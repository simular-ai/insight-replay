# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""InsightReplay agent loop.

Two-phase rollout that injects a fixed "Wait, before I commit..." reflection
prompt whenever phase-1 ends with a natural EOS, then continues generation as
phase-2.

Layout in the response (only when injection happens):
  [phase-1 tokens incl. natural EOS, mask=1]
  [wait_text tokens, mask=0]                   <-- injected, no PG gradient
  [phase-2 tokens incl. natural EOS, mask=1]

If phase-1 hits the response_length cap without natural EOS, no injection
happens and the rollout is returned as-is (treated as overlong by reward).
"""
import logging
import os
from typing import Any
from uuid import uuid4

from verl.experimental.agent_loop.agent_loop import AgentLoopBase, AgentLoopOutput, register
from verl.utils.profiler import simple_timer
from verl.utils.rollout_trace import rollout_trace_op
from verl.workers.rollout.replica import TokenOutput

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


# DAPO prompt wrapping that we strip off to recover the bare math question.
DAPO_PREAMBLE = (
    "Solve the following math problem step by step. The last line of your "
    "response should be of the form Answer: $Answer (without quotes) where "
    "$Answer is the answer to the problem."
)
DAPO_POSTAMBLE = 'Remember to put your answer on its own line after "Answer:".'

# Inject text — the InsightReplay turn-2 prompt that steers the model toward
# verification via a *completely different method* rather than re-running the
# same derivation. Without this steering, RL-trained phase 2 collapses into
# "echo phase 1" and the reflection adds no value.
WAIT_TEMPLATE = (
    "\n\nWait, before I commit to a final answer, let me restate "
    "what's being asked and cross-verify by a completely different "
    "method than what I used above — e.g. plug in specific numeric "
    "values to test an equation, try a closed-form where I enumerated, "
    "enumerate where I used a formula, or attack the problem from a "
    "different physical/mathematical angle. Do not just re-run the "
    "same derivation.\n\n"
    "The user's request: Solve the math problem step by step. "
    "Final answer must be on its own line as 'Answer: $ANSWER'.\n"
    "Question: {bare_question}\n\n"
    "Key conclusions so far:\n"
)


def _extract_bare_question(prompt: str) -> str:
    """Strip the DAPO preamble + postamble to recover the bare math question.

    Different datasets put the POSTAMBLE in different positions:
      - DAPO-Math-15k: PREAMBLE \\n\\n QUESTION \\n\\n POSTAMBLE
      - AIME-2025:     PREAMBLE \\n\\n POSTAMBLE \\n\\n QUESTION
    Use replace() so we strip POSTAMBLE wherever it appears.
    """
    p = prompt
    if p.startswith(DAPO_PREAMBLE):
        p = p[len(DAPO_PREAMBLE):]
    p = p.replace(DAPO_POSTAMBLE, "")
    return p.strip()


# One-shot debug print: only the first rollout in this worker prints. Each Ray
# worker has its own process so each will print once at training start.
_DEBUG_PRINTED = False


@register("insight_replay_agent")
class InsightReplayAgentLoop(AgentLoopBase):
    """Phase 1 -> if natural EOS, splice fixed wait_text -> phase 2."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.prompt_length = self.rollout_config.prompt_length
        self.response_length = self.rollout_config.response_length

        # Build EOS token id set — for Qwen3, both <|im_end|> (151645) and
        # <|endoftext|> (151643) act as natural stop tokens depending on
        # generation_config. We treat either as a phase-1 termination signal.
        eos_set: set[int] = set()
        eos = getattr(self.tokenizer, "eos_token_id", None)
        if isinstance(eos, list):
            eos_set.update(eos)
        elif eos is not None:
            eos_set.add(eos)
        for tok_str in ("<|im_end|>", "<|endoftext|>"):
            tid = self.tokenizer.convert_tokens_to_ids(tok_str)
            if tid is not None and tid != self.tokenizer.unk_token_id:
                eos_set.add(tid)
        self._eos_token_ids = eos_set

    @rollout_trace_op
    async def run(self, sampling_params: dict[str, Any], **kwargs) -> AgentLoopOutput:
        global _DEBUG_PRINTED

        messages = list(kwargs["raw_prompt"])

        # 1. multi-modal (no-op for math text)
        multi_modal_data = await self.process_vision_info(messages)
        images = multi_modal_data.get("images")
        videos = multi_modal_data.get("videos")

        # 2. tokenize the chat-formatted prompt
        prompt_ids = await self.apply_chat_template(messages, images=images, videos=videos)

        # 3. build the wait_text tokens up front (reused across phases)
        user_msg = next((m for m in messages if m["role"] == "user"), None)
        user_text = user_msg["content"] if user_msg else ""
        bare_question = _extract_bare_question(user_text)
        wait_text = WAIT_TEMPLATE.format(bare_question=bare_question)
        wait_ids = self.tokenizer.encode(wait_text, add_special_tokens=False)

        metrics: dict[str, Any] = {}

        # === phase 1 ===
        with simple_timer("generate_sequences", metrics):
            phase1: TokenOutput = await self.server_manager.generate(
                request_id=uuid4().hex,
                prompt_ids=prompt_ids,
                sampling_params=sampling_params,
                image_data=images,
                video_data=videos,
            )
        if metrics.get("num_preempted") is None:
            metrics["num_preempted"] = phase1.num_preempted if phase1.num_preempted is not None else -1

        phase1_tokens = list(phase1.token_ids)
        phase1_logprobs = list(phase1.log_probs) if phase1.log_probs else None

        # Natural EOS = phase 1 emitted one of our EOS tokens at the very end.
        ended_naturally = (
            len(phase1_tokens) > 0 and phase1_tokens[-1] in self._eos_token_ids
        )

        do_inject = ended_naturally and (
            len(phase1_tokens) + len(wait_ids) < self.response_length
        )

        # === phase 2 (only if we injected) ===
        if do_inject:
            remaining = self.response_length - len(phase1_tokens) - len(wait_ids)
            phase2_sampling = dict(sampling_params)
            phase2_sampling["max_tokens"] = remaining

            phase2_prompt_ids = list(prompt_ids) + phase1_tokens + wait_ids

            with simple_timer("generate_sequences", metrics):
                phase2: TokenOutput = await self.server_manager.generate(
                    request_id=uuid4().hex,
                    prompt_ids=phase2_prompt_ids,
                    sampling_params=phase2_sampling,
                    image_data=images,
                    video_data=videos,
                )
            if phase2.num_preempted is not None:
                metrics["num_preempted"] = (metrics.get("num_preempted") or 0) + phase2.num_preempted

            phase2_tokens = list(phase2.token_ids)
            phase2_logprobs = list(phase2.log_probs) if phase2.log_probs else None

            response_ids = phase1_tokens + wait_ids + phase2_tokens
            response_mask = (
                [1] * len(phase1_tokens)
                + [0] * len(wait_ids)
                + [1] * len(phase2_tokens)
            )
            if phase1_logprobs is not None and phase2_logprobs is not None:
                response_logprobs = (
                    phase1_logprobs + [0.0] * len(wait_ids) + phase2_logprobs
                )
            else:
                response_logprobs = None

            num_turns = 4  # user + assistant_p1 + injected + assistant_p2
        else:
            response_ids = phase1_tokens
            response_mask = [1] * len(phase1_tokens)
            response_logprobs = phase1_logprobs
            num_turns = 2

        # === one-shot debug print (first rollout per worker) ===
        if not _DEBUG_PRINTED:
            _DEBUG_PRINTED = True
            try:
                phase1_tail = self.tokenizer.decode(phase1_tokens[-200:]) if phase1_tokens else ""
                phase2_head = ""
                if do_inject:
                    phase2_tokens_for_decode = response_ids[len(phase1_tokens) + len(wait_ids):]
                    phase2_head = self.tokenizer.decode(phase2_tokens_for_decode[:200])
                lines = [
                    "=" * 70,
                    "[INSIGHT_REPLAY DEBUG] first rollout in this worker:",
                    f"  prompt_ids len            : {len(prompt_ids)}",
                    f"  phase1 tokens len         : {len(phase1_tokens)}",
                    f"  phase1 last token id      : {phase1_tokens[-1] if phase1_tokens else None}",
                    f"  ended_naturally           : {ended_naturally}",
                    f"  wait_text tokens len      : {len(wait_ids)}",
                    f"  do_inject                 : {do_inject}",
                    f"  phase2 tokens len         : {len(response_ids) - len(phase1_tokens) - len(wait_ids) if do_inject else 0}",
                    f"  total response_ids len    : {len(response_ids)}",
                    f"  response_mask sum (=#1s)  : {sum(response_mask)}",
                    f"  response_mask zero count  : {len(response_mask) - sum(response_mask)}",
                    f"  EOS token id set          : {sorted(self._eos_token_ids)}",
                    f"  bare_question  (250ch)    : {bare_question[:250]!r}",
                    f"  wait_text head (250ch)    : {wait_text[:250]!r}",
                    f"  phase1 tail    (250ch)    : {phase1_tail[-250:]!r}",
                    f"  phase2 head    (250ch)    : {phase2_head[:250]!r}",
                    "=" * 70,
                ]
                print("\n" + "\n".join(lines), flush=True)
            except Exception as e:
                print(f"[INSIGHT_REPLAY DEBUG] print failed: {e!r}", flush=True)

        # defensive truncation to response_length cap
        response_ids = response_ids[: self.response_length]
        response_mask = response_mask[: self.response_length]
        if response_logprobs is not None:
            response_logprobs = response_logprobs[: self.response_length]

        output = AgentLoopOutput(
            prompt_ids=prompt_ids,
            response_ids=response_ids,
            response_mask=response_mask,
            response_logprobs=response_logprobs,
            multi_modal_data=multi_modal_data,
            num_turns=num_turns,
            metrics=metrics,
            extra_fields=phase1.extra_fields,
        )
        # keeping schema consistent with single_turn_agent_loop / tool_agent_loop
        output.extra_fields.update({"turn_scores": [], "tool_rewards": []})
        return output
