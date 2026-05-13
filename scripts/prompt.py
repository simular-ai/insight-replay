"""Model- and dataset-aware prompt / answer-extraction / thinking-strip utilities.

Supports the 4-way model × 3-way dataset matrix:

  Models:
    * qwen35_35b_a3b         Qwen/Qwen3.5-35B-A3B                  <think>/</think>,  auto-opens
    * glm41v_9b_thinking     zai-org/GLM-4.1V-9B-Thinking          <think>/</think>,  no auto-open, native <answer>
    * r1_distill_qwen_32b    deepseek-ai/DeepSeek-R1-Distill-Qwen-32B  <think>/</think>,  auto-opens, native \\boxed{}
    * gemma4_31b_it          google/gemma-4-31B-it                 <|channel>thought\\n/<channel|>, no auto-open

  Datasets:
    * aime          integer answer
    * gpqa          letter answer (A/B/C/D)
    * livecodebench python code block
    * hmmt          LaTeX expression (graded via math_verify)
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

_DATA_DIR = Path(os.environ.get(
    "FBE_DATA_DIR", Path(__file__).resolve().parent.parent / "data"))


# ---------------------------------------------------------------------------
# Model configs
# ---------------------------------------------------------------------------

@dataclass
class ModelConfig:
    key: str
    hf_name: str
    local_path: str
    # If True, apply_chat_template(add_generation_prompt=True) already ends with
    # the opening thinking marker (e.g. "<think>\n"), so continuations should
    # NOT manually re-inject it.
    auto_opens_think: bool
    # vLLM: recommend tp × dp layout (for ~8 H100/H200 cluster)
    tp_size: int = 1
    # Pass enable_thinking=True/False to apply_chat_template?  None = don't pass.
    enable_thinking_kwarg: Optional[bool] = True
    # Regex to strip thinking blocks from text.
    # For <think>...</think> style:
    think_open: str = "<think>"
    think_close: str = "</think>"
    # Some templates use asymmetric <|channel>thought\n ... <channel|> (Gemma).
    # If set, these override the above for strip_thinking.
    alt_think_open: Optional[str] = None
    alt_think_close: Optional[str] = None


MODELS: dict[str, ModelConfig] = {
    # `local_path` defaults to the HuggingFace repo ID so vLLM resolves the
    # model via the HF cache. Override via environment variable
    # `FBE_MODEL_PATH_<KEY>` to point at a local snapshot directory if you
    # have one (recommended on shared clusters to avoid re-downloading).
    "qwen35_35b_a3b": ModelConfig(
        key="qwen35_35b_a3b",
        hf_name="Qwen/Qwen3.5-35B-A3B",
        local_path=os.environ.get("FBE_MODEL_PATH_QWEN35_35B_A3B", "Qwen/Qwen3.5-35B-A3B"),
        auto_opens_think=True,
        tp_size=2,
        enable_thinking_kwarg=True,
    ),
    "glm41v_9b_thinking": ModelConfig(
        key="glm41v_9b_thinking",
        hf_name="zai-org/GLM-4.1V-9B-Thinking",
        local_path=os.environ.get("FBE_MODEL_PATH_GLM41V_9B_THINKING", "zai-org/GLM-4.1V-9B-Thinking"),
        auto_opens_think=False,
        tp_size=1,
        enable_thinking_kwarg=None,  # template doesn't accept it
    ),
    "r1_distill_qwen_32b": ModelConfig(
        key="r1_distill_qwen_32b",
        hf_name="deepseek-ai/DeepSeek-R1-Distill-Qwen-32B",
        local_path=os.environ.get("FBE_MODEL_PATH_R1_DISTILL_QWEN_32B", "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B"),
        auto_opens_think=True,
        tp_size=2,
        enable_thinking_kwarg=None,  # no such kwarg; thinking is always on
    ),
    "gemma4_31b_it": ModelConfig(
        key="gemma4_31b_it",
        hf_name="google/gemma-4-31B-it",
        local_path=os.environ.get("FBE_MODEL_PATH_GEMMA4_31B_IT", "google/gemma-4-31B-it"),
        auto_opens_think=False,
        tp_size=2,
        enable_thinking_kwarg=True,
        # Gemma uses channel markers, not <think>/</think>
        think_open="<|channel>thought\n",
        think_close="<channel|>",
    ),
    # ---- 8B-class companions (single-GPU TP=1) ----
    "qwen35_9b": ModelConfig(
        key="qwen35_9b",
        hf_name="Qwen/Qwen3.5-9B",
        local_path=os.environ.get("FBE_MODEL_PATH_QWEN35_9B", "Qwen/Qwen3.5-9B"),
        auto_opens_think=True,
        tp_size=1,
        enable_thinking_kwarg=True,
    ),
    "r1_distill_qwen_7b": ModelConfig(
        key="r1_distill_qwen_7b",
        hf_name="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
        local_path=os.environ.get("FBE_MODEL_PATH_R1_DISTILL_QWEN_7B", "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"),
        auto_opens_think=True,
        tp_size=1,
        enable_thinking_kwarg=None,  # no such kwarg; thinking is always on
    ),
    "gemma4_e4b_it": ModelConfig(
        key="gemma4_e4b_it",
        hf_name="google/gemma-4-E4B-it",
        local_path=os.environ.get("FBE_MODEL_PATH_GEMMA4_E4B_IT", "google/gemma-4-E4B-it"),
        auto_opens_think=False,
        tp_size=1,
        enable_thinking_kwarg=True,
        # Gemma uses channel markers, not <think>/</think>
        think_open="<|channel>thought\n",
        think_close="<channel|>",
    ),
}


# ---------------------------------------------------------------------------
# Dataset configs
# ---------------------------------------------------------------------------

@dataclass
class DatasetConfig:
    key: str
    data_path: str
    # Semantic answer type
    answer_kind: str    # 'integer' | 'letter' | 'code'
    # Human-readable instructions appended to the baseline prompt
    baseline_header: str
    finding_header: str


DATASETS: dict[str, DatasetConfig] = {
    "aime": DatasetConfig(
        key="aime",
        data_path=str(_DATA_DIR / "aime.jsonl"),
        answer_kind="integer",
        baseline_header=(
            "Solve the math problem. "
            "Final answer must be a single non-negative integer in <Answer>...</Answer>."
        ),
        finding_header=(
            "Solve the math problem by reasoning step by step. "
            "Final answer must be a single non-negative integer in <Answer>...</Answer>."
        ),
    ),
    "gpqa": DatasetConfig(
        key="gpqa",
        data_path=str(_DATA_DIR / "gpqa_diamond_test.jsonl"),
        answer_kind="letter",
        baseline_header=(
            "Select the best answer. "
            "Final answer must be a single letter (A, B, C, or D) in <Answer>...</Answer>."
        ),
        finding_header=(
            "Select the best answer by reasoning step by step. "
            "Final answer must be a single letter (A, B, C, or D) in <Answer>...</Answer>."
        ),
    ),
    "livecodebench": DatasetConfig(
        key="livecodebench",
        data_path=str(_DATA_DIR / "livecodebench_v5.jsonl"),
        answer_kind="code",
        baseline_header=(
            "Solve the coding task. "
            "Put your final solution in a single Python code block delimited "
            "by ```python ... ```."
        ),
        finding_header=(
            "Solve the coding task by reasoning step by step. "
            "Put your final solution in a single Python code block delimited "
            "by ```python ... ```."
        ),
    ),
    "hmmt": DatasetConfig(
        key="hmmt",
        data_path=str(_DATA_DIR / "hmmt.jsonl"),
        answer_kind="latex",
        # baseline and finding use the same header — see discussion in
        # scripts/math_verify_util.py. Final answer must be exact (symbolic),
        # not a decimal approximation, otherwise math_verify may reject it.
        baseline_header=(
            "Solve the math problem. Final answer must be the EXACT value "
            "(simplified, no decimal approximations) in \\boxed{...}. "
            "If there are multiple solutions, list all of them inside the "
            "single \\boxed{...} separated by commas."
        ),
        finding_header=(
            "Solve the math problem. Final answer must be the EXACT value "
            "(simplified, no decimal approximations) in \\boxed{...}. "
            "If there are multiple solutions, list all of them inside the "
            "single \\boxed{...} separated by commas."
        ),
    ),
}


# ---------------------------------------------------------------------------
# Question formatting per dataset
# ---------------------------------------------------------------------------

def _format_gpqa_question(example: dict) -> str:
    """Build '<question>\n\nA) ...\nB) ...' from raw GPQA row."""
    q = example.get("question", "")
    choices = example.get("choices") or example.get("options") or []
    if not choices:
        return q
    letters = ["A", "B", "C", "D"]
    lines = [q.strip(), ""]
    for letter, ch in zip(letters, choices):
        lines.append(f"{letter}) {ch}")
    return "\n".join(lines)


def format_question(dataset_key: str, example: dict) -> str:
    if dataset_key == "gpqa":
        return _format_gpqa_question(example)
    if dataset_key == "livecodebench":
        return example.get("question") or example.get("prompt") or ""
    # aime, hmmt
    return example.get("question", "")


def ground_truth(dataset_key: str, example: dict):
    if dataset_key == "aime":
        return int(str(example.get("answer", example.get("gt_answer", ""))).strip())
    if dataset_key == "gpqa":
        # Store as uppercase letter
        ans = str(example.get("answer", "")).strip().upper()
        return ans[0] if ans else None
    if dataset_key == "livecodebench":
        # No scalar GT; evaluation is via test cases (we don't auto-grade code here)
        return example.get("all_test_cases")
    if dataset_key == "hmmt":
        # Raw LaTeX string — graded via math_verify_util.verify_latex
        return str(example.get("answer", "")).strip()
    return None


# ---------------------------------------------------------------------------
# Prompt builders
# ---------------------------------------------------------------------------

def build_baseline_prompt(dataset_key: str, example: dict) -> str:
    """Single-shot CoT user-content (the prompt inside the user turn, before
    chat-template wrapping)."""
    ds = DATASETS[dataset_key]
    q = format_question(dataset_key, example)
    return f"{ds.baseline_header}\n\n{q}\n"


def build_finding_base_prompt(dataset_key: str, example: dict) -> str:
    """User-content used at the BEGINNING of finding's continuation stream.
    Same shape as baseline but with the step-by-step header — this is the
    prompt the model is continuing from across rounds."""
    ds = DATASETS[dataset_key]
    q = format_question(dataset_key, example)
    return f"{ds.finding_header}\n\n{q}\n"


def build_cf_extract_prompt(chunk: str, dataset_key: str, example: dict,
                             prior_findings: list = None) -> str:
    """Ask the model to identify 1-3 VERBATIM anchor sentences from the
    reasoning trace that represent the round's most important established
    results, and restate each as a concise critical finding that can
    include specific numeric values.

    The anchor is used downstream to trim the raw generation back to a
    semantically-meaningful endpoint (the last anchor) rather than just
    the last complete sentence.

    ``prior_findings``: list of CF strings already captured in previous
    rounds. Passed in so the extractor can avoid repetition and build on
    them.
    """
    q = format_question(dataset_key, example)
    chunk_trimmed = trim_to_last_sentence(chunk)
    # Keep more context than before so the extractor sees real progress
    if len(chunk_trimmed) > 7000:
        chunk_trimmed = "..." + chunk_trimmed[-7000:]

    prior_block = ""
    if prior_findings:
        prior_block = (
            "\nFindings already established in previous rounds (do NOT "
            "simply re-state them; report only NEW progress this round "
            "has made beyond these):\n"
            + "\n".join(f"  - {pf}" for pf in prior_findings)
            + "\n"
        )

    # Dataset-specific verification example menu.
    if dataset_key == "aime":
        verify_examples = (
            "    - Plug specific numbers (e.g. y=0, y=-4) into a derived "
            "equation and check both sides match.\n"
            "    - Re-derive the key algebraic step from scratch.\n"
            "    - Confirm a claimed factorization by expanding it.\n"
            "    - Check dimensional / sign consistency.\n"
            "    - Cross-check against a different method.\n"
            "    - Sanity-check the scale (is the answer plausible given "
            "the problem?)\n"
        )
        domain_reminder = (
            "  * Long polynomial coefficients, multi-step sign "
            "manipulations, and meta-claims (\"earlier step had an error\") "
            "REQUIRE a concrete numerical verification — 'looks consistent' "
            "is NOT a verification.\n"
        )
    elif dataset_key == "gpqa":
        verify_examples = (
            "    - Re-read the quoted option against the claim and check "
            "they agree.\n"
            "    - Verify the elimination logic: if claim says 'option X is "
            "ruled out because of reason R', check R against the facts.\n"
            "    - Cross-check unit / magnitude order against known values.\n"
            "    - Look for a counter-example or inconsistency with another "
            "established fact.\n"
            "    - Re-derive the specific scientific / quantitative step.\n"
        )
        domain_reminder = (
            "  * Claims like \"option X is the correct one\" or \"option Y "
            "is ruled out\" REQUIRE a specific reason plus a check against "
            "domain facts — 'seems right' is NOT a verification.\n"
        )
    elif dataset_key == "livecodebench":
        verify_examples = (
            "    - Mentally run a small concrete input through the proposed "
            "algorithm and check the output against expectations.\n"
            "    - Trace an edge case (empty input, single element, all "
            "duplicates, extremes).\n"
            "    - Verify the proposed function signature / return type "
            "matches the problem.\n"
            "    - Check loop invariants and termination conditions.\n"
            "    - Confirm complexity / correctness of a core data-structure "
            "operation (e.g. binary search bounds).\n"
        )
        domain_reminder = (
            "  * Claims about algorithm correctness, signature, or "
            "complexity REQUIRE a concrete trace or invariant argument — "
            "'the algorithm should work' is NOT a verification.\n"
        )
    else:
        verify_examples = (
            "    - Re-derive the key step from scratch.\n"
            "    - Plug in a specific concrete instance and check both "
            "sides / both interpretations agree.\n"
            "    - Cross-check against an alternative method.\n"
            "    - Look for a counter-example or contradiction with another "
            "established fact.\n"
        )
        domain_reminder = (
            "  * Specific numeric or categorical claims REQUIRE a concrete "
            "verification — 'looks right' is NOT a verification.\n"
        )

    return (
        "You are reviewing one round of reasoning on a problem. Your task is "
        "to identify concrete results this round has established — but ONLY "
        "after independently VERIFYING each one. A wrong or unverified "
        "finding is WORSE than no finding, because the next round will build "
        "on it and compound the error. When in doubt, discard.\n\n"
        "PROCESS for each candidate finding (do this carefully, internally):\n"
        "  STEP 1. Pick a candidate result established in this round's "
        "reasoning. Identify the verbatim sentence where it was established.\n"
        "  STEP 2. State the result concisely; it MAY include specific "
        "numeric values, equations, or explicit conditions.\n"
        "  STEP 3. VERIFY the result independently — take as many lines as "
        "you need. Examples:\n"
        f"{verify_examples}"
        "  STEP 4. Judge CORRECT / UNCERTAIN / INCORRECT based on the "
        "verification. ONLY if you reach CORRECT should this finding be "
        "included in the final output.\n\n"
        "RULES:\n"
        "  * Perform STEPS 1-4 thoroughly for every candidate, but OUTPUT "
        "ONLY the (ANCHOR, FINDING) pairs that you judged CORRECT. Discard "
        "UNCERTAIN and INCORRECT ones silently.\n"
        "  * Better to output zero findings than one wrong one. If no "
        "candidate passes verification, output exactly "
        "'NO_VERIFIED_PROGRESS'.\n"
        f"{domain_reminder}"
        "  * Order the retained pairs chronologically (ANCHOR 1 appears "
        "earliest in the reasoning).\n"
        "  * Do NOT propose the final numeric / letter / code answer.\n\n"
        f"Problem: {q}\n{prior_block}\n"
        "This round's reasoning:\n"
        f"{chunk_trimmed}\n\n"
        "OUTPUT FORMAT (only verified findings; up to 3 pairs):\n"
        'ANCHOR 1: "<verbatim quoted sentence>"\n'
        "FINDING 1: <1-2 sentence restatement>\n\n"
        'ANCHOR 2: "..."\n'
        "FINDING 2: ...\n\n"
        "(etc.)\n\n"
        "Output:"
    )


# ---------------------------------------------------------------------------
# CF output parser + anchor locator
# ---------------------------------------------------------------------------

_ANCHOR_LINE = re.compile(
    r"ANCHOR\s*(\d+)\s*:\s*[\"\u201c\u201d](.+?)[\"\u201c\u201d]",
    re.DOTALL | re.IGNORECASE,
)
_FINDING_LINE = re.compile(
    r"FINDING\s*(\d+)\s*:\s*(.+?)(?=\n\s*(?:ANCHOR|FINDING)\s*\d|\Z)",
    re.DOTALL | re.IGNORECASE,
)


def parse_cf_output(raw: str) -> tuple:
    """Return (anchors, findings) parsed from the CF extractor's output.

    The model performs internal verification and only outputs pairs it
    judged CORRECT, so here we just extract all (ANCHOR, FINDING) pairs
    that appear in the final output.

    Returns ([], []) on NO_PROGRESS / NO_VERIFIED_PROGRESS or empty input.
    """
    if not raw:
        return [], []
    head = raw.strip().upper()
    if head.startswith("NO_PROGRESS") or head.startswith("NO_VERIFIED_PROGRESS"):
        return [], []

    anchors_by_id = {m.group(1): m.group(2).strip()
                     for m in _ANCHOR_LINE.finditer(raw)}
    findings_by_id = {m.group(1): m.group(2).strip().rstrip('"\u201c\u201d').strip()
                      for m in _FINDING_LINE.finditer(raw)}

    anchors, findings = [], []
    for gid in sorted(anchors_by_id.keys(),
                      key=lambda x: int(x) if x.isdigit() else 0):
        a = anchors_by_id.get(gid)
        f = findings_by_id.get(gid)
        if a and f:
            anchors.append(a)
            findings.append(f)
    return anchors, findings


def find_last_anchor_end(raw_generation: str, anchors: list) -> int:
    """Return the char position just AFTER the latest anchor in
    raw_generation. Returns -1 if no anchor could be located."""
    if not anchors or not raw_generation:
        return -1
    best = -1
    for a in anchors:
        a_stripped = a.strip()
        if len(a_stripped) < 10:
            continue
        pos = raw_generation.rfind(a_stripped)
        if pos >= 0:
            end = pos + len(a_stripped)
            if end > best:
                best = end
            continue
        # Fuzzy: use a 40-char prefix
        probe = a_stripped[:40].strip()
        if len(probe) >= 12:
            pos = raw_generation.rfind(probe)
            if pos >= 0:
                end = pos + len(probe)
                # Extend to the next sentence terminator if possible
                tail = raw_generation[end:end + 300]
                m = re.search(r"[.!?。！？]['\"\u201d)]?\s", tail)
                if m:
                    end = end + m.end()
                if end > best:
                    best = end
    return best


def format_cf_injection(dataset_key: str, example: dict, cfs: list[str]) -> str:
    """The block that gets spliced into the continuation stream between rounds.

    Phrased as the model's own self-reflection ("Let me summarize ...") so
    that, sitting inside a <think> block, it reads naturally as part of the
    model's internal monologue rather than an externally inserted note.
    Always re-states the user's task + output-format requirement so the
    model doesn't forget it deep into a long thinking stream.
    """
    q = format_question(dataset_key, example)
    header = DATASETS[dataset_key].baseline_header
    lines = [
        "Let me summarize the current progress.",
        f"The user's request is: {header}",
        f"Question: {q}",
    ]
    for i, cf in enumerate(cfs, 1):
        lines.append(f"Critical finding {i}: {cf}")
    lines.append("Let me continue thinking from here.")
    return "\n\n" + "\n".join(lines) + "\n\n"


# ---------------------------------------------------------------------------
# Method B: post-hoc truncation + self-verification injection
# ---------------------------------------------------------------------------
# Drop findings whose body starts with label/control-token keywords — these
# are label leakage or thinking-marker residue, not real findings.
_BAD_CF_PREFIX = re.compile(
    r"^\s*(?:ANCHOR\b|FINDING\b|</?think>|<\|?channel)",
    re.IGNORECASE,
)

# Drop extractor meta-plan bullets: the extractor sometimes writes its own
# task outline as markdown section headers ("**Analyze the Request:**",
# "**Review the Problem:**", "**Recall Key Findings:**") instead of actual
# conclusions. These leak through as findings and dilute / mislead the
# self-verification injection. Require BOTH a meta-verb and a meta-object
# to avoid killing real findings like "Identify the constraints: n1+n2=6".
_META_VERBS = (
    r"Analyze|Review|Recall|Extract|Summariz(?:e|ing)|Summary|"
    r"Restate|Identify|Understand|Clarif(?:y|ication)|Outline|"
    r"Plan|Approach|Strategy|Examine|Re-?examine"
)
_META_OBJECTS = (
    r"the\s+Request|the\s+Problem|the\s+Reasoning(?:\s+Pass)?|"
    r"the\s+Task|the\s+Question|the\s+Goal|the\s+Objective|"
    r"(?:Key\s+|The\s+)?Findings?|(?:Key\s+|The\s+)?Conclusions?|"
    r"the\s+Solution|my\s+Reasoning|Prior\s+Reasoning|Key\s+Points?"
)
_META_FINDING_RE = re.compile(
    rf"^\s*\*{{0,2}}\s*(?:{_META_VERBS})\s+(?:{_META_OBJECTS})",
    re.IGNORECASE,
)
# Markdown section header with no content, e.g. "**Review the Problem:**"
# that leaves nothing useful to verify. Must end with a colon inside the
# bold wrapper — otherwise a fully bolded real finding like
# "**Product 1 is para-nitrotoluene.**" would be wrongly dropped.
_EMPTY_HEADER_RE = re.compile(r"^\s*\*{2,}[^*\n]{1,120}:\s*\*{2,}\s*$")

# Label-only finding: one/two words followed by a colon with nothing after,
# e.g. "Constraints:", "Rules.", "Problem.". These are section-header
# remnants the extractor spits out when it echoes structure without content.
# Require the "word(s) + punctuation + end-of-string" shape exactly.
_LABEL_ONLY_RE = re.compile(
    r"^\s*\*{0,2}\s*\w+(?:\s+\w+){0,2}\s*[:：.]\s*\*{0,2}\s*$"
)

# Problem-statement role labels that echo the problem rather than capturing
# any derived conclusion. These role labels never appear at the start of a
# genuine finding (real findings use concrete nouns like "Triangle ABC",
# "Tangent condition", etc.).
_ECHO_LABEL_RE = re.compile(
    r"^\s*\*{0,2}\s*"
    r"(?:Goal|Format|Problem(?:\s+Statement)?|Question|Given|Find|"
    r"Required|Requirement|Objective|Task|To\s+Find|We\s+need|"
    r"We\s+want|We\s+are\s+asked)"
    r"\s*\*{0,2}\s*:\s*\S",
    re.IGNORECASE,
)


def split_at_think_close(model_key: str, response: str) -> tuple:
    """For Method B: given a full baseline response, return (pre_close, post_close).

    pre_close = everything BEFORE the first close marker (may include the
                opening marker, e.g. '<|channel>thought\\n' for Gemma).
    post_close = everything AFTER the close marker.

    Returns (None, None) if the thinking block never closed — Method B then
    falls back to accepting the baseline as-is with no CF.
    """
    if not response:
        return None, None
    cfg = MODELS[model_key]
    close = cfg.think_close
    idx = response.find(close)
    if idx < 0:
        return None, None
    return response[:idx], response[idx + len(close):]


def thinking_body_only(model_key: str, pre_close: str) -> str:
    """Strip the opening thinking marker from pre_close so the extractor sees
    only the reasoning text."""
    cfg = MODELS[model_key]
    open_m = cfg.think_open
    if open_m and open_m in pre_close:
        return pre_close.split(open_m, 1)[1]
    return pre_close


def adaptive_finding_cap(thinking_tokens: int) -> int:
    """Step-function finding cap based on round-1 thinking size.

      thinking_tokens ≤ 5000         →  2
      5000 < t ≤ 10000               →  3
      10000 < t ≤ 15000              →  4
      ...
      45000 < t ≤ 50000              → 11
      t > 50000                      → 11  (plateau)
    """
    return min(11, 2 + max(0, (thinking_tokens - 1) // 5000))


def build_methodb_extract_prompt(thinking_body: str, dataset_key: str,
                                  example: dict, max_findings: int,
                                  prior_findings: Optional[list] = None) -> str:
    """Pure fact-extraction prompt (no judgment, no anchors).

    Asks the model to list up to `max_findings` key conclusions from its own
    thinking trace. These will be fed back to it as a self-verification
    checklist before it commits to a final answer.

    The full thinking body is sent verbatim — truncating it would hide the
    tail conclusions (often the most load-bearing). Budget is controlled at
    the caller level via MAX_MODEL_LEN.

    ``prior_findings``: list of finding strings already captured in earlier
    passes (used for multi-turn insightreplay). When provided, the extractor
    is asked to report only NEW conclusions reached beyond those. When
    ``None`` or empty, the prompt is identical to the single-turn form.
    """
    q = format_question(dataset_key, example)
    body = thinking_body or ""

    prior_block = ""
    if prior_findings:
        prior_lines = "\n".join(f"- {pf}" for pf in prior_findings)
        prior_block = (
            "You already captured these conclusions in earlier passes — "
            "do NOT simply re-state them. Focus on NEW concrete "
            "conclusions your reasoning has reached since:\n"
            f"{prior_lines}\n\n"
        )

    return (
        "You just finished an internal reasoning pass on the problem below. "
        "List the most important concrete conclusions you reached during "
        "that pass — the facts, equations, intermediate results, or "
        "commitments that your final answer will rest on. These will be "
        "fed back to you as a checklist to verify before you commit.\n\n"
        "RULES:\n"
        f"  * Output AT MOST {max_findings} findings (fewer is fine). Pick "
        f"the {max_findings} most load-bearing — conclusions whose "
        "correctness most affects the final answer.\n"
        "  * Each finding: one short sentence, factual and specific. "
        "Include the numeric values, equations, or named quantities that "
        "were derived.\n"
        "  * Do NOT re-derive, do NOT evaluate correctness, do NOT restate "
        "the final answer. Just list what was concluded.\n"
        "  * Output format: plain bullet list, one finding per line, each "
        "line starting with '- '. No preamble, no closing remarks.\n"
        "  * The RULES above describe HOW to format the output. They are "
        "NOT findings. Do NOT echo these rules back as bullet items.\n\n"
        f"Problem: {q}\n\n"
        "Your prior reasoning:\n"
        f"{body}\n\n"
        f"{prior_block}"
        "Key conclusions to verify (bullet list only):"
    )


def parse_methodb_findings(raw: str, max_findings: int) -> list:
    """Parse bullet list from the Method B extractor's output."""
    if not raw:
        return []
    out = []
    for line in raw.strip().splitlines():
        line = line.strip()
        if not line:
            continue
        m = re.match(r"^[-*\u2022]\s+(.+)$", line)
        if not m:
            m = re.match(r"^\d+[.)]\s+(.+)$", line)
        if not m:
            continue
        body = m.group(1).strip().strip('"\u201c\u201d').strip()
        if not body or _BAD_CF_PREFIX.match(body):
            continue
        if (_META_FINDING_RE.match(body)
                or _EMPTY_HEADER_RE.match(body)
                or _ECHO_LABEL_RE.match(body)
                or _LABEL_ONLY_RE.match(body)):
            continue
        if len(body) > 500:
            body = body[:500].rstrip() + "..."
        out.append(body)
        if len(out) >= max_findings:
            break
    return out


def format_methodb_injection(dataset_key: str, example: dict,
                              findings: list,
                              baseline_answer: Optional[str] = None,
                              variant: str = "default") -> str:
    """Self-verification block spliced in BEFORE the model re-emits the
    thinking-close marker. Phrased as the model's own voice so it reads as
    continuation of the internal monologue.

    Restates the user's request + question (so the model doesn't drift from
    the task deep into a long thinking stream) and then lists the findings
    it's about to verify.

    If baseline_answer is provided, anchors the verification against that
    candidate answer — this breaks bullet-list format momentum and pushes
    the model to explicitly restate or revise a concrete answer rather than
    trailing off in a bullet list.

    ``variant`` controls the opening self-reflection phrase (for multi-turn
    insightreplay):
      * "default" — single-turn / first-turn wording (behavior unchanged)
      * "again"   — "double-check the key conclusions I've been relying on
                    one more time"
      * "final"   — "double-check the key conclusions I've been relying on
                    for the last time"
    """
    if not findings:
        return ""
    q = format_question(dataset_key, example)
    header = DATASETS[dataset_key].baseline_header
    if variant == "again":
        opening = (
            "Wait, before I commit to a final answer, let me double-check "
            "the key conclusions I've been relying on one more time."
        )
    elif variant == "final":
        opening = (
            "Wait, before I commit to a final answer, let me double-check "
            "the key conclusions I've been relying on for the last time."
        )
    else:
        opening = (
            "Wait, before I commit to a final answer, let me restate what's "
            "being asked and double-check the key conclusions I've been "
            "relying on."
        )
    lines = [
        "",
        "",
        opening,
        "",
        f"The user's request: {header}",
        f"Question: {q}",
        "",
        "Key conclusions so far:",
    ]
    for i, f in enumerate(findings, 1):
        lines.append(f"  {i}. {f}")
    lines.append("")
    # For turns 2 and 3, steer the model toward a *different* verification
    # method rather than repeating the same derivation. Turn 1 keeps the
    # original wording unchanged.
    if variant == "again":
        verify_steer = (
            " This time, approach the verification by a completely different "
            "method than I used above — e.g. plug in specific numeric values "
            "to test an equation, try a closed-form where I enumerated, "
            "enumerate where I used a formula, or attack the problem from a "
            "different physical/mathematical angle. Do not just re-run the "
            "same derivation."
        )
    elif variant == "final":
        verify_steer = (
            " This is my last check, so I need an independent cross-"
            "verification — use yet another approach I haven't tried yet "
            "(concrete sanity plug-in, orthogonal derivation, dimensional / "
            "limiting-case check, or a counter-example search). If the new "
            "method disagrees with my current answer, revise; if it agrees, "
            "commit."
        )
    else:
        verify_steer = ""
    if baseline_answer:
        if dataset_key == "livecodebench":
            lines.append(
                "Before finalizing, my current working implementation is:")
            lines.append("")
            lines.append("```python")
            lines.append(baseline_answer.strip())
            lines.append("```")
            lines.append("")
            lines.append(
                "Let me verify each of these conclusions and check whether "
                "they actually support this implementation — or whether "
                "I've missed something that would change it." + verify_steer)
        else:
            lines.append(
                f"Before finalizing, my current working answer is "
                f"{baseline_answer}.")
            lines.append(
                "Let me verify each of these conclusions and check whether "
                "they actually support this answer — or whether I've missed "
                "something that would change it." + verify_steer)
    else:
        lines.append(
            "Let me verify each of these one more time before finalizing "
            "my answer." + verify_steer)
    lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Sentence-boundary trim (used for CF extraction input only)
# ---------------------------------------------------------------------------

_SENT_END_RE = re.compile(r"[.!?。！？]['\"\)\]]?\s")


def trim_to_last_sentence(text: str) -> str:
    """If the text was hard-cut mid-sentence, trim back to the last sentence
    boundary within the final 500 chars. Falls back to last newline."""
    if not text:
        return text
    tail_start = max(0, len(text) - 500)
    best = -1
    for m in _SENT_END_RE.finditer(text, tail_start):
        best = m.end()
    if best > 0:
        return text[:best].rstrip()
    nl = text.rfind("\n", tail_start)
    if nl > 0:
        return text[:nl].rstrip()
    return text


# ---------------------------------------------------------------------------
# Chat-template wrapper + thinking strip (model-aware)
# ---------------------------------------------------------------------------

def apply_chat_template(model_key: str, tokenizer, user_content: str,
                        enable_thinking_override: Optional[bool] = None) -> str:
    """Wrap user_content with the model's chat template.
    If the model supports an `enable_thinking` kwarg we pass it (defaulting to
    the config), otherwise we omit it.  Returns a ready-to-send prompt."""
    cfg = MODELS[model_key]
    messages = [{"role": "user", "content": user_content}]
    eth = (enable_thinking_override if enable_thinking_override is not None
           else cfg.enable_thinking_kwarg)
    try:
        if eth is None:
            return tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True)
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
            enable_thinking=eth)
    except TypeError:
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True)


def strip_thinking(model_key: str, text: str) -> str:
    """Return post-think text. Uses rsplit on the LAST closing marker so that
    if the model emits the marker inside the thinking block (common for
    mid-reasoning self-reference), the real boundary still wins.
    If no closing marker is present, returns the full text (useful for
    continuation-based finding where we want samples to keep going rather
    than being prematurely killed)."""
    cfg = MODELS[model_key]
    open_m = cfg.think_open
    close_m = cfg.think_close
    # First, drop any fully-formed <open>…<close> pairs
    pattern = re.escape(open_m) + r".*?" + re.escape(close_m)
    text = re.sub(pattern, "", text, flags=re.DOTALL)
    if close_m in text:
        return text.rsplit(close_m, 1)[1].strip()
    return text.strip()


def thinking_is_closed(model_key: str, text: str) -> bool:
    """True iff the model has emitted the closing thinking marker anywhere."""
    cfg = MODELS[model_key]
    return cfg.think_close in text


# ---------------------------------------------------------------------------
# Answer extraction per dataset
# ---------------------------------------------------------------------------

# Try multiple answer tag styles in priority order.
_INT_ANSWER_PATTERNS = [
    # Explicit <Answer>N</Answer> / <answer>N</answer> we ask for
    re.compile(r"<[aA]nswer>\s*(-?\d+)\s*</[aA]nswer>"),
    # DeepSeek R1 style \boxed{N}
    re.compile(r"\\boxed\{\s*(-?\d+)\s*\}"),
    # LaTeX-command mimicry: \Answer{N} / \Answer N / \answer{N} (model
    # drifts from XML-tag requirement toward LaTeX command style, especially
    # on math-heavy problems and after multi-turn reinjections; case-
    # insensitive because lowercase \answer{} also appears).
    re.compile(r"\\[aA]nswer\s*\{?\s*(-?\d+)"),
    # Markdown "**Answer**: N" / "**Answer:** N" / "Answer: N". Tolerate
    # markdown bold/italic wrapping the word "Answer" itself, and decorations
    # between colon and the number.
    re.compile(
        r"(?i)[*_`]*\s*(?:final\s+)?answer\s*[*_`]*\s*[:：]\s*"
        r"[*_`\"'(<]*\s*(-?\d+)"
    ),
    # Fallback: open <Answer> tag without a closing </Answer>. Last-resort
    # since the closed form above takes priority when both are present.
    re.compile(r"<[aA]nswer>\s*(-?\d+)\b"),
]

_LETTER_ANSWER_PATTERNS = [
    # Preferred: explicit tag we asked for
    re.compile(r"<[aA]nswer>\s*([A-D])\s*</[aA]nswer>"),
    # R1-style
    re.compile(r"\\boxed\{\s*([A-D])\s*\}"),
    # LaTeX-command mimicry: \Answer{D} or \Answer D (same drift as INT);
    # case-insensitive to also catch \answer{} lowercase variants.
    re.compile(r"\\[aA]nswer\s*\{?\s*([A-D])\b"),
    # "**Answer**: D" / "**Answer:** D" / "Answer: D" / "Final answer: B".
    # Tolerate markdown bold/italic around "Answer" itself AND between colon
    # and the letter. Word-boundary AFTER the letter (so "Abc" isn't captured
    # as "A").
    re.compile(
        r"(?i)[*_`]*\s*(?:final\s+)?answer\s*[*_`]*\s*[:：]?\s*"
        r"[*_`\"'(<]*\s*([A-D])\b"
    ),
    # Fallback: open <Answer> tag with a letter but no </Answer> close
    # (also catches variants like <Answer>\nD. 10\n</Answer> where the
    # strict pattern fails due to trailing text after the letter).
    re.compile(r"<[aA]nswer>\s*([A-D])\b"),
]
# NOTE: we deliberately removed the "\boption\s+([A-D])\b" pattern — it
# matched every "Option A/B/C/D" mention in the reasoning and silently
# picked the last one, overriding the model's actual final answer.

_CODE_BLOCK_RE = re.compile(
    r"```(?:python|py)?\s*\n(.*?)```", re.DOTALL)


def extract_answer(dataset_key: str, text: str) -> Optional[str]:
    """Return the extracted answer string or None. Works on post-think text."""
    if not text:
        return None
    kind = DATASETS[dataset_key].answer_kind
    if kind == "integer":
        for pat in _INT_ANSWER_PATTERNS:
            matches = pat.findall(text)
            if matches:
                return matches[-1].strip()
        return None
    if kind == "letter":
        for pat in _LETTER_ANSWER_PATTERNS:
            matches = pat.findall(text)
            if matches:
                return matches[-1].strip().upper()
        return None
    if kind == "code":
        matches = _CODE_BLOCK_RE.findall(text)
        if matches:
            return matches[-1].strip()
        return None
    if kind == "latex":
        # Brace-counting extractor handles nested {} inside \boxed{}.
        from math_verify_util import extract_boxed
        return extract_boxed(text)
    return None


def grade_answer(dataset_key: str, predicted: Optional[str], gt) -> bool:
    """Deterministic grading for scalar-answer datasets. For livecodebench
    returns False here — real grading requires running test cases."""
    if predicted is None:
        return False
    kind = DATASETS[dataset_key].answer_kind
    if kind == "integer":
        try:
            return int(predicted) == int(gt)
        except (ValueError, TypeError):
            return False
    if kind == "letter":
        return (predicted or "").strip().upper() == str(gt).strip().upper()
    if kind == "latex":
        from math_verify_util import verify_latex
        # Wrap predicted in \boxed{...} so math_verify's default config picks it up.
        return verify_latex(str(gt), f"\\boxed{{{predicted}}}")
    # code: require external grader
    return False


# ---------------------------------------------------------------------------
# Tiny smoke test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # Quick sanity check on answer extraction
    tests = [
        ("aime", "Final answer: <Answer>42</Answer>", "42"),
        ("aime", "So the result is \\boxed{7}.", "7"),
        ("aime", "Answer: 123", "123"),
        ("gpqa", "<answer>C</answer>", "C"),
        ("gpqa", "\\boxed{B}", "B"),
        ("livecodebench", "```python\nprint(1)\n```", "print(1)"),
        ("hmmt", "So the answer is \\boxed{\\frac{1}{2}}.", "\\frac{1}{2}"),
        ("hmmt", "\\boxed{8\\sqrt{10}}", "8\\sqrt{10}"),
    ]
    for ds, text, expect in tests:
        got = extract_answer(ds, text)
        print(f"  [{ds}] {text[:40]!r:45} → {got!r}  "
              + ("OK" if got == expect else f"EXPECTED {expect!r}"))

    # hmmt grading sanity
    grade_tests = [
        ("hmmt", "\\frac{1}{2}", "\\frac{1}{2}", True),
        ("hmmt", "0.5", "\\frac{1}{2}", True),
        ("hmmt", "\\sqrt{640}", "8\\sqrt{10}", True),
        ("hmmt", "0.6", "\\frac{1}{2}", False),
    ]
    for ds, pred, gt, expect in grade_tests:
        got = grade_answer(ds, pred, gt)
        print(f"  [grade {ds}] pred={pred!r} gt={gt!r} → {got}  "
              + ("OK" if got == expect else f"EXPECTED {expect}"))

    # strip_thinking smoke
    for m in ["qwen35_35b_a3b", "gemma4_31b_it"]:
        t = (f"{MODELS[m].think_open}step 1"
             f"{MODELS[m].think_close}Final answer: X")
        print(f"  [{m}] strip: {strip_thinking(m, t)!r}")
