#!/usr/bin/env python3
"""
Sample responses from any configured model × dataset via a vLLM OpenAI-compatible server.

Models (keys defined in prompt.py):
  qwen35_35b_a3b, glm41v_9b_thinking, r1_distill_qwen_32b, gemma4_31b_it
Datasets:
  aime, gpqa, livecodebench

Modes:
  baseline  — single-call CoT generation
  finding   — continuation-based generation with CF injection on truncated rounds

Usage (model_key and dataset_key are REQUIRED):
    python run_sampling.py --group baseline \\
        --model-key qwen35_35b_a3b --dataset-key aime \\
        --out outputs/raw_baseline.jsonl
"""

import argparse
import json
import os
import re
import time
from pathlib import Path

import requests
from tqdm import tqdm
from transformers import AutoTokenizer

from config import (
    FINDING_STEPS,
    MAX_MODEL_LEN,
    MAX_STEPS,
    MAX_TOKENS,
    NUM_SAMPLES,
    TEMPERATURE,
    TOP_P,
    VLLM_PORT,
)
from prompt import (
    DATASETS,
    MODELS,
    adaptive_finding_cap,
    apply_chat_template as apply_chat_template_mk,
    build_baseline_prompt,
    build_cf_extract_prompt,
    build_finding_base_prompt,
    build_methodb_extract_prompt,
    extract_answer,
    find_last_anchor_end,
    format_cf_injection,
    format_methodb_injection,
    grade_answer,
    ground_truth,
    parse_cf_output,
    parse_methodb_findings,
    split_at_think_close,
    strip_thinking as strip_thinking_mk,
    thinking_body_only,
    thinking_is_closed,
    trim_to_last_sentence,
)

VLLM_BASE = f"http://127.0.0.1:{VLLM_PORT}"


# ---------------------------------------------------------------------------
# vLLM server interaction
# ---------------------------------------------------------------------------

def wait_for_server(base_url: str, timeout: int = 600):
    start = time.time()
    while time.time() - start < timeout:
        try:
            r = requests.get(f"{base_url}/v1/models", timeout=5)
            if r.status_code == 200:
                print(f"vLLM server ready at {base_url}")
                return
        except requests.ConnectionError:
            pass
        time.sleep(3)
    raise TimeoutError(f"vLLM server not ready after {timeout}s")


def completions_request(
    prompts: list[str],
    n: int = 1,
    max_tokens: int = MAX_TOKENS,
    base_url: str = VLLM_BASE,
    model: str = "",
) -> list[list[dict]]:
    """Send a batch /v1/completions request, return grouped outputs.
    Each outer list element corresponds to one prompt; inner list has n dicts
    with keys: text, token_count."""
    payload = {
        "model": model,
        "prompt": prompts,
        "max_tokens": max_tokens,
        "temperature": TEMPERATURE,
        "top_p": TOP_P,
        "n": n,
        "skip_special_tokens": False,
    }
    resp = requests.post(f"{base_url}/v1/completions", json=payload,
                         timeout=int(os.environ.get("FBE_HTTP_TIMEOUT", 1800)))
    resp.raise_for_status()
    data = resp.json()

    grouped: list[list[dict]] = [[] for _ in prompts]
    for c in data.get("choices", []):
        idx = c.get("index", 0)
        # For batch prompts, vLLM uses prompt_index (if available)
        prompt_idx = c.get("prompt_index", idx // n if n > 0 else 0)
        if 0 <= prompt_idx < len(grouped):
            text = c.get("text", "")
            # Count completion tokens from usage or estimate
            token_count = c.get("usage", {}).get("completion_tokens", 0)
            finish_reason = c.get("finish_reason", None)
            grouped[prompt_idx].append({
                "text": text,
                "token_count": token_count,
                "finish_reason": finish_reason,
            })

    return grouped


def completions_request_robust(
    prompts: list[str],
    n: int = 1,
    max_tokens: int = MAX_TOKENS,
    base_url: str = VLLM_BASE,
    model: str = "",
    max_retries: int = 3,
) -> list[list[dict]]:
    """completions_request with retry on transient errors."""
    for attempt in range(max_retries):
        try:
            return completions_request(prompts, n, max_tokens, base_url, model)
        except (requests.ConnectionError, requests.Timeout) as e:
            if attempt < max_retries - 1:
                wait = 10 * (attempt + 1)
                print(f"  Request error ({e}), retrying in {wait}s...")
                time.sleep(wait)
            else:
                raise


# ---------------------------------------------------------------------------
# Token counting: vLLM /v1/completions doesn't always return per-choice
# token counts. We estimate from the tokenizer as fallback.
# ---------------------------------------------------------------------------

_tokenizer_cache = {}

def count_tokens(tokenizer, text: str) -> int:
    return len(tokenizer(text, add_special_tokens=False).input_ids)


def calc_max_tokens(tokenizer, prompts: list[str]) -> int:
    """Dynamically compute max_tokens so prompt + completion fits in MAX_MODEL_LEN."""
    max_prompt_len = max(count_tokens(tokenizer, p) for p in prompts)
    available = MAX_MODEL_LEN - max_prompt_len - 100  # safety margin for tokenizer differences
    return min(MAX_TOKENS, available)


# ---------------------------------------------------------------------------
# Baseline (CoT) sampling — single call
# ---------------------------------------------------------------------------

def run_baseline(tokenizer, dataset: list[dict], num_samples: int,
                 out_file, done_keys: set, base_url: str, model: str,
                 batch_size: int = 256,
                 model_key: str = "qwen35_35b_a3b",
                 dataset_key: str = "aime") -> int:
    """Single-call CoT, batched across problems. Writes records incrementally."""
    written = 0

    work = []  # each: (pid, gt, sample_idx, prompt)
    for row_idx, problem in enumerate(dataset):
        pid = (problem.get("id") or problem.get("question_id")
               or f"row_{row_idx}")
        gt = ground_truth(dataset_key, problem)
        user_content = build_baseline_prompt(dataset_key, problem)
        prompt = apply_chat_template_mk(model_key, tokenizer, user_content)
        for s in range(1, num_samples + 1):
            if (pid, s) in done_keys:
                continue
            work.append((pid, gt, s, prompt))

    if not work:
        return 0

    print(f"Baseline: {len(work)} prompts to generate, batch_size={batch_size}")

    for start in tqdm(range(0, len(work), batch_size), desc="Baseline batches"):
        chunk = work[start:start + batch_size]
        prompts = [w[3] for w in chunk]
        mt = calc_max_tokens(tokenizer, prompts)

        outputs = completions_request_robust(prompts, n=1, max_tokens=mt,
                                              base_url=base_url, model=model)

        for (pid, gt, sample_i, _prompt), out_group in zip(chunk, outputs):
            text = out_group[0]["text"] if out_group else ""
            token_count = out_group[0].get("token_count", 0) if out_group else 0
            if token_count == 0:
                token_count = count_tokens(tokenizer, text)

            post_think = strip_thinking_mk(model_key, text)
            predicted = extract_answer(dataset_key, post_think)
            correct = grade_answer(dataset_key, predicted, gt)

            record = {
                "problem_id": pid,
                "sample_idx": sample_i,
                "group": "baseline",
                "gt_answer": gt,
                "predicted_answer": predicted,
                "correct": correct,
                "completion_tokens": token_count,
                "num_steps": 1,
                "response": text,
            }
            out_file.write(json.dumps(record, ensure_ascii=False) + "\n")
            written += 1
        out_file.flush()
        write_pretty_snapshot(out_file.name)

    return written


# ---------------------------------------------------------------------------
# Finding (multi-round) sampling
# ---------------------------------------------------------------------------

MIN_GEN_TOKENS = 512  # minimum generation budget; below this, skip the sample

# Continuation-based finding config
# Finding now runs UNLIMITED rounds — each ROUND_MAX_TOKENS (12000) new tokens,
# continuing until the model naturally emits <Answer> OR the accumulated
# prompt exceeds the model context window. NUM_ROUNDS_SAFETY_CAP is just a
# hard upper bound so a stuck sample doesn't loop forever.
NUM_ROUNDS_SAFETY_CAP = int(os.environ.get("FBE_NUM_ROUNDS_SAFETY_CAP", 50))
ROUND_MAX_TOKENS = int(os.environ.get("FBE_ROUND_MAX_TOKENS", 12000))
CF_MAX_TOKENS = int(os.environ.get("FBE_CF_MAX_TOKENS", 50000))
# Every round writes a full debug snapshot for all watched samples; plus a
# separate single-sample snapshot (randomly picked from ALL samples).
DEBUG_SNAPSHOT_EVERY = int(os.environ.get("FBE_DEBUG_SNAPSHOT_EVERY", 1))
# Drop findings whose body starts with a label/control-token keyword —
# these are label leakage from extractor repetition loops or thinking-token
# residue, not real findings.
_BAD_CF_PREFIX = re.compile(
    r"^\s*(?:ANCHOR\b|FINDING\b|</?think>|<\|?channel)",
    re.IGNORECASE,
)
# Meta-leak detector: catches samples where the extractor echoed the RULES
# text back as 'findings' instead of extracting domain facts. Matches the
# characteristic phrases that only appear in the extractor's RULES section.
_META_LEAK_RE = re.compile(
    r"(?i)("
    r"AT\s+MOST\s+\d+\s+findings|"
    r"Pick\s+the\s+\d+\s+most\s+load[-\s]bearing|"
    r"load[-\s]bearing|"
    r"one\s+short\s+sentence|"
    r"plain\s+bullet\s+list|"
    r"No\s+preamble|No\s+closing\s+remarks|"
    r"Do\s+NOT\s+re-?derive|Do\s+NOT\s+evaluate\s+correctness|"
    r"Do\s+NOT\s+restate\s+the\s+final\s+answer|"
    r"each\s+line\s+starting\s+with|"
    r"The\s+RULES\s+above\s+describe"
    r")"
)


def findings_have_issues(new_findings, prior_findings=None):
    """Return (is_bad, reason) if findings need a retry.

    Flags META-leaks (RULES echoed as findings) and cases where every new
    finding is a near-duplicate of a prior one (no genuinely new progress).
    """
    if not new_findings:
        return True, "empty"
    # META leak: RULES text echoed as findings
    for f in new_findings:
        if _META_LEAK_RE.search(f):
            return True, "meta_leak"
    # Duplicate-of-prior check (only when prior_findings provided)
    if prior_findings:
        def _norm(s):
            return re.sub(r"\s+", " ", s).strip().lower()[:80]
        prior_set = {_norm(f) for f in prior_findings}
        dup_count = sum(1 for f in new_findings if _norm(f) in prior_set)
        if dup_count == len(new_findings):
            return True, "all_duplicates_of_prior"
    return False, None


# kept for backwards-compat env (ignored by new unlimited-rounds logic):
NUM_ROUNDS = int(os.environ.get("FBE_NUM_ROUNDS", NUM_ROUNDS_SAFETY_CAP))


def write_pretty_snapshot(jsonl_path: Path) -> None:
    """Read a jsonl file and dump it as a pretty-indented JSON array alongside
    as `<name>_pretty.json`. Safe to call repeatedly during a run so someone
    can inspect intermediate progress."""
    try:
        p = Path(jsonl_path)
        if not p.exists():
            return
        rows = []
        with p.open() as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
        pretty_path = p.with_name(f"{p.stem}_pretty.json")
        with pretty_path.open("w", encoding="utf-8") as f:
            json.dump(rows, f, ensure_ascii=False, indent=2)
    except Exception as e:
        # Snapshot failures must never block the run
        print(f"  [warn] write_pretty_snapshot({jsonl_path}) failed: {e}")


def run_finding(tokenizer, dataset: list[dict], num_samples: int,
                finding_steps: int, max_steps: int,
                out_file, done_keys: set, base_url: str, model: str,
                batch_size: int = 256, debug_log_path=None,
                debug_per_case: int = 2,
                model_key: str = "qwen35_35b_a3b",
                dataset_key: str = "aime") -> int:
    """Continuation-based finding.

    Each round: let the model keep generating where it left off, capped at
    ROUND_MAX_TOKENS new tokens. After each round:
      case 1: model emitted EOS naturally (token_count < ROUND_MAX_TOKENS) → done
      case 2: hit cap AND output closed </think>  → continue, no CF extraction
      case 3: hit cap AND no </think> this round → summarize this round's output
              into a 1-2 sentence critical finding, and inject
              "(Original question + all CFs so far)" right after the accumulated
              raw thinking; model continues from that point.
    """
    written = 0
    skipped_overlength = 0

    samples = []
    for row_idx, problem in enumerate(dataset):
        pid = (problem.get("id") or problem.get("question_id")
               or f"row_{row_idx}")
        gt = ground_truth(dataset_key, problem)
        for s_idx in range(1, num_samples + 1):
            if (pid, s_idx) in done_keys:
                continue
            samples.append({
                "pid": pid,
                "gt": gt,
                "problem": problem,
                "sample_idx": s_idx,
                "raw_generation": "",          # concat of raw outputs only (no injections)
                "critical_findings": [],        # one CF per round where case 3 triggered
                "round_outputs": [],            # per-round {text, finish_reason, cf}
                "thinking_closed": False,
                "done": False,
                "rounds_run": 0,
                "total_gen_tokens": 0,
                "last_prompt_tokens": 0,
                "last_gen_tokens": 0,
                "skipped": False,
                # --- debug trace ---
                "debug_trace": [],              # full per-round records when debug enabled
            })

    # Anchor-match statistics (logged at end)
    anchor_stats = {
        "cf_extractions": 0,      # total CF extraction calls
        "parsed_ok": 0,           # CF output parsed at least one (anchor,finding) pair
        "anchor_verbatim_hit": 0, # raw_generation trimmed by exact anchor match
        "anchor_fuzzy_hit": 0,    # trimmed by fuzzy (prefix) match
        "anchor_no_match": 0,     # all anchors missed → fell back to sentence trim
        "no_progress": 0,         # model explicitly said NO_PROGRESS
    }

    # Debug: pick a handful of samples and record everything they go through.
    # Log file is rewritten from scratch at the end of each round so progress
    # is visible while the run is still in flight.
    debug_watch = set()
    if debug_log_path:
        seen_pids = {}
        for idx, s in enumerate(samples):
            seen_pids[s["pid"]] = seen_pids.get(s["pid"], 0) + 1
            if seen_pids[s["pid"]] <= 2:   # first 2 samples of each problem
                debug_watch.add(idx)
        print(f"  [debug] tracing {len(debug_watch)} samples; log → {debug_log_path}")

    def _dump_debug(turn_n=None):
        if not debug_log_path:
            return
        if turn_n is not None:
            base = Path(debug_log_path)
            out_path = base.with_name(f"{base.stem}_turn_{turn_n}{base.suffix}")
        else:
            out_path = Path(debug_log_path)
        records = []
        for idx in sorted(debug_watch):
            s = samples[idx]
            post = strip_thinking_mk(model_key, s["raw_generation"])
            pred = extract_answer(dataset_key, post)
            records.append({
                "pid": s["pid"],
                "sample_idx": s["sample_idx"],
                "gt": s["gt"],
                "question": s["problem"].get("question", ""),
                "final_predicted": pred,
                "final_correct": grade_answer(dataset_key, pred, s["gt"]),
                "rounds_run": s["rounds_run"],
                "thinking_closed": s["thinking_closed"],
                "done": s["done"],
                "total_gen_tokens": s["total_gen_tokens"],
                "critical_findings": s["critical_findings"],
                "rounds": s["debug_trace"],
            })
        # Write both jsonl (one per line) and pretty indented JSON array so
        # you can eyeball progress without converting.
        with open(out_path, "w", encoding="utf-8") as dbg:
            for rec in records:
                dbg.write(json.dumps(rec, ensure_ascii=False) + "\n")
        pretty_path = out_path.with_name(f"{out_path.stem}_pretty.json")
        with open(pretty_path, "w", encoding="utf-8") as f:
            json.dump(records, f, ensure_ascii=False, indent=2)

    def _dump_random_sample(turn_n):
        """After each round, pick ONE random sample from the full set and dump
        its current state to a separate file, so we can eyeball a concrete
        sample's trajectory without wading through all 120 watched ones.
        Prefers ACTIVE (not yet done) samples when possible."""
        if not debug_log_path or not samples:
            return
        import random
        # Prefer an active (still running) sample; fall back to any sample
        active_pool = [i for i in range(len(samples)) if not samples[i]["done"]]
        pool = active_pool if active_pool else list(range(len(samples)))
        pick = random.choice(pool)
        s = samples[pick]
        post = strip_thinking_mk(model_key, s["raw_generation"])
        pred = extract_answer(dataset_key, post)
        rec = {
            "sampled_from": "active" if active_pool else "all",
            "total_samples": len(samples),
            "active_samples_this_turn": len(active_pool),
            "pid": s["pid"],
            "sample_idx": s["sample_idx"],
            "gt": s["gt"],
            "question": s["problem"].get("question", ""),
            "current_predicted": pred,
            "current_correct": grade_answer(dataset_key, pred, s["gt"]),
            "rounds_run": s["rounds_run"],
            "thinking_closed": s["thinking_closed"],
            "done": s["done"],
            "total_gen_tokens": s["total_gen_tokens"],
            "critical_findings": s["critical_findings"],
            "raw_generation": s["raw_generation"],
            "round_outputs": s["round_outputs"],
            # debug_trace is only populated for watched samples, include if present
            "debug_trace": s.get("debug_trace") if pick in debug_watch else None,
        }
        base = Path(debug_log_path)
        rand_path = base.with_name(
            f"{base.stem}_turn_{turn_n}_random_sample_pretty.json")
        with open(rand_path, "w", encoding="utf-8") as f:
            json.dump(rec, f, ensure_ascii=False, indent=2)

    if not samples:
        return 0

    print(f"Finding: {len(samples)} samples, rounds=UNLIMITED "
          f"(safety cap {NUM_ROUNDS_SAFETY_CAP}), "
          f"round_max_tokens={ROUND_MAX_TOKENS}, batch_size={batch_size}")

    active_idxs = list(range(len(samples)))

    round_n = 0
    while active_idxs and round_n < NUM_ROUNDS_SAFETY_CAP:
        round_n += 1
        print(f"  === Round {round_n}: {len(active_idxs)} active samples ===")

        # 1. Build continuation prompts
        batch_prompts = []
        batch_active = []
        for i in active_idxs:
            s = samples[i]
            # Base prompt = finding-header CoT prompt; chat template may auto-open <think>
            user_content = build_finding_base_prompt(dataset_key, s["problem"])
            base_prompt = apply_chat_template_mk(model_key, tokenizer, user_content)

            # When injecting CFs, trim raw_generation back to the last complete
            # sentence so the CF block doesn't follow a half-cut line. Update
            # raw_generation in-place so the sample's recorded state matches
            # what the model actually continues from.
            if not s["thinking_closed"] and s["critical_findings"]:
                s["raw_generation"] = trim_to_last_sentence(s["raw_generation"])
                continuation = (s["raw_generation"]
                                + format_cf_injection(dataset_key, s["problem"],
                                                      s["critical_findings"]))
            else:
                continuation = s["raw_generation"]

            prompt = base_prompt + continuation
            prompt_tok = count_tokens(tokenizer, prompt)
            if prompt_tok + ROUND_MAX_TOKENS + 100 >= MAX_MODEL_LEN:
                s["done"] = True
                s["skipped"] = True
                skipped_overlength += 1
                continue
            s["last_prompt_tokens"] = prompt_tok
            batch_prompts.append(prompt)
            batch_active.append(i)
            if i in debug_watch:
                s["debug_trace"].append({
                    "round": round_n,
                    "prompt_tokens": prompt_tok,
                    "prompt_full": prompt,
                })

        if not batch_active:
            break

        # 2. Generate this round. Every round uses ROUND_MAX_TOKENS (12000)
        #    until the model naturally emits <Answer>. No special last-round
        #    logic — there is no "last round" anymore (loop runs until done).
        all_outputs = []
        for start in tqdm(range(0, len(batch_prompts), batch_size),
                           desc=f"  round {round_n} gen", leave=False):
            chunk_prompts = batch_prompts[start:start + batch_size]
            chunk_out = completions_request_robust(chunk_prompts, n=1,
                                                    max_tokens=ROUND_MAX_TOKENS,
                                                    base_url=base_url, model=model)
            all_outputs.extend(chunk_out)

        # 3. Classify each sample's outcome, collect those needing CF extraction
        need_cf = []  # list of (sample_idx, this_round_text)
        for rel_idx, i in enumerate(batch_active):
            s = samples[i]
            out_group = all_outputs[rel_idx] if rel_idx < len(all_outputs) else []
            chunk = out_group[0]["text"] if out_group else ""
            tok = out_group[0].get("token_count", 0) if out_group else 0
            finish = out_group[0].get("finish_reason") if out_group else None
            if tok == 0:
                tok = count_tokens(tokenizer, chunk)

            s["rounds_run"] += 1
            s["total_gen_tokens"] += tok
            s["last_gen_tokens"] = tok
            s["raw_generation"] += chunk
            entry = {"text": chunk, "finish_reason": finish, "cf": None,
                     "token_count": tok}
            s["round_outputs"].append(entry)

            # Classify outcome based on finish_reason (authoritative)
            has_end_think_overall = thinking_is_closed(model_key, s["raw_generation"])

            if finish == "stop":
                # Case 1: model emitted EOS this round — naturally finished
                s["done"] = True
            elif finish == "length":
                # Hit ROUND_MAX_TOKENS — round was truncated
                if has_end_think_overall and not s["thinking_closed"]:
                    # Case 2: </think> appeared this round (or earlier); mark
                    # thinking as closed and continue without CF extraction.
                    s["thinking_closed"] = True
                elif s["thinking_closed"]:
                    # Already past </think>, still truncated in answer phase;
                    # just continue next round without injecting.
                    pass
                else:
                    # Case 3: still in thinking, extract CF from this round.
                    need_cf.append((i, chunk))
            else:
                # Unknown finish_reason (shouldn't happen); safety-terminate
                s["done"] = True

            # Record debug info for this round
            if i in debug_watch and s["debug_trace"]:
                last = s["debug_trace"][-1]
                if finish == "stop":
                    case_name = "CASE 1 (natural stop, done)"
                elif finish == "length" and has_end_think_overall and (not s["thinking_closed"] or last["round"] == s["rounds_run"]):
                    # thinking just closed this round OR had closed earlier
                    case_name = "CASE 2 (hit cap, </think> seen, no CF)"
                elif finish == "length" and s["thinking_closed"]:
                    case_name = "CASE 2-continued (hit cap, past </think>, no CF)"
                elif finish == "length":
                    case_name = "CASE 3 (hit cap, still thinking, CF extracted)"
                else:
                    case_name = f"UNKNOWN (finish={finish})"
                last.update({
                    "output_text": chunk,
                    "output_tokens": tok,
                    "finish_reason": finish,
                    "case": case_name,
                    "thinking_closed_after": s["thinking_closed"],
                    "done_after": s["done"],
                })

        # 4. Batch-extract critical findings for case-3 samples.
        #    The extractor produces (ANCHOR, FINDING) pairs. We use the
        #    latest ANCHOR's position to trim raw_generation back to a
        #    semantically-meaningful endpoint (everything after the last
        #    anchor — usually an incomplete exploration — is discarded).
        if need_cf:
            cf_prompts = []
            for i, chunk in need_cf:
                cf_user = build_cf_extract_prompt(
                    chunk, dataset_key, samples[i]["problem"],
                    prior_findings=samples[i]["critical_findings"])
                # Thinking-on is fine — we strip <think>…</think> from output
                cf_prompts.append(
                    apply_chat_template_mk(model_key, tokenizer, cf_user)
                )
            cf_outputs = []
            for start in tqdm(range(0, len(cf_prompts), batch_size),
                               desc=f"  round {round_n} cf", leave=False):
                sub = cf_prompts[start:start + batch_size]
                max_p = max(count_tokens(tokenizer, p) for p in sub)
                cf_mt = min(CF_MAX_TOKENS, MAX_MODEL_LEN - max_p - 100)
                out = completions_request_robust(sub, n=1,
                                                  max_tokens=cf_mt,
                                                  base_url=base_url, model=model)
                cf_outputs.extend(out)
            for (i, _), out_group in zip(need_cf, cf_outputs):
                cf_raw = out_group[0]["text"] if out_group else ""
                cf_post = strip_thinking_mk(model_key, cf_raw).strip()
                anchor_stats["cf_extractions"] += 1

                # Detect explicit "no progress" signal
                if cf_post.strip().upper().startswith("NO_PROGRESS"):
                    anchor_stats["no_progress"] += 1
                    # Trim to last sentence as fallback
                    samples[i]["raw_generation"] = trim_to_last_sentence(
                        samples[i]["raw_generation"])
                    trim_method = "no_progress_sentence_trim"
                    new_cfs, anchors = [], []
                else:
                    anchors, findings = parse_cf_output(cf_post)
                    if findings or anchors:
                        anchor_stats["parsed_ok"] += 1

                    # Try to trim raw_generation by anchor
                    end_pos = find_last_anchor_end(
                        samples[i]["raw_generation"], anchors)

                    # Determine match method for stats
                    trim_method = "no_anchor_fallback_sentence_trim"
                    if end_pos > 0 and anchors:
                        # Re-run to know if verbatim or fuzzy
                        verbatim_hit = any(
                            a.strip() in samples[i]["raw_generation"]
                            for a in anchors if len(a.strip()) >= 10)
                        if verbatim_hit:
                            anchor_stats["anchor_verbatim_hit"] += 1
                            trim_method = "anchor_verbatim"
                        else:
                            anchor_stats["anchor_fuzzy_hit"] += 1
                            trim_method = "anchor_fuzzy"
                        samples[i]["raw_generation"] = samples[i]["raw_generation"][:end_pos]
                    else:
                        anchor_stats["anchor_no_match"] += 1
                        # Fallback: at least trim to last complete sentence
                        samples[i]["raw_generation"] = trim_to_last_sentence(
                            samples[i]["raw_generation"])

                    new_cfs = findings if findings else ([cf_post[:800]] if cf_post else [])

                new_cfs = [c for c in new_cfs if c and not _BAD_CF_PREFIX.match(c)]
                new_cfs = new_cfs[:1]

                if not new_cfs:
                    new_cfs = ["(no finding produced)"]
                for cf_text in new_cfs:
                    if len(cf_text) > 800:
                        cf_text = cf_text[:800].rstrip() + "..."
                    samples[i]["critical_findings"].append(cf_text)

                samples[i]["round_outputs"][-1]["cf"] = new_cfs
                if i in debug_watch and samples[i]["debug_trace"]:
                    samples[i]["debug_trace"][-1]["cf_extracted"] = new_cfs
                    samples[i]["debug_trace"][-1]["cf_anchors"] = anchors
                    samples[i]["debug_trace"][-1]["cf_raw"] = cf_raw
                    samples[i]["debug_trace"][-1]["trim_method"] = trim_method

        # 5. Flush debug snapshots:
        #    - full watched snapshot every DEBUG_SNAPSHOT_EVERY rounds
        #    - a single random sample's state every round (separate file)
        if round_n % DEBUG_SNAPSHOT_EVERY == 0:
            _dump_debug(turn_n=round_n)
        _dump_random_sample(turn_n=round_n)

        # 6. Incrementally write any samples that just became done this round
        #    so the jsonl (and pretty snapshot) reflect progress mid-run.
        for i in active_idxs:
            s = samples[i]
            if not s["done"] or s.get("_written"):
                continue
            post_think = strip_thinking_mk(model_key, s["raw_generation"])
            predicted = extract_answer(dataset_key, post_think)
            correct = grade_answer(dataset_key, predicted, s["gt"])
            record = {
                "problem_id": s["pid"],
                "sample_idx": s["sample_idx"],
                "group": "finding",
                "gt_answer": s["gt"],
                "predicted_answer": predicted,
                "correct": correct,
                "completion_tokens": s["total_gen_tokens"],
                "last_round_prompt_tokens": s["last_prompt_tokens"],
                "last_round_gen_tokens": s["last_gen_tokens"],
                "last_round_total_tokens": s["last_prompt_tokens"] + s["last_gen_tokens"],
                "num_steps": s["rounds_run"],
                "thinking_closed": s["thinking_closed"],
                "response": s["raw_generation"],
                "all_step_outputs": [r["text"] for r in s["round_outputs"]],
                "round_finish_reasons": [r["finish_reason"] for r in s["round_outputs"]],
                "findings": s["critical_findings"],
                "skipped_overlength": s["skipped"],
            }
            out_file.write(json.dumps(record, ensure_ascii=False) + "\n")
            written += 1
            s["_written"] = True
        out_file.flush()
        write_pretty_snapshot(out_file.name)

        # 7. Update active list
        active_idxs = [i for i in active_idxs if not samples[i]["done"]]

    # After all rounds: write any remaining samples that finished in the last
    # iteration or hit the round cap without becoming done.
    for s in samples:
        if s.get("_written"):
            continue
        post_think = strip_thinking_mk(model_key, s["raw_generation"])
        predicted = extract_answer(dataset_key, post_think)
        correct = grade_answer(dataset_key, predicted, s["gt"])
        record = {
            "problem_id": s["pid"],
            "sample_idx": s["sample_idx"],
            "group": "finding",
            "gt_answer": s["gt"],
            "predicted_answer": predicted,
            "correct": correct,
            "completion_tokens": s["total_gen_tokens"],
            "last_round_prompt_tokens": s["last_prompt_tokens"],
            "last_round_gen_tokens": s["last_gen_tokens"],
            "last_round_total_tokens": s["last_prompt_tokens"] + s["last_gen_tokens"],
            "num_steps": s["rounds_run"],
            "thinking_closed": s["thinking_closed"],
            "response": s["raw_generation"],
            "all_step_outputs": [r["text"] for r in s["round_outputs"]],
            "round_finish_reasons": [r["finish_reason"] for r in s["round_outputs"]],
            "findings": s["critical_findings"],
            "skipped_overlength": s["skipped"],
        }
        out_file.write(json.dumps(record, ensure_ascii=False) + "\n")
        written += 1
        s["_written"] = True
    out_file.flush()
    write_pretty_snapshot(out_file.name)

    if skipped_overlength:
        print(f"  WARNING: {skipped_overlength} samples skipped due to prompt overflow")

    # Anchor-match stats summary
    if anchor_stats["cf_extractions"] > 0:
        total = anchor_stats["cf_extractions"]
        verbatim = anchor_stats["anchor_verbatim_hit"]
        fuzzy = anchor_stats["anchor_fuzzy_hit"]
        nomatch = anchor_stats["anchor_no_match"]
        noprog = anchor_stats["no_progress"]
        parsed = anchor_stats["parsed_ok"]
        print(f"  [cf-stats] extractions={total}  parsed_ok={parsed} "
              f"({parsed/total:.1%})")
        print(f"  [cf-stats]   trim_by_anchor_verbatim: {verbatim} "
              f"({verbatim/total:.1%})")
        print(f"  [cf-stats]   trim_by_anchor_fuzzy:    {fuzzy} "
              f"({fuzzy/total:.1%})")
        print(f"  [cf-stats]   trim_by_sentence_fallback: {nomatch} "
              f"({nomatch/total:.1%})")
        print(f"  [cf-stats]   no_progress_declared:    {noprog} "
              f"({noprog/total:.1%})")

    # Final debug dump (also flushed every round above)
    _dump_debug()
    if debug_log_path:
        print(f"  [debug] wrote {len(debug_watch)} full traces → {debug_log_path}")

    return written


# ---------------------------------------------------------------------------
# Method B (post-hoc): reuse baseline round-1 + single self-verification continuation
# ---------------------------------------------------------------------------

METHODB_EXTRACT_MAX_TOKENS = int(os.environ.get(
    "FBE_INSIGHTREPLAY_EXTRACT_MAX_TOKENS",
    os.environ.get("FBE_METHODB_EXTRACT_MAX_TOKENS", 10000)))
METHODB_CONT_MAX_TOKENS = int(os.environ.get(
    "FBE_INSIGHTREPLAY_CONT_MAX_TOKENS",
    os.environ.get("FBE_METHODB_CONT_MAX_TOKENS", 30000)))

# --- verify_only ablation: naked "wait"-style re-think prompt spliced in before
# the thinking close marker, with no problem restatement, no findings, and no
# baseline_answer anchor. Isolates the contribution of the findings+anchor
# content from the contribution of (extra continuation budget + generic
# self-reflection). Same 30k continuation budget as full methodb.
VERIFY_ONLY_INJECTION = (
    "\n\nWait, let me double-check this before finalizing.\n\n"
)


def load_baseline_records(path: Path) -> dict:
    """Load baseline jsonl keyed by (problem_id, sample_idx)."""
    out = {}
    if not path.exists():
        return out
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            key = (r.get("problem_id"), r.get("sample_idx"))
            out[key] = r
    return out


def run_methodb_finding(tokenizer, dataset: list, num_samples: int,
                        out_file, done_keys: set, base_url: str, model: str,
                        baseline_in: Path, batch_size: int = 256,
                        debug_log_path=None,
                        model_key: str = "r1_distill_qwen_32b",
                        dataset_key: str = "aime",
                        ablation: str = "none",
                        turns: int = 1,
                        insightreplay_in: "Path | None" = None) -> int:
    """Post-hoc Method B: reuse baseline round-1 output; extract findings from
    the completed <think> block; splice a self-verification block in BEFORE
    the close marker; let the model continue to a second close + final answer.

    ablation="verify_only" skips the extractor and replaces the full
    methodb injection with a naked re-think prompt (see VERIFY_ONLY_INJECTION).
    Continuation budget and fallback logic are unchanged, so the only
    difference vs full methodb is the injection content.

    ``turns``: 1 (default) = original single insightreplay pass, unchanged.
    3 = after the first replay, perform 2 additional extract+replay rounds
    with 'one more time' / 'for the last time' wording. Each turn's extractor
    sees the full accumulated thinking (baseline + all prior injections +
    all prior new thinking) and is shown all prior findings so it reports
    NEW conclusions. A turn is only committed if the extractor produces
    findings, the continuation prompt fits in context, and the continuation
    emits a thinking-close marker; otherwise the sample stays at the
    previous successfully-committed turn.  Only supported when
    ablation == 'none'.
    """
    assert ablation in ("none", "verify_only"), f"unknown ablation: {ablation}"
    assert turns in (1, 3), f"unsupported turns={turns} (must be 1 or 3)"
    if turns != 1 and ablation != "none":
        raise ValueError(
            "turns > 1 is only supported with ablation='none'; got "
            f"ablation={ablation}, turns={turns}")
    baseline_by_key = load_baseline_records(baseline_in)
    if not baseline_by_key:
        raise FileNotFoundError(
            f"--baseline-in not usable: {baseline_in} (missing or empty)")
    print(f"  [methodb] loaded {len(baseline_by_key)} baseline records "
          f"from {baseline_in}")

    # Build sample list mirrored after run_finding
    samples = []
    missing = 0
    for row_idx, problem in enumerate(dataset):
        pid = (problem.get("id") or problem.get("question_id")
               or f"row_{row_idx}")
        gt = ground_truth(dataset_key, problem)
        for s_idx in range(1, num_samples + 1):
            if (pid, s_idx) in done_keys:
                continue
            base = baseline_by_key.get((pid, s_idx))
            if base is None:
                missing += 1
                continue
            samples.append({
                "pid": pid,
                "gt": gt,
                "problem": problem,
                "sample_idx": s_idx,
                "baseline_response": base.get("response", ""),
                "baseline_tokens": base.get("completion_tokens", 0),
                "baseline_correct": bool(base.get("correct", False)),
                "baseline_predicted": base.get("predicted_answer"),
                "pre_close": None,
                "post_close": None,
                "thinking_body": None,
                "thinking_tokens": 0,
                "cap": 0,
                "findings": [],
                "continuation": "",
                "continuation_tokens": 0,
                "continuation_finish": None,
                "full_response": "",
                "mode": "pending",   # baseline_fallback_no_close | extract_overflow_fallback |
                                     # ctx_overflow_fallback | no_findings_fallback |
                                     # continuation_length_fallback | methodb
            })
    if missing:
        print(f"  [methodb] WARNING: {missing} (pid,sample) not found in baseline")
    if not samples:
        print("  [methodb] nothing to do")
        return 0

    cfg = MODELS[model_key]
    user_contents = {}
    chat_prompts = {}
    for s in samples:
        pid = s["pid"]
        if pid not in user_contents:
            user_contents[pid] = build_baseline_prompt(dataset_key, s["problem"])
            chat_prompts[pid] = apply_chat_template_mk(
                model_key, tokenizer, user_contents[pid])
    print(f"  [methodb] {len(samples)} samples; chat templates cached for "
          f"{len(chat_prompts)} problems")

    # Anchor helper (hoisted so prior-turn reconstruction can reuse it).
    def _anchor(s):
        if dataset_key not in ("aime", "gpqa", "livecodebench", "hmmt"):
            return None
        ans = s.get("baseline_predicted")
        if not ans or not isinstance(ans, str):
            return None
        ans = ans.strip()
        if not ans or ans.lower() == "none":
            return None
        return ans

    # -------- Phase 0 (optional): reconstruct turn-1 state from a prior
    # 1-turn insightreplay run so we don't have to re-run extract + continue
    # for samples that already succeeded. Samples marked `_prior_loaded=True`
    # are skipped by Phase 1-4 below; the multi-turn block picks them up.
    # --------
    if insightreplay_in is not None:
        prior_records = load_baseline_records(insightreplay_in)
        print(f"  [methodb] prior insightreplay: loaded "
              f"{len(prior_records)} records from {insightreplay_in}")
        ok = fail_reconstruct = carried_fallback = 0
        close_marker = MODELS[model_key].think_close
        fallback_modes = (
            "baseline_fallback_no_close",
            "extract_overflow_fallback",
            "ctx_overflow_fallback",
            "no_findings_fallback",
            "continuation_length_fallback",
        )
        for s in samples:
            prior = prior_records.get((s["pid"], s["sample_idx"]))
            if prior is None:
                continue
            prior_mode = prior.get("mode", "")
            prior_resp = prior.get("response") or ""
            if prior_mode == "methodb":
                prior_findings = prior.get("findings") or []
                if not prior_findings:
                    fail_reconstruct += 1
                    continue
                # The original turn-1 run may or may not have passed a
                # baseline_answer anchor to format_methodb_injection (LCB runs
                # historically skipped it). Try both forms and use whichever
                # literally appears in the stored response.
                anchor_val = _anchor(s)
                I1_candidates = []
                if anchor_val is not None:
                    I1_candidates.append((anchor_val, format_methodb_injection(
                        dataset_key, s["problem"], prior_findings,
                        baseline_answer=anchor_val, variant="default")))
                I1_candidates.append((None, format_methodb_injection(
                    dataset_key, s["problem"], prior_findings,
                    baseline_answer=None, variant="default")))
                matched = None
                for anc, i1 in I1_candidates:
                    if i1 and i1 in prior_resp:
                        matched = (anc, i1)
                        break
                if matched is None:
                    fail_reconstruct += 1
                    continue
                matched_anchor, I1 = matched
                pos = prior_resp.find(I1)
                pre_close_rec = prior_resp[:pos]
                continuation_rec = prior_resp[pos + len(I1):]
                # continuation must carry a close marker, otherwise the
                # stored full_response can't be spliced further.
                if not close_marker or close_marker not in continuation_rec:
                    fail_reconstruct += 1
                    continue
                s["pre_close"] = pre_close_rec
                s["post_close"] = None  # unused downstream
                s["thinking_body"] = thinking_body_only(model_key, pre_close_rec)
                s["thinking_tokens"] = (
                    prior.get("thinking_tokens")
                    or count_tokens(tokenizer, s["thinking_body"]))
                s["cap"] = (prior.get("findings_cap")
                            or adaptive_finding_cap(s["thinking_tokens"]))
                s["findings"] = list(prior_findings)
                s["injection"] = I1
                s["continuation"] = continuation_rec
                s["continuation_tokens"] = prior.get("continuation_tokens", 0)
                s["continuation_finish"] = prior.get("continuation_finish")
                s["mode"] = "methodb"
                s["full_response"] = prior_resp
                s["_prior_loaded"] = True
                # matched_anchor tells us which I1 variant turn 1 used; we
                # log it for transparency but do NOT propagate it to turns
                # 2/3 — those always use the current _anchor convention
                # (option 2: consistent with the live code design).
                s["_turn1_anchor_used"] = matched_anchor
                ok += 1
            elif prior_mode in fallback_modes:
                # Previously fell back at turn 1; nothing to extend. Carry
                # the stored record forward unchanged.
                s["pre_close"] = None
                s["post_close"] = None
                s["thinking_body"] = ""
                s["thinking_tokens"] = prior.get("thinking_tokens", 0)
                s["cap"] = prior.get("findings_cap", 0)
                s["findings"] = prior.get("findings") or []
                s["injection"] = ""
                s["continuation"] = ""
                s["continuation_tokens"] = prior.get("continuation_tokens", 0)
                s["continuation_finish"] = prior.get("continuation_finish")
                s["mode"] = prior_mode
                s["full_response"] = prior_resp or s["baseline_response"]
                s["_prior_loaded"] = True
                carried_fallback += 1
            # else: unknown prior mode — treat as not-loaded, run from scratch
        print(f"  [methodb] prior reuse: {ok} turn-1 reconstructed, "
              f"{carried_fallback} fallback carried, "
              f"{fail_reconstruct} reconstruction failed (will re-run)")

    # -------- Phase 1: split + compute caps + decide fallback --------
    ext_indices = []   # indices into samples needing extractor call
    for idx, s in enumerate(samples):
        if s.get("_prior_loaded"):
            continue
        pre, post = split_at_think_close(model_key, s["baseline_response"])
        if pre is None:
            s["mode"] = "baseline_fallback_no_close"
            s["full_response"] = s["baseline_response"]
            continue
        s["pre_close"] = pre
        s["post_close"] = post
        body = thinking_body_only(model_key, pre)
        s["thinking_body"] = body
        s["thinking_tokens"] = count_tokens(tokenizer, body)
        s["cap"] = adaptive_finding_cap(s["thinking_tokens"])
        ext_indices.append(idx)

    print(f"  [methodb] phase1: {len(ext_indices)}/{len(samples)} candidates "
          f"for extraction; "
          f"{sum(1 for s in samples if s['mode']=='baseline_fallback_no_close')} "
          f"fallback (no close marker)")

    # -------- Phase 2: batched finding extraction --------
    # Send the FULL thinking body to the extractor. If the wrapped prompt
    # doesn't fit in MAX_MODEL_LEN, fall back to baseline for that sample
    # (no truncation — partial-body extraction can miss tail conclusions
    # and cause C→W regressions).
    EXTRACT_OUTPUT_RESERVE = 1024
    TARGET_MAX = MAX_MODEL_LEN - EXTRACT_OUTPUT_RESERVE
    extract_fallback = []
    if ablation == "verify_only":
        print(f"  [methodb] ablation=verify_only → skipping extractor; "
              f"{len(ext_indices)} candidates go straight to Phase 3 with "
              f"naked re-think injection")
    elif ext_indices:
        extract_prompts = []
        extract_sample_indices = []
        for idx in ext_indices:
            s = samples[idx]
            uc = build_methodb_extract_prompt(
                s["thinking_body"], dataset_key, s["problem"], s["cap"])
            prompt_text = apply_chat_template_mk(model_key, tokenizer, uc)
            ptok = count_tokens(tokenizer, prompt_text)
            if ptok > TARGET_MAX:
                extract_fallback.append(idx)
                s["mode"] = "extract_overflow_fallback"
                s["full_response"] = s["baseline_response"]
                continue
            extract_prompts.append(prompt_text)
            extract_sample_indices.append(idx)
        if extract_fallback:
            print(f"  [methodb] {len(extract_fallback)} samples exceed "
                  f"MAX_MODEL_LEN at extract phase → baseline fallback")

        all_ext_out = []
        for start in tqdm(range(0, len(extract_prompts), batch_size),
                           desc="  methodb extract", leave=False):
            sub = extract_prompts[start:start + batch_size]
            max_p = max(count_tokens(tokenizer, p) for p in sub)
            mt = MAX_MODEL_LEN - max_p - 100
            mt = max(256, min(METHODB_EXTRACT_MAX_TOKENS, mt))
            out = completions_request_robust(sub, n=1, max_tokens=mt,
                                              base_url=base_url, model=model)
            all_ext_out.extend(out)

        for rel, idx in enumerate(extract_sample_indices):
            s = samples[idx]
            out_group = all_ext_out[rel] if rel < len(all_ext_out) else []
            raw = out_group[0]["text"] if out_group else ""
            post_think = strip_thinking_mk(model_key, raw).strip()
            s["findings"] = parse_methodb_findings(post_think, s["cap"])

        # Extract-phase retry loop. Originally only active for R1-distill
        # (which ignores the bullet-list format ~17% of the time), but now
        # runs for all models because we also flag META-leaks (RULES echoed
        # as findings) and duplicates. Per-model retry budget still biased
        # to R1 because it has the highest parse-fail rate.
        if extract_sample_indices:
            idx_to_prompt = dict(zip(extract_sample_indices, extract_prompts))
            MAX_EXTRACT_RETRIES = (
                10 if model_key.startswith("r1_distill") else 3)
            for attempt in range(1, MAX_EXTRACT_RETRIES + 1):
                retry_indices = []
                retry_reasons: list[str] = []
                for idx in extract_sample_indices:
                    bad, reason = findings_have_issues(
                        samples[idx]["findings"], prior_findings=None)
                    if bad:
                        retry_indices.append(idx)
                        retry_reasons.append(reason or "")
                if not retry_indices:
                    break
                from collections import Counter as _C
                reason_summary = _C(retry_reasons)
                print(f"  [methodb] extract retry {attempt}/"
                      f"{MAX_EXTRACT_RETRIES}: "
                      f"{len(retry_indices)} samples need re-extract "
                      f"(reasons: {dict(reason_summary)})")
                retry_prompts = [idx_to_prompt[idx] for idx in retry_indices]
                retry_out = []
                for start in tqdm(
                        range(0, len(retry_prompts), batch_size),
                        desc=f"  methodb extract retry{attempt}", leave=False):
                    sub = retry_prompts[start:start + batch_size]
                    max_p = max(count_tokens(tokenizer, p) for p in sub)
                    mt = MAX_MODEL_LEN - max_p - 100
                    mt = max(256, min(METHODB_EXTRACT_MAX_TOKENS, mt))
                    out = completions_request_robust(
                        sub, n=1, max_tokens=mt,
                        base_url=base_url, model=model)
                    retry_out.extend(out)
                for rel, idx in enumerate(retry_indices):
                    s = samples[idx]
                    out_group = (retry_out[rel]
                                 if rel < len(retry_out) else [])
                    raw = out_group[0]["text"] if out_group else ""
                    post_think = strip_thinking_mk(model_key, raw).strip()
                    parsed = parse_methodb_findings(post_think, s["cap"])
                    # Only overwrite if the new parse is at least as good.
                    if parsed:
                        s["findings"] = parsed

    # -------- Phase 3: budget check + build continuation prompts --------
    # Anchor the verification on baseline's extracted answer for AIME/GPQA
    # (breaks bullet-list mimicry; LCB predicted_answer is full code → skip).
    # _anchor is defined above (hoisted for prior-turn reconstruction reuse).
    def _has_close_marker(text: str) -> bool:
        cfg = MODELS[model_key]
        if cfg.think_close and cfg.think_close in text:
            return True
        if cfg.alt_think_close and cfg.alt_think_close in text:
            return True
        return False

    cont_indices = []
    cont_prompts = []
    cont_budgets = []
    for idx in ext_indices:
        s = samples[idx]
        # Skip if already marked fallback by phase 2 (wrapper too big)
        if s["mode"] == "extract_overflow_fallback":
            continue
        if ablation == "verify_only":
            injection = VERIFY_ONLY_INJECTION
        else:
            injection = format_methodb_injection(
                dataset_key, s["problem"], s["findings"],
                baseline_answer=_anchor(s))
        new_prompt = chat_prompts[s["pid"]] + s["pre_close"] + injection
        prompt_tok = count_tokens(tokenizer, new_prompt)
        # Fixed continuation cap: cheap samples don't need more, pathological
        # samples shouldn't be given 100k+ rope that starves the batch and
        # blows the HTTP timeout. If even 30k doesn't fit in ctx → fallback.
        ctx_overflow = prompt_tok + METHODB_CONT_MAX_TOKENS + 100 > MAX_MODEL_LEN
        no_findings = (ablation != "verify_only") and (not s["findings"])
        if ctx_overflow or no_findings:
            s["mode"] = ("ctx_overflow_fallback" if ctx_overflow
                         else "no_findings_fallback")
            s["full_response"] = s["baseline_response"]
            continue
        s["mode"] = "verify_only" if ablation == "verify_only" else "methodb"
        s["injection"] = injection   # cache for Phase 4 / full_response build
        cont_indices.append(idx)
        cont_prompts.append(new_prompt)
        cont_budgets.append(METHODB_CONT_MAX_TOKENS)

    print(f"  [methodb] phase3: {len(cont_indices)} continue, "
          f"{sum(1 for s in samples if s['mode'].endswith('fallback'))} "
          f"fallback")

    # Sort by budget descending so large-budget samples batch together and
    # never get clamped by a small-budget batch-mate. Small-budget samples
    # cluster in the final batch where min(budget) already reflects their
    # own limit — no starvation of healthy neighbors.
    if cont_indices:
        order = sorted(range(len(cont_indices)), key=lambda i: -cont_budgets[i])
        cont_indices = [cont_indices[i] for i in order]
        cont_prompts = [cont_prompts[i] for i in order]
        cont_budgets = [cont_budgets[i] for i in order]

    # -------- Phase 4: run continuations (grouped by budget floor to share max_tokens) --------
    if cont_indices:
        # Use a single shared max_tokens equal to min of budgets in each sub-batch
        for start in tqdm(range(0, len(cont_indices), batch_size),
                           desc="  methodb continue", leave=False):
            sub_idx = cont_indices[start:start + batch_size]
            sub_prompts = cont_prompts[start:start + batch_size]
            sub_budgets = cont_budgets[start:start + batch_size]
            mt = min(sub_budgets)
            out = completions_request_robust(sub_prompts, n=1, max_tokens=mt,
                                              base_url=base_url, model=model)
            for rel, idx in enumerate(sub_idx):
                s = samples[idx]
                og = out[rel] if rel < len(out) else []
                text = og[0]["text"] if og else ""
                tok = og[0].get("token_count", 0) if og else 0
                finish = og[0].get("finish_reason") if og else None
                if tok == 0:
                    tok = count_tokens(tokenizer, text)
                s["continuation"] = text
                s["continuation_tokens"] = tok
                s["continuation_finish"] = finish
                # Truncated mid-thinking: no close marker → grader would fail.
                # Accept the (rare) case where the model happened to commit
                # within the cap.
                if finish == "length" and not _has_close_marker(text):
                    s["mode"] = "continuation_length_fallback"
                    s["full_response"] = s["baseline_response"]
                    continue
                tentative_full = s["pre_close"] + s["injection"] + text
                # Require an actually-parseable answer in the post-think tail.
                # The model sometimes burns the continuation budget re-deriving
                # after </think> and cuts off mid-answer; accepting that as the
                # commit would silently drop a case that baseline already had
                # right. Fall back to baseline to preserve the last known
                # correct answer.
                post = strip_thinking_mk(model_key, tentative_full)
                if extract_answer(dataset_key, post) is None:
                    s["mode"] = "no_answer_fallback"
                    s["full_response"] = s["baseline_response"]
                    continue
                s["full_response"] = tentative_full

    # -------- Multi-turn extension (turns > 1, ablation='none' only) --------
    # No-op when turns == 1, so the single-turn path above is bit-for-bit
    # identical to the pre-refactor behavior.
    if turns > 1 and ablation == "none":
        close_marker = MODELS[model_key].think_close

        def _split_keep_close(text):
            """Split at first close marker; keep the marker on the 'after' side
            so concatenating the pieces back together preserves </think>."""
            if not text or not close_marker:
                return None, None
            idx = text.find(close_marker)
            if idx < 0:
                return None, None
            return text[:idx], text[idx:]

        # Seed per-sample multi-turn state from the turn-1 result.
        for s in samples:
            s["turns_committed"] = 1 if s["mode"] == "methodb" else 0
            s["findings_per_turn"] = []
            s["injections_per_turn"] = []
            s["new_thinking_per_turn"] = []
            s["continuation_tokens_per_turn"] = []
            s["findings_cap_per_turn"] = []
            # Track the committed answer after each turn so I2/I3 can anchor
            # on the model's latest answer instead of the baseline's.
            s["_answer_per_turn"] = []
            if s["mode"] != "methodb":
                continue
            nt1, tail1 = _split_keep_close(s["continuation"])
            if nt1 is None:
                # Phase 4 should have caught this; defensive guard.
                s["turns_committed"] = 0
                continue
            s["findings_per_turn"].append(list(s["findings"]))
            s["injections_per_turn"].append(s["injection"])
            s["new_thinking_per_turn"].append(nt1)
            s["continuation_tokens_per_turn"].append(s["continuation_tokens"])
            s["findings_cap_per_turn"].append(s["cap"])
            s["_tail_after_last_thinking"] = tail1  # "</think> … final answer"
            # Extract the answer the model committed to at end of turn 1.
            turn1_post = strip_thinking_mk(model_key, s["full_response"])
            s["_answer_per_turn"].append(
                extract_answer(dataset_key, turn1_post))

        EXTRACT_OUTPUT_RESERVE_N = 1024
        TARGET_MAX_N = MAX_MODEL_LEN - EXTRACT_OUTPUT_RESERVE_N

        for turn_n in range(2, turns + 1):
            variant = "again" if turn_n == 2 else "final"
            eligible = [i for i, s in enumerate(samples)
                        if s["turns_committed"] == turn_n - 1]
            if not eligible:
                print(f"  [methodb turn{turn_n}] no eligible samples; skipping")
                continue
            print(f"  [methodb turn{turn_n}] {len(eligible)} eligible "
                  f"samples; variant={variant}")

            # --- Phase A: build extract prompts (cumulative thinking + prior findings) ---
            extract_prompts = []
            extract_sample_indices = []
            extract_overflow = 0
            for idx in eligible:
                s = samples[idx]
                parts = [s["thinking_body"]]
                for inj, nt in zip(s["injections_per_turn"],
                                    s["new_thinking_per_turn"]):
                    parts.append(inj)
                    parts.append(nt)
                cumulative = "".join(parts)
                prior = []
                for flist in s["findings_per_turn"]:
                    prior.extend(flist)
                cum_tokens = count_tokens(tokenizer, cumulative)
                cap_now = adaptive_finding_cap(cum_tokens)
                uc = build_methodb_extract_prompt(
                    cumulative, dataset_key, s["problem"], cap_now,
                    prior_findings=prior)
                prompt_text = apply_chat_template_mk(model_key, tokenizer, uc)
                ptok = count_tokens(tokenizer, prompt_text)
                if ptok > TARGET_MAX_N:
                    extract_overflow += 1
                    continue
                s[f"_turn{turn_n}_cap"] = cap_now
                extract_prompts.append(prompt_text)
                extract_sample_indices.append(idx)
            if extract_overflow:
                print(f"  [methodb turn{turn_n}] {extract_overflow} extract-"
                      f"overflow — staying at turn {turn_n - 1}")
            if not extract_prompts:
                continue

            # --- Phase B: run extractor ---
            all_ext_out = []
            for start in tqdm(range(0, len(extract_prompts), batch_size),
                               desc=f"  methodb turn{turn_n} extract",
                               leave=False):
                sub = extract_prompts[start:start + batch_size]
                max_p = max(count_tokens(tokenizer, p) for p in sub)
                mt = MAX_MODEL_LEN - max_p - 100
                mt = max(256, min(METHODB_EXTRACT_MAX_TOKENS, mt))
                out = completions_request_robust(sub, n=1, max_tokens=mt,
                                                  base_url=base_url,
                                                  model=model)
                all_ext_out.extend(out)

            findings_new = {}
            for rel, idx in enumerate(extract_sample_indices):
                s = samples[idx]
                og = all_ext_out[rel] if rel < len(all_ext_out) else []
                raw = og[0]["text"] if og else ""
                post_think = strip_thinking_mk(model_key, raw).strip()
                findings_new[idx] = parse_methodb_findings(
                    post_think, s[f"_turn{turn_n}_cap"])

            # Quality-check retry: empty findings, META leak, or pure
            # duplicates of prior-turn findings → rerun extract. Applies
            # to all models; R1-distill gets a higher retry budget because
            # of its bullet-format miss rate.
            idx_to_prompt = dict(zip(extract_sample_indices, extract_prompts))
            MAX_EXTRACT_RETRIES_N = (
                10 if model_key.startswith("r1_distill") else 3)
            # Precompute prior findings per sample (flat) for dup check
            prior_findings_by_idx = {}
            for idx in extract_sample_indices:
                s = samples[idx]
                prior = []
                for fl in s["findings_per_turn"]:
                    prior.extend(fl)
                prior_findings_by_idx[idx] = prior

            for attempt in range(1, MAX_EXTRACT_RETRIES_N + 1):
                retry_indices = []
                retry_reasons: list[str] = []
                for idx in extract_sample_indices:
                    bad, reason = findings_have_issues(
                        findings_new.get(idx, []),
                        prior_findings=prior_findings_by_idx[idx])
                    if bad:
                        retry_indices.append(idx)
                        retry_reasons.append(reason or "")
                if not retry_indices:
                    break
                from collections import Counter as _C
                reason_summary = _C(retry_reasons)
                print(f"  [methodb turn{turn_n}] extract retry {attempt}/"
                      f"{MAX_EXTRACT_RETRIES_N}: "
                      f"{len(retry_indices)} samples need re-extract "
                      f"(reasons: {dict(reason_summary)})")
                retry_prompts = [idx_to_prompt[i] for i in retry_indices]
                retry_out = []
                for start in tqdm(
                        range(0, len(retry_prompts), batch_size),
                        desc=f"  methodb turn{turn_n} retry{attempt}",
                        leave=False):
                    sub = retry_prompts[start:start + batch_size]
                    max_p = max(count_tokens(tokenizer, p) for p in sub)
                    mt = MAX_MODEL_LEN - max_p - 100
                    mt = max(256, min(METHODB_EXTRACT_MAX_TOKENS, mt))
                    out = completions_request_robust(
                        sub, n=1, max_tokens=mt,
                        base_url=base_url, model=model)
                    retry_out.extend(out)
                for rel, idx in enumerate(retry_indices):
                    s = samples[idx]
                    og = retry_out[rel] if rel < len(retry_out) else []
                    raw = og[0]["text"] if og else ""
                    post_think = strip_thinking_mk(model_key, raw).strip()
                    parsed = parse_methodb_findings(
                        post_think, s[f"_turn{turn_n}_cap"])
                    # Only adopt the retry result if it's non-empty AND
                    # (issues resolve) — otherwise keep the prior attempt
                    # so a single bad retry doesn't overwrite decent findings.
                    if parsed:
                        new_bad, _ = findings_have_issues(
                            parsed, prior_findings=prior_findings_by_idx[idx])
                        if not new_bad:
                            findings_new[idx] = parsed

            # --- Phase C: build continuation prompts ---
            cont_indices = []
            cont_prompts = []
            no_findings = 0
            ctx_overflow_n = 0
            for idx in extract_sample_indices:
                s = samples[idx]
                new_findings = findings_new.get(idx, [])
                if not new_findings:
                    no_findings += 1
                    continue
                # "Key conclusions so far" in the injection is cumulative:
                # F1 ∪ F2 ∪ … ∪ F_turn_n so the model sees every verified
                # conclusion it's been building on, not just the latest batch.
                cumulative_findings: list = []
                for flist in s["findings_per_turn"]:
                    cumulative_findings.extend(flist)
                cumulative_findings.extend(new_findings)
                # The answer anchor tracks the model's current working answer:
                # turn 2's injection uses turn-1's committed answer, turn 3's
                # uses turn-2's. If extraction fails (None), fall back to the
                # baseline answer via _anchor.
                prev_answers = s.get("_answer_per_turn") or []
                latest_answer = prev_answers[-1] if prev_answers else None
                answer_for_turn = latest_answer or _anchor(s)
                injection = format_methodb_injection(
                    dataset_key, s["problem"], cumulative_findings,
                    baseline_answer=answer_for_turn, variant=variant)
                prefix_parts = [s["pre_close"]]
                for inj, nt in zip(s["injections_per_turn"],
                                    s["new_thinking_per_turn"]):
                    prefix_parts.append(inj)
                    prefix_parts.append(nt)
                prefix_parts.append(injection)
                prefix = "".join(prefix_parts)
                new_prompt = chat_prompts[s["pid"]] + prefix
                ptok = count_tokens(tokenizer, new_prompt)
                if ptok + METHODB_CONT_MAX_TOKENS + 100 > MAX_MODEL_LEN:
                    ctx_overflow_n += 1
                    continue
                s[f"_turn{turn_n}_injection"] = injection
                s[f"_turn{turn_n}_findings"] = new_findings
                cont_indices.append(idx)
                cont_prompts.append(new_prompt)
            print(f"  [methodb turn{turn_n}] continue={len(cont_indices)}, "
                  f"no_findings={no_findings}, "
                  f"ctx_overflow={ctx_overflow_n}")
            if not cont_indices:
                continue

            # --- Phase D: run continuations ---
            for start in tqdm(range(0, len(cont_indices), batch_size),
                               desc=f"  methodb turn{turn_n} continue",
                               leave=False):
                sub_idx = cont_indices[start:start + batch_size]
                sub_prompts = cont_prompts[start:start + batch_size]
                out = completions_request_robust(
                    sub_prompts, n=1,
                    max_tokens=METHODB_CONT_MAX_TOKENS,
                    base_url=base_url, model=model)
                for rel, idx in enumerate(sub_idx):
                    s = samples[idx]
                    og = out[rel] if rel < len(out) else []
                    text = og[0]["text"] if og else ""
                    tok = og[0].get("token_count", 0) if og else 0
                    finish = og[0].get("finish_reason") if og else None
                    if tok == 0:
                        tok = count_tokens(tokenizer, text)
                    nt, tail = _split_keep_close(text)
                    if nt is None:
                        # No close marker this turn → sample stays at turn_n-1
                        continue
                    # Also require a parseable answer in this turn's tail.
                    # If the model emitted </think> but ran out of budget
                    # before writing <Answer>…</Answer>, committing this turn
                    # would overwrite the previous turn's valid answer with
                    # None. Stay at turn_n-1 instead.
                    post_this_turn_check = strip_thinking_mk(model_key, tail)
                    turn_answer = extract_answer(
                        dataset_key, post_this_turn_check)
                    if turn_answer is None:
                        continue
                    s["findings_per_turn"].append(s[f"_turn{turn_n}_findings"])
                    s["injections_per_turn"].append(s[f"_turn{turn_n}_injection"])
                    s["new_thinking_per_turn"].append(nt)
                    s["continuation_tokens_per_turn"].append(tok)
                    s["findings_cap_per_turn"].append(s[f"_turn{turn_n}_cap"])
                    s["_tail_after_last_thinking"] = tail
                    s["turns_committed"] = turn_n
                    s[f"_turn{turn_n}_continuation_finish"] = finish
                    # We already extracted turn_answer above (it's non-None,
                    # otherwise we would have skipped this commit).
                    s["_answer_per_turn"].append(turn_answer)

        # Reconstruct full_response for samples that advanced past turn 1
        for s in samples:
            if s.get("turns_committed", 0) < 2:
                continue
            parts = [s["pre_close"]]
            for inj, nt in zip(s["injections_per_turn"],
                                s["new_thinking_per_turn"]):
                parts.append(inj)
                parts.append(nt)
            parts.append(s["_tail_after_last_thinking"])
            s["full_response"] = "".join(parts)
            flat = []
            for flist in s["findings_per_turn"]:
                flat.extend(flist)
            s["findings"] = flat
            s["continuation_tokens"] = sum(s["continuation_tokens_per_turn"])

        turn_counts = {}
        for s in samples:
            tc = s.get("turns_committed", 0)
            turn_counts[tc] = turn_counts.get(tc, 0) + 1
        print(f"  [methodb] turns_committed histogram: {turn_counts}")

    # -------- Phase 5: write records --------
    written = 0
    for s in samples:
        post_think = strip_thinking_mk(model_key, s["full_response"])
        predicted = extract_answer(dataset_key, post_think)
        correct = grade_answer(dataset_key, predicted, s["gt"])
        total_tokens = s["baseline_tokens"] + s["continuation_tokens"]
        turns_committed = s.get("turns_committed")
        if turns_committed is None:
            turns_committed = (1 if s["mode"] in ("methodb", "verify_only")
                               else 0)
        record = {
            "problem_id": s["pid"],
            "sample_idx": s["sample_idx"],
            "group": "methodb",
            "ablation": ablation,
            "gt_answer": s["gt"],
            "predicted_answer": predicted,
            "correct": correct,
            "completion_tokens": total_tokens,
            "baseline_tokens": s["baseline_tokens"],
            "continuation_tokens": s["continuation_tokens"],
            "thinking_tokens": s["thinking_tokens"],
            "findings_cap": s["cap"],
            "num_findings": len(s["findings"]),
            "findings": s["findings"],
            "mode": s["mode"],
            "continuation_finish": s["continuation_finish"],
            "baseline_correct": s["baseline_correct"],
            "methodb_correct": correct,
            "num_steps": 1 + turns_committed,
            "response": s["full_response"],
        }
        if turns > 1:
            record["turns_requested"] = turns
            record["turns_committed"] = turns_committed
            record["findings_per_turn"] = s.get("findings_per_turn", [])
            record["continuation_tokens_per_turn"] = s.get(
                "continuation_tokens_per_turn", [])
            record["findings_cap_per_turn"] = s.get(
                "findings_cap_per_turn", [])
        out_file.write(json.dumps(record, ensure_ascii=False) + "\n")
        written += 1
    out_file.flush()
    write_pretty_snapshot(out_file.name)

    # Summary
    buckets = {"CC": 0, "CW": 0, "WC": 0, "WW": 0}
    for s in samples:
        post_think = strip_thinking_mk(model_key, s["full_response"])
        pred = extract_answer(dataset_key, post_think)
        mc = grade_answer(dataset_key, pred, s["gt"])
        b = s["baseline_correct"]
        buckets["CC" if (b and mc) else "CW" if (b and not mc)
                else "WC" if (not b and mc) else "WW"] += 1
    print(f"  [methodb] transitions: C→C={buckets['CC']}  "
          f"C→W={buckets['CW']}  W→C={buckets['WC']}  W→W={buckets['WW']}")
    mode_counts = {}
    for s in samples:
        mode_counts[s["mode"]] = mode_counts.get(s["mode"], 0) + 1
    print(f"  [methodb] modes: {mode_counts}")

    return written


# ---------------------------------------------------------------------------
# Dataset loading & resume
# ---------------------------------------------------------------------------

def load_dataset(path: str) -> list[dict]:
    data = []
    with open(path) as f:
        for line in f:
            data.append(json.loads(line))
    return data


def load_done_keys(path: Path) -> set[tuple[str, int]]:
    done = set()
    if path.exists():
        with open(path) as f:
            for line in f:
                try:
                    rec = json.loads(line)
                    done.add((rec["problem_id"], rec["sample_idx"]))
                except (json.JSONDecodeError, KeyError):
                    continue
    return done


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--group", required=True, choices=["baseline", "finding", "methodb"])
    parser.add_argument("--baseline-in", default=None,
                        help="(methodb only) Path to baseline raw_baseline.jsonl to reuse")
    parser.add_argument("--model-key", required=True, choices=list(MODELS.keys()),
                        help="Which model config to use (from prompt.py)")
    parser.add_argument("--dataset-key", required=True, choices=list(DATASETS.keys()),
                        help="Which dataset config to use (from prompt.py)")
    parser.add_argument("--out", required=True, help="Output JSONL path")
    parser.add_argument("--data", default=None,
                        help="Override dataset path (default: DATASETS[dataset_key].data_path)")
    parser.add_argument("--model", default=None,
                        help="Override model path served by vLLM "
                             "(default: MODELS[model_key].local_path)")
    parser.add_argument("--port", type=int, default=VLLM_PORT)
    parser.add_argument("--num-samples", type=int, default=NUM_SAMPLES)
    parser.add_argument("--finding-steps", type=int, default=FINDING_STEPS)
    parser.add_argument("--max-steps", type=int, default=MAX_STEPS)
    parser.add_argument("--batch-size", type=int, default=256,
                        help="Max prompts per vLLM request (cross-problem batching)")
    parser.add_argument("--debug-log", default=None,
                        help="(finding only) Path to write full per-round trace "
                             "for a small sample of problems")
    parser.add_argument("--ablation", default="none",
                        choices=["none", "verify_only"],
                        help="(methodb only) 'verify_only' skips the extractor "
                             "and replaces the full injection with a naked "
                             "re-think prompt. 'none' runs standard "
                             "insightreplay.")
    parser.add_argument("--turns", type=int, default=1, choices=[1, 3],
                        help="(methodb, ablation=none only) Number of "
                             "extract+replay turns. 1 = original behavior. "
                             "3 = after the first insightreplay round, extract "
                             "new findings from the full accumulated thinking "
                             "and re-inject with 'one more time' / 'for the "
                             "last time' phrasing, for two additional turns.")
    parser.add_argument("--insightreplay-in", default=None,
                        help="(methodb, turns>1 only) Path to a prior 1-turn "
                             "insightreplay jsonl. When provided, samples "
                             "whose turn-1 state can be reconstructed from "
                             "that file skip Phase 2-4 and go straight to "
                             "turns 2+3, saving roughly a third of the "
                             "inference cost. Samples whose reconstruction "
                             "fails (or that weren't present) are re-run "
                             "from scratch.")
    args = parser.parse_args()

    model_cfg = MODELS[args.model_key]
    dataset_cfg = DATASETS[args.dataset_key]
    data_path = args.data or dataset_cfg.data_path
    model_path = args.model or model_cfg.local_path

    base_url = f"http://127.0.0.1:{args.port}"

    dataset = load_dataset(data_path)
    print(f"Loaded {len(dataset)} problems from {data_path}")
    print(f"Model:   {args.model_key}  ({model_path})")
    print(f"Dataset: {args.dataset_key}  (answer_kind={dataset_cfg.answer_kind})")

    out_path = Path(args.out)
    os.makedirs(out_path.parent, exist_ok=True)

    # Resume
    done_keys = load_done_keys(out_path)
    expected = len(dataset) * args.num_samples
    if len(done_keys) >= expected:
        print(f"Output already has {len(done_keys)} records (expected {expected}). Skipping.")
        return
    if done_keys:
        print(f"Resuming: {len(done_keys)}/{expected} already done")

    # Wait for server
    wait_for_server(base_url)

    print(f"Loading tokenizer from {model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    with open(out_path, "a") as fout:
        if args.group == "baseline":
            written = run_baseline(tokenizer, dataset, args.num_samples,
                                   fout, done_keys, base_url, model_path,
                                   batch_size=args.batch_size,
                                   model_key=args.model_key,
                                   dataset_key=args.dataset_key)
        elif args.group == "methodb":
            if not args.baseline_in:
                raise SystemExit("--baseline-in is required for --group methodb")
            if args.turns != 1 and args.ablation != "none":
                raise SystemExit(
                    "--turns > 1 is only supported with --ablation none "
                    "(multi-turn is for full insightreplay, not ablations)")
            if args.insightreplay_in and args.turns == 1:
                raise SystemExit(
                    "--insightreplay-in only makes sense with --turns > 1 "
                    "(it's used to skip turn-1 re-computation)")
            fr_in = (Path(args.insightreplay_in)
                     if args.insightreplay_in else None)
            if fr_in is not None and not fr_in.is_file():
                raise SystemExit(
                    f"--insightreplay-in not found: {fr_in}")
            written = run_methodb_finding(
                tokenizer, dataset, args.num_samples,
                fout, done_keys, base_url, model_path,
                baseline_in=Path(args.baseline_in),
                batch_size=args.batch_size,
                debug_log_path=args.debug_log,
                model_key=args.model_key,
                dataset_key=args.dataset_key,
                ablation=args.ablation,
                turns=args.turns,
                insightreplay_in=fr_in)
        else:
            written = run_finding(tokenizer, dataset, args.num_samples,
                                  args.finding_steps, args.max_steps,
                                  fout, done_keys, base_url, model_path,
                                  batch_size=args.batch_size,
                                  debug_log_path=args.debug_log,
                                  model_key=args.model_key,
                                  dataset_key=args.dataset_key)

    # Summary
    total_done = len(done_keys) + written
    all_recs = []
    with open(out_path) as f:
        for line in f:
            all_recs.append(json.loads(line))
    # For livecodebench, correct is False until external grader runs; skip pct
    n_correct = sum(1 for r in all_recs if r.get("correct"))
    pct = f"{n_correct/total_done*100:.1f}%" if total_done else "n/a"
    print(f"Done [{args.group}]: {total_done} total responses, "
          f"{n_correct}/{total_done} correct ({pct})")
    if dataset_cfg.answer_kind == "code":
        print("  NOTE: code answers need external grading "
              "(correct=False here means 'not auto-graded').")
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    main()
