#!/usr/bin/env python3
"""Retroactively apply the 'no-answer fallback' policy to existing jsonls.

For each insightreplay / verify_only / insightreplay_3turns record whose
`predicted_answer` is None AND whose `mode` suggests the method was
committed (methodb / verify_only / or any mode ending != baseline fallback),
replace `predicted_answer` and `correct` with the matching baseline record's
values. This mirrors the runtime `no_answer_fallback` added to
run_sampling.py — any turn that committed </think> but failed to emit a
parseable answer would have silently zeroed out a sample the baseline had
right.

Skips livecodebench (handled by grade_livecodebench.py — its 'not_graded'
samples already reflect no-code-block and are a separate concern).

Usage:
  python3 scripts/repair_no_answer.py           # repair all eligible files
  python3 scripts/repair_no_answer.py --dry-run # show what would change
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUTPUTS = ROOT / "outputs"
sys.path.insert(0, str(ROOT / "scripts"))

MODELS = [
    "qwen35_9b", "gemma4_e4b_it", "r1_distill_qwen_7b",
    "qwen35_35b_a3b", "gemma4_31b_it", "r1_distill_qwen_32b",
]
DATASETS = ["aime", "gpqa", "hmmt"]  # skip livecodebench (external grader)
METHODS = {
    "insightreplay":        "raw_insightreplay.jsonl",
    "verify_only":          "raw_verify_only.jsonl",
    "insightreplay_3turns": "raw_insightreplay_3turns.jsonl",
}

# `mode` values that indicate a "committed" (non-fallback) attempt
COMMITTED_MODES = {"methodb", "verify_only"}


def load_baseline_index(model: str, dataset: str) -> dict:
    path = OUTPUTS / f"{model}__{dataset}__unlimited" / "raw_baseline.jsonl"
    out = {}
    if not path.is_file():
        return out
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            out[(r.get("problem_id"), r.get("sample_idx"))] = r
    return out


def repair_file(path: Path, baseline_idx: dict, dry_run: bool) -> dict:
    stats = {"n": 0, "none_to_fix": 0, "repaired": 0,
             "baseline_also_none": 0, "skipped_fallback": 0,
             "new_correct_from_fix": 0}
    rows = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))

    for r in rows:
        stats["n"] += 1
        if r.get("predicted_answer") is not None:
            continue
        # Only repair committed attempts — samples already in a
        # *_fallback mode either used baseline anyway or are a separate
        # issue (e.g. baseline_fallback_no_close).
        mode = r.get("mode", "")
        if mode not in COMMITTED_MODES:
            stats["skipped_fallback"] += 1
            continue
        stats["none_to_fix"] += 1
        key = (r.get("problem_id"), r.get("sample_idx"))
        base = baseline_idx.get(key)
        if base is None:
            continue
        baseline_pred = base.get("predicted_answer")
        if baseline_pred is None:
            stats["baseline_also_none"] += 1
            continue
        # Perform the fallback: overwrite predicted_answer / correct / mode
        old_correct = bool(r.get("correct", False))
        r["predicted_answer"] = baseline_pred
        r["correct"] = bool(base.get("correct", False))
        # Preserve methodb_correct if present
        if "methodb_correct" in r:
            r["methodb_correct"] = bool(base.get("correct", False))
        # Record what we did by renaming the mode; also keep a note.
        r["mode"] = f"{mode}_no_answer_fallback"
        r["no_answer_repair"] = True
        stats["repaired"] += 1
        if r["correct"] and not old_correct:
            stats["new_correct_from_fix"] += 1

    if not dry_run and stats["repaired"] > 0:
        tmp = path.with_suffix(path.suffix + ".tmp")
        with tmp.open("w") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        tmp.replace(path)
    return stats


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    grand = {"n": 0, "none_to_fix": 0, "repaired": 0,
             "baseline_also_none": 0, "skipped_fallback": 0,
             "new_correct_from_fix": 0}
    print(f"{'file':<70s}  {'n':>5s} {'None':>5s} {'fixed':>5s} {'→right':>7s}")
    print("-" * 105)
    for model in MODELS:
        for ds in DATASETS:
            baseline_idx = load_baseline_index(model, ds)
            if not baseline_idx:
                continue
            for suf, fname in METHODS.items():
                path = OUTPUTS / f"{model}__{ds}__{suf}" / fname
                if not path.is_file():
                    continue
                st = repair_file(path, baseline_idx, args.dry_run)
                rel = path.relative_to(ROOT)
                print(f"{str(rel):<70s}  {st['n']:>5d} {st['none_to_fix']:>5d} "
                      f"{st['repaired']:>5d} {st['new_correct_from_fix']:>7d}")
                for k in grand:
                    grand[k] += st[k]
    print("-" * 105)
    print(f"{'TOTAL':<70s}  {grand['n']:>5d} {grand['none_to_fix']:>5d} "
          f"{grand['repaired']:>5d} {grand['new_correct_from_fix']:>7d}")
    if args.dry_run:
        print("\n(dry-run — nothing written)")


if __name__ == "__main__":
    main()
