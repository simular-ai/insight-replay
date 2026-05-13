#!/usr/bin/env python3
"""Audit baseline + insightreplay + verify_only outputs.

For every (model, dataset) combination:
  1. File existence + record count == num_problems * num_samples
  2. No duplicate (problem_id, sample_idx) rows
  3. All required fields present; types sane
  4. completion_tokens > 0
  5. AIME/GPQA: re-grade (predicted_answer vs gt_answer) and confirm the
     stored `correct` field matches
  6. LCB: `correct` is a bool (LCB relies on external grader)
  7. insightreplay / verify_only only:
       - baseline_tokens + continuation_tokens == completion_tokens
       - methodb_correct == correct
       - mode is in the known set
       - num_findings == len(findings) when both present
       - baseline_tokens matches the baseline jsonl's completion_tokens for
         the same (problem_id, sample_idx)
  8. Summary CSVs: per-line n/nc/pct/avg_tok/max_tok/avg_steps recomputed
     from the jsonl should match
  9. Log files (run_*.log, .*_all.log, vllm.log): scan for Traceback /
     Error markers

Prints a compact per-combo table and a summary of all anomalies.
Non-zero exit if any check fails; zero if everything's clean.

Usage:
  python3 scripts/audit_results.py
  python3 scripts/audit_results.py --models qwen35_9b gemma4_31b_it
  python3 scripts/audit_results.py --datasets aime --verbose
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUTPUTS = ROOT / "outputs"
sys.path.insert(0, str(ROOT / "scripts"))

from prompt import DATASETS, grade_answer  # noqa: E402

MODELS = [
    "qwen35_9b",
    "gemma4_e4b_it",
    "r1_distill_qwen_7b",
    "qwen35_35b_a3b",
    "gemma4_31b_it",
    "r1_distill_qwen_32b",
]
DATASET_NAMES = ["aime", "gpqa", "livecodebench", "hmmt"]
METHODS = {
    "baseline":      ("__unlimited",     "raw_baseline.jsonl"),
    "insightreplay": ("__insightreplay", "raw_insightreplay.jsonl"),
    "verify_only":   ("__verify_only",   "raw_verify_only.jsonl"),
}
KNOWN_MODES = {
    "baseline_fallback_no_close",
    "extract_overflow_fallback",
    "ctx_overflow_fallback",
    "no_findings_fallback",
    "continuation_length_fallback",
    "methodb",
    "verify_only",
    "pending",
}
BASE_REQUIRED = ["problem_id", "sample_idx", "correct",
                 "completion_tokens", "predicted_answer",
                 "gt_answer", "response"]
FR_EXTRA = ["baseline_tokens", "continuation_tokens", "thinking_tokens",
            "findings", "mode", "methodb_correct", "baseline_correct"]


def dataset_num_problems(dataset_key: str) -> int:
    path = Path(DATASETS[dataset_key].data_path)
    if not path.is_file():
        return 0
    with path.open() as f:
        return sum(1 for line in f if line.strip())


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open() as f:
        for ln, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as e:
                rows.append({"__parse_error__": str(e), "__line__": ln})
    return rows


def check_records(rows: list[dict], dataset_key: str, method: str,
                  baseline_by_key: dict | None = None) -> list[str]:
    """Return a list of human-readable issues; empty if clean."""
    issues: list[str] = []

    # Parse errors
    bad = [r for r in rows if r.get("__parse_error__")]
    if bad:
        issues.append(f"{len(bad)} lines failed to JSON-parse")

    good = [r for r in rows if not r.get("__parse_error__")]

    # Duplicate keys
    keys = [(r.get("problem_id"), r.get("sample_idx")) for r in good]
    key_counts = Counter(keys)
    dupes = [k for k, c in key_counts.items() if c > 1]
    if dupes:
        issues.append(f"{len(dupes)} duplicate (pid, sample) pairs; "
                      f"e.g. {dupes[:3]}")

    # Required fields
    required = list(BASE_REQUIRED)
    if method != "baseline":
        required += FR_EXTRA
    missing_counts: Counter = Counter()
    for r in good:
        for f in required:
            if f not in r:
                missing_counts[f] += 1
    for f, c in missing_counts.items():
        issues.append(f"{c} records missing field '{f}'")

    # Per-record checks
    grade_mismatch = 0
    correct_not_bool = 0
    tok_zero = 0
    tok_sum_mismatch = 0
    mode_unknown = 0
    mb_correct_mismatch = 0
    nf_mismatch = 0
    baseline_tok_mismatch = 0
    baseline_key_missing = 0

    answer_kind = DATASETS[dataset_key].answer_kind

    for r in good:
        # correct field type
        if not isinstance(r.get("correct"), bool):
            correct_not_bool += 1
        # completion_tokens > 0 (baseline) or >= 0 for fallback FR modes
        if r.get("completion_tokens", 0) <= 0:
            if method == "baseline" or r.get("mode") not in (
                    "baseline_fallback_no_close",):
                tok_zero += 1
        # grading re-check for scalar-answer datasets
        if answer_kind in ("integer", "letter", "latex"):
            recomputed = grade_answer(dataset_key,
                                       r.get("predicted_answer"),
                                       r.get("gt_answer"))
            if bool(recomputed) != bool(r.get("correct")):
                grade_mismatch += 1

        # insightreplay/verify_only extra checks
        if method != "baseline":
            bt = r.get("baseline_tokens", 0) or 0
            ct = r.get("continuation_tokens", 0) or 0
            tot = r.get("completion_tokens", 0) or 0
            if bt + ct != tot:
                tok_sum_mismatch += 1
            mode = r.get("mode")
            if mode not in KNOWN_MODES:
                mode_unknown += 1
            # LCB: `correct` is populated post-hoc by grade_livecodebench.py
            # while `methodb_correct` is frozen at record-write time. So for
            # any LCB code sample the grader flips from False → True, these
            # will disagree by design. Skip the check for LCB.
            if (dataset_key != "livecodebench"
                    and r.get("methodb_correct") is not None
                    and bool(r["methodb_correct"]) != bool(r.get("correct"))):
                mb_correct_mismatch += 1
            findings = r.get("findings")
            nf = r.get("num_findings")
            if findings is not None and nf is not None and nf != len(findings):
                nf_mismatch += 1
            # Cross-check baseline_tokens against the baseline jsonl
            if baseline_by_key is not None:
                key = (r.get("problem_id"), r.get("sample_idx"))
                base = baseline_by_key.get(key)
                if base is None:
                    baseline_key_missing += 1
                elif (base.get("completion_tokens", 0)
                      != r.get("baseline_tokens", 0)):
                    baseline_tok_mismatch += 1

    if grade_mismatch:
        issues.append(f"{grade_mismatch} records where stored `correct` "
                      f"disagrees with re-grade(predicted,gt)")
    if correct_not_bool:
        issues.append(f"{correct_not_bool} records where `correct` is not a bool")
    if tok_zero:
        issues.append(f"{tok_zero} records with completion_tokens <= 0")
    if method != "baseline":
        if tok_sum_mismatch:
            issues.append(f"{tok_sum_mismatch} records where baseline_tokens "
                          f"+ continuation_tokens != completion_tokens")
        if mode_unknown:
            issues.append(f"{mode_unknown} records with unknown `mode`")
        if mb_correct_mismatch:
            issues.append(f"{mb_correct_mismatch} records where "
                          f"`methodb_correct` != `correct`")
        if nf_mismatch:
            issues.append(f"{nf_mismatch} records where num_findings "
                          f"!= len(findings)")
        if baseline_key_missing:
            issues.append(f"{baseline_key_missing} (pid,sample) missing "
                          f"from baseline jsonl")
        if baseline_tok_mismatch:
            issues.append(f"{baseline_tok_mismatch} records where "
                          f"baseline_tokens != baseline jsonl's "
                          f"completion_tokens")

    return issues


def audit_combo(model: str, dataset_key: str, method: str,
                expected_n: int, baseline_by_key: dict | None = None,
                verbose: bool = False) -> tuple[bool, dict]:
    dir_suffix, fname = METHODS[method]
    path = OUTPUTS / f"{model}__{dataset_key}{dir_suffix}" / fname
    result = {"path": str(path.relative_to(ROOT)),
              "exists": path.is_file(),
              "n": 0, "n_correct": 0, "issues": [],
              "avg_tok": 0.0, "max_tok": 0, "avs": 0.0}
    if not path.is_file():
        result["issues"].append("file missing")
        return False, result
    rows = load_jsonl(path)
    good = [r for r in rows if not r.get("__parse_error__")]
    result["n"] = len(good)
    result["n_correct"] = sum(1 for r in good if r.get("correct"))
    toks = [r.get("completion_tokens", 0) for r in good]
    steps = [r.get("num_steps", 1) for r in good]
    result["avg_tok"] = sum(toks) / max(1, len(toks))
    result["max_tok"] = max(toks) if toks else 0
    result["avs"] = sum(steps) / max(1, len(steps))
    if expected_n and len(good) != expected_n:
        result["issues"].append(
            f"count={len(good)} expected={expected_n}")
    result["issues"].extend(
        check_records(rows, dataset_key, method, baseline_by_key))
    return len(result["issues"]) == 0, result


def audit_summary_csv(method: str, all_results: dict,
                      verbose: bool = False) -> list[str]:
    """Compare summary CSVs against recomputed stats."""
    issues: list[str] = []
    fname = f"summary_{method}.csv"
    csv_path = OUTPUTS / fname
    if not csv_path.is_file():
        # baseline pipeline doesn't emit a summary CSV by design; only
        # insightreplay / verify_only do.
        if method in ("insightreplay", "verify_only"):
            issues.append(f"[{fname}] file missing")
        return issues

    # Summary CSVs are append-only; last entry for a (model,dataset) wins
    last_by_key: dict[tuple, dict] = {}
    with csv_path.open() as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) < 10:
                issues.append(f"[{fname}] malformed row: {row}")
                continue
            model_, ds_, grp_, n, nc, pct, avg_tok, max_tok, avs, out_dir = (
                row[0], row[1], row[2], row[3], row[4],
                row[5], row[6], row[7], row[8], row[9])
            last_by_key[(model_, ds_)] = {
                "n": int(n), "nc": int(nc), "pct": float(pct),
                "avg_tok": float(avg_tok), "max_tok": float(max_tok),
                "avs": float(avs), "out_dir": out_dir}

    for (model, ds), entry in last_by_key.items():
        combo_key = (model, ds, method)
        rec = all_results.get(combo_key)
        if rec is None or not rec["exists"]:
            issues.append(f"[{fname}] {model}x{ds}: listed but jsonl missing")
            continue
        # Reuse stats computed during audit_combo (avoid re-parsing 2GB jsonls)
        n_re = rec["n"]
        nc_re = rec["n_correct"]
        avg_tok_re = rec["avg_tok"]
        max_tok_re = rec["max_tok"]
        avs_re = rec["avs"]

        local_issues = []
        if entry["n"] != n_re:
            local_issues.append(f"n {entry['n']}!={n_re}")
        if entry["nc"] != nc_re:
            local_issues.append(f"nc {entry['nc']}!={nc_re}")
        if abs(entry["pct"] - (nc_re / n_re if n_re else 0)) > 1e-3:
            local_issues.append(
                f"pct {entry['pct']}!={nc_re/n_re:.3f}")
        if abs(entry["avg_tok"] - avg_tok_re) > 1.0:
            local_issues.append(
                f"avg_tok {entry['avg_tok']}!={avg_tok_re:.1f}")
        if entry["max_tok"] != max_tok_re:
            local_issues.append(
                f"max_tok {entry['max_tok']}!={max_tok_re}")
        if abs(entry["avs"] - avs_re) > 0.01:
            local_issues.append(f"avs {entry['avs']}!={avs_re:.2f}")
        if local_issues:
            issues.append(f"[{fname}] {model}x{ds}: " + ", ".join(local_issues))

    return issues


LOG_PATTERNS = [
    re.compile(r"Traceback \(most recent call last\)"),
    re.compile(r"\bERROR\b"),
    re.compile(r"CUDA out of memory", re.IGNORECASE),
    re.compile(r"Killed$"),
]


def scan_log(path: Path, max_hits: int = 3) -> list[str]:
    """Return short summaries of concerning lines; empty if clean."""
    if not path.is_file():
        return []
    hits: list[str] = []
    try:
        with path.open(errors="replace") as f:
            for ln, line in enumerate(f, 1):
                for pat in LOG_PATTERNS:
                    if pat.search(line):
                        snippet = line.strip()
                        if len(snippet) > 160:
                            snippet = snippet[:160] + "…"
                        hits.append(f"L{ln}: {snippet}")
                        break
                if len(hits) >= max_hits:
                    break
    except Exception as e:
        return [f"read error: {e}"]
    return hits


def audit_logs(verbose: bool = False) -> list[str]:
    issues: list[str] = []
    log_globs = [
        OUTPUTS.glob("*/run_baseline.log"),
        OUTPUTS.glob("*/run_insightreplay.log"),
        OUTPUTS.glob("*/run_verify_only.log"),
        OUTPUTS.glob("*/run_methodb.log"),
        OUTPUTS.glob(".*_all.log"),
    ]
    for g in log_globs:
        for p in g:
            hits = scan_log(p, max_hits=2)
            if hits:
                rel = p.relative_to(ROOT)
                issues.append(f"[{rel}] " + " | ".join(hits))
    return issues


def fmt_row(label, n, expected, correct, path, issues):
    status = "OK" if not issues else "FAIL"
    acc = f"{correct}/{n}" if n else "—"
    pct = f"{correct/n:.3f}" if n else "—"
    return (f"  {status:4s}  {label:14s}  n={n:>4d}/{expected:<4d}  "
            f"acc={acc:<9s} ({pct})   {path}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--models", nargs="*", default=MODELS)
    ap.add_argument("--datasets", nargs="*", default=DATASET_NAMES)
    ap.add_argument("--methods", nargs="*",
                    default=list(METHODS.keys()),
                    choices=list(METHODS.keys()))
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--skip-logs", action="store_true",
                    help="Skip scanning log files (fast mode)")
    args = ap.parse_args()

    # Precompute expected counts per dataset
    num_problems = {ds: dataset_num_problems(ds) for ds in args.datasets}
    num_samples = int(16)  # k=16 used throughout
    print(f"Dataset sizes: {num_problems}  (num_samples={num_samples})\n")

    all_results: dict[tuple, dict] = {}
    any_issues = False

    for method in args.methods:
        print(f"=== {method} ===")
        # Preload baseline jsonl for insightreplay/verify_only cross-check
        baseline_cache: dict = {}
        for model in args.models:
            for ds in args.datasets:
                expected = num_problems.get(ds, 0) * num_samples
                b_key = None
                if method != "baseline":
                    if (model, ds) not in baseline_cache:
                        bpath = OUTPUTS / f"{model}__{ds}__unlimited" / "raw_baseline.jsonl"
                        by_key = {}
                        if bpath.is_file():
                            for r in load_jsonl(bpath):
                                if r.get("__parse_error__"):
                                    continue
                                by_key[(r.get("problem_id"),
                                        r.get("sample_idx"))] = r
                        baseline_cache[(model, ds)] = by_key
                    b_key = baseline_cache[(model, ds)] or None

                ok, res = audit_combo(model, ds, method, expected,
                                       baseline_by_key=b_key,
                                       verbose=args.verbose)
                all_results[(model, ds, method)] = res
                label = f"{model}x{ds}"
                print(fmt_row(label, res["n"], expected,
                              res["n_correct"], res["path"],
                              res["issues"]))
                if res["issues"]:
                    any_issues = True
                    for iss in res["issues"]:
                        print(f"        - {iss}")
        print()

    print("=== summary CSVs ===")
    for method in args.methods:
        issues = audit_summary_csv(method, all_results, verbose=args.verbose)
        fname = f"summary_{method}.csv"
        if not issues:
            print(f"  OK    {fname}")
        else:
            any_issues = True
            print(f"  FAIL  {fname}")
            for iss in issues:
                print(f"        - {iss}")
    print()

    if not args.skip_logs:
        print("=== log files ===")
        log_issues = audit_logs(verbose=args.verbose)
        if not log_issues:
            print("  OK    no errors / tracebacks found")
        else:
            any_issues = True
            for iss in log_issues:
                print(f"  FAIL  {iss}")
        print()

    if any_issues:
        print("RESULT: ISSUES FOUND — see above")
        sys.exit(1)
    print("RESULT: all checks passed")


if __name__ == "__main__":
    main()
