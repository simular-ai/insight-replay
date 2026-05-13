#!/usr/bin/env python3
"""Grade LiveCodeBench predictions stored in our raw_*.jsonl format.

Wraps the Docker-sandboxed grader in ``sample_eval_livecodebench.py``.
For each record with ``predicted_answer = <code>`` (or None), we run the
code against the test cases of the matching problem and overwrite the
record's ``correct`` field.

Usage:
    python grade_livecodebench.py \\
        --raw outputs/<run>/raw_<suffix>.jsonl \\
        --dataset data/livecodebench_v5.jsonl \\
        --workers 8 --timeout 20
"""

import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
from sample_eval_livecodebench import (  # noqa: E402
    load_dataset as _load_dataset,
    parse_verification_info,
    run_candidate,
)

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", required=True, help="Path to raw_*.jsonl to grade in-place")
    ap.add_argument("--dataset", required=True,
                    help="Path to livecodebench_v5.jsonl (source of tests)")
    ap.add_argument("--docker-image", default="python:3.10-slim")
    ap.add_argument("--timeout", type=int, default=20)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--debug", action="store_true")
    ap.add_argument("--dry-run", action="store_true",
                    help="Don't overwrite the raw file; just print stats.")
    args = ap.parse_args()

    raw_path = Path(args.raw)
    rows = []
    with raw_path.open() as f:
        for line in f:
            rows.append(json.loads(line))
    print(f"loaded {len(rows)} rows from {raw_path}")

    dataset = _load_dataset(args.dataset)
    print(f"dataset has {len(dataset)} problems")

    # Build tasks: (row_idx, code, tests, func_name)
    tasks = []
    skipped_no_code = 0
    skipped_no_tests = 0
    for i, r in enumerate(rows):
        code = r.get("predicted_answer")
        if not code:
            skipped_no_code += 1
            continue
        pid = str(r.get("problem_id"))
        drow = dataset.get(pid)
        if drow is None:
            skipped_no_tests += 1
            continue
        tests, func_name = parse_verification_info(drow)
        if not tests:
            skipped_no_tests += 1
            continue
        tasks.append((i, code, tests, func_name))

    print(f"to grade: {len(tasks)}  (skipped: no_code={skipped_no_code}, "
          f"no_tests={skipped_no_tests})")

    # Run in parallel
    results = {}  # row_idx -> (ok, reason)
    if args.workers <= 1:
        it = tasks if tqdm is None else tqdm(tasks, desc="grading")
        for idx, code, tests, func_name in it:
            ok, reason, _, _ = run_candidate(
                code, func_name, tests, args.docker_image, args.timeout, args.debug)
            results[idx] = (ok, reason)
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            futs = {ex.submit(run_candidate, code, func_name, tests,
                              args.docker_image, args.timeout, args.debug): idx
                    for idx, code, tests, func_name in tasks}
            it = as_completed(futs) if tqdm is None else \
                 tqdm(as_completed(futs), total=len(futs), desc="grading")
            for fut in it:
                idx = futs[fut]
                ok, reason, _, _ = fut.result()
                results[idx] = (ok, reason)

    # Update rows
    for idx, (ok, reason) in results.items():
        rows[idx]["correct"] = bool(ok)
        rows[idx]["grader_reason"] = reason
    # Rows not in results (no code / no tests) get correct=False
    for i, r in enumerate(rows):
        if i not in results:
            r["correct"] = False
            if "grader_reason" not in r:
                r["grader_reason"] = "not_graded"

    correct = sum(1 for r in rows if r.get("correct"))
    print(f"graded: correct={correct}/{len(rows)} = {correct/len(rows):.3%}")

    if args.dry_run:
        print("(dry-run) raw file NOT modified")
        return

    with raw_path.open("w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"overwrote {raw_path}")


if __name__ == "__main__":
    main()
