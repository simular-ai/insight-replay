#!/usr/bin/env python3
"""Rebuild outputs/summary_{baseline,insightreplay,verify_only}.csv from the
current raw_*.jsonl files.

The CSVs written live by run_all.sh / run_ablation.sh are append-only — if a
(model,dataset) cell is re-run or its jsonl is replaced (e.g. after moving in
a fresh methodb_fix snapshot), the CSV can end up with stale or missing rows.
This tool reads every jsonl under outputs/ and emits one canonical row per
(model,dataset) to each summary file (overwriting in place).

Usage:
  python3 scripts/rebuild_summaries.py              # rewrite all 3 summaries
  python3 scripts/rebuild_summaries.py --dry-run    # print the diff only
  python3 scripts/rebuild_summaries.py --methods insightreplay
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUTPUTS = ROOT / "outputs"

MODELS = [
    "qwen35_9b",
    "gemma4_e4b_it",
    "r1_distill_qwen_7b",
    "qwen35_35b_a3b",
    "gemma4_31b_it",
    "r1_distill_qwen_32b",
]
DATASETS = ["aime", "gpqa", "livecodebench", "hmmt"]
METHODS = {
    "baseline":             ("__unlimited",            "raw_baseline.jsonl"),
    "insightreplay":        ("__insightreplay",        "raw_insightreplay.jsonl"),
    "insightreplay_3turns": ("__insightreplay_3turns", "raw_insightreplay_3turns.jsonl"),
    "verify_only":          ("__verify_only",          "raw_verify_only.jsonl"),
}


def compute_stats(path: Path) -> dict | None:
    if not path.is_file():
        return None
    n = nc = 0
    toks: list[int] = []
    steps: list[int] = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            n += 1
            if r.get("correct"):
                nc += 1
            toks.append(int(r.get("completion_tokens", 0)))
            steps.append(int(r.get("num_steps", 1)))
    if n == 0:
        return None
    return {
        "n": n,
        "nc": nc,
        "pct": f"{nc/n:.3f}",
        "avg_tok": f"{sum(toks)/n:.1f}",
        "max_tok": max(toks) if toks else 0,
        "avs": f"{sum(steps)/n:.2f}",
    }


def format_row(model: str, ds: str, method: str, stats: dict,
               out_dir: str) -> str:
    return (f"{model},{ds},{method},{stats['n']},{stats['nc']},"
            f"{stats['pct']},{stats['avg_tok']},{stats['max_tok']},"
            f"{stats['avs']},{out_dir}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--methods", nargs="*", default=list(METHODS.keys()),
                    choices=list(METHODS.keys()))
    ap.add_argument("--dry-run", action="store_true",
                    help="Print new contents without writing")
    args = ap.parse_args()

    for method in args.methods:
        dir_suffix, fname = METHODS[method]
        out_csv = OUTPUTS / f"summary_{method}.csv"
        rows: list[str] = []
        for model in MODELS:
            for ds in DATASETS:
                out_dir = f"outputs/{model}__{ds}{dir_suffix}"
                path = OUTPUTS / f"{model}__{ds}{dir_suffix}" / fname
                stats = compute_stats(path)
                if stats is None:
                    print(f"  skip {model} x {ds} [{method}]: no jsonl at "
                          f"{path.relative_to(ROOT)}")
                    continue
                rows.append(format_row(model, ds, method, stats, out_dir))

        # Preserve any existing contents for a diff view
        existing = out_csv.read_text() if out_csv.is_file() else ""
        new_contents = "\n".join(rows) + ("\n" if rows else "")
        if args.dry_run:
            print(f"\n===== {out_csv.relative_to(ROOT)} =====")
            print(f"--- existing ({existing.count(chr(10))} lines) ---")
            print(existing.rstrip() or "(empty / missing)")
            print(f"--- proposed ({len(rows)} lines) ---")
            print(new_contents.rstrip())
        else:
            out_csv.write_text(new_contents)
            print(f"  wrote {len(rows)} rows → {out_csv.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
