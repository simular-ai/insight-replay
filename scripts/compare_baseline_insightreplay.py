#!/usr/bin/env python3
"""Row-level comparison of baseline vs ours.

Joins the baseline raw_baseline.jsonl with the method's raw output jsonl on
(problem_id, sample_idx) and reports the 4 transition buckets (CC / CW /
WC / WW) plus token / findings stats. Optionally dumps sample rows from
each bucket.

Usage:
  python compare_baseline_insightreplay.py \\
      --baseline      outputs/<model>__<dataset>__unlimited/raw_baseline.jsonl \\
      --insightreplay outputs/<model>__<dataset>__<suffix>/raw_<suffix>.jsonl \\
      [--dump-dir     outputs/.../diff]
"""
import argparse
import json
from pathlib import Path


def load(path):
    out = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            out[(r.get("problem_id"), r.get("sample_idx"))] = r
    return out


def bucket(a, b):
    if a and b: return "CC"
    if a and not b: return "CW"
    if not a and b: return "WC"
    return "WW"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline", required=True)
    # Accept both --insightreplay (canonical) and --methodb (legacy alias).
    ap.add_argument("--insightreplay", "--methodb", dest="insightreplay",
                    required=True)
    ap.add_argument("--dump-dir", default=None,
                    help="If set, dump up to --dump-n rows per bucket")
    ap.add_argument("--dump-n", type=int, default=5)
    args = ap.parse_args()

    base = load(args.baseline)
    mb = load(args.insightreplay)
    keys = sorted(set(base) & set(mb))
    print(f"[compare] baseline={len(base)} insightreplay={len(mb)} "
          f"joined={len(keys)}")

    buckets = {"CC": [], "CW": [], "WC": [], "WW": []}
    mode_counts = {}
    baseline_tok_sum = mb_tok_sum = cont_tok_sum = 0
    nfind_sum = 0
    n_methodb_triggered = 0

    for k in keys:
        b = base[k]
        m = mb[k]
        buckets[bucket(b.get("correct", False), m.get("correct", False))].append(k)
        mode = m.get("mode", "unknown")
        mode_counts[mode] = mode_counts.get(mode, 0) + 1
        baseline_tok_sum += b.get("completion_tokens", 0)
        mb_tok_sum += m.get("completion_tokens", 0)
        cont_tok_sum += m.get("continuation_tokens", 0)
        nfind_sum += m.get("num_findings", 0)
        if mode == "methodb":
            n_methodb_triggered += 1

    n = len(keys)
    print()
    print(f"  Transitions:")
    for b in ("CC", "CW", "WC", "WW"):
        print(f"    {b}: {len(buckets[b]):5d}  ({len(buckets[b])/n:6.2%})")
    b_correct = len(buckets["CC"]) + len(buckets["CW"])
    m_correct = len(buckets["CC"]) + len(buckets["WC"])
    delta = m_correct - b_correct
    print()
    print(f"  baseline      correct:  {b_correct}/{n}  ({b_correct/n:.3%})")
    print(f"  insightreplay  correct:  {m_correct}/{n}  ({m_correct/n:.3%})")
    print(f"  delta:                  {delta:+d}  ({delta/n:+.3%})  "
          f"[W→C={len(buckets['WC'])} gains, "
          f"C→W={len(buckets['CW'])} regressions]")
    print()
    print(f"  mode breakdown: {mode_counts}")
    print(f"  insightreplay triggered (not fallback): "
          f"{n_methodb_triggered}/{n}  ({n_methodb_triggered/n:.2%})")
    print()
    print(f"  avg baseline     tokens:  {baseline_tok_sum/n:8.1f}")
    print(f"  avg insightreplay tokens:  {mb_tok_sum/n:8.1f}  "
          f"(+{(mb_tok_sum-baseline_tok_sum)/n:.1f})")
    if n_methodb_triggered:
        print(f"  avg continuation tokens (triggered only): "
              f"{cont_tok_sum/n_methodb_triggered:.1f}")
        print(f"  avg findings injected (triggered only):  "
              f"{nfind_sum/n_methodb_triggered:.2f}")

    if args.dump_dir:
        out = Path(args.dump_dir)
        out.mkdir(parents=True, exist_ok=True)
        for b in ("WC", "CW", "CC", "WW"):
            rows = []
            for k in buckets[b][:args.dump_n]:
                rows.append({
                    "key": k, "baseline": base[k], "methodb": mb[k],
                })
            with open(out / f"bucket_{b}.json", "w") as f:
                json.dump(rows, f, ensure_ascii=False, indent=2)
        print(f"\n  sample rows → {out}/bucket_{{CC,CW,WC,WW}}.json")


if __name__ == "__main__":
    main()
