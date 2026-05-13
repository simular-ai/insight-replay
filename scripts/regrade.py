#!/usr/bin/env python3
"""Regrade raw_baseline.jsonl / raw_insightreplay.jsonl in-place.

Re-runs extract_answer + grade_answer on each row's `response` field and
overwrites `predicted_answer` / `correct` / `methodb_correct`. Used to apply
an answer-extractor regex fix without re-running the model.

Skips livecodebench (uses external grader; `correct` is set by
grade_livecodebench.py, not grade_answer).

Usage:
  python scripts/regrade.py outputs/<dir>/raw_*.jsonl [<more paths> ...]
"""
import json
import sys
from pathlib import Path

_THIS = Path(__file__).resolve().parent
sys.path.insert(0, str(_THIS))

from prompt import extract_answer, grade_answer, strip_thinking


def _dataset_from_path(path: Path) -> str | None:
    parent = path.parent.name  # e.g. qwen35_9b__gpqa__unlimited
    parts = parent.split("__")
    if len(parts) >= 2:
        ds = parts[1]
        if ds in ("aime", "gpqa", "livecodebench", "hmmt"):
            return ds
    return None


def _model_from_path(path: Path) -> str | None:
    parent = path.parent.name
    parts = parent.split("__")
    return parts[0] if parts else None


def regrade(path: Path) -> dict | None:
    ds = _dataset_from_path(path)
    if ds is None:
        print(f"  [skip] {path}: can't infer dataset from dirname")
        return None
    if ds == "livecodebench":
        print(f"  [skip] {path}: livecodebench (external grader)")
        return None
    model_key = _model_from_path(path)
    if not path.exists() or path.stat().st_size == 0:
        print(f"  [skip] {path}: missing or empty")
        return None

    rows = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))

    pred_changed = 0
    correct_flipped = 0
    new_rights = 0
    new_wrongs = 0
    for r in rows:
        resp = r.get("response", "") or ""
        post = strip_thinking(model_key, resp)
        new_pred = extract_answer(ds, post)
        new_correct = grade_answer(ds, new_pred, r.get("gt_answer"))
        old_pred = r.get("predicted_answer")
        old_correct = bool(r.get("correct"))
        if old_pred != new_pred:
            pred_changed += 1
        if old_correct != bool(new_correct):
            correct_flipped += 1
            if new_correct and not old_correct:
                new_rights += 1
            elif old_correct and not new_correct:
                new_wrongs += 1
        r["predicted_answer"] = new_pred
        r["correct"] = bool(new_correct)
        if "methodb_correct" in r:
            r["methodb_correct"] = bool(new_correct)

    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    tmp.replace(path)

    return {
        "n": len(rows),
        "pred_changed": pred_changed,
        "correct_flipped": correct_flipped,
        "new_rights": new_rights,
        "new_wrongs": new_wrongs,
    }


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    for p in sys.argv[1:]:
        path = Path(p)
        result = regrade(path)
        if result is not None:
            print(
                f"  {path}: n={result['n']:5d}  "
                f"pred_changed={result['pred_changed']:4d}  "
                f"correct_flipped={result['correct_flipped']:4d}  "
                f"(+{result['new_rights']} rights, "
                f"-{result['new_wrongs']} wrongs)"
            )


if __name__ == "__main__":
    main()
