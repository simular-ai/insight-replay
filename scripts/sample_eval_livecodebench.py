#!/usr/bin/env python3
import argparse
import os
import ast
import json
import random
import re
import subprocess
import tempfile
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import sys
import time

try:
    from tqdm import tqdm
except Exception:  # pragma: no cover - optional dependency
    tqdm = None


def parse_args():
    p = argparse.ArgumentParser(description="Sample Docker eval for LiveCodeBench.")
    p.add_argument("--dataset", required=True, help="Path to livecodebench JSONL dataset")
    p.add_argument("--pred", required=True, help="Pass@K shard JSONL or directory of shards")
    p.add_argument("--num", type=int, default=10, help="Number of cases to sample")
    p.add_argument("--max-k", type=int, default=4, help="Max candidates per case")
    p.add_argument("--docker-image", default="python:3.10-slim")
    p.add_argument("--timeout", type=int, default=20)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--workers", type=int, default=1, help="Parallel workers")
    p.add_argument("--case-id", default="", help="Evaluate a single case id/question_id")
    p.add_argument("--debug", action="store_true", help="Print docker stderr/stdout for each candidate")
    return p.parse_args()


def strip_code_fences(text: str) -> str:
    if text is None:
        return ""
    s = text.strip()
    if "```" not in s:
        return s
    start = s.find("```")
    end = s.rfind("```")
    if start == end:
        return s.replace("```", "").strip()
    inner = s[start + 3 : end]
    if inner.startswith("python"):
        inner = inner[len("python") :]
    return inner.strip("\n")


def _safe_literal_eval(text):
    try:
        return ast.literal_eval(text)
    except Exception:
        return text


def _parse_args_lines(inp: str):
    lines = [ln for ln in (inp or "").splitlines() if ln.strip() != ""]
    if not lines:
        return []
    return [_safe_literal_eval(ln) for ln in lines]


def load_dataset(path: str):
    rows = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            pid = obj.get("problem_id") or obj.get("id") or obj.get("question_id")
            if pid is None:
                continue
            rows[str(pid)] = obj
    return rows


def load_preds(path: str):
    p = Path(path)
    files = [p] if p.is_file() else sorted(p.glob("passk_shard_*.jsonl"))
    rows = []
    for fp in files:
        with fp.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rows.append(json.loads(line))
    return rows


def parse_verification_info(row):
    info = row.get("verification_info") or ""
    tests = []
    func_name = None
    if info:
        try:
            info_obj = json.loads(info)
        except Exception:
            info_obj = None
        if info_obj:
            gt_raw = info_obj.get("ground_truth", "[]")
            try:
                tests = json.loads(gt_raw)
            except Exception:
                tests = []
            if tests and isinstance(tests[0], dict):
                func_name = tests[0].get("metadata", {}).get("func_name")
    if not tests:
        tests = row.get("all_test_cases") or []
    if not func_name:
        q = row.get("question") or ""
        m = re.search(r"def\s+(\w+)\s*\(", q)
        if m:
            func_name = m.group(1)
    return tests, func_name


def build_runner_func(func_name: str, tests):
    return (
        "import ast\n"
        "import os\n"
        "import sys\n"
        "import traceback\n\n"
        "DEBUG = os.getenv('LCB_DEBUG') == '1'\n\n"
        "def _safe_literal_eval(text):\n"
        "    try:\n"
        "        return ast.literal_eval(text)\n"
        "    except Exception:\n"
        "        return text\n\n"
        "def _parse_args_lines(inp):\n"
        "    lines = [ln for ln in (inp or '').splitlines() if ln.strip() != '']\n"
        "    if not lines:\n"
        "        return []\n"
        "    return [_safe_literal_eval(ln) for ln in lines]\n\n"
        "from solution import Solution\n\n"
        f"fn = getattr(Solution(), {func_name!r})\n\n"
        f"TESTS = {repr(tests)}\n\n"
        "for t in TESTS:\n"
        "    args = _parse_args_lines(t['input'])\n"
        "    expected = _safe_literal_eval(t['output'])\n"
        "    try:\n"
        "        result = fn(*args)\n"
        "    except Exception:\n"
        "        if DEBUG:\n"
        "            traceback.print_exc()\n"
        "        raise SystemExit(1)\n"
        "    if result != expected:\n"
        "        if DEBUG:\n"
        "            print('MISMATCH', file=sys.stderr)\n"
        "            print('expected:', repr(expected), file=sys.stderr)\n"
        "            print('got:', repr(result), file=sys.stderr)\n"
        "        raise SystemExit(1)\n\n"
        "print('OK')\n"
    )


def build_runner_stdin(tests):
    return (
        "import os\n"
        "import subprocess\n"
        "import sys\n\n"
        "DEBUG = os.getenv('LCB_DEBUG') == '1'\n\n"
        f"TESTS = {repr(tests)}\n\n"
        "for t in TESTS:\n"
        "    inp = t.get('input', '')\n"
        "    expected = t.get('output', '')\n"
        "    try:\n"
        "        p = subprocess.run(\n"
        "            ['python', 'solution.py'],\n"
        "            input=inp,\n"
        "            text=True,\n"
        "            capture_output=True,\n"
        "            check=False,\n"
        "        )\n"
        "    except Exception:\n"
        "        if DEBUG:\n"
        "            print('SUBPROCESS_EXCEPTION', file=sys.stderr)\n"
        "        raise SystemExit(1)\n"
        "    if p.returncode != 0:\n"
        "        if DEBUG:\n"
        "            print('SUBPROCESS_EXIT', p.returncode, file=sys.stderr)\n"
        "            if p.stdout:\n"
        "                print('stdout:', repr(p.stdout), file=sys.stderr)\n"
        "            if p.stderr:\n"
        "                print('stderr:', repr(p.stderr), file=sys.stderr)\n"
        "        raise SystemExit(1)\n"
        "    got = (p.stdout or '').strip()\n"
        "    exp = (expected or '').strip()\n"
        "    if got != exp:\n"
        "        if DEBUG:\n"
        "            print('MISMATCH', file=sys.stderr)\n"
        "            print('expected:', repr(exp), file=sys.stderr)\n"
        "            print('got:', repr(got), file=sys.stderr)\n"
        "        raise SystemExit(1)\n\n"
        "print('OK')\n"
    )


def run_candidate(code: str, func_name: str, tests, image: str, timeout: int, debug: bool):
    code = strip_code_fences(code)
    if not code:
        return False, "empty_code", "", ""
    testtype = None
    if tests and isinstance(tests[0], dict):
        testtype = tests[0].get("testtype")
    if not func_name and testtype != "stdin":
        return False, "missing_func_name", "", ""
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        (tmp_path / "solution.py").write_text(code, encoding="utf-8")
        if func_name:
            runner = build_runner_func(func_name, tests)
        else:
            runner = build_runner_stdin(tests)
        (tmp_path / "runner.py").write_text(runner, encoding="utf-8")
        cmd = [
            "docker",
            "run",
            "--rm",
            "-i",
            "--network",
            "none",
            "--cpus",
            "1",
            "--memory",
            "2g",
            "-u",
            f"{os.getuid()}:{os.getgid()}",
            "-e",
            "PYTHONDONTWRITEBYTECODE=1",
            "-e",
            f"LCB_DEBUG={'1' if debug else '0'}",
            "-v",
            f"{tmp_path}:/work",
            "-w",
            "/work",
            image,
            "python",
            "runner.py",
        ]
        try:
            p = subprocess.run(
                cmd,
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=timeout,
            )
            return True, "ok", p.stdout or "", p.stderr or ""
        except subprocess.CalledProcessError as e:
            return False, f"exit_{e.returncode}", e.stdout or "", e.stderr or ""
        except Exception as e:
            return False, f"exception_{type(e).__name__}", "", str(e)


def eval_one(task):
    pid, drow, candidates, image, timeout, debug = task
    tests, func_name = parse_verification_info(drow)
    testtype = None
    if tests and isinstance(tests[0], dict):
        testtype = tests[0].get("testtype")
    if not tests or not candidates:
        return pid, 0, 0, 0, False
    if not func_name and testtype != "stdin":
        return pid, 0, 0, 0, False
    correct = 0
    wrong = 0
    for idx, code in enumerate(candidates):
        ok, reason, out, err = run_candidate(code, func_name, tests, image, timeout, debug)
        if debug:
            print(f"[debug] case={pid} cand={idx} ok={ok} reason={reason}")
            print("[debug] stdout:")
            print(out if out else "<empty>")
            print("[debug] stderr:")
            print(err if err else "<empty>")
        if ok:
            correct += 1
        else:
            wrong += 1
    total = correct + wrong
    any_ok = correct > 0
    return pid, correct, wrong, total, any_ok


def render_progress(done: int, total: int, started_at: float):
    if total <= 0:
        return
    ratio = min(max(done / total, 0.0), 1.0)
    width = 30
    filled = int(ratio * width)
    bar = "#" * filled + "-" * (width - filled)
    elapsed = max(time.time() - started_at, 0.0)
    rate = done / elapsed if elapsed > 0 else 0.0
    msg = f"[{bar}] {done}/{total} ({ratio*100:5.1f}%) {rate:5.2f} it/s"
    print(msg, end="\r", file=sys.stderr, flush=True)


def main():
    args = parse_args()
    random.seed(args.seed)
    dataset = load_dataset(args.dataset)
    question_index = {}
    for drow in dataset.values():
        q = drow.get("question")
        if q and q not in question_index:
            question_index[q] = drow
    preds = load_preds(args.pred)
    if not preds:
        print("No pred rows found")
        return
    random.shuffle(preds)

    tasks = []
    for row in preds:
        if len(tasks) >= args.num:
            break
        pid = row.get("id") or row.get("problem_id")
        drow = dataset.get(str(pid)) if pid is not None else None
        if not drow:
            q = row.get("question")
            if q:
                drow = question_index.get(q)
                if drow:
                    pid = drow.get("problem_id") or drow.get("id") or drow.get("question_id")
        if not drow:
            continue
        candidates = (row.get("pred_answers") or [])[: args.max_k]
        if not candidates:
            continue
        if args.case_id and str(pid) != str(args.case_id):
            continue
        tasks.append((pid, drow, candidates, args.docker_image, args.timeout, args.debug))

    tested = 0
    passed = 0
    total_correct = 0
    total_wrong = 0
    started_at = time.time()
    if args.case_id:
        args.workers = 1
    if args.workers <= 1:
        iterator = tasks
        if tqdm is not None:
            iterator = tqdm(tasks, total=len(tasks), file=sys.stderr, desc="Evaluating")
        for task in iterator:
            pid, correct, wrong, total, any_ok = eval_one(task)
            tested += 1
            if any_ok:
                passed += 1
            total_correct += correct
            total_wrong += wrong
            print(f"{pid}: correct={correct} wrong={wrong} total={total}")
            if tqdm is None:
                render_progress(tested, len(tasks), started_at)
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            futures = [ex.submit(eval_one, t) for t in tasks]
            if tqdm is None:
                for fut in as_completed(futures):
                    pid, correct, wrong, total, any_ok = fut.result()
                    tested += 1
                    if any_ok:
                        passed += 1
                    total_correct += correct
                    total_wrong += wrong
                    print(f"{pid}: correct={correct} wrong={wrong} total={total}")
                    render_progress(tested, len(tasks), started_at)
            else:
                with tqdm(total=len(futures), file=sys.stderr, desc="Evaluating") as pbar:
                    for fut in as_completed(futures):
                        pid, correct, wrong, total, any_ok = fut.result()
                        tested += 1
                        if any_ok:
                            passed += 1
                        total_correct += correct
                        total_wrong += wrong
                        print(f"{pid}: correct={correct} wrong={wrong} total={total}")
                        pbar.update(1)

    if tasks and tqdm is None:
        print(file=sys.stderr)
    print(
        json.dumps(
            {
                "tested": tested,
                "passed": passed,
                "total_correct": total_correct,
                "total_wrong": total_wrong,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
