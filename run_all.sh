#!/usr/bin/env bash
# Single-shot driver: runs all four methods on every (model × dataset) cell
# in one pass per model.
#
# Methods (per cell, in order):
#   1. baseline       — unconstrained sampling (no continuation injection)
#   2. verify_only    — naked re-think prompt as ablation
#   3. method 1-turn  — extract-and-replay, single round
#   4. method 3-turn  — extract-and-replay, three rounds
#
# Each cell is idempotent: if its output jsonl already exists, that step is
# skipped. vLLM is started once per model and shared across the four methods
# and all datasets, so one model load amortizes 16 method runs (4 methods ×
# 4 datasets). LCB is auto-graded after every run.
#
# Inputs:
#   ./scripts/run_sampling.py                  — per-method sampler
#   ./scripts/grade_livecodebench.py           — LCB code grader
#   ./scripts/compare_baseline_insightreplay.py — A/B vs baseline reporter
#   ./data/{aime,gpqa_diamond_test,livecodebench_v5,hmmt}.jsonl
#
# Outputs (per cell):
#   ./outputs/<model>__<dataset>__unlimited/raw_baseline.jsonl
#   ./outputs/<model>__<dataset>__verify_only/raw_verify_only.jsonl
#   ./outputs/<model>__<dataset>__insightreplay/raw_insightreplay.jsonl
#   ./outputs/<model>__<dataset>__insightreplay_3turns/raw_insightreplay_3turns.jsonl
#   ./outputs/summary_{baseline,verify_only,insightreplay,insightreplay_3turns}.csv
set -euo pipefail
cd "$(dirname "$0")"

export PYTHONPATH="$(pwd)/scripts${PYTHONPATH:+:$PYTHONPATH}"

# --- Sampling defaults (override via env) ---
export FBE_NUM_SAMPLES="${FBE_NUM_SAMPLES:-16}"
export FBE_BATCH_SIZE="${FBE_BATCH_SIZE:-128}"
export FBE_TEMPERATURE="${FBE_TEMPERATURE:-1.0}"
export FBE_TOP_P="${FBE_TOP_P:-0.95}"
export FBE_MAX_TOKENS="${FBE_MAX_TOKENS:-60000}"
export FBE_INSIGHTREPLAY_EXTRACT_MAX_TOKENS="${FBE_INSIGHTREPLAY_EXTRACT_MAX_TOKENS:-10000}"
export FBE_INSIGHTREPLAY_CONT_MAX_TOKENS="${FBE_INSIGHTREPLAY_CONT_MAX_TOKENS:-30000}"
export FBE_HTTP_TIMEOUT="${FBE_HTTP_TIMEOUT:-7200}"
export FBE_VLLM_PORT="${FBE_VLLM_PORT:-8270}"
export FBE_GPU_IDS="${FBE_GPU_IDS:-0,1,2,3,4,5,6,7}"
export FBE_GPU_MEM="${FBE_GPU_MEM:-0.92}"
export FBE_GRADE_WORKERS="${FBE_GRADE_WORKERS:-8}"
export FBE_GRADE_TIMEOUT="${FBE_GRADE_TIMEOUT:-20}"
# 8 DP shards loading concurrently can trip HuggingFace's anonymous rate
# limit; force-offline keeps the load local once the model snapshot is
# cached.
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"

# Datasets to evaluate — the full 4-benchmark grid reported in the paper.
DATASETS=(aime hmmt gpqa livecodebench)

NUM_GPUS=$(echo "$FBE_GPU_IDS" | tr ',' '\n' | wc -l)
mkdir -p outputs
LOG="outputs/.run_all.log"
exec > >(tee -a "$LOG") 2>&1

echo ">>> ================================================================="
echo ">>> driver: baseline + verify_only + 1-turn + 3-turn"
echo ">>> $(date)   port=$FBE_VLLM_PORT  GPUs=$FBE_GPU_IDS  gpu_mem=$FBE_GPU_MEM"
echo ">>> ================================================================="

# Preflight: confirm all required script entry points exist.
python3 - <<'PY' || { echo "### scripts missing required entry points"; exit 1; }
import inspect, sys
sys.path.insert(0, "scripts")
import run_sampling, prompt
for name in ("run_baseline", "run_methodb_finding"):
    assert hasattr(run_sampling, name), f"run_sampling.{name} missing"
sig_method = inspect.signature(run_sampling.run_methodb_finding).parameters
for arg in ("turns", "ablation"):
    assert arg in sig_method, f"run_methodb_finding has no {arg} kwarg"
sig_inj = inspect.signature(prompt.format_methodb_injection).parameters
assert "variant" in sig_inj, "format_methodb_injection has no variant kwarg"
print("[preflight] all entry points present")
PY

# --- vLLM lifecycle ---
VLLM_PID=""
cleanup_vllm() {
    if [ -n "${VLLM_PID:-}" ] && kill -0 "$VLLM_PID" 2>/dev/null; then
        echo "### stopping vLLM (pid=$VLLM_PID)"
        kill "$VLLM_PID" 2>/dev/null || true
        for _ in $(seq 1 30); do
            kill -0 "$VLLM_PID" 2>/dev/null || break
            sleep 1
        done
        kill -9 "$VLLM_PID" 2>/dev/null || true
        wait "$VLLM_PID" 2>/dev/null || true
    fi
    VLLM_PID=""
}
trap cleanup_vllm EXIT INT TERM

start_vllm() {
    local MODEL_KEY="$1"
    local CTX="$2"
    local VLLM_LOG="outputs/.run_all_${MODEL_KEY}_vllm.log"
    read MODEL_PATH TP_SIZE <<< "$(python3 - <<EOF
from prompt import MODELS
m = MODELS["$MODEL_KEY"]
print(m.local_path, m.tp_size)
EOF
)"
    local DP_SIZE=$((NUM_GPUS / TP_SIZE))
    echo "### starting vLLM for $MODEL_KEY  TP=$TP_SIZE DP=$DP_SIZE ctx=$CTX"

    CUDA_VISIBLE_DEVICES=$FBE_GPU_IDS python3 -m vllm.entrypoints.openai.api_server \
        --model "$MODEL_PATH" \
        --port "$FBE_VLLM_PORT" \
        --tensor-parallel-size "$TP_SIZE" \
        --data-parallel-size "$DP_SIZE" \
        --gpu-memory-utilization "$FBE_GPU_MEM" \
        --max-model-len "$CTX" \
        --no-enable-log-requests \
        --trust-remote-code \
        > "$VLLM_LOG" 2>&1 &
    VLLM_PID=$!
    echo "### vLLM PID=$VLLM_PID  log=$VLLM_LOG"

    local ready=false
    for i in $(seq 1 900); do
        if curl -s "http://127.0.0.1:${FBE_VLLM_PORT}/v1/models" > /dev/null 2>&1; then
            ready=true; echo "### vLLM ready after ${i}x5s"; break
        fi
        if ! kill -0 $VLLM_PID 2>/dev/null; then
            echo "### vLLM died during startup!"; tail -200 "$VLLM_LOG"; return 1
        fi
        sleep 5
    done
    $ready || { echo "### vLLM never ready"; tail -200 "$VLLM_LOG"; return 1; }
}

_lcb_data_path() {
    python3 - <<EOF
from prompt import DATASETS
print(DATASETS["$1"].data_path)
EOF
}

# --- Per-method runners. Each is idempotent: if the output jsonl already
# exists the function is a no-op. The first time a cell is touched the
# method's output dir, jsonl, log, and (for LCB) grader log are all created.

_grade_lcb_if_code() {
    local DATASET_KEY="$1"
    local OUT_JSONL="$2"
    local LOG_PATH="$3"
    if [ "$DATASET_KEY" = "livecodebench" ]; then
        echo "--- grading livecodebench ---"
        python3 scripts/grade_livecodebench.py \
            --raw "$OUT_JSONL" \
            --dataset "$(_lcb_data_path "$DATASET_KEY")" \
            --workers "$FBE_GRADE_WORKERS" \
            --timeout "$FBE_GRADE_TIMEOUT" \
            2>&1 | tee -a "$LOG_PATH"
    fi
}

_write_summary() {
    local METHOD="$1"
    local MODEL_KEY="$2"
    local DATASET_KEY="$3"
    local OUT_JSONL="$4"
    local OUT_DIR="$5"
    python3 - <<PY
import json
rows = [json.loads(l) for l in open("$OUT_JSONL") if l.strip()]
n = len(rows); nc = sum(1 for r in rows if r.get("correct"))
toks = [r.get("completion_tokens", 0) for r in rows]
steps = [r.get("num_steps", 1) for r in rows]
avg = sum(toks)/len(toks) if toks else 0
mx  = max(toks) if toks else 0
avs = sum(steps)/len(steps) if steps else 0
pct = f"{nc/n:.3f}" if n else "nan"
with open("outputs/summary_${METHOD}.csv", "a") as f:
    f.write(f"$MODEL_KEY,$DATASET_KEY,$METHOD,{n},{nc},{pct},{avg:.1f},{mx},{avs:.2f},$OUT_DIR\n")
print(f"[summary $METHOD] {nc}/{n} acc={pct} avg_tok={avg:.0f} max_tok={mx} avg_steps={avs:.1f}")
PY
}

run_baseline() {
    local MODEL_KEY="$1" DATASET_KEY="$2" CTX="$3"
    local OUT_DIR="outputs/${MODEL_KEY}__${DATASET_KEY}__unlimited"
    local OUT_JSONL="$OUT_DIR/raw_baseline.jsonl"
    [ -f "$OUT_JSONL" ] && { echo "### [baseline] reuse $OUT_JSONL"; return 0; }
    mkdir -p "$OUT_DIR"
    echo ">>> [baseline] ${MODEL_KEY} × ${DATASET_KEY}  $(date +%H:%M:%S)"
    FBE_MAX_TOKENS="$CTX" \
    python3 scripts/run_sampling.py \
        --group baseline \
        --model-key "$MODEL_KEY" \
        --dataset-key "$DATASET_KEY" \
        --out "$OUT_JSONL" \
        --port "$FBE_VLLM_PORT" \
        --num-samples "$FBE_NUM_SAMPLES" \
        --batch-size "$FBE_BATCH_SIZE" \
        2>&1 | tee "$OUT_DIR/run_baseline.log"
    _grade_lcb_if_code "$DATASET_KEY" "$OUT_JSONL" "$OUT_DIR/grader.log"
    _write_summary baseline "$MODEL_KEY" "$DATASET_KEY" "$OUT_JSONL" "$OUT_DIR"
}

run_verify_only() {
    local MODEL_KEY="$1" DATASET_KEY="$2"
    local BASE_JSONL="outputs/${MODEL_KEY}__${DATASET_KEY}__unlimited/raw_baseline.jsonl"
    local OUT_DIR="outputs/${MODEL_KEY}__${DATASET_KEY}__verify_only"
    local OUT_JSONL="$OUT_DIR/raw_verify_only.jsonl"
    [ -f "$OUT_JSONL" ] && { echo "### [verify_only] reuse $OUT_JSONL"; return 0; }
    [ -f "$BASE_JSONL" ] || { echo "### [verify_only] missing baseline; skipping"; return 0; }
    mkdir -p "$OUT_DIR"
    echo ">>> [verify_only] ${MODEL_KEY} × ${DATASET_KEY}  $(date +%H:%M:%S)"
    python3 scripts/run_sampling.py \
        --group methodb \
        --ablation verify_only \
        --model-key "$MODEL_KEY" \
        --dataset-key "$DATASET_KEY" \
        --out "$OUT_JSONL" \
        --baseline-in "$BASE_JSONL" \
        --port "$FBE_VLLM_PORT" \
        --num-samples "$FBE_NUM_SAMPLES" \
        --batch-size "$FBE_BATCH_SIZE" \
        2>&1 | tee "$OUT_DIR/run_verify_only.log"
    _grade_lcb_if_code "$DATASET_KEY" "$OUT_JSONL" "$OUT_DIR/grader.log"
    _write_summary verify_only "$MODEL_KEY" "$DATASET_KEY" "$OUT_JSONL" "$OUT_DIR"
}

run_method_1turn() {
    local MODEL_KEY="$1" DATASET_KEY="$2"
    local BASE_JSONL="outputs/${MODEL_KEY}__${DATASET_KEY}__unlimited/raw_baseline.jsonl"
    local OUT_DIR="outputs/${MODEL_KEY}__${DATASET_KEY}__insightreplay"
    local OUT_JSONL="$OUT_DIR/raw_insightreplay.jsonl"
    [ -f "$OUT_JSONL" ] && { echo "### [1-turn] reuse $OUT_JSONL"; return 0; }
    [ -f "$BASE_JSONL" ] || { echo "### [1-turn] missing baseline; skipping"; return 0; }
    mkdir -p "$OUT_DIR"
    echo ">>> [1-turn] ${MODEL_KEY} × ${DATASET_KEY}  $(date +%H:%M:%S)"
    python3 scripts/run_sampling.py \
        --group methodb \
        --model-key "$MODEL_KEY" \
        --dataset-key "$DATASET_KEY" \
        --out "$OUT_JSONL" \
        --baseline-in "$BASE_JSONL" \
        --port "$FBE_VLLM_PORT" \
        --num-samples "$FBE_NUM_SAMPLES" \
        --batch-size "$FBE_BATCH_SIZE" \
        2>&1 | tee "$OUT_DIR/run_insightreplay.log"
    _grade_lcb_if_code "$DATASET_KEY" "$OUT_JSONL" "$OUT_DIR/grader.log"
    _write_summary insightreplay "$MODEL_KEY" "$DATASET_KEY" "$OUT_JSONL" "$OUT_DIR"
    echo "--- A/B vs baseline ---"
    python3 scripts/compare_baseline_insightreplay.py \
        --baseline "$BASE_JSONL" \
        --insightreplay "$OUT_JSONL" \
        --dump-dir "$OUT_DIR/bucket_samples" \
        2>&1 | tee "$OUT_DIR/compare.log"
}

run_method_3turns() {
    local MODEL_KEY="$1" DATASET_KEY="$2"
    local BASE_JSONL="outputs/${MODEL_KEY}__${DATASET_KEY}__unlimited/raw_baseline.jsonl"
    local PRIOR_FR="outputs/${MODEL_KEY}__${DATASET_KEY}__insightreplay/raw_insightreplay.jsonl"
    local OUT_DIR="outputs/${MODEL_KEY}__${DATASET_KEY}__insightreplay_3turns"
    local OUT_JSONL="$OUT_DIR/raw_insightreplay_3turns.jsonl"
    [ -f "$OUT_JSONL" ] && { echo "### [3-turn] reuse $OUT_JSONL"; return 0; }
    [ -f "$BASE_JSONL" ] || { echo "### [3-turn] missing baseline; skipping"; return 0; }
    mkdir -p "$OUT_DIR"
    echo ">>> [3-turn] ${MODEL_KEY} × ${DATASET_KEY}  $(date +%H:%M:%S)"
    # If the 1-turn output exists, reuse it for turn 1 (saves ~33% of work).
    local PRIOR_FLAGS=()
    if [ -f "$PRIOR_FR" ]; then
        echo "### [3-turn] reusing prior 1-turn: $PRIOR_FR"
        PRIOR_FLAGS+=(--insightreplay-in "$PRIOR_FR")
    fi
    python3 scripts/run_sampling.py \
        --group methodb \
        --turns 3 \
        --model-key "$MODEL_KEY" \
        --dataset-key "$DATASET_KEY" \
        --out "$OUT_JSONL" \
        --baseline-in "$BASE_JSONL" \
        "${PRIOR_FLAGS[@]}" \
        --port "$FBE_VLLM_PORT" \
        --num-samples "$FBE_NUM_SAMPLES" \
        --batch-size "$FBE_BATCH_SIZE" \
        2>&1 | tee "$OUT_DIR/run_insightreplay_3turns.log"
    _grade_lcb_if_code "$DATASET_KEY" "$OUT_JSONL" "$OUT_DIR/grader.log"
    _write_summary insightreplay_3turns "$MODEL_KEY" "$DATASET_KEY" "$OUT_JSONL" "$OUT_DIR"
}

# --- One model, all 4 methods × all datasets, behind a single vLLM startup. ---
run_model() {
    local MODEL_KEY="$1" CTX="$2"
    # Optional 3rd arg overrides FBE_BATCH_SIZE for this model.
    local PREV_BATCH="$FBE_BATCH_SIZE"
    export FBE_BATCH_SIZE="${3:-$FBE_BATCH_SIZE}"
    export FBE_MAX_MODEL_LEN="$CTX"

    # Skip vLLM startup entirely if every (method, dataset) cell is done.
    local ALL_DONE=1
    for DS in "${DATASETS[@]}"; do
        for SUF in unlimited verify_only insightreplay insightreplay_3turns; do
            local FN
            case "$SUF" in
                unlimited)             FN=raw_baseline.jsonl;;
                verify_only)           FN=raw_verify_only.jsonl;;
                insightreplay)         FN=raw_insightreplay.jsonl;;
                insightreplay_3turns)  FN=raw_insightreplay_3turns.jsonl;;
            esac
            if [ ! -f "outputs/${MODEL_KEY}__${DS}__${SUF}/${FN}" ]; then
                ALL_DONE=0; break 2
            fi
        done
    done
    if [ "$ALL_DONE" = "1" ]; then
        echo "### ${MODEL_KEY}: all methods × datasets done — skipping vLLM"
        export FBE_BATCH_SIZE="$PREV_BATCH"
        return 0
    fi

    start_vllm "$MODEL_KEY" "$CTX"
    for DS in "${DATASETS[@]}"; do
        run_baseline       "$MODEL_KEY" "$DS" "$CTX"
        run_verify_only    "$MODEL_KEY" "$DS"
        run_method_1turn   "$MODEL_KEY" "$DS"
        run_method_3turns  "$MODEL_KEY" "$DS"
    done
    cleanup_vllm
    export FBE_BATCH_SIZE="$PREV_BATCH"
}

# ---- 8B tier (TP=1 × DP=8). Cheap; deeper HTTP queue is fine. ----
run_model qwen35_9b           200000  1000
run_model gemma4_e4b_it       131072  1000
run_model r1_distill_qwen_7b  131072  1000

# ---- 30B tier (TP=2 × DP=4). Default batch=128. ----
run_model qwen35_35b_a3b      200000
run_model gemma4_31b_it       200000
run_model r1_distill_qwen_32b 131072

echo ">>> ================================================================="
echo ">>> all done  $(date)"
echo ">>> ================================================================="
