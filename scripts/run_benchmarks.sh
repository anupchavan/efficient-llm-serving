#!/bin/bash
# Runs the benchmark grid used in the report.
# For each configuration: start a vLLM server, send 1,000 requests at each rate,
# repeat each rate REPEATS times, save one JSON per run, then stop the server.
#
# Usage: ./scripts/run_benchmarks.sh            (from the repository root)
# Already-finished runs are skipped, so the script can be restarted.

export VLLM_USE_FLASHINFER_SAMPLER=0
export VLLM_ATTENTION_BACKEND=FLASH_ATTN

BASE_MODEL="Qwen/Qwen2.5-3B-Instruct"
AWQ_MODEL="Qwen/Qwen2.5-3B-Instruct-AWQ"
DRAFT_MODEL="Qwen/Qwen2.5-0.5B"

REQUEST_RATES=(2 4 8 12 16 24 32 48 64)
REPEATS=6
INPUT_LEN=1024
OUTPUT_LEN=128
NUM_PROMPTS=1000

RESULTS_DIR="./data/raw"
mkdir -p "$RESULTS_DIR"

ARG_AWQ="--quantization awq"
ARG_SPEC="--speculative-config {\"method\":\"draft_model\",\"model\":\"$DRAFT_MODEL\",\"num_speculative_tokens\":5}"

run_config() {
    local MODEL=$1 EXTRA_ARGS=$2 NAME=$3

    echo "=== $NAME: $MODEL $EXTRA_ARGS"
    # EXTRA_ARGS is left unquoted on purpose so the flags are split.
    vllm serve --model "$MODEL" \
        --gpu-memory-utilization 0.95 \
        --max-model-len 8192 \
        $EXTRA_ARGS &
    SERVER_PID=$!

    until curl -s http://localhost:8000/health > /dev/null; do
        if ! kill -0 $SERVER_PID 2>/dev/null; then
            echo "Server for $NAME exited during startup"; return 1
        fi
        sleep 5
    done

    for RATE in "${REQUEST_RATES[@]}"; do
        for REP in $(seq 1 $REPEATS); do
            OUT="$RESULTS_DIR/${NAME}_rate${RATE}_rep${REP}.json"
            [ -f "$OUT" ] && { echo "skip $OUT"; continue; }
            vllm bench serve \
                --model "$MODEL" \
                --dataset-name random \
                --random-input-len $INPUT_LEN \
                --random-output-len $OUTPUT_LEN \
                --num-prompts $NUM_PROMPTS \
                --request-rate $RATE \
                --save-result \
                --result-filename "$OUT"
            sleep 2
        done
    done

    kill $SERVER_PID
    wait $SERVER_PID 2>/dev/null
    sleep 5
}

run_config "$BASE_MODEL" ""                     "baseline"
run_config "$AWQ_MODEL"  "$ARG_AWQ"             "awq"
run_config "$BASE_MODEL" "$ARG_SPEC"            "spec"
run_config "$AWQ_MODEL"  "$ARG_AWQ $ARG_SPEC"   "awq_spec"
