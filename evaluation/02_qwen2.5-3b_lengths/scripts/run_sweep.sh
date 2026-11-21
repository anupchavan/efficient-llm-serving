#!/bin/bash
export PATH="$HOME/.local/bin:$PATH"
export VLLM_USE_FLASHINFER_SAMPLER=0
export VLLM_ATTENTION_BACKEND=FLASH_ATTN

BASE_MODEL="Qwen/Qwen2.5-3B-Instruct"
QUANT_MODEL="Qwen/Qwen2.5-3B-Instruct-AWQ"
DRAFT_MODEL="Qwen/Qwen2.5-0.5B"

# Denser grid focused on the interesting region (collapse of pure speculative decoding)
REQUEST_RATES=(2 4 8 12 16 24 32 48 64)
INPUT_LENGTHS=(512 1024 1536 2048 3072 4096)   # dropped 6144; added middle points
OUTPUT_LEN=128

RESULTS_DIR="./benchmark_results"
mkdir -p $RESULTS_DIR

run_benchmark() {
local MODEL_NAME=$1
local EXTRA_SERVER_ARGS=$2
local REGIME_NAME=$3

echo "--------------------------------------------------------"
echo "Starting Server: $MODEL_NAME"
echo "Regime: $REGIME_NAME | Args: $EXTRA_SERVER_ARGS"
# We do not quote $EXTRA_SERVER_ARGS here so bash parses the flags correctly
vllm serve --model $MODEL_NAME \
--gpu-memory-utilization 0.95 \
--max-model-len 8192 \
        $EXTRA_SERVER_ARGS &
    SERVER_PID=$!



echo "Waiting for vLLM server to initialize..."
while ! curl -s http://localhost:8000/health > /dev/null; do
    sleep 5
done
echo "Server is up! Starting benchmark runs..."



for RATE in "${REQUEST_RATES[@]}"; do
    for LEN in "${INPUT_LENGTHS[@]}"; do
        SAFE_MODEL=$(echo $MODEL_NAME | awk -F/ '{print $NF}')
        OUT_FILE="$RESULTS_DIR/${SAFE_MODEL}_${REGIME_NAME}_rate${RATE}_len${LEN}.json"

        # >>> RESUME LOGIC <<<
        # If you get kicked off mid-run, it skips what is already done!
        if [ -f "$OUT_FILE" ]; then
            echo "Skipping Rate=$RATE, InputLen=$LEN - Result already exists!"
            continue
        fi

        echo "Benchmarking: Rate=$RATE req/s, InputLen=$LEN"

        # Shared prefix for prefix-caching tests (same dataset for all runs → clean CR comparison)
        if [ "$LEN" -ge 1024 ]; then
            PREFIX_LEN=512
        else
            PREFIX_LEN=$((LEN / 2))
        fi
        SUFFIX_LEN=$((LEN - PREFIX_LEN))

        vllm bench serve \
            --model "$MODEL_NAME" \
            --dataset-name prefix_repetition \
            --request-rate "$RATE" \
            --prefix-repetition-prefix-len "$PREFIX_LEN" \
            --prefix-repetition-suffix-len "$SUFFIX_LEN" \
            --prefix-repetition-output-len "$OUTPUT_LEN" \
            --prefix-repetition-num-prefixes 32 \
            --save-result \
            --result-filename "$OUT_FILE"

        sleep 2 
    done
done



echo "Shutting down server (PID: $SERVER_PID)..."
kill $SERVER_PID
wait $SERVER_PID 2>/dev/null
sleep 5
}



# # Grab the command line argument
# RUN_ID=$1



# if [ -z "$RUN_ID" ]; then
# echo "Error: You must provide a Run ID (1-16)."
# echo "Example usage: ./test.sh 5"
# exit 1
# fi


# Reusable Argument Snippets
ARG_WQ="--quantization awq"
ARG_KV="--kv-cache-dtype fp8"
ARG_CR="--enable-prefix-caching"
ARG_MT="--speculative-config {\"method\":\"draft_model\",\"model\":\"$DRAFT_MODEL\",\"num_speculative_tokens\":5}"



run_benchmark "$BASE_MODEL" "" "baseline"
run_benchmark "$BASE_MODEL" "$ARG_MT" "MT"
run_benchmark "$BASE_MODEL" "$ARG_CR" "CR"
run_benchmark "$BASE_MODEL" "$ARG_CR $ARG_MT" "CR_MT"
run_benchmark "$BASE_MODEL" "$ARG_KV" "KV"
run_benchmark "$BASE_MODEL" "$ARG_KV $ARG_MT" "KV_MT"
run_benchmark "$BASE_MODEL" "$ARG_KV $ARG_CR" "KV_CR"
run_benchmark "$BASE_MODEL" "$ARG_KV $ARG_CR $ARG_MT" "KV_CR_MT"
run_benchmark "$QUANT_MODEL" "$ARG_WQ" "WQ"
run_benchmark "$QUANT_MODEL" "$ARG_WQ $ARG_MT" "WQ_MT"
run_benchmark "$QUANT_MODEL" "$ARG_WQ $ARG_CR" "WQ_CR"
run_benchmark "$QUANT_MODEL" "$ARG_WQ $ARG_CR $ARG_MT" "WQ_CR_MT"
run_benchmark "$QUANT_MODEL" "$ARG_WQ $ARG_KV" "WQ_KV"
run_benchmark "$QUANT_MODEL" "$ARG_WQ $ARG_KV $ARG_MT" "WQ_KV_MT"
run_benchmark "$QUANT_MODEL" "$ARG_WQ $ARG_KV $ARG_CR" "WQ_KV_CR"
run_benchmark "$QUANT_MODEL" "$ARG_WQ $ARG_KV $ARG_CR $ARG_MT" "WQ_KV_CR_MT"



echo "Finished Run ID $RUN_ID."