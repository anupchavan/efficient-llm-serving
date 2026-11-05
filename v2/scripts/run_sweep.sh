#!/bin/bash
# Larger-model sweep with real prompts. Run from the repository root on the GPU host.
#
#   PILOT=1 ./v2/scripts/run_sweep.sh   short pilot: measures capacity and time per run
#   ./v2/scripts/run_sweep.sh           full sweep (set RATES after the pilot)
#
# Finished runs are skipped, so the script can be restarted after a disconnect.
# Prefix caching is OFF for every main configuration, so repeated or similar
# prompts never reuse cached work. It is only switched on in the prefix experiment,
# which runs the same workload with caching off and on.
set -u
export VLLM_USE_FLASHINFER_SAMPLER=0

BASE_MODEL=${BASE_MODEL:-Qwen/Qwen3-14B}
AWQ_MODEL=${AWQ_MODEL:-Qwen/Qwen3-14B-AWQ}
DRAFT_MODEL=${DRAFT_MODEL:-Qwen/Qwen3-0.6B}
PROMPTS=${PROMPTS:-./v2/prompts}
OUT=${OUT:-./v2/data/raw}
OUTPUT_LEN=128
MAX_MODEL_LEN=8192
# Qwen's recommended sampling for non-thinking mode, set explicitly so every run matches.
SAMPLING="--temperature 0.7 --top-p 0.8 --top-k 20"

if [ "${PILOT:-0}" = 1 ]; then
    OUT=./v2/data/pilot
    RATES=(1 2 4 8)
    WORKLOADS=(len1024 len4096)
    REPEATS=1
    NUM_PROMPTS=200
    CONFIGS=(fp16 awq_spec)
    PREFIX_EXPERIMENT=0
else
    RATES=(${RATES:-0.5 1 2 3 4 6 8 16 32})   # placeholder: set from the pilot
    WORKLOADS=(${WORKLOADS:-natural len512 len1024 len2048 len4096})
    REPEATS=${REPEATS:-3}
    NUM_PROMPTS=${NUM_PROMPTS:-500}
    CONFIGS=(${CONFIGS:-fp16 awq spec awq_spec})
    PREFIX_EXPERIMENT=${PREFIX_EXPERIMENT:-1}
fi
mkdir -p "$OUT" ./v2/logs

SPEC="--speculative-config {\"method\":\"draft_model\",\"model\":\"$DRAFT_MODEL\",\"num_speculative_tokens\":5}"
NO_CACHE="--no-enable-prefix-caching"
CACHE="--enable-prefix-caching"

model_of() { case $1 in awq*) echo "$AWQ_MODEL" ;; *) echo "$BASE_MODEL" ;; esac; }
args_of() {
    case $1 in
        fp16)     echo "$NO_CACHE" ;;
        awq)      echo "--quantization awq $NO_CACHE" ;;
        spec)     echo "$SPEC $NO_CACHE" ;;
        awq_spec) echo "--quantization awq $SPEC $NO_CACHE" ;;
        fp16_cacheoff) echo "$NO_CACHE" ;;
        fp16_cacheon)  echo "$CACHE" ;;
        awq_cacheoff)  echo "--quantization awq $NO_CACHE" ;;
        awq_cacheon)   echo "--quantization awq $CACHE" ;;
    esac
}

start_server() {
    local NAME=$1 MODEL=$2 ARGS=$3
    echo "=== $NAME: $MODEL $ARGS"
    # ARGS is unquoted on purpose so the flags are split.
    vllm serve "$MODEL" --gpu-memory-utilization 0.95 --max-model-len $MAX_MODEL_LEN \
        --seed 0 $ARGS > "./v2/logs/server_$NAME.log" 2>&1 &
    SERVER_PID=$!
    local WAITED=0
    until curl -s http://localhost:8000/health > /dev/null; do
        kill -0 $SERVER_PID 2>/dev/null || { echo "server $NAME died, see v2/logs/server_$NAME.log"; return 1; }
        sleep 5; WAITED=$((WAITED + 5))
        [ $WAITED -gt 1800 ] && { echo "server $NAME not up after 30 min"; kill $SERVER_PID; return 1; }
    done
    # Record what the server actually chose (KV blocks, flags) next to the data.
    grep -E "KV cache|num_gpu_blocks|Maximum concurrency|prefix|speculative" \
        "./v2/logs/server_$NAME.log" > "$OUT/server_$NAME.txt" || true
    nvidia-smi --query-gpu=name,memory.total,memory.used,driver_version --format=csv >> "$OUT/server_$NAME.txt"
}

stop_server() { kill $SERVER_PID; wait $SERVER_PID 2>/dev/null; sleep 5; }

# With prefix caching on, a prompt the server has already seen would be served
# from cache. So each prefix run gets its own slice of user prompts that no
# earlier run on this server used; only the system prompts repeat.
prefix_slice() {
    local K=$1 FILE="$PROMPTS/prefix.jsonl" S="$PROMPTS/.slice.jsonl"
    local FROM=$((K * NUM_PROMPTS + 1)) TO=$(((K + 1) * NUM_PROMPTS))
    [ "$(wc -l < "$FILE")" -ge $TO ] || { echo "prefix.jsonl too short for run $K" >&2; return 1; }
    sed -n "${FROM},${TO}p" "$FILE" > "$S"
    echo "$S"
}

run_grid() {
    local NAME=$1 MODEL=$2; shift 2
    local WLS=("$@") K=0
    for WL in "${WLS[@]}"; do
        for RATE in "${RATES[@]}"; do
            for REP in $(seq 1 $REPEATS); do
                local F="$OUT/${NAME}_${WL}_rate${RATE}_rep${REP}.json"
                local DATA="$PROMPTS/$WL.jsonl" ORDER=""
                if [ "$WL" = prefix ]; then
                    DATA=$(prefix_slice $K) || return 1
                    ORDER="--disable-shuffle"
                fi
                K=$((K + 1))
                [ -f "$F" ] && { echo "skip $F"; continue; }
                # --seed REP draws a different shuffle of the prompt file per repeat.
                vllm bench serve --model "$MODEL" \
                    --dataset-name custom --dataset-path "$DATA" $ORDER --skip-chat-template \
                    --custom-output-len $OUTPUT_LEN --ignore-eos $SAMPLING \
                    --num-prompts $NUM_PROMPTS --request-rate $RATE --seed $REP \
                    --save-result --save-detailed --result-filename "$F" \
                    --metadata config=$NAME workload=$WL rep=$REP \
                    >> "./v2/logs/bench_$NAME.log" 2>&1 || {
                    # Stop at the first failure instead of burning GPU time on a broken setup.
                    echo "FAILED $F, see v2/logs/bench_$NAME.log"; return 1; }
                sleep 2
            done
        done
    done
}

for C in "${CONFIGS[@]}"; do
    M=$(model_of $C)
    start_server $C "$M" "$(args_of $C)" || continue
    run_grid $C "$M" "${WORKLOADS[@]}" || { stop_server; echo "stopping: $C failed"; exit 1; }
    stop_server
done

if [ "$PREFIX_EXPERIMENT" = 1 ]; then
    for C in fp16_cacheoff fp16_cacheon awq_cacheoff awq_cacheon; do
        M=$(model_of $C)
        start_server $C "$M" "$(args_of $C)" || continue
        run_grid $C "$M" prefix || { stop_server; echo "stopping: $C failed"; exit 1; }
        stop_server
    done
fi
echo "done"
