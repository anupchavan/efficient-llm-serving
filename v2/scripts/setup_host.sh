#!/bin/bash
# One-time setup on the rented GPU host. Run from the repository root.
set -euo pipefail
python3 -m venv ~/venv && source ~/venv/bin/activate
pip install "vllm[bench]==0.30.0"   # same vLLM as the earlier sweeps; [bench] adds pandas etc. for vllm bench
nvidia-smi
mkdir -p v2/sources v2/prompts
HF=https://huggingface.co/datasets
curl -L -o v2/sources/sharegpt_v3.json \
    $HF/anon8231489123/ShareGPT_Vicuna_unfiltered/resolve/main/ShareGPT_V3_unfiltered_cleaned_split.json
curl -L -o v2/sources/role_prompts.csv $HF/fka/awesome-chatgpt-prompts/resolve/main/prompts.csv
head -c 300 v2/sources/role_prompts.csv; echo
# Download the models now, so server start-up time is not counted later.
for m in Qwen/Qwen3-14B Qwen/Qwen3-14B-AWQ Qwen/Qwen3-0.6B; do
    hf download $m > /dev/null || huggingface-cli download $m > /dev/null
done
python3 v2/scripts/prepare_prompts.py --tokenizer Qwen/Qwen3-14B \
    --sharegpt v2/sources/sharegpt_v3.json --role-prompts v2/sources/role_prompts.csv --out v2/prompts
