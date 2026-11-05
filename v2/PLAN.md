# Larger-model sweep with real prompts: plan

Status: plan and scripts ready, nothing rented or run yet. Prices were read on 2026-11-05 and change daily.

## 1. What we are repeating

The final presentation (16 slides, 2026-10-29) used Qwen2.5-3B-Instruct on one RTX PRO 4000 Blackwell (24 GB):
FP16, AWQ, speculative decoding (Qwen2.5-0.5B draft, 5 tokens) and AWQ + speculative, prompts of 512 to 4,096 tokens from vLLM's sonnet text, 2 to 64 requests/s, one run per setting, plus a prefix-sharing comparison.
The analyses were: light-load TPOT/TTFT, capacity (requests/s served), what sets the limit (prompt tokens/s), goodput within TTFT < 1 s and TPOT < 50 ms, the prefix-caching gain, and a "which setup when" table.

The new sweep produces the same analyses on a larger model with real prompts.

## 2. Scope check: multi-turn follow-up evaluation

Not in the plan. Project 24 in `Project-1-DAT.xlsx` (M1 optimised serving, M2 prefix reuse and speculative decoding, M3 attribution and operating regimes) and the evaluation-plan CSV never mention multi-turn or follow-up prompts. The only mention found is a cited reference in `llm-serving-report.pdf`. So it is a **deferred stretch goal**, written down in section 8 and not built now.

## 3. Things found while checking that change the plan

1. **Our recorded GPU is 24 GB, not 48 GB.** The README, the report and slide 7 all say RTX PRO 4000 Blackwell, 24 GB. Please confirm with Vishank. If a 48 GB card is available, option B below runs on it for free.
2. **Qwen2.5 cannot be kept for the larger model as-is.** Qwen2.5-7B, 14B and 32B have `vocab_size` 152064, while the 0.5B and 1.5B drafts have 151936. vLLM 0.30.0 refuses a draft model whose vocab size differs (`verify_equal_vocab_size_if_draft_model` in `vllm/config/speculative.py`). The only built-in workaround (`use_heterogeneous_vocab`) is a different algorithm that only supports greedy drafting, so it would not be the same technique we measured. Qwen3 has the same vocab size (151936) for the 0.6B draft and the 14B and 32B targets, and Qwen publishes official AWQ versions (`Qwen/Qwen3-14B-AWQ`, `Qwen/Qwen3-32B-AWQ`). So the plan moves to **Qwen3**. This was checked in the vLLM source and the Hugging Face configs, not yet on a GPU; the pilot run confirms it.
3. **Prefix caching was on in every earlier run** (vLLM's default). With real prompts, repeated or similar prompts would get free cache hits and mix into every result. In the new sweep it is off for the four main configurations, and the prefix experiment runs the same workload with caching off and on in one sweep. That also fixes the old limitation "prefix-caching gain compares two sweeps".
4. **vLLM's built-in ShareGPT loader drops every prompt over 1,024 tokens** (`is_valid_sequence` in `vllm/benchmarks/datasets/datasets.py`) and does not apply the chat template. So we build our own prompt files and use the `custom` dataset.

## 4. Model and GPU options

Memory figures are my estimates from the model configs (weights in BF16 = 2 bytes per parameter; KV cache per token = 2 × layers × KV heads × head size × 2 bytes) and assume `--gpu-memory-utilization 0.95`, about 3.5 GB for vLLM overhead and 1.5 GB for the draft weights. vLLM logs the real KV-cache size at start, and the script saves it with the data.

| | A. Qwen3-14B on A100 80GB (recommended) | B. Qwen3-14B on a 48 GB card (L40S / RTX 6000 Ada) | C. Qwen3-32B on H100 80GB |
|---|---|---|---|
| Weights FP16 / AWQ | 29.6 GB / about 10 GB | same | 65.6 GB / about 20 GB |
| KV cache per token (target + 0.6B draft) | 160 KiB + 112 KiB | same | 256 KiB + 112 KiB |
| KV room left, FP16 + spec (estimate) | about 47 GB, roughly 40 requests of 4,224 tokens at once | about 11 GB, roughly 9–10 requests | about 11 GB, roughly 7 requests |
| Memory bandwidth (spec sheet) | 1.9–2.0 TB/s | 0.86 TB/s (L40S) | 3.35 TB/s (SXM) |
| Listed price, 1 GPU | Vast.ai about $1.00–1.18/h; TensorDock $1.63/h; Lambda $2.79/h | Vast.ai L40S $0.73/h, RTX 6000 Ada $0.60/h; Massed Compute L40S $0.97/h | Vast.ai about $2.27/h; Lambda $3.99/h |
| Verdict | Fits all four setups with room, so FP16 is not starved of memory and comparisons stay fair | Fits, but FP16 + spec has little KV room at 4,096 tokens, and decode can be up to about 2.3× slower (bandwidth ratio), so runs take longer; free if we own a 48 GB card | FP16 weights take about 80% of memory, so the FP16 baseline is limited by memory more than the 14B case, and it costs about twice as much per hour |

Price sources: [computeprices.com Vast.ai page](https://computeprices.com/providers/vast) (collected 2026-11-05), [gpuperhour.com](https://gpuperhour.com/) (2026-11-05 11:30 UTC), [TensorDock A100 page](https://www.tensordock.com/gpu-a100.html), [Lambda pricing](https://lambda.ai/pricing) (whether Lambda rents a single GPU at that price is not stated). I could not query the Vast.ai or Hugging Face APIs from my machine, so these are listed prices from those pages; the price shown in the Vast.ai console when renting is what counts. Vast.ai is a marketplace: pick a "verified" datacenter host with fast disk and enough download bandwidth (about 41 GB of model files), and note it charges for disk while an instance is stopped.

**Recommendation: A.** Qwen3-14B (about 4.8× the parameters of the 3B), Qwen3-14B-AWQ, Qwen3-0.6B as the draft, one A100 80GB on Vast.ai.

Optional add-on: run Qwen3-4B on the same GPU with the same prompts (short, cheap). Then "small vs large" is the same family, GPU and prompts, which answers the presentation feedback directly. I have not checked that an official Qwen3-4B-AWQ exists.

## 5. Workloads (real prompts)

Built by `v2/scripts/prepare_prompts.py` on the GPU host:

- **User prompts:** the first user turn of each ShareGPT V3 conversation (public), exact duplicates removed, chat template applied with thinking off.
- **Length buckets:** `len512`, `len1024`, `len2048`, `len4096` hold prompts whose real length is within ±20% of the target. Real prompts are not cut to an exact length, so the length axis is bucketed. The script reports how many prompts each bucket has; long real prompts may be scarce, and if `len4096` has fewer than 500 we decide then (widen the bucket, or add a public long-document set).
- **Natural mix:** all prompts up to 4,096 tokens, in their real length mix.
- **Shared prefixes:** one of 32 community-written role prompts (awesome-chatgpt-prompts, public) as the system message, then a user prompt. The script reports the real shared fraction.
- Output fixed at 128 tokens (`--ignore-eos`), as in the earlier sweeps, so TPOT compares cleanly. Sampling set explicitly to Qwen's non-thinking defaults (temperature 0.7, top-p 0.8, top-k 20).

## 6. Run design

- Configurations: FP16, AWQ, speculative, AWQ + speculative, all with prefix caching off; then FP16 and AWQ with caching off vs on, on the shared-prefix workload. With caching on, every run gets user prompts that no earlier run on that server used, so only the system prompts can be reused.
- **3 repeats per setting** (the final presentation had one run per setting), each repeat a different sample of prompts.
- 500 requests per run. Request rates are set **after the pilot**, around the measured capacity (the 3B rates of 2–64 would mostly be far past a 14B model's limit).
- Every run saves per-request data. `v2/analysis/summarise.py` recomputes median TTFT and TPOT from it, checks for failed requests and wrong output lengths, and flags any run that does not match. No number goes into slides from a flagged run.

## 7. Cost and steps

1. **Pilot** (`PILOT=1`): FP16 and AWQ + speculative, 1,024 and 4,096 tokens, 4 rates, 200 requests. About 1–2 hours including downloads, about $2. It confirms the Qwen3 draft works, measures capacity, and gives the real time per run.
2. **Full sweep:** my rough guess is 30–60 GPU-hours for 3 repeats (about $35–70 at $1.0–1.2/h). That rests on an assumed capacity of 1–3 requests/s, which I have not measured, so treat it as a guess until the pilot. Ways to cut it: 2 repeats, fewer overloaded rates, or dropping one length bucket.
3. Same analyses and figures as the final presentation, from `summary.csv`, with spread across repeats shown.

## 8. Deferred stretch goal: multi-turn follow-ups

Measure a conversation, not just the first reply: send a prompt, take the model's answer, send a follow-up built from it, and so on. This tests prefix caching on the growing history and speculative decoding on later turns. vLLM 0.30.0 ships a tool for this (`benchmarks/multi_turn/benchmark_serving_multi_turn.py`), and the saved `generated_texts` can seed follow-ups. Build only if time permits after the main sweep.

## 9. Decisions needed before spending

1. Host and GPU (recommended: Vast.ai A100 80GB), or a 48 GB card if we have one.
2. Model family switch to Qwen3 (recommended: Qwen3-14B + 0.6B draft), and whether to add the Qwen3-4B control.
3. Budget cap (pilot about $2; full sweep guessed $35–70).
