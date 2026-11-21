# Study 2: Qwen2.5-3B, six prompt lengths, shared prefixes

864 runs on the lab GPU, 15–26 October 2026: 16 configurations × 9 request rates × 6 prompt lengths, one run each. This is the data behind the progress presentation (`docs/progress/2026-10-31_progress_presentation.pptx`). Only 4 of the 16 configurations are used; the rest are kept in `data/raw` and marked excluded in `data/summary.csv`.

| | |
|---|---|
| GPU | NVIDIA RTX PRO 4000 Blackwell, 24 GB (shared lab server) |
| Server | vLLM 0.30, `--gpu-memory-utilization 0.95 --max-model-len 8192` |
| Models | Qwen2.5-3B-Instruct, Qwen2.5-3B-Instruct-AWQ, Qwen2.5-0.5B as draft (5 tokens) |
| Load | `vllm bench serve --dataset-name prefix_repetition`, 992 requests per run, 2–64 requests/s |
| Workload | 512, 1,024, 1,536, 2,048, 3,072 or 4,096 prompt tokens; the first half (at most 512 tokens) is one of 32 shared prefixes; 128 output tokens |

## Which runs are used

| Folder name in `data/raw` | Flags | Status |
|---|---|---|
| `CR` | `--enable-prefix-caching` | Used as **FP16**. Prefix caching is on by default in vLLM 0.30, so this is the default server. |
| `WQ` | AWQ | Used as **AWQ** |
| `MT` | speculative decoding | Used as **speculative** |
| `WQ_MT` | AWQ + speculative | Used as **AWQ + speculative** |
| `CR_MT` | caching flag + speculative | Valid, duplicate of `MT` (capacity within 3%) |
| `KV`, `KV_MT`, `KV_CR`, `KV_CR_MT`, `WQ_KV`, `WQ_KV_MT`, `WQ_KV_CR`, `WQ_KV_CR_MT` | `--kv-cache-dtype fp8` | **Excluded.** All 432 runs completed 0 requests. The FP8 server did not serve; the host has no CUDA compiler (`nvcc`), which the FP8 path needed. |
| `WQ_CR`, `WQ_CR_MT` | AWQ + caching (+ speculative) | **Excluded.** The two sweeps ran at the same time against one server (start times 1 s apart, no speculation statistics in the "speculative" one), so each saw the other's load. |
| `baseline` | none | **Excluded.** Same server flags as `CR`, run 15–16 October, but it served at most 2.1 requests/s at 1,024 tokens where `CR` served 7.1. We did not find the cause; the GPU is shared. One of its files is corrupt. |

Consequence: this sweep has no caching-off run, so it cannot measure the gain from prefix caching. The progress presentation estimated that gain by comparing this sweep with study 1 (capacity 7.1 against 4.0 requests/s at 1,024 tokens). That comparison is across two sweeps two weeks apart, and the excluded `baseline` run shows the same server could differ by more than that between days. Study 3 measures the caching gain with off and on in one sweep.

## Results

Generated tables: [`results/tables.md`](results/tables.md). Main points, all single runs:

- At 2 requests/s and 1,024 tokens, median TPOT is 22.2 ms (FP16), 8.4 ms (AWQ, 2.63×), 10.6 ms (speculative, 2.09×), 7.1 ms (AWQ + speculative, 3.14×).
- Capacity at 1,024 tokens is 7.1–7.3 requests/s for all four. From 1,536 tokens, the speculative configurations serve 21–27% more than FP16; at 512 tokens they serve 18% less.
- Without speculation, the server prefills 3.4–3.9k new prompt tokens/s at every length.

Study 3 did not reproduce the speculative-decoding results on a larger model (see its section 9).

## Reproduce

```bash
python3 analysis/summarise.py   # data/raw -> data/summary.csv, with the status column
python3 analysis/tables.py      # summary.csv -> results/tables.md
```

`scripts/run_sweep.sh` is the script as it was run on the lab server, unchanged. Its health check only waits for port 8000 to answer, so when a server failed to start the benchmark ran anyway and recorded failures; the study 3 script checks that the server process is alive.
