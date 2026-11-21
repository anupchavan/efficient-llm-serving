# Demo walkthrough

A screen recording of about 6 minutes. Everything shown runs on a laptop from a fresh clone.

The GPU benchmarks are not rerun in the demo. The 14B model needs an 80 GB GPU and the full sweep takes about 15 hours. The demo shows the recorded output of those runs (result files and logs in the repository) and reruns everything that comes after them: the checks, the tables and the tests.

## Before recording

```bash
git clone https://github.com/anupchavan/efficient-llm-serving
cd efficient-llm-serving
pip install -r requirements.txt
```

Open one terminal at the repository root with a font large enough to read in the video.

## 1. What the project is (30 s)

Show `README.md` at the status table.

Say: Project 24. We measured AWQ quantisation, speculative decoding and prefix caching on a vLLM server with one GPU, to find which of them to switch on at which load. The final results are from Qwen3-14B on an A100 with real ShareGPT prompts, 504 runs. FP8 KV cache and the AWQ quality check are not done.

## 2. How a run is produced (60 s)

```bash
cd evaluation/03_qwen3-14b_sharegpt
less scripts/run_sweep.sh
```

Show the `CONFIGS` list and the loop over workloads, rates and repeats.

Say: for each configuration the script starts `vllm serve` with one set of flags, waits for the health check, then calls `vllm bench serve` for every workload, rate and repeat. Each call sends 200 requests at Poisson arrival times and writes one JSON file.

Then show the recorded output of one benchmark call:

```bash
zcat logs/bench_awq.log.gz | sed -n '10,34p'
```

This is the first AWQ run (natural prompt mix, 1 request/s): 200 successful requests, 0 failed, median TPOT 8.01 ms.

## 3. One run, from raw file to summary row (60 s)

```bash
zcat data/raw/awq_len1024_rate1_rep1.json.gz | python3 -c "
import json, sys
d = json.load(sys.stdin)
print(d['config'], d['workload'], d['request_rate'], 'completed', d['completed'], 'failed', d['failed'])
print('median TPOT', round(d['median_tpot_ms'], 2), 'ms   median TTFT', round(d['median_ttft_ms'], 1), 'ms')
print('per-request TTFTs saved:', len(d['ttfts']))"
```

Expected output:

```
awq len1024 1.0 completed 200 failed 0
median TPOT 9.47 ms   median TTFT 179.7 ms
per-request TTFTs saved: 200
```

Say: the file holds the timing of every request, so the medians can be recomputed and goodput can be checked per request.

## 4. Checks and tables from the raw files (90 s)

```bash
python3 analysis/summarise.py
python3 analysis/goodput.py
python3 analysis/extract_per_run.py
python3 analysis/analyse.py --no-figures
git status --short data results
```

Expected: `504 runs, 0 flagged`, and `git status` prints nothing, because the regenerated summary and tables are identical to the committed ones. (`--no-figures` skips the PNGs, whose bytes depend on the installed fonts.)

Say: `summarise.py` recomputes each run's medians from the per-request timings and flags a run with a failed request or a mismatch. None of the 504 is flagged.

Open `results/tables.md` and show, in this order:

- Table 1, row `len1024`: FP16 21.81 ms, AWQ 9.45 ms (2.31×), speculative 14.91 ms, AWQ + speculative 13.83 ms. AWQ is fastest; adding speculative decoding to AWQ makes it slower.
- Table 2 (capacity), `len1024`: FP16 5.60 requests/s, AWQ 4.66. AWQ serves fewer requests under load.
- Table 5 (prefix caching), AWQ at 12 requests/s: TTFT 510 ms with caching off, 65 ms with caching on.

Open `figures/saturation.png`: requests finished against requests sent.

## 5. Speculative decoding acceptance, from the server log (30 s)

```bash
zcat logs/server_awq_spec.log.gz | grep -m3 "SpecDecoding metrics"
```

Say: vLLM logs how many drafted tokens the 14B model accepted. Across the sweep it is 34–39%, which is why speculative decoding gains less here than on the 3B model with synthetic prompts (58–69%).

## 6. The earlier studies and what was excluded (45 s)

```bash
cd ../02_qwen2.5-3b_lengths
python3 analysis/summarise.py
cut -d, -f1,5 data/summary.csv | sort | uniq -c
```

Open `README.md` at "Which runs are used".

Say: study 2 is the 864-run sweep on the lab GPU behind the progress presentation. We use 216 of the runs. All 432 FP8 KV-cache runs completed zero requests, so there is no FP8 result. Two configurations ran at the same time against one server, and the `baseline` configuration, with the same flags as `CR`, served 2.1 requests/s at 1,024 tokens where `CR` served 7.1. These are kept in the repository and marked excluded.

## 7. Tests (30 s)

```bash
cd ../..
pytest
```

Expected: `27 passed` in about 20 seconds.

Say: the tests rebuild the summaries and tables of all three studies from the raw files and compare them with the committed files, check each exclusion rule against the data, and check the headline numbers in the report.

## 8. Close (15 s)

Show `docs/` in the file browser: `Project_Report.pdf`, `Architecture_Decision_Records.md`, `Final_Presentation.pptx`.

Say: the report, the decision records and the slides are in `docs/`. What is not done is listed in section 8 of the report: FP8 KV cache, the AWQ quality check, and batching and scheduling left at vLLM's defaults.
