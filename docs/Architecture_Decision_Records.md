# Architecture Decision Records

CS5903 Distributed AI Training, Fall 2026. Group 4, Project 24: Efficient LLM Serving.
Vishank Singh (AI23BTECH11032), Anup Chavan (CS23BTECH11005).

All records were written on 21 November 2026. Each record gives the date the decision was taken, which is earlier. Where no dated document from that time exists, the record says what the earliest dated evidence is.

Paths are relative to the repository root. "Study 1/2/3" are the folders under `evaluation/`.

| ID | Decision | Decided | Status |
|---|---|---|---|
| ADR-001 | Measure on vLLM instead of building a server | September 2026 | Accepted |
| ADR-002 | Methodology reaffirmed at the 9 October checkpoint | 9 October 2026 | Accepted, later extended by ADR-006 |
| ADR-003 | Open-loop load at fixed request rates, fixed output length | October 2026 | Accepted |
| ADR-004 | Synthetic prompts for the 3B studies | October 2026 | Superseded by ADR-006 |
| ADR-005 | Use 4 of the 16 configurations of study 2 | 29 October 2026 | Accepted |
| ADR-006 | Repeat the study on Qwen3-14B with real prompts | 5 November 2026 | Accepted |
| ADR-007 | Prefix caching off by default, on only in a paired experiment | 5 November 2026 | Accepted |
| ADR-008 | Three repeats and per-request verification of every run | 5 November 2026 | Accepted |
| ADR-009 | Goodput target: TTFT under 1 s and TPOT under 50 ms per request | 29 October 2026, revised 9 November | Accepted |

---

## ADR-001 — Measure on vLLM instead of building a server

**Date, status, scope.** Decided in the design phase, September 2026. Accepted. Affects M1, M2, M3 and every script in `evaluation/`.

**Context and decision drivers.** The project asks which infrastructure-level optimisations improve the latency and throughput of a served open-weight model, how they interact, and in which regimes each one pays. That is a measurement question. The techniques themselves (weight quantisation, reduced-precision KV cache, prefix reuse, speculative decoding, continuous batching) already exist in open-source serving engines. We needed all of them switchable in one engine, so that a difference between two runs comes from one flag.

**Options considered.**

| Option | For | Against |
|---|---|---|
| vLLM | AWQ, FP8 KV cache, prefix caching and draft-model speculative decoding are all server flags; ships a load generator (`vllm bench serve`) that reports TTFT and TPOT | Results are specific to vLLM's scheduler and kernels |
| SGLang | Prefix reuse is its main design point | We did not confirm that it offers all four techniques as independent switches; not benchmarked |
| Write our own serving loop | Full control over batching and scheduling | Two people, ten weeks; the result would measure our implementation, not the techniques |

**Decision and rationale.** vLLM. This is a judgment about fit, not a measured comparison of engines: we did not benchmark SGLang or any other engine.

**Consequences and trade-offs.** We wrote no serving code. Our code is the benchmark harness, the prompt builder, the run checks and the analysis. Tuned batching, prefill scheduling and parallelism strategy (named in M1) were left at vLLM's defaults and were not studied. Every finding holds for vLLM 0.30.0 and may not hold for another engine or version.

**Evidence and validation.** `evaluation/*/scripts/` start `vllm serve` with one set of flags per configuration. Server start-up records: `evaluation/03_qwen3-14b_sharegpt/data/server_*.txt`.

**Revisit trigger.** A required technique is missing or broken in vLLM (this happened for FP8 KV cache on the lab host, see ADR-005), or a question needs control over the scheduler that flags do not give.

---

## ADR-002 — Methodology reaffirmed at the 9 October checkpoint

**Date, status, scope.** Checkpoint: 9 October 2026. This record was written on 21 November 2026. Accepted. Affects all milestones.

**Context.** The evaluation framework asks each team to state by 9 October whether the proposed approach stands or changes.

**Options considered.** (a) Keep the proposed approach: a benchmark study on a single GPU with vLLM, switching AWQ, FP8 KV cache, prefix caching and speculative decoding on and off across request rates and prompt lengths. (b) Change to building a custom scheduler or an adaptive controller that picks a configuration at run time.

**Decision and rationale.** Reaffirmed, no change. The approach in use on 9 October was the proposed one, and the first results gave no reason to change it: the pipeline ran end to end and separated the configurations clearly at light load (AWQ 2.5× lower TPOT). Option (b) needs the regime map as input, so it could only follow this study.

**Consequences.** Milestones M1–M3 unchanged. The approach was extended on 5 November (larger model, real prompts, repeats); that change is ADR-006.

**Evidence and validation.** We have no document written on 9 October. The earliest dated evidence is the study 1 result files, which carry timestamps from 7 to 10 October 2026 (`date` field in `evaluation/01_qwen2.5-3b_1024tok/data/raw/*.json`), and the progress report of 15 October, which describes the same approach (`docs/progress/2026-10-15_progress_report.pdf`).

**Revisit trigger.** Results on the small model that do not separate the configurations, or evaluator feedback that the method is insufficient. The second occurred at the progress presentation (ADR-006).

---

## ADR-003 — Open-loop load at fixed request rates, fixed output length

**Date, status, scope.** October 2026, before study 1. Accepted. Affects all three studies.

**Context and decision drivers.** M2 asks for the load level beyond which speculative decoding stops helping, and M3 for regimes by concurrency. We needed load as an independent variable that the server cannot influence, and a TPOT that is comparable across configurations.

**Options considered.**

| Option | Behaviour |
|---|---|
| Open loop: requests arrive at a set average rate with Poisson gaps | Load does not slow down when the server does. Shows the saturation point and queue growth. |
| Closed loop: fixed number of concurrent clients | Load adapts to the server, so a slower configuration receives fewer requests. Never overloads; hides the capacity limit. |

For output length: fixed 128 tokens with `--ignore-eos`, or let the model stop on its own.

**Decision and rationale.** Open loop, `vllm bench serve --request-rate`, 128 output tokens forced. With a fixed output length every request does the same decode work in every configuration, so TPOT differences come from the configuration and not from answer length. AWQ changes the weights and could change answer length if the model were allowed to stop.

**Consequences and trade-offs.** Above capacity the queue grows for the whole run, so TTFT in overloaded runs depends on run length and is not a steady-state value. In study 3 a run at 12 requests/s lasts about 17 s of arrivals. Real answer-length variation is not represented. Concurrency is an outcome (`max_concurrent_requests`), not a controlled variable.

**Evidence and validation.** `evaluation/03_qwen3-14b_sharegpt/scripts/run_sweep.sh` (`--request-rate`, `--custom-output-len 128 --ignore-eos`); `tests/test_study3.py::test_every_run_is_complete` confirms 128 output tokens per request through the run checks.

**Revisit trigger.** A question about steady-state behaviour near capacity, which needs longer runs or fixed concurrency.

---

## ADR-004 — Synthetic prompts for the 3B studies

**Date, status, scope.** October 2026. Superseded by ADR-006 on 5 November 2026. Affects studies 1 and 2.

**Context.** The first studies needed exact control of prompt length and of how much of a prompt is shared.

**Options considered.** (a) vLLM's built-in synthetic datasets: `random` (random tokens of an exact length) and `prefix_repetition` (a shared prefix of set length plus a unique suffix). (b) A real chat dataset.

**Decision and rationale.** (a). Exact lengths made the length axis clean, and `prefix_repetition` let us set the shared fraction to one half.

**Consequences and trade-offs.** Speculative decoding's acceptance rate depends on how predictable the text is. On synthetic text it was 52–69%; on real prompts in study 3 it was 34–39%. The speculative-decoding gains measured in studies 1 and 2 therefore do not carry over to real prompts. This was the main weakness raised at the progress presentation.

**Evidence and validation.** Acceptance rates: `evaluation/02_qwen2.5-3b_lengths/results/tables.md` and section 6 of `evaluation/03_qwen3-14b_sharegpt/README.md`.

**Revisit trigger.** Any result that depends on text content, which speculative decoding does. Triggered; see ADR-006.

---

## ADR-005 — Use 4 of the 16 configurations of study 2

**Date, status, scope.** 29 October 2026, when the 864 runs were analysed. Accepted. Affects M1 (FP8 KV cache), M2 (prefix caching) and every number quoted from study 2.

**Context.** Study 2 ran 16 configurations. On inspection, three groups could not be compared fairly:

1. All 8 configurations with `--kv-cache-dtype fp8` completed 0 of 53,568 requests each. The FP8 server did not serve on the lab host, which has no CUDA compiler.
2. `WQ_CR` and `WQ_CR_MT` (AWQ with the caching flag, without and with speculation) were benchmarked at the same time against one server. Their start times are 1 s apart and the run labelled speculative has no speculation statistics.
3. `baseline` has the same server flags as `CR` (prefix caching is on by default), but served at most 2.1 requests/s at 1,024 tokens where `CR` served 7.1. The two were run six days apart on a shared GPU.

**Options considered.** (a) Report all runs as recorded. This would show `baseline` against `CR` as a 3.4× gain from prefix caching, which the flags do not support. (b) Use only the configurations that are complete and were run alone: `CR` as FP16, `WQ`, `MT`, `WQ_MT`. (c) Rerun the excluded configurations.

**Decision and rationale.** (b). Every number quoted from study 2 comes from these four. (c) was not done: within a week the work moved to the larger model (ADR-006), where caching off and on were measured in one sweep.

**Consequences and trade-offs.** No FP8 KV-cache result exists for any model in this project. Study 2 has no caching-off run, so it gives no prefix-caching gain. The cause of the slow `baseline` run was not found. The excluded files stay in `data/raw`, and `data/summary.csv` carries a `status` column with the reason.

**Evidence and validation.** `evaluation/02_qwen2.5-3b_lengths/README.md`; `tests/test_study2.py` checks each of the three observations against the raw files.

**Revisit trigger.** FP8 becomes runnable (a host with the CUDA toolkit), or the 3B numbers are needed for a claim that the larger-model study does not cover.

---

## ADR-006 — Repeat the study on Qwen3-14B with real prompts

**Date, status, scope.** 5 November 2026. Accepted. This is a methodology change made after the 9 October checkpoint. It extends ADR-002 and supersedes ADR-004. Affects M1–M3; the final results come from this study.

**Old approach.** Qwen2.5-3B on the 24 GB lab GPU, synthetic prompts, one run per setting (studies 1 and 2).

**New evidence.** At the progress presentation (deck dated 31 October; the feedback was acted on from 5 November) the evaluator's feedback was that effects and interactions measured on a 3B model are too noisy to trust, and that the analysis should be repeated on a larger model. The change was requested verbally; there is no written approval in the repository. Our own review of study 2 added three problems: synthetic prompts (ADR-004), one run per setting, and prefix caching on in every run.

**Options considered.**

| Option | Assessment |
|---|---|
| Keep the 3B model, add repeats | Does not answer the feedback |
| Qwen2.5-7B or 14B with the Qwen2.5-0.5B draft | Not possible in vLLM 0.30.0: the larger Qwen2.5 models have vocabulary size 152,064 and the draft has 151,936, and vLLM rejects a draft with a different vocabulary size |
| Qwen3-14B, Qwen3-14B-AWQ, Qwen3-0.6B draft on one A100 80 GB | Same vocabulary across target and draft; official AWQ weights exist; all four configurations fit with KV-cache room left |
| Qwen3-14B on a 48 GB GPU | Fits, but FP16 with a draft model leaves little KV-cache memory at 4,096 tokens, which would handicap that configuration |
| Qwen3-32B on an H100 80 GB | FP16 weights take about 80% of memory, so the FP16 baseline would be memory-starved |

**Decision and rationale.** Qwen3-14B on one A100 80 GB with real prompts from ShareGPT. The memory estimates behind the table were calculated from the model configurations before the run; the KV-cache sizes vLLM reported at start-up are in `data/server_*.txt`. A 16-run pilot on 5 November confirmed that the Qwen3-0.6B draft loads and set the request-rate grid.

**Consequences and trade-offs.** The comparison between the 3B and 14B results is confounded: model family, size, GPU, prompts, repeats, requests per run and prefix caching all changed together. Differences between the studies cannot be attributed to model size alone. The lab GPU cannot hold the 14B model, so the study ran on a rented GPU and cannot be rerun on lab hardware. Real prompts are bucketed by length (within ±20% of the target), so the length axis is no longer exact. Milestones did not change; the evidence for them did.

**Evidence and validation.** `evaluation/03_qwen3-14b_sharegpt/` (scripts, 504 raw runs, README). Pilot output: `data/pilot/`.

**Revisit trigger.** A control run of a small Qwen3 model on the same GPU and prompts, which would separate model size from the other changes.

---

## ADR-007 — Prefix caching off by default, on only in a paired experiment

**Date, status, scope.** 5 November 2026. Accepted. Affects M2 and all capacity numbers in study 3.

**Context.** vLLM 0.30 enables prefix caching by default. In studies 1 and 2 it was on in every run. With the half-shared prompts of study 2, every capacity number included cache hits, and there was no caching-off run to compare with.

**Options considered.** (a) Leave the default on everywhere. (b) Turn it off for the four main configurations and measure it separately: the same shared-prefix workload with caching off and on, in one sweep on one server build.

**Decision and rationale.** (b). The main configurations run with `--no-enable-prefix-caching`. The prefix experiment uses one of 32 role prompts as the system message followed by a real user prompt. In the caching-on runs, each run gets user prompts that no earlier run on that server has sent, so only the role prompts can be served from cache.

**Consequences and trade-offs.** The main results describe a server with caching off, which is not vLLM's default; a default deployment with repeated prompts would do better. The caching experiment covers FP16 and AWQ only. Caching combined with speculative decoding was not run on the 14B model.

**Evidence and validation.** `scripts/run_sweep.sh` (`args_of`, `prefix_slice`); `tests/test_scripts.py` checks the flags in the script and in the recorded server start-up lines. Result: section 5 of the study 3 README.

**Revisit trigger.** A workload where prompts repeat across requests in ways other than a shared system prompt (multi-turn conversations).

---

## ADR-008 — Three repeats and per-request verification of every run

**Date, status, scope.** 5 November 2026. Accepted. Affects study 3.

**Context.** Study 2 had one run per setting, so its run-to-run spread is unknown. Its script waited only for port 8000 to answer, so when a server failed to start, benchmarks still ran and recorded failures (ADR-005). Study 1 did not save per-request data, so its reported medians cannot be checked.

**Options considered.** Repeats: 1, 2 or 3 per setting, against GPU time. Requests per run: about 1,000 as before, or 200. Verification: trust the summary that `vllm bench` prints, or recompute from per-request timings.

**Decision and rationale.** 3 repeats, each with a different sample of prompts (`--seed` = repeat number). 200 requests per run, because only 215 distinct real prompts fall in the 4,096-token bucket and a run should not repeat a prompt. Every run saves per-request timings (`--save-detailed`). `analysis/summarise.py` recomputes median TTFT and TPOT, checks that all 200 requests completed with 128 output tokens, and flags any run that fails; the analysis refuses to run if a flagged run is present. The sweep script checks that the server process is alive and stops at the first failed run.

**Consequences and trade-offs.** 200 requests per run is short: at 12 requests/s the arrivals span about 17 s, so overloaded runs show a queue building, not a steady state. In return the spread is measured: the median spread of TPOT across repeats is 3.1%, and the large spreads are all at rates near a configuration's capacity. One run lost a request to a dropped connection, was flagged, and was rerun.

**Evidence and validation.** `data/checks.txt` ("all runs passed"); `tests/test_study3.py::test_broken_run_is_flagged`; section 7 of the study 3 README.

**Revisit trigger.** A claim that depends on a difference smaller than the measured spread, or on steady-state behaviour near capacity.

---

## ADR-009 — Goodput target: TTFT under 1 s and TPOT under 50 ms per request

**Date, status, scope.** First set on 29 October 2026 for the progress presentation; revised on 9 November 2026. Accepted. Affects M1 ("improvement at a stated latency target").

**Context.** M1 asks for an improvement over the default deployment at a stated latency target. The project statement does not give the target.

**Options considered.** The threshold values, and how to apply them: (a) per run, using the run's median TTFT and TPOT, as in the progress presentation; (b) per request, counting the requests that meet both limits.

**Decision and rationale.** TTFT under 1 s and TPOT under 50 ms (20 tokens/s, faster than reading speed). These values are our choice, a judgment about interactive use, not derived from data. From study 3 the target is applied per request (b): goodput = requests meeting both limits / run duration. A run whose median meets the target can still have close to half its requests outside it, which (a) hides.

**Consequences and trade-offs.** Goodput numbers from the progress presentation (per run) and from study 3 (per request) are not comparable. The request-rate grid is coarse, so the best goodput is known only at grid points. A different target would move the numbers; the ranking of configurations at long prompts follows capacity and is less sensitive to it.

**Evidence and validation.** `analysis/goodput.py`, `data/goodput.csv`, section 4 of the study 3 README.

**Revisit trigger.** A deployment with a stated service-level objective.
