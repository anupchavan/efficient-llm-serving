#set document(title: "Efficient LLM Serving on a Single GPU: Project Report", author: ("Vishank Singh", "Anup Chavan"))
#set page(paper: "a4", margin: (x: 2.1cm, top: 2.0cm, bottom: 2.1cm), numbering: "1", number-align: center)
#set text(font: "Libertinus Serif", size: 10.2pt, lang: "en")
#set par(justify: true, leading: 0.58em, spacing: 0.85em)
#set heading(numbering: none)
#show heading.where(level: 1): it => block(above: 1.3em, below: 0.7em, text(size: 13pt, weight: "bold", it.body))
#show heading.where(level: 2): it => block(above: 1.0em, below: 0.5em, text(size: 10.6pt, weight: "bold", it.body))
#show link: it => text(fill: rgb("#1a4f8b"), it)
#show raw: set text(font: "DejaVu Sans Mono", size: 8.2pt)
#set table(stroke: none, inset: (x: 5pt, y: 3.4pt))
#show table: set text(size: 8.8pt)
#show table: set par(justify: false, leading: 0.45em)
#show figure.caption: set text(size: 8.8pt)
#set figure(gap: 0.6em)
#show figure: set block(above: 1.0em, below: 1.0em)

#let repo = "https://github.com/anupchavan/efficient-llm-serving"
#let blob(path, label) = link(repo + "/blob/cs5903-final-2026/" + path, label)
#let tree(path, label) = link(repo + "/tree/cs5903-final-2026/" + path, label)
#let tbl(cols, align: left, ..cells) = table(
  columns: cols, align: align,
  table.hline(stroke: 0.7pt),
  ..cells,
  table.hline(stroke: 0.7pt),
)
#let hd(..c) = table.header(..c.pos().map(x => strong(x)), table.hline(stroke: 0.4pt))
#let s3 = "evaluation/03_qwen3-14b_sharegpt/"

#align(center)[
  #text(size: 17pt, weight: "bold")[Efficient LLM Serving on a Single GPU]
  #v(-0.3em)
  #text(size: 11.5pt)[Project Report · CS5903 Distributed AI Training, Fall 2026]
  #v(0.2em)
  Project 24 · Group 4 \
  Vishank Singh (AI23BTECH11032) · Anup Chavan (CS23BTECH11005) \
  #v(0.1em)
  Repository: #link(repo) \
  Final revision: tag `cs5903-final-2026` \
  Code, data and documents are as of commit `197ff656722a328ce460d3269ca4605fd2ec9867`. \
  The commits after it, up to the tag, change only these lines. \
  21 November 2026
]

= Executive summary

*Problem.* A serving engine offers several optimisations for running an open-weight LLM on one GPU. They interact, and each helps only under some loads and prompt lengths, so an operator cannot tell which to enable.

*Delivered.* A benchmark harness around vLLM 0.30.0 with run checks and analysis, and three studies: 201 and 864 runs on Qwen2.5-3B (24 GB GPU), and 504 runs on Qwen3-14B (A100 80 GB) with real prompts and three repeats.

*What works today.* The harness, checks and analysis run end to end. Every table regenerates from the committed raw files; 27 automated tests check this and the headline numbers quoted here. On the 14B model: AWQ lowers time per output token 2.3× for prompts up to about 1,000 tokens, but serves 17% fewer requests per second from 1,024 tokens. Speculative decoding with a 0.6B draft is slower than AWQ alone and lowers capacity. Prefix caching on shared system prompts cut time to first token from 510 to 65 ms near capacity. Prompt processing sets the capacity limit.

*Completion.* M2 and M3 are met with the gaps listed in section 8. M1 is partly met: FP8 KV cache, tuning of batching and scheduling, and a check of the AWQ model's output quality were not done.

= 1. Problem, scope and success criteria

*Actors.* The operator of an LLM endpoint on a single GPU, who chooses the server configuration. The endpoint's users, who experience time to first token (TTFT) and time per output token (TPOT).

*Problem statement.* For a vLLM server on one GPU, determine which of weight quantisation, reduced-precision KV cache, prefix caching and speculative decoding improve latency and throughput over the default deployment, how much each contributes when combined, and under which request rates and prompt lengths each stops helping.

#tbl((1fr, 1fr),
  hd([In scope], [Out of scope]),
  [One GPU, one model replica], [Multi-GPU serving, sharding against replication],
  [Techniques available as vLLM server flags], [Changes to vLLM's scheduler or kernels],
  [First-turn chat prompts, 128 output tokens], [Multi-turn conversations, variable answer length],
  [Latency and throughput], [Output quality (required by M1; not measured, see section 8)],
)

*Assumptions and constraints.* The lab GPU (RTX PRO 4000 Blackwell, 24 GB) is shared and cannot hold a 14B model in 16-bit weights, so the final study ran on one rented A100 80 GB. Findings are specific to vLLM 0.30.0, these models and these GPUs.

*Approved milestones* (from the project sheet) and the 'done when' condition we read from each:

#tbl((auto, 1.25fr, 1fr),
  hd([ID], [Milestone], [Done when]),
  [M1], [Optimised single-instance serving: quantised weights, reduced-precision KV cache, tuned batching and prefill scheduling, a parallelism strategy; improvement over the default deployment at a stated latency target, with output quality preserved], [Goodput at the latency target is higher than default vLLM, and output quality is verified],
  [M2], [Context reuse and multi-token generation: prefix caching and speculative decoding; the conditions under which each contributes, including the load beyond which speculative decoding stops helping], [Caching gain measured with caching off and on; request rate found where speculative TPOT is no longer below the baseline],
  [M3], [Attribution and operating regimes: contribution of each technique when combined, the binding constraint, and the regimes where the chosen configuration is and is not right], [Factorial comparison of the techniques; limiting resource identified; a table of regimes],
)

*Acceptance tests.* No acceptance tests were frozen in writing with the mentor. We use the milestone text above as the criterion, and section 8 states for each clause whether it was met. The repository's tests are listed in section 4.

*Target metrics.* TTFT under 1 s and TPOT under 50 ms for each request. Goodput is the number of requests per second that meet both. The project statement does not give a target; these values are our choice (#blob("docs/Architecture_Decision_Records.md", "ADR-009")).

= 2. Related work, baselines and alternatives

#tbl((auto, 1fr),
  hd([Work], [Relevance here]),
  [vLLM, PagedAttention [1]], [Stores the KV cache in fixed-size blocks so that many requests share GPU memory; the engine we measure],
  [Sarathi-Serve [2]], [Shows that long prefills stall ongoing decodes and proposes chunked prefill; vLLM's chunked prefill was on in study 3 (server log), and prefill is the limit we observe],
  [SGLang [3]], [Reuses the KV cache of shared prompt prefixes through a radix tree; the same idea as vLLM's prefix caching],
  [Speculative decoding [4]], [A small draft model proposes several tokens and the target verifies them in one pass; output distribution unchanged],
  [AWQ [5]], [4-bit weight quantisation that protects weights with large activations; we use published AWQ checkpoints],
)

*Approaches considered.* (A) Measure the techniques in an existing engine, switching one flag at a time. (B) Build a scheduler or controller that selects a configuration at run time. We chose A: B needs the regime map as its input, and with two people a custom engine would have measured our implementation and not the techniques (#blob("docs/Architecture_Decision_Records.md", "ADR-001")). Within A, vLLM was chosen over SGLang because all four techniques are independent server flags in vLLM; we did not benchmark SGLang.

*Reused components.*

#tbl((auto, auto, auto, 1fr),
  hd([Component], [Version], [Licence], [Use]),
  [vLLM, with `vllm bench serve`], [0.30.0], [Apache-2.0], [Server and load generator],
  [Qwen3-14B, Qwen3-14B-AWQ, Qwen3-0.6B], [Hugging Face `main`, Nov 2026], [Apache-2.0], [Target, quantised target, draft (study 3)],
  [Qwen2.5-3B-Instruct and its AWQ version], [Hugging Face `main`, Oct 2026], [Qwen Research Licence], [Target (studies 1, 2); research use],
  [Qwen2.5-0.5B], [same], [Apache-2.0], [Draft (studies 1, 2)],
  [ShareGPT V3 (unfiltered, cleaned split)], [SHA-256 in `prompts/stats.json`], [Apache-2.0], [User prompts (study 3)],
  [awesome-chatgpt-prompts], [same], [CC0-1.0], [Shared system prompts (study 3)],
)

Model revisions were not pinned to a commit hash. Prompt files are built on the benchmark host and are not committed; the raw result files contain the generated answers but not the prompts.

*What we added.* The sweep scripts, the prompt builder (length buckets, natural mix, shared-prefix workload with unique user prompts per cache-on run), per-run checks that recompute medians from per-request timings, per-request goodput, the analysis and figures, and the tests.

*Working baseline.* Default vLLM serving the 16-bit model: `vllm serve <model> --gpu-memory-utilization 0.95 --max-model-len 8192`. In study 3 prefix caching is also turned off in the baseline and in every main configuration, so that it can be measured separately (#blob("docs/Architecture_Decision_Records.md", "ADR-007")).

= 3. Architecture and implementation

#figure(
  {
    set text(size: 8.4pt)
    let b(body, w: 3.5cm, fill: rgb("#f3f1ea")) = box(width: w, inset: 5pt, radius: 2pt, stroke: 0.5pt + rgb("#888"), fill: fill, align(center, body))
    let ar = box(width: 0.55cm, align(center, sym.arrow.r))
    grid(columns: 1, row-gutter: 6pt, align: center,
      grid(columns: 7, align: horizon,
        b[*prepare\_prompts.py* \ ShareGPT, role prompts → `prompts/*.jsonl`], ar,
        b(fill: rgb("#e6eef7"))[*vllm bench serve* \ 200 requests, Poisson arrivals, one rate], ar,
        b(fill: rgb("#e6eef7"))[*vllm serve* \ one configuration: weights, draft, caching], ar,
        b[#strong("data/raw/") \ one JSON per run, per-request timings],
      ),
      text(size: 8pt)[`run_sweep.sh` starts and stops the server and loops over repeats, configurations, workloads and rates],
      grid(columns: 7, align: horizon,
        b[*summarise.py* \ recompute medians, flag bad runs → `summary.csv`], ar,
        b[*goodput.py*, *extract\_per\_run.py* \ per-request targets, percentiles], ar,
        b[*analyse.py* \ mean and range over repeats], ar,
        b[`results/tables.md`, `figures/` \ `tests/` compare with committed files],
      ),
    )
  },
  caption: [Components and data flow of study 3. Blue boxes are vLLM; the rest is our code. Server and load generator run on the same host and talk over `localhost:8000`.],
)

#tbl((auto, 1fr, auto),
  hd([Component], [Responsibility and interface], [Source]),
  [Prompt builder], [Reads ShareGPT and the role prompts, applies the chat template (thinking off), counts tokens with the model's tokenizer, writes one JSONL file per workload and `stats.json` with counts, length percentiles and source hashes], [#blob(s3 + "scripts/prepare_prompts.py", "prepare_prompts.py")],
  [Sweep script], [Maps a configuration name to server flags, starts `vllm serve`, waits for `/health` while checking the process is alive, records what the server reported at start-up, runs the grid, stops at the first failed run; finished runs are skipped on restart], [#blob(s3 + "scripts/run_sweep.sh", "run_sweep.sh")],
  [Run checks], [For each run: all 200 requests completed, 128 output tokens each, median TTFT and TPOT recomputed from per-request data equal those vLLM reported; writes `summary.csv` and `checks.txt`], [#blob(s3 + "analysis/summarise.py", "summarise.py")],
  [Goodput], [Counts requests with TTFT < 1 s and TPOT < 50 ms, divides by run duration], [#blob(s3 + "analysis/goodput.py", "goodput.py")],
  [Analysis], [Refuses to run if any run is flagged; mean, minimum and maximum over the three repeats; writes every table and figure], [#blob(s3 + "analysis/analyse.py", "analyse.py")],
)

*Data.* One file per run, `<config>_<workload>_rate<R>_rep<N>.json.gz`, as written by `vllm bench serve --save-detailed`: run totals, and per request the prompt length, TTFT, inter-token latencies, total latency and generated text. There is no database. Derived files are CSV.

*External dependencies.* vLLM and its Python dependencies on the benchmark host; Hugging Face for model and dataset downloads. The analysis needs only pandas and matplotlib.

*Trust boundaries.* The server listens on localhost and receives only our benchmark traffic. No credentials are needed or stored. ShareGPT is a public dataset of user-shared conversations; we do not redistribute its prompts.

Decision records: #blob("docs/Architecture_Decision_Records.md", "docs/Architecture_Decision_Records.md"). Source folders: #tree("evaluation", "evaluation/"), #tree("tests", "tests/").

= 4. Methodology and experiment setup

*Evaluation questions.*
+ How much does each technique lower TPOT and TTFT at light load, across prompt lengths? (M1, M2)
+ How many requests per second can each configuration serve, and what limits it? (M1, M3)
+ How many requests per second meet the latency target? (M1)
+ What does prefix caching gain when requests share a system prompt? (M2)
+ At what load does speculative decoding stop helping, and how do AWQ and speculative decoding combine? (M2, M3)

*Studies.*

#tbl((auto, auto, auto, auto),
  hd([], [Study 1], [Study 2], [Study 3 (final)]),
  [Dates (2026)], [7–10 Oct], [15–26 Oct], [pilot 5 Nov; sweep 8–9 Nov],
  [GPU], [RTX PRO 4000, 24 GB], [RTX PRO 4000, 24 GB], [A100 SXM4, 80 GB],
  [Target / draft], [Qwen2.5-3B / 0.5B], [Qwen2.5-3B / 0.5B], [Qwen3-14B / 0.6B],
  [Prompts], [random tokens, 1,024], [synthetic, 512–4,096, first half shared], [ShareGPT first user turns],
  [Request rates (per s)], [2–64], [2–64], [1, 2, 3, 4, 6, 8, 12],
  [Requests per run], [1,000], [992], [200],
  [Repeats], [2–6, uneven], [1], [3, different prompt sample each],
  [Prefix caching], [on (default), nothing shared], [on (default)], [off; off against on in a paired experiment],
  [Per-request data saved], [no], [no], [yes],
  [Runs], [201], [864, of which 216 used], [504],
)

The conclusions of this report rest on study 3. The change from the 3B model to the 14B model after the progress presentation is recorded in #blob("docs/Architecture_Decision_Records.md", "ADR-006").

*Study 3 configurations.* FP16 (Qwen3-14B in BF16), AWQ (`Qwen/Qwen3-14B-AWQ`, `--quantization awq`), speculative (`Qwen/Qwen3-0.6B` draft, 5 tokens per step), AWQ + speculative; and for the caching experiment FP16 and AWQ with caching off and on. All servers: `--gpu-memory-utilization 0.95 --max-model-len 8192 --seed 0`. Sampling: temperature 0.7, top-p 0.8, top-k 20, thinking off. The load generator's seed is the repeat number.

*Study 3 workloads.*

#tbl((auto, auto, auto, 1fr),
  hd([Workload], [Prompts available], [Median prompt tokens], [Content]),
  [natural], [46,261], [44], [All deduplicated first user turns up to 4,096 tokens, in their real length mix],
  [len512, len1024, len2048, len4096], [1,643 / 1,497 / 939 / 215], [496 / 1,008 / 1,973 / 3,629], [Prompts within ±20% of the target length],
  [prefix], [43,178], [298], [One of 32 role prompts as the system message, then a user prompt; about 80% of each prompt is the shared role prompt],
)

*Metrics.* Median TTFT and median TPOT per run, where TPOT = (latency − TTFT) / (output tokens − 1); requests per second served; prompt tokens per second (total prompt tokens / run duration); goodput at the target of section 1; speculative acceptance rate. 'Capacity' is the served rate at 12 requests/s sent, and is called a capacity only where the served rate at 12 is within 3% of that at 8.

*Run scripts.* #blob(s3 + "scripts/setup_host.sh", "setup_host.sh"), #blob(s3 + "scripts/run_sweep.sh", "run_sweep.sh"); for studies 1 and 2, the `scripts/` folder of each study.

*Tests* (`pytest`, 27 tests, no GPU): summaries and tables regenerate byte-for-byte from raw files; every run of study 3 is complete; a run with a failed request and a short output is flagged; the three exclusion rules of study 2 hold in the raw data; the headline numbers of section 5 match the data; shell scripts parse and the recorded server flags match the configuration names. The end-to-end GPU workflow is the pilot (`PILOT=1 ./scripts/run_sweep.sh`); its recorded output of 5 November is in #tree(s3 + "data/pilot", "data/pilot/") and is checked by the tests. The benchmarks themselves were not rerun for this submission.

= 5. Results, observations and analysis

All numbers in 5.1–5.6 are from study 3 and are means of three repeats. Full tables with ranges: #blob(s3 + "results/tables.md", "results/tables.md"). Raw files: #tree(s3 + "data/raw", "data/raw/").

== 5.1 Light load: AWQ is fastest

#figure(
  tbl((auto, auto, auto, auto, auto), align: (left, right, right, right, right),
    hd([Workload], [FP16], [AWQ], [Speculative], [AWQ + speculative]),
    [natural], [19.3], [8.0 (2.41×)], [13.5 (1.43×)], [13.0 (1.48×)],
    [len512], [20.2], [8.6 (2.34×)], [13.5 (1.50×)], [12.8 (1.58×)],
    [len1024], [21.8], [9.5 (2.31×)], [14.9 (1.46×)], [13.8 (1.58×)],
    [len2048], [25.7], [12.9 (1.99×)], [19.7 (1.30×)], [17.9 (1.44×)],
    [len4096], [40.9], [29.5 (1.39×)#super[a]], [40.8 (1.00×)], [71.3 (0.57×)#super[a]],
  ),
  caption: [Median TPOT in ms at 1 request/s, with speed-up over FP16. #super[a] 1 request/s is above 80% of that configuration's capacity at this length, so the cell is not light load.],
  kind: table,
)

AWQ lowers TPOT 2.3–2.4× for prompts up to about 1,000 tokens. Speculative decoding gives 1.3–1.5× over FP16 and raises TTFT (59 to 103 ms on natural prompts, 172 to 216 ms at len1024), because the draft model also processes the prompt. AWQ + speculative is slower than AWQ alone at every length. Across the three repeats TPOT differs by 3.1% at the median setting.

AWQ's advantage shrinks as load approaches capacity. At len1024 its speed-up over FP16 is 2.31× at 1 request/s, 2.10× at 2, 1.40× at 4, and at 6 requests/s AWQ is slower than FP16 (191 against 123 ms), because AWQ saturates at a lower rate (5.2).

#figure(image("overview.png", width: 100%), caption: [Study 3 at 1 request/s (left, middle) and served rate at 12 requests/s sent (right).])

== 5.2 Capacity: FP16 serves the most once prompts are long

#figure(
  tbl((auto, auto, auto, auto, auto, auto, auto), align: (left, right, right, right, right, right, right),
    hd([Workload], [FP16], [AWQ], [Speculative], [AWQ + spec.], [FP16 prompt tok/s], [AWQ prompt tok/s]),
    [len1024], [5.60], [4.66 (0.83×)], [4.46 (0.80×)], [3.76 (0.67×)], [5,648], [4,703],
    [len2048], [2.79], [2.32 (0.83×)], [2.49 (0.89×)], [2.07 (0.74×)], [5,597], [4,660],
    [len4096], [1.45], [1.21 (0.83×)], [1.35 (0.93×)], [1.13 (0.78×)], [5,537], [4,621],
  ),
  caption: [Requests/s served at 12 requests/s sent, relative to FP16 in brackets, and prompt tokens processed per second. The range over repeats is under 2% of each value.],
  kind: table,
)

For natural and len512 prompts most configurations were still following the sent rate at 12 requests/s, so the sweep gives only lower bounds there (FP16 ≥ 9.2 and AWQ ≥ 8.6 at len512; FP16 ≥ 9.8 and AWQ ≥ 10.6 on natural) and we do not rank capacity for short prompts.

== 5.3 What sets the limit: prompt processing

FP16 processes 5.5–5.6k prompt tokens/s whether prompts are 1,000 or 3,600 tokens long, and AWQ 4.6–4.7k. A constant token rate across a 3.6× range of prompt length means the time spent on prompts (prefill) is what limits the server, not the number of requests or KV-cache memory. This is an inference from throughput; we did not profile the GPU.

AWQ prefills 17% slower than FP16 although it decodes faster. vLLM ran the AWQ model with its Marlin kernel (server log). A possible reason is the cost of unpacking 4-bit weights, which matters when the GPU is compute-bound in prefill and not when decode is waiting on memory. We did not test this explanation.

== 5.4 Goodput at the latency target

#figure(
  tbl((auto, auto, auto, auto, auto), align: (left, right, right, right, right),
    hd([Workload], [FP16], [AWQ], [Speculative], [AWQ + speculative]),
    [natural], [7.17 at 8/s], [9.20 at 12/s#super[b]], [6.75 at 8/s], [5.46 at 6/s],
    [len512], [5.55 at 6/s], [5.80 at 6/s], [5.44 at 6/s], [3.87 at 4/s],
    [len1024], [2.88 at 3/s], [3.13 at 4/s], [2.88 at 3/s], [2.31 at 3/s],
    [len2048], [1.21 at 2/s], [1.14 at 2/s], [0.96 at 2/s], [0.83 at 1/s],
    [len4096], [0.54 at 1/s], [0.38 at 1/s], [0.32 at 1/s], [0.09 at 1/s],
  ),
  caption: [Best goodput (requests/s with TTFT < 1 s and TPOT < 50 ms) over the tested rates, and the sent rate where it occurs. #super[b] Range over repeats 6.06–10.84; in one repeat median TPOT rose to 45 ms and 41% of requests missed the TPOT limit.],
  kind: table,
)

The rate grid is coarse, so the true optimum usually lies between two tested rates and small differences should not be read. Against default FP16, AWQ's goodput is 4–9% higher at len512 and len1024, higher but unstable on natural prompts, and lower from 2,048 tokens. At len4096 no configuration reaches 1 request/s: even at 1 request/s only 54% of FP16 requests meet both limits. Speculative decoding never has higher goodput than FP16.

== 5.5 Prefix caching

#figure(
  tbl((auto, auto, auto, auto, auto), align: (left, right, right, right, right),
    hd([Weights], [Sent /s], [TTFT ms, off → on], [TPOT ms, off → on], [Served /s, off → on]),
    [FP16], [1], [84 → 61], [19.8 → 19.3], [0.99 → 0.99],
    [FP16], [8], [153 → 78], [38.9 → 26.0], [7.17 → 7.23],
    [FP16], [12], [300 → 99], [70.2 → 32.4], [9.66 → 10.16],
    [AWQ], [1], [68 → 31], [8.3 → 8.1], [0.99 → 0.99],
    [AWQ], [8], [150 → 45], [24.6 → 11.4], [7.60 → 7.67],
    [AWQ], [12], [510 → 65], [91.4 → 17.2], [9.41 → 11.10],
  ),
  caption: [Shared-prefix workload with prefix caching off and on, in one sweep.],
  kind: table,
)

At light load caching removes 23 ms (FP16) and 37 ms (AWQ) of TTFT. Near capacity the effect is larger, because skipping prefill for the shared part frees GPU time for decode. With AWQ the best goodput rises from 7.57 (at 8/s, caching off) to 11.10 requests/s (at 12/s, caching on). 12 requests/s is the top of the grid and the caching-on runs had not flattened, so their capacity may be higher.

== 5.6 Speculative decoding: where it stops helping, and how it combines with AWQ

#figure(
  tbl((auto, auto, auto, auto, auto, auto, auto, auto), align: (left, right, right, right, right, right, right, right),
    hd([Workload], [1/s], [2/s], [3/s], [4/s], [6/s], [8/s], [12/s]),
    [natural], [1.43], [1.37], [1.32], [1.31], [1.24], [0.98#super[c]], [—],
    [len512], [1.50], [1.43], [1.35], [1.29], [1.12#super[c]], [—], [—],
    [len1024], [1.46], [1.35], [1.25], [1.05#super[c]], [—], [—], [—],
    [len2048], [1.30], [1.03#super[c]], [—], [—], [—], [—], [—],
    [len4096], [1.00], [—], [—], [—], [—], [—], [—],
  ),
  caption: [TPOT of FP16 divided by TPOT of speculative decoding, by sent rate. #super[c] The rate is above 80% of the speculative server's capacity. — the sent rate is above that capacity, so TPOT reflects queueing and is not compared.],
  kind: table,
)

The speed-up over FP16 falls as load rises and is gone (0.98–1.05×) at 8 requests/s for natural prompts, 4 at len1024 and 2 at len2048. At len512 it is 1.12× at 6 requests/s and the server is overloaded at 8. At len4096 there is none even at 1 request/s. In each case this is where the speculative server passes about 80% of its capacity. It accepts 34–39% of drafted tokens (2.7–2.9 tokens per step) in every workload and rate, with FP16 and AWQ targets alike.

The two techniques do not add on TPOT. At len1024 and 1 request/s:

#tbl((auto, auto, auto, auto), align: (left, right, right, right),
  hd([], [Without speculative], [With speculative], [Effect of speculative]),
  [FP16 weights], [21.8 ms], [14.9 ms], [−6.9 ms],
  [AWQ weights], [9.5 ms], [13.8 ms], [+4.4 ms],
  [Effect of AWQ], [−12.4 ms], [−1.1 ms], [],
)

With speculation on, TPOT is 13–15 ms whichever weights the target uses, so the time of a speculative step is dominated by something AWQ does not shorten; the five draft passes per step are the likely part, which we did not measure separately. On capacity the two costs multiply without interaction: speculative decoding costs 7–20% with FP16 weights and the same 7–19% with AWQ weights (Table 2).

== 5.7 Which configuration when (M3)

#tbl((1.2fr, auto, 1.4fr),
  hd([Regime], [Choice], [Evidence]),
  [Prompts up to about 1,000 tokens, load below about half of capacity], [AWQ], [2.3–2.4× lower TPOT; best or equal goodput up to len1024],
  [Prompts of 2,000 tokens or more, or load near capacity], [FP16], [17% more requests/s than AWQ at len1024–4096; best goodput from len2048],
  [Requests share a system prompt], [Prefix caching on, with either weight format], [TTFT 510 → 65 ms and TPOT 91 → 17 ms with AWQ at 12/s],
  [Speculative decoding with the 0.6B draft and 5 tokens], [Off], [Slower than AWQ alone at every light-load point; lowers capacity; never raises goodput],
)

Binding constraint: prefill compute (5.3). The regime boundary between AWQ and FP16 is where the workload turns from decode-bound, where AWQ's smaller weights help, to prefill-bound, where AWQ is slower.

== 5.8 What studies 1 and 2 showed, and what did not hold

#tbl((1.25fr, auto, auto),
  hd([At about 1,024 prompt tokens], [Study 2: Qwen2.5-3B], [Study 3: Qwen3-14B]),
  [AWQ TPOT speed-up at 2 requests/s], [2.63×], [2.10×],
  [Speculative TPOT speed-up at 2 requests/s], [2.09×], [1.35×],
  [Lowest TPOT at light load], [AWQ + speculative, 7.1 ms], [AWQ alone, 9.5 ms],
  [Capacity, AWQ against FP16], [3% higher], [17% lower],
  [Capacity, speculative against FP16, 2,048+ tokens], [26–27% higher], [7–11% lower],
  [Speculative acceptance rate], [58–69%], [34–39%],
)

Study 1 (one prompt length, random tokens) gave AWQ 2.46× and speculative 1.56× lower TPOT at 2 requests/s, and AWQ + speculative slower than AWQ alone. Study 2 numbers: #blob("evaluation/02_qwen2.5-3b_lengths/results/tables.md", "study 2 tables").

Two findings hold across all three studies: AWQ gives the largest light-load gain, and capacity is limited by prefill. The speculative-decoding gains of study 2 did not hold. The studies differ in model family, model size, GPU, prompts, repeats and caching at once, so we cannot say which difference is responsible. The acceptance rate, which is lower on real prompts, is one measured difference.

== 5.9 Failures, excluded data and tests not run

- *FP8 KV cache.* All 432 FP8 runs of study 2 completed 0 requests: the FP8 server did not serve on the lab host, which has no CUDA compiler. It was also attempted during study 1 and failed; those files are not in the repository. FP8 was not run on the 14B model. There is no FP8 result.
- *Study 2 exclusions.* Two AWQ-with-caching sweeps ran concurrently against one server, and the `baseline` sweep served under a third of the identical `CR` configuration for a reason we did not find. Both are excluded; 216 of 864 runs are used (#blob("docs/Architecture_Decision_Records.md", "ADR-005")).
- *Prefix-caching claim in the progress presentation.* It reported 1.8–2.0× capacity from shared prefixes by comparing study 2 with study 1. That comparison is across two sweeps and is not a valid measurement; 5.5 replaces it.
- *Study 3.* One run lost 1 of 200 requests to a dropped connection, was flagged by the checks and rerun. The first pilot attempt failed every request because vLLM's benchmark extras were not installed; the setup script now installs them.
- *Not run:* output-quality comparison of AWQ against FP16; caching combined with speculative decoding on the 14B model; draft lengths other than 5 and drafts other than 0.6B; a same-GPU small-model control; request rates above 12/s for short prompts; multi-turn conversations; tuning of batching or scheduling parameters.

= 8. Current state, limitations and risks

#tbl((auto, 1fr, auto),
  hd([Milestone clause], [State], [Evidence]),
  [M1 quantised weights], [*Working.* Measured on 3B and 14B], [5.1, 5.2],
  [M1 reduced-precision KV cache], [*Missing.* Attempted on the lab GPU and failed to serve; not run on the A100], [5.9],
  [M1 tuned batching, prefill scheduling, parallelism], [*Missing.* vLLM defaults, one GPU], [ADR-001],
  [M1 improvement at a latency target], [*Partial.* AWQ: goodput +4–9% at 512–1,024 tokens, lower from 2,048 tokens. With shared system prompts, AWQ + caching reaches 11.1 against 7.0 requests/s for FP16 without caching], [5.4, 5.5],
  [M1 output quality preserved], [*Missing.* Not measured. Speculative decoding does not change the output distribution by construction; AWQ changes the weights and was not checked], [5.9],
  [M2 prefix caching], [*Working* for FP16 and AWQ; not combined with speculative decoding], [5.5],
  [M2 speculative decoding and its load limit], [*Working* for one draft model and one draft length], [5.6],
  [M3 attribution], [*Working* for AWQ × speculative (2 × 2). Caching is measured separately, not in the factorial], [5.6],
  [M3 binding constraint], [*Partial.* Prefill compute, inferred from a constant prompt-token rate; not profiled], [5.3],
  [M3 operating regimes], [*Working*], [5.7],
)

*Limitations.*
- 200 requests per run. At 12 requests/s the arrivals last about 17 s, so overloaded runs show a queue building and not a steady state.
- The rate grid stops at 12 requests/s; capacity for prompts under about 1,000 tokens is only bounded from below.
- Output length is fixed at 128 tokens. Real answers vary in length.
- TPOT spread across repeats is 3.1% at the median setting but reaches 76% at rates near a configuration's capacity, where a different prompt sample tips the server into queueing.
- The explanation for AWQ's slower prefill, and for speculative decoding not adding to AWQ, are hypotheses.
- Studies 1 and 2 ran on a shared GPU with one run per setting (study 2); their absolute numbers carry unknown run-to-run variation.

*Known defects and technical debt.* The study 2 script waits only for port 8000, so it benchmarks whatever answers there; the study 3 script checks its own server process but would still be misled by another process already on port 8000. The three analysis folders duplicate the file-reading and table code. Model revisions are not pinned. `analyse.py` asserts exactly 504 runs, so it must be edited for a different grid.

*Risks and dependencies.* Rerunning study 3 needs a rented 80 GB GPU for about 15 hours. vLLM's defaults change between releases (prefix caching is on by default in 0.30), so results depend on the pinned version. The Qwen2.5-3B weights are under a research licence. ShareGPT contains text that users shared publicly; we publish generated answers but not the prompts. No credentials or personal data are stored. Open work is listed in section 9; GitHub issues were not opened for it.

= 9. Way forward

#tbl((auto, 1.1fr, 1.2fr, 1.2fr),
  hd([], [Task and start point], [Why, and what it depends on], [Done when]),
  [1], [AWQ output-quality check. New script next to `run_sweep.sh` that sends a fixed prompt set to the FP16 and AWQ servers at temperature 0], [M1 requires quality to be preserved, and AWQ is the recommended configuration for short prompts. Needs the 80 GB GPU for about an hour and a choice of benchmark], [Accuracy of FP16 and AWQ on the same public benchmark subset is reported with its sampling error],
  [2], [FP8 KV cache on the A100. Add `fp16_kv8` and `awq_kv8` to `args_of` in `run_sweep.sh`], [The only M1 technique with no result. Needs a host image with the CUDA toolkit. Expected to matter only where KV memory limits batch size, which 5.3 suggests is not the case here], [Both configurations pass the run checks for len1024 and len4096, and their capacity is compared with Table 2],
  [3], [Same-GPU control with a small Qwen3 model, same prompts and script (`BASE_MODEL`, `AWQ_MODEL` variables)], [Separates model size from the other differences between studies 2 and 3. Blocked if no official AWQ checkpoint exists for the small model], [Section 5.8 can be restated with one variable changed],
  [4], [Speculative decoding with 2–3 draft tokens, and with caching on. `num_speculative_tokens` in `run_sweep.sh`; add `awq_spec_cacheon`], [Acceptance is 2.7–2.9 tokens of 5, so most draft work is discarded; and caching × speculative is the untested cell of the M3 factorial], [The table in 5.6 is extended with the new draft length and the caching column],
  [5], [Profile prefill for FP16 and AWQ at len2048 (PyTorch profiler or `nvidia-smi dmon` during a saturated run)], [Tests the explanation in 5.3 for AWQ's 17% lower capacity], [Time per prefill token is split into kernel time and the rest for both weight formats],
)

= 10. Contributions, AI use and attribution

#tbl((auto, 1fr),
  hd([Member], [Areas]),
  [Vishank Singh], [Studies 1 and 2 on the lab GPU: vLLM setup, benchmark scripts (#blob("evaluation/02_qwen2.5-3b_lengths/scripts/run_sweep.sh", "study 2 script")) and the 1,065 result files],
  [Anup Chavan], [Study 3: model and GPU selection, prompt workloads, sweep and pilot (#tree(s3, "study 3 folder")); analysis of all three studies; repository, tests, report and presentations. Commits are pushed from this account; the commits that add Vishank Singh's scripts and result files name him as co-author (#link(repo + "/commits/cs5903-final-2026", "commit history"))],
)

*AI assistance.* An AI coding assistant (Claude) was used throughout: to write the study 3 scripts, the analysis code and the tests under our direction, to cross-check numbers, and to draft and edit this report, the decision records and the slides. How it was checked: every run's medians are recomputed from per-request data and compared with vLLM's own output; every table is regenerated from raw files by the tests; the headline numbers of section 5 are asserted by tests that read them from the data; and results that contradicted our earlier presentation are reported as such. The assistant also operated the rented GPU host for study 3 under our supervision. We chose the experiments and are responsible for the content.

*Third-party material.* vLLM, the Qwen models, ShareGPT and awesome-chatgpt-prompts are used under the licences listed in section 2 and are downloaded by the setup script, not redistributed. No code was copied from other projects.

= References

#set text(size: 9pt)
#set par(justify: false)
[1] W. Kwon et al. Efficient Memory Management for Large Language Model Serving with PagedAttention. SOSP 2023. \
[2] A. Agrawal et al. Taming Throughput-Latency Tradeoff in LLM Inference with Sarathi-Serve. OSDI 2024. \
[3] L. Zheng et al. SGLang: Efficient Execution of Structured Language Model Programs. NeurIPS 2024. \
[4] Y. Leviathan, M. Kalman, Y. Matias. Fast Inference from Transformers via Speculative Decoding. ICML 2023. \
[5] J. Lin et al. AWQ: Activation-aware Weight Quantization for On-Device LLM Compression and Acceleration. MLSys 2024. \
[6] vLLM documentation and source, version 0.30.0. #link("https://github.com/vllm-project/vllm") \
[7] Qwen Team. Qwen3 Technical Report, 2025; Qwen2.5 Technical Report, 2024.

= Appendix: where the evidence is

#tbl((auto, 1fr),
  hd([Item], [Location]),
  [Raw results, study 3 (504 runs, per-request)], [#tree(s3 + "data/raw", s3 + "data/raw/")],
  [Run checks output], [#blob(s3 + "data/checks.txt", s3 + "data/checks.txt")],
  [Server start-up records (KV cache size, flags, GPU)], [#tree(s3 + "data", s3 + "data/server_*.txt")],
  [Server and benchmark logs], [#tree(s3 + "logs", s3 + "logs/")],
  [All generated tables], [#blob(s3 + "results/tables.md", s3 + "results/tables.md")],
  [Pilot run (end-to-end GPU workflow)], [#tree(s3 + "data/pilot", s3 + "data/pilot/")],
  [Studies 1 and 2], [#tree("evaluation/01_qwen2.5-3b_1024tok", "evaluation/01_…"), #tree("evaluation/02_qwen2.5-3b_lengths", "evaluation/02_…")],
  [Earlier reports and slides], [#tree("docs/progress", "docs/progress/")],
)

Mentor feedback and response: the feedback at the progress presentation (results on a 3B model are too noisy; repeat on a larger model) is answered by study 3 and recorded in ADR-006. No bonus claims are made.
