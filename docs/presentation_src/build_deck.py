"""Build docs/Final_Presentation.pptx from template.pptx (the progress-presentation deck,
used for its layout) and the study 3 tables. Chart values are read from
evaluation/03_qwen3-14b_sharegpt/results/by_setting.csv, not typed in.

    python3 docs/presentation_src/build_deck.py
"""
import copy, os
import pandas as pd
from lxml import etree
from pptx import Presentation
from pptx.chart.data import CategoryChartData

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, '..', '..')
A = pd.read_csv(os.path.join(ROOT, 'evaluation', '03_qwen3-14b_sharegpt', 'results', 'by_setting.csv')
                ).set_index(['workload', 'config', 'rate'])
CONFIGS = [('fp16', 'FP16'), ('awq', 'AWQ'), ('spec', 'Speculative'), ('awq_spec', 'AWQ + speculative')]
NS = '{http://schemas.openxmlformats.org/drawingml/2006/main}'

prs = Presentation(os.path.join(HERE, 'template.pptx'))
old = list(prs.slides)


def shape(slide, name):
    return next(s for s in slide.shapes if s.name == name)


def charts(slide):
    return [s.chart for s in slide.shapes if getattr(s, 'has_chart', False) and s.has_chart]


# Paragraph templates, taken from the template deck so fonts and spacing stay as designed.
_body14 = shape(old[13], 'Text Placeholder 2').text_frame.paragraphs
T_HEAD = copy.deepcopy(_body14[0]._p)                 # level 0, bold run only
T_SUB = copy.deepcopy(_body14[1]._p)                  # level 1, plain run
_p8 = shape(old[7], 'Text Placeholder 2').text_frame.paragraphs[0]._p
T_LEAD = copy.deepcopy(_p8)                           # level 0, bold lead + plain run
T_PLAIN = copy.deepcopy(shape(old[1], 'Text Placeholder 2').text_frame.paragraphs[0]._p)


def _runs(p):
    return [r for r in p if r.tag == NS + 'r']


def para(kind, *texts):
    """kind: 'lead' (bold, plain), 'plain' (plain), 'head' (bold), 'sub' (level-1 plain)."""
    p = copy.deepcopy({'lead': T_LEAD, 'plain': T_PLAIN, 'head': T_HEAD, 'sub': T_SUB}[kind])
    runs = _runs(p)
    assert len(runs) >= len(texts), kind
    for r, t in zip(runs, texts):
        r.find(NS + 't').text = t
    for r in runs[len(texts):]:
        p.remove(r)
    return p


def set_body(slide, paras, name='Text Placeholder 2'):
    txBody = shape(slide, name).text_frame._txBody
    for p in txBody.findall(NS + 'p'):
        txBody.remove(p)
    for p in paras:
        txBody.append(p)


def set_text(shape_, text):
    """Replace a shape's text, keeping the formatting of its first run."""
    tf = shape_.text_frame
    first = tf.paragraphs[0]
    for p in tf.paragraphs[1:]:
        tf._txBody.remove(p._p)
    lines = text.split('\n')
    runs = first.runs
    runs[0].text = lines[0]
    for r in runs[1:]:
        first._p.remove(r._r)
    for line in lines[1:]:
        p = copy.deepcopy(first._p)
        _runs(p)[0].find(NS + 't').text = line
        tf._txBody.append(p)


def set_title(slide, title, eyebrow=None):
    set_text(shape(slide, 'Title 1'), title)
    if eyebrow:
        set_text(shape(slide, 'Eyebrow'), eyebrow)


def set_table(table, rows):
    tbl = table._tbl
    trs = tbl.findall(NS + 'tr')
    while len(trs) < len(rows):
        tbl.append(copy.deepcopy(trs[-1]))
        trs = tbl.findall(NS + 'tr')
    for tr in trs[len(rows):]:
        tbl.remove(tr)
    for r, row in enumerate(rows):
        for c, text in enumerate(row):
            set_text(table.cell(r, c), text)


def set_chart(chart, categories, series, title=None, axis_title=None, ymax=None):
    data = CategoryChartData()
    data.categories = categories
    for name, values in series:
        data.add_series(name, [round(float(v), 3) for v in values])
    chart.replace_data(data)
    if title:
        chart.chart_title.text_frame.paragraphs[0].runs[0].text = title
    if axis_title:
        chart.category_axis.axis_title.text_frame.paragraphs[0].runs[0].text = axis_title
    if ymax:
        chart.value_axis.maximum_scale = ymax


def notes(slide, text):
    slide.notes_slide.notes_text_frame.text = text


def duplicate(src):
    new = prs.slides.add_slide(src.slide_layout)
    tree = new.shapes._spTree
    for el in list(tree)[2:]:
        tree.remove(el)
    for el in list(src.shapes._spTree)[2:]:
        tree.append(copy.deepcopy(el))
    return new


def table_of(slide):
    return next(s for s in slide.shapes if getattr(s, 'has_table', False) and s.has_table).table


def col(workloads, config, rate, metric, scale=1.0):
    return [A.loc[(w, config, rate), metric] * scale for w in workloads]


# Extra slides are copies of template slides; made before any text is changed.
s_changed = duplicate(old[7])      # eyebrow + title + bullets (charts removed below)
s_noadd = duplicate(old[4])        # eyebrow + title + 3-column table + footnote
s_nothold = duplicate(old[4])
s_status = duplicate(old[13])      # title + two-level bullets
for el in [s._element for s in s_changed.shapes if getattr(s, 'has_chart', False) and s.has_chart]:
    s_changed.shapes._spTree.remove(el)

# 1 Title: unchanged.
notes(old[0], "We are Group 4, Project 24, efficient LLM serving. We measured how much quantisation, prefix "
              "caching and speculative decoding speed up a model served on one GPU, and when each is worth "
              "switching on. This is the final presentation; the results are from a 14B model with real prompts.")

# 2 Motivation: text unchanged.
# 3 Problem statement: text unchanged.

# 4 How a request is served
set_body(old[3], [
    para('plain', 'TTFT: time to first token.  TPOT: time per output token after it'),
    para('plain', 'Prefill is limited by compute; decode reads all the weights for every token '
                  '(about 30 GB for a 14B model in 16 bits), so memory bandwidth limits it'),
])
notes(old[3], "A request waits in a queue, then the model reads the whole prompt in one pass: prefill. Then it "
              "writes the answer one token per pass: decode. State for every token is kept in GPU memory, the "
              "KV cache. TTFT is queue plus prefill. TPOT is the time per decode pass. Prefill is compute-bound "
              "and decode is memory-bound, and that difference explains most of the results.")

# 5 The techniques
t = table_of(old[4])
set_text(t.cell(4, 1), 'A 0.6B draft proposes 5 tokens; the 14B model checks them in one pass')
notes(old[4], "Four techniques. AWQ stores the weights in 4 bits, so decode reads less. FP8 KV cache halves the "
              "KV cache. Prefix caching reuses the KV cache of a prompt prefix seen before. Speculative decoding "
              "lets a small draft model propose tokens that the large model checks in one pass. We have no FP8 "
              "result: the FP8 server did not serve on the lab GPU and we did not run it on the 14B model.")

# 6 Benchmark pipeline
set_body(old[5], [
    para('plain', 'For each configuration: start the server, then run every workload and request rate'),
    para('plain', 'Each run: 200 requests at random arrival times, timings saved per request'),
    para('plain', 'Runs with a failed request are rejected'),
])
set_text(shape(old[5], 'Rounded Rectangle 902'), 'Serve\nQwen3-14B on one GPU')
set_text(shape(old[5], 'Rounded Rectangle 903'), 'Record\none JSON per run')
set_text(shape(old[5], 'Rounded Rectangle 904'), 'Analyse\nchecks, tables, plots')
set_text(shape(old[5], 'TextBox 909'), 'Repeat for every rate (1–12 requests/s), workload and repeat (3), '
                                       'then the next configuration')
notes(old[5], "A script starts the vLLM server with one set of flags. vLLM's benchmark tool sends 200 requests "
              "at a fixed rate and saves per-request timings. The script goes through every rate, workload and "
              "repeat, then restarts the server with the next configuration. Our analysis recomputes the medians "
              "of every run and refuses to use a run with a failed request. All 504 runs passed.")

# 7 Experimental setup
set_table(table_of(old[6]), [
    ['GPU', 'NVIDIA A100 SXM4, 80 GB'],
    ['Server', 'vLLM 0.30.0; prefix caching off, except in the caching experiment'],
    ['Models', 'Qwen3-14B (16-bit or AWQ 4-bit); Qwen3-0.6B draft, 5 tokens'],
    ['Prompts', 'ShareGPT first user turns: natural mix, and buckets near 512, 1,024, 2,048 and 4,096 tokens'],
    ['Shared prefixes', 'One of 32 role prompts as the system message, about 80% of each prompt'],
    ['Load', '1–12 requests/s, 200 requests per run, 128 output tokens'],
    ['Runs', '4 configurations plus caching off and on, 3 repeats each: 504 runs'],
])
notes(old[6], "One A100 with 80 GB. vLLM 0.30.0 with Qwen3-14B in 16 bits or AWQ 4 bits, and Qwen3-0.6B as the "
              "draft model. Prompts are real first messages from ShareGPT conversations, either in their natural "
              "length mix, which is mostly short, or grouped near a target length. Output is fixed at 128 tokens "
              "so TPOT is comparable. Three repeats per setting, each with a different sample of prompts.")

# 8 What changed since the progress presentation
set_title(s_changed, 'What changed since the progress presentation', 'Approach')
set_body(s_changed, [
    para('lead', 'Model: ', 'Qwen2.5-3B on a 24 GB GPU → Qwen3-14B on an 80 GB GPU'),
    para('lead', 'Prompts: ', 'synthetic text → real ShareGPT prompts'),
    para('lead', 'Repeats: ', 'one run per setting → three, each with a different prompt sample'),
    para('lead', 'Prefix caching: ', 'on in every run → off, and measured off against on in one sweep'),
    para('lead', 'Checks: ', 'per-request timings saved, medians recomputed for every run'),
])
_b = shape(s_changed, 'Text Placeholder 2')
_b.height = int(3.2 * 914400)
for _el in _b.text_frame._txBody.find(NS + 'bodyPr'):
    _b.text_frame._txBody.find(NS + 'bodyPr').remove(_el)
notes(s_changed, "At the progress presentation our results were from a 3B model, and the feedback was that effects "
                 "on a model that small are too noisy to trust. So we repeated the study on a 14B model. We also "
                 "fixed three weaknesses of our own: synthetic prompts, one run per setting, and prefix caching "
                 "being on in every run, which meant we had no caching-off comparison.")

# 9 Light load
W4 = ['natural', 'len512', 'len1024', 'len2048']
C4 = ['natural mix', '512', '1,024', '2,048']
set_title(old[7], 'Light load: 1 request/s')
set_body(old[7], [
    para('lead', 'AWQ: ', '2.3× lower TPOT than FP16 at 1,024 tokens (21.8 → 9.5 ms)'),
    para('lead', 'Speculative: ', '1.5× lower TPOT than FP16, and higher TTFT (the draft model prefills too)'),
    para('lead', 'Both together: ', 'slower than AWQ alone at every length (13.8 against 9.5 ms)'),
])
c = charts(old[7])
set_chart(c[0], C4, [(n, col(W4, k, 1.0, 'tpot')) for k, n in CONFIGS], 'TPOT (ms)', 'Prompt length (tokens)')
set_chart(c[1], C4, [(n, col(W4, k, 1.0, 'ttft')) for k, n in CONFIGS], 'TTFT (ms)', 'Prompt length (tokens)')
notes(old[7], "Light load, 1 request per second. On the left, time per output token. AWQ brings it from 21.8 to "
              "9.5 milliseconds at 1,024 tokens, 2.3 times lower, because decode is limited by memory and the AWQ "
              "weights are about a third of the size. Speculative decoding gives about 1.5 times, and it raises "
              "time to first token, on the right, because the draft model also reads the prompt. The new result: "
              "AWQ with speculative decoding is slower than AWQ alone at every length. On the 3B model it was the "
              "fastest. The numbers are means of three repeats, which differ by about 3 percent.")

# 10 Capacity
R = [1.0, 2.0, 3.0, 4.0, 6.0, 8.0, 12.0]
RC = ['1', '2', '3', '4', '6', '8', '12']
set_body(old[8], [
    para('lead', 'Limit at 1,024 tokens: ', 'FP16 5.6, AWQ 4.7, speculative 4.5, both 3.8 requests/s'),
    para('plain', 'Past the limit, requests queue: median TTFT reaches 5–15 s at 12 requests/s'),
])
c = charts(old[8])
set_chart(c[0], RC, [('Sent', R)] + [(n, [A.loc[('len1024', k, r), 'thr'] for r in R]) for k, n in CONFIGS],
          'Finished (requests/s)', 'Sent (requests/s)', 12)
set_chart(c[1], RC, [(n, [A.loc[('len1024', k, r), 'ttft'] / 1000 for r in R]) for k, n in CONFIGS],
          'Median TTFT (s)', None, 16)
notes(old[8], "Now we raise the load, with prompts of about 1,024 tokens. On the left, the top line is what we send "
              "and the others are what the server finishes. Each one follows the sent rate and then goes flat. "
              "FP16 goes flat at 5.6 requests per second, AWQ at 4.7. So AWQ, the fastest at light load, serves 17 "
              "percent fewer requests under load, and speculative decoding costs capacity too. On the right, once "
              "a server is past its limit, requests queue and time to first token grows to many seconds. Runs are "
              "200 requests long, so this shows a queue building, not a steady state.")

# 11 What sets the limit
W3 = ['len1024', 'len2048', 'len4096']
set_body(old[9], [
    para('lead', 'FP16: ', '5.5–5.6k prompt tokens/s at every length, so prefill compute sets the limit'),
    para('lead', 'AWQ: ', '4.6–4.7k. It decodes faster but prefills 17% slower; the cause is not profiled'),
    para('lead', 'Speculative: ', '7–20% fewer requests/s, with FP16 or with AWQ weights'),
])
set_chart(charts(old[9])[0], ['1,024', '2,048', '4,096'],
          [(n, col(W3, k, 12.0, 'ptok', 1e-3)) for k, n in CONFIGS],
          'Prompt tokens/s at capacity (thousands)', 'Prompt length (tokens)', 6)
notes(old[9], "Here capacity is turned into prompt tokens processed per second. For FP16 it is about 5,500 whether "
              "prompts are 1,000 or 4,000 tokens long. A constant token rate means prefill compute is the limit, "
              "not memory and not the number of requests. AWQ is at about 4,650, so it prefills 17 percent "
              "slower. vLLM runs AWQ with a kernel that unpacks the 4-bit weights; that cost may matter when the "
              "GPU is busy with prefill. We did not profile it, so that is a hypothesis. We infer the limit from "
              "throughput, not from a profiler.")

# 12 Within a latency target
W5 = ['natural', 'len512', 'len1024', 'len2048', 'len4096']
set_body(old[10], [
    para('lead', 'Target: ', 'TTFT < 1 s and TPOT < 50 ms, checked for every request'),
    para('lead', 'AWQ: ', '4–9% more than FP16 at 512 and 1,024 tokens, less from 2,048 tokens'),
    para('lead', 'Speculative: ', 'never more than FP16. At 4,096 tokens no setup reaches 1 request/s'),
])
set_chart(charts(old[10])[0], ['natural mix', '512', '1,024', '2,048', '4,096'],
          [(n, [A.loc[(w, k), 'goodput'].max() for w in W5]) for k, n in CONFIGS],
          'Best rate of requests within the target (requests/s)', 'Prompt length (tokens)', 10)
notes(old[10], "Capacity is what a server can finish, but users care about waiting. Our target is time to first "
               "token under one second and time per token under 50 milliseconds, and we count the requests per "
               "second that meet both. The target values are our choice. AWQ is a little ahead of FP16 up to "
               "1,024 tokens and behind from 2,048. On the natural mix AWQ shows 9.2 against 7.2, but that point "
               "varied between 6.1 and 10.8 over the three repeats. The rate grid is coarse, so small differences "
               "should not be read. Speculative decoding never beats FP16 here.")

# 13 Prefix caching
set_title(old[11], 'Shared system prompts')
set_body(old[11], [
    para('plain', 'Same workload with prefix caching off and on; only the shared system prompt can be reused'),
    para('lead', 'Near capacity, AWQ at 12 requests/s: ', 'TTFT 510 → 65 ms, TPOT 91 → 17 ms'),
    para('lead', 'Light load: ', '23–37 ms off TTFT, TPOT unchanged'),
])
c = charts(old[11])
pf = lambda cache, metric: [A.loc[('prefix', f'{q}_{cache}', 12.0), metric] for q in ('fp16', 'awq')]
set_chart(c[0], ['FP16', 'AWQ'], [('Caching off', pf('cacheoff', 'ttft')), ('Caching on', pf('cacheon', 'ttft'))],
          'TTFT at 12 requests/s (ms)', None, 600)
set_chart(c[1], ['FP16', 'AWQ'], [('Caching off', pf('cacheoff', 'tpot')), ('Caching on', pf('cacheon', 'tpot'))],
          'TPOT at 12 requests/s (ms)', None, 100)
notes(old[11], "Prefix caching. Each prompt starts with one of 32 role prompts as the system message, about 80 "
               "percent of the prompt, followed by a real user message. We ran the same workload with caching off "
               "and on. With caching on, every run got user messages the server had not seen, so only the system "
               "prompt could come from the cache. At light load it saves 23 to 37 milliseconds of time to first "
               "token. Near capacity it is much larger, because skipping most of the prefill frees the GPU for "
               "decode: with AWQ at 12 requests per second, time to first token goes from 510 to 65 milliseconds. "
               "In the progress presentation we compared two different sweeps for this; this is the proper "
               "measurement. We did not run caching together with speculative decoding.")

# 14 AWQ and speculative decoding do not add
set_title(s_noadd, 'AWQ and speculative decoding do not add', 'Results')
set_table(table_of(s_noadd), [
    ['TPOT at 1,024 tokens, 1 request/s', 'Without speculative', 'With speculative'],
    ['FP16 weights', '21.8 ms', '14.9 ms'],
    ['AWQ weights', '9.5 ms', '13.8 ms'],
    ['Effect of AWQ', '−12.4 ms', '−1.1 ms'],
])
set_text(shape(s_noadd, 'TextBox 901'),
         'The draft\'s tokens are accepted 34–39% of the time on real prompts. Against FP16, the speculative '
         'gain is gone at about 80% of capacity.')
shape(s_noadd, 'TextBox 901').top = int(3.95 * 914400)
notes(s_noadd, "This is the attribution. Speculative decoding takes 6.9 milliseconds off FP16, but adds 4.4 to "
               "AWQ. With speculation on, TPOT is 13 to 15 milliseconds whichever weights are used, so the cost of "
               "a speculative step is in something AWQ does not shorten, most likely the five draft passes. We did "
               "not measure that separately. The draft's tokens are accepted 34 to 39 percent of the time on real "
               "prompts; on the synthetic prompts of the 3B study it was 58 to 69 percent. Against FP16, the "
               "speculative gain shrinks as load rises and is gone at about 80 percent of capacity: 8 requests per "
               "second on short prompts, 4 at 1,024 tokens, 2 at 2,048.")

# 15 Which setup, when
s15 = old[12]
for el in list(s15.shapes._spTree):
    if etree.QName(el).localname == 'AlternateContent':
        s15.shapes._spTree.remove(el)
s15.shapes._spTree.append(copy.deepcopy(next(
    el for el in old[4].shapes._spTree if etree.QName(el).localname == 'graphicFrame')))
set_table(table_of(s15), [
    ['Workload', 'Best setup', 'Why'],
    ['Up to ~1,000 tokens, light load', 'AWQ', '2.3× lower TPOT, best goodput'],
    ['2,000+ tokens, or near capacity', 'FP16', '17% more requests/s than AWQ'],
    ['Shared system prompts', 'Prefix caching on', 'TTFT 510 → 65 ms at 12 requests/s'],
    ['Speculative decoding, 0.6B draft', 'Off', 'Slower than AWQ, costs capacity'],
])
set_text(shape(s15, 'TextBox 901'),
         'The limit under load is prefill compute: about 5,500 prompt tokens/s for FP16 on this GPU.')
shape(s15, 'TextBox 901').top = int(4.92 * 914400)
notes(s15, "Putting it together. For prompts up to about a thousand tokens at light load, AWQ. For long prompts or "
           "a server near capacity, FP16, because AWQ prefills slower. If requests share a system prompt, turn "
           "prefix caching on. Speculative decoding with this draft model is not worth enabling: AWQ alone is "
           "faster and it costs capacity. The boundary between AWQ and FP16 is where the work turns from mostly "
           "decode to mostly prefill.")

# 16 What did not hold
set_title(s_nothold, 'What did not hold from the 3B results', 'Results')
set_table(table_of(s_nothold), [
    ['At about 1,024 tokens', 'Qwen2.5-3B (progress)', 'Qwen3-14B (final)'],
    ['Lowest TPOT at light load', 'AWQ + speculative', 'AWQ alone'],
    ['Capacity, AWQ against FP16', '3% higher', '17% lower'],
    ['Capacity, speculative, 2,048+ tokens', '26–27% higher', '7–11% lower'],
    ['Speculative acceptance rate', '58–69%', '34–39%'],
])
set_text(shape(s_nothold, 'TextBox 901'),
         'Model, GPU, prompts, repeats and caching all changed between the two, so the cause of each difference '
         'is not isolated.')
shape(s_nothold, 'TextBox 901').top = int(4.92 * 914400)
notes(s_nothold, "Compared with what we showed at the progress presentation. Two things hold on both models: AWQ "
                 "gives the largest gain at light load, and prefill limits capacity. The speculative decoding "
                 "results did not hold. On the 3B model AWQ plus speculative was fastest and speculative decoding "
                 "added capacity on long prompts; on the 14B model neither is true. We cannot say which change "
                 "caused this, because the model, the GPU, the prompts and the caching setting all changed "
                 "together. One measured difference is the acceptance rate, which is much lower on real prompts.")

# 17 What is done and what is not
set_title(s_status, 'What is done and what is not')
set_body(s_status, [
    para('head', 'Done'),
    para('sub', 'AWQ, speculative decoding and both, measured on 3B and 14B models'),
    para('sub', 'Prefix caching off against on; goodput at a latency target; which setup when'),
    para('sub', 'Every table regenerates from the raw result files; 27 automated tests'),
    para('head', 'Not done'),
    para('sub', 'FP8 KV cache: the server did not serve on the lab GPU; not run on the 14B model'),
    para('sub', 'AWQ output quality; tuning of batching and scheduling'),
    para('sub', 'Prefix caching together with speculative decoding'),
])
notes(s_status, "Against the milestones. M2 and M3 are covered: caching, speculative decoding, the attribution and "
                "the regimes. M1 is partly covered. We have no FP8 KV cache result: in the 864-run sweep on the lab "
                "GPU all 432 FP8 runs completed zero requests, because the FP8 server did not serve there, and we "
                "did not run it on the 14B model. We did not check AWQ's output quality, which M1 asks for. And we "
                "left batching and scheduling at vLLM's defaults.")

# 18 Limitations and next steps
set_title(old[13], 'Limitations and next steps')
set_body(old[13], [
    para('head', 'Limitations'),
    para('sub', 'One engine (vLLM 0.30.0), one draft model, 128-token outputs'),
    para('sub', '200 requests per run; the rate grid stops at 12 requests/s'),
    para('sub', 'Why AWQ prefills slower is a hypothesis, not profiled'),
    para('head', 'Next'),
    para('sub', 'AWQ quality on a public benchmark; FP8 KV cache on the A100'),
    para('sub', 'A small Qwen3 model on the same GPU and prompts, to isolate model size'),
    para('sub', 'Fewer draft tokens; profile prefill for FP16 and AWQ'),
])
notes(old[13], "Limitations. Everything is on vLLM 0.30.0 with one draft model and fixed 128-token outputs. Runs are "
               "200 requests, so overloaded runs are short, and for short prompts our rates did not reach the "
               "server's limit. Next: the AWQ quality check and FP8, which are the missing parts of M1; a small "
               "Qwen3 model on the same GPU and prompts, which would tell us whether model size explains the "
               "differences from the 3B study; and fewer draft tokens, since most drafted tokens are rejected.")

# Order: 1-7, changed, 8-12 (results), no-add, 13 (which setup), not-hold, status, 14 (limits), 15 (refs), 16 (end)
for _p in shape(old[14], 'Text Placeholder 2').text_frame.paragraphs:
    for _r in _p.runs:
        _r.text = _r.text.replace('Qwen2.5 Technical Report', 'Qwen3 Technical Report').replace(
            'arXiv 2412.15115, 2024', 'arXiv 2505.09388, 2025')
order = old[:7] + [s_changed] + old[7:12] + [s_noadd, old[12], s_nothold, s_status] + old[13:]
lst = prs.slides._sldIdLst
by_id = {int(el.get('id')): el for el in list(lst)}
order_ids = [s.slide_id for s in order]
for el in list(lst):
    lst.remove(el)
for i in order_ids:
    lst.append(by_id[i])

prs.save(os.path.join(HERE, '..', 'Final_Presentation.pptx'))
print(f'wrote docs/Final_Presentation.pptx ({len(order)} slides)')
