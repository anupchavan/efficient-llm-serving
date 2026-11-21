"""data/raw -> data/summary.csv (one row per run) for the 864-run sweep.

Each row gets a `status`: `valid`, or `excluded: <reason>`. The exclusion rules
are listed in ../README.md and are applied here, per configuration:
  KV*, WQ_KV*   every request failed (the FP8 KV-cache server never served)
  WQ_CR, WQ_CR_MT   both ran against one server at the same time
  baseline      same server flags as CR, but less than two thirds of its capacity
"""
import csv, os, re
from rawio import runs

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
NAME = re.compile(r'Qwen2\.5-3B-Instruct(-AWQ)?_(?P<cfg>.+)_rate(?P<rate>\d+)_len(?P<len>\d+)\.json$')
KEYS = ['date', 'model_id', 'num_prompts', 'completed', 'failed', 'duration', 'request_throughput',
        'output_throughput', 'total_input_tokens', 'total_output_tokens',
        'mean_ttft_ms', 'median_ttft_ms', 'p99_ttft_ms', 'mean_tpot_ms', 'median_tpot_ms',
        'p99_tpot_ms', 'max_concurrent_requests',
        'spec_decode_acceptance_rate', 'spec_decode_acceptance_length', 'unreadable']
# Names used in the report for the four configurations that are compared.
LABEL = {'CR': 'fp16', 'WQ': 'awq', 'MT': 'spec', 'WQ_MT': 'awq_spec'}


def status(cfg):
    if 'KV' in cfg:
        return 'excluded: all requests failed (FP8 KV-cache server did not serve)'
    if cfg in ('WQ_CR', 'WQ_CR_MT'):
        return 'excluded: ran concurrently with another benchmark on one server'
    if cfg == 'baseline':
        return 'excluded: same flags as CR but less than two thirds of its capacity; cause unknown'
    if cfg == 'CR_MT':
        return 'valid (duplicate of MT: prefix caching is on by default)'
    return 'valid'


rows = []
for name, d in runs(os.path.join(ROOT, 'data', 'raw')):
    m = NAME.search(name)
    rows.append(dict(cfg=m['cfg'], label=LABEL.get(m['cfg'], ''), rate=int(m['rate']),
                     prompt_len=int(m['len']), status=status(m['cfg']),
                     **{k: d.get(k) for k in KEYS}))
rows.sort(key=lambda r: (r['cfg'], r['prompt_len'], r['rate']))
with open(os.path.join(ROOT, 'data', 'summary.csv'), 'w', newline='') as out:
    w = csv.DictWriter(out, fieldnames=list(rows[0]))
    w.writeheader()
    w.writerows(rows)
print(f'{len(rows)} runs -> data/summary.csv')
