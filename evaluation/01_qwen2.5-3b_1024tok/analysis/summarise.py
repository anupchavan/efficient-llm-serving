"""Collect every result JSON in data/raw into data/summary.csv (one row per run)."""
import csv, glob, json, os, re

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
KEYS = ['completed', 'failed', 'request_throughput', 'output_throughput',
        'mean_ttft_ms', 'median_ttft_ms', 'p99_ttft_ms', 'mean_tpot_ms', 'median_tpot_ms',
        'mean_itl_ms', 'median_itl_ms', 'max_concurrent_requests', 'duration',
        'spec_decode_acceptance_rate', 'spec_decode_acceptance_length']

rows = []
for f in glob.glob(os.path.join(ROOT, 'data', 'raw', '*.json')):
    config, rate, rep = re.match(r'(.+)_rate(\d+)_rep(\d+)\.json', os.path.basename(f)).groups()
    d = json.load(open(f))
    rows.append(dict(config=config, rate=int(rate), rep=int(rep),
                     input_tokens_per_request=d['total_input_tokens'] // d['completed'],
                     **{k: d.get(k) for k in KEYS}))

rows.sort(key=lambda r: (r['config'], r['rate'], r['rep']))
with open(os.path.join(ROOT, 'data', 'summary.csv'), 'w', newline='') as out:
    w = csv.DictWriter(out, fieldnames=rows[0].keys())
    w.writeheader()
    w.writerows(rows)
print(f'{len(rows)} runs -> data/summary.csv')
