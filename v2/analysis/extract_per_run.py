"""v2/data/raw -> v2/data/per_run_extra.csv: compact per-run numbers not in summary.csv.

TTFT and per-request TPOT percentiles are recomputed from the per-request arrays
(same request filter and TPOT definition as summarise.py). Percentiles use linear
interpolation, matching numpy.percentile's default, which vllm bench also uses.
"""
import csv, glob, json, os, re, sys

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
RAW = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, 'data', 'raw')
OUT = sys.argv[2] if len(sys.argv) > 2 else os.path.join(ROOT, 'data', 'per_run_extra.csv')
NAME = re.compile(r'(?P<config>.+?)_(?P<workload>natural|prefix|len\d+)_rate(?P<rate>[\d.]+)_rep(?P<rep>\d+)\.json$')
KEYS = ['total_input_tokens', 'total_output_tokens', 'duration', 'completed', 'failed',
        'request_throughput']


def pct(xs, q):
    xs = sorted(xs)
    if not xs:
        return None
    k = (len(xs) - 1) * q / 100
    lo = int(k)
    hi = min(lo + 1, len(xs) - 1)
    return xs[lo] + (xs[hi] - xs[lo]) * (k - lo)


rows = []
for f in sorted(glob.glob(os.path.join(RAW, '*.json'))):
    m = NAME.search(os.path.basename(f))
    if not m:
        continue
    d = json.load(open(f))
    ok = [i for i, n in enumerate(d.get('output_lens', [])) if n > 0]
    ttft = [d['ttfts'][i] * 1000 for i in ok]
    tpot = [(d['latencies'][i] - d['ttfts'][i]) / (d['output_lens'][i] - 1) * 1000
            for i in ok if d['output_lens'][i] > 1]
    row = dict(config=m['config'], workload=m['workload'], rate=float(m['rate']), rep=int(m['rep']))
    row.update({k: d.get(k) for k in KEYS})
    for name, xs in (('ttft', ttft), ('tpot', tpot)):
        for q, label in ((50, 'median'), (90, 'p90'), (99, 'p99')):
            row[f'{label}_{name}_ms'] = pct(xs, q)
    row['mean_tpot_ms'] = d.get('mean_tpot_ms')
    row['spec_decode_acceptance_rate'] = d.get('spec_decode_acceptance_rate')
    row['spec_decode_acceptance_length'] = d.get('spec_decode_acceptance_length')
    rows.append(row)

if not rows:
    sys.exit(f'no result files in {RAW}')
with open(OUT, 'w', newline='') as out:
    w = csv.DictWriter(out, fieldnames=list(rows[0]))
    w.writeheader()
    w.writerows(rows)
print(f'{len(rows)} runs -> {OUT}')
