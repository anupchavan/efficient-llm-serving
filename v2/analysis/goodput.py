"""v2/data/raw -> v2/data/goodput.csv: per-request SLO check for every run.

A request meets the target when its TTFT < 1 s and its TPOT < 50 ms, with
TPOT = (latency - TTFT) / (output tokens - 1), the same formula as summarise.py.
goodput = requests meeting both / run duration (requests/s).
"""
import csv, glob, json, os, re, sys

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
RAW = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, 'data', 'raw')
TTFT_S, TPOT_S = 1.0, 0.050
NAME = re.compile(r'(?P<config>.+?)_(?P<workload>natural|prefix|len\d+)_rate(?P<rate>[\d.]+)_rep(?P<rep>\d+)\.json$')

rows = []
for f in sorted(glob.glob(os.path.join(RAW, '*.json'))):
    m = NAME.search(os.path.basename(f))
    if not m:
        continue
    d = json.load(open(f))
    n = len(d['ttfts'])
    meet_ttft = meet_tpot = meet_both = 0
    for i in range(n):
        out = d['output_lens'][i]
        if out < 2:
            continue
        ttft = d['ttfts'][i]
        tpot = (d['latencies'][i] - ttft) / (out - 1)
        a, b = ttft < TTFT_S, tpot < TPOT_S
        meet_ttft += a; meet_tpot += b; meet_both += a and b
    rows.append(dict(config=m['config'], workload=m['workload'], rate=float(m['rate']),
                     rep=int(m['rep']), requests=n, duration=d['duration'],
                     meet_ttft=meet_ttft, meet_tpot=meet_tpot, meet_both=meet_both,
                     attainment=meet_both / n, goodput=meet_both / d['duration']))
if not rows:
    sys.exit(f'no result files in {RAW}')
out = os.path.join(os.path.dirname(os.path.normpath(RAW)), 'goodput.csv')
with open(out, 'w', newline='') as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0]))
    w.writeheader(); w.writerows(rows)
print(f'{len(rows)} runs -> {out}')
