"""v2/data/raw -> v2/data/summary.csv, with checks on every run.

For each run the median TTFT and TPOT are recomputed from the per-request
arrays (saved with --save-detailed) and compared with what vllm bench reported.
Any run that fails a check is listed in v2/data/checks.txt and flagged in the
summary, so it is never used without being looked at first.
"""
import csv, glob, json, os, re, statistics, sys

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
RAW = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, 'data', 'raw')
OUTPUT_LEN = 128
KEYS = ['completed', 'failed', 'duration', 'request_throughput', 'output_throughput',
        'mean_ttft_ms', 'median_ttft_ms', 'p99_ttft_ms', 'mean_tpot_ms', 'median_tpot_ms',
        'p99_tpot_ms', 'median_itl_ms', 'max_concurrent_requests',
        'spec_decode_acceptance_rate', 'spec_decode_acceptance_length']
NAME = re.compile(r'(?P<config>.+?)_(?P<workload>natural|prefix|len\d+)_rate(?P<rate>[\d.]+)_rep(?P<rep>\d+)\.json$')


def close(a, b, rel=1e-6):
    return a is not None and b is not None and abs(a - b) <= rel * max(1.0, abs(b))


def check(d):
    problems = []
    ok = [i for i, n in enumerate(d.get('output_lens', [])) if n > 0]
    if not d.get('ttfts'):
        return ['no per-request data (run without --save-detailed)'], {}
    if d['failed']:
        problems.append(f"{d['failed']} failed requests")
    wrong = [d['output_lens'][i] for i in ok if d['output_lens'][i] != OUTPUT_LEN]
    if wrong:
        problems.append(f'{len(wrong)} requests not {OUTPUT_LEN} output tokens')
    ttft = [d['ttfts'][i] * 1000 for i in ok]
    tpot = [(d['latencies'][i] - d['ttfts'][i]) / (d['output_lens'][i] - 1) * 1000
            for i in ok if d['output_lens'][i] > 1]
    mine = {'median_ttft_ms': statistics.median(ttft), 'median_tpot_ms': statistics.median(tpot)}
    for k, v in mine.items():
        if not close(v, d.get(k)):
            problems.append(f'{k}: recomputed {v:.3f}, reported {d.get(k)}')
    ins = [d['input_lens'][i] for i in ok]
    extra = {'input_tokens_mean': round(statistics.mean(ins), 1),
             'input_tokens_median': statistics.median(ins)}
    return problems, extra


rows, report = [], []
for f in sorted(glob.glob(os.path.join(RAW, '*.json'))):
    m = NAME.search(os.path.basename(f))
    if not m:
        continue
    d = json.load(open(f))
    problems, extra = check(d)
    rows.append(dict(config=m['config'], workload=m['workload'], rate=float(m['rate']),
                     rep=int(m['rep']), **extra, **{k: d.get(k) for k in KEYS},
                     checks='ok' if not problems else 'FLAGGED'))
    if problems:
        report.append(f"{os.path.basename(f)}: " + '; '.join(problems))

if not rows:
    sys.exit(f'no result files in {RAW}')
fields = list(dict.fromkeys(k for r in rows for k in r))
out_dir = os.path.dirname(os.path.normpath(RAW))
with open(os.path.join(out_dir, 'summary.csv'), 'w', newline='') as out:
    w = csv.DictWriter(out, fieldnames=fields)
    w.writeheader()
    w.writerows(rows)
with open(os.path.join(out_dir, 'checks.txt'), 'w') as out:
    out.write('\n'.join(report) + '\n' if report else 'all runs passed\n')
print(f'{len(rows)} runs, {len(report)} flagged -> {out_dir}/summary.csv, checks.txt')
