"""Figures and the numbers quoted in the report, from data/summary.csv.

Run from the repository root:  python3 analysis/summarise.py && python3 analysis/plots.py
"""
import csv, os, statistics as st
from collections import defaultdict
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, '..', 'figures')
os.makedirs(OUT, exist_ok=True)

CONFIGS = [  # regime key, label, colour
    ('baseline', 'Baseline (FP16)', '#52514e'),
    ('awq', 'AWQ 4-bit weights', '#2a78d6'),
    ('spec', 'Speculative decoding', '#eb6834'),
    ('awq_spec', 'AWQ + speculative', '#1baf7a'),
]
INK, MUTED, GRID = '#0b0b0b', '#52514e', '#e4e3df'
plt.rcParams.update({'font.family': 'Helvetica', 'font.size': 9, 'axes.edgecolor': MUTED,
                     'axes.labelcolor': INK, 'xtick.color': MUTED, 'ytick.color': MUTED,
                     'axes.spines.top': False, 'axes.spines.right': False,
                     'axes.grid': True, 'grid.color': GRID, 'grid.linewidth': .6,
                     'legend.frameon': False, 'pdf.fonttype': 42})

runs = defaultdict(list)  # (config, rate) -> runs
for r in csv.DictReader(open(os.path.join(HERE, '..', 'data', 'summary.csv'))):
    runs[(r['config'], int(r['rate']))].append({k: float(v) for k, v in r.items()
                                                 if k != 'config' and v})
RATES = [2, 4, 8, 12, 16, 24, 32, 48, 64]

def stat(regime, rate, key):
    v = [r[key] for r in runs[(regime, rate)]]
    return (st.median(v), min(v), max(v)) if v else (None, None, None)

# ---- Figure 1: the GPU saturates at ~4 req/s -------------------------------------------
fig, (a, b) = plt.subplots(1, 2, figsize=(8.2, 2.9))
for key, label, c in CONFIGS:
    a.plot(RATES, [stat(key, x, 'request_throughput')[0] for x in RATES], '-o', color=c, lw=2, ms=4, label=label)
    b.plot(RATES, [stat(key, x, 'median_ttft_ms')[0] / 1000 for x in RATES], '-o', color=c, lw=2, ms=4)
a.plot([2, 8], [2, 8], ':', color=MUTED, lw=1)
a.text(5.6, 6.6, 'served = sent', color=MUTED, fontsize=7.5, rotation=0)
for ax in (a, b):
    ax.set_xscale('log', base=2); ax.set_xticks(RATES); ax.set_xticklabels(RATES)
    ax.set_xlabel('Requests sent per second')
a.set_ylim(0, 8); a.set_ylabel('Requests served per second')
a.set_title('Throughput', fontsize=9.5, color=INK, loc='left')
b.set_yscale('log'); b.set_ylabel('Median time to first token (s)')
b.set_yticks([0.1, 1, 10, 100]); b.set_yticklabels(['0.1', '1', '10', '100'])
b.set_title('Time to first token', fontsize=9.5, color=INK, loc='left')
fig.legend(loc='lower center', ncol=4, bbox_to_anchor=(.5, -.02), fontsize=8)
fig.tight_layout(rect=(0, .07, 1, 1))
fig.savefig(os.path.join(OUT, 'saturation.pdf')); fig.savefig(os.path.join(OUT, 'saturation.png'), dpi=160)

# ---- Figure 2: below saturation, per-token speed and first-token time ------------------
fig, (a, b) = plt.subplots(1, 2, figsize=(8.2, 2.9))
loads, w = [2, 4], .19
for i, (key, label, c) in enumerate(CONFIGS):
    for ax, metric, scale in ((a, 'mean_tpot_ms', 1), (b, 'median_ttft_ms', 1)):
        xs = [j + (i - 1.5) * (w + .02) for j in range(len(loads))]
        s = [stat(key, x, metric) for x in loads]
        ax.bar(xs, [m for m, _, _ in s], w, color=c, label=label if ax is a else None,
               yerr=[[m - lo for m, lo, _ in s], [hi - m for m, _, hi in s]],
               error_kw=dict(ecolor=INK, lw=.8, capsize=2))
        for x, (m, _, hi) in zip(xs, s):
            ax.text(x, hi + (1 if ax is a else 15), f'{m:.0f}', ha='center', va='bottom', fontsize=7, color=INK)
for ax in (a, b):
    ax.set_xticks(range(len(loads))); ax.set_xticklabels([f'{x} requests/s' for x in loads])
    ax.grid(axis='x', visible=False)
a.set_ylabel('Time per output token (ms)'); a.set_ylim(0, 68)
a.set_title('TPOT (lower is better)', fontsize=9.5, color=INK, loc='left')
b.set_ylabel('Median time to first token (ms)'); b.set_ylim(0, 1100)
b.set_title('TTFT (lower is better)', fontsize=9.5, color=INK, loc='left')
fig.legend(loc='lower center', ncol=4, bbox_to_anchor=(.5, -.02), fontsize=8)
fig.tight_layout(rect=(0, .07, 1, 1))
fig.savefig(os.path.join(OUT, 'low_load.pdf')); fig.savefig(os.path.join(OUT, 'low_load.png'), dpi=160)

# ---- Numbers quoted on slides ----------------------------------------------------------
for rate in (2, 4, 16, 64):
    for key, label, _ in CONFIGS:
        t, ft, tp = (stat(key, rate, k)[0] for k in ('mean_tpot_ms', 'median_ttft_ms', 'request_throughput'))
        acc = stat(key, rate, 'spec_decode_acceptance_rate')[0] if 'spec' in key else None
        print(f'rate {rate:>2} {key:9} n={len(runs[(key, rate)])} TPOT {t:6.1f} ms  TTFT {ft:9.0f} ms  served {tp:.2f}/s'
              + (f'  accept {acc:.1f}%' if acc else ''))
