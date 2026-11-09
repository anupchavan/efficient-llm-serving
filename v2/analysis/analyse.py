"""v2/data/*.csv -> v2/results/tables.md and v2/figures/*.png.

Every number is a mean over the 3 repeats, with the min-max range across repeats
where it matters. Nothing here reads a run that summarise.py flagged.
"""
import os
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
D = lambda *p: os.path.join(ROOT, *p)
CONFIGS = ['fp16', 'awq', 'spec', 'awq_spec']
LABEL = {'fp16': 'FP16 (BF16)', 'awq': 'AWQ', 'spec': 'Speculative', 'awq_spec': 'AWQ + speculative',
         'fp16_cacheoff': 'FP16, cache off', 'fp16_cacheon': 'FP16, cache on',
         'awq_cacheoff': 'AWQ, cache off', 'awq_cacheon': 'AWQ, cache on'}
COLOR = {'fp16': '#4c6ef5', 'awq': '#e8590c', 'spec': '#2f9e44', 'awq_spec': '#ae3ec9'}
WORKLOADS = ['natural', 'len512', 'len1024', 'len2048', 'len4096']

s = pd.read_csv(D('data', 'summary.csv'))
assert (s.checks == 'ok').all(), 'flagged runs present'
assert len(s) == 504
x = pd.read_csv(D('data', 'per_run_extra.csv'))
s = s.merge(x[['config', 'workload', 'rate', 'rep', 'total_input_tokens', 'p90_ttft_ms', 'p90_tpot_ms']],
            on=['config', 'workload', 'rate', 'rep'], validate='1:1')
s['prompt_tok_s'] = s.total_input_tokens / s.duration
gp_path = D('data', 'goodput.csv')
if os.path.exists(gp_path):
    g = pd.read_csv(gp_path)
    s = s.merge(g[['config', 'workload', 'rate', 'rep', 'goodput', 'attainment']],
                on=['config', 'workload', 'rate', 'rep'], validate='1:1')

agg = s.groupby(['workload', 'config', 'rate']).agg(
    n=('rep', 'count'),
    tpot=('median_tpot_ms', 'mean'), tpot_lo=('median_tpot_ms', 'min'), tpot_hi=('median_tpot_ms', 'max'),
    ttft=('median_ttft_ms', 'mean'), ttft_lo=('median_ttft_ms', 'min'), ttft_hi=('median_ttft_ms', 'max'),
    thr=('request_throughput', 'mean'), thr_lo=('request_throughput', 'min'), thr_hi=('request_throughput', 'max'),
    ptok=('prompt_tok_s', 'mean'), inmed=('input_tokens_median', 'mean'),
    acc=('spec_decode_acceptance_rate', 'mean'), acclen=('spec_decode_acceptance_length', 'mean'),
    **({'goodput': ('goodput', 'mean'), 'gp_lo': ('goodput', 'min'), 'gp_hi': ('goodput', 'max'),
        'att': ('attainment', 'mean')} if 'goodput' in s else {})).reset_index()
assert (agg.n == 3).all()
A = agg.set_index(['workload', 'config', 'rate'])
out = []
P = out.append


def table(head, rows):
    P('| ' + ' | '.join(head) + ' |')
    P('|' + '---|' * len(head))
    for r in rows:
        P('| ' + ' | '.join(r) + ' |')
    P('')


def saturated(w, c):
    # served at 12/s within 3% of served at 8/s: the curve has flattened
    return A.loc[(w, c, 12.0), 'thr'] / A.loc[(w, c, 8.0), 'thr'] < 1.03


def near(w, c, rate):
    return rate > 0.8 * A.loc[(w, c, 12.0), 'thr']


# 1. light load
P('## 1. Light load: median TPOT and TTFT at 1 request/s\n')
P('Mean of 3 repeats; range across repeats in brackets. Speed-up = FP16 TPOT / config TPOT. '
  '* = 1 request/s is above 80% of that config\'s capacity, so this is not light load for it.\n')
rows = []
for w in WORKLOADS:
    base = A.loc[(w, 'fp16', 1.0), 'tpot']
    for c in CONFIGS:
        r = A.loc[(w, c, 1.0)]
        rows.append([w, LABEL[c] + (' *' if near(w, c, 1.0) else ''), f'{r.tpot:.2f} [{r.tpot_lo:.2f}–{r.tpot_hi:.2f}]', f'{base / r.tpot:.2f}×',
                     f'{r.ttft:.0f} [{r.ttft_lo:.0f}–{r.ttft_hi:.0f}]'])
table(['Workload', 'Config', 'TPOT ms', 'TPOT speed-up', 'TTFT ms'], rows)

P('### TPOT speed-up over FP16 at 1, 2 and 4 requests/s\n')
P('A cell is marked * when the rate is above 80% of the capacity of that config or of FP16 (section 2), '
  'so TPOT there includes queueing and is not a clean decode-speed comparison.\n')
rows = []
for w in WORKLOADS:
    for c in CONFIGS[1:]:
        cells = []
        for rate in (1.0, 2.0, 4.0):
            r, b = A.loc[(w, c, rate)], A.loc[(w, 'fp16', rate)]
            star = '*' if near(w, c, rate) or near(w, 'fp16', rate) else ''
            cells.append(f'{b.tpot / r.tpot:.2f}×{star}')
        rows.append([w, LABEL[c]] + cells)
table(['Workload', 'Config', '1/s', '2/s', '4/s'], rows)

# 2. capacity
P('## 2. Capacity: requests/s served at 12 requests/s sent\n')
P('Mean [min–max] over 3 repeats. "not saturated" = served at 12/s is more than 3% above served at 8/s, '
  'so the curve has not flattened and the number is a lower bound on capacity.\n')
rows = []
for w in WORKLOADS:
    cells = []
    for c in CONFIGS:
        r = A.loc[(w, c, 12.0)]
        tag = '' if saturated(w, c) else ' (not saturated)'
        cells.append(f'{r.thr:.2f} [{r.thr_lo:.2f}–{r.thr_hi:.2f}]{tag}')
    rows.append([w] + cells)
table(['Workload'] + [LABEL[c] for c in CONFIGS], rows)
rows = []
for w in WORKLOADS:
    b = A.loc[(w, 'fp16', 12.0), 'thr']
    rows.append([w] + [f'{A.loc[(w, c, 12.0), "thr"] / b:.2f}×' for c in CONFIGS])
P('Capacity relative to FP16:\n')
table(['Workload'] + [LABEL[c] for c in CONFIGS], rows)

# 3. what sets the limit
P('## 3. What sets the limit: prompt tokens/s at 12 requests/s\n')
P('prompt tokens/s = total prompt tokens / run duration. If it stays flat across prompt lengths, prefill compute is the limit.\n')
rows = []
for w in WORKLOADS:
    rows.append([w, f'{A.loc[(w, "fp16", 12.0), "inmed"]:.0f}'] +
                [f'{A.loc[(w, c, 12.0), "ptok"]:,.0f}' for c in CONFIGS])
table(['Workload', 'Median prompt tokens'] + [LABEL[c] for c in CONFIGS], rows)

# 4. goodput
if 'goodput' in agg:
    P('## 4. Goodput: requests/s with TTFT < 1 s and TPOT < 50 ms (per request)\n')
    P('Best goodput over the tested rates (mean of 3 repeats at that rate, [min–max]), and the rate where it occurs.\n')
    rows = []
    for w in WORKLOADS + ['prefix']:
        cfgs = CONFIGS if w != 'prefix' else ['fp16_cacheoff', 'fp16_cacheon', 'awq_cacheoff', 'awq_cacheon']
        cells = []
        for c in cfgs:
            sub = agg[(agg.workload == w) & (agg.config == c)]
            best = sub.loc[sub.goodput.idxmax()]
            cells.append(f'{best.goodput:.2f} [{best.gp_lo:.2f}–{best.gp_hi:.2f}] @ {best.rate:g}/s')
        rows.append([w] + cells + [''] * (4 - len(cells)))
    table(['Workload', 'FP16 / FP16 cache off', 'AWQ / FP16 cache on', 'Spec / AWQ cache off', 'AWQ+spec / AWQ cache on'], rows)

# 5. prefix caching
P('## 5. Prefix caching off vs on (shared role prompts)\n')
rows = []
for q in ('fp16', 'awq'):
    for rate in (1.0, 4.0, 8.0, 12.0):
        off, on = A.loc[('prefix', f'{q}_cacheoff', rate)], A.loc[('prefix', f'{q}_cacheon', rate)]
        rows.append([LABEL[q], f'{rate:g}', f'{off.ttft:.0f} → {on.ttft:.0f}', f'{off.tpot:.1f} → {on.tpot:.1f}',
                     f'{off.thr:.2f} → {on.thr:.2f}'])
table(['Weights', 'Rate /s', 'TTFT ms (off → on)', 'TPOT ms (off → on)', 'Served /s (off → on)'], rows)

# 6. acceptance
P('## 6. Speculative decoding acceptance (Qwen3-0.6B draft, 5 tokens)\n')
rows = []
for w in WORKLOADS:
    sub = agg[(agg.workload == w) & agg.config.isin(['spec', 'awq_spec'])]
    for c in ('spec', 'awq_spec'):
        t = sub[sub.config == c]
        rows.append([w, LABEL[c], f'{t.acc.min():.1f}–{t.acc.max():.1f}%', f'{t.acclen.min():.2f}–{t.acclen.max():.2f}'])
table(['Workload', 'Config', 'Acceptance rate (range over rates)', 'Mean accepted length'], rows)

# 7. repeat spread
P('## 7. Spread across repeats\n')
agg['tpot_spread'] = (agg.tpot_hi - agg.tpot_lo) / agg.tpot * 100
agg['thr_spread'] = (agg.thr_hi - agg.thr_lo) / agg.thr * 100
P(f'Over all {len(agg)} settings: TPOT (max−min)/mean median {agg.tpot_spread.median():.1f}%, '
  f'90th pct {agg.tpot_spread.quantile(.9):.1f}%, max {agg.tpot_spread.max():.1f}%; '
  f'served/s median {agg.thr_spread.median():.1f}%, max {agg.thr_spread.max():.1f}%.\n')
worst = agg.sort_values('tpot_spread', ascending=False).head(5)
table(['Workload', 'Config', 'Rate', 'TPOT spread %'],
      [[r.workload, r.config, f'{r.rate:g}', f'{r.tpot_spread:.1f}'] for r in worst.itertuples()])

open(D('results', 'tables.md'), 'w').write('# v2 sweep tables (generated by analysis/analyse.py)\n\n' + '\n'.join(out))
agg.to_csv(D('results', 'by_setting.csv'), index=False)

# figures
plt.rcParams.update({'font.size': 9, 'axes.spines.top': False, 'axes.spines.right': False})
fig, ax = plt.subplots(1, 3, figsize=(12, 3.6))
for c in CONFIGS:
    t = agg[(agg.config == c) & (agg.rate == 1.0)].set_index('workload').loc[WORKLOADS]
    ax[0].plot(range(5), t.tpot, 'o-', color=COLOR[c], label=LABEL[c])
    ax[1].plot(range(5), t.ttft, 'o-', color=COLOR[c])
    t12 = agg[(agg.config == c) & (agg.rate == 12.0)].set_index('workload').loc[WORKLOADS[1:]]
    ax[2].plot(range(4), t12.thr, 'o-', color=COLOR[c])
for a, yl in zip(ax, ['Median TPOT (ms) at 1 req/s', 'Median TTFT (ms) at 1 req/s', 'Served req/s at 12 req/s sent']):
    a.set_ylabel(yl)
ax[0].set_xticks(range(5), WORKLOADS); ax[1].set_xticks(range(5), WORKLOADS)
ax[2].set_xticks(range(4), WORKLOADS[1:]); ax[2].set_ylim(0, 10)
ax[0].legend(frameon=False, fontsize=8)
fig.suptitle('Qwen3-14B on one A100 80GB, real prompts (mean of 3 repeats)')
fig.tight_layout(); fig.savefig(D('figures', 'overview.png'), dpi=160)

fig, ax = plt.subplots(2, 5, figsize=(15, 5.6), sharex=True)
for j, w in enumerate(WORKLOADS):
    for c in CONFIGS:
        t = agg[(agg.workload == w) & (agg.config == c)]
        ax[0, j].errorbar(t.rate, t.thr, yerr=[t.thr - t.thr_lo, t.thr_hi - t.thr], fmt='o-', ms=3, color=COLOR[c], label=LABEL[c])
        ax[1, j].plot(t.rate, t.ttft / 1000, 'o-', ms=3, color=COLOR[c])
    ax[0, j].plot([0, 12], [0, 12], ':', color='grey', lw=.8)
    ax[0, j].set_title(w); ax[1, j].set_yscale('log'); ax[1, j].set_xlabel('Requests/s sent')
    ax[1, j].axhline(1, ls=':', color='grey', lw=.8)
ax[0, 0].set_ylabel('Served req/s'); ax[1, 0].set_ylabel('Median TTFT (s)')
ax[0, 0].legend(frameon=False, fontsize=7)
fig.tight_layout(); fig.savefig(D('figures', 'saturation.png'), dpi=160)

fig, ax = plt.subplots(1, 2, figsize=(9, 3.4))
for q, col in (('fp16', COLOR['fp16']), ('awq', COLOR['awq'])):
    for cache, ls in (('cacheoff', '--'), ('cacheon', '-')):
        t = agg[(agg.workload == 'prefix') & (agg.config == f'{q}_{cache}')]
        ax[0].plot(t.rate, t.ttft, 'o' + ls, ms=3, color=col, label=LABEL[f'{q}_{cache}'])
        ax[1].plot(t.rate, t.tpot, 'o' + ls, ms=3, color=col)
ax[0].set_ylabel('Median TTFT (ms)'); ax[1].set_ylabel('Median TPOT (ms)')
for a in ax: a.set_xlabel('Requests/s sent')
ax[0].legend(frameon=False, fontsize=8)
fig.suptitle('Prefix caching, shared role prompts (mean of 3 repeats)')
fig.tight_layout(); fig.savefig(D('figures', 'prefix.png'), dpi=160)
print('wrote results/tables.md, results/by_setting.csv, figures/*.png')
