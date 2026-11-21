"""Study 2 (Qwen2.5-3B, six prompt lengths): the exclusion rules and the numbers quoted from it."""
import glob, os
import pandas as pd
import pytest
from conftest import S2, unchanged_after


@pytest.fixture(scope='module')
def d():
    unchanged_after(S2, 'summarise.py', ['data/summary.csv'])
    return pd.read_csv(os.path.join(S2, 'data', 'summary.csv'))


def test_all_864_runs_present(d):
    assert len(glob.glob(os.path.join(S2, 'data', 'raw', '*.json.gz'))) == 864
    assert len(d) == 864 and d.cfg.nunique() == 16


def test_fp8_runs_completed_nothing(d):
    kv = d[d.cfg.str.contains('KV')]
    assert len(kv) == 432 and kv.completed.sum() == 0
    assert kv.status.str.startswith('excluded').all()


def test_awq_caching_runs_overlapped(d):
    a = d[d.cfg == 'WQ_CR'].set_index(['rate', 'prompt_len']).date
    b = d[d.cfg == 'WQ_CR_MT'].set_index(['rate', 'prompt_len']).date
    assert abs(int(a.min()[-6:]) - int(b.min()[-6:])) <= 1 and a.min()[:8] == b.min()[:8]  # started 1 s apart
    # the run labelled speculative has no speculation statistics
    assert d[d.cfg == 'WQ_CR_MT'].spec_decode_acceptance_rate.isna().all()
    assert d[d.cfg == 'WQ_MT'].spec_decode_acceptance_rate.notna().all()


def test_baseline_disagrees_with_identical_config(d):
    cap = lambda c: d[(d.cfg == c) & (d.prompt_len == 1024)].request_throughput.max()
    assert cap('baseline') < 0.5 * cap('CR')


def test_used_runs_are_complete(d):
    used = d[d.label.notna()]
    assert len(used) == 216 and (used.failed == 0).all() and (used.completed == 992).all()


def test_tables_regenerate_and_quoted_numbers(d):
    unchanged_after(S2, 'tables.py', ['results/tables.md'])
    A = d[d.label.notna()].set_index(['label', 'prompt_len', 'rate'])
    t = lambda c: A.loc[(c, 1024, 2), 'median_tpot_ms']
    assert t('fp16') / t('awq') == pytest.approx(2.63, abs=0.02)
    assert t('fp16') / t('spec') == pytest.approx(2.09, abs=0.02)
    assert t('awq_spec') == pytest.approx(7.1, abs=0.05)
    cap = d[d.label.notna() & (d.rate >= 16)].groupby(['label', 'prompt_len']).request_throughput.mean()
    assert cap[('fp16', 1024)] == pytest.approx(7.13, abs=0.02)
    assert cap[('spec', 2048)] / cap[('fp16', 2048)] == pytest.approx(1.26, abs=0.02)
