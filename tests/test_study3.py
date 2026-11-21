"""Study 3 (Qwen3-14B, real prompts): regeneration from raw data, run checks, and every
number the report quotes from this study."""
import glob, gzip, json, os, shutil
import pandas as pd
import pytest
from conftest import S3, run, unchanged_after


def test_raw_files_present():
    assert len(glob.glob(os.path.join(S3, 'data', 'raw', '*.json.gz'))) == 504


def test_summary_and_checks_regenerate():
    unchanged_after(S3, 'summarise.py', ['data/summary.csv', 'data/checks.txt'])
    assert open(os.path.join(S3, 'data', 'checks.txt')).read() == 'all runs passed\n'


def test_goodput_and_extras_regenerate():
    unchanged_after(S3, 'goodput.py', ['data/goodput.csv'])
    unchanged_after(S3, 'extract_per_run.py', ['data/per_run_extra.csv'])


def test_tables_regenerate():
    unchanged_after(S3, 'analyse.py', ['results/tables.md', 'results/by_setting.csv'], '--no-figures')


def test_every_run_is_complete():
    s = pd.read_csv(os.path.join(S3, 'data', 'summary.csv'))
    assert len(s) == 504 and (s.checks == 'ok').all()
    assert (s.completed == 200).all() and (s.failed == 0).all()
    assert (s.groupby(['config', 'workload', 'rate']).rep.count() == 3).all()


def test_pilot_summary_regenerates():
    """The 16-run pilot is the recorded output of the end-to-end GPU workflow (PILOT=1)."""
    unchanged_after(S3, 'summarise.py', ['data/pilot/summary.csv', 'data/pilot/checks.txt'], 'data/pilot')
    assert len(pd.read_csv(os.path.join(S3, 'data', 'pilot', 'summary.csv'))) == 16


def test_broken_run_is_flagged(tmp_path):
    """A run with a failed request and a short output must be flagged, not averaged in."""
    src = os.path.join(S3, 'data', 'raw', 'fp16_len512_rate1_rep1.json.gz')
    d = json.load(gzip.open(src, 'rt'))
    d['failed'] = 1
    d['output_lens'][0] = 100
    json.dump(d, open(tmp_path / 'fp16_len512_rate1_rep1.json', 'w'))
    shutil.copy(os.path.join(S3, 'data', 'raw', 'fp16_len512_rate2_rep1.json.gz'), tmp_path)
    run(S3, 'summarise.py', str(tmp_path))
    checks = open(tmp_path / 'checks.txt').read()
    assert 'fp16_len512_rate1_rep1.json' in checks and '1 failed requests' in checks
    assert 'not 128 output tokens' in checks and 'rate2' not in checks
    s = pd.read_csv(tmp_path / 'summary.csv').set_index('rate')
    assert s.loc[1.0, 'checks'] == 'FLAGGED' and s.loc[2.0, 'checks'] == 'ok'


def test_excluded_run_is_the_one_with_a_lost_request():
    p = os.path.join(S3, 'data', 'raw', 'fp16_natural_rate1_rep1.failed199.json.gz.excluded')
    d = json.load(gzip.open(p, 'rt'))
    assert d['completed'] == 199 and d['failed'] == 1


@pytest.fixture(scope='module')
def A():
    return pd.read_csv(os.path.join(S3, 'results', 'by_setting.csv')).set_index(['workload', 'config', 'rate'])


def test_light_load_tpot(A):
    t = lambda w, c: A.loc[(w, c, 1.0), 'tpot']
    assert t('len1024', 'fp16') == pytest.approx(21.8, abs=0.05)
    assert t('len1024', 'awq') == pytest.approx(9.5, abs=0.05)
    assert t('len1024', 'spec') == pytest.approx(14.9, abs=0.05)
    assert t('len1024', 'awq_spec') == pytest.approx(13.8, abs=0.05)
    for w, s in (('natural', 2.41), ('len512', 2.34), ('len1024', 2.31), ('len2048', 1.99), ('len4096', 1.39)):
        assert t(w, 'fp16') / t(w, 'awq') == pytest.approx(s, abs=0.01)
    for w in ('natural', 'len512', 'len1024', 'len2048', 'len4096'):
        assert t(w, 'awq') < t(w, 'awq_spec')   # speculation never helps on top of AWQ
        assert t(w, 'awq') < t(w, 'spec')


def test_capacity(A):
    c = lambda w, k: A.loc[(w, k, 12.0), 'thr']
    assert c('len1024', 'fp16') == pytest.approx(5.60, abs=0.01)
    assert c('len2048', 'fp16') == pytest.approx(2.79, abs=0.01)
    assert c('len4096', 'fp16') == pytest.approx(1.45, abs=0.01)
    for w in ('len1024', 'len2048', 'len4096'):
        assert c(w, 'awq') / c(w, 'fp16') == pytest.approx(0.83, abs=0.01)
        assert c(w, 'spec') < c(w, 'fp16') and c(w, 'awq_spec') < c(w, 'awq')
        # saturated: served rate at 12/s within 3% of served rate at 8/s
        assert c(w, 'fp16') / A.loc[(w, 'fp16', 8.0), 'thr'] < 1.03


def test_prefill_rate_is_flat_across_lengths(A):
    p = [A.loc[(w, 'fp16', 12.0), 'ptok'] for w in ('len1024', 'len2048', 'len4096')]
    assert max(p) / min(p) < 1.03 and 5500 < min(p) and max(p) < 5700
    q = [A.loc[(w, 'awq', 12.0), 'ptok'] for w in ('len1024', 'len2048', 'len4096')]
    assert max(q) / min(q) < 1.03 and 4600 < min(q) and max(q) < 4750


def test_goodput(A):
    best = lambda w, c: A.loc[(w, c), 'goodput'].max()
    assert best('len1024', 'fp16') == pytest.approx(2.88, abs=0.01)
    assert best('len1024', 'awq') == pytest.approx(3.13, abs=0.01)
    assert best('len2048', 'fp16') == pytest.approx(1.21, abs=0.01)
    assert best('len2048', 'awq') == pytest.approx(1.14, abs=0.01)
    assert best('len4096', 'fp16') == pytest.approx(0.54, abs=0.01)
    for w in ('natural', 'len512', 'len1024', 'len2048', 'len4096'):
        assert best(w, 'spec') <= best(w, 'fp16') + 0.005       # speculation never raises goodput
        assert best(w, 'awq_spec') < best(w, 'awq')


def test_prefix_caching(A):
    v = lambda c, r, m: A.loc[('prefix', c, r), m]
    assert v('awq_cacheoff', 12.0, 'ttft') == pytest.approx(510, abs=1)
    assert v('awq_cacheon', 12.0, 'ttft') == pytest.approx(65, abs=1)
    assert v('awq_cacheoff', 12.0, 'tpot') == pytest.approx(91.4, abs=0.1)
    assert v('awq_cacheon', 12.0, 'tpot') == pytest.approx(17.2, abs=0.1)
    assert v('fp16_cacheoff', 12.0, 'thr') == pytest.approx(9.66, abs=0.01)
    assert v('fp16_cacheon', 12.0, 'thr') == pytest.approx(10.16, abs=0.01)
    assert v('awq_cacheon', 12.0, 'goodput') == pytest.approx(11.10, abs=0.01)
    assert A.loc[('prefix', 'awq_cacheoff'), 'goodput'].max() == pytest.approx(7.57, abs=0.01)


def test_speculative_acceptance(A):
    acc = A.loc[(slice(None), ['spec', 'awq_spec'], slice(None)), 'acc']
    assert 34 < acc.min() and acc.max() < 39


def test_speculative_speedup_fades_with_load(A):
    r = lambda w, rate: A.loc[(w, 'fp16', rate), 'tpot'] / A.loc[(w, 'spec', rate), 'tpot']
    assert [round(r('natural', x), 2) for x in (1.0, 2.0, 3.0, 4.0, 6.0, 8.0)] == [1.43, 1.37, 1.32, 1.31, 1.24, 0.98]
    assert [round(r('len512', x), 2) for x in (1.0, 2.0, 3.0, 4.0, 6.0)] == [1.50, 1.43, 1.35, 1.29, 1.12]
    assert [round(r('len1024', x), 2) for x in (1.0, 2.0, 3.0, 4.0)] == [1.46, 1.35, 1.25, 1.05]
    assert [round(r('len2048', x), 2) for x in (1.0, 2.0)] == [1.30, 1.03]
    assert round(r('len4096', 1.0), 2) == 1.00


def test_awq_and_speculative_do_not_add(A):
    t = lambda c: A.loc[('len1024', c, 1.0), 'tpot']
    assert t('spec') - t('fp16') == pytest.approx(-6.9, abs=0.05)       # speculative helps FP16
    assert t('awq_spec') - t('awq') == pytest.approx(4.4, abs=0.05)     # and hurts AWQ
    assert t('awq') - t('fp16') == pytest.approx(-12.4, abs=0.05)
    assert t('awq_spec') - t('spec') == pytest.approx(-1.1, abs=0.05)
    # AWQ's own advantage is gone before capacity: slower than FP16 at 6 requests/s
    assert A.loc[('len1024', 'awq', 6.0), 'tpot'] > A.loc[('len1024', 'fp16', 6.0), 'tpot']
    # on capacity the two costs multiply
    c = lambda w, k: A.loc[(w, k, 12.0), 'thr']
    for w in ('len1024', 'len2048', 'len4096'):
        assert c(w, 'awq_spec') / c(w, 'awq') == pytest.approx(c(w, 'spec') / c(w, 'fp16'), abs=0.015)
