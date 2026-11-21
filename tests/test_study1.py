"""Study 1 (Qwen2.5-3B, 1,024-token prompts): data integrity and the numbers quoted from it."""
import glob, os
import pandas as pd
import pytest
from conftest import S1, unchanged_after


def test_summary_regenerates_from_raw():
    assert len(glob.glob(os.path.join(S1, 'data', 'raw', '*.json'))) == 201
    unchanged_after(S1, 'summarise.py', ['data/summary.csv'])


def test_quoted_numbers():
    d = pd.read_csv(os.path.join(S1, 'data', 'summary.csv'))
    assert (d.input_tokens_per_request == 1024).all()
    tpot = d.groupby(['config', 'rate']).mean_tpot_ms.mean()
    speedup = lambda c, r: tpot[('baseline', r)] / tpot[(c, r)]
    assert speedup('awq', 2) == pytest.approx(2.46, abs=0.02)
    assert speedup('awq', 4) == pytest.approx(2.52, abs=0.02)
    assert speedup('spec', 2) == pytest.approx(1.56, abs=0.02)
    assert speedup('spec', 4) == pytest.approx(0.98, abs=0.02)
    assert speedup('awq_spec', 2) == pytest.approx(2.11, abs=0.02)
    served = d[d.rate >= 16].groupby('config').request_throughput.mean()
    assert served['baseline'] == pytest.approx(3.99, abs=0.02)
    assert served['spec'] == pytest.approx(4.42, abs=0.02)
