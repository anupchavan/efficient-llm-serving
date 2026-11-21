import os, subprocess, sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
S1 = os.path.join(ROOT, 'evaluation', '01_qwen2.5-3b_1024tok')
S2 = os.path.join(ROOT, 'evaluation', '02_qwen2.5-3b_lengths')
S3 = os.path.join(ROOT, 'evaluation', '03_qwen3-14b_sharegpt')


def run(study, script, *args):
    """Run an analysis script the way the README says to; fail the test if it fails."""
    r = subprocess.run([sys.executable, os.path.join('analysis', script), *args],
                       cwd=study, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    return r


def unchanged_after(study, script, files, *args):
    """The committed output files must be reproduced byte-for-byte from the raw data."""
    before = {f: open(os.path.join(study, f), 'rb').read() for f in files}
    run(study, script, *args)
    for f in files:
        assert open(os.path.join(study, f), 'rb').read() == before[f], f'{f} changed when regenerated'
