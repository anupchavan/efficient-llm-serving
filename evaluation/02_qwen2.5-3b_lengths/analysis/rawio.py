"""Read the run files in a raw-data folder: <name>.json or <name>.json.gz."""
import glob, gzip, json, os


def runs(folder):
    """Yield (file name without .gz, parsed JSON) for every run file, sorted by name."""
    paths = glob.glob(os.path.join(folder, '*.json')) + glob.glob(os.path.join(folder, '*.json.gz'))
    for p in sorted(paths, key=lambda p: os.path.basename(p).removesuffix('.gz')):
        opener = gzip.open if p.endswith('.gz') else open
        with opener(p, 'rt') as fh:
            text = fh.read()
        try:
            d = json.loads(text)
        except json.JSONDecodeError:
            # One file (baseline, rate 12, len 4096) was saved corrupt. It is kept in
            # data/raw as received and reported as unreadable; its config is excluded anyway.
            d = {'unreadable': True}
        yield os.path.basename(p).removesuffix('.gz'), d
