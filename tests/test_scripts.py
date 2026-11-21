"""The benchmark scripts need a GPU, so here they are only checked for syntax and for
the server flags each configuration is documented to use."""
import glob, os, subprocess
from conftest import ROOT, S3


def test_shell_scripts_parse():
    scripts = glob.glob(os.path.join(ROOT, 'evaluation', '*', 'scripts', '*.sh'))
    assert len(scripts) == 4
    for s in scripts:
        assert subprocess.run(['bash', '-n', s]).returncode == 0, s


def test_study3_main_configurations_run_with_prefix_caching_off():
    src = open(os.path.join(S3, 'scripts', 'run_sweep.sh')).read()
    body = src[src.index('args_of()'):src.index('start_server()')]
    for line in body.splitlines():
        name = line.strip().split(')')[0]
        if name in ('fp16', 'awq', 'spec', 'awq_spec', 'fp16_cacheoff', 'awq_cacheoff'):
            assert '$NO_CACHE' in line and '$CACHE' not in line.replace('$NO_CACHE', ''), line
        if name in ('fp16_cacheon', 'awq_cacheon'):
            assert '$CACHE' in line and '$NO_CACHE' not in line, line


def test_recorded_server_flags_match():
    """What each server logged at start-up agrees with the configuration name."""
    for name, want in (('fp16', "'enable_prefix_caching': False"), ('awq', "'quantization': 'awq'"),
                       ('spec', "'num_speculative_tokens': 5"), ('awq_cacheon', "'enable_prefix_caching': True"),
                       ('fp16_cacheoff', "'enable_prefix_caching': False")):
        assert want in open(os.path.join(S3, 'data', f'server_{name}.txt')).read(), name
