import ubelt as ub
import os
import pytest


def test_top_level_cli_help_message():
    verbose = 0
    exe = 'python -m geowatch'

    env = dict(os.environ)
    env['WATCH_PREIMPORT'] = '0'

    kwargs = {
        'verbose': verbose,
        'env': env,
    }

    info = ub.cmd(f'{exe} --help', **kwargs)
    assert info['ret'] == 0


def test_subcommand_cli_help_message():
    if not int(os.environ.get('SLOW_TESTS', '0')):
        pytest.skip('skip slower test')

    verbose = 0
    exe = 'python -m geowatch'

    env = dict(os.environ)
    env['WATCH_PREIMPORT'] = '0'

    kwargs = {
        'verbose': verbose,
        'env': env,
    }

    info = ub.cmd(f'{exe} align --help', **kwargs)
    assert info['ret'] == 0

    info = ub.cmd(f'{exe} dvcdir --help', **kwargs)
    assert info['ret'] == 0

    info = ub.cmd(f'{exe} find_dvc --help', **kwargs)
    assert info['ret'] == 0

    info = ub.cmd(f'{exe} stats --help', **kwargs)
    assert info['ret'] == 0

    info = ub.cmd(f'{exe} add_fields --help', **kwargs)
    assert info['ret'] == 0

    info = ub.cmd(f'{exe} model_stats --help', **kwargs)
    assert info['ret'] == 0

    info = ub.cmd(f'{exe} reproject --help', **kwargs)
    assert info['ret'] == 0

    info = ub.cmd(f'{exe} average_features --help', **kwargs)
    assert info['ret'] == 0

    info = ub.cmd(f'{exe} time_combine --help', **kwargs)
    assert info['ret'] == 0

    info = ub.cmd(f'{exe} iarpa_eval --help', **kwargs)
    assert info['ret'] == 0

    info = ub.cmd(f'{exe} spectra --help', **kwargs)
    assert info['ret'] == 0

    info = ub.cmd(f'{exe} visualize --help', **kwargs)
    assert info['ret'] == 0
