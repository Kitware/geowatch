

def test_cli_help_message():
    verbose = 3
    import ubelt as ub
    exe = 'python -m watch'

    info = ub.cmd(f'{exe} --help', verbose=verbose)
    assert info['ret'] == 0

    info = ub.cmd(f'{exe} align --help', verbose=verbose)
    assert info['ret'] == 0

    info = ub.cmd(f'{exe} dvcdir --help', verbose=verbose)
    assert info['ret'] == 0

    info = ub.cmd(f'{exe} find_dvc --help', verbose=verbose)
    assert info['ret'] == 0

    info = ub.cmd(f'{exe} stats --help', verbose=verbose)
    assert info['ret'] == 0

    info = ub.cmd(f'{exe} add_fields --help', verbose=verbose)
    assert info['ret'] == 0

    info = ub.cmd(f'{exe} model_stats --help', verbose=verbose)
    assert info['ret'] == 0

    info = ub.cmd(f'{exe} reproject --help', verbose=verbose)
    assert info['ret'] == 0

    info = ub.cmd(f'{exe} average_features --help', verbose=verbose)
    assert info['ret'] == 0

    info = ub.cmd(f'{exe} time_combine --help', verbose=verbose)
    assert info['ret'] == 0

    info = ub.cmd(f'{exe} iarpa_eval --help', verbose=verbose)
    assert info['ret'] == 0

    info = ub.cmd(f'{exe} spectra --help', verbose=verbose)
    assert info['ret'] == 0

    info = ub.cmd(f'{exe} visualize --help', verbose=verbose)
    assert info['ret'] == 0
