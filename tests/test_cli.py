

def test_cli_help_message():
    verbose = 0

    import ubelt as ub
    info = ub.cmd('python -m watch.cli --help', verbose=verbose)
    assert info['ret'] == 0

    info = ub.cmd('python -m watch.cli align --help', verbose=verbose)
    assert info['ret'] == 0

    info = ub.cmd('python -m watch.cli dvcdir --help', verbose=verbose)
    assert info['ret'] == 0

    info = ub.cmd('python -m watch.cli find_dvc --help', verbose=verbose)
    assert info['ret'] == 0

    info = ub.cmd('python -m watch.cli stats --help', verbose=verbose)
    assert info['ret'] == 0

    info = ub.cmd('python -m watch.cli add_fields --help', verbose=verbose)
    assert info['ret'] == 0

    info = ub.cmd('python -m watch.cli model_stats --help', verbose=verbose)
    assert info['ret'] == 0

    info = ub.cmd('python -m watch.cli reproject --help', verbose=verbose)
    assert info['ret'] == 0

    info = ub.cmd('python -m watch.cli average_features --help', verbose=verbose)
    assert info['ret'] == 0

    info = ub.cmd('python -m watch.cli time_combine --help', verbose=verbose)
    assert info['ret'] == 0

    info = ub.cmd('python -m watch.cli iarpa_eval --help', verbose=verbose)
    assert info['ret'] == 0

    info = ub.cmd('python -m watch.cli spectra --help', verbose=verbose)
    assert info['ret'] == 0

    info = ub.cmd('python -m watch.cli visualize --help', verbose=verbose)
    assert info['ret'] == 0
