def test_legacy_cold_invocation():
    """"
    There was an issue in redoing eval18 with cold_config=None and with_cold=1.
    Test for it.
    """
    from geowatch.cli.queue_cli.prepare_teamfeats import prep_feats
    import ubelt as ub
    cold_config = None
    config = {
        'src_kwcocos': './PRETEND_BUNDLE/data.kwcoco.json',
        'gres': [0, 1],
        'expt_dvc_dpath': './PRETEND_EXPT_DVC',
        'virtualenv_cmd': 'conda activate geowatch',
        'cold_config': cold_config,
        'with_cold': 1,
        'run': 0,
        'check': False,
        'skip_existing': False,
        'backend': 'serial',
    }
    config['backend'] = 'serial'
    outputs = prep_feats(cmdline=False, **config)
    outputs['queue'].print_commands(0, 0)
    output_paths = outputs['final_output_paths']
    print('output_paths = {}'.format(ub.urepr(output_paths, nl=1)))

    assert len(output_paths) == 1
    assert '_C.kwcoco.zip' in output_paths[0].name, (
        'if the naming convention changes this might break, and we could remove this test.')
