import ubelt as ub


def test_init_from_pretrained_state():
    # Train a demo model (in the future grab a pretrained demo model)
    import ubelt as ub
    import kwcoco

    import pytest
    pytest.skip('slow test: switch to lightning')

    from geowatch.tasks.fusion.fit import fit_model  # NOQA
    # import xdev
    # xdev.make_warnings_print_tracebacks()

    # args = None
    # cmdline = False
    gpus = None
    test_dpath = ub.Path.appdir('geowatch/tests/fusion/').ensuredir()
    results_path = ub.ensuredir((test_dpath, 'predict'))
    ub.delete(results_path)
    ub.ensuredir(results_path)
    package_fpath = test_dpath / 'my_test_package.pt'
    # train_dset = kwcoco.CocoDataset.demo('special:vidshapes4-multispectral', num_frames=5, gsize=(128, 128))
    test_dset = kwcoco.CocoDataset.demo('special:vidshapes2-multispectral', num_frames=3, gsize=(128, 128))
    fit_kwargs = {
         'train_dataset': test_dset.fpath,
         'datamodule': 'KWCocoVideoDataModule',
         'workdir': ub.ensuredir((test_dpath, 'train')),
         'package_fpath': package_fpath,
         'max_epochs': 1,
         'time_steps': 3,
         'chip_size': 64,
         'max_steps': 1,
         'learning_rate': 1e-5,
         'diff_inputs': False,
         'num_workers': 1,
         'gpus': gpus,
    }
    package_fpath = fit_model(**fit_kwargs)

    # Start a new training run but try to init from the pretrianed state
    # (even though it is different)
    fit_kwargs2 = fit_kwargs.copy()
    fit_kwargs2['init'] = package_fpath
    fit_kwargs2['diff_inputs'] = 0
    fit_model(**fit_kwargs2)

    # Start a new training run but try to init from the pretrianed state
    # (even though it is different)  # TODO: had to remove diff-inputs, make more different to test
    fit_kwargs3 = fit_kwargs.copy()
    fit_kwargs3['init'] = 'kaiming_normal'
    fit_kwargs3['diff_inputs'] = 0
    fit_model(**fit_kwargs2)


def test_init_from_phase1_models():
    import geowatch
    import os
    try:
        phase1_dvc_dpath = geowatch.find_dvc_dpath(tags='phase1')
    except Exception:
        import pytest
        pytest.skip('dvc repo is not available')

    from geowatch.tasks.fusion import production
    found = None
    for row in production.PRODUCTION_MODELS:
        if row['name'] == 'Drop3_SpotCheck_V323_epoch=18-step=12976.pt':
            found = row
            break
    assert found
    model_fpath = phase1_dvc_dpath / found['file_name']
    if not model_fpath.exists():
        import pytest
        pytest.skip('dvc repo is not available')

    init = model_fpath

    # First test it works by itself
    from geowatch.tasks.fusion.fit import coerce_initializer
    initializer = coerce_initializer(init)
    fpath = initializer._rectify_fpath()
    from torch_liberator.initializer import _torch_load
    with open(fpath, 'rb') as file:
        state_dict = _torch_load(file)
    assert state_dict

    # gpus = None
    test_dpath = ub.Path.appdir('geowatch/tests/fusion/').ensuredir()
    results_path = ub.ensuredir((test_dpath, 'predict'))
    ub.delete(results_path)
    ub.ensuredir(results_path)
    package_fpath = test_dpath / 'my_test_package.pt'
    import kwcoco
    from geowatch.tasks.fusion.fit import fit_model  # NOQA
    test_dset = kwcoco.CocoDataset.demo('special:vidshapes2-multispectral', num_frames=3, gsize=(128, 128))
    fit_kwargs = {
         'train_dataset': test_dset.fpath,
         'datamodule': 'KWCocoVideoDataModule',
         'workdir': ub.ensuredir((test_dpath, 'train')),
         'package_fpath': os.fspath(package_fpath),
         'max_epochs': 1,
         'time_steps': 3,
         'chip_size': 64,
         'max_steps': 1,
         'learning_rate': 1e-5,
         'diff_inputs': False,
         'num_workers': 1,
         'init': os.fspath(model_fpath),
         'devices': None,
         # 'gpus': gpus,
    }
    package_fpath = fit_model(**fit_kwargs)


if __name__ == '__main__':
    """
    CommandLine:
        python ~/code/geowatch/tests/test_init_from_pretrained.py
    """
    test_init_from_pretrained_state()
