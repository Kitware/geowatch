
def test_init_from_pretrained_state():
    # Train a demo model (in the future grab a pretrained demo model)
    import ubelt as ub
    from os.path import join
    from watch.tasks.fusion.fit import fit_model  # NOQA
    import kwcoco

    import pytest
    pytest.skip('slow test')

    # import xdev
    # xdev.make_warnings_print_tracebacks()

    # args = None
    # cmdline = False
    gpus = None
    test_dpath = ub.ensure_app_cache_dir('watch/test/fusion/')
    results_path = ub.ensuredir((test_dpath, 'predict'))
    ub.delete(results_path)
    ub.ensuredir(results_path)
    package_fpath = join(test_dpath, 'my_test_package.pt')
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
    import watch
    watch.fin
    pass


if __name__ == '__main__':
    """
    CommandLine:
        python ~/code/watch/tests/test_init_from_pretrained.py
    """
    test_init_from_pretrained_state()
