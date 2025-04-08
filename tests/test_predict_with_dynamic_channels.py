def test_predict_with_dynamic_channels():
    """
    Test that dynamic channels and robust normalize are handled by the predict
    script correctly.
    """
    from geowatch.tasks.fusion.predict import predict
    import kwcoco
    import ubelt as ub
    import os
    from geowatch.utils.lightning_ext.monkeypatches import disable_lightning_hardware_warnings
    disable_lightning_hardware_warnings()
    devices = None
    test_dpath = ub.Path.appdir('geowatch/tests/fusion/').ensuredir()
    results_path = (test_dpath / 'predict_with_dynamic_channels').ensuredir()

    # Train a dummy model (TODO: need to build a mechanism so a trained model
    # is created and cached once for all tests that need a model)
    results_path.delete()
    results_path.ensuredir()
    train_dset = kwcoco.CocoDataset.demo('special:vidshapes2-gsize64-frames9')
    test_dset = kwcoco.CocoDataset.demo('special:vidshapes1-gsize64-frames9')
    root_dpath = ub.Path(test_dpath, 'train').ensuredir()
    fit_config = {
        'subcommand': 'fit',
        'fit.data.train_dataset': train_dset.fpath,
        'fit.data.time_steps': 2,
        'fit.data.time_span': "2m",
        'fit.data.chip_dims': 64,
        'fit.data.time_sampling': 'contiguous',
        'fit.data.num_workers': 0,
        'fit.data.channels': 'r|negative_r1|r_sqrt|negative_r2',
        'fit.data.dynamic_channels': ub.codeblock(
            '''
            - name: negative_r1
              expr: -r
            - name: negative_r2
              expr: -r
            - name: r_sqrt
              expr: sqrt(r)
            '''),
        'fit.data.robust_normalize': ub.codeblock(
            '''
            groups:
              - sensorchan: "negative_r1|r"
              - sensorchan: "r_sqrt"
                high: 0.9
                low: 0.1
            '''),
        #'package_fpath': package_fpath,
        'fit.model.class_path': 'geowatch.tasks.fusion.methods.MultimodalTransformer',
        'fit.model.init_args.global_change_weight': 0.0,
        'fit.model.init_args.global_class_weight': 0.0,
        'fit.model.init_args.global_saliency_weight': 1.0,
        'fit.optimizer.class_path': 'torch.optim.SGD',
        'fit.optimizer.init_args.lr': 1e-5,
        'fit.trainer.max_steps': 10,
        'fit.trainer.accelerator': 'cpu',
        'fit.trainer.devices': 1,
        'fit.trainer.max_epochs': 3,
        'fit.trainer.log_every_n_steps': 1,
        'fit.trainer.default_root_dir': os.fspath(root_dpath),
    }
    from geowatch.tasks.fusion import fit_lightning
    package_fpath = root_dpath / 'final_package.pt'
    fit_lightning.main(fit_config)

    # TODO: it would be nice to test if the batches are normalized correctly
    # here. This will require some ability to write raw batches to disk so they
    # can be checked.

    # Unfortunately, its not as easy to get the package path of
    # this call..
    assert ub.Path(package_fpath).exists()

    # Predict via that model
    predict_kwargs = {
        'package_fpath': package_fpath,
        'pred_dataset': ub.Path(results_path) / 'pred.kwcoco.json',
        'test_dataset': test_dset.fpath,
        'datamodule': 'KWCocoVideoDataModule',
        'batch_size': 1,
        'num_workers': 0,
        'devices': devices,
        'draw_batches': 1,
    }
    result_dataset = predict(**predict_kwargs)
    dset = result_dataset

    # test that the predicted config was able to introspect the dynamic
    # channels and robust normalize
    pred_config = dset.dataset['info'][-1]['properties']['config']
    fit_config = dset.dataset['info'][-1]['properties']['extra']['fit_config']

    assert pred_config['robust_normalize'] == fit_config['data']['robust_normalize']
    assert pred_config['dynamic_channels'] == fit_config['data']['dynamic_channels']
    assert pred_config['robust_normalize'] == {
        'groups': [{'sensorchan': 'negative_r1|r'},
                   {'high': 0.9, 'low': 0.1, 'sensorchan': 'r_sqrt'}]}
    assert pred_config['dynamic_channels'] == [{'expr': '-r', 'name': 'negative_r1'},
                                               {'expr': '-r', 'name': 'negative_r2'},
                                               {'expr': 'sqrt(r)', 'name': 'r_sqrt'}]
