def test_predict_override_mean_std():
    """
    Test that the user can override the meanstd at predict time.
    """
    from geowatch.tasks.fusion.predict import predict
    import ubelt as ub
    import os
    from geowatch.utils.lightning_ext.monkeypatches import disable_lightning_hardware_warnings
    disable_lightning_hardware_warnings()
    devices = None
    test_dpath = ub.Path.appdir('geowatch/tests/fusion/').ensuredir()
    results_path = (test_dpath / 'predict').ensuredir()

    # Train a dummy model (TODO: need to build a mechanism so a trained model
    # is created and cached once for all tests that need a model)
    results_path.delete()
    results_path.ensuredir()
    import kwcoco
    train_dset = kwcoco.CocoDataset.demo('special:vidshapes2-gsize64-frames9-speed0.5-multispectral')
    test_dset = kwcoco.CocoDataset.demo('special:vidshapes1-gsize64-frames9-speed0.5-multispectral')
    root_dpath = ub.Path(test_dpath, 'train').ensuredir()
    fit_config = {
        'subcommand': 'fit',
        'fit.data.train_dataset': train_dset.fpath,
        'fit.data.time_steps': 2,
        'fit.data.time_span': "2m",
        'fit.data.chip_dims': 64,
        'fit.data.time_sampling': 'hardish3',
        'fit.data.num_workers': 0,
        #'package_fpath': package_fpath,
        'fit.model.class_path': 'geowatch.tasks.fusion.methods.MultimodalTransformer',
        'fit.model.init_args.global_change_weight': 1.0,
        'fit.model.init_args.global_class_weight': 1.0,
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
        'override_meanstd': ub.codeblock(
            '''
            - sensor: '*'
              channels: 'B1|B10|B11|B8|B8a'
              mean: [0, 0, 0, 0, 0]
              std: [1, 1, 1, 1, 1]
            '''),
        'draw_batches': 1,
    }
    result_dataset = predict(**predict_kwargs)
    dset = result_dataset
    dset.dataset['info'][-1]['properties']['config']['time_sampling']
    # Check that the result format looks correct
    for vidid in dset.index.videos.keys():
        # Note: only some of the images in the pred sequence will get
        # a change predictoion, depending on the temporal sampling.
        images = dset.images(dset.index.vidid_to_gids[1])
        pred_chans = [[a['channels'] for a in aux] for aux in images.lookup('auxiliary')]
        assert any('change' in cs for cs in pred_chans), 'some frames should have change'
        assert not all('change' in cs for cs in pred_chans), 'some frames should not have change'
        # Test number of annots in each frame
        frame_to_cathist = {
            img['frame_index']: ub.dict_hist(annots.cnames, labels=result_dataset.object_categories())
            for img, annots in zip(images.objs, images.annots)
        }
        assert frame_to_cathist[0]['change'] == 0, 'first frame should have no change polygons'
        # This test may fail with very low probability, so warn
        import warnings
        if sum(d['change'] for d in frame_to_cathist.values()) == 0:
            warnings.warn('should have some change predictions elsewhere')
    coco_img = dset.images().coco_images[1]
    # Test that new quantization does not existing APIs
    pred1 = coco_img.imdelay('salient', nodata_method='float').finalize()
    assert pred1.max() <= 1
    # new delayed image does not make it easy to remove dequantization
    # add test back in if we add support for that.
