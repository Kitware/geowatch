
def test_small_predict_region():
    import pytest
    pytest.skip('TODO: switch to lightning')
    from geowatch.tasks.fusion.fit import fit_model  # NOQA
    import ubelt as ub
    from geowatch.tasks.fusion.predict import predict
    import geowatch

    gpus = None
    test_dpath = ub.Path.appdir('geowatch/tests/fusion/')
    results_path = (test_dpath / 'predict').ensuredir()
    ub.delete(results_path)
    ub.ensuredir(results_path)
    package_fpath = test_dpath / 'my_test_package.pt'

    train_dset = geowatch.coerce_kwcoco(
        'geowatch-msi', num_videos=3, num_frames=5, image_size=(128, 128),
        rng=3213,
    )

    # Predict via that model on a smaller dataset
    test_dset = geowatch.coerce_kwcoco(
        'geowatch-msi', num_videos=1, num_frames=1, image_size=(32, 32),
        rng=90312,
    )
    from geowatch.utils import kwcoco_extensions  # NOQA
    test_chans = kwcoco_extensions.coco_channel_stats(test_dset)
    train_chans = kwcoco_extensions.coco_channel_stats(train_dset)
    print('test_chans = {}'.format(ub.urepr(test_chans, nl=1)))
    print('train_chans = {}'.format(ub.urepr(train_chans, nl=1)))

    fit_kwargs = {
        'train_dataset': train_dset.fpath,
        'datamodule': 'KWCocoVideoDataModule',
        'workdir': ub.ensuredir((test_dpath, 'train')),
        'package_fpath': package_fpath,
        'max_epochs': 1,
        'time_steps': 3,
        'chip_size': 64,
        'time_sampling': 'hardish3',
        'global_change_weight': 0.0,
        'global_class_weight': 0.0,
        'global_saliency_weight': 1.0,
        'max_steps': 1,
        'learning_rate': 1e-5,
        'num_workers': 0,
        'gpus': gpus,
    }

    package_fpath = fit_model(**fit_kwargs)

    from geowatch.cli import torch_model_stats
    torch_model_stats.main(src=package_fpath)

    predict_kwargs = {
        'package_fpath': package_fpath,
        'pred_dataset': results_path / 'pred.kwcoco.json',
        'test_dataset': test_dset.fpath,
        'datamodule': 'KWCocoVideoDataModule',
        'write_probs': True,
        'write_preds': False,
        'with_change': False,
        'with_class': False,
        'with_saliency': True,
        'batch_size': 8,
        'tta_time': 1,
        'tta_fliprot': 0,
        'chip_overlap': 0.3,
        'num_workers': 0,
        'gpus': gpus,
    }
    result_dataset = predict(**predict_kwargs)
    pred_chans = kwcoco_extensions.coco_channel_stats(result_dataset)
    print('pred_chans = {}'.format(ub.urepr(pred_chans, nl=1)))

    from geowatch.cli import watch_coco_stats
    watch_coco_stats.WatchCocoStats.main(src=train_dset.fpath)
    watch_coco_stats.WatchCocoStats.main(src=train_dset.fpath)
    watch_coco_stats.WatchCocoStats.main(src=result_dataset.fpath)

    dset = result_dataset
    # Check that the result format looks correct
    for vidid in dset.index.videos.keys():
        coco_imgs = dset.images(dset.index.vidid_to_gids[1]).coco_images
        for coco_img in coco_imgs:
            assert 'salient' in coco_img.channels

    coco_img = dset.images().coco_images[0]
    # Test that new quantization does not existing APIs
    pred1 = coco_img.imdelay('salient').finalize(nodata='float')
    pred2 = coco_img.imdelay('salient').finalize(nodata='float', dequantize=False)
    assert pred1.max() <= 1
    assert pred2.max() > 1


if __name__ == '__main__':
    """
    CommandLine:
        python ~/code/geowatch/tests/test_predict_small_region.py
    """
    test_small_predict_region()
