import kwcoco
import ubelt as ub


def test_predict_latest_models():
    """
    Test the predict step with the latest and greatest
    """
    import watch
    import pytest
    from watch.tasks.fusion import production
    model_info = production.PRODUCTION_MODELS[-1]
    tags = model_info['tags']

    try:
        expt_dvc_dpath = watch.find_smart_dvc_dpath(tags=tags)
        data_dvc_dpath = watch.find_smart_dvc_dpath(tags='phase2_data')
    except Exception:
        pytest.skip('dvc path does not exist')

    model_fpath = expt_dvc_dpath / model_info['file_name']

    from watch.cli import torch_model_stats
    torch_model_stats.main(cmdline=False, src=model_fpath)

    from watch.utils.simple_dvc import SimpleDVC
    dvc = SimpleDVC()
    dvc.request(model_fpath)

    bundle_dpath = (data_dvc_dpath / 'Aligned-Drop4-2022-08-08-TA1-S2-L8-ACC')
    vali_fpath = (bundle_dpath / 'data_vali.kwcoco.json')

    output_dpath = ub.Path.appdir('watch/tests/pred/latest').ensuredir()
    pred_fpath = output_dpath / 'pred_bundle/pred.kwcoco.json'
    dset = kwcoco.CocoDataset(vali_fpath)
    subset = make_small_kwcoco_subset(dset, output_dpath)

    from watch.tasks.fusion import predict
    pred_kwargs = {
        'test_dataset': subset.fpath,
        'pred_dataset': pred_fpath,
        'package_fpath': model_fpath,
        'chip_overlap': 0.0,
        'gpus': "auto:1",
        'num_workers': 0,
        'set_cover_algo': 'approx',
    }
    kwargs = pred_kwargs  # NOQA
    predict.predict(**pred_kwargs)

    if 0:
        import xdev
        xdev.view_directory(pred_fpath.parent)

        from watch.cli import coco_visualize_videos
        coco_visualize_videos.main(cmdline=0, src=pred_fpath)


def make_small_kwcoco_subset(dset, output_dpath):
    import pytest
    from watch.utils import kwcoco_extensions
    vidid = dset.videos().peek()['id']
    gids = list(dset.images(vidid=vidid))[0:11]
    subset = dset.subset(gids)
    if subset.missing_images(check_aux=True):
        pytest.skip('data has not been pulled down')
    subset.reroot(absolute=True)
    subset.fpath = str(output_dpath / 'test_input.kwcoco.json')

    stats = kwcoco_extensions.coco_channel_stats(subset)
    print('stats = {}'.format(ub.repr2(stats, nl=2)))

    subset.dump(subset.fpath, newlines=True)
    return subset
