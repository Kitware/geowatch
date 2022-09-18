import kwcoco
import pytest
import watch
import ubelt as ub


def get_production_model_test_info(task):
    from watch.tasks.fusion import production
    MODELS = [row for row in production.PRODUCTION_MODELS
              if row.get('task', None) == task]
    model_info = MODELS[-1]
    tags = model_info['tags']

    try:
        expt_dvc_dpath = watch.find_smart_dvc_dpath(tags=tags)
    except Exception:
        pytest.skip('dvc path does not exist')

    model_fpath = expt_dvc_dpath / model_info['file_name']

    from watch.cli import torch_model_stats
    torch_model_stats.main(cmdline=False, src=model_fpath)

    from watch.utils.simple_dvc import SimpleDVC
    dvc = SimpleDVC()
    dvc.request(model_fpath)
    return model_fpath


DEBUGGING_NOW = 0


def test_predict_latest_bas_model():
    """
    Test the predict step with the latest and greatest
    """
    model_fpath = get_production_model_test_info(task='BAS')

    try:
        data_dvc_dpath = watch.find_smart_dvc_dpath(tags='phase2_data')
    except Exception:
        pytest.skip('dvc path does not exist')
    bundle_dpath = (data_dvc_dpath / 'Aligned-Drop4-2022-08-08-TA1-S2-L8-ACC')
    vali_fpath = (bundle_dpath / 'data_vali.kwcoco.json')

    output_dpath = ub.Path.appdir('watch/tests/pred/bas_latest').ensuredir()

    pred_fpath = output_dpath / 'pred_bundle/pred.kwcoco.json'
    dset = kwcoco.CocoDataset(vali_fpath)
    subset = make_small_kwcoco_subset(dset, output_dpath)

    from watch.tasks.fusion import predict as predict_mod
    pred_kwargs = {
        'test_dataset': subset.fpath,
        'pred_dataset': pred_fpath,
        'package_fpath': model_fpath,
        'chip_overlap': 0.3,
        'gpus': "auto:1",
        'num_workers': 0,
        'write_preds': False,
        'clear_annots': False,
        # 'set_cover_algo': 'approx',
        'set_cover_algo': 'approx',
        'space_scale': '15GSD',
        'window_space_scale': '15GSD',
        'use_cloudmask': False,
        'resample_invalid_frames': False,
    }
    kwargs = pred_kwargs  # NOQA
    predict_mod.predict(cmdline=0, **pred_kwargs)

    if DEBUGGING_NOW:
        import xdev
        xdev.view_directory(pred_fpath.parent)

        from watch.cli import coco_visualize_videos
        coco_visualize_videos.main(cmdline=0, src=pred_fpath,
                                   channels='red|green|blue,salient',
                                   skip_missing=False, animate=True)


def test_predict_latest_sc_model():
    model_fpath = get_production_model_test_info(task='SC')

    try:
        data_dvc_dpath = watch.find_smart_dvc_dpath(tags='phase2_data')
    except Exception:
        pytest.skip('dvc path does not exist')
    bundle_dpath = (data_dvc_dpath / 'Aligned-Drop4-2022-08-08-TA1-S2-WV-PD-ACC')
    vali_fpath = (bundle_dpath / 'data_vali.kwcoco.json')

    output_dpath = ub.Path.appdir('watch/tests/pred/sc_latest').ensuredir()
    pred_fpath = output_dpath / 'pred_bundle/pred.kwcoco.json'
    dset = kwcoco.CocoDataset(vali_fpath)
    subset = make_small_kwcoco_subset(dset, output_dpath)

    from watch.tasks.fusion import predict as predict_mod
    pred_kwargs = {
        'test_dataset': subset.fpath,
        'pred_dataset': pred_fpath,
        'package_fpath': model_fpath,
        'chip_overlap': 0.3,
        'gpus': "auto:1",
        'num_workers': 0,
        'with_class': True,
        'with_saliency': False,
        'with_change': False,
        'write_preds': False,
        'clear_annots': False,
        'use_cloudmask': False,
        'resample_invalid_frames': False,
        'set_cover_algo': 'approx',
    }
    kwargs = pred_kwargs  # NOQA
    cmdline = False
    predict_mod.predict(cmdline=cmdline, **pred_kwargs)

    if DEBUGGING_NOW:
        import xdev
        xdev.view_directory(pred_fpath.parent)

        from watch.cli import coco_visualize_videos
        coco_visualize_videos.main(cmdline=0, src=pred_fpath)


def make_small_kwcoco_subset(dset, output_dpath):
    import pytest
    import numpy as np
    from watch.utils import kwcoco_extensions

    if 1:
        # Find a spot with a lot of changes.
        found = None
        for tid, aids in dset.index.trackid_to_aids.items():
            aids = list(aids)
            track_aids = dset.index.trackid_to_aids[tid]
            cids = dset.annots(track_aids).lookup('category_id')
            changes = np.where(np.diff(cids))[0]
            if len(changes) > 2:
                change_annots = list(ub.take(aids, changes))
                found = change_annots
                break

        if found is not None:
            gids = dset.annots(change_annots).get('image_id')
            vidid = list(dset.images(gids).lookup('video_id'))[0]
            vid_gids = list(dset.images(vidid=vidid))
            vid_gids = ub.oset(vid_gids)
            remain_gids = vid_gids - ub.oset(gids)
            import kwarray
            rng = kwarray.ensure_rng(0)
            num_want = 20
            chosen = list(rng.choice(remain_gids, num_want - len(gids)))
            final_gids = vid_gids & (set(gids) | set(chosen))
        else:
            raise Exception
    # else:
    #     vidid = dset.videos().peek()['id']
    #     video_images = dset.images(vidid=vidid)
    #     final_gids = list(dset.images(vidid=vidid))[0:11]

    subset = dset.subset(final_gids)
    if subset.missing_images(check_aux=True):
        pytest.skip('data has not been pulled down')
    subset.reroot(absolute=True)
    subset.fpath = str(output_dpath / 'test_input.kwcoco.json')

    stats = kwcoco_extensions.coco_channel_stats(subset)
    print('stats = {}'.format(ub.repr2(stats, nl=2)))

    subset.dump(subset.fpath, newlines=True)
    return subset
