import kwcoco
import pytest
import geowatch
import ubelt as ub


def get_production_model_test_info(task):
    from geowatch.tasks.fusion import production
    import json
    candidates = []

    candidates = [row for row in production.PRODUCTION_MODELS
                  if row.get('task', None) == task]

    for item in production.NEW_PRODUCTION_MODELS:
        if isinstance(item, str):
            item = json.loads(item)
            if 'tasks' not in item:
                item['tasks'] = []
                if item['fit_params']['global_saliency_weight'] > 0:
                    item['tasks'].append('BAS')
                if item['fit_params']['global_class_weight'] > 0:
                    item['tasks'].append('SC')
            if task in item['tasks']:
                candidates.append(item)
            item['file_name']

    model_info = candidates[-1]
    tags = model_info.get('tags', 'phase2_expt')

    try:
        expt_dvc_dpath = geowatch.find_smart_dvc_dpath(tags=tags)
    except Exception:
        pytest.skip('dvc path does not exist')

    model_fpath = expt_dvc_dpath / model_info['file_name']

    if task == 'SC' and 'pred_params' not in model_info:
        model_info['pred_params'] = {
            'chip_overlap': 0.3,
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
    if task == 'BAS' and 'pred_params' not in model_info:
        model_info['pred_params'] = {
            'chip_overlap': 0.3,
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

    model_info['pred_params']['package_fpath'] = model_fpath

    from geowatch.utils.simple_dvc import SimpleDVC
    dvc = SimpleDVC()
    dvc.request(model_fpath)

    from geowatch.cli import torch_model_stats
    torch_model_stats.main(cmdline=False, src=model_fpath)
    return model_info


def get_test_dataset_fpath(task):
    try:
        data_dvc_dpath = geowatch.find_smart_dvc_dpath(tags='phase2_data')
    except Exception:
        pytest.skip('dvc path does not exist')
    if task == 'BAS':
        bundle_dpath = (data_dvc_dpath / 'Aligned-Drop4-2022-08-08-TA1-S2-L8-ACC')
        vali_fpath = (bundle_dpath / 'data_vali.kwcoco.json')
    elif task == 'SC':
        bundle_dpath = (data_dvc_dpath / 'Aligned-Drop4-2022-08-08-TA1-S2-WV-PD-ACC')
        vali_fpath = (bundle_dpath / 'data_vali.kwcoco.json')
    else:
        raise KeyError(task)
    return vali_fpath


DEBUGGING_NOW = 0


def test_predict_latest_bas_model():
    """
    Test the predict step with the latest and greatest
    """
    import pytest
    pytest.skip('slow and a little outdated')
    model_info = get_production_model_test_info(task='BAS')
    pred_params = model_info['pred_params']

    output_dpath = ub.Path.appdir('geowatch/tests/pred/bas_latest').ensuredir()

    vali_fpath = get_test_dataset_fpath(task='BAS')
    dset = kwcoco.CocoDataset(vali_fpath)
    subset = make_small_kwcoco_subset(dset, output_dpath)

    pred_fpath = output_dpath / 'pred_bundle/pred.kwcoco.json'

    from geowatch.tasks.fusion import predict as predict_mod
    pred_kwargs = ub.udict(pred_params) | {
        'test_dataset': subset.fpath,
        'pred_dataset': pred_fpath,
        'gpus': "auto:1",
        # "output_space_scale": "15GSD",
        'num_workers': 2,
    }
    kwargs = pred_kwargs  # NOQA
    predict_mod.predict(cmdline=0, **pred_kwargs)

    if DEBUGGING_NOW:
        import xdev
        xdev.view_directory(pred_fpath.parent)

        from geowatch.cli import coco_visualize_videos
        coco_visualize_videos.main(cmdline=0, src=pred_fpath,
                                   channels='red|green|blue,salient',
                                   stack='only', skip_missing=False, animate=True)


def test_predict_latest_sc_model():
    import pytest
    pytest.skip('slow and a little outdated')
    model_info = get_production_model_test_info(task='SC')
    pred_params = model_info['pred_params']

    output_dpath = ub.Path.appdir('geowatch/tests/pred/sc_latest').ensuredir()

    vali_fpath = get_test_dataset_fpath(task='SC')
    dset = kwcoco.CocoDataset(vali_fpath)
    subset = make_small_kwcoco_subset(dset, output_dpath)

    pred_fpath = output_dpath / 'pred_bundle/pred.kwcoco.json'

    from geowatch.tasks.fusion import predict as predict_mod
    pred_kwargs = ub.udict(pred_params) | {
        'test_dataset': subset.fpath,
        'pred_dataset': pred_fpath,
        'gpus': "auto:1",
        'num_workers': 0,
    }
    kwargs = pred_kwargs  # NOQA
    cmdline = False
    predict_mod.predict(cmdline=cmdline, **pred_kwargs)

    if DEBUGGING_NOW:
        import xdev
        xdev.view_directory(pred_fpath.parent)

        from geowatch.cli import coco_visualize_videos
        coco_visualize_videos.main(cmdline=0, src=pred_fpath, channels='auto',
                                   stack='only', workers=0)


def make_small_kwcoco_subset(dset, output_dpath):
    import pytest
    import numpy as np
    from geowatch.utils import kwcoco_extensions

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
            vid_gids = list(dset.images(video_id=vidid))
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
    #     video_images = dset.images(video_id=vidid)
    #     final_gids = list(dset.images(video_id=vidid))[0:11]

    subset = dset.subset(final_gids)
    if subset.missing_images(check_aux=True):
        pytest.skip('data has not been pulled down')
    subset.reroot(absolute=True)
    subset.fpath = str(output_dpath / 'test_input.kwcoco.json')

    stats = kwcoco_extensions.coco_channel_stats(subset)
    print('stats = {}'.format(ub.urepr(stats, nl=2)))

    subset.dump(subset.fpath, newlines=True)
    return subset


if __name__ == '__main__':
    """
    CommandLine:
        python ~/code/geowatch/tests/test_predict_latest_models.py
    """
    # test_predict_latest_bas_model()
    test_predict_latest_sc_model()
