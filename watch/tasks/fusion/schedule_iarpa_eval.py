import ubelt as ub


def _suggest_track_paths(pred_fpath, track_cfg, eval_dpath=None):
    """
    Helper for reasonable paths to keep everything organized for tracking eval
    """
    human_opts = ub.dict_isect(track_cfg, {'thresh'})
    other_opts = ub.dict_diff(track_cfg, human_opts)
    human_part = ub.repr2(human_opts, compact=1)
    track_cfgstr = human_part + '_' + ub.hash_data(other_opts)[0:8]
    pred_bundle_dpath = pred_fpath.parent
    track_cfg_dname = f'tracking/trackcfg_{track_cfgstr}'
    track_cfg_base = pred_bundle_dpath / track_cfg_dname
    track_out_fpath = track_cfg_base / 'tracks.json'

    if eval_dpath is None:
        iarpa_eval_dpath = track_cfg_base / 'iarpa_eval'
    else:
        iarpa_eval_dpath = eval_dpath / track_cfg_dname / 'iarpa_eval'

    iarpa_merge_fpath = iarpa_eval_dpath / 'scores' / 'merged' / 'summary2.json'

    track_suggestions = {
        'iarpa_eval_dpath': iarpa_eval_dpath,
        'track_cfgstr': track_cfgstr,
        'track_out_fpath': track_out_fpath,
        'iarpa_merge_fpath': iarpa_merge_fpath,
    }
    return track_suggestions


def _build_bas_track_job(pred_fpath, track_out_fpath, thresh=0.2):
    """
    Given a predicted kwcoco file submit tracking and iarpa eval jobs

    Args:
        pred_fpath (PathLike): path to predicted kwcoco file
        task (str): bas or sc
        annotations_dpath (PathLike): path to IARPA annotations file ($dvc/annotations)

    Ignore:
        pred_fpath = '/home/joncrall/data/dvc-repos/smart_watch_dvc/_tmp/_tmp_pred_00/pred.kwcoco.json'
        annotations_dpath = '/home/joncrall/data/dvc-repos/smart_watch_dvc/annotations'
        thresh = 0.2
        task = 'bas'
    """
    pred_fpath = ub.Path(pred_fpath)

    track_cfg = {
        'thresh': thresh,
    }

    task = 'bas'
    if task == 'bas':
        import shlex
        import json
        track_kwargs_str = shlex.quote(json.dumps(track_cfg))
        bas_args = f'--default_track_fn saliency_heatmaps --track_kwargs {track_kwargs_str}'  # NOQA
        task_args = bas_args
    elif task == 'sc':
        sc_args = r'--track_fn watch.tasks.tracking.from_heatmap.TimeAggregatedHybrid --track_kwargs "{\"coco_dset_sc\": \"' + str(pred_fpath) + r'\"}"'  # NOQA
        task_args = bas_args
    else:
        raise KeyError

    # Because BAS is the first step we want ensure we clear annotations so
    # everything that comes out is a track from BAS.

    sites_dpath = track_out_fpath.parent / 'tracked_sites'
    command = ub.codeblock(
        fr'''
        python -m watch.cli.kwcoco_to_geojson \
            "{pred_fpath}" \
             {task_args} \
            --clear_annots \
            --out_dir "{sites_dpath}" \
            --out_fpath "{track_out_fpath}"
        ''')

    track_info = {
        'command': command,
        'track_out_fpath': track_out_fpath,
    }
    return track_info


def _build_sc_track_job(pred_fpath, track_out_fpath, thresh=0.2):
    r"""
    Given a predicted kwcoco file submit tracking and iarpa eval jobs


    Notes:

        DVC_DPATH=$(smartwatch_dvc --hardware=hdd)
        PRED_DATASET=$DVC_DPATH/models/fusion/eval3_sc_candidates/pred/CropDrop3_SC_V006/pred_CropDrop3_SC_V006_epoch=71-step=18431/Cropped-Drop3-TA1-2022-03-10_combo_DL_s2_wv_vali.kwcoco/predcfg_464eb52f/pred.kwcoco.json
        SITE_SUMMARY_GLOB="$DVC_DPATH/annotations/region_models/KR_*.geojson"

        ANNOTATIONS_DPATH=$DVC_DPATH/annotations
        ls $ANNOTATIONS_DPATH

        python -m watch.cli.kwcoco_to_geojson \
            "$PRED_DATASET" \
            --default_track_fn class_heatmaps \
            --site_summary "$SITE_SUMMARY_GLOB" \
            --track_kwargs '{"boundaries_as": "polys"}' \
            --out_dir ./tmp/site_models \
            --out_fpath ./tmp/site_models_stamp.json

        python -m watch.cli.run_metrics_framework \
            --merge \
            --gt_dpath "$ANNOTATIONS_DPATH" \
            --tmp_dir "./tmp/iarpa/tmp" \
            --out_dir "./tmp/iarpa/scores" \
            --name "mytest" \
            --merge_fpath "./tmp/iarpa/merged.json" \
            --inputs_are_paths \
            ./tmp/site_models_stamp.json

    Args:
        pred_fpath (PathLike): path to predicted kwcoco file
        task (str): bas or sc
        annotations_dpath (PathLike): path to IARPA annotations file ($dvc/annotations)

    Ignore:
        pred_fpath = '/home/joncrall/data/dvc-repos/smart_watch_dvc/_tmp/_tmp_pred_00/pred.kwcoco.json'
        annotations_dpath = '/home/joncrall/data/dvc-repos/smart_watch_dvc/annotations'
        thresh = 0.2
        task = 'bas'
    """
    pred_fpath = ub.Path(pred_fpath)

    track_cfg = {
        'thresh': thresh,
    }

    task = 'bas'
    if task == 'bas':
        import shlex
        import json
        track_kwargs_str = shlex.quote(json.dumps(track_cfg))
        bas_args = f'--default_track_fn saliency_heatmaps --track_kwargs {track_kwargs_str}'  # NOQA
        task_args = bas_args
    elif task == 'sc':
        sc_args = r'--track_fn watch.tasks.tracking.from_heatmap.TimeAggregatedHybrid --track_kwargs "{\"coco_dset_sc\": \"' + str(pred_fpath) + r'\"}"'  # NOQA
        task_args = bas_args
    else:
        raise KeyError

    # Because BAS is the first step we want ensure we clear annotations so
    # everything that comes out is a track from BAS.

    sites_dpath = track_out_fpath.parent / 'tracked_sites'
    command = ub.codeblock(
        fr'''
        python -m watch.cli.kwcoco_to_geojson \
            "{pred_fpath}" \
             {task_args} \
            --clear_annots \
            --out_dir "{sites_dpath}" \
            --out_fpath "{track_out_fpath}"
        ''')

    track_info = {
        'command': command,
        'track_out_fpath': track_out_fpath,
    }
    return track_info


def _build_iarpa_eval_job(track_out_fpath, iarpa_merge_fpath, iarpa_eval_dpath, annotations_dpath, name=None):
    import shlex
    tmp_dir = iarpa_eval_dpath / 'tmp'
    out_dir = iarpa_eval_dpath / 'scores'
    merge_dpath = out_dir / 'merged'

    command = ub.codeblock(
        fr'''
        python -m watch.cli.run_metrics_framework \
            --merge \
            --gt_dpath "{annotations_dpath}" \
            --tmp_dir "{tmp_dir}" \
            --out_dir "{out_dir}" \
            --name {shlex.quote(str(name))} \
            --merge_fpath "{iarpa_merge_fpath}" \
            --inputs_are_paths \
            {track_out_fpath}
        ''')

    iarpa_summary_fpath = merge_dpath / 'summary2.json'

    iarpa_eval_info = {
        'command': command,
        'out_dir': out_dir,
        'iarpa_summary_fpath': iarpa_summary_fpath,
    }
    return iarpa_eval_info


"""
# Note: change backend to tmux if slurm is not installed
DVC_DPATH=$(smartwatch_dvc)
DATASET_CODE=Aligned-Drop3-TA1-2022-03-10/
EXPT_GROUP_CODE=eval3_candidates
KWCOCO_BUNDLE_DPATH=$DVC_DPATH/$DATASET_CODE
VALI_FPATH=$KWCOCO_BUNDLE_DPATH/combo_LM_nowv_vali.kwcoco.json
python -m watch.tasks.fusion.schedule_evaluation schedule_evaluation \
        --gpus="0,1,2,3,4,5,6,7,8" \
        --model_globstr="$DVC_DPATH/models/fusion/$EXPT_GROUP_CODE/packages/*/*.pt" \
        --test_dataset="$VALI_FPATH" \
        --skip_existing=True \
        --enable_pred=0 \
        --enable_eval=0 \
        --enable_track=1 \
        --enable_iarpa_eval=1 \
        --backend=tmux --run=1
"""
