import json
import ubelt as ub


def _suggest_bas_path(pred_fpath, bas_track_cfg, eval_dpath=None):
    """
    Helper for reasonable paths to keep everything organized for tracking eval
    """
    # human_opts = ub.dict_isect(bas_track_cfg, {'thresh'})
    human_opts = ub.dict_isect(bas_track_cfg, {})
    other_opts = ub.dict_diff(bas_track_cfg, human_opts)
    if len(human_opts):
        human_part = ub.repr2(human_opts, compact=1) + '_'
    else:
        human_part = ''
    cfgstr = human_part + ub.hash_data(other_opts)[0:8]
    pred_bundle_dpath = pred_fpath.parent
    track_cfg_dname = f'tracking/trackcfg_{cfgstr}'
    track_cfg_base = pred_bundle_dpath / track_cfg_dname
    bas_out_fpath = track_cfg_base / 'tracks.json'

    if eval_dpath is None:
        iarpa_eval_dpath = track_cfg_base / 'iarpa_eval'
    else:
        iarpa_eval_dpath = eval_dpath / track_cfg_dname / 'iarpa_eval'

    iarpa_merge_fpath = iarpa_eval_dpath / 'scores' / 'merged' / 'summary2.json'

    bas_suggestions = {
        'iarpa_eval_dpath': iarpa_eval_dpath,
        'bas_cfgstr': cfgstr,
        'bas_out_fpath': bas_out_fpath,
        'iarpa_merge_fpath': iarpa_merge_fpath,
    }
    return bas_suggestions


def _suggest_act_paths(pred_fpath, actcfg, eval_dpath=None):
    """
    Helper for reasonable paths to keep everything organized for tracking eval
    """
    human_opts = ub.dict_isect(actcfg, {})
    other_opts = ub.dict_diff(actcfg, human_opts)
    if len(human_opts):
        human_part = ub.repr2(human_opts, compact=1) + '_'
    else:
        human_part = ''
    cfgstr = human_part + ub.hash_data(other_opts)[0:8]
    pred_bundle_dpath = pred_fpath.parent
    cfg_dname = f'actclf/actcfg_{cfgstr}'
    cfg_base = pred_bundle_dpath / cfg_dname
    act_out_fpath = cfg_base / 'activity_tracks.json'

    if eval_dpath is None:
        iarpa_eval_dpath = cfg_base / 'iarpa_sc_eval'
    else:
        iarpa_eval_dpath = eval_dpath / cfg_dname / 'iarpa_sc_eval'

    iarpa_merge_fpath = iarpa_eval_dpath / 'scores' / 'merged' / 'summary3.json'

    act_suggestions = {
        'iarpa_eval_dpath': iarpa_eval_dpath,
        'act_cfgstr': cfgstr,
        'act_out_fpath': act_out_fpath,
        'iarpa_merge_fpath': iarpa_merge_fpath,
    }
    return act_suggestions


def _build_bas_track_job(pred_fpath, bas_out_fpath, bas_track_cfg):
    """
    Given a predicted kwcoco file submit tracking and iarpa eval jobs

    Args:
        pred_fpath (PathLike): path to predicted kwcoco file
    """
    import shlex
    pred_fpath = ub.Path(pred_fpath)

    cfg = bas_track_cfg.copy()

    from watch.utils.lightning_ext import util_globals

    if isinstance(cfg['thresh_hysteresis'], str):
        cfg['thresh_hysteresis'] = util_globals.restricted_eval(
            cfg['thresh_hysteresis'].format(**cfg))

    if cfg['moving_window_size'] is None:
        cfg['moving_window_size'] = 'heatmaps_to_polys'
    else:
        cfg['moving_window_size'] = 'heatmaps_to_polys_moving_window'

    track_kwargs_str = shlex.quote(json.dumps(cfg))
    bas_args = f'--default_track_fn saliency_heatmaps --track_kwargs {track_kwargs_str}'
    # Because BAS is the first step we want ensure we clear annotations so
    # everything that comes out is a track from BAS.
    sites_dpath = bas_out_fpath.parent / 'tracked_sites'
    command = ub.codeblock(
        fr'''
        python -m watch.cli.kwcoco_to_geojson \
            "{pred_fpath}" \
            {bas_args} \
            --clear_annots \
            --out_dir "{sites_dpath}" \
            --out_fpath "{bas_out_fpath}"
        ''')

    return command


def _build_sc_actclf_job(pred_fpath, region_model_dpath, act_out_fpath, actcfg):
    r"""
    Given a predicted kwcoco file submit tracking and iarpa eval jobs

    We use truth annotations so this can be scored independently of SC

    Notes:
        DVC_DPATH=$(smartwatch_dvc --hardware=hdd)
        PRED_DATASET=$DVC_DPATH/models/fusion/eval3_sc_candidates/pred/CropDrop3_SC_V006/pred_CropDrop3_SC_V006_epoch=71-step=18431/Cropped-Drop3-TA1-2022-03-10_combo_DL_s2_wv_vali.kwcoco/predcfg_464eb52f/pred.kwcoco.json
        SITE_SUMMARY_GLOB="$DVC_DPATH/annotations/region_models/KR_*.geojson"

        ANNOTATIONS_DPATH=$DVC_DPATH/annotations
        ls $ANNOTATIONS_DPATH

        python -m watch.tasks.fusion.predict \
                pass

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
            --enable_viz=True \
            ./tmp/site_models_stamp.json

    Args:
        pred_fpath (PathLike): path to predicted kwcoco file
        region_model_dpath (PathLike): path to IARPA region file ($dvc/annotations/region_models)
    """

    # SITE_SUMMARY_GLOB="$DVC_DPATH/annotations/region_models

    import shlex
    pred_fpath = ub.Path(pred_fpath)

    actclf_cfg = {
        'boundaries_as': 'polys',
    }
    actclf_cfg.update(actcfg)

    kwargs_str = shlex.quote(json.dumps(actclf_cfg))
    sc_args = f'--default_track_fn class_heatmaps --track_kwargs {kwargs_str}'

    site_summary_glob = (region_model_dpath / '*.geojson')

    sites_dpath = act_out_fpath.parent / 'classified_sites'
    command = ub.codeblock(
        fr'''
        python -m watch.cli.kwcoco_to_geojson \
            "{pred_fpath}" \
            --site_summary '{site_summary_glob}' \
            {sc_args} \
            --out_dir "{sites_dpath}" \
            --out_fpath "{act_out_fpath}"
        ''')
    return command


def _build_iarpa_eval_job(track_out_fpath, iarpa_merge_fpath, iarpa_eval_dpath,
                          annotations_dpath, name=None):
    import shlex
    tmp_dir = iarpa_eval_dpath / 'tmp'
    out_dir = iarpa_eval_dpath / 'scores'
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
    return command


r"""
# Note: change backend to tmux if slurm is not installed
DVC_DPATH=$(smartwatch_dvc --hardware=hdd)
DATASET_CODE=Aligned-Drop3-TA1-2022-03-10/
KWCOCO_BUNDLE_DPATH=$DVC_DPATH/$DATASET_CODE
VALI_FPATH=$KWCOCO_BUNDLE_DPATH/combo_LM_nowv_vali.kwcoco.json


python -m watch.tasks.fusion.schedule_evaluation schedule_evaluation \
        --gpus="0" \
        --model_globstr="$DVC_DPATH/models/fusion/eval3_candidates/packages/Drop3_SpotCheck_V323/Drop3_SpotCheck_V323_epoch=18-step=12976.pt" \
        --test_dataset="$VALI_FPATH" \
        --skip_existing=0 \
        --enable_pred=1 \
        --enable_eval=1 \
        --enable_track=1 \
        --enable_iarpa_eval=1 \
        --backend=serial --run=0


python -m watch.cli.kwcoco_to_geojson \
    "$HOME/data/dvc-repos/smart_watch_dvc-hdd/models/fusion/eval3_candidates/pred/Drop3_SpotCheck_V323/pred_Drop3_SpotCheck_V323_epoch=18-step=12976/Aligned-Drop3-TA1-2022-03-10_combo_LM_nowv_vali.kwcoco/predcfg_abd043ec/pred.kwcoco.json" \
        --default_track_fn saliency_heatmaps --track_kwargs '{"thresh": 0.1, "polygon_fn": "heatmaps_to_polys_moving_window"}' \
    --clear_annots \
    --out_dir   "./_tmp/_testbas2/tracks" \
    --out_fpath "./_tmp/_testbas2/tracks/tracks.json"


  python -m watch.cli.run_metrics_framework \
    --merge \
    --name testing \
    --gt_dpath "$HOME/data/dvc-repos/smart_watch_dvc-hdd/annotations" \
    --tmp_dir "./_tmp/_testbas2/iarpa_eval/tmp" \
    --out_dir "./_tmp/_testbas2/iarpa_eval/scores" \
    --merge_fpath "./_tmp/_testbas2/iarpa_eval/scores/merged/summary2.json" \
    --inputs_are_paths \
    "./_tmp/_testbas2/tracks/tracks.json"



python -m watch.cli.kwcoco_to_geojson \
    "$HOME/data/dvc-repos/smart_watch_dvc-hdd/models/fusion/eval3_candidates/pred/Drop3_SpotCheck_V323/pred_Drop3_SpotCheck_V323_epoch=18-step=12976/Aligned-Drop3-TA1-2022-03-10_combo_LM_nowv_vali.kwcoco/predcfg_abd043ec/pred.kwcoco.json" \
        --default_track_fn saliency_heatmaps --track_kwargs '{"thresh": 0.1}' \
    --clear_annots \
    --out_dir   "./_tmp/_testbas/tracks" \
    --out_fpath "./_tmp/_testbas/tracks/tracks.json"


  python -m watch.cli.run_metrics_framework \
    --merge \
    --name testing \
    --gt_dpath "$HOME/data/dvc-repos/smart_watch_dvc-hdd/annotations" \
    --tmp_dir "./_tmp/_testbas/iarpa_eval/tmp" \
    --out_dir "./_tmp/_testbas/iarpa_eval/scores" \
    --merge_fpath "./_tmp/_testbas/iarpa_eval/scores/merged/summary2.json" \
    --inputs_are_paths \
    "./_tmp/_testbas/tracks/tracks.json"






#### SC Notes:

curl https://raw.githubusercontent.com/Erotemic/local/main/init/utils.sh -o utils.sh
source utils.sh


export DVC_DPATH=$(smartwatch_dvc --hardware=hdd)
DATASET_CODE=Cropped-Drop3-TA1-2022-03-10
KWCOCO_BUNDLE_DPATH=$DVC_DPATH/$DATASET_CODE
VALI_FPATH=$KWCOCO_BUNDLE_DPATH/combo_DL_s2_wv_vali.kwcoco.json

ls $DVC_DPATH/models/fusion/eval3_sc_candidates/pred/*/*/*/*/pred.kwcoco.json
ls $DVC_DPATH/models/fusion/eval3_sc_candidates/eval/*/*/*/*/

MODEL_GLOB_PARTS=(
    $DVC_DPATH/
    models/fusion/eval3_sc_candidates/packages/
    CropDrop3_SC_wvonly_D_V011/
    CropDrop3_SC_wvonly_D_V011_epoch=81-step=167935.pt
)
MODEL_GLOBSTR=$(join_by "" "${MODEL_GLOB_PARTS[@]}")

python -m watch.tasks.fusion.schedule_evaluation schedule_evaluation \
        --gpus="0" \
        --model_globstr="$MODEL_GLOBSTR" \
        --test_dataset="$VALI_FPATH" \
        --skip_existing=0 \
        --enable_pred=1 \
        --enable_eval=1 \
        --enable_actclf=1 \
        --enable_actclf_eval=1 \
        --backend=serial --run=0

PRED_PATH_PART=(
    $DVC_DPATH/
    models/fusion/eval3_sc_candidates/pred/CropDrop3_SC_wvonly_D_V011/
    pred_CropDrop3_SC_wvonly_D_V011_epoch=81-step=167935/
    Cropped-Drop3-TA1-2022-03-10_combo_DL_s2_wv_vali.kwcoco/
    predcfg_abd043ec/
    pred.kwcoco.json
)
PRED_DATASET=$(join_by "" "${PRED_PATH_PART[@]}")
echo "PRED_DATASET = $PRED_DATASET"

#SITE_SUMMARY_GLOB="$DVC_DPATH/annotations/region_models/KR_*.geojson"
SITE_SUMMARY_GLOB="$DVC_DPATH/annotations/region_models/KR_R001.geojson"

python -m watch.cli.kwcoco_to_geojson \
    "$PRED_DATASET" \
    --default_track_fn class_heatmaps \
    --site_summary "$SITE_SUMMARY_GLOB" \
    --track_kwargs '{"boundaries_as": "polys"}' \
    --out_dir ./tmp/site_models \
    --out_fpath ./tmp/site_models_stamp.json

python -m watch.cli.run_metrics_framework \
    --merge \
    --gt_dpath "$DVC_DPATH/annotations" \
    --tmp_dir "./tmp/iarpa/tmp" \
    --out_dir "./tmp/iarpa/scores" \
    --name "mytest" \
    --merge_fpath "./tmp/iarpa/merged.json" \
    --inputs_are_paths \
    --enable_viz=True \
    ./tmp/site_models_stamp.json


"""
