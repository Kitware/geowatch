def main():
    """
    import sys, ubelt
    sys.path.append(ubelt.expandpath('~/code/watch/dev/reports'))
    from report_2022_09_xx import *  # NOQA
    """
    import ubelt as ub
    import pandas as pd  # NOQA
    from watch import heuristics
    from watch.mlops import expt_manager
    from watch.mlops import expt_report
    expt_dvc_dpath = heuristics.auto_expt_dvc()
    data_dvc_dpath = heuristics.auto_expt_dvc()

    # I messed up the name of the dataset I was working on.
    # it is marked as train, but it should have been vali.

    # dataset_code = 'Aligned-Drop4-2022-08-08-TA1-S2-L8-ACC'
    dataset_code = '*'

    state = expt_manager.ExperimentState(
        expt_dvc_dpath, dataset_code=dataset_code,
        data_dvc_dpath=data_dvc_dpath, model_pattern='*')
    self = state  # NOQA

    # state.patterns['test_dset'] = 'Aligned-Drop4-2022-08-08-TA1-S2-L8-ACC_data_train_subset.kwcoco'
    # state.patterns['test_dset'] = (
    #     'Aligned-Drop4-2022-08-08-TA1-S2-L8-ACC_data_vali_10GSD_KR_R001.kwcoco')
    # state.patterns['test_dset'] = (
    #     'Aligned-Drop4-2022-08-08-TA1-S2-L8-ACC_data_septest.kwcoco')
    # state.patterns['test_dset'] = ('Aligned-Drop4-2022-08-08-TA1-S2-L8-ACC_data_kr1br2.kwcoco')
    state.patterns['test_trk_dset'] = 'Aligned-Drop4-2022-08-08-TA1-S2-L8-ACC_data.kwcoco'
    state.patterns['test_act_dset'] = 'NA'
    # state.patterns['test_dset'] = ('*')

    state._build_path_patterns()
    state.summarize()
    # state._make_cross_links()
    # state._block_non_existing_rhashes()

    reporter = expt_report.EvaluationReporter(state)
    reporter.load1()
    reporter.load2()

    reporter.state.summarize()
    df = reporter.orig_merged_df

    non_dotted_cols = ub.oset([c for c in df.columns if '.' not in c])
    non_dotted_cols = non_dotted_cols - {
        'expt_dvc_dpath', 'raw', 'dvc', 'has_dvc', 'has_raw', 'needs_pull',
        'is_link', 'is_broken', 'unprotected', 'needs_push',
        'dataset_code', 'has_teamfeat',
        'crop_id', 'crop_cfg',
    }
    print(df[list(non_dotted_cols)].to_string())
    print(df[['type', 'trk.poly.thresh']].value_counts(dropna=False))

    # dpath = reporter.dpath
    dpath = ub.Path.appdir('watch/expt-report/2022-10-xx').ensuredir()

    # Dump details out about the best models
    cohort = ub.timestamp()
    best_models_dpath = (dpath / 'best_models' / cohort).ensuredir()
    groupid_to_shortlist = reporter.report_best(show_configs=True, verbose=1, top_k=4)

    viz_cmds = []

    # from watch.utils import util_param_grid
    colnames = ub.oset(reporter.orig_merged_df.columns)
    # column_nestings = util_param_grid.dotkeys_to_nested(colnames)
    dotted = ub.oset([c for c in colnames if '.' in c])
    metric_cols = ub.oset([c for c in dotted if 'metrics.' in c])
    meta_cols = ub.oset([c for c in dotted if 'meta.' in c])
    resource_cols = ub.oset([c for c in dotted if 'resource.' in c])
    fit_cols = ub.oset([c for c in dotted if 'fit.' in c])
    param_cols = dotted - (metric_cols | fit_cols | resource_cols | meta_cols)

    # df['trk.pxl.properties.test_dataset_fname'] = df['trk.pxl.test_dataset'].apply(lambda x: ub.Path(x).name)
    # flags = df['trk.pxl.properties.test_dataset_fname'] == 'data.kwcoco.json'
    # df = df[flags]
    # TODO: shrink rows where data isn't all nan

    # subdf = df[~df['trk.poly.thresh'].isnull()]
    # subdf = subdf[['trk.poly.thresh', 'trk.poly.metrics.bas_f1']]
    # ~subdf.isnull(axis=1)
    # trk.poly.metrics.]]
    # df[~df['trk.poly.thresh'].isnull()]

    cols_of_interest = ['trk.poly.thresh', 'trk.poly.metrics.bas_f1', 'trk.poly.metrics.bas_tp', 'trk.poly.metrics.bas_fp', 'trk.poly.metrics.bas_fn', 'trk.poly.metrics.bas_ppv', 'trk.poly.metrics.bas_tpr']
    col_order = (ub.oset(cols_of_interest) | metric_cols) & ub.oset(df.columns)
    import rich
    df2 = df[cols_of_interest].sort_values('trk.poly.thresh')
    rich.print(df2)

    import kwplot
    sns = kwplot.autosns()
    sns.lineplot(data=df2, x='trk.poly.metrics.bas_fp', y='trk.poly.metrics.bas_fp')
    sns.lineplot(data=df2, x='trk.poly.metrics.bas_tpr', y='trk.poly.metrics.bas_ppv')
    sns.lineplot(data=df2, x='trk.poly.thresh', y='trk.poly.metrics.bas_f1')
    sns.lineplot(data=df2, x='trk.poly.metrics.bas_tpr', y='trk.poly.metrics.bas_ppv')
    sns.lineplot(data=df2, x='trk.poly.metrics.bas_tpr', y='trk.poly.metrics.bas_ppv')
    sns.lineplot(data=df2, x='trk.poly.metrics.bas_fp', y='trk.poly.metrics.bas_tpr')
    # hue='trk.poly.thresh')

    df.loc[:, df.columns & (list(metric_cols) + ['trk.poly.thresh'])]


def shrink_na_cols(df):
    import pandas as pd
    isnull = pd.isnull(df)
    df = df.loc[:, ~isnull.all(axis=0)]
    return df

"""

Test connors models:

DATASET_CODE=Aligned-Drop4-2022-08-08-TA1-S2-L8-ACC
DATA_DVC_DPATH=$(smartwatch_dvc --tags="phase2_data" --hardware="ssd")
DVC_EXPT_DPATH=$(smartwatch_dvc --tags="phase2_expt")
# TEST_DATASET=$DATA_DVC_DPATH/$DATASET_CODE/data.kwcoco.json
TEST_DATASET=$DATA_DVC_DPATH/$DATASET_CODE/BR_R001.kwcoco.json


python -m watch.mlops.schedule_evaluation \
    --params="
        matrix:
            trk.pxl.model:
                # - $DVC_EXPT_DPATH/models/fusion/Aligned-Drop4-2022-08-08-TA1-S2-L8-ACC/packages/Drop4_BAS_Multi_Native/package_epoch10_step200000.pt
                - /home/joncrall/remote/toothbrush/data/dvc-repos/smart_expt_dvc/training/toothbrush/joncrall/Aligned-Drop4-2022-08-08-TA1-S2-L8-ACC/runs/Drop4_BAS_invariants_30GSD_V016/lightning_logs/version_4/checkpoints/Drop4_BAS_invariants_30GSD_V016_epoch=10-step=5632.pt
                # - $DVC_EXPT_DPATH/models/fusion/eval3_candidates/packages/Drop3_SpotCheck_V323/Drop3_SpotCheck_V323_epoch=18-step=12976.pt
            trk.pxl.data.test_dataset:
                - $DATA_DVC_DPATH/$DATASET_CODE/combo_BR_R001_I.kwcoco.json
                - $DATA_DVC_DPATH/$DATASET_CODE/combo_KR_R001_I.kwcoco.json
                - $DATA_DVC_DPATH/$DATASET_CODE/combo_KR_R002_I.kwcoco.json
                - $DATA_DVC_DPATH/$DATASET_CODE/combo_US_R007_I.kwcoco.json
            trk.pxl.data.window_scale_space:
                - "auto"
            trk.pxl.data.time_sampling:
                - "auto"
            trk.pxl.data.input_scale_space:
                - "auto"
            trk.poly.thresh:
                - 0.1
            crop.src:
                - /home/joncrall/remote/toothbrush/data/dvc-repos/smart_data_dvc/online_v1/kwcoco_for_sc_fielded.json
            crop.regions:
                - trk.poly.output
            act.pxl.data.test_dataset:
                - /home/joncrall/remote/toothbrush/data/dvc-repos/smart_expt_dvc/models/fusion/Aligned-Drop4-2022-08-08-TA1-S2-L8-ACC/crop/online_v1_kwcoco_for_sc_fielded/trk_poly_id_0408400f/crop_f64d5b9a/crop_id_59ed6e1b/crop.kwcoco.json
            act.pxl.data.input_scale_space:
                - 3GSD
            act.pxl.data.time_steps:
                - 3
            act.pxl.data.chip_overlap:
                - 0.35
            act.poly.thresh:
                - 0.01
            act.poly.use_viterbi:
                - 0
            act.pxl.model:
                - $DVC_EXPT_DPATH/models/fusion/Aligned-Drop4-2022-08-08-TA1-S2-WV-PD-ACC/packages/Drop4_SC_RGB_scratch_V002/Drop4_SC_RGB_scratch_V002_epoch=99-step=50300-v1.pt.pt
        include:
            - act.pxl.data.chip_dims: 256,256
              act.pxl.data.window_scale_space: 3GSD
              act.pxl.data.input_scale_space: 3GSD
              act.pxl.data.output_scale_space: 3GSD
    " \
    --enable_pred_trk_pxl=1 \
    --enable_pred_trk_poly=1 \
    --enable_eval_trk_pxl=1 \
    --enable_eval_trk_poly=1 \
    --enable_crop=0 \
    --enable_pred_act_pxl=0 \
    --enable_pred_act_poly=0 \
    --enable_eval_act_pxl=0 \
    --enable_eval_act_poly=0 \
    --enable_viz_pred_trk_poly=0 \
    --enable_viz_pred_act_poly=0 \
    --enable_links=1 \
    --devices="1," --queue_size=2 \
    --queue_name='nov-eval' \
    --backend=tmux --skip_existing=1 \
    --run=0




# HOROLOGIC
DATASET_CODE=Drop4-BAS
DATA_DVC_DPATH=$(smartwatch_dvc --tags="phase2_data" --hardware="ssd")
DVC_EXPT_DPATH=$(smartwatch_dvc --tags="phase2_expt")
# TEST_DATASET=$DATA_DVC_DPATH/$DATASET_CODE/data.kwcoco.json
TEST_DATASET=$DATA_DVC_DPATH/$DATASET_CODE/BR_R001.kwcoco.json
python -m watch.mlops.schedule_evaluation \
    --params="
        matrix:
            trk.pxl.model:
                - $DVC_EXPT_DPATH/models/fusion/eval3_candidates/packages/Drop3_SpotCheck_V323/Drop3_SpotCheck_V323_epoch=18-step=12976.pt
            trk.pxl.data.test_dataset:
                - $DATA_DVC_DPATH/$DATASET_CODE/BR_R002.kwcoco.json
                - $DATA_DVC_DPATH/$DATASET_CODE/KR_R001.kwcoco.json
                - $DATA_DVC_DPATH/$DATASET_CODE/KR_R002.kwcoco.json
                - $DATA_DVC_DPATH/$DATASET_CODE/US_R007.kwcoco.json
            trk.pxl.data.window_scale_space:
                - "auto"
            trk.pxl.data.input_scale_space:
                # - "auto"
                - 10GSD
            trk.pxl.data.time_sampling:
                - "auto"
            trk.poly.thresh:
                - 0.1
            crop.src:
                - /home/joncrall/remote/toothbrush/data/dvc-repos/smart_data_dvc/online_v1/kwcoco_for_sc_fielded.json
            crop.regions:
                - trk.poly.output
            act.pxl.data.test_dataset:
                - /home/joncrall/remote/toothbrush/data/dvc-repos/smart_expt_dvc/models/fusion/Aligned-Drop4-2022-08-08-TA1-S2-L8-ACC/crop/online_v1_kwcoco_for_sc_fielded/trk_poly_id_0408400f/crop_f64d5b9a/crop_id_59ed6e1b/crop.kwcoco.json
            act.pxl.data.input_scale_space:
                - 3GSD
            act.pxl.data.time_steps:
                - 3
            act.pxl.data.chip_overlap:
                - 0.35
            act.poly.thresh:
                - 0.01
            act.poly.use_viterbi:
                - 0
            act.pxl.model:
                - $DVC_EXPT_DPATH/models/fusion/Aligned-Drop4-2022-08-08-TA1-S2-WV-PD-ACC/packages/Drop4_SC_RGB_scratch_V002/Drop4_SC_RGB_scratch_V002_epoch=99-step=50300-v1.pt.pt
        include:
            - act.pxl.data.chip_dims: 256,256
              act.pxl.data.window_scale_space: 3GSD
              act.pxl.data.input_scale_space: 3GSD
              act.pxl.data.output_scale_space: 3GSD
            - trk.pxl.data.window_scale_space: 10GSD
              trk.pxl.data.input_scale_space: 10GSD
              trk.pxl.data.output_scale_space: 10GSD
            - trk.pxl.data.window_scale_space: 15GSD
              trk.pxl.data.input_scale_space: 15GSD
              trk.pxl.data.output_scale_space: 15GSD
            - trk.pxl.data.window_scale_space: auto
              trk.pxl.data.input_scale_space: auto
              trk.pxl.data.output_scale_space: auto
    " \
    --enable_pred_trk_pxl=1 \
    --enable_pred_trk_poly=1 \
    --enable_eval_trk_pxl=1 \
    --enable_eval_trk_poly=1 \
    --enable_crop=0 \
    --enable_pred_act_pxl=0 \
    --enable_pred_act_poly=0 \
    --enable_eval_act_pxl=0 \
    --enable_eval_act_poly=0 \
    --enable_viz_pred_trk_poly=0 \
    --enable_viz_pred_act_poly=0 \
    --enable_links=1 \
    --devices="1," --queue_size=2 \
    --queue_name='nov-eval' \
    --backend=tmux --skip_existing=1 \
    --run=1


python -m watch.cli.run_metrics_framework --merge=True --name Drop4_BAS_invariants_30GSD_V016_epoch=10-step=5632.pt-trk_pxl_ca8e6033-trk_poly_9f08fb8c \
    --true_site_dpath /home/joncrall/remote/toothbrush/data/dvc-repos/smart_data_dvc/annotations/site_models \
    --true_region_dpath /home/joncrall/remote/toothbrush/data/dvc-repos/smart_data_dvc/annotations/region_models \
    --pred_sites /home/joncrall/remote/toothbrush/data/dvc-repos/smart_expt_dvc/models/fusion/dset_code_unknown/pred/trk/Drop4_BAS_invariants_30GSD_V016_epoch=10-step=5632.pt/Aligned-Drop4-2022-08-08-TA1-S2-L8-ACC_combo_KR_R001_I.kwcoco/trk_pxl_ca8e6033/trk_poly_9f08fb8c/site_tracks_manifest.json \
    --tmp_dir /home/joncrall/remote/toothbrush/data/dvc-repos/smart_expt_dvc/models/fusion/dset_code_unknown/eval/trk/Drop4_BAS_invariants_30GSD_V016_epoch=10-step=5632.pt/Aligned-Drop4-2022-08-08-TA1-S2-L8-ACC_combo_KR_R001_I.kwcoco/trk_pxl_ca8e6033/trk_poly_9f08fb8c/_tmp \
    --out_dir /home/joncrall/remote/toothbrush/data/dvc-repos/smart_expt_dvc/models/fusion/dset_code_unknown/eval/trk/Drop4_BAS_invariants_30GSD_V016_epoch=10-step=5632.pt/Aligned-Drop4-2022-08-08-TA1-S2-L8-ACC_combo_KR_R001_I.kwcoco/trk_pxl_ca8e6033/trk_poly_9f08fb8c \
    --merge_fpath /home/joncrall/remote/toothbrush/data/dvc-repos/smart_expt_dvc/models/fusion/dset_code_unknown/eval/trk/Drop4_BAS_invariants_30GSD_V016_epoch=10-step=5632.pt/Aligned-Drop4-2022-08-08-TA1-S2-L8-ACC_combo_KR_R001_I.kwcoco/trk_pxl_ca8e6033/trk_poly_9f08fb8c/merged/summary2.json




"""

# ------

"""
SC

DATASET_CODE=Drop4-SC
DATA_DVC_DPATH=$(smartwatch_dvc --tags="phase2_data" --hardware="ssd")
DVC_EXPT_DPATH=$(smartwatch_dvc --tags="phase2_expt")
ls $DATA_DVC_DPATH/$DATASET_CODE
TEST_DATASET=$DATA_DVC_DPATH/$DATASET_CODE/data_vali.kwcoco.json
python -m watch.mlops.schedule_evaluation \
    --params="
        matrix:
            trk.pxl.model:
                - foo
            trk.pxl.data.test_dataset:
                - $DATA_DVC_DPATH/$DATASET_CODE/combo_US_R007_I.kwcoco.json
            trk.pxl.data.window_scale_space:
                - "auto"
            trk.pxl.data.time_sampling:
                - "auto"
            trk.pxl.data.input_scale_space:
                - "auto"
            trk.poly.thresh:
                - 0.1
            crop.src:
                - /home/joncrall/remote/toothbrush/data/dvc-repos/smart_data_dvc/online_v1/kwcoco_for_sc_fielded.json
            crop.regions:
                - trk.poly.output
            act.pxl.data.test_dataset:
                - $DATA_DVC_DPATH/$DATASET_CODE/data_vali.kwcoco.json
            act.pxl.data.input_scale_space:
                - 3GSD
            act.pxl.data.time_steps:
                - 12
            act.pxl.data.chip_overlap:
                - 0.3
            act.poly.thresh:
                - 0.01
            act.poly.use_viterbi:
                - 0
            act.pxl.model:
                - $DVC_EXPT_DPATH/models/fusion/eval3_sc_candidates/packages/CropDrop3_SC_s2wv_invar_scratch_V030/CropDrop3_SC_s2wv_invar_scratch_V030_epoch=78-step=53956-v1.pt
        include:
            - act.pxl.data.chip_dims: 256,256
              act.pxl.data.window_scale_space: 3GSD
              act.pxl.data.input_scale_space: 3GSD
              act.pxl.data.output_scale_space: 3GSD
    " \
    --enable_pred_trk_pxl=0 \
    --enable_pred_trk_poly=0 \
    --enable_eval_trk_pxl=0 \
    --enable_eval_trk_poly=0 \
    --enable_crop=0 \
    --enable_pred_act_pxl=1 \
    --enable_pred_act_poly=1 \
    --enable_eval_act_pxl=1 \
    --enable_eval_act_poly=1 \
    --enable_viz_pred_trk_poly=0 \
    --enable_viz_pred_act_poly=0 \
    --enable_links=1 \
    --devices="1," --queue_size=2 \
    --queue_name='nov-eval' \
    --backend=tmux --skip_existing=1 \
    --run=0




kwcoco subset \
        --src ~/data/dvc-repos/smart_data_dvc-ssd/Drop4-SC/data_vali.kwcoco.json \
        --dst ~/data/dvc-repos/smart_data_dvc-ssd/Drop4-SC/US_R007_0055_box.kwcoco.json \
        --select_videos '.name == "US_R007_0055_box"'

python -m watch.tasks.fusion.predict \
    --package_fpath ~/data/dvc-repos/smart_expt_dvc/models/fusion/eval3_sc_candidates/packages/CropDrop3_SC_s2wv_invar_scratch_V030/CropDrop3_SC_s2wv_invar_scratch_V030_epoch=78-step=53956-v1.pt \
    --test_dataset ~/data/dvc-repos/smart_data_dvc-ssd/Drop4-SC/US_R007_0055_box.kwcoco.json \
    --pred_dataset ~/data/dvc-repos/smart_expt_dvc/models/fusion/eval3_sc_candidates/pred/act/CropDrop3_SC_s2wv_invar_scratch_V030_epoch=78-step=53956-v1/Drop4-SC_data_vali.kwcoco/act_pxl_d31324d9/test/pred.kwcoco.json \
    --input_scale_space=3GSD \
    --time_steps=12 \
    --chip_overlap=0.3  \
    --num_workers=4 \
    --devices=0, \
    --accelerator=gpu \
    --batch_size=1


/home/joncrall/remote/toothbrush/data/dvc-repos/smart_expt_dvc/models/fusion/eval3_sc_candidates/pred/act/CropDrop3_SC_s2wv_invar_scratch_V030_epoch=78-step=53956-v1/Drop4-SC_data_vali.kwcoco/act_pxl_d31324d9/_viz_act_pxl_d31324d9_pred.kwcoco_d4d18d08
"""
