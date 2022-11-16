"""

python -m watch mlops "status" --dataset_codes Drop4-SC
python -m watch mlops "push packages" --dataset_codes Drop4-SC
python -m watch mlops "pull packages" --dataset_codes Drop4-SC



models/fusion/Drop4-SC/packages/Drop4_tune_V30_V2/Drop4_tune_V30_V2_epoch=6-step=83790.pt.pt
models/fusion/Drop4-SC/packages/Drop4_tune_V30_8GSD_V3/package_epoch3_step22551.pt.pt
models/fusion/Drop4-SC/packages/Drop4_tune_V30_2GSD_V3/package_epoch0_step57.pt.pt



# Quick validation sites
KR_R001_0000_box
KR_R001_0015_box
KR_R002_0025_box
KR_R002_0030_box  # negative
US_R007_0045_box
US_R007_0015_box

python -m watch find_dvc   --hardware=auto --tags=phase2_expt

DATASET_CODE=Drop4-SC
DATA_DVC_DPATH=$(smartwatch_dvc --tags="phase2_data" --hardware="auto")
EXPT_DVC_DPATH=$(smartwatch_dvc --tags="phase2_expt" --hardware="auto")

echo "
DATA_DVC_DPATH = $DATA_DVC_DPATH
EXPT_DVC_DPATH = $EXPT_DVC_DPATH
"

# Setup the small dataset
kwcoco subset \
    --src $DATA_DVC_DPATH/Drop4-SC/data_vali.kwcoco.json \
    --dst $DATA_DVC_DPATH/Drop4-SC/data_vali_small.kwcoco.json \
    --select_videos '((.name == "KR_R001_0000_box") or
                      (.name == "KR_R001_0015_box") or
                      (.name == "KR_R002_0025_box") or
                      (.name == "KR_R002_0030_box") or
                      (.name == "US_R007_0045_box") or
                      (.name == "US_R007_0015_box"))'



{'Drop4_tune_V30_8GSD_V3_epoch=0-step=5778-v1.pt',
 'Drop4_tune_V30_8GSD_V3_epoch=1-step=11556.pt',
 'Drop4_tune_V30_8GSD_V3_epoch=2-step=17334.pt',
 'Drop4_tune_V30_V2_epoch=1-step=23940.pt',
 'Drop4_tune_V30_V2_epoch=2-step=35910-v1.pt',
 'Drop4_tune_V30_V2_epoch=3-step=47880-v1.pt',
 'package_epoch0_step171.pt',
 'package_epoch0_step57.pt',
 'package_epoch0_step6587.pt',
 'package_epoch3_step22551.pt'}



DATASET_CODE=Drop4-SC
DATA_DVC_DPATH=$(smartwatch_dvc --tags="phase2_data" --hardware="auto")
EXPT_DVC_DPATH=$(smartwatch_dvc --tags="phase2_expt" --hardware="auto")
python -m watch.mlops.schedule_evaluation \
    --params="
        matrix:
            trk.pxl.model: unused
            trk.pxl.data.test_dataset: unused
            trk.pxl.data.window_space_scale: unused
            trk.pxl.data.time_sampling: unused
            trk.pxl.data.input_space_scale: unused
            trk.poly.thresh: unused
            crop.src: unused
            crop.regions: truth
            act.pxl.data.test_dataset:
                - $DATA_DVC_DPATH/$DATASET_CODE/data_vali_small.kwcoco.json
            act.pxl.data.input_space_scale:
                # - auto
                - 8GSD
            act.pxl.data.time_steps:
                - auto
            act.pxl.data.chip_overlap:
                - 0.3
            act.poly.thresh:
                - 0.07
                - 0.1
                - 0.13
            act.poly.use_viterbi:
                - 0
            act.pxl.model:
                - $EXPT_DVC_DPATH/models/fusion/Drop4-SC/packages/Drop4_tune_V30_V2/Drop4_tune_V30_V2_epoch=1-step=23940.pt.pt
                - $EXPT_DVC_DPATH/models/fusion/Drop4-SC/packages/Drop4_tune_V30_V2/Drop4_tune_V30_V2_epoch=2-step=35910-v1.pt.pt
                - $EXPT_DVC_DPATH/models/fusion/Drop4-SC/packages/Drop4_tune_V30_V2/Drop4_tune_V30_V2_epoch=3-step=47880-v1.pt.pt
                - $EXPT_DVC_DPATH/models/fusion/Drop4-SC/packages/Drop4_tune_V30_8GSD_V3/package_epoch3_step22551.pt.pt
                - $EXPT_DVC_DPATH/models/fusion/Drop4-SC/packages/Drop4_tune_V30_2GSD_V3/package_epoch0_step57.pt.pt
                - $EXPT_DVC_DPATH/models/fusion/Drop4-SC/packages/Drop4_tune_V30_V2/package_epoch0_step171.pt.pt
                - $EXPT_DVC_DPATH/models/fusion/Drop4-SC/packages/Drop4_tune_V30_V2/package_epoch0_step6587.pt.pt
                - $EXPT_DVC_DPATH/models/fusion/Drop4-SC/packages/Drop4_tune_V30_8GSD_V3/Drop4_tune_V30_8GSD_V3_epoch=0-step=5778-v1.pt.pt
                - $EXPT_DVC_DPATH/models/fusion/Drop4-SC/packages/Drop4_tune_V30_8GSD_V3/Drop4_tune_V30_8GSD_V3_epoch=1-step=11556.pt.pt
                - $EXPT_DVC_DPATH/models/fusion/Drop4-SC/packages/Drop4_tune_V30_8GSD_V3/Drop4_tune_V30_8GSD_V3_epoch=2-step=17334.pt.pt

                # - $EXPT_DVC_DPATH/models/fusion/Drop4-SC/packages/Drop4_tune_V30_V2/Drop4_tune_V30_V2_epoch=4-step=59850.pt.pt
                # - $EXPT_DVC_DPATH/models/fusion/Drop4-SC/packages/Drop4_tune_V30_V2/Drop4_tune_V30_V2_epoch=5-step=71820.pt.pt
                # - $EXPT_DVC_DPATH/models/fusion/Drop4-SC/packages/Drop4_tune_V30_V2/Drop4_tune_V30_V2_epoch=6-step=83790.pt.pt
                # - $EXPT_DVC_DPATH/models/fusion/Drop4-SC/packages/Drop4_tune_V30_V2/Drop4_tune_V30_V2_epoch=0-step=11970.pt.pt
                # - $EXPT_DVC_DPATH/models/fusion/Drop4-SC/packages/Drop4_tune_V30_V2/package_epoch0_step0.pt.pt
                # - $EXPT_DVC_DPATH/models/fusion/Drop4-SC/packages/Drop4_tune_V30_V2/package_epoch0_step11970.pt.pt
                # - $EXPT_DVC_DPATH/models/fusion/Drop4-SC/packages/Drop4_tune_V30_V2/package_epoch0_step1661.pt.pt
                # - $EXPT_DVC_DPATH/models/fusion/Drop4-SC/packages/Drop4_tune_V30_V2/package_epoch7_step95760.pt.pt
            include:
                - act.pxl.data.chip_dims: 256,256
                  act.pxl.data.window_space_scale: 8GSD
                  act.pxl.data.input_space_scale: 8GSD
                  act.pxl.data.output_space_scale: 8GSD
                # - act.pxl.data.chip_dims: 256,256
                #   act.pxl.data.window_space_scale: 4GSD
                #   act.pxl.data.input_space_scale: 4GSD
                #   act.pxl.data.output_space_scale: 4GSD
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
    --devices="0,1" --queue_size=2 \
    --queue_name='nov-sc-eval2' \
    --backend=tmux --skip_existing=1 \
    --run=1


python -m watch.cli.run_tracker /home/joncrall/remote/Ooo/data/dvc-repos/smart_expt_dvc/models/fusion/Drop4-SC/pred/act/package_epoch3_step22551.pt/Drop4-SC_data_vali_small.kwcoco/act_pxl_e836a34c/pred.kwcoco.json --default_track_fn class_heatmaps --track_kwargs '{"boundaries_as": "polys", "thresh": 0.1, "use_viterbi": 0}' --site_summary /home/joncrall/remote/Ooo/data/dvc-repos/smart_expt_dvc/models/fusion/dset_code_unknown/pred/trk/foo/Drop4-SC_combo_US_R007_I.kwcoco/trk_pxl_ca8e6033/trk_poly_9f08fb8c/site_summary_tracks_manifest.json --out_sites_fpath /home/joncrall/remote/Ooo/data/dvc-repos/smart_expt_dvc/models/fusion/Drop4-SC/pred/act/package_epoch3_step22551.pt/Drop4-SC_data_vali_small.kwcoco/act_pxl_e836a34c/act_poly_e50c1c4f/site_activity_manifest.json --out_sites_dir /home/joncrall/remote/Ooo/data/dvc-repos/smart_expt_dvc/models/fusion/Drop4-SC/pred/act/package_epoch3_step22551.pt/Drop4-SC_data_vali_small.kwcoco/act_pxl_e836a34c/act_poly_e50c1c4f/sites --out_kwcoco /home/joncrall/remote/Ooo/data/dvc-repos/smart_expt_dvc/models/fusion/Drop4-SC/pred/act/package_epoch3_step22551.pt/Drop4-SC_data_vali_small.kwcoco/act_pxl_e836a34c/act_poly_e50c1c4f/activity_tracks.kwcoco.json


TEST_DATASET=$DATA_DVC_DPATH/$DATASET_CODE/BR_R001.kwcoco.json

"""


def bas_report():
    import rich  # NOQA
    import ubelt as ub  # NOQA
    import pandas as pd  # NOQA
    from watch import heuristics
    from watch.mlops import expt_state
    from watch.mlops import expt_report
    expt_dvc_dpath = heuristics.auto_expt_dvc()
    data_dvc_dpath = heuristics.auto_expt_dvc()

    # I messed up the name of the dataset I was working on.
    # it is marked as train, but it should have been vali.
    # dataset_code = ['Drop4-BAS', 'dset_code_unknown', 'eval3_candidates', 'Aligned-Drop4-2022-08-08-TA1-S2-L8-ACC']
    dataset_code = ['Drop4-BAS']
    # 'Aligned-Drop4-2022-08-08-TA1-S2-L8-ACC']
    # dataset_code = '*'
    state = expt_state.ExperimentState(
        expt_dvc_dpath, dataset_code=dataset_code,
        data_dvc_dpath=data_dvc_dpath, model_pattern='*')
    self = state  # NOQA
    state._build_path_patterns()
    state.summarize()

    reporter = expt_report.EvaluationReporter(state)
    reporter.load1()
    reporter.load2()

    df = reporter.orig_merged_df
    from watch.utils.util_param_grid import DotDictDataFrame
    df = DotDictDataFrame(df)
    groupid_to_shortlist = reporter.report_best(show_configs=True, verbose=1, top_k=10)


def main():
    """
    import sys, ubelt
    sys.path.append(ubelt.expandpath('~/code/watch/dev/reports'))
    from report_2022_09_xx import *  # NOQA
    """
    import rich
    import ubelt as ub
    import pandas as pd  # NOQA
    from watch import heuristics
    from watch.mlops import expt_state
    from watch.mlops import expt_report
    expt_dvc_dpath = heuristics.auto_expt_dvc()
    data_dvc_dpath = heuristics.auto_expt_dvc()

    # I messed up the name of the dataset I was working on.
    # it is marked as train, but it should have been vali.
    dataset_code = 'Drop4-SC'
    state = expt_state.ExperimentState(
        expt_dvc_dpath, dataset_code=dataset_code,
        data_dvc_dpath=data_dvc_dpath, model_pattern='*')
    self = state  # NOQA
    state._build_path_patterns()
    state.summarize()

    reporter = expt_report.EvaluationReporter(state)
    reporter.load1()
    reporter.load2()

    reporter.state.summarize()
    df = reporter.orig_merged_df

    groupid_to_shortlist = reporter.report_best(show_configs=True, verbose=1, top_k=4)

    from watch.utils.util_param_grid import DotDictDataFrame
    sdf = DotDictDataFrame(groupid_to_shortlist[('Drop4-SC_data_vali_small.kwcoco', 'eval_act_poly_fpath')])

    keepers = []
    for x in groupid_to_shortlist.values():
        keepers.extend(x['act_model'])

    from watch.utils.util_param_grid import DotDictDataFrame
    df = DotDictDataFrame(df)
    print('df.nested_columns = {}'.format(ub.repr2(df.nested_columns, nl=True)))

    space = DotDictDataFrame(df['input_space_scale'])
    (space['act.fit.input_space_scale'] != space['act.pxl.input_space_scale']).sum()

    act_poly_df = DotDictDataFrame(df[df['type'] == 'eval_act_poly_fpath'])

    rich.print(act_poly_df.sort_values('act.poly.metrics.sc_macro_f1').iloc[-3:].drop(['raw'], axis=1).T.to_string())

    dotted = df.find_columns('*.*')
    metric_cols = df.find_columns('*metrics.*')
    meta_cols = df.find_columns('*meta.*')
    resource_cols = df.find_columns('*resource.*')
    fit_cols = df.find_columns('*fit.*')

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

    cols_of_interest = [
        'trk.poly.thresh',
        'trk.poly.metrics.bas_f1',
        'trk.poly.metrics.bas_tp',
        'trk.poly.metrics.bas_fp',
        'trk.poly.metrics.bas_fn',
        'trk.poly.metrics.bas_ppv',
        'trk.poly.metrics.bas_tpr',
        'act.poly.metrics.sc_macro_f1',
        'act.poly.resource.total_hours',
        'act.pxl.resource.total_hours',
        'act.fit.input_space_scale',
        'act.pxl.model',

        'act.fit.input_space_scale',
        'act.poly.pxl.input_space_scale',
    ]
    col_order = (ub.oset(cols_of_interest) | metric_cols) & ub.oset(df.columns)
    df2 = df[col_order]

    df['total_hours']
    df['act.total_hours'] = df['act.resource.total_hours'].sum(axis=1)
    df._clear_column_caches()
    df['act.pxl.model_name'] = df['act.pxl.package_fpath'].apply(lambda x: ub.Path(x).name)
    df['act.pxl.properties.step']

    rich.print(df2)

    import kwplot
    sns = kwplot.autosns()
    sns.scatterplot(data=df, x='act.pxl.properties.step', y='act.poly.metrics.sc_macro_f1', hue='act.pxl.model_name')
    sns.scatterplot(data=df, x='act.pxl.input_space_scale', y='act.poly.metrics.sc_macro_f1', hue='act.pxl.model_name')
    sns.scatterplot(data=df, x='act.total_hours', y='act.poly.metrics.sc_macro_f1', hue='act.pxl.model_name')

    (df['act.fit.input_space_scale'] != df['act.pxl.input_space_scale'])
    (df['act.fit.input_space_scale'] != df['act.pxl.input_space_scale'])

    sns.lineplot(data=df2, x='trk.poly.metrics.bas_fp', y='trk.poly.metrics.bas_fp')
    sns.lineplot(data=df2, x='trk.poly.metrics.bas_tpr', y='trk.poly.metrics.bas_ppv')
    sns.lineplot(data=df2, x='trk.poly.thresh', y='trk.poly.metrics.bas_f1')
    sns.lineplot(data=df2, x='trk.poly.metrics.bas_tpr', y='trk.poly.metrics.bas_ppv')
    sns.lineplot(data=df2, x='trk.poly.metrics.bas_tpr', y='trk.poly.metrics.bas_ppv')
    sns.lineplot(data=df2, x='trk.poly.metrics.bas_fp', y='trk.poly.metrics.bas_tpr')
    # hue='trk.poly.thresh')

    df.loc[:, df.columns & (list(metric_cols) + ['trk.poly.thresh'])]

    # dpath = reporter.dpath
    dpath = ub.Path.appdir('watch/expt-report/2022-10-xx').ensuredir()

    # Dump details out about the best models
    cohort = ub.timestamp()
    best_models_dpath = (dpath / 'best_models' / cohort).ensuredir()


def shrink_na_cols(df):
    import pandas as pd
    isnull = pd.isnull(df)
    df = df.loc[:, ~isnull.all(axis=0)]
    return df

"""

Test connors models:

DATASET_CODE=Aligned-Drop4-2022-08-08-TA1-S2-L8-ACC
DATA_DVC_DPATH=$(smartwatch_dvc --tags="phase2_data" --hardware="ssd")
EXPT_DVC_DPATH=$(smartwatch_dvc --tags="phase2_expt")
# TEST_DATASET=$DATA_DVC_DPATH/$DATASET_CODE/data.kwcoco.json
TEST_DATASET=$DATA_DVC_DPATH/$DATASET_CODE/BR_R001.kwcoco.json


python -m watch.mlops.schedule_evaluation \
    --params="
        matrix:
            trk.pxl.model:
                # - $EXPT_DVC_DPATH/models/fusion/Aligned-Drop4-2022-08-08-TA1-S2-L8-ACC/packages/Drop4_BAS_Multi_Native/package_epoch10_step200000.pt
                - /home/joncrall/remote/toothbrush/data/dvc-repos/smart_expt_dvc/training/toothbrush/joncrall/Aligned-Drop4-2022-08-08-TA1-S2-L8-ACC/runs/Drop4_BAS_invariants_30GSD_V016/lightning_logs/version_4/checkpoints/Drop4_BAS_invariants_30GSD_V016_epoch=10-step=5632.pt
                # - $EXPT_DVC_DPATH/models/fusion/eval3_candidates/packages/Drop3_SpotCheck_V323/Drop3_SpotCheck_V323_epoch=18-step=12976.pt
            trk.pxl.data.test_dataset:
                - $DATA_DVC_DPATH/$DATASET_CODE/combo_BR_R001_I.kwcoco.json
                - $DATA_DVC_DPATH/$DATASET_CODE/combo_KR_R001_I.kwcoco.json
                - $DATA_DVC_DPATH/$DATASET_CODE/combo_KR_R002_I.kwcoco.json
                - $DATA_DVC_DPATH/$DATASET_CODE/combo_US_R007_I.kwcoco.json
            trk.pxl.data.window_space_scale:
                - "auto"
            trk.pxl.data.time_sampling:
                - "auto"
            trk.pxl.data.input_space_scale:
                - "auto"
            trk.poly.thresh:
                - 0.1
            crop.src:
                - /home/joncrall/remote/toothbrush/data/dvc-repos/smart_data_dvc/online_v1/kwcoco_for_sc_fielded.json
            crop.regions:
                - trk.poly.output
            act.pxl.data.test_dataset:
                - /home/joncrall/remote/toothbrush/data/dvc-repos/smart_expt_dvc/models/fusion/Aligned-Drop4-2022-08-08-TA1-S2-L8-ACC/crop/online_v1_kwcoco_for_sc_fielded/trk_poly_id_0408400f/crop_f64d5b9a/crop_id_59ed6e1b/crop.kwcoco.json
            act.pxl.data.input_space_scale:
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
                - $EXPT_DVC_DPATH/models/fusion/Aligned-Drop4-2022-08-08-TA1-S2-WV-PD-ACC/packages/Drop4_SC_RGB_scratch_V002/Drop4_SC_RGB_scratch_V002_epoch=99-step=50300-v1.pt.pt
        include:
            - act.pxl.data.chip_dims: 256,256
              act.pxl.data.window_space_scale: 3GSD
              act.pxl.data.input_space_scale: 3GSD
              act.pxl.data.output_space_scale: 3GSD
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




# BAS TUNING




DATASET_CODE=Drop4-BAS
DATA_DVC_DPATH=$(smartwatch_dvc --tags="phase2_data" --hardware="auto")
EXPT_DVC_DPATH=$(smartwatch_dvc --tags="phase2_expt" --hardware="auto")
python -m watch.mlops.schedule_evaluation \
    --params="
        matrix:
            trk.pxl.model:
                # - $EXPT_DVC_DPATH/models/fusion/eval3_candidates/packages/Drop3_SpotCheck_V323/Drop3_SpotCheck_V323_epoch=18-step=12976.pt
                # - $EXPT_DVC_DPATH/models/fusion/Aligned-Drop4-2022-08-08-TA1-S2-L8-ACC/packages/Drop4_BAS_Continue_15GSD_BGR_V004/Drop4_BAS_Continue_15GSD_BGR_V004_epoch=78-step=323584.pt.pt
                - $EXPT_DVC_DPATH/models/fusion/Drop4-BAS/packages/Drop4_TuneV323_BAS_30GSD_BGRNSH_V2/Drop4_TuneV323_BAS_30GSD_BGRNSH_V2_epoch=0-step=7501.pt.pt
                - $EXPT_DVC_DPATH/models/fusion/Drop4-BAS/packages/Drop4_TuneV323_BAS_30GSD_BGRNSH_V2/Drop4_TuneV323_BAS_30GSD_BGRNSH_V2_epoch=0-step=23012.pt.pt
                - $EXPT_DVC_DPATH/models/fusion/Drop4-BAS/packages/Drop4_TuneV323_BAS_30GSD_BGRNSH_V2/Drop4_TuneV323_BAS_30GSD_BGRNSH_V2_epoch=0-step=7501-v1.pt.pt
                - $EXPT_DVC_DPATH/models/fusion/Drop4-BAS/packages/Drop4_TuneV323_BAS_30GSD_BGRNSH_V2/Drop4_TuneV323_BAS_30GSD_BGRNSH_V2_epoch=0-step=23012-v1.pt.pt
                # - $EXPT_DVC_DPATH/models/fusion/Drop4-BAS/packages/Drop4_TuneV323_BAS_30GSD_BGRNSH_V2/package_epoch0_step41.pt.pt
                # - $EXPT_DVC_DPATH/models/fusion/Drop4-BAS/packages/Drop4_TuneV323_BAS_30GSD_BGRNSH_V2/package_epoch0_step24.pt.pt
            trk.pxl.data.test_dataset:
                - $DATA_DVC_DPATH/$DATASET_CODE/KR_R001.kwcoco.json
                # - $DATA_DVC_DPATH/$DATASET_CODE/BR_R002.kwcoco.json
                # - $DATA_DVC_DPATH/$DATASET_CODE/KR_R002.kwcoco.json
                # - $DATA_DVC_DPATH/$DATASET_CODE/US_R007.kwcoco.json
            trk.pxl.data.window_space_scale:
                # - "auto"
                # - "10GSD"
                - "15GSD"
                - "30GSD"
                - "40GSD"
                - "60GSD"
            trk.pxl.data.time_steps:
                - auto
                - 2
                - 3
                - 4
                - 5
                - 6
            # trk.pxl.data.input_space_scale:
            #     # - "auto"
            #     - 10GSD
            trk.pxl.data.time_sampling:
                - "auto"
            trk.poly.thresh:
                - 0.07
                - 0.09
                - 0.1
                - 0.11
                - 0.13
            crop.src:
                - /home/joncrall/remote/toothbrush/data/dvc-repos/smart_data_dvc/online_v1/kwcoco_for_sc_fielded.json
            crop.regions:
                - trk.poly.output
            act.pxl.data.test_dataset:
                - /home/joncrall/remote/toothbrush/data/dvc-repos/smart_expt_dvc/models/fusion/Aligned-Drop4-2022-08-08-TA1-S2-L8-ACC/crop/online_v1_kwcoco_for_sc_fielded/trk_poly_id_0408400f/crop_f64d5b9a/crop_id_59ed6e1b/crop.kwcoco.json
            act.pxl.data.input_space_scale:
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
                - $EXPT_DVC_DPATH/models/fusion/Aligned-Drop4-2022-08-08-TA1-S2-WV-PD-ACC/packages/Drop4_SC_RGB_scratch_V002/Drop4_SC_RGB_scratch_V002_epoch=99-step=50300-v1.pt.pt
            include:
                - trk.pxl.data.window_space_scale: 60GSD
                  trk.pxl.data.input_space_scale: 60GSD
                  trk.pxl.data.output_space_scale: 60GSD
                - trk.pxl.data.window_space_scale: 40GSD
                  trk.pxl.data.input_space_scale: 40GSD
                  trk.pxl.data.output_space_scale: 40GSD
                - trk.pxl.data.window_space_scale: 30GSD
                  trk.pxl.data.input_space_scale: 30GSD
                  trk.pxl.data.output_space_scale: 30GSD
                - trk.pxl.data.window_space_scale: 15GSD
                  trk.pxl.data.input_space_scale: 15GSD
                  trk.pxl.data.output_space_scale: 15GSD
                # - trk.pxl.data.window_space_scale: 10GSD
                #   trk.pxl.data.input_space_scale: 10GSD
                #   trk.pxl.data.output_space_scale: 10GSD
                # - trk.pxl.data.window_space_scale: auto
                #   trk.pxl.data.input_space_scale: auto
                #   trk.pxl.data.output_space_scale: auto
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
    --devices="0,1" --queue_size=2 \
    --queue_name='bas-eval' \
    --backend=tmux --skip_existing=0 \
    --run=1



"""

# ------

"""

#### BAS CHECKS

python -m watch mlops "status" --dataset_codes Drop4-BAS
python -m watch mlops "push packages" --dataset_codes Drop4-BAS
python -m watch mlops "pull packages" --dataset_codes Drop4-BAS

"""
