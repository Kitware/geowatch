r"""

smartwatch mlops status


DATASET_CODE=Aligned-Drop4-2022-08-08-TA1-S2-L8-ACC
# DATA_DVC_DPATH=$(smartwatch_dvc --tags="phase2_data" --hardware="ssd")
DATA_DVC_DPATH=$(smartwatch_dvc --tags="phase2_data")
DVC_EXPT_DPATH=$(smartwatch_dvc --tags="phase2_expt")

VALI_DATASET_SUBSET=$DATA_DVC_DPATH/$DATASET_CODE/data_vali_KR_R001.kwcoco.json
if [ ! -f "$VALI_DATASET_SUBSET" ]; then
    VALI_DATASET_BIG=$DATA_DVC_DPATH/$DATASET_CODE/data_vali.kwcoco.json
    kwcoco subset "$VALI_DATASET_BIG" "$VALI_DATASET_SUBSET" --select_videos '.name | test("KR_R001")'
fi


VALI_DATASET_SUBSET=$DATA_DVC_DPATH/$DATASET_CODE/data_vali_10GSD_KR_R001.kwcoco.json
if [ ! -f "$VALI_DATASET_SUBSET" ]; then
    VALI_DATASET_BIG=$DATA_DVC_DPATH/$DATASET_CODE/data_vali.kwcoco.json
    kwcoco subset "$VALI_DATASET_BIG" "$VALI_DATASET_SUBSET" --select_videos '.name | test("KR_R001")'
    jq .videos[0] $VALI_DATASET_SUBSET
    smartwatch coco_add_watch_fields --src="$VALI_DATASET_SUBSET" --dst="$VALI_DATASET_SUBSET" --target_gsd=10
    jq .videos[0] $VALI_DATASET_SUBSET
fi


# Then you should be able to evaluate that model
# MODEL_OF_INTEREST="Drop4_BAS_Retrain_V002_epoch=14-step=7680"
MODEL_OF_INTEREST="Drop4_BAS_Retrain_V002_epoch=31-step=16384"
MODEL_OF_INTEREST="Drop4_BAS_Continue_15GSD_BGR_V004_epoch=78-step=323584"


echo "
Drop4_BAS_Retrain_V001_epoch=54-step=28160.pt
Drop4_BAS_Retrain_V002_epoch=14-step=7680.pt
Drop4_BAS_Retrain_V002_epoch=31-step=16384.pt
Drop4_BAS_Continue_15GSD_BGR_V004_epoch=78-step=323584*
" > models_of_interest.txt
DATASET_CODE=Aligned-Drop4-2022-08-08-TA1-S2-L8-ACC
VALI_DATASET_SUBSET=$DATA_DVC_DPATH/$DATASET_CODE/data_vali_10GSD_KR_R001.kwcoco.json
python -m watch.mlops.expt_manager "evaluate" \
    --dataset_codes "$DATASET_CODE" \
    --test_dataset="$VALI_DATASET_SUBSET" \
    --enable_track=1 \
    --enable_iarpa_eval=1 \
    --bas_thresh=0.0,0.01,0.1 \
    --skip_existing=True \
    --model_pattern="models_of_interest.txt" \
    --hack_bas_grid=True \
    --json_grid_pred_pxl='{
        "input_space_scale": ["10GSD", "15GSD"],
        "window_space_scale": ["10GSD"],
        "use_cloudmask": [0,1],
        "resample_invalid_frames": [0,1],
        "chip_overlap": [0.5],
        "set_cover_algo": ["approx", null]
    }' \
    --devices="0,1" --enable_pred=1 --run=1


# --model_pattern="${MODEL_OF_INTEREST}*" \
# --test_dataset="$TRAIN_DATASET_SUBSET" \
# TRAIN_DATASET_SUBSET=$DATA_DVC_DPATH/$DATASET_CODE/data_train_subset.kwcoco.json
# TRAIN_DATASET_BIG=$DATA_DVC_DPATH/$DATASET_CODE/data_train.kwcoco.json
# kwcoco subset "$TRAIN_DATASET_BIG" "$TRAIN_DATASET_SUBSET" --select_videos '.name | test(".*_R.*")'



# ENSURE THE MODEL WE CARE ABOUT IS EVALUTATED

MODEL_OF_INTEREST="Drop4_BAS_Continue_15GSD_BGR_V004_epoch=78-step=323584.pt"
DATASET_CODE=Aligned-Drop4-2022-08-08-TA1-S2-L8-ACC
DATA_DVC_DPATH=$(smartwatch_dvc --tags="phase2_data" --hardware="ssd")
DVC_EXPT_DPATH=$(smartwatch_dvc --tags="phase2_expt")
VALI_DATASET_SUBSET=$DATA_DVC_DPATH/$DATASET_CODE/data_train_subset.kwcoco.json
python -m watch.mlops.expt_manager "evaluate" \
    --model_pattern="${MODEL_OF_INTEREST}*" \
    --dataset_codes "$DATASET_CODE" \
    --test_dataset="$VALI_DATASET_SUBSET" \
    --enable_eval=1 \
    --enable_track=1 \
    --enable_iarpa_eval=1 \
    --bas_thresh=0.01,0.1 \
    --skip_existing=True \
    --json_grid_pred_pxl='{
        "input_space_scale": ["10GSD", "15GSD"],
        "window_space_scale": ["10GSD"],
        "use_cloudmask": [0],
        "resample_invalid_frames": [0, 1],
        "chip_overlap": [0.3, 0.0],
        "set_cover_algo": ["approx", null]
    }' \
    --devices="0,1" --enable_pred=1 --run=1 --check_other_sessions=0


# # Then you should be able to evaluate that model
# # MODEL_OF_INTEREST="Drop4_BAS_Retrain_V002_epoch=14-step=7680"
# MODEL_OF_INTEREST="Drop4_BAS_Retrain_V002_epoch=31-step=16384"
# MODEL_OF_INTEREST="Drop4_BAS_Continue_15GSD_BGR_V004_epoch=78-step=323584"
# python -m watch.mlops.expt_manager "evaluate" \
#     --dataset_codes "$DATASET_CODE" \
#     --test_dataset="$VALI_DATASET_SUBSET" \
#     --enable_track=1 \
#     --enable_iarpa_eval=1 \
#     --bas_thresh=0.00,0.01,0.1 \
#     --skip_existing=True \
#     --model_pattern="${MODEL_OF_INTEREST}*" \
#     --json_grid_pred_pxl='{
#         "input_space_scale": ["10GSD", "15GSD"],
#         "window_space_scale": ["10GSD"],
#         "use_cloudmask": [0],
#         "resample_invalid_frames": [0],
#         "chip_overlap": [0.3],
#         "set_cover_algo": ["approx", null]
#     }' \
#     --devices="0,1" --enable_pred=1 --run=1

models/fusion/Aligned-Drop4-2022-08-08-TA1-S2-L8-ACC/packages/Drop4_BAS_Continue_15GSD_BGR_V004/Drop4_BAS_Continue_15GSD_BGR_V004_epoch=78-step=323584.pt.pt
"""


def main():
    """
    import sys, ubelt
    sys.path.append(ubelt.expandpath('~/code/watch/dev/reports'))
    from report_2022_09_xx import *  # NOQA
    """
    from watch import heuristics
    from watch.mlops import expt_manager
    from watch.mlops import expt_report
    expt_dvc_dpath = heuristics.auto_expt_dvc()
    data_dvc_dpath = heuristics.auto_expt_dvc()

    # I messed up the name of the dataset I was working on.
    # it is marked as train, but it should have been vali.

    dataset_code = 'Aligned-Drop4-2022-08-08-TA1-S2-L8-ACC'

    state = expt_manager.ExperimentState(
        expt_dvc_dpath, dataset_code=dataset_code,
        data_dvc_dpath=data_dvc_dpath, model_pattern='*')
    self = state  # NOQA

    # state.patterns['test_dset'] = 'Aligned-Drop4-2022-08-08-TA1-S2-L8-ACC_data_train_subset.kwcoco'
    state.patterns['test_dset'] = (
        'Aligned-Drop4-2022-08-08-TA1-S2-L8-ACC_data_vali_10GSD_KR_R001.kwcoco')
    state._build_path_patterns()
    state._make_cross_links()
    # state._block_non_existing_rhashes()

    reporter = expt_report.EvaluationReporter(state)
    reporter.load()

    reporter.state.summarize()

    test_dset_to_shortlist = reporter.report_best(show_configs=True)
    for test_dset, shortlist in test_dset_to_shortlist.items():
        for _, row in best_rows[::-1].iterrows():
            break

    reporter = reporter
    orig_merged_df = reporter.orig_merged_df
    # predcfg_to_label = reporter.predcfg_to_label
    # actcfg_to_label = reporter.actcfg_to_label
    human_mapping = reporter.human_mapping

    merged_df = orig_merged_df.copy()

    common_plotkw = {
        # 'connect': 'expt',
        # 'mesh': 'model',
        # 'clique': 'model',
        # 'style': 'has_teamfeat',
        'star': 'in_production',
        'starkw': {'s': 500},
        's': 120,
    }

    import kwplot
    kwplot.autosns()

    from watch.mlops import plots

    # dpath = reporter.dpath
    import ubelt as ub
    dpath = ub.Path.appdir('watch/expt-report/2022-09-xx').ensuredir()

    plotter = plots.Plotter.from_reporter(
        reporter, common_plotkw=common_plotkw, dpath=dpath)

    # Takes a long time to load these
    # plots.dataset_summary_tables(dpath)

    plots.initial_summary(reporter, dpath)

    plots.describe_varied(merged_df, dpath, human_mapping=human_mapping)

    tracked_metrics = ['salient_AP', 'BAS_F1']
    for metrics in tracked_metrics:
        plotter.plot_groups(
            'metric_over_training_time', metrics=metrics, huevar='expt')

    plotter.plot_groups('plot_pixel_ap_verus_iarpa', huevar='expt')

    plotter.plot_groups('plot_pixel_ap_verus_auc', huevar='expt')

    plotter.plot_groups('plot_resource_versus_metric', huevar='expt')

    import xdev
    xdev.view_directory(dpath)
    model = shortlist_models()[0]


def single_model_analysis(reporter, model):
    import ubelt as ub
    orig_merged_df = reporter.orig_merged_df
    df = orig_merged_df
    df = df[df['model'].str.startswith(model)]

    metric_names = reporter.metric_registry.name
    # cfg_names = ['pred_cfg', 'act_cfg', 'trk_cfg']
    id_names = ['model']  # + cfg_names
    # id_names = ['test_dset', 'model']
    cols = id_names + list(metric_names)
    cols = ub.oset(cols) & df.columns

    # state = reporter.state
    # param_names = ['fit_params', 'pred_params', 'track_params']
    # known_hashes = build_hash_lut(state)

    type_to_all_params = ub.udict()
    type_to_all_params['fit'] = df['fit_params'].tolist()
    type_to_all_params['pred_pxl'] = df['pred_params'].tolist()
    type_to_all_params['pred_trk'] = df['track_params'].tolist()
    type_to_varied_params = type_to_all_params.map_values(lambda vs: ub.varied_values(vs, default='?'))
    type_to_varied_params['pred_pxl'] = ub.udict(type_to_varied_params['pred_pxl']) - {
        'pred_model_fpath', 'pred_in_dataset_fpath', 'pred_model_name', 'expt_name'}

    varied_pred_params = {k for k, v in type_to_varied_params['pred_pxl'].items() if len(v) > 1}
    varied_trk_params = {k for k, v in type_to_varied_params['pred_trk'].items() if len(v) > 1}
    print('type_to_varied_params = {}'.format(ub.repr2(type_to_varied_params, nl=2)))

    for type, varied_params in type_to_varied_params.items():
        varied_params

    new_rows = []
    for row in df.to_dict('records'):
        row = ub.udict(row)
        subrow = row.subdict(cols)
        pred_params = ub.udict(row.get('pred_params', {})).subdict(varied_pred_params, default='?')
        track_params = ub.udict(row.get('track_params', {})).subdict(varied_trk_params, default='?')
        subrow.update(pred_params)
        subrow.update(track_params)
        new_rows.append(subrow)
    import pandas as pd
    new_df = pd.DataFrame(new_rows)
    print(new_df)


def shortlist_models():
    return [
        'Drop4_BAS_Continue_15GSD_BGR_V004_epoch=78-step=323584',
        'Drop4_BAS_Retrain_V001_epoch=54-step=28160.pt',
        'Drop4_BAS_Retrain_V002_epoch=14-step=7680.pt',
        'Drop4_BAS_Retrain_V002_epoch=31-step=16384.pt',
    ]


def custom_investigate(reporter):
    metric = 'BAS_F1'
    expt_name = 'Drop4_BAS_Retrain_V002'

    reporter = reporter
    orig_merged_df = reporter.orig_merged_df
    df = orig_merged_df

    mfound = df[df['expt'].str.match(expt_name)]
    mfound = mfound.sort_values(metric, ascending=False)
    import ubelt as ub
    for type, group in mfound.groupby('type'):
        print(f'type={type}')
        print(ub.repr2(group.to_dict('records')[0]))

    mfound[['model', 'BAS_F1', 'pred_cfg', 'trk_cfg']]

    table = reporter.state.evaluation_table()
    found = table[table['expt'] == expt_name].reset_index()


def make_cross_links():
    pass

    # from pandasql import sqldf
    # import ubelt as ub
    # results = sqldf(ub.codeblock(
    #     '''
    #     SELECT * FROM d
    #     WHERE d.expt == "Drop4_BAS_Retrain_V002"
    #     '''), {'d': orig_merged_df})


def build_hash_lut(state):
    import ubelt as ub
    known_hashes = ub.ddict(list)
    missing = []
    state._build_path_patterns()
    orig_eval_table = state.evaluation_table()
    for cfgkey in state.hashed_cfgkeys:
        if cfgkey in orig_eval_table:
            unique_keys = orig_eval_table[cfgkey].dropna().unique()
            for key in unique_keys:
                from watch.utils.reverse_hashid import ReverseHashTable
                candidates = ReverseHashTable.query(key, verbose=0)
                if candidates:
                    known_hashes[cfgkey].append(candidates)
                else:
                    missing.append(cfgkey)
    return known_hashes


if __name__ == '__main__':
    """
    CommandLine:
        python ~/code/watch/dev/reports/report_2022_09_xx.py


        models/fusion/Aligned-Drop4-2022-08-08-TA1-S2-WV-PD-ACC/eval/Drop4_SC_RGB_from_sc006_V003_cont/Drop4_SC_RGB_from_sc006_V003_cont_epoch=51-step=34892.pt/Aligned-Drop4-2022-08-08-TA1-S2-WV-PD-ACC_data_vali.kwcoco/predcfg_e1c3673c/eval/actclf/actcfg_6d2b0de0/iarpa_sc_eval/scores/merged/summary3.json.dvc

        is git ignored
    """
    main()
