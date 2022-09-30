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
    --enable_pred_trk=1 \
    --enable_eval_trk=1 \
    --bas_thresh=0.05,0.01 \
    --skip_existing=True \
    --model_pattern="models_of_interest.txt" \
    --json_grid_pred_trk=False \
    --json_grid_pred_pxl='{
        "input_space_scale": ["10GSD", "15GSD"],
        "window_space_scale": ["auto"],
        "use_cloudmask": [0,1],
        "resample_invalid_frames": [0,1],
        "chip_overlap": [0.3],
        "set_cover_algo": ["approx", null],
        "time_sampling": ["auto"],
        "time_span": ["auto"]
    }' \
    --devices="0,1" --enable_pred_pxl=1 --run=1
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

    state.patterns['test_dset'] = (
        'Aligned-Drop4-2022-08-08-TA1-S2-L8-ACC_data_septest.kwcoco')

    state._build_path_patterns()
    state._make_cross_links()
    # state._block_non_existing_rhashes()

    reporter = expt_report.EvaluationReporter(state)
    reporter.load()
    reporter.state.summarize()

    # dpath = reporter.dpath
    import ubelt as ub
    import pandas as pd
    dpath = ub.Path.appdir('watch/expt-report/2022-09-xx').ensuredir()

    # Dump details out about the best models
    cohort = ub.timestamp()
    best_models_dpath = (dpath / 'best_models' / cohort).ensuredir()
    groupid_to_shortlist = reporter.report_best(show_configs=True, verbose=1, top_k=4)

    rlut = reporter._build_cfg_rlut(None)

    # Ideally we would have more consistent name conventions, but just
    # register things for now
    cfg_param_relationship = [
        {'cfg_key': 'trk_cfg', 'param_key': 'track_params', 'type': 'trk'},
        {'cfg_key': 'pred_cfg', 'param_key': 'pred_params', 'type': 'pxl'},
        # TODO: act
    ]

    for groupid, shortlist in groupid_to_shortlist.items():
        test_dset, type = groupid
        _dpath = (best_models_dpath / f'{test_dset}_{type}').ensuredir()

        for rank, row in enumerate(shortlist[::-1].to_dict('records'), start=1):
            parts = [p for p in [
                f'rank_{rank:03d}',
                row.get('model', None),
                # row.get('test_dset', None)
                row.get('pred_cfg', None),
                row.get('trk_cfg', None),
                row.get('act_cfg', None),
            ] if not pd.isnull(p)]
            dname = '_'.join(parts)
            row_dpath = (_dpath / dname).ensuredir()

            linkable_keys = [
                f'{a}_{b}'
                for a in ['eval', 'pred']
                for b in ['pxl', 'trk', 'act']
            ]
            linkables = {}
            for key in linkable_keys:
                try:
                    linkables[key] = ub.Path(state.templates[key].format(**row))
                except KeyError:
                    ...

            for key, real_fpath in linkables.items():
                if real_fpath.exists():
                    link_dpath = row_dpath / f'_link_{key}'
                    # hack:
                    if key == 'eval_pxl':
                        real_dpath = real_fpath.parent.parent
                    else:
                        real_dpath = real_fpath.parent
                    ub.symlink(real_dpath, link_dpath, verbose=0, overwrite=1)

            # Check if the registered params agree with the rlut params
            for relate in cfg_param_relationship:
                cfg_key = relate['cfg_key']
                param_key = relate['param_key']
                if param_key in row and cfg_key in row:
                    cfg_hashid = row[cfg_key]
                    cfg_params = row[param_key]
                    if cfg_hashid in rlut:
                        resolved = rlut[cfg_hashid]

            metric_names = reporter.metric_registry.name
            metric_cols = (ub.oset(metric_names) & row.keys())
            primary_metrics = (ub.oset(['mean_f1', 'BAS_F1']) & row.keys())
            metric_cols = list((metric_cols & primary_metrics) | (metric_cols - primary_metrics))

            from kwcoco.util.util_json import ensure_json_serializable
            row = ensure_json_serializable(row)
            walker = ub.IndexableWalker(row)
            for path, val in walker:
                if hasattr(val, 'spec'):
                    walker[path] = val.spec

            import json
            details = json.dumps(row, indent='   ')
            deatils_fpath = row_dpath / 'details.json'
            deatils_fpath.write_text(details)

            # hack to get back to regular names
            pred_params = ub.udict({k[5:] : v for k, v in row['pred_params'].items() if k.startswith('pred_')})
            pred_params = pred_params - {'in_dataset_name', 'model_name', 'in_dataset_fpath', 'model_fpath'}

            # hack to get back to regular names
            track_params = ub.udict({
                k[4:] : v for k, v in row['track_params'].items() if k.startswith('trk_')})
            track_params = track_params - {
                'in_dataset_name', 'model_name', 'in_dataset_fpath', 'model_fpath'}

            row_ = row.copy()
            row['rank'] = (rank, cohort)
            row_['expt_dvc_dpath'] = '.'
            pkg_fpath = state.templates['pkg'].format(**row_)

            metrics = ub.udict(row) & metric_cols
            metrics['test_dset'] = test_dset
            summary = {
                'rank': (rank, cohort),
                'model': row['model'],
                'file_name': pkg_fpath,
                'pred_params': pred_params,
                'track_params': track_params,
                'metrics': metrics,
            }
            summary_fpath = row_dpath / 'summary.json'
            summary_fpath.write_text(json.dumps(summary, indent='   '))
            print(f'summary_fpath={summary_fpath}')

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

    plotter.plot_groups('plot_violinplots', metrics='BAS_F1')

    import xdev
    xdev.view_directory(dpath)
    # model = shortlist_models()[0]


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

"""


2022-09-28 EVAL RUN


DATASET_CODE=Aligned-Drop4-2022-08-08-TA1-S2-L8-ACC
DATA_DVC_DPATH=$(smartwatch_dvc --tags="phase2_data")
DVC_EXPT_DPATH=$(smartwatch_dvc --tags="phase2_expt")

TEST_DATASET=$DATA_DVC_DPATH/$DATASET_CODE/data_septest.kwcoco.json
if [ ! -f "$TEST_DATASET" ]; then
    DATASET_BIG=$DATA_DVC_DPATH/$DATASET_CODE/data.kwcoco.json
    kwcoco subset "$DATASET_BIG" "$TEST_DATASET" \
        --select_videos '((.name | test("KR_R001")) or (.name | test("KR_R002")) or (.name | test("BR_R002")) or (.name | test("US_R007")))'
fi

# Drop4_BAS_Retrain_V001_epoch=54-step=28160.pt
# Drop4_BAS_Continue_15GSD_BGR_V004_epoch=78-step=323584*

echo "
Drop4_BAS_Retrain_V002_epoch=31-step=16384.pt
" > models_of_interest.txt
DATASET_CODE=Aligned-Drop4-2022-08-08-TA1-S2-L8-ACC
python -m watch.mlops.expt_manager "evaluate" \
    --dataset_codes "$DATASET_CODE" \
    --test_dataset="$TEST_DATASET" \
    --model_pattern="models_of_interest.txt" \
    --json_grid_pred_trk=auto \
    --json_grid_pred_pxl='{
        "input_space_scale": ["10GSD", "15GSD"],
        "window_space_scale": ["10GSD"],
        "use_cloudmask": [0,1],
        "resample_invalid_frames": [0,1],
        "chip_overlap": [0.3],
        "set_cover_algo": ["approx", null],
        "time_sampling": ["auto", "contiguous"],
        "time_span": ["auto"]
    }' \
    --devices="0,1" --queue_size=2 \
    --enable_pred_pxl=1 --enable_eval_pxl=1 \
    --enable_pred_trk=1 --enable_eval_trk=1 --enable_pred_trk_viz=0  \
    --skip_existing=1 \
    --run=1


echo "
Drop4_BAS_Retrain_V001_epoch=54-step=28160.pt
Drop4_BAS_Retrain_V002_epoch=31-step=16384.pt
Drop4_BAS_Continue_15GSD_BGR_V004_epoch=78-step=323584*
" > models_of_interest.txt
DATASET_CODE=Aligned-Drop4-2022-08-08-TA1-S2-L8-ACC
python -m watch.mlops.expt_manager "evaluate" \
    --dataset_codes "$DATASET_CODE" \
    --test_dataset="$TEST_DATASET" \
    --model_pattern="models_of_interest.txt" \
    --json_grid_pred_trk=auto \
    --json_grid_pred_pxl='{
        "input_space_scale": ["10GSD", "15GSD"],
        "window_space_scale": ["10GSD"],
        "use_cloudmask": [0,1],
        "resample_invalid_frames": [0,1],
        "chip_overlap": [0.3],
        "set_cover_algo": ["approx", null],
        "time_sampling": ["auto", "contiguous"],
        "time_span": ["auto"]
    }' \
    --devices="0,1" --queue_size=8 \
    --enable_pred_pxl=0 --enable_eval_pxl=1 \
    --enable_pred_trk=1 --enable_eval_trk=1 --enable_pred_trk_viz=0  \
    --skip_existing=1 \
    --run=1



echo "
Drop4_BAS_Retrain_V002_epoch=31-step=16384.pt
" > models_of_interest.txt
DATASET_CODE=Aligned-Drop4-2022-08-08-TA1-S2-L8-ACC
python -m watch.mlops.expt_manager "evaluate" \
    --dataset_codes "$DATASET_CODE" \
    --test_dataset="$TEST_DATASET" \
    --model_pattern="models_of_interest.txt" \
    --json_grid_pred_trk=auto \
    --json_grid_pred_pxl='{
        "input_space_scale": ["10GSD"],
        "window_space_scale": ["10GSD"],
        "use_cloudmask": [0],
        "resample_invalid_frames": [0],
        "chip_overlap": [0.3],
        "set_cover_algo": ["approx"],
        "time_sampling": ["auto"],
        "time_span": ["auto"]
    }' \
    --devices="0,1" --queue_size=8 \
    --enable_pred_pxl=0 --enable_eval_pxl=0 \
    --enable_pred_trk=0 --enable_eval_trk=1 --enable_pred_trk_viz=0  \
    --skip_existing=1 \
    --run=0
"""
