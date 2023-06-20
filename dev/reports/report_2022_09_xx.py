r"""

smartwatch mlops status


DATASET_CODE=Aligned-Drop4-2022-08-08-TA1-S2-L8-ACC
# DATA_DVC_DPATH=$(geowatch_dvc --tags="phase2_data" --hardware="ssd")
DATA_DVC_DPATH=$(geowatch_dvc --tags="phase2_data")
DVC_EXPT_DPATH=$(geowatch_dvc --tags="phase2_expt")

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
    --grid_pred_trk=False \
    --grid_pred_pxl='{
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
    import ubelt as ub
    import pandas as pd
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
    state.patterns['test_dset'] = (
        'Aligned-Drop4-2022-08-08-TA1-S2-L8-ACC_data_vali_10GSD_KR_R001.kwcoco')

    state.patterns['test_dset'] = (
        'Aligned-Drop4-2022-08-08-TA1-S2-L8-ACC_data_septest.kwcoco')

    state.patterns['test_dset'] = ('Aligned-Drop4-2022-08-08-TA1-S2-L8-ACC_data_kr1br2.kwcoco')

    state.patterns['test_dset'] = ('*')

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

    colnames = ub.oset(reporter.orig_merged_df.columns)
    # column_nestings = util_dotdict.dotkeys_to_nested(colnames)
    dotted = ub.oset([c for c in colnames if '.' in c])
    metric_cols = ub.oset([c for c in dotted if 'metrics.' in c])
    meta_cols = ub.oset([c for c in dotted if 'meta.' in c])
    resource_cols = ub.oset([c for c in dotted if 'resource.' in c])
    fit_cols = ub.oset([c for c in dotted if 'fit.' in c])
    param_cols = dotted - (metric_cols | fit_cols | resource_cols | meta_cols)

    if 0:
        idx = 368
        df.loc[idx]

    dset_blocklist = [
        'KR_R001_0.1BASThresh_40cloudcover_debug10_kwcoco_cropped_kwcoco_for_bas',
    ]
    type_blocklist = [
        'eval_trk_poly_fpath'
    ]

    import rich as rich_mod
    from kwutil import util_yaml
    for groupid, shortlist in groupid_to_shortlist.items():
        test_dset, type = groupid
        if test_dset in dset_blocklist:
            continue
        if type in type_blocklist:
            continue
        _dpath = (best_models_dpath / f'{test_dset}_{type}').ensuredir()

        for rank, row in reversed(list(enumerate(shortlist[::-1].to_dict('records'), start=1))):
            row = ub.udict(row)

            parts = [p for p in [
                f'rank_{rank:03d}',
                row.get('trk_model', None),
                row.get('act_model', None),
                # row.get('test_dset', None)
                row.get('trk_pxl_cfg', None),
                row.get('trk_poly_cfg', None),
                row.get('act_pxl_cfg', None),
                row.get('act_poly_cfg', None),
            ] if not pd.isnull(p)]
            dname = '_'.join(parts)
            row_dpath = (_dpath / dname).ensuredir()

            # metric_names = reporter.metric_registry.name
            # metric_cols = (ub.oset(metric_names) & row.keys())
            # primary_metrics = (ub.oset(['sc_macro_f1', 'BAS_F1']) & row.keys())
            metric_cols = [c for c in row.keys() if 'metrics.' in c]
            # metric_cols = list((metric_cols & primary_metrics) | (metric_cols - primary_metrics))

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
            # pred_params = ub.udict({k[5:] : v for k, v in row['pred_params'].items() if k.startswith('pred_')})
            # pred_params = pred_params - {'in_dataset_name', 'model_name', 'in_dataset_fpath', 'model_fpath'}

            # # hack to get back to regular names
            # track_params = ub.udict({
            #     k[4:] : v for k, v in row['track_params'].items() if k.startswith('trk_')})
            # track_params = track_params - {
            #     'in_dataset_name', 'model_name', 'in_dataset_fpath', 'model_fpath'}

            # fit_params = ub.udict({
            #     k : v for k, v in row['fit_params'].items()})
            # fit_params = fit_params - {
            #     'in_dataset_name', 'model_name', 'in_dataset_fpath', 'model_fpath'}

            row = row.copy()
            row['rank'] = (rank, cohort)
            # row_['expt_dvc_dpath'] = '.'
            # pkg_fpath = state.templates['pkg_trk_pxl_fpath'].format(**row_)
            # act_fpath = state.templates['pkg_act_pxl_fpath'].format(**row_)

            if 1:
                pred_act_poly_kwcoco = ub.Path(state.templates['pred_act_poly_kwcoco'].format(**row))
                # name = row.get('model', '') + row.get('pred_cfg', '') + row.get('trk_cfg', '')
                name = '<todo: more context>'
                # TODO: allow specification of truth fpath as well?
                viz_track_cmd = ub.codeblock(
                    fr'''
                    smartwatch visualize \
                        "{pred_act_poly_kwcoco}" \
                        --channels="red|green|blue,No Activity|Site Preparation|Active Construction|Post Construction" \
                        --viz_dpath={row_dpath}/_viz \
                        --stack=only \
                        --skip_missing=False \
                        --draw_imgs=True \
                        --draw_anns=True \
                        --workers=avail/2 \
                        --animate=True \
                        --draw_valid_region=False \
                        --extra_header="\nRank#{rank} {cohort}\n{name}"
                    ''')
                print(viz_track_cmd)
                viz_cmds.append(viz_track_cmd)

            # metrics['test_dset'] = test_dset
            summary = {
                'rank': (rank, cohort),
                # 'model': row['model'],
                # 'file_name': pkg_fpath,
                'pred_params': row & param_cols,
                'fit_params': row & fit_cols,
                'resources': row & resource_cols,
                'meta': row & meta_cols,
                'metrics': row & metric_cols,
            }
            summary_fpath = row_dpath / 'summary.yaml'
            yaml_text = util_yaml.Yaml.dumps(summary)
            summary_fpath.write_text(yaml_text)
            # summary_fpath.write_text(json.dumps(summary, indent='   '))
            rich_mod.print('summary = {}'.format(ub.urepr(summary, nl=2)))
            print(f'summary_fpath={summary_fpath}')

    for viz_cmd in viz_cmds:
        print(viz_cmd)
        print('')
        print('')

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
    # cls = plots.Plotter
    plotter = plots.Plotter.from_reporter(
        reporter, common_plotkw=common_plotkw, dpath=dpath)
    analysis = plotter.analysis = reporter.build_analysis()

    params_of_interest = [s['param_name'] for s in analysis.statistics][::-1]
    print('params_of_interest = {}'.format(ub.urepr(params_of_interest, nl=1)))
    plotter.params_of_interest = params_of_interest

    # Takes a long time to load these
    # plots.dataset_summary_tables(dpath)

    # plots.initial_summary(reporter, dpath)

    # plots.describe_varied(merged_df, dpath, human_mapping=human_mapping)

    for code_type, group in plotter.expt_groups.items():
        pass

    # tracked_metrics = ['salient_AP', 'BAS_F1']
    tracked_metrics = [
        # 'trk.poly.metrics.bas_f1',
        'act.poly.metrics.bas_f1',
        # 'trk.pxl.metrics.salient_AP',
        # 'act.poly.metrics.macro_f1',
    ]
    for param in params_of_interest:
        ...
        # plot_name = 'plot_pixel_ap_verus_iarpa'
        plot_name = 'plot_relationship'
        try:
            plotter.plot_groups(plot_name, huevar=param)
        except Exception as ex:
            print(f'ex={ex}')

    # plotter.plot_groups(
    #     plot_name='metric_over_training_time', metrics=metrics, huevar='expt')

    # plotter.plot_groups('plot_pixel_ap_verus_auc', huevar='expt')

    # plotter.plot_groups(
    #     plot_name='plot_resource_versus_metric',
    #     huevar='expt')

    # params_of_interest = list(analysis.varied)

    # params_of_interest = [
    #     # 'pred_use_cloudmask',
    #     # 'pred_resample_invalid_frames',
    #     # 'pred_input_space_scale',
    #     # 'pred_window_space_scale',
    #     'trk_thresh',
    # ]

    plotter.plot_groups('plot_param_analysis', metrics='act.poly.metrics.macro_f1',
                        params_of_interest=params_of_interest)

    plotter.plot_groups('plot_param_analysis', metrics='act.poly.metrics.bas_f1',
                        params_of_interest=params_of_interest)

    plotter.plot_groups('plot_param_analysis', metrics='total_hours',
                        params_of_interest=params_of_interest)

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
    print('type_to_varied_params = {}'.format(ub.urepr(type_to_varied_params, nl=2)))

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
        print(ub.urepr(group.to_dict('records')[0]))

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
DATA_DVC_DPATH=$(geowatch_dvc --tags="phase2_data")
DVC_EXPT_DPATH=$(geowatch_dvc --tags="phase2_expt")

TEST_DATASET=$DATA_DVC_DPATH/$DATASET_CODE/data_septest.kwcoco.json
if [ ! -f "$TEST_DATASET" ]; then
    DATASET_BIG=$DATA_DVC_DPATH/$DATASET_CODE/data.kwcoco.json
    kwcoco subset "$DATASET_BIG" "$TEST_DATASET" \
        --select_videos '((.name | test("KR_R001")) or (.name | test("KR_R002")) or (.name | test("BR_R002")) or (.name | test("US_R007")))'
fi

TEST_DATASET=$DATA_DVC_DPATH/$DATASET_CODE/data_kr1br2.kwcoco.json
if [ ! -f "$TEST_DATASET" ]; then
    DATASET_BIG=$DATA_DVC_DPATH/$DATASET_CODE/data.kwcoco.json
    kwcoco subset "$DATASET_BIG" "$TEST_DATASET" \
        --select_videos '((.name | test("KR_R001")) or (.name | test("BR_R002")))'
fi


echo "
Drop4_BAS_Retrain_V001_epoch=54-step=28160.pt
Drop4_BAS_Continue_15GSD_BGR_V004_epoch=78-step=323584*
Drop4_BAS_Retrain_V002_epoch=31-step=16384.pt
" > models_of_interest.txt
DATASET_CODE=Aligned-Drop4-2022-08-08-TA1-S2-L8-ACC
python -m watch.mlops.expt_manager "evaluate" \
    --dataset_codes "$DATASET_CODE" \
    --test_dataset="$TEST_DATASET" \
    --model_pattern="models_of_interest.txt" \
    --grid_pred_trk=auto \
    --grid_pred_pxl='{
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
    --enable_pred_trk=1 --enable_eval_trk=1 \
    --skip_existing=1 --run=1


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
    --grid_pred_trk=auto \
    --grid_pred_pxl='{
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
    --grid_pred_trk=auto \
    --grid_pred_pxl='{
        matrix:
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
    --run=1
"""


"""

NEWEST

2022-09-28 EVAL RUN

AWS_DEFAULT_PROFILE=iarpa GDAL_DISABLE_READDIR_ON_OPEN=EMPTY_DIR smartwatch add_fields kwcoco_for_sc.json kwcoco_for_sc_fielded.json \
    --target_gsd=4 \
    --enable_video_stats=True \
    --enable_valid_region=True \
    --workers=auto


DATASET_CODE=Aligned-Drop4-2022-08-08-TA1-S2-L8-ACC
DATA_DVC_DPATH=$(geowatch_dvc --tags="phase2_data")
DVC_EXPT_DPATH=$(geowatch_dvc --tags="phase2_expt")

TEST_DATASET=$DATA_DVC_DPATH/$DATASET_CODE/data_kr1br2.kwcoco.json
if [ ! -f "$TEST_DATASET" ]; then
    DATASET_BIG=$DATA_DVC_DPATH/$DATASET_CODE/data.kwcoco.json
    kwcoco subset "$DATASET_BIG" "$TEST_DATASET" \
        --select_videos '((.name | test("KR_R001")) or (.name | test("BR_R002")))'
fi


python -m watch.mlops.schedule_evaluation \
    --params="
        matrix:
            trk.pxl.model:
                - $DVC_EXPT_DPATH/models/fusion/Aligned-Drop4-2022-08-08-TA1-S2-L8-ACC/packages/Drop4_BAS_Retrain_V002/Drop4_BAS_Retrain_V002_epoch=31-step=16384.pt.pt
            trk.pxl.data.test_dataset:
                - $TEST_DATASET
            trk.pxl.data.window_scale_space: 15GSD
            trk.pxl.data.time_sampling:
                - "auto"
                - "contiguous"
            trk.pxl.data.input_scale_space:
                - "15GSD"
                - "10GSD"
            crop.src:
                # FIXME: should be cropping from a dataset with WV
                - /home/joncrall/remote/toothbrush/data/dvc-repos/smart_data_dvc/online_v1/kwcoco_for_sc_fielded.json
                # - $TEST_DATASET
            crop.regions:
                - trk.poly.output
            act.pxl.data.test_dataset:
                - crop.dst
            act.pxl.data.window_scale_space:
                - 15GSD
                - auto
            trk.poly.thresh:
                - 0.003
                - 0.007
                - 0.01
                - 0.03
                - 0.05
                - 0.07
                - 0.1
            act.poly.thresh:
                - 0.01
                - 0.05
                - 0.1
                - 0.2
            act.poly.use_viterbi:
                - 0
                - 1
            act.pxl.model:
                - $DVC_EXPT_DPATH/models/fusion/Aligned-Drop4-2022-08-08-TA1-S2-WV-PD-ACC/packages/Drop4_SC_RGB_scratch_V002/Drop4_SC_RGB_scratch_V002_epoch=99-step=50300-v1.pt.pt
                - $DVC_EXPT_DPATH/models/fusion/Aligned-Drop4-2022-08-08-TA1-S2-WV-PD-ACC/packages/Drop4_SC_RGB_scratch_V002/Drop4_SC_RGB_scratch_V002_epoch=155-step=78468.pt.pt
    " \
    --enable_pred_trk_pxl=0 \
    --enable_pred_trk_poly=0 \
    --enable_eval_trk_pxl=0 \
    --enable_eval_trk_poly=0 \
    --enable_crop=0 \
    --enable_pred_act_pxl=0 \
    --enable_pred_act_poly=1 \
    --enable_eval_act_pxl=1 \
    --enable_eval_act_poly=1 \
    --enable_viz_pred_trk_poly=0 \
    --enable_viz_pred_act_poly=0 \
    --devices="0,1" --queue_size=2 \
    --backend=tmux --skip_existing=1 \
    --run=1
"""












"""
ALT:



AWS_DEFAULT_PROFILE=iarpa GDAL_DISABLE_READDIR_ON_OPEN=EMPTY_DIR smartwatch add_fields kwcoco_for_sc.json kwcoco_for_sc_fielded.json \
    --target_gsd=4 \
    --enable_video_stats=True \
    --enable_valid_region=True \
    --workers=auto


DATASET_CODE=Aligned-Drop4-2022-08-08-TA1-S2-L8-ACC
DATA_DVC_DPATH=$(geowatch_dvc --tags="phase2_data")
DVC_EXPT_DPATH=$(geowatch_dvc --tags="phase2_expt")

TEST_DATASET=$DATA_DVC_DPATH/$DATASET_CODE/data_kr1br2.kwcoco.json
if [ ! -f "$TEST_DATASET" ]; then
    DATASET_BIG=$DATA_DVC_DPATH/$DATASET_CODE/data.kwcoco.json
    kwcoco subset "$DATASET_BIG" "$TEST_DATASET" \
        --select_videos '((.name | test("KR_R001")) or (.name | test("BR_R002")))'
fi


python -m watch.mlops.schedule_evaluation \
    --params="
        matrix:
            trk.pxl.model:
                - $DVC_EXPT_DPATH/models/fusion/Aligned-Drop4-2022-08-08-TA1-S2-L8-ACC/packages/Drop4_BAS_Retrain_V002/Drop4_BAS_Retrain_V002_epoch=31-step=16384.pt.pt
            trk.pxl.data.test_dataset:
                - $TEST_DATASET
            trk.pxl.data.window_scale_space: 15GSD
            trk.pxl.data.time_sampling:
                - "auto"
                - "contiguous"
            trk.pxl.data.input_scale_space:
                - "15GSD"
                - "10GSD"
            trk.poly.thresh:
                - 0.003
                - 0.007
                - 0.01
                - 0.03
                - 0.05
                - 0.07
                - 0.1
            crop.src:
                # FIXME: should be cropping from a dataset with WV
                - /home/joncrall/remote/toothbrush/data/dvc-repos/smart_data_dvc/online_v1/kwcoco_for_sc_fielded.json
                # - $TEST_DATASET
            crop.regions:
                - trk.poly.output
            act.pxl.data.test_dataset:
                - crop.dst
            act.pxl.data.input_scale_space:
                - 10GSD
                - 5GSD
            act.pxl.data.time_steps:
                - 3
                - 5
            act.poly.thresh:
                - 0.003
                - 0.007
                - 0.01
                - 0.05
                - 0.1
                - 0.2
            act.poly.use_viterbi:
                - 0
                - 1
            act.pxl.model:
                - $DVC_EXPT_DPATH/models/fusion/Aligned-Drop4-2022-08-08-TA1-S2-WV-PD-ACC/packages/Drop4_SC_RGB_scratch_V002/Drop4_SC_RGB_scratch_V002_epoch=99-step=50300-v1.pt.pt
                - $DVC_EXPT_DPATH/models/fusion/Aligned-Drop4-2022-08-08-TA1-S2-WV-PD-ACC/packages/Drop4_SC_RGB_scratch_V002/Drop4_SC_RGB_scratch_V002_epoch=155-step=78468.pt.pt
        include:
            - act.pxl.data.chip_dims: 480,480
              act.pxl.data.window_scale_space: 5GSD
              act.pxl.data.input_scale_space: 5GSD
              act.pxl.data.output_scale_space: 5GSD
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
    --devices="0,1" --queue_size=2 \
    --queue_name='secondary-eval' \
    --backend=tmux --skip_existing=1 \
    --run=1


DATASET_CODE=Aligned-Drop4-2022-08-08-TA1-S2-L8-ACC
DATA_DVC_DPATH=$(geowatch_dvc --tags="phase2_data")
DVC_EXPT_DPATH=$(geowatch_dvc --tags="phase2_expt")
TEST_DATASET=$DATA_DVC_DPATH/$DATASET_CODE/data_kr1br2.kwcoco.json
python -m watch.mlops.schedule_evaluation \
    --params="
        matrix:
            trk.pxl.model:
                - $DVC_EXPT_DPATH/models/fusion/Aligned-Drop4-2022-08-08-TA1-S2-L8-ACC/packages/Drop4_BAS_Retrain_V002/Drop4_BAS_Retrain_V002_epoch=31-step=16384.pt.pt
            trk.pxl.data.test_dataset:
                - $TEST_DATASET
            trk.pxl.data.window_scale_space: 15GSD
            trk.pxl.data.time_sampling:
                - "contiguous"
            trk.pxl.data.input_scale_space:
                - "15GSD"
            trk.poly.thresh:
                - 0.07
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
    --enable_pred_trk_pxl=0 \
    --enable_pred_trk_poly=0 \
    --enable_eval_trk_pxl=0 \
    --enable_eval_trk_poly=0 \
    --enable_crop=1 \
    --enable_pred_act_pxl=1 \
    --enable_pred_act_poly=1 \
    --enable_eval_act_pxl=0 \
    --enable_eval_act_poly=1 \
    --enable_viz_pred_trk_poly=0 \
    --enable_viz_pred_act_poly=1 \
    --enable_links=0 \
    --devices="0,1" --queue_size=2 \
    --queue_name='secondary-eval' \
    --backend=serial --skip_existing=0 \
    --run=1



DATASET_CODE=Aligned-Drop4-2022-08-08-TA1-S2-L8-ACC
DATA_DVC_DPATH=$(geowatch_dvc --tags="phase2_data")
DVC_EXPT_DPATH=$(geowatch_dvc --tags="phase2_expt")
TEST_DATASET=$DATA_DVC_DPATH/$DATASET_CODE/data.kwcoco.json
python -m watch.mlops.schedule_evaluation \
    --params="
        matrix:
            trk.pxl.model:
                - $DVC_EXPT_DPATH/models/fusion/Aligned-Drop4-2022-08-08-TA1-S2-L8-ACC/packages/Drop4_BAS_Retrain_V002/Drop4_BAS_Retrain_V002_epoch=31-step=16384.pt.pt
            trk.pxl.data.test_dataset:
                - $TEST_DATASET
            trk.pxl.data.window_scale_space: 15GSD
            trk.pxl.data.time_sampling:
                - "contiguous"
            trk.pxl.data.input_scale_space:
                - "15GSD"
            trk.poly.thresh:
                - 0.07
                - 0.08
                - 0.09
                - 0.095
                - 0.1
                - 0.105
                - 0.11
                - 0.12
                - 0.13
                - 0.14
                - 0.15
                - 0.16
                - 0.175
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
    --devices="0,1" --queue_size=2 \
    --queue_name='secondary-eval' \
    --backend=tmux --skip_existing=1 \
    --run=1


    python -m watch.tasks.fusion.predict --package_fpath=/home/joncrall/remote/toothbrush/data/dvc-repos/smart_expt_dvc/models/fusion/Aligned-Drop4-2022-08-08-TA1-S2-WV-PD-ACC/packages/Drop4_SC_RGB_scratch_V002/Drop4_SC_RGB_scratch_V002_epoch=155-step=78468.pt.pt --test_dataset=/home/joncrall/remote/toothbrush/data/dvc-repos/smart_expt_dvc/models/fusion/Aligned-Drop4-2022-08-08-TA1-S2-L8-ACC/crop/online_v1_kwcoco_for_sc_fielded/trk_poly_id_4aa82814/crop_4ee34f2e/crop_id_e7ec2c85/crop.kwcoco.json --pred_dataset=/home/joncrall/remote/toothbrush/data/dvc-repos/smart_expt_dvc/models/fusion/Aligned-Drop4-2022-08-08-TA1-S2-WV-PD-ACC/pred/act/Drop4_SC_RGB_scratch_V002_epoch=155-step=78468.pt/crop_id_e7ec2c85_crop.kwcoco/act_pxl_f447d96a/pred.kwcoco.json --input_scale_space=10GSD --output_scale_space=10GSD --window_scale_space=10GSD --chip_dims=512,512 --time_steps=5 --num_workers=4 --devices=0, --accelerator=gpu --batch_size=1

"""
