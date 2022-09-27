"""

python -m watch mlops "list"
python -m watch mlops "status"
python -m watch mlops "push pull evals"
python -m watch mlops "pull evals"
python -m watch mlops "pull packages"
python -m watch mlops "push evals"

"""
import ubelt as ub
import math
import numpy as np
import pandas as pd
import functools  # NOQA
# APPLY Monkey Patches
from watch.tasks.fusion import monkey  # NOQA
from watch import heuristics


fit_param_keys = heuristics.fit_param_keys
pred_param_keys = heuristics.pred_param_keys
DSET_CODE_TO_GSD = heuristics.DSET_CODE_TO_GSD


def evaluation_report():
    """
    MAIN FUNCTION

    from watch.mlops.expt_report import *  # NOQA

    row = reporter.orig_merged_df.loc[121]
    print(ub.repr2(row.to_dict(), nl=1))
    path = reporter.orig_merged_df.loc[121]['raw']

    import platform
    host = platform.node()
    path.shrinkuser(home=f'$HOME/remote/{host}')


    print(ub.repr2(row.to_dict(), nl=1))
    """
    import kwplot
    kwplot.autosns()
    from watch import heuristics
    from watch.mlops import expt_manager
    dvc_expt_dpath = heuristics.auto_expt_dvc()
    manager = expt_manager.DVCExptManager.coerce(dvc_expt_dpath)
    state = manager
    reporter = EvaluationReporter(state)
    reporter.load()
    reporter.status()
    reporter.plot()

    if 0:
        merged_df = reporter.orig_merged_df.copy()
        merged_df[merged_df.expt.str.contains('invar')]['mean_f1']
        merged_df[merged_df.in_production]['mean_f1']

        selected = merged_df[merged_df.in_production].sort_values('mean_f1')
        selected = selected[['siteprep_f1', 'active_f1', 'mean_f1', 'model']]
        selected['coi_mean_f1'] = selected[['siteprep_f1', 'active_f1']].mean(axis=1)
        selected = selected.sort_values('coi_mean_f1')
        print(selected)


class EvaluationReporter:
    """
    Manages handing the data off to experiment plotting functions.
    """

    def __init__(reporter, state):
        """
        Args:
            state (DVCExptManager | ExperimentState)
        """
        reporter.state = state
        reporter.dvc_expt_dpath = state.expt_dvc_dpath

        reporter.raw_df = None
        reporter.filt_df = None
        reporter.comp_df = None

        reporter.dpath = ub.Path.appdir('watch/expt-report').ensuredir()

        reporter.metric_registry = pd.DataFrame([
            {'name': 'coi_mAP', 'tasks': ['sc'], 'human': 'Pixelwise mAP (classes of interest)'},
            {'name': 'coi_mAUC', 'tasks': ['sc'], 'human': 'Pixelwise mAUC (classes of interest)'},
            {'name': 'mean_f1', 'tasks': ['sc'], 'human': 'IARPA SC mean F1'},

            {'name': 'salient_AP', 'tasks': ['bas'], 'human': 'Pixelwise Salient AP'},
            {'name': 'salient_AUC', 'tasks': ['bas'], 'human': 'Pixelwise Salient AUC'},
            {'name': 'BAS_F1', 'tasks': ['bas'], 'human': 'IARPA BAS F1'},
        ])
        reporter.metric_registry['type'] = 'metric'

        # TODO: add column types
        column_meanings = [
            {'name': 'raw', 'help': 'A full path to a file on disk that contains this info'},
            {'name': 'dvc', 'help': 'A path to a DVC sidecar file if it exists.'},
            {'name': 'type', 'help': 'The type of the row'},
            {'name': 'step', 'help': 'The number of steps taken by the most recent training run associated with the row'},
            {'name': 'total_steps', 'help': 'An estimate of the total number of steps the model associated with the row took over all training runs.'},
            {'name': 'model', 'help': 'The name of the learned model associated with this row'},
            {'name': 'test_dset', 'help': 'The name of the test dataset used to compute a metric associated with this row'},
            {'name': 'expt', 'help': 'The name of the experiment, i.e. training session that might have made several models'},
            {'name': 'dataset_code', 'help': 'The higher level dataset code associated with this row'},

            {'name': 'pred_cfg', 'help': 'A hash of the configuration used for pixel heatmap prediction'},
            {'name': 'trk_cfg', 'help': 'A hash of the configuration used for BAS tracking'},
            {'name': 'act_cfg', 'help': 'A hash of the configuration used for SC classification'},

            {'name': 'total_steps', 'help': 'An estimate of the total number of steps the model associated with the row took over all training runs.'},
        ]
        reporter.column_meanings = column_meanings

        # COLUMNS TO DOCUMENT
        # 'raw',
        # 'dvc',
        # 'type',
        # 'has_dvc',
        # 'has_raw',
        # 'needs_pull',
        # 'is_link',
        # 'is_broken',
        # 'unprotected',
        # 'needs_push',
        # 'dataset_code',
        # 'expt_dvc_dpath',
        # 'expt',
        # 'model',
        # 'test_dset',
        # 'pred_cfg',
        # 'trk_cfg',
        # 'act_cfg': 'SC Tracking Config',

        # 'has_teamfeat',
        # 'model_fpath',
        # 'pred_start_time',
        # 'class_mAP',
        # 'class_mAUC',
        # 'class_mAPUC',
        # 'coi_mAP',
        # 'coi_mAUC',
        # 'coi_mAPUC',
        # 'salient_AP',
        # 'salient_AUC',
        # 'salient_APUC',
        # 'co2_kg',
        # 'vram_gb',

        # 'total_hours',
        # 'iters_per_second',
        # 'cpu_name',
        # 'gpu_name',

        # 'disk_type',
        # 'accumulate_grad_batches',
        # 'arch_name',
        # 'channels',
        # 'chip_overlap',
        # 'class_loss',
        # 'decoder',
        # 'init',
        # 'learning_rate',
        # 'modulate_class_weights',
        # 'optimizer',
        # 'saliency_loss',
        # 'stream_channels',
        # 'temporal_dropout',
        # 'time_sampling',
        # 'time_span',
        # 'time_steps',
        # 'tokenizer',
        # 'upweight_centers',
        # 'use_cloudmask',
        # 'use_grid_positives',
        # 'bad_channels',
        # 'sensorchan',
        # 'true_multimodal',
        # 'hardware',
        # 'epoch',
        # 'step',
        # 'total_steps',
        # 'in_production',
        # 'Processing',

        # 'trk_use_viterbi': 'Viterbi Enabled',
        # 'trk_thresh': 'SC Tracking Threshold',
        # 'co2_kg': 'CO2 Emissions (kg)',
        # 'total_hours': 'Time (hours)',
        # 'sensorchan': 'Sensor/Channel',
        # 'has_teamfeat': 'Has Team Features',
        # 'eval_act+pxl': 'SC',
        # 'eval_trk+pxl': 'BAS',
        # ]

    def status(reporter, table=None):
        reporter.state.summarize()
        reporter.report_best(verbose=1)
        # if 0:
        #     if table is None:
        #         table = reporter.state.evaluation_table()
        #     loaded_table = load_extended_data(table, reporter.dvc_expt_dpath)
        #     loaded_table = pd.DataFrame(loaded_table)
        #     # dataset_summary_tables(dpath)
        #     initial_summary(table, loaded_table, reporter.dpath)

    def report_best(reporter, show_configs=0, verbose=0, top_k=2):
        import rich
        orig_merged_df = reporter.orig_merged_df
        metric_names = reporter.metric_registry.name
        cfg_names = ['pred_cfg', 'act_cfg', 'trk_cfg']
        id_names = ['model'] + cfg_names

        metric_names = reporter.metric_registry.name
        metric_cols = (ub.oset(metric_names) & orig_merged_df.columns)
        primary_metrics = (ub.oset(['mean_f1', 'BAS_F1']) & metric_cols)
        metric_cols = list((metric_cols & primary_metrics) | (metric_cols - primary_metrics))
        # print('orig_merged_df.columns = {}'.format(ub.repr2(list(orig_merged_df.columns), nl=1)))
        id_cols = list(ub.oset(id_names) & orig_merged_df.columns)

        test_datasets = orig_merged_df['test_dset'].dropna().unique().tolist()

        if verbose:
            rich.print('[orange1]-- REPORTING BEST --')
            print('test_datasets = {}'.format(ub.repr2(test_datasets, nl=1)))

        grouped_shortlists = {}
        group_keys = ['test_dset', 'type']

        for groupid, subdf in orig_merged_df.groupby(group_keys):
            if verbose:
                print('')
                rich.print(f'[orange1] -- <BEST ON: {groupid}> --')

            top_indexes = set()
            for metric in metric_cols:
                # TODO: maximize or minimize
                best_rows = subdf[metric].sort_values()
                top_indexes.update(best_rows.iloc[-top_k:].index)

            idxs = sorted(top_indexes)
            shortlist = subdf.loc[idxs]
            shortlist = shortlist.sort_values(metric_cols, na_position='first')

            show_configs = show_configs
            if show_configs:
                keys = list(set([
                    v for v in shortlist[ub.oset(cfg_names) & shortlist.columns].values.ravel()
                    if not pd.isnull(v)
                ]))
                resolved = reporter._build_cfg_rlut(keys)
                if verbose and show_configs:
                    rich.print('resolved = {}'.format(ub.repr2(resolved, nl=2)))

            # test_dset_to_best[test_dset] =
            if verbose:
                shortlist_small = shortlist[id_cols + metric_cols]
                rich.print(shortlist_small.to_string())
                print('')
                rich.print(f'[orange1] -- </BEST ON: {groupid}> --')
                print('')
            grouped_shortlists[groupid] = shortlist

        return grouped_shortlists

    def _build_cfg_rlut(reporter, keys):
        from watch.utils.reverse_hashid import ReverseHashTable
        if keys is None:
            candidates = ReverseHashTable.query(verbose=0)
        else:
            candidates = []
            for key in keys:
                candidates += ReverseHashTable.query(key, verbose=0)
        resolved = ub.ddict(list)
        for cand in candidates:
            resolved[cand['key']].extend(
                [f['data'] for f in cand['found']])
        for k in resolved.keys():
            if len(resolved[k]) == 1:
                resolved[k] = resolved[k][0]
        return resolved

    def load1(reporter):
        """
        Load basic data
        """
        table = reporter.state.evaluation_table()
        reporter.state.summarize()
        evaluations = table[~table['raw'].isnull()]
        reporter.raw_df = raw_df = pd.DataFrame(evaluations)

        if 0:
            col_stats_df = unique_col_stats(raw_df)
            print('Column Unique Value Frequencies')
            print(col_stats_df.to_string())

        test_dset_freq = raw_df['test_dset'].value_counts()
        print(f'test_dset_freq={test_dset_freq}')

        print('\nRaw')
        num_files_summary(raw_df)

        # Remove duplicate predictions on effectively the same dataset.
        # reporter.filt_df = filt_df = deduplicate_test_datasets(raw_df)
        reporter.raw_df = filt_df = raw_df

        print('\nDeduplicated (over test dataset)')
        num_files_summary(filt_df)

        USE_COMP = 0
        if USE_COMP:
            eval_types_to_locs = ub.ddict(list)
            for k, group in filt_df.groupby(['test_dset', 'model', 'pred_cfg']):
                eval_types = tuple(sorted(group['type'].unique()))
                eval_types_to_locs[eval_types].extend(group.index)
            print('Cross-Metric Comparable Locs')
            print(ub.repr2(ub.map_vals(len, eval_types_to_locs)))
            comparable_locs = list(ub.flatten(v for k, v in eval_types_to_locs.items() if len(k) > 0))
            reporter.comp_df = comp_df = filt_df.loc[comparable_locs]

            print('\nCross-Metric Comparable')
            num_files_summary(comp_df)

    def load2(reporter):
        """
        Load detailed data that might cross reference files
        """
        df = reporter.raw_df
        dvc_expt_dpath = reporter.dvc_expt_dpath
        reporter.big_rows = load_extended_data(df, dvc_expt_dpath)
        # reporter.big_rows = load_extended_data(reporter.comp_df, reporter.dvc_expt_dpath)
        # set(r['expt'] for r in reporter.big_rows)

        big_rows = reporter.big_rows
        orig_merged_df, other = clean_loaded_data(big_rows)
        reporter.orig_merged_df = orig_merged_df
        reporter.other = other

        # hard coded values
        human_mapping = {
            'coi_mAP': 'Pixelwise mAP (classes of interest)',
            'coi_mAUC': 'Pixelwise mAUC (classes of interest)',
            'salient_AP': 'Pixelwise Salient AP',
            'salient_AUC': 'Pixelwise Salient AUC',
            'mean_f1': 'IARPA SC mean F1',
            'BAS_F1': 'IARPA BAS F1',
            'act_cfg': 'SC Tracking Config',
            'trk_use_viterbi': 'Viterbi Enabled',
            'trk_thresh': 'SC Tracking Threshold',
            'co2_kg': 'CO2 Emissions (kg)',
            'total_hours': 'Time (hours)',
            'sensorchan': 'Sensor/Channel',
            'has_teamfeat': 'Has Team Features',
            'eval_act+pxl': 'SC',
            'eval_trk+pxl': 'BAS',
            'pred_input_space_scale': 'Input Scale',
            'pred_use_cloudmask': 'Cloudmask',
            'pred_resample_invalid_frames': 'Resample Invalid Frames',
            'pred_window_scale_space': 'Window Scale',
        }
        reporter.human_mapping = human_mapping
        reporter.iarpa_metric_lut = {
            'eval_act+pxl': 'mean_f1',
            'eval_trk+pxl': 'BAS_F1',
        }
        reporter.pixel_metric_lut = {
            'eval_act+pxl': 'coi_mAP',
            'eval_trk+pxl': 'salient_AP',
        }
        reporter.actcfg_to_label = other['actcfg_to_label']
        reporter.predcfg_to_label = other['predcfg_to_label']
        reporter.human_mapping.update(reporter.actcfg_to_label)
        reporter.human_mapping.update(reporter.predcfg_to_label)

    def load(reporter):
        reporter.load1()
        reporter.load2()

    def plot(reporter):
        plot_merged(reporter)


def plot_merged(reporter):
    from watch.mlops.plots import plot_pixel_ap_verus_auc
    from watch.mlops.plots import plot_pixel_ap_verus_iarpa
    from watch.mlops.plots import plot_resource_versus_metric

    reporter = reporter
    dpath = reporter.dpath
    orig_merged_df = reporter.orig_merged_df
    iarpa_metric_lut = reporter.iarpa_metric_lut
    pixel_metric_lut = reporter.pixel_metric_lut
    predcfg_to_label = reporter.predcfg_to_label
    actcfg_to_label = reporter.actcfg_to_label
    human_mapping = reporter.human_mapping

    # ['trk_thresh',
    #  'trk_morph_kernel',
    #  'trk_agg_fn',
    #  'trk_thresh_hysteresis',
    #  'trk_moving_window_size']

    common_plotkw = {
        # 'connect': 'expt',
        # 'mesh': 'model',
        # 'clique': 'model',
        'style': 'has_teamfeat',
        # 'star': 'in_production',
        'starkw': {'s': 500},
        's': 120,
    }

    merged_df = orig_merged_df.copy()

    if 0:
        from watch.utils import util_time
        deadline = util_time.coerce_datetime('2022-04-19')
        before_deadline = ((merged_df['pred_start_time'] < deadline) | merged_df['pred_start_time'].isnull())
        # after_deadline = ~before_deadline
        # merged_df = merged_df[after_deadline]
        merged_df = merged_df[before_deadline]

    if 0:
        chosen_pred_cfg = ub.invert_dict(predcfg_to_label)['pred_tta_time=0']
        # chosen_act_cfg = ub.invert_dict(actcfg_to_label)['trk_thresh=0.01,trk_use_viterbi=v1,v6']
        chosen_act_cfg = ub.invert_dict(actcfg_to_label)['trk_thresh=0,trk_use_viterbi=0']
        # chosen_pred_cfg = 'predcfg_abd043ec'
        # chosen_pred_cfg = 'predcfg_4d9147b0'
        # chosen_pred_cfg = 'predcfg_036fdb96'
        # chosen_act_cfg = 'actcfg_f1456a39'

        merged_df = merged_df[(
            (merged_df['pred_cfg'] == chosen_pred_cfg) &
            ((merged_df['act_cfg'] == chosen_act_cfg) | merged_df['act_cfg'].isnull())
        )]

    if 0:
        metrics = [
            'mean_f1',
            'BAS_F1',
            # 'salient_AP',
            # 'coi_mAP'
        ]
        # merged_df['pred_cfg'].value_counts()
        # merged_df['act_cfg'].value_counts()
        # HACK: need to maximize comparability, not the metric here.
        # Do this for viz purposes, dont present if it changes the conclusion
        # but might need to do this for visual clarity.
        rows = []
        for model, group in merged_df.groupby('model'):
            if len(group) > 1:
                chosen_idxs = [group[m].argmax() for m in metrics]
                row = group.iloc[sorted(set(chosen_idxs))]
            else:
                row = group
            rows.append(row)
        merged_df = pd.concat(rows)

    # describe_varied(merged_df, dpath, human_mapping=human_mapping)

    # plot_ta1_vs_l1(merged_df, human_mapping, iarpa_metric_lut, pixel_metric_lut, common_plotkw, dpath)

    plot_pixel_ap_verus_auc(merged_df, human_mapping, iarpa_metric_lut, pixel_metric_lut, common_plotkw, dpath)

    plot_pixel_ap_verus_iarpa(merged_df, human_mapping, iarpa_metric_lut, pixel_metric_lut, common_plotkw, dpath)

    plot_resource_versus_metric(merged_df, human_mapping, iarpa_metric_lut, pixel_metric_lut, common_plotkw, dpath)

    # plot_viterbii_analysis(merged_df, human_mapping, iarpa_metric_lut, pixel_metric_lut, common_plotkw, dpath)


def num_files_summary(df):
    expt_group = dict(list(df.groupby('dataset_code')))
    filt_summaries = []
    groups = sorted(expt_group.items())[::-1]
    for dataset_code, group in groups:
        # print('dataset_code = {!r}'.format(dataset_code))
        # print('Column Unique Value Frequencies')
        # col_stats_df2 = unique_col_stats(group)
        # print(col_stats_df2.to_string())
        row = {}
        type_evals = ub.dict_hist(group['type'])
        row['dataset_code'] = dataset_code
        row['num_experiments'] = len(group['expt'].unique())
        row['num_models'] = len(group['model'].unique())
        row['num_pxl_evals'] = type_evals.get('eval_pxl', 0)
        row['num_bas_evals'] = type_evals.get('eval_trk', 0)
        row['num_sc_evals'] = type_evals.get('eval_act', 0)
        filt_summaries.append(row)
    _summary_df = pd.DataFrame(filt_summaries)
    total_row = _summary_df.sum().to_dict()
    total_row['dataset_code'] = '*'
    summary_df = pd.DataFrame(filt_summaries + [total_row])
    print('Number of Models & Evaluations')
    print(summary_df.to_string(index=False))


def unique_col_stats(df):
    col_stats = ub.ddict(dict)
    import kwarray
    import numpy as np
    for key in df.columns:
        col_freq = np.array(list(ub.dict_hist(df[key]).values()))
        stats = kwarray.stats_dict(col_freq, median=True)
        stats['num_unique'] = stats.pop('shape')[0]
        col_stats[key] = stats
        # ['num_unique'] = len(unique_cols)
        # col_stats[key]['max'] = stats['max']
        # col_stats[key]['max'] = stats['max']
    col_stats_df = pd.DataFrame(col_stats)
    # Hack
    col_stats_df = col_stats_df.drop(['dataset_code', 'dvc', 'raw'], axis=1)
    col_stats_df = col_stats_df.astype(int)
    return col_stats_df


def load_extended_data(df, dvc_expt_dpath):
    from watch.tasks.fusion import aggregate_results as agr
    rows = df.to_dict('records')
    big_rows = []
    errors = []
    for row in ub.ProgIter(rows, desc='load'):
        big_row = row.copy()
        fpath = row['raw']
        try:
            if row['type'] == 'eval_pxl':
                pxl_info = agr.load_pxl_eval(fpath, dvc_expt_dpath)
                big_row['info'] = pxl_info
            elif row['type'] == 'eval_act':
                sc_info = agr.load_sc_eval(fpath, dvc_expt_dpath)
                big_row['info'] = sc_info
            elif row['type'] == 'eval_trk':
                bas_info = agr.load_bas_eval(fpath, dvc_expt_dpath)
                big_row['info'] = bas_info
            else:
                raise KeyError(row['type'])
            big_rows.append(big_row)
        except Exception as ex:
            errors.append((ex, row))
    import rich
    if len(errors):
        rich.print('[red] ' + repr(errors[0]))
        rich.print(f'[red] {len(errors)=}')
    else:
        print(f'{len(errors)=}')
    return big_rows


def clean_loaded_data(big_rows):
    """
    Turn the nested "loaded" data into flat data for tabulation.
    Also combine eval types together into a single row per model / config.
    """
    from watch.tasks.fusion import aggregate_results as agr
    import kwcoco

    def _is_teamfeat(sensorchan):
        unique_chans = sum([s.chans for s in sensorchan.streams()]).fuse().to_set()
        if isinstance(unique_chans, float) and math.isnan(unique_chans):
            return False
        return any([a in unique_chans for a in ['depth', 'invariant', 'invariants', 'matseg', 'land']])

    _actcfg_to_track_config = ub.ddict(list)
    _trkcfg_to_track_config = ub.ddict(list)
    _prdcfg_to_pred_config = ub.ddict(list)
    simple_rows = []
    missing_models = []
    blocklist = {
        'S2:|R|G',
        'S2:|G|R|,invariants:16)',
        'S2:(RGB|land:8,R|G,R|G|land:8)',
    }

    passlist = {
        'BGR',
        'BGRN',
        'RGB|near-ir1|near-ir2|red-edge|yellow',
        'BGR|near-ir1',
        'BGRNSH|land:8|matseg:4|mat_up5:64',
        'BGRNSH',
        'BGR|near-ir1|depth',
        'RGB',
        'RGB|near-ir1',
        'RGB|land:8',
        'RGB|near-ir1|near-ir2|depth',
        'RGB|near_ir1|near_ir2|depth',
        'land:8',
        'invariants:16',
        'matseg:4',
        'matseg:4|mat_up5:64',
        'G|R|N|S|H|land:8',
    }
    chan_blocklist = {
        'R|G',
        'G|R',
        'G|R|N|S|H',
        'R|G|land:8',
        'RGB|near-ir1|depth',
        'G|R|N|S|H|land:8|matseg:4|mat_up5:64',
        'BGRNSH|land:8',
    }

    for big_row in ub.ProgIter(big_rows, desc='big rows'):
        # fpath = big_row['raw']
        row = ub.dict_diff(big_row, {'info'})
        info = big_row['info']

        param_type = info['param_types']

        meta = param_type['meta']
        fit_params = param_type['fit']
        pred_params = param_type['pred']
        model_fpath = pred_params['pred_model_fpath']

        track_params = info['param_types'].get('track', {})

        # Shrink and check the sensorchan spec
        request_sensorchan = kwcoco.SensorChanSpec.coerce(
            agr.shrink_channels(fit_params['channels']))
        fit_params['channels'] = request_sensorchan.spec
        sensorchan = request_sensorchan

        if 0:
            # Hack for Phase1 Models with improper sensorchan.
            # This can likely be removed as we move forward in Phase 2.

            # Dont trust what the model info says about channels, look
            # at the model stats to be sure.
            if model_fpath and model_fpath.exists():
                stats = resolve_model_info(model_fpath)
                real_chan_parts = ub.oset()
                senschan_parts = []
                real_sensors = []
                for input_row in stats['model_stats']['known_inputs']:
                    known_sensorchan = agr.shrink_channels(input_row['sensor'] + ':' + input_row['channel'])
                    known_sensorchan = kwcoco.SensorChanSpec.coerce(known_sensorchan)
                    real_chan = known_sensorchan.chans.spec
                    if real_chan not in chan_blocklist:
                        if real_chan not in passlist:
                            print(f'Unknown real_chan={real_chan}')
                        real_chan_parts.add(real_chan)
                        real_sensors.append(input_row['sensor'])
                        senschan_parts.append('{}:{}'.format(input_row['sensor'], real_chan))
                model_sensorchan = ','.join(sorted(set(senschan_parts)))
                model_sensorchan = kwcoco.SensorChanSpec.coerce(model_sensorchan)

                model_parts = model_sensorchan.normalize().spec.split(',')
                request_parts = request_sensorchan.normalize().spec.split(',')
                if not request_parts.issubset(model_parts):
                    fit_params['bad_channels'] = True
                else:
                    fit_params['bad_channels'] = False
            else:
                missing_models.append(model_fpath)

                if 'Cropped' in big_row['test_dset']:
                    # Hack
                    sensors = ['WV', 'S2']
                elif 'Cropped' in big_row['test_dset']:
                    sensors = ['S2', 'L8']
                else:
                    sensors = ['*']

                channels = kwcoco.ChannelSpec.coerce(fit_params['channels'])
                senschan_parts = []
                for sensor in sensors:
                    for chan in channels.streams():
                        senschan_parts.append(f'{sensor}:{chan.spec}')

                sensorchan = ','.join(sorted(senschan_parts))
                sensorchan = kwcoco.SensorChanSpec.coerce(sensorchan)
                fit_params['bad_channels'] = True

            # MANUAL HACK:
            if 1:
                sensorchan = ','.join([p.spec for p in sensorchan.streams() if p.chans.spec not in blocklist])
        else:
            fit_params['bad_channels'] = False

        fit_params['sensorchan'] = sensorchan
        row['has_teamfeat'] = _is_teamfeat(sensorchan)

        fit_param_keys2 = list(fit_param_keys) + ['bad_channels', 'channels']
        selected_fit_params = ub.dict_isect(fit_params, fit_param_keys2)

        act_cfg = row.get('act_cfg', None)
        if not is_null(act_cfg):
            track_cfg = param_type.get('track', None)
            row.update(track_cfg)
            _actcfg_to_track_config[act_cfg].append(track_cfg)

        trk_cfg = row.get('trk_cfg', None)
        if not is_null(trk_cfg):
            track_cfg = param_type.get('track', None)
            row.update(track_cfg)
            _trkcfg_to_track_config[trk_cfg].append(track_cfg)

        pred_cfg = row.get('pred_cfg', None)
        if not is_null(trk_cfg):
            pred_config = param_type.get('pred', None)
            pred_config = ub.dict_isect(pred_config, pred_param_keys)
            if pred_config.get('pred_tta_time', None) is None:
                pred_config['pred_tta_time'] = 0
            if pred_config.get('pred_tta_fliprot', None) is None:
                pred_config['pred_tta_fliprot'] = 0
            row.update(pred_config)
            _prdcfg_to_pred_config[pred_cfg].append(pred_config)

        resource = param_type.get('resource', {})
        row['model_fpath'] = model_fpath
        row.update(**meta)
        row.update(info['metrics'])
        row.update(resource)
        row.update(selected_fit_params)
        row['pred_params'] = pred_params
        row['track_params'] = track_params
        row['fit_params'] = fit_params
        simple_rows.append(row)

    simple_df = pd.DataFrame(simple_rows)
    print(f'{len(simple_df)=}')
    # simple_df['sensorchan'].unique()
    # simple_df[simple_df['sensorchan'].isnull()]
    # simple_df['bad_channels']

    actcfg_to_track_config = {}
    for actcfg, track_cfgs in _actcfg_to_track_config.items():
        unique_configs = list(ub.unique(track_cfgs, key=ub.hash_data))
        assert len(unique_configs) == 1
        actcfg_to_track_config[actcfg] = unique_configs[0]

    trkcfg_to_track_config = {}
    for trkcfg, track_cfgs in _trkcfg_to_track_config.items():
        unique_configs = list(ub.unique(track_cfgs, key=ub.hash_data))
        assert len(unique_configs) == 1
        trkcfg_to_track_config[trkcfg] = unique_configs[0]

    prdcfg_to_pred_config = {}
    for predcfg, track_cfgs in _prdcfg_to_pred_config.items():
        unique_configs = list(ub.unique(track_cfgs, key=ub.hash_data))
        if len(unique_configs) == 1:
            prdcfg_to_pred_config[predcfg] = unique_configs[0]
        else:
            print(f'{unique_configs=}')
            print('predcfg = {}'.format(ub.repr2(predcfg, nl=1)))

    if True:
        # Get activity config labels
        actcfg_to_label = {}
        varied_act = ub.varied_values(actcfg_to_track_config.values(), 1, default=None)
        varied_act_keys = sorted(varied_act.keys())
        for k, v in actcfg_to_track_config.items():
            c = ub.dict_isect(v, varied_act_keys)
            label = ub.repr2(c, compact=1)
            actcfg_to_label[k] = label

        # Get activity config labels
        predcfg_to_label = {}
        varied_act = ub.varied_values(prdcfg_to_pred_config.values(), 1, default=None)
        varied_act_keys = sorted(varied_act.keys())
        for k, v in prdcfg_to_pred_config.items():
            c = ub.dict_isect(v, varied_act_keys)
            label = ub.repr2(c, compact=1)
            predcfg_to_label[k] = label

    bad_expts = simple_df[simple_df['bad_channels']]['expt']
    print('bad_expts =\n{}'.format(ub.repr2(bad_expts, nl=1)))

    ub.dict_hist(simple_df['channels'])
    merged_rows = []
    for pred_key, group in simple_df.groupby(['model', 'pred_cfg']):
        # Can propogate pixel metrics to child groups
        type_to_subgroup = dict(list(group.groupby('type')))
        pxl_group = type_to_subgroup.pop('eval_pxl', None)
        if pxl_group is not None:
            if len(pxl_group) > 1:
                print(f'Warning more than one pixel group for {pred_key}')
            pxl_row = pxl_group.iloc[0]

            if len(type_to_subgroup) == 0:
                pxl_row = pxl_row.to_dict()
                if not math.isnan(pxl_row.get('coi_mAP', np.nan)):
                    srow = pxl_row.copy()
                    srow['type'] = 'eval_act+pxl'
                    merged_rows.append(srow)

                if not math.isnan(pxl_row.get('salient_AP', np.nan)):
                    srow = pxl_row.copy()
                    srow['type'] = 'eval_trk+pxl'
                    merged_rows.append(srow)

            for type, subgroup in type_to_subgroup.items():
                for srow in subgroup.to_dict('records'):
                    srow['type'] = srow['type'] + '+pxl'
                    for k1, v1 in pxl_row.items():
                        v2 = srow.get(k1, None)
                        if v2 is None or (isinstance(v2, float) and math.isnan(v2)):
                            srow[k1] = v1
                    merged_rows.append(srow)

    merged_df = pd.DataFrame(merged_rows)
    merged_df['sensorchan'] = merged_df['sensorchan'].apply(str)
    print(f'{len(merged_df)=}')

    total_carbon_cost = simple_df[simple_df['type'] == 'eval_pxl']['co2_kg'].sum()
    # total_carbon_cost = merged_df['co2_kg'].sum()
    print(f'{total_carbon_cost=}')
    merged_df['gpu_name'] = merged_df['gpu_name'].fillna('?')
    merged_df['cpu_name'] = merged_df['cpu_name'].fillna('?')
    merged_df['hardware'] = merged_df['hardware'].fillna('?')
    # cpu_names = merged_df['cpu_name'].apply(lambda x: x.replace('Intel(R) Core(TM) ', ''))
    # gpu_names = merged_df['gpu_name']
    # merged_df['hardware'] = ['{} {}'.format(c, g) for c, g in zip(cpu_names, gpu_names)]

    other = {
        'actcfg_to_label': actcfg_to_label,
        'predcfg_to_label': predcfg_to_label,
    }

    # TODO: compute total steps including with initialized continuations
    epoch_info = merged_df['model'].apply(checkpoint_filepath_info).values
    merged_df['epoch'] = [e.get('epoch', None) if e else None for e in epoch_info]
    merged_df['step'] = [e.get('step', None) if e else None for e in epoch_info]

    merged_df['init'] = merged_df['init'].apply(lambda x: x[:-3] if x.endswith('.pt') else x)

    known_models = merged_df['model'].unique()
    init_models = merged_df['init'].unique()
    traceable = sorted(set(init_models) & set(known_models))

    merged_df = merged_df.reset_index(drop=True)  # because of enumerate
    init_to_idxs = ub.map_vals(sorted, ub.invert_dict(dict(enumerate(merged_df['init'])), 0))
    model_to_idxs = ub.map_vals(sorted, ub.invert_dict(dict(enumerate(merged_df['model'])), 0))
    import networkx as nx
    g = nx.DiGraph()
    for model in traceable:
        pred_idxs = model_to_idxs[model]
        succ_idxs = init_to_idxs[model]
        pred_df = merged_df.loc[pred_idxs]
        succ_df = merged_df.loc[succ_idxs]
        parents = pred_df['model'].unique()
        children = succ_df['model'].unique()
        for p in parents:
            for c in children:
                g.add_edge(p, c)

    if 0:
        from cmd_queue.util.util_networkx import write_network_text
        print(write_network_text(g))

    # Ensure we compute total epochs for earlier models first
    merged_df['total_steps'] = merged_df['step']
    for model in list(nx.topological_sort(g)):
        pred_idxs = model_to_idxs[model]
        succ_idxs = init_to_idxs.get(model, [])
        if len(succ_idxs):
            pred_df = merged_df.loc[pred_idxs]
            succ_df = merged_df.loc[succ_idxs]
            assert len(pred_df['total_steps'].unique()) == 1
            prev = pred_df['total_steps'].iloc[0]
            merged_df.loc[succ_idxs, 'total_steps'] += prev

    if 0:
        print(ub.repr2(merged_df.columns.tolist()))

    # Flag which models went into production.
    from watch.tasks.fusion import production
    import kwarray
    production_models = [row['name'].replace('.pt', '') for row in production.PRODUCTION_MODELS]
    model_names = np.array([n.replace('.pt', '') for n in merged_df['model']])
    stared_models = set(model_names) & set(production_models)
    star_flags = kwarray.isect_flags(model_names, stared_models)
    merged_df['in_production'] = star_flags

    if 'trk_use_viterbi' in merged_df.columns:
        merged_df.loc[merged_df['trk_use_viterbi'].isnull(), 'trk_use_viterbi'] = 0

    if 'track_agg_fn' in merged_df.columns:
        merged_df['track_agg_fn'] = merged_df['trk_agg_fn'].fillna('probs')
        flags = 1 - group['trk_thresh_hysteresis'].isnull()
        merged_df['trk_thresh_hysteresis'] = merged_df['trk_thresh_hysteresis'].fillna(0) + (flags * merged_df['trk_thresh'])

    actcfg_to_label = other['actcfg_to_label']
    predcfg_to_label = other['predcfg_to_label']
    label_to_cfgstr = ub.invert_dict(actcfg_to_label)
    try:
        # hack
        a = label_to_cfgstr['trk_thresh=0,trk_use_viterbi=0']
        b = label_to_cfgstr['trk_thresh=0.0,trk_use_viterbi=0']
        merged_df.loc[merged_df['act_cfg'] == b, 'act_cfg'] = a
    except Exception:
        pass

    is_l1 = np.array(['L1' in c for c in merged_df['dataset_code']])
    is_ta1 = np.array(['TA1' in c for c in merged_df['dataset_code']])
    merged_df.loc[is_l1, 'Processing'] = 'L1'
    merged_df.loc[is_ta1, 'Processing'] = 'TA1'

    return merged_df, other


def is_null(x):
    return (isinstance(x, float) and math.isnan(x)) or x is None or not bool(x)


def resolve_model_info(model_fpath):
    cacher = ub.Cacher('model_info_memo', depends=[str(model_fpath)], appname='watch')
    stats = cacher.tryload()
    if stats is None:
        from watch.cli.torch_model_stats import torch_model_stats
        stats = torch_model_stats(model_fpath)
        cacher.save(stats)
    return stats


def checkpoint_filepath_info(fname):
    """
    Finds information encoded in the checkpoint/model file path.

    hacky

    TODO:
        We need to ensure this info is encoded inside the file header as well!

    Ignore
        parse.parse('{prefix}foo={bar}', 'foo=3')
        parse.parse('{prefix}foo={bar}', 'afoao=3')

    Example:
        >>> fnames = [
        >>>     'epoch=1-step=10.foo',
        >>>     'epoch=1-step=10-v2.foo',
        >>>     'epoch=1-step=10',
        >>>     'epoch=1-step=10-v2',
        >>>     'junkepoch=1-step=10.foo',
        >>>     'junk/epoch=1-step=10-v2.foo',
        >>>     'junk-epoch=1-step=10',
        >>>     'junk_epoch=1-step=10-v2',
        >>> ]
        >>> for fname in fnames:
        >>>     info = checkpoint_filepath_info(fname)
        >>>     print(f'info={info}')
        info={'epoch': 1, 'step': 10, 'ckpt_ver': 'v0'}
        info={'epoch': 1, 'step': 10, 'ckpt_ver': 'v2'}
        info={'epoch': 1, 'step': 10, 'ckpt_ver': 'v0'}
        info={'epoch': 1, 'step': 10, 'ckpt_ver': 'v2'}
        info={'epoch': 1, 'step': 10, 'ckpt_ver': 'v0'}
        info={'epoch': 1, 'step': 10, 'ckpt_ver': 'v2'}
        info={'epoch': 1, 'step': 10, 'ckpt_ver': 'v0'}
        info={'epoch': 1, 'step': 10, 'ckpt_ver': 'v2'}
    """
    import parse
    # We assume it must have this
    suffix = ''.join(fname.partition('epoch=')[1:])
    # Hack: making name assumptions
    parsers = [
        parse.Parser('epoch={epoch:d}-step={step:d}-{ckpt_ver}.{ext}'),
        parse.Parser('epoch={epoch:d}-step={step:d}.{ext}'),
        parse.Parser('epoch={epoch:d}-step={step:d}-{ckpt_ver}'),
        parse.Parser('epoch={epoch:d}-step={step:d}'),
    ]
    # results = parser.parse(str(path))
    info = None
    for parsers in parsers:
        result = parsers.parse(suffix)
        if result:
            break
    if result:
        info = result.named
        if 'ckpt_ver' not in info:
            info['ckpt_ver'] = 'v0'
        info = ub.dict_diff(info, {'ext', 'prefix'})
    return info


# def deduplicate_test_datasets(raw_df):
#     """
#     The same model might have been run on two variants of the dataset.
#     E.g. a RGB model might have run on data_vali.kwcoco.json and
#     combo_DILM.kwcoco.json. The system sees these as different datasets
#     even though the model will use the same subset of both. We define
#     a heuristic ordering and then take just one of them.
#     """
#     preference = {
#         'Cropped-Drop3-TA1-2022-03-10_combo_DLM_s2_wv_vali.kwcoco': 0,
#         'Cropped-Drop3-TA1-2022-03-10_combo_DL_s2_wv_vali.kwcoco': 1,
#         'Cropped-Drop3-TA1-2022-03-10_data_wv_vali.kwcoco': 2,
#         'Aligned-Drop3-TA1-2022-03-10_combo_LM_nowv_vali.kwcoco': 0,
#         'Aligned-Drop3-TA1-2022-03-10_combo_LM_vali.kwcoco': 1,
#     }
#     FILTER_DUPS = 1
#     if FILTER_DUPS:
#         keep_locs = []
#         for k, group in raw_df.groupby(['dataset_code', 'model', 'pred_cfg', 'type']):
#             prefs = group['test_dset'].apply(lambda x: preference.get(x, 0))
#             keep_flags = prefs == prefs.min()
#             keep_locs.extend(group[keep_flags].index)
#         print(f'Keep {len(keep_locs)} / {len(raw_df)} drop3 evals')
#         filt_df = raw_df.loc[keep_locs]
#     else:
#         filt_df = raw_df.copy()
#     return filt_df
