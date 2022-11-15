"""

python -m watch mlops "list"
python -m watch mlops "status"
python -m watch mlops "push pull evals"
python -m watch mlops "pull evals"
python -m watch mlops "pull packages"
python -m watch mlops "push evals"

SeeAlso:
    ~/code/watch/dev/reports/report_2022_09_xx.py

"""
import ubelt as ub
import math
import numpy as np
import pandas as pd
import functools  # NOQA
# APPLY Monkey Patches
from watch.tasks.fusion import monkey  # NOQA
from watch import heuristics
from watch.mlops import smart_pipeline as smart


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
    expt_dvc_dpath = heuristics.auto_expt_dvc()
    manager = expt_manager.DVCExptManager.coerce(expt_dvc_dpath)
    state = manager
    reporter = EvaluationReporter(state)
    reporter.load()
    reporter.status()
    reporter.plot()

    if 0:
        merged_df = reporter.orig_merged_df.copy()
        merged_df[merged_df.expt.str.contains('invar')]['sc_macro_f1']
        merged_df[merged_df.in_production]['sc_macro_f1']

        selected = merged_df[merged_df.in_production].sort_values('sc_macro_f1')
        selected = selected[['siteprep_f1', 'active_f1', 'sc_macro_f1', 'model']]
        selected['sc_macro_f1'] = selected[['siteprep_f1', 'active_f1']].mean(axis=1)
        selected = selected.sort_values('sc_macro_f1')
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
        reporter.expt_dvc_dpath = state.expt_dvc_dpath

        reporter.raw_df = None
        reporter.filt_df = None
        reporter.comp_df = None

        reporter.dpath = ub.Path.appdir('watch/expt-report').ensuredir()

        reporter.metric_registry = pd.DataFrame([
            {'name': 'coi_mAP', 'tasks': ['sc'], 'human': 'Pixelwise mAP (classes of interest)'},
            {'name': 'coi_mAUC', 'tasks': ['sc'], 'human': 'Pixelwise mAUC (classes of interest)'},
            {'name': 'sc_macro_f1', 'tasks': ['sc'], 'human': 'IARPA SC mean F1'},

            {'name': 'salient_AP', 'tasks': ['bas'], 'human': 'Pixelwise Salient AP'},
            {'name': 'salient_AUC', 'tasks': ['bas'], 'human': 'Pixelwise Salient AUC'},
            {'name': 'BAS_F1', 'tasks': ['bas'], 'human': 'IARPA BAS F1'},
        ])
        reporter.metric_registry['type'] = 'metric'

        # TODO: add column types
        column_meanings = smart.get_column_meanings()
        reporter.column_meanings = column_meanings

    def status(reporter, table=None):
        reporter.state.summarize()
        reporter.report_best(verbose=1)
        # if 0:
        #     if table is None:
        #         table = reporter.state.evaluation_table()
        #     loaded_table = load_extended_data(table, reporter.expt_dvc_dpath)
        #     loaded_table = pd.DataFrame(loaded_table)
        #     # dataset_summary_tables(dpath)
        #     initial_summary(table, loaded_table, reporter.dpath)

    def report_best(reporter, show_configs=0, verbose=0, top_k=2):
        """
        Ignore:
            show_configs = 0
            verbose = 1
            top_k = 2
        """
        import rich
        orig_merged_df = reporter.orig_merged_df
        metric_names = reporter.metric_registry.name

        cfg_names = reporter.state.hashed_cfgkeys
        id_names = ['trk_model', 'act_model'] + cfg_names

        # metric_names = reporter.metric_registry.name
        metric_names = [c for c in orig_merged_df.columns if 'metrics.' in c]
        resource_names = [c for c in orig_merged_df.columns if 'resource.' in c]
        metric_cols = (ub.oset(metric_names) & orig_merged_df.columns)
        resource_cols = (ub.oset(resource_names) & orig_merged_df.columns)

        metric_cols = [c for c in metric_cols if '.tau' not in c and '.rho' not in c]
        metric_cols = [c for c in metric_cols if '.salient_AUC' not in c]
        metric_cols = [c for c in metric_cols if '.salient_APUC' not in c]
        metric_cols = [c for c in metric_cols if '.macro_f1_active' not in c]
        metric_cols = [c for c in metric_cols if '.macro_f1_siteprep' not in c]
        metric_cols = [c for c in metric_cols if '.mAPUC' not in c]
        metric_cols = [c for c in metric_cols if '.mAUC' not in c]
        metric_cols = [c for c in metric_cols if ' ' not in c]
        # resource_cols = [c for c in resource_cols if 'total' in c or 'kwh' in c]
        resource_cols = [c for c in resource_cols if '.hardware' not in c]
        resource_cols = list(resource_cols)

        primary_metrics = (ub.oset(['act.poly.metrics.macro_f1', 'trk.poly.metrics.f1']) & metric_cols)
        metric_cols = list((metric_cols & primary_metrics) | (metric_cols - primary_metrics))

        # print('orig_merged_df.columns = {}'.format(ub.repr2(list(orig_merged_df.columns), nl=1)))
        id_cols = list(ub.oset(id_names) & orig_merged_df.columns)

        # test_datasets = orig_merged_df['test_dset'].dropna().unique().tolist()
        # if verbose:
        #     rich.print('[orange1]-- REPORTING BEST --')
        #     print('test_datasets = {}'.format(ub.repr2(test_datasets, nl=1)))

        grouped_shortlists = {}
        group_keys = ub.oset(['test_trk_dset', 'test_act_dset', 'type'])
        group_keys = list(group_keys & orig_merged_df.columns.intersection(group_keys))

        def _condense_report_df(df):
            col_mapper = {c: c for c in df.columns}
            _metric_cols = []
            for k in col_mapper.keys():
                k2 = k.replace('metrics.', '')
                k2 = k2.replace('resource.', '')
                col_mapper[k] = k2
                _metric_cols.append(k2)
            df = df.rename(col_mapper, axis=1)
            drop_cols = [k for k in _metric_cols if df[k].isnull().all()]
            df = df.drop(drop_cols, axis=1)
            cpu_cols = [c for c in df.columns if '.cpu_name' in c]
            gpu_cols = [c for c in df.columns if '.gpu_name' in c]
            for c in cpu_cols:
                df[c] = df[c].apply(lambda x: x if not isinstance(x, str) else x.replace('Intel(R) Core(TM) ', ''))
            for c in gpu_cols:
                df[c] = df[c].apply(lambda x: x if not isinstance(x, str) else x.replace('NVIDIA GeForce ', ''))
            return df

        for groupid, subdf in orig_merged_df.groupby(group_keys):
            # group_row = ub.dzip(group_keys, groupid)
            # if '_poly_' not in group_row['type']:
            #     continue
            if verbose:
                print('')
                rich.print(f'[orange1] -- <BEST ON: {groupid}> --')

            top_indexes = set()
            for metric in metric_cols:
                best_rows = subdf[metric].sort_values()
                top_indexes.update(best_rows.iloc[-top_k:].index)

            idxs = sorted(top_indexes)
            shortlist = subdf.loc[idxs]
            shortlist = shortlist.sort_values(metric_cols, na_position='first')

            show_configs = show_configs

            if show_configs:
                cfg_cols = shortlist[ub.oset(cfg_names) & shortlist.columns]
                keys = [
                    v for v in ub.unique(cfg_cols.values[::-1].ravel())
                    if not pd.isnull(v)][::-1]
                resolved = reporter._build_cfg_rlut(keys)
                resolved = ub.udict(resolved).subdict(keys)
                if verbose and show_configs:
                    rich.print('resolved = {}'.format(ub.repr2(resolved, sort=0, nl=2)))

            # test_dset_to_best[test_dset] =
            if verbose:
                shortlist_small = shortlist[id_cols + metric_cols + resource_cols]
                shortlist_small = _condense_report_df(shortlist_small)
                # rich.print(shortlist_small.T.to_string())
                rich.print(shortlist_small.to_string())
                print('')
                rich.print(f'[orange1] -- </BEST ON: {groupid}> --')
                print('')
            grouped_shortlists[groupid] = shortlist

        return grouped_shortlists

    def serialize_rows(reporter):
        from kwcoco.util.util_json import ensure_json_serializable
        table = reporter.orig_merged_df
        # state = reporter.state

        colnames = ub.oset(reporter.orig_merged_df.columns)
        # column_nestings = util_param_grid.dotkeys_to_nested(colnames)
        dotted = ub.oset([c for c in colnames if '.' in c])
        metric_cols = ub.oset([c for c in dotted if 'metrics.' in c])
        meta_cols = ub.oset([c for c in dotted if 'meta.' in c])
        resource_cols = ub.oset([c for c in dotted if 'resource.' in c])
        fit_cols = ub.oset([c for c in dotted if 'fit.' in c])
        param_cols = dotted - (metric_cols | fit_cols | resource_cols | meta_cols)

        for row in table.to_dict('records'):
            row = ub.udict(ensure_json_serializable(row))
            walker = ub.IndexableWalker(row)
            for path, val in walker:
                if hasattr(val, 'spec'):
                    walker[path] = val.spec

            summary = {
                # 'model': row['model'],
                # 'file_name': pkg_fpath,
                'pred_params': row & param_cols,
                'fit_params': row & fit_cols,
                'resources': row & resource_cols,
                'meta': row & meta_cols,
                'metrics': row & metric_cols,
            }
            yield summary, row

    def build_analysis(reporter):
        from watch.utils.result_analysis import ResultAnalysis
        from watch.utils.result_analysis import Result
        # import json

        results = []
        for summary, details in reporter.serialize_rows():
            params = {
                **summary['fit_params'],
                **summary['pred_params'],
            }
            for k, v in params.items():
                if isinstance(v, list):
                    params[k] = repr(v)
            name = 'row_' + ub.hash_data(summary)[0:8]
            metrics = ub.udict(summary['metrics'])  # - {'properties'}
            result = Result(name, params, metrics)
            results.append(result)

        analysis = ResultAnalysis(
            results,
            metric_objectives={
                'act.poly.metrics.bas_f1': 'max',
                'act.poly.metrics.macro_f1': 'max'
            },
            metrics=[
                # 'act.poly.metrics.bas_f1',
                'act.poly.metrics.macro_f1'
            ]
        )
        analysis.build()
        analysis.report()
        return analysis

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

        # test_dset_freq = raw_df['test_dset'].value_counts()
        # print(f'test_dset_freq={test_dset_freq}')

        print('\nRaw')
        num_files_summary(raw_df)

        # Remove duplicate predictions on effectively the same dataset.
        # reporter.filt_df = filt_df = deduplicate_test_datasets(raw_df)
        reporter.raw_df = filt_df = raw_df

        # print('\nDeduplicated (over test dataset)')
        # num_files_summary(filt_df)

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
        expt_dvc_dpath = reporter.expt_dvc_dpath
        big_rows = reporter.big_rows = load_extended_data(df, expt_dvc_dpath)
        # reporter.big_rows = load_extended_data(reporter.comp_df, reporter.expt_dvc_dpath)
        # set(r['expt'] for r in reporter.big_rows)
        big_rows = reporter.big_rows
        cleaned_df = clean_loaded_data(big_rows, expt_dvc_dpath)
        # reporter.other = other
        reporter.orig_cleaned_df = cleaned_df

        if 0:
            # Remove non-varying (interesting) columns to make development more
            # sane
            keep_cols = []
            for col in cleaned_df.columns:
                keep = True
                try:
                    unique_vals = cleaned_df[col].unique()
                    unique_vals = unique_vals[~pd.isnull(unique_vals)]
                    if len(unique_vals) == 1:
                        if '.' in col:
                            keep = False
                except Exception:
                    pass
                if keep:
                    keep_cols.append(col)
            cleaned_df = cleaned_df[keep_cols]

        if 1:
            # Merge rows from earlier pipeline steps to all of the descendant
            # rows that depend on it.
            from watch.utils import util_param_grid
            colnames = ub.oset(cleaned_df.columns)
            column_nestings = util_param_grid.dotkeys_to_nested(colnames)
            # non_nested = [k for k, v in column_nestings.items() if k == v]
            print(ub.repr2(column_nestings, sort=0))
            # column_nestings['trk']
            # column_nestings['act']

            type_to_idkeys = {
                'eval_trk_pxl_fpath': ['trk_model', 'test_trk_dset', 'trk_pxl_cfg'],
                'eval_trk_poly_fpath': ['trk_model', 'test_trk_dset', 'trk_pxl_cfg', 'trk_poly_cfg'],
            }

            metric_names = [c for c in cleaned_df.columns if 'metrics.' in c]
            metric_cols = (ub.oset(metric_names) & cleaned_df.columns)

            def link_types(smaller_row_type, larger_row_type, col_prefix):
                move_cols = [c for c in metric_cols if col_prefix in c]
                if move_cols:
                    smaller_keys = type_to_idkeys[smaller_row_type]
                    smaller = cleaned_df[cleaned_df.type == smaller_row_type]
                    larger = cleaned_df[cleaned_df.type == larger_row_type]
                    new_larger = my_nonstandard_merge(smaller, larger, smaller_keys, move_cols)
                    cleaned_df.loc[new_larger.index, move_cols] = new_larger.loc[:, move_cols].values

            smaller_row_type = 'eval_trk_pxl_fpath'
            larger_row_type = 'eval_trk_poly_fpath'
            col_prefix = 'trk.pxl'
            link_types(smaller_row_type, larger_row_type, col_prefix)

            smaller_row_type = 'eval_trk_pxl_fpath'
            larger_row_type = 'eval_act_poly_fpath'
            col_prefix = 'trk.pxl'
            link_types(smaller_row_type, larger_row_type, col_prefix)
            # move_cols = [c for c in metric_cols if col_prefix in c]
            # smaller_keys = type_to_idkeys[smaller_row_type]
            # smaller = cleaned_df[cleaned_df.type == smaller_row_type]
            # larger = cleaned_df[cleaned_df.type == larger_row_type]
            # new_larger = my_nonstandard_merge(smaller, larger, smaller_keys, move_cols)
            # cleaned_df.loc[new_larger.index, move_cols] = new_larger.loc[:, move_cols].values

            smaller_row_type = 'eval_trk_poly_fpath'
            larger_row_type = 'eval_act_poly_fpath'
            col_prefix = 'trk.poly'
            link_types(smaller_row_type, larger_row_type, col_prefix)
            # move_cols = [c for c in metric_cols if col_prefix in c]
            # smaller_keys = type_to_idkeys[smaller_row_type]
            # smaller = cleaned_df[cleaned_df.type == smaller_row_type]
            # larger = cleaned_df[cleaned_df.type == larger_row_type]
            # new_larger = my_nonstandard_merge(smaller, larger, smaller_keys, move_cols)
            # cleaned_df.loc[new_larger.index, move_cols] = new_larger.loc[:, move_cols].values

            # larger[larger['type'] == 'eval_act_poly_fpath']
            cleaned_df[cleaned_df['type'] == 'eval_trk_pxl_fpath'][metric_cols]
            cleaned_df[cleaned_df['type'] == 'eval_act_poly_fpath'][metric_cols].T

        reporter.orig_merged_df = cleaned_df

        # import xdev
        # xdev.search_replace
        # xdev.set_overlaps(a1['trk_poly_id'], a2['trk_poly_id'])

        reporter.orig_merged_df

        # hard coded values
        human_mapping = {
            'coi_mAP': 'Pixelwise mAP (classes of interest)',
            'coi_mAUC': 'Pixelwise mAUC (classes of interest)',
            'salient_AP': 'Pixelwise Salient AP',
            'salient_AUC': 'Pixelwise Salient AUC',
            'sc_macro_f1': 'IARPA SC mean F1',
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
            'eval_act+pxl': 'sc_macro_f1',
            'eval_trk+pxl': 'BAS_F1',
        }
        reporter.pixel_metric_lut = {
            'eval_act+pxl': 'coi_mAP',
            'eval_trk+pxl': 'salient_AP',
        }
        # reporter.actcfg_to_label = other['actcfg_to_label']
        # reporter.predcfg_to_label = other['predcfg_to_label']
        # reporter.human_mapping.update(reporter.actcfg_to_label)
        # reporter.human_mapping.update(reporter.predcfg_to_label)

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
    # predcfg_to_label = reporter.predcfg_to_label
    # actcfg_to_label = reporter.actcfg_to_label
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
        row['dataset_code'] = dataset_code
        if 'trk_model' in df.columns:
            row['num_trk_models'] = len(df['trk_model'].dropna().unique())
        else:
            row['num_trk_models'] = 0
        if 'act_model' in df.columns:
            row['num_act_models'] = len(df['act_model'].dropna().unique())
        else:
            row['num_act_models'] = 0
        row.update(
            ub.udict(df['type'].value_counts().to_dict()).map_keys(
                lambda x: 'num_' + x)
        )
        # row['num_experiments'] = len(group['expt'].unique())
        # row['num_models'] = len(group['model'].unique())
        # row['num_pxl_evals'] = type_evals.get('eval_pxl', 0)
        # row['num_bas_evals'] = type_evals.get('eval_trk', 0)
        # row['num_sc_evals'] = type_evals.get('eval_act', 0)
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


def load_extended_data(df, expt_dvc_dpath):
    """
    """
    from watch.mlops import smart_pipeline as frp
    rows = df.to_dict('records')
    big_rows = []
    errors = []

    import os
    WATCH_EVAL_LOAD_STRICT = os.environ.get('WATCH_EVAL_LOAD_STRICT', 1)
    for row in ub.ProgIter(rows, desc='load'):
        big_row = row.copy()
        fpath = row['raw']
        try:
            if row['type'] == 'eval_trk_pxl_fpath':
                pxl_info = frp.load_pxl_eval(fpath, expt_dvc_dpath, arg_prefix='trk.')
                big_row['info'] = pxl_info
            elif row['type'] == 'eval_act_pxl_fpath':
                pxl_info = frp.load_pxl_eval(fpath, expt_dvc_dpath, arg_prefix='act.')
                big_row['info'] = pxl_info
            elif row['type'] == 'eval_act_poly_fpath':
                sc_info = frp.load_eval_act_poly(fpath, expt_dvc_dpath)
                big_row['info'] = sc_info
            elif row['type'] == 'eval_trk_poly_fpath':
                bas_info = frp.load_eval_trk_poly(fpath, expt_dvc_dpath)
                big_row['info'] = bas_info
            else:
                raise KeyError('Unknown row type: ' + str(row['type']))
            big_rows.append(big_row)
        except Exception as ex:
            errors.append((ex, row))
            if WATCH_EVAL_LOAD_STRICT:
                raise

    import rich
    if len(errors):
        rich.print('[red] ' + repr(errors[0]))
        rich.print(f'[red] {len(errors)=}')
    else:
        print(f'{len(errors)=}')
    return big_rows


def clean_loaded_data(big_rows, expt_dvc_dpath):
    """
    Turn the nested "loaded" data into flat data for tabulation.
    Also combine eval types together into a single row per model / config.
    """

    def fix_none(v):
        return "None" if v is None else v

    # from watch.utils.util_param_grid import dotdict_to_nested
    import kwcoco

    from watch.mlops import expt_manager
    state = expt_manager.ExperimentState(expt_dvc_dpath, '*')

    simple_rows = []
    if 0:
        big_row = big_rows[10]
        big_row = big_rows[-1]

    for big_row in ub.ProgIter(big_rows, desc='big rows'):
        # fpath = big_row['raw']
        row = ub.udict(big_row) - {'info'}
        info = big_row['info']
        param_types = info['param_types']
        params = ub.udict().union(*param_types.values())
        params = params.map_values(fix_none)

        ADD_CROPID_HACK = 1
        if ADD_CROPID_HACK:
            from watch.utils import util_path
            # special handling for adding tracking / cropping
            # params to the activity row. We should figure out a
            # way of making this more general in the future.
            if row['type'] == 'eval_act_poly_fpath':
                if row['test_act_dset'].startswith('crop'):
                    # Fixme dataset name ids need a rework
                    crop_id = row['test_act_dset'].split('_crop.kwcoco')[0]
                    # There needs to be a search step for the crop
                    # dataset, which is not ideal.
                    pats = state.patterns.copy()
                    pats['crop_id'] = crop_id
                    pats = ub.udict(pats).map_values(str)
                    pat = state.templates['crop_fpath'].format(**pats)
                    _found = util_path.coerce_patterned_paths(pat)
                    if _found:
                        assert len(_found) == 1, 'should not have dups here'
                        found = _found[0]
                        _crop_attrs = ub.udict(state._parse_pattern_attrs(state.templates['crop_fpath'], found))
                        _crop_attrs = _crop_attrs - row
                        print(f'_crop_attrs={_crop_attrs}')
                        row.update(_crop_attrs)
            if row['type'] == 'eval_trk_poly_fpath':
                ...

            # Some of the ids from the experiments state may not be build we
            # should do that.
            for k, vs in state.hashid_dependencies.items():
                if k not in row:
                    deps = row & vs
                    if not any(pd.isnull(_) for _ in deps.values()):
                        v = state._condense_cfg(deps, k)
                        row[k] = v

            if row['type'] == 'eval_act_poly_fpath':
                try:
                    if row['regions_id'].startswith('trk_poly_id'):
                        row['trk_poly_id'] = row['regions_id']
                except KeyError:
                    ...

        FIX_FOR_POSTLOAD_TRK_INFO = 1
        if FIX_FOR_POSTLOAD_TRK_INFO:
            extra_attrs = info['other'].get('extra_attrs', None)
            if extra_attrs is None:
                extra_attrs = {}
            extra = ub.udict(extra_attrs) - {k for k, v in row.items() if not pd.isnull(v)}
            row.update(extra)

        for k, v in params.items():
            if k.endswith('.channels'):
                k3 = k.replace('channels', 'has_teamfeat')
                request_sensorchan = kwcoco.SensorChanSpec.coerce(smart.shrink_channels(v))
                row[k3] = smart.is_teamfeat(request_sensorchan)

        row.update(params)
        # row.update(info['metrics'])

        # fit_params = param_type['fit']
        # pred_params = param_type['pred']
        # track_params = info['param_types'].get('track', {})

        # # Shrink and check the sensorchan spec
        # request_sensorchan = kwcoco.SensorChanSpec.coerce(
        #     frp.shrink_channels(fit_params['channels']))
        # fit_params['channels'] = request_sensorchan.spec
        # sensorchan = request_sensorchan

        # fit_params['sensorchan'] = sensorchan
        # row['has_teamfeat'] = _is_teamfeat(sensorchan)

        # fit_param_keys2 = list(fit_param_keys) + ['channels']
        # selected_fit_params = ub.dict_isect(fit_params, fit_param_keys2)
        # row.update(selected_fit_params)
        simple_rows.append(row)

    cleaned_df = pd.DataFrame(simple_rows)
    print(f'{cleaned_df.shape=}')
    # simple_df['sensorchan'].unique()
    # simple_df[simple_df['sensorchan'].isnull()]

    if 0:
        sensorchan_keys = [k for k in cleaned_df.keys() if 'sensorchan' in k]
        print(f'sensorchan_keys={sensorchan_keys}')
        chan_keys = [k for k in cleaned_df.keys() if 'channels' in k]
        print(f'chan_keys={chan_keys}')

    # for gkey, group in cleaned_df.groupby('trk_pxl_cfg'):
    #     pass
    return cleaned_df


def is_null(x):
    return (isinstance(x, float) and math.isnan(x)) or x is None or not bool(x)


def resolve_model_info(model_fpath):
    cacher = ub.Cacher('model_info_memo', depends=[str(model_fpath)], appname='watch/model_info_memo')
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


def my_nonstandard_merge(smaller, larger, smaller_keys, move_cols, mode=0):
    """
    We are copying specific columns from a single row in a smaller dataframe
    into multiple rows in a larger data frame.

    Args:
        smaller (pd.DataFrame): a data frame to copy from
        larger (pd.DataFrame): a data frame to copy into
        smaller_keys (List[str]): columns that specify a single row in
            `smaller` and groups of rows in `larger`
        move_cols (List[str]): the columns to move.

    Example:
        >>> from watch.mlops.expt_report import *  # NOQA
        >>> smaller = pd.DataFrame([
        >>>     {'k1': 1, 'k2': 1, 'm1': 2, 'm2': 2, 's': 2, 'u1': 9},
        >>>     {'k1': 3, 'k2': 3, 'm1': 4, 'm2': 3, 's': 2, 'u2': 8},
        >>>     {'k1': 5, 'k2': 5, 'm1': 6, 'm2': 5, 's': 2, 'u3': 7},
        >>> ])
        >>> larger = pd.DataFrame([
        >>>     {'k1': 1, 'k2': 1, 'm1': np.nan, 'u2': 1, 's': 3},
        >>>     {'k1': 1, 'k2': 1, 'm1': np.nan, 'u2': 2, 's': 3},
        >>>     {'k1': 3, 'k2': 3, 'm1': np.nan, 'u2': 3, 's': 3},
        >>>     {'k1': 3, 'k2': 3, 'm1': np.nan, 'u2': 4, 's': 3},
        >>>     {'k1': 5, 'k2': 5, 'm1': np.nan, 'u2': 5, 's': 3},
        >>>     {'k1': 5, 'k2': 5, 'm1': np.nan, 'u2': 6, 's': 3},
        >>> ])
        >>> smaller_keys = ['k1', 'k2']  # should be usable as an index
        >>> move_cols = ['m1', 'm2']  # columns to move
        >>> larger1 = my_nonstandard_merge(smaller, larger, smaller_keys, move_cols, mode=0)
        >>> print(larger1)
        >>> larger2 = my_nonstandard_merge(smaller, larger, smaller_keys, move_cols, mode=1)
        >>> print(larger2)

    Ignore:
        import timerit
        ti = timerit.Timerit(100, bestof=10, verbose=2)
        for timer in ti.reset('time'):
            my_nonstandard_merge(smaller, larger, smaller_keys, move_cols, mode=1)
        for timer in ti.reset('time'):
            my_nonstandard_merge(smaller, larger, smaller_keys, move_cols, mode=0)
    """
    if mode == 0:
        smaller_suffix = None
        larger_suffix = '_y'
        smaller_subset = smaller[smaller_keys + move_cols]
        new_larger = pd.merge(smaller_subset, larger, on=smaller_keys,
                              suffixes=[smaller_suffix, larger_suffix],
                              validate='one_to_many')
        drop_cols = [k + larger_suffix for k in larger.columns.intersection(move_cols)]
        new_larger = new_larger.drop(drop_cols, axis=1)
        return new_larger
    else:
        smaller_lut = smaller.set_index(smaller_keys)
        new_larger = larger.copy()
        for smaller_key, group in dict(list(larger.groupby(smaller_keys))).items():
            small_match = smaller_lut.loc[smaller_key, move_cols]
            # Why is .values needed here? TODO: understand
            new_larger.loc[group.index, move_cols] = small_match.values
            # larger.loc[group.index, move_cols] = small_match
        return new_larger


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
