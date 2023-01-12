"""
Loads results from an evaluation and aggregates them

Ignore:

    # Real data

    Given results from schedule_evaluation

    # SC
    DVC_DATA_DPATH=$(smartwatch_dvc --tags='phase2_data' --hardware=auto)
    DVC_EXPT_DPATH=$(smartwatch_dvc --tags='phase2_expt' --hardware=auto)
    python -m watch.mlops.aggregate \
        --pipeline=sc \
        --root_dpath="$DVC_EXPT_DPATH/_testsc"


    # BAS
    DVC_DATA_DPATH=$(smartwatch_dvc --tags='phase2_data' --hardware=auto)
    DVC_EXPT_DPATH=$(smartwatch_dvc --tags='phase2_expt' --hardware=auto)
    python -m watch.mlops.aggregate \
        --pipeline=bas \
        --root_dpath="$DVC_EXPT_DPATH/_testpipe"

"""
import scriptconfig as scfg
import kwarray
import math
import parse
import ubelt as ub
from watch.mlops import smart_pipeline
from watch.utils import util_pattern
from watch.mlops import smart_result_parser
import json
from watch.utils.util_param_grid import DotDictDataFrame
from watch.utils import slugify_ext
from typing import Dict, Any
import pandas as pd


class AggregateEvluationConfig(scfg.DataConfig):
    """
    Aggregates results from multiple DAG evaluations.
    """
    root_dpath = scfg.Value('auto', help='Where do dump all results. If "auto", uses <expt_dvc_dpath>/dag_runs')
    pipeline = scfg.Value('joint_bas_sc', help='the name of the pipeline to run')


def main(cmdline=True, **kwargs):
    """
    Ignore:
        >>> from watch.mlops.aggregate import *  # NOQA
        >>> import watch
        >>> data_dvc_dpath = watch.find_dvc_dpath(tags='phase2_data', hardware='auto')
        >>> expt_dvc_dpath = watch.find_dvc_dpath(tags='phase2_expt', hardware='auto')
        >>> cmdline = 0
        >>> kwargs = {
        >>>     'root_dpath': expt_dvc_dpath / '_testpipe',
        >>>     'pipeline': 'bas',
        >>>     # 'pipeline': 'joint_bas_sc_nocrop',
        >>>     # 'root_dpath': expt_dvc_dpath / '_testsc',
        >>>     #'pipeline': 'sc',
        >>> }
        >>> ## Execute
        >>> main(cmdline=cmdline, **kwargs)
    """
    config = AggregateEvluationConfig.legacy(cmdline=cmdline, data=kwargs)
    cacher = ub.Cacher(
        'table_cacher', appname='watch/mlops/aggregate', depends=dict(config),
        enabled=0,
    )
    tables = cacher.tryload()
    if tables is None:
        eval_type_to_results = build_tables(config)
        cacher.save(eval_type_to_results)

    eval_type_to_aggregator = build_aggregators(eval_type_to_results)

    agg = eval_type_to_aggregator.get('sc_poly_eval', None)
    if agg is not None:
        agg.analyze()

    agg = eval_type_to_aggregator.get('bas_poly_eval', None)
    if agg is not None:
        ...
        model_col = 'bas_poly_eval.params.bas_pxl.package_fpath'
        metric_col = 'bas_poly_eval.metrics.bas_faa_f1'
        table = agg.table
        agg.report_best(top_k=10)
        # table = table[~table[model_col].isnull()]
        for region_id, region_table in table.groupby('region_id'):
            print(f' --- region_id={region_id} --- ')
            region_table = region_table.sort_values(metric_col)
            for _, row in region_table.iterrows():
                score = row[metric_col]
                try:
                    model_name = ub.Path(row[model_col]).name
                except Exception:
                    continue
                    model_name = '?'
                print(score, model_name)

    plot_tables()

    plot_examples()  # TODO


def build_tables(config):
    import pandas as pd
    print('config = {}'.format(ub.repr2(dict(config), nl=1)))
    dag = smart_pipeline.make_smart_pipeline(config['pipeline'])
    dag.print_graphs()
    dag.configure(config=None, root_dpath=config['root_dpath'])

    # patterns = {
    #     'bas_pxl_id': '*',
    #     'bas_poly_id': '*',
    #     'bas_pxl_eval_id': '*',
    #     'bas_poly_eval_id': '*',
    #     'bas_poly_viz_id': '*',
    # }

    # Hard coded nodes of interest to gather. Should abstract later.
    node_eval_infos = [
        {'name': 'bas_pxl_eval', 'out_key': 'eval_pxl_fpath',
         'result_loader': smart_result_parser.load_pxl_eval},
        {'name': 'bas_poly_eval', 'out_key': 'eval_fpath',
         'result_loader': smart_result_parser.load_eval_trk_poly},
        {'name': 'sc_poly_eval', 'out_key': 'eval_fpath',
         'result_loader': smart_result_parser.load_eval_act_poly},
    ]

    eval_type_to_results = {}
    for node_eval_info in node_eval_infos:
        node_name = node_eval_info['name']
        out_key = node_eval_info['out_key']
        result_loader_fn = node_eval_info['result_loader']

        if node_name not in dag.nodes:
            continue

        node = dag.nodes[node_name]
        out_node = node.outputs[out_key]

        fpaths = out_node_matching_fpaths(out_node)

        # Pattern match
        # node.template_out_paths[out_node.name]

        cols = {
            'metrics': [],
            'index': [],
            'params': [],
            'param_types': [],
            'fpaths': [],
            # 'json_info': [],
        }
        for fpath in ub.ProgIter(fpaths, desc='loading'):
            result = result_loader_fn(fpath)

            # TODO: better way to get config
            job_config_fpath = fpath.parent / 'job_config.json'
            if job_config_fpath.exists():
                config_ = json.loads(job_config_fpath.read_text())
            else:
                config_ = {}

            region_ids = result['json_info']['region_ids']
            if True:
                # Munge data to get the region ids we expect
                import re
                region_pat = re.compile(r'[A-Z][A-Z]_R\d\d\d')
                region_ids = ','.join(list(region_pat.findall(region_ids)))

            index = {
                'type': out_node.key,
                'region_id': region_ids,
            }
            metrics = smart_result_parser._add_prefix(node_name + '.metrics.', result['metrics'])
            params = smart_result_parser._add_prefix(node_name + '.params.', config_)

            if config_:
                cols['metrics'].append(metrics)
                cols['params'].append(params)
                cols['index'].append(index)
                cols['param_types'].append(result['param_types'])
                cols['fpaths'].append(fpath)
                # cols['json_info'].append(result['json_info'])

        params = pd.DataFrame(cols['params'])
        trunc_params, mappings = truncate_dataframe_items(params)
        results = {
            'mappings': mappings,
            'fpaths': cols['fpaths'],
            'index': pd.DataFrame(cols['index']),
            'metrics': pd.DataFrame(cols['metrics']),
            'params': pd.DataFrame(cols['params']),
            'trunc_params': trunc_params,
            'param_types': cols['param_types'],
        }
        eval_type_to_results[node_name] = results

    return eval_type_to_results


def truncate_dataframe_items(params):
    """
    Truncates long, typically path-like items in a data frame.
    """
    def truncate(x):
        if not isinstance(x, str):
            return x
        return slugify_ext.smart_truncate(x, max_length=16, trunc_loc=0,
                                          hash_len=4, head='', tail='')
    mappings = {}
    if len(params):
        x = params.loc[0]
        trunc_cols = [k for k, v in x.items() if isinstance(v, str) and len(v) > 16]
        trunc_params = params.copy()
        trunc_params[trunc_cols] = trunc_params[trunc_cols].applymap(truncate)
        for c in trunc_params[trunc_cols]:
            v2 = pd.Categorical(params[c])
            params[c] = v2
            v1 = v2.map(truncate)
            mapping = list(zip(v1.categories, v2.categories))
            mappings[c] = mapping
    else:
        mapping = {}
        trunc_params = params
    return trunc_params, mappings


def build_aggregators(eval_type_to_results):
    eval_type_to_aggregator = {}
    for key, results in eval_type_to_results.items():
        agg = Aggregator(results, type=key)
        agg.build()
        eval_type_to_aggregator[key] = agg
        # TODO : nicer replacement of long paths for params
        # metrics['sc_poly_eval.metrics.sc_macro_f1']
    return eval_type_to_aggregator


class Aggregator:
    def __init__(agg, results, type=None):
        agg.results = results
        agg.type = type
        agg.metrics = results['metrics']
        agg.params = results['params']
        agg.index = results['index']

    def build(agg):
        if agg.type == 'bas_poly_eval':
            _primary_metrics_suffixes = [
                'bas_faa_f1'
            ]
        elif agg.type == 'sc_poly_eval':
            _primary_metrics_suffixes = [
                'sc_macro_f1', 'bas_faa_f1'
            ]
        elif agg.type == 'bas_pxl_eval':
            _primary_metrics_suffixes = [
                'bas_pxl_eval.metrics.salient_AP',
                'bas_pxl_eval.metrics.salient_APUC',
                'bas_pxl_eval.metrics.salient_AUC',
            ]
        elif agg.type == 'sc_pxl_eval':
            _primary_metrics_suffixes = [
                'bas_pxl_eval.metrics.coi_mAP',
                'bas_pxl_eval.metrics.coi_mAPUC',
                'bas_pxl_eval.metrics.coi_mAUC',
            ]
        else:
            raise NotImplementedError(agg.type)

        agg.primary_metric_cols = pandas_suffix_columns(
            agg.metrics, _primary_metrics_suffixes)

        _model_suffixes = ['package_fpath']
        agg.model_cols = pandas_suffix_columns(
            agg.params, _model_suffixes)

        _testdset_suffixes = ['test_dataset']
        agg.test_dset_cols = pandas_suffix_columns(
            agg.params, _testdset_suffixes)

        agg.table = pd.concat([agg.metrics, agg.index, agg.params], axis=1)

        effective_params, mappings, hashid_to_params = agg.build_effective_params()
        agg.hashid_to_params = ub.udict(hashid_to_params)
        agg.mappings = mappings
        agg.effective_params = effective_params

        agg.macro_key_to_regions = {}
        agg.region_to_tables = {}
        for region_id, idx_group in agg.index.groupby('region_id'):
            agg.region_to_tables[region_id] = {
                'metrics': agg.metrics.loc[idx_group.index],
                'params': agg.params.loc[idx_group.index],
                'index': agg.index.loc[idx_group.index],
                'effective_params': agg.effective_params.loc[idx_group.index],
            }

        agg.build_macro_averaged_comparable_tables()

    def analyze(agg):
        from watch.utils import result_analysis
        metrics_of_interest = [
            'sc_poly_eval.metrics.sc_macro_f1',
        ]
        analysis = result_analysis.ResultAnalysis(
            agg.results, metrics=metrics_of_interest)
        analysis.results
        analysis.analysis()

    def report_best(agg, top_k=3):
        import rich
        for region_id, group in agg.region_to_tables.items():
            metric_group = group['metrics']
            top_idxs = pandas_argmaxima(metric_group, agg.primary_metric_cols, k=top_k)
            top_idxs = top_idxs[::-1]

            top_metrics = metric_group.loc[top_idxs][agg.primary_metric_cols]
            top_metrics = top_metrics[agg.primary_metric_cols]
            top_indexes = group['index'].loc[top_idxs]
            # top_params = group['effective_params'].loc[top_idxs].drop(agg.test_dset_cols, axis=1)
            param_lut = agg.hashid_to_params.subdict(top_indexes['param_hashid'])
            # hashed_params, param_lut = pandas_hashed_rows(top_params, hashed_colname='param_hashid')
            summary_table = pd.concat([top_indexes, top_metrics], axis=1)

            rich.print('param_lut = {}'.format(ub.urepr(param_lut, nl=2)))
            rich.print(summary_table.to_string())

    def build_effective_params(agg):
        """
        Consolodate / cleanup / expand information
        """
        params = agg.params

        effective_params = params.copy()

        HACK_FIX_JUNK_PARAMS = True
        if HACK_FIX_JUNK_PARAMS:
            junk_suffixes = ['space_basale']
            junk_cols = pandas_suffix_columns(effective_params, junk_suffixes)
            effective_params = effective_params.drop(junk_cols, axis=1)

        model_cols = agg.model_cols
        test_dset_cols = agg.test_dset_cols

        mappings : Dict[str, Dict[Any, str]] = {}
        for colname in model_cols:
            raw_paths = params[colname].tolist()
            condensed = []
            for p in raw_paths:
                p = ub.Path(p).stem
                p = '.'.join(p.split('.')[0:1])
                condensed.append(p)
            effective_params[colname] = condensed
            col_mapping = ub.dzip(condensed, raw_paths)
            mappings[colname] = col_mapping

        for colname in test_dset_cols:
            raw_paths = params[colname].tolist()
            condensed = []
            for p in raw_paths:
                p = ub.Path(p).stem
                p = '.'.join(p.split('.')[0:1])
                condensed.append(p)
            effective_params[colname] = condensed
            col_mapping = ub.dzip(condensed, raw_paths)
            mappings[colname] = col_mapping

        # For each unique set of effective parameters compute a hashid
        param_cols = ub.oset(effective_params.columns).difference(agg.test_dset_cols)
        param_cols = list(param_cols - {'region_id', 'type'})
        new_hashids = pd.Series([None] * len(agg.index), index=agg.index.index)
        hashid_to_params = {}
        for param_vals, group in effective_params.groupby(param_cols, dropna=False):
            unique_params = ub.dzip(param_cols, param_vals)
            hashid = ub.hash_data(unique_params)[0:8]
            hashid_to_params[hashid] = unique_params
            new_hashids.loc[group.index] = hashid

        # Update the index with an effective parameter hashid
        agg.index['param_hashid'] = new_hashids

        return effective_params, mappings, hashid_to_params

    def find_macro_comparable(agg):
        """
        Search for groups that have the same parameters over multiple regions.
        """
        table = pd.concat([agg.index, agg.metrics, agg.effective_params], axis=1)

        # Macro aggregation over regions.
        macro_compatible = ub.ddict(list)
        for param_hashid, group in table.groupby('param_hashid'):
            test_regions = frozenset(group['region_id'].tolist())
            macro_compatible[test_regions].append(group)

        macro_compatible_num = ub.udict(macro_compatible).map_values(len)

        region_to_num_compatible = ub.ddict(lambda: 0)
        for region_id in ub.unique(ub.flatten(macro_compatible_num)):
            for group, num in macro_compatible_num.items():
                if region_id in group:
                    region_to_num_compatible[region_id] += num
        print('macro_compatible_num = {}'.format(ub.urepr(macro_compatible_num, nl=1)))
        print('region_to_num_compatible = {}'.format(ub.urepr(region_to_num_compatible, nl=1)))
        return macro_compatible

    def build_macro_averaged_comparable_tables(agg):

        macro_compatible = agg.find_macro_comparable()

        # Given a specific group of regions,
        # Get all measurements that can be averaged over the chosen regions
        # regions_of_interest = {'BR_R002', 'KR_R001', 'KR_R002', 'US_R007'}
        # regions_of_interest = {'KR_R001', 'KR_R002', 'BR_R002'}
        regions_of_interest = {'KR_R001', 'KR_R002'}
        comparable_groups = []
        for key in macro_compatible.keys():
            avail = (key & regions_of_interest)
            if avail == regions_of_interest:
                groups = macro_compatible[key]
                for group in groups:
                    flags = kwarray.isect_flags(group['region_id'], avail)
                    comparable_groups.append(group[flags])

        macro_key = f'macro_{len(regions_of_interest):02d}_{ub.hash_data(sorted(regions_of_interest))[0:6]}'

        sum_cols = [
            'bas_poly_eval.metrics.bas_tp',
            'bas_poly_eval.metrics.bas_fp',
            'bas_poly_eval.metrics.bas_fn',
            'bas_poly_eval.metrics.bas_ntrue',
            'bas_poly_eval.metrics.bas_npred',
        ]
        mean_cols = [
            'bas_poly_eval.metrics.bas_ppv',
            'bas_poly_eval.metrics.bas_tpr',
            'bas_poly_eval.metrics.bas_ffpa',
            'bas_poly_eval.metrics.bas_f1',
            'bas_poly_eval.metrics.bas_space_FAR',
            'bas_poly_eval.metrics.bas_time_FAR',
            'bas_poly_eval.metrics.bas_image_FAR',
            'bas_poly_eval.metrics.bas_faa_f1',
            'bas_poly_eval.metrics.sc_macro_f1',
            'bas_poly_eval.metrics.macro_f1_siteprep',
            'bas_poly_eval.metrics.macro_f1_active',
            'bas_poly_eval.metrics.sc_micro_f1',
        ]
        agg_id_cols = [
            'region_id',
        ] + agg.test_dset_cols

        if len(comparable_groups) > 0:
            group = comparable_groups[0]
            map_cols = [c for c in group.columns if 'metrics' in c and c.endswith(('AP', 'AUC', 'APUC'))]
            mean_cols.extend(map_cols)

        def macro_aggregate(group):
            other_cols = group.columns.difference(mean_cols).difference(sum_cols).difference(agg_id_cols)
            group[other_cols]

            for c in mean_cols:
                pass

            aggid_df = group[group.columns.intersection(agg_id_cols)]
            sum_df = group[group.columns.intersection(sum_cols)]
            mean_df = group[group.columns.intersection(mean_cols)]
            other_df = group[other_cols]

            if len(aggid_df) == 1:
                aggid_row = aggid_df.iloc[0]
            else:
                aggid_row = pd.Series({
                    k: f'macro_{len(v):02d}_{ub.hash_data(sorted(v.to_list()))[0:6]}'
                    for k, v in aggid_df.T.iterrows()
                })
            aggid_row['macro_size'] = len(aggid_df)

            is_safe_cols = {
                k: ub.allsame(vs, eq=nan_eq)
                for k, vs in other_df.T.iterrows()}
            warn_cols = {k: v for k, v in is_safe_cols.items() if not v}
            assert not warn_cols

            sum_row = sum_df.sum(axis=0)
            mean_row = mean_df.mean(axis=0, numeric_only=False)
            other_row = other_df.iloc[0]

            macro_row = pd.concat([aggid_row, mean_row, sum_row, other_row], axis=0)
            return macro_row

        # Macro average comparable groups
        macro_rows = []
        for group in comparable_groups:
            macro_row = macro_aggregate(group)
            macro_rows.append(macro_row)

        macro_df = pd.DataFrame(macro_rows)

        # main_metric = 'bas_poly_eval.metrics.bas_faa_f1'
        # main_metric = 'bas_poly_eval.metrics.bas_tp'
        # macro_df = macro_df.sort_values(main_metric, ascending=False)
        macro_results = {
            'index': macro_df[agg.index.columns],
            'effective_params': macro_df[agg.effective_params.columns],
            'params': macro_df[agg.effective_params.columns],  # fixme, use real
            'metrics': macro_df[agg.metrics.columns],
        }
        agg.macro_key_to_regions = {
            macro_key: regions_of_interest,
        }
        agg.region_to_tables[macro_key] = macro_results

    def macro_analysis(agg):
        from watch.utils import result_analysis
        results = []

        region_id = ub.peek(agg.macro_key_to_regions.keys())
        regions_of_interest = agg.macro_key_to_regions[region_id]
        tables = agg.region_to_tables[region_id]

        effective_params = tables['effective_params']
        metrics = tables['metrics']
        index = tables['index']

        table = pd.concat([index, effective_params, metrics], axis=1)
        table = table.fillna('None')

        main_metric = agg.primary_metric_cols[0]

        for idx, row in enumerate(table.to_dict('records')):
            row = ub.udict(row)
            row_metrics = row & set(metrics.keys())
            row_params = row & set(effective_params.keys())
            result = result_analysis.Result(str(idx), row_metrics, row_params)
            results.append(result)

        analysis = result_analysis.ResultAnalysis(
            results, metrics=[main_metric],
            metric_objectives={main_metric: 'max'}
        )
        analysis.analysis()
        analysis.report()

        model_cols = agg.model_cols
        import kwplot
        sns = kwplot.autosns()
        sns = kwplot.autosns()
        plt = kwplot.autoplt()
        kwplot.figure()
        x = 'bas_poly_eval.params.bas_poly.thresh'
        sns.lineplot(data=table, x=x, y=main_metric, hue=model_cols[0], style=model_cols[0])
        ax = plt.gca()
        ax.set_title(f'BAS Macro Average over {regions_of_interest}')

    def visualize_cases(agg):
        region_id = 'macro_02_19bfe3'
        regions_of_interest = agg.macro_key_to_regions.get(region_id, [region_id])
        param_hashid = 'ab43161b'

        to_inspect_idxs = []
        for region_id in regions_of_interest:
            tables = agg.region_to_tables[region_id]

            grouped_indexes = tables['index'].groupby('param_hashid')
            rows = grouped_indexes.get_group(param_hashid)
            assert len(rows) == 1
            to_inspect_idxs.append(rows.index[0])

        for index in to_inspect_idxs:
            pass
        index = to_inspect_idxs[0]

        agg.index.iloc[index]

        agg.results['param_types'][index]
        eval_fpath = agg.results['fpaths'][index]  # NOQA


def bas_poly_eval_confusion_analysis(eval_fpath):
    eval_fpath.parent / '.pred'
    bas_poly_dpath = list((eval_fpath.parent / '.pred/bas_poly').glob('*'))[0]
    pred_sites_fpath = bas_poly_dpath / 'sites_manifest.json'

    info = smart_result_parser.load_eval_trk_poly(eval_fpath)
    bas_row = info['json_info']['best_bas_rows']['data'][0]
    region_id = bas_row['region_id']
    rho = bas_row['rho']
    tau = bas_row['tau']
    dpath = (eval_fpath.parent / region_id / 'overall/bas')
    assign_fpaths1 = list(dpath.glob(f'detections_tau={tau}_rho={rho}_min_area*.csv'))
    assign_fpaths2 = list(dpath.glob(f'proposals_tau={tau}_rho={rho}_min_area*.csv'))
    assert len(assign_fpaths1) == 1
    assert len(assign_fpaths2) == 1
    assign_fpath1 = assign_fpaths1[0]
    assign_fpath2 = assign_fpaths2[0]

    import watch
    dvc_dpath = watch.find_dvc_dpath(tags='phase2_data', hardware='auto')
    true_site_dpath = dvc_dpath / 'annotations/site_models'
    # true_region_dpath = dvc_dpath / 'annotations/region_models'

    from watch.utils import util_gis

    ### Assign a confusion label to each truth and predicted annotation
    # Note to get the confusion, the metrics cant be run with the sequestered
    # flag.
    performer_id = 'kit'
    assign1 = pd.read_csv(assign_fpath1)
    assign2 = pd.read_csv(assign_fpath2)

    true_confusion_rows = []
    pred_confusion_rows = []
    site_to_status = {}
    from watch import heuristics
    for row in assign1.to_dict('records'):
        true_site_id = row['truth site'].split('_te_')[0]
        pred_site_ids = []
        truth_status = row['site type']
        site_to_status[true_site_id] = truth_status
        if isinstance(row['matched site models'], str):
            for name in row['matched site models'].split(','):
                pred_site_id = name.strip().split(f'_{performer_id}_')[0]
                pred_site_ids.append(pred_site_id)
        has_positive_match = len(pred_site_ids)
        true_cfsn = heuristics.iarpa_assign_truth_confusion(truth_status, has_positive_match)
        true_confusion_rows.append({
            'true_site_id': true_site_id,
            'pred_site_ids': pred_site_ids,
            'true_confusion': true_cfsn,
            'role': 'true_confusion',
        })

    for row in assign2.to_dict('records'):
        pred_site_id = row['site model'].split(f'_{performer_id}_')[0]
        true_site_ids = []
        truth_match_statuses = []
        if isinstance(row['matched truth sites'], str):
            for name in row['matched truth sites'].split(','):
                true_site_id = name.strip().split('_te_')[0]
                truth_match_statuses.append(site_to_status[true_site_id])
                true_site_ids.append(true_site_id)
        pred_cfsn = heuristics.iarpa_assign_pred_confusion(truth_match_statuses)
        pred_confusion_rows.append({
            'pred_site_id': pred_site_id,
            'true_site_ids': true_site_ids,
            'pred_confusion': pred_cfsn,
            'role': 'pred_confusion',
        })

    for true_row in true_confusion_rows:
        true_row['confusion_color'] = heuristics.IARPA_CONFUSION_COLORS.get(true_row['true_confusion'])
        true_row['role'] = 'true_confusion'

    for pred_row in pred_confusion_rows:
        pred_row['confusion_color'] = heuristics.IARPA_CONFUSION_COLORS.get(pred_row['pred_confusion'])
        pred_row['role'] = 'true_confusion'

    """
    True Confusion Spec
    -------------------

    "misc_info":  {
        "true_site_id": str,          # redundant site id information,
        "pred_site_ids": List[str],   # the matching predicted site ids,
        "true_confusion": str,        # the type of true confusion assigned by T&E
        "confusion_color": str,       # a named color coercable via kwimage.Color.coerce
        "role": "true_confusion",     # constant
    }

    Predicted Confusion Spec
    -------------------

    "misc_info":  {
        "pred_site_id": str,          # redundant site id information,
        "true_site_ids": List[str],   # the matching predicted site ids,
        "pred_confusion": str,        # the type of predicted confusion assigned by T&E
        "confusion_color": str,       # a named color coercable via kwimage.Color.coerce
        "role": "pred_confusion",     # constant
    }

    # The possible confusion codes and the corresponding confusion_color they
    # will be assigned is.
    IARPA_CONFUSION_COLORS = {}
    IARPA_CONFUSION_COLORS['gt_true_neg'] = 'darkgreen'  # no IARPA color for this, make one up.
    IARPA_CONFUSION_COLORS['gt_true_pos'] = 'lime'
    IARPA_CONFUSION_COLORS['gt_false_pos'] = 'red'
    IARPA_CONFUSION_COLORS['gt_false_neg'] = 'black'
    IARPA_CONFUSION_COLORS['gt_positive_unbounded'] = "darkviolet"
    IARPA_CONFUSION_COLORS['gt_ignore'] = "lightsalmon"
    IARPA_CONFUSION_COLORS['gt_seen'] = "gray"
    IARPA_CONFUSION_COLORS['sm_pos_match'] = "orange"
    IARPA_CONFUSION_COLORS['sm_partially_wrong'] = "aquamarine"
    IARPA_CONFUSION_COLORS['sm_completely_wrong'] = "magenta"
    """

    # confusion vectors -- unused
    if 0:
        pred_to_row = {r['pred_site_id']: r for r in pred_confusion_rows}
        confusion_vectors = []
        for true_row in true_confusion_rows:
            if len(true_row['pred_site_ids']):
                for pred_site_id in true_row['pred_site_ids']:
                    pred_row = pred_to_row[pred_site_id]
                    confusion_vectors.append({
                        'true_site_id': true_row['true_site_id'],
                        'true_confusion': true_row['true_confusion'],
                        'pred_site_id': pred_site_id,
                        'pred_confusion': pred_row['pred_confusion'],
                        'num_other_true': len(true_row['pred_site_ids']) - 1,
                        'num_other_pred': len(pred_row['true_site_ids']) - 1,
                    })
            else:
                confusion_vectors.append({
                    'true_site_id': true_row['true_site_id'],
                    'true_confusion': true_row['true_confusion'],
                    'pred_site_id': None,
                    'pred_confusion': None,
                    'num_other_true': 0,
                    'num_other_pred': 0,
                })

        for pred_row in pred_confusion_rows:
            if not pred_row['true_site_ids']:
                confusion_vectors.append({
                    'pred_site_id': pred_row['pred_site_id'],
                    'pred_confusion': pred_row['pred_confusion'],
                    'true_site_id': None,
                    'true_confusion': None,
                    'num_other_true': 0,
                    'num_other_pred': 0,
                })
    # /confusion vectors -- unused

    # Add the confusion info as misc data in new site files and reproject them
    # onto the truth for visualization.
    pred_site_fpaths = list(util_gis.coerce_geojson_paths(pred_sites_fpath))
    # rm_files = list(true_region_dpath.glob(region_id + '*.geojson'))
    gt_files = list(true_site_dpath.glob(region_id + '*.geojson'))
    sm_files = pred_site_fpaths
    true_site_infos = list(util_gis.coerce_geojson_datas(gt_files, format='json'))
    pred_site_infos = list(util_gis.coerce_geojson_datas(sm_files, format='json'))

    id_to_true_data = {ub.Path(d['fpath']).stem: d for d in true_site_infos}
    id_to_pred_data = {ub.Path(d['fpath']).stem: d for d in pred_site_infos}

    for true_row in true_confusion_rows:
        info = id_to_true_data[true_row['true_site_id']]
        for feat in info['data']['features']:
            if 'misc_info' in feat['properties']:
                feat['properties']['misc_info'].update(true_row)
            else:
                feat['properties']['misc_info'] = true_row.copy()

    for pred_row in pred_confusion_rows:
        info = id_to_pred_data[pred_row['pred_site_id']]
        for feat in info['data']['features']:
            if 'misc_info' in feat['properties']:
                feat['properties']['misc_info'].update(pred_row)
            else:
                feat['properties']['misc_info'] = pred_row.copy()

    # Check misc info is populated correctly and add role to site model
    for pred_site_id, pred_site in id_to_pred_data.items():
        for feat in pred_site['data']['features']:
            props = feat['properties']

            import kwimage
            geom = kwimage.MultiPolygon.coerce(feat['geometry']).to_shapely()
            simple_geom = geom.simplify(0.0002)  # Hack, should do this properly in the tracker
            new_geom = kwimage.MultiPolygon.coerce(simple_geom).to_geojson()
            feat['geometry'] = new_geom
            misc_info = props['misc_info']
            print('misc_info = {}'.format(ub.urepr(misc_info, nl=1)))

    for true_site_id, true_site in id_to_true_data.items():
        for feat in true_site['data']['features']:
            props = feat['properties']
            assert 'misc_info' in props
            misc_info = props['misc_info']
            print('misc_info = {}'.format(ub.urepr(misc_info, nl=1)))

    cfsn_dpath = bas_poly_dpath / 'confusion_sites'
    true_cfsn_dpath = (cfsn_dpath / 'true').ensuredir()
    pred_cfsn_dpath = (cfsn_dpath / 'pred').ensuredir()

    # Dump confusion site models to disk
    for pred_site_id, pred_site in id_to_pred_data.items():
        fpath = pred_cfsn_dpath / (pred_site_id + '.geojson')
        text = json.dumps(pred_site['data'], indent='    ')
        fpath.write_text(text)

    for true_site_id, true_site in id_to_true_data.items():
        fpath = true_cfsn_dpath / (true_site_id + '.geojson')
        text = json.dumps(true_site['data'], indent='    ')
        fpath.write_text(text)

    # Project confusion site models onto kwcoco for visualization
    from watch.cli import project_annotations
    import kwcoco
    src_fpath = bas_poly_dpath / 'poly.kwcoco.json'
    dst_fpath = bas_poly_dpath / 'poly_toviz.kwcoco.json'
    src_dset = kwcoco.CocoDataset(src_fpath)
    dst_dset = src_dset.copy()
    dst_dset.fpath = dst_fpath
    cmdline = 0

    true_site_infos2 = list(util_gis.coerce_geojson_datas(
        id_to_true_data.values(), format='dataframe', allow_raw=True))
    pred_site_infos2 = list(util_gis.coerce_geojson_datas(
        id_to_pred_data.values(), format='dataframe', allow_raw=True))

    for info in pred_site_infos2:
        site_df = info['data']

    for info in true_site_infos2:
        site_df = info['data']
        project_annotations.validate_site_dataframe(site_df)

    dst_dset.clear_annotations()
    common_kwargs = ub.udict(
        clear_existing=False,
        src=dst_dset,
        dst='return',
        workers=0,
    )
    true_kwargs = common_kwargs | ub.udict(
        role='truth_confusion',
        # propogate_strategy=False,
        # propogate_strategy=False,
        site_models=true_site_infos2,
        # viz_dpath=(bas_poly_dpath / '_true_projection'),
    )
    kwargs = true_kwargs
    pred_kwargs = common_kwargs | ub.udict(
        role='pred_confusion',
        site_models=pred_site_infos2,
        # viz_dpath=(bas_poly_dpath / '_pred_projection'),
    )
    # I don't know why this isn't in-place. Maybe it is a scriptconfig thing?
    repr1 = str(dst_dset.annots())
    print(f'repr1={repr1}')
    dst_dset = project_annotations.main(cmdline=cmdline, **true_kwargs)
    repr2 = str(dst_dset.annots())
    print(f'repr1={repr1}')
    print(f'repr2={repr2}')
    pred_kwargs['src'] = dst_dset
    dst_dset = project_annotations.main(cmdline=cmdline, **pred_kwargs)
    repr3 = str(dst_dset.annots())
    print(f'repr1={repr1}')
    print(f'repr2={repr2}')
    print(f'repr3={repr3}')

    set(dst_dset.annots().lookup('role', None))

    dst_dset.annots().take([0, 1, 2])

    from watch.cli import coco_visualize_videos
    kwargs = dict(
        src=dst_dset,
        smart=True,
        role_order=['truth_confusion', 'pred_confusion'],
        resolution='5 GSD',
        workers=0,
    )
    coco_visualize_videos.main(cmdline=cmdline, **kwargs)


def plot_examples():
    pass


def plot_tables(agg):
    # from watch.mlops import smart_result_parser
    # for fpath in fpaths:
    #     ...
    #     result = smart_result_parser.load_eval_act_poly(fpath, None)
    #     print(result['metrics']['sc_macro_f1'])

    import kwplot
    sns = kwplot.autosns()
    plt = kwplot.autoplt()
    # metric_cols = [c for c in df.columns if 'metrics.' in c]

    metrics_of_interset = [
        'sc_poly_eval.metrics.macro_f1_siteprep',
        'sc_poly_eval.metrics.macro_f1_active',
        'sc_poly_eval.metrics.sc_macro_f1',
        'sc_poly_eval.metrics.sc_micro_f1',

        'bas_poly_eval.metrics.bas_faa_f1',

        'bas_pxl_eval.metrics.salient_AP',
        # 'sc_pxl_eval.metrics.coi_mAP',
    ]
    kwplot.close_figures()

    metric = 'sc_poly_eval.metrics.sc_macro_f1'

    # df['sc_poly_eval.metrics.macro_f1_active']
    for metric in metrics_of_interset:
        node_id = metric.split('.')[0]
        metric_name = metric.split('.')[-1]
        DotDictDataFrame(agg.metrics)[node_id]
        df = pd.concat([agg.metrics, agg.index, agg.params], axis=1)
        plt.figure()
        ax = sns.boxplot(data=df, x='region_id', y=metric)
        ax.set_ylabel(metric_name)
        ax.set_title(node_id + ' ' + metric_name)


# def node_matching_outputs(node):
#     from watch.utils import util_pattern
#     import ubelt as ub
#     # print(f'node.template_out_paths={node.template_out_paths}')
#     # out_key = eval_to_outkey[node.name]
#     found = {}
#     for out_key, out_template in node.template_out_paths.items():
#         out_template = node.template_out_paths[out_key]

#         # self._parse_pattern_attrs(self.templates[key], path)
#         pat = node.root_dpath / out_template.format(**patterns)
#         mpat = util_pattern.Pattern.coerce(pat)
#         fpaths = list(mpat.paths())
#         found[out_key] = fpaths

#     print(ub.map_vals(len, found))
#     # print('fpaths = {}'.format(ub.urepr(fpaths, nl=1)))


def out_node_matching_fpaths(out_node):
    out_template = out_node.template_value
    parser = parse.Parser(str(out_template))
    patterns = {n: '*' for n in parser.named_fields}
    pat = out_template.format(**patterns)
    mpat = util_pattern.Pattern.coerce(pat)
    fpaths = list(mpat.paths())
    return fpaths


def pandas_argmaxima(data, columns, k=1):
    """
    Finds the top K indexes for each column.

    Args:
        data : pandas data frame
        columns : columns to maximize
        k : number of top entries per column

    Returns:
        List: indexes into subset of data that are in the top k for any of the
            requested columns.

    Example:
        >>> from watch.mlops.aggregate import *  # NOQA
        >>> import numpy as np
        >>> data = pd.DataFrame({k: np.random.rand(10) for k in 'abcde'})
        >>> columns = ['b', 'd', 'e']
        >>> k = 1
        >>> top_indexes = pandas_argmaxima(data=data, columns=columns, k=k)
        >>> print(data.loc[top_indexes])
    """
    top_indexes = None
    for col in columns:
        ranked_data = data[col].sort_values(ascending=False)
        ranked_idxs = ranked_data.index
        if top_indexes is None:
            top_indexes = ranked_idxs[0:k]
        else:
            top_indexes = top_indexes.union(ranked_idxs[0:k], sort=False)
    return top_indexes


def pandas_suffix_columns(data, suffixes):
    """
    Return columns that end with this suffix
    """
    return [c for c in data.columns if any(c.endswith(s) for s in suffixes)]


def pandas_hashed_rows(data, hashed_colname='hashed'):
    # data = top_params
    hashid_to_row = {}
    hash_rows = []
    for row in data.to_dict('records'):
        hashid = ub.hash_data(row)[0:8]
        hashid_to_row[hashid] = row
        hash_rows.append(hashid)

    hashed_df = pd.DataFrame(hash_rows, columns=[hashed_colname], index=data.index)
    return hashed_df, hashid_to_row


def nan_to_None(x):
    if isinstance(x, float) and math.isnan(x):
        return None
    else:
        return x


def nan_eq(a, b):
    if isinstance(a, float) and isinstance(b, float) and math.isnan(a) and math.isnan(b):
        return True
    else:
        return a == b


if __name__ == '__main__':
    """
    CommandLine:
        python ~/code/watch/watch/mlops/aggregate_evaluation.py --help
    """
    main()
