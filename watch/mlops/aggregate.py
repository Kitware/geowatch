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
import parse
import ubelt as ub
from watch.mlops import smart_pipeline
from watch.utils import util_pattern
from watch.mlops import smart_result_parser
import json
from watch.utils.util_param_grid import DotDictDataFrame
from watch.utils import slugify_ext
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
        'table_cacher', appname='watch', depends=dict(config),
        enabled=0,
    )
    tables = cacher.tryload()
    if tables is None:
        eval_type_to_results = build_tables(config)
        cacher.save(eval_type_to_results)

    eval_type_to_aggregator = build_aggregators(eval_type_to_results)

    agg = eval_type_to_aggregator.get('sc_poly_eval', None)
    if agg is None:
        agg.analyze()

    agg = eval_type_to_aggregator.get('bas_poly_eval', None)
    if agg is not None:
        model_col = 'bas_poly_eval.params.bas_pxl.package_fpath'
        metric_col = 'bas_poly_eval.metrics.bas_faa_f1'
        table = agg.table
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
            index = {
                'type': out_node.key,
                'region_id': result['json_info']['region_ids'],
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

    @property
    def table(agg):
        table = pd.concat([agg.metrics, agg.index, agg.params], axis=1)
        return table

    def analyze(agg):
        from watch.utils import result_analysis
        metrics_of_interest = [
            'sc_poly_eval.metrics.sc_macro_f1',
        ]
        analysis = result_analysis.ResultAnalysis(
            agg.results, metrics=metrics_of_interest)
        analysis.results
        analysis.analysis()

    def build_effective_params(agg):
        """
        Consolodate information
        """
        params = agg.params

        effective_params = params.copy()

        model_cols = [c for c in params.columns if c.endswith('package_fpath')]
        test_dset_cols = [c for c in params.columns if c.endswith('test_dataset')]

        from typing import Dict, Any
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

        agg.effective_params = effective_params

    def build_macro_averaged_comparable_tables(agg):
        import kwplot
        sns = kwplot.autosns()
        table = pd.concat([agg.index, agg.metrics, agg.effective_params], axis=1)
        index_params = pd.concat([agg.index, agg.effective_params], axis=1)

        # Macro aggregation over regions.
        macro_compatible = ub.ddict(list)
        test_dset_cols = [c for c in agg.effective_params.columns if c.endswith('test_dataset')]
        param_cols = ub.oset(index_params.columns).difference(test_dset_cols)
        param_cols = list(param_cols - {'region_id', 'type'})
        for param_vals, group in table.groupby(param_cols, dropna=False):
            # unique_params = ub.dzip(param_cols, param_vals)
            test_regions = frozenset(group['region_id'].tolist())
            macro_compatible[test_regions].append(group)

        # all_common = frozenset.intersection(*macro_compatible.keys())

        # Get all measurements that can be averaged over the chosen regions
        import kwarray
        regions_of_interest = {'BR_R002', 'KR_R001', 'KR_R002', 'US_R007'}
        # regions_of_interest = {'KR_R001', 'KR_R002', 'BR_R002'}
        # regions_of_interest = {'KR_R001', 'KR_R002', 'BR_R002'}
        comparable = []
        for key in macro_compatible.keys():
            avail = (key & regions_of_interest)
            if avail == regions_of_interest:
                groups = macro_compatible[key]
                for group in groups:
                    flags = kwarray.isect_flags(group['region_id'], avail)
                    comparable.append(group[flags])

        def macro_aggregate(group):
            # if len(group) == 1:
            #     return group
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
                'bas_poly_eval.params.bas_pxl.test_dataset',
                'region_id',
            ]
            other_cols = group.columns.difference(mean_cols).difference(sum_cols).difference(agg_id_cols)
            group[other_cols]

            sum_df = group[sum_cols]
            mean_df = group[mean_cols]
            other_df = group[other_cols]
            aggid_df = group[agg_id_cols]

            if len(aggid_df) == 1:
                aggid_row = aggid_df.iloc[0]
            else:
                aggid_row = pd.Series({k: 'macro_' + ub.hash_data(sorted(v.to_list()))[0:6] for k, v in aggid_df.T.iterrows()})
            aggid_row['macro_size'] = len(aggid_df)

            other_df.apply(ub.allsame, axis=1)
            is_safe_cols = {k: ub.allsame(vs) for k, vs in other_df.T.iterrows()}
            warn_cols = {k: v for k, v in is_safe_cols.items() if not v}
            assert not warn_cols

            sum_row = sum_df.sum(axis=0)
            mean_row = mean_df.mean(axis=0)
            other_row = other_df.iloc[0]

            macro_row = pd.concat([aggid_row, mean_row, sum_row, other_row], axis=0)
            return macro_row

        # Macro average comparable groups
        macro_rows = pd.DataFrame([macro_aggregate(group) for group in comparable])
        macro_df = macro_rows
        main_metric = 'bas_poly_eval.metrics.bas_faa_f1'
        main_metric = 'bas_poly_eval.metrics.bas_tp'
        macro_df = macro_df.sort_values(main_metric, ascending=False)
        metrics_of_interest = [
            main_metric,
        ]
        from watch.utils import result_analysis
        results = []
        for idx, row in enumerate(macro_df.to_dict('records')):
            row = ub.udict(row)
            metrics = row & set(agg.metrics.keys())
            params = row & set(agg.params.keys())
            result = result_analysis.Result(str(idx), params, metrics)
            results.append(result)

        analysis = result_analysis.ResultAnalysis(
            results, metrics=metrics_of_interest,
            metric_objectives={main_metric: 'max'}
        )
        analysis.analysis()
        analysis.report()

        model_cols = [c for c in agg.params.columns if c.endswith('package_fpath')]
        test_dset_cols = [c for c in agg.params.columns if c.endswith('test_dataset')]

        sns = kwplot.autosns()
        plt = kwplot.autoplt()
        kwplot.figure()
        x = 'bas_poly_eval.params.bas_poly.thresh'
        sns.lineplot(data=macro_df, x=x, y=main_metric, hue=model_cols[0], style=model_cols[0])
        ax = plt.gca()
        ax.set_title(f'BAS Macro Average over {regions_of_interest}')


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


if __name__ == '__main__':
    """
    CommandLine:
        python ~/code/watch/watch/mlops/aggregate_evaluation.py --help
    """
    main()
