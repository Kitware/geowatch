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


def build_tables(config):
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

    tables = {}

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

        rows = []
        for fpath in fpaths:
            row = {}
            result = result_loader_fn(fpath)

            # TODO: better way to get config
            job_config_fpath = fpath.parent / 'job_config.json'
            config_ = json.loads(job_config_fpath.read_text())

            row['type'] = out_node.key
            row['region_id'] = result['json_info']['region_ids']
            metrics = smart_result_parser._add_prefix(node_name + '.metrics.', result['metrics'])
            params = smart_result_parser._add_prefix(node_name + '.params.', config_)
            row.update(metrics)
            row.update(params)
            rows.append(row)

        df = pd.DataFrame(rows)
        tables[node_name] = df

    eval_type_to_results = {}

    for key, df in tables.items():
        ddf = DotDictDataFrame(df)
        index = ddf[['type', 'region']]
        params = ddf[key + '.params']
        metrics = ddf[key + '.metrics']
        trunc_params, mappings = truncate_dataframe(params)
        results = {
            'index': index,
            'params': trunc_params,
            'metrics': metrics,
            'mappings': mappings,
        }
        eval_type_to_results[key] = results
    return eval_type_to_results


def truncate(x):
    return slugify_ext.smart_truncate(x, max_length=16, trunc_loc=0, hash_len=4, head='', tail='')


def truncate_dataframe(params):
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


def analyze_tables(eval_type_to_results):
    for key, results in eval_type_to_results.items():
        agg = Aggregator(results, type=key)
        # TODO : nicer replacement of long paths for params
        # metrics['sc_poly_eval.metrics.sc_macro_f1']


class Aggregator:
    def __init__(agg, results, type=None):
        agg.results = results
        agg.type = type
        agg.metrics = results['metrics']
        agg.params = results['params']

    def analyze(agg):
        from watch.utils import result_analysis
        metrics_of_interest = [
            'sc_poly_eval.metrics.sc_macro_f1',
        ]
        analysis = result_analysis.ResultAnalysis(
            agg.results, metrics=metrics_of_interest)
        analysis.results
        analysis.analysis()


def plot_examples():
    pass


def plot_tables():
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

    # df['sc_poly_eval.metrics.macro_f1_active']
    for metric in metrics_of_interset:
        node_id = metric.split('.')[0]
        metric_name = metric.split('.')[-1]
        df = tables[node_id]
        plt.figure()
        ax = sns.boxplot(data=df, x='region_id', y=metric)
        ax.set_ylabel(metric_name)
        ax.set_title(node_id + ' ' + metric_name)


def main(cmdline=True, **kwargs):
    """
    Ignore:
        >>> from watch.mlops.aggregate import *  # NOQA
        >>> import watch
        >>> data_dvc_dpath = watch.find_dvc_dpath(tags='phase2_data', hardware='auto')
        >>> expt_dvc_dpath = watch.find_dvc_dpath(tags='phase2_expt', hardware='auto')
        >>> cmdline = 0
        >>> kwargs = {
        >>>     # 'root_dpath': expt_dvc_dpath / '_testpipe',
        >>>     # 'pipeline': 'joint_bas_sc_nocrop',
        >>>     'root_dpath': expt_dvc_dpath / '_testsc',
        >>>     'pipeline': 'sc',
        >>> }
        >>> ## Execute
        >>> main(cmdline=cmdline, **kwargs)
    """
    config = AggregateEvluationConfig.legacy(cmdline=cmdline, data=kwargs)
    cacher = ub.Cacher(
        'table_cacher', appname='watch', depends=dict(config),
        enabled=1,
    )
    tables = cacher.tryload()
    if tables is None:
        eval_type_to_results = build_tables(config)
        cacher.save(eval_type_to_results)

    analyze_tables(eval_type_to_results)

    plot_tables()

    plot_examples()  # TODO


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
