#!/usr/bin/env python
r"""
Loads results from an evaluation, aggregates them, and reports text or visual
results.

This is the main entry point for the mlops.aggregate CLI. It contains the logic
to consolidate rows of results into macro averages and compute a parameter
hashid (param_hashid) for each row. It also contains the basic text report
logic (although maybe that should be moved out?). It relies on several other
files in this directory

* aggregate_loader.py - handles the loading of individual rows from mlops output

* aggregate_plots.py - handles plotting relationships between parameters and metrics

* smart_global_helper.py - quick and dirty project specific stuff that ideally wont
    get in the way of general use-cases but should eventually be factored out.

Ignore:

    # Real data

    Given results from schedule_evaluation

    # SC
    DVC_DATA_DPATH=$(geowatch_dvc --tags='phase2_data' --hardware=auto)
    DVC_EXPT_DPATH=$(geowatch_dvc --tags='phase2_expt' --hardware=auto)
    python -m geowatch.mlops.aggregate \
        --pipeline=sc \
        --root_dpath="$DVC_EXPT_DPATH/_testsc"


    # BAS
    DVC_DATA_DPATH=$(geowatch_dvc --tags='phase2_data' --hardware=auto)
    DVC_EXPT_DPATH=$(geowatch_dvc --tags='phase2_expt' --hardware=auto)
    python -m geowatch.mlops.aggregate \
        --pipeline=bas \
        --root_dpath="$DVC_EXPT_DPATH/_testpipe"

    # BAS
    DVC_DATA_DPATH=$(geowatch_dvc --tags='phase2_data' --hardware=auto)
    DVC_EXPT_DPATH=$(geowatch_dvc --tags='phase2_expt' --hardware=auto)
    python -m geowatch.mlops.aggregate \
        --pipeline=bas \
        --io_workers=0 \
        --target \
            "$DVC_EXPT_DPATH/_testpipe" \
            "$DVC_EXPT_DPATH/_timekernel_test_drop4" \
        --output_dpath=./my_aggregate \
        --export_tables=True

    # BAS
    python -m geowatch.mlops.aggregate \
        --target ./my_aggregate/*.csv.zip \
        --stdout_report=True --rois KR_R001,KR_R002

    python -m geowatch.mlops.aggregate \
        --target ./my_aggregate/bas_pxl_eval_2023-02-22T215702-5.csv.zip \
        --plot_params=True --rois KR_R001,KR_R002



TODO:
    - [ ] The package_fpath (i.e. model_cols) reporting does heuristics to
          shorten the path to the package, but we shouldn't do this. We should
          make a new column that indicates it is a shortened name for the
          model, otherwise it is confusing.

"""
import math
import ubelt as ub
from typing import Dict, Any
from scriptconfig import DataConfig, Value

try:
    from line_profiler import profile
except ImportError:
    profile = ub.identity


class AggregateLoader(DataConfig):
    """
    Base config that will be mixed in to the :class:`AggregateEvluationConfig`.
    This config just defines parts related to constructing the
    :class:`Aggregator` objects (i.e. loading the tables).
    """

    target = Value(None, help=ub.paragraph(
        '''
        The input to the aggregator, which can take several forms:
        (1) the root directory of an mlops evaluation,
        (2) one or more pre-aggregated files,
        '''), nargs='+', position=1)

    pipeline = Value('joint_bas_sc', help=ub.paragraph(
        '''
        The pipeline to run. This can be a name of an internally registered
        pipeline, or it can point to a function that defines a pipeline
        in a Python file. E.g. ``user_module.pipelines.custom_pipeline_func()``
        or ``$HOME/my_code/my_pipeline.py::make_my_pipeline("arg")``.
        '''))

    io_workers = Value('avail', help='number of processes to load results')

    eval_nodes = Value(None, help='eval nodes to look at')

    primary_metric_cols = Value('auto', help='Either auto, or a YAML list of metrics in order of importance used to sort the output')

    display_metric_cols = Value('auto', help='Either auto, or a YAML list of metrics in order for display')

    cache_resolved_results = Value(True, isflag=True, help=ub.paragraph(
        '''
        if True, avoid recomputing parameter resolution if possible.
        Set to False if the specific resolved parameter / result parsers have
        changed.
        '''))

    def __post_init__(self):
        from kwutil.util_yaml import Yaml
        self.eval_nodes = Yaml.coerce(self.eval_nodes)
        self.primary_metric_cols = Yaml.coerce(self.primary_metric_cols)
        self.display_metric_cols = Yaml.coerce(self.display_metric_cols)
        ####
        # Pre-corece patterned inputs for nicer reporting?
        inputs = self.target
        if inputs is not None:
            from ruamel.yaml.composer import ComposerError
            resolved = []

            def resolve_item(item):
                try:
                    loaded = Yaml.loads(item)
                except (ComposerError, TypeError):
                    loaded = item
                if ub.iterable(loaded):
                    yield from loaded
                else:
                    yield loaded
            if ub.iterable(inputs):
                for item in inputs:
                    resolved.extend(list(resolve_item(item)))
            else:
                resolved.extend(list(resolve_item(inputs)))
            self.target = resolved

    @profile
    def coerce_aggregators(config):
        from kwutil import util_path
        from geowatch.mlops.aggregate_loader import build_tables
        import pandas as pd
        input_targets = util_path.coerce_patterned_paths(config.target)
        eval_type_to_tables = ub.ddict(list)

        print('Coerce aggregators for pipeline:')
        from geowatch.mlops import pipeline_nodes
        dag = pipeline_nodes.coerce_pipeline(config.pipeline)
        dag.print_graphs()

        print(f'Found {len(input_targets)} input targets')
        for target in ub.ProgIter(input_targets, desc='loading targets', verbose=3):
            if target.is_dir():
                # Assume Pipeline Output dir
                root_dpath = target
                dag.configure(config=None, root_dpath=root_dpath)

                eval_nodes = config.eval_nodes
                io_workers = config.io_workers
                cache_resolved_results = config.cache_resolved_results
                eval_type_to_results = build_tables(
                    root_dpath, dag, io_workers, eval_nodes,
                    cache_resolved_results=cache_resolved_results)
                for node_type, results in eval_type_to_results.items():
                    table = pd.concat(list(results.values()), axis=1)
                    eval_type_to_tables[node_type].append(table)
            if target.is_file():
                # Assume CSV file
                table = pd.read_csv(target, low_memory=False)
                if len(table):
                    node_type = table['node'].iloc[0]
                    eval_type_to_tables[node_type].append(table)

        eval_type_to_aggregator = {}
        for eval_type, tables in eval_type_to_tables.items():
            table = tables[0] if len(tables) == 1 else pd.concat(tables).reset_index(drop=True)
            agg = Aggregator(table,
                             primary_metric_cols=config.primary_metric_cols,
                             display_metric_cols=config.display_metric_cols,
                             dag=dag)
            agg.build()
            eval_type_to_aggregator[eval_type] = agg
        return eval_type_to_aggregator


class AggregateEvluationConfig(AggregateLoader):
    """
    Aggregates results from multiple DAG evaluations.
    """
    __command__ = 'aggregate'
    __alias__ = ['mlops_aggregate']

    output_dpath = Value('./aggregate', help=ub.paragraph(
        '''
        The path where the aggregator can write results (e.g. tables / plots).
        '''))

    export_tables = Value(False, isflag=True, help='if True, aggregated tables will be written to the output directory')

    plot_params = Value(False, isflag=True, help=ub.paragraph(
        '''
        if True, param plots will be drawn. This can also be a YAML dictionary
        with items that give finder grained control over plotting.
        An example set if items might look like:
        ``{"enabled": 0, "stats_ranking": 0, "min_variations": 1,
        "params_of_interest": ["params.bas_poly.thresh",
        "resolved_params.bas_pxl.channels"]}``
        '''))

    stdout_report = Value(True, type=str, isflag=True, help=ub.paragraph(
        '''
        if True, print a report to stdout. This can also be a YAML dictionary.
        An example set if items might look like:
        ``{"top_k": 100, "per_group": 1, "macro_analysis": 0, "analyze": 1,
         "print_models": true, "reference_region": "final", "concise": 1,
         "show_csv": 0}``
        '''))

    resource_report = Value(False, isflag=True, help='if True report resource utilization')

    symlink_results = Value(False, isflag=True, help='if True make symlinks based on region and param hashids')

    rois = Value('auto', help='Comma separated regions of interest')

    inspect = Value(None, help='param hashid to look at')

    query = Value(None, type=str, help=ub.paragraph(
        '''
        a pandas query to restrict the rows of the table we consider.
        E.g. "df['param_hashid'] == 'blpiinmvwgng'"
        '''
    ))

    custom_query = Value(None, type=str, help=ub.paragraph(
        '''
        This is raw Python code executed after a query which can be used to
        create complex filters not directly supported by other arguments.
        The code must define a name "new_eval_type_to_aggregator", which should
        be a filtered version of eval_type_to_aggregator.
        Ideally we can determine common cases and codify them without this
        arbitrary code execution. Use only if necessary.
        This is highly experimental and may be removed.
        '''
    ))

    embed = Value(False, isflag=True, help='if True, embed into IPython. Prefer snapshot over embed.')

    snapshot = Value(False, isflag=True, help='if True, make a snapshot suitable for IPython or a notebook. (requires xdev)')

    def __post_init__(self):
        super().__post_init__()
        from kwutil.util_yaml import Yaml
        self.plot_params = Yaml.coerce(self.plot_params)
        if self.query is not None:
            self.query = ub.paragraph(self.query)
        if isinstance(self.plot_params, int):
            self.plot_params = {
                'enabled': bool(self.plot_params)
            }
        self.stdout_report = Yaml.coerce(self.stdout_report)

    @profile
    def coerce_aggregators(config):
        eval_type_to_aggregator = super().coerce_aggregators()
        output_dpath = ub.Path(config['output_dpath'])
        for agg in eval_type_to_aggregator.values():
            agg.output_dpath = output_dpath
        return eval_type_to_aggregator


def main(cmdline=True, **kwargs):
    """
    Aggregate entry point.

    Loads results for each evaluation node_type, constructs aggregator objects,
    and then executes user specified commands that could include filtering,
    macro-averaging, reporting, plotting, etc...

    Ignore:
        >>> from geowatch.mlops.aggregate import *  # NOQA
        >>> import geowatch
        >>> data_dvc_dpath = geowatch.find_dvc_dpath(tags='phase2_data', hardware='auto')
        >>> expt_dvc_dpath = geowatch.find_dvc_dpath(tags='phase2_expt', hardware='auto')
        >>> cmdline = 0
        >>> kwargs = {
        >>>     'target': [expt_dvc_dpath / '_testpipe', expt_dvc_dpath / '_timekernel_test_drop4'],
        >>>     'pipeline': 'bas',
        >>>     'io_workers': 10,
        >>> }

        config = AggregateEvluationConfig.cli(cmdline=cmdline, data=kwargs)
        agg_dpath = ub.Path(config['root_dpath']) / 'aggregate'
        from geowatch.mlops.aggregate_loader import build_tables
        eval_type_to_results = build_tables(config)
        eval_type_to_aggregator = build_aggregators(eval_type_to_results, agg_dpath)
        agg = ub.peek(eval_type_to_aggregator.values())
        agg = eval_type_to_aggregator.get('bas_poly_eval', None)
        agg = eval_type_to_aggregator.get('bas_pxl_eval', None)

        ## Execute

        cmdline = 0
        main(cmdline=cmdline, **kwargs)
    """
    config = AggregateEvluationConfig.cli(cmdline=cmdline, data=kwargs, strict=True)
    import rich
    rich.print('config = {}'.format(ub.urepr(config, nl=1)))
    run_aggregate(config)


def run_aggregate(config):
    import rich
    from kwutil.util_yaml import Yaml
    eval_type_to_aggregator = config.coerce_aggregators()
    orig_eval_type_to_aggregator = eval_type_to_aggregator  # NOQA

    if config.eval_nodes is not None:
        eval_type_to_aggregator = ub.udict(eval_type_to_aggregator) & config.eval_nodes

    output_dpath = ub.Path(config['output_dpath'])
    for agg in eval_type_to_aggregator.values():
        agg.output_dpath = output_dpath

    rois = config.rois
    # rois = {'KR_R001', 'KR_R002', 'BR_R002'}

    if config.query:
        print('Running query')
        new_eval_type_to_aggregator = {}
        for key, agg in eval_type_to_aggregator.items():
            new_agg = agg.filterto(query=config.query)
            new_eval_type_to_aggregator[key] = new_agg
            rich.print(f'Query {key} filtered to {len(new_agg)}/{len(agg)} rows')
        eval_type_to_aggregator = new_eval_type_to_aggregator

    if config.custom_query:
        vars_ = dict(globals()) | dict(locals())
        code = ub.codeblock(config.custom_query)
        print('Executing custom query')
        print(ub.highlight_code(code, backend='rich'))
        res = exec(code, vars_)
        new_eval_type_to_aggregator = vars_['new_eval_type_to_aggregator']
        print(f'new_eval_type_to_aggregator={new_eval_type_to_aggregator}')
        # new_eval_type_to_aggregator = {}
        # for key, agg in eval_type_to_aggregator.items():
        #     chosen_idxs = []
        #     for group_id, group in agg.table.groupby('resolved_params.heatmap_pred_fit.trainer.default_root_dir'):
        #         group['metrics.heatmap_eval.salient_AP'].argsort()
        #         keep_idxs = group['metrics.heatmap_eval.salient_AP'].sort_values()[-5:].index
        #         chosen_idxs.extend(keep_idxs)
        #     new_agg = agg.filterto(index=chosen_idxs)
        #     rich.print(f'Special filter {key} filtered to {len(new_agg)}/{len(agg)} rows')
        #     new_eval_type_to_aggregator[key] = new_agg
        eval_type_to_aggregator = new_eval_type_to_aggregator

    if config.embed or config.snapshot:
        # Sneaky way around linting filters, but also a more concise than
        # try/except, and perhaps we can generalize to people's favorite
        # shells?
        embedding_modpath = ub.modname_to_modpath('xdev')
        if embedding_modpath is None:
            print('missing embed module')
        if embedding_modpath is not None:

            print(f'eval_type_to_aggregator = {ub.urepr(eval_type_to_aggregator, nl=1)}')
            for node_type, agg in eval_type_to_aggregator.items():
                print(f'agg={agg}')

            embed_module = ub.import_module_from_name('xdev')
            if config.snapshot:
                embed_module.snapshot()
                # Exit after taking the snapshot
                import sys
                sys.exit(1)
            elif config.embed:
                embed_module.embed()
            else:
                raise AssertionError('unreachable')

    for eval_type, agg in eval_type_to_aggregator.items():
        print(f'agg={agg}')

    timestamp = ub.timestamp()
    if config.export_tables:
        import platform
        hostname = platform.node()
        for eval_type, agg in eval_type_to_aggregator.items():
            num_results = len(agg)
            if num_results > 0:
                agg.output_dpath.ensuredir()
                fname = f'{agg.node_type}_{hostname}_{num_results:05d}_{timestamp}.csv.zip'
                csv_fpath = agg.output_dpath / fname
                print(f'Exported tables to: {csv_fpath}')
                agg.table.to_csv(csv_fpath, index_label=False)

    if config.stdout_report:
        if config.stdout_report is not True:
            report_config = Yaml.coerce(config.stdout_report)
        else:
            report_config = {}
        print(f'report_config = {ub.urepr(report_config, nl=1)}')
        for eval_type, agg in eval_type_to_aggregator.items():
            if len(agg):

                if rois is not None:
                    agg.build_macro_tables(rois)

                reportkw = ub.compatible(report_config, agg.report_best)
                agg.report_best(**reportkw)
                if report_config.get('analyze', False):
                    agg.analyze()
                if report_config.get('macro_analysis', False):
                    agg.macro_analysis()

    if config.symlink_results:
        for eval_type, agg in eval_type_to_aggregator.items():
            if len(agg):
                agg.make_result_node_symlinks()

    if config.resource_report:
        for eval_type, agg in eval_type_to_aggregator.items():
            if len(agg):
                agg.report_resources()

    if config.plot_params['enabled']:
        for eval_type, agg in eval_type_to_aggregator.items():
            if len(agg):
                plot_config = ub.udict(config.plot_params) - {'enabled'}
                agg.plot_all(rois, plot_config)
                # TODO: have text reports in a separate group
                agg.dump_varied_parameter_report()

    if config.inspect:
        agg = eval_type_to_aggregator['bas_pxl_eval']
        for eval_type, agg in eval_type_to_aggregator.items():
            if len(agg):
                subagg = agg.filterto(param_hashids=config.inspect if ub.iterable(config.inspect) else [config.inspect])
                if len(subagg):
                    subagg.make_summary_analysis(config)
                    # from geowatch.mlops import confusor_analysis
                    # for region_id, group in subagg.index.groupby('region_id'):
                    #     group_agg = subagg.filterto(index=group.index)
                    #     # confusor_analysis.main(cmdline=0, )
    return eval_type_to_aggregator


class TopResultsReport:
    """
    Object to hold the result of :func:`Aggregator.report_best`.
    """

    def __init__(self, region_id_to_summary, top_param_lut):
        self.top_param_lut = top_param_lut
        self.region_id_to_summary = region_id_to_summary


class AggregatorAnalysisMixin:
    """
    Analysis methods for :class:`Aggregator`.
    """
    def macro_analysis(agg):
        import pandas as pd
        from geowatch.utils import result_analysis
        from geowatch.utils import util_pandas

        macro_keys = list(agg.macro_key_to_regions.keys())
        if len(macro_keys) == 0:
            raise Exception('Build a macro result first')

        # regions_of_interest = agg.macro_key_to_regions[agg.primary_macro_region]
        tables = util_pandas.DotDictDataFrame(agg.region_to_tables[agg.primary_macro_region])

        resolved_params = tables['resolved_params']
        metrics = tables['metrics']
        # index = tables['index']
        index = tables[['node', 'region_id']]
        table = pd.concat([index, resolved_params, metrics], axis=1)
        table = table.fillna('None')

        main_metric = agg.primary_metric_cols[0]
        table = table.applymap(lambda x: str(x) if isinstance(x, list) else x)

        results = []
        for idx, row in enumerate(table.to_dict('records')):
            row = ub.udict(row)
            row_metrics = row & set(metrics.keys())
            row_params = row & set(resolved_params.keys())
            result = result_analysis.Result(str(idx), row_params, row_metrics)
            results.append(result)

        analysis = result_analysis.ResultAnalysis(
            results, metrics=[main_metric],
            metric_objectives={main_metric: 'max'}
        )
        # self = analysis
        analysis.analysis()
        analysis.report()
        return analysis, table

    def varied_param_counts(agg, min_variations=2, dropna=False):
        from geowatch.utils import util_pandas
        params = util_pandas.DataFrame(agg.resolved_params)
        params = params.applymap(lambda x: str(x) if isinstance(x, list) else x)
        varied_counts = params.varied_value_counts(dropna=dropna, min_variations=min_variations)
        varied_counts = ub.udict(varied_counts).sorted_values(key=len)
        return varied_counts

    def dump_varied_parameter_report(agg):
        """
        Write the varied parameter report to disk
        """
        import kwutil
        import rich
        report = agg.varied_parameter_report()
        list(kwutil.Json.find_unserializable(report))
        fixed_report = kwutil.Json.ensure_serializable(report, verbose=3, normalize_containers=True)
        # kwutil.Json.dumps(fixed_report)
        try:
            yaml_text = kwutil.Yaml.dumps(fixed_report)
        except Exception:
            # not sure why ruamel.yaml will cause an error here
            yaml_text = kwutil.Yaml.dumps(fixed_report, backend='pyyaml')

        agg.output_dpath.ensuredir()
        report_fpath = agg.output_dpath / 'varied_param_report.yaml'
        rich.print(f'Write varied parameter report to: {report_fpath}')
        report_fpath.write_text(yaml_text)

    def varied_parameter_report(agg, concise=True,
                                concise_value_char_threshold=80):
        """
        Dump a machine and human readable varied parameter report.

        Args:
            concise (bool):
                if True, sacrifice row homogeneity for shorter encodings
        """
        from geowatch.utils import util_pandas
        concise_value_char_threshold = 80
        report = {}

        # varied_counts = util_pandas.DataFrame(agg.effective_params).varied_value_counts()
        # on_error='placeholder')

        # from geowatch.utils import result_analysis
        resolved_params = util_pandas.DataFrame(agg.resolved_params)
        varied_counts = resolved_params.varied_value_counts(on_error='placeholder')
        # varied_counts = agg.table.varied_value_counts(on_error='placeholder')
        param_summary = {}
        for key, value_counts in varied_counts.items():
            type_counts = ub.ddict(lambda: 0)
            for value, count in value_counts.items():
                type_counts[type(value).__name__] += count
            type_counts = ub.odict(type_counts)
            summary = {
                'num_variations': len(value_counts),
            }
            if concise and len(type_counts) == 1:
                # Just indicate what the type of all values was
                summary['type'] = ub.peek(type_counts.keys())
            else:
                summary['type_counts'] = type_counts

            if concise:
                if len(ub.urepr(value_counts)) < concise_value_char_threshold:
                    # Give the varied values if they are short
                    summary['value_counts'] = value_counts
            else:
                summary['value_counts'] = value_counts
            param_summary[key] = summary

        param_summary = ub.udict(param_summary).sorted_values(lambda x: x['num_variations'])

        top_level_descendants = ub.ddict(set)
        for key in list(varied_counts.keys()):
            parts = key.split('.')
            top = parts[0]
            top_level_descendants[top].add(tuple(parts[1:]))

        column_summary = []
        for top, parts in top_level_descendants.items():
            partlens = list(map(len, parts))
            length_hist = ub.dict_hist(partlens)
            column_summary.append({
                'top': top,
                'num_subcolumns': len(parts),
                'subcolumn_depth_hist': length_hist,
            })

        report['param_summary'] = param_summary
        report['column_summary'] = column_summary
        return report

    def analyze(agg, metrics_of_interest=None):
        """
        Does a stats analysis on each varied parameter. Note this makes
        independence assumptions that may not hold in general.
        """
        from geowatch.utils import util_pandas
        from geowatch.utils import result_analysis
        resolved_params = util_pandas.DataFrame(agg.resolved_params)
        if metrics_of_interest is None:
            metrics_of_interest = agg.primary_metric_cols

        metrics = agg.metrics[metrics_of_interest]
        resolved_params = resolved_params.applymap(lambda x: str(x) if isinstance(x, list) else x)

        varied_counts = resolved_params.varied_value_counts(dropna=True)

        if 1:
            # Only look at reasonable groupings
            chosen_params = []
            for param, counts in varied_counts.items():
                if len(counts) > 1 and len(counts) < 10:
                    chosen_params.append(param)
                    ...
        else:
            chosen_params = None

        results = {
            'params': resolved_params,
            'metrics': metrics,
        }
        analysis = result_analysis.ResultAnalysis(results, params=chosen_params)
        # analysis.results
        analysis.analysis()

    def report_best(agg, top_k=100, shorten=True, per_group=None, verbose=1,
                    reference_region=None, print_models=False, concise=False,
                    show_csv=False, grouptop=None) -> TopResultsReport:
        """
        Report the top k pointwise results for each region / macro-region.

        Note:
            Results are chosen per-region independently. To get comparable
            results for a specific set of parameters choose a
            ``reference_region``, which could be a macro region.

        Args:
            top_k (int): number of top results for each region

            shorten (bool): if True, shorten the columns by removing
                non-ambiguous prefixes wrt to a known node eval_type.

            concise (bool):
                if True, remove certain columns that communicate context for a
                more concise report.

            reference_region (str | None):
                if specified filter the top results in all other regions to
                only be with respect to the top results in this region (or
                macro region). Can be set to the special key "final" to choose
                the last region, which is typically a macro region.

            show_csv (bool):
                also print as a CSV suitable for copy/paste into google sheets.

            grouptop (str | List[str]):
                if specified, these are a list of columns that a
                "suboptimized", which means that we group the table by these
                columns (e.g. the model column) and then only consider the
                "best" scoring results within these groups. This can help
                remove clutter if attempting to choose between a specific
                parameter.

        TODO:
            This might need to become a class that builds the TopResultsReport
            as it is getting somewhat complex.

        Returns:
            TopResultsReport:
                contains:
                region_id_to_summary (T1=Dict[str, DataFrame]):
                    mapping from region_id to top k results
                top_param_lut (T2=Dict[str, DataFrame]):
                    mapping from param hash to invocation details

        Example:
            >>> from geowatch.mlops.aggregate import *  # NOQA
            >>> agg = Aggregator.demo(rng=0, num=100).build()
            >>> agg.report_best(print_models=True, top_k=3)
            >>> agg.report_best(print_models=True, top_k=3, grouptop='special:model')
            >>> agg.report_best(print_models=True, top_k=3, grouptop='special:model', reference_region='region1')
        """
        import rich
        import pandas as pd
        import numpy as np
        from geowatch.utils import util_pandas

        if isinstance(per_group, float) and math.isinf(per_group):
            per_group = None
        if isinstance(top_k, float) and math.isinf(top_k):
            top_k = None

        primary_metric_objectives = [
            agg._metric_info[c]['objective'] for c in
            agg.primary_metric_cols
        ]

        if reference_region:
            # In every region group, restrict to only the top values for the
            # reference region. The idea is to make things comparable to the
            # macro scores.
            if reference_region == 'final':
                reference_region = region_id = list(agg.region_to_tables.keys())[-1]
            else:
                region_id = reference_region

            # Lookup the table corresponding to the reference region
            group = agg.region_to_tables[region_id]
            if len(group) == 0:
                region_to_len = ub.udict(agg.region_to_tables).map_values(len)
                print('region_to_len = {}'.format(ub.urepr(region_to_len, nl=1)))
                raise Exception(f'reference {region_id=} group is empty')

            # TODO: consolidate this logic with the similar per-region logic
            # In the next section: `for region_id, group in _agg.region_to_tables.items()`
            # of this function

            # Rank reference region param_hashids of the primary metrics
            if grouptop is not None:
                grouptop = _coerce_grouptop(grouptop, aliases={
                    'special:model': agg.model_cols
                })
                # Find the top k results per group.
                sublocs = []
                for subkey, subgroup in group.groupby(grouptop['params']):
                    locs = util_pandas.DataFrame.argextrema(
                        subgroup, agg.primary_metric_cols,
                        objective=primary_metric_objectives, k=grouptop['top_k'])
                    sublocs.extend(locs)
                group_to_rank = group.loc[sublocs]
                if verbose > 3:
                    print(f'Filtering by group to {len(sublocs)} / {len(group)} param hashids in reference region')
            else:
                group_to_rank = group

            try:
                top_locs = util_pandas.DataFrame.argextrema(
                    group_to_rank, agg.primary_metric_cols,
                    objective=primary_metric_objectives, k=top_k)
            except Exception:
                print("FIXME: Something when wrong when sorting the reference region")
                raise
            top_param_hashids = group.loc[top_locs]['param_hashid']

            if verbose > 3:
                print(f'Filtering to top {len(top_locs)} / {len(group)} param hashids in reference region')

            # Filter the agg object to consider only the top parameters
            _agg = agg.filterto(param_hashids=top_param_hashids)

            if region_id in agg.macro_key_to_regions:
                rois = agg.macro_key_to_regions[region_id]
                _agg.build_macro_tables(rois)
            reference_hashids = top_param_hashids
            reference_hashid_to_rank = {
                hashid: rank for rank, hashid in enumerate(reference_hashids)
            }

            if verbose > 3:
                # Print out information on how much was filtered per region
                for region_id in agg.region_to_tables.keys():
                    old_table = agg.region_to_tables[region_id]
                    new_table = _agg.region_to_tables[region_id]
                    print(f'Filter reduces {region_id} to {len(new_table)} / {len(old_table)}')

        else:
            # If no reference region is given, subsequent code will sort each
            # region independently.
            reference_hashids = None
            reference_hashid_to_rank = None
            _agg = agg

        metric_display_cols = list(ub.oset(
            _agg.primary_metric_cols + _agg.display_metric_cols))

        # For each region determine what information will be returned / shown
        region_id_to_summary = {}
        big_param_lut = {}
        region_id_to_ntotal = {}
        for region_id, group in _agg.region_to_tables.items():
            if len(group) == 0:
                continue
            index_cols = group.columns.intersection(_agg.index.columns)

            if reference_hashids is None:
                # Rank the rows for this region individually

                if grouptop is not None:
                    grouptop = _coerce_grouptop(grouptop, aliases={
                        'special:model': _agg.model_cols
                    })
                    # Find the top k results per group.
                    sublocs = []
                    for subkey, subgroup in group.groupby(grouptop['params']):
                        locs = util_pandas.DataFrame.argextrema(
                            subgroup, _agg.primary_metric_cols,
                            objective=primary_metric_objectives, k=grouptop['top_k'])
                        sublocs.extend(locs)
                    group_to_rank = group.loc[sublocs]
                else:
                    group_to_rank = group

                ranked_locs = util_pandas.DataFrame.argextrema(
                    group_to_rank, _agg.primary_metric_cols,
                    objective=primary_metric_objectives, k=top_k)
            else:
                # Rank the rows for this region by the reference rank
                # len(reference_hashid_to_rank)
                def make_rank_getter(d):  # no closure for embed debug
                    return lambda x: d.get(x, float('inf'))
                rank_getter = make_rank_getter(reference_hashid_to_rank)
                ranking = group['param_hashid'].apply(rank_getter)
                ranking = ranking[np.isfinite(ranking)]
                ranked_locs = ranking.sort_values().index

            # Note: this report will only display requested params, but there
            # might be more detailed variations of interest.
            ranked_group = group.loc[ranked_locs]
            param_lut = _agg.hashid_to_effective_params.subdict(ranked_group['param_hashid'])
            big_param_lut.update(param_lut)

            have_metric_display_cols = list(ub.oset(metric_display_cols) & list(ranked_group.columns))
            summary_cols = list(index_cols) + have_metric_display_cols
            summary_table = ranked_group[summary_cols]

            if shorten:
                summary_table = util_pandas.pandas_shorten_columns(summary_table)

            region_id_to_summary[region_id] = summary_table
            region_id_to_ntotal[region_id] = len(group)

        # In reverse order (so they correspond with last region table)
        # get a unique list of all params reported in the top k sorted
        # to be easy to reference with the topk tables.
        # Do initial sorting to the best config from the last table
        # is first. Sort by table first, and then score.
        param_hashid_order = ub.oset()
        for summary_table in reversed(region_id_to_summary.values()):
            param_hashids = summary_table['param_hashid'].values
            param_hashid_order.update(param_hashids)

        param_hashid_order = param_hashid_order[::-1]
        top_param_lut = ub.udict(big_param_lut).subdict(param_hashid_order)

        if verbose:
            from geowatch.utils.result_analysis import varied_value_counts

            PARAMTER_DISPLAY_MODE = 'auto'

            if PARAMTER_DISPLAY_MODE == 'auto':
                PARAMTER_DISPLAY_MODE = 'varied-requested'
                if len(top_param_lut) == 1:
                    PARAMTER_DISPLAY_MODE = 'full-requested'

            varied = varied_value_counts(top_param_lut.values(), dropna=True,
                                         min_variations=2, default=None)

            if PARAMTER_DISPLAY_MODE == 'full-requested':
                # Show full requested parameters for each hash
                rich.print('Parameter LUT: {}'.format(ub.urepr(top_param_lut, nl=2)))
            elif PARAMTER_DISPLAY_MODE == 'varied-requested':
                # Show all unvaried requested parameters and then the varied
                # requested parameters for each hash
                varied_param_names = set(varied.keys())
                top_varied_param_lut = {k: ub.udict(v) & varied_param_names
                                        for k, v in top_param_lut.items()}

                top_nonvaried_param_lut = {
                    k: ub.udict(v) - varied_param_names
                    for k, v in top_param_lut.items()}

                non_varied_params = ub.udict().union(*top_nonvaried_param_lut.values())
                rich.print('Varied Basis: = {}'.format(ub.urepr(varied, nl=2)))
                rich.print('Constant Params: {}'.format(ub.urepr(non_varied_params, nl=2)))
                rich.print('Varied Parameter LUT: {}'.format(ub.urepr(top_varied_param_lut, nl=2)))
            else:
                raise KeyError(PARAMTER_DISPLAY_MODE)

            if show_csv:
                ub.schedule_deprecation(
                    'geowatch', 'show_csv', 'param',
                    migration='This option should not be relied on.',
                    deprecate='0.18.3', error='1.0.0', remove='1.1.0')
                varied_keys = list(varied.keys())
                param_table = pd.DataFrame.from_dict(top_param_lut).T
                param_table.index.name = 'param_hashid'
                param_table = util_pandas.DataFrame(param_table)
                param_table = param_table.reorder(varied_keys, axis=1, missing='drop')
                print(ub.paragraph(
                    '''
                    Note, to paste into sheets, there will be an icon after you
                    paste (that looks like a clipboard) you can click and it
                    give you an option: Split text to columns
                    '''))
                print(param_table.to_csv(header=True, index=True))

            # Check for a common special case that we can make more concise output for
            only_one_top_item = all(len(t) == 1 for t in region_id_to_summary.values())
            only_one_source_item = all(n == 1 for n in region_id_to_ntotal.values())

            if only_one_top_item and len(region_id_to_summary):
                # In this case there is only a single top result per-region, so
                # we can show them all in the same table rather than having on
                # table per-region.
                justone = pd.concat(list(region_id_to_summary.values()), axis=0)
                submacro = ub.udict(_agg.macro_key_to_regions) & justone['region_id'].values

                if only_one_source_item:
                    # Not sure why I differentiated this case, but keeping
                    # code consistent
                    if submacro:
                        print('Macro Regions LUT: ' +  ub.urepr(submacro, nl=1))
                _justone = util_pandas.DataFrame(justone)
                if concise:
                    _justone = _justone.safe_drop(['node'], axis=1)
                    _justone = _justone.safe_drop(['fpath'], axis=1)
                rich.print(_justone.to_string())

                if not only_one_source_item:
                    # again, not sure why this is different
                    rich.print('agg.macro_key_to_regions = {}'.format(
                        ub.urepr(_agg.macro_key_to_regions, nl=1)))
            else:
                # In the more common case, we have multiple results per region,
                # so we display a table for each region separately.
                for region_id, summary_table in region_id_to_summary.items():
                    ntotal = region_id_to_ntotal[region_id]
                    rich.print('---')
                    ref_text = ''
                    if reference_region:
                        ref_text = f' wrt to reference region {reference_region}'

                    if region_id in _agg.macro_key_to_regions:
                        macro_regions = _agg.macro_key_to_regions[region_id]
                        rich.print(f'Top {len(summary_table)} / {ntotal} for {agg.node_type}, {region_id} = {macro_regions}{ref_text}')
                    else:
                        rich.print(f'Top {len(summary_table)} / {ntotal} for {agg.node_type}, {region_id}{ref_text}')

                    _summary_table = util_pandas.DataFrame(summary_table)
                    _summary_table_csv = _summary_table
                    if concise:
                        _summary_table = _summary_table.safe_drop(['node'], axis=1)
                        _summary_table = _summary_table.safe_drop(['fpath'], axis=1)
                        _summary_table_csv = _summary_table_csv.safe_drop(['fpath'], axis=1)

                    text = _summary_table.iloc[::-1].to_string(index=False)

                    RICH_LINKS = 1
                    if RICH_LINKS:
                        # Make the param hashids link to their directory
                        lut = summary_table.set_index('param_hashid')
                        if 'fpath' in lut.columns:
                            for param_hashid in _summary_table['param_hashid']:
                                try:
                                    fpath = ub.Path(lut.loc[param_hashid]['fpath'])
                                    if fpath.exists():
                                        text = text.replace(param_hashid, f'[link={fpath.parent}]{param_hashid}[/link]')
                                except TypeError:
                                    # can happen if lut has multiple results for the same hashid
                                    ...

                    rich.print(text)

                    if show_csv:
                        print(ub.paragraph(
                            '''
                            Note, to paste into sheets, there will be an icon
                            after you paste (that looks like a clipboard) you
                            can click and it give you an option: Split text to
                            columns
                        '''))
                        print(_summary_table_csv.iloc[::-1].to_csv(header=True, index=False))
                        ...
                    rich.print('')

        if print_models:
            import itertools as it
            from kwutil.util_yaml import Yaml

            # FIXME: handle macro regions?

            # The idea is that we get the list of models that did the best
            # according to each region. If they were ordered by a reference,
            # then this ordering will be wrt to the reference, otherwise it is
            # a ranking of the models that did the best on some region.

            tocombine_indexes = []
            for region_id, summary in region_id_to_summary.items():
                if not region_id.startswith('macro_'):
                    tocombine_indexes.append(list(summary.index))

            top_locs = list(ub.oset([x for x in ub.flatten(
                it.zip_longest(*tocombine_indexes)) if x is not None]))

            table = _agg.table.copy()
            table.loc[top_locs, 'rank'] = np.arange(len(top_locs))
            table = table.sort_values('rank')

            if len(agg.model_cols) == 0:
                print('No model columns are availble')
            else:
                model_col = agg.model_cols[0]

                # HACK: we want to group models that came from the same training
                # run so we report a more diverse set of models. We typically group
                # models together in a folder, but this is not robust, so we Only
                # do this grouping if the parent folder has a special name

                model_paths = [
                    ub.Path(p)
                    if not pd.isnull(p) else None
                    for p in table[model_col].tolist()]
                hacked_groups = [
                    p if p is not None and p.parent.name.startswith('Drop') else p
                    for p in model_paths]
                table['_hackgroup'] = hacked_groups

                chosen_locs = []
                for expt, group in table.groupby('_hackgroup'):
                    # group[model_col].tolist()
                    # flags = (group[_agg.primary_metric_cols] > 0).any(axis=1)
                    # group = group[flags]
                    group = group.sort_values('rank')
                    chosen_locs.extend(group.index[0:per_group])
                chosen_locs = table.loc[chosen_locs, 'rank'].sort_values().index

                top_k = 40
                chosen_locs = chosen_locs[:top_k]
                chosen_table = table.loc[chosen_locs]

                print('Model shortlist (lower rank is a better scoring model):')
                for chosen_row in ub.unique(chosen_table.to_dict('records'), key=lambda row: row[model_col]):
                    model_fpath = chosen_row[model_col]
                    param_hashid = chosen_row['param_hashid']
                    rank = chosen_row['rank']
                    rich.print(f'[blue]# Best Rank: [cyan] {rank} [blue]{param_hashid}')
                    print(Yaml.dumps([model_fpath]).strip())

        report = TopResultsReport(region_id_to_summary, top_param_lut)
        return report

    def resource_summary_table(agg):
        """
        Sumarize resource usage of the pipeline
        """
        import pandas as pd
        from kwutil import util_time
        table = agg.table.copy()
        resources = agg.resources

        duration_cols = [
            k for k in resources.keys()
            if k.endswith('.duration')
        ]
        for k in duration_cols:
            new_vals = table.loc[:, k].apply(lambda x: util_time.coerce_timedelta(x) if not pd.isnull(x) else x)
            table[k] = new_vals

        resource_summary = []
        for duration_key in duration_cols:
            a, b, c = duration_key.split('.')
            uuid_key = f'context.{b}.uuid'

            # Later stages in the pipeline may be based on the same earlier
            # result. We deduplicate to avoid double-counting resource usage
            chosen = []
            for _, group in table.groupby(uuid_key):
                try:
                    idx = group[duration_key].idxmax()
                except TypeError:
                    idx = 0
                chosen.append(idx)

            asec = util_time.timedelta(seconds=1)

            unique_rows = table.loc[chosen]
            row = {
                'node': b,
                'resource': c,
                'num': len(chosen),
            }
            row['total'] = unique_rows[duration_key].sum()
            row['mean'] = unique_rows[duration_key].mean()

            try:
                row['total'] = row['total'].round(asec)
            except AttributeError:
                ...
            try:
                row['mean'] = row['mean'].round(asec)
            except AttributeError:
                ...

            resource_summary.append(row)

            co2_key = f'{a}.{b}.co2_kg'
            if co2_key in table:
                unique_rows[co2_key]
                row = {
                    'node': b,
                    'resource': 'co2_kg',
                    'total': unique_rows[co2_key].sum(),
                    'mean': unique_rows[co2_key].mean(),
                    'num': len(chosen),
                }
                resource_summary.append(row)

            co2_key = f'{a}.{b}.kwh'
            if co2_key in table:
                unique_rows[co2_key]
                row = {
                    'node': b,
                    'resource': 'kwh',
                    'total': unique_rows[co2_key].sum(),
                    'mean': unique_rows[co2_key].mean(),
                    'num': len(chosen),
                }
                resource_summary.append(row)
        resource_summary_df = pd.DataFrame(resource_summary)
        return resource_summary_df

    def resource_summary_table_friendly(agg):
        resource_summary_df = agg.resource_summary_table()
        # TODO: nicer report
        import kwutil
        import pandas as pd
        def format_kwh(v):
            return str(round(v, 2)) + ' kWh'

        def format_co2(v):
            return str(round(v, 2)) + ' CO2Kg'

        def format_duration(v):
            v = kwutil.timedelta.coerce(v, nan_policy='return-nan', none_policy='return-nan')
            if pd.isnull(v):
                return v
            return v.format(unit={'value': 'auto', 'min_unit': 'hour', 'exclude_units': ['week', 'month', 'min']}, precision=2)
            # return v.format(unit='hour', precision=2)

        # Make more human friendly
        new_groups = []
        for resource, group in resource_summary_df.groupby('resource'):
            if resource == 'kwh':
                group['total'] = group['total'].apply(format_kwh)
                group['mean'] = group['mean'].apply(format_kwh)
            elif resource == 'duration':
                group['total'] = group['total'].apply(format_duration)
                group['mean'] = group['mean'].apply(format_duration)
            elif resource == 'co2_kg':
                group['total'] = group['total'].apply(format_co2)
                group['mean'] = group['mean'].apply(format_co2)
            else:
                raise NotImplementedError(resource)
            new_groups.append(group)
        friendly = pd.concat(new_groups).loc[resource_summary_df.index]

        # Better column names when we include units in the value
        mapper = {
            'duration': 'time',
            'kwh': 'electricity',
            'co2_kg': 'emissions',
        }
        friendly['resource'] = friendly['resource'].map(lambda x: mapper.get(x, x))
        return friendly

    def report_resources(agg):
        import rich
        resource_summary_df = agg.resource_summary_table()

        if 1:
            friendly = agg.resource_summary_table_friendly()
            rich.print(friendly.to_string())
            print(friendly.to_csv())
            print(friendly.to_latex(index=False, escape=False))

        rich.print(resource_summary_df.to_string())

    def make_summary_analysis(subagg, config):
        output_dpath = ub.Path(config['output_dpath']) / 'aggregate'
        agg_group_dpath = output_dpath / ('agg_summary_params2_v3')
        agg_group_dpath = agg_group_dpath.ensuredir()

        import rich
        rich.print(f'agg_group_dpath: [link={agg_group_dpath}]{agg_group_dpath}[/link]')

        # Given these set of A/B values, visualize each region
        for region_id, group in ub.ProgIter(list(subagg.index.groupby('region_id')), desc='Inspect Region'):
            group_agg = subagg.filterto(index=group.index)
            for id, row in group_agg.index.iterrows():
                ...
                inspect_node(subagg, id, row, group_agg, agg_group_dpath)

        rich.print(f'agg_group_dpath: [link={agg_group_dpath}]{agg_group_dpath}[/link]')

    def make_result_node_symlinks(agg):
        """
        Builds symlinks to results node paths based on region and param
        hashids.
        """
        assert agg.output_dpath is not None
        assert agg.node_type is not None
        base_dpath = (agg.output_dpath / 'param_links' / agg.node_type)
        byregion_dpath = (base_dpath / 'by_region').ensuredir()
        byparamid_dpath = (base_dpath / 'by_param_hashid').ensuredir()
        print('base_dpath = {}'.format(ub.urepr(base_dpath, nl=1)))
        print('byregion_dpath = {}'.format(ub.urepr(byregion_dpath, nl=1)))
        print('byparamid_dpath = {}'.format(ub.urepr(byparamid_dpath, nl=1)))
        grouped = agg.table.groupby(['param_hashid', 'region_id'])
        for group_vals, group in ub.ProgIter(list(grouped), desc='symlink nodes'):
            # handle the fact that there can be multiple runs of the same param hashid.
            # TODO: sort the groups in a consistent way if possible
            for group_idx, row in enumerate(group.to_dict('records'), start=1):
                region_id = row['region_id']
                param_hashid = row['param_hashid']
                version_id = f'version_{group_idx}'
                eval_fpath = ub.Path(row['fpath'])
                node_dpath = eval_fpath.parent
                node_byregion_dpath = (byregion_dpath / region_id / param_hashid / version_id)
                node_byparamid_dpath = (byparamid_dpath / param_hashid / region_id / version_id)
                node_byregion_dpath.parent.ensuredir()
                node_byparamid_dpath.parent.ensuredir()
                ub.symlink(real_path=node_dpath, link_path=node_byparamid_dpath, overwrite=1)
                ub.symlink(real_path=node_dpath, link_path=node_byregion_dpath, overwrite=1)

        import rich
        rich.print(f'Made Param Links: [link={base_dpath}]{base_dpath}[/link]')

    def build_plotter(agg, rois=None, plot_config=None):
        if rois is None:
            ...
        if plot_config is None:
            plot_config = {}
        from geowatch.mlops import aggregate_plots
        if isinstance(rois, str):
            # fixme: ensure rois are coerced before this point.
            rois = agg._coerce_rois(rois)
        # agg.macro_key_to_regions
        plotter = aggregate_plots.build_plotter(agg, rois, plot_config)
        return plotter

    def plot_all(agg, rois=None, plot_config=None):
        plotter = agg.build_plotter(rois, plot_config)
        plotter.plot_requested()

    def _wip_build_per_region_variance_tables(agg):
        from geowatch.utils import util_pandas
        table = util_pandas.DataFrame(agg.table)

        def stats_aggregate(subgroup, metric_keys):
            from geowatch.utils import util_dotdict
            metric_description = subgroup[metric_keys].describe()
            stats_row = {}
            for key, stats in metric_description.T.iterrows():
                stats = ub.udict(stats.to_dict())
                count = stats.pop('count')
                stats = stats - {'50%', '75%', '25%'}
                keystats = util_dotdict.DotDict(stats).add_prefix(key)
                stats_row.update(keystats)
                stats_row['count'] = count
            return stats_row

        group_rows = []
        metric_keys = ub.oset(list(agg.primary_metric_cols + agg.display_metric_cols))

        for _, subgroup in table.groupby(['region_id']):
            region_id = subgroup.iloc[0]['region_id']
            index_row = ub.udict({'region_id': region_id})
            stats_row = stats_aggregate(subgroup, metric_keys)
            stats_row = index_row | stats_row
            group_rows.append(stats_row)

        group_stats = util_pandas.DataFrame(group_rows)
        group_stats_show, col_mapping = util_pandas.pandas_shorten_columns(group_stats, min_length=2, return_mapping=True)
        # rich.print(group_stats_show)

        metrics_with_std = []
        for srow in group_stats.to_dict('records'):
            group_count = None
            for metric in metric_keys:
                mean = srow[metric + '.mean']
                std = srow[metric + '.std']
                count = srow['count']
                region_id = srow['region_id']
                if math.isnan(mean):
                    cell = '-'
                else:
                    if count == 1:
                        cell = f'{mean:0.2f}'
                    else:
                        cell = f'{mean:0.2f}±{std:0.2f}'
                if group_count is None:
                    # Only add count once
                    group_count = count
                    metrics_with_std.append({
                        'cell': count,
                        'metric': 'count',
                        'region_id': region_id,
                    })
                else:
                    assert group_count == count
                metrics_with_std.append({
                    'cell': cell,
                    'metric': metric,
                    'region_id': region_id,
                })
        longform = util_pandas.DataFrame(metrics_with_std)
        all_metric_table = longform.pivot(
            index=['region_id'], columns=['metric'], values='cell')
        all_metric_table = util_pandas.pandas_shorten_columns(all_metric_table, min_length=0)
        return all_metric_table


class _AggregatorDeprecatedMixin:
    """
    Old property names for backwards compatability
    """

    @property
    def params(self):
        ub.schedule_deprecation(
            modname='geowatch', name='params', type='property',
            migration='use requested_params instead',
            deprecate='0.15.0', error='1.0.0', remove='1.1.0',
        )
        return self.subtables['params']

    @property
    def hashid_to_params(self):
        ub.schedule_deprecation(
            modname='geowatch', name='hashid_to_params', type='property',
            migration='use hashid_to_effective_params instead',
            deprecate='0.18.4', error='1.0.0', remove='1.1.0',
        )
        return self.hashid_to_effective_params

    @property
    def type(self):
        ub.schedule_deprecation(
            modname='geowatch', name='type', type='property',
            migration='use node_type instead',
            deprecate='0.18.4', error='1.0.0', remove='1.1.0',
            stacklevel=2,
        )
        return self.node_type


class Aggregator(ub.NiceRepr, AggregatorAnalysisMixin, _AggregatorDeprecatedMixin):
    """
    Stores multiple data frames that separate metrics, parameters, and other
    information using consistent pandas indexing. Can be filtered to a
    comparable subsets of choice. Can also handle building macro averaged
    results over different "regions" with the same parameters.

    Set config based on your problem

    Example:
        >>> from geowatch.mlops.aggregate import *  # NOQA
        >>> agg = Aggregator.demo(rng=0, num=3).build()
        >>> print(f'agg.config = {ub.urepr(agg.config, nl=1)}')
        >>> print('--- The table of only metrics ---')
        >>> print(agg.metrics)
        >>> print('--- The table of resource utilization ---')
        >>> print(agg.resources)
        >>> print('--- The table of explicitly requested hyperparameters (to distinguish from defaults) ---')
        >>> print(agg.resolved_params)
        >>> print('--- The table of resolved hyperparameters ---')
        >>> print(agg.resolved_params)
        >>> print('--- The table with unique indexes for each experiment ---')
        >>> print(agg.index)
        >>> print('--- The entire joined table ---')
        >>> print(agg.table)
    """
    def __init__(agg, table, output_dpath=None,
                 node_type=None,
                 primary_metric_cols='auto',
                 display_metric_cols='auto',
                 dag=None):
        """
        Args:
            table (pandas.DataFrame):
                a table with a specific column structure (e.g. built by the
                aggregate_loader). See the demo for an example. Needs more docs
                here.

            output_dpath (None | PathLike):
                Path where output aggregate results should be written

            node_type (str | None):
                should not need to specify this anymore. This should just be
                the "node" column in the table.

            primary_metric_cols (List[str] | Literal['auto']):
                if "auto", then the "node_type" must be known by the global helpers.
                Otherwise list the metric columns in the priority that should
                be used to rank the rows.

            display_metric_cols (List[str] | Literal['auto']):
                if "auto", then the "node_type" must be known by the global helpers.
                Otherwise list the metric columns in the order they should be
                displayed (after the primary metrics).

            dag (geowatch.mlops.Pipeline):
                The pipeline that the evaluation table corresponds to.
                Only needed if introspection if necessary.
                If all "auto" params are specified, this should not be needed.
        """
        agg.output_dpath = output_dpath

        from geowatch.utils import util_pandas
        if not isinstance(table, util_pandas.DataFrame):
            table = util_pandas.DataFrame(table)

        agg.table = table
        agg.node_type = node_type
        agg.dag = dag
        agg.subtables = None
        agg.config = {
            'display_metric_cols': display_metric_cols,
            'primary_metric_cols': primary_metric_cols,
        }

        # This attribute will hold columns that store paths to model files
        agg.model_cols = None

        # This attribute will hold columns that store paths test datasets
        agg.test_dset_cols = None

        agg.hashid_to_effective_params = None
        agg.mappings = None
        agg.effective_params = None
        agg.macro_key_to_regions = None
        agg.region_to_tables = None
        agg.macro_compatible = None

    @classmethod
    def demo(cls, num=10, rng=None):
        """
        Construct a demo aggregator for testing.

        This gives an example of the very particular column format that is
        expected as input the the aggregator.

        Args:
            num (int): number of rows
            rng (int | None): random number generator / state

        Returns:
            Aggregator

        Example:
            >>> from geowatch.mlops.aggregate import *  # NOQA
            >>> agg = Aggregator.demo(rng=0, num=100)
            >>> print(agg.table)
            >>> agg.build()
            >>> agg.analyze()
            >>> agg.resource_summary_table()
            >>> agg.report_best()
        """
        from kwarray import distributions as dmod
        import pandas as pd
        import kwarray
        import uuid
        rng = kwarray.ensure_rng(rng)

        # An aggregator needs to correspond to a specific "evaluation node" in
        # some DAG.
        node = 'demo_node'

        # Define distributions to generate random rows for our tables.
        distributions = {}
        distributions['params'] = {
            f'{node}.param1': dmod.Categorical(['a', 'b', 'c']),
            f'{node}.param2': dmod.Categorical(['e', 'f', 'g']),
            f'{node}.param3': dmod.Distribution.random(rng=rng),
            f'{node}.test_dataset': dmod.Categorical([
                '/path/to/test_dataset1.kwcoco.zip',
                '/incompatable/paths/to/another/test_dataset2.kwcoco.zip',
                'relative_path/to/test_dataset2.kwcoco.zip',
                'test_dataset3.kwcoco.zip',
            ], rng=rng),
            f'{node}.package_fpath': dmod.Categorical([
                '/path/to/model1.pt',
                '/incompatable/paths/to/another/model2.pt',
                'relative_path/to/model3.pt',
                'model4.pt',
            ], rng=rng)
        }
        distributions['metrics'] = {
            f'{node}.metric1': dmod.Distribution.random(rng=rng),
            f'{node}.metric2': dmod.Distribution.random(rng=rng),
            f'{node}.metric3': dmod.Distribution.random(rng=rng),
        }
        distributions['resources'] = {
            f'{node}.duration': dmod.Uniform(1, 100, rng=rng),
            f'{node}.kwh': dmod.Uniform(1, 100, rng=rng),
            f'{node}.co2_kg': dmod.Uniform(1, 100, rng=rng),
        }

        columns = {}

        columns['node'] = [node] * num

        # We currently expect something called a "region_id" which corresponds
        # roughly to a test dataset, but abstracts away specific pre-processing
        # of that dataset. This is hard coded for SMART, but should be
        # generalized later.
        columns['region_id'] = ['region1'] * num

        # Can we make numpy use 128 bit types?
        # maxint = 340_282_366_920_938_463_463_374_607_431_768_211_455
        _pyrng = kwarray.ensure_rng(rng, api='python')
        def _seeded_uuid():
            # uuid.uuid4()
            _int = int.from_bytes(_pyrng.randbytes(16), byteorder='big')
            # _int = int.from_bytes(rng.randbytes(16))
            return uuid.UUID(int=_int, version=4)

        columns[f'context.{node}.uuid'] = [str(_seeded_uuid()) for _ in range(num)]
        columns[f'machine.{node}.host'] = ['pc1'] * num

        # Sample from the distributions to construct the demo rows
        for key1, distris in distributions.items():
            for key2, distri in distris.items():
                key = f'{key1}.{key2}'
                columns[key] = distri.sample(num)

        # Sometimes parameters are "auto", which means that they need to be
        # resolved to get the real value they used.
        for param_key in distributions['params'].keys():
            columns['resolved_params.' + param_key] = columns['params.' + param_key]

        # For parameters, they need an extra set of columns to indicate if they
        # were specified - or somehow inferred.
        for param_key in distributions['params'].keys():
            columns['specified.params.' + param_key] = [1] * num

        table = pd.DataFrame(columns)

        primary_metric_cols = [f'metrics.{node}.metric1']
        display_metric_cols = [f'metrics.{node}.metric3']
        agg = cls(table, primary_metric_cols=primary_metric_cols,
                  display_metric_cols=display_metric_cols)
        return agg

    def build(agg):
        """
        Inspect the aggregator's table and build supporting information

        Returns:
            Self: returns self for method chaining
        """
        from geowatch.utils import util_pandas
        agg.__dict__.update(**agg.config)

        if len(agg.table) == 0:
            agg.node_type = 'unknown-node_type-empty-table'
            return

        _table = util_pandas.DotDictDataFrame(agg.table)

        known_index_columns = ['node', 'region_id', 'param_hashid', 'fpath']
        agg.index_columns = list(ub.oset(known_index_columns) & set(agg.table.columns))

        subtables = {
            'index': agg.table[agg.index_columns].copy(),
        }
        _expected_top_level = [
            'metrics', 'params', 'specified', 'resolved_params',
            'resources', 'machine', 'context'
        ]
        subtables.update({
            c: _table.subframe(c, drop_prefix=False)
            for c in _expected_top_level
        })
        unknown_cols = agg.table.columns.difference(set(ub.flatten(([v.columns for v in subtables.values()]))))
        if len(unknown_cols):
            raise Exception(str(unknown_cols))
        agg.subtables = subtables

        if agg.node_type is None:
            agg.node_type = agg.table['node'].iloc[0]

        # Construct primary / display model columns and metric column info lut
        agg._build_metrics_column_preferences()

        # FIXME: HARD CODED from SMART
        # TODO: add mechanism where nodes can tag their parameters with these
        # categories like isa model suffix or isa test dataset.
        _model_suffixes = ['package_fpath']
        _testdset_suffixes = ['test_dataset', 'crop_src_fpath']

        agg.model_cols = util_pandas.pandas_suffix_columns(
            agg.requested_params, _model_suffixes)
        agg.test_dset_cols = util_pandas.pandas_suffix_columns(
            agg.requested_params, _testdset_suffixes)

        # def _ensure_prefixed_names(names, prefix):
        #     """
        #     If names are given without the appropriate prefix, then append it.
        #     """
        #     prefix_ = prefix + '.'
        #     new_names = []
        #     for c in names:
        #         if not c.startswith(prefix_):
        #             c = prefix_ + c
        #         new_names.append(c)
        #     return new_names
        # agg.display_metric_cols = _ensure_prefixed_names(agg.display_metric_cols, metrics_prefix)
        # agg.primary_metric_cols = _ensure_prefixed_names(agg.primary_metric_cols, metrics_prefix)
        # agg.model_cols = _ensure_prefixed_names(agg.model_cols, 'params')
        # agg.test_dset_cols = _ensure_prefixed_names(agg.test_dset_cols, 'params')

        # util_pandas.pandas_suffix_columns(agg.resolved_params, _testdset_suffixes)

        agg.build_effective_params()

        agg.macro_key_to_regions = {}
        agg.region_to_tables = {}
        # FIXME; region-id is a SMART thing, needs to be generalized
        for region_id, idx_group in agg.index.groupby('region_id'):
            agg.region_to_tables[region_id] = agg.table.loc[idx_group.index]
        agg.macro_compatible = agg.find_macro_comparable()
        return agg

    def _build_metrics_column_preferences(agg):
        """
        Builds a table indexed by column name for the metrics columns.

        The table attribute is ``_metric_info``. Each value should have keys:

            name
            suffix
            objective
            primary
            display
            aggregator

        """
        from geowatch.mlops.smart_global_helper import SMART_HELPER
        import warnings
        # TODO: need to be able to specify what the objective is for each
        # metric. Either minimize or maximize.
        metrics_prefix = f'metrics.{agg.node_type}'
        agg._metric_info = {}

        if agg.dag is not None:
            node = agg.dag.nodes[agg.node_type]
        else:
            node = None
        # Build a lookup table about how different metrics are interpreted
        try:
            user_metric_info = node._default_metrics2()
        except (AttributeError, NotImplementedError):
            print(f'User did not specify _default_metrics2 for {node}')
        else:
            for info in user_metric_info:
                suffix = info['suffix']
                name = f'{metrics_prefix}.{suffix}'
                agg._metric_info[name] = info.copy()
                agg._metric_info[name]['name'] = name

        if agg.primary_metric_cols == 'auto' or agg.display_metric_cols == 'auto':
            if agg._metric_info:
                # If the metrics info was specified, then dont use the old _default_metrics
                if agg.primary_metric_cols == 'auto':
                    agg.primary_metric_cols = [info['name'] for info in agg._metric_info.values() if info.get('primary', False)]
                    if len(agg.primary_metric_cols) == 0:
                        warnings.warn(f'No metrics for {node} were marked as primary, forcing at least one')
                        agg.primary_metric_cols = [ub.peek(agg._metric_info.values())['name']]
                if agg.display_metric_cols == 'auto':
                    agg.display_metric_cols = [info['name'] for info in agg._metric_info.values() if info.get('display', False)]
                    agg.display_metric_cols = list(ub.oset(agg.primary_metric_cols + agg.display_metric_cols))
            else:
                # TODO: deprecate the old _default_metrics stuff entirely
                try:
                    # TODO: deprecate SMART-stuff
                    _primary_metrics_suffixes, _display_metrics_suffixes = SMART_HELPER._default_metrics(agg)
                except Exception:
                    if hasattr(node, '_default_metrics'):
                        _primary_metrics_suffixes, _display_metrics_suffixes = node._default_metrics()
                        # should we prevent double prefixes?
                        _primary_metrics = [f'{metrics_prefix}.{s}' for s in _primary_metrics_suffixes]
                        _display_metrics = [f'{metrics_prefix}.{s}' for s in _display_metrics_suffixes]
                    else:
                        # fallback to something
                        _display_metrics = list(agg.table.search_columns('metrics'))[0:3]
                        _primary_metrics = _display_metrics[0:1]
                    if agg.primary_metric_cols == 'auto':
                        agg.primary_metric_cols = _primary_metrics
                    if agg.display_metric_cols == 'auto':
                        agg.display_metric_cols = _display_metrics

        # If specified primary metrics are not in the info table, use
        # assumptions
        for c in ub.flatten([agg.primary_metric_cols, agg.display_metric_cols]):
            info = agg._metric_info.get(c, {})
            if 'suffix' not in info:
                info['suffix'] = c.split('.')[-1]
            if 'name' not in info:
                info['name'] = c
            if c not in agg._metric_info:
                agg._metric_info[c] = info

        for info in agg._metric_info.values():
            if 'objective' not in info:
                # Assumption
                warnings.warn(f'Assuming {info} has objective=maximize')
                info['objective'] = 'maximize'

        for c in agg.primary_metric_cols:
            info = agg._metric_info[c]
            if not info.get('primary', False):
                info['primary'] = True

        for c in agg.display_metric_cols:
            info = agg._metric_info[c]
            if not info.get('display', False):
                info['display'] = True

        # Normalize key names
        for info in agg._metric_info.values():
            objective = info['objective']
            if objective in {'max'}:
                info['objective'] = 'maximize'
            if objective in {'min'}:
                info['objective'] = 'minimize'

        print(f'agg._metric_info = {ub.urepr(agg._metric_info, nl=2)}')

    def __nice__(self):
        return f'{self.node_type}, n={len(self)}'

    def __len__(self):
        return len(self.table)

    @property
    def primary_macro_region(agg):
        macro_keys = list(agg.macro_key_to_regions.keys())
        if len(macro_keys) == 0:
            region_keys = list(agg.region_to_tables.keys())
            assert len(region_keys) == 1
            key = region_keys[0]
        else:
            key = macro_keys[-1]
        return key

    def filterto(agg, index=None, models=None, param_hashids=None, query=None):
        """
        Build a new aggregator with a subset of rows from this one.

        Args:
            index (List | pd.Index): a subset of pandas row indexes to restrict to

            models (List[str]): list of effective model names (not paths) to restrict to.

            param_hashids (List[str]): list of parameter hashids to restrict to

            query(str):
                A custom query string currently parsed :func:`our_hack_query`,
                which can either be a DataFrame.query or a simple eval using
                ``df`` as the dataframe variable (i.e. ``agg.table``) that
                should resolve to flags or indexes to indicates which rows to
                take. See the example for demo usage.

        Returns:
            Aggregator: A new aggregator with a subset of data

        Example:
            >>> from geowatch.mlops.aggregate import *  # NOQA
            >>> agg = Aggregator.demo(rng=0, num=100)
            >>> agg.build()
            >>> subagg = agg.filterto(query='df["context.demo_node.uuid"].str.startswith("c")')
            >>> assert len(subagg) > 0, 'query should return something'
            >>> assert subagg.table['context.demo_node.uuid'].str.startswith('c').all()
            >>> assert not agg.table['context.demo_node.uuid'].str.startswith('c').all()
            >>> print(subagg.table['context.demo_node.uuid'])

        FIXME:
            On 2024-02-12 CI failed this test with. Not sure where
            non-determinisim came from.
            assert len(subagg) > 0, 'query should return something'
            AssertionError: query should return something

            Another instance on 2024-04-19. Job log is:
            https://gitlab.kitware.com/computer-vision/geowatch/-/jobs/9652752

            This is likely because of unseeded UUIDs, which should now be
            fixed.
        """
        import numpy as np
        import kwarray
        final_flags = 1

        if index is not None:
            flags = kwarray.isect_flags(agg.index.index, index)
            final_flags = np.logical_and(final_flags, flags)

        if param_hashids is not None:
            if not ub.iterable(param_hashids):
                param_hashids = [param_hashids]
            flags = kwarray.isect_flags(agg.index['param_hashid'].values, param_hashids)
            final_flags = np.logical_and(final_flags, flags)

        if models is not None:
            if not ub.iterable(models):
                models = [models]
            flags = kwarray.isect_flags(agg.effective_params[agg.model_cols[0]].values, models)
            final_flags = np.logical_and(final_flags, flags)

        def our_hacky_query(df, query):
            try:
                from pandas.core.computation.ops import UndefinedVariableError
            except Exception:
                from pandas.errors import UndefinedVariableError

            new_table = None
            if 'df[' in query:
                # HACK for more expressive queries
                try:
                    flags = eval(query)
                except Exception as ex:
                    print(f'warning, eval query unsuccessful: ex={ex}')
                else:
                    new_table = df[flags]
            else:
                try:
                    new_table = df.query(query)
                except UndefinedVariableError as ex:
                    print(f'warning, failed to query: ex={ex}')
            return new_table

        if query is not None:
            if isinstance(final_flags, int):
                table_so_far = agg.table
            else:
                table_so_far = agg.table[final_flags]
            if len(table_so_far) > 0:
                df = table_so_far
                new_table = our_hacky_query(df, query)
                if new_table is not None:
                    flags = kwarray.isect_flags(agg.index.index, new_table.index)
                    final_flags = np.logical_and(final_flags, flags)

        if isinstance(final_flags, int):
            new_agg = agg
        else:
            new_agg = agg.compress(final_flags)

        return new_agg

    def compress(agg, flags):
        new_table = agg.table[flags].copy()
        new_agg = Aggregator(new_table, node_type=agg.node_type,
                             dag=agg.dag, output_dpath=agg.output_dpath,
                             **agg.config)
        new_agg.build()
        return new_agg

    @property
    def metrics(self):
        return self.subtables['metrics']

    @property
    def resources(self):
        return self.subtables['resources']

    @property
    def index(self):
        return self.subtables['index']

    @property
    def requested_params(self):
        return self.subtables['params']

    @property
    def specified_params(self):
        return self.subtables['specified']

    @property
    def resolved_params(self):
        return self.subtables['resolved_params']

    @property
    def default_vantage_points(self):
        from geowatch.mlops.smart_global_helper import SMART_HELPER
        try:
            if self.dag is not None:
                node = self.dag.nodes[self.node_type]
                vantage_points = node.default_vantage_points
        except Exception:
            vantage_points = SMART_HELPER.default_vantage_points(self.node_type)
        return vantage_points

    def build_effective_params(self):
        """
        Consolodate / cleanup / expand information

        THIS COMPUTES THE ``param_hashid`` COLUMN!

        The "effective params" normalize the full set of given parameters so we
        can compute more consistent param_hashid. This is done by condensing
        paths (which is a debatable design decision) as well as mapping
        non-hashable data to strings.

        Populates:

            * ``self.hashid_to_effective_params`` : Dict[str, Dict[str, Any]]

            * ``self.mappings``

            * ``self.effective_params``

        """
        import pandas as pd
        from geowatch.utils import util_pandas
        from geowatch.mlops.smart_global_helper import SMART_HELPER
        requested_params = self.requested_params
        effective_params = requested_params.copy()

        HACK_FIX_JUNK_PARAMS = True
        if HACK_FIX_JUNK_PARAMS:
            # hacks to remove junk params that happen to be in our tables
            junk_suffixes = ['space_basale']
            junk_cols = util_pandas.pandas_suffix_columns(effective_params, junk_suffixes)
            effective_params = effective_params.drop(junk_cols, axis=1)

        model_cols = self.model_cols
        test_dset_cols = self.test_dset_cols

        mappings : Dict[str, Dict[Any, str]] = {}
        path_colnames = model_cols + test_dset_cols
        path_colnames = path_colnames + SMART_HELPER.EXTRA_PATH_COLUMNS
        existing_path_colnames = requested_params.columns.intersection(path_colnames)

        for colname in existing_path_colnames:
            colvals = requested_params[colname]
            condensed, mapper = util_pandas.pandas_condense_paths(colvals)
            mappings[colname] = mapper
            effective_params[colname] = condensed

        for colname in SMART_HELPER.EXTRA_HASHID_IGNORE_COLUMNS:
            effective_params[colname] = 'ignore'

        _specified = util_pandas.DotDictDataFrame(self.specified_params)
        _specified_params = _specified.subframe('specified')
        is_param_included = _specified_params > 0

        # For each unique set of effective parameters compute a hashid
        # TODO: better mechanism for user-specified ignore param columns
        hashid_ignore_columns = list(self.test_dset_cols)
        hashid_ignore_columns += SMART_HELPER.EXTRA_HASHID_IGNORE_COLUMNS

        param_cols = ub.oset(effective_params.columns).difference(hashid_ignore_columns)
        param_cols = list(param_cols - {'region_id', 'node'})

        try:
            list(effective_params.groupby(param_cols, dropna=False))
        except Exception:
            effective_params = effective_params.applymap(lambda x: str(x) if isinstance(x, list) else x)

        if 0:
            # dev helper to check which params are being varied. This can help
            # find ones that you would not expect to be varied, so they can
            # be manually excluded.
            from geowatch.utils.result_analysis import varied_value_counts
            varied_value_counts(effective_params, min_variations=2)

        if 0:
            # check behavior of groupby with None:
            df = pd.DataFrame([
                {'a': None, 'b': 4, 'c': 1},
                {'a': None, 'b': 1, 'c': 1},
                {'a': None, 'b': 1, 'c': 2},
                {'a': 1, 'b': 2, 'c': 2},
                {'a': 1, 'b': 3, 'c': 2},
            ], dtype=object)
            group_keys = ['a', 'c']
            grouped = df.groupby(group_keys, dropna=False)
            for group_vals, group in grouped:
                group_key1 = dict(zip(group_keys, group_vals))
                group_key2 = group.iloc[0][group_keys].to_dict()
                print('---')
                print(f'group_key1={group_key1}')
                print(f'group_key2={group_key2}')

        # Preallocate a series with the appropriate index
        hashids_v1 = pd.Series([None] * len(self.index), index=self.index.index)
        hashid_to_effective_params = {}

        if len(param_cols) > 0:
            param_groups = effective_params.groupby(param_cols, dropna=False)
        else:
            # fallback case, something is probably wrong if we are here
            param_groups = {None: effective_params}.items()

        for param_vals, group in param_groups:
            # Further subdivide the group so each row only computes its hash
            # with the parameters that were included in its row
            is_group_included = is_param_included.loc[group.index]

            # NOTE: groupby will replace None with NaN in the returned
            # iteration values
            # Work around this by choosing the first item from the group
            # itself.
            unique_params = group.iloc[0][param_cols]

            if len(param_cols) > 0:
                # TODO: Used the fixed groupby to avoid the need to ensure
                # param_flags is a list.
                param_subgroups = is_group_included.groupby(param_cols, dropna=False)
            else:
                # fallback case, something is probably wrong if we are here
                param_subgroups = {tuple(): is_group_included}.items()

            for param_flags, subgroup in param_subgroups:
                if not ub.iterable(param_flags):
                    param_flags = [param_flags]
                _flags = list(param_flags)
                valid_unique_params = unique_params[_flags].to_dict()

                hashid = hash_param(valid_unique_params, version=1)
                hashid_to_effective_params[hashid] = valid_unique_params
                hashids_v1.loc[subgroup.index] = hashid

        # Update the index with an effective parameter hashid
        self.index.loc[hashids_v1.index, 'param_hashid'] = hashids_v1
        self.table.loc[hashids_v1.index, 'param_hashid'] = hashids_v1
        self.hashid_to_effective_params = ub.udict(hashid_to_effective_params)
        self.mappings = mappings
        self.effective_params = effective_params

    def find_macro_comparable(agg, verbose=0):
        """
        Search for groups that have the same parameters over multiple regions.

        We determine if two columns have the same parameters by using the
        param_hashid, so the details of how that is computed (and which
        parameters are ignored when computing it - e.g. paths to datasets) has
        a big impact on the behavior of this function.

        SeeAlso:
            :meth:`Aggregator.build_effective_params` -
                the method that determines what parameters go into the
                param_hashid, and how to normalize them.
        """
        import pandas as pd
        table = pd.concat([agg.index, agg.metrics, agg.resolved_params], axis=1)
        # table[['param_hashid']]

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

        if verbose:
            macro_compatible_num = macro_compatible_num.sorted_values()

            macro_compatible_cumsum = {}

            if 0:
                for k1, val in macro_compatible_num.items():
                    if k1 not in macro_compatible_cumsum:
                        macro_compatible_cumsum[k1] = val
                    for k2 in macro_compatible_cumsum:
                        if k1 != k2 and k1.issuperset(k2):
                            macro_compatible_cumsum[k2] += val
                x = ub.udict(macro_compatible_cumsum).sorted_values()
                print('x = {}'.format(ub.urepr(x, nl=1)))

            print('macro_compatible_num = {}'.format(ub.urepr(macro_compatible_num, nl=1)))
            print('region_to_num_compatible = {}'.format(ub.urepr(region_to_num_compatible, nl=1)))
        return macro_compatible

    def gather_macro_compatable_groups(agg, regions_of_interest):
        """
        Given a set of ROIs, find groups in the comparable regions that contain
        all of the requested ROIs.
        """
        import kwarray
        comparable_groups = []
        macro_compatible = agg.macro_compatible
        regions_of_interest = set(regions_of_interest)
        for key in macro_compatible.keys():
            avail = (key & regions_of_interest)
            if avail == regions_of_interest:
                groups = macro_compatible[key]
                for group in groups:
                    flags = kwarray.isect_flags(group['region_id'], avail)
                    comparable_groups.append(group[flags])
        return comparable_groups

    def _coerce_rois(agg, rois=None):
        if rois is None:
            rois = 'max'
        if isinstance(rois, str):
            if rois == 'max' or rois == 'auto':
                regions_of_interest = ub.argmax(agg.macro_compatible, key=len)
            else:
                from kwutil.util_yaml import Yaml
                regions_of_interest = Yaml.coerce(rois)
                if isinstance(regions_of_interest, str):
                    regions_of_interest = [regions_of_interest]
        else:
            regions_of_interest = rois
        return regions_of_interest

    def build_macro_tables(agg, rois=None, **kwargs):
        """
        Builds one or more macro tables
        """
        if rois == 'auto':
            rois = [key for key in agg.macro_compatible.keys() if len(key) > 1]
            if len(rois) == 0:
                rois = [key for key in agg.macro_compatible.keys()]
        if isinstance(rois, list) and len(rois) and ub.iterable(rois[0]):
            # Asked for multiple groups of ROIS.
            print(f'Building multiple ({len(rois)}) macro tables')
            for single_rois in rois:
                agg.build_single_macro_table(single_rois, **kwargs)
        else:
            print(f'Building a single macro table: rois={rois!r}')
            agg.build_single_macro_table(rois, **kwargs)

    @profile
    def build_single_macro_table(agg, rois, average='mean'):
        """
        Builds a single macro table for a choice of regions.

        A **macro table** is a table of paramters and metrics macro averaged
        over multiple regions of interest.

        There is some hard-coded values in this function, but the core idea is
        general, and they just need to be parameterized correctly.

        Args:
            rois (List[str]): names of regions to average together
            average (str): mean or gmean

        Returns:
            DataFrame | None:
        """

        import pandas as pd
        import numpy as np
        from geowatch.utils.util_pandas import DotDictDataFrame
        # Given a specific group of regions,

        regions_of_interest = agg._coerce_rois(rois)
        macro_key = hash_regions(regions_of_interest)

        # Define how to aggregate each column
        # FIXME: handle in general.
        sum_cols = [c for c in agg.metrics.columns if c.endswith((
            '_tp', '_fp', '_fn', '_ntrue', '_npred'))]
        average_cols = [c for c in agg.metrics.columns if c.endswith((
            'mAP', 'APUC', 'mAPUC', 'mAUC', 'AP', 'AUC', 'f1', 'FAR', 'ppv',
            'tpr', 'ffpa', 'f1', 'f1_siteprep', 'f1_active'))]
        ignore_cols = [c for c in agg.metrics.columns if c.endswith(('rho', 'tau'))]
        sum_cols = agg.metrics.columns.intersection(sum_cols)

        start_time_cols = DotDictDataFrame.search_columns(agg.table, 'start_timestamp')
        stop_time_cols = DotDictDataFrame.search_columns(agg.table, 'stop_timestamp')

        # FIXME: SMART-specific
        ignore_cols = [c for c in agg.metrics.columns if c.endswith(('rho', 'tau'))]

        # NEW: this is a more general way to handle definition of aggregators
        # This code can be cleaned up considerably
        _hacked_col_types = {
            'mean': average_cols,
            'sum': sum_cols,
            'ignore': ignore_cols,
            'min': start_time_cols,
            'max': stop_time_cols,
        }
        for metric_info in agg._metric_info.values():
            column_name = metric_info['name']
            aggregator = metric_info.get('aggregator', 'mean')
            col_type = _hacked_col_types[aggregator]
            if column_name not in col_type:
                col_type.append(column_name)

        average_cols = agg.metrics.columns.intersection(average_cols)
        other_metric_cols = agg.metrics.columns.difference(sum_cols).difference(average_cols)
        other_metric_cols = other_metric_cols.difference(ignore_cols)
        if len(other_metric_cols):
            print(f'ignoring agg {other_metric_cols}')

        if average == 'mean':
            average = 'mean'
            aggregator = {c: 'mean' for c in average_cols}
        elif average == 'gmean':
            import scipy.stats.mstats
            gmean = scipy.stats.mstats.gmean
            average = gmean
            aggregator = {c: gmean for c in average_cols}
        else:
            raise KeyError(average)
        aggregator.update({c: 'sum' for c in sum_cols})
        aggregator.update({c: 'sum' for c in agg.resources.select_dtypes(np.number).columns})
        aggregator.update({c: 'min' for c in start_time_cols})
        aggregator.update({c: 'max' for c in stop_time_cols})

        # Gather groups that can be aggregated
        comparable_groups = agg.gather_macro_compatable_groups(regions_of_interest)
        if len(comparable_groups) == 0:
            import rich
            rich.print(ub.paragraph(
                f'''
                [yellow]WARNING: Failed to build macro results. No comparable groups
                for rois={rois}
                '''))
            DEBUG = 1
            if DEBUG:
                # Give the user a hint as to why...
                agg.find_macro_comparable(verbose=1)
        else:
            # Macro aggregaet comparable groups
            macro_rows = []
            for group in ub.ProgIter(comparable_groups, desc='macro aggregate'):
                if len(group) > 0:
                    macro_row = macro_aggregate(agg, group, aggregator, average=average)
                    macro_rows.append(macro_row)

            macro_table = pd.DataFrame(macro_rows).reset_index(drop=True)
            agg.region_to_tables.pop(macro_key, None)
            agg.macro_key_to_regions.pop(macro_key, None)
            agg.macro_key_to_regions[macro_key] = regions_of_interest
            agg.region_to_tables[macro_key] = macro_table
            return macro_table


def inspect_node(subagg, id, row, group_agg, agg_group_dpath):
    # FIXME: SMART specific
    from geowatch.utils import util_pandas
    # eval_fpath = group_agg.fpaths[id]
    eval_fpath = ub.Path(group_agg.table['fpath'].loc[id])
    param_hashid = row['param_hashid']
    region_id = row['region_id']
    dname = f'{region_id}_{param_hashid}'
    link_dpath = agg_group_dpath / dname
    real_dpath = eval_fpath.parent
    node_dpath = real_dpath
    ub.symlink(real_path=node_dpath, link_path=link_dpath)
    import kwimage
    from kwcoco.metrics.drawing import concice_si_display
    if 'poly_eval' in row['node']:
        region_viz_fpaths = list((node_dpath / 'region_viz_overall').glob('*_detailed.png'))
        assert len(region_viz_fpaths) == 1
        region_viz_fpath = region_viz_fpaths[0]
        viz_img = kwimage.imread(region_viz_fpath)
        scores_of_interest = util_pandas.pandas_shorten_columns(subagg.metrics).loc[id, ['bas_tp', 'bas_fp', 'bas_fn', 'bas_f1']]
        scores_of_interest = ub.udict(scores_of_interest.to_dict())
        text = ub.urepr(scores_of_interest.map_values(concice_si_display), nobr=1, si=1, compact=1)
        new_img = kwimage.draw_header_text(viz_img, param_hashid + '\n' + text)
        kwimage.imwrite(agg_group_dpath / f'summary_{region_id}_{param_hashid}.jpg', new_img)

        # FIXME
        import geowatch
        data_dvc_dpath = geowatch.find_dvc_dpath(tags='phase2_data', hardware='auto')
        # expt_dvc_dpath = geowatch.find_dvc_dpath(tags='phase2_expt', hardware='auto')
        true_region_dpath = data_dvc_dpath / 'annotations/drop6/region_models'
        true_site_dpath = data_dvc_dpath / 'annotations/drop6/site_models'

        confusion_fpaths = list((eval_fpath.parent / 'bas_summary_viz').glob('confusion_*.jpg'))
        if len(confusion_fpaths) == 0:
            from geowatch.mlops import confusor_analysis
            src_kwcoco = list((node_dpath / '.pred/bas_poly/').glob('*/poly.kwcoco.zip'))[0]
            pred_sites_dpath = list((node_dpath / '.pred/bas_poly/').glob('*/sites'))[0]
            confusor_config = confusor_analysis.ConfusorAnalysisConfig(
                bas_metric_dpath=(node_dpath / region_id / 'overall' / 'bas'),
                src_kwcoco=src_kwcoco,
                pred_sites=pred_sites_dpath,
                region_id=region_id,
                out_dpath=agg_group_dpath,
                true_site_dpath=true_site_dpath,
                true_region_dpath=true_region_dpath,
            )
            confusor_analysis.main(cmdline=0, **confusor_config)

        confusion_fpaths = list((eval_fpath.parent / 'bas_summary_viz').glob('confusion_*.jpg'))
        if len(confusion_fpaths):
            assert len(confusion_fpaths) == 1
            confusion_fpath = confusion_fpaths[0]
            im = kwimage.imread(confusion_fpath)
            scores_of_interest = util_pandas.pandas_shorten_columns(subagg.metrics).loc[id, ['bas_tp', 'bas_fp', 'bas_fn', 'bas_f1']]
            scores_of_interest = ub.udict(scores_of_interest.to_dict())
            text = ub.urepr(scores_of_interest.map_values(concice_si_display), nobr=1, si=1, compact=1)
            model_name = group_agg.effective_params[group_agg.model_cols[0]].loc[id]
            im = kwimage.draw_header_text(im, param_hashid + ' - ' + model_name + '\n' + text)
            kwimage.imwrite(agg_group_dpath / f'confusion_{region_id}_{param_hashid}.jpg', im)


@profile
def aggregate_param_cols(df, aggregator=None, hash_cols=None, allow_nonuniform=False):
    """
    Aggregates parameter columns. Specified hash_cols should be
    dataset-specific columns to be hashed. All other columns should
    be effectively the same, otherwise we will warn.

    Args:
        hash_cols (None | List[str]):
            columns whos values should be hashed together.

    Returns:
        pandas.Series: a single row representing the combined rows

    TODO:
        - [ ] optimize this
        - [ ] Rectify with ~/code/watch/geowatch/utils/util_pandas.py :: aggregate_columns

    Example:
        >>> from geowatch.mlops.aggregate import *  # NOQA
        >>> import pandas as pd
        >>> agg = Aggregator.demo(num=3)
        >>> agg.build()
        >>> df = pd.concat([agg.table] * 3).reset_index()
        >>> import scipy.stats.mstats
        >>> gmean = scipy.stats.mstats.gmean
        >>> aggregator = {'metrics.demo_node.metric1': gmean}
        >>> hash_cols = 'param_hashid'
        >>> allow_nonuniform = True
        >>> hash_cols = ['region_id'] + agg.test_dset_cols
        >>> agg_row = aggregate_param_cols(df, aggregator=aggregator, hash_cols=hash_cols, allow_nonuniform=allow_nonuniform)
        >>> print(agg_row)
    """
    import pandas as pd
    import numpy as np
    import rich
    agg_row = df.iloc[0]
    if len(df) == 1:
        return agg_row
    else:
        if hash_cols:
            df_comparable = df.drop(hash_cols, axis=1)
            df_hashable = df[hash_cols]
            hashed = {}
            for col, values in df_hashable.T.iterrows():
                try:
                    hashed[col] = hash_regions(values)
                except TypeError:
                    rich.print(ub.codeblock(
                        f'''
                        [red]ERROR[/red] when hashing column: {col=}
                        '''))
                    raise
        else:
            df_comparable = df
            hashed = {}

        if aggregator is not None:
            # Handle columns that can be aggregated
            aggregated = []
            for agg_op, cols in ub.group_items(aggregator.keys(), aggregator.values()).items():
                toagg = df_comparable[cols]
                # Drop non-numeric
                toagg = toagg.select_dtypes(np.number)
                aggregated.append(toagg.aggregate(agg_op))
            agg_parts = pd.concat(aggregated)
            df_comparable = df_comparable.drop(list(aggregator.keys()), axis=1)
        else:
            agg_parts = None

        is_safe_cols = {
            k: ub.allsame(vs, eq=nan_eq)
            for k, vs in df_comparable.T.iterrows()}

        nonuniform_cols = {k: v for k, v in is_safe_cols.items() if not v}
        if allow_nonuniform:
            agg_row = agg_row.drop(nonuniform_cols, axis=0)
        else:
            if nonuniform_cols:
                raise AssertionError(f'Values not identical: {nonuniform_cols}')
        if hashed or agg_parts is not None:
            agg_row = agg_row.copy()
            for c, v in hashed.items():
                agg_row[c] = v
            if agg_parts is not None:
                agg_row.update(agg_parts)
    return agg_row


@profile
def macro_aggregate(agg, group, aggregator, average='mean'):
    """
    Helper function
    """
    import pandas as pd
    blocklist = {'fpath'}
    hash_cols = ['region_id'] + agg.test_dset_cols

    table = agg.table.loc[group.index].drop(blocklist, axis=1)

    # Check if there is more than one run per-region per-param and
    # average them to keep the stats balanced.
    runs_per_region = table['region_id'].value_counts()
    has_multiple_param_runs = (runs_per_region > 1).any()

    allow_nonuniform = True

    if has_multiple_param_runs:

        # All aggregations are the mean when combining over the same region id
        sub_aggregator = {c: average for c in aggregator.keys()}
        sub_aggregator.update({c: average for c in agg.resources.columns})

        sub_hash_cols = agg.test_dset_cols
        subgroups = table.groupby('region_id')
        subrows = []
        try:
            for _, subgroup in subgroups:
                subrow = aggregate_param_cols(df=subgroup,
                                              aggregator=sub_aggregator,
                                              hash_cols=sub_hash_cols,
                                              allow_nonuniform=allow_nonuniform)
                subrows.append(subrow)
        except Exception:
            print(f'_={_}')
            print('len(subgroup) = {}'.format(ub.urepr(len(subgroup), nl=1)))
            raise
        # Now each region is in exactly one row.
        table = pd.DataFrame(subrows)

    macro_row = aggregate_param_cols(df=table, aggregator=aggregator,
                                     hash_cols=hash_cols,
                                     allow_nonuniform=allow_nonuniform)
    return macro_row


def hash_param(row, version=1):
    """
    Rule of thumb for probability of a collision:

    Ignore:
        base, length = 16, 8
        rule_of_thumb = np.sqrt(base ** length)
        rule_of_thumb = base ** (length // 2)
        print(f'rule_of_thumb={rule_of_thumb}')

        base, length = 26, 12
        rule_of_thumb = np.sqrt(base ** length)
        rule_of_thumb = base ** (length // 2)
        print(f'rule_of_thumb={rule_of_thumb}')
    """
    # TODO: something like multibase
    # https://github.com/multiformats/multibase
    if version == 1:
        param_hashid = ub.hash_data(row, base=26)[0:12]
    elif version == 0:
        param_hashid = ub.hash_data(row)[0:8]
    else:
        raise KeyError(version)
    return param_hashid


def hash_regions(rois):
    try:
        suffix = ub.hash_data(sorted(rois), base=16)[0:6]
    except Exception:
        print('Error---')
        print('rois = {}'.format(ub.urepr(rois, nl=1)))
        print('Error---')
        raise
    macro_key = f'macro_{len(rois):02d}_{suffix}'
    return macro_key


def nan_eq(a, b):
    if isinstance(a, float) and isinstance(b, float) and math.isnan(a) and math.isnan(b):
        return True
    else:
        return a == b


def _coerce_grouptop(grouptop, aliases=None):
    """
    Given a user input for "grouptop", coerce it into dictionary form. This is
    a helper for :func:`Aggregator.report_best`

    Args:
        grouptop (List[str] | str | dict):
            parameter name(s) or a dictionary of precise options.

        aliases (None | Dict[str, str | List[str]]):
            parameter aliases. User inputs that match keys will be expanded to
            the corresponding values.

    Returns:
        Dict[str, Any]: the dictionary form of "grouptop" with the form:
            .. code::
                {
                    'params': List[str],
                    'k': int,
                }

    Example:
        >>> from geowatch.mlops.aggregate import *  # NOQA
        >>> from geowatch.mlops.aggregate import _coerce_grouptop
        >>> grouptop = 'param1'
        >>> result1 = _coerce_grouptop(grouptop)
        >>> grouptop = ['param1', 'param2']
        >>> result2 = _coerce_grouptop(grouptop)
        >>> special_cols = {
        >>>     'special:1': 'a.special.alias',
        >>>     'special:2': ['two.special.alias', 'and.the.second.one'],
        >>> }
        >>> grouptop = {'params': ['special:1', 'param2', 'special:2'], 'top_k': 3}
        >>> result3 = _coerce_grouptop(grouptop, special_cols)
        >>> print(f'result1 = {ub.urepr(result1, nl=1)}')
        >>> print(f'result2 = {ub.urepr(result2, nl=1)}')
        >>> print(f'result3 = {ub.urepr(result3, nl=1)}')
        result1 = {
            'params': ['param1'],
            'top_k': 1,
        }
        result2 = {
            'params': ['param1', 'param2'],
            'top_k': 1,
        }
        result3 = {
            'params': ['a.special.alias', 'param2', 'two.special.alias', 'and.the.second.one'],
            'top_k': 3,
        }
    """
    new_grouptop = {
        'params': [],
        'top_k': 1,
    }
    if isinstance(grouptop, dict):
        # Given as a full dictionary
        new_grouptop.update(grouptop)
    else:
        # Given as a str | List[str]
        new_grouptop['params'] = grouptop

    # Now in a dictionary form, coerce params to always be a List[str]
    if isinstance(new_grouptop['params'], str):
        # Coerce params to always be a list
        new_grouptop['params'] = [new_grouptop['params']]

    # Resolve parameter aliases
    if aliases:
        resolved = []
        for param in new_grouptop["params"]:
            param = aliases.get(param, param)
            if isinstance(param, str):
                param = [param]
            resolved.extend(param)
        new_grouptop['params'] = resolved
    return new_grouptop


__cli__ = AggregateEvluationConfig
__cli__.main = main


if __name__ == '__main__':
    """
    CommandLine:
        python ~/code/watch/geowatch/mlops/aggregate_evaluation.py --help
    """
    main()
