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

"""
import math
import ubelt as ub
from typing import Dict, Any
from scriptconfig import DataConfig, Value

try:
    from xdev import profile
except ImportError:
    profile = ub.identity


class AggregateLoader(DataConfig):
    """
    Base config that will be mixed in to the :class:`AggregateEvluationConfig`.
    This config just defines parts related to constructing the
    :class:`Aggregator` objects (i.e.  loading the tables).
    """

    target = Value(None, help=ub.paragraph(
        '''
        The input to the aggregator, which can take several forms:
        (1) the root directory of an mlops evaluation,
        (2) one or more pre-aggregated files,
        '''), nargs='+', position=1)

    pipeline = Value('joint_bas_sc', help='the name of the pipeline to run')

    io_workers = Value('avail', help='number of processes to load results')

    eval_nodes = Value(None, help='eval nodes to look at')

    cache_resolved_results = Value(True, isflag=True, help=ub.paragraph(
        '''
        if True, avoid recomputing parameter resolution if possible.
        Set to False if the specific resolved parameter / result parsers have
        changed.
        '''))

    def __post_init__(self):
        from kwutil.util_yaml import Yaml
        self.eval_nodes = Yaml.coerce(self.eval_nodes)
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
        print(f'Found {len(input_targets)} input targets')
        for target in ub.ProgIter(input_targets, desc='loading targets', verbose=3):
            if target.is_dir():
                # Assume Pipeline Output dir
                root_dpath = target
                pipeline = config.pipeline
                eval_nodes = config.eval_nodes
                io_workers = config.io_workers
                cache_resolved_results = config.cache_resolved_results
                eval_type_to_results = build_tables(
                    root_dpath, pipeline, io_workers, eval_nodes,
                    cache_resolved_results=cache_resolved_results)
                for type, results in eval_type_to_results.items():
                    # print('GOT RESULTS')
                    # print(results['resolved_params']['resolved_params.sc_poly.smoothing'])
                    table = pd.concat(list(results.values()), axis=1)
                    # print('TABLE')
                    # print(table['resolved_params.sc_poly.smoothing'])
                    eval_type_to_tables[type].append(table)
            if target.is_file():
                # Assume CSV file
                table = pd.read_csv(target, low_memory=False)
                if len(table):
                    type = table['node'].iloc[0]
                    eval_type_to_tables[type].append(table)

        eval_type_to_aggregator = {}
        for type, tables in eval_type_to_tables.items():
            table = tables[0] if len(tables) == 1 else pd.concat(tables).reset_index(drop=True)
            # print('TABLE2')
            # print(table['resolved_params.sc_poly.smoothing'])
            agg = Aggregator(table)
            agg.build()
            # print('agg.TABLE')
            # print(agg.table['resolved_params.sc_poly.smoothing'])
            eval_type_to_aggregator[type] = agg
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

    plot_params = Value(False, isflag=True, help='if True, param plots will be drawn')

    stdout_report = Value(True, isflag=True, help='if True, print a report to stdout')

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


def main(cmdline=True, **kwargs):
    """
    Aggregate entry point.

    Loads results for each evaluation type, constructs aggregator objects, and
    then executes user specified commands that could include filtering,
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
    from kwutil.util_yaml import Yaml
    rich.print('config = {}'.format(ub.urepr(config, nl=1)))

    eval_type_to_aggregator = config.coerce_aggregators()
    orig_eval_type_to_aggregator = eval_type_to_aggregator  # NOQA

    if config.eval_nodes is not None:
        eval_type_to_aggregator = ub.udict(eval_type_to_aggregator) & config.eval_nodes

    output_dpath = ub.Path(config['output_dpath'])
    for agg in eval_type_to_aggregator.values():
        agg.output_dpath = output_dpath

    rois = config.rois
    # rois = {'KR_R001', 'KR_R002', 'BR_R002'}

    if config.embed or config.snapshot:
        # Sneaky way around linting filters, but also a more concise than
        # try/except, and perhaps we can generalize to people's favorite
        # shells?
        embedding_modpath = ub.modname_to_modpath('xdev')
        if embedding_modpath is None:
            print('missing embed module')
        if embedding_modpath is not None:

            print(f'eval_type_to_aggregator = {ub.urepr(eval_type_to_aggregator, nl=1)}')
            for type, agg in eval_type_to_aggregator.items():
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

    if config.query:
        print('Running query')
        new_eval_type_to_aggregator = {}
        for key, agg in eval_type_to_aggregator.items():
            new_agg = agg.filterto(query=config.query)
            new_eval_type_to_aggregator[key] = new_agg
            rich.print(f'Query {key} filtered to {len(new_agg)}/{len(agg)} rows')
        eval_type_to_aggregator = new_eval_type_to_aggregator

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
                fname = f'{agg.type}_{hostname}_{num_results:05d}_{timestamp}.csv.zip'
                csv_fpath = agg.output_dpath / fname
                print(f'Exported tables to: {csv_fpath}')
                agg.table.to_csv(csv_fpath, index_label=False)

    if config.stdout_report:
        if config.stdout_report is not True:
            report_config = Yaml.coerce(config.stdout_report)
        else:
            report_config = {}
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

        regions_of_interest = agg.macro_key_to_regions[agg.primary_macro_region]
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
        if 0:
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

            x = 'bas_poly_eval.params.bas_pxl.output_space_scale'
            sns.boxplot(data=table, x=x, y=main_metric)
            ax = plt.gca()
            ax.set_title(f'BAS Macro Average over {regions_of_interest}')
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
        from kwutil.util_yaml import Yaml
        import rich
        report = agg.varied_parameter_report()
        yaml_text = Yaml.dumps(report)
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

        varied_counts = util_pandas.DataFrame(agg.effective_params).varied_value_counts()
        # on_error='placeholder')

        # from geowatch.utils import result_analysis
        varied_counts = agg.table.varied_value_counts(on_error='placeholder')
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

        # if 0:
        #     from geowatch.utils import util_dotdict
        #     nested = util_dotdict.dotdict_to_nested(varied_counts)
        #     graph = util_dotdict.indexable_to_graph(nested)

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
                'subcolcolumn_depth_hist': length_hist,
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
        params = util_pandas.DataFrame(agg.resolved_params)
        if metrics_of_interest is None:
            metrics_of_interest = agg.primary_metric_cols
            # metrics_of_interest = ['metrics.bas_pxl_eval.salient_AP']

        metrics = agg.metrics[metrics_of_interest]
        params = params.applymap(lambda x: str(x) if isinstance(x, list) else x)

        varied_counts = params.varied_value_counts(dropna=True)

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
            'params': params,
            'metrics': metrics,
        }
        analysis = result_analysis.ResultAnalysis(results, params=chosen_params)
        # analysis.results
        analysis.analysis()

    def report_best(agg, top_k=100, shorten=True, per_group=None, verbose=1,
                    reference_region=None, print_models=False, concise=False,
                    show_csv=False) -> TopResultsReport:
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

        Returns:
            TopResultsReport:
                contains:
                region_id_to_summary (T1=Dict[str, DataFrame]):
                    mapping from region_id to top k results
                top_param_lut (T2=Dict[str, DataFrame]):
                    mapping from param hash to invocation details
        """
        import rich
        import pandas as pd
        import numpy as np
        from geowatch.utils import util_pandas

        if isinstance(per_group, float) and math.isinf(per_group):
            per_group = None
        if isinstance(top_k, float) and math.isinf(top_k):
            top_k = None

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

            # Rank reference region param_hashids of the primary metrics
            metric_cols = group.columns.intersection(agg.metrics.columns)
            metric_group = group[metric_cols]
            top_locs = util_pandas.pandas_argmaxima(metric_group, agg.primary_metric_cols, k=top_k)
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
                ranked_locs = util_pandas.pandas_argmaxima(
                    group, _agg.primary_metric_cols, k=top_k)
            else:
                # Rank the rows for this region by the reference rank
                # len(reference_hashid_to_rank)
                def make_rank_getter(d):  # no closure for embed debug
                    return lambda x: d.get(x, float('inf'))
                rank_getter = make_rank_getter(reference_hashid_to_rank)
                ranking = group['param_hashid'].apply(rank_getter)
                ranking = ranking[np.isfinite(ranking)]
                ranked_locs = ranking.sort_values().index

            ranked_group = group.loc[ranked_locs]

            param_lut = _agg.hashid_to_params.subdict(ranked_group['param_hashid'])
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

            if show_csv:
                varied_keys = list(varied.keys())
                param_table = pd.DataFrame.from_dict(top_param_lut).T
                param_table.index.name = 'param_hashid'
                param_table = util_pandas.DataFrame(param_table)
                param_table = param_table.reorder(varied_keys, axis=1, intersect=1)
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

            if only_one_top_item:
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
                rich.print(_justone)

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
                        rich.print(f'Top {len(summary_table)} / {ntotal} for {agg.type}, {region_id} = {macro_regions}{ref_text}')
                    else:
                        rich.print(f'Top {len(summary_table)} / {ntotal} for {agg.type}, {region_id}{ref_text}')

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

            model_col = agg.model_cols[0]

            # HACK: we want to group models that came from the same training
            # run so we report a more diverse set of models. We typically group
            # models together in a folder, but this is not robust, so we Only
            # do this grouping if the parent folder has a special name

            model_paths = [ub.Path(p) for p in table[model_col].tolist()]
            hacked_groups = [
                p if p.parent.name.startswith('Drop') else p for p in model_paths]
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

            # all_models_fpath = ub.Path('$HOME/code/watch/dev/reports/split1_all_models.yaml').expand()
            # known_models = Yaml.coerce(all_models_fpath)
            # set(known_models).issuperset(set(chosen_models))
            # if 0:
            #     new_models_fpath = ub.Path('$HOME/code/watch/dev/reports/unnamed_shortlist.yaml').expand()
            # new_models_fpath.write_text(shortlist_text)

        report = TopResultsReport(region_id_to_summary, top_param_lut)
        return report

    def resource_summary_table(agg):
        import pandas as pd
        from kwutil import util_time
        table = agg.table.copy()
        resources = agg.resources

        duration_cols = [
            k for k in resources.keys()
            if k.endswith('.duration')
        ]
        for k in duration_cols:
            table.loc[:, k] = table.loc[:, k].apply(lambda x: util_time.coerce_timedelta(x) if not pd.isnull(x) else x)

        resource_summary = []
        for duration_key in duration_cols:
            a, b, c = duration_key.split('.')
            uuid_key = f'context.{b}.uuid'

            chosen = []
            for _, group in table.groupby(uuid_key):
                idx = group[duration_key].idxmax()
                chosen.append(idx)

            asec = util_time.coerce_timedelta('1second')

            unique_rows = table.loc[chosen]
            row = {
                'node': b,
                'resource': c,
                'total': unique_rows[duration_key].sum().round(asec),
                'mean': unique_rows[duration_key].mean().round(asec),
                'num': len(chosen),
            }
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

    def report_resources(agg):
        import rich
        resource_summary_df = agg.resource_summary_table()
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
        assert agg.type is not None
        base_dpath = (agg.output_dpath / 'param_links' / agg.type)
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


class Aggregator(ub.NiceRepr, AggregatorAnalysisMixin):
    """
    Stores multiple data frames that separate metrics, parameters, and other
    information using consistent pandas indexing. Can be filtered to a
    comparable subsets of choice. Can also handle building macro averaged
    results over different "regions" with the same parameters.

    Set config based on your problem
    """
    def __init__(agg, table, output_dpath=None,
                 type=None,
                 primary_metric_cols='auto',
                 display_metric_cols='auto'):
        """
        Args:
            table (pandas.DataFrame):
                a table with a specific column structure (e.g. built by the
                aggregate_loader). See the demo for an example. Needs more docs
                here.

            output_dpath (None | PathLike):
                Path where output aggregate results should be written

            type (str | None):
                should not need to specify this anymore. This should just be
                the "node" column in the table.

            primary_metric_cols (List[str] | Literal['auto']):
                if "auto", then the "type" must be known by the global helpers.
                Otherwise list the metric columns in the priority that should
                be used to rank the rows.

            display_metric_cols (List[str] | Literal['auto']):
                if "auto", then the "type" must be known by the global helpers.
                Otherwise list the metric columns in the order they should be
                displayed (after the primary metrics).
        """
        agg.output_dpath = output_dpath

        from geowatch.utils import util_pandas
        if not isinstance(table, util_pandas.DataFrame):
            table = util_pandas.DataFrame(table)

        agg.table = table
        agg.type = type
        agg.subtables = None
        agg.config = {
            'display_metric_cols': display_metric_cols,
            'primary_metric_cols': primary_metric_cols,
        }

        # This attribute will hold columns that store paths to model files
        agg.model_cols = None

        # This attribute will hold columns that store paths test datasets
        agg.test_dset_cols = None

        agg.hashid_to_params = None
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

        columns[f'context.{node}.uuid'] = [str(uuid.uuid4()) for _ in range(num)]
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

    # def __export(agg):
    #     ...
    #     agg.table
    #     fname = f'{agg.type}_{agg.output_dpath.parent.name}.csv'
    #     agg.table.to_csv(fpath, index_label=False)
    #     fpath = 'bas_results_2023-01.csv.zip'
    #     agg.table.to_csv(fpath, index_label=False)

    def build(agg):
        """
        Inspect the aggregator's table and build supporting information
        """
        from geowatch.mlops.smart_global_helper import SMART_HELPER
        from geowatch.utils import util_pandas
        agg.__dict__.update(**agg.config)

        if len(agg.table) == 0:
            agg.type = 'unknown-type-empty-table'
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

        if agg.type is None:
            agg.type = agg.table['node'].iloc[0]

        metrics_prefix = f'metrics.{agg.type}'
        # params_prefix = f'params.{agg.type}'
        if agg.primary_metric_cols == 'auto' or agg.display_metric_cols == 'auto':
            _primary_metrics_suffixes, _display_metrics_suffixes = SMART_HELPER._default_metrics(agg)

            if agg.primary_metric_cols == 'auto':
                # agg.primary_metric_cols = util_pandas.pandas_suffix_columns(  # fixme sorting
                #     agg.metrics, _primary_metrics_suffixes)
                agg.primary_metric_cols = [f'{metrics_prefix}.{s}' for s in _primary_metrics_suffixes]
            if agg.display_metric_cols == 'auto':
                # agg.display_metric_cols = util_pandas.pandas_suffix_columns(  # fixme sorting
                #     agg.metrics, _display_metrics_suffixes)
                agg.display_metric_cols = [f'{metrics_prefix}.{s}' for s in _display_metrics_suffixes]

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
        # effective_params, mappings, hashid_to_params = agg.build_effective_params()
        # agg.hashid_to_params = ub.udict(hashid_to_params)
        # agg.mappings = mappings
        # agg.effective_params = effective_params

        agg.macro_key_to_regions = {}
        agg.region_to_tables = {}
        for region_id, idx_group in agg.index.groupby('region_id'):
            agg.region_to_tables[region_id] = agg.table.loc[idx_group.index]
        agg.macro_compatible = agg.find_macro_comparable()

    def __nice__(self):
        return f'{self.type}, n={len(self)}'

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
        new_agg = Aggregator(new_table, type=agg.type,
                             output_dpath=agg.output_dpath, **agg.config)
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
    def params(self):
        ub.schedule_deprecation(None, migration='use requested_params instead')
        return self.subtables['params']

    @property
    def requested_params(self):
        return self.subtables['params']

    @property
    def specified_params(self):
        return self.subtables['specified']

    @property
    def resolved_params(self):
        return self.subtables['resolved_params']

    def build_effective_params(self):
        """
        Consolodate / cleanup / expand information

        THIS COMPUTES THE ``param_hashid`` COLUMN!

        The "effective params" normalize the full set of given parameters so we
        can compute more consistent param_hashid. This is done by condensing
        paths (which is a debatable design decision) as well as mapping
        non-hashable data to strings.

        Populates:

            * ``self.hashid_to_params``

            * ``self.mappings``

            * ``self.effective_params``

        """
        import pandas as pd
        from geowatch.utils import util_pandas
        from geowatch.mlops.smart_global_helper import SMART_HELPER
        params = self.requested_params
        effective_params = params.copy()

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
        existing_path_colnames = params.columns.intersection(path_colnames)

        for colname in existing_path_colnames:
            colvals = params[colname]
            condensed, mapper = util_pandas.pandas_condense_paths(colvals)
            mappings[colname] = mapper
            effective_params[colname] = condensed

        # TODO: Give the user more customization and control over how effective
        # params are built. I'm going to hard code my use-case for now,
        # refactor later.
        # Q: Do we need to be using "resolved parameters" here?
        if 0:
            resolved_params = util_pandas.DataFrame(self.subtables['resolved_params'])
            channel_cols = resolved_params.match_columns('*.channels')
            unique_channels = sorted(set(ub.flatten(resolved_params[channel_cols].value_counts().index)))

            CHANNEL_LUT = {
                'blue_COLD_cv|green_COLD_cv|red_COLD_cv|nir_COLD_cv|swir16_COLD_cv|swir22_COLD_cv|blue_COLD_a0|green_COLD_a0|red_COLD_a0|nir_COLD_a0|swir16_COLD_a0|swir22_COLD_a0|blue_COLD_rmse|green_COLD_rmse|red_COLD_rmse|nir_COLD_rmse|swir16_COLD_rmse|swir22_COLD_rmse': 'COLD.0:18',
            }
            import delayed_image
            channel_mapping = {}
            for orig_c in unique_channels:
                c = orig_c
                for k, v in CHANNEL_LUT.items():
                    c = c.replace(k, v)
                c = delayed_image.sensorchan_spec.SensorChanSpec.coerce(c)
                print(c.normalize().concise())
                channel_mapping[orig_c] = c

            for col in channel_cols:
                self.subtables['resolved_params'][col] = self.subtables['resolved_params'][col].apply(channel_mapping.get)

            effective_params = util_pandas.DataFrame(effective_params)

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
        hashid_to_params = {}
        for param_vals, group in effective_params.groupby(param_cols, dropna=False):
            # Further subdivide the group so each row only computes its hash
            # with the parameters that were included in its row
            is_group_included = is_param_included.loc[group.index]

            # NOTE: groupby will replace None with NaN in the returned
            # iteration values
            # Work around this by choosing the first item from the group
            # itself.
            unique_params = group.iloc[0][param_cols]

            for param_flags, subgroup in is_group_included.groupby(param_cols, dropna=False):
                # valid_param_cols = list(ub.compress(param_cols, param_flags))
                # valid_param_vals = list(ub.compress(param_vals, param_flags))
                # valid_unique_params = ub.dzip(valid_param_cols, valid_param_vals)

                valid_unique_params = unique_params[list(param_flags)].to_dict()

                hashid = hash_param(valid_unique_params, version=1)
                hashid_to_params[hashid] = valid_unique_params
                hashids_v1.loc[subgroup.index] = hashid

        # Update the index with an effective parameter hashid
        self.index.loc[hashids_v1.index, 'param_hashid'] = hashids_v1
        self.table.loc[hashids_v1.index, 'param_hashid'] = hashids_v1
        self.hashid_to_params = ub.udict(hashid_to_params)
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
            for single_rois in rois:
                agg.build_single_macro_table(single_rois, **kwargs)
        else:
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
        sum_cols = [c for c in agg.metrics.columns if c.endswith((
            '_tp', '_fp', '_fn', '_ntrue', '_npred'))]
        average_cols = [c for c in agg.metrics.columns if c.endswith((
            'mAP', 'APUC', 'mAPUC', 'mAUC', 'AP', 'AUC', 'f1', 'FAR', 'ppv',
            'tpr', 'ffpa', 'f1', 'f1_siteprep', 'f1_active'))]
        ignore_cols = [c for c in agg.metrics.columns if c.endswith(('rho', 'tau'))]
        sum_cols = agg.metrics.columns.intersection(sum_cols)

        start_time_cols = DotDictDataFrame.search_columns(agg.table, 'start_timestamp')
        stop_time_cols = DotDictDataFrame.search_columns(agg.table, 'stop_timestamp')

        ignore_cols = [c for c in agg.metrics.columns if c.endswith(('rho', 'tau'))]

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
            # rich.print(ub.urepr(confusor_config))
            # cmdline = 0
            # kwargs = confusor_config
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
        >>> aggregate_param_cols(df, aggregator=aggregator, hash_cols=hash_cols, allow_nonuniform=allow_nonuniform)
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


__config__ = AggregateEvluationConfig
__config__.main = main


if __name__ == '__main__':
    """
    CommandLine:
        python ~/code/watch/geowatch/mlops/aggregate_evaluation.py --help
    """
    main()
