r"""
Loads results from an evaluation and aggregates them

Ignore:

    # Real data

    Given results from schedule_evaluation

    # SC
    DVC_DATA_DPATH=$(geowatch_dvc --tags='phase2_data' --hardware=auto)
    DVC_EXPT_DPATH=$(geowatch_dvc --tags='phase2_expt' --hardware=auto)
    python -m watch.mlops.aggregate \
        --pipeline=sc \
        --root_dpath="$DVC_EXPT_DPATH/_testsc"


    # BAS
    DVC_DATA_DPATH=$(geowatch_dvc --tags='phase2_data' --hardware=auto)
    DVC_EXPT_DPATH=$(geowatch_dvc --tags='phase2_expt' --hardware=auto)
    python -m watch.mlops.aggregate \
        --pipeline=bas \
        --root_dpath="$DVC_EXPT_DPATH/_testpipe"

    # BAS
    DVC_DATA_DPATH=$(geowatch_dvc --tags='phase2_data' --hardware=auto)
    DVC_EXPT_DPATH=$(geowatch_dvc --tags='phase2_expt' --hardware=auto)
    python -m watch.mlops.aggregate \
        --pipeline=bas \
        --io_workers=0 \
        --target \
            "$DVC_EXPT_DPATH/_testpipe" \
            "$DVC_EXPT_DPATH/_timekernel_test_drop4" \
        --output_dpath=./my_aggregate \
        --export_tables=True

    # BAS
    python -m watch.mlops.aggregate \
        --target ./my_aggregate/*.csv.zip \
        --stdout_report=True --rois KR_R001,KR_R002

    python -m watch.mlops.aggregate \
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

# import xdev
# xdev.make_warnings_print_tracebacks()


class AggregateLoader(DataConfig):
    """
    Just the part of the config related to loading
    """

    target = Value(None, help=ub.paragraph(
        '''
        The input to the aggregator, which can take several forms:
        (1) the root directory of an mlops evaluation,
        (2) one or more pre-aggregated files,
        '''), nargs='+')

    pipeline = Value('joint_bas_sc', help='the name of the pipeline to run')

    io_workers = Value('avail', help='number of processes to load results')

    eval_nodes = Value(None, help='eval nodes to look at')

    def __post_init__(self):
        from kwutil.util_yaml import Yaml
        self.eval_nodes = Yaml.coerce(self.eval_nodes)

    @profile
    def coerce_aggregators(config):
        from kwutil import util_path
        from watch.mlops.aggregate_loader import build_tables
        import pandas as pd
        input_targets = util_path.coerce_patterned_paths(config.target)
        eval_type_to_tables = ub.ddict(list)
        print(f'Found {len(input_targets)} input targets')
        for target in ub.ProgIter(input_targets, desc='loading targets', verbose=3):
            if target.is_dir():
                # Assume Pipeline Output dir
                eval_type_to_results = build_tables(target, config.pipeline, config.io_workers, config.eval_nodes)
                for type, results in eval_type_to_results.items():
                    table = pd.concat(list(results.values()), axis=1)
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
            agg = Aggregator(table)
            agg.build()
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

    stdout_report = Value(False, isflag=True, help='if True, print a report to stdout')

    resource_report = Value(False, isflag=True, help='if True report resource utilization')

    rois = Value('auto', help='Comma separated regions of interest')

    inspect = Value(None, help='param hashid to look at')

    query = Value(None, type=str, help='a pandas query to restrict the rows of the table we consider')

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

    Ignore:
        >>> from watch.mlops.aggregate import *  # NOQA
        >>> import watch
        >>> data_dvc_dpath = watch.find_dvc_dpath(tags='phase2_data', hardware='auto')
        >>> expt_dvc_dpath = watch.find_dvc_dpath(tags='phase2_expt', hardware='auto')
        >>> cmdline = 0
        >>> kwargs = {
        >>>     'target': [expt_dvc_dpath / '_testpipe', expt_dvc_dpath / '_timekernel_test_drop4'],
        >>>     'pipeline': 'bas',
        >>>     'io_workers': 10,
        >>> }

        config = AggregateEvluationConfig.cli(cmdline=cmdline, data=kwargs)
        agg_dpath = ub.Path(config['root_dpath']) / 'aggregate'
        from watch.mlops.aggregate_loader import build_tables
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

    eval_type_to_aggregator = config.coerce_aggregators()

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

    for type, agg in eval_type_to_aggregator.items():
        print(f'agg={agg}')

    timestamp = ub.timestamp()
    if config.export_tables:
        for type, agg in eval_type_to_aggregator.items():
            if len(agg):
                agg.output_dpath.ensuredir()
                fname = f'{agg.type}_{timestamp}.csv.zip'
                csv_fpath = agg.output_dpath / fname
                print(f'csv_fpath={csv_fpath}')
                agg.table.to_csv(csv_fpath, index_label=False)

    if config.stdout_report:
        from kwutil.util_yaml import Yaml
        if config.stdout_report is not True:
            report_config = Yaml.coerce(config.stdout_report)
        else:
            report_config = {}
        for type, agg in eval_type_to_aggregator.items():
            if len(agg):

                if rois is not None:
                    agg.build_macro_tables(rois)

                agg.report_best(**ub.compatible(report_config, agg.report_best))
                if report_config.get('analyze', False):
                    agg.analyze()
                if report_config.get('macro_analysis', False):
                    agg.macro_analysis()

    if config.resource_report:
        for type, agg in eval_type_to_aggregator.items():
            if len(agg):
                agg.report_resources()

    if config.plot_params['enabled']:
        for type, agg in eval_type_to_aggregator.items():
            if len(agg):
                build_all_param_plots(agg, rois, config)
    # automated_analysis(eval_type_to_aggregator, config)

    if config.inspect:
        agg = eval_type_to_aggregator['bas_pxl_eval']
        for type, agg in eval_type_to_aggregator.items():
            if len(agg):
                subagg = agg.filterto(param_hashids=config.inspect if ub.iterable(config.inspect) else [config.inspect])
                if len(subagg):
                    subagg.make_summary_analysis(config)
                    # from watch.mlops import confusor_analysis
                    # for region_id, group in subagg.index.groupby('region_id'):
                    #     group_agg = subagg.filterto(index=group.index)
                    #     # confusor_analysis.main(cmdline=0, )


@profile
def build_all_param_plots(agg, rois, config):
    from watch.mlops.smart_global_helper import SMART_HELPER
    from watch.utils import util_pandas

    def build_special_columns(agg):
        resolved_params = util_pandas.DotDictDataFrame(agg.resolved_params)
        part1 = resolved_params.query_column('batch_size')
        if len(part1) > 1:
            # Disambiguate fit and pred batch size
            part1_ = [p for p in part1 if '_fit' in p]
            if len(part1_) == 1:
                part1 = part1_

        part2 = resolved_params.query_column('accumulate_grad_batches')
        prefix_to_batchsize = ub.group_items(part1, key=lambda x: x.rsplit('.', 1)[0])
        prefix_to_accumbatch = ub.group_items(part2, key=lambda x: x.rsplit('.', 1)[0])

        for prefix in set(prefix_to_batchsize) | set(prefix_to_accumbatch):
            cols1 = prefix_to_batchsize.get(prefix, None)
            cols2 = prefix_to_accumbatch.get(prefix, None)
            val_accum = 1
            val_bsize = resolved_params[cols1[0]]
            if cols2 is not None:
                assert len(cols2) == 1
                val_accum = resolved_params[cols2[0]].copy()
                val_accum[val_accum.isnull()] = 1
                val_accum[val_accum == 'None'] = 1
            val_effective_bsize = val_bsize * val_accum
            agg.table.loc[:, prefix + '.effective_batch_size'] = val_effective_bsize

    plot_config = ub.udict(config.plot_params) - {'enabled'}

    build_special_columns(agg)
    agg.build()

    def preprocess_table(table):
        fillna_cols = table.columns.intersection(agg.resolved_params.columns.union(agg.resolved_params.columns))
        table.loc[:, fillna_cols] = table.loc[:, fillna_cols].fillna('None')
        table = table.applymap(lambda x: str(x) if isinstance(x, list) else x)
        return table

    single_table = table = agg.table
    single_table = preprocess_table(table)

    MARK_DELIVERED = plot_config.get('mark_delivered', False)
    if MARK_DELIVERED:
        SMART_HELPER.mark_delivery(single_table)

    modifier = SMART_HELPER.label_modifier()

    if rois is not None:
        agg.build_macro_tables(rois)
        macro_table = agg.region_to_tables[agg.primary_macro_region].copy()
        macro_table = preprocess_table(macro_table)

        if MARK_DELIVERED:
            SMART_HELPER.mark_delivery(macro_table)
        if 0:
            SMART_HELPER.old_hacked_model_case(macro_table)
        param_to_palette = SMART_HELPER.shared_palletes(macro_table)
        if 0:
            SMART_HELPER.mark_star_models(macro_table)
    else:
        macro_table = None
        param_to_palette = SMART_HELPER.shared_palletes(single_table)

    # agg = plotter.agg
    agg_group_dpath = (agg.output_dpath / ('all_params' + ub.timestamp())).ensuredir()

    plotter = ParamPlotter(agg)

    plotter.agg_group_dpath = agg_group_dpath
    plotter.param_to_palette = param_to_palette
    plotter.modifier = modifier
    plotter.macro_table = macro_table
    plotter.single_table = single_table
    plotter.rois = rois
    plotter.plot_config = plot_config

    vantage = plotter.vantage_points[0]

    for vantage in plotter.vantage_points:
        print('Plot vantage overview: ' + vantage['name'])
        plotter.plot_vantage_overview(vantage)
        print('Plot vantage params: ' + vantage['name'])
        # plotter.plot_vantage_overview(vantage)
        plotter.plot_vantage_params(vantage)


class ParamPlotter:
    """
    Builds the scatter plots and barcharts over different params.
    Working in cleaning this up
    """
    def __init__(plotter, agg):
        from watch.mlops.smart_global_helper import SMART_HELPER
        plotter.agg = agg

        # We will conduct analysis under serveral different vantage points
        vantage_points = SMART_HELPER.default_vantage_points(agg.type)
        for vantage in vantage_points:
            pm = vantage['metric1'].split('.')[-1]
            sm = vantage['metric2'].split('.')[-1]
            name = f'{pm}-vs-{sm}'
            vantage['name'] = name
        plotter.vantage_points = vantage_points

    # def plot_vantage(plotter, vantage):
    #     plotter.plot_vantage_overview(vantage)
    #     plotter.plot_vantage_params(vantage)

    def _add_sv_hack_lines(plotter, ax, table, x, y):
        import matplotlib as mpl

        def add_arrows_to_lines(line_collection, position=None, direction='right', size=15, color=None):
            """
            add an arrow to a line.

            line:       Line2D object
            position:   x-position of the arrow. If None, mean of xdata is taken
            direction:  'left' or 'right'
            size:       size of the arrow in fontsize points
            color:      if None, line color is taken.

            References:
                .. [SO34017866] https://stackoverflow.com/questions/34017866/arrow-on-a-line-plot-with-matplotlib
            """
            if color is None:
                color = line_collection.get_color()

            for segment in line_collection.get_segments():
                xdata = segment[:, 0]
                ydata = segment[:, 1]

                if position is None:
                    position = xdata.mean()
                # find closest index
                import numpy as np
                start_ind = np.argmin(np.absolute(xdata - position))
                start_ind = 0
                if direction == 'right':
                    end_ind = start_ind + 1
                else:
                    end_ind = start_ind - 1

                line_collection.axes.annotate(
                    '',
                    xytext=(xdata[start_ind], ydata[start_ind]),
                    xy=(xdata[end_ind], ydata[end_ind]),
                    arrowprops=dict(arrowstyle="->", color=color),
                    size=size, zorder=0,
                )
        # Hack to compare before/after SV
        # import matplotlib as mpl
        # ax = sns.scatterplot(data=single_table, x=x, y=y, hue='region_id', legend=False)
        # sns.scatterplot(data=single_table, x=x_prev, y=y_prev, ax=ax, legend=False)
        if 'sv_poly_eval' in x.split('.'):
            print('SV HACK!!!')
            x_prev = x.replace('sv_poly_eval', 'bas_poly_eval')
            y_prev = y.replace('sv_poly_eval', 'bas_poly_eval')

            # xy1 = table[[x_prev, y_prev]].values
            # xy2 = table[[x, y]].values
            # uv = xy2 - xy1
            # ax.quiver(xy1.T[0], xy1.T[1], uv.T[0], uv.T[1])

            segments = []
            # patches = []
            for x1, y1, x2, y2 in table[[x_prev, y_prev, x, y]].values:
                segments.append([(x1, y1), (x2, y2)])
                # patch = mpl.patches.FancyArrow(
                #     x1, y1, x2 - x1, y2 - y1,
                #     width=0.001, length_includes_head=True,
                #     head_width=0.001,
                #     head_length=0.001,
                # )
                # patches.append(patch)
                ...
            # collection = mpl.collections.PatchCollection(patches)
            # ax.add_collection(collection)
            line_collection = mpl.collections.LineCollection(segments, color='blue', alpha=0.5, linewidths=1)
            ax.add_collection(line_collection)
            add_arrows_to_lines(line_collection)
            # pts1 = [s[0] for s in segments]
            # pts2 = [s[1] for s in segments]
            # ax.plot(*zip(*pts1), 'rx', label='before SV')
            # ax.plot(*zip(*pts2), 'bo', label='after SV')

    def plot_vantage_overview(plotter, vantage):
        from watch.utils import util_kwplot
        from watch.utils.util_kwplot import scatterplot_highlight
        import numpy as np
        import kwplot
        import kwimage
        from watch.mlops.smart_global_helper import SMART_HELPER
        sns = kwplot.autosns()
        plt = kwplot.autoplt()  # NOQA
        kwplot.close_figures()

        agg = plotter.agg
        rois = plotter.rois
        macro_table = plotter.macro_table
        single_table = plotter.single_table

        modifier = plotter.modifier

        vantage_dpath = (plotter.agg_group_dpath / vantage['name']).ensuredir()

        main_metric = y = vantage['metric1']
        yscale = vantage['scale1']
        x = vantage['metric2']
        xscale = vantage['scale2']

        # main_metric = 'bas_poly_eval.metrics.bas_f1'
        # main_metric = 'bas_poly_eval.metrics.bas_faa_f1'
        main_metric = agg.primary_metric_cols[0]

        finalize_figure = util_kwplot.FigureFinalizer(
            dpath=vantage_dpath,
            size_inches=np.array([6.4, 4.8]) * 1.0,
        )

        fig = kwplot.figure(fnum=2, doclf=True)
        ax = sns.scatterplot(data=single_table, x=x, y=y, hue='region_id', legend=False)
        if plotter.plot_config.get('compare_sv_hack', False):
            # Hack to compare before/after SV
            if 'sv_poly_eval' in x.split('.'):
                plotter._add_sv_hack_lines(ax, single_table, x, y)
        if 'delivered_params' in single_table:
            val_to_color = SMART_HELPER.delivery_to_color
            if 0:
                kwplot.imshow(kwplot.make_legend_img(val_to_color, mode='star', dpi=300))
            scatterplot_highlight(data=single_table, x=x, y=y,
                                  highlight='delivered_params', ax=ax,
                                  color='group',
                                  size=300, val_to_color=val_to_color)

        ax.set_title(f'BAS Per-Region Results (n={len(agg)})')
        ax.set_xscale(xscale)
        ax.set_yscale(yscale)
        finalize_figure.finalize(fig, 'single_results.png')
        # ax.set_xlim(0, np.quantile(agg.metrics[x], 0.99))
        # ax.set_xlim(1e-2, np.quantile(agg.metrics[x], 0.99))

        try:
            fig = kwplot.figure(fnum=90, doclf=True)
            ax = sns.boxplot(data=single_table, x='region_id', y=main_metric)
            ax.set_title(f'BAS Per-Region Results (n={len(agg)})')
            param_histogram = single_table.groupby('region_id').size().to_dict()
            util_kwplot.LabelModifier({
                param_value: f'{param_value}\n(n={num})'
                for param_value, num in param_histogram.items()
            }).relabel_xticks(ax)
            modifier.relabel(ax, ticks=False)
            finalize_figure.finalize(fig, 'single_results_boxplot.png')

        except Exception as ex:
            print(f'ex={ex}')

        if macro_table is not None:
            fig = kwplot.figure(fnum=3, doclf=True)
            ax = fig.gca()
            region_ids = macro_table['region_id'].unique()
            assert len(region_ids) == 1
            macro_region_id = region_ids[0]
            palette = {
                macro_region_id: kwimage.Color('kitware_darkgray').as01()
            }
            ax = sns.scatterplot(data=macro_table, x=x, y=y, hue='region_id', ax=ax, palette=palette)
            if plotter.plot_config.get('compare_sv_hack', False):
                # Hack to compare before/after SV
                if 'sv_poly_eval' in x.split('.'):
                    plotter._add_sv_hack_lines(ax, macro_table, x, y)
            if 'is_star' in macro_table:
                scatterplot_highlight(
                    data=macro_table, x=x, y=y, highlight='is_star', ax=ax,
                    size=300)
            if 'delivered_params' in macro_table:
                import kwimage
                val_to_color = SMART_HELPER.delivery_to_color
                if 0:
                    kwplot.imshow(kwplot.make_legend_img(val_to_color, mode='star', dpi=300))
                scatterplot_highlight(data=macro_table, x=x, y=y,
                                      highlight='delivered_params', ax=ax,
                                      color='group', size=300,
                                      val_to_color=val_to_color)
            ax.set_title(f'BAS Results (n={len(macro_table)})\n'
                         f'Macro Analysis over {ub.urepr(rois, sv=1, nl=0)}')
            ax.set_xscale(xscale)
            ax.set_yscale(yscale)
            finalize_figure.finalize(fig, 'macro_results.png')
            # ax.set_xlim(1e-2, npe.quantile(agg.metrics[x], 0.99))
            # ax.set_xlim(1e-2, 0.7)

    def plot_vantage_params(plotter, vantage):
        import numpy as np
        import kwplot
        import kwarray
        import pandas as pd
        from kwcoco.metrics.drawing import concice_si_display
        from watch.utils import util_pandas
        from watch.utils import util_kwplot
        from watch.utils.util_kwplot import scatterplot_highlight

        sns = kwplot.autosns()
        plt = kwplot.autoplt()  # NOQA
        kwplot.close_figures()

        rois = plotter.rois
        macro_table = plotter.macro_table

        modifier = plotter.modifier
        param_to_palette = plotter.param_to_palette

        param_group_dpath = plotter.agg_group_dpath / 'params'
        vantage_dpath = ((plotter.agg_group_dpath / vantage['name']).ensuredir()).resolve()

        main_metric = y = vantage['metric1']
        yscale = vantage['scale1']
        main_objective = vantage['objective1']

        secondary_metric = x = vantage['metric2']
        xscale = vantage['scale2']

        metric_objectives = {main_metric: main_objective}

        finalize_figure = util_kwplot.FigureFinalizer(
            dpath=vantage_dpath,
            size_inches=np.array([6.4, 4.8]) * 1.0,
        )

        from watch.mlops.smart_global_helper import SMART_HELPER
        blocklist = SMART_HELPER.VIZ_BLOCKLIST

        resolved_params = util_pandas.DotDictDataFrame(macro_table).subframe('resolved_params', drop_prefix=False)
        valid_cols = resolved_params.columns.difference(blocklist)
        resolved_params = resolved_params[valid_cols]

        DO_STAT_ANALYSIS = plotter.plot_config.get('stats_ranking', False)
        if DO_STAT_ANALYSIS:
            ### Build param analysis
            from watch.utils import result_analysis
            metrics_table = util_pandas.DotDictDataFrame(macro_table).subframe('metrics', drop_prefix=False)
            results = {'params': resolved_params,
                       'metrics': metrics_table}
            # agg.primary_metric_cols)
            analysis = result_analysis.ResultAnalysis(
                results, metrics=[main_metric], metric_objectives=metric_objectives)
            analysis.build()
            analysis.analysis()
            print('analysis.varied = {}'.format(ub.urepr(analysis.varied, nl=2)))
            ranked_stats = list(sorted(analysis.statistics, key=lambda x: x['anova_rank_p']))
            param_name_to_stats = {s['param_name']: s for s in ranked_stats}
            ranked_params = ub.oset(param_name_to_stats.keys())
        else:
            ranked_params = []
            for col in resolved_params.columns:
                if len(macro_table[col].unique()) > 1:
                    ranked_params.append(col)
            param_name_to_stats = {}

        # ranked_params = ['bas_poly_eval.params.bas_pxl.package_fpath']
        if len(ranked_params):
            print('Warning: no ranked params')

        def shrink_param_names(param_histogram):
            text_len_thresh = 20
            param_labels = [str(p) for p in param_histogram]
            text_label_size = len(''.join(param_labels))
            if text_label_size > text_len_thresh:
                had_value_remap = True
                # Param names are too long. need to map parameter names to codes.
                param_valname_map = {}
                prefixchar = param_name.split('.')[-1][0].upper()
                for idx, value in enumerate(sorted(param_histogram.keys())):
                    old_name = str(value)
                    new_name = f'{prefixchar}{idx:02d}'
                    param_valname_map[old_name] = new_name
            else:
                had_value_remap = False
                param_valname_map = ub.dzip(param_labels, param_labels)
            return param_valname_map, had_value_remap

        for rank, param_name in enumerate(ub.ProgIter(ranked_params, desc='plot param for ' + vantage['name'], verbose=3)):

            stats = param_name_to_stats.get(param_name, {})
            # stats['moments']
            anova_rank_p = stats.get('anova_rank_p', None)
            # param_name = stats['param_name']

            snskw = {}
            if param_name in param_to_palette:
                snskw['palette'] = param_to_palette[param_name]

            try:
                macro_table = macro_table.sort_values(param_name)
            except Exception as ex:
                print(f'warning ex={ex}')
                ...

            # Number of samples we have for each value of this parameter
            param_histogram = ub.udict(macro_table.groupby(param_name).size().to_dict())
            param_histogram = param_histogram.map_keys(str)

            sub_macro_table = macro_table

            min_variations = plotter.plot_config.get('min_variations', 1)
            if min_variations > 1:
                ignore_params = [k for k, v in param_histogram.items() if v < min_variations]
                param_histogram = ub.udict(param_histogram) - set(ignore_params)
                row_is_ignored = kwarray.isect_flags(macro_table[param_name], ignore_params)
                sub_macro_table = macro_table[~row_is_ignored]
                if len(param_histogram) == 1:
                    print('Skip plot')
                    continue
                ...

            header_lines = [
                f'BAS Results (n={len(sub_macro_table)})',
                f'Macro Analysis over {ub.urepr(rois, sv=1, nl=0)}',
            ]
            if anova_rank_p is not None:
                header_lines.append(f'Effect of {param_name}: anova_rank_p={concice_si_display(anova_rank_p)}')
            header_text = '\n'.join(header_lines)

            param_dpath = (param_group_dpath / param_name).ensuredir().resolve()

            param_valname_map, had_value_remap = shrink_param_names(param_histogram)

            # Mapper for the scatterplot legend
            if had_value_remap:
                freq_mapper_scatter = util_kwplot.LabelModifier({
                    param_value: f'{param_value}\n{param_valname_map[param_value]} (n={num})'
                    for param_value, num in param_histogram.items()
                })
            else:
                freq_mapper_scatter = util_kwplot.LabelModifier({
                    param_value: f'{param_value}\n(n={num})'
                    for param_value, num in param_histogram.items()
                })

            freq_mapper_box = util_kwplot.LabelModifier({
                param_value: f'{param_valname_map[param_value]}\n(n={num})'
                for param_value, num in param_histogram.items()
            })

            fname_prefix = f'macro_results_{rank:03d}_{param_name}'
            param_prefix = f'macro_results_{param_name}'
            param_metric_prefix = f'{param_prefix}_{main_metric}'
            param_metric2_prefix = f'{param_prefix}_{main_metric}_{secondary_metric}'

            # SCATTER
            fig = kwplot.figure(fnum=4, doclf=True)
            ax = sns.scatterplot(data=sub_macro_table, x=x, y=y, hue=param_name, legend=True, **snskw)
            ax.set_title(header_text)
            if 'is_star' in sub_macro_table:
                scatterplot_highlight(data=sub_macro_table, x=x, y=y, highlight='is_star', ax=ax, size=300)

            if plotter.plot_config.get('compare_sv_hack', False):
                # Hack to compare before/after SV
                if 'sv_poly_eval' in x.split('.'):
                    plotter._add_sv_hack_lines(ax, sub_macro_table, x, y)

            ax.set_xscale(xscale)
            ax.set_yscale(yscale)
            modifier.relabel(ax, ticks=False)

            vantage_fpath = vantage_dpath / f'{fname_prefix}_PLT01_scatter_legend.png'
            param_fpath = param_dpath / f'{param_metric2_prefix}_PLT01_scatter_legend.png'
            finalize_figure.finalize(fig, vantage_fpath)
            ub.symlink(real_path=vantage_fpath, link_path=param_fpath, overwrite=True)

            # Scatter legend  (doesnt care about the vantage)
            try:
                param_fpath = param_dpath / f'{param_prefix}_PLT03_scatter_onlylegend.png'
                vantage_fpath = vantage_dpath / f'{fname_prefix}_PLT03_scatter_onlylegend.png'
                if not param_fpath.exists():
                    legend_ax = util_kwplot.extract_legend(ax)
                    freq_mapper_scatter.relabel(legend_ax, ticks=False)
                    finalize_figure.finalize(legend_ax.figure, param_fpath)
                ub.symlink(real_path=param_fpath, link_path=vantage_fpath, overwrite=True)
            except RuntimeError:
                ...
            else:
                ax.get_legend().remove()

            vantage_fpath = vantage_dpath / f'{fname_prefix}_PLT02_scatter_nolegend.png'
            param_fpath = param_dpath / f'{param_metric2_prefix}_PLT02_scatter_nolegend.png'
            finalize_figure.finalize(fig, vantage_fpath)
            ub.symlink(real_path=vantage_fpath, link_path=param_fpath, overwrite=True)

            # BOX
            vantage_fpath = vantage_dpath / f'{fname_prefix}_PLT04_box.png'
            param_fpath = param_dpath / f'{param_metric_prefix}_PLT04_box.png'
            print(f'param_fpath={param_fpath}')
            if not param_fpath.exists():
                fig = kwplot.figure(fnum=5, doclf=True)
                ax = sns.boxplot(data=sub_macro_table, x=param_name, y=y, **snskw)
                freq_mapper_box.relabel_xticks(ax)
                ax.set_title(header_text)
                modifier.relabel(ax, ticks=False)
                modifier.relabel_xticks(ax)
                finalize_figure.finalize(fig, param_fpath)
            ub.symlink(real_path=param_fpath, link_path=vantage_fpath, overwrite=True)

            # Varied value table (doesnt care about the vantage)
            param_fpath = param_dpath / f'{param_prefix}_PLT05_table.png'
            vantage_fpath = vantage_dpath / f'{fname_prefix}_PLT05_table.png'
            if not param_fpath.exists():
                param_code_lut = []
                for old_name, new_name in param_valname_map.items():
                    param_code_lut.append({
                        'code': new_name,
                        'value': old_name,
                        'num': param_histogram[old_name],
                    })
                param_code_lut = pd.DataFrame(param_code_lut)
                if not had_value_remap:
                    param_code_lut = param_code_lut.drop('code', axis=1)
                param_title = 'Key: ' + modifier._modify_text(param_name)
                lut_style = param_code_lut.style.set_caption(param_title)
                util_kwplot.dataframe_table(lut_style, param_fpath, title=param_title)
            ub.symlink(real_path=param_fpath, link_path=vantage_fpath, overwrite=True)


def automated_analysis(eval_type_to_aggregator, config):
    timestamp = ub.timestamp()
    output_dpath = ub.Path(config['root_dpath']) / 'aggregate'
    macro_groups = None
    selector = None

    agg0 = eval_type_to_aggregator.get('bas_poly_eval', None)
    if agg0 is not None:
        ...

        subagg2 = generic_analysis(agg0, macro_groups, selector)

        to_visualize_fpaths = list(subagg2.results['fpaths']['fpath'])
        agg_group_dpath = output_dpath / ('bas_poly_agg_' + timestamp)
        agg_group_dpath = agg_group_dpath.ensuredir()
        # make a analysis link to the final product
        for eval_fpath in to_visualize_fpaths[::-1]:
            print((eval_fpath.parent / 'job_config.json').read_text())
            print(f'eval_fpath={eval_fpath}')
            ub.symlink(real_path=eval_fpath.parent, link_path=agg_group_dpath / eval_fpath.parent.name)
            from watch.mlops import confusion_visualization
            eval_dpath = confusion_visualization.bas_poly_eval_confusion_analysis(eval_fpath)
            # TODO: use the region_id.
            ub.symlink(real_path=eval_dpath, link_path=agg_group_dpath / eval_dpath.name)

    agg0 = eval_type_to_aggregator.get('bas_pxl_eval')
    if agg0 is not None:
        # agg[agg.primary_metric_cols]
        generic_analysis(agg0, macro_groups, selector)

    agg0 = eval_type_to_aggregator.get('sc_poly_eval', None)
    if agg0 is not None:
        ...
        # agg0.analyze()


def fix_duplicate_param_hashids(agg0):
    import kwarray
    # There are some circumstances where we can have duplicates region / param
    # hash ids due to munging of the param fields. In this case they should
    # have the same or similar results. Hack to deduplicate them.
    ideally_unique = list(map(ub.hash_data, agg0.index[['region_id', 'param_hashid']].to_dict('records')))
    dupxs = ub.find_duplicates(ideally_unique)
    remove_idxs = []
    for k, dup_idxs in dupxs.items():
        # dup_df = agg0.metrics.iloc[dup_idxs]
        mtimes = [ub.Path(fpath).stat().st_mtime for fpath in agg0.results['fpaths'].iloc[dup_idxs]['fpath']]
        keep_idx = dup_idxs[ub.argmax(mtimes)]
        remove_idxs.extend(set(dup_idxs) - {keep_idx})

        # is_safe_cols = {
        #     k: ub.allsame(vs, eq=nan_eq)
        #     for k, vs in dup_df.T.iterrows()}
        ...
    flags = ~kwarray.boolmask(remove_idxs, shape=len(agg0.index.index))
    print(f'hack to remove {len(remove_idxs)} / {len(agg0.index.index)} duplicates')
    agg0_ = agg0.compress(flags)
    return agg0_


def generic_analysis(agg0, macro_groups=None, selector=None):
    import pandas as pd
    HACK_DEDUPLICATE = 1
    if HACK_DEDUPLICATE:
        agg0_ = fix_duplicate_param_hashids(agg0)
    else:
        agg0_ = agg0

    if macro_groups is None:
        n_to_keys = ub.group_items(agg0_.macro_compatible, key=len)
        chosen_macro_rois = []
        for n, keys in sorted(n_to_keys.items()):
            if n > 1:
                chosen = max(keys, key=lambda k: (len(agg0_.macro_compatible[k]), k))
                chosen_macro_rois.append(chosen)
    else:
        chosen_macro_rois = macro_groups

    if selector is None:
        selector = chosen_macro_rois[-1]

    print('chosen_macro_rois = {}'.format(ub.urepr(chosen_macro_rois, nl=1)))
    agg0_.build_macro_tables(chosen_macro_rois)

    agg_best, param_lut = agg0_.report_best(top_k=1)
    params_of_interest = pd.concat(agg_best.values())['param_hashid'].value_counts()

    params_of_interest = list(param_lut.keys())
    n1 = len(params_of_interest)
    n2 = len(agg0_.index['param_hashid'])
    print(f'Restrict to {n1} / {n2} top parameters')

    subagg1 = agg0_.filterto(param_hashids=params_of_interest)
    subagg1.build_macro_tables(chosen_macro_rois)
    models_of_interest = subagg1.effective_params[subagg1.model_cols].value_counts()
    print('models_of_interest = {}'.format(ub.urepr(models_of_interest, nl=1)))

    agg1_best, param_lut1 = subagg1.report_best(top_k=1)
    param_hashid = agg1_best[hash_regions(selector)]['param_hashid'].iloc[0]
    params_of_interest1 = [param_hashid]
    # params_of_interest1 = [list(agg1_best.values())[-1]['param_hashid'].iloc[0]]

    n1 = len(params_of_interest1)
    n2 = len(agg0_.index['param_hashid'])
    print(f'Restrict to {n1} / {n2} top parameters')
    subagg2 = agg0_.filterto(param_hashids=params_of_interest1)
    subagg2.build_macro_tables(chosen_macro_rois)
    agg2_best, param_lut2 = subagg2.report_best(top_k=1)  # NOQA
    return subagg2


class AggregatorAnalysisMixin:
    def macro_analysis(agg):
        import pandas as pd
        from watch.utils import result_analysis
        from watch.utils import util_pandas

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

    def analyze(agg):
        """
        Does a stats analysis on each varied parameter. Note this makes
        independence assumptions that may not hold in general.
        """
        from watch.utils import result_analysis
        metrics_of_interest = agg.primary_metric_cols
        # metrics_of_interest = ['metrics.bas_pxl_eval.salient_AP']

        params = agg.resolved_params
        metrics = agg.metrics[metrics_of_interest]
        params = params.applymap(lambda x: str(x) if isinstance(x, list) else x)

        from watch.utils.result_analysis import varied_value_counts
        varied_counts = varied_value_counts(params, dropna=True)

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

    def report_best(agg, top_k=3, shorten=True, per_group=2, verbose=1, reference_region=None, print_models=False):
        """
        Report the top k pointwise results for each region / macro-region.

        Note:
            Results are chosen per-region independently. To get comparable
            results for a specific set of parameters, filter to them and then
            report the top results for that filtering.

        Args:
            top_k (int): number of top results for each region

            shorten (bool): if True, shorten the columns by removing
                non-ambiguous prefixes wrt to a known node type.

            reference_region (str | None):
                if specified filter the top results in all other regions to
                only be with respect to the top results in this region (or
                macro region). Can be set to the special key "final" to choose
                the last region, which is typically a macro region.


        Returns:
            Tuple[T1, T2]:
                region_id_to_summary (T1=Dict[str, DataFrame]):
                    mapping from region_id to top k results
                top_param_lut (T2=Dict[str, DataFrame]):
                    mapping from param hash to invocation details
        """
        import rich
        import pandas as pd
        from watch.utils import util_pandas
        region_id_to_summary = {}
        big_param_lut = {}
        region_id_to_ntotal = {}
        if reference_region:
            # In every region group, restrict to only the top values for the
            # reference region. The idea is to make things comparable to the
            # macro scores.
            if reference_region == 'final':
                reference_region = region_id = list(agg.region_to_tables.keys())[-1]
            else:
                region_id = reference_region
            group = agg.region_to_tables[region_id]
            if len(group) == 0:
                region_to_len = ub.udict(agg.region_to_tables).map_values(len)
                print('region_to_len = {}'.format(ub.urepr(region_to_len, nl=1)))
                raise Exception(f'reference {region_id=} group is empty')
            metric_group = group[group.columns.intersection(agg.metrics.columns)]
            metric_group = metric_group.sort_values(agg.primary_metric_cols)
            top_idxs = util_pandas.pandas_argmaxima(metric_group, agg.primary_metric_cols, k=top_k)
            param_hashids = group.loc[top_idxs]['param_hashid']
            _agg = agg.filterto(param_hashids=param_hashids)
            if region_id in agg.macro_key_to_regions:
                rois = agg.macro_key_to_regions[region_id]
                _agg.build_macro_tables(rois)
            reference_hashids = param_hashids
            reference_hashid_to_rank = {
                hashid: rank for rank, hashid in enumerate(reference_hashids)
            }
        else:
            reference_hashids = None
            _agg = agg

        for region_id, group in _agg.region_to_tables.items():
            if len(group) == 0:
                continue
            metric_group = group[group.columns.intersection(_agg.metrics.columns)]
            metric_group = metric_group.sort_values(_agg.primary_metric_cols)

            top_idxs = util_pandas.pandas_argmaxima(metric_group, _agg.primary_metric_cols, k=top_k)

            final_display_cols = list(ub.oset(_agg.primary_metric_cols + _agg.display_metric_cols))
            top_metrics = metric_group.loc[top_idxs][final_display_cols]
            # top_metrics = top_metrics[_agg.primary_metric_cols + _agg.display_metric_cols]
            top_indexes = group[group.columns.intersection(_agg.index.columns)].loc[top_idxs]
            param_lut = _agg.hashid_to_params.subdict(top_indexes['param_hashid'])
            big_param_lut.update(param_lut)
            summary_table = pd.concat([top_indexes, top_metrics], axis=1)
            if shorten:
                summary_table = util_pandas.pandas_shorten_columns(summary_table)

            if reference_hashids is not None:
                # When a reference region is given, order all per-region rows
                # to align with the reference region.
                ranking = summary_table['param_hashid'].apply(lambda x: reference_hashid_to_rank.get(x, len(reference_hashid_to_rank)))
                sortx = ranking.sort_values().index
                summary_table = summary_table.loc[sortx]

            region_id_to_summary[region_id] = summary_table
            region_id_to_ntotal[region_id] = len(metric_group)

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
            rich.print('Parameter LUT: {}'.format(ub.urepr(top_param_lut, nl=2)))

            # Check for a common special case that we can make more concise output for
            only_one_top_item = all(len(t) == 1 for t in region_id_to_summary.values())
            only_one_source_item = all(n == 1 for n in region_id_to_ntotal.values())

            if only_one_source_item and only_one_top_item:
                justone = pd.concat(list(region_id_to_summary.values()), axis=0)
                submacro = ub.udict(_agg.macro_key_to_regions) & justone['region_id'].values
                if submacro:
                    print('Macro Regions LUT: ' +  ub.urepr(submacro, nl=1))
                rich.print(justone)
            elif only_one_top_item:
                justone = pd.concat(list(region_id_to_summary.values()), axis=0)
                # submacro = ub.udict(_agg.macro_key_to_regions) & justone['region_id'].values
                # if submacro:
                #     print('Macro Regions LUT: ' +  ub.urepr(submacro, nl=1))
                rich.print(justone)
                rich.print('agg.macro_key_to_regions = {}'.format(ub.urepr(_agg.macro_key_to_regions, nl=1)))
            else:
                for region_id, summary_table in region_id_to_summary.items():
                    ntotal = region_id_to_ntotal[region_id]
                    rich.print('---')
                    ref_text = ''
                    if reference_region:
                        ref_text = f' wrt to reference region {reference_region}'

                    if region_id in _agg.macro_key_to_regions:
                        macro_regions = _agg.macro_key_to_regions[region_id]
                        rich.print(f'Top {len(summary_table)} / {ntotal} for {region_id} = {macro_regions}{ref_text}')
                    else:
                        rich.print(f'Top {len(summary_table)} / {ntotal} for {region_id}{ref_text}')
                    rich.print(summary_table.iloc[::-1].to_string())
                    rich.print('')

        if print_models:
            # FIXME: handle macro regions
            tocombine_indexes = []
            for region_id, summary in region_id_to_summary.items():
                if not region_id.startswith('macro_'):
                    tocombine_indexes.append(list(summary.index))

            import itertools as it
            top_indexes = list(ub.oset([x for x in ub.flatten(
                it.zip_longest(*tocombine_indexes)) if x is not None]))

            table = _agg.table.copy()
            table.loc[top_indexes, 'rank'] = list(range(len(top_indexes)))
            table = table.sort_values('rank')

            model_col = agg.model_cols[0]

            chosen_indexes = []
            if 0:
                for expt, group in table.groupby('resolved_params.bas_pxl_fit.name'):
                    group[model_col].tolist()
                    group = group.sort_values('rank')
                    chosen_indexes.extend(group.index[0:2])
                chosen_indexes = table.loc[chosen_indexes, 'rank'].sort_values().index
            else:
                table['_hackname'] = [ub.Path(p).parent.name for p in table[model_col].tolist()]
                for expt, group in table.groupby('_hackname'):
                    # group[model_col].tolist()
                    flags = (group[_agg.primary_metric_cols] > 0).any(axis=1)
                    group = group[flags]
                    group = group.sort_values('rank')
                    chosen_indexes.extend(group.index[0:per_group])
                chosen_indexes = table.loc[chosen_indexes, 'rank'].sort_values().index

            top_k = 40
            chosen_indexes = chosen_indexes[:top_k]

            chosen_table = table.loc[chosen_indexes]

            # # Need to remove invariants for now
            # flags = ~np.array(['invariants' in chan for chan in chosen_table['resolved_params.bas_pxl_fit.channels']])
            # chosen_table = chosen_table[flags]

            from kwutil.util_yaml import Yaml
            chosen_models = list(ub.oset(chosen_table[model_col].tolist()))
            shortlist_text = Yaml.dumps(chosen_models)
            print('Model shortlist (top of the list is a higher scoring model):')
            print(shortlist_text)

            # all_models_fpath = ub.Path('$HOME/code/watch/dev/reports/split1_all_models.yaml').expand()
            # known_models = Yaml.coerce(all_models_fpath)
            # set(known_models).issuperset(set(chosen_models))
            # if 0:
            #     new_models_fpath = ub.Path('$HOME/code/watch/dev/reports/unnamed_shortlist.yaml').expand()
            # new_models_fpath.write_text(shortlist_text)

        return region_id_to_summary, top_param_lut

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
        agg.output_dpath = output_dpath
        agg.table = table
        agg.type = type
        agg.subtables = None
        agg.config = {
            'display_metric_cols': display_metric_cols,
            'primary_metric_cols': primary_metric_cols,
        }

    # def __export(agg):
    #     ...

    #     agg.table

    #     fname = f'{agg.type}_{agg.output_dpath.parent.name}.csv'
    #     agg.table.to_csv(fpath, index_label=False)

    #     fpath = 'bas_results_2023-01.csv.zip'
    #     agg.table.to_csv(fpath, index_label=False)

    def build(agg):
        from watch.mlops.smart_global_helper import SMART_HELPER
        from watch.utils import util_pandas
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
        _primary_metrics_suffixes, _display_metrics_suffixes = SMART_HELPER._default_metrics(agg)

        if agg.primary_metric_cols == 'auto':
            # agg.primary_metric_cols = util_pandas.pandas_suffix_columns(  # fixme sorting
            #     agg.metrics, _primary_metrics_suffixes)
            agg.primary_metric_cols = [f'metrics.{agg.type}.{s}' for s in _primary_metrics_suffixes]
        if agg.display_metric_cols == 'auto':
            # agg.display_metric_cols = util_pandas.pandas_suffix_columns(  # fixme sorting
            #     agg.metrics, _display_metrics_suffixes)
            agg.display_metric_cols = [f'metrics.{agg.type}.{s}' for s in _display_metrics_suffixes]

        _model_suffixes = ['package_fpath']
        _testdset_suffixes = ['test_dataset', 'crop_src_fpath']

        agg.model_cols = util_pandas.pandas_suffix_columns(
            agg.requested_params, _model_suffixes)
        agg.test_dset_cols = util_pandas.pandas_suffix_columns(
            agg.requested_params, _testdset_suffixes)

        # util_pandas.pandas_suffix_columns(agg.resolved_params, _testdset_suffixes)

        effective_params, mappings, hashid_to_params = agg.build_effective_params()
        agg.hashid_to_params = ub.udict(hashid_to_params)
        agg.mappings = mappings
        agg.effective_params = effective_params

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

    def filterto(agg, models=None, param_hashids=None, index=None, query=None):
        import numpy as np
        import kwarray
        final_flags = 1
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

        if index is not None:
            flags = kwarray.isect_flags(agg.index.index, index)
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
        new_agg = Aggregator(new_table, type=agg.type, **agg.config)
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

    def build_effective_params(agg):
        """
        Consolodate / cleanup / expand information
        """
        import pandas as pd
        from watch.utils import util_pandas
        params = agg.params
        effective_params = params.copy()

        HACK_FIX_JUNK_PARAMS = True
        if HACK_FIX_JUNK_PARAMS:
            junk_suffixes = ['space_basale']
            junk_cols = util_pandas.pandas_suffix_columns(effective_params, junk_suffixes)
            effective_params = effective_params.drop(junk_cols, axis=1)

        model_cols = agg.model_cols
        test_dset_cols = agg.test_dset_cols

        mappings : Dict[str, Dict[Any, str]] = {}
        path_colnames = model_cols + test_dset_cols
        for colname in path_colnames:
            colvals = params[colname]
            condensed, mapper = util_pandas.pandas_condense_paths(colvals)
            mappings[colname] = mapper
            effective_params[colname] = condensed

        _specified = util_pandas.DotDictDataFrame(agg.specified_params)
        _specified_params = _specified.subframe('specified')
        is_param_included = _specified_params > 0

        # For each unique set of effective parameters compute a hashid
        param_cols = ub.oset(effective_params.columns).difference(agg.test_dset_cols)
        param_cols = list(param_cols - {'region_id', 'node'})
        hashids_v1 = pd.Series([None] * len(agg.index), index=agg.index.index)
        # hashids_v0 = pd.Series([None] * len(agg.index), index=agg.index.index)
        hashid_to_params = {}
        try:
            list(effective_params.groupby(param_cols, dropna=False))
        except Exception:
            effective_params = effective_params.applymap(lambda x: str(x) if isinstance(x, list) else x)
        for param_vals, group in effective_params.groupby(param_cols, dropna=False):
            # Further subdivide the group so each row only computes its hash
            # with the parameters that were included in its row
            for param_flags, subgroup in is_param_included.loc[group.index].groupby(param_cols, dropna=False):
                valid_param_cols = list(ub.compress(param_cols, param_flags))
                valid_param_vals = list(ub.compress(param_vals, param_flags))
                unique_params = ub.dzip(valid_param_cols, valid_param_vals)
                hashid = hash_param(unique_params, version=1)
                hashid_to_params[hashid] = unique_params
                hashids_v1.loc[subgroup.index] = hashid
                # hashids_v0.loc[subgroup.index] = hash_param(unique_params, version=0)

        # Update the index with an effective parameter hashid
        agg.index.loc[hashids_v1.index, 'param_hashid'] = hashids_v1
        agg.table.loc[hashids_v1.index, 'param_hashid'] = hashids_v1
        # agg.index.loc[hashids_v0.index, 'param_hashid_v0'] = hashids_v0

        return effective_params, mappings, hashid_to_params

    def find_macro_comparable(agg):
        """
        Search for groups that have the same parameters over multiple regions.
        """
        import pandas as pd
        table = pd.concat([agg.index, agg.metrics, agg.resolved_params], axis=1)
        table[['param_hashid']]

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
        # print('macro_compatible_num = {}'.format(ub.urepr(macro_compatible_num, nl=1)))
        # print('region_to_num_compatible = {}'.format(ub.urepr(region_to_num_compatible, nl=1)))
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

    def build_macro_tables(agg, rois=None):
        """
        Builds one or more macro tables
        """
        if rois == 'auto':
            rois = [key for key in agg.macro_compatible.keys() if len(key) > 1]
        if isinstance(rois, list) and len(rois) and ub.iterable(rois[0]):
            # Asked for multiple groups of ROIS.
            for single_rois in rois:
                agg.build_single_macro_table(single_rois)
        else:
            agg.build_single_macro_table(rois)

    @profile
    def build_single_macro_table(agg, rois):
        """
        Builds a single macro table for a choice of regions.
        """

        import pandas as pd
        import numpy as np
        # Given a specific group of regions,

        regions_of_interest = agg._coerce_rois(rois)
        macro_key = hash_regions(regions_of_interest)

        # Define how to aggregate each column
        sum_cols = [c for c in agg.metrics.columns if c.endswith((
            '_tp', '_fp', '_fn', '_ntrue', '_npred'))]
        mean_cols = [c for c in agg.metrics.columns if c.endswith((
            'mAP', 'APUC', 'mAPUC', 'mAUC', 'AP', 'AUC', 'f1', 'FAR', 'ppv',
            'tpr', 'ffpa', 'f1', 'f1_siteprep', 'f1_active'))]
        sum_cols = agg.metrics.columns.intersection(sum_cols)
        mean_cols = agg.metrics.columns.intersection(mean_cols)
        other_metric_cols = agg.metrics.columns.difference(sum_cols).difference(mean_cols)
        if len(other_metric_cols):
            print(f'ignoring agg {other_metric_cols}')
        aggregator = {c: 'mean' for c in mean_cols}
        aggregator.update({c: 'sum' for c in sum_cols})
        aggregator.update({c: 'sum' for c in agg.resources.select_dtypes(np.number).columns})

        # Gather groups that can be aggregated
        comparable_groups = agg.gather_macro_compatable_groups(regions_of_interest)
        if len(comparable_groups) == 0:
            import rich
            rich.print(ub.paragraph(
                f'''
                [yellow]WARNING: Failed to build macro results. No comparable groups
                for rois={rois}
                '''))
        else:
            # Macro aggregaet comparable groups
            macro_rows = []
            for group in comparable_groups:
                if len(group) > 0:
                    macro_row = macro_aggregate(agg, group, aggregator)
                    macro_rows.append(macro_row)

            macro_table = pd.DataFrame(macro_rows).reset_index(drop=True)
            agg.region_to_tables.pop(macro_key, None)
            agg.macro_key_to_regions.pop(macro_key, None)
            agg.macro_key_to_regions[macro_key] = regions_of_interest
            agg.region_to_tables[macro_key] = macro_table
            return macro_table

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


def inspect_node(subagg, id, row, group_agg, agg_group_dpath):
    from watch.utils import util_pandas
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
        import watch
        data_dvc_dpath = watch.find_dvc_dpath(tags='phase2_data', hardware='auto')
        # expt_dvc_dpath = watch.find_dvc_dpath(tags='phase2_expt', hardware='auto')
        true_region_dpath = data_dvc_dpath / 'annotations/drop6/region_models'
        true_site_dpath = data_dvc_dpath / 'annotations/drop6/site_models'

        confusion_fpaths = list((eval_fpath.parent / 'bas_summary_viz').glob('confusion_*.jpg'))
        if len(confusion_fpaths) == 0:
            from watch.mlops import confusor_analysis
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
        - [ ] Rectify with ~/code/watch/watch/utils/util_pandas.py :: aggregate_columns
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
def macro_aggregate(agg, group, aggregator):
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
        sub_aggregator = {c: 'mean' for c in aggregator.keys()}
        sub_aggregator.update({c: 'mean' for c in agg.resources.columns})

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
        python ~/code/watch/watch/mlops/aggregate_evaluation.py --help
    """
    main()
