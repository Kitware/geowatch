"""
Plotting logic for :class:`Aggregator`. These plots illustrate the relationshp
between parameters and metrics from various vantage points.

Used by ./aggregate.py
"""
import ubelt as ub

try:
    from line_profiler import profile
except ImportError:
    profile = ub.identity


def build_plotter(agg, rois, plot_config):
    """
    Used by :class:`Aggregator`, which will generally immediately call
    :func:`ParamPlotter.plot_requested`.

    Returns:
        ParamPlotter
    """
    from geowatch.mlops.smart_global_helper import SMART_HELPER
    from geowatch.utils import util_kwplot
    from kwutil import Yaml

    build_special_columns(agg)
    agg.build()
    single_table = table = agg.table
    single_table = preprocess_table_for_seaborn(agg, table)

    MARK_DELIVERED = plot_config.get('mark_delivered', False)
    if MARK_DELIVERED:
        SMART_HELPER.mark_delivery(single_table)

    modifier: util_kwplot.LabelModifier = SMART_HELPER.label_modifier()

    config_label_mappings = plot_config.get('label_mappings', {})
    modifier.update(config_label_mappings)

    if rois is not None:
        agg.build_macro_tables(rois)
        macro_table = agg.region_to_tables[agg.primary_macro_region].copy()
        macro_table = preprocess_table_for_seaborn(agg, macro_table)

        if MARK_DELIVERED:
            SMART_HELPER.mark_delivery(macro_table)
        param_to_palette = SMART_HELPER.shared_palettes(macro_table)
        if 0:
            SMART_HELPER.mark_star_models(macro_table)
    else:
        macro_table = None
        param_to_palette = SMART_HELPER.shared_palettes(single_table)

    plot_dpath = plot_config.get('plot_dpath', None)
    if plot_dpath is None:
        from geowatch.mlops.aggregate import hash_regions
        if rois is not None:
            region_hash = hash_regions(rois)
        else:
            region_hash = 'allrois'
        plot_dpath = (agg.output_dpath / 'plots')
    else:
        plot_dpath = ub.Path(plot_dpath)

    macro_plot_dpath = (plot_dpath / (f'macro-plots-{region_hash}'))

    USE_EFFECTIVE_HACK = 1
    if USE_EFFECTIVE_HACK:
        # Relabel the resolved params to use the "effective-params"
        # instead.
        # print(f'agg.mappings={agg.mappings}')
        for col, lut in agg.mappings.items():
            resolved_col = 'resolved_' + col
            for c in [col, resolved_col]:
                for table in [macro_table, single_table]:
                    if table is not None:
                        if resolved_col in table:
                            new = table[resolved_col].apply(lambda x: lut.get(x, x))
                            table[resolved_col] = new

    vantage_points = plot_config.get('vantage_points', None)
    plotter = ParamPlotter(agg, vantage_points=vantage_points)

    plotter.plot_dpath = plot_dpath
    plotter.macro_plot_dpath = macro_plot_dpath

    # Maps paramter values to colors
    plotter.param_to_palette = param_to_palette

    # This is for remapping parameter values
    plotter.param_to_valmap = Yaml.coerce(plot_config.get('param_to_valmap', {}))
    plotter.param_to_valmap = plotter.param_to_valmap or {}

    # The modifier is for remapping parameter names
    plotter.modifier = modifier
    plotter.label_modifier = modifier

    # The main data
    plotter.macro_table = macro_table
    plotter.single_table = single_table

    plotter.rois = rois
    plotter.plot_config = plot_config
    plotter.roi_attr = 'region_id'
    return plotter


@profile
def build_all_param_plots(agg, rois, plot_config):
    """
    Main entry point for plotting results from an :class:`Aggregator`.
    """
    plotter = build_plotter(agg, rois, plot_config)
    plotter.plot_requested()


def build_special_columns(agg):
    from geowatch.utils import util_pandas
    resolved_params = util_pandas.DotDictDataFrame(agg.resolved_params)
    part1 = resolved_params.query_column('batch_size')
    if len(part1) > 1:
        # Disambiguate fit and pred batch size
        part1_ = [p for p in part1 if '_fit' in p]
        if len(part1_) == 1:
            part1 = part1_
    part2 = resolved_params.query_column('accumulate_grad_batches')
    prefix_to_batchsize = ub.group_items(part1, key=lambda x: x.rsplit('.', 2)[0])
    prefix_to_accumbatch = ub.group_items(part2, key=lambda x: x.rsplit('.', 2)[0])
    prefixes = set(prefix_to_batchsize) | set(prefix_to_accumbatch)

    for prefix in prefixes:
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

    # agg.table['normalized_params.bas_pxl_fit.initializer.init'] = agg.table['resolved_params.bas_pxl_fit.initializer.init'].apply(lambda x: '/'.join(x.split('/')[-3:]))
    # HACK!!!
    # agg.table['normalized_params.bas_pxl_fit.initializer.init']


def preprocess_table_for_seaborn(agg, table):
    fillna_cols = table.columns.intersection(agg.resolved_params.columns.union(agg.resolved_params.columns))
    table.loc[:, fillna_cols] = table.loc[:, fillna_cols].fillna('None')
    table = table.applymap(lambda x: str(x) if isinstance(x, list) else x)

    from geowatch.utils import util_pandas
    from geowatch.mlops.smart_global_helper import SMART_HELPER
    table = util_pandas.DataFrame(table)
    channel_cols = table.match_columns('*.channels')
    if len(channel_cols):
        unique_channels = list(set(ub.flatten(table[channel_cols].value_counts().index)))
        try:
            unique_channels = sorted(unique_channels)
        except TypeError:
            ...
        channel_mapping = SMART_HELPER.custom_channel_relabel_mapping(unique_channels, coarsen=False)
        for col in channel_cols:
            table[col] = table[col].apply(channel_mapping.get)

    if 'resolved_params.bas_pxl_fit.initializer.init' in table:
        # SMART hack
        table['resolved_params.bas_pxl_fit.initializer.init'] = table['resolved_params.bas_pxl_fit.initializer.init'].apply(lambda x: '/'.join(x.split('/')[-3:]))
    # from geowatch.utils import util_pandas
    # table = util_pandas.DataFrame(table)
    # # table.match_columns('.channels')
    return table


class ParamPlotter:
    """
    Builds the scatter and box-and-whisker plots over different params.
    Working on cleaning this up
    """
    def __init__(plotter, agg, vantage_points=None):
        plotter.agg = agg

        # We will conduct analysis under serveral different vantage points
        if vantage_points is None:
            vantage_points = agg.default_vantage_points

        vantage_points = [Vantage(**vantage) for vantage in vantage_points]

        for vantage in vantage_points:
            pm = vantage['metric1'].split('.')[-1]
            sm = vantage['metric2'].split('.')[-1]
            name = f'{pm}-vs-{sm}'
            vantage['name'] = name
        plotter.vantage_points = vantage_points

    def plot_requested(plotter):
        """
        Simplified entry point
        """
        plot_config = plotter.plot_config
        plotter.plot_dpath.ensuredir()

        flag = plot_config.get('plot_resources', 'try')
        if flag:
            try:
                plotter.plot_resources()
            except Exception as ex:
                print(f'Error plotting resources ex={ex}')
                if flag != 'try':
                    raise

        if plot_config.get('plot_overviews', 1):
            plotter.plot_overviews()

        if plot_config.get('plot_params', 1):
            plotter.plot_params()

    def plot_resources(plotter):
        """
        Draw tables that summarize the resource usage of the experiments.
        """
        import rich
        from geowatch.utils import util_kwplot
        plotter.plot_dpath.ensuredir()
        agg = plotter.agg
        rich.print('[green] ### Plot Resources')
        # resource_summary_df = agg.resource_summary_table()
        resource_summary_df = agg.resource_summary_table_friendly()
        rich.print(resource_summary_df.to_string())
        resource_summary_df

        table = resource_summary_df
        table_title = 'resources'
        table_fpath = plotter.plot_dpath / f'{table_title}.png'
        table_style = table.style.set_caption(table_title)
        util_kwplot.dataframe_table(table_style, table_fpath, title=table_title)
        rich.print(f'Dpath: [link={plotter.plot_dpath}]{plotter.plot_dpath}[/link]')

        table_tex_fpath = plotter.plot_dpath / f'{table_title}.tex'
        tex = table.to_latex(index=False)
        tex = '\\begin{table*}[t]\n' + tex.rstrip('\n') + '\n\\end{table*}'
        print(tex)
        table_tex_fpath.write_text(tex)

    def plot_overviews(plotter):
        """
        Draw the overview for each vantage point.
        Draw tables that summarize the resource usage of the experiments.
        """
        from kwutil.util_progress import ProgressManager
        import rich
        pman = ProgressManager()

        rich.print('[green]### Plot Overviews')
        rich.print(f'Dpath: [link={plotter.plot_dpath}]{plotter.plot_dpath}[/link]')
        with pman:
            for vantage in pman.progiter(plotter.vantage_points, desc='plotting vantage overviews'):
                plotter.plot_vantage_per_region_overview(vantage)

                if plotter.macro_table is not None:
                    plotter.plot_vantage_macro_overview(vantage)

        rich.print(f'Dpath: [link={plotter.plot_dpath}]{plotter.plot_dpath}[/link]')

    def plot_params(plotter):
        from kwutil.util_progress import ProgressManager
        import rich
        pman = ProgressManager()

        rich.print('[green] ### Plot Params')
        if plotter.macro_table is None:
            rich.print(f'[red] Cannot plot params plotter.macro_table={plotter.macro_table}')
            return
        rich.print(f'Dpath: [link={plotter.macro_plot_dpath}]{plotter.macro_plot_dpath}[/link]')
        with pman:
            for vantage in pman.progiter(plotter.vantage_points, desc='plotting vantage params'):
                plotter.plot_vantage_params(vantage, pman=pman)
        rich.print(f'Dpath: [link={plotter.macro_plot_dpath}]{plotter.macro_plot_dpath}[/link]')

    def plot_vantage_per_region_overview(plotter, vantage):
        """
        Draw scatter plots and box plots that that distinguish each region with
        respect to a vantage point.

        Args:
            vantage (Dict):
                The primary and secondary metric to look at results with.
                This must have keys: metric1, metric2 and can optionally
                contain keys: scale1, scale2, objective1, and objective2.
        """
        from geowatch.utils import util_kwplot
        from geowatch.utils.util_kwplot import scatterplot_highlight
        import numpy as np
        import kwplot
        import kwimage
        import rich
        from geowatch.mlops.smart_global_helper import SMART_HELPER
        sns = kwplot.autosns()
        plt = kwplot.autoplt()  # NOQA
        kwplot.close_figures()

        rich.print('[white]### Plot Vantage Region Overview:')
        rich.print(f'[white] * {vantage}')

        agg = plotter.agg
        single_table = plotter.single_table

        name = vantage['name']
        y = vantage['metric1']
        x = vantage['metric2']
        yscale = vantage['scale1']
        xscale = vantage['scale2']
        main_metric = y
        roi_attr = plotter.roi_attr

        plotter.plot_dpath.ensuredir()
        finalize_figure = util_kwplot.FigureFinalizer(
            dpath=plotter.plot_dpath,
            size_inches=np.array([6.4, 4.8]) * 1.0,
        )
        fig = kwplot.figure(fnum=2, doclf=True)

        snskw = {}
        if roi_attr in plotter.param_to_palette:
            roi_to_color = util_kwplot.Palette.coerce(plotter.param_to_palette[roi_attr])
        else:
            unique_rois = single_table[roi_attr].unique()
            roi_to_color = util_kwplot.Palette.coerce(unique_rois)
        snskw['palette'] = roi_to_color

        s = plotter.plot_config.get('scatter.markersize', None)
        scatterkw = snskw.copy()
        if s is not None:
            scatterkw['s'] = s
        ax = sns.scatterplot(data=single_table, x=x, y=y, hue=roi_attr,
                             legend=False, **scatterkw)

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
        ax.set_title(f'Per-Region Results (n={len(agg)})')
        ax.set_xscale(xscale)
        ax.set_yscale(yscale)
        plotter.modifier.relabel(ax, ticks=False)
        finalize_figure.finalize(fig, f'overview-{name}.png')
        rich.print(f'[green] made overview-{name}.png')

        try:
            macro_fig_final = finalize_figure.copy()
            macro_fig_final.update(size_inches=[12, 4], dpi=300)
            fig = kwplot.figure(fnum=90, doclf=True)

            region_order = plotter.plot_config.get('region_order', None)
            boxsns_kw = {}
            if region_order is not None:
                boxsns_kw['order'] = region_order
            boxsns_kw.update(snskw)

            util_kwplot.fix_seaborn_palette_issue(x=roi_attr, snskw=boxsns_kw)
            ax = sns.boxplot(data=single_table, x=roi_attr, y=main_metric, **boxsns_kw)
            ax.set_title(f'Per-Region Results (n={len(agg)})')
            param_histogram = single_table.groupby(roi_attr).size().to_dict()
            util_kwplot.LabelModifier({
                param_value: f'{param_value}\n(n={num})'
                for param_value, num in param_histogram.items()
            }).relabel_xticks(ax)
            plotter.modifier.relabel(ax, ticks=False)
            macro_fig_final.finalize(fig, f'overview-boxplot-{roi_attr}-vs-{main_metric}.png')

        except Exception as ex:
            rich.print(f'[red] warning, unable to plot overview-boxplot-region-vs-{main_metric}.png ex={ex}')
        else:
            rich.print(f'[green] made overview boxplot overview-boxplot-region-vs-{main_metric}.png')

        roi_legend = roi_to_color.make_legend_img(dpi=300)
        roi_legend_fpath = (plotter.plot_dpath / 'roi_legend.png')
        kwimage.imwrite(roi_legend_fpath, roi_legend)
        rich.print('[green] made roi_legend.png')

    def plot_vantage_macro_overview(plotter, vantage):
        """
        Draw a scatter plot that gives an overview of the requested macro table
        wrt to a metric vantage point.

        Args:
            vantage (Dict):
                The primary and secondary metric to look at results with.
                This must have keys: metric1, metric2 and can optionally
                contain keys: scale1, scale2, objective1, and objective2.
        """
        from geowatch.utils import util_kwplot
        from geowatch.utils.util_kwplot import scatterplot_highlight
        import kwplot
        import kwimage
        import numpy as np
        import rich
        from geowatch.mlops.smart_global_helper import SMART_HELPER

        rich.print('[white]### Plot Vantage Macro Overview:')
        rich.print(f'[white] * {vantage}')

        sns = kwplot.autosns()
        plt = kwplot.autoplt()  # NOQA
        kwplot.close_figures()

        rois = plotter.rois
        macro_table = plotter.macro_table
        roi_attr = plotter.roi_attr

        name = vantage['name']
        y = vantage['metric1']
        x = vantage['metric2']
        yscale = vantage['scale1']
        xscale = vantage['scale2']

        roi_finalizer = util_kwplot.FigureFinalizer(
            dpath=plotter.macro_plot_dpath,
            size_inches=np.array([6.4, 4.8]) * 1.0,
        )

        fig = kwplot.figure(fnum=3, doclf=True)
        ax = fig.gca()
        region_ids = macro_table[roi_attr].unique()
        assert len(region_ids) == 1
        macro_region_id = region_ids[0]
        palette = {
            macro_region_id: kwimage.Color('kitware_darkgray').as01()
        }

        s = plotter.plot_config.get('scatter.markersize', None)
        snskw = {}
        scatterkw = snskw.copy()
        if s is not None:
            scatterkw['s'] = s
        ax = sns.scatterplot(data=macro_table, x=x, y=y, hue=roi_attr, ax=ax,
                             palette=palette, **snskw)
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

        from geowatch.utils.util_kwplot import TitleBuilder
        title_builder = TitleBuilder()
        title_builder.add_part(f'Results (n={len(macro_table)})')
        if [r for r in rois if r]:
            title_builder.ensure_newline()
            title_builder.add_part(f'Macro Analysis over {ub.urepr(rois, sv=1, nl=0)}')
        ax.set_title(title_builder.finalize())
        ax.set_xscale(xscale)
        ax.set_yscale(yscale)
        plotter.modifier.relabel(ax, ticks=False)
        roi_finalizer.finalize(fig, f'overview-macro_results-{name}.png')
        rich.print('[green] made overview-macro_results-{name}.png')

    def plot_vantage_params(plotter, vantage, pman=None, params_of_interest=None):
        """
        The main parameter inspection plots.

        A vantage point specifies the metrics to visualize and analyze.  Makes
        scatter plot, box plots, and attempts to draw legends useful for
        communicating results.

        The main per-paramter logic is in :func:`_plot_single_vantage_param`.

        Writes two folders in the "macro-plots-*" folder: params, and vantage.
        Both folders contain the same data, but it is laid out differently.

        1. The params folder contains a sub-folder for each parameter, and that
           folder contains plots for all vantage points.

        2. The "vantage" folder contains a sub-folder for each vantage point
           and all parameters plotted with that vantage point are plotted here.
           This folder will typically be symlinks into the above "params"
           folder.

        Args:
            vantage (Dict):
                The primary and secondary metric to look at results with.
                This must have keys: metric1, metric2 and can optionally
                contain keys: scale1, scale2, objective1, and objective2.

            pman (ProgressManager | None):
                a progress manager used to indicate sub-progress of plotting
        """
        import kwplot
        import rich
        from geowatch.utils import util_pandas
        rich.print('[white]### Plot Vantage Params:')
        rich.print(f'[white] * {vantage}')

        plt = kwplot.autoplt()  # NOQA
        kwplot.autosns()
        kwplot.close_figures()

        macro_table = plotter.macro_table

        main_metric = vantage['metric1']
        main_objective = vantage['objective1']
        metric_objectives = {main_metric: main_objective}

        from geowatch.mlops.smart_global_helper import SMART_HELPER
        blocklist = SMART_HELPER.VIZ_BLOCKLIST

        resolved_params = util_pandas.DotDictDataFrame(macro_table).subframe('resolved_params', drop_prefix=False)
        resolved_params['param_hashid'] = macro_table['param_hashid']
        valid_cols = resolved_params.columns.difference(blocklist)
        resolved_params = resolved_params[valid_cols]

        from kwutil.util_yaml import Yaml
        if params_of_interest is None:
            params_of_interest = Yaml.coerce(plotter.plot_config.get('params_of_interest', None))

        if params_of_interest is not None:
            chosen_params = params_of_interest
            params_of_interest = set(params_of_interest)
            valid_params_of_interest = list(resolved_params.columns.intersection(params_of_interest))
            missing = sorted(set(params_of_interest) - set(valid_params_of_interest))
            chosen_params = valid_params_of_interest
            if missing:
                rich.print('[yellow]WARNING: unknown params of interest!')
                rich.print('missing: {}'.format(ub.repr2(missing)))
                print('chosen_params = {}'.format(ub.urepr(chosen_params, nl=1)))
                suggest_did_you_mean(missing, resolved_params.columns)
        else:
            print('params_of_interest is unspecified, automatically choosing')

        # TODO: cleanup logic
        DO_STAT_ANALYSIS = plotter.plot_config.get('stats_ranking', False)
        if DO_STAT_ANALYSIS:
            ### Build param analysis
            from geowatch.utils import result_analysis
            metrics_table = util_pandas.DotDictDataFrame(macro_table).subframe('metrics', drop_prefix=False)
            results = {'params': resolved_params,
                       'metrics': metrics_table}
            # agg.primary_metric_cols)
            # TODO: params_of_interest in analysis
            analysis = result_analysis.ResultAnalysis(
                results, metrics=[main_metric], metric_objectives=metric_objectives)
            analysis.build()
            analysis.analysis()
            print('analysis.varied = {}'.format(ub.urepr(analysis.varied, nl=2)))
            ranked_stats = list(sorted(analysis.statistics, key=lambda x: x['anova_rank_p']))
            param_name_to_stats = {s['param_name']: s for s in ranked_stats}
            ranked_params = ub.oset(param_name_to_stats.keys())
            chosen_params = ranked_params
            if params_of_interest is not None:
                chosen_params = params_of_interest
        else:
            if params_of_interest is None:
                chosen_params = []
                for col in resolved_params.columns:
                    if len(macro_table[col].unique()) > 1:
                        chosen_params.append(col)
            param_name_to_stats = {}

        # ranked_params = ['bas_poly_eval.params.bas_pxl.package_fpath']
        if not len(chosen_params):
            print('Warning: no chosen params')
        else:
            print('chosen_params = {}'.format(ub.urepr(chosen_params, nl=1)))

        from kwutil.util_progress import ProgressManager
        owns_pman = 0

        results = []
        if pman is None:
            pman = ProgressManager()
            pman.__enter__()
            owns_pman = 1
        try:
            for rank, param_name in enumerate(pman.progiter(chosen_params, desc='plot param for ' + vantage['name'], verbose=3)):
                try:
                    drawn_rows = plotter._plot_single_vantage_param(
                        rank, macro_table, param_name, vantage,
                        params_of_interest, param_name_to_stats)
                    results.append(drawn_rows)
                except SkipPlot:
                    continue

            rich.print(f'Dpath: [link={plotter.macro_plot_dpath}]{plotter.macro_plot_dpath}[/link]')
            # agg0.analyze()
        finally:
            if owns_pman:
                pman.__exit__()
        return results

    def _plot_single_vantage_param(plotter, rank, macro_table, param_name,
                                   vantage, params_of_interest,
                                   param_name_to_stats):
        """
        Inner loop for :func:`ParamPlotter.plot_vantage_params`,
        todo: reduce arguments
        """
        import rich
        import kwplot
        import kwarray
        import pandas as pd
        from kwcoco.metrics.drawing import concice_si_display
        from geowatch.utils import util_kwplot
        from geowatch.utils.util_kwplot import scatterplot_highlight
        import numpy as np
        import seaborn as sns

        rich.print(f'param_name = {ub.urepr(param_name, nl=1)}')

        param_group_dpath = plotter.macro_plot_dpath / 'params'
        param_to_palette = plotter.param_to_palette
        vantage_dpath = ((plotter.macro_plot_dpath / 'vantage' / vantage['name']).ensuredir()).resolve()
        vantage_flat_dpath = (vantage_dpath / '_flat').ensuredir()
        finalize_figure = util_kwplot.FigureFinalizer(
            size_inches=np.array([6.4, 4.8]) * 1.0,
        )

        modifier = plotter.modifier

        rois = plotter.rois
        main_metric = y = vantage['metric1']
        secondary_metric = x = vantage['metric2']
        yscale = vantage['scale1']
        xscale = vantage['scale2']

        stats = param_name_to_stats.get(param_name, {})
        # stats['moments']
        anova_rank_p = stats.get('anova_rank_p', None)
        # param_name = stats['param_name']

        # if 0:
        #     # macro_table[param_name] = macro_table[param_name].apply(lambda x: config_label_mappings.get(x, x))
        #     param_to_palette[param_name] = SMART_HELPER.make_param_palette(macro_table[param_name].unique())

        try:
            macro_table = macro_table.sort_values(param_name)
        except Exception as ex:
            rich.print(f'[yellow] warning, unable to sort values by {param_name} ex={ex}')

        # Number of samples we have for each value of this parameter
        raw_param_histogram = ub.udict(macro_table.groupby(param_name).size().to_dict())
        orig_unique_params = ub.udict()
        param_histogram = ub.udict()
        for k, v in raw_param_histogram.items():
            str_key = str(k)
            if str_key in param_histogram:
                param_histogram[str_key] += v
                orig_unique_params[str_key].append(k)
            else:
                orig_unique_params[str_key] = [k]
                param_histogram[str_key] = v

        sub_macro_table = macro_table
        num_macro_rows = len(macro_table)

        min_variations = plotter.plot_config.get('min_variations', 1)  # if less than this number of groups, skip the plot
        max_variations = plotter.plot_config.get('max_variations', float('inf'))  # if less than this number of groups, skip the plot
        min_support = plotter.plot_config.get('min_support', 1)  # minimum amount of values in a group, otherwise remove that group

        explicitly_requested = (params_of_interest is not None and param_name in params_of_interest)

        if min_support > 1 and not explicitly_requested:
            ignore_params = [k for k, v in param_histogram.items() if v < min_support]
            param_histogram = ub.udict(param_histogram) - set(ignore_params)
            row_is_ignored = kwarray.isect_flags(macro_table[param_name], ignore_params)
            sub_macro_table = macro_table[~row_is_ignored]

        num_filtered_rows = num_macro_rows - len(sub_macro_table)
        n_support_stats = kwarray.stats_dict(list(param_histogram.values()), quantile=False, median=True)
        unique_val_types = list(map(type, ub.flatten(orig_unique_params.take(param_histogram))))
        value_type_hist = ub.dict_hist(unique_val_types)
        column_dtype = str(sub_macro_table[param_name].dtype)
        param_summary = {
            'param_name': param_name,
            'num_rows': len(sub_macro_table),
            'num_filtered_rows': num_filtered_rows,
            # Does this need a better name? It is showing the distribution of
            # support for the different groupings.
            'n_support_stats': n_support_stats,
            'column_dtype': column_dtype,
            'value_type_hist': value_type_hist,
        }
        rich.print(f'param_summary = {ub.urepr(param_summary, nl=2)}')

        if len(param_histogram) < min_variations:
            msg = f'skip plot for param_name={param_name} due to min_variations'
            rich.print(f'[red] {msg}')
            raise SkipPlot(msg)

        if len(param_histogram) > max_variations:
            msg = f'skip plot for param_name={param_name} due to max_variations'
            rich.print(f'[red] {msg}')
            raise SkipPlot(msg)

        if 1:
            # HACK: if the parameter we are plotting doesn't have a palette, make one
            # probably want to build palettes outside of this function?
            if param_name not in param_to_palette:
                # We may want to sort the input here?
                param_to_palette[param_name] = util_kwplot.Palette.coerce(macro_table[param_name].unique())

        snskw = {}
        if param_name in param_to_palette:
            snskw['palette'] = param_to_palette[param_name]

        s = plotter.plot_config.get('scatter.markersize', None)
        scatterkw = snskw.copy()
        if s is not None:
            scatterkw['s'] = s

        from geowatch.utils.util_kwplot import TitleBuilder
        title_builder = TitleBuilder()
        title_builder.add_part(f'Results (n={len(sub_macro_table)})')
        if [r for r in rois if r]:
            title_builder.ensure_newline()
            title_builder.add_part(f'Macro Analysis over {ub.urepr(rois, sv=1, nl=0)}')
        if anova_rank_p is not None:
            title_builder.ensure_newline()
            title_builder.append(f'Effect of {param_name}: anova_rank_p={concice_si_display(anova_rank_p)}')
        header_text = title_builder.finalize()

        param_valname_map, had_value_remap = shrink_param_names(param_name, list(param_histogram))
        print(f'had_value_remap={had_value_remap}')

        if param_name in plotter.param_to_valmap:
            had_value_remap = True
            # User can overload the mappign of the parameter value names
            param_valname_map.update(plotter.param_to_valmap[param_name])
            ...

        # Mapper for the legend
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

        vantage_prefix = f'macro_results_{rank:03d}_{param_name}'
        param_prefix = f'macro_results_{param_name}'
        param_metric_prefix = f'{param_prefix}_{main_metric}'
        param_metric2_prefix = f'{param_prefix}_{main_metric}_vs_{secondary_metric}'

        drawn_rows = []

        # SCATTER
        fig = kwplot.figure(fnum=4, doclf=True)
        # Scatter with legend
        ax = sns.scatterplot(data=sub_macro_table, x=x, y=y,
                             hue=param_name, legend=True, **scatterkw)
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

        param_dpath = (param_group_dpath / param_name).ensuredir().resolve()
        param_dpath_flat = (param_dpath / '_flat').ensuredir()

        param_fpath = param_dpath_flat / f'{param_metric2_prefix}_PLT01_scatter_legend.png'
        finalize_figure.finalize(fig, param_fpath)
        rich.print(f'[green] wrote {param_fpath.name}')
        drawn_rows.append({
            'type': 'vantage_metric_plot',
            'plot_type': 'scatter_legend',
            'param_fpath': param_fpath,
            'suffix': 'PLT01_scatter_legend.png',
            'prefix': param_metric2_prefix,
        })

        force_regen = True

        # Scatter legend (doesnt care about the vantage)
        param_fpath = param_dpath_flat / f'{param_prefix}_PLT03_scatter_onlylegend.png'
        try:
            if not param_fpath.exists() or force_regen:
                legend_ax = util_kwplot.extract_legend(ax)
                freq_mapper_scatter.relabel(legend_ax, ticks=False)
                finalize_figure.finalize(legend_ax.figure, param_fpath)
                drawn_rows.append({
                    'type': 'main_param_legend',
                    'param_fpath': param_fpath,
                    'suffix': 'PLT03_scatter_onlylegend.png',
                    'prefix': param_prefix,
                })
        except RuntimeError:
            rich.print(f'[red] failed to write {param_fpath.name}')
        else:
            rich.print(f'[green] wrote {param_fpath.name}')
            legend = ax.get_legend()
            if legend is not None:
                legend.remove()
        # Scatter without legend
        param_fpath = param_dpath_flat / f'{param_metric2_prefix}_PLT02_scatter_nolegend.png'
        finalize_figure.finalize(fig, param_fpath)
        rich.print(f'[green] wrote {param_fpath.name}')
        drawn_rows.append({
            'type': 'vantage_metric_plot',
            'plot_type': 'scatter_nolegend',
            'param_fpath': param_fpath,
            'suffix': 'PLT02_scatter_nolegend.png',
            'prefix': param_metric2_prefix,
        })

        # BOX
        param_fpath = param_dpath_flat / f'{param_metric_prefix}_PLT04_box.png'
        if not param_fpath.exists() or force_regen:
            fig = kwplot.figure(fnum=5, doclf=True)
            util_kwplot.fix_seaborn_palette_issue(x=param_name, snskw=snskw)
            try:
                ax = sns.boxplot(data=sub_macro_table, x=param_name, y=y,
                                 legend=False, **snskw)
            except Exception as box_ex:
                box_ex = box_ex
                rich.print(ub.codeblock(
                    f'''
                    [red]Error with boxplot

                    x={param_name}
                    y={y}

                    ex={box_ex}
                    '''))
            else:
                freq_mapper_box.relabel_xticks(ax)
                ax.set_title(header_text)
                modifier.relabel(ax, ticks=False)
                modifier.relabel_xticks(ax)
                finalize_figure.finalize(fig, param_fpath)
                rich.print(f'[green] wrote {param_fpath.name}')
                drawn_rows.append({
                    'plot_type': 'box',
                    'type': 'main_metric_plot',
                    'param_fpath': param_fpath,
                    'suffix': 'PLT04_box.png',
                    'prefix': param_metric_prefix,
                })

        # Output dataframe table for larger legend
        # Varied value table (doesn't care about the vantage)
        param_fpath = param_dpath_flat / f'{param_prefix}_PLT05_table.png'
        if not param_fpath.exists() or force_regen:
            # TODO: don't check for existence, check if we generated it
            # previously in this sessions (we want this to regenerate if
            # we modify code).
            param_code_lut = []
            for old_name, new_name in param_valname_map.items():
                param_code_lut.append({
                    'code': new_name,
                    'value': old_name,
                    'num': param_histogram[old_name],
                })
            param_code_lut = pd.DataFrame(param_code_lut, columns=['code', 'value', 'num'])
            if not had_value_remap:
                param_code_lut = param_code_lut.drop('code', axis=1)
            param_title = 'Key: ' + modifier._modify_text(param_name)
            lut_style = param_code_lut.style.set_caption(param_title)
            util_kwplot.dataframe_table(lut_style, param_fpath, title=param_title)
            rich.print(f'[green] wrote {param_fpath.name}')
            drawn_rows.append({
                'type': 'main_param_legend',
                'param_fpath': param_fpath,
                'suffix': 'PLT05_table.png',
                'prefix': param_prefix,
            })

        # Symlink to other organization structures
        for row in drawn_rows:
            param_fpath = row['param_fpath']
            suffix = row['suffix']
            prefix = row['prefix']
            vantage_fpath = vantage_flat_dpath / f'{vantage_prefix}_{suffix}'
            ub.symlink(real_path=param_fpath, link_path=vantage_fpath, overwrite=True)

            if row.get('plot_type') is not None:
                param_plot_type_dpath = (param_dpath / row['plot_type']).ensuredir()
                link_fpath = param_plot_type_dpath / f'{prefix}_{suffix}'
                ub.symlink(real_path=param_fpath, link_path=link_fpath, overwrite=True)

                vantage_plot_type_dpath = (vantage_dpath / row['plot_type']).ensuredir()
                link_fpath = vantage_plot_type_dpath / f'{vantage_prefix}_{suffix}'
                ub.symlink(real_path=param_fpath, link_path=link_fpath, overwrite=True)

            if row['type'] == 'main_param_legend':
                link_fpath = param_dpath / f'{prefix}_{suffix}'
                ub.symlink(real_path=param_fpath, link_path=link_fpath, overwrite=True)
            if row['type'] == 'vantage_metric_plot':
                ...
            if row['type'] == 'main_metric_plot':
                ...
        # ub.symlink(real_path=param_fpath, link_path=vantage_fpath, overwrite=True)
        return drawn_rows

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


def edit_distance(string1, string2):
    """
    Edit distance algorithm. String1 and string2 can be either
    strings or lists of strings

    Args:
        string1 (str | List[str]):
        string2 (str | List[str]):

    Requirements:
        pip install python-Levenshtein

    Returns:
        float | List[float] | List[List[float]]

    Example:
        >>> # xdoctest: +REQUIRES(module:Levenshtein)
        >>> string1 = 'hello world'
        >>> string2 = ['goodbye world', 'rofl', 'hello', 'world', 'lowo']
        >>> edit_distance(['hello', 'one'], ['goodbye', 'two'])
        >>> edit_distance('hello', ['goodbye', 'two'])
        >>> edit_distance(['hello', 'one'], 'goodbye')
        >>> edit_distance('hello', 'goodbye')
        >>> distmat = edit_distance(string1, string2)
        >>> result = ('distmat = %s' % (ub.repr2(distmat),))
        >>> print(result)
        >>> [7, 9, 6, 6, 7]
    """

    import Levenshtein
    isiter1 = ub.iterable(string1)
    isiter2 = ub.iterable(string2)
    strs1 = string1 if isiter1 else [string1]
    strs2 = string2 if isiter2 else [string2]
    distmat = [
        [Levenshtein.distance(str1, str2) for str2 in strs2]
        for str1 in strs1
    ]
    # broadcast
    if not isiter2:
        distmat = [row[0] for row in distmat]
    if not isiter1:
        distmat = distmat[0]
    return distmat


def suggest_did_you_mean(invalid_options, valid_choices):
    """
    Args:
        missing (List[str]): the invalid options the user chose
        valid_choices (List[str]): the valid available options

    Example:
        >>> from geowatch.mlops.aggregate_plots import *  # NOQA
        >>> invalid_options = ['fuber', 'yam', 'spam']
        >>> valid_choices = ['spam', 'ham', 'foobar']
        >>> suggest_did_you_mean(invalid_options, valid_choices)
    """
    import numpy as np
    import rich
    try:
        distances = np.array(edit_distance(invalid_options, valid_choices))
        for got, dists in zip(invalid_options, distances):
            min_idx = dists.argmin()
            if dists[min_idx] != 0:
                alternative = valid_choices[min_idx]
                rich.print(f'[yellow] Got: {got}. Did you mean {alternative}?')
    except ImportError:
        rich.print('[yellow] Warning: unable to suggest existing alternatives')


def shrink_param_names(param_name, param_values, text_len_thresh=20):
    param_labels = [str(p) for p in param_values]
    text_label_size = len(''.join(param_labels))
    if text_label_size > text_len_thresh:
        had_value_remap = True
        # Param names are too long. need to map parameter names to codes.
        param_valname_map = {}
        prefixchar = param_name.split('.')[-1][0].upper()
        for idx, value in enumerate(sorted(param_values)):
            old_name = str(value)
            new_name = f'{prefixchar}{idx:02d}'
            param_valname_map[old_name] = new_name
    else:
        had_value_remap = False
        param_valname_map = ub.dzip(param_labels, param_labels)
    return param_valname_map, had_value_remap


class SkipPlot(Exception):
    ...


class Vantage2(dict):
    """
    The primary and secondary metric to look at results with.
    This must have keys: metric1, metric2 and can optionally
    contain keys: scale1, scale2, objective1, and objective2.
    """

    @property
    def name(self):
        return self['name']

    @property
    def metric1(self):
        return self['metric1']

    @property
    def metric2(self):
        return self['metric2']

    @property
    def scale1(self):
        return self.get('scale1', 'linear')

    @property
    def scale2(self):
        return self.get('scale2', 'linear')


from dataclasses import dataclass  # NOQA


@dataclass
class Vantage:
    """
    The primary and secondary metric to look at results with.
    This must have keys: metric1, metric2 and can optionally
    contain keys: scale1, scale2, objective1, and objective2.

    Example:
        >>> from geowatch.mlops.aggregate_plots import *  # NOQA
        >>> Vantage('metrics.ppv', 'metrics.tpr')
    """
    metric1: str
    metric2: str
    scale1: str = 'linear'
    scale2: str = 'linear'
    objective1: str = 'maximize'
    objective2: str = 'maximize'
    name: str = None

    def __getitem__(self, key):
        return getattr(self, key)

    def __setitem__(self, key, value):
        return setattr(self, key, value)
