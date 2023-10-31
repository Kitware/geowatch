"""
Plotting logic for :class:`Aggregator`. These plots illustrate the relationshp
between parameters and metrics from various vantage points.

Used by ./aggregate.py
"""
import ubelt as ub

try:
    from xdev import profile
except ImportError:
    profile = ub.identity


def build_plotter(agg, rois, plot_config):
    from watch.mlops.smart_global_helper import SMART_HELPER
    from watch.utils import util_kwplot

    build_special_columns(agg)
    agg.build()
    single_table = table = agg.table
    single_table = preprocess_table_for_seaborn(agg, table)

    MARK_DELIVERED = plot_config.get('mark_delivered', False)
    if MARK_DELIVERED:
        SMART_HELPER.mark_delivery(single_table)

    modifier: util_kwplot.LabelModifier = SMART_HELPER.label_modifier()

    if rois is not None:
        agg.build_macro_tables(rois)
        macro_table = agg.region_to_tables[agg.primary_macro_region].copy()
        macro_table = preprocess_table_for_seaborn(agg, macro_table)

        if MARK_DELIVERED:
            SMART_HELPER.mark_delivery(macro_table)
        # if 0:
        #     SMART_HELPER.old_hacked_model_case(macro_table)
        param_to_palette = SMART_HELPER.shared_palettes(macro_table)
        if 0:
            SMART_HELPER.mark_star_models(macro_table)
    else:
        macro_table = None
        param_to_palette = SMART_HELPER.shared_palettes(single_table)

    # agg = plotter.agg
    plot_dpath = plot_config.get('plot_dpath', None)
    if plot_dpath is None:
        from watch.mlops.aggregate import hash_regions
        region_hash = hash_regions(rois)
        # agg_group_dpath = (agg.output_dpath / ('all_params' + ub.timestamp())).ensuredir()
        agg_group_dpath = (agg.output_dpath / (f'all_params-{region_hash}')).ensuredir()
    else:
        agg_group_dpath = ub.Path(plot_dpath).ensuredir()

    USE_EFFECTIVE = 1
    if USE_EFFECTIVE:
        # Relabel the resolved params to use the "effective-params"
        # instead.
        print(f'agg.mappings={agg.mappings}')
        for col, lut in agg.mappings.items():
            resolved_col = 'resolved_' + col
            for c in [col, resolved_col]:
                for table in [macro_table, single_table]:
                    if table is not None:
                        new = table[resolved_col].apply(lambda x: lut.get(x, x))
                        table[resolved_col] = new

    vantage_points = plot_config.get('vantage_points', None)
    plotter = ParamPlotter(agg, vantage_points=vantage_points)

    plotter.agg_group_dpath = agg_group_dpath
    plotter.param_to_palette = param_to_palette
    plotter.modifier = modifier
    plotter.macro_table = macro_table
    plotter.single_table = single_table
    plotter.rois = rois
    plotter.plot_config = plot_config
    return plotter


@profile
def build_all_param_plots(agg, rois, plot_config):
    """
    Main entry point for plotting results from an :class:`Aggregator`.
    """
    plotter = build_plotter(agg, rois, plot_config)
    plotter.plot_all()


def build_special_columns(agg):
    from watch.utils import util_pandas
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


def preprocess_table_for_seaborn(agg, table):
    fillna_cols = table.columns.intersection(agg.resolved_params.columns.union(agg.resolved_params.columns))
    table.loc[:, fillna_cols] = table.loc[:, fillna_cols].fillna('None')
    table = table.applymap(lambda x: str(x) if isinstance(x, list) else x)
    return table


class ParamPlotter:
    """
    Builds the scatter plots and barcharts over different params.
    Working on cleaning this up
    """
    def __init__(plotter, agg, vantage_points=None):
        from watch.mlops.smart_global_helper import SMART_HELPER
        plotter.agg = agg

        # We will conduct analysis under serveral different vantage points
        if vantage_points is None:
            vantage_points = SMART_HELPER.default_vantage_points(agg.type)

        for vantage in vantage_points:
            pm = vantage['metric1'].split('.')[-1]
            sm = vantage['metric2'].split('.')[-1]
            name = f'{pm}-vs-{sm}'
            vantage['name'] = name
        plotter.vantage_points = vantage_points

    def plot_resources(plotter):
        import rich
        from watch.utils import util_kwplot
        agg = plotter.agg
        resource_summary_df = agg.resource_summary_table()
        rich.print(resource_summary_df.to_string())
        resource_summary_df

        table = resource_summary_df
        table_title = 'resources'
        table_fpath = plotter.agg_group_dpath / f'{table_title}.png'
        table_style = table.style.set_caption(table_title)
        util_kwplot.dataframe_table(table_style, table_fpath, title=table_title)

    def plot_all(plotter):
        vantage = plotter.vantage_points[0]
        plot_config = plotter.plot_config

        from kwutil.util_progress import ProgressManager
        import rich
        pman = ProgressManager()

        rich.print(f'Dpath: [link={plotter.agg_group_dpath}]{plotter.agg_group_dpath}[/link]')
        with pman:
            if plot_config.get('plot_overviews', 1):
                for vantage in pman.progiter(plotter.vantage_points, desc='plotting vantage overviews'):
                    print('Plot vantage overview: ' + vantage['name'])
                    plotter.plot_vantage_overview(vantage)

            if plot_config.get('plot_params', 1):
                for vantage in pman.progiter(plotter.vantage_points, desc='plotting vantage params'):
                    print('Plot vantage params: ' + vantage['name'])
                    plotter.plot_vantage_params(vantage, pman=pman)

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
        import rich
        from watch.mlops.smart_global_helper import SMART_HELPER
        sns = kwplot.autosns()
        plt = kwplot.autoplt()  # NOQA
        kwplot.close_figures()

        agg = plotter.agg
        rois = plotter.rois
        macro_table = plotter.macro_table
        single_table = plotter.single_table

        modifier = plotter.modifier
        name = vantage['name']

        vantage_dpath2 = (plotter.agg_group_dpath / name).ensuredir()
        vantage_dpath = (plotter.agg_group_dpath).ensuredir()

        main_metric = y = vantage['metric1']
        yscale = vantage['scale1']
        x = vantage['metric2']
        xscale = vantage['scale2']

        region_attr = 'region_id'
        print('vantage = {}'.format(ub.urepr(vantage, nl=1)))
        print('main_metric = {}'.format(ub.urepr(main_metric, nl=1)))

        # main_metric = 'bas_poly_eval.metrics.bas_f1'
        # main_metric = 'bas_poly_eval.metrics.bas_faa_f1'
        # main_metric = agg.primary_metric_cols[0]

        finalize_figure = util_kwplot.FigureFinalizer(
            dpath=vantage_dpath,
            size_inches=np.array([6.4, 4.8]) * 1.0,
        )
        fig = kwplot.figure(fnum=2, doclf=True)

        unique_regions = single_table[region_attr].unique()
        print(f'unique_regions={unique_regions}')
        num_regions = len(unique_regions)
        print(f'num_regions={num_regions}')
        if num_regions < 10:
            colors = sns.color_palette(n_colors=num_regions)
        else:
            colors = kwimage.Color.distinct(num_regions, legacy=False)
            colors = [kwimage.Color.coerce(c).adjust(saturate=-0.3, lighten=-0.1).as01()
                      for c in kwimage.Color.distinct(num_regions, legacy=False)]
            # c = colors[0]
        regionid_to_color = ub.dzip(unique_regions, colors)
        snskw = {}
        snskw['palette'] = regionid_to_color

        ax = sns.scatterplot(data=single_table, x=x, y=y, hue=region_attr, legend=False, **snskw)

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
        modifier.relabel(ax, ticks=False)
        finalize_figure.finalize(fig, f'overview-{name}.png')
        rich.print(f'[green] made overview-{name}.png')
        # ax.set_xlim(0, np.quantile(agg.metrics[x], 0.99))
        # ax.set_xlim(1e-2, np.quantile(agg.metrics[x], 0.99))

        try:
            macro_fig_final = finalize_figure.copy()
            macro_fig_final.update(
                size_inches=[12, 4],
                dpi=300,
            )
            fig = kwplot.figure(fnum=90, doclf=True)

            region_order = plotter.plot_config.get('region_order', None)
            boxsns_kw = {}
            if region_order is not None:
                ...
                # known_regions = single_table[region_attr].unique()
                # # Todo: give the user control over x-axis order
                # # hack: put imerit regions last
                # orig_order = ub.oset(known_regions)
                # trailing_regions = [r for r in orig_order if '_C' in r]
                boxsns_kw['order'] = region_order
            boxsns_kw.update(snskw)

            ax = sns.boxplot(data=single_table, x=region_attr, y=main_metric, **boxsns_kw)
            ax.set_title(f'Per-Region Results (n={len(agg)})')
            param_histogram = single_table.groupby(region_attr).size().to_dict()
            util_kwplot.LabelModifier({
                param_value: f'{param_value}\n(n={num})'
                for param_value, num in param_histogram.items()
            }).relabel_xticks(ax)
            modifier.relabel(ax, ticks=False)
            macro_fig_final.finalize(fig, f'overview-boxplot-{region_attr}-vs-{main_metric}.png')

        except Exception as ex:
            rich.print(f'[yellow] warning, unable to plot overview-boxplot-region-vs-{main_metric}.png ex={ex}')
        else:
            rich.print(f'[green] made overview boxplot overview-boxplot-region-vs-{main_metric}.png')

        if macro_table is not None:
            fig = kwplot.figure(fnum=3, doclf=True)
            ax = fig.gca()
            region_ids = macro_table[region_attr].unique()
            assert len(region_ids) == 1
            macro_region_id = region_ids[0]
            palette = {
                macro_region_id: kwimage.Color('kitware_darkgray').as01()
            }
            ax = sns.scatterplot(data=macro_table, x=x, y=y, hue=region_attr, ax=ax, palette=palette)
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
            ax.set_title(f'Results (n={len(macro_table)})\n'
                         f'Macro Analysis over {ub.urepr(rois, sv=1, nl=0)}')
            ax.set_xscale(xscale)
            ax.set_yscale(yscale)
            roi_finalizer = finalize_figure.copy()
            roi_finalizer.update(
                dpath=vantage_dpath2,
            )
            modifier.relabel(ax, ticks=False)
            roi_finalizer.finalize(fig, f'overview-macro_results-{name}.png')
            # ax.set_xlim(1e-2, npe.quantile(agg.metrics[x].png')
            # ax.set_xlim(1e-2, npe.quantile(agg.metrics[x], 0.99))
            # ax.set_xlim(1e-2, 0.7)

    def plot_vantage_params(plotter, vantage, pman=None):
        import numpy as np
        import kwplot
        import kwarray
        import pandas as pd
        import rich
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
        resolved_params['param_hashid'] = macro_table['param_hashid']
        valid_cols = resolved_params.columns.difference(blocklist)
        resolved_params = resolved_params[valid_cols]

        from kwutil.util_yaml import Yaml
        params_of_interest = Yaml.coerce(plotter.plot_config.get('params_of_interest', None))

        if params_of_interest is not None:
            chosen_params = params_of_interest

            params_of_interest = set(params_of_interest)
            # if 'param_hashid' in params_of_interest:
            #     params_of_interest.remove(
            valid_params_of_interest = list(resolved_params.columns.intersection(params_of_interest))
            missing = sorted(set(params_of_interest) - set(valid_params_of_interest))
            chosen_params = valid_params_of_interest
            if missing:
                rich.print('[yellow]WARNING: unknown params of interest!')
                rich.print('missing: {}'.format(ub.repr2(missing)))
                print('chosen_params = {}'.format(ub.urepr(chosen_params, nl=1)))

                try:
                    distances = np.array(edit_distance(missing, resolved_params.columns))
                    for got, dists in zip(missing, distances):
                        alternative = resolved_params.columns[dists.argmin()]
                        rich.print(f'[yellow] Got: {got}. Did you mean {alternative}?')
                except ImportError:
                    ...

        # TODO: cleanup logic
        DO_STAT_ANALYSIS = plotter.plot_config.get('stats_ranking', False)
        if DO_STAT_ANALYSIS:
            ### Build param analysis
            from watch.utils import result_analysis
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
        if pman is None:
            pman = ProgressManager()
            pman.__enter__()
            owns_pman = 1
        try:
            for rank, param_name in enumerate(pman.progiter(chosen_params, desc='plot param for ' + vantage['name'], verbose=3)):
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
                    rich.print(f'[yellow] warning, unable to sort values by {param_name} ex={ex}')
                    ...

                # Number of samples we have for each value of this parameter
                param_histogram = ub.udict(macro_table.groupby(param_name).size().to_dict())
                param_histogram = param_histogram.map_keys(str)

                sub_macro_table = macro_table

                min_variations = plotter.plot_config.get('min_variations', 1)
                if min_variations > 1 and not (params_of_interest is not None and param_name in params_of_interest):
                    ignore_params = [k for k, v in param_histogram.items() if v < min_variations]
                    param_histogram = ub.udict(param_histogram) - set(ignore_params)
                    row_is_ignored = kwarray.isect_flags(macro_table[param_name], ignore_params)
                    sub_macro_table = macro_table[~row_is_ignored]
                    if len(param_histogram) == 1:
                        print('Skip plot')
                        continue
                    ...

                header_lines = [
                    f'Results (n={len(sub_macro_table)})',
                    f'Macro Analysis over {ub.urepr(rois, sv=1, nl=0)}',
                ]
                if anova_rank_p is not None:
                    header_lines.append(f'Effect of {param_name}: anova_rank_p={concice_si_display(anova_rank_p)}')
                header_text = '\n'.join(header_lines)

                param_dpath = (param_group_dpath / param_name).ensuredir().resolve()

                param_valname_map, had_value_remap = shrink_param_names(param_name, list(param_histogram))

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

                # Scatter with legend
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

                # Scatter legend (doesnt care about the vantage)
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
                    legend = ax.get_legend()
                    if legend is not None:
                        legend.remove()
                # Scatter without legend
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
                    try:
                        ax = sns.boxplot(data=sub_macro_table, x=param_name, y=y, **snskw)
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
                if param_fpath.exists():
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
                    param_code_lut = pd.DataFrame(param_code_lut, columns=['code', 'value', 'num'])
                    if not had_value_remap:
                        param_code_lut = param_code_lut.drop('code', axis=1)
                    param_title = 'Key: ' + modifier._modify_text(param_name)
                    lut_style = param_code_lut.style.set_caption(param_title)
                    util_kwplot.dataframe_table(lut_style, param_fpath, title=param_title)
                ub.symlink(real_path=param_fpath, link_path=vantage_fpath, overwrite=True)

            rich.print(f'Dpath: [link={plotter.agg_group_dpath}]{plotter.agg_group_dpath}[/link]')
            # agg0.analyze()
        finally:
            if owns_pman:
                pman.__exit__()


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
