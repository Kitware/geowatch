# import math
# import pandas as pd
import ubelt as ub
# from watch.mlops.aggregate import hash_param
# from watch.mlops.aggregate import fix_duplicate_param_hashids
# from watch.utils import util_pandas


def _sitevisit_2023_july_report():
    import watch
    from watch.mlops.aggregate import AggregateLoader
    import pandas as pd
    from watch.utils.util_pandas import DotDictDataFrame
    expt_dvc_dpath = watch.find_dvc_dpath(tags='phase2_expt', hardware='auto')

    load_kwargs = {
        'target': [
            expt_dvc_dpath / 'aggregate_results/dzyne/bas_poly_eval_2023-07-10T131639-5.csv.zip',
            expt_dvc_dpath / 'aggregate_results/dzyne/bas_poly_eval_2023-07-10T164254-5.csv.zip',
            expt_dvc_dpath / 'aggregate_results/dzyne/sv_poly_eval_2023-07-10T164254-5.csv.zip',
            expt_dvc_dpath / 'aggregate_results/horologic/bas_poly_eval_2023-07-10T155903-5.csv.zip',
            expt_dvc_dpath / 'aggregate_results/horologic/sv_poly_eval_2023-07-10T155903-5.csv.zip',
            expt_dvc_dpath / 'aggregate_results/namek/bas_poly_eval_2023-04-19T113433-5.csv.zip',
            expt_dvc_dpath / 'aggregate_results/namek/bas_poly_eval_2023-07-10T161857-5.csv.zip',
            expt_dvc_dpath / 'aggregate_results/namek/bas_pxl_eval_2023-04-19T113433-5.csv.zip',
            expt_dvc_dpath / 'aggregate_results/namek/bas_pxl_eval_2023-07-10T161857-5.csv.zip',
            expt_dvc_dpath / 'aggregate_results/namek/sv_poly_eval_2023-07-10T161857-5.csv.zip',
            expt_dvc_dpath / 'aggregate_results/toothbrush/bas_poly_eval_2023-04-19T105718-5.csv.zip',
            expt_dvc_dpath / 'aggregate_results/toothbrush/bas_poly_eval_2023-07-10T150132-5.csv.zip',
            expt_dvc_dpath / 'aggregate_results/toothbrush/bas_pxl_eval_2023-04-19T105718-5.csv.zip',
            expt_dvc_dpath / 'aggregate_results/toothbrush/bas_pxl_eval_2023-07-10T150132-5.csv.zip',
            expt_dvc_dpath / 'aggregate_results/toothbrush/sv_poly_eval_2023-04-19T105718-5.csv.zip',
            expt_dvc_dpath / 'aggregate_results/toothbrush/sv_poly_eval_2023-07-10T150132-5.csv.zip',
            expt_dvc_dpath / 'aggregate_results/uconn/COLD_candidates_0705.zip',
            expt_dvc_dpath / 'aggregate_results/wu/bas_pxl_eval_2023-07-11T180910+0.csv.zip',
            expt_dvc_dpath / 'aggregate_results/wu/bas_poly_eval_2023-07-11T180910+0.csv.zip',
            expt_dvc_dpath / 'aggregate_results/wu/bas_pxl_eval_2023-07-11T181515+0.csv.zip',
            expt_dvc_dpath / 'aggregate_results/wu/bas_poly_eval_2023-07-11T181515+0.csv.zip',
            expt_dvc_dpath / 'aggregate_results/wu/bas_pxl_eval_2023-07-11T213433+0.csv.zip',
            expt_dvc_dpath / 'aggregate_results/wu/bas_poly_eval_2023-07-11T213433+0.csv.zip',
            expt_dvc_dpath / 'aggregate_results/connor/bas_poly_eval_2023-07-11T134348-5.csv.zip',
            expt_dvc_dpath / 'aggregate_results/connor/bas_pxl_eval_2023-07-11T134348-5.csv.zip',
        ],
        'pipeline': 'bas_building_and_depth_vali',
        'io_workers': 'avail',
    }
    with ub.Timer('load'):
        loader = AggregateLoader(**load_kwargs)
        eval_type_to_agg = loader.coerce_aggregators()

    agg0 = eval_type_to_agg['bas_poly_eval']

    from watch.mlops.smart_global_helper import SMART_HELPER
    SMART_HELPER.populate_test_dataset_bundles(agg0)

    rois = [
        'KR_R002',
        'NZ_R001', 'CH_R001',
        'KR_R001',
        'AE_R001',
        # 'PE_R001',
        # 'BR_R004'
    ]
    multi_rois = [
        ['KR_R002'],
        ['PE_R001'],
        ['NZ_R001'],
        ['CH_R001'],
        ['KR_R002', 'NZ_R001', 'CH_R001', 'KR_R001', 'AE_R001', 'BR_R002'],
        ['KR_R002', 'NZ_R001', 'CH_R001', 'KR_R001', 'AE_R001', 'BR_R002', 'PE_R001', 'BR_R004'],
    ]
    # agg0.build_macro_tables(rois)

    # Filter to experiments only on a particualr bundle
    flags = agg0.table['params.bas_pxl.test_dataset_bundle'] == 'Drop7-MedianNoWinter10GSD'
    subagg = agg0.filterto(index=flags[flags].index)
    subagg.output_dpath = (ub.Path.home() / 'site_visit_2023-07-13').ensuredir()
    subagg.build_macro_tables(rois=rois, average='mean')

    # subagg.build_macro_tables(average='gmean')

    subagg.report_best(top_k=10, print_models=1, reference_region='final')
    SMART_HELPER.print_minmax_times(subagg.table)

    resources = subagg.resource_summary_table()
    from watch.utils import util_kwplot
    util_kwplot.dataframe_table(resources, 'resource_summary_bas.png')

    import kwplot
    sns = kwplot.autosns()
    plt = kwplot.autoplt()  # NOQA
    kwplot.close_figures()

    from watch.mlops.aggregate import build_special_columns, preprocess_table_for_seaborn
    build_special_columns(subagg)
    # single_table = preprocess_table_for_seaborn(subagg, subagg.table)

    dpath = (subagg.output_dpath / 'coarse_channels').ensuredir()
    import numpy as np
    finalize_figure = util_kwplot.FigureFinalizer(
        dpath=dpath,
        size_inches=np.array([6.4, 4.8]) * 1.3,
    )
    modifier = SMART_HELPER.label_modifier()

    new_columns = SMART_HELPER.custom_channel_relabel(subagg.table, channel_key='resolved_params.bas_pxl.channels', coarsen=True)

    # param_name = 'bas_pxl.channels'
    # param_presenter = ParamPresenter(param_name, new_columns.unique())

    param_name = 'resolved_params.bas_pxl_fit.init'
    agg0.table[param_name].unique()
    subagg.table[param_name]
    param_presenter = ParamPresenter(param_name, new_columns.unique())

    snskw = {}
    # import kwimage
    param_presenter.param_palette['raw'] = (0.1, 0.1, 0.1)
    snskw['palette'] = param_presenter.param_palette

    ### PER REGION PLOTS
    # make_per_region_channel_sitevisit_plots(subagg)
    for rois in multi_rois:
        ...
        subagg.build_macro_tables(rois=rois, average='gmean')
        macro_table = subagg.region_to_tables[subagg.primary_macro_region].copy()
        macro_table = DotDictDataFrame(macro_table)
        print(f'subagg.primary_macro_region={subagg.primary_macro_region}')
        macro_table = preprocess_table_for_seaborn(subagg, macro_table)
        # param_to_palette = SMART_HELPER.shared_palettes(macro_table)

        sub_macro_table = macro_table
        sub_macro_table = sub_macro_table[sub_macro_table['resolved_params.bas_pxl.channels'] != 'None']
        sub_macro_table = sub_macro_table[~sub_macro_table['resolved_params.bas_pxl.channels'].isna()]

        new_columns = SMART_HELPER.custom_channel_relabel(sub_macro_table, channel_key='resolved_params.bas_pxl.channels', coarsen=True)
        param_name = 'bas_pxl.channels'
        sub_macro_table['bas_pxl.channels'] = new_columns

        y = 'metrics.bas_poly_eval.bas_f1'
        x = 'metrics.bas_poly_eval.bas_ffpa'

        if 0:
            y_threshold = 0.1
            sub_macro_table = SMART_HELPER.threshold_param_groups(sub_macro_table, param_name, y, y_threshold)

        MARK_DELIVERED = 1
        if MARK_DELIVERED:
            SMART_HELPER.mark_delivery(sub_macro_table, include={'Baseline2023-07'})
            baseline_param_name = sub_macro_table[sub_macro_table['delivered_params'] == 'Baseline2023-07'][param_name].iloc[0]
            # Color to match the baseline parameter
            val_to_color = ub.udict(SMART_HELPER.delivery_to_color)
            val_to_color['Baseline2023-07'] = param_presenter.param_palette[baseline_param_name]

        fig = kwplot.figure(fnum=1, doclf=True)
        ax = sns.scatterplot(data=sub_macro_table, x=x, y=y,
                             hue=param_name,
                             legend=True, **snskw)
        ax.set_xscale('log')
        ax.set_ylim(0, 1)

        if 'delivered_params' in sub_macro_table:
            from watch.utils.util_kwplot import scatterplot_highlight
            # scatterplot_highlight(data=sub_macro_table, x=x, y=y, highlight='is_star', ax=ax, size=300)
            val_to_color = val_to_color & set(sub_macro_table['delivered_params'].unique())
            scatterplot_highlight(data=sub_macro_table, x=x, y=y,
                                  highlight='delivered_params', ax=ax,
                                  color='group',
                                  size=300, val_to_color=val_to_color, linewidths=2.0)
            # linewidths=1.5)
            # mytable = sub_macro_table[sub_macro_table['delivered_params'] == 'Baseline2023-07']
            # from watch.utils.result_analysis import varied_value_counts
            # zzz = varied_value_counts(mytable, min_variations=2, dropna=True)
        ax.set_title(f'BAS Results (n={len(sub_macro_table)})\n'
                     f'Macro Analysis over {ub.urepr(rois, sv=1, nl=0)}')
        modifier.relabel(ax, ticks=False)
        modifier.relabel_xticks(ax)
        from watch.mlops.aggregate import hash_regions
        macro_key = hash_regions(rois)
        finalize_figure(fig, dpath / macro_key + '.png')

    fig = kwplot.figure(fnum=2, doclf=True)
    y = 'metrics.bas_poly_eval.bas_f1'
    orig_x = 'context.bas_poly_eval.stop_timestamp'
    fixed_x = 'bas_poly_eval.stop_timestamp'
    sub_macro_table[fixed_x] = util_kwplot.fix_matplotlib_dates(
        sub_macro_table[orig_x], format='datetime')
    x = fixed_x

    ax = sns.scatterplot(data=sub_macro_table, x=x, y=y,
                         hue='machine.bas_poly.user',
                         legend=True)
    ax.set_ylim(0, 1)
    ax.set_title(f'BAS Results (n={len(sub_macro_table)})\n'
                 f'Macro Analysis over {ub.urepr(rois, sv=1, nl=0)}')

    relabelers, param_code_lut = build_param_relabelers(sub_macro_table, param_name)
    # freq_mapper_scatter = relabelers['scatter']
    freq_mapper_box = relabelers['box']

    param_histogram = ub.udict(sub_macro_table.groupby(param_name).size().to_dict())
    param_histogram = param_histogram.map_keys(str)
    param_presenter = ParamPresenter(param_name, param_histogram)
    param_valname_map = param_presenter.param_valname_map
    had_value_remap = param_presenter.had_value_remap

    sub_macro_table = sub_macro_table.sort_values(param_name)

    snskw = {}
    # if param_name in param_to_palette:
    #     snskw['palette'] = param_to_palette[param_name]

    modifier = SMART_HELPER.label_modifier()

    header_lines = [
        f'BAS Results (n={len(sub_macro_table)})',
        f'Macro Analysis over {ub.urepr(rois, sv=1, nl=0)}',
    ]
    header_text = '\n'.join(header_lines)

    #### BOX BOX BOX
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
    param_code_lut = param_code_lut.set_index('value')

    param_stats = sub_macro_table.groupby(param_name)[[y]].describe()
    param_stats = param_stats.droplevel(0, axis=1)
    statcols = ['max', '50%', 'mean']
    param_code_lut[statcols] = param_stats.loc[param_code_lut.index, statcols]
    param_code_lut = param_code_lut.reset_index()
    param_code_lut = param_code_lut.set_index('code')
    param_code_lut = param_code_lut.sort_values('max', ascending=False)

    fig = kwplot.figure(fnum=5, doclf=True)  # NOQA
    ax = sns.boxplot(data=sub_macro_table, x=param_name, y=y, **snskw)
    freq_mapper_box.relabel_xticks(ax)
    ax.set_title(header_text)
    modifier.relabel(ax, ticks=False)
    modifier.relabel_xticks(ax)

    param_title = 'Key: ' + modifier._modify_text(param_name)
    lut_style = param_code_lut.style.set_caption(param_title)
    param_fpath = 'param_lut.png'
    util_kwplot.dataframe_table(lut_style, param_fpath, title=param_title, show='eog')


class ParamPresenter:
    """
    Helper to determine reasonable ways to simplify how we will present a
    parameter. This includes mapping it to a short code if needed and building
    a color pallet.

    Helper to shrink parameters to a consistent set of codes if needed.
    """

    def __init__(self, param_name, param_values, text_len_thresh=20):
        self.param_name = param_name
        self.param_values = param_values
        self.text_len_thresh = text_len_thresh
        self.param_valname_map = None
        self.had_value_remap = None
        self.param_palette = None
        self._build()

    def _build(self):
        from watch.mlops.aggregate import shrink_param_names
        param_valname_map, had_value_remap = shrink_param_names(
            self.param_name, self.param_values, text_len_thresh=self.text_len_thresh)
        self.param_valname_map = param_valname_map
        self.had_value_remap = had_value_remap
        from watch.mlops.smart_global_helper import SMART_HELPER
        self.param_palette = SMART_HELPER.make_param_palette(self.param_values)

    def build_lut(self, param_histogram=None):
        import pandas as pd
        param_code_lut = []
        for old_name, new_name in self.param_valname_map.items():
            row = {
                'code': new_name,
                'value': old_name,
            }
            if param_histogram is not None:
                if old_name not in param_histogram:
                    continue
                row['num'] = param_histogram[old_name]
            param_code_lut.append(row)
        param_code_lut = pd.DataFrame(param_code_lut)
        if not self.had_value_remap:
            param_code_lut = param_code_lut.drop('code', axis=1)
        return param_code_lut


def build_param_relabelers(table, param_name, y, param_presenter):
    # from watch.mlops.aggregate import shrink_param_names
    # import pandas as pd
    from watch.utils import util_kwplot

    param_table_groups = table.groupby(param_name)

    # Determine the max score each parameter achieved (and other stats).
    # TODO: maximize or minimize
    param_stats = param_table_groups[y].describe()
    statcols = ['max', '50%', 'mean']
    # param_ranking = param_stats['max'].sort_values(ascending=False).index

    if 1:
        known = set(param_presenter.param_values)
        assert known.issuperset(set(param_stats.index))

    # Count the number of time each parameter value occurred
    param_histogram = ub.udict(param_table_groups.size().to_dict())
    param_histogram = param_histogram.map_keys(str)

    param_code_lut = param_presenter.build_lut(param_histogram)

    param_code_lut = param_code_lut.set_index('value')
    param_code_lut[statcols] = param_stats.loc[param_code_lut.index, statcols]
    param_code_lut = param_code_lut.reset_index()
    param_code_lut = param_code_lut.set_index('code')
    param_code_lut = param_code_lut.sort_values('max', ascending=False)

    # Mapper for the scatterplot legend
    if param_presenter.had_value_remap:
        freq_mapper_scatter = util_kwplot.LabelModifier({
            param_value: f'{param_value}\n{param_presenter.param_valname_map[param_value]} (n={num})'
            for param_value, num in param_histogram.items()
        })
    else:
        freq_mapper_scatter = util_kwplot.LabelModifier({
            param_value: f'{param_value}\n(n={num})'
            for param_value, num in param_histogram.items()
        })

    freq_mapper_box = util_kwplot.LabelModifier({
        param_value: f'{param_presenter.param_valname_map[param_value]}\n(n={num})'
        for param_value, num in param_histogram.items()
    })

    relabelers = {
        'scatter': freq_mapper_scatter,
        'box': freq_mapper_box,
    }
    return relabelers, param_code_lut


def make_per_region_channel_sitevisit_plots(subagg):
    from watch.mlops.smart_global_helper import SMART_HELPER
    import kwplot
    from watch.utils import util_kwplot
    import pandas as pd
    sns = kwplot.autosns()
    import numpy as np

    # x = 'metrics.bas_poly_eval.bas_ffpa'
    y = 'metrics.bas_poly_eval.bas_f1'

    sub_table = subagg.table
    sub_table = sub_table[sub_table['resolved_params.bas_pxl.channels'] != 'None']
    sub_table = sub_table[~sub_table['resolved_params.bas_pxl.channels'].isna()]

    new_columns = SMART_HELPER.custom_channel_relabel(sub_table, channel_key='resolved_params.bas_pxl.channels')
    param_name = 'bas_pxl.channels'
    sub_table['bas_pxl.channels'] = new_columns

    # Filter out of channels that never score above a threshold
    if False:
        y_threshold = 0.5
        param_groups = sub_table.groupby(param_name)
        passed_thresh = param_groups[y].describe()['max'] > y_threshold
        sub_table = pd.concat(list((ub.udict(list(param_groups)) & passed_thresh[passed_thresh].index).values()))

    if True:
        # Filter to regions with enough data
        region_to_table = dict(list(sub_table.groupby('region_id')))
        inspect_regions = {k for k, v in ub.udict(region_to_table).map_values(len).items() if v > 10}
        sub_table = pd.concat(list((ub.udict(region_to_table) & inspect_regions).values()))

    dpath = (subagg.output_dpath / 'per_region_channels').ensuredir()
    finalize_figure = util_kwplot.FigureFinalizer(
        dpath=dpath,
        size_inches=np.array([6.4, 4.8]) * 1.3,
    )

    modifier = SMART_HELPER.label_modifier()

    # Get a global set of relabels for consistency
    param_values = np.unique(sub_table[param_name].values)
    param_presenter = ParamPresenter(param_name, param_values)
    snskw = {}
    if param_presenter.param_palette is not None:
        snskw['palette'] = param_presenter.param_palette

    region_to_table = dict(list(sub_table.groupby('region_id')))

    relevant_indexes = []

    region_to_top_params = {}

    for region_id in inspect_regions:
        table = region_to_table[region_id]

        rois = [region_id]

        header_lines = [
            f'BAS Results (n={len(table)})',
            f'Macro Analysis over {ub.urepr(rois, sv=1, nl=0)}',
        ]
        header_text = '\n'.join(header_lines)

        # plot_name = region_id + '_scatter'
        # fig = kwplot.figure(fnum=1, doclf=True)
        # ax = fig.gca()
        # ax = sns.scatterplot(data=table, x=x, y=y,
        #                      hue=param_name,
        #                      legend=True)
        # ax.set_xscale('log')
        # ax.set_ylim(0, 1)
        # ax.set_title(header_text)
        # finalize_figure(fig, plot_name + '.png')

        fig = kwplot.figure(fnum=5, doclf=True)
        ax = fig.gca()

        relabelers, param_code_lut = build_param_relabelers(table, param_name, y, param_presenter)
        param_ranking = param_code_lut.loc[param_code_lut['max'].sort_values(ascending=False).index]['value'].values

        top_k = 5
        param_ranking = param_ranking[0:top_k]
        region_to_top_params[region_id] = param_ranking

        reordered_table = pd.concat(list(ub.udict(list(table.groupby(param_name))).take(param_ranking)))

        relevant_indexes.append(reordered_table.index)

        ax = sns.boxplot(data=reordered_table, x=param_name, y=y, ax=ax, **snskw)
        ax.set_ylim(0, 1)

        freq_mapper_box = relabelers['box']
        freq_mapper_box.relabel_xticks(ax)
        ax.set_title(header_text)
        modifier.relabel(ax, ticks=False)
        modifier.relabel_xticks(ax)
        finalize_figure(fig, region_id + '_box.png')

        param_title = 'Key: ' + modifier._modify_text(param_name)
        lut_style = param_code_lut.style.set_caption(param_title)

        param_fpath = dpath / region_id + '_param_lut.png'

        util_kwplot.dataframe_table(lut_style, param_fpath, title=param_title)

        # show='eog')
        # break

    # votes = ub.dict_hist(list(ub.flatten(region_to_top_params.values())))
    param_code_lut = param_presenter.build_lut()

    # subsub_agg = subagg.filterto(sub_table.index)
    rois = {
        # 'AE_R001',
        # 'BR_R001',
        # 'BR_R002',
        # 'BR_R004',
        'CH_R001',
        'KR_R001',
        'KR_R002',
        'NZ_R001',
        # 'PE_R001',
        # 'US_R001'
    }
    subagg.build_macro_tables(rois=rois, average='mean')

    macro_table = subagg.region_to_tables[subagg.primary_macro_region]

    sub_macro_table = macro_table
    sub_macro_table = sub_macro_table[sub_macro_table['resolved_params.bas_pxl.channels'] != 'None']
    sub_macro_table = sub_macro_table[~sub_macro_table['resolved_params.bas_pxl.channels'].isna()]

    new_columns = SMART_HELPER.custom_channel_relabel(sub_macro_table, channel_key='resolved_params.bas_pxl.channels')
    param_name = 'bas_pxl.channels'
    sub_macro_table['bas_pxl.channels'] = new_columns

    fig = kwplot.figure(fnum=1, doclf=True)
    y = 'metrics.bas_poly_eval.bas_f1'
    x = 'metrics.bas_poly_eval.bas_ffpa'
    ax = sns.scatterplot(data=sub_macro_table, x=x, y=y,
                         hue=param_name,
                         legend=True)
    # **snskw)
    ax.set_xscale('log')
    ax.set_ylim(0, 1)
    ax.set_title(f'BAS Results (n={len(sub_macro_table)})\n'
                 f'Macro Analysis over {ub.urepr(rois, sv=1, nl=0)}')

    fig = kwplot.figure(fnum=2, doclf=True)
    y = 'metrics.bas_poly_eval.bas_f1'
    orig_x = 'context.bas_poly_eval.stop_timestamp'
    fixed_x = 'bas_poly_eval.stop_timestamp'
    sub_macro_table[fixed_x] = util_kwplot.fix_matplotlib_dates(
        sub_macro_table[orig_x], format='datetime')
    x = fixed_x

    ax = sns.scatterplot(data=sub_macro_table, x=x, y=y,
                         # hue='machine.bas_poly.user',
                         hue=param_name,
                         legend=False)
    ax.set_ylim(0, 1)
    ax.set_title(f'BAS Results (n={len(sub_macro_table)})\n'
                 f'Macro Analysis over {ub.urepr(rois, sv=1, nl=0)}')
