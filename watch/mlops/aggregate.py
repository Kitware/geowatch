r"""
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

    # BAS
    DVC_DATA_DPATH=$(smartwatch_dvc --tags='phase2_data' --hardware=auto)
    DVC_EXPT_DPATH=$(smartwatch_dvc --tags='phase2_expt' --hardware=auto)
    XDEV_PROFILE=1 python -m watch.mlops.aggregate \
        --pipeline=bas \
        --io_workers=0 \
        --root_dpath="$DVC_EXPT_DPATH/_timekernel_test_drop4"

"""
import kwarray
import math
import ubelt as ub
from watch.utils import util_pandas
from typing import Dict, Any
import pandas as pd
from scriptconfig import DataConfig, Value as _V
from watch.mlops.aggregate_loader import build_tables
from watch.mlops.smart_global_helper import SMART_HELPER


class AggregateEvluationConfig(DataConfig):
    """
    Aggregates results from multiple DAG evaluations.
    """
    root_dpath   = _V(None, help='The mlops output directory used in schedule evaluation')
    pipeline     = _V('joint_bas_sc', help='the name of the pipeline to run')
    io_workers   = _V('avail', help='number of processes to load results')
    freeze_cache = _V(False, help='set to a specific cache string to freeze a cache with the current results')


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
        >>>     'io_workers': 10,
        >>>     'freeze_cache': 0,
        >>>     # 'pipeline': 'joint_bas_sc_nocrop',
        >>>     # 'root_dpath': expt_dvc_dpath / '_testsc',
        >>>     #'pipeline': 'sc',
        >>> }

        config = AggregateEvluationConfig.legacy(cmdline=cmdline, data=kwargs)
        agg_dpath = ub.Path(config['root_dpath']) / 'aggregate'

        eval_type_to_results = build_tables(config)
        eval_type_to_aggregator = build_aggregators(eval_type_to_results, agg_dpath)
        agg = ub.peek(eval_type_to_aggregator.values())
        agg = eval_type_to_aggregator.get('bas_poly_eval', None)

        >>> ## Execute
        >>> main(cmdline=cmdline, **kwargs)
    """
    config = AggregateEvluationConfig.legacy(cmdline=cmdline, data=kwargs)

    agg_dpath = ub.Path(config['root_dpath']) / 'aggregate'
    cache_dpath = agg_dpath / '_cache'

    cacher = ub.Cacher(
        # Caching may be important depending on how much data we need to load
        'table_cacher', dpath=cache_dpath, depends=dict(config),
        enabled=config['freeze_cache'],
        verbose=3,
    )
    tables = cacher.tryload()
    if tables is None:
        eval_type_to_results = build_tables(config)
        cacher.save(eval_type_to_results)

    eval_type_to_aggregator = build_aggregators(eval_type_to_results, agg_dpath)

    # automated_analysis(eval_type_to_aggregator, config)
    agg = eval_type_to_aggregator.get('bas_poly_eval', None)
    # for agg in eval_type_to_aggregator.values():
    if agg is not None:
        rois = {'KR_R001', 'KR_R002', 'BR_R002'}
        # rois = {'KR_R001', 'KR_R002'}
        build_all_param_plots(agg, rois, config)
        ...


def build_all_param_plots(agg, rois, config):
    agg_dpath = ub.Path(config['root_dpath'] / 'aggregate')
    agg.agg_dpath = agg_dpath

    # Hack in fit params
    if 0:
        resolved_params = pd.concat([
            agg.resolved_info['resolved_params'],
            agg.resolved_info['fit_params']], axis=1)
        agg.resolved_info['resolved_params'] = resolved_params
        agg.resolved_params = resolved_params

    resolved_params = util_pandas.DotDictDataFrame(agg.results['resolved_params'])

    part1 = resolved_params.query_column('batch_size')
    part2 = resolved_params.query_column('accumulate_grad_batches')
    prefix_to_batchsize = ub.group_items(part1, key=lambda x: x.rsplit('.', 1)[0])
    prefix_to_accumbatch = ub.group_items(part2, key=lambda x: x.rsplit('.', 1)[0])

    for prefix in set(prefix_to_batchsize) | set(prefix_to_accumbatch):
        cols1 = prefix_to_batchsize.get(prefix, None)
        cols2 = prefix_to_accumbatch.get(prefix, None)
        assert len(cols1) == 1
        val_accum = 1
        val_bsize = resolved_params[cols1[0]]
        if cols2 is not None:
            assert len(cols2) == 1
            val_accum = resolved_params[cols2[0]]
        val_effective_bsize = val_bsize * val_accum
        resolved_params[prefix + '.effective_batch_size'] = val_effective_bsize

    agg.build_macro_tables(rois)

    macro_results = agg.region_to_tables[agg.primary_macro_region].copy()
    single_results = agg.results

    macro_table = pd.concat(list(macro_results.values()), axis=1)
    single_table = pd.concat(list(single_results.values()), axis=1)

    single_table.loc[:, agg.resolved_params.columns] = single_table.loc[:, agg.resolved_params.columns].fillna('None')
    macro_table.loc[:, agg.resolved_params.columns] = macro_table.loc[:, agg.resolved_params.columns].fillna('None')

    macro_table = macro_table.applymap(lambda x: str(x) if isinstance(x, list) else x)
    single_table = single_table.applymap(lambda x: str(x) if isinstance(x, list) else x)

    if 0:
        SMART_HELPER.old_hacked_model_case(macro_table)

    modifier = SMART_HELPER.label_modifier()
    param_to_palette = SMART_HELPER.shared_palletes(macro_table)
    if 0:
        SMART_HELPER.mark_star_models(macro_table)

    # agg = plotter.agg
    agg_group_dpath = (agg.agg_dpath / ('all_params' + ub.timestamp())).ensuredir()

    plotter = ParamPlotter(agg)

    plotter.macro_results = macro_results
    plotter.agg_group_dpath = agg_group_dpath
    plotter.param_to_palette = param_to_palette
    plotter.modifier = modifier
    plotter.macro_table = macro_table
    plotter.single_table = single_table
    plotter.rois = rois

    for vantage in plotter.vantage_points:
        print(vantage['name'])
        # plotter.plot_vantage(vantage)
        # plotter.plot_vantage_overview(vantage)
        plotter.plot_vantage_params(vantage)


class ParamPlotter:
    """
    Builds the scatter plots and barcharts over different params.
    Working in cleaning this up
    """
    def __init__(plotter, agg):
        plotter.agg = agg

        # We will conduct analysis under serveral different vantage points
        vantage_points = SMART_HELPER.default_vantage_points()
        for vantage in vantage_points:
            pm = vantage['metric1'].split('.')[-1]
            sm = vantage['metric2'].split('.')[-1]
            name = f'{pm}-vs-{sm}'
            vantage['name'] = name
        plotter.vantage_points = vantage_points

    def plot_vantage(plotter, vantage):
        plotter.plot_vantage_overview(vantage)
        plotter.plot_vantage_params(vantage)

    def plot_vantage_overview(plotter, vantage):
        from watch.utils import util_kwplot
        import numpy as np
        import kwplot
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
        main_metric = 'bas_poly_eval.metrics.bas_faa_f1'

        finalize_figure = util_kwplot.FigureFinalizer(
            dpath=vantage_dpath,
            size_inches=np.array([6.4, 4.8]) * 1.0,
        )

        fig = kwplot.figure(fnum=2, doclf=True)
        ax = sns.scatterplot(data=single_table, x=x, y=y, hue='region_id')
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
        except Exception:
            ...

        from watch.utils.util_kwplot import scatterplot_highlight
        fig = kwplot.figure(fnum=3, doclf=True)
        ax = fig.gca()
        ax = sns.scatterplot(data=macro_table, x=x, y=y, hue='region_id', ax=ax)
        if 'is_star' in macro_table:
            scatterplot_highlight(data=macro_table, x=x, y=y, highlight='is_star', ax=ax, size=300)
        ax.set_title(f'BAS Results (n={len(macro_table)})\n'
                     f'Macro Analysis over {ub.urepr(rois, sv=1, nl=0)}')
        ax.set_xscale(xscale)
        ax.set_yscale(yscale)
        finalize_figure.finalize(fig, 'macro_results.png')
        # ax.set_xlim(1e-2, npe.quantile(agg.metrics[x], 0.99))
        # ax.set_xlim(1e-2, 0.7)

    def plot_vantage_params(plotter, vantage):
        from watch.utils import util_kwplot
        import numpy as np
        import kwplot
        from watch.utils.util_kwplot import scatterplot_highlight

        sns = kwplot.autosns()
        plt = kwplot.autoplt()  # NOQA
        kwplot.close_figures()

        rois = plotter.rois
        macro_table = plotter.macro_table
        macro_results = plotter.macro_results

        modifier = plotter.modifier
        param_to_palette = plotter.param_to_palette

        param_group_dpath = plotter.agg_group_dpath / 'params'
        vantage_dpath = (plotter.agg_group_dpath / vantage['name']).ensuredir()

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

        DO_STAT_ANALYSIS = True
        if DO_STAT_ANALYSIS:
            ### Build param analysis
            from watch.utils import result_analysis
            results = {'params': macro_table[macro_results['resolved_params'].columns],
                       'metrics': macro_table[macro_results['metrics'].columns]}
            # agg.primary_metric_cols)
            analysis = result_analysis.ResultAnalysis(
                results, metrics=[main_metric], metric_objectives=metric_objectives)
            analysis.build()
            # analysis.analysis()
            # print('analysis.varied = {}'.format(ub.urepr(analysis.varied, nl=2)))
            ranked_stats = list(sorted(analysis.statistics, key=lambda x: x['anova_rank_p']))
            param_name_to_stats = {s['param_name']: s for s in ranked_stats}
            ranked_params = ub.oset(param_name_to_stats.keys())
        else:
            ...

        ranked_params = ['bas_poly_eval.params.bas_pxl.package_fpath']

        from kwcoco.metrics.drawing import concice_si_display
        for rank, param_name in enumerate(ub.ProgIter(ranked_params, desc='plot param for ' + vantage['name'], verbose=3)):

            param_dpath = (param_group_dpath / param_name).ensuredir()

            stats = param_name_to_stats[param_name]
            stats['moments']
            anova_rank_p = stats['anova_rank_p']
            param_name = stats['param_name']

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
            ax = sns.scatterplot(data=macro_table, x=x, y=y, hue=param_name, legend=True, **snskw)
            ax.set_title(f'BAS Results (n={len(macro_table)})\n'
                         f'Macro Analysis over {ub.urepr(rois, sv=1, nl=0)}\n'
                         f'Effect of {param_name}: anova_rank_p={concice_si_display(anova_rank_p)}')
            if 'is_star' in macro_table:
                scatterplot_highlight(data=macro_table, x=x, y=y, highlight='is_star', ax=ax, size=300)
            ax.set_xscale(xscale)
            ax.set_yscale(yscale)
            modifier.relabel(ax, ticks=False)
            vantage_fpath = vantage_dpath / f'{fname_prefix}_PLT01_scatter_legend.png'
            param_fpath = param_dpath / f'{param_metric2_prefix}_PLT01_scatter_legend.png'
            finalize_figure.finalize(fig, vantage_fpath)
            ub.symlink(real_path=vantage_fpath, link_path=param_fpath, overwrite=True)

            # Scatter legend  (doesnt care about the vantage)
            param_fpath = param_dpath / f'{param_prefix}_PLT03_scatter_onlylegend.png'
            vantage_fpath = vantage_dpath / f'{fname_prefix}_PLT03_scatter_onlylegend.png'
            if not param_fpath.exists():
                legend_ax = util_kwplot.extract_legend(ax)
                freq_mapper_scatter.relabel(legend_ax, ticks=False)
                finalize_figure.finalize(legend_ax.figure, param_fpath)
            ub.symlink(real_path=param_fpath, link_path=vantage_fpath, overwrite=True)

            ax.get_legend().remove()
            vantage_fpath = vantage_dpath / f'{fname_prefix}_PLT02_scatter_nolegend.png'
            param_fpath = param_dpath / f'{param_metric2_prefix}_PLT03_scatter_onlylegend.png'
            finalize_figure.finalize(fig, vantage_fpath)
            ub.symlink(real_path=vantage_fpath, link_path=param_fpath, overwrite=True)

            # BOX
            vantage_fpath = vantage_dpath / f'{fname_prefix}_PLT04_box.png'
            param_fpath = param_dpath / f'{param_metric_prefix}_PLT04_box.png'
            print(f'param_fpath={param_fpath}')
            if not param_fpath.exists():
                fig = kwplot.figure(fnum=5, doclf=True)
                ax = sns.boxplot(data=macro_table, x=param_name, y=y, **snskw)
                freq_mapper_box.relabel_xticks(ax)
                ax.set_title(f'BAS Results (n={len(macro_table)})\n'
                             f'Macro Analysis over {ub.urepr(rois, sv=1, nl=0)}\n'
                             f'Effect of {param_name}: anova_rank_p={concice_si_display(anova_rank_p)}')
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

    agg_dpath = ub.Path(config['root_dpath']) / 'aggregate'

    # TODO: save this for custom analysis, let automatic choose
    # for generality
    # macro_groups = [
    #     {'KR_R001', 'KR_R002'},
    #     {'KR_R001', 'KR_R002', 'US_R007'},
    #     {'KR_R001', 'KR_R002', 'BR_R002', 'AE_R001'},
    #     {'KR_R001', 'KR_R002', 'BR_R002', 'AE_R001', 'US_R007'},
    # ]
    # rois = macro_groups  # NOQA
    # selector = {'BR_R002', 'KR_R001', 'KR_R002', 'AE_R001'}
    # selector = {'BR_R002', 'KR_R001', 'KR_R002'}
    macro_groups = None
    selector = None

    agg0 = eval_type_to_aggregator.get('bas_poly_eval', None)
    if agg0 is not None:
        ...

        subagg2 = generic_analysis(agg0, macro_groups, selector)

        to_visualize_fpaths = list(subagg2.results['fpaths']['fpath'])
        agg_group_dpath = agg_dpath / ('bas_poly_agg_' + timestamp)
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


def foldin_resolved_info(agg):
    """
    Uses the params after they have been resolved in the results files.  This
    is still pretty dirty and needs to be refined to fit into mlops in a nicer
    way.
    """
    # make these just parse nicer, but for now munge the data.
    from watch.utils.util_pandas import DotDictDataFrame
    from watch.utils.util_pandas import pandas_add_prefix
    param_types = DotDictDataFrame(agg.results['param_types'])

    # param_types['pxl.meta']
    fit_params = util_pandas.pandas_shorten_columns(param_types['fit'])
    resources = util_pandas.pandas_shorten_columns(param_types['pxl.resource'])
    properties = util_pandas.pandas_shorten_columns(param_types['pxl.properties'])
    meta = util_pandas.pandas_shorten_columns(param_types['pxl.meta'])
    pred_params = param_types['pxl']
    pred_params = pred_params.drop([
        c for c in pred_params.columns
        if '.meta' in c or '.resource' in c or '.properties' in c],
        axis=1)
    pred_params = util_pandas.pandas_shorten_columns(pred_params)

    node_name = agg.type

    if True:
        # HACK: This doesn't work generally, we need to be more careful
        # about how we load resolved configurations.
        if 'bas' in agg.type:
            disk_resolved_params = pandas_add_prefix(pred_params, node_name + '.params.bas_pxl.')
        else:
            disk_resolved_params = pandas_add_prefix(pred_params, node_name + '.params.sc_pxl.')

    else:
        disk_resolved_params = pandas_add_prefix(pred_params, node_name + '.params.')

    properties = pandas_add_prefix(properties, node_name + '.properties.')
    fit_params = pandas_add_prefix(fit_params, node_name + '.fit.')
    resources = pandas_add_prefix(resources, node_name + '.resources.')
    meta = pandas_add_prefix(meta, node_name + '.meta.')

    resolved_info = {
        'resources': resources,
        'fit_params': fit_params,
        'properties': properties,
        'meta': meta,
    }

    if True:
        is_specified = agg.results['specified_params']
        effective_params = agg.effective_params
        d1 = effective_params
        d2 = disk_resolved_params
        # Handle autos and things by placing in defaults
        c1 = d1.columns
        c2 = d2.columns
        always_defaulted_cols = c2.difference(c1)
        common_cols = c1.intersection(c2)

        path_colnames = agg.model_cols + agg.test_dset_cols
        for colname in path_colnames:
            try:
                colvals = d2[colname]
                condensed, mapper = util_pandas.pandas_condense_paths(colvals)
                d2[colname] = condensed
            except Exception as ex:
                print(f'warning: ex={ex}')
                ...

        resolved_is_specified = is_specified.copy()
        resolved_is_specified.loc[:, always_defaulted_cols] = 0
        resolved_params = d1.copy()
        resolved_params[always_defaulted_cols] = disk_resolved_params[always_defaulted_cols]

        a = d1[common_cols]
        b = d2[common_cols]
        has_diff = ~util_pandas.pandas_nan_eq(a, b)

        for colname, colflags in has_diff.T.iterrows():
            resolved_params.loc[colflags, colname] = d2.loc[colflags, common_cols]

        # Fix issues with chipdims
        resolved_params = resolved_params.applymap(lambda x: str(x) if isinstance(x, list) else x)

        resolved_info['resolved_params'] = resolved_params
    return resolved_info


def make_summary_analysis(agg1, config, dpath=None):
    if dpath is None:
        agg_dpath = ub.Path(config['root_dpath'] / 'aggregate')
    else:
        agg_dpath = dpath
    agg_group_dpath = agg_dpath / ('agg_summary_params2_v3')
    agg_group_dpath = agg_group_dpath.ensuredir()

    # Given these set of A/B values, visualize each region
    for region_id, group in agg1.index.groupby('region_id'):
        group_agg = agg1.filterto(index=group.index)
        for id, row in group_agg.index.iterrows():
            eval_fpath = group_agg.fpaths[id]
            param_hashid = row['param_hashid']
            region_id = row['region_id']
            dname = f'{region_id}_{param_hashid}'
            link_dpath = agg_group_dpath / dname
            real_dpath = eval_fpath.parent
            ub.symlink(real_path=real_dpath, link_path=link_dpath)
            import kwimage
            from kwcoco.metrics.drawing import concice_si_display
            region_viz_fpaths = list((eval_fpath.parent / 'region_viz_overall').glob('*_detailed.png'))
            assert len(region_viz_fpaths) == 1
            region_viz_fpath = region_viz_fpaths[0]
            viz_img = kwimage.imread(region_viz_fpath)
            scores_of_interest = util_pandas.pandas_shorten_columns(agg1.metrics).loc[id, ['bas_tp', 'bas_fp', 'bas_fn', 'bas_f1']]
            scores_of_interest = ub.udict(scores_of_interest.to_dict())
            text = ub.urepr(scores_of_interest.map_values(concice_si_display), nobr=1, si=1, compact=1)
            new_img = kwimage.draw_header_text(viz_img, param_hashid + '\n' + text)
            kwimage.imwrite(agg_group_dpath / f'summary_{region_id}_{param_hashid}.jpg', new_img)

    for region_id, group in list(agg1.index.groupby('region_id')):
        group_agg = agg1.filterto(index=group.index)
        for id, row in list(group_agg.index.iterrows()):
            param_hashid = row['param_hashid']
            region_id = row['region_id']
            eval_fpath = group_agg.fpaths[id]
            confusion_fpaths = list((eval_fpath.parent / 'bas_summary_viz').glob('confusion_*.jpg'))
            if len(confusion_fpaths) == 0:
                from watch.mlops import confusion_visualization
                confusion_visualization.bas_poly_eval_confusion_analysis(eval_fpath)
            confusion_fpaths = list((eval_fpath.parent / 'bas_summary_viz').glob('confusion_*.jpg'))
            assert len(confusion_fpaths) == 1
            confusion_fpath = confusion_fpaths[0]
            im = kwimage.imread(confusion_fpath)
            scores_of_interest = util_pandas.pandas_shorten_columns(agg1.metrics).loc[id, ['bas_tp', 'bas_fp', 'bas_fn', 'bas_f1']]
            scores_of_interest = ub.udict(scores_of_interest.to_dict())
            text = ub.urepr(scores_of_interest.map_values(concice_si_display), nobr=1, si=1, compact=1)
            model_name = group_agg.effective_params[group_agg.model_cols[0]].loc[id]
            im = kwimage.draw_header_text(im, param_hashid + ' - ' + model_name + '\n' + text)
            kwimage.imwrite(agg_group_dpath / f'confusion_{region_id}_{param_hashid}.jpg', im)


def fix_duplicate_param_hashids(agg0):
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

    print('chosen_macro_rois = {}'.format(ub.repr2(chosen_macro_rois, nl=1)))
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


def build_aggregators(eval_type_to_results, agg_dpath):
    eval_type_to_aggregator = {}
    for key, results in eval_type_to_results.items():
        agg = Aggregator(results, type=key, agg_dpath=agg_dpath)
        # FIXME: if there are no results, don't try to build?
        agg.build()
        eval_type_to_aggregator[key] = agg
        # TODO : nicer replacement of long paths for params
        # metrics['sc_poly_eval.metrics.sc_macro_f1']
    return eval_type_to_aggregator


class Aggregator(ub.NiceRepr):
    """
    Stores multiple data frames that separate metrics, parameters, and other
    information using consistent pandas indexing. Can be filtered to a
    comparable subsets of choice. Can also handle building macro averaged
    results over different "regions" with the same parameters.

    Set config based on your problem
    """
    def __init__(agg, results, type=None, agg_dpath=None,
                 primary_metric_cols='auto',
                 display_metric_cols='auto'):
        agg.agg_dpath = agg_dpath
        agg.results = results
        agg.type = type
        agg.metrics = results['metrics']
        agg.requested_params = results['requested_params']
        agg.resolved_params = results['resolved_params']
        agg.specified_params = results['specified_params']
        agg.params = agg.requested_params
        agg.index = results['index']
        agg.fpaths = results['fpath']

        agg.config = {
            'display_metric_cols': display_metric_cols,
            'primary_metric_cols': primary_metric_cols,
        }

    def __export(agg):
        ...
        ub.udict(agg.results).map_values(type)
        for col in agg.table.columns:
            vs = agg.table[col]
            if list in set(vs.apply(type)):
                print(col)
        fpath = 'bas_results_2023-01.csv.zip'
        agg.table.to_csv(fpath, index_label=False)

    def __nice__(self):
        return f'{self.type}, n={len(self)}'

    def __len__(self):
        return len(self.index)

    def view_directory(agg):
        import xdev
        # TODO: make agg.dpath
        xdev.view_directory(agg.dpath)

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

    def filterto(agg, models=None, param_hashids=None, index=None):
        import numpy as np
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

        if isinstance(final_flags, int):
            new_agg = agg
        else:
            new_agg = agg.compress(final_flags)
        return new_agg

    def compress(agg, flags):
        import pandas as pd
        new_results = {}
        for key, val in agg.results.items():
            if isinstance(val, list):
                new_results[key] = list(ub.compress(val, flags))
            elif isinstance(val, pd.DataFrame):
                new_results[key] = val[flags].copy()
            else:
                new_results[key] = val
        new_agg = Aggregator(new_results, agg.type, **agg.config)
        new_agg.build()
        return new_agg

    def build(agg):
        agg.__dict__.update(**agg.config)

        _primary_metrics_suffixes, _display_metrics_suffixes = SMART_HELPER._default_metrics(agg)

        if agg.primary_metric_cols == 'auto':
            agg.primary_metric_cols = util_pandas.pandas_suffix_columns(  # fixme sorting
                agg.metrics, _primary_metrics_suffixes)

        if agg.display_metric_cols == 'auto':
            agg.display_metric_cols = util_pandas.pandas_suffix_columns(  # fixme sorting
                agg.metrics, _display_metrics_suffixes)

        _model_suffixes = ['package_fpath']
        _testdset_suffixes = ['test_dataset']

        agg.model_cols = util_pandas.pandas_suffix_columns(
            agg.requested_params, _model_suffixes)
        agg.test_dset_cols = util_pandas.pandas_suffix_columns(
            agg.requested_params, _testdset_suffixes)

        # util_pandas.pandas_suffix_columns(agg.resolved_params, _testdset_suffixes)

        effective_params, mappings, hashid_to_params = agg.build_effective_params()
        # agg.results['effective_params'] = effective_params
        agg.hashid_to_params = ub.udict(hashid_to_params)
        agg.mappings = mappings
        agg.effective_params = effective_params

        # agg.resolved_info = foldin_resolved_info(agg)
        # agg.fit_params = agg.resolved_info['fit_params']
        # agg.resolved_params = agg.resolved_info['resolved_params']
        # agg.effective_table = pd.concat([agg.metrics, agg.index, agg.effective_params], axis=1)

        agg.macro_key_to_regions = {}
        agg.region_to_tables = {}
        for region_id, idx_group in agg.index.groupby('region_id'):
            agg.region_to_tables[region_id] = {
                k: v.loc[idx_group.index] for k, v in agg.results.items()
            }
        agg.macro_compatible = agg.find_macro_comparable()

    def table(agg):
        table = pd.concat(list(agg.results.values()), axis=1, verify_integrity=True)
        return table

    def macro_analysis(agg):
        from watch.utils import result_analysis

        macro_keys = list(agg.macro_key_to_regions.keys())
        if len(macro_keys) == 0:
            raise Exception('Build a macro result first')

        regions_of_interest = agg.macro_key_to_regions[agg.primary_macro_region]
        tables = agg.region_to_tables[agg.primary_macro_region]
        resolved_params = tables['resolved_params']
        metrics = tables['metrics']
        index = tables['index']
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
        from watch.utils import result_analysis
        metrics_of_interest = agg.primary_metric_cols
        analysis = result_analysis.ResultAnalysis(
            agg.results, metrics=metrics_of_interest)
        analysis.results
        analysis.analysis()

    def report_best(agg, top_k=3, shorten=True):
        """
        Report the top k pointwise results for each region / macro-region.

        Note:
            Results are chosen per-region independently. To get comparable
            results for a specific set of parameters, filter to them and then
            report the top results for that filtering.

        Args:
            k (int): number of top results for each region

        Returns:
            Tuple[T1, T2]:
                region_id_to_summary (T1=Dict[str, DataFrame]):
                    mapping from region_id to top k results
                top_param_lut (T2=Dict[str, DataFrame]):
                    mapping from param hash to invocation details
        """
        import rich
        region_id_to_summary = {}
        big_param_lut = {}
        region_id_to_ntotal = {}

        for region_id, group in agg.region_to_tables.items():
            metric_group = group['metrics']
            metric_group = metric_group.sort_values(agg.primary_metric_cols)

            top_idxs = util_pandas.pandas_argmaxima(metric_group, agg.primary_metric_cols, k=top_k)

            final_display_cols = list(ub.oset(agg.primary_metric_cols + agg.display_metric_cols))
            top_metrics = metric_group.loc[top_idxs][final_display_cols]
            # top_metrics = top_metrics[agg.primary_metric_cols + agg.display_metric_cols]
            top_indexes = group['index'].loc[top_idxs]
            # top_params = group['effective_params'].loc[top_idxs].drop(agg.test_dset_cols, axis=1)
            param_lut = agg.hashid_to_params.subdict(top_indexes['param_hashid'])
            big_param_lut.update(param_lut)
            summary_table = pd.concat([top_indexes, top_metrics], axis=1)
            if shorten:
                summary_table = util_pandas.pandas_shorten_columns(summary_table)
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

        rich.print('Parameter LUT: {}'.format(ub.urepr(top_param_lut, nl=2)))

        # Check for a common special case that we can make more concise output for
        only_one_top_item = all(len(t) == 1 for t in region_id_to_summary.values())
        only_one_source_item = all(n == 1 for n in region_id_to_ntotal.values())

        if only_one_source_item and only_one_top_item:
            justone = pd.concat(list(region_id_to_summary.values()), axis=0)
            submacro = ub.udict(agg.macro_key_to_regions) & justone['region_id'].values
            if submacro:
                print('Macro Regions LUT: ' +  ub.urepr(submacro, nl=1))
            rich.print(justone)
        elif only_one_top_item:
            justone = pd.concat(list(region_id_to_summary.values()), axis=0)
            # submacro = ub.udict(agg.macro_key_to_regions) & justone['region_id'].values
            # if submacro:
            #     print('Macro Regions LUT: ' +  ub.urepr(submacro, nl=1))
            rich.print(justone)
            rich.print('agg.macro_key_to_regions = {}'.format(ub.repr2(agg.macro_key_to_regions, nl=1)))
        else:
            for region_id, summary_table in region_id_to_summary.items():
                ntotal = region_id_to_ntotal[region_id]
                if region_id in agg.macro_key_to_regions:
                    macro_regions = agg.macro_key_to_regions[region_id]
                    rich.print(f'Top {len(summary_table)} / {ntotal} for {region_id} = {macro_regions}')
                else:
                    rich.print(f'Top {len(summary_table)} / {ntotal} for {region_id}')
                rich.print(summary_table.iloc[::-1].to_string())

        return region_id_to_summary, top_param_lut

    def build_effective_params(agg):
        """
        Consolodate / cleanup / expand information
        """
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

        _specified = util_pandas.DotDictDataFrame(agg.results['specified_params'])
        _specified_params = _specified.subframe('specified')
        is_param_included = _specified_params > 0

        # For each unique set of effective parameters compute a hashid
        param_cols = ub.oset(effective_params.columns).difference(agg.test_dset_cols)
        param_cols = list(param_cols - {'region_id', 'node'})
        hashids_v1 = pd.Series([None] * len(agg.index), index=agg.index.index)
        # hashids_v0 = pd.Series([None] * len(agg.index), index=agg.index.index)
        hashid_to_params = {}
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
        # agg.index.loc[hashids_v0.index, 'param_hashid_v0'] = hashids_v0

        return effective_params, mappings, hashid_to_params

    def find_macro_comparable(agg):
        """
        Search for groups that have the same parameters over multiple regions.
        """
        table = pd.concat([agg.index, agg.metrics, agg.resolved_params], axis=1)

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
        comparable_groups = []
        macro_compatible = agg.macro_compatible
        for key in macro_compatible.keys():
            avail = (key & regions_of_interest)
            if avail == regions_of_interest:
                groups = macro_compatible[key]
                for group in groups:
                    flags = kwarray.isect_flags(group['region_id'], avail)
                    # if len(group[flags]) > 3:
                    #     raise Exception
                    comparable_groups.append(group[flags])
        return comparable_groups

    def _coerce_rois(agg, rois=None):
        if rois is None:
            rois = 'max'
        if isinstance(rois, str):
            if rois == 'max':
                regions_of_interest = ub.argmax(agg.macro_compatible, key=len)
        else:
            regions_of_interest = rois
        return regions_of_interest

    def build_macro_tables(agg, rois=None):
        """
        Builds one or more macro tables
        """
        if isinstance(rois, list) and len(rois) and ub.iterable(rois[0]):
            # Asked for multiple groups of ROIS.
            for single_rois in rois:
                agg.build_single_macro_table(single_rois)
        else:
            agg.build_single_macro_table(rois)

    def build_single_macro_table(agg, rois):
        """
        Builds a single macro table for a choice of regions.
        """
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

        # Gather groups that can be aggregated
        comparable_groups = agg.gather_macro_compatable_groups(regions_of_interest)
        if len(comparable_groups) == 0:
            print(ub.paragraph(
                f'''
                WARNING: Failed to build macro results. No comparable groups
                for rois={rois}
                '''))
        else:
            if 0:
                # gather debug info about all of the comparable groups and check
                # basic assumptions
                stat_accum = {
                    'size': []
                }
                seen_indexes = set()
                for group in comparable_groups:
                    assert len(group.param_hashid.unique()) == 1
                    assert len(group.param_hashid) >= 1
                    assert len(group.index.unique()) == len(group)
                    stat_accum['size'].append(len(group))
                    assert not (set(group.index) & seen_indexes)
                    seen_indexes.update(group.index)
                    if len(group) > 2:
                        break

                print(pd.DataFrame(stat_accum).describe().T)

            # Macro aggregaet comparable groups
            macro_parts = ub.ddict(list)
            for group in comparable_groups:
                agg_parts = macro_aggregate(agg, group, aggregator)
                for k, v in agg_parts.items():
                    macro_parts[k].append(v)

            # main_metric = 'bas_poly_eval.metrics.bas_faa_f1'
            # main_metric = 'bas_poly_eval.metrics.bas_tp'
            # macro_df = macro_df.sort_values(main_metric, ascending=False)
            macro_results = {
                k: pd.DataFrame(vs).reset_index(drop=True)
                for k, vs in macro_parts.items()
            }
            agg.region_to_tables.pop(macro_key, None)
            agg.macro_key_to_regions.pop(macro_key, None)
            agg.macro_key_to_regions[macro_key] = regions_of_interest
            agg.region_to_tables[macro_key] = macro_results
            return macro_results


def aggregate_param_cols(df, hash_cols=None, allow_nonuniform=False):
    """
    Aggregates parameter columns. Specified hash_cols should be
    dataset-specific columns to be hashed. All other columns should
    be effectively the same, otherwise we will warn.
    """
    agg_row = df.iloc[0]
    if len(df) == 1:
        return agg_row
    else:
        if hash_cols:
            df_comparable = df.drop(hash_cols, axis=1)
            df_hashable = df[hash_cols]
            hashed = {c: hash_regions(v) for c, v in df_hashable.T.iterrows()}
        else:
            df_comparable = df
            hashed = {}
        is_safe_cols = {
            k: ub.allsame(vs, eq=nan_eq)
            for k, vs in df_comparable.T.iterrows()}
        nonuniform_cols = {k: v for k, v in is_safe_cols.items() if not v}
        if allow_nonuniform:
            df = df.drop(nonuniform_cols, axis=1)
        else:
            if nonuniform_cols:
                raise AssertionError(f'Values not identical: {nonuniform_cols}')
        if hashed:
            agg_row = agg_row.copy()
            for c, v in hashed.items():
                agg_row[c] = v
    return agg_row


def macro_aggregate(agg, group, aggregator):
    """
    Helper function
    """
    blocklist = {'fpath'}
    group_index = agg.index.loc[group.index]
    if (group_index.value_counts('region_id') > 1).any():
        # Check if there is more than one run per-region per-param and
        # average them to keep the stats balanced.
        subgroups = group_index.groupby('region_id')
        subagg_parts = ub.ddict(list)
        for _, subgroup in subgroups:
            subgroup_parts = {}
            for k, v in agg.results.items():
                if k not in blocklist:
                    subgroup_parts[k] = v.loc[subgroup.index]

            _group_parts = subgroup_parts.copy()
            subagg_part = {}
            subagg_part['index']  = aggregate_param_cols(_group_parts.pop('index'), ['region_id'])
            subagg_part['metrics'] = _group_parts.pop('metrics').aggregate('mean')
            subagg_part['other'] = _group_parts.pop('other').mean(numeric_only=True)
            subagg_part['resolved_params']  = aggregate_param_cols(_group_parts.pop('resolved_params'), agg.test_dset_cols)
            subagg_part['requested_params']  = aggregate_param_cols(_group_parts.pop('requested_params'), agg.test_dset_cols, allow_nonuniform=True)
            subagg_part['specified_params']  = aggregate_param_cols(_group_parts.pop('specified_params'), allow_nonuniform=True)
            assert len(_group_parts) == 0

            for k, v in subagg_part.items():
                subagg_parts[k].append(v)

        group_parts = {}
        for k, v in subagg_parts.items():
            group_parts[k] = pd.DataFrame(v).reset_index(drop=True)

    else:
        # Each region has only one result, can use it as is.
        group_parts = {}
        for k, v in agg.results.items():
            if k not in blocklist:
                group_parts[k] = v.loc[group.index]

    _group_parts = group_parts.copy()
    agg_parts = {}
    agg_parts['index']  = aggregate_param_cols(_group_parts.pop('index'), ['region_id'])
    agg_parts['metrics'] = _group_parts.pop('metrics').aggregate(aggregator)
    agg_parts['other'] = _group_parts.pop('other').sum(numeric_only=True)
    agg_parts['resolved_params']  = aggregate_param_cols(_group_parts.pop('resolved_params'), agg.test_dset_cols)
    agg_parts['requested_params']  = aggregate_param_cols(_group_parts.pop('requested_params'), agg.test_dset_cols, allow_nonuniform=True)
    agg_parts['specified_params']  = aggregate_param_cols(_group_parts.pop('specified_params'), allow_nonuniform=True)
    assert len(_group_parts) == 0
    return agg_parts


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
    suffix = ub.hash_data(sorted(rois), base=16)[0:6]
    macro_key = f'macro_{len(rois):02d}_{suffix}'
    return macro_key


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
