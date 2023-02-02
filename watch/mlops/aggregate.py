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

"""
import kwarray
import math
import parse
import ubelt as ub
from watch.mlops import smart_pipeline
from watch.utils import util_pattern
from watch.mlops import smart_result_parser
from watch.utils import util_pandas
from watch.utils import util_parallel
from typing import Dict, Any
import pandas as pd
import json
from scriptconfig import DataConfig, Value as _V


class AggregateEvluationConfig(DataConfig):
    """
    Aggregates results from multiple DAG evaluations.
    """
    root_dpath   = _V('auto', help='Where do dump all results. If "auto", uses <expt_dvc_dpath>/dag_runs')
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
        eval_type_to_results = build_tables(config)
        eval_type_to_aggregator = build_aggregators(eval_type_to_results)
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

    eval_type_to_aggregator = build_aggregators(eval_type_to_results)

    # automated_analysis(eval_type_to_aggregator, config)
    agg = eval_type_to_aggregator.get('bas_poly_eval', None)
    # for agg in eval_type_to_aggregator.values():
    if agg is not None:
        # rois = {'KR_R001', 'KR_R002', 'BR_R002'}
        rois = {'KR_R001', 'KR_R002'}
        build_all_param_plots(agg, rois, config)
        ...


def build_all_param_plots(agg, rois, config):
    from watch.utils import util_kwplot
    import numpy as np
    import kwplot
    sns = kwplot.autosns()
    plt = kwplot.autoplt()  # NOQA
    # metric_cols = [c for c in df.columns if 'metrics.' in c]
    kwplot.close_figures()

    agg_dpath = ub.Path(config['root_dpath'] / 'aggregate')
    agg_group_dpath = (agg_dpath / ('all_params' + ub.timestamp())).ensuredir()

    # Hack in fit params
    if 1:
        resolved_params = pd.concat([agg.resolved_info['resolved_params'], agg.resolved_info['fit_params']], axis=1)
        agg.resolved_info['resolved_params'] = resolved_params
        agg.resolved_params = resolved_params

    agg.build_macro_tables(rois)

    macro_results = agg.region_to_tables[agg.primary_macro_region].copy()
    single_results = {
        'index': agg.index,
        'metrics': agg.metrics,
        'resolved_params': agg.resolved_params,
        'resources': agg.resolved_info['resources'],
    }

    macro_results['resolved_params']['bas_poly_eval.fit.effective_batch_size'] = (
        macro_results['resolved_params']['bas_poly_eval.fit.accumulate_grad_batches'] *
        macro_results['resolved_params']['bas_poly_eval.fit.batch_size']
    )

    _parts = list((ub.udict(macro_results) & {
        'index', 'metrics', 'resolved_params', 'resources'}).values())
    macro_table = pd.concat(_parts, axis=1)
    single_table = pd.concat(list(single_results.values()), axis=1)
    single_table = single_table.fillna('None')
    macro_table = macro_table.fillna('None')
    macro_table = macro_table.applymap(lambda x: str(x) if isinstance(x, list) else x)
    single_table = single_table.applymap(lambda x: str(x) if isinstance(x, list) else x)

    if 0:
        agg.model_cols
        from watch.utils.util_param_grid import DotDictDataFrame
        fit_params = DotDictDataFrame(macro_table)['fit']
        unique_packages = macro_table['bas_pxl.package_fpath'].drop_duplicates()
        # unique_fit_params = fit_params.loc[unique_packages.index]
        pkgmap = {}
        pkgver = {}
        for id, pkg in unique_packages.items():
            pkgver[pkg] = 'M{:02d}'.format(len(pkgver))
            pid = pkgver[pkg]
            out_gsd = fit_params.loc[id, 'fit.output_space_scale']
            in_gsd = fit_params.loc[id, 'fit.input_space_scale']
            assert in_gsd == out_gsd
            new_name = f'{pid}'
            if pkg == 'package_epoch0_step41':
                new_name = f'{pid}_NOV'
            pkgmap[pkg] = new_name
        macro_table['bas_pxl.package_fpath'] = macro_table['bas_pxl.package_fpath'].apply(lambda x: pkgmap.get(x, x))

    modifier = util_kwplot.LabelModifier()

    modifier.add_mapping({
        'blue|green|red|nir': 'BGRN',
        'blue|green|red|nir,invariants.0:17': 'invar',
        'blue|green|red|nir|swir16|swir22': 'BGNRSH'
    })

    @modifier.add_mapping
    def humanize_label(text):
        text = text.replace('package_epoch0_step41', 'EVAL7')
        text = text.replace('bas_poly_eval.params.', '')
        text = text.replace('bas_poly_eval.metrics.', '')
        text = text.replace('bas_poly_eval.fit.', 'fit.')
        return text

    # Pre determine some palettes
    shared_palette_groups = [
        ['bas_poly_eval.params.bas_poly.thresh'],
        ['bas_poly_eval.fit.learning_rate'],
        ['bas_poly_eval.fit.learning_rate'],
        ['bas_poly_eval.params.bas_pxl.chip_dims', 'bas_poly_eval.fit.chip_dims'],
        ['bas_poly_eval.params.bas_pxl.output_space_scale', 'bas_poly_eval.fit.output_space_scale', 'bas_poly_eval.params.bas_poly.resolution'],
    ]
    param_to_palette = {}
    for group_params in shared_palette_groups:
        unique_vals = np.unique(macro_table[group_params].values)
        # 'Spectral'
        if len(unique_vals) > 5:
            unique_colors = sns.color_palette('Spectral', n_colors=len(unique_vals))
            # kwplot.imshow(_draw_color_swatch(unique_colors), fnum=32)
        else:
            unique_colors = sns.color_palette(n_colors=len(unique_vals))
        palette = ub.dzip(unique_vals, unique_colors)
        param_to_palette.update({p: palette for p in group_params})

    if 1:
        #### Hack for models of interest.
        star_params = []
        p1 = macro_table[(
            # (macro_table['bas_poly.moving_window_size'] == 200) &
            (macro_table['bas_poly_eval.params.bas_pxl.package_fpath'] == 'package_epoch0_step41') &
            (macro_table['bas_poly_eval.params.bas_pxl.chip_dims'] == '[128, 128]') &
            (macro_table['bas_poly_eval.params.bas_poly.thresh'] == 0.12)  &
            (macro_table['bas_poly_eval.params.bas_poly.max_area_sqkm'] == 'None') &
            (macro_table['bas_poly_eval.params.bas_poly.moving_window_size'] == 'None')
        )]['param_hashid'].iloc[0]
        star_params = [p1]
        p2 = macro_table[(
            # (macro_table['bas_poly.moving_window_size'] == 200) &
            (macro_table['bas_poly_eval.params.bas_pxl.package_fpath'] == 'Drop4_BAS_2022_12_15GSD_BGRN_V10_epoch=0-step=4305') &
            (macro_table['bas_poly_eval.params.bas_pxl.chip_dims'] == '[224, 224]') &
            (macro_table['bas_poly_eval.params.bas_poly.thresh'] == 0.13)  &
            (macro_table['bas_poly_eval.params.bas_poly.max_area_sqkm'] == 'None') &
            (macro_table['bas_poly_eval.params.bas_poly.moving_window_size'] == 200)
        )]['param_hashid'].iloc[0]
        star_params += [p2]
        p3 = macro_table[(
            # (macro_table['bas_poly.moving_window_size'] == 200) &
            (macro_table['bas_poly_eval.params.bas_pxl.package_fpath'] == 'Drop4_BAS_15GSD_BGRNSH_invar_V8_epoch=16-step=8704') &
            (macro_table['bas_poly_eval.params.bas_pxl.chip_dims'] == '[256, 256]') &
            (macro_table['bas_poly_eval.params.bas_poly.thresh'] == 0.17)  &
            (macro_table['bas_poly_eval.params.bas_poly.max_area_sqkm'] == 'None') &
            (macro_table['bas_poly_eval.params.bas_poly.moving_window_size'] == 'None')
        )]['param_hashid'].iloc[0]
        star_params += [p3]
        macro_table['is_star'] = kwarray.isect_flags(macro_table['param_hashid'], star_params)

    # x = 'bas_poly_eval.metrics.bas_tpr'
    # y = 'bas_poly_eval.metrics.bas_ppv'
    # x = 'bas_poly_eval.metrics.bas_space_FAR'
    # y = 'bas_poly_eval.metrics.bas_tpr'
    # x = 'bas_poly_eval.metrics.bas_ffpa'
    # xscale = 'log'

    x = 'bas_poly_eval.metrics.bas_tpr'
    xscale = 'linear'

    y = 'bas_poly_eval.metrics.bas_f1'
    y = 'bas_poly_eval.metrics.bas_faa_f1'

    # main_metric = 'bas_poly_eval.metrics.bas_f1'
    main_metric = 'bas_poly_eval.metrics.bas_faa_f1'
    metric_objectives = {main_metric: 'maximize'}

    def finalize_figure(fig, fpath):
        fig.set_size_inches(np.array([6.4, 4.8]) * 1.0)
        fig.tight_layout()
        fig.savefig(fpath)
        util_kwplot.cropwhite_ondisk(fpath)

    fig = kwplot.figure(fnum=2, doclf=True)
    ax = sns.scatterplot(data=single_table, x=x, y=y, hue='region_id')
    ax.set_title(f'BAS Per-Region Results (n={len(agg)})')
    ax.set_xscale('log')
    fpath = agg_group_dpath / 'single_results.png'
    finalize_figure(fig, fpath)
    # ax.set_xlim(0, np.quantile(agg.metrics[x], 0.99))
    # ax.set_xlim(1e-2, np.quantile(agg.metrics[x], 0.99))

    fig = kwplot.figure(fnum=90, doclf=True)
    ax = sns.boxplot(data=single_table, x='region_id', y=main_metric)
    ax.set_title(f'BAS Per-Region Results (n={len(agg)})')
    util_kwplot.LabelModifier({
        param_value: f'{param_value}\n(n={num})'
        for param_value, num in single_table.groupby('region_id').size().to_dict().items()
    }).relabel_xticks(ax)
    modifier.relabel(ax)
    fpath = agg_group_dpath / 'single_results_boxplot.png'
    finalize_figure(fig, fpath)

    from watch.utils.util_kwplot import scatterplot_highlight
    fig = kwplot.figure(fnum=3, doclf=True)
    ax = fig.gca()
    ax = sns.scatterplot(data=macro_table, x=x, y=y, hue='region_id', ax=ax)
    if 'is_star' in macro_table:
        scatterplot_highlight(data=macro_table, x=x, y=y, highlight='is_star', ax=ax, size=300)
    ax.set_title(f'BAS Results (n={len(macro_table)})\n'
                 f'Macro Analysis over {ub.urepr(rois, sv=1, nl=0)}')
    ax.set_xscale(xscale)
    fpath = agg_group_dpath / 'macro_results.png'
    finalize_figure(fig, fpath)
    # ax.set_xlim(1e-2, npe.quantile(agg.metrics[x], 0.99))
    # ax.set_xlim(1e-2, 0.7)

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
        analysis.analysis()
        print('analysis.varied = {}'.format(ub.urepr(analysis.varied, nl=2)))
        ranked_stats = list(sorted(analysis.statistics, key=lambda x: x['anova_rank_p']))
        param_name_to_stats = {s['param_name']: s for s in ranked_stats}
        ranked_params = ub.oset(param_name_to_stats.keys())
    else:
        ...

    if 0:
        import xdev
        xdev.view_directory(agg_group_dpath)

    if 1:
        ranked_params = [p for p in ranked_params if 'resolution' in p]

    from kwcoco.metrics.drawing import concice_si_display
    for rank, param_name in ub.ProgIter(enumerate(ranked_params)):
        stats = param_name_to_stats[param_name]
        stats['moments']
        anova_rank_p = stats['anova_rank_p']
        param_name = stats['param_name']

        snskw = {}
        if param_name in param_to_palette:
            snskw['palette'] = param_to_palette[param_name]

        fig = kwplot.figure(fnum=4, doclf=True)
        # ax = sns.scatterplot(data=macro_table, x=x, y=y, hue=agg.model_cols[0])
        ax = sns.scatterplot(data=macro_table, x=x, y=y, hue=param_name, legend=False, **snskw)
        # scatterplot_highlight(data=macro_table, x=x, y=y, highlight='is_star', ax=ax, size=300)
        ax.set_title(f'BAS Results (n={len(macro_table)})\n'
                     f'Macro Analysis over {ub.urepr(rois, sv=1, nl=0)}\n'
                     f'Effect of {param_name}: anova_rank_p={concice_si_display(anova_rank_p)}')
        if 'is_star' in macro_table:
            scatterplot_highlight(data=macro_table, x=x, y=y, highlight='is_star', ax=ax, size=300)
        ax.set_xscale(xscale)
        modifier.relabel(ax)
        fpath = agg_group_dpath / f'macro_results_{rank:03d}_{param_name}.png'
        finalize_figure(fig, fpath)

        fig = kwplot.figure(fnum=5, doclf=True)
        ax = sns.boxplot(data=macro_table, x=param_name, y=y, **snskw)
        util_kwplot.LabelModifier({
            param_value: f'{param_value}\n(n={num})'
            for param_value, num in macro_table.groupby(param_name).size().to_dict().items()
        }).relabel_xticks(ax)
        ax.set_title(f'BAS Results (n={len(macro_table)})\n'
                     f'Macro Analysis over {ub.urepr(rois, sv=1, nl=0)}\n'
                     f'Effect of {param_name}: anova_rank_p={concice_si_display(anova_rank_p)}')
        modifier.relabel(ax)
        fpath = agg_group_dpath / f'macro_results_{rank:03d}_{param_name}_box.png'
        finalize_figure(fig, fpath)


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
    from watch.utils.util_param_grid import DotDictDataFrame
    from watch.utils.util_param_grid import pandas_add_prefix
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


def make_summary_analysis(agg1, config):
    agg_dpath = ub.Path(config['root_dpath'] / 'aggregate')
    agg_group_dpath = agg_dpath / ('agg_summary_params2_v2')
    agg_group_dpath = agg_group_dpath.ensuredir()

    # agg2 =

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


def build_tables(config):
    import pandas as pd
    from watch.utils import util_progress
    print('config = {}'.format(ub.repr2(dict(config), nl=1)))
    dag = smart_pipeline.make_smart_pipeline(config['pipeline'])
    dag.print_graphs()
    dag.configure(config=None, root_dpath=config['root_dpath'])

    io_workers = util_parallel.coerce_num_workers(config.io_workers)
    print(f'io_workers={io_workers}')

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
        {'name': 'sc_poly_eval', 'out_key': 'eval_fpath',
         'result_loader': smart_result_parser.load_eval_act_poly},
        {'name': 'bas_poly_eval', 'out_key': 'eval_fpath',
         'result_loader': smart_result_parser.load_eval_trk_poly},
    ]

    from concurrent.futures import as_completed
    pman = util_progress.ProgressManager(backend='rich')
    with pman:
        eval_type_to_results = {}
        eval_node_prog = pman.progiter(node_eval_infos, desc='Loading node results')

        for node_eval_info in eval_node_prog:
            node_name = node_eval_info['name']
            out_key = node_eval_info['out_key']
            # result_loader_fn = node_eval_info['result_loader']

            if node_name not in dag.nodes:
                continue

            node = dag.nodes[node_name]
            out_node = node.outputs[out_key]
            out_node_key = out_node.key

            fpaths = out_node_matching_fpaths(out_node)

            # Pattern match
            # node.template_out_paths[out_node.name]
            cols = {
                'metrics': [],
                'index': [],
                'params': [],
                'specified_params': [],
                'param_types': [],
                'fpaths': [],
                # 'json_info': [],
            }
            executor = ub.Executor(mode='process', max_workers=io_workers)
            jobs = []
            submit_prog = pman.progiter(
                fpaths, desc=f'  * submit load jobs: {node_name}',
                transient=True)
            for fpath in submit_prog:
                job = executor.submit(load_result_worker, fpath, node_name,
                                      out_node_key)
                jobs.append(job)

            num_ignored = 0
            job_iter = as_completed(jobs)
            del jobs
            collect_prog = pman.progiter(
                job_iter, total=len(fpaths),
                desc=f'  * loading node results: {node_name}')
            for job in collect_prog:
                job.result()
                fpath, index, metrics, params, param_types = job.result()
                if params:
                    cols['metrics'].append(metrics)
                    cols['params'].append(params)
                    cols['index'].append(index)
                    cols['param_types'].append(param_types)
                    # To make hashing consistent we need to know which
                    # parameters were explicitly specified
                    cols['specified_params'].append({k: 1 for k in params})
                    cols['fpaths'].append(fpath)
                    # cols['json_info'].append(result['json_info'])
                else:
                    num_ignored += 1

            params = pd.DataFrame(cols['params'])
            # trunc_params, mappings = util_pandas.pandas_truncate_items(params)
            results = {
                # 'mappings': mappings,
                'fpaths': pd.DataFrame(cols['fpaths'], columns=['fpath']),
                'index': pd.DataFrame(cols['index']),
                'metrics': pd.DataFrame(cols['metrics']),
                'params': pd.DataFrame(cols['params']),
                'specified_params': pd.DataFrame(cols['specified_params']),
                # 'trunc_params': trunc_params,
                'param_types': pd.DataFrame(cols['param_types']),
            }
            eval_type_to_results[node_name] = results

    return eval_type_to_results


def load_result_worker(fpath, node_name, out_node_key):

    if node_name == 'bas_pxl_eval':
        result_loader_fn = smart_result_parser.load_pxl_eval
    elif node_name == 'bas_poly_eval':
        result_loader_fn = smart_result_parser.load_eval_trk_poly
    elif node_name == 'sc_poly_eval':
        result_loader_fn = smart_result_parser.load_eval_act_poly
    else:
        raise KeyError(node_name)

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
        'type': out_node_key,
        'region_id': region_ids,
    }
    metrics = smart_result_parser._add_prefix(node_name + '.metrics.', result['metrics'])
    params = smart_result_parser._add_prefix(node_name + '.params.', config_)
    # These are the old way of loading params, but they are also
    # the resolved params, which we should take into account
    param_types = result['param_types']
    param_types = ub.udict.union({}, *list(param_types.values()))

    return fpath, index, metrics, params, param_types


def build_aggregators(eval_type_to_results):
    eval_type_to_aggregator = {}
    for key, results in eval_type_to_results.items():
        agg = Aggregator(results, type=key)
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
    """
    def __init__(agg, results, type=None):
        agg.results = results
        agg.type = type
        agg.metrics = results['metrics']
        agg.params = results['params']
        agg.index = results['index']
        agg.fpaths = results['fpaths']['fpath']

    def __nice__(self):
        return f'{self.type}, n={len(self)}'

    def __len__(self):
        return len(self.index)

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
        new_agg = Aggregator(new_results, agg.type)
        new_agg.build()
        return new_agg

    def build(agg):
        _display_metrics_suffixes = []
        if agg.type == 'bas_poly_eval':
            _display_metrics_suffixes = [
                'bas_poly_eval.metrics.bas_tp',
                'bas_poly_eval.metrics.bas_fp',
                'bas_poly_eval.metrics.bas_fn',
                'bas_poly_eval.metrics.bas_f1',
                'bas_poly_eval.metrics.bas_ffpa',
            ]
            _primary_metrics_suffixes = [
                # 'bas_faa_f1'
                'bas_poly_eval.metrics.bas_faa_f1',
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

        agg.primary_metric_cols = util_pandas.pandas_suffix_columns(  # fixme sorting
            agg.metrics, _primary_metrics_suffixes)

        agg.display_metric_cols = util_pandas.pandas_suffix_columns(  # fixme sorting
            agg.metrics, _display_metrics_suffixes)

        _model_suffixes = ['package_fpath']
        agg.model_cols = util_pandas.pandas_suffix_columns(
            agg.params, _model_suffixes)

        _testdset_suffixes = ['test_dataset']
        agg.test_dset_cols = util_pandas.pandas_suffix_columns(
            agg.params, _testdset_suffixes)

        effective_params, mappings, hashid_to_params = agg.build_effective_params()
        agg.hashid_to_params = ub.udict(hashid_to_params)
        agg.mappings = mappings
        agg.effective_params = effective_params

        agg.resolved_info = foldin_resolved_info(agg)
        agg.resolved_params = agg.resolved_info['resolved_params']

        agg.effective_table = pd.concat([agg.metrics, agg.index, agg.effective_params], axis=1)

        agg.macro_key_to_regions = {}
        agg.region_to_tables = {}
        for region_id, idx_group in agg.index.groupby('region_id'):
            agg.region_to_tables[region_id] = {
                'metrics': agg.metrics.loc[idx_group.index],
                'params': agg.params.loc[idx_group.index],
                'index': agg.index.loc[idx_group.index],
                'effective_params': agg.effective_params.loc[idx_group.index],
                'resolved_params': agg.resolved_params.loc[idx_group.index],
            }
        agg.macro_compatible = agg.find_macro_comparable()
        agg.table = pd.concat([agg.metrics, agg.index, agg.params], axis=1)
        # agg.build_macro_tables()

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

            top_metrics = metric_group.loc[top_idxs][agg.primary_metric_cols + agg.display_metric_cols]
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

        specified_params = agg.results['specified_params']
        is_param_included = specified_params > 0

        # For each unique set of effective parameters compute a hashid
        param_cols = ub.oset(effective_params.columns).difference(agg.test_dset_cols)
        param_cols = list(param_cols - {'region_id', 'type'})
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
    group_index = agg.index.loc[group.index]
    if (group_index.value_counts('region_id') > 1).any():
        # Check if there is more than one run per-region per-param and
        # average them to keep the stats balanced.
        subgroups = group_index.groupby('region_id')
        subagg_parts = ub.ddict(list)
        for _, subgroup in subgroups:
            subgroup_parts = {}
            subgroup_parts['index'] = agg.results['index'].loc[subgroup.index]
            subgroup_parts['params'] = agg.results['params'].loc[subgroup.index]
            subgroup_parts['specified_params'] = agg.results['specified_params'].loc[subgroup.index]
            subgroup_parts['resolved_params'] = agg.resolved_info['resolved_params'].loc[subgroup.index]
            subgroup_parts['metrics'] = agg.results['metrics'].loc[subgroup.index]
            subgroup_parts['resources'] = agg.resolved_info['resources'].loc[subgroup.index]

            subagg_part = {}
            subagg_part['index'] = aggregate_param_cols(subgroup_parts['index'])
            subagg_part['params'] = aggregate_param_cols(subgroup_parts['params'], agg.test_dset_cols, allow_nonuniform=True)
            subagg_part['resolved_params'] = aggregate_param_cols(subgroup_parts['resolved_params'], agg.test_dset_cols)
            subagg_part['specified_params'] = aggregate_param_cols(subgroup_parts['specified_params'], allow_nonuniform=True)
            # Always do mean within-regions
            subagg_part['metrics'] = subgroup_parts['metrics'].aggregate('mean')
            subagg_part['resources']  = subgroup_parts['resources'].mean(numeric_only=True)
            for k, v in subagg_part.items():
                subagg_parts[k].append(v)
        group_parts = {}
        for k, v in subagg_parts.items():
            # try:
            group_parts[k] = pd.DataFrame(v).reset_index(drop=True)
            # except Exception:
            #     new = pd.concat(v, axis=1)
            #     x = pd.DataFrame(new).T
            #     group_parts[k] = x.reset_index(drop=True)
            # group_parts[k] = pd.DataFrame([_.reset_index() for _ in v]).reset_index(drop=True)
        # group_parts = {k: pd.DataFrame(v).reset_index(drop=True) for k, v in subagg_parts.items()}
    else:
        # Each region has only one result, can use it as is.
        group_parts = {}
        group_parts['metrics'] = agg.metrics.loc[group.index]
        group_parts['index'] = agg.index.loc[group.index]
        group_parts['params'] = agg.params.loc[group.index]
        group_parts['specified_params'] = agg.results['specified_params'].loc[group.index]
        group_parts['resolved_params'] = agg.resolved_info['resolved_params'].loc[group.index]
        group_parts['resources'] = agg.resolved_info['resources'].loc[group.index]

    agg_parts = {}
    agg_parts['metrics'] = group_parts['metrics'].aggregate(aggregator)
    agg_parts['resources']  = group_parts['resources'].sum(numeric_only=True)
    agg_parts['index']  = aggregate_param_cols(group_parts['index'], ['region_id'])
    agg_parts['params']  = aggregate_param_cols(group_parts['params'], agg.test_dset_cols, allow_nonuniform=True)
    agg_parts['resolved_params']  = aggregate_param_cols(group_parts['resolved_params'], agg.test_dset_cols)
    agg_parts['specified_params']  = aggregate_param_cols(group_parts['specified_params'], allow_nonuniform=True)
    return agg_parts


def out_node_matching_fpaths(out_node):
    out_template = out_node.template_value
    parser = parse.Parser(str(out_template))
    patterns = {n: '*' for n in parser.named_fields}
    pat = out_template.format(**patterns)
    mpat = util_pattern.Pattern.coerce(pat)
    fpaths = list(mpat.paths())
    return fpaths


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
