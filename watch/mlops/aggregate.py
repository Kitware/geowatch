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
import kwarray
import math
import ubelt as ub
import numpy as np
from watch.utils import util_pandas
from typing import Dict, Any
import pandas as pd
from scriptconfig import DataConfig, Value
from watch.mlops.aggregate_loader import build_tables
from watch.mlops.smart_global_helper import SMART_HELPER

try:
    from xdev import profile
except ImportError:
    profile = ub.identity


class AggregateEvluationConfig(DataConfig):
    """
    Aggregates results from multiple DAG evaluations.
    """
    target = Value(None, help=ub.paragraph(
        '''
        The input to the aggregator, which can take several forms:
        (1) the root directory of an mlops evaluation,
        (2) one or more pre-aggregated files,
        '''), nargs='+')

    output_dpath = Value('./aggregate', help=ub.paragraph(
        '''
        The path where the aggregator can write results (e.g. tables / plots).
        '''))

    pipeline = Value('joint_bas_sc', help='the name of the pipeline to run')

    export_tables = Value(False, help='if True, aggregated tables will be written to the output directory')

    plot_params = Value(False, help='if True, param plots will be drawn')

    stdout_report = Value(False, help='if True, print a report to stdout')

    io_workers = Value('avail', help='number of processes to load results')

    rois = Value('auto', help='Comma separated regions of interest')


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

        eval_type_to_results = build_tables(config)
        eval_type_to_aggregator = build_aggregators(eval_type_to_results, agg_dpath)
        agg = ub.peek(eval_type_to_aggregator.values())
        agg = eval_type_to_aggregator.get('bas_poly_eval', None)

        agg = eval_type_to_aggregator.get('bas_pxl_eval', None)

        >>> ## Execute
        >>> main(cmdline=cmdline, **kwargs)
    """

    config = AggregateEvluationConfig.cli(cmdline=cmdline, data=kwargs)
    print('config = {}'.format(ub.repr2(dict(config), nl=1)))

    eval_type_to_aggregator = coerce_aggregators(config)

    rois = config.rois
    # rois = {'KR_R001', 'KR_R002', 'BR_R002'}

    for type, agg in eval_type_to_aggregator.items():
        print(f'agg={agg}')

    timestamp = ub.timestamp()
    if config.export_tables:
        for type, agg in eval_type_to_aggregator.items():
            agg.output_dpath.ensuredir()
            fname = f'{agg.type}_{timestamp}.csv.zip'
            csv_fpath = agg.output_dpath / fname
            print(f'csv_fpath={csv_fpath}')
            agg.table.to_csv(csv_fpath, index_label=False)

    if config.stdout_report:
        for type, agg in eval_type_to_aggregator.items():
            if rois is not None:
                agg.build_macro_tables(rois)
            agg.report_best()

    if config.plot_params:
        for type, agg in eval_type_to_aggregator.items():
            build_all_param_plots(agg, rois, config)
    # automated_analysis(eval_type_to_aggregator, config)


@profile
def coerce_aggregators(config):
    from watch.utils import util_path
    input_targets = util_path.coerce_patterned_paths(config.target)
    eval_type_to_tables = ub.ddict(list)
    for target in input_targets:
        if target.is_dir():
            # Assume Pipeline Output dir
            eval_type_to_results = build_tables(target, config.pipeline, config.io_workers)
            for type, results in eval_type_to_results.items():
                table = pd.concat(list(results.values()), axis=1)
                eval_type_to_tables[type].append(table)
        if target.is_file():
            # Assume CSV file
            table = pd.read_csv(target, low_memory=False)
            if len(table):
                type = table['node'].iloc[0]
                eval_type_to_tables[type].append(table)

    output_dpath = ub.Path(config['output_dpath'])

    eval_type_to_aggregator = {}
    for type, tables in eval_type_to_tables.items():
        table = tables[0] if len(tables) == 1 else pd.concat(tables).reset_index(drop=True)
        agg = Aggregator(table)
        agg.output_dpath = output_dpath
        agg.build()
        eval_type_to_aggregator[type] = agg
    return eval_type_to_aggregator


@profile
def build_all_param_plots(agg, rois, config):
    resolved_params = util_pandas.DotDictDataFrame(agg.resolved_params)

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
        agg.table.loc[:, prefix + '.effective_batch_size'] = val_effective_bsize
    agg.build()

    agg.build_macro_tables(rois)

    macro_table = agg.region_to_tables[agg.primary_macro_region].copy()
    single_table = agg.table

    fillna_cols = single_table.columns.intersection(agg.resolved_params.columns.union(agg.resolved_params.columns))
    fillna_cols = macro_table.columns.intersection(agg.resolved_params.columns.union(agg.resolved_params.columns))
    single_table.loc[:, fillna_cols] = single_table.loc[:, fillna_cols].fillna('None')
    macro_table.loc[:, fillna_cols] = macro_table.loc[:, fillna_cols].fillna('None')

    macro_table = macro_table.applymap(lambda x: str(x) if isinstance(x, list) else x)
    single_table = single_table.applymap(lambda x: str(x) if isinstance(x, list) else x)

    if 0:
        SMART_HELPER.old_hacked_model_case(macro_table)

    modifier = SMART_HELPER.label_modifier()
    param_to_palette = SMART_HELPER.shared_palletes(macro_table)
    if 0:
        SMART_HELPER.mark_star_models(macro_table)

    # agg = plotter.agg
    agg_group_dpath = (agg.output_dpath / ('all_params' + ub.timestamp())).ensuredir()

    plotter = ParamPlotter(agg)

    plotter.agg_group_dpath = agg_group_dpath
    plotter.param_to_palette = param_to_palette
    plotter.modifier = modifier
    plotter.macro_table = macro_table
    plotter.single_table = single_table
    plotter.rois = rois

    for vantage in plotter.vantage_points:
        print(vantage['name'])
        plotter.plot_vantage(vantage)
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
        vantage_points = SMART_HELPER.default_vantage_points(agg.type)
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
        # main_metric = 'bas_poly_eval.metrics.bas_faa_f1'
        main_metric = agg.primary_metric_cols[0]

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
        except Exception as ex:
            print(f'ex={ex}')

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

        blocklist = {
            'resolved_params.bas_poly_eval.pred_sites',
            'resolved_params.bas_poly_eval.gt_dpath',
            'resolved_params.bas_poly_eval.true_site_dpath',
            'resolved_params.bas_poly_eval.true_region_dpath',
            'resolved_params.bas_poly_eval.out_dir',
            'resolved_params.bas_poly_eval.merge',
            'resolved_params.bas_poly_eval.merge_fpath',
            'resolved_params.bas_poly_eval.merge_fbetas',
            'resolved_params.bas_poly_eval.tmp_dir',
            'resolved_params.bas_poly_eval.enable_viz',
            'resolved_params.bas_poly_eval.name',
            'resolved_params.bas_poly_eval.use_cache',
            'resolved_params.bas_poly_eval.load_workers',
            'resolved_params.bas_poly.in_file',
            'resolved_params.bas_poly.out_kwcoco',
            'resolved_params.bas_poly.out_sites_dir',
            'resolved_params.bas_poly.out_site_summaries_dir',
            'resolved_params.bas_poly.out_sites_fpath',
            'resolved_params.bas_poly.out_site_summaries_fpath',
            'resolved_params.bas_poly.in_file_gt',
            'resolved_params.bas_poly.region_id',
            'resolved_params.bas_poly.default_track_fn',
            'resolved_params.bas_poly.site_summary',
            'resolved_params.bas_poly.clear_annots',
            'resolved_params.bas_poly.append_mode',
            'resolved_params.bas_pxl.config_file',
            'resolved_params.bas_pxl.write_out_config_file_to_this_path',
            'resolved_params.bas_pxl.datamodule',
            'resolved_params.bas_pxl.pred_dataset',
            'resolved_params.bas_pxl.devices',
            'resolved_params.bas_pxl.with_change',
            'resolved_params.bas_pxl.with_class',
            'resolved_params.bas_pxl.with_saliency',
            'resolved_params.bas_pxl.compress',
            'resolved_params.bas_pxl.track_emissions',
            'resolved_params.bas_pxl.quantize',
            'resolved_params.bas_pxl.clear_annots',
            'resolved_params.bas_pxl.write_workers',
            'resolved_params.bas_pxl.write_preds',
            'resolved_params.bas_pxl.write_probs',
            'resolved_params.bas_pxl.train_dataset',
            'resolved_params.bas_pxl.vali_dataset',
            'resolved_params.bas_pxl.test_dataset',
            'resolved_params.bas_pxl.batch_size',
            'resolved_params.bas_pxl.normalize_inputs',
            'resolved_params.bas_pxl.num_workers',
            'resolved_params.bas_pxl.torch_sharing_strategy',
            'resolved_params.bas_pxl.torch_start_method',
            'resolved_params.bas_pxl.sqlview',
            'resolved_params.bas_pxl.max_epoch_length',
            'resolved_params.bas_pxl.use_centered_positives',
            'resolved_params.bas_pxl.use_grid_positives',
            'resolved_params.bas_pxl.use_grid_valid_regions',
            'resolved_params.bas_pxl.neg_to_pos_ratio',
            'resolved_params.bas_pxl.use_grid_cache',
            'resolved_params.bas_pxl.ignore_dilate',
            'resolved_params.bas_pxl.weight_dilate',
            'resolved_params.bas_pxl.min_spacetime_weight',
            'resolved_params.bas_pxl.upweight_centers',
            'resolved_params.bas_pxl.upweight_time',
            'resolved_params.bas_pxl.dist_weights',
            'resolved_params.bas_pxl.balance_areas',
            'resolved_params.bas_pxl.resample_invalid_frames',
            'resolved_params.bas_pxl.downweight_nan_regions',
            'resolved_params.bas_pxl.temporal_dropout',
            'resolved_params.bas_pxl_fit.accelerator',
            'resolved_params.bas_pxl_fit.accumulate_grad_batches',
            'resolved_params.bas_pxl_fit.datamodule',
            'resolved_params.bas_pxl_fit.devices',
            'resolved_params.bas_pxl_fit.gradient_clip_algorithm',
            'resolved_params.bas_pxl_fit.gradient_clip_val',
            'resolved_params.bas_pxl_fit.max_epochs',
            'resolved_params.bas_pxl_fit.max_steps',
            'resolved_params.bas_pxl_fit.method',
            'resolved_params.bas_pxl_fit.name',
            'resolved_params.bas_pxl_fit.patience',
            'resolved_params.bas_pxl_fit.precision',
            'resolved_params.bas_pxl_fit.sqlview',
            'resolved_params.bas_pxl_fit.stochastic_weight_avg',
            'resolved_params.bas_pxl_fit.inference_mode',
            'resolved_params.bas_pxl_fit.use_grid_cache',
            'resolved_params.bas_pxl_fit.use_grid_valid_regions',
            'resolved_params.bas_pxl_eval.balance_area',
            'resolved_params.bas_pxl_eval.draw_curves',
            'resolved_params.bas_pxl_eval.draw_heatmaps',
            'resolved_params.bas_pxl_eval.draw_workers',
            'resolved_params.bas_pxl_eval.eval_dpath',
            'resolved_params.bas_pxl_eval.eval_fpath',
            'resolved_params.bas_pxl_eval.pred_dataset',
            'resolved_params.bas_pxl_eval.resolution',
            'resolved_params.bas_pxl_eval.score_space',
            'resolved_params.bas_pxl_eval.true_dataset',
            'resolved_params.bas_pxl_eval.viz_thresh',
            'resolved_params.bas_pxl_eval.workers',
        }
        resolved_params = util_pandas.DotDictDataFrame(macro_table).subframe('resolved_params', drop_prefix=False)
        valid_cols = resolved_params.columns.difference(blocklist)
        resolved_params = resolved_params[valid_cols]

        DO_STAT_ANALYSIS = True
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
            # print('analysis.varied = {}'.format(ub.urepr(analysis.varied, nl=2)))
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

        from kwcoco.metrics.drawing import concice_si_display
        for rank, param_name in enumerate(ub.ProgIter(ranked_params, desc='plot param for ' + vantage['name'], verbose=3)):

            param_dpath = (param_group_dpath / param_name).ensuredir().resolve()

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
            if anova_rank_p is not None:
                ax.set_title(f'BAS Results (n={len(macro_table)})\n'
                             f'Macro Analysis over {ub.urepr(rois, sv=1, nl=0)}\n'
                             f'Effect of {param_name}: anova_rank_p={concice_si_display(anova_rank_p)}')
            else:
                ax.set_title(f'BAS Results (n={len(macro_table)})\n'
                             f'Macro Analysis over {ub.urepr(rois, sv=1, nl=0)}')
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
                ax = sns.boxplot(data=macro_table, x=param_name, y=y, **snskw)
                freq_mapper_box.relabel_xticks(ax)
                if anova_rank_p is not None:
                    ax.set_title(f'BAS Results (n={len(macro_table)})\n'
                                 f'Macro Analysis over {ub.urepr(rois, sv=1, nl=0)}')
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


def make_summary_analysis(agg1, config, dpath=None):
    if dpath is None:
        output_dpath = ub.Path(config['root_dpath'] / 'aggregate')
    else:
        output_dpath = dpath
    agg_group_dpath = output_dpath / ('agg_summary_params2_v3')
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


class AggregatorAnalysisMixin:
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
            agg.table, metrics=metrics_of_interest)
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
            metric_group = group[group.columns.intersection(agg.metrics.columns)]
            metric_group = metric_group.sort_values(agg.primary_metric_cols)

            top_idxs = util_pandas.pandas_argmaxima(metric_group, agg.primary_metric_cols, k=top_k)

            final_display_cols = list(ub.oset(agg.primary_metric_cols + agg.display_metric_cols))
            top_metrics = metric_group.loc[top_idxs][final_display_cols]
            # top_metrics = top_metrics[agg.primary_metric_cols + agg.display_metric_cols]
            top_indexes = group[group.columns.intersection(agg.index.columns)].loc[top_idxs]
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
        """
        Given a set of ROIs, find groups in the comparable regions that contain
        all of the requested ROIs.
        """
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

    @profile
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
        aggregator.update({c: 'sum' for c in agg.resources.select_dtypes(np.number).columns})

        # Gather groups that can be aggregated
        comparable_groups = agg.gather_macro_compatable_groups(regions_of_interest)
        if len(comparable_groups) == 0:
            print(ub.paragraph(
                f'''
                WARNING: Failed to build macro results. No comparable groups
                for rois={rois}
                '''))
        else:
            # Macro aggregaet comparable groups
            macro_rows = []
            for group in comparable_groups:
                macro_row = macro_aggregate(agg, group, aggregator)
                macro_rows.append(macro_row)

            macro_table = pd.DataFrame(macro_rows).reset_index(drop=True)
            agg.region_to_tables.pop(macro_key, None)
            agg.macro_key_to_regions.pop(macro_key, None)
            agg.macro_key_to_regions[macro_key] = regions_of_interest
            agg.region_to_tables[macro_key] = macro_table
            return macro_table


@profile
def aggregate_param_cols(df, aggregator=None, hash_cols=None, allow_nonuniform=False):
    """
    Aggregates parameter columns. Specified hash_cols should be
    dataset-specific columns to be hashed. All other columns should
    be effectively the same, otherwise we will warn.

    TODO:
        - [ ] optimizatize this
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
    blocklist = {'fpath'}
    hash_cols = ['region_id'] + agg.test_dset_cols

    table = agg.table.loc[group.index].drop(blocklist, axis=1)

    # Check if there is more than one run per-region per-param and
    # average them to keep the stats balanced.
    has_multiple_param_runs = (table['region_id'].value_counts() > 1).any()
    if has_multiple_param_runs:

        # All aggregations are the mean when combining over the same region id
        sub_aggregator = {c: 'mean' for c in aggregator.keys()}
        sub_aggregator.update({c: 'mean' for c in agg.resources.columns})

        sub_hash_cols = agg.test_dset_cols
        subgroups = table.groupby('region_id')
        subrows = []
        for _, subgroup in subgroups:
            subrow = aggregate_param_cols(subgroup, aggregator=sub_aggregator, hash_cols=sub_hash_cols, allow_nonuniform=True)
            subrows.append(subrow)
        # Now each region is in exactly one row.
        table = pd.DataFrame(subrows)

    macro_row = aggregate_param_cols(table, aggregator=aggregator, hash_cols=hash_cols, allow_nonuniform=True)
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
