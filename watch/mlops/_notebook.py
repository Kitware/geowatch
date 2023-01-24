# flake8: noqa
import math
import pandas as pd
import ubelt as ub
from watch.mlops.aggregate import hash_param
from watch.mlops.aggregate import fix_duplicate_param_hashids
from watch.utils import util_pandas


def _setup():
    from watch.mlops.aggregate import AggregateEvluationConfig
    from watch.mlops.aggregate import build_tables
    from watch.mlops.aggregate import build_aggregators
    import watch
    expt_dvc_dpath = watch.find_dvc_dpath(tags='phase2_expt', hardware='auto')
    cmdline = 0
    kwargs = {
        'root_dpath': expt_dvc_dpath / '_testpipe',
        'pipeline': 'bas',
        'io_workers': 10,
        'freeze_cache': 0,
        # 'pipeline': 'joint_bas_sc_nocrop',
        # 'root_dpath': expt_dvc_dpath / '_testsc',
        #'pipeline': 'sc',
    }
    config = AggregateEvluationConfig.legacy(cmdline=cmdline, data=kwargs)
    eval_type_to_results = build_tables(config)
    eval_type_to_aggregator = build_aggregators(eval_type_to_results)
    agg = ub.peek(eval_type_to_aggregator.values())
    agg = eval_type_to_aggregator.get('bas_poly_eval', None)
    print(f'agg={agg}')
    rois = {'KR_R001', 'KR_R002', 'BR_R002'}
    print(f'rois={rois}')


def build_all_param_plots(agg):
    from watch.utils import util_kwplot
    import numpy as np
    import kwplot
    sns = kwplot.autosns()
    plt = kwplot.autoplt()
    # metric_cols = [c for c in df.columns if 'metrics.' in c]
    kwplot.close_figures()

    rois = {'KR_R001', 'KR_R002', 'BR_R002'}
    macro_tables = agg.build_macro_tables(rois)

    macro_results = agg.region_to_tables[agg.primary_macro_region].copy()
    single_results = {
        'index': agg.index,
        'metrics': agg.metrics,
        'resolved_params': agg.resolved_params,
        'resources': agg.resolved_info['resources'],
    }

    if True:
        # Shorten columns
        mappers = {}
        mappers['metrics'] = {c: c.split('.')[-1] for c in macro_results['metrics'].columns}
        mappers['resolved_params'] = {c: c.replace('bas_poly_eval.params.', '')
                                      for c in macro_results['resolved_params'].columns}

        for k, mapper in mappers.items():
            macro_results[k] = macro_results[k].rename(mappers[k], axis=1)
            single_results[k] = single_results[k].rename(mappers[k], axis=1)

    _parts = list((ub.udict(macro_results) & {
        'index', 'metrics', 'resolved_params', 'resources'}).values())
    macro_table = pd.concat(_parts, axis=1)
    single_table = pd.concat(list(single_results.values()), axis=1)
    single_table = single_table.fillna('None')
    macro_table = macro_table.fillna('None')

    # x = 'bas_poly_eval.metrics.bas_tpr'
    # y = 'bas_poly_eval.metrics.bas_ppv'
    # x = 'bas_poly_eval.metrics.bas_space_FAR'
    # y = 'bas_poly_eval.metrics.bas_tpr'
    x = 'bas_ffpa'
    y = 'bas_f1'

    agg_dpath = ub.Path(config['root_dpath'] / 'aggregate')
    agg_group_dpath = (agg_dpath / ('all_params' + ub.timestamp())).ensuredir()

    def finalize_figure(fig, fpath):
        fig.set_size_inches(np.array([6.4, 4.8]) * 1.0)
        fig.tight_layout()
        fig.savefig(fpath)
        util_kwplot.cropwhite_ondisk(fpath)

    fig = kwplot.figure(fnum=2, doclf=True)
    ax = sns.scatterplot(data=single_table, x=x, y=y, hue='region_id')
    ax.set_title(f'BAS Experiment Results (n={len(agg)})')
    ax.set_xscale('log')
    fpath = agg_group_dpath / 'single_results.png'
    finalize_figure(fig, fpath)
    # ax.set_xlim(0, np.quantile(agg.metrics[x], 0.99))
    # ax.set_xlim(1e-2, np.quantile(agg.metrics[x], 0.99))

    macro_table = pd.concat(_parts, axis=1)
    fig = kwplot.figure(fnum=3, doclf=True)
    ax = sns.scatterplot(data=macro_table, x=x, y=y, hue='region_id')
    ax.set_title(f'BAS Experiment Results (n={len(macro_table)})\nMacro Results over ' + str(rois))
    ax.set_xscale('log')
    fpath = agg_group_dpath / 'macro_results.png'
    finalize_figure(fig, fpath)
    # ax.set_xlim(1e-2, npe.quantile(agg.metrics[x], 0.99))
    # ax.set_xlim(1e-2, 0.7)

    ### Build param analysis
    from watch.utils import result_analysis
    results = {'params': macro_table[macro_results['resolved_params'].columns],
               'metrics': macro_table[macro_results['metrics'].columns]}
    analysis = result_analysis.ResultAnalysis(results, metrics=agg.primary_metric_cols)
    analysis.build()
    analysis.analysis()
    analysis.varied

    for stats in analysis.statistics:
        stats['moments']
        anova_rank_p = stats['anova_rank_p']
        hue = stats['param_name']
        print(f'anova_rank_p={anova_rank_p}')
        print(f'hue={hue}')

        fig = kwplot.figure(fnum=4, doclf=True)
        # ax = sns.scatterplot(data=macro_table, x=x, y=y, hue=agg.model_cols[0])
        ax = sns.scatterplot(data=macro_table, x=x, y=y, hue=hue)
        ax.set_title(f'BAS Experiment Results (n={len(macro_table)})\n'
                     f'Macro Results over {rois}')
        ax.set_xscale('log')

    # ax.set_xlim(1e-2, npe.quantile(agg.metrics[x], 0.99))
    # ax.set_xlim(1e-2, 0.7)

def check_baseline(eval_type_to_aggregator):
    from watch.utils.util_param_grid import DotDictDataFrame
    models_of_interest = [
        'package_epoch0_step41',
    ]
    agg0 = eval_type_to_aggregator.get('bas_poly_eval', None)
    agg1 = agg0.filterto(models=models_of_interest)
    _ = agg1.report_best()

    DotDictDataFrame(agg1.effective_params)
    flags = DotDictDataFrame(agg1.effective_params)['thresh'] == 0.1
    agg2 = agg1.compress(flags.values)

    agg3 = agg2.filterto(param_hashids=['peercprsgafm'])
    agg3.build_macro_tables()
    agg3.report_best()


def compare_simplify_tolerence(agg0, config):
    # Filter to only parameters that are A / B comparable
    parameter_of_interest = 'bas_poly_eval.params.bas_poly.polygon_simplify_tolerance'
    # groups = agg0.effective_params.groupby(parameter_of_interest)
    if parameter_of_interest not in agg0.effective_params.columns:
        raise ValueError(f'Missing parameter of interest: {parameter_of_interest}')

    other_params = agg0.effective_params.columns.difference({parameter_of_interest}).tolist()
    candidate_groups = agg0.effective_params.groupby(other_params, dropna=False)

    comparable_groups = []
    for value, group in candidate_groups:
        # if len(group) > 1:
        if len(group) == 4:
            if len(group[parameter_of_interest].unique()) == 4:
                comparable_groups.append(group)

    comparable_idxs = sorted(ub.flatten([group.index for group in comparable_groups]))
    comparable_agg1 = agg0.filterto(index=comparable_idxs)
    agg = comparable_agg1

    # _ = comparable_agg1.report_best()

    selector = {'KR_R001', 'KR_R002', 'BR_R002'}
    # Find the best result for the macro region
    rois = selector  # NOQA
    macro_results = comparable_agg1.build_single_macro_table(selector)
    top_macro_id = macro_results['metrics'].sort_values(comparable_agg1.primary_metric_cols, ascending=False).index[0]

    print(macro_results['index'].loc[top_macro_id])

    # Find the corresponding A / B for each region.
    # Get the param hash each region should have
    top_params = macro_results['effective_params'].loc[top_macro_id]
    # top_value = top_params[parameter_of_interest]
    top_flags = macro_results['specified_params'].loc[top_macro_id] > 0
    top_flags[comparable_agg1.test_dset_cols] = False
    comparable_agg1.hashid_to_params[macro_results['index'].loc[top_macro_id]['param_hashid']]

    params_of_interest = []
    for value in comparable_agg1.effective_params[parameter_of_interest].unique():
        params_row = top_params.copy()
        flags = top_flags.copy()
        if isinstance(value, float) and math.isnan(value):
            flags[parameter_of_interest] = False | flags[parameter_of_interest]
        else:
            flags[parameter_of_interest] = True | flags[parameter_of_interest]
        params_row[parameter_of_interest] = value
        params = params_row[flags].to_dict()
        param_hashid = hash_param(params)
        if not (param_hashid in comparable_agg1.index['param_hashid'].values):
            flags[parameter_of_interest] = not flags[parameter_of_interest]
            params_row[parameter_of_interest] = value
            params = params_row[flags].to_dict()
            param_hashid = hash_param(params)
            if not (param_hashid in comparable_agg1.index['param_hashid'].values):
                raise AssertionError
        params_of_interest.append(param_hashid)

    comparable_agg2 = comparable_agg1.filterto(param_hashids=params_of_interest)

    _ = comparable_agg2.report_best(top_k=10)

    agg_dpath = ub.Path(config['root_dpath'] / 'aggregate')
    agg_group_dpath = (agg_dpath / ('inspect_simplify_' + ub.timestamp())).ensuredir()

    # Given these set of A/B values, visualize each region
    for region_id, group in comparable_agg2.index.groupby('region_id'):
        group_agg = comparable_agg2.filterto(index=group.index)
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
            scores_of_interest = util_pandas.pandas_shorten_columns(comparable_agg2.metrics).loc[id, ['bas_tp', 'bas_fp', 'bas_fn', 'bas_f1']]
            scores_of_interest = ub.udict(scores_of_interest.to_dict())
            text = ub.urepr(scores_of_interest.map_values(concice_si_display), nobr=1, si=1, compact=1)
            new_img = kwimage.draw_header_text(viz_img, param_hashid + '\n' + text)
            kwimage.imwrite(agg_group_dpath / f'summary_{region_id}_{param_hashid}.jpg', new_img)

    # Hacked result analysis

    from watch.utils import result_analysis
    metrics_of_interest = comparable_agg1.primary_metric_cols

    params = comparable_agg1.results['params']
    specified = comparable_agg1.results['specified_params']  # NOQA

    params.loc[:, parameter_of_interest] = params.loc[:, parameter_of_interest].fillna('None')

    analysis = result_analysis.ResultAnalysis(
        comparable_agg1.results, metrics=metrics_of_interest)
    analysis.results
    analysis.analysis()

    ###
    comparable_agg1.build_macro_tables()

    # Check to see if simplify shrinks the size of our files
    rows = []
    for idx, fpath in comparable_agg1.fpaths.iteritems():
        poly_coco_fpath = list((fpath.parent / '.pred').glob('bas_poly/*/poly.kwcoco.json'))[0]
        sites_dpath = list((fpath.parent / '.pred').glob('bas_poly/*/site_summaries'))[0]
        size1 = poly_coco_fpath.stat().st_size
        size2 = sum([p.stat().st_size for p in sites_dpath.ls()])
        value = comparable_agg1.effective_params.loc[idx, parameter_of_interest]
        rows.append({
            'coco_size': size1,
            'site_size': size2,
            'param_value': value,
        })
    for val, group in pd.DataFrame(rows).groupby('param_value', dropna=False):
        print(f'val={val}')
        import xdev
        print(len(group))
        size1 = xdev.byte_str(group['coco_size'].sum())
        size2 = xdev.byte_str(group['site_size'].sum())
        print(f'size1={size1}')
        print(f'size2={size2}')

    macro_region_id = list(comparable_agg1.macro_key_to_regions.keys())[-1]
    regions_of_interest = comparable_agg1.macro_key_to_regions[macro_region_id]  # NOQA
    tables = comparable_agg1.region_to_tables[macro_region_id]

    effective_params = tables['effective_params']
    metrics = tables['metrics']
    index = tables['index']

    table = pd.concat([index, effective_params, metrics], axis=1)
    table = table.fillna('None')

    main_metric = agg.primary_metric_cols[0]

    results = []
    for idx, row in enumerate(table.to_dict('records')):
        row = ub.udict(row)
        row_metrics = row & set(metrics.keys())
        row_params = row & set(effective_params.keys())
        result = result_analysis.Result(str(idx), row_params, row_metrics)
        results.append(result)

    analysis = result_analysis.ResultAnalysis(
        results, metrics=[main_metric],
        metric_objectives={main_metric: 'max'}
    )
    # self = analysis
    analysis.analysis()
    analysis.report()

    pd.Index.union(*[group.index for group in comparable_groups])


def comparse_single_param(agg0, config):

    agg0 = fix_duplicate_param_hashids(agg0)
    agg0.resolved_params = agg0.resolved_params.applymap(lambda x: str(x) if isinstance(x, list) else x)
    resolved_params = agg0.resolved_params.fillna('None')
    parameters_of_interest = [
        'bas_poly_eval.params.bas_pxl.output_space_scale',
        'bas_poly_eval.params.bas_pxl.input_space_scale',
        'bas_poly_eval.params.bas_pxl.window_space_scale',
    ]
    other_params = resolved_params.columns.difference(parameters_of_interest).tolist()
    candidate_groups = resolved_params.groupby(other_params, dropna=False)

    comparable_groups = []
    for value, group in candidate_groups:
        if len(group) > 1:
            has_variation = ~group[parameters_of_interest].apply(ub.allsame, axis=0)
            if has_variation.any():
                comparable_groups.append(group)
            else:
                print('unsure why this can happen')

    comparable_idxs = sorted(ub.flatten([group.index for group in comparable_groups]))
    comparable_agg1 = agg0.filterto(index=comparable_idxs)
    agg = comparable_agg1   # NOQA

    selector = {'KR_R001', 'KR_R002', 'BR_R002'}
    # Find the best result for the macro region
    rois = selector  # NOQA
    macro_results = comparable_agg1.build_single_macro_table(selector)
    top_macro_id = macro_results['metrics'].sort_values(comparable_agg1.primary_metric_cols, ascending=False).index[0]

    print(macro_results['index'].loc[top_macro_id])

    # Find the corresponding A / B for each region.
    # Get the param hash each region should have
    top_params = macro_results['resolved_params'].loc[top_macro_id]  # NOQA
    # top_value = top_params[parameter_of_interest]
    top_flags = macro_results['specified_params'].loc[top_macro_id] > 0
    top_flags[comparable_agg1.test_dset_cols] = False
    comparable_agg1.hashid_to_params[macro_results['index'].loc[top_macro_id]['param_hashid']]

    # comparable_agg1
    # params_of_interest = []
    # for value in comparable_agg1.resolved_params[parameter_of_interest].unique():
    #     params_row = top_params.copy()
    #     flags = top_flags.copy()
    #     if isinstance(value, float) and math.isnan(value):
    #         flags[parameter_of_interest] = False | flags[parameter_of_interest]
    #     else:
    #         flags[parameter_of_interest] = True | flags[parameter_of_interest]
    #     params_row[parameter_of_interest] = value
    #     params = params_row[flags].to_dict()
    #     param_hashid = hash_param(params)
    #     if not (param_hashid in comparable_agg1.index['param_hashid'].values):
    #         flags[parameter_of_interest] = not flags[parameter_of_interest]
    #         params_row[parameter_of_interest] = value
    #         params = params_row[flags].to_dict()
    #         param_hashid = hash_param(params)
    #         if not (param_hashid in comparable_agg1.index['param_hashid'].values):
    #             raise AssertionError
    #     params_of_interest.append(param_hashid)


def visualize_cases(agg):
    region_id = 'macro_02_19bfe3'
    regions_of_interest = agg.macro_key_to_regions.get(region_id, [region_id])
    param_hashid = 'ab43161b'

    to_inspect_idxs = []
    for region_id in regions_of_interest:
        tables = agg.region_to_tables[region_id]

        grouped_indexes = tables['index'].groupby('param_hashid')
        rows = grouped_indexes.get_group(param_hashid)
        assert len(rows) == 1
        to_inspect_idxs.append(rows.index[0])

    for index in to_inspect_idxs:
        pass
    index = to_inspect_idxs[0]

    agg.index.iloc[index]

    agg.results['param_types'][index]
    eval_fpath = agg.results['fpaths'][index]  # NOQA


def plot_stats_tables(agg, config):
    """
    agg = eval_type_to_aggregator.get('bas_poly_eval', None)
    """
    # from watch.mlops import smart_result_parser
    # for fpath in fpaths:
    #     ...
    #     result = smart_result_parser.load_eval_act_poly(fpath, None)
    #     print(result['metrics']['sc_macro_f1'])

    import numpy as np
    from watch.utils import util_kwplot
    import kwplot
    sns = kwplot.autosns()
    plt = kwplot.autoplt()
    # metric_cols = [c for c in df.columns if 'metrics.' in c]
    kwplot.close_figures()

    metric = 'sc_poly_eval.metrics.sc_macro_f1'

    agg_dpath = ub.Path(config['root_dpath'] / 'aggregate')

    rois = {'BR_R002', 'KR_R001', 'KR_R002'}
    agg.build_single_macro_table(rois)
    macro_key = agg.primary_macro_region

    agg_group_dpath = (agg_dpath / (f'stats_tables_{macro_key}' + ub.timestamp())).ensuredir()

    # df['sc_poly_eval.metrics.macro_f1_active']
    for metric in agg.primary_metric_cols:
        node_id = metric.split('.')[0]
        metric_name = metric.split('.')[-1]

        df = pd.concat([agg.metrics, agg.index, agg.resolved_params], axis=1)

        plt.figure()
        ax = sns.boxplot(data=df, x='region_id', y=metric)
        fig = ax.figure
        ax.set_ylabel(metric_name)
        ax.set_title(f'{node_id} {metric_name} n={len(agg)}')
        xtick_mapping = {
            region_id: f'{region_id}\n(n={num})'
            for region_id, num in df.groupby('region_id').size().to_dict().items()
        }
        util_kwplot.relabel_xticks(xtick_mapping, ax=ax)

        fname = f'boxplot_{metric}.png'
        fpath = agg_group_dpath / fname
        fig.set_size_inches(np.array([6.4, 4.8]) * 1.4)
        fig.tight_layout()
        fig.savefig(fpath)
        util_kwplot.cropwhite_ondisk(fpath)

        ### Macro stats analysis
        regions_of_interest = agg.macro_key_to_regions[macro_key]
        print(f'regions_of_interest={regions_of_interest}')
        tables = agg.region_to_tables[agg.primary_macro_region]
        table = pd.concat(list((ub.udict(tables) & {
            'index', 'resolved_params', 'metrics'}).values()), axis=1)
        table = table.fillna('None')

        # Fix
        table['bas_poly_eval.params.bas_pxl.chip_dims'] = table['bas_poly_eval.params.bas_pxl.chip_dims'].apply(lambda x: str(x) if isinstance(x, list) else x)

        from watch.utils import result_analysis
        results = []
        for idx, row in enumerate(table.to_dict('records')):
            row = ub.udict(row)
            row_metrics = row & set(tables['metrics'].keys())
            row_params = row & set(tables['resolved_params'].keys())
            result = result_analysis.Result(str(idx), row_params, row_metrics)
            results.append(result)
        analysis = result_analysis.ResultAnalysis(
            results, metrics=[metric],
            metric_objectives={metric: 'max'}
        )
        # self = analysis
        analysis.analysis()
        analysis.report()

        kwplot.close_figures()

        for param in analysis.varied:
            fig = plt.figure()
            fig.clf()
            ax = sns.boxplot(data=table, x=param, y=metric)
            fig = ax.figure
            ax.set_ylabel(metric_name)
            ax.set_title(f'{node_id} {macro_key} {metric_name} {param} n={len(table)}')

            xtick_mapping = {
                str(value): f'{value}\nn={num}'
                for value, num in table.groupby(param, dropna=False).size().to_dict().items()
            }
            util_kwplot.relabel_xticks(xtick_mapping, ax=ax)

            # util_kwplot.relabel_xticks(xtick_mapping, ax=ax)
            fname = f'boxplot_{macro_key}_{metric}_{param}.png'
            fpath = agg_group_dpath / fname
            # fig.set_size_inches(np.array([6.4, 4.8]) * 1.4)
            fig.set_size_inches(np.array([16, 9]) * 1.0)
            fig.tight_layout()
            fig.savefig(fpath)
            util_kwplot.cropwhite_ondisk(fpath)


def custom_analysis(eval_type_to_aggregator, config):

    macro_groups = [
        {'KR_R001', 'KR_R002'},
        {'KR_R001', 'KR_R002', 'US_R007'},
        {'BR_R002', 'KR_R001', 'KR_R002', 'AE_R001'},
        {'BR_R002', 'KR_R001', 'KR_R002', 'AE_R001', 'US_R007'},
    ]
    rois = macro_groups  # NOQA

    agg0 = eval_type_to_aggregator.get('bas_poly_eval', None)
    agg_dpath = ub.Path(config['root_dpath'] / 'aggregate')

    if agg0 is not None:
        agg0_ = fix_duplicate_param_hashids(agg0)
        # params_of_interest = ['414d0b37']
        # v0_params_of_interest = [
        #     '414d0b37', 'ab43161b', '34bed2b3', '8ac5594b']
        params_of_interest = list({
            '414d0b37': 'ffmpktiwwpbx',
            'ab43161b': 'nriolrrxfbco',
            '34bed2b3': 'eflhsmpfhgsj',
            '8ac5594b': 'lbxwewpkfwcd',
        }.values())

        param_of_interest = params_of_interest[0]
        subagg1 = agg0_.filterto(param_hashids=param_of_interest)
        subagg1.build_macro_tables(macro_groups)
        _ = subagg1.report_best()

        print(ub.repr2(subagg1.results['fpaths']['fpath'].to_list()))

        agg_group_dpath = agg_dpath / (f'agg_params_{param_of_interest}')
        agg_group_dpath = agg_group_dpath.ensuredir()

        # Make a directory with a summary over all the regions
        summary_dpath = (agg_group_dpath / 'summary').ensuredir()

        # make a analysis link to the final product
        for idx, fpath in ub.ProgIter(list(subagg1.fpaths.iteritems())):
            region_id = subagg1.index.loc[idx]['region_id']
            param_hashid = subagg1.index.loc[idx]['param_hashid']
            link_dpath = agg_group_dpath / region_id
            ub.symlink(real_path=fpath.parent, link_path=link_dpath)
            # ub.symlink(real_path=region_viz_fpath, link_path=summary_dpath / region_viz_fpath.name)
            import kwimage
            from kwcoco.metrics.drawing import concice_si_display
            region_viz_fpaths = list((fpath.parent / 'region_viz_overall').glob('*_detailed.png'))
            assert len(region_viz_fpaths) == 1
            region_viz_fpath = region_viz_fpaths[0]
            viz_img = kwimage.imread(region_viz_fpath)
            scores_of_interest = util_pandas.pandas_shorten_columns(subagg1.metrics).loc[idx, ['bas_tp', 'bas_fp', 'bas_fn', 'bas_f1']]
            scores_of_interest = ub.udict(scores_of_interest.to_dict())
            text = ub.urepr(scores_of_interest.map_values(concice_si_display), nobr=1, si=1, compact=1)
            new_img = kwimage.draw_header_text(viz_img, param_hashid + '\n' + text)
            kwimage.imwrite(summary_dpath / f'summary_{region_id}.jpg', new_img)

            # eval_dpath = bas_poly_eval_confusion_analysis(eval_fpath)
            # TODO: use the region_id.
            # ub.symlink(real_path=eval_dpath, link_path=agg_group_dpath / eval_dpath.name)

        #### FIND BEST FOR EACH MODEL

        agg0_.build_macro_tables(macro_groups)

        params_of_interest = []
        macro_metrics = agg0_.region_to_tables[agg0_.primary_macro_region]['metrics']
        macro_params = agg0_.region_to_tables[agg0_.primary_macro_region]['effective_params']
        macro_index = agg0_.region_to_tables[agg0_.primary_macro_region]['index']
        for model_name, group in macro_params.groupby(agg0_.model_cols):
            group_metrics = macro_metrics.loc[group.index]
            group_metrics = group_metrics.sort_values(agg0_.primary_metric_cols, ascending=False)
            group_metrics.iloc[0]
            param_hashid = macro_index.loc[group_metrics.index[0]]['param_hashid']
            params_of_interest.append(param_hashid)

        agg1 = agg0_.filterto(param_hashids=params_of_interest)
        agg1.build_macro_tables(macro_groups)

        # Get a shortlist of the top models
        hash_part = agg1.region_to_tables[agg1.primary_macro_region]['index']['param_hashid']
        model_part = agg1.region_to_tables[agg1.primary_macro_region]['effective_params'][agg0_.model_cols]
        metric_part = agg1.region_to_tables[agg1.primary_macro_region]['metrics'][agg0_.primary_metric_cols]
        table = pd.concat([hash_part, model_part, metric_part], axis=1)
        table = table.sort_values(agg0_.primary_metric_cols, ascending=False)

        blocklist = {
            'Drop4_BAS_2022_12_15GSD_BGRN_V10_epoch=0-step=512-v1',
            'Drop4_BAS_2022_12_15GSD_BGRN_V10_epoch=6-step=3584',
            'Drop4_BAS_2022_12_15GSD_BGRN_V10_epoch=4-step=2560',
            'Drop4_BAS_2022_12_15GSD_BGRN_V5_epoch=1-step=77702',
            'Drop4_BAS_2022_12_15GSD_BGRN_V5_epoch=5-step=233106',
        }
        flags = ~kwarray.isect_flags(table['bas_poly_eval.params.bas_pxl.package_fpath'].values, blocklist)
        params_of_interest = table['param_hashid'].loc[flags]
        agg1 = agg0_.filterto(param_hashids=params_of_interest)
        agg1.build_macro_tables(macro_groups)

        # rois = {'BR_R002', 'KR_R001', 'KR_R002', 'AE_R001', 'US_R007'}/
        # rois = {'KR_R001', 'KR_R002'}
        # rois = {'KR_R001', 'KR_R002', 'US_R007'}
        # rois = {'BR_R002', 'KR_R001', 'KR_R002', 'AE_R001'}
        # _ = agg.build_macro_tables(rois)
        # params_of_interest = ['414d0b37']
        # params_of_interest = ['ab43161b']
        # params_of_interest = ['34bed2b3']
        # params_of_interest = ['8ac5594b']

        # model_of_interest =
        # models_of_interest = [
        #     'package_epoch0_step41',
        #     'Drop4_BAS_15GSD_BGRNSH_invar_V8_epoch=16-step=8704',
        #     'Drop4_BAS_2022_12_15GSD_BGRN_V10_epoch=0-step=4305',
        #     'Drop4_BAS_2022_12_15GSD_BGRN_V5_epoch=1-step=77702-v1',
        # ]
        # subagg2 = agg.filterto(models=params_of_interest1)
        # _ = subagg2.build_macro_tables({'KR_R001', 'KR_R002'})
        # _ = subagg2.build_macro_tables(rois)
        # subagg2.macro_analysis()
        # _ = subagg2.build_macro_tables('max')
        # _ = subagg2.build_macro_tables({'KR_R001', 'KR_R002', 'US_R007'})
        # _ = subagg2.build_macro_tables({'KR_R001', 'KR_R002'})
        # _ = subagg2.build_macro_tables({'AE_R001'})
        # _ = subagg2.build_macro_tables({'BR_R002'})
        # _ = subagg2.build_macro_tables({'US_R007'})
        # _ = subagg2.build_macro_tables({'KR_R002'})
        # subagg2.macro_analysis()
        # _ = subagg2.report_best()

        # rois = {'BR_R002', 'KR_R001', 'KR_R002', 'AE_R001'}
        # macro_results = subagg.build_macro_tables(rois)
        # top_idxs = macro_results['metrics'].sort_values(subagg.primary_metric_cols).index[0:3]
        # top_param_hashids = macro_results['index']['param_hashid'].iloc[top_idxs]
        # flags = kwarray.isect_flags(subagg.index['param_hashid'].values, top_param_hashids)
        # final_agg = subagg.compress(flags)
        # macro_results = final_agg.build_macro_tables(rois)
        # # top_idx = macro_results['metrics'].sort_values(subagg.primary_metric_cols).index[0]
        # final_scores = final_agg.report_best()
        # region_id_to_summary = subagg.report_best()
        # region_id_to_summary['macro_02_19bfe3']


