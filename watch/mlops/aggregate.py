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
import scriptconfig as scfg
import kwarray
import math
import parse
import ubelt as ub
from watch.mlops import smart_pipeline
from watch.utils import util_pattern
from watch.mlops import smart_result_parser
import json
from watch.utils.util_stringalgo import shortest_unique_suffixes
from watch.utils import slugify_ext
from watch.utils import util_parallel
from typing import Dict, Any
import pandas as pd


class AggregateEvluationConfig(scfg.DataConfig):
    """
    Aggregates results from multiple DAG evaluations.
    """
    root_dpath = scfg.Value('auto', help='Where do dump all results. If "auto", uses <expt_dvc_dpath>/dag_runs')
    pipeline = scfg.Value('joint_bas_sc', help='the name of the pipeline to run')
    io_workers = scfg.Value('avail', help='number of processes to load results')
    freeze_cache = scfg.Value(False, help='set to a specific cache string to freeze a cache with the current results')


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
        >>>     'freeze_cache': 1,
        >>>     # 'pipeline': 'joint_bas_sc_nocrop',
        >>>     # 'root_dpath': expt_dvc_dpath / '_testsc',
        >>>     #'pipeline': 'sc',
        >>> }

        config = AggregateEvluationConfig.legacy(cmdline=cmdline, data=kwargs)
        eval_type_to_results = build_tables(config)
        eval_type_to_aggregator = build_aggregators(eval_type_to_results)
        agg = ub.peek(eval_type_to_aggregator.values())

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

    automated_analysis(eval_type_to_aggregator, config)


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

    plot_tables()
    plot_examples()  # TODO


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


def foldin_resolved_info(agg):
    # make these just parse nicer, but for now munge the data.
    from watch.utils.util_param_grid import DotDictDataFrame
    from watch.utils.util_param_grid import pandas_add_prefix
    param_types = DotDictDataFrame(agg.results['param_types'])

    # param_types['pxl.meta']
    fit_params = pandas_shorten_columns(param_types['fit'])
    resources = pandas_shorten_columns(param_types['pxl.resource'])
    properties = pandas_shorten_columns(param_types['pxl.properties'])
    meta = pandas_shorten_columns(param_types['pxl.meta'])
    pred_params = param_types['pxl']
    pred_params = pred_params.drop([
        c for c in pred_params.columns
        if '.meta' in c or '.resource' in c or '.properties' in c],
        axis=1)
    pred_params = pandas_shorten_columns(pred_params)

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
        'disk_resolved_params': disk_resolved_params,
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
            colvals = d2[colname]
            condensed, mapper = pandas_condense_paths(colvals)
            d2[colname] = condensed

        resolved_is_specified = is_specified.copy()
        resolved_is_specified.loc[:, always_defaulted_cols] = 0
        resolved_params = d1.copy()
        resolved_params[always_defaulted_cols] = disk_resolved_params[always_defaulted_cols]

        a = d1[common_cols]
        b = d2[common_cols]
        has_diff = ~pandas_nan_eq(a, b)

        for colname, colflags in has_diff.T.iterrows():
            resolved_params.loc[colflags, colname] = d2.loc[colflags, common_cols]

        resolved_info['resolved_params'] = resolved_params
        # ...
        # print(f'colname={colname}')
        # print(a[colname][colflags])
        # print(b[colname][colflags])

    return resolved_info


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
            scores_of_interest = pandas_shorten_columns(subagg1.metrics).loc[idx, ['bas_tp', 'bas_fp', 'bas_fn', 'bas_f1']]
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
            scores_of_interest = pandas_shorten_columns(agg1.metrics).loc[id, ['bas_tp', 'bas_fp', 'bas_fn', 'bas_f1']]
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
            scores_of_interest = pandas_shorten_columns(agg1.metrics).loc[id, ['bas_tp', 'bas_fp', 'bas_fn', 'bas_f1']]
            scores_of_interest = ub.udict(scores_of_interest.to_dict())
            text = ub.urepr(scores_of_interest.map_values(concice_si_display), nobr=1, si=1, compact=1)
            model_name = group_agg.effective_params[group_agg.model_cols[0]].loc[id]
            im = kwimage.draw_header_text(im, param_hashid + ' - ' + model_name + '\n' + text)
            kwimage.imwrite(agg_group_dpath / f'confusion_{region_id}_{param_hashid}.jpg', im)


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
            scores_of_interest = pandas_shorten_columns(comparable_agg2.metrics).loc[id, ['bas_tp', 'bas_fp', 'bas_fn', 'bas_f1']]
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

    # pman = util_progress.ProgressManager(backend='rich')
    # eval_node_prog = pman(node_eval_infos, desc='load node type')
    # for node_eval_info in eval_node_prog:
    #     node_name = node_eval_info['name']
    #     eval_node_prog.set_postfix_str(node_name)
    #     out_key = node_eval_info['out_key']
    #     result_loader_fn = node_eval_info['result_loader']
    #     if node_name not in dag.nodes:
    #         continue
    #     node = dag.nodes[node_name]
    #     out_node = node.outputs[out_key]
    #     fpaths = out_node_matching_fpaths(out_node)
    #     fpath_prog = pman(fpaths, desc=f'loading node {node_name} results')
    #     for fpath in fpath_prog:
    #         import time
    #         time.sleep(0.1)

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
            trunc_params, mappings = truncate_dataframe_items(params)
            results = {
                'mappings': mappings,
                'fpaths': pd.DataFrame(cols['fpaths'], columns=['fpath']),
                'index': pd.DataFrame(cols['index']),
                'metrics': pd.DataFrame(cols['metrics']),
                'params': pd.DataFrame(cols['params']),
                'specified_params': pd.DataFrame(cols['specified_params']),
                'trunc_params': trunc_params,
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


def truncate_dataframe_items(params):
    """
    Truncates long, typically path-like items in a data frame.
    """
    def truncate(x):
        if not isinstance(x, str):
            return x
        return slugify_ext.smart_truncate(x, max_length=16, trunc_loc=0,
                                          hash_len=4, head='', tail='')
    mappings = {}
    if len(params):
        x = params.loc[0]
        trunc_cols = [k for k, v in x.items() if isinstance(v, str) and len(v) > 16]
        trunc_params = params.copy()
        trunc_params[trunc_cols] = trunc_params[trunc_cols].applymap(truncate)
        for c in trunc_params[trunc_cols]:
            v2 = pd.Categorical(params[c])
            params[c] = v2
            v1 = v2.map(truncate)
            mapping = list(zip(v1.categories, v2.categories))
            mappings[c] = mapping
    else:
        mapping = {}
        trunc_params = params
    return trunc_params, mappings


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

        agg.primary_metric_cols = pandas_suffix_columns(  # fixme sorting
            agg.metrics, _primary_metrics_suffixes)

        agg.display_metric_cols = pandas_suffix_columns(  # fixme sorting
            agg.metrics, _display_metrics_suffixes)

        _model_suffixes = ['package_fpath']
        agg.model_cols = pandas_suffix_columns(
            agg.params, _model_suffixes)

        _testdset_suffixes = ['test_dataset']
        agg.test_dset_cols = pandas_suffix_columns(
            agg.params, _testdset_suffixes)

        agg.table = pd.concat([agg.metrics, agg.index, agg.params], axis=1)

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
        # agg.build_macro_tables()

    def macro_analysis(agg):
        from watch.utils import result_analysis

        macro_keys = list(agg.macro_key_to_regions.keys())
        if len(macro_keys) == 0:
            raise Exception('Build a macro result first')

        regions_of_interest = agg.macro_key_to_regions[agg.primary_macro_region]
        tables = agg.region_to_tables[agg.primary_macro_region]
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

            top_idxs = pandas_argmaxima(metric_group, agg.primary_metric_cols, k=top_k)

            top_metrics = metric_group.loc[top_idxs][agg.primary_metric_cols + agg.display_metric_cols]
            # top_metrics = top_metrics[agg.primary_metric_cols + agg.display_metric_cols]
            top_indexes = group['index'].loc[top_idxs]
            # top_params = group['effective_params'].loc[top_idxs].drop(agg.test_dset_cols, axis=1)
            param_lut = agg.hashid_to_params.subdict(top_indexes['param_hashid'])
            big_param_lut.update(param_lut)
            summary_table = pd.concat([top_indexes, top_metrics], axis=1)
            if shorten:
                summary_table = pandas_shorten_columns(summary_table)
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
            junk_cols = pandas_suffix_columns(effective_params, junk_suffixes)
            effective_params = effective_params.drop(junk_cols, axis=1)

        model_cols = agg.model_cols
        test_dset_cols = agg.test_dset_cols

        mappings : Dict[str, Dict[Any, str]] = {}
        path_colnames = model_cols + test_dset_cols
        for colname in path_colnames:
            colvals = params[colname]
            condensed, mapper = pandas_condense_paths(colvals)
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
        table = pd.concat([agg.index, agg.metrics, agg.effective_params], axis=1)

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

    def build_macro_tables(agg, rois=None):
        if isinstance(rois, list) and len(rois) and ub.iterable(rois[0]):
            # Asked for multiple groups of ROIS.
            for single_rois in rois:
                agg.build_single_macro_table(single_rois)
        else:
            agg.build_single_macro_table(rois)

    def gather_macro_compatable_groups(agg, regions_of_interest):
        comparable_groups = []
        macro_compatible = agg.macro_compatible
        for key in macro_compatible.keys():
            avail = (key & regions_of_interest)
            if avail == regions_of_interest:
                groups = macro_compatible[key]
                for group in groups:
                    flags = kwarray.isect_flags(group['region_id'], avail)
                    comparable_groups.append(group[flags])
        return comparable_groups

    def build_single_macro_table(agg, rois):
        # Given a specific group of regions,
        if rois is None:
            rois = 'max'

        if isinstance(rois, str):
            if rois == 'max':
                regions_of_interest = ub.argmax(agg.macro_compatible, key=len)
        else:
            regions_of_interest = rois

        comparable_groups = agg.gather_macro_compatable_groups(regions_of_interest)

        macro_key = hash_regions(regions_of_interest)

        sum_cols = [
            'bas_poly_eval.metrics.bas_tp',
            'bas_poly_eval.metrics.bas_fp',
            'bas_poly_eval.metrics.bas_fn',
            'bas_poly_eval.metrics.bas_ntrue',
            'bas_poly_eval.metrics.bas_npred',
        ]
        mean_cols = [
            'bas_poly_eval.metrics.bas_ppv',
            'bas_poly_eval.metrics.bas_tpr',
            'bas_poly_eval.metrics.bas_ffpa',
            'bas_poly_eval.metrics.bas_f1',
            'bas_poly_eval.metrics.bas_space_FAR',
            'bas_poly_eval.metrics.bas_time_FAR',
            'bas_poly_eval.metrics.bas_image_FAR',
            'bas_poly_eval.metrics.bas_faa_f1',
            'bas_poly_eval.metrics.sc_macro_f1',
            'bas_poly_eval.metrics.macro_f1_siteprep',
            'bas_poly_eval.metrics.macro_f1_active',
            'bas_poly_eval.metrics.sc_micro_f1',
        ]
        agg_id_cols = [
            'region_id',
        ] + agg.test_dset_cols

        if len(comparable_groups) > 0:
            group = comparable_groups[0]
            map_cols = [c for c in group.columns if 'metrics' in c and c.endswith(('AP', 'AUC', 'APUC'))]
            mean_cols.extend(map_cols)

        def macro_aggregate(group):
            other_cols = group.columns.difference(mean_cols).difference(sum_cols).difference(agg_id_cols)
            group[other_cols]

            for c in mean_cols:
                pass

            aggid_df = group[group.columns.intersection(agg_id_cols)]
            sum_df = group[group.columns.intersection(sum_cols)]
            mean_df = group[group.columns.intersection(mean_cols)]
            other_df = group[other_cols]

            if len(aggid_df) == 1:
                aggid_row = aggid_df.iloc[0]
            else:
                aggid_row = pd.Series({
                    k: hash_regions(v)
                    for k, v in aggid_df.T.iterrows()
                })
            aggid_row['macro_size'] = len(aggid_df)

            is_safe_cols = {
                k: ub.allsame(vs, eq=nan_eq)
                for k, vs in other_df.T.iterrows()}
            warn_cols = {k: v for k, v in is_safe_cols.items() if not v}
            assert not warn_cols

            sum_row = sum_df.sum(axis=0)
            mean_row = mean_df.mean(axis=0, numeric_only=False)
            other_row = other_df.iloc[0]

            macro_row = pd.concat([aggid_row, mean_row, sum_row, other_row], axis=0)
            return macro_row

        # Macro average comparable groups
        macro_rows = []
        macro_specified = []
        for group in comparable_groups:
            macro_row = macro_aggregate(group)
            macro_rows.append(macro_row)
            # Add in the new specified params flags
            specifed_row = agg.results['specified_params'].loc[group.index[0]]
            macro_specified.append(specifed_row)

        macro_df = pd.DataFrame(macro_rows)
        macro_specified_params = pd.DataFrame(macro_specified).reset_index(drop=True)

        # main_metric = 'bas_poly_eval.metrics.bas_faa_f1'
        # main_metric = 'bas_poly_eval.metrics.bas_tp'
        # macro_df = macro_df.sort_values(main_metric, ascending=False)
        macro_results = {
            'index': macro_df[agg.index.columns],
            'effective_params': macro_df[agg.effective_params.columns],
            'specified_params': macro_specified_params,
            'params': macro_df[agg.effective_params.columns],  # fixme, use real
            'metrics': macro_df[agg.metrics.columns],
        }
        agg.region_to_tables.pop(macro_key, None)
        agg.macro_key_to_regions.pop(macro_key, None)
        agg.macro_key_to_regions[macro_key] = regions_of_interest
        agg.region_to_tables[macro_key] = macro_results
        return macro_results

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


def plot_examples():
    pass


def plot_tables(agg):
    ...


def plot_stats_tables(agg, config):
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

    agg.build_single_macro_table({'BR_R002', 'KR_R001', 'KR_R002'})
    macro_key = agg.primary_macro_region

    agg_group_dpath = (agg_dpath / (f'stats_tables_{macro_key}' + ub.timestamp())).ensuredir()

    # df['sc_poly_eval.metrics.macro_f1_active']
    for metric in agg.primary_metric_cols:
        node_id = metric.split('.')[0]
        metric_name = metric.split('.')[-1]
        df = pd.concat([agg.metrics, agg.index, agg.params], axis=1)

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
        effective_params = tables['effective_params']
        metrics = tables['metrics']
        index = tables['index']
        table = pd.concat([index, effective_params, metrics], axis=1)
        table = table.fillna('None')

        from watch.utils import result_analysis
        results = []
        for idx, row in enumerate(table.to_dict('records')):
            row = ub.udict(row)
            row_metrics = row & set(metrics.keys())
            row_params = row & set(effective_params.keys())
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


# def node_matching_outputs(node):
#     from watch.utils import util_pattern
#     import ubelt as ub
#     # print(f'node.template_out_paths={node.template_out_paths}')
#     # out_key = eval_to_outkey[node.name]
#     found = {}
#     for out_key, out_template in node.template_out_paths.items():
#         out_template = node.template_out_paths[out_key]

#         # self._parse_pattern_attrs(self.templates[key], path)
#         pat = node.root_dpath / out_template.format(**patterns)
#         mpat = util_pattern.Pattern.coerce(pat)
#         fpaths = list(mpat.paths())
#         found[out_key] = fpaths

#     print(ub.map_vals(len, found))
#     # print('fpaths = {}'.format(ub.urepr(fpaths, nl=1)))


def out_node_matching_fpaths(out_node):
    out_template = out_node.template_value
    parser = parse.Parser(str(out_template))
    patterns = {n: '*' for n in parser.named_fields}
    pat = out_template.format(**patterns)
    mpat = util_pattern.Pattern.coerce(pat)
    fpaths = list(mpat.paths())
    return fpaths


def pandas_argmaxima(data, columns, k=1):
    """
    Finds the top K indexes for each column.

    Args:
        data : pandas data frame
        columns : columns to maximize
        k : number of top entries per column

    Returns:
        List: indexes into subset of data that are in the top k for any of the
            requested columns.

    Example:
        >>> from watch.mlops.aggregate import *  # NOQA
        >>> import numpy as np
        >>> data = pd.DataFrame({k: np.random.rand(10) for k in 'abcde'})
        >>> columns = ['b', 'd', 'e']
        >>> k = 1
        >>> top_indexes = pandas_argmaxima(data=data, columns=columns, k=k)
        >>> print(data.loc[top_indexes])
    """
    top_indexes = None
    for col in columns:
        ranked_data = data[col].sort_values(ascending=False)
        ranked_idxs = ranked_data.index
        if top_indexes is None:
            top_indexes = ranked_idxs[0:k]
        else:
            top_indexes = top_indexes.union(ranked_idxs[0:k], sort=False)
    return top_indexes


def pandas_suffix_columns(data, suffixes):
    """
    Return columns that end with this suffix
    """
    return [c for c in data.columns if any(c.endswith(s) for s in suffixes)]


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


def nan_to_None(x):
    if isinstance(x, float) and math.isnan(x):
        return None
    else:
        return x


def nan_eq(a, b):
    if isinstance(a, float) and isinstance(b, float) and math.isnan(a) and math.isnan(b):
        return True
    else:
        return a == b


def pandas_nan_eq(a, b):
    nan_flags1 = pd.isna(a)
    nan_flags2 = pd.isna(b)
    eq_flags = a == b
    both_nan = nan_flags1 & nan_flags2
    flags = eq_flags | both_nan
    return flags


def pandas_shorten_columns(summary_table):
    import ubelt as ub
    # fixme
    old_cols = summary_table.columns
    new_cols = shortest_unique_suffixes(old_cols, sep='.')
    mapping = ub.dzip(old_cols, new_cols)
    summary_table = summary_table.rename(columns=mapping)
    return summary_table


def pandas_condense_paths(colvals):
    is_valid = ~pd.isnull(colvals)
    valid_vals = colvals[is_valid]
    unique_valid_vals = valid_vals.unique()
    unique_short_vals = shortest_unique_suffixes(unique_valid_vals, sep='/')
    new_vals = [p.split('.')[0] for p in unique_short_vals]
    mapper = ub.dzip(unique_valid_vals, new_vals)
    condensed = colvals.apply(lambda x: mapper.get(x, x))
    return condensed, mapper


if __name__ == '__main__':
    """
    CommandLine:
        python ~/code/watch/watch/mlops/aggregate_evaluation.py --help
    """
    main()
