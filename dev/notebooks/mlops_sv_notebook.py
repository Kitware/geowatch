# import math
# import pandas as pd
import numpy as np
import ubelt as ub
# from watch.mlops.aggregate import hash_param
# from watch.mlops.aggregate import fix_duplicate_param_hashids
# from watch.utils import util_pandas


def _sitevisit_2023_july_report():
    import watch
    from watch.mlops.aggregate import AggregateLoader
    # import pandas  as pd
    # from watch.utils.util_pandas import DotDictDataFrame
    expt_dvc_dpath = watch.find_dvc_dpath(tags='phase2_expt', hardware='auto')

    load_kwargs = {
        'target': [
            expt_dvc_dpath / 'aggregate_results/dzyne/bas_poly_eval_2023-07-10T131639-5.csv.zip',
            expt_dvc_dpath / 'aggregate_results/dzyne/bas_poly_eval_2023-07-10T164254-5.csv.zip',
            expt_dvc_dpath / 'aggregate_results/dzyne/sv_poly_eval_2023-07-10T164254-5.csv.zip',
            # expt_dvc_dpath / 'aggregate_results/horologic/bas_poly_eval_2023-07-10T155903-5.csv.zip',
            # expt_dvc_dpath / 'aggregate_results/horologic/sv_poly_eval_2023-07-10T155903-5.csv.zip',
            # expt_dvc_dpath / 'aggregate_results/namek/bas_poly_eval_2023-04-19T113433-5.csv.zip',
            # expt_dvc_dpath / 'aggregate_results/namek/bas_poly_eval_2023-07-10T161857-5.csv.zip',
            # expt_dvc_dpath / 'aggregate_results/namek/bas_pxl_eval_2023-04-19T113433-5.csv.zip',
            # expt_dvc_dpath / 'aggregate_results/namek/bas_pxl_eval_2023-07-10T161857-5.csv.zip',
            # expt_dvc_dpath / 'aggregate_results/namek/sv_poly_eval_2023-07-10T161857-5.csv.zip',
            # expt_dvc_dpath / 'aggregate_results/toothbrush/bas_poly_eval_2023-04-19T105718-5.csv.zip',
            # expt_dvc_dpath / 'aggregate_results/toothbrush/bas_poly_eval_2023-07-10T150132-5.csv.zip',
            # expt_dvc_dpath / 'aggregate_results/toothbrush/bas_pxl_eval_2023-04-19T105718-5.csv.zip',
            # expt_dvc_dpath / 'aggregate_results/toothbrush/bas_pxl_eval_2023-07-10T150132-5.csv.zip',
            # expt_dvc_dpath / 'aggregate_results/toothbrush/sv_poly_eval_2023-04-19T105718-5.csv.zip',
            # expt_dvc_dpath / 'aggregate_results/toothbrush/sv_poly_eval_2023-07-10T150132-5.csv.zip',
            # expt_dvc_dpath / 'aggregate_results/uconn/COLD_candidates_0705.zip',
            # expt_dvc_dpath / 'aggregate_results/wu/bas_pxl_eval_2023-07-11T180910+0.csv.zip',
            # expt_dvc_dpath / 'aggregate_results/wu/bas_poly_eval_2023-07-11T180910+0.csv.zip',
            # expt_dvc_dpath / 'aggregate_results/wu/bas_pxl_eval_2023-07-11T181515+0.csv.zip',
            # expt_dvc_dpath / 'aggregate_results/wu/bas_poly_eval_2023-07-11T181515+0.csv.zip',
            # expt_dvc_dpath / 'aggregate_results/wu/bas_pxl_eval_2023-07-11T213433+0.csv.zip',
            # expt_dvc_dpath / 'aggregate_results/wu/bas_poly_eval_2023-07-11T213433+0.csv.zip',
            # expt_dvc_dpath / 'aggregate_results/connor/bas_poly_eval_2023-07-11T134348-5.csv.zip',
            # expt_dvc_dpath / 'aggregate_results/connor/bas_pxl_eval_2023-07-11T134348-5.csv.zip',
        ],
        'pipeline': 'bas_building_and_depth_vali',
        'io_workers': 'avail',
    }
    with ub.Timer('load'):
        loader = AggregateLoader(**load_kwargs)
        eval_type_to_agg = loader.coerce_aggregators()

    # agg0 = eval_type_to_agg['sv_poly_eval']
    # pxl_agg = eval_type_to_agg['bas_pxl_eval']
    poly_agg = eval_type_to_agg['bas_poly_eval']
    sv_poly_agg = eval_type_to_agg['sv_poly_eval']

    # from watch.mlops.smart_global_helper import SMART_HELPER
    # SMART_HELPER.populate_test_dataset_bundles(agg0)

    rois = ['CH_R001', 'KR_R001', 'KR_R002', 'NZ_R001']
    poly_agg.build_macro_tables(rois)
    _ = poly_agg.report_best()

    # print(ub.urepr(ub.udict(sv_poly_agg.macro_compatible).map_values(len)))
    # print(ub.urepr(ub.udict(poly_agg.macro_compatible).map_values(len)))

    agg = sv_poly_agg
    flags = agg.table['params.sv_crop.crop_src_fpath'].isnull()
    subagg = agg.compress(~flags)
    subagg.build_macro_tables(rois)
    _ = subagg.report_best()

    macro_key = list(subagg.region_to_tables.keys())[-1]
    macro = subagg.region_to_tables[macro_key]

    macro = macro.sort_values('metrics.sv_poly_eval.bas_f1')
    points = macro[['region_id', 'param_hashid',
                    'metrics.bas_poly_eval.bas_f1',
                    'metrics.sv_poly_eval.bas_f1',
                    'metrics.bas_poly_eval.bas_tpr',
                    'metrics.sv_poly_eval.bas_tpr']]

    import kwplot
    import matplotlib as mpl
    kwplot.autosns()
    kwplot.plt.ion()
    kwplot.figure()
    ax = kwplot.plt.gca()
    ax.cla()

    segments = []
    # min_x = float('inf')
    # max_x = -float('inf')
    for row in macro.to_dict('records'):
        x1 = row['metrics.bas_poly_eval.bas_tpr']
        x2 = row['metrics.sv_poly_eval.bas_tpr']

        # y1 = row['metrics.bas_poly_eval.bas_fp']
        # y2 = row['metrics.sv_poly_eval.bas_fp']
        y1 = row['metrics.bas_poly_eval.bas_f1']
        y2 = row['metrics.sv_poly_eval.bas_f1']
        segments.append([(x1, y1), (x2, y2)])
    macro['metrics.bas_poly_eval.bas_f1']

    pts1 = [s[0] for s in segments]
    pts2 = [s[1] for s in segments]
    data_lines = mpl.collections.LineCollection(segments, color='blue', alpha=0.5, linewidths=1)
    ax.add_collection(data_lines)
    ax.plot(*zip(*pts1), 'rx', label='before SV')
    ax.plot(*zip(*pts2), 'bo', label='after SV')
    ax.legend()
    ax.set_xlabel('bas_tpr')
    ax.set_ylabel('bas_f1')
    # ax.set_ylabel('bas_fp')
    ax.set_title(f'Effect of SV: {rois}')

    # kwplot.sns.scatterplot(data=macro, y='metrics.bas_poly_eval.bas_f1', x='metrics.bas_poly_eval.bas_tpr', markers='x', ax=ax, hue='resolved_params.bas_poly.thresh')

    # kwplot.sns.scatterplot(data=macro, y='metrics.sv_poly_eval.bas_f1', x='metrics.sv_poly_eval.bas_tpr', markers='o', hue='resolved_params.sv_dino_filter.end_min_score', ax=ax)
    # kwplot.sns.scatterplot(data=macro, y='metrics.sv_poly_eval.bas_f1', x='metrics.sv_poly_eval.bas_tpr', markers='o', hue='resolved_params.bas_poly.thresh', ax=ax)

    if 0:
        # Point A
        target_pt = (0.6227, 0.57730)
        delta = points[['metrics.bas_poly_eval.bas_tpr', 'metrics.bas_poly_eval.bas_f1']].values - target_pt
        dist = np.linalg.norm(delta, axis=1)
        point_A = points.iloc[np.argmin(dist)]
        ax.plot(point_A['metrics.bas_poly_eval.bas_tpr'], point_A['metrics.bas_poly_eval.bas_f1'], '*', markersize=20, color='orange')
        ax.plot(point_A['metrics.sv_poly_eval.bas_tpr'], point_A['metrics.sv_poly_eval.bas_f1'], '*', markersize=20, color='orange')
        ax.text(*target_pt, 'A', fontdict={'weight': 'bold'})

        # Pt C
        target_pt = (0.656, 0.60)
        delta = points[['metrics.sv_poly_eval.bas_tpr', 'metrics.sv_poly_eval.bas_f1']].values - target_pt
        dist = np.linalg.norm(delta, axis=1)
        point_C = points.iloc[np.argmin(dist)]
        ax.plot(point_C['metrics.bas_poly_eval.bas_tpr'], point_C['metrics.bas_poly_eval.bas_f1'], '*', markersize=20, color='orange')
        ax.plot(point_C['metrics.sv_poly_eval.bas_tpr'], point_C['metrics.sv_poly_eval.bas_f1'], '*', markersize=20, color='orange')
        ax.text(*target_pt, 'C', fontdict={'weight': 'bold'})

        # Point B
        target_pt = (0.7952, 0.5581)
        delta = points[['metrics.sv_poly_eval.bas_tpr', 'metrics.sv_poly_eval.bas_f1']].values - target_pt
        dist = np.linalg.norm(delta, axis=1)
        point_B = points.iloc[np.argmin(dist)]
        ax.plot(point_B['metrics.bas_poly_eval.bas_tpr'], point_B['metrics.bas_poly_eval.bas_f1'], '*', markersize=20, color='orange')
        ax.plot(point_B['metrics.sv_poly_eval.bas_tpr'], point_B['metrics.sv_poly_eval.bas_f1'], '*', markersize=20, color='orange')
        ax.text(*target_pt, 'B', fontdict={'weight': 'bold'})

        special_params = {}
        special_params['point_A'] = agg.hashid_to_params[point_A['param_hashid']]
        special_params['point_B'] = agg.hashid_to_params[point_B['param_hashid']]
        special_params['point_C'] = agg.hashid_to_params[point_C['param_hashid']]
        print('special_params = {}'.format(ub.urepr(special_params, nl=2)))

        if 0:
            chosen_param_hashid = point_C['param_hashid']
            chosen_agg = agg.filterto(param_hashids=[chosen_param_hashid])
            _ = chosen_agg.report_best()
            print(ub.repr2(chosen_agg.table['fpath'].tolist()))
            out_dpath = ub.Path('./chosen')
            eval_links = (out_dpath / 'eval_links').ensuredir()
            for _, row in chosen_agg.table.iterrows():
                node_dpath = ub.Path(row['fpath']).parent
                link_fpath = eval_links / (row['region_id'] + '_' + node_dpath.name)
                ub.symlink(node_dpath, link_fpath)

    if 0:
        # Find high TPR example:
        idx = subagg.table['metrics.bas_poly_eval.bas_tpr'].idxmax()
        high_tpr_param_hashid = subagg.table.loc[idx]['param_hashid']
        high_tpr_agg = agg.filterto(param_hashids=[high_tpr_param_hashid])
        _ = high_tpr_agg.report_best()
        print(ub.repr2(high_tpr_agg.table['fpath'].tolist()))

        out_dpath = ub.Path('./high-tpr')
        eval_links = (out_dpath / 'eval_links').ensuredir()
        for _, row in high_tpr_agg.table.iterrows():
            node_dpath = ub.Path(row['fpath']).parent
            link_fpath = eval_links / (row['region_id'] + '_' + node_dpath.name)
            ub.symlink(node_dpath, link_fpath)
            ...

    if 0:
        # Inspect one of the best ones.
        ###
        out_dpath = ub.Path('./chosen')
        root_dpath = expt_dvc_dpath / '_toothbrush_split6_landcover_MeanYear10GSD-V2/'
        out_dpath = root_dpath / '_custom'
        max_id = macro['metrics.sv_poly_eval.bas_f1'].idxmax()
        chosen_param_hashid = macro.loc[max_id]['param_hashid']

        chosen_agg = agg.filterto(param_hashids=[chosen_param_hashid])
        _ = chosen_agg.report_best()
        print(ub.repr2(chosen_agg.table['fpath'].tolist()))
        eval_links = (out_dpath / 'eval_links').ensuredir()
        for _, row in chosen_agg.table.iterrows():
            node_dpath = ub.Path(row['fpath']).parent
            link_fpath = eval_links / (row['region_id'] + '_' + node_dpath.name)
            ub.symlink(node_dpath, link_fpath)

    # agg.table['resolved_params.sv_crop.src'].unique()
    # bad_values['resolved_params.sv_crop.src'].unique()
    # good_values['resolved_params.sv_crop.src'].unique()
    # bad_values = agg.table[flags]
    # good_values = agg.table[~flags]
