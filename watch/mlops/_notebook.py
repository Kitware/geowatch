# flake8: noqa
import math
import pandas as pd
import ubelt as ub
from watch.mlops.aggregate import hash_param
from watch.mlops.aggregate import fix_duplicate_param_hashids
from watch.utils import util_pandas

def _gather_namek_shortlist_results():
    """

    smartwatch model_stats models/fusion/Drop4-BAS/packages/Drop4_BAS_2022_12_15GSD_BGRNSH_BGR_V4/Drop4_BAS_2022_12_15GSD_BGRNSH_BGR_V4_v0_epoch44_step46014.pt

    """
    from watch.mlops.aggregate import AggregateEvluationConfig
    from watch.mlops.aggregate import coerce_aggregators
    import watch
    expt_dvc_dpath = watch.find_dvc_dpath(tags='phase2_expt', hardware='auto')
    cmdline = 0
    kwargs = {
        'target': expt_dvc_dpath / '_namek_split1_eval_small',
        # 'target': expt_dvc_dpath / '_namek_split2_eval_small',
        'pipeline': 'bas',
        'io_workers': 20,
    }
    config = AggregateEvluationConfig.cli(cmdline=cmdline, data=kwargs)
    eval_type_to_aggregator = coerce_aggregators(config)
    agg = eval_type_to_aggregator['bas_pxl_eval']

    region_id_to_summary, top_param_lut = agg.report_best(1000, verbose=0)

    tocombine_indexes = []
    for region, summary in region_id_to_summary.items():
        tocombine_indexes.append(list(summary.index))
    import itertools as it
    top_indexes = list(ub.oset([x for x in ub.flatten(
        it.zip_longest(*tocombine_indexes)) if x is not None]))

    table = agg.table.copy()
    table.loc[top_indexes, 'rank'] = list(range(len(top_indexes)))
    table = table.sort_values('rank')

    chosen_indexes = []
    for expt, group in table.groupby('resolved_params.bas_pxl_fit.name'):
        group['params.bas_pxl.package_fpath'].tolist()
        group = group.sort_values('rank')
        chosen_indexes.extend(group.index[0:2])
    chosen_indexes = table.loc[chosen_indexes, 'rank'].sort_values().index

    all_models_fpath = ub.Path('$HOME/code/watch/dev/reports/split1_all_models.yaml').expand()
    from watch.utils.util_yaml import Yaml
    known_models = Yaml.coerce(all_models_fpath)

    top_k = 40
    chosen_indexes = chosen_indexes[:top_k]

    chosen_table = table.loc[chosen_indexes]

    # Need to remove invariants for now
    flags = ~np.array(['invariants' in chan for chan in chosen_table['resolved_params.bas_pxl_fit.channels']])
    chosen_table = chosen_table[flags]

    chosen_models = chosen_table['params.bas_pxl.package_fpath'].tolist()
    set(known_models).issuperset(set(chosen_models))

    new_models_fpath = ub.Path('$HOME/code/watch/dev/reports/split1_models_filter1.yaml').expand()
    new_models_fpath.write_text(Yaml.dumps(chosen_models))


    subagg = agg.filterto(index=chosen_indexes)
    subagg.table['resolved_params.bas_pxl_fit.name']

    # top_table = agg.table.loc[top_indexes]

    # top_models = top_table['resolved_params.bas_pxl_fit.name'].unique()

    # from watch.utils.util_pandas import DotDictDataFrame
    # resolved = DotDictDataFrame(agg.resolved_params)
    # fit_params = resolved.subframe('resolved_params.bas_pxl_fit')
    # top_fit_params = fit_params.loc[top_indexes]

    num_top_models = len(top_models)
    num_total_models = len(agg.params['params.bas_pxl.package_fpath'].unique())
    print(f'num_top_models={num_top_models} / {num_total_models}')

    is_highres = [float(a.split('G')[0]) < 8 for a in agg.table['resolved_params.bas_pxl_fit.window_space_scale']]
    has_wv = ['WV' in a for a in agg.table['resolved_params.bas_pxl_fit.channels']]
    flags = (np.array(has_wv) & np.array(is_highres))
    subagg = agg.compress(flags)


def _namek_check_pipeline_status():
    from watch.mlops import aggregate_loader
    import watch
    expt_dvc_dpath = watch.find_dvc_dpath(tags='phase2_expt', hardware='auto')
    root_dpath = expt_dvc_dpath / '_namek_eval'
    pipeline = 'bas'
    io_workers = 16
    # eval_type_to_results = aggregate_loader.build_tables(root_dpath, pipeline, io_workers)
    # eval_type_to_results['bas_pxl_eval']

    from watch.mlops import smart_pipeline
    dag = smart_pipeline.make_smart_pipeline(pipeline)
    dag.configure(config=None, root_dpath=root_dpath)
    # node_to_fpaths = {}
    # for node_name, node in ub.ProgIter(dag.nodes.items()):
    #     node_fpaths = {}
    #     for out_node_key, out_node in node.outputs.items():
    #         node_fpaths[out_node_key] = aggregate_loader.out_node_matching_fpaths(out_node)
    #     node_to_fpaths[node_name] = node_fpaths
    # node_to_fpaths = ub.udict(node_to_fpaths).map_values(ub.udict)
    # num_existing_outs = node_to_fpaths.map_values(lambda x: x.map_values(len))

    # stages_of_interest = ['bas_pxl', 'bas_pxl_eval', 'bas_poly', 'bas_poly_eval']
    stages_of_interest = ['bas_pxl', 'bas_pxl_eval']
    existing = {}
    for stage in stages_of_interest:
        rows = dag.nodes[stage].find_template_outputs(workers=8)
        if 1:
            rows = [row for row in rows if row.get('request.bas_pxl.test_dataset', None) is not None]
        existing[stage] = rows

    bad_rows = []
    for stage in stages_of_interest:
        from watch.utils import util_dotdict  # NOQA
        for row in ub.ProgIter(existing[stage]):
            node = dag.nodes[stage]
            row = util_dotdict.DotDict(row)
            config = row.prefix_get('request')
            dag.configure(config, cache=False)
            row_dpath = ub.Path(row['dpath'])
            row['process_id'] = node.process_id
            # print(node.final_command())
            if node.final_node_dpath != row['dpath']:
                # print('config = {}'.format(ub.urepr(config, nl=1)))
                # print((row_dpath / 'job_config.json').read_text())
                # print((row_dpath / 'invoke.sh').read_text())
                # print(node.final_command())
                bad_rows.append(row)

    dfs = {}
    from watch.utils.util_pandas import DotDictDataFrame
    for stage in stages_of_interest:
        dfs[stage] = DotDictDataFrame(existing[stage])

    stage1 = 'bas_pxl'
    stage2 = 'bas_pxl_eval'
    df1 = dfs[stage1]
    df2 = dfs[stage2]

    from watch.utils import util_dotdict  # NOQA
    bad_rows = []
    for row in existing[stage1]:
        wanted_succ = ub.Path(row['dpath']) / '.succ' / stage2
        if not wanted_succ.exists():
            node = dag.nodes[stage]
            row = util_dotdict.DotDict(row)
            config = row.prefix_get('request')
            dag.configure(config, cache=False)
            if node.does_exist:
                raise Exception

        node1 = dag.nodes[stage1]
        node2 = dag.nodes[stage2]
        ...


    df1 = dfs['bas_pxl'].subframe('request')
    df2 = dfs['bas_pxl_eval'].subframe('request')

    df1 = df1[~df1['bas_pxl.package_fpath'].isnull()]
    df2 = df2[~df2['bas_pxl.package_fpath'].isnull()]

    from kwcoco._helpers import _delitems

    hashids1 = [ub.hash_data(row) for row in df1.to_dict('records')]
    hashids2 = [ub.hash_data(row) for row in df2.to_dict('records')]

    # There shouldn't be duplicates here, except in pathological circumstances
    to_drop1 = []
    for _, idxs in ub.find_duplicates(hashids1).items():
        print('Dropping')
        print(df1.iloc[idxs])
        to_drop1.extend(idxs)
    to_drop2 = []
    for _, idxs in ub.find_duplicates(hashids2).items():
        print('Dropping')
        print(df2.iloc[idxs])
        to_drop2.extend(idxs)
    df1 = df1.drop(df1.index[to_drop1], axis=0)
    df2 = df2.drop(df2.index[to_drop2], axis=0)
    _delitems(hashids1, to_drop1)
    _delitems(hashids2, to_drop2)

    assert not ub.find_duplicates(hashids1)
    assert not ub.find_duplicates(hashids2)

    hashids1 = ub.oset(hashids1)
    hashids2 = ub.oset(hashids2)

    df1['hashid'] = hashids1
    df2['hashid'] = hashids2
    common = hashids1 & hashids2

    df1 = df1.set_index('hashid')
    df2 = df2.set_index('hashid')

    missing1 = hashids2 - hashids1
    assert len(missing1) == 0, 'should not be possible without error'

    missing2 = hashids1 - hashids2
    # These have not had a computation done for them.
    missing_df = df1.loc[missing2]


    # WANTED: Algorithm that takes a list of grid points and determines if any
    # of the rows can be expressed conciesly as a matrix.
    """
    given = [
        {'x': 1, 'y': 1, 'z': 1},
        {'x': 1, 'y': 2, 'z': 1},
        {'x': 1, 'y': 3, 'z': 1},
        {'x': 2, 'y': 1, 'z': 2},
        {'x': 2, 'y': 2, 'z': 2},
        {'x': 2, 'y': 3, 'z': 2},
        {'x': 3, 'y': 1, 'z': 2},
        {'x': 4, 'y': 3, 'z': 2},
        {'x': 4, 'y': 4, 'z': 2},
        {'x': 4, 'y': 3, 'z': 3},
        {'x': 4, 'y': 4, 'z': 3},
    ]

    should output:
        - x: [1, 2]
          y: [1, 2, 3]
          z: [1, 2]
        - x: [3]
          y: [1]
          z: [2]
        - x: [4]
          y: [3, 4]
          z: [2, 3]

    It seems like a solution here is non-unique, or at least the greedy way of
    finding the solution will result in different answers depending on which
    variable you try to collapse first. Is this a well-studied algorithm? Seems
    like a decomposition problem.
    """

    import sys, ubelt
    from watch.utils.result_analysis import varied_value_counts

    submatrices = []
    for _, group in missing_df.groupby('bas_pxl.test_dataset'):
        group_records = group.to_dict('records')
        group_varied = varied_value_counts(group_records)
        for row in group_records:
            row = {k: v for k, v in row.items() if not isinstance(v, float) or not math.isnan(v)}
            row['bas_pxl.enabled'] = False
            row['bas_pxl_eval.enabled'] = True
            row['bas_poly.enabled'] = True
            row['bas_poly_eval.enabled'] = True
            submatrices.append(row)

    from watch.utils import util_yaml
    submat_text = util_yaml.yaml_dumps({'submatrices': submatrices})
    fpath = ub.Path('foo.yaml').absolute()
    fpath.write_text(submat_text)

    # Generate code to run the missing experiments
    invocation = ub.codeblock(
        fr'''
        python -m watch.mlops.schedule_evaluation \
            --params={fpath} \
            --root_dpath="{root_dpath}" \
            --devices="0,1" --queue_size=4 \
            --backend=tmux --queue_name "_explicit_recompute" \
            --pipeline=bas --skip_existing=0 \
            --print_varied=1 \
            --run=0
        ''')
    print(invocation)



    # For two levels in the node figure out:
    # What paths on the parent are are in common.
    # What paths on the child have yet to be computed.
    node_fpaths = node_to_fpaths['bas_pxl']


def _gather_all_results():
    r"""
    # On Namek
    DVC_EXPT_DPATH=$(smartwatch_dvc --tags='phase2_expt' --hardware=auto)
    python -m watch.mlops.aggregate \
        --pipeline=bas \
        --target "
            - $DVC_EXPT_DPATH/_timekernel_test_drop4
            - $DVC_EXPT_DPATH/_namek_eval
        " \
        --export_tables=True \
        --output_dpath="$DVC_EXPT_DPATH/namek_agg"

    # On Toothbrush
    DVC_EXPT_DPATH=$(smartwatch_dvc --tags='phase2_expt' --hardware=auto)
    rsync -avprPR namek:data/dvc-repos/smart_expt_dvc/./namek_agg "$DVC_EXPT_DPATH"

    python -m watch.mlops.aggregate \
        --pipeline=bas \
        --target "
            - namek_agg/*.csv.zip
            - $DVC_EXPT_DPATH/_timekernel_test_drop4
            - $DVC_EXPT_DPATH/_testpipe
            - $DVC_EXPT_DPATH/_evaluations
            - $DVC_EXPT_DPATH/_testpipe2
        " \
        --export_tables=True \
        --output_dpath="$DVC_EXPT_DPATH/all_agg_2022-02-24"

    python -m watch.mlops.aggregate \
        --pipeline=bas \
        --target "
            - all_agg_2022-02-24/*.csv.zip
        " \
        --stdout_report=True \
        --output_dpath="$DVC_EXPT_DPATH/all_agg_2022-02-24/reports"
    """
    from watch.mlops.aggregate import AggregateEvluationConfig
    from watch.mlops.aggregate import coerce_aggregators
    import watch
    expt_dvc_dpath = watch.find_dvc_dpath(tags='phase2_expt', hardware='auto')
    cmdline = 0
    kwargs = {
        # 'target': expt_dvc_dpath / 'all_agg_2022-02-24/*.csv.zip',
        'target': expt_dvc_dpath / '_split6_toothbrush_meanyear',
        'pipeline': 'bas',
        'io_workers': 20,
    }
    config = AggregateEvluationConfig.cli(cmdline=cmdline, data=kwargs)
    eval_type_to_aggregator = coerce_aggregators(config)

    agg = eval_type_to_aggregator.get('bas_pxl_eval', None)
    agg = eval_type_to_aggregator.get('bas_poly_eval', None)

    from watch.mlops.aggregate import coerce_aggregators
    build_all_param_plots(agg, rois, config)


def resource_usage_report(agg):
    import pandas as pd
    from watch.utils import util_time
    table = agg.table.copy()
    resources = agg.resources

    duration_cols = [
        k for k in agg.resources.keys()
        if k.endswith('.duration')
    ]
    for k in duration_cols:
        table.loc[:, k] = table.loc[:, k].apply(lambda x: util_time.coerce_timedelta(x) if not pd.isnull(x) else x)

    resource_summary = []
    for k in duration_cols:
        a, b, c = k.split('.')
        uuid_key = f'context.{b}.uuid'

        chosen = []
        for _, group in table.groupby(uuid_key):
            idx = group[k].idxmax()
            chosen.append(idx)

        unique_rows = table.loc[chosen]
        row = {
            'node': b,
            'resource': c,
            'total': unique_rows[k].sum(),
            'mean': unique_rows[k].mean(),
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
    pd.DataFrame(resource_summary)

    agg.resources['resources.bas_poly_eval.duration']
    agg.resources['resources.bas_poly.duration']
    agg.resources['resources.bas_poly.co2_kg']

    unique_resources = {}
    unique_resources['bas_pxl'] = poly_agg.table.groupby('context.bas_pxl.uuid')
    unique_resources['bas_poly'] = poly_agg.table.groupby('context.bas_poly.uuid')
    unique_resources['bas_poly_eval'] = poly_agg.table.groupby('context.bas_poly_eval.uuid')

    for k, v in unique_resources.items():
        v[f'resources.{k}.duration'].max().apply(util_time.coerce_timedelta).sum()
        co2_key = f'resources.{k}.co2_kg'
        if co2_key in v:
            ...
        # v[].max().sum()

    pxl_time = poly_agg.table.groupby('context.bas_pxl.uuid')['resources.bas_pxl.duration'].max().apply(util_time.coerce_timedelta).sum()
    poly_time = poly_agg.table.groupby('context.bas_poly.uuid')['resources.bas_poly.duration'].max().apply(util_time.coerce_timedelta).sum()
    poly_eval_time = poly_agg.table.groupby('context.bas_poly_eval.uuid')['resources.bas_poly.duration'].max().apply(util_time.coerce_timedelta).sum()
    total_time = pxl_time + poly_time + poly_eval_time

    poly_agg.table['context.bas_poly.uuid']
    poly_agg.table['context.bas_poly_eval.uuid']


def _check_high_tpr_case(agg, config):
    macro_results = agg.region_to_tables[agg.primary_macro_region].copy()

    from watch.utils.util_pandas import DotDictDataFrame
    macro_metrics = DotDictDataFrame(macro_results['metrics'])
    tpr_col = macro_metrics.find_column('bas_tpr')
    macro_metrics = macro_metrics.sort_values(tpr_col, ascending=False)
    inspect_idxs = macro_metrics.index[0:1]

    agg_dpath = ub.Path(config['root_dpath'] / 'aggregate')
    agg_group_dpath = (agg_dpath / ('top_tpr_cases' + ub.timestamp())).ensuredir()

    for rank, idx in enumerate(inspect_idxs):
        param_hashid = macro_results['index'].loc[idx]['param_hashid']

        subagg = agg.filterto(param_hashids=[param_hashid])
        subagg.build_macro_tables(rois)
        subagg.report_best()

        dpath = (agg_group_dpath / f'top_{rank:03d}_tpr_case').ensuredir()

        for loc in subagg.fpaths.index:
            index_row = subagg.index.loc[loc]
            eval_fpath = subagg.fpaths.loc[loc]
            eval_dpath = eval_fpath.parent
            link_dpath = dpath / f'eval_link_{index_row.region_id}_{index_row.param_hashid}'
            ub.symlink(real_path=eval_dpath, link_path=link_dpath)
            ...

        agg.index['param_hashid'] == param_hashid

        subagg.fpaths.tolist()
        from watch.mlops.aggregate import make_summary_analysis
        agg1 = subagg
        make_summary_analysis(agg1, config, dpath)
    ...


def _namek_eval():
    from watch.mlops.aggregate import AggregateEvluationConfig
    from watch.mlops.aggregate import coerce_aggregators
    import watch
    data_dvc_dpath = watch.find_dvc_dpath(tags='phase2_data', hardware='auto')
    expt_dvc_dpath = watch.find_dvc_dpath(tags='phase2_expt', hardware='auto')
    cmdline = 0
    kwargs = {
        'target': expt_dvc_dpath / '_namek_eval',
        'pipeline': 'bas',
        'io_workers': 0,
    }
    config = AggregateEvluationConfig.cli(cmdline=cmdline, data=kwargs)
    eval_type_to_aggregator = coerce_aggregators(config)

    poly_agg = eval_type_to_aggregator.get('bas_poly_eval', None)
    pxl_agg = eval_type_to_aggregator.get('bas_pxl_eval', None)

    if 1:
        # from watch.utils import util_dotdict  # NOQA
        # Join the pixel and polygon results
        from watch.utils import util_pandas
        a = util_pandas.DotDictDataFrame(poly_agg.params).subframe('params.bas_pxl')
        b = util_pandas.DotDictDataFrame(pxl_agg.params).subframe('params.bas_pxl')

        a_hashids = [ub.hash_data(row) for row in a.to_dict('records')]
        b_hashids = [ub.hash_data(row) for row in b.to_dict('records')]
        a['hashid.bas_pxl'] = a_hashids
        b['hashid.bas_pxl'] = b_hashids

        hashid_to_idxs1 =ub.find_duplicates(a_hashids, k=0)
        hashid_to_idxs2 =ub.find_duplicates(b_hashids, k=0)

        missing1 = set(hashid_to_idxs1) - set(hashid_to_idxs2)
        missing2 = set(hashid_to_idxs2) - set(hashid_to_idxs1)
        common = set(hashid_to_idxs2) & set(hashid_to_idxs1)

        pxl_eval_cols = [c for c in pxl_agg.table.columns if 'bas_pxl_eval' in c.split('.')]
        sub_table = pxl_agg.table[pxl_eval_cols].copy()
        sub_table.loc[:, 'hashid.bas_pxl'] = b_hashids

        main_table = poly_agg.table.copy()
        main_table.loc[:, 'hashid.bas_pxl'] = a_hashids
        joined_table = pd.merge(main_table, sub_table, on='hashid.bas_pxl')

        joined_table[['metrics.bas_poly_eval.bas_f1', 'metrics.bas_pxl_eval.salient_AP']]

        joined_table[['resolved_params.bas_poly.agg_fn']]

        import kwplot
        if 0:
            sns = kwplot.autosns()
            plt = kwplot.autoplt()
            # sns.scatterplot(
            #     data=joined_table,
            #     x='metrics.bas_poly_eval.bas_f1',
            #     y='metrics.bas_pxl_eval.salient_AP',
            #     hue='params.bas_pxl.package_fpath',
            #     legend=False,
            # )
            sns.scatterplot(
                data=joined_table,
                x='metrics.bas_poly_eval.bas_f1',
                y='metrics.bas_pxl_eval.salient_AP',
                hue='resolved_params.bas_poly.agg_fn',
                legend=False,
            )

        # ['params']).find_column('saliency_loss')
        # row = util_dotdict.DotDict(row)
        ...

    poly_agg.resources['resources.bas_poly_eval.duration']
    poly_agg.resources['resources.bas_poly.duration']

    from watch.utils import util_time
    unique_resources = {}
    unique_resources['bas_pxl'] = poly_agg.table.groupby('context.bas_pxl.uuid')
    unique_resources['bas_poly'] = poly_agg.table.groupby('context.bas_poly.uuid')
    unique_resources['bas_poly_eval'] = poly_agg.table.groupby('context.bas_poly_eval.uuid')

    for k, v in unique_resources.items():
        v[f'resources.{k}.duration'].max().apply(util_time.coerce_timedelta).sum()
        co2_key = f'resources.{k}.co2_kg'
        if co2_key in v:
            ...
        # v[].max().sum()

    pxl_time = poly_agg.table.groupby('context.bas_pxl.uuid')['resources.bas_pxl.duration'].max().apply(util_time.coerce_timedelta).sum()
    poly_time = poly_agg.table.groupby('context.bas_poly.uuid')['resources.bas_poly.duration'].max().apply(util_time.coerce_timedelta).sum()
    poly_eval_time = poly_agg.table.groupby('context.bas_poly_eval.uuid')['resources.bas_poly.duration'].max().apply(util_time.coerce_timedelta).sum()
    total_time = pxl_time + poly_time + poly_eval_time

    poly_agg.table['context.bas_poly.uuid']
    poly_agg.table['context.bas_poly_eval.uuid']


    agg = ub.peek(eval_type_to_aggregator.values())
    agg.build_macro_tables()

    agg.primary_display_cols = ['bas_poly_eval.metrics.bas_faa_f1', 'bas_poly_eval.metrics.bas_f1', 'bas_poly_eval.metrics.bas_tpr', 'bas_poly_eval.metrics.bas_ppv']

    agg.primary_metric_cols = ['bas_poly_eval.metrics.bas_tpr', 'bas_poly_eval.metrics.bas_ppv']
    _ = agg.report_best()

    agg.primary_metric_cols = ['bas_poly_eval.metrics.bas_f1']
    _ = agg.report_best()

    # agg.primary_metric_cols = ['bas_poly_eval.metrics.bas_ppv', 'bas_poly_eval.metrics.bas_tpr']
    # _ = agg.report_best()

    from watch.mlops.aggregate import build_all_param_plots
    # rois = {'KR_R001', 'KR_R002'}
    rois = {'KR_R001'}
    build_all_param_plots(agg, rois, config)

    # Find best dicefocal model
    from watch.utils import util_pandas
    col = util_pandas.DotDictDataFrame(agg.fit_params).find_column('saliency_loss')
    flags = agg.fit_params[col] == 'dicefocal'
    subagg1 = agg.compress(flags)
    agg.primary_metric_cols = ['bas_poly_eval.metrics.bas_ppv', 'bas_poly_eval.metrics.bas_tpr']
    _ = subagg1.report_best()


def _timekernel_analysis():
    from watch.mlops.aggregate import AggregateEvluationConfig
    from watch.mlops.aggregate import build_tables
    from watch.mlops.aggregate import build_aggregators
    import watch
    data_dvc_dpath = watch.find_dvc_dpath(tags='phase2_data', hardware='auto')
    expt_dvc_dpath = watch.find_dvc_dpath(tags='phase2_expt', hardware='auto')
    cmdline = 0
    kwargs = {
        'root_dpath': expt_dvc_dpath / '_timekernel_test_drop4',
        'pipeline': 'bas',
        'io_workers': 10,
        'freeze_cache': 0,
    }
    config = AggregateEvluationConfig.cli(cmdline=cmdline, data=kwargs)
    eval_type_to_results = build_tables(config)
    agg_dpath = ub.Path(config['root_dpath']) / 'aggregate'
    eval_type_to_aggregator = build_aggregators(eval_type_to_results, agg_dpath)
    agg = ub.peek(eval_type_to_aggregator.values())
    agg = eval_type_to_aggregator.get('bas_poly_eval', None)

    from watch.mlops.aggregate import build_all_param_plots
    rois = {'KR_R001', 'KR_R002'}
    rois = {'KR_R001'}
    _ = agg.report_best()

    build_all_param_plots(agg, rois, config)

    build_all_param_plots(agg, rois, config)
    from watch.mlops.aggregate import build_all_param_plots
    rois = {'KR_R001', 'KR_R002'}
    # rois = {'KR_R001'}
    build_all_param_plots(agg, rois, config)

    agg = eval_type_to_aggregator.get('bas_pxl_eval', None)


def _setup_sc_analysis():
    from watch.mlops.aggregate import AggregateEvluationConfig
    from watch.mlops.aggregate import build_tables
    from watch.mlops.aggregate import build_aggregators
    import watch
    expt_dvc_dpath = watch.find_dvc_dpath(tags='phase2_expt', hardware='auto')
    cmdline = 0
    kwargs = {
        'root_dpath': expt_dvc_dpath / '_testpipe',
        'pipeline': 'joint_bas_sc',
        'io_workers': 20,
        'freeze_cache': 0,
        # 'pipeline': 'joint_bas_sc_nocrop',
        # 'root_dpath': expt_dvc_dpath / '_testsc',
        #'pipeline': 'sc',
    }
    config = AggregateEvluationConfig.cli(cmdline=cmdline, data=kwargs)
    eval_type_to_results = build_tables(config)
    agg_dpath = ub.Path(config['root_dpath']) / 'aggregate'
    eval_type_to_aggregator = build_aggregators(eval_type_to_results, agg_dpath)
    agg = ub.peek(eval_type_to_aggregator.values())
    agg = eval_type_to_aggregator.get('sc_poly_eval', None)
    print(f'agg={agg}')
    rois = {'KR_R001', 'KR_R002', 'BR_R002'}
    print(f'rois={rois}')


def _setup_bas():
    from watch.mlops.aggregate import AggregateEvluationConfig
    from watch.mlops.aggregate import build_tables
    from watch.mlops.aggregate import build_aggregators
    import watch
    expt_dvc_dpath = watch.find_dvc_dpath(tags='phase2_expt', hardware='auto')
    cmdline = 0
    kwargs = {
        'root_dpath': expt_dvc_dpath / '_testpipe',
        'pipeline': 'bas',
        'io_workers': 20,
        'freeze_cache': 0,
        # 'pipeline': 'joint_bas_sc_nocrop',
        # 'root_dpath': expt_dvc_dpath / '_testsc',
        #'pipeline': 'sc',
    }
    config = AggregateEvluationConfig.cli(cmdline=cmdline, data=kwargs)
    eval_type_to_results = build_tables(config)
    agg_dpath = ub.Path(config['root_dpath']) / 'aggregate'
    eval_type_to_aggregator = build_aggregators(eval_type_to_results, agg_dpath)
    agg = ub.peek(eval_type_to_aggregator.values())

    agg = eval_type_to_aggregator.get('bas_poly_eval', None)
    fname = f'{agg.type}_{agg.agg_dpath.parent.name}.csv'
    agg.table().to_csv(fname, index_label=False)

    agg = eval_type_to_aggregator.get('bas_pxl_eval', None)
    fname = f'{agg.type}_{agg.agg_dpath.parent.name}.csv'
    agg.table().to_csv(fname, index_label=False)

    print(f'agg={agg}')
    rois = {'KR_R001', 'KR_R002'}
    # rois = {'KR_R001', 'KR_R002', 'BR_R002'}
    print(f'rois={rois}')


def _resource_table(eval_type_to_aggregator):
    agg1 = eval_type_to_aggregator.get('bas_poly_eval', None)
    agg2 = eval_type_to_aggregator.get('bas_pxl_eval', None)

    agg2.resolved_info['resources'].sum()

    mapping = {
        'bas_poly_eval.params.bas_pxl.package_fpath': 'models',
        'bas_poly_eval.params.bas_poly.moving_window_size': 'tracking_window_size',
        'bas_poly_eval.params.bas_poly.thresh': 'tracking_threshold',
        'bas_poly_eval.params.bas_pxl.input_space_scale': 'heatmap_gsd',
        'bas_poly_eval.params.bas_poly.max_area_sqkm': 'max_area_km_threshold',
        'bas_poly_eval.params.bas_poly.min_area_sqkm': 'min_area_km_threshold',
        'bas_poly_eval.params.bas_poly.min_area_square_meters': 'min_area_meters',
        'bas_poly_eval.params.bas_poly.max_area_square_meters': 'max_area_meters',
        'bas_poly_eval.params.bas_poly.polygon_simplify_tolerance': 'polygon_simplify_tolerence',
        'bas_poly_eval.params.bas_pxl.chip_dims': 'input_window_dimensions',
        'bas_poly_eval.params.bas_pxl.test_dataset': 'regions',
    }
    summary_rows = []
    for colname, col in agg1.effective_params.fillna('None').T.iterrows():
        histo = col.value_counts()
        if len(histo) > 1:
            if colname in mapping:
                summary_rows.append({
                    'param': mapping[colname],
                    'num_unique': len(histo),
                })
            print(f'colname={colname}')
            print(histo)
    df = pd.DataFrame(summary_rows)
    from watch.utils import util_kwplot
    dfh = humanize_dataframe(df)
    dataframe_table(dfh, 'foo.png')

    from watch.utils import util_pandas
    resource_df = util_pandas.pandas_shorten_columns(agg2.resolved_info['resources'])
    resource_df = resource_df.drop(['vram_gb'], axis=1)
    resouce_table = resource_df.sum(numeric_only=True, axis=0).to_frame('bas_pxl')
    resouce_table.loc['num_params', 'bas_pxl'] = len(agg2)
    resouce_table.loc[:, 'bas_poly'] = '<pending>'
    resouce_table.loc['num_params', 'bas_poly'] = len(agg1)
    dfh = humanize_dataframe(resouce_table, title='Pipeline Summary')
    dataframe_table(dfh, 'pipeline_summary.png')


def report_top_results(agg):
    rois = {'KR_R001', 'KR_R002', 'BR_R002', 'AE_R001'}
    agg.build_macro_tables(rois)

    macro_results = agg.region_to_tables[agg.primary_macro_region].copy()

    z = agg.report_best()
    best_macro_param_hashid = list(z[0].values())[-1]['param_hashid'].iloc[0]
    agg_best = agg.filterto(param_hashids=[best_macro_param_hashid])

    agg_best.build_macro_tables()
    agg_best.report_best()


def generate_kr2_heatmaps(agg):
    agg1 = agg.filterto(models=['package_epoch0_step41'])
    rois = {'KR_R001', 'KR_R002', 'BR_R002'}
    agg1 = agg1.compress(agg1.params['bas_poly_eval.params.bas_poly.thresh'] == 0.12)
    agg1 = agg1.compress(agg1.params['bas_poly_eval.params.bas_poly.moving_window_size'].isna())
    agg1 = agg1.filterto(param_hashids=['txfrewydmfeb'])
    agg1.build_single_macro_table(rois)
    agg1 = agg1.compress(agg1.index['region_id'] == 'KR_R002')

    eval_fpath = agg1.fpaths.iloc[0]
    from watch.mlops import confusion_visualization
    confusion_visualization.bas_poly_eval_confusion_analysis(eval_fpath)

    rois = {'KR_R001', 'KR_R002', 'BR_R002'}
    agg1 = agg.filterto(models=['Drop4_BAS_15GSD_BGRNSH_invar_V8_epoch=16-step=8704'])
    # agg1 = agg.filterto(param_hashids=['cvvlwbictocz'])
    agg1.build_single_macro_table(rois)
    best = agg1.report_best()
    param_hashid = list(best[0].values())[-1].iloc[0]['param_hashid']
    agg1 = agg1.filterto(param_hashids=[param_hashid])
    agg1 = agg1.compress(agg1.index['region_id'] == 'KR_R002')
    eval_fpath = agg1.fpaths.iloc[0]
    print(f'eval_fpath={eval_fpath}')

    from watch.mlops import confusion_visualization
    confusion_visualization.bas_poly_eval_confusion_analysis(eval_fpath)

    rois = {'KR_R001', 'KR_R002', 'BR_R002'}
    agg1 = agg.filterto(models=['Drop4_BAS_2022_12_15GSD_BGRN_V10_epoch=0-step=4305'])
    # agg1 = agg.filterto(param_hashids=['cvvlwbictocz'])
    agg1.build_single_macro_table(rois)
    best = agg1.report_best()
    param_hashid = list(best[0].values())[-1].iloc[0]['param_hashid']
    agg1 = agg1.filterto(param_hashids=[param_hashid])
    agg1 = agg1.compress(agg1.index['region_id'] == 'KR_R002')
    eval_fpath = agg1.fpaths.iloc[0]
    print(f'eval_fpath={eval_fpath}')

    from watch.mlops import confusion_visualization
    confusion_visualization.bas_poly_eval_confusion_analysis(eval_fpath)


def check_baseline(eval_type_to_aggregator):
    from watch.utils.util_pandas import DotDictDataFrame
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
    #     result = smart_result_parser.load_sc_poly_eval(fpath, None)
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


def quick_heatmap_viz():
    import ubelt as ub
    import kwcoco
    import kwarray
    dmj_dpath = ub.Path('/data/david.joy/DataFor2023Jan31Delivery/KW_R001_eval_8/2021-08-31/split/mono/products/bas-fusion')
    kwcoco_fpath = dmj_dpath / 'bas_fusion_kwcoco.json'
    dset = kwcoco.CocoDataset(kwcoco_fpath)
    video_name = 'KW_R001'
    video = dset.index.name_to_video[video_name]
    video_id = video['id']
    images = dset.images(video_id=video_id)

    # Average all heatmaps together
    running = kwarray.RunningStats()
    for coco_img in ub.ProgIter(images.coco_images, desc='loading images'):
        delayed = coco_img.imdelay('salient', resolution='10 GSD', nodata_method='float')
        heatmap = delayed.finalize()
        running.update(heatmap)

    import kwplot
    import kwimage
    from watch.utils import util_kwimage
    stats = running.current()
    average_heatmap = stats['mean']
    average_heatmap = util_kwimage.exactly_1channel(average_heatmap)
    canvas = kwplot.make_heatmask(average_heatmap)[:, :, 0:3]
    canvas = kwimage.ensure_uint255(canvas)
    kwimage.imwrite('average_heatmap.png', canvas)

    import kwplot
    kwplot.autompl()

    kwplot.imshow(stats['min'], cmap='plasma', data_colorbar=True, title='min response', fnum=1)
    kwplot.imshow(stats['max'], cmap='plasma', data_colorbar=True, title='max response', fnum=2)
    kwplot.imshow(stats['mean'], cmap='plasma', data_colorbar=True, title='mean response', fnum=3)
    kwplot.imshow(stats['std'], cmap='plasma', data_colorbar=True, title='std response', fnum=4)


def remove_specific_runs_by_region():
    """
    Find specific regions and remove runs for them
    """
    from watch.mlops import aggregate_loader
    import watch
    expt_dvc_dpath = watch.find_dvc_dpath(tags='phase2_expt', hardware='auto')
    root_dpath = expt_dvc_dpath / '_split6_toothbrush_meanyear'
    pipeline = 'bas'
    io_workers = 16
    # eval_type_to_results = aggregate_loader.build_tables(root_dpath, pipeline, io_workers)
    # eval_type_to_results['bas_pxl_eval']

    from watch.mlops import smart_pipeline
    dag = smart_pipeline.make_smart_pipeline(pipeline)
    dag.configure(config=None, root_dpath=root_dpath)

    remove_regions = [
        'NZ_R001',
        'BR_R001',
        'BR_R002',
    ]

    stages_of_interest = ['bas_pxl', 'bas_pxl_eval', 'bas_poly_eval']
    existing = {}

    bad_dpaths = []
    for stage in stages_of_interest:
        rows = dag.nodes[stage].find_template_outputs(workers=8)
        for row in rows:
            test_dset_fpath = ub.Path(row['request.bas_pxl.test_dataset'])
            if any([r in test_dset_fpath.name for r in remove_regions]):
                bad_dpath = row['dpath']
                bad_dpaths.append(bad_dpath)

    def expand_pred(dpath):
        pred_link_dpath = dpath / '.pred'
        for pred_dpath in list(pred_link_dpath.glob('*/*')):
            yield pred_dpath
            yield from expand_pred(pred_dpath)

    def expand_succ(dpath):
        succ_link_dpath = dpath / '.succ'
        for succ_dpath in list(succ_link_dpath.glob('*/*')):
            yield succ_dpath
            yield from expand_succ(succ_dpath)

    complement_bad_dpaths = set()

    for dpath in list(bad_dpaths):
        if dpath not in complement_bad_dpaths:
            complement_bad_dpaths.update(list(expand_pred(dpath)))
            complement_bad_dpaths.update(list(expand_succ(dpath)))

    closed_bad_dpaths = sorted([p.resolve() for p in list(complement_bad_dpaths) + bad_dpaths])
    print('closed_bad_dpaths = {}'.format(ub.urepr(closed_bad_dpaths, nl=1)))

    for dpath in closed_bad_dpaths:
        dpath.delete()
