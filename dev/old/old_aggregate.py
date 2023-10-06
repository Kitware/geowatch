"""
Old aggregate code that hasnt been used in awhile and likely needs to be
refactored or removed.
"""


def automated_analysis(eval_type_to_aggregator, config):
    timestamp = ub.timestamp()
    output_dpath = ub.Path(config['root_dpath']) / 'aggregate'
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
            from watch.mlops import confusor_analysis
            eval_dpath = eval_fpath.parent
            cfsn_dpath = eval_dpath / 'confusion_analysis'
            confusor_analysis.main(cmdline=0, metrics_node_dpath=eval_dpath,
                                   out_dpath=cfsn_dpath)
            # TODO: use the region_id.
            ub.symlink(real_path=eval_dpath, link_path=agg_group_dpath / eval_dpath.name)

    agg0 = eval_type_to_aggregator.get('bas_pxl_eval')
    if agg0 is not None:
        # agg[agg.primary_metric_cols]
        generic_analysis(agg0, macro_groups, selector)

    agg0 = eval_type_to_aggregator.get('sc_poly_eval', None)
    if agg0 is not None:
        ...


def generic_analysis(agg0, macro_groups=None, selector=None):
    import pandas as pd
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

    print('chosen_macro_rois = {}'.format(ub.urepr(chosen_macro_rois, nl=1)))
    agg0_.build_macro_tables(chosen_macro_rois)

    report0 = agg0_.report_best(top_k=1)
    params_of_interest = pd.concat(report0.region_id_to_summary.values())['param_hashid'].value_counts()
    params_of_interest = list(report0.top_param_lut.keys())
    n1 = len(params_of_interest)
    n2 = len(agg0_.index['param_hashid'])
    print(f'Restrict to {n1} / {n2} top parameters')

    subagg1 = agg0_.filterto(param_hashids=params_of_interest)
    subagg1.build_macro_tables(chosen_macro_rois)
    models_of_interest = subagg1.effective_params[subagg1.model_cols].value_counts()
    print('models_of_interest = {}'.format(ub.urepr(models_of_interest, nl=1)))

    report1 = subagg1.report_best(top_k=1)
    param_hashid = report1.region_id_to_summary[hash_regions(selector)]['param_hashid'].iloc[0]
    params_of_interest1 = [param_hashid]
    # params_of_interest1 = [list(report1.region_id_to_summary.values())[-1]['param_hashid'].iloc[0]]

    n1 = len(params_of_interest1)
    n2 = len(agg0_.index['param_hashid'])
    print(f'Restrict to {n1} / {n2} top parameters')
    subagg2 = agg0_.filterto(param_hashids=params_of_interest1)
    subagg2.build_macro_tables(chosen_macro_rois)
    report2 = subagg2.report_best(top_k=1)  # NOQA
    return subagg2


def fix_duplicate_param_hashids(agg0):
    import kwarray
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
