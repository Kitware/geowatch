"""
Loads and summarizes pre-computed metrics over multiple experiments


The main function is :func:`gather_measures`.
"""
import json
import pandas as pd
import numpy as np
import ubelt as ub
import yaml
import shutil
import scriptconfig as scfg


class GatherResultsConfig(scfg.Config):
    """
    TODO: write good docs for the gather command line tool.

    Basic idea:

        Grabs a selection of precomputed metrics on a particular dataset

        Compares models against each other with statistical tests and plots.

        Tries to figure out which configs did best.
    """
    default = {
        'measure_globstr': scfg.Value('measures2.json', help='a group of measures2.json files from kwcoco metrics, specified by list or glob pattern'),
        'out_dpath': scfg.Value('./agg_results', help='A location where aggregate results can be written and compared'),
        'show': scfg.Value(False, help='if true, does a plt.show')
    }


class Found(Exception):
    pass


def _writefig(fig, dpath, fname, figsize, verbose, tight):
    fig_fpath = dpath / fname
    if verbose:
        print('write fig_fpath = {!r}'.format(fig_fpath))
    fig.set_size_inches(figsize)
    if tight:
        fig.tight_layout()
    fig.savefig(fig_fpath)


def load_measure(measure_fpath):
    """
    Workers to load a single measure path. Has a hack to fix old configs.
    This can eventually be removed.
    """
    with open(measure_fpath, 'r') as file:
        info = json.load(file)
    if True:
        # Hack to ensure fit config is properly serialized
        try:
            predict_meta = None
            for meta_item in info['meta']['info']:
                if meta_item['type'] == 'process':
                    if meta_item['properties']['name'] == 'watch.tasks.fusion.predict':
                        predict_meta = meta_item
                        raise Found
        except Found:
            pass
        else:
            raise Exception('no prediction metadata')
        process_props = predict_meta['properties']
        predict_args = process_props['args']
        cand_remote = process_props['hostname']
        need_fit_config_hack = 'fit_config' not in process_props
        if need_fit_config_hack:
            # Hack, for models where I forgot to serialize the fit
            # configuration.
            print('Hacking in fit-config')
            package_fpath = predict_args['package_fpath']
            # hack, dont have enough into to properly remove the user directory
            hack_home = ub.expandpath(f'$HOME/remote/{cand_remote}')
            cand_remote_home = ub.Path(hack_home)
            tmp = ub.Path(package_fpath)
            possible_home_dirs = [
                ub.Path('/home/local/KHQ'),
                ub.Path('/home'),
            ]
            cand_suffix = None
            for possible_home in possible_home_dirs:
                possible_home_parts = possible_home.parts
                n = len(possible_home_parts)
                if tmp.parts[:n] == possible_home_parts:
                    cand_suffix = '/'.join(tmp.parts[n + 1:])
                    break
            if cand_suffix is None:
                raise Exception
            cand_remote_fpath = cand_remote_home / cand_suffix
            if cand_remote_fpath.exists():
                base_file = ub.zopen(cand_remote_fpath, ext='.pt')
                found = None
                for subfile in base_file.namelist():
                    if 'package_header/fit_config.yaml' in subfile:
                        found = subfile
                file = ub.zopen(cand_remote_fpath / found, ext='.pt')
                fit_config = yaml.safe_load(file)
                # TODO: this should have already existed
                process_props['fit_config'] = fit_config
                print('Backup measures: {}'.format(measure_fpath))
                shutil.copy(measure_fpath, ub.augpath(measure_fpath, suffix='.bak'))
                with open(measure_fpath, 'w') as file:
                    json.dump(info, file)
            else:
                raise Exception
    return info


def prepare_results(all_infos):
    from kwcoco.coco_evaluator import CocoSingleResult
    from watch.utils import result_analysis
    class_rows = []
    mean_rows = []
    all_results = []
    results_list2 = []
    for info in ub.ProgIter(all_infos):
        # Note: for now, the nocls part will refer to the saliency metrics and
        # the ovr part will be the class metrics. Should make per-head results
        # in the future.
        result = CocoSingleResult.from_json(info)
        all_results.append(result)

        class_aps = []
        class_aucs = []

        meta = info['meta']

        try:
            predict_meta = None
            for meta_item in meta['info']:
                if meta_item['type'] == 'process':
                    if meta_item['properties']['name'] == 'watch.tasks.fusion.predict':
                        predict_meta = meta_item
                        raise Found
        except Found:
            pass
        else:
            raise Exception('no prediction metadata')

        pred_fpath = predict_meta['properties']['args']['pred_dataset']

        _ = ub.Path(pred_fpath)
        model_fpath = (_.parent.parent.parent / (_.parent.parent.name.split('pred_')[-1] + '.pt'))

        title = meta['title']

        if 'package_name' not in meta:
            if ' ' not in title:
                package_name = title
            else:
                raise AssertionError
        else:
            package_name = meta['package_name']

        # Hack to get the epoch/step/expt_name
        epoch = int(package_name.split('epoch=')[1].split('-')[0])
        step = int(package_name.split('step=')[1].split('-')[0])
        expt_name = package_name.split('epoch=')[0]

        salient_measures = info['nocls_measures']
        class_measures = info['ovr_measures']

        expt_class_rows = []
        for catname, bin_measure in class_measures.items():
            class_aps.append(bin_measure['ap'])
            class_aucs.append(bin_measure['auc'])
            class_row = {}
            class_row['AP'] = bin_measure['ap']
            class_row['AUC'] = bin_measure['auc']
            class_row['APUC'] = np.nanmean([bin_measure['ap'], bin_measure['auc']])
            class_row['catname'] = catname
            class_row['title'] = title
            class_row['package_name'] = package_name
            class_row['expt_name'] = expt_name
            class_row['epoch'] = epoch
            class_row['step'] = step
            class_row['pred_fpath'] = pred_fpath
            class_row['model_fpath'] = str(model_fpath)
            expt_class_rows.append(class_row)
        class_rows.extend(expt_class_rows)

        row = {}
        row['class_mAP'] = np.nanmean(class_aps) if len(class_aps) else np.nan
        row['class_mAUC'] = np.nanmean(class_aucs) if len(class_aucs) else np.nan
        row['class_mAPUC'] = np.nanmean([row['class_mAUC'], row['class_mAP']])

        row['salient_AP'] = salient_measures['ap']
        row['salient_AUC'] = salient_measures['auc']
        row['salient_APUC'] = np.nanmean([row['salient_AP'], row['salient_AUC']])

        row['catname'] = 'all'
        row['package_name'] = package_name
        row['title'] = title
        row['expt_name'] = expt_name
        row['epoch'] = epoch
        row['step'] = step
        row['pred_fpath'] = pred_fpath
        row['model_fpath'] = str(model_fpath)

        mean_rows.append(row)

        if predict_meta is not None:
            process_props = predict_meta['properties']
            predict_args = process_props['args']
            cand_remote = process_props['hostname']

            if 'fit_config' in process_props:
                fit_config = process_props['fit_config']
                # Add in defaults for new params
                fit_config.setdefault('normalize_perframe', False)
                result.meta['fit_config'] = fit_config
            else:
                raise Exception('Fit config was not serialized correctly')

            metrics = {
                'class_mAP': row['class_mAP'],
                'class_mAUC': row['class_mAUC'],
                'class_mAPUC': row['class_mAPUC'],
                'salient_AP': row['salient_AP'],
                'salient_AUC': row['salient_AUC'],
                'salient_APUC': row['salient_APUC'],
            }
            for class_row in expt_class_rows:
                metrics[class_row['catname'] + '_AP'] = class_row['AP']
                metrics[class_row['catname'] + '_AUC'] = class_row['AUC']

            # Add relevant train params here
            row['channels'] = fit_config['channels']
            row['time_steps'] = fit_config['time_steps']
            row['chip_size'] = fit_config['chip_size']
            row['arch_name'] = fit_config['arch_name']
            row['normalize_perframe'] = fit_config.get('normalize_perframe', False)
            row['normalize_inputs'] = fit_config.get('normalize_inputs', False)
            row['train_remote'] = cand_remote

            def hack_smartcast(x):
                try:
                    return int(x)
                except Exception:
                    pass

            from scriptconfig import smartcast
            # hack
            fit_config2 = {}
            for k, v in fit_config.items():
                if k not in {'channels', 'init'}:
                    fit_config2[k] = smartcast.smartcast(v)
                else:
                    fit_config2[k] = v
            fit_config = fit_config2
            # fit_config = ub.map_vals(smartcast.smartcast, fit_config)

            predict_args  # add predict window overlap
            # row['train_remote'] = cand_remote

            result2 = result_analysis.Result(
                 name=result.meta['title'],
                 params=fit_config,
                 metrics=metrics,
            )
            results_list2.append(result2)

    best_candidates(class_rows, mean_rows)

    return class_rows, mean_rows, all_results, results_list2


def best_candidates(class_rows, mean_rows):
    # TOP CANDIDATE MODELS - FIND TOP K MODELS FOR EVERY METRIC
    K = 7
    max_per_metric_per_expt = 3
    cand_expt_names = set()

    mean_metrics = ['class_mAP', 'class_mAUC', 'salient_AP', 'salient_AUC', 'class_mAPUC', 'salient_APUC']
    class_metrics = ['AP', 'AUC', 'APUC', 'catname']

    subsets = {}
    if len(class_rows):
        class_df = pd.DataFrame(class_rows)
        class_candidate_indexes = []
        for class_metric in class_metrics:
            for catname, group in class_df.groupby('catname'):
                valid_indexes = []
                for expt_name, subgroup in group.groupby('expt_name'):
                    best_subgroup = subgroup.sort_values(class_metric, ascending=False).iloc[0:max_per_metric_per_expt]
                    valid_indexes.extend(best_subgroup.index.tolist())
                valid_indexes = sorted(set(valid_indexes))
                valid_group = group.loc[valid_indexes]
                top_group = valid_group.sort_values(class_metric, ascending=False).iloc[0:K]
                class_candidate_indexes.extend(top_group.index)
        top_class_indexes = sorted(set(class_candidate_indexes))
        class_subset = class_df.loc[top_class_indexes]
        subsets['class'] = class_subset = class_subset.sort_values('AP')
        cand_expt_names.update(set(class_subset['model_fpath'].tolist()))

    else:
        class_subset = []
        top_class_indexes = []

    if len(mean_rows):
        mean_df = pd.DataFrame(mean_rows)
        mean_candidate_indexes = []
        for metric in mean_metrics:
            valid_indexes = []
            for expt_name, subgroup in mean_df.groupby('expt_name'):
                best_subgroup = subgroup.sort_values(metric, ascending=False).iloc[0:max_per_metric_per_expt]
                valid_indexes.extend(best_subgroup.index.tolist())
            valid_indexes = sorted(set(valid_indexes))
            valid_group = mean_df.loc[valid_indexes]
            top_group = valid_group.sort_values(metric, ascending=False).iloc[0:K]
            mean_candidate_indexes.extend(top_group.index)
        top_mean_indexes = sorted(set(mean_candidate_indexes))
        mean_subset = mean_df.loc[top_mean_indexes]
        subsets['mean'] = mean_subset = mean_subset.sort_values('class_mAPUC')
        cand_expt_names.update(set(mean_subset['model_fpath'].tolist()))

        sc_mean_subset = mean_subset[~mean_subset['class_mAPUC'].isnull()].sort_values('class_mAPUC')
        bas_mean_subset = mean_subset[~mean_subset['salient_APUC'].isnull()].sort_values('salient_APUC')
    else:
        mean_subset = []
        top_mean_indexes = []
        sc_mean_subset = []
        bas_mean_subset = []

    model_candidates = ub.ddict(list)
    pred_candidates = ub.ddict(list)

    if len(class_subset):
        print('Best Subset Table (Per-Class):')
        print(class_subset[class_metrics + ['package_name']].to_string())
        model_candidates['sc'].append(class_subset['model_fpath'].values.tolist())
        pred_candidates['sc'].append(class_subset['pred_fpath'].values.tolist())

    if len(bas_mean_subset):
        print('Best Subset Table (Mean-BAS):')
        print(bas_mean_subset[mean_metrics + ['package_name']].to_string())
        model_candidates['bas'].append(bas_mean_subset['model_fpath'].values.tolist())
        pred_candidates['bas'].append(bas_mean_subset['pred_fpath'].values.tolist())

    if len(sc_mean_subset):
        print('Best Subset Table (Mean-SC):')
        print(sc_mean_subset[mean_metrics + ['package_name']].to_string())
        model_candidates['sc'].append(sc_mean_subset['model_fpath'].values.tolist())
        pred_candidates['sc'].append(sc_mean_subset['pred_fpath'].values.tolist())

    for n, s in subsets.items():
        print('n = {!r}'.format(n))
        print(shrink_notations(s, drop=1))

    sc_model_candidates = list(ub.unique(ub.flatten(model_candidates['sc'])))
    bas_model_candidates = list(ub.unique(ub.flatten(model_candidates['bas'])))

    sc_pred_candidates = list(ub.unique(ub.flatten(pred_candidates['sc'])))
    bas_pred_candidates = list(ub.unique(ub.flatten(pred_candidates['bas'])))

    print('sc_model_candidates = {}'.format(ub.repr2(sc_model_candidates, nl=1)))
    print('bas_model_candidates = {}'.format(ub.repr2(bas_model_candidates, nl=1)))

    print('sc_pred_candidates = {}'.format(ub.repr2(sc_pred_candidates, nl=1)))
    print('bas_pred_candidates = {}'.format(ub.repr2(bas_pred_candidates, nl=1)))

    cand_expt_names = sorted(cand_expt_names)
    print('cand_expt_names = {}'.format(ub.repr2(cand_expt_names, nl=1)))
    return cand_expt_names


def shrink_notations(df, drop=0):
    import kwcoco
    import re
    from watch.utils import util_regex
    b = util_regex.PythonRegexBuilder()
    pat0 = r'v\d+'
    pat1 = '^{pat}$'.format(pat=pat0)
    pat2 = b.lookbehind('_') + pat0 + b.optional(b.lookahead('_'))
    pat_text = b.oneof(*map(b.group, (pat1, pat2)))
    pat = re.compile(pat_text)

    df = df.copy()

    if 0:
        df['expt_name'] = (
            df['expt_name'].apply(
                lambda x: pat.search(x).group()
            ))
    if 'channels' in df:
        df['channels'] = (
            df['channels'].apply(
                lambda x: kwcoco.ChannelSpec.coerce(x.replace('matseg_', 'matseg.')).concise().spec
            ))
        df['channels'] = (
            df['channels'].apply(
                lambda x: x.replace('blue|green|red|nir|swir16|swir22', 'BGRNSH'))
        )
        df['channels'] = (
            df['channels'].apply(
                lambda x: x.replace('brush|bare_ground|built_up', 'seg:3'))
        )

    if drop:
        drop_cols = set(df.columns) & {
            'title', 'catname', 'normalize_perframe', 'normalize_inputs',
            'train_remote', 'step', 'arch_name', 'package_name',
            'pred_fpath', 'model_fpath',
        }
        df = df.drop(drop_cols, axis=1)
    return df


def _oldhack():
    """
    if 0:
        # Hack to move over data into a comparable eval
        k1 = 'Drop1-Aligned-L1-2022-01_combo_DILM_nowv_vali.kwcoco'
        k2 = 'combo_DILM_nowv_vali.kwcoco'
        a = sorted(dset_groups[k1])
        b = sorted(dset_groups[k2])
        for x in b:
            y = ub.Path(str(x).replace(k2, k1))
            print(y in a)
            if not y.exists():
                p1 = x.parent.parent.parent
                p2 = y.parent.parent.parent
                ub.symlink(p1, p2, verbose=1)
    """


def gather_measures(cmdline=False, **kwargs):
    """
    Ignore:
        from watch.tasks.fusion.gather_results import *  # NOQA
        import watch
        dvc_dpath = watch.find_smart_dvc_dpath()
        measure_globstr = 'models/fusion/SC-20201117/*/*/*/eval/curves/measures2.json'
        measure_globstr = 'models/fusion/SC-20201117/*_TA1*/*/*/eval/curves/measures2.json'
        kwargs['measure_globstr'] = dvc_dpath / measure_globstr

        if 0:
            remote = 'namek'
            remote_dpath = ub.Path(ub.shrinkuser(dvc_dpath, home=ub.expandpath(f'$HOME/remote/{remote}')))
            dvc_dpath = remote_dpath
    """
    import watch
    from watch.utils import result_analysis
    from watch.utils import util_path
    import matplotlib as mpl

    config = GatherResultsConfig(cmdline=cmdline, **kwargs)
    print('config = {}'.format(ub.repr2(config.asdict(), nl=1)))

    measure_globstr = config['measure_globstr']
    out_dpath = ub.Path(config['out_dpath']).ensuredir()

    # TODO: high level results for a model should be serialized to DVC
    if measure_globstr is None:
        # model_dpath = ub.Path(dvc_dpath) / 'models/fusion/unevaluated-activity-2021-11-12'
        # model_dpath = fusion_model_dpath / 'unevaluated-activity-2021-11-12'
        # fusion_model_dpath = dvc_dpath / 'models/fusion/'
        # model_dpath = fusion_model_dpath / 'SC-20201117'
        # measure_fpaths = list(model_dpath.glob('eval_links/*/curves/measures2.json'))
        dvc_dpath = watch.find_smart_dvc_dpath()
        measure_globstr = 'models/fusion/SC-20201117/*/*/*/eval/curves/measures2.json'
        measure_fpaths = list(dvc_dpath.glob(measure_globstr))
    else:
        measure_fpaths = util_path.coerce_patterned_paths(measure_globstr)

    measure_fpaths = [ub.Path(p) for p in measure_fpaths]

    dset_groups = ub.group_items(measure_fpaths, lambda x: x.parent.parent.parent.name)
    print('dset_groups = {}'.format(ub.repr2(dset_groups, nl=2)))

    predict_group_freq = ub.map_vals(len, dset_groups)
    print('These are the different datasets prediction was run on.')
    print('TODO: need to choose exactly 1 or a compatible set of them')
    print('predict_group_freq = {}'.format(ub.repr2(predict_group_freq, nl=1)))

    # measure_fpaths = dset_groups['combo_train_US_R001_small_nowv.kwcoco']
    # measure_fpaths = dset_groups['combo_vali_nowv.kwcoco']
    # measure_fpaths = dset_groups['combo_DILM_nowv_vali.kwcoco']

    # dataset_key = 'Drop1-Aligned-TA1-2022-01_vali_data_nowv.kwcoco'
    # dataset_key = 'Drop1-Aligned-L1-2022-01_combo_DILM_nowv_vali.kwcoco'
    # dataset_key = 'combo_DILM.kwcoco_vali'
    # dataset_key = 'Drop1-Aligned-L1-2022-01_vali_data_nowv.kwcoco'

    # TODO: this makes this script non-portable. Need to parameterize

    # dataset_key = 'Drop2-Aligned-TA1-2022-01_data_nowv_vali.kwcoco'
    dataset_keys = [

        # 'Drop1-Aligned-L1-2022-01_combo_DILM_nowv_vali.kwcoco',
        # 'Drop1-Aligned-L1-2022-01_vali_data_nowv.kwcoco',
        # 'Drop1-Aligned-TA1-2022-01_vali_data_nowv.kwcoco',
        # 'Drop1-Aligned-L1-2022-01_vali_data_nowv.kwcoco',
        # 'Drop1-Aligned-L1-2022-01_combo_DILM_nowv_vali.kwcoco'
        # 'Drop1-Aligned-TA1-2022-01_vali_data_nowv.kwcoco',
        # 'Drop2-Aligned-TA1-2022-01_data_nowv_vali.kwcoco',

        # 'Drop2-Aligned-TA1-2022-01_data_nowv_vali.kwcoco',
        'Drop2-Aligned-TA1-2022-01_combo_L_nowv_vali.kwcoco',
        # 'Drop2-Aligned-TA1-2022-01_combo_L_nowv.kwcoco',
    ]

    # dataset_key = 'combo_vali_nowv.kwcoco'

    measure_fpaths = list(ub.flatten([dset_groups[k] for k in dataset_keys]))
    # dataset_key = 'Drop2-Aligned-TA1-2022-01_data_nowv_vali.kwcoco'
    # measure_fpaths = dset_groups[dataset_key]

    print(len(measure_fpaths))
    # dataset_to_evals = ub.group_items(eval_dpaths, lambda x: x.parent.name)

    jobs = ub.JobPool('thread', max_workers=10)
    all_infos = []
    for measure_fpath in ub.ProgIter(measure_fpaths):
        job = jobs.submit(load_measure, measure_fpath)
        job.measure_fpath = measure_fpath

    # job = next(iter(jobs.as_completed(desc='collect jobs')))
    # all_infos = [job.result()]
    failed_jobs = []
    for job in jobs.as_completed(desc='collect jobs'):
        try:
            all_infos.append(job.result())
        except Exception:
            failed_jobs.append(job.measure_fpath)
            print('Failed job.measure_fpath = {!r}'.format(job.measure_fpath))
            pass

    if 0:
        to_unlink = []
        for fpath in failed_jobs:
            if 'Drop2' in fpath.parent.parent.parent.name:
                dvc_fpath = ub.Path(str(fpath) + '.dvc')
                if dvc_fpath.exists():
                    to_unlink.append(fpath)
                    to_unlink.append(dvc_fpath)
                    print('fpath = {!r}'.format(fpath))
            print(len(list(fpath.parent.parent.parent.glob('*'))))
        for p in to_unlink:
            p.unlink()

    print(f'Failed Jobs {len(failed_jobs)=}/{len(jobs)}')

    class_rows, mean_rows, all_results, results_list2 = prepare_results(all_infos)

    ignore_params = {
        'default_root_dir', 'name', 'enable_progress_bar'
        'prepare_data_per_node', 'enable_model_summary', 'checkpoint_callback',
        'detect_anomaly', 'gpus', 'terminate_on_nan', 'train_dataset',
        'workdir', 'config', 'num_workers', 'amp_backend',
        'enable_progress_bar', 'flush_logs_every_n_steps',
        'enable_checkpointing', 'prepare_data_per_node', 'amp_level',
        'vali_dataset', 'test_dataset', 'package_fpath',
    }
    ignore_metrics = {
        'positive_AUC',
        'positive_AP',
        # 'nocls_auc',
        # 'nocls_ap',
        # 'map',
        # 'mauc',
    }

    analysis = result_analysis.ResultAnalysis(
        results_list2, ignore_params=ignore_params,
        # metrics=['class_mAPUC', 'salient_APUC'],
        metrics=['salient_AP'],
        ignore_metrics=ignore_metrics,
    )
    try:
        analysis.run()
    except Exception:
        print('Warning: Statistical analysis failed. Probably needs more data.')
    else:
        print('analysis.varied = {}'.format(ub.repr2(analysis.varied, nl=2)))
        if len(analysis.stats_table):
            analysis.stats_table = analysis.stats_table.sort_values('anova_rank_p')
            print(analysis.stats_table)

    class_df = pd.DataFrame(class_rows)
    mean_df = pd.DataFrame(mean_rows)
    class_df = class_df.drop(set(class_df.columns) & {'title', 'pred_fpath', 'package_name'}, axis=1)
    mean_df = mean_df.drop(set(mean_df.columns) & {'title', 'pred_fpath', 'package_name'}, axis=1)

    class_df = shrink_notations(class_df, drop=1)
    mean_df = shrink_notations(mean_df, drop=1)

    if 'class_mAPUC' in mean_df.columns:
        print('\nSort by class_mAPUC')
        print(mean_df.sort_values('class_mAPUC').to_string())

    if 'salient_APUC' in mean_df.columns:
        print('\nSort by salient_APUC')
        print(mean_df.sort_values('salient_APUC').to_string())

    if 'AP' in class_df.columns:
        print('\nClass: Sort by AP')
        print(class_df[~class_df['AP'].isnull()].sort_values('AP').to_string())

    if 'AUC' in class_df.columns:
        print('\nClass: Sort by AUC')
        print(class_df[~class_df['AUC'].isnull()].sort_values('AUC').to_string())

    # mean_df['title'].apply(lambda x: int(x.split('epoch=')[1].split('-')[0]))
    def group_by_best(mean_df, metric_key, shrink=False):
        bests = []
        for t, subdf in mean_df.groupby('expt_name'):
            idx = subdf[[metric_key]].idxmax()
            import math
            if not math.isnan(idx.item()):
                best = subdf.loc[idx]
                bests.append(best)
        best_per_expt = pd.concat(bests)
        if shrink:
            best_per_expt = shrink_notations(best_per_expt, drop=1)
            best_per_expt = best_per_expt.rename({'time_steps': 'time', 'chip_size': 'space'}, axis=1)
        return best_per_expt

    print('\nBest Class Models')
    try:
        best_per_expt = group_by_best(mean_df, 'class_mAP', shrink=True)
        best_per_expt = best_per_expt[~best_per_expt['class_mAP'].isnull()]
        print(best_per_expt.sort_values('class_mAP').to_string())

        if 0:
            import dataframe_image as dfi
            dfi.export(
                best_per_expt,
                "./tmp.png",
                table_conversion="chrome",
                fontsize=28,
                max_rows=-1,
            )
            import kwplot
            fig, _ = kwplot.imshow('./tmp.png', fnum=10)
            fig.tight_layout()
    except ValueError:
        pass

    # salient_metric = 'salient_APUC'
    # class_metric = 'class_mAPUC'
    salient_metric = 'salient_AP'
    class_metric = 'class_mAP'

    try:
        print('\nBest Salient Models')
        best_per_expt = group_by_best(mean_df, salient_metric, shrink=True)
        print(best_per_expt.sort_values(salient_metric).to_string())
    except ValueError:
        pass

    import kwplot
    sns = kwplot.autosns()
    plt = kwplot.autoplt()  # NOQA

    dataset_title_part = "-".join(dataset_keys)

    def plot_summary_over_epochs(y):
        data = mean_df[~mean_df[y].isnull()]
        ax = sns.lineplot(data=data, x='epoch', y=y, hue='expt_name', marker='o', style='channels')
        h, ell = ax.get_legend_handles_labels()
        ax.legend(h, ell, loc='lower right')
        ax.set_title(f'Pixelwise {y} metrics: {dataset_title_part}')  # todo: add train name
        # ax.set_title('Pixelwise mAP AC metrics: KR_R001 + KR_R002')
        fig = ax.figure
        return fig

    figsize = 'auto'
    verbose = 1
    if figsize == 'auto':
        figsize = (9, 7)

    kwplot.figure(fnum=1, doclf=True)
    y = class_metric
    fig1 = plot_summary_over_epochs(y)
    _writefig(fig1, out_dpath, 'epoch_summary_class.png', figsize, verbose, tight=True)

    kwplot.figure(fnum=2, doclf=True)
    y = salient_metric
    fig2 = plot_summary_over_epochs(y)
    _writefig(fig2, out_dpath, 'epoch_summary_salient.png', figsize, verbose, tight=True)

    # kwplot.figure(fnum=2, doclf=True)
    # ax = sns.lineplot(data=mean_df, x='epoch', y='class_mAUC', hue='expt_name', marker='o', style='channels')
    # ax.set_title('Pixelwise mAUC AC metrics: KR_R001 + KR_R002')

    # import kwimage
    import kwarray
    # distinct_colors_selection = kwimage.Color.distinct(255)

    def hash_color(data):
        import distinctipy
        key_hash = ub.hash_data(data, hasher='blake3')
        key_tensor = np.frombuffer(memoryview(key_hash.encode()), dtype=np.int32)
        rng = kwarray.ensure_rng(rng=key_tensor.sum(), api='python')
        color = distinctipy.get_random_color(rng=rng)
        return color

    def plot_individual_class_curves(catname, fnum, metric='ap'):
        from kwcoco.metrics import drawing
        max_num_curves = 16
        max_per_expt = None
        max_per_expt = 10
        fig = kwplot.figure(fnum=fnum, doclf=True)

        def lookup_metric(x):
            return x.ovr_measures[catname][metric]

        relevant_results = [r for r in all_results if catname in r.ovr_measures]
        # ub.group_items(relevant_results)

        if 1:
            # Take best per experiment
            groups = ub.group_items(relevant_results, key=lambda x: x.meta['fit_config']['name'])
            ordered_groups = []
            for name, group in groups.items():
                # if not ('v53' in name or 'v54' in name):
                #     continue
                ordered_group = sorted(group, key=lookup_metric)[::-1][:max_per_expt]
                ordered_groups.append(ordered_group)
            ordered_groups = sorted(ordered_groups, key=lambda g: lookup_metric(g[0]))[::-1]
            import itertools as it
            sorted_results = [x for x in ub.flatten(it.zip_longest(*ordered_groups)) if x is not None]
        else:
            sorted_results = sorted(relevant_results, key=lookup_metric)[::-1]

        results_to_plot = sorted_results[0:max_num_curves]
        results_to_plot = sorted(results_to_plot, key=lookup_metric)[::-1]
        # sorted_results = sorted(relevant_results, key=lookup_metric)[::-1]
        results_to_plot = sorted_results[0:max_num_curves]
        colors = kwplot.Color.distinct(len(results_to_plot))
        for idx, result in enumerate(results_to_plot):
            color = colors[idx]
            color = [kwplot.Color(color).as01()]
            measure = result.ovr_measures[catname]
            if 'package_name' in result.meta:
                prefix = result.meta['package_name']
            elif 'title' in result.meta:
                prefix = result.meta['title']
            else:
                prefix = '?label-unknown?'

            color = hash_color(prefix)

            kw = {'fnum': fnum}
            if metric == 'ap':
                drawing.draw_prcurve(measure, prefix=prefix, color=color, **kw)
            elif metric == 'auc':
                drawing.draw_roc(measure, prefix=prefix, color=color, **kw)
            else:
                raise KeyError
        fig.gca().set_title(f'Comparison of runs {metric}: {catname} -\n{dataset_title_part}')
        return fig

    def plot_individual_salient_curves(fnum, metric='ap'):
        from kwcoco.metrics import drawing
        max_num_curves = 16
        max_per_expt = None
        max_per_expt = 10
        fig = kwplot.figure(fnum=fnum, doclf=True)
        relevant_results = [r for r in all_results if r.nocls_measures and r.nocls_measures['nsupport'] > 0]

        for result in relevant_results:
            if 'package_name' in result.meta:
                prefix = result.meta['package_name']
            elif 'title' in result.meta:
                prefix = result.meta['title']
            else:
                prefix = '?label-unknown?'
            result.meta['prefix'] = prefix

        def lookup_metric(x):
            return x.nocls_measures[metric]

        if 1:
            # Take best per experiment
            groups = ub.group_items(relevant_results, key=lambda x: x.meta['fit_config']['name'])

            if 0:
                # HACK!!!!
                groups2 = {}
                for name in groups.keys():
                    group = groups[name]
                    if not ('v53' in name or 'v54' in name):
                        continue
                    group2 = []
                    for g in group:
                        flag1 = 'v53_epoch=15' in g.meta['prefix']
                        flag2 = 'v54_epoch=13' in g.meta['prefix']
                        if flag2 or flag1:
                            group2.append(g)
                    group = group2
                    if group:
                        groups2[name] = group
                groups = groups2
            ordered_groups = []
            for name, group in groups.items():
                ordered_group = sorted(group, key=lookup_metric)[::-1][:max_per_expt]
                ordered_groups.append(ordered_group)
            ordered_groups = sorted(ordered_groups, key=lambda g: lookup_metric(g[0]))[::-1]
            import itertools as it
            sorted_results = [x for x in ub.flatten(it.zip_longest(*ordered_groups)) if x is not None]
        else:
            sorted_results = sorted(relevant_results, key=lookup_metric)[::-1]

        results_to_plot = sorted_results[0:max_num_curves]
        results_to_plot = sorted(results_to_plot, key=lookup_metric)[::-1]

        colors = kwplot.Color.distinct(len(results_to_plot))
        for idx, result in enumerate(results_to_plot):
            color = colors[idx]
            color = [kwplot.Color(color).as01()]
            measure = result.nocls_measures
            prefix = result.meta['prefix']
            color = hash_color(prefix)
            kw = {'fnum': fnum}
            if metric == 'ap':
                drawing.draw_prcurve(measure, prefix=prefix, color=color, **kw)
            elif metric == 'auc':
                drawing.draw_roc(measure, prefix=prefix, color=color, **kw)
            else:
                raise KeyError
        fig.gca().set_title(f'Comparison of runs {metric}: Salient -\n{dataset_title_part}')
        return fig

    fnum = 3
    catname = 'Active Construction'
    fig3 = plot_individual_class_curves(catname, fnum, 'ap')
    _writefig(fig3, out_dpath, f'{catname}_ap_curve.png', figsize, verbose, tight=True)

    fnum = 4
    catname = 'Site Preparation'
    fig4: mpl.figure.Figure = plot_individual_class_curves(catname, fnum, 'ap')
    _writefig(fig4, out_dpath, f'{catname}_ap_curve.png', figsize, verbose, tight=True)

    fnum = 5
    fig5 = plot_individual_salient_curves(fnum, metric='ap')
    _writefig(fig5, out_dpath, 'salient_ap_curve.png', figsize, verbose, tight=True)
    # print(best_per_expt.sort_values('mAP').to_string())

    # # if 1:
    #     # fig3.set_size_inches(np.array([6.4, 4.8]) * 2.0)
    #     # fig3.tight_layout()
    #     # fig4.set_size_inches(np.array([6.4, 4.8]) * 2.0)
    #     # fig4.tight_layout()
    #     # fig5.set_size_inches(np.array([5.4, 2.8]) * 2.0)
    #     # fig5.tight_layout()

    _ = best_candidates(class_rows, mean_rows)

    if config['show']:
        plt.show()


if __name__ == '__main__':
    """
    CommandLine:
        # DVC_DPATH=$(python -m watch.cli.find_dvc)
        DVC_DPATH=$HOME/data/dvc-repos/smart_watch_dvc
        MEASURE_GLOBSTR=$DVC_DPATH/models/fusion/SC-20201117/*_TA1*/*/*/eval/curves/measures2.json
        # ls $MEASURE_GLOBSTR
        # echo "$MEASURE_GLOBSTR"
        python -m watch.tasks.fusion.gather_results \
            --measure_globstr="$MEASURE_GLOBSTR" \
            --out_dpath="$DVC_DPATH/agg_results"
    """
    gather_measures(cmdline=True)
