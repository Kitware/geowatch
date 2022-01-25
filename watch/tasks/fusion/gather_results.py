"""
Loads and summarizes pre-computed metrics over multiple experiments
"""


def gather_measures():
    import watch
    import json
    import pandas as pd
    import numpy as np
    import ubelt as ub
    from kwcoco.coco_evaluator import CocoSingleResult
    import yaml
    from watch.utils import result_analysis
    import shutil

    # TODO: high level results for a model should be serialized to DVC

    class Found(Exception):
        pass

    dvc_dpath = watch.utils.util_data.find_smart_dvc_dpath()

    if True:
        # Prefer a remote (the machine where data is being evaluated)
        remote = 'horologic'
        remote = 'namek'
        remote = 'toothrush'
        # Hack for pointing at a remote
        remote_dpath = ub.Path(ub.shrinkuser(dvc_dpath, home=ub.expandpath(f'$HOME/remote/{remote}')))
        if remote_dpath.exists():
            dvc_dpath = remote_dpath

    # model_dpath = ub.Path(dvc_dpath) / 'models/fusion/unevaluated-activity-2021-11-12'
    fusion_model_dpath = dvc_dpath / 'models/fusion/'
    print(ub.repr2(list(fusion_model_dpath.glob('*'))))
    # model_dpath = fusion_model_dpath / 'unevaluated-activity-2021-11-12'
    model_dpath = fusion_model_dpath / 'SC-20201117'

    # measure_fpaths = list(model_dpath.glob('eval_links/*/curves/measures2.json'))
    measure_fpaths = list(model_dpath.glob('*/*/*/eval/curves/measures2.json'))

    dset_groups = ub.group_items(measure_fpaths, lambda x: x.parent.parent.parent.name)

    predict_group_freq = ub.map_vals(len, dset_groups)
    print('These are the different datasets prediction was run on.')
    print('TODO: need to choose exactly 1 or a compatible set of them')
    print('predict_group_freq = {}'.format(ub.repr2(predict_group_freq, nl=1)))

    # measure_fpaths = dset_groups['combo_train_US_R001_small_nowv.kwcoco']
    # measure_fpaths = dset_groups['combo_vali_nowv.kwcoco']
    measure_fpaths = dset_groups['combo_DILM_nowv_vali.kwcoco']
    print(len(measure_fpaths))
    # dataset_to_evals = ub.group_items(eval_dpaths, lambda x: x.parent.name)

    jobs = ub.JobPool('thread', max_workers=10)
    all_infos = []
    def load_data(measure_fpath):
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
    for measure_fpath in ub.ProgIter(measure_fpaths):
        jobs.submit(load_data, measure_fpath)

    # job = next(iter(jobs.as_completed(desc='collect jobs')))
    # all_infos = [job.result()]
    for job in jobs.as_completed(desc='collect jobs'):
        all_infos.append(job.result())

    class_rows = []
    mean_rows = []
    all_results = []
    results_list2 = []

    for info in ub.ProgIter(all_infos):
        result = CocoSingleResult.from_json(info)
        all_results.append(result)

        class_aps = []
        class_aucs = []

        title = info['meta']['title']

        # Hack to get the epoch/step/expt_name
        epoch = int(title.split('epoch=')[1].split('-')[0])
        step = int(title.split('step=')[1].split('-')[0])
        expt_name = title.split('epoch=')[0]

        expt_class_rows = []
        for catname, bin_measure in info['ovr_measures'].items():
            class_aps.append(bin_measure['ap'])
            class_aucs.append(bin_measure['auc'])
            class_row = {}
            class_row['AP'] = bin_measure['ap']
            class_row['AUC'] = bin_measure['auc']
            class_row['catname'] = catname
            class_row['title'] = title
            class_row['expt_name'] = expt_name
            class_row['epoch'] = epoch
            class_row['step'] = step
            expt_class_rows.append(class_row)
        class_rows.extend(expt_class_rows)

        row = {}
        row['mAP'] = np.nanmean(class_aps)
        row['mAUC'] = np.nanmean(class_aucs)
        row['catname'] = 'all'
        row['title'] = title
        row['expt_name'] = expt_name
        row['epoch'] = epoch
        row['step'] = step
        mean_rows.append(row)

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

            bin_measure = info['ovr_measures']
            metrics = {
                'map': row['mAP'],
                'mauc': row['mAUC'],
                'nocls_ap': info['nocls_measures']['ap'],
                'nocls_auc': info['nocls_measures']['auc'],
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

            predict_args  # add predict window overlap
            # row['train_remote'] = cand_remote

            result2 = result_analysis.Result(
                 name=result.meta['title'],
                 params=fit_config,
                 metrics=metrics,
            )
            results_list2.append(result2)

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
        'nocls_auc',
        'nocls_ap',
        # 'map',
        # 'mauc',
    }

    def shrink_notations(df):
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
        return df

    self = result_analysis.ResultAnalysis(
        results_list2, ignore_params=ignore_params,
        ignore_metrics=ignore_metrics,
    )
    self.analysis()
    stats_table = pd.DataFrame([ub.dict_diff(d, {'pairwise', 'param_values', 'moments'}) for d in self.statistics])
    stats_table = stats_table.sort_values('anova_rank_p')
    print(stats_table)

    mean_df = pd.DataFrame(mean_rows)
    mean_df = shrink_notations(mean_df)
    print('Sort by mAP')
    print(mean_df.sort_values('mAP').to_string())

    mean_df['title'].apply(lambda x: int(x.split('epoch=')[1].split('-')[0]))

    best_per_expt = pd.concat([subdf.loc[subdf[['mAP']].idxmax()] for t, subdf in mean_df.groupby('expt_name')])
    if True:
        best_per_expt = shrink_notations(best_per_expt)
        best_per_expt = best_per_expt.drop('title', axis=1)
        best_per_expt = best_per_expt.drop('catname', axis=1)
        best_per_expt = best_per_expt.drop('normalize_perframe', axis=1)
        best_per_expt = best_per_expt.drop('normalize_inputs', axis=1)
        best_per_expt = best_per_expt.drop('train_remote', axis=1)
        best_per_expt = best_per_expt.drop('step', axis=1)
        best_per_expt = best_per_expt.drop('arch_name', axis=1)
        best_per_expt = best_per_expt.rename({'time_steps': 'time', 'chip_size': 'space'}, axis=1)
    print(best_per_expt.sort_values('mAP').to_string())

    print('Sort by mAUC')
    print(mean_df[~mean_df['mAP'].isnull()].sort_values('mAUC').to_string())

    class_df = pd.DataFrame(class_rows)
    print('Sort by AP')
    print(class_df[~class_df['AP'].isnull()].sort_values('AP').to_string())

    print('Sort by AUC')
    print(class_df[~class_df['AUC'].isnull()].sort_values('AUC').to_string())

    import kwplot
    sns = kwplot.autosns()
    plt = kwplot.autoplt()  # NOQA

    kwplot.figure(fnum=1, doclf=True)
    ax = sns.lineplot(data=mean_df, x='epoch', y='mAP', hue='expt_name', marker='o', style='channels')
    h, ell = ax.get_legend_handles_labels()
    ax.legend(h, ell, loc='lower right')
    # ax.set_title('Pixelwise mAP AC metrics: KR_R001 + KR_R002')
    ax.set_title('Pixelwise mAP AC metrics')  # todo: add train name

    kwplot.figure(fnum=2, doclf=True)
    ax = sns.lineplot(data=mean_df, x='epoch', y='mAUC', hue='expt_name', marker='o', style='channels')
    ax.set_title('Pixelwise mAUC AC metrics: KR_R001 + KR_R002')

    max_num_curves = 16

    from kwcoco.metrics import drawing
    fig = kwplot.figure(fnum=3, doclf=True)
    catname = 'Active Construction'
    # catname = 'Site Preparation'

    sorted_results = sorted(all_results, key=lambda x: x.ovr_measures[catname]['ap'])[::-1]
    results_to_plot = sorted_results[0:max_num_curves]
    colors = kwplot.Color.distinct(len(results_to_plot))
    for idx, result in enumerate(results_to_plot):
        color = colors[idx]
        color = [kwplot.Color(color).as01()]
        measure = result.ovr_measures[catname]
        print(measure['ap'])
        prefix = result.meta['title']
        kw = {'fnum': 3}
        drawing.draw_prcurve(measure, prefix=prefix, color=color, **kw)
    fig.gca().set_title('Comparison of runs AP: {}'.format(catname))

    fig = kwplot.figure(fnum=4, doclf=True)
    # catname = 'Active Construction'
    sorted_results = sorted(all_results, key=lambda x: x.ovr_measures[catname]['auc'])[::-1]
    # catname = 'Site Preparation'
    results_to_plot = sorted_results[0:max_num_curves]
    colors = kwplot.Color.distinct(len(results_to_plot))
    for idx, result in enumerate(results_to_plot):
        color = colors[idx]
        color = [kwplot.Color(color).as01()]
        measure = result.ovr_measures[catname]
        prefix = result.meta['title']
        kw = {'fnum': 4}
        drawing.draw_roc(measure, prefix=prefix, color=color, **kw)
    ax = fig.gca()
    # ax.set_xlabel('fpr (false positive rate)')
    # ax.set_xlabel('tpr (true positive rate)')
    ax.set_title('Comparison of runs AUC: {}'.format(catname))

    print(best_per_expt.sort_values('mAP').to_string())

    if 1:
        plt.show()


if __name__ == '__main__':
    """
    CommandLine:
        python -m watch.tasks.fusion.gather_results
    """
    gather_measures()
    import xdoctest
    xdoctest.doctest_module(__file__)
