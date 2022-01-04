"""
Helper for scheduling a set of prediction + evaluation jobs

python -m watch.tasks.fusion.schedule_inference schedule_evaluation
"""
import pathlib
import ubelt as ub
import itertools as it
import stat
import os


class TMUXQueue(ub.NiceRepr):
    """
    A lightweight, but limited job queue

    Example:
        >>> TMUXQueue('foo', 'foo')
    """
    def __init__(self, name, dpath, environ=None):
        self.name = name
        self.environ = environ
        self.dpath = pathlib.Path(dpath)
        self.fpath = self.dpath / (name + '.sh')
        self.header = ['#!/bin/bash']
        self.commands = []

    def __nice__(self):
        return f'{self.name} - {len(self.commands)}'

    def finalize_text(self):
        script = self.header
        if self.environ:
            script = script + [
                f'export {k}="{v}"' for k, v in self.environ.items()]
        script = script + self.commands
        text = '\n'.join(script)
        return text

    def submit(self, command):
        self.commands.append(command)

    def write(self):
        text = self.finalize_text()
        with open(self.fpath, 'w') as file:
            file.write(text)
        os.chmod(self.fpath, (
            stat.S_IXUSR | stat.S_IXGRP | stat.S_IRUSR |
            stat.S_IWUSR | stat.S_IRGRP | stat.S_IWGRP))
        return self.fpath


class TMUXMultiQueue(ub.NiceRepr):
    """
    Create multiple sets of jobs to start in detatched tmux sessions

    Example:
        >>> from watch.tasks.fusion.schedule_inference import *  # NOQA
        >>> self = TMUXMultiQueue('foo', 2)
        >>> print('self = {!r}'.format(self))
        >>> self.submit('echo hello')
        >>> self.submit('echo world')
        >>> self.submit('echo foo')
        >>> self.submit('echo bar')
        >>> self.submit('echo bazbiz')
        >>> self.write()
        >>> self.rprint()
    """
    def __init__(self, name, size=1, environ=None, dpath=None, gres=None):
        if dpath is None:
            dpath = ub.ensure_app_cache_dir('watch/tmux_queue')
        self.dpath = pathlib.Path(dpath)
        self.name = name
        self.size = size
        self.environ = environ
        self.fpath = self.dpath / f'run_queues_{self.name}.sh'

        per_worker_environs = [environ] * size
        if gres:
            # TODO: more sophisticated GPU policy?
            per_worker_environs = [
                ub.dict_union(e, {
                    'CUDA_VISIBLE_DEVICES': str(cvd),
                })
                for cvd, e in zip(gres, per_worker_environs)]

        self.workers = [
            TMUXQueue(
                name='queue_{}_{}'.format(self.name, worker_idx),
                dpath=self.dpath,
                environ=e
            )
            for worker_idx, e in enumerate(per_worker_environs)
        ]
        self._worker_cycle = it.cycle(self.workers)

    def __nice__(self):
        return ub.repr2(self.workers)

    def __iter__(self):
        yield from self._worker_cycle

    def submit(self, command):
        return next(self._worker_cycle).submit(command)

    def finalize_text(self):
        # Create a driver script
        driver_lines = [ub.codeblock(
            '''
            #!/bin/bash
            # Driver script to start the tmux-queue
            echo "submitting jobs"
            ''')]
        for queue in self.workers:
            # run_command_in_tmux_queue(command, name)
            part = ub.codeblock(
                f'''
                ### Run Queue: {queue.name}
                tmux new-session -d -s {queue.name} "bash"
                tmux send -t {queue.name} "source {queue.fpath}" Enter
                ''').format()
            driver_lines.append(part)
        driver_lines += ['echo "jobs submitted"']
        driver_text = '\n\n'.join(driver_lines)
        return driver_text

    def write(self):
        text = self.finalize_text()
        for queue in self.workers:
            queue.write()
        with open(self.fpath, 'w') as file:
            file.write(text)
        os.chmod(self.fpath, (
            stat.S_IXUSR | stat.S_IXGRP | stat.S_IRUSR |
            stat.S_IWUSR | stat.S_IRGRP | stat.S_IWGRP))
        return self.fpath

    def run(self):
        return ub.cmd(f'bash {self.fpath}', verbose=3, check=True)

    def rprint(self):
        from rich.panel import Panel
        from rich.syntax import Syntax
        from rich.console import Console
        console = Console()
        for queue in self.workers:
            code = queue.finalize_text()
            console.print(Panel(Syntax(code, 'bash'), title=str(queue.fpath)))
        code = self.finalize_text()
        console.print(Panel(Syntax(code, 'bash'), title=str(self.fpath)))


def schedule_evaluation(model_globstr=None, test_dataset=None, gpus='auto', run=False):
    """
    First ensure that models have been copied to the DVC repo in the
    appropriate path. (as noted by model_dpath)

    DVC_DPATH=$HOME/data/dvc-repos/smart_watch_dvc
    KWCOCO_TEST_FPATH=$DVC_DPATH/Drop1-Aligned-L1/combo_vali_nowv.kwcoco.json

    kwcoco subset $DVC_DPATH/Drop1-Aligned-L1/combo_vali_nowv.kwcoco.json
    smartwatch stats --src $DVC_DPATH/Drop1-Aligned-L1/combo_train_nowv.kwcoco.json

    # Hack to test on the train set for a sanity check
    kwcoco subset --src $DVC_DPATH/Drop1-Aligned-L1/combo_train_nowv.kwcoco.json \
            --dst $DVC_DPATH/Drop1-Aligned-L1/combo_train_US_R001_small_nowv.kwcoco.json \
            --select_videos '.name | startswith("US_R001")' \
            --select_images '.id % 4 == 0'
    smartwatch stats $DVC_DPATH/Drop1-Aligned-L1/combo_train_US_R001_small_nowv.kwcoco.json

    MODEL_GLOB=$DVC_DPATH/'models/fusion/SC-20201117/*/*.pt'
    echo "$MODEL_GLOB"

    cd $DVC_DPATH
    dvc pull -r aws --recursive models/fusion/SC-20201117

        --model_globstr="$MODEL_GLOB"
        --test_dataset="$KWCOCO_TEST_FPATH"

    Ignore:

    TODO:
        - [ ] Specify the model_dpath as an arg
        - [ ] Specify target dataset as an argument
        - [ ] Skip models that were already evaluated
    """
    import watch
    from watch.tasks.fusion import organize
    import json
    import ubelt as ub

    if model_globstr is None and test_dataset is None:
        dvc_dpath = watch.utils.util_data.find_smart_dvc_dpath()
        model_globstr = str(dvc_dpath / 'models/fusion/SC-20201117/*/*.pt')
        test_dataset = dvc_dpath / 'Drop1-Aligned-L1/combo_vali_nowv.kwcoco.json'

        # hack for train set
        # test_dataset = dvc_dpath / 'Drop1-Aligned-L1/combo_train_US_R001_small_nowv.kwcoco.json'
        gpus = 'auto'

    dvc_dpath = watch.utils.util_data.find_smart_dvc_dpath()

    # with_saliency = 'auto'
    # with_class = 'auto'
    with_saliency = 'auto'
    with_class = 'auto'

    with_pred = True  # TODO: allow caching
    with_eval = True
    workers_per_queue = 5
    recompute = False

    # HARD CODED
    # model_dpath = dvc_dpath / 'models/fusion/unevaluated-activity-2021-11-12'
    # test_dataset_fpath = dvc_dpath / 'Drop1-Aligned-L1/vali_combo11.kwcoco.json'

    # model_dpath = dvc_dpath / 'models/fusion/unevaluated-activity-2021-11-12'
    # model_dpath = dvc_dpath / 'models/fusion/SC-20201117'
    # test_dataset_fpath = dvc_dpath / 'Drop1-Aligned-L1/combo_vali_nowv.kwcoco.json'
    test_dataset_fpath = pathlib.Path(test_dataset)
    assert test_dataset_fpath.exists()

    stamp = ub.timestamp() + '_' + ub.hash_data([])[0:8]

    def package_metadata(package_fpath):
        # Hack for choosing one model from this "type"
        epoch_num = int(package_fpath.name.split('epoch=')[1].split('-')[0])
        expt_name = package_fpath.name.split('_epoch')[0]
        info = {
            'name': expt_name,
            'epoch': epoch_num,
            'fpath': package_fpath,
        }
        return info

    packages_to_eval = []
    import glob
    for package_fpath in glob.glob(model_globstr, recursive=True):
        package_info = package_metadata(pathlib.Path(package_fpath))
        packages_to_eval.append(package_info)

    shuffle_jobs = True
    if shuffle_jobs:
        import kwarray
        packages_to_eval = kwarray.shuffle(packages_to_eval)

    # # for subfolder in model_dpath.glob('*'):
    #     # package_fpaths = list(subfolder.glob('*.pt'))
    #     subfolder_infos = [package_metadata(package_fpath)
    #                        for package_fpath in package_fpaths]
    #     subfolder_infos = sorted(subfolder_infos, key=lambda x: x['epoch'], reverse=True)
    #     for info in subfolder_infos:
    #         if 'rutgers_v5' in info['name']:
    #             break
    #         packages_to_eval.append(info)
    #         # break

    tmux_schedule_dpath = dvc_dpath / '_tmp_tmux_schedule'
    tmux_schedule_dpath.mkdir(exist_ok=True)

    if gpus == 'auto':
        # Use all unused gpus
        import netharn as nh
        GPUS = []
        for gpu_idx, gpu_info in nh.device.gpu_info().items():
            if len(gpu_info['procs']) == 0:
                GPUS.append(gpu_idx)

    # GPUS = [0, 1, 2, 3]
    # GPUS = [0]
    environ = {
        'DVC_DPATH': dvc_dpath,
    }

    jobs = TMUXMultiQueue(name=stamp, size=len(GPUS), environ=environ,
                          gres=GPUS, dpath=tmux_schedule_dpath)
    for info, queue in zip(packages_to_eval, jobs):
        package_fpath = info['fpath']
        suggestions = organize.suggest_paths(package_fpath=package_fpath, test_dataset=test_dataset_fpath)
        suggestions = json.loads(suggestions)

        pred_dataset_fpath = pathlib.Path(suggestions['pred_dataset'])
        eval_metrics_fpath = pathlib.Path(suggestions['eval_dpath']) / 'curves/measures2.json'

        suggestions['eval_metrics'] = eval_metrics_fpath
        suggestions['test_dataset'] = test_dataset_fpath
        suggestions['true_dataset'] = test_dataset_fpath
        suggestions['package_fpath'] = package_fpath
        suggestions['with_class'] = with_class
        suggestions['with_saliency'] = with_saliency
        suggestions = ub.map_vals(lambda x: str(x).replace(
            str(dvc_dpath), '$DVC_DPATH'), suggestions)
        predictkw = {
            'workers_per_queue': workers_per_queue,
        }

        if with_pred:
            pred_command = ub.codeblock(
                r'''
                python -m watch.tasks.fusion.predict \
                    --write_probs=True \
                    --write_preds=False \
                    --with_class={with_class} \
                    --with_saliency={with_saliency} \
                    --with_change=False \
                    --package_fpath={package_fpath} \
                    --pred_dataset={pred_dataset} \
                    --test_dataset={test_dataset} \
                    --num_workers=5 \
                    --compress=DEFLATE \
                    --gpus=0 \
                    --batch_size=1
                ''').format(**suggestions, **predictkw)
            if not recompute:
                # Only run the command if its expected output does not exist
                pred_command = (
                    '[[ -f "{pred_dataset}" ]] || '.format(**suggestions) +
                    pred_command
                )

            if recompute or not pred_dataset_fpath.exists():
                queue.submit(pred_command)

        if with_eval:
            eval_command = ub.codeblock(
                r'''
                python -m watch.tasks.fusion.evaluate \
                    --true_dataset={true_dataset} \
                    --pred_dataset={pred_dataset} \
                      --eval_dpath={eval_dpath}
                ''').format(**suggestions)
            if not recompute:
                # TODO: use a real stamp file
                # Only run the command if its expected output does not exist
                eval_command = (
                    '[[ -f "{eval_metrics}" ]] || '.format(**suggestions) +
                    eval_command
                )
            if recompute or not eval_metrics_fpath.exists():
                queue.submit(eval_command)

    jobs.rprint()

    driver_fpath = jobs.write()
    # RUN
    if run:
        ub.cmd('bash ' + str(driver_fpath), verbose=3, check=True)
    else:
        print('Wrote script: to run execute:\n{}'.format(driver_fpath))

    """
    # Now postprocess script:

    python ~/code/watch/watch/tasks/fusion/organize.py make_eval_symlinks

    ls models/fusion/unevaluated-activity-2021-11-12/eval_links

    cd /home/joncrall/remote/horologic/smart_watch_dvc

    ARR=($(ls -a1 models/fusion/unevaluated-activity-2021-11-12/eval_links/*/curves/measures2.json))
    for ARG in "${ARR[@]}"; do
        echo "ARG = $ARG"
        cat "$ARG" | jq '.ovr_measures[] | with_entries(select(.key | in({"node":1, "ap":1, "auc": 1})))'
    done
    print('MEASURE_FPATHS = {!r}'.format(MEASURE_FPATHS))
    feh models/fusion/unevaluated-activity-2021-11-12/eval_links/*/curves/ovr_roc.png
    feh models/fusion/unevaluated-activity-2021-11-12/eval_links/*/curves/ovr_ap.png
    """


def gather_measures():
    import watch
    import json
    import pandas as pd
    import numpy as np
    import ubelt as ub
    from kwcoco.coco_evaluator import CocoSingleResult
    import yaml
    from watch.tasks.fusion import result_analysis
    import shutil
    # import pathlib

    class Found(Exception):
        pass

    dvc_dpath = watch.utils.util_data.find_smart_dvc_dpath()

    if True:
        # Prefer a remote (the machine where data is being evaluated)
        remote = 'horologic'
        remote = 'namek'
        # Hack for pointing at a remote
        remote_dpath = pathlib.Path(ub.shrinkuser(dvc_dpath, home=ub.expandpath(f'$HOME/remote/{remote}')))
        if remote_dpath.exists():
            dvc_dpath = remote_dpath

    # model_dpath = pathlib.Path(dvc_dpath) / 'models/fusion/unevaluated-activity-2021-11-12'
    fusion_model_dpath = dvc_dpath / 'models/fusion/'
    print(ub.repr2(list(fusion_model_dpath.glob('*'))))
    # model_dpath = fusion_model_dpath / 'unevaluated-activity-2021-11-12'
    model_dpath = fusion_model_dpath / 'SC-20201117'

    # measure_fpaths = list(model_dpath.glob('eval_links/*/curves/measures2.json'))
    measure_fpaths = list(model_dpath.glob('*/*/*/eval/curves/measures2.json'))

    dset_groups = ub.group_items(measure_fpaths, lambda x: x.parent.parent.parent.name)
    # measure_fpaths = dset_groups['combo_train_US_R001_small_nowv.kwcoco']
    measure_fpaths = dset_groups['combo_vali_nowv.kwcoco']

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
                cand_remote_home = pathlib.Path(hack_home)
                tmp = pathlib.Path(package_fpath)
                possible_home_dirs = [
                    pathlib.Path('/home/local/KHQ'),
                    pathlib.Path('/home'),
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


# def run_command_in_tmux_queue(command, name):
#     """
#     Ignore:
#         # Start a new bash session
#         tmux new-session -d -s test-bash-session "bash"
#         # Send that session a command
#         tmux send -t test-bash-session "ls" Enter
#     """
#     import ubelt as ub
#     info1 = ub.cmd('tmux new-session -d -s {} "bash"'.format(name), check=True, verbose=2)
#     info2 = ub.cmd('tmux send -t {} "{}" Enter'.format(name, command), verbose=2, check=True)

#     # # First start a new bash session
#     # session_id = 'fds'
#     # 'tmux new-session -d -s {session_id} "bash" Enter'


if __name__ == '__main__':
    """
    CommandLine:
        python ~/code/watch/watch/tasks/fusion/schedule_inference.py schedule_evaluation

        python ~/code/watch/watch/tasks/fusion/organize.py make_nice_dirs
        python ~/code/watch/watch/tasks/fusion/organize.py make_eval_symlinks
        python ~/code/watch/watch/tasks/fusion/organize.py make_pred_symlinks

        python ~/code/watch/watch/tasks/fusion/schedule_inference.py gather_measures
    """
    import fire
    fire.Fire()
