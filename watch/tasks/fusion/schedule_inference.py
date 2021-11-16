"""
Helper for scheduling a set of prediction + evaluation jobs
"""


def gather_candidate_models():
    import watch
    from watch.tasks.fusion import organize
    import json
    import ubelt as ub
    dvc_dpath = watch.utils.util_data.find_smart_dvc_dpath()

    # HARD CODED
    model_dpath = dvc_dpath / 'models/fusion/unevaluated-activity-2021-11-12'
    test_dataset_fpath = dvc_dpath / 'Drop1-Aligned-L1/vali_combo11.kwcoco.json'

    stamp = ub.timestamp() + '_' + ub.hash_data([])[0:8]

    with_pred = True  # TODO: allow caching
    with_eval = True

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
    for subfolder in model_dpath.glob('*'):
        package_fpaths = list(subfolder.glob('*.pt'))
        subfolder_infos = [package_metadata(package_fpath)
                           for package_fpath in package_fpaths]
        subfolder_infos = sorted(subfolder_infos, key=lambda x: x['epoch'], reverse=True)
        for info in subfolder_infos:
            if 'rutgers_v5' in info['name']:
                break
            packages_to_eval.append(info)
            break

    tmux_schedule_dpath = dvc_dpath / '_tmp_tmux_schedule'
    tmux_schedule_dpath.mkdir(exist_ok=True)

    GPUS = [0, 2, 3]
    parallel_job_ids = range(len(GPUS))
    parallel_queues = {
        queue_index: {
            'name': 'queue_{}_{}'.format(stamp, queue_index),
            'index': queue_index,
            'gpu_index': GPUS[queue_index],
            'commands': [],
        }
        for queue_index in parallel_job_ids
    }

    for queue in parallel_queues.values():
        queue['fpath'] = tmux_schedule_dpath / (queue['name'] + '.sh')
        queue['commands'].append(
            '#!/bin/bash'
        )
        queue['commands'].append(
            'export DVC_DPATH="{}"'.format(str(dvc_dpath))
        )
        queue['commands'].append(
            'export CUDA_VISIBLE_DEVICES={}'.format(str(queue['gpu_index']))
        )
    import itertools as it
    script_cycle = it.cycle(parallel_queues.values())
    for idx, info in enumerate(packages_to_eval):
        script = next(script_cycle)

        package_fpath = info['fpath']
        suggestions = organize.suggest_paths(package_fpath=package_fpath, test_dataset=test_dataset_fpath)
        suggestions = json.loads(suggestions)
        suggestions['test_dataset'] = test_dataset_fpath
        suggestions['true_dataset'] = test_dataset_fpath
        suggestions['package_fpath'] = package_fpath

        if with_pred:
            pred_command = ub.codeblock(
                r'''
                python -m watch.tasks.fusion.predict \
                    --write_probs=True \
                    --write_preds=False \
                    --with_class=True \
                    --with_saliency=False \
                    --with_change=False \
                    --package_fpath={package_fpath} \
                    --pred_dataset={pred_dataset} \
                    --test_dataset={test_dataset} \
                    --num_workers=5 \
                    --gpus=0 \
                    --batch_size=1
                ''').format(**suggestions).replace(str(dvc_dpath), '$DVC_DPATH')
            script['commands'].append(pred_command)

        if with_eval:
            eval_command = ub.codeblock(
                r'''
                python -m watch.tasks.fusion.evaluate \
                        --true_dataset={true_dataset} \
                        --pred_dataset={pred_dataset} \
                          --eval_dpath={eval_dpath}
                ''').format(**suggestions).replace(str(dvc_dpath), '$DVC_DPATH')
            script['commands'].append(eval_command)

    import stat
    import os
    for queue in parallel_queues.values():
        fpath = queue['fpath']
        text = '\n\n'.join(queue['commands'])
        with open(fpath, 'w') as file:
            file.write(text)
        print(text)
        os.chmod(fpath, stat.S_IXUSR | stat.S_IXGRP | stat.S_IRUSR | stat.S_IWUSR | stat.S_IRGRP | stat.S_IWGRP)

    # Create a driver script

    # Start the jobs in a detatched temux session
    driver_lines = ['#!/bin/bash']
    driver_lines = ['# Driver script to start the tmux-queue']
    driver_lines = ['echo "submitting jobs"']
    import stat
    import os
    for queue in parallel_queues.values():
        fpath = str(queue['fpath'])
        name = str(queue['name'])
        command = 'source {}'.format(fpath)
        # run_command_in_tmux_queue(command, name)
        part = ub.codeblock(
            '''
            ### Run Queue: {name}
            tmux new-session -d -s {name} "bash"
            tmux send -t {name} "{command}" Enter
            ''').format(name=name, command=command)
        driver_lines.append(part)
    driver_lines = ['echo "jobs submitted"']

    driver_text = '\n\n'.join(driver_lines)
    driver_fpath = tmux_schedule_dpath / 'run_queues_{}'.format(stamp)
    with open(driver_fpath, 'w') as file:
        file.write(driver_text)
    os.chmod(driver_fpath, stat.S_IXUSR | stat.S_IXGRP | stat.S_IRUSR | stat.S_IWUSR | stat.S_IRGRP | stat.S_IWGRP)

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
    dvc_dpath = watch.utils.util_data.find_smart_dvc_dpath()
    model_dpath = dvc_dpath / 'models/fusion/unevaluated-activity-2021-11-12'

    class_rows = []
    mean_rows = []
    for measure_fpath in list(model_dpath.glob('eval_links/*/curves/measures2.json')):
        with open(measure_fpath, 'r') as file:
            info = json.load(file)

        class_aps = []
        class_aucs = []
        for catname, bin_measure in info['ovr_measures'].items():
            class_aps.append(bin_measure['ap'])
            class_aucs.append(bin_measure['auc'])
            row = {}
            row['AP'] = bin_measure['ap']
            row['AUC'] = bin_measure['auc']
            row['catname'] = catname
            row['title'] = info['meta']['title']
            class_rows.append(row)

        row = {}
        row['mAP'] = np.nanmean(class_aps)
        row['mAUC'] = np.nanmean(class_aucs)
        row['catname'] = 'all'
        row['title'] = info['meta']['title']
        mean_rows.append(row)

    df = pd.DataFrame(mean_rows)
    print('Sort by mAP')
    print(df.sort_values('mAP').to_string())

    print('Sort by mAUC')
    print(df.sort_values('mAUC').to_string())

    df = pd.DataFrame(class_rows)
    print('Sort by AP')
    print(df.sort_values('AP').to_string())

    print('Sort by AUC')
    print(df.sort_values('AUC').to_string())


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
        python ~/code/watch/watch/tasks/fusion/schedule_inference.py gather_candidate_models

        python ~/code/watch/watch/tasks/fusion/organize.py make_nice_dirs
        python ~/code/watch/watch/tasks/fusion/organize.py make_eval_symlinks
        python ~/code/watch/watch/tasks/fusion/organize.py make_pred_symlinks

        python ~/code/watch/watch/tasks/fusion/schedule_inference.py gather_measures
    """
    import fire
    fire.Fire()
