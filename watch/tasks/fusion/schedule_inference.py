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
    stamp = ub.hash_data(ub.timestamp())[0:8]
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

        eval_command = ub.codeblock(
            r'''
            python -m watch.tasks.fusion.evaluate \
                    --true_dataset={true_dataset} \
                    --pred_dataset={pred_dataset} \
                      --eval_dpath={eval_dpath}
            ''').format(**suggestions).replace(str(dvc_dpath), '$DVC_DPATH')

        script['commands'].append(pred_command)
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

    # Start the jobs in a detatched temux session
    import stat
    import os
    for queue in parallel_queues.values():
        fpath = str(queue['fpath'])
        name = str(queue['name'])
        command = 'source {}'.format(fpath)
        run_command_in_tmux_queue(command, name)


def run_command_in_tmux_queue(command, name):
    """
    Ignore:
        # Start a new bash session
        tmux new-session -d -s test-bash-session "bash"
        # Send that session a command
        tmux send -t test-bash-session "ls" Enter
    """
    import ubelt as ub
    info1 = ub.cmd('tmux new-session -d -s {} "bash"'.format(name), check=True, verbose=2)
    info2 = ub.cmd('tmux send -t {} "{}" Enter'.format(name, command), verbose=2, check=True)

    # # First start a new bash session
    # session_id = 'fds'
    # 'tmux new-session -d -s {session_id} "bash" Enter'
