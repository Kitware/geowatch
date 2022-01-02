"""
Helper script for packaging a checkpoint into a torch package
"""


def repackage(checkpoint_fpath, force=False):
    """

    checkpoint_fpath

    Ignore:
        >>> import ubelt as ub
        >>> checkpoint_fpath = ub.expandpath(
        ...     '/home/joncrall/data/dvc-repos/smart_watch_dvc/models/fusion/checkpoint_DirectCD_smt_it_joint_p8_raw9common_v5_tune_from_onera_epoch=2-step=2147.ckpt')

    checkpoint_fpath = '/home/joncrall/remote/namek/smart_watch_dvc/training/namek/joncrall/Drop1_October2021/runs/Saliency_smt_it_joint_p8_rgb_uconn_ukyshared_v001/lightning_logs/version_1/checkpoints/epoch=53-step=28457.ckpt'

    """
    import ubelt as ub
    import pathlib
    import os
    import yaml
    # For now there is only one model, but in the future we will need
    # some sort of modal switch to package the correct metadata
    from watch.tasks.fusion import methods
    # If we have a checkpoint path we can load it if we make assumptions
    # init method from checkpoint.
    checkpoint_fpath = os.fspath(checkpoint_fpath)

    x = pathlib.Path(ub.augpath(checkpoint_fpath, ext='.pt'))
    package_name = x.name

    # Can we precompute the package name of this checkpoint?
    train_dpath_hint = None
    if checkpoint_fpath.endswith('.ckpt'):
        path_ = pathlib.Path(checkpoint_fpath)
        if path_.parent.stem == 'checkpoints':
            train_dpath_hint = path_.parent.parent

    if train_dpath_hint is not None:
        candidates = list(train_dpath_hint.glob('fit_config.yaml'))
        if len(candidates) == 1:
            meta_fpath = candidates[0]
            with open(meta_fpath, 'r') as file:
                data = yaml.safe_load(file)
            # Hack to put experiment name in package name
            expt_name = pathlib.Path(data['default_root_dir']).name
            if expt_name not in package_name:
                package_name = expt_name + '_' + package_name

    package_fpath = x.parent / package_name

    if force or not package_fpath.exists():
        import netharn as nh
        xpu = nh.XPU.coerce('cpu')
        checkpoint = xpu.load(checkpoint_fpath)

        # checkpoint = torch.load(checkpoint_fpath)
        print(list(checkpoint.keys()))
        hparams = checkpoint['hyper_parameters']
        if 'input_channels' in hparams:
            from kwcoco.channel_spec import ChannelSpec
            # Hack for strange pickle issue
            chan = hparams['input_channels']
            if not hasattr(chan, '_spec') and hasattr(chan, '_info'):
                chan = ChannelSpec.coerce(chan._info['spec'])
                hparams['input_channels'] = chan
            else:
                hparams['input_channels'] = ChannelSpec.coerce(chan.spec)

        method = methods.MultimodalTransformer(**hparams)
        state_dict = checkpoint['state_dict']
        method.load_state_dict(state_dict)

        if train_dpath_hint is not None:
            method.train_dpath_hint = train_dpath_hint

        method.save_package(str(package_fpath))
    return str(package_fpath)


def gather_checkpoints(dvc_dpath=None, storage_dpath=None, train_dpath=None):
    """
    Package and copy checkpoints into the DVC folder for evaluation.

    Ignore:
        from watch.tasks.fusion.repackage import *  # NOQA

    CommandLine:
        DVC_DPATH=$HOME/data/dvc-repos/smart_watch_dvc


    python -m watch.tasks.fusion.repackage gather_checkpoints \
        --dvc_dpath=$DVC_DPATH \
        --storage_dpath=$DVC_DPATH/models/fusion/SC-20201117 \
        --train_dpath=$DVC_DPATH/training/$HOSTNAME/$USER/Drop1-20201117

    """
    from watch.utils import util_data
    import pathlib
    import ubelt as ub
    import shutil
    import os

    if dvc_dpath is None:
        dvc_dpath = util_data.find_smart_dvc_dpath()

    # storage_dpath = dvc_dpath / 'models/fusion/unevaluated-activity-2021-11-12'
    if storage_dpath is None:
        storage_dpath = dvc_dpath / 'models/fusion/SC-20201117'

    if train_dpath is None:
        dataset_names = [
            # 'Drop1_October2021',
            # 'Drop1_November2021',
            'Drop1-20201117',
        ]
        train_base = dvc_dpath / 'training'
        user_machine_dpaths = list(train_base.glob('*/*'))
        all_checkpoint_paths = []
        dset_dpaths = []
        for um_dpath in user_machine_dpaths:
            for dset_name in dataset_names:
                dset_dpath = um_dpath / dset_name
                dset_dpaths.append(dset_dpath)
    else:
        dset_dpaths = [train_dpath]

    for dset_dpath in dset_dpaths:
        lightning_log_dpaths = list((dset_dpath / 'runs').glob('*/lightning_logs'))
        for ll_dpath in lightning_log_dpaths:
            if not ll_dpath.parent.name.startswith(('Activity', 'SC_')):  # HACK
                continue
            for checkpoint_fpath in list((ll_dpath).glob('*/checkpoints/*.ckpt')):
                parts = checkpoint_fpath.name.split('-')
                if int(parts[0].split('epoch=')[1]) > 2 and parts[-1].startswith('step='):
                    print('checkpoint_fpath = {!r}'.format(checkpoint_fpath))
                    all_checkpoint_paths.append(checkpoint_fpath)

    storage_dpath = pathlib.Path(storage_dpath)
    storage_dpath.mkdir(exist_ok=True, parents=True)

    to_copy = []
    for p in ub.ProgIter(all_checkpoint_paths):
        package_fpath = repackage(p)
        package_fpath = pathlib.Path(package_fpath)
        name = package_fpath.name.split('_epoch')[0]
        name_dpath = storage_dpath / name
        name_dpath.mkdir(exist_ok=True, parents=True)
        name_fpath = name_dpath / package_fpath.name
        if not name_fpath.exists():
            to_copy.append((package_fpath, name_dpath))
    print(f'Copy {len(to_copy)} new checkpoints')
    for package_fpath, name_fpath in ub.ProgIter(to_copy):
        shutil.copy(package_fpath, name_fpath)

    dvc_to_add = []
    for package_dpath in list(storage_dpath.glob('*/*.pt')):
        package_dvc_fpath = pathlib.Path(str(package_dpath) + '.dvc')
        if not package_dvc_fpath.exists():
            dvc_to_add.append(str(package_dpath.relative_to(dvc_dpath)))

    dvc_info = ub.cmd(['dvc', 'add'] + dvc_to_add, cwd=dvc_dpath, verbose=3, check=True)
    start = False
    gitlines = []
    for line in dvc_info['out'].split('\n'):
        if start:
            gitlines.append(line.strip())
        if 'To track the changes with git, run:' in line:
            start = True
    gitcmd = ''.join(gitlines)
    git_info1 = ub.cmd(gitcmd, verbose=3, check=True, cwd=dvc_dpath)
    assert git_info1['ret'] == 0

    # git_info3 = ub.cmd('git commit -am "new models"', verbose=3, check=True, cwd=dvc_dpath)  # dangerous?
    # assert git_info3['ret'] == 0
    # git_info2 = ub.cmd('git push', verbose=3, check=True, cwd=dvc_dpath)
    # assert git_info2['ret'] == 0

    import dvc.main
    # from dvc import main
    saved_cwd = os.getcwd()
    try:
        os.chdir(dvc_dpath)
        remote = 'aws'
        dvc_command = ['push', '-r', remote, '--recursive', str(storage_dpath.relative_to(dvc_dpath))]
        dvc.main.main(dvc_command)
    finally:
        os.chdir(saved_cwd)

    """
    # on remote
    DVC_DPATH=$HOME/data/dvc-repos/smart_watch_dvc
    cd $DVC_DPATH
    git pull
    dvc pull -r aws --recursive models/fusion/SC-20201117

    python ~/code/watch/watch/tasks/fusion/schedule_inference.py schedule_evaluation --gpus=None
    """


if __name__ == '__main__':
    """
    CommandLine:
        python ~/code/watch/watch/tasks/fusion/repackage.py /home/joncrall/data/dvc-repos/smart_watch_dvc/training/toothbrush/joncrall/Drop1_Raw_Holdout/runs/ActivityClf_smt_it_joint_p8_raw_v019/lightning_logs/version_0/checkpoints/epoch=9-step=2299.ckpt

        python -m watch.tasks.fusion.repackage /home/joncrall/remote/namek/smart_watch_dvc/training/namek/joncrall/Drop1_October2021/runs/Saliency_smt_it_joint_p8_rgb_uconn_ukyshared_v001/lightning_logs/version_1/checkpoints/epoch=53-step=28457.ckpt

        python -m watch.tasks.fusion.repackage /home/joncrall/remote/namek/smart_watch_dvc/training/namek/joncrall/Drop1_October2021/runs/Saliency_smt_it_joint_p8_rgb_uconn_ukyshared_v001/lightning_logs/version_1/checkpoints/epoch=98-step=52172.ckpt

        python -m watch.tasks.fusion.repackage /home/joncrall/remote/namek/smart_watch_dvc/training/namek/joncrall/Drop1_October2021/runs/Saliency_smt_it_joint_p8_rgb_uconn_ukyshared_v001/lightning_logs/version_1/checkpoints/epoch=21-step=11593.ckpt

        python -m watch.tasks.fusion.repackage /home/joncrall/remote/namek/smart_watch_dvc/training/namek/joncrall/Drop1_October2021/runs/Saliency_smt_it_joint_p8_rgb_uconn_ukyshared_v001/lightning_logs/version_1/checkpoints/epoch=31-step=16863.ckpt
        python -m watch.tasks.fusion.repackage /home/joncrall/remote/namek/smart_watch_dvc/training/namek/joncrall/Drop1_October2021/runs/Saliency_smt_it_joint_p8_rgb_uconn_ukyshared_v001/lightning_logs/version_1/checkpoints/epoch=43-step=23187.ckpt

        python -m watch.tasks.fusion.repackage /home/joncrall/remote/namek/smart_watch_dvc/training/namek/joncrall/Drop1_October2021/runs/Saliency_smt_it_joint_p8_raw_v001/lightning_logs/version_1/checkpoints/epoch=145-step=76941.ckpt


        DVC_DPATH=$HOME/data/dvc-repos/smart_watch_dvc
        ls $DVC_DPATH/training/*/*/Drop1_October2021/runs/*/lightning_logs



    """
    import fire
    fire.Fire(repackage)
