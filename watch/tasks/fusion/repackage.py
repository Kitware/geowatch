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
    import os
    import yaml
    # For now there is only one model, but in the future we will need
    # some sort of modal switch to package the correct metadata
    from watch.tasks.fusion import methods
    from watch.utils import util_path
    checkpoint_fpaths = util_path.coerce_patterned_paths(checkpoint_fpath)
    package_fpaths = []
    for checkpoint_fpath in checkpoint_fpaths:
        # If we have a checkpoint path we can load it if we make assumptions
        # init method from checkpoint.
        checkpoint_fpath = os.fspath(checkpoint_fpath)

        x = ub.Path(ub.augpath(checkpoint_fpath, ext='.pt'))
        package_name = x.name

        # Can we precompute the package name of this checkpoint?
        train_dpath_hint = None
        if checkpoint_fpath.endswith('.ckpt'):
            path_ = ub.Path(checkpoint_fpath)
            if path_.parent.stem == 'checkpoints':
                train_dpath_hint = path_.parent.parent

        if train_dpath_hint is not None:
            candidates = list(train_dpath_hint.glob('fit_config.yaml'))
            if len(candidates) == 1:
                meta_fpath = candidates[0]
                with open(meta_fpath, 'r') as file:
                    data = yaml.safe_load(file)
                # Hack to put experiment name in package name
                expt_name = ub.Path(data['default_root_dir']).name
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
        package_fpaths.append(str(package_fpath))
    return package_fpaths


def gather_checkpoints(dvc_dpath=None, storage_dpath=None, train_dpath=None,
                       git_commit=False):
    """
    Package and copy checkpoints into the DVC folder for evaluation.

    Ignore:
        from watch.tasks.fusion.repackage import *  # NOQA
        import xdev
        globals().update(xdev.get_func_kwargs(gather_checkpoints))
        git_commit = True

    CommandLine:
        DVC_DPATH=$HOME/data/dvc-repos/smart_watch_dvc
        python -m watch.tasks.fusion.repackage gather_checkpoints \
            --dvc_dpath=$DVC_DPATH \
            --storage_dpath=$DVC_DPATH/models/fusion/SC-20201117 \
            --train_dpath=$DVC_DPATH/training/$HOSTNAME/$USER/Drop1-20201117 \
            --git_commit=True

        DVC_DPATH=$HOME/data/dvc-repos/smart_watch_dvc
        python -m watch.tasks.fusion.repackage gather_checkpoints \
            --dvc_dpath=$DVC_DPATH \
            --storage_dpath=$DVC_DPATH/models/fusion/SC-20201117 \
            --train_dpath=$DVC_DPATH/training/$HOSTNAME/$USER/Drop1-20201117 \
            --git_commit=True
    """
    from watch.utils import util_data
    import ubelt as ub
    import shutil
    import os

    if dvc_dpath is None:
        dvc_dpath = util_data.find_smart_dvc_dpath()
    else:
        dvc_dpath = ub.Path(dvc_dpath)

    # storage_dpath = dvc_dpath / 'models/fusion/unevaluated-activity-2021-11-12'
    if storage_dpath is None:
        storage_dpath = dvc_dpath / 'models/fusion/SC-20201117'
    else:
        storage_dpath = ub.Path(storage_dpath)

    if train_dpath is None:
        dataset_names = [
            # 'Drop1_October2021',
            # 'Drop1_November2021',
            'Drop1-20201117',
        ]
        train_base = dvc_dpath / 'training'
        user_machine_dpaths = list(train_base.glob('*/*'))
        dset_dpaths = []
        for um_dpath in user_machine_dpaths:
            for dset_name in dataset_names:
                dset_dpath = um_dpath / dset_name
                dset_dpaths.append(dset_dpath)
    else:
        dset_dpaths = [ub.Path(train_dpath)]

    all_checkpoint_paths = []
    for dset_dpath in dset_dpaths:
        # Find all paths with lightning logs
        lightning_log_dpaths = list((dset_dpath / 'runs').glob('*/lightning_logs'))
        for ll_dpath in lightning_log_dpaths:
            if not ll_dpath.parent.name.startswith(('Activity', 'SC_', 'BOTH_', 'BAS_')):  # HACK
                continue
            checkpoint_fpaths = list((ll_dpath).glob('*/checkpoints/*.ckpt'))
            for checkpoint_fpath in checkpoint_fpaths:
                parts = checkpoint_fpath.name.split('-')
                epoch = int(parts[0].split('epoch=')[1])
                # Dont add the -v2 versions
                if epoch >= 0 and parts[-1].startswith('step='):
                    print('checkpoint_fpath = {!r}'.format(checkpoint_fpath))
                    all_checkpoint_paths.append(checkpoint_fpath)

    storage_dpath.ensuredir()

    to_copy = []
    failed = []
    for p in ub.ProgIter(all_checkpoint_paths):
        try:
            package_fpath = repackage(p)
            package_fpath = ub.Path(package_fpath)
            name = package_fpath.name.split('_epoch')[0]
            name_dpath = storage_dpath / name
            name_dpath.ensuredir()
            name_fpath = name_dpath / package_fpath.name
            print('package_fpath = {!r}'.format(package_fpath))
            print('name_fpath = {!r}'.format(name_fpath))
            if not name_fpath.exists():
                to_copy.append((package_fpath, name_dpath))
        except Exception as ex:
            print(f'Failed to repackage {p=} {ex=}')
            failed.append(p)

    print(f'failed to repackage = {failed=!r}')
    print(f'failed to repackage = {len(failed)=!r}')

    print(f'Copy {len(to_copy)} new checkpoints')
    for package_fpath, name_fpath in ub.ProgIter(to_copy):
        shutil.copy(package_fpath, name_fpath)

    dvc_to_add = []
    for package_fpath in list(storage_dpath.glob('*/*.pt')):
        package_dvc_fpath = ub.Path(str(package_fpath) + '.dvc')
        if not package_dvc_fpath.exists() and package_fpath.is_file():
            dvc_to_add.append(str(package_fpath.relative_to(dvc_dpath)))

    print('New models to add to DVC: {}'.format(ub.repr2(dvc_to_add)))
    dvc_info = ub.cmd(['dvc', 'add'] + dvc_to_add, cwd=dvc_dpath, verbose=3, check=True)

    # Determine if DVC will autostage the new files
    # (It should for SMART)
    has_autostage = ub.cmd('dvc config core.autostage', cwd=dvc_dpath, check=True)['out'].split() == 'true'

    if not has_autostage:
        # Note: Using autostageadd means we do not need this, check
        # the setting and disable if necessary
        start = False
        gitlines = []
        for line in dvc_info['out'].split('\n'):
            if start:
                gitlines.append(line.strip())
            if 'To track the changes with git, run:' in line:
                start = True
        gitcmd = ''.join(gitlines)
        if gitcmd:
            git_info1 = ub.cmd(gitcmd, verbose=3, check=True, cwd=dvc_dpath)
            assert git_info1['ret'] == 0

    if git_commit:
        git_info3 = ub.cmd('git commit -am "new models"', verbose=3, check=True, cwd=dvc_dpath)  # dangerous?
        assert git_info3['ret'] == 0
        git_info2 = ub.cmd('git push', verbose=3, check=True, cwd=dvc_dpath)
        assert git_info2['ret'] == 0

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

    print(ub.codeblock(
        """
        # On the evaluation remote you need to run something like:
        DVC_DPATH=$HOME/data/dvc-repos/smart_watch_dvc
        cd $DVC_DPATH
        git pull
        dvc pull -r aws --recursive models/fusion/SC-20201117

        python ~/code/watch/watch/tasks/fusion/schedule_inference.py schedule_evaluation --gpus=auto --run=True
        """))


if __name__ == '__main__':
    """
    CommandLine:
        ls $HOME/data/dvc-repos/smart_watch_dvc/training/namek/joncrall/Drop2-Aligned-TA1-2022-02-15/runs/BASELINE_EXPERIMENT_V001/lightning_logs/version_0/checkpoints/*.ckpt

        python -m watch.tasks.fusion.repackage

        python -m watch.tasks.fusion.repackage repackage "$HOME/data/dvc-repos/smart_watch_dvc/training/*/*/Drop1-20201117/runs/BAS_TA1_KOREA_v083/lightning_logs/*/checkpoints/*.ckpt"

        DVC_DPATH=$HOME/data/dvc-repos/smart_watch_dvc
        ls $DVC_DPATH/training/*/*/Drop1_October2021/runs/*/lightning_logs
    """
    import fire
    fire.Fire()
