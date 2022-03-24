"""
Helper script for packaging a checkpoint into a torch package
"""
import os
import ubelt as ub


@ub.memoize
def load_meta(fpath):
    import yaml
    with open(fpath, 'r') as file:
        data = yaml.safe_load(file)
    return data


def repackage(checkpoint_fpath, force=False, dry=False):
    """

    checkpoint_fpath

    Ignore:
        >>> import ubelt as ub
        >>> checkpoint_fpath = ub.expandpath(
        ...     '/home/joncrall/data/dvc-repos/smart_watch_dvc/models/fusion/checkpoint_DirectCD_smt_it_joint_p8_raw9common_v5_tune_from_onera_epoch=2-step=2147.ckpt')

    checkpoint_fpath = '/home/joncrall/remote/namek/smart_watch_dvc/training/namek/joncrall/Drop1_October2021/runs/Saliency_smt_it_joint_p8_rgb_uconn_ukyshared_v001/lightning_logs/version_1/checkpoints/epoch=53-step=28457.ckpt'
    """
    # import os
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
            # Look at the training config file to get info about this
            # experiment
            candidates = list(train_dpath_hint.glob('fit_config.yaml'))
            if len(candidates) == 1:
                meta_fpath = candidates[0]
                data = load_meta(meta_fpath)
                if 'name' in data:
                    # Use the metadata package name if it exists
                    expt_name = data['name']
                else:
                    # otherwise, hack to put experiment name in package name
                    # based on an assumed directory structure
                    expt_name = ub.Path(data['default_root_dir']).name
                if expt_name not in package_name:
                    package_name = expt_name + '_' + package_name

        package_fpath = x.parent / package_name

        if force or not package_fpath.exists():
            if not dry:
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
                       mode='list', dvc_remote='aws'):
    """
    Package and copy checkpoints into the DVC folder for evaluation.

    Args:
        mode (str): can be list, repackage, copy, dvc-commit, or commit

    Ignore:
        from watch.tasks.fusion.repackage import *  # NOQA
        import xdev
        globals().update(xdev.get_func_kwargs(gather_checkpoints))
        commit = True

    CommandLine:
        DVC_DPATH=$HOME/data/dvc-repos/smart_watch_dvc
        python -m watch.tasks.fusion.repackage gather_checkpoints \
            --dvc_dpath=$DVC_DPATH \
            --storage_dpath=$DVC_DPATH/models/fusion/SC-20201117 \
            --train_dpath=$DVC_DPATH/training/$HOSTNAME/$USER/Drop2-Aligned-TA1-2022-02-15/runs/* \
            --mode=copy

        DVC_DPATH=$HOME/data/dvc-repos/smart_watch_dvc
        python -m watch.tasks.fusion.repackage gather_checkpoints \
            --dvc_dpath=$DVC_DPATH \
            --storage_dpath=$DVC_DPATH/models/fusion/eval3_candidates/packages \
            --train_dpath="$DVC_DPATH/training/$HOSTNAME/$USER/Drop2-Aligned-TA1-2022-02-15/runs/*" \
            --mode=list
    """
    from watch.utils import util_data
    from watch.utils import util_path
    import shutil

    if dvc_dpath is None:
        dvc_dpath = util_data.find_smart_dvc_dpath()
    else:
        dvc_dpath = ub.Path(dvc_dpath)

    # storage_dpath = dvc_dpath / 'models/fusion/unevaluated-activity-2021-11-12'
    # if storage_dpath is None:
    #     storage_dpath = dvc_dpath / 'models/fusion/SC-20201117'
    # else:
    storage_dpath = ub.Path(storage_dpath)

    # if train_dpath is None:
    #     train_dpath = [
    #         dvc_dpath / 'training/*/*/Drop1-20201117'
    #     ]

    # dset_dpaths = util_path.coerce_patterned_paths(train_dpath)
    # dset_dpaths = [ub.Path(p) for p in dset_dpaths]
    # # all_checkpoint_paths = [p / 'runs/*/lightning_logs/' for p in dset_dpaths]
    # all_checkpoint_paths = dset_dpaths

    lightning_log_dpaths = util_path.coerce_patterned_paths(train_dpath)
    lightning_log_dpaths = [ub.Path(p) for p in lightning_log_dpaths]
    print('lightning_log_dpaths = {}'.format(ub.repr2(lightning_log_dpaths, nl=1)))

    # for p in lightning_log_dpaths:
    #     pass

    # Collect checkpoints from the training path
    gathered = []
    for dpath in lightning_log_dpaths:
        # Hack to allow the user to specify the regular training path or the
        # lightning logs dirs themselves
        if dpath.stem == 'lightning_logs':
            checkpoint_fpaths = util_path.coerce_patterned_paths(
                dpath / '*/checkpoints/*.ckpt')
        elif dpath.parent.stem == 'lightning_logs':
            checkpoint_fpaths = util_path.coerce_patterned_paths(
                dpath / 'checkpoints/*.ckpt')
        else:
            ll_dpath = dpath / 'lightning_logs'
            if ll_dpath.exists():
                checkpoint_fpaths = util_path.coerce_patterned_paths(
                    dpath / 'lightning_logs/*/checkpoints/*.ckpt')
            else:
                checkpoint_fpaths = util_path.coerce_patterned_paths(
                    dpath / '*.ckpt')

        if not checkpoint_fpaths:
            checkpoint_fpaths = util_path.coerce_patterned_paths(dpath)
        print('checkpoint_fpaths = {}'.format(ub.repr2(checkpoint_fpaths, nl=1)))

        for checkpoint_fpath in checkpoint_fpaths:
            checkpoint_fpath = ub.Path(checkpoint_fpath)
            parts = checkpoint_fpath.name.split('-')
            epoch = int(parts[0].split('epoch=')[1])
            # print('checkpoint_fpath = {!r}'.format(checkpoint_fpath))
            # print('parts = {!r}'.format(parts))
            # print('epoch = {!r}'.format(epoch))
            # Dont add the -v2 versions
            if epoch >= 0:  # and parts[-1].startswith('step='):
                # print('checkpoint_fpath = {!r}'.format(checkpoint_fpath))
                gathered.append({
                    'epoch': epoch,
                    'checkpoint_fpath': checkpoint_fpath
                })

    for row in ub.ProgIter(gathered, desc='Gather checkpoint info'):
        p = row['checkpoint_fpath']
        package_fpath = repackage(str(p), dry=True)[0]
        package_fpath = ub.Path(package_fpath)
        name = package_fpath.name.split('_epoch')[0]
        name_dpath = storage_dpath / name
        name_fpath = name_dpath / package_fpath.name
        row['package_fpath'] = package_fpath
        row['name'] = name
        row['name_fpath'] = name_fpath
        row['needs_repackage'] = not package_fpath.exists()
        row['needs_copy'] = not name_fpath.exists()
        row['failed_repackage'] = 0
        # name_dpath.ensuredir()
        # print('package_fpath = {!r}'.format(package_fpath))
        # print('name_fpath = {!r}'.format(name_fpath))

    import rich
    if 1:
        import pandas as pd
        df = pd.DataFrame(gathered)
        rich.print('[blue] Gathered Data')
        if len(df) == 0:
            print(df)
            raise Exception('No data gathered')
        print(df.groupby('name')['needs_copy', 'needs_repackage'].sum())

    if mode == 'list':
        # import xdev
        # xdev.embed()
        return

    to_repackage = [r for r in gathered if r['needs_repackage']]
    for row in ub.ProgIter(to_repackage, desc='repackage'):
        try:
            repackage(row['checkpoint_fpath'])[0]
        except Exception:
            row['failed_repackage'] = True

    if 1:
        import pandas as pd
        df = pd.DataFrame(gathered)
        rich.print('[blue] Repackaged')
        if len(df):
            print(df.groupby('name')[['failed_repackage']].sum())

    if mode == 'repackage':
        return

    storage_dpath.ensuredir()
    to_copy = [r for r in gathered if r['needs_copy']]

    staged_expt_fpaths = [r['name_fpath'] for r in gathered]

    # Find the unique directories we stage to DVC
    # staged_expt_dpaths = sorted({r['name_fpath'].parent for r in to_copy})
    # for dpath in staged_expt_dpaths:
    #     dpath.ensuredir()

    for row in ub.ProgIter(to_copy, desc='Copy packages to DVC dir'):
        row['name_fpath'].parent.ensuredir()
        shutil.copy(row['package_fpath'], row['name_fpath'])

    if mode == 'copy':
        return

    from watch.utils.simple_dvc import SimpleDVC
    dvc_api = SimpleDVC(dvc_dpath)
    # dvc_api.add(staged_expt_dpaths)
    dvc_api.add(staged_expt_fpaths)
    dvc_api.push(storage_dpath, remote=dvc_remote)

    if mode == 'dvc-commit':
        return

    git_info3 = ub.cmd('git commit -am "new models"', verbose=3, check=True, cwd=dvc_dpath)  # dangerous?
    assert git_info3['ret'] == 0
    git_info2 = ub.cmd('git push', verbose=3, check=True, cwd=dvc_dpath)
    assert git_info2['ret'] == 0

    if mode == 'commit':
        return

    rel_storage_dpath = storage_dpath.relative_to(dvc_dpath)

    print(ub.codeblock(
        f"""
        # On the evaluation remote you need to run something like:
        DVC_DPATH=$(python -m watch.cli.find_dvc)
        cd $DVC_DPATH
        git pull
        dvc pull -r aws --recursive models/fusion/{rel_storage_dpath}

        python -m tasks.fusion.schedule_inference schedule_evaluation --gpus=auto --run=True
        """))

    # if 0:
    #     dvc_to_add = []
    #     for package_fpath in list(storage_dpath.glob('*/*.pt')):
    #         package_dvc_fpath = ub.Path(str(package_fpath) + '.dvc')
    #         if not package_dvc_fpath.exists() and package_fpath.is_file():
    #             dvc_to_add.append(str(package_fpath.relative_to(dvc_dpath)))
    #     # Broken parts
    #     print('New models to add to DVC: {}'.format(ub.repr2(dvc_to_add)))
    #     dvc_info = ub.cmd(['dvc', 'add'] + dvc_to_add, cwd=dvc_dpath, verbose=3, check=True)

    #     # Determine if DVC will autostage the new files
    #     # (It should for SMART)
    #     has_autostage = ub.cmd('dvc config core.autostage', cwd=dvc_dpath, check=True)['out'].split() == 'true'

    #     if not has_autostage:
    #         # Note: Using autostageadd means we do not need this, check
    #         # the setting and disable if necessary
    #         start = False
    #         gitlines = []
    #         for line in dvc_info['out'].split('\n'):
    #             if start:
    #                 gitlines.append(line.strip())
    #             if 'To track the changes with git, run:' in line:
    #                 start = True
    #         gitcmd = ''.join(gitlines)
    #         if gitcmd:
    #             git_info1 = ub.cmd(gitcmd, verbose=3, check=True, cwd=dvc_dpath)
    #             assert git_info1['ret'] == 0

    #     if git_commit:
    #         git_info3 = ub.cmd('git commit -am "new models"', verbose=3, check=True, cwd=dvc_dpath)  # dangerous?
    #         assert git_info3['ret'] == 0
    #         git_info2 = ub.cmd('git push', verbose=3, check=True, cwd=dvc_dpath)
    #         assert git_info2['ret'] == 0

    #     import dvc.main
    #     # from dvc import main
    #     saved_cwd = os.getcwd()
    #     try:
    #         os.chdir(dvc_dpath)
    #         remote = 'aws'
    #         dvc_command = ['push', '-r', remote, '--recursive', str(storage_dpath.relative_to(dvc_dpath))]
    #         dvc.main.main(dvc_command)
    #     finally:
    #         os.chdir(saved_cwd)

    #     print(ub.codeblock(
    #         """
    #         # On the evaluation remote you need to run something like:
    #         DVC_DPATH=$HOME/data/dvc-repos/smart_watch_dvc
    #         cd $DVC_DPATH
    #         git pull
    #         dvc pull -r aws --recursive models/fusion/SC-20201117

    #         python ~/code/watch/watch/tasks/fusion/schedule_inference.py schedule_evaluation --gpus=auto --run=True
    #         """))


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
