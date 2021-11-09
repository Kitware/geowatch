"""
Helper script for packaging a checkpoint into a torch package
"""


def repackage(checkpoint_fpath):
    """

    checkpoint_fpath

    Ignore:
        >>> import ubelt as ub
        >>> checkpoint_fpath = ub.expandpath(
        ...     '/home/joncrall/data/dvc-repos/smart_watch_dvc/models/fusion/checkpoint_DirectCD_smt_it_joint_p8_raw9common_v5_tune_from_onera_epoch=2-step=2147.ckpt')

    checkpoint_fpath = '/home/joncrall/remote/namek/smart_watch_dvc/training/namek/joncrall/Drop1_October2021/runs/Saliency_smt_it_joint_p8_rgb_uconn_ukyshared_v001/lightning_logs/version_1/checkpoints/epoch=53-step=28457.ckpt'


    """
    import torch
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
    checkpoint = torch.load(checkpoint_fpath)
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

    train_dpath_hint = None
    if checkpoint_fpath.endswith('.ckpt'):
        path_ = pathlib.Path(checkpoint_fpath)
        if path_.parent.stem == 'checkpoints':
            train_dpath_hint = path_.parent.parent
            method.train_dpath_hint = train_dpath_hint

    x = ub.augpath(checkpoint_fpath, ext='.pt')
    x = pathlib.Path(x)
    package_name = x.name
    # package_name = x.name.replace('checkpoint_', 'package_')

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

    package_fpath = str(x.parent / package_name)
    method.save_package(str(package_fpath))

    return package_fpath


def gather_checkpoints():
    """
    Hack function to move all checkpoints into a directory for evaluation
    """
    from watch.utils import util_data
    dvc_dpath = util_data.find_smart_dvc_dpath()
    train_base = dvc_dpath / 'training'
    dataset_names = [
        'Drop1_October2021',
        'Drop1_November2021',
    ]
    user_machine_dpaths = list(train_base.glob('*/*'))

    all_checkpoint_paths = []
    for um_dpath in user_machine_dpaths:
        for dset_name in dataset_names:
            dset_dpath = um_dpath / dset_name
            lightning_log_dpaths = list((dset_dpath / 'runs').glob('*/lightning_logs'))
            for ll_dpath in lightning_log_dpaths:
                for checkpoint_fpath in list((ll_dpath).glob('*/checkpoints/*.ckpt')):
                    parts = checkpoint_fpath.name.split('-')
                    if int(parts[0].split('epoch=')[1]) > 10 and parts[-1].startswith('step='):
                        print('checkpoint_fpath = {!r}'.format(checkpoint_fpath))
                        all_checkpoint_paths.append(checkpoint_fpath)

    unevaled_dpath = dvc_dpath / 'models/fusion/unevaluated'
    unevaled_dpath.mkdir(exist_ok=True, parents=True)

    import ubelt as ub
    import shutil
    for p in ub.ProgIter(all_checkpoint_paths):
        package_fpath = repackage(p)
        package_fpath = pathlib.Path(package_fpath)
        name = package_fpath.name.split('_epoch')[0]
        name_dpath = unevaled_dpath / name
        name_dpath.mkdir(exist_ok=True, parents=True)
        shutil.copy(package_fpath, name_dpath)


    import os
    for r, ds, fs in os.walk(train_base):
        if r.endswith('/lightning_logs'):
            print('r = {!r}'.format(r))
            break
        pass


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
