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
    """
    import torch
    # For now there is only one model, but in the future we will need
    # some sort of modal switch to package the correct metadata
    from watch.tasks.fusion import methods
    # If we have a checkpoint path we can load it if we make assumptions
    # init method from checkpoint.
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
    import ubelt as ub
    import pathlib
    x = ub.augpath(checkpoint_fpath, ext='.pt')
    x = pathlib.Path(x)
    package_fpath = str(x.parent / x.name.replace('checkpoint_', 'package_'))
    method.save_package(str(package_fpath))
    return package_fpath


if __name__ == '__main__':
    """
    CommandLine:
        python ~/code/watch/watch/tasks/fusion/repackage.py /home/joncrall/data/dvc-repos/smart_watch_dvc/training/toothbrush/joncrall/Drop1_Raw_Holdout/runs/ActivityClf_smt_it_joint_p8_raw_v019/lightning_logs/version_0/checkpoints/epoch=9-step=2299.ckpt

    """
    import fire
    fire.Fire(repackage)
