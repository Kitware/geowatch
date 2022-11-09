import ubelt as ub
import scriptconfig as scfg


class RepackageConfig(scfg.DataConfig):
    """
    Standalone repackage logic
    """
    checkpoint_fpath = scfg.Value(None, position=1, help='the checkpoint path to repackage')


def main(**kwargs):
    config = RepackageConfig.cli(data=kwargs)
    checkpoint_fpath = config['checkpoint_fpath']
    repackage(checkpoint_fpath)


def repackage(checkpoint_fpath, force=False, dry=False):
    """
    TODO:
        generalize this beyond the fusion model, also refactor.

    checkpoint_fpath

    Ignore:
        >>> import ubelt as ub
        >>> checkpoint_fpath = ub.expandpath(
        ...     '/home/joncrall/data/dvc-repos/smart_watch_dvc/models/fusion/checkpoint_DirectCD_smt_it_joint_p8_raw9common_v5_tune_from_onera_epoch=2-step=2147.ckpt')

    checkpoint_fpath = '/home/joncrall/remote/namek/smart_watch_dvc/training/namek/joncrall/Drop1_October2021/runs/Saliency_smt_it_joint_p8_rgb_uconn_ukyshared_v001/lightning_logs/version_1/checkpoints/epoch=53-step=28457.ckpt'
    """
    import os
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

        meta_fpath = None
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

                if 'input_sensorchan' not in hparams:
                    # HACK: we had old models that did not save their hparams
                    # correctly. Try to fix them up here. The best we can do
                    # is try to start a small training run with the exact same
                    # settings and capture fixed model state from that.
                    if meta_fpath is None:
                        raise Exception('we cant do a fix without the meta fpath')

                    hackfix_hparams_fpath = meta_fpath.augment(prefix='hackfix_')
                    if not hackfix_hparams_fpath.exists():
                        # Do this once per experiment group to save time.
                        import tempfile
                        tmp_dpath = ub.Path(tempfile.mkdtemp())
                        tmp_root = (tmp_dpath / package_name)
                        ub.cmd(f'python -m watch.tasks.fusion.fit '
                               f'--config "{meta_fpath}" --default_root_dir "{tmp_root}" '
                               f'--max_epochs=0 --max_epoch_length=1', system=1, verbose=3)
                        tmp_llogs_dpaths = sorted((tmp_root / 'lightning_logs').glob('*'))
                        assert tmp_llogs_dpaths, 'cannot fix this model'
                        tmp_hparams_fpath = tmp_llogs_dpaths[-1] / 'hparams.yaml'
                        import shutil
                        shutil.copy(tmp_hparams_fpath, hackfix_hparams_fpath)

                    import yaml
                    with open(hackfix_hparams_fpath, 'r') as file:
                        hacked_hparams = yaml.load(file, yaml.Loader)
                    hacked_hparams = ub.udict(hacked_hparams)
                    # Select the known problematic variables
                    problem_hparams = hacked_hparams.subdict([
                        'classes', 'dataset_stats', 'input_sensorchan',
                        'input_channels'])
                    hparams.update(problem_hparams)
                    # hacked_hparams - hparams

                if 'input_channels' in hparams:
                    import kwcoco
                    # Hack for strange pickle issue
                    chan = hparams['input_channels']
                    if chan is not None:
                        if not hasattr(chan, '_spec') and hasattr(chan, '_info'):
                            chan = kwcoco.ChannelSpec.coerce(chan._info['spec'])
                            hparams['input_channels'] = chan
                        else:
                            hparams['input_channels'] = kwcoco.ChannelSpec.coerce(chan.spec)

                method = methods.MultimodalTransformer(**hparams)
                state_dict = checkpoint['state_dict']
                method.load_state_dict(state_dict)

                if train_dpath_hint is not None:
                    method.train_dpath_hint = train_dpath_hint

                method.save_package(os.fspath(package_fpath))
                print(f'wrote: package_fpath={package_fpath}')
        package_fpaths.append(os.fspath(package_fpath))
    return package_fpaths


@ub.memoize
def load_meta(fpath):
    import yaml
    with open(fpath, 'r') as file:
        data = yaml.safe_load(file)
    return data


if __name__ == '__main__':
    """
    CommandLine:
        python -m watch.mlops.repackager
    """
    main()
