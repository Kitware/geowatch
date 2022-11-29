import ubelt as ub
import os
import scriptconfig as scfg

import importlib
import yaml

class RepackageConfig(scfg.DataConfig):
    r"""
    Attempts to combine checkpoint weights with its associated model code in a
    standalone torch package.

    To do this we must be able to infer how to construct an instance of the
    model to load the weights into. Currently we implement hard coded
    heuristics that only work for specific fusion models.

    Note:
        The output filenames are chosen automatically. In the future we may
        give the user more control here. We may also look for ways to provide
        more hints for determening how to construct model instances either from
        context or via these configuration arguments.

    Ignore:
        python -m watch.mlops.repackager  \
            $HOME/data/dvc-repos/smart_expt_dvc/training/yardrat/jon.crall/Drop4-SC/runs/Drop4_tune_V30_V1/lightning_logs/version_6/checkpoints/epoch=35-step=486072.ckpt \
            $HOME/data/dvc-repos/smart_expt_dvc/training/yardrat/jon.crall/Drop4-SC/runs/Drop4_tune_V30_V1/lightning_logs/version_6/checkpoints/epoch=12-step=175526-v1.ckpt \
            $HOME/data/dvc-repos/smart_expt_dvc/training/yardrat/jon.crall/Drop4-SC/runs/Drop4_tune_V30_V1/lightning_logs/version_6/checkpoints/epoch=21-step=297044-v2.ckpt \
            $HOME/data/dvc-repos/smart_expt_dvc/training/yardrat/jon.crall/Drop4-SC/runs/Drop4_tune_V30_V1/lightning_logs/version_6/checkpoints/epoch=32-step=445566.ckpt \
            $HOME/data/dvc-repos/smart_expt_dvc/training/yardrat/jon.crall/Drop4-SC/runs/Drop4_tune_V30_V1/lightning_logs/version_6/checkpoints/epoch=36-step=499574.ckpt \
            $HOME/data/dvc-repos/smart_expt_dvc/training/yardrat/jon.crall/Drop4-SC/runs/Drop4_tune_V30_V1/lightning_logs/version_6/checkpoints/epoch=37-step=513076.ckpt \
            $HOME/data/dvc-repos/smart_expt_dvc/training/yardrat/jon.crall/Drop4-SC/runs/Drop4_tune_V30_V1/lightning_logs/version_6/checkpoints/epoch=37-step=513076.ckpt \
            $HOME/data/dvc-repos/smart_expt_dvc/training/yardrat/jon.crall/Drop4-SC/runs/Drop4_tune_V30_V1/lightning_logs/version_6/checkpoints/epoch=89-step=1215180.ckpt


        python -m watch.mlops.repackager \
            $HOME/data/dvc-repos/smart_expt_dvc/training/yardrat/jon.crall/Drop4-SC/runs/Drop4_tune_V30_V1/lightning_logs/version_6/checkpoints/epoch=3*.ckpt
    """
    checkpoint_fpath = scfg.Value(None, position=1, nargs='+', help=ub.paragraph(
        '''
        One or more checkpoint paths to repackage. This can be a path to a file
        or a glob pattern.
        '''))

    force = scfg.Value(False, isflag=True, help='if True, rewrite the packages even if they exist')


def main(**kwargs):
    config = RepackageConfig.cli(data=kwargs)
    print('config = {}'.format(ub.repr2(config.to_dict(), nl=1)))
    checkpoint_fpath = config['checkpoint_fpath']
    repackage(checkpoint_fpath, force=config['force'])


def parse_and_init_config(config):
    assert isinstance(config, dict)

    if ("class_path" in config) and ("init_args" in config):
        class_path = config["class_path"]
        init_args = config["init_args"]

        init_args = parse_and_init_config(init_args)

        # https://stackoverflow.com/a/8719100
        package_name, method_name = class_path.rsplit(".", 1)
        package = importlib.import_module(package_name)
        method = getattr(package, method_name)
        module = method(**init_args)
        return module

    return {
        key: (
            parse_and_init_config(value)
            if isinstance(value, dict)
            else value
        )
        for key, value in config.items()
    }


def repackage(checkpoint_fpath, force=False, dry=False):
    """
    Logic for handling multiple checkpoint repackages at a time.
    Automatically chooses the new package name.

    TODO:
        generalize this beyond the fusion model, also refactor.

    Ignore:
        >>> import ubelt as ub
        >>> checkpoint_fpath = ub.expandpath(
        ...     '/home/joncrall/data/dvc-repos/smart_watch_dvc/models/fusion/checkpoint_DirectCD_smt_it_joint_p8_raw9common_v5_tune_from_onera_epoch=2-step=2147.ckpt')
    """
    from watch.utils import util_path
    checkpoint_fpaths = util_path.coerce_patterned_paths(checkpoint_fpath)
    print('Begin repackage')
    print('checkpoint_fpaths = {}'.format(ub.repr2(checkpoint_fpaths, nl=1)))
    package_fpaths = []
    for checkpoint_fpath in checkpoint_fpaths:
        # If we have a checkpoint path we can load it if we make assumptions
        # init method from checkpoint.

        checkpoint_fpath = os.fspath(checkpoint_fpath)
        print(checkpoint_fpath)

        x = ub.Path(ub.augpath(checkpoint_fpath, ext='.pt'))
        package_name = x.name

        # Can we precompute the package name of this checkpoint?
        train_dpath_hint = None
        if checkpoint_fpath.endswith('.ckpt'):
            # .resolve() is necessary if we are running within the checkpoint dir
            path_ = ub.Path(checkpoint_fpath).resolve()
            if path_.parent.stem == 'checkpoints':
                train_dpath_hint = path_.parent.parent

        model_config = None
        meta_fpath = None
        if train_dpath_hint is not None:
            candidates = list(train_dpath_hint.glob('config.yaml'))
            if len(candidates) == 1:
                meta_fpath = candidates[0]
                data = load_meta(meta_fpath)

                if "model" in data:
                    model_config = data["model"]

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

                if model_config is None:
                    model = methods.MultimodalTransformer(**hparams)
                else:
                    model_config["init_args"] = hparams | model_config["init_args"]
                    model = parse_and_init_config(model_config)

                state_dict = checkpoint['state_dict']
                model.load_state_dict(state_dict)

                if train_dpath_hint is not None:
                    model.train_dpath_hint = train_dpath_hint

                model.save_package(os.fspath(package_fpath))
                print(f'wrote: package_fpath={package_fpath}')

        package_fpaths.append(os.fspath(package_fpath))
    print('package_fpaths = {}'.format(ub.repr2(package_fpaths, nl=1)))
    from watch.utils import util_yaml
    package_fpaths_ = [ub.shrinkuser(p, home='$HOME') for p in package_fpaths]
    print(util_yaml.yaml_dumps(package_fpaths_))
    return package_fpaths


def inspect_checkpoint_context(checkpoint_fpath):
    """
    Use heuristics to attempt to find the context in which this checkpoint was
    trained.
    """
    context = {}
    checkpoint_fpath = ub.Path(checkpoint_fpath)
    package_name = checkpoint_fpath.augment(ext='.pt').name

    # Can we precompute the package name of this checkpoint?
    train_dpath_hint = None
    if checkpoint_fpath.name.endswith('.ckpt'):
        if checkpoint_fpath.parent.stem == 'checkpoints':
            train_dpath_hint = checkpoint_fpath.parent.parent

    fit_config_fpath = None
    hparams_fpath = None
    if train_dpath_hint is not None:
        # Look at the training config file to get info about this
        # experiment
        candidates = list(train_dpath_hint.glob('fit_config.yaml'))
        if len(candidates) == 1:
            fit_config_fpath = candidates[0]
        candidates = list(train_dpath_hint.glob('hparams.yaml'))
        if len(candidates) == 1:
            hparams_fpath = candidates[0]

    context['package_name'] = package_name
    context['train_dpath_hint'] = train_dpath_hint
    context['checkpoint_fpath'] = checkpoint_fpath
    context['fit_config_fpath'] = fit_config_fpath
    context['hparams_fpath'] = hparams_fpath
    return context


def suggest_package_name_for_checkpoint(context):
    """
    Suggest a more distinguishable name for the checkpoint based on context
    """
    checkpoint_fpath = ub.Path(context['checkpoint_fpath'])
    package_name = checkpoint_fpath.augment(ext='.pt').name
    meta_fpath = context.get('fit_config_fpath', None)
    if meta_fpath is not None:
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
    return package_name


def repackage_single_checkpoint(checkpoint_fpath, package_fpath,
                                train_dpath_hint=None):
    """
    Primary logic for repackaging a checkpoint into a torch package.

    To do this we need to have some information about how to construct the
    specific module to associate with the weights. We have some heuristics
    built in to take care of this for specific known models, but new models
    will need new logic to handle them. It would be nice to find a way to
    generalize this.

    Example:
        >>> import ubelt as ub
        >>> import torch
        >>> dpath = ub.Path.appdir('watch/tests/repackage').delete().ensuredir()
        >>> package_fpath = dpath / 'my_package.pt'
        >>> checkpoint_fpath = dpath / 'my_checkpoint.ckpt'
        >>> assert not package_fpath.exists()
        >>> # Create an instance of a model, and save a checkpoint to disk
        >>> from watch.tasks.fusion import methods
        >>> from watch.tasks.fusion import datamodules
        >>> model = self = methods.MultimodalTransformer(
        >>>     arch_name="smt_it_joint_p2", input_sensorchan=5,
        >>>     change_head_hidden=0, saliency_head_hidden=0,
        >>>     class_head_hidden=0)
        >>> # Save a checkpoint to disk.
        >>> model_state = model.state_dict()
        >>> # (fixme: how to get a lightning style checkpoint structure?)
        >>> checkpoint = {
        >>>     'state_dict': model_state,
        >>>     'hyper_parameters': model._hparams,
        >>> }
        >>> with open(checkpoint_fpath, 'wb') as file:
        ...     torch.save(checkpoint, file)
        >>> from watch.mlops.repackager import *  # NOQA
        >>> repackage_single_checkpoint(checkpoint_fpath, package_fpath)
        >>> assert package_fpath.exists()
        >>> # Test we can reload the package
        >>> from watch.tasks.fusion.utils import load_model_from_package
        >>> model2 = load_model_from_package(package_fpath)
        >>> # TODO: get allclose working on the nested dict
        >>> params1 = dict(model.named_parameters())
        >>> params2 = dict(model2.named_parameters())
        >>> k = 'encoder.layers.0.mlp.3.weight'
        >>> assert torch.allclose(params1[k], params2[k])
        >>> assert params1[k] is not params2[k]
        >>> params1 = ub.IndexableWalker(dict(model.named_parameters()))
        >>> params2 = ub.IndexableWalker(dict(model2.named_parameters()))
        >>> for k, v in params1:
        ...     assert torch.allclose(params1[k], params2[k])
        ...     assert params1[k] is not params2[k]
        >>> # Test that we can get model stats
        >>> from watch.cli import torch_model_stats
        >>> torch_model_stats.torch_model_stats(package_fpath)
    """
    from torch_liberator.xpu_device import XPU
    xpu = XPU.coerce('cpu')
    checkpoint = xpu.load(checkpoint_fpath)

    hparams = checkpoint['hyper_parameters']

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

    # Construct the model we want to repackage.  For now we just hard code
    # this. But in the future we could use context from the lightning output
    # directory to figure this out more generally.

    from watch.tasks.fusion import methods
    method = methods.MultimodalTransformer(**hparams)
    state_dict = checkpoint['state_dict']
    method.load_state_dict(state_dict)

    if train_dpath_hint is not None:
        method.train_dpath_hint = train_dpath_hint

    # We assume the module has its own save package method implemented.
    method.save_package(os.fspath(package_fpath))
    print(f'wrote: package_fpath={package_fpath}')


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
