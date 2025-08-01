"""
Script for converting a checkpoint (that lives in a training directory) into a
pytorch package with appropriate metadata.
"""
import ubelt as ub
import os
import scriptconfig as scfg
import importlib


class RepackageConfig(scfg.DataConfig):
    r"""
    Convert a raw torch checkpoint into a torch package.

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

    Usage:
        python -m geowatch.mlops.repackager <path-to-ckpt>
    """
    __command__ = 'repackage'
    checkpoint_fpath = scfg.Value(None, position=1, nargs='+', help=ub.paragraph(
        '''
        One or more checkpoint paths to repackage. This can be a path to a file
        or a glob pattern.
        '''))

    force = scfg.Value(False, isflag=True, help='if True, rewrite the packages even if they exist')
    strict = scfg.Value(False, isflag=True, help='if True, fail if there are any errors')


def main(cmdline=True, **kwargs):
    import os
    os.environ['HACK_SAVE_ANYWAY'] = '1'
    config = RepackageConfig.cli(cmdline=cmdline, data=kwargs)
    print('config = {}'.format(ub.urepr(config.to_dict(), nl=1)))
    checkpoint_fpath = config['checkpoint_fpath']
    repackage(checkpoint_fpath, strict=config.strict, force=config.force)


__cli__ = RepackageConfig
__cli__.main = main


def repackage(checkpoint_fpath, force=False, strict=False, dry=False):
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
    from kwutil import util_path
    from kwutil import util_exception
    checkpoint_fpaths = util_path.coerce_patterned_paths(checkpoint_fpath)
    print('Begin repackage')
    print('checkpoint_fpaths = {}'.format(ub.urepr(checkpoint_fpaths, nl=1)))
    package_fpaths = []
    for checkpoint_fpath in checkpoint_fpaths:
        # If we have a checkpoint path we can load it if we make assumptions
        # init method from checkpoint.
        checkpoint_fpath = ub.Path(checkpoint_fpath)
        context = inspect_checkpoint_context(checkpoint_fpath)
        package_name = suggest_package_name_for_checkpoint(context)
        package_fpath = checkpoint_fpath.parent / package_name
        if force or not package_fpath.exists():
            if not dry:
                train_dpath_hint = context['train_dpath_hint']
                model_config_fpath = context.get("config_fpath", None)
                try:
                    repackage_single_checkpoint(checkpoint_fpath, package_fpath,
                                                train_dpath_hint, model_config_fpath)
                except Exception as ex:
                    note = f'ERROR: Failed to package checkpoint={checkpoint_fpath!r}'
                    print(f'{note} ex={ex!r}')
                    if strict:
                        raise util_exception.add_exception_note(ex, note)
        package_fpaths.append(os.fspath(package_fpath))
    print('package_fpaths = {}'.format(ub.urepr(package_fpaths, nl=1)))
    from kwutil import util_yaml
    package_fpaths_ = [ub.shrinkuser(p, home='$HOME') for p in package_fpaths]
    print(util_yaml.Yaml.dumps(package_fpaths_))
    return package_fpaths


def looks_like_training_directory(candidate_dpath):
    paths = [
        candidate_dpath / 'fit_config.yaml',
        candidate_dpath / 'hparams.yaml',
        candidate_dpath / 'config.yaml',
    ]
    return sum(p.exists() for p in paths)


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
        # .resolve() is necessary if we are running within the checkpoint dir
        path_ = ub.Path(checkpoint_fpath).resolve()
        if path_.parent.stem == 'checkpoints':
            train_dpath_hint = path_.parent.parent
        else:
            if looks_like_training_directory(path_.parent):
                train_dpath_hint = path_.parent
            elif looks_like_training_directory(path_.parent.parent):
                train_dpath_hint = path_.parent.parent

    fit_config_fpath = None
    hparams_fpath = None
    config_fpath = None
    if train_dpath_hint is not None:
        # Look at the training config file to get info about this
        # experiment
        candidates = list(train_dpath_hint.glob('fit_config.yaml'))
        if len(candidates) == 1:
            fit_config_fpath = candidates[0]

        candidates = list(train_dpath_hint.glob('hparams.yaml'))
        if len(candidates) == 1:
            hparams_fpath = candidates[0]

        candidates = list(train_dpath_hint.glob('config.yaml'))
        if len(candidates) == 1:
            config_fpath = candidates[0]

    context['package_name'] = package_name
    context['train_dpath_hint'] = train_dpath_hint
    context['checkpoint_fpath'] = checkpoint_fpath
    context['fit_config_fpath'] = fit_config_fpath
    context['hparams_fpath'] = hparams_fpath
    context['config_fpath'] = config_fpath
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


def parse_and_init_config(config):
    assert isinstance(config, dict)

    if ("class_path" in config) and ("init_args" in config):
        class_path = config["class_path"]
        init_args = config["init_args"]

        # I'm not sure how these keys got into the init args, but
        # we can hack around them and remove them. Hopefully nobody
        # uses these names as real config options
        _block_init_args = {'_instantiator', '_class_path'}
        init_args = ub.udict(init_args) - _block_init_args

        init_args = parse_and_init_config(init_args)

        # https://stackoverflow.com/a/8719100
        package_name, method_name = class_path.rsplit(".", 1)
        package = importlib.import_module(package_name)
        module_cls = getattr(package, method_name)
        try:
            module = module_cls(**init_args)
        except Exception:
            # try to restrict to the arguments the class takes, if it supports
            # introspecting this.
            if hasattr(module_cls, 'compatible'):
                init_args = module_cls.compatible(init_args)
                module = module_cls(**init_args)
            else:
                raise
        return module

    return {
        key: (
            parse_and_init_config(value)
            if isinstance(value, dict)
            else value
        )
        for key, value in config.items()
    }


def torch_load_cpu(checkpoint_fpath):
    from packaging.version import parse as LooseVersion
    import torch
    def _cpu_map_location(storage, location):
        return storage
    loadkw = {
        'map_location': _cpu_map_location,
    }
    _TORCH_IS_GE_2_4_0 = LooseVersion(torch.__version__) >= LooseVersion('2.4.0')
    if _TORCH_IS_GE_2_4_0:
        loadkw['weights_only'] = False
    checkpoint = torch.load(checkpoint_fpath, **loadkw)
    return checkpoint


def repackage_single_checkpoint(checkpoint_fpath, package_fpath,
                                train_dpath_hint=None, model_config_fpath=None):
    """
    Primary logic for repackaging a checkpoint into a torch package.

    To do this we need to have some information about how to construct the
    specific module to associate with the weights. We have some heuristics
    built in to take care of this for specific known models, but new models
    will need new logic to handle them. It would be nice to find a way to
    generalize this.

    CommandLine:
        xdoctest -m geowatch.mlops.repackager repackage_single_checkpoint

    Example:
        >>> import ubelt as ub
        >>> import torch
        >>> dpath = ub.Path.appdir('geowatch/tests/repackage').delete().ensuredir()
        >>> package_fpath = dpath / 'my_package.pt'
        >>> checkpoint_fpath = dpath / 'my_checkpoint.ckpt'
        >>> assert not package_fpath.exists()
        >>> # Create an instance of a model, and save a checkpoint to disk
        >>> from geowatch.tasks.fusion import methods
        >>> from geowatch.tasks.fusion import datamodules
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
        >>> from geowatch.mlops.repackager import *  # NOQA
        >>> repackage_single_checkpoint(checkpoint_fpath, package_fpath)
        >>> assert package_fpath.exists()
        >>> # Test we can reload the package
        >>> from geowatch.tasks.fusion.utils import load_model_from_package
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
        >>> from geowatch.cli import torch_model_stats
        >>> row = torch_model_stats.torch_model_stats(package_fpath)
        >>> print(f'row = {ub.urepr(row, nl=2)}')
    """
    checkpoint = torch_load_cpu(checkpoint_fpath)
    # Can use this if we update torch liberator.
    # from torch_liberator.xpu_device import XPU
    # xpu = XPU.coerce('cpu')
    # checkpoint = xpu.load(checkpoint_fpath)

    context = inspect_checkpoint_context(checkpoint_fpath)

    hparams = checkpoint.get('hyper_parameters', None)

    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint

    HACK_WORKAROUND_HPARAM_MISMATCH = True
    if HACK_WORKAROUND_HPARAM_MISMATCH:
        # Due to issues with pytorch-lightning 2.3.0 we check for if hparams
        # are inconsistent and use the one on disk (which is easier to hack
        # into a valid state).
        # References:
        #     https://github.com/Lightning-AI/pytorch-lightning/issues/20311
        hparams_fpath = context['hparams_fpath']
        if hparams_fpath is not None:
            assert hparams_fpath.exists()
            import kwutil
            ondisk_hparams = kwutil.Yaml.load(hparams_fpath, backend='pyyaml')
            if hparams is None:
                common_ondisk_hparams = ub.udict(ondisk_hparams)
            else:
                common_ondisk_hparams = ub.udict(ondisk_hparams) & hparams.keys()

            if hparams != common_ondisk_hparams:
                print(ub.paragraph(
                    '''
                    Warning: hparams in checkpoint differ from the file in the
                    training directory. Using the one from disk instead.
                    '''))
                hparams = common_ondisk_hparams

    if 'input_channels' in hparams:
        from delayed_image.channel_spec import ChannelSpec
        # Hack for strange pickle issue
        chan = hparams['input_channels']

        if chan is not None:
            if not hasattr(chan, '_spec') and hasattr(chan, '_info'):
                chan = ChannelSpec.coerce(chan._info['spec'])
                hparams['input_channels'] = chan
            else:
                hparams['input_channels'] = ChannelSpec.coerce(chan.spec)

    # Construct the model we want to repackage.  For now we just hard code
    # this. But in the future we could use context from the lightning output
    # directory to figure this out more generally.

    if model_config_fpath is None:
        from geowatch.tasks.fusion import methods
        model_cls = methods.MultimodalTransformer
        try:
            model = model_cls(**hparams)
        except Exception:
            # try to restrict to the arguments the class takes, if it supports
            # introspecting this.
            if hasattr(model_cls, 'compatible'):
                hparams = model_cls.compatible(hparams)
                model = model_cls(**hparams)
            else:
                raise
    else:

        # new stuff seems to be missing dataset stats and classes.

        data = load_meta(model_config_fpath)
        if "model" in data:
            model_config = data["model"]

        model_config["init_args"] = hparams | model_config["init_args"]
        try:
            model = parse_and_init_config(model_config)
        except Exception as ex:
            if 'input_sensorchan=None and dataset_stats=None' in str(ex):
                # Can we hack the missing dataset stats case?
                input_norm_keys = [k for k in state_dict if k.startswith('input_norms.')]
                # head_keys = [k for k in state_dict if k.startswith('heads.')]
                # class_keys = [k for k in state_dict if k.startswith('class')]
                unique_sensor_modes = set()
                input_stats = ub.ddict(dict)
                for k in input_norm_keys:
                    _, sensor, chan, stat = k.split('.')
                    unique_sensor_modes.add((sensor, chan))
                    value = state_dict[k]
                    input_stats[(sensor, chan)][stat] = value
                assert model_config['init_args'].get('input_sensorchan') is None, (
                    'should not get this error if it is populated')
                assert model_config['init_args'].get('dataset_stats') is None, (
                    'should not get this error if it is populated')
                # hacking in dataset stats for models that exclusively use it
                model_config['init_args']['dataset_stats'] = {
                    'input_stats': input_stats,
                    'class_freq': None,
                    'unique_sensor_modes': unique_sensor_modes,
                }
                model = parse_and_init_config(model_config)

    model.load_state_dict(state_dict)

    if train_dpath_hint is not None:
        model.train_dpath_hint = train_dpath_hint

    try:
        context['epoch'] = checkpoint['epoch']
    except KeyError:
        ...
    try:
        context['global_step'] = checkpoint['global_step']
    except KeyError:
        ...

    from kwcoco.util import util_json
    context = util_json.ensure_json_serializable(context)

    # We assume the module has its own save package method implemented.
    model.save_package(package_fpath, context=context)
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
        python -m geowatch.mlops.repackager
    """
    main()
