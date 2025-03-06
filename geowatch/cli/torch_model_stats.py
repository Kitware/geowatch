"""
Print stats about a torch model

Exapmle Usage:
    DVC_DPATH=$(geowatch_dvc)
    PACKAGE_FPATH=$DVC_DPATH/models/fusion/SC-20201117/BOTH_smt_it_stm_p8_L1_DIL_v55/BOTH_smt_it_stm_p8_L1_DIL_v55_epoch=5-step=53819.pt
    python -m geowatch.cli.torch_model_stats $PACKAGE_FPATH
"""
import scriptconfig as scfg
import ubelt as ub


class TorchModelStatsConfig(scfg.DataConfig):
    """
    Print stats about a torch model.

    Currently some things are hard-coded for fusion models
    """
    src = scfg.PathList(help='path to one or more torch models', position=1)
    stem_stats = scfg.Value(True, isflag=True, help='if True, print more verbose model mean/std')
    hparams = scfg.Value(True, isflag=True, help='if True, print fit hyperparameters')


def main(cmdline=False, **kwargs):
    """
    Ignore:
        import geowatch
        from geowatch.cli.torch_model_stats import *  # NOQA
        dvc_dpath = geowatch.find_dvc_dpath()
        package_fpath1 = dvc_dpath / 'models/fusion/SC-20201117/BOTH_smt_it_stm_p8_L1_DIL_v55/BOTH_smt_it_stm_p8_L1_DIL_v55_epoch=5-step=53819.pt'
        package_fpath2 = dvc_dpath / 'models/fusion/SC-20201117/BAS_smt_it_stm_p8_TUNE_L1_RAW_v58/BAS_smt_it_stm_p8_TUNE_L1_RAW_v58_epoch=3-step=81135.pt'
        package_fpath = dvc_dpath / 'models/fusion/SC-20201117/BAS_smt_it_stm_p8_L1_raw_v53/BAS_smt_it_stm_p8_L1_raw_v53_epoch=3-step=85011.pt'
        kwargs = {
            'src': [package_fpath2, package_fpath1]
        }
        main(cmdline=False, **kwargs)

    """
    config = TorchModelStatsConfig.cli(cmdline=cmdline, data=kwargs, strict=True)
    import rich
    from rich.markup import escape
    rich.print('config = {}'.format(escape(ub.urepr(config, nl=1))))
    package_paths = config['src']

    import geowatch
    import warnings
    if not ub.iterable(package_paths):
        package_paths = [package_paths]

    try:
        dvc_dpath = geowatch.find_dvc_dpath()
    except Exception:
        dvc_dpath = None

    package_rows = []
    for package_fpath in package_paths:
        print('--------')
        print(f'package_fpath={package_fpath}')

        stem_stats = config['stem_stats']
        try:
            row = torch_model_stats(package_fpath, stem_stats=stem_stats, dvc_dpath=dvc_dpath)
        except RuntimeError as pkg_ex:
            print('Reading as a package failed, attempting to read as a checkpoint')
            checkpoint_fpath = package_fpath
            try:
                checkpoint_stats = torch_checkpoint_stats(checkpoint_fpath)
            except Exception as ckpt_ex:
                warnings.warn(f'Unable to read input as a checkpoint due to: {ckpt_ex}.')
                raise pkg_ex
            else:
                warnings.warn(f'Unable to read input as a package due to: {pkg_ex}. Interpreting as a checkpoint instead')

            rich.print('checkpoint_stats = {}'.format(ub.urepr(checkpoint_stats, nl=2, sort=0, precision=2)))
        else:
            model_stats = row.get('model_stats', None)
            fit_config = row.pop('fit_config', None)
            config_cli_yaml = row.pop('config_cli_yaml', None)
            if config.hparams:
                rich.print('fit_config = {}'.format(ub.urepr(fit_config, nl=1)))
                rich.print('config_cli_yaml = {}'.format(ub.urepr(config_cli_yaml, nl=2)))
            rich.print('model_stats = {}'.format(ub.urepr(model_stats, nl=2, sort=0, precision=2)))
            package_rows.append(row)

    print('package_rows = {}'.format(ub.urepr(package_rows, nl=2, sort=0)))


def torch_checkpoint_stats(checkpoint_fpath):
    """
    A fallback set of statistics we can make for checkpoints only.

    Summarizes a PyTorch checkpoint by extracting useful metadata about the model's state_dict
    and optimizer states.

    Args:
        checkpoint_path (str | PathLike): Path to the checkpoint file.

    Returns:
        dict: Summary statistics of the checkpoint.
    """
    import torch
    data = torch.load(checkpoint_fpath)
    top_level_keys = list(data.keys())
    print(f'top_level_keys = {ub.urepr(top_level_keys, nl=1)}')

    # TODO: Given a checkpoint path, we may be able to read more information
    # about it if we can find the corresponding haprams.yaml or
    # train_config.yaml

    nested_keys = {
        'state_dict',
        'optimizer_states',
        'loops'
    }
    nested = ub.udict.intersection(data, nested_keys)
    non_nested = ub.udict.difference(data, nested_keys)
    stats = {
        'path': checkpoint_fpath,
        **non_nested
    }

    if 'optimizer_states' in nested:
        optimizer_states_stats = {}
        optimizer_states = nested['optimizer_states']
        optimizer_states_stats['num_optimizer_states'] = len(optimizer_states)
        opt_stat_list = []
        for opt_state in optimizer_states:
            opt_state_stats = {}
            # TODO: Can we do more here?
            opt_state_stats['num_param_groups'] = len(opt_state['param_groups'])
            opt_stat_list.append(opt_state_stats)
        optimizer_states_stats['total_num_param_groups'] = sum(r['num_param_groups'] for r in opt_stat_list)
    else:
        optimizer_states_stats = None

    if 'state_dict' in nested:
        state_dict_stats = {}
        state_dict = nested['state_dict']
        len(state_dict)
        # state_shape = ub.udict.map_values(state_dict, lambda x: x.shape)
        state_numel = ub.udict.map_values(state_dict, lambda x: x.numel())
        total_params = sum(state_numel.values())
        # TODO: Can we use the optimizer to determine the trainable params?
        state_dict_stats['num_tensors'] = len(state_dict)
        state_dict_stats['total_params'] = total_params
    else:
        state_dict_stats = None

    if 'loops' in nested:
        loop_stats = {}
        # TODO: extract appopriate stats
        loops = nested['loops']
    else:
        loops = None
        loop_stats = None
    # Just use the loops
    loop_stats = loops

    stats['loops'] = loop_stats
    stats['state_dict'] = state_dict_stats
    stats['optimizer_states_stats'] = optimizer_states_stats
    return stats


def torch_model_stats(package_fpath, stem_stats=True, dvc_dpath=None):
    import kwcoco
    from geowatch.tasks.fusion import utils
    from geowatch.utils import util_netharn
    from geowatch.monkey import monkey_torchmetrics
    monkey_torchmetrics.fix_torchmetrics_compatability()

    package_fpath = ub.Path(package_fpath)

    if not package_fpath.exists():
        if package_fpath.augment(tail='.dvc').exists():
            raise Exception('model does not exist, but its dvc file does')
        else:
            raise Exception('model does not exist')

    file_stat = package_fpath.stat()

    # TODO: generalize the load-package
    raw_module, package_header = utils.load_model_from_package(package_fpath, with_header=True)

    if hasattr(raw_module, 'module'):
        module = raw_module.module
    else:
        module = raw_module

    # TODO: get the category freq

    model_stats = {}
    num_params = util_netharn.number_of_parameters(module)

    print(ub.urepr(utils.model_json(module, max_depth=3), nl=-1, sort=0))
    # print(ub.urepr(utils.model_json(module, max_depth=2), nl=-1, sort=0))

    # import xdev
    # with xdev.embed_on_exception_context:
    try:
        state = module.state_dict()
    except Exception:
        if hasattr(module, 'head_metrics'):
            module.head_metrics.clear()
            state = module.state_dict()
        else:
            raise
    state_keys = list(state.keys())
    # print('state_keys = {}'.format(ub.urepr(state_keys, nl=1)))

    unique_sensors = set()
    config_cli_yaml = None
    train_dataset = None
    prenorm_stats = None
    fit_config = {}
    if hasattr(module, 'dataset_stats') and module.dataset_stats is not None:

        dataset_stats = module.dataset_stats.copy()

        if 'modality_input_stats' in dataset_stats:
            # This is too much info to print
            dataset_stats.pop('modality_input_stats', None)

        known_input_stats = []
        unknown_input_stats = []
        sensor_modes_with_stats = set()

        for sens_chan_key, stats in dataset_stats['input_stats'].items():
            sensor, channel = sens_chan_key
            channel = kwcoco.ChannelSpec.coerce(channel).concise().spec
            sensor_modes_with_stats.add((sensor, channel))
            unique_sensors.add(sensor)
            sensor_stat = {
                'sensor': sensor,
                'channel': channel,
            }
            if stem_stats:
                import numpy as np
                sensor_stat.update({
                    'mean': np.asarray(stats['mean']).ravel().tolist(),
                    'std': np.asarray(stats['std']).ravel().tolist(),
                })
            known_input_stats.append(sensor_stat)

        unique_sensor_modes = list(dataset_stats['unique_sensor_modes'])
        for sensor, channel in unique_sensor_modes:
            channel = kwcoco.ChannelSpec.coerce(channel).concise().spec
            key = (sensor, channel)
            if key not in sensor_modes_with_stats:
                unique_sensors.add(sensor)
                unknown_input_stats.append(
                    {
                        'sensor': sensor,
                        'channel': channel,
                    }
                )

        mb_size = file_stat.st_size / (2.0 ** 20)
        size_str = ub.urepr(mb_size, precision=2) + ' MB'

        # Add in some params about how this model was trained
        if hasattr(raw_module, 'config_cli_yaml'):
            config_cli_yaml = raw_module.config_cli_yaml
        else:
            config_cli_yaml = None

        if hasattr(raw_module, 'fit_config'):
            # Old non-cli modules
            fit_config = raw_module.fit_config
        else:
            # new lightning cli modules
            fit_config = (
                ub.udict(getattr(raw_module, 'datamodule_hparams', {})) |
                ub.udict(raw_module.hparams)
            )

        if 'train_dataset' in fit_config:
            train_dataset = ub.Path(fit_config['train_dataset'])
        else:
            if config_cli_yaml is not None:
                train_dataset = config_cli_yaml.get('data', {}).get('train_dataset', None)
            else:
                train_dataset = None

        if dvc_dpath is not None and train_dataset is not None:
            try:
                if str(train_dataset).startswith(str(dvc_dpath)):
                    train_dataset = train_dataset.relative_to(dvc_dpath)

                if str(package_fpath).startswith(str(dvc_dpath)):
                    package_fpath = package_fpath.relative_to(dvc_dpath)
            except Exception:
                ...

        heads = []
        if fit_config['global_class_weight']:
            heads.append('class')

        if fit_config['global_saliency_weight']:
            heads.append('saliency')

        spacetime_stats = ub.udict(fit_config) & [
            'chip_size',
            'time_steps',
            'time_sampling',
            'time_span',
            'chip_dims',
            'window_space_scale',
            'input_space_scale',
        ]

        # spacetime_stats = {
        #     'chip_size': fit_config['chip_size'],
        #     'time_steps': fit_config['time_steps'],
        #     'time_sampling': fit_config['time_sampling'],
        #     'time_span': fit_config['time_span'],
        # }

        model_stats['size'] = size_str
        model_stats['num_params'] = num_params
        model_stats['num_states'] = len(state_keys)
        model_stats['heads'] = heads
        model_stats['train_dataset'] = None if train_dataset is None else str(train_dataset)
        model_stats['spacetime_stats'] = spacetime_stats
        model_stats['classes'] = list(module.classes)
        model_stats['known_inputs'] = known_input_stats
        model_stats['unknown_inputs'] = unknown_input_stats

        # Normalization done in the dataloader
        prenorm_stats = {
            'normalize_peritem': fit_config.get('normalize_peritem'),
        }

    param_stats = {
        name: {
            "size": param.size(),
            "min": param.min().item(),
            "max": param.max().item(),
            "mean": param.mean().item(),
            "std": param.std().item(),
        }
        for name, param in module.named_parameters()
    }
    param_stats_summary = {
        "min": min([summary["min"] for summary in param_stats.values()]),
        "max": min([summary["max"] for summary in param_stats.values()]),
        "mean": min([summary["mean"] for summary in param_stats.values()]),
    }

    try:
        unique_sensors = sorted(unique_sensors)
    except TypeError:
        ...

    row = {
        'name': package_fpath.stem,
        'task': 'TODO',
        'file_name': str(package_fpath),
        'sensors': unique_sensors,
        'train_dataset': str(train_dataset),
        'fit_config': fit_config,
        'config_cli_yaml': config_cli_yaml,
        'model_stats': model_stats,
        'prenorm_stats': prenorm_stats,
        'param_stats': param_stats_summary,
        'package_header': package_header,
    }

    if hasattr(module, 'input_sensorchan'):
        input_sensorchan = module.input_sensorchan.concise().spec
        row['input_sensorchan'] = input_sensorchan
    elif hasattr(module, 'input_channels'):
        input_channels = module.input_channels.concise().spec
        row['input_channels'] = input_channels

    return row


def fallback(fpath):
    import zipfile
    zfile = zipfile.ZipFile(fpath)
    for internal_path in zfile.namelist():
        if internal_path.endswith('.yaml'):
            data = zfile.read(internal_path)
            print(data.decode('utf8'))


__cli__ = TorchModelStatsConfig

if __name__ == '__main__':
    """
    CommandLine:
        python -m geowatch.cli.torch_model_stats
    """
    main(cmdline=True)
