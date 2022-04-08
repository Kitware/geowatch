"""
Print stats about a torch model

Exapmle Usage:
    DVC_DPATH=$(WATCH_PREIMPORT=0 python -m watch.cli.find_dvc)
    PACKAGE_FPATH=$DVC_DPATH/models/fusion/SC-20201117/BOTH_smt_it_stm_p8_L1_DIL_v55/BOTH_smt_it_stm_p8_L1_DIL_v55_epoch=5-step=53819.pt
    python -m watch.cli.torch_model_stats $PACKAGE_FPATH
"""
import scriptconfig as scfg
import ubelt as ub


class TorchModelStatsConfig(scfg.Config):
    """
    Print stats about a torch model.

    Currently some things are hard-coded for fusion models
    """
    default = {
        'src': scfg.PathList(help='path to one or more torch models', position=1),
        'stem_stats': scfg.Value(help='if True, print more verbose model mean/std', position=2),
    }


def main(cmdline=False, **kwargs):
    """
    Ignore:
        import watch
        from watch.cli.torch_model_stats import *  # NOQA
        dvc_dpath = watch.find_smart_dvc_dpath()
        package_fpath1 = dvc_dpath / 'models/fusion/SC-20201117/BOTH_smt_it_stm_p8_L1_DIL_v55/BOTH_smt_it_stm_p8_L1_DIL_v55_epoch=5-step=53819.pt'
        package_fpath2 = dvc_dpath / 'models/fusion/SC-20201117/BAS_smt_it_stm_p8_TUNE_L1_RAW_v58/BAS_smt_it_stm_p8_TUNE_L1_RAW_v58_epoch=3-step=81135.pt'
        package_fpath = dvc_dpath / 'models/fusion/SC-20201117/BAS_smt_it_stm_p8_L1_raw_v53/BAS_smt_it_stm_p8_L1_raw_v53_epoch=3-step=85011.pt'
        kwargs = {
            'src': [package_fpath2, package_fpath1]
        }
        main(cmdline=False, **kwargs)

    """
    from watch.tasks.fusion import utils
    import xdev
    import watch
    import netharn as nh

    config = TorchModelStatsConfig(cmdline=cmdline, data=kwargs)
    print('config = {}'.format(ub.repr2(dict(config), nl=1)))
    package_paths = config['src']

    dvc_dpath = watch.find_smart_dvc_dpath()

    package_rows = []
    for package_fpath in package_paths:

        package_fpath = ub.Path(package_fpath)
        file_stat = package_fpath.stat()

        # TODO: generalize the load-package
        raw_module = utils.load_model_from_package(package_fpath)
        if hasattr(raw_module, 'module'):
            module = raw_module.module
        else:
            module = raw_module

        # TODO: get the category freq

        model_stats = {}
        num_params = nh.util.number_of_parameters(module)

        print(ub.repr2(utils.model_json(module, max_depth=3), nl=-1, sort=0))
        # print(ub.repr2(utils.model_json(module, max_depth=2), nl=-1, sort=0))

        state = module.state_dict()
        state_keys = list(state.keys())
        # print('state_keys = {}'.format(ub.repr2(state_keys, nl=1)))

        import kwcoco
        if hasattr(module, 'dataset_stats'):
            module.dataset_stats.keys()

            known_input_stats = []
            unknown_input_stats = []
            sensor_modes_with_stats = set()

            unique_sensors = set()
            for sens_chan_key, stats in module.dataset_stats['input_stats'].items():
                sensor, channel = sens_chan_key
                channel = kwcoco.ChannelSpec.coerce(channel).concise().spec
                sensor_modes_with_stats.add((sensor, channel))
                unique_sensors.add(sensor)
                sensor_stat = {
                    'sensor': sensor,
                    'channel': channel,
                }
                if config['stem_stats']:
                    sensor_stat.update({
                        'mean': stats['mean'].ravel().tolist(),
                        'std': stats['std'].ravel().tolist(),
                    })
                known_input_stats.append(sensor_stat)

            unique_sensor_modes = list(module.dataset_stats['unique_sensor_modes'])
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

            size_str = xdev.byte_str(file_stat.st_size)

            # Add in some params about how this model was trained
            fit_config = raw_module.fit_config

            train_dataset = ub.Path(fit_config['train_dataset'])
            if str(train_dataset).startswith(str(dvc_dpath)):
                train_dataset = train_dataset.relative_to(dvc_dpath)

            if str(package_fpath).startswith(str(dvc_dpath)):
                package_fpath = package_fpath.relative_to(dvc_dpath)

            heads = []
            if fit_config['global_class_weight']:
                heads.append('class')

            if fit_config['global_saliency_weight']:
                heads.append('saliency')

            spacetime_stats = {
                'chip_size': fit_config['chip_size'],
                'time_steps': fit_config['time_steps'],
                'time_sampling': fit_config['time_sampling'],
                'time_span': fit_config['time_span'],
            }

            model_stats['size'] = size_str
            model_stats['num_params'] = num_params
            model_stats['num_states'] = len(state_keys)
            model_stats['heads'] = heads
            model_stats['train_dataset'] = str(train_dataset)
            model_stats['spacetime_stats'] = spacetime_stats
            model_stats['classes'] = list(module.classes)
            model_stats['known_inputs'] = known_input_stats
            model_stats['unknown_inputs'] = unknown_input_stats

            row = {
                'name': package_fpath.stem,
                'task': 'TODO',
                'file_name': str(package_fpath),
                'input_channels': module.input_channels.concise().spec,
                'sensors': sorted(unique_sensors),
                'train_dataset': str(train_dataset),
                'model_stats': model_stats,
            }
            package_rows.append(row)
            print('model_stats = {}'.format(ub.repr2(model_stats, nl=2, sort=0, precision=2)))

        print('package_rows = {}'.format(ub.repr2(package_rows, nl=2, sort=0)))


_CLI = TorchModelStatsConfig

if __name__ == '__main__':
    """
    CommandLine:
        python -m watch.cli.torch_model_stats
    """
    main(cmdline=True)
