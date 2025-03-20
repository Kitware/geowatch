"""
Main entrypoint for a fusion fit job.

For unit tests see:
    ../../../tests/test_lightning_cli_fit.py

For tutorials see:
    ../../../docs/source/manual/tutorial/tutorial1_rgb_network.sh
"""
# TODO: lets avoid the import * here.
# I think these need to be exposed as options
from geowatch.tasks.fusion.methods import MultimodalTransformer  # NOQA
from geowatch.tasks.fusion.methods import HeterogeneousModel  # NOQA
from geowatch.tasks.fusion.methods import UNetBaseline  # NOQA
from geowatch.tasks.fusion.methods import NoopModel  # NOQA
from geowatch.tasks.fusion.methods import channelwise_transformer  # NOQA
from geowatch.tasks.fusion.methods import heterogeneous  # NOQA
from geowatch.tasks.fusion.methods import noop_model  # NOQA
from geowatch.tasks.fusion.methods import unet_baseline  # NOQA

from geowatch.tasks.fusion.datamodules.kwcoco_datamodule import KWCocoVideoDataModule
from geowatch.tasks.fusion._lightning_components import SmartTrainer, SmartLightningCLI, DDP_WORKAROUND
from geowatch.utils import lightning_ext as pl_ext

import pytorch_lightning as pl
import ubelt as ub

import yaml
from jsonargparse import set_loader, set_dumper
# from pytorch_lightning.utilities.rank_zero import rank_zero_only


from geowatch.monkey import monkey_numpy  # NOQA
# monkey_numpy.patch_numpy_dtypes()
monkey_numpy.patch_numpy_2x()


# Not very safe, but needed to parse tuples e.g. datamodule.dataset_stats
# TODO: yaml.SafeLoader + tuple parsing
def custom_yaml_load(stream):
    return yaml.load(stream, Loader=yaml.FullLoader)


set_loader('yaml_unsafe_for_tuples', custom_yaml_load)


def custom_yaml_dump(data):
    return yaml.dump(data, Dumper=yaml.Dumper)


set_dumper('yaml_unsafe_for_tuples', custom_yaml_dump)


def make_cli(config=None):
    """
    Main entrypoint that creates the CLI and works around issues when config is
    passed as a parameter rather than via ``sys.argv`` itself.

    Args:
        config (None | Dict):
            if specified disables sys.argv usage and executes a training run
            with the specified config.

    Returns:
        SmartLightningCLI

    Note:
        Currently, creating the CLI will invoke it. We could modify this
        function to have the option to not invoke by specifying ``run=False``
        to :class:`LightningCLI`, but for some reason that changes the expected
        form of the config (you must specify subcommand if run=True but must
        not if run=False). We need to understand exactly what's going on there
        before we expose a way to set run=False.
    """

    if isinstance(config, str):
        try:
            if len(config) > 200:
                raise Exception
            if ub.Path(config).exists():
                config = config.read_text()
        except Exception:
            ...

        def nested_to_jsonnest(nested):
            config = {}
            for p, v in ub.IndexableWalker(nested):
                if not isinstance(v, (dict, list)):
                    k = '.'.join(list(map(str, p)))
                    config[k] = v
            return config
        from kwutil import util_yaml
        print('Passing string-based config:')
        print(ub.highlight_code(config, 'yaml'))

        # Need to use pyyaml backend, otherwise jsonargparse will balk at the
        # ruamel.yaml types. EVEN THOUGH THEY ARE DUCKTYPED!
        # Rant: People see the mathematical value of typing, and then they take
        # it too far.
        nested = util_yaml.Yaml.loads(config, backend='pyyaml')
        # print('nested = {}'.format(ub.urepr(nested, nl=1)))
        config = nested_to_jsonnest(nested)
        # print('config = {}'.format(ub.urepr(config, nl=1)))

    clikw = {'run': True}
    if config is not None:
        # overload the argument parsing with a programatic config
        clikw['args'] = config
        # Note: we may not need manual mode by setting run to False once we
        # have a deeper understanding of how lightning CLI works.
        # clikw['run'] = False

    default_callbacks = []
    import os

    if os.environ.get('SLURM_JOBID', ''):
        # slurm does not play well with the rich progress bar
        # The default TQDM iter seems to work well enough.
        # from geowatch.utils.lightning_ext.callbacks.progiter_progress import ProgIterProgressBar
        # default_callbacks.append(ProgIterProgressBar())
        ...
    else:
        default_callbacks.append(pl.callbacks.RichProgressBar())
        # pl.callbacks.LearningRateMonitor(logging_interval='step', log_momentum=True),

    default_callbacks.extend([
        pl.callbacks.LearningRateMonitor(logging_interval='epoch', log_momentum=True),
        # pl.callbacks.ModelCheckpoint(monitor='train_loss', mode='min', save_top_k=4),
        # pl.callbacks.ModelCheckpoint(monitor='val_loss', mode='min', save_top_k=4),

        # leaving always on breaks when correspinding metric isnt
        # tracked because loss_weight==0
        # FIXME: can we conditionally apply these if they make sense?
        # or can we make them robust to the case where the key isn't logged?
        # pl.callbacks.ModelCheckpoint(
        #     monitor='val_change_f1', mode='max', save_top_k=4),
        # pl.callbacks.ModelCheckpoint(
        #     monitor='val_saliency_f1', mode='max', save_top_k=4),
        # pl.callbacks.ModelCheckpoint(
        #     monitor='val_class_f1_micro', mode='max', save_top_k=4),
        # pl.callbacks.ModelCheckpoint(
        #     monitor='val_class_f1_macro', mode='max', save_top_k=4),
    ])

    if not DDP_WORKAROUND:
        # FIXME: Why aren't the rank zero checks enough here?

        try:
            # There has to be a tool with less dependencies the matplotlib
            # auto-plotters can hook into.
            import tensorboard  # NOQA
        except ImportError:
            import rich
            rich.print('[yellow]warning: tensorboard not available')
        else:
            # Only use tensorboard if we have it.
            default_callbacks.append(pl_ext.callbacks.TensorboardPlotter())

        default_callbacks.append(pl_ext.callbacks.LightningTelemetry())
    else:
        # TODO: write the redraw script at the start
        # pl_ext.callbacks.TensorboardPlotter()
        ...

    # NOTE: We want to be able to swap the dataloader, but jsonargparse is
    # becoming untenable. I think we just need do a rewrite with regular
    # lightning, I'm pretty over LightningCLI. Its too intrusive.

    # from kwcoco_dataloader.tasks.fusion.datamodules.kwcoco_datamodule import KWCocoVideoDataModule
    datamodule_class = KWCocoVideoDataModule

    cli = SmartLightningCLI(
        model_class=pl.LightningModule,  # TODO: factor out common components of the two models and put them in base class models inherit from
        datamodule_class=datamodule_class,
        trainer_class=SmartTrainer,
        subclass_mode_model=True,

        # save_config_overwrite=True,
        save_config_kwargs={
            'overwrite': True,
        },

        # subclass_mode_data=True,
        parser_kwargs=dict(
            parser_mode='yaml_unsafe_for_tuples',
            error_handler=None,
            exit_on_error=False,
        ),
        trainer_defaults=dict(
            # The following works, but it might be better to move some of these callbacks into the cli
            # (https://pytorch-lightning.readthedocs.io/en/latest/cli/lightning_cli_expert.html#configure-forced-callbacks)
            # Another option is to have a base_config.yaml that includes these, which would make them fully configurable
            # without modifying source code.
            # TODO: find good way to reenable profiling, but not by default
            # profiler=pl.profilers.AdvancedProfiler(dirpath=".", filename="perf_logs"),

            callbacks=default_callbacks,
        ),
        **clikw,
    )
    return cli


def main(config=None):
    """
    Thin wrapper around :func:`make_cli`.

    Args:
        config (None | Dict):
            if specified disables sys.argv usage and executes a training run
            with the specified config.

    CommandLine:
        xdoctest -m geowatch.tasks.fusion.fit_lightning main:0

    Example:
        >>> import os
        >>> from geowatch.utils.lightning_ext.monkeypatches import disable_lightning_hardware_warnings
        >>> from geowatch.tasks.fusion.fit_lightning import *  # NOQA
        >>> disable_lightning_hardware_warnings()
        >>> dpath = ub.Path.appdir('geowatch/tests/test_fusion_fit/demo_main_noop').delete().ensuredir()
        >>> config = {
        >>>     'subcommand': 'fit',
        >>>     'fit.model': 'geowatch.tasks.fusion.methods.noop_model.NoopModel',
        >>>     'fit.trainer.default_root_dir': os.fspath(dpath),
        >>>     'fit.data.train_dataset': 'special:vidshapes2-frames9-gsize32',
        >>>     'fit.data.vali_dataset': 'special:vidshapes1-frames9-gsize32',
        >>>     'fit.data.chip_dims': 32,
        >>>     'fit.trainer.accelerator': 'cpu',
        >>>     'fit.trainer.devices': 1,
        >>>     'fit.trainer.max_steps': 2,
        >>>     'fit.trainer.num_sanity_val_steps': 0,
        >>>     'fit.trainer.add_to_registery': 0,
        >>> }
        >>> cli = main(config=config)

    Example:
        >>> import os
        >>> from geowatch.utils.lightning_ext.monkeypatches import disable_lightning_hardware_warnings
        >>> from geowatch.tasks.fusion.fit_lightning import *  # NOQA
        >>> disable_lightning_hardware_warnings()
        >>> dpath = ub.Path.appdir('geowatch/tests/test_fusion_fit/demo_main_heterogeneous').delete().ensuredir()
        >>> config = {
        >>>     # 'model': 'geowatch.tasks.fusion.methods.MultimodalTransformer',
        >>>     #'model': 'geowatch.tasks.fusion.methods.UNetBaseline',
        >>>     'subcommand': 'fit',
        >>>     'fit.model.class_path': 'geowatch.tasks.fusion.methods.heterogeneous.HeterogeneousModel',
        >>>     'fit.optimizer.class_path': 'torch.optim.SGD',
        >>>     'fit.optimizer.init_args.lr': 1e-3,
        >>>     'fit.trainer.default_root_dir': os.fspath(dpath),
        >>>     'fit.data.train_dataset': 'special:vidshapes2-gsize64-frames9-speed0.5-multispectral',
        >>>     'fit.data.vali_dataset': 'special:vidshapes1-gsize64-frames9-speed0.5-multispectral',
        >>>     'fit.data.chip_dims': 64,
        >>>     'fit.trainer.accelerator': 'cpu',
        >>>     'fit.trainer.devices': 1,
        >>>     'fit.trainer.max_steps': 2,
        >>>     'fit.trainer.num_sanity_val_steps': 0,
        >>>     'fit.trainer.add_to_registery': 0,
        >>> }
        >>> main(config=config)

    Ignore:
        ...

        # export stats
        from geowatch.utils.lightning_ext.monkeypatches import disable_lightning_hardware_warnings
        from geowatch.tasks.fusion.fit_lightning import *  # NOQA
        from kwutil.util_yaml import Yaml
        disable_lightning_hardware_warnings()

        def export_dataset_stats(cli):
            input_stats = cli.trainer.datamodule.dataset_stats['input_stats']
            rows = []
            for sensorchan, stats in input_stats.items():
                mean = list(map(float, stats['mean'].ravel().tolist()))
                std = list(map(float, stats['std'].ravel().tolist()))
                row = {
                    'sensor': sensorchan[0],
                    'channels': sensorchan[1],
                    'mean': Yaml.InlineList(mean),
                    'std': Yaml.InlineList(std),
                }
                rows.append(row)
            from kwutil import util_yaml
            import ruamel.yaml
            import io
            file = io.StringIO()
            ruamel.yaml.round_trip_dump(rows, file, Dumper=ruamel.yaml.RoundTripDumper)
            print(file.getvalue())
        dataset_stats = ub.codeblock(
            '''
            - sensor: '*'
              channels: r|g|b
              mean: [87.572401, 87.572401, 87.572401]
              std: [99.449996, 99.449996, 99.449996]
            ''')
        dpath = ub.Path.appdir('geowatch/tests/test_fusion_fit/demo_main_noop').delete().ensuredir()
        config = {
            'subcommand': 'fit',
            'fit.model': 'geowatch.tasks.fusion.methods.noop_model.NoopModel',
            'fit.trainer.default_root_dir': dpath,
            'fit.data.train_dataset': 'special:vidshapes4-frames9-gsize32',
            'fit.data.vali_dataset': 'special:vidshapes1-frames9-gsize32',
            'fit.data.window_dims': 32,
            'fit.data.dataset_stats': dataset_stats,
            'fit.trainer.max_steps': 2,
            'fit.trainer.num_sanity_val_steps': 0,
        }
        cli = main(config=config)


    """
    cli = make_cli(config)
    return cli


if __name__ == "__main__":
    r"""
    CommandLine:
        python -m geowatch.tasks.fusion.fit_lightning fit --help

        python -m geowatch.tasks.fusion.fit_lightning fit \
                --model.help=MultimodalTransformer

        python -m geowatch.tasks.fusion.fit_lightning fit \
                --model.help=NoopModel

        # Simple run CLI style
        python -m geowatch.tasks.fusion.fit_lightning fit \
            --data.train_dataset=special:vidshapes8-frames9-speed0.5 \
            --data.window_dims=64 \
            --data.workers=4 \
            --trainer.accelerator=gpu \
            --trainer.devices=0, \
            --data.batch_size=1 \
            --model.class_path=MultimodalTransformer \
            --optimizer.class_path=torch.optim.Adam \
            --trainer.default_root_dir ./demo_train

        # Simple run YAML config CLI style
        srun \
        python -m geowatch.tasks.fusion.fit_lightning fit --config="
            data:
                train_dataset: special:vidshapes8-frames9-speed0.5
                window_dims: 64
                num_workers: 4
                batch_size: 4
                normalize_inputs:
                    input_stats:
                        - sensor: '*'
                          channels: r|g|b
                          video: video1
                          mean: [87.572401, 87.572402, 87.572403]
                          std: [99.449997, 99.449998, 99.449999]
            model:
                class_path: MultimodalTransformer
            optimizer:
                class_path: torch.optim.Adam
            trainer:
                accelerator: gpu
                devices: 1
                default_root_dir: ./demo_train
        "

        # Multi GPU run with DDP and CLI config
        python -m geowatch.tasks.fusion.fit_lightning fit \
            --data.train_dataset=special:vidshapes8-frames9-speed0.5 \
            --data.window_dims=64 \
            --data.workers=4 \
            --trainer.accelerator=gpu \
            --trainer.strategy=ddp \
            --trainer.devices=0,1 \
            --data.batch_size=4 \
            --model.class_path=HeterogeneousModel \
            --optimizer.class_path=torch.optim.Adam \
            --trainer.default_root_dir ./demo_train

        # Multi GPU run with DDP and YAML config
        python -m geowatch.tasks.fusion.fit_lightning fit --config="
            data:
                train_dataset: special:vidshapes8-frames9-speed0.5
                window_dims: 64
                workers: 4
                batch_size: 4
                normalize_inputs:
                    input_stats:
                        - sensor: '*'
                          channels: r|g|b
                          video: video1
                          mean: [87.572401, 87.572401, 87.572401]
                          std: [99.449996, 99.449996, 99.449996]
            model:
                class_path: HeterogeneousModel
            optimizer:
                class_path: torch.optim.Adam
            trainer:
                accelerator: gpu
                strategy: ddp
                devices: 0,1
                default_root_dir: ./demo_train
        "

        # Note: setting fast_dev_run seems to disable directory output.

        python -m geowatch.tasks.fusion.fit_lightning fit \
            --data.train_dataset=special:vidshapes8-frames9-speed0.5-multispectral \
            --trainer.accelerator=gpu \
            --trainer.devices=0, \
            --trainer.precision=16 \
            --trainer.fast_dev_run=5 \
            --model=NoopModel
    """
    main()
