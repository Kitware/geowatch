"""
Main entrypoint for a fusion fit job.

For unit tests see:
    ../../../tests/test_lightning_cli_fit.py

For tutorials see:
    ../../../docs/source/manual/tutorial/tutorial1_rgb_network.sh
"""
# TODO: lets avoid the import * here.
from geowatch.tasks.fusion.methods import *  # NOQA

from geowatch.tasks.fusion.datamodules.kwcoco_datamodule import KWCocoVideoDataModule
from geowatch.utils.lightning_ext.lightning_cli_ext import LightningCLI_Extension
from geowatch.utils.lightning_ext.lightning_cli_ext import LightningArgumentParser
from geowatch.utils import lightning_ext as pl_ext

import pytorch_lightning as pl
import ubelt as ub
from torch import optim
from typing import Any

import yaml
from jsonargparse import set_loader, set_dumper
# from pytorch_lightning.utilities.rank_zero import rank_zero_only

from kwutil import util_environ

from geowatch.monkey import monkey_numpy  # NOQA
# monkey_numpy.patch_numpy_dtypes()
monkey_numpy.patch_numpy_2x()


# FIXME: we should be able to use our callbacks when ddp is enabled.
# Not sure why we get a freeze.
DDP_WORKAROUND = util_environ.envflag('DDP_WORKAROUND')


# Not very safe, but needed to parse tuples e.g. datamodule.dataset_stats
# TODO: yaml.SafeLoader + tuple parsing
def custom_yaml_load(stream):
    return yaml.load(stream, Loader=yaml.FullLoader)


set_loader('yaml_unsafe_for_tuples', custom_yaml_load)


def custom_yaml_dump(data):
    return yaml.dump(data, Dumper=yaml.Dumper)


set_dumper('yaml_unsafe_for_tuples', custom_yaml_dump)


class SmartTrainer(pl.Trainer):
    """
    Simple trainer subclass so we can ensure a print happens directly before
    the training loop. (so annoying that we can't reorder callbacks)
    """

    def __init__(self, *args, add_to_registery=True, **kwargs):
        super().__init__(*args, **kwargs)
        self.add_to_registery = add_to_registery

    def _run_stage(self, *args, **kwargs):
        # All I want is to print this  directly before training starts.
        # Is that so hard to do?
        self._on_before_run()

        super()._run_stage(*args, **kwargs)

    @property
    def log_dpath(self):
        """
        Get path to the the log directory if it exists.
        """
        if self.logger is None:
            raise Exception('cannot get a log_dpath when no logger exists')
        if self.logger.log_dir is None:
            raise Exception('cannot get a log_dpath when logger.log_dir is None')
        return ub.Path(self.logger.log_dir)

    def _on_before_run(self):
        """
        Our custom "callback"
        """
        print(f'self.global_rank={self.global_rank}')
        if self.global_rank == 0:
            # TODO: add condition so tests do not add to the registery
            if self.add_to_registery:
                self._add_to_registery()
            self._write_inspect_helper_scripts()
            self._handle_restart_details()

        if hasattr(self.datamodule, '_notify_about_tasks'):
            # Not sure if this is the best place, but we want datamodule to be
            # able to determine what tasks it should be producing data for.
            # We currently infer this from information in the model.
            self.datamodule._notify_about_tasks(model=self.model)

    def _add_to_registery(self):
        """
        Register this training run with the registery.
        This allows the user to lookup recent via:

        .. code:: bash

            python -m geowatch.tasks.fusion.fit_registery peek

        """
        from geowatch.tasks.fusion.fit_registery import FitRegistery
        registery = FitRegistery()
        dpath = ub.Path(self.logger.log_dir)
        # TODO: more config options here.
        registery.append(
            path=dpath,
        )

    def _handle_restart_details(self):
        """
        Handle chores when restarting from a previous checkpoint.
        """
        if self.ckpt_path:
            print('Detected that you are restarting from a previous checkpoint')
            ckpt_path = ub.Path(self.ckpt_path)
            assert ckpt_path.parent.name == 'checkpoints'
            old_event_fpaths = list(ckpt_path.parent.parent.glob('events.out.tfevents.*'))
            if len(old_event_fpaths):
                print('Copying tensorboard events to new training directory directory')
                for old_fpath in old_event_fpaths:
                    new_fpath = self.log_dpath / old_fpath.name
                    old_fpath.copy(new_fpath)

    def _write_inspect_helper_scripts(self):
        """
        Write helper scripts to the main training log dir that the user can run
        to inspect the status of training. This helps workaround the
        ddp-workaround because we can run tasks that caused hangs indepenently
        of the main train script.
        """
        import rich
        dpath = self.log_dpath
        rich.print(f"Trainer log dpath:\n\n[link={dpath}]{dpath}[/link]\n")
        from geowatch.tasks.fusion import helper_scripts

        try:
            vali_coco_fpath = self.datamodule.vali_dataset.coco_dset.fpath
        except Exception:
            vali_coco_fpath = None

        try:
            train_coco_fpath = self.datamodule.train_dataset.coco_dset.fpath
        except Exception:
            train_coco_fpath = None

        # Generate helper script and write them to disk
        scripts = helper_scripts.train_time_helper_scripts(dpath, train_coco_fpath, vali_coco_fpath)
        for key, script in scripts.items():
            script['fpath'].write_text(script['text'])

        # Add executable permission
        try:
            for key, script in scripts.items():
                fpath = script['fpath']
                # Can use new ubelt
                ub.Path(fpath).chmod('u+x')
        except Exception as ex:
            print('WARNING ex = {}'.format(ub.urepr(ex, nl=1)))


class TorchGlobals(pl.callbacks.Callback):
    """
    Callback to setup torch globals

    Args:
        float32_matmul_precision (str):
            can be 'medium', 'high', 'default', or 'auto'.
            The 'default' value does not change any setting.
            The 'auto' value defaults to 'medium' if the training devices have
                ampere cores.
    """

    def __init__(self, float32_matmul_precision='default'):
        self.float32_matmul_precision = float32_matmul_precision

    def setup(self, trainer, pl_module, stage):
        import torch
        float32_matmul_precision = self.float32_matmul_precision
        if float32_matmul_precision == 'default':
            float32_matmul_precision = None
        elif float32_matmul_precision == 'auto':
            # Detect if we have Ampere tensor cores
            # Ampere (V8) and later leverage tensor cores, where medium
            # float32_matmul_precision becomes useful
            if torch.cuda.is_available():
                device_versions = [torch.cuda.get_device_capability(device_id)[0]
                                   for device_id in trainer.device_ids]
                if all(v >= 8 for v in device_versions):
                    float32_matmul_precision = 'medium'
                else:
                    float32_matmul_precision = None
            else:
                float32_matmul_precision = None
        if float32_matmul_precision is not None:
            torch.set_float32_matmul_precision(float32_matmul_precision)


class WeightInitializer(pl.callbacks.Callback):
    """
    Network weight initializer with support for partial weight loading.

    Attributes:
        init (str | PathLike): either "noop" to use default weight
            initialization, or a path to a pretrained model that has at least
            some similarity to the model to be trained.

        association (str):
            Either "embedding" or "isomorphism". The "embedding" case is more
            flexible allowing similar subcomponents of the network to be
            disconnected.  In the "isomorphism" case, the transfered part must
            be a proper subgraph in both models.  See torch-libertor's partial
            weight initializtion for more details.

        remember_initial_state (bool):
            If True saves the initial state in a "analysis_checkpoints" folder.
            Defaults to True.

        verbose (int):
            if 1 prints some info. If 3 prints the explicit association found.
            if 0, prints nothing.
    """

    def __init__(self, init='noop', association='embedding',
                 remember_initial_state=True, verbose=1):
        self.init = init
        self.association = association
        self.remember_initial_state = remember_initial_state
        self.verbose = verbose
        self._did_once = False

    def setup(self, trainer, pl_module, stage):
        if self._did_once:
            # Only weight initialize once, on whatever stage happens first.
            return
        self._did_once = True

        _verbose = self.verbose and trainer.global_rank == 0
        if _verbose:
            print('🏋 Initializing weights')

        if self.init == 'noop':
            ...
        else:
            from geowatch.tasks.fusion.fit import coerce_initializer
            from kwutil import util_pattern
            initializer = coerce_initializer(self.init)

            model = pl_module

            # Hack to always preserve specific values
            # (TODO add to torch_liberator as an option, allow to be configured
            # as argument to this class, or allow the model to set some property
            # that says which weights should not be touched.)
            old_state = model.state_dict()
            ignore_pattern = util_pattern.MultiPattern.coerce(['*tokenizers*.0.mean', '*tokenizers*.0.std'])
            ignore_keys = [key for key in old_state.keys() if ignore_pattern.match(key)]
            if self.verbose:
                print('Finding keys to not initializer')
            to_preserve = ub.udict(old_state).subdict(ignore_keys).map_values(lambda v: v.clone())

            # TODO: read the config of the model we initialize from and save it
            # so we can remember the lineage.

            initializer.association = self.association
            info = initializer.forward(model)  # NOQA
            if _verbose >= 3:
                if info:
                    mapping = info.get('mapping', None)
                    unset = info.get('self_unset', None)
                    unused = info.get('self_unused', None)
                    print('mapping = {}'.format(ub.urepr(mapping, nl=1)))
                    print(f'unused={unused}')
                    print(f'unset={unset}')

            if _verbose:
                print('🏋 - Finalize weight initialization')
            updated = model.state_dict() | to_preserve
            model.load_state_dict(updated)

        if self.remember_initial_state and trainer.global_rank == 0:
            import torch
            # Remember what the initial state was.  Save this to a directory
            # where we will store checkpoints useful for analysis, but might
            # not be relevant for finding the best model.
            print('Saving initial weights to disk')
            analysis_checkpoint_dpath = (trainer.log_dpath / 'analysis_checkpoints')
            analysis_checkpoint_dpath.ensuredir()
            initial_state_fpath = analysis_checkpoint_dpath / 'initial_state.ckpt'
            if initial_state_fpath.exists():
                raise FileExistsError('Initial state already exists. This should not happen.')
            initial_state = pl_module.state_dict()
            torch.save(initial_state, initial_state_fpath)


class SmartLightningCLI(LightningCLI_Extension):
    """
    Our extension of LightningCLI class that adds custom arguments and
    functionality to the CLI. See :func:`add_arguments_to_parser` for more
    details.
    """

    @staticmethod
    def configure_optimizers(
        lightning_module: pl.LightningModule, optimizer: optim.Optimizer, lr_scheduler=None
    ) -> Any:
        """Override to customize the :meth:`~pytorch_lightning.core.module.LightningModule.configure_optimizers`
        method.

        TODO: why is this overloaded?

        Args:
            lightning_module: A reference to the model.
            optimizer: The optimizer.
            lr_scheduler: The learning rate scheduler (if used).
        """

        if lr_scheduler is None:
            return optimizer

        if isinstance(lr_scheduler, pl.cli.ReduceLROnPlateau):
            return {
                "optimizer": optimizer,
                "lr_scheduler": {"scheduler": lr_scheduler, "monitor": lr_scheduler.monitor},
            }

        if isinstance(lr_scheduler, optim.lr_scheduler.OneCycleLR):
            return {
                "optimizer": optimizer,
                "lr_scheduler": {"scheduler": lr_scheduler, 'interval': 'step'},
            }

        return [optimizer], [lr_scheduler]

    def add_arguments_to_parser(self, parser: LightningArgumentParser):
        """
        1. Adds custom extensions like "initializer" and "torch_globals"
        2. Adds the packager callback (not sure why this is having a problem when used as a real callback)
        3. Helps the dataset / model notify each other about relevant settings.
        """

        # TODO: separate final_package dir and fpath for more configuration
        # pl_ext.callbacks.Packager(package_fpath=args.package_fpath),
        parser.add_lightning_class_args(pl_ext.callbacks.Packager, "packager")

        parser.add_lightning_class_args(WeightInitializer, "initializer")

        parser.add_lightning_class_args(TorchGlobals, "torch_globals")

        if not DDP_WORKAROUND:
            # Fixme: disabled for multi-gpu training
            # Force disable batch plotter when ddp is enabled,
            parser.add_lightning_class_args(pl_ext.callbacks.BatchPlotter, "batch_plotter")

        # pl_ext.callbacks.BatchPlotter(  # Fixme: disabled for multi-gpu training with deepspeed
        #     num_draw=2,  # args.num_draw,
        #     draw_interval="5min",  # args.draw_interval
        # ),

        # parser.set_defaults({"packager.package_fpath": "???"}) # "$DEFAULT_ROOT_DIR"/final_package.pt
        parser.link_arguments(
            "trainer.default_root_dir",
            "packager.package_fpath",
            compute_fn=_final_pkg_compute_fn,
            # lambda root: None if root is None else str(ub.Path(root) / "final_package.pt")
            # apply_on="instantiate",
        )

        # Reference:
        # https://github.com/omni-us/jsonargparse/issues/170
        # https://github.com/omni-us/jsonargparse/pull/326
        # Ensure the datamodule
        if hasattr(parser, 'add_instantiator'):
            parser.add_instantiator(
                instantiate_datamodule,
                class_type=pl.LightningDataModule
            )

        # pass dataset stats to model after initialization datamodule
        parser.link_arguments(
            "data",
            "model.init_args.dataset_stats",
            compute_fn=_data_value_getter('dataset_stats'),
            apply_on="instantiate")

        # TODO: we don't want to force the model to take a classes argument.
        # The classes a model predicts is really a property of one or more of
        # its heads. We need to determine a better way to have the dataloader
        # notify the appropriate model head about what classes it can produce
        # training examples for.
        parser.link_arguments(
            "data",
            "model.init_args.classes",
            compute_fn=_data_value_getter('predictable_classes'),
            apply_on="instantiate")

        super().add_arguments_to_parser(parser)


def instantiate_datamodule(cls, *args, **kwargs):
    """
    Custom instantiator for the datamodule that simply calls setup after
    creating the instance.
    """
    self = cls(*args, **kwargs)
    if not self.did_setup:
        self.setup('fit')
    return self


def _final_pkg_compute_fn(root):
    # cant be a lambda for pickle
    return None if root is None else str(ub.Path(root) / "final_package.pt")


class _ValueGetter:
    # Made into a class instead of a closure for pickling issues
    def __init__(self, key):
        self.key = key

    def __call__(self, data):
        if not data.did_setup:
            data.setup('fit')

        if data.config.num_workers not in {0, '0'}:
            # Hack to disconnect SQL coco databases before we fork
            # TODO: find a way to handle this more ellegantly
            if hasattr(data, 'coco_datasets'):
                for k, v in data.coco_datasets.items():
                    if hasattr(v, 'disconnect'):
                        if v.session is not None:
                            v.session.close()
                        v.disconnect()
        return getattr(data, self.key)


def _data_value_getter(key):
    # Hack to call setup on the datamodule before linking args
    get_value = _ValueGetter(key)
    return get_value


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

    cli = SmartLightningCLI(
        model_class=pl.LightningModule,  # TODO: factor out common components of the two models and put them in base class models inherit from
        datamodule_class=KWCocoVideoDataModule,
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
