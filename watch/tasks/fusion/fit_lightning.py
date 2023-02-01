# Import models for the CLI registry
from watch.tasks.fusion.methods import *  # NOQA
from watch.tasks.fusion.datamodules.kwcoco_datamodule import KWCocoVideoDataModule
from watch.utils.lightning_ext.lightning_cli_ext import LightningCLI_Extension
from watch.utils import lightning_ext as pl_ext

import pytorch_lightning as pl
import ubelt as ub
from torch import optim
from typing import Any

import yaml
from jsonargparse import set_loader, set_dumper
# , lazy_instance


# Not very safe, but needed to parse tuples e.g. datamodule.dataset_stats
# TODO: yaml.SafeLoader + tuple parsing
def custom_yaml_load(stream):
    return yaml.load(stream, Loader=yaml.FullLoader)
set_loader('yaml_unsafe_for_tuples', custom_yaml_load)


def custom_yaml_dump(data):
    return yaml.dump(data, Dumper=yaml.Dumper)
set_dumper('yaml_unsafe_for_tuples', custom_yaml_dump)


# class PartialWeightInitializer(pl.callbacks.Callback):

#     def __init__(self, init='noop'):
#         ...

#     def on_fit_start(self, trainer, pl_module):
#         if 0:
#             # TODO: make this work
#             import ubelt as ub
#             from watch.tasks.fusion.fit import coerce_initializer
#             from watch.utils import util_pattern
#             init_fpath = '/home/joncrall/remote/toothbrush/data/dvc-repos/smart_expt_dvc/models/fusion/Drop4-BAS/packages/Drop4_TuneV323_BAS_30GSD_BGRNSH_V2/package_epoch0_step41.pt.pt'
#             initializer = coerce_initializer(init_fpath)
#             # other_model = getattr(initializer, 'other_model', None)

#             # Hack to preserve specific values
#             print('Initializing weights')
#             old_state = pl_module.state_dict()
#             ignore_pattern = util_pattern.MultiPattern.coerce(['*tokenizers*.0.mean', '*tokenizers*.0.std'])
#             ignore_keys = [key for key in old_state.keys() if ignore_pattern.match(key)]
#             print('Finding keys to not initializer')
#             to_preserve = ub.udict(old_state).subdict(ignore_keys).map_values(lambda v: v.clone())

#             initializer.association = 'embedding'
#             info = initializer.forward(pl_module)  # NOQA
#             if info:
#                 mapping = info.get('mapping', None)
#                 unset = info.get('self_unset', None)
#                 unused = info.get('self_unused', None)
#                 print('mapping = {}'.format(ub.repr2(mapping, nl=1)))
#                 print(f'unused={unused}')
#                 print(f'unset={unset}')


class SmartLightningCLI(LightningCLI_Extension):

    @staticmethod
    def configure_optimizers(
        lightning_module: pl.LightningModule, optimizer: optim.Optimizer, lr_scheduler=None
    ) -> Any:
        """Override to customize the :meth:`~pytorch_lightning.core.module.LightningModule.configure_optimizers`
        method.

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

    # TODO: import initialization code from fit.py
    def add_arguments_to_parser(self, parser):
        # TODO: separate final_package dir and fpath for more configuration
        # pl_ext.callbacks.Packager(package_fpath=args.package_fpath),
        parser.add_lightning_class_args(pl_ext.callbacks.Packager, "packager")
        # parser.set_defaults({"packager.package_fpath": "???"}) # "$DEFAULT_ROOT_DIR"/final_package.pt
        parser.link_arguments(
            "trainer.default_root_dir",
            "packager.package_fpath",
            compute_fn=lambda root: None if root is None else str(ub.Path(root) / "final_package.pt")
            # apply_on="instantiate",
        )

        parser.add_argument(
            '--profile',
            action='store_true',
            help=ub.paragraph(
                '''
                Fit does nothing with this flag. This just allows for `@xdev.profile`
                profiling which checks sys.argv separately.

                DEPRECATED: there is no longer any reason to use this. Set the
                XDEV_PROFILE environment variable instead.
                '''))

        def data_value_getter(key):
            # Hack to call setup on the datamodule before linking args
            def get_value(data):
                if not data.did_setup:
                    data.setup('fit')
                return getattr(data, key)
            return get_value

        # pass dataset stats to model after initialization datamodule
        parser.link_arguments(
            "data",
            "model.init_args.dataset_stats",
            compute_fn=data_value_getter('dataset_stats'),
            apply_on="instantiate")
        parser.link_arguments(
            "data",
            "model.init_args.classes",
            compute_fn=data_value_getter('classes'),
            apply_on="instantiate")

        super().add_arguments_to_parser(parser)


def make_cli(config=None):

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
                    # print('--' + k + '=' + str(v) + ' \\\\')
            return config
        from watch.utils import util_yaml
        print(ub.highlight_code(config, 'yaml'))
        nested = util_yaml.yaml_loads(config, backend='pyyaml')
        print('nested = {}'.format(ub.urepr(nested, nl=1)))
        config = nested_to_jsonnest(nested)
        print('config = {}'.format(ub.urepr(config, nl=1)))

    manual_mode = False
    clikw = {'run': True}
    if config is not None:
        # overload the argument parsing with a programatic config
        clikw['args'] = config
        clikw['run'] = False
        # Note: we may not need manual mode once we have a deeper understanding
        # of how lightning CLI works.
        manual_mode = True

    cli = SmartLightningCLI(
        model_class=pl.LightningModule,  # TODO: factor out common components of the two models and put them in base class models inherit from
        datamodule_class=KWCocoVideoDataModule,
        subclass_mode_model=True,
        # subclass_mode_data=True,
        parser_kwargs=dict(
            parser_mode='yaml_unsafe_for_tuples',
            error_handler=None,
        ),
        trainer_defaults=dict(
            # The following works, but it might be better to move some of these callbacks into the cli
            # (https://pytorch-lightning.readthedocs.io/en/latest/cli/lightning_cli_expert.html#configure-forced-callbacks)
            # Another option is to have a base_config.yaml that includes these, which would make them fully configurable
            # without modifying source code.
            # TODO: find good way to reenable profiling, but not by default
            # profiler=pl.profilers.AdvancedProfiler(dirpath=".", filename="perf_logs"),

            callbacks=[
                pl_ext.callbacks.BatchPlotter(  # Fixme: disabled for multi-gpu training with deepspeed
                    num_draw=2,  # args.num_draw,
                    draw_interval="5min",  # args.draw_interval
                ),
                pl.callbacks.LearningRateMonitor(logging_interval='step', log_momentum=True),

                pl.callbacks.ModelCheckpoint(monitor='train_loss', mode='min', save_top_k=1),
                # leaving always on breaks when correspinding metric isnt
                # tracked because loss_weight==0
                # pl.callbacks.ModelCheckpoint(
                #     monitor='val_change_f1', mode='max', save_top_k=4),
                # pl.callbacks.ModelCheckpoint(
                #     monitor='val_saliency_f1', mode='max', save_top_k=4),
                # pl.callbacks.ModelCheckpoint(
                #     monitor='val_class_f1_micro', mode='max', save_top_k=4),
                # pl.callbacks.ModelCheckpoint(
                #     monitor='val_class_f1_macro', mode='max', save_top_k=4),
            ]
        ),
        **clikw,
    )
    cli.manual_mode = manual_mode
    return cli


def main(config=None):
    """
    Args:
        config (None | Dict):
            if specified disables sys.argv usage and executes a training run
            with the specified config.

    Example:
        >>> from watch.utils.lightning_ext.monkeypatches import disable_lightning_hardware_warnings
        >>> from watch.tasks.fusion.fit_lightning import *  # NOQA
        >>> disable_lightning_hardware_warnings()
        >>> dpath = ub.Path.appdir('watch/tests/test_fusion_fit/demo_main_noop').ensuredir()
        >>> config = {
        >>>     'model': 'NoopModel',
        >>>     'trainer.default_root_dir': dpath,
        >>>     'data.train_dataset': 'special:vidshapes8-frames9-speed0.5-multispectral',
        >>>     'data.vali_dataset': 'special:vidshapes4-frames9-speed0.5-multispectral',
        >>>     'trainer.max_steps': 3,
        >>>     'trainer.num_sanity_val_steps'
        >>> }
        >>> main(config=config)

    Example:
        >>> from watch.utils.lightning_ext.monkeypatches import disable_lightning_hardware_warnings
        >>> from watch.tasks.fusion.fit_lightning import *  # NOQA
        >>> disable_lightning_hardware_warnings()
        >>> dpath = ub.Path.appdir('watch/tests/test_fusion_fit/demo_main_heterogeneous').ensuredir()
        >>> config = {
        >>>     # 'model': 'watch.tasks.fusion.methods.MultimodalTransformer',
        >>>     #'model': 'watch.tasks.fusion.methods.UNetBaseline',
        >>>     'model.class_path': 'watch.tasks.fusion.methods.heterogeneous.HeterogeneousModel',
        >>>     'model.init_args.position_encoder.class_path': 'watch.tasks.fusion.methods.heterogeneous.ScaleAgnostictPositionalEncoder',
        >>>     'model.init_args.position_encoder.init_args.in_dims': 3,
        >>>     'model.init_args.position_encoder.init_args.max_freq': 3,
        >>>     'model.init_args.position_encoder.init_args.num_freqs': 10,
        >>>     'model.init_args.backbone.class_path': 'watch.tasks.fusion.architectures.transformer.TransformerEncoderDecoder',
        >>>     'optimizer.class_path': 'torch.optim.Adam',
        >>>     'optimizer.init_args.lr': 1e-4,
        >>>     'optimizer.init_args.weight_decay': 1e-2,
        >>>     'optimizer.init_args.betas': [0.9, 0.98],
        >>>     'optimizer.init_args.eps': 1e-12,
        >>>     'trainer.default_root_dir': dpath,
        >>>     'data.train_dataset': 'special:vidshapes8-frames9-speed0.5-multispectral',
        >>>     'data.vali_dataset': 'special:vidshapes4-frames9-speed0.5-multispectral',
        >>>     'trainer.max_steps': 3,
        >>>     'trainer.num_sanity_val_steps': 0,
        >>> }
        >>> main(config=config)
    """
    cli = make_cli(config)
    if cli.manual_mode:
        # Do the running ourself
        print('trainer.logger.log_dir = {!r}'.format(cli.trainer.logger.log_dir))
        cli.trainer.fit(cli.model, cli.datamodule)


if __name__ == "__main__":
    r"""
    CommandLine:
        python -m watch.tasks.fusion.fit_lightning fit --help

        python -m watch.tasks.fusion.fit_lightning fit \
                --model.help=MultimodalTransformer

        python -m watch.tasks.fusion.fit_lightning fit \
                --model.help=NoopModel

        python -m watch.tasks.fusion.fit_lightning fit \
            --data.train_dataset=special:vidshapes8-frames9-speed0.5-multispectral \
            --trainer.accelerator=gpu --trainer.devices=0, \
            --trainer.precision=16  \
            --trainer.fast_dev_run=5 \
            --model=HeterogeneousModel \
            --model.tokenizer=linconv \
            --trainer.default_root_dir ./demo_train

        # Note: setting fast_dev_run seems to disable directory output.

        python -m watch.tasks.fusion.fit_lightning fit \
            --data.train_dataset=special:vidshapes8-frames9-speed0.5-multispectral \
            --trainer.accelerator=gpu \
            --trainer.devices=0, \
            --trainer.precision=16 \
            --trainer.fast_dev_run=5 \
            --model=NoopModel
    """
    main()
