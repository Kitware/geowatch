# Import models for the CLI registry
from watch.tasks.fusion.methods import *  # NOQA
from watch.tasks.fusion.datamodules.kwcoco_datamodule import KWCocoVideoDataModule
from watch.utils.lightning_ext.lightning_cli_ext import LightningCLI_Extension
from watch.utils import lightning_ext as pl_ext

import pytorch_lightning as pl
import ubelt as ub
from torch import optim
from typing import Optional, Any

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
        lightning_module: pl.LightningModule, optimizer: optim.Optimizer, lr_scheduler = None
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

def main():
    SmartLightningCLI(
        model_class=pl.LightningModule,  # TODO: factor out common components of the two models and put them in base class models inherit from
        # MultimodalTransformer,
        datamodule_class=KWCocoVideoDataModule,
        subclass_mode_model=True,
        # subclass_mode_data=True,
        parser_kwargs=dict(parser_mode='yaml_unsafe_for_tuples'),
        trainer_defaults=dict(
            # The following works, but it might be better to move some of these callbacks into the cli
            # (https://pytorch-lightning.readthedocs.io/en/latest/cli/lightning_cli_expert.html#configure-forced-callbacks)
            # Another option is to have a base_config.yaml that includes these, which would make them fully configurable
            # without modifying source code.
            profiler=pl.profilers.AdvancedProfiler(dirpath=".", filename="perf_logs"),
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
    )


if __name__ == "__main__":
    r"""
    CommandLine:
        python -m watch.tasks.fusion.fit_lightning fit \
                --model.help=MultimodalTransformer

        python -m watch.tasks.fusion.fit_lightning fit \
                --model.help=NoopModel

        python -m watch.tasks.fusion.fit_lightning fit \
            --data.train_dataset=special:vidshapes8-frames9-speed0.5-multispectral \
            --trainer.accelerator=gpu --trainer.devices=0, \
            --trainer.precision=16  \
            --trainer.fast_dev_run=5 \
            --model=MultimodalTransformer \
            --model.tokenizer=linconv \
            --trainer.default_root_dir ./demo_train

        # Note: setting fast_dev_run seems to disable directory output.

        python -m watch.tasks.fusion.fit_lightning fit \
            --data.train_dataset=special:vidshapes8-frames9-speed0.5-multispectral \
            --trainer.accelerator=gpu \
            --trainer.devices=0, \
            --trainer.precision=16 \
            --trainer.fast_dev_run=5 \
            --model=NoopModel\
            --model.tokenizer=linconv
    """
    main()
