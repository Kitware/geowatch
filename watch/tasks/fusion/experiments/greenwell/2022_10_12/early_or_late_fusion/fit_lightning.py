from watch.tasks.fusion.datamodules.kwcoco_datamodule import KWCocoVideoDataModule
# Import models for the CLI registry
from watch.tasks.fusion.methods import SequenceAwareModel  # NOQA
from watch.tasks.fusion.methods import MultimodalTransformer  # NOQA

"""
The Wrapped class below are examples of why we should eventually factor out the current configuraiton system. LightningCLI interogates the __init__ methods belonging to LightningModule and LightningDataModule to decide which parameters can be configured.
"""


def main():
    from watch.utils.lightning_ext.lightning_cli_ext import LightningCLI_Extension
    import pytorch_lightning as pl
    from watch.utils import lightning_ext as pl_ext
    import ubelt as ub

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

    class MyLightningCLI(LightningCLI_Extension):

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

    MyLightningCLI(
        # SequenceAwareModel,
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
            ]
        ),
    )


if __name__ == "__main__":
    r"""
    CommandLine:
        python -m watch.tasks.fusion.fit_lightning fit \
                --model.help=MultimodalTransformer

        python -m watch.tasks.fusion.fit_lightning fit \
                --model.help=SequenceAwareModel

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
            --model=SequenceAwareModel \
            --model.tokenizer=linconv
    """
    main()
