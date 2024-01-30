"""
LightningTelemetry callback to interface with torch.package
"""
import pytorch_lightning as pl
import ubelt as ub
# from pytorch_lightning.utilities.rank_zero import rank_zero_only


class LightningTelemetry(pl.callbacks.Callback):
    """
    The idea is that we wrap a fit job with ProcessContext

    Example:
        >>> from geowatch.utils.lightning_ext.callbacks.telemetry import *  # NOQA
        >>> from geowatch.utils.lightning_ext.demo import LightningToyNet2d
        >>> from geowatch.utils.lightning_ext.callbacks import StateLogger
        >>> import ubelt as ub
        >>> from geowatch.utils.lightning_ext.monkeypatches import disable_lightning_hardware_warnings
        >>> disable_lightning_hardware_warnings()
        >>> default_root_dir = ub.Path.appdir('geowatch/lightning_ext/test/telemetry')
        >>> default_root_dir.delete().ensuredir()
        >>> self = LightningTelemetry()
        >>> # Test starting a model without any existing checkpoints
        >>> trainer = pl.Trainer(default_root_dir=default_root_dir, callbacks=[
        >>>     self,
        >>>     StateLogger()
        >>> ], max_epochs=2, accelerator='cpu', devices=1)
        >>> model = LightningToyNet2d()
        >>> trainer.fit(model)
    """

    def __init__(self):
        from geowatch.utils.process_context import ProcessContext
        self.context = ProcessContext(
            name='lightning_fit',
            # TODO: how to get the config here?
            # config=config,
            # track_emissions='offline'
        )

    @classmethod
    def add_argparse_args(cls, parent_parser):
        """
        Example:
            >>> from geowatch.utils.lightning_ext.callbacks.telemetry import *  # NOQA
            >>> from geowatch.utils.configargparse_ext import ArgumentParser
            >>> cls = LightningTelemetry
            >>> parent_parser = ArgumentParser(formatter_class='defaults')
            >>> cls.add_argparse_args(parent_parser)
            >>> parent_parser.print_help()
        """
        from geowatch.utils.lightning_ext import argparse_ext
        arg_infos = argparse_ext.parse_docstring_args(cls)
        argparse_ext.add_arginfos_to_parser(parent_parser, arg_infos)
        return parent_parser

    def setup(self, trainer, pl_module, stage=None):
        """
        Finalize initialization step.
        Resolve the paths where files will be written.

        Args:
            trainer (pl.Trainer):
            pl_module (pl.LightningModule):
            stage (str | None):

        Returns:
            None
        """
        if trainer.is_global_zero:
            self._after_initialization(trainer)

    def _after_initialization(self, trainer):
        if trainer.is_global_zero:
            print('initialize process context')
            root_dir = ub.Path(trainer.default_root_dir)
            print('root_dir = {!r}'.format(root_dir))
            self.context.add_disk_info(root_dir)
            # trainer.logger.log_dir

    def on_fit_start(self, trainer: 'pl.Trainer', pl_module: 'pl.LightningModule') -> None:
        """
        TODO:
            - [ ] Write out the uninitialized topology
        """
        if not trainer.is_global_zero:
            return
        self.context.start()
        self._dump(trainer)

    def on_fit_end(self, trainer: 'pl.Trainer', pl_module: 'pl.LightningModule') -> None:
        if not trainer.is_global_zero:
            return
        self._dump(trainer)

    # Causes ddp hang
    # def on_train_epoch_end(self, trainer, logs=None):
    #     if trainer.global_rank != 0:
    #         return
    #     print('Training is complete, dumping telemetry')
    #     self._dump(trainer)

    def on_exception(self, trainer: 'pl.Trainer', pl_module: 'pl.LightningModule', *args, **kw) -> None:
        if trainer.global_rank != 0:
            return
        print('Exception, dumping telemetry')
        self._dump(trainer)

    def _dump(self, trainer):
        if not trainer.is_global_zero:
            return
        if trainer.log_dir is None:
            print('Trainer run without a log_dir, cannot dump telemetry')
            return
        import json
        log_dpath = ub.Path(trainer.logger.log_dir)
        obj = self.context.flush()
        tel_fpath = log_dpath / 'telemetry.json'
        tel_fpath.write_text(json.dumps(obj))
