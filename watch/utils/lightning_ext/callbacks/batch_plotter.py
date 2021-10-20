# -*- coding: utf-8 -*-
import pytorch_lightning as pl
import ubelt as ub
import numpy as np
from os.path import join
import kwimage
import pytimeparse
import numbers
import datetime

try:
    import xdev
    profile = xdev.profile
except Exception:
    profile = ub.identity

__all__ = ['BatchPlotter']


class BatchPlotter(pl.callbacks.Callback):
    """
    These are callbacks used to monitor the training.

    To be used, the trainer datamodule must have a `draw_batch` method that
    returns an ndarray to draw a batch.

    Args:
        num_draw (int):
            number of batches to draw at the start of each epoch

        draw_interval (datetime.timedelta | str | numbers.Number):
            This is the amount of time to wait before drawing the next batch
            item within an epoch. Can be given as a timedelta, a string
            parsable by `pytimeparse` (e.g.  '1m') or a numeric number of
            seconds.

    TODO:
        - [ ] Doctest

    Example:
        >>> #
        >>> from watch.utils.lightning_ext.callbacks.batch_plotter import *  # NOQA
        >>> from watch.utils.lightning_ext import demo
        >>> self = demo.LightningToyNet2d(num_train=55)
        >>> default_root_dir = ub.ensure_app_cache_dir('lightning_ext/tests/TensorboardPlotter')
        >>> #
        >>> trainer = pl.Trainer(callbacks=[BatchPlotter()],
        >>>                      default_root_dir=default_root_dir,
        >>>                      max_epochs=3)
        >>> trainer.fit(self)
        >>> import pathlib
        >>> train_dpath = pathlib.Path(trainer.log_dir)
        >>> list((train_dpath / 'monitor').glob('*'))
        >>> print('trainer.logger.log_dir = {!r}'.format(train_dpath))

    Ignore:
        >>> from watch.tasks.fusion.fit import make_lightning_modules # NOQA
        >>> args = None
        >>> cmdline = False
        >>> kwargs = {
        ...     'train_dataset': 'special:vidshapes8-multispectral',
        ...     'datamodule': 'KWCocoVideoDataModule',
        ... }
        >>> modules = make_lightning_modules(args=None, cmdline=cmdline, **kwargs)

    References:
        https://pytorch-lightning.readthedocs.io/en/latest/extensions/callbacks.html
    """

    def __init__(self, num_draw=4, draw_interval='1m'):
        super().__init__()
        self.num_draw = num_draw

        if isinstance(draw_interval, datetime.timedelta):
            delta = draw_interval
            num_seconds = delta.total_seconds()
        elif isinstance(draw_interval, numbers.Number):
            num_seconds = draw_interval
        else:
            num_seconds = pytimeparse.parse(draw_interval)
            if num_seconds is None:
                raise ValueError(f'{draw_interval} is not a parsable delta')

        self.draw_interval_seconds = num_seconds
        self.draw_timer = None

    def setup(self, trainer, pl_module, stage):
        self.draw_timer = ub.Timer().tic()

    @classmethod
    def add_argparse_args(cls, parent_parser):
        """
        Example:
            >>> from watch.utils.lightning_ext.callbacks.batch_plotter import *  # NOQA
            >>> from watch.utils.configargparse_ext import ArgumentParser
            >>> cls = BatchPlotter
            >>> parent_parser = ArgumentParser(formatter_class='defaults')
            >>> cls.add_argparse_args(parent_parser)
            >>> parent_parser.print_help()
            >>> parent_parser.parse_known_args()
        """
        from watch.utils.lightning_ext import argparse_ext
        arg_infos = argparse_ext.parse_docstring_args(cls)
        argparse_ext.add_arginfos_to_parser(parent_parser, arg_infos)
        return parent_parser

    # def compute_model_cfgstr(self, model):
    #     type(model)

    # @classmethod
    # TODO
    # def demo(cls):
    #     utils()

    @profile
    def draw_batch(self, trainer, outputs, batch, batch_idx):
        from watch.utils import util_kwimage

        model = trainer.model
        # TODO: get step number
        if hasattr(model, 'get_cfgstr'):
            model_cfgstr = model.get_cfgstr()
        else:
            from watch.utils.slugify_ext import smart_truncate
            model_config = {
                'type': str(model.__class__),
                'hp': smart_truncate(ub.repr2(model.hparams, compact=1, nl=0), max_length=8),
            }
            model_cfgstr = smart_truncate(ub.repr2(
                model_config, compact=1, nl=0), max_length=64)

        datamodule = trainer.datamodule
        if datamodule is None:
            # must have datamodule to draw batches
            canvas = kwimage.draw_text_on_image({'width': 512, 'height': 512}, 'Implement draw_batch in your datamodule', org=(1, 1))
        else:
            canvas = datamodule.draw_batch(batch, outputs=outputs)

        canvas = np.nan_to_num(canvas)

        stage = trainer.state.stage.lower()
        epoch = trainer.current_epoch
        step = trainer.global_step

        if stage.startswith('val'):
            title = f'{stage}_bx{batch_idx:04d}_epoch{epoch:08d}_step{step:08d}'
        else:
            title = f'{stage}_epoch{epoch:08d}_step{step:08d}_bx{batch_idx:04d}'

        canvas = util_kwimage.draw_header_text(
            image=canvas,
            text=f'{model_cfgstr}',
            stack=True,
        )

        canvas = util_kwimage.draw_header_text(image=canvas, text=title,
                                               stack=True)

        dump_dpath = ub.ensuredir((trainer.log_dir, 'monitor', stage, 'batch'))
        dump_fname = f'pred_{title}.jpg'
        fpath = join(dump_dpath, dump_fname)
        # print('write to fpath = {!r}'.format(fpath))
        kwimage.imwrite(fpath, canvas)

    def draw_if_ready(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
        do_draw = batch_idx < self.num_draw
        if self.draw_interval_seconds > 0:
            do_draw |= self.draw_timer.toc() > self.draw_interval_seconds
        if do_draw:
            self.draw_batch(trainer, outputs, batch, batch_idx)
            self.draw_timer.tic()

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
        self.draw_if_ready(trainer, pl_module, outputs, batch, batch_idx, dataloader_idx)

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
        self.draw_if_ready(trainer, pl_module, outputs, batch, batch_idx, dataloader_idx)

    def on_test_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
        self.draw_if_ready(trainer, pl_module, outputs, batch, batch_idx, dataloader_idx)
