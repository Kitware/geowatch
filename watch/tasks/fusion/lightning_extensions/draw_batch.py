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


class DrawBatchCallback(pl.callbacks.Callback):
    """
    These are callbacks used to monitor the training.

    To be used, the trainer datamodule must have a `draw_batch` method that
    returns an ndarray to draw a batch.

    Args:
        num_draw (int):
            number of batches to draw at the start of each epoch

        draw_interval (datetime.timedelta | str | numbers.Number, default='10m'):
            This is the amount of time to wait before drawing the next batch
            item within an epoch. Can be given as a timedelta, a string
            parsable by `pytimeparse` (e.g.  '1m') or a numeric number of
            seconds.

    TODO:
        - [ ] Doctest

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

    @profile
    def draw_batch(self, trainer, outputs, batch, batch_idx):

        datamodule = trainer.datamodule
        if datamodule is None:
            # must have datamodule to draw batches
            return

        canvas = datamodule.draw_batch(batch, outputs=outputs)

        canvas = np.nan_to_num(canvas)

        stage = trainer.state.stage.lower()
        epoch = trainer.current_epoch

        canvas = kwimage.draw_text_on_image(
            canvas, f'{stage}_epoch{epoch:08d}_bx{batch_idx:04d}', org=(1, 1),
            valign='top')

        dump_dpath = ub.ensuredir((trainer.log_dir, 'monitor', stage, 'batch'))
        dump_fname = f'pred_{stage}_epoch{epoch:08d}_bx{batch_idx:04d}.jpg'
        fpath = join(dump_dpath, dump_fname)
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
