import kwimage
import numpy as np
import pytorch_lightning as pl
import ubelt as ub
import warnings
import traceback
from kwutil import util_time
from kwutil.slugify_ext import smart_truncate
from geowatch.utils import util_kwimage
from geowatch.utils.lightning_ext import util_model

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

    See [LightningCallbacks]_.

    Args:
        num_draw (int):
            number of batches to draw at the start of each epoch

        draw_interval (datetime.timedelta | str | numbers.Number):
            This is the amount of time to wait before drawing the next batch
            item within an epoch. Can be given as a timedelta, a string
            parsable by `coerce_timedelta` (e.g.  '1M') or a numeric number of
            seconds.

        max_items (int):
            Maximum number of items within this batch to draw in a single
            figure. Defaults to 2.

        overlay_on_image (bool):
            if True overlay annotations on image data for a more compact
            view. if False separate annotations / images for a less
            cluttered view.

    FIXME:
        - [ ] This breaks when using strategy=DDP and multiple gpus

    TODO:
        - [ ] Doctest

    Example:
        >>> from geowatch.utils.lightning_ext.callbacks.batch_plotter import *  # NOQA
        >>> import ubelt as ub
        >>> import pytorch_lightning as pl
        >>> from geowatch.utils.lightning_ext import demo
        >>> from geowatch.monkey import monkey_lightning
        >>> monkey_lightning.disable_lightning_hardware_warnings()
        >>> model = demo.LightningToyNet2d(num_train=55)
        >>> default_root_dir = ub.Path.appdir('lightning_ext/tests/BatchPlotter').ensuredir()
        >>> #
        >>> self = BatchPlotter()
        >>> trainer = pl.Trainer(callbacks=[self],
        >>>                      default_root_dir=default_root_dir,
        >>>                      max_epochs=3, accelerator='cpu', devices=1)
        >>> trainer.fit(model)
        >>> train_dpath = ub.Path(trainer.log_dir)
        >>> outputs = list((train_dpath / 'monitor/train/batch').glob('*'))
        >>> print('trainer.logger.log_dir = {!r}'.format(train_dpath))

    Ignore:
        >>> from geowatch.tasks.fusion.fit import make_lightning_modules # NOQA
        >>> args = None
        >>> cmdline = False
        >>> kwargs = {
        ...     'train_dataset': 'special:vidshapes8-multispectral',
        ...     'datamodule': 'KWCocoVideoDataModule',
        ... }
        >>> modules = make_lightning_modules(args=None, cmdline=cmdline, **kwargs)

    References:
        .. [LightningCallbacks] https://pytorch-lightning.readthedocs.io/en/latest/extensions/callbacks.html
    """

    def __init__(self, num_draw=2, draw_interval='5minutes', max_items=2, overlay_on_image=False):
        super().__init__()
        self.num_draw = num_draw

        delta = util_time.coerce_timedelta(draw_interval)
        num_seconds = delta.total_seconds()

        # if isinstance(draw_interval, datetime.timedelta):
        #     delta = draw_interval
        #     num_seconds = delta.total_seconds()
        # elif isinstance(draw_interval, numbers.Number):
        #     num_seconds = draw_interval
        # else:
        #     num_seconds = pytimeparse.parse(draw_interval)
        #     if num_seconds is None:
        #         raise ValueError(f'{draw_interval} is not a parsable delta')

        # Keyword arguments passed to the datamodule draw batch function
        self.draw_batch_kwargs = {
            'max_items': max_items,
            'overlay_on_image': overlay_on_image,
        }

        self.draw_interval_seconds = num_seconds
        self.draw_timer = None
        self._ready_to_draw = False

    def setup(self, trainer, pl_module, stage):
        self.draw_timer = ub.Timer().tic()

    @classmethod
    def add_argparse_args(cls, parent_parser):
        """
        Example:
            >>> from geowatch.utils.lightning_ext.callbacks.batch_plotter import *  # NOQA
            >>> from geowatch.utils.configargparse_ext import ArgumentParser
            >>> cls = BatchPlotter
            >>> parent_parser = ArgumentParser(formatter_class='defaults')
            >>> cls.add_argparse_args(parent_parser)
            >>> parent_parser.print_help()
            >>> parent_parser.parse_known_args()
        """
        from geowatch.utils.lightning_ext import argparse_ext
        arg_infos = argparse_ext.parse_docstring_args(cls)
        argparse_ext.add_arginfos_to_parser(parent_parser, arg_infos)
        return parent_parser

    # def compute_model_cfgstr(self, model):
    #     type(model)

    # @classmethod
    # TODO
    # def demo(cls):
    #     utils()

    def _on_batch_start(self, trainer, pl_module, batch, batch_idx, dataloader_idx=0):
        if self._is_ready_to_draw(trainer, pl_module, batch, batch_idx):
            # Mark state to indicate that we are ready to draw.
            self._ready_to_draw = True
            # If the model or batch supports it, notify it that we require
            # drawable outputs for this batch. TODO: we need to define this
            # protocol clearly and in a way that is easy to add to models.
            # For now we will use a special internal method belonging to the
            # lightning module called "_notify"
            if hasattr(pl_module, '_notify'):
                pl_module._notify({'draw': True}, str(id(self)))

    def _on_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        try:
            self.draw_if_ready(trainer, pl_module, outputs, batch, batch_idx)
        except Exception as e:
            print("========")
            print("Exception raised during batch rendering callback: _on_batch_end")
            print("========")
            print(traceback.format_exc())
            print(repr(e))
        self._ready_to_draw = False
        if hasattr(pl_module, '_notify'):
            pl_module._notify({'draw': False}, str(id(self)))

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx, dataloader_idx=0):
        self._on_batch_start(trainer, pl_module, batch, batch_idx, dataloader_idx=dataloader_idx)

    def on_validation_batch_start(self, trainer, pl_module, batch, batch_idx, dataloader_idx=0):
        self._on_batch_start(trainer, pl_module, batch, batch_idx, dataloader_idx=dataloader_idx)

    def on_test_batch_start(self, trainer, pl_module, batch, batch_idx, dataloader_idx=0):
        self._on_batch_start(trainer, pl_module, batch, batch_idx, dataloader_idx=dataloader_idx)

    #  New
    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        self._on_batch_end(trainer, pl_module, outputs, batch, batch_idx, dataloader_idx)

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        self._on_batch_end(trainer, pl_module, outputs, batch, batch_idx, dataloader_idx)

    def on_test_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        self._on_batch_end(trainer, pl_module, outputs, batch, batch_idx, dataloader_idx)

    @profile
    def draw_batch(self, trainer, outputs, batch, batch_idx):
        from kwutil import util_environ
        if util_environ.envflag('DISABLE_BATCH_PLOTTER'):
            return

        if trainer.log_dir is None:
            warnings.warn('The trainer logdir is not set. Cannot dump a batch plot')
            return

        model = trainer.model
        # TODO: get step number
        if hasattr(model, 'get_cfgstr'):
            model_cfgstr = model.get_cfgstr()
        else:
            hparams = util_model.model_hparams(model)
            model_config = {
                'type': str(model.__class__),
                'hp': smart_truncate(ub.urepr(hparams, compact=1, nl=0), max_length=8),
            }
            model_cfgstr = smart_truncate(ub.urepr(
                model_config, compact=1, nl=0), max_length=64)

        datamodule = trainer.datamodule
        if datamodule is None:
            # must have datamodule to draw batches
            canvas = kwimage.draw_text_on_image(
                {'width': 512, 'height': 512},
                'Implement draw_batch in your datamodule',
                org=(1, 1))
        else:
            stage = trainer.state.stage.value
            if stage == 'validate':
                stage = 'vali'
            canvas = datamodule.draw_batch(batch, outputs=outputs,
                                           stage=stage,
                                           **self.draw_batch_kwargs)

        canvas = np.nan_to_num(canvas)

        stage = trainer.state.stage.lower()
        epoch = trainer.current_epoch
        step = trainer.global_step

        # This is more trouble than it's worth
        # if stage.startswith('val'):
        #     title = f'{stage}_bx{batch_idx:04d}_epoch{epoch:08d}_step{step:08d}'

        fstem = f'{stage}_epoch{epoch:08d}_step{step:08d}_bx{batch_idx:04d}'
        if trainer.global_rank is not None:
            fstem += '_' + str(trainer.global_rank)

        canvas = util_kwimage.draw_header_text(
            image=canvas,
            text=f'{model_cfgstr}',
            stack=True,
        )

        title = fstem + '\n' + f'batch_size={len(batch)}'
        canvas = util_kwimage.draw_header_text(image=canvas, text=title,
                                               stack=True)

        log_dpath = ub.Path(trainer.log_dir)
        dump_dpath = (log_dpath / 'monitor' / stage / 'batch').ensuredir()
        fpath = dump_dpath / f'pred_{fstem}.jpg'

        # print(f'[rank {trainer.global_rank}] write to fpath = {fpath}')
        kwimage.imwrite(fpath, canvas)
        self.draw_timer.tic()

    def _is_ready_to_draw(self, trainer, pl_module, outputs, batch_idx):
        """
        Check if we are ready to draw
        """
        do_draw = batch_idx < self.num_draw
        if self.draw_interval_seconds > 0:
            do_draw |= self.draw_timer.toc() > self.draw_interval_seconds
        if trainer.log_dir is not None:
            # UNDOCUMENTED HIDDEN DEVELOPER SUPER HACK:
            # (very useful if you know about it)
            # By making this file we can let the user see a lot of batches
            # quickly.
            if (ub.Path(trainer.log_dir) / 'please_draw').exists():
                do_draw = True
        return do_draw

    # def draw_if_ready(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
    # @rank_zero_only
    def draw_if_ready(self, trainer, pl_module, outputs, batch, batch_idx):
        # print(f'IN DRAW BATCH: trainer.global_rank={trainer.global_rank}')
        # if trainer.global_rank != 0:
        #     return
        if self._ready_to_draw:
            self.draw_batch(trainer, outputs, batch, batch_idx)
        # print(f'FINISH DRAW BATCH: trainer.global_rank={trainer.global_rank}')
