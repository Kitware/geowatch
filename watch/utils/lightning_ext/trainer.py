"""
Core extensions to the trainer

~/.pyenv/versions/3.8.6/envs/pyenv3.8.6/lib/python3.8/site-packages/pytorch_lightning/__init__.py

~/.pyenv/versions/3.8.6/envs/pyenv3.8.6/lib/python3.8/site-packages/pytorch_lightning/trainer/trainer.py

~/.pyenv/versions/3.8.6/envs/pyenv3.8.6/lib/python3.8/site-packages/pytorch_lightning/trainer/connectors/logger_connector/logger_connector.py

~/.pyenv/versions/3.8.6/envs/pyenv3.8.6/lib/python3.8/site-packages/pytorch_lightning/callbacks/model_checkpoint.py

~/.pyenv/versions/3.8.6/envs/pyenv3.8.6/lib/python3.8/site-packages/pytorch_lightning/trainer/connectors/checkpoint_connector.py
"""
import pytorch_lightning as pl
from typing import Dict, Any, Optional


class KitwareCallbacks(pl.callbacks.Callback):
    """
    Extra steps we want to take
    """
    def setup(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", stage: Optional[str] = None) -> None:
        print('setup kitware callbacks')
        print('trainer.train_dpath = {!r}'.format(trainer.train_dpath))

    def teardown(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", stage: Optional[str] = None) -> None:
        print('teardown kitware callbacks')

    def on_init_start(self, trainer: "pl.Trainer") -> None:
        print('on_init_start')

    def on_init_end(self, trainer: "pl.Trainer") -> None:
        print('on_init_start')

    def on_fit_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        print('on_fit_start')

    def on_fit_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        print('on_fit_end')

    # def on_train_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
    #     print('on_train_start')

    # def on_train_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
    #     print('on_train_end')

    def on_save_checkpoint(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", checkpoint: Dict[str, Any]) -> dict:
        return

    def on_load_checkpoint(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", callback_state: Dict[str, Any]) -> None:
        print('on_load_checkpoint')

    def on_sanity_check_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        print('on_sanity_check_start')

    def on_sanity_check_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        print('on_sanity_check_end')

    def on_keyboard_interrupt(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        print('on_keyboard_interrupt')
        print('KEYBOARD INTERUPT')
        print('trainer.train_dpath = {!r}'.format(trainer.train_dpath))
        print('trainer.log_dir = {!r}'.format(trainer.log_dir))


def kitware_trainer(*args, **kwargs):
    """
    Example:
        >>> from watch.utils.lightning_ext.demo import *  # NOQA
        >>> import ubelt as ub
        >>> default_root_dir = ub.ensure_app_cache_dir('lightning_ext/test/kwtrainer')
        >>> ub.delete(default_root_dir)
        >>> model = LightningToyNet2d(num_train=55)
        >>> trainer = kitware_trainer(default_root_dir=default_root_dir, max_epochs=10)
        >>> print('trainer.train_dpath = {!r}'.format(trainer.train_dpath))
        >>> print('trainer.log_dir = {!r}'.format(trainer.log_dir))
        >>> trainer.fit(model)
        >>> train_dpath = trainer.logger.log_dir
        >>> print('trainer.log_dir = {!r}'.format(trainer.log_dir))
    """

    # It seems we have to override the init, not sure
    resume_from_checkpoint = kwargs.get('resume_from_checkpoint', 'auto')
    train_dpath = kwargs.get('default_root_dir', None)
    assert train_dpath is not None, 'must specify'

    if resume_from_checkpoint == 'auto':
        resume_from_checkpoint = find_most_recent_checkpoint(train_dpath)
        kwargs['resume_from_checkpoint'] = resume_from_checkpoint

    callbacks = kwargs.get('callbacks', 'auto')
    if callbacks == 'auto':
        print('callbacks = {!r}'.format(callbacks))
        from watch.utils.lightning_ext.callbacks import TensorboardPlotter
        from watch.utils.lightning_ext.callbacks import BatchPlotter
        from pytorch_lightning.callbacks import EarlyStopping
        # callbacks = []
        # callbacks += [TensorboardPlotter()]
        # kwargs['callbacks'] = callbacks

        callbacks = [
            KitwareCallbacks(),
            BatchPlotter(
                num_draw=kwargs.get('num_draw', 4),
                draw_interval=kwargs.get('draw_interval', '10m'),
            ),
            TensorboardPlotter(),  # draw tensorboard
            pl.callbacks.LearningRateMonitor(logging_interval='epoch', log_momentum=True),
            pl.callbacks.LearningRateMonitor(logging_interval='step', log_momentum=True),

            pl.callbacks.ModelCheckpoint(monitor='train_loss', mode='min', save_top_k=2),
            # pl.callbacks.GPUStatsMonitor(),
        ]
        # if args.vali_dataset is not None:
        if kwargs.get('vali_dataset', None) is not None:
            callbacks += [
                EarlyStopping(
                    monitor='val_loss', mode='min',
                    patience=kwargs.get('patience', 10),
                    verbose=True),
                pl.callbacks.ModelCheckpoint(
                    monitor='val_loss', mode='min', save_top_k=2),
            ]

        kwargs['callbacks'] = callbacks

    # TODO: explititly initialize the tensorboard logger
    # logger = [
    #     pl.loggers.TensorBoardLogger(
    #         save_dir=args.default_root_dir, version=self.trainer.slurm_job_id, name="lightning_logs"
    #     )
    # ]

    trainer = pl.Trainer(*args, **kwargs)
    trainer.train_dpath = train_dpath
    return trainer


def ensurepath(path_like):
    import pathlib
    if isinstance(path_like, pathlib.Path):
        return path_like
    else:
        return pathlib.Path(path_like)


def find_most_recent_checkpoint(train_dpath):
    train_dpath = ensurepath(train_dpath)
    candidates = list(train_dpath.glob('*/*/checkpoints/*.ckpt'))
    if len(candidates):
        chosen = sorted(candidates)[-1]
    else:
        chosen = None
    return chosen
