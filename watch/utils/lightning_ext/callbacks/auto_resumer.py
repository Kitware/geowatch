# -*- coding: utf-8 -*-
import pytorch_lightning as pl
from watch.utils import util_path

__all__ = ['AutoResumer']


class AutoResumer(pl.callbacks.Callback):
    """
    Auto-resumes from the most recent checkpoint

    Example:
        >>> from watch.utils.lightning_ext.callbacks.auto_resumer import AutoResumer
        >>> from watch.utils import util_path
        >>> from watch.utils.lightning_ext.demo import LightningToyNet2d
        >>> from watch.utils.lightning_ext.callbacks import StateLogger
        >>> import pytorch_lightning as pl
        >>> import ubelt as ub
        >>> default_root_dir = ub.get_app_cache_dir('lightning_ext/test/auto_resume')
        >>> ub.delete(default_root_dir)
        >>> # Test starting a model without any existing checkpoints
        >>> trainer = pl.Trainer(default_root_dir=default_root_dir, callbacks=[AutoResumer(), StateLogger()], max_epochs=5)
        >>> model = LightningToyNet2d()
        >>> trainer.fit(model)
        >>> assert len(list((util_path.coercepath(trainer.logger.log_dir) / 'checkpoints').glob('*'))) > 0
        >>> # See contents written
        >>> print(ub.repr2(list(util_path.tree(default_root_dir)), sort=0))
        >>> #
        >>> # Make a new trainer that should auto-resume
        >>> trainer = pl.Trainer(default_root_dir=default_root_dir, callbacks=[AutoResumer(), StateLogger()], max_epochs=5)
        >>> model = LightningToyNet2d()
        >>> trainer.fit(model)
        >>> print(ub.repr2(list(util_path.tree(default_root_dir)), sort=0))
        >>> # max_epochs should prevent auto-resume from doing anything
        >>> assert len(list((util_path.coercepath(trainer.logger.log_dir) / 'checkpoints').glob('*'))) == 0
        >>> #
        >>> # Increasing max epochs will let it train for longer
        >>> trainer = pl.Trainer(default_root_dir=default_root_dir, callbacks=[AutoResumer(), StateLogger()], max_epochs=6)
        >>> model = LightningToyNet2d()
        >>> trainer.fit(model)
        >>> print(ub.repr2(list(util_path.tree(util_path.coercepath(default_root_dir))), sort=0))
        >>> # max_epochs should prevent auto-resume from doing anything
        >>> assert len(list((util_path.coercepath(trainer.logger.log_dir) / 'checkpoints').glob('*'))) > 0
    """

    def __init__(self):
        """
        TODO:
            - [ ] Configure how to find which checkpoint to resume from
        """
        pass

    def on_init_start(self, trainer: "pl.Trainer") -> None:
        train_dpath = trainer.default_root_dir
        prev_states = self.recent_checkpoints(train_dpath)
        print('There are {} existing checkpoints'.format(len(prev_states)))
        if len(prev_states) > 0:
            resume_from_checkpoint = prev_states[-1]
            print('Will resume from {!r}'.format(resume_from_checkpoint))
            # A Trainer will construct its checkpoint connector before this is
            # called, but it wont use it, so it is currently safe to overwrite
            # it with a new one that has the "correct" resume_from_checkpoint
            # argument.
            CheckpointConnector = trainer.checkpoint_connector.__class__
            trainer.checkpoint_connector = CheckpointConnector(
                trainer, resume_from_checkpoint)

    def recent_checkpoints(self, train_dpath):
        """
        Return a list of existing checkpoints in some Trainer root directory
        """
        train_dpath = util_path.coercepath(train_dpath)
        candidates = sorted(train_dpath.glob('*/*/checkpoints/*.ckpt'))
        return candidates
