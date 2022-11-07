"""
https://github.com/Lightning-AI/lightning/issues/10894
"""

import pytorch_lightning as pl
from watch.utils import util_path
from packaging.version import parse as Version
from typing import Optional  # NOQA

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
        >>> from watch.utils.lightning_ext.monkeypatches import disable_lightning_hardware_warnings
        >>> disable_lightning_hardware_warnings()
        >>> default_root_dir = ub.Path.appdir('lightning_ext/test/auto_resume')
        >>> default_root_dir.delete()
        >>> #
        >>> # STEP 1:
        >>> # Test starting a model without any existing checkpoints
        >>> import pytest
        >>> try:
        >>>     AutoResumer()
        >>> except NotImplementedError:
        >>>     pytest.skip()
        >>> trainer_orig = pl.Trainer(default_root_dir=default_root_dir, callbacks=[AutoResumer(), StateLogger()], max_epochs=2)
        >>> model = LightningToyNet2d()
        >>> trainer_orig.fit(model)
        >>> assert len(list((util_path.coercepath(trainer_orig.logger.log_dir) / 'checkpoints').glob('*'))) > 0
        >>> # See contents written
        >>> print(ub.repr2(list(util_path.tree(default_root_dir)), sort=0))
        >>> #
        >>> # CHECK 1:
        >>> # Make a new trainer that should auto-resume
        >>> self = AutoResumer()
        >>> trainer = trainer_resume1 = pl.Trainer(default_root_dir=default_root_dir, callbacks=[self, StateLogger()], max_epochs=2)
        >>> model = LightningToyNet2d()
        >>> trainer_resume1.fit(model)
        >>> print(ub.repr2(list(util_path.tree(default_root_dir)), sort=0))
        >>> # max_epochs should prevent auto-resume from doing anything
        >>> assert len(list((util_path.coercepath(trainer_resume1.logger.log_dir) / 'checkpoints').glob('*'))) == 0
        >>> #
        >>> # CHECK 2:
        >>> # Increasing max epochs will let it train for longer
        >>> trainer_resume2 = pl.Trainer(default_root_dir=default_root_dir, callbacks=[AutoResumer(), StateLogger()], max_epochs=3)
        >>> model = LightningToyNet2d()
        >>> trainer_resume2.fit(model)
        >>> print(ub.repr2(list(util_path.tree(util_path.coercepath(default_root_dir))), sort=0))
        >>> # max_epochs should prevent auto-resume from doing anything
        >>> assert len(list((util_path.coercepath(trainer_resume2.logger.log_dir) / 'checkpoints').glob('*'))) > 0
    """

    def __init__(self):
        """
        TODO:
            - [ ] Configure how to find which checkpoint to resume from
        """
        if Version(pl.__version__) >= Version('1.8.0'):
            raise NotImplementedError(
                'Lightning 1.8.0 broke on_init_start, and we havent fixed it. '
                'This component should be non-critical. Avoid using in the '
                'meantime'
            )

    # @classmethod
    # def add_argparse_args(cls, parent_parser):
    #     """
    #     Example:
    #         >>> from watch.utils.lightning_ext.callbacks.auto_resumer import *  # NOQA
    #         >>> from watch.utils.configargparse_ext import ArgumentParser
    #         >>> cls = AutoResumer
    #         >>> parent_parser = ArgumentParser(formatter_class='defaults')
    #         >>> cls.add_argparse_args(parent_parser)
    #         >>> parent_parser.print_help()
    #     """
    #     from watch.utils.lightning_ext import argparse_ext
    #     arg_infos = argparse_ext.parse_docstring_args(cls)
    #     argparse_ext.add_arginfos_to_parser(parent_parser, arg_infos)
    #     return parent_parser

    # FIXME: this doesn't work in new lightning versions

    # def setup(self, trainer: 'pl.Trainer', pl_module: 'pl.LightningModule', stage: Optional[str] = None) -> None:
    def on_init_start(self, trainer: 'pl.Trainer') -> None:
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

            attrname = '_checkpoint_connector'
            if not hasattr(trainer, attrname):
                # lightning < 1.6
                attrname = 'checkpoint_connector'
            checkpoint_connector = getattr(trainer, attrname)

            CheckpointConnector = checkpoint_connector.__class__
            setattr(trainer, attrname, CheckpointConnector(
                trainer, resume_from_checkpoint))

    def recent_checkpoints(self, train_dpath):
        """
        Return a list of existing checkpoints in some Trainer root directory
        """
        train_dpath = util_path.coercepath(train_dpath)
        candidates = sorted(train_dpath.glob('*/*/checkpoints/*.ckpt'))
        return candidates
