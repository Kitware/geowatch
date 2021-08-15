import pytorch_lightning as pl
# from typing import Dict, Any
import ubelt as ub
from os.path import join


class Packager(pl.callbacks.Callback):
    """
    Packages the best checkpoint at the end of training and at various other
    key phases of the training loop.

    The lightning module must have a "save_package" method that can be called
    with a filepath

    Example:
        >>> from watch.utils.lightning_ext.callbacks.packager import *  # NOQA
        >>> from watch.utils.lightning_ext.demo import LightningToyNet2d
        >>> from watch.utils.lightning_ext.callbacks import StateLogger
        >>> import ubelt as ub
        >>> default_root_dir = ub.get_app_cache_dir('lightning_ext/test/packager')
        >>> ub.delete(default_root_dir)
        >>> # Test starting a model without any existing checkpoints
        >>> trainer = pl.Trainer(default_root_dir=default_root_dir, callbacks=[
        >>>     Packager(join(default_root_dir, 'final_package.pt')),
        >>>     StateLogger()
        >>> ], max_epochs=2)
        >>> model = LightningToyNet2d()
        >>> trainer.fit(model)

    """

    def __init__(self, package_fpath='auto'):
        # self.package_on_interrupt = True
        self.package_fpath = package_fpath

    def on_init_end(self, trainer: "pl.Trainer") -> None:
        # Rectify paths if we need to
        print('on_init_start')
        if self.package_fpath == 'auto':
            self.package_fpath = join(trainer.default_root_dir, 'final_package.pt')

    def on_fit_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        if True:
            print('Training is starting, checking that the model can be packaged')
            # self._save_package(trainer.model)
            package_dpath = ub.ensuredir((trainer.log_dir, 'packages'))
            package_fpath = join(package_dpath, '_test_package_epoch{}_step{}.pt'.format(trainer.current_epoch, trainer.global_step))
            self._save_package(pl_module, package_fpath)

    def on_fit_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        print('Training is complete, packaging model')
        # package_fpath = join()
        # join(package_fpath,
        # self._save_package(trainer.model)
        package_dpath = ub.ensuredir((trainer.log_dir, 'packages'))
        package_fpath = join(package_dpath, 'package_epoch{}_step{}.pt'.format(trainer.current_epoch, trainer.global_step))
        self._save_package(pl_module, package_fpath)
        # Symlink to "BEST" package at the end.
        # TODO: write some script such that any checkpoint can be packaged.
        final_package_fpath = self.package_fpath
        if final_package_fpath is not None:
            final_package_fpath = join(trainer.default_root_dir, 'final_package.pt')
            ub.symlink(package_fpath, final_package_fpath, overwrite=True, verbose=3)
            print('final_package_fpath = {!r}'.format(final_package_fpath))

    # def on_save_checkpoint(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", checkpoint: Dict[str, Any]) -> dict:
    #     print('on_save_checkpoint - checkpoint.keys() = {}'.format(ub.repr2(checkpoint.keys(), nl=1)))
    #     package_dpath = ub.ensuredir((trainer.log_dir, 'packages'))
    #     package_fpath = join(package_dpath, 'package_epoch{}_step{}.pt'.format(trainer.current_epoch, trainer.global_step))
    #     self._save_package(pl_module, package_fpath)

    def on_keyboard_interrupt(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        print('Attempting to package model before exiting')
        self._save_package(trainer.model)
        package_dpath = ub.ensuredir((trainer.log_dir, 'packages-interrupt'))
        package_fpath = join(package_dpath, 'package_epoch{}_step{}.pt'.format(trainer.current_epoch, trainer.global_step))
        self._save_package(pl_module, package_fpath)

    def _save_package(self, model, package_fpath):
        if hasattr(model, 'save_package'):
            model.save_package(package_fpath)
        else:
            print('model has no save_package method required by Packager')
            default_save_package(model, package_fpath)
        print('save package_fpath = {!r}'.format(package_fpath))


def default_save_package(model, package_path, verbose=1):
    import copy
    import torch.package
    # shallow copy of self, to apply attribute hacks to
    model = copy.copy(model)
    model.trainer = None
    model.train_dataloader = None
    model.val_dataloader = None
    model.test_dataloader = None
    model_name = "model.pkl"
    module_name = 'default_save_module_name'
    with torch.package.PackageExporter(package_path, verbose=verbose) as exp:
        # TODO: this is not a problem yet, but some package types (mainly binaries) will need to be excluded and added as mocks
        # exp.extern("**", exclude=["watch.tasks.fusion.**"])
        exp.extern("**")
        # exclude=["watch.tasks.fusion.**"])
        # exp.intern("watch.tasks.fusion.**")
        exp.save_pickle(module_name, model_name, model)
