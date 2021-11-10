# -*- coding: utf-8 -*-
"""
Packager callback to interface with torch.package
"""
import pytorch_lightning as pl
import ubelt as ub
import copy
from os.path import join
from typing import Dict, Any


class Packager(pl.callbacks.Callback):
    """
    Packages the best checkpoint at the end of training and at various other
    key phases of the training loop.

    The lightning module must have a "save_package" method that can be called
    with a filepath.


    TODO:
        - [ ] Package the "best" checkpoints according to the monitor

        - [ ] Package arbitrary checkpoints:
            - [ ] Package model topology without any weights
            - [ ] Copy checkpoint weights into a package to get a package with
                  that "weight state".

        - [ ] Initializer should be able to point at a package and use
            torch-liberator partial load to transfer the weights.

        - [ ] Replace print statements with logging statements

        - [ ] Create a trainer-level logger instance (similar to netharn)

        - [ ] what is the right way to handle running eval after fit?
              There may be multiple candidate models that need to be tested, so
              we can't just specify one package, one prediction dumping ground,
              and one evaluation dataset, maybe we specify the paths where the
              "best" ones are written?.

    Args:
        package_fpath (PathLike):
            Specifies a path where a torch packaged model will be written (or
            symlinked) to.

    References:
        https://discuss.pytorch.org/t/packaging-pytorch-topology-first-and-checkpoints-later/129478/2

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
        self.package_on_interrupt = True
        self.package_fpath = package_fpath
        self.package_verbose = 0

    @classmethod
    def add_argparse_args(cls, parent_parser):
        """
        Example:
            >>> from watch.utils.lightning_ext.callbacks.packager import *  # NOQA
            >>> from watch.utils.configargparse_ext import ArgumentParser
            >>> cls = Packager
            >>> parent_parser = ArgumentParser(formatter_class='defaults')
            >>> cls.add_argparse_args(parent_parser)
            >>> parent_parser.print_help()
        """
        from watch.utils.lightning_ext import argparse_ext
        arg_infos = argparse_ext.parse_docstring_args(cls)
        argparse_ext.add_arginfos_to_parser(parent_parser, arg_infos)
        return parent_parser

    def on_init_end(self, trainer: 'pl.Trainer') -> None:
        """
        Finalize initialization step.
        Resolve the paths where files will be written.
        """
        # Rectify paths if we need to
        print('on_init_start')
        if self.package_fpath == 'auto':
            self.package_fpath = join(trainer.default_root_dir, 'final_package.pt')
            print('setting auto self.package_fpath = {!r}'.format(self.package_fpath))

        # Hack this in. TODO: what is the best way to expose this?
        trainer.package_fpath = self.package_fpath
        print('will save trainer.package_fpath = {!r}'.format(trainer.package_fpath))

    def on_fit_start(self, trainer: 'pl.Trainer', pl_module: 'pl.LightningModule') -> None:
        """
        TODO:
            - [ ] Write out the uninitialized topology
        """
        if False:
            print('Training is starting, checking that the model can be packaged')
            package_dpath = ub.ensuredir((trainer.log_dir, 'packages'))
            package_fpath = join(package_dpath,
                                 '_test_package_epoch{}_step{}.pt'.format(
                                     trainer.current_epoch,
                                     trainer.global_step))
            self._save_package(pl_module, package_fpath)

    def on_fit_end(self, trainer: 'pl.Trainer', pl_module: 'pl.LightningModule') -> None:
        """
        Create the final package (or a list of candidate packages) for
        evaluation and deployment.

        TODO:
            - [ ] how do we properly package all of the candidate checkpoints?
            - [ ] Symlink to "BEST" package at the end.
            - [ ] write some script such that any checkpoint can be packaged.
        """
        print('Training is complete, packaging model')
        package_fpath = self._make_package_fpath(trainer)
        self._save_package(pl_module, package_fpath)
        final_package_fpath = self.package_fpath
        ub.symlink(package_fpath, final_package_fpath, overwrite=True, verbose=3)
        print('final_package_fpath = {!r}'.format(final_package_fpath))

    def on_save_checkpoint(self, trainer: 'pl.Trainer', pl_module: 'pl.LightningModule', checkpoint: Dict[str, Any]) -> dict:
        """
        TODO:
            - [ ] Do we create a package for every checkpoint?
        """
        # package_fpath = self._make_package_fpath(trainer)
        # self._save_package(pl_module, package_fpath)

    def on_keyboard_interrupt(self, trainer: 'pl.Trainer', pl_module: 'pl.LightningModule') -> None:
        """
        Saving a package on keyboard interrupt is useful for manual early
        stopping.

        TODO:
            - [X] Package current model state
            - [ ] Package "best" model state
        """
        if self.package_on_interrupt:
            print('Attempting to package model before exiting')
            package_fpath = self._make_package_fpath(
                trainer, dname='package-interupt')
            self._save_package(pl_module, package_fpath)

    def _make_package_fpath(self, trainer, dname='packages'):
        package_dpath = ub.ensuredir((trainer.log_dir, dname))
        package_fpath = join(package_dpath,
                             'package_epoch{}_step{}.pt'.format(
                                 trainer.current_epoch, trainer.global_step))
        return package_fpath

    def _save_package(self, model, package_fpath):
        if hasattr(model, 'save_package'):
            print('calling model.save_package')
            model.save_package(package_fpath, verbose=self.package_verbose)
        else:
            print('model has no save_package method required by Packager')
            default_save_package(model, package_fpath, verbose=self.package_verbose)
        print('save package_fpath = {!r}'.format(package_fpath))


def default_save_package(model, package_path, verbose=1):
    import torch.package
    # shallow copy of self, to apply attribute hacks to
    model = copy.copy(model)
    model.trainer = None
    model.train_dataloader = None
    model.val_dataloader = None
    model.test_dataloader = None
    model_name = 'model.pkl'
    module_name = 'default_save_module_name'
    if verbose:
        print('Packaging package_path = {!r}'.format(package_path))

    with torch.package.PackageExporter(package_path) as exp:
        # TODO: this is not a problem yet, but some package types (mainly binaries) will need to be excluded and added as mocks
        # exp.extern("**", exclude=["watch.tasks.fusion.**"])
        exp.extern('**')
        # exclude=["watch.tasks.fusion.**"])
        # exp.intern("watch.tasks.fusion.**")
        exp.save_pickle(module_name, model_name, model)
