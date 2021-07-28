"""
Notes:
    There are parts of netharn that could be ported to lightning

    The logging stuff
        - [ ] loss curves (odd they aren't in tensorboard)

    The auto directory structure
        - [ ] save multiple checkpoints
        - [ ] delete them intelligently

    The run managment
        - [ ] The netharn/cli/manage_runs.py

    The auto-deploy files
        - [ ] Use Torch 1.9 Packages instead of Torch-Liberator

    Automated dynamics / plugins?

CommandLine:
    CUDA_VISIBLE_DEVICES=1 DVC_DPATH=$HOME/data/dvc-repos/smart_watch_dvc xdoctest -m watch.tasks.fusion.fit __doc__:0 -- --profile
    CUDA_VISIBLE_DEVICES=1 DVC_DPATH=$HOME/data/dvc-repos/smart_watch_dvc xdoctest -m watch.tasks.fusion.fit __doc__:0
    CUDA_VISIBLE_DEVICES=0 DVC_DPATH=$HOME/data/dvc-repos/smart_watch_dvc xdoctest -m watch.tasks.fusion.fit __doc__:0

Example:
    >>> # xdoctest: +REQUIRES(env:DVC_DPATH)
    >>> from watch.tasks.fusion.fit import *  # NOQA
    >>> from os.path import join
    >>> import os
    >>> _default = ub.expandpath('$HOME/data/dvc-repos/smart_watch_dvc')
    >>> dvc_dpath = os.environ.get('DVC_DPATH', _default)
    >>> train_fpath = join(dvc_dpath, 'drop1_S2_aligned_c1/train_data.kwcoco.json')
    >>> vali_fpath = join(dvc_dpath, 'drop1_S2_aligned_c1/vali_data.kwcoco.json')

    >>> import kwcoco
    >>> dset = kwcoco.CocoDataset(train_fpath)
    >>> available_channel_profiles = {
    >>>     frozenset(aux.get('channels', None) for aux in img.get('auxiliary', []))
    >>>      for img in dset.index.imgs.values()}
    >>> print('available_channel_profiles = {!r}'.format(available_channel_profiles))

    >>> args = None
    >>> cmdline = False
    >>> kwargs = {
    ...     'train_dataset': train_fpath,
    ...     'vali_dataset': vali_fpath,
    ...     'dataset': 'WatchDataModule',
    ...     #'method': 'MultimodalTransformerDirectCD',
    ...     'method': 'MultimodalTransformerDotProdCD',
    ...     'channels': 'coastal|blue|green|red|nir|swir16|swir22',
    ...     #'channels': 'blue|green|red|nir',
    ...     #'channels': None,
    ...     'time_steps': 8,
    ...     #'chip_size': 128,
    ...     'chip_size': 192,
    ...     #'chip_size': 256,
    ...     'batch_size': 1,
    ...     'accumulate_grad_batches': 12,
    ...     'model_name': 'smt_it_stm_p8',
    ...     'num_workers': 12,
    ...     'attention_impl': 'exact',
    ...     #'attention_impl': 'performer',  # note: exact seems to be faster and less memory at this scale
    ...     'gradient_clip_val': 0.5,
    ...     'gradient_clip_algorithm': 'value',
    ...     'gpus': 1,
    ... }
    >>> #modules = make_lightning_modules(args=None, cmdline=cmdline, **kwargs)
    >>> fit_model(args=args, cmdline=cmdline, **kwargs)
"""

import pytorch_lightning as pl

from watch.tasks.fusion import datasets
from watch.tasks.fusion import methods
from watch.tasks.fusion import models
from watch.tasks.fusion import utils

# import scriptconfig as scfg
import ubelt as ub
import sys
import pathlib

try:
    import xdev
    profile = xdev.profile
except Exception:
    profile = ub.identity

# available_methods = dir(methods)
available_methods = [
    'End2EndVotingModel',
    'MultimodalTransformerDirectCD',
    'MultimodalTransformerDotProdCD',
    'MultimodalTransformerSegmentation',
    'TransformerChangeDetector',
    'UNetChangeDetector',
    'VotingModel',
]

# Model names define the transformer encoder used by the method
available_models = list(models.transformer.encoder_configs.keys())

# dir(datasets)
# TODO: rename to datamodules
available_datasets = [
    # 'Drop0AlignMSI_S2',
    # 'Drop0Raw_S2',
    # 'OneraCD_2018',

    'WatchDataModule',
]

# TODO: is there a better way to mark these?
learning_irrelevant = {
    'workdir',
    'num_workers',
    'gpus',
    'limit_val_batches',
    'limit_test_batches',
    'limit_predict_batches',
    'val_check_interval',
    'flush_logs_every_n_steps',
    'reload_dataloaders_every_epoch',
    'progress_bar_refresh_rate'
    'log_every_n_steps',
    'log_gpu_memory',
    'logger',
    'checkpoint_callback',
}


class DrawBatchCallback(pl.callbacks.Callback):
    """
    These are callbacks used to monitor the training

    Args:
        num_draw (int): number of batches to draw at the start of each epoch
        draw_interval (int): if nothing has been drawn in this many minutes,
            draw something.

    References:
        https://pytorch-lightning.readthedocs.io/en/latest/extensions/callbacks.html
    """
    def __init__(self, num_draw=2, draw_interval=10):
        super().__init__()
        self.num_draw = num_draw
        self.draw_interval = draw_interval
        self.draw_timer = None

    def setup(self, trainer, pl_module, stage):
        self.draw_timer = ub.Timer().tic()

    @profile
    def draw_batch(self, trainer, outputs, batch, batch_idx):
        import numpy as np
        import kwimage
        from os.path import join

        datamodule = trainer.datamodule
        canvas = datamodule.draw_batch(batch, outputs=outputs)

        # dataset = datamodule.torch_datasets[stage]
        # images = batch['images']
        # if 0:
        #     import kwplot
        #     kwplot.autompl()
        #     kwplot.imshow(canvas)

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
        if self.draw_interval > 0:
            do_draw |= self.draw_timer.toc() > 60 * self.draw_interval
        if do_draw:
            self.draw_batch(trainer, outputs, batch, batch_idx)
            self.draw_timer.tic()

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
        self.draw_if_ready(trainer, pl_module, outputs, batch, batch_idx, dataloader_idx)

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
        self.draw_if_ready(trainer, pl_module, outputs, batch, batch_idx, dataloader_idx)

    def on_test_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
        self.draw_if_ready(trainer, pl_module, outputs, batch, batch_idx, dataloader_idx)

    # def on_init_start(self, trainer):
    #     pass

    # def on_init_end(self, trainer):
    #     print('trainer is init now')

    # def on_train_end(self, trainer, pl_module):
    #     print('do something when training ends')


def make_fit_config(args=None, cmdline=False, **kwargs):
    """
    Args:
        args : namespace that overrides defaults
        cmdline (bool): if True, will override defaults based on sys.argv
        **kwargs: dictionary that overrides defaults

    Example:
        >>> from watch.tasks.fusion.fit import *  # NOQA
        >>> args = None
        >>> cmdline = False
        >>> kwargs = {}
        >>> args = make_fit_config(args=args, cmdline=cmdline, **kwargs)
    """
    import argparse
    import configargparse

    class RawDescriptionDefaultsHelpFormatter(
            argparse.RawDescriptionHelpFormatter,
            argparse.ArgumentDefaultsHelpFormatter):
        pass

    parser = configargparse.ArgumentParser(
        add_config_file_help=False,
        description='Training script for the fused change/segmentation task',
        formatter_class=RawDescriptionDefaultsHelpFormatter,
    )

    # Setup scriptconfig-like special arguments to set a config via a file or
    # dump some config to stdout or disk
    config_parser = parser.add_argument_group("Config")
    config_parser.add('--config', is_config_file=True, help=ub.paragraph(
        '''
        A path to a config file path that will overwrite the defaults.
        '''))

    config_parser.add_argument('--dump', default=None, help=ub.paragraph(
        '''
        If specified, dump this config to this filepath on disk and exit.
        '''))

    config_parser.add_argument('--dumps', action='store_true', help=ub.paragraph(
        '''
        If specified, dump this config stdout and exit.
        '''))

    # Setup common fields and modal switches
    modal_parser = parser.add_argument_group("Modal")
    modal_parser.add_argument(
        '--dataset', choices=available_datasets, default='WatchDataModule',
        help=ub.paragraph(
            '''
            Modal parameter indicating the family of dataset to train on.
            See the watch.tasks.fusion.datasets submodule for details
            '''))

    modal_parser.add_argument(
        '--method', default='MultimodalTransformerDotProdCD',
        choices=available_methods, help=ub.paragraph(
            '''
            Modal parameter indicating the family of model to train.
            See the watch.tasks.fusion.methods submodule for details
            ''')
    )

    # override common defaults with user settings
    parser.set_defaults(**kwargs)
    # The specific parser will depend on the modal arguments
    modal, _ = parser.parse_known_args(ignore_help_args=True)

    print(kwargs)
    print(modal)

    common_parser = parser.add_argument_group("Common")
    common_parser.add_argument(
        '--workdir', default='./_trained_models',
        help=ub.paragraph(
            '''
            Directory where training data can be written.
            Overrides default_root_dir,
            ''')
    )

    print(modal)

    # Get subcomponents
    method_class = getattr(methods, modal.method)
    dataset_class = getattr(datasets, modal.dataset)

    # Extend the parser based on the chosen dataset / method modes
    dataset_class.add_data_specific_args(parser)
    method_parser = parser.add_argument_group("Method")
    method_class.add_model_specific_args(method_parser)
    pl.Trainer.add_argparse_args(parser)

    # Remove parameters that we will fill in with special logic
    # Apparently this is hard to do, argparse is such a mess.

    # to_remove = ['default_root_dir']
    # dest_to_actions = ub.group_items(parser._actions, lambda x: x.dest)
    # for rmkey in to_remove:
    #     for action in dest_to_actions[rmkey]:
    #         parser._remove_action(action)
    #         for optstr in action.option_strings:
    #             parser._option_string_actions.pop(optstr)
    # for grp in parser._action_groups:
    #     dest_to_actions = ub.group_items(grp._actions, lambda x: x.dest)
    #     for rmkey in to_remove:
    #         for action in dest_to_actions[rmkey]:
    #             print('action = {!r}'.format(action))
    #             grp._actions.remove(action)

    # override modal-specific defaults with user settings
    parser.set_defaults(**kwargs)

    args, _ = parser.parse_known_args(args=args)

    # Do scriptconfig like dump logic
    dump_fpath = args.dump
    do_dumps = args.dumps
    # Remove special arguments
    del args.dump
    del args.dumps
    del args.config
    if do_dumps:
        config_items = parser.get_items_for_config_file_output(parser._source_to_settings, args)
        file_contents = parser._config_file_parser.serialize(config_items)
        print(file_contents)
        sys.exit(0)

    if dump_fpath is not None:
        config_items = parser.get_items_for_config_file_output(parser._source_to_settings, args)
        file_contents = parser._config_file_parser.serialize(config_items)
        with open(dump_fpath, 'w') as file:
            file.write(file_contents)
        sys.exit(0)

    learning_config = ub.dict_diff(args.__dict__, learning_irrelevant)

    # Construct a netharn-like training directory based on relevant hyperparams
    args.train_hashid = ub.hash_data(ub.map_vals(str, learning_config))[0:16]
    args.train_name = "{method}-{train_hashid}".format(**args.__dict__)
    args.default_root_dir = pathlib.Path(args.workdir) / args.train_name

    # TODO:
    # Add dump and --dumps commands to write the config to a file
    # similar to how scriptconfig works
    return args


def make_lightning_modules(args=None, cmdline=False, **kwargs):
    """
    Example:
        >>> from watch.tasks.fusion.fit import *  # NOQA
        >>> args = None
        >>> cmdline = False
        >>> kwargs = {
        ...     'train_dataset': 'special:vidshapes8-multispectral',
        ...     'dataset': 'WatchDataModule',
        ... }
        >>> modules = make_lightning_modules(args=None, cmdline=cmdline, **kwargs)
    """
    args = make_fit_config(args=args, cmdline=cmdline, **kwargs)
    print("{train_name}\n====================".format(**args.__dict__))

    method_class = getattr(methods, args.method)
    dataset_class = getattr(datasets, args.dataset)

    # init dataset from args
    # TODO: compute and cache mean / std if it is not provided. Pass this to
    # the model so it can whiten the inputs.
    dataset_var_dict = utils.filter_args(args.__dict__, dataset_class.__init__)
    # dataset_var_dict["preprocessing_step"] = model.preprocessing_step
    datamodule = dataset_class(**dataset_var_dict)
    datamodule.setup("fit")

    # init method from args
    method_var_dict = args.__dict__

    # TODO: need a better way to indicate that a method needs parameters from a
    # dataset, and maybe the reverse too
    if hasattr(dataset_class, "bce_weight"):
        method_var_dict["pos_weight"] = getattr(dataset_class, "bce_weight")

    method_var_dict = utils.filter_args(method_var_dict, method_class.__init__)

    if hasattr(datamodule, "input_stats"):
        print('datamodule.input_stats = {}'.format(ub.repr2(datamodule.input_stats, nl=2)))
        method_var_dict["input_stats"] = datamodule.input_stats
    # Note: Changed name from method to model
    model = method_class(**method_var_dict)

    # init trainer from args
    trainer = pl.Trainer.from_argparse_args(args, callbacks=[
        DrawBatchCallback()
    ])

    modules = {
        'datamodule': datamodule,
        'model': model,
        'trainer': trainer,
    }
    return modules


def fit_model(args=None, cmdline=False, **kwargs):
    """
    Example:
        >>> from watch.tasks.fusion.fit import *  # NOQA
        >>> args = None
        >>> cmdline = False
        >>> kwargs = {
        ...     'train_dataset': 'special:vidshapes8-multispectral',
        ...     'dataset': 'WatchDataModule',
        ... }
        >>> args = make_fit_config(args=None, cmdline=cmdline, **kwargs)
        >>> print('args.__dict__ = {}'.format(ub.repr2(args.__dict__, nl=1)))
        >>> fit_model(**kwargs)
    """
    modules = make_lightning_modules(args=args, cmdline=cmdline, **kwargs)
    trainer = modules['trainer']
    datamodule = modules['datamodule']
    model = modules['model']

    # prime the model, incase it has a lazy layer
    batch = next(iter(datamodule.train_dataloader()))

    # batch_shapes = ub.map_vals(lambda x: x.shape, batch)
    # print('batch_shapes = {}'.format(ub.repr2(batch_shapes, nl=1)))

    # result = model(batch["images"][[0], ...].float())
    import torch
    with torch.set_grad_enabled(False):
        model.forward_step(batch)

    # if requested, tune model
    trainer.tune(model, dataset)

    # fit the model
    trainer.fit(model, datamodule)

    # save model to package
    utils.create_package(model, args.default_root_dir / "package.pt")

    # save learning relevant training options
    learning_config = ub.dict_diff(args.__dict__, learning_irrelevant)


def main(args=None, **kwargs):
    """

    CommandLine:
        # Set the path to your data
        DVC_DPATH=$HOME/Projects/smart_watch_dvc
        DVC_DPATH=$HOME/data/dvc-repos/smart_watch_dvc
        TRAIN_FPATH=$DVC_DPATH/extern/onera_2018/onera_train.kwcoco.json

        # View the help docs
        python -m watch.tasks.fusion.fit --help

        python -m watch.tasks.fusion.fit --dumps

        # Invoke the training script

        # This task required 5.12GB on a 3090
        python -m watch.tasks.fusion.fit \
            --model_name=smt_it_stm_p8 \
            --method=MultimodalTransformerDotProdCD \
            --dataset=WatchDataModule \
            --train_dataset=vidshapes8-multispectral \
            --batch_size=4 \
            --num_workers=4 \
            --chip_size=96 \
            --workdir=$HOME/work/watch/fit

        # This task required 5.12GB on a 3090
        python -m watch.tasks.fusion.fit \
            --model_name=smt_it_stm_p8 \
            --method=MultimodalTransformerDotProdCD \
            --train_dataset=$TRAIN_FPATH \
            --batch_size=4 \
            --num_workers=4 \
            --chip_size=96 \
            --workdir=$HOME/work/watch/fit

        # This task required 17GB on a 3090
        python -m watch.tasks.fusion.fit \
            --model_name=smt_it_stm_p8 \
            --method=MultimodalTransformerDotProdCD \
            --train_dataset=$TRAIN_FPATH \
            --batch_size=16 \
            --num_workers=4 \
            --chip_size=128 \
            --workdir=$HOME/work/watch/fit

        # This task required 20GB on a 3090
        python -m watch.tasks.fusion.onera_channelwisetransformer_train \
            --model_name=smt_it_joint_p8 \
            --method=MultimodalTransformerDirectCD \
            --train_dataset=$TRAIN_FPATH \
            --batch_size=4 \
            --num_workers=4 \
            --chip_size=96 \
            --workdir=$HOME/work/watch/fit
    """
    args = make_fit_config(args=args, cmdline=True, **kwargs)
    fit_model(args=args, **kwargs)


if __name__ == "__main__":
    main()
