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


Experiments:
    experiments/crall/onera_experiments.sh


TODO:
    - [ ] Rename --dataset argument to --datamodule

    - [ ] Rename WatchDataModule to ChangeDataModule

    - [ ] Need to figure out how to connect configargparse with ray.tune

    - [ ] Distributed Training:
        - [ ] How do do DistributedDataParallel
        - [ ] On one machine
        - [ ] On multiple machines

    - [ ] Add Data Modules:
        - [ ] SegmentationDataModule
        - [ ] ClassificationDataModule
        - [ ] DetectionDataModule
        - [ ] <Problem>DataModule

CommandLine:
    CUDA_VISIBLE_DEVICES=1 DVC_DPATH=$HOME/data/dvc-repos/smart_watch_dvc xdoctest -m watch.tasks.fusion.fit __doc__:0 -- --profile
    CUDA_VISIBLE_DEVICES=1 DVC_DPATH=$HOME/data/dvc-repos/smart_watch_dvc xdoctest -m watch.tasks.fusion.fit __doc__:0
    CUDA_VISIBLE_DEVICES=0 DVC_DPATH=$HOME/data/dvc-repos/smart_watch_dvc xdoctest -m watch.tasks.fusion.fit __doc__:0


    # Takes ~18GB on a 3090
    DVC_DPATH=$HOME/data/dvc-repos/smart_watch_dvc
    DVC_SUBPATH=$DVC_DPATH/drop1_S2_aligned_c1
    CUDA_VISIBLE_DEVICES=0 \
    python -m watch.tasks.fusion.fit \
        --train_dataset=$DVC_SUBPATH/train_data.kwcoco.json \
        --vali_dataset=$DVC_SUBPATH/vali_data.kwcoco.json \
        --time_steps=8 \
        --channels="coastal|blue|green|red|nir|swir16|swir22" \
        --chip_size=192 \
        --method="MultimodalTransformerDotProdCD" \
        --model_name=smt_it_stm_p8 \
        --batch_size=2 \
        --accumulate_grad_batches=8 \
        --num_workers=12 \
        --gpus=1
    2>/dev/null


    # Takes ~14GB on a 3090
    DVC_DPATH=$HOME/data/dvc-repos/smart_watch_dvc
    DVC_SUBPATH=$DVC_DPATH/drop1_S2_aligned_c1
    CUDA_VISIBLE_DEVICES=0 \
    python -m watch.tasks.fusion.fit \
        --train_dataset=$DVC_SUBPATH/train_data.kwcoco.json \
        --vali_dataset=$DVC_SUBPATH/vali_data.kwcoco.json \
        --time_steps=7 \
        --channels="coastal|blue|green|red|nir|swir16|swir22" \
        --chip_size=192 \
        --chip_overlap=0.66 \
        --time_overlap=0.3 \
        --method="MultimodalTransformerDotProdCD" \
        --model_name=smt_it_stm_small \
        --batch_size=4 \
        --accumulate_grad_batches=4 \
        --num_workers=12 \
        --gpus=1

    2>/dev/null

    # Can run on a 1080ti
    DVC_DPATH=$HOME/data/dvc-repos/smart_watch_dvc
    DVC_SUBPATH=$DVC_DPATH/drop1-S2-L8-LS-aligned-v2
    CUDA_VISIBLE_DEVICES=1 \
    python -m watch.tasks.fusion.fit \
        --train_dataset=$DVC_SUBPATH/train_data.kwcoco.json \
        --vali_dataset=$DVC_SUBPATH/vali_data.kwcoco.json \
        --time_steps=7 \
        --channels="coastal|blue|green|red|nir|swir16|swir22" \
        --chip_size=192 \
        --method="MultimodalTransformerDirectCD" \
        --model_name=smt_it_stm_p8 \
        --batch_size=1 \
        --accumulate_grad_batches=8 \
        --num_workers=12 \
        --gpus=1 2>/dev/null

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
    >>> fit_model(cmdline=cmdline, **kwargs)


Example:
    >>> # [WIP] Demo for end-to-end fit-predict-test pipeline
    >>> from watch.tasks.fusion.fit import *  # NOQA
    >>> from os.path import join
    >>> import os
    >>> import kwcoco
    >>> train_dset = kwcoco.CocoDataset.demo('special:vidshapes8-multispectral', num_frames=5, gsize=(128, 128))
    >>> vali_dset = kwcoco.CocoDataset.demo('special:vidshapes4-multispectral', num_frames=10, gsize=(128, 128), num_tracks=3)
    >>> available_channel_profiles = {
    >>>     frozenset(aux.get('channels', None) for aux in img.get('auxiliary', []))
    >>>      for img in train_dset.index.imgs.values()}
    >>> print('available_channel_profiles = {!r}'.format(available_channel_profiles))
    >>> kwargs = {
    ...     'train_dataset': train_dset.fpath,
    ...     'vali_dataset': vali_dset.fpath,
    ...     'dataset': 'WatchDataModule',
    ...     'method': 'MultimodalTransformerDirectCD',
    ...     'channels': 'B11|B1|B10|B8a',
    ...     'time_steps': 4,
    ...     'chip_size': 96,
    ...     'batch_size': 2,
    ...     'accumulate_grad_batches': 4,
    ...     'model_name': 'smt_it_stm_p8',
    ...     'num_workers': 2,
    ...     'gradient_clip_val': 0.5,
    ...     'gradient_clip_algorithm': 'value',
    ...     'gpus': 1,
    ... }
    >>> cmdline = False
    >>> #modules = make_lightning_modules(args=None, cmdline=cmdline, **kwargs)
    >>> fit_model(cmdline=cmdline, **kwargs)

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

    def __init__(self, num_draw=4, draw_interval=10):
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


@profile
def make_fit_config(cmdline=False, **kwargs):
    """
    Args:
        args : namespace that overrides defaults
        cmdline (bool): if True, will override defaults based on sys.argv
        **kwargs: dictionary that overrides defaults

    Example:
        >>> from watch.tasks.fusion.fit import *  # NOQA
        >>> cmdline = False
        >>> kwargs = {}
        >>> args = make_fit_config(cmdline=cmdline, **kwargs)
        >>> print('args.__dict__ = {}'.format(ub.repr2(args.__dict__, nl=1, sort=0)))
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

    config_parser.add_argument('--profile', action='store_true', help=ub.paragraph(
        '''
        Fit does nothing with this flag. This just allows for `@xdev.profile`
        profiling.
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
        '--method', default='MultimodalTransformerDirectCD',
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

    # I strongly recommend that ~/data is a symlink to a drive with more
    # storage space.
    default_workdir = './_trained_models'

    # Write to a sensible default location
    ENABLE_SMART_DEFAULT_WORKDIR = 1
    if ENABLE_SMART_DEFAULT_WORKDIR:
        dvc_repos_dpath = pathlib.Path('~/data/dvc-repos/').expanduser()
        if dvc_repos_dpath.exists():
            smart_dvc_dpath = dvc_repos_dpath / 'smart_watch_dvc'
            if smart_dvc_dpath.exists():
                import platform
                import getpass
                user_info = {
                    'user': getpass.getuser(),
                    'hostname': platform.node(),
                }
                default_workdir = (smart_dvc_dpath / 'experiments' /
                                   user_info['user'] / user_info['hostname'])
                default_workdir.mkdir(exist_ok=True)

    common_parser = parser.add_argument_group("Common")
    common_parser.add_argument(
        '--workdir', default=default_workdir,
        help=ub.paragraph(
            '''
            Directory where training data can be written.
            Overrides default_root_dir.
            ''')
    )

    # import netharn as nh
    # xpu = nh.XPU.coerce('auto')
    # auto_device = xpu.device.index

    import netharn as nh
    has_gpu = nh.XPU.coerce('auto').device.type == 'cpu'

    # Get subcomponents
    method_class = getattr(methods, modal.method)
    dataset_class = getattr(datasets, modal.dataset)

    # Extend the parser based on the chosen dataset / method modes
    dataset_class.add_data_specific_args(parser)
    method_parser = parser.add_argument_group("Method")
    method_class.add_model_specific_args(method_parser)
    pl.Trainer.add_argparse_args(parser)

    # Hard code custom default settings for lightning to enable certain tricks
    # by default
    parser.set_defaults(**{
        'default_root_dir': None,  # we override the default based on workdir
        'gradient_clip_val': 0.5,
        'gradient_clip_algorithm': 'value',
        'train_dataset': 'special:vidshapes8-multispectral',
        'vali_dataset': None,
        'test_dataset': None,
        'num_workers': 4,
        'gpus': 1 if has_gpu else None,
        'auto_select_gpus': True,
    })

    # override modal-specific defaults with user settings
    parser.set_defaults(**kwargs)

    if isinstance(cmdline, list):
        args = cmdline
    else:
        # setting args to [] disable command line parsting
        args = None if cmdline else []

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

    if args.default_root_dir is None:
        args.default_root_dir = pathlib.Path(args.workdir) / args.train_name

    # TODO:
    # Add dump and --dumps commands to write the config to a file
    # similar to how scriptconfig works
    return args


@profile
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

    args_dict = args.__dict__
    print("{train_name}\n====================".format(**args_dict))
    print('args_dict = {}'.format(ub.repr2(args_dict, nl=1, sort=0)))

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
        print('datamodule.input_stats = {}'.format(ub.repr2(datamodule.input_stats, nl=2, sort=0)))
        method_var_dict["input_stats"] = datamodule.input_stats
    # Note: Changed name from method to model
    model = method_class(**method_var_dict)

    # init trainer from args

    callbacks = [
        DrawBatchCallback(),
        pl.callbacks.LearningRateMonitor(logging_interval='epoch'),
        pl.callbacks.LearningRateMonitor(logging_interval='step'),
        pl.callbacks.ModelCheckpoint(monitor='loss', mode='min', save_top_k=1),
    ]
    if args.vali_dataset is not None:
        callbacks += [
            pl.callbacks.ModelCheckpoint(
                monitor='val_loss', mode='min', save_top_k=4),
        ]

    # TODO:
    # - [ ] Save multiple checkpoints based on metrics
    # https://github.com/PyTorchLightning/pytorch-lightning/issues/2908
    trainer = pl.Trainer.from_argparse_args(args, callbacks=callbacks)

    modules = {
        'datamodule': datamodule,
        'model': model,
        'trainer': trainer,
    }
    return modules


@profile
def fit_model(args=None, cmdline=False, **kwargs):
    """
    Example:
        >>> # xdoctest: +REQUIRES(--gpu)
        >>> from watch.tasks.fusion.fit import *  # NOQA
        >>> args = None
        >>> cmdline = False
        >>> workdir = ub.ensure_app_cache_dir('watch', 'tests', 'fusion', 'fit')
        >>> kwargs = {
        ...     'train_dataset': 'special:vidshapes8-multispectral',
        ...     'vali_dataset': 'special:vidshapes2-multispectral',
        ...     'test_dataset': 'special:vidshapes1-multispectral',
        ...     'dataset': 'WatchDataModule',
        ...     'workdir': workdir,
        ...     'gpus': 1,
        ...     'max_epochs': 3,
        ...     #'max_steps': 1,
        ...     'learning_rate': 1e-5,
        ...     'num_workers': 1,
        ... }
        >>> #args = make_fit_config(args=None, cmdline=cmdline, **kwargs)
        >>> #print('args.__dict__ = {}'.format(ub.repr2(args.__dict__, nl=1)))
        >>> fit_model(**kwargs)
    """
    modules = make_lightning_modules(cmdline=cmdline, **kwargs)
    trainer = modules['trainer']
    datamodule = modules['datamodule']
    model = modules['model']

    print(ub.repr2(utils.model_json(model, max_depth=3), nl=-1, sort=0))

    # prime the model, incase it has a lazy layer
    batch = next(iter(datamodule.train_dataloader()))

    # batch_shapes = ub.map_vals(lambda x: x.shape, batch)
    # print('batch_shapes = {}'.format(ub.repr2(batch_shapes, nl=1)))

    # result = model(batch["images"][[0], ...].float())
    import torch
    with torch.set_grad_enabled(False):
        model.forward_step(batch)

    # if requested, tune model with lightning default tuners
    trainer.tune(model, datamodule)

    # fit the model
    trainer.fit(model, datamodule)

    # TODO: Package the best epoch based on validation metrics
    package_fpath = pathlib.Path(trainer.default_root_dir) / "package.pt"

    # Record the dataset hparams this was trained with.
    model.datamodule_hparams = model.trainer.datamodule.hparams
    # Unload non-picklable parts from the data module
    # Get rid of problematic pickel variables
    # (is this desirable?)
    model.trainer = None
    model.train_dataloader = None
    model.val_dataloader = None
    model.test_dataloader = None

    # save model to package
    utils.create_package(model, package_fpath)

    return package_fpath

    # save learning relevant training options
    # learning_config = ub.dict_diff(args.__dict__, learning_irrelevant)


@profile
def main(**kwargs):
    """

    CommandLine:
        # Set the path to your data
        DVC_DPATH=$HOME/Projects/smart_watch_dvc
        DVC_DPATH=$HOME/data/dvc-repos/smart_watch_dvc
        TRAIN_FPATH=$DVC_DPATH/extern/onera_2018/onera_train.kwcoco.json

        # View the help docs
        python -m watch.tasks.fusion.fit --help

        python -m watch.tasks.fusion.fit --dumps

        # Running without any args should train a demo model
        python -m watch.tasks.fusion.fit

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
    import logging
    # configure logging at the root level of lightning
    logging.getLogger("pytorch_lightning").setLevel(logging.DEBUG)

    fit_model(cmdline=True, **kwargs)


if __name__ == "__main__":
    # import xdev
    # xdev.make_warnings_print_tracebacks()

    main()
