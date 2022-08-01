#!/usr/bin/env python
"""
Trains a fusion machine learning model on target dataset.


SeeAlso:
    README.md
    fit.py
    predict.py
    evaluate.py
    experiments/crall/onera_experiments.sh
    experiments/crall/drop1_experiments.sh
    experiments/crall/toy_experiments.sh


CommandLine:
    CUDA_VISIBLE_DEVICES=1 DVC_DPATH=$HOME/data/dvc-repos/smart_watch_dvc xdoctest -m watch.tasks.fusion.fit __doc__:0 -- --profile
    CUDA_VISIBLE_DEVICES=1 DVC_DPATH=$HOME/data/dvc-repos/smart_watch_dvc xdoctest -m watch.tasks.fusion.fit __doc__:0
    CUDA_VISIBLE_DEVICES=0 DVC_DPATH=$HOME/data/dvc-repos/smart_watch_dvc xdoctest -m watch.tasks.fusion.fit __doc__:0


    python -m watch.tasks.fusion.fit --help
    python -m watch.tasks.fusion.fit --dump foo.yml


Example:
    >>> # xdoctest: +REQUIRES(env:DVC_DPATH)
    >>> from watch.tasks.fusion.fit import *  # NOQA
    >>> import watch
    >>> dvc_dpath = watch.utils.util_data.find_smart_dvc_dpath()
    >>> train_fpath = dvc_dpath / 'drop1-S2-L8-aligned/train_data.kwcoco.json'
    >>> vali_fpath = dvc_dpath / 'drop1-S2-L8-aligned/vali_data.kwcoco.json'

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
    ...     'datamodule': 'KWCocoVideoDataModule',
    ...     'method': 'MultimodalTransformer',
    ...     'channels': 'coastal|blue|green|red|nir|swir16|swir22',
    ...     #'channels': 'blue|green|red|nir',
    ...     #'channels': None,
    ...     'time_steps': 8,
    ...     #'chip_size': 128,
    ...     'chip_size': 192,
    ...     #'chip_size': 256,
    ...     'batch_size': 1,
    ...     'accumulate_grad_batches': 12,
    ...     'arch_name': 'smt_it_stm_p8',
    ...     'num_workers': 12,
    ...     'attention_impl': 'exact',
    ...     #'attention_impl': 'performer',  # note: exact seems to be faster and less memory at this scale
    ...     'gradient_clip_val': 0.5,
    ...     'gradient_clip_algorithm': 'value',
    ...     'devices': 1,
    ... }
    >>> #modules = make_lightning_modules(args=None, cmdline=cmdline, **kwargs)
    >>> fit_model(cmdline=cmdline, **kwargs)
"""
import pytorch_lightning as pl
import ubelt as ub
import platform
import getpass
from os.path import join

from watch.utils import lightning_ext as pl_ext

try:
    import xdev
    profile = xdev.profile
except Exception:
    profile = ub.identity

available_methods = [
    'MultimodalTransformer',
]

available_datamodules = [
    'KWCocoVideoDataModule',
]

# TODO: is there a better way to mark these?
learning_irrelevant = {
    'workdir',
    'num_workers',
    'devices',
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
    'move_metrics_to_cpu',
    'distributed_backend',
}


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
        >>> args, parser = make_fit_config(cmdline=cmdline, **kwargs)
        >>> print('args.__dict__ = {}'.format(ub.repr2(args.__dict__, nl=1, sort=0)))
    """
    from watch.utils import configargparse_ext
    from watch.tasks.fusion import datamodules
    from watch.tasks.fusion import methods

    parser = configargparse_ext.ArgumentParser(
        add_config_file_help=False,
        description='Training script for the fused change/segmentation task',
        auto_env_var_prefix='WATCH_FUSION_FIT_',
        add_env_var_help=True,
        formatter_class='defaults',
        config_file_parser_class='yaml',
        args_for_setting_config_path=['--config'],
        args_for_writing_out_config_file=['--dump'],
    )

    # Setup scriptconfig-like special arguments to set a config via a file or
    # dump some config to stdout or disk
    config_parser = parser.add_argument_group('Other')

    parser.add_argument('--profile', action='store_true', help=ub.paragraph(
        '''
        Fit does nothing with this flag. This just allows for `@xdev.profile`
        profiling which checks sys.argv separately.
        '''))

    config_parser.add_argument('--init', default='noop', help=ub.paragraph(
        '''
        Initialization strategy. Can be a path to a pretrained network.
        '''))

    config_parser.add_argument('--eval_after_fit', default=False, help=ub.paragraph(
        '''
        If true, attempts to run default prediction and evaluation on the final
        packaged state.
        '''))

    callback_parser = parser.add_argument_group('Callbacks')

    # our extension callbacks have arg parsers
    pl_ext.callbacks.BatchPlotter.add_argparse_args(callback_parser)
    pl_ext.callbacks.Packager.add_argparse_args(callback_parser)

    callback_parser.add_argument('--patience', default=100, type=int, help=ub.paragraph(
        '''Number of epochs with no improvement before early stopping'''))

    # Setup common fields and modal switches
    modal_parser = parser.add_argument_group('Modal')

    modal_parser.add_argument(
        '--datamodule', choices=available_datamodules, default='KWCocoVideoDataModule',
        help=ub.paragraph(
            '''
            Modal parameter indicating the family of datamodule to train on.
            See the watch.tasks.fusion.datamodules submodule for details
            '''))

    modal_parser.add_argument(
        '--method', default='MultimodalTransformer',
        choices=available_methods, help=ub.paragraph(
            '''
            Modal parameter indicating the family of model to train.
            See the watch.tasks.fusion.methods submodule for details
            # TODO: change name to model?
            ''')
    )

    # override common defaults with user settings
    parser.set_defaults(**kwargs)
    # The specific parser will depend on the modal arguments
    modal, _ = parser.parse_known_args(ignore_help_args=True,
                                       ignore_write_args=True)

    # NOTE: if default_root_dir is specified, this workdir can be ignored
    # I strongly recommend that ~/data is a symlink to a drive with more
    # storage space.
    default_workdir = './_trained_models'
    ENABLE_SMART_DEFAULT_WORKDIR = 1
    if ENABLE_SMART_DEFAULT_WORKDIR:
        # Write to a sensible default location instead of CWD
        dvc_repos_dpath = ub.Path('~/data/dvc-repos/').expanduser()
        if dvc_repos_dpath.exists():
            smart_dvc_dpath = dvc_repos_dpath / 'smart_watch_dvc'
            if smart_dvc_dpath.exists():
                user_info = {
                    'user': getpass.getuser(),
                    'hostname': platform.node(),
                }
                default_workdir = (smart_dvc_dpath / 'experiments' /
                                   user_info['user'] / user_info['hostname'])

    common_parser = parser.add_argument_group('Common')
    common_parser.add_argument(
        '--workdir', default=str(default_workdir),
        help=ub.paragraph(
            '''
            Directory where training data can be written.
            This will override the default_root_dir.
            ''')
    )

    # import netharn as nh
    # xpu = nh.XPU.coerce('auto')
    # auto_device = xpu.device.index
    # import netharn as nh
    # has_gpu = nh.XPU.coerce('auto').device.type == 'cpu'

    # Get subcomponents
    method_class = getattr(methods, modal.method)
    datamodule_class = getattr(datamodules, modal.datamodule)

    # Extend the parser based on the chosen dataset / method modes
    datamodule_class.add_argparse_args(parser)
    # method_parser = parser.add_argument_group("Method")
    method_class.add_argparse_args(parser)
    # from watch.utils import util_globals
    # import watch.utils

    pl.Trainer.add_argparse_args(parser)

    # Hard code custom default settings for lightning to enable certain tricks
    # by default
    parser.set_defaults(**{
        'default_root_dir': None,  # we override the default based on workdir

        # Trick Defaults
        'gradient_clip_val': 0.5,
        'gradient_clip_algorithm': 'value',

        # Data defaults
        'train_dataset': 'special:vidshapes8-multispectral',
        'vali_dataset': None,
        'test_dataset': None,

        'num_workers': 4,

        'accumulate_grad_batches': 1,

        'max_epochs': None,
        'max_steps': -1,
        'max_time': None,

        'check_val_every_n_epoch': 1,

        'log_every_n_steps': 50,
    })

    # override modal-specific defaults with user settings
    parser.set_defaults(**kwargs)

    if isinstance(cmdline, list):
        args = cmdline
    else:
        # setting args to [] disable command line parsting
        args = None if cmdline else []

    args, _ = parser.parse_known_args(args=args)

    if args.normalize_inputs == 'True':
        args.normalize_inputs = True
    if args.normalize_inputs == 'False':
        args.normalize_inputs = False

    if args.max_steps is None:
        args.max_steps = -1  # Hack to supress warning

    # print('args.__dict__ = {}'.format(ub.repr2(args.__dict__, nl=1)))
    learning_config = ub.dict_diff(args.__dict__, learning_irrelevant)

    # Construct a netharn-like training directory based on relevant hyperparams
    args.train_hashid = ub.hash_data(ub.map_vals(str, learning_config))[0:16]
    args.train_name = '{method}-{train_hashid}'.format(**args.__dict__)

    if args.default_root_dir is None:
        args.default_root_dir = ub.Path(args.workdir) / args.train_name
    return args, parser


@profile
def make_lightning_modules(args=None, cmdline=False, **kwargs):
    """

    CommandLine:
        xdoctest -m /home/joncrall/code/watch/watch/tasks/fusion/fit.py make_lightning_modules

    Example:
        >>> from watch.tasks.fusion.fit import *  # NOQA
        >>> args = None
        >>> cmdline = False
        >>> kwargs = {
        ...     'train_dataset': 'special:vidshapes8-multispectral',
        ...     'datamodule': 'KWCocoVideoDataModule',
        ... }
        >>> modules = make_lightning_modules(args=None, cmdline=cmdline, **kwargs)
    """
    from watch.tasks.fusion import datamodules
    from watch.tasks.fusion import methods
    args, parser = make_fit_config(args=args, cmdline=cmdline, **kwargs)

    args_dict = args.__dict__
    print('{train_name}\n===================='.format(**args_dict))
    print('args_dict = {}'.format(ub.repr2(args_dict, nl=1, sort=1)))

    ub.Path(args.workdir).ensuredir()

    method_class = getattr(methods, args.method)
    datamodule_class = getattr(datamodules, args.datamodule)

    # init datamodule from args
    datamodule_vars = datamodule_class.compatible(args.__dict__)
    datamodule = datamodule_class(**datamodule_vars)
    datamodule.setup('fit')

    # init method from args
    method_var_dict = args.__dict__

    # TODO: need a better way to indicate that a method needs parameters from a
    # datamodule, and maybe the reverse too
    if hasattr(datamodule_class, 'bce_weight'):
        method_var_dict['pos_weight'] = getattr(datamodule_class, 'bce_weight')

    method_var_dict = method_class.compatible(method_var_dict)

    _needs_transfer = False
    if args.resume_from_checkpoint is None:
        _needs_transfer = True
        import netharn as nh
        init_cls, init_kw = nh.api.Initializer.coerce(init=args.init)
        other_model = None
        if 'fpath' in init_kw:
            # Hack: try and add support for torch.package
            # from torch import package
            try:
                from watch.tasks.fusion import utils
                other_model = utils.load_model_from_package(init_kw['fpath'])
            except Exception:
                pass
            else:
                import torch
                import tempfile
                tfile = tempfile.NamedTemporaryFile()

                state_dict = other_model.state_dict()

                # HACK:
                # Remove the normalization keys, we don't want to transfer them
                # in this step. They will be set correctly depending on if
                # normalize_inputs=transfer or not.
                bad_keys = [key for key in state_dict if 'input_norms' in key]
                for k in bad_keys:
                    state_dict.pop(k)
                print(ub.repr2(sorted(state_dict.keys())))

                torch.save(state_dict, tfile.name)
                init_kw['fpath'] = tfile.name
        initializer = init_cls(**init_kw)

    if hasattr(datamodule, 'dataset_stats'):

        # TODO: Allow manual override of any of the dataset stats or allow them
        # to be combined with a prior with some level of confidence.

        # Compute mean/std
        # TODO: allow for hardcoding per-sensor/channel mean/std in the
        # heuristics and then using those to have the option to skip computing
        # them for new datasets.
        # method_var_dict['input_channels'] = datamodule.input_channels
        method_var_dict['input_sensorchan'] = datamodule.input_sensorchan

        if args.normalize_inputs == 'transfer':
            assert other_model is not None
            method_var_dict['dataset_stats'] = other_model.dataset_stats
            print('other_model.dataset_stats = {}'.format(
                ub.repr2(other_model.dataset_stats, nl=3, sort=0)))
        else:
            print('datamodule.dataset_stats = {}'.format(
                ub.repr2(datamodule.dataset_stats, nl=3, sort=0)))
            method_var_dict['dataset_stats'] = datamodule.dataset_stats

    method_var_dict['classes'] = datamodule.classes
    # Note: Changed name from method to model
    model = method_class(**method_var_dict)

    # Tell the datamodule what tasks the datasets will need to generate data
    # for.
    datamodule._notify_about_tasks(model=model)

    if _needs_transfer:
        # Execute transfer
        # NOTE: This will overwrite any new dataset mean/std
        # TODO: allow the user to specify if they want to use new stats or old
        # stats when training this model.
        info = initializer(model)  # NOQA

    # init trainer from args
    if 1:
        # OLD:
        #     ('S2', 'blue|green|red'): {
        #         'mean': np.array([[[311.356341]],[[523.429458]],[[690.607918]]], dtype=np.float64),
        #         'std': np.array([[[2129.805325]],[[2222.820243]],[[2348.7701  ]]], dtype=np.float64),
        #     },
        #     ('L8', 'blue|green|red'): {
        #         'mean': np.array([[[ 949.545103]],[[1397.652219]],[[1568.746219]]], dtype=np.float64),
        #         'std': np.array([[[ 817.128624]],[[ 986.585357]],[[1210.746754]]], dtype=np.float64),
        #     },
        # },
        #
        callbacks = [
            # pl_ext.callbacks.AutoResumer(),
            # pl_ext.callbacks.StateLogger(),
            pl_ext.callbacks.TextLogger(args),
            pl_ext.callbacks.Packager(package_fpath=args.package_fpath),
            pl_ext.callbacks.BatchPlotter(
                num_draw=args.num_draw,
                draw_interval=args.draw_interval
            ),
            pl_ext.callbacks.TensorboardPlotter(),
            pl.callbacks.LearningRateMonitor(logging_interval='epoch', log_momentum=True),
            pl.callbacks.LearningRateMonitor(logging_interval='step', log_momentum=True),

            pl.callbacks.ModelCheckpoint(monitor='train_loss', mode='min', save_top_k=1),
            # pl.callbacks.GPUStatsMonitor(),  # enabling this breaks CPU tests
        ]
        if args.vali_dataset is not None:
            callbacks += [
                pl.callbacks.EarlyStopping(
                    monitor='val_loss', mode='min', patience=args.patience,
                    verbose=True, strict=False),
                pl.callbacks.ModelCheckpoint(
                    monitor='val_loss', mode='min', save_top_k=8),
                pl.callbacks.ModelCheckpoint(every_n_epochs=10),
            ]

            if datamodule.requested_tasks['change']:
                callbacks += [
                    pl.callbacks.ModelCheckpoint(
                        monitor='val_change_f1', mode='max', save_top_k=4),
                ]

            if datamodule.requested_tasks['saliency']:
                callbacks += [
                    pl.callbacks.ModelCheckpoint(
                        monitor='val_saliency_f1', mode='max', save_top_k=4),
                ]

            if datamodule.requested_tasks['class']:
                callbacks += [
                    pl.callbacks.ModelCheckpoint(
                        monitor='val_class_f1_micro', mode='max', save_top_k=4),
                    pl.callbacks.ModelCheckpoint(
                        monitor='val_class_f1_macro', mode='max', save_top_k=4),
                ]
    else:
        callbacks = []

    # TODO: explititly initialize the tensorboard logger?
    # logger = [
    #     pl.loggers.TensorBoardLogger(
    #         save_dir=args.default_root_dir, version=self.trainer.slurm_job_id, name="lightning_logs"
    #     )
    # ]

    # TODO:
    # - [ ] Save multiple checkpoints based on metrics
    # https://github.com/PyTorchLightning/pytorch-lightning/issues/2908
    trainer = pl.Trainer.from_argparse_args(args, callbacks=callbacks)

    print('trainer.logger.log_dir = {!r}'.format(trainer.logger.log_dir))
    # hack, this should be a callback, but it is not easy to pass the right
    # vars along without using lambdas, had issues with pickling objects
    if 1:
        import os
        import pathlib
        for k in dir(args):
            v = getattr(args, k)
            if isinstance(v, pathlib.Path):
                setattr(args, k, os.fspath(v))
        fpath = join(trainer.log_dir, 'fit_config.yaml')
        ub.Path(trainer.log_dir).ensuredir()
        parser.write_config_file(args, [fpath])

    modules = {
        'datamodule': datamodule,
        'model': model,
        'trainer': trainer,
        'args': args,
        'parser': parser,  # return parser so we can write the config
    }
    return modules


@profile
def fit_model(args=None, cmdline=False, **kwargs):
    """
    CommandLine:
        CUDA_VISIBLE_DEVICES=0 DVC_DPATH=$HOME/data/dvc-repos/smart_watch_dvc xdoctest -m watch.tasks.fusion.fit fit_model:0 -- --gpu

    Example:
        >>> # xdoctest: +REQUIRES(--gpu)
        >>> from watch.tasks.fusion.fit import *  # NOQA
        >>> args = None
        >>> cmdline = False
        >>> workdir = ub.ensure_app_cache_dir('watch', 'tests', 'fusion', 'fit')
        >>> kwargs = {
        ...     'train_dataset': 'special:vidshapes2-multispectral',
        ...     'vali_dataset': 'special:vidshapes1-multispectral',
        ...     'test_dataset': 'special:vidshapes1-multispectral',
        ...     'datamodule': 'KWCocoVideoDataModule',
        ...     'workdir': workdir,
        ...     'num_sanity_val_steps': 0,
        ...     'eval_after_fit': True,
        ...     'devices': 1,
        ...     'max_epochs': 2,
        ...     #'max_steps': 1,
        ...     'learning_rate': 1e-5,
        ...     'auto_lr_find': False,
        ...     'num_workers': 2,
        ... }
        >>> fit_model(**kwargs)
    """
    # cv2.setNumThreads(0)
    from watch.tasks.fusion import utils
    import warnings

    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', '.*GPU available but not used.*')
        modules = make_lightning_modules(cmdline=cmdline, **kwargs)

    # args = modules['args']
    trainer = modules['trainer']
    datamodule = modules['datamodule']
    model = modules['model']
    print(ub.repr2(utils.model_json(model, max_depth=1), nl=-1, sort=0))

    print('Tune if requested')
    # if requested, tune model with lightning default tuners
    tune_result = trainer.tune(model, datamodule)
    if tune_result:
        finder = tune_result['lr_find']
        print('finder.lr_max = {!r}'.format(finder.lr_max))
        print('finder.lr_min = {!r}'.format(finder.lr_min))
        suggestion_lr = finder.suggestion()
        print('suggestion_lr = {!r}'.format(suggestion_lr))

        if 0:
            import kwplot
            kwplot.autompl()
            finder.plot()
            kwplot.show_if_requested()
        print('tune_result = {}'.format(ub.repr2(tune_result, nl=1)))

    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', '.*is a deprecated alias for the builtin.*')
        warnings.filterwarnings('ignore', '.*GPU available but not used.*')
        warnings.filterwarnings('ignore', '.*Skipping val loop.*')
        warnings.filterwarnings('ignore', '.*does not have many workers which may be a bottleneck.*')
        warnings.filterwarnings('ignore', '.*Set a lower value for log_every_n_steps if you want to see logs for the training.*')

        # fit the model
        print('Fit starting')
        trainer.fit(model, datamodule)
        print('Fit finished')

    # Hack: what is the best way to get at this info?
    package_fpath = trainer.package_fpath

    args = modules['args']
    if args.eval_after_fit:
        if not package_fpath:
            print('package fpath was not set, so we cant eval')
        else:
            # Perhaps this doesn't happen in the same Python process maybe we
            # simply decouple this and force using a cmd-queue like scheduler?
            print('Attempting to unload resources after fit')
            # Unload fit resources
            trainer = None
            model = None
            datamodule = None
            modules = None

            import gc
            gc.collect()

            import torch
            torch.cuda.empty_cache()

            # TODO: evaluate multiple checkpoints?

            from watch.tasks.fusion import organize
            suggestions = organize.suggest_paths(
                test_dataset=args.test_dataset,
                package_fpath=package_fpath)
            import json
            suggestions = json.loads(suggestions)
            print('suggestions = {}'.format(ub.repr2(suggestions, nl=1)))
            from watch.tasks.fusion import predict
            from watch.tasks.fusion import evaluate
            predict_cfg = {
                'package_fpath': package_fpath,
                'test_dataset': args.test_dataset,
                'pred_dataset': suggestions['pred_dataset'],
                'num_workers': args.num_workers,
                'devices': args.devices,
            }
            eval_cfg = {
                'pred_dataset': suggestions['pred_dataset'],
                'true_dataset': args.test_dataset,
                'eval_dpath': suggestions['eval_dpath'],
            }
            predict.main(cmdline=False, **predict_cfg)
            evaluate.main(cmdline=False, **eval_cfg)

    # TODO:
    # Run prediction code here
    # Run evaluation code here
    return package_fpath


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
            --arch_name=smt_it_stm_p8 \
            --method=MultimodalTransformerDotProdCD \
            --datamodule=KWCocoVideoDataModule \
            --train_dataset=vidshapes8-multispectral \
            --batch_size=4 \
            --num_workers=4 \
            --chip_size=96 \
            --workdir=$HOME/work/watch/fit
    """
    # TODO: how to make this work in a distributed (ideally elastic train case)
    # def setup(rank, world_size):
    #     # https://pytorch.org/tutorials/intermediate/dist_tuto.html
    #     import torch.distributed as dist
    #     import os
    #     os.environ['MASTER_ADDR'] = 'localhost'
    #     os.environ['MASTER_PORT'] = '12355'
    #     # initialize the process group
    #     dist.init_process_group("gloo", rank=rank, world_size=world_size)
    # setup(0, 1)

    import logging
    # configure logging at the root level of lightning
    # logging.getLogger('pytorch_lightning').setLevel(logging.DEBUG)
    logging.getLogger('pytorch_lightning').setLevel(logging.INFO)
    fit_model(cmdline=True, **kwargs)


if __name__ == '__main__':
    if ub.argflag('--warntb'):
        import xdev
        xdev.make_warnings_print_tracebacks()
    # from watch.tasks.fusion import fit as this_module
    # this_module.main()
    main()
