"""
Defines a lightning DataModule for kwcoco video data.

The parameters to each are handled by scriptconfig objects, which prevents us
from needing to specify what the available options are in multiple places.
"""
import kwcoco
import kwimage
# import ndsampler
import pytorch_lightning as pl
import ubelt as ub
import scriptconfig as scfg

from geowatch.utils import util_globals
from kwutil import util_parallel
from geowatch.tasks.fusion import utils
from geowatch.tasks.fusion.datamodules.kwcoco_dataset import KWCocoVideoDatasetConfig, KWCocoVideoDataset

from typing import Dict

try:
    import xdev
    profile = xdev.profile
except Exception:
    profile = ub.identity


class KWCocoVideoDataModuleConfig(KWCocoVideoDatasetConfig):
    """
    These are the argument accepted by the KWCocoDataModule.

    The scriptconfig class is not used directly as it normally would be here.
    Instead we use it as a convinience to minimize lightning boilerplate later
    when it constructs its own argparse object, and for handling arguments
    passed directly to the KWCocoDataModule

    In the future this might be convertable to, or handled by omegaconfig
    """
    train_dataset = scfg.Value(None, help='path to the train kwcoco file', group='datasets')
    vali_dataset = scfg.Value(None, help='path to the validation kwcoco file', group='datasets')
    test_dataset = scfg.Value(None, help='path to the test kwcoco file', group='datasets')

    batch_size = scfg.Value(4, type=int, help=None)

    normalize_inputs = scfg.Value(True, help=ub.paragraph(
            '''
            if True, computes the mean/std for this dataset on each mode
            so this can be passed to the model.
            If set to a number it will only draw that many samples to estimate
            the mean/std.
            '''))

    num_workers = scfg.Value(4, type=str, alias=['workers'], help=ub.paragraph(
            '''
            number of background workers. Can be auto or an avail
            expression.
            '''))

    request_rlimit_nofile = scfg.Value('auto', help=ub.paragraph(
        '''
        As a convinience, on Linux systems this automatically requests that
        ulimit raises the maximum number of open files allowed. Auto currently
        simply sets this to 8192, so use a number higher than this if you run
        into too many open file errors, or set your ulimit explicitly before
        running this software.
        '''), group='resources')

    torch_sharing_strategy = scfg.Value('default', help=ub.paragraph(
            '''
            Torch multiprocessing sharing strategy. Can be 'default',
            "file_descriptor", "file_system". On linux, the default is
            "file_descriptor". See https://pytorch.org/docs/stable/multi
            processing.html#sharing-strategies for descriptions of
            options. When using sqlview=True, using "file_system" can
            help prevent the "received 0 items of ancdata" Error. It is
            unclear why using "file_descriptor" fails in this case for
            some datasets.
            '''), group='resources')

    torch_start_method = scfg.Value('default', help=ub.paragraph(
            '''
            Torch multiprocessing sharing strategy. Can be "default",
            "fork", "spawn", "forkserver". The default method on Linux
            is "spawn".
            '''), group='resources')

    sampler_backend = scfg.Value(None, help=ub.paragraph(
        '''
        Can be None, 'npy', or 'cog'.
        '''))

    test_with_annot_info = scfg.Value(False, isflag=1, help=ub.paragraph(
        '''
        If True, the test dataset is allowed to use annotations to refine the
        sampling. This is useful at predict time for drawing batches.
        '''))

    sqlview = scfg.Value(False, help=ub.paragraph(
            '''
            If False, reads the COCO dataset as a json file. Otherwise
            it can be sqlite or postgresql to cache json file in an SQL
            database for faster responce times and lower memory
            footprint.
            '''))


class KWCocoVideoDataModule(pl.LightningDataModule):
    """
    Prepare the kwcoco dataset as torch video datamodules

    Example:
        >>> # Demo of the data module on auto-generated toy data
        >>> from geowatch.tasks.fusion.datamodules.kwcoco_datamodule import *  # NOQA
        >>> import geowatch
        >>> import kwcoco
        >>> coco_dset = geowatch.coerce_kwcoco('vidshapes8-geowatch')
        >>> channels = None
        >>> batch_size = 1
        >>> time_steps = 3
        >>> chip_size = 416
        >>> self = KWCocoVideoDataModule(
        >>>     train_dataset=coco_dset,
        >>>     test_dataset=None,
        >>>     batch_size=batch_size,
        >>>     normalize_inputs=8,
        >>>     channels=channels,
        >>>     num_workers=0,
        >>>     time_steps=time_steps,
        >>>     chip_size=chip_size,
        >>>     neg_to_pos_ratio=0,
        >>> )
        >>> self.setup('fit')
        >>> dl = self.train_dataloader()
        >>> dataset = dl.dataset
        >>> batch = next(iter(dl))
        >>> batch = [dl.dataset[0]]
        >>> # Visualize
        >>> canvas = self.draw_batch(batch)
        >>> # xdoctest: +REQUIRES(--show)
        >>> import kwplot
        >>> kwplot.autompl()
        >>> kwplot.imshow(canvas)
        >>> kwplot.show_if_requested()

    Example:
        >>> # xdoctest: +REQUIRES(env:DVC_DPATH)
        >>> # Run the following tests on real geowatch data if DVC is available
        >>> from geowatch.tasks.fusion.datamodules.kwcoco_datamodule import *  # NOQA
        >>> import geowatch
        >>> import kwcoco
        >>> dvc_dpath = geowatch.find_dvc_dpath()
        >>> coco_fpath = dvc_dpath / 'Drop2-Aligned-TA1-2022-02-15/combo_ILM.kwcoco.json'
        >>> #coco_fpath = dvc_dpath / 'Aligned-Drop2-TA1-2022-03-07/combo_DILM.kwcoco.json'
        >>> #coco_fpath = dvc_dpath / 'Drop2-Aligned-TA1-2022-02-15/combo_DILM.kwcoco.json'
        >>> dset = kwcoco.CocoDataset(coco_fpath)
        >>> images = dset.images()
        >>> train_dataset = dset
        >>> #sub_images = dset.videos(names=['KR_R002']).images[0]
        >>> #train_dataset = dset.subset(sub_images.lookup('id'))
        >>> test_dataset = None
        >>> img = ub.peek(train_dataset.imgs.values())
        >>> chan_info = kwcoco_extensions.coco_channel_stats(dset)
        >>> #channels = chan_info['common_channels']
        >>> channels = 'blue|green|red|nir|swir16|swir22,forest|bare_ground,matseg_0|matseg_1|matseg_2,invariants.0:3,cloudmask'
        >>> #channels = 'blue|green|red|depth'
        >>> #chan_spec = kwcoco.channel_spec.FusedChannelSpec.coerce(channels)
        >>> #channels = None
        >>> #
        >>> batch_size = 1
        >>> time_steps = 8
        >>> chip_size = 512
        >>> datamodule = KWCocoVideoDataModule(
        >>>     train_dataset=train_dataset,
        >>>     test_dataset=test_dataset,
        >>>     batch_size=batch_size,
        >>>     channels=channels,
        >>>     num_workers=0,
        >>>     normalize_inputs=8,
        >>>     time_steps=time_steps,
        >>>     chip_size=chip_size,
        >>>     neg_to_pos_ratio=0,
        >>>     min_spacetime_weight=0.5,
        >>> )
        >>> datamodule.setup('fit')
        >>> dl = datamodule.train_dataloader()
        >>> dataset = dl.dataset
        >>> dataset.requested_tasks['change'] = False
        >>> dataset.disable_augmenter = True
        >>> target = 0
        >>> item, *_ = batch = [dataset[target]]
        >>> #item, *_ = batch = next(iter(dl))
        >>> # Visualize
        >>> canvas = datamodule.draw_batch(batch)
        >>> # xdoctest: +REQUIRES(--show)
        >>> import kwplot
        >>> kwplot.autompl()
        >>> kwplot.imshow(canvas, doclf=1)
        >>> kwplot.show_if_requested()

    Example:
        >>> # xdoctest: +SKIP
        >>> # NOTE: I DONT KNOW WHY THIS IS FAILING ON CI AT THE MOMENT. FIXME!
        >>> # Run the data module on coco demo datamodules for the CI
        >>> from geowatch.tasks.fusion.datamodules.kwcoco_datamodule import *  # NOQA
        >>> import kwcoco
        >>> train_dataset = kwcoco.CocoDataset.demo('vidshapes2-multispectral', num_frames=5)
        >>> test_dataset = kwcoco.CocoDataset.demo('vidshapes1-multispectral', num_frames=5)
        >>> channels = '|'.join([aux['channels'] for aux in train_dataset.imgs[1]['auxiliary']])
        >>> chan_spec = kwcoco.channel_spec.FusedChannelSpec.coerce(channels)
        >>> #
        >>> batch_size = 2
        >>> time_steps = 3
        >>> chip_size = 128
        >>> channels = channels
        >>> self = KWCocoVideoDataModule(
        >>>     train_dataset=train_dataset,
        >>>     test_dataset=test_dataset,
        >>>     batch_size=batch_size,
        >>>     channels=channels,
        >>>     num_workers=0,
        >>>     time_steps=time_steps,
        >>>     chip_size=chip_size,
        >>>     normalize_inputs=True,
        >>> )
        >>> self.setup('fit')
        >>> dl = self.train_dataloader()
        >>> item, *_ = batch = next(iter(dl))
        >>> expect_shape = (batch_size, time_steps, len(chan_spec), chip_size, chip_size)
        >>> assert len(batch) == batch_size
        >>> for item in batch:
        ...     assert len(item['frames']) == time_steps
        ...     for mode_key, mode_val in item['frames'][0]['modes'].items():
        ...         assert mode_val.shape[1:3] == (chip_size, chip_size)
    """

    __scriptconfig__ = KWCocoVideoDataModuleConfig

    def __init__(self, verbose=1, **kwargs):
        """
        For details on accepted arguments see KWCocoVideoDataModuleConfig
        """
        super().__init__()
        self.verbose = verbose
        self.config = KWCocoVideoDataModuleConfig(**kwargs)
        cfgdict = self.config.to_dict()
        self.save_hyperparameters(cfgdict)
        # Backwards compatibility. Previous iterations had the
        # config saved directly as datamodule arguments
        self.__dict__.update(cfgdict)
        self.train_kwcoco = self.config['train_dataset']
        self.vali_kwcoco = self.config['vali_dataset']
        self.test_kwcoco = self.config['test_dataset']

        if ub.WIN32:
            from kwutil import util_windows
            self.train_kwcoco = util_windows.fix_msys_path(self.train_kwcoco)
            self.vali_kwcoco = util_windows.fix_msys_path(self.vali_kwcoco)
            self.test_kwcoco = util_windows.fix_msys_path(self.test_kwcoco)

        common_keys = set(KWCocoVideoDatasetConfig.__default__.keys())
        # Pass the relevant parts of the config to the underlying datasets
        self.train_dataset_config = ub.dict_subset(cfgdict, common_keys)
        # with small changes made for validation and test datasets.
        self.vali_dataset_config = self.train_dataset_config.copy()
        self.vali_dataset_config['chip_overlap'] = 0.0
        # TODO: reconsider this hard-coded decision. It may bias our validation
        # check towards too many false positives. That is what we want if we
        # are having trouble there, but that setting should be configurable.
        self.vali_dataset_config['neg_to_pos_ratio'] = 0.0
        self.vali_dataset_config['use_grid_positives'] = True
        self.vali_dataset_config['use_centered_positives'] = False

        self.test_dataset_config = self.train_dataset_config.copy()
        self.test_dataset_config['test_with_annot_info'] = self.config.test_with_annot_info

        self.num_workers = util_parallel.coerce_num_workers(cfgdict['num_workers'])
        self.dataset_stats = None

        # will only correspond to train
        self.classes = None
        self.predictable_classes = None
        # self.input_channels = None
        self.input_sensorchan = None

        # Can we get rid of inject method?
        # Unfortunately lightning seems to only enable / disables
        # validation depending on the methods that are defined, so we are
        # not able to statically define them.
        ub.inject_method(self, lambda self: self._make_dataloader('train', shuffle=True), 'train_dataloader')

        # Store train / test / vali
        self.torch_datasets: Dict[str, KWCocoVideoDataset] = {}
        self.coco_datasets: Dict[str, kwcoco.CocoDataset] = {}

        self.requested_tasks = None
        self.did_setup = False

        if self.verbose:
            print('Init KWCocoVideoDataModule')
            print('self.train_kwcoco = {!r}'.format(self.train_kwcoco))
            print('self.vali_kwcoco = {!r}'.format(self.vali_kwcoco))
            print('self.test_kwcoco = {!r}'.format(self.test_kwcoco))
            print('self.input_sensorchan = {!r}'.format(self.input_sensorchan))
            print('self.time_steps = {!r}'.format(self.time_steps))
            print('self.chip_dims = {!r}'.format(self.chip_dims))
            print('self.window_space_scale = {!r}'.format(self.window_space_scale))
            print('self.input_space_scale = {!r}'.format(self.input_space_scale))
            print('self.output_space_scale = {!r}'.format(self.output_space_scale))

    def setup(self, stage):

        if self.did_setup:
            print('datamodules are already setup. Ignoring extra setup call')
            return

        import geowatch
        if self.verbose:
            print('Setup DataModule: stage = {!r}'.format(stage))

        util_globals.configure_global_attributes(**{
            'num_workers': self.num_workers,
            'torch_sharing_strategy': self.torch_sharing_strategy,
            'torch_start_method': self.torch_start_method,
            'request_rlimit_nofile': self.request_rlimit_nofile,
        })
        sqlview = self.config['sqlview']

        # Clear existing coco datasets so a reload occurs (should never happen
        # if the user doesnt touch `self.did_setup`).
        self.coco_datasets.clear()
        # make a temp mapping from train/vali/test to the specified coco inputs
        _coco_inputs = {
            'train': self.train_kwcoco,
            'vali': self.vali_kwcoco,
            'test': self.test_kwcoco,
        }

        def _read_kwcoco_split(_key):
            """
            Quick and dirty helper originally used to debug an issue. Keeping
            something similar to ensure train/test/vali kwcoco are read in the
            same way.

            This modifies the self.coco_datasets attribute.
            """
            _coco_input = _coco_inputs[_key]
            _coco_output = self.coco_datasets.get(_key, None)
            if _coco_output is None and _coco_input is not None:
                if self.verbose:
                    print(f'Read {_key} kwcoco dataset')
                # Use the demo coerce function to read the kwcoco file because
                # it allows for special demo inputs useful in doctests.
                _coco_output = geowatch.coerce_kwcoco(_coco_input, sqlview=sqlview)
                self.coco_datasets[_key] = _coco_output
            return _coco_output

        if stage == 'fit' or stage is None:
            train_coco_dset = _read_kwcoco_split('train')
            self.coco_datasets['train'] = train_coco_dset

            # HACK: load the validation kwcoco before we do any further
            # processing.
            _read_kwcoco_split('vali')

            if self.verbose:
                print('Build train kwcoco dataset')

            print('self.exclude_sensors', self.exclude_sensors)
            # coco_train_sampler = ndsampler.CocoSampler(train_coco_dset)
            coco_train_sampler = train_coco_dset
            train_dataset = KWCocoVideoDataset(
                coco_train_sampler, mode='fit', **self.train_dataset_config,
            )

            self.classes = train_dataset.classes
            self.predictable_classes = train_dataset.predictable_classes
            self.torch_datasets['train'] = train_dataset

            if self.input_sensorchan is None:
                self.input_sensorchan = train_dataset.input_sensorchan

            stats_params = {
                'num': None,
                'with_intensity': False,
                'with_class': True,
                'num_workers': self.num_workers,
                'batch_size': self.batch_size,
            }
            if isinstance(self.prenormalize_inputs, list):
                # The user specified normalization info
                ...

            if self.normalize_inputs:
                if isinstance(self.normalize_inputs, str):
                    if self.normalize_inputs == 'transfer':
                        # THIS MEANS WE EXPECT THAT WE CAN TRANSFER FROM AN
                        # EXISTING MODEL. THE FIT METHOD MUST HANDLE THIS
                        stats_params = None
                    else:
                        raise NotImplementedError(
                            'TODO: handle special normalization keys, '
                            'e.g. imagenet')
                else:
                    if isinstance(self.normalize_inputs, int):
                        stats_params['num'] = self.normalize_inputs
                    else:
                        stats_params['num'] = None
            else:
                stats_params['with_intensity'] = False

            # Hack for now:
            # TODO: Note: also need for class weights
            if stats_params is not None:
                print(f'stats_params={stats_params}')
                self.dataset_stats = train_dataset.cached_dataset_stats(**stats_params)

            if self.vali_kwcoco is not None:
                vali_coco_dset = _read_kwcoco_split('vali')
                if self.verbose:
                    print('Build validation kwcoco dataset')
                # vali_coco_sampler = ndsampler.CocoSampler(vali_coco_dset)
                vali_coco_sampler = vali_coco_dset
                vali_dataset = KWCocoVideoDataset(
                    vali_coco_sampler, mode='vali', **self.vali_dataset_config)
                self.torch_datasets['vali'] = vali_dataset
                ub.inject_method(self, lambda self: self._make_dataloader('vali', shuffle=False), 'val_dataloader')

        if stage == 'test' or stage is None:
            test_coco_dset = _read_kwcoco_split('test')
            if self.verbose:
                print('Build test kwcoco dataset')
            # test_coco_sampler = ndsampler.CocoSampler(test_coco_dset)
            test_coco_sampler = test_coco_dset
            self.coco_datasets['test'] = test_coco_dset
            self.torch_datasets['test'] = KWCocoVideoDataset(
                test_coco_sampler, mode='test', **self.test_dataset_config,
            )
            ub.inject_method(self, lambda self: self._make_dataloader('test', shuffle=False), 'test_dataloader')

        print('self.torch_datasets = {}'.format(ub.urepr(self.torch_datasets, nl=1)))
        self._notify_about_tasks(self.requested_tasks)
        self.did_setup = True

    # Can we use these instead of inject method?
    # def train_dataloader(self):
    #     return self._make_dataloader('train', shuffle=True)
    # def val_dataloader(self):
    #     return self._make_dataloader('vali', shuffle=True)
    # def test_dataloader(self):
    #     return self._make_dataloader('test', shuffle=True)

    @property
    def train_dataset(self):
        return self.torch_datasets.get('train', None)

    @property
    def test_dataset(self):
        return self.torch_datasets.get('test', None)

    @property
    def vali_dataset(self):
        return self.torch_datasets.get('vali', None)

    def _make_dataloader(self, stage, shuffle=False):
        """
        If the stage doesn't exist, resturns None.

        Returns:
            torch.utils.data.DataLoader | None
        """
        dataset = self.torch_datasets.get(stage, None)
        if dataset is None:
            return None
        loader = dataset.make_loader(
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=shuffle,
            pin_memory=True,
        )
        return loader

    def _notify_about_tasks(self, requested_tasks=None, model=None, predictable_classes=None):
        """
        Hacky method. Given the multimodal model, tell all the datasets which
        tasks they will need to generate data for. (This helps make the
        visualizations cleaner).
        """
        if model is not None:
            assert requested_tasks is None
            if hasattr(model, 'global_head_weights'):
                requested_tasks = {k: w > 0 for k, w in model.global_head_weights.items()}
            if hasattr(model, 'predictable_classes'):
                predictable_classes = model.predictable_classes
            else:
                import warnings
                warnings.warn(ub.paragraph(
                    f'''
                    Model {model.__class__} does not have the structure needed
                    to notify the dataset about tasks. A better design to make
                    specifying tasks easier is needed without relying on the
                    ``global_head_weights``.
                    '''))
        print(f'datamodule notified: requested_tasks={requested_tasks} predictable_classes={predictable_classes}')
        if requested_tasks is not None:
            self.requested_tasks = requested_tasks
            for dataset in self.torch_datasets.values():
                dataset._notify_about_tasks(requested_tasks, predictable_classes=predictable_classes)

    @classmethod
    def add_argparse_args(cls, parent_parser):
        """
        Previously the arguments were in multiple places including here.  This
        has been updated to use the :class:`KWCocoVideoDataModuleConfig` as the
        single point where arguments are defined. The functionality of this
        method is roughly the same as it used to be given that scriptconfig
        objects can be transformed into argparse objects.

        CommandLine:
            xdoctest -m /home/joncrall/code/watch/geowatch/tasks/fusion/datamodules/kwcoco_datamodule.py add_argparse_args

        Example:
            >>> from geowatch.tasks.fusion.datamodules.kwcoco_datamodule import *  # NOQA
            >>> cls = KWCocoVideoDataModule
            >>> # TODO: make use of geowatch.utils.lightning_ext import argparse_ext
            >>> import argparse
            >>> parent_parser = argparse.ArgumentParser()
            >>> cls.add_argparse_args(parent_parser)
            >>> parent_parser.print_help()
            >>> args, _ = parent_parser.parse_known_args(['--use_grid_positives=True'])
            >>> assert args.use_grid_positives
            >>> args, _ = parent_parser.parse_known_args(['--use_grid_positives=False'])
            >>> assert not args.use_grid_positives
            >>> args, _ = parent_parser.parse_known_args(['--exclude_sensors=l8,f3'])
            >>> assert args.exclude_sensors == 'l8,f3'
            >>> args, _ = parent_parser.parse_known_args(['--exclude_sensors=l8'])
            >>> assert args.exclude_sensors == 'l8'
        """
        # from functools import partial
        parser = parent_parser.add_argument_group('kwcoco_datamodule')
        config = KWCocoVideoDataModuleConfig()
        config.argparse(parser)
        return parent_parser

    @classmethod
    def compatible(cls, cfgdict):
        """
        Given keyword arguments, find the subset that is compatible with this
        constructor. This is somewhat hacked because of usage of scriptconfig,
        but could be made nicer by future updates.
        """
        # init_kwargs = ub.compatible(config, cls.__init__)
        import inspect
        nameable_kinds = {inspect.Parameter.POSITIONAL_OR_KEYWORD,
                          inspect.Parameter.KEYWORD_ONLY}
        cls_sig = inspect.signature(cls)
        explicit_argnames = [
            argname for argname, argtype in cls_sig.parameters.items()
            if argtype.kind in nameable_kinds
        ]
        valid_argnames = explicit_argnames + list(KWCocoVideoDataModuleConfig.__default__.keys())
        datamodule_vars = ub.dict_isect(cfgdict, valid_argnames)
        return datamodule_vars

    def draw_batch(self, batch, stage='train', outputs=None, max_items=2,
                   overlay_on_image=False, classes=None, **kwargs):
        r"""
        Visualize a batch produced by a KWCocoVideoDataset.

        Args:
            batch (Dict[str, List[Tensor]]): dictionary of uncollated lists of Dataset Items
                change: [ [T-1, H, W] \in [0, 1] \forall examples ]
                saliency: [ [T, H, W, 2] \in [0, 1] \forall examples ]
                class: [ [T, H, W, 10] \in [0, 1] \forall examples ]

            outputs (Dict[str, Tensor]):
                maybe-collated list of network outputs?

            max_items (int):
                Maximum number of items within this batch to draw in a single
                figure. Defaults to 2.

            overlay_on_image (bool):
                if True overlay annotations on image data for a more compact
                view. if False separate annotations / images for a less
                cluttered view.

        Example:
            >>> from geowatch.tasks.fusion.datamodules.kwcoco_datamodule import *  # NOQA
            >>> from geowatch.tasks.fusion import datamodules
            >>> self = datamodules.KWCocoVideoDataModule(
            >>>     train_dataset='special:vidshapes8-multispectral', channels='auto', num_workers=0)
            >>> self.setup('fit')
            >>> loader = self.train_dataloader()
            >>> batch = next(iter(loader))
            >>> item = batch[0]
            >>> # Visualize
            >>> B = len(batch)
            >>> C, H, W = ub.peek(item['frames'][0]['modes'].values()).shape
            >>> T = len(item['frames'])
            >>> import torch
            >>> outputs = {'change_probs': [torch.rand(T - 1, H, W) for _ in range(B)]}
            >>> outputs.update({'class_probs': [torch.rand(T, H, W, 10) for _ in range(B)]})
            >>> stage = 'train'
            >>> canvas = self.draw_batch(batch, stage=stage, outputs=outputs)
            >>> # xdoctest: +REQUIRES(--show)
            >>> import kwplot
            >>> kwplot.autompl()
            >>> kwplot.imshow(canvas)
            >>> kwplot.show_if_requested()


        Example:
            >>> # xdoctest: +REQUIRES(--slow)
            >>> from geowatch.tasks.fusion.datamodules.kwcoco_datamodule import *  # NOQA
            >>> from geowatch.tasks.fusion import datamodules
            >>> import geowatch
            >>> train_dataset = geowatch.demo.demo_kwcoco_multisensor()
            >>> self = datamodules.KWCocoVideoDataModule(
            >>>     train_dataset=train_dataset, chip_size=256, time_steps=5, num_workers=0, batch_size=3)
            >>> self.setup('fit')
            >>> loader = self.train_dataloader()
            >>> batch_iter = iter(loader)
            >>> batch = next(batch_iter)
            >>> batch[1] = None  # simulate a dropped batch item
            >>> batch[0] = None  # simulate a dropped batch item
            >>> #item = batch[0]
            >>> # Visualize
            >>> B = len(batch)
            >>> outputs = {'change_probs': [], 'class_probs': [], 'saliency_probs': []}
            >>> # Add dummy outputs
            >>> import torch
            >>> for item in batch:
            >>>     if item is None:
            >>>         [v.append([None]) for v in outputs.values()]
            >>>     else:
            >>>         [v.append([]) for v in outputs.values()]
            >>>         for frame_idx, frame in enumerate(item['frames']):
            >>>             H, W = frame['class_idxs'].shape
            >>>             if frame_idx > 0:
            >>>                 outputs['change_probs'][-1].append(torch.rand(H, W))
            >>>             outputs['class_probs'][-1].append(torch.rand(H, W, 10))
            >>>             outputs['saliency_probs'][-1].append(torch.rand(H, W, 2))
            >>> from geowatch.utils import util_nesting
            >>> print(ub.urepr(util_nesting.shape_summary(outputs), nl=1, sort=0))
            >>> stage = 'train'
            >>> canvas = self.draw_batch(batch, stage=stage, outputs=outputs, max_items=4)
            >>> # xdoctest: +REQUIRES(--show)
            >>> import kwplot
            >>> kwplot.autompl()
            >>> kwplot.imshow(canvas)
            >>> kwplot.show_if_requested()

        Example:
            >>> from geowatch.tasks.fusion.datamodules.kwcoco_datamodule import *  # NOQA
            >>> from geowatch.tasks.fusion import datamodules
            >>> self = datamodules.KWCocoVideoDataModule(
            >>>     batch_size = 3,
            >>>     train_dataset='special:vidshapes8-multispectral', channels='auto', num_workers=0)
            >>> self.setup('fit')
            >>> loader = self.train_dataloader()
            >>> batch = next(iter(loader))
            >>> batch[1] = None
            >>> item = batch[0]
            >>> # Visualize
            >>> B = len(batch)
            >>> C, H, W = ub.peek(item['frames'][0]['modes'].values()).shape
            >>> T = len(item['frames'])
            >>> import torch
            >>> outputs = {'change_probs': [torch.rand(T - 1, H, W) for _ in range(B)]}
            >>> outputs.update({'class_probs': [torch.rand(T, H, W, 10) for _ in range(B)]})
            >>> stage = 'train'
            >>> canvas = self.draw_batch(batch, stage=stage, outputs=outputs)
            >>> # xdoctest: +REQUIRES(--show)
            >>> import kwplot
            >>> kwplot.autompl()
            >>> kwplot.imshow(canvas)
            >>> kwplot.show_if_requested()
        """
        dataset = self.torch_datasets[stage]
        # Get the raw dataset class
        while hasattr(dataset, 'dataset'):
            dataset = dataset.dataset

        # assume collation is disabled
        batch_items = batch
        # batch_items = [ex for ex in batch if (ex is not None)]

        DEBUG_INCOMING_DATA = 1
        if DEBUG_INCOMING_DATA:
            stats = {}
            stats['batch_size'] = len(batch_items)
            stats['num_None_batch_items'] = 0
            for item_idx, item in enumerate(batch_items):
                if item is None:
                    stats['num_None_batch_items'] += 1

        KNOWN_HEADS = ['change_probs', 'class_probs', 'saliency_probs']

        canvas_list = []
        for item_idx, item in zip(range(max_items), batch_items):
            # HACK: I'm not sure how general accepting outputs is
            # TODO: more generic handling of outputs.
            # Should be able to accept
            # - [ ] binary probability of change
            # - [ ] fine-grained probability of change
            # - [ ] per-frame semenatic segmentation
            # - [ ] detections with box results!

            if item is None:
                continue

            if outputs is not None:
                # Extract outputs only for this specific batch item.
                item_output = ub.AutoDict()
                for head_key in KNOWN_HEADS:
                    if head_key in outputs:
                        item_output[head_key] = []
                        head_outputs = outputs[head_key]
                        head_item_output = head_outputs[item_idx]
                        if head_item_output is not None:
                            for frame_out in head_item_output:
                                item_output[head_key].append(frame_out.data.cpu().numpy())
                        else:
                            item_output[head_key].append(None)
            else:
                item_output = {}

            part = dataset.draw_item(
                item, item_output=item_output,
                overlay_on_image=overlay_on_image, classes=classes, **kwargs)

            canvas_list.append(part)

        num_images = len(canvas_list)
        # import xdev
        # xdev.embed()
        if 1:
            # Choose a sensible chunksize for the grid based on the input image
            # aspect ratios

            # TODO: could add this as a grid heuristic.
            import numpy as np
            hs = np.array([c.shape[0] for c in canvas_list])
            ws = np.array([c.shape[1] for c in canvas_list])

            h_majorness = hs > (ws * 1.2)
            w_majorness = ws > (hs * 1.2)
            if h_majorness.sum() >= w_majorness.sum():
                majors, minors = hs, ws
                stack_axis = 0
            else:
                majors, minors = ws, hs
                stack_axis = 1

            majors_per_minor = (majors / minors).mean()
            # Not sure if this is quite right
            chunksize = int(np.ceil(np.sqrt(majors_per_minor * num_images)))

            """
            import sympy as sym
            majors_per_minor, num_imgs = sym.symbols('majors_per_minor, num_imgs')
            real_grid_major, real_grid_minor = sym.symbols('real_grid_w, real_grid_h')
            ideal_grid_dim = sym.symbols('ideal_grid_dim')
            sym.sqrt(num_imgs)
            vars = (majors_per_minor, num_imgs, real_grid_major, real_grid_minor, ideal_grid_dim)

            # TODO: get the system that solves for the number of images we
            # stack across the minor dimension such that we roughly get a
            # square image in the end.

            equations = [
                sym.Eq(ideal_grid_dim * ideal_grid_dim, num_imgs * majors_per_minor),
                sym.Eq(majors_per_minor * ideal_grid_dim, real_grid_minor),
                sym.Eq(ideal_grid_dim, real_grid_major),
            ]
            print('equations = {}'.format(ub.urepr(equations, nl=1)))
            from sympy import solve
            solutions = solve(equations, *vars, dict=True)
            print('solutions = {}'.format(ub.urepr(solutions, nl=2)))
            solutions = solve(equations, real_grid_major, dict=True)
            print('solutions = {}'.format(ub.urepr(solutions, nl=2)))
            solutions = solve(equations, real_grid_minor, dict=True)
            print('solutions = {}'.format(ub.urepr(solutions, nl=2)))
            """
        else:
            stack_axis = 1
            chunksize = int(np.ceil(np.sqrt(num_images)))

        canvas = kwimage.stack_images_grid(
            canvas_list, chunksize=chunksize, axis=stack_axis, overlap=-12, bg_value=[64, 60, 60])

        with_legend = self.requested_tasks is None or self.requested_tasks.get('class', True)
        # with_legend = True
        if with_legend:
            if classes is None:
                classes = dataset.classes
            utils.category_tree_ensure_color(classes)
            label_to_color = {
                node: data['color']
                for node, data in classes.graph.nodes.items()}
            label_to_color = ub.sorted_keys(label_to_color)
            legend_img = utils._memo_legend(label_to_color)
            canvas = kwimage.stack_images([canvas, legend_img], axis=1)

        return canvas
