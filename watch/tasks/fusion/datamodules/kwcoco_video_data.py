"""
Defines a torch Dataset and lightning DataModule for kwcoco video data.

The parameters to each are handled by scriptconfig objects, which prevents us
from needing to specify what the available options are in multiple places.
"""
import os
import einops
import kwarray
import kwcoco
import kwimage
import ndsampler
import numpy as np
import pathlib
import pandas as pd
import pytorch_lightning as pl
# import random  # NOQA
import torch
import ubelt as ub
from kwcoco import channel_spec
from torch.utils import data
from typing import Dict, List  # NOQA
import scriptconfig as scfg


from watch import heuristics
from watch.utils import kwcoco_extensions
from watch.utils import util_bands
from watch.utils import util_iter
from watch.utils import util_kwimage
from watch.utils import util_time
from watch.utils.lightning_ext import util_globals
from watch.tasks.fusion import utils
from watch.tasks.fusion.datamodules.spacetime_grid_builder import sample_video_spacetime_targets

# __all__ = ['KWCocoVideoDataModule', 'KWCocoVideoDataset']

try:
    import xdev
    profile = xdev.profile
except Exception:
    profile = ub.identity


class KWCocoVideoDatasetConfig(scfg.Config):
    """
    This is the configuration for a single dataset that could be used for
    train, test, or validation.

    In the future this might be convertable to, or handled by omegaconfig
    """
    default = {
        'time_steps': scfg.Value(2, help='number of temporal sampler per batch'),

        'chip_size': scfg.Value(128, help='spatial width and height per batch. DEPRECATED. Use chip_dims instead.'),

        'chip_dims': scfg.Value(None, help=ub.paragraph(
            '''
            spatial height/width per batch. If given as a single number, used
            as both width and height. Default is currently taken from
            deprecated chip_size, but in the future will be 128.
            ''')),

        # 'time_overlap': scfg.Value(0.0, help='fraction of time steps to overlap'),
        'chip_overlap': scfg.Value(0.0, help='fraction of space steps to overlap'),

        'channels': scfg.Value(None, type=str, help=ub.paragraph(
            '''
            channels to use should be ChannelSpec coercable
            ''')),

        'diff_inputs': scfg.Value(False, help=ub.paragraph(
            '''
            if True, also includes a difference between consecutive
            frames in the inputs produced. NO LONGER WORKS
            ''')),

        'dist_weights': scfg.Value(0, help=ub.paragraph(
            '''
            To use distance-transform based weights on annotations or
            not
            ''')),

        'exclude_sensors': scfg.Value(None, type=str, help=ub.paragraph(
            '''
            comma delimited list of sensors to avoid, such as S2 or L8
            ''')),

        'ignore_dilate': scfg.Value(11, help='Dilation applied to ignore masks.'),

        'match_histograms': scfg.Value(False, help='undocumented - ignored'),

        'max_epoch_length': scfg.Value(None, help=ub.paragraph(
            '''
            If specified, restricts number of steps per epoch
            ''')),

        'min_spacetime_weight': scfg.Value(0.5, help='Minimum space-time dilation weight'),

        'normalize_perframe': scfg.Value(False, help='undocumented - ignored'),

        'resample_invalid_frames': scfg.Value(True, help=ub.paragraph(
            '''
            if True, will attempt to resample any frame without valid
            data
            ''')),

        'set_cover_algo': scfg.Value(None, choices=[None, 'approx', 'exact'], help=ub.paragraph(
            '''
            Set cover algorithm to remove redundant gids when building space
            time targets. Options are 'approx' (a greedy solution) or 'exact'
            (an ILP solution). If None is passed, set cover is not computed.
            The 'exact' method requires the pulp package (and can be very slow
            so it is generally not recommended).
            ''')),

        'temporal_dropout': scfg.Value(0.0, type=float, help=ub.paragraph(
            '''
            Drops frames in a fraction of training batches
            ''')),

        'time_sampling': scfg.Value('contiguous', type=str, help=ub.paragraph(
            '''
            Strategy for expanding the time window across non-contiguous
            frames. Can be auto, contiguous, hard+distribute, or
            dilate_affinity
            ''')),

        'time_span': scfg.Value('2y', type=str, help=ub.paragraph(
            '''
            how long a time window should roughly span by default
            ''')),

        'true_multimodal': scfg.Value(True, help=ub.paragraph(
            '''
            Enables new logic for sampling multimodal data. Old logic
            probably doesn't work anymore.
            ''')),

        'use_centered_positives': scfg.Value(False, help=ub.paragraph(
            '''
            Use centers of annotations as window centers
            ''')),

        'upweight_centers': scfg.Value(True, help='undocumented'),

        'use_cloudmask': scfg.Value(1, type=int, help=ub.paragraph(
            '''
            Allow the dataloader to use the cloud mask to skip frames
            ''')),

        'use_conditional_classes': scfg.Value(True, help=ub.paragraph(
            '''
            Include no-activity, post-construction in predictions when
            their conditions are met.
            ''')),

        'use_grid_positives': scfg.Value(True, help=ub.paragraph(
            '''
            Use annotation overlaps with grid as positives
            ''')),

        # Overwritten for non-train

        'neg_to_pos_ratio': scfg.Value(1.0, type=float, help=ub.paragraph(
            '''
            maximum ratio of samples with no annotations to samples with
            annots
            ''')),
    }

    def normalize(self):
        if isinstance(self['exclude_sensors'], str):
            self['exclude_sensors'] = [s.strip() for s in self['exclude_sensors'].split(',')]
        self['time_steps'] = int(self['time_steps'])

        if self['chip_dims'] is None:
            d = int(self['chip_size'])
            self['chip_dims'] = [d, d]  # has to be a list not a tuple for yaml

        self['chip_size'] = None


class KWCocoVideoDataModuleConfig(scfg.Config):
    """
    These are the argument accepted by the KWCocoDataModule.

    The scriptconfig class is not used directly as it normally would be here.
    Instead we use it as a convinience to minimize lightning boilerplate later
    when it constructs its own argparse object, and for handling arguments
    passed directly to the KWCocoDataModule

    In the future this might be convertable to, or handled by omegaconfig
    """
    default = ub.dict_union({
        'train_dataset': scfg.Value(None, help='path to the train kwcoco file'),
        'vali_dataset': scfg.Value(None, help='path to the validation kwcoco file'),
        'test_dataset': scfg.Value(None, help='path to the test kwcoco file'),

        'batch_size': scfg.Value(4, type=int),
        'normalize_inputs': scfg.Value(True, help=ub.paragraph(
            '''
            if True, computes the mean/std for this dataset on each mode
            so this can be passed to the model.
            ''')),

        'num_workers': scfg.Value(4, type=str, help=ub.paragraph(
            '''
            number of background workers. Can be auto or an avail
            expression. TODO: rename to data_workers?
            ''')),

        'torch_sharing_strategy': scfg.Value('default', help=ub.paragraph(
            '''
            Torch multiprocessing sharing strategy. Can be default,
            file_descriptor, file_system
            ''')),

        'torch_start_method': scfg.Value('default', help=ub.paragraph(
            '''
            Torch multiprocessing sharing strategy. Can be fork, spawn,
            forkserver
            ''')),
        # Mixin the dataset config
    }, KWCocoVideoDatasetConfig.default)

    def normalize(self):
        # hack because we dont have proper inheritence
        KWCocoVideoDatasetConfig.normalize(self)


class KWCocoVideoDataModule(pl.LightningDataModule):
    """
    Prepare the kwcoco dataset as torch video datamodules

    Example:
        >>> # Demo of the data module on auto-generated toy data
        >>> from watch.tasks.fusion.datamodules.kwcoco_video_data import *  # NOQA
        >>> import watch
        >>> import kwcoco
        >>> coco_dset = watch.coerce_kwcoco('vidshapes8-watch')
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
        >>> # Run the following tests on real watch data if DVC is available
        >>> from watch.tasks.fusion.datamodules.kwcoco_video_data import *  # NOQA
        >>> import watch
        >>> import kwcoco
        >>> dvc_dpath = watch.find_smart_dvc_dpath()
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
        >>>     use_conditional_classes=1,
        >>> )
        >>> datamodule.setup('fit')
        >>> dl = datamodule.train_dataloader()
        >>> dataset = dl.dataset
        >>> dataset.requested_tasks['change'] = False
        >>> dataset.disable_augmenter = True
        >>> tr = 0
        >>> item, *_ = batch = [dataset[tr]]
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
        >>> from watch.tasks.fusion.datamodules.kwcoco_video_data import *  # NOQA
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

    def __init__(self, verbose=1, **kwargs):
        """
        For details on accepted arguments see KWCocoVideoDataModuleConfig
        """
        super().__init__()
        self.verbose = verbose
        self.config = KWCocoVideoDataModuleConfig(cmdline=0, data=kwargs)
        cfgdict = self.config.to_dict()
        self.save_hyperparameters(cfgdict)
        # Backwards compatibility. Previous iterations had the
        # config saved directly as datamodule arguments
        print('cfgdict = {}'.format(ub.repr2(cfgdict, nl=1)))
        self.__dict__.update(cfgdict)
        self.train_kwcoco = self.config['train_dataset']
        self.vali_kwcoco = self.config['vali_dataset']
        self.test_kwcoco = self.config['test_dataset']

        common_keys = set(KWCocoVideoDatasetConfig.default.keys())
        # Pass the relevant parts of the config to the underlying datasets
        self.train_dataset_config = ub.dict_subset(cfgdict, common_keys)
        # with small changes made for validation and test datasets.
        self.vali_dataset_config = self.train_dataset_config.copy()
        self.vali_dataset_config['chip_overlap'] = 0.0
        self.vali_dataset_config['neg_to_pos_ratio'] = 0.0
        self.test_dataset_config = self.vali_dataset_config.copy()

        self.num_workers = util_globals.coerce_num_workers(cfgdict['num_workers'])
        self.dataset_stats = None

        # will only correspond to train
        self.classes = None
        self.input_channels = None
        self.input_sensorchan = None

        # Store train / test / vali
        self.torch_datasets: Dict[str, KWCocoVideoDataset] = {}
        self.coco_datasets: Dict[str, kwcoco.CocoDataset] = {}

        self.requested_tasks = None

        if self.verbose:
            print('Init KWCocoVideoDataModule')
            print('self.train_kwcoco = {!r}'.format(self.train_kwcoco))
            print('self.vali_kwcoco = {!r}'.format(self.vali_kwcoco))
            print('self.test_kwcoco = {!r}'.format(self.test_kwcoco))
            print('self.time_steps = {!r}'.format(self.time_steps))
            print('self.chip_dims = {!r}'.format(self.chip_dims))
            print('self.channels = {!r}'.format(self.channels))

    def setup(self, stage):
        import watch
        if self.verbose:
            print('Setup DataModule: stage = {!r}'.format(stage))

        util_globals.configure_global_attributes(**{
            'num_workers': self.num_workers,
            'torch_sharing_strategy': self.torch_sharing_strategy,
            'torch_start_method': self.torch_start_method,
        })

        if stage == 'fit' or stage is None:
            train_data = self.train_kwcoco
            if isinstance(train_data, pathlib.Path):
                train_data = os.fspath(train_data.expanduser())

            if self.verbose:
                print('Build train kwcoco dataset')
            train_coco_dset = watch.demo.coerce_kwcoco(train_data)
            self.coco_datasets['train'] = train_coco_dset

            print('self.exclude_sensors', self.exclude_sensors)
            coco_train_sampler = ndsampler.CocoSampler(train_coco_dset)
            train_dataset = KWCocoVideoDataset(
                coco_train_sampler, mode='fit', **self.train_dataset_config,
            )

            # Unfortunately lightning seems to only enable / disables
            # validation depending on the methods that are defined, so we are
            # not able to statically define them.
            self.classes = train_dataset.classes
            self.torch_datasets['train'] = train_dataset
            ub.inject_method(self, lambda self: self._make_dataloader('train', shuffle=True), 'train_dataloader')

            if self.input_channels is None:
                self.input_channels = train_dataset.input_channels

            if self.input_sensorchan is None:
                self.input_sensorchan = train_dataset.input_sensorchan

            stats_params = {
                'num': None,
                'with_intensity': False,
                'with_class': True,
                'num_workers': self.num_workers,
                'batch_size': self.batch_size,
            }
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
                self.dataset_stats = train_dataset.cached_dataset_stats(**stats_params)

            if self.vali_kwcoco is not None:
                # Explicit validation dataset should be prefered
                vali_data = self.vali_kwcoco
                if isinstance(vali_data, pathlib.Path):
                    vali_data = os.fspath(vali_data.expanduser())
                if self.verbose:
                    print('Build validation kwcoco dataset')
                kwcoco_ds = watch.demo.coerce_kwcoco(vali_data)
                vali_coco_sampler = ndsampler.CocoSampler(kwcoco_ds)
                vali_dataset = KWCocoVideoDataset(
                    vali_coco_sampler, mode='vali', **self.vali_dataset_config)
                self.torch_datasets['vali'] = vali_dataset
                ub.inject_method(self, lambda self: self._make_dataloader('vali', shuffle=False), 'val_dataloader')

        if stage == 'test' or stage is None:
            test_data = self.test_kwcoco
            if isinstance(test_data, pathlib.Path):
                test_data = os.fspath(test_data.expanduser())
            if self.verbose:
                print('Build test kwcoco dataset')
            test_coco_dset = watch.demo.coerce_kwcoco(test_data)
            test_coco_sampler = ndsampler.CocoSampler(test_coco_dset)
            self.coco_datasets['test'] = test_coco_dset
            self.torch_datasets['test'] = KWCocoVideoDataset(
                test_coco_sampler, mode='test', **self.test_dataset_config,
            )
            ub.inject_method(self, lambda self: self._make_dataloader('test', shuffle=False), 'test_dataloader')

        print('self.torch_datasets = {}'.format(ub.repr2(self.torch_datasets, nl=1)))

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
        # import nonechucks
        # nonechucks.SafeDataset
        return data.DataLoader(
            self.torch_datasets[stage],
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            collate_fn=ub.identity,  # disable collation
            shuffle=shuffle,
            pin_memory=True,
        )

    def _notify_about_tasks(self, requested_tasks=None, model=None):
        """
        Hacky method. Given the multimodal model, tell all the datasets which
        tasks they will need to generate data for. (This helps make the
        visualizations cleaner).
        """
        if model is not None:
            assert requested_tasks is None
            requested_tasks = {k: w > 0 for k, w in model.global_head_weights.items()}
        assert requested_tasks is not None
        self.requested_tasks = requested_tasks
        for dataset in self.torch_datasets.values():
            dataset._notify_about_tasks(requested_tasks)

    @classmethod
    def add_argparse_args(cls, parent_parser):
        """
        Previously the arguments were in multiple places including here.  This
        has been updated to use the :class:`KWCocoVideoDataModuleConfig` as the
        single point where arguments are defined. The functionality of this
        method is roughly the same as it used to be given that scriptconfig
        objects can be transformed into argparse objects.

        Example:
            >>> from watch.tasks.fusion.datamodules.kwcoco_video_data import *  # NOQA
            >>> cls = KWCocoVideoDataModule
            >>> # TODO: make use of watch.utils.lightning_ext import argparse_ext
            >>> import argparse
            >>> parent_parser = argparse.ArgumentParser()
            >>> cls.add_argparse_args(parent_parser)
            >>> parent_parser.print_help()
            >>> args, _ = parent_parser.parse_known_args(['--diff_inputs=True'])
            >>> assert args.diff_inputs
            >>> args, _ = parent_parser.parse_known_args(['--diff_inputs=False'])
            >>> assert not args.diff_inputs
            >>> args, _ = parent_parser.parse_known_args(['--exclude_sensors=l8,f3'])
            >>> assert args.exclude_sensors == 'l8,f3'
            >>> args, _ = parent_parser.parse_known_args(['--exclude_sensors=l8'])
            >>> assert args.exclude_sensors == 'l8'
        """
        # from functools import partial
        parser = parent_parser.add_argument_group('kwcoco_video_data')
        config = KWCocoVideoDataModuleConfig(cmdline=0)
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
        valid_argnames = explicit_argnames + list(KWCocoVideoDataModuleConfig.default.keys())
        datamodule_vars = ub.dict_isect(cfgdict, valid_argnames)
        return datamodule_vars

    def draw_batch(self, batch, stage='train', outputs=None, max_items=2,
                   overlay_on_image=False, **kwargs):
        """
        Visualize a batch produced by this DataSet.

        Args:
            batch (List[Dict]): uncollated list of Dataset Items

            outputs (Dict[str, Tensor]):
                maybe-collated list of network outputs?

        Example:
            >>> from watch.tasks.fusion.datamodules.kwcoco_video_data import *  # NOQA
            >>> from watch.tasks.fusion import datamodules
            >>> self = datamodules.KWCocoVideoDataModule(
            >>>     train_dataset='special:vidshapes8-multispectral', num_workers=0)
            >>> self.setup('fit')
            >>> loader = self.train_dataloader()
            >>> batch = next(iter(loader))
            >>> item = batch[0]
            >>> # Visualize
            >>> B = len(batch)
            >>> C, H, W = ub.peek(item['frames'][0]['modes'].values()).shape
            >>> T = len(item['frames'])
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
            >>> from watch.tasks.fusion.datamodules.kwcoco_video_data import *  # NOQA
            >>> from watch.tasks.fusion import datamodules
            >>> import watch
            >>> train_dataset = watch.demo.demo_kwcoco_multisensor()
            >>> self = datamodules.KWCocoVideoDataModule(
            >>>     train_dataset=train_dataset, chip_size=256, time_steps=5, num_workers=0, batch_size=3, true_multimodal=True)
            >>> self.setup('fit')
            >>> loader = self.train_dataloader()
            >>> batch_iter = iter(loader)
            >>> batch = next(batch_iter)
            >>> item = batch[0]
            >>> # Visualize
            >>> B = len(batch)
            >>> outputs = {'change_probs': [], 'class_probs': [], 'saliency_probs': []}
            >>> # Add dummy outputs
            >>> for item in batch:
            >>>     [v.append([]) for v in outputs.values()]
            >>>     for frame_idx, frame in enumerate(item['frames']):
            >>>         H, W = frame['class_idxs'].shape
            >>>         if frame_idx > 0:
            >>>             outputs['change_probs'][-1].append(torch.rand(H, W))
            >>>         outputs['class_probs'][-1].append(torch.rand(H, W, 10))
            >>>         outputs['saliency_probs'][-1].append(torch.rand(H, W, 2))
            >>> stage = 'train'
            >>> canvas = self.draw_batch(batch, stage=stage, outputs=outputs, max_items=4)
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

        canvas_list = []
        for item_idx, item in zip(range(max_items), batch_items):
            # HACK: I'm not sure how general accepting outputs is
            # TODO: more generic handling of outputs.
            # Should be able to accept
            # - [ ] binary probability of change
            # - [ ] fine-grained probability of change
            # - [ ] per-frame semenatic segmentation
            item_output = {}
            if outputs is not None:
                # print('outputs = {!r}'.format(outputs))
                item_output = ub.AutoDict()
                # output_walker = ub.IndexableWalker(item_output)
                # input_walker = ub.IndexableWalker(outputs)
                # for p, v in input_walker:
                #     if isinstance(v, torch.Tensor):
                #         d = input_walker
                #         for k in p[:-1]:
                #             d = d[k]
                #         d[p[-1]] = v.data.cpu().numpy()
                # print('item_output = {!r}'.format(item_output))
                for head_key in ['change_probs', 'class_probs', 'saliency_probs']:
                    if head_key in outputs:
                        item_output[head_key] = [f.data.cpu().numpy() for f in outputs[head_key][item_idx]]

            part = dataset.draw_item(item, item_output=item_output, overlay_on_image=overlay_on_image, **kwargs)
            canvas_list.append(part)
        canvas = kwimage.stack_images_grid(
            canvas_list, axis=1, overlap=-12, bg_value=[64, 60, 60])

        with_legend = True
        if with_legend:
            label_to_color = {
                node: data['color']
                for node, data in dataset.classes.graph.nodes.items()}
            label_to_color = ub.sorted_keys(label_to_color)
            legend_img = utils._memo_legend(label_to_color)
            canvas = kwimage.stack_images([canvas, legend_img], axis=1)

        return canvas


class KWCocoVideoDataset(data.Dataset):
    """
    Accepted keyword arguments are specified in
    :class:`KWCocoVideoDatasetConfig`

    Example:
        >>> from watch.tasks.fusion.datamodules.kwcoco_video_data import *  # NOQA
        >>> import ndsampler
        >>> import kwcoco
        >>> coco_dset = kwcoco.CocoDataset.demo('vidshapes2-multispectral', num_frames=10)
        >>> sampler = ndsampler.CocoSampler(coco_dset)
        >>> channels = 'B10|B8a|B1|B8'
        >>> sample_shape = (3, 256, 256)
        >>> self = KWCocoVideoDataset(sampler, sample_shape=sample_shape, channels=channels, time_sampling='soft2+distribute', diff_inputs=0, match_histograms=0, temporal_dropout=0.5)
        >>> index = len(self) // 4
        >>> item = self[index]
        >>> canvas = self.draw_item(item, overlay_on_image=1)
        >>> # xdoctest: +REQUIRES(--show)
        >>> import kwplot
        >>> kwplot.autompl()
        >>> kwplot.imshow(canvas)
        >>> kwplot.show_if_requested()

    Ignore:
        import kwplot
        kwplot.autompl()
        import xdev
        sample_indices = list(range(len(self)))
        for index in xdev.InteractiveIter(sample_indices):
            item = self[index]
            canvas = self.draw_item(item)
            kwplot.imshow(canvas)
            xdev.InteractiveIter.draw()

    Example:
        >>> from watch.tasks.fusion.datamodules.kwcoco_video_data import *  # NOQA
        >>> import ndsampler
        >>> import kwcoco
        >>> import watch
        >>> coco_dset = watch.demo.coerce_kwcoco('watch')
        >>> print({c.get('sensor_coarse') for c in coco_dset.images().coco_images})
        >>> print({c.channels.spec for c in coco_dset.images().coco_images})
        >>> sampler = ndsampler.CocoSampler(coco_dset)
        >>> sample_shape = (2, 128, 128)
        >>> self = KWCocoVideoDataset(sampler, sample_shape=sample_shape,
        >>>                           channels=None, diff_inputs=False, true_multimodal=True)
        >>> index = 0
        >>> item = self[index]
        >>> canvas = self.draw_item(item)
        >>> # xdoctest: +REQUIRES(--show)
        >>> import kwplot
        >>> kwplot.autompl()
        >>> kwplot.imshow(canvas)
        >>> kwplot.show_if_requested()

    Example:
        >>> # xdoctest: +REQUIRES(env:DVC_DPATH)
        >>> # Run the following tests on real watch data if DVC is available
        >>> from watch.tasks.fusion.datamodules.kwcoco_video_data import *  # NOQA
        >>> import os
        >>> from os.path import join
        >>> import ndsampler
        >>> import kwcoco
        >>> import watch
        >>> dvc_dpath = watch.find_smart_dvc_dpath()
        >>> coco_fpath = dvc_dpath / 'drop1-S2-L8-aligned/data.kwcoco.json'
        >>> coco_dset = kwcoco.CocoDataset(coco_fpath)
        >>> sampler = ndsampler.CocoSampler(coco_dset)
        >>> sample_shape = (7, 128, 128)
        >>> self = KWCocoVideoDataset(sampler, sample_shape=sample_shape, channels='red|green|blue|swir16|swir22|nir', match_histograms=0)
        >>> item = self[4]
        >>> canvas = self.draw_item(item)
        >>> # xdoctest: +REQUIRES(--show)
        >>> import kwplot
        >>> kwplot.autompl()
        >>> kwplot.imshow(canvas)
        >>> kwplot.show_if_requested()

    Ignore:
        >>> # xdoctest: +REQUIRES(env:DVC_DPATH)
        >>> # Debug issues seen in training
        >>> from watch.tasks.fusion.datamodules.kwcoco_video_data import *  # NOQA
        >>> import os
        >>> from os.path import join
        >>> import ndsampler
        >>> import kwcoco
        >>> import watch
        >>> dvc_dpath = watch.find_smart_dvc_dpath()
        >>> coco_fpath = dvc_dpath / 'Aligned-Drop3-TA1-2022-03-10/data_nowv_vali.kwcoco.json'
        >>> coco_dset = kwcoco.CocoDataset(coco_fpath)
        >>> sampler = ndsampler.CocoSampler(coco_dset)
        >>> self = KWCocoVideoDataset(
        >>>     sampler,
        >>>     sample_shape=(5, 128, 128),
        >>>     window_overlap=0,
        >>>     channels="blue|green|red|nir|swir16",
        >>>     neg_to_pos_ratio=0, time_sampling='auto', diff_inputs=0, mode='fit', match_histograms=0,
        >>> )
        >>> item = self[0]
        >>> canvas = self.draw_item(item)
        >>> print(ub.repr2(item['tr'], nl=-1))
        >>> # xdoctest: +REQUIRES(--show)
        >>> import kwplot
        >>> kwplot.autompl()
        >>> canvas = self.draw_item(
        >>>     item, max_channels=7, overlay_on_image=False)
        >>> kwplot.imshow(canvas)
        >>> import xdev
        >>> sample_indices = list(range(len(self)))
        >>> for index in xdev.InteractiveIter(sample_indices):
        >>>     item = self[index]
        >>>     canvas = self.draw_item(item, max_channels=7,
        >>>                             overlay_on_image=False)
        >>>     kwplot.imshow(canvas)
        >>>     xdev.InteractiveIter.draw()
    """
    # TODO: add torchvision.transforms or albumentations

    def __init__(self, sampler, sample_shape=None, window_overlap=None, mode='fit', **kwargs):
        config = KWCocoVideoDatasetConfig(cmdline=0, data=kwargs)
        BACKWARDS_COMPATIBILITY = True
        if BACKWARDS_COMPATIBILITY:
            if window_overlap is not None:
                config['chip_overlap'] = window_overlap
            if sample_shape is not None:
                config['time_steps'] = sample_shape[0]
                config['chip_dims'] = sample_shape[1:3]

        chip_dims = config['chip_dims']
        chip_h, chip_w = chip_dims
        window_dims = (config['time_steps'], chip_h, chip_w)
        window_overlap = config['chip_overlap']

        self.window_dims = window_dims
        self.config = config
        self.__dict__.update(self.config.to_dict())
        self.sampler = sampler
        # TODO: the set of "valid" background classnames should be defined
        # by the inputs, not hard-coded in the dataloader. This can either be a
        # list of names provided to the training config, or something baked
        # into the kwcoco spec marking a class as some type of "background"
        # if not self.use_conditional_classes:
        #     # TODO: CONDITIONAL
        #     raise NotImplementedError

        # Add extra categories if we need to and construct a new classes object
        graph = self.sampler.classes.graph
        if 0:
            import networkx as nx
            print(nx.forest_str(graph, with_labels=True))

        # Update with heuristics
        # HACK: Overwrite kwcoco data
        for _catinfo in heuristics.CATEGORIES:
            name = _catinfo['name']
            exists_flag = name in graph.nodes
            if not exists_flag and _catinfo.get('required'):
                graph.add_node(name, **_catinfo)
            if exists_flag:
                graph.nodes[name].update(**_catinfo)

        self.background_classes = set(heuristics.BACKGROUND_CLASSES) & set(graph.nodes)
        self.negative_classes = set(heuristics.NEGATIVE_CLASSES) & set(graph.nodes)
        self.ignore_classes = set(heuristics.IGNORE_CLASSNAMES) & set(graph.nodes)
        self.undistinguished_classes = set(heuristics.UNDISTINGUISHED_CLASSES) & set(graph.nodes)
        self.classes = kwcoco.CategoryTree(graph)

        channels = config['channels']
        time_sampling = config['time_sampling']
        exclude_sensors = config['exclude_sensors']
        use_centered_positives = config['use_centered_positives']
        use_grid_positives = config['use_grid_positives']
        set_cover_algo = config['set_cover_algo']
        time_span = config['time_span']
        neg_to_pos_ratio = config['neg_to_pos_ratio']
        max_epoch_length = config['max_epoch_length']

        if time_sampling == 'auto':
            time_sampling = 'hard+distribute'

        if mode == 'custom':
            new_sample_grid = None
            self.length = 1
        elif mode == 'test':
            # In test mode we have to sample everything for BAS
            # (TODO: for activity clf, we should only focus on candidate regions)
            new_sample_grid = sample_video_spacetime_targets(
                sampler.dset, window_dims=window_dims,
                window_overlap=window_overlap,
                keepbound=True,
                use_annot_info=False,
                exclude_sensors=exclude_sensors,
                time_sampling=time_sampling,
                time_span=time_span,
                set_cover_algo=set_cover_algo,
            )
            self.length = len(new_sample_grid['targets'])
        else:
            negative_classes = (
                self.ignore_classes | self.background_classes | self.negative_classes)
            new_sample_grid = sample_video_spacetime_targets(
                sampler.dset, window_dims=window_dims,
                window_overlap=window_overlap,
                negative_classes=negative_classes,
                keepbound=False,
                use_annot_info=True,
                exclude_sensors=exclude_sensors,
                time_sampling=time_sampling,
                time_span=time_span,
                use_centered_positives=use_centered_positives,
                use_grid_positives=use_grid_positives,
                set_cover_algo=set_cover_algo,
            )

            n_pos = len(new_sample_grid['positives_indexes'])
            n_neg = len(new_sample_grid['negatives_indexes'])

            # max_neg = min(int(max(0, (neg_to_pos_ratio * n_pos))), n_neg)
            # if n_neg > max_neg:
            #     print('restrict to max_neg = {!r}'.format(max_neg))

            target_vidids = [v['video_id'] for v in new_sample_grid['targets']]

            # Hack: determine if videos should be grouped together

            target_posbit = kwarray.boolmask(
                new_sample_grid['positives_indexes'],
                len(new_sample_grid['targets']))

            # import kwarray
            # rng = kwarray.ensure_rng(None)

            if 1:
                vidnames = self.sampler.dset.videos(target_vidids).lookup('name')
                df = pd.DataFrame({
                    'vidid': target_vidids,
                    'vidname': vidnames,
                    'is_positive': target_posbit,
                }).reset_index(drop=False)

                # Hack, because we didn't encode the region in the cropped site
                # (rookie move)
                import watch
                pat = watch.utils.util_pattern.Pattern.coerce(r'\w+_R\d+_\d+', 'regex')
                vidname_to_region_name = {}
                for vidname in set(vidnames):
                    if pat.match(vidname):
                        vidname_to_region_name[vidname] = vidname.rsplit('_', 1)[0]

                self.vidname_to_region_name = vidname_to_region_name

                if vidname_to_region_name:
                    df['region'] = df['vidname'].apply(vidname_to_region_name.__getitem__)
                else:
                    df['region'] = df['vidname']

                key_to_group = dict(list(df.groupby(['region', 'is_positive'])))
                vidname_to_pool = {}
                for key, group in key_to_group.items():
                    vidname, flag = key
                    if flag:
                        pos_vid_idxs = group['index']
                        other_key = (vidname, False)
                        if other_key in key_to_group:
                            other = key_to_group[other_key]
                            neg_vid_idxs = other['index']
                        else:
                            neg_vid_idxs = []
                            other = []
                        n_pos = len(group)
                        n_neg = len(other)
                        max_neg = min(int(max(1, (neg_to_pos_ratio * n_pos))), n_neg)
                        print(f'restrict to {max_neg=} in {vidname=} with {n_pos=} and {n_neg=}')
                        # neg_vid_idxs = posneg_groups[False]['index'].values
                        neg_vid_pool_ = list(util_iter.chunks(neg_vid_idxs, nchunks=max_neg))
                        pos_vid_pool_ = list(util_iter.chunks(pos_vid_idxs, nchunks=n_pos))
                        vid_pool = pos_vid_pool_ + neg_vid_pool_
                        vidname_to_pool[vidname] = [p for p in vid_pool if p]

                freqs = list(map(len, vidname_to_pool.values()))
                if len(freqs) == 0:
                    max_per_vid = 100
                    import warnings
                    warnings.warn('Warning: no video pool')
                else:
                    max_per_vid = int(np.median(freqs))
                all_chunks = []
                for vidname, vid_pool in vidname_to_pool.items():
                    # print(len(vid_pool[0]))
                    # print(len(vid_pool[-1]))
                    rechunked_video_pool = list(util_iter.chunks(vid_pool, nchunks=max_per_vid))
                    all_chunks.extend(rechunked_video_pool)

                self.nested_pool = NestedPool(all_chunks)

            if 0:
                import netharn as nh
                nh.data.collate._debug_inbatch_shapes(all_chunks)

            self.length = len(self.nested_pool)

            if max_epoch_length is not None:
                self.length = min(self.length, max_epoch_length)

        self.new_sample_grid = new_sample_grid

        bg_catname = ub.peek(sorted(self.background_classes))
        self.bg_idx = self.classes.node_to_idx[bg_catname]

        # bg_node = graph.nodes['background']
        # if 'color' not in bg_node:
        #     bg_node['color'] = (0., 0., 0.)
        utils.category_tree_ensure_color(self.classes)

        self.special_inputs = {}

        if channels is None:
            # Hack to use all channels in the first image.
            # (Does not handle heterogeneous channels yet)
            # chan_info = kwcoco_extensions.coco_channel_stats(sampler.dset)
            sensorchan_hist = kwcoco_extensions.coco_channel_stats(sampler.dset)['sensorchan_hist']
            sensorchans = ','.join(sorted([f'{sensor}:{chans}' for sensor, chan_hist in sensorchan_hist.items() for chans in chan_hist.keys()]))
            sensorchans = kwcoco.SensorChanSpec.coerce(sensorchans)
            # channels = ','.join(sorted(chan_info['chan_hist']))
            if len(sensorchan_hist) > 0:
                import warnings
                warnings.warn('Channels are unspecified, but the dataset has a complex set of channels with multiple sensors. Being explicit would be better')
            # channels = chan_info['all_channels']
        else:
            # hack
            sensorchan_hist = None
            sensorchans = channels
            # channels = ','.join(sorted([s.chans.spec for s in self.sensorchan.streams()]))

        self.sensorchan = kwcoco.SensorChanSpec.coerce(sensorchans).normalize()

        # handle generic * sensors, the idea is that we find matches
        # in the dataset that can support the requested channels.
        if '*' in [s.sensor.spec for s in self.sensorchan.streams()]:
            # handle * sensor in a way that works with previous models
            # This code is a little messy and should be cleaned up
            if sensorchan_hist is None:
                sensorchan_hist = kwcoco_extensions.coco_channel_stats(sampler.dset)['sensorchan_hist']

            expanded_input_sensorchan_streams = []
            for fused_sensorchan in self.sensorchan.streams():
                sensor = fused_sensorchan.sensor
                chans = fused_sensorchan.chans
                if sensor.spec == '*':
                    for cand_sensor, cand_chans in sensorchan_hist.items():
                        valid_chan_cands = []
                        for cand_chan_group in cand_chans:
                            cand_chan_group = kwcoco.FusedChannelSpec.coerce(cand_chan_group)
                            chan_isect = chans & cand_chan_group
                            if chan_isect.spec == chans.spec:
                                valid_chan_cands.append(valid_chan_cands)
                                expanded_input_sensorchan_streams.append(cand_sensor + ':' + chans.spec)
                                break
                else:
                    expanded_input_sensorchan_streams.append('{}:{}'.format(sensor, chans))

            if not expanded_input_sensorchan_streams:
                print('sensorchan_hist = {}'.format(ub.repr2(sensorchan_hist, nl=1)))
                raise ValueError('The generic sensor * was given, but no data in the kwcoco file matched')

            self.sensorchan = kwcoco.SensorChanSpec.coerce(','.join(list(ub.unique(expanded_input_sensorchan_streams)))).normalize()

        # Hack away sensors
        print(f'self.sensorchan={self.sensorchan=!r}')
        channels = ','.join(sorted(ub.unique([s.chans.spec for s in self.sensorchan.streams()])))
        channels = channel_spec.ChannelSpec.coerce(channels).normalize()
        self.channels = channels

        NEW = 1
        if NEW:
            # TODO: Clean up this code.
            _input_channels = []
            _sample_channels = []
            _input_sensorchans = []
            _sample_sensorchans = []
            for fused_sensorchan in self.sensorchan.streams():
                sensor = fused_sensorchan.sensor
                chans = fused_sensorchan.chans
                _stream = chans.as_oset()
                _sample_stream = _stream.copy()
                special_bands = _stream & util_bands.SPECIALIZED_BANDS
                if special_bands:
                    raise NotImplementedError('This is broken ATM')
                    # TODO: introspect which extra bands are needed for to compute
                    # the sample, but hard code for now
                    _sample_stream -= special_bands
                    _sample_stream = _sample_stream | ub.oset('blue|green|red|nir|swir16|swir22'.split('|'))
                    self.special_inputs[key] = special_bands
                if self.diff_inputs:
                    raise NotImplementedError('This is broken ATM')
                    _stream = [s + p for p in _stream for s in ['', 'D']]
                _input_sensorchans.append(sensor.spec + ':' + '|'.join(_stream))
                _sample_sensorchans.append(sensor.spec + ':' + '|'.join(_sample_stream))
                _input_channels.append('|'.join(_stream))
                _sample_channels.append('|'.join(_sample_stream))

                #### New: input_sensorchan will replace input_channels
                self.sample_sensorchan = kwcoco.SensorChanSpec(
                    ','.join(_sample_sensorchans)
                )

                self.input_sensorchan = kwcoco.SensorChanSpec.coerce(
                    ','.join(_input_sensorchans)
                )
        else:
            _input_channels = []
            _sample_channels = []
            for key, stream in channels.parse().items():
                _stream = stream.as_oset()
                _sample_stream = _stream.copy()
                special_bands = _stream & util_bands.SPECIALIZED_BANDS
                if special_bands:
                    # TODO: introspect which extra bands are needed for to compute
                    # the sample, but hard code for now
                    _sample_stream -= special_bands
                    _sample_stream = _sample_stream | ub.oset('blue|green|red|nir|swir16|swir22'.split('|'))
                    self.special_inputs[key] = special_bands
                if self.diff_inputs:
                    _stream = [s + p for p in _stream for s in ['', 'D']]
                _input_channels.append('|'.join(_stream))
                _sample_channels.append('|'.join(_sample_stream))

        # DEPRECATE channel only stuff. Use sensorchan everywhere
        # * sample_channels
        # * input_channels

        # Some of the channels are computed on the fly.
        # This is the list of ones that are loaded from disk.
        self.sample_channels = kwcoco.channel_spec.ChannelSpec(
            ','.join(_sample_channels)
        )
        self.input_channels = kwcoco.channel_spec.ChannelSpec.coerce(
            ','.join(_input_channels)
        )

        # TODO:
        # We need to know all of the combinations of channels each data item
        # could produce

        self.mode = mode

        self.true_multimodal = config['true_multimodal']
        self.disable_augmenter = False

        # hidden option for now (todo: expose this)
        self.inference_only = False
        self.with_change = True
        self.requested_tasks = {
            'change': True,
            'class': True,
            'saliency': True,
        }

        # Hacks: combinable channels can be visualized as RGB images.
        # The only reason this is a hack is because of the hardcoded names
        # otherwise it is a cool feature.
        self.default_combinable_channels = [
            ub.oset(['red', 'green', 'blue']),
            ub.oset(['Dred', 'Dgreen', 'Dblue']),
            ub.oset(['r', 'g', 'b']),
        ] + heuristics.HUERISTIC_COMBINABLE_CHANNELS

    def __len__(self):
        return self.length

    def _notify_about_tasks(self, requested_tasks=None, model=None):
        """
        Hacky method. Given the multimodal model, tell all the datasets which
        tasks they will need to generate data for. (This helps make the
        visualizations cleaner).
        """
        if model is not None:
            assert requested_tasks is None
            requested_tasks = {k: w > 0 for k, w in model.global_head_weights.items()}
        assert requested_tasks is not None
        self.requested_tasks = requested_tasks

    # def _make_augmenter():
    #     # TODO: how to make this work with kwimage polygons?
    #     import albumentations as A

    #     tf = A.HorizontalFlip(p=0.5)

    #     # Declare an augmentation pipeline
    #     transform = A.Compose([
    #         # A.RandomCrop(width=256, height=256),
    #         A.HorizontalFlip(p=0.5),
    #         # A.RandomBrightnessContrast(p=0.2),
    #     ], keypoint_params=A.KeypointParams(format='xy'))

    #     transform(image=np.random.rand(10, 10), keypoints=[[2, 2]])
    #     transform(image=np.random.rand(10, 10), keypoints=[[2, 2]])
    #     transform( keypoints=[[2, 2]])

    def check_nested_pool(self, num=4096):
        """
        Example:
            >>> # xdoctest: +REQUIRES(env:DVC_DPATH)
            >>> from watch.tasks.fusion.datamodules.kwcoco_video_data import *  # NOQA
            >>> import watch
            >>> import ndsampler
            >>> import kwcoco
            >>> dvc_dpath = watch.find_smart_dvc_dpath()
            >>> coco_fpath = dvc_dpath / 'Cropped-Drop3-TA1-2022-03-10/data_nowv_train.kwcoco.json'
            >>> coco_dset = kwcoco.CocoDataset(coco_fpath)
            >>> sampler = ndsampler.CocoSampler(coco_dset)
            >>> self = KWCocoVideoDataset(
            >>>     sampler,
            >>>     sample_shape=(11, 256, 256),
            >>>     window_overlap=0,
            >>>     #channels="ASI|MF_Norm|AF|EVI|red|green|blue|swir16|swir22|nir",
            >>>     channels="blue|green|red|nir|swir16|swir22",
            >>>     neg_to_pos_ratio=0, time_sampling='soft2', diff_inputs=0, temporal_dropout=0.5,
            >>> )
            >>> #self.requested_tasks['change'] = False

            if 0:
                infos = []
                for num in [500, 1000, 2500, 5000, 7500, 10000, 20000]:
                    row = self.check_nested_pool(num=num)
                    infos.append(row)
                df = pd.DataFrame(infos)
                import kwplot
                sns = kwplot.autosns()

                data = df.melt(id_vars=['num'])
                data['style'] = 'raw'
                data.loc[data.variable.apply(lambda x: 'gids' in x), 'style'] = 'gids'
                data.loc[data.variable.apply(lambda x: 'region' in x), 'style'] = 'region'
                data['region'] = data.variable.apply(lambda x: x.split('_', 2)[-1].replace('seen', '') if 'R' in x else x)
                sns.lineplot(data=data, x='num', y='value', style='style', hue='region')

                    frac_seen = info['frac_gids_seen']
                    frac_seen['num'] = num
                    frac_seen['ideal_seen'] = ideal_seen
                    frac_seen['ideal_frac'] = ideal_frac
        """
        # Check the nested pool
        dset = self.sampler.dset
        vidid_to_name = dset.videos().lookup('name', keepid=True)
        idx_hist = ub.dict_hist(self.nested_pool.sample() for _ in range(num))
        targets = self.new_sample_grid['targets']

        gid_freq = ub.ddict(lambda: 0)
        vidid_freq = ub.ddict(lambda: 0)
        region_seen_gids = ub.ddict(set)
        for idx, freq in ub.ProgIter(list(idx_hist.items())):
            tr = targets[idx]
            gids = tr['gids']
            for gid in gids:
                # frame_index = dset.index.imgs[gid]['frame_index']
                gid_freq[gid] += freq
            vidid = tr['video_id']
            vidname = vidid_to_name[vidid]
            region = self.vidname_to_region_name[vidname]
            vidid_freq[vidid] += freq
            region_seen_gids[region].update(gids)

        vidname_to_freq = ub.map_keys(vidid_to_name, vidid_freq)

        # TODO: these should be some concept of video groups
        region_freq = ub.ddict(lambda: 0)
        for vidname, freq in vidname_to_freq.items():
            region_name = self.vidname_to_region_name[vidname]
            region_freq[region_name] += freq

        _region_total_gids = ub.ddict(lambda: 0)
        for vidid, gids in dset.index.vidid_to_gids.items():
            vidname = vidid_to_name[vidid]
            region_name = self.vidname_to_region_name[vidname]
            _region_total_gids[region_name] += len(gids)
        region_total_num_gids = pd.Series(_region_total_gids)
        region_seen_num_gids = pd.Series(ub.map_vals(len, region_seen_gids))

        frac_gids_seen = region_seen_num_gids / region_total_num_gids

        _count = pd.Series(region_freq)
        _prob = _count / _count.sum()
        seen_gids = set(gid_freq.keys())
        total_gids = set(dset.images())
        num_seen = len(seen_gids)
        num_total = len(total_gids)
        ideal_seen = (num * len(tr['gids']))
        seen_frac = num_seen / num_total
        ideal_frac = min(ideal_seen / num_total, 1.0)

        row = frac_gids_seen.add_prefix('frac_gids_seen')
        row = pd.concat([row, _prob.add_prefix('region_freq_')])

        row['seen_frac'] = seen_frac
        row['ideal_frac'] = ideal_frac
        row['num'] = num
        return row

        if 0:
            rows = []
            for idx, freq in ub.ProgIter(list(idx_hist.items())):
                tr = targets[idx]
                for gid in tr['gids']:
                    vidid = tr['video_id']
                    vidname = vidid_to_name[vidid]
                    region = self.vidname_to_region_name[vidname]
                    frame_index = dset.index.imgs[gid]['frame_index']
                    rows.append({
                        'idx': idx,
                        'gid': gid,
                        'vidid': vidid,
                        'frame_index': frame_index,
                        'vidname': vidname,
                        'region': region,
                        'freq': freq,
                    })
            df = pd.DataFrame(rows)
            region_freq = df.groupby('region')['freq'].sum()
            region_freq = region_freq / region_freq.sum()
            _freq = df.groupby('video_id')['freq'].sum()
            _freq = _freq / _freq.sum()
            _freq = df.groupby(['video_id', 'frame_index'])['freq'].sum()
            _freq = _freq / _freq.sum()

        # vidid_to_name = dset.videos(list(vidid_freq.keys())).lookup('name', keepid=True)

    def _expand_targets_time(self, n_time_expands):
        """
        Increase the number of test-time targets by expanding them in time.
        """
        sample_grid = self.new_sample_grid
        expanded_targets = []
        assert not sample_grid['positives_indexes'], 'unhandled'
        assert not sample_grid['negatives_indexes'], 'unhandled'
        targets = sample_grid['targets']
        for tr in targets:
            seen_ = set()
            # Add the original sample
            expanded_targets.append(tr)
            seen_.add(tuple(tr['gids']))
            # Add the expanded samples
            for _ in range(n_time_expands):
                tr_ = tr.copy()
                tr_ = self._augment_target_time(tr_)
                new_gids = tuple(tr_['gids'])
                if new_gids not in seen_:
                    expanded_targets.append(tr_)
                    seen_.add(tuple(tr_['gids']))
        print(f'Temporal augmentation expanded {len(targets)=} '
              f'to {len(expanded_targets)=}')
        sample_grid['targets'] = expanded_targets
        self.length = len(expanded_targets)

    def _expand_targets_fliprot(self, n_fliprot):
        """
        Increase the number of test-time targets via flips
        """
        sample_grid = self.new_sample_grid
        expanded_targets = []
        assert not sample_grid['positives_indexes'], 'unhandled'
        assert not sample_grid['negatives_indexes'], 'unhandled'
        targets = sample_grid['targets']
        unique_fliprots = [
            {'rot_k': 0, 'flip_axis': None},
            {'rot_k': 0, 'flip_axis': (0,)},
            {'rot_k': 1, 'flip_axis': None},
            {'rot_k': 1, 'flip_axis': (0,)},
            {'rot_k': 2, 'flip_axis': None},
            {'rot_k': 2, 'flip_axis': (0,)},
            {'rot_k': 3, 'flip_axis': None},
            {'rot_k': 3, 'flip_axis': (0,)},
        ]
        for tr in targets:
            # Add the original sample
            expanded_targets.append(tr)
            # Add the expanded samples
            assert n_fliprot <= 7
            for idx in range(1, n_fliprot + 1):
                tr_ = tr.copy()
                tr_['fliprot_params'] = unique_fliprots[idx]
                expanded_targets.append(tr_)

        print(f'Fliprot augmentation expanded {len(targets)=} '
              f'to {len(expanded_targets)=}')

        sample_grid['targets'] = expanded_targets
        self.length = len(expanded_targets)

    def _augment_target_time(self, tr_):
        """
        Jitters the time sample in a target
        """
        vidid = tr_['video_id']
        time_sampler = self.new_sample_grid['vidid_to_time_sampler'][vidid]
        valid_gids = self.new_sample_grid['vidid_to_valid_gids'][vidid]
        new_idxs = time_sampler.sample(tr_['main_idx'])
        new_gids = list(ub.take(valid_gids, new_idxs))
        tr_['gids'] = new_gids
        return tr_

    def _augment_spacetime_target(self, tr_):
        """
        Given a target dictionary, shift around the space and time slice

        Ignore:
            >>> from watch.tasks.fusion.datamodules.kwcoco_video_data import *  # NOQA
            >>> import ndsampler
            >>> import kwcoco
            >>> import watch
            >>> coco_dset = watch.demo.coerce_kwcoco('watch')
            >>> sampler = ndsampler.CocoSampler(coco_dset)
            >>> sample_shape = (2, 128, 128)
            >>> self = KWCocoVideoDataset(sampler, sample_shape=sample_shape)
            >>> index = 0
            >>> tr = self.new_sample_grid['targets'][index]
            >>> tr_ = tr.copy()
            >>> tr_ = self._augment_spacetime_target(tr_)
            >>> print('tr  = {!r}'.format(tr))
            >>> print('tr_ = {!r}'.format(tr_))
        """

        # TODO: make a nice "augmenter" pipeline
        # TODO: parameteraize
        temporal_augment_rate = 0.8
        spatial_augment_rate = 0.9

        do_shift = False
        if not self.disable_augmenter and self.mode == 'fit':
            # do_shift = np.random.rand() > 0.5
            do_shift = True
        if do_shift:
            # Spatial augmentation
            rng = kwarray.ensure_rng(None)

            vidid = tr_['video_id']
            video = self.sampler.dset.index.videos[vidid]
            vid_width = video['width']
            vid_height = video['height']

            # Spatial augmentation:
            if rng.rand() < spatial_augment_rate:
                # space_box = kwimage.Boxes.from_slice(tr_['space_slice'], endpoint=False)
                y_sl, x_sl = tr_['space_slice']
                space_box = kwimage.Boxes([
                    [x_sl.start, y_sl.start, x_sl.stop - 1, y_sl.stop - 1],
                ], 'ltrb')
                w = space_box.width.ravel()[0]
                h = space_box.height.ravel()[0]
                # hack: this prevents us from assuming there is a target in the
                # window, but it lets us get the benefit of chip_overlap=0.5 while
                # still having it at 0 for faster epochs.
                aff = kwimage.Affine.coerce(offset=(
                    rng.randint(-w // 2.7, w // 2.7),
                    rng.randint(-h // 2.7, h // 2.7)))
                space_box = space_box.warp(aff).quantize()

                # prevent shifting the target off the edge of the video
                snap_target = kwimage.Boxes([[0, 0, vid_width, vid_height]], 'ltrb')
                space_box = _boxes_snap_to_edges(space_box, snap_target)

                tr_['space_slice'] = space_box.astype(int).to_slices()[0]

            # Temporal augmentation
            if rng.rand() < temporal_augment_rate:
                self._augment_target_time(tr_)

            temporal_dropout_rate = self.temporal_dropout
            do_temporal_dropout = rng.rand() < temporal_dropout_rate
            if do_temporal_dropout:
                # Temporal dropout
                gids = tr_['gids']
                main_gid = tr_['main_gid']
                main_frame_idx = gids.index(main_gid)
                flags = rng.rand(len(gids)) > 0.5
                flags[main_frame_idx] = True
                flags[0] = True
                flags[-1] = True
                gids = list(ub.compress(gids, flags))
                # tr_['main_idx'] = gids.index(main_gid)
                tr_['gids'] = gids

        # force_flip = tr_.get('flip_axis', None)
        unique_fliprots = [
            {'rot_k': 0, 'flip_axis': None},
            {'rot_k': 1, 'flip_axis': None},
            {'rot_k': 2, 'flip_axis': None},
            {'rot_k': 3, 'flip_axis': None},
            {'rot_k': 0, 'flip_axis': (0,)},
            {'rot_k': 1, 'flip_axis': (0,)},
            {'rot_k': 2, 'flip_axis': (0,)},
            {'rot_k': 3, 'flip_axis': (0,)},
        ]

        # Force an augmentation
        FLIP_AUGMENTATION = (not self.disable_augmenter and self.mode == 'fit')
        if FLIP_AUGMENTATION:
            # Choose a unique flip/rot
            fliprot_idx = rng.randint(0, len(unique_fliprots))
            fliprot_params = unique_fliprots[fliprot_idx]
            tr_['fliprot_params'] = fliprot_params

        return tr_

    @profile
    def _sample_one_frame(self, gid, sampler, coco_dset, tr_, with_annots,
                          gid_to_isbad, gid_to_sample):
        # helper that was previously a nested function moved out for profiling
        coco_img = coco_dset.coco_image(gid)

        sensor_coarse = coco_img.img.get('sensor_coarse', '*')

        matching_sensorchan = self.sample_sensorchan.matching_sensor(sensor_coarse)
        # if matching_sensorchan == 0:

        sensor_channels = matching_sensorchan.chans

        # Require
        # self.sample_sensorchan

        # sensor_channels = (self.sample_channels & coco_img.channels).normalize()
        tr_frame = tr_.copy()
        tr_frame['gids'] = [gid]
        sample_streams = {}
        first_with_annot = with_annots

        # TODO: Use the cloudmask here

        # Flag will be set to true if any heuristic on any channel stream
        # forces us to mark this image as bad.
        force_bad = False

        # TODO: separate ndsampler annotation loading function
        USE_CLOUDMASK = self.use_cloudmask
        # NOTES ON CLOUDMASK
        # https://github.com/GERSL/Fmask#46-version
        # The cloudmask band is a class-idx based raster with labels
        # 0 => clear land pixel
        # 1 => clear water pixel
        # 2 => cloud shadow
        # 3 => snow
        # 4 => cloud
        # 255 => no observation

        # However, in my data I seem to see:
        # Unique values   8,  16,  65, 128

        # These are specs
        # https://smartgitlab.com/TE/standards/-/wikis/Data-Output-Specifications#quality-band
        # TODO: this could be a specially handled frame like ASI.
        # Bits
        # 0 T&E binary mask
        # 1 Dilated Cloud
        # 2 Cirrus
        # 3 Cloud
        # 4 Cloud Shadow
        # 5 Snow
        # 6 Clear
        # 7 Water
        if USE_CLOUDMASK:
            if 'cloudmask' in coco_img.channels:
                tr_cloud = tr_frame.copy()
                tr_cloud['channels'] = 'cloudmask'
                # tr_cloud['channels'] = 'red|green|blue'
                tr_cloud['antialias'] = False
                tr_cloud['interpolation'] = 'nearest'
                tr_cloud['nodata'] = None
                cloud_sample = sampler.load_sample(
                    tr_cloud, with_annots=None,
                    padkw={'constant_values': 255},
                    dtype=np.float32
                )
                cloud_im = cloud_sample['im']
                cloud_bits = 1 << np.array([1, 2, 3])
                is_cloud_iffy = np.logical_or.reduce([cloud_im == b for b in cloud_bits])
                cloud_frac = is_cloud_iffy.mean()
                if cloud_frac > 0.5:
                    # print('cloud_frac = {!r}'.format(cloud_frac))
                    force_bad = 'too cloudy'
                    # valid_cloud_vals = cloud_im[np.isnan(cloud_im)]

                # if 0:
                #     obj = coco_img.find_asset_obj('cloudmask')
                #     fpath = ub.Path(coco_img.bundle_dpath) / obj['file_name']

                # Skip if more then 50% cloudy

        if sensor_channels.numel() == 0:
            force_bad = 'Missing requested channels'

        for stream in sensor_channels.streams():
            if force_bad:
                break
            tr_frame['channels'] = stream
            sample = sampler.load_sample(
                tr_frame, with_annots=first_with_annot,
                nodata='float',
                padkw={'constant_values': np.nan},
                dtype=np.float32
            )

            WV_NODATA_HACK = 1
            if WV_NODATA_HACK:
                # Should be fixed in drop3
                if coco_img.img.get('sensor_coarse') == 'WV':
                    if set(stream).issubset({'blue', 'green', 'red'}):
                        # Check to see if the nodata value is known in the
                        # image metadata
                        obj = coco_img.find_asset_obj('red')
                        band_metas = obj.get('band_metas', [{}])
                        nodata_vals = [m.get('nodata', None) for m in band_metas]
                        # TODO: could be more careful about what band metas
                        # we are looking at. Assuming they are all the same
                        # here. The idea is only do this hack if the nodata
                        # value is not set (like in L1 data, but dont do it
                        # when the) values are set (like in TA1 data)
                        if any(v is None for v in nodata_vals):
                            mask = (sample['im'] == 0)
                            sample['im'][mask] = np.nan

            invalid_mask = np.isnan(sample['im'])

            any_invalid = np.any(invalid_mask)
            none_invalid = not any_invalid
            if none_invalid:
                all_invalid = False
            else:
                all_invalid = np.all(invalid_mask)

            if any_invalid:
                sample['invalid_mask'] = invalid_mask
            else:
                sample['invalid_mask'] = None

            if not all_invalid:
                sample_streams[stream.spec] = sample
                if 'annots' in sample:
                    # dont ask for annotations multiple times
                    first_with_annot = False
            else:
                # HACK: if the red channel is all bad, discard the frame
                # This can be removed once nodata is correctly propogated
                # in the team features. OR we can add a feature where we
                # keep track of an image wide observation mask and use that
                # instead of using red as a proxy for it.
                if 'red' in set(stream):
                    force_bad = 'invalid red channel'
                    break

            # TODO: mark frame as invalid when a red band is all 0
            # We are going to try to generalize this with a concept of an
            # "iffy" mask with will flag pixels that are minimum, zero, or
            # nan.
            RGB_IFFY_HACK = 1
            if RGB_IFFY_HACK and set(stream).issubset({'blue', 'green', 'red'}):
                import warnings
                with warnings.catch_warnings():
                    warnings.filterwarnings('ignore', 'empty slice', RuntimeWarning)
                    warnings.filterwarnings('ignore', 'All-NaN', RuntimeWarning)
                    if any_invalid:
                        chan_mins = np.nanmin(sample['im'], axis=(0, 1, 2), keepdims=1)
                        invalid_mask
                        is_min_mask = (sample['im'] == chan_mins)
                        is_zero_mask = sample['im'] == 0
                        is_iffy_mask = (invalid_mask | is_min_mask | is_zero_mask)
                        chan_num_iffy = is_iffy_mask.sum(axis=(0, 1, 2))
                        chan_num_pxls = np.prod(is_iffy_mask.shape[0:3])
                        chan_frac_iffy = chan_num_iffy / chan_num_pxls
                        chan_is_bad = chan_frac_iffy > 0.4
                        if np.any(chan_is_bad):
                            force_bad = 'iffy RGB channel'
                            break

        if not force_bad:
            if len(sample_streams) == 0:
                force_bad = 'no-streams'

        gid_to_isbad[gid] = force_bad
        gid_to_sample[gid] = sample_streams

    def _input_grid_stats(self):
        targets = self.new_sample_grid['targets']

        freqs = ub.ddict(lambda: ub.ddict(lambda: 0))

        for tr in ub.ProgIter(targets, desc='loop over targets'):
            vidid = tr['video_id']
            freqs['vidid'][vidid] += 1
            gids = tr['gids']
            for gid in gids:
                freqs['gid'][gid] += 1
            freqs['label'][tr['label']] += 1

        dset = self.sampler.dset
        for gid, freq in freqs['gid'].items():
            sensor_coarse = dset.coco_image(gid).img.get('sensor_coarse', '*')
            sensor_coarse = dset.coco_image(gid).img.get('sensor_coarse', '*')
            freqs['sensor'][sensor_coarse] += 1

        print(ub.repr2(ub.dict_diff(freqs, {'gid'})))

    @profile
    def __getitem__(self, index):
        """

        CommandLine:
            DVC_DPATH=$(smartwatch_dvc --hardware="hdd") xdoctest -m watch.tasks.fusion.datamodules.kwcoco_video_data KWCocoVideoDataset.__getitem__:0 --profile

        Example:
            >>> # xdoctest: +REQUIRES(env:DVC_DPATH)
            >>> # Debug issues seen in training
            >>> from watch.tasks.fusion.datamodules.kwcoco_video_data import *  # NOQA
            >>> import watch
            >>> import ndsampler
            >>> import kwcoco
            >>> dvc_dpath = watch.find_smart_dvc_dpath()
            >>> #coco_fpath = dvc_dpath / 'Drop2-Aligned-TA1-2022-01/data.kwcoco.json'
            >>> coco_fpath = dvc_dpath / 'Aligned-Drop3-TA1-2022-03-10/data_nowv_train.kwcoco.json'
            >>> coco_fpath = dvc_dpath / 'Aligned-Drop3-TA1-2022-03-10/data_nowv_vali.kwcoco.json'
            >>> coco_dset = kwcoco.CocoDataset(coco_fpath)
            >>> #rng = kwarray.ensure_rng(0)
            >>> #vidid = rng.choice(coco_dset.videos())
            >>> #coco_dset = coco_dset.subset(coco_dset.images(vidid=vidid))
            >>> sampler = ndsampler.CocoSampler(coco_dset)
            >>> self = KWCocoVideoDataset(
            >>>     sampler,
            >>>     sample_shape=(7, 380, 380),
            >>>     window_overlap=0,
            >>>     #channels="ASI|MF_Norm|AF|EVI|red|green|blue|swir16|swir22|nir",
            >>>     channels="blue|green|red|nir|swir16|swir22,blue|green|red",
            >>>     neg_to_pos_ratio=0, time_sampling='soft2', diff_inputs=0, temporal_dropout=0.5,
            >>> )
            >>> #self.requested_tasks['change'] = False
            >>> item = self[5]
            >>> item = self[6]
            >>> item = self[7]
            >>> item = self[8]
            >>> # xdoctest: +REQUIRES(--show)
            >>> canvas = self.draw_item(item, max_channels=10, overlay_on_image=0)
            >>> import kwplot
            >>> kwplot.autompl()
            >>> kwplot.imshow(canvas)
            >>> kwplot.show_if_requested()

        Ignore:
            import kwplot
            kwplot.autompl()
            import xdev
            sample_indices = list(range(len(self)))
            for index in xdev.InteractiveIter(sample_indices):
                item = self[index]
                canvas = self.draw_item(item)
                kwplot.imshow(canvas)
                xdev.InteractiveIter.draw()

        Example:
            >>> from watch.tasks.fusion.datamodules.kwcoco_video_data import *  # NOQA
            >>> import ndsampler
            >>> import kwcoco
            >>> coco_dset = kwcoco.CocoDataset.demo('vidshapes2-multispectral', num_frames=5)
            >>> sampler = ndsampler.CocoSampler(coco_dset)
            >>> channels = 'B10|B8a|B1|B8'
            >>> sample_shape = (5, 530, 610)
            >>> self = KWCocoVideoDataset(sampler, sample_shape=sample_shape, channels=channels, diff_inputs=0, dist_weights=1, temporal_dropout=0.5)
            >>> item = self[0]
            >>> canvas = self.draw_item(item)
            >>> # xdoctest: +REQUIRES(--show)
            >>> import kwplot
            >>> kwplot.autompl()
            >>> kwplot.imshow(canvas)
            >>> kwplot.show_if_requested()

        Example:
            >>> from watch.tasks.fusion.datamodules.kwcoco_video_data import *  # NOQA
            >>> import ndsampler
            >>> import kwcoco
            >>> coco_dset = kwcoco.CocoDataset.demo('vidshapes2-multispectral', num_frames=5)
            >>> sampler = ndsampler.CocoSampler(coco_dset)
            >>> channels = 'B10|B8|B1'
            >>> sample_shape = (4, 96, 96)
            >>> self = KWCocoVideoDataset(sampler, sample_shape=sample_shape, channels=channels, neg_to_pos_ratio=0.1, true_multimodal=True)
            >>> item = self[-1]
            >>> canvas = self.draw_item(item)
            >>> # xdoctest: +REQUIRES(--show)
            >>> import kwplot
            >>> kwplot.autompl()
            >>> kwplot.imshow(canvas)
            >>> kwplot.show_if_requested()

        Example:
            >>> # xdoctest: +REQUIRES(env:DVC_DPATH)
            >>> # Run the following tests on real watch data if DVC is available
            >>> from watch.tasks.fusion.datamodules.kwcoco_video_data import *  # NOQA
            >>> import ndsampler
            >>> import kwcoco
            >>> import watch
            >>> dvc_dpath = watch.find_smart_dvc_dpath()
            >>> coco_fpath = dvc_dpath / 'Drop1-Aligned-L1/vali_data_nowv.kwcoco.json'
            >>> coco_dset = kwcoco.CocoDataset(coco_fpath)
            >>> sampler = ndsampler.CocoSampler(coco_dset)
            >>> self = KWCocoVideoDataset(sampler, sample_shape=(5, 256, 256), channels='red|green|blue|swir16|pan|lwir11|lwir12', normalize_perframe=False, true_multimodal=True)
            >>> self.disable_augmenter = True
            >>> index = 300
            >>> item = self[index]
            >>> canvas = self.draw_item(item)
            >>> # xdoctest: +REQUIRES(--show)
            >>> import kwplot
            >>> kwplot.autompl()
            >>> kwplot.imshow(canvas, doclf=1)
            >>> kwplot.show_if_requested()

        Example:
            >>> from watch.tasks.fusion.datamodules.kwcoco_video_data import *  # NOQA
            >>> import ndsampler
            >>> import kwcoco
            >>> import watch
            >>> coco_dset = watch.demo.demo_kwcoco_multisensor()
            >>> sampler = ndsampler.CocoSampler(coco_dset)
            >>> #channels = '|'.join(sorted(set(ub.flatten([c.channels.fuse().as_list() for c in coco_dset.images().coco_images]))))
            >>> #channels = '|'.join(sorted(set(ub.flatten([kwcoco.ChannelSpec.coerce(c).fuse().as_list() for c in groups.keys()]))))
            >>> # Case where all channels from sensors are aligned and padded with nans
            >>> # channels = '(sensor0,sensor1,sensor2):' + '|'.join(sorted(set(ub.flatten([c.channels.fuse().as_list() for c in coco_dset.images().coco_images]))))
            >>> # Each sensor uses all of its own channels
            >>> channels = None
            >>> self = KWCocoVideoDataset(sampler, sample_shape=(5, 256, 256), channels=channels, normalize_perframe=False, true_multimodal=True)
            >>> self.disable_augmenter = False
            >>> index = 0
            >>> index = tr = self.new_sample_grid['targets'][0]
            >>> item = self[index]
            >>> canvas = self.draw_item(item)
            >>> # xdoctest: +REQUIRES(--show)
            >>> import kwplot
            >>> kwplot.autompl()
            >>> kwplot.imshow(canvas)
            >>> kwplot.show_if_requested()

        Ignore:
            ### Useful Debugging Code. Dont remove. (Maybe could move somewhere else though)
            ### Currently there is a problem where the image data isn't loading
            ### correctly. This code verifies it for a gid and band of interest at
            ### multiple levels of abstraction: the batch viz level, the dataloader
            ### level, the ndsampler level, the kwcoco level (which seems correct,
            ### so stopping there)

            # Select the frame index of the image of interest causing issues
            # As well as the band
            frame_index = 1
            band = 'X.2'

            gid = tr['gids'][frame_index]
            sl = tr['space_slice']

            index = tr
            item = self[index]
            canvas = self.draw_item(item)
            kwplot.imshow(canvas, fnum=1)

            ### Verify the Dataset item level
            dset = self.sampler.dset
            # Also we are making an assumption about streams here, fixme if needed
            code, dataset_frame = ub.peek(item['frames'][frame_index]['modes'].items())
            band_idx = kwcoco.FusedChannelSpec.coerce(code).as_list().index(band)
            real_chan = dataset_frame[band_idx].numpy()
            dset_level_norm = kwimage.normalize_intensity(real_chan)
            kwplot.imshow(dset_level_norm, fnum=2, title='Dataloader Level')

            ### Verify (visually) the ndsampler level
            coco_img = coco_dset.coco_image(gid)
            sensor_coarse = coco_img.img.get('sensor_coarse', '*')
            matching_sensorchan = self.sample_sensorchan.matching_sensor(sensor_coarse)
            sensor_channels = matching_sensorchan.chans
            stream = ub.peek(sensor_channels.streams())
            tr_frame = tr.copy()
            tr_frame['gids'] = [gid]
            tr_frame['channels'] = stream
            sample = sampler.load_sample(
                tr_frame, with_annots=0,
                nodata='float',
                padkw={'constant_values': np.nan},
                dtype=np.float32
            )
            chan_idx = stream.as_list().index(band)
            raw_loaded = sample['im'][0, :, :, chan_idx]
            loaded_norm = kwimage.normalize_intensity(raw_loaded)
            kwplot.imshow(loaded_norm, fnum=3, title='NDsampler Level')

            ### Verify (visually) the kwcoco level
            delayed = dset.coco_image(gid).delay(space='video').take_channels(band)
            delayed_chip = delayed.crop(sl)
            chip = delayed_chip.finalize()
            full_norm = kwimage.normalize_intensity(delayed.finalize())
            full_norm = kwimage.Boxes.from_slice(sl).draw_on(full_norm)
            kwplot.imshow(full_norm, fnum=4, pnum=(1, 2, 1), title='KWcoco Full Level')
            kwplot.imshow(kwimage.normalize_intensity(chip), fnum=4, pnum=(1, 2, 2), title='KWcoco Chip Level')
        """
        if isinstance(index, dict):
            # User can specify a tr dict directly
            tr = index
            index = 'given-as-dictionary'
        else:
            if self.mode == 'test':
                # Test mode gets determenistic indexes
                tr = self.new_sample_grid['targets'][index]
            else:
                # The idea of the pool is to handle balanced sampling
                try:
                    tr_idx = self.nested_pool.sample()
                except Exception as ex:
                    print(f'Failed sample: {ex=}')
                    return None
                tr = self.new_sample_grid['targets'][tr_idx]

        tr_ = tr.copy()

        # get positive sample definition
        # collect sample
        sampler = self.sampler
        coco_dset = self.sampler.dset
        tr_['as_xarray'] = False
        tr_['use_experimental_loader'] = 1

        tr_ = self._augment_spacetime_target(tr_)

        if self.inference_only:
            with_annots = []
        else:
            with_annots = ['boxes', 'segmentation']

        NEW_TRUE_MULTIMODAL = self.true_multimodal
        ALLOW_RESAMPLE = self.resample_invalid_frames
        ALLOW_FEWER_FRAMES = 1

        if not NEW_TRUE_MULTIMODAL:
            raise NotImplementedError('old mode is gone')

        # New true-multimodal data items
        gid_to_sample: Dict[str, Dict] = {}
        gid_to_isbad: Dict[str, bool] = {}

        for gid in tr_['gids']:
            self._sample_one_frame(gid, sampler, coco_dset, tr_, with_annots,
                                   gid_to_isbad, gid_to_sample)

        if 'video_id' not in tr_:
            arbitrary_sample = ub.peek(ub.peek(gid_to_sample.values()).values())
            tr_['video_id'] = arbitrary_sample['tr']['vidid']

        vidid = tr_['video_id']
        video = coco_dset.index.videos[vidid]
        time_sampler = self.new_sample_grid['vidid_to_time_sampler'][vidid]
        video_gids = time_sampler.video_gids

        # If we skipped the main gid, record why
        main_gid = tr.get('main_gid', None)
        if main_gid is not None and gid_to_isbad[main_gid]:
            main_skip_reason = gid_to_isbad[main_gid]
        else:
            main_skip_reason = None

        if ALLOW_FEWER_FRAMES:
            error_level = 0
        else:
            error_level = 1

        if ALLOW_RESAMPLE:
            # If any image is junk allow for a resample
            if any(gid_to_isbad.values()):
                vidid = tr_['video_id']
                time_sampler = self.new_sample_grid['vidid_to_time_sampler'][vidid]
                max_tries = 30  # parameterize
                for iter_idx in range(max_tries):
                    good_gids = np.array([gid for gid, flag in gid_to_isbad.items() if not flag])
                    if len(good_gids) == len(tr['gids']):
                        break
                    bad_gids = np.array([gid for gid, flag in gid_to_isbad.items() if flag])
                    # print('resampling: {}'.format(index))
                    include_idxs = np.where(kwarray.isect_flags(time_sampler.video_gids, good_gids))[0]
                    exclude_idxs = np.where(kwarray.isect_flags(time_sampler.video_gids, bad_gids))[0]
                    try:
                        chosen = time_sampler.sample(include=include_idxs,
                                                     exclude=exclude_idxs,
                                                     error_level=error_level,
                                                     return_info=False)
                    except Exception:
                        if ALLOW_FEWER_FRAMES:
                            break
                        else:
                            raise
                    new_idxs = np.setdiff1d(chosen, include_idxs)
                    new_gids = video_gids[new_idxs]
                    # print('new_gids = {!r}'.format(new_gids))
                    if not len(new_gids):
                        print('exhausted resample possibilities')
                        # Exhausted all possibilities
                        break
                    for gid in new_gids:
                        self._sample_one_frame(gid, sampler, coco_dset, tr_,
                                               with_annots, gid_to_isbad,
                                               gid_to_sample)

        good_gids = [gid for gid, flag in gid_to_isbad.items() if not flag]
        if len(good_gids) == 0:
            # Cannot force any good sample, try and return None
            return None
            if 0:
                # We cant always do this
                # Force at least a few to be "good"
                for gid in tr['gids']:
                    gid_to_isbad[gid] = False
                good_gids = [gid for gid, flag in gid_to_isbad.items() if not flag]

        final_gids = ub.oset(video_gids) & good_gids
        num_frames = len(final_gids)
        if num_frames == 0:
            raise Exception('0 frames')

        # coco_dset.images(final_gids).lookup('date_captured')
        tr_['gids'] = final_gids

        if self.window_dims is None:
            # Do something better
            input_dsize = ub.peek(gid_to_sample[final_gids[0]])['im'].shape[1:3][::-1]
        else:
            input_dsize = self.window_dims[-2:][::-1]

        if not self.inference_only:
            # Learn more from the center of the space-time patch
            time_weights = kwimage.gaussian_patch((1, num_frames))[0]
            time_weights = time_weights / time_weights.max()
            time_weights = time_weights.clip(0, 1)
            time_weights = np.maximum(time_weights, self.min_spacetime_weight)
            space_weights = util_kwimage.upweight_center_mask(input_dsize[::-1])
            space_weights = np.maximum(space_weights, self.min_spacetime_weight)

        if 0:
            # Replace nans with windows stats
            # TODO: handle nans outside of the dataloader
            # The dataloader **should** return nan values, it is up to the
            # method to handle them. So we have to fix RunningStats
            # ALSO: this is broken.
            for gid in final_gids:
                stream_sample = gid_to_sample[gid]
                for sample in stream_sample.values():
                    im = sample['im']
                    invalid_mask = np.isnan(im)
                    if np.any(invalid_mask):
                        if np.all(invalid_mask):
                            im[:] = 0
                        else:
                            # TODO: Should use the global stream mean/std for this
                            # If that is not available, use in-window means
                            window_chan_med = np.nanmedian(im, axis=(0, 1, 2))
                            window_chan_mean = np.nanmean(im, axis=(0, 1, 2))
                            window_chan_ave = (window_chan_med + window_chan_mean) / 2
                            im[invalid_mask.any(axis=3), :] = window_chan_ave[None, None, None, :]

        if self.special_inputs:
            raise NotImplementedError(f'{self.special_inputs=}')

        if self.diff_inputs:
            raise NotImplementedError(f'{self.diff_inputs=}')

        if self.match_histograms:
            raise NotImplementedError(f'{self.match_histograms=}')

        if not self.inference_only:
            # build up info about the tracks
            dset = self.sampler.dset
            gid_to_dets: Dict[int, kwimage.Detections] = {}
            tid_to_aids = ub.ddict(list)
            tid_to_cids = ub.ddict(list)
            # tid_to_catnames = ub.ddict(list)
            for gid in final_gids:
                stream_sample = gid_to_sample[gid]
                frame_dets = None
                for sample in stream_sample.values():
                    if 'annots' in sample:
                        frame_dets: kwimage.Detections = sample['annots']['frame_dets'][0]
                        break
                if frame_dets is None:
                    raise AssertionError(ub.paragraph(
                        f'''
                        Did not sample correctly.
                        Please send this info to jon.crall@kitware.com:
                        {dset=!r}
                        {gid=!r}
                        {tr=!r}
                        {tr_=!r}
                        '''
                    ))
                gid_to_dets[gid] = frame_dets

            for gid, frame_dets in gid_to_dets.items():
                aids = frame_dets.data['aids']
                cids = frame_dets.data['cids']
                tids = dset.annots(aids).lookup('track_id', None)
                frame_dets.data['tids'] = tids
                for tid, aid, cid in zip(tids, aids, cids):
                    tid_to_aids[tid].append(aid)
                    tid_to_cids[tid].append(cid)

            # tid_to_cnames = ub.map_vals(
            #     lambda cids: list(ub.take(self.classes.id_to_node, cids, None)),
            #     tid_to_cids
            # )

            tid_to_frame_cids = ub.ddict(list)
            for gid, frame_dets in gid_to_dets.items():
                cids = frame_dets.data['cids']
                tids = frame_dets.data['tids']
                frame_tid_to_cid = ub.dzip(tids, cids)
                for tid in tid_to_aids.keys():
                    cid = frame_tid_to_cid.get(tid, None)
                    tid_to_frame_cids[tid].append(cid)

            # TODO: be more efficient at this
            tid_to_frame_cnames = ub.map_vals(
                lambda cids: list(ub.take(self.classes.id_to_node, cids, None)),
                tid_to_frame_cids
            )

            task_tid_to_cnames = {
                'saliency': {},
                'class': {},
            }
            for tid, cnames in tid_to_frame_cnames.items():
                task_tid_to_cnames['class'][tid] = heuristics.hack_track_categories(cnames, 'class')
                task_tid_to_cnames['saliency'][tid] = heuristics.hack_track_categories(cnames, 'saliency')
            # print('task_tid_to_cnames = {}'.format(ub.repr2(task_tid_to_cnames, nl=3)))
            # print('tid_to_frame_cnames = {}'.format(ub.repr2(tid_to_frame_cnames, nl=2)))

        # TODO: handle all augmentation before we construct any labels
        frame_items = []
        for time_idx, gid in enumerate(final_gids):
            img = coco_dset.index.imgs[gid]

            stream_sample = gid_to_sample[gid]
            assert len(stream_sample) > 0

            mode_imdata = {}
            mode_invalid_masks = {}
            for mode_key, sample in stream_sample.items():
                # TODO: get nodata value here
                # FIXME: nodata value needs to be handled in the kwcoco delay
                frame_chans = sample['tr']['channels'].fuse().as_list()
                mode_key = '|'.join(frame_chans)

                frame_invalid_mask = sample.get('invalid_mask', None)
                if frame_invalid_mask is not None:
                    invalid_mask = kwimage.imresize(frame_invalid_mask[0].astype(np.uint8),
                                                    dsize=input_dsize,
                                                    interpolation='nearest')
                else:
                    invalid_mask = None

                frame_imdata = sample['im'][0]
                frame, info = kwimage.imresize(frame_imdata, dsize=input_dsize,
                                               interpolation='linear',
                                               antialias=True,
                                               return_info=True)

                frame = np.asarray(frame, dtype=np.float32)

                # ensure channel dim is not squeezed
                frame_hwc = kwarray.atleast_nd(frame, 3)
                # catch nans (TODO: the network should really do this instead)
                # frame_hwc[np.isnan(frame_hwc)] = -1.
                # rearrange image axes for pytorch
                input_chw = einops.rearrange(frame_hwc, 'h w c -> c h w')
                mode_imdata[mode_key] = input_chw
                mode_invalid_masks[mode_key] = invalid_mask

            if not self.inference_only:
                frame_dets = gid_to_dets[gid]
                if frame_dets is None:
                    print('frame_dets = {!r}'.format(frame_dets))
                    raise AssertionError

            dt_captured = img.get('date_captured', None)
            if dt_captured:
                dt_captured = util_time.coerce_datetime(dt_captured)
                timestamp = dt_captured.timestamp()
            else:
                timestamp = np.nan

            sensor = img.get('sensor_coarse', '*')

            frame_item = {
                'gid': gid,
                'date_captured': img.get('date_captured', ''),
                'timestamp': timestamp,
                'time_index': time_idx,
                'sensor': sensor,
                'modes': mode_imdata,
                'change': None,
                'class_idxs': None,
                'saliency': None,
                'change_weights': None,
                'class_weights': None,
                'saliency_weights': None,
            }

            if not self.inference_only:
                # Remember to apply any transform to the dets as well
                # TODO: the info scale is on a per-mode basis, need to
                # normalize it first or compute a mode-to-truth transform.
                dets = frame_dets.scale(info['scale'])
                dets = dets.translate(info['offset'])

                # Create truth masks
                bg_idx = self.bg_idx
                space_shape = frame.shape[:2]
                frame_cidxs = np.full(space_shape, dtype=np.int32,
                                      fill_value=bg_idx)

                class_ohe_shape = (len(self.classes),) + space_shape
                salient_shape = space_shape

                # A "Salient" class is anything that is a foreground class
                # Not sure if this should be a dataloader thing or not
                frame_saliency = np.zeros(salient_shape, dtype=np.uint8)

                frame_class_ohe = np.zeros(class_ohe_shape, dtype=np.uint8)
                saliency_ignore = np.zeros(space_shape, dtype=np.uint8)
                frame_class_ignore = np.zeros(space_shape, dtype=np.uint8)

                task_target_ohe = {}
                task_target_ohe['saliency'] = frame_saliency
                task_target_ohe['class'] = frame_class_ohe

                task_target_ignore = {}
                task_target_ignore['saliency'] = saliency_ignore
                task_target_ignore['class'] = frame_class_ignore

                # Rasterize frame targets
                ann_polys = dets.data['segmentations'].to_polygon_list()
                ann_aids = dets.data['aids']
                ann_cids = dets.data['cids']
                ann_tids = dets.data['tids']

                frame_poly_weights = np.ones(space_shape, dtype=np.float32)

                # Note: it is important to respect class indexes, ids, and
                # name mappings
                # TODO: layer ordering? Multiclass prediction?
                for poly, aid, cid, tid in zip(ann_polys, ann_aids, ann_cids, ann_tids):  # NOQA

                    if self.use_conditional_classes:
                        # VERY HACKY, NEEDS REWRITE

                        if self.requested_tasks['saliency']:
                            # orig_cname = self.classes.id_to_node[cid]
                            new_salient_catname = task_tid_to_cnames['saliency'][tid][time_idx]
                            if new_salient_catname not in self.background_classes:
                                poly.fill(frame_saliency, value=1)
                            if new_salient_catname in self.ignore_classes:
                                poly.fill(saliency_ignore, value=1)

                        if self.requested_tasks['class']:
                            new_class_catname = task_tid_to_cnames['class'][tid][time_idx]
                            new_class_cidx = self.classes.node_to_idx[new_class_catname]
                            orig_cidx = self.classes.id_to_idx[cid]
                            if new_class_catname in self.ignore_classes:
                                poly.fill(frame_class_ignore, value=1)
                                poly.fill(frame_class_ohe[orig_cidx], value=1)
                            elif new_class_catname not in self.background_classes:
                                poly.fill(frame_class_ohe[new_class_cidx], value=1)
                    else:
                        cidx = self.classes.id_to_idx[cid]
                        catname = self.classes.id_to_node[cid]

                        if catname in self.background_classes:
                            pass
                        elif catname in self.ignore_classes:
                            poly.fill(saliency_ignore, value=1)
                            poly.fill(frame_class_ignore, value=1)
                            # weights should allow us to distinguish ignore from
                            # background. It shouldn't be learned on in any case.
                            poly.fill(frame_class_ohe[cidx], value=1)
                            poly.fill(frame_saliency, value=1)
                        else:
                            # Indistinguishable classes should be ignored
                            # for classification, but not saliency
                            if catname in self.undistinguished_classes:
                                poly.fill(frame_class_ignore, value=1)
                                # poly.fill(frame_class_ohe[cidx], value=0)
                                # poly.fill(frame_class_ohe[cidx], value=0)
                            poly.fill(frame_class_ohe[cidx], value=1)
                            poly.fill(frame_saliency, value=1)

                    if self.dist_weights:
                        # New feature where we encode that we care much more about
                        # segmenting the inside of the object than the outside.
                        # Effectively boundaries become uncertain.
                        import cv2
                        poly_mask = np.zeros_like(frame_class_ohe[0])
                        poly_mask = poly.fill(poly_mask, value=1)
                        dist = cv2.distanceTransform(poly_mask, cv2.DIST_L2, 3)
                        max_dist = dist.max()
                        if max_dist > 0:
                            dist_weight = dist / max_dist
                            weight_mask = dist_weight + (1 - poly_mask)
                            frame_poly_weights = frame_poly_weights * weight_mask

                frame_poly_weights = np.maximum(frame_poly_weights, self.min_spacetime_weight)

                # Postprocess (Dilate?) the truth map
                for cidx, class_map in enumerate(frame_class_ohe):
                    # class_map = util_kwimage.morphology(class_map, 'dilate', kernel=5)
                    frame_cidxs[class_map > 0] = cidx

                if self.upweight_centers:
                    frame_weights = space_weights * time_weights[time_idx] * frame_poly_weights
                else:
                    frame_weights = frame_poly_weights

                # Note: ensure this is resampled into target output space
                # Module the pixelwise weights by the 1 - the fraction of modes
                # that have nodata.
                DOWNWEIGHT_NAN_REGIONS = 1
                if DOWNWEIGHT_NAN_REGIONS:
                    nodata_total = 0.0
                    for mask in mode_invalid_masks.values():
                        if mask is None:
                            nodata_total += 0
                        else:
                            if len(mask.shape) == 3:
                                nodata_total += ((mask.sum(axis=2) / mask.shape[2])).astype(float)
                            else:
                                nodata_total += mask.astype(float)
                    # nodata_total = np.add.reduce([0 if mask is None else mask.sum(axis=2) / mask.shape[2] for mask in mode_invalid_masks.values()])
                    total_bands = len(mode_invalid_masks)
                    nodata_frac = nodata_total / total_bands
                    nodata_weight = 1 - nodata_frac
                    frame_weights = frame_weights * nodata_weight

                # Dilate ignore masks (dont care about the surrounding area # either)
                # frame_saliency = util_kwimage.morphology(frame_saliency, 'dilate', kernel=ignore_dilate)
                saliency_ignore = util_kwimage.morphology(saliency_ignore, 'dilate', kernel=self.ignore_dilate)
                frame_class_ignore = util_kwimage.morphology(frame_class_ignore, 'dilate', kernel=self.ignore_dilate)

                saliency_weights = frame_weights * (1 - saliency_ignore)
                class_weights = frame_weights * (1 - frame_class_ignore)
                saliency_weights = saliency_weights.clip(0, 1)
                frame_weights = frame_weights.clip(0, 1)

            if not self.inference_only:
                if self.requested_tasks['class'] or self.requested_tasks['change']:
                    frame_item['class_idxs'] = frame_cidxs
                    frame_item['class_weights'] = class_weights
                if self.requested_tasks['saliency']:
                    frame_item['saliency'] = frame_saliency
                    frame_item['saliency_weights'] = saliency_weights

            frame_items.append(frame_item)

        if self.normalize_perframe:
            for frame_item in frame_items:
                frame_modes = frame_item['modes']
                for mode_key in list(frame_modes.keys()):
                    mode_data = frame_modes[mode_key]
                    to_restack = []
                    for item in mode_data:
                        # TODO: use real nodata values? Ideally they have
                        # already been converted into nans
                        mask = (item != 0) & np.isfinite(item)
                        norm_item = kwimage.normalize_intensity(item, params={
                            'high': 0.90,
                            'mid': 0.5,
                            'low': 0.01,
                            'mode': 'linear',
                        }, mask=mask)
                        to_restack.append(norm_item)
                    mode_data_normed = np.stack(to_restack, axis=0)
                    frame_modes[mode_key] = mode_data_normed

        # Add in change truth
        if not self.inference_only:
            if self.requested_tasks['change']:
                if frame_items:
                    frame1 = frame_items[0]
                for frame1, frame2 in ub.iter_window(frame_items, 2):
                    frame_change = (frame1['class_idxs'] != frame2['class_idxs']).astype(np.uint8)
                    frame_change = util_kwimage.morphology(frame_change, 'open', kernel=3)
                    change_weights = frame1['class_weights'] * frame2['class_weights']
                    frame2['change'] = frame_change
                    frame2['change_weights'] = change_weights.clip(0, 1)

        truth_keys = [
            'change', 'class_idxs',
            'saliency', 'class_weights',
            'saliency_weights', 'change_weights'
        ]

        # If we are augmenting
        fliprot_params = tr_.get('fliprot_params', None)
        if fliprot_params is not None:
            for frame_item in frame_items:
                frame_modes = frame_item['modes']
                for mode_key in list(frame_modes.keys()):
                    mode_data = frame_modes[mode_key]
                    frame_modes[mode_key] = fliprot(mode_data, **fliprot_params, axes=[1, 2])
                for key in truth_keys:
                    data = frame_item.get(key, None)
                    if data is not None:
                        frame_item[key] = fliprot(data, **fliprot_params, axes=[-2, -1])

        # Convert data to torch
        for frame_item in frame_items:
            frame_modes = frame_item['modes']
            for mode_key in list(frame_modes.keys()):
                mode_data = frame_modes[mode_key]
                frame_modes[mode_key] = kwarray.ArrayAPI.tensor(mode_data)
            for key in truth_keys:
                data = frame_item.get(key, None)
                if data is not None:
                    frame_item[key] = kwarray.ArrayAPI.tensor(data)

        positional_tensors = None
        if True:
            # TODO: what is the standard way to do the learned embedding
            # "input vector"?

            # TODO: preprocess any auxiliary learnable information into a
            # Tensor. It is likely ideal to pre-stack whenever possible, but we
            # need to keep the row-form data to make visualization
            # straight-forward. We could use a flag to toggle it depending on
            # if we need to visualize or not.
            permode_datas = ub.ddict(list)
            prev_timestamp = None

            time_index_encoding = utils.ordinal_position_encoding(len(frame_items), 8).numpy()

            for frame_item in frame_items:

                k = 'timestamp'
                frame_timestamp = np.array([frame_item[k]]).astype(np.float32)

                for mode_code in frame_item['modes'].keys():
                    # Maybe this should be a model responsibility.
                    # I dont like defining the positional encoding in the
                    # dataset
                    key_tensor = _string_to_hashvec(mode_code)
                    permode_datas['mode_tensor'].append(key_tensor)
                    #
                    k = 'time_index'
                    time_index = frame_item[k]
                    # v = np.array([frame_item[k]]).astype(np.float32)
                    v = time_index_encoding[time_index]
                    permode_datas[k].append(v)

                    if prev_timestamp is None:
                        time_offset = np.array([0]).astype(np.float32)
                    else:
                        time_offset = frame_timestamp - prev_timestamp

                    # TODO: add seasonal positional encoding

                    permode_datas['time_offset'].append(time_offset)

                    k = 'sensor'
                    key_tensor = _string_to_hashvec(k)
                    permode_datas[k].append(key_tensor)

                frame_item['time_offset'] = time_offset
                prev_timestamp = frame_timestamp

            positional_arrays = ub.map_vals(np.stack, permode_datas)
            time_offset = positional_arrays.pop('time_offset', None)
            if time_offset is not None:
                time_offset = time_offset + 1
                time_offset[np.isnan(time_offset)] = 0.1
                positional_arrays['time_offset'] = np.log(time_offset)
            else:
                print(list(permode_datas.keys()))

            # This is flattened for each frame for each mode.
            # A bit hacky, not in love with it.
            positional_tensors = ub.map_vals(torch.from_numpy, positional_arrays)

        # Only pass back some of the metadata (because I think torch
        # multiprocessing makes a new file descriptor for every Python object
        # or something like that)
        tr_subset = ub.dict_isect(tr_, {
            'gids', 'space_slice', 'vidid', 'fliprot_params',
        })
        if main_skip_reason:
            tr_subset['main_skip_reason'] = main_skip_reason
        item = {
            # TODO: breakup modes into different items
            'index': index,
            'frames': frame_items,
            'positional_tensors': positional_tensors,
            'video_id': vidid,
            'video_name': video['name'],
            'tr': tr_subset
        }
        return item

    def cached_dataset_stats(self, num=None, num_workers=0, batch_size=2,
                             with_intensity=True, with_class=True):
        """
        Compute the normalization stats, and caches them

        TODO:
            - [ ] Does this dataset have access to the workdir?
            - [ ] Cacher needs to depend on config of this dataset
        """
        # Get stats on the dataset (todo: nice way to disable augmentation temporarilly for this)
        depends = ub.odict([
            ('num', num),
            ('hashid', self.sampler.dset._build_hashid()),
            ('sensorchan', self.input_sensorchan.concise().spec),
            ('normalize_perframe', self.normalize_perframe),
            ('with_intensity', with_intensity),
            ('with_class', with_class),
            ('depends_version', 15),  # bump if `compute_dataset_stats` changes
        ])
        workdir = None
        cacher = ub.Cacher('dset_mean', dpath=workdir, depends=depends)
        dataset_stats = cacher.tryload()
        if dataset_stats is None or ub.argflag('--force-recompute-stats'):
            dataset_stats = self.compute_dataset_stats(
                num, num_workers=num_workers, batch_size=batch_size)
            cacher.save(dataset_stats)
        return dataset_stats

    def compute_dataset_stats(self, num=None, num_workers=0, batch_size=2,
                              with_intensity=True, with_class=True, with_vidid=True):
        """
        Args:
            num (int | None): number of input items to compute stats for

        Example:
            >>> from watch.tasks.fusion.datamodules.kwcoco_video_data import *  # NOQA
            >>> import ndsampler
            >>> import kwcoco
            >>> coco_dset = kwcoco.CocoDataset.demo('vidshapes2-multispectral', num_frames=3)
            >>> sampler = ndsampler.CocoSampler(coco_dset)
            >>> sample_shape = (2, 256, 256)
            >>> self = KWCocoVideoDataset(sampler, sample_shape=sample_shape, channels=None, true_multimodal=True)
            >>> self.compute_dataset_stats()

        Example:
            >>> from watch.tasks.fusion.datamodules.kwcoco_video_data import *  # NOQA
            >>> import ndsampler
            >>> import kwcoco
            >>> coco_dset = kwcoco.CocoDataset.demo('vidshapes2')
            >>> sampler = ndsampler.CocoSampler(coco_dset)
            >>> sample_shape = (2, 256, 256)
            >>> self = KWCocoVideoDataset(sampler, sample_shape=sample_shape, channels=None)
            >>> stats = self.compute_dataset_stats()
            >>> assert stats['class_freq']['star'] > 0 or stats['class_freq']['superstar'] > 0 or stats['class_freq']['eff'] > 0
            >>> assert stats['class_freq']['background'] > 0

        CommandLine:
            DVC_DPATH=$HOME/data/dvc-repos/smart_watch_dvc xdoctest -m watch.tasks.fusion.datamodules.kwcoco_video_data KWCocoVideoDataset.compute_dataset_stats:1

        Ignore:
            import xdev
            from watch.tasks.fusion.datamodules.kwcoco_video_data import *  # NOQA
            globals().update(xdev.get_func_kwargs(KWCocoVideoDataset.compute_dataset_stats))

        Example:
            >>> # xdoctest: +REQUIRES(env:DVC_DPATH)
            >>> # Run the following tests on real watch data if DVC is available
            >>> from watch.tasks.fusion.datamodules.kwcoco_video_data import *  # NOQA
            >>> import ndsampler
            >>> import kwcoco
            >>> import watch
            >>> dvc_dpath = watch.find_smart_dvc_dpath()
            >>> coco_fpath = dvc_dpath / 'drop1-S2-L8-aligned/data.kwcoco.json'
            >>> coco_dset = kwcoco.CocoDataset(coco_fpath)
            >>> sampler = ndsampler.CocoSampler(coco_dset)
            >>> sample_shape = (6, 256, 256)
            >>> channels = 'blue|green|red|nir|swir16'
            >>> #channels = 'rice_field|cropland|water|inland_water|river_or_stream|sebkha|snow_or_ice_field|bare_ground|sand_dune|built_up|grassland|brush|forest|wetland|road'
            >>> #channels = 'matseg_0|matseg_1|matseg_2|matseg_3|matseg_4'
            >>> self = KWCocoVideoDataset(sampler, sample_shape=sample_shape, channels=channels, neg_to_pos_ratio=1.0)
            >>> item = self[100]
            >>> #self.compute_dataset_stats(num=10)
            >>> num_workers = 0
            >>> num = 100
            >>> batch_size = 6
            >>> self.compute_dataset_stats(num=num, num_workers=num_workers, batch_size=batch_size)

        Example:
            >>> from watch.tasks.fusion.datamodules.kwcoco_video_data import *  # NOQA
            >>> import watch
            >>> from watch.tasks.fusion import datamodules
            >>> num = 10
            >>> datamodule = datamodules.KWCocoVideoDataModule(
            >>>     train_dataset='vidshapes-watch', chip_size=64, time_steps=3,
            >>>     num_workers=0, batch_size=3, true_multimodal=True,
            >>>     normalize_inputs=num)
            >>> datamodule.setup('fit')
            >>> self = datamodule.torch_datasets['train']
            >>> coco_dset = self.sampler.dset
            >>> print({c.get('sensor_coarse') for c in coco_dset.images().coco_images})
            >>> print({c.channels.spec for c in coco_dset.images().coco_images})
            >>> num_workers = 0
            >>> batch_size = 6
            >>> s = (self.compute_dataset_stats(num=num))
            >>> print('s = {}'.format(ub.repr2(s, nl=3)))
            >>> self.compute_dataset_stats(num=num, with_intensity=False)
            >>> self.compute_dataset_stats(num=num, with_class=False)
            >>> self.compute_dataset_stats(num=num, with_class=False, with_intensity=False)

        Example:
            >>> # xdoctest: +REQUIRES(env:DVC_DPATH)
            >>> # Run the following tests on real watch data if DVC is available
            >>> from watch.tasks.fusion.datamodules.kwcoco_video_data import *  # NOQA
            >>> import ndsampler
            >>> import kwcoco
            >>> import watch
            >>> dvc_dpath = watch.find_smart_dvc_dpath()
            >>> coco_fpath = dvc_dpath / 'Aligned-Drop3-TA1-2022-03-10/combo_LM_nowv_vali.kwcoco.json'
            >>> coco_dset = kwcoco.CocoDataset(coco_fpath)
            >>> sampler = ndsampler.CocoSampler(coco_dset)
            >>> sample_shape = (6, 256, 256)
            >>> channels = '(L8,S2):(blue|green|red|nir|swir16),S2:red|green|blue'
            >>> self = KWCocoVideoDataset(sampler, sample_shape=sample_shape, channels=channels, neg_to_pos_ratio=1.0)
            >>> item = self[100]
            >>> #self.compute_dataset_stats(num=10)
            >>> num_workers = 0
            >>> num = 100
            >>> batch_size = 6
            >>> self.compute_dataset_stats(num=num, num_workers=num_workers, batch_size=batch_size, with_vidid=True)
        """
        num = num if isinstance(num, int) and num is not True else 1000
        if not with_class and not with_intensity:
            num = 1  # efficiency hack
        stats_idxs = kwarray.shuffle(np.arange(len(self)), rng=0)[0:min(num, len(self))]
        stats_subset = torch.utils.data.Subset(self, stats_idxs)

        # Hack: disable augmentation if we are doing that
        self.disable_augmenter = True

        loader = torch.utils.data.DataLoader(
            stats_subset,
            collate_fn=ub.identity, num_workers=num_workers, shuffle=True,
            batch_size=batch_size)

        # dataset_sensors = set(
        #     self.sampler.dset.images().lookup('sensor_coarse', None))
        # Track moving average of each fused channel stream
        # channel_stats = {
        #     sensor: {key: kwarray.RunningStats()
        #              for key in self.input_channels.keys()}
        #     for sensor in dataset_sensors
        # }
        channel_stats = ub.AutoDict()

        timer = ub.Timer().tic()
        timer.first = 1

        classes = self.classes
        num_classes = len(classes)
        bins = np.arange(num_classes + 1)
        total_freq = np.zeros(num_classes, dtype=np.int64)

        sensor_mode_hist = ub.ddict(lambda: 0)

        video_id_histogram = {}

        # TODO: we should ensure instance level frequency data as well
        # as pixel level frequency data.

        # TODO: we should ensure we include at least one sample from each type
        # of modality.
        # Note: the requested order of the channels could be different that
        # what is registered in the dataset. Need to find a good way to account
        # for this.

        # Make a list of all unique modes in the dataset.
        NEW = 1
        if NEW:
            # User specifies all of this explicitly now
            unique_sensor_modes = set(
                (s.sensor.spec, s.chans.spec)
                for s in self.input_sensorchan.streams())
        else:
            FIX_CSPEC = 1
            # Make a list of all unique modes in the dataset.
            unique_sensor_modes = set(sensor_mode_hist.keys())
            if True:
                print('Looking for unique modes')
                # This looks at the entire dataset, might want to
                # make a better way of getting this info.
                # self.sampler.dset.videos().images
                coco_images = self.sampler.dset.images().coco_images
                hacked = set()
                for c in coco_images:
                    sspec = c.img.get('sensor_coarse', '*')
                    img_chans = c.channels.fuse().normalize()

                    # TODO: the input_channels should eventually define
                    # the sensor so we can do a specific lookup.
                    # In the meantime, hack it.

                    # Get only the requested bands for this sensor.
                    sensor_input_chans = self.input_sensorchan.matching_sensor(sspec).chans

                    if FIX_CSPEC:
                        # # Get only the requested bands for this sensor.
                        # sensor_input_chans = self.input_sensorchan.matching_sensor(sspec).chans
                        # canidates = sensor_input_chans & img_chans
                        canidates = sensor_input_chans & img_chans
                        for a in canidates.streams():
                            for b in sensor_input_chans.streams():
                                if a.spec == b.spec:
                                    hacked.add((sspec, a.spec))
                    else:
                        # Ensure channels are returned in requested order
                        cspec = (self.input_channels & c.channels.fuse().normalize()).fuse().normalize().spec
                        if cspec:
                            hacked.add((sspec, cspec))
                unique_sensor_modes.update(hacked)

        print('unique_sensor_modes = {}'.format(ub.repr2(unique_sensor_modes, nl=1)))

        # TODO: we can compute the intensity histogram much more efficiently by
        # only doing it for unique channels (which might be duplicated)
        prog = ub.ProgIter(loader, desc='estimate dataset stats')
        for batch_items in prog:
            for item in batch_items:
                if item is None:
                    continue
                if with_vidid:
                    vidid = item['video_id']
                    if vidid not in set(video_id_histogram.keys()):
                        video_id_histogram[vidid] = 0
                    video_id_histogram[vidid] += 1
                for frame_item in item['frames']:
                    if with_class:
                        class_idxs = frame_item['class_idxs']
                        # print(np.unique(class_idxs))
                        item_freq = np.histogram(class_idxs.ravel(), bins=bins)[0]
                        total_freq += item_freq
                    if with_intensity:
                        sensor_code = frame_item['sensor']
                        modes = frame_item['modes']

                        for mode_code, mode_val in modes.items():
                            sensor_mode_hist[(sensor_code, mode_code)] += 1
                            running = channel_stats[sensor_code][mode_code]
                            if not running:
                                running = kwarray.RunningStats()
                                channel_stats[sensor_code][mode_code] = running
                            val = mode_val.numpy()
                            flags = np.isfinite(val)
                            if not np.all(flags):
                                # Hack it:
                                val[~flags] = 0
                            running.update(val.astype(np.float64))

            if timer.first or timer.toc() > 5:
                from watch.utils.slugify_ext import smart_truncate
                if with_class:
                    intermediate = ub.sorted_vals(ub.dzip(classes, total_freq), reverse=True)
                    intermediate_text = ub.repr2(intermediate, compact=1)
                    intermediate_text = smart_truncate(intermediate_text, max_length=40, trunc_loc=0.8)
                else:
                    intermediate_text = ''

                if with_intensity:
                    curr = ub.dict_isect(running.summarize(keepdims=False), {'mean', 'std', 'max', 'min'})
                    curr = ub.map_vals(float, curr)
                    text = ub.repr2(curr, compact=1, precision=1, nl=0) + ' ' + intermediate_text
                else:
                    text = intermediate_text
                prog.set_postfix_str(text)
                timer.first = 0
                timer.tic()
        self.disable_augmenter = False

        channel_stats = channel_stats.to_dict()

        # Return the raw counts and let the model choose how to handle it
        if with_class:
            class_freq = ub.dzip(classes, total_freq)
        else:
            class_freq = None

        if with_intensity:
            input_stats = {}
            for sensor, submodes in channel_stats.items():
                for chan_key, running in submodes.items():
                    perchan_stats = running.summarize(axis=(1, 2))
                    input_stats[(sensor, chan_key)] = {
                        'mean': perchan_stats['mean'].round(3),  # only take 3 sigfigs
                        'std': np.maximum(perchan_stats['std'], 1e-3).round(3),
                    }
        else:
            input_stats = None

        dataset_stats = {
            'unique_sensor_modes': unique_sensor_modes,
            'sensor_mode_hist': dict(sensor_mode_hist),
            'input_stats': input_stats,
            'class_freq': class_freq,
            'video_id_histogram': video_id_histogram,
        }
        return dataset_stats

    @profile
    def draw_item(self, item, item_output=None, combinable_extra=None,
                  max_channels=5, max_dim=224, norm_over_time=0,
                  overlay_on_image=False, draw_weights=True):
        """
        Visualize an item produced by this DataSet.

        Each channel will be a row, and each column will be a timestep.

        Args:
            item (Dict): An item returned from the torch Dataset.

            overlay_on_image (bool):
                if True, the truth and prediction is drawn on top of
                an image, otherwise it is drawn on a black image.

            max_dim (int):
                max dimension to resize each grid cell to.

            max_channels (int) :
                maximum number of channel rows to draw

            item_output (Dict):
                Special task keys that we know how to plot.
                These should be some sort of binary or class prediction from
                the network. I'm not sure how best to pass the details
                of how they should be interpreted.

                Known keys:
                    change_probs
                    class_probs

        Example:
            >>> from watch.tasks.fusion.datamodules.kwcoco_video_data import *  # NOQA
            >>> import ndsampler
            >>> import kwcoco
            >>> coco_dset = kwcoco.CocoDataset.demo('vidshapes2-multispectral', num_frames=5)
            >>> sampler = ndsampler.CocoSampler(coco_dset)
            >>> channels = 'B10|B8a|B1|B8|B11'
            >>> combinable_extra = [['B10', 'B8', 'B8a']]  # special behavior
            >>> # combinable_extra = None  # uncomment for raw behavior
            >>> sample_shape = (5, 530, 610)
            >>> self = KWCocoVideoDataset(sampler, sample_shape=sample_shape, channels=channels)
            >>> index = len(self) // 4
            >>> item = self[index]
            >>> fliprot_params = item['tr'].get('fliprot_params', None)
            >>> # Calculate the probability of change for each frame
            >>> item_output = {}
            >>> change_prob_list = []
            >>> for _ in range(1, sample_shape[0]):
            >>>     change_prob = kwimage.Heatmap.random(
            >>>         dims=sample_shape[1:3], classes=1).data['class_probs'][0]
            >>>     if fliprot_params:
            >>>         change_prob = fliprot(change_prob, **fliprot_params)
            >>>     change_prob_list += [change_prob]
            >>> change_probs = np.stack(change_prob_list)
            >>> item_output['change_probs'] = change_probs  # first frame does not have change
            >>> #
            >>> # Probability of each class for each frame
            >>> class_prob_list = []
            >>> for _ in range(0, sample_shape[0]):
            >>>     class_prob = kwimage.Heatmap.random(
            >>>         dims=sample_shape[1:3], classes=list(sampler.classes)).data['class_probs']
            >>>     class_prob_ = einops.rearrange(class_prob, 'c h w -> h w c')
            >>>     if fliprot_params:
            >>>         class_prob_ = fliprot(class_prob_, **fliprot_params)
            >>>     class_prob_list += [class_prob_]
            >>> class_probs = np.stack(class_prob_list)
            >>> item_output['class_probs'] = class_probs  # first frame does not have change
            >>> #binprobs[0][:] = 0  # first change prob should be all zeros
            >>> canvas = self.draw_item(item, item_output, combinable_extra=combinable_extra, overlay_on_image=1)
            >>> canvas2 = self.draw_item(item, item_output, combinable_extra=combinable_extra, max_channels=3, overlay_on_image=0)
            >>> # xdoctest: +REQUIRES(--show)
            >>> import kwplot
            >>> kwplot.autompl()
            >>> kwplot.imshow(canvas, fnum=1, pnum=(1, 2, 1))
            >>> kwplot.imshow(canvas2, fnum=1, pnum=(1, 2, 2))
            >>> kwplot.show_if_requested()

        Ignore:
            import netharn as nh
            nh.data.collate._debug_inbatch_shapes(item)
            nh.data.collate._debug_inbatch_shapes(item_output)

        Example:
            >>> # xdoctest: +REQUIRES(env:DVC_DPATH)
            >>> from watch.tasks.fusion.datamodules.kwcoco_video_data import *  # NOQA
            >>> import os
            >>> from os.path import join
            >>> import ndsampler
            >>> import kwcoco
            >>> import watch
            >>> dvc_dpath = watch.find_smart_dvc_dpath()
            >>> #coco_fpath = dvc_dpath / 'drop1-S2-L8-aligned/combo_data.kwcoco.json'
            >>> coco_fpath = dvc_dpath / 'Drop1-Aligned-L1/vali_data_nowv.kwcoco.json'
            >>> coco_dset = kwcoco.CocoDataset(coco_fpath)
            >>> sampler = ndsampler.CocoSampler(coco_dset)
            >>> sample_shape = (3, 128, 128)
            >>> self = KWCocoVideoDataset(sampler, sample_shape=sample_shape, channels='swamp|red|green|blue|swir22|lwir12|pan|nir', true_multimodal=True)
            >>> vidid = self.sampler.dset.videos()[1]
            >>> from watch.cli.coco_visualize_videos import video_track_info
            >>> tid_to_info = video_track_info(coco_dset, vidid)
            >>> for track_info in tid_to_info.values():
            >>>     cnames = coco_dset.annots(track_info['track_aids']).cnames
            >>>     print(ub.oset(cnames))
            >>> track_info = list(tid_to_info.values())[1]
            >>> index = {
            >>>     'space_slice': track_info['full_vid_box'].quantize().to_slices()[0],
            >>>     'gids': track_info['track_gids'][2:9],
            >>>     'video_id': vidid,
            >>> }
            >>> max_channels = 5
            >>> max_dim = 256
            >>> norm_over_time = True
            >>> overlay_on_image = False
            >>> combinable_extra = None
            >>> self.disable_augmenter = True
            >>> item = self.__getitem__(index)
            >>> canvas = self.draw_item(item, overlay_on_image=overlay_on_image, norm_over_time=norm_over_time)
            >>> # xdoctest: +REQUIRES(--show)
            >>> import kwplot
            >>> kwplot.autompl()
            >>> kwplot.imshow(canvas)
            >>> kwplot.show_if_requested()

        Ignore:
            import xdev
            globals().update(xdev.get_func_kwargs(KWCocoVideoDataset.draw_item))
        """
        if item is None:
            # BIG RED X
            # h, w = vertical_stack[-1].shape[0:2]
            h = w = (max_dim or 224)
            bad_canvas = kwimage.draw_text_on_image(
                {'width': w, 'height': h}, 'X', org=(w // 2, h // 2),
                valign='center', halign='center', fontScale=10,
                color='red')
            return bad_canvas

        from watch.tasks.fusion.datamodules.batch_visualization import BatchVisualizationBuilder

        builder = BatchVisualizationBuilder(
            item=item, item_output=item_output,
            default_combinable_channels=self.default_combinable_channels,
            norm_over_time=norm_over_time, max_dim=max_dim,
            max_channels=max_channels, overlay_on_image=overlay_on_image,
            draw_weights=draw_weights, combinable_extra=combinable_extra,
            classes=self.classes, requested_tasks=self.requested_tasks)
        canvas = builder.build()
        return canvas

    def make_loader(self, batch_size=1, num_workers=0, shuffle=False,
                    pin_memory=False):
        """
        Example:
            >>> from watch.tasks.fusion.datamodules.kwcoco_video_data import *  # NOQA
            >>> import ndsampler
            >>> import kwcoco
            >>> coco_dset = kwcoco.CocoDataset.demo('vidshapes2-multispectral', num_frames=5)
            >>> sampler = ndsampler.CocoSampler(coco_dset)
            >>> self = KWCocoVideoDataset(sampler, sample_shape=(3, 530, 610))
            >>> loader = self.make_loader(batch_size=2)
            >>> batch = next(iter(loader))
        """
        loader = torch.utils.data.DataLoader(
            self, batch_size=batch_size, num_workers=num_workers,
            shuffle=shuffle, pin_memory=pin_memory, collate_fn=ub.identity)
        return loader


class NestedPool(list):
    """
    Example:
        >>> from watch.tasks.fusion.datamodules.kwcoco_video_data import *  # NOQA
        >>> nested1 = NestedPool([[[1], [2, 3], [4, 5, 6], [7, 8, 9, 0]], [[11, 12, 13]]])
        >>> print({nested1.sample() for i in range(100)})
        >>> nested2 = NestedPool([[101], [102, 103], [104, 105, 106], [107, 8, 9, 0]])
        >>> print({nested2.sample() for i in range(100)})
        >>> nested3 = NestedPool([nested1, nested2, [4, 59, 9, [], []]])
        >>> print({nested3.sample() for i in range(100)})
        >>> print(ub.repr2(ub.dict_hist(nested3.sample() for i in range(100))))
    """
    def __init__(nested, pools, rng=None):
        super().__init__(pools)
        nested.rng = rng = kwarray.ensure_rng(rng)
        nested.pools = pools

    def sample(nested):
        # Hack for empty lists
        chosen = nested
        i = 0
        while ub.iterable(chosen):
            chosen = nested
            i += 1
            while ub.iterable(chosen):
                i += 1
                num = len(chosen)
                if i > 100000:
                    raise Exception('Too many samples. Bad balance?')
                if not num:
                    break
                idx = nested.rng.randint(0, num)
                chosen = chosen[idx]

        return chosen


def _boxes_snap_to_edges(given_box, snap_target):
    """
    Ignore:
        given_box = space_box
        , snap_target

        >>> from watch.tasks.fusion.datamodules.kwcoco_video_data import *  # NOQA
        >>> from watch.tasks.fusion.datamodules.kwcoco_video_data import _boxes_snap_to_edges
        >>> snap_target = kwimage.Boxes([[0, 0, 10, 10]], 'ltrb')
        >>> given_box = kwimage.Boxes([[-3, 5, 3, 13]], 'ltrb')
        >>> adjusted_box = _boxes_snap_to_edges(given_box, snap_target)
        >>> print('adjusted_box = {!r}'.format(adjusted_box))

        _boxes_snap_to_edges(kwimage.Boxes([[-3, 3, 20, 13]], 'ltrb'), snap_target)
        _boxes_snap_to_edges(kwimage.Boxes([[-3, -3, 3, 3]], 'ltrb'), snap_target)
        _boxes_snap_to_edges(kwimage.Boxes([[7, 7, 15, 15]], 'ltrb'), snap_target)
    """
    s_x1, s_y1, s_x2, s_y2 = snap_target.components
    g_x1, g_y1, g_x2, g_y2 = given_box.components

    xoffset1 = -np.minimum((g_x1 - s_x1), 0)
    yoffset1 = -np.minimum((g_y1 - s_y1), 0)

    xoffset2 = np.minimum((s_x2 - g_x2), 0)
    yoffset2 = np.minimum((s_y2 - g_y2), 0)

    xoffset = (xoffset1 + xoffset2).ravel()[0]
    yoffset = (yoffset1 + yoffset2).ravel()[0]

    adjusted_box = given_box.translate((xoffset, yoffset))
    return adjusted_box


def fliprot(img, rot_k=0, flip_axis=None, axes=(0, 1)):
    """
    Args:
        img (ndarray): H, W, C

        rot_k (int): number of ccw rotations

        flip_axis(Tuple[int, ...]):
            either [], [0], [1], or [0, 1].
            0 is the y axis and 1 is the x axis.

        axes (Typle[int, int]): the location of the y and x axes

    Example:
        >>> img = np.arange(16).reshape(4, 4)
        >>> unique_fliprots = [
        >>>     {'rot_k': 0, 'flip_axis': None},
        >>>     {'rot_k': 0, 'flip_axis': (0,)},
        >>>     {'rot_k': 1, 'flip_axis': None},
        >>>     {'rot_k': 1, 'flip_axis': (0,)},
        >>>     {'rot_k': 2, 'flip_axis': None},
        >>>     {'rot_k': 2, 'flip_axis': (0,)},
        >>>     {'rot_k': 3, 'flip_axis': None},
        >>>     {'rot_k': 3, 'flip_axis': (0,)},
        >>> ]
        >>> for params in unique_fliprots:
        >>>     img_fw = fliprot(img, **params)
        >>>     img_inv = inv_fliprot(img_fw, **params)
        >>>     assert np.all(img == img_inv)
    """
    if rot_k != 0:
        img = np.rot90(img, k=rot_k, axes=axes)
    if flip_axis is not None:
        _flip_axis = np.asarray(axes)[flip_axis]
        img = np.flip(img, axis=_flip_axis)
    return img


def inv_fliprot(img, rot_k=0, flip_axis=None, axes=(0, 1)):
    """
    Undo a fliprot

    Args:
        img (ndarray): H, W, C
    """
    if flip_axis is not None:
        _flip_axis = np.asarray(axes)[flip_axis]
        img = np.flip(img, axis=_flip_axis)
    if rot_k != 0:
        img = np.rot90(img, k=-rot_k, axes=axes)
    return img


@ub.memoize
def _string_to_hashvec(key):
    """
    Transform a string into a 16D float32 uniformly distributed random Tensor
    based on the hash of the string.
    """
    key_hash = ub.hash_data(key, base=16, hasher='blake3').encode()
    key_tensor = np.frombuffer(memoryview(key_hash), dtype=np.int32).astype(np.float32)
    key_tensor = key_tensor / np.linalg.norm(key_tensor)
    return key_tensor
