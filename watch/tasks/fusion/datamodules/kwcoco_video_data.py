"""
Defines a torch Dataset and lightning DataModule for kwcoco video data.
"""
import einops
import kwarray
import kwcoco
import kwimage
import ndsampler
import numpy as np
import pathlib
import pytorch_lightning as pl
import random
import torch
import ubelt as ub
from kwcoco import channel_spec
from torch.utils import data
from watch import heuristics
from watch.utils import kwcoco_extensions
from watch.utils import util_bands
from watch.utils import util_iter
from watch.utils import util_kwimage
from watch.utils import util_time
# from watch.utils import util_norm
from watch.utils.lightning_ext import util_globals
from watch.tasks.fusion import utils
from typing import Dict, List  # NOQA

# __all__ = ['KWCocoVideoDataModule', 'KWCocoVideoDataset']

try:
    import xdev
    profile = xdev.profile
except Exception:
    profile = ub.identity


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

        if 0:
            tr = {
                'gids': ([816, 817, 818, 822, 824, 825, 905]),
                'space_slice': (slice(0, 512, None), slice(0, 512, None))
            }

            tr = {
                'gids': ([816, 817, 818, 822]),
                'space_slice': (slice(0, 512, None), slice(0, 512, None))}

            tr = {
                'gids': ([905]),
                'space_slice': (slice(0, 512, None), slice(0, 512, None))}

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

    def __init__(
        self,
        train_dataset=None,
        vali_dataset=None,
        test_dataset=None,
        time_steps=2,
        chip_size=128,
        time_overlap=0,
        chip_overlap=0.1,
        neg_to_pos_ratio=1.0,
        time_sampling='contiguous',
        time_span='2y',
        exclude_sensors=None,
        channels=None,
        batch_size=4,
        normalize_inputs=False,
        normalize_perframe=False,
        match_histograms=False,
        upweight_centers=True,
        diff_inputs=False,
        verbose=1,
        num_workers=4,
        torch_sharing_strategy='default',
        torch_start_method='default',
        resample_invalid_frames=True,
        true_multimodal=True,
        use_grid_positives=True,
        use_centered_positives=False,
        temporal_dropout=0.0,
        max_epoch_length=None,
        use_conditional_classes=True,
        ignore_dilate=11,
        min_spacetime_weight=0.5,
        dist_weights=False,
        use_cloudmask=True,
    ):
        """
        Args:
            train_dataset : path to the train kwcoco file
            vali_dataset : path to the validation kwcoco file
            test_dataset : path to the test kwcoco file
            time_steps (int) : number of time steps in an item
            chip_size (int) : width and height of an item
            time_overlap (float): fraction of time steps to overlap
            chip_overlap (float): fraction of space steps to overlap
            neg_to_pos_ratio (float): maximum ratio of samples with no annotations to samples with annots
            time_sampling (str): Strategy for expanding the time window across non-contiguous frames
            channels : channels to use should be ChannelSpec coercable
            batch_size (int) : number of items per batch
            num_workers (int) : number of background workers
            normalize_inputs : if True, computes the mean/std for this dataset on
                each mode so this can be passed to the model.
        """
        super().__init__()
        self.verbose = verbose
        self.save_hyperparameters()
        self.train_kwcoco = train_dataset
        self.vali_kwcoco = vali_dataset
        self.test_kwcoco = test_dataset
        self.time_steps = int(time_steps)
        self.chip_size = int(chip_size)
        self.time_overlap = time_overlap
        self.chip_overlap = chip_overlap
        self.neg_to_pos_ratio = neg_to_pos_ratio
        self.batch_size = batch_size
        self.normalize_inputs = normalize_inputs
        self.time_span = time_span
        self.max_epoch_length = max_epoch_length
        self.use_conditional_classes = use_conditional_classes

        # self.channels = channels
        # self.time_sampling = time_sampling
        # self.exclude_sensors = exclude_sensors
        # self.diff_inputs = diff_inputs
        # self.match_histograms = match_histograms
        # self.resample_invalid_frames = resample_invalid_frames
        # self.upweight_centers = upweight_centers
        # self.normalize_perframe = normalize_perframe
        # self.true_multimodal = true_multimodal
        # self.use_centered_positives = use_centered_positives
        # self.use_grid_positives = use_grid_positives
        # self.temporal_dropout = temporal_dropout

        # TODO: reduce redundency between this, the argparse args piece
        self.common_dataset_kwargs = dict(
            channels=channels,
            time_sampling=time_sampling,
            diff_inputs=diff_inputs,
            exclude_sensors=exclude_sensors,
            match_histograms=match_histograms,
            upweight_centers=upweight_centers,
            resample_invalid_frames=resample_invalid_frames,
            normalize_perframe=normalize_perframe,
            true_multimodal=true_multimodal,
            use_centered_positives=use_centered_positives,
            use_grid_positives=use_grid_positives,
            temporal_dropout=temporal_dropout,
            max_epoch_length=max_epoch_length,
            use_conditional_classes=use_conditional_classes,
            ignore_dilate=ignore_dilate,
            min_spacetime_weight=min_spacetime_weight,
            dist_weights=dist_weights,
            use_cloudmask=use_cloudmask,
        )
        for _k, _v in self.common_dataset_kwargs.items():
            setattr(self, _k, _v)

        self.num_workers = util_globals.coerce_num_workers(num_workers)
        self.torch_start_method = torch_start_method
        self.torch_sharing_strategy = torch_sharing_strategy

        self.dataset_stats = None

        # will only correspond to train
        self.classes = None
        self.input_channels = None

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
            print('self.chip_size = {!r}'.format(self.chip_size))
            print('self.channels = {!r}'.format(self.channels))

    @classmethod
    def add_argparse_args(cls, parent_parser):
        """
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
            >>> assert args.exclude_sensors == ['l8', 'f3']
            >>> args, _ = parent_parser.parse_known_args(['--exclude_sensors=l8'])
            >>> assert args.exclude_sensors == ['l8']
        """
        from scriptconfig.smartcast import smartcast
        from functools import partial
        parser = parent_parser.add_argument_group('kwcoco_video_data')
        parser.add_argument('--train_dataset', default=None, help='path to the train kwcoco file')
        parser.add_argument('--vali_dataset', default=None, help='path to the validation kwcoco file')
        parser.add_argument('--test_dataset', default=None, help='path to the test kwcoco file')
        parser.add_argument('--time_steps', default=2, type=smartcast)
        parser.add_argument('--chip_size', default=128, type=smartcast)
        parser.add_argument('--time_overlap', default=0.0, type=smartcast, help='fraction of time steps to overlap')
        parser.add_argument('--chip_overlap', default=0.1, type=smartcast, help='fraction of space steps to overlap')
        parser.add_argument('--neg_to_pos_ratio', default=1.0, type=float, help='maximum ratio of samples with no annotations to samples with annots')
        parser.add_argument('--time_sampling', default='contiguous', type=str, help=ub.paragraph(
            '''
            Strategy for expanding the time window across non-contiguous frames.
            Can be auto, contiguous, hard+distribute, or dilate_affinity
            '''))
        parser.add_argument('--exclude_sensors', type=partial(smartcast, astype=list), help='comma delimited list of sensors to avoid, such as S2 or L8')
        parser.add_argument('--channels', default=None, type=str, help='channels to use should be ChannelSpec coercable')
        parser.add_argument('--batch_size', default=4, type=int)
        parser.add_argument('--time_span', default='2y', type=str, help='how long a time window should roughly span by default')
        parser.add_argument('--resample_invalid_frames', default=True, help='if True, will attempt to resample any frame without valid data')
        parser.add_argument('--temporal_dropout', default=0.0, type=float, help='Drops frames in a fraction of training batches'),

        parser.add_argument('--max_epoch_length', default=None, type=smartcast, help='If specified, restricts number of steps per epoch'),

        parser.add_argument(
            '--normalize_inputs', default=True, type=smartcast, help=ub.paragraph(
                '''
                if True, computes the mean/std for this dataset on each mode
                so this can be passed to the model.
                '''))

        parser.add_argument(
            '--match_histograms', default=False, type=smartcast, help=ub.paragraph(
                '''
                undocumented
                '''))

        parser.add_argument(
            '--normalize_perframe', default=False, type=smartcast, help=ub.paragraph(
                '''
                undocumented
                '''))

        parser.add_argument(
            '--upweight_centers', default=True, type=smartcast, help=ub.paragraph(
                '''
                undocumented
                '''))
        parser.add_argument(
            '--diff_inputs', default=False, type=smartcast, help=ub.paragraph(
                '''
                if True, also includes a difference between consecutive frames
                in the inputs produced.
                '''))

        # Backend infastructure-based arguments
        parser.add_argument(
            '--num_workers', default=4, type=str, help=ub.paragraph(
                '''
                number of background workers. Can be auto or an avail
                expression. TODO: rename to data_workers?
                '''
            ))

        parser.add_argument(
            '--true_multimodal', default=True, type=smartcast, help=ub.paragraph(
                '''
                Enables new logic for sampling multimodal data.
                Old logic probably doesn't work anymore.
                '''))

        parser.add_argument(
            '--use_conditional_classes', default=True, type=smartcast, help=ub.paragraph(
                '''
                Include no-activity, post-construction in predictions when
                their conditions are met.
                '''))

        parser.add_argument(
            '--torch_sharing_strategy', default='default', help=ub.paragraph(
                '''
                Torch multiprocessing sharing strategy.
                Can be default, file_descriptor, file_system
                '''))

        parser.add_argument(
            '--torch_start_method', default='default', help=ub.paragraph(
                '''
                Torch multiprocessing sharing strategy.
                Can be fork, spawn, forkserver
                '''))

        parser.add_argument(
            '--use_centered_positives', default=False, type=smartcast, help=ub.paragraph('Use centers of annotations as window centers'))
        parser.add_argument(
            '--use_grid_positives', default=True, type=smartcast, help=ub.paragraph('Use annotation overlaps with grid as positives'))
        parser.add_argument(
            '--ignore_dilate', default=11, type=smartcast, help=ub.paragraph('Dilation applied to ignore masks.'))
        parser.add_argument(
            '--min_spacetime_weight', default=0.5, type=smartcast, help=ub.paragraph('Minimum space-time dilation weight'))

        parser.add_argument(
            '--dist_weights', default=0, type=smartcast, help=ub.paragraph('To use distance-transform based weights on annotations or not'))

        parser.add_argument(
            '--use_cloudmask', default=1, type=int, help=ub.paragraph('Allow the dataloader to use the cloud mask to skip frames'))

        return parent_parser

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
                train_data = str(train_data.expanduser())

            if self.verbose:
                print('Build train kwcoco dataset')
            train_coco_dset = watch.demo.coerce_kwcoco(train_data)
            self.coco_datasets['train'] = train_coco_dset

            print('self.exclude_sensors', self.exclude_sensors)
            coco_train_sampler = ndsampler.CocoSampler(train_coco_dset)
            train_dataset = KWCocoVideoDataset(
                coco_train_sampler,
                sample_shape=(self.time_steps, self.chip_size, self.chip_size),
                mode='fit',
                # window_overlap=(self.time_overlap, self.chip_overlap, self.chip_overlap),
                window_overlap=self.chip_overlap,  # FIXME
                neg_to_pos_ratio=self.neg_to_pos_ratio,
                **self.common_dataset_kwargs,
            )

            # Unfortunately lightning seems to only enable / disables
            # validation depending on the methods that are defined, so we are
            # not able to statically define them.
            self.classes = train_dataset.classes
            self.torch_datasets['train'] = train_dataset
            ub.inject_method(self, lambda self: self._make_dataloader('train', shuffle=True), 'train_dataloader')

            if self.input_channels is None:
                self.input_channels = train_dataset.input_channels

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
            # Note: also need for class weights
            if stats_params is not None:
                self.dataset_stats = train_dataset.cached_dataset_stats(**stats_params)

            if self.vali_kwcoco is not None:
                # Explicit validation dataset should be prefered
                vali_data = self.vali_kwcoco
                if isinstance(vali_data, pathlib.Path):
                    vali_data = str(vali_data.expanduser())
                if self.verbose:
                    print('Build validation kwcoco dataset')
                kwcoco_ds = watch.demo.coerce_kwcoco(vali_data)
                vali_coco_sampler = ndsampler.CocoSampler(kwcoco_ds)
                vali_dataset = KWCocoVideoDataset(
                    vali_coco_sampler,
                    sample_shape=(self.time_steps, self.chip_size, self.chip_size),
                    mode='vali',
                    window_overlap=0,
                    neg_to_pos_ratio=0,
                    **self.common_dataset_kwargs)
                self.torch_datasets['vali'] = vali_dataset
                ub.inject_method(self, lambda self: self._make_dataloader('vali', shuffle=False), 'val_dataloader')

        if stage == 'test' or stage is None:
            test_data = self.test_kwcoco
            if isinstance(test_data, pathlib.Path):
                test_data = str(test_data.expanduser())
            if self.verbose:
                print('Build test kwcoco dataset')
            test_coco_dset = watch.demo.coerce_kwcoco(test_data)
            self.coco_datasets['test'] = test_coco_dset
            test_coco_sampler = ndsampler.CocoSampler(test_coco_dset)
            self.torch_datasets['test'] = KWCocoVideoDataset(
                test_coco_sampler,
                sample_shape=(self.time_steps, self.chip_size, self.chip_size),
                window_overlap=self.chip_overlap,  # FIXME
                mode='test',
                **self.common_dataset_kwargs,
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
        >>> coco_dset = kwcoco.CocoDataset(coco_fpath)
        >>> sampler = ndsampler.CocoSampler(coco_dset)
        >>> self = KWCocoVideoDataset(
        >>>     sampler,
        >>>     sample_shape=(5, 128, 128),
        >>>     window_overlap=0,
        >>>     channels="blue|green|red|nir|swir16",
        >>>     neg_to_pos_ratio=0, time_sampling='auto', diff_inputs=1, mode='fit', match_histograms=0,
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

    def __init__(
        self,
        sampler,
        sample_shape,
        channels=None,
        mode='fit',
        window_overlap=0,
        neg_to_pos_ratio=1.0,
        time_sampling='auto',
        diff_inputs=False,
        time_span='2y',
        exclude_sensors=None,
        match_histograms=False,
        resample_invalid_frames=True,
        upweight_centers=True,
        normalize_perframe=False,
        true_multimodal=True,
        use_grid_positives=True,
        use_centered_positives=False,
        temporal_dropout=0.0,
        max_epoch_length=None,
        use_conditional_classes=True,
        ignore_dilate=11,
        min_spacetime_weight=0.5,
        dist_weights=False,
        use_cloudmask=1,
    ):
        self.use_cloudmask = use_cloudmask
        self.dist_weights = dist_weights
        self.match_histograms = match_histograms
        self.normalize_perframe = normalize_perframe
        self.resample_invalid_frames = resample_invalid_frames
        self.upweight_centers = upweight_centers
        self.temporal_dropout = temporal_dropout
        self.max_epoch_length = max_epoch_length
        self.use_conditional_classes = use_conditional_classes
        self.ignore_dilate = ignore_dilate
        self.min_spacetime_weight = min_spacetime_weight

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
        self.ignore_classes = set(heuristics.IGNORE_CLASSNAMES) & set(graph.nodes)
        self.undistinguished_classes = set(heuristics.UNDISTINGUISHED_CLASSES) & set(graph.nodes)
        self.classes = kwcoco.CategoryTree(graph)

        if channels is None:
            # Hack to use all channels in the first image.
            # (Does not handle heterogeneous channels yet)
            chan_info = kwcoco_extensions.coco_channel_stats(sampler.dset)
            # channels = ','.join(sorted(chan_info['chan_hist']))
            channels = chan_info['all_channels']
        channels = channel_spec.ChannelSpec.coerce(channels).normalize()

        if time_sampling == 'auto':
            time_sampling = 'hard+distribute'

        if mode == 'custom':
            new_sample_grid = None
            self.length = 1
        elif mode == 'test':
            # In test mode we have to sample everything for BAS
            # (TODO: for activity clf, we should only focus on candidate regions)
            new_sample_grid = sample_video_spacetime_targets(
                sampler.dset, window_dims=sample_shape,
                window_overlap=window_overlap,
                keepbound=True,
                use_annot_info=False,
                exclude_sensors=exclude_sensors,
                time_sampling=time_sampling,
                time_span=time_span,
            )
            self.length = len(new_sample_grid['targets'])
        else:
            negative_classes = (self.ignore_classes | self.background_classes)
            new_sample_grid = sample_video_spacetime_targets(
                sampler.dset, window_dims=sample_shape,
                window_overlap=window_overlap,
                negative_classes=negative_classes,
                keepbound=False,
                use_annot_info=True,
                exclude_sensors=exclude_sensors,
                time_sampling=time_sampling,
                time_span=time_span,
                use_centered_positives=use_centered_positives,
                use_grid_positives=use_grid_positives,
            )

            n_pos = len(new_sample_grid['positives_indexes'])
            n_neg = len(new_sample_grid['negatives_indexes'])

            max_neg = min(int(max(0, (neg_to_pos_ratio * n_pos))), n_neg)
            if n_neg > max_neg:
                print('restrict to max_neg = {!r}'.format(max_neg))

            # We have too many negatives, so we are going to "group" negatives
            # and when we select one we will really just randomly select from
            # within the pool
            if max_neg > 0:
                negative_pool = list(util_iter.chunks(new_sample_grid['negatives_indexes'], nchunks=max_neg))
                self.negative_pool = negative_pool
                neg_pool_chunksizes = set(map(len, self.negative_pool))
                print('neg_pool_chunksizes = {!r}'.format(neg_pool_chunksizes))
            else:
                self.negative_pool = []

            # This is in a per-iteration basis
            self.n_pos = n_pos
            self.n_neg = len(self.negative_pool)
            self.length = self.n_pos + self.n_neg

            if max_epoch_length is not None:
                self.length = min(self.length, max_epoch_length)

            # print('len(neg_pool) ' + str(len(self.negative_pool)))
            # print('self.n_pos = {!r}'.format(self.n_pos))
            # print('self.n_neg = {!r}'.format(self.n_neg))
            # print('self.length = {!r}'.format(self.length))
        self.new_sample_grid = new_sample_grid

        self.window_overlap = window_overlap

        bg_catname = ub.peek(sorted(self.background_classes))
        self.bg_idx = self.classes.node_to_idx[bg_catname]

        # bg_node = graph.nodes['background']
        # if 'color' not in bg_node:
        #     bg_node['color'] = (0., 0., 0.)
        utils.category_tree_ensure_color(self.classes)

        self.sample_shape = sample_shape
        self.channels = channels

        self.diff_inputs = diff_inputs

        self.special_inputs = {}

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

        self.true_multimodal = true_multimodal
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
                space_box = kwimage.Boxes.from_slice(tr_['space_slice'])
                w = space_box.width.ravel()[0]
                h = space_box.height.ravel()[0]
                # hack: this prevents us from assuming there is a target in the
                # window, but it lets us get the benefit of chip_overlap=0.5 while
                # still having it at 0 for faster epochs.
                aff = kwimage.Affine.coerce(offset=(
                    rng.randint(-w // 2.7, w // 2.7),
                    rng.randint(-h // 2.7, h // 2.7)))
                space_box = space_box.warp(aff).quantize()
                space_box = kwimage.Boxes.from_slice(tr_['space_slice']).warp(aff).quantize()

                # prevent shifting the target off the edge of the video
                snap_target = kwimage.Boxes([[0, 0, vid_width, vid_height]], 'ltrb')
                _boxes_snap_to_edges(space_box, snap_target)

                tr_['space_slice'] = space_box.astype(int).to_slices()[0]

            # Temporal augmentation
            if rng.rand() < temporal_augment_rate:
                # old_gids = tr_['gids']
                time_sampler = self.new_sample_grid['vidid_to_time_sampler'][vidid]
                valid_gids = self.new_sample_grid['vidid_to_valid_gids'][vidid]
                new_gids = list(ub.take(valid_gids, time_sampler.sample(tr_['main_idx'])))
                tr_['gids'] = new_gids

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

        return tr_

    @profile
    def __getitem__(self, index):
        """

        Ignore:
            >>> # xdoctest: +REQUIRES(env:DVC_DPATH)
            >>> # Debug issues seen in training
            >>> from watch.tasks.fusion.datamodules.kwcoco_video_data import *  # NOQA
            >>> import watch
            >>> import ndsampler
            >>> import kwcoco
            >>> dvc_dpath = watch.find_smart_dvc_dpath()
            >>> #coco_fpath = dvc_dpath / 'Drop2-Aligned-TA1-2022-01/data.kwcoco.json'
            >>> coco_fpath = dvc_dpath / 'Aligned-Drop3-TA1-2022-03-10/data_nowv_train.kwcoco.json'
            >>> coco_dset = kwcoco.CocoDataset(coco_fpath)
            >>> rng = kwarray.ensure_rng(0)
            >>> vidid = rng.choice(coco_dset.videos())
            >>> coco_dset = coco_dset.subset(coco_dset.images(vidid=vidid))
            >>> sampler = ndsampler.CocoSampler(coco_dset)
            >>> self = KWCocoVideoDataset(
            >>>     sampler,
            >>>     sample_shape=(5, 224, 224),
            >>>     window_overlap=0,
            >>>     #channels="ASI|MF_Norm|AF|EVI|red|green|blue|swir16|swir22|nir",
            >>>     channels="red|green|blue|nir",
            >>>     neg_to_pos_ratio=0, time_sampling='auto', diff_inputs=0, temporal_dropout=0.5,
            >>> )
            >>> self.requested_tasks['change'] = False
            >>> item = self[5]
            >>> canvas = self.draw_item(item, max_channels=10, overlay_on_image=0)
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
            >>> item = self[self.n_pos + 1]
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
            >>> channels = '|'.join(sorted(set(ub.flatten([c.channels.fuse().as_list() for c in coco_dset.images().coco_images]))))
            >>> #channels = '|'.join(sorted(set(ub.flatten([kwcoco.ChannelSpec.coerce(c).fuse().as_list() for c in groups.keys()]))))
            >>> self = KWCocoVideoDataset(sampler, sample_shape=(5, 256, 256), channels=channels, normalize_perframe=False, true_multimodal=True)
            >>> self.disable_augmenter = True
            >>> index = 0
            >>> item = self[index]
            >>> canvas = self.draw_item(item)
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
        """

        if isinstance(index, dict):
            tr = index
            index = 'given-as-dictionary'
        else:
            if self.mode == 'test':
                tr = self.new_sample_grid['targets'][index]
            else:
                # Hack: we will make all of the first indexes positives
                # in the non-shuffled case. A negative index will randomly get
                # assigned a real negative target from its "group"

                # TODO: we can generalize this into generic pools
                # that happend to correspond to positive / negative or any
                # other distribution of examples we want
                if index < self.n_pos:
                    tr_idx = self.new_sample_grid['positives_indexes'][index]
                else:
                    neg_chunk = self.negative_pool[self.n_pos - index]
                    tr_idx = random.choice(neg_chunk)
                tr = self.new_sample_grid['targets'][tr_idx]

        tr_ = tr.copy()

        # get positive sample definition
        # collect sample
        sampler = self.sampler
        coco_dset = self.sampler.dset
        tr_['as_xarray'] = False
        tr_['use_experimental_loader'] = 1

        tr_ = self._augment_spacetime_target(tr_)

        if self.channels:
            tr_['channels'] = self.sample_channels

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

        def sample_one_frame(gid):
            coco_img = coco_dset.coco_image(gid)
            sensor_channels = (self.sample_channels & coco_img.channels).normalize()
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
                        print('cloud_frac = {!r}'.format(cloud_frac))
                        force_bad = True
                        # valid_cloud_vals = cloud_im[np.isnan(cloud_im)]

                    # if 0:
                    #     obj = coco_img.find_asset_obj('cloudmask')
                    #     fpath = ub.Path(coco_img.bundle_dpath) / obj['file_name']

                    # Skip if more then 50% cloudy

            for stream in sensor_channels.streams():
                if force_bad:
                    break
                tr_frame['channels'] = stream
                # TODO: FIXME: Use the correct nodata value here!
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
                            band_metas = coco_img.find_asset_obj('red').get('band_metas', [{}])
                            nodata_vals = [m.get('nodata', None) for m in band_metas]
                            # TODO: could be more careful about what band metas
                            # we are looking at. Assuming they are all the same
                            # here. The idea is only do this hack if the nodata
                            # value is not set (like in L1 data, but dont do it
                            # when the) values are set (like in TA1 data)
                            if any(v is None for v in nodata_vals):
                                mask = (sample['im'] == 0)
                                sample['im'][mask] = np.nan

                # dont ask for annotations multiple times
                invalid_mask = np.isnan(sample['im'])

                any_invalid = np.any(invalid_mask)
                none_invalid = not any_invalid
                if none_invalid:
                    all_invalid = False
                    # some_invalid = False
                else:
                    all_invalid = np.all(invalid_mask)
                    # some_invalid = not all_invalid and any_invalid

                if any_invalid:
                    sample['invalid_mask'] = invalid_mask
                else:
                    sample['invalid_mask'] = None

                if not all_invalid:
                    sample_streams[stream.spec] = sample
                    if 'annots' in sample:
                        first_with_annot = False
                else:
                    # HACK: if the red channel is all bad, discard the frame
                    # This can be removed once nodata is correctly propogated
                    # in the team features. OR we can add a feature where we
                    # keep track of an image wide observation mask and use that
                    # instead of using red as a proxy for it.
                    if 'red' in set(stream):
                        force_bad = True

                # TODO: mark frame as invalid when a red band is all 0
                # We are going to try to generalize this with a concept of an
                # "iffy" mask with will flag pixels that are minimum, zero, or
                # nan.
                RGB_IFFY_HACK = 1
                if RGB_IFFY_HACK and set(stream).issubset({'blue', 'green', 'red'}):
                    import warnings
                    with warnings.catch_warnings():
                        warnings.filterwarnings('ignore', 'empty slice')
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
                                force_bad = True

            gid_to_isbad[gid] = force_bad or len(sample_streams) == 0
            gid_to_sample[gid] = sample_streams

        for gid in tr_['gids']:
            pass
            sample_one_frame(gid)

        if 'video_id' not in tr_:
            arbitrary_sample = ub.peek(ub.peek(gid_to_sample.values()).values())
            tr_['video_id'] = arbitrary_sample['tr']['vidid']

        vidid = tr_['video_id']
        video = coco_dset.index.videos[vidid]
        time_sampler = self.new_sample_grid['vidid_to_time_sampler'][vidid]
        video_gids = time_sampler.video_gids

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
                        sample_one_frame(gid)

        good_gids = [gid for gid, flag in gid_to_isbad.items() if not flag]
        if len(good_gids) == 0:
            # Force at least a few to be "good"
            for gid in tr['gids']:
                gid_to_isbad[gid] = False
            good_gids = [gid for gid, flag in gid_to_isbad.items() if not flag]

        final_gids = ub.oset(video_gids) & good_gids
        # requested_channel_order = self.input_channels.spec.split('|')
        num_frames = len(final_gids)
        if num_frames == 0:
            raise Exception('0 frames')

        # coco_dset.images(final_gids).lookup('date_captured')
        tr_['gids'] = final_gids

        if self.sample_shape is None:
            # Do something better
            input_dsize = ub.peek(gid_to_sample[final_gids[0]])['im'].shape[1:3][::-1]
        else:
            input_dsize = self.sample_shape[-2:][::-1]

        if not self.inference_only:
            # Learn more from the center of the space-time patch
            time_weights = kwimage.gaussian_patch((1, num_frames))[0]
            time_weights = time_weights / time_weights.max()
            time_weights = time_weights.clip(0, 1)
            time_weights = np.maximum(time_weights, self.min_spacetime_weight)
            space_weights = util_kwimage.upweight_center_mask(input_dsize[::-1])
            space_weights = np.maximum(space_weights, self.min_spacetime_weight)

        if 1:
            # Replace nans with windows stats
            # TODO: handle nans outside of the dataloader
            # The dataloader **should** return nan values, it is up to the
            # method to handle them. So we have to fix RunningStats
            for gid in final_gids:
                stream_sample = gid_to_sample[gid]
                for sample in stream_sample.values():
                    im = sample['im']
                    mask = np.isnan(im)
                    if np.any(mask):
                        if np.all(mask):
                            im[:] = 0
                        else:
                            # TODO: Should use the global stream mean/std for this
                            # If that is not available, use in-window means
                            window_chan_med = np.nanmedian(im, axis=(0, 1, 2))
                            window_chan_mean = np.nanmean(im, axis=(0, 1, 2))
                            window_chan_ave = (window_chan_med + window_chan_mean) / 2
                            im[mask.any(axis=3), :] = window_chan_ave[None, None, None, :]

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
                        Did not sample correctly. Please send this info to Jon:
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
                # catch nans
                frame_hwc[np.isnan(frame_hwc)] = -1.
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

            sensor = img.get('sensor_coarse', '')

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
                    nodata_total = 0
                    for mask in mode_invalid_masks.values():
                        if mask is None:
                            nodata_total += 0
                        else:
                            if len(mask.shape) == 3:
                                nodata_total += (mask.sum(axis=2) / mask.shape[2])
                            else:
                                nodata_total += mask
                    # nodata_total = np.add.reduce([0 if mask is None else mask.sum(axis=2) / mask.shape[2] for mask in mode_invalid_masks.values()])
                    total_bands = len(mode_invalid_masks)
                    nodata_frac = nodata_total / total_bands
                    nodata_weight = 1 - nodata_frac
                    frame_weights = frame_weights * nodata_weight

                # Dilate ignore masks (dont care about the surrounding area
                # either)
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

        FLIP_AUGMENTATION = (not self.disable_augmenter and self.mode == 'fit')
        if FLIP_AUGMENTATION:
            # TODO: make a nice "augmenter" pipeline
            rng = kwarray.ensure_rng(None)
            do_hflip = rng.rand() > 0.5
            do_vflip = rng.rand() > 0.5
            flip_axis = []
            # Space dims are last at this point in the pipeline
            if do_vflip:
                flip_axis += [-2]
            if do_hflip:
                flip_axis += [-1]
            flip_axis = tuple(flip_axis)

            for frame_item in frame_items:
                frame_modes = frame_item['modes']
                for mode_key in list(frame_modes.keys()):
                    mode_data = frame_modes[mode_key]
                    frame_modes[mode_key] = np.flip(mode_data, axis=flip_axis)
                for key in truth_keys:
                    data = frame_item.get(key, None)
                    if data is not None:
                        frame_item[key] = np.flip(data, axis=flip_axis)

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

            @ub.memoize
            def _string_to_hashvec(key):
                # Maybe this should be a model responsibility.
                # I dont like defining the positional encoding in the dataset
                key_hash = ub.hash_data(key, base=16, hasher='blake3').encode()
                key_tensor = np.frombuffer(memoryview(key_hash), dtype=np.int32).astype(np.float32)
                key_tensor = key_tensor / np.linalg.norm(key_tensor)
                return key_tensor

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
        # tr_subset = ub.dict_isect(sample['tr'], {
        #     'gids', 'space_slice', 'vidid',
        # })
        tr_subset = ub.dict_isect(tr_, {
            'gids', 'space_slice', 'vidid',
        })
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
            ('channels', self.input_channels.__json__()),
            # ('sample_shape', self.sample_shape),
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
                              with_intensity=True, with_class=True):
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
            >>> sample_shape = (3, 96, 96)
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

        # TODO: we should ensure instance level frequency data as well
        # as pixel level frequency data.

        # TODO: we should ensure we include at least one sample from each type
        # of modality.
        # Note: the requested order of the channels could be different that
        # what is registered in the dataset. Need to find a good way to account
        # for this.

        prog = ub.ProgIter(loader, desc='estimate dataset stats')
        for batch_items in prog:
            for item in batch_items:
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

        # Make a list of all unique modes in the dataset.
        unique_sensor_modes = set(sensor_mode_hist.keys())

        if True:
            # This looks at the entire dataset, might want to
            # make a better way of getting this info.
            # self.sampler.dset.videos().images
            coco_images = self.sampler.dset.images().coco_images
            hacked = set()
            for c in coco_images:
                sspec = c.img.get('sensor_coarse', '')
                # Ensure channels are returned in requested order
                cspec = (self.input_channels & c.channels.fuse().normalize()).fuse().normalize().spec
                if cspec:
                    hacked.add((sspec, cspec))
            unique_sensor_modes.update(hacked)

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
            >>> # Calculate the probability of change for each frame
            >>> item_output = {}
            >>> change_prob_list = []
            >>> for _ in range(1, sample_shape[0]):
            >>>     change_prob = kwimage.Heatmap.random(
            >>>         dims=sample_shape[1:3], classes=1).data['class_probs'][0]
            >>>     change_prob_list += [change_prob]
            >>> change_probs = np.stack(change_prob_list)
            >>> item_output['change_probs'] = change_probs  # first frame does not have change
            >>> #
            >>> # Probability of each class for each frame
            >>> class_prob_list = []
            >>> for _ in range(0, sample_shape[0]):
            >>>     class_prob = kwimage.Heatmap.random(
            >>>         dims=sample_shape[1:3], classes=list(sampler.classes)).data['class_probs']
            >>>     class_prob_list += [einops.rearrange(class_prob, 'c h w -> h w c')]
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


class BatchVisualizationBuilder:
    """
    Helper object to build a batch visualization.

    The basic logic is that we will build a column for each timestep and then
    arrange them from left to right to show how the scene changes over time.
    Each column will be made of "cells" which could show either the truth, a
    prediction, loss weights, or raw input channels.

    Example:
        >>> from watch.tasks.fusion.datamodules.kwcoco_video_data import *  # NOQA
        >>> import ndsampler
        >>> import watch
        >>> coco_dset = watch.coerce_kwcoco('vidshapes2-watch', num_frames=5)
        >>> sampler = ndsampler.CocoSampler(coco_dset)
        >>> channels = 'r|g|b,B10|B8a|B1|B8|B11,X.2|Y.2'
        >>> combinable_extra = [['B10', 'B8', 'B8a']]  # special behavior
        >>> # combinable_extra = None  # uncomment for raw behavior
        >>> sample_shape = (5, 530, 610)
        >>> self = KWCocoVideoDataset(sampler, sample_shape=sample_shape, channels=channels, use_centered_positives=True, neg_to_pos_ratio=0)
        >>> index = len(self) // 4
        >>> item = self[index]
        >>> # Calculate the probability of change for each frame
        >>> item_output = {}
        >>> change_prob_list = []
        >>> for _ in range(1, sample_shape[0]):
        >>>     change_prob = kwimage.Heatmap.random(
        >>>         dims=sample_shape[1:3], classes=1).data['class_probs'][0]
        >>>     change_prob_list += [change_prob]
        >>> change_probs = np.stack(change_prob_list)
        >>> item_output['change_probs'] = change_probs  # first frame does not have change
        >>> #
        >>> # Probability of each class for each frame
        >>> class_prob_list = []
        >>> for _ in range(0, sample_shape[0]):
        >>>     class_prob = kwimage.Heatmap.random(
        >>>         dims=sample_shape[1:3], classes=list(sampler.classes)).data['class_probs']
        >>>     class_prob_list += [einops.rearrange(class_prob, 'c h w -> h w c')]
        >>> class_probs = np.stack(class_prob_list)
        >>> item_output['class_probs'] = class_probs  # first frame does not have change
        >>> #
        >>> # Probability of "saliency" (i.e. non-background) for each frame
        >>> saliency_prob_list = []
        >>> for _ in range(0, sample_shape[0]):
        >>>     saliency_prob = kwimage.Heatmap.random(
        >>>         dims=sample_shape[1:3], classes=1).data['class_probs']
        >>>     saliency_prob_list += [einops.rearrange(saliency_prob, 'c h w -> h w c')]
        >>> saliency_probs = np.stack(saliency_prob_list)
        >>> item_output['saliency_probs'] = saliency_probs
        >>> #binprobs[0][:] = 0  # first change prob should be all zeros
        >>> builder = BatchVisualizationBuilder(
        >>>     item, item_output, classes=self.classes, requested_tasks=self.requested_tasks,
        >>>     default_combinable_channels=self.default_combinable_channels, combinable_extra=combinable_extra)
        >>> #builder.overlay_on_image = 1
        >>> #canvas = builder.build()
        >>> builder.max_channels = 3
        >>> builder.overlay_on_image = 0
        >>> canvas2 = builder.build()
        >>> # xdoctest: +REQUIRES(--show)
        >>> import kwplot
        >>> kwplot.autompl()
        >>> #kwplot.imshow(canvas, fnum=1, pnum=(1, 2, 1))
        >>> #kwplot.imshow(canvas2, fnum=1, pnum=(1, 2, 2))
        >>> kwplot.imshow(canvas2, fnum=1, doclf=True)
        >>> kwplot.show_if_requested()
    """

    def __init__(builder, item, item_output=None, combinable_extra=None,
                 max_channels=5, max_dim=224, norm_over_time=0,
                 overlay_on_image=False, draw_weights=True, classes=None,
                 default_combinable_channels=None,
                 requested_tasks=None):
        builder.max_channels = max_channels
        builder.max_dim = max_dim
        builder.norm_over_time = norm_over_time
        builder.combinable_extra = combinable_extra
        builder.item_output = item_output
        builder.item = item
        builder.overlay_on_image = overlay_on_image
        builder.draw_weights = draw_weights
        builder.requested_tasks = requested_tasks

        builder.classes = classes
        builder.default_combinable_channels = default_combinable_channels

        combinable_channels = default_combinable_channels
        if combinable_extra is not None:
            combinable_channels = combinable_channels.copy()
            combinable_channels += list(map(ub.oset, combinable_extra))
        builder.combinable_channels = combinable_channels
        # print('builder.combinable_channels = {}'.format(ub.repr2(builder.combinable_channels, nl=1)))

    def build(builder):
        frame_metas = builder._prepare_frame_metadata()
        if 0:
            for idx, frame_meta in enumerate(frame_metas):
                print('---')
                print('idx = {!r}'.format(idx))
                frame_weight_shape = ub.map_vals(lambda x: x.shape, frame_meta['frame_weight'])
                print('frame_weight_shape = {}'.format(ub.repr2(frame_weight_shape, nl=1)))
                frame_meta['frame_weight']
        canvas = builder._build_canvas(frame_metas)
        return canvas

    def _prepare_frame_metadata(builder):
        import more_itertools
        item = builder.item
        combinable_channels = builder.combinable_channels

        truth_keys = []
        weight_keys = []
        if builder.requested_tasks['class']:
            truth_keys.append('class_idxs')
            weight_keys.append('class_weights')
        if builder.requested_tasks['saliency']:
            truth_keys.append('saliency')
            weight_keys.append('saliency_weights')
        if builder.requested_tasks['change']:
            truth_keys.append('change')
            weight_keys.append('change_weights')

        # Prepare metadata on each frame
        frame_metas = []
        for frame_idx, frame_item in enumerate(item['frames']):
            # Gather ground truth rasters
            frame_truth = {}
            for truth_key in truth_keys:
                truth_data = frame_item[truth_key]
                if truth_data is not None:
                    truth_data = truth_data.data.cpu().numpy()
                    frame_truth[truth_key] = truth_data

            frame_weight = {}
            for weight_key in weight_keys:
                weight_data = frame_item[weight_key]
                if weight_data is not None:
                    weight_data = weight_data.data.cpu().numpy()
                    frame_weight[weight_key] = weight_data
                else:
                    # HACK so saliency weights align correctly
                    frame_weight[weight_key] = None
                    # np.full((2, 2), fill_value=np.nan)

            # Breakup all of the modes into 1-channel per array
            frame_chan_names = []
            frame_chan_datas = []
            frame_modes = frame_item['modes']
            for mode_code, mode_data in frame_modes.items():
                mode_data = mode_data.data.cpu().numpy()
                code_list = kwcoco.FusedChannelSpec.coerce(mode_code).normalize().as_list()
                for chan_data, chan_name in zip(mode_data, code_list):
                    frame_chan_names.append(chan_name)
                    frame_chan_datas.append(chan_data)
            full_mode_code = ','.join(list(frame_item['modes'].keys()))

            # Determine what single and combinable channels exist per stream
            perstream_available = []
            for mode_code in frame_modes.keys():
                code_list = kwcoco.FusedChannelSpec.coerce(mode_code).normalize().as_list()
                code_set = ub.oset(code_list)
                stream_combinables = []
                for combinable in combinable_channels:
                    if combinable.issubset(code_set):
                        stream_combinables.append(combinable)
                remain = code_set - set(ub.flatten(stream_combinables))
                stream_singletons = [(c,) for c in remain]
                # Prioritize combinable channels in each stream first
                stream_available = list(map(tuple, stream_combinables)) + stream_singletons
                perstream_available.append(stream_available)

            # Prioritize choosing a balance of channels from each stream
            frame_available_chans = list(more_itertools.roundrobin(*perstream_available))

            frame_meta = {
                'full_mode_code': full_mode_code,
                'frame_idx': frame_idx,
                'frame_item': frame_item,
                'frame_chan_names': frame_chan_names,
                'frame_chan_datas': frame_chan_datas,
                'frame_available_chans': frame_available_chans,
                'frame_truth': frame_truth,
                'frame_weight': frame_weight,
                'sensor': frame_item.get('sensor', ''),
            }
            frame_metas.append(frame_meta)

        # Determine which frames to visualize For each frame choose N channels
        # such that common channels are aligned, visualize common channels in
        # the first rows and then fill with whatever is left
        # chan_freq = ub.dict_hist(ub.flatten(frame_meta['frame_available_chans']
        #                                     for frame_meta in frame_metas))
        # chan_priority = {k: (v, len(k), -idx) for idx, (k, v)
        #                  in enumerate(chan_freq.items())}
        for frame_meta in frame_metas:
            chan_keys = frame_meta['frame_available_chans']
            # print('chan_keys = {!r}'.format(chan_keys))
            # frame_priority = ub.dict_isect(chan_priority, chan_keys)
            # chosen = ub.argsort(frame_priority, reverse=True)[0:builder.max_channels]
            # print('chosen = {!r}'.format(chosen))
            chosen = chan_keys[0:builder.max_channels]
            frame_meta['chans_to_use'] = chosen

        # Gather channels to visualize
        for frame_meta in frame_metas:
            chans_to_use = frame_meta['chans_to_use']
            frame_chan_names = frame_meta['frame_chan_names']
            frame_chan_datas = frame_meta['frame_chan_datas']
            chan_idx_lut = {name: idx for idx, name in enumerate(frame_chan_names)}
            # Prepare and normalize the channels for visualization
            chan_rows = []
            for chan_names in chans_to_use:
                chan_code = '|'.join(chan_names)
                chanxs = list(ub.take(chan_idx_lut, chan_names))
                parts = list(ub.take(frame_chan_datas, chanxs))
                raw_signal = np.stack(parts, axis=2)
                row = {
                    'raw_signal': raw_signal,
                    'chan_code': chan_code,
                    'signal_text': f'{chan_code}',
                    'sensor': frame_meta['sensor'],
                }
                chan_rows.append(row)
            frame_meta['chan_rows'] = chan_rows
            assert len(chan_rows) > 0, 'no channels to draw on'

        if builder.draw_weights:
            # Normalize weights for visualization
            all_weight_overlays = []
            for frame_meta in frame_metas:
                frame_meta['weight_overlays'] = {}
                for weight_key, weight_data in frame_meta['frame_weight'].items():
                    overlay_row = {
                        'weight_key': weight_key,
                        'raw': weight_data,
                    }
                    frame_meta['weight_overlays'][weight_key] = overlay_row
                    all_weight_overlays.append(overlay_row)

            for weight_key, group in ub.group_items(all_weight_overlays, lambda x: x['weight_key']).items():
                # print('weight_key = {!r}'.format(weight_key))
                # maxval = -float('inf')
                # minval = float('inf')
                # for cell in group:
                #     maxval = max(maxval, cell['raw'].max())
                #     minval = min(minval, cell['raw'].min())
                # print('maxval = {!r}'.format(maxval))
                # print('minval = {!r}'.format(minval))
                for cell in group:
                    weight_data = cell['raw']
                    if weight_data is None:
                        h = w = builder.max_dim
                        weight_overlay = kwimage.draw_text_on_image(
                            {'width': w, 'height': h}, 'X', org=(w // 2, h // 2),
                            valign='center', halign='center', fontScale=10,
                            color='red')
                        weight_overlay = kwimage.ensure_float01(weight_overlay)
                    else:
                        weight_overlay = kwimage.atleast_3channels(weight_data)
                    # weight_overlay = kwimage.ensure_alpha_channel(weight_overlay)
                    # weight_overlay[:, 3] = 0.5
                    cell['overlay'] = weight_overlay

        # Normalize raw signal into visualizable range
        if builder.norm_over_time:
            # Normalize all cells with the same channel code across time
            channel_cells = [cell for frame_meta in frame_metas for cell in frame_meta['chan_rows']]
            # chan_to_cells = ub.group_items(channel_cells, lambda c: (c['chan_code'])
            chan_to_cells = ub.group_items(channel_cells, lambda c: (c['chan_code'], c['sensor']))
            for chan_code, cells in chan_to_cells.items():
                flat = [c['raw_signal'].ravel() for c in cells]
                cums = np.cumsum(list(map(len, flat)))
                combo = np.hstack(flat)
                try:
                    combo_normed = kwimage.normalize_intensity(combo, nodata=0).copy()
                except Exception:
                    combo_normed = combo.copy()
                flat_normed = np.split(combo_normed, cums)
                for cell, flat_item in zip(cells, flat_normed):
                    norm_signal = flat_item.reshape(*cell['raw_signal'].shape)
                    norm_signal = kwimage.atleast_3channels(norm_signal)
                    norm_signal = np.nan_to_num(norm_signal)
                    cell['norm_signal'] = norm_signal
        else:
            # Normalize each timestep by itself
            for frame_meta in frame_metas:
                for row in frame_meta['chan_rows']:
                    raw_signal = row['raw_signal']
                    needs_norm = np.nanmin(raw_signal) < 0 or np.nanmax(raw_signal) > 1
                    if needs_norm:
                        try:
                            norm_signal = kwimage.normalize_intensity(raw_signal).copy()
                        except Exception:
                            norm_signal = raw_signal.copy()
                    else:
                        norm_signal = raw_signal.copy()
                    norm_signal = np.nan_to_num(norm_signal)
                    norm_signal = util_kwimage.ensure_false_color(norm_signal)
                    norm_signal = kwimage.atleast_3channels(norm_signal)
                    row['norm_signal'] = norm_signal

        return frame_metas

    def _build_canvas(builder, frame_metas):

        # Given prepared frame metadata, build a vertical stack of per-chanel
        # information, and then horizontally stack the timesteps.
        horizontal_stack = []

        truth_overlay_keys = set(ub.flatten([m['frame_truth'] for m in frame_metas]))
        weight_overlay_keys = set(ub.flatten([m['frame_weight'] for m in frame_metas]))

        for frame_meta in frame_metas:
            frame_canvas = builder._build_frame(
                frame_meta, truth_overlay_keys, weight_overlay_keys)
            horizontal_stack.append(frame_canvas)

        body_canvas = kwimage.stack_images(horizontal_stack, axis=1, pad=5)
        body_canvas = body_canvas[..., 0:3]  # drop alpha
        body_canvas = kwimage.ensure_uint255(body_canvas)  # convert to uint8

        width = body_canvas.shape[1]

        vid_text = f'video: {builder.item["video_id"]} - {builder.item["video_name"]}'
        vid_header = kwimage.draw_text_on_image(
            {'width': width}, vid_text, org=(width // 2, 3), valign='top',
            halign='center', color='pink')

        canvas = kwimage.stack_images([vid_header, body_canvas], axis=0, pad=3)
        return canvas

    def _build_frame_header(builder, frame_meta):
        """
        Make the text header for each timestep (frame)
        """
        header_stack = []

        frame_item = frame_meta['frame_item']
        frame_idx = frame_meta['frame_idx']
        gid = frame_item['gid']

        # Build column headers
        header_dims = {'width': builder.max_dim}
        header_part = util_kwimage.draw_header_text(
            image=header_dims, fit=False,
            text=f't={frame_idx} gid={gid}', color='salmon')
        header_stack.append(header_part)

        sensor = frame_item.get('sensor', '')
        if sensor:
            header_part = util_kwimage.draw_header_text(
                image=header_dims, fit=False, text=f'{sensor}',
                color='salmon')
            header_stack.append(header_part)

        date_captured = frame_item.get('date_captured', '')
        if date_captured:
            header_part = util_kwimage.draw_header_text(
                header_dims, fit='shrink', text=f'{date_captured}',
                color='salmon')
            header_stack.append(header_part)
        return header_stack

    def _build_frame(builder, frame_meta, truth_overlay_keys, weight_overlay_keys):
        """
        Build a vertical stack for a single frame
        """
        classes = builder.classes
        item_output = builder.item_output

        vertical_stack = []

        frame_idx = frame_meta['frame_idx']
        chan_rows = frame_meta['chan_rows']

        frame_truth = frame_meta['frame_truth']
        # frame_weight = frame_meta['frame_weight']

        # Build column headers
        header_stack = builder._build_frame_header(frame_meta)
        vertical_stack.extend(header_stack)

        # Build truth / metadata overlays
        overlay_shape = ub.peek(frame_truth.values()).shape[0:2]

        # Create overlays for training objective targets
        overlay_items = []

        # Create the the true class label overlay
        overlay_key = 'class_idxs'
        if overlay_key in truth_overlay_keys and builder.requested_tasks['class']:
            class_idxs = frame_truth.get(overlay_key, None)
            true_heatmap = kwimage.Heatmap(class_idx=class_idxs, classes=classes)
            class_overlay = true_heatmap.colorize('class_idx')
            class_overlay[..., 3] = 0.5
            overlay_items.append({
                'overlay': class_overlay,
                'label_text': 'true class',
            })

        # Create the the true saliency label overlay
        overlay_key = 'saliency'
        if overlay_key in truth_overlay_keys and builder.requested_tasks['saliency']:
            saliency = frame_truth.get(overlay_key, None)
            if saliency is not None:
                if 1:
                    saliency_overlay = kwimage.make_heatmask(saliency.astype(np.float32), cmap='plasma').clip(0, 1)
                    saliency_overlay[..., 3] *= 0.5
                else:
                    saliency_overlay = np.zeros(saliency.shape + (4,), dtype=np.float32)
                    saliency_overlay = kwimage.Mask(saliency, format='c_mask').draw_on(saliency_overlay, color='dodgerblue')
                    saliency_overlay = kwimage.ensure_alpha_channel(saliency_overlay)
                    saliency_overlay[..., 3] = (saliency > 0).astype(np.float32) * 0.5
                overlay_items.append({
                    'overlay': saliency_overlay,
                    'label_text': 'true saliency',
                })

        # Create the true change label overlay
        overlay_key = 'change'
        if overlay_key in truth_overlay_keys and builder.requested_tasks['change']:
            change_overlay = np.zeros(overlay_shape + (4,), dtype=np.float32)
            changes = frame_truth.get(overlay_key, None)
            if changes is not None:
                if 1:
                    change_overlay = kwimage.make_heatmask(changes.astype(np.float32), cmap='viridis').clip(0, 1)
                    change_overlay[..., 3] *= 0.5
                else:
                    change_overlay = kwimage.Mask(changes, format='c_mask').draw_on(change_overlay, color='lime')
                    change_overlay = kwimage.ensure_alpha_channel(change_overlay)
                    change_overlay[..., 3] = (changes > 0).astype(np.float32) * 0.5
            overlay_items.append({
                'overlay': change_overlay,
                'label_text': 'true change',
            })

        weight_items = []
        if builder.draw_weights:
            weight_overlays = frame_meta['weight_overlays']
            for overlay_key in weight_overlay_keys:
                weight_overlay_info = weight_overlays.get(overlay_key, None)
                if weight_overlay_info is not None:
                    weight_items.append({
                        'overlay': weight_overlay_info['overlay'],
                        'label_text': overlay_key,
                    })

        resizekw = {
            'dsize': (builder.max_dim, builder.max_dim),
            # 'max_dim': builder.max_dim,
            # 'letterbox': False,
            'letterbox': True,
            'interpolation': 'nearest',
            # 'interpolation': 'linear',
        }

        # TODO: clean up logic
        key = 'class_probs'
        overlay_index = 0
        if item_output and key in item_output and builder.requested_tasks['class']:
            if builder.overlay_on_image:
                norm_signal = chan_rows[overlay_index]['norm_signal']
            else:
                norm_signal = np.zeros_like(chan_rows[min(overlay_index, len(chan_rows) - 1)]['norm_signal'])
            x = item_output[key][frame_idx]
            class_probs = einops.rearrange(x, 'h w c -> c h w')
            class_heatmap = kwimage.Heatmap(class_probs=class_probs, classes=classes)
            pred_part = class_heatmap.draw_on(norm_signal, with_alpha=0.7)
            # TODO: we might want to overlay the prediction on one or
            # all of the channels
            pred_part = kwimage.imresize(pred_part, **resizekw).clip(0, 1)
            pred_text = f'pred class t={frame_idx}'
            pred_part = kwimage.draw_text_on_image(
                pred_part, pred_text, (1, 1), valign='top',
                color='dodgerblue', border=3)
            vertical_stack.append(pred_part)

        key = 'saliency_probs'
        if item_output and  key in item_output and builder.requested_tasks['saliency']:
            if builder.overlay_on_image:
                norm_signal = chan_rows[0]['norm_signal']
            else:
                norm_signal = np.zeros_like(chan_rows[min(overlay_index, len(chan_rows) - 1)]['norm_signal'])
            x = item_output[key][frame_idx]
            saliency_probs = einops.rearrange(x, 'h w c -> c h w')
            # Hard coded index, dont like
            is_salient_probs = saliency_probs[1]
            # saliency_heatmap = kwimage.Heatmap(class_probs=saliency_probs)
            # pred_part = saliency_heatmap.draw_on(norm_signal, with_alpha=0.7)
            pred_part = kwimage.make_heatmask(is_salient_probs, cmap='plasma')
            pred_part[..., 3] = 0.7
            # TODO: we might want to overlay the prediction on one or
            # all of the channels
            pred_part = kwimage.imresize(pred_part, **resizekw).clip(0, 1)
            pred_text = f'pred saliency t={frame_idx}'
            pred_part = kwimage.draw_text_on_image(
                pred_part, pred_text, (1, 1), valign='top',
                color='dodgerblue', border=3)
            vertical_stack.append(pred_part)

        key = 'change_probs'
        overlay_index = 1
        if item_output and  key in item_output and builder.requested_tasks['change']:
            # Make a probability heatmap we can either display
            # independently or overlay on a rendered channel
            if frame_idx == 0:
                # BIG RED X
                # h, w = vertical_stack[-1].shape[0:2]
                h = w = builder.max_dim
                pred_mask = kwimage.draw_text_on_image(
                    {'width': w, 'height': h}, 'X', org=(w // 2, h // 2),
                    valign='center', halign='center', fontScale=10,
                    color='red')
                pred_part = pred_mask
            else:
                pred_raw = item_output[key][frame_idx - 1]
                # Draw predictions on the first item
                pred_mask = kwimage.make_heatmask(pred_raw, cmap='viridis')
                norm_signal = chan_rows[min(overlay_index, len(chan_rows) - 1)]['norm_signal']
                if builder.overlay_on_image:
                    norm_signal = norm_signal
                else:
                    norm_signal = np.zeros_like(norm_signal)
                pred_layers = [pred_mask, norm_signal]
                pred_part = kwimage.overlay_alpha_layers(pred_layers)
                # TODO: we might want to overlay the prediction on one or
                # all of the channels
                pred_part = kwimage.imresize(pred_part, **resizekw).clip(0, 1)
                pred_text = f'pred change t={frame_idx}'
                pred_part = kwimage.draw_text_on_image(
                    pred_part, pred_text, (1, 1), valign='top',
                    color='dodgerblue', border=3)
            vertical_stack.append(pred_part)

        if not builder.overlay_on_image:
            # FIXME: might be broken
            # Draw the overlays by themselves
            for overlay_info in overlay_items:
                label_text = overlay_info['label_text']
                row_canvas = overlay_info['overlay'][..., 0:3]
                row_canvas = kwimage.imresize(row_canvas, **resizekw).clip(0, 1)
                signal_bottom_y = 1  # hack: hardcoded
                row_canvas = kwimage.ensure_uint255(row_canvas)
                row_canvas = kwimage.draw_text_on_image(
                    row_canvas, label_text, (1, signal_bottom_y + 1),
                    valign='top', color='lime', border=3)
                vertical_stack.append(row_canvas)

        for overlay_info in weight_items:
            label_text = overlay_info['label_text']
            row_canvas = overlay_info['overlay'][..., 0:3]
            row_canvas = row_canvas.copy()
            row_canvas = kwimage.imresize(row_canvas, **resizekw).clip(0, 1)
            signal_bottom_y = 1  # hack: hardcoded
            row_canvas = kwimage.ensure_uint255(row_canvas)
            row_canvas = kwimage.draw_text_on_image(
                row_canvas, label_text, (1, signal_bottom_y + 1),
                valign='top', color='lime', border=3)
            vertical_stack.append(row_canvas)

        for iterx, row in enumerate(chan_rows):
            layers = []
            label_text = None
            if builder.overlay_on_image:
                # Draw truth on the image itself
                if iterx < len(overlay_items):
                    overlay_info = overlay_items[iterx]
                    layers.append(overlay_info['overlay'])
                    label_text = overlay_info['label_text']

            layers.append(row['norm_signal'])
            row_canvas = kwimage.overlay_alpha_layers(layers)[..., 0:3]

            row_canvas = kwimage.imresize(row_canvas, **resizekw).clip(0, 1)
            row_canvas = kwimage.ensure_uint255(row_canvas)
            row_canvas = kwimage.draw_text_on_image(
                row_canvas, row['signal_text'], (1, 1), valign='top',
                color='white', border=3)

            if label_text:
                # TODO: make draw_text_on_image able to return the
                # geometry of what it drew and use that.
                signal_bottom_y = 31  # hack: hardcoded
                row_canvas = kwimage.draw_text_on_image(
                    row_canvas, label_text, (1, signal_bottom_y + 1),
                    valign='top', color='lime', border=3)
            vertical_stack.append(row_canvas)

        vertical_stack = [kwimage.ensure_uint255(p) for p in vertical_stack]
        frame_canvas = kwimage.stack_images(vertical_stack, overlap=-3)
        return frame_canvas


def visualize_sample_grid(dset, sample_grid, max_vids=2, max_frames=6):
    """
    Debug visualization for sampling grid

    Draws multiple frames.

    Draws a yellow polygon over invalid spatial regions.

    Places a red dot where there is a negative sample (at the center of the negative window)

    Places a blue dot where there is a positive sample

    Notes:
        * Dots are more intense when there are more temporal coverage of that dot.

        * Dots are placed on the center of the window.
          They do not indicate its extent.

        * Dots are blue if they overlap any annotation in their temporal region
          so they may visually be near an annotation.

    Example:
        >>> from watch.tasks.fusion.datamodules.kwcoco_video_data import *  # NOQA
        >>> from watch.demo.smart_kwcoco_demodata import demo_kwcoco_multisensor
        >>> dset = coco_dset = demo_kwcoco_multisensor(dates=True, geodata=True, heatmap=True)
        >>> window_overlap = 0.0
        >>> window_dims = (3, 32, 32)
        >>> keepbound = False
        >>> time_sampling = 'soft2+distribute'
        >>> use_centered_positives = True
        >>> use_grid_positives = 0
        >>> sample_grid = sample_video_spacetime_targets(
        >>>     dset, window_dims, window_overlap, time_sampling=time_sampling,
        >>>     use_grid_positives=use_grid_positives, use_centered_positives=use_centered_positives)
        >>> # xdoctest: +REQUIRES(--show)
        >>> import kwplot
        >>> kwplot.autompl()
        >>> canvas = visualize_sample_grid(dset, sample_grid, max_vids=1,
        >>>                                max_frames=6)
        >>> kwplot.imshow(canvas, doclf=1)
        >>> kwplot.show_if_requested()

    Example:
        >>> # xdoctest: +REQUIRES(env:DVC_DPATH)
        >>> from watch.tasks.fusion.datamodules.kwcoco_video_data import *  # NOQA
        >>> import watch
        >>> # dset = coco_dset = demo_kwcoco_multisensor(dates=True, geodata=True, heatmap=True)
        >>> dvc_dpath = watch.find_smart_dvc_dpath()
        >>> #coco_fpath = dvc_dpath / 'Drop2-Aligned-TA1-2022-02-15/combo_DILM_train.kwcoco.json'
        >>> coco_fpath = dvc_dpath / 'Aligned-Drop3-TA1-2022-03-10/data_nowv_vali.kwcoco.json'
        >>> big_dset = kwcoco.CocoDataset(coco_fpath)
        >>> dset = big_dset.subset(big_dset.videos(names=['BR_R002']).images.lookup('id')[0])
        >>> window_overlap = 0.0
        >>> window_dims = (3, 32, 32)
        >>> keepbound = False
        >>> time_sampling = 'soft2+distribute'
        >>> use_centered_positives = True
        >>> use_grid_positives = 0
        >>> sample_grid = sample_video_spacetime_targets(
        >>>     dset, window_dims, window_overlap, time_sampling=time_sampling,
        >>>     use_grid_positives=True, use_centered_positives=use_centered_positives)
        >>> # xdoctest: +REQUIRES(--show)
        >>> import kwplot
        >>> kwplot.autompl()
        >>> canvas = visualize_sample_grid(dset, sample_grid, max_vids=1, max_frames=12)
        >>> kwplot.imshow(canvas, doclf=1)
        >>> kwplot.show_if_requested()
    """
    # Visualize the sample grid
    import pandas as pd
    targets = pd.DataFrame(sample_grid['targets'])

    dataset_canvases = []

    # max_vids = 2
    # max_frames = 6

    vidid_to_videodf = dict(list(targets.groupby('video_id')))

    orientation = 0

    for vidid, video_df in vidid_to_videodf.items():
        video = dset.index.videos[vidid]
        vidname = video['name']
        gid_to_infos = ub.ddict(list)
        for _, row in video_df.iterrows():
            for gid in row['gids']:
                gid_to_infos[gid].append({
                    'gid': gid,
                    'space_slice': row['space_slice'],
                    'label': row['label'],
                })

        video_canvases = []
        common = ub.oset(dset.images(vidid=vidid)) & (gid_to_infos)

        if True:
            # HACK: Use a temporal sampler once to get a nice overview of the
            # dataset in time.
            from dateutil import parser
            from watch.tasks.fusion.datamodules import temporal_sampling as tsm  # NOQA
            images = dset.images(common)
            datetimes = [None if date is None else parser.parse(date) for date in images.lookup('date_captured', None)]
            unixtimes = np.array([np.nan if dt is None else dt.timestamp() for dt in datetimes])
            sensors = images.lookup('sensor_coarse', None)
            time_sampler = tsm.TimeWindowSampler(
                unixtimes=unixtimes, sensors=sensors, time_window=max_frames,
                time_span='1y', affinity_type='hardish3',
                update_rule='distribute+pairwise')
            sample = time_sampler.sample()
            common = list(ub.take(common, sample))

        for gid in common:
            infos = gid_to_infos[gid]
            label_to_items = ub.group_items(infos, key=lambda x: x['label'])
            video = dset.index.videos[vidid]

            shape = (video['height'], video['width'], 4)
            canvas = np.zeros(shape, dtype=np.float32)
            shape = (2, video['height'], video['width'])
            accum = np.zeros(shape, dtype=np.float32)

            for label, items in label_to_items.items():
                label_idx = {'positive_grid': 1, 'positive_center': 1,
                             'negative_grid': 0}[label]
                for info in items:
                    space_slice = info['space_slice']
                    y_sl, x_sl = space_slice
                    # ww = x_sl.stop - x_sl.start
                    # wh = y_sl.stop - y_sl.start
                    ss = accum[(label_idx,) + space_slice].shape
                    if np.prod(ss) > 0:
                        vals = util_kwimage.upweight_center_mask(ss)
                        vals = np.maximum(vals, 0.1)
                        accum[(label_idx,) + space_slice] += vals
                        # Add extra weight to borders for viz
                        accum[label_idx, y_sl.start:y_sl.start + 1, x_sl.start: x_sl.stop] += 0.15
                        accum[label_idx, y_sl.stop - 1:y_sl.stop, x_sl.start:x_sl.stop] += 0.15
                        accum[label_idx, y_sl.start:y_sl.stop, x_sl.start: x_sl.start + 1] += 0.15
                        accum[label_idx, y_sl.start:y_sl.stop, x_sl.stop - 1: x_sl.stop] += 0.15

            neg_accum = accum[0]
            pos_accum = accum[1]

            neg_alpha = neg_accum / (neg_accum.max() * 2)
            pos_alpha = pos_accum / (pos_accum.max() * 2)
            bg_canvas = canvas.copy()
            bg_canvas[..., 0:4] = [0., 0., 0., 1.0]
            pos_canvas = canvas.copy()
            neg_canvas = canvas.copy()
            pos_canvas[..., 0:3] = kwimage.Color('dodgerblue').as01()
            neg_canvas[..., 0:3] = kwimage.Color('orangered').as01()
            neg_canvas[..., 3] = neg_alpha
            pos_canvas[..., 3] = pos_alpha
            neg_canvas = np.nan_to_num(neg_canvas)
            pos_canvas = np.nan_to_num(pos_canvas)

            warp_vid_from_img = kwimage.Affine.coerce(
                dset.index.imgs[gid]['warp_img_to_vid'])

            vid_poly = kwimage.Boxes([[0, 0, video['width'], video['height']]], 'xywh').to_polygons()[0]
            coco_poly = dset.index.imgs[gid].get('valid_region', None)
            if coco_poly is None:
                kw_invalid_poly = None
            else:
                kw_poly_img = kwimage.MultiPolygon.coerce(coco_poly)
                valid_poly = kw_poly_img.warp(warp_vid_from_img)
                sh_invalid_poly = vid_poly.to_shapely().difference(valid_poly.to_shapely())
                kw_invalid_poly = kwimage.MultiPolygon.coerce(sh_invalid_poly)

            final_canvas = kwimage.overlay_alpha_layers([pos_canvas, neg_canvas, bg_canvas])
            final_canvas = kwimage.ensure_uint255(final_canvas)

            annot_dets = dset.annots(gid=gid).detections
            vid_annot_dets = annot_dets.warp(warp_vid_from_img)

            if 1:
                final_canvas = vid_annot_dets.draw_on(final_canvas, color='white')

            if kw_invalid_poly is not None:
                final_canvas = kw_invalid_poly.draw_on(final_canvas, color='yellow', alpha=0.5)

            # from watch import heuristics
            img = dset.index.imgs[gid]
            header_lines = heuristics.build_image_header_text(
                img=img, vidname=vidname)
            header_text = '\n'.join(header_lines)

            final_canvas = kwimage.draw_header_text(final_canvas, header_text)
            video_canvases.append(final_canvas)

            if len(video_canvases) >= max_frames:
                break

        if max_vids == 1:
            video_canvas = kwimage.stack_images_grid(
                video_canvases, axis=orientation, pad=10)
        else:
            video_canvas = kwimage.stack_images(video_canvases, axis=1 - orientation, pad=10)
        dataset_canvases.append(video_canvas)
        if len(dataset_canvases) >= max_vids:
            break

    dataset_canvas = kwimage.stack_images(dataset_canvases, axis=orientation, pad=20)
    if 0:
        import kwplot
        kwplot.autompl()
        kwplot.imshow(dataset_canvas, doclf=1)
    return dataset_canvas


def sample_video_spacetime_targets(dset, window_dims, window_overlap=0.0,
                                   negative_classes=None, keepbound=False,
                                   exclude_sensors=None,
                                   time_sampling='hard+distribute',
                                   time_span='2y', use_annot_info=True,
                                   use_grid_positives=True,
                                   use_centered_positives=True):
    """
    This is the main driver that builds the sample grid.

    The basic idea is that you will slide a spacetime window over the dataset
    and mark where positive andnegative "windows" are. We also put windows
    directly on positive annotations if desired.

    See the above :func:`visualize_sample_grid` for a visualization of what the
    sample grid looks like.

    Ask jon about what the params mean if you need this.
    This code badly needs a refactor.

    Example:
        >>> # xdoctest: +REQUIRES(env:DVC_DPATH)
        >>> import os
        >>> from watch.tasks.fusion.datamodules.kwcoco_video_data import *  # NOQA
        >>> import watch
        >>> dvc_dpath = watch.find_smart_dvc_dpath()
        >>> coco_fpath = dvc_dpath / 'drop1-S2-L8-aligned/data.kwcoco.json'
        >>> dset = kwcoco.CocoDataset(coco_fpath)
        >>> window_overlap = 0.0
        >>> window_dims = (2, 128, 128)
        >>> keepbound = False
        >>> exclude_sensors = None
        >>> sample_grid = sample_video_spacetime_targets(dset, window_dims, window_overlap)
        >>> time_sampling = 'hard+distribute'
        >>> positives = list(ub.take(sample_grid['targets'], sample_grid['positives_indexes']))
        _ = xdev.profile_now(sample_video_spacetime_targets)(dset, window_dims, window_overlap)

        >>> import os
        >>> from watch.tasks.fusion.datamodules.kwcoco_video_data import *  # NOQA
        >>> import watch
        >>> dvc_dpath = watch.find_smart_dvc_dpath()
        >>> coco_fpath = dvc_dpath / 'Drop1-Aligned-L1/vali_data_wv.kwcoco.json'
        >>> dset = kwcoco.CocoDataset(coco_fpath)
        >>> window_overlap = 0.0
        >>> window_dims = (2, 128, 128)
        >>> keepbound = False
        >>> exclude_sensors = None
        >>> sample_grid = sample_video_spacetime_targets(dset, window_dims, window_overlap)
        >>> time_sampling = 'hard+distribute'
        >>> time_span = '2y'
        >>> use_annot_info = True
        >>> time_sampling = 'hard+distribute'
        >>> positives = list(ub.take(sample_grid['targets'], sample_grid['positives_indexes']))

    Example:
        >>> from watch.tasks.fusion.datamodules.kwcoco_video_data import *  # NOQA
        >>> import ndsampler
        >>> import kwcoco
        >>> dset = kwcoco.CocoDataset.demo('vidshapes2-multispectral', num_frames=30)
        >>> window_overlap = 0.0
        >>> window_dims = (3, 32, 32)
        >>> keepbound = False
        >>> time_sampling = 'soft2+distribute'
        >>> sample_grid1 = sample_video_spacetime_targets(dset, window_dims, window_overlap, time_sampling='soft2+distribute')
        >>> sample_grid2 = sample_video_spacetime_targets(dset, window_dims, window_overlap, time_sampling='contiguous+pairwise')

        ub.peek(sample_grid1['vidid_to_time_sampler'].values()).show_summary(fnum=1)
        ub.peek(sample_grid2['vidid_to_time_sampler'].values()).show_summary(fnum=2)
        _ = xdev.profile_now(sample_video_spacetime_targets)(dset, window_dims, window_overlap)

        import xdev
        globals().update(xdev.get_func_kwargs(sample_video_spacetime_targets))

    Ignore:
        >>> from watch.tasks.fusion.datamodules.kwcoco_video_data import *  # NOQA
        >>> import watch
        >>> dvc_dpath = watch.find_smart_dvc_dpath()
        >>> coco_fpath = dvc_dpath / 'Drop1-Aligned-TA1-2022-01/data.kwcoco.json'
        >>> dset = kwcoco.CocoDataset(coco_fpath)
        >>> window_overlap = 0.0
        >>> window_dims = (3, 128, 128)
    """
    # Create a sliding window object for each specific image (because they may
    # have different sizes, technically we could memoize this)
    import pyqtree
    from watch.tasks.fusion.datamodules import temporal_sampling as tsm  # NOQA

    window_space_dims = window_dims[1:3]
    window_time_dims = window_dims[0]
    print('window_time_dims = {!r}'.format(window_time_dims))

    # It is important that keepbound is True at test time, otherwise
    # we may not predict on the bottom right of the image.
    keepbound = True

    vidid_to_space_slider = {}
    for vidid, video in dset.index.videos.items():
        full_dims = [video['height'], video['width']]
        window_dims_ = full_dims if window_space_dims == 'full' else window_space_dims
        slider = kwarray.SlidingWindow(full_dims, window_dims_,
                                       overlap=window_overlap,
                                       keepbound=keepbound,
                                       allow_overshoot=True)
        vidid_to_space_slider[vidid] = slider

    # from ndsampler import isect_indexer
    # _isect_index = isect_indexer.FrameIntersectionIndex.from_coco(dset)
    targets = []
    positive_idxs = []
    negative_idxs = []

    vidid_to_time_sampler = {}
    vidid_to_valid_gids = {}

    parts = set(time_sampling.split('+'))
    affinity_type_parts = parts & {'hard', 'hardish', 'contiguous', 'soft2', 'soft', 'hardish2', 'hardish3'}
    update_rule_parts = parts & {'distribute', 'pairwise'}
    unknown = (parts - affinity_type_parts) - update_rule_parts
    if unknown:
        raise ValueError('Unknown time-sampling parts: {}'.format(unknown))

    affinity_type = '+'.join(list(affinity_type_parts))
    update_rule = '+'.join(list(update_rule_parts))
    if not update_rule:
        update_rule = 'distribute'

    dset_hashid = dset._cached_hashid()

    @ub.memoize
    def get_image_valid_region_in_vidspace(gid):
        coco_poly = dset.index.imgs[gid].get('valid_region', None)
        if not coco_poly:
            sh_poly_vid = None
        else:
            warp_vid_from_img = kwimage.Affine.coerce(
                dset.index.imgs[gid]['warp_img_to_vid'])
            kw_poly_img = kwimage.MultiPolygon.coerce(coco_poly)
            kw_poly_vid = kw_poly_img.warp(warp_vid_from_img)
            sh_poly_vid = kw_poly_vid.to_shapely()
        return sh_poly_vid

    if negative_classes is None:
        negative_classes = heuristics.BACKGROUND_CLASSES

    # Given an video
    all_vid_ids = list(dset.index.videos.keys())
    for video_id in ub.ProgIter(all_vid_ids, desc='sample video regions', verbose=3):
        slider = vidid_to_space_slider[video_id]

        video_info = dset.index.videos[video_id]
        all_video_gids = list(dset.index.vidid_to_gids[video_id])

        if exclude_sensors is not None:
            sensor_coarse = dset.images(all_video_gids).lookup('sensor_coarse', '')
            flags = [s not in exclude_sensors for s in sensor_coarse]
            video_gids = list(ub.compress(all_video_gids, flags))
        else:
            video_gids = all_video_gids
        # video_frame_idxs = np.array(list(range(len(video_gids))))

        # TODO: allow for multiple time samplers
        time_sampler = tsm.TimeWindowSampler.from_coco_video(
            dset, video_id, gids=video_gids, time_window=window_time_dims,
            affinity_type=affinity_type, update_rule=update_rule,
            name=video_info['name'], time_span=time_span)
        time_sampler.video_gids = np.array(video_gids)
        time_sampler.determenistic = True

        depends = [
            dset_hashid,
            negative_classes,
            affinity_type,
            update_rule,
            video_info['name'],
            window_dims, window_overlap,
            negative_classes, keepbound,
            exclude_sensors,
            time_sampling,
            time_span, use_annot_info,
            use_grid_positives,
            use_centered_positives
        ]
        cacher = ub.Cacher('sliding-window-cache', appname='watch',
                           depends=depends)
        _cached = cacher.tryload()
        if _cached is None:

            video_targets = []
            video_positive_idxs = []
            video_negative_idxs = []
            # For each frame, determenistically compute an initial list of which
            # supporting frames we will look at when making a prediction for the
            # "main" frame. Initially this is only based on temporal metadata.  We
            # may modify this later depending on spatial properties.
            main_idx_to_gids = {
                main_idx: list(ub.take(video_gids, time_sampler.sample(main_idx)))
                for main_idx in time_sampler.main_indexes
            }

            if use_annot_info:
                # Build a distribution of where annotations exist in this dataset
                qtree = pyqtree.Index((0, 0, video_info['width'], video_info['height']))
                qtree.aid_to_tlbr = {}
                # qtree.idx_to_tlbr = {}
                tid_to_infos = ub.ddict(list)
                video_aids = dset.images(video_gids).annots.lookup('id')
                annot_vid_tlbr = []
                aids_to_track = []
                for aids, gid in zip(video_aids, video_gids):
                    warp_vid_from_img = kwimage.Affine.coerce(
                        dset.index.imgs[gid]['warp_img_to_vid'])
                    img_info = dset.index.imgs[gid]
                    frame_index = img_info['frame_index']
                    tids = dset.annots(aids).lookup('track_id', None)
                    cids = dset.annots(aids).lookup('category_id', None)
                    cnames = dset.categories(cids).name

                    for tid, aid, cid, cname in zip(tids, aids, cids, cnames):
                        if cname not in negative_classes:
                            aids_to_track.append(aid)
                            imgspace_box = kwimage.Boxes([dset.index.anns[aid]['bbox']], 'xywh')
                            vidspace_box = imgspace_box.warp(warp_vid_from_img)
                            tlbr_box = vidspace_box.to_tlbr().data[0]
                            annot_vid_tlbr.append(tlbr_box)
                            if tid is not None:
                                tid_to_infos[tid].append({
                                    'gid': gid,
                                    'cid': cid,
                                    'frame_index': frame_index,
                                    'vidspace_box': tlbr_box,
                                    'cname': dset._resolve_to_cat(cid)['name'],
                                    'aid': aid,
                                })
                            qtree.insert(aid, tlbr_box)
                            qtree.aid_to_tlbr[aid] = tlbr_box
                            # qtree.idx_to_tlbr[aid] = tlbr_box

                # if len(annot_vid_tlbr):
                #     unique_tlbr = util_kwarray.unique_rows(np.array(annot_vid_tlbr))
                # else:
                #     unique_tlbr = []
                # for idx, tlbr_box in enumerate(unique_tlbr):
                #     qtree.insert(idx, tlbr_box)
                #    qtree.idx_to_tlbr[idx] = tlbr_box

                # tid_to_dframe = ub.map_vals(kwarray.DataFrameLight.from_dict, tid_to_infos)
                # for track_dframe in tid_to_dframe.values():
                #     track_dframe['gid'] = np.array(track_dframe['gid'])
                #     track_dframe['frame_index'] = np.array(track_dframe['frame_index'])
                #     # Precompute for speed
                #     track_boxes = kwimage.Boxes(np.array(track_dframe['vidspace_box']), 'ltrb')
                #     track_dframe['track_pairwise_ious'] = track_boxes.ious(track_boxes)
                #     track_dframe['track_boxes'] = track_boxes

            RESPECT_VALID_REGIONS = True
            for space_region in ub.ProgIter(list(slider), desc='Sliding window'):
                y_sl, x_sl = space_region

                kw_space_box = kwimage.Boxes.from_slice(space_region).to_tlbr()

                # Find all annotations that pass through this spatial region
                if use_annot_info:
                    query = kw_space_box.data[0]
                    isect_aids = list(qtree.intersect(query))
                    # isect_aids = set(isect_aids)
                    isect_gids = set(dset.annots(isect_aids).lookup('image_id'))

                if RESPECT_VALID_REGIONS:
                    # Reselect the keyframes if we overlap an invalid region
                    # (as denoted in the metadata, further filtering may happen later)
                    # todo: refactor to be cleaner
                    try:
                        main_idx_to_gids2, resampled = _refine_time_sample(
                            dset, main_idx_to_gids, kw_space_box, time_sampler,
                            get_image_valid_region_in_vidspace)
                    except tsm.TimeSampleError:
                        # Hack, just skip the region
                        # We might be able to sample less and still be ok
                        continue
                else:
                    main_idx_to_gids2 = main_idx_to_gids
                    resampled = False

                for main_idx, gids in main_idx_to_gids2.items():
                    main_gid = time_sampler.video_gids[main_idx]
                    label = 'unknown'

                    if use_annot_info:
                        if isect_aids:
                            has_annot = bool(isect_gids & set(gids))
                        else:
                            has_annot = False
                        if has_annot:
                            label = 'positive_grid'
                        else:
                            # Hack: exclude all annotated regions from negative sampling
                            label = 'negative_grid'

                    # Or do that on the fly?
                    if False:
                        for gid in gids:
                            coco_img = dset.coco_image(gid)
                            coco_img.channels
                            part = coco_img.delay(space='video')
                            cropped = part.crop(space_region)
                            arr = cropped.finalize(as_xarray=True)
                            if np.all(arr == 0):
                                print('BLACK REGION')

                    if label == 'positive_grid':
                        if not use_grid_positives:
                            continue
                        video_positive_idxs.append(len(video_targets))
                    elif label == 'negative_grid':
                        video_negative_idxs.append(len(video_targets))

                    video_targets.append({
                        'main_idx': main_idx,
                        'video_id': video_id,
                        'gids': gids,
                        'main_gid': main_gid,
                        'space_slice': space_region,
                        'resampled': resampled,
                        'label': label,
                    })

            INSERT_CENTERED_ANNOT_WINDOWS = use_centered_positives
            if INSERT_CENTERED_ANNOT_WINDOWS and use_annot_info:
                # FIXME: This code is too slow
                # in addition to the sliding window sample, add positive samples
                # centered around each annotation.
                for tid, infos in ub.ProgIter(list(tid_to_infos.items()), desc='Centered annots'):
                    # existing_gids = [info['gid'] for info in infos]
                    for info in infos:
                        main_gid = info['gid']
                        ann_box = kwimage.Boxes([info['vidspace_box']], 'tlbr').to_cxywh()
                        ann_box.data[:, 2] = window_space_dims[1]
                        ann_box.data[:, 3] = window_space_dims[0]
                        kw_space_region = ann_box.to_tlbr().quantize()
                        space_region = kw_space_region.to_slices()[0]
                        #  FIXME, this code is ugly
                        # TODO: we could make frames where the phase transitions
                        # more likely here.
                        _hack_main_idx = np.where(time_sampler.video_gids == main_gid)[0][0]
                        sample_gids = list(ub.take(video_gids, time_sampler.sample(_hack_main_idx)))
                        _hack = {_hack_main_idx: sample_gids}
                        if 0:
                            # Too slow to handle here, will have to handle
                            # in getitem or be more efficient
                            # 86% of the time is spent here
                            _hack2, _ = _refine_time_sample(
                                dset, _hack, kw_space_box, time_sampler,
                                get_image_valid_region_in_vidspace)
                        else:
                            _hack2 = _hack
                        if _hack2:
                            gids = _hack2[_hack_main_idx]
                            label = 'positive_center'
                            video_positive_idxs.append(len(video_targets))
                            video_targets.append({
                                'main_idx': _hack_main_idx,
                                'video_id': video_id,
                                'gids': gids,
                                'main_gid': main_gid,
                                'space_slice': space_region,
                                'label': label,
                                'resampled': -1,
                            })

            _cached = {
                'video_targets': video_targets,
                'video_positive_idxs': video_positive_idxs,
                'video_negative_idxs': video_negative_idxs,
            }
            cacher.save(_cached)

        offset = len(targets)
        targets.extend(_cached['video_targets'])
        positive_idxs.extend([idx + offset for idx in _cached['video_positive_idxs']])
        negative_idxs.extend([idx + offset for idx in _cached['video_negative_idxs']])

        # Disable determenism
        time_sampler.determenistic = False
        vidid_to_time_sampler[video_id] = time_sampler
        vidid_to_valid_gids[video_id] = video_gids

    print('Found {} targets'.format(len(targets)))
    if use_annot_info:
        print('Found {} positives'.format(len(positive_idxs)))
        print('Found {} negatives'.format(len(negative_idxs)))

    sample_grid = {
        'positives_indexes': positive_idxs,
        'negatives_indexes': negative_idxs,
        'targets': targets,
        'vidid_to_valid_gids': vidid_to_valid_gids,
        'vidid_to_time_sampler': vidid_to_time_sampler,
    }
    return sample_grid


def _refine_time_sample(dset, main_idx_to_gids, kw_space_box, time_sampler, get_image_valid_region_in_vidspace):
    """
    Refine the time sample based on spatial information
    """
    from watch.tasks.fusion.datamodules import temporal_sampling as tsm  # NOQA
    video_gids = time_sampler.video_gids

    iooa_thresh = 0.2  # parametarize?

    gid_to_isbad = {}
    for gid in video_gids:
        valid_poly = get_image_valid_region_in_vidspace(gid)
        gid_to_isbad[gid] = False
        if valid_poly is not None:
            sh_space_poly = kw_space_box.to_shapley()[0]
            # flag = valid_poly.intersects(sh_space_poly)
            isect = valid_poly.intersection(sh_space_poly)
            iooa = isect.area / sh_space_poly.area
            if iooa < iooa_thresh:
                gid_to_isbad[gid] = True

    all_bad_gids = [gid for gid, flag in gid_to_isbad.items() if flag]

    try:
        resampled = 0
        refined_sample = {}
        for main_idx, gids in main_idx_to_gids.items():
            main_gid = video_gids[main_idx]
            # Skip the sample when the "main" frame is bad.
            if not gid_to_isbad[main_gid]:
                good_gids = [gid for gid in gids if not gid_to_isbad[gid]]
                if good_gids != gids:
                    include_idxs = np.where(kwarray.isect_flags(video_gids, good_gids))[0]
                    exclude_idxs = np.where(kwarray.isect_flags(video_gids, all_bad_gids))[0]
                    chosen = time_sampler.sample(include=include_idxs, exclude=exclude_idxs, error_level=1, return_info=False)
                    new_gids = list(ub.take(video_gids, chosen))
                    # Are we allowed to return less than the initial expected
                    # number of frames? For transformers yes, but we should be
                    # careful to ask the user if they expect this.
                    new_are_bad = [g for g in new_gids if gid_to_isbad[g]]
                    if not new_are_bad:
                        resampled += 1
                        refined_sample[main_idx] = new_gids
                else:
                    refined_sample[main_idx] = gids
    except tsm.TimeSampleError:
        raise

    return refined_sample, resampled


def lookup_track_info(coco_dset, tid):
    """
    Find the spatio-temporal extent of a track
    """
    track_aids = coco_dset.index.trackid_to_aids[tid]
    vidspace_boxes = []
    track_gids = []
    for aid in track_aids:
        ann = coco_dset.index.anns[aid]
        gid = ann['image_id']
        track_gids.append(gid)
        img = coco_dset.index.imgs[gid]
        bbox = ann['bbox']
        vid_from_img = kwimage.Affine.coerce(img.get('warp_img_to_vid', None))
        imgspace_box = kwimage.Boxes([bbox], 'xywh')
        vidspace_box = imgspace_box.warp(vid_from_img)
        vidspace_boxes.append(vidspace_box)
    all_vidspace_boxes = kwimage.Boxes.concatenate(vidspace_boxes)
    full_vid_box = all_vidspace_boxes.bounding_box().to_xywh()

    frame_index = coco_dset.images(track_gids).lookup('frame_index')
    track_gids = list(ub.take(track_gids, ub.argsort(frame_index)))

    track_info = {
        'tid': tid,
        'full_vid_box': full_vid_box,
        'track_gids': track_gids,
    }
    return track_info


def make_track_based_spatial_samples(coco_dset):
    """
    Ignore:
        >>> # xdoctest: +REQUIRES(env:DVC_DPATH)
        >>> import os
        >>> from watch.tasks.fusion.datamodules.kwcoco_video_data import *  # NOQA
        >>> import watch
        >>> dvc_dpath = watch.find_smart_dvc_dpath()
        >>> coco_fpath = dvc_dpath / 'Drop1-Aligned-L1/data.kwcoco.json'
        >>> coco_dset = kwcoco.CocoDataset(coco_fpath)
    """
    tid_list = list(coco_dset.index.trackid_to_aids.keys())
    tid_to_trackinfo = {}
    for tid in tid_list:
        track_info = lookup_track_info(coco_dset, tid)
        gid = track_info['track_gids'][0]
        vidid = coco_dset.index.imgs[gid]['video_id']
        track_info['vidid'] = vidid
        tid_to_trackinfo[tid] = track_info

    vidid_to_tracks = ub.group_items(tid_to_trackinfo.values(), key=lambda x: x['vidid'])

    window_space_dims = [96, 96]

    for vidid, trackinfos in vidid_to_tracks.items():
        positive_boxes = []
        for track_info in trackinfos:
            boxes = track_info['full_vid_box']
            positive_boxes.append(boxes.to_cxywh())
        positives = kwimage.Boxes.concatenate(positive_boxes)
        positives_samples = positives.to_cxywh()
        positives_samples.data[:, 2] = window_space_dims[0]
        positives_samples.data[:, 3] = window_space_dims[1]
        print('positive_boxes = {}'.format(ub.repr2(positive_boxes, nl=1)))

        video = coco_dset.index.videos[vidid]
        full_dims = [video['height'], video['width']]
        window_overlap = 0.0
        keepbound = 0

        window_dims_ = full_dims if window_space_dims == 'full' else window_space_dims
        slider = kwarray.SlidingWindow(full_dims, window_dims_,
                                       overlap=window_overlap,
                                       keepbound=keepbound,
                                       allow_overshoot=True)

        sliding_boxes = kwimage.Boxes.concatenate(list(map(kwimage.Boxes.from_slice, slider)))
        ious = sliding_boxes.ious(positives)
        overlaps = ious.sum(axis=1)
        negative_boxes = sliding_boxes.compress(overlaps == 0)

        if 1:
            import kwplot
            kwplot.autompl()
            fig = kwplot.figure(fnum=vidid)
            ax = fig.gca()
            ax.set_title(video['name'])
            negative_boxes.draw(setlim=1, color='red', fill=True)
            positives.draw(color='limegreen')
            positives_samples.draw(color='green')


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
