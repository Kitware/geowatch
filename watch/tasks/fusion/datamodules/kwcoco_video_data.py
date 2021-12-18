import einops
import kwarray
import kwcoco
import kwimage
import ndsampler
import numpy as np
import pathlib
import pytorch_lightning as pl
import torch
import ubelt as ub
import random
from kwcoco import channel_spec
from torch.utils import data
from watch.tasks.fusion import utils
from watch import heuristics
from watch.utils import kwcoco_extensions
from watch.utils import util_bands
from watch.utils import util_iter
from watch.utils import util_kwimage
from watch.utils import util_time
from watch.utils import util_norm
from watch.utils.lightning_ext import util_globals

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
        >>> # xdoctest: +REQUIRES(env:DVC_DPATH)
        >>> # Run the following tests on real watch data if DVC is available
        >>> from watch.tasks.fusion.datamodules.kwcoco_video_data import *  # NOQA
        >>> from os.path import join
        >>> from watch.utils.util_data import find_smart_dvc_dpath
        >>> dvc_dpath = find_smart_dvc_dpath()
        >>> coco_fpath = dvc_dpath / 'drop1-S2-L8-aligned/data.kwcoco.json'
        >>> import kwcoco
        >>> dset = train_dataset = kwcoco.CocoDataset(coco_fpath)
        >>> test_dataset = None
        >>> img = ub.peek(train_dataset.imgs.values())
        >>> chan_info = kwcoco_extensions.coco_channel_stats(dset)
        >>> channels = chan_info['common_channels']
        >>> #chan_spec = kwcoco.channel_spec.FusedChannelSpec.coerce(channels)
        >>> #channels = None
        >>> #
        >>> batch_size = 1
        >>> time_steps = 2
        >>> chip_size = 330
        >>> self = KWCocoVideoDataModule(
        >>>     train_dataset=train_dataset,
        >>>     test_dataset=test_dataset,
        >>>     batch_size=batch_size,
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
        >>> # Visualize
        >>> canvas = self.draw_batch(batch)
        >>> # xdoctest: +REQUIRES(--show)
        >>> import kwplot
        >>> kwplot.autompl()
        >>> kwplot.imshow(canvas)
        >>> kwplot.show_if_requested()

    Example:
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
        >>> batch = next(iter(dl))
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
        exclude_sensors=['L8'],  # NOQA
        channels=None,
        batch_size=4,
        preprocessing_step=None,
        tfms_channel_subset=None,  # DEPRECATE
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
        true_multimodal=False,
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
        self.channels = channels
        self.batch_size = batch_size
        self.preprocessing_step = preprocessing_step
        self.normalize_inputs = normalize_inputs
        self.time_sampling = time_sampling
        self.exclude_sensors = exclude_sensors
        self.diff_inputs = diff_inputs
        self.time_span = time_span
        self.match_histograms = match_histograms
        self.resample_invalid_frames = resample_invalid_frames
        self.upweight_centers = upweight_centers
        self.normalize_perframe = normalize_perframe
        self.true_multimodal = true_multimodal

        self.common_dataset_kwargs = dict(
            channels=self.channels,
            time_sampling=self.time_sampling,
            diff_inputs=self.diff_inputs,
            exclude_sensors=self.exclude_sensors,
            match_histograms=self.match_histograms,
            upweight_centers=self.upweight_centers,
            resample_invalid_frames=self.resample_invalid_frames,
            normalize_perframe=self.normalize_perframe,
            true_multimodal=self.true_multimodal,
        )

        self.num_workers = util_globals.coerce_num_workers(num_workers)
        self.torch_start_method = torch_start_method
        self.torch_sharing_strategy = torch_sharing_strategy

        self.input_stats = None
        self.dataset_stats = None

        # will only correspond to train
        self.classes = None
        self.input_channels = None

        # Store train / test / vali
        self.torch_datasets = {}
        self.coco_datasets = {}

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

        parser.add_argument(
            '--normalize_inputs', default=True, type=smartcast, help=ub.paragraph(
                '''
                if True, computes the mean/std for this dataset on each mode
                so this can be passed to the model.
                '''))

        parser.add_argument(
            '--match_histograms', default=True, type=smartcast, help=ub.paragraph(
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
                expression
                '''
            ))

        parser.add_argument(
            '--true_multimodal', default=False, type=smartcast, help=ub.paragraph(
                '''
                Enables new logic for sampling multimodal data
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

        return parent_parser

    def setup(self, stage):
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
            train_coco_dset = kwcoco.CocoDataset.coerce(train_data)
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

            if self.normalize_inputs:
                if isinstance(self.normalize_inputs, str):
                    raise NotImplementedError(
                        'TODO: handle special normalization keys, '
                        'e.g. imagenet')
                else:
                    if isinstance(self.normalize_inputs, int):
                        num = self.normalize_inputs
                    else:
                        num = None
                    self.dataset_stats = train_dataset.cached_dataset_stats(
                        num=num, num_workers=self.num_workers,
                        batch_size=self.batch_size)

                    # Hack for now:
                    self.input_stats = self.dataset_stats

            if self.vali_kwcoco is not None:
                # Explicit validation dataset should be prefered
                vali_data = self.vali_kwcoco
                if isinstance(vali_data, pathlib.Path):
                    vali_data = str(vali_data.expanduser())
                if self.verbose:
                    print('Build validation kwcoco dataset')
                kwcoco_ds = kwcoco.CocoDataset.coerce(vali_data)
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
            test_coco_dset = kwcoco.CocoDataset.coerce(test_data)
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

    def _make_dataloader(self, stage, shuffle=False):
        return data.DataLoader(
            self.torch_datasets[stage],
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            collate_fn=ub.identity,  # disable collation
            shuffle=shuffle,
            pin_memory=True,
        )

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
            >>> kwplot.show_if_requestedV


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
                item_output = ub.AutoDict()
                # output_walker = ub.IndexableWalker(item_output)
                input_walker = ub.IndexableWalker(outputs)
                for p, v in input_walker:
                    if isinstance(v, torch.Tensor):
                        input_walker[p] = v.data.cpu().numpy()
                # for k in ['change_probs', 'class_probs', 'saliency_probs']:
                #     if k in outputs:
                #         item_output[k] = outputs[k][item_idx].data.cpu().numpy()

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
        >>> self = KWCocoVideoDataset(sampler, sample_shape=sample_shape, channels=channels, time_sampling='soft+distribute', diff_inputs=True, match_histograms=True)
        >>> index = len(self) // 4
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

    Example:
        >>> from watch.tasks.fusion.datamodules.kwcoco_video_data import *  # NOQA
        >>> import ndsampler
        >>> import kwcoco
        >>> coco_dset = kwcoco.CocoDataset.demo('vidshapes2-multispectral', num_frames=5)
        >>> sampler = ndsampler.CocoSampler(coco_dset)
        >>> sample_shape = (2, 128, 128)
        >>> self = KWCocoVideoDataset(sampler, sample_shape=sample_shape,
        >>>                           channels=None, diff_inputs=True)
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
        >>> from watch.utils.util_data import find_smart_dvc_dpath
        >>> dvc_dpath = find_smart_dvc_dpath()
        >>> coco_fpath = dvc_dpath / 'drop1-S2-L8-aligned/data.kwcoco.json'
        >>> coco_dset = kwcoco.CocoDataset(coco_fpath)
        >>> sampler = ndsampler.CocoSampler(coco_dset)
        >>> sample_shape = (7, 128, 128)
        >>> self = KWCocoVideoDataset(sampler, sample_shape=sample_shape, channels='red|green|blue|swir16|swir22|nir|ASI', match_histograms=True)
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
        >>> from watch.utils.util_data import find_smart_dvc_dpath
        >>> dvc_dpath = find_smart_dvc_dpath()
        >>> coco_dset = kwcoco.CocoDataset(coco_fpath)
        >>> sampler = ndsampler.CocoSampler(coco_dset)
        >>> self = KWCocoVideoDataset(
        >>>     sampler,
        >>>     sample_shape=(5, 128, 128),
        >>>     window_overlap=0,
        >>>     channels="blue|green|red|nir|swir16",
        >>>     neg_to_pos_ratio=0, time_sampling='auto', diff_inputs=1, mode='fit', match_histograms=True,
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
        true_multimodal=False,
    ):

        # TODO: the set of "valid" background classnames should be defined
        # by the inputs, not hard-coded in the dataloader. This can either be a
        # list of names provided to the training config, or something baked
        # into the kwcoco spec marking a class as some type of "background"
        self._hueristic_background_classnames = heuristics.BACKGROUND_CLASSES
        self._heuristic_ignore_classnames = heuristics.IGNORE_CLASSNAMES
        self._heuristic_undistinguished_classnames = heuristics.UNDISTINGUISHED_CLASSES

        self.match_histograms = match_histograms
        self.normalize_perframe = normalize_perframe
        self.resample_invalid_frames = resample_invalid_frames
        self.upweight_centers = upweight_centers

        if channels is None:
            # Hack to use all channels in the first image.
            # (Does not handle heterogeneous channels yet)
            chan_info = kwcoco_extensions.coco_channel_stats(sampler.dset)
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
            negative_classes = (
                self._heuristic_ignore_classnames |
                self._hueristic_background_classnames
            )
            new_sample_grid = sample_video_spacetime_targets(
                sampler.dset, window_dims=sample_shape,
                window_overlap=window_overlap,
                negative_classes=negative_classes,
                keepbound=False,
                use_annot_info=True,
                exclude_sensors=exclude_sensors,
                time_sampling=time_sampling,
                time_span=time_span,
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
            print('len(neg_pool) ' + str(len(self.negative_pool)))
            print('self.n_pos = {!r}'.format(self.n_pos))
            print('self.n_neg = {!r}'.format(self.n_neg))
            print('self.length = {!r}'.format(self.length))
        self.new_sample_grid = new_sample_grid

        self.window_overlap = window_overlap
        self.sampler = sampler

        # Add extra categories if we need to and construct a new classes object
        graph = self.sampler.classes.graph
        if 0:
            import networkx as nx
            print(nx.forest_str(graph, with_labels=True))

        self.background_classes = self._hueristic_background_classnames & set(graph.nodes)
        if not len(self.background_classes):
            graph.add_node('background')
            self.background_classes = self._hueristic_background_classnames & set(graph.nodes)

        self.ignore_classes = self._heuristic_ignore_classnames & set(graph.nodes)
        if not len(self.ignore_classes):
            graph.add_node('ignore')
            self.ignore_classes = self._heuristic_ignore_classnames & set(graph.nodes)

        self.undistinguished_classes = self._heuristic_undistinguished_classnames & set(graph.nodes)

        self.classes = kwcoco.CategoryTree(graph)

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

        self.mode = mode

        self.true_multimodal = true_multimodal
        self.augment = False
        self.disable_augmenter = False

        # hidden option for now (todo: expose this)
        self.inference_only = False
        self.with_change = True
        self.with_class = True

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
        """
        do_shift = False
        if not self.disable_augmenter and self.mode == 'fit':
            # do_shift = np.random.rand() > 0.5
            do_shift = True
        if do_shift:
            # Spatial augmentation
            rng = kwarray.ensure_rng(None)
            space_box = kwimage.Boxes.from_slice(tr_['space_slice'])
            w = space_box.width.ravel()[0]
            h = space_box.height.ravel()[0]

            # hack: this prevents us from assuming there is a target in the
            # window, but it lets us get the benefit of chip_overlap=0.5 while
            # still having it at 0 for faster epochs.
            # TODO: dont shift off the edge.
            aff = kwimage.Affine.coerce(offset=(
                rng.randint(-w // 2, w // 2),
                rng.randint(-h // 2, h // 2)))

            space_box = space_box.warp(aff).quantize()
            # aff = kwimage.Affine.coerce(offset=rng.randint(-8, 8, size=2))
            space_box = kwimage.Boxes.from_slice(tr_['space_slice']).warp(aff).quantize()
            tr_['space_slice'] = space_box.astype(int).to_slices()[0]

            # Temporal augmentation
            vidid = tr_['video_id']
            time_sampler = self.new_sample_grid['vidid_to_time_sampler'][vidid]
            valid_gids = self.new_sample_grid['vidid_to_valid_gids'][vidid]
            tr_['gids'] = list(ub.take(valid_gids, time_sampler.sample(tr_['main_idx'])))
        return tr_

    @profile
    def __getitem__(self, index):
        """
        Example:
            >>> from watch.tasks.fusion.datamodules.kwcoco_video_data import *  # NOQA
            >>> import ndsampler
            >>> import kwcoco
            >>> coco_dset = kwcoco.CocoDataset.demo('vidshapes2-multispectral', num_frames=5)
            >>> sampler = ndsampler.CocoSampler(coco_dset)
            >>> channels = 'B10|B8a|B1|B8'
            >>> sample_shape = (4, 530, 610)
            >>> self = KWCocoVideoDataset(sampler, sample_shape=sample_shape, channels=channels, diff_inputs=True)
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
            >>> self = KWCocoVideoDataset(sampler, sample_shape=sample_shape, channels=channels, neg_to_pos_ratio=0.1)
            >>> item = self[self.n_pos + 1]
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
            >>> _default = ub.expandpath('$HOME/data/dvc-repos/smart_watch_dvc')
            >>> dvc_dpath = os.environ.get('DVC_DPATH', _default)
            >>> coco_fpath = join(dvc_dpath, 'drop1-S2-L8-aligned/data.kwcoco.json')
            >>> coco_dset = kwcoco.CocoDataset(coco_fpath)
            >>> sampler = ndsampler.CocoSampler(coco_dset)
            >>> self = KWCocoVideoDataset(
            >>>     sampler,
            >>>     sample_shape=(2, 128, 128),
            >>>     window_overlap=0,
            >>>     channels="ASI|MF_Norm|AF|EVI|red|green|blue|swir16|swir22|nir",
            >>>     neg_to_pos_ratio=0, time_sampling='auto', diff_inputs=0,
            >>> )
            >>> item = self[5]
            >>> canvas = self.draw_item(item, max_channels=10, overlay_on_image=0)
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
            >>> from watch.utils.util_data import find_smart_dvc_dpath
            >>> dvc_dpath = find_smart_dvc_dpath()
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
            >>> kwplot.imshow(canvas)
            >>> kwplot.show_if_requested()

        Example:
            >>> from watch.tasks.fusion.datamodules.kwcoco_video_data import *  # NOQA
            >>> import ndsampler
            >>> import kwcoco
            >>> import watch
            >>> coco_dset = watch.demo.demo_kwcoco_multisensor()
            >>> channels = '|'.join(sorted(set(ub.flatten([kwcoco.ChannelSpec.coerce(c).fuse().as_list() for c in groups.keys()]))))
            >>> sampler = ndsampler.CocoSampler(coco_dset)
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

        if NEW_TRUE_MULTIMODAL:
            # New true-multimodal data items
            gid_to_sample = {}
            gid_to_isbad = {}

            def sample_one_frame(gid):
                coco_img = coco_dset.coco_image(gid)
                sensor_channels = self.sample_channels & coco_img.channels
                tr_frame = tr_.copy()
                tr_frame['gids'] = [gid]
                tr_frame['channels'] = sensor_channels
                sample = sampler.load_sample(
                    tr_frame, with_annots=with_annots,
                    padkw={'constant_values': np.nan}
                )
                gid_to_isbad[gid] = np.all(np.isnan(sample['im']))
                gid_to_sample[gid] = sample

            for gid in tr_['gids']:
                sample_one_frame(gid)

            vidid = tr_['video_id']
            video = coco_dset.index.videos[vidid]
            time_sampler = self.new_sample_grid['vidid_to_time_sampler'][vidid]
            video_gids = time_sampler.video_gids

            if ALLOW_RESAMPLE:
                # If any image is junk allow for a resample
                if any(gid_to_isbad.values()):
                    vidid = tr_['video_id']
                    time_sampler = self.new_sample_grid['vidid_to_time_sampler'][vidid]
                    max_tries = 30
                    for iter_idx in range(max_tries):
                        good_gids = np.array([gid for gid, flag in gid_to_isbad.items() if not flag])
                        if len(good_gids) == len(tr['gids']):
                            break
                        bad_gids = np.array([gid for gid, flag in gid_to_isbad.items() if flag])
                        # print('resampling: {}'.format(index))
                        include_idxs = np.where(kwarray.isect_flags(time_sampler.video_gids, good_gids))[0]
                        exclude_idxs = np.where(kwarray.isect_flags(time_sampler.video_gids, bad_gids))[0]
                        chosen = time_sampler.sample(include=include_idxs, exclude=exclude_idxs, error_level=1, return_info=False)
                        new_idxs = np.setdiff1d(chosen, include_idxs)
                        new_gids = video_gids[new_idxs]
                        print('new_gids = {!r}'.format(new_gids))
                        if not len(new_gids):
                            # Exhausted all possibilities
                            break
                        for gid in new_gids:
                            sample_one_frame(gid)

            good_gids = [gid for gid, flag in gid_to_isbad.items() if not flag]
            final_gids = ub.oset(video_gids) & good_gids
            # coco_dset.images(final_gids).lookup('date_captured')
            tr_['gids'] = final_gids

            if self.sample_shape is None:
                input_dsize = gid_to_sample[good_gids[0]]['im'].shape[1:3][::-1]
            else:
                input_dsize = self.sample_shape[-2:][::-1]

            requested_channel_order = self.input_channels.spec.split('|')

            if not self.inference_only:
                # Learn more from the center of the space-time patch
                num_frames = len(good_gids)
                time_weights = kwimage.gaussian_patch((1, num_frames))[0]
                time_weights = time_weights / time_weights.max()
                space_weights = util_kwimage.upweight_center_mask(input_dsize[::-1])

            if self.special_inputs:
                raise NotImplementedError

            if self.diff_inputs:
                raise NotImplementedError

            if self.match_histograms:
                raise NotImplementedError

            frame_items = []
            for time_idx, gid in enumerate(final_gids):
                img = coco_dset.index.imgs[gid]
                sample = gid_to_sample[gid]
                # TODO: get nodata value here
                # FIXME: nodata value needs to be handled in the kwcoco delay
                frame_chans = sample['tr']['channels'].fuse().as_list()
                frame_imdata = sample['im'][0]
                mode_key = '|'.join(frame_chans)

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

                dt_captured = img.get('date_captured', None)
                if dt_captured:
                    dt_captured = util_time.coerce_datetime(dt_captured)
                    timestamp = dt_captured.timestamp()
                else:
                    timestamp = np.nan

                frame_item = {
                    'gid': gid,
                    'date_captured': img.get('date_captured', ''),
                    'timestamp': timestamp,
                    'sensor_coarse': img.get('sensor_coarse', ''),
                    'modes': {
                        mode_key: input_chw,
                    },
                    'change': None,
                    'class_idxs': None,
                    'ignore': None,
                }

                if not self.inference_only:
                    # Remember to apply any transform to the dets as well
                    frame_dets = sample['annots']['frame_dets'][0]
                    dets = frame_dets.scale(info['scale'])
                    dets = dets.translate(info['offset'])

                    # Create truth masks
                    bg_idx = self.bg_idx
                    space_shape = frame.shape[:2]
                    frame_cidxs = np.full(space_shape, dtype=np.int32,
                                          fill_value=bg_idx)

                    ohe_shape = (len(self.classes),) + space_shape
                    frame_class_ohe = np.zeros(ohe_shape, dtype=np.uint8)
                    saliency_ignore = np.zeros(space_shape, dtype=np.uint8)
                    frame_class_ignore = np.zeros(space_shape, dtype=np.uint8)

                    # Rasterize frame targets
                    ann_polys = dets.data['segmentations'].to_polygon_list()
                    ann_aids = dets.data['aids']
                    ann_cids = dets.data['cids']
                    # Note: it is important to respect class indexes, ids, and
                    # name mappings
                    # TODO: layer ordering? Multiclass prediction?
                    for poly, aid, cid in zip(ann_polys, ann_aids, ann_cids):  # NOQA
                        cidx = self.classes.id_to_idx[cid]
                        catname = self.classes.id_to_node[cid]
                        if catname in self.background_classes:
                            pass
                        elif catname in self.ignore_classes:
                            poly.fill(saliency_ignore, value=1)
                            poly.fill(frame_class_ignore, value=1)
                            # weights should allow us to distinguish ignore
                            # from background. It shouldn't be learned on in
                            # any case.
                            poly.fill(frame_class_ohe[cidx], value=1)
                        else:
                            # Indistinguishable classes should be ignored
                            # for classification, but not saliency
                            if catname in self.undistinguished_classes:
                                poly.fill(frame_class_ignore, value=1)
                                # poly.fill(frame_class_ohe[cidx], value=0)
                                # poly.fill(frame_class_ohe[cidx], value=0)
                            poly.fill(frame_class_ohe[cidx], value=1)

                    # Postprocess (Dilate?) the truth map
                    for cidx, class_map in enumerate(frame_class_ohe):
                        # class_map = util_kwimage.morphology(class_map, 'dilate', kernel=5)
                        frame_cidxs[class_map > 0] = cidx

                    if self.upweight_centers:
                        frame_weights = space_weights * time_weights[time_idx]
                    else:
                        frame_weights = 1.0

                    saliency_weights = frame_weights * (1 - saliency_ignore)
                    class_weights = frame_weights * (1 - frame_class_ignore)

                if not self.inference_only:
                    frame_item.update({
                        'class_idxs': frame_cidxs,
                        'ignore': saliency_ignore,
                        'class_weights': class_weights,
                        'saliency_weights': saliency_weights,
                    })
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
                            norm_item = util_norm.normalize_intensity(item, params={
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
                if self.with_change:
                    for frame1, frame2 in ub.iter_window(frame_items, 2):
                        frame_change = (frame1['class_idxs'] != frame2['class_idxs']).astype(np.uint8)
                        frame_change = util_kwimage.morphology(frame_change, 'open', kernel=3)
                        frame2['change'] = frame_change

            # Convert data to torch
            for frame_item in frame_items:
                truth_keys = ['change', 'class_idxs', 'ignore', 'class_weights', 'saliency_weights']
                for key in truth_keys:
                    data = frame_item.get(key, None)
                    frame_modes = frame_item['modes']
                    for mode_key in list(frame_modes.keys()):
                        mode_data = frame_modes[mode_key]
                        frame_modes[mode_key] = kwarray.ArrayAPI.tensor(mode_data)
                    if data is not None:
                        frame_item[key] = kwarray.ArrayAPI.tensor(data)

        else:
            # ------------------
            # OLD CODE

            # collect sample
            sample = sampler.load_sample(
                tr_, with_annots=with_annots,
                padkw={'constant_values': np.nan}
            )

            if ALLOW_RESAMPLE:
                # If any image is junk allow for a resample
                is_frame_bad = np.isnan(sample['im']).all(axis=(1, 2, 3))
                if np.any(is_frame_bad):
                    gids = np.array(tr_['gids'])
                    good_gids = gids[~is_frame_bad].tolist()
                    vidid = tr_['video_id']
                    time_sampler = self.new_sample_grid['vidid_to_time_sampler'][vidid]
                    video_gids = time_sampler.video_gids
                    bad_gids = gids[is_frame_bad].tolist()
                    new_bad_gids = bad_gids
                    iter_idx = 0
                    while len(new_bad_gids):
                        print('resampling: {}'.format(index))
                        include_idxs = np.where(kwarray.isect_flags(time_sampler.video_gids, good_gids))[0]
                        exclude_idxs = np.where(kwarray.isect_flags(time_sampler.video_gids, bad_gids))[0]
                        chosen, info = time_sampler.sample(include=include_idxs, exclude=exclude_idxs, error_level=1, return_info=True)
                        new_idxs = np.setdiff1d(chosen, include_idxs)
                        new_gids = video_gids[new_idxs]

                        tr_test = tr_.copy()
                        tr_test['gids'] = new_gids
                        test_sample = sampler.load_sample(
                            tr_test, with_annots=False, padkw={'constant_values': np.nan}
                        )
                        new_is_frame_bad = np.isnan(test_sample['im']).all(axis=(1, 2, 3))
                        new_good_gids = new_gids[~new_is_frame_bad].tolist()
                        new_bad_gids = new_gids[new_is_frame_bad].tolist()
                        bad_gids.extend(new_bad_gids)
                        good_gids.extend(new_good_gids)
                        iter_idx += 1
                        if iter_idx > 100:
                            raise Exception('Something is wrong')

                    resampled_gids = ub.oset(video_gids) & good_gids
                    tr_['gids'] = resampled_gids
                    # finalize resample sample
                    sample = sampler.load_sample(
                        tr_, with_annots=with_annots,
                        padkw={'constant_values': np.nan}
                    )

            vidid = sample['tr']['vidid']
            video = self.sampler.dset.index.videos[vidid]

            if self.normalize_perframe:
                im = sample['im']
                # mask = ~np.isnan(im)
                # mask[im == 0] = False
                reorg = einops.rearrange(im, 't h w c -> (t c) h w')
                to_restack = []
                for item in reorg:
                    mask = (item != 0) & np.isfinite(item)
                    norm_item = util_norm.normalize_intensity(item, params={
                        'high': 0.90,
                        'mid': 0.5,
                        'low': 0.01,
                        'mode': 'linear',
                    }, mask=mask)
                    to_restack.append(norm_item)
                norm_im = einops.rearrange(np.stack(to_restack, axis=0), '(t c) h w -> t h w c', c=im.shape[3])
                sample['im'] = norm_im

            if self.special_inputs or self.diff_inputs:
                import xarray as xr
                im = sample['im']
                # chan_coords = list(self.sample_channels.values())[0].split('|')
                chan_coords = self.sample_channels.streams()[0].parsed
                sample_im = xr.DataArray(
                    im, dims=('t', 'h', 'w', 'c'),
                    coords={'c': chan_coords})
                special_ims = []
                if self.special_inputs:
                    bands = {c: sample_im.sel(c=c).data for c in chan_coords}
                    indexes = util_bands.specialized_index_bands(bands=bands)
                    indexes = ub.map_vals(np.nan_to_num, indexes)
                    special_ims = [
                        xr.DataArray(
                            indexes[v][..., None],
                            dims=('t', 'h', 'w', 'c'),
                            coords={'c': [v]}
                        )
                        for _, values in self.special_inputs.items()
                        for v in values
                    ]
                concat1 = xr.concat([sample_im] + special_ims, dim='c')

                main_idx_ = tr.get('main_idx', 0)
                if self.match_histograms:
                    nodata_mask = np.isnan(concat1)  # NOQA
                    tmp = np.nan_to_num(concat1)
                    # Hack: do before diff
                    from skimage import exposure  # NOQA
                    from skimage.exposure import match_histograms
                    main_idx_ = min(main_idx_, len(tmp) - 1)
                    reference = tmp[main_idx_]
                    for idx, raw_frame in enumerate(tmp):
                        if idx != main_idx_:
                            new_frame = match_histograms(raw_frame, reference, multichannel=True)
                            tmp[idx] = new_frame
                    concat1[...] = tmp

                # TODO: add the matching step somewhere around here

                if self.diff_inputs:
                    diff_ims = np.abs(concat1.diff(dim='t'))
                    diff_ims.coords.update({
                        'c': ['D' + s for s in diff_ims.coords['c'].data]
                    })
                    diff_ims = diff_ims.pad({'t': (1, 0)}).fillna(0)
                    concat2 = xr.concat([concat1, diff_ims], dim='c')
                else:
                    concat2 = concat1

                # TODO: multi-modal inputs
                requested_channel_order = self.input_channels.spec.split('|')
                final = concat2.sel(c=requested_channel_order)
                raw_frame_list = final
            else:
                # Access the sampled image and annotation data
                raw_frame_list = sample['im']

            # TODO: use this
            # TODO: read QA bands, input other special QA bands
            nodata_mask = np.isnan(raw_frame_list)  # NOQA
            raw_frame_list = np.nan_to_num(raw_frame_list)

            if not self.special_inputs and not self.diff_inputs:
                main_idx_ = tr.get('main_idx', 0)
                if self.match_histograms:
                    nodata_mask = np.isnan(raw_frame_list)  # NOQA
                    raw_frame_list = np.nan_to_num(raw_frame_list)
                    # Hack: do before diff
                    from skimage import exposure  # NOQA
                    from skimage.exposure import match_histograms
                    main_idx_ = min(main_idx_, len(raw_frame_list) - 1)
                    reference = raw_frame_list[main_idx_]
                    for idx, raw_frame in enumerate(raw_frame_list):
                        if idx != main_idx_:
                            new_frame = match_histograms(raw_frame, reference, multichannel=True)
                            raw_frame_list[idx] = new_frame

            raw_det_list = sample['annots']['frame_dets']
            raw_gids = sample['tr']['gids']

            # channel_keys = sample['tr']['_coords']['c'].values.tolist()
            # print('channel_keys = {!r}'.format(channel_keys))

            stream_specs = self.input_channels.streams()
            assert len(stream_specs) == 1, 'no late fusion yet'
            mode_key = self.input_channels.fuse().spec

            # print('mode_key = {!r}'.format(mode_key))
            # mode_key = '|'.join(channel_keys)

            # Break data down on a per-frame basis so we can apply image-based
            # augmentations.
            frame_items = []

            if self.sample_shape is None:
                input_dsize = raw_frame_list[0].shape[0:2][::-1]
            else:
                input_dsize = self.sample_shape[-2:][::-1]

            # hack for augmentation
            # TODO: make a nice "augmenter" pipeline
            do_hflip = False
            do_vflip = False
            if not self.disable_augmenter and self.mode == 'fit':
                def make_hflipper(width):
                    def hflip(pt):
                        new = np.hstack([width - pt[:, 0:1], pt[:, 1:2]])
                        return new
                    return hflip
                hflipper = make_hflipper(input_dsize[0])
                do_hflip = np.random.rand() > 0.5

                def make_vflipper(height):
                    def vflip(pt):
                        new = np.hstack([pt[:, 0:1], height - pt[:, 1:2]])
                        return new
                    return vflip
                vflipper = make_vflipper(input_dsize[1])
                do_vflip = np.random.rand() > 0.5

            prev_frame_cidxs = None

            if not self.inference_only:
                num_frames = len(raw_frame_list)
                frame0 = raw_frame_list[0]
                # from watch.utils import util_kwimage
                # Learn more from the center of the space-time patch
                time_weights = kwimage.gaussian_patch((1, num_frames))[0]
                time_weights = time_weights / time_weights.max()
                space_weights = util_kwimage.upweight_center_mask(frame0.shape[0:2])
                # time_weights[1:] = 0

            for time_idx, (frame, dets, gid) in enumerate(zip(raw_frame_list, raw_det_list, raw_gids)):
                img = self.sampler.dset.imgs[gid]

                frame = np.asarray(frame, dtype=np.float32)

                if do_hflip:
                    frame = np.fliplr(frame)
                    dets = dets.warp(hflipper)

                if do_vflip:
                    frame = np.flipud(frame)
                    dets = dets.warp(vflipper)

                # Resize the sampled window to the target space for the network
                frame, info = kwimage.imresize(frame, dsize=input_dsize,
                                               interpolation='linear',
                                               antialias=True,
                                               return_info=True)
                # Remember to apply any transform to the dets as well
                dets = dets.scale(info['scale'])
                dets = dets.translate(info['offset'])

                # ensure channel dim is not squeezed
                frame_hwc = kwarray.atleast_nd(frame, 3)
                # catch nans
                frame_hwc[np.isnan(frame_hwc)] = -1.
                # rearrange image axes for pytorch
                frame_chw = einops.rearrange(frame_hwc, 'h w c -> c h w')
                input_chw = frame_chw

                if not self.inference_only:
                    # allocate class masks
                    bg_idx = self.bg_idx

                    space_shape = frame.shape[:2]
                    frame_cidxs = np.full(space_shape, dtype=np.int32,
                                          fill_value=bg_idx)

                    ohe_shape = (len(self.classes),) + space_shape
                    frame_class_ohe = np.full(ohe_shape, dtype=np.uint8,
                                              fill_value=0)

                    # Ignore for saliency
                    saliency_ignore = np.full(space_shape, dtype=np.uint8,
                                              fill_value=0)

                    # Ignore for class
                    frame_class_ignore = np.full(space_shape, dtype=np.uint8,
                                                 fill_value=0)

                    # Rasterize frame targets
                    ann_polys = dets.data['segmentations'].to_polygon_list()
                    ann_aids = dets.data['aids']
                    ann_cids = dets.data['cids']
                    # Note: it is important to respect class indexes, ids, and name
                    # mappings
                    # TODO: layer ordering? Multiclass prediction?
                    for poly, aid, cid in zip(ann_polys, ann_aids, ann_cids):  # NOQA
                        cidx = self.classes.id_to_idx[cid]
                        catname = self.classes.id_to_node[cid]
                        if catname in self.background_classes:
                            pass
                        elif catname in self.ignore_classes:
                            poly.fill(saliency_ignore, value=1)
                            poly.fill(frame_class_ignore, value=1)
                        else:
                            if catname in self.undistinguished_classes:
                                poly.fill(frame_class_ohe[cidx], value=0)
                            else:
                                poly.fill(frame_class_ohe[cidx], value=1)

                    # Postprocess (Dilate?) the truth map
                    for cidx, class_map in enumerate(frame_class_ohe):
                        # class_map = util_kwimage.morphology(class_map, 'dilate', kernel=5)
                        frame_cidxs[class_map > 0] = cidx

                    if self.upweight_centers:
                        frame_weights = space_weights * time_weights[time_idx]
                    else:
                        frame_weights = 1.0

                    saliency_weights = frame_weights * (1 - saliency_ignore)
                    class_weights = frame_weights * (1 - frame_class_ignore)

                    # convert annotations into a change detection task suitable for
                    # the network.
                    if self.with_change:
                        if prev_frame_cidxs is None:
                            frame_change = None
                        else:
                            frame_change = (frame_cidxs != prev_frame_cidxs).astype(np.uint8)
                            # Clean up the change target
                            frame_change = util_kwimage.morphology(frame_change, 'open', kernel=3)
                            frame_change = torch.from_numpy(frame_change)
                    else:
                        frame_change = None

                # convert to torch
                frame_item = {
                    'gid': gid,
                    'date_captured': img.get('date_captured', ''),
                    'sensor_coarse': img.get('sensor_coarse', ''),
                    'modes': {
                        mode_key: torch.from_numpy(input_chw),
                    },
                    'change': None,
                    'class_idxs': None,
                    'ignore': None,
                }

                if not self.inference_only:
                    frame_item.update({
                        'change': frame_change,
                        'class_idxs': torch.from_numpy(frame_cidxs),
                        'ignore': torch.from_numpy(saliency_ignore),
                        'class_weights': torch.from_numpy(class_weights),
                        'saliency_weights': torch.from_numpy(saliency_weights),
                    })
                    prev_frame_cidxs = frame_cidxs
                frame_items.append(frame_item)

        # Only pass back some of the metadata (because I think torch
        # multiprocessing makes a new file descriptor for every Python object
        # or something like that)
        tr_subset = ub.dict_isect(sample['tr'], {
            'gids', 'space_slice', 'vidid',
        })

        item = {
            # TODO: breakup modes into different items
            'index': index,
            'frames': frame_items,
            'video_id': vidid,
            'video_name': video['name'],
            'tr': tr_subset
        }
        return item

    def cached_dataset_stats(self, num=None, num_workers=0, batch_size=2):
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
            ('depends_version', 7),  # bump if `compute_dataset_stats` changes
        ])
        workdir = None
        cacher = ub.Cacher('dset_mean', dpath=workdir, depends=depends)
        input_stats = cacher.tryload()
        if input_stats is None or ub.argflag('--force-recompute-stats'):
            input_stats = self.compute_dataset_stats(
                num, num_workers=num_workers, batch_size=batch_size)
            cacher.save(input_stats)
        return input_stats

    def compute_dataset_stats(self, num=None, num_workers=0, batch_size=2):
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
            >>> self = KWCocoVideoDataset(sampler, sample_shape=sample_shape, channels=None)
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

        Example:
            >>> # xdoctest: +REQUIRES(env:DVC_DPATH)
            >>> # Run the following tests on real watch data if DVC is available
            >>> from watch.tasks.fusion.datamodules.kwcoco_video_data import *  # NOQA
            >>> import ndsampler
            >>> import kwcoco
            >>> from watch.utils.util_data import find_smart_dvc_dpath
            >>> dvc_dpath = find_smart_dvc_dpath()
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
        """
        num = num if isinstance(num, int) and num is not True else 1000
        stats_idxs = kwarray.shuffle(np.arange(len(self)), rng=0)[0:min(num, len(self))]
        stats_subset = torch.utils.data.Subset(self, stats_idxs)

        # Hack: disable augmentation if we are doing that
        self.disable_augmenter = True

        loader = torch.utils.data.DataLoader(
            stats_subset,
            collate_fn=ub.identity, num_workers=num_workers, shuffle=True,
            batch_size=batch_size)

        # Track moving average of each fused channel stream
        channel_stats = {key: kwarray.RunningStats()
                         for key in self.input_channels.keys()}

        timer = ub.Timer().tic()
        timer.first = 1

        classes = self.classes
        num_classes = len(classes)
        bins = np.arange(num_classes + 1)
        total_freq = np.zeros(num_classes, dtype=np.int64)

        prog = ub.ProgIter(loader, desc='estimate mean/std')
        for batch_items in prog:
            for item in batch_items:
                for frame_item in item['frames']:

                    class_idxs = frame_item['class_idxs']
                    # print(np.unique(class_idxs))
                    item_freq = np.histogram(class_idxs.ravel(), bins=bins)[0]
                    total_freq += item_freq

                    for mode_code, mode_val in frame_item['modes'].items():
                        running = channel_stats[mode_code]
                        val = mode_val.numpy()
                        flags = np.isfinite(val)
                        if not np.all(flags):
                            # Hack it:
                            val[~flags] = 0
                        running.update(val.astype(np.float64))

            if timer.first or timer.toc() > 5:
                from watch.utils.slugify_ext import smart_truncate
                intermediate = ub.sorted_vals(ub.dzip(classes, total_freq), reverse=True)
                intermediate_text = ub.repr2(intermediate, compact=1)
                intermediate_text = smart_truncate(intermediate_text, max_length=40, trunc_loc=0.8)
                curr = ub.dict_isect(running.summarize(keepdims=False), {'mean', 'std', 'max', 'min'})
                curr = ub.map_vals(float, curr)
                text = ub.repr2(curr, compact=1, precision=1, nl=0) + ' ' + intermediate_text
                prog.set_postfix_str(text)
                timer.first = 0
                timer.tic()

        # Return the raw counts and let the model choose how to handle it
        class_freq = ub.dzip(classes, total_freq)

        input_stats = {}
        for key, running in channel_stats.items():
            perchan_stats = running.summarize(axis=(1, 2))
            input_stats[key] = {
                'mean': perchan_stats['mean'].round(3),  # only take 3 sigfigs
                'std': np.maximum(perchan_stats['std'], 1e-3).round(3),
            }
        self.disable_augmenter = False

        dataset_stats = {
            'input_stats': input_stats,
            'class_freq': class_freq,
        }
        return dataset_stats

    @profile
    def draw_item(self, item, item_output=None, combinable_extra=None,
                  max_channels=5, max_dim=256, norm_over_time=0,
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
            >>> canvas2 = self.draw_item(item, item_output, combinable_extra=combinable_extra, max_channels=1, overlay_on_image=0)
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
            >>> from watch.utils.util_data import find_smart_dvc_dpath
            >>> dvc_dpath = find_smart_dvc_dpath()
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
        """
        classes = self.classes

        combinable_channels = self.default_combinable_channels
        if combinable_extra is not None:
            combinable_channels += list(map(ub.oset, combinable_extra))

        truth_keys = ['class_idxs', 'change']
        weight_keys = ['class_weights', 'saliency_weights']

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

            unused_frame_chan_names_set = ub.oset(frame_chan_names)
            frame_available_chans = []
            for combinable in combinable_channels:
                if combinable.issubset(unused_frame_chan_names_set):
                    frame_available_chans.append(tuple(combinable))
                    unused_frame_chan_names_set.difference_update(combinable)
            frame_available_chans.extend(
                [(c,) for c in unused_frame_chan_names_set])

            frame_meta = {
                'full_mode_code': full_mode_code,
                'frame_idx': frame_idx,
                'frame_item': frame_item,
                'frame_chan_names': frame_chan_names,
                'frame_chan_datas': frame_chan_datas,
                'frame_available_chans': frame_available_chans,
                'frame_truth': frame_truth,
                'frame_weight': frame_weight,
                'sensor_coarse': frame_item.get('sensor_coarse', ''),
            }
            frame_metas.append(frame_meta)

        # Determine which frames to visualize For each frame choose N channels
        # such that common channels are aligned, visualize common channels in
        # the first rows and then fill with whatever is left
        chan_freq = ub.dict_hist(ub.flatten(frame_meta['frame_available_chans']
                                            for frame_meta in frame_metas))
        chan_priority = {k: (v, len(k), -idx) for idx, (k, v)
                         in enumerate(chan_freq.items())}
        for frame_meta in frame_metas:
            chan_keys = frame_meta['frame_available_chans']
            frame_priority = ub.dict_isect(chan_priority, chan_keys)
            chosen = ub.argsort(frame_priority, reverse=True)[0:max_channels]
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
                    'sensor_coarse': frame_meta['sensor_coarse'],
                }
                chan_rows.append(row)
            frame_meta['chan_rows'] = chan_rows
            assert len(chan_rows) > 0, 'no channels to draw on'

        # Normalize raw signal into visualizable range
        if norm_over_time:
            # Normalize all cells with the same channel code across time
            channel_cells = [cell for frame_meta in frame_metas for cell in frame_meta['chan_rows']]
            # chan_to_cells = ub.group_items(channel_cells, lambda c: (c['chan_code'])
            chan_to_cells = ub.group_items(channel_cells, lambda c: (c['chan_code'], c['sensor_coarse']))
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
                    try:
                        norm_signal = kwimage.normalize_intensity(raw_signal, nodata=0).copy()
                    except Exception:
                        norm_signal = raw_signal.copy()
                    # norm_signal = kwimage.normalize(raw_signal).copy()
                    norm_signal = np.nan_to_num(norm_signal)
                    norm_signal = util_kwimage.ensure_false_color(norm_signal)
                    norm_signal = kwimage.atleast_3channels(norm_signal)
                    row['norm_signal'] = norm_signal

        if draw_weights:
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
                    weight_overlay = kwimage.atleast_3channels(cell['raw'])
                    weight_overlay = kwimage.ensure_alpha_channel(weight_overlay)
                    weight_overlay[:, 3] = 0.5
                    cell['overlay'] = weight_overlay

        # Given prepared frame metadata, build a vertical stack of per-chanel
        # information, and then horizontally stack the timesteps.
        horizontal_stack = []

        truth_overlay_keys = set(ub.flatten([m['frame_truth'] for m in frame_metas]))
        weight_overlay_keys = set(ub.flatten([m['frame_weight'] for m in frame_metas]))

        for frame_meta in frame_metas:
            vertical_stack = []

            frame_idx = frame_meta['frame_idx']
            frame_item = frame_meta['frame_item']
            chan_rows = frame_meta['chan_rows']

            frame_truth = frame_meta['frame_truth']
            frame_weight = frame_meta['frame_weight']

            gid = frame_item['gid']

            # Build column headers
            header_dims = {'width': max_dim}
            header_part = util_kwimage.draw_header_text(
                image=header_dims, fit=False,
                text=f't={frame_idx} gid={gid}', color='salmon')
            vertical_stack.append(header_part)

            sensor_coarse = frame_item.get('sensor_coarse', '')
            if sensor_coarse:
                header_part = util_kwimage.draw_header_text(
                    image=header_dims, fit=False, text=f'{sensor_coarse}',
                    color='salmon')
                vertical_stack.append(header_part)

            date_captured = frame_item.get('date_captured', '')
            if date_captured:
                header_part = util_kwimage.draw_header_text(
                    header_dims, fit='shrink', text=f'{date_captured}',
                    color='salmon')
                vertical_stack.append(header_part)

            # Build truth / metadata overlays
            overlay_shape = ub.peek(frame_truth.values()).shape[0:2]

            # Create overlays for training objective targets
            overlay_items = []

            # Create the the true class label overlay
            overlay_key = 'class_idxs'
            if overlay_key in truth_overlay_keys:
                class_idxs = frame_truth.get(overlay_key, None)
                true_heatmap = kwimage.Heatmap(class_idx=class_idxs, classes=classes)
                class_overlay = true_heatmap.colorize('class_idx')
                class_overlay[..., 3] = 0.5
                overlay_items.append({
                    'overlay': class_overlay,
                    'label_text': 'true class',
                })

            # Create the true change label overlay
            overlay_key = 'change'
            if overlay_key in truth_overlay_keys:
                change_overlay = np.zeros(overlay_shape + (4,), dtype=np.float32)
                changes = frame_truth.get(overlay_key, None)
                if changes is not None:
                    change_overlay = kwimage.Mask(changes, format='c_mask').draw_on(change_overlay, color='lime')
                    change_overlay = kwimage.ensure_alpha_channel(change_overlay)
                    change_overlay[..., 3] = (changes > 0).astype(np.float32) * 0.5
                overlay_items.append({
                    'overlay': change_overlay,
                    'label_text': 'true change',
                })

            if draw_weights:
                weight_overlays = frame_meta['weight_overlays']
                for overlay_key in weight_overlay_keys:
                    weight_overlay_info = weight_overlays.get(overlay_key, None)
                    overlay_items.append({
                        'overlay': weight_overlay_info['overlay'],
                        'label_text': overlay_key,
                    })

            if not overlay_on_image:
                # Draw the overlays by themselves
                for overlay_info in overlay_items:
                    label_text = overlay_info['label_text']
                    row_canvas = overlay_info['overlay'][..., 0:3]
                    row_canvas = kwimage.imresize(row_canvas, max_dim=max_dim).clip(0, 1)
                    signal_bottom_y = 1  # hack: hardcoded
                    row_canvas = kwimage.draw_text_on_image(
                        row_canvas, label_text, (1, signal_bottom_y + 1),
                        valign='top', color='lime', border=3)
                    vertical_stack.append(row_canvas)

            for iterx, row in enumerate(chan_rows):
                layers = []
                label_text = None
                if overlay_on_image:
                    # Draw truth on the image itself
                    if iterx < len(overlay_items):
                        overlay_info = overlay_items[iterx]
                        layers.append(overlay_info['overlay'])
                        label_text = overlay_info['label_text']

                layers.append(row['norm_signal'])
                row_canvas = kwimage.overlay_alpha_layers(layers)[..., 0:3]

                row_canvas = kwimage.imresize(row_canvas, max_dim=max_dim).clip(0, 1)
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

            # TODO: clean up logic
            key = 'class_probs'
            overlay_index = 0
            if item_output and  key in item_output:
                if overlay_on_image:
                    norm_signal = chan_rows[overlay_index]['norm_signal']
                else:
                    norm_signal = np.zeros_like(chan_rows[min(overlay_index, len(chan_rows) - 1)]['norm_signal'])
                x = item_output[key][frame_idx]
                class_probs = einops.rearrange(x, 'h w c -> c h w')
                class_heatmap = kwimage.Heatmap(class_probs=class_probs, classes=classes)
                pred_part = class_heatmap.draw_on(norm_signal, with_alpha=0.7)
                # TODO: we might want to overlay the prediction on one or
                # all of the channels
                pred_part = kwimage.imresize(pred_part, max_dim=max_dim).clip(0, 1)
                pred_text = f'pred class t={frame_idx}'
                pred_part = kwimage.draw_text_on_image(
                    pred_part, pred_text, (1, 1), valign='top',
                    color='dodgerblue', border=3)
                vertical_stack.append(pred_part)

            key = 'change_probs'
            overlay_index = 1
            if item_output and  key in item_output:
                # Make a probability heatmap we can either display
                # independently or overlay on a rendered channel
                if frame_idx == 0:
                    # BIG RED X
                    h, w = vertical_stack[-1].shape[0:2]
                    pred_mask = kwimage.draw_text_on_image(
                        {'width': w, 'height': h}, 'X', org=(w // 2, h // 2),
                        valign='center', halign='center', fontScale=10,
                        color='red')
                    pred_part = pred_mask
                else:
                    pred_raw = item_output[key][frame_idx - 1]
                    # Draw predictions on the first item
                    pred_mask = kwimage.make_heatmask(pred_raw)
                    norm_signal = chan_rows[min(overlay_index, len(chan_rows) - 1)]['norm_signal']
                    if overlay_on_image:
                        norm_signal = norm_signal
                    else:
                        norm_signal = np.zeros_like(norm_signal)
                    pred_layers = [pred_mask, norm_signal]
                    pred_part = kwimage.overlay_alpha_layers(pred_layers)
                    # TODO: we might want to overlay the prediction on one or
                    # all of the channels
                    pred_part = kwimage.imresize(pred_part, max_dim=max_dim).clip(0, 1)
                    pred_text = f'pred change t={frame_idx}'
                    pred_part = kwimage.draw_text_on_image(
                        pred_part, pred_text, (1, 1), valign='top',
                        color='dodgerblue', border=3)
                vertical_stack.append(pred_part)

            key = 'saliency_probs'
            if item_output and  key in item_output:
                if overlay_on_image:
                    norm_signal = chan_rows[0]['norm_signal']
                else:
                    norm_signal = np.zeros_like(chan_rows[min(overlay_index, len(chan_rows) - 1)]['norm_signal'])
                x = item_output[key][frame_idx]
                saliency_probs = einops.rearrange(x, 'h w c -> c h w')
                saliency_heatmap = kwimage.Heatmap(class_probs=saliency_probs)
                pred_part = saliency_heatmap.draw_on(norm_signal, with_alpha=0.7)
                # TODO: we might want to overlay the prediction on one or
                # all of the channels
                pred_part = kwimage.imresize(pred_part, max_dim=max_dim).clip(0, 1)
                pred_text = f'pred saliency t={frame_idx}'
                pred_part = kwimage.draw_text_on_image(
                    pred_part, pred_text, (1, 1), valign='top',
                    color='dodgerblue', border=3)
                vertical_stack.append(pred_part)

            vertical_stack = [kwimage.ensure_uint255(p) for p in vertical_stack]
            frame_canvas = kwimage.stack_images(vertical_stack, overlap=-3)
            horizontal_stack.append(frame_canvas)

        canvas = kwimage.stack_images(horizontal_stack, axis=1, overlap=-5)
        canvas = canvas[..., 0:3]  # drop alpha
        canvas = kwimage.ensure_uint255(canvas)  # convert to uint8

        width = canvas.shape[1]

        vid_text = f'video: {item["video_id"]} - {item["video_name"]}'
        vid_header = kwimage.draw_text_on_image(
            {'width': width}, vid_text, org=(width // 2, 3), valign='top',
            halign='center', color='pink')

        canvas = kwimage.stack_images([vid_header, canvas], axis=0, overlap=-3)
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


def sample_video_spacetime_targets(dset, window_dims, window_overlap=0.0,
                                   negative_classes=None, keepbound=False,
                                   exclude_sensors=None,
                                   time_sampling='hard+distribute',
                                   time_span='2y', use_annot_info=True):
    """
    Example:
        >>> # xdoctest: +REQUIRES(env:DVC_DPATH)
        >>> import os
        >>> from watch.tasks.fusion.datamodules.kwcoco_video_data import *  # NOQA
        >>> from watch.utils.util_data import find_smart_dvc_dpath
        >>> dvc_dpath = find_smart_dvc_dpath()
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
        >>> from watch.utils.util_data import find_smart_dvc_dpath
        >>> dvc_dpath = find_smart_dvc_dpath()
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
        >>> window_dims = (3, 96, 96)
        >>> keepbound = False
        >>> sample_grid1 = sample_video_spacetime_targets(dset, window_dims, window_overlap, time_sampling='soft+distribute')
        >>> sample_grid2 = sample_video_spacetime_targets(dset, window_dims, window_overlap, time_sampling='contiguous+pairwise')

        ub.peek(sample_grid1['vidid_to_time_sampler'].values()).show_summary(fnum=1)
        ub.peek(sample_grid2['vidid_to_time_sampler'].values()).show_summary(fnum=2)
        _ = xdev.profile_now(sample_video_spacetime_targets)(dset, window_dims, window_overlap)
    """
    # Create a sliding window object for each specific image (because they may
    # have different sizes, technically we could memoize this)
    import pyqtree
    from watch.tasks.fusion.datamodules import temporal_sampling as tsm  # NOQA
    from watch.utils import util_kwarray

    window_space_dims = window_dims[1:3]
    window_time_dims = window_dims[0]
    print('window_time_dims = {!r}'.format(window_time_dims))
    keepbound = False
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
    affinity_type_parts = parts & {'hard', 'hardish', 'contiguous', 'soft'}
    affinity_type = '+'.join(list(affinity_type_parts))
    update_rule = '+'.join(list(parts - affinity_type_parts))
    if not update_rule:
        update_rule = 'distribute'

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

        time_sampler = tsm.TimeWindowSampler.from_coco_video(
            dset, video_id, gids=video_gids, time_window=window_time_dims,
            affinity_type=affinity_type, update_rule=update_rule,
            name=video_info['name'], time_span=time_span)
        time_sampler.video_gids = np.array(video_gids)
        time_sampler.determenistic = True

        if use_annot_info:
            qtree = pyqtree.Index((0, 0, video_info['width'], video_info['height']))
            qtree.aid_to_tlbr = {}
            qtree.idx_to_tlbr = {}
            tid_to_infos = ub.ddict(list)
            video_aids = dset.images(video_gids).annots.lookup('id')
            all_vid_tlbr = []
            for aids, gid in zip(video_aids, video_gids):
                warp_vid_from_img = kwimage.Affine.coerce(
                    dset.index.imgs[gid]['warp_img_to_vid'])
                img_info = dset.index.imgs[gid]
                frame_index = img_info['frame_index']
                tids = dset.annots(aids).lookup('track_id')
                cids = dset.annots(aids).lookup('category_id')
                for tid, aid, cid in zip(tids, aids, cids):
                    imgspace_box = kwimage.Boxes([dset.index.anns[aid]['bbox']], 'xywh')
                    vidspace_box = imgspace_box.warp(warp_vid_from_img)
                    tlbr_box = vidspace_box.to_tlbr().data[0]
                    all_vid_tlbr.append(tlbr_box)
                    tid_to_infos[tid].append({
                        'gid': gid,
                        'cid': cid,
                        'frame_index': frame_index,
                        'vidspace_box': tlbr_box,
                        'cname': dset._resolve_to_cat(cid)['name'],
                        'aid': aid,
                    })

            if len(all_vid_tlbr):
                unique_tlbr = util_kwarray.unique_rows(np.array(all_vid_tlbr))
            else:
                unique_tlbr = []
            for idx, tlbr_box in enumerate(unique_tlbr):
                qtree.insert(idx, tlbr_box)
                qtree.idx_to_tlbr[idx] = tlbr_box

            # tid_to_dframe = ub.map_vals(kwarray.DataFrameLight.from_dict, tid_to_infos)
            # for track_dframe in tid_to_dframe.values():
            #     track_dframe['gid'] = np.array(track_dframe['gid'])
            #     track_dframe['frame_index'] = np.array(track_dframe['frame_index'])
            #     # Precompute for speed
            #     track_boxes = kwimage.Boxes(np.array(track_dframe['vidspace_box']), 'ltrb')
            #     track_dframe['track_pairwise_ious'] = track_boxes.ious(track_boxes)
            #     track_dframe['track_boxes'] = track_boxes

        # Compute determenistic base sample
        main_idx_to_gids = {
            main_idx: list(ub.take(video_gids, time_sampler.sample(main_idx)))
            for main_idx in time_sampler.main_indexes
        }

        for space_region in ub.ProgIter(list(slider)):
            y_sl, x_sl = space_region

            # Find all annotations that pass through this spatial region
            if use_annot_info:
                vid_box = kwimage.Boxes.from_slice((y_sl, x_sl))
                query = vid_box.to_tlbr().data[0]
                isect_aids = qtree.intersect(query)
                # isect_aids = set(isect_aids)

            for main_idx, gids in main_idx_to_gids.items():
                if use_annot_info:
                    if isect_aids:
                        has_annot = True
                        # has_annot = any(
                        #     bool(isect_aids & dset.index.gid_to_aids[gid])
                        #     for gid in gids)
                    else:
                        has_annot = False
                    if has_annot:
                        positive_idxs.append(len(targets))
                    else:
                        # Hack: exclude all annotated regions from negative sampling
                        negative_idxs.append(len(targets))

                # Reselect the keyframes if we are overlaping a nodata region?
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

                targets.append({
                    'main_idx': main_idx,
                    'video_id': video_id,
                    'gids': gids,
                    'space_slice': space_region,
                    # 'changes': ','.join(changes),
                    # 'region_tracks': region_tracks,
                })

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
        >>> from watch.utils.util_data import find_smart_dvc_dpath
        >>> dvc_dpath = find_smart_dvc_dpath()
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
