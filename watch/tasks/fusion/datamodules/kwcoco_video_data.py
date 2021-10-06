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
import math
import datetime
import itertools as it
from dateutil import parser
from kwcoco import channel_spec
from torch.utils import data
from watch.tasks.fusion import utils
from watch.utils import kwcoco_extensions
from watch.utils import util_iter
from watch.utils import util_kwimage
from watch.utils import util_kwarray
from watch.utils.lightning_ext import util_globals

# __all__ = ['KWCocoVideoDataModule', 'KWCocoVideoDataset']

try:
    import xdev
    profile = xdev.profile
except Exception:
    profile = ub.identity


# FIXME: Hard-coded category aliases.
# The correct way to handle these would be to have some information in the
# kwcoco category dictionary that specifies how the categories should be
# interpreted.
_HEURISTIC_CATEGORIES = {

    'background': {'background', 'No Activity', 'Post Construction'},

    'pre_background': {'No Activity'},
    'post_background': {'Post Construction'},

    'ignore': {'ignore', 'Unknown', 'clouds'},
}


class KWCocoVideoDataModule(pl.LightningDataModule):
    """
    Prepare the kwcoco dataset as torch video datamodules

    Example:
        >>> # xdoctest: +REQUIRES(env:DVC_DPATH)
        >>> # Run the following tests on real watch data if DVC is available
        >>> from watch.tasks.fusion.datamodules.kwcoco_video_data import *  # NOQA
        >>> from os.path import join
        >>> import os
        >>> _default = ub.expandpath('$HOME/data/dvc-repos/smart_watch_dvc')
        >>> dvc_dpath = os.environ.get('DVC_DPATH', _default)
        >>> coco_fpath = join(dvc_dpath, 'drop1-S2-L8-aligned/data.kwcoco.json')
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
        >>> self.setup("fit")
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
        >>> self.setup("fit")
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
        exclude_sensors=['L8'],  # NOQA
        channels=None,
        batch_size=4,
        num_workers=4,
        preprocessing_step=None,
        tfms_channel_subset=None,
        normalize_inputs=False,
        diff_inputs=False,
        verbose=1,
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
        self.time_steps = time_steps
        self.chip_size = chip_size
        self.time_overlap = time_overlap
        self.chip_overlap = chip_overlap
        self.neg_to_pos_ratio = neg_to_pos_ratio
        self.channels = channels
        self.batch_size = batch_size
        if isinstance(num_workers, str):
            if num_workers == 'auto':
                num_workers = 'avail-2'

            # input normalization
            num_workers = num_workers.replace('available', 'avail')
            base_workers = None

            prefix = 'avail'
            if num_workers.startswith(prefix):
                base_workers = util_globals.request_cpus(max_load=0.5)
                suffix = num_workers[len(prefix):]

            prefix = 'all'
            if num_workers.startswith(prefix):
                import psutil
                base_workers = psutil.cpu_count()
                suffix = num_workers[len(prefix):]

            if base_workers is None:
                raise KeyError(num_workers)

            if suffix:
                expr = '{}{}'.format(base_workers, suffix)
                if len(expr) > 8:
                    raise Exception(
                        'num-workers-hueristic should be small text. '
                        'We want to disallow attempts at crashing python '
                        'by feeding nasty input into eval '
                    )
                # FIME: eval is not very safe, add numexpr dependency instead
                # import numexpr
                # numexpr.evaluate('3 - 2')
                num_workers = max(eval(expr, {}, {}), 0)
            else:
                num_workers = base_workers
            print('Choose num_workers = {!r}'.format(num_workers))

        self.num_workers = num_workers
        self.preprocessing_step = preprocessing_step
        self.normalize_inputs = normalize_inputs
        self.time_sampling = time_sampling
        self.exclude_sensors = exclude_sensors
        self.diff_inputs = diff_inputs

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
        parser = parent_parser.add_argument_group("kwcoco_video_data")
        parser.add_argument("--train_dataset", default=None, help='path to the train kwcoco file')
        parser.add_argument("--vali_dataset", default=None, help='path to the validation kwcoco file')
        parser.add_argument("--test_dataset", default=None, help='path to the test kwcoco file')
        parser.add_argument("--time_steps", default=2, type=int)
        parser.add_argument("--chip_size", default=128, type=int)
        parser.add_argument("--time_overlap", default=0.0, type=float, help='fraction of time steps to overlap')
        parser.add_argument("--chip_overlap", default=0.1, type=float, help='fraction of space steps to overlap')
        parser.add_argument("--neg_to_pos_ratio", default=1.0, type=float, help='maximum ratio of samples with no annotations to samples with annots')
        parser.add_argument("--time_sampling", default='contiguous', type=str, help=ub.paragraph(
            '''
            Strategy for expanding the time window across non-contiguous frames.
            Can be auto, contiguous, dilate_template, or dilate_affinity
            '''))
        parser.add_argument("--exclude_sensors", type=partial(smartcast, astype=list), help='comma delimited list of sensors to avoid, such as S2 or L8')
        parser.add_argument("--channels", default=None, type=str, help='channels to use should be ChannelSpec coercable')
        parser.add_argument("--batch_size", default=4, type=int)
        parser.add_argument("--num_workers", default=4, type=int)

        parser.add_argument(
            "--normalize_inputs", default=True, help=ub.paragraph(
                '''
                if True, computes the mean/std for this dataset on each mode
                so this can be passed to the model.
                '''))
        parser.add_argument(
            "--diff_inputs", default=False, type=smartcast, help=ub.paragraph(
                '''
                if True, also includes a difference between consecutive frames
                in the inputs produced.
                '''))
        return parent_parser

    def setup(self, stage):
        if self.verbose:
            print('Setup DataModule: stage = {!r}'.format(stage))
        if stage == "fit" or stage is None:
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
                window_overlap=(self.time_overlap, self.chip_overlap, self.chip_overlap),
                channels=self.channels,
                neg_to_pos_ratio=self.neg_to_pos_ratio,
                time_sampling=self.time_sampling,
                diff_inputs=self.diff_inputs,
                exclude_sensors=self.exclude_sensors,
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
                    window_overlap=(self.time_overlap, self.chip_overlap, self.chip_overlap),
                    channels=self.channels,
                    time_sampling=self.time_sampling,
                    mode='vali',
                    neg_to_pos_ratio=0,
                    diff_inputs=self.diff_inputs,
                    exclude_sensors=self.exclude_sensors,
                )
                self.torch_datasets['vali'] = vali_dataset
                ub.inject_method(self, lambda self: self._make_dataloader('vali', shuffle=False), 'val_dataloader')

        if stage == "test" or stage is None:
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
                window_overlap=(self.time_overlap, self.chip_overlap, self.chip_overlap),
                channels=self.channels,
                mode='test',
                diff_inputs=self.diff_inputs,
                exclude_sensors=self.exclude_sensors,
            )

            ub.inject_method(self, lambda self: self._make_dataloader('test', shuffle=False), 'test_dataloader')

        print('self.torch_datasets = {}'.format(ub.repr2(self.torch_datasets, nl=1)))

    def _make_dataloader(self, stage, shuffle=False):
        if self.num_workers > 0:
            util_globals.request_nofile_limits()
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
        """
        import kwimage
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
                for k in ['change_probs', 'class_probs']:
                    if k in outputs:
                        item_output[k] = outputs[k][item_idx].data.cpu().numpy()

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
        >>> coco_dset = kwcoco.CocoDataset.demo('vidshapes2-multispectral', num_frames=5)
        >>> sampler = ndsampler.CocoSampler(coco_dset)
        >>> channels = 'B10|B8a|B1|B8'
        >>> sample_shape = (3, 128, 128)
        >>> self = KWCocoVideoDataset(sampler, sample_shape=sample_shape, channels=channels)
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
        >>> _default = ub.expandpath('$HOME/data/dvc-repos/smart_watch_dvc')
        >>> dvc_dpath = os.environ.get('DVC_DPATH', _default)
        >>> coco_fpath = join(dvc_dpath, 'drop1-S2-L8-aligned/data.kwcoco.json')
        >>> coco_dset = kwcoco.CocoDataset(coco_fpath)
        >>> sampler = ndsampler.CocoSampler(coco_dset)
        >>> sample_shape = (7, 128, 128)
        >>> self = KWCocoVideoDataset(sampler, sample_shape=sample_shape, channels=None)
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
        >>> _default = ub.expandpath('$HOME/data/dvc-repos/smart_watch_dvc')
        >>> dvc_dpath = os.environ.get('DVC_DPATH', _default)
        >>> coco_fpath = join(dvc_dpath, 'drop1-S2-L8-aligned/data.kwcoco.json')
        >>> coco_dset = kwcoco.CocoDataset(coco_fpath)
        >>> sampler = ndsampler.CocoSampler(coco_dset)
        >>> self = KWCocoVideoDataset(
        >>>     sampler,
        >>>     sample_shape=(5, 128, 128),
        >>>     window_overlap=0,
        >>>     channels="blue|green|red|nir|swir16",
        >>>     neg_to_pos_ratio=0, time_sampling='auto', diff_inputs=0, mode='train'
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
        mode="fit",
        window_overlap=0,
        transform=None,
        neg_to_pos_ratio=1.0,
        time_sampling='auto',
        diff_inputs=False,
        exclude_sensors=None,
    ):

        self._hueristic_background_classnames = _HEURISTIC_CATEGORIES['background']
        self._heuristic_ignore_classnames = _HEURISTIC_CATEGORIES['ignore']

        if channels is None:
            # Hack to use all channels in the first image.
            # (Does not handle heterogeneous channels yet)
            chan_info = kwcoco_extensions.coco_channel_stats(sampler.dset)
            channels = chan_info['all_channels']
        channels = channel_spec.ChannelSpec.coerce(channels).normalize()

        if transform is not None:
            raise Exception('I do not like injecting the transforms')

        if time_sampling == 'auto':
            time_sampling = 'dilate_template'

        if mode == 'test':
            # In test mode we have to sample everything for BAS
            # (TODO: for activity clf, we should only focus on candidate regions)
            new_sample_grid = sample_video_spacetime_targets(
                sampler.dset, window_dims=sample_shape,
                window_overlap=window_overlap,
                keepbound=True,
                use_annot_info=False,
                exclude_sensors=exclude_sensors,
                time_sampling=time_sampling,
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
            )

            n_pos = len(new_sample_grid["positives_indexes"])
            n_neg = len(new_sample_grid["negatives_indexes"])

            max_neg = min(int(max(0, (neg_to_pos_ratio * n_pos))), n_neg)
            if n_neg > max_neg:
                print('restrict to max_neg = {!r}'.format(max_neg))

            # We have too many negatives, so we are going to "group" negatives
            # and when we select one we will really just randomly select from
            # within the pool
            if max_neg > 0:
                negative_pool = list(util_iter.chunks(new_sample_grid["negatives_indexes"], nchunks=max_neg))
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

        # TODO: the set of "valid" background classnames should be defined
        # by the inputs, not hard-coded in the dataloader. This can either be a
        # list of names provided to the training config, or something baked
        # into the kwcoco spec marking a class as some type of "background"

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
        if self.diff_inputs:
            # Add frame_differences between channels
            self.input_channels = kwcoco.channel_spec.ChannelSpec.coerce(','.join(
                ['|'.join([s + p for p in part for s in ['', 'D']])
                 for part in self.channels.parse().values()]))
        else:
            self.input_channels = channels
        self.mode = mode

        self.augment = False
        self.disable_augmenter = False

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
            >>> sample_shape = (5, 530, 610)
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
            >>>     channels="blue|green|red|nir|swir16",
            >>>     neg_to_pos_ratio=0, time_sampling='auto', diff_inputs=True
            >>> )
            >>> item = self[5]
            >>> canvas = self.draw_item(item, max_channels=10, overlay_on_image=0)
            >>> # xdoctest: +REQUIRES(--show)
            >>> import kwplot
            >>> kwplot.autompl()
            >>> kwplot.imshow(canvas)
            >>> kwplot.show_if_requested()

            kwplot.imshow(self.draw_item(self[4], max_channels=10, overlay_on_image=0))
        """

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
                import random
                neg_chunk = self.negative_pool[self.n_pos - index]
                tr_idx = random.choice(neg_chunk)
            tr = self.new_sample_grid['targets'][tr_idx]

        # get positive sample definition
        if self.channels:
            tr["channels"] = self.channels

        # TODO: perterb the spatial and time sample coordinates
        do_shift = False
        # collect sample
        sampler = self.sampler
        tr['as_xarray'] = False
        tr['use_experimental_loader'] = 1
        if not self.disable_augmenter and self.mode == 'fit':
            # do_shift = np.random.rand() > 0.5
            do_shift = True
        if not do_shift:
            # collect sample
            sample = sampler.load_sample(tr, padkw={'constant_values': np.nan})
        else:
            rng = kwarray.ensure_rng(132)
            tr_ = tr.copy()
            aff = kwimage.Affine.coerce(offset=rng.randint(-8, 8, size=2))
            space_box = kwimage.Boxes.from_slice(tr['space_slice']).warp(aff).quantize()
            tr_['space_slice'] = space_box.astype(int).to_slices()[0]
            sample = sampler.load_sample(tr_, padkw={'constant_values': np.nan})

        # Access the sampled image and annotation data
        raw_frame_list = sample['im']

        # TODO: use this
        nodata_mask = np.isnan(raw_frame_list)  # NOQA
        raw_frame_list = np.nan_to_num(raw_frame_list)

        raw_det_list = sample['annots']['frame_dets']
        raw_gids = sample['tr']['gids']

        # channel_keys = sample['tr']['_coords']['c'].values.tolist()
        # print('channel_keys = {!r}'.format(channel_keys))

        mode_lists = list(self.input_channels.values())
        assert len(mode_lists) == 1, 'no late fusion yet'
        mode_key = '|'.join(mode_lists[0])

        # print('mode_key = {!r}'.format(mode_key))
        # mode_key = '|'.join(channel_keys)

        # Break data down on a per-frame basis so we can apply image-based
        # augmentations.
        frame_items = []

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
        prev_frame_chw = None

        for frame, dets, gid in zip(raw_frame_list, raw_det_list, raw_gids):
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

            # allocate class masks
            bg_idx = self.bg_idx

            space_shape = frame.shape[:2]

            frame_cidxs = np.full(space_shape, dtype=np.int32,
                                  fill_value=bg_idx)

            ohe_shape = (len(self.classes),) + space_shape
            frame_class_ohe = np.full(ohe_shape, dtype=np.uint8,
                                      fill_value=0)

            frame_ignore = np.full(space_shape, dtype=np.uint8,
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
                    poly.fill(frame_ignore, value=1)
                else:
                    poly.fill(frame_class_ohe[cidx], value=1)

            # Dilate the truth map
            for cidx, class_map in enumerate(frame_class_ohe):
                class_map = util_kwimage.morphology(class_map, 'dilate', kernel=5)
                frame_cidxs[class_map > 0] = cidx

            # ensure channel dim is not squeezed
            frame_hwc = kwarray.atleast_nd(frame, 3)
            # catch nans
            frame_hwc[np.isnan(frame_hwc)] = -1.
            # rearrange image axes for pytorch
            frame_chw = einops.rearrange(frame_hwc, 'h w c -> c h w')

            if self.diff_inputs:
                if prev_frame_chw is not None:
                    frame_diff = np.abs(frame_chw - prev_frame_chw)
                    # kwimage.normalize(frame_chw) -
                    # kwimage.normalize(prev_frame_chw))
                else:
                    frame_diff = np.zeros_like(frame_chw)
                """
                # Check:
                hwc = kwimage.ensure_float01(kwimage.grab_test_image('astro'))
                hwc2 = kwimage.gaussian_blur(hwc)
                v1 = einops.rearrange(hwc, 'h w c -> c h w')
                v2 = einops.rearrange(hwc2, 'h w c -> c h w')
                diff = np.abs(v1 - v2)

                kwplot.imshow(kwimage.stack_images(v1))
                kwplot.imshow(kwimage.stack_images(v2))
                kwplot.imshow(kwimage.stack_images(diff))

                input_chw = einops.rearrange([v1, diff], '(c1 c2) c h w -> (c1 c c2) h w', c1=1)
                kwplot.imshow(kwimage.stack_images(input_chw))
                """
                # Interlace/Interweave the diffs and the channels
                parts = list(ub.flatten(zip(frame_chw, frame_diff)))
                input_chw = np.stack(parts, axis=0)
                # input_chw = einops.rearrange([frame_chw, frame_diff], '(c1 c2) c h w -> (c1 c c2) h w', c1=1)
            else:
                input_chw = frame_chw
                pass

            # convert annotations into a change detection task suitable for
            # the network.
            if prev_frame_cidxs is None:
                frame_change = None
            else:
                frame_change = (frame_cidxs != prev_frame_cidxs).astype(np.uint8)
                # Clean up the change target
                frame_change = util_kwimage.morphology(frame_change, 'open', kernel=3)
                frame_change = torch.from_numpy(frame_change)

            # convert to torch
            frame_item = {
                'gid': gid,
                'date_captured': img.get('date_captured', ''),
                'sensor_coarse': img.get('sensor_coarse', ''),
                'modes': {
                    mode_key: torch.from_numpy(input_chw),
                },
                'change': frame_change,
                'class_idxs': torch.from_numpy(frame_cidxs),
                'ignore': torch.from_numpy(frame_ignore),
            }
            prev_frame_cidxs = frame_cidxs
            prev_frame_chw = frame_chw

            frame_items.append(frame_item)

        vidid = sample['tr']['vidid']
        video = self.sampler.dset.index.videos[vidid]

        # Only pass back some of the metadata (because I think torch
        # multiprocessing makes a new file descriptor for every Python object
        # or something like that)
        tr_subset = ub.dict_isect(sample['tr'], {
            'gids', 'space_slice', 'annot_idx', 'slices', 'vidid',
        })

        item = {
            # TODO: breakup modes into different items
            "index": index,
            "frames": frame_items,
            "video_id": sample['tr']['vidid'],
            "video_name": video['name'],
            "tr": tr_subset
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
            ('depends_version', 6),  # bump if `compute_dataset_stats` changes
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
            >>> coco_dset = kwcoco.CocoDataset.demo('vidshapes2-multispectral', num_frames=5)
            >>> sampler = ndsampler.CocoSampler(coco_dset)
            >>> sample_shape = (2, 128, 128)
            >>> self = KWCocoVideoDataset(sampler, sample_shape=sample_shape, channels=None)
            >>> self.compute_dataset_stats()

        Example:
            >>> from watch.tasks.fusion.datamodules.kwcoco_video_data import *  # NOQA
            >>> import ndsampler
            >>> import kwcoco
            >>> coco_dset = kwcoco.CocoDataset.demo('vidshapes2')
            >>> sampler = ndsampler.CocoSampler(coco_dset)
            >>> sample_shape = (2, 128, 128)
            >>> self = KWCocoVideoDataset(sampler, sample_shape=sample_shape, channels=None)
            >>> self.compute_dataset_stats()

        CommandLine:
            DVC_DPATH=$HOME/data/dvc-repos/smart_watch_dvc xdoctest -m watch.tasks.fusion.datamodules.kwcoco_video_data KWCocoVideoDataset.compute_dataset_stats:1

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
            >>> coco_fpath = join(dvc_dpath, 'drop1-S2-L8-aligned/data.kwcoco.json')
            >>> #coco_fpath = join(dvc_dpath, 'drop1-S2-L8-aligned/rutgers_material_seg.kwcoco.json')
            >>> coco_dset = kwcoco.CocoDataset(coco_fpath)
            >>> sampler = ndsampler.CocoSampler(coco_dset)
            >>> sample_shape = (3, 96, 96)
            >>> channels = 'blue|green|red|nir|swir16'
            >>> #channels = 'rice_field|cropland|water|inland_water|river_or_stream|sebkha|snow_or_ice_field|bare_ground|sand_dune|built_up|grassland|brush|forest|wetland|road'
            >>> #channels = 'matseg_0|matseg_1|matseg_2|matseg_3|matseg_4'
            >>> self = KWCocoVideoDataset(sampler, sample_shape=sample_shape, channels=channels, neg_to_pos_ratio=1.0)
            >>> item = self[100]
            >>> #self.compute_dataset_stats(num=10)
            >>> num_workers = 14
            >>> num = 1000
            >>> batch_size = 6
            >>> self.compute_dataset_stats(num=num, num_workers=num_workers, batch_size=batch_size)

        Ignore:
            # TODO: profile and optimize loading in ndsampler / kwcoco
            _ = xdev.profile_now(self.__getitem__)(0)
            _ = xdev.profile_now(self.compute_dataset_stats)(num=10, num_workers=4, batch_size=1)
            tr = self.new_sample_grid['targets'][0]
            tr['channels'] = self.channels
            _ = xdev.profile_now(self.sampler.load_sample)(tr)
            pad = None
            padkw = {}
            tr['use_experimental_loader'] = 0
            item1 = xdev.profile_now(self.sampler._load_slice)(tr, pad, padkw)
            item1['im'].shape
            item1['im'].sum()
            print(item1['im'].mean(axis=(1, 2)))

            tr['use_experimental_loader'] = 1
            item2 = xdev.profile_now(self.sampler._load_slice)(tr, pad, padkw)
            item2['im'].shape
            print(item2['im'].mean(axis=(1, 2)))

            import timerit
            ti = timerit.Timerit(10, bestof=2, verbose=2)
            for timer in ti.reset('time'):
                with timer:
                    tr['use_experimental_loader'] = 0
                    (self.sampler._load_slice)(tr, pad, padkw)

            for timer in ti.reset('time'):
                with timer:
                    tr['use_experimental_loader'] = 1
                    (self.sampler._load_slice)(tr, pad, padkw)
        """
        num = num if isinstance(num, int) and num is not True else 1000
        stats_idxs = kwarray.shuffle(np.arange(len(self)), rng=0)[0:min(num, len(self))]
        stats_subset = torch.utils.data.Subset(self, stats_idxs)

        # Hack: disable augmentation if we are doing that
        self.disable_augmenter = True
        if num_workers > 0:
            util_globals.request_nofile_limits()
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
                  overlay_on_image=True):
        """
        Visualize an item produced by this DataSet.

        Args:
            item (Dict): An item returned from the torch Dataset.
                (It is a dict right? { ಠ ︿ ಠ } )

            overlay_on_image (bool):
                if True, the truth and prediction is drawn on top of
                an image, otherwise it is drawn on a black image.

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
        """
        classes = self.classes

        # Hacks: combinable channels can be visualized as RGB images.
        # The only reason this is a hack is because of the hardcoded names
        # otherwise it is a cool feature.
        default_combinable_channels = [
            ub.oset(['red', 'green', 'blue']),
            ub.oset(['Dred', 'Dgreen', 'Dblue']),
            ub.oset(['r', 'g', 'b']),
            ub.oset(['B04', 'B03', 'B02']),  # for onera
        ]
        combinable_channels = default_combinable_channels
        if combinable_extra is not None:
            combinable_channels += list(map(ub.oset, combinable_extra))

        # Prepare metadata on each frame
        frame_metas = []
        for frame_idx, frame_item in enumerate(item['frames']):
            class_idxs = frame_item['class_idxs'].data.cpu().numpy()
            changes = frame_item['change']
            if changes is None:
                changes = np.zeros_like(class_idxs)
            else:
                changes = changes.data.cpu().numpy()

            # hack just use one of the modes, todo: use them all
            full_mode_code = ','.join(list(frame_item['modes'].keys()))
            mode_code, mode_data = ub.peek(frame_item['modes'].items())
            chan_names = mode_code.split('|')
            frame_chw = mode_data.data.cpu().numpy()

            chan_name_to_idx = {
                chan_name: chan_idx
                for chan_idx, chan_name in enumerate(chan_names)
            }

            unused_names = set(chan_name_to_idx)
            # unused_chan_idx = ub.oset(chan_name_to_idx.values())

            combos_to_use = []
            for combinable in combinable_channels:
                if combinable.issubset(unused_names):
                    combo = [chan_name_to_idx[c] for c in combinable]
                    if len(combos_to_use) < max_channels:
                        combos_to_use.append(combo)
                    unused_names.difference_update(combinable)

            available = sorted(ub.dict_subset(chan_name_to_idx, unused_names).values())

            first_to_use = available[0:max(0, max_channels - len(combos_to_use))]
            chans_to_use = first_to_use + combos_to_use

            # Prepare and normalize the channels for visualization
            chan_rows = []
            for chanxs in chans_to_use:
                if not isinstance(chanxs, list):
                    chanxs = [chanxs]
                chan_name = '|'.join([chan_names[x] for x in chanxs])
                raw_signal = frame_chw[chanxs].transpose(1, 2, 0)
                # normalize across channel?
                signal_text = f'c={ub.repr2(chanxs, nobr=1, compact=1, trailsep=0)}:{chan_name}'
                row = {
                    'raw_signal': raw_signal,
                    'signal_text': signal_text,
                }
                if not norm_over_time:
                    norm_signal = kwimage.normalize_intensity(raw_signal).copy()
                    # norm_signal = kwimage.normalize(raw_signal).copy()
                    norm_signal = kwimage.atleast_3channels(norm_signal)
                    norm_signal = np.nan_to_num(norm_signal)
                    row['norm_signal'] = norm_signal
                chan_rows.append(row)

            assert len(chan_rows) > 0, 'no channels to draw on'

            frame_meta = {
                'full_mode_code': full_mode_code,
                'changes': changes,
                'class_idxs': class_idxs,
                'frame_idx': frame_idx,
                'frame_item': frame_item,
                'chan_rows': chan_rows,
            }
            frame_metas.append(frame_meta)

        if norm_over_time:
            for chans_over_time in zip(*[frame_meta['chan_rows'] for frame_meta in frame_metas]):
                flat = [c['raw_signal'].ravel() for c in chans_over_time]
                cums = np.cumsum(list(map(len, flat)))
                combo = np.hstack(flat)
                combo_normed = kwimage.normalize_intensity(combo).copy()
                # combo_normed = kwimage.normalize(combo).copy()
                flat_normed = np.split(combo_normed, cums)
                for row, flat_item in zip(chans_over_time, flat_normed):
                    norm_signal = flat_item.reshape(*row['raw_signal'].shape)
                    norm_signal = kwimage.atleast_3channels(norm_signal)
                    norm_signal = np.nan_to_num(norm_signal)
                    row['norm_signal'] = norm_signal

        # Given prepared frame metadata, build a vertical stack of per-chanel
        # information, and then horizontally stack the timesteps.
        horizontal_stack = []
        for frame_meta in frame_metas:
            vertical_stack = []

            frame_idx = frame_meta['frame_idx']
            frame_item = frame_meta['frame_item']
            chan_rows = frame_meta['chan_rows']
            full_mode_code = frame_meta['full_mode_code']
            class_idxs = frame_meta['class_idxs']
            changes = frame_meta['changes']
            gid = frame_item['gid']

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

            # Create overlays for training objective targets
            truth_overlays = []

            # Create the the true class label overlay
            true_heatmap = kwimage.Heatmap(class_idx=class_idxs, classes=classes)
            class_overlay = true_heatmap.colorize('class_idx')
            class_overlay[..., 3] = 0.5
            truth_overlays.append({
                'overlay': class_overlay,
                'label_text': 'true class',
            })

            # Create the true change label overlay
            change_overlay = np.zeros(changes.shape[0:2] + (4,), dtype=np.float32)
            change_overlay = kwimage.Mask(changes, format='c_mask').draw_on(change_overlay, color='lime')
            change_overlay = kwimage.ensure_alpha_channel(change_overlay)
            change_overlay[..., 3] = (changes > 0).astype(np.float32) * 0.5
            truth_overlays.append({
                'overlay': change_overlay,
                'label_text': 'true change',
            })
            # change_overlay = kwimage.make_heatmask(changes)

            if not overlay_on_image:
                # Draw the truth by itself
                for overlay_info in truth_overlays:
                    label_text = overlay_info['label_text']
                    row_canvas = overlay_info['overlay'][..., 0:3]
                    row_canvas = kwimage.imresize(row_canvas, max_dim=max_dim).clip(0, 1)
                    # TODO: deduplicate this block of code
                    # TODO: make draw_text_on_image able to return the
                    # geometry of what it drew and use that.
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
                    if iterx < len(truth_overlays):
                        overlay_info = truth_overlays[iterx]
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
            if item_output and  key in item_output:
                if overlay_on_image:
                    norm_signal = chan_rows[0]['norm_signal']
                else:
                    norm_signal = np.zeros_like(chan_rows[0]['norm_signal'])
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
                    norm_signal = chan_rows[1 if len(chan_rows) > 1 else 0]['norm_signal']
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
        if num_workers > 0:
            util_globals.request_nofile_limits()
        loader = torch.utils.data.DataLoader(
            self, batch_size=batch_size, num_workers=num_workers,
            shuffle=shuffle, pin_memory=pin_memory, collate_fn=ub.identity)
        return loader


def debug_video_information(dset, video_id):
    """
    Ignore:
        >>> from watch.tasks.fusion.datamodules.kwcoco_video_data import *  # NOQA
        >>> from watch.utils.util_data import find_smart_dvc_dpath
        >>> dvc_dpath = find_smart_dvc_dpath()
        >>> coco_fpath = dvc_dpath / 'drop1-S2-L8-aligned/data.kwcoco.json'
        >>> dset = kwcoco.CocoDataset(coco_fpath)

        for video_id in dset.index.videos.keys():
            debug_video_information(dset, video_id)
    """
    exclude_sensors = None
    # exclude_sensors = {'L8'}
    video_info = dset.index.videos[video_id]
    video_name = video_info['name']
    all_video_gids = list(dset.index.vidid_to_gids[video_id])

    if exclude_sensors is not None:
        sensor_coarse = dset.images(all_video_gids).lookup('sensor_coarse', '')
        flags = [s not in exclude_sensors for s in sensor_coarse]
        video_gids = list(ub.compress(all_video_gids, flags))
    else:
        video_gids = all_video_gids
    video_gids = np.array(video_gids)

    video_frame_idxs = np.array(list(range(len(video_gids))))

    # If the dataset has dates, we can use that
    gid_to_datetime = {}
    frame_dates = dset.images(video_gids).lookup('date_captured', None)
    for gid, date in zip(video_gids, frame_dates):
        if date is not None:
            gid_to_datetime[gid] = parser.parse(date)
    unixtimes = np.array([
        gid_to_datetime[gid].timestamp()
        if gid in gid_to_datetime else np.nan
        for gid in video_gids])

    window_time_dims = 5

    sample_idxs = dilated_template_sample(unixtimes, window_time_dims)
    sample_pattern_v1 = kwarray.one_hot_embedding(sample_idxs, len(unixtimes), dim=1).sum(axis=2)

    # For each frame, calculate a weight proportional to how much we would
    # like to include any other frame in the sample.
    sensors = np.array(dset.images(video_gids).lookup('sensor_coarse', None))
    dilated_weights = dilated_time_weights(unixtimes)['final']
    same_sensor = sensors[:, None] == sensors[None, :]
    sensor_weights = ((same_sensor * 0.5) + 0.5)
    pair_weights = dilated_weights * sensor_weights
    pair_weights[np.eye(len(pair_weights), dtype=bool)] = 1.0

    # Get track info in this video
    classes = dset.object_categories()
    tid_to_infos = ub.ddict(list)
    video_aids = dset.images(video_gids).annots.lookup('id')
    for aids, gid, frame_idx in zip(video_aids, video_gids, video_frame_idxs):
        tids = dset.annots(aids).lookup('track_id')
        cids = dset.annots(aids).lookup('category_id')
        for tid, aid, cid in zip(tids, aids, cids):
            dset.index.anns[aid]['bbox']
            tid_to_infos[tid].append({
                'gid': gid,
                'cid': cid,
                'aid': aid,
                'cx': classes.id_to_idx[cid],
                'frame_idx': frame_idx,
            })

    nancx = len(classes) + 1
    track_phase_mat = []
    # bg_cid = classes.node_to_cid['No Activity']
    for tid, track_infos in tid_to_infos.items():
        track_phase = np.full(len(video_frame_idxs), fill_value=nancx)
        at_idxs = np.array([row['frame_idx'] for row in track_infos])
        track_cxs = np.array([row['cx'] for row in track_infos])
        track_phase[at_idxs] = track_cxs
        track_phase_mat.append(track_phase)
    track_phase_mat = np.array(track_phase_mat)

    if 1:
        import kwplot
        import pandas as pd
        kwplot.autompl()
        sns = kwplot.autosns()

        fnum = video_id

        utils.category_tree_ensure_color(classes)
        color_lut = np.zeros((nancx + 1, 3))
        for node, node_data in classes.graph.nodes.items():
            cx = classes.id_to_idx[node_data['id']]
            color_lut[cx] = node_data['color']
        color_lut[nancx] = (0, 0, 0)
        colored_track_phase = color_lut[track_phase_mat]

        fig = kwplot.figure(fnum=fnum, pnum=(3, 4, slice(0, 3)), doclf=True)
        ax = fig.gca()
        kwplot.imshow(colored_track_phase, ax=ax)
        ax.set_xlabel('observation index')
        ax.set_ylabel('track')
        ax.set_title(f'{video_name} tracks')

        fig = kwplot.figure(fnum=fnum, pnum=(3, 4, 4))
        label_to_color = {
            node: data['color']
            for node, data in classes.graph.nodes.items()}
        label_to_color = ub.sorted_keys(label_to_color)
        legend_img = utils._memo_legend(label_to_color)
        kwplot.imshow(legend_img)

        # pairwise affinity
        fig = kwplot.figure(fnum=fnum, pnum=(3, 1, 2))
        ax = fig.gca()
        kwplot.imshow(kwimage.normalize(pair_weights), ax=ax)
        ax.set_title('pairwise affinity')

        # =====================
        # Show Sample Pattern in heatmap
        datetimes = np.array([datetime.datetime.fromtimestamp(t) for t in unixtimes])
        # dates = np.array([datetime.datetime.fromtimestamp(t).date() for t in unixtimes])
        #
        df = pd.DataFrame(sample_pattern_v1)
        df.index.name = 'index'
        #
        df.columns = pd.to_datetime(datetimes).date
        df.columns.name = 'date'
        #
        kwplot.figure(fnum=fnum, pnum=(3, 1, 3))
        ax = sns.heatmap(data=df)
        # ax.set_title(f'Sample Pattern wrt Available Observations: {video_name}')
        ax.set_title('Sample pattern')
        ax.set_xlabel('Observation Index')
        ax.set_ylabel('Sample Index')


def sample_video_spacetime_targets(dset, window_dims, window_overlap=0.0,
                                   negative_classes=None, keepbound=False,
                                   exclude_sensors=None,
                                   time_sampling='dilate_template',
                                   use_annot_info=True):
    """
    Example:
        >>> # xdoctest: +REQUIRES(env:DVC_DPATH)
        >>> import os
        >>> from watch.tasks.fusion.datamodules.kwcoco_video_data import *  # NOQA
        >>> from watch.utils.util_data import find_smart_dvc_dpath
        >>> dvc_dpath = find_smart_dvc_dpath()
        >>> coco_fpath = dvc_dpath / 'drop1-S2-L8-aligned/data.kwcoco.json'
        >>> dset = kwcoco.CocoDataset(coco_fpath)
        >>> # Create a sliding window object for each specific image (because they may
        >>> # have different sizes, technically we could memoize this)
        >>> import kwarray
        >>> window_overlap = 0.5
        >>> window_dims = (2, 64, 64)
        >>> keepbound = False
        >>> exclude_sensors = None
        >>> sample_grid = sample_video_spacetime_targets(dset, window_dims, window_overlap)
        >>> time_sampling = 'dilate_template'
        >>> positives = list(ub.take(sample_grid['targets'], sample_grid['positives_indexes']))
        _ = xdev.profile_now(sample_video_spacetime_targets)(dset, window_dims, window_overlap)

    Example:
        >>> from watch.tasks.fusion.datamodules.kwcoco_video_data import *  # NOQA
        >>> import ndsampler
        >>> import kwcoco
        >>> dset = kwcoco.CocoDataset.demo('vidshapes2-multispectral', num_frames=30)
        >>> # Create a sliding window object for each specific image (because they may
        >>> # have different sizes, technically we could memoize this)
        >>> import kwarray
        >>> window_overlap = 0.0
        >>> window_dims = (2, 96, 96)
        >>> keepbound = False
        >>> sample_grid = sample_video_spacetime_targets(dset, window_dims, window_overlap, time_sampling='dilate_template')
        >>> sample_grid = sample_video_spacetime_targets(dset, window_dims, window_overlap, time_sampling='contiguous')

        _ = xdev.profile_now(sample_video_spacetime_targets)(dset, window_dims, window_overlap)
    """
    # Create a sliding window object for each specific image (because they may
    # have different sizes, technically we could memoize this)
    import pyqtree

    # window_overlap = 0.5
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

    if use_annot_info:
        # FIXME: HARD CODED CONSTANTS
        print('dset.cats = {}'.format(ub.repr2(dset.cats, nl=1)))
        special_cids = ub.ddict(set)
        special_aliases = {
            'pre_cids': {'background', 'No Activity'},
            'ignore_cids': {'ignore', 'Unknown', 'clouds'},

            'active': {'Active Construction'},
            'post_cids': {'Post Construction'},
        }
        for key, aliases in special_aliases.items():
            for name in aliases:
                if name in dset.index.name_to_cat:
                    special_cids[key].add(dset.index.name_to_cat[name]['id'])

    # Given an video
    all_vid_ids = list(dset.index.videos.keys())
    for video_id in ub.ProgIter(all_vid_ids, desc='sample video regions'):
        slider = vidid_to_space_slider[video_id]

        video_info = dset.index.videos[video_id]
        all_video_gids = list(dset.index.vidid_to_gids[video_id])

        if exclude_sensors is not None:
            sensor_coarse = dset.images(all_video_gids).lookup('sensor_coarse', '')
            flags = [s not in exclude_sensors for s in sensor_coarse]
            video_gids = list(ub.compress(all_video_gids, flags))
        else:
            video_gids = all_video_gids
        video_frame_idxs = np.array(list(range(len(video_gids))))

        # If the dataset has dates, we can use that
        gid_to_datetime = {}
        frame_dates = dset.images(video_gids).lookup('date_captured', None)
        for gid, date in zip(video_gids, frame_dates):
            if date is not None:
                gid_to_datetime[gid] = parser.parse(date)
        unixtimes = np.array([
            gid_to_datetime[gid].timestamp()
            if gid in gid_to_datetime else np.nan
            for gid in video_gids])

        if use_annot_info:
            qtree = pyqtree.Index((0, 0, video_info['width'], video_info['height']))
            qtree.aid_to_tlbr = {}
            tid_to_infos = ub.ddict(list)
            video_aids = dset.images(video_gids).annots.lookup('id')
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
                    qtree.insert(aid, tlbr_box)
                    qtree.aid_to_tlbr[aid] = tlbr_box
                    dset.index.anns[aid]['bbox']
                    tid_to_infos[tid].append({
                        'gid': gid,
                        'cid': cid,
                        'frame_index': frame_index,
                        'vidspace_box': tlbr_box,
                        'cname': dset._resolve_to_cat(cid)['name'],
                        'aid': aid,
                    })

            tid_to_dframe = ub.map_vals(kwarray.DataFrameLight.from_dict, tid_to_infos)
            for track_dframe in tid_to_dframe.values():
                track_dframe['gid'] = np.array(track_dframe['gid'])
                track_dframe['frame_index'] = np.array(track_dframe['frame_index'])
                # Precompute for speed
                track_boxes = kwimage.Boxes(np.array(track_dframe['vidspace_box']), 'ltrb')
                track_dframe['track_pairwise_ious'] = track_boxes.ious(track_boxes)
                track_dframe['track_boxes'] = track_boxes

        if time_sampling == 'dilate_template':
            sample_idxs = dilated_template_sample(unixtimes, window_time_dims)
            # sample_pattern = kwarray.one_hot_embedding(sample_idxs, len(unixtimes), dim=1).sum(axis=2)

        elif time_sampling == 'contiguous':
            time_slider = kwarray.SlidingWindow(
                (len(unixtimes),), (window_time_dims,), stride=(1,), keepbound=True,
                allow_overshoot=True)
            all_indexes = np.arange(len(unixtimes))
            sample_idxs = [all_indexes[sl] for sl in time_slider]
        else:
            raise NotImplementedError(time_sampling)

        for space_region in list(slider):
            y_sl, x_sl = space_region

            # Find all annotations that pass through this spatial region
            if use_annot_info:
                vid_box = kwimage.Boxes.from_slice((y_sl, x_sl))
                query = vid_box.to_tlbr().data[0]
                isect_aids = set(qtree.intersect(query))

            for frame_idxs in sample_idxs:
                gids = list(ub.take(video_gids, frame_idxs))

                if use_annot_info:
                    if isect_aids:
                        has_annot = any(
                            bool(isect_aids & _aids)
                            for _aids in dset.index.gid_to_aids.values())
                    else:
                        has_annot = False

                    if has_annot:
                        positive_idxs.append(len(targets))
                    else:
                        # Hack: exclude all annotated regions from negative sampling
                        negative_idxs.append(len(targets))

                targets.append({
                    'gids': gids,
                    'space_slice': space_region,
                    # 'changes': ','.join(changes),
                    # 'region_tracks': region_tracks,
                })

        if 0:
            # For each frame, calculate a weight proportional to how much we would
            # like to include any other frame in the sample.
            sensors = np.array(dset.images(video_gids).lookup('sensor_coarse', None))
            dilated_weights = dilated_time_weights(unixtimes)['final']
            same_sensor = sensors[:, None] == sensors[None, :]
            sensor_weights = ((same_sensor * 0.5) + 0.5)
            pair_weights = dilated_weights * sensor_weights
            pair_weights[np.eye(len(pair_weights), dtype=bool)] = 1.0

            classes = dset.object_categories()
            nancx = len(classes) + 1
            track_phase_mat = []
            # bg_cid = classes.node_to_cid['No Activity']
            for tid, track_dframe in tid_to_dframe.items():
                # FIXME; BROKEN, NOT THE RIGHT INDEX
                at_idxs = np.searchsorted(video_frame_idxs, track_dframe['frame_index'])
                track_phase = np.full(len(video_frame_idxs), fill_value=nancx)
                track_cids = np.array(track_dframe['cid'])
                track_cxs = [classes.id_to_idx[cid] for cid in track_cids]
                track_phase[at_idxs] = track_cxs
                track_phase_mat.append(track_phase)
            track_phase_mat = np.array(track_phase_mat)

            if 0:
                utils.category_tree_ensure_color(classes)
                color_lut = np.zeros((nancx + 1, 3))
                for node, node_data in classes.graph.nodes.items():
                    cx = classes.id_to_idx[node_data['id']]
                    color_lut[cx] = node_data['color']
                color_lut[nancx] = (0, 0, 0)
                colored_track_phase = color_lut[track_phase_mat]
                kwplot.imshow(colored_track_phase)

            tid_to_track_changemat = {}
            for tid, track_dframe in tid_to_dframe.items():
                # For each track, find frames where phase boundries occur
                track_cids = np.array(track_dframe['cid'])
                at_idxs = np.searchsorted(video_frame_idxs, track_dframe['frame_index'])
                track_phase = np.full(len(video_frame_idxs), fill_value=np.nan)
                track_phase[at_idxs] = track_cids
                track_missing = np.isnan(track_phase)
                is_change = (track_phase[:, None] != track_phase[None, :])
                is_change[track_missing, :] = 0
                is_change[:, track_missing] = 0
                tid_to_track_changemat[tid] = is_change

            # print('tid_to_info = {}'.format(ub.repr2(tid_to_info, nl=2, sort=0)))
            for space_region in list(slider):
                y_sl, x_sl = space_region

                # Find all annotations that pass through this spatial region
                vid_box = kwimage.Boxes.from_slice((y_sl, x_sl))
                query = vid_box.to_tlbr().data[0]
                isect_aids = sorted(set(qtree.intersect(query)))

                isect_annots = dset.annots(isect_aids)
                unique_tracks = set(isect_annots.lookup('track_id'))
                region_tid_to_info = ub.dict_subset(tid_to_dframe, unique_tracks)

                # precompute track metrics over pairwise frames for speed
                tid_to_space_window_iooa = {}
                window_change_weights = []
                for tid, track_dframe in region_tid_to_info.items():
                    track_boxes = track_dframe['track_boxes']
                    track_fxs = track_dframe['frame_index']
                    # track_boxes = kwimage.Boxes(track_dframe['vidspace_box'], 'tlbr')
                    track_iooa = vid_box.iooas(track_boxes)[0, :]
                    is_visible = track_iooa > 0
                    visible_fxs = track_fxs[is_visible]
                    invisible_fxs = track_fxs[~is_visible]
                    track_change_weight = tid_to_track_changemat[tid].copy().astype(np.float32)
                    track_change_weight[invisible_fxs, :][:, invisible_fxs] = 0
                    track_change_weight[visible_fxs, :][:, invisible_fxs] = 0.3
                    track_change_weight[invisible_fxs, :][:, visible_fxs] = 0.3
                    track_change_weight[invisible_fxs, :][:, visible_fxs] = 0.3
                    tid_to_space_window_iooa[tid] = track_iooa
                    window_change_weights.append(track_change_weight)

                if window_change_weights:
                    has_annot = True
                    frame_change_w = np.add.reduce(window_change_weights)
                    min_p = 0.1
                    frame_change_w = (frame_change_w * (1 - min_p)) + min_p
                    frame_w = (dilated_weights * frame_change_w)
                else:
                    has_annot = False
                    frame_w = (dilated_weights)

                for base_idx in video_frame_idxs:
                    chosen = affinity_sample(
                        frame_w, window_time_dims, include_indices=[base_idx],
                        jit=0)
                    gids = list(ub.take(video_gids, chosen))

                    if has_annot:
                        positive_idxs.append(len(targets))
                    else:
                        # Hack: exclude all annotated regions from negative sampling
                        negative_idxs.append(len(targets))

                    # TODO: would it be more efficient to simply iterate over
                    # spatial positions and then return information about what the
                    # dataset was allowed to to to augment the target?  In other
                    # words, allow training to choose a different dilated temporal
                    # sample every time?

                    targets.append({
                        'gids': gids,
                        'space_slice': space_region,
                        # 'changes': ','.join(changes),
                        # 'region_tracks': region_tracks,
                    })

            if 0:
                # PPR Review hacks:
                # Generate all combinations of sample frames
                # TODO: ITERATING THROUGH ALL COMBINATIONS IS SLOW!
                # Could likely reparametarize to sample implicitly in getitem
                for frame_idxs in list(it.combinations(video_frame_idxs, window_time_dims)):

                    # Default is to assume this spacetime region has no change
                    changes = []

                    any_visible = False

                    gids = list(ub.take(video_gids, frame_idxs))
                    region_tracks = []
                    # For each track that passes through this region
                    for tid, track_dframe in region_tid_to_info.items():

                        # Interpolate / Extrapolate track annotations onto the
                        # sample frame indexes The track might not be annotated on
                        # each frame. For each timestep check the most recent state
                        # of the track.
                        sampled_info = ub.ddict(list)
                        _explicit_track_fxs = track_dframe['frame_index']
                        _space_window_iooa = tid_to_space_window_iooa[tid]
                        most_recent_idxs = np.searchsorted(_explicit_track_fxs, frame_idxs, 'right') - 1
                        # prev_box = None
                        prev_idx = None
                        for idx in most_recent_idxs:
                            if idx < 0:
                                # The sample frame is before this track starts
                                cid = ub.peek(special_cids['pre_cids'])
                                curr_box = None
                                space_window_iooa = 0
                                prev_iou = np.nan
                            elif idx > _explicit_track_fxs[-1]:
                                # The sampled frame is after this track ends
                                cid = track_dframe['cid'][idx]
                                curr_box = track_dframe['vidspace_box'][idx]
                                space_window_iooa = _space_window_iooa[idx]
                                if prev_idx is None:
                                    prev_iou = np.nan
                                else:
                                    prev_iou = track_dframe['track_pairwise_ious'][idx][prev_idx]
                            else:
                                cid = track_dframe['cid'][idx]
                                curr_box = track_dframe['vidspace_box'][idx]
                                space_window_iooa = _space_window_iooa[idx]
                                if prev_idx is None:
                                    prev_iou = np.nan
                                else:
                                    prev_iou = track_dframe['track_pairwise_ious'][idx][prev_idx]
                            sampled_info['cid'].append(cid)
                            sampled_info['box'].append(curr_box)
                            sampled_info['space_window_iooa'].append(space_window_iooa)
                            sampled_info['prev_iou'].append(prev_iou)
                            prev_idx = idx

                        # Heuristic: flag this region as a positive if any of these
                        # heuristics are detected.  TODO: we need to figure out the
                        # best method for determening if a space-time window
                        # contains a positive example of change or not. What is the
                        # best way to encode this in a kwcoco dataset?

                        # Detect if the category of track changes.
                        is_visible = np.array(sampled_info['space_window_iooa']) > 0.1
                        is_change_visible = (is_visible[0:-1] & is_visible[1:])
                        is_moving = np.array(sampled_info['prev_iou'][1:]) < 0.6
                        is_visibly_moving = is_change_visible & is_moving

                        if is_visible.any():
                            any_visible = True

                        if is_visibly_moving.any():
                            changes.append('visibly_moving')

                        if 1:
                            has_pre = set(sampled_info['cid']) & special_cids['pre_cids']
                            has_active = set(sampled_info['cid']) & special_cids['active']
                            if len(has_pre) and len(has_active):
                                changes.append('hack')

                        if 0:
                            unique_cids = set(sampled_info['cid']) - special_cids['ignore_cids']
                            # TODO: dont mark change when post_cid moves to background
                            if len(unique_cids) > 1:
                                changes.append('category change')

                        region_tracks.append({
                            'tid': tid,
                            'sampled_info': sampled_info,
                            # 'is_visibly_moving': is_visibly_moving,
                            # 'is_moving': is_moving,
                            # 'is_change_visible': is_change_visible,
                            # 'is_visible': is_visible,
                        })

                    if changes:
                        positive_idxs.append(len(targets))
                    else:
                        # Hack: exclude all annotated regions from negative sampling
                        if any_visible:
                            continue
                        negative_idxs.append(len(targets))

                    targets.append({
                        'gids': gids,
                        'space_slice': space_region,
                        'changes': ','.join(changes),
                        'region_tracks': region_tracks,
                    })

    print('Found {} targets'.format(len(targets)))
    if use_annot_info:
        print('Found {} positives'.format(len(positive_idxs)))
        print('Found {} negatives'.format(len(negative_idxs)))

    sample_grid = {
        'positives_indexes': positive_idxs,
        'negatives_indexes': negative_idxs,
        'targets': targets,
    }
    return sample_grid


def dilated_template_sample(unixtimes, time_window):
    """
    Args:
        unixtimes (ndarray):
            list of unix timestamps indicating available temporal samples

        time_window (int):
            number of frames per sample

    References:
        https://docs.google.com/presentation/d/1GSOaY31cKNERQObl_L3vk0rGu6zU7YM_ZFLrdksHSC0/edit#slide=id.p

    Example:
        >>> from watch.tasks.fusion.datamodules.kwcoco_video_data import *  # NOQA
        >>> low = datetime.datetime.now().timestamp()
        >>> high = low + datetime.timedelta(days=365 * 5).total_seconds()
        >>> rng = kwarray.ensure_rng(0)
        >>> base_unixtimes = np.array(sorted(rng.randint(low, high, 20)), dtype=float)
        >>> unixtimes = base_unixtimes.copy()
        >>> #unixtimes[rng.rand(*unixtimes.shape) < 0.1] = np.nan
        >>> time_window = 5
        >>> sample_idxs = dilated_template_sample(unixtimes, time_window)
        >>> name = 'demo-data'

        >>> #unixtimes[:] = np.nan
        >>> time_window = 5
        >>> sample_idxs = dilated_template_sample(unixtimes, time_window)
        >>> name = 'demo-data'

    Ignore:
        >>> import os
        >>> from watch.tasks.fusion.datamodules.kwcoco_video_data import *  # NOQA
        >>> from watch.utils.util_data import find_smart_dvc_dpath
        >>> dvc_dpath = find_smart_dvc_dpath()
        >>> bundle_dpath = join(dvc_dpath, 'drop1-S2-L8-aligned')
        >>> coco_fpath = join(bundle_dpath, 'data.kwcoco.json')
        >>> dset = kwcoco.CocoDataset(coco_fpath)
        >>> video_ids = list(ub.sorted_vals(dset.index.vidid_to_gids, key=len).keys())
        >>> vidid = video_ids[0]
        >>> video = dset.index.videos[vidid]
        >>> name = (video['name'])
        >>> print('name = {!r}'.format(name))
        >>> images = dset.images(vidid=vidid)
        >>> datetimes = [parser.parse(date) for date in images.lookup('date_captured')]
        >>> unixtimes = np.array([dt.timestamp() for dt in datetimes])
        >>> time_window = 5
        >>> sample_idxs = dilated_template_sample(unixtimes, time_window)
        >>> #unixtimes[:] = 0

    Ignore:
        >>> import kwplot
        >>> import numpy as np
        >>> sns = kwplot.autosns()

        >>> # =====================
        >>> # Show Sample Pattern in heatmap
        >>> datetimes = np.array([datetime.datetime.fromtimestamp(t) for t in unixtimes])
        >>> dates = np.array([datetime.datetime.fromtimestamp(t).date() for t in unixtimes])
        >>> #
        >>> sample_pattern = kwarray.one_hot_embedding(sample_idxs, len(unixtimes), dim=1).sum(axis=2)
        >>> kwplot.imshow(sample_pattern)
        >>> import pandas as pd
        >>> df = pd.DataFrame(sample_pattern)
        >>> df.index.name = 'index'
        >>> #
        >>> df.columns = pd.to_datetime(datetimes).date
        >>> df.columns.name = 'date'
        >>> #
        >>> kwplot.figure(fnum=1, doclf=True)
        >>> ax = sns.heatmap(data=df)
        >>> ax.set_title(f'Sample Pattern wrt Available Observations: {name}')
        >>> ax.set_xlabel('Observation Index')
        >>> ax.set_ylabel('Sample Index')
        >>> #
        >>> #import matplotlib.dates as mdates
        >>> #ax.figure.autofmt_xdate()

        >>> # =====================
        >>> # Show Sample Pattern WRT to time
        >>> fig = kwplot.figure(fnum=2, doclf=True)
        >>> ax = fig.gca()
        >>> for t in datetimes:
        >>>     ax.plot([t, t], [0, len(sample_idxs) + 1], color='orange')
        >>> for sample_ypos, sample in enumerate(sample_idxs, start=1):
        >>>     ax.plot(datetimes[sample], [sample_ypos] * len(sample), '-x')
        >>> ax.set_title(f'Sample Pattern wrt Time Range: {name}')
        >>> ax.set_xlabel('Time')
        >>> ax.set_ylabel('Sample Index')
        >>> # import matplotlib.dates as mdates
        >>> # ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        >>> # ax.xaxis.set_major_locator(mdates.DayLocator(interval=1))
        >>> # ax.figure.autofmt_xdate()

        >>> # =====================
        >>> # Show Available Samples
        >>> import july
        >>> from july.utils import date_range
        >>> datetimes = [datetime.datetime.fromtimestamp(t) for t in unixtimes]
        >>> grid_dates = date_range(
        >>>     datetimes[0].date().isoformat(),
        >>>     (datetimes[-1] + datetime.timedelta(days=1)).date().isoformat()
        >>> )
        >>> grid_unixtime = np.array([
        >>>     datetime.datetime.combine(d, datetime.datetime.min.time()).timestamp()
        >>>     for d in grid_dates
        >>> ])
        >>> positions = np.searchsorted(grid_unixtime, unixtimes)
        >>> indicator = np.zeros_like(grid_unixtime)
        >>> indicator[positions] = 1
        >>> dates_unixtimes = [d for d in dates]
        >>> july.heatmap(grid_dates, indicator, title=f'Available Observations: {name}', cmap="github")

    Ignore:

        name = 'demo'
        unixtimes = np.arange(11)
        time_window = 5
        time_window = np.array([-5, -1, 0, 1, 5])
        sample_idxs = dilated_template_sample(unixtimes, time_window)

        template_deltas = np.array([-5, -1, 0, 1, 5])

        ideal_time_samples = unixtimes[:, None] + template_deltas[None, :]
        losses = np.abs(ideal_time_samples[None, :, :] - unixtimes[:, None, None])

        losses = np.abs(ideal_time_samples[None, :, :] - unixtimes[:, None, None])


        idx = 5
        all_rows = []
        for idx in range(len(ideal_time_samples)):
            ideal_sample_for_row = ideal_time_samples[idx]
            unixtimes[:, None] - ideal_sample_for_row[None, :]
            loss_for_row = np.abs(ideal_sample_for_row[:, None] - unixtimes[None, :])
            # For each row find the closest available frames to the ideal
            # sample without duplicates.
            candidiates = kwarray.argmaxima(-loss_for_row, axis=1, num=time_window).T
            sample_idxs = sorted(it.islice(ub.unique(candidiates.ravel()), time_window))
            all_rows.append(sample_idxs)
        print('all_rows = {}'.format(ub.repr2(all_rows, nl=1)))
        all_sample_idxs = np.vstack(all_rows)
    """
    import itertools as it
    missing_date = np.isnan(unixtimes)
    missing_any_dates = np.any(missing_date)
    have_any_dates = not np.all(missing_date)

    if isinstance(time_window, int):
        # TODO: formulate how to choose template delta for given window dims
        # Or pass in a delta
        if time_window == 1:
            template_deltas = np.array([
                datetime.timedelta(days=0).total_seconds(),
            ])
        elif time_window == 2:
            template_deltas = np.array([
                datetime.timedelta(days=0).total_seconds(),
                datetime.timedelta(days=+365).total_seconds(),
            ])
        elif time_window == 3:
            template_deltas = np.array([
                datetime.timedelta(days=-365).total_seconds(),
                datetime.timedelta(days=0).total_seconds(),
                datetime.timedelta(days=+365).total_seconds(),
            ])
        # elif time_window == 4:
        #     template_deltas = np.array([
        #         datetime.timedelta(days=-365).total_seconds(),
        #         datetime.timedelta(days=-1).total_seconds(),
        #         datetime.timedelta(days=0).total_seconds(),
        #         datetime.timedelta(days=+365).total_seconds(),
        #     ])
        # elif time_window == 5:
        #     template_deltas = np.array([
        #         datetime.timedelta(days=-365).total_seconds(),
        #         datetime.timedelta(days=-1).total_seconds(),
        #         datetime.timedelta(days=0).total_seconds(),
        #         datetime.timedelta(days=+1).total_seconds(),
        #         datetime.timedelta(days=+365).total_seconds(),
        #     ])
        # elif time_window == 7:
        #     template_deltas = np.array([
        #         datetime.timedelta(days=-365).total_seconds(),
        #         datetime.timedelta(days=-1).total_seconds(),
        #         datetime.timedelta(days=-17).total_seconds(),
        #         datetime.timedelta(days=0).total_seconds(),
        #         datetime.timedelta(days=+1).total_seconds(),
        #         datetime.timedelta(days=+17).total_seconds(),
        #         datetime.timedelta(days=+365).total_seconds(),
        #     ])
        else:
            num_years = 2
            min_time = -datetime.timedelta(days=365).total_seconds() * num_years
            max_time = datetime.timedelta(days=365).total_seconds() * num_years
            template_deltas = np.linspace(min_time, max_time, time_window).round().astype(int)
            # Always include a delta of 0
            template_deltas[np.abs(template_deltas).argmin()] = 0
    else:
        template_deltas = time_window
    print('template_deltas = {!r}'.format(template_deltas))

    num_frames = len(template_deltas)

    # template_deltas = np.array([
    #     datetime.timedelta(days=-365).total_seconds(),
    #     datetime.timedelta(days=-30).total_seconds(),
    #     datetime.timedelta(days=0).total_seconds(),
    #     datetime.timedelta(days=30).total_seconds(),
    #     datetime.timedelta(days=365).total_seconds(),
    # ])
    # unixtimes = np.arange(20)
    # template_deltas = np.array([-5, -1, 0, 1, 5])

    if missing_any_dates:
        if have_any_dates:
            from scipy import interpolate
            frame_idxs = np.arange(len(unixtimes))
            miss_idxs = frame_idxs[missing_date]
            have_idxs = frame_idxs[~missing_date]
            have_values = unixtimes[have_idxs]
            interp = interpolate.interp1d(have_idxs, have_values, fill_value=np.nan)
            interp_vals = interp(miss_idxs)
            unixtimes = unixtimes.copy()
            unixtimes[miss_idxs] = interp_vals
        else:
            unixtimes = np.linspace(0, 1, len(unixtimes))

    ideal_time_samples = unixtimes[:, None] + template_deltas[None, :]

    if 0:
        # Broken, not sure why
        losses = np.abs(ideal_time_samples[None, :, :] - unixtimes[:, None, None])
        all_candidates = kwarray.argmaxima(-losses, axis=1, num=num_frames)
        all_rows = []
        for candidates in all_candidates:
            sample_idxs = sorted(it.islice(ub.unique(candidates.ravel()), num_frames))
            all_rows.append(sample_idxs)
    else:
        # Seems to work correctly?
        all_rows = []
        for idx in range(len(ideal_time_samples)):
            ideal_sample_for_row = ideal_time_samples[idx]
            loss_for_row = np.abs(ideal_sample_for_row[:, None] - unixtimes[None, :])
            loss_for_row[loss_for_row == 0] = -np.inf
            # For each row find the closest available frames to the ideal
            # sample without duplicates.
            if 1:
                sample_idxs = np.array(kwarray.mincost_assignment(loss_for_row)[0]).T[1]
            else:
                candidiates = np.array([zz.argsort()[0:num_frames] for zz in loss_for_row]).T
                # candidates =
                # np.array([zz.argsort()[0:5] for zz in loss_for_row]).T
                # kwarray.argmaxima(-loss_for_row, axis=1, num=num_frames).T
                # candidiates = kwarray.argmaxima(-loss_for_row, axis=1, num=num_frames).T
                sample_idxs = sorted(it.islice(ub.unique(candidiates.ravel()), num_frames))
            all_rows.append(sorted(sample_idxs))

    all_sample_idxs = np.vstack(all_rows)
    sample_idxs = util_kwarray.unique_rows(all_sample_idxs)

    return sample_idxs


def dilated_time_weights(unixtimes):
    """
    Produce a pairwise affinity weights between frames based on a dilated time
    heuristic.

    Example:
        >>> from watch.tasks.fusion.datamodules.kwcoco_video_data import *  # NOQA
        >>> low = datetime.datetime.now().timestamp()
        >>> high = low + datetime.timedelta(days=365 * 5).total_seconds()
        >>> rng = kwarray.ensure_rng(0)
        >>> base_unixtimes = np.array(sorted(rng.randint(low, high, 113)), dtype=float)

        >>> # Test no missing data case
        >>> unixtimes = base_unixtimes.copy()
        >>> allhave_weights = dilated_time_weights(unixtimes)
        >>> #
        >>> # Test all missing data case
        >>> unixtimes = np.full_like(unixtimes, fill_value=np.nan)
        >>> allmiss_weights = dilated_time_weights(unixtimes)
        >>> #
        >>> # Test partial missing data case
        >>> unixtimes = base_unixtimes.copy()
        >>> unixtimes[rng.rand(*unixtimes.shape) < 0.1] = np.nan
        >>> anymiss_weights_1 = dilated_time_weights(unixtimes)
        >>> unixtimes = base_unixtimes.copy()
        >>> unixtimes[rng.rand(*unixtimes.shape) < 0.5] = np.nan
        >>> anymiss_weights_2 = dilated_time_weights(unixtimes)
        >>> unixtimes = base_unixtimes.copy()
        >>> unixtimes[rng.rand(*unixtimes.shape) < 0.9] = np.nan
        >>> anymiss_weights_3 = dilated_time_weights(unixtimes)

        >>> # xdoctest: +REQUIRES(--show)
        >>> import kwplot
        >>> kwplot.autoplt()
        >>> pnum_ = kwplot.PlotNums(nCols=5)
        >>> kwplot.figure(fnum=1, doclf=True)
        >>> # kwplot.imshow(kwimage.normalize(daylight_weights))
        >>> kwplot.imshow(kwimage.normalize(allhave_weights['final']), pnum=pnum_(), title='no missing dates')
        >>> kwplot.imshow(kwimage.normalize(anymiss_weights_1['final']), pnum=pnum_(), title='any missing dates (0.1)')
        >>> kwplot.imshow(kwimage.normalize(anymiss_weights_2['final']), pnum=pnum_(), title='any missing dates (0.5)')
        >>> kwplot.imshow(kwimage.normalize(anymiss_weights_3['final']), pnum=pnum_(), title='any missing dates (0.9)')
        >>> kwplot.imshow(kwimage.normalize(allmiss_weights['final']), pnum=pnum_(), title='all missing dates')

        >>> import pandas as pd
        >>> sns = kwplot.autosns()
        >>> fig = kwplot.figure(fnum=2, doclf=True)
        >>> kwplot.imshow(kwimage.normalize(allhave_weights['final']), pnum=(1, 3, 1), title='pairwise affinity')
        >>> row_idx = 0
        >>> df = pd.DataFrame({k: v[row_idx] for k, v in allhave_weights.items()})
        >>> df['index'] = np.arange(df.shape[0])
        >>> data = df.drop(['final'], axis=1).melt(['index'])
        >>> kwplot.figure(fnum=2, pnum=(1, 3, 2))
        >>> sns.lineplot(data=data, x='index', y='value', hue='variable')
        >>> fig.gca().set_title('Affinity components for row={}'.format(row_idx))
        >>> kwplot.figure(fnum=2, pnum=(1, 3, 3))
        >>> sns.lineplot(data=df, x='index', y='final')
        >>> fig.gca().set_title('Affinity components for row={}'.format(row_idx))
    """
    missing_date = np.isnan(unixtimes)
    missing_any_dates = np.any(missing_date)
    have_any_dates = not np.all(missing_date)

    weights = {}

    if have_any_dates:
        # unixtimes[np.random.rand(*unixtimes.shape) > 0.1] = np.nan
        seconds_per_year = datetime.timedelta(days=365).total_seconds()
        seconds_per_day = datetime.timedelta(days=1).total_seconds()

        second_deltas = np.abs(unixtimes[None, :] - unixtimes[:, None])
        year_deltas = second_deltas / seconds_per_year
        day_deltas = second_deltas / seconds_per_day

        # Upweight similar seasons
        season_weights = (1 + np.cos(year_deltas * math.tau)) / 2.0

        # Upweight similar times of day
        daylight_weights = ((1 + np.cos(day_deltas * math.tau)) / 2.0) * 0.5 + 0.5

        # Upweight times in the future
        # future_weights = year_deltas ** 0.25
        future_weights = util_kwarray.asymptotic(year_deltas)

        weights['daylight'] = daylight_weights
        weights['season'] = season_weights
        weights['future'] = future_weights

        frame_weights = season_weights * future_weights * daylight_weights

    if missing_any_dates:
        # For the frames that don't have dates on them, we use indexes to
        # calculate a proxy weight.
        frame_idxs = np.arange(len(unixtimes))
        frame_dist = np.abs(frame_idxs[:, None] - frame_idxs[None, ])
        index_weight = (frame_dist / len(frame_idxs)) ** 0.33
        weights['index'] = index_weight

        # Interpolate over any existing values
        # https://stackoverflow.com/questions/21690608/numpy-inpaint-nans-interpolate-and-extrapolate
        if have_any_dates:
            from scipy import interpolate
            miss_idxs = frame_idxs[missing_date]
            have_idxs = frame_idxs[~missing_date]

            miss_coords = np.vstack([
                util_kwarray.cartesian_product(miss_idxs, frame_idxs),
                util_kwarray.cartesian_product(have_idxs, miss_idxs)])
            have_coords = util_kwarray.cartesian_product(have_idxs, have_idxs)
            have_values = frame_weights[tuple(have_coords.T)]

            interp = interpolate.LinearNDInterpolator(have_coords, have_values, fill_value=0.8)
            interp_vals = interp(miss_coords)

            miss_coords_fancy = tuple(miss_coords.T)
            frame_weights[miss_coords_fancy] = interp_vals

            # Average interpolation with the base case
            frame_weights[miss_coords_fancy] = (
                frame_weights[miss_coords_fancy] +
                index_weight[miss_coords_fancy]) / 2
        else:
            # No data to use, just use
            frame_weights = index_weight

    weights['final'] = frame_weights
    return weights


@ub.memoize
def cython_aff_samp_mod():
    import os
    from watch.tasks.fusion.datamodules import kwcoco_video_data
    fpath = os.path.join(os.path.dirname(kwcoco_video_data.__file__), 'affinity_sampling.pyx')
    cython_mod = xdev.import_module_from_pyx(fpath, verbose=0, annotate=True)
    return cython_mod


def affinity_sample(affinity, size, include_indices, return_info=False,
                    rng=None, jit=False):
    """
    Choose random samples to maximize

    Args:
        affinity (ndarray):
            pairwise affinity matrix

        size (int):
            Number of sample indices to return

        include_indices (List[int]):
            Indicies that must be included in the sample

        rng (Coercable[RandomState]):
            random state

    Possible Related Work:
        * Random Stratified Sampling Affinity Matrix
        * A quasi-random sampling approach to image retrieval

    Example:
        >>> from watch.tasks.fusion.datamodules.kwcoco_video_data import *  # NOQA
        >>> low = datetime.datetime.now().timestamp()
        >>> high = low + datetime.timedelta(days=365 * 5).total_seconds()
        >>> rng = kwarray.ensure_rng(0)
        >>> unixtimes = np.array(sorted(rng.randint(low, high, 113)), dtype=float)
        >>> #
        >>> affinity = dilated_time_weights(unixtimes)['final']
        >>> include_indices = [5]
        >>> size = 5
        >>> chosen, info = affinity_sample(affinity, size, include_indices, return_info=True)
        >>> # xdoctest: +REQUIRES(--show)
        >>> steps = info['steps']
        >>> import kwplot
        >>> sns = kwplot.autosns()
        >>> plt = kwplot.autoplt()
        >>> pnum_ = kwplot.PlotNums(nCols=2, nSubplots=len(steps) * 2 + 1)
        >>> kwplot.figure(pnum=pnum_(), fnum=1, doclf=True)
        >>> kwplot.imshow(kwimage.normalize(affinity), title='Pairwise Affinity')
        >>> chosen_so_far = list(info['include_indices'])
        >>> for step_idx, step in enumerate(steps, start=len(include_indices)):
        >>>     fig = kwplot.figure(pnum=pnum_())
        >>>     ax = fig.gca()
        >>>     idx = step['next_idx']
        >>>     probs = step['probs']
        >>>     ymax = probs.max()
        >>>     xmax = len(probs)
        >>>     x, y = idx, probs[idx]
        >>>     for x_ in chosen_so_far:
        >>>         ax.plot([x_, x_], [0, ymax], color='gray')
        >>>     ax.plot(np.arange(xmax), probs)
        >>>     xpos = x + xmax * 0.0 if x < (xmax / 2) else x - xmax * 0.0
        >>>     ypos = y + ymax * 0.3 if y < (ymax / 2) else y - ymax * 0.3
        >>>     ax.annotate('chosen', (x, y), xytext=(xpos, ypos), color='black', arrowprops=dict(color='orange', arrowstyle="->"))
        >>>     ax.plot([x, x], [0, ymax], color='orange')
        >>>     #ax.annotate('chosen', (x, y), color='black')
        >>>     ax.set_title('Sample {}'.format(step_idx))
        >>>     chosen_so_far.append(idx)
        >>>     fig = kwplot.figure(pnum=pnum_())
        >>>     ax = fig.gca()
        >>>     ax.plot(np.arange(xmax), step['next_affinity'], color='orange')
        >>>     #ax.annotate('chosen', (x, y), xytext=(xpos, ypos), color='black', arrowprops=dict(color='black', arrowstyle="->"))
        >>>     ax.plot([x, x], [0, step['next_affinity'].max()], color='orange')
        >>>     ax.set_title('New affinity {}'.format(step_idx))
        >>> kwplot.imshow(kwimage.normalize(affinity[sorted(chosen)][:, sorted(chosen)]), pnum=pnum_(), title='Final Affinities')

    Ignore:
        >>> from watch.tasks.fusion.datamodules.kwcoco_video_data import *  # NOQA
        >>> low = datetime.datetime.now().timestamp()
        >>> high = low + datetime.timedelta(days=365 * 5).total_seconds()
        >>> rng = kwarray.ensure_rng(0)
        >>> unixtimes = np.array(sorted(rng.randint(low, high, 113)), dtype=float)
        >>> affinity = dilated_time_weights(unixtimes)['final']
        >>> include_indices = [5]
        >>> size = 20
        >>> xdev.profile_now(affinity_sample)(affinity, size, include_indices)

    CommandLine:
        xdoctest -m /home/joncrall/code/watch/watch/tasks/fusion/datamodules/kwcoco_video_data.py affinity_sample:1 --cython

    Example:
        >>> # xdoctest: +REQUIRES(--cython)
        >>> from watch.tasks.fusion.datamodules.kwcoco_video_data import *  # NOQA
        >>> low = datetime.datetime.now().timestamp()
        >>> high = low + datetime.timedelta(days=365 * 5).total_seconds()
        >>> rng = kwarray.ensure_rng(0)
        >>> unixtimes = np.array(sorted(rng.randint(low, high, 113)), dtype=float)
        >>> affinity = dilated_time_weights(unixtimes)['final']
        >>> include_indices = [5]
        >>> size = 5
        >>> import timerit
        >>> ti = timerit.Timerit(100, bestof=10, verbose=2)
        >>> for timer in ti.reset('python'):
        >>>     with timer:
        >>>         affinity_sample(affinity, size, include_indices, jit=False)
        >>> for timer in ti.reset('cython'):
        >>>     with timer:
        >>>         chosen = affinity_sample(affinity, size, include_indices, jit=True)
        >>> # xdev.profile_now(affinity_sample)(affinity, size, include_indices, jit=True)
        >>> # xdev.profile_now(affinity_sample)(affinity, size, include_indices, jit=False)

        fig.tight_layout()
    """
    # TODO: make this faster
    chosen = list(include_indices)
    if len(chosen) == 1:
        current_weights = affinity[chosen[0]]
    else:
        current_weights = affinity[chosen].prod(axis=0)
    num_sample = size - len(chosen)
    rng = kwarray.ensure_rng(rng)
    if jit:
        cython_mod = cython_aff_samp_mod()
        return cython_mod.cython_affinity_sample(affinity, num_sample, current_weights, chosen, rng)
    current_weights[chosen] = 0
    # available_idxs = np.arange(affinity.shape[0])
    if return_info:
        info = {'steps': [], 'initial_weights': current_weights, 'include_indices': include_indices}
    for _ in range(num_sample):
        # Choose the next image based on combined sample affinity

        # probs = current_weights / current_weights.sum()
        # next_idx = rng.choice(available_idxs, size=1, p=probs)[0]

        cumprobs = current_weights.cumsum()
        dart = rng.rand() * cumprobs[-1]
        next_idx = np.searchsorted(cumprobs, dart)

        next_affinity = affinity[next_idx]
        chosen.append(next_idx)

        if return_info:
            probs = current_weights / current_weights.sum()
            info['steps'].append({
                'probs': probs,
                'next_idx': next_idx,
                'next_affinity': next_affinity,
            })
        # Don't resample the same item
        current_weights = current_weights * next_affinity
        current_weights[next_idx] = 0
    chosen = sorted(chosen)
    if return_info:
        return chosen, info
    else:
        return chosen
