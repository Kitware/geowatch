import einops
import itertools as it
import kwarray
import kwcoco
import kwimage
import ndsampler
import numpy as np
import pathlib
import pytorch_lightning as pl
import torch
import ubelt as ub

import cv2
from functools import partial  # NOQA
from kwcoco import channel_spec
from torch import nn
from torch.utils import data
from torchvision import transforms
from watch.tasks.fusion import utils

try:
    import xdev
    profile = xdev.profile
except Exception:
    profile = ub.identity


class WatchDataModule(pl.LightningDataModule):
    """
    Prepare the kwcoco dataset as torch video datasets

    Example:
        >>> # xdoctest: +REQUIRES(env:DVC_DPATH)
        >>> # Run the following tests on real watch data if DVC is available
        >>> from watch.tasks.fusion.datasets.watch_data import *  # NOQA
        >>> from os.path import join
        >>> import os
        >>> _default = ub.expandpath('$HOME/data/dvc-repos/smart_watch_dvc')
        >>> dvc_dpath = os.environ.get('DVC_DPATH', _default)
        >>> coco_fpath = join(dvc_dpath, 'drop1_S2_aligned_c1/data.kwcoco.json')
        >>> import kwcoco
        >>> train_dataset = kwcoco.CocoDataset(coco_fpath)
        >>> test_dataset = None
        >>> img = ub.peek(train_dataset.imgs.values())
        >>> #channels = '|'.join([aux['channels'] for aux in img['auxiliary']])
        >>> #chan_spec = kwcoco.channel_spec.FusedChannelSpec.coerce(channels)
        >>> channels = None
        >>> #
        >>> batch_size = 2
        >>> time_steps = 3
        >>> chip_size = 330
        >>> self = WatchDataModule(
        >>>     train_dataset=train_dataset,
        >>>     test_dataset=test_dataset,
        >>>     batch_size=batch_size,
        >>>     channels=channels,
        >>>     num_workers=0,
        >>>     time_steps=time_steps,
        >>>     chip_size=chip_size,
        >>> )
        >>> self.setup("fit")
        >>> dl = self.train_dataloader()
        >>> batch = next(iter(dl))
        >>> # Visualize
        >>> canvas = self.draw_batch(batch)
        >>> # xdoctest: +REQUIRES(--show)
        >>> import kwplot
        >>> kwplot.autompl()
        >>> kwplot.imshow(canvas)
        >>> kwplot.show_if_requested()

    Example:
        >>> # Run the data module on coco demo datasets for the CI
        >>> from watch.tasks.fusion.datasets.watch_data import *  # NOQA
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
        >>> self = WatchDataModule(
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
        channels=None,
        valid_pct=0.1,
        batch_size=4,
        num_workers=4,
        preprocessing_step=None,
        tfms_channel_subset=None,
        normalize_inputs=False,
        verbose=1,
    ):
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
        self.channels = channels
        self.valid_pct = valid_pct
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.preprocessing_step = preprocessing_step
        self.normalize_inputs = normalize_inputs
        self.input_stats = None

        if self.verbose:
            print('Init WatchDataModule')
            print('self.train_kwcoco = {!r}'.format(self.train_kwcoco))
            print('self.vali_kwcoco = {!r}'.format(self.vali_kwcoco))
            print('self.test_kwcoco = {!r}'.format(self.test_kwcoco))
            print('self.time_steps = {!r}'.format(self.time_steps))
            print('self.chip_size = {!r}'.format(self.chip_size))
            print('self.channels = {!r}'.format(self.channels))

        # TODO: there is no need for tfms_channel_subset,
        # Remove that parameter and just send ``channels`` in as the requested
        # subset. To handle the case where you want to test how well the model
        # works when channels are missing pass some channel dropout parameter
        # to the dataset.

        if 0:
            tfms_channel_subset = channels if (tfms_channel_subset is None) else tfms_channel_subset
            channel_split = channels.split("|")
            tfms_channel_subset = [
                idx
                for idx, channel in enumerate(tfms_channel_subset.split("|"))
                if channel in channel_split
            ]
            self.tfms_channel_subset = tfms_channel_subset

            self.train_tfms = self.preprocessing_step
            self.test_tfms = transforms.Compose([
                utils.Lambda(lambda x: x[:, tfms_channel_subset]),
            ])

        # Store train / test / vali
        self.torch_datasets = {}
        self.coco_datasets = {}

    def draw_batch(self, batch, stage='train', outputs=None, max_items=2):
        """
        Helper method to draw a batch of data.

        Example:
            >>> from watch.tasks.fusion.datasets.watch_data import *  # NOQA
            >>> from watch.tasks.fusion import datasets
            >>> self = datasets.WatchDataModule(
            >>>     train_dataset='special:vidshapes8-multispectral', num_workers=0)
            >>> self.setup('fit')
            >>> loader = self.train_dataloader()
            >>> batch = next(iter(loader))
            >>> item = batch[0]
            >>> # Visualize
            >>> B = len(batch)
            >>> C, H, W = ub.peek(item['frames'][0]['modes'].values()).shape
            >>> T = len(item['frames'])
            >>> outputs = {'binary_predictions': [torch.rand(T - 1, H, W) for _ in range(B)]}
            >>> stage = 'train'
            >>> canvas = self.draw_batch(batch, stage=stage, outputs=outputs)
            >>> # xdoctest: +REQUIRES(--show)
            >>> import kwplot
            >>> kwplot.autompl()
            >>> kwplot.imshow(canvas)
            >>> kwplot.show_if_requested()
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
            binprobs = None
            if outputs is not None:
                if 'binary_predictions' in outputs:
                    binprobs = outputs['binary_predictions'][item_idx].data.cpu().numpy()

            part = dataset.draw_item(item, binprobs=binprobs)
            canvas_list.append(part)
        canvas = kwimage.stack_images_grid(canvas_list, axis=1, overlap=-12)

        with_legend = True
        if with_legend:
            label_to_color = {
                node: data['color']
                for node, data in dataset.classes.graph.nodes.items()}
            label_to_color = ub.sorted_keys(label_to_color)
            legend_img = _memo_legend(label_to_color)
            canvas = kwimage.stack_images([canvas, legend_img], axis=1)

        return canvas

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

            coco_train_sampler = ndsampler.CocoSampler(train_coco_dset)
            train_dataset = WatchVideoDataset(
                coco_train_sampler,
                sample_shape=(self.time_steps, self.chip_size, self.chip_size),
                window_overlap=(self.time_overlap, self.chip_overlap, self.chip_overlap),
                channels=self.channels,
                # transform=self.train_tfms,
            )

            # Unfortunately lightning seems to only enable / disables
            # validation depending on the methods that are defined, so we are
            # not able to statically define them.
            self.torch_datasets['train'] = train_dataset
            ub.inject_method(self, lambda self: self._make_dataloader('train', shuffle=True), 'train_dataloader')

            if self.normalize_inputs:
                if isinstance(self.normalize_inputs, str):
                    if self.normalize_inputs == 'some-special-key':
                        # TODO: hard code any special input normalization you
                        # want here
                        pass
                    else:
                        raise KeyError(self.normalize_inputs)
                else:
                    if isinstance(self.normalize_inputs, int):
                        num = self.normalize_inputs
                    else:
                        num = None
                    self.input_stats = train_dataset.cached_input_stats(
                        num=num, num_workers=self.num_workers,
                        batch_size=self.batch_size)

            if self.vali_kwcoco is not None:
                # Explicit validation dataset should be prefered
                vali_data = self.vali_kwcoco
                if isinstance(vali_data, pathlib.Path):
                    vali_data = str(vali_data.expanduser())
                if self.verbose:
                    print('Build validation kwcoco dataset')
                kwcoco_ds = kwcoco.CocoDataset.coerce(vali_data)
                vali_coco_sampler = ndsampler.CocoSampler(kwcoco_ds)
                vali_dataset = WatchVideoDataset(
                    vali_coco_sampler,
                    sample_shape=(self.time_steps, self.chip_size, self.chip_size),
                    window_overlap=(self.time_overlap, self.chip_overlap, self.chip_overlap),
                    channels=self.channels,
                    # transform=self.vali_tfms,
                )
                self.torch_datasets['vali'] = vali_dataset
                ub.inject_method(self, lambda self: self._make_dataloader('vali', shuffle=True), 'val_dataloader')
            else:
                if 0:
                    # TODO:
                    raise NotImplementedError(
                        "TODO: Can take different video subsets at the coco level"
                        " but dont do it at the torch dataset level."
                        " too much leakage"
                    )
                    num_examples = len(train_dataset)
                    num_valid = int(self.valid_pct * num_examples)
                    num_train = num_examples - num_valid

                    # FIXME Probably not the right way to do this, too much leakage
                    train_dataset, vali_dataset = data.random_split(
                        train_dataset,
                        [num_train, num_valid],
                    )
                    self.torch_datasets['train'] = train_dataset
                    self.torch_datasets['vali'] = vali_dataset

        if stage == "test" or stage is None:
            test_data = self.test_kwcoco
            if isinstance(test_data, pathlib.Path):
                test_data = str(test_data.expanduser())
            if self.verbose:
                print('Build test kwcoco dataset')
            test_coco_dset = kwcoco.CocoDataset.coerce(test_data)
            self.coco_datasets['test'] = test_coco_dset
            test_coco_sampler = ndsampler.CocoSampler(test_coco_dset)
            self.torch_datasets['test'] = WatchVideoDataset(
                test_coco_sampler,
                sample_shape=(self.time_steps, self.chip_size, self.chip_size),
                window_overlap=(self.time_overlap, self.chip_overlap, self.chip_overlap),
                channels=self.channels,
                mode='test',
                # transform=self.test_tfms,
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

    @staticmethod
    def add_data_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("watch_data")
        parser.add_argument("--train_dataset", default=None, type=pathlib.Path)
        parser.add_argument("--vali_dataset", default=None, type=pathlib.Path)
        parser.add_argument("--test_dataset", default=None, type=pathlib.Path)
        parser.add_argument("--time_steps", default=2, type=int)
        parser.add_argument("--chip_size", default=128, type=int)
        parser.add_argument("--time_overlap", default=0.0, type=float, help='fraction of time steps to overlap')
        parser.add_argument("--chip_overlap", default=0.1, type=float, help='fraction of space steps to overlap')
        parser.add_argument("--channels", default=None, type=str)
        parser.add_argument("--valid_pct", default=0.1, type=float)
        parser.add_argument("--batch_size", default=4, type=int)
        parser.add_argument("--num_workers", default=4, type=int)
        parser.add_argument("--transform_key", default="none", type=str)
        parser.add_argument("--tfms_scale", default=2000., type=float)
        parser.add_argument("--tfms_window_size", default=8, type=int)

        parser.add_argument(
            "--normalize_inputs", default=True, help=ub.paragraph(
                '''
                if True, computes the mean/std for this dataset on each mode
                so this can be passed to the model.
                '''))

        return parent_parser


class AddPositionalEncoding(nn.Module):
    def __init__(self, dest_dim, dims_to_encode):
        super().__init__()
        self.dest_dim = dest_dim
        self.dims_to_encode = dims_to_encode
        assert self.dest_dim not in self.dims_to_encode

    def forward(self, x):

        inds = [
            slice(0, size) if (dim in self.dims_to_encode) else slice(0, 1)
            for dim, size in enumerate(x.shape)
        ]
        inds[self.dest_dim] = self.dims_to_encode

        encoding = torch.cat(torch.meshgrid([
            torch.linspace(0, 1, x.shape[dim]) if (dim in self.dims_to_encode) else torch.tensor(-1.)
            for dim in range(len(x.shape))
        ]), dim=self.dest_dim)[inds]

        expanded_shape = list(x.shape)
        expanded_shape[self.dest_dim] = -1
        x = torch.cat([x, encoding.expand(expanded_shape).type_as(x)], dim=self.dest_dim)
        return x


def coco_channel_profiles(coco_dset, max_checks=float('inf')):
    """
    Example:
        >>> from watch.tasks.fusion.datasets.watch_data import *  # NOQA
        >>> coco_dset = kwcoco.CocoDataset.demo('vidshapes8-multispectral')
        >>> candidates = coco_channel_profiles(coco_dset)
    """
    candidates = ub.oset()
    for check_idx, img in enumerate(coco_dset.index.imgs.values()):

        objs = []
        if img.get('file_name', None):
            objs.append(img)
        objs.extend(img.get('auxiliary', []))

        chan_list = [obj.get('channels', None) for obj in objs]
        candidates.add(tuple(chan_list))

        if check_idx > max_checks:
            break
    if len(candidates) == 0:
        raise Exception('no candidate channel profiles')
    return candidates


class WatchVideoDataset(data.Dataset):
    """
    Example:
        >>> from watch.tasks.fusion.datasets.watch_data import *  # NOQA
        >>> import ndsampler
        >>> import kwcoco
        >>> coco_dset = kwcoco.CocoDataset.demo('vidshapes2-multispectral', num_frames=5)
        >>> coco_dset.ensure_category('background')
        >>> sampler = ndsampler.CocoSampler(coco_dset)
        >>> channels = 'B10|B8a|B1|B8'
        >>> sample_shape = (3, 530, 610)
        >>> self = WatchVideoDataset(sampler, sample_shape=sample_shape, channels=channels)
        >>> index = len(self) // 4
        >>> item = self[index]
        >>> canvas = self.draw_item(item)
        >>> # xdoctest: +REQUIRES(--show)
        >>> import kwplot
        >>> kwplot.autompl()
        >>> kwplot.imshow(canvas)
        >>> kwplot.show_if_requested()

    Example:
        >>> from watch.tasks.fusion.datasets.watch_data import *  # NOQA
        >>> import ndsampler
        >>> import kwcoco
        >>> coco_dset = kwcoco.CocoDataset.demo('vidshapes2-multispectral', num_frames=5)
        >>> coco_dset.ensure_category('background')
        >>> sampler = ndsampler.CocoSampler(coco_dset)
        >>> sample_shape = (2, 128, 128)
        >>> self = WatchVideoDataset(sampler, sample_shape=sample_shape, channels=None)
        >>> index = 0
        >>> item = self[index]
        >>> canvas = self.draw_item(item)
        >>> # xdoctest: +REQUIRES(--show)
        >>> import kwplot
        >>> kwplot.autompl()
        >>> kwplot.imshow(canvas)
        >>> kwplot.show_if_requested()
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
    ):

        if channels is None:
            # Hack to use all channels in the first image.
            # (Does not handle heterogeneous channels yet)
            candidates = coco_channel_profiles(sampler.dset, 3)
            chan_list = candidates[0]
            channels = '|'.join(chan_list)
        channels = channel_spec.ChannelSpec.coerce(channels).normalize()

        if transform is not None:
            raise Exception('I do not like injecting the transforms')

        if mode == 'test':
            # In test mode we have to sample everything
            sample_grid = simple_video_sample_grid(
                sampler.dset, window_dims=sample_shape,
                window_overlap=window_overlap)
        else:
            full_sample_grid = sampler.new_sample_grid(
                "video_detection", sample_shape,
                window_overlap=window_overlap)

            n_pos = len(full_sample_grid["positives"])
            n_neg = len(full_sample_grid["negatives"])

            # TODO: parametarize ratio of positives to negatives
            neg_to_pos_ratio = 2
            max_neg = (neg_to_pos_ratio * n_pos)
            if n_neg > max_neg:
                print('chose max_neg = {!r}'.format(max_neg))
                neg_idxs = kwarray.shuffle(np.arange(n_neg), rng=47789403)[0:max_neg]
                chosen_negs = list(ub.take(full_sample_grid["negatives"], neg_idxs))
            else:
                chosen_negs = full_sample_grid["negatives"]

            sample_grid = list(it.chain(
                full_sample_grid["positives"],
                chosen_negs,
            ))

        self.sample_grid = sample_grid
        self.transform = transform
        self.window_overlap = window_overlap
        self.sampler = sampler
        self.classes = self.sampler.classes

        category_tree_ensure_color(self.classes)

        self.sample_shape = sample_shape
        self.channels = channels
        self.mode = mode

        self.ignore_classnames = {
            'clouds', 'ignore'
        }

        self.augment = False
        self.disable_augmenter = False

    def __len__(self):
        return len(self.sample_grid)

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
            >>> from watch.tasks.fusion.datasets.watch_data import *  # NOQA
            >>> import ndsampler
            >>> import kwcoco
            >>> coco_dset = kwcoco.CocoDataset.demo('vidshapes2-multispectral', num_frames=5)
            >>> coco_dset.ensure_category('background')
            >>> sampler = ndsampler.CocoSampler(coco_dset)
            >>> channels = 'B10|B8a|B1|B8'
            >>> sample_shape = (5, 530, 610)
            >>> self = WatchVideoDataset(sampler, sample_shape=sample_shape, channels=channels)
            >>> item = self[0]
            >>> canvas = self.draw_item(item)
            >>> # xdoctest: +REQUIRES(--show)
            >>> import kwplot
            >>> kwplot.autompl()
            >>> kwplot.imshow(canvas)
            >>> kwplot.show_if_requested()

        Example:
            >>> # xdoctest: +REQUIRES(env:DVC_DPATH)
            >>> # Run the following tests on real watch data if DVC is available
            >>> from watch.tasks.fusion.datasets.watch_data import *  # NOQA
            >>> import os
            >>> from os.path import join
            >>> import ndsampler
            >>> import kwcoco
            >>> _default = ub.expandpath('$HOME/data/dvc-repos/smart_watch_dvc')
            >>> dvc_dpath = os.environ.get('DVC_DPATH', _default)
            >>> #coco_fpath = join(dvc_dpath, 'drop1_S2_aligned_c1/data.kwcoco.json')
            >>> coco_fpath = join(dvc_dpath, 'drop1-S2-L8-LS-aligned-v2/train_data.kwcoco.json')
            >>> coco_dset = kwcoco.CocoDataset(coco_fpath)
            >>> sampler = ndsampler.CocoSampler(coco_dset)
            >>> sample_shape = (7, 128, 128)
            >>> self = WatchVideoDataset(sampler, sample_shape=sample_shape, channels=None)
            >>> item = self[4]
            >>> canvas = self.draw_item(item)
            >>> # xdoctest: +REQUIRES(--show)
            >>> import kwplot
            >>> kwplot.autompl()
            >>> kwplot.imshow(canvas)
            >>> kwplot.show_if_requested()
        """

        # get positive sample definition
        tr = self.sample_grid[index]
        if self.channels:
            tr["channels"] = self.channels

        tr['as_xarray'] = False
        tr['use_experimental_loader'] = 1
        # collect sample
        sample = self.sampler.load_sample(tr)

        channel_keys = sample['tr']['_coords']['c'].values.tolist()
        mode_key = '|'.join(channel_keys)

        # Access the sampled image and annotation data
        raw_frame_list = sample['im']
        raw_det_list = sample['annots']['frame_dets']
        raw_gids = sample['tr']['gids']

        # Break data down on a per-frame basis so we can apply image-based
        # augmentations.
        frame_items = []

        input_dsize = self.sample_shape[-2:][::-1]
        # hack for augmentation
        # TODO: make a nice "augmenter" pipeline
        do_flip = False
        if not self.disable_augmenter and self.mode == 'fit':
            def make_hflipper(width):
                def hflip(pt):
                    new = np.hstack([width - pt[:, 0:1], pt[:, 1:2]])
                    return new
                return hflip
            flipper = make_hflipper(input_dsize[0])
            do_flip = np.random.rand() > 0.5

        prev_frame_cids = None

        for frame, dets, gid in zip(raw_frame_list, raw_det_list, raw_gids):
            img = self.sampler.dset.imgs[gid]

            frame = np.asarray(frame, dtype=np.float32)

            if do_flip:
                frame = np.fliplr(frame)
                dets = dets.warp(flipper)

            # Resize the sampled window to the target space for the network
            frame, info = kwimage.imresize(frame, dsize=input_dsize,
                                           interpolation='linear',
                                           antialias=True,
                                           return_info=True)
            # Remember to apply any transform to the dets as well
            dets = dets.scale(info['scale'])
            dets = dets.translate(info['offset'])

            # allocate class masks
            frame_cids = np.full(frame.shape[:2], dtype=np.int32, fill_value=-1)
            frame_ignore = np.full(frame.shape[:2], dtype=np.uint8, fill_value=0)

            # Rasterize frame targets
            ann_polys = dets.data['segmentations'].to_polygon_list()
            ann_aids = dets.data['aids']
            ann_cids = dets.data['cids']
            # Note: it is important to respect class indexes, ids, and name
            # mappings
            # TODO: layer ordering? Multiclass prediction?
            for poly, aid, cid in zip(ann_polys, ann_aids, ann_cids):
                cidx = self.classes.id_to_idx[cid]
                catname = self.classes.id_to_node[cid]
                if catname in self.ignore_classnames:
                    frame_ignore.fill(frame_cids, value=1)
                else:
                    poly.fill(frame_cids, value=cidx)

            # ensure channel dim is not squeezed
            frame_hwc = kwarray.atleast_nd(frame, 3)
            # catch nans
            frame_hwc[np.isnan(frame_hwc)] = -1.
            # rearrange image axes for pytorch
            frame_chw = einops.rearrange(frame_hwc, 'h w c -> c h w')

            # convert annotations into a change detection task suitable for
            # the network.
            if prev_frame_cids is None:
                frame_change = None
            else:
                frame_change = (frame_cids != prev_frame_cids).astype(np.uint8)
                # Clean up the change target
                frame_change = morphology(frame_change, 'open', kernel=3)
                frame_change = torch.from_numpy(frame_change)

            # convert to torch
            frame_item = {
                'gid': gid,
                'date_captured': img.get('date_captured', ''),
                'sensor_coarse': img.get('sensor_coarse', ''),
                'modes': {
                    mode_key: torch.from_numpy(frame_chw),
                },
                'change': frame_change,
                'labels': torch.from_numpy(frame_cids),
                'ignore': torch.from_numpy(frame_ignore),
            }
            prev_frame_cids = frame_cids
            frame_items.append(frame_item)

        vidid = sample['tr']['vidid']
        video = self.sampler.dset.index.videos[vidid]

        item = {
            # TODO: breakup modes into different items
            "frames": frame_items,
            "video_id": sample['tr']['vidid'],
            "video_name": video['name'],
            "tr": sample['tr'],  # pass all of the metadata
        }
        return item

    def cached_input_stats(self, num=None, num_workers=0, batch_size=2):
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
            ('channels', self.channels.__json__()),
            # ('sample_shape', self.sample_shape),
            ('depends_version', 3),  # bump if `compute_input_stats` changes
        ])
        workdir = None
        cacher = ub.Cacher('dset_mean', dpath=workdir, depends=depends)
        input_stats = cacher.tryload()
        if input_stats is None:
            input_stats = self.compute_input_stats(
                num, num_workers=num_workers, batch_size=batch_size)
            cacher.save(input_stats)
        return input_stats

    def compute_input_stats(self, num=None, num_workers=0, batch_size=2):
        """
        Args:
            num (int | None): number of input items to compute stats for

        Example:
            >>> from watch.tasks.fusion.datasets.watch_data import *  # NOQA
            >>> import ndsampler
            >>> import kwcoco
            >>> coco_dset = kwcoco.CocoDataset.demo('vidshapes2-multispectral', num_frames=5)
            >>> coco_dset.ensure_category('background')
            >>> sampler = ndsampler.CocoSampler(coco_dset)
            >>> sample_shape = (2, 128, 128)
            >>> self = WatchVideoDataset(sampler, sample_shape=sample_shape, channels=None)
            >>> self.compute_input_stats()

        Example:
            >>> from watch.tasks.fusion.datasets.watch_data import *  # NOQA
            >>> import ndsampler
            >>> import kwcoco
            >>> coco_dset = kwcoco.CocoDataset.demo('vidshapes8')
            >>> coco_dset.ensure_category('background')
            >>> sampler = ndsampler.CocoSampler(coco_dset)
            >>> sample_shape = (2, 128, 128)
            >>> self = WatchVideoDataset(sampler, sample_shape=sample_shape, channels=None)
            >>> self.compute_input_stats()

        Example:
            >>> # xdoctest: +REQUIRES(env:DVC_DPATH)
            >>> # Run the following tests on real watch data if DVC is available
            >>> from watch.tasks.fusion.datasets.watch_data import *  # NOQA
            >>> import os
            >>> from os.path import join
            >>> import ndsampler
            >>> import kwcoco
            >>> _default = ub.expandpath('$HOME/data/dvc-repos/smart_watch_dvc')
            >>> dvc_dpath = os.environ.get('DVC_DPATH', _default)
            >>> coco_fpath = join(dvc_dpath, 'drop1_S2_aligned_c1/data.kwcoco.json')
            >>> coco_dset = kwcoco.CocoDataset(coco_fpath)
            >>> sampler = ndsampler.CocoSampler(coco_dset)
            >>> sample_shape = (2, 128, 128)
            >>> self = WatchVideoDataset(sampler, sample_shape=sample_shape, channels=None)
            >>> self.compute_input_stats()
        """
        num = None
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
                         for key in self.channels.keys()}

        for batch_items in ub.ProgIter(loader, desc='estimate mean/std'):
            for item in batch_items:
                for frame_item in item['frames']:
                    for mode_code, mode_val in frame_item['modes'].items():
                        channel_stats[mode_code].update(mode_val.numpy())

        input_stats = {}
        for key, running in channel_stats.items():
            perchan_stats = running.summarize(axis=(1, 2))
            input_stats[key] = {
                'mean': perchan_stats['mean'].round(3),  # only take 3 sigfigs
                'std': perchan_stats['std'].round(3),
            }
        self.disable_augmenter = False
        return input_stats

    @profile
    def draw_item(self, item, binprobs=None):
        """
        Example:
            >>> from watch.tasks.fusion.datasets.watch_data import *  # NOQA
            >>> import ndsampler
            >>> import kwcoco
            >>> coco_dset = kwcoco.CocoDataset.demo('vidshapes2-multispectral', num_frames=5)
            >>> coco_dset.ensure_category('background')
            >>> sampler = ndsampler.CocoSampler(coco_dset)
            >>> channels = 'B10|B8a|B1|B8'
            >>> sample_shape = (5, 530, 610)
            >>> self = WatchVideoDataset(sampler, sample_shape=sample_shape, channels=channels)
            >>> index = len(self) // 4
            >>> item = self[index]
            >>> # Calculate the probability of change for each frame
            >>> #binprobs = np.random.rand(*self.sample_shape)
            >>> binprobs = np.stack([
            >>>     kwimage.Heatmap.random(
            >>>         dims=sample_shape[1:3], classes=1).data['class_probs'][0]
            >>>     for _ in range(sample_shape[0])
            >>> ])
            >>> binprobs = binprobs[1:]  # first frame does not have change
            >>> #binprobs[0][:] = 0  # first change prob should be all zeros
            >>> canvas = self.draw_item(item, binprobs)
            >>> # xdoctest: +REQUIRES(--show)
            >>> import kwplot
            >>> kwplot.autompl()
            >>> kwplot.imshow(canvas)
            >>> kwplot.show_if_requested()
        """
        # TODO: parametarize
        max_dim = 296       # shape of each item when drawing
        limit_channels = 3  # limit drawing to only show a subset of the channels.

        classes = self.sampler.classes
        frame_list = []
        for frame_idx, frame_item in enumerate(item['frames']):

            mask = frame_item['labels'].data.cpu().numpy()
            changes = frame_item['change']
            if changes is None:
                changes = np.zeros_like(mask)
            else:
                changes = changes.data.cpu().numpy()

            # hack just use one of the modes
            mode_code, mode_data = ub.peek(frame_item['modes'].items())
            chan_names = mode_code.split('|')
            frame_chw = mode_data.data.cpu().numpy()

            vertical_stack = []

            # frame_text = f't={frame_idx} - {date_captured}\n{mode_code}'
            # frame_header = kwimage.draw_text_on_image(
            #     {'width': max_dim}, frame_text, org=(max_dim // 2, 5), valign='top',
            #     halign='center', color='purple')

            gid = frame_item['gid']

            def header_text(text, shrink=False):
                """
                If shrink is true, shrinks the text to fit, otherwise text is
                placed in the center at a constant size, but is not guarenteed
                to fit.
                """
                if shrink:
                    header = kwimage.draw_text_on_image(
                        None, text, org=(1, 1),
                        valign='top', halign='left', color='salmon')
                    header = cv2.copyMakeBorder(header, 3, 3, 3, 3,
                                                cv2.BORDER_CONSTANT)
                    header = kwimage.imresize(header, dsize=(max_dim, None))
                else:
                    header = kwimage.draw_text_on_image(
                        {'width': max_dim}, text, org=(max_dim // 2, 1),
                        valign='top', halign='center', color='salmon')
                return header

            header_part = header_text(f't={frame_idx} gid={gid}', shrink=False)
            vertical_stack.append(header_part)

            sensor_coarse = frame_item.get('sensor_coarse', '')
            if sensor_coarse:
                header_part = header_text(f'{sensor_coarse}', shrink=False)
                vertical_stack.append(header_part)

            date_captured = frame_item.get('date_captured', '')
            if date_captured:
                header_part = header_text(f'{date_captured}', shrink=True)
                vertical_stack.append(header_part)

            if 0:
                header_part = header_text(f'{mode_code}', shrink=True)
                vertical_stack.append(header_part)

            signal = None
            for chan_idx, chan in enumerate(frame_chw):
                if limit_channels and chan_idx  >= limit_channels:
                    break
                chan_name = chan_names[chan_idx]

                # TODO: normalize across time?
                chan = kwimage.normalize_intensity(chan)
                signal = kwimage.atleast_3channels(chan).copy()
                signal_text = f'c={chan_idx}:{chan_name}'

                true_layers = []
                if chan_idx == 0:
                    # Draw class label on odd frames
                    true_heatmap = kwimage.Heatmap(class_idx=mask, classes=classes)
                    # true_part = heatmap.draw_on(true_part, with_alpha=0.5)
                    # Hack: -1 is given the last color by colorize, it would
                    # be better if there was a non-negative background class index
                    class_overlay = true_heatmap.colorize('class_idx')
                    class_overlay[..., 3] = 0.5
                    class_overlay[mask == -1, 3] = 0
                    true_layers.append(class_overlay)
                    label_text = 'true class'
                elif chan_idx == 1:
                    # Draw change label on even frames
                    change_overlay = np.zeros(changes.shape[0:2] + (4,), dtype=np.float32)
                    change_overlay = kwimage.Mask(changes, format='c_mask').draw_on(change_overlay, color='lime')
                    change_overlay = kwimage.ensure_alpha_channel(change_overlay)
                    change_overlay[..., 3] = (changes > 0).astype(np.float32) * 0.5
                    # change_overlay = kwimage.make_heatmask(changes)
                    # print('change_overlay = {!r}'.format(change_overlay))
                    true_layers.append(change_overlay)
                    label_text = 'true change'
                else:
                    label_text = None

                true_layers.append(signal)
                true_part = kwimage.overlay_alpha_layers(true_layers)
                true_part = true_part[..., 0:3]

                # Draw change label
                # if chan_idx == 1:
                #     # HACK: fixme put me in above conditional
                #     true_part = kwimage.Mask(changes, format='c_mask').draw_on(true_part, color='lime')

                true_part = kwimage.imresize(true_part, max_dim=max_dim).clip(0, 1)
                true_part = kwimage.draw_text_on_image(
                    true_part, signal_text, (1, 1), valign='top',
                    color='white', border=3)

                if label_text:
                    # TODO: make draw_text_on_image able to return the
                    # geometry of what it drew and use that.
                    signal_bottom_y = 31  # hack: hardcoded
                    true_part = kwimage.draw_text_on_image(
                        true_part, label_text, (1, signal_bottom_y + 1),
                        valign='top', color='lime', border=3)
                vertical_stack.append(true_part)

            if binprobs is not None:
                # Make a probability heatmap we can either display
                # independently or overlay on a rendered channel
                if frame_idx == 0:
                    # BIG RED X
                    pred_mask = kwimage.draw_text_on_image(
                        {'width': max_dim, 'height': max_dim},
                        'X', org=(max_dim // 2, max_dim // 2),
                        valign='center', halign='center', fontScale=10,
                        color='red')
                    pred_part = pred_mask
                else:
                    pred_mask = kwimage.make_heatmask(binprobs[frame_idx - 1])
                    assert signal is not None, 'no channels to draw on'
                    pred_layers = [pred_mask, signal]
                    pred_part = kwimage.overlay_alpha_layers(pred_layers)

                    # TODO: we might want to overlay the prediction on one or all
                    # of the channels
                    pred_part = kwimage.imresize(pred_part, max_dim=max_dim).clip(0, 1)
                    pred_text = f'pred change t={frame_idx}'
                    pred_part = kwimage.draw_text_on_image(
                        pred_part, pred_text, (1, 1), valign='top',
                        color='dodgerblue', border=3)

                vertical_stack.append(pred_part)

            vertical_stack = [kwimage.ensure_uint255(p) for p in vertical_stack]
            frame_canvas = kwimage.stack_images(vertical_stack, overlap=-3)
            frame_list.append(frame_canvas)

        canvas = kwimage.stack_images(frame_list, axis=1, overlap=-5)
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
            >>> from watch.tasks.fusion.datasets.watch_data import *  # NOQA
            >>> import ndsampler
            >>> import kwcoco
            >>> coco_dset = kwcoco.CocoDataset.demo('vidshapes2-multispectral', num_frames=5)
            >>> coco_dset.ensure_category('background')
            >>> sampler = ndsampler.CocoSampler(coco_dset)
            >>> self = WatchVideoDataset(sampler, sample_shape=(3, 530, 610))
            >>> loader = self.make_loader(batch_size=2)
            >>> batch = next(iter(loader))
        """
        loader = torch.utils.data.DataLoader(
            self, batch_size=batch_size, num_workers=num_workers,
            shuffle=shuffle, pin_memory=pin_memory, collate_fn=ub.identity)
        return loader


# TODO: LRU cache?
@ub.memoize
def _morph_kernel_core(h, w):
    return np.ones((h, w), np.uint8)


def _morph_kernel(size):
    if isinstance(size, int):
        h = size
        w = size
    else:
        raise NotImplementedError
    return _morph_kernel_core(h, w)


def morphology(data, mode, kernel=5):
    """
    Executes a morphological operation.

    Args:
        input (ndarray): data
        mode (str) : morphology mode.  currently only open

    Example:
        >>> data = (np.random.rand(32, 32) > 0.5).astype(np.uint8)
        >>> mode = 'open'
        >>> kernel = 5
        >>> morphology(data, mode, kernel=5)

    """
    kernel = _morph_kernel(kernel)
    if mode == 'open':
        new = cv2.morphologyEx(data, cv2.MORPH_OPEN, kernel)
    else:
        raise NotImplementedError
    return new


@ub.memoize
def _memo_legend(label_to_color):
    import kwplot
    legend_img = kwplot.make_legend_img(label_to_color)
    return legend_img


def category_tree_ensure_color(classes):
    """
    Ensures that each category in a CategoryTree has a color

    TODO:
        - [ ] Add to CategoryTree

    Example:
        >>> import kwcoco
        >>> classes = kwcoco.CategoryTree.demo()
        >>> assert not any('color' in data for data in classes.graph.nodes.values())
        >>> category_tree_ensure_color(classes)
        >>> assert all('color' in data for data in classes.graph.nodes.values())
    """
    backup_colors = iter(kwimage.Color.distinct(len(classes)))
    for node in classes.graph.nodes:
        color = classes.graph.nodes[node].get('color', None)
        if color is None:
            color = next(backup_colors)
            classes.graph.nodes[node]['color'] = kwimage.Color(color).as01()


def simple_video_sample_grid(dset, window_dims=None, window_overlap=0.0):
    import kwarray
    keepbound = True

    # Create a sliding window object for each specific image (because they may
    # have different sizes, technically we could memoize this)
    vidid_to_slider = {}
    for vidid, video in dset.index.videos.items():
        gids = dset.index.vidid_to_gids[vidid]
        num_frames = len(gids)
        full_dims = [num_frames, video['height'], video['width']]
        window_dims_ = full_dims if window_dims == 'full' else window_dims
        slider = kwarray.SlidingWindow(full_dims, window_dims_,
                                       overlap=window_overlap,
                                       keepbound=keepbound,
                                       allow_overshoot=True)

        vidid_to_slider[vidid] = slider

    sample_grid = []
    for vidid, slider in vidid_to_slider.items():
        regions = list(slider)
        gids = dset.index.vidid_to_gids[vidid]
        box_gids = []
        for region in regions:
            t_sl, y_sl, x_sl = region
            region_gids = gids[t_sl]
            box_gids.append(region_gids)

        for region, region_gids in zip(regions, box_gids):
            space_slice = region[1:3]
            time_slice = region[0]

            tr = {
                'vidid': vidid,
                'time_slice': time_slice,
                'space_slice': space_slice,
                'gids': region_gids,
            }
            sample_grid.append(tr)
    return sample_grid
