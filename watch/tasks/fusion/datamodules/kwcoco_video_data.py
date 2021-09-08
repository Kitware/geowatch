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
from kwcoco import channel_spec
from torch.utils import data
from watch.tasks.fusion import utils
from watch.utils import kwcoco_extensions

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
        >>> batch_size = 2
        >>> time_steps = 5
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
        neg_to_pos_ratio=2.0,
        channels=None,
        batch_size=4,
        num_workers=4,
        preprocessing_step=None,
        tfms_channel_subset=None,
        normalize_inputs=False,
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
        self.requested_channels = channels
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.preprocessing_step = preprocessing_step
        self.normalize_inputs = normalize_inputs
        self.input_stats = None

        # will only correspond to train
        self.classes = None
        self.channels = None

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
            print('self.requested_channels = {!r}'.format(self.requested_channels))

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
        """
        parser = parent_parser.add_argument_group("kwcoco_video_data")
        parser.add_argument("--train_dataset", default=None, help='path to the train kwcoco file')
        parser.add_argument("--vali_dataset", default=None, help='path to the validation kwcoco file')
        parser.add_argument("--test_dataset", default=None, help='path to the test kwcoco file')
        parser.add_argument("--time_steps", default=2, type=int)
        parser.add_argument("--chip_size", default=128, type=int)
        parser.add_argument("--time_overlap", default=0.0, type=float, help='fraction of time steps to overlap')
        parser.add_argument("--chip_overlap", default=0.1, type=float, help='fraction of space steps to overlap')
        parser.add_argument("--neg_to_pos_ratio", default=2.0, type=float, help='maximum ratio of samples with no annotations to samples with annots')
        parser.add_argument("--channels", default=None, type=str, help='channels to use should be ChannelSpec coercable')
        parser.add_argument("--batch_size", default=4, type=int)
        parser.add_argument("--num_workers", default=4, type=int)

        parser.add_argument(
            "--normalize_inputs", default=True, help=ub.paragraph(
                '''
                if True, computes the mean/std for this dataset on each mode
                so this can be passed to the model.
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

            coco_train_sampler = ndsampler.CocoSampler(train_coco_dset)
            train_dataset = KWCocoVideoDataset(
                coco_train_sampler,
                sample_shape=(self.time_steps, self.chip_size, self.chip_size),
                window_overlap=(self.time_overlap, self.chip_overlap, self.chip_overlap),
                channels=self.requested_channels,
                neg_to_pos_ratio=self.neg_to_pos_ratio,
            )

            # Unfortunately lightning seems to only enable / disables
            # validation depending on the methods that are defined, so we are
            # not able to statically define them.
            self.classes = train_dataset.classes
            self.torch_datasets['train'] = train_dataset
            ub.inject_method(self, lambda self: self._make_dataloader('train', shuffle=True), 'train_dataloader')

            if self.channels is None:
                self.channels = train_dataset.channels

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
                vali_dataset = KWCocoVideoDataset(
                    vali_coco_sampler,
                    sample_shape=(self.time_steps, self.chip_size, self.chip_size),
                    window_overlap=(self.time_overlap, self.chip_overlap, self.chip_overlap),
                    channels=self.requested_channels,
                    neg_to_pos_ratio=0,
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
                channels=self.requested_channels,
                mode='test',
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

    def draw_batch(self, batch, stage='train', outputs=None, max_items=2, **kwargs):
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

            part = dataset.draw_item(item, item_output=item_output, **kwargs)
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
        >>> sample_shape = (3, 530, 610)
        >>> self = KWCocoVideoDataset(sampler, sample_shape=sample_shape, channels=channels)
        >>> index = len(self) // 4
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
        >>> coco_dset = kwcoco.CocoDataset.demo('vidshapes2-multispectral', num_frames=5)
        >>> sampler = ndsampler.CocoSampler(coco_dset)
        >>> sample_shape = (2, 128, 128)
        >>> self = KWCocoVideoDataset(sampler, sample_shape=sample_shape, channels=None)
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
        neg_to_pos_ratio=2.0,
    ):

        self._hueristic_background_classnames = {
            'background', 'No Activity',
        }

        self._heuristic_ignore_classnames = {
            'ignore', 'Unknown', 'clouds',
        }

        if channels is None:
            # Hack to use all channels in the first image.
            # (Does not handle heterogeneous channels yet)
            chan_info = kwcoco_extensions.coco_channel_stats(sampler.dset)
            channels = chan_info['all_channels']
        channels = channel_spec.ChannelSpec.coerce(channels).normalize()

        if transform is not None:
            raise Exception('I do not like injecting the transforms')

        if mode == 'test':
            # In test mode we have to sample everything
            sample_grid = simple_video_sample_grid(
                sampler.dset, window_dims=sample_shape,
                window_overlap=window_overlap)
        else:
            negative_classes = (
                self._heuristic_ignore_classnames |
                self._hueristic_background_classnames
            )
            full_sample_grid = new_video_sample_grid(
                sampler.dset, window_dims=sample_shape,
                window_overlap=window_overlap,
                negative_classes=negative_classes,
            )

            # BUGGED
            # sampler.new_sample_grid(
            #     "video_detection", sample_shape,
            #     window_overlap=window_overlap)

            n_pos = len(full_sample_grid["positives"])
            n_neg = len(full_sample_grid["negatives"])

            max_neg = int(max(1, (neg_to_pos_ratio * n_pos)))
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

        bg_catname = ub.peek(self.background_classes)
        self.bg_idx = self.classes.node_to_idx[bg_catname]

        # bg_node = graph.nodes['background']
        # if 'color' not in bg_node:
        #     bg_node['color'] = (0., 0., 0.)
        utils.category_tree_ensure_color(self.classes)

        self.sample_shape = sample_shape
        self.channels = channels
        self.mode = mode

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
            >>> from watch.tasks.fusion.datamodules.kwcoco_video_data import *  # NOQA
            >>> import ndsampler
            >>> import kwcoco
            >>> coco_dset = kwcoco.CocoDataset.demo('vidshapes2-multispectral', num_frames=5)
            >>> sampler = ndsampler.CocoSampler(coco_dset)
            >>> channels = 'B10|B8a|B1|B8'
            >>> sample_shape = (5, 530, 610)
            >>> self = KWCocoVideoDataset(sampler, sample_shape=sample_shape, channels=channels)
            >>> item = self[0]
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

        # TODO: perterb the spatial and time sample coordinates
        do_shift = False
        sampler = self.sampler
        tr['as_xarray'] = False
        tr['use_experimental_loader'] = 1
        if not self.disable_augmenter and self.mode == 'fit':
            do_shift = np.random.rand() > 0.5
        if not do_shift:
            # collect sample
            sample = sampler.load_sample(tr, padkw={'constant_values': np.nan})
        else:
            rng = kwarray.ensure_rng(132)
            tr_ = tr.copy()
            aff = kwimage.Affine.coerce(offset=rng.randint(-8, 8, size=2))
            space_box = kwimage.Boxes.from_slice(tr['space_slice']).warp(aff).quantize()
            tr_['space_slice'] = space_box.astype(int).to_slices()[0]
            sample = sampler.load_sample(tr_, padkw=dict(constant_values=np.nan))

        if 0:
            # debug
            dset = sampler.dset
            vid_box = kwimage.Boxes.from_slice(tr['space_slice'])
            for gid in tr['gids']:
                warp_img_to_vid = kwimage.Affine.coerce(
                    dset.index.imgs[gid].get('warp_img_to_vid', None))
                img_box = vid_box.warp(warp_img_to_vid.inv())
                aids = sampler.regions._isect_index.overlapping_aids(gid, img_box)
                qtree = sampler.regions._isect_index.qtrees[gid]
                for aid in aids:
                    annot_box = kwimage.Boxes(qtree.aid_to_tlbr[aid][None, :], 'tlbr')
                    annot_box.warp(warp_img_to_vid.inv())
                    img_box.iooas(annot_box)

                    raise Exception
                print('aids = {!r}'.format(aids))

        # Access the sampled image and annotation data
        raw_frame_list = sample['im']

        # TODO: use this
        nodata_mask = np.isnan(raw_frame_list)  # NOQA
        raw_frame_list = np.nan_to_num(raw_frame_list)

        raw_det_list = sample['annots']['frame_dets']
        raw_gids = sample['tr']['gids']

        channel_keys = sample['tr']['_coords']['c'].values.tolist()
        mode_key = '|'.join(channel_keys)

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

        prev_frame_cids = None

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

            frame_cidxs = np.full(frame.shape[:2], dtype=np.int32,
                                  fill_value=bg_idx)

            frame_ignore = np.full(frame.shape[:2], dtype=np.uint8,
                                   fill_value=0)

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
                if catname in self.ignore_classes:
                    poly.fill(frame_ignore, value=1)
                else:
                    poly.fill(frame_cidxs, value=cidx)

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
                frame_change = (frame_cidxs != prev_frame_cids).astype(np.uint8)
                # Clean up the change target
                frame_change = utils.morphology(frame_change, 'open', kernel=3)
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
                'class_idxs': torch.from_numpy(frame_cidxs),
                'ignore': torch.from_numpy(frame_ignore),
            }
            prev_frame_cids = frame_cidxs
            frame_items.append(frame_item)

        vidid = sample['tr']['vidid']
        video = self.sampler.dset.index.videos[vidid]

        item = {
            # TODO: breakup modes into different items
            "index": index,
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
            ('depends_version', 4),  # bump if `compute_input_stats` changes
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
            >>> from watch.tasks.fusion.datamodules.kwcoco_video_data import *  # NOQA
            >>> import ndsampler
            >>> import kwcoco
            >>> coco_dset = kwcoco.CocoDataset.demo('vidshapes2-multispectral', num_frames=5)
            >>> sampler = ndsampler.CocoSampler(coco_dset)
            >>> sample_shape = (2, 128, 128)
            >>> self = KWCocoVideoDataset(sampler, sample_shape=sample_shape, channels=None)
            >>> self.compute_input_stats()

        Example:
            >>> from watch.tasks.fusion.datamodules.kwcoco_video_data import *  # NOQA
            >>> import ndsampler
            >>> import kwcoco
            >>> coco_dset = kwcoco.CocoDataset.demo('vidshapes2')
            >>> sampler = ndsampler.CocoSampler(coco_dset)
            >>> sample_shape = (2, 128, 128)
            >>> self = KWCocoVideoDataset(sampler, sample_shape=sample_shape, channels=None)
            >>> self.compute_input_stats()

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
            >>> coco_fpath = join(dvc_dpath, 'drop1-S2-L8-aligned/combo_train_data.kwcoco.json')
            >>> coco_fpath = join(dvc_dpath, 'drop1-S2-L8-aligned/rutgers_material_seg.kwcoco.json')
            >>> coco_dset = kwcoco.CocoDataset(coco_fpath)
            >>> sampler = ndsampler.CocoSampler(coco_dset)
            >>> sample_shape = (3, 96, 96)
            >>> #channels = 'blue|green|red|nir|inv_sort1|inv_sort2|inv_sort3|inv_sort4|inv_sort5|inv_sort6|inv_sort7|inv_sort8|inv_augment1|inv_augment2|inv_augment3|inv_augment4|inv_augment5|inv_augment6|inv_augment7|inv_augment8|inv_overlap1|inv_overlap2|inv_overlap3|inv_overlap4|inv_overlap5|inv_overlap6|inv_overlap7|inv_overlap8|inv_shared1|inv_shared2|inv_shared3|inv_shared4|inv_shared5|inv_shared6|inv_shared7|inv_shared8|matseg_0|matseg_1|matseg_2|matseg_3|matseg_4|matseg_5|matseg_6|matseg_7|matseg_8|matseg_9|matseg_10|matseg_11|matseg_12|matseg_13|matseg_14|matseg_15|matseg_16|matseg_17|matseg_18|matseg_19|rice_field|cropland|water|inland_water|river_or_stream|sebkha|snow_or_ice_field|bare_ground|sand_dune|built_up|grassland|brush|forest|wetland|road'
            >>> channels = 'rice_field|cropland|water|inland_water|river_or_stream|sebkha|snow_or_ice_field|bare_ground|sand_dune|built_up|grassland|brush|forest|wetland|road'
            >>> channels = 'matseg_0|matseg_1|matseg_2|matseg_3|matseg_4|matseg_5|matseg_6|matseg_7|matseg_8|matseg_9|matseg_10|matseg_11|matseg_12|matseg_13|matseg_14|matseg_15|matseg_16|matseg_17|matseg_18|matseg_19'
            >>> self = KWCocoVideoDataset(sampler, sample_shape=sample_shape, channels=channels)
            >>> item = self[0]
            >>> #self.compute_input_stats(num=10)
            >>> self.compute_input_stats(num=1000, num_workers=4, batch_size=1)


        Ignore:
            _ = xdev.profile_now(self.__getitem__)(0)
            _ = xdev.profile_now(self.compute_input_stats)(num=10, num_workers=4, batch_size=1)
            tr = self.sample_grid[0]
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
        loader = torch.utils.data.DataLoader(
            stats_subset,
            collate_fn=ub.identity, num_workers=num_workers, shuffle=True,
            batch_size=batch_size)

        # Track moving average of each fused channel stream
        channel_stats = {key: kwarray.RunningStats()
                         for key in self.channels.keys()}

        timer = ub.Timer().tic()
        timer.first = 1
        prog = ub.ProgIter(loader, desc='estimate mean/std')
        for batch_items in prog:
            for item in batch_items:
                for frame_item in item['frames']:
                    for mode_code, mode_val in frame_item['modes'].items():
                        running = channel_stats[mode_code]
                        val = mode_val.numpy()
                        flags = np.isfinite(val)
                        if not np.all(flags):
                            # Hack it:
                            val[~flags] = 0
                        running.update(val.astype(np.float64))

            for key, running in channel_stats.items():
                perchan_stats = running.summarize(axis=(1, 2))

            if timer.first or timer.toc() > 5:
                curr = ub.dict_isect(running.summarize(keepdims=False), {'mean', 'std', 'max', 'min'})
                curr = ub.map_vals(float, curr)
                text = ub.repr2(curr, compact=1, precision=1, nl=0)
                prog.set_postfix_str(text)
                timer.first = 0
                timer.tic()

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
    def draw_item(self, item, item_output=None, combinable_extra=None,
                  max_channels=5, max_dim=256, norm_over_time=0):
        """
        Visualize an item produced by this DataSet.

        Args:
            item (Dict): An item returned from the torch Dataset.
                (It is a dict right? { ಠ ︿ ಠ } )

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
            >>> canvas = self.draw_item(item, item_output, combinable_extra=combinable_extra)
            >>> # xdoctest: +REQUIRES(--show)
            >>> import kwplot
            >>> kwplot.autompl()
            >>> kwplot.imshow(canvas)
            >>> kwplot.show_if_requested()
        """
        classes = self.classes

        # Hacks: combinable channels can be visualized as RGB images.
        # The only reason this is a hack is because of the hardcoded names
        # otherwise it is a cool feature.
        default_combinable_channels = [
            ub.oset(['red', 'green', 'blue']),
            ub.oset(['r', 'g', 'b'])
        ]
        combinable_channels = default_combinable_channels
        if combinable_extra is not None:
            combinable_channels += list(map(ub.oset, combinable_extra))

        def make_frame_infos():
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
                for iterx, chanxs in enumerate(chans_to_use):
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
                        norm_signal = kwimage.atleast_3channels(norm_signal)
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

            # normalize across time?
            if norm_over_time:
                for chans_over_time in zip(*[frame_meta['chan_rows'] for frame_meta in frame_metas]):
                    flat = [c['raw_signal'].ravel() for c in chans_over_time]
                    cums = np.cumsum(list(map(len, flat)))
                    combo = np.hstack(flat)
                    combo_normed = kwimage.normalize_intensity(combo).copy()
                    flat_normed = np.split(combo_normed, cums)
                    for row, flat_item in zip(chans_over_time, flat_normed):
                        norm_signal = flat_item.reshape(*row['raw_signal'].shape)
                        norm_signal = kwimage.atleast_3channels(norm_signal)
                        row['norm_signal'] = norm_signal

            # chan = kwimage.normalize_intensity(chan)
            # signal = kwimage.atleast_3channels(chan).copy()

            return frame_metas

        frame_metas = make_frame_infos()

        horizontal_stack = []
        for frame_meta in frame_metas:
            # Start building the visualization
            vertical_stack = []

            # frame_text = f't={frame_idx} - {date_captured}\n{full_mode_code}'
            # frame_header = kwimage.draw_text_on_image(
            #     {'width': max_dim}, frame_text, org=(max_dim // 2, 5), valign='top',
            #     halign='center', color='purple')

            frame_idx = frame_meta['frame_idx']
            frame_item = frame_meta['frame_item']
            chan_rows = frame_meta['chan_rows']
            full_mode_code = frame_meta['full_mode_code']
            class_idxs = frame_meta['class_idxs']
            changes = frame_meta['changes']
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
                header_part = header_text(f'{full_mode_code}', shrink=True)
                vertical_stack.append(header_part)

            for iterx, row in enumerate(chan_rows):
                norm_signal = row['norm_signal']
                true_layers = []
                if iterx == 0:
                    # Draw class label on odd frames
                    true_heatmap = kwimage.Heatmap(class_idx=class_idxs, classes=classes)
                    # true_part = heatmap.draw_on(true_part, with_alpha=0.5)
                    # Hack: -1 is given the last color by colorize, it would
                    # be better if there was a non-negative background class index
                    class_overlay = true_heatmap.colorize('class_idx')
                    class_overlay[..., 3] = 0.5
                    # class_overlay[class_idxs == -1, 3] = 0
                    true_layers.append(class_overlay)
                    label_text = 'true class'
                elif iterx == 1:
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

                true_layers.append(norm_signal)
                true_part = kwimage.overlay_alpha_layers(true_layers)[..., 0:3]

                true_part = kwimage.imresize(true_part, max_dim=max_dim).clip(0, 1)
                true_part = kwimage.draw_text_on_image(
                    true_part, row['signal_text'], (1, 1), valign='top',
                    color='white', border=3)

                if label_text:
                    # TODO: make draw_text_on_image able to return the
                    # geometry of what it drew and use that.
                    signal_bottom_y = 31  # hack: hardcoded
                    true_part = kwimage.draw_text_on_image(
                        true_part, label_text, (1, signal_bottom_y + 1),
                        valign='top', color='lime', border=3)
                vertical_stack.append(true_part)

            key = 'class_probs'
            if item_output and  key in item_output:
                norm_signal = chan_rows[0]['norm_signal']
                # print('norm_signal.shape = {!r}'.format(norm_signal.shape))
                x = item_output[key][frame_idx]
                # print('x.shape = {!r}'.format(x.shape))
                class_probs = einops.rearrange(x, 'h w c -> c h w')
                # print('class_probs.shape = {!r}'.format(class_probs.shape))
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
                        {'width': w, 'height': h},
                        'X', org=(w // 2, h // 2),
                        valign='center', halign='center', fontScale=10,
                        color='red')
                    pred_part = pred_mask
                else:
                    pred_raw = item_output[key][frame_idx - 1]
                    # Draw predictions on the first item
                    pred_mask = kwimage.make_heatmask(pred_raw)
                    norm_signal = chan_rows[1 if len(chan_rows) > 1 else 0]['norm_signal']
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
        loader = torch.utils.data.DataLoader(
            self, batch_size=batch_size, num_workers=num_workers,
            shuffle=shuffle, pin_memory=pin_memory, collate_fn=ub.identity)
        return loader


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


def new_video_sample_grid(dset, window_dims=None, window_overlap=0.0,
                          space_dims=None, time_dim=None,  # TODO
                          classes_of_interest=None, ignore_coverage_thresh=0.6,
                          negative_classes={'ignore', 'background'}):
    """
    PORTED FROM NDSAMPLER TO FIX A BIG

    Example:
        >>> from watch.tasks.fusion.datamodules.kwcoco_video_data import *  # NOQA
        >>> import kwcoco
        >>> dset = kwcoco.CocoDataset.demo('vidshapes8-multispectral', num_frames=5)
        >>> dset.conform()
        >>> window_dims = (2, 224, 224)
        >>> sample_grid = new_video_sample_grid(dset, window_dims)
        >>> print('sample_grid = {}'.format(ub.repr2(sample_grid, nl=2)))
        >>> # Now try to load a sample
        >>> tr = sample_grid['positives'][0]
        >>> import ndsampler
        >>> sampler = ndsampler.CocoSampler(dset)
        >>> tr_ = sampler._infer_target_attributes(tr)
        >>> print('tr_ = {}'.format(ub.repr2(tr_, nl=1)))
        >>> sample = sampler.load_sample(tr)
        >>> assert sample['im'].shape == (2, 224, 224, 5)

    Ignore:
        import xdev
        globals().update(xdev.get_func_kwargs(new_video_sample_grid))
    """
    import kwarray
    from ndsampler import isect_indexer
    keepbound = True

    if classes_of_interest:
        raise NotImplementedError

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

    @ub.memoize
    def _lut_warp(gid):
        warp_img_to_vid = dset.index.imgs[gid].get('warp_img_to_vid', None)
        return kwimage.Affine.coerce(warp_img_to_vid)

    # NOTE: this is in IMAGE space, not video space
    _isect_index = isect_indexer.FrameIntersectionIndex.from_coco(dset)

    positives = []
    negatives = []
    for vidid, slider in vidid_to_slider.items():
        video_regions = list(slider)
        gids = dset.index.vidid_to_gids[vidid]
        for vid_region in video_regions:
            t_sl, y_sl, x_sl = vid_region
            vid_box = kwimage.Boxes.from_slice((y_sl, x_sl))
            region_gids = gids[t_sl]
            region_aids = []
            for gid in region_gids:
                warp_img_to_vid = _lut_warp(gid)
                # Check to see what annotations this window-box overlaps with
                # (in image space!)
                img_box = vid_box.warp(warp_img_to_vid.inv())
                aids = _isect_index.overlapping_aids(gid, img_box)
                for aid in aids:
                    cid = dset.index.anns[aid]['category_id']
                    catname = dset.index.cats[cid]['name']
                    if catname not in negative_classes:
                        region_aids.append(aid)

            pos_aids = sorted(region_aids)
            time_slice = vid_region[0]
            space_slice = vid_region[1:3]
            tr = {
                'vidid': vidid,
                'time_slice': time_slice,
                'space_slice': space_slice,
                # 'slices': region,
                'gids': region_gids,
                'aids': pos_aids,
            }
            if len(pos_aids):
                positives.append(tr)
            else:
                negatives.append(tr)

    print('Found {} positives'.format(len(positives)))
    print('Found {} negatives'.format(len(negatives)))
    sample_grid = {
        'positives': positives,
        'negatives': negatives,
    }
    return sample_grid
