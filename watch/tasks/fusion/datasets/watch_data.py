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

from functools import partial  # NOQA
from kwcoco import channel_spec
from torch import nn
from torch.utils import data
from torchvision import transforms
from watch.tasks.fusion import utils


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
        >>> )
        >>> self.setup("fit")
        >>> dl = self.train_dataloader()
        >>> batch = next(iter(dl))
        >>> expect_shape = (batch_size, time_steps, len(chan_spec), chip_size, chip_size)
        >>> assert len(batch) == batch_size
        >>> for item in batch:
        ...     for mode_key, mode_val in item['modes'].items():
        ...         assert mode_val.shape[0] == time_steps
        ...         assert mode_val.shape[2:4] == (chip_size, chip_size)
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
    ):
        super().__init__()
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
                self.preprocessing_step,
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
            >>> T, C, H, W = ub.peek(item['modes'].values()).shape
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
        return canvas

    def setup(self, stage):
        if stage == "fit" or stage is None:
            train_data = self.train_kwcoco
            if isinstance(train_data, pathlib.Path):
                train_data = str(train_data.expanduser())
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

            if self.vali_kwcoco is not None:
                # Explicit validation dataset should be prefered
                vali_data = self.vali_kwcoco
                if isinstance(vali_data, pathlib.Path):
                    vali_data = str(vali_data.expanduser())
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
            test_coco_dset = kwcoco.CocoDataset.coerce(test_data)
            self.coco_datasets['train'] = test_coco_dset
            test_coco_sampler = ndsampler.CocoSampler(test_coco_dset)
            self.torch_datasets['test'] = WatchVideoDataset(
                test_coco_sampler,
                sample_shape=(self.time_steps, self.chip_size, self.chip_size),
                window_overlap=(self.time_overlap, self.chip_overlap, self.chip_overlap),
                channels=self.channels,
                # transform=self.test_tfms,
            )

            ub.inject_method(self, lambda self: self._make_dataloader('train', shuffle=True), 'test_dataloader')

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
        # parser.add_argument("--vali_dataset", default=None, type=pathlib.Path)
        parser.add_argument("--test_dataset", default=None, type=pathlib.Path)
        parser.add_argument("--time_steps", default=2, type=int)
        parser.add_argument("--chip_size", default=128, type=int)
        parser.add_argument("--time_overlap", default=0, type=int)
        parser.add_argument("--chip_overlap", default=0.1, type=float)
        parser.add_argument("--channels", default=None, type=str)
        parser.add_argument("--valid_pct", default=0.1, type=float)
        parser.add_argument("--batch_size", default=4, type=int)
        parser.add_argument("--num_workers", default=4, type=int)
        parser.add_argument("--transform_key", default="none", type=str)
        parser.add_argument("--tfms_scale", default=2000., type=float)
        parser.add_argument("--tfms_window_size", default=8, type=int)
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
        channels = channel_spec.ChannelSpec.coerce(channels)

        if transform is not None:
            raise Exception('I do not like injecting the transforms')

        full_sample_grid = sampler.new_sample_grid(
            "video_detection", sample_shape,
            window_overlap=window_overlap)

        sample_grid = list(it.chain(
            full_sample_grid["positives"],
            full_sample_grid["negatives"],
        ))

        self.sample_grid = sample_grid
        self.transform = transform
        self.window_overlap = window_overlap
        self.sampler = sampler
        self.sample_shape = sample_shape
        self.channels = channels
        self.mode = mode
        # self.num_channels = len(channels)

    def __len__(self):
        return len(self.sample_grid)

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
        """

        # get positive sample definition
        tr = self.sample_grid[index]
        if self.channels:
            tr["channels"] = self.channels

        tr['as_xarray'] = False
        tr['use_experimental_loader'] = True
        # collect sample
        sample = self.sampler.load_sample(tr)

        channel_keys = sample['tr']['_coords']['c'].values.tolist()
        mode_code = '|'.join(channel_keys)

        # Access the sampled image and annotation data
        raw_frame_list = sample['im']
        raw_det_list = sample['annots']['frame_dets']

        # Break data down on a per-frame basis so we can apply image-based
        # augmentations.
        frame_ims = []
        frame_masks = []
        for frame, dets in zip(raw_frame_list, raw_det_list):
            frame = np.asarray(frame, dtype=np.float32)
            input_dsize = self.sample_shape[-2:][::-1]

            input_dsize = [
                real if (nominal is None) else nominal
                for nominal, real in zip(input_dsize, frame.shape[:2][::-1])
            ]

            # Resize the sampled window to the target space for the network
            frame, info = kwimage.imresize(frame, dsize=input_dsize,
                                           interpolation='linear',
                                           antialias=True,
                                           return_info=True)
            # Remember to apply any transform to the dets as well
            dets = dets.scale(info['scale'])
            dets = dets.translate(info['offset'])

            frame_mask = np.full(frame.shape[:2], dtype=np.int32, fill_value=-1)
            ann_polys = dets.data['segmentations'].to_polygon_list()
            ann_aids = dets.data['aids']
            ann_cids = dets.data['cids']

            for poly, aid, cid in zip(ann_polys, ann_aids, ann_cids):
                cidx = self.sampler.classes.id_to_idx[cid]
                poly.fill(frame_mask, value=cidx)

            # ensure channel dim is not squeezed
            frame = kwarray.atleast_nd(frame, 3)

            frame_masks.append(frame_mask)
            frame_ims.append(frame)

        # stack along temporal axis
        frame_ims = np.stack(frame_ims, axis=0)

        # DO NOT ADD 1, THE CLASS INDEXES SHOULD BE RESPECTED
        # frame_masks = np.stack(frame_masks, axis=0) + 1
        frame_masks = np.stack(frame_masks, axis=0)

        frame_ignores = np.zeros_like(frame_masks)
        # frame_ignores = (frame_masks == self.occlusion_class_id)

        # rearrange image axes for pytorch
        frame_ims = einops.rearrange(frame_ims, "t h w c -> t c h w")

        # catch nans
        frame_ims[np.isnan(frame_ims)] = -1.

        # convert to torch
        frame_ims = torch.from_numpy(frame_ims.astype("float")).float()
        frame_masks = torch.from_numpy(frame_masks)
        frame_ignores = torch.from_numpy(frame_ignores)

        # if self.transform:
        #     frame_ims = self.transform(frame_ims)

        # if self.mode == "predict":
        #     return frame_ims
        # images, labels = batch["images"].float(), batch["labels"]

        # convert annotations into a change detection task suitable for the
        # network.
        changes = frame_masks[1:] != frame_masks[:-1]

        item = {
            # TODO: breakup modes into different items
            "modes": {
                mode_code: frame_ims,
            },
            "labels": frame_masks,
            "changes": changes,
            "ignore": frame_ignores,
        }

        return item

    def cached_input_stats(self):
        """
        Compute the normalization stats, and caches them

        TODO:
            - [ ] Does this dataset have access to the workdir?
            - [ ] Cacher needs to depend on config of this dataset
        """
        # Get stats on the dataset (todo: nice way to disable augmentation temporarilly for this)
        depends = ub.odict([
            ('hashid', self.sampler.dset._build_hashid()),
            ('channels', self.channels.__json__()),
            ('sample_shape', self.sample_shape),
        ])
        workdir = None
        cacher = ub.Cacher('dset_mean', dpath=workdir, depends=depends + 'v8')
        input_stats = cacher.tryload()
        if input_stats is None:
            cacher.save(input_stats)

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

        # TODO: disable augmentation if we are doing that
        loader = torch.utils.data.DataLoader(
            stats_subset,
            collate_fn=ub.identity, num_workers=num_workers, shuffle=True,
            batch_size=batch_size)

        # Track moving average of each fused channel stream
        channel_stats = {key: kwarray.RunningStats()
                         for key in self.channels.keys()}

        for batch_items in ub.ProgIter(loader, desc='estimate mean/std'):
            for item in batch_items:
                for mode_code, mode_val in item['modes'].items():
                    channel_stats[mode_code].update(mode_val.numpy())

        input_stats = {}
        for key, running in channel_stats.items():
            perchan_stats = running.summarize(axis=(0, 2, 3))
            input_stats[key] = {
                'std': perchan_stats['mean'].round(3),
                'mean': perchan_stats['std'].round(3),
            }
        # _dset.disable_augmenter = False  # hack
        return input_stats

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
            >>> binprobs = np.random.rand(*self.sample_shape)
            >>> canvas = self.draw_item(item, binprobs)
            >>> # xdoctest: +REQUIRES(--show)
            >>> import kwplot
            >>> kwplot.autompl()
            >>> kwplot.imshow(canvas)
            >>> kwplot.show_if_requested()
        """
        import watch
        # import kwcoco
        min_dim = 296
        # chan_names = kwcoco.channel_spec.FusedChannelSpec.coerce(self.channels).as_list()
        classes = self.sampler.classes

        # hack just use one of the modes
        mode_code, mode_data = ub.peek(item['modes'].items())
        chan_names = mode_code.split('|')

        frame_ims = mode_data.data.cpu().numpy()
        frame_masks = item['labels'].data.cpu().numpy()
        frame_ims = watch.utils.util_norm.normalize_intensity(frame_ims, axis=1)
        frame_list = []
        for frame_idx, (im_chw, mask) in enumerate(zip(frame_ims, frame_masks)):
            chan_list = []
            for chan_idx, chan in enumerate(im_chw):
                chan_name = chan_names[chan_idx]

                signal = kwimage.atleast_3channels(chan)

                # cidxs = mask

                true_heatmap = kwimage.Heatmap(class_idx=mask, classes=classes)
                # true_part = heatmap.draw_on(true_part, with_alpha=0.5)
                # Hack: -1 is given the last color by colorize, it would
                # be better if there was a non-negative background class index
                class_overlay = true_heatmap.colorize('class_idx')
                class_overlay[..., 3] = 0.5
                class_overlay[mask == -1, 3] = 0
                text = 't={}, c={}:{}'.format(frame_idx, chan_idx, chan_name)

                true_part = kwimage.overlay_alpha_layers([class_overlay, signal])
                true_part = true_part[..., 0:3]
                true_part = kwimage.imresize(true_part, min_dim=min_dim).clip(0, 1)
                true_part = kwimage.draw_text_on_image(
                    true_part, text, (1, 1), valign='top', color='limegreen')

                if binprobs is not None:
                    # TODO: we could make this visualization better
                    if frame_idx == 0:
                        #
                        pred_part = np.zeros_like(true_part)
                    else:
                        # Hack, output is assumed to only be for subsequent
                        # frames it would be better if we had some sort of
                        # standardized output encoding.
                        pred_mask = kwimage.make_heatmask(binprobs[frame_idx - 1])
                        pred_part = kwimage.overlay_alpha_layers([pred_mask, signal])
                        pred_part = pred_part[..., 0:3]
                        pred_part = kwimage.imresize(pred_part, min_dim=min_dim).clip(0, 1)
                    pred_part = kwimage.draw_text_on_image(
                        pred_part, 'pred ' + text, (1, 1), valign='top', color='blue')
                    part = kwimage.stack_images([true_part, pred_part], axis=1)
                else:
                    part = true_part

                chan_list.append(part)
            frame_canvas = kwimage.stack_images(chan_list)
            frame_list.append(frame_canvas)
        canvas = kwimage.stack_images(frame_list, axis=1)
        canvas = canvas[..., 0:3]  # drop alpha
        canvas = kwimage.ensure_uint255(canvas)  # convert to uint8
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
