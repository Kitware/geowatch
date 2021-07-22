import kwimage
import kwarray
import numpy as np
from torch.utils import data
from torch import nn
import torch
import einops
from kwcoco import channel_spec
import itertools as it
import ubelt as ub


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
        >>> from watch.tasks.fusion.datasets.common import *  # NOQA
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


class VideoDataset(data.Dataset):
    """
    Example:
        >>> from watch.tasks.fusion.datasets.common import *  # NOQA
        >>> import ndsampler
        >>> import kwcoco
        >>> coco_dset = kwcoco.CocoDataset.demo('vidshapes2-multispectral', num_frames=5)
        >>> coco_dset.ensure_category('background')
        >>> sampler = ndsampler.CocoSampler(coco_dset)
        >>> channels = 'B10|B8a|B1|B8'
        >>> sample_shape = (3, 530, 610)
        >>> self = VideoDataset(sampler, sample_shape=sample_shape, channels=channels)
        >>> index = len(self) // 4
        >>> item = self[index]
        >>> canvas = self.draw_item(item)
        >>> # xdoctest: +REQUIRES(--show)
        >>> import kwplot
        >>> kwplot.autompl()
        >>> kwplot.imshow(canvas)
        >>> kwplot.show_if_requested()

    Example:
        >>> from watch.tasks.fusion.datasets.common import *  # NOQA
        >>> import ndsampler
        >>> import kwcoco
        >>> coco_dset = kwcoco.CocoDataset.demo('vidshapes2-multispectral', num_frames=5)
        >>> coco_dset.ensure_category('background')
        >>> sampler = ndsampler.CocoSampler(coco_dset)
        >>> sample_shape = (2, 128, 128)
        >>> self = VideoDataset(sampler, sample_shape=sample_shape, channels=None)
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
    def __init__(self, sampler, sample_shape, channels=None, mode="fit",
                 window_overlap=0, transform=None, occlusion_class_id=1):

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
        self.occlusion_class_id = occlusion_class_id
        self.sampler = sampler
        self.sample_shape = sample_shape
        self.channels = channels
        self.mode = mode
        # self.num_channels = len(channels)

    def __len__(self):
        return len(self.sample_grid)

    def __getitem__(self, index):

        # get positive sample definition
        tr = self.sample_grid[index]
        if self.channels:
            tr["channels"] = self.channels

        tr['as_xarray'] = False
        # collect sample
        sample = self.sampler.load_sample(tr)
        # channel_keys = sample['im'].coords['c'].values.tolist()

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

        frame_ignores = (frame_masks == self.occlusion_class_id)

        # rearrange image axes for pytorch
        frame_ims = einops.rearrange(frame_ims, "t h w c -> t c h w")

        # catch nans
        frame_ims[np.isnan(frame_ims)] = -1.

        # convert to torch
        frame_ims = torch.from_numpy(frame_ims.astype("float"))
        frame_masks = torch.from_numpy(frame_masks)
        frame_ignores = torch.from_numpy(frame_ignores)

        # if self.transform:
        #     frame_ims = self.transform(frame_ims)

        # if self.mode == "predict":
        #     return frame_ims

        example = {
            # "channel_keys": channel_keys,
            "images": frame_ims,
            "labels": frame_masks,
            "ignore": frame_ignores,
        }

        return example

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
            >>> from watch.tasks.fusion.datasets.common import *  # NOQA
            >>> import ndsampler
            >>> import kwcoco
            >>> coco_dset = kwcoco.CocoDataset.demo('vidshapes2-multispectral', num_frames=5)
            >>> coco_dset.ensure_category('background')
            >>> sampler = ndsampler.CocoSampler(coco_dset)
            >>> sample_shape = (2, 128, 128)
            >>> self = VideoDataset(sampler, sample_shape=sample_shape, channels=None)

        Example:
            >>> # xdoctest: +REQUIRES(env:DVC_DPATH)
            >>> # Run the following tests on real watch data if DVC is available
            >>> from watch.tasks.fusion.datasets.common import *  # NOQA
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
            >>> self = VideoDataset(sampler, sample_shape=sample_shape, channels=None)
            >>> self.compute_input_stats()
        """
        # Use parallel workers to load data faster
        # from netharn.data.data_containers import container_collate
        # from functools import partial
        # collate_fn = partial(container_collate, num_devices=1)

        num = None
        num = num if isinstance(num, int) and num is not True else 1000
        stats_idxs = kwarray.shuffle(np.arange(len(self)), rng=0)[0:min(num, len(self))]
        stats_subset = torch.utils.data.Subset(self, stats_idxs)

        # TODO: disable augmentation if we are doing that

        loader = torch.utils.data.DataLoader(
            stats_subset,
            # collate_fn=collate_fn,
            num_workers=num_workers, shuffle=True, batch_size=batch_size)

        # Track moving average of each fused channel stream
        channel_stats = {key: kwarray.RunningStats()
                         for key in self.channels.keys()}

        for batch in ub.ProgIter(loader, desc='estimate mean/std'):
            # Current single image case:
            assert len(channel_stats) == 1, 'update me in the future'
            key = ub.peek(channel_stats.keys())
            part = batch['images'].numpy()
            channel_stats[key].update(part)

            # Future multimodal case:
            # if 0:
            #     for key, val in batch['inputs'].items():
            #         try:
            #             for batch_part in val.data:
            #                 for part in batch_part.numpy():
            #                     channel_stats[key].update(part)
            #         except ValueError:  # final batch broadcast error
            #             pass

        input_stats = {}
        for key, running in channel_stats.items():
            perchan_stats = running.summarize(axis=(0, 1, 3, 4))
            input_stats[key] = {
                'std': perchan_stats['mean'].round(3),
                'mean': perchan_stats['std'].round(3),
            }

        return input_stats
        # _dset.disable_augmenter = False  # hack

    def draw_item(self, item):
        import watch
        import kwcoco
        min_dim = 296
        chan_names = kwcoco.channel_spec.FusedChannelSpec.coerce(self.channels).as_list()
        classes = self.sampler.classes
        frame_ims = item['images'].numpy()
        frame_masks = item['labels'].numpy()
        frame_ims = watch.utils.util_norm.normalize_intensity(frame_ims, axis=1)
        frame_list = []
        for frame_idx, (im_chw, mask) in enumerate(zip(frame_ims, frame_masks)):
            chan_list = []
            for chan_idx, chan in enumerate(im_chw):
                chan_name = chan_names[chan_idx]
                # cidxs = mask
                heatmap = kwimage.Heatmap(class_idx=mask, classes=classes)
                # Hack: -1 is given the last color by colorize, it would
                # be better if there was a non-negative background class index
                class_overlay = heatmap.colorize('class_idx')
                class_overlay[..., 3] = 0.5
                class_overlay[mask == -1, 3] = 0
                part = kwimage.overlay_alpha_layers([class_overlay, chan])
                text = 't={}, c={}:{}'.format(frame_idx, chan_idx, chan_name)
                # part = chan
                part = kwimage.atleast_3channels(part)
                # part = heatmap.draw_on(part, with_alpha=0.5)
                part = kwimage.imresize(part, min_dim=min_dim)
                part = part.clip(0, 1)
                part = kwimage.draw_text_on_image(
                    part, text, (1, 1), valign='top', color='limegreen')

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
            >>> from watch.tasks.fusion.datasets.common import *  # NOQA
            >>> import ndsampler
            >>> import kwcoco
            >>> coco_dset = kwcoco.CocoDataset.demo('vidshapes2-multispectral', num_frames=5)
            >>> coco_dset.ensure_category('background')
            >>> sampler = ndsampler.CocoSampler(coco_dset)
            >>> self = VideoDataset(sampler, sample_shape=(3, 530, 610))
            >>> loader = self.make_loader(batch_size=2)
            >>> batch = next(iter(loader))
        """
        loader = torch.utils.data.DataLoader(
            self, batch_size=batch_size, num_workers=num_workers,
            shuffle=shuffle, pin_memory=pin_memory)
        return loader
