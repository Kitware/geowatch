import kwarray
import kwimage
import numpy as np
import torch
import ubelt as ub
from functools import partial
from netharn.data.batch_samplers import PatchedBatchSampler
from netharn.data.data_containers import ItemContainer
from netharn.data.data_containers import container_collate
from netharn.data.batch_samplers import PatchedRandomSampler
from netharn.data.batch_samplers import SubsetSampler
import random


class SequenceDataset(torch.utils.data.Dataset):
    """
    Example:
        >>> from geowatch.tasks.rutgers_material_seg.datasets.iarpa_contrastive_dataset import *  # NOQA
        >>> import ndsampler
        >>> import itertools as it
        >>> sampler = ndsampler.CocoSampler.demo('vidshapes8', image_size=(64, 64))
        >>> channels = 'r|g|b'
        >>> window_dims = (2, 128, 128)
        >>> self = SequenceDataset(sampler, window_dims=window_dims, training=False, channels=channels)
        >>> index_iter = it.count()
        >>> index = next(index_iter)
        >>> item = self[index]
        >>> # xdoctest: +REQUIRES(--show)
        >>> # xdoctest: +REQUIRES(module:kwplot)
        >>> import kwplot
        >>> import einops
        >>> kwplot.autompl()
        >>> frames = item['inputs']['im'].data
        >>> frame_masks = item['label']['class_masks'].data
        >>> frames_ = einops.rearrange(frames, 'c t h w -> t c h w').numpy()
        >>> frames_ = kwimage.normalize_intensity(frames_)
        >>> frames_ = np.nan_to_num(frames_)
        >>> pnum_ = kwplot.PlotNums(nSubplots=len(frames_))
        >>> for frame, mask in zip(frames_, frame_masks):
        >>>     kwplot.imshow(frame, pnum=pnum_())
        >>>     heatmap = kwimage.Heatmap(class_idx=mask, classes=self.sampler.classes)
        >>>     heatmap.draw(with_alpha=0.3)

    Example:
        >>> from geowatch.tasks.rutgers_material_seg.datasets.iarpa_contrastive_dataset import *  # NOQA
        >>> import ndsampler
        >>> import itertools as it
        >>> sampler = ndsampler.CocoSampler.demo('vidshapes8-msi', image_size=(64, 64))
        >>> channels = 'B1|B8|B11'
        >>> window_dims = (2, 128, 128)
        >>> self = SequenceDataset(sampler, window_dims=window_dims, training=False, inference_only=True, channels=channels)
        >>> index_iter = it.count()
        >>> index = next(index_iter)
        >>> item = self[index]
        >>> # xdoctest: +REQUIRES(--show)
        >>> # xdoctest: +REQUIRES(module:kwplot)
        >>> import kwplot
        >>> import einops
        >>> kwplot.autompl()
        >>> frames = item['inputs']['im'].data
        >>> assert 'label' not in item
        >>> frames_ = einops.rearrange(frames, 'c t h w -> t c h w').numpy()
        >>> frames_ = kwimage.normalize_intensity(frames_)
        >>> frames_ = np.nan_to_num(frames_)
        >>> pnum_ = kwplot.PlotNums(nSubplots=len(frames_))
        >>> for frame in frames_:
        >>>     kwplot.imshow(frame, pnum=pnum_())
    """

    def __init__(self, sampler, window_dims, input_dims=None, channels=None,
                 rng=None, training=True, window_overlap=0.0,
                 inference_only=False):
        super().__init__()
        if input_dims is None:
            input_dims = window_dims[-2:]
        # print(sampler)
        self.training = training
        self.rng = kwarray.ensure_rng(rng)
        self.sampler = sampler
        self.window_dims = window_dims
        self.input_dims = input_dims
        self.classes = self.sampler.classes
        self.channels = channels
        self.inference_only = inference_only
        # Build a simple space-time-grid
        sample_grid_spec = {
            'task': 'video_detection',
            'window_dims': window_dims,
            'window_overlap': window_overlap,
        }
        # print(sample_grid_spec)
        self.sample_grid = sampler.new_sample_grid(**sample_grid_spec)
        self.training = training

        if self.training:
            self.chosen_samples = self.sample_grid['positives']
        else:
            # In inference we need to load all positive and negative regions
            self.chosen_samples = (
                self.sample_grid['positives'] +
                self.sample_grid['negatives']
            )
            # Reorganize samples so they iterate through images / videos in
            # order
            grouped = ub.group_items(
                self.chosen_samples,
                lambda x: tuple(
                    [x['vidid']] + [gid for gid in x['gids']]
                )
            )
            grouped = ub.sorted_keys(grouped)
            self.chosen_samples = list(ub.flatten(grouped.values()))

    def __len__(self):
        return len(self.chosen_samples)

    def __getitem__(self, index):

        sampler = self.sampler

        tr = self.chosen_samples[index]
        if self.channels:
            tr['channels'] = self.channels

        if not self.inference_only:
            with_annots = 'segmentation'
        else:
            with_annots = False

        # NOTE: Setting nodata=float ensures this returns data in a floating
        # format and also ensures any values that need to be masked are filled
        # with nans.
        sample: dict = sampler.load_sample(
            tr, with_annots=with_annots, nodata='float',
            padkw={'mode': 'constant', 'constant_values': np.nan})

        if self.training:
            # Only need to get a "negative" in training
            negative_index = random.randint(0, self.__len__() - 2)
            tr_negative = self.chosen_samples[negative_index]
            if self.channels:
                tr_negative['channels'] = self.channels
            negative_sample: dict = sampler.load_sample(
                tr_negative, with_annots=with_annots, nodata='float',
                padkw={'mode': 'constant', 'constant_values': np.nan})

        # Access the sampled image and annotation data
        raw_frame_list = sample['im']

        if not self.inference_only:
            raw_det_list = sample['annots']['frame_dets']
        else:
            raw_det_list = [None] * len(raw_frame_list)

        # Break data down on a per-frame basis so we can apply image-based
        # augmentations.
        frame_ims = []
        frame_masks = []
        for raw_frame, raw_dets in zip(raw_frame_list, raw_det_list):
            frame = raw_frame.astype(np.float32)
            input_dsize = self.input_dims[-2:][::-1]
            # Resize the sampled window to the target space for the network
            frame, info = kwimage.imresize(frame, dsize=input_dsize,
                                           interpolation='linear',
                                           return_info=True)

            # ensure channel dim is not squeezed
            frame = kwarray.atleast_nd(frame, 3)
            frame_ims.append(frame)

            if not self.inference_only:
                # Remember to apply any transform to the dets as well
                dets: kwimage.Detections = raw_dets
                dets = dets.scale(info['scale'])
                # print(info)
                dets = dets.translate(info['offset'])
                # print(frame.shape[0:2])
                frame_mask = np.full(frame.shape[0:2], dtype=np.int32, fill_value=-1)
                ann_polys = dets.data['segmentations'].to_polygon_list()
                # print(ann_polys)
                ann_aids = dets.data['aids']
                ann_cids = dets.data['cids']

                for poly, aid, cid in zip(ann_polys, ann_aids, ann_cids):
                    cidx = self.classes.id_to_idx[cid]
                    poly.fill(frame_mask, value=cidx)
                frame_masks.append(frame_mask)

        # Perpare data for torch
        frame_data = np.concatenate([f[None, ...] for f in frame_ims], axis=0)
        cthw_im = frame_data.transpose(3, 0, 1, 2)
        # print(f"image min:{cthw_im.min()}, max:{cthw_im.max()}")
        inputs = {
            'im': ItemContainer(torch.from_numpy(cthw_im).contiguous(), stack=True),
        }

        if not self.inference_only:
            class_masks = np.concatenate([m[None, ...] for m in frame_masks], axis=0)
            label = {
                'class_masks': ItemContainer(
                    torch.from_numpy(class_masks).contiguous(), stack=False, cpu_only=True),
            }

        if self.training:
            negative_raw_frame_list = negative_sample['im']
            negative_raw_det_list = negative_sample['annots']['frame_dets']
            negative_frame_ims = []
            # negative_frame_masks = []
            for negative_raw_frame, negative_raw_dets in zip(negative_raw_frame_list, negative_raw_det_list):
                frame = negative_raw_frame.astype(np.float32)
                # dets = negative_raw_dets
                input_dsize = self.input_dims[-2:][::-1]
                # Resize the sampled window to the target space for the network
                frame, info = kwimage.imresize(frame, dsize=input_dsize,
                                               interpolation='linear',
                                               return_info=True)
                # Remember to apply any transform to the dets as well
                # dets = dets.scale(info['scale'])
                # dets = dets.translate(info['offset'])
                # frame_mask = np.full(frame.shape[0:2], dtype=np.int32, fill_value=-1)
                # ann_polys = dets.data['segmentations'].to_polygon_list()
                # ann_aids = dets.data['aids']
                # ann_cids = dets.data['cids']
                # for poly, aid, cid in zip(ann_polys, ann_aids, ann_cids):
                #     cidx = self.classes.id_to_idx[cid]
                #     poly.fill(frame_mask, value=cidx)

                # ensure channel dim is not squeezed
                frame = kwarray.atleast_nd(frame, 3)

                # fig = plt.figure()
                # ax1 = fig.add_subplot(1,2,1)
                # ax2 = fig.add_subplot(1,2,2)
                # ax1.imshow(frame[:,:,:3])
                # ax2.imshow(frame_mask)
                # plt.show()

                negative_frame_ims.append(frame)
                # negative_frame_masks.append(frame_mask)

            # Prepare data for torch
            negative_frame_data = np.concatenate([f[None, ...] for f in negative_frame_ims], axis=0)
            negative_cthw_im = negative_frame_data.transpose(3, 0, 1, 2)
            # UNUSED? FIXME?
            # negative_class_masks = np.concatenate([m[None, ...] for m in negative_frame_masks], axis=0)  # NOQA
            inputs['negative_im'] = ItemContainer(torch.from_numpy(negative_cthw_im).contiguous(), stack=True)

        item = {
            'inputs': inputs,
            'tr': ItemContainer(sample['tr'], stack=False),
        }
        if not self.inference_only:
            item['label'] = label

        return item

    def make_loader(self, batch_size=16, num_workers=0, shuffle=False,
                    pin_memory=True, drop_last=True, multiscale=False,
                    num_batches='auto', xpu=None):
        """
        Create a loader for this dataset with custom sampling logic and
        container collation.
        """
        if len(self) == 0:
            raise ValueError('must have some data')

        # The case where where replacement is not allowed
        if num_batches == 'auto':
            num_samples = None
        else:
            num_samples = num_batches * batch_size

        if shuffle:
            item_sampler = PatchedRandomSampler(self, num_samples=num_samples)
        else:
            if num_samples is None:
                item_sampler = torch.utils.data.sampler.SequentialSampler(self)
            else:
                stats_idxs = (np.arange(num_samples) % len(self))
                item_sampler = SubsetSampler(stats_idxs)

        if num_samples is not None:
            # If num_batches is too big, error
            if num_samples > len(self):
                raise IndexError(
                    'num_batches={} and batch_size={} causes '
                    'num_samples={} to be greater than the number '
                    'of data items {}. Try setting num_batches=auto?'.format(
                        num_batches, batch_size, num_samples, len(self)))

        batch_sampler = PatchedBatchSampler(
            item_sampler, batch_size=batch_size, drop_last=drop_last,
            num_batches=num_batches)

        if xpu is None:
            num_devices = 1
        else:
            num_devices = len(xpu.devices)

        collate_fn = partial(container_collate, num_devices=num_devices)

        loader = torch.utils.data.DataLoader(
            self, batch_sampler=batch_sampler,
            collate_fn=collate_fn, num_workers=num_workers,
            pin_memory=pin_memory, worker_init_fn=worker_init_fn)
        return loader


def worker_init_fn(worker_id):
    worker_info = torch.utils.data.get_worker_info()
    self = worker_info.dataset

    if hasattr(self.sampler.dset, 'connect'):
        # Reconnect to the backend if we are using SQL
        self.sampler.dset.connect(readonly=True)
