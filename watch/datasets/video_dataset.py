"""
"""
import kwarray
import kwimage
import numpy as np
import torch

# from kwcoco.channel_spec import ChannelSpec  # NOQA
from functools import partial
from netharn.data.batch_samplers import PatchedBatchSampler
from netharn.data.data_containers import ItemContainer
from netharn.data.data_containers import BatchContainer
from netharn.data.data_containers import container_collate
from netharn.data.batch_samplers import PatchedRandomSampler
from netharn.data.batch_samplers import SubsetSampler


class SimpleVideoDataset(torch.utils.data.Dataset):
    """
    Simple video dataset template / example

    Example:
        >>> from watch.datasets.video_dataset import *  # NOQA
        >>> import ndsampler
        >>> sampler = ndsampler.CocoSampler.demo('vidshapes8-multispectral')
        >>> # window_dims = (3, 300, 300)
        >>> window_dims = (3, None, None)
        >>> input_dims = (128, 128)
        >>> self = SimpleVideoDataset(sampler, window_dims, input_dims)
        >>> index = 2
        >>> item = self[index]
        >>> stacked = draw_multispectral_item(item)
        >>> #
        >>> # xdoctest: +REQUIRES(--show)
        >>> import kwplot
        >>> kwplot.autompl()
        >>> kwplot.imshow(stacked)
        >>> #
        >>> # xdoctest: +REQUIRES(--interact)
        >>> import xdev
        >>> for index in xdev.InteractiveIter(list(range(len(self)))):
        >>>     item = self[index]
        >>>     stacked = draw_multispectral_item(item)
        >>>     kwplot.imshow(stacked)
        >>>     xdev.InteractiveIter.draw()

    Example:
        >>> # Show interaction with a derived loader
        >>> from watch.datasets.video_dataset import *  # NOQA
        >>> import ndsampler
        >>> import ubelt as ub
        >>> sampler = ndsampler.CocoSampler.demo('vidshapes8-multispectral')
        >>> window_dims = (2, 100, 100)
        >>> input_dims = (64, 64)
        >>> self = SimpleVideoDataset(sampler, window_dims, input_dims)
        >>> loader = self.make_loader(batch_size=3)
        >>> batch = ub.peek(loader)
        >>> canvas = draw_multispectral_batch(batch)
        >>> # xdoctest: +REQUIRES(--show)
        >>> import kwplot
        >>> kwplot.autompl()
        >>> kwplot.imshow(canvas)
        >>> #
        >>> # xdoctest: +REQUIRES(--interact)
        >>> import xdev
        >>> loader_iter = iter(loader)
        >>> for index in xdev.InteractiveIter(list(range(len(loader)))):
        >>>     batch = next(loader_iter)
        >>>     stacked = draw_multispectral_batch(batch)
        >>>     kwplot.imshow(stacked)
        >>>     xdev.InteractiveIter.draw()

    Example:
        >>> # xdoctest: +SKIP
        >>> # Example with Real Data
        >>> from watch.datasets.video_dataset import *  # NOQA
        >>> import kwcoco
        >>> import ndsampler
        >>> import ubelt as ub
        >>> coco_fpath = ub.expandpath('~/data/dvc-repos/smart_watch_dvc/drop0_aligned/data.kwcoco.json')
        >>> dset = kwcoco.CocoDataset(coco_fpath)
        >>> #
        >>> sampler = ndsampler.CocoSampler(dset)
        >>> print(ub.repr2(self.sample_grid['positives'], nl=-1))
        >>> #
        >>> window_dims = (3, None, None)
        >>> input_dims = (128, 128)
        >>> channels = 'r|g|b|gray|wv1'
        >>> # channels = 'gray'
        >>> self = SimpleVideoDataset(sampler, window_dims, input_dims, channels)
        >>> index = 2
        >>> item = self[index]
        >>> stacked = draw_multispectral_item(item)
        >>> # xdoctest: +REQUIRES(--show)
        >>> import kwplot
        >>> kwplot.autompl()
        >>> kwplot.imshow(stacked)
        >>> #
        >>> # xdoctest: +REQUIRES(--interact)
        >>> import xdev
        >>> loader = self.make_loader(batch_size=3)
        >>> loader_iter = iter(loader)
        >>> for index in xdev.InteractiveIter(list(range(len(loader)))):
        >>>     batch = next(loader_iter)
        >>>     stacked = draw_multispectral_batch(batch)
        >>>     kwplot.imshow(stacked)
        >>>     xdev.InteractiveIter.draw()

    """

    def __init__(self, sampler, window_dims, input_dims=None, channels=None,
                 rng=None):
        super().__init__()

        if input_dims is None:
            input_dims = window_dims[-2:]

        self.rng = kwarray.ensure_rng(rng)
        self.sampler = sampler
        self.window_dims = window_dims
        self.input_dims = input_dims

        self.classes = self.sampler.classes
        self.channels = channels

        # Build a simple space-time-grid
        sample_grid_spec = {
            'task': 'video_detection',
            'window_dims': window_dims
        }
        self.sample_grid = sampler.new_sample_grid(**sample_grid_spec)

    def __len__(self):
        return len(self.sample_grid['positives'])

    def __getitem__(self, index):
        tr = self.sample_grid['positives'][index]

        if self.channels:
            tr['channels'] = self.channels

        sampler = self.sampler
        sample = sampler.load_sample(tr)

        # Access the sampled image and annotation data
        raw_frame_list = sample['im']
        raw_det_list = sample['annots']['frame_dets']

        # Break data down on a per-frame basis so we can apply image-based
        # augmentations.
        frame_ims = []
        frame_masks = []
        for raw_frame, raw_dets in zip(raw_frame_list, raw_det_list):
            frame = raw_frame.astype(np.float32)
            dets = raw_dets
            input_dsize = self.input_dims[-2:][::-1]

            # Resize the sampled window to the target space for the network
            frame, info = kwimage.imresize(frame, dsize=input_dsize,
                                           interpolation='linear',
                                           return_info=True)
            # Remember to apply any transform to the dets as well
            dets = dets.scale(info['scale'])
            dets = dets.translate(info['offset'])

            frame_mask = np.full(frame.shape[0:2], dtype=np.int32, fill_value=-1)
            ann_polys = dets.data['segmentations'].to_polygon_list()
            ann_aids = dets.data['aids']
            ann_cids = dets.data['cids']

            for poly, aid, cid in zip(ann_polys, ann_aids, ann_cids):
                cidx = self.classes.id_to_idx[cid]
                poly.fill(frame_mask, value=cidx)

            # ensure channel dim is not squeezed
            frame = kwarray.atleast_nd(frame, 3)

            frame_masks.append(frame_mask)
            frame_ims.append(frame)

        # Perpare data for torch
        frame_data = np.concatenate([f[None, ...] for f in frame_ims], axis=0)
        class_masks = np.concatenate([m[None, ...] for m in frame_masks], axis=0)

        cthw_im = frame_data.transpose(3, 0, 1, 2)

        inputs = {
            'im': ItemContainer(torch.from_numpy(cthw_im), stack=True),
        }
        label = {
            'class_masks': ItemContainer(
                torch.from_numpy(class_masks), stack=False, cpu_only=True),
        }

        item = {
            'inputs': inputs,
            'label': label,
            'tr': ItemContainer(sample['tr'], stack=False),
        }
        return item

    def make_loader(self, batch_size=16, num_workers=0, shuffle=False,
                    pin_memory=False, drop_last=False, multiscale=False,
                    balance=False, num_batches='auto', xpu=None):
        """
        Create a loader for this dataset with custom sampling logic and
        container collation.
        """
        if len(self) == 0:
            raise ValueError('must have some data')

        if balance:
            # Can use information in self.sample_grid to balance over
            # categories.
            raise NotImplementedError

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


def draw_multispectral_item(item):
    """
    Very basic code to draw an item returned by the dataset
    """
    import kwarray
    import watch
    import kwimage
    _impl = kwarray.ArrayAPI()
    imdata = _impl.numpy(item['inputs']['im'].data).transpose(1, 2, 3, 0)
    maskdata = _impl.numpy(item['label']['class_masks'].data)

    min_dim = 296

    with_text = 1

    canvas_frames = []
    for framex, (frame_im, frame_mask) in enumerate(zip(imdata, maskdata)):
        chan_canvas_list = []
        for chanx, chan in enumerate(frame_im.transpose(2, 0, 1)):
            # Normalize to make visible (should work reasonably even with mean subtract)
            if not np.allclose(chan, chan.min()):
                canvas = np.nan_to_num(watch.utils.util_norm.normalize_intensity(chan))
            else:
                canvas = (chan.astype(np.float32) / max(chan.max(), np.float32(1))).astype(np.float32)
            canvas = canvas.clip(0, 1)
            canvas = kwimage.atleast_3channels(canvas)

            heatmask = kwimage.make_heatmask((frame_mask > -1).astype(np.float32), with_alpha=0.5)
            canvas = kwimage.overlay_alpha_layers([heatmask, canvas, heatmask])

            # print('canvas.shape = {!r}'.format(canvas.shape))
            canvas, info = kwimage.imresize(canvas, min_dim=min_dim, return_info=True)
            canvas = canvas.clip(0, 1)

            if with_text:
                # chan_name = channel_names[chanx]
                chan_name = str(chanx)
                canvas = kwimage.draw_text_on_image(
                    canvas, 'frame: {}\nchan: {}'.format(framex, chan_name),
                    org=(1, 1), valign='top',
                )

            chan_canvas_list.append(canvas)
        # frame_canvas = kwimage.stack_images(chan_canvas_list, axis=1, overlap=-1)
        canvas_frames.append(chan_canvas_list)

    stack1 = []
    for tup in zip(*canvas_frames):
        stack = kwimage.stack_images(tup, axis=1, overlap=-3)
        stack1.append(stack)
    stacked = kwimage.stack_images(stack1, axis=0, overlap=-6)
    return stacked


def draw_multispectral_batch(batch):
    decollated = decollate_batch(batch)
    canvas_list = []
    for item in decollated:
        part = draw_multispectral_item(item)
        canvas_list.append(part)

    canvas = kwimage.stack_images_grid(canvas_list, axis=1, overlap=-12)
    return canvas


def decollate_batch(batch):
    """
    Breakup a collated batch of BatchContainers back into ItemContainers

    TODO:
        - [ ] This should be a function that lives in netharn or wherever the
              container objects live.

    Example:
        >>> bsize = 5
        >>> batch_items = [
        >>>     {
        >>>         'im': ItemContainer.demo('img'),
        >>>         'label': ItemContainer.demo('labels'),
        >>>         'box': ItemContainer.demo('box'),
        >>>     }
        >>>     for _ in range(bsize)
        >>> ]
        >>> batch = container_collate(batch_items, num_devices=2)
        >>> decollated = decollate_batch(batch)
        >>> assert len(decollated) == len(batch_items)
        >>> assert (decollated[0]['im'].data == batch_items[0]['im'].data).all()
    """
    import ubelt as ub
    from kwcoco.util.util_json import IndexableWalker
    walker = IndexableWalker(batch)
    decollated_dict = ub.AutoDict()
    decollated_walker = IndexableWalker(decollated_dict)
    for path, batch_val in walker:
        if isinstance(batch_val, BatchContainer):
            for bx, item_val in enumerate(ub.flatten(batch_val.data)):
                decollated_walker[[bx] + path] = ItemContainer(item_val)
        elif isinstance(batch_val, torch.Tensor):
            for bx, item_val in enumerate(batch_val):
                decollated_walker[[bx] + path] = item_val
    decollated = list(decollated_dict.to_dict().values())
    return decollated


def __notes__():
    """
    Ignore:
        >>> #
        >>> for img in dset.imgs.values():
        >>>     chan = img.get('channels', None)
        >>>     print('img_chan = {!r}'.format(chan))
        >>>     for aux in img.get('auxiliary', []):
        >>>         chan = aux.get('channels', None)
        >>>         print('aux_chan = {!r}'.format(chan))
    """
