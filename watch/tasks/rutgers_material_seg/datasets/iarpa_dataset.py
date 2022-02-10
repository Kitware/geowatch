import kwarray
import kwimage
import numpy as np
import torch
# import matplotlib.pyplot as plt
# from kwcoco.channel_spec import ChannelSpec  # NOQA
from functools import partial
from netharn.data.batch_samplers import PatchedBatchSampler
from netharn.data.data_containers import ItemContainer
from netharn.data.data_containers import BatchContainer
from netharn.data.data_containers import container_collate
from netharn.data.batch_samplers import PatchedRandomSampler
from netharn.data.batch_samplers import SubsetSampler


class SequenceDataset(torch.utils.data.Dataset):
    def __init__(self, sampler, window_dims, input_dims=None, channels=None,
                 rng=None, window_overlap=0.0):
        super().__init__()
        if input_dims is None:
            input_dims = window_dims[-2:]
        # print(sampler)
        self.rng = kwarray.ensure_rng(rng)
        self.sampler = sampler
        self.window_dims = window_dims
        self.input_dims = input_dims
        self.classes = self.sampler.classes
        self.channels = channels
        # Build a simple space-time-grid
        sample_grid_spec = {
            'task': 'video_detection',
            'window_dims': window_dims,
            'window_overlap': window_overlap,
        }
        # print(sample_grid_spec)
        self.sample_grid = sampler.new_sample_grid(**sample_grid_spec)
        # print(len(self.sample_grid['positives']))

    def __len__(self):
        return len(self.sample_grid['positives'])
        # return len(self.sample_grid['negatives'])

    def __getitem__(self, index):

        tr = self.sample_grid['positives'][index]
        # tr = self.sample_grid['negatives'][index]
        if self.channels:
            tr['channels'] = self.channels

        sampler = self.sampler

        # import pdb
        # pdb.set_trace()
        # print(f"frame min: {frame.min()}, frame max: {frame.max()}")
        sample = sampler.load_sample(tr, with_annots="segmentation")
        # print(sample.keys())
        # print(sample['annots'].keys())
        # print(sample['annots']['rel_ssegs'])
        # print(sample['annots']['frame_dets'])
        # rel_ssegs = sample['annots']['rel_ssegs']
        # print(rel_ssegs.data)
        # seg = kwimage.Segmentation.coerce(rel_ssegs.data)
        # plt.imshow(rel_ssegs)
        # plt.show()
        # Access the sampled image and annotation data
        raw_frame_list = sample['im']
        raw_det_list = sample['annots']['frame_dets']
        # print(sample)
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
            # print(info)
            dets = dets.translate(info['offset'])
            # print(frame.shape[0:2])
            frame_mask = np.full(frame.shape[0:2], dtype=np.int32, fill_value=-1)
            ann_polys = dets.data['segmentations'].to_polygon_list()
            # print(ann_polys)
            ann_aids = dets.data['aids']
            ann_cids = dets.data['cids']
            # print(ann_cids)
            for poly, aid, cid in zip(ann_polys, ann_aids, ann_cids):
                cidx = self.classes.id_to_idx[cid]
                poly.fill(frame_mask, value=cidx)

            # ensure channel dim is not squeezed
            frame = kwarray.atleast_nd(frame, 3)

            # fig = plt.figure()
            # ax1 = fig.add_subplot(1,2,1)
            # ax2 = fig.add_subplot(1,2,2)
            # ax1.imshow(frame[:,:,:3])
            # ax2.imshow(frame_mask)
            # plt.show()

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
                    pin_memory=True, drop_last=True, multiscale=False,
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


def decollate_batch(batch):
    """
    Breakup a collated batch of BatchContainers back into ItemContainers
    """
    import ubelt as ub
    walker = ub.IndexableWalker(batch)
    decollated_dict = ub.AutoDict()
    decollated_walker = ub.IndexableWalker(decollated_dict)
    for path, batch_val in walker:
        if isinstance(batch_val, BatchContainer):
            for bx, item_val in enumerate(ub.flatten(batch_val.data)):
                decollated_walker[[bx] + path] = ItemContainer(item_val)
    decollated = list(decollated_dict.to_dict().values())
    return decollated


# def __notes__():
#     """
#         >>> #
#         >>> for img in dset.imgs.values():
#         >>>     chan = img.get('channels', None)
#         >>>     print('img_chan = {!r}'.format(chan))
#         >>>     for aux in img.get('auxiliary', []):
#         >>>         chan = aux.get('channels', None)
#         >>>         print('aux_chan = {!r}'.format(chan))
#     """
