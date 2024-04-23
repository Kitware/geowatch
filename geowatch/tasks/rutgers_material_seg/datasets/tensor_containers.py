"""
import liberator

from netharn.data.batch_samplers import PatchedBatchSampler
from netharn.data.data_containers import ItemContainer
from netharn.data.data_containers import container_collate
from netharn.data.batch_samplers import PatchedRandomSampler
from netharn.data.batch_samplers import SubsetSampler
lib = liberator.Liberator()
lib.add_dynamic(SubsetSampler)
lib.add_dynamic(PatchedRandomSampler)
lib.add_dynamic(container_collate)
lib.add_dynamic(ItemContainer)
lib.add_dynamic(PatchedBatchSampler)
lib.add_dynamic(padded_colate)
lib.expand(['netharn'])
print(lib.current_sourcecode())
"""

import torch
import ubelt as ub
import torch.nn.functional as F
import numpy as np
import re
import torch.utils.data as torch_data

try:
    import collections.abc as container_abcs
    from six import string_types as string_classes
    from six import integer_types as int_classes
except Exception:
    from torch._six import container_abcs
    from torch._six import string_classes, int_classes


class PatchedBatchSampler(torch.utils.data.sampler.BatchSampler, ub.NiceRepr):
    """
    A modification of the standard torch BatchSampler that allows for
    specification of ``num_batches=auto``

    Example:
        >>> data_source = torch.arange(64)
        >>> sampler = PatchedRandomSampler(data_source, replacement=True, num_samples=None)
        >>> batch_size = 10
        >>> drop_last = False
        >>> num_batches = 'auto'
        >>> batch_sampler = PatchedBatchSampler(sampler, batch_size, drop_last, num_batches)
        >>> assert len(list(batch_sampler)) == 7 == len(batch_sampler)
        >>> batch_sampler = PatchedBatchSampler(sampler, batch_size, drop_last, 3)
        >>> assert len(list(batch_sampler)) == 3 == len(batch_sampler)
        >>> batch_sampler = PatchedBatchSampler(sampler, batch_size, drop_last, 1)
        >>> assert len(list(batch_sampler)) == 1 == len(batch_sampler)
    """
    def __init__(self, sampler, batch_size, drop_last, num_batches='auto'):
        super().__init__(sampler, batch_size, drop_last)
        self.num_batches = num_batches

    def __len__(self):
        if self.drop_last:
            max_num_batches = len(self.sampler) // self.batch_size  # type: ignore
        else:
            max_num_batches = (len(self.sampler) + self.batch_size - 1) // self.batch_size  # type: ignore
        if self.num_batches == 'auto':
            num_batches = max_num_batches
        else:
            num_batches = min(max_num_batches, self.num_batches)
        return num_batches

    def __iter__(self):
        num_batches = len(self)
        for bx, batch in zip(range(num_batches), super().__iter__()):
            yield batch


class ItemContainer(ub.NiceRepr):
    """
    A container for uncollated items that defines a specific collation
    strategy. Based on mmdetections ItemContainer.
    """

    def __init__(
        self,
        data,
        stack=False,
        padding_value=-1,
        cpu_only=False,
        pad_dims=2
    ):
        self._data = data
        assert pad_dims in [None, 1, 2, 3]
        self.meta = {
            'stack': stack,
            'padding_value': padding_value,
            'cpu_only': cpu_only,
            'pad_dims': pad_dims,
        }

    @property
    def nestshape(self):
        return nestshape(self.data)

    def __nice__(self):
        try:
            shape_repr = ub.repr2(self.nestshape, nl=-2)
            return 'nestshape(data)={}'.format(shape_repr)
        except Exception:
            return object.__repr__(self)
            # return super().__repr__()

    @classmethod
    def demo(cls, key='img', rng=None, **kwargs):
        """
        Create data for tests

        Example:
            >>> print(ItemContainer.demo('img'))
            >>> print(ItemContainer.demo('labels'))
            >>> print(ItemContainer.demo('box'))

        """
        import kwarray
        rng = kwarray.ensure_rng(rng)
        if key == 'img':
            shape = kwargs.get('shape', (3, 512, 512))
            data = rng.rand(*shape).astype(np.float32)
            data = torch.from_numpy(data)
            self = cls(data, stack=True)
        elif key == 'labels':
            n = rng.randint(0, 10)
            data = rng.randint(0, 10, n)
            data = torch.from_numpy(data)
            self = cls(data, stack=False)
        elif key == 'box':
            n = rng.randint(0, 10)
            data = rng.rand(n, 4)
            data = torch.from_numpy(data)
            self = cls(data, stack=False)
        else:
            raise KeyError(key)
        return self

    def __getitem__(self, index):
        assert self.stack, 'can only index into stackable items'
        cls = self.__class__
        return cls(self.data[index], **self.meta)

    @property
    def data(self):
        return self._data

    @property
    def datatype(self):
        if isinstance(self.data, torch.Tensor):
            return self.data.type()
        else:
            return type(self.data)

    @property
    def cpu_only(self):
        return self.meta['cpu_only']

    @property
    def stack(self):
        return self.meta['stack']

    @property
    def padding_value(self):
        return self.meta['padding_value']

    @property
    def pad_dims(self):
        return self.meta['pad_dims']

    def size(self, *args, **kwargs):
        return self.data.size(*args, **kwargs)

    @property
    def shape(self):
        return self.data.shape

    def dim(self):
        return self.data.dim()

    @classmethod
    def _collate(cls, inbatch, num_devices=None):
        """
        Collates a sequence of DataContainers

        Args:
            inbatch (Sequence[ItemContainer]): datacontainers with the same
                parameters.

            num_devices (int): number of groups, if None, then uses one group.

        Example:
            >>> print('Collate Image ItemContainer')
            >>> inbatch = [ItemContainer.demo('img') for _ in range(5)]
            >>> print('inbatch = {}'.format(ub.repr2(inbatch)))
            >>> result = ItemContainer._collate(inbatch, num_devices=2)
            >>> print('result1 = {}'.format(ub.repr2(result, nl=1)))
            >>> result = ItemContainer._collate(inbatch, num_devices=1)
            >>> print('result2 = {}'.format(ub.repr2(result, nl=1)))
            >>> result = ItemContainer._collate(inbatch, num_devices=None)
            >>> print('resultN = {}'.format(ub.repr2(result, nl=1)))

            >>> print('Collate Label ItemContainer')
            >>> inbatch = [ItemContainer.demo('labels') for _ in range(5)]
            >>> print('inbatch = {}'.format(ub.repr2(inbatch, nl=1)))
            >>> result = ItemContainer._collate(inbatch, 1)
            >>> print('result1 = {}'.format(ub.repr2(result, nl=1)))
            >>> result = ItemContainer._collate(inbatch, 2)
            >>> print('result2 = {}'.format(ub.repr2(result, nl=1)))
            >>> result = ItemContainer._collate(inbatch, None)
            >>> print('resultN = {}'.format(ub.repr2(result, nl=1)))
        """
        item0 = inbatch[0]
        bsize = len(inbatch)
        if num_devices is None:
            num_devices = 1

        samples_per_device = int(np.ceil(bsize / num_devices))

        # assert bsize % samples_per_device == 0
        stacked = []
        if item0.cpu_only:
            # chunking logic
            stacked = []
            for i in range(0, bsize, samples_per_device):
                stacked.append(
                    [sample.data for sample in inbatch[i:i + samples_per_device]])

        elif item0.stack:
            for i in range(0, bsize, samples_per_device):
                item = inbatch[i]
                pad_dims_ = item.pad_dims
                assert isinstance(item.data, torch.Tensor)

                if pad_dims_ is not None:
                    # Note: can probably reimplement this using padded collate
                    # logic
                    ndim = item.dim()
                    assert ndim > pad_dims_
                    max_shape = [0 for _ in range(pad_dims_)]
                    for dim in range(1, pad_dims_ + 1):
                        max_shape[dim - 1] = item.shape[-dim]
                    for sample in inbatch[i:i + samples_per_device]:
                        for dim in range(0, ndim - pad_dims_):
                            assert item.shape[dim] == sample.shape[dim]
                        for dim in range(1, pad_dims_ + 1):
                            max_shape[dim - 1] = max(max_shape[dim - 1], sample.shape[-dim])
                    padded_samples = []
                    for sample in inbatch[i:i + samples_per_device]:
                        pad = [0 for _ in range(pad_dims_ * 2)]
                        for dim in range(1, pad_dims_ + 1):
                            pad[2 * dim - 1] = max_shape[dim - 1] - sample.shape[-dim]
                        padded_samples.append(
                            F.pad(sample.data, pad, value=sample.padding_value))
                    stacked.append(default_collate(padded_samples))

                elif pad_dims_ is None:
                    stacked.append(
                        default_collate([
                            sample.data
                            for sample in inbatch[i:i + samples_per_device]
                        ]))
                else:
                    raise ValueError(
                        'pad_dims should be either None or integers (1-3)')

        else:
            for i in range(0, bsize, samples_per_device):
                stacked.append(
                    [sample.data for sample in inbatch[i:i + samples_per_device]])
        result = BatchContainer(stacked, **item0.meta)
        return result


def nestshape(data):
    """
    Examine nested shape of the data

    Example:
        >>> data = [np.arange(10), np.arange(13)]
        >>> nestshape(data)
        [(10,), (13,)]

    Ignore:
        >>> # xdoctest: +REQUIRES(module:mmdet)
        >>> from mmdet.core.mask.structures import *  # NOQA
        >>> masks = [
        >>>     [ np.array([0, 0, 10, 0, 10, 10., 0, 10, 0, 0]) ],
        >>>     [ np.array([0, 0, 10, 0, 10, 10., 0, 10, 5., 5., 0, 0]) ]
        >>> ]
        >>> height, width = 16, 16
        >>> polys = PolygonMasks(masks, height, width)
        >>> nestshape(polys)

        >>> dc = BatchContainer([polys], stack=False)
        >>> print('dc = {}'.format(ub.repr2(dc, nl=1)))

        >>> num_masks, H, W = 3, 32, 32
        >>> rng = np.random.RandomState(0)
        >>> masks = (rng.rand(num_masks, H, W) > 0.1).astype(int)
        >>> bitmasks = BitmapMasks(masks, height=H, width=W)
        >>> nestshape(bitmasks)

        >>> dc = BatchContainer([bitmasks], stack=False)
        >>> print('dc = {}'.format(ub.repr2(dc, nl=1)))

    """

    def _recurse(d):
        if isinstance(d, dict):
            return ub.odict(sorted([(k, _recurse(v)) for k, v in d.items()]))

        clsname = type(d).__name__
        if 'Container' in clsname:
            meta = ub.odict(sorted([
                ('stack', d.stack),
                # ('padding_value', d.padding_value),
                # ('pad_dims', d.pad_dims),
                # ('datatype', d.datatype),
                ('cpu_only', d.cpu_only),
            ]))
            meta = ub.repr2(meta, nl=0)
            return {type(d).__name__ + meta: _recurse(d.data)}
        elif isinstance(d, list):
            return [_recurse(v) for v in d]
        elif isinstance(d, tuple):
            return tuple([_recurse(v) for v in d])
        elif isinstance(d, torch.Tensor):
            return d.shape
        elif isinstance(d, np.ndarray):
            return d.shape
        elif isinstance(d, (str, bytes)):
            return d
        elif isinstance(d, (int, float)):
            return d
        elif isinstance(d, slice):
            return d
        elif 'PolygonMasks' == clsname:
            # hack for mmdet
            return repr(d)
        elif 'BitmapMasks' == clsname:
            # hack for mmdet
            return repr(d)
        elif hasattr(d, 'shape'):
            return d.shape
        elif hasattr(d, 'items'):
            # hack for dict-like objects
            return ub.odict(sorted([(k, _recurse(v)) for k, v in d.items()]))
        elif d is None:
            return None
        else:
            raise TypeError(type(d))

    # globals()['_recurse'] = _recurse
    d = _recurse(data)
    return d


class BatchContainer(ub.NiceRepr):
    """
    A container for a set of items in a batch. Usually this is for network
    outputs or a set of items that have already been collated.

    Attributes:
        data (List[Any]): Unlike ItemContainer, data is always a list where
            len(data) is the number of devices this batch will run on.  Each
            item in the list may be either a pre-batched Tensor (in the case
            where the each item in the batch has the same shape) or a list of
            individual item Tensors (in the case where different batch items
            may have different shapes).
    """
    def __init__(self, data, stack=False, padding_value=-1, cpu_only=False,
                 pad_dims=2):
        self.data = data  # type: list
        self.meta = {
            'stack': stack,
            'padding_value': padding_value,
            'cpu_only': cpu_only,
            'pad_dims': pad_dims,
        }

    @property
    def nestshape(self):
        return nestshape(self.data)

    def numel(self):
        """
        The number of scalar elements held by this container
        """
        shapes = self.nestshape
        total = sum([np.prod(s) for s in shapes])
        return total

    @property
    def packshape(self):
        """
        The shape of this data if it was packed
        """
        # shape = np.maximum.reduce(self.nestshape)
        # return shape
        dim = 0
        if self.stack:
            # Should be a straight forward concatenation
            shapes = [d.shape for d in self.data]
            max_shape = np.maximum.reduce(shapes)  # should all be the same here
            stacked_dim = sum([s[dim] for s in shapes])
            max_shape[dim] = stacked_dim
            pack_shape = tuple(max_shape.tolist())
            return pack_shape
        else:
            shapes = nestshape(self.data)
            max_shape = np.maximum.reduce(shapes)
            stacked_dim = sum([s[dim] for s in shapes])
            max_shape[dim] = stacked_dim
            pack_shape = tuple(max_shape.tolist())
            return pack_shape

    def __nice__(self):
        try:
            shape_repr = ub.repr2(self.nestshape, nl=-2)
            return 'nestshape(data)={}'.format(shape_repr)
        except Exception:
            return object.__repr__(self)

    def __getitem__(self, index):
        cls = self.__class__
        return cls([d[index] for d in self.data], **self.meta)

    @property
    def cpu_only(self):
        return self.meta['cpu_only']

    @property
    def stack(self):
        return self.meta['stack']

    @property
    def padding_value(self):
        return self.meta['padding_value']

    @property
    def pad_dims(self):
        return self.meta['pad_dims']

    @classmethod
    def cat(cls, items, dim=0):
        """
        Concatenate data in multiple BatchContainers

        Example:
            >>> d1 = BatchContainer([torch.rand(3, 3, 1, 1), torch.rand(2, 3, 1, 1)])
            >>> d2 = BatchContainer([torch.rand(3, 1, 1, 1), torch.rand(2, 1, 1, 1)])
            >>> items = [d1, d2]
            >>> self = BatchContainer.cat(items, dim=1)
        """
        newdata = []
        num_devices = len(items[0].data)
        for device_idx in range(num_devices):
            parts = [item.data[device_idx] for item in items]
            newpart = torch.cat(parts, dim=dim)
            newdata.append(newpart)
        self = cls(newdata, **items[0].meta)
        return self

    @classmethod
    def demo(cls, key='img', n=5, num_devices=1):
        inbatch = [ItemContainer.demo(key) for _ in range(n)]
        self = ItemContainer._collate(inbatch, num_devices=num_devices)
        return self

    def pack(self):
        """
        Pack all of the data in this container into a single tensor.

        Returns:
            Tensor: packed data, padded with ``self.padding_value`` if
            ``self.stack`` is False.

        Example:
            >>> self = BatchContainer.demo('img')
            >>> print(self.pack())
            >>> self = BatchContainer.demo('box')
            >>> print(self.pack())
            >>> self = BatchContainer.demo('labels')
            >>> print(self.pack())
        """
        if self.stack:
            # Should be a straight forward concatenation
            packed = torch.cat(self.data, dim=0)
        else:
            # Need to account for padding values
            inbatch = list(ub.flatten(self.data))
            packed = padded_collate(inbatch, fill_value=self.padding_value)
        return packed

    def to(self, device):
        """ inplace move data onto a device """
        walker = ub.IndexableWalker(self.data)
        for path, val in walker:
            if torch.is_tensor(val):
                walker[path] = val.to(device)
        return self


default_collate = torch_data.dataloader.default_collate


def container_collate(inbatch, num_devices=None):
    """Puts each data field into a tensor/DataContainer with outer dimension
    batch size.

    Extend default_collate to add support for
    :type:`~mmcv.parallel.DataContainer`. There are 3 cases.

    1. cpu_only = True, e.g., meta data
    2. cpu_only = False, stack = True, e.g., images tensors
    3. cpu_only = False, stack = False, e.g., gt bboxes

    Example:
        >>> item1 = {
        >>>     'im': torch.rand(3, 512, 512),
        >>>     'label': torch.rand(3),
        >>> }
        >>> item2 = {
        >>>     'im': torch.rand(3, 512, 512),
        >>>     'label': torch.rand(3),
        >>> }
        >>> item3 = {
        >>>     'im': torch.rand(3, 512, 512),
        >>>     'label': torch.rand(3),
        >>> }
        >>> batch = batch_items = [item1, item2, item3]
        >>> raw_batch = container_collate(batch_items)
        >>> print('batch_items = {}'.format(ub.repr2(batch_items, nl=2)))
        >>> print('raw_batch = {}'.format(ub.repr2(raw_batch, nl=2)))

        >>> batch = batch_items = [
        >>>     {'im': ItemContainer.demo('img'), 'label': ItemContainer.demo('labels')},
        >>>     {'im': ItemContainer.demo('img'), 'label': ItemContainer.demo('labels')},
        >>>     {'im': ItemContainer.demo('img'), 'label': ItemContainer.demo('labels')},
        >>> ]
        >>> raw_batch = container_collate(batch, num_devices=2)
        >>> print('batch_items = {}'.format(ub.repr2(batch_items, nl=2)))
        >>> print('raw_batch = {}'.format(ub.repr2(raw_batch, nl=2)))

        >>> raw_batch = container_collate(batch, num_devices=6)
        >>> raw_batch = container_collate(batch, num_devices=3)
        >>> raw_batch = container_collate(batch, num_devices=4)
        >>> raw_batch = container_collate(batch, num_devices=1)
        >>> print('batch = {}'.format(ub.repr2(batch, nl=1)))
    """

    if not isinstance(inbatch, container_abcs.Sequence):
        raise TypeError("{} is not supported.".format(inbatch.dtype))
    item0 = inbatch[0]
    if isinstance(item0, ItemContainer):
        return item0.__class__._collate(inbatch, num_devices=num_devices)
    elif isinstance(item0, container_abcs.Sequence):
        transposed = zip(*inbatch)
        return [container_collate(samples,
                                  num_devices=num_devices)
                for samples in transposed]
    elif isinstance(item0, container_abcs.Mapping):
        return {
            key: container_collate([d[key] for d in inbatch],
                                   num_devices=num_devices)
            for key in item0
        }
    else:
        return default_collate(inbatch)
        # return _collate_else(inbatch, container_collate)


class PatchedRandomSampler(torch.utils.data.sampler.Sampler, ub.NiceRepr):
    r"""
    A modification of the standard torch Sampler that allows specification of
    ``num_samples``.

    See: https://github.com/pytorch/pytorch/pull/39214

    Example:
        >>> data_source = torch.arange(10)
        >>> # with replacement
        >>> sampler = PatchedRandomSampler(data_source, replacement=True, num_samples=None)
        >>> assert len(sampler) == 10 == len(list(sampler))
        >>> sampler = PatchedRandomSampler(data_source, replacement=True, num_samples=1)
        >>> assert len(sampler) == 1 == len(list(sampler))
        >>> sampler = PatchedRandomSampler(data_source, replacement=True, num_samples=5)
        >>> assert len(sampler) == 5 == len(list(sampler))
        >>> sampler = PatchedRandomSampler(data_source, replacement=True, num_samples=10)
        >>> assert len(sampler) == 10 == len(list(sampler))
        >>> sampler = PatchedRandomSampler(data_source, replacement=True, num_samples=15)
        >>> assert len(sampler) == 15 == len(list(sampler))
        >>> # without replacement
        >>> sampler = PatchedRandomSampler(data_source, replacement=False, num_samples=None)
        >>> assert len(sampler) == 10 == len(list(sampler))
        >>> sampler = PatchedRandomSampler(data_source, replacement=False, num_samples=1)
        >>> assert len(sampler) == 1 == len(list(sampler))
        >>> sampler = PatchedRandomSampler(data_source, replacement=False, num_samples=5)
        >>> assert len(sampler) == 5 == len(list(sampler))
        >>> sampler = PatchedRandomSampler(data_source, replacement=False, num_samples=10)
        >>> assert len(sampler) == 10 == len(list(sampler))
        >>> sampler = PatchedRandomSampler(data_source, replacement=False, num_samples=15)
        >>> assert len(sampler) == 10 == len(list(sampler))
    """

    def __init__(self, data_source, replacement=False, num_samples=None):
        self.data_source = data_source
        self.replacement = replacement
        self._num_samples = num_samples

        if not isinstance(self.replacement, bool):
            raise ValueError("replacement should be a boolean value, but got "
                             "replacement={}".format(self.replacement))

        if not isinstance(self.num_samples, int) or self.num_samples <= 0:
            raise ValueError("num_samples should be a positive integer "
                             "value, but got num_samples={}".format(self.num_samples))

    @property
    def num_samples(self):
        # dataset size might change at runtime
        if self._num_samples is None:
            num = len(self.data_source)
        else:
            if self.replacement:
                num = self._num_samples
            else:
                num = min(self._num_samples, len(self.data_source))
        return num

    def __iter__(self):
        n = len(self.data_source)
        if self.replacement:
            return iter(torch.randint(high=n, size=(self.num_samples,), dtype=torch.int64).tolist())
        return iter(torch.randperm(n).tolist()[:self.num_samples])

    def __len__(self):
        return self.num_samples


class SubsetSampler(torch.utils.data.sampler.Sampler, ub.NiceRepr):
    """
    Generates sample indices based on a specified order / subset

    Example:
        >>> indices = list(range(10))
        >>> assert indices == list(SubsetSampler(indices))
    """

    def __init__(self, indices):
        self.indices = indices

    def __iter__(self):
        for idx in self.indices:
            yield idx

    def __len__(self):
        return len(self.indices)


def padded_collate(inbatch, fill_value=-1):
    """
    Used for detection datasets with boxes.

    Example:
        >>> import torch
        >>> rng = np.random.RandomState(0)
        >>> inbatch = []
        >>> bsize = 7
        >>> for i in range(bsize):
        >>>     # add an image and some dummy bboxes to the batch
        >>>     img = torch.rand(3, 8, 8)  # dummy 8x8 image
        >>>     n = 11 if i == 3 else rng.randint(0, 11)
        >>>     boxes = torch.rand(n, 4)
        >>>     item = (img, boxes)
        >>>     inbatch.append(item)
        >>> out_batch = padded_collate(inbatch)
        >>> assert len(out_batch) == 2
        >>> assert list(out_batch[0].shape) == [bsize, 3, 8, 8]
        >>> assert list(out_batch[1].shape) == [bsize, 11, 4]

    Example:
        >>> import torch
        >>> rng = np.random.RandomState(0)
        >>> inbatch = []
        >>> bsize = 4
        >>> for _ in range(bsize):
        >>>     # add an image and some dummy bboxes to the batch
        >>>     img = torch.rand(3, 8, 8)  # dummy 8x8 image
        >>>     #boxes = torch.empty(0, 4)
        >>>     boxes = torch.FloatTensor()
        >>>     item = (img, [boxes])
        >>>     inbatch.append(item)
        >>> out_batch = padded_collate(inbatch)
        >>> assert len(out_batch) == 2
        >>> assert list(out_batch[0].shape) == [bsize, 3, 8, 8]
        >>> #assert list(out_batch[1][0].shape) == [bsize, 0, 4]
        >>> assert list(out_batch[1][0].shape) in [[0], []]  # torch .3 a .4

    Example:
        >>> inbatch = [torch.rand(4, 4), torch.rand(8, 4),
        >>>            torch.rand(0, 4), torch.rand(3, 4),
        >>>            torch.rand(0, 4), torch.rand(1, 4)]
        >>> out_batch = padded_collate(inbatch)
        >>> assert list(out_batch.shape) == [6, 8, 4]
    """
    try:
        if torch.is_tensor(inbatch[0]):
            num_items = [len(item) for item in inbatch]
            if ub.allsame(num_items):
                if len(num_items) == 0:
                    batch = torch.FloatTensor()
                elif num_items[0] == 0:
                    batch = torch.FloatTensor()
                else:
                    batch = default_collate(inbatch)
            else:
                max_size = max(num_items)
                real_tail_shape = None
                for item in inbatch:
                    if item.numel():
                        tail_shape = item.shape[1:]
                        if real_tail_shape is not None:
                            assert real_tail_shape == tail_shape
                        real_tail_shape = tail_shape

                padded_inbatch = []
                for item in inbatch:
                    n_extra = max_size - len(item)
                    if n_extra > 0:
                        shape = (n_extra,) + tuple(real_tail_shape)
                        if torch.__version__.startswith('0.3'):
                            extra = torch.Tensor(np.full(shape, fill_value=fill_value))
                        else:
                            extra = torch.full(shape, fill_value=fill_value,
                                               dtype=item.dtype)
                        padded_item = torch.cat([item, extra], dim=0)
                        padded_inbatch.append(padded_item)
                    else:
                        padded_inbatch.append(item)
                batch = inbatch
                batch = default_collate(padded_inbatch)
        else:
            batch = _collate_else(inbatch, padded_collate)
    except Exception as ex:
        if not isinstance(ex, CollateException):
            try:
                _debug_inbatch_shapes(inbatch)
            except Exception:
                pass
            raise CollateException(
                'Failed to collate inbatch={}. Reason: {!r}'.format(inbatch, ex))
        else:
            raise
    return batch


default_collate = torch_data.dataloader.default_collate


# numpy_type_map = torch_data.dataloader.numpy_type_map  # moved in torch 1.1.0
numpy_type_map = {
    'float64': torch.DoubleTensor,
    'float32': torch.FloatTensor,
    'float16': torch.HalfTensor,
    'int64': torch.LongTensor,
    'int32': torch.IntTensor,
    'int16': torch.ShortTensor,
    'int8': torch.CharTensor,
    'uint8': torch.ByteTensor,
}


class CollateException(Exception):
    pass


_DEBUG = False


def _collate_else(batch, collate_func):
    """
    Handles recursion in the else case for these special collate functions

    This is duplicates all non-tensor cases from `torch_data.dataloader.default_collate`
    This also contains support for collating slices.
    """
    error_msg = "batch must contain tensors, numbers, dicts or lists; found {}"
    elem_type = type(batch[0])
    if elem_type.__module__ == 'numpy' and elem_type.__name__ != 'str_' \
            and elem_type.__name__ != 'string_':
        elem = batch[0]
        if elem_type.__name__ == 'ndarray':
            # array of string classes and object
            if re.search('[SaUO]', elem.dtype.str) is not None:
                raise TypeError(error_msg.format(elem.dtype))

            return torch.stack([torch.from_numpy(b) for b in batch], 0)
        if elem.shape == ():  # scalars
            py_type = float if elem.dtype.name.startswith('float') else int
            return numpy_type_map[elem.dtype.name](list(map(py_type, batch)))
    elif isinstance(batch[0], slice):
        batch = default_collate([{
            'start': sl.start,
            'stop': sl.stop,
            'step': 1 if sl.step is None else sl.step
        } for sl in batch])
        return batch
    elif isinstance(batch[0], int_classes):
        return torch.LongTensor(batch)
    elif isinstance(batch[0], float):
        return torch.DoubleTensor(batch)
    elif isinstance(batch[0], string_classes):
        return batch
    elif isinstance(batch[0], container_abcs.Mapping):
        # Hack the mapping collation implementation to print error info
        if _DEBUG:
            collated = {}
            try:
                for key in batch[0]:
                    collated[key] = collate_func([d[key] for d in batch])
            except Exception:
                print('\n!!Error collating key = {!r}\n'.format(key))
                raise
            return collated
        else:
            return {key: collate_func([d[key] for d in batch]) for key in batch[0]}
    elif isinstance(batch[0], tuple) and hasattr(batch[0], '_fields'):  # namedtuple
        return type(batch[0])(*(default_collate(samples) for samples in zip(*batch)))
    elif isinstance(batch[0], container_abcs.Sequence):
        transposed = zip(*batch)
        return [collate_func(samples) for samples in transposed]
    else:
        raise TypeError((error_msg.format(type(batch[0]))))


def _debug_inbatch_shapes(inbatch):
    import ubelt as ub
    print('len(inbatch) = {}'.format(len(inbatch)))
    extensions = ub.util_format.FormatterExtensions()
    #
    @extensions.register((torch.Tensor, np.ndarray))
    def format_shape(data, **kwargs):
        return ub.repr2(dict(type=str(type(data)), shape=data.shape), nl=1, sv=1)
    print('inbatch = ' + ub.repr2(inbatch, extensions=extensions, nl=True))
