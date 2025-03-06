"""
Module to define the :class:`BatchItem` wrapper class.

This is currently a work in progress, and the goal is to abstract the format of
the items produced by the datalaoder to make them ammenable for use with both
our heterogeneous networks as well as more standard networks that require data
be more regular.


This has a lot in common with prior work in:

    ~/code/netharn/netharn/data/data_containers.py
"""
import kwarray
import ubelt as ub
import numpy as np


class BatchItem(dict):
    """
    Ideally a batch item is simply an unstructured dictionary.
    This is the base class for more specific implementations, which are going
    to all be dictionaries, but the class will expose convinience methods.
    """

    def __repr__(self):
        """
        Returns:
            str
        """
        nice = self.__nice__()
        classname = self.__class__.__name__
        return '<{0}({1}) at {2}>'.format(classname, nice, hex(id(self)))

    def __str__(self):
        """
        Returns:
            str
        """
        classname = self.__class__.__name__
        nice = self.__nice__()
        return '<{0}({1})>'.format(classname, nice)

    def __nice__(self):
        return ''

    def asdict(self):
        return dict(self)

    def draw(self, **kwargs):
        raise NotImplementedError

    @classmethod
    def demo(cls, **kwargs):
        raise NotImplementedError

    @classmethod
    def summarize(cls, **kwargs):
        raise NotImplementedError


class HeterogeneousBatchItem(BatchItem):
    """
    A BatchItem is a container to help organize the output of the
    KWCocoVideoDataset. For backwards compatibility it retains the original
    dictionary interface.

    Example:
        >>> from geowatch.tasks.fusion.datamodules.network_io import *  # NOQA
        >>> self = HeterogeneousBatchItem.demo()
        >>> print(self)
        >>> print(ub.urepr(self.summarize(), nl=2))
        >>> canvas = self.draw()
        >>> # xdoctest: +REQUIRES(--show)
        >>> import kwplot
        >>> kwplot.autompl()
        >>> kwplot.imshow(canvas, fnum=1, pnum=(1, 1, 1))
        >>> kwplot.show_if_requested()
    """

    def __nice__(self):
        return f'num_frames={self.num_frames}, sensorchans={self.sensorchan_histogram}'

    @property
    def num_frames(self):
        return len(self['frames'])

    @property
    def sensorchan_histogram(self):
        histogram = ub.dict_hist(
            frame.get('sensor', '*') + ':' + mode_key
            for frame in self['frames']
            for mode_key in frame['modes'].keys()
        )
        return histogram

    @classmethod
    def demo(cls):
        from geowatch.tasks.fusion.datamodules import kwcoco_dataset
        import geowatch
        coco_dset = geowatch.coerce_kwcoco('geowatch-msi', num_frames=10)
        dataset = kwcoco_dataset.KWCocoVideoDataset(
            coco_dset, time_dims=4, window_dims=(300, 300),
            channels='auto')
        dataset.disable_augmenter = True
        index = dataset.sample_grid['targets'][dataset.sample_grid['positives_indexes'][0]]
        item = dataset[index]
        self = cls(**item)
        return self

    def draw(self, item_output=None, combinable_extra=None, max_channels=5,
             max_dim=224, norm_over_time='auto', overlay_on_image=False,
             draw_weights=True, rescale='auto', classes=None,
             predictable_classes=None, show_summary_text=True,
             requested_tasks=None, legend=True, **kwargs):
        """
        Visualize this batch item.
        Corresponds to the dataset :func:`IntrospectMixin.draw_item` method.

        Not finished. The dataset draw_item class has context not currently
        available like predictable classes, that needs to be represented in the
        item itself.
        """
        import kwimage
        item = self

        from geowatch import heuristics
        default_combinable_channels = [
            ub.oset(['red', 'green', 'blue']),
            ub.oset(['Dred', 'Dgreen', 'Dblue']),
            ub.oset(['r', 'g', 'b']),
            ub.oset(['impervious', 'forest', 'water']),
            ub.oset(['baren', 'field', 'water']),
            ub.oset(['landcover_hidden.0', 'landcover_hidden.1', 'landcover_hidden.2']),
            ub.oset(['sam.0', 'sam.1', 'sam.2']),
            ub.oset(['sam.3', 'sam.4', 'sam.5']),
        ] + heuristics.HUERISTIC_COMBINABLE_CHANNELS

        if requested_tasks is None:
            requested_tasks = {'class': True, 'saliency': True, 'change': True, 'outputs': False, 'boxes': False}
        # self.default_combinable_channels
        # if rescale == 'auto':
        #     rescale = self.config['input_space_scale'] != 'native'
        # if norm_over_time == 'auto':
        #     norm_over_time = self.config['normalize_peritem'] is not None

        # Hack to force the categories to draw right for SMART
        # FIXME: Use the correct class colors in visualization.
        # FIXME: requested_tasks from user input is not respected
        requested_tasks = item['requested_tasks']
        predictable_classes = item['predictable_classes']
        if predictable_classes is not None:
            heuristics.ensure_heuristic_category_tree_colors(predictable_classes, force=True)

        from geowatch.tasks.fusion.datamodules.batch_visualization import BatchVisualizationBuilder
        builder = BatchVisualizationBuilder(
            item=item, item_output=item_output,
            default_combinable_channels=default_combinable_channels,
            norm_over_time=norm_over_time, max_dim=max_dim,
            max_channels=max_channels, overlay_on_image=overlay_on_image,
            draw_weights=draw_weights, combinable_extra=combinable_extra,
            classes=predictable_classes, requested_tasks=requested_tasks,
            rescale=rescale, **kwargs)
        canvas = builder.build()

        if show_summary_text:
            try:
                summary = item.summarize()
            except Exception as ex:
                summary = {'summary_error': str(ex)}
            summary = ub.udict(summary) - {'frame_summaries'}
            summary_text = ub.urepr(summary, nobr=1, precision=2, nl=-1)
            header = kwimage.draw_text_on_image(None, text=summary_text, halign='left', color='kitware_blue')
            canvas = kwimage.stack_images([canvas, header])

        if legend:
            from geowatch.tasks.fusion.datamodules.batch_visualization import _memo_legend
            label_to_color = {
                node: data['color']
                for node, data in self['predictable_classes'].graph.nodes.items()}
            legend_img = _memo_legend(label_to_color)
            legend_img = kwimage.imresize(legend_img, scale=4.0)
            canvas = kwimage.stack_images([canvas, legend_img], axis=1)

        return canvas

    def summarize(self, coco_dset=None, stats=False):
        """
        Return debugging stats about the item

        Args:
            coco_dset (CocoDataset):
                The coco dataset used to generate the item.
                If specified, allows the summary to lookup extra information

            stats (bool): if True, include statistics on input datas.

        Returns:
            dict : a summary of the item

        Example:
            >>> from geowatch.tasks.fusion.datamodules.network_io import *  # NOQA
            >>> self = HeterogeneousBatchItem.demo()
            >>> item_summary = self.summarize(stats=0)
            >>> print(f'item_summary = {ub.urepr(item_summary, nl=-2)}')
        """
        item = self
        item_summary = {}
        item_summary['frame_summaries'] = []
        timestamps = []
        for frame in item['frames']:
            frame_summary = {}
            for mode_key, im_mode in frame['modes'].items():
                domain_key = frame['sensor'] + ':' + mode_key
                frame_summary[domain_key] = {}
                if stats:
                    frame_summary[domain_key]['stats'] = kwarray.stats_dict(
                        im_mode, nan=True)
                frame_summary[domain_key]['shape'] = im_mode.shape
            label_keys = [
                'class_idxs', 'class_ohe', 'saliency', 'change'
                'class_weights', 'saliency_weights', 'change_weights',
                'output_weights', 'box_ltrb',
                # 'box_weights', 'box_tids', 'box_cidxs',
            ]
            for key in label_keys:
                if frame.get(key, None) is not None:
                    frame_summary[key] = {}
                    frame_summary[key]['shape'] = frame[key].shape
                    if stats:
                        frame_summary[key]['stats'] = kwarray.stats_dict(frame[key], nan=True)
            item_summary['frame_summaries'].append(frame_summary)
            if frame['date_captured']:
                timestamps.append(ub.timeparse(frame['date_captured']))

            if frame.get('ann_aids') is not None:
                if 0 and coco_dset is not None:
                    # disable as workaround for coco sql issues
                    # (i.e. we dont want to rely on a connection to the sql
                    # database in the main thread)
                    annots = coco_dset.annots(frame['ann_aids'])
                    cids = annots.lookup('category_id')
                    class_hist = ub.dict_hist(ub.udict(self.classes.id_to_node).take(cids))
                    frame_summary['class_hist'] = class_hist
                frame_summary['num_annots'] = len(frame['ann_aids'])

        vidname = item.get('video_name', None)
        if vidname is not None:
            item_summary['video_name'] = vidname
            if coco_dset is not None:
                try:
                    video = coco_dset.index.name_to_video[vidname]
                    vid_w = video['width']
                    vid_h = video['height']
                    item_summary['video_hw'] = (vid_h, vid_w)
                except (KeyError, AttributeError):
                    item_summary['video_hw'] = '?'

        if len(timestamps) > 1:
            deltas = np.diff(timestamps)
            deltas = [d.total_seconds() for d in deltas]
            item_summary['min_time'] = ub.timestamp(min(timestamps))
            item_summary['max_time'] = ub.timestamp(max(timestamps))
            if len(deltas):
                item_summary['min_delta'] = min(deltas)
                item_summary['max_delta'] = max(deltas)
                item_summary['mean_delta'] = np.mean(deltas)
        item_summary['input_gsd'] = item['input_gsd']
        item_summary['output_gsd'] = item['output_gsd']

        if 'requested_target' in item:
            item_summary['requested_target'] = item['requested_target']

        if 'target' in item:
            item_summary['resolved_target'] = item['target']

        item_summary['producer_rank'] = item.get('producer_rank', None)
        item_summary['producer_mode'] = item.get('producer_mode', None)
        item_summary['requested_index'] = item.get('requested_index', None)
        item_summary['resolved_index'] = item.get('resolved_index', None)
        return item_summary


class HomogeneousBatchItem(HeterogeneousBatchItem):
    """
    Ideally this is a simplified representation that "just works" with standard
    off the shelf networks.

    Example:
        >>> from geowatch.tasks.fusion.datamodules.network_io import *  # NOQA
        >>> self = HomogeneousBatchItem.demo()
        >>> print(self)
        >>> print(ub.urepr(self.summarize(), nl=2))
        >>> canvas = self.draw()
        >>> # xdoctest: +REQUIRES(--show)
        >>> import kwplot
        >>> kwplot.autompl()
        >>> kwplot.imshow(canvas, fnum=1, pnum=(1, 1, 1))
        >>> kwplot.show_if_requested()
    """

    @classmethod
    def demo(cls):
        """
        Example:
            cls = HomogeneousBatchItem
            self = cls.demo()
        """
        from geowatch.tasks.fusion.datamodules import kwcoco_dataset
        import geowatch
        coco_dset = geowatch.coerce_kwcoco('vidshapes1', num_frames=10)
        dataset = kwcoco_dataset.KWCocoVideoDataset(
            coco_dset, time_dims=3, window_dims=(300, 300),
            channels='r|g|b')
        dataset.disable_augmenter = True
        index = dataset.sample_grid['targets'][dataset.sample_grid['positives_indexes'][0]]
        item = dataset[index]
        self = cls(**item)
        return self


class RGBImageBatchItem(HomogeneousBatchItem):
    """
    Only allows a single RGB image as the input.

    Example:
        >>> from geowatch.tasks.fusion.datamodules.network_io import *  # NOQA
        >>> self = RGBImageBatchItem.demo()
        >>> print(self)
        >>> print(ub.urepr(self.summarize(), nl=2))
        >>> canvas = self.draw()
        >>> # xdoctest: +REQUIRES(--show)
        >>> import kwplot
        >>> kwplot.autompl()
        >>> kwplot.imshow(canvas, fnum=1, pnum=(1, 1, 1))
        >>> kwplot.show_if_requested()
    """
    def __nice__(self):
        return f'num_frames={self.num_frames}, sensorchans={self.sensorchan_histogram}'

    @property
    def frame(self):
        frame = self['frames'][0]
        return frame

    @property
    def channels(self):
        frame = self.frame
        modes = frame['modes']
        mode_keys = list(modes.keys())
        assert len(mode_keys) == 1
        mode_key = mode_keys[0]
        return mode_key

    @property
    def imdata_chw(self):
        frame = self.frame
        modes = frame['modes']
        mode_keys = list(modes.keys())
        assert len(mode_keys) == 1
        mode_key = mode_keys[0]
        mode_val = modes[mode_key]
        imdata_chw = mode_val
        return imdata_chw

    @property
    def nonlocal_class_ohe(self):
        frame = self.frame
        nonlocal_class_ohe = frame['nonlocal_class_ohe']
        return nonlocal_class_ohe

    @classmethod
    def demo(cls, index=None):
        """
        cls = RGBImageBatchItem
        """
        dataset = _demo_dataset()
        if index is None:
            index = dataset.sample_grid['targets'][dataset.sample_grid['positives_indexes'][0]]
        item = dataset[index]
        self = cls(**item)
        assert len(item['frames']) == 1
        frame = self['frames'][0]
        modes = frame['modes']
        mode_keys = list(modes.keys())
        assert len(mode_keys) == 1
        # mode_key = mode_keys[0]
        # mode_val = modes[mode_key]
        # new_item = ub.udict(item) - {'frames'}
        # new_frame = ub.udict(frame) - {'modes'}
        # new_frame['channels'] = mode_key
        # new_frame['imdata_chw'] = mode_val
        # new_item.update(new_frame)
        return self


@ub.memoize
def _demo_dataset():
    """
    Cached dataset for more efficient testing
    """
    from geowatch.tasks.fusion.datamodules import kwcoco_dataset
    import geowatch
    coco_dset = geowatch.coerce_kwcoco('vidshapes1', num_frames=1)
    dataset = kwcoco_dataset.KWCocoVideoDataset(
        coco_dset, time_dims=1, window_dims=(300, 300),
        channels='r|g|b')

    dataset.requested_tasks['nonlocal_class'] = True
    dataset.disable_augmenter = True
    return dataset


# ----------------------------------
# Non-collated Batch Item Containers
# ----------------------------------


class UncollatedBatch(list):
    """
    A generic list of batch items, which may or may not be collatable.
    """


class HeterogeneousBatch(UncollatedBatch):
    """
    A HeterogeneousBatch a ``List[HeterogeneousBatchItem]``.
    """


class UncollatedRGBImageBatch(UncollatedBatch):
    """
    A list of collatable RGBImageBatchItem
    """

    @classmethod
    def demo(cls, num_items=3):
        """
        """
        self = cls.from_items(RGBImageBatchItem.demo(_) for _ in range(num_items))
        return self

    @classmethod
    def coerce(cls, data):
        if isinstance(data, list):
            self = cls.from_items(data)
        else:
            raise NotImplementedError
        return self

    @classmethod
    def from_items(cls, data):
        self = cls(RGBImageBatchItem(item) for item in data)
        return self

    def collate(self):
        """
        Returns:
            CollatedRGBImageBatch

        Example:
            >>> from geowatch.tasks.fusion.datamodules.network_io import *  # NOQA
            >>> self = UncollatedRGBImageBatch.demo()
            >>> batch = self.collate()
        """
        import torch
        imdatas = [batch_item.imdata_chw for batch_item in self]
        imdata_bchw = torch.stack(imdatas)

        batch = CollatedRGBImageBatch()
        batch['imdata_bchw'] = imdata_bchw

        try:
            nonlocal_class_ohes = [batch_item.nonlocal_class_ohe for batch_item in self]
        except KeyError:
            ...
        else:
            nonlocal_class_ohe = torch.stack(nonlocal_class_ohes)
            batch['nonlocal_class_ohe'] = nonlocal_class_ohe

        return batch


# ------------------------------
# Collated Batch Item Containers
# ------------------------------


class CollatedBatch(dict):

    def asdict(self):
        return dict(self)


class CollatedRGBImageBatch(CollatedBatch):

    def to(self, device):
        for k, v in self.items():
            self[k] = v.to(device)
        return self


# ---------------
# Network Outputs
# ---------------
# The concept of a network input needs to be mirrored by a network output.


class NetworkOutputs(dict):
    """
    Network outputs should ALWAYS be a dictionary, this is the most flexible
    way to encode networks such that they can be extended later.
    """

    def _debug_shape(self):
        # from geowatch.utils.util_netharn import _debug_inbatch_shapes
        # _debug_inbatch_shapes(self)
        import torch
        import ubelt as ub
        inbatch = self
        print('len(inbatch) = {}'.format(len(inbatch)))
        extensions = ub.util_format.FormatterExtensions()
        #
        @extensions.register((torch.Tensor, np.ndarray))
        def format_shape(data, **kwargs):
            return ub.repr2(dict(type=str(type(data)), shape=data.shape), nl=1, sv=1)
        print('inbatch = ' + ub.repr2(inbatch, extensions=extensions, nl=True))


# ------------------------------------
# Collated Network Output Containers
# ------------------------------------

class UncollatedNetworkOutputs(NetworkOutputs):
    ...


# ------------------------------------
# Uncollated Network Output Containers
# ------------------------------------

class CollatedNetworkOutputs(NetworkOutputs):
    """
    Example:
        >>> from geowatch.tasks.fusion.datamodules.network_io import *  # NOQA
        >>> B, H, W, C = 2, 3, 3, 11
        >>> import torch
        >>> logits = {
        >>>     'nonlocal_class': (torch.rand(B, C) - 0.5) * 10,
        >>>     'segmentation_class': (torch.rand(B, W, H, C) - 0.5) * 10,
        >>>     'nonlocal_saliency': (torch.rand(B, 1) - 0.5) * 10,
        >>>     'segmentation_saliency': (torch.rand(B, W, H, 1) - 0.5) * 10,
        >>> }
        >>> self = CollatedNetworkOutputs(
        >>>     logits=logits,
        >>>     probs={k + '_probs': v.sigmoid() for k, v in logits.items()},
        >>>     loss_parts={},
        >>>     loss=10,
        >>> )
        >>> new = self.decollate()
        >>> self._debug_shape()
        >>> new._debug_shape()
    """

    def decollate(self):
        """
        Convert back into a per-item structure for easier analysis / drawing.
        """
        new = UncollatedNetworkOutputs()
        new['loss'] = self.get('loss', None)
        new['item_probs'] = decollate(self['probs'])
        return new


def decollate(collated):
    """
    Breakup a collated batch in a standardized way.
    Returns a list of items for each batch item with a structure that matches
    the collated batch, but without the leading batch dimension in each value.

    Example:
        >>> from geowatch.tasks.fusion.datamodules.network_io import *  # NOQA
        >>> import torch
        >>> B, H, W, C = 5, 2, 3, 7
        >>> collated = {
        >>>     'segmentation_class': torch.rand(B, H, W, C),
        >>>     'nonlocal_class': torch.rand(B, C),
        >>> }
        >>> uncollated = decollate(collated)
        >>> assert len(uncollated) == B
        >>> assert (uncollated[0]['nonlocal_class'] == collated['nonlocal_class'][0]).all()
    """
    import ubelt as ub
    # TODO: make more efficient?
    walker = ub.IndexableWalker(collated)
    uncollated_dict = ub.AutoDict()
    uncollated_walker = ub.IndexableWalker(uncollated_dict)
    for path, batch_val in walker:
        for bx, item_val in enumerate(batch_val):
            uncollated_walker[[bx] + path] = item_val
    uncollated = list(uncollated_dict.to_dict().values())
    return uncollated
