"""
Module to define the :class:`BatchItem` wrapper class.

This is currently a work in progress, and the goal is to abstract the format of
the items produced by the datalaoder to make them ammenable for use with both
our heterogeneous networks as well as more standard networks that require data
be more regular.
"""
import kwarray
import ubelt as ub
import numpy as np


class BatchItem(dict):
    """
    A BatchItem is a container to help organize the output of the
    KWCocoVideoDataset. For backwards compatibility it retains the original
    dictionary interface.

    Example:
        >>> from geowatch.tasks.fusion.datamodules.batch_item import *  # NOQA
        >>> self = BatchItem.demo()
        >>> print(self)
        >>> print(ub.urepr(self.summarize(), nl=2))
    """

    def __nice__(self):
        return f'num_frames={self.num_frames}, sensorchans={self.sensorchan_histogram}'

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

    @classmethod
    def demo(cls):
        from geowatch.tasks.fusion.datamodules import kwcoco_dataset
        import geowatch
        coco_dset = geowatch.coerce_kwcoco('vidshapes1', num_frames=10)
        dataset = kwcoco_dataset.KWCocoVideoDataset(
            coco_dset, time_dims=4, window_dims=(300, 300),
            channels='r|g|b')
        dataset.disable_augmenter = True
        index = dataset.sample_grid['targets'][dataset.sample_grid['positives_indexes'][0]]
        item = dataset[index]
        self = cls(**item)
        return self

    @property
    def num_frames(self):
        return len(self['frames'])

    @property
    def sensorchan_histogram(self):
        histogram = ub.dict_hist(
            frame['sensor'] + ':' + mode_key for frame in self['frames']
            for mode_key in frame['modes'].keys()
        )
        return histogram

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
            >>> from geowatch.tasks.fusion.datamodules.batch_item import *  # NOQA
            >>> self = BatchItem.demo()
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
