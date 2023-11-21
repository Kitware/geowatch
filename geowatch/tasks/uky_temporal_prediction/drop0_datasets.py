import torch
import kwcoco
import kwimage
import random
import itertools as it


class drop0_pairs(torch.utils.data.Dataset):
    """
    Dataset return pairs if images from the subset aligned drop0 data. The
    output is a pair of images along with a pair of dates for the images.
    Sensor may be chosen from S2, LC, or WV. Uses the underlying
    drop0_aligned_segmented class.

    Example:
        >>> # Test with coco demodata
        >>> from geowatch.tasks.uky_temporal_prediction.drop0_datasets import *  # NOQA
        >>> sensor = None
        >>> coco_dset = kwcoco.CocoDataset.demo('special:vidshapes8-multispectral')
        >>> # Hack in date_captured to each image
        >>> # TODO: we could make a demodata wrapper that constructs
        >>> # a demo dataset that works for our purposes
        >>> import dateutil.parser
        >>> import datetime
        >>> base_time = dateutil.parser.parse('2020-03-15')
        >>> delta_time = datetime.timedelta(days=1)
        >>> next_time = base_time
        >>> for vidid, gids in coco_dset.index.vidid_to_gids.items():
        ...     for gid in gids:
        ...         next_time = next_time + delta_time
        ...         img = coco_dset.index.imgs[gid]
        ...         img['date_captured'] = datetime.datetime.isoformat(next_time)
        >>> self = drop0_pairs(coco_dset, sensor=sensor, video=None)
        >>> idx = 0
        >>> item = self[idx]
    """

    def __init__(self,
                 coco_dset,
                 sensor='S2',
                 panchromatic=True,
                 video=1,
                 min_time_step=1,
                 change_labels=list(range(14))):

        self.dataset = drop0_aligned(
            coco_dset=coco_dset,
            sensor=sensor,
            panchromatic=panchromatic,
            video=video,
            change_labels=change_labels)
        self.length = len(self.dataset)
        self.min_time_step = min_time_step

    def __len__(self,):
        return self.length

    def __getitem__(self, idx):
        view1 = self.dataset.__getitem__(idx)
        im1, date1 = view1['image'], view1['date']
        idx2 = idx
        while abs(idx2 - idx) < self.min_time_step:
            idx2 = random.randint(0, self.length - 1)

        view2 = self.dataset.__getitem__(idx2)
        im2, date2 = view2['image'], view2['date']

        date1 = (int(date1[:4]), int(date1[5:7]), int(date1[8:10]))
        date2 = (int(date2[:4]), int(date2[5:7]), int(date2[8:10]))

        if date2 < date1:
            im1, im2 = im2, im1

        date1 = torch.tensor(date1)
        date2 = torch.tensor(date2)

        item = {
            'image1': im1,
            'image2': im2,
            'date1': date1,
            'date2': date2
        }
        return item


class drop0_aligned_change(torch.utils.data.Dataset):
    """
    Example:
        >>> # Test with coco demodata
        >>> from geowatch.tasks.uky_temporal_prediction.drop0_datasets import *  # NOQA
        >>> coco_dset = 'special:vidshapes8-multispectral'
        >>> sensor = None
        >>> self = drop0_aligned_change(coco_dset, sensor=sensor, video=None)
        >>> idx = 0
        >>> item = self[idx]
    """

    def __init__(self, coco_dset,
                 sensor='S2',
                 panchromatic=True,
                 video=1,
                 soften_by=0,
                 change_labels=list(range(14))):
        self.dataset = drop0_aligned_segmented(
            sensor=sensor,
            panchromatic=panchromatic,
            video=video,
            change_labels=change_labels,
            coco_dset=coco_dset)
        self.soften_by = soften_by
        self.length = len(self.dataset)

    def __len__(self,):
        return self.length

    def __getitem__(self, idx):

        # TODO: This will fail if subsequent items are from different videos
        # The constructor should make a list of image-id pairs, which
        # are then sampled from in order to make a more robust dataset.

        item1 = self.dataset.__getitem__(idx)
        im1, seg1, date1 = item1['image'], item1['mask'], item1['date']
        frame_index1 = item1.get('frame_index', None)

        idx2 = idx
        while abs(idx2 - idx) < 1:
            idx2 = random.randint(0, self.length - 1)

        item2 = self.dataset.__getitem__(idx2)
        im2, seg2, date2 = item2['image'], item2['mask'], item2['date']
        frame_index2 = item2.get('frame_index', None)

        if date2 is not None and date1 is not None:
            if date2 < date1:
                im1, im2 = im2, im1
                seg1, seg2 = seg2, seg1
        else:
            if frame_index1 < frame_index2:
                im1, im2 = im2, im1
                seg1, seg2 = seg2, seg1

        cmap = torch.where(seg2 - seg1 != 0, 1., 0. + self.soften_by)

        item = {
            'image1': im1,
            'image2': im2,
            'cmap': cmap
        }
        return item


class drop0_aligned_segmented(torch.utils.data.Dataset):
    """
    Dataset compatible with drop0_aligned_v2 (now just drop0_aligned on DVC).

    Sensor must be 'WV' (Worldview), 'LC' (Land Cover) or 'S2' (Sentinel 2). If
    'WV' is chosen, specify if you want panchromatic (single channel) images by
    setting panchromatic=True. If False, 8 channel multi-spectral images will
    be returned.

    In current drop, all Sentinel 2 images are RGB only. Annotations give
    bounding box/segmentation outlines of construction sites, but we do not
    have pixel level annotations for building segmentation or change detection.

    There are 5 "videos" in the dataset of aligned images across a single
    location. Set video=0 to return images from all videos (note these will not
    all be the same size). Otherwise choose which video to return images from.

    Land Cover: Videos 1,4
    WV multi-sprectral: Video 5
    WV panchromatic: Videos 1,2,5
    S2: Videos 1,4,5
    """

    def __init__(self, coco_dset,
                 sensor='S2', panchromatic=True, video=1, change_labels=[2, 3, 4, 7, 8, 9, 11]):

        self.sensor = sensor

        # by default only take contruction based labels, ignore "transient
        # construction"
        self.accepted_labels = change_labels

        self.video_id = video
        dset = kwcoco.CocoDataset.coerce(coco_dset)

        if self.video_id is None:
            # Use all videos if not specified
            video_ids_of_interest = list(dset.index.videos.keys())
        else:
            video_ids_of_interest = [self.video_id]

        # A flat list of images belonging to those videos
        valid_image_ids = list(it.chain.from_iterable(
            [dset.index.vidid_to_gids[vidid] for vidid in video_ids_of_interest]))

        # An `Images` object for all the valid images
        valid_images = dset.images(valid_image_ids)

        # Restrict to correct sensor
        if sensor is not None:
            valid_images = valid_images.compress(
                [x == sensor for x in valid_images.lookup('sensor_coarse')])

        if 'WV' == sensor:
            if panchromatic:
                valid_images = valid_images.compress(
                    [num_bands == 1 for num_bands in valid_images.lookup('num_bands')])
                self.ms = False
            else:
                valid_images = valid_images.compress(
                    [num_bands == 8 for num_bands in valid_images.lookup('num_bands')])
                self.ms = True

        print('Built drop0_aligned_segmented dataset with {} valid images'.format(
            len(valid_images)))

        self.dset_ids = valid_images.gids
        self.annotations = dset.annots
        self.images = valid_images

        self.dset = dset

    def __len__(self):
        return len(self.dset_ids)

    def __getitem__(self, idx):

        gid = self.dset_ids[idx]
        # annot_ids = self.dset.index.gid_to_aids[gid]

        aids = self.dset.index.gid_to_aids[gid]
        dets = kwimage.Detections.from_coco_annots(
            self.dset.annots(aids).objs, dset=self.dset)

        # bbox = dets.data['boxes'].data
        segmentation = dets.data['segmentations'].data
        category_id = [dets.classes.idx_to_id[cidx]
                       for cidx in dets.data['class_idxs']]

        img = self.dset.index.imgs[gid]
        acquisition_date = img.get('date_captured', None)
        frame_index = img.get('frame_index', None)

        if False:
            # Requires new kwcoco methods
            delayed_image = self.dset.delayed_load(
                gid, channels=..., space='video')
        else:
            # Hack to simply load all channels,
            # TODO: The dataset needs to know what the set of channels that it
            # is supposed to output will be.
            delayed_image = self.dset.delayed_load(gid)
            im = delayed_image.finalize()

        im = torch.from_numpy(im.astype('int16'))

        if len(im.shape) < 3:
            im = im.unsqueeze(0)
            if self.sensor == 'WV':
                im = im / 2048.  # rough normalization
            else:
                im = im / 32000.  # rough normalization

        else:
            im = im.permute(2, 0, 1)
            if self.sensor == 'S2':
                im = im / 255.
            elif self.sensor == 'WV':
                im = im / 2048.

        # create segmentation mask

        # class_idxs = dets.data['class_idxs']
        img_dims = (img['height'], img['width'])
        combined = []

        for sseg, cid in zip(segmentation, category_id):
            assert cid > 0
            np_mask = sseg.to_mask(dims=img_dims).data.astype(float) * cid
            mask = torch.from_numpy(np_mask)
            combined.append(mask.unsqueeze(0))

        if combined:
            overall_mask = torch.max(torch.cat(combined, dim=0), dim=0)[0]
        else:
            overall_mask = torch.zeros_like(im)
        #####

        item = {
            'image': im,
            'mask': overall_mask,
            'date': acquisition_date,
            'frame_index': frame_index,
        }
        return item


class drop0_aligned(torch.utils.data.Dataset):
    """
    Dataset compatible with drop0_aligned_v2 (now just drop0_aligned on DVC).

    Data input can be a generic kwcoco file, but we do expect certain fields
    associated with watch data.

    Sensor must be 'WV' (Worldview), 'LC' (Land Cover) or 'S2' (Sentinel 2). If
    'WV' is chosen, specify if you want panchromatic (single channel) images by
    setting panchromatic=True. If False, 8 channel multi-spectral images will
    be returned.

    In current drop, all Sentinel 2 images are RGB only. Annotations give
    bounding box/segmentation outlines of construction sites, but we do not
    have pixel level annotations for building segmentation or change detection.

    There are 5 "videos" in the dataset of aligned images across a single
    location. Set video=0 to return images from all videos (note these will not
    all be the same size). Otherwise choose which video to return images from.

    Land Cover: Videos 1,4
    WV multi-sprectral: Video 5
    WV panchromatic: Videos 1,2,5
    S2: Videos 1,4,5

    Example:
        >>> # Test with coco demodata
        >>> from geowatch.tasks.uky_temporal_prediction.drop0_datasets import *  # NOQA
        >>> coco_dset = 'special:vidshapes8-multispectral'
        >>> sensor = None
        >>> self = drop0_aligned(coco_dset, sensor=sensor, video=None)
        >>> idx = 0
        >>> item = self[idx]
    """

    def __init__(self, coco_dset, sensor='S2', panchromatic=True,
                 video=None, change_labels=[2, 3, 4, 7, 8, 9, 11]):

        self.sensor = sensor

        # by default only take contruction based labels, ignore "transient
        # construction"
        self.accepted_labels = change_labels

        self.video_id = video
        dset = kwcoco.CocoDataset.coerce(coco_dset)

        if self.video_id is None:
            # Use all videos if not specified
            video_ids_of_interest = list(dset.index.videos.keys())
        else:
            video_ids_of_interest = [self.video_id]

        if 0:
            # print number of images per sensor for each video
            import ubelt as ub
            for vidid, gids in dset.index.vidid_to_gids.items():
                avail_sensors = dset.images(gids).lookup('sensor_coarse', None)
                sensor_freq = ub.dict_hist(avail_sensors)
                print('vidid = {} sensor_freq = {}'.format(vidid, sensor_freq))

        # A flat list of images belonging to those videos
        valid_image_ids = list(it.chain.from_iterable(
            [dset.index.vidid_to_gids[vidid]
             for vidid in video_ids_of_interest]))

        # An `Images` object for all the valid images
        valid_images = dset.images(valid_image_ids)

        # Restrict to correct sensor
        if sensor is not None:
            valid_images = valid_images.compress(
                [x == sensor for x in valid_images.lookup('sensor_coarse')])

        if 'WV' == sensor:
            if panchromatic:
                valid_images = valid_images.compress(
                    [num_bands == 1 for num_bands in valid_images.lookup('num_bands')])
                self.ms = False
            else:
                valid_images = valid_images.compress(
                    [num_bands == 8 for num_bands in valid_images.lookup('num_bands')])
                self.ms = True

        if len(valid_images) == 0:
            raise ValueError('Dataset and filter criteria have no images')

        self.dset_ids = valid_images.gids
        self.annotations = dset.annots
        self.images = valid_images

        self.dset = dset

    def __len__(self):
        return len(self.dset_ids)

    def __getitem__(self, idx):

        gid = self.dset_ids[idx]
        # annot_ids = seldx f.dset.index.gid_to_aids[gid]

        # aids = self.dset.index.gid_to_aids[gid]
        # dets = kwimage.Detections.from_coco_annots(
        #     self.dset.annots(aids).objs, dset=self.dset)

        # bbox = dets.data['boxes'].data
        # segmentation = dets.data['segmentations'].data
        # category_id = [dets.classes.idx_to_id[cidx]
        #                for cidx in dets.data['class_idxs']]

        img = self.dset.index.imgs[gid]
        acquisition_date = img.get('date_captured', None)
        frame_index = img.get('frame_index', None)

        if False:
            # Requires new kwcoco methods
            delayed_image = self.dset.delayed_load(
                gid, channels=..., space='video')
        else:
            # Hack to simply load all channels,
            # TODO: The dataset needs to know what the set of channels that it
            # is supposed to output will be.
            delayed_image = self.dset.delayed_load(gid)
            im = delayed_image.finalize()

        im = torch.from_numpy(im.astype('int16'))

        if len(im.shape) < 3:
            im = im.unsqueeze(0)
            if self.sensor == 'WV':
                im = im / 2048.  # rough normalization
            else:
                im = im / 32000.  # rough normalization

        else:
            im = im.permute(2, 0, 1)
            if self.sensor == 'S2':
                im = im / 255.
            elif self.sensor == 'WV':
                im = im / 2048.

        item = {
            'image': im,
            'date': acquisition_date,
            'frame_index': frame_index,
        }
        return item
