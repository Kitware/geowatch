import torch
import kwcoco
import kwimage
import tifffile
import os.path as osp
import random
import itertools as it


class drop0_pairs(torch.utils.data.Dataset):
    """
    Dataset return pairs if images from the subset aligned drop0 data. The
    output is a pair of images along with a pair of dates for the images.
    Sensor may be chosen from S2, LC, or WV. Uses the underlying
    drop0_aligned_segmented class.
    """

    def __init__(self,
                 root='/localdisk0/SCRATCH/watch/smart_watch_dvc/drop0_aligned/',
                 sensor='S2',
                 panchromatic=True,
                 video=1,
                 min_time_step=1,
                 change_labels=list(range(14))):

        self.dataset = drop0_aligned(
            root=root,
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
        im, date = view1['image'], view1['date']
        idx2 = idx
        while abs(idx2 - idx) < self.min_time_step:
            idx2 = random.randint(0, self.length - 1)

        view2 = self.dataset.__getitem__(idx2)
        im2, date2 = view2['image'], view2['date']

        date = (int(date[:4]), int(date[5:7]), int(date[8:]))
        date2 = (int(date2[:4]), int(date2[5:7]), int(date2[8:]))

        if date2 < date:
            im, im2 = im2, im

        date = torch.tensor(date)
        date2 = torch.tensor(date2)

        return {'image1': im,
                'image2': im2,
                'date1': date,
                'date2': date2
                }


class drop0_aligned_change(torch.utils.data.Dataset):
    def __init__(self, root='/localdisk0/SCRATCH/watch/smart_watch_dvc/drop0_aligned/',
                 sensor='S2',
                 sites='all',
                 panchromatic=True,
                 video=1,
                 soften_by=0,
                 change_labels=list(range(14))):
        self.dataset = drop0_aligned_segmented(
            sensor=sensor,
            sites=sites,
            panchromatic=panchromatic,
            video=video,
            change_labels=change_labels,
            root=root)
        self.soften_by = soften_by
        self.length = len(self.dataset)

    def __len__(self,):
        return self.length

    def __getitem__(self, idx):
        item1 = self.dataset.__getitem__(idx)
        im, seg, date = item1['image'], item1['mask'], item1['date']

        idx2 = idx
        while abs(idx2 - idx) < 1:
            idx2 = random.randint(0, self.length - 1)
        
        item2 = self.dataset.__getitem__(idx2)
        im2, seg2, date2 = item2['image'], item2['mask'], item2['date']
        
        if date2 < date:
            im, im2 = im2, im
            seg, seg2 = seg2, seg

        cmap = torch.where(seg2 - seg != 0, 1., 0. + self.soften_by)

        item = {
            'image1': im,
            'image2': im2,
            'cmap': cmap
        }
        return item


class drop0_aligned_segmented(torch.utils.data.Dataset):
    """
    Dataset compatible with drop0_aligned_v2 (now just drop0_aligned on DVC).
    Sites must be a list from the following:
        ['AE-Dubai-0001',
         'BR-Rio-0270',
         'BR-Rio-0277',
         'KR-Pyeongchang-S2',
         'KR-Pyeongchang-WV',
         'US-Waynesboro-0001']
    or be set as 'all'.

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

    def __init__(self, root='/localdisk0/SCRATCH/watch/smart_watch_dvc/drop0_aligned/',
                 sensor='S2', sites='all', panchromatic=True, video=1, change_labels=[2, 3, 4, 7, 8, 9, 11]):

        self.sensor = sensor

        # by default only take contruction based labels, ignore "transient
        # construction"
        self.accepted_labels = change_labels

        if sites == 'all':
            sites = ['AE-Dubai-0001',
                     'BR-Rio-0270',
                     'BR-Rio-0277',
                     'KR-Pyeongchang-S2',
                     'KR-Pyeongchang-WV',
                     'US-Waynesboro-0001']

        self.video_id = video
        self.root = root
        self.json_file = osp.join(self.root, 'data.kwcoco.json')
        dset = kwcoco.CocoDataset(self.json_file)

        if self.video_id:
            # video_ids = dset.index.vidid_to_gids[self.video_id]
            video_ids_of_interest = [self.video_id]
        else:
            # video_ids = dset.images().gids
            video_ids_of_interest = [1, 2, 3, 4, 5]

        # sensor_list = dset.images().lookup('sensor_coarse', keepid=True)
        # sensor_ids = [ID for ID in sensor_list if sensor_list[ID] == sensor]

        # A flat list of images belonging to those videos
        valid_image_ids = list(it.chain.from_iterable(
            [dset.index.vidid_to_gids[vidid] for vidid in video_ids_of_interest]))

        # An `Images` object for all the valid images
        valid_images = dset.images(valid_image_ids)

        # Restrict to correct sensor
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
        filename = osp.join(self.root, img['file_name'])
        acquisition_date = img['date_captured']

        im = tifffile.imread(filename)
        im = torch.tensor(im.astype('int16'))

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
            # FIXME? This does not seem to actually return annotations
            'annotations': img
        }
        return item


class drop0_aligned(torch.utils.data.Dataset):
    """
    Dataset compatible with drop0_aligned_v2 (now just drop0_aligned on DVC).
    Sites must be a list from the following:

        ['AE-Dubai-0001',
         'BR-Rio-0270',
         'BR-Rio-0277',
         'KR-Pyeongchang-S2',
         'KR-Pyeongchang-WV',
         'US-Waynesboro-0001']

    or be set as 'all'.

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

    def __init__(self, root='/localdisk0/SCRATCH/watch/smart_watch_dvc/drop0_aligned/',
                 sensor='S2', sites='all', panchromatic=True, video=1, change_labels=[2, 3, 4, 7, 8, 9, 11]):

        self.sensor = sensor

        # by default only take contruction based labels, ignore "transient
        # construction"
        self.accepted_labels = change_labels

        if sites == 'all':
            sites = ['AE-Dubai-0001',
                     'BR-Rio-0270',
                     'BR-Rio-0277',
                     'KR-Pyeongchang-S2',
                     'KR-Pyeongchang-WV',
                     'US-Waynesboro-0001']

        self.video_id = video
        self.root = root
        self.json_file = osp.join(self.root, 'data.kwcoco.json')
        dset = kwcoco.CocoDataset(self.json_file)

        if self.video_id:
            # video_ids = dset.index.vidid_to_gids[self.video_id]
            video_ids_of_interest = [self.video_id]
        else:
            # video_ids = dset.images().gids
            video_ids_of_interest = [1, 2, 3, 4, 5]

        # sensor_list = dset.images().lookup('sensor_coarse', keepid=True)
        # sensor_ids = [ID for ID in sensor_list if sensor_list[ID] == sensor]

        # A flat list of images belonging to those videos
        valid_image_ids = list(it.chain.from_iterable(
            [dset.index.vidid_to_gids[vidid] for vidid in video_ids_of_interest]))

        # An `Images` object for all the valid images
        valid_images = dset.images(valid_image_ids)

        # Restrict to correct sensor
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

        self.dset_ids = valid_images.gids
        self.annotations = dset.annots
        self.images = valid_images

        self.dset = dset

    def __len__(self):
        return len(self.dset_ids)

    def __getitem__(self, idx):

        gid = self.dset_ids[idx]
        # annot_ids = self.dset.index.gid_to_aids[gid]

        # aids = self.dset.index.gid_to_aids[gid]
        # dets = kwimage.Detections.from_coco_annots(
        #     self.dset.annots(aids).objs, dset=self.dset)

        # bbox = dets.data['boxes'].data
        # segmentation = dets.data['segmentations'].data
        # category_id = [dets.classes.idx_to_id[cidx]
        #                for cidx in dets.data['class_idxs']]

        img = self.dset.index.imgs[gid]
        filename = osp.join(self.root, img['file_name'])
        acquisition_date = img['date_captured']

        im = tifffile.imread(filename)
        im = torch.tensor(im.astype('int16'))

        im = tifffile.imread(filename)
        im = torch.tensor(im.astype('int16'))

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

        # TODO: Was there intent to use data from dets instead of img in the
        # annotations item?
        item = {
            'image': im,
            'date': acquisition_date,
            'annotations': img
        }
        return item
