from torch.utils.data import Dataset
import torch
import numpy as np
import os
import albumentations as A
from torchvision import transforms
import kwcoco
import kwimage
import kwarray
import random
from pandas import read_csv
import ndsampler
import ubelt as ub
from ..utils.read_sentinel_images import read_sentinel_img_trio


class gridded_dataset(torch.utils.data.Dataset):
    """
    Example:
        >>> # xdoctest: +REQUIRES(env:DVC_DPATH)
        >>> from watch.tasks.invariants.data.datasets import *  # NOQA
        >>> import watch
        >>> dvc_dpath = watch.find_smart_dvc_dpath()
        >>> coco_fpath = dvc_dpath / 'Drop2-Aligned-TA1-2022-02-15/data_nowv_vali.kwcoco.json'
        >>> import kwcoco
        >>> coco_dset = kwcoco.CocoDataset(coco_fpath)
        >>> self = gridded_dataset(coco_dset)
        >>> idx = 0
        >>> out = self[idx]
        >>> rgb1 = out['image1'][0:3].permute(1, 2, 0).numpy()[..., ::-1]
        >>> rgb2 = out['image2'][0:3].permute(1, 2, 0).numpy()[..., ::-1]
        >>> rgb3 = out['offset_image1'][0:3].permute(1, 2, 0).numpy()[..., ::-1]
        >>> rgb4 = out['augmented_image1'][0:3].permute(1, 2, 0).numpy()[..., ::-1]
        >>> # xdoctest: +REQUIRES(--show)
        >>> import kwplot
        >>> kwplot.autompl()
        >>> kwplot.imshow(kwimage.normalize(rgb1), pnum=(1, 4, 1), title='image1')
        >>> kwplot.imshow(kwimage.normalize(rgb2), pnum=(1, 4, 2), title='image2')
        >>> kwplot.imshow(kwimage.normalize(rgb3), pnum=(1, 4, 3), title='offset_image1')
        >>> kwplot.imshow(kwimage.normalize(rgb4), pnum=(1, 4, 4), title='augmented_image1')
        >>> kwplot.show_if_requested()

        loader = torch.utils.data.DataLoader(
            self, num_workers=0, batch_size=1, shuffle=False)
        dliter = iter(loader)
        batch = next(dliter)

    Example:
        >>> # xdoctest: +SKIP
        >>> from watch.tasks.invariants.data.datasets import *  # NOQA
        >>> import watch
        >>> dvc_dpath = watch.find_smart_dvc_dpath()
        >>> coco_fpath = dvc_dpath / 'Drop2-Aligned-TA1-2022-02-15/data_nowv_vali.kwcoco.json'
        >>> import kwcoco
        >>> coco_dset = kwcoco.CocoDataset(coco_fpath)
        >>> self = gridded_dataset(coco_dset)
        >>> dsize = (224, 224)
        >>> # Draw multiple batch items
        >>> rows = []
        >>> max_idx = len(self) // 4 - 2
        >>> indexes = np.linspace(0, max_idx, 10).round().astype(int)
        >>> for idx in indexes:
        >>>     out = self[idx]
        >>>     rgb1 = out['image1'][0:3].permute(1, 2, 0).numpy()
        >>>     rgb2 = out['image2'][0:3].permute(1, 2, 0).numpy()
        >>>     rgb3 = out['offset_image1'][0:3].permute(1, 2, 0).numpy()
        >>>     rgb4 = out['augmented_image1'][0:3].permute(1, 2, 0).numpy()
        >>>     canvas1 = np.nan_to_num(kwimage.imresize(kwimage.normalize(rgb1), dsize=dsize)).clip(0, 1)
        >>>     canvas2 = np.nan_to_num(kwimage.imresize(kwimage.normalize(rgb2), dsize=dsize)).clip(0, 1)
        >>>     canvas3 = np.nan_to_num(kwimage.imresize(kwimage.normalize(rgb3), dsize=dsize)).clip(0, 1)
        >>>     canvas4 = np.nan_to_num(kwimage.imresize(kwimage.normalize(rgb4), dsize=dsize)).clip(0, 1)
        >>>     canvas1 = kwimage.draw_text_on_image(canvas1, 'image1', org=(1, 1), valign='top', color='white', border=2)
        >>>     canvas2 = kwimage.draw_text_on_image(canvas2, 'image2', org=(1, 1), valign='top', color='white', border=2)
        >>>     canvas3 = kwimage.draw_text_on_image(canvas3, 'offset_image1', org=(1, 1), valign='top', color='white', border=2)
        >>>     canvas4 = kwimage.draw_text_on_image(canvas4, 'augmented_image1', org=(1, 1), valign='top', color='white', border=2)
        >>>     row_canvas = kwimage.stack_images([canvas1, canvas2, canvas3, canvas4], axis=1, pad=3)
        >>>     rows.append(row_canvas)
        >>> canvas = kwimage.stack_images(rows, axis=0, pad=10)
        >>> # xdoctest: +REQUIRES(--show)
        >>> import kwplot
        >>> kwplot.autompl()
        >>> kwplot.imshow(canvas)
    """
    S2_l2a_channel_names = [
        'B02.tif', 'B01.tif', 'B03.tif', 'B04.tif', 'B05.tif', 'B06.tif', 'B07.tif', 'B08.tif', 'B09.tif', 'B11.tif', 'B12.tif', 'B8A.tif'
    ]
    S2_channel_names = [
        'coastal', 'blue', 'green', 'red', 'B05', 'B06', 'B07', 'nir', 'B09', 'cirrus', 'swir16', 'swir22', 'B8A'
    ]
    L8_channel_names = [
        'coastal', 'lwir11', 'lwir12', 'blue', 'green', 'red', 'nir', 'swir16', 'swir22', 'pan', 'cirrus'
    ]

    def __init__(self, coco_dset, sensor=['S2', 'L8'], bands=['shared'],
                 segmentation=False, patch_size=128, num_images=2,
                 mode='train', patch_overlap=.25, bas=True, rng=None):
        super().__init__()

        # initialize dataset
        print('load dataset')
        self.coco_dset: kwcoco.CocoDataset = kwcoco.CocoDataset.coerce(coco_dset)

        print('filter dataset')
        # Filter out worldview images (better to use subset than remove)
        images: kwcoco.coco_objects1d.Images = self.coco_dset.images()
        flags = [s != 'WV' for s in images.lookup('sensor_coarse')]
        valid_image_ids : list[int] = list(images.compress(flags))
        self.coco_dset = self.coco_dset.subset(valid_image_ids)

        self.images : kwcoco.coco_objects1d.Images = self.coco_dset.images()
        self.sampler = ndsampler.CocoSampler(self.coco_dset)

        window_dims = [num_images, patch_size, patch_size]

        print('make grid')
        from watch.tasks.fusion.datamodules.kwcoco_video_data import sample_video_spacetime_targets
        sample_grid = sample_video_spacetime_targets(
            self.coco_dset, window_dims=window_dims,
            window_overlap=patch_overlap,
            time_sampling='hardish3', time_span='1y',
            use_annot_info=False,
            keepbound=True,
            exclude_sensors=['WV'],
            use_centered_positives=False,
            # set_cover_algo='approx',
            set_cover_algo=None,
            use_cache=0,
            workers=0,
            window_space_scale=None,
        )
        import copy
        # all_samples = sample_grid['targets']
        all_samples = copy.deepcopy(sample_grid['targets'])
        # import xdev
        # xdev.embed()
        for tr in all_samples:
            tr['vidid'] = tr['video_id']  # hack
            # The second gid is always the main gid in our case
            tr['main_gid'] = tr['gids'][1]
            tr['frame_index'] = coco_dset.imgs[tr['main_gid']]['frame_index']
            tr['main_idx'] = coco_dset.imgs[tr['main_gid']]['frame_index']
            tr['frame_indexes'] = coco_dset.images(tr['gids']).lookup('frame_index')

        if 0:
            # DEBUG:
            # vidid = self.coco_dset.videos()[0]
            # time_sampler = sample_grid['vidid_to_time_sampler'][vidid]
            # gids = set(self.coco_dset.images(video_id=vidid))
            # for loc, main_idx in enumerate(time_sampler.main_indexes):
            #     print(time_sampler.sample(main_idx, exclude=time_sampler.main_indexes[loc:], error_level=0))
            # covered1 = set()
            # covered2 = set()
            # for tr in all_samples:
            #     tr['vidid'] = tr['video_id']  # hack
            #     v = tr['vidid']
            #     if v == vidid:
            #         covered1.add(tr['gids'][0])
            #         covered2.add(tr['gids'][1])
            #         ...
            ...

        # Postprocess the grid we get out of the temporal sampler to make it a
        # little nicer for this problem.

        from watch.utils import util_kwimage
        class HashableBox(util_kwimage.Box):
            def to_tuple(box):
                return tuple([box.format] + box.data.tolist())

        vidid_to_new_samples = {}
        vidid_to_samples = ub.group_items(all_samples, lambda x: x['vidid'])
        for vidid, vid_samples in vidid_to_samples.items():
            time_sampler = sample_grid['vidid_to_time_sampler'][vidid]
            vid_images = coco_dset.images(video_id=vidid)
            frame_idxs = vid_images.lookup('frame_index')
            assert sorted(frame_idxs) == frame_idxs
            frame_to_samples = ub.group_items(vid_samples, lambda x: x['frame_index'])
            missing_frame_idxs = set(frame_idxs) - set(frame_to_samples)

            # Get spatial information about the samples
            from collections import Counter
            spatial_slices = Counter()

            # frame_index_to_timepairs = ub.ddict(Counter)
            for samples in frame_to_samples.values():
                for target in samples:
                    tr = target
                    tr['frame_index']
                    box = HashableBox.from_slice(tr['space_slice'])
                    box_tup = tuple([box.format] + box.data.tolist())
                    spatial_slices.update([box_tup])

            # Make everything valid for this hack
            time_sampler.affinity += np.finfo(np.float32).eps
            # Fill in the gaps the sampler missed
            new_frame_to_samples = ub.ddict(list)
            for idx in missing_frame_idxs:
                if idx == frame_idxs[0]:
                    continue
                # Mask out everything in the future. We must take something
                # from the past.

                exclude_idxs = time_sampler.indexes[time_sampler.indexes > idx]
                sample_idxs = time_sampler.sample(idx, exclude=exclude_idxs)

                frame_index = frame_idxs[idx]
                sample_gids = time_sampler.video_gids[sample_idxs]
                main_gid = sample_gids[1]

                partial_tr = ub.udict({
                    'main_idx':  int(frame_index),
                    'video_id': vidid,
                    'vidid': vidid,
                    'gids': list(map(int, sample_gids)),
                    'main_gid': int(main_gid),
                    'frame_index': int(frame_index),
                    'frame_indexes': list(map(int, ub.take(frame_idxs, sample_idxs))),
                    # 'space_slice': (slice(0, 256, None), slice(0, 256, None)),
                    'resampled': None,
                    'label': None,
                })
                for space_slice in spatial_slices.keys():
                    format, a, b, c, d = space_slice
                    box = util_kwimage.Box.coerce([[a, b, c, d]], format=format)
                    space_slice = box.to_slice()
                    new_tr = partial_tr | {
                        'space_slice': space_slice,
                    }
                    if 1:
                        new_frame_to_samples[frame_index].append(new_tr)

            # For each each main frame, choose only one other frame as it's
            # pair.
            for frame_index, samples in frame_to_samples.items():
                # TODO: spatial coverage
                chosen = ub.udict(ub.group_items(samples, lambda x: x['gids'][0])).peek_value()
                new_frame_to_samples[frame_index] = chosen

            vidid_to_new_samples[vidid] = list(ub.flatten(new_frame_to_samples.values()))

        self.patches = []

        for vidid, samples in vidid_to_new_samples.items():
            samples = sorted(samples, key=lambda x: x['frame_index'])
            self.patches.extend(samples)

            if 0:
                # Check ordering
                prev_frame_index = -1
                for sample in samples:
                    assert sample['frame_index'] >= prev_frame_index
                    prev_frame_index = sample['frame_index']
                    assert np.all(np.array(sample['frame_indexes']) >= sample['frame_indexes'])

        # [x['vidid']] + [self.coco_dset.imgs[gid]['frame_index'] for gid in x['gids']]
        # self.patches : list[dict] = list(ub.flatten(vidid_to_new_samples.values()))
        # self.patches = sorted(list(ub.flatten(vidid_to_new_samples.values())),
        #                       key=lambda x: (x['video_id'], x['frame_index']))

        all_bands = [
            aux.get('channels', None)
            for aux in self.coco_dset.index.imgs[self.images._ids[0]].get('auxiliary', [])]

        if 'r|g|b' in all_bands:
            all_bands.remove('r|g|b')
        self.bands = []
        # no channels selected
        if len(bands) < 1:
            raise ValueError(f'bands must be specified. Options are {", ".join(all_bands)}, or all')
        # all channels selected
        elif len(bands) == 1:
            if bands[0].lower() == 'all':
                self.bands = all_bands
            elif bands[0].lower() == 'shared':
                self.bands = ['red', 'green', 'blue', 'nir', 'swir16', 'swir22']
            elif bands[0] == 'r|g|b':
                self.bands.append('r|g|b')
        else:
            for band in bands:
                if band in all_bands:
                    self.bands.append(band)
        self.num_channels = len(self.bands)
        self.bands = "|".join(self.bands)

        # define augmentations
        print('build augs')
        additional_targets = dict()
        self.num_images = num_images

        for i in range(self.num_images):
            additional_targets['image{}'.format(1 + i)] = 'image'
            additional_targets['seg{}'.format(i + 1)] = 'mask'

        if mode == 'train':
            self.transforms = A.Compose([A.OneOf([
                            A.MotionBlur(p=.5),
                            A.Blur(blur_limit=7, p=1),
                        ], p=.9),
                        A.GaussNoise(var_limit=.002),
                        A.RandomBrightnessContrast(brightness_limit=.3, contrast_limit=.3, brightness_by_max=False, always_apply=True)
                    ],
                    additional_targets=additional_targets)
        else:
            ### deterministic transforms for test mode
            self.transforms = A.Compose([
                            A.Blur(blur_limit=[4, 4], p=1),
                            A.RandomBrightnessContrast(brightness_limit=[.2, .2], contrast_limit=[.2, .2], brightness_by_max=False, always_apply=True)
                    ],
                    additional_targets=additional_targets)

        self.mode = mode
        self.segmentation = segmentation
        self.patch_size = patch_size
        self.bas = bas
        if self.bas:
            self.positive_indices = [0, 1, 3]
            self.ignore_indices = [2, 6]
        else:
            self.positive_indices = [0, 1, 2, 3]
            self.ignore_indices = [6]
        print('finished dataset init')

    def __len__(self):
        return len(self.patches)

    def __getitem__(self, idx):
        tr : dict = self.patches[idx]
        tr['channels'] = self.bands
        # vidid = tr['vidid']
        gids : list[int] = tr['gids']

        im1_id = gids[0]
        img_obj1 : dict = self.coco_dset.index.imgs[im1_id]
        video_obj = self.coco_dset.index.videos[img_obj1['video_id']]

        vid_width = video_obj['width']
        vid_height = video_obj['height']

        # Choose an offset "target" such that it is
        # (1) in the same image as the main image
        # (2) has a different spatial location
        # (3) is in a valid region of the image
        space_box = kwimage.Boxes.from_slice(tr['space_slice'])
        sh_space_box = space_box.to_shapely()[0]
        img_width = space_box.width.ravel()[0]
        img_height = space_box.height.ravel()[0]

        # Get the valid polygon for this coco image in video space
        # TODO: add API for this in CocoImage
        # CocoImage.valid_region(space='video') will be in kwcoco 0.2.26
        valid_coco_poly = img_obj1.get('valid_region', None)
        if valid_coco_poly is None:
            sh_valid_poly = None
        else:
            warp_vid_from_img = kwimage.Affine.coerce(img_obj1['warp_img_to_vid'])
            kw_poly_img = kwimage.MultiPolygon.coerce(valid_coco_poly)
            if kw_poly_img is None:
                sh_valid_poly = None
            else:
                valid_coco_poly = None
                sh_valid_poly = kw_poly_img.warp(warp_vid_from_img).to_shapely()  # shapely.geometry.Polygon

        # Sample valid offset boxes until the conditions are met
        rng = kwarray.ensure_rng(None)
        offset_box = None
        attempts = 0
        while offset_box is None:
            attempts += 1
            offset_box = kwimage.Boxes([[0, 0, img_width, img_height]], 'ltrb')
            offset_x = rng.randint(0, max(vid_width - img_width, 1))
            offset_y = rng.randint(0, max(vid_height - img_height, 1))
            offset_box = offset_box.translate((offset_x, offset_y))
            if attempts > 10:
                # Give up
                break
            sh_box = offset_box.to_shapely()[0]
            orig_overlap = sh_space_box.intersection(sh_box).area / sh_space_box.area
            if orig_overlap > 0.001:
                offset_box = None
            if sh_valid_poly is not None:
                valid_frac = sh_valid_poly.intersection(sh_box).area / sh_box.area
                if valid_frac < 0.5:
                    offset_box = None

        # Create a new target for the offset region
        offset_tr = tr.copy()
        offset_tr['channels'] = self.bands
        offset_tr['space_slice'] = offset_box.astype(int).to_slices()[0]

        import warnings
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', 'empty slice')
            warnings.filterwarnings('ignore', category=RuntimeWarning)

            if self.segmentation:
                sample = self.sampler.load_sample(tr, with_annots='segmentation', nodata='float')
                det_list = sample['annots']['frame_dets']
                segmentation_masks = []
                for det in det_list:
                    frame_mask = np.full([self.patch_size, self.patch_size], dtype=np.int32, fill_value=0)
                    ann_polys = det.data['segmentations'].to_polygon_list()
                    ann_aids = det.data['aids']
                    ann_cids = det.data['cids']
                    for poly, aid, cid in zip(ann_polys, ann_aids, ann_cids):
                        if cid in self.positive_indices:
                            if self.bas:
                                poly.fill(frame_mask, value=1)
                            else:
                                cidx = self.sampler.classes.id_to_idx[cid]
                                poly.fill(frame_mask, value=cidx)
                        elif cid in self.ignore_indices:
                            poly.fill(frame_mask, value=-1)
                    segmentation_masks.append(frame_mask)
            else:
                sample = self.sampler.load_sample(tr, nodata='float')
            offset_sample = self.sampler.load_sample(offset_tr, nodata='float')

            images : np.ndarray = sample['im']
            offset_image = offset_sample['im'][0]

            invalid_mask = np.isnan(images[0])
            if np.all(invalid_mask):
                max_val = 1
            else:
                max_val = np.nanmax(images[0])
            augmented_image = self.transforms(image=images[0].copy() / max_val)['image'] * max_val

            image_dict = {}
            for k, image in enumerate(images):
                imstd = np.nanstd(image)
                if imstd != 0.:
                    image = (image - np.nanmean(image)) / imstd
                else:
                    image = np.zeros_like(image)
                image_dict[1 + k] = image
            offset_imstd = np.nanstd(offset_image)
            if offset_imstd != 0:
                offset_image = (offset_image - np.nanmean(offset_image)) / offset_imstd
            else:
                offset_image = np.zeros_like(offset_image)
            augmented_imgstd = augmented_image.std()
            if augmented_imgstd != 0:
                augmented_image = (augmented_image - np.nanmean(augmented_image)) / augmented_imgstd
            else:
                augmented_image = np.zeros_like(augmented_image)

            for key in image_dict:
                image_dict[key] = torch.tensor(image_dict[key]).permute(2, 0, 1)
            offset_image = torch.tensor(offset_image).permute(2, 0, 1)
            augmented_image = torch.tensor(augmented_image).permute(2, 0, 1)

            date_list = []
            for gid in gids:
                date = self.coco_dset.index.imgs[gid]['date_captured']
                date_list.append((int(date[:4]), int(date[5:7])))
            normalized_date = torch.tensor([date_[0] - 2018 + date_[1] / 12 for date_ in date_list])
            out = dict()

            for m in range(self.num_images):
                out['image{}'.format(1 + m)] = image_dict[1 + m].float()

            out['offset_image1'] = offset_image.float().contiguous()
            out['augmented_image1'] = augmented_image.float().contiguous()
            out['normalized_date'] = normalized_date.float().contiguous()
            out['time_sort_label'] = float(normalized_date[0] < normalized_date[1])
            out['img1_id'] = gids[0]
            # img1_info = self.coco_dset.index.imgs[gids[0]]
            # out['img1_info'] = img1_info
            # out['tr'] = ItemContainer(tr, stack=False)
            if self.segmentation:
                for k in range(self.num_images):
                    out['segmentation{}'.format(1 + k)] = torch.tensor(segmentation_masks[k]).contiguous()
        return out


class kwcoco_dataset(Dataset):
    S2_l2a_channel_names = [
        'B02.tif', 'B01.tif', 'B03.tif', 'B04.tif', 'B05.tif', 'B06.tif', 'B07.tif', 'B08.tif', 'B09.tif', 'B11.tif', 'B12.tif', 'B8A.tif'
    ]
    S2_channel_names = [
        'coastal', 'blue', 'green', 'red', 'B05', 'B06', 'B07', 'nir', 'B09', 'cirrus', 'swir16', 'swir22', 'B8A'
    ]
    L8_channel_names = [
        'coastal', 'lwir11', 'lwir12', 'blue', 'green', 'red', 'nir', 'swir16', 'swir22', 'pan', 'cirrus'
    ]

    def __init__(self, coco_dset, sensor=['S2', 'L8'], bands=['shared'], patch_size=64, change_labels=False, display=False, mode='train'):
        # initialize dataset
        self.dset = kwcoco.CocoDataset.coerce(coco_dset)
        self.images = self.dset.images()
        self.change_labels = change_labels
        self.annotations = self.dset.annots
        self.display = display

        # handle if there are multiple sensors
        if type(sensor) is not list:
            sensor = [sensor]
        if type(bands) is not list:
            bands = [bands]
        print('Using sensors:', sensor)
        print('Using bands:', bands)

        if 'sensor_coarse' in self.images._id_to_obj[self.images._ids[0]].keys():
            # get available sensors
            # avail_sensors = set(self.images.lookup('sensor_coarse'))
            # filter images by desired sensor
            self.images = self.images.compress([x in sensor for x in self.images.lookup('sensor_coarse')])
            assert self.images
        # else:
        #     avail_sensors = None

        # get image ids and videos
        self.dset_ids = self.images.gids
        self.videos = [x['id'] for x in self.dset.videos().objs]

        # get all available channels
        all_channels = [ aux.get('channels', None) for aux in self.dset.index.imgs[self.images._ids[0]].get('auxiliary', []) ]
        if 'r|g|b' in all_channels:
            all_channels.remove('r|g|b')
        self.channels = []
        # no channels selected
        if len(bands) < 1:
            raise ValueError(f'bands must be specified. Options are {", ".join(all_channels)}, shared, or all')
        # all channels selected
        elif len(bands) == 1:
            if bands[0].lower() == 'all':
                self.channels = all_channels
            elif bands[0].lower() == 'shared':
                self.channels = ['red', 'green', 'blue', 'nir', 'swir16', 'swir22']
            elif bands[0] == 'r|g|b':
                self.channels.append('r|g|b')
            # subset of channels selected
        else:
            for band in bands:
                if band in all_channels:
                    self.channels.append(band)
                else:
                    raise ValueError(f'\'{band}\' not recognized as an available band. Options are {", ".join(all_channels)}, or all')

        # define augmentations
        self.transforms = A.Compose([
                A.RandomCrop(height=patch_size, width=patch_size),
                A.RandomRotate90(p=1)
                ],
                additional_targets={'image2': 'image', 'seg1': 'mask', 'seg2': 'mask'})

        self.transforms2 = A.Compose([
                A.RandomCrop(height=patch_size, width=patch_size),
                A.RandomRotate90(p=1),
                A.HorizontalFlip(p=1)],
            additional_targets={'image2': 'image'})

        self.transforms3 = A.Compose([A.OneOf([
                        A.MotionBlur(p=1),
                        A.Blur(blur_limit=3, p=1),
                    ], p=0.9),
                    A.GaussNoise(var_limit=.002),
                    A.RandomBrightnessContrast(brightness_limit=.3, contrast_limit=.3, brightness_by_max=False, always_apply=True)
                ])
        self.num_channels = len(self.channels)
        self.mode = mode

    def __len__(self,):
        return len(self.dset_ids)

    def get_img(self, idx, device=None):
        image_id = self.dset_ids[idx]
        image_info = self.dset.index.imgs[image_id]
        image = self.dset.delayed_load(image_id, channels=self.channels, space='video').finalize(no_data='float').astype(np.float32)
        image = torch.tensor(image)
        if device:
            image = image.to(device)
        # normalize
        if image.std() != 0.0:
            image = (image - image.mean()) / image.std()
        image = image.permute(2, 0, 1).unsqueeze(0)
        return image_id, image_info, image

    def __getitem__(self, idx):
        # get image1 id and the video it is associated with
        img1_id = self.dset_ids[idx]
        if self.mode == 'test':
            img1_info = self.dset.index.imgs[img1_id]
        else:
            img1_info = torch.tensor([])

        img_obj1 : dict = self.dset.index.imgs[img1_id]
        video : int = img_obj1['video_id']

        # randomly select image2 id from the same video (could be before or after image1)
        # make sure image2 is not image1 and image2 is in the set of filtered images by desired sensor
        img2_id = img1_id
        while img2_id == img1_id or img2_id not in self.dset_ids:
            img2_id = random.choice(self.dset.index.vidid_to_gids[video])

        img_obj2 : dict = self.dset.index.imgs[img2_id]

        # get frame indices for each image (used to determine which image was captured first)
        frame_index1 = img_obj1['frame_index']
        frame_index2 = img_obj2['frame_index']
        # get sensors
        im1_sensor = img_obj1['sensor_coarse']
        im2_sensor = img_obj2['sensor_coarse']

        # load images
        img1 = self.dset.delayed_load(img1_id, channels=self.channels, space='video').finalize(no_data='float').astype(np.float32)
        img2 = self.dset.delayed_load(img2_id, channels=self.channels, space='video').finalize(no_data='float').astype(np.float32)

        if not self.change_labels:
            # transformations
            max1 = img1.max()
            max2 = img2.max()
            transformed = self.transforms(image=img1.copy() / max1, image2=img2.copy() / max2)
            transformed2 = self.transforms2(image=img1.copy() / max1)
            img1 = transformed['image'] * max1
            img2 = transformed['image2'] * max2

            if self.display:
                if self.num_channels == 3:
                    display_image1 = img1
                    display_image2 = img2
                else:

                    display_image1 = img1[:, :, [3, 2, 1]]
                    display_image2 = img2[:, :, [3, 2, 1]]
                display_image1 = (2 + display_image1) / 6
                display_image2 = (2 + display_image2) / 6
                display_image1 = torch.tensor(display_image1).permute(2, 0, 1)
                display_image2 = torch.tensor(display_image2).permute(2, 0, 1)
            else:
                display_image1 = torch.tensor([])
                display_image2 = torch.tensor([])

            img3 = transformed2['image'] * max1
            img4 = self.transforms3(image=img1.copy() / img1.max())['image'] * img1.max()
            # convert to tensors

            img1std = np.nanstd(img1)
            if img1std != 0.:
                img1 = (img1 - np.nanmean(img1)) / img1std
            else:
                img1 = np.zeros_like(img1)
            img2std = np.nanstd(img2)
            if img2std != 0.:
                img2 = (img2 - np.nanmean(img2)) / img2std
            else:
                img2 = np.zeros_like(img2)
            img3std = np.nanstd(img3)
            if img3std != 0.:
                img3 = (img3 - np.nanmean(img3)) / img3std
            else:
                img3 = np.zeros_like(img3)
            img4std = np.nanstd(img4)
            if img4std != 0.:
                img4 = (img4 - np.nanmean(img4)) / img4std
            else:
                img4 = np.zeros_like(img4)

            img1 = np.nan_to_num(img1)
            img2 = np.nan_to_num(img2)
            img3 = np.nan_to_num(img3)
            img4 = np.nan_to_num(img4)

            img1 = torch.tensor(img1).permute(2, 0, 1)
            img2 = torch.tensor(img2).permute(2, 0, 1)
            img3 = torch.tensor(img3).permute(2, 0, 1)
            img4 = torch.tensor(img4).permute(2, 0, 1)

            return {
                'image1': img1.float(),
                'image2': img2.float(),
                'offset_image1': img3.float(),
                'augmented_image1': img4.float(),
                'time_sort_label': float(frame_index1 < frame_index2),
                'date1': (frame_index1, frame_index1),
                'date2': (frame_index2, frame_index2),
                'display_image1': display_image1,
                'display_image2': display_image2,
                'sensor_image1': im1_sensor,
                'sensor_image2': im2_sensor,
                'img1_id': img1_id,
                'img1_info': img1_info
            }

        else:
            if frame_index1 > frame_index2:
                img1, img2 = img2, img1
                img1_id, img2_id = img2_id, img1_id
                img_obj1, img_obj2 = img_obj2, img_obj1

            aids1 = self.dset.index.gid_to_aids[img1_id]
            aids2 = self.dset.index.gid_to_aids[img2_id]
            dets1 = kwimage.Detections.from_coco_annots(
                self.dset.annots(aids1).objs, dset=self.dset)
            dets2 = kwimage.Detections.from_coco_annots(
                self.dset.annots(aids2).objs, dset=self.dset)

            vid_from_img1 = kwimage.Affine.coerce(img_obj1['warp_img_to_vid'])
            vid_from_img2 = kwimage.Affine.coerce(img_obj2['warp_img_to_vid'])

            dets1 = dets1.warp(vid_from_img1)
            dets2 = dets2.warp(vid_from_img2)

            # bbox = dets.data['boxes'].data
            segmentation1 = dets1.data['segmentations'].data
            segmentation2 = dets2.data['segmentations'].data
            category_id1 = [dets1.classes.idx_to_id[cidx] for cidx in dets1.data['class_idxs']]
            category_id2 = [dets2.classes.idx_to_id[cidx] for cidx in dets2.data['class_idxs']]

            img_dims = (img1.shape[0], img1.shape[1])

            combined1 = []

            for sseg, cid in zip(segmentation1, category_id1):
                assert cid > 0
                np_mask = sseg.to_mask(dims=img_dims).data.astype(float) * cid
                mask1 = torch.from_numpy(np_mask)
                combined1.append(mask1.unsqueeze(0))

            if combined1:
                overall_mask1 = torch.max(torch.cat(combined1, dim=0), dim=0)[0]
            else:
                overall_mask1 = np.zeros_like(img1[:, :, 0])

            combined2 = []

            for sseg, cid in zip(segmentation2, category_id2):
                assert cid > 0
                np_mask = sseg.to_mask(dims=img_dims).data.astype(float) * cid
                mask2 = torch.from_numpy(np_mask)
                combined2.append(mask2.unsqueeze(0))

            if combined2:
                overall_mask2 = torch.max(torch.cat(combined2, dim=0), dim=0)[0]
            else:
                overall_mask2 = np.zeros_like(img2[:, :, 0])

            transformed = self.transforms(image=img1, image2=img2, seg1=np.array(overall_mask1), seg2=np.array(overall_mask2))
            img1 = transformed['image']
            img2 = transformed['image2']
            segmentation1 = transformed['seg1']
            segmentation2 = transformed['seg2']

            img1 = torch.tensor(img1).permute(2, 0, 1)
            img2 = torch.tensor(img2).permute(2, 0, 1)

            if self.display:
                if self.num_channels == 3:
                    display_image1 = img1
                    display_image2 = img2
                else:
                    display_image1 = img1[[3, 2, 1], :, :]
                    display_image2 = img2[[3, 2, 1], :, :]
                display_image1 = (2 + display_image1) / 6
                display_image2 = (2 + display_image2) / 6
            else:
                display_image1 = torch.tensor([])
                display_image2 = torch.tensor([])

            segmentation1 = torch.tensor(segmentation1)
            segmentation2 = torch.tensor(segmentation2)
            change_map = torch.clamp(segmentation2 - segmentation1, 0, 1)

            img1 = np.nan_to_num(img1)
            img2 = np.nan_to_num(img2)

            return {
                'image1': img1.float(),
                'image2': img2.float(),
                'segmentation1': segmentation1,
                'segmentation2': segmentation2,
                # 'categories1': category_id1,
                # 'categories2': category_id2,
                'segmentation1': segmentation1,
                'segmentation2': segmentation2,
                'change_map': change_map,
                'display_image1': display_image1,
                'display_image2': display_image2,
                'sensor_image1': im1_sensor,
                'sensor_image2': im2_sensor,
                'img1_id': img1_id,
                'img1_info': img1_info
            }


class Onera(Dataset):
    """Change Detection dataset class, used for both training and test data."""

    def __init__(self,
                 data_folder='/localdisk0/SCRATCH/watch/onera/',
                 train=True,
                 patch_size=96,
                 num_channels=13,
                 multihead=False,
                 display=False,
                 class_weight=1,
                 randomize_order=False):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.display = display
        self.randomize_order = randomize_order
        # basics
        self.path = data_folder
        self.num_channels = num_channels
        self.img_subpath = 'images/'
        self.train = train

        if self.train:
            fname = 'train.txt'
            self.label_subpath = 'train_labels/'
            self.transforms1 = A.Compose([
                                                A.HorizontalFlip(p=0.5),
                                                A.RandomRotate90(p=.99),
                                                A.RandomCrop(patch_size, patch_size)
                                            ],
                                            additional_targets={'image1': 'image', 'image2': 'image', 'mask1': 'mask'})
        else:
            fname = 'test.txt'
            self.label_subpath = 'test_labels/'
            self.transforms1 = A.Compose([A.NoOp()],
                                            additional_targets={'image1': 'image', 'image2': 'image', 'mask1': 'mask'})

        self.multihead = multihead
        if multihead:
            if train:
                self.transforms2 = A.Compose([
                                                A.RandomCrop(height=patch_size, width=patch_size),
                                                A.RandomRotate90(p=0.5),
                                                A.HorizontalFlip(p=.75),
                                                A.VerticalFlip(p=.75)
                                                ],
                                                additional_targets={'image2': 'image'})
            else:
                self.transforms2 = A.Compose([
                                                # A.RandomCrop(height=patch_size, width=patch_size),
                                                A.HorizontalFlip(p=.75),
                                                A.VerticalFlip(p=.75)
                                                ],
                                                additional_targets={'image2': 'image'})
            self.transforms3 = A.Compose([
                    A.Blur(p=.3),
                    A.RandomBrightnessContrast(always_apply=True)
            ])

        self.to_tensor = transforms.ToTensor()

        self.loc_names = read_csv(self.path + self.img_subpath + fname).columns

        self.num_channels = num_channels

    def __len__(self):
        if self.train:
            return 2560
        else:
            return 10

    def __getitem__(self, idx):
        if self.train:
            idx = idx % 14

        loc_name = self.loc_names[idx]
        img1, img2, cm = read_sentinel_img_trio(self.path + self.img_subpath + loc_name, self.path + self.label_subpath + loc_name, self.num_channels, True)

        img1 = (img1 - img1.mean()) / img1.std()
        img2 = (img2 - img2.mean()) / img2.std()

        cm = 1 * np.array(cm)

        transformed = self.transforms1(image=img1, image2=img2, mask=cm)
        img1 = transformed['image']
        img2 = transformed['image2']
        change_map = transformed['mask']

        if self.multihead:
            transformed2 = self.transforms2(image=img1)
            img3 = transformed2['image']
            img3 = self.to_tensor(img3)

            transformed3 = self.transforms3(image=img1)
            img4 = transformed3['image']
            img4 = self.to_tensor(img4)

        img1 = self.to_tensor(img1)
        img2 = self.to_tensor(img2)

        date1 = (0, 0)
        date2 = (1, 1)

        if self.randomize_order:
            label = random.choice([0, 1])
        else:
            label = 1

        if not label:
            img1, img2 = img2, img1
            date1, date2 = date2, date1

        if self.display:
            if self.num_channels == 3:
                display_image1 = img1
                display_image2 = img2
            elif self.num_channels == 13:
                display_image1 = img1[[3, 2, 1], : , :]
                display_image2 = img2[[3, 2, 1], : , :]
            else:
                display_image1 = img1[[2, 1, 0], :, :]
                display_image2 = img2[[2, 1, 0], :, :]
            display_image1 = (2 + display_image1) / 3
            display_image2 = (2 + display_image2) / 3
        else:
            display_image1 = torch.tensor([])
            display_image2 = torch.tensor([])

        if not self.multihead:
            return {'image1': img1.float().contiguous(),
                    'image2': img2.float().contiguous(),
                    'change_map': change_map,
                    'label': label,
                    'date1': date1,
                    'date2': date2,
                    'display_image1': display_image1,
                    'display_image2': display_image2,
                    'time_steps': torch.tensor([0, 1])}

        else:
            return {'image1': img1.float(),
                    'image2': img2.float(),
                    'offset_image': img3.float(),
                    'augmented_image': img4.float(),
                    'change_map': change_map,
                    'label': label,
                    'date1': date1,
                    'date2': date2,
                    'display_image1': display_image1.contiguous(),
                    'display_image2': display_image2.contiguous(),
                    'time_steps': torch.tensor([0, 1])}


class SpaceNet7(Dataset):
    normalize_params = [[0.16198677, 0.22665408, 0.1745371], [0.06108317, 0.06515977, 0.04128775]]

    def __init__(self,
                    patch_size=[128, 128],
                    splits='satellite_sort/data/spacenet/splits_unmasked/',  # ### unmasked images
                    train=True,
                    normalize=True,
                    yearly=True,
                    display=False):

        self.display = display
        self.train = train
        self.yearly = yearly
        self.crop = A.Compose([A.RandomCrop(height=patch_size[0], width=patch_size[1])], additional_targets={'image2': 'image', 'mask1': 'mask', 'mask2': 'mask'})
        if self.train:
            self.rotate = A.Compose([A.RandomRotate90()], additional_targets={'image2': 'image',
                                                                               'mask1': 'mask',
                                                                               'mask2': 'mask'})
            self.transforms = A.Compose([
                A.GaussianBlur(),
                A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
                                        ],
                                additional_targets={'image2': 'image'}
            )
        else:
            self.transforms = None

        if train:
            self.images = []
            with open(os.path.join(splits, 'spacenet7_train_images.txt'), 'r') as file:
                for row in file:
                    self.images.append(row.strip('\n'))

        else:
            self.images = []
            with open(os.path.join(splits, 'spacenet7_val_images.txt'), 'r') as file:
                for row in file:
                    self.images.append(row.strip('\n'))

        self.length = len(self.images)
        self.normalize = normalize
        self.normalization = A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), max_pixel_value=255.0)

        self.time_adjust = [-1, 1]

    def __len__(self):
        return (self.length)

    def __getitem__(self, idx):

        im1_path = self.images[idx]
        date1 = (int(im1_path[-47:-43]), int(im1_path[-42:-40]))

        if self.yearly:
            ###Choose image2 as close to one year spread apart as possible
            time_adjust = random.choice(self.time_adjust)
            im2_path = im1_path[:-47] + str(int(im1_path[-47:-43]) + time_adjust) + im1_path[-43:]

            if not os.path.exists(im2_path):
                im2_path = im1_path[:-47] + str(int(im1_path[-47:-43]) - time_adjust) + im1_path[-43:]

                if not os.path.exists(im2_path):
                    im2_path = im1_path[:-47] + str(int(im1_path[-47:-43]) - time_adjust) + im1_path[-43]

                    if not os.path.exists(im2_path):
                        im2_path = im1_path[:-47] + str(int(im1_path[-47:-43]) - time_adjust) + im1_path[-43:]

                        x = 0
                        while not os.path.exists(im2_path):
                            x += 1
                            if not os.path.exists(im2_path):
                                im2_path = im1_path[:-47] + str(int(im1_path[-47:-43]) - time_adjust) + im1_path[-43:-41] + str(int(im1_path[-41]) + x * time_adjust) + im1_path[-40:]
                                if not os.path.exists(im2_path):
                                    im2_path = im1_path[:-47] + str(int(im1_path[-47:-43]) - time_adjust) + im1_path[-43:-41] + str(int(im1_path[-41]) - x * time_adjust) + im1_path[-40:]

                                    if not os.path.exists(im2_path):
                                        im2_path = im1_path[:-47] + str(int(im1_path[-47:-43]) + time_adjust) + im1_path[-43:-41] + str(int(im1_path[-41]) + x * time_adjust) + im1_path[-40:]
                                        if not os.path.exists(im2_path):
                                            im2_path = im1_path[:-47] + str(int(im1_path[-47:-43]) + time_adjust) + im1_path[-43:-41] + str(int(im1_path[-41]) - x * time_adjust) + im1_path[-40:]

        else:
            im_directory, _ = os.path.split(self.images[idx])
            date1 = (int(self.images[idx][-47:-43]), int(self.images[idx][-42:-40]))
            date2 = date1
            while date2 == date1:
                idx2 = torch.randint(len(os.listdir(im_directory)), (1,))[0]
                im2_path = os.path.join(im_directory, sorted(os.listdir(im_directory))[idx2])

        date2 = (int(im2_path[-47:-43]), int(im2_path[-42:-40]))

        '''TO DO: Convert tiffile package commands to kwimage'''
        image = (kwimage.imread(im1_path)).astype("float32")
        image2 = (kwimage.imread(im2_path)).astype("float32")

        cloud_mask1 = image[:, :, 3]
        cloud_mask2 = image2[:, :, 3]

        image = image[:, :, :3]
        image2 = image2[:, :, :3]

        crop = self.crop(image=image, image2=image2, mask1=cloud_mask1, mask2=cloud_mask2)

        image = crop['image']
        image2 = crop['image2']
        cloud_mask1 = crop['mask1']
        cloud_mask2 = crop['mask2']

        if self.display:
            display_image1 = image.astype('uint8')
            display_image2 = image2.astype('uint8')
        else:
            display_image1 = torch.tensor([])
            display_image2 = torch.tensor([])

        if self.normalize:
            image = self.normalization(image=image)['image']
            image2 = self.normalization(image=image2)['image']

        if self.transforms:
            transformed = self.transforms(image=image, image2=image2)
            image = transformed['image']
            image2 = transformed['image2']

            rotated = self.rotate(image=image, image2=image2)
            image = rotated['image']
            image2 = rotated['image2']

        item = {
            'image1': torch.tensor(image).permute(2, 0, 1).contiguous(),
            'image2': torch.tensor(image2).permute(2, 0, 1).contiguous(),
            'label': int(date1 < date2),
            'date1': date1,
            'date2': date2,
            'display_image1': display_image1.contiguous(),
            'display_image2': display_image2.contiguous(),
            'cloud_mask1': cloud_mask1.contiguous(),
            'cloud_mask2': cloud_mask2.contiguous(),
        }
        return item
