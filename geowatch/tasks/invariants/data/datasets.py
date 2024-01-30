import torch
import numpy as np
import kwcoco
import kwimage
import kwarray
import ndsampler
import ubelt as ub
import warnings
from geowatch.monkey import monkey_albumentations

monkey_albumentations.patch_albumentations_for_311()


class GriddedDataset(torch.utils.data.Dataset):
    """

    Example:
        >>> from geowatch.tasks.invariants.data.datasets import *  # NOQA
        >>> import geowatch
        >>> coco_dset = geowatch.coerce_kwcoco('geowatch-msi', dates=True, geodata=True)
        >>> keep_ids = [img.img['id'] for img in coco_dset.images().coco_images if 'B11' in img.channels]
        >>> coco_dset = coco_dset.subset(keep_ids)
        >>> self = GriddedDataset(coco_dset, include_debug_info=True, sensor=None, bands=['B11'], patch_size=32, input_space_scale='3GSD')
        >>> item = self[0]
        >>> item_summary = self.summarize_item(item)
        >>> import rich
        >>> rich.print('item_summary = {}'.format(ub.urepr(item_summary, nl=1, sort=0, align=':')))

    Example:
        >>> # xdoctest: +REQUIRES(env:DVC_DPATH)
        >>> from geowatch.tasks.invariants.data.datasets import *  # NOQA
        >>> import geowatch
        >>> dvc_dpath = geowatch.find_dvc_dpath()
        >>> coco_fpath = dvc_dpath / 'Drop2-Aligned-TA1-2022-02-15/data_nowv_vali.kwcoco.json'
        >>> import kwcoco
        >>> coco_dset = kwcoco.CocoDataset(coco_fpath)
        >>> self = GriddedDataset(coco_dset)
        >>> idx = 0
        >>> item = self[idx]
        >>> rgb1 = item['image1'][0:3].permute(1, 2, 0).numpy()[..., ::-1]
        >>> rgb2 = item['image2'][0:3].permute(1, 2, 0).numpy()[..., ::-1]
        >>> rgb3 = item['offset_image1'][0:3].permute(1, 2, 0).numpy()[..., ::-1]
        >>> rgb4 = item['augmented_image1'][0:3].permute(1, 2, 0).numpy()[..., ::-1]
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
        >>> from geowatch.tasks.invariants.data.datasets import *  # NOQA
        >>> import geowatch
        >>> dvc_dpath = geowatch.find_dvc_dpath(tags='phase2_data', hardware='ssd')
        >>> bundle_dpath = dvc_dpath / 'Drop6'
        >>> coco_fpath = bundle_dpath / 'data_vali_split1.kwcoco.zip'
        >>> import kwcoco
        >>> coco_dset = kwcoco.CocoDataset(coco_fpath)
        >>> self = GriddedDataset(
        >>>     coco_dset,
        >>>     window_space_scale='33GSD',
        >>>     input_space_scale='33GSD',
        >>>     output_space_scale='input',
        >>>     patch_size=64,
        >>>     include_debug_info=True,
        >>> )
        >>> dsize = (224, 224)
        >>> # dsize = None
        >>> # Draw multiple batch items
        >>> rows = []
        >>> max_idx = len(self) // 4 - 2
        >>> indexes = np.linspace(0, max_idx, 4).round().astype(int)
        >>> for idx in indexes:
        >>>     item = self[idx]
        >>>     row_canvas = self.draw_item(item, dsize)
        >>>     rows.append(row_canvas)
        >>> item_summary = self.summarize_item(item)
        >>> import rich
        >>> rich.print('item_summary = {}'.format(ub.urepr(item_summary, nl=1, sort=0, align=':')))
        >>> canvas = kwimage.stack_images(rows, axis=0, pad=10)
        >>> # xdoctest: +REQUIRES(--show)
        >>> import kwplot
        >>> kwplot.autompl()
        >>> kwplot.imshow(canvas)

    Ignore:
        >>> # xdoctest: +SKIP
        >>> from geowatch.tasks.invariants.data.datasets import *  # NOQA
        >>> import geowatch
        >>> import kwcoco
        >>> import rich
        >>> dvc_dpath = geowatch.find_dvc_dpath(tags='phase2_data', hardware='ssd')
        >>> # TEST DROP6
        >>> coco_drop6 = kwcoco.CocoDataset(dvc_dpath / 'Drop6/data_vali_split6.kwcoco.zip')
        >>> data_drop6 = GriddedDataset(coco_drop6, window_space_scale='30GSD')
        >>> item = data_drop6[0]
        >>> item_summary6 = data_drop6.summarize_item(item)
        >>> rich.print('item_summary6 = {}'.format(ub.urepr(item_summary6, nl=1, sort=0, align=':')))
        >>> # TEST DROP7
        >>> coco_drop7 = kwcoco.CocoDataset(dvc_dpath / 'Drop7-MedianNoWinter10GSD/data_vali_EI2LMSC_split6.kwcoco.zip')
        >>> data_drop7 = GriddedDataset(coco_drop7, window_space_scale='30GSD')
        >>> item = data_drop7[0]
        >>> item_summary7 = data_drop7.summarize_item(item)
        >>> rich.print('item_summary7 = {}'.format(ub.urepr(item_summary7, nl=1, sort=0, align=':')))
        >>> ###
        >>> rich.print('item_summary6 = {}'.format(ub.urepr(item_summary6, nl=1, sort=0, align=':')))
        >>> rich.print('item_summary7 = {}'.format(ub.urepr(item_summary7, nl=1, sort=0, align=':')))

        import geowatch
        import kwcoco
        import rich
        dvc_dpath = geowatch.find_dvc_dpath(tags='phase2_data', hardware='ssd')
        coco_drop7 = kwcoco.CocoDataset(dvc_dpath / 'Drop7-MedianNoWinter10GSD/data_vali_EI2LMSC_split6.kwcoco.zip')
        self = GriddedDataset(coco_drop7, window_space_scale='10GSD')
        sampler = self.sampler

        target = {
            'main_idx': 1,
            'video_id': 1,
            'vidid': 1,
            'gids': [54, 62],
            'main_gid': 62,
            'frame_index': 18,
            'frame_indexes': [5, 18],
            'resampled': None,
            'label': None,
            'space_slice': (slice(0, 384, None), slice(0, 384, None)),
            'scale': 0.3333333333333333,
            'channels': 'red|green|blue|nir|swir16|swir22',
            '_input_gsd': 30.0,
            '_native_video_gsd': 10.0,
            'verbose_ndsample': True,
        }
        sample = sampler.load_sample(target, nodata='float', with_annots=False)
        print(sample['im'].shape)

        print(chr(10) * 10 + '----' + chr(10) * 10)

        target = {
            'main_idx': 1,
            'video_id': 1,
            'vidid': 1,
            'gids': [54, 62],
            'main_gid': 62,
            'frame_index': 18,
            'frame_indexes': [5, 18],
            'resampled': None,
            'label': None,
            'space_slice': (slice(0, 128, None), slice(0, 128, None)),
            'scale': 1.0,
            'channels': 'red|green|blue|nir|swir16|swir22',
            '_input_gsd': 10.0,
            '_native_video_gsd': 10.0,
            'verbose_ndsample': True,
        }
        sample = sampler.load_sample(target, nodata='float', with_annots=False)
        visible_thresh = 0.0
        kwargs = dict(nodata='float')
        print(sample['im'].shape)


    """
    # S2_l2a_channel_names = [
    #     'B02.tif', 'B01.tif', 'B03.tif', 'B04.tif', 'B05.tif', 'B06.tif', 'B07.tif', 'B08.tif', 'B09.tif', 'B11.tif', 'B12.tif', 'B8A.tif'
    # ]
    # S2_channel_names = [
    #     'coastal', 'blue', 'green', 'red', 'B05', 'B06', 'B07', 'nir', 'B09', 'cirrus', 'swir16', 'swir22', 'B8A'
    # ]
    # L8_channel_names = [
    #     'coastal', 'lwir11', 'lwir12', 'blue', 'green', 'red', 'nir', 'swir16', 'swir22', 'pan', 'cirrus'
    # ]

    def __init__(self, coco_dset, sensor=['S2', 'L8'], bands=['shared'],
                 segmentation=False, patch_size=128, num_images=2,
                 mode='train', patch_overlap=.25, bas=True, rng=None,
                 window_space_scale=None,
                 input_space_scale='window',
                 output_space_scale='input',
                 include_debug_info=False):
        import albumentations as A
        super().__init__()

        self.include_debug_info = include_debug_info

        if input_space_scale == 'window':
            input_space_scale = window_space_scale
        if output_space_scale == 'input':
            output_space_scale = input_space_scale

        # NOTE: Assumption: all videos have the same target gsd.
        # videos = coco_dset.videos()
        # video = None
        # if len(videos):
        #     video = videos.objs[0]
        # if video:
        #     # Compute scale if we are doing that
        #     # This should live somewhere else, but lets just get it hooked up
        #     vidspace_gsd = video.get('target_gsd', None)

        self.window_space_scale = window_space_scale
        self.input_space_scale = input_space_scale
        self.output_space_scale = output_space_scale

        assert self.output_space_scale == self.input_space_scale

        # initialize dataset
        print('load dataset')
        self.coco_dset: kwcoco.CocoDataset = kwcoco.CocoDataset.coerce(coco_dset)

        print('filter dataset')
        # Filter out worldview images (better to use subset than remove)
        images: kwcoco.coco_objects1d.Images = self.coco_dset.images()

        if sensor is not None:
            requested_sensors = set(sensor)
            flags = [s in requested_sensors for s in images.lookup('sensor_coarse')]
            valid_image_ids : list[int] = list(images.compress(flags))
            self.coco_dset = self.coco_dset.subset(valid_image_ids)

        self.images : kwcoco.coco_objects1d.Images = self.coco_dset.images()
        self.sampler = ndsampler.CocoSampler(self.coco_dset)

        if len(self.images) == 0:
            raise ValueError('No images were provided')

        print('make grid')
        from geowatch.tasks.fusion.datamodules import spacetime_grid_builder
        builder = spacetime_grid_builder.SpacetimeGridBuilder(
            self.coco_dset,
            time_dims=num_images,
            window_dims=[patch_size, patch_size],
            window_overlap=patch_overlap,
            time_sampling='hardish3', time_span='1y',
            use_annot_info=False,
            keepbound=True,
            # exclude_sensors=['WV'],
            use_centered_positives=False,
            # set_cover_algo='approx',
            set_cover_algo=None,
            use_cache=1,
            workers=0,
            window_space_scale=window_space_scale,
        )
        sample_grid = builder.build()
        vidid_to_new_samples = fixup_samples(coco_dset, sample_grid, num_images)

        # Sort the patches into an order where we can
        self.patches = []
        for vidid, samples in ub.ProgIter(vidid_to_new_samples.items(), desc='ordering samples'):
            # TODO: find the best test-time ordering of the samples. For now just do sequential
            samples = sorted(samples, key=lambda x: x['frame_index'])
            if mode != 'train':
                idx_to_final_gids, g = find_complete_image_indexes(samples)
                for idx, gids in idx_to_final_gids.items():
                    if gids and idx is not None:
                        samples[idx]['complete_gids'] = gids
            self.patches.extend(samples)

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
                self.bands.extend(bands)
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
        # TODO / FIXME:
        # I think recent changes broke training. Likely will need to add
        # something to randomly switch image order at train time.
        target : dict = self.patches[idx]

        target = self.update_target_properties(target)

        offset_tr = self.choose_offset_target(target)

        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', 'empty slice')
            warnings.filterwarnings('ignore', category=RuntimeWarning)

            if self.segmentation:
                sample = self.sampler.load_sample(target, with_annots='segmentation', nodata='float')
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
                print('target = {}'.format(ub.urepr(target, nl=1)))
                sample = self.sampler.load_sample(target, nodata='float', with_annots=False)
            offset_sample = self.sampler.load_sample(offset_tr, nodata='float', with_annots=False)

            images : np.ndarray = sample['im']
            offset_image = offset_sample['im'][0]

            invalid_mask = np.isnan(images[0])
            if np.all(invalid_mask):
                max_val = 1
            else:
                max_val = np.nanmax(images[0])

            _aug_input = images[0].copy() / max_val
            _aug_input = np.nan_to_num(_aug_input)
            _aug_output = self.transforms(image=_aug_input)['image']
            augmented_image = _aug_output * max_val

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

        gids : list[int] = target['gids']
        images = self.coco_dset.images(gids)
        date_list = []
        for img in images.objs:
            date = img['date_captured']
            date_list.append((int(date[:4]), int(date[5:7])))

        normalized_date = torch.tensor([date_[0] - 2018 + date_[1] / 12 for date_ in date_list])
        item = dict()

        for m in range(self.num_images):
            item['image{}'.format(1 + m)] = image_dict[1 + m].float()

        item['offset_image1'] = offset_image.float().contiguous()
        item['augmented_image1'] = augmented_image.float().contiguous()
        item['normalized_date'] = normalized_date.float().contiguous()
        item['time_sort_label'] = float(normalized_date[0] < normalized_date[1])
        item['img1_id'] = gids[0]

        vidspace_box = kwimage.Box.from_slice(target['space_slice'])

        if self.include_debug_info:
            item['sampled_input_gsd'] = target['_input_gsd']
            item['native_video_gsd'] = target['_native_video_gsd']
            item['date_list'] = date_list
            item['sensor_list'] = images.lookup('sensor_coarse', 'unknown')
            item['bands'] = self.bands
            item['vidspace_xywh'] = torch.from_numpy(vidspace_box.toformat('xywh').data.astype(np.int32))

        outspace_box = vidspace_box.scale(target['scale']).quantize().astype(np.int32)

        im1_id = gids[0]
        img_obj1 : dict = self.coco_dset.index.imgs[im1_id]
        video_obj = self.coco_dset.index.videos[img_obj1['video_id']]
        # The size of the canvas we will stitch into in video space and output space
        full_stitch_vidspace_box = kwimage.Box.coerce([0, 0, video_obj['width'], video_obj['height']], format='xywh')
        full_stitch_outspace_box = full_stitch_vidspace_box.scale(target['scale']).quantize().astype(np.int32)

        item['full_stitch_outspace_ltrb'] = torch.from_numpy(full_stitch_outspace_box.data)
        item['sample_outspace_ltrb'] = torch.from_numpy(outspace_box.data)
        item['scale_outspace_from_vid'] = target['scale']

        # img1_info = self.coco_dset.index.imgs[gids[0]]
        # item['img1_info'] = img1_info
        # item['target'] = ItemContainer(target, stack=False)
        if self.segmentation:
            for k in range(self.num_images):
                item['segmentation{}'.format(1 + k)] = torch.tensor(segmentation_masks[k]).contiguous()
        return item

    def update_target_properties(self, target):
        """
        Populate the target so it has the correct input scale and bands.
        """
        # Handle target scale
        from geowatch.tasks.fusion.datamodules import data_utils
        gids : list[int] = target['gids']
        im1_id = gids[0]
        img_obj1 : dict = self.coco_dset.index.imgs[im1_id]
        video_obj = self.coco_dset.index.videos[img_obj1['video_id']]
        vidspace_gsd = video_obj.get('target_gsd', None)
        resolved_input_scale = data_utils.resolve_scale_request(request=self.input_space_scale, data_gsd=vidspace_gsd)
        target['scale'] = resolved_input_scale['scale']
        target['channels'] = self.bands
        target['_input_gsd'] = resolved_input_scale['gsd']
        target['_native_video_gsd'] = resolved_input_scale['data_gsd']
        return target

    def choose_offset_target(self, target):
        gids : list[int] = target['gids']

        im1_id = gids[0]
        img_obj1 : dict = self.coco_dset.index.imgs[im1_id]
        video_obj = self.coco_dset.index.videos[img_obj1['video_id']]

        vid_width = video_obj['width']
        vid_height = video_obj['height']

        # Choose an offset "target" such that it is
        # (1) in the same image as the main image
        # (2) has a different spatial location
        # (3) is in a valid region of the image
        space_box = kwimage.Boxes.from_slice(target['space_slice'])
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
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', 'invalid value encountered in intersection')
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
        offset_tr = target.copy()
        offset_tr['channels'] = self.bands
        offset_tr['space_slice'] = offset_box.astype(int).to_slices()[0]
        return offset_tr

    def draw_item(self, item, dsize=(224, 224)):
        """
        Example:
            >>> from geowatch.tasks.invariants.data.datasets import *  # NOQA
            >>> from geowatch.demo import coerce_kwcoco
            >>> coco_dset = coerce_kwcoco('geowatch-msi', dates=True, geodata=True)
            >>> keep_ids = [img.img['id'] for img in coco_dset.images().coco_images if 'B11' in img.channels]
            >>> coco_dset = coco_dset.subset(keep_ids)
            >>> self = GriddedDataset(coco_dset, include_debug_info=True, sensor=None, bands=['B11'])
            >>> item = self[0]
            >>> item_summary = self.summarize_item(item)
            >>> print('item_summary = {}'.format(ub.urepr(item_summary, nl=1)))
            >>> canvas = self.draw_item(item)
            >>> # xdoctest: +REQUIRES(--show)
            >>> import kwplot
            >>> kwplot.autompl()
            >>> kwplot.imshow(canvas)
        """
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', 'All-NaN slice')
            rgb1 = item['image1'][0:3].permute(1, 2, 0).numpy()
            rgb2 = item['image2'][0:3].permute(1, 2, 0).numpy()
            rgb3 = item['offset_image1'][0:3].permute(1, 2, 0).numpy()
            rgb4 = item['augmented_image1'][0:3].permute(1, 2, 0).numpy()
            canvas1 = kwimage.fill_nans_with_checkers(kwimage.imresize(kwimage.normalize(rgb1), dsize=dsize)).clip(0, 1)
            canvas2 = kwimage.fill_nans_with_checkers(kwimage.imresize(kwimage.normalize(rgb2), dsize=dsize)).clip(0, 1)
            canvas3 = kwimage.fill_nans_with_checkers(kwimage.imresize(kwimage.normalize(rgb3), dsize=dsize)).clip(0, 1)
            canvas4 = kwimage.fill_nans_with_checkers(kwimage.imresize(kwimage.normalize(rgb4), dsize=dsize)).clip(0, 1)
            canvas1 = kwimage.draw_text_on_image(canvas1, 'image1', org=(1, 1), valign='top', color='white', border=2)
            canvas2 = kwimage.draw_text_on_image(canvas2, 'image2', org=(1, 1), valign='top', color='white', border=2)
            canvas3 = kwimage.draw_text_on_image(canvas3, 'offset_image1', org=(1, 1), valign='top', color='white', border=2)
            canvas4 = kwimage.draw_text_on_image(canvas4, 'augmented_image1', org=(1, 1), valign='top', color='white', border=2)
            row_canvas = kwimage.stack_images([canvas1, canvas2, canvas3, canvas4], axis=1, pad=3)
        return row_canvas

    def summarize_item(self, item):
        """
        Return debugging stats about the item

        Args:
            item (dict): an item returned by __getitem__

        Returns:
            dict : a summary of the item
        """
        item_summary = {}
        _item = item.copy()
        item_summary['image1.shape'] = _item.pop('image1').shape
        item_summary['image2.shape'] = _item.pop('image2').shape
        item_summary['offset_image1.shape'] = _item.pop('offset_image1').shape
        item_summary['augmented_image1.shape'] = _item.pop('augmented_image1').shape
        for k in list(_item.keys()):
            if 'segmentation' in k:
                item_summary[k + '.shape'] = _item.pop(k).shape
        item_summary.update(_item)
        return item_summary


def find_complete_image_indexes(samples, fast=True):
    """
    Args:
        samples (List[dict]):
            A list of target dictionaries from ndsampler that contains a key
            'gids' which maps to a list of image ids that the sample touches.

    Returns:
        Tuple: sample_to_complete_gids, g
            mapping from sample indexes to what images can be marked as done

            g is the graph used for debug purposes

    References:
        https://cs.stackexchange.com/questions/155186/algorithm-for-minimizing-the-number-of-resources-simultaneously-open-while-itera

    Example:
        >>> from geowatch.tasks.invariants.data.datasets import *  # NOQA
        >>> samples = [
        >>>     {'gids': [0, 3]},
        >>>     {'gids': [0, 3]},
        >>>     {'gids': [1, 2]},
        >>>     {'gids': [1, 4]},
        >>>     {'gids': [2, 5]},
        >>>     {'gids': [0, 5]},
        >>> ]
        >>> sample_to_complete_gids, graphs = find_complete_image_indexes(samples, fast=False)
        >>> sample_to_complete_gids2, _ = find_complete_image_indexes(samples, fast=True)
        >>> assert sample_to_complete_gids == sample_to_complete_gids2
        >>> from cmd_queue.util.util_networkx import write_network_text
        >>> write_network_text(graphs['node_ordered'])
        >>> print('sample_to_complete_gids = {}'.format(ub.urepr(sample_to_complete_gids, nl=1)))

    Example:
        >>> from geowatch.tasks.invariants.data.datasets import *  # NOQA
        >>> samples = [
        >>>     {'gids': [0, 1]},
        >>>     {'gids': [1, 2]},
        >>>     {'gids': [2, 3]},
        >>>     {'gids': [3, 4]},
        >>>     {'gids': [4, 5]},
        >>>     {'gids': [5, 6]},
        >>> ]
        >>> sample_to_complete_gids, graphs = find_complete_image_indexes(samples, fast=False)
        >>> sample_to_complete_gids2, _ = find_complete_image_indexes(samples, fast=True)
        >>> assert sample_to_complete_gids == sample_to_complete_gids2
        >>> from cmd_queue.util.util_networkx import write_network_text
        >>> write_network_text(graphs['node_ordered'])
        >>> print('sample_to_complete_gids = {}'.format(ub.urepr(sample_to_complete_gids, nl=1)))

    Ignore:
        import kwplot
        kwplot.autompl()
        from graphid import util
        util.show_nx(graphs['constraint'])
        util.show_nx(graphs['node_ordered'])

        # util.show_nx(graphs['touchable_graph'])
        # util.show_nx(graphs['untouchable_graph'])
    """
    # Create a graph describing how indexes cover frames so we can
    # know when we are finally done with a particular image in
    # predict mode.
    # We connect an edge from each sample index to the images it
    # needs. We also connect each sample index to the next index.
    # At any sample index, if there is not a path to a particular
    # image, then it is done.
    import networkx as nx
    graphs = {}

    if fast:
        # Faster variant of the metric, just mark the last sample index we saw
        # the image.
        gid_to_last_idx = {}
        sample_to_complete_gids = {}
        for sample_idx, sample in enumerate(samples):
            sample_to_complete_gids[sample_idx] = []
            for gid in sample['gids']:
                gid_to_last_idx[gid] = sample_idx
        sample_to_complete_gids[None] = []
        for gid, idx in gid_to_last_idx.items():
            sample_idx = idx + 1
            if sample_idx == len(samples):
                sample_idx = None
            sample_to_complete_gids[sample_idx].append(gid)
        sample_to_complete_gids = ub.udict(sample_to_complete_gids).map_values(sorted)
        return sample_to_complete_gids, graphs

    SAMPLE = 'sample'
    GID = 'gid'

    # Build the graph where each sample points to the images it uses.
    constraint_graph = graphs['constraint'] = nx.DiGraph()
    for sample_index, sample in enumerate(samples):
        sample_node = (SAMPLE, sample_index)
        for gid in sample['gids']:
            image_node = (GID, gid)
            constraint_graph.add_edge(sample_node, image_node)

    image_nodes = {n for n in constraint_graph.nodes if n[0] == GID}
    sample_nodes = {n for n in constraint_graph.nodes if n[0] == SAMPLE}
    sample_nodes = ub.oset(sorted(sample_nodes))

    # Add in the baseline node ordering
    node_ordered = graphs['node_ordered'] = constraint_graph.copy()
    for s1, s2 in ub.iter_window(sorted(sample_nodes), 2):
        node_ordered.add_edge(s1, s2)

    # Make each sample node point to all of the images that cannot be unloaded
    # touchable_graph = nx.transitive_closure_dag(node_ordered)
    # Find the transative closure edges between samples and remove them
    # tc_sample_edges = set(
    #     edge for edge in touchable_graph.edges if (edge[0][0] == SAMPLE and edge[1][0] == SAMPLE))
    # rm_edges = tc_sample_edges - set(node_ordered.edges)
    # touchable_graph.remove_edges_from(rm_edges)
    # graphs['touchable_graph'] = touchable_graph
    # sample_to_touchable_images = {}
    # for sample_node in sample_nodes:
    #     touchable_images = set(touchable_graph.adj[sample_node]) - sample_nodes
    #     sample_to_touchable_images[sample_node] = touchable_images

    # Seems faster to just find shortest paths
    sample_to_touchable_images = {}
    for sample_node in sample_nodes:
        node_to_path = nx.single_source_shortest_path(node_ordered, sample_node)
        touchable_images = {k for k in node_to_path.keys() if k[0] == GID}
        # assert (set(touchable_graph.adj[sample_node]) - sample_nodes) == touchable_images
        sample_to_touchable_images[sample_node] = touchable_images

    sample_to_untouchable_images = {}
    for source, touchable in sample_to_touchable_images.items():
        untouchable = image_nodes - touchable
        sample_to_untouchable_images[source] = untouchable

    # untouchable_graph = nx.DiGraph()
    # untouchable_graph.add_nodes_from(constraint_graph)
    # untouchable_graph.add_edges_from((s, g) for s, gs in sample_to_untouchable_images.items() for g in gs)
    # graphs['untouchable_graph'] = untouchable_graph

    # Now mark the first sample we see an untouchable image so at
    # that point in interation the predictor knows it can mark it
    # as complete and finalize it.
    sample_to_complete_nodes = {}
    completed_images = set()
    for source, untouchable in sample_to_untouchable_images.items():
        marker = untouchable - completed_images
        completed_images.update(untouchable)
        sample_to_complete_nodes[source] = marker

    # And these are the images that have to wait all the way
    # until the end to complete
    final_nodes = image_nodes - completed_images
    sample_to_complete_nodes[(SAMPLE, None)] = final_nodes
    sample_to_complete_gids = {
        k[1]: [v[1] for v in vs]
        for k, vs in sample_to_complete_nodes.items()}
    sample_to_complete_gids = ub.udict(sample_to_complete_gids).map_values(sorted)
    return sample_to_complete_gids, graphs


class HashableBox(kwimage.Box):
    def to_tuple(box):
        return tuple([box.format] + box.data.tolist())


def fixup_samples(coco_dset, sample_grid, time_dims):
    """
    Takes the output of the sample grid and ensures we get at least one sample
    on each frame. Getting this to happen is something the time sampler should
    take care of, but for now we just hack it in.
    """
    import copy
    # all_samples = sample_grid['targets']

    all_samples = copy.deepcopy(sample_grid['targets'])

    # import xdev
    # xdev.embed()
    for target in all_samples:
        target['vidid'] = target['video_id']  # hack
        time_sampler = sample_grid['vidid_to_time_sampler'][target['video_id']]

        gids = target['gids']
        images = coco_dset.images(gids)
        frame_indexes = images.lookup('frame_index')

        # The second gid is always the main gid in our case.
        # FIXME: is this true? Only do this in test mode?
        main_gid = gids[1]
        main_frame_index = frame_indexes[1]
        time_sampler.gid_to_index = ub.udict(enumerate(time_sampler.video_gids)).invert()
        main_time_sampler_index = time_sampler.gid_to_index[main_gid]

        target['main_gid'] = main_gid
        target['frame_index'] = main_frame_index
        target['main_idx'] = main_time_sampler_index
        target['frame_indexes'] = frame_indexes

    WORKAROUND_NON_UNIQUE_IMAGE_IDS = 1
    if WORKAROUND_NON_UNIQUE_IMAGE_IDS:
        workaround_samples = []
        for tr in all_samples:
            unique_gids = list(ub.unique(tr['gids']))
            if len(unique_gids) == time_dims:
                tr['gids'] = unique_gids
                workaround_samples.append(tr)
        all_samples = workaround_samples

    # Postprocess the grid we get out of the temporal sampler to make it a
    # little nicer for this problem.
    vidid_to_new_samples = {}
    vidid_to_samples = ub.group_items(all_samples, lambda x: x['vidid'])
    for vidid, vid_samples in vidid_to_samples.items():

        time_sampler = sample_grid['vidid_to_time_sampler'][vidid]

        # Make everything valid for this hack
        time_sampler.affinity = np.maximum(
            time_sampler.affinity, np.finfo(np.float32).eps)

        # Only use the image id we filtered the time sampler to.
        vid_images = coco_dset.images(time_sampler.video_gids)
        frame_idxs = vid_images.lookup('frame_index')
        assert sorted(frame_idxs) == frame_idxs

        idx_to_sample = ub.group_items(vid_samples, lambda x: x['main_idx'])
        missing_frame_idxs = set(time_sampler.indexes) - set(idx_to_sample)

        # Get spatial information about the samples
        from collections import Counter
        spatial_slices = Counter()

        # frame_index_to_timepairs = ub.ddict(Counter)
        for samples in idx_to_sample.values():
            for target in samples:
                target = target
                box = HashableBox.from_slice(target['space_slice'])
                box_tup = tuple([box.format] + box.data.tolist())
                spatial_slices.update([box_tup])

        # Fill in the gaps the sampler missed
        new_idx_to_samples = ub.ddict(list)
        for idx in missing_frame_idxs:
            if idx == time_sampler.indexes[0]:
                continue
            # Mask out everything in the future. We must take something
            # from the past.

            exclude_idxs = time_sampler.indexes[time_sampler.indexes > idx]
            sample_idxs = time_sampler.sample(idx, exclude=exclude_idxs)

            sample_gids = time_sampler.video_gids[sample_idxs]
            main_gid = sample_gids[1]
            main_frame_index = coco_dset.index.imgs[main_gid]['frame_index']
            main_time_sampler_index = time_sampler.gid_to_index[main_gid]

            partial_tr = ub.udict({
                'main_idx': main_time_sampler_index,
                'video_id': vidid,
                'vidid': vidid,
                'gids': list(map(int, sample_gids)),
                'main_gid': int(main_gid),
                'frame_index': int(main_frame_index),
                'frame_indexes': list(map(int, ub.take(frame_idxs, sample_idxs))),
                # 'space_slice': (slice(0, 256, None), slice(0, 256, None)),
                'resampled': None,
                'label': None,
            })
            for space_slice in spatial_slices.keys():
                format, a, b, c, d = space_slice
                box = kwimage.Box.coerce([[a, b, c, d]], format=format)
                space_slice = box.to_slice()
                new_tr = partial_tr | {
                    'space_slice': space_slice,
                }
                new_idx_to_samples[main_frame_index].append(new_tr)

        # For each each main frame, choose only one other frame as it's pair.
        for main_idx, samples in idx_to_sample.items():
            # TODO: spatial coverage
            chosen = ub.udict(ub.group_items(samples, lambda x: x['gids'][0])).peek_value()
            new_idx_to_samples[main_idx] = chosen

        vidid_to_new_samples[vidid] = list(ub.flatten(new_idx_to_samples.values()))
    return vidid_to_new_samples


# backwards compat
BACKWARDS_COMPAT = 1
if BACKWARDS_COMPAT:
    gridded_dataset = GriddedDataset
    from geowatch.tasks.invariants.data.other_datasets import kwcoco_dataset, Onera, SpaceNet7  # NOQA
