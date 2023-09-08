"""
Developer notebook
"""


def dzyne_mwe():
    import watch
    import ubelt as ub
    from watch.tasks.fusion.datamodules.kwcoco_dataset import KWCocoVideoDataset
    dvc_dpath = watch.find_dvc_dpath(tags='phase2_data', hardware='auto')
    coco_fpath = dvc_dpath / 'Drop4-SC/yourdata.kwcoco.json'
    channels = 'red|green|blue,change'
    self = KWCocoVideoDataset(coco_fpath,
                              time_steps=5,
                              chip_dims=(196, 196),
                              time_sampling='uniform',
                              input_resolution='3GSD',
                              window_resolution='3GSD',
                              output_resolution='3GSD',
                              channels=channels)
    self.disable_augmenter = True

    # Get a sample target around a positive example.
    # Change properties as necessary to reproduce the error.
    target = self.new_sample_grid['targets'][self.new_sample_grid['positives_indexes'][0]].copy()

    # Execute sampling logic. Dive into the getitem function as necessary to
    # debug.
    item = self[target]
    print('item summary: ' + ub.urepr(self.summarize_item(item), nl=3))

    # Display the sample
    canvas = self.draw_item(item, overlay_on_image=0, rescale=1,
                            max_channels=5)

    import kwplot
    kwplot.autompl()
    kwplot.imshow(canvas, fnum=1)
    kwplot.show_if_requested()


def visualize_cloudmask_batch():
    # import os
    # os.environ['XDEV_PROFILE'] = '1'
    import watch
    import numpy as np  # NOQA
    import kwimage  # NOQA
    from watch.tasks.fusion.datamodules.kwcoco_dataset import KWCocoVideoDataset
    dvc_dpath = watch.find_dvc_dpath(tags='phase2_data', hardware='auto')
    coco_fpath = dvc_dpath / 'Drop4-BAS/combo_vali_I2.kwcoco.json'
    # channels = 'red|green|blue,invariants.0:3,invariants.16,cloudmask'
    channels = 'red|green|blue,cloudmask'
    # coco_fpath = dvc_dpath / 'Drop4-BAS/KR_R001.kwcoco.json'
    # channels = 'red|green|blue,nir|swir16|swir22'
    self = KWCocoVideoDataset(coco_fpath,
                              time_steps=4,
                              chip_dims='full',
                              time_sampling='uniform',
                              input_resolution='native',
                              # input_resolution='3.3GSD',
                              window_resolution='10GSD',
                              output_resolution='10GSD',
                              channels=channels)
    self.disable_augmenter = True
    target = self.new_sample_grid['targets'][self.new_sample_grid['positives_indexes'][100]].copy()

    # vidid = self.sampler.dset.videos()[0]
    # images = self.sampler.dset.images(video_id=vidid)
    # full_gids = []
    # partial_gids = []
    # for obj in images.objs:
    #     poly = kwimage.MultiPolygon.coerce(obj['valid_region'])
    #     covered = poly.area / (obj['width'] * obj['height'])
    #     if covered >= 0.95:
    #         full_gids.append(obj['id'])
    #     else:
    #         partial_gids.append(obj['id'])
    # target['gids'] = full_gids[0:5] + partial_gids[0:2]
    # target['space_slice'] = (slice(-16, 224), slice(-16, 360))

    target['SAMECOLOR_QUALITY_HEURISTIC'] = None
    # target['SAMECOLOR_QUALITY_HEURISTIC'] = 'histogram'
    target['PROPOGATE_NAN_BANDS'] = set({'red'})
    # target['PROPOGATE_NAN_BANDS'] = set()
    target['force_bad_frames'] = 1
    target['mask_low_quality'] = 1
    target['quality_threshold'] = 0.0
    target['observable_threshold'] = 0.0
    target['resample_invalid_frames'] = 1
    item = self[target]

    # import xdev
    # xdev.profile.print_report()

    #print('item summary: ' + ub.urepr(self.summarize_item(item), nl=3))
    canvas1 = self.draw_item(item, overlay_on_image=0, rescale=1,
                             max_channels=5,
                             draw_truth=False,
                             draw_weights=False,
                             # combinable_extra='invariants.0:3'
                             )

    unmasked_target = target.copy()
    unmasked_target['mask_low_quality'] = 0
    item_unmasked = self[unmasked_target]
    canvas2 = self.draw_item(item_unmasked, overlay_on_image=0, rescale=1,
                             max_channels=1,
                             draw_truth=False,
                             draw_weights=False,
                             # combinable_extra='invariants.0:3'
                             )

    canvas = kwimage.stack_images([canvas1, canvas2], axis=0)

    # xdoctest: +REQUIRES(--show)
    import kwplot
    kwplot.autompl()
    kwplot.imshow(canvas, fnum=1)
    kwplot.show_if_requested()

    if 0:
        # resample the same target but no resampling
        target = item['target']
        target['resample_invalid_frames'] = 0
        target['force_bad_frames'] = 1
        target['mask_low_quality'] = 1
        # And different properties
        item = self[target]

        canvas = self.draw_item(item, overlay_on_image=0, rescale=1,
                                max_channels=5,
                                # combinable_extra='invariants.0:3'
                                )
        import kwplot
        kwplot.autompl()
        kwplot.imshow(canvas, fnum=2)
        kwplot.show_if_requested()

    if 0:
        from watch.tasks.fusion.datamodules.qa_bands import QA_SPECS
        import kwarray
        import ubelt as ub
        plt = kwplot.autoplt()
        # Pick a cloudmask that looks questionable
        chosen_index = 3

        frame_item = item['frames'][chosen_index]
        qa_data = frame_item['modes']['cloudmask'].numpy()
        print(ub.urepr(ub.udict(ub.dict_hist(qa_data.ravel())).sorted_keys()))
        # qa_data[np.isnan(qa_data)] = -9999
        # qa_data = np.clip(qa_data, 0, None)
        qa_data = qa_data.astype('int16')[0]

        # We don't have the exact right information here, so we can
        # punt for now and assume "Drop4"
        spec_name = 'ACC-1'
        gid = frame_item['gid']
        coco_img = self.sampler.dset.coco_image(gid)
        sensor = coco_img.img.get('sensor_coarse', '*')
        table = QA_SPECS.find_table(spec_name, sensor)
        print(f'table={table}')
        print('table.spec = {}'.format(ub.urepr(table.spec, nl=2)))

        fpath = ub.Path(coco_img.bundle_dpath) / coco_img.find_asset_obj('cloudmask')['file_name']
        qa_data = kwimage.imread(fpath)

        space = 'video'
        # space = 'image'
        scale = 0.4
        qa_data = coco_img.imdelay('cloudmask', space=space).scale(scale).finalize(interpolation='nearest', antialias=1)
        rgb_img = coco_img.imdelay('red|green|blue', space=space).scale(scale).finalize(interpolation='linear', nodata_method='ma')

        quality_im = qa_data
        result = table.draw_labels(quality_im)
        qa_canvas = result['qa_canvas']
        iffy_qa_names = [
            'cloud',
            # 'dilated_cloud',
            'cirrus',
        ]
        qa_names = iffy_qa_names
        is_cloud_iffy = table.mask_any(qa_data, qa_names)
        # norm_rgb, infos_to_return = kwarray.robust_normalize(rgb_img.clip(0, None), axis=None, params=dict(scaling='sigmoid', low=0.00, mid=0.8, high=0.99, extrema='quantile'), return_info=True)
        norm_rgb, infos_to_return = kwarray.robust_normalize(rgb_img.clip(0, None), axis=None, params=dict(scaling='sigmoid'), return_info=True)
        # norm_rgb = kwarray.robust_normalize(rgb_img)
        kwplot.imshow(norm_rgb, fnum=3, pnum=(1, 4, 1), title='rgb')
        kwplot.imshow(is_cloud_iffy, pnum=(1, 4, 2), title=f'mask matching {iffy_qa_names}')
        kwplot.imshow(qa_canvas, fnum=3, pnum=(1, 4, 3), title='qa bits')
        kwplot.imshow(result['legend'], fnum=3, pnum=(1, 4, 4), title='qa legend')
        kwplot.set_figtitle(f"QA Spec: name={table.spec['qa_spec_name']} sensor={table.spec['sensor']}")
        ax = plt.gca()
        ax.figure.tight_layout()


#     if 0:
#         data = item['frames'][0]['modes']['invariants.0|invariants.1|invariants.2'][..., 0].numpy()


# import timerit
# ti = timerit.Timerit(100, bestof=10, verbose=2)
# for timer in ti.reset('time'):
#     with timer:
#         # Speed up the compuation by doing this at a coarser scale
#         is_samecolor = util_kwimage.find_samecolor_regions(bands, scale=0.4, min_region_size=49)

# import timerit
# ti = timerit.Timerit(100, bestof=10, verbose=2)
# for timer in ti.reset('time'):
#     with timer:
#         util_kwimage.find_high_frequency_values(bands)


def _check_target(self):
    target = self.new_sample_grid['targets'][0]

    dset = self.sampler.dset
    gid_to_isbad = {}
    gid_to_sample = {}
    with_annots = 0
    coco_dset = dset
    sampler = self.sampler
    target_ = target.copy()
    gids = target_['gids']
    gid = gids[0]

    dset.images(target['gids']).lookup('sensor_coarse')

    item = self.getitem(target)

    coco_img = dset.coco_image(gid)

    import kwplot
    kwplot.autompl()
    kwplot.figure()

    kwplot.imshow(self.draw_item(item))

    data = coco_img.imdelay('invariants.0').finalize()
    kwplot.imshow(data)

    target_ = item['target'].copy()
    gid = target_['gids'][0]

    self._sample_one_frame(gid, sampler, coco_dset, target_, with_annots,
                           gid_to_isbad, gid_to_sample)
    sample = gid_to_sample[gid]

    for mode_name, mode in sample.items():
        pass

    # mode_name = 'invariants.0|invariants.1|invariants.2|invariants.3|invariants.4|invariants.5|invariants.6|invariants.7|invariants.8|invariants.9|invariants.10|invariants.11|invariants.12|invariants.13|invariants.14|invariants.15|invariants.16'


def debug_cloudmasks():
    """
    Visualize cloudmasks for all types of sensors and on a variety of real
    datasets.
    """
    import kwcoco
    import watch
    import kwimage
    import ubelt as ub
    import kwarray

    dvc_dpath = watch.find_dvc_dpath(tags='phase2_data', hardware='hdd')

    bundle_dpath = dvc_dpath / 'Aligned-Drop7'

    coco_dataset_fpaths = [
        bundle_dpath / 'KR_R002/imgonly-KR_R002.kwcoco.zip',
        bundle_dpath / 'CN_C000/imgonly-CN_C000.kwcoco.zip',
    ]

    # Directory to write debugging visualizations to
    out_dpath = ub.Path('$HOME/temp/debug_qa').expand().ensuredir()

    coco_datasets = list(kwcoco.CocoDataset.coerce_multiple(coco_dataset_fpaths, workers=16))

    images_of_interest = []

    for coco_dset in ub.ProgIter(coco_datasets, desc='choosing images of interest'):
        coco_images = coco_dset.images().coco_images

        # Select a group of images
        group_to_images = ub.udict(ub.group_items(
            coco_images,
            key=lambda coco_img: (
                coco_img['sensor_coarse'],
                tuple(sorted(coco_img.channels.unique(normalize=True)))
            )
        ))

        # Print info about number of images per type
        print('coco_dset = {}'.format(ub.urepr(coco_dset, nl=1)))
        print(ub.urepr(group_to_images.map_values(len)))

        # Seeded random number generator
        rng = kwarray.ensure_rng(48942398243, api='python')

        for sensor, imgs in group_to_images.items():
            # Randomly pick one image for each sensor
            coco_img = rng.choice(imgs)
            images_of_interest.append(coco_img)

    for coco_img in ub.ProgIter(images_of_interest, desc='draw cloudmask debug image', verbose=3):
        vidname = coco_img.video['name']
        imgname = coco_img['name']
        sensor = coco_img['sensor_coarse']
        fname = f'qa_debug_{vidname}_{sensor}_{imgname}.jpg'
        print(coco_img.dsize)
        print('fname = {}'.format(ub.urepr(fname, nl=1)))

        drawings = debug_single_cloudmask(coco_img)

        tci_canvas = drawings['tci_canvas']
        qa_canvas = drawings['qa_canvas']
        legend = drawings['legend']
        title = drawings['title']

        canvas = kwimage.stack_images([tci_canvas, qa_canvas, legend], axis=1)
        canvas = kwimage.draw_header_text(canvas, title)

        print(f'canvas.shape={canvas.shape}')

        fpath = out_dpath / fname
        kwimage.imwrite(fpath, canvas)

    # import kwplot
    # kwplot.autoplt()
    # kwplot.imshow(canvas, fnum=1)
    # kwplot.imshow(legend, fnum=2)


def debug_single_cloudmask(coco_img):
    """
    """
    import kwimage
    from watch.utils import kwcoco_extensions
    from watch.tasks.fusion.datamodules.qa_bands import QA_SPECS
    from watch.utils import util_time

    tci_channel_priority = [
        'red|green|blue',
        'pan',
    ]
    tci_channels = kwcoco_extensions.pick_channels(coco_img, tci_channel_priority)

    # Load quality and visual bands in image space
    space = 'image'
    qa_delayed = coco_img.imdelay('quality', interpolation='nearest', antialias=False, space=space)
    tci_delayed = coco_img.imdelay(tci_channels, space=space)

    quality_im = qa_delayed.finalize(interpolation='nearest', antialias=False)
    tci_raw = tci_delayed.finalize(nodata_method='float')
    tci_canvas = kwimage.normalize_intensity(tci_raw)

    sensor = coco_img.img.get('sensor_coarse')
    print(f'sensor={sensor}')

    # Lookup the correct QA spec for the image type.
    spec_name = 'ACC-1'
    sensor = coco_img['sensor_coarse']
    table = QA_SPECS.find_table(spec_name, sensor)

    drawings = table.draw_labels(quality_im, legend_dpi=300)
    drawings['tci_canvas'] = tci_canvas
    drawings['tci_canvas'] = kwimage.ensure_uint255(drawings['tci_canvas'])
    drawings['qa_canvas'] = kwimage.ensure_uint255(drawings['qa_canvas'])

    datestr = util_time.coerce_datetime(coco_img['date_captured']).date().isoformat()

    title_parts = [
        coco_img.video['name'],
        coco_img['sensor_coarse'],
        datestr,
    ]

    drawings['title'] = ' - '.join(title_parts)

    if 0:
        import kwplot
        kwplot.autompl()
        tci_canvas = drawings['tci_canvas']
        qa_canvas = drawings['qa_canvas']
        legend = drawings['legend']
        kwplot.imshow(tci_canvas, fnum=1)
        kwplot.imshow(qa_canvas, fnum=2)
        kwplot.imshow(legend, fnum=3)

    return drawings


def debug_specific_qa_masks():

    from watch.utils import util_fsspec
    util_fsspec.FSPath.coerce
    # fs = util_fsspec.S3Path._new_fs(profile='iarpa')

    s3_paths = [util_fsspec.FSPath.coerce('s3://' + p[7:]) for p in  [
        "/vsis3/smart-data-accenture/ta-1/ta1-wv-acc-3/52/S/DG/2018/1/5/18JAN05020545-P1BS-011778196010_01_P001/18JAN05020545-P1BS-011778196010_01_P001_ACC_QA.tif",
        "/vsis3/smart-data-accenture/ta-1/ta1-wv-acc-3/52/S/DG/2018/1/5/18JAN05020545-M1BS-011778196010_01_P001/18JAN05020545-M1BS-011778196010_01_P001_ACC_QA.tif",
        "/vsis3/smart-data-accenture/ta-1/ta1-wv-acc-3/52/S/DG/2018/1/5/18JAN05020545-M1BS-011778196010_01_P002/18JAN05020545-M1BS-011778196010_01_P002_ACC_QA.tif",
        "/vsis3/smart-data-accenture/ta-1/ta1-wv-acc-3/52/S/DG/2018/1/5/18JAN05020545-P1BS-011778196010_01_P002/18JAN05020545-P1BS-011778196010_01_P002_ACC_QA.tif"
    ]]
    import ubelt as ub
    dpath = ub.Path.appdir('watch/temp/').ensuredir()

    local_fpaths = []
    for p in s3_paths:
        fpath = dpath / p.name
        local_fpaths.append(fpath)
        if not fpath.exists():
            p.copy(fpath)

    import kwimage
    from watch.tasks.fusion.datamodules.qa_bands import QA_SPECS
    table = QA_SPECS.find_table('ACC-1', 'WV')

    for fpath in local_fpaths:
        viz_fpath = fpath.augment(prefix='_viz', ext='.png')
        if True or not viz_fpath.exists():
            if 'M1BS' in str(fpath.name):
                quality_im = kwimage.imread(fpath, overview=1)
            if 'P1BS' in fpath.name:
                quality_im = kwimage.imread(fpath, overview=3)
            drawings = table.draw_labels(quality_im, legend_dpi=300, verbose=1)
            qa_canvas = kwimage.ensure_uint255(drawings['qa_canvas'])
            im_legend = kwimage.ensure_uint255(drawings['legend'])
            # middle = kwimage.imcrop(qa_canvas, (1000, 1000), about='cc')
            canvas = kwimage.stack_images([qa_canvas, im_legend], axis=1)
            canvas = kwimage.draw_header_text(canvas, fpath.name)
            kwimage.imwrite(viz_fpath, canvas)

            iffy_qa_names = ['cloud', 'cirrus']
            train_mask = table.mask_any(quality_im, iffy_qa_names)
            avoid_quality_values = ['cloud', 'cloud_shadow', 'cloud_adjacent']
            time_combo_mask = table.mask_any(quality_im, avoid_quality_values)
            viz_fpath = fpath.augment(prefix='_viz_mask1', ext='.png')
            kwimage.imwrite(viz_fpath, kwimage.ensure_uint255(train_mask))
            viz_fpath = fpath.augment(prefix='_viz_mask2', ext='.png')
            kwimage.imwrite(viz_fpath, kwimage.ensure_uint255(time_combo_mask))
