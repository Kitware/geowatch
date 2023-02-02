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
    print('item summary: ' + ub.repr2(self.summarize_item(item), nl=3))

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

    #print('item summary: ' + ub.repr2(self.summarize_item(item), nl=3))
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
        qa_data = coco_img.delay('cloudmask', space=space).scale(scale).finalize(interpolation='nearest', antialias=1)
        rgb_img = coco_img.delay('red|green|blue', space=space).scale(scale).finalize(interpolation='linear', nodata_method='ma')

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

    data = coco_img.delay('invariants.0').finalize()
    kwplot.imshow(data)

    target_ = item['target'].copy()
    gid = target_['gids'][0]

    self._sample_one_frame(gid, sampler, coco_dset, target_, with_annots,
                           gid_to_isbad, gid_to_sample)
    sample = gid_to_sample[gid]

    for mode_name, mode in sample.items():
        pass

    # mode_name = 'invariants.0|invariants.1|invariants.2|invariants.3|invariants.4|invariants.5|invariants.6|invariants.7|invariants.8|invariants.9|invariants.10|invariants.11|invariants.12|invariants.13|invariants.14|invariants.15|invariants.16'
