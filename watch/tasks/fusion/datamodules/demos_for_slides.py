import ubelt as ub
import kwimage
import numpy as np
import ndsampler
import einops
import kwcoco


def demo_visualize_tokenization():
    """
    Helper for slides
    """
    import watch
    from watch.tasks.fusion.datamodules.kwcoco_dataset import KWCocoVideoDataset
    dvc_dpath = watch.find_smart_dvc_dpath()
    coco_fpath = dvc_dpath / 'Drop1-Aligned-L1-2022-01/combo_DILM_train.kwcoco.json'
    coco_dset = kwcoco.CocoDataset(coco_fpath)

    sampler = ndsampler.CocoSampler(coco_dset)
    channels = 'red|green|blue,nir,bare_ground,panchromatic'
    sample_shape = (2, 96, 96)
    self = KWCocoVideoDataset(sampler, sample_shape=sample_shape,
                              channels=channels,
                              time_sampling='soft2+distribute',
                              resample_invalid_frames=False)
    self.requested_tasks['change'] = False

    vidid = coco_dset.index.name_to_video['NZ_R001']['id']
    # vidid = list(coco_dset.videos())[3]
    coco_dset.index.videos[vidid]
    images = coco_dset.images(vidid=vidid)
    sensor_gids = {}
    for coco_img in images.coco_images:
        sensor = coco_img.img.get('sensor_coarse')
        if sensor_gids.get(sensor) is None:
            sensor_gids[sensor] = coco_img.img['id']

    gids = list(sensor_gids.values())
    print('gids = {!r}'.format(gids))

    tr = {
        'main_idx': 0,
        'video_id': vidid,
        # 'space_slice': (slice(0, 96, None), slice(0, 96, None)),
        'space_slice': (slice(200, 296, None), slice(200, 296, None)),
        'gids': gids,
    }
    self.disable_augmenter = True
    item = self[tr]

    canvas = self.draw_item(item)
    import kwplot
    kwplot.autompl()
    kwplot.imshow(canvas, fnum=1)

    to_stack_modes = []
    to_stack_flats = []

    for frame in item['frames']:

        frame_stack_modes = []
        frame_stack_flats = []

        for mode_key, mode in frame['modes'].items():
            imtensor = mode.numpy()
            img = einops.rearrange(imtensor, 'c h w -> h w c')
            img = kwimage.normalize_intensity(img)

            chunks1 = einops.rearrange(img, '(wh h) (ww w) c -> (wh ww) h w c', ww=8, wh=8)

            grid_stack = kwimage.stack_images_grid(chunks1, pad=4, axis=0)
            grid_stack = kwimage.imresize(grid_stack, scale=2.0, interpolation='nearest')

            # chunks1 = einops.rearrange(img, '(wh h) (ww w) c -> (wh ww) (c h) w 1', ww=8, wh=8)
            # chunks1 = einops.rearrange(img, '(wh h) (ww w) c -> (wh ww) (h w) 1 c', ww=8, wh=8)
            chunks2 = einops.rearrange(img, '(wh h) (ww f w) c -> (wh ww) (h w) f c', ww=8, wh=8, f=4)

            flat_stack = kwimage.stack_images(chunks2, pad=4, axis=1)

            flat_stack = kwimage.ensure_uint255(flat_stack)
            flat_stack = kwimage.draw_header_text(flat_stack, f'{mode_key}', fit='shrink')

            grid_stack = kwimage.ensure_uint255(grid_stack)
            grid_stack = kwimage.draw_header_text(grid_stack, f'{mode_key}', fit='shrink')

            frame_stack_modes.append(grid_stack)
            frame_stack_flats.append(flat_stack)

        frame_mode_canvas = kwimage.stack_images(frame_stack_modes, axis=1, pad=16, bg_value=(0, 0, 0))
        frame_flat_canvas = kwimage.stack_images(frame_stack_flats, pad=16, axis=1, bg_value=(0, 0, 0))

        sensor = frame['sensor']
        time_index = frame['time_index']
        time_offset = np.array(frame['time_offset']).ravel()[0]

        frame_mode_canvas = kwimage.draw_header_text(frame_mode_canvas, f'{time_index=}', fit='shrink')
        frame_mode_canvas = kwimage.draw_header_text(frame_mode_canvas, f'{time_offset=}', fit='shrink')
        frame_mode_canvas = kwimage.draw_header_text(frame_mode_canvas, f'{sensor=}', fit='shrink')

        frame_flat_canvas = kwimage.draw_header_text(frame_flat_canvas, f'{time_index=}', fit='shrink')
        frame_flat_canvas = kwimage.draw_header_text(frame_flat_canvas, f'{time_offset=}', fit='shrink')
        frame_flat_canvas = kwimage.draw_header_text(frame_flat_canvas, f'{sensor=}', fit='shrink')

        to_stack_modes.append(frame_mode_canvas)
        to_stack_flats.append(frame_flat_canvas)

    canvas_modes = kwimage.stack_images(to_stack_modes, axis=1, pad=16, bg_value=(255, 255, 255))
    canvas_flats = kwimage.stack_images(to_stack_flats, pad=16, axis=1, bg_value=(255, 255, 255))

    kwplot.imshow(canvas_modes, fnum=2)
    kwplot.imshow(canvas_flats, fnum=3)


def demo_visualize_heterogeneous_inputs():
    # xdoctest: +REQUIRES(env:DVC_DPATH)
    # Run the following tests on real watch data if DVC is available
    import watch
    import ubelt as ub
    from watch.utils import kwcoco_extensions
    from watch.tasks.fusion.datamodules.kwcoco_datamodule import KWCocoVideoDataModule
    import kwcoco
    import kwplot
    kwplot.autompl()

    dvc_dpath = watch.find_smart_dvc_dpath()
    coco_fpath = dvc_dpath / 'Drop1-Aligned-L1-2022-01/combo_DILM.kwcoco.json'
    train_dataset = kwcoco.CocoDataset(coco_fpath)
    chan_info = kwcoco_extensions.coco_channel_stats(train_dataset)

    single_channels_hist = ub.ddict(lambda: 0)
    for chans, freq in chan_info['chan_hist'].items():
        for c in kwcoco.FusedChannelSpec.coerce(chans).as_list():
            single_channels_hist[c] += freq

    print('single_channels_hist = {}'.format(ub.repr2(single_channels_hist, nl=1)))

    channel_groups = [
        'red|green|blue',
        'built_up|forest|water',
        'matseg_0|matseg_1|matseg_2',
        'invariants.0:3',
        'panchromatic',
        'depth',
        'nir|swir16|swir22',
        'lwir11|pan|lwir12'
    ]

    channels = ','.join(channel_groups)
    batch_size = 1
    time_steps = 9
    chip_size = 416
    datamodule = KWCocoVideoDataModule(
        train_dataset=train_dataset,
        test_dataset=None,
        batch_size=batch_size,
        channels=channels,
        num_workers=0,
        time_steps=time_steps,
        chip_size=chip_size,
        neg_to_pos_ratio=0,
        normalize_inputs=1,
        use_centered_positives=True,
        use_grid_positives=False,
        resample_invalid_frames=False,
    )
    datamodule.setup('fit')
    dataset = datamodule.torch_datasets['train']

    coco_dset = train_dataset

    # tr = find_varied_region(coco_dset)

    space_slice = (slice(45, 462, None), slice(850, 1267, None))
    gids = [811, 803, 763, 749, 783, 787, 778]

    vidid = coco_dset.index.name_to_video['NZ_R001']['id']
    tr = {
        'main_idx': 0,
        'video_id': vidid,
        # 'space_slice': (slice(0, 96, None), slice(0, 96, None)),
        'space_slice': space_slice,
        'gids': gids,
    }

    dataset.disable_augmenter = True
    dataset.requested_tasks['change'] = False
    dataset.requested_tasks['saliency'] = False

    item = dataset[tr]

    combinable_extra = [kwcoco.FusedChannelSpec.coerce(g) for g in channel_groups]
    combinable_extra = [p.as_list() for p in combinable_extra if p.numel() == 3]
    canvas = dataset.draw_item(item, draw_weights=False, combinable_extra=combinable_extra, max_dim=416)
    kwplot.imshow(canvas)

    kwimage.imwrite('watch_data.jpg', canvas)

    # fig = kwplot.autoplt().gcf()
    # canvas = kwplot.render_figure_to_image(fig)


def assert_temporal_sampler_consistency(dataset):
    coco_dset = dataset.sampler.dset
    vidid_to_imgs = ub.group_items(coco_dset.dataset['images'], key=lambda img: img['video_id'])
    for vidid, temporal_sampler in dataset.new_sample_grid['vidid_to_time_sampler'].items():
        # vidids = coco_dset.images(temporal_sampler.video_gids).lookup('video_id')
        imgs = vidid_to_imgs[vidid]
        n_imgs = len(imgs)
        n_video_gids = len(temporal_sampler.video_gids)
        assert n_video_gids == n_imgs
        print(f'{vidid=} {n_imgs=} {n_video_gids=}')


def find_varied_region(coco_dset, dataset):
    vidid = coco_dset.index.name_to_video['NZ_R001']['id']
    # vidid = list(coco_dset.videos())[3]
    coco_dset.index.videos[vidid]
    images = coco_dset.images(vidid=vidid)
    sensor_gids = ub.ddict(list)
    for coco_img in images.coco_images:
        sensor = coco_img.img.get('sensor_coarse')
        sensor_gids[sensor].append(coco_img.img['id'])

    import kwarray
    for sensor in sensor_gids.keys():
        sensor_gids[sensor] = kwarray.shuffle(sensor_gids[sensor])

    for gid in sensor_gids['WV']:
        if 'depth' in coco_dset.coco_image(gid).channels:
            print(gid)
            pass
        pass

    gids = list(ub.flatten(list(zip(*sensor_gids.values()))))[0:9]
    coco_dset.images(gids).lookup('sensor_coarse')

    pos_idxs = dataset.new_sample_grid['positives_indexes']
    targets = dataset.new_sample_grid['targets']
    space_slices = []
    for idx in pos_idxs:
        tr = targets[idx]
        if tr['video_id'] == vidid:
            print('tr = {!r}'.format(tr))
            space_slices.append(tr['space_slice'])

    slice_options = list(ub.unique(space_slices, key=ub.hash_data))
    print('slice_options = {}'.format(ub.repr2(slice_options, nl=1)))
    space_slice = slice_options[0]

    # gids = [843, 923, 921, 927]

    # for gid in sensor_gids['WV']:
    #     coco_img = coco_dset.coco_image(gid)
    #     data = coco_img.delay(channels='panchromatic').finalize(space='video')

    # for gid in sensor_gids['WV']:
    #     print(coco_dset.imgs[gid]['valid_region'])

    # gids = list(sensor_gids.values())
    # print('gids = {!r}'.format(gids))
    space_slice = (slice(45, 462, None), slice(850, 1267, None))
    gids = [811, 803, 763, 823, 749, 783, 796, 787, 778]

    tr = {
        'main_idx': 0,
        'video_id': vidid,
        # 'space_slice': (slice(0, 96, None), slice(0, 96, None)),
        'space_slice': space_slice,
        'gids': gids,
    }
    return tr
