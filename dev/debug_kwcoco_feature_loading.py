import xdev


def debug_feature_loading():
    """
    profile loading steps in ndsampler / kwcoco
    """
    import xdev
    from watch.tasks.fusion.datamodules.kwcoco_video_data import KWCocoVideoDataset  # NOQA
    import ndsampler
    import kwcoco
    import ubelt as ub
    from watch.utils.util_data import find_smart_dvc_dpath
    dvc_dpath = find_smart_dvc_dpath()

    print(ub.repr2(list((dvc_dpath / 'drop1-S2-L8-aligned').glob('combo*.json'))))

    coco_fpath = dvc_dpath / 'drop1-S2-L8-aligned/combo_train_data.kwcoco.json'
    coco_dset = kwcoco.CocoDataset(coco_fpath)
    sampler = ndsampler.CocoSampler(coco_dset)
    sample_shape = (3, 96, 96)
    channels = 'blue|green|red|nir|swir16'
    channels = 'blue'

    from watch.cli import watch_coco_stats
    watch_coco_stats.coco_watch_stats(coco_dset)

    channels = 'blue|inv_shared30'

    #channels = 'rice_field|cropland|water|inland_water|river_or_stream|sebkha|snow_or_ice_field|bare_ground|sand_dune|built_up|grassland|brush|forest|wetland|road'
    #channels = 'matseg_0|matseg_1|matseg_2|matseg_3|matseg_4'
    self = KWCocoVideoDataset(sampler, sample_shape=sample_shape, channels=channels, neg_to_pos_ratio=1.0)
    print('self = {!r}'.format(self))

    # _ = xdev.profile_now(self.compute_dataset_stats)(num=10, num_workers=4, batch_size=1)

    # Profile the getitem step (points to load_sample as the bottleneck)
    if 0:
        _ = xdev.profile_now(self.__getitem__)(0)

    tr = self.new_sample_grid['targets'][0]
    tr['channels'] = self.channels
    print('tr = {!r}'.format(tr))

    # Profile load_sample itself (points to sampler._load_slice as bottleneck)
    if 0:
        _ = xdev.profile_now(self.sampler.load_sample)(tr)

    pad = None
    padkw = {}

    # Several longer lived places exist here
    # Try with and without experimental loading
    if 0:
        tr['use_experimental_loader'] = 0
        with ub.Timer() as t1:
            item1 = xdev.profile_now(self.sampler._load_slice)(tr, pad, padkw)

        tr['use_experimental_loader'] = 1
        with ub.Timer() as t2:
            item2 = xdev.profile_now(self.sampler._load_slice)(tr, pad, padkw)

        print('t1.elapsed = {!r}'.format(t1.elapsed))
        print('t2.elapsed = {!r}'.format(t2.elapsed))
        # -- inspect outputs
        item1['im'].shape
        item1['im'].sum()
        print(item1['im'].mean(axis=(1, 2)))

        item2['im'].shape
        print(item2['im'].mean(axis=(1, 2)))

    if 0:
        import timerit
        print('start experimental loader timing')
        ti = timerit.Timerit(1, bestof=1, verbose=3)
        for timer in ti.reset('time'):
            with timer:
                tr['use_experimental_loader'] = 0
                (self.sampler._load_slice)(tr, pad, padkw)

        for timer in ti.reset('time'):
            with timer:
                tr['use_experimental_loader'] = 1
                (self.sampler._load_slice)(tr, pad, padkw)

    print('done experimental loader timing')
    # Dig into the experimental loader
    time_gids = tr['gids']
    gid = time_gids[0]
    dset = coco_dset
    space_slice = tr['space_slice']

    from kwcoco import channel_spec
    request_chanspec = channel_spec.ChannelSpec.coerce(channels)

    if 0:
        obj = dset.imgs[2]['auxiliary'][0]
        xdev.profile_now(dset._delay_load_imglike)(obj)

        from kwcoco.channel_spec import FusedChannelSpec
        xdev.profile_now(FusedChannelSpec.coerce)(obj['channels'])
        xdev.profile_now(FusedChannelSpec.parse)(obj['channels'])

        xdev.profile_now(FusedChannelSpec.coerce('a').normalize)()
        xdev.profile_now(FusedChannelSpec.coerce('a').normalize)()

        with ub.Timer() as t3:
            delayed_frame = xdev.profile_now(dset.delayed_load)(gid, space='video')
        with ub.Timer() as t4:
            delayed_frame = xdev.profile_now(dset.delayed_load)(gid, space='video', channels=request_chanspec)
        print('t3.elapsed = {!r}'.format(t3.elapsed))
        print('t4.elapsed = {!r}'.format(t4.elapsed))

        delayed_frame = xdev.profile_now(delayed_frame.delayed_crop)(space_slice)

    delayed_frame = dset.delayed_load(gid, space='video')
    delayed_frame = delayed_frame.crop(space_slice)
    # if not all_chan:
    #     delayed_frame = delayed_frame.take_channels(request_chanspec)
    # xr_frame = delayed_frame.finalize(as_xarray=True)
    # space_frames.append(xr_frame)

    for i in range(100):
        # Not a huge time difference in the actual delayed load, which seems correct
        gid = 2
        with ub.Timer() as tshrd:
            shared_delayed = (coco_dset.delayed_load)(gid=gid, channels='inv_shared30')
        with ub.Timer() as tblue:
            blue_delayed = (coco_dset.delayed_load)(gid=gid, channels='blue')
        print('---')
        print('shared tshrd.elapsed = {!r}'.format(tshrd.elapsed))
        print('blue tblue.elapsed   = {!r}'.format(tblue.elapsed))

        # The crop operation is also fine
        print('---')
        space_slice = (slice(0, 96, None), slice(0, 96, None))
        gid = 2
        with ub.Timer() as tshrd:
            shared_crop = shared_delayed.delayed_crop(space_slice)
        with ub.Timer() as tblue:
            blue_crop = blue_delayed.delayed_crop(space_slice)
        print('shared tshrd.elapsed = {!r}'.format(tshrd.elapsed))
        print('blue tblue.elapsed   = {!r}'.format(tblue.elapsed))

        # FIRST run of finalize is about 100x slower.
        print('---')
        with ub.Timer() as tblue:
            blue_final = blue_crop.finalize()
        with ub.Timer() as tshrd:
            shared_final = shared_crop.finalize()
        print('shared tshrd.elapsed = {!r}'.format(tshrd.elapsed))
        print('blue tblue.elapsed   = {!r}'.format(tblue.elapsed))

        # SUBSEQUENT runs seem to be hitting a cache
        print('---')
        with ub.Timer() as tshrd:
            shared_final = shared_crop.finalize()
        with ub.Timer() as tblue:
            blue_final = blue_crop.finalize()
        print('shared tshrd.elapsed = {!r}'.format(tshrd.elapsed))
        print('blue tblue.elapsed   = {!r}'.format(tblue.elapsed))

    if 1:
        from os.path import join
        from kwcoco.util.util_delayed_poc import LazyGDalFrameFile
        aux_blue = [aux for aux in coco_dset.imgs[gid]['auxiliary'] if 'blue' in aux['channels']][0]
        blue_fpath = join(dset.bundle_dpath, aux_blue['file_name'])
        blue_frame = LazyGDalFrameFile(blue_fpath)
        sl_blue = (slice(0, 147, None), slice(0, 147, None), slice(None, None, None))

        aux_shared = [aux for aux in coco_dset.imgs[gid]['auxiliary'] if 'inv_shared1' in aux['channels']][0]
        shared_fpath = join(dset.bundle_dpath, aux_shared['file_name'])
        shared_frame = LazyGDalFrameFile(shared_fpath)
        sl_shared = (slice(0, 147, None), slice(0, 147, None), [29])

        print('blue_fpath = {!r}'.format(blue_fpath))
        print('shared_fpath = {!r}'.format(shared_fpath))

        import timerit
        ti = timerit.Timerit(1, bestof=10, verbose=2)
        for timer in ti.reset('blue'):
            with timer:
                blue_frame[sl_blue]

        for timer in ti.reset('shared'):
            with timer:
                shared_frame[sl_shared]

        for timer in ti.reset('shared'):
            with timer:
                shared_frame[sl_shared]

        for timer in ti.reset('blue'):
            with timer:
                blue_frame[sl_blue]


@xdev.profile
def simple_debug_blue():
    simple_debug_feature_loading('blue')


@xdev.profile
def simple_debug_shared():
    simple_debug_feature_loading('inv_shared30')


@xdev.profile
def simple_debug_feature_loading(channels):
    """
    Method to profile a single run of feature loading for some channel selection
    """
    from watch.tasks.fusion.datamodules.kwcoco_video_data import KWCocoVideoDataset  # NOQA
    import ndsampler
    import kwcoco
    from watch.utils.util_data import find_smart_dvc_dpath
    dvc_dpath = find_smart_dvc_dpath()
    coco_fpath = dvc_dpath / 'drop1-S2-L8-aligned/combo_train_data.kwcoco.json'
    coco_dset = kwcoco.CocoDataset(coco_fpath)
    sampler = ndsampler.CocoSampler(coco_dset)
    sample_shape = (3, 96, 96)
    self = KWCocoVideoDataset(sampler, sample_shape=sample_shape, channels=channels)

    self[0]
    self[1]
    self[2]
    self[3]


if __name__ == '__main__':
    """
    CommandLine:
        python ~/code/watch/dev/debug_kwcoco_feature_loading.py --profile
    """
    simple_debug_blue()
    simple_debug_shared()
    # simple_debug_feature_loading('inv_shared30')
    # debug_feature_loading()
