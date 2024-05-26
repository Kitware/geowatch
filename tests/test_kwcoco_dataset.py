def test_dynamic_resolution():
    from geowatch.tasks.fusion.datamodules.kwcoco_dataset import KWCocoVideoDataset
    import geowatch
    import ubelt as ub

    # Ensure the two videos have different sizes
    # (one is much larger than the other)
    coco_dset1 = geowatch.coerce_kwcoco(
        'geowatch', num_videos=1, image_size=(8, 8), num_frames=3,
        multisensor=False,
        multispectral=False,
        geodata={
            'enabled': True,
            'region_geom': 'random-proportional',
            'target_gsd': 2.0
        }, rng=10)
    coco_dset2 = geowatch.coerce_kwcoco(
        'geowatch', num_videos=1, image_size=(64, 64), num_frames=3,
        multisensor=False,
        multispectral=False,
        geodata={
            'enabled': True,
            'region_geom': 'random-proportional',
            'target_gsd': 2.0
        }, rng=11)
    coco_dset = coco_dset1.union(coco_dset2)

    coco_dset.images().lookup(['width', 'height'])
    coco_dset.videos().lookup(['width', 'height'])

    # Enable "dynamic fixed resolution" which should scale down the image
    # so there are a maximum number of windows
    dynamic_fixed_resolution = {
        'max_winspace_full_dims': (4, 4),
    }
    # dynamic_fixed_resolution = None
    self = KWCocoVideoDataset(coco_dset, time_dims=3,
                              window_dims=(4, 4), fixed_resolution='4.0GSD',
                              channels='red|green|blue', autobuild=False, mode='test',
                              use_grid_cache=False,
                              resample_invalid_frames=False,
                              force_bad_frames=True,
                              dynamic_fixed_resolution=dynamic_fixed_resolution)
    self.requested_tasks['change'] = False
    self.requested_tasks['class'] = False
    self.requested_tasks['boxes'] = False
    self.requested_tasks['outputs'] = False
    self._init()
    target1 = (self.sample_grid['targets'][0])
    target2 = (self.sample_grid['targets'][-1])

    # Check that we downsample the big video but keep the small video to some
    # target resolution at its regular resolution.
    import rich
    rich.print(f'target1 = {ub.urepr(target1, nl=1)}')
    rich.print(f'target2 = {ub.urepr(target2, nl=1)}')

    sample1 = self[target1]
    sample2 = self[target2]
    summary1 = self.summarize_item(sample1)
    summary2 = self.summarize_item(sample2)
    import rich
    rich.print(f'summary1 = {ub.urepr(summary1, nl=-2)}')
    rich.print(f'summary2 = {ub.urepr(summary2, nl=-2)}')

    # The dynamic resolution should force the samples to be the same size
    # in the sampled input dimensions, but the first should be at a coarser
    # resolution.
    shape1 = sample1['frames'][0]['modes']['red|green|blue'].shape
    shape2 = sample2['frames'][0]['modes']['red|green|blue'].shape
    assert tuple(shape1) == tuple(shape2)

    # Note: the random number generator which controls some details of the
    # generated video / geo-crs size (even though we specify the image size)
    # influences if the following statement is true.  If this starts to fail,
    # it could be due to RNG seed issues.  It would be nice to ensure this test
    # properly generates all relevant aspects of the data we are trying to
    # test, and that those are documented well.
    scale1 = sample1['frames'][0]['scale_outspace_from_vid']
    scale2 = sample2['frames'][0]['scale_outspace_from_vid']
    import numpy as np
    assert np.all(scale1 < scale2)
