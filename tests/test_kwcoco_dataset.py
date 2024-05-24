def test_dynamic_resolution():
    from geowatch.tasks.fusion.datamodules.kwcoco_dataset import KWCocoVideoDataset
    import geowatch
    import ubelt as ub

    # Ensure the two videos have different sizes
    # (one is much larger than the other)
    coco_dset1 = geowatch.coerce_kwcoco(
        'geowatch', num_videos=1, image_size=(8, 8), num_frames=3,
        geodata={
            'enabled': True,
            'region_geom': 'random-proportional',
            'target_gsd': 2.0
        })
    coco_dset2 = geowatch.coerce_kwcoco(
        'geowatch', num_videos=1, image_size=(64, 64), num_frames=3,
        geodata={
            'enabled': True,
            'region_geom': 'random-proportional',
            'target_gsd': 2.0
        })
    coco_dset = coco_dset1.union(coco_dset2)

    # Enable "dynamic fixed resolution" which should scale down the image
    # so there are a maximum number of windows
    dynamic_fixed_resolution = {
        'max_winspace_full_dims': (16, 16),
    }
    # dynamic_fixed_resolution = None
    self = KWCocoVideoDataset(coco_dset, time_dims=3,
                              window_dims=(4, 4), fixed_resolution='4.0GSD',
                              channels='r|g|b', autobuild=False, mode='test',
                              use_grid_cache=False,
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
    shape1 = sample1['frames'][0]['modes']['r|g|b'].shape
    shape2 = sample2['frames'][0]['modes']['r|g|b'].shape
    assert tuple(shape1) == tuple(shape2)

    scale1 = sample1['frames'][0]['scale_outspace_from_vid']
    scale2 = sample2['frames'][0]['scale_outspace_from_vid']
    import numpy as np
    assert np.all(scale1 < scale2)
