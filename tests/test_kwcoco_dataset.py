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

    # Disable test for the disussed reason. FIXME.
    if False:
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


def distance_weights():
    from geowatch.tasks.fusion.datamodules.kwcoco_dataset import KWCocoVideoDataset
    import ubelt as ub
    # Demo toy data without augmentation
    import kwcoco
    import numpy as np
    #src = geowatch.coerce_kwcoco('geowatch-msi', geodata=True, dates=True)
    coco_dset = kwcoco.CocoDataset.demo('vidshapes2-multispectral', num_frames=10)
    channels = 'B10,B8a|B1,B8'
    self = KWCocoVideoDataset(coco_dset, time_dims=4, window_dims=(300, 300),
                              channels=channels,
                              input_space_scale='native',
                              output_space_scale=None,
                              window_space_scale=1.2,
                              augment_space_shift_rate=0.5,
                              use_grid_negatives=False,
                              use_grid_positives=False,
                              use_centered_positives=True,
                              absolute_weighting=True,
                              time_sampling='uniform',
                              time_kernel='-1year,0,1month,1year',
                              modality_dropout=0.5,
                              channel_dropout=0.5,
                              temporal_dropout=0.7,
                              temporal_dropout_rate=1.0)
    # Add weights to annots
    annots = self.sampler.dset.annots()
    annots.set('weight', 2 + np.random.rand(len(annots)) * 10)
    self.disable_augmenter = False
    # Summarize batch item in text
    index = self.sample_grid['targets'][self.sample_grid['positives_indexes'][3]]
    item = self[index]
    summary = self.summarize_item(item)
    print('item summary: ' + ub.urepr(summary, nl=2))
    # Draw batch item
    canvas = self.draw_item(item, draw_weights=True)
    # xdoctest: +REQUIRES(--show)
    import kwplot
    kwplot.autompl()
    kwplot.imshow(canvas)
    kwplot.show_if_requested()


def test_msi_auto_channels():
    from geowatch.tasks.fusion.datamodules.kwcoco_dataset import KWCocoVideoDataset
    import kwcoco
    import ubelt as ub

    coco_msi = kwcoco.CocoDataset.demo('vidshapes2-msi', num_frames=1)
    coco_rgb = kwcoco.CocoDataset.demo('vidshapes2', num_frames=1)

    msi_spec = coco_msi.coco_image(1).channels.spec
    assert 'B1' in msi_spec, 'kwcoco should provide this'
    assert 'B8' in msi_spec, 'kwcoco should provide this'
    assert 'B8a' in msi_spec, 'kwcoco should provide this'

    sample_kwargs = {
        'time_dims': 1,
        'window_dims': (128, 128)
    }

    datasets = {}
    # Check that regular RGB works fine
    datasets['rgb_auto'] = KWCocoVideoDataset(coco_rgb, **sample_kwargs)
    datasets['rgb_explicit'] = KWCocoVideoDataset(coco_rgb, **sample_kwargs,
                                                  channels='r|g|b')
    datasets['msi_explicit'] = KWCocoVideoDataset(coco_msi, **sample_kwargs,
                                                  channels='B10,B8a|B1,B8')
    datasets['msi_auto'] = KWCocoVideoDataset(coco_msi, **sample_kwargs)

    if 0:
        # First check that the matching sensorchans does what we expect
        # This was broken in delayed image 0.4.2 and fixed in 0.4.3
        datasets['msi_explicit'].sample_sensorchan.matching_sensor('sensor1')

    results = []
    errors = []
    for key, dataset in datasets.items():
        row = {'key': key}
        try:
            item = dataset.getitem(0)
        except Exception as ex:
            row['status'] = 'fail'
            row['ex'] = str(ex)[0:16]
            errors.append(row)
            raise
        else:
            row['item'] = str(item)[0:32]
            row['status'] = 'pass'
        results.append(row)

    if errors:
        import pandas as pd
        import rich
        df = pd.DataFrame(results)
        rich.print(df.to_string())
        for key, dataset in datasets.items():
            print('----')
            print(f'key={key}')
            print(f'dataset.sensorchan        = {ub.urepr(dataset.sensorchan, nl=1)}')
            print(f'dataset.sample_sensorchan = {ub.urepr(dataset.sample_sensorchan, nl=1)}')
            print(f'dataset.input_sensorchan  = {ub.urepr(dataset.input_sensorchan, nl=1)}')
        raise AssertionError('Problems in dataset getitems, maybe something to do with sensorchan configs?')


if __name__ == "__main__":
    distance_weights()
