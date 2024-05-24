def test_dynamic_resolution():
    from geowatch.tasks.fusion.datamodules.kwcoco_dataset import KWCocoVideoDataset
    import geowatch
    coco_dset1 = geowatch.coerce_kwcoco('geowatch', num_videos=1, image_size=(8, 8), num_frames=3, geodata=True)
    coco_dset2 = geowatch.coerce_kwcoco('geowatch', num_videos=1, image_size=(64, 64), num_frames=3, geodata=True)
    coco_dset = coco_dset1.union(coco_dset2)
    self = KWCocoVideoDataset(coco_dset, time_dims=4,
                              window_dims=(128, 128), fixed_resolution='0.3GSD',
                              channels='r|g|b', autobuild=False, mode='test',
                              use_grid_cache=False, dynamic_fixed_resolution=None)
    self._init()

    # TODO:
    # Ensure the two videos have different sizes (ideally one is much larger than the other)
    # Add an option to enable "dynamic resolution"
    # Ensure that we downsample the big video but keep the small video to some
    # target resolution at its regular resolution.
