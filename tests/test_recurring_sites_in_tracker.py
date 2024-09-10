

def test_tracker_time_split_thresh():
    """
    CommandLine:
        xdoctest tests/test_recurring_sites_in_tracker.py test_tracker_time_split_thresh
    """
    import json
    import os
    import kwcoco
    import ubelt as ub
    from geowatch.cli import run_tracker

    coco_dset = build_recurring_sites_coco()

    dpath = ub.Path.appdir('geowatch', 'test', 'tracking', 'recuring_sites').ensuredir()
    dpath.delete().ensuredir()

    regions_dir = dpath / 'regions/'
    bas_coco_fpath = dpath / 'bas_output.kwcoco.json'
    bas_fpath = dpath / 'bas_sites.json'

    viz_outdir = str((dpath / 'track_viz').ensuredir())

    # Run BAS
    bas_argv = [
        '--in_file', os.fspath(coco_dset.fpath),
        '--out_site_summaries_dir', str(regions_dir),
        '--out_site_summaries_fpath',  str(bas_fpath),
        '--out_kwcoco', str(bas_coco_fpath),
        '--track_fn', 'saliency_heatmaps',
        '--sensor_warnings', '0',
        '--viz_out_dir', os.fspath(viz_outdir),
        '--track_kwargs', json.dumps({
            'thresh': 0.3,
            'time_thresh': .8,
            'min_area_square_meters': None,
            'moving_window_size': 2,
            'max_area_square_meters': None,
            'polygon_simplify_tolerance': 1,
            'time_split_thresh': 0.5,
        }),
    ]
    run_tracker.main(argv=bas_argv)

    bas_coco_dset = kwcoco.CocoDataset(bas_coco_fpath)

    ann_trackids = bas_coco_dset.annots().lookup('track_id', None)
    ann_timestamps = bas_coco_dset.annots().images.lookup('date_captured', None)
    track_times = ub.group_items(ann_timestamps, ann_trackids)
    print('track_times = {}'.format(ub.urepr(track_times, nl=1)))
    assert len(track_times) == 3

    if 0:
        ub.cmd(f'geowatch visualize {bas_coco_dset.fpath} --stack=only --smart', system=1)


def build_recurring_sites_coco():
    """
    Build a simple test case where the site blips in and out every year.

    CommandLine:
        xdoctest tests/test_recurring_sites_in_tracker.py build_recurring_sites_coco
    """
    import geowatch
    import ubelt as ub
    import kwarray
    import kwimage
    import numpy as np
    from kwutil import util_time
    dpath = ub.Path.appdir('geowatch/tests/recurring_sites')
    geowatch.coerce_kwcoco('geowatch-msi')
    asset_dpath = (dpath / 'assets').ensuredir()

    rng = kwarray.ensure_rng(0)

    gh, gw = 512, 512
    num_frames = 50

    poly = kwimage.Polygon.random(rng=rng).scale((gw, gh))
    poly = poly.scale(0.3, about='centroid')

    start_time = util_time.coerce_datetime('2020-01-01')
    frame_rate = util_time.coerce_timedelta('1week')

    current_time = start_time

    blip_times = [4, 5, 6, 16, 17, 18, 40, 41, 42]

    from geowatch.demo.smart_kwcoco_demodata import _random_utm_box
    utm_box, utm_crs_info = _random_utm_box()
    auth = utm_crs_info['auth']
    assert auth[0] == 'EPSG'
    epsg_int = int(auth[1])
    ulx, uly, lrx, lry = utm_box.to_ltrb().data[0]

    import kwcoco
    coco_dset = kwcoco.CocoDataset()
    video_id = coco_dset.add_video('video1', width=gw, height=gh)

    for frame_idx in range(num_frames):
        # Write image to disk
        name = f'video_{video_id:03d}_frame_{frame_idx:03d}'
        fname = f'{name}.tif'
        gpath = asset_dpath / fname
        # canvas = np.zeros((gh, gw, 1), dtype=np.float32)
        canvas = rng.rand(gh, gw, 1).astype(np.float32) * 0.1
        if frame_idx in blip_times:
            # Draw blips only sometimes
            canvas = poly.fill(canvas, value=1)
        kwimage.imwrite(gpath, canvas, backend='gdal')
        command = f'gdal_edit.py -a_ullr {ulx} {uly} {lrx} {lry} -a_srs EPSG:{epsg_int} {gpath}'
        ub.cmd(command, shell=True, check=True)

        # Add base container for the image
        image = {
            'video_id': video_id,
            'name': name,
            'file_name': None,
            'sensor_coarse': 'S2',
            'date_captured': current_time.date().isoformat(),
            'frame_index': frame_idx,
            'width': gw,
            'height': gh,
        }
        gid = coco_dset.add_image(**image)
        # Use the coco image to add the asset
        coco_img = coco_dset.coco_image(gid)
        coco_img.add_asset(
            file_name=gpath,
            channels='salient',
            width=gw,
            height=gh
        )
        current_time += frame_rate

    from geowatch.utils import kwcoco_extensions
    # Do a consistent transfer of the hacked seeded geodata to the other images
    kwcoco_extensions.ensure_transfered_geo_data(coco_dset)
    kwcoco_extensions.coco_populate_geo_heuristics(coco_dset)
    kwcoco_extensions.warp_annot_segmentations_to_geos(coco_dset)
    kwcoco_extensions.populate_watch_fields(
        coco_dset, target_gsd=0.3, enable_valid_region=True,
        overwrite=True)

    coco_dset.fpath = dpath / 'test.kwcoco.zip'
    coco_dset.dump()

    if 0:
        ub.cmd(f'geowatch visualize {coco_dset.fpath} --stack=only --smart', system=1)

    return coco_dset

