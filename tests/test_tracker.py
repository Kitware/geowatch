def test_tracker_with_sv():
    """
    Tests tracker with site verification on
    """
    import pytest
    pytest.skip('slow')

    import json
    import kwcoco
    import pandas as pd
    import ubelt as ub
    from watch.cli import kwcoco_to_geojson
    from watch.utils import util_gis
    import watch

    coco_dset = watch.coerce_kwcoco('watch-msi', heatmap=True, geodata=True, dates=True)

    video0 = coco_dset.videos().objs[0]
    video0_images = coco_dset.images(video_id=video0['id'])
    assert len(video0_images) >= 10, 'should have a several frames'
    assert len(set(video0_images.lookup('date_captured'))) > 5, (
        'should be on different dates')

    #coco_dset = smart_kwcoco_demodata.demo_smart_aligned_kwcoco()
    dpath = ub.Path.appdir('watch', 'test', 'tracking', 'unit_test1').ensuredir()
    dpath.delete().ensuredir()

    coco_dset.reroot(absolute=True)
    coco_dset.fpath = dpath / 'bas_input.kwcoco.json'
    coco_dset.clear_annotations()
    coco_dset.dump(coco_dset.fpath, indent=2)

    regions_dir = dpath / 'regions/'
    bas_coco_fpath = dpath / 'bas_output.kwcoco.json'
    sc_coco_fpath = dpath / 'sc_output.kwcoco.json'
    sv_coco_fpath = dpath / 'sv_output.kwcoco.json'
    bas_fpath = dpath / 'bas_sites.json'
    sc_fpath = dpath / 'sc_sites.json'
    sv_fpath = dpath / 'sv_sites.json'
    # Run BAS
    bas_argv = [
        '--in_file', coco_dset.fpath,
        '--out_site_summaries_dir', str(regions_dir),
        '--out_site_summaries_fpath',  str(bas_fpath),
        '--out_kwcoco', str(bas_coco_fpath),
        '--track_fn', 'saliency_heatmaps',
        '--sensor_warnings', '0',
        '--track_kwargs', json.dumps({
            'thresh': 0.5,
            'time_thresh': .8,
            'min_area_square_meters': None,
            'max_area_square_meters': None,
            'polygon_simplify_tolerance': 1,
            # 'moving_window_size': 1,
        }),
    ]
    kwcoco_to_geojson.main(bas_argv)

    bas_coco_dset = kwcoco.CocoDataset(bas_coco_fpath)
    bas_trackids = bas_coco_dset.annots().lookup('track_id', None)
    assert len(bas_trackids) > len(set(bas_trackids)), (
        'should have multiple observations per track')

    # Run SC on the same dset
    sites_dir = dpath / 'sites'
    sc_argv = [
        '--in_file', coco_dset.fpath,
        '--out_sites_dir', str(sites_dir),
        '--out_sites_fpath', str(sc_fpath),
        '--out_kwcoco', str(sc_coco_fpath),
        '--track_fn', 'class_heatmaps',
        '--site_summary', str(bas_fpath),
        '--sensor_warnings', '0',
        '--track_kwargs', json.dumps({
            'thresh': 1e-9, 'min_area_square_meters': None,
            'max_area_square_meters': None,
            'polygon_simplify_tolerance': 1,
            'boundaries_as': 'polys',
            'key': ('salient',)
        }),
    ]
    kwcoco_to_geojson.main(sc_argv)

    # Run SV on the same dset
    sites_dir = dpath / 'sv_sites'
    sv_argv = [
        '--in_file', str(sc_coco_fpath),
        '--out_sites_dir', str(sites_dir),
        '--out_sites_fpath', str(sv_fpath),
        '--out_kwcoco', str(sv_coco_fpath),
        '--track_fn', 'site_validation',
        '--site_summary', str(bas_fpath),
        '--sensor_warnings', '0',
        '--track_kwargs', json.dumps({
            'thresh': 1e-9,
            'min_area_square_meters': None,
            'max_area_square_meters': None,
            'polygon_simplify_tolerance': 1,
            'key': 'salient',
            # 'thresh': 0
        }),
    ]
    kwcoco_to_geojson.main(sv_argv)

    # Check expected results
    sc_coco_dset = kwcoco.CocoDataset(sc_coco_fpath)

    sc_trackids = sc_coco_dset.annots().lookup('track_id', None)

    assert len(bas_trackids) and None not in bas_trackids
    assert len(sc_trackids) and None not in sc_trackids

    summaries = list(util_gis.coerce_geojson_datas(bas_fpath, format='dataframe'))
    sites = list(util_gis.coerce_geojson_datas(sc_fpath, format='dataframe'))

    sc_df = pd.concat([d['data'] for d in sites])
    bas_df = pd.concat([d['data'] for d in summaries])
    ssum_rows = bas_df[bas_df['type'] == 'site_summary']
    site_rows = sc_df[sc_df['type'] == 'site']
    obs_rows = sc_df[sc_df['type'] == 'observation']

    print(ub.codeblock(
        f'''
        len(site_rows) = {len(site_rows)}
        len(ssum_rows) = {len(ssum_rows)}
        len(obs_rows) = {len(obs_rows)}
        '''))
    assert len(site_rows) > 0
    assert len(ssum_rows) > 0
    assert len(ssum_rows) == len(site_rows), (
        'number of site summaries and site headers should always be equal'
    )
    assert len(obs_rows) > len(site_rows), (
        'we should have more than one observation per-track')

    # Cleanup
    # dpath.delete()


def test_tracker_bas_with_boundary_region():
    """
    Runs two variants of the tracker, one with region bounds on and another
    with region bounds off. We generate demo region in a way that should always
    result in some tracks being removed.

    CommandLine:
        pytest tests/test_tracker.py -k test_tracker_bas_with_boundary_region -s
    """

    from watch.demo.smart_kwcoco_demodata import random_inscribed_polygon
    import json
    import kwcoco
    import ubelt as ub
    from watch.cli import kwcoco_to_geojson
    import watch
    from watch.geoannots import geomodels
    from watch.geoannots.geococo_objects import CocoGeoVideo
    from watch.utils import util_gis
    import kwimage
    import kwarray

    coco_dset = watch.coerce_kwcoco('watch-msi', heatmap=True, geodata=True,
                                    dates=True, image_size=(96, 96))
    coco_dset.clear_annotations()

    rng = kwarray.ensure_rng(4329423)

    # Make some region models for this dataset
    import geopandas as gpd
    region_models = []
    crs84 = util_gis.get_crs84()

    video_name_to_crs84_bounds = {}

    for video in coco_dset.videos().objs:
        coco_video = CocoGeoVideo(video=video, dset=coco_dset)

        dates = coco_video.images.lookup('date_captured')
        start_time = min(dates)
        end_time = max(dates)

        utm_part = coco_video.wld_corners_gdf
        utm_poly = kwimage.Polygon.coerce(utm_part.iloc[0]['geometry'])
        # Make a random inscribed polygon to use as the test region
        utm_region_poly = random_inscribed_polygon(utm_poly, rng=rng)

        # Shrink it so we are more likely to remove annotations
        utm_region_poly = utm_region_poly.scale(0.5, about='centroid')

        crs84_region_poly = kwimage.Polygon.coerce(gpd.GeoDataFrame(
            geometry=[utm_region_poly],
            crs=utm_part.crs).to_crs(crs84).iloc[0]['geometry'])

        video_name_to_crs84_bounds[coco_video['name']] = crs84_region_poly

        region_model = geomodels.RegionModel.random(
            region_id=coco_video['name'], region_poly=crs84_region_poly,
            rng=rng, start_time=start_time, end_time=end_time)
        region_models.append(region_model)

    video0 = coco_dset.videos().objs[0]
    video0_images = coco_dset.images(video_id=video0['id'])
    assert len(video0_images) >= 10, 'should have a several frames'
    assert len(set(video0_images.lookup('date_captured'))) > 5, (
        'should be on different dates')

    dpath = ub.Path.appdir('watch', 'test', 'tracking', 'unit_test2').ensuredir()
    dpath.delete().ensuredir()

    dpath1 = (dpath / 'with_boundary_region').ensuredir()
    dpath2 = (dpath / 'without_boundary_region').ensuredir()

    # Write region models to disk
    region_models_dpath = (dpath1 / 'region_models').ensuredir()
    for region_model in region_models:
        region_fpath = region_models_dpath / (region_model.region_id + '.geojson')
        region_fpath.write_text(region_model.dumps(indent=4))

    track_kwargs = {
        'thresh': 0.5,
        'time_thresh': .8,
        'min_area_square_meters': None,
        'max_area_square_meters': None,
        'polygon_simplify_tolerance': 1,
        # 'moving_window_size': 1,
    }

    # Run BAS with region bounds on (1)
    coco_dset.reroot(absolute=True)
    in_coco_fpath1 = dpath1 / 'bas_input.kwcoco.json'
    coco_dset.dump(in_coco_fpath1, indent=2)
    regions_dir1 = dpath1 / 'regions/'
    bas_coco_fpath1 = dpath1 / 'bas_output.kwcoco.json'
    bas_fpath1 = dpath1 / 'bas_sites.json'
    bas_argv1 = [
        '--in_file', in_coco_fpath1,
        '--out_site_summaries_dir', str(regions_dir1),
        '--out_site_summaries_fpath',  str(bas_fpath1),
        '--out_kwcoco', str(bas_coco_fpath1),
        '--track_fn', 'saliency_heatmaps',
        '--boundary_region', region_models_dpath,
        '--track_kwargs', json.dumps(track_kwargs),
        '--sensor_warnings', '0',
    ]
    kwcoco_to_geojson.main(bas_argv1)

    # Run BAS with region bounds off (2)
    in_coco_fpath2 = dpath2 / 'bas_input.kwcoco.json'
    coco_dset.dump(in_coco_fpath2, indent=2)
    regions_dir2 = dpath2 / 'regions/'
    bas_coco_fpath2 = dpath2 / 'bas_output.kwcoco.json'
    bas_fpath2 = dpath2 / 'bas_sites.json'
    bas_argv2 = [
        '--in_file', in_coco_fpath2,
        '--out_site_summaries_dir', str(regions_dir2),
        '--out_site_summaries_fpath',  str(bas_fpath2),
        '--out_kwcoco', str(bas_coco_fpath2),
        '--track_fn', 'saliency_heatmaps',
        '--track_kwargs', json.dumps(track_kwargs),
        '--sensor_warnings', '0',
    ]
    kwcoco_to_geojson.main(bas_argv2)

    bas_coco_dset1 = kwcoco.CocoDataset(bas_coco_fpath1)
    bas_coco_dset2 = kwcoco.CocoDataset(bas_coco_fpath2)

    bas_trackids = bas_coco_dset1.annots().lookup('track_id', None)
    assert len(bas_trackids) > len(set(bas_trackids)), (
        'should have multiple observations per track')

    num_oob_tracks_in_dset1 = 0
    num_oob_tracks_in_dset2 = 0
    import numpy as np
    from shapely.ops import unary_union
    from shapely.geometry import shape
    for video_id in bas_coco_dset1.videos():
        video_name1 = bas_coco_dset1.index.videos[video_id]['name']
        video_name2 = bas_coco_dset1.index.videos[video_id]['name']
        assert video_name1 == video_name2, 'dsets should agree'

        region_bounds = video_name_to_crs84_bounds[video_name1].to_shapely()

        video_annots1 = bas_coco_dset1.images(video_id=video_id).annots
        video_annots2 = bas_coco_dset2.images(video_id=video_id).annots

        annots1 = bas_coco_dset1.annots(list(ub.flatten(video_annots1)))
        annots2 = bas_coco_dset2.annots(list(ub.flatten(video_annots2)))

        assert len(annots1) <= len(annots2), (
            'boundaries should strictly remove annots')

        trackid_to_geoms1 = ub.group_items(annots1.lookup('segmentation_geos'), annots1.lookup('track_id'))
        trackid_to_geoms2 = ub.group_items(annots2.lookup('segmentation_geos'), annots2.lookup('track_id'))

        inbound_flags1 = []
        for tid1, geoms1 in trackid_to_geoms1.items():
            if len(geoms1):
                track_poly1 = unary_union(list(map(shape, geoms1)))
                inbound_flag1 = region_bounds.intersects(track_poly1)
                inbound_flags1.append(inbound_flag1)

        inbound_flags2 = []
        for tid2, geoms2 in trackid_to_geoms2.items():
            if len(geoms2):
                track_poly2 = unary_union(list(map(shape, geoms2)))
                inbound_flag2 = region_bounds.intersects(track_poly2)
                inbound_flags2.append(inbound_flag2)

        assert all(inbound_flags1), (
            'the region_bounds version should never have a fully oob track')

        num_oob_tracks_in_dset1 += (1 - np.array(inbound_flags1, dtype=int)).sum()
        num_oob_tracks_in_dset2 += (1 - np.array(inbound_flags2, dtype=int)).sum()

    assert num_oob_tracks_in_dset1 == 0
    if num_oob_tracks_in_dset2 == 0:
        raise AssertionError(
            'This test should have been written such that some of the generated '
            'tracks should have been out of bounds. However, if the demodata changes'
            'that assumption may break. This error could be a warning if it fails, '
            'but then the tests should be fixed to ensure there are oob sites '
            'being removed'
        )


def test_tracker_nan_params():
    """
    Test that nan params are properly handled
    """
    import json
    import kwcoco
    import ubelt as ub
    from watch.cli import kwcoco_to_geojson
    import watch

    coco_dset = watch.coerce_kwcoco('watch-msi', heatmap=True, geodata=True, dates=True)

    video0 = coco_dset.videos().objs[0]
    video0_images = coco_dset.images(video_id=video0['id'])
    assert len(video0_images) >= 10, 'should have a several frames'
    assert len(set(video0_images.lookup('date_captured'))) > 5, (
        'should be on different dates')

    #coco_dset = smart_kwcoco_demodata.demo_smart_aligned_kwcoco()
    dpath = ub.Path.appdir('watch', 'test', 'tracking', 'unit_test1').ensuredir()
    dpath.delete().ensuredir()

    coco_dset.reroot(absolute=True)
    coco_dset.fpath = dpath / 'bas_input.kwcoco.json'
    coco_dset.clear_annotations()
    coco_dset.dump(coco_dset.fpath, indent=2)

    regions_dir = dpath / 'regions/'
    bas_coco_fpath = dpath / 'bas_output.kwcoco.json'
    bas_fpath = dpath / 'bas_sites.json'
    # Run BAS
    bas_argv = [
        '--in_file', coco_dset.fpath,
        '--out_site_summaries_dir', str(regions_dir),
        '--out_site_summaries_fpath',  str(bas_fpath),
        '--out_kwcoco', str(bas_coco_fpath),
        '--track_fn', 'saliency_heatmaps',
        '--sensor_warnings', '0',
        '--track_kwargs', json.dumps({
            'thresh': 0.5,
            'time_thresh': .8,
            'min_area_square_meters': None,
            'moving_window_size': None,
            'max_area_square_meters': None,
            'polygon_simplify_tolerance': 1,
        }),
    ]
    kwcoco_to_geojson.main(bas_argv)

    bas_coco_dset = kwcoco.CocoDataset(bas_coco_fpath)
    bas_trackids = bas_coco_dset.annots().lookup('track_id', None)
    assert len(bas_trackids) > len(set(bas_trackids)), (
        'should have multiple observations per track')
