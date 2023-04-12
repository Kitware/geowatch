def test_tracker():
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


def random_inscribed_polygon(bounding_polygon, rng=None):
    """
        if 1:
            import kwplot
            kwplot.plt.ion()
            bounding_box.draw(facecolor='blue', alpha=0.8, setlim=1, fill=True, edgecolor='darkblue')
            utm_poly.draw(facecolor='orange', alpha=0.8, setlim=1, fill=True, edgecolor='darkorange')
            rando_utm.draw(facecolor='green', alpha=0.8, setlim=1, fill=True, edgecolor='darkgreen')
            inscribed_utm.draw(facecolor='red', alpha=0.8, setlim=1, fill=True, edgecolor='darkred')
    """
    import kwimage
    # Make a random polygon inscribed in the utm region
    bounding_box = kwimage.Box(bounding_polygon.bounding_box())
    rano_01 = kwimage.Polygon.random(tight=1, rng=rng)
    # Move to the origin, scale to match the box, and then move to the center
    # of the polygon of interest.
    rando = rano_01.translate((-.5, -.5)).scale((
        bounding_box.width, bounding_box.height)).translate(
            bounding_polygon.centroid)
    # Take the intersection ito inscribe
    inscribed = rando.intersection(bounding_polygon)
    return inscribed


def test_tracker_bas_with_boundary_region():
    """
    xdoctest ~/code/watch/tests/test_tracker.py test_tracker_bas_with_boundary_region
    """
    import json
    import kwcoco
    import ubelt as ub
    from watch.cli import kwcoco_to_geojson
    import watch
    from watch.geoannots import geomodels
    from watch.geoannots.geococo_objects import CocoGeoVideo
    from watch.utils import util_gis
    crs84 = util_gis.get_crs84()
    import kwimage
    import kwarray

    coco_dset = watch.coerce_kwcoco('watch-msi', heatmap=True, geodata=True,
                                    dates=True, image_size=(96, 96))

    rng = kwarray.ensure_rng(4329423)

    # Make some region models for this dataset
    import geopandas as gpd
    # crs84_parts = []
    region_models = []
    for video in coco_dset.videos().objs:
        coco_video = CocoGeoVideo(video=video, dset=coco_dset)

        dates = coco_video.images.lookup('date_captured')
        start_time = min(dates)
        end_time = max(dates)

        utm_part = coco_video.wld_corners_gdf
        utm_poly = kwimage.Polygon.coerce(utm_part.iloc[0]['geometry'])
        # Make a random inscribed polygon to use as the test region
        utm_region_poly = random_inscribed_polygon(utm_poly, rng=rng)

        crs84_region_poly = kwimage.Polygon.coerce(gpd.GeoDataFrame(
            geometry=[utm_region_poly],
            crs=utm_part.crs).to_crs(crs84).iloc[0]['geometry'])

        region_model = geomodels.RegionModel.random(
            region_id=coco_video['name'], region_poly=crs84_region_poly,
            rng=rng, start_time=start_time, end_time=end_time)
        region_models.append(region_model)
        # crs84_part = utm_part.to_crs(crs84)
        # crs84_parts.append(crs84_part)

    # import pandas as pd
    # video_gdf = pd.concat(crs84_parts)

    video0 = coco_dset.videos().objs[0]
    video0_images = coco_dset.images(video_id=video0['id'])
    assert len(video0_images) >= 10, 'should have a several frames'
    assert len(set(video0_images.lookup('date_captured'))) > 5, (
        'should be on different dates')

    dpath = ub.Path.appdir('watch', 'test', 'tracking', 'unit_test2').ensuredir()
    dpath.delete().ensuredir()

    region_models_dpath = (dpath / 'region_models').ensuredir()

    # Write region models to disk
    for region_model in region_models:
        region_fpath = region_models_dpath / (region_model.region_id + '.geojson')
        region_fpath.write_text(region_model.dumps(indent=4))

    coco_dset.reroot(absolute=True)
    coco_dset.fpath = dpath / 'bas_input.kwcoco.json'
    coco_dset.clear_annotations()
    coco_dset.dump(coco_dset.fpath, indent=2)

    # region_id = 'dummy_region'

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
        '--boundary_region', region_models_dpath,
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

    # sc_coco_fpath = dpath / 'sc_output.kwcoco.json'
    # sv_coco_fpath = dpath / 'sv_output.kwcoco.json'
    # sc_fpath = dpath / 'sc_sites.json'
    # sv_fpath = dpath / 'sv_sites.json'
