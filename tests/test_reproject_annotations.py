

def test_reproject_annotations_positive_pending():
    from geowatch.geoannots.geomodels import RegionModel, SiteModel
    import kwarray

    start_time = '2012-03-04'
    end_time = '2025-06-07'

    region = RegionModel.random(
        with_sites=False,
        rng=1089283,
        start_time=start_time,
        end_time=end_time,
        num_sites=0,
    )

    sites = []
    rng = kwarray.ensure_rng(60810907460)

    N = 10

    # Create a positive_pending annotation with a null end date
    for _ in range(N):
        site = SiteModel.random(
            region=region, num_observations=2, rng=rng,
            start_site_index=len(sites),
        )
        obs1, obs2 = list(site.observations())
        obs1['properties']['current_phase'] = None
        obs2['properties']['current_phase'] = None
        obs2['properties']['observation_date'] = None
        site.header['properties']['status'] = 'positive_pending'
        site.header['properties']['end_date'] = None
        sites.append(site)

    # Create a positive_pending annotation with a null start date
    for _ in range(N):
        site = SiteModel.random(
            region=region, num_observations=2, rng=rng,
            start_site_index=len(sites),
        )
        obs1, obs2 = list(site.observations())
        obs1['properties']['current_phase'] = None
        obs2['properties']['current_phase'] = None
        obs1['properties']['observation_date'] = None
        site.header['properties']['status'] = 'positive_pending'
        site.header['properties']['start_date'] = None
        sites.append(site)

    # Create a positive_pending annotation with null start and end dates
    for _ in range(N):
        site = SiteModel.random(
            region=region, num_observations=2, rng=rng,
            start_site_index=len(sites),
        )
        obs1, obs2 = list(site.observations())
        obs1['properties']['current_phase'] = None
        obs2['properties']['current_phase'] = None
        obs2['properties']['observation_date'] = None
        obs1['properties']['observation_date'] = None
        site.header['properties']['status'] = 'positive_pending'
        site.header['properties']['end_date'] = None
        site.header['properties']['start_date'] = None
        sites.append(site)

    # Create a positive_pending annotation with both dates
    for _ in range(N):
        site = SiteModel.random(
            region=region, num_observations=2, rng=rng,
            start_site_index=len(sites),
        )
        obs1, obs2 = list(site.observations())
        obs1['properties']['current_phase'] = None
        obs2['properties']['current_phase'] = None
        site.header['properties']['status'] = 'positive_pending'
        sites.append(site)

    summaries = [s.as_summary() for s in sites]
    region['features'].extend(summaries)

    summaries_df = region.pandas_summaries()
    print(summaries_df[['site_id', 'start_date', 'end_date']])

    # Create a dummy kwcoco file that is georegistered to the random region
    import geowatch
    dset = geowatch.coerce_kwcoco(
        'geowatch-msi', num_videos=1,
        geodata={'region_geom': region.geometry},
        dates={'start_time': start_time, 'end_time': end_time}
    )
    dset.clear_annotations()

    dset.images().lookup('date_captured')

    # print(f'dset.fpath={dset.fpath}')
    # gdalinfo /home/joncrall/.cache/geowatch/demo_kwcoco_bundles/watch_vidshapes_a86db771/_assets/auxiliary/aux_B10_B11/img_00010.tif
    # gdalinfo /home/joncrall/.cache/geowatch/demo_kwcoco_bundles/watch_vidshapes_5b7fdde0/_assets/auxiliary/aux_B10_B11/img_00010.tif

    from geowatch.cli import reproject_annotations
    import geowatch
    import ubelt as ub
    viz_dpath = ub.Path.appdir('geowatch/tests/test_reproject/viz').ensuredir()
    cmdline = False
    kwargs = {
        'src': dset,
        'dst': 'return',
        'viz_dpath': viz_dpath,
        'workers': 4,
        'io_workers': 8,
        'site_models': sites,
        'region_models': region,
    }
    new_dset = reproject_annotations.main(cmdline=cmdline, **kwargs)
    print(f'new_dset={new_dset}')

    # TODO: check assertions, but the viz looks ok
