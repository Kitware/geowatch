#!/usr/bin/env python3
# PYTHON_ARGCOMPLETE_OK
import scriptconfig as scfg
import warnings
import ubelt as ub


class SiteFilterConfig(scfg.DataConfig):

    min_area_square_meters = scfg.Value(None, help=ub.paragraph(
        '''
        If specified, any site with an area less than this threshold is
        removed.
        '''))

    max_area_square_meters = scfg.Value(None, help=ub.paragraph(
        '''
        If specified, any site with an area greater than this threshold is
        removed.
        '''))

    apply_clip_to = scfg.Value('polygon', help=ub.paragraph(
        '''
        The type of geometry that clipping is applied to.
        If "none", then this is not applied.
        If "all", then all polygons are clipped (which could cause multipolygons)
        If "polygon", then only apply clipping if it does not create a
        multipolygon.
        '''))

    apply_bounds_filter_to = scfg.Value('multipolygon', help=ub.paragraph(
        '''
        The type of geometry that the bounds filter is applied to.
        If "none", then this is not applied.
        If "all", then all polygons are have the bounds filter applied.
        If "multipolygon", only polygons clipped into multiple parts will have
        this applied.
        '''))

    in_bounds_thresh = scfg.Value(0.6, help=ub.paragraph(
        '''
        For polygons the bounds filter is applied to, this is the fraction of
        the site geometry that must be in bounds to keep it otherwise it is
        removed.
        '''))


class CropSitesToRegionsConfig(SiteFilterConfig):
    r"""
    Crops site models to the bounds of a region model.

    TODO:
        - [ ] Rename this to ClipSitesToRegions?

    Example:
        DVC_DPATH=$(WATCH_PREIMPORT=none python -m geowatch.cli.find_dvc)
        WATCH_PREIMPORT=none python -m geowatch.cli.crop_sites_to_regions \
            --site_models "$DVC_DPATH/annotations/site_models/KR_R002_*.geojson" \
            --region_models "$DVC_DPATH/annotations/region_models/KR_R002.geojson" \
            --new_site_dpath ./cropped_sites
    """
    site_models = scfg.Value(None, help=ub.paragraph(
        '''
        Geospatial geojson "site" annotation files. Either a path to a
        file, or a directory.
        '''))

    region_models = scfg.Value(None, help=ub.paragraph(
        '''
        A single geojson "region" file to crop to.
        '''))

    new_site_dpath = scfg.Value(None, help=ub.paragraph(
        '''
        Destination directory for new site models.
        Note: names of files must be unique.
        '''))

    new_region_dpath = scfg.Value(None, help=ub.paragraph(
        '''
        Destination directory for new site models.
        Note: names of files must be unique.
        '''))

    io_workers = scfg.Value(0, help=ub.paragraph(
        '''
        IO workers to load sites in the background while others are
        cropping.
        '''))

    force_multipolygon = scfg.Value(True, help=ub.paragraph(
        '''
        For output site observations the output geometry type will
        be set to MultiPolygon.  As per the T&E specification
        '''))


USE_LISTS = 0  # turn on for eager debugging


def main(cmdline=False, **kwargs):
    """

    CommandLine:
        xdoctest -m geowatch.cli.crop_sites_to_regions main:0
        xdoctest -m geowatch.cli.crop_sites_to_regions main:1

    Example:
        >>> from geowatch.geoannots import geomodels
        >>> import kwimage
        >>> region = geomodels.RegionModel.random(num_sites=0)
        >>> # Create several clipping cases
        >>> region_poly = kwimage.Polygon.coerce(region.geometry)
        >>> width = region_poly.to_box().width
        >>> height = region_poly.to_box().height
        >>> geoms = {}
        >>> geoms['in_bounds'] = region_poly.scale(0.1, about='centroid')
        >>> geoms['half_oob'] = region_poly.translate((width / 2, 0)).scale(0.5, about='centroid')
        >>> geoms['some_oob'] = region_poly.translate((-width / 2, -height / 2)).scale(0.5, about='centroid').translate(width / 4, height / 4)
        >>> geoms['fully_oob'] = region_poly.translate((width * 2, 0))
        >>> sites = {}
        >>> for key, poly in geoms.items():
        >>>     sites[key] = geomodels.SiteModel.random(region=region, site_poly=poly)
        >>>     region.add_site_summary(sites[key].as_summary())
        >>> # Write demo data to disk
        >>> dpath = ub.Path.appdir('geowatch/tests/cli/crop_sites_to_regions/doctest0')
        >>> dpath.delete().ensuredir()
        >>> region_dpath = (dpath / 'region_models').ensuredir()
        >>> site_dpath = (dpath / 'site_models').ensuredir()
        >>> region_fpath = region_dpath / 'region.geojson'
        >>> region_fpath.write_text(region.dumps())
        >>> for k, site in sites.items():
        >>>     site_fpath = site_dpath / f'{k}.geojson'
        >>>     site_fpath.write_text(site.dumps())
        >>> kwargs = {
        >>>     'site_models': site_dpath,
        >>>     'region_models': region_dpath,
        >>>     'new_site_dpath': dpath / 'new_site_models',
        >>>     'new_region_dpath': dpath / 'new_region_models',
        >>> }
        >>> from geowatch.cli import crop_sites_to_regions
        >>> cmdline = 0
        >>> crop_sites_to_regions.main(cmdline=cmdline, **kwargs)
        >>> new_region = geomodels.RegionModel.coerce(dpath / 'new_region_models')
        >>> new_sites = list(geomodels.SiteModel.coerce_multiple(dpath / 'new_site_models'))
        >>> # xdoctest: +REQUIRES(--show)
        >>> # xdoctest: +REQUIRES(module:kwplot)
        >>> import kwplot
        >>> kwplot.plt.ion()
        >>> ax = kwplot.figure(doclf=True, fnum=2, pnum=(2, 2, 1), title='Sites Before Clip').gca()
        >>> df = region.pandas_region()
        >>> ax = df.plot(edgecolor='black', facecolor=(0.1, 0.8, 0.1, 0.5), ax=ax)
        >>> for site in sites.values():
        >>>     df = site.pandas()
        >>>     ax = df.plot(edgecolor='black', facecolor=(0.1, 0.1, 0.8, 0.5), ax=ax)
        >>> ax = kwplot.figure(fnum=2, pnum=(2, 2, 2), title='Sites After Clip').gca()
        >>> df = new_region.pandas_region()
        >>> ax = df.plot(edgecolor='black', facecolor=(0.1, 0.8, 0.1, 0.5), ax=ax)
        >>> for site in new_sites:
        >>>     df = site.pandas()
        >>>     ax = df.plot(edgecolor='black', facecolor=(0.1, 0.1, 0.8, 0.5), ax=ax)
        >>> ax = kwplot.figure(fnum=2, pnum=(2, 2, 3), title='Region Before Clip').gca()
        >>> df = region.pandas()
        >>> ax = df.plot(edgecolor='black', facecolor=(0.1, 0.8, 0.1, 0.5), ax=ax)
        >>> ax = kwplot.figure(fnum=2, pnum=(2, 2, 4), title='Region After Clip').gca()
        >>> df = new_region.pandas()
        >>> ax = df.plot(edgecolor='black', facecolor=(0.1, 0.8, 0.1, 0.5), ax=ax)

    Example:
        >>> # Convex clipping case
        >>> from geowatch.geoannots import geomodels
        >>> import kwimage
        >>> star = kwimage.Polygon.star()
        >>> p1 = kwimage.Polygon.circle(xy=(0, 0), r=1)
        >>> p2 = kwimage.Polygon.circle(xy=(0.2, 0), r=1)
        >>> p3 = p1.difference(p2).translate(0.3)
        >>> box = kwimage.Box.coerce([-1, .3, 5, 5], format='xywh').to_polygon()
        >>> p3 = p3.difference(box)
        >>> box = kwimage.Box.coerce([-.1, -1, 10, 10], format='xywh').to_polygon()
        >>> p3 = p3.difference(box)
        >>> p3 = p3.difference(kwimage.Polygon.circle(xy=(-.6, -.2), r=0.15))
        >>> region = geomodels.RegionModel.random(region_poly=star, num_sites=0, rng=21)
        >>> region_poly = kwimage.Polygon.coerce(region.geometry)
        >>> width = region_poly.to_box().width
        >>> height = region_poly.to_box().height
        >>> geoms = {}
        >>> geoms['in_bounds'] = region_poly.scale(0.1, about='centroid')
        >>> geoms['half_oob'] = region_poly.translate((width / 2, 0))
        >>> geoms['some_oob'] = region_poly.translate((width / 2, height / 2)).scale(0.5, about='centroid').translate(-width / 3, -height / 3)
        >>> geoms['fully_oob'] = region_poly.translate((width * 2, 0))
        >>> geoms['tiny_oob'] = kwimage.Polygon.circle(xy=(-.20, .27), r=0.1)
        >>> geoms['sliver'] = p3
        >>> sites = {}
        >>> for key, poly in geoms.items():
        >>>     sites[key] = geomodels.SiteModel.random(region=region, site_poly=poly)
        >>>     region.add_site_summary(sites[key].as_summary())
        >>> # Write demo data to disk
        >>> dpath = ub.Path.appdir('geowatch/tests/cli/crop_sites_to_regions/doctest0')
        >>> dpath.delete().ensuredir()
        >>> region_dpath = (dpath / 'region_models').ensuredir()
        >>> site_dpath = (dpath / 'site_models').ensuredir()
        >>> region_fpath = region_dpath / 'region.geojson'
        >>> region_fpath.write_text(region.dumps())
        >>> for k, site in sites.items():
        >>>     site_fpath = site_dpath / f'{k}.geojson'
        >>>     site_fpath.write_text(site.dumps())
        >>> kwargs = {
        >>>     'site_models': site_dpath,
        >>>     'region_models': region_dpath,
        >>>     'new_site_dpath': dpath / 'new_site_models',
        >>>     'new_region_dpath': dpath / 'new_region_models',
        >>>     'min_area_square_meters': 5e8,
        >>> }
        >>> from geowatch.cli import crop_sites_to_regions
        >>> cmdline = 0
        >>> crop_sites_to_regions.main(cmdline=cmdline, **kwargs)
        >>> new_region = geomodels.RegionModel.coerce(dpath / 'new_region_models')
        >>> new_sites = list(geomodels.SiteModel.coerce_multiple(dpath / 'new_site_models'))
        >>> assert len(new_sites) == 2
        >>> assert len(sites) == 6
        >>> # xdoctest: +REQUIRES(--show)
        >>> # xdoctest: +REQUIRES(module:kwplot)
        >>> import kwplot
        >>> kwplot.plt.ion()
        >>> ax = kwplot.figure(doclf=True, fnum=2, pnum=(2, 2, 1), title='Observations Before Clip').gca()
        >>> df = region.pandas_region()
        >>> ax = df.plot(edgecolor='black', facecolor=(0.1, 0.8, 0.1, 0.5), ax=ax)
        >>> for site in sites.values():
        >>>     df = site.pandas_observations()
        >>>     ax = df.plot(edgecolor='black', facecolor=(0.1, 0.1, 0.8, 0.5), ax=ax)
        >>> ax = kwplot.figure(fnum=2, pnum=(2, 2, 2), title='Observations After Clip').gca()
        >>> df = new_region.pandas_region()
        >>> ax = df.plot(edgecolor='black', facecolor=(0.1, 0.8, 0.1, 0.5), ax=ax)
        >>> for site in new_sites:
        >>>     df = site.pandas_observations()
        >>>     ax = df.plot(edgecolor='black', facecolor=(0.1, 0.1, 0.8, 0.5), ax=ax)
        >>> ax = kwplot.figure(fnum=2, pnum=(2, 2, 3), title='Site Summary Before Clip').gca()
        >>> df = region.pandas()
        >>> ax = df.plot(edgecolor='black', facecolor=(0.1, 0.8, 0.1, 0.5), ax=ax)
        >>> ax = kwplot.figure(fnum=2, pnum=(2, 2, 4), title='Site Summary After Clip').gca()
        >>> df = new_region.pandas()
        >>> ax = df.plot(edgecolor='black', facecolor=(0.1, 0.8, 0.1, 0.5), ax=ax)
    """
    from geowatch.utils import util_gis
    from shapely.geometry import MultiPolygon
    import geopandas as gpd
    import safer

    config = CropSitesToRegionsConfig.cli(data=kwargs, cmdline=cmdline,
                                          strict=True)
    print('config = {}'.format(ub.urepr(dict(config), nl=1)))

    # TODO: integrate process context
    # from kwcoco.util.util_json import ensure_json_serializable
    # from geowatch.utils import process_context
    # proc_context = process_context.ProcessContext(
    #     name='crop_sites_to_regions',
    #     type='process',
    #     config=ensure_json_serializable(dict(config))
    # )
    # proc_context.start()
    # process_info = proc_context.obj

    new_site_dpath = config['new_site_dpath']
    assert new_site_dpath is not None, 'must specify new_site_dpath'
    new_site_dpath = ub.Path(new_site_dpath)

    new_region_dpath = config['new_region_dpath']
    assert new_region_dpath is not None, 'must specify new_region_dpath'
    new_region_dpath = ub.Path(new_region_dpath)

    site_geojson_fpaths: list[ub.Path] = util_gis.coerce_geojson_paths(
        config.get('site_models'))

    region_geojson_fpaths: list[ub.Path] = util_gis.coerce_geojson_paths(
        config.get('region_models'))

    # Load a single region
    if len(region_geojson_fpaths) != 1:
        raise ValueError(f'Must specify exactly one region file, Got: {region_geojson_fpaths}')

    regions = list(util_gis.load_geojson_datas(
        region_geojson_fpaths, workers=0, desc='load geojson region-models'))
    old_region_fpath = regions[0]['fpath']
    region_gdf_crs84: gpd.GeoDataFrame = regions[0]['data']

    # Load multiple site models
    io_workers = config['io_workers']
    sites = util_gis.load_geojson_datas(
        site_geojson_fpaths, workers=io_workers,
        desc='load geojson site-models')

    if USE_LISTS:
        sites = list(sites)

    filter_config = SiteFilterConfig(**(ub.udict(config) & SiteFilterConfig.__default__))

    cropped_region, cropped_sites = filter_sites(
        region_gdf_crs84, sites, filter_config)

    if USE_LISTS:
        cropped_sites = ub.ProgIter(
            cropped_sites, desc='Cropping sites',
            total=len(site_geojson_fpaths))
        cropped_sites = list(cropped_sites)

    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', 'pandas.Int64Index is deprecated', FutureWarning)

        # Save the cropped site summaries to disk
        new_region_dpath.ensuredir()
        new_region_fpath = new_region_dpath / old_region_fpath.name

        # Not sure why this insists on bytes. I dont think it was before
        with safer.open(new_region_fpath, temp_file=not ub.WIN32, mode='w') as file:
            cropped_region_json = cropped_region.to_json(na='drop', indent=2, drop_id=True)
            file.write(cropped_region_json)
        print(f'Wrote cropped site summaries to {new_region_fpath}')

        # Save the cropped sites to disk
        new_site_dpath.ensuredir()
        num_valid = 0
        total = 0
        for cropped_site_info in cropped_sites:
            total += 1
            cropped_site = cropped_site_info['data']
            if len(cropped_site):
                num_valid += 1
                old_site_fpath = cropped_site_info['fpath']
                new_site_fpath = new_site_dpath / old_site_fpath.name
                cropped_site['observation_date'] = cropped_site['observation_date'].astype('string')
                cropped_site['start_date'] = cropped_site['start_date'].astype('string')
                cropped_site['end_date'] = cropped_site['end_date'].astype('string')

                if config['force_multipolygon']:
                    cropped_site.loc[cropped_site['type'] == 'observation', 'geometry'] =\
                        cropped_site[cropped_site['type'] == 'observation']['geometry'].apply(
                            lambda x: MultiPolygon((x,)) if not isinstance(x, MultiPolygon) else x)

                if 'predicted_phase_transition_date' in cropped_site:
                    cropped_site['predicted_phase_transition_date'] =\
                        cropped_site['predicted_phase_transition_date'].astype('string')

                with safer.open(new_site_fpath, temp_file=not ub.WIN32, mode='w') as file:
                    cropped_site_json = cropped_site.to_json(
                        na='drop', indent=2, drop_id=True)
                    file.write(cropped_site_json)
                    # cropped_site.to_file(file, driver='GeoJSON')
        print(f'Wrote {num_valid} / {total} valid cropped sites in {new_site_dpath}')


def filter_sites(region_gdf_crs84, sites, filter_config=None):
    """
    Args:
        region_gdf_crs84 (GeoDataFrame):
            the region GDF containing the region geom to crop to and
            the site summary geometry

        sites (Iterable[Dict]):
            List of the loaded geo data frames with a 'data' key
            and the file path in the 'fpath' key.

        filter_config (SiteFilterConfig | None):
            modifies filter behavior.

    Returns:
        Tuple[GeoDataFrame, Iterable[Dict]]:
            Region model with cropped site summaries and a list of site info
            dictionaries containing the new cropped data field.

    Example:
        >>> from geowatch.cli.crop_sites_to_regions import *  # NOQA
        >>> import geopandas as gpd
        >>> import kwimage
        >>> from geowatch.utils import util_gis
        >>> crs84 = util_gis._get_crs84()
        >>> region_poly = kwimage.Polygon.random(rng=0).translate((42, 72))
        >>> site_poly1 = region_poly.translate((0.0001, 0.0001))
        >>> #
        >>> def demo_site_summary(site_id, site_poly):
        >>>     return {
        >>>         'type': 'site_summary',
        >>>         'region_id': None,
        >>>         'site_id': site_id,
        >>>         'start_date': '2020-01-01',
        >>>         'end_date': '2020-01-03',
        >>>         'geometry': site_poly.to_shapely()
        >>>     }
        >>> def demo_site(site_id, site_poly):
        >>>     sh_poly = site_poly.to_shapely()
        >>>     site = gpd.GeoDataFrame([
        >>>         {'type': 'site', 'region_id': 'DemoRegion', 'site_id': site_id, 'geometry': sh_poly},
        >>>         {'type': 'observation', 'observation_date': '2020-01-01', 'current_phase': 'phase1', 'geometry': sh_poly},
        >>>         {'type': 'observation', 'observation_date': '2020-01-02', 'current_phase': 'phase2', 'geometry': sh_poly},
        >>>         {'type': 'observation', 'observation_date': '2020-01-03', 'current_phase': 'phase3', 'geometry': sh_poly},
        >>>     ], crs=crs84)
        >>>     return {'fpath': None, 'data': site}
        >>> region_gdf_crs84 = gpd.GeoDataFrame([
        >>>     {
        >>>         'type': 'region',
        >>>         'region_id': 'DemoRegion',
        >>>         'geometry': region_poly.to_shapely(),
        >>>     },
        >>>     demo_site_summary('DemoRegion_0001', site_poly1),
        >>> ], crs=crs84)
        >>> sites = [
        >>>     demo_site('DemoRegion_0001', site_poly1),
        >>> ]
        >>> cropped_region, cropped_sites = filter_sites(region_gdf_crs84, sites)
        >>> cropped_sites = list(cropped_sites)
        >>> assert len(cropped_sites) == len(sites)
        >>> assert len(cropped_region) == 2

    Example:
        >>> # xdoctest: +REQUIRES(--slow)
        >>> from geowatch.cli.crop_sites_to_regions import *  # NOQA
        >>> import geopandas as gpd
        >>> import kwimage
        >>> from geowatch.utils import util_gis
        >>> crs84 = util_gis._get_crs84()
        >>> region_poly = kwimage.Polygon.random(rng=0).translate((42, 72))
        >>> site_poly0 = region_poly
        >>> site_poly1 = region_poly.translate((0.0001, 0.0001))
        >>> site_poly2 = region_poly.translate((3.0, 3.0))
        >>> site_poly3 = kwimage.Polygon.random().translate((42, 72))
        >>> site_poly4 = kwimage.Polygon.random().translate((43, 73))
        >>> site_poly5 = kwimage.Polygon.random().translate((42.1, 72.1))
        >>> #
        >>> def demo_site_summary(site_id, site_poly):
        >>>     return {
        >>>         'type': 'site_summary',
        >>>         'region_id': None,
        >>>         'site_id': site_id,
        >>>         'start_date': '2020-01-01',
        >>>         'end_date': '2020-01-03',
        >>>         'geometry': site_poly.to_shapely()
        >>>     }
        >>> def demo_site(site_id, site_poly):
        >>>     sh_poly = site_poly.to_shapely()
        >>>     site = gpd.GeoDataFrame([
        >>>         {'type': 'site', 'region_id': 'DemoRegion', 'site_id': site_id, 'geometry': sh_poly},
        >>>         {'type': 'observation', 'observation_date': '2020-01-01', 'current_phase': 'phase1', 'geometry': sh_poly},
        >>>         {'type': 'observation', 'observation_date': '2020-01-02', 'current_phase': 'phase2', 'geometry': sh_poly},
        >>>         {'type': 'observation', 'observation_date': '2020-01-03', 'current_phase': 'phase3', 'geometry': sh_poly},
        >>>     ], crs=crs84)
        >>>     return {'fpath': None, 'data': site}
        >>> region_gdf_crs84 = gpd.GeoDataFrame([
        >>>     {
        >>>         'type': 'region',
        >>>         'region_id': 'DemoRegion',
        >>>         'geometry': region_poly.to_shapely(),
        >>>     },
        >>>     demo_site_summary('DemoRegion_0001', site_poly1),
        >>>     demo_site_summary('DemoRegion_0002', site_poly2),
        >>> ], crs=crs84)
        >>> sites = [
        >>>     demo_site('DemoRegion_0000', site_poly0),
        >>>     demo_site('DemoRegion_0001', site_poly1),
        >>>     demo_site('DemoRegion_0002', site_poly2),
        >>>     demo_site('DemoRegion_0003', site_poly3),
        >>>     demo_site('DemoRegion_0004', site_poly4),
        >>>     demo_site('DemoRegion_0005', site_poly5),
        >>> ]
        >>> cropped_region, cropped_sites = filter_sites(region_gdf_crs84, sites)
        >>> cropped_sites = list(cropped_sites)
        >>> assert len(cropped_sites) == len(sites)
        >>> assert len(cropped_sites[0]['data']) == len(sites[0]['data'])
        >>> assert len(cropped_sites[1]['data']) == len(sites[1]['data'])
        >>> assert len(cropped_sites[2]['data']) == 0
        >>> assert len(cropped_region) == 2
    """
    from geowatch.utils import util_gis
    import pandas as pd
    output_crs = region_gdf_crs84.crs

    if filter_config is None:
        filter_config = SiteFilterConfig()

    _row_type = region_gdf_crs84['type']
    region_rows_crs84 = region_gdf_crs84[_row_type == 'region']
    sitesum_rows_crs84 = region_gdf_crs84[_row_type == 'site_summary']

    num_sitesum_rows = len(sitesum_rows_crs84)
    num_region_rows = len(region_rows_crs84)
    num_total_rows = len(region_gdf_crs84)

    if num_region_rows != 1:
        raise AssertionError(f'can only have one region row with type region. Got {len(region_rows_crs84)}')

    if num_total_rows != num_sitesum_rows + num_region_rows:
        unique_types = region_gdf_crs84['type'].unique()
        raise AssertionError(f'Region file expected to only contain types of region and site_summary. Got {unique_types}')

    # Take only the first row, ignore site-summaries
    assert region_gdf_crs84.iloc[0].type == 'region'
    assert region_gdf_crs84.crs.name.startswith('WGS 84')

    site_summary_rows = region_gdf_crs84.iloc[1:]

    region_row_crs84 = region_rows_crs84.iloc[0:1]
    region_geom_crs84 = region_row_crs84.geometry.iloc[0]

    utm_epsg : int = util_gis.find_local_meter_epsg_crs(region_geom_crs84)
    region_row_utm = region_row_crs84.to_crs(utm_epsg)
    crop_geom_utm = region_row_utm.geometry.iloc[0]

    assert crop_geom_utm.is_valid

    # Crop the site summaries within the region file
    main_type = 'site_summary'
    valid_site_summaries = filter_gdf_in_utm(
        site_summary_rows, crop_geom_utm, utm_epsg, output_crs, main_type,
        filter_config)

    cropped_region = pd.concat([region_rows_crs84, valid_site_summaries])

    # Crop the site models to the region geometry
    cropped_sites = _cropper_gen(sites, crop_geom_utm, utm_epsg, output_crs,
                                 filter_config)

    return cropped_region, cropped_sites


def _cropper_gen(sites, crop_geom_utm, utm_epsg, output_crs, filter_config):
    import geopandas as gpd
    main_type = 'site'
    for site_info in sites:
        site_gdf_crs84: gpd.GeoDataFrame = site_info['data']
        valid_site_gdf_crs84 = filter_gdf_in_utm(
            site_gdf_crs84, crop_geom_utm, utm_epsg, output_crs,
            main_type, filter_config)

        new_site_info: dict = site_info.copy()
        new_site_info['data'] = valid_site_gdf_crs84
        yield new_site_info


def filter_gdf_in_utm(gdf, crop_geom_utm, utm_epsg, output_crs, main_type=None,
                      filter_config=None):
    """
    Crop geometry in a geopandas data frame to specified bounds in UTM space.
    Filter out any rows where the cropped geometry is null or invalid.

    Args:
        gdf (geopandas.GeoDataFrame):
            The data to crop (in CRS84)

        crop_geom_utm (shapely.geometry.polygon.Polygon):
            The UTM polygon to crop to.

        utm_epsg (int):
            The UTM zone to work in

        output_crs (pyproj.crs.crs.CRS):
            The output CRS to wrap back into (should be CRS84)

        main_type (str):
            "site_summary" for region models, and "site" for site models.

        filter_config (SiteFilterConfig | None):
            modifies filter behavior.

    Returns:
        GeoDataFrame
    """
    import pandas as pd
    assert filter_config is not None
    gdf_utm = gdf.to_crs(utm_epsg)

    # Attempt to fix any polygon that became invalid after UTM projection
    invalid_proj = ~gdf_utm.geometry.is_valid
    if invalid_proj.any():
        gdf_utm.geometry[invalid_proj] = gdf_utm.geometry[invalid_proj].buffer(0)
        # HACK: make_valid() here not supported until geopandas version 0.12.0 (strict has 0.10.2)
        # TODO: Update requirements (requires environment rebuild)
        # gdf_utm.geometry[invalid_proj] = gdf_utm.geometry[invalid_proj].make_valid()
    invalid_proj = ~gdf_utm.geometry.is_valid
    assert not invalid_proj.any()

    if main_type is not None:
        is_main = gdf_utm['type'] == main_type
        main_gdf = gdf_utm[is_main]
        other_gdf = gdf_utm[~is_main]
    else:
        main_gdf = gdf_utm
        other_gdf = gdf_utm.iloc[0:0]

    if main_type == 'site':
        assert len(main_gdf) == 1, 'should only have one main polygon for a site'

    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', 'invalid value', RuntimeWarning)
        warnings.filterwarnings('ignore', 'Self-intersection', RuntimeWarning)
        main_isect = main_gdf.intersection(crop_geom_utm)
        other_isect = other_gdf.intersection(crop_geom_utm)

    # First remove everything completely outside of the bounds
    main_is_empty = main_isect.is_empty
    main_isect = main_isect[~main_is_empty]
    main_gdf = main_gdf[~main_is_empty]

    if filter_config.apply_clip_to == 'none':
        do_clip_flags = main_isect.is_empty & False
    elif filter_config.apply_clip_to == 'all':
        do_clip_flags = main_isect.is_empty | False
    elif filter_config.apply_clip_to == 'polygon':
        do_clip_flags = (main_isect.type == 'Polygon')
    else:
        raise KeyError
    do_clip_locs = do_clip_flags[do_clip_flags].index

    # Clip the chosen geometry to the bounds
    if len(do_clip_locs):
        main_gdf.loc[do_clip_locs, 'geometry'] = main_isect.loc[do_clip_locs]

    if filter_config.apply_bounds_filter_to == 'none':
        do_filter_flags = main_isect.is_empty | False
    elif filter_config.apply_bounds_filter_to == 'all':
        do_filter_flags = main_isect.is_empty | False
    elif filter_config.apply_bounds_filter_to == 'multipolygon':
        do_filter_flags = (main_isect.type == 'MultiPolygon')
    else:
        raise KeyError
    do_filter_locs = do_filter_flags[do_filter_flags].index

    # Remove chosen polygons that have a low intersection with the region
    if len(do_filter_locs):
        in_bounds_thresh = filter_config.in_bounds_thresh
        main_to_filter = main_gdf.loc[do_filter_flags]
        isect_to_filter = main_isect.loc[do_filter_flags]
        frac_in_bounds = isect_to_filter.area / main_to_filter.area
        flags = frac_in_bounds <= in_bounds_thresh
        remove_locs = flags[flags].index
        main_gdf = main_gdf.drop(remove_locs, axis=0)

    if filter_config.min_area_square_meters is not None:
        keep_flags = main_gdf.geometry.area >= filter_config.min_area_square_meters
        main_gdf = main_gdf[keep_flags]

    if filter_config.max_area_square_meters is not None:
        keep_flags = main_gdf.geometry.area <= filter_config.max_area_square_meters
        main_gdf = main_gdf[keep_flags]

    if len(other_gdf) > 0:
        # Specialized logic for site models where there will only be
        # one row in main_gdf
        assert main_type == 'site'
        if len(main_gdf) == 0:
            # If we filtered the site, then we should filter all observations
            other_gdf = other_gdf.drop(other_gdf.index, axis=0)
        else:
            # Otherwise we we either clip everything or do nothing.
            if len(do_clip_locs):
                other_gdf = other_gdf.assign(geometry=other_isect)
            # always remove empty ones.
            other_gdf = other_gdf[~other_isect.is_empty]
        valid_gdf_utm = pd.concat([main_gdf, other_gdf], axis=0)
    else:
        valid_gdf_utm = main_gdf

    # Project back to the output CRS
    valid_gdf_crs84 = valid_gdf_utm.to_crs(output_crs)
    return valid_gdf_crs84


__config__ = CropSitesToRegionsConfig


if __name__ == '__main__':
    """
    CommandLine:
        DVC_DPATH=$(geowatch_dvc)
        python -m geowatch.cli.crop_sites_to_regions \
            --site_models=$DVC_DPATH/annotations/site_models/*KR*.geojson \
            --region_models=$DVC_DPATH/annotations/region_models/KR_R001.geojson \
            --new_site_dpath=$DVC_DPATH/tmp/new_site_models \
            --new_region_dpath=$DVC_DPATH/tmp/new_region_models
    """
    main(cmdline=True)
