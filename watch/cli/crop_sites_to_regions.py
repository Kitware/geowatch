import scriptconfig as scfg
import warnings
import ubelt as ub


class CropSitesToRegionsConfig(scfg.Config):
    r"""
    Crops site models to the bounds of a region model

    Example:
        DVC_DPATH=$(WATCH_HACK_IMPORT_ORDER=none python -m watch.cli.find_dvc)
        WATCH_HACK_IMPORT_ORDER=none python -m watch.cli.crop_sites_to_regions \
            --site_models "$DVC_DPATH/annotations/site_models/KR_R002_*.geojson" \
            --region_models "$DVC_DPATH/annotations/region_models/KR_R002.geojson" \
            --new_site_dpath ./cropped_sites
    """
    default = {
        'site_models': scfg.Value(None, help=ub.paragraph(
            '''
            Geospatial geojson "site" annotation files. Either a path to a
            file, or a directory.
            ''')),

        'region_models': scfg.Value(None, help=ub.paragraph(
            '''
            A single geojson "region" file to crop to.
            ''')),

        'new_site_dpath': scfg.Value(None, help=ub.paragraph(
            '''
            Destination directory for new site models.
            Note: names of files must be unique.
            ''')),
    }


def main(cmdline=False, **kwargs):
    from watch.utils import util_gis
    from watch.utils import util_path
    import geopandas as gpd
    config = CropSitesToRegionsConfig(default=kwargs, cmdline=cmdline)
    print('config = {}'.format(ub.repr2(dict(config), nl=1)))

    new_site_dpath = config['new_site_dpath']
    assert new_site_dpath is not None, 'must specify dst dpath'
    new_site_dpath = ub.Path(new_site_dpath)
    site_geojson_fpaths: list[ub.Path] = util_path.coerce_patterned_paths(config['site_models'], '.geojson')
    region_geojson_fpaths: list[ub.Path] = util_path.coerce_patterned_paths(config['region_models'], '.geojson')

    if len(region_geojson_fpaths) != 1:
        raise ValueError(f'Must specify exactly one region file, Got: {region_geojson_fpaths}')

    regions = []
    for region_fpath in ub.ProgIter(region_geojson_fpaths, desc='load geojson region-models'):
        region_gdf_crs84 : gpd.GeoDataFrame = util_gis.read_geojson(region_fpath)
        regions.append(region_gdf_crs84)
    region_gdf_crs84 = regions[0]

    USE_LISTS = 0

    def _load_site_gen():
        # sites = []
        if USE_LISTS:
            desc = 'loading geojson site-models'
        else:
            desc = 'loading / cropping geojson site-models'
        for site_fpath in ub.ProgIter(site_geojson_fpaths, desc=desc):
            site_gdf_crs84 : gpd.GeoDataFrame = util_gis.read_geojson(site_fpath)
            yield site_gdf_crs84
            # sites.append(site_gdf_crs84)

    new_site_dpath.ensuredir()

    sites = _load_site_gen()

    if USE_LISTS:
        sites = list(sites)

    cropped_sites = crop_sites_to_region(region_gdf_crs84, sites)

    if USE_LISTS:
        cropped_sites = ub.ProgIter(cropped_sites, desc='Cropping sites', total=len(site_geojson_fpaths))
        cropped_sites = list(cropped_sites)

    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', 'pandas.Int64Index is deprecated', FutureWarning)
        import safer

        for cropped_site, fpath in zip(cropped_sites, site_geojson_fpaths):
            new_fpath = new_site_dpath / fpath.name
            with safer.open(new_fpath, temp_file=True, mode='w') as file:
                cropped_site.to_file(file, driver='GeoJSON')


def crop_sites_to_region(region_gdf_crs84, sites):
    """
    Args:
        region_gdf_crs84 (GeoDataFrame): the region GDF
        sites (List[GeoDataFrame]): the sites GDFs

    Yields:
        GeoDataFrame: cropped site models in the same order they were given

    Example:
        >>> from watch.cli.crop_sites_to_regions import *  # NOQA
        >>> import geopandas as gpd
        >>> import kwimage
        >>> region_poly = kwimage.Polygon.random(rng=0).translate((42, 72))
        >>> site_poly0 = region_poly
        >>> site_poly1 = region_poly.translate((0.0001, 0.0001))
        >>> site_poly2 = region_poly.translate((3.0, 3.0))
        >>> site_poly3 = kwimage.Polygon.random().translate((42, 72))
        >>> site_poly4 = kwimage.Polygon.random().translate((43, 73))
        >>> site_poly5 = kwimage.Polygon.random().translate((42.1, 72.1))
        >>> region_gdf_crs84 = gpd.GeoDataFrame([
        >>>     {
        >>>         'type': 'region',
        >>>         'region_id': 'DemoRegion',
        >>>         'geometry': region_poly.to_shapely(),
        >>>     }
        >>> ], crs='crs84')
        >>> #
        >>> def demo_site(site_id, site_poly):
        >>>     sh_poly = site_poly.to_shapely()
        >>>     site = gpd.GeoDataFrame([
        >>>         {'type': 'site', 'region_id': 'DemoRegion', 'site_id': site_id, 'geometry': sh_poly},
        >>>         {'type': 'observation', 'observation_date': '2020-01-01', 'current_phase': 'phase1', 'geometry': sh_poly},
        >>>         {'type': 'observation', 'observation_date': '2020-01-02', 'current_phase': 'phase2', 'geometry': sh_poly},
        >>>         {'type': 'observation', 'observation_date': '2020-01-03', 'current_phase': 'phase3', 'geometry': sh_poly},
        >>>     ], crs='crs84')
        >>>     return site
        >>> sites = [
        >>>     demo_site('DemoRegion_0000', site_poly0),
        >>>     demo_site('DemoRegion_0001', site_poly1),
        >>>     demo_site('DemoRegion_0002', site_poly2),
        >>>     demo_site('DemoRegion_0003', site_poly3),
        >>>     demo_site('DemoRegion_0004', site_poly4),
        >>>     demo_site('DemoRegion_0005', site_poly5),
        >>> ]
        >>> cropped_sites = list(crop_sites_to_region(region_gdf_crs84, sites))
        >>> assert len(cropped_sites) == len(sites)
        >>> assert len(cropped_sites[0]) == len(sites[0])
        >>> assert len(cropped_sites[1]) == len(sites[1])
        >>> assert len(cropped_sites[3]) == 0
    """
    from watch.utils import util_gis
    output_crs = region_gdf_crs84.crs

    # Take only the first row, ignore site-summaries
    assert region_gdf_crs84.iloc[0].type == 'region'
    assert region_gdf_crs84.crs.name == 'WGS 84 (CRS84)'
    region_row_crs84 = region_gdf_crs84.iloc[0:1]
    region_geom_crs84 = region_row_crs84.geometry.iloc[0]

    utm_epsg : int = util_gis.find_local_meter_epsg_crs(region_geom_crs84)
    region_row_utm = region_row_crs84.to_crs(utm_epsg)
    region_geom_utm = region_row_utm.geometry.iloc[0]

    assert region_geom_utm.is_valid

    # Read the external CRS84 annotations from the site models
    cropped_sites = []
    for site_gdf_crs84 in sites:
        site_gdf_utm = site_gdf_crs84.to_crs(utm_epsg)

        # Attempt to fix any polygon that became invalid after UTM projection
        invalid_proj = ~site_gdf_utm.geometry.is_valid
        if invalid_proj.any():
            site_gdf_utm.geometry[invalid_proj] = site_gdf_utm.geometry[invalid_proj].buffer(0)
        invalid_proj = ~site_gdf_utm.geometry.is_valid
        assert not invalid_proj.any()

        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', 'invalid value', RuntimeWarning)
            isect = site_gdf_utm.intersection(region_geom_utm)
        flags = isect.area > 0

        valid_isect = isect[flags]
        valid_site_gdf_utm = site_gdf_utm[flags]
        valid_site_gdf_utm['geometry'] = valid_isect

        # Project back to the output CRS
        valid_site_gdf_crs84 = valid_site_gdf_utm.to_crs(output_crs)
        cropped_sites.append(valid_site_gdf_crs84)

    return cropped_sites


if __name__ == '__main__':
    """
    CommandLine:
        python ~/code/watch/watch/cli/crop_sites_to_regions.py
    """
    main(cmdline=True)
