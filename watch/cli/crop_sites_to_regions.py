import scriptconfig as scfg
import warnings
import ubelt as ub


class CropSitesToRegionsConfig(scfg.Config):
    r"""
    Crops site models to the bounds of a region model

    Example:
        DVC_DPATH=$(WATCH_PREIMPORT=none python -m watch.cli.find_dvc)
        WATCH_PREIMPORT=none python -m watch.cli.crop_sites_to_regions \
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

        'new_region_dpath': scfg.Value(None, help=ub.paragraph(
            '''
            Destination directory for new site models.
            Note: names of files must be unique.
            ''')),

        'io_workers': scfg.Value(0, help=ub.paragraph(
            '''
            IO workers to load sites in the background while others are
            cropping.
            ''')),
    }


USE_LISTS = 0  # turn on for eager debugging


def main(cmdline=False, **kwargs):
    from watch.utils import util_gis
    from watch.utils import util_path
    import geopandas as gpd
    import safer

    config = CropSitesToRegionsConfig(default=kwargs, cmdline=cmdline)
    print('config = {}'.format(ub.repr2(dict(config), nl=1)))

    new_site_dpath = config['new_site_dpath']
    assert new_site_dpath is not None, 'must specify new_site_dpath'
    new_site_dpath = ub.Path(new_site_dpath)

    new_region_dpath = config['new_region_dpath']
    assert new_region_dpath is not None, 'must specify new_region_dpath'
    new_region_dpath = ub.Path(new_region_dpath)

    site_geojson_fpaths: list[ub.Path] = util_path.coerce_patterned_paths(config['site_models'], '.geojson')
    region_geojson_fpaths: list[ub.Path] = util_path.coerce_patterned_paths(config['region_models'], '.geojson')

    if len(region_geojson_fpaths) != 1:
        raise ValueError(f'Must specify exactly one region file, Got: {region_geojson_fpaths}')

    regions = []
    for region_fpath in ub.ProgIter(region_geojson_fpaths, desc='load geojson region-models'):
        region_gdf_crs84 : gpd.GeoDataFrame = util_gis.read_geojson(region_fpath)
        regions.append(region_gdf_crs84)
    old_region_fpath = region_fpath
    region_gdf_crs84 = regions[0]

    io_workers = config['io_workers']

    sites = load_site_gen(site_geojson_fpaths, io_workers)

    if USE_LISTS:
        sites = list(sites)

    cropped_region, cropped_sites = crop_sites_to_region(
        region_gdf_crs84, sites)

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
        with safer.open(new_region_fpath, temp_file=True, mode='wb') as file:
            cropped_region.to_file(file, driver='GeoJSON')
        print(f'Wrote cropped site summaries to {new_region_fpath}')

        # Save the cropped sites to disk
        new_site_dpath.ensuredir()
        num_valid = 0
        total = 0
        for cropped_site_info in cropped_sites:
            total += 1
            cropped_site = cropped_site_info['gdf']
            if len(cropped_site):
                num_valid += 1
                old_site_fpath = cropped_site_info['fpath']
                new_site_fpath = new_site_dpath / old_site_fpath.name
                with safer.open(new_site_fpath, temp_file=True, mode='wb') as file:
                    cropped_site.to_file(file, driver='GeoJSON')
        print(f'Wrote {num_valid} / {total} valid cropped sites in {new_site_dpath}')


def load_site_gen(site_geojson_fpaths, io_workers=0, mode='thread'):
    """
    Generator that loads sites (and the path they loaded from) in parallel

    Args:
        site_geojson_fpaths (Iterable[PathLike]):
            geojson paths to load

        io_workers (int):
            number of background loading workers

        mode (str):
            concurrent executor mode

    Yields:
        Dict:
            containing keys, 'fpath' and 'gdf'.
    """
    from watch.utils import util_gis
    import geopandas as gpd
    # sites = []
    if USE_LISTS:
        desc = 'load geojson site-models'
    else:
        desc = 'load / cropping geojson site-models'

    jobs = ub.JobPool(mode=mode, max_workers=io_workers)
    for site_fpath in ub.ProgIter(site_geojson_fpaths, desc='submit ' + desc):
        job = jobs.submit(util_gis.read_geojson, site_fpath)
        job.fpath = site_fpath

    for job in jobs.as_completed(desc=desc):
        site_gdf_crs84 : gpd.GeoDataFrame = job.result()
        site_info = {
            'fpath': job.fpath,
            'gdf': site_gdf_crs84
        }
        yield site_info


def crop_sites_to_region(region_gdf_crs84, sites):
    """
    Args:
        region_gdf_crs84 (GeoDataFrame):
            the region GDF containing the region geom to crop to and
            the site summary geometry

        sites (Iterable[Dict]):
            List of the loaded geo data frames with a 'gdf' key
            and the file path in the 'path' key.

    Returns:
        Tuple[GeoDataFrame, List[Dict]]:
            Region model with cropped site summaries and a list of site info
            dictionaries containing the new cropped gdf field.

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
        >>>     ], crs='crs84')
        >>>     return {'fpath': None, 'gdf': site}
        >>> region_gdf_crs84 = gpd.GeoDataFrame([
        >>>     {
        >>>         'type': 'region',
        >>>         'region_id': 'DemoRegion',
        >>>         'geometry': region_poly.to_shapely(),
        >>>     },
        >>>     demo_site_summary('DemoRegion_0001', site_poly1),
        >>>     demo_site_summary('DemoRegion_0002', site_poly2),
        >>> ], crs='crs84')
        >>> sites = [
        >>>     demo_site('DemoRegion_0000', site_poly0),
        >>>     demo_site('DemoRegion_0001', site_poly1),
        >>>     demo_site('DemoRegion_0002', site_poly2),
        >>>     demo_site('DemoRegion_0003', site_poly3),
        >>>     demo_site('DemoRegion_0004', site_poly4),
        >>>     demo_site('DemoRegion_0005', site_poly5),
        >>> ]
        >>> cropped_region, cropped_sites = crop_sites_to_region(region_gdf_crs84, sites)
        >>> cropped_sites = list(cropped_sites)
        >>> assert len(cropped_sites) == len(sites)
        >>> assert len(cropped_sites[0]['gdf']) == len(sites[0]['gdf'])
        >>> assert len(cropped_sites[1]['gdf']) == len(sites[1]['gdf'])
        >>> assert len(cropped_sites[2]['gdf']) == 0
        >>> assert len(cropped_region) == 2
    """
    from watch.utils import util_gis
    import pandas as pd
    output_crs = region_gdf_crs84.crs

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
    assert region_gdf_crs84.crs.name == 'WGS 84 (CRS84)'

    site_summary_rows = region_gdf_crs84.iloc[1:]

    region_row_crs84 = region_rows_crs84.iloc[0:1]
    region_geom_crs84 = region_row_crs84.geometry.iloc[0]

    utm_epsg : int = util_gis.find_local_meter_epsg_crs(region_geom_crs84)
    region_row_utm = region_row_crs84.to_crs(utm_epsg)
    crop_geom_utm = region_row_utm.geometry.iloc[0]

    assert crop_geom_utm.is_valid

    # Crop the site summaries within the region file
    valid_site_summaries = crop_gdf_in_utm(
        site_summary_rows, crop_geom_utm, utm_epsg, output_crs)

    cropped_region = pd.concat([region_rows_crs84, valid_site_summaries])

    # Crop the site models to the region geometry
    cropped_sites = _cropper_gen(sites, crop_geom_utm, utm_epsg, output_crs)

    return cropped_region, cropped_sites


def _cropper_gen(sites, crop_geom_utm, utm_epsg, output_crs):
    import geopandas as gpd
    for site_info in sites:
        site_gdf_crs84: gpd.GeoDataFrame = site_info['gdf']
        valid_site_gdf_crs84 = crop_gdf_in_utm(
            site_gdf_crs84, crop_geom_utm, utm_epsg, output_crs)

        new_site_info: dict = site_info.copy()
        new_site_info['gdf'] = valid_site_gdf_crs84
        yield new_site_info


def crop_gdf_in_utm(gdf, crop_geom_utm, utm_epsg, output_crs):
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

    Returns:
        GeoDataFrame
    """
    gdf_utm = gdf.to_crs(utm_epsg)

    # Attempt to fix any polygon that became invalid after UTM projection
    invalid_proj = ~gdf_utm.geometry.is_valid
    if invalid_proj.any():
        gdf_utm.geometry[invalid_proj] = gdf_utm.geometry[invalid_proj].buffer(0)
    invalid_proj = ~gdf_utm.geometry.is_valid
    assert not invalid_proj.any()

    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', 'invalid value', RuntimeWarning)
        warnings.filterwarnings('ignore', 'Self-intersection', RuntimeWarning)
        isect = gdf_utm.intersection(crop_geom_utm)
    flags = isect.area > 0

    valid_isect = isect[flags]
    valid_gdf_utm = gdf_utm[flags]
    valid_gdf_utm = valid_gdf_utm.assign(geometry=valid_isect)

    # Project back to the output CRS
    valid_gdf_crs84 = valid_gdf_utm.to_crs(output_crs)
    return valid_gdf_crs84


if __name__ == '__main__':
    """
    CommandLine:
        DVC_DPATH=$(smartwatch_dvc)
        python -m watch.cli.crop_sites_to_regions \
            --site_models=$DVC_DPATH/annotations/site_models/*KR*.geojson \
            --region_models=$DVC_DPATH/annotations/region_models/KR_R001.geojson \
            --new_site_dpath=$DVC_DPATH/tmp/new_site_models \
            --new_region_dpath=$DVC_DPATH/tmp/new_region_models
    """
    main(cmdline=True)