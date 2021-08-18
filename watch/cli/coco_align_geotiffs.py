"""
Given the raw data in kwcoco format, this script will extract orthorectified
regions around areas of intere/t across time.

Notes:

    # Given the output from geojson_to_kwcoco this script extracts
    # orthorectified regions.

    # https://data.kitware.com/#collection/602457272fa25629b95d1718/folder/602c3e9e2fa25629b97e5b5e

    python -m watch.cli.coco_align_geotiffs \
            --src ~/data/dvc-repos/smart_watch_dvc/drop0/drop0.kwcoco.json \
            --dst ~/data/dvc-repos/smart_watch_dvc/drop0_aligned_v2 \
            --context_factor=1.5

    # Archive the data and upload to data.kitware.com
    cd $HOME/data/dvc-repos/smart_watch_dvc/
    7z a ~/data/dvc-repos/smart_watch_dvc/drop0_aligned_v2.zip \
            ~/data/dvc-repos/smart_watch_dvc/drop0_aligned_v2

    stamp=$(date +"%Y-%m-%d")
    # resolve links (7z cant handl)
    rsync -avpL drop0_aligned_v2 drop0_aligned_v2_$stamp
    7z a drop0_aligned_v2_$stamp.zip drop0_aligned_v2_$stamp

    source $HOME/internal/secrets
    cd $HOME/data/dvc-repos/smart_watch_dvc/
    girder-client --api-url https://data.kitware.com/api/v1 upload \
            602c3e9e2fa25629b97e5b5e drop0_aligned_v2_$stamp.zip

    python -m watch.cli.coco_align_geotiffs \
            --src ~/data/dvc-repos/smart_watch_dvc/drop0/drop0-msi.kwcoco.json \
            --dst ~/data/dvc-repos/smart_watch_dvc/drop0_aligned_msi \
            --context_factor=1.5


    python -m watch.cli.coco_align_geotiffs \
            --src ~/data/dvc-repos/smart_watch_dvc/drop0/drop0-msi.kwcoco.json \
            --dst ~/data/dvc-repos/smart_watch_dvc/drop0_aligned_msi_big \
            --context_factor=3.5


Test:

    There was a bug in KR-WV, run the script only on that region to test if we
    have fixed it.

    kwcoco stats ~/data/dvc-repos/smart_watch_dvc/drop0/KR-Pyeongchang-WV/data.kwcoco.json

    jq .images ~/data/dvc-repos/smart_watch_dvc/drop0/KR-Pyeongchang-WV/data.kwcoco.json

    kwcoco subset ~/data/dvc-repos/smart_watch_dvc/drop0/KR-Pyeongchang-WV/data.kwcoco.json --gids=1129,1130 --dst ~/data/dvc-repos/smart_watch_dvc/drop0/KR-Pyeongchang-WV/subtmp.kwcoco.json

    python -m watch.cli.coco_align_geotiffs \
            --src ~/remote/namek/data/dvc-repos/smart_watch_dvc/drop0/KR-Pyeongchang-WV/subtmp.kwcoco.json \
            --dst ~/remote/namek/data/dvc-repos/smart_watch_dvc/drop0_aligned_WV_Fix \
            --rpc_align_method pixel_crop \
            --context_factor=3.5

           # --src ~/data/dvc-repos/smart_watch_dvc/drop0/KR-Pyeongchang-WV/data.kwcoco.json \

    TODO:
        - [ ] Add method for extracting "negative ROIs" that are nearby
            "positive ROIs".
"""
import kwcoco
import kwimage
import numpy as np
import os
import scriptconfig as scfg
import socket
import ubelt as ub
import dateutil.parser
import geopandas as gpd
import datetime
import shapely
from shapely import ops
from os.path import join, exists


class CocoAlignGeotiffConfig(scfg.Config):
    """
    Create a dataset of aligned temporal sequences around objects of interest
    in an unstructured collection of annotated geotiffs.

    High Level Steps:
        * Find a set of geospatial AOIs
        * For each AOI find all images that overlap
        * Orthorectify (or warp) the selected spatial region and its
          annotations to a cannonical space.
    """
    default = {
        'src': scfg.Value('in.geojson.json', help='input dataset to chip'),

        'dst': scfg.Value(None, help='bundle directory for the output'),

        'max_workers': scfg.Value(4, help='number of parallel procs'),
        'aux_workers': scfg.Value(4, help='additional inner threads for aux imgs'),

        'context_factor': scfg.Value(1.0, help=ub.paragraph(
            '''
            scale factor for the clustered ROIs.
            Amount of context to extract around each ROI.
            '''
        )),

        'regions': scfg.Value('annots', help=ub.paragraph(
            '''
            Strategy for extracting regions, if annots, uses the convex hulls
            of clustered annotations. Can also be a path to a geojson file
            to use pre-defined regions.
            ''')),

        # TODO: change this name to just align-method or something
        'rpc_align_method': scfg.Value('orthorectify', help=ub.paragraph(
            '''
            Can be one of:
                (1) orthorectify - which uses gdalwarp with -rpc,
                (2) pixel_crop - which warps annotations onto pixel with RPCs
                    but only crops the original image without distortion,
                (3) affine_warp - which ignores RPCs and uses the affine
                    transform in the geotiff metadata.
            '''
        )),

        'write_subsets': scfg.Value(True, help=ub.paragraph(
            '''
            if True, writes a separate kwcoco file for every discovered ROI
            in addition to the final kwcoco file.
            '''
        )),

        'visualize': scfg.Value(True, help=ub.paragraph(
            '''
            if True, normalize and draw image / annotation sequences when
            extracting.
            '''
        )),

    }


def main(**kw):
    """
    Main function for coco_align_geotiffs.
    See :class:``CocoAlignGeotiffConfig` for details

    Ignore:
        from watch.cli.coco_align_geotiffs import *  # NOQA
        import kwcoco
        src = ub.expandpath('~/data/dvc-repos/smart_watch_dvc/drop0/drop0.kwcoco.json')
        dst = ub.expandpath('~/data/dvc-repos/smart_watch_dvc/drop0_aligned')
        kw = {
            'src': src,
            'dst': dst,
        }

    Example:
        >>> from watch.cli.coco_align_geotiffs import *  # NOQA
        >>> from watch.demo.landsat_demodata import grab_landsat_product
        >>> from watch.gis.geotiff import geotiff_metadata
        >>> # Create a dead simple coco dataset with one image
        >>> import kwcoco
        >>> dset = kwcoco.CocoDataset()
        >>> ls_prod = grab_landsat_product()
        >>> fpath = ls_prod['bands'][0]
        >>> meta = geotiff_metadata(fpath)
        >>> # We need a date captured ATM in a specific format
        >>> dt = dateutil.parser.parse(
        >>>     meta['filename_meta']['acquisition_date'])
        >>> date_captured = dt.strftime('%Y/%m/%d')
        >>> gid = dset.add_image(file_name=fpath, date_captured=date_captured)
        >>> dummy_poly = kwimage.Polygon(exterior=meta['wgs84_corners'])
        >>> dummy_poly = dummy_poly.scale(0.3, about='center')
        >>> sseg_geos = dummy_poly.swap_axes().to_geojson()
        >>> # NOTE: script is not always robust to missing annotation
        >>> # information like segmentation and bad bbox, but for this
        >>> # test config it is
        >>> dset.add_annotation(
        >>>     image_id=gid, bbox=[0, 0, 0, 0], segmentation_geos=sseg_geos)
        >>> #
        >>> # Create arguments to the script
        >>> dpath = ub.ensure_app_cache_dir('smart_watch/test/coco_align_geotiff')
        >>> dst = ub.ensuredir((dpath, 'align_bundle1'))
        >>> ub.delete(dst)
        >>> dst = ub.ensuredir(dst)
        >>> kw = {
        >>>     'src': dset,
        >>>     'dst': dst,
        >>>     'regions': 'annots',
        >>> }
        >>> new_dset = main(**kw)
    """
    config = CocoAlignGeotiffConfig(default=kw, cmdline=True)

    # Store that this dataset is a result of a process.
    # Note what the process is, what its arguments are, and where the process
    # was executed.
    config_dict = config.to_dict()
    if not isinstance(config_dict['src'], str):
        # If the dataset was given in memory we don't know the path and we cant
        # always serialize it, so we punt and mark it as such
        config_dict['src'] = ':memory:'

    process_info = {
        'type': 'process',
        'properties': {
            'name': 'coco_align_geotiffs',
            'args': config_dict,
            'hostname': socket.gethostname(),
            'cwd': os.getcwd(),
            'timestamp': ub.timestamp(),
        }
    }
    print('process_info = {}'.format(ub.repr2(process_info, nl=2)))

    src_fpath = config['src']
    dst_dpath = config['dst']
    regions = config['regions']
    context_factor = config['context_factor']
    rpc_align_method = config['rpc_align_method']
    visualize = config['visualize']
    write_subsets = config['write_subsets']
    max_workers = config['max_workers']
    aux_workers = config['aux_workers']

    output_bundle_dpath = dst_dpath

    if regions == 'annots':
        pass
    elif exists(regions):
        region_df = read_geojson(regions)
        print('region_df = {!r}'.format(region_df))
    else:
        raise KeyError(regions)

    # Load the dataset and extract geotiff metadata from each image.
    dset = kwcoco.CocoDataset.coerce(src_fpath)
    update_coco_geotiff_metadata(dset, serializable=False,
                                 max_workers=max_workers)

    # Construct the "data cube"
    cube = SimpleDataCube(dset)

    if regions == 'annots':
        # Find the clustered ROI regions
        sh_all_rois, kw_all_rois = find_roi_regions(dset)
        region_df = gpd.GeoDataFrame([
            {'geometry': geos, 'start_date': None, 'end_date': None}
            for geos in sh_all_rois
        ], geometry='geometry', crs='epsg:4326')

    # Exapnd the ROI by the context factor and convert to a bounding box
    region_df['geometry'] = region_df['geometry'].apply(shapely_bounding_box)
    if context_factor != 1:
        region_df['geometry'] = region_df['geometry'].scale(
            xfact=context_factor, yfact=context_factor, origin='center')

    # For each ROI extract the aligned regions to the target path
    extract_dpath = ub.expandpath(output_bundle_dpath)
    ub.ensuredir(extract_dpath)

    # Create a new dataset that we will extend as we extract ROIs
    new_dset = kwcoco.CocoDataset()

    new_dset.dataset['info'] = [
        process_info,
    ]

    to_extract = cube.query_image_overlaps2(region_df)

    for image_overlaps in ub.ProgIter(to_extract, desc='extract ROI videos', verbose=3):
        video_name = image_overlaps['video_name']
        print('video_name = {!r}'.format(video_name))

        sub_bundle_dpath = join(extract_dpath, video_name)
        print('sub_bundle_dpath = {!r}'.format(sub_bundle_dpath))

        cube.extract_overlaps(image_overlaps, extract_dpath,
                              rpc_align_method=rpc_align_method,
                              new_dset=new_dset, visualize=visualize,
                              write_subsets=write_subsets,
                              max_workers=max_workers, aux_workers=aux_workers)

    new_dset.fpath = join(extract_dpath, 'data.kwcoco.json')
    print('Dumping new_dset.fpath = {!r}'.format(new_dset.fpath))
    new_dset.reroot(new_root=output_bundle_dpath, absolute=False)
    new_dset.dump(new_dset.fpath, newlines=True)
    print('finished')
    return new_dset


def demo_regions_geojson_text():
    geojson_text = ub.codeblock(
        '''
        {
            "type": "FeatureCollection",
            "features": [
                {
                    "type": "Feature",
                    "properties": {"type": "region", "region_model_id": "US_Jacksonville_R01", "version": "1.0.1", "mgrs": "17RMP", "start_date": "2009-05-09", "end_date": "2020-01-26" },
                    "geometry": {"type": "Polygon", "coordinates": [[[-81.6953, 30.3652], [-81.6942, 30.2984], [-81.5975, 30.2992], [-81.5968, 30.3667], [-81.6953, 30.3652]]]}
                },
                {
                    "type": "Feature",
                    "properties": {"type": "site", "site_id": "17RMP_US_Jacksonville_R01_0000", "start_date": "2016-02-14", "end_date": "2017-11-01"},
                    "geometry": {"type": "Polygon", "coordinates": [[[-81.6364, 30.3209], [-81.6364, 30.3236], [-81.6397, 30.3236], [-81.6397, 30.3209], [-81.6364, 30.3209]]]}
                },
                {
                    "type": "Feature",
                    "properties": {"type": "site", "site_id": "17RMP_US_Jacksonville_R01_0001", "start_date": "2016-07-13", "end_date": "2020-01-26" },
                    "geometry": {"type": "Polygon", "coordinates": [[[-81.6085, 30.3568], [-81.6085, 30.3600], [-81.6120, 30.3600], [-81.6120, 30.3568], [-81.6085, 30.3568]]]}
                }
            ]
        }
        ''')
    return geojson_text


def read_geojson(file, default_axis_mapping='OAMS_TRADITIONAL_GIS_ORDER'):
    """
    Args:
        file (str | file): path or file object containing geojson data.

        axis_mapping (str, default='OAMS_TRADITIONAL_GIS_ORDER'):
            The axis-ordering of the geojson file on disk.  This is assumed to
            be traditional ordering by default according to the geojson spec.

    Returns:
        GeoDataFrame : a dataframe with geo info. This will ALWAYS return
        with an OAMS_AUTHORITY_COMPLIANT wgs84 crs (i.e. lat,lon) even
        though the on disk order is should be OAMS_TRADITIONAL_GIS_ORDER.

    Example:
        >>> import io
        >>> from watch.cli.coco_align_geotiffs import *  # NOQA
        >>> geojson_text = demo_regions_geojson_text()
        >>> file = io.StringIO()
        >>> file.write(geojson_text)
        >>> file.seek(0)
        >>> region_df = read_geojson(file)
    """
    valid_axis_mappings = {
        'OAMS_TRADITIONAL_GIS_ORDER',
        'OAMS_AUTHORITY_COMPLIANT',
    }
    if default_axis_mapping not in valid_axis_mappings:
        raise Exception

    # Read custom ROI regions
    region_df = gpd.read_file(file)

    if default_axis_mapping == 'OAMS_TRADITIONAL_GIS_ORDER':
        # For whatever reason geopandas reads in geojson (which is supposed to
        # be traditional order long/lat) with a authority compliant wgs84
        # lat/long crs
        region_df['geometry'] = region_df['geometry'].apply(
            shapely_flip_xy)
    elif default_axis_mapping == 'OAMS_AUTHORITY_COMPLIANT':
        pass
    else:
        raise KeyError(default_axis_mapping)

    # TODO: can we construct a pyproj.CRS from wgs84, but with traditional
    # order?

    # import pyproj
    # wgs84 = pyproj.CRS.from_epsg(4326)
    # z = pyproj.Transformer.from_crs(4326, 4326, always_xy=True)
    # crs1 = region_df.crs
    # pyproj.CRS.from_dict(crs1.to_json_dict())
    # z = region_df.crs
    # z.to_json_dict()
    return region_df


def geopandas_pairwise_overlaps(gdf1, gdf2, predicate='intersects'):
    """
    Find pairwise relationships between each geometries

    Args:
        gdf1 (GeoDataFrame): query geo data
        gdf2 (GeoDataFrame): database geo data (builds spatial index)
        predicate (str, default='intersects'): a DE-9IM [1] predicate.
           (e.g. if intersection finds intersections between geometries)

    References:
        ..[1] https://en.wikipedia.org/wiki/DE-9IM

    Returns:
        dict: mapping from indexes in gdf1 to overlapping indexes in gdf2

    Example:
        >>> from watch.cli.coco_align_geotiffs import *  # NOQA
        >>> import geopandas as gpd
        >>> gpd.GeoDataFrame()
        >>> gdf1 = gpd.read_file(
        >>>     gpd.datasets.get_path('naturalearth_lowres')
        >>> )
        >>> gdf2 = gpd.read_file(
        >>>     gpd.datasets.get_path('naturalearth_cities')
        >>> )
        >>> mapping = geopandas_pairwise_overlaps(gdf1, gdf2)

    Benchmark:
        import timerit
        ti = timerit.Timerit(10, bestof=3, verbose=2)
        for timer in ti.reset('with sindex O(N * log(M))'):
            with timer:
                fast_mapping = geopandas_pairwise_overlaps(gdf1, gdf2)
        for timer in ti.reset('without sindex O(N * M)'):
            with timer:
                from collections import defaultdict
                slow_mapping = defaultdict(list)
                for idx1, geom1 in enumerate(gdf1.geometry):
                    slow_mapping[idx1] = []
                    for idx2, geom2 in enumerate(gdf2.geometry):
                        if geom1.intersects(geom2):
                            slow_mapping[idx1].append(idx2)
        # check they are the same
        assert set(slow_mapping) == set(fast_mapping)
        for idx1 in slow_mapping:
            slow_idx2s = slow_mapping[idx1]
            fast_idx2s = fast_mapping[idx1]
            assert sorted(fast_idx2s) == sorted(slow_idx2s)
    """
    # Construct the spatial index (requires pygeos and/or rtree)
    sindex2 = gdf2.sindex
    # For each query polygon, lookup intersecting polygons in the spatial index
    idx1_to_idxs2 = {}
    for idx1, row1 in gdf1.iterrows():
        idxs2 = sindex2.query(row1.geometry, predicate=predicate)
        # Record result indexes that "match" given the geometric predicate
        idx1_to_idxs2[idx1] = idxs2
    return idx1_to_idxs2


def latlon_text(lat, lon, precision=6):
    """
    Make a lat,lon string suitable for a filename.

    Pads with leading zeros so file names will align nicely at the same level
    of prcision.

    Args:
        lat (float): degrees latitude

        lon (float): degrees longitude

        precision (float, default=6):
            Number of trailing decimal places. As rule of thumb set this to:
                6 - for ~10cm accuracy,
                5 - for ~1m accuracy,
                2 - for ~1km accuracy,

    Notes:
        1 degree of latitude is *very* roughly the order of 100km, so the
        default precision of 6 localizes down to ~0.1 meters, which will
        usually be sufficient for satellite applications, but be mindful of
        using this text in applications that require more precision. Note 1
        degree of longitude will vary, but will always be at least as precise
        as 1 degree of latitude.

    Example:
        >>> lat = 90
        >>> lon = 180
        >>> print(latlon_text(lat, lon))
        N90.000000E180.000000

        >>> lat = 0
        >>> lon = 0
        >>> print(latlon_text(lat, lon))
        N00.000000E000.000000

    Example:
        >>> print(latlon_text(80.123, 170.123))
        >>> print(latlon_text(10.123, 80.123))
        >>> print(latlon_text(0.123, 0.123))
        N80.123000E170.123000
        N10.123000E080.123000
        N00.123000E000.123000

        >>> print(latlon_text(80.123, 170.123, precision=2))
        >>> print(latlon_text(10.123, 80.123, precision=2))
        >>> print(latlon_text(0.123, 0.123, precision=2))
        N80.12E170.12
        N10.12E080.12
        N00.12E000.12

        >>> print(latlon_text(80.123, 170.123, precision=5))
        >>> print(latlon_text(10.123, 80.123, precision=5))
        >>> print(latlon_text(0.123, 0.123, precision=5))
        N80.12300E170.12300
        N10.12300E080.12300
        N00.12300E000.12300
    """
    def _build_float_precision_fmt(num_leading, num_trailing):
        num2 = num_trailing
        # 2 extra for radix and leading sign
        num1 = num_leading + num_trailing + 2
        fmtparts = ['{:+0', str(num1), '.', str(num2), 'F}']
        fmtstr = ''.join(fmtparts)
        return fmtstr

    assert -90 <= lat <= 90, 'invalid lat'
    assert -180 <= lon <= 180, 'invalid lon'

    # Ensure latitude had 2 leading places and longitude has 3
    latfmt = _build_float_precision_fmt(2, precision)
    lonfmt = _build_float_precision_fmt(3, precision)

    lat_str = latfmt.format(lat).replace('+', 'N').replace('-', 'S')
    lon_str = lonfmt.format(lon).replace('+', 'E').replace('-', 'W')
    text = lat_str + lon_str
    return text


class SimpleDataCube(object):
    """
    Given a CocoDataset containing geotiffs, provide a simple API to extract a
    region in some coordinate space.

    Intended usage is to use :func:`query_image_overlaps` to find images that
    overlap an ROI, then then :func:`extract_overlaps` to warp spatial subsets
    of that data into an aligned temporal sequence.
    """

    def __init__(cube, dset):
        # old way: gid_to_poly is old and should be deprecated
        gid_to_poly = {}

        # new way: put data in the cube into a geopandas data frame
        df_input = []
        for gid, img in dset.imgs.items():
            info = img['geotiff_metadata']
            kw_img_poly = kwimage.Polygon(exterior=info['wgs84_corners'])
            sh_img_poly = kw_img_poly.to_shapely()
            # Create a data frame with space-time regions
            df_input.append({
                'gid': gid,
                'name': img.get('name', None),
                'video_id': img.get('video_id', None),
                'geometry': sh_img_poly,
            })
            # Maintain old way for now
            gid_to_poly[gid] = sh_img_poly

        cube.dset = dset
        cube.gid_to_poly = gid_to_poly
        cube.img_geos_df = gpd.GeoDataFrame(
            df_input, geometry='geometry', crs='epsg:4326')

    @classmethod
    def demo(SimpleDataCube, num_imgs=1, with_region=False):
        from watch.demo.landsat_demodata import grab_landsat_product
        from watch.gis.geotiff import geotiff_metadata
        # Create a dead simple coco dataset with one image
        import kwcoco
        dset = kwcoco.CocoDataset()
        ls_prod = grab_landsat_product()
        fpath = ls_prod['bands'][0]
        meta = geotiff_metadata(fpath)
        # We need a date captured ATM in a specific format
        dt = dateutil.parser.parse(
            meta['filename_meta']['acquisition_date'])
        date_captured = dt.strftime('%Y/%m/%d')

        gid = dset.add_image(file_name=fpath, date_captured=date_captured)
        img_poly = kwimage.Polygon(exterior=meta['wgs84_corners'])
        ann_poly = img_poly.scale(0.1, about='center')
        sseg_geos = ann_poly.swap_axes().to_geojson()
        dset.add_annotation(
            image_id=gid, bbox=[0, 0, 0, 0], segmentation_geos=sseg_geos)

        update_coco_geotiff_metadata(dset, serializable=False, max_workers=0)
        cube = SimpleDataCube(dset)
        if with_region:
            img_poly = kwimage.Polygon(exterior=cube.dset.imgs[1]['geotiff_metadata']['wgs84_corners'])
            img_poly.swap_axes().to_geojson()
            region_geojson =  {
                'type': 'FeatureCollection',
                'features': [
                    {
                        'type': 'Feature',
                        'properties': {'type': 'region', 'region_model_id': 'demo_region', 'version': '1.0.1', 'mgrs': None, 'start_date': None, 'end_date': None},
                        'geometry': img_poly.scale(0.2, about='center').swap_axes().to_geojson(),
                    },
                ]
            }
            region_df = gpd.GeoDataFrame.from_features(region_geojson)
            region_df['geometry'] = region_df['geometry'].apply(shapely_flip_xy)
            return cube, region_df
        return cube

    def query_image_overlaps2(cube, region_df):
        """
        Find the images that overlap with a each space-time region

        For each region, assigns all images that overlap the space-time bounds,
        and constructs arguments to :func:`extract_overlaps`.

        Args:
            region_df (GeoDataFrame): data frame containing all space-time
                region queries.

        Returns:
            dict :
                Information about which images belong to this ROI and their
                temporal sequence. Also contains strings to be used for
                subdirectories in the extract step.

        Example:
            >>> cube, region_df = SimpleDataCube.demo(with_region=True)
            >>> to_extract = cube.query_image_overlaps2(region_df)
        """
        # New maybe faster and safer way of finding overlaps?
        ridx_to_gidsx = geopandas_pairwise_overlaps(region_df, cube.img_geos_df)
        print('ridx_to_gidsx = {}'.format(ub.repr2(ridx_to_gidsx, nl=1)))

        # TODO: maybe check for self-overlap?
        # ridx_to_ridx = geopandas_pairwise_overlaps(region_df, region_df)

        to_extract = []
        for ridx, gidxs in ridx_to_gidsx.items():
            region_row = region_df.iloc[ridx]

            space_region = kwimage.Polygon.from_shapely(region_row.geometry)
            space_box = space_region.bounding_box().to_ltrb()
            latmin, lonmin, latmax, lonmax = space_box.data[0]
            min_pt = latlon_text(latmin, lonmin)
            max_pt = latlon_text(latmax, lonmax)
            space_str = '{}_{}'.format(min_pt, max_pt)

            if region_row.get('type', None) == 'region':
                # Special case where we are extracting a region with a name
                video_name = region_row.get('region_model_id', space_str)
            else:
                video_name = space_str

            if len(gidxs) == 0:
                print('WARNING: No spatial matches to {}'.format(video_name))
            else:

                # TODO: filter dates out of range
                query_start_date = region_row.get('start_date', None)
                query_end_date = region_row.get('end_date', None)

                cand_gids = cube.img_geos_df.iloc[gidxs].gid
                cand_datetimes = cube.dset.images(cand_gids).lookup('datetime_acquisition')

                if 0 and query_start_date is not None:
                    query_start_datetime = dateutil.parser.parse(query_start_date)
                    flags = [dt >= query_start_datetime for dt in cand_datetimes]
                    cand_datetimes = list(ub.compress(cand_datetimes, flags))
                    cand_gids = list(ub.compress(cand_gids, flags))

                if 0 and query_end_date is not None:
                    query_end_datetime = dateutil.parser.parse(query_end_date)
                    flags = [dt <= query_end_datetime for dt in cand_datetimes]
                    cand_datetimes = list(ub.compress(cand_datetimes, flags))
                    cand_gids = list(ub.compress(cand_gids, flags))

                if len(cand_gids) == 0:
                    print('WARNING: No temporal matches to {}'.format(video_name))
                else:
                    date_to_gids = ub.group_items(cand_gids, cand_datetimes)
                    dates = sorted(date_to_gids)
                    print('Found {} overlaps for {} from {} to {}'.format(
                        len(cand_gids),
                        video_name,
                        min(dates).isoformat(),
                        max(dates).isoformat(),
                    ))

                    region_props = ub.dict_diff(
                        region_row.to_dict(), {'geometry'})

                    image_overlaps = {
                        'date_to_gids': date_to_gids,
                        'space_region': space_region,
                        'space_str': space_str,
                        'space_box': space_box,
                        'video_name': video_name,
                        'properties': region_props,
                    }
                    to_extract.append(image_overlaps)
        return to_extract

    def extract_overlaps(cube, image_overlaps, extract_dpath,
                         rpc_align_method='orthorectify', new_dset=None,
                         write_subsets=True, visualize=True, max_workers=0,
                         aux_workers=0):
        """
        Given a region of interest, extract an aligned temporal sequence
        of data to a specified directory.

        Args:
            image_overlaps (dict): Information about images in an ROI and their
                temporal order computed from :func:``query_image_overlaps2``.

            extract_dpath (str):
                where to dump the data extracted from this ROI.

            rpc_align_method (str):
                how to handle RPC information
                (see :class:``CocoAlignGeotiffConfig`` for details)

            new_dset (kwcoco.CocoDataset | None):
                if specified, add extracted images and annotations to this
                dataset, otherwise create a new dataset.

            write_subset (bool, default=True):
                if True, write out a separate manifest file containing only
                information in this ROI.

            visualize (bool, default=True):
                if True, dump image and annotation visalizations parallel to
                the extracted data.

        Returns:
            kwcoco.CocoDataset: the given or new dataset that was modified

        Example:
            >>> cube, region_df = SimpleDataCube.demo(with_region=True)
            >>> extract_dpath = ub.ensure_app_cache_dir('smart_watch/test/coco_align_geotiff/demo_extract_overlaps')
            >>> rpc_align_method = 'orthorectify'
            >>> new_dset = kwcoco.CocoDataset()
            >>> write_subsets = True
            >>> visualize = True
            >>> max_workers = 32
            >>> to_extract = cube.query_image_overlaps2(region_df)
            >>> image_overlaps = to_extract[0]
            >>> cube.extract_overlaps(image_overlaps, extract_dpath,
            >>>                       new_dset=new_dset, visualize=visualize,
            >>>                       max_workers=max_workers)
        """
        # import watch
        dset = cube.dset

        date_to_gids = image_overlaps['date_to_gids']
        space_str = image_overlaps['space_str']
        space_box = image_overlaps['space_box']
        space_region = image_overlaps['space_region']
        video_name = image_overlaps['video_name']
        video_props = image_overlaps['properties']

        sub_bundle_dpath = ub.ensuredir((extract_dpath, video_name))

        latmin, lonmin, latmax, lonmax = space_box.data[0]
        dates = sorted(date_to_gids)

        new_video = {
            'name': video_name,
            'properties': video_props,
        }

        if new_dset is None:
            new_dset = kwcoco.CocoDataset()
        new_vidid = new_dset.add_video(**new_video)

        for cat in dset.cats.values():
            new_dset.ensure_category(**cat)

        bundle_dpath = dset.bundle_dpath
        new_anns = []

        # Manage new ids such that parallelization does not impact their order
        start_gid = new_dset._next_ids.get('images')
        start_aid = new_dset._next_ids.get('annotations')
        frame_index = 0

        # Hueristic to choose if we parallize the inner or outer loop
        # num_imgs_to_warp = sum(map(len, date_to_gids.values()))
        # if num_imgs_to_warp <= max_workers:
        #     img_workers = 0
        #     aux_workers = max_workers
        # else:
        # Internal threading might be beneficial as well
        # aux_workers = 6
        img_workers = max_workers
        # // aux_workers

        # parallelize over images
        from kwcoco.util.util_futures import JobPool
        print('img_workers = {!r}'.format(img_workers))
        print('aux_workers = {!r}'.format(aux_workers))
        pool = JobPool(mode='thread', max_workers=img_workers)

        for date in ub.ProgIter(dates, desc='submit extract jobs', verbose=1):
            gids = date_to_gids[date]
            # TODO: Is there any other consideration we should make when
            # multiple images have the same timestamp?
            for num, gid in enumerate(gids):
                img = dset.imgs[gid]
                anns = [dset.index.anns[aid] for aid in
                        dset.index.gid_to_aids[gid]]
                job = pool.submit(extract_image_job, img, anns, bundle_dpath,
                                  date, num, frame_index, new_vidid,
                                  rpc_align_method, sub_bundle_dpath,
                                  space_str, space_region, space_box,
                                  start_gid, start_aid, aux_workers)
                start_gid = start_gid + 1
                start_aid = start_aid + len(anns)
                frame_index = frame_index + 1

        sub_new_gids = []
        Prog = ub.ProgIter
        # import tqdm
        # Prog = tqdm.tqdm
        for job in Prog(pool.as_completed(), total=len(pool),
                        desc='collect extract jobs'):
            new_img, new_anns = job.result()

            # Hack, the next ids dont update when new images are added
            # with explicit ids. This is a quick fix.
            new_img.pop('id', None)

            new_img['video_id'] = new_vidid

            new_gid = new_dset.add_image(**new_img)
            sub_new_gids.append(new_gid)

            for ann in new_anns:
                ann.pop('id', None)  # quick hack fix
                ann['image_id'] = new_gid
                new_dset.add_annotation(**ann)

            if visualize:
                new_img = new_dset.imgs[new_gid]
                new_anns = new_dset.annots(gid=new_gid).objs
                _write_ann_visualizations(new_dset, new_img, new_anns,
                                          sub_bundle_dpath)

        if True:
            for new_gid in sub_new_gids:
                # Fix json serializability
                new_img = new_dset.index.imgs[new_gid]
                new_objs = [new_img] + new_img.get('auxiliary', [])

                # hack
                for obj in new_objs:
                    if 'warp_to_wld' in obj:
                        obj['warp_to_wld'] = kwimage.Affine.coerce(obj['warp_to_wld']).concise()
                    if 'wld_to_pxl' in obj:
                        obj['wld_to_pxl'] = kwimage.Affine.coerce(obj['wld_to_pxl']).concise()
                    obj.pop('wgs84_to_wld', None)
                from kwcoco.util import util_json
                assert not list(util_json.find_json_unserializable(new_img))

        if write_subsets:
            print('Writing data subset')
            new_dset._check_json_serializable()

            sub_dset = new_dset.subset(sub_new_gids, copy=True)
            sub_dset.fpath = join(sub_bundle_dpath, 'subdata.kwcoco.json')
            sub_dset.reroot(new_root=sub_bundle_dpath, absolute=False)
            sub_dset.dump(sub_dset.fpath, newlines=True)
        return new_dset


def extract_image_job(img, anns, bundle_dpath, date, num, frame_index,
                      new_vidid, rpc_align_method, sub_bundle_dpath, space_str,
                      space_region, space_box, start_gid, start_aid,
                      aux_workers=0):
    """
    Threaded worker function for :func:`SimpleDataCube.extract_overlaps`.
    """
    from watch.utils.kwcoco_extensions import _populate_canvas_obj
    from watch.utils.kwcoco_extensions import _recompute_auxiliary_transforms

    iso_time = datetime.date.isoformat(date.date())
    sensor_coarse = img.get('sensor_coarse', 'unknown')

    # Construct a name for the subregion to extract.
    name = 'crop_{}_{}_{}_{}'.format(iso_time, space_str, sensor_coarse, num)

    auxiliary = img.get('auxiliary', [])

    objs = []
    has_base_image = img.get('file_name', None) is not None
    if has_base_image:
        objs.append(ub.dict_diff(img, {'auxiliary'}))
    objs.extend(auxiliary)

    is_rpcs = [obj['geotiff_metadata']['is_rpc'] for obj in objs]
    assert ub.allsame(is_rpcs)
    is_rpc = ub.peek(is_rpcs)

    if is_rpc and rpc_align_method != 'affine_warp':
        align_method = rpc_align_method
        if align_method == 'pixel_crop':
            align_method = 'pixel_crop'
    else:
        align_method = 'affine_warp'

    dst_dpath = ub.ensuredir((sub_bundle_dpath, sensor_coarse, align_method))

    is_multi_image = len(objs) > 1

    job_list = []

    # Turn off internal threading because we refactored this to thread over all
    # iamges instead
    from kwcoco.util.util_futures import Executor
    Prog = ub.ProgIter
    # import tqdm
    # Prog = tqdm.tqdm
    executor = Executor(mode='serial', max_workers=aux_workers)
    for obj in ub.ProgIter(objs, desc='submit warp auxiliaries', verbose=0):
        job = executor.submit(
            _aligncrop, obj, bundle_dpath, name, sensor_coarse,
            dst_dpath, space_region, space_box, align_method,
            is_multi_image)
        job_list.append(job)

    dst_list = []
    for job in Prog(job_list, total=len(job_list),
                    desc='collect warp auxiliaries {}'.format(name),
                    disable=1):
        dst = job.result()
        dst_list.append(dst)

    if align_method != 'pixel_crop':
        # If we are a pixel crop, we can transform directly
        for dst in dst_list:
            # hack this in for heuristics
            if 'sensor_coarse' in img:
                dst['sensor_coarse'] = img['sensor_coarse']
            # We need to overwrite because we changed the bounds
            # Note: if band info is not popluated above, this
            # might write bad data based on hueristics
            _populate_canvas_obj(bundle_dpath, dst,
                                 overwrite={'warp'}, with_wgs=True)

    new_gid = start_gid

    new_img = {
        'id': new_gid,
        'name': name,
        'align_method': align_method,
    }

    if has_base_image:
        base_dst = dst_list[0]
        new_img.update(base_dst)
        aux_dst = dst_list[1:]
        assert len(aux_dst) == 0, 'cant have aux and base yet'
    else:
        aux_dst = dst_list

    # Hack because heurstics break when fnames change
    for old_aux, new_aux in zip(auxiliary, aux_dst):
        # new_aux['channels'] = old_aux['channels']
        new_aux['parent_file_name'] = old_aux['file_name']

    if len(aux_dst):
        new_img['auxiliary'] = aux_dst
        _recompute_auxiliary_transforms(new_img)

    carry_over = ub.dict_isect(img, {
        'date_captured',
        'approx_elevation',
        'sensor_candidates',
        'num_bands',
        'sensor_coarse',
        'site_tag',
        'channels',
    })

    # Carry over appropriate metadata from original image
    new_img.update(carry_over)
    new_img['parent_file_name'] = img['file_name']  # remember which image this came from
    # new_img['video_id'] = new_vidid  # Done outside of this worker
    new_img['frame_index'] = frame_index
    new_img['timestamp'] = date.toordinal()

    # HANDLE ANNOTATIONS
    """
    It would probably be better to warp pixel coordinates using the
    same transform found by gdalwarp, but I'm not sure how to do
    that. Thus we transform the geocoordinates to the new extracted
    img coords instead. Hopefully gdalwarp preserves metadata
    enough to do this.
    """
    new_anns = []
    geo_poly_list = []
    for ann in anns:
        # Q: WHAT FORMAT ARE THESE COORDINATES IN?
        # A: I'm fairly sure these coordinates are all Traditional-WGS84-Lon-Lat
        # We convert them to authority compliant WGS84 (lat-lon)
        # Hack to support real and orig drop0 geojson
        geo = _fix_geojson_poly(ann['segmentation_geos'])
        geo_poly = kwimage.structs.MultiPolygon.from_geojson(geo).swap_axes()
        # geo_coords = geo['coordinates'][0]
        # exterior = kwimage.Coords(np.array(geo_coords)[:, ::-1])
        # geo_poly = kwimage.Polygon(exterior=exterior)
        geo_poly_list.append(geo_poly)
    geo_polys = kwimage.SegmentationList(geo_poly_list)

    if align_method == 'orthorectify':
        # Is the affine mapping in the destination image good
        # enough after the image has been orthorectified?
        pxl_polys = geo_polys.warp(new_img['wgs84_to_wld']).warp(new_img['wld_to_pxl'])
    elif align_method == 'pixel_crop':
        raise NotImplementedError('fixme')
        yoff, xoff = new_img['transform']['st_offset']
        orig_pxl_poly_list = []
        for ann in anns:
            old_poly = kwimage.Polygon.from_coco(ann['segmentation'])
            orig_pxl_poly_list.append(old_poly)
        orig_pxl_polys = kwimage.MultiPolygon(orig_pxl_poly_list)
        pxl_polys = orig_pxl_polys.translate((-xoff, -yoff))
    elif align_method == 'affine_warp':
        # Warp Auth-WGS84 to whatever the image world space is,
        # and then from there to pixel space.
        pxl_polys = geo_polys.warp(new_img['wgs84_to_wld']).warp(new_img['wld_to_pxl'])
    else:
        raise KeyError(align_method)

    def _test_inbounds(pxl_multi_poly):
        is_any = False
        is_all = True
        for pxl_poly in pxl_multi_poly.data:
            xs, ys = pxl_poly.data['exterior'].data.T
            flags_x1 = xs < 0
            flags_y1 = ys < 0
            flags_x2 = xs >= new_img['width']
            flags_y2 = ys >= new_img['height']
            flags = flags_x1 | flags_x2 | flags_y1 | flags_y2
            n_oob = flags.sum()
            is_any &= (n_oob > 0)
            is_all &= (n_oob == len(flags))
        return is_any, is_all

    flags = [not _test_inbounds(p)[1] for p in pxl_polys]

    valid_anns = [ann.copy() for ann in ub.compress(anns, flags)]
    valid_pxl_polys = list(ub.compress(pxl_polys, flags))

    new_aid = start_aid
    for ann, pxl_poly in zip(valid_anns, valid_pxl_polys):
        ann.pop('image_id', None)
        ann['segmentation'] = pxl_poly.to_coco(style='new')
        pxl_box = pxl_poly.bounding_box().quantize().to_xywh()
        xywh = list(pxl_box.to_coco())[0]
        ann['bbox'] = xywh
        ann['image_id'] = new_gid
        ann['id'] = new_aid
        new_aid = new_aid + 1
        new_anns.append(ann)

    return new_img, new_anns


def _write_ann_visualizations(new_dset, new_img, new_anns, sub_bundle_dpath):
    """
    Helper for :func:`SimpleDataCube.extract_overlaps`.
    """
    # See if we can look at what we made
    from watch.utils.util_norm import normalize_intensity
    sensor_coarse = new_img.get('sensor_coarse', 'unknown')
    align_method = new_img.get('align_method', 'unknown')
    name = new_img.get('name', 'unknown')

    new_delayed = new_dset.delayed_load(new_img['id'])
    if hasattr(new_delayed, 'components'):
        components = new_delayed.components
    else:
        components = [new_delayed]

    for chan in components:
        spec = chan.channels.spec
        canvas = chan.finalize()

        # canvas = kwimage.imread(dst_gpath)
        canvas = normalize_intensity(canvas)
        if len(canvas.shape) > 2 and canvas.shape[2] > 4:
            # hack for wv
            canvas = canvas[..., 0]
        canvas = kwimage.ensure_float01(canvas)

        view_img_dpath = ub.ensuredir(
            (sub_bundle_dpath, sensor_coarse,
             '_view_img_' + align_method))

        view_ann_dpath = ub.ensuredir(
            (sub_bundle_dpath, sensor_coarse,
             '_view_ann_' + align_method))

        view_img_fpath = ub.augpath(name, dpath=view_img_dpath) + '_' + str(spec) + '.view_img.jpg'
        kwimage.imwrite(view_img_fpath, kwimage.ensure_uint255(canvas))

        dets = kwimage.Detections.from_coco_annots(new_anns, dset=new_dset)
        view_ann_fpath = ub.augpath(name, dpath=view_ann_dpath) + '_' + str(spec) + '.view_ann.jpg'
        ann_canvas = dets.draw_on(canvas)
        kwimage.imwrite(view_ann_fpath, kwimage.ensure_uint255(ann_canvas))


def update_coco_geotiff_metadata(dset, serializable=True, max_workers=0):
    """
    if serializable is True, then we should only update with information
    that can be coerced to json.
    """
    if serializable:
        raise NotImplementedError('we dont do this yet')

    bundle_dpath = dset.bundle_dpath

    from kwcoco.util.util_futures import JobPool
    pool = JobPool(mode='thread', max_workers=max_workers)

    img_iter = ub.ProgIter(dset.imgs.values(),
                           total=len(dset.imgs),
                           desc='submit update meta jobs',
                           verbose=1, freq=1, adjust=False)
    for img in img_iter:
        job = pool.submit(single_geotiff_metadata, bundle_dpath, img,
                          serializable=serializable)
        job.img = img

    for job in ub.ProgIter(pool.as_completed(),
                           total=len(pool),
                           desc='collect update meta jobs',
                           verbose=1, freq=1, adjust=False):
        geotiff_metadata, aux_metadata = job.result()
        img = job.img
        # # todo: can be parallelized
        # geotiff_metadata, aux_metadata = single_geotiff_metadata(
        #     bundle_dpath, img, serializable=serializable)
        img['geotiff_metadata'] = geotiff_metadata
        for aux, aux_info in zip(img.get('auxiliary', []), aux_metadata):
            aux['geotiff_metadata'] = aux_info


def single_geotiff_metadata(bundle_dpath, img, serializable=False):
    import watch
    geotiff_metadata = None
    aux_metadata = []

    img['datetime_acquisition'] = (
        dateutil.parser.parse(img['date_captured'])
    )

    # if an image specified its "dem_hint" as ignore, then we set the
    # elevation to 0. NOTE: this convention might be generalized and
    # replaced in the future. I.e. in the future the dem_hint might simply
    # specify the constant elevation to use, or perhaps something else.
    dem_hint = img.get('dem_hint', 'use')
    metakw = {}
    if dem_hint == 'ignore':
        metakw['elevation'] = 0

    # only need rpc info, wgs84_corners, and and warps
    keys_of_interest = {
        'rpc_transform',
        'is_rpc',
        'wgs84_to_wld',
        'wgs84_corners',
        'wld_to_pxl',
    }

    HACK_METADATA = 0
    if HACK_METADATA:
        # HACK: See if we can construct the keys from the metadata
        # in the coco file instead of reading the geotiff
        hack_keys = {
            'utm_corners',
            'warp_img_to_wld',
            'utm_crs_info',
            'wld_crs_info',
        }
        have_hacks = ub.dict_isect(img, hack_keys)
        if len(have_hacks) == len(hack_keys):
            print('have hacks: {}'.format(img['sensor_coarse']))
            from osgeo import osr

            def _make_osgeo_crs(crs_info):
                from osgeo import osr
                axis_mapping_int = getattr(osr, crs_info['axis_mapping'])
                auth = crs_info['auth']
                assert len(auth) == 2
                assert auth[0] == 'EPSG'
                crs = osr.SpatialReference()
                crs.ImportFromEPSG(int(auth[1]))
                crs.SetAxisMappingStrategy(axis_mapping_int)
                return crs

            wgs84_crs = osr.SpatialReference()
            wgs84_crs.ImportFromEPSG(4326)  # 4326 is the EPSG id WGS84 of lat/lon crs

            wld_to_pxl = kwimage.Affine.coerce(img['warp_img_to_wld']).inv()
            utm_crs = _make_osgeo_crs(have_hacks['utm_crs_info'])
            wld_crs = _make_osgeo_crs(have_hacks['wld_crs_info'])
            utm_to_wgs84 = osr.CoordinateTransformation(utm_crs, wgs84_crs)
            wgs84_to_wld = osr.CoordinateTransformation(wgs84_crs, wld_crs)
            utm_corners = kwimage.Coords(np.array(have_hacks['utm_corners']))
            wgs84_corners = utm_corners.warp(utm_to_wgs84)

            hack_info = {
                'rpc_transform': None,
                'is_rpc': False,
                'wgs84_to_wld': wgs84_to_wld,
                'wgs84_corners': wgs84_corners,
                'wld_to_pxl': wld_to_pxl,
            }
            geotiff_metadata = hack_info
            return geotiff_metadata
        else:
            print('missing hacks: {}'.format(img['sensor_coarse']))

    fname = img.get('file_name', None)
    if fname is not None:
        src_gpath = join(bundle_dpath, fname)
        assert exists(src_gpath)
        # img_iter.ensure_newline()
        # print('src_gpath = {!r}'.format(src_gpath))
        img_info = watch.gis.geotiff.geotiff_metadata(src_gpath, **metakw)

        if serializable:
            raise NotImplementedError
        else:
            # info['datetime_acquisition'] = img['datetime_acquisition']
            # info['gpath'] = src_gpath
            img_info = ub.dict_isect(img_info, keys_of_interest)
            geotiff_metadata = img_info

    for aux in img.get('auxiliary', []):
        aux_fpath = join(bundle_dpath, aux['file_name'])
        assert exists(aux_fpath)
        aux_info = watch.gis.geotiff.geotiff_metadata(aux_fpath, **metakw)
        aux_info = ub.dict_isect(aux_info, keys_of_interest)
        if serializable:
            raise NotImplementedError
        else:
            aux_metadata.append(aux_info)
            aux['geotiff_metadata'] = aux_info

    if fname is None:
        # need to choose one of the auxiliary images as the "main" image.
        # We are assuming that there is one auxiliary image that exactly
        # corresponds.
        candidates = []
        for aux in img.get('auxiliary', []):
            if aux['width'] == img['width'] and aux['height'] == img['height']:
                candidates.append(aux)

        if not candidates:
            raise AssertionError(
                'Assumed at least one auxiliary image has identity '
                'transform, but this seems to not be the case')
        aux = ub.peek(candidates)
        geotiff_metadata = aux['geotiff_metadata']

    img['geotiff_metadata'] = geotiff_metadata
    return geotiff_metadata, aux_metadata


def find_roi_regions(dset):
    """
    Given a dataset find spatial regions of interest that contain annotations
    """
    aid_to_poly = {}
    for aid, ann in dset.anns.items():
        geo = _fix_geojson_poly(ann['segmentation_geos'])
        # wgs84 = pyproj.CRS.from_epsg(4326)
        # wgs84_traditional = pyproj.CRS.from_epsg(4326)
        # from pyproj import CRS
        # geopandas.GeoDataFrame.from_dict(geo, crs)
        # latlon = np.array(geo['coordinates'][0])[:, ::-1]
        # kw_poly = kwimage.structs.Polygon(exterior=latlon)

        # Geojson is lon/lat, so swap to wgs84 lat/lon
        kw_poly = kwimage.structs.MultiPolygon.from_geojson(geo).swap_axes()
        aid_to_poly[aid] = kw_poly.to_shapely()

    gid_to_rois = {}
    for gid, aids in dset.index.gid_to_aids.items():
        if len(aids):
            sh_annot_polys = ub.dict_subset(aid_to_poly, aids)
            sh_annot_polys_ = [p.buffer(0) for p in sh_annot_polys.values()]
            sh_annot_polys_ = [p.buffer(0.000001) for p in sh_annot_polys_]

            # What CRS should we be doing this in? Is WGS84 OK?
            # Should we switch to UTM?
            img_rois_ = ops.cascaded_union(sh_annot_polys_)
            try:
                img_rois = list(img_rois_)
            except Exception:
                img_rois = [img_rois_]

            kw_img_rois = [
                kwimage.Polygon.from_shapely(p.convex_hull).bounding_box().to_polygons()[0]
                for p in img_rois]
            sh_img_rois = [p.to_shapely() for p in kw_img_rois]
            gid_to_rois[gid] = sh_img_rois

    # TODO: if there are only midly overlapping regions, we should likely split
    # them up. We can also group by UTM coordinates to reduce computation.
    sh_rois_ = ops.cascaded_union([
        p.buffer(0) for rois in gid_to_rois.values()
        for p in rois
    ])
    try:
        sh_rois = list(sh_rois_.geoms)
    except Exception:
        sh_rois = [sh_rois_]

    kw_rois = list(map(kwimage.Polygon.from_shapely, sh_rois))
    return sh_rois, kw_rois


def find_covered_regions(dset):
    """
    Find the intersection of all image bounding boxes in world space
    to see what spatial regions are covered by the imagery.
    """
    gid_to_poly = {}
    for gid, img in dset.imgs.items():
        info  = img['geotiff_metadata']
        kw_img_poly = kwimage.Polygon(exterior=info['wgs84_corners'])
        sh_img_poly = kw_img_poly.to_shapely()
        gid_to_poly[gid] = sh_img_poly

    # df_input = [
    #     {'gid': gid, 'bounds': poly, 'name': dset.imgs[gid].get('name', None),
    #      'video_id': dset.imgs[gid].get('video_id', None) }
    #     for gid, poly in gid_to_poly.items()
    # ]
    # img_geos = gpd.GeoDataFrame(df_input, geometry='bounds', crs='epsg:4326')

    # Can merge like this, but we lose membership info
    # coverage_df = gpd.GeoDataFrame(img_geos.unary_union)

    coverage_rois_ = ops.unary_union(gid_to_poly.values())
    if hasattr(coverage_rois_, 'geoms'):
        # Iteration over shapely objects was deprecated, test for geoms
        # attribute instead.
        coverage_rois = list(coverage_rois_.geoms)
    else:
        coverage_rois = [coverage_rois_]
    return coverage_rois


def _flip(x, y):
    return (y, x)


def shapely_flip_xy(geom):
    return ops.transform(_flip, geom)


def shapely_bounding_box(geom):
    return shapely.geometry.box(*geom.bounds)


def flip_xy(poly):
    if hasattr(poly, 'reorder_axes'):
        new_poly = poly.reorder_axes((1, 0))
    else:
        kw_poly = kwimage.Polygon.from_shapely(poly)
        kw_poly.data['exterior'].data = kw_poly.data['exterior'].data[:, ::-1]
        sh_poly_ = kw_poly.to_shapely()
        new_poly = sh_poly_
    return new_poly


def coco_geopandas_images(dset):
    df_input = []
    for gid, img in dset.imgs.items():
        info  = img['geotiff_metadata']
        kw_img_poly = kwimage.Polygon(exterior=info['wgs84_corners'])
        sh_img_poly = kw_img_poly.to_shapely()
        df_input.append({
            'gid': gid,
            'name': img.get('name', None),
            'video_id': img.get('video_id', None),
            'bounds': sh_img_poly,
        })
    img_geos_df = gpd.GeoDataFrame(df_input, geometry='bounds', crs='epsg:4326')
    return img_geos_df


def visualize_rois(dset, kw_all_box_rois):
    """
    matplotlib visualization of image and annotation regions on a world map

    Developer function, unused in the script
    """

    sh_coverage_rois = find_covered_regions(dset)
    sh_coverage_rois_trad = [flip_xy(p) for p in sh_coverage_rois]
    kw_coverage_rois_trad = list(map(kwimage.Polygon.from_shapely, sh_coverage_rois_trad))
    print('kw_coverage_rois_trad = {}'.format(ub.repr2(kw_coverage_rois_trad, nl=1)))
    cov_poly_crs = 'epsg:4326'
    cov_poly_gdf = gpd.GeoDataFrame({'cov_rois': sh_coverage_rois_trad},
                                    geometry='cov_rois', crs=cov_poly_crs)

    sh_all_box_rois = [p.to_shapely()for p in kw_all_box_rois]
    sh_all_box_rois_trad = [flip_xy(p) for p in sh_all_box_rois]
    kw_all_box_rois_trad = list(map(kwimage.Polygon.from_shapely, sh_all_box_rois_trad))
    roi_poly_crs = 'epsg:4326'
    roi_poly_gdf = gpd.GeoDataFrame({'roi_polys': sh_all_box_rois_trad},
                                    geometry='roi_polys', crs=roi_poly_crs)
    print('kw_all_box_rois_trad = {}'.format(ub.repr2(kw_all_box_rois_trad, nl=1)))

    if True:
        import kwplot
        import geopandas as gpd
        kwplot.autompl()

        wld_map_gdf = gpd.read_file(
            gpd.datasets.get_path('naturalearth_lowres')
        )
        ax = wld_map_gdf.plot()

        cov_centroids = cov_poly_gdf.geometry.centroid
        cov_poly_gdf.plot(ax=ax, facecolor='none', edgecolor='green', alpha=0.5)
        cov_centroids.plot(ax=ax, marker='o', facecolor='green', alpha=0.5)
        # img_centroids = img_poly_gdf.geometry.centroid
        # img_poly_gdf.plot(ax=ax, facecolor='none', edgecolor='red', alpha=0.5)
        # img_centroids.plot(ax=ax, marker='o', facecolor='red', alpha=0.5)

        roi_centroids = roi_poly_gdf.geometry.centroid
        roi_poly_gdf.plot(ax=ax, facecolor='none', edgecolor='orange', alpha=0.5)
        roi_centroids.plot(ax=ax, marker='o', facecolor='orange', alpha=0.5)

        kw_zoom_roi = kw_all_box_rois_trad[1]
        kw_zoom_roi = kw_coverage_rois_trad[2]
        kw_zoom_roi = kw_all_box_rois_trad[3]

        bb = kw_zoom_roi.bounding_box()

        min_x, min_y, max_x, max_y = bb.scale(1.5, about='center').to_ltrb().data[0]
        ax.set_xlim(min_x, max_x)
        ax.set_ylim(min_y, max_y)


def _fix_geojson_poly(geo):
    """
    We were given geojson polygons with one fewer layers of nesting than
    the spec allows for. Fix this.

    Example:
        >>> geo1 = kwimage.Polygon.random().to_geojson()
        >>> fixed1 = _fix_geojson_poly(geo1)
        >>> #
        >>> geo2 = {'type': 'Polygon', 'coordinates': geo1['coordinates'][0]}
        >>> fixed2 = _fix_geojson_poly(geo2)
        >>> assert fixed1 == fixed2
        >>> assert fixed1 == geo1
        >>> assert fixed2 != geo2
    """
    def check_leftmost_depth(data):
        # quick check leftmost depth of a nested struct
        item = data
        depth = 0
        while isinstance(item, list):
            if len(item) == 0:
                raise Exception('no child node')
            item = item[0]
            depth += 1
        return depth
    if geo['type'] == 'Polygon':
        data = geo['coordinates']
        depth = check_leftmost_depth(data)
        if depth == 2:
            # correctly format by adding the outer nesting
            fixed = geo.copy()
            fixed['coordinates'] = [geo['coordinates']]
        elif depth == 3:
            # already correct
            fixed = geo
        else:
            raise Exception(depth)
    else:
        fixed = geo
    return fixed


def _aligncrop(obj, bundle_dpath, name, sensor_coarse, dst_dpath, space_region,
               space_box, align_method, is_multi_image):
    # NOTE: https://github.com/dwtkns/gdal-cheat-sheet
    latmin, lonmin, latmax, lonmax = space_box.data[0]

    if is_multi_image:
        # obj.get('channels', None)
        multi_dpath = ub.ensuredir((dst_dpath, name))
        dst_gpath = join(multi_dpath, name + '_' + obj['channels'] + '.tif')
    else:
        dst_gpath = join(dst_dpath, name + '.tif')

    fname = obj.get('file_name', None)
    assert fname is not None
    src_gpath = join(bundle_dpath, fname)

    dst = {
        'file_name': dst_gpath,
    }
    if obj.get('channels', None):
        dst['channels'] = obj['channels']
    if obj.get('num_bands', None):
        dst['num_bands'] = obj['num_bands']

    if align_method == 'pixel_crop':
        align_method = 'pixel_crop'
        from ndsampler.utils.util_gdal import LazyGDalFrameFile
        imdata = LazyGDalFrameFile(src_gpath)
        info = obj['geotiff_metadata']
        space_region_pxl = space_region.warp(info['wgs84_to_wld']).warp(info['wld_to_pxl'])
        pxl_xmin, pxl_ymin, pxl_xmax, pxl_ymax = space_region_pxl.bounding_box().to_ltrb().quantize().data[0]
        sl = tuple([slice(pxl_ymin, pxl_ymax), slice(pxl_xmin, pxl_xmax)])
        subim, transform = kwimage.padded_slice(
            imdata, sl, return_info=True)
        kwimage.imwrite(dst_gpath, subim, space=None, backend='gdal')
        dst['img_shape'] = subim.shape
        dst['transform'] = transform
        # TODO: do this with a gdal command so the tiff metdata is preserved

    elif align_method == 'orthorectify':
        # HACK TO FIND an appropirate DEM file
        # from watch.gis import elevation
        # dems = elevation.girder_gtop30_elevation_dem()
        info = obj['geotiff_metadata']
        rpcs = info['rpc_transform']
        dems = rpcs.elevation

        # TODO: reproject to utm
        # https://gis.stackexchange.com/questions/193094/can-gdalwarp-reproject-from-espg4326-wgs84-to-utm
        # '+proj=utm +zone=12 +datum=WGS84 +units=m +no_defs'

        if hasattr(dems, 'find_reference_fpath'):
            dem_fpath, dem_info = dems.find_reference_fpath(latmin, lonmin)
            template = ub.paragraph(
                '''
                gdalwarp
                -te {xmin} {ymin} {xmax} {ymax}
                -te_srs epsg:4326
                -t_srs epsg:4326
                -rpc -et 0
                -to RPC_DEM={dem_fpath}
                -co TILED=YES
                -co BLOCKXSIZE=256
                -co BLOCKYSIZE=256
                -overwrite
                {SRC} {DST}
                ''')
        else:
            dem_fpath = None
            template = ub.paragraph(
                '''
                gdalwarp
                -te {xmin} {ymin} {xmax} {ymax}
                -te_srs epsg:4326
                -t_srs epsg:4326
                -rpc -et 0
                -co TILED=YES
                -co BLOCKXSIZE=256
                -co BLOCKYSIZE=256
                -overwrite
                {SRC} {DST}
                ''')
        command = template.format(
            ymin=latmin,
            xmin=lonmin,
            ymax=latmax,
            xmax=lonmax,

            dem_fpath=dem_fpath,
            SRC=src_gpath, DST=dst_gpath,
        )
        cmd_info = ub.cmd(command, verbose=0)  # NOQA
    elif align_method == 'affine_warp':
        template = (
            'gdalwarp '
            '-te {xmin} {ymin} {xmax} {ymax} '
            '-te_srs epsg:4326 '
            '-overwrite '
            '-co TILED=YES '
            '-co BLOCKXSIZE=256 '
            '-co BLOCKYSIZE=256 '
            '{SRC} {DST}')
        command = template.format(
            ymin=latmin,
            xmin=lonmin,
            ymax=latmax,
            xmax=lonmax,
            SRC=src_gpath, DST=dst_gpath,
        )
        cmd_info = ub.cmd(command, verbose=0)  # NOQA
    else:
        raise KeyError(align_method)

    return dst


_CLI = CocoAlignGeotiffConfig


if __name__ == '__main__':
    """
    CommandLine:
        python -m watch.cli.coco_align_geotiffs --help
    """
    main()
