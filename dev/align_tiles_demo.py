'''
Download all LS tiles overlapping the S2 tiles in a time range and align them to the S2 grid using VRTs.

Inputs:
    AOI (GeoJSON Feature); currently KR bounding box
    Datetime range; currently 01-11-2018 - 30-11-2018
    Credentials for https://watch.resonantgeodata.com/

Outputs:
    Downloaded S2, L7, and L8 tiles from Resonant GeoData in 'align_tiles_demo/'
    Intermediate VRT files in 'align_tiles_demo/vrt/'
    GIF of aligned, cropped, UTM-reprojected tiles 'align_tiles_demo/test.gif'
'''

import os
import json
import functools
import numpy as np
from datetime import datetime, timedelta
from dateutil.parser import isoparse
import imageio
import shapely as shp
import shapely.ops
import shapely.geometry
from osgeo import gdal
import shapely
from copy import deepcopy
from lxml import etree
from tempfile import NamedTemporaryFile
from watch.utils import util_norm, util_rgdc
from watch.gis.spatial_reference import utm_epsg_from_latlon

from rgd_client import Rgdc


def reproject_crop(in_path, aoi, code=None, out_path=None, vrt_root=None):
    """
    Crop an image to an AOI and reproject to its UTM CRS (or another CRS)

    Unfortunately, this cannot be done in a single step in scenes_to_vrt
    because gdal.BuildVRT does not support warping between CRS.
    Cropping alone could be done in scenes_to_vrt. Note gdal.BuildVRTOptions
    has an outputBounds(=-te) kwarg for cropping, but not an equivalent of
    -te_srs.

    This means another intermediate file is necessary for each warp operation.

    TODO check for this quantization error:
        https://gis.stackexchange.com/q/139906

    Args:
        in_path: A georeferenced image. GTiff, VRT, etc.
        aoi: A geojson Feature in epsg:4326 CRS to crop to.
        code: EPSG code [1] of the CRS to convert to.
            if None, use the UTM CRS containing aoi.
        out_path: Name of output file to write to. If None, create a VRT file.
        vrt_root: Root directory for VRT output. If None, same dir as input.

    Returns:
        Path to a new VRT or out_path

    References:
        [1] http://epsg.io/

    Example:
        >>> # xdoctest: +REQUIRES(--network)
        >>> import os
        >>> from osgeo import gdal
        >>> from watch.demo.landsat_demodata import grab_landsat_product
        >>> band1 = grab_landsat_product()['bands'][0]
        >>> #
        >>> # pick the AOI from the drop0 KR site
        >>> # (this doesn't actually intersect the demodata)
        >>> top, left = (128.6643, 37.6601)
        >>> bottom, right = (128.6749, 37.6639)
        >>> geojson_bbox = {
        >>>     "type":
        >>>     "Polygon",
        >>>     "coordinates": [[[top, left], [top, right], [bottom, right],
        >>>                      [bottom, left], [top, left]]]
        >>> }
        >>> #
        >>> out_path = reproject_crop(band1, geojson_bbox)
        >>> #
        >>> # clean up
        >>> os.remove(out_path)
    """
    if out_path is None:
        root, name = os.path.split(in_path)
        if vrt_root is None:
            vrt_root = root
        os.makedirs(vrt_root, exist_ok=True)
        out_path = os.path.join(vrt_root, f'{hash(name + "warp")}.vrt')
        if os.path.isfile(out_path):
            print(f'Warning: {out_path} already exists! Removing...')
            os.remove(out_path)

    if code is None:
        # find the UTM zone(s) of the AOI
        codes = [
            utm_epsg_from_latlon(lat, lon)
            for lon, lat in aoi['coordinates'][0]
        ]
        u, counts = np.unique(codes, return_counts=True)
        if len(u) > 1:
            print(
                f'Warning: AOI crosses UTM zones {u}. Taking majority vote...')
        code = int(u[np.argsort(-counts)][0])

    dst_crs = osr.SpatialReference()
    dst_crs.ImportFromEPSG(code)

    bounds_crs = osr.SpatialReference()
    bounds_crs.ImportFromEPSG(4326)

    opts = gdal.WarpOptions(
        outputBounds=shapely.geometry.shape(aoi).buffer(0).bounds,
        outputBoundsSRS=bounds_crs,
        dstSRS=dst_crs)
    vrt = gdal.Warp(out_path, in_path, options=opts)
    del vrt

    return out_path


def scenes_to_vrt(scenes, vrt_root, relative_to_path):
    """
    Search for band files from compatible scenes and stack them in a single
    mosaicked VRT

    A simple wrapper around make_vrt that performs both
    the 'stacked' and 'mosaicked' modes

    Args:
        scenes: list(scene), where scene := list(path) [of band files]
        vrt_root: root dir to save VRT under

    Returns:
        path to the VRT

    Example:
        >>> # xdoctest: +REQUIRES(--network)
        >>> import os
        >>> from osgeo import gdal
        >>> from watch.demo.landsat_demodata import grab_landsat_product
        >>> bands = grab_landsat_product()['bands']
        >>> #
        >>> # pretend there are more scenes here
        >>> out_path = scenes_to_vrt([sorted(bands)] , vrt_root='.', relative_to_path=os.getcwd())
        >>> with GdalOpen(out_path) as f:
        >>>     print(f.GetDescription())
        >>> #
        >>> # clean up
        >>> os.remove(out_path)
    """
    # first make VRTs for individual tiles
    # TODO use https://rasterio.readthedocs.io/en/latest/topics/memory-files.html
    # for these intermediate files?
    tmp_vrts = [
        make_vrt(scene,
                 os.path.join(vrt_root, f'{hash(scene[0])}.vrt'),
                 mode='stacked',
                 relative_to_path=relative_to_path) for scene in scenes
    ]

    # then mosaic them
    final_vrt = make_vrt(tmp_vrts,
                         os.path.join(vrt_root,
                                      f'{hash(scenes[0][0] + "final")}.vrt'),
                         mode='mosaicked',
                         relative_to_path=relative_to_path)

    if 0:  # can't do this because final_vrt still references them
        for t in tmp_vrts:
            os.remove(t)

    return final_vrt


def make_vrt(in_paths, out_path, mode, relative_to_path=None, **kwargs):
    """
    Stack multiple band files in the same directory into a single VRT

    Args:
        in_paths: list(path)
        out_path: path to save to; standard is '*.vrt'. If None, a path will be
            generated.
        mode:
            'stacked': Stack multiple band files covering the same area
            'mosaicked': Mosaic/merge scenes with overlapping areas. Content
                will be taken from the first in_path without nodata.
        relative_to_path: if this function is being called from another
            process, pass in the cwd of the calling process, to trick gdal into
            creating a rerootable VRT
        kwargs: passed to gdal.BuildVRTOptions [1,2]

    Returns:
        path to VRT

    Example:
        >>> # xdoctest: +REQUIRES(--network)
        >>> import os
        >>> from osgeo import gdal
        >>> from watch.demo.landsat_demodata import grab_landsat_product
        >>> bands = grab_landsat_product()['bands']
        >>> #
        >>> # stack bands from a scene
        >>> make_vrt(sorted(bands), './bands1.vrt', mode='stacked', relative_to_path=os.getcwd())
        >>> # pretend this is a different scene
        >>> make_vrt(sorted(bands), './bands2.vrt', mode='stacked', relative_to_path=os.getcwd())
        >>> # now, if they overlap, mosaic/merge them
        >>> make_vrt(['./bands1.vrt', './bands2.vrt'], 'full_scene.vrt', mode='mosaicked', relative_to_path=os.getcwd())
        >>> with GdalOpen('full_scene.vrt') as f:
        >>>     print(f.GetDescription())
        >>> #
        >>> # clean up
        >>> os.remove('bands1.vrt')
        >>> os.remove('bands2.vrt')
        >>> os.remove('full_scene.vrt')

    References:
        [1] https://gdal.org/programs/gdalbuildvrt.html
        [2] https://gdal.org/python/osgeo.gdal-module.html#BuildVRTOptions
    """

    if mode == 'stacked':
        kwargs['separate'] = True
    elif mode == 'mosaicked':
        kwargs['separate'] = False
        kwargs['srcNodata'] = 0  # this ensures nodata doesn't overwrite data
    else:
        raise ValueError(f'mode: {mode} should be "stacked" or "mosaicked"')

    # set sensible defaults
    if 'resolution' not in kwargs:
        kwargs['resolution'] = 'highest'
    if 'resampleAlg' not in kwargs:
        kwargs['resampleAlg'] = 'bilinear'

    opts = gdal.BuildVRTOptions(**kwargs)

    if len(in_paths) > 1:
        common = os.path.commonpath(in_paths)
    else:
        common = os.path.dirname(in_paths[0])

    if relative_to_path is None:
        relative_to_path = os.path.dirname(os.path.abspath(__file__))

    # validate out_path
    if out_path is not None:
        out_path = os.path.abspath(out_path)
        if os.path.splitext(out_path)[1]:  # is a file
            os.makedirs(os.path.dirname(out_path), exist_ok=True)
        elif os.path.isdir(out_path):  # is a dir
            raise ValueError(f'{out_path} is an existing directory.')

    # generate an unused name
    with NamedTemporaryFile(dir=common,
                            suffix='.vrt',
                            mode='r+',
                            delete=(out_path is not None)) as f:

        # First, create VRT in a place where it can definitely see the input
        # files.  Use a relative instead of absolute path to ensure that
        # <SourceFilename> refs are relative, and therefore the VRT is
        # rerootable
        vrt = gdal.BuildVRT(os.path.relpath(f.name, start=relative_to_path),
                            in_paths,
                            options=opts)
        del vrt  # write to disk

        # then, move it to the desired location
        if out_path is None:
            out_path = f.name
        elif os.path.isfile(out_path):
            print(f'warning: {out_path} already exists! Removing...')
            os.remove(out_path)
        reroot_vrt(f.name, out_path, keep_old=True)

    return out_path


def reroot_vrt(old_path, new_path, keep_old=True):
    """
    Copy a VRT file, fixing relative paths to its component images

    Example:
        >>> # xdoctest: +REQUIRES(--network)
        >>> import os
        >>> from osgeo import gdal
        >>> from watch.demo.landsat_demodata import grab_landsat_product
        >>> bands = grab_landsat_product()['bands']
        >>> #
        >>> # VRT must be created in the imgs' subtree
        >>> tmp_path = os.path.join(os.path.dirname(bands[0]), 'all_bands.vrt')
        >>> # (consider using the wrapper make_vrt instead of this)
        >>> gdal.BuildVRT(tmp_path, sorted(bands))
        >>> # now move it somewhere more convenient
        >>> reroot_vrt(tmp_path, './bands.vrt', keep_old=False)
        >>> #
        >>> # clean up
        >>> os.remove('bands.vrt')
    """
    if os.path.abspath(old_path) == os.path.abspath(new_path):
        return

    path_diff = os.path.relpath(os.path.dirname(os.path.abspath(old_path)),
                                start=os.path.dirname(
                                    os.path.abspath(new_path)))

    tree = deepcopy(etree.parse(old_path))
    for elem in tree.iterfind('.//SourceFilename'):
        if elem.get('relativeToVRT') == '1':
            elem.text = os.path.join(path_diff, elem.text)
        else:
            if not os.path.isabs(elem.text):
                raise ValueError(f'''VRT file:
                    {old_path}
                cannot be rerooted because it contains path:
                    {elem.text}
                relative to an unknown location [the original calling location].
                To produce a rerootable VRT, call gdal.BuildVRT() with out_path relative to in_paths.'''
                                 )
            if not os.path.isfile(elem.text):
                raise ValueError(f'''VRT file:
                    {old_path}
                references an nonexistent path:
                    {elem.text}''')

    with open(new_path, 'wb') as f:
        tree.write(f, encoding='utf-8')

    if not keep_old:
        os.remove(old_path)


def main():

    # pick the AOI from the drop0 KR site

    top, left = (128.6643, 37.6601)
    bottom, right = (128.6749, 37.6639)

    geojson_bbox = {
        "type":
        "Polygon",
        "coordinates": [[[top, left], [top, right], [bottom, right],
                         [bottom, left], [top, left]]]
    }

    # and a date range of 1 month

    dt_min, dt_max = (datetime(2018, 11, 1), datetime(2018, 11, 30))

    # query S2 tiles
    '''
    Get your username and password from https://watch.resonantgeodata.com/
    If you do not enter a pw here, you will be prompted for it
    '''
    client = Rgdc(username='matthew.bernstein@kitware.com',
                  api_url='https://watch.resonantgeodata.com/api/')
    kwargs = {
        'query': json.dumps(geojson_bbox),
        'predicate': 'intersects',
        'acquired': (dt_min, dt_max)
    }

    query_s2 = (client.search(**kwargs, instrumentation='S2A') +
                client.search(**kwargs, instrumentation='S2B'))

    # match AOI and query LS tiles

    # S2 scenes do not overlap perfectly - get their intersection
    s2_bboxes = [
        shp.geometry.shape(q['footprint']).buffer(0) for q in query_s2
    ]
    s2_min_bbox = functools.reduce(lambda b1, b2: b1.intersection(b2),
                                   s2_bboxes)
    s2_min_bbox = shp.geometry.mapping(s2_min_bbox)
    kwargs['query'] = json.dumps(s2_min_bbox)

    query_l7 = client.search(**kwargs, instrumentation='ETM')
    query_l8 = client.search(**kwargs, instrumentation='OLI_TIRS')

    print(f'S2, L7, L8: {len(query_s2)}, {len(query_l7)}, {len(query_l8)}')

    query_s2 = util_rgdc.group_tiles(
        query_s2)  # this has no effect for this AOI
    query_l7 = util_rgdc.group_tiles(query_l7)
    query_l8 = util_rgdc.group_tiles(query_l8)

    def as_stac(scene):
        # this is a temporary workaround until STAC endpoint is in the client
        import requests
        return requests.get(scene['detail'] + '/stac').json()

    def filter_cloud(scenes, max_cloud_frac):
        # not all entries have this field
        cloud_frac = np.mean([
            as_stac(scene)['properties'].get('eo:cloud_cover', 0)
            for scene in scenes
        ])
        return cloud_frac <= max_cloud_frac

    def filter_overlap(scenes, min_overlap):
        polys = [as_stac(scene)['geometry'] for scene in scenes]
        polys = [shp.geometry.shape(p).buffer(0) for p in polys]
        u = shp.ops.cascaded_union(polys)
        aoi = shp.geometry.shape(s2_min_bbox)
        overlap = u.intersection(aoi).area / aoi.area
        return overlap >= min_overlap

    # filter out unwanted scenes before downloading
    # overlap should have no effect for query_s2
    def filter_fn(s):
        return filter_cloud(s, 0.5) and filter_overlap(s, 0.25)

    query = query_s2 + query_l7 + query_l8
    query = list(filter(filter_fn, query))

    print(f'{len(query)} filtered and merged scenes')

    # sort query in correct time order,
    # with sat_codes to keep track of which are from each satellite

    datetimes = [isoparse(q[0]['acquisition_date']) for q in query]
    datetimes, query = zip(*sorted(zip(datetimes, query)))

    nice = {'S2A': 'S2', 'S2B': 'S2', 'OLI_TIRS': 'L8', 'ETM': 'L7'}
    sat_codes = [nice[q[0]['instrumentation']] for q in query]

    # download all tiles

    out_path = './align_tiles_demo/'
    os.makedirs(out_path, exist_ok=True)

    def _paths(query):
        return [
            client.download_raster(search_result,
                                   out_path,
                                   nest_with_name=True,
                                   keep_existing=True)
            for search_result in query
        ]

    paths = [_paths(q) for q in query]

    # convert to VRT

    def _bands(paths, sat):
        if sat == 'S2':
            return util_rgdc.bands_sentinel2(paths)
        elif sat == 'L7' or sat == 'L8':
            return util_rgdc.bands_landsat(paths)

    vrt_root = os.path.join(out_path, 'vrt')

    paths = [
        scenes_to_vrt([_bands(s, sat) for s in scenes], vrt_root, os.getcwd())
        for scenes, sat in zip(paths, sat_codes)
    ]

    # orthorectification would happen here, before cropping away the margins

    paths = [reproject_crop(p, s2_min_bbox) for p in paths]

    # output GIF of thumbnails

    # 1 second per 5 days between images
    diffs = np.diff(datetimes, prepend=(datetimes[0] - timedelta(days=5)))
    duration = list((diffs / timedelta(days=5)).astype(float))
    imageio.mimsave(
        os.path.join(out_path, 'test.gif'),
        [util_norm.thumbnail(p, sat) for p, sat in zip(paths, sat_codes)],
        duration=duration)


if __name__ == '__main__':
    main()
