'''
Download all LS tiles overlapping the S2 tiles in a time range and align them to the S2 grid using VRTs.

Inputs:
    AOI (GeoJSON Feature); currently KR bounding box
    Datetime range; currently 01-11-2018 - 30-11-2018
    Credentials for https://www.resonantgeodata.com/

Outputs:
    Downloaded S2, L7, and L8 tiles from Resonant GeoData in 'align_tiles_demo/'
    Intermediate VRT files in 'align_tiles_demo/vrt/'
    GIF of aligned, cropped, UTM-reprojected tiles 'align_tiles_demo/test.gif'
'''

import os
import json
import functools
from osgeo import gdal, osr
import numpy as np
from datetime import datetime, timedelta
from dateutil.parser import isoparse
import utm
import pyproj
import imageio
import shapely as shp
import shapely.ops
import shapely.geometry

from watch.utils import util_raster, util_norm, util_rgdc

from rgdc import Rgdc


def scenes_to_vrt(scenes, vrt_root):
    '''
    Search for band files from compatible scenes and stack them in a single mosaicked VRT
    
    A simple wrapper around watch.utils.util_raster.make_vrt that performs both
    the 'stacked' and 'mosaicked' modes
    
    Args:
        scenes: list(scene), where scene := list(path) [of band files]
        vrt_root: root dir to save VRT under

    Returns:
        path to the VRT
    '''
    # first make VRTs for individual tiles
    # TODO use https://rasterio.readthedocs.io/en/latest/topics/memory-files.html
    # for these intermediate files?
    tmp_vrts = [
        util_raster.make_vrt(
            scene,
            os.path.join(vrt_root, f'{hash(scene[0])}.vrt'),
            mode='stacked',
            relative_to_path=os.getcwd(),
        ) for scene in scenes
    ]

    # then mosaic them
    final_vrt = util_raster.make_vrt(
        tmp_vrts,
        os.path.join(vrt_root, f'{hash(scenes[0][0] + "final")}.vrt'),
        mode='mosaicked',
        relative_to_path=os.getcwd(),
        srcNodata=0)  # this ensures nodata doesn't overwrite data in LS

    if 0:  # can't do this because final_vrt still references them
        for t in tmp_vrts:
            os.remove(t)

    return final_vrt


def epsg_code_from_latlon(lat, lon):
    _, _, zone, south = utm.from_latlon(lat, lon)
    return pyproj.CRS({
        'proj': 'utm',
        'zone': zone,
        'south': (south == 'S')
    }).to_epsg()


def reproject_crop(in_path, aoi, epsg_code=None, vrt_root=None):
    '''
    Reproject to a new CRS and crop to a common AOI

    Unfortunately, this cannot be done in a single step in scenes_to_vrt
    because gdal.BuildVRT does not support warping between CRS.
    Cropping alone could be done in scenes_to_vrt. Note gdal.BuildVRTOptions has
    an outputBounds(=-te) kwarg for cropping, but not an equivalent of -te_srs.

    This means another intermediate file is necessary for each warp operation.
    
    TODO check for this quantization error: https://gis.stackexchange.com/q/139906

    Args:
        in_path: A georeferenced image. GTiff, VRT, etc.
        aoi: A geojson Feature in epsg:4326 CRS to crop to
        epsg_code: EPSG code [1] of the CRS to convert to.
            if None, use the UTM CRS containing aoi.
        vrt_root: Root directory for output. If None, same dir as input.

    Returns:
        Path to a new VRT

    References:
        [1] http://epsg.io/
    '''
    root, name = os.path.split(in_path)
    if vrt_root is None:
        vrt_root = root
    os.makedirs(vrt_root, exist_ok=True)
    out_path = os.path.join(vrt_root, f'{hash(name + "warp")}.vrt')
    if os.path.isfile(out_path):
        print(f'Warning: {out_path} already exists! Removing...')
        os.remove(out_path)

    # find the UTM zone(s) of the AOI
    codes = [
        epsg_code_from_latlon(lat, lon) for lon, lat in aoi['coordinates'][0]
    ]
    u, counts = np.unique(codes, return_counts=True)
    if len(u) > 1:
        print(f'Warning: AOI crosses UTM zones {u}. Taking majority vote...')
    code = int(u[np.argsort(-counts)][0])

    dst_crs = osr.SpatialReference()
    dst_crs.ImportFromEPSG(code)

    bounds_crs = osr.SpatialReference()
    bounds_crs.ImportFromEPSG(4326)

    opts = gdal.WarpOptions(
        outputBounds=shp.geometry.shape(aoi).buffer(0).bounds,
        outputBoundsSRS=bounds_crs,
        dstSRS=dst_crs)
    vrt = gdal.Warp(out_path, in_path, options=opts)
    del vrt

    return out_path


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
        'datatype': 'raster',
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
    filter_fn = lambda s: filter_cloud(s, 0.5) and filter_overlap(s, 0.25)

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
        scenes_to_vrt([_bands(s, sat) for s in scenes], vrt_root)
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
