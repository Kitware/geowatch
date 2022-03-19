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

from watch.utils import util_gdal
from watch.utils import util_norm, util_rgdc

from rgd_client import Rgdc


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
        util_gdal.scenes_to_vrt([_bands(s, sat) for s in scenes], vrt_root, os.getcwd())
        for scenes, sat in zip(paths, sat_codes)
    ]

    # orthorectification would happen here, before cropping away the margins

    paths = [util_gdal.reproject_crop(p, s2_min_bbox) for p in paths]

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
