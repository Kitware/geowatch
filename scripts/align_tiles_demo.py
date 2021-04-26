'''
Download all the LS tiles matching a reference S2 tile in a time range and align them to the S2 grid using VRTs
'''

import os
import json
import gdal
import numpy as np
import kwimage as ki
from datetime import date, datetime, timedelta
from watch.utils import util_raster

from rgdc import Rgdc

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
Get your username and password from https://www.resonantgeodata.com/
If you do not enter a pw here, you will be prompted for it
'''
client = Rgdc(username='matthew.bernstein@kitware.com')
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
s2_bboxes = ki.Boxes.concatenate(
    [ki.Polygon.from_geojson(q['footprint']).to_boxes() for q in query_s2])
s2_min_bbox = s2_bboxes[0]
for b in s2_bboxes:
    s2_min_bbox = s2_min_bbox.intersection(b)
s2_min_bbox = s2_min_bbox.to_polygons()[0].to_geojson()
kwargs['query'] = json.dumps(s2_min_bbox)

query_l7 = client.search(**kwargs, instrumentation='ETM')
query_l8 = client.search(**kwargs, instrumentation='OLI_TIRS')

print(f'S2, L7, L8: {len(query_s2)}, {len(query_l7)}, {len(query_l8)}')


def _dt(q):
    return datetime.fromisoformat(q['acquisition_date'].strip('Z'))


def _path_row(q):
    path_row = q['subentry_name'].split('_')[2]
    path, row = path_row[:3], path_row[3:]
    return path, row


def group_landsat_tiles(query, timediff_sec=300):
    '''
    Given a RGDC search query result on Landsat tiles, nest results so that scenes
    that can be merged are grouped together
  
    Args:
        timediff_sec: max allowed time between adjacent scenes.
            Should nominally be 23 seconds for Landsat 7 and 8. Some buffer is included.
    Example:
       >>> from rgdc import Rgdc
       >>> client = Rgdc('username', 'password')
       >>> query = (client.search(**kwargs, instrumentation='ETM') +
       >>>          client.search(**kwargs, instrumentation='OLI_TIRS'))
       >>> # query == [scene1, scene2, scene3, scene4]
       >>> query = group_landsat_tiles(query)
       >>> # query == [[scene1], [scene2, scene3], [scene4]]
    '''
    # ensure we're only working with one satellite at a time
    query = sorted(query, key=lambda q: q['acquisition_date'])
    sensors, ixs = np.unique([q['instrumentation'] for q in query],
                             return_index=True)
    assert set(sensors).issubset({'ETM', 'OLI_TIRS'}), sensors
    to_process = np.split(query, ixs[1:])

    result = []
    for sensor in to_process:
        # extract and split by datetimes
        dts = [_dt(q) for q in sensor]
        diffs = np.diff([dt.timestamp() for dt in dts],
                        prepend=dts[0].timestamp())
        ixs = np.where(diffs > timediff_sec)[0]
        for split in np.split(sensor, ixs):
            # each split should consist of adjacent rows in the same path
            paths, rows = zip(*[_path_row(q) for q in split])
            assert len(np.unique(paths)) == 1
            assert len(np.unique(rows)) == len(split)
            result.append(split)

    return result


query_l7 = group_landsat_tiles(query_l7)
query_l8 = group_landsat_tiles(query_l8)

# download all tiles

out_path = './align_tiles_demo/'
os.makedirs(out_path, exist_ok=True)


def _paths(query):
    return [
        client.download_raster(search_result,
                               out_path,
                               nest_with_name=True,
                               keep_existing=True) for search_result in query
    ]


paths_s2 = _paths(query_s2)
paths_l7 = list(map(_paths, query_l7))
paths_l8 = list(map(_paths, query_l8))

# convert to VRT

vrt_root = os.path.join(out_path, 'vrt')


def path_to_vrt_s2(paths):
    '''
    Search for Sentinel-2 band files and stack them in a VRT

    Args:
        paths: RasterMetaEntry

    Returns:
        path to the VRT
    '''
    def _bands(paths):
        return [str(p) for p in paths.images if p.match('*_B*.jp2')]

    return util_raster.make_vrt(_bands(paths),
                                os.path.join(vrt_root,
                                             paths.path.stem + '.vrt'),
                                mode='stacked',
                                relative_to_path=os.getcwd())


def path_to_vrt_ls(paths):
    '''
    Search for Landsat band files from compatible scenes and stack them in a single mosaicked VRT

    Args:
        paths: list(RasterMetaEntry)

    Returns:
        path to the VRT
    '''
    def _bands(paths):
        return [str(p) for p in paths.images if 'BQA' not in p.stem]

    # first make VRTs for individual tiles
    tmp_vrts = [
        util_raster.make_vrt(_bands(p),
                             os.path.join(vrt_root, p.path.stem + '.vrt'),
                             mode='stacked',
                             relative_to_path=os.getcwd()) for p in paths
    ]

    # then mosaic them
    final_vrt = util_raster.make_vrt(
        tmp_vrts,
        os.path.join(vrt_root, paths[0].path.stem + f'_{len(paths)}.vrt'),
        mode='mosaicked',
        relative_to_path=os.getcwd())

    for t in tmp_vrts:
        os.remove(t)

    return final_vrt


paths_s2 = [path_to_vrt_s2(p) for p in paths_s2]
paths_l7 = [path_to_vrt_ls(p) for p in paths_l7]
paths_l8 = [path_to_vrt_ls(p) for p in paths_l8]
