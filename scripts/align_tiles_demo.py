'''
Download all LS tiles overlapping an S2 tile in a time range and align them to the S2 grid using VRTs
'''

import os
import json
import itertools
import functools
from osgeo import gdal, osr
import numpy as np
import kwimage as ki
from datetime import datetime
from PIL import Image
import imageio
import rasterio

from watch.utils import util_raster, util_norm

from rgdc import Rgdc

from fels import safedir_to_datetime, landsatdir_to_date


def _dt(q):
    return datetime.fromisoformat(q['acquisition_date'].strip('Z'))


def _path_row(q):
    '''
    Returns LS path and row as strings. eg ('115', '034')
    '''
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
    query = sorted(query, key=_dt)
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
            path_rows = [_path_row(s) for s in split]
            paths, rows = zip(*path_rows)
            assert len(np.unique(paths)) == 1
            assert len(np.unique(rows)) == len(split)
            # skip splits that don't contain the main overlapping image
            if ('115', '034') not in path_rows:
                continue
            result.append(split)

    return result


def path_to_vrt_s2(paths, vrt_root):
    '''
    Search for Sentinel-2 band files and stack them in a VRT

    Args:
        paths: RasterMetaEntry

    Returns:
        path to the VRT
    '''
    def _bands(paths):
        return [str(p) for p in paths.images if p.match('*_B*.jp2')]

    # TODO use https://rasterio.readthedocs.io/en/latest/topics/memory-files.html
    # for these intermediate files?
    return util_raster.make_vrt(_bands(paths),
                                os.path.join(vrt_root,
                                             f'{hash(paths.path.stem)}.vrt'),
                                mode='stacked',
                                relative_to_path=os.getcwd())


def path_to_vrt_ls(paths, vrt_root):
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
        util_raster.make_vrt(
            _bands(p),
            os.path.join(vrt_root, f'{hash(p.path.stem)}.vrt'),
            mode='stacked',
            relative_to_path=os.getcwd(),
            #outputBounds=(top, left, bottom, right)
        ) for p in paths
    ]

    # then mosaic them
    final_vrt = util_raster.make_vrt(
        tmp_vrts,
        os.path.join(vrt_root, f'{hash(paths[0].path.stem + "final")}.vrt'),
        mode='mosaicked',
        relative_to_path=os.getcwd())

    if 0:  # can't do this because final_vrt still references them
        for t in tmp_vrts:
            os.remove(t)

    return final_vrt


def by_date(path):
    '''
    Sort function for a list of combined S2/LS paths
    '''
    basename = os.path.splitext(os.path.basename(path))[0]
    try:
        return safedir_to_datetime(basename).date()
    except ValueError:
        return landsatdir_to_date(basename)


def reproject_crop(in_path, bbox, vrt_root=None):
    '''
    Convert to epsg:4326 CRS and crop to common bounding box

    Unfortunately, this cannot be done in a single step in path_to_vrt_{ls|s2}
    because gdal.BuildVRT does not support warping between CRS, and the bbox wanted
    is given in epsg:4326 (not the tiles' original CRS). gdal.BuildVRTOptions has an
    outputBounds(=-te) kwarg for cropping, but not an equivalent of -te_srs.

    This means another intermediate file is necessary for each warp operation.
    
    Args:
        in_path: A georeferenced image. GTiff, VRT, etc.
        bbox: A tlbr geospatial bounding box in epsg:4326 CRS to crop to
        vrt_root: Root directory for output. If None, same dir as input.

    Returns:
        Path to a new VRT
    '''
    root, name = os.path.split(in_path)
    if vrt_root is None:
        vrt_root = root
    os.makedirs(vrt_root, exist_ok=True)
    out_path = os.path.join(vrt_root, f'{hash(name + "warp")}.vrt')
    if os.path.isfile(out_path):
        print(f'Warning: {out_path} already exists! Removing...')
        os.remove(out_path)

    dst_crs = osr.SpatialReference()
    dst_crs.ImportFromEPSG(4326)
    opts = gdal.WarpOptions(outputBounds=bbox, dstSRS=dst_crs)
    vrt = gdal.Warp(out_path, in_path, options=opts)
    del vrt

    return out_path


def thumbnail(in_path, out_path=None):
    '''
    Create a small, true-color [1] thumbnail from a satellite image.

    Args:
        in_path: path to a S2, L7, or L8 scene readable by gdal
        out_path: if None, return image content in memory

    Returns:
        out_path or image content

    References:
        [1] https://github.com/sentinel-hub/custom-scripts/tree/master
    '''
    with rasterio.open(in_path) as f:

        sat_code = os.path.basename(in_path)[:2]
        if sat_code == 'S2':
            bands = np.stack([f.read(4), f.read(3), f.read(2)], axis=-1)
        elif sat_code == 'LC':  # L8
            bands = np.stack([f.read(4), f.read(3), f.read(2)], axis=-1)
        elif sat_code == 'LE':  # L7
            bands = np.stack([f.read(3), f.read(2), f.read(1)], axis=-1)

    return Image.fromarray(util_norm.normalize_intensity(bands), 'RGB').resize(
        (1000, 1000), resample=Image.BILINEAR)


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
    s2_min_bbox = functools.reduce(lambda b1, b2: b1.intersection(b2),
                                   s2_bboxes)
    s2_min_bbox = s2_min_bbox.to_polygons()[0].to_geojson()
    kwargs['query'] = json.dumps(s2_min_bbox)

    query_l7 = client.search(**kwargs, instrumentation='ETM')
    query_l8 = client.search(**kwargs, instrumentation='OLI_TIRS')

    print(f'S2, L7, L8: {len(query_s2)}, {len(query_l7)}, {len(query_l8)}')

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
                                   keep_existing=True)
            for search_result in query
        ]

    paths_s2 = _paths(query_s2)
    paths_l7 = list(map(_paths, query_l7))
    paths_l8 = list(map(_paths, query_l8))

    '''
    # add in nodata mask (not needed for this script, just a good idea in general)
    for p in itertools.chain(paths_s2, *paths_l7, *paths_l8):
        for i in p.images:
            with rasterio.open(i, 'r+') as f:
                if not f.nodata:
                    print(f)
                    f.nodata = 0
    '''

    # convert to VRT

    vrt_root = os.path.join(out_path, 'vrt')

    paths_s2 = [path_to_vrt_s2(p, vrt_root) for p in paths_s2]
    paths_l7 = [path_to_vrt_ls(p, vrt_root) for p in paths_l7]
    paths_l8 = [path_to_vrt_ls(p, vrt_root) for p in paths_l8]

    # combine all tiles in correct time order

    datetimes = ([_dt(q) for q in query_s2] +
                 [_dt(q[0]) for q in query_l7] +
                 [_dt(q[0]) for q in query_l8])
    all_paths = [
        p for _, p in sorted(zip(datetimes, paths_s2 + paths_l7 + paths_l8))
    ]

    # orthorectification would happen here, before cropping away the margins

    all_paths = [
        reproject_crop(p, (s2_min_bbox['coordinates'][0][0] +
                           s2_min_bbox['coordinates'][0][2]))
        for p in all_paths
    ]

    imageio.mimsave('test.gif', [thumbnail(p) for p in all_paths])


if __name__ == '__main__':
    main()

# dead code:


def _bbox_from_epsg4326(path, tlbr):
    '''
    Transform a EPSG:4326 lon-lat bounding box into the CRS of a dataset

    This would be easier with rasterio, but it doesn't play nice with gdal due to
    https://rasterio.readthedocs.io/en/latest/faq.html#why-can-t-rasterio-find-proj-db-rasterio-from-pypi-versions-1-2-0
    Args:
        path: path to the dataset
        tlbr: tlbr lon-lat bounding box
    '''
    src_crs = osr.SpatialReference()
    src_crs.ImportFromEPSG(4326)
    with util_raster.gdal_open(path) as f:
        dst_crs = osr.SpatialReference(wkt=f.GetProjection())
    tfm = osr.CoordinateTransformation(src_crs, dst_crs)
    l, t, _ = tfm.TransformPoint(tlbr[1], tlbr[0])
    r, b, _ = tfm.TransformPoint(tlbr[3], tlbr[2])
    return (l, t, r, b)


def _to_uint8(arr):
    if np.issubdtype(arr.dtype, np.integer):
        return ((arr / np.iinfo(arr.dtype).max) *
                np.iinfo(np.uint8).max).astype(np.uint8)
    raise TypeError


def gif(imgs, out_path):
    '''
    Deprecated. This produces serious Bayer-pattern artifacts in output.

    Args:
        imgs: list of PIL.Image
        out_path: path to save to

    References:
        https://stackoverflow.com/a/57751793
    '''
    first, *rest = imgs
    first.save(fp=out_path,
               format='GIF',
               append_images=rest,
               save_all=True,
               duration=200,
               loop=0)
