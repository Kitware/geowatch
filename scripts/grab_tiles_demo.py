'''
A demo for grabbing the same set of Landsat and Sentinel-2 tiles 2 ways.

1. from Google Cloud using https://gitlab.kitware.com/smart/fetchLandsatSentinelFromGoogleCloud

2. from Resonant GeoData (ultimately from Google Cloud as well) using https://pypi.org/project/rgd-client/
'''
import os
import json
import dateutil
from datetime import datetime, timedelta
from fels import run_fels
# from fels import safedir_to_datetime, landsatdir_to_date
from rgd_client import Rgdc
import ubelt as ub
import scriptconfig as scfg


class GrabTilesConfig(scfg.Config):
    default = {
        'regions': scfg.Value('regions.json', help='file containing geojson space-time bounds'),
        'backend': scfg.Value('rgdc', help='either rgdc or fels'),
        'out_dpath': scfg.Value('./grab_tiles_out', help='output directory'),

        'rgdc_username': scfg.Value(None, help='username if using rgdc backend'),
        'rgdc_password': scfg.Value(None, help='password if using rgdc backend'),

        'with_l7': scfg.Value(True, help='if True, grab landsat7'),
        'with_l8': scfg.Value(True, help='if True, grab landsat8'),
        'with_s2': scfg.Value(True, help='if True, grab sentinel2'),
    }


def try_fels(geojson_bbox, dt_min, dt_max, with_l7=True, with_l8=True,
             with_s2=True, out_dpath=None):
    '''
    This provides access to:
        https://cloud.google.com/storage/docs/public-datasets/sentinel-2
        https://cloud.google.com/storage/docs/public-datasets/landsat
    '''
    if out_dpath is None:
        # put this wherever you're ok with dumping 6GB of indexes
        cats_path = os.path.expanduser('~/smart/data/fels/')
        out_path = './grab_tiles_demo/fels/'
    else:
        out_path = os.path.join(out_dpath, 'fels')
        cats_path = os.path.join(out_dpath, 'cats')

    # fels works on dates, not datetimes, and is exclusive on start and end
    pad = timedelta(days=0.5)
    date_bounds = (dt_min - pad, dt_max + pad)

    os.makedirs(out_path, exist_ok=True)

    # args are identical to the fels CLI
    # for a full list, $ fels -h

    geometry = json.dumps(geojson_bbox)

    if with_s2:
        s2_urls = run_fels(
            None,  # positional arg for tile - not needed with geometry kwarg
            'S2',
            *date_bounds,
            geometry=geometry,
            includeoverlap=True,
            outputcatalogs=cats_path,
            output=out_path,
            reject_old=True
        )  # for S2, skip redundant old-format (before Nov 2016) images

    if with_l7:
        l7_urls = run_fels(None,
                           'L7',
                           *date_bounds,
                           geometry=geometry,
                           includeoverlap=True,
                           outputcatalogs=cats_path,
                           output=out_path)

    if with_l8:
        l8_urls = run_fels(None,
                           'L8',
                           *date_bounds,
                           geometry=geometry,
                           includeoverlap=True,
                           outputcatalogs=cats_path,
                           output=out_path)

    # just for fun, print the urls and datetimes
    if with_s2:
        print('Sentinel-2:')
        print(s2_urls)
        # print([safedir_to_datetime(u.split('/')[-1]) for u in s2_urls])

    if with_l7:
        print('Landsat-7:')
        print(l7_urls)
        # print([landsatdir_to_date(u.split('/')[-1]) for u in l7_urls])

    if with_l8:
        print('Landsat-8:')
        print(l8_urls)
        # print([landsatdir_to_date(u.split('/')[-1]) for u in l8_urls])


def try_rgdc(geojson_bbox, dt_min, dt_max, out_dpath=None, username=None,
             password=None, with_l7=True, with_l8=True, with_s2=True):
    '''
    The WATCH instance of RGD is at https://watch.resonantgeodata.com/.
    You can go there to make a username and password.
    Landsat/Sentinel2 for all drop0 sites is being ingested here.
    Eventually, commercial (WV/Planet) data will live there as well.

    If that instance is not working for some reason, you can use
    the default public instance at https://www.resonantgeodata.com/.
    It has LS/S2 ingested over the KR site.

    If you do not enter your password you will be prompted for it.
    '''
    if out_dpath is None:
        out_path = './grab_tiles_demo/rgdc/'
    else:
        out_path = os.path.join(out_dpath, 'rgdc')

    client = Rgdc(username=username, password=password,
                  api_url='https://watch.resonantgeodata.com/api')
    kwargs = {
        'query': json.dumps(geojson_bbox),
        'predicate': 'intersects',
        'acquired': (dt_min, dt_max),
        #'limit': int(1e3)
    }

    results = {}

    if with_s2:
        results['s2'] = (client.search(**kwargs, instrumentation='S2A')['results'] +
                         client.search(**kwargs, instrumentation='S2B')['results'])

    if with_l7:
        results['l7'] = client.search(**kwargs, instrumentation='ETM')['results']

    if with_l8:
        results['l8'] = client.search(**kwargs, instrumentation='OLI_TIRS')['results']

    result_len = ub.map_vals(len, results)
    print('result_len = {}'.format(ub.repr2(result_len, nl=1)))

    os.makedirs(out_path, exist_ok=True)

    for search_result in ub.flatten(results.values()):
        paths = client.download_raster(search_result,
                                       out_path,
                                       nest_with_name=True,
                                       keep_existing=True)
        print(paths.path)


def coerce_regions(regions):
    if isinstance(regions, str):
        fpath = regions
        with open(fpath, 'r') as file:
            final = json.load(file)
    elif isinstance(regions, list):
        final = regions
    elif isinstance(regions, dict):
        final = [regions]
    else:
        raise TypeError(regions)
    return final


def __example__(self):
    # AOIs from drop0:
    # (coords from smart_watch_dvc/drop0_aligned/)

    # AE
    # top, left = (128.664256, 37.660143)
    # bottom, right = (128.674901, 37.663936)

    # BR
    # top, left = (-043.342788, 22.960878)
    # bottom, right = (-043.336003, 22.954186)

    # US
    # top, left = (-081.776670, 33.132338)
    # bottom, right = (-081.764686, 33.146012)

    # KR
    top, left = (128.6643, 37.6601)
    bottom, right = (128.6749, 37.6639)

    geojson_bbox = {
        "type":
        "Polygon",
        "coordinates": [[[top, left], [top, right], [bottom, right],
                         [bottom, left], [top, left]]]
    }

    # and a date range of 1 week

    dt_min, dt_max = (datetime(2018, 11, 1), datetime(2018, 11, 8))

    # run FetchLandsatSentinel

    try_fels(geojson_bbox, dt_min, dt_max)

    # run ResonantGeoDataClient

    try_rgdc(geojson_bbox, dt_min, dt_max)


def main(**kwargs):
    """
    grab_tiles_demo main

    Example:
        >>> region = {
        >>>     "type": "Feature",
        >>>     "geometry": {
        >>>         "type": "Polygon",
        >>>         "coordinates": [
        >>>             [
        >>>                 [-81.77458099915803, 33.134616000917106 ],
        >>>                 [-81.7746519999415, 33.13462000078607 ],
        >>>                 [-81.77467199973971, 33.134656000662886 ],
        >>>                 [-81.77467199973971, 33.14364699999685 ],
        >>>                 [-81.77466799967921, 33.14369799968376 ],
        >>>                 [-81.77463399945849, 33.14370499982124 ],
        >>>                 [-81.77448799923498, 33.14371699987569 ],
        >>>                 [-81.77332899925237, 33.143731999734904 ],
        >>>                 [-81.76709400004009, 33.143731999734904 ],
        >>>                 [-81.76674600019226, 33.14371699987569 ],
        >>>                 [-81.76669300004107, 33.14368199996038 ],
        >>>                 [-81.76668300022116, 33.14365299999399 ],
        >>>                 [-81.76668300022116, 33.13473900015227 ],
        >>>                 [-81.7666850003536, 33.13470800000192 ],
        >>>                 [-81.76670400030774, 33.134616000917106 ],
        >>>                 [-81.77458099915803, 33.134616000917106 ]
        >>>             ]
        >>>         ]
        >>>     },
        >>>     "properties": {
        >>>         "min_time": "2019-12-17T00:00:00",
        >>>         "max_time": "2019-12-19T00:00:00",
        >>>         "crs": {
        >>>             "note": "geojson is lon-lat"
        >>>         }
        >>>     }
        >>> }
        >>> kwargs = {}
        >>> kwargs['regions'] = region
        >>> main(**kwargs)

        >>> region = {
        >>>     "type": "Feature",
        >>>     "geometry": {
        >>>         "type": "Polygon",
        >>>         "coordinates": [
        >>>             [
        >>>                 [-81.77458099915803, 33.134616000917106 ],
        >>>                 [-81.7746519999415, 33.13462000078607 ],
        >>>                 [-81.77467199973971, 33.134656000662886 ],
        >>>                 [-81.77467199973971, 33.14364699999685 ],
        >>>                 [-81.77466799967921, 33.14369799968376 ],
        >>>                 [-81.77463399945849, 33.14370499982124 ],
        >>>                 [-81.77448799923498, 33.14371699987569 ],
        >>>                 [-81.77332899925237, 33.143731999734904 ],
        >>>                 [-81.76709400004009, 33.143731999734904 ],
        >>>                 [-81.76674600019226, 33.14371699987569 ],
        >>>                 [-81.76669300004107, 33.14368199996038 ],
        >>>                 [-81.76668300022116, 33.14365299999399 ],
        >>>                 [-81.76668300022116, 33.13473900015227 ],
        >>>                 [-81.7666850003536, 33.13470800000192 ],
        >>>                 [-81.76670400030774, 33.134616000917106 ],
        >>>                 [-81.77458099915803, 33.134616000917106 ]
        >>>             ]
        >>>         ]
        >>>     },
        >>>     "properties": {
        >>>         "min_time": "2019-12-17T00:00:00",
        >>>         "max_time": "2019-12-19T00:00:00",
        >>>         "crs": {
        >>>             "note": "geojson is lon-lat"
        >>>         }
        >>>     }
        >>> }
        >>> kwargs = {}
        >>> kwargs['regions'] = region
        >>> main(**kwargs)
    """

    config = GrabTilesConfig(default=kwargs, cmdline=True)
    regions = coerce_regions(config['regions'])
    out_dpath = config['out_dpath']

    sensor_flags = ub.dict_isect(config, {'with_l7', 'with_l8', 'with_s2'})

    for region in ub.ProgIter(regions, desc='query for geo regions', verbose=3):
        geojson_bbox = region['geometry']
        dt_min = dateutil.parser.isoparse(region['properties']['min_time'])
        dt_max = dateutil.parser.isoparse(region['properties']['max_time'])

        if config['backend'] == 'rgdc':
            try_rgdc(geojson_bbox, dt_min, dt_max, out_dpath=out_dpath,
                     username=config['rgdc_username'],
                     password=config['rgdc_password'], **sensor_flags)
        elif config['backend'] == 'fels':
            try_fels(geojson_bbox, dt_min, dt_max, out_dpath=out_dpath,
                     **sensor_flags)
        else:
            raise KeyError(config['backend'])


if __name__ == '__main__':
    main()
