'''
A demo for grabbing the same set of Landsat and Sentinel-2 tiles 2 ways.

1. from Google Cloud using https://gitlab.kitware.com/smart/fetchLandsatSentinelFromGoogleCloud

2. from Resonant GeoData (ultimately from Google Cloud as well) using https://pypi.org/project/rgdc/
'''
import os
import json
from datetime import datetime, timedelta
from fels import run_fels, safedir_to_datetime, landsatdir_to_date
from rgdc import Rgdc
import scriptconfig as scfg


class GrabTilesConfig(scfg.Config):
    default = {
        'regions': scfg.Value('regions.json', help='file containing geojson space-time bounds'),
        'backend': scfg.Value('rgdc', help='either rgdc or fels'),
        'out_dpath': scfg.Value('./grab_tiles_out', help='output directory'),

        'rgdc_username': scfg.Value(None, help='username if using rgdc backend'),
        'rgdc_password': scfg.Value(None, help='password if using rgdc backend'),
    }


def try_fels(geojson_bbox, dt_min, dt_max, out_dpath=None):
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
    date_bounds = (dt_min - timedelta(days=1), dt_max + timedelta(days=1))

    os.makedirs(out_path, exist_ok=True)

    # args are identical to the fels CLI
    # for a full list, $ fels -h
    s2_urls = run_fels(
        None,  # positional arg for tile - not needed with geometry kwarg
        'S2',
        *date_bounds,
        geometry=json.dumps(geojson_bbox),
        outputcatalogs=cats_path,
        output=out_path,
        reject_old=True
    )  # for S2, skip redundant old-format (before Nov 2016) images

    l7_urls = run_fels(None,
                       'L7',
                       *date_bounds,
                       geometry=json.dumps(geojson_bbox),
                       outputcatalogs=cats_path,
                       output=out_path)

    l8_urls = run_fels(None,
                       'L8',
                       *date_bounds,
                       geometry=json.dumps(geojson_bbox),
                       outputcatalogs=cats_path,
                       output=out_path)

    # just for fun, print the urls and datetimes
    print('Sentinel-2:')
    print(s2_urls)
    print([safedir_to_datetime(u.split('/')[-1]) for u in s2_urls])

    print('Landsat-7:')
    print(l7_urls)
    print([landsatdir_to_date(u.split('/')[-1]) for u in l7_urls])

    print('Landsat-8:')
    print(l8_urls)
    print([landsatdir_to_date(u.split('/')[-1]) for u in l8_urls])


def try_rgdc(geojson_bbox, dt_min, dt_max, out_dpath=None, username=None,
             password=None):
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

    client = Rgdc(username=username, password=password, api_url='https://watch.resonantgeodata.com/api')
    kwargs = {
        'query': json.dumps(geojson_bbox),
        'predicate': 'intersects',
        'datatype': 'raster',
        'acquired': (dt_min, dt_max),
        #'limit': int(1e3)
    }

    query_s2 = (client.search(**kwargs, instrumentation='S2A') +
                client.search(**kwargs, instrumentation='S2B'))
    query_l7 = client.search(**kwargs, instrumentation='ETM')
    query_l8 = client.search(**kwargs, instrumentation='OLI_TIRS')

    print(f'S2, L7, L8: {len(query_s2)}, {len(query_l7)}, {len(query_l8)}')

    os.makedirs(out_path, exist_ok=True)

    for search_result in query_s2 + query_l7 + query_l8:
        paths = client.download_raster(search_result,
                                       out_path,
                                       nest_with_name=True,
                                       keep_existing=True)
        print(paths.path)


def coerce_regions(regions):
    if isinstance(regions, str):
        fpath = regions
        with open(fpath, 'r'):
            final = json.load(fpath)
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
    region = {
        "region_geos": {
            "type": "Polygon",
            "coordinates": [
                [
                    [ 23.960642000147992, 52.22258200006882 ],
                    [ 23.96059500018642, 52.22263300002096 ],
                    [ 23.96055600043741, 52.222677000030835 ],
                    [ 23.96055600043741, 52.24351699884394 ],
                    [ 23.960570000193663, 52.24358099939994 ],
                    [ 23.960658000159356, 52.24380599979307 ],
                    [ 23.973363999981217, 52.24380599979307 ],
                    [ 23.974296999922018, 52.24374199985515 ],
                    [ 23.974383999746383, 52.243709998811276 ],
                    [ 23.974490999993762, 52.24363999881372 ],
                    [ 23.974613999896015, 52.24212599985994 ],
                    [ 23.974613999896015, 52.22564900009239 ],
                    [ 23.974490999993762, 52.2227520000801 ],
                    [ 23.974416999751664, 52.22263300002096 ],
                    [ 23.974383999746383, 52.22261800008218 ],
                    [ 23.973412999963436, 52.22258200006882 ],
                    [ 23.960642000147992, 52.22258200006882 ]
                ]
            ]
        },
        "max_time": "2020-01-02",
        "min_time": "2013-04-01"
    }
    kwargs = {}
    kwargs['regions'] = region
    """

    config = GrabTilesConfig(default=kwargs, cmdline=True)
    regions = coerce_regions(config['regions'])
    out_dpath = config['out_dpath']

    from watch.utils.util_time import Timestamp
    for region in regions:
        geojson_bbox = region['region_geos']
        dt_min = Timestamp.coerce(region['min_time']).to_datetime()
        dt_max = Timestamp.coerce(region['max_time']).to_datetime()

        if config['backend'] == 'rgdc':
            try_rgdc(geojson_bbox, dt_min, dt_max, out_dpath=out_dpath,
                     username=config['rgdc_username'],
                     password=config['rgdc_password'])
        elif config['backend'] == 'fels':
            try_fels(geojson_bbox, dt_min, dt_max, out_dpath=out_dpath)
        else:
            raise KeyError(config['backend'])


if __name__ == '__main__':
    main()
