'''
A demo for grabbing the same set of Landsat and Sentinel-2 tiles 2 ways.

1. from Google Cloud using https://gitlab.kitware.com/smart/fetchLandsatSentinelFromGoogleCloud [not installed with this repo]

2. from Resonant GeoData (ultimately from Google Cloud as well) using https://pypi.org/project/rgdc/ [installed with this repo]
'''

import os
import json
import numpy as np
from datetime import date, datetime, timedelta

from fels import run_fels, safedir_to_datetime, landsatdir_to_date

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

# and a date range of 1 week

dt_min, dt_max = (datetime(2018, 11, 1), datetime(2018, 11, 8))

# run FetchLandsatSentinel


def try_fels():
    # put this wherever you're ok with dumping 6GB of indexes
    cats_path = os.path.expanduser('~/smart/data/fels/')

    # fels works on dates, not datetimes, and is exclusive on start and end
    date_bounds = (dt_min - timedelta(days=1), dt_max + timedelta(days=1))

    out_path = './grab_tiles_demo/fels/'
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


# try_fels()


def try_rgdc(username, password):
    '''
    get your username and password from https://www.resonantgeodata.com/
    '''
    client = Rgdc(username=username,
                  password=password)
    kwargs = {
        'query': json.dumps(geojson_bbox),
        'predicate': 'intersects',
        'datatype': 'raster',
        'acquired': (dt_min, dt_max),
        'limit': int(1e3)
    }

    def _search(instrumentation):
        query = client.search(**kwargs, instrumentation=instrumentation)
        # fix dates
        query = ([
            entry for entry in query if (dt_min <= datetime.fromisoformat(
                entry['acquisition_date'].strip('Z')) <= dt_max)
        ])
        # fix dupes
        _, ixs = np.unique([entry['detail'] for entry in query],
                           return_index=True)
        query = list(np.array(query)[ixs])
        return query

    query_s2 = _search('S2A') + _search('S2B')
    query_l7 = _search('ETM')
    query_l8 = _search('OLI_TIRS')

    # we are missing a couple due to
    # https://github.com/ResonantGeoData/ResonantGeoData/issues/354
    print(f'S2, L7, L8: {len(query_s2)}, {len(query_l7)}, {len(query_l8)}')

    out_path = './grab_tiles_demo/rgdc/'
    os.makedirs(out_path, exist_ok=True)

    for search_result in query_s2 + query_l7 + query_l8:
        paths = client.download_raster_entry(search_result['subentry_pk'],
                                             out_path,
                                             nest_with_name=True)
        print(paths.path)

try_rgdc(username='matthew.bernstein@kitware.com', password='UMa7KqKXCaaiDmR')

