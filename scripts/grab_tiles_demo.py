'''
A demo for grabbing the same set of Landsat and Sentinel-2 tiles 2 ways.

1. from Google Cloud using https://gitlab.kitware.com/smart/fetchLandsatSentinelFromGoogleCloud
You will need to pip install this manually.

2. from Resonant GeoData (ultimately from Google Cloud as well) using https://pypi.org/project/rgdc/
'''

import os
import json
import numpy as np
from datetime import date, datetime, timedelta

from fels import run_fels, safedir_to_datetime, landsatdir_to_date

from rgdc import Rgdc

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


def try_fels():
    '''
    This provides access to:
        https://cloud.google.com/storage/docs/public-datasets/sentinel-2
        https://cloud.google.com/storage/docs/public-datasets/landsat
    '''
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


try_fels()


def try_rgdc():
    '''
    The default public instance of RGD is https://www.resonantgeodata.com/.
    You can go there to make a username and password.
    
    The WATCH instance is at rgd.beamio.co; it is still under construction.
    Eventually, commercial (WV/Planet) data will live there as well.
    Connect to that by passing
        api_url="rgd.beamio.co/api"

    Both have LS/S2 ingested over the KR site.

    If you do not enter your password you will be prompted for it
    '''
    client = Rgdc(username='matthew.bernstein@kitware.com')
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

    out_path = './grab_tiles_demo/rgdc/'
    os.makedirs(out_path, exist_ok=True)

    for search_result in query_s2 + query_l7 + query_l8:
        paths = client.download_raster(search_result,
                                       out_path,
                                       nest_with_name=True,
                                       keep_existing=True)
        print(paths.path)


try_rgdc()
