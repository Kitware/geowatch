#!/usr/bin/env python

import json
import os

from pystac_client import Client
from shapely.geometry import shape

from config import DEFAULT_STAC_CONFIG
from file_utils import create_working_dir, get_file_from_s3, send_file_to_s3
from log_utils import get_logger

stac_config = DEFAULT_STAC_CONFIG
logger = None


def search_stac_by_geometry(
    provider, geom, collections, start, end, outfile, query, headers
):
    logger.info('Processing ' + provider)
    catalog = Client.open(provider, headers=headers)
    daterange = [start, end]
    search = catalog.search(
        collections=collections,
        datetime=daterange,
        intersects=geom,
        query=query)

    items = search.get_all_items()
    logger.info('Search found %s items' % str(len(items)))
    for item in items:
        with open(outfile, 'a') as the_file:
            the_file.write(json.dumps(item.to_dict()) + '\n')
    logger.info('Saved STAC results to: ' + outfile)


def search_stac_by_id(
    provider, collections, stac_id, outfile, query, headers
):
    logger.info('Processing ' + stac_id)
    catalog = Client.open(provider, headers=headers)
    if stac_id[-4:] == '_TCI':
        stac_id = stac_id[0:-4]
    search = catalog.search(
        collections=collections, ids=[stac_id], query=query)

    items = search.get_all_items()
    logger.info('Item found')
    for item in items:
        with open(outfile, 'a') as the_file:
            the_file.write(json.dumps(item.to_dict()) + '\n')
    logger.info('Saved STAC result to: ' + outfile)


def get_stac_query(search_item):
    if 'query' in search_item:
        query = s['query']
    else:
        query = {}
    return query


def get_stac_headers(search_item):
    if 'headers' in search_item:
        headers = s['headers']
    else:
        headers = {}
    return headers


if __name__ == '__main__':

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-o',
        '--outfile',
        required=True,
        help='output file name for STAC items'
    )
    parser.add_argument(
        '-rf',
        '--region_file',
        help='path to a region geojson file; required if mode is area'
    )
    parser.add_argument(
        '-sj',
        '--search_json',
        help='json string or path to json file containing '
             'STAC search parameters'
    )
    parser.add_argument(
        '-sf',
        '--site_file',
        help='path to a site geojson file; required if mode is id'
    )
    parser.add_argument(
        '-m',
        '--mode',
        default='id',
        help='"area" to search a bbox or "id" to provide a list of stac IDs'
    )
    parser.add_argument(
        '-s',
        '--s3_dest',
        help='s3 URI for output file'
    )
    parser.add_argument(
        '-v',
        '--verbosity',
        type=int,
        default=2,
        help='verbosity of logging [0, 1 or 2]'
    )
    args = parser.parse_args()

    logger = get_logger(verbose=args.verbosity)

    outdir = os.path.dirname(args.outfile)

    if outdir == '':
        temp_dir = create_working_dir()
        logger.info('Created temp folder: ' + temp_dir)

        dest_path = os.path.join(temp_dir, args.outfile)
    else:
        os.makedirs(outdir, exist_ok=True)

        dest_path = args.outfile

    if args.mode == 'area':
        if not hasattr(args, 'search_json'):
            raise ValueError('Missing stac search parameters')

        if not hasattr(args, 'region_file'):
            raise ValueError('Missing region file')

        logger.info('Reading STAC search JSON')
        try:
            search_params = json.loads(args.search_json)
        except json.decoder.JSONDecodeError:
            with open(args.search_json) as f:
                search_params = json.load(f)

        if args.region_file.startswith('s3://'):
            r_file_loc = get_file_from_s3(args.region_file, temp_dir)
        else:
            r_file_loc = args.region_file

        logger.info('Opening region file')
        with open(r_file_loc, 'r') as r_file:
            region = json.loads(r_file.read())

        regions = [
            f for f in region['features'] if (
                f['properties']['type'].lower() == 'region')
        ]
        if len(regions) > 0:
            # assume only 1 region per region model file
            geom = shape(regions[0]['geometry'])
            for s in search_params['stac_search']:
                search_stac_by_geometry(
                    s['endpoint'],
                    geom,
                    s['collections'],
                    s['start_date'],
                    s['end_date'],
                    dest_path,
                    get_stac_query(s),
                    get_stac_headers(s)
                )
    else:
        if args.site_file.startswith('s3://'):
            s_file_loc = get_file_from_s3(args.site_file, temp_dir)
        else:
            s_file_loc = args.site_file

        logger.info('Opening site file')
        with open(s_file_loc, 'r') as s_file:
            site = json.loads(s_file.read())

        features = site['features']

        for f in features:
            props = f['properties']
            if props['type'] == 'observation':
                sensor = props['sensor_name']
                if sensor.lower() != "worldview":
                    params = stac_config[sensor]
                    search_stac_by_id(
                        params['provider'],
                        params['collections'],
                        props['source'],
                        dest_path,
                        params['query'],
                        params['headers']
                    )

    if args.s3_dest is not None:
        logger.info('Saving output to S3')
        send_file_to_s3(dest_path, args.s3_dest)
    else:
        logger.info('--s3_dest parameter not present; skipping S3 output')

    logger.info('Search complete')

