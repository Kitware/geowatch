#!/usr/bin/env python
import json
import os
import subprocess
import uuid

from pystac_client import Client
from shapely.geometry import shape
import logging.config


if 'SMART_STAC_API_KEY' in os.environ:
    smart_stac_api_key = os.environ['SMART_STAC_API_KEY']
else:
    smart_stac_api_key = None


DEFAULT_STAC_CONFIG = {
    #"Landsat 8": {
    #    "provider": "https://api.smart-stac.com",
    #    "collections": ["landsat-c2l1"],
    #    "headers": {
    #        "x-api-key": smart_stac_api_key
    #    },
    #    "query": {}
    #},
    "Landsat 8": {
        "provider": "https://landsatlook.usgs.gov/stac-server/",
        "collections": ["landsat-c2l1"],
        "headers": {},
        "query": {}
    },
    "Sentinel-2": {
        "provider": "https://earth-search.aws.element84.com/v0",
        "collections": ["sentinel-s2-l1c"],
        "query": {},
        "headers": {}
    }
}

stac_config = DEFAULT_STAC_CONFIG


def setup_logging(verbose=1):
    """
    Define logging level

    Args:
        verbose (int):
            Accepted values:
                * 0: no logging
                * 1: INFO level
                * 2: DEBUG level

    TODO:
        - [ ] standardized loggers should probably be in watch.util
    """

    log_med = "%(asctime)s-15s %(name)-32s [%(levelname)-8s] %(message)s"
    log_large = "%(asctime)s-15s %(name)-32s [%(levelname)-8s] "
    log_large += "(%(module)-17s) %(message)s"

    log_config = {}

    if verbose == 0:
        log_config = {
            "version": 1,
            "disable_existing_loggers": False,
            "handlers": {
                "null": {"level": "DEBUG", "class": "logging.NullHandler"}
            },
            "loggers": {
                "watchlog": {
                    "handlers": ["null"],
                    "propagate": True,
                    "level": "INFO"
                }
            },
        }
    elif verbose == 1:
        log_config = {
            "version": 1,
            "disable_existing_loggers": False,
            "formatters": {
                "standard": {
                    "format": log_med
                }
            },
            "handlers": {
                "console": {
                    "level": "DEBUG",
                    "class": "logging.StreamHandler",
                    "formatter": "standard",
                }
            },
            "loggers": {
                "watchlog": {
                    "handlers": ["console"],
                    "propagate": True,
                    "level": "INFO",
                }
            },
        }
    elif verbose == 2:
        log_config = {
            "version": 1,
            "disable_existing_loggers": False,
            "formatters": {
                "verbose": {
                    "format": log_large
                }
            },
            "handlers": {
                "console": {
                    "level": "DEBUG",
                    "class": "logging.StreamHandler",
                    "formatter": "verbose",
                }
            },
            "loggers": {
                "watchlog": {
                    "handlers": ["console"],
                    "propagate": True,
                    "level": "DEBUG",
                }
            },
        }
    else:
        raise ValueError("'verbose' must be one of: 0, 1, 2")
    return log_config


def get_logger(verbose=1):
    logcfg = setup_logging(verbose)
    logging.config.dictConfig(logcfg)
    logger = logging.getLogger('watchlog')
    return logger


def create_working_dir():
    rand_id = uuid.uuid4().hex
    temp_dir = os.path.join('/tmp', rand_id)
    os.makedirs(temp_dir, exist_ok=True)

    return temp_dir


def get_file_from_s3(uri, path):
    dst_path = os.path.join(path, os.path.basename(uri))

    try:
        subprocess.check_call(
            ['aws', 's3', 'cp', '--quiet', uri, dst_path]
        )
    except Exception:
        raise OSError('Error getting file from s3URI: ' + uri)

    return dst_path


def send_file_to_s3(path, uri):
    try:
        subprocess.check_call(
            ['aws', 's3', 'cp', '--quiet', path, uri]
        )
    except Exception:
        raise OSError('Error sending file to s3URI: ' + uri)

    return uri


class StacSearcher:
    def __init__(self, logger):
        self.logger = logger

    def by_geometry(self, provider, geom, collections, start, end, outfile,
                    query, headers):
        self.logger.info('Processing ' + provider)
        catalog = Client.open(provider, headers=headers)
        daterange = [start, end]
        search = catalog.search(
            collections=collections,
            datetime=daterange,
            intersects=geom,
            query=query)

        items = search.get_all_items()
        self.logger.info('Search found %s items' % str(len(items)))
        for item in items:
            with open(outfile, 'a') as the_file:
                the_file.write(json.dumps(item.to_dict()) + '\n')
        self.logger.info('Saved STAC results to: ' + outfile)

    def by_id(self, provider, collections, stac_id, outfile, query, headers):
        self.logger.info('Processing ' + stac_id)
        catalog = Client.open(provider, headers=headers)
        if stac_id[-4:] == '_TCI':
            stac_id = stac_id[0:-4]
        search = catalog.search(
            collections=collections, ids=[stac_id], query=query)

        items = search.get_all_items()
        self.logger.info('Item found')
        for item in items:
            with open(outfile, 'a') as the_file:
                the_file.write(json.dumps(item.to_dict()) + '\n')
        self.logger.info('Saved STAC result to: ' + outfile)


def main():
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
    search_stac = StacSearcher(logger)

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
                search_stac.by_geometry(
                    s['endpoint'],
                    geom,
                    s['collections'],
                    s['start_date'],
                    s['end_date'],
                    dest_path,
                    s.get('query', {}),
                    s.get('headers', {})
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
                    search_stac.by_id(
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


if __name__ == '__main__':
    main()
