#!/usr/bin/env python
import json
import os
import subprocess
import uuid

from pystac_client import Client
from shapely.geometry import shape
from watch.utils import util_logging
import scriptconfig as scfg


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


class StacSearchConfig(scfg.Config):
    """
    """
    default = {
        'outfile': scfg.Value(
            None,
            help='output file name for STAC items',
            short_alias=['o'],
            required=True
        ),
        'region_file': scfg.Value(
            None,
            help='path to a region geojson file; required if mode is area',
            short_alias=['rf']
        ),
        'search_json': scfg.Value(
            None,
            help='json string or path to json file containing STAC search parameters',
            short_alias=['sj']
        ),
        'site_file': scfg.Value(
            None,
            help='path to a site geojson file; required if mode is id',
            short_alias=['sf']
        ),
        'mode': scfg.Value(
            'id',
            help='"area" to search a bbox or "id" to provide a list of stac IDs',
            short_alias=['m']
        ),
        's3_dest': scfg.Value(
            None,
            help='s3 URI for output file',
            short_alias=['s']
        ),
        'verbosity': scfg.Value(
            2,
            help='verbosity of logging [0, 1 or 2]',
            type=int,
            short_alias=['v']
        ),
    }


def _make_parser():
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
    if 0:
        import scriptconfig as scfg
        import argparse
        text = scfg.Config.port_argparse(parser, 'StacSearchConfig')
        print(text)
    return parser


def main(cmdline=True, **kwargs):
    r"""
    CommandLine:
        xdoctest ~/code/watch/watch/demo/demo_region.py demo_region_fpath
        region_file=$HOME/.cache/watch/demo/regions/KHQ_R001.geojson
        start_date=$(jq -r '.features[] | select(.properties.type=="region") | .properties.start_date' $region_file)
        end_date=$(jq -r '.features[] | select(.properties.type=="region") | .properties.end_date' $region_file)
        region_id=$(jq -r '.features[] | select(.properties.type=="region") | .properties.region_id' $region_file)
        echo "start_date = $start_date"
        echo "end_date = $end_date"
        echo "region_id = $region_id"
        SMART_STAC_API_KEY='myapikey' python -m watch.cli.make_stac_search_json \
            --start_date=2018-01-01 \
            --end_date=2020-01-01 \
            --out_fpath ./stac_search.json
        cat ./stac_search.json
        python -m watch.cli.stac_search \
            -rf "$region_file" \
            -sj ./stac_search.json \
            -m area \
            --verbose 3 \
            -o "all_sensors_kit/${region_id}.input"

    Example:
        from watch.demo import demo_region
        region_fpath = demo_region.demo_khq_region_fpath()

        kwargs = {

        }
        cmdline = 0
    """
    if 1:
        config = StacSearchConfig(cmdline=cmdline, data=kwargs)
        args = config.namespace
    else:
        parser = _make_parser()
        args = parser.parse_args()

    logger = util_logging.get_logger(verbose=args.verbosity)
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
