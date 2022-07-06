#!/usr/bin/env python
r"""
Performs the stac search to create the .input file needed for
prepare_ta2_dataset

CommandLine:
    # Create a demo region file
    xdoctest watch.demo.demo_region demo_khq_region_fpath

    DATASET_SUFFIX=DemoKHQ-2022-06-10-V2
    DEMO_DPATH=$HOME/.cache/watch/demo/datasets
    REGION_FPATH="$HOME/.cache/watch/demo/annotations/KHQ_R001.geojson"
    SITE_GLOBSTR="$HOME/.cache/watch/demo/annotations/KHQ_R001_sites/*.geojson"
    START_DATE=$(jq -r '.features[] | select(.properties.type=="region") | .properties.start_date' "$REGION_FPATH")
    END_DATE=$(jq -r '.features[] | select(.properties.type=="region") | .properties.end_date' "$REGION_FPATH")
    REGION_ID=$(jq -r '.features[] | select(.properties.type=="region") | .properties.region_id' "$REGION_FPATH")
    SEARCH_FPATH=$DEMO_DPATH/stac_search.json
    RESULT_FPATH=$DEMO_DPATH/all_sensors_kit/${REGION_ID}.input

    mkdir -p "$DEMO_DPATH"

    # Create the search json wrt the sensors and processing level we want
    python -m watch.cli.stac_search_build \
        --start_date="$START_DATE" \
        --end_date="$END_DATE" \
        --cloud_cover=40 \
        --sensors=L2 \
        --out_fpath "$SEARCH_FPATH"
    cat "$SEARCH_FPATH"

    # Delete this to prevent duplicates
    rm -f "$RESULT_FPATH"
    # Create the .input file
    python -m watch.cli.stac_search \
        -rf "$REGION_FPATH" \
        -sj "$SEARCH_FPATH" \
        -m area \
        --verbose 2 \
        -o "${RESULT_FPATH}"

    # Construct the TA2-ready dataset
    python -m watch.cli.prepare_ta2_dataset \
        --dataset_suffix=$DATASET_SUFFIX \
        --s3_fpath "${RESULT_FPATH}" \
        --collated False \
        --dvc_dpath="$DEMO_DPATH" \
        --aws_profile=iarpa \
        --region_globstr="$REGION_FPATH" \
        --site_globstr="$SITE_GLOBSTR" \
        --fields_workers=8 \
        --convert_workers=8 \
        --align_workers=26 \
        --cache=0 \
        --ignore_duplicates=0 \
        --visualize=True \
        --serial=True --run=1
"""
import json
import os
import subprocess
import uuid
import pystac_client
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
        catalog = pystac_client.Client.open(provider, headers=headers)
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
        catalog = pystac_client.Client.open(provider, headers=headers)
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
        'verbose': scfg.Value(
            2,
            help='verbose of logging [0, 1 or 2]',
            type=int,
            short_alias=['v']
        ),
    }


def main(cmdline=True, **kwargs):
    r"""
    Execute the stac search and write the input file

    Example:
        >>> from watch.cli.stac_search import *  # NOQA
        >>> from watch.demo import demo_region
        >>> from watch.cli import stac_search_build
        >>> from watch.utils import util_gis
        >>> import ubelt as ub
        >>> dpath = ub.Path.appdir('watch/tests/test-stac-search').ensuredir()
        >>> search_fpath = dpath / 'stac_search.json'
        >>> region_fpath = demo_region.demo_khq_region_fpath()
        >>> region = util_gis.read_geojson(region_fpath)
        >>> result_fpath = dpath / 'demo.input'
        >>> start_date = region['start_date'].iloc[0]
        >>> end_date = region['end_date'].iloc[0]
        >>> stac_search_build.main(
        >>>     cmdline=0,
        >>>     start_date=start_date,
        >>>     end_date=end_date,
        >>>     cloud_cover=10,
        >>>     out_fpath=search_fpath,
        >>> )
        >>> kwargs = {
        >>>     'region_file': str(region_fpath),
        >>>     'search_json': str(search_fpath),
        >>>     'mode': 'area',
        >>>     'verbose': 2,
        >>>     'outfile': str(result_fpath),
        >>> }
        >>> result_fpath.delete()
        >>> cmdline = 0
        >>> main(cmdline=cmdline, **kwargs)
        >>> # results are in the
        >>> from watch.cli.baseline_framework_ingress import read_input_stac_items
        >>> items = read_input_stac_items(result_fpath)
        >>> len(items)
        >>> for item in items:
        >>>     print(item['properties']['eo:cloud_cover'])
        >>>     print(item['properties']['datetime'])
    """
    config = StacSearchConfig(cmdline=cmdline, data=kwargs)
    import ubelt as ub
    print('config = {}'.format(ub.repr2(dict(config), nl=1)))
    args = config.namespace

    logger = util_logging.get_logger(verbose=args.verbose)
    searcher = StacSearcher(logger)

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
                searcher.by_geometry(
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
                    searcher.by_id(
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
