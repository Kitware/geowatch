#!/usr/bin/env python
r"""
Performs the stac search to create the .input file needed for
prepare_ta2_dataset


SeeAlso:
    ../demo/demo_region.py
    ../stac/stac_search_builder.py

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
    python -m watch.stac.stac_search_builder \
        --start_date="$START_DATE" \
        --end_date="$END_DATE" \
        --cloud_cover=40 \
        --sensors=sentinel-s2-l2a-cogs \
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


CommandLine:
    # Alternate invocation
    # Create a demo region file
    xdoctest watch.demo.demo_region demo_khq_region_fpath

    DATASET_SUFFIX=DemoKHQ-2022-06-10-V3
    DEMO_DPATH=$HOME/.cache/watch/demo/datasets
    REGION_FPATH="$HOME/.cache/watch/demo/annotations/KHQ_R001.geojson"
    REGION_ID=$(jq -r '.features[] | select(.properties.type=="region") | .properties.region_id' "$REGION_FPATH")
    RESULT_FPATH=$DEMO_DPATH/all_sensors_kit/${REGION_ID}.input

    mkdir -p "$DEMO_DPATH"

    # Define SMART_STAC_API_KEY
    source "$HOME"/code/watch/secrets/secrets

    # Delete this to prevent duplicates
    rm -f "$RESULT_FPATH"
    # Create the .input file
    python -m watch.cli.stac_search \
        --region_file "$REGION_FPATH" \
        --api_key=env:SMART_STAC_API_KEY \
        --search_json "auto" \
        --cloud_cover 10 \
        --sensors=TA1-L8-ACC \
        --mode area \
        --verbose 2 \
        --outfile "${RESULT_FPATH}"

CommandLine:
    # Alternate invocation
    # Create a demo region file

    DVC_DPATH=$(smartwatch_dvc)
    REGION_FPATH=$DVC_DPATH/annotations/region_models/BR_R005.geojson

    # Define SMART_STAC_API_KEY
    source "$HOME"/code/watch/secrets/secrets

    # Delete this to prevent duplicates
    rm -f "$RESULT_FPATH"
    # Create the .input file
    python -m watch.cli.stac_search \
        --region_file "$REGION_FPATH" \
        --api_key=env:SMART_STAC_API_KEY \
        --search_json "auto" \
        --cloud_cover 100 \
        --sensors=TA1-S2-L8-WV-PD-ACC \
        --mode area \
        --verbose 2 \
        --outfile "./result.input"
"""
import json
import tempfile
import pystac_client
from shapely.geometry import shape as geom_shape
from watch.utils import util_logging
from watch.utils import util_s3
import ubelt as ub
import scriptconfig as scfg


class StacSearchConfig(scfg.Config):
    """
    Execute a STAC query
    """
    default = {
        'outfile': scfg.Value(
            None,
            help='output file name for STAC items',
            short_alias=['o'],
            required=True
        ),

        'region_globstr': scfg.Value(None, help='if specified, run over multiple region files and ignore "region_file" and "site_file"'),

        'max_products_per_region': scfg.Value(None, help='does uniform affinity sampling over time to filter down to this many results per region'),

        'region_file': scfg.Value(
            None,
            help='path to a region geojson file; required if mode is area',
            short_alias=['rf']
        ),
        'search_json': scfg.Value(
            None, help=ub.paragraph(
                '''
                json string or path to json file containing STAC search
                parameters. If "auto", then parameters are inferred from
                '''),
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

        'cloud_cover': scfg.Value(10, help='maximum cloud cover percentage (only used if search_json is "auto")'),
        'sensors': scfg.Value("L2", help='(only used if search_json is "auto")'),
        'api_key': scfg.Value('env:SMART_STAC_API_KEY', help='The API key or where to get it (only used if search_json is "auto")'),
    }


class StacSearcher:
    r"""
    Example:
        >>> # xdoctest: +REQUIRES(env:WATCH_ENABLE_NETWORK_TESTS)
        >>> from watch.cli.stac_search import *  # NOQA
        >>> import tempfile
        >>> provider = "https://earth-search.aws.element84.com/v0"
        >>> geom = {'type': 'Polygon',
        >>>         'coordinates': [[[54.960669, 24.782276],
        >>>                          [54.960669, 25.03516],
        >>>                          [55.268326, 25.03516],
        >>>                          [55.268326, 24.782276],
        >>>                          [54.960669, 24.782276]]]}
        >>> collections = ["sentinel-s2-l2a-cogs"]
        >>> start = '2020-03-14'
        >>> end = '2020-05-04'
        >>> query = {}
        >>> headers = {}
        >>> outfile = ub.Path(tempfile.mktemp())
        >>> self = StacSearcher()
        >>> self.by_geometry(provider, geom, collections, start_date, end_date,
        >>>                  outfile, query, headers)
        >>> results_text = outfile.read_text()
        >>> for result in [r for r in results_text.split('\n') if r]:
        >>>     item = json.loads(result)
        >>>     print('item = {}'.format(ub.repr2(item, nl=-1)))
    """

    def __init__(self, logger=None):
        if logger is None:
            logger = util_logging.PrintLogger(verbose=1)
        self.logger = logger

    def by_geometry(self, provider, geom, collections, start, end, outfile,
                    query, headers, max_products_per_region=None):
        self.logger.info('Processing ' + provider)
        catalog = pystac_client.Client.open(provider, headers=headers)

        daterange = [start, end]

        search = catalog.search(
            collections=collections,
            datetime=daterange,
            intersects=geom,
            max_items=None,
            query=query)

        # Found features
        try:
            items_gen = search.items()
            items = list(items_gen)
            # items = search.get_all_items()
        except pystac_client.exceptions.APIError as ex:
            print('ERROR ex = {}'.format(ub.repr2(ex, nl=1)))
            if 'no such index' in str(ex):
                print('You may have the wrong collection. Listing available')
                available_collections = list(catalog.get_all_collections())
                print('available_collections = {}'.format(ub.repr2(available_collections, nl=1)))
                pass
            raise

        features = [d.to_dict() for d in items]

        dates_found = [item.datetime for item in items]
        min_date_found = min(dates_found).date().isoformat()
        max_date_found = min(dates_found).date().isoformat()
        self.logger.info(f'Search found {len(items)} items for {collections} between {min_date_found} and {max_date_found}')

        if max_products_per_region and max_products_per_region < len(features):
            # Filter to a max number of items per region for testing
            # Sample over time uniformly
            from watch.utils import util_time
            from watch.tasks.fusion.datamodules import temporal_sampling
            import kwarray
            import ubelt as ub
            datetimes = [util_time.coerce_datetime(item['properties']['datetime'])
                         for item in features]
            # TODO: Can we get a linear variant that doesn't need the N**2
            # affinity matrix?  Greedy set cover maybe? Or mean-shift
            sampler = temporal_sampling.TimeWindowSampler.from_datetimes(
                datetimes, time_span='full', time_window=max_products_per_region,
                affinity_type='soft2', update_rule='pairwise+distribute',
                affkw={'heuristics': []},
            )
            rng = kwarray.ensure_rng(sampler.unixtimes.sum())
            take_idxs = sampler.sample(rng=rng)

            features = list(ub.take(features, take_idxs))
            self.logger.info(f'Filtered to {len(features)} items')

        with open(outfile, 'a') as the_file:
            for item in features:
                the_file.write(json.dumps(item) + '\n')
        self.logger.info(f'Saved STAC results to: {outfile}')

    def by_id(self, provider, collections, stac_id, outfile, query, headers):
        raise NotImplementedError
        self.logger.info(f'Processing {stac_id}')
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
        self.logger.info(f'Saved STAC result to: {outfile}')


def main(cmdline=True, **kwargs):
    r"""
    Execute the stac search and write the input file

    Example:
        >>> from watch.cli.stac_search import *  # NOQA
        >>> from watch.demo import demo_region
        >>> from watch.stac import stac_search_builder
        >>> from watch.utils import util_gis
        >>> import ubelt as ub
        >>> dpath = ub.Path.appdir('watch/tests/test-stac-search').ensuredir()
        >>> search_fpath = dpath / 'stac_search.json'
        >>> region_fpath = demo_region.demo_khq_region_fpath()
        >>> region = util_gis.read_geojson(region_fpath)
        >>> result_fpath = dpath / 'demo.input'
        >>> start_date = region['start_date'].iloc[0]
        >>> end_date = region['end_date'].iloc[0]
        >>> stac_search_builder.main(
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
    print('config = {}'.format(ub.repr2(dict(config), nl=1)))
    args = config.namespace

    logger = util_logging.get_logger(verbose=args.verbose)
    searcher = StacSearcher(logger)

    dest_path = ub.Path(args.outfile)
    outdir = dest_path.parent

    temp_dir = ub.Path(tempfile.mkdtemp(prefix='stac_search'))
    logger.info(f'Created temp folder: {temp_dir}')
    if str(outdir) == '':
        dest_path = temp_dir / args.outfile
    else:
        outdir.ensuredir()

    ub.Path(dest_path).parent.ensuredir()

    if args.mode == 'area':
        if config['region_globstr'] is not None:
            from watch.utils import util_path
            region_file_fpaths = util_path.coerce_patterned_paths(config['region_globstr'])
            assert args.mode == 'area'
        else:
            if not hasattr(args, 'region_file'):
                raise ValueError('Missing region file')
            region_file_fpaths = [args.region_file]
        print('region_file_fpaths = {}'.format(ub.repr2(region_file_fpaths, nl=1)))

        if not hasattr(args, 'search_json'):
            raise ValueError('Missing stac search parameters')
        search_json = args.search_json

        # Might be reasonable to parallize this, but will need locks around
        # writes to the same file, or write to separate files and then combine
        for region_fpath in region_file_fpaths:
            area_query(region_fpath, search_json, searcher, temp_dir, dest_path, config, logger)
    else:
        id_query(searcher, logger, dest_path, temp_dir, args)

    if args.s3_dest is not None:
        logger.info('Saving output to S3')
        util_s3.send_file_to_s3(dest_path, args.s3_dest)
    else:
        logger.info('--s3_dest parameter not present; skipping S3 output')

    logger.info('Search complete')


def _auto_search_params_from_region(r_file_loc, config):
    from watch.utils import util_gis
    from watch.utils import util_time
    from watch.stac.stac_search_builder import build_search_json
    region_df = util_gis.read_geojson(r_file_loc)
    region_row = region_df[region_df['type'] == 'region'].iloc[0]
    end_date = util_time.coerce_datetime(region_row['end_date'])
    start_date = util_time.coerce_datetime(region_row['start_date'])
    if end_date is None:
        end_date = util_time.coerce_datetime('now').date()
    if start_date is None:
        start_date = util_time.coerce_datetime('2010-01-01').date()
    # Hack to avoid pre-constructing the search json
    cloud_cover = config['cloud_cover']  # TODO parametarize this
    sensors = config['sensors']
    api_key = config['api_key']
    search_params = build_search_json(
        start_date=start_date, end_date=end_date,
        sensors=sensors, api_key=api_key,
        cloud_cover=cloud_cover)
    return search_params


def area_query(region_fpath, search_json, searcher, temp_dir, dest_path, config, logger):
    logger.info(f'Query region file: {region_fpath}')

    if str(region_fpath).startswith('s3://'):
        r_file_loc = util_s3.get_file_from_s3(region_fpath, temp_dir)
    else:
        r_file_loc = region_fpath

    if search_json == 'auto':
        # hack to construct the search params here.
        search_params = _auto_search_params_from_region(r_file_loc, config)
    else:
        # Assume it is a path
        try:
            search_params = json.loads(search_json)
        except json.decoder.JSONDecodeError:
            with open(search_json) as f:
                search_params = json.load(f)

    with open(r_file_loc, 'r') as r_file:
        region = json.loads(r_file.read())

    regions = [
        f for f in region['features'] if (
            f['properties']['type'].lower() == 'region')
    ]
    if len(regions) != 1:
        raise AssertionError(
            f'Region file {r_file_loc!r} should have exactly 1 feature with '
            f'type "region", but we found {len(regions)}')

    max_products_per_region = config['max_products_per_region']
    # assume only 1 region per region model file
    geom = geom_shape(regions[0]['geometry'])

    searches = search_params['stac_search']
    logger.info(f'Performing {len(searches)} geometry stac searches')

    for s in search_params['stac_search']:
        searcher.by_geometry(
            s['endpoint'],
            geom,
            s['collections'],
            s['start_date'],
            s['end_date'],
            dest_path,
            s.get('query', {}),
            s.get('headers', {}),
            max_products_per_region=max_products_per_region
        )


def id_query(searcher, logger, dest_path, temp_dir, args):
    # FIXME
    raise NotImplementedError('This doesnt have the right stac endpoints setup for it.')
    # DEPRECATE FOR ITEMS IN STAC_BUILDER (which maybe moves somewhere else?)
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
    if args.site_file.startswith('s3://'):
        s_file_loc = util_s3.get_file_from_s3(args.site_file, temp_dir)
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


if __name__ == '__main__':
    main()
