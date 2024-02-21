#!/usr/bin/env python3
r"""
Performs the stac search to create the .input file needed for
:mod:`geowatch.cli.prepare_ta2_dataset`.


SeeAlso:
    ../demo/demo_region.py
    ../stac/stac_search_builder.py

CommandLine:
    # Create a demo region file
    xdoctest geowatch.demo.demo_region demo_khq_region_fpath

    DATASET_SUFFIX=DemoKHQ-2022-06-10-V2
    DEMO_DPATH=$HOME/.cache/geowatch/demo/datasets

    REGION_FPATH="$HOME/.cache/geowatch/demo/annotations/KHQ_R001.geojson"
    SITE_GLOBSTR="$HOME/.cache/geowatch/demo/annotations/KHQ_R001_sites/*.geojson"

    START_DATE=$(jq -r '.features[] | select(.properties.type=="region") | .properties.start_date' "$REGION_FPATH")
    END_DATE=$(jq -r '.features[] | select(.properties.type=="region") | .properties.end_date' "$REGION_FPATH")
    REGION_ID=$(jq -r '.features[] | select(.properties.type=="region") | .properties.region_id' "$REGION_FPATH")
    SEARCH_FPATH=$DEMO_DPATH/stac_search.json
    RESULT_FPATH=$DEMO_DPATH/all_sensors_kit/${REGION_ID}.input

    mkdir -p "$DEMO_DPATH"

    # Create the search json wrt the sensors and processing level we want
    python -m geowatch.stac.stac_search_builder \
        --start_date="$START_DATE" \
        --end_date="$END_DATE" \
        --cloud_cover=40 \
        --sensors=sentinel-2-l2a \
        --out_fpath "$SEARCH_FPATH"
    cat "$SEARCH_FPATH"

    # Delete this to prevent duplicates
    rm -f "$RESULT_FPATH"
    # Create the .input file
    python -m geowatch.cli.stac_search \
        -rf "$REGION_FPATH" \
        -sj "$SEARCH_FPATH" \
        -m area \
        --verbose 2 \
        -o "${RESULT_FPATH}"

    # Construct the TA2-ready dataset
    python -m geowatch.cli.prepare_ta2_dataset \
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
        --backend=serial --run=0


CommandLine:
    # Alternate invocation
    # Create a demo region file
    xdoctest geowatch.demo.demo_region demo_khq_region_fpath

    DATASET_SUFFIX=DemoKHQ-2022-06-10-V3
    DEMO_DPATH=$HOME/.cache/geowatch/demo/datasets
    REGION_FPATH="$HOME/.cache/geowatch/demo/annotations/KHQ_R001.geojson"
    REGION_ID=$(jq -r '.features[] | select(.properties.type=="region") | .properties.region_id' "$REGION_FPATH")
    RESULT_FPATH=$DEMO_DPATH/all_sensors_kit/${REGION_ID}.input

    mkdir -p "$DEMO_DPATH"

    # Define SMART_STAC_API_KEY
    source "$HOME"/code/watch/secrets/secrets

    # Delete this to prevent duplicates
    rm -f "$RESULT_FPATH"
    # Create the .input file
    python -m geowatch.cli.stac_search \
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

    DVC_DPATH=$(geowatch_dvc --tags="phase2_data" --hardware=auto)
    REGION_FPATH=$DVC_DPATH/annotations/region_models/BR_R005.geojson

    # Define SMART_STAC_API_KEY
    source "$HOME"/code/watch/secrets/secrets

    # Delete this to prevent duplicates
    rm -f "$RESULT_FPATH"
    # Create the .input file
    python -m geowatch.cli.stac_search \
        --region_file "$REGION_FPATH" \
        --api_key=env:SMART_STAC_API_KEY \
        --search_json "auto" \
        --cloud_cover 100 \
        --sensors=TA1-S2-L8-WV-PD-ACC \
        --mode area \
        --verbose 2 \
        --outfile "./result.input"

    ###
    ### - Debug case

    DVC_DPATH=$(geowatch_dvc --tags="phase2_data" --hardware=auto)

    # Load SMART_STAC_API_KEY
    source "$HOME"/code/watch/secrets/secrets

    python -m geowatch.cli.stac_search \
        --region_file "$DVC_DPATH/annotations/region_models/US_R007.geojson" \
        --search_json "auto" \
        --cloud_cover "0" \
        --sensors "TA1-S2-L8-WV-PD-ACC-1" \
        --api_key "env:SMART_STAC_API_KEY" \
        --max_products_per_region "None" \
        --append_mode=False \
        --mode area \
        --verbose 100000 \
        --outfile "./test_result.input"
"""
import json
import tempfile
import pystac_client
from geowatch.utils import util_logging
from geowatch.utils import util_s3
import kwarray
import ubelt as ub
import scriptconfig as scfg


class StacSearchConfig(scfg.DataConfig):
    """
    Execute a STAC query
    """
    __default__ = {
        'outfile': scfg.Value(
            None,
            help='output file name for STAC items',
            short_alias=['o'],
            required=True
        ),

        'region_globstr': scfg.Value(None, help='if specified, run over multiple region files and ignore "region_file" and "site_file"'),

        'max_products_per_region': scfg.Value(None, help='does uniform affinity sampling over time to filter down to this many results per region'),

        'append_mode': scfg.Value(True, help='if True appends to the existing output file. If false will overwrite an existing output file'),

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

        'allow_failure': scfg.Value(False, isflag=True, help='if True keeps running if one region fails'),

        'query_workers': scfg.Value(0, help='The number of queries to run in parallel'),

        'cloud_cover': scfg.Value(10, help='maximum cloud cover percentage (only used if search_json is "auto")'),
        'sensors': scfg.Value("L2", help='(only used if search_json is "auto")'),
        'api_key': scfg.Value('env:SMART_STAC_API_KEY', help='The API key or where to get it (only used if search_json is "auto")'),
    }


def main(cmdline=True, **kwargs):
    r"""
    Execute the stac search and write the input file

    Example:
        >>> # xdoctest: +REQUIRES(env:SLOW_DOCTEST)
        >>> # xdoctest: +REQUIRES(--network)
        >>> from geowatch.cli.stac_search import *  # NOQA
        >>> from geowatch.demo import demo_region
        >>> from geowatch.stac import stac_search_builder
        >>> from geowatch.utils import util_gis
        >>> import ubelt as ub
        >>> dpath = ub.Path.appdir('geowatch/tests/test-stac-search').ensuredir()
        >>> search_fpath = dpath / 'stac_search.json'
        >>> region_fpath = demo_region.demo_khq_region_fpath()
        >>> region = util_gis.load_geojson(region_fpath)
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
        >>> from geowatch.cli.baseline_framework_ingress import read_input_stac_items
        >>> items = read_input_stac_items(result_fpath)
        >>> len(items)
        >>> for item in items:
        >>>     print(item['properties']['eo:cloud_cover'])
        >>>     print(item['properties']['datetime'])
    """
    config = StacSearchConfig.cli(cmdline=cmdline, data=kwargs, strict=True)
    import rich
    from geowatch.utils import util_gis
    from kwutil import slugify_ext
    from kwutil import util_progress
    from kwutil import util_parallel
    from geowatch.utils import util_pandas
    from kwutil import util_time
    import pandas as pd
    import rich.markup
    rich.print('config = {}'.format(ub.urepr(config, nl=1)))
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

    if config['append_mode']:
        # Ensure we are not appending to an existing file
        dest_path.delete()

    if args.mode == 'area':
        if config['region_globstr'] is not None:
            region_file_fpaths = util_gis.coerce_geojson_paths(config['region_globstr'])
            assert args.mode == 'area'
        else:
            if not hasattr(args, 'region_file'):
                raise ValueError('Missing region file')
            region_file_fpaths = [args.region_file]

        print('region_file_fpaths = {}'.format(slugify_ext.smart_truncate(
            ub.urepr(region_file_fpaths, nl=1), max_length=1000)))

        if not hasattr(args, 'search_json'):
            raise ValueError('Missing stac search parameters')
        search_json = args.search_json

        overall_status = {
            'regions_with_results': 0,
            'regions_without_results': 0,
            'regions_with_errors': 0,
            'total_regions': len(region_file_fpaths),
        }

        query_workers = util_parallel.coerce_num_workers(config['query_workers'])
        print(f'query_workers={query_workers}')

        result_summary_rows = []

        def aggregate_datetime(values):
            dts = [util_time.coerce_datetime(v) for v in values]
            min_dt = min(dts)
            max_dt = max(dts)
            min_str = min_dt.date().isoformat()
            max_str = max_dt.date().isoformat()
            return f'{min_str} - {max_str}'

        property_aggregators = {
            'platform': 'unique',
            'constellation': 'unique',
            'mission': 'hist',
            # 'gsd': 'min-max',
            # 'eo:cloudcover': 'min-max',
            'datetime': aggregate_datetime,
            # 'quality_info:filled_percentage': 'min-max',
            # 'proj:shape': 'drop',
            # 'proj:epsg': 'hist'
        }

        if len(region_file_fpaths) == 1:
            # Force serial if there is just one
            query_workers = 0

        pool = ub.JobPool(mode='thread', max_workers=query_workers)
        pman = util_progress.ProgressManager(backend='rich' if query_workers > 0 else 'progiter')
        with pman:
            for region_fpath in pman.progiter(region_file_fpaths, desc='submit query jobs'):
                job = pool.submit(area_query, region_fpath, search_json,
                                  searcher, temp_dir, config, logger,
                                  verbose=(query_workers == 0))
                job.region_name = ub.Path(region_fpath).name.split('.')[0]

            with open(dest_path, 'a') as the_file:
                for job in pman.progiter(pool.as_completed(),
                                         total=len(region_file_fpaths),
                                         desc='collect query jobs', verbose=3):
                    try:
                        area_results = job.result()
                    except Exception:
                        overall_status['regions_with_errors'] += 1
                        area_results = []
                        if not args.allow_failure:
                            raise

                    had_results = False
                    for result in area_results:
                        querykw = result['querykw']
                        features = result['features']
                        if len(features):
                            had_results = True
                        for item in features:
                            the_file.write(json.dumps(item) + '\n')

                        proprows = [
                            ub.udict(f['properties']) & property_aggregators.keys()
                            for f in result['features']
                        ]
                        propdf = pd.DataFrame(proprows)
                        aggrow = util_pandas.aggregate_columns(
                            propdf, property_aggregators, nonconst_policy='drop')
                        row = {
                            'q_region': job.region_name,
                            'q_time': f'{querykw["start"]} - {querykw["end"]}',
                            'q_collections': ','.join(querykw['collections']),
                            'num_results': len(features),
                            **aggrow,
                        }
                        result_summary_rows.append(row)

                    if had_results:
                        overall_status['regions_with_results'] += 1
                    else:
                        overall_status['regions_without_results'] += 1

                    summary_df = pd.DataFrame(result_summary_rows)
                    rich.print(rich.markup.escape(summary_df.to_string()))
                    pman.update_info('overall_status = {}'.format(ub.urepr(overall_status, nl=2)))

        summary_df = pd.DataFrame(result_summary_rows)
        rich.print(rich.markup.escape(summary_df.to_string()))
    else:
        raise NotImplementedError(f'only area is implemented. Got {args.mode=}')

    if args.s3_dest is not None:
        logger.info('Saving output to S3')
        util_s3.send_file_to_s3(dest_path, args.s3_dest)
    else:
        logger.info('--s3_dest parameter not present; skipping S3 output')

    logger.info('Search complete')


class StacSearcher:
    r"""
    Example:
        >>> # xdoctest: +REQUIRES(env:WATCH_ENABLE_NETWORK_TESTS)
        >>> # xdoctest: +REQUIRES(env:SLOW_DOCTEST)
        >>> from geowatch.cli.stac_search import *  # NOQA
        >>> import tempfile
        >>> provider = "https://earth-search.aws.element84.com/v0"
        >>> geom = {'type': 'Polygon',
        >>>         'coordinates': [[[54.960669, 24.782276],
        >>>                          [54.960669, 25.03516],
        >>>                          [55.268326, 25.03516],
        >>>                          [55.268326, 24.782276],
        >>>                          [54.960669, 24.782276]]]}
        >>> collections = ["sentinel-2-l2a"]
        >>> start = '2020-03-14'
        >>> end = '2020-05-04'
        >>> query = {}
        >>> headers = {}
        >>> self = StacSearcher()
        >>> features = self.by_geometry(provider, geom, collections, start_date, end_date,
        >>>                  query, headers)
        >>> print('features = {}'.format(ub.urepr(features, nl=1)))
    """

    def __init__(self, logger=None):
        if logger is None:
            logger = util_logging.PrintLogger(verbose=1)
        self.logger = logger

    def by_geometry(self, provider, geom, collections, start, end,
                    query, headers, max_products_per_region=None, verbose=1):
        if verbose:
            self.logger.info('Processing ' + provider)
        # hack: is there a better way to declare (i.e. in the
        # stac_search_builder) that the sign inplace is required to access the
        # planetary computer data?
        client_kwargs = {}
        if 'planetarycomputer' in provider:
            import planetary_computer as pc
            client_kwargs['modifier'] = pc.sign_inplace

        catalog = pystac_client.Client.open(provider, headers=headers,
                                            **client_kwargs)

        daterange = [start, end]

        if verbose:
            self.logger.info(f'Query {collections} between {start} and {end}')
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
            print('ERROR ex = {}'.format(ub.urepr(ex, nl=1)))
            if 'no such index' in str(ex):
                print('You may have the wrong collection. Listing available')
                available_collections = list(catalog.get_all_collections())
                print('available_collections = {}'.format(ub.urepr(available_collections, nl=1)))
                pass
            raise

        features = [d.to_dict() for d in items]

        dates_found = [item.datetime for item in items]
        if verbose:
            if dates_found:
                min_date_found = min(dates_found).date().isoformat()
                max_date_found = max(dates_found).date().isoformat()
                self.logger.info(f'Search found {len(items)} items for {collections} between {min_date_found} and {max_date_found}')
            else:
                self.logger.warning(f'Search found {len(items)} items for {collections}')

        if max_products_per_region and max_products_per_region < len(features):
            # Filter to a max number of items per region for testing
            # Sample over time uniformly
            from kwutil import util_time
            from geowatch.tasks.fusion.datamodules import temporal_sampling
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
            if verbose:
                self.logger.info(f'Filtered to {len(features)} items')
        return features

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


def _auto_search_params_from_region(r_file_loc, config):
    from geowatch.utils import util_gis
    from kwutil import util_time
    from geowatch.stac.stac_search_builder import build_search_json
    region_df = util_gis.load_geojson(r_file_loc)
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


def area_query(region_fpath, search_json, searcher, temp_dir, config, logger, verbose=1):
    from geowatch.geoannots import geomodels
    from shapely.geometry import shape as geom_shape
    if verbose:
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
        except (json.decoder.JSONDecodeError, TypeError):
            with open(search_json, 'r') as f:
                search_params = json.load(f)

    if verbose:
        logger.info(f'Query with params: {ub.urepr(search_params)}')

    region = geomodels.RegionModel.coerce(r_file_loc)
    # region.validate(strict=False)
    # region._()

    geom = geom_shape(region.header['geometry'])

    searches = search_params['stac_search']
    if verbose:
        logger.info(f'Performing {len(searches)} geometry stac searches')

    area_results = []
    for s in search_params['stac_search']:
        querykw = dict(
            provider=s['endpoint'],
            geom=geom,
            collections=s['collections'],
            start=s['start_date'],
            end=s['end_date'],
            query=s.get('query', {}),
            headers=s.get('headers', {}),
            max_products_per_region=config['max_products_per_region'],
        )
        features = searcher.by_geometry(verbose=verbose, **querykw)
        result = {
            'querykw': querykw,
            'features': features,
        }
        area_results.append(result)

    if verbose:
        total_results = sum(len(r['features']) for r in area_results)
        if total_results:
            logger.info(f'Total results for region: {total_results}')
        else:
            logger.warning(f'Total results for region: {total_results}')

    return area_results


if __name__ == '__main__':
    main()
