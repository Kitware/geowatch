"""
A tool to help build a STAC search json.

This does contain a command line interface, for legacy reasons, but
is not intended to be a fully supported part of the WATCH CLI.

SeeAlso:
    ../cli/stac_search.py
"""
import os
import ubelt as ub
import scriptconfig as scfg


class StacSearchBuilderConfig(scfg.Config):
    """
    Helper to create STAC search json queries
    """
    default = {
        'start_date': scfg.Value(None, help='iso starting date'),
        'end_date': scfg.Value(None, help='iso ending date'),
        'cloud_cover': scfg.Value(10, help='maximum cloud cover percentage'),
        'out_fpath': scfg.Value(None, help='if unspecified, write to stdout'),
        'api_key': scfg.Value('env:SMART_STAC_API_KEY', help='The API key or where to get it'),
        'sensors': scfg.Value('L2-S2', help=''),
    }


_KITWARE_PHASE1_TA1_PRODUCTS = {
    # Kitware Phase 1 TA-1 Products
    'ta1-s2-kit': {
        'endpoint': "https://api.smart-stac.com",
        'collections': ['ta1-s2-kit'],
    },
    'ta1-l8-kit': {
        'endpoint': "https://api.smart-stac.com",
        'collections': ['ta1-l8-kit'],
    },
    'ta1-wv-kit': {
        'endpoint': "https://api.smart-stac.com",
        'collections': ['ta1-wv-kit'],
        "query": {
            "nitf:imd": {
                "eq": "true"
            },
        }
    },
}

"""
Accenture Notes:

    New Procesing 2022-11-21:
        https://smart-research.slack.com/?redir=%2Ffiles%2FU028UQGN1N0%2FF04B998ANRL%2Faccenture_ta1_productdoc_phaseii_20211117.pptx%3Forigin_team%3DTN3QR7WAH%26origin_channel%3DC03QTAXU7GF

        https://smartgitlab.com/TE/evaluations/-/wikis/Accenture-TA-1-Processing-Status

"""

_ACCENTURE_PHASE2_TA1_PRODUCTS = {
    # Accenture Phase 2 TA-1 Products
    'ta1-s2-acc': {
        'endpoint': "https://api.smart-stac.com",
        'collections': ['ta1-s2-acc'],
    },
    'ta1-ls-acc': {
        'endpoint': "https://api.smart-stac.com",
        'collections': ['ta1-ls-acc'],
    },
    'ta1-pd-acc': {
        'endpoint': "https://api.smart-stac.com",
        'collections': ['ta1-pd-acc'],
    },
    'ta1-wv-acc': {
        'endpoint': "https://api.smart-stac.com",
        'collections': ['ta1-wv-acc'],
        "query": {
            "nitf:imd": {
                "eq": "true"
            },
        }
    },

    # Accenture Phase 2 TA-1 Products
    'ta1-s2-acc-1': {
        'endpoint': "https://api.smart-stac.com",
        'collections': ['ta1-s2-acc-1'],
    },
    'ta1-ls-acc-1': {
        'endpoint': "https://api.smart-stac.com",
        'collections': ['ta1-ls-acc-1'],
    },
    'ta1-pd-acc-1': {
        'endpoint': "https://api.smart-stac.com",
        'collections': ['ta1-pd-acc-1'],
    },
    'ta1-wv-acc-1': {
        'endpoint': "https://api.smart-stac.com",
        'collections': ['ta1-wv-acc-1'],
        "query": {
            "nitf:imd": {
                "eq": "true"
            },
        }
    },

    'ta1-s2-acc-2': {
        'endpoint': "https://api.smart-stac.com",
        'collections': ['ta1-s2-acc-2'],
    },
    'ta1-ls-acc-2': {
        'endpoint': "https://api.smart-stac.com",
        'collections': ['ta1-ls-acc-2'],
    },
    'ta1-pd-acc-2': {
        'endpoint': "https://api.smart-stac.com",
        'collections': ['ta1-pd-acc-2'],
    },
    'ta1-wv-acc-2': {
        'endpoint': "https://api.smart-stac.com",
        'collections': ['ta1-wv-acc-2'],
        "query": {
            "nitf:imd": {
                "eq": "true"
            },
        }
    },

}


# ta1-mixedgsd-acc


_SMARTSTAC_PRODUCTS = {
    # Non public, but non-performer products
    'planet-dove': {
        'endpoint': "https://api.smart-stac.com",
        'collections': ['planet-dove'],
    },
    'worldview-nitf': {
        'endpoint': "https://api.smart-stac.com",
        'collections': ['worldview-nitf'],
        "query": {
            "nitf:imd": {
                "eq": "true"
            },
        }
    },

    'smart-landsat-c2l1': {
        "collections": ["landsat-c2l1"],
        'endpoint': "https://api.smart-stac.com",
        "query": {
            "platform": {
                "eq": "LANDSAT_8"
            }
        }
    },

    'smart-landsat-c2l2-sr': {
        "collections": ["landsat-c2l2-sr"],
        'endpoint': "https://api.smart-stac.com",
        "query": {
            "platform": {
                "eq": "LANDSAT_8"
            }
        }
    },

}


_PUBLIC_L1_PRODUCTS = {
    # https://landsatlook.usgs.gov/stac-server/
    # Public L1 Products
    'landsat-c2l1': {
        "collections": ["landsat-c2l1"],
        "endpoint": "https://landsatlook.usgs.gov/stac-server/",
        "query": {
            "platform": {
                "eq": "LANDSAT_8"
            }
        }
    },
    'sentinel-s2-l1c': {
        "collections": ["sentinel-s2-l1c"],
        "endpoint": "https://earth-search.aws.element84.com/v0",
    },
}


# NOTE;
# Info about QA bands:
# https://d9-wret.s3.us-west-2.amazonaws.com/assets/palladium/production/s3fs-public/media/files/LSDS-1435%20Landsat%20C2%20US%20ARD%20Data%20Format%20Control%20Book-v3.pdf
_PUBLIC_L2_PRODUCTS = {
    # Public L2 Products
    'sentinel-s2-l2a-cogs': {
        "collections": ["sentinel-s2-l2a-cogs"],
        "endpoint": "https://earth-search.aws.element84.com/v0",
    },

    # https://stacindex.org/catalogs/usgs-landsat-collection-2-api#/
    # Surface Reflectance
    'landsat-c2l2-sr': {
        # Note: AWS_REQUEST_PAYER='requester' is required to grab the data
        "collections": ["landsat-c2l2-sr"],
        "endpoint": "https://landsatlook.usgs.gov/stac-server/",
        "query": {
            "platform": {
                "eq": "LANDSAT_8"
            }
        }
    },

    # https://stacindex.org/catalogs/usgs-landsat-collection-2-api#/
    # Brightness Temperature
    'landsat-c2l2-bt': {
        # Note: AWS_REQUEST_PAYER='requester' is required to grab the data
        "collections": ["landsat-c2l2-bt"],
        "endpoint": "https://landsatlook.usgs.gov/stac-server/",
        "query": {
            "platform": {
                "eq": "LANDSAT_8"
            }
        }
    },
}


_PUBLIC_ARD_PRODUCTS = {
    # https://stacindex.org/catalogs/usgs-landsat-collection-2-api#/
    # Surface Reflectance
    'landsat-c2ard-sr': {
        # Note: AWS_REQUEST_PAYER='requester' is required to grab the data
        "collections": ["landsat-c2ard-sr"],
        "endpoint": "https://landsatlook.usgs.gov/stac-server/",
        "query": {
            "platform": {
                "eq": "LANDSAT_8"
            }
        }
    },

    # https://stacindex.org/catalogs/usgs-landsat-collection-2-api#/
    # Brightness Temperature
    'landsat-c2ard-bt': {
        # Note: AWS_REQUEST_PAYER='requester' is required to grab the data
        "collections": ["landsat-c2ard-bt"],
        "endpoint": "https://landsatlook.usgs.gov/stac-server/",
        "query": {
            "platform": {
                "eq": "LANDSAT_8"
            }
        }
    },
}


def print_provider_debug_information():
    """
    Helper to debug STAC endpoints and data availability.

    Summarize information about our hard-coded registered endpoints and query
    information about what other endpoints might exist.

    CommandLine:
        xdoctest -m watch.stac.stac_search_builder print_provider_debug_information
    """
    from rich import print
    print('Printing debug information about known and discoverable providers')
    rows = []
    for stac_code, stac_info in SENSOR_TO_DEFAULTS.items():

        endpoint = stac_info['endpoint']

        for collection in stac_info['collections']:
            rows.append({
                'stac_code': stac_code,
                'endpoint': endpoint,
                'collection': collection,
            })

    import pandas as pd
    df = pd.DataFrame(rows)
    print('Registered endpoints / collections / codes')
    print(df.to_string())
    unique_endpoints = df['endpoint'].unique()

    collection_to_endpoint = ub.udict(ub.group_items(df['endpoint'], df['collection']))
    endpoint_to_collections = ub.udict(ub.group_items(df['collection'], df['endpoint']))
    collection_to_num_endpoints = collection_to_endpoint.map_values(len)
    num_endpoints_to_collections = collection_to_num_endpoints.invert(unique_vals=False)
    multiendpoint_collections = num_endpoints_to_collections - {1}
    single_endpoint_collections = num_endpoints_to_collections.get(1, set())
    num_multi_collections = sum(map(len, multiendpoint_collections.values()))
    print(f'There are {len(unique_endpoints)} unique endpoints')
    print(f'There are {len(single_endpoint_collections)} collection that are unique to an endpoint')
    print(f'There are {num_multi_collections} collections that exist in multiple endpoints')
    print('single_endpoint_collections = {}'.format(ub.repr2(single_endpoint_collections, nl=1)))
    print('multiendpoint_collections = {}'.format(ub.repr2(multiendpoint_collections, nl=1)))
    print('unique_endpoints = {}'.format(ub.repr2(unique_endpoints, nl=1)))

    smart_stac_header = {
        'x-api-key': os.environ['SMART_STAC_API_KEY']
    }

    found_endpoint_to_catalog = {}
    found_endpoint_to_collections = {}

    import pystac_client
    for endpoint in unique_endpoints:
        print(f'Query {endpoint=}')

        pystac_headers = {}
        if endpoint == 'https://api.smart-stac.com':
            pystac_headers.update(smart_stac_header)

        try:
            catalog = pystac_client.Client.open(endpoint, headers=pystac_headers)
            collections = list(catalog.get_collections())
        except Exception:
            print(f'Failed to query {endpoint=}')
        else:
            print(f'Found {len(collections)} collections')
            found_endpoint_to_catalog[endpoint] = catalog
            found_endpoint_to_collections[endpoint] = collections

    unregistered_rows = []
    for endpoint, collections in found_endpoint_to_collections.items():
        print(f'\nCollections in endpoint={endpoint}')
        known_collection_names = set(endpoint_to_collections[endpoint])
        found_collection_names = {c.id for c in collections}
        registered_names = known_collection_names & found_collection_names
        misregistered_names = known_collection_names - found_collection_names
        unregistered_names = found_collection_names - known_collection_names
        if registered_names:
            print('Valid registered collections = {}'.format(ub.repr2(registered_names, nl=1)))
        if misregistered_names:
            print(f'!!! Collections are registered that dont exist {misregistered_names=}')
        if unregistered_names:
            print('There are unregistered collections = {}'.format(ub.repr2(unregistered_names, nl=1)))
            for name in unregistered_names:
                unregistered_rows.append({
                    'stac_code': None,
                    'endpoint': endpoint,
                    'collection': name,
                })
    print('Unregistered Collections')
    unregistered_df = pd.DataFrame(unregistered_rows)
    print(unregistered_df.to_string())

    full_df = pd.concat([df, unregistered_df])
    new_rows = full_df.to_dict(orient='records')
    for row in ub.ProgIter(new_rows, verbose=3):
        collection_name = row['collection']
        endpoint = row['endpoint']
        if endpoint in found_endpoint_to_collections:
            name_to_col = {c.id: c for c in found_endpoint_to_collections[endpoint]}
            if collection_name in name_to_col:
                catalog = found_endpoint_to_catalog[endpoint]
                collection = name_to_col[collection_name]
                row['title'] = collection.title

                is_unregistered = (unregistered_df[['endpoint', 'collection']] == [endpoint, collection_name]).all(axis=1).sum()
                is_registered = (df[['endpoint', 'collection']] == [endpoint, collection_name]).all(axis=1).sum()
                if is_registered:
                    assert not is_unregistered
                is_bad = not is_unregistered and not is_registered
                assert not is_bad
                row['registered'] = bool(is_registered)

                result = catalog.search(
                    collections=[collection_name],
                    max_items=1
                )
                found = list(result.items())
                row['has_items'] = len(found)

                print(row)
                print(collection.summaries.lists)

    for row in new_rows:
        if 'smart-stac' in row['endpoint']:
            collection_name = row['collection']
            if collection_name.startswith('ta1-'):
                parts = collection_name.split('-')
                if parts[-1] in {'1', '2', '3'}:
                    processing = '-'.join(parts[-2:])
                else:
                    processing = parts[-1]
                row['processing'] = processing

    new_df = pd.DataFrame(new_rows)
    new_df = new_df.sort_values(['processing', 'endpoint', 'collection'])
    # new_df['has_items'] = new_df['has_items'].fillna(False)
    new_df.loc[new_df['has_items'] == 1, 'has_items'] = True
    new_df.loc[new_df['has_items'] == 0, 'has_items'] = False
    new_df.loc[new_df['registered'] == 1, 'registered'] = True
    new_df.loc[new_df['registered'] == 0, 'registered'] = False
    print(new_df.to_string())


def check_processed_regions():
    """
    Print out a table of how many images / region / collection there are.

    CommandLine:
        xdoctest -m watch.stac.stac_search_builder check_processed_regions
    """
    import json
    import pystac_client
    from datetime import datetime as datetime_cls
    import watch

    dvc_data_dpath = watch.find_dvc_dpath(tags='phase2_data', hardware='auto')

    # MODIFY AS NEEDED
    headers = {
        'x-api-key': os.environ['SMART_STAC_API_KEY']
    }

    base = ((dvc_data_dpath / 'annotations') / 'drop6')
    # base = dvc_data_dpath / 'annotations'

    region_dpath = base / 'region_models'
    # region_fpaths = list(region_dpath.glob('*.geojson'))
    region_fpaths = list(region_dpath.glob('*_C*.geojson'))

    # region_dpath = dvc_data_dpath / 'golden_regions/region_models'
    # region_fpaths = list(region_dpath.glob('*_S*.geojson'))

    provider = "https://api.smart-stac.com"
    catalog = pystac_client.Client.open(provider, headers=headers)

    all_collections = list(catalog.get_collections())

    collections_of_interest = ['planet-dove', 'ta1-pd-acc', 'ta1-pd-ara', 'ta1-pd-str']

    from watch.utils import util_pattern
    pat = util_pattern.Pattern.coerce('ta1-*-acc*').to_regex()
    collections_of_interest = [c.id for c in all_collections if pat.match(c.id)]

    collections_of_interest = [
        # 'ta1-s2-acc',
        # 'ta1-s2-acc-1',
        'ta1-s2-acc-2',

        # 'ta1-ls-acc',
        # 'ta1-ls-acc-1',
        'ta1-ls-acc-2',

        # 'ta1-wv-acc',
        # 'ta1-wv-acc-1',
        # 'ta1-wv-acc-2',
    ]

    #     'ta1-pd-acc',
    #     'ta1-pd-acc-1',
    #     'ta1-mixedgsd-acc-1',
    #     'ta1-30m-acc-1'
    # ]
    rows = []

    import rich
    import rich.progress
    progress = rich.progress.Progress(
        "[progress.description]{task.description}",
        rich.progress.BarColumn(),
        rich.progress.MofNCompleteColumn(),
        # "[progress.percentage]{task.percentage:>3.0f}%",
        rich.progress.TimeRemainingColumn(),
        rich.progress.TimeElapsedColumn(),
    )
    with progress:
        collection_task = progress.add_task("[cyan] Query Collection...", total=len(collections_of_interest))
        region_task = None

        # Check that planet items exist
        for collection in collections_of_interest:

            if region_task is not None:
                progress.remove_task(region_task)
            progress.update(collection_task, advance=1)
            region_task = progress.add_task("[green] Query Regions...", total=len(region_fpaths))

            # Check that planet items exist in our regions
            region_to_results = {}
            for region_fpath in region_fpaths:

                progress.update(region_task, advance=1)

                with open(region_fpath) as file:
                    region_data = json.load(file)
                region_row = [f for f in region_data['features'] if f['properties']['type'] == 'region'][0]
                region_id = region_row['properties']['region_id']
                geom = region_row['geometry']
                start = region_row['properties']['start_date']
                end = region_row['properties']['end_date']
                if end is None:
                    # end = datetime_cls.utcnow().date()
                    end = datetime_cls.now().date().isoformat()

                item_search = catalog.search(
                    collections=[collection],
                    datetime=(start, end),
                    intersects=geom,
                    max_items=1000,
                )
                search_iter = item_search.items()
                # result0 = next(search_iter)
                results = list(search_iter)
                region_to_results[region_id] = results

                rows.append({
                    'collection': collection,
                    'region_id': region_id,
                    'num_results': len(results),
                    'start_date': start,
                    'end_date': end,
                })

    for row in rows:
        if row['collection'].endswith('acc-2'):
            row['processing'] = 'acc-2'
        elif row['collection'].endswith('acc-1'):
            row['processing'] = 'acc-1'
        elif row['collection'].endswith('acc'):
            row['processing'] = 'acc'

        if '-ls' in row['collection']:
            row['sensor'] = 'L8'
        elif '-s2' in row['collection']:
            row['sensor'] = 'S2'
        elif '-wv' in row['collection']:
            row['sensor'] = 'WV'
        elif '-pd' in row['collection']:
            row['sensor'] = 'PD'

    import pandas as pd
    from rich import print
    df = pd.DataFrame(rows)
    print(df.to_string())

    # df = df.sort_values(['processing', 'sensor', 'collection'])
    # piv = df.pivot(['region_id'], ['processing', 'sensor', 'collection'], ['num_results'])

    df = df.sort_values(['sensor', 'processing', 'collection'])
    piv = df.pivot(['region_id'], ['sensor', 'processing', 'collection'], ['num_results'])
    # piv = piv.astype(bool)
    print(piv.to_string())

    # print(df.to_string())
    # print(f'region_id={region_id}')
    # print(f'results={results}')
    # print(f'collection={collection}')
    # print('region_to_results = {}'.format(ub.repr2(region_to_results, nl=1)))


def _devcheck_providers_exist():
    """
    develoepr logic to test to see if providers are working

    """
    # from watch.stac.stac_search_builder import _ACCENTURE_PHASE2_TA1_PRODUCTS
    # provider = _ACCENTURE_PHASE2_TA1_PRODUCTS['ta1-pd-acc']['endpoint']
    import pystac_client
    import os
    import ubelt as ub

    headers = {
        'x-api-key': os.environ['SMART_STAC_API_KEY']
    }
    provider = "https://api.smart-stac.com"
    catalog = pystac_client.Client.open(provider, headers=headers)
    print(ub.repr2(list(catalog.get_collections())))

    # item_search = catalog.search(collections=["ta1-mixedgsd-acc-1"])
    # print(f'item_search={item_search}')

    item_search = catalog.search(collections=["ta1-s2-acc"])
    item_search = catalog.search(collections=["ta1-wv-acc"])
    if 1:
        item_iter = iter(item_search.items())
        # View cloud cover
        item = next(item_iter)
        ccs = []
        for item in item_iter:
            cc = item.to_dict()['properties']['eo:cloud_cover']
            ccs.append(cc)
        import kwplot
        sns = kwplot.autosns()
        import pandas as pd
        df = pd.DataFrame({'cc': ccs})
        sns.histplot(data=df, x='cc')

    item_search = catalog.search(collections=["ta1-pd-acc"])
    item_search = catalog.search(collections=["ta1-pd-ara"])
    item_search = catalog.search(collections=["ta1-pd-str"])
    print(f'item_search={item_search}')


### Available smartstac collections:
# <CollectionClient id=landsat-c2l2-sr>,
# <CollectionClient id=ta1-s2-ara>,
# <CollectionClient id=ta1-pd-ara>,
# <CollectionClient id=ssh-wv-acc>,
# <CollectionClient id=ssh-ls-acc>,
# <CollectionClient id=ta1-wv-ara>,
# <CollectionClient id=ta1-ls-ara>,
# <CollectionClient id=ta1-s2-acc>,
# <CollectionClient id=worldview-nitf>,
# <CollectionClient id=ta1-wv-str>,
# <CollectionClient id=landsat-c2l1>,
# <CollectionClient id=ta1-ls-str>,
# <CollectionClient id=ta1-pd-str>,
# <CollectionClient id=ta1-pd-kit>,
# <CollectionClient id=planet-dove>,
# <CollectionClient id=ta1-ls-kit>,
# <CollectionClient id=ta1-wv-acc>,
# <collectionclient id=ta1-dsm-ara>,
# <CollectionClient id=ssh-pd-acc>,
# <CollectionClient id=ssh-s2-acc>,
# <CollectionClient id=ta1-s2-kit>,
# <CollectionClient id=ta1-wv-kit>,
# <CollectionClient id=ta1-pd-acc>,
# <CollectionClient id=ta1-ls-acc>,
# <CollectionClient id=ta1-s2-str>,


SENSOR_TO_DEFAULTS = ub.dict_union(
    _KITWARE_PHASE1_TA1_PRODUCTS,
    _ACCENTURE_PHASE2_TA1_PRODUCTS,
    _PUBLIC_L1_PRODUCTS,
    _PUBLIC_L2_PRODUCTS,
    _PUBLIC_ARD_PRODUCTS,
    _SMARTSTAC_PRODUCTS,
)


# Simplified codes for the CLI
CONVINIENCE_SENSOR_GROUPS = {
    'TA1': [
        'ta1-s2-kit',
        'ta1-l8-kit',
        'ta1-wv-kit',
    ],
    'L2-S2': [
        'sentinel-s2-l2a-cogs',
    ],
    'L2-L8': [
        'landsat-c2l2-sr',
        'landsat-c2l2-bt',
    ],
    'L2-S2-L8': [
        'sentinel-s2-l2a-cogs',
        'landsat-c2l2-sr',
        'landsat-c2l2-bt',
    ],
    'ARD-L8': [
        'landsat-c2ard-sr',
        'landsat-c2ard-bt',
    ],
    'ARD-S2-L8': [
        'sentinel-s2-l2a-cogs',
        'landsat-c2ard-sr',
        'landsat-c2ard-bt',
    ],

    'TA1-S2-L8-ACC': [
        'ta1-s2-acc',
        'ta1-ls-acc',
    ],
    'TA1-S2-L8-WV-PD-ACC-2': [
        'ta1-s2-acc-2',
        'ta1-ls-acc-2',
        'ta1-pd-acc-2',
        'ta1-wv-acc-2',
    ],
    'TA1-S2-L8-WV-PD-ACC-1': [
        'ta1-s2-acc-1',
        'ta1-ls-acc-1',
        'ta1-pd-acc-1',
        'ta1-wv-acc-1',
    ],
    'TA1-S2-L8-WV-PD-ACC': [
        'ta1-s2-acc',
        'ta1-ls-acc',
        'ta1-pd-acc',
        'ta1-wv-acc',
    ],
    'TA1-S2-WV-PD-ACC': [
        'ta1-s2-acc',
        'ta1-pd-acc',
        'ta1-wv-acc',
    ],
    'TA1-WV-PD-ACC': [
        'ta1-pd-acc',
        'ta1-wv-acc',
    ],
    'TA1-S2-ACC': [
        'ta1-s2-acc',
    ],
    'TA1-L8-ACC': [
        'ta1-ls-acc',
    ],
}


def build_search_json(start_date, end_date, sensors, api_key, cloud_cover):
    """
    Construct the json that can be used for a stac search

    Example:
        >>> from watch.stac.stac_search_builder import build_search_json
        >>> start_date = '2017-01-01'
        >>> end_date = '2020-01-01'
        >>> sensors = 'L2-S2'
        >>> api_key = None
        >>> cloud_cover = 20
        >>> search_json = build_search_json(start_date, end_date, sensors, api_key, cloud_cover)
        >>> print('search_json = {}'.format(ub.repr2(search_json, nl=-1)))
        search_json = {
            'stac_search': [
                {
                    'collections': ['sentinel-s2-l2a-cogs'],
                    'end_date': '2020-01-01',
                    'endpoint': 'https://earth-search.aws.element84.com/v0',
                    'headers': {},
                    'query': {
                        'eo:cloud_cover': {'lt': 20}
                    },
                    'start_date': '2017-01-01'
                }
            ]
        }
    """
    from watch.utils import util_time

    if api_key is not None and api_key.startswith('env:'):
        api_environ_key = api_key.split(':')[1]
        api_key = os.environ.get(api_environ_key, None)

    if isinstance(sensors, str):
        if sensors in SENSOR_TO_DEFAULTS:
            sensors = [sensors]
        else:
            sensors = CONVINIENCE_SENSOR_GROUPS[sensors]

    headers = {}
    if api_key is not None:
        headers['x-api-key'] = api_key

    start_date = util_time.coerce_datetime(start_date, default_timezone='utc')
    end_date = util_time.coerce_datetime(end_date, default_timezone='utc')

    if start_date is None or end_date is None:
        raise ValueError('need start and end date')

    search_item_list = []
    for sensor in sensors:
        search_item = SENSOR_TO_DEFAULTS[sensor]
        item_query = search_item.setdefault('query', {})
        search_item['headers'] = headers
        search_item['start_date'] = start_date.date().isoformat()
        search_item['end_date'] = end_date.date().isoformat()
        if cloud_cover is not None:
            item_query['eo:cloud_cover'] = {
                'lt': cloud_cover,
            }
        search_item_list.append(search_item)

    search_json = {
        'stac_search': search_item_list,
    }
    return search_json


def main(cmdline=1, **kwargs):
    """
    Example:
        >>> # xdoctest: +SKIP
        >>> from watch.stac.stac_search_builder import main
        >>> cmdline = 0
        >>> kwargs = {
        >>>     'start_date': '2017-01-01',
        >>>     'end_date': '2020-01-01',
        >>>     'sensors': 'L2-S2',
        >>> }
        >>> main(cmdline=cmdline, **kwargs)
    """
    import json
    config = StacSearchBuilderConfig(cmdline=cmdline, data=kwargs)

    search_json = build_search_json(**ub.compatible(config, build_search_json))
    text = json.dumps(search_json, indent='    ')

    if config['out_fpath'] is not None:
        out_fpath = ub.Path(config['out_fpath'])
        out_fpath.write_text(text)
    else:
        print(text)


if __name__ == '__main__':
    """
    CommandLine:
        python ~/code/watch/watch/stac/stac_search_build.py
    """
    main()
