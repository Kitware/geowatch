"""
References:
    https://smartgitlab.com/TE/evaluations/-/wikis/Accenture-TA-1-Processing-Status
"""
import os
import ubelt as ub
from watch.stac.stac_search_builder import SENSOR_TO_DEFAULTS


def print_provider_debug_information():
    """
    Helper to debug STAC endpoints and data availability.

    Summarize information about our hard-coded registered endpoints and query
    information about what other endpoints might exist.

    CommandLine:
        source $HOME/code/watch/secrets/secrets
        xdoctest dev/notebooks/stac_notebook.py print_provider_debug_information
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
    print('single_endpoint_collections = {}'.format(ub.urepr(single_endpoint_collections, nl=1)))
    print('multiendpoint_collections = {}'.format(ub.urepr(multiendpoint_collections, nl=1)))
    print('unique_endpoints = {}'.format(ub.urepr(unique_endpoints, nl=1)))

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
            print('Valid registered collections = {}'.format(ub.urepr(registered_names, nl=1)))
        if misregistered_names:
            print(f'!!! Collections are registered that dont exist {misregistered_names=}')
        if unregistered_names:
            print('There are unregistered collections = {}'.format(ub.urepr(unregistered_names, nl=1)))
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
        source $HOME/code/watch/secrets/secrets
        xdoctest $HOME/code/watch/dev/notebooks/stac_notebook.py check_processed_regions
    """
    import json
    import pystac_client
    from datetime import datetime as datetime_cls
    import watch
    import pandas as pd
    from rich import print

    dvc_data_dpath = watch.find_dvc_dpath(tags='phase2_data', hardware='hdd')

    # MODIFY AS NEEDED
    headers = {
        'x-api-key': os.environ['SMART_STAC_API_KEY']
    }

    base = ((dvc_data_dpath / 'annotations') / 'drop7')
    # base = dvc_data_dpath / 'annotations'

    region_dpath = base / 'region_models'
    region_fpaths = list(region_dpath.glob('*.geojson'))
    # region_fpaths = list(region_dpath.glob('*_C*.geojson'))
    # region_fpaths = list(region_dpath.glob('*_R*.geojson'))
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
        # 'ta1-s2-acc-2',
        # 'ta1-ls-acc-2',
        # 'ta1-wv-acc-2',
        # 'ta1-pd-acc-2',

        'ta1-10m-tsmoothed-acc-3',
        # 'ta1-10m-acc-3',

        'ta1-s2-acc-4',
        'ta1-ls-acc-4',
        'ta1-wv-acc-4',
        'ta1-pd-acc-4',

        'ta1-s2-acc-3',
        'ta1-ls-acc-3',
        'ta1-wv-acc-3',
        'ta1-pd-acc-3',

        # 'ta1-s2-acc',
        # 'ta1-s2-acc-1',
        # 'ta1-ls-acc',
        # 'ta1-ls-acc-1',
        # 'ta1-wv-acc',
        # 'ta1-wv-acc-1',
    ]

    #     'ta1-pd-acc',
    #     'ta1-pd-acc-1',
    #     'ta1-mixedgsd-acc-1',
    #     'ta1-30m-acc-1'
    # ]
    from kwutil import util_progress
    from watch.cli.stac_to_kwcoco import summarize_stac_item

    peryear_rows = []
    peritem_rows = []
    raw_stac_items = []

    mprog = util_progress.ProgressManager()
    jobs = ub.JobPool(mode='thread', max_workers=20)

    region_to_results = ub.ddict(list)

    with mprog, jobs:
        # Check that planet items exist
        for collection in mprog.progiter(collections_of_interest, desc='Query collections'):
            # Check that planet items exist in our regions
            region_iter = mprog.progiter(region_fpaths, desc=f'Submit query regions for {str(collection)}')
            for region_fpath in region_iter:
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
                job = jobs.submit(list, item_search.items())
                job.region_id = region_id
                job.collection = collection

        collect_errors = []

        import rich
        for job in mprog(jobs.as_completed(), total=len(jobs), desc='collect results'):
            region_id = job.region_id
            collection = job.collection
            try:
                results = job.result()
            except Exception as ex:
                rich.print(f'[red]ERROR IN {region_id} for {collection}: {ex}')
                collect_errors.append(ex)
                continue
            region_to_results[region_id] += results

            year_to_results = ub.udict(ub.group_items(results, key=lambda r: r.get_datetime().year))

            for year, year_results in year_to_results.items():
                for r in year_results:
                    summary = summarize_stac_item(r)
                    summary['collection'] = collection
                    summary['region_id'] = region_id
                    summary['year'] = year
                    r.extra_fields['collection'] = collection
                    r.extra_fields['region_id'] = region_id
                    r.extra_fields['year'] = year
                    peritem_rows.append(summary)
                    raw_stac_items.append(r)

                year_dates = [r.get_datetime() for r in year_results]
                min_date = min(year_dates)
                max_date = max(year_dates)
                peryear_rows.append({
                    'collection': collection,
                    'region_id': region_id,
                    'num_results': len(year_results),
                    'year': year,
                    'min_date': min_date.isoformat(),
                    'max_date': max_date.isoformat(),
                    # **year_oo_num
                })

    region_to_num_results = ub.udict(region_to_results).map_values(len)
    region_to_num_results = ub.udict(region_to_num_results).sorted_values()
    print('region_to_num_results = {}'.format(ub.urepr(region_to_num_results, nl=1)))

    # for r, v in region_to_num_results.items():
    #     if v:
    #         print('- $DVC_DATA_DPATH/annotations/drop7/region_models/' + r + '.geojson')

    if collect_errors:
        print("ERROR")
        print('collect_errors = {}'.format(ub.urepr(collect_errors, nl=1)))
        print("ERROR")

    for row in peryear_rows + peritem_rows:
        if row['collection'].endswith('acc-2'):
            row['processing'] = 'acc-2'
        elif row['collection'].endswith('acc-3'):
            row['processing'] = 'acc-3'
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
        if 'sensor' not in row:
            row['sensor'] = 'unknown'

    item_df = pd.DataFrame(peritem_rows)

    if 0:
        # Specific queries
        for item in raw_stac_items:
            summary = summarize_stac_item(item)
            b = summary['eo_bands']
            b = b if isinstance(b, str) else '|'.join(sorted(b))
            if b != 'pan':
                if summary['datetime'].year < 2016 and item.extra_fields['region_id'] == 'KR_R002':
                    # print(f'item.extra_fields={item.extra_fields}')
                    # print('summary = {}'.format(ub.urepr(summary, nl=1)))
                    print(item.id)

        sub = item_df[item_df.year < 2016]
        sub = sub[sub.sensor == 'worldview']
        sub.loc[:, 'eo_bands'] = [b if isinstance(b, str) else '|'.join(sorted(b)) for b in sub['eo_bands']]
        sub.loc[:, 'asset_names'] = [b if isinstance(b, str) else '|'.join(sorted(b)) for b in sub['asset_names']]
        for region_id, group in sub.groupby('region_id'):
            histo = group.value_counts(['region_id', 'sensor', 'year', 'eo_bands', 'eo_cloud_cover'])
            print(histo.to_string())

    def pandas_aggregate2(data, func, axis=0):
        assert isinstance(func, dict)
        agg_func = {}
        drop_cols = []
        for k, v in func.items():
            if isinstance(v, str) and v == 'drop':
                drop_cols.append(k)
            else:
                agg_func[k] = v
        df2 = data.drop(drop_cols, axis=1)
        out = df2.iloc[0:1].reset_index(drop=True).iloc[0].copy()
        agg_cols = df2.agg(agg_func, axis=axis)
        out.loc[agg_cols.index] = agg_cols
        return out

    # Aggregate over years
    agg_rows = []
    df = pd.DataFrame(peryear_rows)
    # print(df.to_string())

    for region_id, group in df.groupby(['region_id', 'collection']):
        agg = pandas_aggregate2(group, func={
            'num_results': 'sum',
            'min_date': 'min',
            'max_date': 'max',
            'year': 'drop',
        }, axis=0)
        agg_rows.append(agg)
    agg_df = pd.DataFrame(agg_rows)

    agg_df = agg_df.sort_values(['region_id', 'collection'])
    print(agg_df.to_string())

    # df = df.sort_values(['processing', 'sensor', 'collection'])
    # piv = df.pivot(['region_id'], ['processing', 'sensor', 'collection'], ['num_results'])

    agg_df = agg_df.sort_values(['sensor', 'processing', 'collection'])
    piv = agg_df.pivot(index=['region_id'], columns=['sensor', 'processing', 'collection'], values=['num_results'])
    # piv = piv.astype(bool)
    print(piv.to_string())

    df = df.sort_values(['sensor', 'processing', 'collection', 'year'])
    piv = df.pivot(index=['region_id'], columns=['sensor', 'processing', 'collection', 'year'], values=['num_results'])
    print(piv.to_string())

    for c, group in df.groupby('collection'):
        group = group.sort_values('year')
        piv = group.pivot(index=['region_id'], columns=['sensor', 'processing', 'collection', 'year'], values=['num_results'])
        print(piv.to_string())

    requested_regions = {p.name.split('.')[0] for p in region_fpaths}
    missing_regions = requested_regions - set(piv.index)
    print('No results for regions = {}'.format(ub.urepr(missing_regions, nl=0)))

    # print(df.to_string())
    # print(f'region_id={region_id}')
    # print(f'results={results}')
    # print(f'collection={collection}')
    # print('region_to_results = {}'.format(ub.urepr(region_to_results, nl=1)))


def _devcheck_providers_exist():
    """
    developer logic to test to see if providers are working
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
    print(ub.urepr(list(catalog.get_collections())))

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


def check_single_colletion():
    """
    source $HOME/code/watch/secrets/secrets
    COLLECTION=ta1-10m-tsmoothed-acc-3
    xdoctest -m watch.stac._notebook check_single_endpoint
    """
    import os
    import pystac_client
    import ubelt as ub
    collection = os.environ.get('COLLECTION', "ta1-10m-tsmoothed-acc-3")
    # item_search = catalog.search(collections=[collection])

    provider = "https://api.smart-stac.com"
    headers = {
        'x-api-key': os.environ['SMART_STAC_API_KEY']
    }
    catalog = pystac_client.Client.open(provider, headers=headers)

    item_search = catalog.search(collections=[collection])

    item_iter = iter(item_search.items())
    # View cloud cover
    first_item = next(item_iter)
    first_item_dict = first_item.to_dict()
    import rich
    rich.print('first_item_dict = {}'.format(ub.urepr(first_item_dict, nl=-1)))

    print(f'first_item={first_item}')
    print(f'first_item.assets={first_item.assets}')
    asset = ub.peek(first_item.assets.values())
    print(asset.to_dict())

    import watch
    from watch.geoannots import geomodels
    dvc_data_dpath = watch.find_dvc_dpath(tags='phase2_data', hardware='auto')
    base = ((dvc_data_dpath / 'annotations') / 'drop6')
    region_fpath = base / 'region_models/NZ_R001.geojson'
    region = geomodels.RegionModel.coerce(region_fpath)
    geom = region.geometry
    query = {}
    daterange = ('2010-01-01', '2020-01-01')

    search = catalog.search(
        collections=[collection],
        intersects=geom,
        max_items=None,
        query=query,
        datetime=daterange,
    )

    items_gen = search.items()
    items = list(items_gen)
    num_found = len(items)
    print(f'num_found={num_found}')
