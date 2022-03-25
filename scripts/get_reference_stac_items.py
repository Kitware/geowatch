import os
import time
import json
from typing import List
from collections import defaultdict
import itertools
import pystac_client
import pystac
from progiter import ProgIter
import ubelt as ub


def convert_annotated_input_to_l1(
        path=('/home/local/KHQ/matthew.bernstein/smart/scratch/'
              'ALL_ANNOTATED_REGIONS_TA-1_PROCESSED_20220222.unique.input')):

    def _process(line):
        item = json.loads(line)
        orig_item_href = item['properties']['smart:source']
        if 'smart-stac' in orig_item_href:
            return line
        else:
            return json.dumps(pystac.Item.from_file(orig_item_href).to_dict())

    exc = ub.Executor('thread', max_workers=32)
    jobs = [exc.submit(_process, line) for line in open(path).readlines()]

    with open(path + '.l1', 'w') as f:
        for job in ProgIter(jobs):
            f.write(job.result())
            f.write('\n')


def convert_imerit_input_to_l1(
        path=('/home/local/KHQ/matthew.bernstein/smart/scratch/'
              'iMERIT_COMBINED.unique.input')):

    def _process(line):
        item = json.loads(line)
        orig_item_href = None
        for link in item['links']:
            if link['rel'] == 'original':
                orig_item_href = link['href']
                break
        if 'smart-stac' in orig_item_href:
            return line
        else:
            return json.dumps(pystac.Item.from_file(orig_item_href).to_dict())

    exc = ub.Executor('thread', max_workers=32)
    jobs = [exc.submit(_process, line) for line in open(path).readlines()]

    with open(path + '.l1', 'w') as f:
        for job in ProgIter(jobs):
            f.write(job.result())
            f.write('\n')


def convert_l1_input_to_l2(path=(
    '/home/local/KHQ/matthew.bernstein/smart/scratch/'
    'ALL_ANNOTATED_REGIONS_TA-1_PROCESSED_20220222.unique.input.l1')):

    ls_cli = pystac_client.Client.open(
        'https://landsatlook.usgs.gov/stac-server')

    def match_ls(item: pystac.Item) -> List[pystac.Item]:
        '''
        LS catalog doesn't support the Filter or Query extension, so have to
        filter manually after searching
        https://pystac-client.readthedocs.io/en/latest/usage.html#api-conformance
        '''
        kwargs = {
            'datetime': item.properties['datetime'],
            'collections': 'landsat-c2l2-sr'
        }
        items = list(ls_cli.search(**kwargs).get_all_items())
        items = [
            i for i in items if (i.properties['platform'] == 'LANDSAT_8'
                                 and i.properties['landsat:wrs_path'] ==
                                 item.properties['landsat:wrs_path']
                                 and i.properties['landsat:wrs_row'] ==
                                 item.properties['landsat:wrs_row']
                                 and i.properties['landsat:scene_id'] ==
                                 item.properties['landsat:scene_id'])
        ]
        return items

    s2_cli = pystac_client.Client.open(
        'https://earth-search.aws.element84.com/v0')

    def match_s2(item: pystac.Item) -> List[pystac.Item]:
        if 0:
            # these fields don't always exist in S2 items for some reason
            # filter requires pystac_client>=0.3
            latitude_band = item.properties.get(
                'mgrs:latitude_band',
                item.properties['sentinel:latitude_band'])
            grid_square = item.properties.get(
                'mgrs:grid_square', item.properties['sentinel:grid_square'])
            utm_zone = item.properties.get(
                'mgrs:utm_zone', item.properties['sentinel:utm_zone'])
            kwargs = {
                'filter': {
                    'eq': [{
                        'property': 'platform'
                    }, item.properties['platform'], {
                        'property': 'mgrs:latitude_band'
                    }, latitude_band, {
                        'property': 'mgrs:grid_square'
                    }, grid_square, {
                        'property': 'mgrs:utm_zone'
                    }, utm_zone]
                },
                'datetime': item.properties['datetime'],
                'collections': 'sentinel-s2-l2a-cogs'
            }
            # do MGRS tile matching manually
            # items = [i for i in items if i.id[:9] == item.id[:9]]
        else:
            # luckily, IDs just match up
            kwargs = {
                'ids': item.id.replace('_L1C', '_L2A'),
                'collections': 'sentinel-s2-l2a-cogs'
            }
        items = list(s2_cli.search(**kwargs).get_all_items())
        return items

    def _process(line):
        item = pystac.Item.from_dict(json.loads(line))

        platform = item.properties['platform']
        if platform.startswith('sentinel-2'):
            matched_items = match_s2(item)
        elif platform == 'LANDSAT_8':
            matched_items = match_ls(item)
        else:
            # probably a WV, but could be an edge case S2 or L8 we missed
            print(f'{item.id=} from unknown or WV source {platform=}')
            matched_items = [item]

        if len(matched_items) != 1:
            print(f'imperfect match {item} vs {matched_items}')
        else:
            return json.dumps(matched_items[0].to_dict())

    exc = ub.Executor('thread', max_workers=64)
    jobs = [exc.submit(_process, line) for line in open(path).readlines()]

    n_missing_items = 0
    with open(path.replace('.l1', '.l2'), 'w') as f:
        for job in ProgIter(jobs):
            line = job.result()
            if line is None:
                n_missing_items += 1
            else:
                f.write(line)
                f.write('\n')
    print(f'{n_missing_items=}')


def process_qfabric(
        root=('/home/local/KHQ/matthew.bernstein/smart/data/smart_watch_dvc/'
              'annotations-qfabric/orig/')):
    '''
    Region durations:
        np.histogram([(dateutil.parser.parse(r['features'][0]['properties']['end_date']) - dateutil.parser.parse(r['features'][0]['properties']['start_date'])).days for r in regions])
        (array([13, 10, 28, 55, 59, 53, 42, 27,  3,  2]),
 array([1081., 1247., 1413., 1579., 1745., 1911., 2077., 2243., 2409.,
        2575., 2741.]))

    Sites per region:
        Counter([len(r['features'])-1 for r in regions]).most_common()
        [(1, 97),
         (3, 48),
         (2, 47),
         (6, 17),
         (4, 15),
         (5, 14),
         (8, 7),
         (7, 7),
         (9, 6),
         (15, 5),
         (10, 5),
         (16, 2),
         (12, 2),
         (24, 2),
         (13, 2),
         (90, 1),
         (21, 1),
         (28, 1),
         (293, 1),
         (11, 1),
         (23, 1),
         (30, 1),
         (55, 1),
         (39, 1),
         (20, 1),
         (41, 1),
         (48, 1),
         (45, 1),
         (147, 1),
         (25, 1),
         (132, 1)]

    Sites per region with names:
        nsites_nm = [(len(r['features'])-1, r['features'][0]['properties']['region_id']) for r in regions]
        nsites, nm = zip(*nsites_nm)
        ub.group_items(nm, nsites)

    Observations per site:
        Counter([len(r['features'])-1 for r in sites])
        Counter({5: 2027, 11: 2, 4: 1})

    Categories:
        Counter(itertools.chain.from_iterable([[f['properties']['current_phase'] for f in s['features']] for s in sites]))
        Counter({None: 2032,
         'No Activity': 2717,
         'Active Construction': 1572,
         'Post Construction': 1949,
         'Site Preparation': 3921})

    Null sites (with only No Activity):
        n=0
        for s in sites:
            ixs = [i for i,f in enumerate(s['features']) if f['properties']['type'] == 'observation' and f['properties']['current_phase'] == 'No Activity']
            if len(ixs) == len(s['features']) - 1:
                print(ixs, s['features'][0]['properties']['site_id'])
                n += 1

        223

    Sites with full s2l8 coverage:
        ns_all, ns_notall = (0,0)
        s2l8_sites = []
        for site in sites2:
            if any(f['properties']['source'] is None for f in site['features'][1:]):
                ns_notall += 1
            else:
                ns_all += 1
                s2l8_sites.append(site)
        ns_all, ns_notall == (6, 2018)
        s2l8_rids = {s['features'][0]['properties']['region_id'] for s in s2l8_sites}
    '''

    root = ub.Path(root)

    # these regions have malformed geometry
    SKIP_REGIONS = ['BLA_QFABRIC_R089']
    # these sites have <5 observations
    SKIP_SITES = ['BLA_QFABRIC_R271_0067']
    MIN_SITES_PER_REGION = 1
    regions = [
        json.load(open(f)) for f in (root / 'region_models').iterdir()
        if (f.suffix == '.geojson' and (not any(s in f.name
                                                for s in SKIP_REGIONS)))
    ]
    regions = [
        r for r in regions if len(r['features']) - 1 >= MIN_SITES_PER_REGION
    ]

    rid_to_region_dct = {
        r['features'][0]['properties']['region_id']: r
        for r in regions
    }
    if 0:
        sites = [
            json.load(open(f)) for f in (root / 'site_models').iterdir()
            if f.suffix == '.geojson' and not any(s in f.name
                                                  for s in SKIP_SITES)
        ]
    else:
        # get sites associated with these regions
        region_names = set(rid_to_region_dct.keys())
        sites = [
            json.load(open(f)) for f in (root / 'site_models').iterdir()
            if (f.suffix == '.geojson' and any(
                s in f.name
                for s in region_names) and not any(s in f.name
                                                   for s in SKIP_SITES))
        ]

    # these sites have duplicated site and observation features
    DUPE_SITES = ['BLA_QFABRIC_R140_1457', 'BLA_QFABRIC_R129_0775']
    for site in sites:
        if site['features'][0]['properties']['site_id'] in DUPE_SITES:
            site['features'] = site['features'][1:7]

    rid_to_site_dct = defaultdict(list)
    for s in sites:
        rid_to_site_dct[s['features'][0]['properties']['region_id']].append(s)

    observations = list(
        itertools.chain.from_iterable([[
            f for f in site['features']
            if f['properties']['type'] == 'observation'
        ] for site in sites]))
    region_feats = list(
        itertools.chain.from_iterable([[
            r for r in rr['features'] if r['properties']['type'] == 'region'
        ] for rr in regions]))

    def _from(obs_or_reg, cli, coll, **kwargs):
        '''
        works on an observation or region feature
        '''
        if 'observation_date' in obs_or_reg['properties']:
            datetime = obs_or_reg['properties']['observation_date']
        else:
            datetime = '/'.join((obs_or_reg['properties']['start_date'],
                                 obs_or_reg['properties']['end_date']))
        s = cli.search(collections=coll,
                       datetime=datetime,
                       intersects=obs_or_reg['geometry'],
                       **kwargs)
        # returns an empty list if no matches
        try:
            tmp = s.get_all_items().items
            # print(s.matched())
            return tmp
        except pystac_client.exceptions.APIError:
            name = obs_or_reg['properties']['region_id']
            print(
                f'{s.matched()} {coll} {name} APIError - probably too many items requested'
            )
            # time.sleep(1)  # retry in 1 second
            # import xdev
            # with xdev.embed_on_exception_context():
            #     return s.get_all_items()
            return []

    FILTER_CLOUD = 0
    if FILTER_CLOUD:
        s2cli = pystac_client.Client.open(
            'https://earth-search.aws.element84.com/v0')
        s2_from = lambda obs: _from(obs,
                                    s2cli,
                                    'sentinel-s2-l1c',
                                    query={'eo:cloud_cover': {
                                        'lt': 10
                                    }})

        lscli = pystac_client.Client.open(
            'https://landsatlook.usgs.gov/stac-server')
        ls_from = lambda obs: _from(obs,
                                    lscli,
                                    'landsat-c2l1',
                                    query={
                                        'platform': {
                                            'eq': 'LANDSAT_8'
                                        },
                                        'eo:cloud_cover': {
                                            'lt': 10
                                        }
                                    })

        wvcli = pystac_client.Client.open(
            'https://api.smart-stac.com/',
            headers={'x-api-key': os.environ['API_KEY']})
        wv_from = lambda obs: filter(
            lambda: 1,
            _from(obs,
                  wvcli,
                  'worldview-nitf',
                  query={
                      'nitf:imd': {
                          'eq': 'true'
                      },
                      'eo:cloud_cover': {
                          'lt': 10
                      }
                  }))
    else:
        s2cli = pystac_client.Client.open(
            'https://earth-search.aws.element84.com/v0')
        s2_from = lambda obs: _from(obs, s2cli, 'sentinel-s2-l1c')

        lscli = pystac_client.Client.open(
            'https://landsatlook.usgs.gov/stac-server')
        ls_from = lambda obs: _from(obs,
                                    lscli,
                                    'landsat-c2l1',
                                    query={'platform': {
                                        'eq': 'LANDSAT_8'
                                    }})

        wvcli = pystac_client.Client.open(
            'https://api.smart-stac.com/',
            headers={'x-api-key': os.environ['API_KEY']})
        wv_from = lambda obs: _from(
            obs, wvcli, 'worldview-nitf', query={'nitf:imd': {
                'eq': 'true'
            }})

    # have to search the catalog to fill in the source field
    assert {o['properties']['source'] for o in observations} == {'smart-maxar'}

    exc = ub.Executor('thread', 64)

    # check loose coverage
    if 0:
        jobs1 = [exc.submit(s2_from, r) for r in region_feats]
        s2s = []
        for j in ProgIter(jobs1):
            s2s.append(j.result())
        s2s = list(itertools.chain.from_iterable(s2s))

        jobs2 = [exc.submit(ls_from, r) for r in region_feats]
        lss = []
        for j in ProgIter(jobs2):
            lss.append(j.result())
        lss = list(itertools.chain.from_iterable(lss))

        jobs3 = [exc.submit(wv_from, r) for r in region_feats]
        wvs = []
        for j in ProgIter(jobs3):
            wvs.append(j.result())
        wvs = list(itertools.chain.from_iterable(wvs))

    # check strict coverage
    else:
        jobs1 = [exc.submit(s2_from, o) for o in observations]
        s2s = []
        for j in ProgIter(jobs1):
            s2s.append(j.result())
        # s2s = list(itertools.chain.from_iterable(s2s))

        jobs2 = [exc.submit(ls_from, o) for o in observations]
        lss = []
        for j in ProgIter(jobs2):
            lss.append(j.result())
        # lss = list(itertools.chain.from_iterable(lss))
        # return s2s, lss

        jobs3 = [exc.submit(wv_from, o) for o in observations]
        wvs = []
        for j in ProgIter(jobs3):
            wvs.append(j.result())
        # wvs = list(itertools.chain.from_iterable(wvs))

    _dct = lambda its: dict(
        filter(
            lambda kv: len(kv[1]) > 0,
            zip([(o['properties']['site_id'], o['properties'][
                'observation_date']) for o in observations], its)))

    #
    # make s2l8-only dataset
    #

    s2l8_root = (root.parent / 's2l8').ensuredir()

    s2_dct = _dct(s2s)
    ls_dct = _dct(lss)

    n_has = 0
    n_s2 = 0
    n_nothas = 0
    sites2 = sites.copy()
    import xdev
    with xdev.embed_on_exception_context():
        for site in sites2:
            for obs in site['features']:
                if obs['properties']['type'] == 'observation':
                    key = (obs['properties']['site_id'],
                           obs['properties']['observation_date'])
                    if key in s2_dct:
                        obs['properties']['source'] = s2_dct[key][0].id
                        n_has += 1
                        n_s2 += 1
                    elif key in ls_dct:
                        obs['properties']['source'] = ls_dct[key][0].id
                        n_has += 1
                    else:
                        obs['properties']['source'] = None
                        n_nothas += 1
    assert n_has + n_nothas == len(observations)
    # n_has, n_s2, n_nothas
    # (2478, 1495, 7642)

    # limit this dataset to regions with at least one site with full s2l8 coverage
    s2l8_rids = ['BLA_QFABRIC_R234']
    s2l8_sites = [ s for s in sites2 if
        s['features'][0]['properties']['region_id'] in s2l8_rids]
    s2l8_regions = [ r for r in regions if
        r['features'][0]['properties']['region_id'] in s2l8_rids]

    (s2l8_root / 'region_models').ensuredir()
    for r in s2l8_regions:
        rid = r['features'][0]['properties']['region_id']
        with open(
                str((s2l8_root / 'region_models' /
                     f'{rid}.geojson')), 'w') as f:
            json.dump(r, f)
    (s2l8_root / 'site_models').ensuredir()
    for s in s2l8_sites:
        rid = s['features'][0]['properties']['region_id']
        with open(
                str((s2l8_root / 'site_models' /
                     f'{rid}.geojson')), 'w') as f:
            json.dump(s, f)

    with open(str(s2l8_root / 'qfabric_s2l8.input.l1'), 'w') as f:
        sources = set(
            list(
                filter(None, (itertools.chain.from_iterable(
                    [f['properties']['source'] for f in s['features']]
                    for s in sites)))))
        items = {
            it.id: it
            for it in itertools.chain.from_iterable(s2s + lss)
            if it.id in sources
        }
        for item in items.values():
            f.write(json.dumps(item.to_dict()))
            f.write('\n')

    convert_l1_input_to_l2(str(s2l8_root / 'qfabric_s2l8.input.l1'))

    #
    # make wvs2l8 dataset
    #

    wvs2l8_root = (root.parent / 'wvs2l8').ensuredir()

    wv_dct = _dct(wvs)

    sites3 = sites.copy()
    for site in sites3:
        for obs in site['features']:
            if obs['properties']['type'] == 'observation':
                key = (obs['properties']['site_id'],
                       obs['properties']['observation_date'])
                if key in wv_dct:
                    obs['properties']['source'] = wv_dct[key][0].id
                elif key in s2_dct:
                    obs['properties']['source'] = s2_dct[key][0].id
                elif key in ls_dct:
                    obs['properties']['source'] = ls_dct[key][0].id
                else:
                    obs['properties']['source'] = None

    wvs2l8_sites = sites3
    wvs2l8_regions = regions

    (s2l8_root / 'region_models').ensuredir()
    for rid, r in wvs2l8_regions.items():
        with open(
                str((wvs2l8_root / 'region_models' /
                     f'{rid}.geojson')), 'w') as f:
            json.dump(r, f)
    (s2l8_root / 'site_models').ensuredir()
    for rid, s in wvs2l8_sites.items():
        with open(
                str((wvs2l8_root / 'site_models' /
                     f'{rid}.geojson')), 'w') as f:
            json.dump(s, f)

    with open(str(wvs2l8_root / 'qfabric_wvs2l8_s2l8.input.l1'), 'w') as f:
        for item in itertools.chain.from_iterable(s2s + lss):
            f.write(json.dumps(item.to_dict()))
            f.write('\n')

    convert_l1_input_to_l2(str(wvs2l8_root / 'qfabric_wvs2l8_s2l8.input.l1'))

    with open(str(wvs2l8_root / 'qfabric_wvs2l8_wv.input'), 'w') as f:
        for item in itertools.chain.from_iterable(wvs):
            f.write(json.dumps(item.to_dict()))
            f.write('\n')



if __name__ == '__main__':
    process_qfabric()
