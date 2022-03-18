import pystac_client
import pystac
from progiter import ProgIter
import json
from typing import List
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


def convert_l1_input_to_l2(
        path=('/home/local/KHQ/matthew.bernstein/smart/scratch/'
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
        items = [i for i in items if
                 (i.properties['platform'] == 'LANDSAT_8' and
                  i.properties['landsat:wrs_path'] == item.properties['landsat:wrs_path'] and
                  i.properties['landsat:wrs_row'] == item.properties['landsat:wrs_row'] and
                  i.properties['landsat:scene_id'] == item.properties['landsat:scene_id'])]
        return items

    s2_cli = pystac_client.Client.open(
        'https://earth-search.aws.element84.com/v0')
    def match_s2(item: pystac.Item) -> List[pystac.Item]:
        if 0:
            # these fields don't always exist in S2 items for some reason
            # filter requires pystac_client>=0.3
            latitude_band = item.properties.get('mgrs:latitude_band', item.properties['sentinel:latitude_band'])
            grid_square = item.properties.get('mgrs:grid_square', item.properties['sentinel:grid_square'])
            utm_zone = item.properties.get('mgrs:utm_zone', item.properties['sentinel:utm_zone'])
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
