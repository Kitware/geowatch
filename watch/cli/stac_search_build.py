"""
A simple script to build a STAC search json
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


SENSOR_TO_DEFAULTS = {

    # https://landsatlook.usgs.gov/stac-server/

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

    # Accenture Phase 2 TA-1 Products
    'ta1-s2-acc': {
        'endpoint': "https://api.smart-stac.com",
        'collections': ['ta1-s2-acc'],
    },
    'ta1-l8-acc': {
        'endpoint': "https://api.smart-stac.com",
        'collections': ['ta1-l8-acc'],
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
        # https://stacindex.org/catalogs/usgs-landsat-collection-2-api#/
        "collections": ["sentinel-s2-l1c"],
        "endpoint": "https://earth-search.aws.element84.com/v0",
    },



    # Public L2 Products
    'sentinel-s2-l2a-cogs': {
        "collections": ["sentinel-s2-l2a-cogs"],
        "endpoint": "https://earth-search.aws.element84.com/v0",
    },

    'landsat-c2l2alb-sr': {
        "collections": ["landsat-c2l2alb-sr"],
        "endpoint": "https://landsatlook.usgs.gov/stac-server/",
        "query": {
            "platform": {
                "eq": "LANDSAT_8"
            }
        }
    },
}


def build_search_json(start_date, end_date, sensors, api_key, cloud_cover):
    from watch.utils import util_time

    if api_key.startswith('env:'):
        api_environ_key = api_key.split(':')[1]
        api_key = os.environ.get(api_environ_key, None)

    if sensors == 'TA1':
        sensors = [
            'ta1-s2-kit',
            'ta1-l8-kit',
            'ta1-wv-kit',
        ]
    elif sensors == 'L2-S2':
        sensors = [
            'sentinel-s2-l2a-cogs',
        ]
    elif sensors == 'L2-L8':
        sensors = [
            'landsat-c2l2alb-sr',
        ]
    elif sensors == 'L2-S2-L8':
        sensors = [
            'sentinel-s2-l2a-cogs',
            'landsat-c2l2alb-sr',
        ]
    elif sensors == 'TA1-S2-L8-ACC':
        sensors = [
            'ta1-s2-acc',
            'ta1-s2-acc',
        ]

    headers = {
            "x-api-key": api_key,
    }
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
        from watch.cli.stac_search_build import *  # NOQA
        cmdline = 0
        kwargs = {
            'start_date': '2017-01-01',
            'end_date': '2020-01-01',
            'sensors': 'TA1',
        }
        main(cmdline=cmdline, **kwargs)
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
        python ~/code/watch/watch/cli/stac_search_build.py
    """
    main()
