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

_ACCENTURE_PHASE2_TA1_PRODUCTS = {
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
        # https://stacindex.org/catalogs/usgs-landsat-collection-2-api#/
        "collections": ["sentinel-s2-l1c"],
        "endpoint": "https://earth-search.aws.element84.com/v0",
    },
}


_PUBLIC_L2_PRODUCTS = {
    # Public L2 Products
    'sentinel-s2-l2a-cogs': {
        "collections": ["sentinel-s2-l2a-cogs"],
        "endpoint": "https://earth-search.aws.element84.com/v0",
    },

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
}


SENSOR_TO_DEFAULTS = ub.dict_union(
    _KITWARE_PHASE1_TA1_PRODUCTS,
    _ACCENTURE_PHASE2_TA1_PRODUCTS,
    _PUBLIC_L1_PRODUCTS,
    _PUBLIC_L2_PRODUCTS,
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
        'landsat-c2ard-sr',
    ],
    'L2-S2-L8': [
        'sentinel-s2-l2a-cogs',
        'landsat-c2ard-sr',
    ],
    'TA1-S2-L8-ACC': [
        'ta1-s2-acc',
        'ta1-l8-acc',
    ],
    'TA1-S2-ACC': [
        'ta1-s2-acc',
        'ta1-l8-acc',
    ],
}


def build_search_json(start_date, end_date, sensors, api_key, cloud_cover):
    from watch.utils import util_time

    if api_key.startswith('env:'):
        api_environ_key = api_key.split(':')[1]
        api_key = os.environ.get(api_environ_key, None)

    if isinstance(sensors, str):
        sensors = CONVINIENCE_SENSOR_GROUPS[sensors]

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
