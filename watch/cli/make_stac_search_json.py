"""
A simple script to create a STAC search json
"""
import os
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
        'sensors': scfg.Value('L2', help=''),
    }


SENSOR_TO_DEFAULTS = {

    # https://landsatlook.usgs.gov/stac-server/
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
        'collections': ['ta1-l8-kit'],
        "query": {
            "nitf:imd": {
                "eq": "true"
            },
        }
    },
    'landsat-c2l1': {
        "collections": ["landsat-c2l1"],
        "endpoint": "https://landsatlook.usgs.gov/stac-server/",
        "query": {
            "platform": {
                "eq": "LANDSAT_8"
            }
        }
    },

    'sentinel-s2-l2a-cogs': {
        "collections": ["sentinel-s2-l2a-cogs"],
        "endpoint": "https://earth-search.aws.element84.com/v0",
    }
}


def main(cmdline=1, **kwargs):
    """
    Example:
        from watch.cli.make_stac_search_json import *  # NOQA
        cmdline = 0
        kwargs = {
            'start_date': '2017-01-01',
            'end_date': '2020-01-01',
            'sensors': 'TA1',
        }
        main(cmdline=cmdline, **kwargs)

    """
    config = StacSearchBuilderConfig(cmdline=cmdline, data=kwargs)

    if config['api_key'].startswith('env:'):
        api_environ_key = config['api_key'].split(':')[1]
        config['api_key'] = os.environ.get(api_environ_key, None)

    if config['sensors'] == 'TA1':
        config['sensors'] = [
            'ta1-s2-kit',
            'ta1-l8-kit',
            'ta1-wv-kit',
        ]
    elif config['sensors'] == 'L2':
        config['sensors'] = [
            'sentinel-s2-l2a-cogs',
        ]

    headers = {
            "x-api-key": config['api_key'],
    }

    # ub.timeparse()
    from watch.utils import util_time
    start_date = util_time.coerce_datetime(
        config['start_date'], default_timezone='utc')
    end_date = util_time.coerce_datetime(
        config['end_date'], default_timezone='utc')

    # if end_date is None:
    #     end_date = start_date
    # if start_date is None:
    #     start_date = end_date

    if start_date is None or end_date is None:
        raise ValueError('need start and end date')

    search_item_list = []
    for sensor in config['sensors']:
        search_item = SENSOR_TO_DEFAULTS[sensor]
        item_query = search_item.setdefault('query', {})
        search_item['headers'] = headers
        search_item['start_date'] = start_date.date().isoformat()
        search_item['end_date'] = end_date.date().isoformat()
        if config['cloud_cover'] is not None:
            item_query['eo:cloud_cover'] = config['cloud_cover']
        search_item_list.append(search_item)

    search_json = {
        'stac_search': search_item_list,
    }
    import json
    text = json.dumps(search_json, indent='    ')

    if config['out_fpath'] is not None:
        import ubelt as ub
        out_fpath = ub.Path(config['out_fpath'])
        out_fpath.write_text(text)
    else:
        print(text)

if __name__ == '__main__':
    """
    CommandLine:
        python ~/code/watch/watch/cli/make_stac_search_json.py
    """
    main()
