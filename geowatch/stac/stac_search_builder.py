"""
A tool to help build a STAC search json.

This does contain a command line interface, for legacy reasons, but
is not intended to be a fully supported part of the GEOWATCH CLI.

SeeAlso:
    ../cli/stac_search.py

Accenture Notes:

    New Procesing 2022-11-21:
        https://smart-research.slack.com/?redir=%2Ffiles%2FU028UQGN1N0%2FF04B998ANRL%2Faccenture_ta1_productdoc_phaseii_20211117.pptx%3Forigin_team%3DTN3QR7WAH%26origin_channel%3DC03QTAXU7GF

        https://smartgitlab.com/TE/evaluations/-/wikis/Accenture-TA-1-Processing-Status
"""
import os
import ubelt as ub
import scriptconfig as scfg


class StacSearchBuilderConfig(scfg.DataConfig):
    """
    Helper to create STAC search json queries
    """
    __default__ = {
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

    'ta1-s2-acc-3': {
        'endpoint': "https://api.smart-stac.com",
        'collections': ['ta1-s2-acc-3'],
    },
    'ta1-ls-acc-3': {
        'endpoint': "https://api.smart-stac.com",
        'collections': ['ta1-ls-acc-3'],
    },
    'ta1-pd-acc-3': {
        'endpoint': "https://api.smart-stac.com",
        'collections': ['ta1-pd-acc-3'],
    },
    'ta1-wv-acc-3': {
        'endpoint': "https://api.smart-stac.com",
        'collections': ['ta1-wv-acc-3'],
        "query": {
            "nitf:imd": {
                "eq": "true"
            },
        }
    },

    'ta1-10m-tsmoothed-acc-3': {
        'endpoint': "https://api.smart-stac.com",
        'collections': ['ta1-10m-tsmoothed-acc-3'],
    }

}

_ARA_PHASE3_TA1_PRODUCTS = {
    'ta1-ls-ara-4': {
        'endpoint': "https://api.smart-stac.com",
        'collections': ['ta1-ls-ara-4'],
    },
    'ta1-pd-ara-4': {
        'endpoint': "https://api.smart-stac.com",
        'collections': ['ta1-pd-ara-4'],
    },
    'ta1-s2-ara-4': {
        'endpoint': "https://api.smart-stac.com",
        'collections': ['ta1-s2-ara-4'],
    },
    'ta1-wv-ara-4': {
        'endpoint': "https://api.smart-stac.com",
        'collections': ['ta1-wv-ara-4'],
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
    'sentinel-2-l1c': {
        "collections": ["sentinel-2-l1c"],
        "endpoint": "https://earth-search.aws.element84.com/v1",
    },
}


# NOTE;
# Info about QA bands:
# https://d9-wret.s3.us-west-2.amazonaws.com/assets/palladium/production/s3fs-public/media/files/LSDS-1435%20Landsat%20C2%20US%20ARD%20Data%20Format%20Control%20Book-v3.pdf

# Updated L2 Products
# https://www.element84.com/geospatial/introducing-earth-search-v1-new-datasets-now-available/

_PUBLIC_L2_PRODUCTS = {
    # Public L2 Products
    # https://docs.sentinel-hub.com/api/latest/data/sentinel-2-l2a/
    'sentinel-2-l2a': {
        "collections": ["sentinel-2-l2a"],
        "endpoint": "https://earth-search.aws.element84.com/v1",
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


# https://www.drivendata.org/competitions/143/tick-tick-bloom/page/650/#sentinel-2-1
# https://planetarycomputer.microsoft.com/dataset/landsat-c2-l2
_PUBLIC_L2_PRODUCTS['planetarycomputer'] = {
    'collections': ['landsat-c2-l2', 'sentinel-2-l2a'],
    "endpoint": "https://planetarycomputer.microsoft.com/api/stac/v1",
}

_PUBLIC_L2_PRODUCTS['planetarycomputer_l8'] = {
    'collections': ['landsat-c2-l2'],
    "endpoint": "https://planetarycomputer.microsoft.com/api/stac/v1",
}

_PUBLIC_L2_PRODUCTS['planetarycomputer_s2'] = {
    'collections': ['sentinel-2-l2a'],
    "endpoint": "https://planetarycomputer.microsoft.com/api/stac/v1",
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


# Note: these functions moved into the _notebook.py file in this directory
def print_provider_debug_information():
    raise NotImplementedError('Moved to the _notebook.py in this folder')


def check_processed_regions():
    raise NotImplementedError('Moved to the _notebook.py in this folder')


def _devcheck_providers_exist():
    raise NotImplementedError('Moved to the _notebook.py in this folder')

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
    _ARA_PHASE3_TA1_PRODUCTS,
)


# Simplified codes for the CLI
CONVINIENCE_SENSOR_GROUPS = {
    'TA1': [
        'ta1-s2-kit',
        # 'ta1-l8-kit',
        'ta1-wv-kit',
    ],
    'L2-S2': [
        'sentinel-2-l2a',
    ],
    'L2-L8': [
        'landsat-c2l2-sr',
        'landsat-c2l2-bt',
    ],
    'L2-S2-L8': [
        'sentinel-2-l2a',
        'landsat-c2l2-sr',
        'landsat-c2l2-bt',
    ],
    'ARD-L8': [
        'landsat-c2ard-sr',
        'landsat-c2ard-bt',
    ],
    'ARD-S2-L8': [
        'sentinel-2-l2a',
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
    'TA1-S2-L8-WV-PD-ACC-3': [
        'ta1-s2-acc-3',
        'ta1-ls-acc-3',
        'ta1-pd-acc-3',
        'ta1-wv-acc-3',
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
    'TA1-10M-TSMOOTH-ACC-3': [
        'ta1-10m-tsmoothed-acc-3',
    ],
    'TA1-S2-ACC-3': [
        'ta1-s2-acc-3',
    ],
}


def build_search_json(start_date, end_date, sensors, api_key, cloud_cover):
    """
    Construct the json that can be used for a stac search

    Example:
        >>> from geowatch.stac.stac_search_builder import build_search_json
        >>> start_date = '2017-01-01'
        >>> end_date = '2020-01-01'
        >>> sensors = 'L2-S2'
        >>> api_key = None
        >>> cloud_cover = 20
        >>> search_json = build_search_json(start_date, end_date, sensors, api_key, cloud_cover)
        >>> print('search_json = {}'.format(ub.urepr(search_json, nl=-1)))
        search_json = {
            'stac_search': [
                {
                    'collections': ['sentinel-2-l2a'],
                    'endpoint': 'https://earth-search.aws.element84.com/v1',
                    'query': {
                        'eo:cloud_cover': {'lt': 20}
                    },
                    'headers': {},
                    'start_date': '2017-01-01',
                    'end_date': '2020-01-01'
                }
            ]
        }

    """
    from kwutil import util_time

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
        >>> # xdoctest: +REQUIRES(env:SLOW_DOCTEST)
        >>> from geowatch.stac.stac_search_builder import main
        >>> cmdline = 0
        >>> kwargs = {
        >>>     'start_date': '2017-01-01',
        >>>     'end_date': '2020-01-01',
        >>>     'sensors': 'L2-S2',
        >>> }
        >>> main(cmdline=cmdline, **kwargs)
    """
    import json
    # import rich
    config = StacSearchBuilderConfig.cli(cmdline=cmdline, data=kwargs,
                                         strict=True)
    # rich.print(f'config = {ub.urepr(config, nl=2)}')

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
        python ~/code/geowatch/geowatch/stac/stac_search_builder.py
    """
    main()
