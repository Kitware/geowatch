'''
Collected information about satellite bands from https://github.com/stac-extensions/eo/

This should be mostly independent of the data source used.
(eg Google Cloud vs element84 AWS, Collection-1 vs Collection-2)
No guarantees that every value will be indentical, but close enough for heuristics.

Each constant is a list of 'eo:bands' Band Object dicts.
They are sorted in the same order in which they appear if the bands come stacked in a single image,
or in lexicographic order if the bands come in separate images.

Coverage of the catalogs is inconsistent. Where necessary, info has been filled in by hand.


Notes:

    Sentinal 2 Band Table
    =====================
    Band    Resolution    Central Wavelength    Description
    B1            60 m                443 nm    Ultra blue (Coastal and Aerosol)
    B2            10 m                490 nm    Blue
    B3            10 m                560 nm    Green
    B4            10 m                665 nm    Red
    B5            20 m                705 nm    Visible and Near Infrared (VNIR)
    B6            20 m                740 nm    Visible and Near Infrared (VNIR)
    B7            20 m                783 nm    Visible and Near Infrared (VNIR)
    B8            10 m                842 nm    Visible and Near Infrared (VNIR)
    B8a           20 m                865 nm    Visible and Near Infrared (VNIR)
    B9            60 m                940 nm    Short Wave Infrared (SWIR)
    B10           60 m               1375 nm    Short Wave Infrared (SWIR)
    B11           20 m               1610 nm    Short Wave Infrared (SWIR)
    B12           20 m               2190 nm    Short Wave Infrared (SWIR)


    Landsat 8 Band Table
    =====================
    Band    Resolution    Central Wavelength    Description
    1            30 m                 430 nm    Coastal aerosol
    2            30 m                 450 nm    Blue
    3            30 m                 530 nm    Green
    4            30 m                 640 nm    Red
    5            30 m                 850 nm    Near Infrared (NIR)
    6            30 m                1570 nm    SWIR 1
    7            30 m                2110 nm    SWIR 2
    8            15 m                 500 nm    Panchromatic
    9            30 m                1360 nm    Cirrus
    10           100 m              10600 nm    Thermal Infrared (TIRS) 1
    11           100 m              11500 nm    Thermal Infrared (TIRS) 2


    Worldview 3 MUL Band Table
    ==========================
    Band    Resolution    Central Wavelength    Description
    1           1.38 m                 400 nm    Coastal aerosol
    2           1.38 m                 450 nm    Blue
    3           1.38 m                 510 nm    Green
    4           1.38 m                 585 nm    Yellow
    5           1.38 m                 630 nm    Red
    6           1.38 m                 705 nm    Red edge
    7           1.38 m                 770 nm    Near-IR1
    8           1.38 m                 860 nm    Near-IR2

    Worldview 3 PAN Band Table
    ==========================
    1           0.34 m                 450-800 nm  Panchromatic

References:
    https://gis.stackexchange.com/questions/290796/how-to-edit-the-metadata-for-individual-bands-of-a-multiband-raster-preferably
    https://gisgeography.com/sentinel-2-bands-combinations/
    https://earth.esa.int/eogateway/missions/worldview-3
    https://www.usgs.gov/faqs/what-are-band-designations-landsat-satellites?qt-news_science_products=0#qt-news_science_products
'''


def dicts_contain(d_list, dsub_list):
    def contains(ds):
        return all(ds[0][k] == ds[1][k] for k in ds[1])
    return all(map(contains, zip(d_list, dsub_list)))

'''
This band info is taken from the sentinelhub AWS catalog.
It will need to be updated to match RGD's when that is STAC-compliant.

Note this is for S2B; S2A has slightly different center_wavelength and full_width_half_max

References:
    https://www.element84.com/earth-search/
    https://docs.sentinel-hub.com/api/latest/data/sentinel-2-l1c/

Example:
    >>> from pystac_client import Client
    >>> cat = Client.open('https://earth-search.aws.element84.com/v0')
    >>> search = cat.search(bbox=[-110, 39.5, -105, 40.5], max_items=1, collections=['sentinel-s2-l1c'])
    >>> i = list(search.items())[0]
    >>> # one image per band
    >>> bands = [v.to_dict()['eo:bands'][0] for k,v in i.assets.items() if k.startswith('B')]
    >>>
    >>> from watch.utils.util_bands import *
    >>> assert dicts_contain(SENTINEL2, bands)
'''
SENTINEL2 = [
        { 'name': 'B01', 'common_name': 'coastal', 'gsd': 60, 'center_wavelength': 0.4439, 'full_width_half_max': 0.027 },
        { 'name': 'B02', 'common_name': 'blue', 'gsd': 10, 'center_wavelength': 0.4966, 'full_width_half_max': 0.098 },
        { 'name': 'B03', 'common_name': 'green', 'gsd': 10, 'center_wavelength': 0.56, 'full_width_half_max': 0.045 },
        { 'name': 'B04', 'common_name': 'red', 'gsd': 10, 'center_wavelength': 0.6645, 'full_width_half_max': 0.038 },
        { 'name': 'B05', 'gsd': 20, 'center_wavelength': 0.7039, 'full_width_half_max': 0.019 },
        { 'name': 'B06', 'gsd': 20, 'center_wavelength': 0.7402, 'full_width_half_max': 0.018 },
        { 'name': 'B07', 'gsd': 20, 'center_wavelength': 0.7825, 'full_width_half_max': 0.028 },
        { 'name': 'B08', 'gsd': 10, 'common_name': 'nir', 'center_wavelength': 0.8351, 'full_width_half_max': 0.145 },
        { 'name': 'B8A', 'gsd': 20, 'center_wavelength': 0.8648, 'full_width_half_max': 0.033 },
        { 'name': 'B09', 'gsd': 60, 'center_wavelength': 0.945, 'full_width_half_max': 0.026 },
        { 'name': 'B10', 'gsd': 60, 'common_name': 'cirrus', 'center_wavelength': 1.3735, 'full_width_half_max': 0.075 },
        { 'name': 'B11', 'gsd': 20, 'common_name': 'swir16', 'center_wavelength': 1.6137, 'full_width_half_max': 0.143 },
        { 'name': 'B12', 'gsd': 20, 'common_name': 'swir22', 'center_wavelength': 2.22024, 'full_width_half_max': 0.242 }
]


'''
This band info is taken from the sentinelhub AWS catalog.
It will need to be updated to match RGD's when that is STAC-compliant.

References:
    https://www.element84.com/earth-search/
    https://docs.sentinel-hub.com/api/latest/data/landsat-8/

Example:
    >>> from pystac_client import Client
    >>> # for Collection 1
    >>> cat = Client.open('https://earth-search.aws.element84.com/v0')
    >>> search = cat.search(bbox=[-110, 39.5, -105, 40.5], max_items=1, collections=['landsat-8-l1-c1'])
    >>> # for Collection 2
    >>> # cat = Client.open('https://earth-search.aws.element84.com/v1')
    >>> # search = cat.search(bbox=[-110, 39.5, -105, 40.5], max_items=1, collections=['landsat-ot-l1'])
    >>> i = list(search.items())[0]
    >>> # one image for all bands
    >>> bands = [v.to_dict()['eo:bands'][0] for k,v in i.assets.items() if k.startswith('B') and (k != 'BQA')]
    >>>
    >>> from watch.utils.util_bands import *
    >>> assert dicts_contain(LANDSAT8, bands)
'''
LANDSAT8 = [
        { 'name': 'B1', 'common_name': 'coastal', 'gsd': 30, 'center_wavelength': 0.48, 'full_width_half_max': 0.02 },
        { 'name': 'B2', 'common_name': 'blue', 'gsd': 30, 'center_wavelength': 0.44, 'full_width_half_max': 0.06 },
        { 'name': 'B3', 'common_name': 'green', 'gsd': 30, 'center_wavelength': 0.56, 'full_width_half_max': 0.06 },
        { 'name': 'B4', 'common_name': 'red', 'gsd': 30, 'center_wavelength': 0.65, 'full_width_half_max': 0.04 },
        { 'name': 'B5', 'common_name': 'nir', 'gsd': 30, 'center_wavelength': 0.86, 'full_width_half_max': 0.03 },
        { 'name': 'B6', 'common_name': 'swir16', 'gsd': 30, 'center_wavelength': 1.6, 'full_width_half_max': 0.08 },
        { 'name': 'B7', 'common_name': 'swir22', 'gsd': 30, 'center_wavelength': 2.2, 'full_width_half_max': 0.2 },
        { 'name': 'B8', 'common_name': 'pan', 'gsd': 15, 'center_wavelength': 0.59, 'full_width_half_max': 0.18 },
        { 'name': 'B9', 'common_name': 'cirrus', 'gsd': 30, 'center_wavelength': 1.37, 'full_width_half_max': 0.02 },
        { 'name': 'B10', 'common_name': 'lwir11', 'gsd': 100, 'center_wavelength': 10.9, 'full_width_half_max': 0.8 },
        { 'name': 'B11', 'common_name': 'lwir12', 'gsd': 100, 'center_wavelength': 12, 'full_width_half_max': 1 }
]


'''
This band info is taken from the USGS Landsat catalog.

This is for Collection-2 Level-1; may be slightly different from
Collection-1 Level-1 (RGD's current source)

Example:
    >>> # not compatible with pystac_client for some reason
    >>> import requests
    >>> item = requests.get(('https://landsatlook.usgs.gov/sat-api/collections'
    >>>                      '/landsat-c2l1/items/LE07_L1TP_026043_20210518_20210518_02_RT')).json()
    >>> assets = item['assets']
    >>> keys = sorted(k for k in assets.keys() if 'B' in k)
    >>> bands = [assets[k]['eo:bands'][0] for k in keys]
    >>>
    >>> from watch.utils.util_bands import *
    >>> assert dicts_contain(LANDSAT7, bands)
'''
LANDSAT7 = [
    {'name': 'B1', 'common_name': 'blue', 'gsd': 30, 'center_wavelength': 0.49},
    {'name': 'B2', 'common_name': 'green', 'gsd': 30, 'center_wavelength': 0.56},
    {'name': 'B3', 'common_name': 'red', 'gsd': 30, 'center_wavelength': 0.66},
    {'name': 'B4', 'common_name': 'nir08', 'gsd': 30, 'center_wavelength': 0.84},
    {'name': 'B5', 'common_name': 'swir16', 'gsd': 30, 'center_wavelength': 1.65},
    {'name': 'B6', 'common_name': 'tir', 'gsd': 30, 'center_wavelength': 11.45},  # B6_VCID_1.TIF
    {'name': 'B6', 'common_name': 'tir', 'gsd': 30, 'center_wavelength': 11.45},  # B6_VCID_2.TIF
    {'name': 'B7', 'common_name': 'swir22', 'gsd': 30, 'center_wavelength': 2.22},
    {'name': 'B8', 'common_name': 'pan', 'gsd': 30, 'center_wavelength': 0.71}
]


'''
This band info is taken from the IARPA T&E STAC catalog.

Example:
    >>> # xdoctest: +SKIP
    >>> # requires the api_key for this catalog
    >>> from pystac_client import Client
    >>> catalog = Client.open('https://api.smart-stac.com/', headers={"x-api-key": api_key})
    >>> search = catalog.search(collections=['worldview-nitf'], bbox=[128.662489, 37.659517, 128.676673, 37.664560])
    >>> items = list(search.items())
    >>> props = [i.to_dict()['properties'] for i in items]
    >>>
    >>> wv01 = [p for p in props if p['mission'] == 'WV01']
    >>> assert np.unique([p['instruments'] for p in wv01]) == ['panchromatic']
    >>> wv01_pan = [p for p in wv01 if p['instruments'] == ['panchromatic']][0]['eo:bands']
    >>>
    >>> wv02 = [p for p in props if p['mission'] == 'WV02']
    >>> assert np.all(np.unique([p['instruments'] for p in wv02]) == ['panchromatic', 'vis-multi'])
    >>> assert np.all(np.unique([len(p['eo:bands']) for p in wv02]) == [1, 4, 8])
    >>> wv02_pan = [p for p in wv02 if p['instruments'] == ['panchromatic']][0]['eo:bands']
    >>> wv02_ms = [p['eo:bands'] for p in wv02 if p['instruments'] == ['vis-multi']]
    >>> wv02_ms4 = [p for p in wv02_ms if len(p) == 4][0]
    >>> wv02_ms8 = [p for p in wv02_ms if len(p) == 8][0]
    >>>
    >>> wv03 = [p for p in props if p['mission'] == 'WV03']
    >>> assert np.all(np.unique([p['instruments'] for p in wv03]) == ['panchromatic', 'vis-multi'])
    >>> assert np.all(np.unique([len(p['eo:bands']) for p in wv03]) == [1, 8])
    >>> wv03_pan = [p for p in wv03 if p['instruments'] == ['panchromatic']][0]['eo:bands']
    >>> wv03_ms = [p['eo:bands'] for p in wv03 if p['instruments'] == ['vis-multi']]
    >>> wv03_ms8 = [p for p in wv03_ms if len(p) == 8][0]
    >>>
    >>> # not sure if this must be true, but it is
    >>> assert wv02_pan == wv03_pan
    >>> assert wv02_ms8 == wv03_ms8
    >>>
    >>> from watch.utils.util_bands import *
    >>> assert dicts_contain(WORLDVIEW1_PAN, wv01_pan)
    >>> assert dicts_contain(WORLDVIEW2_PAN, wv02_pan)
    >>> assert dicts_contain(WORLDVIEW2_MS4, wv02_ms4)
    >>> assert dicts_contain(WORLDVIEW2_MS8, wv02_ms8)
    >>> assert dicts_contain(WORLDVIEW3_PAN, wv03_pan)
    >>> assert dicts_contain(WORLDVIEW3_MS8, wv03_ms8)
'''
WORLDVIEW1_PAN = [
    {'name': 'PAN', 'common_name': 'panchromatic', 'center_wavelength': 0.65}
]

WORLDVIEW2_PAN = [
    {'name': 'PAN', 'common_name': 'panchromatic', 'center_wavelength': 0.625}
]
WORLDVIEW2_MS4 = [
    {'name': 'B2', 'common_name': 'blue', 'center_wavelength': 0.48},
    {'name': 'B3', 'common_name': 'green', 'center_wavelength': 0.545},
    {'name': 'B5', 'common_name': 'red', 'center_wavelength': 0.66},
    {'name': 'B7', 'common_name': 'near-ir1', 'center_wavelength': 0.833}
]
WORLDVIEW2_MS8 = [
    {'name': 'B1', 'common_name': 'coastal', 'center_wavelength': 0.425},
    {'name': 'B2', 'common_name': 'blue', 'center_wavelength': 0.48},
    {'name': 'B3', 'common_name': 'green', 'center_wavelength': 0.545},
    {'name': 'B4', 'common_name': 'yellow', 'center_wavelength': 0.605},
    {'name': 'B5', 'common_name': 'red', 'center_wavelength': 0.66},
    {'name': 'B6', 'common_name': 'red-edge', 'center_wavelength': 0.725},
    {'name': 'B7', 'common_name': 'near-ir1', 'center_wavelength': 0.833},
    {'name': 'B8', 'common_name': 'near-ir2', 'center_wavelength': 0.95}
]


WORLDVIEW3_PAN = [
    {'name': 'PAN', 'common_name': 'panchromatic', 'center_wavelength': 0.625}
]
# TODO does WORLDVIEW3_MS4 not exist or did we just happen to not get it in the initial catalog?
WORLDVIEW3_MS8 = [
    {'name': 'B1', 'common_name': 'coastal', 'center_wavelength': 0.425},
    {'name': 'B2', 'common_name': 'blue', 'center_wavelength': 0.48},
    {'name': 'B3', 'common_name': 'green', 'center_wavelength': 0.545},
    {'name': 'B4', 'common_name': 'yellow', 'center_wavelength': 0.605},
    {'name': 'B5', 'common_name': 'red', 'center_wavelength': 0.66},
    {'name': 'B6', 'common_name': 'red-edge', 'center_wavelength': 0.725},
    {'name': 'B7', 'common_name': 'near-ir1', 'center_wavelength': 0.833},
    {'name': 'B8', 'common_name': 'near-ir2', 'center_wavelength': 0.95}
]

'''
TODO

fix wv doctest
'''

ALL_BANDS = (
    SENTINEL2 + LANDSAT8 + LANDSAT7 + WORLDVIEW1_PAN + WORLDVIEW2_PAN +
    WORLDVIEW2_MS4 + WORLDVIEW2_MS8 + WORLDVIEW3_PAN + WORLDVIEW3_MS8)

'''
WIP
Collect synonyms for allowed common_names values (not enforced by STAC)
TODO do we even need to conform to this? Should we only collect
"true" synonyms like {'pan': 'panchromatic'} ?

Example:
    >>> from watch.utils.util_bands import *
    >>> import itertools
    >>> names = set(b.get('common_name', '') for b in ALL_BANDS)
    >>> accounted_names = set(EO_COMMONNAMES.keys()).union(
    >>>     set(itertools.chain.from_iterable(EO_COMMONNAMES.values())))
    >>> todo = names.difference(accounted_names)
    >>> # not sure what to do with these
    >>> print(todo)
    {'', 'tir'}

References:
    https://github.com/stac-extensions/eo/blob/main/json-schema/schema.json#L151
'''
EO_COMMONNAMES = {
        "coastal": [],
        "blue": [],
        "green": [],
        "red": [],
        "rededge": ['red-edge'],
        "yellow": [],
        "pan": ['panchromatic'],
        "nir": ['near-ir1', 'near-ir2'],
        "nir08": [],
        "nir09": [],
        "cirrus": [],
        "swir16": [],
        "swir22": [],
        "lwir": [],
        "lwir11": [],
        "lwir12": []
}


'''
WIP
Bands that are used to observe targets on the ground
This is just a rough first pass

Example:
    >>> from watch.utils.util_bands import *
    >>> assert GROUND.issubset(set(EO_COMMONNAMES.keys()))
'''
GROUND = {
        "coastal",
        "blue",
        "green",
        "red",
        "rededge",
        "yellow",
        "pan",
        "nir",
        "nir08",
        "nir09",
}

'''
These band fields can be accessed as python objects as well using pystac

Example:
    >>> from pystac.extensions.eo import Band
    >>> from watch.utils.util_bands import *
    >>> for band in ALL_BANDS:
    >>>     band.pop('gsd', None)  # pystac doesn't support this yet
    >>>     b = Band.create(**band)
'''
