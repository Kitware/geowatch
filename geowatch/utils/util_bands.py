"""
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
    .. [MODIS-SPEC] https://modis.gsfc.nasa.gov/about/specifications.php
    .. [SentinelHub] https://apps.sentinel-hub.com/eo-browser/?zoom=15&lat=42.87425&lng=-73.83164&themeId=DEFAULT-THEME&visualizationUrl=https%3A%2F%2Fservices.sentinel-hub.com%2Fogc%2Fwms%2Fbd86bcc0-f318-402b-a145-015f85b9427e&datasetId=S2L2A&fromTime=2022-02-16T00%3A00%3A00.000Z&toTime=2022-02-16T23%3A59%3A59.999Z&layerId=4-FALSE-COLOR-URBAN
"""


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
    https://docs.sentinel-hub.com/api/latest/data/sentinel-2-l1c

Example:
    >>> from pystac_client import Client
    >>> from geowatch.utils.util_bands import *
    >>> cat = Client.open('https://earth-search.aws.element84.com/v1')
    >>> search = cat.search(bbox=[-110, 39.5, -105, 40.5], max_items=1, collections=['sentinel-2-l1c'])
    >>> i = list(search.items())[0]
    >>> # one image per band
    >>> bands = [v.to_dict()['eo:bands'][0] for k,v in i.assets.items() if k.startswith('B')]
    >>> assert dicts_contain(SENTINEL2, bands)
'''
SENTINEL2 = [
    {'name': 'B01', 'common_name': 'coastal', 'gsd': 60,
     'center_wavelength': 0.4439, 'full_width_half_max': 0.027},
    {'name': 'B02', 'common_name': 'blue', 'gsd': 10,
     'center_wavelength': 0.4966, 'full_width_half_max': 0.098},
    {'name': 'B03', 'common_name': 'green', 'gsd': 10,
     'center_wavelength': 0.56, 'full_width_half_max': 0.045},
    {'name': 'B04', 'common_name': 'red', 'gsd': 10,
     'center_wavelength': 0.6645, 'full_width_half_max': 0.038},
    {'name': 'B05',
     'gsd': 20,
     'center_wavelength': 0.7039,
     'full_width_half_max': 0.019},
    {'name': 'B06',
     'gsd': 20,
     'center_wavelength': 0.7402,
     'full_width_half_max': 0.018},
    {'name': 'B07',
     'gsd': 20,
     'center_wavelength': 0.7825,
     'full_width_half_max': 0.028},
    {'name': 'B08', 'gsd': 10, 'common_name': 'nir',
     'center_wavelength': 0.8351, 'full_width_half_max': 0.145},
    {'name': 'B8A',
     'gsd': 20,
     'center_wavelength': 0.8648,
     'full_width_half_max': 0.033},
    {'name': 'B09',
     'gsd': 60,
     'center_wavelength': 0.945,
     'full_width_half_max': 0.026},
    {'name': 'B10', 'gsd': 60, 'common_name': 'cirrus',
     'center_wavelength': 1.3735, 'full_width_half_max': 0.075},
    {'name': 'B11', 'gsd': 20, 'common_name': 'swir16',
     'center_wavelength': 1.6137, 'full_width_half_max': 0.143},
    {'name': 'B12', 'gsd': 20, 'common_name': 'swir22',
     'center_wavelength': 2.22024, 'full_width_half_max': 0.242}
]


'''
This band info is taken from the sentinelhub AWS catalog.
It will need to be updated to match RGD's when that is STAC-compliant.

References:
    https://www.element84.com/earth-search/
    https://docs.sentinel-hub.com/api/latest/data/landsat-8/
    https://landsat.gsfc.nasa.gov/satellites/landsat-8/
    https://planetarycomputer.microsoft.com/dataset/landsat-c2-l2

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
    >>> from geowatch.utils.util_bands import *
    >>> assert dicts_contain(LANDSAT8, bands)
'''
LANDSAT8 = [
    {'name': 'B1', 'common_name': 'coastal', 'gsd': 30,
     'center_wavelength': 0.48, 'full_width_half_max': 0.02,
     'alias': ['aerosol']},

    {'name': 'B2', 'common_name': 'blue', 'gsd': 30,
     'center_wavelength': 0.44, 'full_width_half_max': 0.06},

    {'name': 'B3', 'common_name': 'green', 'gsd': 30,
     'center_wavelength': 0.56, 'full_width_half_max': 0.06},

    {'name': 'B4', 'common_name': 'red', 'gsd': 30,
     'center_wavelength': 0.65, 'full_width_half_max': 0.04},

    {'name': 'B5', 'common_name': 'nir', 'gsd': 30,
     'center_wavelength': 0.86, 'full_width_half_max': 0.03,
     'alias': ['nir08']},

    {'name': 'B6', 'common_name': 'swir16', 'gsd': 30,
     'center_wavelength': 1.6, 'full_width_half_max': 0.08},

    {'name': 'B7', 'common_name': 'swir22', 'gsd': 30,
     'center_wavelength': 2.2, 'full_width_half_max': 0.2},

    {'name': 'B8', 'common_name': 'pan', 'gsd': 15,
     'center_wavelength': 0.59, 'full_width_half_max': 0.18},

    {'name': 'B9', 'common_name': 'cirrus', 'gsd': 30,
     'center_wavelength': 1.37, 'full_width_half_max': 0.02},

    {'name': 'B10', 'common_name': 'lwir11', 'gsd': 100,
     'center_wavelength': 10.9, 'full_width_half_max': 0.8,
     'alias': ['tir1'], 'notes': 'thermal-ir'},

    {'name': 'B11', 'common_name': 'lwir12', 'gsd': 100,
     'center_wavelength': 12, 'full_width_half_max': 1,
     'alias': ['tir2'], 'notes': 'thermal-ir'},
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
    >>> from geowatch.utils.util_bands import *
    >>> assert dicts_contain(LANDSAT7, bands)
'''
LANDSAT7 = [
    {'name': 'B1', 'common_name': 'blue', 'gsd': 30, 'center_wavelength': 0.49},
    {'name': 'B2', 'common_name': 'green', 'gsd': 30, 'center_wavelength': 0.56},
    {'name': 'B3', 'common_name': 'red', 'gsd': 30, 'center_wavelength': 0.66},
    {'name': 'B4', 'common_name': 'nir08', 'gsd': 30, 'center_wavelength': 0.84},
    {'name': 'B5', 'common_name': 'swir16', 'gsd': 30, 'center_wavelength': 1.65},
    {'name': 'B6', 'common_name': 'tir', 'gsd': 30,
        'center_wavelength': 11.45},  # B6_VCID_1.TIF
    {'name': 'B6', 'common_name': 'tir', 'gsd': 30,
        'center_wavelength': 11.45},  # B6_VCID_2.TIF
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
    >>> from geowatch.utils.util_bands import *
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
# TODO does WORLDVIEW3_MS4 not exist or did we just happen to not get it
# in the initial catalog?
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

PLANETSCOPE_3BAND = [
    {'name': 'B1', 'common_name': 'red'},
    {'name': 'B2', 'common_name': 'green'},
    {'name': 'B3', 'common_name': 'blue'}
]

PLANETSCOPE_4BAND = [
    {'name': 'B1', 'common_name': 'red'},
    {'name': 'B2', 'common_name': 'green'},
    {'name': 'B3', 'common_name': 'blue'},
    {'name': 'B4', 'common_name': 'nir'}
]

PLANETSCOPE_8BAND = [
    {'name': 'B1', 'common_name': 'coastal'},
    {'name': 'B2', 'common_name': 'blue'},
    {'name': 'B3', 'common_name': 'green1'},
    {'name': 'B4', 'common_name': 'green'},
    {'name': 'B5', 'common_name': 'yellow'},
    {'name': 'B6', 'common_name': 'red'},
    {'name': 'B7', 'common_name': 'red-edge'},
    {'name': 'B8', 'common_name': 'nir'}
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
    >>> from geowatch.utils.util_bands import *
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
    'coastal': [],
    'blue': [],
    'green': [],
    'red': [],
    'rededge': ['red-edge'],
    'yellow': [],
    'pan': ['panchromatic'],
    'nir': ['near-ir1', 'near-ir2'],
    'nir08': [],
    'nir09': [],
    'cirrus': [],
    'swir16': [],
    'swir22': [],
    'lwir': [],
    'lwir11': [],
    'lwir12': []
}


'''
WIP
Bands that are used to observe targets on the ground
This is just a rough first pass

Example:
    >>> from geowatch.utils.util_bands import *
    >>> assert GROUND.issubset(set(EO_COMMONNAMES.keys()))
'''
GROUND = {
    'coastal',
    'blue',
    'green',
    'red',
    'rededge',
    'yellow',
    'pan',
    'nir',
    'nir08',
    'nir09',
}

'''
These band fields can be accessed as python objects as well using pystac

Example:
    >>> from pystac.extensions.eo import Band
    >>> from geowatch.utils.util_bands import *
    >>> for band in ALL_BANDS:
    >>>     band.pop('gsd', None)  # pystac doesn't support this yet
    >>>     b = Band.create(**band)
'''


# https://modis.gsfc.nasa.gov/about/specifications.php
MODIS = {
    'name': 'MODIS',
    'quantization': '12 bits',
    'bands': [
        {'band': 1, 'gsd': '250m', 'bandwidth': '620-670nm', 'spectral_radiance': 21.8, 'required_snr': 128, 'primary_use': 'Land/Cloud/Aerosols Boundaries'},
        {'band': 2, 'gsd': '250m', 'bandwidth': '841-876nm', 'spectral_radiance': 24.7, 'required_snr': 201, 'primary_use': 'Land/Cloud/Aerosols Boundaries'},

        {'band': 3, 'gsd': '500m', 'bandwidth': '459 - 479nm', 'spectral_radiance': 35.3, 'required_snr': 243, 'primary_use': 'Land/Cloud/Aerosols Properties'},
        {'band': 5, 'gsd': '500m', 'bandwidth': '545 - 565nm', 'spectral_radiance': 29.0, 'required_snr': 228, 'primary_use': 'Land/Cloud/Aerosols Properties'},
        {'band': 5, 'gsd': '500m', 'bandwidth': '1230 - 1250nm', 'spectral_radiance': 5.4, 'required_snr': 74, 'primary_use': 'Land/Cloud/Aerosols Properties'},
        {'band': 6, 'gsd': '500m', 'bandwidth': '1628 - 1652nm', 'spectral_radiance': 7.3, 'required_snr': 275, 'primary_use': 'Land/Cloud/Aerosols Properties'},
        {'band': 7, 'gsd': '500m', 'bandwidth': '2105 - 2155nm', 'spectral_radiance': 1.0, 'required_snr': 110, 'primary_use': 'Land/Cloud/Aerosols Properties'},

        {'band': 8, 'gsd': '1000m', 'bandwidth': '405 - 420nm', 'primary_use': 'Ocean Color/Phytoplankton/Biogeochemistry'},
        {'band': 9, 'gsd': '1000m', 'primary_use': 'Ocean Color/Phytoplankton/Biogeochemistry'},
        {'band': 10, 'gsd': '1000m', 'primary_use': 'Ocean Color/Phytoplankton/Biogeochemistry'},
        {'band': 11, 'gsd': '1000m', 'primary_use': 'Ocean Color/Phytoplankton/Biogeochemistry'},
        {'band': 12, 'gsd': '1000m', 'primary_use': 'Ocean Color/Phytoplankton/Biogeochemistry'},
        {'band': 13, 'gsd': '1000m', 'primary_use': 'Ocean Color/Phytoplankton/Biogeochemistry'},
        {'band': 14, 'gsd': '1000m', 'primary_use': 'Ocean Color/Phytoplankton/Biogeochemistry'},
        {'band': 15, 'gsd': '1000m', 'primary_use': 'Ocean Color/Phytoplankton/Biogeochemistry'},
        {'band': 16, 'gsd': '1000m', 'bandwidth': '862 - 877nm', 'primary_use': 'Ocean Color/Phytoplankton/Biogeochemistry'},
    ]
}


# https://ladsweb.modaps.eosdis.nasa.gov/missions-and-measurements/viirs/
# https://lpdaac.usgs.gov/data/get-started-data/collection-overview/missions/s-npp-nasa-viirs-overview/
# https://rammb.cira.colostate.edu/projects/npp/VIIRS_bands_and_bandwidths.pdf
# The VIIRS instrument provides 22 spectral bands from 412 nanometers (nm) to 12
# micrometers (µm) at two spatial resolutions, 375 meters (m) and 750 m, which
# are resampled to 500 m, 1 km, and 0.05 degrees in the NASA produced data
# products to promote consistency with the MODIS heritage.
VIIRS = {
    'name': 'VIIRS',
    'quantization': '12 bits',
    'gds': '750m',
    'bands': [
        {'name': 'I1', 'gsd': '375m', 'reflected_range': '0.6 - 0.68um', 'desc': 'Near Infrared', 'common_name': 'blue'},
        {'name': 'I2', 'gsd': '375m', 'common_name': 'green'},
        {'name': 'I3', 'gsd': '375m', 'common_name': 'red'},
        {'name': 'I4', 'gsd': '375m'},
        {'name': 'I5', 'gsd': '375m'},

        {'name': 'M1', 'gsd': '750m'},
        {'name': 'M2', 'gsd': '750m'},
        {'name': 'M3', 'gsd': '750m'},
        {'name': 'M4', 'gsd': '750m'},
        {'name': 'M5', 'gsd': '750m'},
        {'name': 'M6', 'gsd': '750m'},
        {'name': 'M7', 'gsd': '750m'},
        {'name': 'M8', 'gsd': '750m'},
        {'name': 'M9', 'gsd': '750m'},
        {'name': 'M10', 'gsd': '750m'},
        {'name': 'M11', 'gsd': '750m'},
        {'name': 'M12', 'gsd': '750m'},
        {'name': 'M13', 'gsd': '750m'},
        {'name': 'M14', 'gsd': '750m'},
        {'name': 'M15', 'gsd': '750m'},
        {'name': 'M16', 'gsd': '750m', 'reflected_range': '11.54 - 12.49um'},

        {'name': 'DNB', 'gsd': '750m', 'reflected_range': '0.5 - 0.9um', 'desc': 'Visible/Reflective'},
    ]
}


SENTINEL1 = {
    'name': 'Sentinel1',
    'notes': 'SAR',
    'gsd': '1.7x4.3m - 3.6x4.9m',
}


# TODO: move to a new "Spectral-Index" module

def specialized_index_bands(bands=None, coco_img=None, symbolic=False):
    r"""
    Ported from code from by (Yongquan Zhao on 26 April 2017)

    References:
        https://mail.google.com/mail/u/1/#chat/space/AAAAE5jpxTc

    Ignore:
        DVC_DPATH=$HOME/data/dvc-repos/smart_watch_dvc
        jq '.images[0].id' $DVC_DPATH/drop1-S2-L8-aligned/data.kwcoco.json
        kwcoco subset --src $DVC_DPATH/drop1-S2-L8-aligned/data.kwcoco.json --gids=2, --dst=./one_image_data/data.kwcoco.json --copy_assets=True

    Example:
        >>> # xdoctest: +REQUIRES(env:DVC_DPATH)
        >>> from geowatch.utils.util_bands import *  # NOQA
        >>> from geowatch.utils.util_data import find_dvc_dpath
        >>> import kwcoco
        >>> import ubelt as ub
        >>> dvc_dpath = find_dvc_dpath()
        >>> coco_fpath = dvc_dpath / 'drop1-S2-L8-aligned/data.kwcoco.json'
        >>> dset = kwcoco.CocoDataset(coco_fpath)
        >>> from geowatch.utils import kwcoco_extensions
        >>> gid = ub.peek(dset.index.imgs.keys())
        >>> vidid = dset.index.name_to_video['BH_Manama_R01']['id']
        >>> gid = dset.index.vidid_to_gids[vidid][0]
        >>> coco_img = dset.coco_image(gid)
        >>> print('coco_img.channels = {!r}'.format(coco_img.channels))
        >>> symbolic = False
        >>> indexes = specialized_index_bands(coco_img=coco_img)
        >>> #indexes = ub.dict_isect(indexes, {"ASI", 'AF_Norm', 'SSF_Norm', 'VSF_Norm', 'MF_Norm'})
        >>> indexes = ub.dict_isect(indexes, {"ASI"})
        >>> import kwarray
        >>> print(ub.urepr(ub.map_vals(kwarray.stats_dict, indexes), nl=1))
        >>> import pandas as pd
        >>> print(pd.DataFrame(ub.map_vals(kwarray.stats_dict, indexes)))
        >>> # xdoctest: +REQUIRES(--show)
        >>> import kwplot
        >>> import kwimage
        >>> kwplot.autompl()
        >>> pnum_ = kwplot.PlotNums(nSubplots=len(indexes))
        >>> #value = kwimage.normalize_intensity(coco_img.imdelay('red|green|blue').finalize())
        >>> #kwplot.imshow(value, title='red|green|blue', pnum=pnum_())
        >>> for key, value in indexes.items():
        >>>     kwplot.imshow(kwimage.normalize(value), title=key, pnum=pnum_())


    # Example:
    #     >>> # xdoctest: +REQUIRES(module:sympy)
    #     >>> from geowatch.utils.util_bands import *  # NOQA
    #     >>> symbolic = True
    #     >>> indexes = specialized_index_bands(coco_img=None, symbolic=symbolic)
    #     >>> import sympy as sym
    #     >>> for key, index in indexes.items():
    #     >>>     print('===============')
    #     >>>     print('key = {!r}'.format(key))
    #     >>>     print('\nOrig {}'.format(key))
    #     >>>     print(index)
    #     >>>     print('\nSimplified {}'.format(key))
    #     >>>     print(sym.simplify(index))
    """
    import ubelt as ub
    import numpy as np
    import kwimage

    if bands is not None:
        allbands = np.stack(list(ub.take(bands, ['blue', 'green', 'red', 'swir16', 'swir22', 'nir'])))
        allbands = kwimage.normalize_intensity(allbands)
        Blue, Green, Red, SWIR1, SWIR2, NIR = allbands

    # Raw bands
    elif symbolic:
        # Sympy can help explore different forms of these equations.
        import sympy as sym
        Blue, Green, Red, SWIR1, SWIR2, NIR = sym.symbols(
            'Blue, Green, Red, SWIR1, SWIR2, NIR')
    else:
        delayed = coco_img.imdelay()
        rgbir123 = delayed.take_channels('blue|green|red|swir16|swir22|nir')
        chw = rgbir123.finalize().transpose(2, 0, 1)

        chw = kwimage.normalize_intensity(chw)

        Blue = chw[0]    # NOQA
        Green = chw[1]  # NOQA
        Red = chw[2]      # NOQA
        SWIR1 = chw[3]  # NOQA
        SWIR2 = chw[4]  # NOQA
        NIR = chw[5]      # NOQA

    Blue = Blue     # NOQA
    Grn = Green   # NOQA
    Red = Red      # NOQA
    SWIR2 = SWIR1   # NOQA
    SWIR1 = SWIR2  # NOQA
    NIR = NIR     # NOQA

    def hist_cut(band, fill_value=0, k=1, minmax='std'):
        if minmax == 'std':
            mean = band.mean()
            std = band.std()
            low_val = (mean + k * std)
            high_val = (mean + k * std)
        else:
            low_val, high_val = minmax
        # is_low = band < low_val
        # is_high = band > high_val
        # [is_high] = fill
        band = band.clip(low_val, high_val)
        return band

    def minmax_norm(band, mask):
        mask = mask & np.isfinite(band)
        if mask.sum() > 0:
            max_val = band[mask].max()
            min_val = band[mask].min()
            extent = max_val - min_val
            if extent > 0:
                shifted = band - min_val
                scaled = shifted / extent
                band[mask] = scaled[mask]
        return band

    # TODO:
    # Need a valid_mask
    valid_mask = None
    if valid_mask is None:
        valid_mask = np.ones(Blue.shape, dtype=bool)

    MinMaxNorm = minmax_norm
    HistCut = hist_cut
    MaskValid = valid_mask

    # constants
    G = 2.5
    C1 = 6
    C2 = 7.5

    # L = 1
    L = 1000

    fillV = 0

    # Might be tricky
    # toplogical_effects = False

    # Formulas

    # Artificial surface Factor (AF).
    AF = ((SWIR1 + NIR) / 2 - Blue) / ((SWIR1 + NIR) / 2 + Blue)
    AF = HistCut(AF, fillV, 6, [-1, 1])
    AF_Norm = MinMaxNorm(AF, MaskValid)

    # Vegetation Suppressing Factor (VSF).
    # EVI is better in general cases, but its adjustment for mountain shadows
    # (with vegetation) is not as good as NDVI.
    EVI = G * ((NIR - Red) / (NIR + C1 * Red - C2 * Blue + L))
    EVI = HistCut(EVI, fillV, 6, [-1, 1])
    VSF = 1 - EVI
    # NDVI = (NIR - Red) / (NIR + Red)
    #  NDVI  = HistCut( NDVI, fillV, 6, [-1 1])
    # VSF = 1 - NDVI
    VSF_Norm = MinMaxNorm(VSF, MaskValid)

    # Soil Suppressing Factor (SSF).
    # Derive the Modified Bare soil Index (MBI).
    MBI = (SWIR1 - SWIR2 - NIR) / (SWIR1 + SWIR2 + NIR) + 0.5
    MBI = HistCut(MBI, fillV, 6, [-0.5, 1.5])
    MBI_Norm = MinMaxNorm(MBI, MaskValid)
    # Deriving Enhanced-MBI based on MBI and MNDWI.
    MNDWI = (Grn - SWIR1) / (Grn + SWIR1)
    MNDWI = HistCut(MNDWI, fillV, 6, [-1, 1])
    MNDWI_Norm = MinMaxNorm(MNDWI, MaskValid)
    EMBI = ((MBI_Norm) - (MNDWI_Norm)) / ((MBI_Norm) + (MNDWI_Norm))
    EMBI_Norm = MinMaxNorm(EMBI, MaskValid)

    invalid = (MBI_Norm == 0)
    MBI_Norm[invalid] = 1
    C = EMBI_Norm / MBI_Norm
    C[invalid] = 0

    # Derive the Bare Soil Index (BSI).
    BSI = ((SWIR1 + Red) - (NIR + Blue)) / ((SWIR1 + Red) + (NIR + Blue))
    MNDWI = HistCut(MNDWI, fillV, 6, [-1, 1])
    EMBI = HistCut(EMBI, fillV, 6, [-1, 1])
    BSI = HistCut(BSI, fillV, 6, [-1, 1])
    BSI_Norm = MinMaxNorm(BSI, MaskValid)
    # Deriving Eenhanced-BSI based on BSI and MNDWI.
    EBSI = ((BSI_Norm) - (MNDWI_Norm)) / ((BSI_Norm) + (MNDWI_Norm))
    MNDWI = HistCut(MNDWI, fillV, 6, [-1, 1])
    EMBI = HistCut(EMBI, fillV, 6, [-1, 1])
    BSI = HistCut(BSI, fillV, 6, [-1, 1])
    EBSI = HistCut(EBSI, fillV, 6, [-1, 1])

    # Derive SSF.
    SSF = C * (1 - EBSI)
    SSF_Norm = MinMaxNorm(SSF, MaskValid)

    # Modulation Factor (MF).
    MF = (Blue + Grn - NIR - SWIR1) / (Blue + Grn + NIR + SWIR1)
    MNDWI = HistCut(MNDWI, fillV, 6, [-1, 1])
    EMBI = HistCut(EMBI, fillV, 6, [-1, 1])
    BSI = HistCut(BSI, fillV, 6, [-1, 1])
    EBSI = HistCut(EBSI, fillV, 6, [-1, 1])

    MF = HistCut(MF, fillV, 6, [-1, 1])
    MF_Norm = MinMaxNorm(MF, MaskValid)

    # Derive ASI.
    ASI = AF_Norm * SSF_Norm * VSF_Norm * MF_Norm

    MNDWI = HistCut(MNDWI, fillV, 6, [-1, 1])
    EMBI = HistCut(EMBI, fillV, 6, [-1, 1])
    BSI = HistCut(BSI, fillV, 6, [-1, 1])
    EBSI = HistCut(EBSI, fillV, 6, [-1, 1])

    MF = HistCut(MF, fillV, 6, [-1, 1])
    ASI = HistCut(ASI, fillV, 6, [0, 1])

    # # The Artificial surface Factor (AF)
    # AF = (((SWIR1 + NIR) / 2) - Blue) / (((SWIR1 + NIR) / 2) + Blue) + 1

    # # %%%%% Vegetation Suppressing Factor (VSF).
    # # % EVI is better in general cases, but its adjustment for mountain shadows (with vegetation) is not as good as NDVI.
    # # EVI = G * ((NIR - Red) / (NIR + C1 * Red - C2 * Blue + L))
    # # https://en.wikipedia.org/wiki/Enhanced_vegetation_index
    # # Enhanced vegetation index
    # EVI = G * (NIR - Red) / (NIR + C1 * Red - C2 * Blue + L)
    # EVI = hist_cut(EVI, fillV)

    # # Normalized Difference Vegetation Index
    # # https://gisgeography.com/ndvi-normalized-difference-vegetation-index/
    # NDVI = (NIR - Red) / (NIR + Red)

    # # bare soil index
    # # https://www.geo.university/pages/blog?p=spectral-indices-with-multispectral-satellite-data#:~:text=Bare%20Soil%20Index%20(BSI)%20is,used%20in%20a%20normalized%20manner.
    # BSI = ((SWIR1 + Red) - (NIR + Blue)) / ((SWIR1 + Red) + (NIR + Blue))
    # BSI  = hist_cut( BSI, fillV, 6, minmax=[-1, 1])
    # BSI_Norm = minmax_norm(BSI, valid_mask)

    # # Modified Bare Soil Index
    # MBI = ((SWIR1 - SWIR2 - NIR) / (SWIR1 + SWIR2 + NIR)) + 0.5
    # MBI = hist_cut(MBI, fillV, minmax=[-0.5, 1.5])

    # # Note that BSI, MBI, and MNDWI need to be normalized when calculation EBSI
    # # and EMBI to avoid unmeaningful values caused by the negative values in
    # # BSI/MBI/MNDW
    # MNDWI = (Green - SWIR1) / (Green + SWIR1)
    # MNDWI  = hist_cut( MNDWI, fillV, 6, minmax=[-1, 1])
    # MNDWI_Norm = minmax_norm(MNDWI, valid_mask)

    # # EBSI/EMBI were designed based on BSI/MBI to enhance barren further,
    # # however, artificial surface still has relative high magnitude, thereby
    # # leading to the over-depressing of artificial surface.

    # # EBSI = (BSI - MNDWI) / (BSI + MNDWI)
    # EBSI = ((BSI_Norm) - (MNDWI_Norm)) / ((BSI_Norm) + (MNDWI_Norm));

    # EBSI  = hist_cut( EBSI, fillV, 6, minmax=[-1, 1])

    # EMBI = (MBI - MNDWI) / (MBI + MNDWI)
    # EMBI  = hist_cut( EMBI, fillV, 6, minmax=[-1, 1])

    # # Soil Depressing Factor
    # SDF = (EMBI + 1) / (MBI + 0.5) * (1 - EBSI)

    # # Vegetation Depressing Factor
    # # VDF is based on NDVI (Rouse et al. 1974) or EVI (Huete et al. 1997),
    # VDF = 1 - EVI if toplogical_effects else 1 - NDVI

    # # Therefore, we design a modulation factor to depress the dark bare land
    # # and enhance dark artificial surfaces simultaneous
    # MF = ((Blue + Green) - (NIR + SWIR1)) / ((Blue + Green) + (NIR + SWIR1)) + 1
    # MF  = hist_cut( MF, fillV, 6, minmax=[-1, 1])

    # ASI = (AF * SDF * VDF * MF) + 1
    # ASI  = hist_cut( ASI, fillV, 6, minmax=[0, 1])

    indexes = {
        'ASI': ASI,
        'MF': MF,
        # 'VDF': VDF,
        # 'SDF': SDF,
        'AF': AF,
        'EBSI': EBSI,
        'MNDWI': MNDWI,
        'MBI': MBI,
        'BSI': BSI,
        'BSI_Norm': BSI_Norm,
        'EVI': EVI,

        'EMBI': EMBI,

        'SSF_Norm': SSF_Norm,
        'AF_Norm': AF_Norm,
        'VSF_Norm': VSF_Norm,
        'MF_Norm': MF_Norm,
        # 'NDVI': NDVI,
    }

    return indexes


SPECIALIZED_BANDS = {
    'ASI',
    'MF',
    # 'VDF',
    # 'SDF',
    'AF',
    'EBSI',
    'MNDWI',
    'MBI',
    'BSI',
    'BSI_Norm',
    'EVI',

    'EMBI',

    'SSF_Norm',
    'AF_Norm',
    'VSF_Norm',
    'MF_Norm',
}


def specialized_index_bands2(delayed=None):
    r"""
    Ported from code from by (Yongquan Zhao on 26 April 2017)

    References:
        https://mail.google.com/mail/u/1/#chat/space/AAAAE5jpxTc

    Ignore:
        DVC_DPATH=$HOME/data/dvc-repos/smart_watch_dvc
        jq '.images[0].id' $DVC_DPATH/drop1-S2-L8-aligned/data.kwcoco.json
        kwcoco subset --src $DVC_DPATH/drop1-S2-L8-aligned/data.kwcoco.json --gids=2, --dst=./one_image_data/data.kwcoco.json --copy_assets=True

    Example:
        >>> # xdoctest: +REQUIRES(env:DVC_DPATH)
        >>> from geowatch.utils.util_bands import *  # NOQA
        >>> from geowatch.utils.util_data import find_dvc_dpath
        >>> import kwcoco
        >>> dvc_dpath = find_dvc_dpath()
        >>> coco_fpath = dvc_dpath / 'Aligned-Drop4-2022-07-28-c20-TA1-S2-L8-ACC/data_vali.kwcoco.json'
        >>> dset = kwcoco.CocoDataset(coco_fpath)
        >>> #from geowatch.utils import kwcoco_extensions
        >>> gid = dset.images().compress([s == 'L8' for s in dset.images().get('sensor_coarse')]).objs[200]['id']
        >>> #gid = ub.peek(dset.index.imgs.keys())
        >>> coco_img = dset.coco_image(gid)
        >>> #print('coco_img.channels = {!r}'.format(coco_img.channels))
        >>> delayed = coco_img.imdelay(space='video', nodata_method='float')
        >>> symbolic = False
        >>> indexes = specialized_index_bands2(delayed)
        >>> import kwarray
        >>> #print(ub.urepr(ub.map_vals(kwarray.stats_dict, indexes), nl=1))
        >>> #import pandas as pd
        >>> #print(pd.DataFrame(ub.map_vals(kwarray.stats_dict, indexes)))
        >>> # xdoctest: +REQUIRES(--show)
        >>> import kwplot
        >>> import kwimage
        >>> kwplot.autompl()
        >>> pnum_ = kwplot.PlotNums(nSubplots=len(indexes) + 1)
        >>> kwplot.figure(fnum=3)
        >>> kwplot.imshow(value, title=key, pnum=pnum_(), cmap=None if key == 'MaskValid' else 'viridis', data_colorbar=True)
        >>> rgb = delayed.take_channels('red|green|blue').finalize()
        >>> rgb_canvas = kwimage.normalize_intensity(rgb)
        >>> kwplot.imshow(rgb_canvas, title='rgb', pnum=pnum_())
        >>> indexes['MaskValid'] = indexes['MaskValid'].astype(np.float32)
        >>> for key, value in indexes.items():
        >>>     value = value.astype(np.float32)
        >>>     #value = kwimage.normalize(value.astype(np.float32))
        >>>     kwplot.imshow(value, title=key, pnum=pnum_(), cmap=None if key == 'MaskValid' else 'viridis', data_colorbar=True)

    Ignore:
        >>> # xdoctest: +SKIP("Something is wrong with grabbing L8 images")
        >>> from geowatch.utils.util_bands import *  # NOQA
        >>> import geowatch
        >>> dset = geowatch.demo.demo_smart_raw_kwcoco()
        >>> coco_img = [img for img in dset.images().coco_images if img.get('sensor_coarse', None) == 'L8'][0]
        >>> delayed = coco_img.imdelay().crop((slice(4000, 5000), slice(4000, 5000)))
        >>> indexes = specialized_index_bands2(delayed)
        >>> # xdoctest: +REQUIRES(--show)
        >>> import kwplot
        >>> import kwimage
        >>> kwplot.autompl()
        >>> pnum_ = kwplot.PlotNums(nSubplots=len(indexes))
        >>> kwplot.figure(fnum=3)
        >>> indexes['MaskValid'] = indexes['MaskValid'].astype(np.float32)
        >>> for key, value in indexes.items():
        >>>     value = value.astype(np.float32)
        >>>     #value = kwimage.normalize(value.astype(np.float32))
        >>>     kwplot.imshow(value, title=key, pnum=pnum_(), cmap=None if key == 'MaskValid' else 'viridis', data_colorbar=True)


    Ignore:
        >>> delayed = coco_img.imdelay()
        >>> rgbir123 = delayed.take_channels('blue|green|red|swir16|swir22|nir')
        >>> chw = rgbir123.finalize().transpose(2, 0, 1).astype(np.float32)
        >>> Blue, Green, Red, SWIR1, SWIR2, NIR = chw
        >>> df = pd.DataFrame(chw.reshape(6, -1).T, columns=['blue', 'red', 'green', 'swir1', 'swir2', 'nir'])
        >>> df = df.melt()
        >>> import kwplot
        >>> sns = kwplot.autosns()
        >>> palette = {
        >>>     'red': 'red', 'blue': 'blue', 'green': 'green',
        >>>     'swir1': 'purple', 'swir2': 'orange', 'nir': 'pink',
        >>> }
        >>> kwplot.figure(fnum=2)
        >>> sns.histplot(data=df, x='value', hue='variable', palette=palette)

        >>> kwplot.figure(fnum=4)
        >>> raw = {
        >>>     'Red': Red,
        >>>     'Blue': Blue,
        >>>     'Green': Green,
        >>>     'NIR': NIR,
        >>>     'SWIR1': SWIR1,
        >>>     'SWIR2': SWIR2,
        >>> }
        >>> pnum_ = kwplot.PlotNums(nSubplots=len(raw))
        >>> for key, value in raw.items():
        >>>     #value = kwimage.normalize(value.astype(np.float32))
        >>>     value = value.astype(np.float32)
        >>>     kwplot.imshow(value, title=key, pnum=pnum_(), cmap='viridis', data_colorbar=True)
    """
    import numpy as np
    # Raw bands
    # if symbolic:
    #     # Sympy can help explore different forms of these equations.
    #     import sympy as sym
    #     Blue, Green, Red, SWIR1, SWIR2, NIR = sym.symbols(
    #         'Blue, Green, Red, SWIR1, SWIR2, NIR')
    # else:
    rgbir123 = delayed.take_channels('blue|green|red|swir16|swir22|nir')
    chw = rgbir123.finalize().transpose(2, 0, 1).astype(np.float32)

    # Artificial Surface Index (ASI) is designed based the surface reflectance imagery of Landsat 8.
    # The data value range of Landsat surface reflectance [0, 1] is
    # transformed to [0, 1*Scale].
    # Scale = 10000
    Scale = 15000

    hack_norm = 0
    if hack_norm:
        # How do we apply this to S2?
        import kwimage
        chw = kwimage.normalize_intensity(chw)
        Scale = 1

    Blue, Green, Red, SWIR1, SWIR2, NIR = chw

    def hist_cut(band, fill_value=0, k=3, minmax='std'):
        if minmax == 'std':
            mean = band.mean()
            std = band.std()
            low_val = (mean - k * std)  # Corrected.
            high_val = (mean + k * std)
        else:
            low_val, high_val = minmax
        band = band.clip(low_val, high_val)
        return band

    def minmax_norm(band, mask):
        max_val = band[mask].max()
        min_val = band[mask].min()
        extent = max_val - min_val
        if extent != 0:
            shifted = band - min_val
            scaled = shifted / extent
            band[mask] = scaled[mask]
        return band

    fillV = 0

    # Surface reflectance should be within [0, 1*Scale]
    max_vals = np.maximum.reduce([Blue, Green, Red, NIR, SWIR1, SWIR2])
    min_vals = np.minimum.reduce([Blue, Green, Red, NIR, SWIR1, SWIR2])
    MaskValid = (0 < min_vals) & (max_vals < Scale)
    # MaskValid = MaskValid.astype(np.uint8)

    # Artificial surface Factor (AF).
    AF = (NIR - Blue) / (NIR + Blue)
    AF = hist_cut(AF, fillV, 6, [-1, 1])
    AF_Norm = minmax_norm(AF, MaskValid)

    # Vegetation Suppressing Factor (VSF).
    # Modified Soil Adjusted Vegetation Index (MSAVI).
    MSAVI = (2 * NIR + 1 * Scale -
             ((2 * NIR + 1 * Scale)**2 - 8 * (NIR - Red)) ** 0.5) / 2
    MSAVI = hist_cut(MSAVI, fillV, 6, [-1, 1])
    NDVI = (NIR - Red) / (NIR + Red)
    NDVI = hist_cut(NDVI, fillV, 6, [-1, 1])
    VSF = 1 - MSAVI * NDVI
    VSF_Norm = minmax_norm(VSF, MaskValid)

    # Soil Suppressing Factor (SSF).
    # Derive the Modified Bare soil Index (MBI).
    MBI = (SWIR1 - SWIR2 - NIR) / (SWIR1 + SWIR2 + NIR) + 0.5
    MBI = hist_cut(MBI, fillV, 6, [-0.5, 1.5])
    # Deriving Enhanced-MBI based on MBI and MNDWI.
    MNDWI = (Green - SWIR1) / (Green + SWIR1)
    MNDWI = hist_cut(MNDWI, fillV, 6, [-1, 1])
    EMBI = ((MBI + 0.5) - (MNDWI + 1)) / ((MBI + 0.5) + (MNDWI + 1))
    EMBI = hist_cut(EMBI, fillV, 6, [-1, 1])
    # Derive SSF.
    SSF = (1 - EMBI)
    SSF_Norm = minmax_norm(SSF, MaskValid)

    # Modulation Factor (MF).
    MF = (Blue + Green - NIR - SWIR1) / (Blue + Green + NIR + SWIR1)
    MF = hist_cut(MF, fillV, 6, [-1, 1])
    MF_Norm = minmax_norm(MF, MaskValid)

    # Derive Artificial Surface Index (ASI).
    ASI = AF_Norm * SSF_Norm * VSF_Norm * MF_Norm
    ASI = hist_cut(ASI, fillV, 6, [0, 1])
    ASI = ASI * MaskValid

    indexes = {
        'MaskValid': MaskValid,
        'ASI': ASI,

        'MF': MF,
        'MF_Norm': MF_Norm,

        'AF': AF,
        'AF_Norm': AF_Norm,

        'SSF_Norm': SSF_Norm,
        'VSF_Norm': VSF_Norm,

        'MSAVI': MSAVI,
        'NDVI': NDVI,
        'MNDWI': MNDWI,
        'MBI': MBI,

        'EMBI': EMBI,
    }

    return indexes
