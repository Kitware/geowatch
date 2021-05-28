'''
Collected information about satellite bands from https://github.com/stac-extensions/eo/

This should be mostly independent of the data source used.
(eg Google Cloud vs element84 AWS, Collection-1 vs Collection-2)
No guarantees that every value will be indentical, but close enough for heuristics.

Each constant is a list of 'eo:bands' Band Object dicts.
They are sorted in the same order in which they appear if the bands come stacked in a single image,
or in lexicographic order if the bands come in separate images.

Coverage of the catalogs is inconsistent. Where necessary, info has been filled in by hand.
'''

def dicts_contain(d_list, dsub_list):
    contains = lambda ds: all(ds[0][k] == ds[1][k] for k in ds[1])
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
    >>> from watch.datacube.reflectance.bands import *
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
    >>> from watch.datacube.reflectance.bands import *
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
    >>> from watch.datacube.reflectance.bands import *
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
    >>> wv03 = [p for p in props if p['mission'] == 'WV02']
    >>> assert np.all(np.unique([p['instruments'] for p in wv03]) == ['panchromatic', 'vis-multi'])
    >>> assert np.all(np.unique([len(p['eo:bands']) for p in wv03]) == [1, 8])
    >>> wv03_pan = [p for p in wv03 if p['instruments'] == ['panchromatic']][0]['eo:bands']
    >>> 
    >>> # not sure if this must be true, but it is
    >>> assert wv02_pan == wv03_pan
    >>> assert wv02_ms8 == wv03_ms8
    >>> 
    >>> from watch.datacube.reflectance.bands import *
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

