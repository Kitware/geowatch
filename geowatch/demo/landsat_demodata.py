import ubelt as ub
from os.path import join


def grab_landsat_product(product_id=None, demo_index=0):
    """
    Download and cache all items for a landsat product.

    Args:
        product_id (str, default=None):
            The product id to download (currently NotImplemented).
            If unspecified, an arbitrary scene is returned.

        demo_index (int):
            hack, can be 0, 1, or 2. Regions 1 and 2 should overlap.

    Returns:
        Dict[str, object]:
            groupings of files associated with this landsat product

    Example:
        >>> # xdoctest: +REQUIRES(--network)
        >>> from geowatch.demo.landsat_demodata import *  # NOQA
        >>> product = grab_landsat_product()
        >>> # xdoctest: +IGNORE_WANT
        >>> print('product = {}'.format(ub.urepr(product, nl=2)))
        product = {
            'bands': [
                '.../LC08/01/037/029/LC08_L1TP_037029_20130602_20170310_01_T1/LC08_L1TP_037029_20130602_20170310_01_T1_B1.TIF',
                '.../LC08/01/037/029/LC08_L1TP_037029_20130602_20170310_01_T1/LC08_L1TP_037029_20130602_20170310_01_T1_B2.TIF',
                '.../LC08/01/037/029/LC08_L1TP_037029_20130602_20170310_01_T1/LC08_L1TP_037029_20130602_20170310_01_T1_B3.TIF',
                '.../LC08/01/037/029/LC08_L1TP_037029_20130602_20170310_01_T1/LC08_L1TP_037029_20130602_20170310_01_T1_B4.TIF',
                '.../LC08/01/037/029/LC08_L1TP_037029_20130602_20170310_01_T1/LC08_L1TP_037029_20130602_20170310_01_T1_B5.TIF',
                '.../LC08/01/037/029/LC08_L1TP_037029_20130602_20170310_01_T1/LC08_L1TP_037029_20130602_20170310_01_T1_B6.TIF',
                '.../LC08/01/037/029/LC08_L1TP_037029_20130602_20170310_01_T1/LC08_L1TP_037029_20130602_20170310_01_T1_B7.TIF',
                '.../LC08/01/037/029/LC08_L1TP_037029_20130602_20170310_01_T1/LC08_L1TP_037029_20130602_20170310_01_T1_B8.TIF',
                '.../LC08/01/037/029/LC08_L1TP_037029_20130602_20170310_01_T1/LC08_L1TP_037029_20130602_20170310_01_T1_B9.TIF',
                '.../LC08/01/037/029/LC08_L1TP_037029_20130602_20170310_01_T1/LC08_L1TP_037029_20130602_20170310_01_T1_B10.TIF',
                '.../LC08/01/037/029/LC08_L1TP_037029_20130602_20170310_01_T1/LC08_L1TP_037029_20130602_20170310_01_T1_B11.TIF',
            ],
            'meta': {
                'bqa': '.../LC08/01/037/029/LC08_L1TP_037029_20130602_20170310_01_T1/LC08_L1TP_037029_20130602_20170310_01_T1_BQA.TIF',
                'mtl': '.../LC08/01/037/029/LC08_L1TP_037029_20130602_20170310_01_T1/LC08_L1TP_037029_20130602_20170310_01_T1_MTL.txt',
            },
        }

    References:
        .. [1] https://github.com/dgketchum/Landsat578#-1
        .. [2] https://tilemill-project.github.io/tilemill/docs/guides/landsat-8-imagery/
        .. [3] https://earth.esa.int/documents/700255/1834061/Landsat+ETM%2B%20Data+Format+Control+Book/4bfb7121-e97d-46ca-8d3f-02f1c6bf309c;jsessionid=F97499AA37A1E9AF0EED42900CF66760?version=1.1

    SeeAlso:
        geowatch.gis.geotiff.parse_landsat_product_id

    Ignore:
        # Use console page for the first data
        https://console.cloud.google.com/storage/browser/gcp-public-data-landsat/LC08/01/037/029/LC08_L1TP_037029_20130602_20170310_01_T1

    TODO:
        - [ ] parametarize scene name / identifier
        - [ ] bundle bands in a single file (gdal VRT?)
        - [X] separate data and metadata files in return structure?
    """
    if product_id is not None:
        raise NotImplementedError('Must use the default scene')

    if demo_index == 0:
        scene_name = 'LC08_L1TP_037029_20130602_20170310_01_T1'
        scene_path = join('LC08', '01', '037', '029', scene_name)
    elif demo_index == 1:
        scene_name = 'LC08_L1TP_187022_20191020_20191030_01_T1'
        scene_path = join('LC08', '01', '187', '022', scene_name)
    elif demo_index == 2:
        scene_name = 'LC08_L1TP_187021_20191020_20191030_01_T1'
        scene_path = join('LC08', '01', '187', '021', scene_name)
    else:
        raise KeyError(demo_index)

    if False:
        # overkill?
        from geowatch.gis.geotiff import parse_landsat_product_id
        ls_meta = parse_landsat_product_id(scene_name)
        sat_code = ls_meta['sat_code']
    else:
        # Equivalent to above, but more fragile and less dependencies
        sat_code = scene_name[2:4]

        try:
            int(sat_code.lstrip('0'))
        except Exception:
            raise AssertionError('scene name does have landsat spec')

    uri_prefix = 'http://storage.googleapis.com/gcp-public-data-landsat'

    # Each satellite has a different set of files that it will produce
    sat_code_to_suffixes = {
        '01': {
            'bands': ['B1.TIF', 'B2.TIF', 'B3.TIF', 'B4.TIF', 'B5.TIF', 'B6.TIF', 'B7.TIF'],
            'meta': {'mtl': 'MTL.txt'}
        },

        '02': {
            'bands': ['B1.TIF', 'B2.TIF', 'B3.TIF', 'B4.TIF', 'B5.TIF', 'B6.TIF', 'B7.TIF'],
            'meta': {'mtl': 'MTL.txt'}
        },

        '03': {
            'bands': ['B1.TIF', 'B2.TIF', 'B3.TIF', 'B4.TIF', 'B5.TIF', 'B6.TIF', 'B7.TIF'],
            'meta': {'mtl': 'MTL.txt', 'bqa': 'BQA.TIF'}
        },

        '04': {
            'bands': ['B1.TIF', 'B2.TIF', 'B3.TIF', 'B4.TIF', 'B5.TIF', 'B6.TIF', 'B7.TIF'],
            'meta': {'mtl': 'MTL.txt', 'bqa': 'BQA.TIF'}
        },

        '05': {
            'bands': ['B1.TIF', 'B2.TIF', 'B3.TIF', 'B4.TIF', 'B5.TIF', 'B6.TIF', 'B7.TIF'],
            'meta': {'mtl': 'MTL.txt', 'bqa': 'BQA.TIF'}
        },

        '07': {
            'bands': ['B1.TIF', 'B2.TIF', 'B3.TIF', 'B4.TIF', 'B5.TIF', 'B6_VCID_1.TIF', 'B6_VCID_2.TIF'],
            'meta': {'mtl': 'MTL.txt', 'bqa': 'BQA.TIF'}
        },

        '08': {
            'bands': ['B1.TIF', 'B2.TIF', 'B3.TIF', 'B4.TIF', 'B5.TIF', 'B6.TIF',
                      'B7.TIF', 'B8.TIF', 'B9.TIF', 'B10.TIF', 'B11.TIF'],
            'meta': {'mtl': 'MTL.txt', 'bqa': 'BQA.TIF'},
        }
    }

    # B6_VCID_1 = Band 6 Visual Channel Identifier (VCID) 1
    # B6_VCID_2 = Band 6 VCID
    # MTL = metadata
    # BQA: quality assurance pseudo-band

    item_suffixes = sat_code_to_suffixes[sat_code]

    # By default cache to the $XDG_CACHE_HOME/geowatch
    dset_dpath = ub.Path.appdir('geowatch/demo/landsat').ensuredir()

    # Cache the scene using the same path used by google cloud storage
    scene_dpath = dset_dpath / scene_path
    scene_dpath.ensuredir()

    product = {
        'bands': [],
        'meta': {},
        'scene_name': scene_name,
    }
    # Download band product-items
    for suffix in item_suffixes['bands']:
        fname = '{}_{}'.format(scene_name, suffix)
        uri_suffix = join(scene_path, fname)
        item_uri = join(uri_prefix, uri_suffix)
        fpath = ub.grabdata(item_uri, dpath=scene_dpath)

        stamp = ub.CacheStamp(fname + '.nodata.stamp', depends=[],
                              dpath=scene_dpath)
        if stamp.expired():
            import rasterio
            # TODO: cache this step
            with rasterio.open(fpath, 'r+') as img:
                if img.nodata is None:
                    img.nodata = 0
            stamp.renew()

        product['bands'].append(fpath)

    # Download meta product-items
    for key, suffix in item_suffixes['meta'].items():
        fname = '{}_{}'.format(scene_name, suffix)
        uri_suffix = join(scene_path, fname)
        item_uri = join(uri_prefix, uri_suffix)
        fpath = ub.grabdata(item_uri, dpath=scene_dpath)
        product['meta'][key] = fpath

    return product
