import ubelt as ub
from os.path import join


def grab_landsat_item():
    """
    Download and cache all items in a given landsat scene.

    TODO:
        - [ ] parametarize scene name / identifier
    """
    scene_name = 'LC08_L1TP_037029_20130602_20170310_01_T1'
    scene_path = join('LC08', '01', '037', '029', scene_name)

    if False:
        # overkill?
        from watch.gis.geotiff import parse_landsat_scene_name
        ls_meta = parse_landsat_scene_name(scene_name)
        sat_code = ls_meta['sat_code']
    else:
        # Equivalent to above, but more fragile and less dependencies
        sat_code = scene_name[2:4]

    uri_prefix = 'http://storage.googleapis.com/gcp-public-data-landsat'

    # Each satellite has a different set of files that it will produce
    sat_code_to_suffixes = {
        '01': ['B1.TIF', 'B2.TIF', 'B3.TIF', 'B4.TIF', 'B5.TIF', 'B6.TIF', 'B7.TIF', 'MTL.txt'],
        '02': ['B1.TIF', 'B2.TIF', 'B3.TIF', 'B4.TIF', 'B5.TIF', 'B6.TIF', 'B7.TIF', 'MTL.txt'],
        '03': ['B1.TIF', 'B2.TIF', 'B3.TIF', 'B4.TIF', 'B5.TIF', 'B6.TIF', 'B7.TIF', 'MTL.txt'],
        '04': ['B1.TIF', 'B2.TIF', 'B3.TIF', 'B4.TIF', 'B5.TIF', 'B6.TIF', 'B7.TIF', 'MTL.txt'],
        '05': ['B1.TIF', 'B2.TIF', 'B3.TIF', 'B4.TIF', 'B5.TIF', 'B6.TIF', 'B7.TIF', 'BQA.TIF', 'MTL.txt'],
        '07': ['B1.TIF', 'B2.TIF', 'B3.TIF', 'B4.TIF', 'B5.TIF', 'B6_VCID_1.TIF', 'B6_VCID_2.TIF', 'B7.TIF', 'B8.TIF', 'BQA.TIF', 'MTL.txt'],
        '08': ['B1.TIF', 'B2.TIF', 'B3.TIF', 'B4.TIF', 'B5.TIF', 'B6.TIF', 'B7.TIF', 'B8.TIF', 'B9.TIF', 'B10.TIF', 'B11.TIF', 'BQA.TIF', 'MTL.txt']
    }

    item_suffixes = sat_code_to_suffixes[sat_code]

    # By default cache to the $XDG_CACHE_HOME/smart_watch
    dset_dpath = ub.ensure_app_cache_dir('smart_watch')

    # Cache the scene using the same path used by google cloud storage
    scene_dpath = ub.ensuredir((dset_dpath, scene_path))

    item_fpaths = []
    for suffix in item_suffixes:
        fname = '{}_{}'.format(scene_name, suffix)
        uri_suffix = join(scene_path, fname)
        item_uri = join(uri_prefix, uri_suffix)
        fpath = ub.grabdata(item_uri, dpath=scene_dpath)
        item_fpaths.append(fpath)

    return item_fpaths
