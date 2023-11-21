#!/usr/bin/env python3
"""
Attempts to register directory of geotiffs into a kwcoco dataset
"""
import scriptconfig as scfg
import ubelt as ub


class KWCocoFromGeotiffConfig(scfg.DataConfig):
    """
    Create a kwcoco manifest of a set of on-disk geotiffs
    """
    __default__ = {
        'geotiff_dpath': scfg.Value(None, help='path containing geotiffs'),
        'relative': scfg.Value(False, help='if true make paths relative'),
        'dst': scfg.Value(None, help='path to write new kwcoco file'),
        'workers': scfg.Value(0, help='number of parallel jobs'),
        'strict': scfg.Value(False, isflag=True, help='it True, will raise an error if any job fails'),
    }


def main(cmdline=1, **kwargs):
    """
    Ignore:
        geotiff_dpath = '/home/joncrall/data/grab_tiles_out/fels'
        dst = '/home/joncrall/data/grab_tiles_out/fels/data.kwcoco.json'
        kwargs = {
            'geotiff_dpath': geotiff_dpath,
            'dst': dst,
        }
        dset1 = kwcoco.CocoDataset(ub.expandpath('$HOME/data/dvc-repos/smart_watch_dvc/drop0/drop0.kwcoco.json'))
        dset2 = kwcoco.CocoDataset(ub.expandpath('$HOME/data/grab_tiles_out/fels/data.kwcoco.json'))
    """
    config = KWCocoFromGeotiffConfig.cli(data=kwargs, cmdline=cmdline)
    geotiff_dpath = config['geotiff_dpath']
    dst = config['dst']

    imgs = find_geotiffs(geotiff_dpath, workers=config['workers'],
                         strict=config['strict'])

    import kwcoco
    dset = kwcoco.CocoDataset()
    for img in imgs:
        dset.add_image(**img)
    return dset
    dset.fpath = dst

    if config['relative']:
        dset.reroot(dset.bundle_dpath, absolute=False)

    dset.dump(dset.fpath, newlines=True)


def filter_band_files(fpaths, band_list, with_tci=True):
    """
    band_list is any subset of util_bands.ALL_BANDS

    with_tci: include true color thumbnail
    """
    from os.path import basename, splitext
    band_names = set(b['name'] for b in band_list)
    if with_tci:
        band_names.add('TCI')
    # use endswith() instead of in
    # to avoid false positives, eg from a tile code in the filename

    def is_band_file(path):
        return any(splitext(basename(path))[0].endswith(b) for b in band_names)
    return list(filter(is_band_file, fpaths))


def ingest_landsat_directory(lc_dpath):
    import geowatch
    from geowatch.utils import util_bands
    from dateutil.parser import isoparse
    from os.path import join, basename, normpath
    import glob
    name = basename(normpath(lc_dpath))
    tiffs = sorted(glob.glob(join(lc_dpath, '*.TIF')))
    if len(tiffs) == 0:
        tiffs = sorted(glob.glob(join(lc_dpath, '**', '*.TIF'), recursive=True))
    baseinfo = geowatch.gis.geotiff.geotiff_filepath_info(name)
    capture_time = isoparse(baseinfo['filename_meta']['acquisition_date']).isoformat()
    sensor_coarse = 'LS'
    if baseinfo['filename_meta']['sensor_code'] == 'C':
        sensor_coarse = 'L8'
    elif baseinfo['filename_meta']['sensor_code'] == 'E':
        sensor_coarse = 'L7'
    # take L8 as the default guess for a mangled name
    tiffs = filter_band_files(tiffs, (util_bands.LANDSAT7 if sensor_coarse == 'L7' else util_bands.LANDSAT8))
    img = make_coco_img_from_auxiliary_geotiffs(tiffs, name)
    img['date_captured'] = capture_time
    img['sensor_coarse'] = sensor_coarse
    return img


def ingest_sentinel2_directory(s2_dpath):
    # Are we in the safedir, the granuledir or some arbitrary dir?
    # Try to use the granuledir as name if available;
    # it's a better unique ID.
    import geowatch
    from geowatch.utils import util_bands
    import datetime as datetime_mod
    from dateutil.parser import isoparse
    from os.path import join, basename, normpath
    import glob
    granules = sorted(glob.glob(join(s2_dpath, 'GRANULE', '*')))
    if len(granules) == 1:
        granule = granules[0]
        tiffs = sorted(glob.glob(join(granule, 'IMG_DATA', '*.jp2')))
        name = basename(normpath(granule))
    else:
        tiffs = sorted(glob.glob(join(s2_dpath, 'GRANULE', '*', 'IMG_DATA', '*.jp2')))
        if len(tiffs) == 0:
            tiffs = sorted(glob.glob(join(s2_dpath, '**', '*.jp2'), recursive=True))
        name = basename(normpath(s2_dpath)).replace('.SAFE', '')
    # Then grab the bands.
    tiffs = filter_band_files(tiffs, util_bands.SENTINEL2)
    img = make_coco_img_from_auxiliary_geotiffs(tiffs, name)

    baseinfo = geowatch.gis.geotiff.geotiff_filepath_info(s2_dpath)
    capture_time = isoparse(baseinfo['filename_meta']['sense_start_time'])
    img['date_captured'] = datetime_mod.datetime.isoformat(capture_time)
    img['sensor_coarse'] = 'S2'
    return img


def make_coco_img_from_geotiff(tiff_fpath, name=None, force_affine=True,
                               with_info=False):
    """
    TODO: move to coco extensions

    Example:
        >>> from geowatch.demo.landsat_demodata import grab_landsat_product  # NOQA
        >>> product = grab_landsat_product()
        >>> tiffs = product['bands'] + [product['meta']['bqa']]
        >>> tiff_fpath = product['bands'][0]
        >>> name = None
        >>> img = make_coco_img_from_geotiff(tiff_fpath)
        >>> print('img = {}'.format(ub.urepr(img, nl=1)))
    """
    import geowatch
    from geowatch.utils import util_bands
    import kwimage
    img = {}
    if name is not None:
        img['name'] = name

    info = geowatch.gis.geotiff.geotiff_metadata(tiff_fpath)
    # only affine transformations are supported in auxiliary channels
    # TODO support RPC
    info.update(**geowatch.gis.geotiff.geotiff_crs_info(tiff_fpath, force_affine=force_affine))

    warp_pxl_from_wld = kwimage.Affine.coerce(info['pxl_to_wld'])
    height, width = info['img_shape']
    file_meta = info['filename_meta']
    channels = file_meta.get('channels', None)

    if channels is None:
        # fix this for known WV channel signature, which isn't obvious from filename
        if file_meta.get('product_guess') == 'worldview':

            from osgeo import gdal
            bands = gdal.Info(tiff_fpath, format='json')['bands']

            # the channel names are the same for all WV, just the center_wavelength is different
            # so we can safely use this info from WV2
            def _code(band_dicts):
                return '|'.join(b.get('common_name', b['name']) for b in band_dicts)

            if len(bands) == 1:
                channels = _code(util_bands.WORLDVIEW2_PAN)
            elif len(bands) == 4:
                channels = _code(util_bands.WORLDVIEW2_MS4)
            elif len(bands) == 8:
                channels = _code(util_bands.WORLDVIEW2_MS8)
            else:
                raise Exception('unknown channel signature for WV')
        else:
            raise Exception('must be able to introspect channels')

    wld_crs_info = ub.dict_diff(info['wld_crs_info'], {'type'})
    utm_crs_info = ub.dict_diff(info['utm_crs_info'], {'type'})

    img.update({
        'file_name': tiff_fpath,
        'width': width,
        'height': height,
        'channels': channels,
        'num_bands': info['num_bands'],
        'approx_meter_gsd': info['approx_meter_gsd'],
        'warp_pxl_to_wld': warp_pxl_from_wld,
        'geos_corners': info['geos_corners'],
        'wld_crs_info': wld_crs_info,
        'utm_crs_info': utm_crs_info,
    })
    if with_info:
        img['info'] = info
    return img


def make_coco_img_from_auxiliary_geotiffs(tiffs, name):
    """
    TODO: move to coco extensions

    Example:
        >>> from geowatch.demo.landsat_demodata import grab_landsat_product  # NOQA
        >>> product = grab_landsat_product()
        >>> tiffs = product['bands'] + [product['meta']['bqa']]
        >>> name = product['scene_name']
        >>> img = make_coco_img_from_auxiliary_geotiffs(tiffs, name)
        >>> print('img = {}'.format(ub.urepr(img, nl=-1, sort=0)))
    """
    auxiliary = []
    for fpath in tiffs:
        aux = make_coco_img_from_geotiff(fpath, force_affine=True)
        auxiliary.append(aux)
    return make_coco_img_from_auxiliary_dicts(auxiliary, name)


def make_coco_img_from_auxiliary_dicts(auxiliary, name):
    img = {
        'name': name,
        'file_name': None,
    }
    # Choose a base image canvas and the relationship between auxiliary images
    idx = ub.argmax(auxiliary, lambda x: (x['width'] * x['height']))
    base = auxiliary[idx]
    warp_wld_from_img = base['warp_pxl_to_wld']
    warp_img_from_wld = warp_wld_from_img.inv()
    img['warp_img_to_wld'] = warp_wld_from_img.concise()
    img.update(ub.dict_isect(base, {'geos_corners', 'wld_crs_info', 'utm_crs_info'}))

    for aux in auxiliary:
        aux.pop('geos_corners')
        aux.pop('utm_crs_info')
        aux.pop('wld_crs_info')
        warp_wld_from_aux = aux.pop('warp_pxl_to_wld')
        warp_img_from_aux = warp_img_from_wld @ warp_wld_from_aux
        aux['warp_aux_to_img'] = warp_img_from_aux.concise()

    img['width'] = base['width']
    img['height'] = base['height']
    img['auxiliary'] = auxiliary
    return img


def find_geotiffs(geotiff_dpath, workers=0, strict=False):
    """
    Search a directory for any geotiffs and return a set of kwcoco-style image
    dictionaries detailing the results.

    Args:
        geotiff_dpath (str): directory to search

    Returns:
        List[Dict]: a list of kwcoco-style image dictionaries

    geotiff_dpath = '/home/joncrall/data/grab_tiles_out/fels'
    """
    import os
    import geowatch
    from os.path import basename
    dpath_list = list(geowatch.gis.geotiff.walk_geotiff_products(geotiff_dpath))

    print(f'Found candidate {len(dpath_list)} geotiff products')

    jobs = ub.JobPool(mode='thread', max_workers=workers)

    loose_files = []
    unknown_products = []
    for dpath in ub.ProgIter(dpath_list, desc='submit geotiffs jobs'):
        if os.path.isfile(dpath):
            # if we dont't get a directory we have a loose geotiff file
            loose_files.append(dpath)
        else:
            dname = basename(dpath)
            if dname.startswith(('LC', 'LE07')):
                lc_dpath = dpath
                job = jobs.submit(ingest_landsat_directory, lc_dpath)
                job.dpath = dpath
            elif dname.startswith('S2'):
                s2_dpath = dpath
                # FIXME: undefined name
                job = jobs.submit(ingest_sentinel2_directory, s2_dpath)
                job.dpath = dpath
            else:
                msg = ('unknown dpath = {!r}'.format(dpath))
                print(msg)
                unknown_products.append(msg)

    if unknown_products:
        print('\n'.join(unknown_products))

    imgs = []
    errors = []
    for job in ub.ProgIter(jobs, desc='collect results'):
        try:
            img = job.result()
        except Exception as ex:
            if strict:
                raise
            err = (job, job.dpath, ex)
            print('err = {!r}'.format(err))
            errors.append(err)
        else:
            imgs.append(img)

    if loose_files:
        # Handle loose files (try grouping them by spacetime)
        groups = ub.ddict(list)

        # jobs = ub.JobPool(mode='thread', max_workers=workers)
        for fpath in ub.ProgIter(loose_files, desc='process loose files'):
            info = geowatch.gis.geotiff.geotiff_filepath_info(fpath, fast=True)
            file_meta = info['filename_meta']
            file_meta.get('tile_number', None)
            date_captured = next(iter(ub.dict_isect(file_meta, ['sense_start_time', 'acquisition_date']).values()), None)
            tile_num = next(iter(ub.dict_isect(file_meta, ['tile_number']).values()), None)
            groupid = (file_meta['product_guess'], tile_num, date_captured)
            img = make_coco_img_from_geotiff(fpath, with_info=True)
            info = img.pop('info')
            img['date_captured'] = info['filename_meta']['date_captured']
            if 'landsat' in info['filename_meta']['product_guess']:
                sensor_coarse = 'LS'
                if info['filename_meta']['sensor_code'] == 'C':
                    sensor_coarse = 'L8'
                elif info['filename_meta']['sensor_code'] == 'E':
                    sensor_coarse = 'L7'
            elif 'sentinel' in info['filename_meta']['product_guess']:
                sensor_coarse = 'S2'
            else:
                sensor_coarse = 'null'
            img['sensor_coarse'] = sensor_coarse
            groups[groupid].append(img)

        for groupid, group in groups.items():
            img = make_coco_img_from_auxiliary_dicts(group, name=groupid)
            img['date_captured'] = group[0]['date_captured']
            img['sensor_coarse'] = group[0]['sensor_coarse']
            imgs.append(img)

    print('Got {} errors'.format(len(errors)))
    print('Found {} images to add'.format(len(imgs)))
    return imgs


__config__ = KWCocoFromGeotiffConfig


if __name__ == '__main__':
    """
    CommandLine:
        python -m geowatch.cli.geotiffs_to_kwcoco.py

    CommandLine:
        python -m geowatch.cli.coco_extract_geo_bounds \
          --src $HOME/data/dvc-repos/smart_watch_dvc/drop0/drop0.kwcoco.json \
          --breakup_times=True \
          --dst $HOME/data/grab_tiles_out/regions.geojson.json

        source ~/internal/safe/secrets
        python ~/code/watch/scripts/grab_tiles_demo.py \
            --regions $HOME/data/grab_tiles_out/regions.geojson.json \
            --out_dpath $HOME/data/grab_tiles_out \
            --rgdc_username=$WATCH_RGD_USERNAME \
            --rgdc_password=$WATCH_RGD_PASSWORD \
            --backend rgdc

        python ~/code/watch/scripts/grab_tiles_demo.py \
            --regions $HOME/data/grab_tiles_out/regions.geojson.json \
            --out_dpath $HOME/data/grab_tiles_out \
            --backend fels --profile

        python -m geowatch.cli.geotiffs_to_kwcoco.py \
            --geotiff_dpath ~/data/grab_tiles_out/fels \
            --dst $HOME/data/grab_tiles_out/fels/data.kwcoco.json --profile

        python -m geowatch.cli.geotiffs_to_kwcoco.py \
            --geotiff_dpath ~/data/dvc-repos/smart_watch_dvc/unannotated/AE-Dubai-0001 \
            --dst ~/data/dvc-repos/smart_watch_dvc/unannotated/dubai-msi.kwcoco.json

        cat ~/data/dvc-repos/smart_watch_dvc/unannotated/dubai-msi.kwcoco.json

        python -m geowatch.cli.geotiffs_to_kwcoco.py \
            --geotiff_dpath ~/data/dvc-repos/smart_watch_dvc/unannotated/KR-Pyeongchang-S2 \
            --dst ~/data/dvc-repos/smart_watch_dvc/unannotated/korea-msi.kwcoco.json

        python -m -m geowatch.cli.geotiffs_to_kwcoco.py \
            --geotiff_dpath ~/data/dvc-repos/smart_watch_dvc/unannotated/US-Waynesboro-0001 \
            --dst ~/data/dvc-repos/smart_watch_dvc/unannotated/waynesboro-msi.kwcoco.json
    """
    main()
