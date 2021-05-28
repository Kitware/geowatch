"""
Attempts to register directory of geotiffs into a kwcoco dataset
"""

from dateutil.parser import isoparse
from kwcoco.util import util_futures
from kwimage.transform import Affine
from os.path import join, basename, normpath
import datetime
import glob
import kwcoco
import scriptconfig as scfg
import ubelt as ub
import watch

from watch.utils import util_bands


class KWCocoFromGeotiffConfig(scfg.Config):
    default = {
        'geotiff_dpath': scfg.Value(None, help='path containing geotiffs'),
        'dst': scfg.Value(None, help='path to write new kwcoco file')
    }


def ingest_landsat_directory(lc_dpath):
    name = basename(normpath(lc_dpath))
    tiffs = sorted(glob.glob(join(lc_dpath, '*.TIF')))
    band_names = [b['name'] for b in (util_bands.BANDS_LANDSAT7 +
                                      util_bands.BANDS_LANDSAT8)]
    tiffs = [t for t in tiffs if any(b in t for b in band_names)]
    img = make_coco_img_from_auxiliary_geotiffs(tiffs, name)
    baseinfo = watch.gis.geotiff.geotiff_filepath_info(name)
    capture_time = isoparse(baseinfo['filename_meta']['acquisition_date'])
    img['date_captured'] = datetime.datetime.isoformat(capture_time)
    if name.startswith('LC'):
        img['sensor_coarse'] = 'L8'
    elif name.startswith('LE'):
        img['sensor_coarse'] = 'L7'
    else:
        img['sensor_coarse'] = 'LS'
    return img


def ingest_sentinal2_directory(s2_dpath):
    name = basename(normpath(s2_dpath)).rstrip('.SAFE')
    tiffs = sorted(glob.glob(join(s2_dpath, 'GRANULE', '*', 'IMG_DATA', '*.jp2')))
    band_names = [b['name'] for b in util_bands.BANDS_SENTINEL2]
    tiffs = [t for t in tiffs if any(b in t for b in band_names)]
    img = make_coco_img_from_auxiliary_geotiffs(tiffs, name)

    baseinfo = watch.gis.geotiff.geotiff_filepath_info(name)
    capture_time = isoparse(baseinfo['filename_meta']['sense_start_time'])
    img['date_captured'] = datetime.datetime.isoformat(capture_time)
    img['sensor_coarse'] = 'S2'
    return img


def make_coco_img_from_geotiff(tiff_fpath, name=None):
    """
    TODO: move to coco extensions

    Example:
        >>> from watch.demo.landsat_demodata import grab_landsat_product  # NOQA
        >>> product = grab_landsat_product()
        >>> tiffs = product['bands'] + [product['meta']['bqa']]
        >>> tiff_fpath = product['bands'][0]
        >>> name = None
        >>> img = make_coco_img_from_geotiff(tiff_fpath)
        >>> print('img = {}'.format(ub.repr2(img, nl=1)))
    """
    img = {}
    if name is not None:
        img['name'] = name

    info = watch.gis.geotiff.geotiff_metadata(tiff_fpath)

    warp_pxl_to_wld = Affine.coerce(info['pxl_to_wld'])
    height, width = info['img_shape']
    file_meta = info['filename_meta']
    # print('file_meta = {!r}'.format(file_meta))
    channels = file_meta.get('channels', None)

    if channels is None:
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
        'warp_pxl_to_wld': warp_pxl_to_wld,
        'utm_corners': info['utm_corners'].data.tolist(),
        'wld_crs_info': wld_crs_info,
        'utm_crs_info': utm_crs_info,
    })

    return img


def Affine_concise(aff):
    """
    TODO: add to kwimage.Affine
    """
    import numpy as np
    params = aff.decompose()
    params['type'] = 'affine'
    if np.allclose(params['offset'], (0, 0)):
        params.pop('offset')
    if np.allclose(params['scale'], (1, 1)):
        params.pop('scale')
    if np.allclose(params['shear'], 0):
        params.pop('shear')
    if np.allclose(params['theta'], 0):
        params.pop('theta')
    return params


def make_coco_img_from_auxiliary_geotiffs(tiffs, name):
    """
    TODO: move to coco extensions

    Example:
        >>> from watch.demo.landsat_demodata import grab_landsat_product  # NOQA
        >>> product = grab_landsat_product()
        >>> tiffs = product['bands'] + [product['meta']['bqa']]
        >>> name = product['scene_name']
        >>> img = make_coco_img_from_auxiliary_geotiffs(tiffs, name)
        >>> print('img = {}'.format(ub.repr2(img, nl=-1, sort=0)))
    """
    img = {
        'name': name,
        'file_name': None,
    }
    auxiliary = []

    for fpath in tiffs:
        aux = make_coco_img_from_geotiff(fpath)
        auxiliary.append(aux)

    # Choose a base image canvas and the relationship between auxiliary images
    idx = ub.argmax(auxiliary, lambda x: (x['width'] * x['height']))
    base = auxiliary[idx]
    warp_img_to_wld = base['warp_pxl_to_wld']
    warp_wld_to_img = warp_img_to_wld.inv()
    img['warp_img_to_wld'] = Affine_concise(warp_img_to_wld)
    img.update(ub.dict_isect(base, {'utm_corners', 'wld_crs_info', 'utm_crs_info'}))

    # img[' = aux.pop('utm_corners')
    # aux.pop('utm_crs_info')
    # aux.pop('wld_crs_info')

    for aux in auxiliary:
        aux.pop('utm_corners')
        aux.pop('utm_crs_info')
        aux.pop('wld_crs_info')
        warp_aux_to_img = warp_wld_to_img @ aux.pop('warp_pxl_to_wld')
        aux['warp_aux_to_img'] = Affine_concise(warp_aux_to_img)

    img['width'] = base['width']
    img['height'] = base['height']
    img['auxiliary'] = auxiliary
    return img


def find_geotiffs(geotiff_dpath):
    """
    geotiff_dpath = '/home/joncrall/data/grab_tiles_out/fels'
    """
    dpath_list = list(watch.gis.geotiff.walk_geotiff_products(geotiff_dpath))

    jobs = util_futures.JobPool(mode='thread', max_workers=14)

    unknown_products = []
    for dpath in ub.ProgIter(dpath_list, desc='submit geotiffs jobs'):
        dname = basename(dpath)
        if dname.startswith(('LC', 'LE07')):
            lc_dpath = dpath
            job = jobs.submit(ingest_landsat_directory, lc_dpath)
            job.dpath = dpath
        elif dname.startswith('S2'):
            s2_dpath = dpath
            job = jobs.submit(ingest_sentinal2_directory, s2_dpath)
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
            errors.append((job, job.dpath, ex))
        else:
            imgs.append(img)

    dset = kwcoco.CocoDataset()

    for img in imgs:
        dset.add_image(**img)

    return dset


def main(**kwargs):
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
    config = KWCocoFromGeotiffConfig(default=kwargs, cmdline=True)
    geotiff_dpath = config['geotiff_dpath']
    dst = config['dst']

    dset = find_geotiffs(geotiff_dpath)

    dset.fpath = dst
    dset.dump(dset.fpath, newlines=True)


if __name__ == '__main__':
    """
    CommandLine:
        python ~/code/watch/scripts/geotiffs_to_kwcoco.py

    CommandLine:
        python ~/code/watch/scripts/coco_extract_geo_bounds.py \
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

        python ~/code/watch/scripts/geotiffs_to_kwcoco.py \
            --geotiff_dpath ~/data/grab_tiles_out/fels \
            --dst $HOME/data/grab_tiles_out/fels/data.kwcoco.json

    """
    main()
