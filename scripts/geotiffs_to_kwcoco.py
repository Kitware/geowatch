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
import os
import scriptconfig as scfg
import ubelt as ub
import watch


class GeotiffToKwcocoConfig(scfg.Config):
    default = {
        'geotiff_dpath': scfg.Value(None, help='path containing geotiffs'),
        'dst': scfg.Value(None, help='path to write new kwcoco file')
    }


def fake_band_info():
    """
    References:
        https://gis.stackexchange.com/questions/290796/how-to-edit-the-metadata-for-individual-bands-of-a-multiband-raster-preferably
        https://gisgeography.com/sentinel-2-bands-combinations/
        https://earth.esa.int/eogateway/missions/worldview-3
        https://www.usgs.gov/faqs/what-are-band-designations-landsat-satellites?qt-news_science_products=0#qt-news_science_products

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
    """


def ingest_landsat_directory(lc_dpath):
    tiffs = sorted(glob.glob(join(lc_dpath, '*.TIF')))
    name = basename(normpath(lc_dpath))
    img = make_coco_img_from_auxiliary_geotiffs(tiffs, name)
    baseinfo = watch.gis.geotiff.geotiff_filepath_info(name)
    capture_time = isoparse(baseinfo['filename_meta']['acquisition_date'])
    img['date_captured'] = datetime.datetime.isoformat(capture_time)
    img['sensor_coarse'] = 'LC'
    return img


def ingest_sentinal2_directory(s2_dpath):
    name = basename(normpath(s2_dpath)).rstrip('.SAFE')
    tiffs = sorted(glob.glob(join(s2_dpath, 'GRANULE', '*', 'IMG_DATA', '*.jp2')))
    img = make_coco_img_from_auxiliary_geotiffs(tiffs, name)

    baseinfo = watch.gis.geotiff.geotiff_filepath_info(name)
    capture_time = isoparse(baseinfo['filename_meta']['sense_start_time'])
    img['date_captured'] = datetime.datetime.isoformat(capture_time)
    img['sensor_coarse'] = 'S2'
    return img


def make_coco_img_from_auxiliary_geotiffs(tiffs, name):
    img = {
        'name': name,
        'file_name': None,
    }
    auxiliary = []

    def shrink_json(aff):
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

    for fpath in tiffs:
        info = watch.gis.geotiff.geotiff_metadata(fpath)
        warp_aux_to_wld = Affine.coerce(info['pxl_to_wld'])
        height, width = info['img_shape']
        file_meta = info['filename_meta']
        # print('file_meta = {!r}'.format(file_meta))
        if 'suffix' in file_meta:
            channel = file_meta['suffix']
            if channel == 'TCI':
                channel = 'r|g|b'
        elif 'discriminator' in file_meta:
            channel = file_meta['discriminator']
        else:
            raise Exception
        aux = {
            'file_name': fpath,
            'width': width,
            'height': height,
            'channels': channel,
            'num_bands': info['num_bands'],
            'approx_meter_gsd': info['approx_meter_gsd'],
            'warp_aux_to_wld': warp_aux_to_wld,
        }
        auxiliary.append(aux)

    # Choose a base image canvas and the relationship between auxiliary images
    idx = ub.argmax(auxiliary, lambda x: (x['width'] * x['height']))
    base = auxiliary[idx]
    warp_wld_to_img = base['warp_aux_to_wld'].inv()

    for aux in auxiliary:
        warp_aux_to_img = warp_wld_to_img @ aux['warp_aux_to_wld']
        aux['warp_aux_to_img'] = warp_aux_to_img

    for aux in auxiliary:
        aux.pop('warp_aux_to_wld')
        aux['warp_aux_to_img'] = shrink_json(aux['warp_aux_to_img'])

    img['width'] = base['width']
    img['height'] = base['height']
    img['auxiliary'] = auxiliary
    return img


def walk_geotiff_products(dpath, recursive=True):
    """
    Walks a file path and returns directories and files that look
    like standalone geotiff products.
    """
    # blocklist = set()
    GEOTIFF_EXTENSIONS = ('.vrt', '.tiff', '.tif', '.jp2')

    for r, ds, fs in os.walk(dpath):
        handled = []
        for didx, dname in enumerate(ds):
            if dname.startswith('LE07'):
                dpath = join(r, dname)
                handled.append(didx)
                yield dpath
            elif dname.startswith('LC08_'):
                dpath = join(r, dname)
                handled.append(didx)
                yield dpath
            elif dname.startswith(('S2A_', 'S2B_')):
                handled.append(didx)
                dpath = join(r, dname)
                yield dpath
            else:
                pass
        for didx in reversed(handled):
            del ds[didx]

        for fname in fs:
            if fname.lower().endswith(GEOTIFF_EXTENSIONS):
                fpath = join(r, fname)
                yield fpath

        if not recursive:
            break


def find_geotiffs(geotiff_dpath):
    jobs = util_futures.JobPool(mode='thread', max_workers=14)

    dpath_list = list(walk_geotiff_products(geotiff_dpath))

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
        dst = join(geotiff_dpath, '/home/joncrall/data/grab_tiles_out/fels/data.kwcoco.json')
        kwargs = {
            'geotiff_dpath': geotiff_dpath
            'dst': dst,
        }

        dset1 = dset
        dset2 = kwcoco.CocoDataset(ub.expandpath('$HOME/data/dvc-repos/smart_watch_dvc/drop0/drop0.kwcoco.json'))
    """
    config = GeotiffToKwcocoConfig(default=kwargs, cmdline=True)
    geotiff_dpath = config['geotiff_dpath']

    dset = find_geotiffs(geotiff_dpath)
    dset.fpath = config['dst']
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
