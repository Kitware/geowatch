import argparse
import sys
from dateutil.parser import isoparse, parse
from concurrent.futures import as_completed
import os
import json
import re

import pystac
from osgeo import gdal
import ubelt as ub
import kwimage
import kwcoco

import watch
from watch.utils import util_bands
from watch.utils import util_raster
from watch.utils import kwcoco_extensions


def main():
    parser = argparse.ArgumentParser(
        description="Convert a STAC catalog to a KWCOCO manifest")

    parser.add_argument('input_stac_catalog',
                        type=str,
                        help="Input STAC Catalog to convert to KWCOCO")
    parser.add_argument("-o", "--outpath",
                        type=str,
                        help="Output path for updated STAC catalog")
    parser.add_argument("--assume-relative",
                        action='store_true',
                        help="Assume the data is in subdirectories relative "
                             "to the output KWCOCO manifest")
    parser.add_argument("-j", "--jobs",
                        type=int,
                        default=1,
                        required=False,
                        help="Number of jobs to run in parallel")

    ta1_stac_to_kwcoco(**vars(parser.parse_args()))

    return 0


SUPPORTED_S2_PLATFORMS = {'S2A',
                          'S2B',
                          'sentinel-2a',
                          'sentinel-2b'}  # Sentinel
SUPPORTED_LS_PLATFORMS = {'OLI_TIRS',
                          'LANDSAT_8'}  # Landsat
SUPPORTED_WV_PLATFORMS = {'DigitalGlobe'}  # Worldview
SUPPORTED_PLATFORMS = (SUPPORTED_S2_PLATFORMS |
                       SUPPORTED_LS_PLATFORMS |
                       SUPPORTED_WV_PLATFORMS)

SENSOR_COARSE_MAPPING = {**{p: 'S2' for p in SUPPORTED_S2_PLATFORMS},
                         **{p: 'L8' for p in SUPPORTED_LS_PLATFORMS},
                         **{p: 'WV' for p in SUPPORTED_WV_PLATFORMS}}

L8_CHANNEL_ALIAS = {band['name']: band['common_name']
                    for band in util_bands.LANDSAT8 if 'common_name' in band}
S2_CHANNEL_ALIAS = {band['name']: band['common_name']
                    for band in util_bands.SENTINEL2 if 'common_name' in band}
# ...except for TCI, which is not a true band, but often included anyway
# and this channel code is more specific to kwcoco
S2_CHANNEL_ALIAS.update({'TCI': 'r|g|b'})


def _determine_s2_channels(asset_dict):
    asset_href = asset_dict['href']
    eo_band_names = [eob['name'] for eob in asset_dict.get('eo:bands', ())]

    if len(eo_band_names) > 0:
        return '|'.join((S2_CHANNEL_ALIAS.get(eobn, eobn)
                         for eobn in eo_band_names))
    elif re.search(r'cloudmask\.(tiff?|jp2)$', asset_href, re.I):
        return 'cloudmask'
    elif re.search(r'SR_AEROSOL\.(tiff?|jp2)$', asset_href, re.I):
        return 'sr_aerosol_mask'
    else:
        return None


def _determine_l8_channels(asset_dict):
    asset_href = asset_dict['href']
    eo_band_names = [eob['name'] for eob in asset_dict.get('eo:bands', ())]

    if len(eo_band_names) > 0:
        return '|'.join((L8_CHANNEL_ALIAS.get(eobn, eobn)
                         for eobn in eo_band_names))
    elif re.search(r'cloudmask\.(tiff?|jp2)$', asset_href, re.I):
        return 'cloudmask'
    elif m := re.search(r'(QA_PIXEL|QA_RADSAT|SR_QA_AEROSOL)\.(tiff?|jp2)$',  # NOQA
                        asset_href, re.I):
        return m.group(1).lower()
    else:
        return None


def _determine_wv_channels(asset_dict):
    asset_href = asset_dict['href']
    bands = gdal.Info(asset_href, format='json')['bands']

    # the channel names are the same for all WV, just the
    # center_wavelength is different so we can safely use this
    # info from WV2
    def _code(band_dicts):
        return '|'.join(b.get('common_name', b['name'])
                        for b in band_dicts)

    if len(bands) == 1:
        channels = _code(util_bands.WORLDVIEW2_PAN)
    elif len(bands) == 4:
        channels = _code(util_bands.WORLDVIEW2_MS4)
    elif len(bands) == 8:
        channels = _code(util_bands.WORLDVIEW2_MS8)
    else:
        raise Exception('unknown channel signature for WV')


def make_coco_img_from_stac_asset(asset_dict,
                                  platform,
                                  name=None,
                                  force_affine=True,
                                  assume_relative=False):
    img = {}
    if name is not None:
        img['name'] = name

    asset_href = asset_dict['href']

    # Skip assets with metadata extensions
    if re.search(r'\.(txt|csv|json|xml)$', asset_href, re.I):
        return None

    # Largely a copy-paste of
    # `watch.gis.geotiff.geotiff_metadata(asset_href)` without
    # attempting to parse metadata from the filename / path
    infos = {}
    try:
        ref = gdal.Open(asset_href, gdal.GA_ReadOnly)
        if ref is None:
            msg = gdal.GetLastErrorMsg()
            # gdal.GetLastErrorType()
            # gdal.GetLastErrorNo()
            print("* Warning * Couldn't open asset_href '{}' with "
                  "GDAL:".format(asset_href))
            print(msg)
            return None

        infos['crs'] = watch.gis.geotiff.geotiff_crs_info(
            ref, force_affine=force_affine)
        infos['header'] = watch.gis.geotiff.geotiff_header_info(ref)
    finally:
        del ref

    # Combine sensor candidates
    sensor_candidates = list(ub.flatten([
        v.get('sensor_candidates', []) for v in infos.values()]))

    info = ub.dict_union(*infos.values())
    info['sensor_candidates'] = sensor_candidates

    warp_pxl_to_wld = kwimage.Affine.coerce(info['pxl_to_wld'])
    height, width = info['img_shape']

    if platform in SUPPORTED_S2_PLATFORMS:
        channels = _determine_s2_channels(asset_dict)
    elif platform in SUPPORTED_LS_PLATFORMS:
        channels = _determine_l8_channels(asset_dict)
    elif platform in SUPPORTED_WV_PLATFORMS:
        channels = _determine_wv_channels(asset_dict)
    else:
        raise NotImplementedError(
            "Unsupported platform '{}'".format(platform))

    if channels is None:
        print("* Warning * Couldn't determine channels for asset "
              "at: '{}'".format(asset_href))
        return None

    wld_crs_info = ub.dict_diff(info['wld_crs_info'], {'type'})
    utm_crs_info = ub.dict_diff(info['utm_crs_info'], {'type'})

    if assume_relative:
        file_name = os.path.join(os.path.basename(os.path.dirname(asset_href)),
                                 os.path.basename(asset_href))
    else:
        file_name = asset_href

    img.update({
        'file_name': file_name,
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


def _stac_item_to_kwcoco_image(stac_item, assume_relative=False):
    stac_item_dict = stac_item.to_dict()

    platform = stac_item_dict['properties']['platform']

    if platform not in SUPPORTED_PLATFORMS:
        print("* Warning * platform '{}' not supported, not adding to "
              "KWCOCO output!".format(platform))
        return None

    img = {
        'name': stac_item.id,
        'file_name': None,
    }
    auxiliary = []

    for asset_name, asset in stac_item_dict.get('assets', {}).items():
        aux = make_coco_img_from_stac_asset(
            asset,
            platform,
            force_affine=True,
            assume_relative=assume_relative)
        if aux is None:
            continue
        auxiliary.append(aux)

    # Choose a base image canvas and the relationship between auxiliary images
    idx = ub.argmax(auxiliary, lambda x: (x['width'] * x['height']))
    base = auxiliary[idx]
    warp_img_to_wld = base['warp_pxl_to_wld']
    warp_wld_to_img = warp_img_to_wld.inv()
    img['warp_img_to_wld'] = warp_img_to_wld.concise()
    img['warp_to_wld'] = warp_img_to_wld.concise()
    img['approx_meter_gsd'] = base['approx_meter_gsd']
    img.update(ub.dict_isect(base,
                             {'utm_corners', 'wld_crs_info', 'utm_crs_info'}))

    for aux in auxiliary:
        aux.pop('utm_corners')
        aux.pop('utm_crs_info')
        aux.pop('wld_crs_info')
        aux['warp_to_wld'] = aux['warp_pxl_to_wld'].concise()
        warp_aux_to_img = warp_wld_to_img @ aux.pop('warp_pxl_to_wld')
        aux['warp_aux_to_img'] = warp_aux_to_img.concise()

    img['width'] = base['width']
    img['height'] = base['height']
    img['auxiliary'] = auxiliary

    date = stac_item_dict['properties']['datetime']
    date = isoparse(date).isoformat()
    img['date_captured'] = date

    sensor_coarse = SENSOR_COARSE_MAPPING[platform]
    img['sensor_coarse'] = sensor_coarse

    return img


def ta1_stac_to_kwcoco(input_stac_catalog,
                       outpath,
                       assume_relative=False,
                       jobs=1):
    if isinstance(input_stac_catalog, str):
        catalog = pystac.read_file(href=input_stac_catalog).full_copy()
    elif isinstance(input_stac_catalog, dict):
        catalog = pystac.Catalog.from_dict(input_stac_catalog).full_copy()
    else:
        catalog = input_stac_catalog.full_copy()

    outdir = os.path.dirname(outpath)
    os.makedirs(outdir, exist_ok=True)

    executor = ub.Executor(mode='process' if jobs > 1 else 'serial',
                           max_workers=jobs)

    jobs = [executor.submit(_stac_item_to_kwcoco_image,
                            stac_item, assume_relative=assume_relative)
            for stac_item in catalog.get_all_items()]

    output_dset = kwcoco.CocoDataset()
    output_dset.fpath = outpath
    # TODO: Should make this name the MGRS tile
    video_id = output_dset.add_video(name=ub.hash_data(catalog.to_dict()))
    for kwcoco_img in (job.result() for job in as_completed(jobs)):
        if kwcoco_img is not None:
            output_dset.add_image(**kwcoco_img)

    ordered_images = sorted(
        output_dset.images().objs,
        key=lambda obj: parse(obj['date_captured']))

    for i, img in enumerate(ordered_images):
        img['frame_index'] = i
        img['video_id'] = video_id

    output_dset.index.build(output_dset)

    kwcoco_extensions.coco_populate_geo_video_stats(
        output_dset, video_id, target_gsd=10.0)

    with open(outpath, 'w') as f:
        f.write(json.dumps(output_dset.dataset, indent=2))

    return output_dset


if __name__ == "__main__":
    sys.exit(main())
