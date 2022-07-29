import argparse
import sys
from dateutil.parser import isoparse, parse
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
from watch.utils import kwcoco_extensions

from os.path import basename, dirname, join

try:
    from xdev import profile
except Exception:
    profile = ub.probile


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
                        default=False,
                        help="Assume the data is in subdirectories relative "
                             "to the output KWCOCO manifest")
    parser.add_argument("--from-collated",
                        action='store_true',
                        default=False,
                        help="Data to convert has been run through TA-1 "
                             "collation")
    parser.add_argument("--ignore_duplicates",
                        action='store_true',
                        default=False,
                        help="Ignore duplicate items when creating the kwcoco file")
    parser.add_argument("--populate-watch-fields",
                        action='store_true',
                        default=False,
                        help="Populate video / watch fields")
    parser.add_argument("-j", "--jobs",
                        type=str,
                        default=1,
                        required=False,
                        help="Number of jobs to run in parallel")

    ta1_stac_to_kwcoco(**vars(parser.parse_args()))
    return 0


SUPPORTED_COARSE_PLATFORMS = {
    'S2': {'S2A', 'S2B', 'sentinel-2a', 'sentinel-2b', 'S2'},  # Sentinel-2
    'L8': {'OLI_TIRS', 'LANDSAT_8', 'L8'},  # Landsat-8
    'WV': {'DigitalGlobe', 'worldview-2', 'worldview-3', 'WV'},  # Worldview
    'PD': {'PlanetScope', 'dove', 'PD'},  # Planet
}

SUPPORTED_PLATFORMS = set.union(
    *SUPPORTED_COARSE_PLATFORMS.values(),
    set(SUPPORTED_COARSE_PLATFORMS.keys()))


SENSOR_COARSE_MAPPING = {
    v: k
    for k, vals in SUPPORTED_COARSE_PLATFORMS.items()
    for v in vals
}

L8_CHANNEL_ALIAS = {band['name']: band['common_name']
                    for band in util_bands.LANDSAT8 if 'common_name' in band}
S2_CHANNEL_ALIAS = {band['name']: band['common_name']
                    for band in util_bands.SENTINEL2 if 'common_name' in band}
# ...except for TCI, which is not a true band, but often included anyway
# and this channel code is more specific to kwcoco
S2_CHANNEL_ALIAS.update({'TCI': 'tci:3'})
L8_CHANNEL_ALIAS.update({'TCI': 'tci:3'})


def _determine_channels_collated(asset_name, asset_dict):
    """
    Note:
        The term "collated" means that each band is its own asset and it has
        the eo:bands property. For more details see:
        https://smartgitlab.com/TE/standards/-/wikis/STAC-and-Storage-Specifications
    """
    eo_band_names = [eob.get('common_name', eob['name'])
                     for eob in asset_dict.get('eo:bands', ())]

    if len(eo_band_names) > 0:
        return '|'.join(eo_band_names)
    elif asset_name == 'quality':
        return 'cloudmask'


def _determine_s2_channels(asset_name, asset_dict):
    """
        >>> from watch.cli.ta1_stac_to_kwcoco import *  # NOQA
        >>> from watch.cli.ta1_stac_to_kwcoco import _determine_s2_channels
        >>> test_hrefs = [
        >>>     '/vsis3/smart-data-accenture/ta-1/ta1-s2-acc/15/T/TF/2020/9/21/S2A_14TQL_20200921_0_L1C_ACC/S2A_14TQL_20200921_0_L1C_ACC_QA.tif',
        >>>     '/vsis3/smart-data-accenture/ta-1/ta1-ls-acc/14/T/QK/2020/8/9/LC08_L1TP_028032_20200809_20200917_02_T1_ACC/LC08_L1TP_028032_20200809_20200917_02_T1_ACC_cloud_mask.tif',
        >>>     '/vsis3/smart-data-accenture/ta-1/ta1-ls-acc/14/T/QK/2020/8/25/LC08_L1TP_028032_20200825_20200905_02_T1_ACC/LC08_L1TP_028032_20200825_20200905_02_T1_ACC_cloud_mask.tif',
        >>>     '/vsis3/smart-data-accenture/ta-1/ta1-s2-acc/14/T/QL/2020/10/14/S2A_14TQL_20201014_0_L1C_ACC/S2A_MSI_L1C_T14TQL_20201014_20201014_B02.img',
        >>>     '/vsis3/smart-data-accenture/ta-1/ta1-s2-acc/14/T/QL/2020/10/14/S2A_14TQL_20201014_0_L1C_ACC/S2A_MSI_L1C_T14TQL_20201014_20201014_B02.hdr',
        >>>     '/vsis3/smart-data-accenture/ta-1/ta1-s2-acc/14/T/QL/2020/10/14/S2A_14TQL_20201014_0_L1C_ACC/HLS.S10.T14TQL.2020288.T173227.v1.5.hdf',
        >>>     '/vsis3/smart-data-accenture/ta-1/ta1-s2-acc/14/T/QL/2020/10/14/S2A_14TQL_20201014_0_L1C_ACC/angle_output.hdf',
        >>>     '/vsis3/smart-data-accenture/ta-1/ta1-s2-acc/14/T/QL/2020/10/14/S2A_14TQL_20201014_0_L1C_ACC/S2A_14TQL_20201014_0_L1C_ACC_ac_mask.tif',
        >>>     '/vsis3/smart-data-accenture/ta-1/ta1-s2-acc/14/T/QL/2020/10/14/S2A_14TQL_20201014_0_L1C_ACC/S2A_14TQL_20201014_0_L1C_ACC_cloud_mask.tif',
        >>>     '/vsis3/smart-data-accenture/ta-1/ta1-s2-acc/14/T/QL/2020/10/14/S2A_14TQL_20201014_0_L1C_ACC/S2A_14TQL_20201014_0_L1C_ACC_QA.tif',
        >>>     '/vsis3/smart-data-accenture/ta-1/ta1-ls-acc/14/T/QM/2021/7/27/LC08_L1TP_028031_20210727_20210804_02_T1_ACC/LC08_L1TP_028031_20210727_20210804_02_T1_ACC_cloud_mask.tif',
        >>>     '/vsis3/smart-data-accenture/ta-1/ta1-ls-acc/14/T/QM/2021/8/28/LC08_L1TP_028031_20210828_20210901_02_T1_ACC/LC08_L1TP_028031_20210828_20210901_02_T1_ACC_cloud_mask.tif',
        >>>     '/vsis3/smart-data-accenture/ta-1/ta1-ls-acc/14/T/QK/2021/6/9/LC08_L1TP_028032_20210609_20210615_02_T1_ACC/LC08_L1TP_028032_20210609_20210615_02_T1_ACC_cloud_mask.tif',
        >>>     '/vsis3/smart-data-accenture/ta-1/ta1-ls-acc/14/T/QK/2021/7/27/LC08_L1TP_028032_20210727_20210804_02_T1_ACC/LC08_L1TP_028032_20210727_20210804_02_T1_ACC_cloud_mask.tif',
        >>>     '/vsis3/smart-data-accenture/ta-1/ta1-ls-acc/14/T/QM/2021/4/22/LC08_L1TP_028031_20210422_20210430_02_T1_ACC/LC08_L1TP_028031_20210422_20210430_02_T1_ACC_cloud_mask.tif',
        >>> ]
        >>> for href in test_hrefs:
        ...     asset_name = None
        ...     asset_dict = {'href': href}
        ...     channels = _determine_s2_channels(asset_name, asset_dict)
        ...     print(f'channels={channels}')
    """
    asset_href = asset_dict['href']
    eo_band_names = [eob['name'] for eob in asset_dict.get('eo:bands', ())]
    # print(f'asset_href={asset_href}')
    # print(f'eo_band_names={eo_band_names}')

    if re.search(r'TCI\.(tiff?|jp2)$', asset_href, re.I):
        return S2_CHANNEL_ALIAS.get('TCI', 'tci:3')
    elif re.search(r'PVI\.(tiff?|jp2)$', asset_href, re.I):
        return 'pvi:3'
    elif re.search(r'PVI\.(tiff?|jp2)$', asset_href, re.I):
        # PVI is preview image
        return 'pvi:3'
    elif re.search(r'AOT\.(tiff?|jp2)$', asset_href, re.I):
        # AOT is Aerosol Optical Thickness
        return 'aot'
    elif re.search(r'WVP\.(tiff?|jp2)$', asset_href, re.I):
        # https://sentinels.copernicus.eu/web/sentinel/user-guides/sentinel-2-msi/processing-levels/level-2
        # WV is Water Vapour
        return 'wvp'
    elif re.search(r'SCL\.(tiff?|jp2)$', asset_href, re.I):
        # https://sentinels.copernicus.eu/web/sentinel/user-guides/sentinel-2-msi/processing-levels/level-2
        # SCL Scene Classification map
        return 'scl'
    elif re.search(r'cloudmask\.(tiff?|jp2)$', asset_href, re.I):
        return 'cloudmask'
    elif re.search(r'SR_AEROSOL\.(tiff?|jp2)$', asset_href, re.I):
        return 'sr_aerosol_mask'
    elif len(eo_band_names) > 0:
        return '|'.join((S2_CHANNEL_ALIAS.get(eobn, eobn)
                         for eobn in eo_band_names))
    elif m := re.search(r'(B\w{2})\.(tiff?|jp2)$', asset_href, re.I):  # NOQA
        return S2_CHANNEL_ALIAS.get(m.group(1), m.group(1))
    else:
        href_path = ub.Path(asset_href)
        stem = href_path.stem
        known_suffixes = [
            'QA',
            'ac_mask',
            'cloud_mask',
        ]
        for suffix in known_suffixes:
            if stem.endswith('_' + suffix):
                return suffix
        return None


def _determine_l8_channels(asset_name, asset_dict):
    """
    Example:
        >>> from watch.cli.ta1_stac_to_kwcoco import *  # NOQA
        >>> from watch.cli.ta1_stac_to_kwcoco import _determine_l8_channels
        >>> test_hrefs = [
        >>>     '/vsis3/smart-data-accenture/ta-1/ta1-ls-acc/52/S/EG/2017/12/2/LC08_L1TP_114034_20171202_20200902_02_T1_ACC/LC08_L1TP_114034_20171202_20200902_02_T1_ACC_QA.tif',
        >>>     '/vsis3/smart-data-accenture/ta-1/ta1-ls-acc/52/S/EG/2017/12/2/LC08_L1TP_114034_20171202_20200902_02_T1_ACC/LC08_L1TP_114034_20171202_20200902_02_T1_ACC_TCI.tif',
        >>>     '/vsis3/smart-data-accenture/ta-1/ta1-ls-acc/52/S/DG/2017/9/20/LC08_L1TP_115034_20170920_20200903_02_T1_ACC/LC08_L1TP_115034_20170920_20200903_02_T1_ACC_ac_mask.tif',
        >>>     '/vsis3/smart-data-accenture/ta-1/ta1-ls-acc/52/S/DG/2017/9/20/LC08_L1TP_115034_20170920_20200903_02_T1_ACC/LC08_L1TP_115034_20170920_20200903_02_T1_ACC_solar_zenith_angle.tif',
        >>>     '/vsis3/smart-data-accenture/ta-1/ta1-ls-acc/52/S/DG/2017/9/20/LC08_L1TP_115034_20170920_20200903_02_T1_ACC/LC08_L1TP_115034_20170920_20200903_02_T1_ACC_solar_azimuth_angle.tif',
        >>>     '/vsis3/smart-data-accenture/ta-1/ta1-ls-acc/52/S/DG/2017/9/20/LC08_L1TP_115034_20170920_20200903_02_T1_ACC/LC08_L1TP_115034_20170920_20200903_02_T1_ACC_view_zenith_angle.tif',
        >>>     '/vsis3/smart-data-accenture/ta-1/ta1-ls-acc/52/S/DG/2017/9/20/LC08_L1TP_115034_20170920_20200903_02_T1_ACC/LC08_L1TP_115034_20170920_20200903_02_T1_ACC_view_azimuth_angle.tif',
        >>>     '/vsis3/smart-data-accenture/ta-1/ta1-ls-acc/52/S/DG/2017/9/20/LC08_L1TP_115034_20170920_20200903_02_T1_ACC/LC08_L1TP_115034_20170920_20200903_02_T1_ACC_QA.tif',
        >>>     '/vsis3/smart-data-accenture/ta-1/ta1-ls-acc/52/S/DG/2017/9/20/LC08_L1TP_115034_20170920_20200903_02_T1_ACC/LC08_L1TP_115034_20170920_20200903_02_T1_ACC_TCI.tif',
        >>>     '/vsis3/smart-data-accenture/ta-1/ta1-ls-acc/52/S/EG/2017/9/13/LC08_L1TP_114034_20170913_20200903_02_T1_ACC/LC08_L1TP_114034_20170913_20200903_02_T1_ACC_ac_mask.tif',
        >>>     '/vsis3/smart-data-accenture/ta-1/ta1-ls-acc/52/S/EG/2017/9/13/LC08_L1TP_114034_20170913_20200903_02_T1_ACC/LC08_L1TP_114034_20170913_20200903_02_T1_ACC_solar_zenith_angle.tif',
        >>>     '/vsis3/smart-data-accenture/ta-1/ta1-ls-acc/52/S/EG/2017/9/13/LC08_L1TP_114034_20170913_20200903_02_T1_ACC/LC08_L1TP_114034_20170913_20200903_02_T1_ACC_solar_azimuth_angle.tif',
        >>>     '/vsis3/smart-data-accenture/ta-1/ta1-ls-acc/52/S/EG/2017/9/13/LC08_L1TP_114034_20170913_20200903_02_T1_ACC/LC08_L1TP_114034_20170913_20200903_02_T1_ACC_view_zenith_angle.tif',
        >>>     '/vsis3/smart-data-accenture/ta-1/ta1-ls-acc/52/S/DG/2020/6/8/LC08_L1TP_115034_20200608_20200824_02_T1_ACC/LC08_L1TP_115034_20200608_20200824_02_T1_ACC_cloud_mask.tif',
        >>> ]
        >>> for href in test_hrefs:
        ...     asset_name = None
        ...     asset_dict = {'href': href}
        ...     channels = _determine_l8_channels(asset_name, asset_dict)
        ...     print(f'channels={channels}')
    """
    asset_href = asset_dict['href']
    eo_band_names = []
    for eob in asset_dict.get('eo:bands', []):
        if isinstance(eob, dict):
            eo_band_names.append(eob['name'])
        elif isinstance(eob, str):
            eo_band_names.append(eob)
        else:
            raise TypeError(f'type(eob) = {type(eob)}')

    if len(eo_band_names) > 0:
        mapped_names = list(ub.unique([L8_CHANNEL_ALIAS.get(eobn, eobn) for eobn in eo_band_names]))
        return '|'.join(mapped_names)
    elif re.search(r'cloudmask\.(tiff?|jp2)$', asset_href, re.I):
        return 'cloudmask'
    elif m := re.search(r'(QA_PIXEL|QA_RADSAT|SR_QA_AEROSOL)\.(tiff?|jp2)$',  # NOQA
                        asset_href, re.I):
        return m.group(1).lower()
    elif m := re.search(r'(B\w{1,2})\.(tiff?|jp2)$', asset_href, re.I):  # NOQA
        return L8_CHANNEL_ALIAS.get(m.group(1), m.group(1))
    else:
        stem = ub.Path(asset_href).stem
        if stem.endswith('_TCI'):
            return 'tci:3'
        known_suffixes = [
            'QA',
            'ac_mask',
            'solar_zenith_angle',
            'view_zenith_angle',
            'solar_zenith_angle',
            'view_azimuth_angle',
            'view_zenith_angle',
            'solar_azimuth_angle',
            'solar_zenith_angle',
            'cloud_mask',
        ]
        for suffix in known_suffixes:
            if stem.endswith('_' + suffix):
                return suffix
        return None


def _determine_wv_channels(asset_name, asset_dict):
    asset_href = asset_dict['href']

    eo_band_names = []
    for eob in asset_dict.get('eo:bands', []):
        if isinstance(eob, dict):
            eo_band_names.append(eob['name'])
        elif isinstance(eob, str):
            eo_band_names.append(eob)
        else:
            raise TypeError(f'type(eob) = {type(eob)}')

    if eo_band_names:
        channels = '|'.join(eo_band_names)
    else:
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
    return channels


@profile
def make_coco_aux_from_stac_asset(asset_name,
                                  asset_dict,
                                  platform,
                                  name=None,
                                  force_affine=True,
                                  assume_relative=False,
                                  from_collated=False,
                                  populate_watch_fields=True):
    """
    Converts a single STAC asset into an "auxiliary" item / asset that will
    belong to a kwcoco image.
    """
    img = {}
    if name is not None:
        img['name'] = name

    asset_href = asset_dict['href']

    # Skip assets with metadata or thumbnail extensions
    if re.search(r'\.(txt|csv|json|xml|vrt|jpe?g)$', asset_href, re.I):
        return None

    if re.search(r'\.(img|hdr|hdf|imd)$', asset_href, re.I):
        return None

    # HACK Skip common TCI (true color images) and PVI (preview images)
    # naming schemes
    if re.search(r'TCI\.jp2$', asset_href, re.I):
        return None
    if re.search(r'_PVI\.tif$', asset_href, re.I):
        return None

    if from_collated and platform in SUPPORTED_PLATFORMS:
        channels = _determine_channels_collated(asset_name, asset_dict)
    elif platform in SUPPORTED_COARSE_PLATFORMS['S2']:
        channels = _determine_s2_channels(asset_name, asset_dict)
    elif platform in SUPPORTED_COARSE_PLATFORMS['L8']:
        channels = _determine_l8_channels(asset_name, asset_dict)
    elif platform in SUPPORTED_COARSE_PLATFORMS['WV']:
        channels = _determine_wv_channels(asset_name, asset_dict)
    else:
        raise NotImplementedError(
            "Unsupported platform '{}'".format(platform))

    # Hard-coded
    ignore_channels = [
        'solar_zenith_angle',
        'view_zenith_angle',
        'solar_zenith_angle',
        'view_azimuth_angle',
        'view_zenith_angle',
        'solar_azimuth_angle',
        'solar_zenith_angle',
        'tci:3',
    ]
    if channels is not None:
        if channels in ignore_channels:
            return None

    if channels is None:
        HACK_AWAY_SOME_WARNINGS = 1
        if HACK_AWAY_SOME_WARNINGS:
            # FIXME: parametarize or make robust
            IGNORE_SUFFIXES = (
                '_SAA.TIF', '_VZA.TIF', '_VAA.TIF', '_SZA.TIF'
            )
            if asset_href.endswith(IGNORE_SUFFIXES):
                return None
        # Collated output must always have eo:bands, so dont warn
        if not from_collated:
            print("* Warning * Couldn't determine channels for asset "
                  "at: '{}'. Asset will be ignored.".format(asset_href))
        return None

    if assume_relative:
        file_name = join(basename(dirname(asset_href)), basename(asset_href))
    else:
        file_name = asset_href

    img.update({
        'file_name': file_name,
        'channels': channels,
    })

    if populate_watch_fields:
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
            ref = None

        # Combine sensor candidates
        sensor_candidates = list(ub.flatten([
            v.get('sensor_candidates', []) for v in infos.values()]))
        info = ub.dict_union(*infos.values())
        info['sensor_candidates'] = sensor_candidates
        warp_pxl_to_wld = kwimage.Affine.coerce(info['pxl_to_wld'])
        height, width = info['img_shape']
        wld_crs_info = ub.dict_diff(info['wld_crs_info'], {'type'})
        utm_crs_info = ub.dict_diff(info['utm_crs_info'], {'type'})
        img.update({
            'width': width,
            'height': height,
            'num_bands': info['num_bands'],
            'approx_meter_gsd': info['approx_meter_gsd'],
            'warp_pxl_to_wld': warp_pxl_to_wld,
            'utm_corners': info['utm_corners'].data.tolist(),
            'wld_crs_info': wld_crs_info,
            'utm_crs_info': utm_crs_info,
        })

    return img


@profile
def _stac_item_to_kwcoco_image(stac_item,
                               assume_relative=False,
                               from_collated=False,
                               populate_watch_fields=True):
    stac_item_dict = stac_item.to_dict()

    platform = stac_item_dict['properties']['platform']
    if 'constellation' in stac_item_dict['properties']:
        if stac_item_dict['properties']['constellation'] == 'dove':
            platform = 'PD'

    if platform not in SUPPORTED_PLATFORMS:
        print("* Warning * platform '{}' not supported, not adding to "
              "KWCOCO output!".format(platform))
        return None

    img = {
        'name': stac_item.id,
        'file_name': None,
    }
    auxiliary = []

    for asset_name, asset_dict in stac_item_dict.get('assets', {}).items():
        aux = make_coco_aux_from_stac_asset(
            asset_name,
            asset_dict,
            platform,
            force_affine=True,
            assume_relative=assume_relative,
            from_collated=from_collated,
            populate_watch_fields=populate_watch_fields
        )
        if aux is not None:
            auxiliary.append(aux)

    # Choose a base image canvas and the relationship between auxiliary images
    if populate_watch_fields:
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

    if len(auxiliary) == 0:
        img['failed'] = stac_item

    img['auxiliary'] = auxiliary

    date = stac_item_dict['properties']['datetime']
    date = isoparse(date).isoformat()
    img['date_captured'] = date

    sensor_coarse = SENSOR_COARSE_MAPPING[platform]
    img['sensor_coarse'] = sensor_coarse

    return img


@profile
def ta1_stac_to_kwcoco(input_stac_catalog,
                       outpath,
                       assume_relative=False,
                       populate_watch_fields=False,
                       jobs=1,
                       from_collated=False,
                       ignore_duplicates=False):

    from watch.utils.lightning_ext import util_globals
    jobs = util_globals.coerce_num_workers(jobs)

    if isinstance(input_stac_catalog, str):
        catalog = pystac.read_file(href=input_stac_catalog).full_copy()
    elif isinstance(input_stac_catalog, dict):
        catalog = pystac.Catalog.from_dict(input_stac_catalog).full_copy()
    else:
        catalog = input_stac_catalog.full_copy()

    outdir = dirname(outpath)
    os.makedirs(outdir, exist_ok=True)

    executor = ub.JobPool(mode='process' if jobs > 1 else 'serial',
                          max_workers=jobs)

    all_items = [stac_item for stac_item in catalog.get_all_items()]

    if 1:
        # Sumamrize items before processing
        sensorchan_hist = ub.ddict(lambda: 0)
        sensorasset_hist = ub.ddict(lambda: 0)
        for stac_item in all_items:
            # TODO: we can use this data to prepopulate the kwcoco file
            # so it takes far less time to field it.
            stac_dict = stac_item.to_dict()
            # stac_dict['geometry']
            sensor = stac_dict['properties'].get(
                'constellation', stac_dict['properties'].get('platform', None))
            # proc_level = stac_dict['landsat:correction']
            asset_names = stac_dict['assets'].keys()
            eo_bands = []
            for asset_name, asset_item in stac_dict['assets'].items():
                if 'roles' in asset_item and 'data' in asset_item['roles']:
                    if 'eo:bands' in asset_item:
                        for eo_band in asset_item['eo:bands']:
                            if isinstance(eo_band, dict):
                                if 'common_name' in eo_band:
                                    eo_bands.append(eo_band['common_name'])
                                elif 'name' in eo_band:
                                    eo_bands.append(eo_band['name'])
                                else:
                                    raise AssertionError
                            elif isinstance(eo_band, str):
                                eo_bands.append(eo_band)
            eo_bands = list(ub.unique(eo_bands))
            eo_bands = list(ub.unique(eo_bands))
            sensorchan = kwcoco.SensorChanSpec.coerce(f'{sensor}:' + '|'.join(eo_bands))
            sensorchan_hist[sensorchan.spec] += 1
            sensorasset = kwcoco.SensorChanSpec.coerce(f'{sensor}:' + '|'.join(sorted(asset_names)))
            sensorasset_hist[sensorasset.spec] += 1
        print('sensorchan_hist = {}'.format(ub.repr2(sensorchan_hist, nl=1)))
        print('sensorasset_hist = {}'.format(ub.repr2(sensorasset_hist, nl=1)))

    for stac_item in all_items:
        executor.submit(_stac_item_to_kwcoco_image, stac_item,
                        assume_relative=assume_relative,
                        from_collated=from_collated,
                        populate_watch_fields=populate_watch_fields)

    output_dset = kwcoco.CocoDataset()
    output_dset.fpath = outpath

    # TODO: Should make this name the MGRS tile
    for job in executor.as_completed(desc='collect jobs'):
        kwcoco_img = job.result()
        if kwcoco_img is not None:
            # Ignore iamges with 0 auxiliary items
            if len(kwcoco_img.get('auxiliary', [])) == 0:
                print('Failed kwcoco_img = {}'.format(ub.repr2(kwcoco_img, nl=1)))
                continue
            try:
                output_dset.add_image(**kwcoco_img)
            except kwcoco.exceptions.DuplicateAddError:
                if not ignore_duplicates:
                    print(ub.paragraph(
                        '''
                        Error encountered duplicate item. Debugging duplicates.
                        Did you append to the same input list multiple times?
                        '''))
                    raise

    if populate_watch_fields:
        video_id = output_dset.add_video(name=ub.hash_data(catalog.to_dict()))

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
        json.dump(output_dset.dataset, f, indent=2)

    print('Wrote: {}'.format(outpath))

    return output_dset


if __name__ == "__main__":
    sys.exit(main())

"""
Test:
    import pandas as pd

    freq = ub.ddict(list)
    for img in dset.dataset['images']:
        sensor = img['sensor_coarse']
        freq[sensor].append(len(img['auxiliary']))

    for sensor, auxfreq in freq.items():
        print('sensor = {!r}'.format(sensor))
        print(ub.dict_hist(auxfreq))
        pass


"""
