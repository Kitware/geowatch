#!/usr/bin/env python3
r"""
Convert a STAC catalog to a kwcoco file.

The following example illustrates how to start from a STAC query, resolve that
into a STAC catalog and finally use this script to convert it to KWCOCO.

CommandLine:
    # Create a demo region file
    xdoctest geowatch.demo.demo_region demo_khq_region_fpath

    DATASET_SUFFIX=DemoKHQ-2022-06-10-V2
    DEMO_DPATH=$HOME/.cache/geowatch/demo/datasets

    REGION_FPATH="$HOME/.cache/geowatch/demo/annotations/KHQ_R001.geojson"
    SITE_GLOBSTR="$HOME/.cache/geowatch/demo/annotations/KHQ_R001_sites/*.geojson"

    START_DATE=$(jq -r '.features[] | select(.properties.type=="region") | .properties.start_date' "$REGION_FPATH")
    END_DATE=$(jq -r '.features[] | select(.properties.type=="region") | .properties.end_date' "$REGION_FPATH")
    # Shrink time window to test with less data
    START_DATE=2016-12-02
    END_DATE=2020-12-31
    REGION_ID=$(jq -r '.features[] | select(.properties.type=="region") | .properties.region_id' "$REGION_FPATH")
    SEARCH_FPATH=$DEMO_DPATH/stac_search.json
    RESULT_FPATH=$DEMO_DPATH/all_sensors_kit/${REGION_ID}.input
    CATALOG_FPATH=$DEMO_DPATH/all_sensors_kit/${REGION_ID}_catalog.json
    KWCOCO_FPATH=$DEMO_DPATH/all_sensors_kit/${REGION_ID}.kwcoco.zip

    mkdir -p "$DEMO_DPATH"

    # Create the search json wrt the sensors and processing level we want
    python -m geowatch.stac.stac_search_builder \
        --start_date="$START_DATE" \
        --end_date="$END_DATE" \
        --cloud_cover=40 \
        --sensors=sentinel-2-l2a \
        --out_fpath "$SEARCH_FPATH"
    cat "$SEARCH_FPATH"

    # Delete this to prevent duplicates
    rm -f "$RESULT_FPATH"

    # Create the .input file
    python -m geowatch.cli.stac_search \
        --region_file "$REGION_FPATH" \
        --search_json "$SEARCH_FPATH" \
        --mode area \
        --verbose 2 \
        --outfile "${RESULT_FPATH}"

    cat "$RESULT_FPATH"

    python -m geowatch.cli.baseline_framework_ingress \
        --input_path="$RESULT_FPATH" \
        --catalog_fpath="${CATALOG_FPATH}" \
        --virtual=True \
        --jobs=avail \
        --aws_profile=iarpa \
        --requester_pays=0

    AWS_DEFAULT_PROFILE=iarpa python -m geowatch.cli.stac_to_kwcoco \
        --input_stac_catalog="${CATALOG_FPATH}" \
        --outpath="$KWCOCO_FPATH" \
        --jobs=8 \
        --from_collated=False \
        --ignore_duplicates=0

    # Check that the resulting kwcoco has what you want in it
    geowatch stats "$KWCOCO_FPATH"

    # Use kwcoco info to dump a single image dictionary
    kwcoco info "$KWCOCO_FPATH" -g 1 -i 0

SeeAlso:
    ~/code/watch/geowatch/cli/stac_search.py
    ~/code/watch/geowatch/cli/prepare_ta2_dataset.py
"""
import os
import re
import sys
import ubelt as ub
import scriptconfig as scfg

from os.path import basename, dirname, join

try:
    from xdev import profile
except Exception:
    profile = ub.identity


class StacToCocoConfig(scfg.DataConfig):
    """
    Convert a STAC catalog to a KWCOCO manifest
    """
    input_stac_catalog = scfg.Value(None, type=str, position=1, required=True, help='Input STAC Catalog to convert to KWCOCO')

    outpath = scfg.Value(None, type=str, short_alias=['o'], help='Output path for updated STAC catalog')

    assume_relative = scfg.Value(False, isflag=True, help=ub.paragraph(
            '''
            Assume the data is in subdirectories relative to the output
            KWCOCO manifest
            '''))

    from_collated = scfg.Value(False, isflag=True, help=ub.paragraph(
            '''
            Data to convert has been run through TA-1 collation
            '''))

    ignore_duplicates = scfg.Value(False, isflag=True, help=ub.paragraph(
            '''
            Ignore duplicate items when creating the kwcoco file
            '''))

    populate_watch_fields = scfg.Value(False, isflag=True, help='Populate video / geowatch feilds')

    verbose = scfg.Value(1, help='verbosity')

    jobs = scfg.Value(0, type=str, short_alias=['j'], help='Number of jobs to run in parallel (Use zero, dont parallelize this)')


def main(cmdline=True, **kwargs):
    config = StacToCocoConfig.cli(cmdline=cmdline, data=kwargs, strict=True)
    import rich
    rich.print('config = {}'.format(ub.urepr(config, nl=1)))
    stac_to_kwcoco(**config)
    return 0


SUPPORTED_COARSE_PLATFORMS = {
    'S2': {'S2A', 'S2B', 'sentinel-2a', 'sentinel-2b', 'S2'},  # Sentinel-2
    'L8': {'OLI_TIRS', 'LANDSAT_8', 'L8'},  # Landsat-8
    'WV': {'DigitalGlobe', 'worldview-2', 'worldview-3', 'WV', 'WV02', 'WV03', 'wv-2', 'wv-3'},  # Worldview
    'WV1': {'WV01', 'wv-1'},  # Worldview 1
    'PD': {'PlanetScope', 'dove', 'PD'},  # Planet
}


def normalize_str(s):
    return s.lower().replace('-', '_')


PLATFORM_NORMALIZED_TO_PLATFORM_STANDARD_CASE = {
    normalize_str(v): v
    for vs in SUPPORTED_COARSE_PLATFORMS.values()
    for v in vs
}

SUPPORTED_PLATFORMS = set.union(
    *SUPPORTED_COARSE_PLATFORMS.values(),
    set(SUPPORTED_COARSE_PLATFORMS.keys()))


SENSOR_COARSE_MAPPING = {
    v: k
    for k, vals in SUPPORTED_COARSE_PLATFORMS.items()
    for v in vals
}


def _construct_sensor_channel_alias():
    """
    Construct mappings from possible names for each bands to the ones that we
    want to use in kwcoco.
    """
    from geowatch.utils import util_bands
    UTIL_BAND_INFOS = {
        'S2': util_bands.SENTINEL2,
        'L8': util_bands.LANDSAT8,
    }
    SENSOR_CHANNEL_ALIAS = {}
    for sensor_code, band_infos in UTIL_BAND_INFOS.items():
        alias_lut = SENSOR_CHANNEL_ALIAS[sensor_code] = {}
        for band in band_infos:
            if 'common_name' in band:
                alias_lut[band['name']] = band['common_name']
                alias_lut[band['common_name']] = band['common_name']
                for alias in band.get('alias', []):
                    alias_lut[alias] = band['common_name']
        # TCI is not a true band, but often included anyway
        # and this channel code is more specific to kwcoco
        alias_lut['TCI'] = 'tci:3'
    return SENSOR_CHANNEL_ALIAS


SENSOR_CHANNEL_ALIAS = _construct_sensor_channel_alias()


def _determine_channels_collated(asset_name, asset_dict, platform):
    """
    Note:
        The term "collated" means that each band is its own asset and it has
        the eo:bands property. For more details see:
        https://smartgitlab.com/TE/standards/-/wikis/STAC-and-Storage-Specifications

    """
    sensor_coarse = SENSOR_COARSE_MAPPING.get(platform, platform)

    eo_band_names = [eob.get('common_name', eob['name'])
                     for eob in asset_dict.get('eo:bands', ())]

    # Map aliases for this sensor if we have registered it
    channel_alias_lut = SENSOR_CHANNEL_ALIAS.get(sensor_coarse, None)
    if channel_alias_lut is not None:
        eo_band_names = [
            channel_alias_lut.get(name, name) for name in eo_band_names]

    if len(eo_band_names) > 0:
        return '|'.join(eo_band_names)
    elif asset_name == 'quality':
        return 'quality'
        # return 'cloudmask'


def _determine_s2_channels(asset_name, asset_dict):
    """
        >>> from geowatch.cli.stac_to_kwcoco import *  # NOQA
        >>> from geowatch.cli.stac_to_kwcoco import _determine_s2_channels
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
        return SENSOR_CHANNEL_ALIAS['S2'].get('TCI', 'tci:3')
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
        return '|'.join((SENSOR_CHANNEL_ALIAS['S2'].get(eobn, eobn)
                         for eobn in eo_band_names))
    elif m := re.search(r'(B\w{2})\.(tiff?|jp2)$', asset_href, re.I):  # NOQA
        return SENSOR_CHANNEL_ALIAS['S2'].get(m.group(1), m.group(1))
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
        >>> from geowatch.cli.stac_to_kwcoco import *  # NOQA
        >>> from geowatch.cli.stac_to_kwcoco import _determine_l8_channels
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
        mapped_names = list(ub.unique([SENSOR_CHANNEL_ALIAS['L8'].get(eobn, eobn) for eobn in eo_band_names]))
        return '|'.join(mapped_names)
    elif re.search(r'cloudmask\.(tiff?|jp2)$', asset_href, re.I):
        return 'cloudmask'
    elif m := re.search(r'(QA_PIXEL|QA_RADSAT|QA_LINEAGE|SR_QA_AEROSOL)\.(tiff?|jp2)$',  # NOQA
                        asset_href, re.I):
        return m.group(1).lower()
    elif m := re.search(r'(B\w{1,2})\.(tiff?|jp2)$', asset_href, re.I):  # NOQA
        return SENSOR_CHANNEL_ALIAS['L8'].get(m.group(1), m.group(1))
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
    from geowatch.utils import util_bands
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
        from osgeo import gdal
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
                                  verbose=0):
    """
    Converts a single STAC asset into an "auxiliary" item / asset that will
    belong to a kwcoco image.
    """
    asset = {}
    if name is not None:
        asset['name'] = name

    asset_href = asset_dict['href']

    # Skip assets with metadata or thumbnail extensions
    if re.search(r'\.(txt|csv|json|xml)$', asset_href, re.I):
        if verbose:
            print(f'SKIP META asset: {asset_href}')
        return None

    if re.search(r'\.(vrt|jpe?g)$', asset_href, re.I):
        if verbose:
            print(f'SKIP THUMB asset: {asset_href}')
        return None

    if re.search(r'\.(img|hdr|hdf|imd)$', asset_href, re.I):
        if verbose:
            print(f'SKIP asset: {asset_href}')
        return None

    # HACK Skip common TCI (true color images) and PVI (preview images)
    # naming schemes
    if re.search(r'TCI\.jp2$', asset_href, re.I):
        if verbose:
            print(f'SKIP TCI asset: {asset_href}')
        return None
    if re.search(r'_PVI\.tif$', asset_href, re.I):
        if verbose:
            print(f'SKIP PVI asset: {asset_href}')
        return None

    if from_collated and platform in SUPPORTED_PLATFORMS:
        if verbose:
            print('Detected collated channels')
        channels = _determine_channels_collated(asset_name, asset_dict,
                                                platform)
    elif platform in SUPPORTED_COARSE_PLATFORMS['S2']:
        if verbose:
            print('Detected S2 channels')
        channels = _determine_s2_channels(asset_name, asset_dict)
    elif platform in SUPPORTED_COARSE_PLATFORMS['L8']:
        if verbose:
            print('Detected L8 channels')
        channels = _determine_l8_channels(asset_name, asset_dict)
    elif platform in SUPPORTED_COARSE_PLATFORMS['WV']:
        if verbose:
            print('Detected WV channels')
        channels = _determine_wv_channels(asset_name, asset_dict)
    elif platform in SUPPORTED_COARSE_PLATFORMS['WV1']:
        if verbose:
            print('Detected WV channels')
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
            if verbose:
                print(f'SKIP ignored asset: {asset_href}')
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
        elif verbose:
            print(f'SKIP Unknown asset: {asset_href}')
        return None

    if verbose:
        print(f'ADD asset: {asset_href}')

    if assume_relative:
        file_name = join(basename(dirname(asset_href)), basename(asset_href))
    else:
        file_name = asset_href

    roles = asset_dict.get('roles', [])
    assert isinstance(roles, list)
    if channels == 'quality':
        if 'quality' not in roles:
            roles.append('quality')

    asset.update({
        'file_name': file_name,
        'channels': channels,
        'stac_asset_key': asset_name,
        'roles': roles,
    })
    return asset


@profile
def _stac_item_to_kwcoco_image(stac_item,
                               assume_relative=False,
                               from_collated=False,
                               verbose=0):
    """
    Example:
        >>> from geowatch.cli.stac_to_kwcoco import *  # NOQA
        >>> stac_item = _demo_item()
        >>> from_collated = True
        >>> img = _stac_item_to_kwcoco_image(stac_item, from_collated=from_collated)
        >>> import kwcoco
        >>> coco_img = kwcoco.CocoImage(img)
    """
    from kwutil import util_time
    stac_item_dict = stac_item.to_dict()

    platform = stac_item_dict['properties']['platform']
    if 'constellation' in stac_item_dict['properties']:
        if stac_item_dict['properties']['constellation'] == 'dove':
            platform = 'PD'

    if platform == 'DigitalGlobe' and stac_item_dict['properties']['mission'] == 'WV01':
        # Ensure we can differentiate between WV01 (which sometimes
        # has 'DigitalGlobe' as the 'platform') and other WV imagery
        platform = 'WV01'

    # Convet to standard case
    platform = PLATFORM_NORMALIZED_TO_PLATFORM_STANDARD_CASE.get(normalize_str(platform), platform)

    if platform not in SUPPORTED_PLATFORMS:
        print("* Warning * platform '{}' not supported, not adding to "
              "KWCOCO output!".format(platform))
        return None

    img = {
        'name': stac_item.id,
        'file_name': None,
    }
    assets = []
    for asset_name, asset_dict in stac_item_dict.get('assets', {}).items():
        coco_asset = make_coco_aux_from_stac_asset(
            asset_name,
            asset_dict,
            platform,
            force_affine=True,
            assume_relative=assume_relative,
            from_collated=from_collated,
            verbose=verbose,
        )
        if coco_asset is not None:
            assets.append(coco_asset)

    DISAMBIGUATE_CHANNELS_HACK = 1
    if DISAMBIGUATE_CHANNELS_HACK:
        # Fixes some cases where the common names of multiple assets are
        # duplicate.
        orig_channel_names = [a['channels'] for a in assets]
        dup_idxs = ub.find_duplicates(orig_channel_names)
        if dup_idxs:
            import warnings
            warnings.warn('Data had duplicate channel names, attempting to resolve')
            for k, idxs in dup_idxs.items():
                for idx in idxs:
                    asset = assets[idx]
                    asset['channels'] = asset['stac_asset_key']
            fixed_channel_names = [a['channels'] for a in assets]
            if ub.find_duplicates(fixed_channel_names):
                warnings.warn('Creating CocoImage with duplicate channel names. There will be no way to distinguish them.')

    if len(assets) == 0:
        print("* Warning * Empty auxiliary assets for "
              "STAC Item '{}', skipping!".format(stac_item.id))
        return None

    if len(assets) == 0:
        img['failed'] = stac_item

    from geowatch import heuristics
    img[heuristics.COCO_ASSETS_KEY] = assets
    img['stac_properties'] = stac_item_dict['properties']
    date = stac_item_dict['properties']['datetime']
    date = util_time.coerce_datetime(date).isoformat()
    img['date_captured'] = date

    sensor_coarse = SENSOR_COARSE_MAPPING[platform]
    img['sensor_coarse'] = sensor_coarse

    return img


@profile
def stac_to_kwcoco(input_stac_catalog,
                   outpath,
                   assume_relative=False,
                   populate_watch_fields=False,
                   jobs=0,
                   from_collated=False,
                   ignore_duplicates=False,
                   verbose=1):

    if populate_watch_fields:
        raise NotImplementedError('REMOVED: use coco_add_watch_feilds '
                                  'as a secondary step instead')

    from kwutil import util_parallel
    import pystac
    import kwcoco
    jobs = util_parallel.coerce_num_workers(jobs)

    if isinstance(input_stac_catalog, str):
        catalog = pystac.read_file(href=input_stac_catalog).full_copy()
    elif isinstance(input_stac_catalog, dict):
        catalog = pystac.Catalog.from_dict(input_stac_catalog).full_copy()
    else:
        catalog = input_stac_catalog.full_copy()

    outdir = dirname(outpath)
    os.makedirs(outdir, exist_ok=True)

    if jobs == 1:
        jobs = 0
    if jobs > 0:
        import warnings
        warnings.warn(ub.paragraph(
            '''
            There is usually no benefit to having jobs > 0, here.
            It often makes the process go (a lot) slower.
            '''))

    executor = ub.JobPool(mode='process', max_workers=jobs)

    all_items = [stac_item for stac_item in catalog.get_all_items()]

    if 1:
        if verbose > 5:
            # Printout one item per sensor
            sensor_to_one_item = {}
            for stac_item in all_items:
                # TODO: we can use this data to prepopulate the kwcoco file
                # so it takes far less time to field it.
                stac_dict = stac_item.to_dict()
                # stac_dict['geometry']
                sensor = stac_dict['properties'].get(
                    'constellation', stac_dict['properties'].get('platform', None))
                sensor_to_one_item[sensor] = stac_dict
            print('sensor_to_one_item = {}'.format(ub.urepr(sensor_to_one_item, nl=True)))

        # Sumamrize items before processing
        sensorchan_hist = ub.ddict(lambda: 0)
        sensorasset_hist = ub.ddict(lambda: 0)
        for stac_item in all_items:
            summary = summarize_stac_item(stac_item)
            # print('summary = {}'.format(ub.urepr(summary, nl=1)))
            sensor = summary['sensor']
            eo_bands = summary['eo_bands']
            asset_names = summary['asset_names']
            sensorchan = kwcoco.SensorChanSpec.coerce(f'{sensor}:' + '|'.join(eo_bands))
            sensorchan_hist[sensorchan.spec] += 1
            sensorasset = kwcoco.SensorChanSpec.coerce(f'{sensor}:' + '|'.join(sorted(asset_names)))
            sensorasset_hist[sensorasset.spec] += 1
            # TODO: stac_item['geometry'] - we can prepopulate geo information
            # stac_dict['properties']

        print('sensorchan_hist = {}'.format(ub.urepr(sensorchan_hist, nl=1)))
        print('sensorasset_hist = {}'.format(ub.urepr(sensorasset_hist, nl=1)))

    for stac_item in all_items:
        executor.submit(_stac_item_to_kwcoco_image, stac_item,
                        assume_relative=assume_relative,
                        from_collated=from_collated,
                        verbose=verbose > 1)

    output_dset = kwcoco.CocoDataset()
    output_dset.fpath = outpath

    # TODO: Should make this name the MGRS tile
    for job in executor.as_completed(desc='collect jobs', progkw={'verbose': verbose}):
        kwcoco_img = job.result()
        if kwcoco_img is not None:
            # Ignore iamges with 0 auxiliary items
            if len(kwcoco_img.get('auxiliary', [])) == 0:
                print('Failed kwcoco_img = {}'.format(ub.urepr(kwcoco_img, nl=1)))
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

    output_dset.dump(indent=2)
    print('Wrote: {}'.format(outpath))
    return output_dset


def summarize_stac_item(stac_item):
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
    eo_cloud_cover = stac_dict['properties'].get('eo:cloud_cover', None)

    summary = {
        'sensor': sensor,
        'asset_names': asset_names,
        'eo_bands': eo_bands,
        'eo_cloud_cover': eo_cloud_cover,
        'datetime': stac_item.get_datetime(),
    }
    return summary


def _demo_item():
    import pystac
    feat = {
        "type": "Feature",
        "stac_version": "1.0.0",
        "id": "S2B_18NTG_20200824_0_L2A",
        "properties": {
            "mgrs:utm_zone": 18,
            "mgrs:latitude_band": "N",
            "mgrs:grid_square": "TG",
            "proj:epsg": 32618,
            "proj:transform": [10, 0, 199980, 0, -10, 200040],
            "proj:shape": [10980, 10980],
            "eo:cloud_cover": 19.272,
            "gsd": 10,
            "platform": "Sentinel-2B",
            "qa_percent:clouds": 19.21,
            "qa_percent:aligned": 100,
            "constellation": "sentinel-2",
            "datetime": "2020-08-24T15:42:51.278000Z"
        },
        "bbox": [ -10.0, 0.31415, -10.0, 0.31415 ],
        "geometry": None,
        "links": [
            { "rel": "self", "href": "https://foobar.com/items/S2B_18NTG_20200824_0_L2A", "type": "application/geo+json" },
        ],
        "assets": {
            "quality": {
                "href": "s3://foobar/S2B_18NTG_20200824_0_L2A_QA.tif",
                "type": "image/tiff; application=geotiff",
                "title": "Quality Assurance Band",
                "description": "Quality assurance bitmask image"
            },
            "visual": {
                "href": "s3://foobar/S2B_18NTG_20200824_0_L2A_TCI.tif",
                "type": "image/tiff; application=geotiff",
                "title": "True Color Image",
                "description": "True color image (RGB), for visualization"
            },
            "B01": {
                "href": "s3://foobar/S2B_18NTG_20200824_0_L2A_B01.tif",
                "type": "image/tiff; application=geotiff",
                "title": "B01 - CoastalAerosol",
                "eo:bands": [ { "name": "B01", "center_wavelength": 442, "common_name": "coastal" } ]
            },
            "B09": {
                "href": "s3://foobar/S2B_18NTG_20200824_0_L2A_B09.tif",
                "type": "image/tiff; application=geotiff",
                "title": "B09 - WaterVapor",
                "eo:bands": [ { "name": "B09", "center_wavelength": 943 } ]
            },
            "B02": {
                "href": "s3://foobar/S2B_18NTG_20200824_0_L2A_B02.tif",
                "type": "image/tiff; application=geotiff",
                "title": "B02 - Blue",
                "eo:bands": [ { "name": "B02", "center_wavelength": 492, "common_name": "blue" } ]
            },
            "B03": {
                "href": "s3://foobar/S2B_18NTG_20200824_0_L2A_B03.tif",
                "type": "image/tiff; application=geotiff",
                "title": "B03 - Green",
                "eo:bands": [ { "name": "B03", "center_wavelength": 559, "common_name": "green" } ]
            },
            "B04": {
                "href": "s3://foobar/S2B_18NTG_20200824_0_L2A_B04.tif",
                "type": "image/tiff; application=geotiff",
                "title": "B04 - Red",
                "eo:bands": [ { "name": "B04", "center_wavelength": 665, "common_name": "red" } ]
            },
            "B08": {
                "href": "s3://foobar/S2B_18NTG_20200824_0_L2A_B08.tif",
                "type": "image/tiff; application=geotiff",
                "title": "B08 - NIR",
                "eo:bands": [ { "name": "B08", "center_wavelength": 833, "common_name": "nir" } ]
            },
            "B08A": {
                "href": "s3://foobar/S2B_18NTG_20200824_0_L2A_B08A.tif",
                "type": "image/tiff; application=geotiff",
                "title": "B08A - NarrowNIR",
                "eo:bands": [ { "name": "B08A", "center_wavelength": 864, "common_name": "rededge" } ]
            },
            "B11": {
                "href": "s3://foobar/S2B_18NTG_20200824_0_L2A_B11.tif",
                "type": "image/tiff; application=geotiff",
                "title": "B11 - SWIR1",
                "eo:bands": [ { "name": "B11", "center_wavelength": 1610, "common_name": "swir16" } ]
            },
            "B12": {
                "href": "s3://foobar/S2B_18NTG_20200824_0_L2A_B12.tif",
                "type": "image/tiff; application=geotiff",
                "title": "B12 - SWIR2",
                "eo:bands": [ { "name": "B12", "center_wavelength": 2186, "common_name": "swir22" } ]
            },
            "B05": {
                "href": "s3://foobar/S2B_18NTG_20200824_0_L2A_B05.tif",
                "type": "image/tiff; application=geotiff",
                "title": "B05 - RedEdge1",
                "eo:bands": [ { "name": "B05", "center_wavelength": 704, "common_name": "rededge" } ]
            },
            "B06": {
                "href": "s3://foobar/S2B_18NTG_20200824_0_L2A_B06.tif",
                "type": "image/tiff; application=geotiff",
                "title": "B06 - RedEdge2",
                "eo:bands": [ { "name": "B06", "center_wavelength": 739, "common_name": "rededge" } ]
            },
            "B07": {
                "href": "s3://foobar/S2B_18NTG_20200824_0_L2A_B07.tif",
                "type": "image/tiff; application=geotiff",
                "title": "B07 - RedEdge3",
                "eo:bands": [{ "name": "B07", "center_wavelength": 780, "common_name": "rededge" } ]
            }
        },
        "stac_extensions": [
            "https://stac-extensions.github.io/projection/v1.0.0/schema.json",
            "https://stac-extensions.github.io/eo/v1.0.0/schema.json",
            "https://stac-extensions.github.io/view/v1.0.0/schema.json",
            "https://stac-extensions.github.io/mgrs/v1.0.0/schema.json"
        ]

    }
    item = pystac.Item.from_dict(feat)
    # catalog_dict = {
    #     "type": "Catalog",
    #     "id": "Baseline Framework ingress catalog",
    #     "stac_version": "1.0.0",
    #     "description": "STAC catalog of SMART search results",
    #     'links': [{
    #         # "rel": "root",
    #         "href": os.fspath(fpath),
    #         "type": "application/json",
    #     }]
    # }
    # catalog = pystac.Catalog.from_dict(catalog_dict)
    # catalog.to_dict()

    return item


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
