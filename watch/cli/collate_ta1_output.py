import argparse
import sys
import json
import os
import tempfile
import subprocess
from dateutil.parser import parse
from concurrent.futures import as_completed
import re
from functools import partial

import ubelt as ub
import pystac
from osgeo import gdal

from watch.utils import util_bands


SUPPORTED_S2_PLATFORMS = {'S2A',
                          'S2B',
                          'sentinel-2a',
                          'sentinel-2b'}  # Sentinel
SUPPORTED_LS_PLATFORMS = {'OLI_TIRS',
                          'LANDSAT_8'}  # Landsat
SUPPORTED_WV_PLATFORMS = {'DigitalGlobe',
                          'worldview-2',
                          'worldview-3'}  # Worldview
SUPPORTED_PLATFORMS = (SUPPORTED_S2_PLATFORMS |
                       SUPPORTED_LS_PLATFORMS |
                       SUPPORTED_WV_PLATFORMS)
PLATFORM_SHORTHAND = {**{p: 's2' for p in SUPPORTED_S2_PLATFORMS},
                      **{p: 'ls' for p in SUPPORTED_LS_PLATFORMS},
                      **{p: 'wv' for p in SUPPORTED_WV_PLATFORMS}}

# Maps Sentinel 2 STAC item asset names to the suffixes they should
# use in the collated output.  Note that this also serves as a filter
# as any asset name not found here will be excluded from the output
S2_ASSET_NAME_MAP = {'image-B01': 'B01',
                     'image-B02': 'B02',
                     'image-B03': 'B03',
                     'image-B04': 'B04',
                     'image-B05': 'B05',
                     'image-B06': 'B06',
                     'image-B07': 'B07',
                     'image-B08': 'B08',
                     'image-B09': 'B09',
                     'image-B10': 'B10',
                     'image-B11': 'B11',
                     'image-B12': 'B12',
                     'image-B8A': 'B8A',
                     'image-cloudmask': 'QA'}

# Maps Sentinel 2 STAC item asset names to the suffixes they should
# use in the collated output for SSH scoring.  Note that this also
# serves as a filter as any asset name not found here will be excluded
# from the output
S2_SSH_ASSET_NAME_MAP = {'image-B02': '10m_B02',
                         'image-B03': '10m_B03',
                         'image-B04': '10m_B04',
                         'image-B8A': 'B05',
                         'image-B11': 'B06',
                         'image-B12': 'B07',
                         'image-cloudmask': 'QA'}

# Maps Landsat 8 STAC item asset names to the suffixes they should
# use in the collated output.  Note that this also serves as a filter
# as any asset name not found here will be excluded from the output
L8_ASSET_NAME_MAP = {'image-B1': 'B01',
                     'image-B2': 'B02',
                     'image-B3': 'B03',
                     'image-B4': 'B04',
                     'image-B5': 'B05',
                     'image-B6': 'B06',
                     'image-B7': 'B07',
                     'image-B8': 'B08',
                     'image-B9': 'B09',
                     'image-B10': 'B10',
                     'image-B11': 'B11',
                     'image-cloudmask': 'QA'}

# Maps Landsat 8 STAC item asset names to the suffixes they should use
# in the collated output for SSH scoring.  Note that this also serves
# as a filter as any asset name not found here will be excluded from
# the output
L8_SSH_ASSET_NAME_MAP = {'image-B2': 'B02',
                         'image-B3': 'B03',
                         'image-B4': 'B04',
                         'image-B5': 'B05',
                         'image-B6': 'B06',
                         'image-B7': 'B07',
                         'image-cloudmask': 'QA'}

# Helper map to take asset suffixes (if different) from maps above to
# asset names as they should appear in the output STAC items
ASSET_SUFFIX_TO_NAME_MAP = {'QA': 'quality',
                            'TCI': 'visual'}


def main():
    parser = argparse.ArgumentParser(
        description="Collate TA-1 output data for T&E consumption")
    parser.add_argument('stac_catalog',
                        type=str,
                        help="Path to input STAC catalog")
    parser.add_argument('output_bucket',
                        type=str,
                        help="S3 bucket path for collated data")
    parser.add_argument("--aws_profile",
                        required=False,
                        type=str,
                        help="AWS Profile to use for AWS S3 CLI commands")
    parser.add_argument("-d", "--dryrun",
                        action='store_true',
                        default=False,
                        help="Run AWS CLI commands with --dryrun flag")
    parser.add_argument("--performer_code",
                        default='kit',
                        type=str,
                        help="Performer code suffix for output "
                             "directories / files (default: 'kit')")
    parser.add_argument("--eval_num",
                        default='1',
                        type=str,
                        help="Evaluation number string for building "
                             "output paths (default: '1')")
    parser.add_argument("-j", "--jobs",
                        type=int,
                        default=1,
                        required=False,
                        help="Number of jobs to run in parallel")

    collate_ta1_output(**vars(parser.parse_args()))

    return 0


def _reformat_bandname(band):
    """
    E.g. convert "B7" to "B07"
    """
    m = re.match(r'B(\d)', band)
    if m is not None:
        return "B{:0>2}".format(m.group(1))
    else:
        return band


def _load_input(path):
    try:
        with open(path) as f:
            input_json = json.load(f)
        return input_json['stac'].get('features', [])
    # Excepting KeyError here in case of a single line STAC item input
    except (json.decoder.JSONDecodeError, KeyError):
        # Support for simple newline separated STAC items
        with open(path) as f:
            return [json.loads(line) for line in f]


def _remap_quality_mask(quality_mask_path, outdir):
    # The cloud mask is a uint8, 30m GSD raster with the
    # following values [2]:
    #
    # 0: null
    # 1: clear
    # 2: cloud
    # 3: shadow
    # 4: snow
    # 5: water
    # 6-7: unused
    #
    # [2] https://github.com/ubarsc/python-fmask/blob/master/fmask/fmask.py#L82  # noqa
    #
    # Remapping to Landsat QA standard (first bit in the remapped
    # output indicates whether the pixel should be used for scoring or
    # not with '1' indicating yes it should be used for scoring, and
    # '0' indicating no)
    output_path = os.path.join(outdir, 'out_qa.tif')
    subprocess.run(['gdal_calc.py',
                    '-A', quality_mask_path,
                    '--outfile', output_path,
                    '--overwrite',
                    '--quiet',
                    '--calc',
                    '1*(A==1)+64*(A==1)+128*(A==5)+16*(A==3)+32*(A==4)+8*(A==2)+255*(A==255)',  # noqa
                    '--NoDataValue', '255'], check=True)

    return output_path


def collate_item(stac_item,
                 working_dir,
                 aws_base_command,
                 output_bucket,
                 performer_code,
                 eval_num):
    # TODO: Make use of `working_dir` argument here; not currently
    # used but expected by streaming decorators (in util_framework)
    if isinstance(stac_item, dict):
        stac_item = pystac.Item.from_dict(stac_item)

    platform = stac_item.properties['platform']

    if platform not in SUPPORTED_PLATFORMS:
        print("* Warning unknown platform: '{}' for item, "
              "skipping!".format(platform))
        return None

    output_stac_collection_id = 'ta1-{}-{}'.format(
        PLATFORM_SHORTHAND[platform], performer_code)

    if 'watch:original_item_id' in stac_item.properties:
        original_id = stac_item.properties['watch:original_item_id']
    elif platform in SUPPORTED_LS_PLATFORMS:
        original_id = stac_item.properties.get(
            'landsat:scene_id', stac_item.id)
    elif platform in SUPPORTED_S2_PLATFORMS:
        original_id = stac_item.properties.get(
            'sentinel:product_id', stac_item.id)
    elif platform in SUPPORTED_WV_PLATFORMS:
        if 'nitf:auxiliary_image_identifier' in stac_item.properties:
            original_id, *_ = stac_item.properties.get(
                'nitf:auxiliary_image_identifier').split()
        else:
            original_id = stac_item.id

    mgrs_utm_zone = str(stac_item.properties.get('mgrs:utm_zone', 'ZZ'))
    mgrs_lat_band = str(stac_item.properties.get('mgrs:latitude_band', 'B'))
    mgrs_grid_square = str(stac_item.properties.get('mgrs:grid_square', 'SS'))

    if platform in SUPPORTED_LS_PLATFORMS:
        output_item_id = "{}_{}{}{}_{}".format(
            original_id,
            mgrs_utm_zone,
            mgrs_lat_band,
            mgrs_grid_square,
            performer_code)
    else:
        output_item_id = "{}_{}".format(original_id, performer_code)

    item_datetime = parse(stac_item.properties['datetime'])
    # NOTE ** Assumes that we're compliant with the MGRS STAC
    # extension (but use dummy values just in case):
    # https://github.com/stac-extensions/mgrs
    item_s3_outdir = '/'.join((
        output_bucket,
        output_stac_collection_id,
        mgrs_utm_zone,
        mgrs_lat_band,
        mgrs_grid_square,
        "{:0>4}".format(item_datetime.year),
        "{:0>2}".format(item_datetime.month),
        "{:0>2}".format(item_datetime.day),
        output_item_id))

    eval_name = 'eval-{}'.format(eval_num)
    ssh_outdir = '/'.join(
        (output_bucket, eval_name))

    if platform in SUPPORTED_LS_PLATFORMS:
        platform_collation_fn = partial(generic_collate_item,
                                        L8_ASSET_NAME_MAP,
                                        L8_SSH_ASSET_NAME_MAP,
                                        util_bands.LANDSAT8)
    elif platform in SUPPORTED_S2_PLATFORMS:
        platform_collation_fn = partial(generic_collate_item,
                                        S2_ASSET_NAME_MAP,
                                        S2_SSH_ASSET_NAME_MAP,
                                        util_bands.SENTINEL2,
                                        additional_ssh_qa_resolutions=[10])
    elif platform in SUPPORTED_WV_PLATFORMS:
        platform_collation_fn = collate_wv_item

    output_stac_item = platform_collation_fn(stac_item,
                                             aws_base_command,
                                             original_id,
                                             item_s3_outdir,
                                             ssh_outdir,
                                             performer_code)

    # Completely discard item if platform collation fails
    if output_stac_item is None:
        return None

    stac_item_outpath = '/'.join((item_s3_outdir,
                                  '{}.json'.format(original_id)))
    output_stac_item.set_self_href(stac_item_outpath)

    original_links = output_stac_item.get_links('original')
    if len(original_links) > 0:
        original_stac_item_uri = original_links[0].get_absolute_href()
    else:
        original_stac_item_uri = ''

    output_stac_item.id = output_item_id
    output_stac_item.properties['smart:performer'] = performer_code
    output_stac_item.properties['smart:evaluation'] = eval_num
    output_stac_item.properties['smart:source'] = original_stac_item_uri
    output_stac_item.collection_id = output_stac_collection_id

    return output_stac_item


def convert_to_cog(input_filepath, resampling='AVERAGE'):
    # Citing: https://smartgitlab.com/TE/standards/-/wikis/Data-Output-Specifications#cloud-optomized-geotiff-cog  # noqa
    # Pixel interleaving
    # Internal tiling with block size 256x256 pixels
    # Internal overviews with block size 128x128 pixels and
    # downsampling levels of 2, 4, 8, 16, 32, and 64
    # Compression with the "deflate" algorithm
    output_filepath = '_cog'.join(os.path.splitext(input_filepath))

    subprocess.run(['gdal_translate',
                    input_filepath, output_filepath,
                    '-q',  # quiet
                    '-of', 'cog',
                    '-co', 'COMPRESS=DEFLATE',
                    '-co', 'BLOCKSIZE=256',
                    '-co', 'OVERVIEW_RESAMPLING={}'.format(resampling.upper()),
                    '--config', 'GDAL_TIFF_OVR_BLOCKSIZE', '128'], check=True)

    return output_filepath


def _get_eo_bands_info(asset_name, eo_bands_list, replacement_name=None):
    band_name = asset_name.replace('image-', '')

    for b in eo_bands_list:
        if b['name'] == band_name:
            out_b = b.copy()

            if replacement_name is not None:
                out_b['name'] = replacement_name

            return [out_b]

    return None


def generic_collate_item(asset_name_map,
                         ssh_asset_name_map,
                         eo_bands_list,
                         stac_item,
                         aws_base_command,
                         original_id,
                         item_outdir,
                         ssh_outdir,
                         performer_code,
                         additional_ssh_qa_resolutions=[]):
    item_outdir_base = os.path.basename(item_outdir)
    output_assets = {}
    for asset_name, asset in stac_item.assets.items():
        # Don't output asset if not included in map
        asset_suffix = asset_name_map.get(asset_name)
        ssh_asset_suffix = ssh_asset_name_map.get(asset_name)

        if asset_suffix is None:
            continue

        stac_asset_outpath_basename = "{}_{}_{}.tif".format(
            original_id, performer_code, asset_suffix)
        stac_asset_outpath = '/'.join(
            (item_outdir, stac_asset_outpath_basename))

        eo_bands_info = _get_eo_bands_info(asset_name,
                                           eo_bands_list,
                                           replacement_name=asset_suffix)

        # Default to asset_suffix if a map isn't found
        output_asset_name = ASSET_SUFFIX_TO_NAME_MAP.get(
            asset_suffix, asset_suffix)
        output_asset_dict = {'href': stac_asset_outpath,
                             'title': '/'.join((item_outdir_base,
                                                stac_asset_outpath_basename)),
                             'roles': ['data']}
        if eo_bands_info is not None:
            output_asset_dict['eo:bands'] = eo_bands_info

        output_assets[output_asset_name] =\
            pystac.Asset.from_dict(output_asset_dict)

        # Copy assets up to S3
        with tempfile.TemporaryDirectory() as tmpdirname:
            asset_href = asset.href

            if asset_suffix == 'QA':
                # Remap QA band
                print("* Remapping QA band ..")

                asset_href = _remap_quality_mask(asset_href, tmpdirname)

                for qa_res in additional_ssh_qa_resolutions:
                    local_resized_qa_outpath = os.path.join(
                        tmpdirname, 'qa_{}.tif'.format(qa_res))

                    if not os.path.isfile(local_resized_qa_outpath):
                        subprocess.run(['gdalwarp',
                                        '-overwrite',
                                        '-of', 'GTiff',
                                        '-r', 'near',
                                        '-q',
                                        '-tr', str(qa_res), str(qa_res),
                                        asset_href,
                                        local_resized_qa_outpath], check=True)

                    local_resized_qa_outpath = convert_to_cog(
                        local_resized_qa_outpath,
                        resampling='NEAREST')

                    resized_qa_ssh_outpath = '/'.join(
                        (ssh_outdir, "{}_{}_SSH_{}m_{}.tif".format(
                            original_id,
                            performer_code,
                            int(qa_res),
                            ssh_asset_suffix)))
                    subprocess.run([*aws_base_command,
                                    local_resized_qa_outpath,
                                    resized_qa_ssh_outpath], check=True)

                asset_href = convert_to_cog(asset_href, resampling='NEAREST')
            else:
                asset_href = convert_to_cog(asset_href, resampling='AVERAGE')

            subprocess.run([*aws_base_command,
                            asset_href, stac_asset_outpath], check=True)

            if ssh_asset_suffix is not None:
                ssh_asset_outpath = '/'.join(
                    (ssh_outdir, "{}_{}_SSH_{}.tif".format(
                        original_id, performer_code, ssh_asset_suffix)))

                subprocess.run([*aws_base_command,
                                asset_href, ssh_asset_outpath], check=True)

    with tempfile.NamedTemporaryFile() as temporary_file:
        datetime = stac_item.properties['datetime']

        with open(temporary_file.name, 'w') as f:
            print(datetime, file=f)

        datetime_outpath = '/'.join(
                    (ssh_outdir, "{}_{}_SSH_datetime.txt".format(
                            original_id, performer_code)))
        subprocess.run([*aws_base_command,
                        temporary_file.name, datetime_outpath], check=True)

        for qa_res in additional_ssh_qa_resolutions:
            qa_res_datetime_outpath = '/'.join(
                (ssh_outdir, "{}_{}_SSH_{}m_datetime.txt".format(
                    original_id, performer_code, qa_res)))

            subprocess.run([*aws_base_command,
                            temporary_file.name, qa_res_datetime_outpath],
                           check=True)

    stac_item.assets = output_assets

    return stac_item


def collate_wv_item(stac_item,
                    aws_base_command,
                    original_id,
                    item_outdir,
                    ssh_outdir,
                    performer_code):
    # WV items only have a single "data" asset containing all bands
    data_asset = stac_item.assets.get('data')
    if data_asset is None:
        print("** Error ** Missing expected 'data' asset from "
              "Worldview STAC Item skipping!")
        return None

    def _out_bands(band_dicts):
        return [(_reformat_bandname(b['name']), b.copy())
                for b in band_dicts]

    bands = gdal.Info(data_asset.href, format='json')['bands']

    if len(bands) == 1:
        output_bands = _out_bands(util_bands.WORLDVIEW2_PAN)
    elif len(bands) == 4:
        output_bands = _out_bands(util_bands.WORLDVIEW2_MS4)
    elif len(bands) == 8:
        output_bands = _out_bands(util_bands.WORLDVIEW2_MS8)
    else:
        print('unknown channel signature for WV')
        return None

    item_outdir_base = os.path.basename(item_outdir)
    output_assets = {}
    for band_i, band in enumerate(output_bands, start=1):
        asset_suffix, eo_band_dict = band
        eo_band_dict['name'] = asset_suffix
        with tempfile.NamedTemporaryFile(suffix='.tif') as temporary_file:
            if len(output_bands) > 1:
                # Extract band as a seperate image
                output_band_path = temporary_file.name
                subprocess.run(['gdal_calc.py',
                                '--quiet',
                                '--calc', 'A',
                                '--outfile', output_band_path,
                                '-A', data_asset.href,
                                '--A_band', str(band_i),
                                '--overwrite'], check=True)
            else:
                # Only a single band output file, don't need to
                # split our input image in this case
                output_band_path = data_asset.href

            output_band_path = convert_to_cog(output_band_path,
                                              resampling='AVERAGE')

            stac_asset_outpath_basename = "{}_{}_{}.tif".format(
                original_id, performer_code, asset_suffix)
            stac_asset_outpath = '/'.join(
                (item_outdir, stac_asset_outpath_basename))

            # Default to asset_suffix if a map isn't found
            output_asset_name = ASSET_SUFFIX_TO_NAME_MAP.get(
                asset_suffix, asset_suffix)
            output_assets[output_asset_name] = pystac.Asset.from_dict(
                {'href': stac_asset_outpath,
                 'title': '/'.join((item_outdir_base,
                                    stac_asset_outpath_basename)),
                 'roles': ['data'],
                 'eo:bands': [eo_band_dict]})

            # Copy assets up to S3
            subprocess.run([*aws_base_command,
                            output_band_path, stac_asset_outpath],
                           check=True)

    stac_item.assets = output_assets

    return stac_item


def build_and_upload_stac_collections(stac_items_by_collection,
                                      aws_base_command,
                                      output_bucket,
                                      performer_code):
    for collection_id, stac_items in stac_items_by_collection.items():
        collection_output_path = '/'.join((
            output_bucket,
            collection_id,
            'collection.json'))

        output_collection = pystac.Collection(
            collection_id,
            "STAC Collection '{}' for SMART program from team '{}'".format(
                collection_id, performer_code),
            pystac.Extent.from_items(stac_items),
            href=collection_output_path)

        for stac_item in stac_items:
            prior_self_href = stac_item.get_self_href()
            output_collection.add_item(stac_item)
            # Reset item's self href as `pystac.Collection.add_item`
            # changes it
            stac_item.set_self_href(prior_self_href)

            with tempfile.NamedTemporaryFile() as temporary_file:
                with open(temporary_file.name, 'w') as f:
                    json.dump(stac_item.to_dict(), f, indent=2)

                # Assumes that the STAC item's self href has been set
                # in the per item collation function(s)
                subprocess.run([*aws_base_command,
                                temporary_file.name,
                                stac_item.get_self_href()], check=True)

        with tempfile.NamedTemporaryFile() as temporary_file:
            with open(temporary_file.name, 'w') as f:
                json.dump(output_collection.to_dict(), f, indent=2)

            subprocess.run([*aws_base_command,
                            temporary_file.name,
                            collection_output_path], check=True)


def collate_ta1_output(stac_catalog,
                       output_bucket,
                       aws_profile=None,
                       dryrun=False,
                       performer_code='kit',
                       eval_num='1',
                       jobs=1):
    if isinstance(stac_catalog, str):
        catalog = pystac.read_file(href=stac_catalog).full_copy()
    else:
        catalog = stac_catalog.full_copy()

    if aws_profile is not None:
        aws_base_command =\
            ['aws', 's3', '--profile', aws_profile, 'cp']
    else:
        aws_base_command = ['aws', 's3', 'cp']

    if dryrun:
        aws_base_command.append('--dryrun')

    input_stac_items = [item.to_dict() for item in catalog.get_all_items()]

    executor = ub.Executor(mode='process' if jobs > 1 else 'serial',
                           max_workers=jobs)
    collation_jobs = [executor.submit(collate_item, stac_item_dict,
                                      None,  # working_dir (not currently used)
                                      aws_base_command,
                                      output_bucket,
                                      performer_code,
                                      eval_num)
                      for stac_item_dict in input_stac_items]

    output_stac_items_by_collection = {}
    for collation_job in ub.ProgIter(as_completed(collation_jobs),
                                     total=len(collation_jobs),
                                     desc='collation jobs'):
        try:
            stac_item = collation_job.result()
        except Exception as e:
            print("Exception occurred (printed below), dropping item!")
            print(e)
            continue
        else:
            if stac_item is not None:
                output_stac_items_by_collection.setdefault(
                    stac_item.collection_id, []).append(stac_item)

    build_and_upload_stac_collections(output_stac_items_by_collection,
                                      aws_base_command,
                                      output_bucket,
                                      performer_code)


if __name__ == "__main__":
    sys.exit(main())
