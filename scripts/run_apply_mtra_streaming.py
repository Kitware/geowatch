import argparse
import sys
import traceback
from concurrent.futures import as_completed
import os
import subprocess
from glob import glob

import ubelt
import pystac

from watch.cli.baseline_framework_egress import upload_output_stac_items
from watch.cli.run_mtra import (apply_harmonization_item_map,
                                ensure_map_at_res)
from watch.utils.util_framework import (CacheItemOutputS3Wrapper,
                                        IngressProcessEgressWrapper)
from watch.cli.baseline_framework_ingress import load_input_stac_items


def main():
    parser = argparse.ArgumentParser(
        description="Apply MTRA coefficients to sentinel items")

    parser.add_argument('input_path',
                        type=str,
                        help="Path to input T&E Baseline Framework JSON")
    parser.add_argument('output_path',
                        type=str,
                        help="S3 path for output JSON")
    parser.add_argument("--aws_profile",
                        required=False,
                        type=str,
                        help="AWS Profile to use for AWS S3 CLI commands")
    parser.add_argument("-d", "--dryrun",
                        action='store_true',
                        default=False,
                        help="Run AWS CLI commands with --dryrun flag")
    parser.add_argument('-s', '--show-progress',
                        action='store_true',
                        default=False,
                        help='Show progress for AWS CLI commands')
    parser.add_argument("-m", "--mtra-maps-bucket",
                        type=str,
                        required=True,
                        help="S3 bucket container MTRA harmonization "
                             "coefficient maps")
    parser.add_argument("-o", "--outbucket",
                        type=str,
                        required=True,
                        help="S3 bucket for harmonized output")
    parser.add_argument("-j", "--jobs",
                        type=int,
                        default=1,
                        required=False,
                        help="Number of jobs to run in parallel")

    run_apply_mtra_coefficients(**vars(parser.parse_args()))

    return 0


SELECTED_PLATFORMS = {'S2A',
                      'S2B',
                      'sentinel-2a',
                      'sentinel-2b'}
BANDS_TO_HARMONIZE = ['B02', 'B03', 'B04', 'B8A', 'B11', 'B12']


def _asset_selector(asset_name, asset):
    # WV items only have a single "data" asset containing all bands
    return ((asset_name in BANDS_TO_HARMONIZE) or
            (asset_name.replace("image-", "") in BANDS_TO_HARMONIZE))


def _item_selector(stac_item):
    if isinstance(stac_item, dict):
        return stac_item['properties'].get('platform') in SELECTED_PLATFORMS
    else:
        return stac_item.properties.get('platform') in SELECTED_PLATFORMS


def apply_harmonization_item_map_wrapper(stac_item,
                                         outdir,
                                         item_selector,
                                         bands_to_harmonize,
                                         slope_maps_by_mgrs,
                                         intercept_maps_by_mgrs):
    # MTRA map MGRS string contains 'T' prefix (e.g. 'T32RLU')
    mgrs_tile = ''.join(('T',
                         stac_item.properties.get('mgrs:utm_zone', '??'),
                         stac_item.properties.get('mgrs:latitude_band', '?'),
                         stac_item.properties.get('mgrs:grid_square', '??')))

    slope_map = slope_maps_by_mgrs[mgrs_tile]
    intercept_map = intercept_maps_by_mgrs[mgrs_tile]

    return apply_harmonization_item_map(stac_item,
                                        outdir,
                                        item_selector,
                                        bands_to_harmonize,
                                        slope_map,
                                        intercept_map)


def run_apply_mtra_coefficients(input_path,
                                output_path,
                                mtra_maps_bucket,
                                outbucket,
                                aws_profile=None,
                                dryrun=False,
                                show_progress=False,
                                requester_pays=False,
                                jobs=1):
    if aws_profile is not None:
        aws_base_command =\
            ['aws', 's3', '--profile', aws_profile, 'cp']
    else:
        aws_base_command = ['aws', 's3', 'cp']

    if dryrun:
        aws_base_command.append('--dryrun')

    if not show_progress:
        aws_base_command.append('--no-progress')

    if requester_pays:
        aws_base_command.extend(['--request-payer', 'requester'])

    print("* Downloading harmonization maps")
    mtra_maps_outdir = '/tmp/mtra_maps'
    os.makedirs(mtra_maps_outdir, exist_ok=True)
    subprocess.run([*aws_base_command, '--recursive',
                    mtra_maps_bucket, mtra_maps_outdir],
                   check=True)

    def _mgrs_from_map_name(map_name):
        base, _ = os.path.splitext(os.path.basename(map_name))
        _, mgrs = base.split('_')

        return mgrs

    print("* Creating resized harmonization maps")
    # Precompute different GSD slope & intercept map files to avoid
    # having difference processes try to do it at the same time.
    slope_maps = {}
    for slope_map_path in glob(os.path.join(mtra_maps_outdir,
                                            'Slope_*.tif')):
        ensure_map_at_res(slope_map_path, 10.0, 10.0)
        ensure_map_at_res(slope_map_path, 20.0, 20.0)

        slope_maps[_mgrs_from_map_name(slope_map_path)] =\
            slope_map_path

    intercept_maps = {}
    for intercept_map_path in glob(os.path.join(mtra_maps_outdir,
                                                'Intercept_*.tif')):
        ensure_map_at_res(intercept_map_path, 10.0, 10.0)
        ensure_map_at_res(intercept_map_path, 20.0, 20.0)

        intercept_maps[_mgrs_from_map_name(intercept_map_path)] =\
            intercept_map_path

    print("* Applying harmonization")
    input_stac_items = load_input_stac_items(input_path, aws_base_command)

    executor = ubelt.Executor(mode='process' if jobs > 1 else 'serial',
                              max_workers=jobs)

    # Skipping ingress / egress here as the collation function performs a
    # specialized ingress / egress
    ingress_process_egress_map = IngressProcessEgressWrapper(
        apply_harmonization_item_map_wrapper,
        outbucket,
        aws_base_command,
        dryrun=dryrun,
        stac_item_selector=_item_selector,
        asset_selector=_asset_selector)
    caching_item_map = CacheItemOutputS3Wrapper(
        ingress_process_egress_map,
        outbucket,
        aws_profile=aws_profile)
    harmonization_jobs = [executor.submit(caching_item_map,
                                          stac_item,
                                          _item_selector,
                                          BANDS_TO_HARMONIZE,
                                          slope_maps, intercept_maps)
                          for stac_item in input_stac_items]

    output_stac_items = []
    for harmonization_job in as_completed(harmonization_jobs):
        try:
            stac_item = harmonization_job.result()
        except Exception:
            print("Exception occurred (printed below), dropping item!")
            traceback.print_exception(*sys.exc_info())
            continue
        else:
            if isinstance(stac_item, dict):
                stac_item = pystac.Item.from_dict(stac_item)
                output_stac_items.append(stac_item)
            elif isinstance(stac_item, pystac.Item):
                output_stac_items.append(stac_item)
            else:
                for si in stac_item:
                    if isinstance(si, dict):
                        si = pystac.Item.from_dict(si)
                        output_stac_items.append(si)
                    elif isinstance(si, pystac.Item):
                        output_stac_items.append(si)

    te_output = upload_output_stac_items(
        output_stac_items, output_path, aws_base_command, newline=True)

    return te_output


if __name__ == "__main__":
    sys.exit(main())
