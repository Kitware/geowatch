import argparse
import sys
import traceback
import os
import json
import subprocess
from concurrent.futures import as_completed
import tempfile
import re

import ubelt
import pystac

from watch.utils.util_framework import (CacheItemOutputS3Wrapper,
                                        IngressProcessEgressWrapper)
from watch.cli.baseline_framework_ingress import (load_input_stac_items,
                                                  ingress_item)
from watch.cli.baseline_framework_egress import upload_output_stac_items
from watch.cli.wv_ortho import ortho_map
from watch.cli.wv_coreg import coreg_map


def main():
    parser = argparse.ArgumentParser(
        description="Run UMD coregistered as baseline framework component")

    parser.add_argument('input_path',
                        type=str,
                        help="Path to input T&E Baseline Framework JSON")
    parser.add_argument('baseline_scenes_bucket',
                        type=str,
                        help="S3 Path to precomputed baseline scenes")
    parser.add_argument('item_association_s3_path',
                        type=str,
                        help="Path to input T&E Baseline Framework JSON")
    parser.add_argument('output_path',
                        type=str,
                        help="S3 path for output JSON")
    parser.add_argument("--aws_profile",
                        required=False,
                        type=str,
                        help="AWS Profile to use for AWS S3 CLI commands")
    parser.add_argument("--aws_storage_class",
                        required=False,
                        type=str,
                        default=None,
                        help="AWS S3 storage class to use for egress AWS S3 "
                             "CLI commands (e.g. 'ONEZONE_IA', default: None)")
    parser.add_argument("-d", "--dryrun",
                        action='store_true',
                        default=False,
                        help="Run AWS CLI commands with --dryrun flag")
    parser.add_argument('-s', '--show-progress',
                        action='store_true',
                        default=False,
                        help='Show progress for AWS CLI commands')
    parser.add_argument("-r", "--requester_pays",
                        action='store_true',
                        default=False,
                        help="Run AWS CLI commands with "
                             "`--requestor_payer requester` flag")
    parser.add_argument("-o", "--outbucket",
                        type=str,
                        required=True,
                        help="S3 Output directory for STAC item / asset "
                             "egress")
    parser.add_argument("-n", "--newline",
                        action='store_true',
                        default=False,
                        help="Output as simple newline separated STAC items")
    parser.add_argument("-j", "--jobs",
                        type=int,
                        default=1,
                        required=False,
                        help="Number of jobs to run in parallel")

    run_wv_ortho_and_coreg_streaming(**vars(parser.parse_args()))

    return 0


SUPPORTED_PLATFORMS = {'DigitalGlobe',
                       'worldview-2',
                       'worldview-3'}  # Worldview


def _item_selector(stac_item):
    return stac_item['properties'].get('platform') in SUPPORTED_PLATFORMS


def _item_map(stac_item, working_dir, baseline_s2_items, item_pairs_dict):
    ortho_item = ortho_map(stac_item,
                           os.path.join(working_dir, 'ortho'),
                           drop_empty=True)

    coreg_item = coreg_map(ortho_item,
                           os.path.join(working_dir, 'wv_coreg'),
                           baseline_s2_items,
                           item_pairs_dict,
                           drop_empty=False)

    return [coreg_item]


def b04_asset_filter(asset_name, asset):
    return (('eo:bands' in asset and
             len(asset['eo:bands']) == 1 and
             asset['eo:bands'][0]['name'] == 'B04') or
            re.match(r'B04(\.tiff?|\.jp2)?', asset_name, re.I))


def run_wv_ortho_and_coreg_streaming(input_path,
                                     baseline_scenes_bucket,
                                     item_association_s3_path,
                                     output_path,
                                     outbucket,
                                     aws_profile=None,
                                     aws_storage_class=None,
                                     dryrun=False,
                                     show_progress=False,
                                     requester_pays=False,
                                     newline=False,
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

    if aws_storage_class is not None:
        aws_base_command.extend(['--storage-class', aws_storage_class])

    input_stac_items = load_input_stac_items(input_path, aws_base_command)

    baseline_s2_items = []
    for input_item in input_stac_items:
        if input_item['properties'].get(
                'watch:s2_coreg_l1c:is_baseline') is True:
            baseline_s2_items.append(input_item)

    baseline_scenes_outdir = os.path.join(
        '/tmp', 'coreg_baseline_scenes')
    # Build list of baseline S2 items
    for stac_item in baseline_s2_items:
        ingress_item(stac_item.to_dict(),
                     baseline_scenes_outdir,
                     aws_base_command,
                     dryrun=dryrun,
                     relative=False,
                     asset_selector=b04_asset_filter)

    with tempfile.NamedTemporaryFile() as temporary_file:
        subprocess.run([*aws_base_command,
                        item_association_s3_path,
                        temporary_file.name], check=True)

        with open(temporary_file.name) as f:
            item_pairs_dict = json.load(f)

    executor = ubelt.Executor(mode='process' if jobs > 1 else 'serial',
                              max_workers=jobs)

    ingress_process_egress_map = IngressProcessEgressWrapper(
        _item_map,
        outbucket,
        aws_base_command,
        dryrun=dryrun,
        stac_item_selector=_item_selector)
    caching_item_map = CacheItemOutputS3Wrapper(
        ingress_process_egress_map,
        outbucket,
        aws_profile=aws_profile)
    coreg_jobs = [executor.submit(caching_item_map,
                                  stac_item,
                                  baseline_s2_items,
                                  item_pairs_dict)
                  for stac_item in input_stac_items]

    output_stac_items = []
    for job in as_completed(coreg_jobs):
        try:
            mapped_item = job.result()
        except Exception:
            print("Exception occurred (printed below), dropping item!")
            traceback.print_exception(*sys.exc_info())
            continue
        else:
            if isinstance(mapped_item, dict):
                output_stac_items.append(mapped_item)
            elif isinstance(mapped_item, pystac.Item):
                output_stac_items.append(mapped_item.to_dict())
            else:
                for mi in mapped_item:
                    if isinstance(mi, dict):
                        output_stac_items.append(mi)
                    elif isinstance(mi, pystac.Item):
                        output_stac_items.append(mi.to_dict())

    te_output = upload_output_stac_items(
        output_stac_items, output_path, aws_base_command, newline=newline)

    return te_output


if __name__ == "__main__":
    sys.exit(main())
