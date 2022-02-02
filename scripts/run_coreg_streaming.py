import argparse
import sys
import traceback
import os
from concurrent.futures import as_completed
import subprocess
import json

import ubelt
import pystac

from watch.utils.util_framework import (CacheItemOutputS3Wrapper,
                                        IngressProcessEgressWrapper)
from watch.cli.baseline_framework_ingress import load_input_stac_items
from watch.cli.baseline_framework_egress import upload_output_stac_items
from watch.cli.run_brdf import brdf_item_map
from watch.cli.s2_coreg import coreg_stac_item


def main():
    parser = argparse.ArgumentParser(
        description="Run coregistration for LS/S2 as a streaming process")

    parser.add_argument('input_path',
                        type=str,
                        help="Path to input T&E Baseline Framework JSON")
    parser.add_argument('baseline_scenes_bucket',
                        type=str,
                        help="S3 Path to precomputed baseline scenes")
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

    run_coreg_streaming(**vars(parser.parse_args()))

    return 0


SUPPORTED_PLATFORMS = {'S2A',
                       'S2B',
                       'sentinel-2a',
                       'sentinel-2b',
                       'LANDSAT_8',
                       'OLI_TIRS'}


def _item_selector(stac_item):
    return stac_item['properties'].get('platform') in SUPPORTED_PLATFORMS


def _item_map(stac_item, working_dir, baseline_scenes):
    coreg_items = coreg_stac_item(
        stac_item,
        os.path.join(working_dir, 'coreg'),
        baseline_scenes)

    # Coreg returns a list of STAC items per input item (as LS items
    # span multiple MGRS tiles)
    brdf_items = []
    for coreg_item in coreg_items:
        brdf_item = brdf_item_map(
            coreg_item,
            os.path.join(working_dir, 'brdf'))

        brdf_items.append(brdf_item)

    return brdf_items


def run_coreg_streaming(input_path,
                        baseline_scenes_bucket,
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

    # Fetch and parse baseline scenes
    # IMPORTANT * This path should match the path used in
    # `run_compute_coreg_baseline_scenes.py` as STAC items for the
    # baseline scenes will assume files are in this directory
    baseline_scenes_outdir = '/tmp/coreg_baseline_scenes'
    os.makedirs(baseline_scenes_outdir, exist_ok=True)
    subprocess.run([*aws_base_command, '--recursive',
                    baseline_scenes_bucket, baseline_scenes_outdir],
                   check=True)

    baseline_scenes_path =\
        os.path.join(baseline_scenes_outdir, 'baseline_scenes.json')

    with open(baseline_scenes_path) as f:
        baseline_scenes = json.load(f)

    input_stac_items = load_input_stac_items(input_path, aws_base_command)

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
                                  baseline_scenes)
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
