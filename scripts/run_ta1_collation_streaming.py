import argparse
import sys
import traceback
from concurrent.futures import as_completed
import itertools

import ubelt
import pystac

from watch.cli.baseline_framework_egress import upload_output_stac_items
from watch.cli.collate_ta1_output import (
    collate_item,
    build_and_upload_stac_collections,
    S2_ASSET_NAME_MAP,
    L8_ASSET_NAME_MAP,
    ASSET_SUFFIX_TO_NAME_MAP)
from watch.utils.util_framework import (CacheItemOutputS3Wrapper,
                                        IngressProcessEgressWrapper)
from watch.cli.baseline_framework_ingress import load_input_stac_items


def main():
    parser = argparse.ArgumentParser(
        description="Run TA-1 output collation script in a streaming fashion")

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
    parser.add_argument("-u", "--upload-collections",
                        action='store_true',
                        default=False,
                        help="Build and upload STAC Collections from "
                             "collated items")
    parser.add_argument('-s', '--show-progress',
                        action='store_true',
                        default=False,
                        help='Show progress for AWS CLI commands')
    parser.add_argument("-r", "--requester_pays",
                        action='store_true',
                        default=False,
                        help="Run AWS CLI commands with "
                             "`--requestor_payer requester` flag")
    parser.add_argument("-o", "--destination-outbucket",
                        type=str,
                        required=True,
                        help="S3 bucket for collated output")
    parser.add_argument("-w", "--working-outbucket",
                        type=str,
                        required=True,
                        help="S3 bucket for status files and output STAC list")
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

    run_ta1_collation_streaming(**vars(parser.parse_args()))

    return 0


def _asset_selector(asset_name, asset):
    # WV items only have a single "data" asset containing all bands
    return (asset_name in S2_ASSET_NAME_MAP or
            asset_name in L8_ASSET_NAME_MAP or
            asset_name in ASSET_SUFFIX_TO_NAME_MAP or
            asset_name == 'data')


def run_ta1_collation_streaming(input_path,
                                output_path,
                                destination_outbucket,
                                working_outbucket,
                                aws_profile=None,
                                dryrun=False,
                                upload_collections=False,
                                show_progress=False,
                                requester_pays=False,
                                performer_code='kit',
                                eval_num='1',
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

    input_stac_items = load_input_stac_items(input_path, aws_base_command)

    executor = ubelt.Executor(mode='process' if jobs > 1 else 'serial',
                              max_workers=jobs)

    # Skipping ingress / egress here as the collation function performs a
    # specialized ingress / egress
    ingress_process_egress_map = IngressProcessEgressWrapper(
        collate_item,
        working_outbucket,
        aws_base_command,
        dryrun=dryrun,
        asset_selector=_asset_selector,
        skip_egress=True)
    caching_item_map = CacheItemOutputS3Wrapper(
        ingress_process_egress_map,
        working_outbucket,
        aws_profile=aws_profile)
    collation_jobs = [executor.submit(caching_item_map,
                                      stac_item,
                                      aws_base_command,
                                      destination_outbucket,
                                      performer_code,
                                      eval_num)
                      for stac_item in input_stac_items]

    output_stac_items_by_collection = {}
    for collation_job in as_completed(collation_jobs):
        try:
            stac_item = collation_job.result()
        except Exception:
            print("Exception occurred (printed below), dropping item!")
            traceback.print_exception(*sys.exc_info())
            continue
        else:
            if isinstance(stac_item, dict):
                stac_item = pystac.Item.from_dict(stac_item)
                output_stac_items_by_collection.setdefault(
                    stac_item.collection_id, []).append(stac_item)
            elif isinstance(stac_item, pystac.Item):
                output_stac_items_by_collection.setdefault(
                    stac_item.collection_id, []).append(stac_item)
            else:
                for si in stac_item:
                    if isinstance(si, dict):
                        si = pystac.Item.from_dict(si)
                        output_stac_items_by_collection.setdefault(
                            si.collection_id, []).append(si)
                    elif isinstance(si, pystac.Item):
                        output_stac_items_by_collection.setdefault(
                            si.collection_id, []).append(si)

    if upload_collections:
        build_and_upload_stac_collections(output_stac_items_by_collection,
                                          aws_base_command,
                                          destination_outbucket,
                                          performer_code)

    output_stac_items = [item.to_dict() for item in
                         itertools.chain(
                             *output_stac_items_by_collection.values())]

    te_output = upload_output_stac_items(
        output_stac_items, output_path, aws_base_command, newline=True)

    return te_output


if __name__ == "__main__":
    sys.exit(main())
