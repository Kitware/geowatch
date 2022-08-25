import sys
import argparse
import json
import subprocess
import tempfile

import pystac

from watch.cli.baseline_framework_egress import upload_output_stac_items
from watch.cli.baseline_framework_ingress import load_input_stac_items
from watch.stac.util_stac import associate_msi_pan


def main():
    parser = argparse.ArgumentParser(
        description="Compute S2 baseline scenes for coregistration")

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

    run_associate_wv_msi_pan(**vars(parser.parse_args()))

    return 0


def run_associate_wv_msi_pan(input_path,
                             output_path,
                             outbucket,
                             aws_profile=None,
                             dryrun=False,
                             requester_pays=False,
                             newline=False):
    if aws_profile is not None:
        aws_base_command =\
            ['aws', 's3', '--profile', aws_profile, 'cp']
    else:
        aws_base_command = ['aws', 's3', 'cp']

    if dryrun:
        aws_base_command.append('--dryrun')

    if requester_pays:
        aws_base_command.extend(['--request-payer', 'requester'])

    input_stac_items = [pystac.Item.from_dict(item) for item in
                        load_input_stac_items(input_path, aws_base_command)]
    input_stac_items_dict = {}
    for item in input_stac_items:
        item.remove_links('root')
        input_stac_items_dict[item.id] = item.to_dict()

    # Establish the MSI-PAN mapping in the WV catalog
    # and remove the paired PAN items so they don't get duplicated
    item_pairs_dict = associate_msi_pan(input_stac_items)
    for item in item_pairs_dict.values():
        del input_stac_items_dict[item.id]

    # Remove 'root' link (can cause errors with `.to_dict()` calls)
    # and convert pystac.Item values to dicts
    output_item_pairs_dict = {}
    for item_id, paired_item in item_pairs_dict.items():
        paired_item.remove_links('root')
        output_item_pairs_dict[item_id] = paired_item.to_dict()

    with tempfile.NamedTemporaryFile() as temporary_file:
        with open(temporary_file.name, 'w') as f:
            json.dump(output_item_pairs_dict, f, indent=2)

        subprocess.run([*aws_base_command,
                        temporary_file.name,
                        '/'.join((outbucket,
                                  'associated_wv_msi_and_pan_items.json'))],
                       check=True)

    te_output = upload_output_stac_items(
        list(input_stac_items_dict.values()),
        output_path,
        aws_base_command,
        newline=newline)

    return te_output


if __name__ == "__main__":
    sys.exit(main())
