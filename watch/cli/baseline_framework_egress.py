import argparse
import sys
import json
import os
import tempfile
import subprocess

import pystac


def main():
    parser = argparse.ArgumentParser(
        description="Egress data to T&E baseline framework structure")

    parser.add_argument('stac_catalog',
                        type=str,
                        help="Path to input STAC catalog")
    parser.add_argument('output_path',
                        type=str,
                        help="S3 path for output JSON")
    parser.add_argument("-o", "--outbucket",
                        type=str,
                        required=True,
                        help="S3 Output directory for STAC item / asset "
                             "egress")
    parser.add_argument("--aws_profile",
                        required=False,
                        type=str,
                        help="AWS Profile to use for AWS S3 CLI commands")
    parser.add_argument("-d", "--dryrun",
                        action='store_true',
                        default=False,
                        help="Run AWS CLI commands with --dryrun flag")

    baseline_framework_egress(**vars(parser.parse_args()))

    return 0


def baseline_framework_egress(stac_catalog,
                              output_path,
                              outbucket,
                              aws_profile=None,
                              dryrun=False):
    if isinstance(stac_catalog, str):
        catalog = pystac.read_file(href=stac_catalog).full_copy()
    elif isinstance(stac_catalog, dict):
        catalog = pystac.Catalog.from_dict(stac_catalog).full_copy()
    else:
        catalog = stac_catalog.full_copy()

    if aws_profile is not None:
        aws_base_command =\
            ['aws', 's3', '--profile', aws_profile, 'cp']
    else:
        aws_base_command = ['aws', 's3', 'cp']

    if dryrun:
        aws_base_command.append('--dryrun')

    output_stac_items = []
    for stac_item in catalog.get_all_items():
        stac_item_outpath = os.path.join(
            outbucket, "{}.json".format(stac_item.id))
        stac_item.set_self_href(stac_item_outpath)

        assets_outdir = os.path.join(outbucket, stac_item.id)

        stac_item_dict = stac_item.to_dict()
        for asset_name, asset in stac_item_dict.get('assets', {}).items():
            asset_basename = os.path.basename(asset['href'])

            asset_outpath = os.path.join(assets_outdir, asset_basename)

            command = [*aws_base_command, asset['href'], asset_outpath]

            print("Running: {}".format(' '.join(command)))
            # TODO: Manually check return code / output
            subprocess.run(command, check=True)

            # Update feature asset href to point to local outpath
            asset['href'] = asset_outpath

        with tempfile.NamedTemporaryFile() as temporary_file:
            with open(temporary_file.name, 'w') as f:
                print(json.dumps(stac_item_dict, indent=2), file=f)

            command = [*aws_base_command,
                       temporary_file.name, stac_item_outpath]

            subprocess.run(command, check=True)

        output_stac_items.append(stac_item_dict)

    te_output = {'raw_images': [],
                 'stac': {
                     'type': 'FeatureCollection',
                     'features': output_stac_items}}

    with tempfile.NamedTemporaryFile() as temporary_file:
        with open(temporary_file.name, 'w') as f:
            print(json.dumps(te_output, indent=2), file=f)

        command = [*aws_base_command, temporary_file.name, output_path]

        subprocess.run(command, check=True)

    return te_output


if __name__ == "__main__":
    sys.exit(main())
