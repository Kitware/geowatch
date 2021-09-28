import argparse
import sys
import json
import os
import tempfile
import subprocess

import pystac


def main():
    parser = argparse.ArgumentParser(
        description="Align all STAC item data assets to the same CRS")

    parser.add_argument('input_path',
                        type=str,
                        help="Path to input T&E Baseline Framework JSON")
    parser.add_argument("-o", "--outdir",
                        type=str,
                        help="Output directory for ingressed assets an output "
                             "STAC Catalog")
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

    baseline_framework_ingress(**vars(parser.parse_args()))

    return 0


def baseline_framework_ingress(input_path,
                               outdir,
                               aws_profile=None,
                               dryrun=False,
                               requester_pays=False):
    os.makedirs(outdir, exist_ok=True)

    catalog_outpath = os.path.join(outdir, 'catalog.json')
    catalog = pystac.Catalog('Baseline Framework ingress catalog',
                             'STAC catalog of SMART search results',
                             href=catalog_outpath)
    catalog.set_root(catalog)

    input_path = input_path

    if aws_profile is not None:
        aws_base_command =\
            ['aws', 's3', '--profile', aws_profile]
    else:
        aws_base_command = ['aws', 's3']

    if requester_pays:
        requester_args = ['--request-payer', 'requester']
    else:
        requester_args = []

    if input_path.startswith('s3'):
        with tempfile.NamedTemporaryFile() as temporary_file:
            subprocess.run(
                [*aws_base_command, 'cp', *requester_args, input_path,
                 temporary_file.name],
                check=True)

            with open(temporary_file.name) as f:
                input_json = json.load(f)
    else:
        with open(input_path) as f:
            input_json = json.load(f)

    input_stac = input_json['stac']
    for feature in input_stac.get('features', ()):
        for asset_name, asset in feature.get('assets').items():
            if asset_name == "index":
                continue

            asset_basename = os.path.basename(asset['href'])

            feature_output_dir = os.path.join(
                outdir, feature['id'])
            asset_outpath = os.path.join(
                feature_output_dir, asset_basename)

            if aws_profile is not None:
                command =\
                    ['aws', 's3', '--profile', aws_profile, 'cp']
            else:
                command = ['aws', 's3', 'cp']

            if dryrun:
                command.append('--dryrun')
            else:
                os.makedirs(feature_output_dir, exist_ok=True)

            command.extend([*requester_args, asset['href'], asset_outpath])

            # TODO: Manually check return code / output
            print("Running: {}".format(' '.join(command)))
            subprocess.run(command, check=True)

            # Update feature asset href to point to local outpath
            asset['href'] = asset_outpath

        item = pystac.Item.from_dict(feature)
        item.set_collection(None)  # Clear the collection if present
        item.set_self_href(os.path.join(outdir,
                                        feature['id'],
                                        feature['id'] + '.json'))
        catalog.add_item(item)

    catalog.save(catalog_type=pystac.CatalogType.ABSOLUTE_PUBLISHED)

    return catalog


if __name__ == "__main__":
    sys.exit(main())
