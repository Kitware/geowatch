import argparse
import sys
import os
import subprocess
import json

from watch.utils.util_framework import download_region


def main():
    parser = argparse.ArgumentParser(
        description="Download, update, and upload region model for "
                    "pseudo-live")

    parser.add_argument('input_region_path',
                        type=str,
                        help="Path to input T&E Baseline Framework Region "
                             "definition JSON")
    parser.add_argument("--aws_profile",
                        required=False,
                        type=str,
                        help="AWS Profile to use for AWS S3 CLI commands")
    parser.add_argument("-o", "--outbucket",
                        type=str,
                        required=True,
                        help="S3 Output directory for STAC item / asset "
                             "egress")
    parser.add_argument("-e", "--updated_end_date",
                        type=str,
                        required=True,
                        help="Set uploaded region model end date to this "
                             "value")

    run_copy_previous_region_model(**vars(parser.parse_args()))

    return 0


def _update_region(region, updated_end_date):
    for feature in region.get('features', ()):
        if feature['properties']['type'] == 'region':
            feature['properties']['end_date'] = updated_end_date
            break

    return region


def run_copy_previous_region_model(input_region_path,
                                   outbucket,
                                   updated_end_date,
                                   aws_profile=None):
    if aws_profile is not None:
        aws_base_command =\
            ['aws', 's3', '--profile', aws_profile, 'cp']
    else:
        aws_base_command = ['aws', 's3', 'cp']

    # 1. Download region file
    print("* Downloading and pruning region file *")
    local_region_path = '/tmp/region.json'
    local_region_path = download_region(input_region_path,
                                        local_region_path,
                                        aws_profile=aws_profile,
                                        strip_nonregions=False)

    with open(local_region_path) as f:
        region = json.load(f)

    updated_region = _update_region(region, updated_end_date)

    local_updated_region_path = '/tmp/updated_region.json'
    with open(local_updated_region_path, 'w') as f:
        print(json.dumps(updated_region, indent=2), file=f)

    # 2. Upload updated region file
    print("* Uploading updated region file *")
    upload_region_path = os.path.join(
        outbucket, os.path.basename(input_region_path))
    subprocess.run([*aws_base_command,
                    local_updated_region_path,
                    upload_region_path], check=True)


if __name__ == "__main__":
    sys.exit(main())
