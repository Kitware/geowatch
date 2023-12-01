#!/usr/bin/env python3
"""
See Old Version:
    ~/code/watch/scripts/run_copy_previous_region_model_pseudolive.py

SeeAlso:
    ~/code/watch-smartflow-dags/KIT_TA2_PREEVAL10_PYENV_V13.py
"""
import os
import subprocess
import json
import ubelt as ub
import scriptconfig as scfg

from geowatch.utils.util_framework import download_region


class PsuedoliveCopyPreviousConfig(scfg.DataConfig):
    """
    Download, update, and upload region model for pseudo-live
    """
    input_region_path = scfg.Value(None, type=str, position=1, required=True, help=ub.paragraph(
            '''
            Path to input T&E Baseline Framework Region definition JSON
            '''))
    aws_profile = scfg.Value(None, type=str, help=ub.paragraph(
            '''
            AWS Profile to use for AWS S3 CLI commands
            '''))
    outbucket = scfg.Value(None, type=str, required=True, short_alias=['o'], help=ub.paragraph(
            '''
            S3 Output directory for STAC item / asset egress
            '''))
    updated_end_date = scfg.Value(None, type=str, required=True, short_alias=['e'], help=ub.paragraph(
            '''
            Set uploaded region model end date to this value
            '''))


def main():
    config = PsuedoliveCopyPreviousConfig.cli(strict=True)
    print('config = {}'.format(ub.urepr(dict(config), nl=1, align=':')))
    run_copy_previous_region_model(**config)


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
        aws_base_command = ['aws', 's3', '--profile', aws_profile, 'cp']
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
    main()
