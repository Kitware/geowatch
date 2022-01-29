import argparse
import sys
import re
import glob
import os
import json
import shutil
import subprocess

import pystac

from watch.cli.baseline_framework_ingress import (
    baseline_framework_ingress, ingress_item)
from watch.cli.s2_coreg import (compute_baseline_scenes_only,
                                SUPPORTED_S2_PLATFORMS)


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

    run_coreg_for_baseline(**vars(parser.parse_args()))

    return 0


def s2_stac_item_filter(stac_item):
    return stac_item['properties'].get('platform') in SUPPORTED_S2_PLATFORMS


def mtd_asset_filter(asset_name, asset):
    return asset_name == 'metadata'


def b04_asset_filter(asset_name, asset):
    return (('eo:bands' in asset and
             len(asset['eo:bands']) == 1 and
             asset['eo:bands'][0]['name'] == 'B04') or
            re.match(r'B04(\.tiff?|\.jp2)?', asset_name, re.I))


def run_coreg_for_baseline(input_path,
                           output_path,
                           outbucket,
                           aws_profile=None,
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

    ingress_dir = '/tmp/ingress'
    baseline_scenes_outdir = '/tmp/coreg_baseline_scenes'
    os.makedirs(baseline_scenes_outdir, exist_ok=True)

    print("* Running ingress *")
    ingress_catalog = baseline_framework_ingress(
        input_path,
        ingress_dir,
        aws_profile=aws_profile,
        dryrun=dryrun,
        requester_pays=requester_pays,
        relative=False,
        jobs=jobs,
        item_selector=s2_stac_item_filter,
        asset_selector=mtd_asset_filter)

    print("* Running coregistration *")
    coreg_baseline_scenes = compute_baseline_scenes_only(
        ingress_catalog,
        baseline_scenes_outdir)

    # Gather up baseline scene STAC items
    baseline_scene_stac_items = []
    for baseline_scene in coreg_baseline_scenes.values():
        stac_item = None
        for json_path in glob.glob(os.path.join(baseline_scene, '*.json')):
            try:
                stac_item = pystac.Item.from_file(json_path)
            except pystac.errors.STACTypeError:
                continue

        if stac_item is None:
            raise RuntimeError("Couldn't find STAC item for baseline scene")
        else:
            baseline_scene_stac_items.append(stac_item)

    for stac_item in baseline_scene_stac_items:
        # Write out original STAC item for baseline scene as well
        stac_item_outpath = os.path.join(
            baseline_scenes_outdir,
            stac_item.id, "{}.json".format(stac_item.id))
        os.makedirs(os.path.dirname(stac_item_outpath), exist_ok=True)
        with open(stac_item_outpath, 'w') as f:
            json.dump(stac_item.to_dict(), f)

        ingress_item(stac_item.to_dict(),
                     baseline_scenes_outdir,
                     aws_base_command,
                     dryrun=dryrun,
                     relative=False,
                     asset_selector=b04_asset_filter)

        shutil.copy(
            os.path.join(ingress_dir, stac_item.id, 'MTD_TL.xml'),
            os.path.join(baseline_scenes_outdir, stac_item.id, 'MTD_TL.xml'))

    # Update baseline scene paths prior to upload
    coreg_baseline_scenes = {k: os.path.join(baseline_scenes_outdir,
                                             os.path.basename(v))
                             for k, v in coreg_baseline_scenes.items()}

    with open(os.path.join(baseline_scenes_outdir,
                           'baseline_scenes.json'), 'w') as f:
        json.dump(coreg_baseline_scenes, f, indent=2)

    subprocess.run([*aws_base_command, '--recursive',
                    baseline_scenes_outdir, outbucket],
                   check=True)

    subprocess.run([*aws_base_command, input_path, output_path])


if __name__ == "__main__":
    sys.exit(main())
