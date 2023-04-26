#!/usr/bin/env python3
"""
See Old Version:
    ../../../scripts/run_sc_fusion_eval3_for_baseline.py

SeeAlso:
    ~/code/watch-smartflow-dags/KIT_TA2_PREEVAL10_PYENV.py
"""
import json
import os
import shutil
import subprocess
import sys
import traceback

import scriptconfig as scfg
import ubelt as ub
from watch.mlops.smart_pipeline import DinoBoxDetector, DinoBuildingFilter

from glob import glob


class DinoSVConfig(scfg.DataConfig):
    """
    Run TA-2 SC fusion as baseline framework component
    """
    input_path = scfg.Value(None, type=str, position=1, required=True, help=ub.paragraph(
            '''
            Path to input T&E Baseline Framework JSON
            '''))
    input_region_path = scfg.Value(None, type=str, position=2, required=True, help=ub.paragraph(
            '''
            Path to input T&E Baseline Framework Region definition JSON
            '''))
    output_path = scfg.Value(None, type=str, position=3, required=True, help='S3 path for output JSON')
    aws_profile = scfg.Value(None, type=str, help=ub.paragraph(
            '''
            AWS Profile to use for AWS S3 CLI commands
            '''))
    dryrun = scfg.Value(False, isflag=True, short_alias=['d'], help='Run AWS CLI commands with --dryrun flag')
    outbucket = scfg.Value(None, type=str, required=True, short_alias=['o'], help=ub.paragraph(
            '''
            S3 Output directory for STAC item / asset egress
            '''))
    dino_detect_config = scfg.Value(None, type=str, help=ub.paragraph(
            '''
            Raw json/yaml or a path to a json/yaml file that specifies the
            config for DinoBoxDetector.
            '''))
    dino_filter_config = scfg.Value(None, type=str, help=ub.paragraph(
            '''
            Raw json/yaml or a path to a json/yaml file that specifies the
            config for DinoBuildingFilter.
            '''))


def main():
    config = DinoSVConfig.cli(strict=True)
    print('config = {}'.format(ub.urepr(dict(config), nl=1, align=':')))
    run_dino_sv(config)


def _upload_region(aws_base_command,
                   local_region_dir,
                   local_input_region_path,
                   destination_region_s3):
    with open(local_input_region_path) as f:
        region = json.load(f)

    region_id = None
    for feature in region.get('features', ()):
        props = feature['properties']
        if props['type'] == 'region':
            region_id = props.get('region_model_id', props.get('region_id'))
            break

    if region_id is not None:
        updated_region_path = os.path.join(local_region_dir,
                                           '{}.geojson'.format(region_id))

        print("** Uploading updated region file")
        subprocess.run([*aws_base_command,
                        updated_region_path, destination_region_s3],
                       check=True)
    else:
        print("** Error: Couldn't parse region_id from region file, "
              "not uploading")


def run_dino_sv(config):
    from watch.cli.baseline_framework_kwcoco_ingress import baseline_framework_kwcoco_ingress
    from watch.cli.baseline_framework_kwcoco_egress import baseline_framework_kwcoco_egress
    from watch.utils.util_framework import download_region, determine_region_id
    from watch.utils.util_yaml import Yaml

    input_path = config.input_path
    input_region_path = config.input_region_path
    output_path = config.output_path

    outbucket = config.outbucket
    aws_profile = config.aws_profile
    dryrun = config.dryrun

    dino_detect_config = config.dino_detect_config
    dino_filter_config = config.dino_filter_config

    if aws_profile is not None:
        aws_base_command =\
            ['aws', 's3', '--profile', aws_profile, 'cp']
    else:
        aws_base_command = ['aws', 's3', 'cp']

    if dryrun:
        aws_base_command.append('--dryrun')

    # 1. Ingress data
    print("* Running baseline framework kwcoco ingress *")
    ingress_dir = '/tmp/ingress'
    ingress_kwcoco_path = baseline_framework_kwcoco_ingress(
        input_path,
        ingress_dir,
        aws_profile,
        dryrun)

    # # 2. Download and prune region file
    print("* Downloading and pruning region file *")
    local_region_path = '/tmp/region.json'

    download_region(
        input_region_path=input_region_path,
        output_region_path=local_region_path,
        aws_profile=aws_profile,
        strip_nonregions=True,
    )

    # Determine the region_id in the region file.
    region_id = determine_region_id(local_region_path)

    dino_boxes_kwcoco_path = os.path.join(
        ingress_dir, 'dino_boxes_kwcoco.json')

    cropped_region_models_bas = os.path.join(ingress_dir,
                                             'cropped_region_models_bas')
    cropped_site_models_bas = os.path.join(ingress_dir,
                                           'cropped_site_models_bas')

    site_models_outdir = os.path.join(ingress_dir, 'sv_out_site_models')
    os.makedirs(site_models_outdir, exist_ok=True)
    region_models_outdir = os.path.join(ingress_dir, 'sv_out_region_models')
    os.makedirs(region_models_outdir, exist_ok=True)

    site_models_manifest_outdir = os.path.join(
        ingress_dir, 'tracking_manifests_sv')
    os.makedirs(site_models_manifest_outdir, exist_ok=True)
    site_models_manifest_outpath = os.path.join(
        site_models_manifest_outdir, 'site_models_manifest.json')
    # Copy input region model into region_models outdir to be updated
    # (rather than generated from tracking, which may not have the
    # same bounds as the original)
    shutil.copy(local_region_path, os.path.join(
        region_models_outdir, '{}.geojson'.format(region_id)))

    # 3.1. Check that we have at least one "video" (BAS identified
    # site) to run over; if not skip SV fusion and KWCOCO to GeoJSON
    with open(ingress_kwcoco_path) as f:
        ingress_kwcoco_data = json.load(f)

    if len(ingress_kwcoco_data.get('videos', ())) > 0:
        # 3.2 Run DinoBoxDetector
        print("* Running Dino Detect *")

        default_dino_detect_config = ub.udict({
            'package_fpath': None,
            'batch_size': 1,
            'num_workers': 2,
            'device': 0})
        dino_detect_config = (default_dino_detect_config
                              | Yaml.coerce(dino_detect_config or {}))

        if dino_detect_config.get('package_fpath', None) is None:
            raise ValueError('Requires package_fpath')

        dino_box_detector = DinoBoxDetector(root_dpath='/tmp/ingress')
        dino_box_detector.configure({
            'out_coco_fpath': dino_boxes_kwcoco_path,
            **dino_detect_config})

        ub.cmd(dino_box_detector.command(), check=True)

        # 3.3 Run DinoBuildingFilter
        print("* Running Dino Building Filter *")

        default_dino_filter_config = ub.udict({})
        dino_filter_config = (default_dino_filter_config
                              | Yaml.coerce(dino_filter_config or {}))

        input_region_model_bas = os.path.join(cropped_region_models_bas,
                                              '{}.geojson'.format(region_id))
        output_region_model = os.path.join(region_models_outdir,
                                           '{}.geojson'.format(region_id))
        dino_building_filter = DinoBuildingFilter(root_dpath='/tmp/ingress')
        dino_building_filter.configure({
            'input_kwcoco': dino_boxes_kwcoco_path,
            'input_region': input_region_model_bas,
            'input_sites': cropped_site_models_bas,
            'output_region_fpath': output_region_model,
            'output_sites_dpath': site_models_outdir,
            'output_site_manifest_fpath': site_models_manifest_outpath,
            })

        ub.cmd(dino_building_filter.command(), check=True)

    # 4. Egress (envelop KWCOCO dataset in a STAC item and egress;
    #    will need to recursive copy the kwcoco output directory up to
    #    S3 bucket)
    print("* Egressing KWCOCO dataset and associated STAC item *")
    baseline_framework_kwcoco_egress(dino_boxes_kwcoco_path,
                                     local_region_path,
                                     output_path,
                                     outbucket,
                                     aws_profile=None,
                                     dryrun=False,
                                     newline=False)


if __name__ == "__main__":
    main()
