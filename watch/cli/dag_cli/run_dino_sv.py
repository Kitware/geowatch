#!/usr/bin/env python3
"""
See Old Version:
    ../../../scripts/run_sc_fusion_eval3_for_baseline.py

SeeAlso:
    ~/code/watch-smartflow-dags/KIT_TA2_PREEVAL10_PYENV.py
"""
import json
import os
import subprocess

import scriptconfig as scfg
import ubelt as ub
from watch.mlops.smart_pipeline import DinoBoxDetector, SV_DinoFilter

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
    ta2_s3_collation_bucket = scfg.Value(None, type=str, help=ub.paragraph(
            '''
            S3 Location for collated TA-2 output (bucket name should
            include up to eval name)
            '''))
    dino_detect_config = scfg.Value(None, type=str, help=ub.paragraph(
            '''
            Raw json/yaml or a path to a json/yaml file that specifies the
            config for DinoBoxDetector.
            '''))
    dino_filter_config = scfg.Value(None, type=str, help=ub.paragraph(
            '''
            Raw json/yaml or a path to a json/yaml file that specifies the
            config for SV_DinoFilter.
            '''))


def main():
    config = DinoSVConfig.cli(strict=True)
    print('config = {}'.format(ub.urepr(dict(config), nl=1, align=':')))
    run_dino_sv(config)


def _ta2_collate_output(aws_base_command,
                        local_region_dir,
                        local_sites_dir,
                        destination_s3_bucket,
                        performer_suffix='KIT'):
    def _get_suffixed_basename(local_path):
        base, ext = os.path.splitext(os.path.basename(local_path))
        return "{}_{}{}".format(base, performer_suffix, ext)

    for region in glob(os.path.join(local_region_dir, '*.geojson')):
        region_s3_outpath = '/'.join((destination_s3_bucket,
                                      'region_models',
                                      _get_suffixed_basename(region)))
        subprocess.run([*aws_base_command,
                        region,
                        region_s3_outpath], check=True)

    for site in glob(os.path.join(local_sites_dir, '*.geojson')):
        site_s3_outpath = '/'.join((destination_s3_bucket,
                                    'site_models',
                                    _get_suffixed_basename(site)))
        subprocess.run([*aws_base_command,
                        site,
                        site_s3_outpath], check=True)


def run_dino_sv(config):
    from watch.cli.baseline_framework_kwcoco_ingress import baseline_framework_kwcoco_ingress
    from watch.cli.baseline_framework_kwcoco_egress import baseline_framework_kwcoco_egress
    from watch.utils.util_framework import download_region, determine_region_id
    from watch.utils.util_yaml import Yaml

    input_path = config.input_path
    input_region_path = config.input_region_path
    output_path = config.output_path

    outbucket = config.outbucket
    ta2_s3_collation_bucket = config.ta2_s3_collation_bucket
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
    ingress_dir = ub.Path('/tmp/ingress')
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

    dino_boxes_kwcoco_path = ingress_dir / 'dino_boxes_kwcoco.json'

    # FIXME: these are hard coded to point at the output of DZYNE depth
    # site validation, the path to the region / sites directories should be
    # parameters passed to us from the DAG (so we can shift the order in
    # which operations are applied at the DAG level)
    input_region_dpath = ingress_dir / 'dyzne_parallel_site_vali_region_models'
    input_sites_dpath = ingress_dir / 'filtered_site'
    input_region_fpath = input_region_dpath / f'{region_id}.geojson'

    # FIXME: these output directories for region / site models should be passed
    # to us from the DAG
    output_sites_dpath = ingress_dir / 'sv_out_site_models'
    output_region_dpath = ingress_dir / 'sv_out_region_models'
    output_site_manifest_dpath = ingress_dir / 'tracking_manifests_sv'
    output_region_fpath = output_region_dpath / f'{region_id}.geojson'
    output_site_manifest_fpath = output_site_manifest_dpath / 'site_models_manifest.json'

    output_sites_dpath.ensuredir()
    output_region_dpath.ensuredir()
    output_site_manifest_dpath.ensuredir()

    # Copy input region model into region_models outdir to be updated
    # (rather than generated from tracking, which may not have the
    # same bounds as the original)
    # CHECKME: Is this necessary? I don't think it is, because it will get
    # overwritten in the DINO filter, so I'm commenting it out.
    # local_region_path.copy(output_region_fpath)

    # 3.1. Check that we have at least one "video" (BAS identified
    # site) to run over; if not skip SV fusion and KWCOCO to GeoJSON
    with open(ingress_kwcoco_path) as f:
        ingress_kwcoco_data = json.load(f)

    if len(ingress_kwcoco_data.get('videos', ())) > 0:
        # 3.2 Run DinoBoxDetector
        print("* Running Dino Detect *")

        default_dino_detect_config = ub.udict({
            'coco_fpath': ingress_kwcoco_path,
            'package_fpath': None,
            'batch_size': 1,
            'device': 0})
        dino_detect_config = (default_dino_detect_config
                              | Yaml.coerce(dino_detect_config or {}))

        if dino_detect_config.get('package_fpath', None) is None:
            raise ValueError('Requires package_fpath')

        dino_box_detector = DinoBoxDetector(root_dpath='/tmp/ingress')
        dino_box_detector.configure({
            'out_coco_fpath': dino_boxes_kwcoco_path,
            **dino_detect_config})

        ub.cmd(dino_box_detector.command(), check=True, verbose=3, system=True)

        # 3.3 Run SV_DinoFilter
        print("* Running Dino Building Filter *")

        default_dino_filter_config = ub.udict({})
        dino_filter_config = (default_dino_filter_config
                              | Yaml.coerce(dino_filter_config or {}))

        dino_building_filter = SV_DinoFilter(root_dpath='/tmp/ingress')

        dino_building_filter.configure({
            'input_kwcoco': dino_boxes_kwcoco_path,
            'input_region': input_region_fpath,
            'input_sites': input_sites_dpath,
            'output_region_fpath': output_region_fpath,
            'output_sites_dpath': output_sites_dpath,
            'output_site_manifest_fpath': output_site_manifest_fpath,
            })

        ub.cmd(dino_building_filter.command(), check=True, verbose=3, system=True)

    # 4. (Optional) collate TA-2 output
    if ta2_s3_collation_bucket is not None:
        print("* Collating TA-2 output")
        _ta2_collate_output(aws_base_command,
                            output_region_dpath,
                            output_sites_dpath,
                            ta2_s3_collation_bucket)

    # 5. Egress (envelop KWCOCO dataset in a STAC item and egress;
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
