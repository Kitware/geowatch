#!/usr/bin/env python3
import os
import subprocess
import json

from watch.cli.baseline_framework_kwcoco_ingress import baseline_framework_kwcoco_ingress  # noqa: 501
from watch.cli.baseline_framework_kwcoco_egress import baseline_framework_kwcoco_egress  # noqa: 501
from watch.utils.util_framework import download_region
import ubelt as ub
import scriptconfig as scfg
from watch.mlops.smart_pipeline import SV_Cropping
from watch.utils.util_yaml import Yaml


class SVDatasetConfig(scfg.DataConfig):
    """
    Generate cropped KWCOCO dataset for SC
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
    newline = scfg.Value(False, isflag=True, short_alias=['n'], help=ub.paragraph(
            '''
            Output as simple newline separated STAC items
            '''))
    jobs = scfg.Value(1, type=int, short_alias=['j'], help='Number of jobs to run in parallel')
    dont_recompute = scfg.Value(False, isflag=True, help=ub.paragraph(
            '''
            Will not recompute if output_path already exists
            '''))
    force_one_job_for_cropping = scfg.Value(False, isflag=True, help='Force jobs=1 for cropping')
    sv_cropping_config = scfg.Value(None, type=str, help=ub.paragraph(
            '''
            Raw json/yaml or a path to a json/yaml file that specifies the
            config for SV_Cropping.
            '''))


def main():
    config = SVDatasetConfig.cli(strict=True)
    print('config = {}'.format(ub.urepr(dict(config), nl=1, align=':')))
    run_generate_sv_cropped_kwcoco(**config)


def run_generate_sv_cropped_kwcoco(input_path,
                                   input_region_path,
                                   output_path,
                                   outbucket,
                                   aws_profile=None,
                                   dryrun=False,
                                   newline=False,
                                   jobs=1,
                                   dont_recompute=False,
                                   force_one_job_for_cropping=False,
                                   sv_cropping_config=None):
    if dont_recompute:
        if aws_profile is not None:
            aws_ls_command = ['aws', 's3', '--profile', aws_profile, 'ls']
        else:
            aws_ls_command = ['aws', 's3', 'ls']

        try:
            subprocess.run([*aws_ls_command, output_path], check=True)
        except subprocess.CalledProcessError:
            # Continue processing
            pass
        else:
            # If output_path file was there, nothing to do
            return

    # 1. Ingress data
    print("* Running baseline framework kwcoco ingress *")
    ingress_dir = '/tmp/ingress'
    _ = baseline_framework_kwcoco_ingress(
        input_path,
        ingress_dir,
        aws_profile,
        dryrun)

    # 2. Download and prune region file
    print("* Downloading and pruning region file *")
    local_region_path = '/tmp/region.json'
    local_region_path = download_region(input_region_path,
                                        local_region_path,
                                        aws_profile=aws_profile,
                                        strip_nonregions=True)

    # Parse region_id from original region file
    with open(local_region_path) as f:
        region = json.load(f)

        region_id = None
        for feature in region.get('features', ()):
            props = feature['properties']
            if props['type'] == 'region':
                region_id = props.get('region_model_id',
                                      props.get('region_id'))
                break

    if region_id is None:
        raise RuntimeError("Couldn't parse 'region_id' from input region file")

    # Paths to inputs generated in previous pipeline steps
    bas_region_path = os.path.join(ingress_dir,
                                   'cropped_region_models_bas',
                                   '{}.geojson'.format(region_id))
    ta1_sc_kwcoco_path = os.path.join(ingress_dir,
                                      'kwcoco_for_sc.json')

    # 4. Crop ingress KWCOCO dataset to region for SV
    print("* Cropping KWCOCO dataset to region for SV*")
    ta1_sv_cropped_kwcoco_path = os.path.join(ingress_dir,
                                              'cropped_kwcoco_for_sv.json')

    sv_cropping_config = Yaml.coerce(sv_cropping_config or {})

    sv_cropping = SV_Cropping(root_dpath='/tmp/ingress')
    sv_cropping.configure({
        'crop_src_fpath': ta1_sc_kwcoco_path,
        'regions': bas_region_path,
        'crop_dset_fpath': ta1_sv_cropped_kwcoco_path})

    subprocess.run(sv_cropping.command(), check=True)

    # 5. Egress (envelop KWCOCO dataset in a STAC item and egress;
    #    will need to recursive copy the kwcoco output directory up to
    #    S3 bucket)
    print("* Egressing KWCOCO dataset and associated STAC item *")
    baseline_framework_kwcoco_egress(ta1_sv_cropped_kwcoco_path,
                                     local_region_path,
                                     output_path,
                                     outbucket,
                                     aws_profile=None,
                                     dryrun=False,
                                     newline=False)


if __name__ == "__main__":
    main()
