"""
See Old Version:
    ../../../scripts/run_uky_invariants_for_baseline.py

SeeAlso:
    ~/code/watch-smartflow-dags/KIT_TA2_PREEVAL10_PYENV.py
"""
import os
import subprocess
import ubelt as ub
import scriptconfig as scfg

from watch.cli.baseline_framework_kwcoco_egress import baseline_framework_kwcoco_egress  # noqa: 501
from watch.cli.baseline_framework_kwcoco_ingress import baseline_framework_kwcoco_ingress  # noqa: 501


class TeamFeatLandcoverConfig(scfg.DataConfig):
    """
    Run DZYNE landcover feature computation as baseline framework component
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
    model_path = scfg.Value(None, type=str, required=True, help='File path to landcover model')
    aws_profile = scfg.Value(None, type=str, help=ub.paragraph(
            '''
            AWS Profile to use for AWS S3 CLI commands
            '''))
    dryrun = scfg.Value(False, isflag=True, short_alias=['d'], help='Run AWS CLI commands with --dryrun flag')
    outbucket = scfg.Value(None, type=str, required=True, short_alias=['o'], help=ub.paragraph(
            '''
            S3 Output directory for STAC item / asset egress
            '''))
    requester_pays = scfg.Value(False, isflag=True, short_alias=['r'], help=ub.paragraph(
            '''
            Run AWS CLI commands with `--requestor_payer requester` flag
            '''))
    newline = scfg.Value(False, isflag=True, short_alias=['n'], help=ub.paragraph(
            '''
            Output as simple newline separated STAC items
            '''))
    jobs = scfg.Value(1, type=int, short_alias=['j'], help='Number of jobs to run in parallel')


def main():
    config = TeamFeatLandcoverConfig.cli(strict=True)
    print('config = {}'.format(ub.urepr(dict(config), nl=1, align=':')))
    run_landcover_for_baseline(**config)


def run_landcover_for_baseline(input_path,
                               input_region_path,
                               output_path,
                               model_path,
                               outbucket,
                               aws_profile=None,
                               dryrun=False,
                               requester_pays=False,
                               newline=False,
                               jobs=1):
    from watch.utils.util_framework import download_region
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

    # 2. Download and prune region file
    print("* Downloading and pruning region file *")
    local_region_path = '/tmp/region.json'
    download_region(
        input_region_path=input_region_path,
        output_region_path=local_region_path,
        aws_profile=aws_profile,
        strip_nonregions=True,
    )

    # 2. Generate Landcover features
    print("* Generating landcover features *")
    dzyne_landcover_features_kwcoco_path = os.path.join(
        ingress_dir, 'dzyne_landcover_kwcoco.json')

    subprocess.run(['python', '-m', 'watch.tasks.landcover.predict',
                    '--dataset', ingress_kwcoco_path,
                    '--deployed', model_path,
                    '--output', dzyne_landcover_features_kwcoco_path,
                    '--num_workers', '2',
                    '--with_hidden', '32',
                    '--select_images', '.sensor_coarse == "S2"',
                    '--device', '0'],
                   check=True)

    # 3. Combining landcover features with input features to pass to BAS
    print("* Combining input features with computed landcover features *")
    combo_features_kwcoco_path = os.path.join(
        ingress_dir, 'features_combo_with_landcover_kwcoco.json')
    subprocess.run(['python', '-m', 'watch.cli.coco_combine_features',
                    '--src',
                    ingress_kwcoco_path,
                    dzyne_landcover_features_kwcoco_path,
                    '--dst',
                    combo_features_kwcoco_path],
                   check=True)

    # 4. Egress (envelop KWCOCO dataset in a STAC item and egress;
    #    will need to recursive copy the kwcoco output directory up to
    #    S3 bucket)
    print("* Egressing KWCOCO dataset and associated STAC item *")
    baseline_framework_kwcoco_egress(combo_features_kwcoco_path,
                                     local_region_path,
                                     output_path,
                                     outbucket,
                                     aws_profile=None,
                                     dryrun=False,
                                     newline=False)


if __name__ == "__main__":
    main()
