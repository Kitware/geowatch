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


class TeamFeatInvariantsConfig(scfg.DataConfig):
    """
    Run UKY invariant feature computation as baseline framework component
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
    model_path = scfg.Value(None, type=str, required=True, help='File path to UKY invariants model')
    pca_projection_path = scfg.Value(None, type=str, required=True, help=ub.paragraph(
            '''
            File path to UKY invariants PCA projections file
            '''))
    aws_profile = scfg.Value(None, type=str, help=ub.paragraph(
            '''
            AWS Profile to use for AWS S3 CLI commands
            '''))
    dryrun = scfg.Value(False, isflag=True, short_alias=['d'], help='Run AWS CLI commands with --dryrun flag')
    outbucket = scfg.Value(None, type=str, required=True, short_alias=['o'], help=ub.paragraph(
            '''
            S3 Output directory for STAC item / asset egress
            '''))
    do_pca = scfg.Value(False, isflag=True, help='Perform PCA on invariants')
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
    config = TeamFeatInvariantsConfig.cli(strict=True)
    print('config = {}'.format(ub.urepr(dict(config), nl=1, align=':')))
    run_uky_invariants_for_baseline(**config)


def run_uky_invariants_for_baseline(input_path,
                                    input_region_path,
                                    output_path,
                                    model_path,
                                    pca_projection_path,
                                    outbucket,
                                    do_pca=False,
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

    # 2. Generate Invariants
    print("* Generating UKY invariant features for L8 *")
    sc_invariants_kwcoco_path = os.path.join(
        ingress_dir, 'sc_invariants_kwcoco.json')
    subprocess.run(['python', '-m', 'watch.tasks.invariants.predict',
                    '--input_kwcoco', ingress_kwcoco_path,
                    '--output_kwcoco', sc_invariants_kwcoco_path,
                    '--pretext_package_path', model_path,
                    '--pca_projection_path', pca_projection_path,
                    '--input_space_scale', '30GSD',
                    '--window_space_scale', '30GSD',
                    '--patch_size', '256',
                    '--do_pca', str(1 if do_pca else 0),
                    '--patch_overlap', '0.3',
                    '--num_workers', '2',
                    '--write_workers', '0',
                    '--tasks', 'before_after', 'pretext'],
                   check=True)

    # 3. Egress (envelop KWCOCO dataset in a STAC item and egress;
    #    will need to recursive copy the kwcoco output directory up to
    #    S3 bucket)
    print("* Egressing KWCOCO dataset and associated STAC item *")
    baseline_framework_kwcoco_egress(sc_invariants_kwcoco_path,
                                     local_region_path,
                                     output_path,
                                     outbucket,
                                     aws_profile=None,
                                     dryrun=False,
                                     newline=False)


if __name__ == "__main__":
    main()
