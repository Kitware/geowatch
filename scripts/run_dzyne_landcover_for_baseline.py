import argparse
import sys
import os
import subprocess

from watch.cli.baseline_framework_kwcoco_egress import baseline_framework_kwcoco_egress  # noqa: 501
from watch.cli.baseline_framework_kwcoco_ingress import baseline_framework_kwcoco_ingress  # noqa: 501
from watch.utils.util_framework import download_region


def main():
    parser = argparse.ArgumentParser(
        description="Run DZYNE landcover feature computation as baseline "
                    "framework component")

    parser.add_argument('input_path',
                        type=str,
                        help="Path to input T&E Baseline Framework JSON")
    parser.add_argument('input_region_path',
                        type=str,
                        help="Path to input T&E Baseline Framework Region "
                             "definition JSON")
    parser.add_argument('output_path',
                        type=str,
                        help="S3 path for output JSON")
    parser.add_argument("--landcover_model_path",
                        required=True,
                        type=str,
                        help="File path to DZYNE landcover model")
    parser.add_argument("--aws_profile",
                        required=False,
                        type=str,
                        help="AWS Profile to use for AWS S3 CLI commands")
    parser.add_argument("-d", "--dryrun",
                        action='store_true',
                        default=False,
                        help="Run AWS CLI commands with --dryrun flag")
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

    run_dzyne_landcover_for_baseline(**vars(parser.parse_args()))

    return 0


def run_dzyne_landcover_for_baseline(input_path,
                                     input_region_path,
                                     output_path,
                                     landcover_model_path,
                                     outbucket,
                                     aws_profile=None,
                                     dryrun=False,
                                     newline=False,
                                     jobs=1):
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
    local_region_path = download_region(input_region_path,
                                        local_region_path,
                                        aws_profile=aws_profile,
                                        strip_nonregions=True)

    # 3. Generate DZYNE landcover features
    print("* Generating DZYNE landcover features*")
    dzyne_landcover_features_kwcoco_path = os.path.join(
        ingress_dir, 'dzyne_landcover_kwcoco.json')

    subprocess.run(['python', '-m', 'watch.tasks.landcover.predict',
                    '--dataset', ingress_kwcoco_path,
                    '--deployed', landcover_model_path,
                    '--output', dzyne_landcover_features_kwcoco_path,
                    '--num_workers', 'auto',
                    '--device', '0'],
                   check=True)

    # 4. Egress (envelop KWCOCO dataset in a STAC item and egress;
    #    will need to recursive copy the kwcoco output directory up to
    #    S3 bucket)
    print("* Egressing KWCOCO dataset and associated STAC item *")
    baseline_framework_kwcoco_egress(dzyne_landcover_features_kwcoco_path,
                                     local_region_path,
                                     output_path,
                                     outbucket,
                                     aws_profile=None,
                                     dryrun=False,
                                     newline=False)


if __name__ == "__main__":
    sys.exit(main())
