import argparse
import sys

from watch.cli.baseline_framework_ingress import baseline_framework_ingress
from watch.cli.baseline_framework_egress import baseline_framework_egress
from watch.cli.run_fmask import run_fmask
from watch.cli.add_angle_bands import add_angle_bands


def main():
    parser = argparse.ArgumentParser(
        description="Run fmask as baseline framework component")

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


def run_coreg_for_baseline(input_path,
                           output_path,
                           outbucket,
                           aws_profile=None,
                           dryrun=False,
                           requester_pays=False,
                           newline=False,
                           jobs=1):
    ingress_catalog = baseline_framework_ingress(
        input_path,
        '/tmp/ingress',
        aws_profile,
        dryrun,
        requester_pays,
        jobs)

    fmask_catalog = run_fmask(
        ingress_catalog,
        '/tmp/fmask',
        jobs)

    angle_bands_catalog = add_angle_bands(
        fmask_catalog,
        '/tmp/add_angle_bands',
        jobs)

    te_output = baseline_framework_egress(
        angle_bands_catalog,
        output_path,
        outbucket,
        aws_profile,
        dryrun,
        newline,
        jobs)

    return te_output


if __name__ == "__main__":
    sys.exit(main())
