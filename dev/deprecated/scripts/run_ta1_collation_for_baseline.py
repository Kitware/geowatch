import argparse
import sys

from watch.cli.collate_ta1_output import collate_ta1_output
from watch.cli.baseline_framework_ingress import baseline_framework_ingress


def main():
    parser = argparse.ArgumentParser(
        description="Run TA-1 output collation script as baseline "
                    "framework component")

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
                        help="S3 bucket for collated output")
    parser.add_argument("--performer_code",
                        default='kit',
                        type=str,
                        help="Performer code suffix for output "
                             "directories / files (default: 'kit')")
    parser.add_argument("--eval_num",
                        default='1',
                        type=str,
                        help="Evaluation number string for building "
                             "output paths (default: '1')")
    parser.add_argument("-j", "--jobs",
                        type=int,
                        default=1,
                        required=False,
                        help="Number of jobs to run in parallel")

    run_ta1_collation_for_baseline(**vars(parser.parse_args()))

    return 0


def run_ta1_collation_for_baseline(input_path,
                                   output_path,
                                   outbucket,
                                   aws_profile=None,
                                   dryrun=False,
                                   requester_pays=False,
                                   performer_code='kit',
                                   eval_num='1',
                                   jobs=1):
    print("* Running ingress *")
    ingress_catalog = baseline_framework_ingress(
        input_path,
        '/tmp/ingress',
        aws_profile=aws_profile,
        dryrun=dryrun,
        requester_pays=requester_pays,
        relative=False,
        jobs=jobs)

    print("* Running TA-1 Collation *")
    collate_ta1_output(
        ingress_catalog,
        outbucket,
        aws_profile=aws_profile,
        dryrun=dryrun,
        performer_code=performer_code,
        eval_num=eval_num,
        jobs=jobs)


if __name__ == "__main__":
    sys.exit(main())
