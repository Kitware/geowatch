import argparse
import sys
import os
import subprocess

from watch.cli.pseudolive_consolidate import pseudolive_consolidate


def main():
    parser = argparse.ArgumentParser(
        description="Run pseudolive consolidation script for TA-2 "
                    "region / site model outputs")

    parser.add_argument('region_id',
                        type=str,
                        help="Region ID")
    parser.add_argument('previous_consolidated_output',
                        type=str,
                        help="S3 path to consolidated regions / sites from "
                             "previous iteration")
    parser.add_argument('current_output',
                        type=str,
                        help="S3 path to regions / sites from current "
                             "iteration")
    parser.add_argument("--aws_profile",
                        required=False,
                        type=str,
                        help="AWS Profile to use for AWS S3 CLI commands")
    parser.add_argument("-o", "--outbucket",
                        type=str,
                        required=True,
                        help="S3 Output directory for STAC item / asset "
                             "egress")
    parser.add_argument("-s", "--performer-suffix",
                        type=str,
                        default=None,
                        help="Performer suffix if present, e.g. KIT")
    parser.add_argument("-i", "--iou-threshold",
                        type=float,
                        default=0.5,
                        help="IOU Threshold for determining duplicates"
                             "(default: 0.5)")

    run_pseudolive_consolidate(**vars(parser.parse_args()))

    return 0


def run_pseudolive_consolidate(region_id,
                               previous_consolidated_output,
                               current_output,
                               outbucket,
                               iou_threshold,
                               aws_profile=None,
                               performer_suffix=None):
    if aws_profile is not None:
        aws_base_command =\
            ['aws', 's3', '--profile', aws_profile, 'cp']
    else:
        aws_base_command = ['aws', 's3', 'cp']

    local_previous_dir = os.path.join('/tmp', 'previous_data')
    local_current_dir = os.path.join('/tmp', 'current_data')

    subprocess.run([*aws_base_command, '--recursive',
                    previous_consolidated_output, local_previous_dir],
                   check=True)

    # Quickly check and short-circuit if no "previous" data was copied
    # over (e.g. if there was no previous data)
    if len(os.listdir(local_previous_dir)) == 0:
        subprocess.run([*aws_base_command, '--recursive',
                        current_output, outbucket],
                       check=True)
        return

    subprocess.run([*aws_base_command, '--recursive',
                    current_output, local_current_dir],
                   check=True)

    local_consolidated_dir = os.path.join('/tmp', 'consolidated_out')
    pseudolive_consolidate(region_id,
                           local_previous_dir,
                           local_current_dir,
                           local_consolidated_dir,
                           iou_threshold,
                           performer_suffix=performer_suffix)

    subprocess.run([*aws_base_command, '--recursive',
                    local_consolidated_dir, outbucket],
                   check=True)


if __name__ == "__main__":
    sys.exit(main())
