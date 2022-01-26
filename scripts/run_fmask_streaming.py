import argparse
import sys
import traceback
import os
import json
import tempfile
import subprocess
from concurrent.futures import as_completed

import ubelt
import pystac

from watch.cli.baseline_framework_ingress import ingress_item
from watch.cli.baseline_framework_egress import egress_item
from watch.cli.run_fmask import run_fmask_for_item
from watch.cli.add_angle_bands import add_angle_bands_to_item


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

    run_fmask_streaming(**vars(parser.parse_args()))

    return 0


SUPPORTED_PLATFORMS = {'S2A',
                       'S2B',
                       'sentinel-2a',
                       'sentinel-2b',
                       'LANDSAT_8',
                       'OLI_TIRS'}


def _item_map(stac_item, outbucket, aws_base_command, dryrun):
    with tempfile.TemporaryDirectory() as tmpdirname:
        status_file_basename = '{}.done'.format(stac_item['id'])
        status_item_s3_path = os.path.join(
            outbucket, 'status', status_file_basename)
        status_item_local_path = os.path.join(
            tmpdirname, status_file_basename)

        try:
            subprocess.run([*aws_base_command,
                            status_item_s3_path,
                            status_item_local_path],
                           check=True)
        except subprocess.CalledProcessError:
            pass
        else:
            print("* Item: {} previously processed, not "
                  "re-processing".format(stac_item['id']))
            with open(status_item_local_path, 'r') as f:
                return json.load(f)

        print("* Processing item: {}".format(stac_item['id']))

        if stac_item['properties'].get('platform') not in SUPPORTED_PLATFORMS:
            output_item = stac_item
        else:
            ingressed_item = ingress_item(
                stac_item,
                os.path.join(tmpdirname, 'ingress'),
                aws_base_command,
                dryrun)

            fmask_item = run_fmask_for_item(
                ingressed_item,
                os.path.join(tmpdirname, 'fmask'))

            angles_item = add_angle_bands_to_item(
                fmask_item,
                os.path.join(tmpdirname, 'angles'))

            output_item = egress_item(angles_item,
                                      outbucket,
                                      aws_base_command)

        output_status_file = os.path.join(
            tmpdirname, '{}.output.done'.format(stac_item['id']))
        with open(output_status_file, 'w') as outf:
            if isinstance(output_item, pystac.Item):
                print(json.dumps(output_item.to_dict()), file=outf)
            else:
                print(json.dumps(output_item.to_dict()), file=outf)

            subprocess.run([*aws_base_command,
                            output_status_file,
                            status_item_s3_path], check=True)

        return output_item


def run_fmask_streaming(input_path,
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

    def _load_input(path):
        try:
            with open(path) as f:
                input_json = json.load(f)
            return input_json['stac'].get('features', [])
        # Excepting KeyError here in case of a single line STAC item input
        except (json.decoder.JSONDecodeError, KeyError):
            # Support for simple newline separated STAC items
            with open(path) as f:
                return [json.loads(line) for line in f]

    if input_path.startswith('s3'):
        with tempfile.NamedTemporaryFile() as temporary_file:
            subprocess.run(
                [*aws_base_command, input_path, temporary_file.name],
                check=True)

            input_stac_items = _load_input(temporary_file.name)
    else:
        input_stac_items = _load_input(input_path)

    executor = ubelt.Executor(mode='process' if jobs > 1 else 'serial',
                              max_workers=jobs)

    fmask_jobs = [executor.submit(_item_map,
                                  stac_item,
                                  outbucket,
                                  aws_base_command,
                                  dryrun)
                  for stac_item in input_stac_items]

    output_stac_items = []
    for job in as_completed(fmask_jobs):
        try:
            mapped_item = job.result()
        except Exception:
            print("Exception occurred (printed below), dropping item!")
            traceback.print_exception(*sys.exc_info())
            continue
        else:
            if isinstance(mapped_item, dict):
                output_stac_items.append(mapped_item)
            else:
                output_stac_items.append(mapped_item.to_dict())

    if newline:
        te_output = '\n'.join((json.dumps(item) for item in output_stac_items))
    else:
        te_output = {'raw_images': [],
                     'stac': {
                         'type': 'FeatureCollection',
                         'features': output_stac_items}}

    with tempfile.NamedTemporaryFile() as temporary_file:
        with open(temporary_file.name, 'w') as f:
            if newline:
                print(te_output, end='', file=f)
            else:
                print(json.dumps(te_output, indent=2), file=f)

        command = [*aws_base_command, temporary_file.name, output_path]

        subprocess.run(command, check=True)

    return te_output


if __name__ == "__main__":
    sys.exit(main())
