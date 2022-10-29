import argparse
import sys
import json
import os
import tempfile
import subprocess


def main():
    parser = argparse.ArgumentParser(
        description="Egress KWCOCO data to T&E baseline framework structure")

    parser.add_argument('input_path',
                        type=str,
                        help="Path to input T&E Baseline Framework JSON")
    parser.add_argument("-o", "--outdir",
                        type=str,
                        required=True,
                        help="Output directory for ingressed assets an output "
                             "STAC Catalog")
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

    baseline_framework_kwcoco_ingress(**vars(parser.parse_args()))

    return 0


def baseline_framework_kwcoco_ingress(input_path,
                                      outdir,
                                      aws_profile=None,
                                      dryrun=False,
                                      show_progress=False):
    os.makedirs(outdir, exist_ok=True)

    if aws_profile is not None:
        aws_base_command =\
            ['aws', 's3', '--profile', aws_profile, 'cp']
    else:
        aws_base_command = ['aws', 's3', 'cp']

    if dryrun:
        aws_base_command.append('--dryrun')

    if not show_progress:
        aws_base_command.append('--only-show-errors')

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

    # Our baseline KWCOCO egress script should only ever write out a
    # single KWCOCO STAC item
    if len(input_stac_items) != 1:
        raise RuntimeError("Expecting one and only one STAC item from input")
    kwcoco_stac_item = input_stac_items[0]

    try:
        kwcoco_dataset_href = kwcoco_stac_item['assets']['kwcoco']['href']
    except KeyError:
        raise RuntimeError("Expecting asset named 'kwcoco' in input "
                           "KWCOCO STAC item")

    # Assumes that all the necessary KWCOCO dataset assets are in the
    # same directory as the KWCOCO dataset itself
    kwcoco_dataset_dir = os.path.dirname(kwcoco_dataset_href)
    subprocess.run([*aws_base_command, '--recursive',
                    kwcoco_dataset_dir, outdir],
                   check=True)

    # Returns local path to retreived KWCOCO dataset
    return os.path.join(outdir, os.path.basename(kwcoco_dataset_href))


if __name__ == "__main__":
    sys.exit(main())
