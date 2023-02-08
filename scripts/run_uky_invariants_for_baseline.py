import argparse
import sys
from urllib.parse import urlparse
import os
import subprocess
import tempfile
import json

from watch.cli.baseline_framework_kwcoco_egress import baseline_framework_kwcoco_egress  # noqa: 501
from watch.cli.baseline_framework_kwcoco_ingress import baseline_framework_kwcoco_ingress  # noqa: 501


def main():
    parser = argparse.ArgumentParser(
        description="Run UKY invariant feature computation as baseline "
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
    parser.add_argument("--model_path",
                        required=True,
                        type=str,
                        help="File path to UKY invariants model")
    parser.add_argument("--pca_projection_path",
                        required=True,
                        type=str,
                        help="File path to UKY invariants PCA "
                             "projections file")
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
    parser.add_argument("--do_pca",
                        action='store_true',
                        default=False,
                        help="Perform PCA on invariants")
    parser.add_argument("-r", "--requester_pays",
                        action='store_true',
                        default=False,
                        help="Run AWS CLI commands with "
                             "`--requestor_payer requester` flag")
    parser.add_argument("-n", "--newline",
                        action='store_true',
                        default=False,
                        help="Output as simple newline separated STAC items")
    parser.add_argument("-j", "--jobs",
                        type=int,
                        default=1,
                        required=False,
                        help="Number of jobs to run in parallel")

    run_uky_invariants_for_baseline(**vars(parser.parse_args()))

    return 0


def _download_region(aws_base_command,
                     input_region_path,
                     output_region_path,
                     strip_nonregions=False,
                     replace_originator=True):
    scheme, *_ = urlparse(input_region_path)
    if scheme == 's3':
        with tempfile.NamedTemporaryFile() as temporary_file:
            command = [*aws_base_command,
                       input_region_path,
                       temporary_file.name]

            print("Running: {}".format(' '.join(command)))
            # TODO: Manually check return code / output
            subprocess.run(command, check=True)

            with open(temporary_file.name) as f:
                out_region_data = json.load(f)
    elif scheme == '':
        with open(input_region_path) as f:
            out_region_data = json.load(f)
    else:
        raise NotImplementedError("Don't know how to pull down region file "
                                  "with URI scheme: '{}'".format(scheme))

    if strip_nonregions:
        out_region_data['features'] =\
            [feature
             for feature in out_region_data.get('features', ())
             if ('properties' in feature
                 and feature['properties'].get('type') == 'region')]

    if replace_originator:
        for feature in out_region_data.get('features', ()):
            if feature['properties']['type'] == 'region':
                feature['properties']['originator'] = 'kit'

    region_id = None
    for feature in out_region_data.get('features', ()):
        props = feature['properties']
        if props['type'] == 'region':
            region_id = props.get('region_model_id', props.get('region_id'))
            break

    with open(output_region_path, 'w') as f:
        print(json.dumps(out_region_data, indent=2), file=f)

    return output_region_path, region_id


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
    local_region_path, region_id = _download_region(aws_base_command,
                                                    input_region_path,
                                                    local_region_path,
                                                    strip_nonregions=True,
                                                    replace_originator=True)

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
    sys.exit(main())
