import argparse
import sys
from urllib.parse import urlparse
import os
import subprocess
import tempfile
import json

from watch.cli.baseline_framework_ingress import baseline_framework_ingress
from watch.cli.baseline_framework_kwcoco_egress import baseline_framework_kwcoco_egress  # noqa: 501
from watch.cli.ta1_stac_to_kwcoco import ta1_stac_to_kwcoco


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
    parser.add_argument("--l8_model_path",
                        required=True,
                        type=str,
                        help="File path to UKY invariants model for "
                             "Landsat 8 imagery")
    parser.add_argument("--s2_model_path",
                        required=True,
                        type=str,
                        help="File path to UKY invariants model for "
                             "Sentinel 2 imagery")
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


def _download_region(input_region_path,
                     output_region_path,
                     aws_profile=None,
                     dryrun=False,
                     strip_nonregions=False):
    if aws_profile is not None:
        aws_base_command =\
            ['aws', 's3', '--profile', aws_profile, 'cp']
    else:
        aws_base_command = ['aws', 's3', 'cp']

    if dryrun:
        aws_base_command.append('--dryrun')

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

    with open(output_region_path, 'w') as f:
        print(json.dumps(out_region_data, indent=2), file=f)

    return output_region_path


def run_uky_invariants_for_baseline(input_path,
                                    input_region_path,
                                    output_path,
                                    l8_model_path,
                                    s2_model_path,
                                    outbucket,
                                    aws_profile=None,
                                    dryrun=False,
                                    requester_pays=False,
                                    newline=False,
                                    jobs=1):
    # 1. Ingress data
    print("* Running baseline framework ingress *")
    ingress_dir = '/tmp/ingress'
    ingress_catalog = baseline_framework_ingress(
        input_path,
        ingress_dir,
        aws_profile,
        dryrun,
        requester_pays,
        jobs)

    # 2. Download and prune region file
    print("* Downloading and pruning region file *")
    local_region_path = '/tmp/region.json'
    local_region_path = _download_region(input_region_path,
                                         local_region_path,
                                         aws_profile=aws_profile,
                                         dryrun=dryrun,
                                         strip_nonregions=True)

    # 3. Convert ingressed STAC catalog to KWCOCO
    print("* Converting STAC to KWCOCO *")
    ta1_kwcoco_path = os.path.join(ingress_dir, 'ingress_kwcoco.json')
    ta1_stac_to_kwcoco(ingress_catalog,
                       ta1_kwcoco_path,
                       assume_relative=False,
                       populate_watch_fields=False,
                       jobs=jobs)

    # 4. Crop ingress KWCOCO dataset to region
    print("* Cropping KWCOCO dataset to region *")
    ta1_cropped_dir = '/tmp/cropped_kwcoco/'
    ta1_cropped_kwcoco_path = os.path.join(ta1_cropped_dir,
                                           'cropped_kwcoco.json')
    subprocess.run(['python', '-m', 'watch.cli.coco_align_geotiffs',
                    '--visualize', 'True',
                    '--src', ta1_kwcoco_path,
                    '--dst', ta1_cropped_kwcoco_path,
                    '--regions', local_region_path,
                    '--repc_align_method', 'affine_warp'], check=True)

    # 5. Add WATCH specific fields to cropped KWCOCO dataset
    print("* Adding WATCH fields to cropped KWCOCO dataset *")
    ta1_cropped_watch_kwcoco_path = os.path.join(ta1_cropped_dir,
                                                 'cropped_watch_kwcoco.json')
    # With overwrite set to True channel names were being clobbered
    subprocess.run(['python', '-m', 'watch.cli.coco_add_watch_fields',
                    '--src', ta1_cropped_kwcoco_path,
                    '--dst', ta1_cropped_watch_kwcoco_path,
                    '--target_gsd', '10.0',
                    '--overwrite', 'False'],
                   check=True)

    # 6. Generate L8 features
    print("* Generating UKY invariant features for L8 *")
    ta1_uky_l8_features_kwcoco_path = os.path.join(
        ta1_cropped_dir, 'uky_l8_invariants_kwcoco.json')
    subprocess.run(['python', '-m', 'watch.tasks.invariants.predict',
                    '--ckpt_path', l8_model_path,
                    '--sensor', 'L8',
                    '--device', 'cuda',
                    '--input_kwcoco', ta1_cropped_watch_kwcoco_path,
                    '--output_kwcoco', ta1_uky_l8_features_kwcoco_path,
                    '--bands', 'coastal|lwir11|lwir12|blue|green|red|nir|swir16|swir22|pan|cirrus'],  # noqa: 501
                   check=True)

    # 7. Generate S2 features
    print("* Generating UKY invariant features for S2 *")
    ta1_uky_s2_features_kwcoco_path = os.path.join(
        ta1_cropped_dir, 'uky_s2_invariants_kwcoco.json')
    subprocess.run(['python', '-m', 'watch.tasks.invariants.predict',
                    '--ckpt_path', s2_model_path,
                    '--sensor', 'S2',
                    '--device', 'cuda',
                    '--input_kwcoco', ta1_cropped_watch_kwcoco_path,
                    '--output_kwcoco', ta1_uky_s2_features_kwcoco_path,
                    '--bands', 'coastal|blue|green|red|B05|B06|B07|nir|B09|cirrus|swir16|swir22|B8A'],  # noqa: 501
                   check=True)

    # 8. Combine features
    print("* Combining UKY invariant features into single KWCOCO dataset *")
    ta1_uky_combined_features_kwcoco_path = os.path.join(
        ta1_cropped_dir, 'uky_combined_invariants_kwcoco.json')
    subprocess.run(['python', '-m', 'watch.cli.coco_combine_features',
                    '--src', ta1_uky_l8_features_kwcoco_path, ta1_uky_s2_features_kwcoco_path,  # noqa: 501
                    '--dst', ta1_uky_combined_features_kwcoco_path],
                   check=True)

    # 9. Egress (envelop KWCOCO dataset in a STAC item and egress;
    #    will need to recursive copy the kwcoco output directory up to
    #    S3 bucket)
    print("* Egressing KWCOCO dataset and associated STAC item *")
    baseline_framework_kwcoco_egress(ta1_uky_combined_features_kwcoco_path,
                                     local_region_path,
                                     output_path,
                                     outbucket,
                                     aws_profile=None,
                                     dryrun=False,
                                     newline=False)


if __name__ == "__main__":
    sys.exit(main())
