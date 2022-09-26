import argparse
import sys
import os
import subprocess
import json

from watch.cli.baseline_framework_kwcoco_ingress import baseline_framework_kwcoco_ingress  # noqa: 501
from watch.cli.baseline_framework_kwcoco_egress import baseline_framework_kwcoco_egress  # noqa: 501
from watch.utils.util_framework import download_region


def main():
    parser = argparse.ArgumentParser(
        description="Generate cropped KWCOCO dataset for SC")

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
    parser.add_argument("--dont-recompute",
                        action='store_true',
                        default=False,
                        help="Will not recompute if output_path "
                             "already exists")
    parser.add_argument("--force_one_job_for_cropping",
                        action='store_true',
                        default=False,
                        help="Force jobs=1 for cropping")

    run_generate_sc_cropped_kwcoco(**vars(parser.parse_args()))

    return 0


def run_generate_sc_cropped_kwcoco(input_path,
                                   input_region_path,
                                   output_path,
                                   outbucket,
                                   aws_profile=None,
                                   dryrun=False,
                                   newline=False,
                                   jobs=1,
                                   dont_recompute=False,
                                   force_one_job_for_cropping=False):
    if dont_recompute:
        if aws_profile is not None:
            aws_ls_command = ['aws', 's3', '--profile', aws_profile, 'ls']
        else:
            aws_ls_command = ['aws', 's3', 'ls']

        try:
            subprocess.run([*aws_ls_command, output_path], check=True)
        except subprocess.CalledProcessError:
            # Continue processing
            pass
        else:
            # If output_path file was there, nothing to do
            return

    # 1. Ingress data
    print("* Running baseline framework kwcoco ingress *")
    ingress_dir = '/tmp/ingress'
    _ = baseline_framework_kwcoco_ingress(
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

    # Parse region_id from original region file
    with open(local_region_path) as f:
        region = json.load(f)

        region_id = None
        for feature in region.get('features', ()):
            props = feature['properties']
            if props['type'] == 'region':
                region_id = props.get('region_model_id',
                                      props.get('region_id'))
                break

    if region_id is None:
        raise RuntimeError("Couldn't parse 'region_id' from input region file")

    # Paths to inputs generated in previous pipeline steps
    bas_region_path = os.path.join(ingress_dir,
                                   'region_models',
                                   '{}.geojson'.format(region_id))
    ta1_sc_kwcoco_path = os.path.join(ingress_dir,
                                      'kwcoco_for_sc.json')

    # 4. Crop ingress KWCOCO dataset to region for SC
    print("* Cropping KWCOCO dataset to region for SC*")
    ta1_sc_cropped_kwcoco_path = os.path.join(ingress_dir,
                                              'cropped_kwcoco_for_sc.json')
    # Crops to BAS generated site_summaries
    subprocess.run(['python', '-m', 'watch.cli.coco_align_geotiffs',
                    '--visualize', 'False',
                    '--src', ta1_sc_kwcoco_path,
                    '--dst', ta1_sc_cropped_kwcoco_path,
                    '--regions', bas_region_path,
                    '--force_nodata', '-9999',
                    '--include_channels', 'red|green|blue|cloudmask',
                    '--site_summary', 'True',
                    '--geo_preprop', 'auto',
                    '--keep', 'none',
                    '--target_gsd', '3',  # TODO: Expose as cli parameter
                    '--context_factor', '1.5',  # TODO: Expose as cli parameter
                    '--workers', '1' if force_one_job_for_cropping else str(jobs),  # noqa: 501
                    '--aux_workers', '4',
                    '--rpc_align_method', 'affine_warp'], check=True)

    # 5. Egress (envelop KWCOCO dataset in a STAC item and egress;
    #    will need to recursive copy the kwcoco output directory up to
    #    S3 bucket)
    print("* Egressing KWCOCO dataset and associated STAC item *")
    baseline_framework_kwcoco_egress(ta1_sc_cropped_kwcoco_path,
                                     local_region_path,
                                     output_path,
                                     outbucket,
                                     aws_profile=None,
                                     dryrun=False,
                                     newline=False)


if __name__ == "__main__":
    sys.exit(main())
