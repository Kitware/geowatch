import argparse
import sys
import os
import subprocess

from watch.cli.baseline_framework_ingress import baseline_framework_ingress
from watch.cli.baseline_framework_kwcoco_egress import baseline_framework_kwcoco_egress  # noqa: 501
from watch.cli.ta1_stac_to_kwcoco import ta1_stac_to_kwcoco
from watch.utils.util_framework import download_region


def main():
    parser = argparse.ArgumentParser(
        description="Generate cropped KWCOCO dataset from TA-1 output")

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
    parser.add_argument("--from-collated",
                        action='store_true',
                        default=False,
                        help="Data to convert has been run through TA-1 "
                             "collation")
    parser.add_argument("--aws_profile",
                        required=False,
                        type=str,
                        help="AWS Profile to use for AWS S3 CLI commands")
    parser.add_argument("-d", "--dryrun",
                        action='store_true',
                        default=False,
                        help="Run AWS CLI commands with --dryrun flag")
    parser.add_argument("--virtual",
                        action='store_true',
                        default=False,
                        help="Ingress will be virtual (using GDAL's virtual "
                             "file system)")
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
    parser.add_argument("--dont-recompute",
                        action='store_true',
                        default=False,
                        help="Will not recompute if output_path "
                             "already exists")
    parser.add_argument("--force_one_job_for_cropping",
                        action='store_true',
                        default=False,
                        help="Force jobs=1 for cropping")

    run_stac_to_cropped_kwcoco(**vars(parser.parse_args()))

    return 0


def run_stac_to_cropped_kwcoco(input_path,
                               input_region_path,
                               output_path,
                               outbucket,
                               from_collated=False,
                               aws_profile=None,
                               dryrun=False,
                               requester_pays=False,
                               newline=False,
                               jobs=1,
                               virtual=False,
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
    print("* Running baseline framework ingress *")
    ingress_dir = '/tmp/ingress'
    ingress_catalog = baseline_framework_ingress(
        input_path,
        ingress_dir,
        aws_profile=aws_profile,
        dryrun=dryrun,
        requester_pays=requester_pays,
        relative=False,
        jobs=jobs,
        virtual=virtual)

    # 2. Download and prune region file
    print("* Downloading and pruning region file *")
    local_region_path = '/tmp/region.json'
    local_region_path = download_region(input_region_path,
                                        local_region_path,
                                        aws_profile=aws_profile,
                                        strip_nonregions=True)

    # 3. Convert ingressed STAC catalog to KWCOCO
    print("* Converting STAC to KWCOCO *")
    ta1_kwcoco_path = os.path.join(ingress_dir, 'ingress_kwcoco.json')
    ta1_stac_to_kwcoco(ingress_catalog,
                       ta1_kwcoco_path,
                       assume_relative=False,
                       populate_watch_fields=True,
                       jobs=jobs,
                       from_collated=from_collated,
                       ignore_duplicates=True)

    # `ta1_cropped_dir` is the directory that gets recursively copied
    # up to S3, want to put any kwcoco manifests we may need
    # downstream into this directory.  TODO: rename variable to
    # include something like upload_dir or output_dir
    ta1_cropped_dir = '/tmp/cropped_kwcoco'
    os.makedirs(ta1_cropped_dir, exist_ok=True)

    # 3a. Filter KWCOCO dataset by sensors used for BAS
    print("* Filtering KWCOCO dataset for BAS")
    ta1_bas_kwcoco_path = os.path.join(ta1_cropped_dir,
                                       'kwcoco_for_bas.json')
    subprocess.run(['kwcoco', 'subset',
                    '--src', ta1_kwcoco_path,
                    '--dst', ta1_bas_kwcoco_path,
                    '--absolute', 'False',
                    '--select_images',
                    '.sensor_coarse == "L8" or .sensor_coarse == "S2"'],
                   check=True)

    # 3a. Filter KWCOCO dataset by sensors used for BAS
    print("* Filtering KWCOCO dataset for SC")
    ta1_sc_kwcoco_path = os.path.join(ta1_cropped_dir,
                                      'kwcoco_for_sc.json')
    subprocess.run(['kwcoco', 'subset',
                    '--src', ta1_kwcoco_path,
                    '--dst', ta1_sc_kwcoco_path,
                    '--absolute', 'False',
                    '--select_images',
                    '.sensor_coarse == "WV" or .sensor_coarse == "S2"'],
                   check=True)

    # 4. Crop ingress KWCOCO dataset to region for BAS
    print("* Cropping KWCOCO dataset to region for BAS*")
    ta1_cropped_kwcoco_path = os.path.join(ta1_cropped_dir,
                                           'cropped_kwcoco_for_bas.json')
    include_channels = 'blue|green|red|nir|swir16|swir22|cloudmask'
    subprocess.run(['python', '-m', 'watch.cli.coco_align_geotiffs',
                    '--visualize', 'False',
                    '--src', ta1_bas_kwcoco_path,
                    '--dst', ta1_cropped_kwcoco_path,
                    '--regions', local_region_path,
                    '--force_nodata', '-9999',
                    '--include_channels', include_channels,  # noqa
                    '--geo_preprop', 'auto',
                    '--keep', 'none',
                    '--target_gsd', '10',  # TODO: Expose as cli parameter
                    '--context_factor', '1',
                    '--workers', '1' if force_one_job_for_cropping else str(jobs),  # noqa: 501
                    '--aux_workers', include_channels.count('|') + 1,
                    '--rpc_align_method', 'affine_warp'], check=True)

    # 5. Egress (envelop KWCOCO dataset in a STAC item and egress;
    #    will need to recursive copy the kwcoco output directory up to
    #    S3 bucket)
    print("* Egressing KWCOCO dataset and associated STAC item *")
    baseline_framework_kwcoco_egress(ta1_cropped_kwcoco_path,
                                     local_region_path,
                                     output_path,
                                     outbucket,
                                     aws_profile=None,
                                     dryrun=False,
                                     newline=False)


if __name__ == "__main__":
    sys.exit(main())
