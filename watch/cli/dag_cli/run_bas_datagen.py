#!/usr/bin/env python3
"""
See Old Script:
    ~/code/watch/scripts/run_stac_to_cropped_kwcoco.py
"""
import os
import subprocess
import json

from watch.cli.baseline_framework_ingress import baseline_framework_ingress, load_input_stac_items  # noqa: 501
from watch.cli.baseline_framework_kwcoco_egress import baseline_framework_kwcoco_egress  # noqa: 501
from watch.cli.ta1_stac_to_kwcoco import ta1_stac_to_kwcoco
from watch.cli import coco_add_watch_fields
from watch.utils.util_framework import download_region
import ubelt as ub
import scriptconfig as scfg


class BASDatasetConfig(scfg.DataConfig):
    """
    Generate cropped KWCOCO dataset for BAS from STAC
    """
    input_path = scfg.Value(None, type=str, position=1, required=True, help=ub.paragraph(
            '''
            Path to input T&E Baseline Framework JSON
            '''))

    input_region_path = scfg.Value(None, type=str, position=2, required=True, help=ub.paragraph(
            '''
            Path to input T&E Baseline Framework Region definition JSON
            '''))

    output_path = scfg.Value(None, type=str, position=3, required=True, help='S3 path for output JSON')

    from_collated = scfg.Value(False, isflag=True, help=ub.paragraph(
            '''
            Data to convert has been run through TA-1 collation
            '''))

    aws_profile = scfg.Value(None, type=str, help=ub.paragraph(
            '''
            AWS Profile to use for AWS S3 CLI commands
            '''))

    dryrun = scfg.Value(False, isflag=True, short_alias=['d'], help='Run AWS CLI commands with --dryrun flag')

    virtual = scfg.Value(False, isflag=True, help=ub.paragraph(
            '''
            Ingress will be virtual (using GDAL's virtual file system)
            '''))

    outbucket = scfg.Value(None, type=str, required=True, short_alias=['o'], help=ub.paragraph(
            '''
            S3 Output directory for STAC item / asset egress
            '''))

    requester_pays = scfg.Value(False, isflag=True, short_alias=['r'], help=ub.paragraph(
            '''
            Run AWS CLI commands with `--requestor_payer requester` flag
            '''))

    newline = scfg.Value(False, isflag=True, short_alias=['n'], help=ub.paragraph(
            '''
            Output as simple newline separated STAC items
            '''))

    jobs = scfg.Value(1, type=int, short_alias=['j'], help='Number of jobs to run in parallel')

    dont_recompute = scfg.Value(False, isflag=True, help=ub.paragraph(
            '''
            Will not recompute if output_path already exists
            '''))

    previous_input_path = scfg.Value(None, type=str, help=ub.paragraph(
            '''
            STAC json input file for previous interval
            '''))

    target_gsd = scfg.Value(10, type=int, help='Target GSD of output KWCOCO video space')

    time_combine = scfg.Value(False, help=ub.paragraph(
            '''
            Perform time combine on BAS dataset
            '''))


def main():
    config = BASDatasetConfig.cli(strict=True)
    print('config = {}'.format(ub.urepr(dict(config), nl=1, align=':')))
    run_stac_to_cropped_kwcoco(**config)


def build_combined_kwcoco(input_path,
                          previous_input_path,
                          aws_profile,
                          ta1_cropped_dir,
                          dryrun=False,
                          requester_pays=False,
                          jobs=1,
                          virtual=False,
                          from_collated=False):
    if aws_profile is not None:
        aws_ls_command = ['aws', 's3', '--profile', aws_profile, 'ls']
    else:
        aws_ls_command = ['aws', 's3', 'ls']

    if aws_profile is not None:
        aws_cp_command = ['aws', 's3', '--profile', aws_profile, 'cp']
    else:
        aws_cp_command = ['aws', 's3', 'cp']

    input_stac_items = load_input_stac_items(input_path, aws_cp_command)

    combined_stac_items_path = os.path.join(
        ta1_cropped_dir, 'combined_input_stac_items.jsonl')

    # Confirm that the previous interval input path actually exists on
    # S3 (for first iteration it will not)
    try:
        subprocess.run([*aws_ls_command,
                        previous_input_path], check=True)
    except subprocess.CalledProcessError:
        # If we don't have previous interval input path, set the input
        # as the "combined" for next interval
        with open(combined_stac_items_path, 'w') as f:
            print('\n'.join((json.dumps(item)
                             for item in input_stac_items)), file=f)
        return

    previous_input_stac_items = load_input_stac_items(previous_input_path,
                                                      aws_cp_command)
    input_stac_items.extend(previous_input_stac_items)

    with open(combined_stac_items_path, 'w') as f:
        print('\n'.join((json.dumps(item)
                         for item in input_stac_items)), file=f)

    combined_working_dir = '/tmp/combined'
    os.makedirs(combined_working_dir, exist_ok=True)
    combined_ingress_catalog = baseline_framework_ingress(
        combined_stac_items_path,
        combined_working_dir,
        aws_profile=aws_profile,
        dryrun=dryrun,
        requester_pays=requester_pays,
        relative=False,
        jobs=jobs,
        virtual=virtual)

    # 3. Convert ingressed STAC catalog to KWCOCO
    print("* Converting STAC to KWCOCO *")
    ta1_kwcoco_path_for_sc = os.path.join(combined_working_dir,
                                          'combined_ingress_kwcoco.json')
    ta1_stac_to_kwcoco(combined_ingress_catalog,
                       ta1_kwcoco_path_for_sc,
                       assume_relative=False,
                       populate_watch_fields=True,
                       jobs=jobs,
                       from_collated=from_collated,
                       ignore_duplicates=True)

    return ta1_kwcoco_path_for_sc


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
                               previous_input_path=None,
                               target_gsd=10,
                               time_combine=False):
    if aws_profile is not None:
        aws_ls_command = ['aws', 's3', '--profile', aws_profile, 'ls']
    else:
        aws_ls_command = ['aws', 's3', 'ls']

    if dont_recompute:
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
                       populate_watch_fields=False,
                       jobs=jobs,
                       from_collated=from_collated,
                       ignore_duplicates=True)
    # Add watch fields
    print("* Adding watch fields *")
    coco_add_watch_fields.main(cmdline=False,
                               src=ta1_kwcoco_path,
                               dst=ta1_kwcoco_path,
                               target_gsd=target_gsd,
                               workers=jobs)

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
                    # '--select_images',
                    # '.sensor_coarse == "L8" or .sensor_coarse == "S2"'
                    ],
                   check=True)

    # 3.1. Combine previous interval `kwcoco_for_sc.json` if provided
    # such that SC has full time range of data to work with
    if previous_input_path is not None:
        combined_kwcoco_path = build_combined_kwcoco(
            input_path,
            previous_input_path,
            aws_profile,
            ta1_cropped_dir,
            dryrun=dryrun,
            requester_pays=requester_pays,
            jobs=jobs,
            virtual=virtual,
            from_collated=from_collated)

        if combined_kwcoco_path is None:
            ta1_kwcoco_path_for_sc = ta1_kwcoco_path
        else:
            ta1_kwcoco_path_for_sc = combined_kwcoco_path
    else:
        ta1_kwcoco_path_for_sc = ta1_kwcoco_path

    # 3a. Filter KWCOCO dataset by sensors used for BAS
    # TODO: move this to run_sc_datagen
    print("* Filtering KWCOCO dataset for SC")
    ta1_sc_kwcoco_path = os.path.join(ta1_cropped_dir,
                                      'kwcoco_for_sc.json')
    subprocess.run(['kwcoco', 'subset',
                    '--src', ta1_kwcoco_path_for_sc,
                    '--dst', ta1_sc_kwcoco_path,
                    '--absolute', 'False',
                    '--select_images',
                    '.sensor_coarse == "WV" or .sensor_coarse == "S2"'],
                   check=True)

    # 4. Crop ingress KWCOCO dataset to region for BAS
    print("* Cropping KWCOCO dataset to region for BAS*")
    ta1_cropped_kwcoco_path = os.path.join(ta1_cropped_dir,
                                           'cropped_kwcoco_for_bas.json')
    # include_channels = 'blue|green|red|nir|swir16|swir22|quality'
    include_channels = 'coastal|blue|green|red|B05|B06|B07|nir|B8A|B09|cirrus|swir16|swir22|pan|quality'
    subprocess.run(['python', '-m', 'watch.cli.coco_align',
                    '--visualize', 'False',
                    '--src', ta1_bas_kwcoco_path,
                    '--dst', ta1_cropped_kwcoco_path,
                    '--regions', local_region_path,
                    '--force_nodata', '-9999',
                    '--include_channels', include_channels,  # noqa
                    '--geo_preprop', 'auto',
                    '--keep', 'none',
                    '--convexify_regions', 'True',
                    '--target_gsd', str(target_gsd),  # noqa
                    '--context_factor', '1',
                    '--force_min_gsd', '10',
                    '--workers', str(jobs),  # noqa: 501
                    '--aux_workers', str(include_channels.count('|') + 1),
                    '--rpc_align_method', 'affine_warp'], check=True)

    # 5. "Clean" dataset
    # Ensure that only data bands are used here (i.e. not quality)
    subprocess.run(['python', '-m', 'watch.cli.coco_clean_geotiffs',
                    '--src', ta1_cropped_kwcoco_path,
                    '--channels', 'coastal|blue|green|red|B05|B06|B07|nir|B8A|B09|cirrus|swir16|swir22',
                    '--prefilter_channels', "red",
                    '--min_region_size', '256',
                    '--nodata_value', '-9999',
                    '--workers', str(jobs),  # noqa: 501
                    ], check=True)

    subprocess.run(['python', '-m', 'watch.cli.coco_clean_geotiffs',
                    '--src', ta1_cropped_kwcoco_path,
                    '--channels', "pan",
                    '--prefilter_channels', "pan",
                    '--min_region_size', '256',
                    '--nodata_value', '-9999',
                    '--workers', str(jobs),  # noqa: 501
                    ], check=True)

    # 6. Do the time_combine for BAS
    from watch.utils.util_yaml import Yaml
    default_time_combine_config = ub.udict(
        time_window='1y',
        channels='coastal|blue|green|red|B05|B06|B07|nir|B8A|B09|cirrus|swir16|swir22|pan',
        resolution='10GSD',
        workers='avail',
        start_time='1970-01-01',
        merge_method='mean',
        assets_dname='raw_bands',
    )
    user_time_combine_config = Yaml.coerce(time_combine)

    if user_time_combine_config not in {False, None}:
        if user_time_combine_config is True:
            user_time_combine_config = {}

        time_combine_config = (default_time_combine_config
                               | user_time_combine_config)

        from watch.cli import coco_time_combine
        preproc_kwcoco_fpath = ub.Path(ta1_cropped_kwcoco_path).augment(
            stemsuffix='_timecombined', ext='.kwcoco.zip', multidot=True)
        coco_time_combine.main(
            cmdline=0,
            input_kwcoco_fpath=ta1_cropped_kwcoco_path,
            output_kwcoco_fpath=preproc_kwcoco_fpath,
            **time_combine_config
        )
        ta1_cropped_kwcoco_path = os.fspath(preproc_kwcoco_fpath)

        # Add watch fields
        print("* Adding watch fields to time combined data *")
        coco_add_watch_fields.main(cmdline=False,
                                   src=ta1_cropped_kwcoco_path,
                                   dst=ta1_cropped_kwcoco_path,
                                   target_gsd=target_gsd,
                                   workers=jobs)

    # 7. Egress (envelop KWCOCO dataset in a STAC item and egress;
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
    main()
