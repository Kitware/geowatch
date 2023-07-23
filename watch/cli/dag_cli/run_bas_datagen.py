#!/usr/bin/env python3
"""
See Old Script:
    ~/code/watch/scripts/run_stac_to_cropped_kwcoco.py
"""
import os
import subprocess
import json
import shutil

from watch.cli.baseline_framework_ingress import baseline_framework_ingress, load_input_stac_items
from watch.cli.smartflow_egress import smartflow_egress
from watch.cli.stac_to_kwcoco import stac_to_kwcoco
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

    previous_outbucket = scfg.Value(None, type=str, help=ub.paragraph(
            '''
            Outbucket for previous interval BAS DatasetGen
            '''))

    bas_align_config = scfg.Value(None, type=str, help=ub.paragraph(
        '''
        The configuration for the coco-align step
        '''))

    time_combine_config = scfg.Value(None, type=str, help=ub.paragraph(
        '''
        If specified, perform time combine on BAS dataset. The special key
        enabled will disable the computation.
        '''), alias=['time_combine'])

    def __post_init__(self):
        if self.time_combine_config in {False, None, 'False', 'None'}:
            self.time_combine_config = {'enabled': False}
        elif self.time_combine_config in {True, 'True'}:
            self.time_combine_config = {'enabled': True}


def main():
    config = BASDatasetConfig.cli(strict=True)
    print('config = {}'.format(ub.urepr(config, nl=1, align=':')))
    run_stac_to_cropped_kwcoco(config)


def build_combined_kwcoco(input_path,
                          previous_input_path,
                          aws_profile,
                          ta1_cropped_dir,
                          dryrun=False,
                          requester_pays=False,
                          jobs=1,
                          virtual=False,
                          from_collated=False):

    from watch.utils.util_framework import AWS_S3_Command
    aws_ls = AWS_S3_Command('ls', profile=aws_profile)
    aws_cp = AWS_S3_Command('cp', profile=aws_profile)
    aws_cp_command = aws_cp.finalize()
    aws_ls_command = aws_ls.finalize()

    input_stac_items = load_input_stac_items(input_path, aws_cp_command)

    combined_stac_items_path = os.path.join(
        ta1_cropped_dir, 'combined_input_stac_items.jsonl')

    # Confirm that the previous interval input path actually exists on
    # S3 (for first iteration it will not)
    try:
        ub.cmd([*aws_ls_command, previous_input_path], check=True, verbose=3,
               capture=False)
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

    combined_working_dir = ub.Path('/tmp/combined')
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
    ta1_kwcoco_path_for_sc = combined_working_dir / 'combined_ingress_kwcoco.json'
    stac_to_kwcoco(combined_ingress_catalog,
                   ta1_kwcoco_path_for_sc,
                   assume_relative=False,
                   populate_watch_fields=True,
                   jobs=jobs,
                   from_collated=from_collated,
                   ignore_duplicates=True)

    return ta1_kwcoco_path_for_sc


def run_stac_to_cropped_kwcoco(config):
    from watch.utils.util_framework import AWS_S3_Command
    from watch.utils import util_framework
    from kwutil.util_yaml import Yaml
    from delayed_image.channel_spec import ChannelSpec
    # from kwcoco import ChannelSpec
    from watch.cli import coco_align
    from watch.cli import coco_time_combine
    aws_ls = AWS_S3_Command('ls', profile=config.aws_profile)
    aws_cp = AWS_S3_Command('cp', profile=config.aws_profile)
    aws_base_command = aws_cp.finalize()
    aws_ls_command = aws_ls.finalize()

    align_config_default = ub.udict(Yaml.coerce(ub.codeblock(
        f'''
        force_nodata: -9999
        include_channels: "coastal|blue|green|red|B05|B06|B07|nir|B8A|B09|cirrus|swir16|swir22|pan|quality"
        geo_preprop: auto
        keep: null
        convexify_regions: True
        target_gsd: 10GSD
        context_factor: 1
        force_min_gsd: 10
        img_workers: {str(config.jobs)}
        aux_workers: auto
        rpc_align_method: affine_warp
        ''')))

    time_combine_config_default = ub.udict(Yaml.coerce(ub.codeblock(
        '''
        enabled: True
        time_window: '1y'
        channels: 'coastal|blue|green|red|B05|B06|B07|nir|B8A|B09|cirrus|swir16|swir22|pan'
        resolution: '10GSD'
        workers: 'avail'
        start_time: '1970-01-01'
        merge_method: 'mean'
        assets_dname: 'raw_bands'
        ''')))

    align_config = align_config_default | Yaml.coerce(config.bas_align_config)
    if align_config['aux_workers'] == 'auto':
        align_config['aux_workers'] = align_config['include_channels'].count('|') + 1
    time_combine_config = time_combine_config_default | Yaml.coerce(config.time_combine_config)

    if time_combine_config['channels'] == 'auto':
        # Default time combine channels to the align channels minus quality.
        time_combine_config['channels'] = ChannelSpec.coerce(align_config['include_channels']) - {'quality'}

    time_combine_enabled = time_combine_config.pop('enabled', True)

    target_gsd = align_config['target_gsd']

    if config.dont_recompute:
        try:
            ub.cmd([*aws_ls_command, config.output_path],
                   check=True, capture=False, verbose=3)
        except subprocess.CalledProcessError:
            # Continue processing
            pass
        else:
            # If output_path file was there, nothing to do
            return

    # `ta1_cropped_dir` is the directory that gets recursively copied
    # up to S3, want to put any kwcoco manifests we may need
    # downstream into this directory.  TODO: rename variable to
    # include something like upload_dir or output_dir
    ta1_cropped_dir = ub.Path('/tmp/cropped_kwcoco')
    local_region_path = ub.Path('/tmp/region.json')
    ingress_dir = ub.Path('/tmp/ingress')

    ta1_cropped_dir.ensuredir()

    ta1_kwcoco_path = ingress_dir / 'ingress_kwcoco.json'
    ta1_bas_kwcoco_path = ta1_cropped_dir / 'kwcoco_for_bas.json'
    ta1_cropped_kwcoco_path = ta1_cropped_dir / 'cropped_kwcoco_for_bas.json'

    align_config['src'] = ta1_bas_kwcoco_path
    align_config['dst'] = ta1_cropped_kwcoco_path
    align_config['regions'] = local_region_path
    # Validate config before running stuff
    align_config = coco_align.CocoAlignGeotiffConfig(**align_config)
    print('align_config = {}'.format(ub.urepr(align_config, nl=1)))

    if time_combine_enabled:
        preproc_kwcoco_fpath = ub.Path(ta1_cropped_kwcoco_path).augment(
            stemsuffix='_timecombined', ext='.kwcoco.zip', multidot=True)
        time_combine_config['input_kwcoco_fpath'] = ta1_cropped_kwcoco_path
        time_combine_config['output_kwcoco_fpath'] = preproc_kwcoco_fpath
        # Validate config before running stuff
        time_combine_config = coco_time_combine.TimeCombineConfig(**time_combine_config)
        print('time_combine_config = {}'.format(ub.urepr(time_combine_config, nl=1)))

    # 1. Ingress data
    print("* Running baseline framework ingress *")
    ingress_catalog = baseline_framework_ingress(
        config.input_path,
        ingress_dir,
        aws_profile=config.aws_profile,
        dryrun=config.dryrun,
        requester_pays=config.requester_pays,
        relative=False,
        jobs=config.jobs,
        virtual=config.virtual)

    # 2. Download and prune region file
    print("* Downloading and pruning region file *")
    local_region_path = download_region(config.input_region_path,
                                        local_region_path,
                                        aws_profile=config.aws_profile,
                                        strip_nonregions=True)

    # HACK: this is what coco-align outputs by default. We should have this be
    # explicit and configurable so we can set it to what we want here.
    from watch.utils.util_framework import determine_region_id
    region_id = determine_region_id(local_region_path)
    ta1_cropped_rawband_dpath = ta1_cropped_dir / region_id

    # 3. Convert ingressed STAC catalog to KWCOCO
    print("* Converting STAC to KWCOCO *")
    stac_to_kwcoco(ingress_catalog,
                       ta1_kwcoco_path,
                       assume_relative=False,
                       populate_watch_fields=False,
                       jobs=config.jobs,
                       from_collated=config.from_collated,
                       ignore_duplicates=True)
    # Add watch fields
    print("* Adding watch fields *")
    coco_add_watch_fields.main(cmdline=False,
                               src=ta1_kwcoco_path,
                               inplace=True,
                               target_gsd=target_gsd,
                               workers=config.jobs)

    # 3a. Filter KWCOCO dataset by sensors used for BAS
    print("* Filtering KWCOCO dataset for BAS")
    ub.cmd([
        'kwcoco', 'subset',
        '--src', ta1_kwcoco_path,
        '--dst', ta1_bas_kwcoco_path,
        '--absolute', 'False',
        # '--select_images',
        # '.sensor_coarse == "L8" or .sensor_coarse == "S2"'
    ], check=True, verbose=3, capture=False)

    # 3.1. Combine previous interval `kwcoco_for_sc.json` if provided
    # such that SC has full time range of data to work with
    if config.previous_input_path is not None:
        combined_kwcoco_path = build_combined_kwcoco(
            config.input_path,
            config.previous_input_path,
            config.aws_profile,
            ta1_cropped_dir,
            dryrun=config.dryrun,
            requester_pays=config.requester_pays,
            jobs=config.jobs,
            virtual=config.virtual,
            from_collated=config.from_collated)

        if combined_kwcoco_path is None:
            ta1_kwcoco_path_for_sc = ta1_kwcoco_path
        else:
            ta1_kwcoco_path_for_sc = combined_kwcoco_path
    else:
        ta1_kwcoco_path_for_sc = ta1_kwcoco_path

    # 3a. Filter KWCOCO dataset by sensors used for BAS
    # TODO: move this to run_sc_datagen
    print("* Filtering KWCOCO dataset for SC")
    ta1_sc_kwcoco_path = ta1_cropped_dir / 'kwcoco_for_sc.json'
    ub.cmd(['kwcoco', 'subset',
            '--src', ta1_kwcoco_path_for_sc,
            '--dst', ta1_sc_kwcoco_path,
            '--absolute', 'False',
            '--select_images',
            '.sensor_coarse == "WV" or .sensor_coarse == "S2"'],
           check=True, verbose=3, capture=False)

    # 4. Crop ingress KWCOCO dataset to region for BAS
    print("* Cropping KWCOCO dataset to region for BAS*")

    ALIGN_EXEC_MODE = 'cmd'
    # Not sure if one is more stable than the other
    if ALIGN_EXEC_MODE == 'import':
        coco_align.main(cmdline=False, **align_config)
    elif ALIGN_EXEC_MODE == 'cmd':
        align_arglist = util_framework._make_arglist(align_config)
        ub.cmd(['python', '-m', 'watch.cli.coco_align'] + align_arglist,
               check=True, capture=False, verbose=3)
    else:
        raise KeyError(ALIGN_EXEC_MODE)

    # 5. Do the time_combine for BAS
    if time_combine_enabled:
        coco_time_combine.main(
            cmdline=0,
            **time_combine_config
        )
        # Add watch fields
        print("* Adding watch fields to time combined data *")
        coco_add_watch_fields.main(cmdline=False,
                                   src=preproc_kwcoco_fpath,
                                   dst=preproc_kwcoco_fpath,
                                   target_gsd=target_gsd,
                                   workers=config.jobs)
        final_interval_bas_kwcoco_path = preproc_kwcoco_fpath
    else:
        final_interval_bas_kwcoco_path = ta1_cropped_kwcoco_path

    # 6.1. Combine previous interval time-combined data for BAS
    if config.previous_outbucket is not None:
        print('* Combining previous interval time combined kwcoco with'
              'current *')
        combined_timecombined_kwcoco_path = ta1_cropped_dir / 'combined_timecombined_kwcoco.json'

        previous_ingress_dir = ub.Path('/tmp/ingress_previous')
        ub.cmd([*aws_base_command, '--recursive',
                config.previous_outbucket, previous_ingress_dir],
               check=True, verbose=3, capture=False)

        previous_timecombined_kwcoco_path = previous_ingress_dir / 'combined_timecombined_kwcoco.json'

        # On first interval nothing will be copied down so need to
        # check that we have the input explicitly
        from watch.cli.concat_kwcoco_videos import concat_kwcoco_datasets
        if previous_timecombined_kwcoco_path.is_file():
            concat_kwcoco_datasets(
                (previous_timecombined_kwcoco_path, final_interval_bas_kwcoco_path),
                combined_timecombined_kwcoco_path)
            # Copy saliency assets from previous bas fusion
            shutil.copytree(
                previous_ingress_dir / 'raw_bands',
                ta1_cropped_dir / 'raw_bands',
                dirs_exist_ok=True)
        else:
            # Copy current bas_fusion_kwcoco_path to combined path as
            # this is the first interval
            shutil.copy(final_interval_bas_kwcoco_path,
                        combined_timecombined_kwcoco_path)
    else:
        combined_timecombined_kwcoco_path = final_interval_bas_kwcoco_path

    # 7. Egress (envelop KWCOCO dataset in a STAC item and egress;
    #    will need to recursive copy the kwcoco output directory up to
    #    S3 bucket)

    timecombined_rawband_dpath = ta1_cropped_dir / 'raw_bands'
    timecombined_teamfeat_dpath = ta1_cropped_dir / '_teamfeats'
    # Put a dummy file in this directory so we can upload a nearly-empty folder
    # to S3
    timecombined_teamfeat_dpath.ensuredir()
    (timecombined_teamfeat_dpath / 'dummy').write_text('dummy')

    print("* Egressing KWCOCO dataset and associated STAC item *")
    assets_to_egress = {
        'timecombined_kwcoco_file_for_bas': combined_timecombined_kwcoco_path,
        'timecombined_kwcoco_file_for_bas_assets': timecombined_rawband_dpath,

        # This is an alias to the BAS dataset and assets that team feature
        # scripts will update.
        'enriched_bas_kwcoco_file': combined_timecombined_kwcoco_path,
        'enriched_bas_kwcoco_teamfeats': timecombined_teamfeat_dpath,
        'enriched_bas_kwcoco_rawbands': timecombined_rawband_dpath,

        # TODO: @DMJ: I dont think anything uses this? Can it be removed?
        'kwcoco_for_sc': ta1_sc_kwcoco_path,

        # We need to egress the temporally dense dataset for COLD
        'timedense_bas_kwcoco_file': ta1_cropped_kwcoco_path,
        'timedense_bas_kwcoco_rawbands': ta1_cropped_rawband_dpath,
    }
    smartflow_egress(assets_to_egress,
                     local_region_path,
                     config.output_path,
                     config.outbucket,
                     aws_profile=config.aws_profile,
                     dryrun=config.dryrun,
                     newline=config.newline)


if __name__ == "__main__":
    main()
