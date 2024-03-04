#!/usr/bin/env python3
"""
See Old Script:
    ~/code/watch/scripts/run_stac_to_cropped_kwcoco.py
"""
import sys
import traceback
import os
import json
import shutil

from geowatch.cli.baseline_framework_ingress import baseline_framework_ingress, load_input_stac_items
from geowatch.cli.smartflow_egress import smartflow_egress
from geowatch.cli.stac_to_kwcoco import stac_to_kwcoco
from geowatch.cli import coco_add_watch_fields
from geowatch.utils.util_framework import download_region
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

    previous_interval_output = scfg.Value(None, type=str, help=ub.paragraph(
            '''
            Output path for previous interval BAS DatasetGen step
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

    skip_timecombine_on_fail = scfg.Value(False, help=ub.paragraph(
            '''
            If an error occurs during the timecombine call, output empty KWCOCO.
            '''))

    def __post_init__(self):
        if self.time_combine_config in {False, None, 'False', 'None'}:
            self.time_combine_config = {'enabled': False}
        elif self.time_combine_config in {True, 'True'}:
            self.time_combine_config = {'enabled': True}


def main():
    config = BASDatasetConfig.cli(strict=True)
    print('config = {}'.format(ub.urepr(config, nl=1, align=':')))
    run_stac_to_cropped_kwcoco(config)


def build_combined_stac(previous_stac_input_path,
                        stac_input_path,
                        combined_stac_output_path):
    previous_stac_items = load_input_stac_items(previous_stac_input_path, None)
    current_stac_items = load_input_stac_items(stac_input_path, None)

    combined_stac_items = previous_stac_items.copy()
    combined_stac_items.extend(current_stac_items)

    with open(combined_stac_output_path, 'w') as f:
        print('\n'.join((json.dumps(item)
                         for item in combined_stac_items)), file=f)

    return combined_stac_output_path


def input_stac_to_kwcoco(stac_items_path,
                         working_dir,
                         out_kwcoco_filename,
                         target_gsd,
                         aws_profile=None,
                         dryrun=False,
                         requester_pays=False,
                         jobs=1,
                         virtual=False,
                         from_collated=False):
    working_dir = ub.Path(working_dir)
    os.makedirs(working_dir, exist_ok=True)
    print("* Running baseline framework ingress *")
    ingressed_catalog = baseline_framework_ingress(
        stac_items_path,
        working_dir,
        aws_profile=aws_profile,
        dryrun=dryrun,
        requester_pays=requester_pays,
        relative=False,
        jobs=jobs,
        virtual=virtual)

    # 3. Convert ingressed STAC catalog to KWCOCO
    print("* Converting STAC to KWCOCO *")
    out_kwcoco_path = working_dir / out_kwcoco_filename
    stac_to_kwcoco(ingressed_catalog,
                   out_kwcoco_path,
                   assume_relative=False,
                   populate_watch_fields=False,
                   jobs=jobs,
                   from_collated=from_collated,
                   ignore_duplicates=True)

    print("* Adding geowatch feilds *")
    coco_add_watch_fields.main(cmdline=False,
                               src=out_kwcoco_path,
                               inplace=True,
                               target_gsd=target_gsd,
                               workers=jobs)

    return out_kwcoco_path


def run_stac_to_cropped_kwcoco(config):
    from geowatch.utils import util_fsspec
    from kwutil.util_yaml import Yaml
    from delayed_image.channel_spec import ChannelSpec
    # from kwcoco import ChannelSpec
    from geowatch.cli import coco_align
    from geowatch.cli import coco_time_combine
    from geowatch.mlops.pipeline_nodes import ProcessNode
    from geowatch.cli.smartflow_ingress import smartflow_ingress
    import kwcoco

    from geowatch.utils.util_framework import NodeStateDebugger
    node_state = NodeStateDebugger()
    node_state.print_environment()

    if config.aws_profile is not None:
        # This should be sufficient, but it is not tested.
        util_fsspec.S3Path._new_fs(profile=config.aws_profile)

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
        # Auto gives each channel its own aux worker
        num_channels = align_config['include_channels'].count('|') + 1
        align_config['aux_workers'] = num_channels

    time_combine_config = time_combine_config_default | Yaml.coerce(config.time_combine_config)

    if time_combine_config['channels'] == 'auto':
        # Default time combine channels to the align channels minus quality.
        time_combine_config['channels'] = ChannelSpec.coerce(align_config['include_channels']) - {'quality'}

    time_combine_enabled = time_combine_config.pop('enabled', True)

    target_gsd = align_config['target_gsd']

    if config.dont_recompute:
        output_path = util_fsspec.FSPath.coerce(config.output_path)
        if output_path.exists():
            print('Dont recompute is True. Early stopping')
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
    ingress_dir.ensuredir()

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

    # 2. Download and prune region file
    print("* Downloading and pruning region file *")
    local_region_path = download_region(config.input_region_path,
                                        local_region_path,
                                        aws_profile=config.aws_profile,
                                        strip_nonregions=True)

    # HACK: this is what coco-align outputs by default. We should have this be
    # explicit and configurable so we can set it to what we want here.
    from geowatch.utils.util_framework import determine_region_id
    region_id = determine_region_id(local_region_path)
    ta1_cropped_rawband_dpath = ta1_cropped_dir / region_id

    node_state.print_current_state(ingress_dir)

    from geowatch.geoannots.geomodels import RegionModel
    region = RegionModel.coerce(local_region_path)

    # Returned as datetime
    current_interval_end_date = region.end_date

    if current_interval_end_date.month == 1 and current_interval_end_date.day == 1:
        # If current interval ends at the start of a year,
        # consider the "current" year to be the previous one
        current_interval_year = current_interval_end_date.year - 1
    else:
        current_interval_year = current_interval_end_date.year

    # Download STAC input file locally
    local_stac_path = ingress_dir / 'input_stac.jsonl'
    input_stac_path = util_fsspec.FSPath.coerce(config.input_path)
    input_stac_path.copy(util_fsspec.FSPath.coerce(local_stac_path))

    # 3. Generate KWCOCO dataset from input STAC
    current_interval_kwcoco_path = input_stac_to_kwcoco(
        config.input_path,
        ingress_dir,
        ta1_kwcoco_path,
        target_gsd,
        aws_profile=config.aws_profile,
        dryrun=config.dryrun,
        requester_pays=config.requester_pays,
        jobs=config.jobs,
        virtual=config.virtual,
        from_collated=config.from_collated)

    # Overwritten if incremental mode (and not first interval)
    combined_kwcoco_path = current_interval_kwcoco_path

    # Setting 'combined_stac_input' here to ensure we have something
    # from the first interval.  Gets overwritten if not the first
    # interval
    incremental_assets_for_egress = {'combined_stac_input': local_stac_path}
    previous_ingressed_assets = None
    if config.previous_interval_output is not None:
        print('* Combining previous interval time combined kwcoco with '
              'current *')
        previous_ingress_dir = ub.Path('/tmp/ingress_previous')
        try:
            previous_ingressed_assets = smartflow_ingress(
                config.previous_interval_output,
                ['combined_stac_input',
                 'timecombined_kwcoco_file_for_bas',
                 'timecombined_kwcoco_file_for_bas_assets'],
                previous_ingress_dir,
                config.aws_profile,
                config.dryrun)
        except FileNotFoundError:
            print("** Warning: Couldn't ingress previous interval output; "
                  "assuming this is the first interval **")
        else:
            combined_stac_path = ingress_dir / 'combined_stac_items.jsonl'

            build_combined_stac(
                previous_ingressed_assets['combined_stac_input'],
                local_stac_path,
                combined_stac_path)

            incremental_assets_for_egress['combined_stac_input'] =\
                combined_stac_path

            # Perform the filtering
            selected_stac_items = []
            with open(combined_stac_path) as f:
                for line in f:
                    try:
                        stac_item = json.loads(line)
                    except json.decoder.JSONDecodeError:
                        print("** Warning: Couldn't parse STAC item from "
                              "'combined_stac_input', skipping!")
                        continue

                    if stac_item['properties']['datetime'].startswith(
                            str(current_interval_year)):
                        selected_stac_items.append(stac_item)

            filtered_stac_items_path = ingress_dir / 'filtered_stac_items.jsonl'
            with open(filtered_stac_items_path, 'w') as f:
                print('\n'.join((json.dumps(item)
                                 for item in selected_stac_items)), file=f)

            incremental_assets_for_egress['filtered_stac_input'] =\
                filtered_stac_items_path

            combined_kwcoco_path = ingress_dir / 'combined_timecombined_kwcoco.json'

            combined_kwcoco_path = input_stac_to_kwcoco(
                filtered_stac_items_path,
                ingress_dir,
                combined_kwcoco_path,
                target_gsd,
                aws_profile=config.aws_profile,
                dryrun=config.dryrun,
                requester_pays=config.requester_pays,
                jobs=config.jobs,
                virtual=config.virtual,
                from_collated=config.from_collated)

    # 3a. Filter KWCOCO dataset by sensors used for BAS
    # Will use either the combined KWCOCO dataset (for incremental
    # mode) or strictly the input STAC items
    print("* Filtering KWCOCO dataset for BAS")
    ub.cmd([
        'kwcoco', 'subset',
        '--src', combined_kwcoco_path,
        '--dst', ta1_bas_kwcoco_path,
        '--absolute', 'False',
        # '--select_images',
        # '.sensor_coarse == "L8" or .sensor_coarse == "S2"'
    ], check=True, verbose=3, capture=False)

    # 3b. Filter KWCOCO dataset by sensors used for SC
    # TODO: move this to run_sc_datagen
    print("* Filtering KWCOCO dataset for SC")
    ta1_sc_kwcoco_path = ta1_cropped_dir / 'kwcoco_for_sc.json'
    ub.cmd(['kwcoco', 'subset',
            '--src', ta1_kwcoco_path,
            '--dst', ta1_sc_kwcoco_path,
            '--absolute', 'False',
            # '--select_images',
            # '.sensor_coarse == "WV1" or .sensor_coarse == "WV" or .sensor_coarse == "S2"'
            ],
           check=True, verbose=3, capture=False)

    # 4. Crop ingress KWCOCO dataset to region for BAS
    print("* Cropping KWCOCO dataset to region for BAS*")

    ALIGN_EXEC_MODE = 'cmd'
    # Not sure if one is more stable than the other
    if ALIGN_EXEC_MODE == 'import':
        coco_align.main(cmdline=False, **align_config)
    elif ALIGN_EXEC_MODE == 'cmd':
        align_node = ProcessNode(
            command='python -m geowatch.cli.coco_align',
            config=align_config,
        )
        command = align_node.final_command()
        ub.cmd(command, check=True, capture=False, verbose=3, shell=True)
    else:
        raise KeyError(ALIGN_EXEC_MODE)

    ### Filter / clean geotiffs (probably should be a separate step)
    CLEAN_GEOTIFFS = 0
    if CLEAN_GEOTIFFS:
        # Detect blocky black regions in geotiffs and switch them to NODATA
        # Modifies geotiffs inplace
        remove_bad_images_node = ProcessNode(
            command='geowatch clean_geotiffs',
            in_paths={
                'src': ta1_cropped_kwcoco_path,
            },
            config={
                'prefilter_channels': 'red',
                'channels': 'red|green|blue|nir',
                'workers': 'avail',
                'dry': False,
                'probe_scale': None,
                'nodata_value': -9999,
                'min_region_size': 256,
            },
            node_dpath='.'
        )
        command = remove_bad_images_node.final_command()
        ub.cmd(command, shell=True, capture=False, verbose=3, check=True)

    REMOVE_BAD_IMAGES = 0
    if REMOVE_BAD_IMAGES:
        # Remove images that are nearly all nan
        remove_bad_images_node = ProcessNode(
            command='geowatch remove_bad_images',
            in_paths={
                'src': ta1_cropped_kwcoco_path,
            },
            out_paths={
                'dst': ta1_cropped_kwcoco_path,  # hack: this is inplace, fix it if we enable.
            },
            config={
                'workers': 'avail',
                'interactive': False,
                'overview': 0,
            },
            node_dpath='.'
        )
        command = remove_bad_images_node.final_command()
        ub.cmd(command, shell=True, capture=False, verbose=3, check=True)
    else:
        print('Not removing bad images. TODO: add support')
        # ta1_sc_cropped_kwcoco_prefilter_path.copy(ta1_sc_cropped_kwcoco_path)

    # Reroot the kwcoco files to be relative and make it easier to work with
    # downloaded results
    ub.cmd(['kwcoco', 'reroot', f'--src={ta1_cropped_kwcoco_path}', '--inplace=1', '--absolute=0'])

    node_state.print_current_state(ingress_dir)

    # 5. Do the time_combine for BAS
    if time_combine_enabled:
        try:
            coco_time_combine.main(
                cmdline=0,
                **time_combine_config
            )
        except Exception:
            if config.skip_timecombine_on_fail:
                print("WARNING: Exception occurred (printed below), generating empty KWCOCO for time-combined output")
                traceback.print_exception(*sys.exc_info())

                empty_dset_path = ta1_cropped_dir / 'empty.json'
                empty_dset = kwcoco.CocoDataset()
                empty_dset.dump(empty_dset_path)

                final_interval_bas_kwcoco_path = empty_dset_path
            else:
                raise
        else:
            # Add geowatch feilds
            print("* Adding geowatch feilds to time combined data *")
            coco_add_watch_fields.main(cmdline=False,
                                       src=preproc_kwcoco_fpath,
                                       dst=preproc_kwcoco_fpath,
                                       target_gsd=target_gsd,
                                       workers=config.jobs)
            final_interval_bas_kwcoco_path = preproc_kwcoco_fpath

        # Reroot the kwcoco files to be relative and make it easier to work with
        # downloaded results
        ub.cmd(['kwcoco', 'reroot', f'--src={final_interval_bas_kwcoco_path}', '--inplace=1', '--absolute=0'])

    else:
        final_interval_bas_kwcoco_path = ta1_cropped_kwcoco_path

    # 6.1. Combine previous interval time-combined data for BAS
    if config.previous_interval_output is not None and previous_ingressed_assets is not None:
        combined_timecombined_kwcoco_path =\
            ta1_cropped_dir / 'combined_timecombined_kwcoco.json'

        previous_timecombined_kwcoco_path = ub.Path(
            previous_ingressed_assets['timecombined_kwcoco_file_for_bas'])

        import kwcoco
        previous_timecombined_dset = kwcoco.CocoDataset(
            previous_timecombined_kwcoco_path)

        image_ids_to_remove =\
            [o["id"] for o in previous_timecombined_dset.images().objs
             if o['date_captured'].startswith(str(current_interval_year))]

        previous_timecombined_dset.remove_images(image_ids_to_remove)

        filtered_previous_timecombined_kwcoco_path =\
            previous_ingress_dir / 'filtered_combined_timecombined_kwcoco.json'

        incremental_assets_for_egress['filtered_combined_timecombined_kwcoco'] =\
            filtered_previous_timecombined_kwcoco_path

        previous_timecombined_dset.dump(
            filtered_previous_timecombined_kwcoco_path)

        # On first interval nothing will be copied down so need to
        # check that we have the input explicitly
        from geowatch.cli.concat_kwcoco_videos import concat_kwcoco_datasets
        if filtered_previous_timecombined_kwcoco_path.is_file() and len(previous_timecombined_dset.images()) > 0:
            # Don't bother to concatenate if previous (now filtered)
            # dset is empty (has no images)
            concat_kwcoco_datasets(
                (filtered_previous_timecombined_kwcoco_path,
                 final_interval_bas_kwcoco_path),
                combined_timecombined_kwcoco_path)
            # Copy saliency assets from previous bas fusion
            shutil.copytree(
                previous_ingress_dir / 'raw_bands',
                ta1_cropped_dir / 'raw_bands',
                dirs_exist_ok=True)
        else:
            # This is either the first interval, or previous
            # interval(s) only contain images from the current
            # interval year
            combined_timecombined_kwcoco_path = final_interval_bas_kwcoco_path
    else:
        combined_timecombined_kwcoco_path = final_interval_bas_kwcoco_path

    # 7. Egress (envelop KWCOCO dataset in a STAC item and egress;
    #    will need to recursive copy the kwcoco output directory up to
    #    S3 bucket)

    timecombined_rawband_dpath = ta1_cropped_dir / 'raw_bands'
    timecombined_teamfeat_dpath = ta1_cropped_dir / '_teamfeats'
    # Put a dummy file in these directories so we can upload a nearly-empty folder
    # to S3
    timecombined_rawband_dpath.ensuredir()
    (timecombined_rawband_dpath / 'dummy').write_text('dummy')
    timecombined_teamfeat_dpath.ensuredir()
    (timecombined_teamfeat_dpath / 'dummy').write_text('dummy')
    ta1_cropped_rawband_dpath.ensuredir()
    (ta1_cropped_rawband_dpath / 'dummy').write_text('dummy')

    node_state.print_current_state(ingress_dir)

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
        # JPC: Seems like the answer is no, for now. I've seen this used later
        # on in the sc datagen node, although the stac-to-kwcoco for sc could
        # be run there.
        'kwcoco_for_sc': ta1_sc_kwcoco_path,

        # We need to egress the temporally dense dataset for COLD
        'timedense_bas_kwcoco_file': ta1_cropped_kwcoco_path,
        'timedense_bas_kwcoco_rawbands': ta1_cropped_rawband_dpath,
        **incremental_assets_for_egress
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
