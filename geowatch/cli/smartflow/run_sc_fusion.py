#!/usr/bin/env python3
"""
SeeAlso:
    ~/code/watch-smartflow-dags/KIT_TA2_PREEVAL10_PYENV.py
"""
import os
import shutil
import sys
import traceback
import scriptconfig as scfg
import ubelt as ub



__debugging__ = r"""

LOCAL_WORK_DPATH=$HOME/temp/debug_smartflow_v2_kr2-sv/ingress
mkdir -p $LOCAL_WORK_DPATH
cd $LOCAL_WORK_DPATH

docker run \
    --runtime=nvidia \
    --volume "$LOCAL_WORK_DPATH":/tmp/ingress \
    --volume $HOME/.aws:/root/.aws:ro \
    --volume "$HOME/code":/extern_code:ro \
    --volume "$HOME/data":/extern_data:ro \
    --volume "$HOME"/.cache/pip:/pip_cache \
    --env AWS_PROFILE=iarpa \
    -it registry.smartgitlab.com/kitware/geowatch:0.16.2-17a6765ff-strict-pyenv3.11.2-20240410T192350-0400-from-0da55667 bash

ipython
from geowatch.cli.smartflow.run_sc_fusion import *  # NOQA

config = SCFusionConfig(**{
    'input_path'                    : 's3://smartflow-023300502152-us-west-2/smartflow/env/kw-v3-0-0/work/preeval21_batch_v193/batch/kit/KR_R002/split_work/52SDG98/products/ac_datagen/items.jsonl',
    'input_region_path'             : 's3://smartflow-023300502152-us-west-2/smartflow/env/kw-v3-0-0/work/preeval21_batch_v193/batch/kit/KR_R002/split_input/52SDG98/region_models/KR_R002.geojson',
    'output_path'                   : 's3://smartflow-023300502152-us-west-2/smartflow/env/kw-v3-0-0/work/preeval21_batch_v193/batch/kit/KR_R002/split_work/52SDG98/products/sc-fusion/items.jsonl',
    'aws_profile'                   : None,
    'dryrun'                        : False,
    'outbucket'                     : 's3://smartflow-023300502152-us-west-2/smartflow/env/kw-v3-0-0/work/preeval21_batch_v193/batch/kit/KR_R002/split_work/52SDG98/products/sc-fusion',
    'ta2_s3_collation_bucket'       : None,
    'sc_pxl_config'                 : 'batch_size: 1\nchip_dims: auto\nchip_overlap: 0.3\ndrop_unused_frames: true\ninput_space_scale: 4GSD\nmask_low_quality: true\nnum_workers: 8\nobservable_threshold: 0.0\noutput_space_scale: 4GSD\npackage_fpath: /root/data/smart_expt_dvc/models/fusion/Drop7-Cropped2GSD/packages/Drop7-Cropped2GSD_SC_bgrn_gnt_split6_V84/Drop7-Cropped2GSD_SC_bgrn_gnt_split6_V84_epoch17_step1548.pt\nresample_invalid_frames: 3\nset_cover_algo: null\ntta_fliprot: 0.0\ntta_time: 0.0\nwindow_space_scale: 4GSD\nwrite_workers: 0',
    'sc_poly_config'                : 'boundaries_as: bounds\nmin_area_square_meters: 7200\nnew_algo: crall\npolygon_simplify_tolerance: 1\nresolution: 8GSD\nsite_score_thresh: 0.3\nsmoothing: 0.0\nthresh: 0.3',
    'input_region_models_asset_name': 'sv_out_region_models',
    'input_site_models_asset_name'  : 'sv_out_site_models',
    'egress_intermediate_outputs'   : 1,
})

"""


class SCFusionConfig(scfg.DataConfig):
    """
    Run TA-2 SC fusion as baseline framework component
    """
    input_path = scfg.Value(None, type=str, position=1, required=True, help=ub.paragraph(
        '''
        Path to the STAC items this step can use as inputs.
        This is usually an S3 Path.
        '''), alias=['input_stac_path'])

    input_region_path = scfg.Value(None, type=str, position=2, required=True, help=ub.paragraph(
        '''
        Path to input T&E Baseline Framework Region definition JSON
        '''))

    output_path = scfg.Value(None, type=str, position=3, required=True, help=ub.paragraph(
        '''
        Path to the STAC items that register the outputs of this stage.
        This is usually an S3 Path.
        '''), alias=['output_stac_path'])

    aws_profile = scfg.Value(None, type=str, help=ub.paragraph(
        '''
        AWS Profile to use for AWS S3 CLI commands
        '''))
    dryrun = scfg.Value(False, isflag=True, short_alias=['d'], help='DEPRECATED. DO NOT USE')
    outbucket = scfg.Value(None, type=str, required=True, short_alias=['o'], help=ub.paragraph(
        '''
        S3 Output directory for STAC item / asset egress
        '''))

    ta2_s3_collation_bucket = scfg.Value(None, type=str, help=ub.paragraph(
        '''
        S3 Location for collated TA-2 output (bucket name should
        include up to eval name)
        '''))

    sc_pxl_config = scfg.Value(None, type=str, help=ub.paragraph(
        '''
        Raw json/yaml or a path to a json/yaml file that specifies the
        config for fusion.predict.
        '''))

    sc_poly_config = scfg.Value(None, type=str, help=ub.paragraph(
        '''
        Raw json/yaml or a path to a json/yaml file that specifies the
        config for SC tracking.
        '''))

    input_region_models_asset_name = scfg.Value('sv_out_region_models', type=str, required=False, help=ub.paragraph(
        '''
        Which region model assets to use as input
        '''))

    input_site_models_asset_name = scfg.Value('sv_out_site_models', type=str, required=False, help=ub.paragraph(
        '''
        Which site model assets to to use as input
        '''))

    egress_intermediate_outputs = scfg.Value(False, isflag=True, help=ub.paragraph(
        '''
        If true egress intermediate heatmaps, otherwise only egress the geojson
        '''))


def main(cmdline=1, **kwargs):
    config = SCFusionConfig.cli(cmdline=cmdline, data=kwargs, strict=True)
    print('config = {}'.format(ub.urepr(config, nl=1, align=':')))
    run_sc_fusion_for_baseline(config)


def run_sc_fusion_for_baseline(config):
    from geowatch.cli.smartflow_ingress import smartflow_ingress
    from geowatch.cli.smartflow_egress import smartflow_egress
    from geowatch.tasks.fusion.predict import predict  # NOQA
    from geowatch.tasks.fusion.datamodules.temporal_sampling import TimeSampleError
    from geowatch.utils.util_framework import download_region, determine_region_id
    from kwutil.util_yaml import Yaml
    from geowatch.utils import util_framework
    from geowatch.mlops import smart_pipeline
    import kwcoco

    if config.aws_profile is not None:
        # This should be sufficient, but it is not tested.
        # TODO: use the new register_bucket API.
        from geowatch.utils import util_fsspec
        util_fsspec.S3Path._new_fs(profile=config.aws_profile)

    from geowatch.utils.util_framework import NodeStateDebugger
    node_state = NodeStateDebugger()
    node_state.print_environment()
    node_state.print_local_invocation(config)

    # 1. Ingress data
    print("* Running baseline framework kwcoco ingress *")

    # ingress_dir = ub.Path('/home/joncrall/data/dvc-repos/smart_expt_dvc/_airflow/temp').ensuredir()
    ingress_dir = ub.Path('/tmp/ingress')

    # input_region_asset_name = 'sv_out_region_models'
    input_region_asset_name = config.input_region_models_asset_name

    ingressed_assets = smartflow_ingress(
        input_path=config.input_path,
        assets=[
            # {'key': 'cropped_region_models_bas'},
            # {'key': 'sv_out_region_models', 'allow_missing': False},

            {'key': input_region_asset_name, 'allow_missing': False},

            # {'key': 'cropped_kwcoco_for_sc'},
            # {'key': 'cropped_kwcoco_for_sc_assets'}
            {'key': 'enriched_acsc_kwcoco_file'},
            {'key': 'enriched_acsc_kwcoco_teamfeats'},
            {'key': 'enriched_acsc_kwcoco_rawbands'},
        ],
        outdir=ingress_dir,
        aws_profile=config.aws_profile,
        dryrun=config.dryrun
    )

    input_site_summary_dpath = ingressed_assets[input_region_asset_name]
    assert os.path.exists(input_site_summary_dpath)
    print(f'Found input site summary dpath: {input_site_summary_dpath}')
    ub.cmd(f'geowatch site_stats "{input_site_summary_dpath}/*.geojson"', verbose=3)

    # # 2. Download and prune region file
    print("* Downloading and pruning region file *")
    local_region_path = '/tmp/region.json'

    download_region(
        input_region_path=config.input_region_path,
        output_region_path=local_region_path,
        aws_profile=config.aws_profile,
        strip_nonregions=True,
    )

    # Determine the region_id in the region file.
    region_id = determine_region_id(local_region_path)

    sc_fusion_kwcoco_path = ingress_dir / 'sc_fusion_kwcoco.json'
    tracked_sc_kwcoco_path = sc_fusion_kwcoco_path.augment(stemsuffix='_tracked')

    site_models_outdir = (ingress_dir / 'sc_out_site_models').ensuredir()
    region_models_outdir = (ingress_dir / 'sc_out_region_models').ensuredir()
    site_models_manifest_outdir = (ingress_dir / 'tracking_manifests_sc').ensuredir()

    site_models_manifest_outpath = site_models_manifest_outdir / 'site_models_manifest.json'
    # Copy input region model into region_models outdir to be updated
    # (rather than generated from tracking, which may not have the
    # same bounds as the original)
    shutil.copy(local_region_path, region_models_outdir / f'{region_id}.geojson')

    region_models_manifest_fpath = ingress_dir / 'sc_out_region_models_manifest.json'

    node_state.print_current_state(ingress_dir)

    # 3.1. Check that we have at least one "video" (BAS identified
    # site) to run over; if not skip SC fusion and KWCOCO to GeoJSON

    # TODO: could use kwcoco info to get lazy loading of just the header.
    input_kwcoco_fpath = ingressed_assets['enriched_acsc_kwcoco_file']
    ingress_dset = kwcoco.CocoDataset(input_kwcoco_fpath, autobuild=False)
    if ingress_dset.n_videos > 0:
        # 3. Run fusion
        print('*********************')
        print("* Running SC fusion *")

        ub.cmd(f'kwcoco stats {input_kwcoco_fpath}', verbose=3)
        ub.cmd(f'geowatch stats {input_kwcoco_fpath}', verbose=3)

        # The params should be fully given in the DAG.
        sc_pxl_config = Yaml.coerce(config.sc_pxl_config or {})
        if sc_pxl_config.get('package_fpath', None) is None:
            raise ValueError('Requires package_fpath')

        # Debugging
        # foo = '/root/data/smart_expt_dvc/models/fusion/Drop7-Cropped2GSD/packages/Drop7-Cropped2GSD_SC_bgrn_gnt_split6_V84/Drop7-Cropped2GSD_SC_bgrn_gnt_split6_V84_epoch17_step1548.pt'
        # sc_pxl_config['package_fpath'] = foo
        # sc_pxl_config['set_cover_algo'] = 'approx'

        sc_pxl = smart_pipeline.SC_HeatmapPrediction(root_dpath=ingress_dir)
        sc_pxl.configure({
            'pred_pxl_fpath': sc_fusion_kwcoco_path,
            'test_dataset': ingressed_assets['enriched_acsc_kwcoco_file'],
        } | sc_pxl_config)

        if False:
            try:
                # Quick inspection of how big the predicted videos are going to be.
                from geowatch.utils import util_resolution
                from geowatch.utils import util_units
                import numpy as np
                # from geowatch.tasks.fusion.datamodules.kwcoco_dataset import KWCocoVideoDatasetConfig
                # sc_pxl_config['chip_dims']
                # input_resolution = util_resolution.ResolvedUnit.coerce(sc_pxl_config['input_space_scale'])
                num_output_channels = 6  # hard coded for the quick check
                for video in ingress_dset.dataset['videos']:
                    output_resolution = util_resolution.ResolvedUnit.coerce(sc_pxl_config['output_space_scale'])
                    output_resolution = util_resolution.ResolvedUnit.coerce('4GSD')
                    vidname = video['name']
                    vid_width = video['width']
                    vid_height = video['height']
                    num_frames = video['num_frames']
                    video_resolution = util_resolution.ResolvedUnit.coerce(video['resolution'])
                    video_dsize = util_resolution.ResolvedWindow((vid_width, vid_height), video_resolution)
                    # input_dsize = video_dsize.at_resolution(input_resolution)
                    output_dsize = video_dsize.at_resolution(output_resolution)
                    output_frame_pixels = np.prod(output_dsize.window)
                    bytes_per_cell = np.dtype('float32').itemsize
                    num_raster_cells = output_frame_pixels * num_frames
                    total_bytes = num_raster_cells * bytes_per_cell
                    ureg = util_units.unit_registry()
                    size_per_channel = (total_bytes * ureg.bytes).to('gigabytes')
                    total_size = num_output_channels * size_per_channel
                    print(f'{vidname} - {total_size:0.2f} - WH={output_dsize},  T={num_frames}, C={num_output_channels}')

                    memory_max_gb = 8
                    factor = np.sqrt(total_size.m / memory_max_gb)
                    if factor > 1:
                        output_resolution = util_resolution.ResolvedUnit(output_resolution.mag * factor, output_resolution.unit)
                        output_dsize = video_dsize.at_resolution(output_resolution)
                        output_frame_pixels = np.prod(output_dsize.window)
                        bytes_per_cell = np.dtype('float32').itemsize
                        num_raster_cells = output_frame_pixels * num_frames
                        total_bytes = num_raster_cells * bytes_per_cell
                        ureg = util_units.unit_registry()
                        size_per_channel = (total_bytes * ureg.bytes).to('gigabytes')
                        total_size = num_output_channels * size_per_channel
                        # output_resolution = util_resolution.ResolvedUnit.coerce(sc_pxl_config['output_space_scale'])
                    print(f'{vidname} - {total_size:0.2f} - WH={output_dsize},  T={num_frames}, C={num_output_channels}')

                ingress_dset.videos()
            except Exception as ex:
                print('Quick Introspection failed:')
                print(f'ex = {ub.urepr(ex, nl=1)}')

        command = sc_pxl.command()
        print(command)
        try:
            ub.cmd(command, check=True, verbose=3, system=True)
            node_state.print_current_state(ingress_dir)
            # ub.cmd(['kwcoco', 'reroot', f'--src={sc_fusion_kwcoco_path}', '--inplace=1', '--absolute=0'])
            # ub.cmd(['geowatch', 'visualize', f'{sc_fusion_kwcoco_path}', '--smart'], verbose=3)
        except TimeSampleError:
            # FIXME: wont work anymore with mlops. Not sure if needed.
            # Can always catch a CalledProcessError and inspect stdout
            print("* Error with time sampling during SC Predict "
                  "(shown below) -- attempting to continue anyway")
            traceback.print_exception(*sys.exc_info())
        else:
            ub.cmd(f'kwcoco stats {sc_fusion_kwcoco_path}', verbose=3)
            ub.cmd(f'geowatch stats {sc_fusion_kwcoco_path}', verbose=3)

            # 4. Compute tracks (SC)
            print('*************************')
            print("* Computing tracks (SC) *")

            # Params are fully specified in the DAG
            sc_track_kwargs = Yaml.coerce(config.sc_poly_config or {})
            final_sc_poly_config = {
                'pred_pxl_fpath': sc_fusion_kwcoco_path,               # Sets --input_kwcoco
                'site_summaries_fpath': region_models_manifest_fpath,  # Sets --out_site_summaries_fpath
                'site_summaries_dpath': region_models_outdir,          # Sets --out_site_summaries_dir
                'sites_dpath': site_models_outdir,                     # Sets --out_sites_dir
                'sites_fpath': site_models_manifest_outpath,           # Sets --out_sites_fpath
                'poly_kwcoco_fpath': tracked_sc_kwcoco_path,           # Sets --out_kwcoco
                'site_summary': ub.Path(input_site_summary_dpath) / '*.geojson',  # Sets --site_summary
                'append_mode': True,
            } | sc_track_kwargs
            # See: ~/code/watch/geowatch/mlops/smart_pipeline.py
            sc_poly = smart_pipeline.SC_PolygonPrediction(root_dpath=ingress_dir)
            sc_poly.configure(final_sc_poly_config)
            command = sc_poly.command()
            ub.cmd(command, check=True, verbose=3, system=True)

            node_state.print_current_state(ingress_dir)

            ub.cmd(f'kwcoco stats {tracked_sc_kwcoco_path}', verbose=3)
            ub.cmd(f'geowatch stats {tracked_sc_kwcoco_path}', verbose=3)
    else:
        print('Warning: No Videos in Ingress Dataset, Skipping Predict!')

    cropped_site_models_outdir = ingress_dir / 'cropped_site_models_sc'
    cropped_region_models_outdir = ingress_dir / 'cropped_region_models_sc'
    sc_heatmap_dpath = (ingress_dir / '_assets').ensuredir()

    cropped_site_models_outdir.ensuredir()
    cropped_region_models_outdir.ensuredir()

    ub.cmd([
        'python', '-m', 'geowatch.cli.crop_sites_to_regions',
        '--site_models', site_models_outdir / '*.geojson',
        '--region_models', region_models_outdir / f'{region_id}.geojson',
        '--new_site_dpath', cropped_site_models_outdir,
        '--new_region_dpath', cropped_region_models_outdir
    ], check=True, verbose=3, capture=False)

    node_state.print_current_state(ingress_dir)

    # Validate and fix all outputs
    print('Fixup and validate outputs')
    util_framework.fixup_and_validate_site_and_region_models(
        region_dpath=cropped_region_models_outdir,
        site_dpath=cropped_site_models_outdir,
    )

    node_state.print_current_state(ingress_dir)
    node_state.print_directory_contents(cropped_site_models_outdir)
    node_state.print_directory_contents(cropped_region_models_outdir)

    # 5. Egress (envelop KWCOCO dataset in a STAC item and egress;
    #    will need to recursive copy the kwcoco output directory up to
    #    S3 bucket)
    print("* Egressing KWCOCO dataset and associated STAC item *")
    ingressed_assets['cropped_site_models_sc'] = cropped_site_models_outdir
    ingressed_assets['cropped_region_models_sc'] = cropped_region_models_outdir

    # Add in intermediate outputs for debugging
    EGRESS_INTERMEDIATE_OUTPUTS = config.egress_intermediate_outputs
    if EGRESS_INTERMEDIATE_OUTPUTS:
        # Reroot kwcoco files to make downloaded results easier to work with
        if sc_fusion_kwcoco_path.exists():
            ub.cmd(['kwcoco', 'reroot', f'--src={sc_fusion_kwcoco_path}', '--inplace=1', '--absolute=0'])
            ingressed_assets['sc_heatmap_kwcoco_file'] = sc_fusion_kwcoco_path
        if tracked_sc_kwcoco_path.exists():
            ub.cmd(['kwcoco', 'reroot', f'--src={tracked_sc_kwcoco_path}', '--inplace=1', '--absolute=0'])
            ingressed_assets['sc_tracked_kwcoco_file'] = tracked_sc_kwcoco_path
        ingressed_assets['sc_heatmap_assets'] = sc_heatmap_dpath
        ingressed_assets['sc_tracking_manifest_dpath'] = site_models_manifest_outdir
        if region_models_manifest_fpath.exists():
            ingressed_assets['sc_tracking_manifest_fpath'] = region_models_manifest_fpath

    smartflow_egress(ingressed_assets,
                     local_region_path,
                     config.output_path,
                     config.outbucket,
                     aws_profile=config.aws_profile,
                     dryrun=False,
                     newline=False)

    # 6. (Optional) collate TA-2 output
    if config.ta2_s3_collation_bucket is not None:
        print("* Collating TA-2 output")
        util_framework.ta2_collate_output(None,
                                          cropped_region_models_outdir,
                                          cropped_site_models_outdir,
                                          config.ta2_s3_collation_bucket)


if __name__ == "__main__":
    main()
