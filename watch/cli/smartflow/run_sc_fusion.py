#!/usr/bin/env python3
"""
See Old Version:
    ../../../scripts/run_sc_fusion_eval3_for_baseline.py

SeeAlso:
    ~/code/watch-smartflow-dags/KIT_TA2_PREEVAL10_PYENV.py
"""
import json
import os
import shutil
import sys
import traceback
import scriptconfig as scfg
import ubelt as ub


class SCFusionConfig(scfg.DataConfig):
    """
    Run TA-2 SC fusion as baseline framework component
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


def main():
    config = SCFusionConfig.cli(strict=True)
    print('config = {}'.format(ub.urepr(dict(config), nl=1, align=':')))
    run_sc_fusion_for_baseline(config)


def run_sc_fusion_for_baseline(config):
    from watch.cli.smartflow_ingress import smartflow_ingress
    from watch.cli.smartflow_egress import smartflow_egress
    from watch.tasks.fusion.predict import predict  # NOQA
    from watch.tasks.fusion.datamodules.temporal_sampling import TimeSampleError
    from watch.utils.util_framework import download_region, determine_region_id
    from kwutil.util_yaml import Yaml
    from watch.utils import util_framework

    if config.aws_profile is not None:
        # This should be sufficient, but it is not tested.
        from watch.utils import util_fsspec
        util_fsspec.S3Path._new_fs(profile=config.aws_profile)

    # 1. Ingress data
    print("* Running baseline framework kwcoco ingress *")
    ingress_dir = ub.Path('/tmp/ingress')
    ingressed_assets = smartflow_ingress(
        config.input_path,
        ['cropped_region_models_bas',
         'cropped_kwcoco_for_sc',
         'cropped_kwcoco_for_sc_assets'],
        ingress_dir,
        config.aws_profile,
        config.dryrun)

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
    cropped_region_models_bas = ingressed_assets['cropped_region_models_bas']

    site_models_outdir = ingress_dir / 'sc_out_site_models'
    os.makedirs(site_models_outdir, exist_ok=True)
    region_models_outdir = ingress_dir / 'sc_out_region_models'
    os.makedirs(region_models_outdir, exist_ok=True)

    site_models_manifest_outdir = ingress_dir / 'tracking_manifests_sc'
    os.makedirs(site_models_manifest_outdir, exist_ok=True)
    site_models_manifest_outpath = site_models_manifest_outdir / 'site_models_manifest.json'
    # Copy input region model into region_models outdir to be updated
    # (rather than generated from tracking, which may not have the
    # same bounds as the original)
    shutil.copy(local_region_path, region_models_outdir / f'{region_id}.geojson')

    # 3.1. Check that we have at least one "video" (BAS identified
    # site) to run over; if not skip SC fusion and KWCOCO to GeoJSON
    with open(ingressed_assets['cropped_kwcoco_for_sc']) as f:
        ingress_kwcoco_data = json.load(f)

    if len(ingress_kwcoco_data.get('videos', ())) > 0:
        # 3. Run fusion
        print('*********************')
        print("* Running SC fusion *")

        # TODO: remove these defaults or replace them with whatever is the
        # default in predict. The params should be fully given in the DAG, not
        # here.
        default_sc_pxl_config = ub.udict({
            # 'tta_fliprot': 0.0,
            # 'tta_time': 0.0,
            # 'chip_overlap': 0.3,
            # 'input_space_scale': '8GSD',
            # 'window_space_scale': '8GSD',
            # 'output_space_scale': '8GSD',
            # 'time_span': '6m',
            # 'time_sampling': 'auto',
            # 'time_steps': '12',
            # 'chip_dims': 'auto',
            # 'set_cover_algo': None,
            # 'resample_invalid_frames': 3,
            # 'observable_threshold': 0.0,
            # 'mask_low_quality': True,
            # 'drop_unused_frames': True,
            # 'num_workers': 2,
            # 'batch_size': 1,
            # 'write_workers': 0,
            # 'package_fpath': None,
        })

        sc_pxl_config = default_sc_pxl_config | Yaml.coerce(config.sc_pxl_config or {})

        if sc_pxl_config.get('package_fpath', None) is None:
            raise ValueError('Requires package_fpath')

        from watch.mlops import smart_pipeline
        sc_pxl = smart_pipeline.SC_HeatmapPrediction(root_dpath=ingress_dir)
        sc_pxl.configure({
            'pred_pxl_fpath': sc_fusion_kwcoco_path,
            'test_dataset': ingressed_assets['cropped_kwcoco_for_sc'],
        } | sc_pxl_config)

        try:
            ub.cmd(sc_pxl.command(), check=True, verbose=3, system=True)
            # Old explicit invocation doesnt update with mlops
            # Remove if the new mlops way works.
            # predict(devices='0,',
            #         write_preds=False,
            #         write_probs=True,
            #         with_change=False,
            #         with_saliency=False,
            #         with_class=True,
            #         test_dataset=ingressed_assets['cropped_kwcoco_for_sc'],
            #         pred_dataset=sc_fusion_kwcoco_path,
            #         **sc_pxl_config)
        except TimeSampleError:
            # FIXME: wont work anymore with mlops. Not sure if needed.
            # Can always catch a CalledProcessError and inspect stdout
            print("* Error with time sampling during SC Predict "
                  "(shown below) -- attempting to continue anyway")
            traceback.print_exception(*sys.exc_info())
        else:
            # 4. Compute tracks (SC)
            print('*************************')
            print("* Computing tracks (SC) *")

            # NOTE: These params are fully specified and overwritten in the DAG
            default_sc_track_kwargs = ub.udict({
                "boundaries_as": "polys",
                "resolution": 8,
                "min_area_square_meters": 7200,
                "thresh": 0.07,
            })
            sc_track_kwargs = default_sc_track_kwargs | Yaml.coerce(config.sc_poly_config or {})

            tracked_sc_kwcoco_path = '_tracked'.join(
                os.path.splitext(sc_fusion_kwcoco_path))
            region_models_manifest_fpath = ingress_dir / 'sc_out_region_models_manifest.json'

            sc_poly = smart_pipeline.SC_PolygonPrediction(root_dpath=ingress_dir)
            sc_poly.configure({
                'pred_pxl_fpath': sc_fusion_kwcoco_path,
                'site_summaries_fpath': region_models_manifest_fpath,
                'site_summaries_dpath': region_models_outdir,
                'sites_dpath': site_models_outdir,
                'sites_fpath': site_models_manifest_outpath,
                'append_mode': True,
                'poly_kwcoco_fpath': tracked_sc_kwcoco_path,
                'site_summary': ub.Path(cropped_region_models_bas) / '*.geojson',
            } | sc_track_kwargs)
            command = sc_poly.command()
            print(command)
            ub.cmd(command, check=True, verbose=3, system=True)

            # Old explicit invocation doesnt update with mlops
            # Remove if the new mlops way works.
            # These args are passed on the top level command line
            # Rather than in track-kwargs
            # external_args = {'site_score_thresh', 'smoothing'}
            # sc_extra_kwargs = sc_track_kwargs & external_args
            # sc_track_kwargs = sc_track_kwargs - external_args
            # sc_extra_argv = [f'--{k}={v}' for k, v in sc_extra_kwargs.items()]
            # tracker_argv = [
            #     'python', '-m', 'watch.cli.run_tracker',
            #     '--input_kwcoco', sc_fusion_kwcoco_path,
            #     '--out_site_summaries_dir', region_models_outdir,
            #     '--out_sites_dir', site_models_outdir,
            #     '--out_sites_fpath', site_models_manifest_outpath,
            #     '--out_kwcoco', tracked_sc_kwcoco_path,
            #     '--default_track_fn', 'class_heatmaps',
            #     '--site_summary', ub.Path(cropped_region_models_bas) / '*.geojson',
            #     '--append_mode', 'True',
            #     '--track_kwargs', json.dumps(sc_track_kwargs)
            # ] + sc_extra_argv
            # ub.cmd(tracker_argv, check=True, verbose=3, capture=False)

    cropped_site_models_outdir = ingress_dir / 'cropped_site_models'
    os.makedirs(cropped_site_models_outdir, exist_ok=True)
    cropped_region_models_outdir = ingress_dir / 'cropped_region_models'
    os.makedirs(cropped_region_models_outdir, exist_ok=True)

    ub.cmd([
        'python', '-m', 'watch.cli.crop_sites_to_regions',
        '--site_models', site_models_outdir / '*.geojson',
        '--region_models', region_models_outdir / f'{region_id}.geojson',
        '--new_site_dpath', cropped_site_models_outdir,
        '--new_region_dpath', cropped_region_models_outdir
    ], check=True, verbose=3, capture=False)

    # Validate and fix all outputs
    print('Fixup and validate outputs')
    util_framework.fixup_and_validate_site_and_region_models(
        region_dpath=cropped_region_models_outdir,
        site_dpath=cropped_site_models_outdir,
    )

    # 5. Egress (envelop KWCOCO dataset in a STAC item and egress;
    #    will need to recursive copy the kwcoco output directory up to
    #    S3 bucket)
    print("* Egressing KWCOCO dataset and associated STAC item *")
    ingressed_assets['cropped_site_models_sc'] = cropped_site_models_outdir
    ingressed_assets['cropped_region_models_sc'] = cropped_region_models_outdir

    # Add in intermediate outputs for debugging
    ingressed_assets['sc_heatmap_kwcoco_file'] = sc_fusion_kwcoco_path
    ingressed_assets['sc_tracked_kwcoco_file'] = tracked_sc_kwcoco_path

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
