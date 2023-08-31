#!/usr/bin/env python3
"""
See Old Version:
    ../../../scripts/run_bas_fusion_eval3_for_baseline.py

SeeAlso:
    ~/code/watch-smartflow-dags/KIT_TA2_PREEVAL10_PYENV.py
"""
import json
import os
import scriptconfig as scfg
import shutil
import ubelt as ub


class BasFusionConfig(scfg.DataConfig):
    """
    Run TA-2 BAS fusion as baseline framework component

    python ~/code/watch/watch/cli/smartflow/run_bas_fusion.py
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
            AWS Profile to use for AWS S3 CLI commands.
            '''))

    dryrun = scfg.Value(False, isflag=True, short_alias=['d'], help='DEPRECATED. DO NOT USE.')

    outbucket = scfg.Value(None, type=str, required=True, short_alias=['o'], help=ub.paragraph(
            '''
            S3 Output directory for STAC item / asset egress
            '''))

    ta2_s3_collation_bucket = scfg.Value(None, type=str, help=ub.paragraph(
            '''
            S3 Location for collated TA-2 output (bucket name should
            include up to eval name)
            '''))

    previous_bas_outbucket = scfg.Value(None, type=str, help=ub.paragraph(
            '''
            S3 Output directory for previous interval BAS fusion output
            '''))

    bas_pxl_config = scfg.Value(None, type=str, help=ub.paragraph(
            '''
            Raw json/yaml or a path to a json/yaml file that specifies the
            config for fusion.predict.
            '''))

    bas_poly_config = scfg.Value(None, type=str, help=ub.paragraph(
            '''
            Raw json/yaml or a path to a json/yaml file that specifies the
            config for bas tracking.
            '''))


def main():
    config = BasFusionConfig.cli(strict=True)
    print('config = {}'.format(ub.urepr(dict(config), nl=1, align=':')))
    run_bas_fusion_for_baseline(config)


def run_bas_fusion_for_baseline(config):
    from watch.cli.smartflow_ingress import smartflow_ingress
    from watch.cli.smartflow_egress import smartflow_egress
    from watch.cli.concat_kwcoco_videos import concat_kwcoco_datasets
    from watch.utils.util_framework import download_region, determine_region_id
    from watch.tasks.fusion.predict import predict
    from kwutil.util_yaml import Yaml
    from watch.utils import util_framework
    from watch.utils import util_fsspec

    input_path = config.input_path
    input_region_path = config.input_region_path
    outbucket = config.outbucket
    aws_profile = config.aws_profile
    dryrun = config.dryrun
    previous_bas_outbucket = config.previous_bas_outbucket
    ta2_s3_collation_bucket = config.ta2_s3_collation_bucket

    if aws_profile is not None:
        # This should be sufficient, but it is not tested.
        util_fsspec.S3Path._new_fs(profile=aws_profile)

    assert not dryrun, 'unsupported'

    # 1. Ingress data
    print("* Running baseline framework kwcoco ingress *")
    ingress_dir = ub.Path('/tmp/ingress')
    ingressed_assets = smartflow_ingress(
        input_path,
        ['enriched_bas_kwcoco_file',
         'enriched_bas_kwcoco_teamfeats',
         'enriched_bas_kwcoco_rawbands',
         {"key": 'hacked_cold_assets', "allow_missing": True},
         {"key": 'landcover_assets', "allow_missing": True},
         ],
        ingress_dir,
        aws_profile,
        dryrun)

    # # 2. Download and prune region file
    print("* Downloading and pruning region file *")
    local_region_path = '/tmp/region.json'

    download_region(
        input_region_path=input_region_path,
        output_region_path=local_region_path,
        aws_profile=aws_profile,
        strip_nonregions=True,
        ensure_comments=True,
    )

    # Determine the region_id in the region file.
    region_id = determine_region_id(local_region_path)

    # 3. Run fusion
    print("* Running BAS fusion *")
    bas_fusion_kwcoco_path = ingress_dir / 'bas_fusion_kwcoco.json'

    # TODO: remove these defaults or replace them with whatever is the
    # default in predict. The params should be fully given in the DAG, not
    # here.
    default_predict_config = ub.udict({
          "chip_overlap": 0.3,
          "chip_dims": "auto",
          "time_span": "auto",
          "time_sampling": "auto",
          "drop_unused_frames": True,
          "batch_size": 1,
          "num_workers": 2,
          'package_fpath': None,
    })
    bas_pxl_config = default_predict_config | Yaml.coerce(config.bas_pxl_config or {})

    if bas_pxl_config.get('package_fpath', None) is None:
        raise ValueError('Requires package_fpath')

    ingress_kwcoco_path = ingressed_assets['enriched_bas_kwcoco_file']
    predict(devices='0,',
            write_preds=False,
            write_probs=True,
            with_change=False,
            with_saliency=True,
            with_class=False,
            test_dataset=ingress_kwcoco_path,
            pred_dataset=bas_fusion_kwcoco_path,
            **bas_pxl_config)

    # 3.1. If a previous interval was run; concatenate BAS fusion
    # output KWCOCO files for tracking
    if previous_bas_outbucket is not None:
        combined_bas_fusion_kwcoco_path = ingress_dir / 'combined_bas_fusion_kwcoco.json'

        previous_ingress_dir = ub.Path('/tmp/ingress_previous')

        previous_bas_outbucket = util_fsspec.FSPath.coerce(previous_bas_outbucket)
        previous_ingress_dir = util_fsspec.FSPath.coerce(previous_ingress_dir)
        previous_bas_outbucket.copy(previous_ingress_dir, recursive=True)

        previous_bas_fusion_kwcoco_path = previous_ingress_dir / 'combined_bas_fusion_kwcoco.json'

        # On first interval nothing will be copied down so need to
        # check that we have the input explicitly
        if os.path.isfile(previous_bas_fusion_kwcoco_path):
            concat_kwcoco_datasets(
                (previous_bas_fusion_kwcoco_path, bas_fusion_kwcoco_path),
                combined_bas_fusion_kwcoco_path)
            # Copy saliency assets from previous bas fusion
            shutil.copy_tree(
                previous_ingress_dir / '_assets/pred_saliency',
                ingress_dir / '_assets/pred_saliency'
            )

            # Copy original assets from previous bas rusion
            shutil.copy_tree(
                previous_ingress_dir / region_id,
                ingress_dir / region_id
            )
        else:
            # Copy current bas_fusion_kwcoco_path to combined path as
            # this is the first interval
            shutil.copy(bas_fusion_kwcoco_path,
                        combined_bas_fusion_kwcoco_path)
    else:
        combined_bas_fusion_kwcoco_path = bas_fusion_kwcoco_path

    # 4. Compute tracks (BAS)
    print("* Computing tracks (BAS) *")
    region_models_outdir = ingress_dir / 'region_models'
    os.makedirs(region_models_outdir, exist_ok=True)

    bas_site_models_outdir = ingress_dir / 'site_models_bas'
    os.makedirs(bas_site_models_outdir, exist_ok=True)

    region_models_manifest_outdir = ingress_dir / 'tracking_manifests_bas'
    os.makedirs(region_models_manifest_outdir, exist_ok=True)
    region_models_manifest_outpath = region_models_manifest_outdir / 'region_models_manifest.json'

    site_models_manifest_outpath = region_models_manifest_outdir / 'site_models_manifest.json'

    # Copy input region model into region_models outdir to be updated
    # (rather than generated from tracking, which may not have the
    # same bounds as the original)
    shutil.copy(local_region_path, region_models_outdir / f'{region_id}.geojson')

    # TODO: remove these defaults or replace them with whatever is the
    # default in tracker. The params should be fully given in the DAG,
    # not here.
    default_bas_tracking_config = ub.udict({
        "thresh": 0.1,
        "moving_window_size": None,
        "polygon_simplify_tolerance": 1,
        "max_area_behavior": 'ignore'
    })
    bas_tracking_config = (default_bas_tracking_config
                           | Yaml.coerce(config.bas_poly_config or {}))

    min_area_square_meters = bas_tracking_config.get('min_area_square_meters', None)

    tracked_bas_kwcoco_path = '_tracked'.join(
        os.path.splitext(bas_fusion_kwcoco_path))
    ub.cmd([
        'python', '-m', 'watch.cli.run_tracker',
        combined_bas_fusion_kwcoco_path,
        '--out_sites_dir', bas_site_models_outdir,
        '--out_sites_fpath', site_models_manifest_outpath,
        '--out_site_summaries_dir', region_models_outdir,
        '--out_site_summaries_fpath',
        region_models_manifest_outpath,
        '--out_kwcoco', tracked_bas_kwcoco_path,
        '--default_track_fn', 'saliency_heatmaps',
        '--append_mode', 'True',
        # TODO:
        # use boundary_region here?
        '--track_kwargs', json.dumps(bas_tracking_config)],
        check=True, verbose=3)

    # Remove after boundary_region here
    cropped_region_models_outdir = ingress_dir / 'cropped_region_models_bas'
    os.makedirs(cropped_region_models_outdir, exist_ok=True)

    cropped_site_models_outdir = ingress_dir / 'cropped_site_models_bas'
    os.makedirs(cropped_site_models_outdir, exist_ok=True)

    crop_cmd = [
        'python', '-m', 'watch.cli.crop_sites_to_regions',
        '--site_models', bas_site_models_outdir / '*.geojson',
        '--region_models', region_models_outdir / '*.geojson',
        '--new_site_dpath', cropped_site_models_outdir,
        '--new_region_dpath', cropped_region_models_outdir,
        '--min_area_square_meters', str(min_area_square_meters)
    ]
    ub.cmd(crop_cmd, check=True, verbose=3)

    # Validate and fix all outputs
    try:
        util_framework.fixup_and_validate_site_and_region_models(
            region_dpath=cropped_region_models_outdir,
            site_dpath=cropped_site_models_outdir,
        )
    except Exception as ex:  # noqa
        print(f'Encountered Exception, ex={ex}, uploading debug informaiton and then exiting')
        debug_s3_outdir = os.path.join(outbucket, '_debug/')
        cropped_region_models_outdir = util_fsspec.FSPath.coerce(cropped_region_models_outdir)
        debug_s3_outdir = util_fsspec.FSPath.coerce(debug_s3_outdir)
        cropped_region_models_outdir.copy(debug_s3_outdir, recursive=True)
        cropped_site_models_outdir.copy(debug_s3_outdir, recursive=True)
        raise

    # 6. (Optional) collate TA-2 output
    if ta2_s3_collation_bucket is not None:
        # only used if we are going to submit our BAS predictions as the final
        # ones?
        print("* Collating TA-2 output")
        util_framework.ta2_collate_output(None,
                                          cropped_region_models_outdir,
                                          cropped_site_models_outdir,
                                          ta2_s3_collation_bucket)

    # 6. Egress (envelop KWCOCO dataset in a STAC item and egress;
    #    will need to recursive copy the kwcoco output directory up to
    #    S3 bucket)
    print("* Egressing KWCOCO dataset and associated STAC item *")
    ingressed_assets['cropped_region_models_bas'] = cropped_region_models_outdir
    ingressed_assets['cropped_site_models_bas'] = cropped_site_models_outdir
    smartflow_egress(ingressed_assets,
                     local_region_path,
                     config.output_path,
                     config.outbucket,
                     aws_profile=None,
                     dryrun=False,
                     newline=False)


if __name__ == "__main__":
    main()
