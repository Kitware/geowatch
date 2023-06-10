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
import subprocess
import ubelt as ub
from glob import glob


class BasFusionConfig(scfg.DataConfig):
    """
    Run TA-2 BAS fusion as baseline framework component

    python ~/code/watch/watch/cli/dag_cli/run_bas_fusion.py
    """
    __fuzzy_hyphens__ = True

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

    dryrun = scfg.Value(False, isflag=True, short_alias=['d'], help='Run AWS CLI commands with --dryrun flag')

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


def _upload_region(aws_base_command,
                   local_region_dir,
                   local_input_region_path,
                   destination_region_s3):
    with open(local_input_region_path) as f:
        region = json.load(f)

    region_id = None
    for feature in region.get('features', ()):
        props = feature['properties']
        if props['type'] == 'region':
            region_id = props.get('region_model_id', props.get('region_id'))
            break

    if region_id is not None:
        updated_region_path = os.path.join(local_region_dir,
                                           '{}.geojson'.format(region_id))

        print("** Uploading updated region file")
        subprocess.run([*aws_base_command,
                        updated_region_path, destination_region_s3],
                       check=True)
    else:
        print("** Error: Couldn't parse region_id from region file, "
              "not uploading")


def _ta2_collate_output(aws_base_command,
                        local_region_dir,
                        local_sites_dir,
                        destination_s3_bucket,
                        performer_suffix='KIT'):
    def _get_suffixed_basename(local_path):
        base, ext = os.path.splitext(os.path.basename(local_path))
        return "{}_{}{}".format(base, performer_suffix, ext)

    for region in glob(os.path.join(local_region_dir, '*.geojson')):
        region_s3_outpath = '/'.join((destination_s3_bucket,
                                      'region_models',
                                      _get_suffixed_basename(region)))
        subprocess.run([*aws_base_command,
                        region,
                        region_s3_outpath], check=True)

    for site in glob(os.path.join(local_sites_dir, '*.geojson')):
        site_s3_outpath = '/'.join((destination_s3_bucket,
                                    'site_models',
                                    _get_suffixed_basename(site)))
        subprocess.run([*aws_base_command,
                        site,
                        site_s3_outpath], check=True)


def run_bas_fusion_for_baseline(config):
    from watch.cli.baseline_framework_kwcoco_ingress import baseline_framework_kwcoco_ingress
    from watch.cli.baseline_framework_kwcoco_egress import baseline_framework_kwcoco_egress
    from watch.cli.concat_kwcoco_videos import concat_kwcoco_datasets
    from watch.utils.util_framework import download_region, determine_region_id
    from watch.tasks.fusion.predict import predict
    from watch.utils.util_yaml import Yaml

    input_path = config.input_path
    input_region_path = config.input_region_path
    output_path = config.output_path
    outbucket = config.outbucket
    aws_profile = config.aws_profile
    dryrun = config.dryrun
    previous_bas_outbucket = config.previous_bas_outbucket
    ta2_s3_collation_bucket = config.ta2_s3_collation_bucket

    if aws_profile is not None:
        aws_base_command = ['aws', 's3', '--profile', aws_profile, 'cp']
    else:
        aws_base_command = ['aws', 's3', 'cp']

    if dryrun:
        aws_base_command.append('--dryrun')

    # 1. Ingress data
    print("* Running baseline framework kwcoco ingress *")
    ingress_dir = ub.Path('/tmp/ingress')
    ingress_kwcoco_path = baseline_framework_kwcoco_ingress(
        input_path,
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

        ub.cmd([*aws_base_command, '--recursive',
                previous_bas_outbucket, previous_ingress_dir],
               check=True, verbose=3)

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
        # use boundary_region here
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
<<<<<<< HEAD
    ]
    ub.cmd(crop_cmd, check=True, verbose=3)
=======
    ], check=True, verbose=3)
>>>>>>> d6e50df4 (Patch int error)

    # 6. (Optional) collate TA-2 output
    if ta2_s3_collation_bucket is not None:
        print("* Collating TA-2 output")
        _ta2_collate_output(aws_base_command,
                            cropped_region_models_outdir,
                            cropped_site_models_outdir,
                            ta2_s3_collation_bucket)

    # 6. Egress (envelop KWCOCO dataset in a STAC item and egress;
    #    will need to recursive copy the kwcoco output directory up to
    #    S3 bucket)
    print("* Egressing KWCOCO dataset and associated STAC item *")
    baseline_framework_kwcoco_egress(bas_fusion_kwcoco_path,
                                     local_region_path,
                                     output_path,
                                     outbucket,
                                     aws_profile=None,
                                     dryrun=False,
                                     newline=False)


if __name__ == "__main__":
    main()
