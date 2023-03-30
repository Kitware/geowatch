#!/usr/bin/env python3
# PYTHON_ARGCOMPLETE_OK
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
import tempfile
import ubelt as ub

from glob import glob
from urllib.parse import urlparse


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

    bas_fusion_model_path = scfg.Value(None, type=str, required=True, help='File path to BAS fusion model')

    aws_profile = scfg.Value(None, type=str, help=ub.paragraph(
            '''
            AWS Profile to use for AWS S3 CLI commands
            '''))

    dryrun = scfg.Value(False, isflag=True, short_alias=['d'], help='Run AWS CLI commands with --dryrun flag')

    outbucket = scfg.Value(None, type=str, required=True, short_alias=['o'], help=ub.paragraph(
            '''
            S3 Output directory for STAC item / asset egress
            '''))

    bas_thresh = scfg.Value(0.1, type=float, help=ub.paragraph(
            '''
            Threshold for BAS tracking (kwarg 'thresh')
            '''))

    previous_bas_outbucket = scfg.Value(None, type=str, help=ub.paragraph(
            '''
            S3 Output directory for previous interval BAS fusion output
            '''))

    time_combine = scfg.Value(False, isflag=True, help=ub.paragraph(
            '''
            Quick and dirty hack to run time combine before fusion
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
    """
    Ignore:
        import sys, ubelt
        sys.path.append(ubelt.expandpath('~/code/watch'))
        from watch.cli.dag_cli.run_bas_fusion import *  # NOQA
        argv = [
            'script',
            '--bas_fusion_model_path', '/root/data/smart_expt_dvc/models/fusion/Drop6-MeanYear10GSD/packages/Drop6_TCombo1Year_BAS_10GSD_split6_V42_cont2/Drop6_TCombo1Year_BAS_10GSD_split6_V42_cont2_epoch3_step941.pt',
            '--outbucket', 's3://smartflow-023300502152-us-west-2/smartflow/env/kitware-prod-v4/work/ta2_preeval10_pyenv_v01/batch/kit/KR_R001/2021-08-31/split/mono/products/bas-fusion',
            '--input_path', 's3://smartflow-023300502152-us-west-2/smartflow/env/kitware-prod-v4/work/ta2_preeval10_pyenv_v01/batch/kit/KR_R001/2021-08-31/split/mono/products/kwcoco-dataset/items.jsonl', '--input_region_path',
            's3://smartflow-023300502152-us-west-2/smartflow/env/kitware-prod-v4/work/ta2_preeval10_pyenv_v01/batch/kit/KR_R001/2021-08-31/input/mono/region_models/KR_R001.geojson', '--output_path', 's3://smartflow-023300502152-us-west-2/smartflow/env/kitware-prod-v4/work/ta2_preeval10_pyenv_v01/batch/kit/KR_R001/2021-08-31/split/mono/products/bas-fusion/items.jsonl',
            '--time_combine', 'True', '--bas_pxl_config',
            'num_workers: 24\nchip_overlap: 0.3\nchip_dims: auto\ndrop_unused_frames: true\ntime_sampling: auto\ntime_span: auto,\ntime_kernel: auto\nbatch_size: 1', '--bas_poly_config', 'inner_window_size: null\ninner_agg_fn: mean\nnorm_ord: 1\nagg_fn: probs\nresolution: 10GSD\nmoving_window_size: null\nmin_area_square_meters: 7200\nmax_area_square_meters: 9000000', '--bas_thresh', '0.33'
            ]
        config = BasFusionConfig.cli(argv=argv, strict=True)
        print('config = {}'.format(ub.urepr(dict(config), nl=1, align=':')))
    """
    config = BasFusionConfig.cli(strict=True)
    import sys
    print(f'sys.argv={sys.argv}')
    print('config = {}'.format(ub.urepr(dict(config), nl=1, align=':')))
    run_bas_fusion_for_baseline(config)


def _download_region(aws_base_command,
                     input_region_path,
                     output_region_path,
                     strip_nonregions=False,
                     replace_originator=True):
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

    if replace_originator:
        for feature in out_region_data.get('features', ()):
            if feature['properties']['type'] == 'region':
                feature['properties']['originator'] = 'kit'

    region_id = None
    for feature in out_region_data.get('features', ()):
        props = feature['properties']
        if props['type'] == 'region':
            # Ensure the region feature has a "comments" field
            props['comments'] = props.get('comments', '')
            region_id = props.get('region_model_id', props.get('region_id'))
            break

    with open(output_region_path, 'w') as f:
        print(json.dumps(out_region_data, indent=2), file=f)

    return output_region_path, region_id


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
    from watch.tasks.fusion.predict import predict
    from watch.utils.util_yaml import Yaml

    input_path = config.input_path
    input_region_path = config.input_region_path
    output_path = config.output_path
    bas_fusion_model_path = config.bas_fusion_model_path
    outbucket = config.outbucket
    aws_profile = config.aws_profile
    dryrun = config.dryrun
    bas_thresh = config.bas_thresh
    previous_bas_outbucket = config.previous_bas_outbucket

    if aws_profile is not None:
        aws_base_command = ['aws', 's3', '--profile', aws_profile, 'cp']
    else:
        aws_base_command = ['aws', 's3', 'cp']

    if dryrun:
        aws_base_command.append('--dryrun')

    # 1. Ingress data
    print("* Running baseline framework kwcoco ingress *")
    ingress_dir = '/tmp/ingress'
    ingress_kwcoco_path = baseline_framework_kwcoco_ingress(
        input_path,
        ingress_dir,
        aws_profile,
        dryrun)

    # # 2. Download and prune region file
    print("* Downloading and pruning region file *")
    local_region_path = '/tmp/region.json'
    local_region_path, region_id = _download_region(aws_base_command,
                                                    input_region_path,
                                                    local_region_path,
                                                    strip_nonregions=True,
                                                    replace_originator=True)

    if config.time_combine:
        from watch.cli import coco_time_combine
        preproc_kwcoco_fpath = ub.Path(ingress_kwcoco_path).augment(
            stemsuffix='_timecombined', ext='.kwcoco.zip', multidot=True)
        coco_time_combine.main(
            cmdline=0,
            input_kwcoco_fpath=ingress_kwcoco_path,
            output_kwcoco_fpath=preproc_kwcoco_fpath,
            time_window='1y',
            resolution='10GSD',
            workers='avail',
        )
        predict_input_fpath = os.fspath(preproc_kwcoco_fpath)
    else:
        predict_input_fpath = ingress_kwcoco_path

    # 3. Run fusion
    print("* Running BAS fusion *")
    bas_fusion_kwcoco_path = os.path.join(
        ingress_dir, 'bas_fusion_kwcoco.json')

    default_predict_config = ub.udict({
          "chip_overlap": 0.3,
          "chip_dims": "auto",
          "time_span": "auto",
          "time_sampling": "auto",
          "drop_unused_frames": True,
          "batch_size": 1,
          "num_workers": 2,
    })
    bas_pxl_config = default_predict_config | Yaml.coerce(config.bas_pxl_config or {})

    predict(devices='0,',
            write_preds=False,
            write_probs=True,
            with_change=False,
            with_saliency=True,
            with_class=False,
            test_dataset=predict_input_fpath,
            pred_dataset=bas_fusion_kwcoco_path,
            package_fpath=bas_fusion_model_path,
            **bas_pxl_config)

    # 3.1. If a previous interval was run; concatenate BAS fusion
    # output KWCOCO files for tracking
    if previous_bas_outbucket is not None:
        combined_bas_fusion_kwcoco_path = os.path.join(
                ingress_dir, 'combined_bas_fusion_kwcoco.json')

        previous_ingress_dir = '/tmp/ingress_previous'
        subprocess.run([*aws_base_command, '--recursive',
                        previous_bas_outbucket, previous_ingress_dir],
                       check=True)

        previous_bas_fusion_kwcoco_path = os.path.join(
            previous_ingress_dir, 'combined_bas_fusion_kwcoco.json')

        # On first interval nothing will be copied down so need to
        # check that we have the input explicitly
        if os.path.isfile(previous_bas_fusion_kwcoco_path):
            concat_kwcoco_datasets(
                (previous_bas_fusion_kwcoco_path, bas_fusion_kwcoco_path),
                combined_bas_fusion_kwcoco_path)
            # Copy saliency assets from previous bas fusion
            shutil.copy_tree(
                os.path.join(previous_ingress_dir, '_assets', 'pred_saliency'),
                os.path.join(ingress_dir, '_assets', 'pred_saliency'))

            # Copy original assets from previous bas rusion
            shutil.copy_tree(
                os.path.join(previous_ingress_dir, region_id),
                os.path.join(ingress_dir, region_id))
        else:
            # Copy current bas_fusion_kwcoco_path to combined path as
            # this is the first interval
            shutil.copy(bas_fusion_kwcoco_path,
                        combined_bas_fusion_kwcoco_path)
    else:
        combined_bas_fusion_kwcoco_path = bas_fusion_kwcoco_path

    # 4. Compute tracks (BAS)
    print("* Computing tracks (BAS) *")
    region_models_outdir = os.path.join(ingress_dir, 'region_models')
    os.makedirs(region_models_outdir, exist_ok=True)

    region_models_manifest_outdir = os.path.join(
        ingress_dir, 'tracking_manifests_bas')
    os.makedirs(region_models_manifest_outdir, exist_ok=True)
    region_models_manifest_outpath = os.path.join(
        region_models_manifest_outdir, 'region_models_manifest.json')
    # Copy input region model into region_models outdir to be updated
    # (rather than generated from tracking, which may not have the
    # same bounds as the original)
    shutil.copy(local_region_path, os.path.join(
        region_models_outdir, '{}.geojson'.format(region_id)))

    default_bas_tracking_config = ub.udict({
        "thresh": bas_thresh,
        "moving_window_size": None,
        "polygon_simplify_tolerance": 1,
        "max_area_behavior": 'ignore'
    })
    bas_tracking_config = default_bas_tracking_config | Yaml.coerce(config.bas_poly_config or {})

    tracked_bas_kwcoco_path = '_tracked'.join(
        os.path.splitext(bas_fusion_kwcoco_path))
    subprocess.run(['python', '-m', 'watch.cli.run_tracker',
                    combined_bas_fusion_kwcoco_path,
                    '--out_site_summaries_dir', region_models_outdir,
                    '--out_site_summaries_fpath',
                    region_models_manifest_outpath,
                    '--out_kwcoco', tracked_bas_kwcoco_path,
                    '--default_track_fn', 'saliency_heatmaps',
                    '--append_mode', 'True',
                    '--track_kwargs', json.dumps(bas_tracking_config)],
                   check=True)

    cropped_region_models_outdir = os.path.join(ingress_dir,
                                                'cropped_region_models_bas')
    subprocess.run(['python', '-m', 'watch.cli.crop_sites_to_regions',
                    '--region_models',
                    os.path.join(region_models_outdir, '*.geojson'),
                    '--new_site_dpath', cropped_region_models_outdir,
                    '--new_region_dpath', cropped_region_models_outdir],
                   check=True)

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
