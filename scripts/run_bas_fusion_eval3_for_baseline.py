import argparse
import sys
from urllib.parse import urlparse
import os
import subprocess
import tempfile
import json
from glob import glob
import shutil

from watch.cli.baseline_framework_kwcoco_egress import baseline_framework_kwcoco_egress  # noqa: 501
from watch.cli.baseline_framework_kwcoco_ingress import baseline_framework_kwcoco_ingress  # noqa: 501
from watch.tasks.fusion.predict import predict
from watch.cli.concat_kwcoco_videos import concat_kwcoco_datasets


def main():
    parser = argparse.ArgumentParser(
        description="Run TA-2 BAS fusion as "
                    "baseline framework component")

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
    parser.add_argument("--bas_fusion_model_path",
                        required=True,
                        type=str,
                        help="File path to BAS fusion model")
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
    parser.add_argument("--force_zero_num_workers",
                        action='store_true',
                        default=False,
                        help="Force predict scripts to use --num_workers=0")
    parser.add_argument("--bas_thresh",
                        default=0.1,
                        type=float,
                        required=False,
                        help="Threshold for BAS tracking (kwarg 'thresh')")
    parser.add_argument("--previous_bas_outbucket",
                        type=str,
                        required=False,
                        help="S3 Output directory for previous interval BAS "
                             "fusion output")

    run_bas_fusion_for_baseline(**vars(parser.parse_args()))

    return 0


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


def run_bas_fusion_for_baseline(
        input_path,
        input_region_path,
        output_path,
        bas_fusion_model_path,
        outbucket,
        aws_profile=None,
        dryrun=False,
        newline=False,
        jobs=1,
        force_zero_num_workers=False,
        bas_thresh=0.1,
        previous_bas_outbucket=None):
    if aws_profile is not None:
        aws_base_command =\
            ['aws', 's3', '--profile', aws_profile, 'cp']
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

    # 3. Run fusion
    print("* Running BAS fusion *")
    bas_fusion_kwcoco_path = os.path.join(
        ingress_dir, 'bas_fusion_kwcoco.json')

    predict_config = json.loads("""
{
      "tta_fliprot": 0,
      "tta_time": 0,
      "chip_overlap": 0.3,
      "input_space_scale": "15GSD",
      "window_space_scale": "15GSD",
      "output_space_scale": "15GSD",
      "time_span": "6m",
      "time_sampling": "auto",
      "time_steps": 5,
      "chip_dims": [
         128,
         128
      ],
      "set_cover_algo": null,
      "resample_invalid_frames": true,
      "use_cloudmask": 1
}
    """)

    predict(devices='0,',
            write_preds=False,
            write_probs=True,
            with_change=False,
            with_saliency=True,
            with_class=False,
            test_dataset=ingress_kwcoco_path,
            package_fpath=bas_fusion_model_path,
            pred_dataset=bas_fusion_kwcoco_path,
            num_workers=('0' if force_zero_num_workers else str(jobs)),  # noqa: 501
            batch_size=1,
            **predict_config)

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
    # Copy input region model into region_models outdir to be updated
    # (rather than generated from tracking, which may not have the
    # same bounds as the original)
    shutil.copy(local_region_path, os.path.join(
        region_models_outdir, '{}.geojson'.format(region_id)))

    bas_tracking_config = {
        "thresh": bas_thresh,
        "polygon_fn": "heatmaps_to_polys",
        "agg_fn": "probs"}

    tracked_bas_kwcoco_path = '_tracked'.join(
        os.path.splitext(bas_fusion_kwcoco_path))
    subprocess.run(['python', '-m', 'watch.cli.kwcoco_to_geojson',
                    combined_bas_fusion_kwcoco_path,
                    '--out_site_summaries_dir', region_models_outdir,
                    '--out_kwcoco', tracked_bas_kwcoco_path,
                    '--default_track_fn', 'saliency_heatmaps',
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
    sys.exit(main())
