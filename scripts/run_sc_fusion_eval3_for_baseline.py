import argparse
import sys
from urllib.parse import urlparse
import os
import subprocess
import tempfile
import json
from glob import glob

from watch.cli.baseline_framework_kwcoco_egress import baseline_framework_kwcoco_egress  # noqa: 501
from watch.cli.baseline_framework_kwcoco_ingress import baseline_framework_kwcoco_ingress  # noqa: 501


def main():
    parser = argparse.ArgumentParser(
        description="Run TA-2 SC fusion as "
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
    parser.add_argument("--sc_fusion_model_path",
                        required=True,
                        type=str,
                        help="File path to SC fusion model")
    parser.add_argument("--sc_track_fn",
                        required=False,
                        default='class_heatmaps',  # noqa: 501
                        type=str,
                        help="Tracking function to use for generating sites")
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
                        default=0.2,
                        type=float,
                        required=False,
                        help="Threshold for BAS tracking (kwarg 'thresh')")
    parser.add_argument("--sc_thresh",
                        default=0.01,
                        type=float,
                        required=False,
                        help="Threshold for SC tracking (kwarg 'thresh')")
    parser.add_argument("--ta2_s3_collation_bucket",
                        type=str,
                        required=False,
                        default=None,
                        help="S3 Location for collated TA-2 output (bucket "
                             "name should include up to eval name)")

    run_sc_fusion_for_baseline(**vars(parser.parse_args()))

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


def run_sc_fusion_for_baseline(
        input_path,
        input_region_path,
        output_path,
        sc_fusion_model_path,
        outbucket,
        sc_track_fn='class_heatmaps',  # noqa: E501
        aws_profile=None,
        dryrun=False,
        newline=False,
        jobs=1,
        force_zero_num_workers=False,
        ta2_s3_collation_bucket=None,
        bas_thresh=0.2,
        sc_thresh=0.01):
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
    print("* Running SC fusion *")
    sc_fusion_kwcoco_path = os.path.join(
        ingress_dir, 'sc_fusion_kwcoco.json')

    subprocess.run(['python', '-m', 'watch.tasks.fusion.predict',
                    '--gpus', '0',
                    '--write_preds', 'False',
                    '--write_probs', 'True',
                    '--with_change', 'False',
                    '--with_saliency', 'False',
                    '--with_class', 'True',
                    '--test_dataset', ingress_kwcoco_path,
                    '--package_fpath', sc_fusion_model_path,
                    '--pred_dataset', sc_fusion_kwcoco_path,
                    '--num_workers', '0' if force_zero_num_workers else str(jobs),  # noqa: 501
                    '--batch_size', '8',
                    '--tta_time', '1',
                    '--tta_fliprot', '0',
                    '--chip_overlap', '0.3'], check=True)

    # 4. Compute tracks (SC)
    print("* Computing tracks (SC) *")
    site_models_outdir = os.path.join(ingress_dir, 'site_models')

    region_models_outdir = os.path.join(ingress_dir, 'region_models')

    sc_track_kwargs = {"boundaries_as": "polys"}
    subprocess.run(['python', '-m', 'watch.cli.kwcoco_to_geojson',
                    sc_fusion_kwcoco_path,
                    '--out_dir', site_models_outdir,
                    '--default_track_fn', sc_track_fn,
                    '--site_summary',
                    os.path.join(region_models_outdir, '*.geojson'),
                    '--track_kwargs', json.dumps(sc_track_kwargs)],
                   check=True)

    cropped_region_models_outdir = os.path.join(ingress_dir,
                                                'cropped_region_models')
    cropped_site_models_outdir = os.path.join(ingress_dir,
                                              'cropped_site_models')

    subprocess.run(['python', '-m', 'watch.cli.crop_sites_to_regions',
                    '--site_models',
                    os.path.join(site_models_outdir, '*.geojson'),
                    '--region_models',
                    os.path.join(region_models_outdir, '*.geojson'),
                    '--new_site_dpath', cropped_site_models_outdir,
                    '--new_region_dpath', cropped_region_models_outdir],
                   check=True)

    # 5. Update region model with computed sites
    # ** NOTE ** This is a destructive operation as the region file
    # ** gets modified in place (locally or on S3)
    # Referencing region model already computed during BAS fusion
    print("* Updating region *")
    _upload_region(aws_base_command,
                   cropped_region_models_outdir,
                   local_region_path,
                   input_region_path)

    # 6. Egress (envelop KWCOCO dataset in a STAC item and egress;
    #    will need to recursive copy the kwcoco output directory up to
    #    S3 bucket)
    print("* Egressing KWCOCO dataset and associated STAC item *")
    baseline_framework_kwcoco_egress(sc_fusion_kwcoco_path,
                                     local_region_path,
                                     output_path,
                                     outbucket,
                                     aws_profile=None,
                                     dryrun=False,
                                     newline=False)

    # 7. (Optional) collate TA-2 output
    if ta2_s3_collation_bucket is not None:
        print("* Collating TA-2 output")
        _ta2_collate_output(aws_base_command,
                            cropped_region_models_outdir,
                            cropped_site_models_outdir,
                            ta2_s3_collation_bucket)


if __name__ == "__main__":
    sys.exit(main())