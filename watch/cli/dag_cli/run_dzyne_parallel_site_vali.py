#!/usr/bin/env python3
import json
import os
import scriptconfig as scfg
import subprocess
import ubelt as ub
from glob import glob
import pathlib


class DzyneParallelSiteValiConfig(scfg.DataConfig):
    """
    Run DZYNE's parallel site validation framework component

    python ~/code/watch/watch/cli/dag_cli/run_dzyne_parallel_site_vali.py
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

    depth_model_fpath = scfg.Value("/models/depthPCD/basicModel2.h5", type=str, position=4, required=True, help='path to depth model weights')

    aws_profile = scfg.Value(None, type=str, help=ub.paragraph(
            '''
            AWS Profile to use for AWS S3 CLI commands
            '''))

    dryrun = scfg.Value(False, isflag=True, short_alias=['d'], help='Run AWS CLI commands with --dryrun flag')

    outbucket = scfg.Value(None, type=str, required=True, short_alias=['o'], help=ub.paragraph(
            '''
            S3 Output directory for STAC item / asset egress
            '''))


def main():
    config = DzyneParallelSiteValiConfig.cli(strict=True)
    print('config = {}'.format(ub.urepr(dict(config), nl=1, align=':')))
    run_dzyne_parallel_site_vali_for_baseline(config)


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


def run_dzyne_parallel_site_vali_for_baseline(config):
    from watch.cli.baseline_framework_kwcoco_ingress import baseline_framework_kwcoco_ingress
    from watch.cli.baseline_framework_kwcoco_egress import baseline_framework_kwcoco_egress
    # from watch.cli.concat_kwcoco_videos import concat_kwcoco_datasets
    from watch.utils.util_framework import download_region, determine_region_id
    # from watch.tasks.fusion.predict import predict
    # from watch.utils.util_yaml import Yaml

    input_path = config.input_path
    input_region_path = config.input_region_path
    output_path = config.output_path
    outbucket = config.outbucket
    aws_profile = config.aws_profile
    dryrun = config.dryrun

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
    print(f'ingress_kwcoco_path={ingress_kwcoco_path}')

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

    # 3. Run the Site Validation Filter
    print("* Running the Site Validation Filter *")
    from watch.tasks import depthPCD

    sv_dir = pathlib.Path(ingress_dir) / "dyzne_parallel_site_vali"
    sv_dir.mkdir(exists_ok=True)
    site_vali_kwcoco_path = sv_dir / "filtered_poly.kwcoco.zip"

    depthPCD.score_tracks(
        in_file=sv_dir / "poly.kwcoco.zip",
        images_kwcoco=ingress_dir / region_id / "subdata.kwcoco.json",
        model_fpath=config.depth_model_fpath,
        out_site_summaries_fpath=sv_dir / "filtered_site_summaries_manifest.json",
        out_site_summaries_dir=sv_dir / "filtered_site_summaries",
        out_sites_fpath=sv_dir / "filtered_sites_manifest.json",
        out_sites_dir=sv_dir / "filtered_site",
        out_kwcoco=site_vali_kwcoco_path,
        threshold=0.4,
    )

    # 4. Egress (envelop KWCOCO dataset in a STAC item and egress;
    #    will need to recursive copy the kwcoco output directory up to
    #    S3 bucket)
    print("* Egressing KWCOCO dataset and associated STAC item *")
    baseline_framework_kwcoco_egress(site_vali_kwcoco_path,
                                     local_region_path,
                                     output_path,
                                     outbucket,
                                     aws_profile=None,
                                     dryrun=False,
                                     newline=False)


if __name__ == "__main__":
    main()
