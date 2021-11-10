import argparse
import sys
from urllib.parse import urlparse
import subprocess
import tempfile
import json


def main():
    parser = argparse.ArgumentParser(
        description="Run TA-2 SC as baseline framework component")

    parser.add_argument('input_path',
                        type=str,
                        help="Path to input T&E Baseline Framework JSON")
    parser.add_argument('input_site_id',
                        type=str,
                        help="ID of the input Site")
    parser.add_argument('input_region_path',
                        type=str,
                        help="Path to input T&E Baseline Framework Region "
                             "definition JSON")
    parser.add_argument('output_path',
                        type=str,
                        help="S3 path for output JSON")
    parser.add_argument("--bas_s3_root",
                        required=True,
                        type=str,
                        help="Root directory for BAS results")
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
    parser.add_argument("--force_v2",
                        action='store_true',
                        default=False,
                        help="Force site model version 2")

    run_sc_for_baseline(**vars(parser.parse_args()))

    return 0


def _download_region(input_region_path,
                     output_region_path,
                     aws_profile=None,
                     dryrun=False,
                     strip_nonregions=False):
    if aws_profile is not None:
        aws_base_command =\
            ['aws', 's3', '--profile', aws_profile, 'cp']
    else:
        aws_base_command = ['aws', 's3', 'cp']

    if dryrun:
        aws_base_command.append('--dryrun')

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

    with open(output_region_path, 'w') as f:
        print(json.dumps(out_region_data, indent=2), file=f)

    return output_region_path


def run_sc_for_baseline(
        input_path,
        input_site_id,
        input_region_path,
        output_path,
        outbucket,
        bas_s3_root,
        aws_profile=None,
        dryrun=False,
        newline=False,
        jobs=1,
        force_v2=False):
    if aws_profile is not None:
        aws_base_command =\
            ['aws', 's3', '--profile', aws_profile, 'cp']
    else:
        aws_base_command = ['aws', 's3', 'cp']

    if dryrun:
        aws_base_command.append('--dryrun')

    local_region_path = _download_region(input_region_path, '/tmp/region.json')
    with open(local_region_path) as f:
        region = json.load(f)

    region_feature = None
    for feature in region.get('features', []):
        if feature['properties']['type'] == 'region':
            region_feature = feature
            break

    if region_feature is None:
        raise RuntimeError("Couldn't find feature of type 'region' in input "
                           "region file, aborting!")

    region_id = region_feature['properties'].get(
        'region_id', region_feature['properties'].get('region_model_id'))

    if region_id is None:
        raise RuntimeError("Couldn't parse 'region_id' (or 'region_model_id') "
                           "from region feature, aborting!")

    # s3://kitware-smart-watch-data/nifi-test/ta2_bas/bas_fusion/BR_Rio_R01_23KPQ/site_models/BR_Rio_R01_23KPQ_0241.geojson
    bas_site_model_path = '/'.join([bas_s3_root,
                                    'bas_fusion',
                                    region_id,
                                    'site_models',
                                    '{}.geojson'.format(input_site_id)])

    site_model_outpath = '/'.join([outbucket,
                                   '{}.geojson'.format(input_site_id)])

    print('bas_site_model_path: {}'.format(bas_site_model_path))
    print('site_model_outpath: {}'.format(site_model_outpath))

    subprocess.run([*aws_base_command,
                    bas_site_model_path, site_model_outpath], check=True)


if __name__ == "__main__":
    sys.exit(main())
