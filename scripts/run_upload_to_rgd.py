import argparse
import sys
import subprocess
import json
import traceback
from concurrent.futures import as_completed
from glob import glob
import os

import requests
import ubelt as ub


def main():
    parser = argparse.ArgumentParser(
        description="Run TA-2 BAS fusion as "
                    "baseline framework component")

    parser.add_argument('input_site_models_s3',
                        type=str,
                        help="Path to S3 directory of site models")
    parser.add_argument("--rgd_aws_region",
                        required=True,
                        type=str,
                        help="AWS region where RGD instance is running")
    parser.add_argument("--rgd_deployment_name",
                        required=True,
                        type=str,
                        help="Name of RGD deployment "
                             "(e.g. 'resonantgeodatablue'")
    parser.add_argument("--aws_profile",
                        required=False,
                        type=str,
                        help="AWS Profile to use for AWS S3 CLI commands")
    parser.add_argument("--title",
                        required=True,
                        type=str,
                        help="Title of the model run")
    parser.add_argument("--performer_shortcode",
                        required=True,
                        type=str,
                        default='KIT',
                        help='Performer shortcode (e.g. "KIT")')
    parser.add_argument("--rgd_endpoint_override",
                        type=str,
                        help="Use this RGD URL instead of looking up "
                             "via aws tools")
    parser.add_argument("-j", "--jobs",
                        type=int,
                        default=8,
                        required=False,
                        help="Number of jobs to run in parallel")

    upload_to_rgd(**vars(parser.parse_args()))


def upload_to_rgd(input_site_models_s3,
                  rgd_aws_region,
                  rgd_deployment_name,
                  title,
                  aws_profile=None,
                  performer_shortcode='KIT',
                  jobs=8,
                  rgd_endpoint_override=None):
    # Ensure performer_shortcode is uppercase
    performer_shortcode = performer_shortcode.upper()

    if aws_profile is not None:
        aws_base_command =\
            ['aws', 's3', '--profile', aws_profile, 'cp']
    else:
        aws_base_command = ['aws', 's3', 'cp']

    local_site_models_dir = '/tmp/site_models'
    subprocess.run([*aws_base_command, '--recursive',
                    input_site_models_s3, local_site_models_dir],
                   check=True)

    if rgd_endpoint_override is None:
        try:
            endpoint_result = subprocess.run(
                ['aws',
                 *(('--profile', aws_profile)
                   if aws_profile is not None else ()),
                 '--region', rgd_aws_region,
                 'elbv2', 'describe-load-balancers',
                 '--region', rgd_aws_region,
                 '--names', "{}-internal-alb".format(rgd_deployment_name)],
                check=True,
                capture_output=True)
        except subprocess.CalledProcessError as e:
            print(e.stderr, file=sys.stderr)
            raise e

        rgd_instance_details = json.loads(endpoint_result.stdout)
        rgd_endpoint = rgd_instance_details['LoadBalancers'][0]['DNSName']
    else:
        rgd_endpoint = rgd_endpoint_override

    # Check that our run doesn't already exist
    model_run_results_url = f"https://{rgd_endpoint}/api/model-runs"
    model_runs_result = requests.get(model_run_results_url,
                                     params={'limit', '0'})

    existing_model_run = None
    for model_run in model_runs_result.json():
        if(model_run['title'] == title and
           model_run['performer'] == performer_shortcode):
            existing_model_run = model_run
            break

    if existing_model_run is not None:
        model_run_id = model_run['id']
    else:
        post_model_url = f"https://{rgd_endpoint}/api/model-runs"
        post_model_data = {"performer": performer_shortcode,
                           "title": title,
                           "parameters": {}}
        post_model_result = requests.post(post_model_url,
                                          data=json.dumps(post_model_data))

        model_run_id = post_model_result.json()['id']

    post_site_url =\
        f"https://{rgd_endpoint}/api/model-runs/{model_run_id}/site-model"

    executor = ub.Executor(mode='process' if jobs > 1 else 'serial',
                           max_workers=jobs)
    site_post_jobs = [executor.submit(post_site, post_site_url, site_filepath)
                      for site_filepath in glob(os.path.join(
                          local_site_models_dir, '*.geojson'))]

    # TODO: Only upload sites that aren't already uploaded (by
    # checking site_id)?
    for site_post_job in ub.ProgIter(as_completed(site_post_jobs),
                                     total=len(site_post_jobs),
                                     desc='Uploading sites..'):
        try:
            result = site_post_job.result()
        except Exception:
            print("Exception occurred (printed below)")
            traceback.print_exception(*sys.exc_info())
            continue
        else:
            if result.status_code != 200:
                print(f"Error uploading site, status "
                      f"code: [{result.status_code}]")


def post_site(post_site_url, site_filepath):
    return requests.post(post_site_url,
                         files={'file': (site_filepath,
                                         open(site_filepath, 'rb'),
                                         'application/json')})


if __name__ == "__main__":
    sys.exit(main())
