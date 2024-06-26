#!/usr/bin/env python
import sys
import subprocess
import json
import traceback
import requests
import ubelt as ub
import scriptconfig as scfg


class UploadRGDConfig(scfg.DataConfig):
    """
    Run TA-2 BAS fusion as baseline framework component
    """
    input_site_models_s3 = scfg.Value(None, type=str, position=1, help='Path to S3 directory of site models')

    rgd_aws_region = scfg.Value(None, type=str, help='AWS region where RGD instance is running')

    rgd_deployment_name = scfg.Value(None, type=str, help=ub.paragraph(
        '''
        Name of RGD deployment (e.g. 'resonantgeodatablue'
        '''))

    aws_profile = scfg.Value(None, type=str, help=ub.paragraph(
        '''
        AWS Profile to use for AWS S3 CLI commands
        '''))

    title = scfg.Value(None, type=str, help='Title of the model run')

    region_id = scfg.Value(None, type=str, help='Region ID (e.g. "KR_R002")')

    performer_shortcode = scfg.Value('KIT', type=str, help='Performer shortcode (e.g. "KIT")')

    rgd_endpoint_override = scfg.Value(None, type=str, help=ub.paragraph(
        '''
        Use this RGD URL instead of looking up via aws tools
        '''))

    jobs = scfg.Value(8, type=int, short_alias=['j'], help='Number of jobs to run in parallel')

    expiration_time = scfg.Value(None, type=int, short_alias=['x'], help=ub.paragraph(
        '''
        Number of days to keep system run output in RGD
        '''))


def main(cmdline=1, **kwargs):
    config = UploadRGDConfig.cli(cmdline=cmdline, data=kwargs)
    print('config = {}'.format(ub.urepr(dict(config), nl=1, align=':')))
    assert config.rgd_aws_region is not None
    assert config.input_site_models_s3 is not None
    assert config.rgd_deployment_name is not None
    assert config.title is not None
    assert config.region_id is not None
    upload_to_rgd(config)


def get_model_results(model_run_results_url):
    model_runs_result = requests.get(model_run_results_url, params={
        'limit': '0'})
    request_json = model_runs_result.json()
    request_results = request_json.get('results', ())
    return request_results


def upload_to_rgd(config):

    input_site_models_s3 = config.input_site_models_s3
    rgd_aws_region = config.rgd_aws_region
    rgd_deployment_name = config.rgd_deployment_name
    title = config.title
    region_id = config.region_id
    aws_profile = config.aws_profile
    performer_shortcode = config.performer_shortcode
    jobs = config.jobs
    rgd_endpoint_override = config.rgd_endpoint_override
    expiration_time = config.expiration_time

    from geowatch.utils.util_framework import NodeStateDebugger
    node_state = NodeStateDebugger()
    node_state.print_environment()
    node_state.print_local_invocation(config)

    # Ensure performer_shortcode is uppercase
    performer_shortcode = performer_shortcode.upper()

    if aws_profile is not None:
        aws_base_command = ['aws', 's3', '--profile', aws_profile, 'cp']
    else:
        aws_base_command = ['aws', 's3', 'cp']

    local_site_models_dir = '/tmp/site_models'
    ub.cmd([*aws_base_command, '--recursive',
            input_site_models_s3, local_site_models_dir],
           check=True, verbose=3)

    if rgd_endpoint_override is None:
        try:
            endpoint_result = ub.cmd(
                ['aws',
                 *(('--profile', aws_profile)
                   if aws_profile is not None else ()),
                 '--region', rgd_aws_region,
                 'elbv2', 'describe-load-balancers',
                 '--region', rgd_aws_region,
                 '--names', "{}-internal-alb".format(rgd_deployment_name)],
                check=True,
                verbose=3)
        except subprocess.CalledProcessError as e:
            print(e.stderr, file=sys.stderr)
            raise e

        rgd_instance_details = json.loads(endpoint_result.stdout)
        rgd_endpoint = rgd_instance_details['LoadBalancers'][0]['DNSName']
    else:
        rgd_endpoint = rgd_endpoint_override

    # Check that our run doesn't already exist
    model_run_results_url = f"http://{rgd_endpoint}/api/model-runs/"

    # from retry.api import retry_call
    from geowatch.utils.util_retry import retry_call
    from geowatch.utils import util_framework
    logger = util_framework.PrintLogger()
    print('\nAttempt to post to RGD model results endpoint')
    print(f'model_run_results_url = {ub.urepr(model_run_results_url, nl=1)}')

    request_results = retry_call(
        get_model_results, fargs=[model_run_results_url], tries=3,
        exceptions=(Exception,), delay=3, logger=logger)

    print('Got Results')
    existing_model_run = None

    for model_run in request_results:
        if (model_run['title'] != title or
             model_run['performer']['short_code'] != performer_shortcode):
            continue

        model_region = model_run.get('region')
        if model_region is None:
            continue

        if (isinstance(model_region, dict) and
             model_run['region'].get('name') == region_id):
            existing_model_run = model_run
            break
        elif (isinstance(model_region, str) and
               model_region == region_id):  # noqa
            existing_model_run = model_run
            break

    if existing_model_run is not None:
        model_run_id = model_run['id']
    else:
        post_model_url = f"http://{rgd_endpoint}/api/model-runs/"
        post_model_data = {"performer": performer_shortcode,
                           "title": title,
                           "region": {"name": region_id},
                           "parameters": {}}

        if expiration_time is not None:
            post_model_data['expiration_time'] = expiration_time

        post_model_result = requests.post(
            post_model_url,
            json=post_model_data,
            headers={"Content-Type": "application/json"})

        model_run_id = post_model_result.json()['id']

    post_site_url = (
        f"http://{rgd_endpoint}/api/model-runs/{model_run_id}/site-model/")

    local_site_models_dir = ub.Path(local_site_models_dir)

    jobs = ub.JobPool(mode='process', max_workers=jobs)
    for site_filepath in local_site_models_dir.glob('*.geojson'):
        jobs.submit(post_site, post_site_url, site_filepath)

    for site_post_job in ub.ProgIter(jobs.as_completed(),
                                     total=len(jobs),
                                     desc='Uploading sites..'):
        try:
            result = site_post_job.result()
        except Exception:
            print("Exception occurred (printed below)")
            traceback.print_exception(*sys.exc_info())
            continue
        else:
            if result.status_code != 201:
                print(f"Error uploading site, status "
                      f"code: [{result.status_code}]")
                print(result.text)


def post_site(post_site_url, site_filepath):
    with open(site_filepath, 'r') as f:
        response = requests.post(
            post_site_url,
            json=json.load(f),
            headers={"Content-Type": "application/json"})

    return response


if __name__ == "__main__":
    main()
