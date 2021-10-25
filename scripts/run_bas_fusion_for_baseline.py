import argparse
import sys
from urllib.parse import urlparse
import os
import subprocess
import tempfile
import json

from watch.cli.baseline_framework_kwcoco_egress import baseline_framework_kwcoco_egress  # noqa: 501
from watch.cli.baseline_framework_kwcoco_ingress import baseline_framework_kwcoco_ingress  # noqa: 501


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
    parser.add_argument("--track_fn",
                        required=False,
                        default='watch.tasks.tracking.from_heatmap.time_aggregated_polys',  # noqa: 501
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

    run_bas_fusion_for_baseline(**vars(parser.parse_args()))

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


def run_bas_fusion_for_baseline(
        input_path,
        input_region_path,
        output_path,
        bas_fusion_model_path,
        outbucket,
        track_fn='watch.tasks.tracking.from_heatmap.time_aggregated_polys',
        aws_profile=None,
        dryrun=False,
        newline=False,
        jobs=1):
    # 1. Ingress data
    print("* Running baseline framework kwcoco ingress *")
    ingress_dir = '/tmp/ingress'
    ingress_kwcoco_path = baseline_framework_kwcoco_ingress(
        input_path,
        ingress_dir,
        aws_profile,
        dryrun)

    # 2. Download and prune region file
    print("* Downloading and pruning region file *")
    local_region_path = '/tmp/region.json'
    local_region_path = _download_region(input_region_path,
                                         local_region_path,
                                         aws_profile=aws_profile,
                                         dryrun=dryrun,
                                         strip_nonregions=True)

    # 3. Run fusion
    print("* Running fusion *")
    bas_fusion_kwcoco_path = os.path.join(
        ingress_dir, 'bas_fusion_kwcoco.json')

    subprocess.run(['python', '-m', 'watch.tasks.fusion.predict',
                    '--write_preds', 'False',
                    '--write_probs', 'True',
                    '--write_change', 'False',
                    '--write_saliency', 'True',
                    '--write_class', 'False',
                    '--test_dataset', ingress_kwcoco_path,
                    '--package_fpath', bas_fusion_model_path,
                    '--pred_dataset', bas_fusion_kwcoco_path,
                    '--num_workers', str(jobs),
                    '--batch_size', '8',
                    '--gpus', '0'], check=True)

    # 4. Compute tracks
    print("* Computing tracks *")
    subprocess.run(['python', '-m', 'watch.cli.kwcoco_to_geojson',
                    '--in_file', bas_fusion_kwcoco_path,
                    '--out_dir', ingress_dir,
                    '--track_fn', track_fn],  # noqa: 501
                   check=True)

    # 5. Update region model with computed sites
    # TODO

    # 6. Egress (envelop KWCOCO dataset in a STAC item and egress;
    #    will need to recursive copy the kwcoco output directory up to
    #    S3 bucket)
    print("* Egressing KWCOCO dataset and associated STAC item *")
    baseline_framework_kwcoco_egress(bas_fusion_kwcoco_path,
                                     output_path,
                                     outbucket,
                                     local_region_path,
                                     aws_profile=None,
                                     dryrun=False,
                                     newline=False)


if __name__ == "__main__":
    sys.exit(main())
