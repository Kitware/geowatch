#!/usr/bin/env python3
import subprocess
from watch.cli.smartflow_ingress import smartflow_ingress
from watch.cli.smartflow_egress import smartflow_egress
from watch.utils.util_framework import download_region
import ubelt as ub
import scriptconfig as scfg
from watch.mlops.smart_pipeline import SV_Cropping
from kwutil.util_yaml import Yaml


class SVDatasetConfig(scfg.DataConfig):
    """
    Generate cropped KWCOCO dataset for SC
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
            AWS Profile to use for AWS S3 CLI commands
            '''))
    dryrun = scfg.Value(False, isflag=True, short_alias=['d'], help='Run AWS CLI commands with --dryrun flag')
    outbucket = scfg.Value(None, type=str, required=True, short_alias=['o'], help=ub.paragraph(
            '''
            S3 Output directory for STAC item / asset egress
            '''))
    newline = scfg.Value(False, isflag=True, short_alias=['n'], help=ub.paragraph(
            '''
            Output as simple newline separated STAC items
            '''))
    jobs = scfg.Value(1, type=int, short_alias=['j'], help='Number of jobs to run in parallel')
    dont_recompute = scfg.Value(False, isflag=True, help=ub.paragraph(
            '''
            Will not recompute if output_path already exists
            '''))
    sv_cropping_config = scfg.Value(None, type=str, help=ub.paragraph(
            '''
            Raw json/yaml or a path to a json/yaml file that specifies the
            config for SV_Cropping.
            '''))


def main():
    config = SVDatasetConfig.cli(strict=True)
    print('config = {}'.format(ub.urepr(dict(config), nl=1, align=':')))
    run_generate_sv_cropped_kwcoco(**config)


def run_generate_sv_cropped_kwcoco(input_path,
                                   input_region_path,
                                   output_path,
                                   outbucket,
                                   aws_profile=None,
                                   dryrun=False,
                                   newline=False,
                                   jobs=1,
                                   dont_recompute=False,
                                   sv_cropping_config=None):

    from watch.utils.util_framework import AWS_S3_Command
    from watch.utils import util_framework
    if dont_recompute:
        aws_ls = AWS_S3_Command('ls', profile=aws_profile)
        aws_ls_command = aws_ls.finalize()
        try:
            ub.cmd([*aws_ls_command, output_path], check=True, verbose=3, capture=False)
        except subprocess.CalledProcessError:
            # Continue processing
            pass
        else:
            # If output_path file was there, nothing to do
            return

    # 1. Ingress data
    print("* Running baseline framework kwcoco ingress *")
    ingress_dir = ub.Path('/tmp/ingress')
    ingressed_assets = smartflow_ingress(
        input_path,
        ['kwcoco_for_sc',
         'cropped_region_models_bas',
         'cropped_site_models_bas'],
        ingress_dir,
        aws_profile,
        dryrun)

    # 2. Download and prune region file
    print("* Downloading and pruning region file *")
    local_region_path = '/tmp/region.json'
    local_region_path = download_region(input_region_path,
                                        local_region_path,
                                        aws_profile=aws_profile,
                                        strip_nonregions=True)

    # Parse region_id from original region file
    region_id = util_framework.determine_region_id(local_region_path)

    if region_id is None:
        raise RuntimeError("Couldn't parse 'region_id' from input region file")

    # Paths to inputs generated in previous pipeline steps
    bas_region_path = ub.Path(ingressed_assets['cropped_region_models_bas']) / f'{region_id}.geojson'
    ta1_sc_kwcoco_path = ingressed_assets['kwcoco_for_sc']

    # 4. Crop ingress KWCOCO dataset to region for SV
    print("* Cropping KWCOCO dataset to region for SV*")
    ta1_sv_cropped_kwcoco_path = ingress_dir / 'cropped_kwcoco_for_sv.json'

    sv_cropping_config = Yaml.coerce(sv_cropping_config or {})

    sv_cropping = SV_Cropping(root_dpath=ingress_dir)
    sv_cropping.configure({
        'crop_src_fpath': ta1_sc_kwcoco_path,
        'regions': bas_region_path,
        'crop_dst_fpath': ta1_sv_cropped_kwcoco_path})

    ub.cmd(sv_cropping.command(), check=True, verbose=3, system=True)

    # 5. Egress (envelop KWCOCO dataset in a STAC item and egress;
    #    will need to recursive copy the kwcoco output directory up to
    #    S3 bucket)
    print("* Egressing KWCOCO dataset and associated STAC item *")
    ingressed_assets['cropped_kwcoco_for_sv'] = ta1_sv_cropped_kwcoco_path
    ingressed_assets['cropped_kwcoco_for_sv_assets'] = ingress_dir / f'{region_id}'
    smartflow_egress(ingressed_assets,
                     local_region_path,
                     output_path,
                     outbucket,
                     aws_profile=None,
                     dryrun=False,
                     newline=False)


if __name__ == "__main__":
    main()
