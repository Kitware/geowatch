#!/usr/bin/env python3
import os
from geowatch.cli.smartflow_ingress import smartflow_ingress
from geowatch.cli.smartflow_egress import smartflow_egress
from geowatch.utils.util_framework import download_region
import ubelt as ub
import scriptconfig as scfg
from geowatch.mlops import smart_pipeline
from kwutil.util_yaml import Yaml


_debug_notes_ = r"""

docker run \
    --runtime=nvidia \
    --volume "$HOME/temp/debug_smartflow_v2/ingress":/tmp/ingress \
    --volume $HOME/.aws:/root/.aws:ro \
    --volume "$HOME/code":/extern_code:ro \
    --volume "$HOME/data":/extern_data:ro \
    --volume "$HOME"/.cache/pip:/pip_cache \
    --env AWS_PROFILE=iarpa \
    -it registry.smartgitlab.com/kitware/watch:0.12.1-98697830f-strict-pyenv3.11.2-20231112T181927-0500-from-de082898 bash

ipython

from geowatch.cli.smartflow.run_sv_datagen import *  # NOQA
config = {
    'input_path'        : 's3://smartflow-023300502152-us-west-2/smartflow/env/kw-v3-0-0/work/preeval17_batch_v134/batch/kit/KR_R001/2021-08-31/split/mono/products/bas-fusion/items.jsonl',
    'input_region_path' : 's3://smartflow-023300502152-us-west-2/smartflow/env/kw-v3-0-0/work/preeval17_batch_v134/batch/kit/KR_R001/2021-08-31/input/mono/region_models/KR_R001.geojson',
    'output_path'       : 's3://smartflow-023300502152-us-west-2/smartflow/env/kw-v3-0-0/work/preeval17_batch_v134/batch/kit/KR_R001/2021-08-31/split/mono/products/site-cropped-kwcoco-for-sv/items.jsonl',
    'aws_profile'       : None,
    'dryrun'            : False,
    'outbucket'         : 's3://smartflow-023300502152-us-west-2/smartflow/env/kw-v3-0-0/work/preeval17_batch_v134/batch/kit/KR_R001/2021-08-31/split/mono/products/site-cropped-kwcoco-for-sv',
    'newline'           : True,
    'jobs'              : 16,
    'dont_recompute'    : False,
    'sv_cropping_config': 'context_factor: 1.6\nforce_min_gsd: 1GSD\nminimum_size: 256x256@3GSD\nnum_end_frames: 3.0\nnum_start_frames: 3.0\ntarget_gsd: 2GSD',
}
config = SVDatasetConfig(**config)

"""


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
    jobs = scfg.Value(1, type=int, short_alias=['j'], help='UNUSED AND WILL BE REMOVED')
    dont_recompute = scfg.Value(False, isflag=True, help=ub.paragraph(
            '''
            Will not recompute if output_path already exists
            '''))
    sv_cropping_config = scfg.Value(None, type=str, help=ub.paragraph(
            '''
            Raw json/yaml or a path to a json/yaml file that specifies the
            config for SV_Cropping.
            '''))

    input_region_models_asset_name = scfg.Value('cropped_region_models_bas', type=str, required=False, help=ub.paragraph(
            '''
            Which region model assets to use as input
            '''))

    input_site_models_asset_name = scfg.Value('cropped_site_models_bas', type=str, required=False, help=ub.paragraph(
            '''
            Which site model assets to to use as input
            '''))


def main():
    config = SVDatasetConfig.cli(strict=True)
    print('config = {}'.format(ub.urepr(dict(config), nl=1, align=':')))
    run_generate_sv_cropped_kwcoco(config)


def run_generate_sv_cropped_kwcoco(config):
    from geowatch.utils import util_framework
    from geowatch.utils import util_fsspec

    input_path = config.input_path
    input_region_path = config.input_region_path
    output_path = config.output_path
    outbucket = config.outbucket
    aws_profile = config.aws_profile
    dryrun = config.dryrun

    # newline = config.newline

    dont_recompute = config.dont_recompute
    sv_cropping_config = config.sv_cropping_config

    if aws_profile is not None:
        # This should be sufficient, but it is not tested.
        util_fsspec.S3Path._new_fs(profile=aws_profile)

    ####
    # DEBUGGING:
    # Print info about what version of the code we are running on
    from geowatch.utils.util_framework import NodeStateDebugger
    node_state = NodeStateDebugger()
    node_state.print_environment()

    if dont_recompute:
        output_path = util_fsspec.FSPath.coerce(output_path)
        if output_path.exists():
            # If output_path file was there, nothing to do
            return

    # 1. Ingress data
    print("* Running baseline framework kwcoco ingress *")
    ingress_dir = ub.Path('/tmp/ingress')
    ingressed_assets = smartflow_ingress(
        input_path,
        ['kwcoco_for_sc',
         config.input_region_models_asset_name,
         config.input_site_models_asset_name,
         ],
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
    bas_region_path = ub.Path(ingressed_assets[config.input_region_models_asset_name]) / f'{region_id}.geojson'
    ta1_sc_kwcoco_path = ingressed_assets['kwcoco_for_sc']

    node_state.print_current_state(ingress_dir)

    # 4. Crop ingress KWCOCO dataset to region for SV
    print("* Cropping KWCOCO dataset to region for SV*")
    ta1_sv_cropped_kwcoco_path = ingress_dir / 'cropped_kwcoco_for_sv.json'

    sv_cropping_config = Yaml.coerce(sv_cropping_config or {})

    sv_cropping = smart_pipeline.SV_Cropping(root_dpath=ingress_dir)
    sv_cropping.configure({
        'crop_src_fpath': ta1_sc_kwcoco_path,
        'regions': bas_region_path,
        'crop_dst_fpath': ta1_sv_cropped_kwcoco_path})

    ub.cmd(sv_cropping.command(), check=True, verbose=3, system=True)

    # 5. Egress (envelop KWCOCO dataset in a STAC item and egress;
    #    will need to recursive copy the kwcoco output directory up to
    #    S3 bucket)

    assets_dpath = ingress_dir / f'{region_id}'
    assets_dpath.ensuredir()

    # Ensure at least one file exists in the directory we want to upload
    # so aws doesn't break.
    (assets_dpath / '_empty').touch()

    print("* Egressing KWCOCO dataset and associated STAC item *")
    ingressed_assets['cropped_kwcoco_for_sv'] = ta1_sv_cropped_kwcoco_path
    ingressed_assets['cropped_kwcoco_for_sv_assets'] = ingress_dir / f'{region_id}'
    # Ensure that the assets directory at least exists before copying
    # (for incremental mode it's possible there are no assets for the
    # given interval; egress will fail if a source directory / file
    # doesn't exist)
    if not os.path.isdir(ingressed_assets['cropped_kwcoco_for_sv_assets']):
        os.makedirs(ingressed_assets['cropped_kwcoco_for_sv_assets'], exist_ok=True)
        ub.Path(ingressed_assets['cropped_kwcoco_for_sv_assets'] / '.empty').touch()

    node_state.print_current_state(ingress_dir)

    smartflow_egress(ingressed_assets,
                     local_region_path,
                     output_path,
                     outbucket,
                     aws_profile=None,
                     dryrun=False,
                     newline=False)


if __name__ == "__main__":
    main()
