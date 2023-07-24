#!/usr/bin/env python3
"""
See Old Script:
    ~/code/watch/scripts/run_generate_sc_cropped_kwcoco.py
"""
import subprocess
from watch.cli.smartflow_ingress import smartflow_ingress
from watch.cli.smartflow_egress import smartflow_egress
from watch.utils.util_framework import download_region
import ubelt as ub
import scriptconfig as scfg


class SCDatasetConfig(scfg.DataConfig):
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
    dryrun = scfg.Value(False, isflag=True, short_alias=['d'], help='DEPRECATD. DO NOT USE')
    outbucket = scfg.Value(None, type=str, required=True, short_alias=['o'], help=ub.paragraph(
            '''
            S3 Output directory for STAC item / asset egress
            '''))
    newline = scfg.Value(False, isflag=True, short_alias=['n'], help=ub.paragraph(
            '''
            Output as simple newline separated STAC items

            UNUSED.
            '''))
    jobs = scfg.Value(1, type=int, short_alias=['j'], help='Number of jobs to run in parallel')
    dont_recompute = scfg.Value(False, isflag=True, help=ub.paragraph(
            '''
            Will not recompute if output_path already exists
            '''))
    sc_align_config = scfg.Value(None, help=ub.paragraph(
        '''
        The configuration for the coco-align step
        '''))


def main():
    config = SCDatasetConfig.cli(strict=True)
    print('config = {}'.format(ub.urepr(config, nl=1, align=':')))
    run_generate_sc_cropped_kwcoco(config)


def run_generate_sc_cropped_kwcoco(config):
    from kwutil.util_yaml import Yaml
    from watch.utils import util_framework
    from watch.cli import coco_align
    from watch.utils import util_fsspec

    if config.aws_profile is not None:
        # This should be sufficient, but it is not tested.
        util_fsspec.S3Path._new_fs(profile=config.aws_profile)

    if config.dont_recompute:
        output_path = util_fsspec.FSPath.coerce(config.output_path)
        if output_path.exists():
            # If output_path file was there, nothing to do
            return

    ingress_dir = ub.Path('/tmp/ingress')
    local_region_path = ub.Path('/tmp/region.json')

    # 1. Ingress data
    print("* Running baseline framework kwcoco ingress *")
    ingressed_assets = smartflow_ingress(
        input_path=config.input_path,
        assets=['kwcoco_for_sc',
                'sv_out_region_models',
                'cropped_region_models_bas'],
        outdir=ingress_dir,
        aws_profile=config.aws_profile,
        dryrun=config.dryrun,
        dont_error_on_missing_asset=True)

    # 2. Download and prune region file
    print("* Downloading and pruning region file *")
    local_region_path = download_region(config.input_region_path,
                                        local_region_path,
                                        aws_profile=config.aws_profile,
                                        strip_nonregions=True)

    # Parse region_id from original region file
    region_id = util_framework.determine_region_id(local_region_path)
    print(f'region_id={region_id}')

    if region_id is None:
        raise RuntimeError("Couldn't parse 'region_id' from input region file")

    # Paths to inputs generated in previous pipeline steps
    if 'sv_out_region_models' in ingressed_assets:
        input_region_path = ub.Path(ingressed_assets['sv_out_region_models']) / f'{region_id}.geojson'
    else:
        print("* Didn't find region output from SV; using region output "
              "from BAS *")
        input_region_path = ub.Path(ingressed_assets['cropped_region_models_bas']) / f'{region_id}.geojson'

    ta1_sc_kwcoco_path = ingressed_assets['kwcoco_for_sc']

    align_config_default = ub.udict(Yaml.coerce(ub.codeblock(
        f'''
        force_nodata: -9999
        include_channels: "red|green|blue|quality"
        site_summary: True
        geo_preprop: auto
        keep: null
        convexify_regions: True
        target_gsd: 2
        context_factor: 1.5
        force_min_gsd: 4
        img_workers: {str(config.jobs)}
        aux_workers: 2
        rpc_align_method: affine_warp
        image_timeout: 20minutes
        asset_timeout: 10minutes
        verbose: 1
        ''')))
    align_config = align_config_default | Yaml.coerce(config.sc_align_config)
    if align_config['aux_workers'] == 'auto':
        align_config['aux_workers'] = align_config['include_channels'].count('|') + 1

    # 4. Crop ingress KWCOCO dataset to region for SC
    print("* Cropping KWCOCO dataset to region for SC*")
    ta1_sc_cropped_kwcoco_path = ingress_dir / 'cropped_kwcoco_for_sc.json'

    align_config['src'] = ta1_sc_kwcoco_path
    align_config['dst'] = ta1_sc_cropped_kwcoco_path
    align_config['regions'] = input_region_path
    # Validate align config before running anything
    align_config = coco_align.CocoAlignGeotiffConfig(**align_config)
    print('align_config = {}'.format(ub.urepr(align_config, nl=1)))

    EXEC_MODE = 'cmd'
    # Not sure if one is more stable than the other
    if EXEC_MODE == 'import':
        coco_align.main(cmdline=False, **align_config)
    elif EXEC_MODE == 'cmd':
        align_arglist = util_framework._make_arglist(align_config)
        ub.cmd(['python', '-m', 'watch.cli.coco_align'] + align_arglist,
               check=True, capture=False, verbose=3)
    else:
        raise KeyError(EXEC_MODE)

    # 5. Egress (envelop KWCOCO dataset in a STAC item and egress;
    #    will need to recursive copy the kwcoco output directory up to
    #    S3 bucket)
    print("* Egressing KWCOCO dataset and associated STAC item *")
    ingressed_assets['cropped_kwcoco_for_sc'] = ta1_sc_cropped_kwcoco_path
    ingressed_assets['cropped_kwcoco_for_sc_assets'] = ingress_dir / f'{region_id}'
    smartflow_egress(ingressed_assets,
                     local_region_path,
                     config.output_path,
                     config.outbucket,
                     aws_profile=config.aws_profile,
                     dryrun=config.dryrun,
                     newline=False)


if __name__ == "__main__":
    main()
