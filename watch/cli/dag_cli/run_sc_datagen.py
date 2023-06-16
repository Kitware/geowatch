#!/usr/bin/env python3
"""
See Old Script:
    ~/code/watch/scripts/run_generate_sc_cropped_kwcoco.py
"""
import os
import subprocess
import json

from watch.cli.baseline_framework_kwcoco_ingress import baseline_framework_kwcoco_ingress  # noqa: 501
from watch.cli.baseline_framework_kwcoco_egress import baseline_framework_kwcoco_egress  # noqa: 501
from watch.utils.util_framework import download_region
from watch.utils.util_framework import determine_region_id
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
    sc_align_config = scfg.Value(None, help=ub.paragraph(
        '''
        The configuration for the coco-align step
        '''))


def main():
    config = SCDatasetConfig.cli(strict=True)
    print('config = {}'.format(ub.urepr(dict(config), nl=1, align=':')))
    run_generate_sc_cropped_kwcoco(**config, config=config)


def run_generate_sc_cropped_kwcoco(input_path,
                                   input_region_path,
                                   output_path,
                                   outbucket,
                                   aws_profile=None,
                                   dryrun=False,
                                   newline=False,
                                   jobs=1,
                                   dont_recompute=False,
                                   sc_align_config=None,
                                   config=None):

    from watch.utils.util_framework import AWS_S3_Command
    if dont_recompute:
        aws_ls = AWS_S3_Command('ls', profile=profile)
        aws_ls_command = aws_ls.finalize()

        try:
            ub.cmd([*aws_ls_command, output_path], check=True, verbose=3, capture=False)
        except subprocess.CalledProcessError:
            # Continue processing
            pass
        else:
            # If output_path file was there, nothing to do
            return

    ingress_dir = ub.Path('/tmp/ingress')
    local_region_path = ub.Path('/tmp/region.json')

    # 1. Ingress data
    print("* Running baseline framework kwcoco ingress *")
    _ = baseline_framework_kwcoco_ingress(
        input_path,
        ingress_dir,
        aws_profile,
        dryrun)

    # 2. Download and prune region file
    print("* Downloading and pruning region file *")
    local_region_path = download_region(input_region_path,
                                        local_region_path,
                                        aws_profile=aws_profile,
                                        strip_nonregions=True)

    # Parse region_id from original region file
    region_id = determine_region_id(local_region_path)
    print(f'region_id={region_id}')


    if region_id is None:
        raise RuntimeError("Couldn't parse 'region_id' from input region file")

    # Paths to inputs generated in previous pipeline steps
    input_region_path = ingress_dir / 'sv_out_region_models' / f'{region_id}.geojson'
    if not os.path.isfile(input_region_path):
        print("* Didn't find region output from SV; using region output "
              "from BAS *")
        input_region_path = ingress_dir / 'cropped_region_models_bas' / f'{region_id}.geojson'

    ta1_sc_kwcoco_path = ingress_dir / 'kwcoco_for_sc.json'

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
        img_workers: {str(jobs}}
        aux_workers: 2
        rpc_align_method: affine_warp
        image_timeout: 20minutes
        asset_timeout: 10minutes
        verbose: 1
        ''')))
    align_config = align_config_default | Yaml.coerce(config.sc_align_config)
    if align_config.aux_workers == 'auto':
        align_config.aux_workers = align_config.include_channels.count('|') + 1
    target_gsd = align_config['target_gsd']

    # 4. Crop ingress KWCOCO dataset to region for SC
    print("* Cropping KWCOCO dataset to region for SC*")
    ta1_sc_cropped_kwcoco_path = ingress_dir / 'cropped_kwcoco_for_sc.json'

    align_config['src'] = ta1_sc_kwcoco_path
    align_config['dst'] = ta1_sc_cropped_kwcoco_path
    align_config['regions'] = input_region_path

    EXEC_MODE = 'cmd'
    # Not sure if one is more stable than the other
    if EXEC_MODE == 'import':
        from watch.cli import coco_align
        coco_align.main(cmdline=False, **align_config)
    elif EXEC_MODE == 'cmd':
        align_arglist = _make_arglist(align_config)
        ub.cmd(['python', '-m', 'watch.cli.coco_align'] + align_arglist,
               check=True, capture=False, verbose=3)
    else:
        raise KeyError(EXEC_MODE)
    # # Crops to BAS generated site_summaries
    # ub.cmd(['python', '-m', 'watch.cli.coco_align',
    #         '--visualize', 'False',
    #         '--src', ta1_sc_kwcoco_path,
    #         '--dst', ta1_sc_cropped_kwcoco_path,
    #         '--regions', input_region_path,
    #         '--force_nodata', '-9999',
    #         '--include_channels', include_channels,
    #         '--site_summary', 'True',
    #         '--geo_preprop', 'auto',
    #         '--keep', 'none',
    #         '--convexify_regions', 'True',
    #         '--target_gsd', '4',  # TODO: Expose as cli parameter
    #         '--context_factor', '1.5',  # TODO: Expose as cli parameter
    #         '--workers', '1' if force_one_job_for_cropping else str(jobs),  # noqa: 501
    #         '--aux_workers', '2',  # str(include_channels.count('|') + 1),  # noqa: 501
    #         '--rpc_align_method', 'affine_warp',  # Maybe needs to change to "orthorectified"  # noqa
    #         '--force_min_gsd', '8',
    #         '--verbose', '4',
    #         '--image_timeout', '20minutes',
    #         '--asset_timeout', '10minutes',
    #        ], check=True, verbose=3, capture=False)

    # 5. Egress (envelop KWCOCO dataset in a STAC item and egress;
    #    will need to recursive copy the kwcoco output directory up to
    #    S3 bucket)
    print("* Egressing KWCOCO dataset and associated STAC item *")
    baseline_framework_kwcoco_egress(ta1_sc_cropped_kwcoco_path,
                                     local_region_path,
                                     output_path,
                                     outbucket,
                                     aws_profile=None,
                                     dryrun=False,
                                     newline=False)


if __name__ == "__main__":
    main()
