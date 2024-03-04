#!/usr/bin/env python3
"""
See Old Version:
    ../../../scripts/run_sc_fusion_eval3_for_baseline.py

SeeAlso:
    ~/code/watch-smartflow-dags/KIT_TA2_PREEVAL10_PYENV.py
"""
import sys
import traceback
import shutil

import scriptconfig as scfg
import ubelt as ub
from geowatch.mlops.smart_pipeline import DinoBoxDetector, SV_DinoFilter


class DinoSVConfig(scfg.DataConfig):
    """
    Run TA-2 SC fusion as baseline framework component
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
    aws_profile = scfg.Value(None, help=ub.paragraph(
            '''
            AWS Profile to use for AWS S3 CLI commands
            '''))
    dryrun = scfg.Value(False, isflag=True, short_alias=['d'], help='Run AWS CLI commands with --dryrun flag')
    outbucket = scfg.Value(None, type=str, required=True, short_alias=['o'], help=ub.paragraph(
            '''
            S3 Output directory for STAC item / asset egress
            '''))
    dino_detect_config = scfg.Value(None, type=str, help=ub.paragraph(
            '''
            Raw json/yaml or a path to a json/yaml file that specifies the
            config for DinoBoxDetector.
            '''))
    dino_filter_config = scfg.Value(None, type=str, help=ub.paragraph(
            '''
            Raw json/yaml or a path to a json/yaml file that specifies the
            config for SV_DinoFilter.
            '''))
    skip_on_fail = scfg.Value(False, help=ub.paragraph(
            '''
            If an error occurs, pass through input region / sites unchanged.
            '''))

    input_region_models_asset_name = scfg.Value('depth_filtered_regions', type=str, required=False, help=ub.paragraph(
            '''
            Which region model assets to use as input
            '''))

    input_site_models_asset_name = scfg.Value('depth_filtered_sites', type=str, required=False, help=ub.paragraph(
            '''
            Which site model assets to to use as input
            '''))


def main():
    config = DinoSVConfig.cli(strict=True)
    print('config = {}'.format(ub.urepr(dict(config), nl=1, align=':')))
    run_dino_sv(config)


def run_dino_sv(config):
    from geowatch.cli.smartflow_ingress import smartflow_ingress
    from geowatch.cli.smartflow_egress import smartflow_egress
    from geowatch.utils.util_framework import download_region, determine_region_id
    from kwutil.util_yaml import Yaml
    from geowatch.utils import util_framework

    input_path = config.input_path
    input_region_path = config.input_region_path
    output_path = config.output_path

    outbucket = config.outbucket
    aws_profile = config.aws_profile
    dryrun = config.dryrun

    ####
    # DEBUGGING:
    # Print info about what version of the code we are running on
    from geowatch.utils.util_framework import NodeStateDebugger
    node_state = NodeStateDebugger()
    node_state.print_environment()

    # 1. Ingress data
    print("* Running baseline framework kwcoco ingress *")
    ingress_dir = ub.Path('/tmp/ingress')
    ingressed_assets = smartflow_ingress(
        input_path=input_path,
        assets=[
            'cropped_kwcoco_for_sv',
            'cropped_kwcoco_for_sv_assets',
            config.input_region_models_asset_name,
            config.input_site_models_asset_name,
        ],
        outdir=ingress_dir,
        aws_profile=aws_profile,
        dryrun=dryrun
    )

    # # 2. Download and prune region file
    print("* Downloading and pruning region file *")
    local_region_path = '/tmp/region.json'

    download_region(
        input_region_path=input_region_path,
        output_region_path=local_region_path,
        aws_profile=aws_profile,
        strip_nonregions=True,
    )

    # Determine the region_id in the region file.
    region_id = determine_region_id(local_region_path)
    print(f'region_id={region_id}')

    dino_boxes_kwcoco_path = ingress_dir / 'dino_boxes_kwcoco.json'

    input_region_dpath = ub.Path(ingressed_assets[config.input_region_models_asset_name])
    input_sites_dpath = ub.Path(ingressed_assets[config.input_site_models_asset_name])

    input_region_fpath = ub.Path(input_region_dpath) / f'{region_id}.geojson'
    assert input_region_fpath.exists()

    # NOTE; we want to be using the output of SV crop, not necesarilly the the
    # dzyne output referenced by ingress_kwcoco_path
    # input_kwcoco_fpath = ingress_kwcoco_path
    input_kwcoco_fpath = ingressed_assets['cropped_kwcoco_for_sv']

    # FIXME: these output directories for region / site models should be passed
    # to us from the DAG
    output_sites_dpath = ingress_dir / 'sv_out_site_models'
    output_region_dpath = ingress_dir / 'sv_out_region_models'
    output_site_manifest_dpath = ingress_dir / 'tracking_manifests_sv'
    output_region_fpath = output_region_dpath / f'{region_id}.geojson'
    output_site_manifest_fpath = output_site_manifest_dpath / 'site_models_manifest.json'

    node_state.print_current_state(ingress_dir)

    # 3.1. Check that we have at least one "video" (BAS identified
    # site) to run over; if not skip SV fusion and KWCOCO to GeoJSON
    import kwcoco
    input_coco_dset = kwcoco.CocoDataset(input_kwcoco_fpath)
    print('input_coco_dset = {}'.format(ub.urepr(input_coco_dset, nl=1)))
    num_videos = input_coco_dset.n_videos
    # Note: cant open with json here because kwcoco will save compressed files
    # with open(input_kwcoco_fpath) as f:
    #     ingress_kwcoco_data = json.load(f)
    # num_videos = len(ingress_kwcoco_data.get('videos', ()))
    print(f'num_videos={num_videos}')
    output_region_dpath.ensuredir()

    if num_videos == 0:
        # Copy input region model into region_models outdir to be updated
        # (rather than generated from tracking, which may not have the
        # same bounds as the original)

        # Not sure if the above case is the right comment, but leaving this
        # here to guarentee the region with site summaries is passed forward
        # TODO: the dino code should just be robust to this.
        input_sites_dpath.copy(output_sites_dpath)
        input_region_fpath.copy(output_region_fpath)
    else:
        output_site_manifest_dpath.ensuredir()
        output_sites_dpath.ensuredir()
        # 3.2 Run DinoBoxDetector
        print("* Running Dino Detect *")

        default_dino_detect_config = ub.udict({
            'coco_fpath': input_kwcoco_fpath,
            'package_fpath': None,
            'batch_size': 1,
            'device': 0})
        dino_detect_config = (default_dino_detect_config
                              | Yaml.coerce(config.dino_detect_config or {}))

        if dino_detect_config.get('package_fpath', None) is None:
            raise ValueError('Requires package_fpath')

        dino_box_detector = DinoBoxDetector(root_dpath='/tmp/ingress')
        dino_box_detector.configure({
            'out_coco_fpath': dino_boxes_kwcoco_path,
            **dino_detect_config})

        try:
            ub.cmd(dino_box_detector.command(), check=True, verbose=3, system=True)

            # 3.3 Run SV_DinoFilter
            print("* Running Dino Building Filter *")

            default_dino_filter_config = ub.udict({})
            dino_filter_config = (default_dino_filter_config
                                  | Yaml.coerce(config.dino_filter_config or {}))

            dino_building_filter = SV_DinoFilter(root_dpath='/tmp/ingress')

            dino_building_filter.configure({
                'input_kwcoco': dino_boxes_kwcoco_path,
                'input_region': input_region_fpath,
                'input_sites': input_sites_dpath,
                'output_region_fpath': output_region_fpath,
                'output_sites_dpath': output_sites_dpath,
                'output_site_manifest_fpath': output_site_manifest_fpath,
                **dino_filter_config,
            })

            ub.cmd(dino_building_filter.command(), check=True, verbose=3, system=True)
        except Exception:
            if config.skip_on_fail:
                print("WARNING: Exception occurred (printed below), passing input sites / region models as output")
                traceback.print_exception(*sys.exc_info())

                shutil.copytree(input_sites_dpath, output_sites_dpath, dirs_exist_ok=True)
                shutil.copytree(input_region_dpath, output_region_dpath, dirs_exist_ok=True)
            else:
                raise
        else:
            # Validate and fix all outputs
            util_framework.fixup_and_validate_site_and_region_models(
                region_dpath=output_region_fpath.parent,
                site_dpath=output_sites_dpath,
            )

    node_state.print_current_state(ingress_dir)

    # 5. Egress (envelop KWCOCO dataset in a STAC item and egress;
    #    will need to recursive copy the kwcoco output directory up to
    #    S3 bucket)
    print("* Egressing KWCOCO dataset and associated STAC item *")
    ingressed_assets['sv_out_site_models'] = output_sites_dpath
    ingressed_assets['sv_out_region_models'] = output_region_dpath
    if dino_boxes_kwcoco_path.exists():
        # Reroot kwcoco files to make downloaded results easier to work with
        ub.cmd(['kwcoco', 'reroot', f'--src={dino_boxes_kwcoco_path}', '--inplace=1', '--absolute=0'])
        ingressed_assets['sv_dino_boxes_kwcoco'] = dino_boxes_kwcoco_path

    smartflow_egress(ingressed_assets,
                     local_region_path,
                     output_path,
                     outbucket,
                     aws_profile=None,
                     dryrun=False,
                     newline=False)


if __name__ == "__main__":
    main()
