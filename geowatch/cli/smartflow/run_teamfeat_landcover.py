#!/usr/bin/env python3
"""
TODO: rectify with run_teamfeat_acsc_landcover.py
"""
import ubelt as ub
import scriptconfig as scfg
from geowatch.cli.smartflow_ingress import smartflow_ingress
from geowatch.cli.smartflow_egress import smartflow_egress


class TeamFeatLandcoverConfig(scfg.DataConfig):
    """
    Run DZYNE landcover feature computation as baseline framework component
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
    model_path = scfg.Value(None, type=str, required=True, help='File path to landcover model')
    aws_profile = scfg.Value(None, type=str, help=ub.paragraph(
            '''
            AWS Profile to use for AWS S3 CLI commands
            '''))
    dryrun = scfg.Value(False, isflag=True, short_alias=['d'], help='Run AWS CLI commands with --dryrun flag')
    outbucket = scfg.Value(None, type=str, required=True, short_alias=['o'], help=ub.paragraph(
            '''
            S3 Output directory for STAC item / asset egress
            '''))
    requester_pays = scfg.Value(False, isflag=True, short_alias=['r'], help=ub.paragraph(
            '''
            Run AWS CLI commands with `--requestor_payer requester` flag
            '''))
    newline = scfg.Value(False, isflag=True, short_alias=['n'], help=ub.paragraph(
            '''
            Output as simple newline separated STAC items
            '''))


def main():
    config = TeamFeatLandcoverConfig.cli(strict=True)
    print('config = {}'.format(ub.urepr(dict(config), nl=1, align=':')))
    run_landcover_for_baseline(config)


def run_landcover_for_baseline(config):
    from geowatch.utils.util_framework import download_region

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
        config.input_path,
        ['enriched_bas_kwcoco_file',
         'enriched_bas_kwcoco_teamfeats',
         'enriched_bas_kwcoco_rawbands'],
        ingress_dir,
        config.aws_profile,
        config.dryrun)

    # 2. Download and prune region file
    print("* Downloading and pruning region file *")
    local_region_path = '/tmp/region.json'
    download_region(
        input_region_path=config.input_region_path,
        output_region_path=local_region_path,
        aws_profile=config.aws_profile,
        strip_nonregions=True,
    )

    # 2. Generate Landcover features
    print("* Generating landcover features *")
    dzyne_landcover_features_kwcoco_path = ingress_dir / 'dzyne_landcover_kwcoco.json'

    enriched_bas_kwcoco_file = ingressed_assets['enriched_bas_kwcoco_file']
    ub.cmd([
        'python', '-m', 'geowatch.tasks.landcover.predict',
        '--dataset', enriched_bas_kwcoco_file,
        '--deployed', config.model_path,
        '--output', dzyne_landcover_features_kwcoco_path,
        '--num_workers', '2',
        '--with_hidden', '32',
        '--assets_dname', '_teamfeats',
        '--select_images', '.sensor_coarse == "S2"',
        '--device', '0'
    ], check=True, verbose=3, capture=False)

    # 3. Combining landcover features with input features to pass to BAS
    print("* Combining input features with computed landcover features *")
    combo_features_kwcoco_path = ingress_dir / 'features_combo_with_landcover_kwcoco.json'
    ub.cmd([
        'python', '-m', 'geowatch.cli.coco_combine_features',
        '--src', enriched_bas_kwcoco_file, dzyne_landcover_features_kwcoco_path,
        '--dst', combo_features_kwcoco_path,
    ], check=True, verbose=3, capture=False)

    # Reroot kwcoco files to make downloaded results easier to work with
    ub.cmd(['kwcoco', 'reroot', f'--src={combo_features_kwcoco_path}', '--inplace=1', '--absolute=0'])

    # 4. Egress (envelop KWCOCO dataset in a STAC item and egress;
    #    will need to recursive copy the kwcoco output directory up to
    #    S3 bucket)
    print("* Egressing KWCOCO dataset and associated STAC item *")
    # Add new assets to be egressed
    ingressed_assets['timecombined_kwcoco_file_for_bas_with_landcover'] = (
        combo_features_kwcoco_path)
    ingressed_assets['landcover_assets'] = ingress_dir / '_teamfeats'

    ingressed_assets['enriched_bas_kwcoco_file'] = combo_features_kwcoco_path
    ingressed_assets['enriched_bas_kwcoco_teamfeats'] = ingress_dir / '_teamfeats'

    smartflow_egress(ingressed_assets,
                     local_region_path,
                     config.output_path,
                     config.outbucket,
                     aws_profile=None,
                     dryrun=False,
                     newline=False)


if __name__ == "__main__":
    main()
