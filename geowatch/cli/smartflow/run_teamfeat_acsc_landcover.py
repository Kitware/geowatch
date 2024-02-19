#!/usr/bin/env python3
"""
TODO: rectify with run_teamfeat_landcover.py
"""
import ubelt as ub
import scriptconfig as scfg
from geowatch.cli.smartflow_ingress import smartflow_ingress
from geowatch.cli.smartflow_egress import smartflow_egress


class TeamFeatLandcover(scfg.DataConfig):
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
    expt_dvc_dpath = scfg.Value('/root/data/smart_expt_dvc', help='location of the experiment DVC repo')


def main():
    config = TeamFeatLandcover.cli(strict=True)

    print('config = {}'.format(ub.urepr(config, nl=1, align=':')))
    from geowatch.utils.util_framework import download_region

    from geowatch.utils.util_framework import NodeStateDebugger
    node_state = NodeStateDebugger()
    node_state.print_environment()

    # 1. Ingress data
    print("* Running baseline framework kwcoco ingress *")

    # TODO: these input output bucket names need to be configurable so they can
    # be run at BAS or at ACSC time and composed at the DAG level.
    ingress_dir = ub.Path('/tmp/ingress')
    ingressed_assets = smartflow_ingress(
        config.input_path,
        [
            # Pull the current teamfeature-enriched dataset to modify
            'enriched_acsc_kwcoco_file',
            'enriched_acsc_kwcoco_teamfeats',
            'enriched_acsc_kwcoco_rawbands',
        ],
        ingress_dir,
        config.aws_profile,
        config.dryrun)

    print('ingressed_assets = {}'.format(ub.urepr(ingressed_assets, nl=1)))

    # 2. Download and prune region file
    print("* Downloading and pruning region file *")
    local_region_path = ub.Path('/tmp/region.json')
    download_region(
        input_region_path=config.input_region_path,
        output_region_path=local_region_path,
        aws_profile=config.aws_profile,
        strip_nonregions=True,
    )

    ingress_dir_contents1 = list(ingress_dir.ls())
    print('ingress_dir_contents1 = {}'.format(ub.urepr(ingress_dir_contents1, nl=1)))

    input_kwcoco_fpath = ingressed_assets['enriched_acsc_kwcoco_file']

    # TOOD: better passing of configs

    # Use the existing prepare teamfeat script to get the features invocation.
    # This has a specific output pattern that we hard code here.
    from geowatch.cli import prepare_teamfeats
    base_fpath = ub.Path(input_kwcoco_fpath)
    # watch_coco_stats.main(cmdline=0, src=base_fpath)
    # coco_stats._CLI.main(cmdline=0, src=[base_fpath])

    node_state.print_current_state(ingress_dir)

    # ub.cmd(f'kwcoco validate {base_fpath}', verbose=3)
    ub.cmd(f'kwcoco stats {base_fpath}', verbose=3)
    ub.cmd(f'geowatch stats {base_fpath}', verbose=3)

    teamfeat_info = prepare_teamfeats.main(
        cmdline=0,
        with_wv_landcover=1,
        with_s2_landcover=1,
        num_wv_landcover_hidden=0,
        num_s2_landcover_hidden=0,
        expt_dvc_dpath=config.expt_dvc_dpath,
        base_fpath=base_fpath,
        assets_dname='_teamfeats',
        run=1,
        backend='serial',
    )
    final_output_paths = teamfeat_info['final_output_paths']
    assert len(final_output_paths) == 1
    full_output_kwcoco_fpath = final_output_paths[0]

    ingress_dir_contents2 = list(ingress_dir.ls())
    print('ingress_dir_contents2 = {}'.format(ub.urepr(ingress_dir_contents2, nl=1)))

    # Reroot kwcoco files to make downloaded results easier to work with
    ub.cmd(['kwcoco', 'reroot', f'--src={full_output_kwcoco_fpath}', '--inplace=1', '--absolute=0'])

    ub.cmd(f'kwcoco stats {full_output_kwcoco_fpath}', verbose=3)
    ub.cmd(f'geowatch stats {full_output_kwcoco_fpath}', verbose=3)

    print("* Egressing KWCOCO dataset and associated STAC item *")

    # This is the location that COLD features will be written to.
    teamfeat_dpath = (ingress_dir / '_teamfeats').ensuredir()
    (teamfeat_dpath / 'dummy').touch()
    ingressed_assets['enriched_acsc_kwcoco_teamfeats'] = teamfeat_dpath
    # This is the kwcoco file with the all teamfeature outputs (i.e. previous
    # team features + this one)
    ingressed_assets['enriched_acsc_kwcoco_file'] = full_output_kwcoco_fpath

    node_state.print_current_state(ingress_dir)

    smartflow_egress(ingressed_assets,
                     local_region_path,
                     config.output_path,
                     config.outbucket,
                     aws_profile=config.aws_profile,
                     dryrun=config.dryrun,
                     newline=config.newline)

    print('Finish run_teamfeat_cold')


if __name__ == "__main__":
    main()
