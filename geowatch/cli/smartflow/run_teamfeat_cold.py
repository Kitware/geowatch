#!/usr/bin/env python3
import ubelt as ub
import scriptconfig as scfg
from geowatch.cli.smartflow_ingress import smartflow_ingress
from geowatch.cli.smartflow_egress import smartflow_egress


class TeamFeatColdConfig(scfg.DataConfig):
    """
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

    cold_workers = scfg.Value(4, type=int, help='Number of parallel workers that COLD will use')


def main():
    # import os
    # os.environ['NO_COLOR'] = '1'
    config = TeamFeatColdConfig.cli(strict=True)
    print('config = {}'.format(ub.urepr(config, nl=1, align=':')))
    from geowatch.utils.util_framework import download_region
    from geowatch.mlops.pipeline_nodes import ProcessNode
    from geowatch.utils.util_framework import NodeStateDebugger

    node_state = NodeStateDebugger()
    node_state.print_environment()

    # 1. Ingress data
    print("* Running baseline framework kwcoco ingress *")
    ingress_dir = ub.Path('/tmp/ingress')

    node_state.print_current_state(ingress_dir)

    ingressed_assets = smartflow_ingress(
        config.input_path,
        [
            # Pull the current teamfeature-enriched dataset to modify
            'enriched_bas_kwcoco_file',
            'enriched_bas_kwcoco_teamfeats',
            'enriched_bas_kwcoco_rawbands',

            # Pull the dense temporal data needed by COLD
            'timedense_bas_kwcoco_file',
            'timedense_bas_kwcoco_rawbands'
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

    # NOTE:
    # For COLD we need to compute on the full non-time-combined data,
    # and then transfer the features to the time-combined data.
    node_state.print_current_state(ingress_dir)

    full_input_kwcoco_fpath = ingressed_assets['timedense_bas_kwcoco_file']
    timecombined_input_kwcoco_fpath = ingressed_assets['enriched_bas_kwcoco_file']

    timecombined_output_kwcoco_fpath = ub.Path(timecombined_input_kwcoco_fpath).augment(
        stemsuffix='_cold', ext='.kwcoco.zip', multidot=True)

    from geowatch.cli import watch_coco_stats
    from kwcoco.cli import coco_stats
    watch_coco_stats.main(cmdline=0, src=full_input_kwcoco_fpath)
    coco_stats._CLI.main(cmdline=0, src=[full_input_kwcoco_fpath])

    print('Print some disk and machine statistics (again)')
    ub.cmd('df -h', verbose=3)

    # TOOD: better passing of configs

    # Quick and dirty, just the existing prepare teamfeat script to get the
    # cold invocation. This has a specific output pattern that we hard code
    # here.
    from geowatch.cli import prepare_teamfeats
    base_fpath = ub.Path(full_input_kwcoco_fpath)
    prepare_teamfeats.main(
        cmdline=0,
        with_cold=1,
        expt_dvc_dpath=config.expt_dvc_dpath,
        base_fpath=full_input_kwcoco_fpath,
        cold_workers=config.cold_workers,
        assets_dname='_teamfeats',
        cold_workermode='process',
        run=1,
        backend='serial',
    )

    # Hard coded-specific output pattern.
    subset_name = base_fpath.name.split('.')[0]
    combo_code = 'C'
    base_combo_fpath = base_fpath.parent / (f'combo_{subset_name}_{combo_code}.kwcoco.zip')
    full_output_kwcoco_fpath = base_combo_fpath

    node_state.print_current_state(ingress_dir)

    watch_coco_stats.main(cmdline=0, src=full_output_kwcoco_fpath)
    coco_stats._CLI.main(cmdline=0, src=[full_output_kwcoco_fpath])

    watch_coco_stats.main(cmdline=0, src=timecombined_input_kwcoco_fpath)
    coco_stats._CLI.main(cmdline=0, src=[timecombined_input_kwcoco_fpath])

    ###
    # Execute the transfer of COLD features to the time-combined dataset
    transfer_node = ProcessNode(
        command=ub.codeblock(
            r'''
            python -m geowatch.tasks.cold.transfer_features
            '''),
        in_paths={
            'coco_fpath': full_output_kwcoco_fpath,
            'combine_fpath': timecombined_input_kwcoco_fpath,
        },
        out_paths={
            'new_coco_fpath': timecombined_output_kwcoco_fpath,
        },
        config={
            'copy_assets': True,
            'io_workers': 4,
        },
        node_dpath='.',
    )
    command = transfer_node.final_command()
    ub.cmd(command, shell=True, capture=False, verbose=3, check=True)

    # Reroot kwcoco files to make downloaded results easier to work with
    ub.cmd(['kwcoco', 'reroot', f'--src={timecombined_output_kwcoco_fpath}', '--inplace=1', '--absolute=0'])

    watch_coco_stats.main(cmdline=0, src=timecombined_output_kwcoco_fpath)
    coco_stats._CLI.main(cmdline=0, src=[timecombined_output_kwcoco_fpath])

    node_state.print_current_state(ingress_dir)

    print("* Egressing KWCOCO dataset and associated STAC item *")

    # This is the location that COLD features will be written to.
    (ingress_dir / '_teamfeats').ensuredir()
    (ingress_dir / '_teamfeats/dummy').touch()
    ingressed_assets['enriched_bas_kwcoco_teamfeats'] = ingress_dir / '_teamfeats'

    # HACK: teamfeats is not ACTUALLY where the features were written. They are
    # in the reccg folder, we should fix this, but for now lets just get an
    # end-to-end run.
    ingressed_assets['hacked_cold_assets'] = ingress_dir / 'reccg'
    # This is the kwcoco file with the all teamfeature outputs (i.e. previous
    # team features + COLD)
    ingressed_assets['enriched_bas_kwcoco_file'] = timecombined_output_kwcoco_fpath

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
