#!/usr/bin/env python3
import ubelt as ub
import scriptconfig as scfg
from geowatch.cli.smartflow_ingress import smartflow_ingress
from geowatch.cli.smartflow_egress import smartflow_egress

__DEBUG_INFO__ = """

import sys, ubelt
sys.path.append(ubelt.expandpath('~/code/geowatch'))
from geowatch.cli.smartflow.run_teamfeat_cold import *  # NOQA

config = TeamFeatColdConfig(**{
    'input_path'       : 's3://smartflow-023300502152-us-west-2/smartflow/env/kw-v3-0-0/work/preeval21_batch_v174/batch/kit/KR_R001/split_work/52SDG67/products/kwcoco-dataset/items.jsonl',
    'input_region_path': 's3://smartflow-023300502152-us-west-2/smartflow/env/kw-v3-0-0/work/preeval21_batch_v174/batch/kit/KR_R001/split_input/52SDG67/region_models/KR_R001.geojson',
    'output_path'      : 's3://smartflow-023300502152-us-west-2/smartflow/env/kw-v3-0-0/work/preeval21_batch_v174/batch/kit/KR_R001/split_work/52SDG67/products/cold/items.jsonl',
    'aws_profile'      : None,
    'dryrun'           : False,
    'outbucket'        : 's3://smartflow-023300502152-us-west-2/smartflow/env/kw-v3-0-0/work/preeval21_batch_v174/batch/kit/KR_R001/split_work/52SDG67/products/cold',
    'newline'          : True,
    'expt_dvc_dpath'   : '/root/data/smart_expt_dvc',
    'cold_workers'     : 2,
    'cold_config'      : None,
})

"""


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

    cold_workers = scfg.Value(2, type=int, help='Number of parallel workers that COLD will use. DEPRECATED and IGNORED, pass workers in cold_config')

    cold_config = scfg.Value(None, type=str, help=ub.paragraph(
            '''
            Raw json/yaml or a path to a json/yaml file that specifies the
            config for cold teamfeats.
            '''))


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
    node_state.print_local_invocation(config)

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

    full_input_kwcoco_fpath = ub.Path(ingressed_assets['timedense_bas_kwcoco_file'])
    timecombined_input_kwcoco_fpath = ub.Path(ingressed_assets['enriched_bas_kwcoco_file'])

    timecombined_output_kwcoco_fpath = timecombined_input_kwcoco_fpath.augment(
        stemsuffix='_cold', ext='.kwcoco.zip', multidot=True)

    from geowatch.cli import watch_coco_stats
    from kwcoco.cli import coco_stats
    watch_coco_stats.main(cmdline=0, src=full_input_kwcoco_fpath)
    coco_stats._CLI.main(cmdline=0, src=[full_input_kwcoco_fpath])

    print('Print some disk and machine statistics (again)')
    ub.cmd('df -h', verbose=3)

    # Eval18 cold config
    # sensors: L8,S2
    # adj_cloud: false
    # method: COLD
    # prob: 0.99
    # conse: 8
    # cm_interval: 60
    # year_lowbound:
    # year_highbound:
    # coefs: cv,rmse,a0,a1,b1,c1
    # coefs_bands: 0,1,2,3,4,5
    # timestamp: false
    # combine: false
    # resolution: 10GSD

    # TOOD: better passing of configs
    # Quick and dirty, just the existing prepare teamfeat script to get the
    # cold invocation. This has a specific output pattern that we hard code
    # here.
    from geowatch.cli.queue_cli import prepare_teamfeats
    from kwutil.util_yaml import Yaml
    base_fpath = ub.Path(full_input_kwcoco_fpath)
    cold_config = Yaml.coerce(config.cold_config) or {}
    cold_config['enabled'] = True

    prepare_teamfeats.main(
        cmdline=0,
        cold_config=config.cold_config,
        with_cold=True,
        expt_dvc_dpath=config.expt_dvc_dpath,
        base_fpath=full_input_kwcoco_fpath,
        assets_dname='_teamfeats',
        run=1,
        backend='serial',
    )

    # Hard coded-specific output pattern.
    subset_name = base_fpath.name.split('.')[0]
    combo_code = 'C'
    base_combo_fpath = base_fpath.parent / (f'combo_{subset_name}_{combo_code}.kwcoco.zip')
    full_output_kwcoco_fpath = base_combo_fpath

    node_state.print_current_state(ingress_dir)

    if not full_output_kwcoco_fpath.exists():
        raise FileNotFoundError(
            f'The COLD kwcoco file: {full_output_kwcoco_fpath} does not seem to exist')

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
