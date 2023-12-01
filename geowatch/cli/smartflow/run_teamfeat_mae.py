#!/usr/bin/env python3
import ubelt as ub
import scriptconfig as scfg
from geowatch.cli.smartflow_ingress import smartflow_ingress
from geowatch.cli.smartflow_egress import smartflow_egress


__debugging__ = r"""


IMAGE_NAME=watch:0.11.0-42f5cc56a-strict-pyenv3.11.2-20231013T095024-0400-from-dc16b29e

docker run \
    --runtime=nvidia \
    --volume "$HOME/temp/debug_smartflow/ingress":/tmp/ingress \
    --volume $HOME/.aws:/root/.aws:ro \
    --volume "$HOME/code":/extern_code:ro \
    --volume "$HOME/data":/extern_data:ro \
    --volume "$HOME"/.cache/pip:/pip_cache \
    --env AWS_PROFILE=iarpa \
    -it "$IMAGE_NAME" bash

(cd /root/code/watch && git remote add tmp /extern_code/watch/.git)
(cd /root/code/watch && git fetch tmp)
(cd /root/code/watch && git checkout dev/0.11.0)
(cd /root/code/watch && git pull tmp)

(cd /root/data/smart_expt_dvc && dvc remote add tmp /extern_data/dvc-repos/smart_expt_dvc/.dvc/cache)
(cd /root/data/smart_expt_dvc && dvc pull -r tmp models/wu/wu_mae_2023_04_21.dvc)

ipython

import sys, ubelt
sys.path.append(ubelt.expandpath('~/code/watch'))
from geowatch.cli.smartflow.run_teamfeat_mae import *  # NOQA


# Copied from a smartflow run that failed,

config = TeamFeatMAE(**{
    'input_path'       : 's3://smartflow-023300502152-us-west-2/smartflow/env/kw-v3-0-0/work/preeval17_batch_v103/batch/kit/KR_R001/2021-08-31/split/mono/products/site-cropped-kwcoco/items.jsonl',
    'input_region_path': 's3://smartflow-023300502152-us-west-2/smartflow/env/kw-v3-0-0/work/preeval17_batch_v103/batch/kit/KR_R001/2021-08-31/input/mono/region_models/KR_R001.geojson',
    'output_path'      : 's3://smartflow-023300502152-us-west-2/smartflow/env/kw-v3-0-0/work/preeval17_batch_v103/batch/kit/KR_R001/2021-08-31/split/mono/products/acsc_mae/items.jsonl',
    'aws_profile'      : None,
    'dryrun'           : False,
    'outbucket'        : 's3://smartflow-023300502152-us-west-2/smartflow/env/kw-v3-0-0/work/preeval17_batch_v103/batch/kit/KR_R001/2021-08-31/split/mono/products/acsc_mae',
    'newline'          : True,
    'expt_dvc_dpath'   : '/root/data/smart_expt_dvc',
})


###

python -m geowatch.tasks.mae.predict \
        --input_kwcoco=/tmp/ingress/cropped_kwcoco_for_sc.json \
        --mae_ckpt_path=/root/data/smart_expt_dvc/models/wu/wu_mae_2023_04_21/Drop6-epoch=01-val_loss=0.20.ckpt \
        --output_kwcoco=/tmp/ingress/cropped_kwcoco_for_sc_wu_mae.kwcoco.zip \
        --workers=2 \
        --assets_dname=_teamfeats



import sys, ubelt
sys.path.append(ubelt.expandpath('~/code/watch'))
from geowatch.tasks.mae.predict import *  # NOQA
kwargs = {
    'device': 'cuda:0',
    'mae_ckpt_path': '/root/data/smart_expt_dvc/models/wu/wu_mae_2023_04_21/Drop6-epoch=01-val_loss=0.20.ckpt',
    'batch_size': 1,
    'workers': 2,
    'io_workers': 8,
    'window_resolution': 1.0,
    'sensor': [
        'S2',
        'L8',
    ],
    'bands': [
        'shared',
    ],
    'patch_overlap': 0.25,
    'input_kwcoco': '/tmp/ingress/cropped_kwcoco_for_sc.json',
    'output_kwcoco': '/tmp/ingress/cropped_kwcoco_for_sc_wu_mae.kwcoco.zip',
    'assets_dname': '_teamfeats',
}

"""


class TeamFeatMAE(scfg.DataConfig):
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


def main():
    # import os
    # os.environ['NO_COLOR'] = '1'
    config = TeamFeatMAE.cli(strict=True)

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
        with_mae=1,
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
    # This is the kwcoco file with the all teamfeature outputs
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
