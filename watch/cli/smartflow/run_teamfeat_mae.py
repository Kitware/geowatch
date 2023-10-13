#!/usr/bin/env python3
import ubelt as ub
import scriptconfig as scfg
from watch.cli.smartflow_ingress import smartflow_ingress
from watch.cli.smartflow_egress import smartflow_egress


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
    import os
    os.environ['NO_COLOR'] = '1'
    config = TeamFeatMAE.cli(strict=True)
    print('config = {}'.format(ub.urepr(config, nl=1, align=':')))
    from watch.utils.util_framework import download_region
    from watch.cli import watch_coco_stats
    from kwcoco.cli import coco_stats

    ####
    # DEBUGGING:
    # Print info about what version of the code we are running on
    import watch
    print('Print current version of the code')
    ub.cmd('git log -n 1', verbose=3, cwd=ub.Path(watch.__file__).parent)

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

    output_kwcoco_fpath = ub.Path(input_kwcoco_fpath).augment(
        stemsuffix='_mae', ext='.kwcoco.zip', multidot=True)

    # TOOD: better passing of configs

    # Use the existing prepare teamfeat script to get the features invocation.
    # This has a specific output pattern that we hard code here.
    from watch.cli import prepare_teamfeats
    base_fpath = ub.Path(input_kwcoco_fpath)

    watch_coco_stats.main(cmdline=0, src=base_fpath)
    coco_stats._CLI.main(cmdline=0, src=[base_fpath])

    prepare_teamfeats.main(
        cmdline=0,
        with_mae=1,
        expt_dvc_dpath=config.expt_dvc_dpath,
        base_fpath=base_fpath,
        assets_dname='_teamfeats',
        run=1,
        backend='serial',
    )
    # Hard coded-specific output pattern.
    subset_name = base_fpath.name.split('.')[0]
    combo_code = 'E'
    base_combo_fpath = base_fpath.parent / (f'combo_{subset_name}_{combo_code}.kwcoco.zip')
    full_output_kwcoco_fpath = base_combo_fpath

    ingress_dir_contents2 = list(ingress_dir.ls())
    print('ingress_dir_contents2 = {}'.format(ub.urepr(ingress_dir_contents2, nl=1)))

    # Reroot kwcoco files to make downloaded results easier to work with
    ub.cmd(['kwcoco', 'reroot', f'--src={full_output_kwcoco_fpath}', '--inplace=1', '--absolute=0'])

    watch_coco_stats.main(cmdline=0, src=full_output_kwcoco_fpath)
    coco_stats._CLI.main(cmdline=0, src=[full_output_kwcoco_fpath])

    print("* Egressing KWCOCO dataset and associated STAC item *")

    # This is the location that COLD features will be written to.
    teamfeat_dpath = (ingress_dir / '_teamfeats').ensuredir()
    (teamfeat_dpath / 'dummy').touch()
    ingressed_assets['enriched_acsc_kwcoco_teamfeats'] = teamfeat_dpath
    # This is the kwcoco file with the all teamfeature outputs (i.e. previous
    # team features + MAE)
    ingressed_assets['enriched_acsc_kwcoco_file'] = output_kwcoco_fpath

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
