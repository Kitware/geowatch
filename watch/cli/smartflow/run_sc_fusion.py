#!/usr/bin/env python3
"""
See Old Version:
    ../../../scripts/run_sc_fusion_eval3_for_baseline.py

SeeAlso:
    ~/code/watch-smartflow-dags/KIT_TA2_PREEVAL10_PYENV.py
"""
import json
import os
import shutil
import sys
import traceback
import scriptconfig as scfg
import ubelt as ub


class SCFusionConfig(scfg.DataConfig):
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

    aws_profile = scfg.Value(None, type=str, help=ub.paragraph(
            '''
            AWS Profile to use for AWS S3 CLI commands
            '''))
    dryrun = scfg.Value(False, isflag=True, short_alias=['d'], help='DEPRECATED. DO NOT USE')
    outbucket = scfg.Value(None, type=str, required=True, short_alias=['o'], help=ub.paragraph(
            '''
            S3 Output directory for STAC item / asset egress
            '''))

    ta2_s3_collation_bucket = scfg.Value(None, type=str, help=ub.paragraph(
            '''
            S3 Location for collated TA-2 output (bucket name should
            include up to eval name)
            '''))

    sc_pxl_config = scfg.Value(None, type=str, help=ub.paragraph(
            '''
            Raw json/yaml or a path to a json/yaml file that specifies the
            config for fusion.predict.
            '''))

    sc_poly_config = scfg.Value(None, type=str, help=ub.paragraph(
            '''
            Raw json/yaml or a path to a json/yaml file that specifies the
            config for SC tracking.
            '''))


def main():
    config = SCFusionConfig.cli(strict=True)
    print('config = {}'.format(ub.urepr(dict(config), nl=1, align=':')))
    run_sc_fusion_for_baseline(config)


def run_sc_fusion_for_baseline(config):
    from watch.cli.smartflow_ingress import smartflow_ingress
    from watch.cli.smartflow_egress import smartflow_egress
    from watch.tasks.fusion.predict import predict  # NOQA
    from watch.tasks.fusion.datamodules.temporal_sampling import TimeSampleError
    from watch.utils.util_framework import download_region, determine_region_id
    from kwutil.util_yaml import Yaml
    from watch.utils import util_framework
    from watch.mlops import smart_pipeline

    if config.aws_profile is not None:
        # This should be sufficient, but it is not tested.
        from watch.utils import util_fsspec
        util_fsspec.S3Path._new_fs(profile=config.aws_profile)

    ####
    # DEBUGGING:
    # Print info about what version of the code we are running on
    import watch
    print('Print current version of the code')
    ub.cmd('git log -n 1', verbose=3, cwd=ub.Path(watch.__file__).parent)

    # 1. Ingress data
    print("* Running baseline framework kwcoco ingress *")

    # ingress_dir = ub.Path('/home/joncrall/data/dvc-repos/smart_expt_dvc/_airflow/temp').ensuredir()
    ingress_dir = ub.Path('/tmp/ingress')

    ingressed_assets = smartflow_ingress(
        input_path=config.input_path,
        assets=[
            {'key': 'cropped_region_models_bas'},
            {'key': 'sv_out_region_models', 'allow_missing': False},
            {'key': 'cropped_kwcoco_for_sc'},
            {'key': 'cropped_kwcoco_for_sc_assets'}
        ],
        outdir=ingress_dir,
        aws_profile=config.aws_profile,
        dryrun=config.dryrun
    )

    # Get the first set of BAS site summaries that are available
    region_model_key_priority = [
        # Use filtered SV site summaries when possible
        # Otherwise fallback to bas site summaries
        'sv_out_region_models',
        'cropped_region_models_bas',
    ]
    for key in region_model_key_priority:
        input_site_summary_dpath = ingressed_assets[key]
        if os.path.exists(input_site_summary_dpath):
            break
    assert os.path.exists(input_site_summary_dpath)
    print(f'Found input site summary dpath: {input_site_summary_dpath}')

    # # 2. Download and prune region file
    print("* Downloading and pruning region file *")
    local_region_path = '/tmp/region.json'

    download_region(
        input_region_path=config.input_region_path,
        output_region_path=local_region_path,
        aws_profile=config.aws_profile,
        strip_nonregions=True,
    )

    # Determine the region_id in the region file.
    region_id = determine_region_id(local_region_path)

    sc_fusion_kwcoco_path = ingress_dir / 'sc_fusion_kwcoco.json'

    tracked_sc_kwcoco_path = '_tracked'.join(
        os.path.splitext(sc_fusion_kwcoco_path))

    site_models_outdir = (ingress_dir / 'sc_out_site_models').ensuredir()
    region_models_outdir = (ingress_dir / 'sc_out_region_models').ensuredir()
    site_models_manifest_outdir = (ingress_dir / 'tracking_manifests_sc').ensuredir()

    site_models_manifest_outpath = site_models_manifest_outdir / 'site_models_manifest.json'
    # Copy input region model into region_models outdir to be updated
    # (rather than generated from tracking, which may not have the
    # same bounds as the original)
    shutil.copy(local_region_path, region_models_outdir / f'{region_id}.geojson')

    region_models_manifest_fpath = ingress_dir / 'sc_out_region_models_manifest.json'

    print('* Printing current directory contents (1/5)')
    cwd_paths = sorted([p.resolve() for p in ingress_dir.glob('*')])
    print('cwd_paths = {}'.format(ub.urepr(cwd_paths, nl=1)))

    # 3.1. Check that we have at least one "video" (BAS identified
    # site) to run over; if not skip SC fusion and KWCOCO to GeoJSON
    with open(ingressed_assets['cropped_kwcoco_for_sc']) as f:
        ingress_kwcoco_data = json.load(f)

    if len(ingress_kwcoco_data.get('videos', ())) > 0:
        # 3. Run fusion
        print('*********************')
        print("* Running SC fusion *")

        # The params should be fully given in the DAG.
        sc_pxl_config = Yaml.coerce(config.sc_pxl_config or {})
        if sc_pxl_config.get('package_fpath', None) is None:
            raise ValueError('Requires package_fpath')

        sc_pxl = smart_pipeline.SC_HeatmapPrediction(root_dpath=ingress_dir)
        sc_pxl.configure({
            'pred_pxl_fpath': sc_fusion_kwcoco_path,
            'test_dataset': ingressed_assets['cropped_kwcoco_for_sc'],
        } | sc_pxl_config)
        command = sc_pxl.command()

        try:
            ub.cmd(command, check=True, verbose=3, system=True)
            print('* Printing current directory contents (2/5)')
            cwd_paths = sorted([p.resolve() for p in ingress_dir.glob('*')])
            print('cwd_paths = {}'.format(ub.urepr(cwd_paths, nl=1)))
        except TimeSampleError:
            # FIXME: wont work anymore with mlops. Not sure if needed.
            # Can always catch a CalledProcessError and inspect stdout
            print("* Error with time sampling during SC Predict "
                  "(shown below) -- attempting to continue anyway")
            traceback.print_exception(*sys.exc_info())
        else:

            # 4. Compute tracks (SC)
            print('*************************')
            print("* Computing tracks (SC) *")

            # Params are fully specified in the DAG
            sc_track_kwargs = Yaml.coerce(config.sc_poly_config or {})
            tracked_sc_kwcoco_path = '_tracked'.join(
                os.path.splitext(sc_fusion_kwcoco_path))
            final_sc_poly_config = {
                'pred_pxl_fpath': sc_fusion_kwcoco_path,               # Sets --input_kwcoco
                'site_summaries_fpath': region_models_manifest_fpath,  # Sets --out_site_summaries_fpath
                'site_summaries_dpath': region_models_outdir,          # Sets --out_site_summaries_dir
                'sites_dpath': site_models_outdir,                     # Sets --out_sites_dir
                'sites_fpath': site_models_manifest_outpath,           # Sets --out_sites_fpath
                'poly_kwcoco_fpath': tracked_sc_kwcoco_path,           # Sets --out_kwcoco
                'site_summary': ub.Path(input_site_summary_dpath) / '*.geojson',  # Sets --site_summary
                'append_mode': True,
            } | sc_track_kwargs
            sc_poly = smart_pipeline.SC_PolygonPrediction(root_dpath=ingress_dir)
            sc_poly.configure(final_sc_poly_config)
            command = sc_poly.command()
            ub.cmd(command, check=True, verbose=3, system=True)

            print('* Printing current directory contents (3/5)')
            cwd_paths = sorted([p.resolve() for p in ingress_dir.glob('*')])
            print('cwd_paths = {}'.format(ub.urepr(cwd_paths, nl=1)))

            # Add in intermediate outputs for debugging
            ingressed_assets['sc_heatmap_kwcoco_file'] = sc_fusion_kwcoco_path
            ingressed_assets['sc_tracked_kwcoco_file'] = tracked_sc_kwcoco_path

    cropped_site_models_outdir = ingress_dir / 'cropped_site_models'
    os.makedirs(cropped_site_models_outdir, exist_ok=True)
    cropped_region_models_outdir = ingress_dir / 'cropped_region_models'
    sc_heatmap_dpath = ingress_dir / '_assets'
    os.makedirs(cropped_region_models_outdir, exist_ok=True)

    ub.cmd([
        'python', '-m', 'watch.cli.crop_sites_to_regions',
        '--site_models', site_models_outdir / '*.geojson',
        '--region_models', region_models_outdir / f'{region_id}.geojson',
        '--new_site_dpath', cropped_site_models_outdir,
        '--new_region_dpath', cropped_region_models_outdir
    ], check=True, verbose=3, capture=False)

    print('* Printing current directory contents (4/5)')
    cwd_paths = sorted([p.resolve() for p in ingress_dir.glob('*')])
    print('cwd_paths = {}'.format(ub.urepr(cwd_paths, nl=1)))

    # Validate and fix all outputs
    print('Fixup and validate outputs')
    util_framework.fixup_and_validate_site_and_region_models(
        region_dpath=cropped_region_models_outdir,
        site_dpath=cropped_site_models_outdir,
    )

    print('* Printing current directory contents (5/5)')
    cwd_paths = sorted([p.resolve() for p in ingress_dir.glob('*')])
    print('cwd_paths = {}'.format(ub.urepr(cwd_paths, nl=1)))

    # 5. Egress (envelop KWCOCO dataset in a STAC item and egress;
    #    will need to recursive copy the kwcoco output directory up to
    #    S3 bucket)
    print("* Egressing KWCOCO dataset and associated STAC item *")
    ingressed_assets['cropped_site_models_sc'] = cropped_site_models_outdir
    ingressed_assets['cropped_region_models_sc'] = cropped_region_models_outdir

    # Add in intermediate outputs for debugging
    EGRESS_FUSION_HEATMAPS = True
    if EGRESS_FUSION_HEATMAPS:
        ingressed_assets['sc_heatmap_kwcoco_file'] = sc_fusion_kwcoco_path
        ingressed_assets['sc_tracked_kwcoco_file'] = tracked_sc_kwcoco_path
        ingressed_assets['sc_heatmap_assets'] = sc_heatmap_dpath
        ingressed_assets['sc_tracking_manifest_dpath'] = site_models_manifest_outdir
        if region_models_manifest_fpath.exists():
            ingressed_assets['sc_tracking_manifest_fpath'] = region_models_manifest_fpath

    smartflow_egress(ingressed_assets,
                     local_region_path,
                     config.output_path,
                     config.outbucket,
                     aws_profile=config.aws_profile,
                     dryrun=False,
                     newline=False)

    # 6. (Optional) collate TA-2 output
    if config.ta2_s3_collation_bucket is not None:
        print("* Collating TA-2 output")
        util_framework.ta2_collate_output(None,
                                          cropped_region_models_outdir,
                                          cropped_site_models_outdir,
                                          config.ta2_s3_collation_bucket)


if __name__ == "__main__":
    main()
