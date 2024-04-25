#!/usr/bin/env python3
"""
See Old Version:
    ../../../scripts/run_bas_fusion_eval3_for_baseline.py

SeeAlso:
    ~/code/watch-smartflow-dags/KIT_TA2_PREEVAL10_PYENV.py
"""
import json
import os
import scriptconfig as scfg
import shutil
import ubelt as ub
from datetime import datetime, timezone
from dateutil import parser


class BasFusionConfig(scfg.DataConfig):
    """
    Run TA-2 BAS fusion as baseline framework component

    python ~/code/watch/geowatch/cli/smartflow/run_bas_fusion.py
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
            AWS Profile to use for AWS S3 CLI commands.
            '''))

    dryrun = scfg.Value(False, isflag=True, short_alias=['d'], help='DEPRECATED. DO NOT USE.')

    outbucket = scfg.Value(None, type=str, required=True, short_alias=['o'], help=ub.paragraph(
            '''
            S3 Output directory for STAC item / asset egress
            '''))

    models_outbucket = scfg.Value(None, type=str, required=False, help=ub.paragraph(
            '''
            S3 Output directory for output region and site models (if
            not specified, defaults to using the `outbucket`
            parameter)
            '''))

    ta2_s3_collation_bucket = scfg.Value(None, type=str, help=ub.paragraph(
            '''
            S3 Location for collated TA-2 output (bucket name should
            include up to eval name)
            '''))

    previous_bas_outbucket = scfg.Value(None, type=str, help=ub.paragraph(
            '''
            S3 Output directory for previous interval BAS fusion output
            '''))

    bas_pxl_config = scfg.Value(None, type=str, help=ub.paragraph(
            '''
            Raw json/yaml or a path to a json/yaml file that specifies the
            config for fusion.predict.
            '''))

    bas_poly_config = scfg.Value(None, type=str, help=ub.paragraph(
            '''
            Raw json/yaml or a path to a json/yaml file that specifies the
            config for bas tracking.
            '''))

    previous_interval_output = scfg.Value(None, type=str, help=ub.paragraph(
            '''
            Output path for previous interval BAS DatasetGen step
            '''))

    num_years_historical = scfg.Value(None, type=int, help=ub.paragraph(
            '''
            Number of years worth of historical data to consider; does
            nothing if 'previous_interval_output' is not specified
            (i.e. for incremental mode)
            '''))

    time_dense = scfg.Value(False, isflag=True, help=ub.paragraph(
            '''
            Use time_dense imagery. Defaults to False and uses time averaged data.
            '''))

    egress_intermediate_outputs = scfg.Value(False, isflag=True, help=ub.paragraph(
        '''
        If true egress intermediate heatmaps, otherwise only egress the geojson
        '''))


__debug_notes__ = r"""
config = {
    'input_path'             : 's3://smartflow-023300502152-us-west-2/smartflow/env/kw-v3-0-0/work/preeval17_batch_v130/batch/kit/KR_R002/2021-08-31/split/mono/products/cold/items.jsonl',
    'input_region_path'      : 's3://smartflow-023300502152-us-west-2/smartflow/env/kw-v3-0-0/work/preeval17_batch_v130/batch/kit/KR_R002/2021-08-31/input/mono/region_models/KR_R002.geojson',
    'output_path'            : 's3://smartflow-023300502152-us-west-2/smartflow/env/kw-v3-0-0/work/preeval17_batch_v130/batch/kit/KR_R002/2021-08-31/split/mono/products/bas-fusion/items.jsonl',
    'aws_profile'            : None,
    'dryrun'                 : False,
    'outbucket'              : 's3://smartflow-023300502152-us-west-2/smartflow/env/kw-v3-0-0/work/preeval17_batch_v130/batch/kit/KR_R002/2021-08-31/split/mono/products/bas-fusion',
    'ta2_s3_collation_bucket': None,
    'previous_bas_outbucket' : None,
    'bas_pxl_config'         : 'chip_dims: auto\nchip_overlap: 0.3\nfixed_resolution: 10GSD\nnum_workers: 24\npackage_fpath: /root/data/smart_expt_dvc/models/fusion/uconn/D7-V2-COLD-candidate/epoch=203-step=4488.pt\ntime_sampling: soft4\ntime_span: auto\ntta_fliprot: 3\ntta_time: 3',
    'bas_poly_config'        : 'agg_fn: probs\ninner_agg_fn: mean\ninner_window_size: 1y\nmax_area_square_meters: 8000000\nmin_area_square_meters: 7200\nmoving_window_size: null\nnorm_ord: inf\npoly_merge_method: v2\npolygon_simplify_tolerance: 1\nresolution: 10GSD\nthresh: 0.375\ntime_thresh: 0.8',
}
config = BasFusionConfig(**config)

"""


def main():
    config = BasFusionConfig.cli(strict=True)
    print('config = {}'.format(ub.urepr(dict(config), nl=1, align=':')))
    run_bas_fusion_for_baseline(config)


def filter_kwcoco_images_by_datetime(input_kwcoco_fpath,
                                     output_kwcoco_fpath,
                                     date,
                                     mode='remove-after'):
    assert mode in {'remove-after', 'remove-before'}

    import kwcoco
    from operator import lt, ge

    if mode == 'remove-before':
        op = lt
    elif mode == 'remove-after':
        op = ge
    else:
        raise RuntimeError("Expecting kwarg `mode` to be one "
                           "of: {'remove-before', 'remove-after'}")

    input_dset = kwcoco.CocoDataset(input_kwcoco_fpath)

    image_ids_to_remove = []
    for o in input_dset.images().objs:
        odate = parser.parse(o['date_captured'])

        if op(odate, date):
            image_ids_to_remove.append(o['id'])

    input_dset.remove_images(image_ids_to_remove)

    input_dset.dump(output_kwcoco_fpath)


def run_bas_fusion_for_baseline(config):
    from geowatch.cli.smartflow_ingress import smartflow_ingress
    from geowatch.cli.smartflow_egress import smartflow_egress
    from geowatch.cli.concat_kwcoco_videos import concat_kwcoco_datasets
    from geowatch.utils.util_framework import download_region
    from geowatch.tasks.fusion.predict import predict
    from kwutil.util_yaml import Yaml
    from geowatch.utils import util_framework
    from geowatch.utils import util_fsspec

    ####
    from geowatch.utils.util_framework import NodeStateDebugger
    node_state = NodeStateDebugger()
    node_state.print_environment()

    input_path = config.input_path
    input_region_path = config.input_region_path
    outbucket = config.outbucket
    aws_profile = config.aws_profile
    dryrun = config.dryrun
    previous_bas_outbucket = config.previous_bas_outbucket
    ta2_s3_collation_bucket = config.ta2_s3_collation_bucket

    if aws_profile is not None:
        # This should be sufficient, but it is not tested.
        util_fsspec.S3Path._new_fs(profile=aws_profile)

    assert not dryrun, 'unsupported'

    # 1. Ingress data
    print("* Running baseline framework kwcoco ingress *")
    ingress_dir = ub.Path('/tmp/ingress')
    assets = [
        'enriched_bas_kwcoco_file',
        'enriched_bas_kwcoco_teamfeats',
        'enriched_bas_kwcoco_rawbands',
        {"key": 'hacked_cold_assets', "allow_missing": True},
        {"key": 'landcover_assets', "allow_missing": True},
    ]
    if config.time_dense:
        assets += [
            'timedense_bas_kwcoco_file',
            'timedense_bas_kwcoco_rawbands',
        ]

    ingressed_assets = smartflow_ingress(
        input_path=input_path,
        assets=assets,
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
        ensure_comments=True,
    )

    from geowatch.geoannots.geomodels import RegionModel
    region = RegionModel.coerce(local_region_path)

    # Returned as datetime
    current_interval_end_date = region.end_date

    if current_interval_end_date.month == 1 and current_interval_end_date.day == 1:
        # If current interval ends at the start of a year,
        # consider the "current" year to be the previous one
        current_interval_year = current_interval_end_date.year - 1
    else:
        current_interval_year = current_interval_end_date.year

    # Granularity of filtering only by year currently since we're
    # using data time-averaged over a year
    if config.num_years_historical is None:
        min_date = datetime(
            current_interval_year,
            1, 1, tzinfo=timezone.utc)
    else:
        min_date = datetime(
            current_interval_year - (config.num_years_historical - 1),
            1, 1, tzinfo=timezone.utc)

    # Determine the region_id in the region file.
    region_id = region.region_id

    # 3. Run fusion
    print("* Running BAS fusion *")
    bas_fusion_kwcoco_path = ingress_dir / 'bas_fusion_kwcoco.json'

    node_state.print_current_state(ingress_dir)

    # TODO: remove these defaults or replace them with whatever is the
    # default in predict. The params should be fully given in the DAG, not
    # here.
    default_predict_config = ub.udict({
          "chip_overlap": 0.3,
          "chip_dims": "auto",
          "time_span": "auto",
          "time_sampling": "auto",
          "drop_unused_frames": True,
          "batch_size": 1,
          "num_workers": 2,
          'package_fpath': None,
    })
    bas_pxl_config = default_predict_config | Yaml.coerce(config.bas_pxl_config or {})

    if bas_pxl_config.get('package_fpath', None) is None:
        raise ValueError('Requires package_fpath')

    print('bas_pxl_config = {}'.format(ub.urepr(bas_pxl_config, nl=1)))

    ingress_kwcoco_path = ingressed_assets['enriched_bas_kwcoco_file']

    incremental_assets_for_egress = {}
    if config.previous_interval_output is None:
        filtered_ingress_kwcoco_path = ingress_kwcoco_path
    else:
        filtered_ingress_kwcoco_path =\
            ingress_dir / 'filtered_enriched_kwcoco.json'

        filter_kwcoco_images_by_datetime(
            ingress_kwcoco_path,
            filtered_ingress_kwcoco_path,
            min_date, mode='remove-before')

        incremental_assets_for_egress['filtered_enriched_kwcoco'] =\
            filtered_ingress_kwcoco_path

    if 0:
        import kwcoco
        src_dset = kwcoco.CocoDataset(ingress_kwcoco_path)
        src_dset.validate()

    predict(devices='0,',
            write_preds=False,
            write_probs=True,
            with_change=False,
            with_saliency=True,
            with_class=False,
            test_dataset=filtered_ingress_kwcoco_path,
            pred_dataset=bas_fusion_kwcoco_path,
            **bas_pxl_config)

    node_state.print_current_state(ingress_dir)

    # 3.1. If a previous interval was run; concatenate BAS fusion
    # output KWCOCO files for tracking
    if previous_bas_outbucket is not None:
        combined_bas_fusion_kwcoco_path = ingress_dir / 'combined_bas_fusion_kwcoco.json'

        previous_ingress_dir = ub.Path('/tmp/ingress_previous')

        previous_bas_outbucket = util_fsspec.FSPath.coerce(previous_bas_outbucket)
        previous_ingress_dir = util_fsspec.FSPath.coerce(previous_ingress_dir)
        previous_bas_outbucket.copy(previous_ingress_dir, recursive=True)

        previous_bas_fusion_kwcoco_path = previous_ingress_dir / 'combined_bas_fusion_kwcoco.json'

        # On first interval nothing will be copied down so need to
        # check that we have the input explicitly
        if os.path.isfile(previous_bas_fusion_kwcoco_path):
            concat_kwcoco_datasets(
                (previous_bas_fusion_kwcoco_path, bas_fusion_kwcoco_path),
                combined_bas_fusion_kwcoco_path)
            # Copy saliency assets from previous bas fusion
            shutil.copy_tree(
                previous_ingress_dir / '_assets/pred_saliency',
                ingress_dir / '_assets/pred_saliency'
            )

            # Copy original assets from previous bas rusion
            shutil.copy_tree(
                previous_ingress_dir / region_id,
                ingress_dir / region_id
            )
        else:
            # Copy current bas_fusion_kwcoco_path to combined path as
            # this is the first interval
            shutil.copy(bas_fusion_kwcoco_path,
                        combined_bas_fusion_kwcoco_path)
    else:
        combined_bas_fusion_kwcoco_path = bas_fusion_kwcoco_path

    previous_ingressed_assets = None
    if config.previous_interval_output is not None:
        print('* Combining previous interval time combined kwcoco with '
              'current *')
        previous_ingress_dir = ub.Path('/tmp/ingress_previous')
        try:
            previous_ingressed_assets = smartflow_ingress(
                config.previous_interval_output,
                ['bas_pred_saliency_assets',
                 'combined_bas_fusion_kwcoco_path'],
                previous_ingress_dir,
                config.aws_profile,
                config.dryrun)
        except FileNotFoundError:
            print("** Warning: Couldn't ingress previous interval output; "
                  "assuming this is the first interval **")
            combined_kwcoco_path = combined_bas_fusion_kwcoco_path
        else:
            combined_kwcoco_path = ingress_dir / 'combined_bas_fusion_kwcoco.json'

            filtered_previous_bas_fusion_kwcoco_path =\
                previous_ingress_dir / 'filtered_bas_fusion_kwcoco.json'

            filter_kwcoco_images_by_datetime(
                previous_ingressed_assets['combined_bas_fusion_kwcoco_path'],
                filtered_previous_bas_fusion_kwcoco_path,
                min_date, mode='remove-after')

            incremental_assets_for_egress['filtered_bas_fusion_kwcoco'] =\
                filtered_previous_bas_fusion_kwcoco_path

            from geowatch.cli.concat_kwcoco_videos import concat_kwcoco_datasets
            concat_kwcoco_datasets(
                (filtered_previous_bas_fusion_kwcoco_path,
                 combined_bas_fusion_kwcoco_path),
                combined_kwcoco_path)

            # Copy saliency assets from previous bas fusion
            shutil.copytree(previous_ingress_dir / 'pred_saliency',
                            ingress_dir / '_assets' / 'pred_saliency',
                            dirs_exist_ok=True)
    else:
        combined_kwcoco_path = combined_bas_fusion_kwcoco_path

    # 4. Compute tracks (BAS)
    print("* Computing tracks (BAS) *")
    bas_region_models_outdir = (ingress_dir / 'region_models').ensuredir()
    bas_site_models_outdir = (ingress_dir / 'site_models_bas').ensuredir()
    region_models_manifest_outdir = (ingress_dir / 'tracking_manifests_bas').ensuredir()
    region_models_manifest_outpath = region_models_manifest_outdir / 'region_models_manifest.json'
    site_models_manifest_outpath = region_models_manifest_outdir / 'site_models_manifest.json'

    # Copy input region model into region_models outdir to be updated
    # (rather than generated from tracking, which may not have the
    # same bounds as the original)
    shutil.copy(local_region_path, bas_region_models_outdir / f'{region_id}.geojson')

    # TODO: remove these defaults or replace them with whatever is the
    # default in tracker. The params should be fully given in the DAG,
    # not here.
    default_bas_tracking_config = ub.udict({
        "thresh": 0.1,
        "moving_window_size": None,
        "polygon_simplify_tolerance": 1,
        "max_area_behavior": 'ignore'
    })
    bas_tracking_config = (default_bas_tracking_config
                           | Yaml.coerce(config.bas_poly_config or {}))

    min_area_square_meters = bas_tracking_config.get('min_area_square_meters', None)
    time_pad_after = bas_tracking_config.pop('time_pad_after', None)
    time_pad_before = bas_tracking_config.pop('time_pad_before', None)
    # TODO: use smart_pipeline.BAS_PolygonPrediction
    tracked_bas_kwcoco_path = '_tracked'.join(
        os.path.splitext(combined_kwcoco_path))
    ub.cmd([
        'python', '-m', 'geowatch.cli.run_tracker',
        combined_kwcoco_path,
        '--out_sites_dir', bas_site_models_outdir,
        '--out_sites_fpath', site_models_manifest_outpath,
        '--out_site_summaries_dir', bas_region_models_outdir,
        '--out_site_summaries_fpath', region_models_manifest_outpath,
        '--out_kwcoco', tracked_bas_kwcoco_path,
        '--default_track_fn', 'saliency_heatmaps',
        '--append_mode', 'True',
        '--time_pad_after', str(time_pad_after),
        '--time_pad_before', str(time_pad_before),
        # TODO:
        # use boundary_region here?
        '--track_kwargs', json.dumps(bas_tracking_config)],
        check=True, verbose=3)

    # Remove after boundary_region here
    cropped_region_models_outdir = (ingress_dir / 'cropped_region_models_bas').ensuredir()
    cropped_site_models_outdir = (ingress_dir / 'cropped_site_models_bas').ensuredir()

    node_state.print_current_state(ingress_dir)

    crop_cmd = [
        'python', '-m', 'geowatch.cli.crop_sites_to_regions',
        '--site_models', bas_site_models_outdir / '*.geojson',
        '--region_models', bas_region_models_outdir / '*.geojson',
        '--new_site_dpath', cropped_site_models_outdir,
        '--new_region_dpath', cropped_region_models_outdir,
        '--min_area_square_meters', str(min_area_square_meters)
    ]
    ub.cmd(crop_cmd, check=True, verbose=3)

    # Validate and fix all outputs
    try:
        util_framework.fixup_and_validate_site_and_region_models(
            region_dpath=cropped_region_models_outdir,
            site_dpath=cropped_site_models_outdir,
        )
    except Exception as ex:  # noqa
        print(f'Encountered Exception, ex={ex}, uploading debug informaiton and then exiting')
        debug_s3_outdir = os.path.join(outbucket, '_debug/')
        cropped_region_models_outdir = util_fsspec.FSPath.coerce(cropped_region_models_outdir)
        debug_s3_outdir = util_fsspec.FSPath.coerce(debug_s3_outdir)
        cropped_region_models_outdir.copy(debug_s3_outdir, recursive=True)
        cropped_site_models_outdir.copy(debug_s3_outdir, recursive=True)
        raise

    if __debug__:
        from geowatch.geoannots import geomodels
        all_regions_models = list(geomodels.RegionModel.coerce_multiple(cropped_region_models_outdir))
        all_site_models = list(geomodels.SiteModel.coerce_multiple(cropped_site_models_outdir))
        assert len(all_regions_models) == 1, 'should only be 1 output region model'
        out_region_model = all_regions_models[0]
        num_site_summaries = len(list(out_region_model.site_summaries()))
        num_sites = len(all_site_models)
        assert num_sites == num_site_summaries, 'number of site summaries should be the same as the number of sites'

    # 6. (Optional) collate TA-2 output
    if ta2_s3_collation_bucket is not None:
        # only used if we are going to submit our BAS predictions as the final
        # ones?
        print("* Collating TA-2 output")
        util_framework.ta2_collate_output(None,
                                          cropped_region_models_outdir,
                                          cropped_site_models_outdir,
                                          ta2_s3_collation_bucket)

    EGRESS_INTERMEDIATE_OUTPUTS = config.egress_intermediate_outputs
    if EGRESS_INTERMEDIATE_OUTPUTS:
        # Reroot kwcoco files to make downloaded results easier to work with
        ub.cmd(['kwcoco', 'reroot', f'--src={bas_fusion_kwcoco_path}', '--inplace=1', '--absolute=0'])
        ub.cmd(['kwcoco', 'reroot', f'--src={tracked_bas_kwcoco_path}', '--inplace=1', '--absolute=0'])
        # Add BAS saliency outputs to egressed attributes for debugging
        ingressed_assets['bas_pred_saliency_assets'] = ingress_dir / '_assets/pred_saliency'
        ingressed_assets['bas_fusion_kwcoco_path'] = bas_fusion_kwcoco_path
        ingressed_assets['combined_bas_fusion_kwcoco_path'] = combined_kwcoco_path
        ingressed_assets['bas_original_site_models_outdir'] = bas_site_models_outdir
        ingressed_assets['bas_original_region_models_outdir'] = bas_region_models_outdir
        ingressed_assets['tracked_bas_kwcoco_path'] = tracked_bas_kwcoco_path

    node_state.print_current_state(ingress_dir)

    models_to_egress = {'cropped_region_models_bas': cropped_region_models_outdir,
                        'cropped_site_models_bas': cropped_site_models_outdir}
    if config.models_outbucket is not None:
        # Egress region models to a dedicated bucket
        models_egressed = smartflow_egress(
            models_to_egress,
            local_region_path,
            os.path.join(config.models_outbucket, 'models_only_items.jsonl'),
            config.models_outbucket,
            aws_profile=None,
            dryrun=False,
            newline=False)

        ingressed_assets = {**ingressed_assets, **models_egressed}
    else:
        ingressed_assets = {**ingressed_assets, **models_to_egress}

    # 6. Egress (envelop KWCOCO dataset in a STAC item and egress;
    #    will need to recursive copy the kwcoco output directory up to
    #    S3 bucket)
    print("* Egressing KWCOCO dataset and associated STAC item *")
    smartflow_egress(ingressed_assets,
                     local_region_path,
                     config.output_path,
                     config.outbucket,
                     aws_profile=None,
                     dryrun=False,
                     newline=False)


if __name__ == "__main__":
    main()
