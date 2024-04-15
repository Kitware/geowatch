#!/usr/bin/env python3
"""
SeeAlso:
    ~/code/watch/geowatch/cli/coco_time_combine.py
    ~/code/watch/geowatch/cli/queue_cli/prepare_teamfeats.py
"""
import scriptconfig as scfg
import ubelt as ub
from cmd_queue.cli_boilerplate import CMDQueueConfig


class PrepareTimeAverages(CMDQueueConfig):
    """
    Prepare a temporally averaged dataset on multiple regions
    """
    regions = scfg.Value('all', type=str, help=ub.paragraph(
        '''
        The regions to time average (this is not a robust implementation)
        '''))

    resolution = '10GSD'

    time_window = '1year'
    merge_method = 'mean'
    remove_seasons = scfg.Value([], nargs='+')
    spatial_tile_size = None
    mask_low_quality = scfg.Value(True)

    combine_workers = scfg.Value(4, help='number of workers per combine job')

    input_bundle_dpath = scfg.Value(None)
    output_bundle_dpath = scfg.Value(None)

    skip_existing = scfg.Value(True)
    cache = scfg.Value(True)

    reproject = scfg.Value(False, isflag=True, help='Enable reprojection of annotations. Requires true_site_dpath and true_region_dpath be specified')
    true_site_dpath = scfg.Value(None)
    true_region_dpath = scfg.Value(None)

    max_images_per_group = scfg.Value(None, help=ub.paragraph(
        '''
        If specified, the averaging operation (e.g. mean/median) will only
        consider a subset of the images within each temporal window.  This can
        greatly reduce the resources required to run this script at the cost of
        quality. Currently a heuristic is used to select the "highest quality"
        subset of images.
        '''))

    queue_name = scfg.Value('time-ave-queue', help='overwrite the default queue name', group='cmd-queue')


def _find_valid_regions():
    import geowatch
    dvc_data_dpath = geowatch.find_dvc_dpath(tags='phase2_data', hardware='auto')
    coco_fpaths = list((dvc_data_dpath / 'Drop6').glob('imgonly*.kwcoco*'))
    import kwcoco
    dsets = list(kwcoco.CocoDataset.coerce_multiple(coco_fpaths, workers='avail'))

    for dset in dsets:
        if dset.n_images > 0:
            print('- ' + ub.Path(dset.fpath).name.split('-')[1].split('.')[0])


def main(cmdline=1, **kwargs):
    """
    Example:
        >>> # xdoctest: +SKIP
        >>> cmdline = 0
        >>> kwargs = dict(
        >>> )
        >>> main(cmdline=cmdline, **kwargs)
    """
    config = PrepareTimeAverages.cli(cmdline=cmdline, data=kwargs, strict=True)
    import rich
    from kwutil.partial_format import subtemplate
    rich.print('config = ' + ub.urepr(config, nl=1))
    assert config.output_bundle_dpath is not None
    assert config.input_bundle_dpath is not None
    # import geowatch
    # dvc_data_dpath = geowatch.find_dvc_dpath(tags='phase2_data', hardware='auto')

    if config.regions == 'all':
        all_regions = [p.name.split('.')[0] for p in (ub.Path(config.true_region_dpath)).ls()]
        chosen_regions = all_regions
    elif config.regions == 'all_tne':
        all_regions = [p.name.split('.')[0] for p in (ub.Path(config.true_region_dpath)).ls()]
        tne_regions = [r for r in all_regions if r.split('_')[1].startswith('R')]
        chosen_regions = tne_regions
    else:
        from kwutil.util_yaml import Yaml
        chosen_regions = Yaml.coerce(config.regions)

    from geowatch.mlops.pipeline_nodes import ProcessNode

    rich.print('chosen_regions = {}'.format(ub.urepr(chosen_regions, nl=1)))

    # time_duration = '1year'
    # time_duration = '3months'
    # all_regions = [
    #     'KR_R001',
    #     'KR_R002',
    #     'NZ_R001',
    #     'CH_R001',
    #     'BR_R001',
    #     'BR_R002',
    #     'BH_R001',
    #     'AE_R001',
    # ]

    queue = config.create_queue()

    # Need these for landcover
    other_s2_bands = '|coastal|cirrus|B05|B06|B07|B8A|B09'

    def submit_job_step(node, depends=None):
        if config.skip_existing and node.outputs_exist:
            job = None
        else:
            node.cache = config.cache
            job = queue.submit(node.final_command(), depends=depends, name=node.name)
        return job

    for region in chosen_regions:

        fmtdict = dict(
            # DVC_DATA_DPATH=dvc_data_dpath,
            INPUT_BUNDLE_DPATH=config.input_bundle_dpath,
            OUTPUT_BUNDLE_DPATH=config.output_bundle_dpath,
            REGION=region,
            # SUFFIX='MeanYear',
            # TIME_DURATION='3months',
            # SUFFIX='Mean3Month10GSD',
            TRUE_SITE_DPATH=config.true_site_dpath,
            TRUE_REGION_DPATH=config.true_region_dpath,
            CHANNELS='red|green|blue|nir|swir16|swir22|pan' + other_s2_bands,
            remove_seasons_str=None if not config.remove_seasons else ','.join(config.remove_seasons),
            mask_low_quality=config.mask_low_quality,
        )
        fmtdict.update(config)

        input_bundle_dpath = ub.Path(config.input_bundle_dpath)
        output_bundle_dpath = ub.Path(config.output_bundle_dpath)

        INPUT_KWCOCO_FPATH = input_bundle_dpath / region / f'imgonly-{region}-rawbands.kwcoco.zip'
        TAVE_KWCOCO_FPATH = output_bundle_dpath / region / f'_unfielded_imgonly-{region}-rawbands.kwcoco.zip'
        FIELDED_KWCOCO_FPATH = output_bundle_dpath / region / f'imgonly-{region}-rawbands.kwcoco.zip'
        FINAL_KWCOCO_FPATH = output_bundle_dpath / region / f'imganns-{region}-rawbands.kwcoco.zip'

        fmtdict['INPUT_KWCOCO_FPATH'] = INPUT_KWCOCO_FPATH
        fmtdict['TAVE_KWCOCO_FPATH'] = TAVE_KWCOCO_FPATH
        fmtdict['FIELDED_KWCOCO_FPATH'] = FIELDED_KWCOCO_FPATH
        fmtdict['FINAL_KWCOCO_FPATH'] = FINAL_KWCOCO_FPATH

        code = subtemplate(ub.codeblock(
            r'''
            python -m geowatch.cli.coco_time_combine \
                --kwcoco_fpath="$INPUT_KWCOCO_FPATH" \
                --output_kwcoco_fpath="$TAVE_KWCOCO_FPATH" \
                --channels="$CHANNELS" \
                --resolution="$resolution" \
                --time_window=$time_window \
                --remove_seasons=$remove_seasons_str \
                --merge_method=$merge_method \
                --spatial_tile_size=$spatial_tile_size \
                --mask_low_quality=$mask_low_quality \
                --max_images_per_group=$max_images_per_group \
                --start_time=2010-03-01 \
                --assets_dname="raw_bands" \
                --workers=$combine_workers
            '''), fmtdict)
        node = ProcessNode(
            name=f'combine-time-{region}',
            command=code,
            in_paths={'kwcoco_fpath': subtemplate('$INPUT_KWCOCO_FPATH', fmtdict)},
            out_paths={'output_kwcoco_fpath': subtemplate('$TAVE_KWCOCO_FPATH', fmtdict)},
            _no_outarg=True,
            _no_inarg=True,
        )
        combine_job = submit_job_step(node)

        code = subtemplate(ub.codeblock(
            r'''
            python -m geowatch add_fields \
                --src $TAVE_KWCOCO_FPATH \
                --dst $FIELDED_KWCOCO_FPATH
            '''), fmtdict)
        node = ProcessNode(
            name=f'add-fields-{region}',
            command=code,
            in_paths={'src': subtemplate('$TAVE_KWCOCO_FPATH', fmtdict)},
            out_paths={'dst': subtemplate('$FIELDED_KWCOCO_FPATH', fmtdict)},
            _no_outarg=True,
            _no_inarg=True,
        )
        field_job = submit_job_step(node, depends=[combine_job])

        if config.reproject:
            code = subtemplate(ub.codeblock(
                r'''
                python -m geowatch reproject \
                    --src $FIELDED_KWCOCO_FPATH \
                    --dst $FINAL_KWCOCO_FPATH \
                    --status_to_catname="positive_excluded: positive" \
                    --regions="$TRUE_REGION_DPATH/${REGION}.geojson" \
                    --sites="$TRUE_SITE_DPATH/${REGION}_*.geojson"
                '''), fmtdict)
            node = ProcessNode(
                command=code,
                name=f'reproject-ann-{region}',
                in_paths={'src': subtemplate('$FIELDED_KWCOCO_FPATH', fmtdict)},
                out_paths={'dst': subtemplate('$FINAL_KWCOCO_FPATH', fmtdict)},
                _no_outarg=True,
                _no_inarg=True,
            )
            field_job = submit_job_step(node, depends=[field_job])

    config.run_queue(queue)


SUMMER_CONFIG = """

DVC_DATA_DPATH=$(geowatch_dvc --tags='phase2_data' --hardware=hdd)
SSD_DVC_DATA_DPATH=$(geowatch_dvc --tags='phase2_data' --hardware=ssd)
python -m geowatch.cli.queue_cli.prepare_time_combined_dataset \
    --regions="[
            # T&E Regions
            AE_R001, BH_R001, BR_R001, BR_R002, BR_R004, BR_R005, CH_R001,
            KR_R001,
            KR_R002, LT_R001, NZ_R001, US_R001, US_R004, US_R005,
            US_R006, US_R007,
            # iMerit Regions
            AE_C001,
            AE_C002,
            AE_C003, PE_C001, QA_C001, SA_C005, US_C000, US_C010,
            US_C011, US_C012,
    ]" \
    --input_bundle_dpath=$SSD_DVC_DATA_DPATH/Drop6 \
    --output_bundle_dpath=$DVC_DATA_DPATH/Drop6-NoWinterMedian10GSD \
    --true_site_dpath=$DVC_DATA_DPATH/annotations/drop6_hard_v1/site_models \
    --true_region_dpath=$DVC_DATA_DPATH/annotations/drop6_hard_v1/region_models \
    --spatial_tile_size=256 \
    --merge_method=median \
    --remove_seasons=winter \
    --tmux_workers=2 \
    --time_window=1y \
    --combine_workers=4 \
    --resolution=10GSD \
    --backend=tmux \
    --run=1

# Drop 6
export CUDA_VISIBLE_DEVICES="0,1"
DVC_DATA_DPATH=$(geowatch_dvc --tags='phase2_data' --hardware=auto)
DVC_EXPT_DPATH=$(geowatch_dvc --tags='phase2_expt' --hardware='auto')
BUNDLE_DPATH=$DVC_DATA_DPATH/Drop6-NoWinterMedian10GSD
python -m geowatch.cli.queue_cli.prepare_teamfeats \
    --base_fpath "$BUNDLE_DPATH"/imganns-*[0-9].kwcoco.zip \
    --expt_dvc_dpath="$DVC_EXPT_DPATH" \
    --with_landcover=1 \
    --with_invariants2=1 \
    --with_materials=0 \
    --with_depth=0 \
    --with_cold=0 \
    --skip_existing=1 \
    --assets_dname=teamfeats \
    --gres=0,1 --tmux_workers=4 --backend=tmux --run=1

# DVC_DATA_DPATH=$(geowatch_dvc --tags='phase2_data' --hardware=auto)
# python -m geowatch.cli.queue_cli.prepare_splits \
#     --base_fpath=$DVC_DATA_DPATH/Drop6-NoWinterMedian10GSD/combo_imganns*_L*.kwcoco.zip \
#     --constructive_mode=True \
#     --suffix=L \
#     --backend=tmux --workers=6 \
#     --run=1

DVC_DATA_DPATH=$(geowatch_dvc --tags='phase2_data' --hardware=auto)
python -m geowatch.cli.queue_cli.prepare_splits \
    --base_fpath=$DVC_DATA_DPATH/Drop6-NoWinterMedian10GSD/combo_imganns*_I2L*.kwcoco.zip \
    --constructive_mode=True \
    --suffix=I2L \
    --backend=tmux --tmux_workers=6 \
    --run=1

# DVC_DATA_DPATH=$(geowatch_dvc --tags='phase2_data' --hardware=auto)
# python -m geowatch.cli.queue_cli.prepare_splits \
#     --base_fpath=$DVC_DATA_DPATH/Drop6-NoWinterMedian10GSD/imganns-*.kwcoco.zip \
#     --constructive_mode=True \
#     --suffix=rawbands \
#     --backend=tmux --tmux_workers=6 \
#     --run=1

DVC_DATA_DPATH=$(geowatch_dvc --tags='phase2_data' --hardware=auto)
DVC_EXPT_DPATH=$(geowatch_dvc --tags='phase2_expt' --hardware='auto')
TRUE_SITE_DPATH=$DVC_DATA_DPATH/annotations/drop6_hard_v1/site_models
OUTPUT_BUNDLE_DPATH=$DVC_DATA_DPATH/Drop6-NoWinterMedian10GSD
geowatch reproject \
    --src $DVC_DATA_DPATH/Drop6-NoWinterMedian10GSD/data_vali_I2L_split6.kwcoco.zip \
    --inplace=True \
    --status_to_catname="positive_excluded: positive" \
    --site_models=$TRUE_SITE_DPATH
geowatch reproject \
    --src $DVC_DATA_DPATH/Drop6-NoWinterMedian10GSD/data_train_I2L_split6.kwcoco.zip \
    --inplace=True \
    --status_to_catname="positive_excluded: positive" \
    --site_models=$TRUE_SITE_DPATH

"""


if __name__ == '__main__':
    r"""

    CommandLine:

        DVC_DATA_DPATH=$(geowatch_dvc --tags='phase2_data' --hardware=auto)
        python geowatch.cli.queue_cli.prepare_time_combined_dataset.py \
            --regions="[
                    # T&E Regions
                    AE_R001, BH_R001, BR_R001, BR_R002, BR_R004, BR_R005, CH_R001,
                    KR_R001, KR_R002, LT_R001, NZ_R001, US_R001, US_R004, US_R005,
                    US_R006, US_R007,
                    # iMerit Regions
                    AE_C001,
                    AE_C002,
                    AE_C003, PE_C001, QA_C001, SA_C005, US_C000, US_C010,
                    US_C011, US_C012,
            ]" \
            --input_bundle_dpath=$DVC_DATA_DPATH/Drop6 \
            --output_bundle_dpath=$DVC_DATA_DPATH/Drop6-MeanYear10GSD-V2 \
            --true_site_dpath=$DVC_DATA_DPATH/annotations/drop6_hard_v1/site_models \
            --true_region_dpath=$DVC_DATA_DPATH/annotations/drop6_hard_v1/region_models \
            --backend=tmux \
            --tmux_workers=4 \
            --combine_workers=2 \
            --resolution=10GSD \
            --run=1

        DVC_DATA_DPATH=$(geowatch_dvc --tags='phase2_data' --hardware=auto)
        python -m geowatch.cli.queue_cli.prepare_splits \
            --base_fpath=$DVC_DATA_DPATH/Drop6-MeanYear10GSD/imganns-*.kwcoco.zip \
            --constructive_mode=True \
            --suffix=rawbands \
            --backend=tmux --tmux_workers=6 \
            --run=1

        geowatch visualize /home/joncrall/remote/toothbrush/data/dvc-repos/smart_data_dvc-ssd/Drop6-MeanYear10GSD/imganns-NZ_R001.kwcoco.zip --smart

        # Drop 6
        export CUDA_VISIBLE_DEVICES="0,1"
        DVC_DATA_DPATH=$(geowatch_dvc --tags='phase2_data' --hardware=auto)
        DVC_EXPT_DPATH=$(geowatch_dvc --tags='phase2_expt' --hardware='auto')
        BUNDLE_DPATH=$DVC_DATA_DPATH/Drop6-MeanYear10GSD-V2
        python -m geowatch.cli.queue_cli.prepare_teamfeats \
            --base_fpath "$BUNDLE_DPATH"/imganns-*[0-9].kwcoco.zip \
            --expt_dvc_dpath="$DVC_EXPT_DPATH" \
            --with_landcover=1 \
            --with_invariants2=1 \
            --with_materials=0 \
            --with_depth=0 \
            --with_cold=0 \
            --skip_existing=1 \
            --assets_dname=teamfeats \
            --gres=0,1 --tmux_workers=4 --backend=tmux --run=0

        DVC_DATA_DPATH=$(geowatch_dvc --tags='phase2_data' --hardware=auto)
        python -m geowatch.cli.queue_cli.prepare_splits \
            --base_fpath=$DVC_DATA_DPATH/Drop6-MeanYear10GSD-V2/combo_imganns*_I2L*.kwcoco.zip \
            --constructive_mode=True \
            --suffix=I2L \
            --backend=tmux --tmux_workers=6 \
            --run=1

        DVC_DATA_DPATH=$(geowatch_dvc --tags='phase2_data' --hardware=auto)
        DVC_EXPT_DPATH=$(geowatch_dvc --tags='phase2_expt' --hardware='auto')
        TRUE_SITE_DPATH=$DVC_DATA_DPATH/annotations/drop6_hard_v1/site_models
        OUTPUT_BUNDLE_DPATH=$DVC_DATA_DPATH/Drop6-MeanYear10GSD-V2
        python -m geowatch reproject \
            --src data_vali_I2L_split6.kwcoco.zip \
            --dst data_vali_I2L_split6.kwcoco.zip \
            --site_models=$TRUE_SITE_DPATH

        DVC_DATA_DPATH=$(geowatch_dvc --tags='phase2_data' --hardware=auto)
        DVC_EXPT_DPATH=$(geowatch_dvc --tags='phase2_expt' --hardware='auto')
        TRUE_SITE_DPATH=$DVC_DATA_DPATH/annotations/drop6_hard_v1/site_models
        OUTPUT_BUNDLE_DPATH=$DVC_DATA_DPATH/Drop6-MeanYear10GSD-V2
        python -m geowatch reproject \
            --src data_train_I2L_split6.kwcoco.zip \
            --dst data_train_I2L_split6.kwcoco.zip \
            --site_models=$TRUE_SITE_DPATH
    """
    main()
