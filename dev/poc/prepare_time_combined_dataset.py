#!/usr/bin/env python3
"""
SeeAlso:
    ~/code/watch/watch/cli/coco_time_combine.py
    ~/code/watch/watch/cli/prepare_teamfeats.py
"""
import scriptconfig as scfg
import ubelt as ub
from cmd_queue.cli_boilerplate import CMDQueueConfig


class PrepareTimeAverages(CMDQueueConfig):
    """
    Prepare a temporally averaged dataset on multiple regions
    """
    regions = scfg.Value('all', type=str, help='The regions to time average (this is not a robust implementation)')

    reproject = scfg.Value(True, isflag=True, help='Enable reprojection of annotations')

    resolution = '10GSD'

    integration_window = '1year'

    combine_workers = scfg.Value(4, help='number of workers per combine job')

    input_bundle_dpath = scfg.Value(None)
    output_bundle_dpath = scfg.Value(None)

    true_site_dpath = scfg.Value(None)
    true_region_dpath = scfg.Value(None)


def _find_valid_regions():
    import watch
    dvc_data_dpath = watch.find_dvc_dpath(tags='phase2_data', hardware='auto')
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
    from watch.utils.partial_format import subtemplate
    rich.print('config = ' + ub.urepr(config, nl=1))
    assert config.output_bundle_dpath is not None
    assert config.input_bundle_dpath is not None
    # import watch
    # dvc_data_dpath = watch.find_dvc_dpath(tags='phase2_data', hardware='auto')

    if config.regions == 'all':
        all_regions = [p.name.split('.')[0] for p in (ub.Path(config.true_region_dpath)).ls()]
    else:
        from watch.utils.util_yaml import Yaml
        all_regions = Yaml.coerce(config.regions)

    rich.print('all_regions = {}'.format(ub.urepr(all_regions, nl=1)))

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

    for region in all_regions:

        fmtdict = dict(
            # DVC_DATA_DPATH=dvc_data_dpath,
            OUTPUT_BUNDLE_DPATH=config.output_bundle_dpath,
            INPUT_BUNDLE_DPATH=config.input_bundle_dpath,
            REGION=region,
            TIME_DURATION=config.integration_window,
            # SUFFIX='MeanYear',
            # TIME_DURATION='3months',
            # SUFFIX='Mean3Month10GSD',
            RESOLUTION=config.resolution,
            WORKERS=config.combine_workers,
            TRUE_SITE_DPATH=config.true_site_dpath,
            CHANNELS='red|green|blue|nir|swir16|swir22|pan' + other_s2_bands,
        )

        code = subtemplate(ub.codeblock(
            r'''
            python -m watch.cli.coco_time_combine \
                --kwcoco_fpath="$INPUT_BUNDLE_DPATH/imgonly-${REGION}.kwcoco.json" \
                --output_kwcoco_fpath="$OUTPUT_BUNDLE_DPATH/imgonly-${REGION}.kwcoco.zip" \
                --channels="$CHANNELS" \
                --resolution="$RESOLUTION" \
                --time_window=$TIME_DURATION \
                --start_time=2010-01-01 \
                --merge_method=mean \
                --assets_dname="raw_bands" \
                --workers=$WORKERS
            '''), fmtdict)
        combine_job = queue.submit(code, name=f'combine-time-{region}')

        if config.reproject:
            code = subtemplate(
                r'''
                python -m watch add_fields \
                    --src $OUTPUT_BUNDLE_DPATH/imgonly-${REGION}.kwcoco.zip \
                    --dst $OUTPUT_BUNDLE_DPATH/imgonly-${REGION}.kwcoco.zip \
                ''', fmtdict)
            field_job = queue.submit(code, depends=[combine_job], name=f'add-fields-{region}')
            code = subtemplate(
                r'''
                python -m watch reproject \
                    --src $OUTPUT_BUNDLE_DPATH/imgonly-${REGION}.kwcoco.zip \
                    --dst $OUTPUT_BUNDLE_DPATH/imganns-${REGION}.kwcoco.zip \
                    --site_models="$TRUE_SITE_DPATH/${REGION}_*.geojson"
                ''', fmtdict)
            queue.submit(code, depends=[field_job], name=f'reproject-ann-{region}')

    config.run_queue(queue)


if __name__ == '__main__':
    r"""

    CommandLine:

        DVC_DATA_DPATH=$(smartwatch_dvc --tags='phase2_data' --hardware=auto)
        python ~/code/watch/dev/poc/prepare_time_combined_dataset.py \
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

        DVC_DATA_DPATH=$(smartwatch_dvc --tags='phase2_data' --hardware=auto)
        python -m watch.cli.prepare_splits \
            --base_fpath=$DVC_DATA_DPATH/Drop6-MeanYear10GSD/imganns-*.kwcoco.zip \
            --constructive_mode=True \
            --suffix=rawbands \
            --backend=tmux --tmux_workers=6 \
            --run=1

        smartwatch visualize /home/joncrall/remote/toothbrush/data/dvc-repos/smart_data_dvc-ssd/Drop6-MeanYear10GSD/imganns-NZ_R001.kwcoco.zip --smart

        # Drop 6
        export CUDA_VISIBLE_DEVICES="0,1"
        DVC_DATA_DPATH=$(smartwatch_dvc --tags='phase2_data' --hardware=auto)
        DVC_EXPT_DPATH=$(smartwatch_dvc --tags='phase2_expt' --hardware='auto')
        BUNDLE_DPATH=$DVC_DATA_DPATH/Drop6-MeanYear10GSD-V2
        python -m watch.cli.prepare_teamfeats \
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

        DVC_DATA_DPATH=$(smartwatch_dvc --tags='phase2_data' --hardware=auto)
        python -m watch.cli.prepare_splits \
            --base_fpath=$DVC_DATA_DPATH/Drop6-MeanYear10GSD-V2/combo_imganns*_I2L*.kwcoco.zip \
            --constructive_mode=True \
            --suffix=I2L \
            --backend=tmux --tmux_workers=6 \
            --run=1
    """
    main()


__notes__ = r"""

python -m watch.cli.coco_time_combine \
    --kwcoco_fpath="/home/joncrall/remote/toothbrush/data/dvc-repos/smart_data_dvc-ssd/Drop6/imgonly-KR_R001.kwcoco.json" \
    --output_kwcoco_fpath="/home/joncrall/remote/toothbrush/data/dvc-repos/smart_data_dvc-ssd/Drop6_MeanYear/imgonly-KR_R001.kwcoco.zip" \
    --channels="red|green|blue" \
    --resolution="10GSD" \
    --temporal_window_duration=1year \
    --merge_method=mean \
    --workers=4


python -m watch.cli.coco_time_combine \
    --kwcoco_fpath="/home/joncrall/remote/toothbrush/data/dvc-repos/smart_data_dvc-ssd/Drop6/imgonly-NZ_R001.kwcoco.json" \
    --output_kwcoco_fpath="/home/joncrall/remote/toothbrush/data/dvc-repos/smart_data_dvc-ssd/Drop6_MeanYear/imgonly-NZ_R001.kwcoco.zip" \
    --channels="red|green|blue" \
    --resolution="10GSD" \
    --temporal_window_duration=1year \
    --merge_method=mean \
    --workers=4


python -m watch.tasks.fusion.predict \
    --package_fpath=/home/joncrall/remote/toothbrush/data/dvc-repos/smart_expt_dvc/models/fusion/Drop4-BAS/packages/Drop4_TuneV323_BAS_30GSD_BGRNSH_V2/package_epoch0_step41.pt.pt \
    --test_dataset=/home/joncrall/remote/toothbrush/data/dvc-repos/smart_data_dvc-ssd/Drop6_MeanYear/imgonly-KR_R001.kwcoco.zip \
    --pred_dataset=/home/joncrall/remote/toothbrush/data/dvc-repos/smart_data_dvc-ssd/Drop6_MeanYear/pred-KR_R001.kwcoco.zip \
    --chip_overlap="0.3" \
    --chip_dims="auto" \
    --time_span="auto" \
    --time_sampling="auto" \
    --drop_unused_frames="True"  \
    --num_workers="0" \
    --devices="0," \
    --batch_size="1" \
    --with_saliency="True" \
    --with_class="False" \
    --with_change="False"


smartwatch stats /home/joncrall/remote/toothbrush/data/dvc-repos/smart_data_dvc-ssd/Drop6_MeanYear/imgonly-KR_R001.kwcoco.zip
smartwatch visualize /home/joncrall/remote/toothbrush/data/dvc-repos/smart_data_dvc-ssd/Drop6_MeanYear/imgonly-KR_R001.kwcoco.zip --smart
smartwatch visualize /home/joncrall/remote/toothbrush/data/dvc-repos/smart_data_dvc-ssd/Drop6_MeanYear/pred-KR_R001.kwcoco.zip --smart
"""
