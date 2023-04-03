#!/usr/bin/env python3
"""
SeeAlso:
    ~/code/watch/watch/cli/coco_time_combine.py
"""
import scriptconfig as scfg
import ubelt as ub


class _CMDQueueBoilerplateConfig(scfg.DataConfig):
    """
    A helper to carry around the common boilerplate for cmd-queue CLI's.
    The __default__ attribute should be used to update another config's
    __default__ attribute, and the static methods can be used to create /
    execute the queue.

    This or something like it may eventually be ported to cmdqueue itself.
    """

    backend = scfg.Value('tmux', help=('The cmd_queue backend. Can be tmux, slurm, or serial'), group='cmd-queue')

    print_commands = scfg.Value('auto', isflag=True, help='enable / disable rprint before exec', group='cmd-queue')

    print_queue = scfg.Value('auto', isflag=True, help='print the cmd queue DAG', group='cmd-queue')

    run = scfg.Value(False, isflag=True, help='if False, only prints the commands, otherwise executes them', group='cmd-queue')

    with_textual = scfg.Value('auto', isflag=True, help='setting for cmd-queue monitoring', group='cmd-queue')

    other_session_handler = scfg.Value('ask', help='for tmux backend only. How to handle conflicting sessions. Can be ask, kill, or ignore, or auto', group='cmd-queue')

    virtualenv_cmd = scfg.Value(None, type=str, help=ub.paragraph(
        '''
        Command to start the appropriate virtual environment if your bashrc
        does not start it by default.'''), group='cmd-queue')

    tmux_workers = scfg.Value(4, help='number of tmux workers in the queue for the tmux backend', group='cmd-queue')

    queue_name = scfg.Value(None, help='overwrite the default queue name', group='cmd-queue')

    @staticmethod
    def _create_queue(config):
        import cmd_queue
        queue = cmd_queue.Queue.create(
            backend=config.backend, size=config.tmux_workers,
            name=config.queue_name or 'combine-time-queue')
        if config.virtualenv_cmd:
            queue.add_header_command(config.virtualenv_cmd)
        return queue

    @staticmethod
    def _run_queue(config, queue):
        print_thresh = 30
        if config['print_commands'] == 'auto':
            if len(queue) < print_thresh:
                config['print_commands'] = 1
            else:
                print(f'More than {print_thresh} jobs, skip queue.print_commands. '
                      'If you want to see them explicitly specify print_commands=1')
                config['print_commands'] = 0

        if config['print_queue'] == 'auto':
            if len(queue) < print_thresh:
                config['print_queue'] = 1
            else:
                print(f'More than {print_thresh} jobs, skip queue.print_graph. '
                      'If you want to see them explicitly specify print_queue=1')
                config['print_queue'] = 0

        if config.print_commands:
            queue.print_commands()

        if config.print_queue:
            queue.print_graph()

        if config.run:
            queue.run(with_textual=config.with_textual,
                      other_session_handler=config.other_session_handler)


class PrepareTimeAverages(scfg.DataConfig):
    """
    Prepare a temporally averaged dataset on multiple regions
    """
    __fuzzy_hyphens__ = 1
    # src = scfg.Value(None, help='input')
    __default__ = ub.udict({}) | _CMDQueueBoilerplateConfig.__default__

    regions = scfg.Value('all', help='The regions to time average (this is not a robust implementation)')

    reproject = scfg.Value(True, isflag=True, help='Enable reprojection of annotations')

    resolution = '10GSD'

    integration_window = '1year'

    combine_workers = scfg.Value(4, help='number of workers per combine job')

    input_bundle_dpath = scfg.Value(None)
    output_bundle_dpath = scfg.Value(None)

    true_site_dpath = scfg.Value(None)
    true_region_dpath = scfg.Value(None)


def main(cmdline=1, **kwargs):
    """
    Example:
        >>> # xdoctest: +SKIP
        >>> cmdline = 0
        >>> kwargs = dict(
        >>> )
        >>> main(cmdline=cmdline, **kwargs)
    """
    from watch.utils.partial_format import subtemplate
    config = PrepareTimeAverages.cli(cmdline=cmdline, data=kwargs, strict=True)
    print('config = ' + ub.urepr(dict(config), nl=1))
    assert config.output_bundle_dpath is not None
    assert config.input_bundle_dpath is not None
    # import watch
    # dvc_data_dpath = watch.find_dvc_dpath(tags='phase2_data', hardware='auto')

    if config.regions == 'all':
        all_regions = [p.name.split('.')[0] for p in (ub.Path(config.true_region_dpath)).ls()]
    else:
        from watch.utils.util_yaml import Yaml
        all_regions = Yaml.coerce(config.regions)

    # time_duration = '1year'
    # time_duration = '3months'
    all_regions = [
        'KR_R001',
        'KR_R002',
        'NZ_R001',
        'CH_R001',
        'BR_R001',
        'BR_R002',
        'BH_R001',
        'AE_R001',
    ]

    queue = _CMDQueueBoilerplateConfig._create_queue(config)

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
            CHANNELS='red|green|blue|nir|swir16|swir22|pan',
        )

        code = subtemplate(ub.codeblock(
            r'''
            python -m watch.cli.coco_time_combine \
                --kwcoco_fpath="$INPUT_BUNDLE_DPATH/imgonly-${REGION}.kwcoco.json" \
                --output_kwcoco_fpath="$OUTPUT_BUNDLE_DPATH/imgonly-${REGION}.kwcoco.zip" \
                --channels="$CHANNELS" \
                --resolution="$RESOLUTION" \
                --temporal_window_duration=$TIME_DURATION \
                --merge_method=mean \
                --workers=$WORKERS
            '''), fmtdict)
        combine_job = queue.submit(code, name=f'combine-time-{region}')

        if config.reproject:
            code = subtemplate(
                r'''
                python -m watch reproject_annotations \
                    --src $OUTPUT_BUNDLE_DPATH/imgonly-${REGION}.kwcoco.zip \
                    --dst $OUTPUT_BUNDLE_DPATH/imganns-${REGION}.kwcoco.zip \
                    --site_models="$TRUE_SITE_DPATH/${REGION}_*.geojson"
                ''', fmtdict)
            queue.submit(code, depends=[combine_job], name=f'reproject-ann-{region}')

    _CMDQueueBoilerplateConfig._run_queue(config, queue)


if __name__ == '__main__':
    r"""

    CommandLine:

        DVC_DATA_DPATH=$(smartwatch_dvc --tags='phase2_data' --hardware=auto)
        python ~/code/watch/dev/poc/prepare_time_combined_dataset.py \
            --regions=all \
            --input_bundle_dpath=$DVC_DATA_DPATH/Drop6 \
            --output_bundle_dpath=$DVC_DATA_DPATH/Drop6-MeanYear10GSD \
            --true_site_dpath=$DVC_DATA_DPATH/annotations/drop6/site_models \
            --true_region_dpath=$DVC_DATA_DPATH/annotations/drop6/region_models \
            --backend=tmux \
            --tmux_workers=4 \
            --resolution=10GSD \
            --run=1

        smartwatch visualize /home/joncrall/remote/toothbrush/data/dvc-repos/smart_data_dvc-ssd/Drop6-MeanYear10GSD/imganns-NZ_R001.kwcoco.zip --smart

        # Drop 6
        export CUDA_VISIBLE_DEVICES="0,1"
        DVC_DATA_DPATH=$(smartwatch_dvc --tags='phase2_data' --hardware=auto)
        DVC_EXPT_DPATH=$(smartwatch_dvc --tags='phase2_expt' --hardware='auto')
        BUNDLE_DPATH=$DVC_DATA_DPATH/Drop6-MeanYear10GSD
        python -m watch.cli.prepare_teamfeats \
            --base_fpath "$BUNDLE_DPATH"/imganns-*.kwcoco.zip \
            --expt_dpath="$DVC_EXPT_DPATH" \
            --with_landcover2=1 \
            --with_landcover=0 \
            --with_materials=0 \
            --with_invariants=1 \
            --with_depth=0 \
            --with_cold=0 \
            --do_splits=0 \
            --skip_existing=1 \
            --gres=0,1 --workers=4 --backend=serial --run=1

        DVC_DATA_DPATH=$(smartwatch_dvc --tags='phase2_data' --hardware=auto)
        python -m watch.cli.prepare_splits \
            --base_fpath=$DVC_DATA_DPATH/Drop6-MeanYear10GSD/imganns-*.kwcoco.zip \
            --constructive_mode=True \
            --suffix=rawbands \
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



Ignore:
    import kwplot
    kwplot.plt.ion()
    import kwcoco
    dset = kwcoco.CocoDataset('/home/joncrall/remote/toothbrush/data/dvc-repos/smart_data_dvc-ssd/Drop6_MeanYear/imgonly-KR_R001.kwcoco.zip')

    from watch.utils import util_gdal

    import kwimage
    video = dset.videos().objs[0]
    video_box = kwimage.Box.from_dsize((video['width'], video['height']))

    for coco_img in coco_images:
        print(f'coco_img.shape={coco_img.dsize}')
        img = ub.udict(coco_img.img)
        img = img - {'has_predictions'}
        print('coco_img.img = {}'.format(ub.urepr(img, nl=-1)))

        for asset in coco_img.assets:
            fpath = ub.Path(coco_img.bundle_dpath) / asset['file_name']
            info = util_gdal.GdalOpen(fpath).info()
            assert len(info['bands']) == coco_img.channels.numel()
            ...




python -m watch.cli.coco_time_combine \
    --kwcoco_fpath="/home/joncrall/remote/toothbrush/data/dvc-repos/smart_data_dvc-ssd/Drop6/imgonly-NZ_R001.kwcoco.json" \
    --output_kwcoco_fpath="/home/joncrall/remote/toothbrush/data/dvc-repos/smart_data_dvc-ssd/Drop6-MeanYear10GSD/imgonly-NZ_R001.kwcoco.json" \
    --channels="red|green|blue|nir|swir16|swir22|pan" \
    --resolution="10GSD" \
    --temporal_window_duration=1year \
    --merge_method=mean \
    --workers=4


python -m watch.cli.coco_add_watch_fields \
    --src=/home/joncrall/remote/toothbrush/data/dvc-repos/smart_data_dvc-ssd/Drop6-MeanYear10GSD/imgonly-NZ_R001.kwcoco.json \
    --dst=/home/joncrall/remote/toothbrush/data/dvc-repos/smart_data_dvc-ssd/Drop6-MeanYear10GSD/imgonly-NZ_R001_v2.kwcoco.json \
    --overwrite=True \
    --target_gsd=10

#
### Command 2 / 2 - reproject-ann-NZ_R001
python -m watch reproject_annotations \
    --src /home/joncrall/remote/toothbrush/data/dvc-repos/smart_data_dvc-ssd/Drop6-MeanYear10GSD/imgonly-NZ_R001_v2.kwcoco.json \
    --dst /home/joncrall/remote/toothbrush/data/dvc-repos/smart_data_dvc-ssd/Drop6-MeanYear10GSD/imganns-NZ_R001_v2.kwcoco.json \
    --site_models="/home/joncrall/remote/toothbrush/data/dvc-repos/smart_data_dvc-ssd/annotations/drop6/site_models/*.geojson"


smartwatch visualize /home/joncrall/remote/toothbrush/data/dvc-repos/smart_data_dvc-ssd/Drop6-MeanYear10GSD/imganns-NZ_R001_v2.kwcoco.json --smart




python -m watch.cli.coco_add_watch_fields \
    --src=/home/joncrall/remote/toothbrush/data/dvc-repos/smart_data_dvc-ssd/Drop6-MeanYear10GSD/data_vali_rawbands_split6.kwcoco.zip \
    --dst=/home/joncrall/remote/toothbrush/data/dvc-repos/smart_data_dvc-ssd/Drop6-MeanYear10GSD/data_vali_rawbands_split6.kwcoco.zip \
    --overwrite=True \
    --target_gsd=10

python -m watch.cli.coco_add_watch_fields \
    --src=/home/joncrall/remote/toothbrush/data/dvc-repos/smart_data_dvc-ssd/Drop6-MeanYear10GSD/data_train_rawbands_split6.kwcoco.zip \
    --dst=/home/joncrall/remote/toothbrush/data/dvc-repos/smart_data_dvc-ssd/Drop6-MeanYear10GSD/data_train_rawbands_split6.kwcoco.zip \
    --overwrite=True \
    --target_gsd=10

### Command 2 / 2 - reproject-ann-NZ_R001
python -m watch reproject_annotations \
    --src /home/joncrall/remote/toothbrush/data/dvc-repos/smart_data_dvc-ssd/Drop6-MeanYear10GSD/data_train_rawbands_split6.kwcoco.zip \
    --dst /home/joncrall/remote/toothbrush/data/dvc-repos/smart_data_dvc-ssd/Drop6-MeanYear10GSD/data_train_rawbands_split6.kwcoco.zip \
    --site_models="/home/joncrall/remote/toothbrush/data/dvc-repos/smart_data_dvc-ssd/annotations/drop6/site_models/*.geojson"

### Command 2 / 2 - reproject-ann-NZ_R001
python -m watch reproject_annotations \
    --src /home/joncrall/remote/toothbrush/data/dvc-repos/smart_data_dvc-ssd/Drop6-MeanYear10GSD/data_vali_rawbands_split6.kwcoco.zip \
    --dst /home/joncrall/remote/toothbrush/data/dvc-repos/smart_data_dvc-ssd/Drop6-MeanYear10GSD/data_vali_rawbands_split6.kwcoco.zip \
    --site_models="/home/joncrall/remote/toothbrush/data/dvc-repos/smart_data_dvc-ssd/annotations/drop6/site_models/*.geojson"




### Command 2 / 2 - reproject-ann-BR_R002
python -m watch reproject_annotations \
    --src /home/joncrall/remote/toothbrush/data/dvc-repos/smart_data_dvc-ssd/Drop6/imgonly-BR_R002.kwcoco.json \
    --dst /home/joncrall/remote/toothbrush/data/dvc-repos/smart_data_dvc-ssd/Drop6/TEST-BR_R002.kwcoco.json \
    --site_models="/home/joncrall/remote/toothbrush/data/dvc-repos/smart_data_dvc-ssd/annotations/drop6/site_models/BR_R002*.geojson"

smartwatch visualize /home/joncrall/remote/toothbrush/data/dvc-repos/smart_data_dvc-ssd/Drop6/TEST-BR_R002.kwcoco.json --smart



python -m watch.cli.coco_time_combine \
    --kwcoco_fpath="/home/joncrall/remote/toothbrush/data/dvc-repos/smart_data_dvc-ssd/Drop6/imgonly-BR_R002.kwcoco.json" \
    --output_kwcoco_fpath="/home/joncrall/remote/toothbrush/data/dvc-repos/smart_data_dvc-ssd/Drop6-MeanYear10GSD/imgonly-BR_R002.kwcoco.zip" \
    --channels="red|green|blue|nir|swir16|swir22|pan" \
    --resolution="10GSD" \
    --temporal_window_duration=1year \
    --merge_method=mean \
    --workers=4


#
### Command 2 / 2 - reproject-ann-BR_R002
python -m watch reproject_annotations \
    --src /home/joncrall/remote/toothbrush/data/dvc-repos/smart_data_dvc-ssd/Drop6-MeanYear10GSD/imgonly-BR_R002.kwcoco.zip \
    --dst /home/joncrall/remote/toothbrush/data/dvc-repos/smart_data_dvc-ssd/Drop6-MeanYear10GSD/imganns-BR_R002.kwcoco.zip \
    --site_models="/home/joncrall/remote/toothbrush/data/dvc-repos/smart_data_dvc-ssd/annotations/drop6/site_models/BR_R002*.geojson"


smartwatch visualize /home/joncrall/remote/toothbrush/data/dvc-repos/smart_data_dvc-ssd/Drop6-MeanYear10GSD/imganns-BR_R002.kwcoco.zip --smart



smartwatch visualize /home/joncrall/remote/toothbrush/data/dvc-repos/smart_data_dvc-ssd/Drop6-MeanYear10GSD/imganns-CH_R001.kwcoco.zip --smart


smartwatch reproject_annotations \
    --src /home/joncrall/remote/toothbrush/data/dvc-repos/smart_data_dvc-ssd/Drop6-MeanYear10GSD/imgonly-CH_R001.kwcoco.zip \
    --dst /home/joncrall/remote/toothbrush/data/dvc-repos/smart_data_dvc-ssd/Drop6-MeanYear10GSD/imganns-CH_R001.kwcoco.zip \
    --site_models="/home/joncrall/remote/toothbrush/data/dvc-repos/smart_data_dvc-hdd/annotations/drop6/site_models/*.geojson"

smartwatch visualize /home/joncrall/remote/toothbrush/data/dvc-repos/smart_data_dvc-ssd/Drop6-MeanYear10GSD/imganns-CH_R001.kwcoco.zip --smart
smartwatch visualize /home/joncrall/remote/toothbrush/data/dvc-repos/smart_data_dvc-ssd/Drop6-MeanYear10GSD/imganns-CH_R0012.kwcoco.zip --smart

"""