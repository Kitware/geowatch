r"""
Main prediction script for cold


SeeAlso:

    ../../cli/prepare_teamfeats.py

    predict.py *

    prepare_kwcoco.py

    tile_processing_kwcoco.py

    export_cold_result_kwcoco.py

    assemble_cold_result_kwcoco.py

CommandLine:

    ##############
    ### SMALL TEST
    ##############

    DATA_DVC_DPATH=$(geowatch_dvc --tags=phase2_data --hardware="auto")
    EXPT_DVC_DPATH=$(geowatch_dvc --tags=phase2_expt --hardware="auto")

    mkdir -p $DATA_DVC_DPATH/Drop6-SMALL
    kwcoco subset \
        --src "$DATA_DVC_DPATH/Drop6/imgonly-KR_R001.kwcoco.json" \
        --dst "$DATA_DVC_DPATH/Drop6-SMALL/imgonly-KR_R001.kwcoco.json" \
        --select_images '(.sensor_coarse == "L8")'

    # Pull out a small selection of images just so we can test.
    python -c "if 1:
        import ubelt as ub
        import kwcoco
        dset = kwcoco.CocoDataset('$DATA_DVC_DPATH/Drop6-SMALL/imgonly-KR_R001.kwcoco.json')
        from kwutil import util_time
        images = dset.images()
        dates = list(map(util_time.coerce_datetime, images.lookup('date_captured')))
        flags = [d.year < 2017 for d in dates]
        chosen = images.compress(flags)
        sub = dset.subset(chosen)
        sub.fpath = dset.fpath
        sub.dump()
    "

    DATA_DVC_DPATH=$(geowatch_dvc --tags=phase2_data --hardware="auto")
    EXPT_DVC_DPATH=$(geowatch_dvc --tags=phase2_expt --hardware="auto")
    python -m geowatch.tasks.cold.predict \
        --coco_fpath="$DATA_DVC_DPATH/Drop6-SMALL/imgonly-KR_R001.kwcoco.json" \
        --out_dpath="$DATA_DVC_DPATH/Drop6-SMALL/_pycold" \
        --sensors='L8' \
        --resolution=30GSD \
        --mod_coco_fpath="$DATA_DVC_DPATH/Drop6-SMALL/_pycold/imgonly-KR_R001-cold.kwcoco.json" \
        --adj_cloud=False \
        --method='COLD' \
        --prob=0.99 \
        --conse=6 \
        --cm_interval=60 \
        --year_lowbound=None \
        --year_highbound=None \
        --coefs=cv \
        --coefs_bands=0,1,2,3,4,5 \
        --timestamp=False \
        --workermode='process' \
        --workers=16

    kwcoco reroot \
        --src="$DATA_DVC_DPATH"/Drop6-SMALL/_pycold/imgonly-KR_R001-cold.kwcoco.json \
        --dst="$DATA_DVC_DPATH"/Drop6-SMALL/_pycold/imgonly-KR_R001-cold.fixed.kwcoco.zip \
        --old_prefix="KR_R001" --new_prefix="../KR_R001"

    geowatch visualize \
        "$DATA_DVC_DPATH"/Drop6-SMALL/_pycold/imgonly-KR_R001-cold.fixed.kwcoco.zip \
        --channels="L8:(red|green|blue,red_COLD_cv|green_COLD_cv|blue_COLD_cv)" \
        --exclude_sensors="S2" \
        --smart=True --skip_aggressive=True

    ###################################################################################
    ### FULL REGION TEST: COLD FEATURES WITH HIGH TEMPORAL RESOLUTION (HTR) + L8/S2 ###
    ###################################################################################

    DATA_DVC_DPATH=$(geowatch_dvc --tags=phase2_data --hardware="auto")
    EXPT_DVC_DPATH=$(geowatch_dvc --tags=phase2_expt --hardware="auto")
    python -m geowatch.tasks.cold.predict \
        --coco_fpath="$DATA_DVC_DPATH/Aligned-Drop7/KR_R001/imgonly-KR_R001.kwcoco.zip" \
        --out_dpath="$DATA_DVC_DPATH/Aligned-Drop7/_pycold_L8S2_HTR" \
        --mod_coco_fpath="$DATA_DVC_DPATH/Aligned-Drop7/KR_R001/imgonly_KR_R001_cold-L8S2-HTR.kwcoco.zip" \
        --sensors='L8,S2' \
        --coefs=cv,rmse,a0,a1,b1,c1 \
        --prob=0.99 \
        --conse=8 \
        --coefs_bands=0,1,2,3,4,5 \
        --combine=False \
        --resolution='10GSD' \
        --workermode='process' \
        --workers=8

    ######################################################################
    ### FULL REGION TEST: TRANSFER COLD FEATURE FROM RAW TO COMBINED INPUT
    ######################################################################

    DATA_DVC_DPATH=$(geowatch_dvc --tags=phase2_data --hardware="auto")
    EXPT_DVC_DPATH=$(geowatch_dvc --tags=phase2_expt --hardware="auto")
    python -m geowatch.tasks.cold.transfer_features \
        --coco_fpath="$DATA_DVC_DPATH/Drop6/imgonly_KR_R001_cold-HTR.kwcoco.zip" \
        --combine_fpath="$DATA_DVC_DPATH/Drop6-MeanYear10GSD-V2/imgonly-KR_R001.kwcoco.zip" \
        --new_coco_fpath="$DATA_DVC_DPATH/Drop6-MeanYear10GSD-V2/imganns-KR_R001_uconn_cold.kwcoco.zip"

    kwcoco stats "$DATA_DVC_DPATH/Drop6-MeanYear10GSD-V2/imganns-KR_R001_uconn_cold.kwcoco.zip"
    geowatch stats "$DATA_DVC_DPATH/Drop6-MeanYear10GSD-V2/imganns-KR_R001_uconn_cold.kwcoco.zip"
    kwcoco validate "$DATA_DVC_DPATH/Drop6-MeanYear10GSD-V2/imganns-KR_R001_uconn_cold.kwcoco.zip"

    DATA_DVC_DPATH=$(geowatch_dvc --tags=phase2_data --hardware="auto")
    geowatch visualize \
        "$DATA_DVC_DPATH/Drop6-MeanYear10GSD-V2/imganns-KR_R001_uconn_cold.kwcoco.zip" \
        --channels="L8:(red|green|blue,red_COLD_a1|green_COLD_a1|blue_COLD_a1,red_COLD_cv|green_COLD_cv|blue_COLD_cv,red_COLD_rmse|green_COLD_rmse|blue_COLD_rmse)" \
        --exclude_sensors=WV,PD,S2 \
        --smart=True

    ########################
    ### MULTIPLE REGION TEST
    ########################
    DVC_DATA_DPATH=$(geowatch_dvc --tags='phase2_data' --hardware=auto)

    "$BUNDLE_DPATH"/imganns-*BR_[RC]*.kwcoco.zip \
    "$BUNDLE_DPATH"/imganns-*KR_[RC]*.kwcoco.zip \
    "$BUNDLE_DPATH"/imganns-*NZ_[RC]*.kwcoco.zip \
    "$BUNDLE_DPATH"/imganns-*US_[RC]*.kwcoco.zip \

    echo "$DVC_DATA_DPATH"
    BUNDLE_DPATH=$DVC_DATA_DPATH/Drop6
    python -m geowatch.cli.prepare_teamfeats \
        --base_fpath \
            "$BUNDLE_DPATH"/imganns-*AE_[RC]*.kwcoco.zip \
            "$BUNDLE_DPATH"/imganns-*BH_[RC]*.kwcoco.zip \
            "$BUNDLE_DPATH"/imganns-*CH_[RC]*.kwcoco.zip \
            "$BUNDLE_DPATH"/imganns-*LT_[RC]*.kwcoco.zip \
            "$BUNDLE_DPATH"/imganns-*NZ_[RC]*.kwcoco.zip \
            "$BUNDLE_DPATH"/imganns-*PE_[RC]*.kwcoco.zip \
            "$BUNDLE_DPATH"/imganns-*QA_[RC]*.kwcoco.zip \
            "$BUNDLE_DPATH"/imganns-*SA_[RC]*.kwcoco.zip \
            "$BUNDLE_DPATH"/imganns-*US_C*.kwcoco.zip \
        --with_cold=1 \
        --with_landcover=0 \
        --with_materials=0 \
        --with_invariants=0 \
        --with_depth=0 \
        --skip_existing=1 \
        --cold_workers=8 \
        --cold_workermode=thread \
        --tmux_workers=2 \
        --backend=tmux --run=0
"""
import scriptconfig as scfg
import ubelt as ub
import json
import logging
import os

logger = logging.getLogger(__name__)


try:
    from xdev import profile
except ImportError:
    profile = ub.identity


class ColdPredictConfig(scfg.DataConfig):
    """
    The docstring will be the description in the CLI help
    """

    coco_fpath = scfg.Value(None, position=1, help=ub.paragraph(
        '''
        a path to a file to input kwcoco file (to predict on)
        '''))

    mod_coco_fpath = scfg.Value(None, help=ub.paragraph(
        '''
        The modified output kwcoco file, which is a copy of the input
        kwcoco file enriched with COLD features.
        '''
    ), alias=['output_kwcoco'])

    combined_coco_fpath = scfg.Value(None, help=ub.paragraph(
        '''
        a path to a file to combined input kwcoco file (to merge with)
        '''))

    out_dpath = scfg.Value(None, help=ub.paragraph(
        '''
        output directory for the output. If unspecified uses the input kwcoco bundle
        '''))

    write_kwcoco = scfg.Value(True, help='writing kwcoco file based on COLD feature, Default is True')
    sensors = scfg.Value('L8', type=str, help='sensor type, default is "L8"')
    adj_cloud = scfg.Value(False, help='How to treat QA band, default is False: ignoring adj. cloud class')
    method = scfg.Value('COLD', choices=['COLD', 'HybridCOLD', 'OBCOLD'], help='type of cold algorithms')
    prob = scfg.Value(0.99, help='change probability of chi-distribution, e.g., 0.99')
    conse = scfg.Value(6, help='consecutive observation to confirm change, e.g., 6')
    cm_interval = scfg.Value(60, help='CM output inverval, e.g., 60')
    year_lowbound = scfg.Value(None, help='min year for saving geotiff, e.g., 2017')
    year_highbound = scfg.Value(None, help='max year for saving geotiff, e.g., 2022')
    coefs = scfg.Value(None, type=str, help="list of COLD coefficients for saving geotiff, e.g., a0,c1,a1,b1,a2,b2,a3,b3,cv,rmse")
    coefs_bands = scfg.Value(None, type=str, help='indicate the ba_nds for output coefs_bands, e.g., 0,1,2,3,4,5')
    timestamp = scfg.Value(False, help='True: exporting cold result by timestamp, False: exporting cold result by year, Default is False')
    combine = scfg.Value(False, help='for temporal combined mode, Default is False')
    track_emissions = scfg.Value(True, help='if True use codecarbon for emission tracking')
    resolution = scfg.Value('30GSD', help='if specified then data is processed at this resolution')
    exclude_first = scfg.Value(True, help='exclude first date of image from each sensor, Default is True')
    workers = scfg.Value(16, help='total number of workers')
    workermode = scfg.Value('process', help='Can be process, serial, or thread')


@profile
def cold_predict_main(cmdline=1, **kwargs):
    """
    Args:
        cmdline (int, optional): _description_. Defaults to 1.

    Ignore:
        python -m geowatch.tasks.cold.predict --help
        TEST_COLD=1 xdoctest -m geowatch.tasks.cold.predict cold_predict_main

     Example:
        >>> # xdoctest: +REQUIRES(env:TEST_COLD)
        >>> from geowatch.tasks.cold.predict import cold_predict_main
        >>> from geowatch.tasks.cold.predict import *
        >>> kwargs= dict(
        >>>    coco_fpath = ub.Path('/gpfs/scratchfs1/zhz18039/jws18003/new-repos/smart_data_dvc2/Drop6/imgonly-KR_R001.kwcoco.json'),
        >>>    out_dpath = ub.Path.appdir('/gpfs/scratchfs1/zhz18039/jws18003/new-repos/smart_data_dvc2/Drop6/_pycold_combine_V2'),
        >>>    write_kwcoco = False,
        >>>    sensors = 'L8',
        >>>    adj_cloud = False,
        >>>    method = 'COLD',
        >>>    prob = 0.99,
        >>>    conse = 6,
        >>>    cm_interval = 60,
        >>>    year_lowbound = None,
        >>>    year_highbound = None,
        >>>    coefs = 'cv,rmse,a0,a1,b1,c1',
        >>>    coefs_bands = '0,1,2,3,4,5',
        >>>    timestamp = False,
        >>>    combine = False,
        >>>    resolution = '10GSD',
        >>>    workermode = 'process',
        >>>    )
        >>> cmdline=0
        >>> cold_predict_main(cmdline, **kwargs)
    """
    from geowatch.tasks.cold import prepare_kwcoco
    from geowatch.tasks.cold import tile_processing_kwcoco
    from geowatch.tasks.cold import export_cold_result_kwcoco
    from geowatch.tasks.cold import assemble_cold_result_kwcoco

    config = ColdPredictConfig.cli(cmdline=cmdline, data=kwargs)
    import rich
    rich.print('config = {}'.format(ub.urepr(config, nl=1)))

    from geowatch.utils import process_context
    from kwutil import util_parallel
    from kwutil import util_progress
    from kwutil import util_json
    resolved_config = config.to_dict()
    resolved_config = util_json.ensure_json_serializable(resolved_config)

    proc_context = process_context.ProcessContext(
        name='geowatch.tasks.cold.predict',
        type='process',
        config=resolved_config,
        track_emissions=config['track_emissions'],
    )

    coco_fpath = ub.Path(config['coco_fpath'])
    if config['out_dpath'] is None:
        config['out_dpath'] = coco_fpath.parent
    out_dpath = ub.Path(config['out_dpath']).ensuredir()
    write_kwcoco = config['write_kwcoco']
    sensors = config['sensors']
    adj_cloud = config['adj_cloud']
    method = config['method']
    workers = util_parallel.coerce_num_workers(config['workers'])

    use_subprogress = workers == 0 or config['workermode'] != 'process'

    proc_context.start()
    proc_context.add_disk_info(out_dpath)

    pman = util_progress.ProgressManager(backend='rich')
    with pman:
        main_prog = pman.progiter(total=4, desc='Predict PyCOLD:')

        # ============
        # 1 / 4 Prepare Step
        # ============
        main_prog.set_postfix('Step 1: Prepare')

        metadata = None
        if (out_dpath / 'stacked').exists():
            for region in os.listdir(out_dpath / 'stacked'):
                if region in str(config['coco_fpath']):
                    if os.path.exists(out_dpath / 'reccg' / region):
                        logger.info('Skipping step 1 because the stacked image already exists...')
                        for root, dirs, files in os.walk(out_dpath / 'stacked' / region):
                            for file in files:
                                if file.endswith(".json"):
                                    json_path = os.path.join(root, file)

                                    with open(json_path, "r") as f:
                                        metadata = json.load(f)
                                break

        if metadata is None:
            meta_fpath = prepare_kwcoco.prepare_kwcoco_main(
                cmdline=0, coco_fpath=coco_fpath, out_dpath=out_dpath, sensors=sensors,
                adj_cloud=adj_cloud, method=method, workers=workers,
                resolution=config.resolution,
            )

            with open(meta_fpath, "r") as f:
                metadata = json.load(f)

        main_prog.step()

        # =========
        # 2 / 4 Tile Step
        # =========
        main_prog.set_postfix('Step 2: Process')
        logger.info('Starting COLD tile-processing...')
        tile_kwargs = tile_processing_kwcoco.TileProcessingKwcocoConfig().to_dict()
        tile_kwargs['stack_path'] = out_dpath / 'stacked' / metadata['region_id']
        tile_kwargs['reccg_path'] = out_dpath / 'reccg' / metadata['region_id']
        tile_kwargs['method'] = method
        tile_kwargs['prob'] = config['prob']
        tile_kwargs['conse'] = config['conse']
        tile_kwargs['cm_interval'] = config['cm_interval']
        if use_subprogress:
            tile_kwargs['pman'] = pman

        tile_log_fpath = out_dpath / 'reccg' / metadata['region_id'] / 'log.json'

        if os.path.exists(tile_log_fpath):
            logger.info('Skipping step 2 because COLD processing already finished...')
        else:
            jobs = ub.JobPool(mode=config['workermode'], max_workers=workers)
            with jobs:
                for i in pman.progiter(range(workers + 1), desc='submit process jobs', transient=True):
                    tile_kwargs['rank'] = i
                    tile_kwargs['n_cores'] = max(workers, 1)
                    jobs.submit(tile_processing_kwcoco.tile_process_main, cmdline=0, **tile_kwargs)

                tile_iter = pman.progiter(jobs.as_completed(), desc='Collect process jobs', total=len(jobs))
                for job in tile_iter:
                    job.result()
        main_prog.step()

        # ===========
        # 3 / 4 Export Step
        # ===========
        main_prog.set_postfix('Step 3: Export')
        logger.info('Writting tmp file of COLD output...')
        export_kwargs = export_cold_result_kwcoco.ExportColdKwcocoConfig().to_dict()
        export_kwargs['stack_path'] = out_dpath / 'stacked' / metadata['region_id']
        export_kwargs['reccg_path'] = out_dpath / 'reccg' / metadata['region_id']
        export_kwargs['combined_coco_fpath'] = config['combined_coco_fpath']
        export_kwargs['year_lowbound'] = config['year_lowbound']
        export_kwargs['year_highbound'] = config['year_highbound']
        export_kwargs['coefs'] = config['coefs']
        export_kwargs['combine'] = config['combine']
        export_kwargs['coefs_bands'] = config['coefs_bands']
        export_kwargs['timestamp'] = config['timestamp']
        export_kwargs['exclude_first'] = config['exclude_first']
        export_kwargs['sensors'] = sensors
        if use_subprogress:
            export_kwargs['pman'] = pman

        jobs = ub.JobPool(mode=config['workermode'], max_workers=workers)
        with jobs:
            for i in pman.progiter(range(workers + 1), desc='submit export jobs', transient=True):
                export_kwargs['rank'] = i
                export_kwargs['n_cores'] = max(workers, 1)
                jobs.submit(export_cold_result_kwcoco.export_cold_main, cmdline=0, **export_kwargs)

            tmp_iter = pman.progiter(jobs.as_completed(), desc='Collect export jobs', total=len(jobs))
            for job in tmp_iter:
                job.result()
        main_prog.step()

        # =============
        # 4 / 4 Assemble Step
        # =============
        main_prog.set_postfix('Step 4: Assemble')
        logger.info('Writting geotiff of COLD output...')
        assemble_kwargs = assemble_cold_result_kwcoco.AssembleColdKwcocoConfig().to_dict()
        assemble_kwargs['stack_path'] = out_dpath / 'stacked' / metadata['region_id']
        assemble_kwargs['reccg_path'] = out_dpath / 'reccg' / metadata['region_id']
        assemble_kwargs['coco_fpath'] = coco_fpath
        assemble_kwargs['combined_coco_fpath'] = config['combined_coco_fpath']
        assemble_kwargs['mod_coco_fpath'] = config['mod_coco_fpath']
        assemble_kwargs['write_kwcoco'] = write_kwcoco
        assemble_kwargs['year_lowbound'] = config['year_lowbound']
        assemble_kwargs['year_highbound'] = config['year_highbound']
        assemble_kwargs['coefs'] = config['coefs']
        assemble_kwargs['coefs_bands'] = config['coefs_bands']
        assemble_kwargs['timestamp'] = config['timestamp']
        assemble_kwargs['combine'] = config['combine']
        assemble_kwargs['exclude_first'] = config['exclude_first']
        assemble_kwargs['resolution'] = config.resolution
        assemble_kwargs['sensors'] = sensors

        if True:
            assemble_kwargs['pman'] = pman
        assemble_cold_result_kwcoco.assemble_main(
            cmdline=0, proc_context=proc_context, **assemble_kwargs)
        main_prog.step()

        # To keep meta data, this script won't clean up stack_path
        # remove stacked image
        # main_prog.set_postfix('Cleanup')
        # shutil.rmtree(tile_kwargs['stack_path'])
        # main_prog.step()


@profile
def read_json_metadata(folder_path):
    stacked_path = folder_path / 'stacked'
    for root, dirs, files in os.walk(stacked_path):
        for file in files:
            if file.endswith(".json"):
                json_path = os.path.join(root, file)

                with open(json_path, "r") as f:
                    metadata = json.load(f)
                    return metadata
    return None


if __name__ == '__main__':
    cold_predict_main()
