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

    codeblock()
    {
        __doc__='
        helper for python -c
        '
        echo "$1" | python -c "import sys; from textwrap import dedent; print(dedent(sys.stdin.read()).strip('\n'))"
    }

    # Pull out a small selection of images just so we can test.
    python -c "$(codeblock "
        import ubelt as ub
        import kwcoco
        dset = kwcoco.CocoDataset('$DATA_DVC_DPATH/Drop6-SMALL/imgonly-KR_R001.kwcoco.json')
        from watch.utils import util_time
        images = dset.images()
        dates = list(map(util_time.coerce_datetime, images.lookup('date_captured')))
        flags = [d.year < 2017 for d in dates]
        chosen = images.compress(flags)
        sub = dset.subset(chosen)
        sub.fpath = dset.fpath
        sub.dump()
        ")"

    DATA_DVC_DPATH=$(geowatch_dvc --tags=phase2_data --hardware="auto")
    EXPT_DVC_DPATH=$(geowatch_dvc --tags=phase2_expt --hardware="auto")
    python -m watch.tasks.cold.predict \
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

    smartwatch visualize \
        "$DATA_DVC_DPATH"/Drop6-SMALL/_pycold/imgonly-KR_R001-cold.fixed.kwcoco.zip \
        --channels="L8:(red|green|blue,red_COLD_cv|green_COLD_cv|blue_COLD_cv)" \
        --exclude_sensors="S2" \
        --smart=True --skip_aggressive=True

    ####################
    ### FULL REGION TEST
    ####################

    DATA_DVC_DPATH=$(geowatch_dvc --tags=phase2_data --hardware="auto")
    EXPT_DVC_DPATH=$(geowatch_dvc --tags=phase2_expt --hardware="auto")
    python -m watch.tasks.cold.predict \
        --coco_fpath="$DATA_DVC_DPATH/Drop6/imgonly-KR_R001.kwcoco.json" \
        --combined_coco_fpath="$DATA_DVC_DPATH/Drop6-MeanYear10GSD-V2/imgonly-KR_R001.kwcoco.zip" \
        --out_dpath="$DATA_DVC_DPATH/Drop6-MeanYear10GSD-V2/_pycold_combine" \
        --mod_coco_fpath="$DATA_DVC_DPATH/Drop6-MeanYear10GSD-V2/imgonly_KR_R001_cold.kwcoco.zip" \
        --sensors='L8' \
        --adj_cloud=False \
        --method='COLD' \
        --prob=0.99 \
        --conse=6 \
        --cm_interval=60 \
        --year_lowbound=None \
        --year_highbound=None \
        --coefs=cv,rmse,a0,a1,b1,c1 \
        --coefs_bands=0,1,2,3,4,5 \
        --timestamp=False \
        --combine=False \
        --resolution='10GSD' \
        --workermode='serial' \
        --workers=0

    kwcoco stats "$DATA_DVC_DPATH"/Drop6-MeanYear10GSD-V2/imgonly_KR_R001_cold.kwcoco.zip
    geowatch stats "$DATA_DVC_DPATH"/Drop6-MeanYear10GSD-V2/imgonly_KR_R001_cold.kwcoco.zip

    # Fix path problem because we wrote a different directory
    # TODO: fix this script so the output always uses absolute paths?
    # or at least doesn't write invalid data that needs fixing?
    DATA_DVC_DPATH=$(geowatch_dvc --tags=phase2_data --hardware="auto")
    kwcoco reroot \
        --src="$DATA_DVC_DPATH"/Drop6-MeanYear10GSD-V2/imgonly_KR_R001_cold.kwcoco.zip \
        --dst="$DATA_DVC_DPATH"/Drop6-MeanYear10GSD-V2/imgonly_KR_R001_cold_fixed.kwcoco.zip \
        --old_prefix="KR_R001" --new_prefix="../KR_R001"

    DATA_DVC_DPATH=$(geowatch_dvc --tags=phase2_data --hardware="auto")
    smartwatch visualize \
        "$DATA_DVC_DPATH"/Drop6-MeanYear10GSD-V2/imgonly_KR_R001_cold.kwcoco.zip \
        --channels="L8:(red|green|blue,red_COLD_a1|green_COLD_a1|blue_COLD_a1,red_COLD_cv|green_COLD_cv|blue_COLD_cv,red_COLD_rmse|green_COLD_rmse|blue_COLD_rmse)" \
        --smart=True


    ########################
    ### MULTIPLE REGION TEST
    ########################
    DVC_DATA_DPATH=$(geowatch_dvc --tags='phase2_data' --hardware=auto)

    echo "$DVC_DATA_DPATH"
    BUNDLE_DPATH=$DVC_DATA_DPATH/Drop6
    python -m watch.cli.prepare_teamfeats \
        --base_fpath \
            "$BUNDLE_DPATH"/imganns-*BR_R*.kwcoco.zip \
            "$BUNDLE_DPATH"/imganns-*KR_R*.kwcoco.zip \
            "$BUNDLE_DPATH"/imganns-*NZ_R*.kwcoco.zip \
            "$BUNDLE_DPATH"/imganns-*US_R*.kwcoco.zip \
        --expt_dpath="$DVC_EXPT_DPATH" \
        --with_cold=1 \
        --with_landcover=0 \
        --with_materials=0 \
        --with_invariants=0 \
        --with_depth=0 \
        --do_splits=0 \
        --skip_existing=1 \
        --cold_workers=8 \
        --cold_workermode=thread \
        --workers=2 \
        --backend=tmux --run=1
"""
import scriptconfig as scfg
import ubelt as ub
import json
import logging
import shutil

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
    combined_coco_fpath = scfg.Value(None, help=ub.paragraph(
        '''
        a path to a file to combined input kwcoco file (to merge with)
        '''))
    mod_coco_fpath = scfg.Value(None, help='file path for modified output coco json')

    out_dpath = scfg.Value(None, help='output directory for the output. If unspecified uses the output kwcoco bundle')
    sensors = scfg.Value('L8', type=str, help='sensor type, default is "L8"')
    adj_cloud = scfg.Value(False, help='How to treat QA band, default is False: ignoring adj. cloud class')
    method = scfg.Value('COLD', choices=['COLD', 'HybridCOLD', 'OBCOLD'], help='type of cold algorithms')
    prob = scfg.Value(None, help='change probability of chi-distribution, e.g., 0.99')
    conse = scfg.Value(None, help='consecutive observation to confirm change, e.g., 6')
    cm_interval = scfg.Value(None, help='CM output inverval, e.g., 60')
    year_lowbound = scfg.Value(None, help='min year for saving geotiff, e.g., 2017')
    year_highbound = scfg.Value(None, help='max year for saving geotiff, e.g., 2022')
    coefs = scfg.Value(None, type=str, help="list of COLD coefficients for saving geotiff, e.g., a0,c1,a1,b1,a2,b2,a3,b3,cv,rmse")
    coefs_bands = scfg.Value(None, type=str, help='indicate the ba_nds for output coefs_bands, e.g., 0,1,2,3,4,5')
    timestamp = scfg.Value(False, help='True: exporting cold result by timestamp, False: exporting cold result by year, Default is False')
    combine = scfg.Value(True, help='for temporal combined mode, Default is True')
    track_emissions = scfg.Value(True, help='if True use codecarbon for emission tracking')
    resolution = scfg.Value('30GSD', help='if specified then data is processed at this resolution')
    workers = scfg.Value(16, help='total number of workers')
    workermode = scfg.Value('process', help='Can be process, serial, or thread')


@profile
def cold_predict_main(cmdline=1, **kwargs):
    """
    Args:
        cmdline (int, optional): _description_. Defaults to 1.

    Ignore:
        python -m watch.tasks.cold.predict --help
        TEST_COLD=1 xdoctest -m watch.tasks.cold.predict cold_predict_main

     Example:
        >>> # xdoctest: +REQUIRES(env:TEST_COLD)
        >>> from watch.tasks.cold.predict import cold_predict_main
        >>> from watch.tasks.cold.predict import *
        >>> kwargs= dict(
        >>>   coco_fpath = ub.Path('/home/jws18003/data/dvc-repos/smart_data_dvc/Aligned-Drop6-2022-12-01-c30-TA1-S2-L8-WV-PD-ACC-2/imgonly-KR_R001.kwcoco.json'),
        >>>   out_dpath = ub.Path.appdir('/gpfs/scratchfs1/zhz18039/jws18003/kwcoco'),
        >>>   sensors = 'L8, S2',
        >>>   adj_cloud = False,
        >>>   method = 'COLD',
        >>>   prob = 0.99,
        >>>   conse = 6,
        >>>   cm_interval = 60,
        >>>   year_lowbound = None,
        >>>   year_highbound = None,
        >>>   coefs = 'a0', 'cv',
        >>>   coefs_bands = '0, 1, 2, 3, 4, 5',
        >>>   timestamp = True,
        >>>   workermode = 'process',
        >>>   mod_coco_fpath = ub.Path('/home/jws18003/data/dvc-repos/smart_data_dvc/Aligned-Drop6-2022-12-01-c30-TA1-S2-L8-WV-PD-ACC-2/KR_R001/imgonly-KR_R001.kwcoco.modified.json'),
        >>> )
        >>> cmdline=0
        >>> cold_predict_main(cmdline, **kwargs)
    """
    from watch.tasks.cold import prepare_kwcoco
    from watch.tasks.cold import tile_processing_kwcoco
    from watch.tasks.cold import export_cold_result_kwcoco
    from watch.tasks.cold import assemble_cold_result_kwcoco

    config = ColdPredictConfig.cli(cmdline=cmdline, data=kwargs, strict=True)
    import rich
    rich.print('config = {}'.format(ub.urepr(config, nl=1)))

    from watch.utils import process_context
    from watch.utils import util_parallel
    from watch.utils import util_progress
    from watch.utils import util_json
    resolved_config = config.to_dict()
    resolved_config = util_json.ensure_json_serializable(resolved_config)

    proc_context = process_context.ProcessContext(
        name='watch.tasks.cold.predict',
        type='process',
        config=resolved_config,
        track_emissions=config['track_emissions'],
    )

    coco_fpath = ub.Path(config['coco_fpath'])
    if config['out_dpath'] is None:
        config['out_dpath'] = coco_fpath.parent
    out_dpath = ub.Path(config['out_dpath']).ensuredir()
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
        meta_fpath = prepare_kwcoco.prepare_kwcoco_main(
            cmdline=0, coco_fpath=coco_fpath, out_dpath=out_dpath, sensors=sensors,
            adj_cloud=adj_cloud, method=method, workers=workers,
            resolution=config.resolution,
        )
        with open(meta_fpath, 'r') as meta:
            metadata = json.load(meta)
        main_prog.step()

        # =========
        # 2 / 4 Tile Step
        # =========
        main_prog.set_postfix('Step 2: Process')
        logger.info('Starting COLD tile-processing...')
        tile_kwargs = tile_processing_kwcoco.TileProcessingKwcocoConfig().to_dict()
        tile_kwargs['stack_path'] = out_dpath / 'stacked' / metadata['region_id']
        tile_kwargs['reccg_path'] = out_dpath / 'reccg' / metadata['region_id']
        tile_kwargs['meta_fpath'] = meta_fpath
        tile_kwargs['method'] = method
        tile_kwargs['prob'] = config['prob']
        tile_kwargs['conse'] = config['conse']
        tile_kwargs['cm_interval'] = config['cm_interval']
        if use_subprogress:
            tile_kwargs['pman'] = pman

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
        export_kwargs['stack_path'] = tile_kwargs['stack_path']
        export_kwargs['reccg_path'] = tile_kwargs['reccg_path']
        export_kwargs['meta_fpath'] = meta_fpath
        export_kwargs['combined_coco_fpath'] = config['combined_coco_fpath']
        export_kwargs['year_lowbound'] = config['year_lowbound']
        export_kwargs['year_highbound'] = config['year_highbound']
        export_kwargs['coefs'] = config['coefs']
        export_kwargs['combine'] = config['combine']
        export_kwargs['coefs_bands'] = config['coefs_bands']
        export_kwargs['timestamp'] = config['timestamp']
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
        assemble_kwargs['stack_path'] = tile_kwargs['stack_path']
        assemble_kwargs['reccg_path'] = tile_kwargs['reccg_path']
        assemble_kwargs['coco_fpath'] = coco_fpath
        assemble_kwargs['combined_coco_fpath'] = config['combined_coco_fpath']
        assemble_kwargs['mod_coco_fpath'] = config['mod_coco_fpath']
        assemble_kwargs['meta_fpath'] = meta_fpath
        assemble_kwargs['year_lowbound'] = config['year_lowbound']
        assemble_kwargs['year_highbound'] = config['year_highbound']
        assemble_kwargs['coefs'] = config['coefs']
        assemble_kwargs['coefs_bands'] = config['coefs_bands']
        assemble_kwargs['timestamp'] = config['timestamp']
        assemble_kwargs['combine'] = config['combine']
        assemble_kwargs['resolution'] = config.resolution
        assemble_kwargs['sensors'] = sensors

        if True:
            assemble_kwargs['pman'] = pman
        assemble_cold_result_kwcoco.assemble_main(
            cmdline=0, proc_context=proc_context, **assemble_kwargs)
        main_prog.step()

        # remove stacked image
        main_prog.set_postfix('Cleanup')
        shutil.rmtree(tile_kwargs['stack_path'])
        main_prog.step()


if __name__ == '__main__':
    cold_predict_main()
