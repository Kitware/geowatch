r"""
Main prediction script for cold

CommandLine:

    DATA_DVC_DPATH=$(smartwatch_dvc --tags=phase2_data --hardware="auto")
    EXPT_DVC_DPATH=$(smartwatch_dvc --tags=phase2_expt --hardware="auto")
    python -m watch.tasks.cold.predict \
        --coco_fpath="$DATA_DVC_DPATH/Drop6/imgonly-KR_R001.kwcoco.json" \
        --out_dpath="$DATA_DVC_DPATH/Drop6/_pycold" \
        --mod_coco_fpath="$DATA_DVC_DPATH/Drop6/_pycold/imgonly-KR_R001-cold.kwcoco.json" \
        --adj_cloud=False \
        --method='COLD' \
        --prob=0.99 \
        --conse=6 \
        --cm_interval=60 \
        --year_lowbound=None \
        --year_highbound=None \
        --coefs=cv,a0,a1,b1,c1,rmse \
        --coefs_bands=0,1,2,3,4,5 \
        --timestamp=True \
        --mode='process' \
        --workers=8

    # Fix path problem because we wrote a different directory
    # TODO: fix this script so the output always uses absolute paths?
    # or at least doesn't write invalid data that needs fixing?
    DATA_DVC_DPATH=$(smartwatch_dvc --tags=phase2_data --hardware="auto")
    kwcoco reroot \
        --src="$DATA_DVC_DPATH"/Drop6/_pycold/imgonly-KR_R001-cold.kwcoco.json \
        --dst="$DATA_DVC_DPATH"/Drop6/_pycold/imgonly-KR_R001-cold.fixed.kwcoco.zip \
        --old_prefix="KR_R001" --new_prefix="../KR_R001"

    DATA_DVC_DPATH=$(smartwatch_dvc --tags=phase2_data --hardware="auto")
    smartwatch visualize \
        "$DATA_DVC_DPATH"/Drop6/_pycold/imgonly-KR_R001-cold.fixed.kwcoco.zip \
        --channels="L8:(red|green|blue,red_COLD_a1|green_COLD_a1|blue_COLD_a1,red_COLD_cv|green_COLD_cv|blue_COLD_cv,red_COLD_rmse|green_COLD_rmse|blue_COLD_rmse)" \
        --smart=True


    ### Multiple Regions:

    DVC_DATA_DPATH=$(smartwatch_dvc --tags='phase2_data' --hardware=auto)
    BUNDLE_DPATH=$DVC_DATA_DPATH/Drop6
    python -m watch.cli.prepare_teamfeats \
        --base_fpath "$BUNDLE_DPATH"/imganns-*.kwcoco.zip \
        --expt_dpath="$DVC_EXPT_DPATH" \
        --with_cold=1 \
        --with_landcover=0 \
        --with_materials=0 \
        --with_invariants=0 \
        --with_depth=0 \
        --do_splits=0 \
        --skip_existing=1 \
        --cold_workers=2 \
        --workers=4 \
        --backend=tmux --run=0

"""
import scriptconfig as scfg
import ubelt as ub
import json
import logging

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
        a path to a file to input kwcoco file
        '''))
    out_dpath = scfg.Value(None, help='output directory for the output. If unspecified uses the output kwcoco bundle')
    adj_cloud = scfg.Value(False, help='How to treat QA band, default is False: ignoring adj. cloud class')
    method = scfg.Value('COLD', choices=['COLD', 'HybridCOLD', 'OBCOLD'], help='type of cold algorithms')
    prob = scfg.Value(None, help='change probability of chi-distribution, e.g., 0.99')
    conse = scfg.Value(None, help='consecutive observation to confirm change, e.g., 6')
    cm_interval = scfg.Value(None, help='CM output inverval, e.g., 60')
    year_lowbound = scfg.Value(None, help='min year for saving geotiff, e.g., 2017')
    year_highbound = scfg.Value(None, help='max year for saving geotiff, e.g., 2022')
    coefs = scfg.Value(None, type=str, help="list of COLD coefficients for saving geotiff, e.g., a0,c1,a1,b1,a2,b2,a3,b3,cv,rmse")
    coefs_bands = scfg.Value(None, type=str, help='indicate the ba_nds for output coefs_bands, e.g., 0,1,2,3,4,5')
    timestamp = scfg.Value(True, help='True: exporting cold result by timestamp, False: exporting cold result by year, Default is False')
    mode = scfg.Value('process', help='Can be process, serial, or thread')
    mod_coco_fpath = scfg.Value(None, help='file path for modified coco json')
    track_emissions = scfg.Value(True, help='if True use codecarbon for emission tracking')
    workers = scfg.Value(16, help='total number of workers')


@profile
def cold_predict_main(cmdline=1, **kwargs):
    """
    Args:
        cmdline (int, optional): _description_. Defaults to 1.

    Ignore:
        python -m watch.tasks.cold.predict --help
        TEST_COLD=1 xdoctest -m watch.tasks.cold.predict cold_predict_main

     Example:
        >>> # xdoctest: +REQUIRES(gg:TEST_COLD)
        >>> from watch.tasks.cold.predict import cold_predict_main
        >>> from watch.tasks.cold.predict import *
        >>> kwargs= dict(
        >>>   coco_fpath = ub.Path('/home/jws18003/data/dvc-repos/smart_data_dvc/Aligned-Drop6-2022-12-01-c30-TA1-S2-L8-WV-PD-ACC-2/imgonly-KR_R001.kwcoco.json'),
        >>>   out_dpath = ub.Path.appdir('/gpfs/scratchfs1/zhz18039/jws18003/kwcoco'),
        >>>   adj_cloud = False,
        >>>   method = 'COLD',
        >>>   prob = 0.99,
        >>>   conse = 6,
        >>>   cm_interval = 60,
        >>>   year_lowbound = None,
        >>>   year_highbound = None,
        >>>   coefs = ['a0', 'cv'],
        >>>   coefs_bands = [0, 1, 2, 3, 4, 5],
        >>>   timestamp = True,
        >>>   mode = 'process',
        >>>   mod_coco_fpath = ub.Path('/home/jws18003/data/dvc-repos/smart_data_dvc/Aligned-Drop6-2022-12-01-c30-TA1-S2-L8-WV-PD-ACC-2/KR_R001/imgonly-KR_R001.kwcoco.modified.json'),
        >>> )
        >>> cmdline=0
        >>> cold_predict_main(cmdline, **kwargs)
    """
    from watch.tasks.cold import prepare_kwcoco
    from watch.tasks.cold import tile_processing_kwcoco
    from watch.tasks.cold import export_cold_result_kwcoco
    from watch.tasks.cold import assemble_cold_result_kwcoco

    config = ColdPredictConfig.cli(cmdline=cmdline, data=kwargs)
    print('config = {}'.format(ub.urepr(dict(config), nl=1)))

    from watch.utils import process_context
    from watch.utils import util_parallel
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
    adj_cloud = config['adj_cloud']
    method = config['method']
    workers = util_parallel.coerce_num_workers(config['workers'])

    proc_context.start()
    proc_context.add_disk_info(out_dpath)

    meta_fpath = prepare_kwcoco.prepare_kwcoco_main(
        cmdline=0, coco_fpath=coco_fpath, out_dpath=out_dpath,
        adj_cloud=adj_cloud, method=method)
    with open(meta_fpath, 'r') as meta:
        metadata = json.load(meta)

    logger.info('Starting COLD tile-processing...')
    tile_kwargs = tile_processing_kwcoco.TileProcessingKwcocoConfig().to_dict()
    tile_kwargs['stack_path'] = out_dpath / 'stacked' / metadata['region_id']
    tile_kwargs['reccg_path'] = out_dpath / 'reccg' / metadata['region_id']
    tile_kwargs['meta_fpath'] = meta_fpath
    tile_kwargs['method'] = method
    tile_kwargs['prob'] = config['prob']
    tile_kwargs['conse'] = config['conse']
    tile_kwargs['cm_interval'] = config['cm_interval']

    jobs = ub.JobPool(mode=config['mode'], max_workers=workers)
    for i in range(workers + 1):
        #jobs.submit(func, arg1, arg2, arg3=34)
        #func(arg, arg3, arg3=34)
        tile_kwargs['rank'] = i
        tile_kwargs['n_cores'] = max(workers, 1)
        jobs.submit(tile_processing_kwcoco.tile_process_main, cmdline=0, **tile_kwargs)

    for job in jobs.as_completed(desc='Collect tile jobs', progkw={'verbose': 3}):
        job.result()

    logger.info('Writting tmp file of COLD output...')
    export_kwargs = export_cold_result_kwcoco.ExportColdKwcocoConfig().to_dict()
    export_kwargs['stack_path'] = tile_kwargs['stack_path']
    export_kwargs['reccg_path'] = tile_kwargs['reccg_path']
    export_kwargs['meta_fpath'] = meta_fpath
    export_kwargs['year_lowbound'] = config['year_lowbound']
    export_kwargs['year_highbound'] = config['year_highbound']
    export_kwargs['coefs'] = config['coefs']
    export_kwargs['coefs_bands'] = config['coefs_bands']
    export_kwargs['timestamp'] = config['timestamp']

    jobs = ub.JobPool(mode=config['mode'], max_workers=workers)
    for i in range(workers + 1):
        export_kwargs['rank'] = i
        export_kwargs['n_cores'] = max(workers, 1)
        jobs.submit(export_cold_result_kwcoco.export_cold_main, cmdline=0, **export_kwargs)

    for job in jobs.as_completed(desc='Collect tmp jobs', progkw={'verbose': 3}):
        job.result()

    logger.info('Writting geotiff of COLD output...')
    assemble_kwargs = assemble_cold_result_kwcoco.AssembleColdKwcocoConfig().to_dict()
    assemble_kwargs['stack_path'] = tile_kwargs['stack_path']
    assemble_kwargs['reccg_path'] = tile_kwargs['reccg_path']
    assemble_kwargs['coco_fpath'] = coco_fpath
    assemble_kwargs['mod_coco_fpath'] = config['mod_coco_fpath']
    assemble_kwargs['meta_fpath'] = meta_fpath
    assemble_kwargs['year_lowbound'] = config['year_lowbound']
    assemble_kwargs['year_highbound'] = config['year_highbound']
    assemble_kwargs['coefs'] = config['coefs']
    assemble_kwargs['coefs_bands'] = config['coefs_bands']
    assemble_kwargs['timestamp'] = config['timestamp']
    assemble_cold_result_kwcoco.assemble_main(
        cmdline=0, proc_context=proc_context, **assemble_kwargs)

if __name__ == '__main__':
    cold_predict_main()
