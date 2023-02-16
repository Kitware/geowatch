import scriptconfig as scfg
import ubelt as ub
import json
import kwcoco
import logging

logger = logging.getLogger(__name__)

class ColdPredictConfig(scfg.DataConfig):
    """
    The docstring will be the description in the CLI help
    """
    
    coco_fpath = scfg.Value(None, position=1, help=ub.paragraph(
        '''
        a path to a file to input kwcoco file
        '''))
    out_dpath = scfg.Value(None, help='output directory for the output')
    adj_cloud = scfg.Value(False, help='How to treat QA band, default is False: ignoring adj. cloud class')
    method = scfg.Value('COLD', choices= ['COLD', 'HybridCOLD', 'OBCOLD'], help='type of cold algorithms')
    prob = scfg.Value(None, help='change probability of chi-distribution, e.g., 0.99')
    conse = scfg.Value(None, help='consecutive observation to confirm change, e.g., 6')
    cm_interval = scfg.Value(None, help='CM output inverval, e.g., 60')
    ref_path = scfg.Value(None, help='refererence image path')
    year_lowbound = scfg.Value(None, help='min year for saving geotiff, e.g., 2017')
    year_highbound = scfg.Value(None, help='max year for saving geotiff, e.g., 2022')
    coefs = scfg.Value(None, help="list of COLD coefficients for saving geotiff, e.g., ['a0', 'c1', 'a1', 'b1', 'a2', 'b2', 'a3', 'b3', 'cv', 'rmse']")
    coefs_bands = scfg.Value(None, help='indicate the ba_nds for output coefs_bands, e.g., [0, 1, 2, 3, 4, 5]')
    timestamp = scfg.Value(True, help='True: exporting cold result by timestamp, False: exporting cold result by year, Default is False')
    mode = scfg.Value('process', help='Can be process, serial, or thread')
    
    #n_core #

    # b_c2 = scfg.Value(True, help='indicate if it is c2 or not') 
    
def main(cmdline=1, **kwargs):
    """

    Args:
        cmdline (int, optional): _description_. Defaults to 1. 
        
    Ignore:
        python -m watch.tasks.cold.predict --help
        TEST_COLD=1 xdoctest -m watch.tasks.cold.predict main

     Example:
        >>> # xdoctest: +REQUIRES(env:TEST_COLD)       
        >>> from watch.tasks.cold.predict import main
        >>> from watch.tasks.cold.predict import *
        >>> kwargs= dict(        
        >>>   coco_fpath = '/home/jws18003/data/dvc-repos/smart_data_dvc/Aligned-Drop4-2022-08-08-TA1-S2-L8-ACC/US_C000/data_US_C000.kwcoco.json',
        >>>   out_dpath = ub.Path.appdir('/gpfs/scratchfs1/zhz18039/jws18003/kwcoco'),
        >>>   adj_cloud = False,
        >>>   method = 'COLD',
        >>>   prob = 0.99,
        >>>   conse = 6,
        >>>   cm_interval = 60,
        >>>   ref_path = "/home/jws18003/Document/kwcoco_working/US_C000_rowcol2.tif",
        >>>   year_lowbound = 2020,
        >>>   year_highbound = 2020,
        >>>   coefs = ['cv'],
        >>>   coefs_bands = [0, 1, 2, 3, 4, 5],
        >>>   timestamp = True,       
        >>>   mode = 'process', 
        >>> )    
        >>> cmdline=0    
        >>> main(cmdline, **kwargs)
    """
    from watch.tasks.cold import prepare_kwcoco   
    from watch.tasks.cold import tile_processing_kwcoco
    from watch.tasks.cold import export_cold_result_kwcoco
    
    config = ColdPredictConfig.legacy(cmdline=cmdline, data=kwargs)
    coco_fpath = config['coco_fpath']
    out_dpath = ub.Path(config['out_dpath']).ensuredir()
    adj_cloud = config['adj_cloud']
    method = config['method']
    meta_fpath = prepare_kwcoco.main(cmdline=0, coco_fpath=coco_fpath, out_dpath=out_dpath, adj_cloud=adj_cloud, method=method)  
    # meta_fpath = '/gpfs/scratchfs1/zhz18039/jws18003/kwcoco/stacked/US_C000/block_x9_y9/crop_20210716T150000Z_N38.904157W077.594580_N39.117177W077.375621_L8_0.json'  
    meta = open(meta_fpath)
    metadata = json.load(meta)
    
    logger.info('Starting COLD tile-processing...')
    tile_kwargs = tile_processing_kwcoco.TileProcessingKwcocoConfig().to_dict()
    tile_kwargs['stack_path'] = out_dpath / 'stacked' / metadata['region_id']
    tile_kwargs['reccg_path'] = out_dpath / 'reccg' / metadata['region_id']
    tile_kwargs['meta_fpath'] = meta_fpath
    tile_kwargs['method'] = config['method']
    tile_kwargs['prob'] = config['prob']
    tile_kwargs['conse'] = config['conse']
    tile_kwargs['cm_interval'] = config['cm_interval']
   
    workers = 8
    jobs = ub.JobPool(mode=config['mode'], max_workers=workers)
    for i in range(workers + 1):
        #jobs.submit(func, arg1, arg2, arg3=34)
        #func(arg, arg3, arg3=34)
        tile_kwargs['rank'] = i
        tile_kwargs['n_cores'] = workers
        jobs.submit(tile_processing_kwcoco.main, cmdline=0, **tile_kwargs)    
    
    for job in jobs.as_completed(desc='Collect tile jobs', progkw={'verbose': 3}):
        ret = job.result()    
     
    
    logger.info('Writting geotiff of COLD output...')    
    export_kwargs = export_cold_result_kwcoco.ExportColdKwcocoConfig().to_dict()
    export_kwargs['stack_path'] = tile_kwargs['stack_path']
    export_kwargs['reccg_path'] = tile_kwargs['reccg_path']
    export_kwargs['out_path'] = tile_kwargs['reccg_path']
    export_kwargs['meta_fpath'] = meta_fpath
    export_kwargs['reference_path'] = config['ref_path']
    export_kwargs['region_id'] = metadata['region_id']
    export_kwargs['year_lowbound'] = config['year_lowbound']
    export_kwargs['year_highbound'] = config['year_highbound']
    export_kwargs['coefs'] = config['coefs']
    export_kwargs['coefs_bands'] = config['coefs_bands']
    export_kwargs['timestamp'] = config['timestamp']
    export_cold_result_kwcoco.main(cmdline=0, **export_kwargs)  

    
if __name__ == '__main__':
    main()