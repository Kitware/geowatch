import os
import numpy as np
import pandas as pd
from osgeo import gdal
import datetime as datetime
from os.path import join
import json
from mpi4py import MPI
import pickle
import yaml
import watch
import kwcoco
import kwimage
import scriptconfig as scfg
import ubelt as ub
import logging
logger = logging.getLogger(__name__)

class AssembleColdKwcocoConfig(scfg.DataConfig):
    """
    The docstring will be the description in the CLI help
    """
    stack_path = scfg.Value(None, help='folder directory of stacked data')
    reccg_path = scfg.Value(None, help='folder directory of cold processing result')
    coco_fpath = scfg.Value(None, help='file path of coco json')
    mod_coco_fpath = scfg.Value(None, help='file path for modified coco json')
    meta_fpath = scfg.Value(None, help='file path of metadata json created by prepare_kwcoco script')    
    region_id = scfg.Value(None, help='region name of Kwcoco data, e.g., US_C000')    
    year_lowbound = scfg.Value(None, help='min year for saving geotiff, e.g., 2017')
    year_highbound = scfg.Value(None, help='max year for saving geotiff, e.g., 2022')
    coefs = scfg.Value(None, help="list of COLD coefficients for saving geotiff, e.g., ['a0', 'c1', 'a1', 'b1', 'a2', 'b2', 'a3', 'b3', 'cv', 'rmse']")
    coefs_bands = scfg.Value(None, help='indicate the ba_nds for output coefs_bands, e.g., [0, 1, 2, 3, 4, 5]')
    timestamp = scfg.Value(True, help='True: exporting cold result by timestamp, False: exporting cold result by year, Default is False')   
    
    # out_path = scfg.Value(None, help='folder directory of output geotiff image')
    # rank = scfg.Value("MPI",  help='rank id')
    # n_cores = scfg.Value("MPI" ,help='number of processor')
    # reference_path = scfg.Value(None, help='file path of reference image')

 
def main(cmdline=1, **kwargs):
    """_summary_

    Args:
        cmdline (int, optional): _description_. Defaults to 1.
        
    Ignore:
        python -m watch.tasks.cold.export_cold_result_kwcoco --help
        TEST_COLD=1 xdoctest -m watch.tasks.cold.export_cold_result_kwcoco main
    Example:
    >>> # xdoctest: +REQUIRES(env:TEST_COLD) 
    >>> from watch.tasks.cold.export_cold_result_kwcoco import main
    >>> from watch.tasks.cold.export_cold_result_kwcoco import *
    >>> kwargs= dict(
    >>>    stack_path = "/gpfs/scratchfs1/zhz18039/jws18003/kwcoco/stacked/KR_R001",
    >>>    reccg_path = "/gpfs/scratchfs1/zhz18039/jws18003/kwcoco/reccg/KR_R001",
    >>>    coco_fpath = ub.Path('/home/jws18003/data/dvc-repos/smart_data_dvc/Aligned-Drop6-2022-12-01-c30-TA1-S2-L8-WV-PD-ACC-2/imgonly-KR_R001.kwcoco.json'),
    >>>    mod_coco_fpath = ub.Path('/home/jws18003/data/dvc-repos/smart_data_dvc/Aligned-Drop6-2022-12-01-c30-TA1-S2-L8-WV-PD-ACC-2/KR_R001/imgonly-KR_R001.kwcoco.modified.json'),
    >>>    meta_fpath = '/gpfs/scratchfs1/zhz18039/jws18003/kwcoco/stacked/KR_R001/block_x10_y1/crop_20140115T020000Z_N37.643680E128.649453_N37.683356E128.734073_L8_0.json',
    >>>    region_id = "KR_R001",
    >>>    coefs = ['cv'],
    >>>    year_lowbound = 2017,
    >>>    year_highbound = 2022,
    >>>    coefs_bands = [0, 1, 2, 3, 4, 5],
    >>>    timestamp = True,
    >>>    )    
    >>> cmdline=0    
    >>> main(cmdline, **kwargs)
    """ 
   
    rank = 0
    n_cores = 1
    config_in = AssembleColdKwcocoConfig.legacy(cmdline=cmdline, data=kwargs)
    # rank = config_in['rank']
    # n_cores = config_in['n_cores']
    stack_path = config_in['stack_path']
    reccg_path = config_in['reccg_path']
    coco_fpath = config_in['coco_fpath']
    mod_coco_fpath = config_in['mod_coco_fpath']
    out_path = os.path.join(reccg_path, 'cold_feature')
    meta_fpath = config_in['meta_fpath']    
    region = config_in['region_id']
    year_lowbound = config_in['year_lowbound']
    year_highbound = config_in['year_highbound']
    coefs = config_in['coefs']
    coefs_bands = config_in['coefs_bands']
    timestamp = config_in['timestamp']
    
    # TODO: MPI mode
    # if config_in['rank'] == 'MPI':
    #     ## MPI mode
    #     raise NotImplementedError('todo')
    #     MPI = 'TODO'
    #     comm = MPI.COMM_WORLD
    #     rank = comm.Get_rank()
    #     n_cores = comm.Get_size()
    # else:
    #     rank = config_in['rank']
    #     n_cores = config_in['n_cores']
                
    meta = open(meta_fpath)
    config = json.load(meta)    
    vid_w = config['video_w']
    vid_h = config['video_h']
           
    coco_dset = kwcoco.CocoDataset(coco_fpath)
    coco_img = coco_dset.images().coco_images[0]
    primary_asset = coco_img.primary_asset()
    primary_fpath = os.path.join(ub.Path(coco_img.bundle_dpath), primary_asset['file_name'])
    ref_image = gdal.Open(primary_fpath, gdal.GA_ReadOnly)
    trans = ref_image.GetGeoTransform()
    proj = ref_image.GetProjection()
    
    # Get original transform from projection to image space
    c, a, b, f, d, e = trans
    original = kwimage.Affine(np.array([
        [a, b, c],
        [d, e, f],
        [0, 0, 1],
    ]))
    
    warp_vid_from_img = kwimage.Affine.coerce(coco_img.img['warp_img_to_vid']).inv()
    new_geotrans =  original @ warp_vid_from_img
    a, b, c, d, e, f, g, h, i = np.array(new_geotrans).ravel().tolist()
    new_gdal_transform = (c, a, b, f, d, e)     
        
    # define variables
    n_cols = config['padded_n_cols']
    n_rows = config['padded_n_rows']
    n_block_x = config['n_block_x']
    n_block_y = config['n_block_y']        
    block_width = int(n_cols / n_block_x)  # width of a block
    block_height = int(n_rows / n_block_y)  # height of a block
    n_blocks = n_block_x * n_block_y  # total number of blocks   
    
    log = open(os.path.join(reccg_path, 'log.json'))
    cold_param = json.load(log)
    method = cold_param['algorithm']
    prob = cold_param['prob']
    conse = cold_param['conse']
        
    coef_names = ['cv', 'rmse', 'a0', 'a1', 'b1', 'a2', 'b2', 'a3', 'b3', 'c1']
    band_names = [0, 1, 2, 3, 4, 5]
    SLOPE_SCALE = 10000

    BAND_INFO = {0: 'blue',
                1: 'green',
                2: 'red',
                3: 'nir',
                4: 'swir16',
                5: 'swir22'}
            
    if coefs is not None:
        try:
            coefs = list(coefs.split(","))
            # coefs = [str(coef) for coef in coefs]
        except:
            print("Illegal coefs inputs: example, --coefs='a0, c1, a1, b1, a2, b2, a3, b3, cv, rmse'")

        try:
            coefs_bands = list(coefs_bands.split(","))
            coefs_bands = [int(coefs_band) for coefs_band in coefs_bands]
        except:
            print("Illegal coefs_bands inputs: example, --coefs_bands='0, 1, 2, 3, 4, 5, 6'")

    dt = np.dtype([('t_start', np.int32),
                   ('t_end', np.int32),
                   ('t_break', np.int32),
                   ('pos', np.int32),
                   ('num_obs', np.int32),
                   ('category', np.short),
                   ('change_prob', np.short),
                   ('coefs', np.float32, (7, 8)),   # note that the slope coefficient was scaled up by 10000
                   ('rmse', np.float32, 7),
                   ('magnitude', np.float32, 7)])

    if coefs is not None:
        assert all(elem in coef_names for elem in coefs)
        assert all(elem in band_names for elem in coefs_bands)

    if not os.path.exists(out_path):
        os.makedirs(out_path)

    ranks_percore = int(np.ceil(n_blocks / n_cores))
    for i in range(ranks_percore):
        iblock = n_cores * i + rank
        if iblock >= n_blocks:
            break
        current_block_y = int(np.floor(iblock / n_block_x)) + 1
        current_block_x = iblock % n_block_x + 1
        if method == 'OBCOLD':
            filename = 'record_change_x{}_y{}_obcold.npy'.format(current_block_x, current_block_y)
        elif method == 'COLD':
            filename = 'record_change_x{}_y{}_cold.npy'.format(current_block_x, current_block_y)
        elif method == 'HybridCOLD':
            filename = 'record_change_x{}_y{}_hybridcold.npy'.format(current_block_x, current_block_y)

        block_folder = os.path.join(stack_path, 'block_x{}_y{}'.format(current_block_x, current_block_y))

        if timestamp == True:
            meta_files = [m for m in os.listdir(block_folder) if m.endswith('.json')]
            
            # sort image files by ordinal dates
            img_dates = []
            img_names = []
            
            # read metadata and
            for meta in meta_files:
                metadata = open(join(block_folder, meta))
                meta_config = json.load(metadata)
                ordinal_date = meta_config['ordinal_date']
                img_name = meta_config['image_name'] + '.npy'
                img_dates.append(ordinal_date)
                img_names.append(img_name)
            
            if year_lowbound is None:
                year_low_ordinal = min(img_dates)
                year_lowbound = pd.Timestamp.fromordinal(year_low_ordinal).year
            else:                
                year_low_ordinal = pd.Timestamp.toordinal(datetime.datetime(int(year_lowbound), 1, 1))
            
            img_dates, img_names = zip(*filter(lambda x: x[0] >= year_low_ordinal,
                                                zip(img_dates, img_names)))
            if year_highbound is None:
                year_high_ordinal = max(img_dates)                
                year_highbound = pd.Timestamp.fromordinal(year_high_ordinal).year
            else:
                year_high_ordinal = pd.Timestamp.toordinal(datetime.datetime(int(year_highbound + 1), 1, 1))
            
            img_dates, img_names = zip(*filter(lambda x: x[0] < year_high_ordinal,
                                                   zip(img_dates, img_names)))
            img_dates = sorted(img_dates)
            img_names = sorted(img_names)
            ordinal_day_list = img_dates
            results_block = [np.full((block_height, block_width), -9999, dtype=np.int16)
                             for t in range(len(ordinal_day_list))]

            if coefs is not None:
                results_block_coefs = np.full(
                    (block_height, block_width, len(coefs) * len(coefs_bands),
                     len(ordinal_day_list)), -9999, dtype=np.float32)
                
    # assemble     
    if coefs is not None:
        for day in range(len(ordinal_day_list)):
            tmp_map_blocks = [np.load(
                os.path.join(out_path, 'tmp_coefmap_block{}_{}.npy'.format(x + 1, ordinal_day_list[day])))
                for x in range(n_blocks)]

            results = np.hstack(tmp_map_blocks)
            results = np.vstack(np.hsplit(results, n_block_x))
            ninput = 0
            for band_idx, band_name in enumerate(coefs_bands):
                for coef_index, coef in enumerate(coefs):
                    # if coef == 'cv':
                    #     results[results == -9999.0] = 0
                        
                    kwcoco_img_name = img_names[day]
                    band = BAND_INFO[band_name]
                    outname = '%s_%s_%s_%s.tif' % (
                    kwcoco_img_name[:-4], band, method, coef)
                    outfile = os.path.join(out_path, outname)

                    outdriver1 = gdal.GetDriverByName("GTiff")
                    outdata = outdriver1.Create(outfile, vid_w, vid_h, 1, gdal.GDT_Float32)
                    outdata.GetRasterBand(1).WriteArray(results[:vid_h, :vid_w, ninput])
                    outdata.FlushCache()
                    outdata.SetGeoTransform(new_gdal_transform)
                    outdata.FlushCache()
                    outdata.SetProjection(proj)
                    outdata.FlushCache()
                    ninput = ninput + 1

            for x in range(n_blocks):
                os.remove(
                    os.path.join(out_path, 'tmp_coefmap_block{}_{}.npy'.format(x + 1, ordinal_day_list[day])))
    
    logger.info('Generated COLD output geotiff')
    
    logger.info('Starting adding new asset to kwcoco json')
    
    # add new asset to each image
    for image_id in coco_dset.images():
        # Create a CocoImage object for each image.
        coco_image: kwcoco.CocoImage = coco_dset.coco_image(image_id)

        image_name = coco_image.img['name']
        # img_w = coco_image.img['width']
        # img_h = coco_image.img['height']

        asset_w = vid_w
        asset_h = vid_h
        
        for band_name in band_names:
            for coef in coef_names:
                band = BAND_INFO[band_name]
                new_fpath = os.path.join(out_path, f'{image_name}_{band}_{method}_{coef}.tif')
                if os.path.exists(new_fpath):
                    channels = kwcoco.ChannelSpec.coerce(f'{band}_{method}_{coef}')
                    
                    # COLD output was wrote based on transform information of coco_dset, so it aligned
                    warp_aux_to_img = coco_image.warp_img_from_vid
                    
                    # Use the CocoImage helper which will augment the coco dictionary with
                    # your information.
                    coco_image.add_asset(new_fpath, channels=channels, width=asset_w,
                                            height=asset_h, warp_aux_to_img=warp_aux_to_img) 
                    logger.info(f'Added to the asset {new_fpath}')
    
    # Write a modified kwcoco.json file
    coco_dset.fpath = mod_coco_fpath
    coco_dset.dump()
   