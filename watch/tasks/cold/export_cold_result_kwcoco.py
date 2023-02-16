"""
This script is for exporting COLD algorithm results (change vector, coefficients, RMSEs)
to geotiff raster with kwcoco dataset.
See original code: ~/code/pycold/src/python/pycold/imagetool/export_change_map.py
"""

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

class ExportColdKwcocoConfig(scfg.DataConfig):
    """
    The docstring will be the description in the CLI help
    """
    stack_path = scfg.Value(None, help='folder directory of stacked data')
    reccg_path = scfg.Value(None, help='folder directory of cold processing result')
    out_path = scfg.Value(None, help='folder directory of output geotiff image')
    reference_path = scfg.Value(None, help='file path of reference image')
    meta_fpath = scfg.Value(None, help='file path of metadata json created by prepare_kwcoco script')    
    region_id = scfg.Value(None, help='region name of Kwcoco data, e.g., US_C000')    
    year_lowbound = scfg.Value(None, help='min year for saving geotiff, e.g., 2017')
    year_highbound = scfg.Value(None, help='max year for saving geotiff, e.g., 2022')
    coefs = scfg.Value(None, help="list of COLD coefficients for saving geotiff, e.g., ['a0', 'c1', 'a1', 'b1', 'a2', 'b2', 'a3', 'b3', 'cv', 'rmse']")
    coefs_bands = scfg.Value(None, help='indicate the ba_nds for output coefs_bands, e.g., [0, 1, 2, 3, 4, 5]')
    timestamp = scfg.Value(True, help='True: exporting cold result by timestamp, False: exporting cold result by year, Default is False')   
    # rank = scfg.Value("MPI",  help='rank id')
    # n_cores = scfg.Value("MPI" ,help='number of processor')


# def main(stack_path, reccg_path, out_path, reference_path, yaml_path, method, region, probability, conse, year_lowbound,
        #  year_highbound, coefs, coefs_bands, timestamp)    
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
    >>>    stack_path = "/gpfs/scratchfs1/zhz18039/jws18003/kwcoco/stacked/US_C000",
    >>>    reccg_path = "/gpfs/scratchfs1/zhz18039/jws18003/kwcoco/reccg/US_C000",
    >>>    out_path = "/gpfs/scratchfs1/zhz18039/jws18003/kwcoco/reccg/US_C000",
    >>>    reference_path = "/home/jws18003/Document/kwcoco_working/US_C000_rowcol2.tif",
    >>>    meta_fpath = '/gpfs/scratchfs1/zhz18039/jws18003/kwcoco/stacked/US_C000/block_x9_y9/crop_20210716T150000Z_N38.904157W077.594580_N39.117177W077.375621_L8_0.json',
    >>>    region_id = "US_C000",
    >>>    coefs = ['cv'],
    >>>    year_lowbound = 2027,
    >>>    year_highbound = 2027,
    >>>    coefs_bands = [5],
    >>>    timestamp = True,
    >>>    )    
    >>> cmdline=0    
    >>> main(cmdline, **kwargs)
    """ 
    rank = 0
    n_cores = 1
    config_in = ExportColdKwcocoConfig.legacy(cmdline=cmdline, data=kwargs)
    # rank = config_in['rank']
    # n_cores = config_in['n_cores']
    stack_path = config_in['stack_path']
    reccg_path = config_in['reccg_path']
    out_path = config_in['out_path']
    reference_path = config_in['reference_path']
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
    
    ##################################################################
    # HELP: Get info from coco_image, not additional reference image #
    ##################################################################
    
    # I used reference image to get information about GeoTransform and Projection to use these info when I wrote COLD output to geotiff.
    # However, I want to find a way not to use reference image. Instead, I want to get these info from kwcoco image...    
    ref_image = gdal.Open(reference_path, gdal.GA_ReadOnly)
    trans = ref_image.GetGeoTransform()
    proj = ref_image.GetProjection()
    
    # Get original transform from projection to image space
    c, a, b, f, d, e = trans
    original = kwimage.Affine(np.array([
        [a, b, c],
        [d, e, f],
        [0, 0, 1],
    ]))
    
    # I want to use following line not to use reference image anymore...
    demo_dpath = ub.Path.appdir('/home/jws18003/data/dvc-repos/smart_data_dvc/Aligned-Drop4-2022-08-08-TA1-S2-L8-ACC/US_C000').ensuredir()
    dset = watch.coerce_kwcoco(demo_dpath)
    for image_id in dset.images():
        coco_image: kwcoco.CocoImage = dset.coco_image(image_id)
        img_w = coco_image.img['width']
        img_h = coco_image.img['height']
        
        # Get the modifier transform to move from image space to video space
        warp_vid_from_img = kwimage.Affine.coerce(coco_image.img['warp_img_to_vid']).inv()
        break
    
    # Combine transforms to get a new transform that goes
    # from the projection to video space
    new_geotrans =  original @ warp_vid_from_img
    a, b, c, d, e, f, g, h, i = np.array(new_geotrans).ravel().tolist()
    new_gdal_transform = (c, a, b, f, d, e) 
    
    ########################################################################
    
    # define variables
    n_cols = config['padded_n_cols']
    n_rows = config['padded_n_rows']
    n_block_x = config['n_block_x']
    n_block_y = config['n_block_y']        
    block_width = int(n_cols / n_block_x)  # width of a block
    block_height = int(n_rows / n_block_y)  # height of a block
    n_blocks = n_block_x * n_block_y  # total number of blocks   
    # vid_w = config['n_cols'] 
    # vid_h = config['n_rows'] 
    
    log = open(os.path.join(reccg_path, 'log.json'))
    cold_param = json.load(log)
    method = cold_param['algorithm']
    prob = cold_param['prob']
    conse = cold_param['conse']
        
    coef_names = ['a0', 'c1', 'a1', 'b1', 'a2', 'b2', 'a3', 'b3', 'cv', 'rmse']
    band_names = [0, 1, 2, 3, 4, 5, 6]
    SLOPE_SCALE = 10000

    BAND_INFO = {0: 'blue',
                1: 'green',
                2: 'red',
                3: 'nir',
                4: 'swir16',
                5: 'swir22'}
        
    # NOTE: tmp file will be saved in reccg_path
    if method == 'OBCOLD':
        reccg_path = os.path.join(reccg_path, 'obcold')
        if timestamp == True:
            tmp_path = os.path.join(reccg_path, 'obcold_maps', 'by_timestamp')
        else:
            tmp_path = os.path.join(reccg_path, 'obcold_maps', 'by_year')

    elif method == 'COLD' or method == 'HybridCOLD':
        if timestamp == True:
            tmp_path = os.path.join(reccg_path, 'cold_maps', 'by_timestamp')
        else:
            tmp_path = os.path.join(reccg_path, 'cold_maps', 'by_year')
            
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

    # if rank == 0:
    if not os.path.exists(out_path):
        os.makedirs(out_path)
    if not os.path.exists(tmp_path):
        os.makedirs(tmp_path)
    # MPI mode
    # trans = comm.bcast(trans, root=0)
    # proj = comm.bcast(proj, root=0)
    # cols = comm.bcast(cols, root=0)
    # rows = comm.bcast(rows, root=0)
    # config = comm.bcast(config, root=0)

    ranks_percore = int(np.ceil(n_blocks / n_cores))
    print(ranks_percore)
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

            if year_lowbound > 0:
                year_low_ordinal = pd.Timestamp.toordinal(datetime.datetime(int(year_lowbound), 1, 1))
                img_dates, img_names = zip(*filter(lambda x: x[0] >= year_low_ordinal,
                                                   zip(img_dates, img_names)))
            if year_highbound > 0:
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

            print('processing the rec_cg file {}'.format(os.path.join(reccg_path, filename)))
            if not os.path.exists(os.path.join(reccg_path, filename)):
                print('the rec_cg file {} is missing'.format(os.path.join(reccg_path, filename)))                

        cold_block = np.array(np.load(os.path.join(reccg_path, filename)), dtype=dt)

        cold_block.sort(order='pos')
        current_processing_pos = cold_block[0]['pos']
        current_dist_type = 0

        for count, curve in enumerate(cold_block):
            if curve['pos'] != current_processing_pos:
                current_processing_pos = curve['pos']
                current_dist_type = 0

            if curve['change_prob'] < 100 or curve['t_break'] == 0 or count == (len(cold_block) - 1):  # last segment
                continue

            i_col = int((curve["pos"] - 1) % n_cols) - \
                    (current_block_x - 1) * block_width
            i_row = int((curve["pos"] - 1) / n_cols) - \
                    (current_block_y - 1) * block_height
            if i_col < 0:
                dat_pth = '?'
                print('Processing {} failed: i_row={}; i_col={} for {}'.format(filename, i_row, i_col, dat_pth))
                return

            if method == 'OBCOLD':
                current_dist_type = getcategory_obcold(cold_block, count, current_dist_type)
            else:
                current_dist_type = getcategory_cold(cold_block, count)
            break_year = pd.Timestamp.fromordinal(curve['t_break']).year
            if break_year < year_lowbound or break_year > year_highbound:
                continue
            results_block[break_year - year_lowbound][i_row][i_col] = current_dist_type * 1000 + curve['t_break'] - \
                (pd.Timestamp.toordinal(datetime.date(break_year, 1, 1))) + 1

        if coefs is not None:
            cold_block_split = np.split(cold_block, np.argwhere(np.diff(cold_block['pos']) != 0)[:, 0] + 1)
            for element in cold_block_split:
                # the relative column number in the block
                i_col = int((element[0]["pos"] - 1) % n_cols) - \
                        (current_block_x - 1) * block_width
                i_row = int((element[0]["pos"] - 1) / n_cols) - \
                        (current_block_y - 1) * block_height

                for band_idx, band in enumerate(coefs_bands):
                    feature_row = extract_features(element, band, ordinal_day_list, -9999, timestamp,
                                                    feature_outputs=coefs)
                    for index, coef in enumerate(coefs):
                        results_block_coefs[i_row][i_col][index + band_idx * len(coefs)][:] = \
                            feature_row[index]

        # save the temp dataset out
        if timestamp == True:
            for day in range(len(ordinal_day_list)):
                if coefs is not None:
                    outfile = os.path.join(tmp_path,
                                            'tmp_coefmap_block{}_{}.npy'.format(iblock + 1,
                                                                                ordinal_day_list[day]))
                    np.save(outfile, results_block_coefs[:, :, :, day])

    # MPI mode (wait for all processes)
    # comm.Barrier()

    # assemble     
    if coefs is not None:
        for day in range(len(ordinal_day_list)):
            tmp_map_blocks = [np.load(
                os.path.join(tmp_path, 'tmp_coefmap_block{}_{}.npy'.format(x + 1, ordinal_day_list[day])))
                for x in range(n_blocks)]

            results = np.hstack(tmp_map_blocks)
            results = np.vstack(np.hsplit(results, n_block_x))
            ninput = 0
            for band_idx, band_name in enumerate(coefs_bands):
                for coef_index, coef in enumerate(coefs):
                    if coef == 'cv':
                        results[results == -9999.0] = 0
                    # FIXME: Best name for outputs...?
                    kwcoco_img_name = img_names[day]
                    print(kwcoco_img_name)
                    # date = str(datetime.date.fromordinal(ordinal_day_list[day]))
                    band = BAND_INFO[band_name]
                    outname = '%s_%s_%s_%s.tif' % (
                    kwcoco_img_name[:-4], band, method, coef)
                    print(outname)
                    # out_fpath = os.path.join(out_path, region, 'L8', 'affine_warp', kwcoco_img_name[:-4])
                    
                    outfile = os.path.join(tmp_path, outname)
                    outdriver1 = gdal.GetDriverByName("GTiff")
                    outdata = outdriver1.Create(outfile, img_w, img_h, 1, gdal.GDT_Float32)
                    outdata.GetRasterBand(1).WriteArray(results[:img_h, :img_w, ninput])
                    outdata.FlushCache()
                    outdata.SetGeoTransform(new_gdal_transform)
                    outdata.FlushCache()
                    outdata.SetProjection(proj)
                    outdata.FlushCache()
                    ninput = ninput + 1

            for x in range(n_blocks):
                os.remove(
                    os.path.join(tmp_path, 'tmp_coefmap_block{}_{}.npy'.format(x + 1, ordinal_day_list[day])))
    
    logger.info('Generated COLD output geotiff')
    
    logger.info('Starting adding new asset to kwcoco json')
    # add new asset
    dset = watch.coerce_kwcoco(demo_dpath)
    # We are going to add a new asset to each image.
    for image_id in dset.images():
        # Create a CocoImage object for each image.
        coco_image: kwcoco.CocoImage = dset.coco_image(image_id)

        image_name = coco_image.img['name']

        img_w = coco_image.img['width']
        img_h = coco_image.img['height']

        asset_w = img_w
        asset_h = img_h
        
        new_fpath = os.path.join(tmp_path, '%s_swir16_COLD_cv.tif'%(image_name))
        if os.path.exists(new_fpath):
            
            ###################################################################################################################  
            # QUESTION: COLD output is generated in separate band. Then Do I need to go through this process by each channel? #
            ###################################################################################################################
            # For each band, possible COLD output would be 'cv', 'a0', 'a1', 'b1', 'c1', 'rmse'.            
            channels = 'swir16_COLD_cv'

            # We need the transform that warps from asset space to image space
            # FIXME: I am not sure which variable is the correct one for warp_aux_to_img, 1) coco_image.img['warp_img_to_vid']).inv(), 2) 'warp_vid_from_img', 3) 'coco_image['warp_img_to_vid'] or other...?
            warp_aux_to_img = coco_image.img(['warp_img_to_vid']).inv() #kwimage.Affine.eye()

            # Use the CocoImage helper which will augment the coco dictionary with
            # your information.
            coco_image.add_asset(new_fpath, channels=channels, width=asset_w,
                                    height=asset_h, warp_aux_to_img=warp_aux_to_img) 
            logger.info(f'Added to the asset {new_fpath}')
            

# copy from /pycold/src/python/pycold/pyclassifier.py because MPI has conflicts with the pycold package in UCONN HPC.
# Dirty approach!
def extract_features(cold_plot, band, ordinal_day_list, nan_val, timestamp, feature_outputs=['a0', 'a1', 'b1']):
    """
    generate features for classification based on a plot-based rec_cg and a list of days to be predicted
    Parameters
    ----------
    cold_plot: nested array
        plot-based rec_cg
    band: integer
        the predicted band number range from 0 to 6
    ordinal_day_list: list
        a list of days that this function will predict every days as a list as output
    nan_val: integer
        NA value assigned to the output
    feature_outputs: a list of outputted feature name
        it must be within [a0, c1, a1, b1, a2, b2, a3, b3, cv, rmse]
    Returns
    -------
        feature: a list (length = n_feature) of 1-array [len(ordinal_day_list)]
    """
    features = [np.full(len(ordinal_day_list), nan_val, dtype=np.double) for x in range(len(feature_outputs))]
    SLOPE_SCALE = 10000
    for index, ordinal_day in enumerate(ordinal_day_list):
        for idx, cold_curve in enumerate(cold_plot):
            if idx == len(cold_plot) - 1:
                last_year = pd.Timestamp.fromordinal(cold_plot[idx]['t_end']).year
                max_days = datetime.date(last_year, 12, 31).toordinal()
            else:
                max_days = cold_plot[idx + 1]['t_start']
            break_year = pd.Timestamp.fromordinal(cold_curve['t_break']).year if(cold_curve['t_break'] > 0 and cold_curve['change_prob'] == 100) else -9999

            if cold_curve['t_start'] <= ordinal_day < max_days:
                for n, feature in enumerate(feature_outputs):
                    if feature not in feature_outputs:
                        raise Exception('the outputted feature must be in [a0, c1, a1, b1,a2, b2, a3, b3, cv, rmse]')
                    if feature == 'a0':
                        features[n][index] = cold_curve['coefs'][band][0] + cold_curve['coefs'][band][1] * \
                                             ordinal_day / SLOPE_SCALE
                        if np.isnan(features[n][index]):
                            features[n][index] = 0
                    elif feature == 'c1':
                        features[n][index] = cold_curve['coefs'][band][1] / SLOPE_SCALE
                        if np.isnan(features[n][index]):
                            features[n][index] = 0
                    elif feature == 'a1':
                        features[n][index] = cold_curve['coefs'][band][2]
                        if np.isnan(features[n][index]):
                            features[n][index] = 0
                    elif feature == 'b1':
                        features[n][index] = cold_curve['coefs'][band][3]
                        if np.isnan(features[n][index]):
                            features[n][index] = 0
                    elif feature == 'a2':
                        features[n][index] = cold_curve['coefs'][band][4]
                        if np.isnan(features[n][index]):
                            features[n][index] = 0
                    elif feature == 'b2':
                        features[n][index] = cold_curve['coefs'][band][5]
                        if np.isnan(features[n][index]):
                            features[n][index] = 0
                    elif feature == 'a3':
                        features[n][index] = cold_curve['coefs'][band][6]
                        if np.isnan(features[n][index]):
                            features[n][index] = 0
                    elif feature == 'b3':
                        features[n][index] = cold_curve['coefs'][band][7]
                        if np.isnan(features[n][index]):
                            features[n][index] = 0
                    elif feature == 'rmse':
                        features[n][index] = cold_curve['rmse'][band]
                        if np.isnan(features[n][index]):
                            features[n][index] = 0                    
                break

        if 'cv' in feature_outputs:
            # ordinal_day_years = [pd.Timestamp.fromordinal(day).year for day in ordinal_day_list]
            for index, ordinal_day in enumerate(ordinal_day_list):
                ordinal_year = pd.Timestamp.fromordinal(ordinal_day).year
                for cold_curve in cold_plot:
                    if (cold_curve['t_break'] == 0) or (cold_curve['change_prob'] != 100):
                        continue
                    break_year = pd.Timestamp.fromordinal(cold_curve['t_break']).year
                    if timestamp == True:
                        if ordinal_day == cold_curve['t_break']:
                            features[feature_outputs.index('cv')][index] = cold_curve['magnitude'][band]
                            continue
                    else:
                        if break_year == ordinal_year:
                            features[feature_outputs.index('cv')][index] = cold_curve['magnitude'][band]
                            continue

    return features


def getcategory_cold(cold_plot, i_curve):
    t_c = -200
    if cold_plot[i_curve]['magnitude'][3] > t_c and cold_plot[i_curve]['magnitude'][2] < -t_c and \
            cold_plot[i_curve]['magnitude'][4] < -t_c:
        if cold_plot[i_curve + 1]['coefs'][3, 1] > np.abs(cold_plot[i_curve]['coefs'][3, 1]) and \
                cold_plot[i_curve + 1]['coefs'][2, 1] < -np.abs(cold_plot[i_curve]['coefs'][2, 1]) and \
                cold_plot[i_curve + 1]['coefs'][4, 1] < -np.abs(cold_plot[i_curve]['coefs'][4, 1]):
            return 3  # aforestation
        else:
            return 2  # regrowth
    else:
        return 1  # land disturbance

def getcategory_obcold(cold_plot, i_curve, last_dist_type):
    t_c = -250
    if cold_plot[i_curve]['magnitude'][3] > t_c and cold_plot[i_curve]['magnitude'][2] < -t_c and \
            cold_plot[i_curve]['magnitude'][4] < -t_c:
        if cold_plot[i_curve + 1]['coefs'][3, 1] > np.abs(cold_plot[i_curve]['coefs'][3, 1]) and \
                cold_plot[i_curve + 1]['coefs'][2, 1] < -np.abs(cold_plot[i_curve]['coefs'][2, 1]) and \
                cold_plot[i_curve + 1]['coefs'][4, 1] < -np.abs(cold_plot[i_curve]['coefs'][4, 1]):
            return 3  # aforestation
        else:
            return 2  # regrowth
    else:
        if i_curve > 0:
            if (cold_plot[i_curve]['t_break'] - cold_plot[i_curve - 1]['t_break'] > 365.25 * 5) or (
                    last_dist_type != 1):
                return 1
            flip_count = 0
            for b in range(5):
                if cold_plot[i_curve]['magnitude'][b + 1] * cold_plot[i_curve - 1]['magnitude'][b + 1] < 0:
                    flip_count = flip_count + 1
            if flip_count >= 4:
                return 4
            else:
                return 1
        else:
            return 1  # land disturbance

if __name__ == '__main__':    
    main()