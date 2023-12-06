"""
This is step 2/4 in predict.py and the step that runs pycold

SeeAlso

* predict.py

* prepare_kwcoco.py

* tile_processing_kwcoco.py

* export_cold_result_kwcoco.py

* assemble_cold_result_kwcoco.py

This script is for running COLD algorithm with kwcoco dataset.
See original code: ~/code/pycold/src/python/pycold/imagetool/tile_processing.py
"""

import json
import numpy as np
import os
import pandas as pd
import scriptconfig as scfg
import time

import pycold  # NOQA

from datetime import datetime as datetime_cls
from pathlib import Path
from pytz import timezone
from scipy.stats import chi2

from pycold import cold_detect
from pycold.ob_analyst import ObjectAnalystHPC
from pycold.utils import get_rowcol_intile, get_doy, assemble_cmmaps

try:
    from xdev import profile
except ImportError:
    from ubelt import identity as profile


class TileProcessingKwcocoConfig(scfg.DataConfig):
    """
    The docstring will be the description in the CLI help
    """
    rank = scfg.Value(None, help='rank id')
    n_cores = scfg.Value(None, help='total cores assigned (parent context, not workers used by this process)')
    stack_path = scfg.Value(None, help='directory of stacked data')
    reccg_path = scfg.Value(None, help='directory where cold record will be saved')
    method = scfg.Value('COLD', choices=['COLD', 'HybridCOLD', 'OBCOLD'], help='type of COLD algorithms, e.g., COLD, HybridCOLD, OBCOLD')
    b_c2 = scfg.Value(True, help='indicate if it is c2 or not')
    prob = scfg.Value(0.99, help='change probability of chi-distribution, e.g., 0.99')
    conse = scfg.Value(6, help='consecutive observation to confirm change, e.g., 6')
    cm_interval = scfg.Value(60, help='CM output inverval, e.g., 60')


@profile
def tile_process_main(cmdline=1, **kwargs):
    """
    Args:
        n_cores (type=int): _description_
        stack_path (_type_): _description_
        reccg_path (_type_): _description_
        method (_type_): _description_
        year_lowbound (_type_): _description_
        year_highbound (_type_): _description_
        b_c2 (_type_): _description_
        cmdline (int, optional): _description_. Defaults to 1.

    Ignore:
        python -m geowatch.tasks.cold.tile_processing_kwcoco --help
        TEST_COLD=1 xdoctest -m geowatch.tasks.cold.tile_processing_kwcoco tile_process_main

    Example:
        >>> # xdoctest: +REQUIRES(env:TEST_COLD)
        >>> from geowatch.tasks.cold.tile_processing_kwcoco import tile_process_main
        >>> from geowatch.tasks.cold.tile_processing_kwcoco import *
        >>> import ubelt as ub
        >>> kwargs= dict(
        >>>    rank = 1,
        >>>    n_cores = 1,
        >>>    stack_path = ub.Path('/gpfs/scratchfs1/zhz18039/jws18003/kwcoco/stacked/KR_R001'),
        >>>    reccg_path = ub.Path('/gpfs/scratchfs1/zhz18039/jws18003/kwcoco/reccg/KR_R001'),
        >>>    method = 'COLD',
        >>>    b_c2 = True,
        >>>    prob = 0.99,
        >>>    conse = 6,
        >>>    cm_interval = 60,
        >>> )
        >>> cmdline=0
        >>> tile_process_main(cmdline, **kwargs)
    """
    # Hacky way to pass in progress manager
    pman = kwargs.pop('pman', None)

    # setting config
    config_in = TileProcessingKwcocoConfig.cli(cmdline=cmdline, data=kwargs)
    rank = config_in['rank']
    n_cores = config_in['n_cores']
    stack_path = Path(config_in['stack_path'])
    reccg_path = Path(config_in['reccg_path'])
    method = config_in['method']
    b_c2 = config_in['b_c2']
    prob = config_in['prob']
    conse = config_in['conse']
    cm_output_interval = config_in['cm_interval']

    config = read_json_metadata(stack_path)
    n_cols = config['padded_n_cols']
    n_rows = config['padded_n_rows']
    n_block_x = config['n_block_x']
    n_block_y = config['n_block_y']
    block_width = int(n_cols / n_block_x)  # width of a block
    block_height = int(n_rows / n_block_y)  # height of a block
    year_lowbound = None
    year_highbound = None

    tz = timezone('US/Eastern')
    start_time = datetime_cls.now(tz)

    # Define year_low_ordinal and year_high_ordinal to filter year for COLD processing
    if year_lowbound is None:
        year_lowbound = 0
    else:
        year_lowbound = pd.Timestamp.toordinal(datetime_cls(int(year_lowbound), 1, 1))

    if year_highbound is None:
        year_highbound = 0
    else:
        year_highbound = pd.Timestamp.toordinal(datetime_cls(int(year_highbound + 1), 1, 1))

    if (n_cols % block_width != 0) or (n_rows % block_height != 0):
        print('padded_n_cols, padded_n_rows must be divisible respectively by block_width, block_height! Please double '
              'check your config yaml')
        exit()

    # set up additional parameters for obcold
    if method == 'OBCOLD':
        # we need read 'global starting date' to save CM which will be only used for ob-cold
        try:
            starting_date, n_cm_maps = reading_start_dates_nmaps(stack_path, year_lowbound, year_highbound,
                                                                 cm_output_interval)
            year_lowbound = pd.Timestamp.fromordinal(starting_date).year
            year_highbound = pd.Timestamp.fromordinal(
                starting_date + (n_cm_maps - 1) * cm_output_interval).year
        except IOError:
            print(f"reading start dates errors: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
            exit()

    # logging and folder preparation
    # if rank == 0:
    reccg_path.mkdir(parents=True, exist_ok=True)
    if method == 'OBCOLD':
        cm_maps_fpath = reccg_path / 'cm_maps'
        cm_maps_fpath.mkdir(parents=True, exist_ok=True)
    print(f"The per-pixel time series processing begins: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")

    if not stack_path.exists():
        print("Failed to locate stack folders. The program ends: "
                f"{datetime_cls.now(tz).strftime('%Y-%m-%d %H:%M:%S')}")
        return

    #########################################################################
    #                        per-pixel COLD procedure                       #
    #########################################################################
    threshold = chi2.ppf(prob, 5)

    nblock_eachcore = int(np.ceil(n_block_x * n_block_y * 1.0 / n_cores))
    i_iter = range(nblock_eachcore + 1)
    if pman is not None:
        i_iter = pman.progiter(i_iter, desc=f'Process Tile \\[rank {rank}]',
                               total=nblock_eachcore, transient=True)
    for i in i_iter:
        block_id = n_cores * i + rank  # started from 1, i.e., rank, rank + n_cores, rank + 2 * n_cores
        if block_id > n_block_x * n_block_y :
            break
        block_y = int((block_id - 1) / n_block_x ) + 1  # note that block_x and block_y start from 1
        block_x = int((block_id - 1) % n_block_x ) + 1

        finished_fpath = reccg_path / f'COLD_block{block_id}_finished.txt'
        if finished_fpath.exists():
            now = datetime_cls.now(tz).strftime('%Y-%m-%d %H:%M:%S')
            print(f"Per-pixel COLD processing is finished for block_x{block_x}_y{block_y} ({now})")
            continue
        img_tstack, img_dates_sorted = get_stack_date(block_x, block_y, stack_path, year_lowbound,
                                                      year_highbound)

        # Define empty list
        result_collect = []
        date_collect = []
        CM_collect = []

        if img_tstack is None:  # empty block
            if method == 'OBCOLD':
                for pos in range(block_width * block_height):
                    CM_collect.append(np.full(n_cm_maps, -9999, dtype=np.short))
                    date_collect.append(np.full(n_cm_maps, -9999, dtype=np.short))
                np.save(reccg_path / f'CM_date_x{block_x}_y{block_y}.npy', np.hstack(date_collect))
                np.save(reccg_path / f'CM_x{block_x}_y{block_y}.npy', np.hstack(CM_collect))

        else:
            # start looping every pixel in the block
            if method == "COLD" or method == "HybridCOLD" or method == "OBCOLD":
                for pos in range(block_width * block_height):
                    original_row, original_col = get_rowcol_intile(pos, block_width,
                                                                   block_height, block_x, block_y)
                    try:
                        if method == 'OBCOLD':
                            [cold_result, CM, CM_date] = cold_detect(img_dates_sorted,
                                                                     img_tstack[pos, 0, :].astype(np.int64),
                                                                     img_tstack[pos, 1, :].astype(np.int64),
                                                                     img_tstack[pos, 2, :].astype(np.int64),
                                                                     img_tstack[pos, 3, :].astype(np.int64),
                                                                     img_tstack[pos, 4, :].astype(np.int64),
                                                                     img_tstack[pos, 5, :].astype(np.int64),
                                                                     img_tstack[pos, 6, :].astype(np.int64),
                                                                     img_tstack[pos, 7, :].astype(np.int64),
                                                                     pos=n_cols * (original_row - 1) + original_col,
                                                                     conse=conse,
                                                                     starting_date=starting_date,
                                                                     n_cm=n_cm_maps, b_c2=b_c2,
                                                                     cm_output_interval=cm_output_interval,
                                                                     b_output_cm=True)
                        else:
                            cold_result = cold_detect(img_dates_sorted,
                                                      img_tstack[pos, 0, :].astype(np.int64),
                                                      img_tstack[pos, 1, :].astype(np.int64),
                                                      img_tstack[pos, 2, :].astype(np.int64),
                                                      img_tstack[pos, 3, :].astype(np.int64),
                                                      img_tstack[pos, 4, :].astype(np.int64),
                                                      img_tstack[pos, 5, :].astype(np.int64),
                                                      img_tstack[pos, 6, :].astype(np.int64),
                                                      img_tstack[pos, 7, :].astype(np.int64),
                                                      t_cg=threshold,
                                                      conse=conse, b_c2=b_c2,
                                                      pos=n_cols * (original_row - 1) + original_col)

                    except RuntimeError:
                        now = datetime_cls.now(tz).strftime('%Y-%m-%d %H:%M:%S')
                        print(f"COLD fails at original_row {original_row}, original_col {original_col} ({now})")
                    except Exception:
                        if method == 'OBCOLD':
                            CM = np.full(n_cm_maps, -9999, dtype=np.short)
                            CM_date = np.full(n_cm_maps, -9999, dtype=np.short)
                    else:
                        result_collect.append(cold_result)
                    finally:
                        if method == 'OBCOLD':
                            CM_collect.append(CM)
                            date_collect.append(CM_date)

                # save the dataset
                if len(result_collect) == 0:
                    with open(reccg_path / f'record_change_x{block_x}_y{block_y}_cold.status.json', 'w') as f:
                        json.dump({'status': 'failed', 'output': 'invalid'}, f)
                if len(result_collect) > 0:
                    if method == 'HybridCOLD':
                        np.save(reccg_path / f'record_change_x{block_x}_y{block_y}_hybridcold.npy',
                                np.hstack(result_collect))
                    elif method == 'COLD':
                        np.save(reccg_path / f'record_change_x{block_x}_y{block_y}_cold.npy',
                                np.hstack(result_collect))
                    with open(reccg_path / f'record_change_x{block_x}_y{block_y}_cold.status.json', 'w') as f:
                        json.dump({'status': 'completed', 'output': 'valid'}, f)
                if method == 'OBCOLD':
                    np.save(reccg_path / 'CM_date_x{block_x}_y{block_y}.npy', np.hstack(date_collect))
                    np.save(reccg_path / 'CM_x{block_x}_y{block_y}.npy', np.hstack(CM_collect))

            with open(reccg_path / f'COLD_block{block_id}_finished.txt', 'w'):
                pass

            now = datetime_cls.now(tz).strftime('%Y-%m-%d %H:%M:%S')
            print(f"Per-pixel COLD processing is finished for block_x{block_x}_y{block_y} ({now})")

    # wait for all cores to be finished
    if method == 'OBCOLD':
        while not is_finished_cold_blockfinished(reccg_path, n_block_x * n_block_y):
            time.sleep(30)

    # if rank == 1:
    #     cold_timepoint = datetime_cls.now(tz)
    # for i in os.listdir(reccg_path):
    #     if i.endswith('.txt'):
    #         os.remove(reccg_path / i)

    #################################################################################
    #                        the below is object-based process                      #
    #################################################################################
    if method == 'OBCOLD':
        # if seedmap_path is None:
        ob_analyst = ObjectAnalystHPC(config, starting_date=starting_date, stack_path=stack_path,
                                          result_path=reccg_path)
        # else:
        #     pyclassifier = PyClassifierHPC(config, record_path=reccg_path, year_lowbound=year_lowbound,
        #                                    year_uppbound=year_highbound, seedmap_path=seedmap_path)
        #     ob_analyst = ObjectAnalystHPC(config, starting_date=starting_date, stack_path=stack_path,
        #                                   result_path=reccg_path, thematic_path=reccg_path / 'feature_maps')
        if rank == 1:
            # need to create folders first
            # if seedmap_path is not None:
            #     pyclassifier.hpc_preparation()
            ob_analyst.hpc_preparation()

        #########################################################################
        #                        reorganize cm snapshots                        #
        #########################################################################

        if not is_finished_assemble_cmmaps(reccg_path / 'cm_maps', n_cm_maps,
                                           starting_date, cm_output_interval):
            if rank == 1:
                assemble_cmmaps(config, reccg_path, reccg_path / 'cm_maps', starting_date, n_cm_maps, 'CM',
                                clean=False)
            elif rank == 2:
                assemble_cmmaps(config, reccg_path, reccg_path / 'cm_maps', starting_date, n_cm_maps,
                                'CM_date',
                                clean=False)

            while not is_finished_assemble_cmmaps(reccg_path / 'cm_maps', n_cm_maps,
                                                  starting_date, cm_output_interval):
                time.sleep(15)

        #########################################################################
        #                      producing classification maps                    #
        #########################################################################
        # if seedmap_path is not None:  # we used thematic info
        #     if not pyclassifier.is_finished_step4_assemble():
        #         if rank == 1:
        #             print("Starts predicting features: {}".format(datetime_cls.now(tz).strftime('%Y-%m-%d %H:%M:%S')))
        #         for i in range(nblock_eachcore):
        #             if n_cores * i + rank > config['n_block_x'] * config['n_block_y']:
        #                 break
        #             pyclassifier.step1_feature_generation(block_id=n_cores * i + rank)

        #         if rank == 1:  # serial mode for producing rf
        #             pyclassifier.step2_train_rf()
        #             print("Training rf ends: {}".format(datetime_cls.now(tz).strftime('%Y-%m-%d %H:%M:%S')))

        #         for i in range(nblock_eachcore):
        #             if n_cores * i + rank > config['n_block_x'] * config['n_block_y']:
        #                 break
        #             pyclassifier.step3_classification(block_id=n_cores * i + rank)

        #         if rank == 1:  # serial mode for assemble
        #             pyclassifier.step4_assemble()
        #     while not pyclassifier.is_finished_step4_assemble():
        #         time.sleep(15)
        #     if rank == 1:
        #         print("Assemble classification map ends: {}".format(datetime_cls.now(tz).strftime('%Y-%m-%d %H:%M:%S')))
        #########################################################################
        #                      object-based image analysis                      #
        #########################################################################
        if not ob_analyst.is_finished_object_analysis(np.arange(starting_date,
                                                                starting_date + cm_output_interval * n_cm_maps,
                                                                cm_output_interval)):
            n_map_percore = int(np.ceil(n_cm_maps / n_cores))
            max_date = starting_date + (n_cm_maps - 1) * cm_output_interval
            for i in range(n_map_percore):
                if starting_date + (rank - 1 + i * n_cores) * cm_output_interval > max_date:
                    break
                date = starting_date + (rank - 1 + i * n_cores) * cm_output_interval
                ob_analyst.obia_execute(date)

            while not ob_analyst.is_finished_object_analysis(np.arange(starting_date,
                                                                       starting_date + cm_output_interval * n_cm_maps,
                                                                       cm_output_interval)):
                time.sleep(15)
        if rank == 1:
            print(f"OBIA ends: {datetime_cls.now(tz).strftime('%Y-%m-%d %H:%M:%S')}")

        #########################################################################
        #                        reconstruct change records                     #
        #########################################################################
        for i in range(nblock_eachcore):
            block_id = n_cores * i + rank  # started from 1, i.e., rank, rank + n_cores, rank + 2 * n_cores
            if block_id > n_block_x * n_block_y:
                break
            block_y = int((block_id - 1) / n_block_y) + 1  # note that block_x and block_y start from 1
            block_x = int((block_id - 1) % n_block_x) + 1
            img_tstack, img_dates_sorted = get_stack_date(block_x, block_y, stack_path)
            result_collect = ob_analyst.reconstruct_reccg(block_id=block_id,
                                                          img_stack=img_tstack,
                                                          img_dates_sorted=img_dates_sorted)
            ob_analyst.save_obcoldrecords(block_id=block_id, result_collect=result_collect)

    # if rank == 1:
    #     # tile_based report
    log = {
        'algorithm': method,
        'prob': prob,
        'conse': conse,
    }

    log_fpath = reccg_path / 'log.json'
    log_fpath.write_text(json.dumps(log))

    # if method == 'OBCOLD':
    #     tileprocessing_report(reccg_path / 'tile_processing_report.log',
    #                           stack_path, pycold.__version__, method, config, start_time, cold_timepoint, tz,
    #                           n_cores, starting_date, n_cm_maps, year_lowbound, year_highbound)
    # else:
    #     tileprocessing_report(reccg_path / 'tile_processing_report.log', stack_path, pycold.__version__,
    #                           method, config, start_time, cold_timepoint, tz, n_cores)
    # print("The whole procedure finished: {}".format(datetime_cls.now(tz).strftime('%Y-%m-%d %H:%M:%S')))


# def tileprocessing_report(result_log_path, stack_path, version, algorithm, config, startpoint, cold_timepoint, tz,
#                           n_cores, starting_date=0, n_cm_maps=0, year_lowbound=0, year_highbound=0):
#     """
#     output tile-based processing report
#     Parameters
#     ----------
#     result_log_path: string
#         outputted log path
#     stack_path: string
#         stack path
#     version: string
#     algorithm: string
#     config: dictionary structure
#     startpoint: a time point, when the program starts
#     tz: string, time zone
#     n_cores: the core number used
#     starting_date: the first date of the total dataset
#     n_cm_maps: the number of snapshots
#     year_lowbound: the low bound of year range
#     year_highbound: the upper bound of year range
#     Returns
#     -------
#     """
#     endpoint = datetime_cls.now(tz)
#     file = open(result_log_path, "w")
#     file.write("PYCOLD V{} \n".format(version))
#     file.write("Author: Su Ye(remoteseningsuy@gmail.com)\n")
#     file.write("Algorithm: {} \n".format(algorithm))
#     file.write("Starting_time: {}\n".format(startpoint.strftime('%Y-%m-%d %H:%M:%S')))
#     file.write("Change probability threshold: {}\n".format(change_probability))
#     file.write("Conse: {}\n".format(conse))
#     file.write("stack_path: {}\n".format(stack_path))
#     file.write("The number of requested cores: {}\n".format(n_cores))
#     file.write("The program starts at {}\n".format(startpoint.strftime('%Y-%m-%d %H:%M:%S')))
#     file.write("The COLD ends at {}\n".format(cold_timepoint.strftime('%Y-%m-%d %H:%M:%S')))
#     file.write("The program ends at {}\n".format(endpoint.strftime('%Y-%m-%d %H:%M:%S')))
#     file.write("The program lasts for {:.2f}mins\n".format((endpoint - startpoint) / datetime_cls.timedelta(minutes=1)))
#     file.close()
@profile
def read_json_metadata(stacked_path):
    for root, dirs, files in os.walk(stacked_path):
        for file in files:
            if file.endswith(".json"):
                json_path = os.path.join(root, file)

                with open(json_path, "r") as f:
                    metadata = json.load(f)
                    return metadata


@profile
def is_finished_cold_blockfinished(reccg_path, nblocks):
    """
    check if the COLD algorithm finishes all blocks

    Args:
        reccg_path (str): the path that save COLD results
        nblocks (int): the block number

    Returns:
        bool: True if all block finished
    """
    for n in range(nblocks):
        fpath = reccg_path / f'COLD_block{n + 1}_finished.txt'
        if not fpath.exists():
            return False
    return True


@profile
def get_stack_date(block_x, block_y, stack_path, year_lowbound=0, year_highbound=0, nband=8):
    """
    Args:
        block_x: block id at x axis
        block_y: block id at y axis
        stack_path: stack path
        year_lowbound: ordinal data of low bounds of selection date range
        year_highbound: ordinal data of upper bounds of selection date range

    Returns:
        Tuple:
        img_tstack, img_dates_sorted
        img_tstack - 3-d array (block_width * block_height, nband, nimage)
    """
    block_folder = stack_path / f'block_x{block_x}_y{block_y}'
    meta_files = [m for m in os.listdir(block_folder) if m.endswith('.json')]

    # sort image files by ordinal dates
    img_dates = []
    img_files = []

    # read metadata and
    for meta in meta_files:
        config = json.loads((block_folder / meta).read_text())
        ordinal_date = config['ordinal_date']
        img_name = config['image_name'] + '.npy'
        img_dates.append(ordinal_date)
        img_files.append(img_name)

    if len(img_files) == 0:
        return None, None

    sample_np = np.load(block_folder / img_files[0])
    block_width = sample_np.shape[1]
    block_height = sample_np.shape[0]

    if year_lowbound > 0:
        year_low_ordinal = pd.Timestamp.toordinal(datetime_cls(int(year_lowbound), 1, 1))
        img_dates, img_files = zip(*filter(lambda x: x[0] >= year_low_ordinal,
                                           zip(img_dates, img_files)))
    if year_highbound > 0:
        year_high_ordinal = pd.Timestamp.toordinal(datetime_cls(int(year_highbound + 1), 1, 1))
        img_dates, img_files = zip(*filter(lambda x: x[0] < year_high_ordinal,
                                           zip(img_dates, img_files)))

    files_date_zip = sorted(zip(img_dates, img_files))
    img_files_sorted = [x[1] for x in files_date_zip]
    img_dates_sorted = np.asarray([x[0] for x in files_date_zip])
    img_tstack = np.dstack([np.load(block_folder / f).reshape(block_width * block_height, nband)
                            for f in img_files_sorted])
    return img_tstack, img_dates_sorted


#########################################################################
#                     function for OB-COLD procedure                    #
#########################################################################


@profile
def reading_start_dates_nmaps(stack_path, year_lowbound, year_highbound, cm_interval):
    """
    Args:
        stack_path (str): stack_path for saving starting_last_dates.txt
        cm_interval (interval): day interval for outputting change magnitudes

    Returns:
        Tuple:
        (starting_date, n_cm_maps)
        starting_date - starting date is the first date of the whole dataset,
        n_cm_maps - the number of change magnitudes to be outputted per pixel per band
    """
    # read starting and ending dates, note that all blocks only has one starting and last date (mainly for obcold)
    try:
        block_folder = stack_path / 'block_x1_y1'
        meta_files = [m for m in os.listdir(block_folder) if m.endswith('.json')]

        # sort image files by ordinal dates
        img_dates = []

        # read metadata and
        for meta in meta_files:
            config = json.loads((block_folder / meta).read_text())
            ordinal_date = config['ordinal_date']
            img_dates.append(ordinal_date)
        sorted_img_dates = sorted(img_dates)

        if year_lowbound > 0:
            year_low_ordinal = pd.Timestamp.toordinal(datetime_cls(int(year_lowbound), 1, 1))
            img_dates = (lambda x: x >= year_low_ordinal, img_dates)
        if year_highbound > 0:
            year_high_ordinal = pd.Timestamp.toordinal(datetime_cls(int(year_highbound + 1), 1, 1))
            img_dates = (lambda x: x < year_high_ordinal, img_dates)

    except IOError:
        raise
    else:
        starting_date = sorted_img_dates[0]
        ending_date = sorted_img_dates[-1]
        n_cm_maps = int((ending_date - starting_date + 1) / cm_interval) + 1
        return starting_date, n_cm_maps


@profile
def is_finished_assemble_cmmaps(cmmap_path, n_cm, starting_date, cm_interval):
    """
    Args:
        cmmap_path: the path for saving change magnitude maps
        n_cm: the number of change magnitudes outputted per pixel
        starting_date: the starting date of the whole dataset
        cm_interval: the day interval for outputting change magnitudes

    Returns:
        bool: True -> assemble finished
    """
    for count in range(n_cm):
        ordinal_date = starting_date + count * cm_interval
        year = pd.Timestamp.fromordinal(ordinal_date).year
        doy = get_doy(ordinal_date)
        cm_fpath = cmmap_path / f'CM_maps_{str(ordinal_date)}_{year}{doy}.npy'
        cm_date_fpath = cmmap_path / 'CM_date_maps_{str(ordinal_date)}_{year}{doy}.npy'
        if not cm_fpath.exists():
            return False
        if not cm_date_fpath.exists():
            return False
    return True


if __name__ == '__main__':
    tile_process_main()
