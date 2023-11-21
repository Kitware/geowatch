"""
This is step 3/4 in predict.py


SeeAlso:

    predict.py

    prepare_kwcoco.py

    tile_processing_kwcoco.py

    export_cold_result_kwcoco.py *

    assemble_cold_result_kwcoco.py

This script is for exporting COLD algorithm results (change vector, coefficients, RMSEs)
to geotiff raster with kwcoco dataset.
See original code: ~/code/pycold/src/python/pycold/imagetool/export_change_map.py
"""

import os
import numpy as np
import pandas as pd
import datetime as datetime_mod
import json
import scriptconfig as scfg
import logging
import itertools
import ubelt as ub
import pytz
from datetime import datetime as datetime_cls
import gc
import kwcoco
logger = logging.getLogger(__name__)


try:
    from xdev import profile
except ImportError:
    from ubelt import identity as profile


class ExportColdKwcocoConfig(scfg.DataConfig):
    """
    The docstring will be the description in the CLI help
    """
    rank = scfg.Value(None, help='rank id')
    n_cores = scfg.Value(None, help='total cores assigned')
    stack_path = scfg.Value(None, help='folder directory of stacked data')
    reccg_path = scfg.Value(
        None, help='folder directory of cold processing result')
    meta_fpath = scfg.Value(
        None, help='file path of metadata json created by prepare_kwcoco script')
    combined_coco_fpath = scfg.Value(None, help=ub.paragraph(
        '''
        a path to a file to combined kwcoco file
        '''))
    year_lowbound = scfg.Value(
        None, help='min year for saving geotiff, e.g., 2017')
    year_highbound = scfg.Value(
        None, help='max year for saving geotiff, e.g., 2022')
    coefs = scfg.Value(
        None,
        type=str,
        help="list of COLD coefficients for saving geotiff, e.g., a0,c1,a1,b1,a2,b2,a3,b3,cv,rmse")
    coefs_bands = scfg.Value(
        None,
        type=str,
        help='indicate the ba_nds for output coefs_bands, e.g., 0,1,2,3,4,5')
    timestamp = scfg.Value(
        False,
        help='True: exporting cold result by timestamp, False: exporting cold result by year, Default is False')
    combine = scfg.Value(False, help='for temporal combined mode, Default is True')
    exclude_first = scfg.Value(True, help='exclude first date of image from each sensor, Default is True')
    sensors = scfg.Value('L8', type=str, help='sensor type, default is "L8"')


@profile
def export_cold_main(cmdline=1, **kwargs):
    """
    Args:
        cmdline (int, optional): _description_. Defaults to 1.

    Ignore:
        python -m geowatch.tasks.cold.export_cold_result_kwcoco --help
        TEST_COLD=1 xdoctest -m geowatch.tasks.cold.export_cold_result_kwcoco export_cold_main

    Example:
    >>> # xdoctest: +REQUIRES(env:TEST_COLD)
    >>> from geowatch.tasks.cold.export_cold_result_kwcoco import export_cold_main
    >>> from geowatch.tasks.cold.export_cold_result_kwcoco import *
    >>> kwargs= dict(
    >>>    rank = 0,
    >>>    n_cores = 1,
    >>>    stack_path = "/gpfs/scratchfs1/zhz18039/jws18003/new-repos/smart_data_dvc2/Drop6-MeanYear10GSD-V2/_pycold_combine2/stacked/KR_R001/",
    >>>    reccg_path = "/gpfs/scratchfs1/zhz18039/jws18003/new-repos/smart_data_dvc2/Drop6-MeanYear10GSD-V2/_pycold_combine2/reccg/KR_R001/",
    >>>    combined_coco_fpath = "/gpfs/scratchfs1/zhz18039/jws18003/new-repos/smart_data_dvc2/Drop6-MeanYear10GSD-V2/imgonly-KR_R001.kwcoco.zip",
    >>>    coefs = 'cv,rmse,a0,a1,b1,c1',
    >>>    year_lowbound = None,
    >>>    year_highbound = None,
    >>>    coefs_bands = '0,1,2,3,4,5',
    >>>    timestamp = False,
    >>>    combine = True,
    >>>    )
    >>> cmdline=0
    >>> export_cold_main(cmdline, **kwargs)
    """
    pman = kwargs.pop('pman', None)
    config_in = ExportColdKwcocoConfig.cli(cmdline=cmdline, data=kwargs, strict=True)
    rank = config_in['rank']
    n_cores = config_in['n_cores']
    stack_path = ub.Path(config_in['stack_path'])
    reccg_path = ub.Path(config_in['reccg_path'])
    year_lowbound = config_in['year_lowbound']
    year_highbound = config_in['year_highbound']
    coefs = config_in['coefs']
    coefs_bands = config_in['coefs_bands']
    combine = config_in['combine']
    timestamp = config_in['timestamp']
    exclude_first = config_in['exclude_first']
    sensors = config_in['sensors']

    if combine:
        combined_coco_fpath = ub.Path(config_in['combined_coco_fpath'])
    else:
        combined_coco_fpath = None

    # assert (combine and not timestamp) or (not combine and timestamp), "combine and timestamp must be either True and False, or False and True."
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

    # define variables
    config = read_json_metadata(stack_path)
    n_cols = config['padded_n_cols']
    n_rows = config['padded_n_rows']
    n_block_x = config['n_block_x']
    n_block_y = config['n_block_y']
    block_width = int(n_cols / n_block_x)  # width of a block
    block_height = int(n_rows / n_block_y)  # height of a block
    n_blocks = n_block_x * n_block_y  # total number of blocks

    cold_param = json.loads((reccg_path / 'log.json').read_text())
    method = cold_param['algorithm']

    # coef_names = ['cv', 'rmse', 'a0', 'a1', 'b1', 'a2', 'b2', 'a3', 'b3', 'c1']
    # band_names = [0, 1, 2, 3, 4, 5]

    if coefs is not None:
        try:
            coefs = list(coefs.split(","))
        except Exception:
            print(f'coefs={coefs}')
            print(
                "Illegal coefs inputs: example, --coefs='a0, c1, a1, b1, a2, b2, a3, b3, cv, rmse'")
            raise

        try:
            coefs_bands = list(coefs_bands.split(","))
            coefs_bands = [int(coefs_band) for coefs_band in coefs_bands]
        except Exception:
            print(f'coefs_bands={coefs_bands}')
            print(
                "Illegal coefs_bands inputs: example, --coefs_bands='0, 1, 2, 3, 4, 5, 6'")
            raise

    dt = np.dtype([('t_start', np.int32),
                   ('t_end', np.int32),
                   ('t_break', np.int32),
                   ('pos', np.int32),
                   ('num_obs', np.int32),
                   ('category', np.short),
                   ('change_prob', np.short),
                   # note that the slope coefficient was scaled up by 10000
                   ('coefs', np.float32, (7, 8)),
                   ('rmse', np.float32, 7),
                   ('magnitude', np.float32, 7)])

    # if coefs is not None:
    #     assert all(elem in coef_names for elem in coefs)
    #     assert all(elem in band_names for elem in coefs_bands)

    out_path = reccg_path / 'cold_feature'
    tmp_path = reccg_path / 'cold_feature' / 'tmp'

    tz = pytz.timezone('US/Eastern')

    if rank == 0:
        out_path.ensuredir()
        tmp_path.ensuredir()

    # MPI mode
    # trans = comm.bcast(trans, root=0)
    # proj = comm.bcast(proj, root=0)
    # cols = comm.bcast(cols, root=0)
    # rows = comm.bcast(rows, root=0)
    # config = comm.bcast(config, root=0)

    # Get ordinal list from sample block_folder
    block_folder = stack_path / 'block_x1_y1'
    meta_files = [m for m in os.listdir(block_folder) if m.endswith('.json')]

    # sort image files by ordinal dates
    img_dates = []
    img_names = []
    img_dates_L8 = []
    img_names_L8 = []
    img_dates_S2 = []
    img_names_S2 = []

    # read metadata and
    for meta in meta_files:
        meta_config = json.loads((block_folder / meta).read_text())
        ordinal_date = meta_config['ordinal_date']
        img_name = meta_config['image_name'] + '.npy'
        img_dates.append(ordinal_date)
        img_names.append(img_name)
    for meta in meta_files:
        meta_config = json.loads((block_folder / meta).read_text())
        if '_L8_' in meta_config['image_name']:
            ordinal_date_L8 = meta_config['ordinal_date']
            img_name_L8 = meta_config['image_name'] + '.npy'
            img_dates_L8.append(ordinal_date_L8)
            img_names_L8.append(img_name_L8)
        elif '_S2_' in meta_config['image_name']:
            ordinal_date_S2 = meta_config['ordinal_date']
            img_name_S2 = meta_config['image_name'] + '.npy'
            img_dates_S2.append(ordinal_date_S2)
            img_names_S2.append(img_name_S2)

    if year_lowbound is None:
        year_low_ordinal = min(img_dates)
        year_lowbound = pd.Timestamp.fromordinal(year_low_ordinal).year
    else:
        year_low_ordinal = pd.Timestamp.toordinal(
            datetime_mod.datetime(int(year_lowbound), 1, 1))

    if year_highbound is None:
        year_high_ordinal = max(img_dates)
        year_highbound = pd.Timestamp.fromordinal(year_high_ordinal).year
    else:
        year_high_ordinal = pd.Timestamp.toordinal(
            datetime_mod.datetime(int(year_highbound + 1), 1, 1))

    img_dates, img_names = zip(*filter(lambda x: x[0] >= year_low_ordinal,
                                       zip(img_dates, img_names)))
    img_dates, img_names = zip(*filter(lambda x: x[0] < year_high_ordinal,
                                       zip(img_dates, img_names)))
    img_dates = sorted(img_dates)
    img_names = sorted(img_names)
    if 'L8' in sensors:
        img_dates_L8, img_names_L8 = zip(*filter(lambda x: x[0] >= year_low_ordinal,
                                                 zip(img_dates_L8, img_names_L8)))
        img_dates_L8, img_names_L8 = zip(*filter(lambda x: x[0] < year_high_ordinal,
                                                 zip(img_dates_L8, img_names_L8)))
        img_dates_L8 = sorted(img_dates_L8)
        img_names_L8 = sorted(img_names_L8)
    if 'S2' in sensors:
        img_dates_S2, img_names_S2 = zip(*filter(lambda x: x[0] >= year_low_ordinal,
                                                 zip(img_dates_S2, img_names_S2)))
        img_dates_S2, img_names_S2 = zip(*filter(lambda x: x[0] < year_high_ordinal,
                                                 zip(img_dates_S2, img_names_S2)))
        img_dates_S2 = sorted(img_dates_S2)
        img_names_S2 = sorted(img_names_S2)

    if timestamp:
        ordinal_day_list = img_dates
    if not timestamp:
        first_ordinal_dates_L8 = []
        first_img_names_L8 = []
        first_ordinal_dates_S2 = []
        first_img_names_S2 = []
        last_year_L8 = None
        last_year_S2 = None
        if 'L8' in sensors:
            for ordinal_day_L8, img_name_L8 in zip(img_dates_L8, img_names_L8):
                year_L8 = pd.Timestamp.fromordinal(ordinal_day_L8).year
                if year_L8 != last_year_L8:
                    # print("L8", img_name_L8)
                    first_ordinal_dates_L8.append(ordinal_day_L8)
                    first_img_names_L8.append(img_name_L8)
                    last_year_L8 = year_L8
        if 'S2' in sensors:
            for ordinal_day_S2, img_name_S2 in zip(img_dates_S2, img_names_S2):
                year_S2 = pd.Timestamp.fromordinal(ordinal_day_S2).year
                if year_S2 != last_year_S2:
                    # print("S2", img_name_S2)
                    first_ordinal_dates_S2.append(ordinal_day_S2)
                    first_img_names_S2.append(img_name_S2)
                    last_year_S2 = year_S2
        if exclude_first:
            if 'L8' in sensors and 'S2' in sensors:
                ordinal_day_list = first_ordinal_dates_L8[1:] + first_ordinal_dates_S2[1:]
            elif 'L8' in sensors and 'S2' not in sensors:
                ordinal_day_list = first_ordinal_dates_L8[1:]
            elif 'S2' in sensors and 'L8' not in sensors:
                ordinal_day_list = first_ordinal_dates_S2[1:]
        else:
            if 'L8' in sensors and 'S2' in sensors:
                ordinal_day_list = first_ordinal_dates_L8 + first_ordinal_dates_S2
            elif 'L8' in sensors and 'S2' not in sensors:
                ordinal_day_list = first_ordinal_dates_L8
            elif 'S2' in sensors and 'L8' not in sensors:
                ordinal_day_list = first_ordinal_dates_S2
        ordinal_day_list.sort()
    if combine:
        combined_coco_dset = kwcoco.CocoDataset(combined_coco_fpath)

        # filter by sensors
        all_images = combined_coco_dset.images(list(ub.flatten(combined_coco_dset.videos().images)))
        flags = [s in sensors for s in all_images.lookup('sensor_coarse')]
        all_images = all_images.compress(flags)
        image_id_iter = iter(all_images)

        # Get ordinal date of combined coco image
        ordinal_dates = []
        ordinal_dates_july_first = []
        for image_id in image_id_iter:
            combined_coco_image: kwcoco.CocoImage = combined_coco_dset.coco_image(image_id)
            ts = combined_coco_image.img['timestamp']
            timestamp_local = datetime_cls.fromtimestamp(ts, tz=tz)
            timestamp_utc = timestamp_local.astimezone(pytz.utc)
            july_first = datetime_mod.date(timestamp_utc.year, 7, 1).toordinal()
            ordinal = timestamp_utc.toordinal()
            ordinal_dates.append(ordinal)
            ordinal_dates_july_first.append(july_first)

        ordinal_day_list = ordinal_dates

    ranks_percore = int(np.ceil(n_blocks / n_cores))
    i_iter = range(ranks_percore)
    if pman is not None:
        i_iter = pman.progiter(i_iter, total=ranks_percore,
                               desc=f'Export COLD \\[rank {rank}]', transient=True)

    for i in i_iter:
        iblock = n_cores * i + rank
        if iblock >= n_blocks:
            break
        current_block_y = int(np.floor(iblock / n_block_x)) + 1
        current_block_x = iblock % n_block_x + 1
        if method == 'OBCOLD':
            filename = f'record_change_x{current_block_x}_y{current_block_y}_obcold.npy'
        elif method == 'COLD':
            filename = f'record_change_x{current_block_x}_y{current_block_y}_cold.npy'
        elif method == 'HybridCOLD':
            filename = f'record_change_x{current_block_x}_y{current_block_y}_hybridcold.npy'

        block_folder = stack_path / \
            f'block_x{current_block_x}_y{current_block_y}'
        reccg_fpath = reccg_path / filename

        now = datetime_cls.now(tz).strftime('%Y-%m-%d %H:%M:%S')
        print(f'processing the rec_cg file {reccg_fpath} ({now})')

        # if not reccg_fpath.exists():
        #     print(f'the rec_cg file {reccg_fpath} is missing')

        if coefs is not None:
            results_block_coefs = np.full(
                (block_height, block_width, len(coefs) * len(coefs_bands),
                    len(ordinal_day_list)), -9999, dtype=np.float32)

            status_fpath = reccg_fpath.with_suffix('.status.json')
            if status_fpath.exists():
                with open(status_fpath, 'r') as f:
                    status_data = json.load(f)
                if status_data.get('status') == 'failed':
                    print(f'the rec_cg file {reccg_fpath} has failed status')
                    results_block_coefs
                else:
                    cold_block = np.array(np.load(reccg_fpath), dtype=dt)
                    cold_block_split = np.split(cold_block, np.argwhere(np.diff(cold_block['pos']) != 0)[:, 0] + 1)

                    nan_val = -9999
                    feature_outputs = coefs
                    feature_set = set(feature_outputs)

                    if not feature_set.issubset({'a0', 'c1', 'a1', 'b1', 'a2', 'b2', 'a3', 'b3', 'cv', 'rmse'}):
                        raise Exception('the outputted feature must be in [a0, c1, a1, b1, a2, b2, a3, b3, cv, rmse]')

                    for element in cold_block_split:
                        # Degugging mode
                        # if element[0]["pos"] == 4:
                        #     print(element)
                        # the relative column number in the block
                        i_col = int((element[0]["pos"] - 1) % n_cols) - \
                                (current_block_x - 1) * block_width
                        i_row = int((element[0]["pos"] - 1) / n_cols) - \
                                (current_block_y - 1) * block_height

                        for band_idx, band in enumerate(coefs_bands):
                            feature_row = extract_features(element, band, ordinal_day_list, nan_val, timestamp,
                                                            feature_outputs, feature_set)
                            # Degugging mode
                            # if current_block_x == 1 and current_block_y == 1:
                            #     if i_col == 3 and i_row == 0:
                            #         print(element[0]["pos"])
                            #         print((current_block_x, current_block_y), (i_col, i_row), (band_idx, band))
                            #         print('feature_row', feature_row)

                            for index, coef in enumerate(coefs):
                                # Degugging mode
                                # if i_col == 3 and i_row == 0 and element[0]["pos"] == 4:
                                #     print(feature_row[index])
                                results_block_coefs[i_row][i_col][index + band_idx * len(coefs)][:] = \
                                    feature_row[index]

        # save the temp dataset out
        for day in range(len(ordinal_day_list)):
            if coefs is not None:
                outfile = tmp_path / \
                    f'tmp_coefmap_block{iblock + 1}_{ordinal_day_list[day]}.npy'
                np.save(outfile, results_block_coefs[:, :, :, day])
    # MPI mode (wait for all processes)
    # comm.Barrier()

    gc.collect()


class NoMatchingColdCurve(Exception):
    ...


@profile
def extract_features(cold_plot, band, ordinal_day_list,
                     nan_val, timestamp, feature_outputs, feature_set):
    # NOTE: this function is a bottleneck, speedups are needed here

    features = np.full(
        (len(feature_outputs),
         len(ordinal_day_list)),
        nan_val,
        dtype=np.double)
    SLOPE_SCALE = 10000

    max_days_list = []
    for i in range(len(cold_plot)):
        last_year = pd.Timestamp.fromordinal(cold_plot[i]['t_end']).year
        max_days_list.append(datetime_mod.date(last_year, 12, 31).toordinal())

    # last_year = pd.Timestamp.fromordinal(cold_plot[i]['t_end']).year
    # max_days_list = [datetime_mod.date(last_year, 12, 31).toordinal()] * len(cold_plot)

    break_year_list = [-9999 if not (curve['t_break'] > 0 and curve['change_prob'] == 100) else
                       pd.Timestamp.fromordinal(curve['t_break']).year for curve in cold_plot]

    possible_features = ['a0', 'c1', 'a1', 'b1', 'rmse', 'cv']
    fk_to_idx = {
        fk: feature_outputs.index(fk)
        for fk in possible_features
        if fk in feature_set
    }

    a0_idx = fk_to_idx.get('a0', None)
    c1_idx = fk_to_idx.get('c1', None)
    a1_idx = fk_to_idx.get('a1', None)
    b1_idx = fk_to_idx.get('b1', None)
    rmse_idx = fk_to_idx.get('rmse', None)
    cv_idx = fk_to_idx.get('cv', None)

    # Precompute as much as possible before running the product
    idx_day_year_list = [
        (day_idx, ordinal_day, pd.Timestamp.fromordinal(ordinal_day).year)
        for day_idx, ordinal_day in enumerate(ordinal_day_list)
    ]

    mday_byear_curve_list = [
        (max_days_list[idx], break_year_list[idx], cold_curve)
        for idx, cold_curve in enumerate(cold_plot)]

    try:
        idxs_iter = itertools.product(idx_day_year_list, mday_byear_curve_list)
        for (day_idx, ordinal_day, ord_year), (max_day,
                                               break_year, cold_curve) in idxs_iter:
            if cv_idx is not None:
                if cold_curve['t_break'] != 0 and cold_curve['change_prob'] == 100:
                    if (timestamp and ordinal_day == cold_curve['t_break']) or (
                            not timestamp and break_year == ord_year):
                        features[cv_idx, day_idx] = cold_curve['magnitude'][band]
            if cold_curve['t_start'] <= ordinal_day < max_day:
                if a0_idx is not None:
                    features[a0_idx, day_idx] = (
                        cold_curve['coefs'][band][0] +
                        cold_curve['coefs'][band][1] *
                        ordinal_day / SLOPE_SCALE)
                if c1_idx is not None:
                    features[c1_idx, day_idx] = cold_curve['coefs'][band][1] / SLOPE_SCALE
                if a1_idx is not None:
                    features[a1_idx, day_idx] = cold_curve['coefs'][band][2]
                if b1_idx is not None:
                    features[b1_idx, day_idx] = cold_curve['coefs'][band][3]
                if rmse_idx is not None:
                    features[rmse_idx, day_idx] = cold_curve['rmse'][band]
        # print(features)
                    # In this case, we didn't find a matching cold
                    # curve for the current ordinal day, stop
                    # processing.
                    # raise NoMatchingColdCurve
    except NoMatchingColdCurve:
        ...

    return features


@profile
def read_json_metadata(stacked_path):
    for root, dirs, files in os.walk(stacked_path):
        for file in files:
            if file.endswith(".json"):
                json_path = os.path.join(root, file)

                with open(json_path, "r") as f:
                    metadata = json.load(f)
                    return metadata


if __name__ == '__main__':
    export_cold_main()
