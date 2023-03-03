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
logger = logging.getLogger(__name__)


try:
    from xdev import profile
except ImportError:
    profile = ub.identity


class ExportColdKwcocoConfig(scfg.DataConfig):
    """
    The docstring will be the description in the CLI help
    """
    rank = scfg.Value(None, help='rank id')
    n_cores = scfg.Value(None, help='total cores assigned')
    stack_path = scfg.Value(None, help='folder directory of stacked data')
    reccg_path = scfg.Value(None, help='folder directory of cold processing result')
    meta_fpath = scfg.Value(None, help='file path of metadata json created by prepare_kwcoco script')
    year_lowbound = scfg.Value(None, help='min year for saving geotiff, e.g., 2017')
    year_highbound = scfg.Value(None, help='max year for saving geotiff, e.g., 2022')
    coefs = scfg.Value(None, type=str, help="list of COLD coefficients for saving geotiff, e.g., a0,c1,a1,b1,a2,b2,a3,b3,cv,rmse")
    coefs_bands = scfg.Value(None, type=str, help='indicate the ba_nds for output coefs_bands, e.g., 0,1,2,3,4,5')
    timestamp = scfg.Value(True, help='True: exporting cold result by timestamp, False: exporting cold result by year, Default is False')


@profile
def export_cold_main(cmdline=1, **kwargs):
    """_summary_

    Args:
        cmdline (int, optional): _description_. Defaults to 1.

    Ignore:
        python -m watch.tasks.cold.export_cold_result_kwcoco --help
        TEST_COLD=1 xdoctest -m watch.tasks.cold.export_cold_result_kwcoco export_cold_main

    Example:
    >>> # xdoctest: +REQUIRES(env:TEST_COLD)
    >>> from watch.tasks.cold.export_cold_result_kwcoco import export_cold_main
    >>> from watch.tasks.cold.export_cold_result_kwcoco import *
    >>> kwargs= dict(
    >>>    rank = 0,
    >>>    n_cores = 1,
    >>>    stack_path = "/gpfs/scratchfs1/zhz18039/jws18003/kwcoco/stacked/KR_R001",
    >>>    reccg_path = "/gpfs/scratchfs1/zhz18039/jws18003/kwcoco/reccg/KR_R001",
    >>>    meta_fpath = '/gpfs/scratchfs1/zhz18039/jws18003/kwcoco/stacked/KR_R001/block_x10_y1/crop_20140115T020000Z_N37.643680E128.649453_N37.683356E128.734073_L8_0.json',
    >>>    coefs = ['cv'],
    >>>    year_lowbound = 2017,
    >>>    year_highbound = 2022,
    >>>    coefs_bands = [0, 1, 2, 3, 4, 5],
    >>>    timestamp = True,
    >>>    )
    >>> cmdline=0
    >>> export_cold_main(cmdline, **kwargs)
    """
    pman = kwargs.pop('pman', None)
    config_in = ExportColdKwcocoConfig.cli(cmdline=cmdline, data=kwargs)
    rank = config_in['rank']
    n_cores = config_in['n_cores']
    stack_path = ub.Path(config_in['stack_path'])
    reccg_path = ub.Path(config_in['reccg_path'])
    meta_fpath = ub.Path(config_in['meta_fpath'])
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

    # define variables
    config = json.loads(meta_fpath.read_text())
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
            print("Illegal coefs inputs: example, --coefs='a0, c1, a1, b1, a2, b2, a3, b3, cv, rmse'")
            raise

        try:
            coefs_bands = list(coefs_bands.split(","))
            coefs_bands = [int(coefs_band) for coefs_band in coefs_bands]
        except Exception:
            print(f'coefs_bands={coefs_bands}')
            print("Illegal coefs_bands inputs: example, --coefs_bands='0, 1, 2, 3, 4, 5, 6'")
            raise

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

    # if coefs is not None:
    #     assert all(elem in coef_names for elem in coefs)
    #     assert all(elem in band_names for elem in coefs_bands)

    out_path = reccg_path / 'cold_feature'

    if rank == 0:
        out_path.ensuredir()

    # MPI mode
    # trans = comm.bcast(trans, root=0)
    # proj = comm.bcast(proj, root=0)
    # cols = comm.bcast(cols, root=0)
    # rows = comm.bcast(rows, root=0)
    # config = comm.bcast(config, root=0)

    # Get ordinal list from sample block_folder
    block_folder = stack_path / 'block_x1_y1'
    if timestamp:
        meta_files = [m for m in os.listdir(block_folder) if m.endswith('.json')]

        # sort image files by ordinal dates
        img_dates = []
        img_names = []

        # read metadata and
        for meta in meta_files:
            meta_config = json.loads((block_folder / meta).read_text())
            ordinal_date = meta_config['ordinal_date']
            img_name = meta_config['image_name'] + '.npy'
            img_dates.append(ordinal_date)
            img_names.append(img_name)

        if year_lowbound is None:
            year_low_ordinal = min(img_dates)
            year_lowbound = pd.Timestamp.fromordinal(year_low_ordinal).year
        else:
            year_low_ordinal = pd.Timestamp.toordinal(datetime_mod.datetime(int(year_lowbound), 1, 1))

        img_dates, img_names = zip(*filter(lambda x: x[0] >= year_low_ordinal,
                                            zip(img_dates, img_names)))
        if year_highbound is None:
            year_high_ordinal = max(img_dates)
            year_highbound = pd.Timestamp.fromordinal(year_high_ordinal).year
        else:
            year_high_ordinal = pd.Timestamp.toordinal(datetime_mod.datetime(int(year_highbound + 1), 1, 1))

        img_dates, img_names = zip(*filter(lambda x: x[0] < year_high_ordinal,
                                                zip(img_dates, img_names)))
        img_dates = sorted(img_dates)
        img_names = sorted(img_names)
        ordinal_day_list = img_dates

    ranks_percore = int(np.ceil(n_blocks / n_cores))
    i_iter = range(ranks_percore)
    if pman is not None:
        i_iter = pman.progiter(i_iter, total=ranks_percore,
                               desc=f'Export COLD: Rank {rank}', transient=True)

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

        block_folder = stack_path / f'block_x{current_block_x}_y{current_block_y}'
        reccg_fpath = reccg_path / filename

        if timestamp:
            if coefs is not None:
                results_block_coefs = np.full(
                    (block_height, block_width, len(coefs) * len(coefs_bands),
                     len(ordinal_day_list)), -9999, dtype=np.float32)

            print(f'processing the rec_cg file {reccg_fpath}')
            if not reccg_fpath.exists():
                print(f'the rec_cg file {reccg_fpath} is missing')

        cold_block = np.array(np.load(reccg_fpath), dtype=dt)

        if coefs is not None:
            cold_block_split = np.split(cold_block, np.argwhere(np.diff(cold_block['pos']) != 0)[:, 0] + 1)

            rcs = []
            for element in cold_block_split:
                # the relative column number in the block
                i_col = int((element[0]["pos"] - 1) % n_cols) - \
                        (current_block_x - 1) * block_width
                i_row = int((element[0]["pos"] - 1) / n_cols) - \
                        (current_block_y - 1) * block_height
                rcs.append((i_row, i_col))

            rcb_iter = itertools.product(rcs, enumerate(coefs_bands))
            if pman is not None:
                rcb_total = len(rcs) * len(coefs_bands)
                rcb_iter = pman.progiter(rcb_iter, desc=f'Extract COLD Rank {rank}', total=rcb_total)

            nan_val = -9999
            feature_outputs = coefs
            feature_set = set(feature_outputs)
            if not feature_set.issubset({'a0', 'c1', 'a1', 'b1', 'a2', 'b2', 'a3', 'b3', 'cv', 'rmse'}):
                raise Exception('the outputted feature must be in [a0, c1, a1, b1, a2, b2, a3, b3, cv, rmse]')

            for (i_row, i_col), (band_idx, band) in rcb_iter:
                feature_row = extract_features(
                    element, band, ordinal_day_list, nan_val, timestamp,
                    feature_outputs, feature_set)
                for index, coef in enumerate(coefs):
                    bx = index + band_idx * len(coefs)
                    results_block_coefs[i_row][i_col][bx][:] = feature_row[index]

        # save the temp dataset out
        if timestamp:
            for day in range(len(ordinal_day_list)):
                if coefs is not None:
                    outfile = out_path / f'tmp_coefmap_block{iblock + 1}_{ordinal_day_list[day]}.npy'
                    np.save(outfile, results_block_coefs[:, :, :, day])
    # MPI mode (wait for all processes)
    # comm.Barrier()


class NoMatchingColdCurve(Exception):
    ...


def extract_features(cold_plot, band, ordinal_day_list, nan_val, timestamp, feature_outputs, feature_set):

    features = np.full((len(feature_outputs), len(ordinal_day_list)), nan_val, dtype=np.double)
    SLOPE_SCALE = 10000

    last_year = pd.Timestamp.fromordinal(cold_plot[-1]['t_end']).year
    max_days_list = [datetime_mod.date(last_year, 12, 31).toordinal()] * len(cold_plot)
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

    idx_day_year_list = [
        (day_idx, ordinal_day, pd.Timestamp.fromordinal(ordinal_day).year)
        for day_idx, ordinal_day in enumerate(ordinal_day_list)
    ]

    # # --- B  # NOQA
    # idxs_iter = itertools.product(idx_day_year_list, enumerate(cold_plot))
    # try:
    #     for (day_idx, ordinal_day, ord_year), (idx, cold_curve) in idxs_iter:
    # # --- B  # NOQA

    # --- A  # NOQA
    for day_idx, ordinal_day in idx_day_year_list:
        ord_year = pd.Timestamp.fromordinal(ordinal_day).year
        for idx, cold_curve in enumerate(cold_plot):
    # --- A  # NOQA

            if cold_curve['t_start'] <= ordinal_day < max_days_list[idx]:
                if a0_idx is not None:
                    features[a0_idx][day_idx] = (
                        cold_curve['coefs'][band][0] +
                        cold_curve['coefs'][band][1] *
                        ordinal_day / SLOPE_SCALE)
                if c1_idx is not None:
                    features[c1_idx][day_idx] = cold_curve['coefs'][band][1] / SLOPE_SCALE
                if a1_idx is not None:
                    features[a1_idx][day_idx] = cold_curve['coefs'][band][2]
                if b1_idx is not None:
                    features[b1_idx][day_idx] = cold_curve['coefs'][band][3]
                if rmse_idx is not None:
                    features[rmse_idx][day_idx] = cold_curve['rmse'][band]

                if cv_idx is not None:
                    if cold_curve['t_break'] != 0 and cold_curve['change_prob'] == 100:
                        break_year = break_year_list[idx]
                        if (timestamp and ordinal_day == cold_curve['t_break']) or (not timestamp and break_year == ord_year):
                            features[cv_idx][day_idx] = cold_curve['magnitude'][band]
    # ---A  # NOQA
                            break
        else:
            # This else block runs only if the loop completed without a break statement being executed
            # In this case, we didn't find a matching cold curve for the current ordinal day
            continue
        break
    # ---A  # NOQA

    # # --- B  # NOQA
    #                         # Pretty sure this exception logic is equivalent to the
    #                         # loop with the else
    #                         raise NoMatchingColdCurve
    # except NoMatchingColdCurve:
    #     ...
    # # --- B  # NOQA

    return features


if __name__ == '__main__':
    export_cold_main()
