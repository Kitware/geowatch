"""
Proof-of-concept script to reformat all of the images in a directory (in-place)
with a specific COG format.

TODO:
    - [ ] Prevent reencoding of lossy-compression formats unless the user forces it
"""

import os
import ubelt as ub
from os.path import join


def main():
    # CHANGE GDAL COMPRESSION
    jobs = ub.JobPool(mode='thread', max_workers=8)

    # dpath = os.path.abspath('.')
    dpaths = [
        '/media/joncrall/raid/home/joncrall/data/dvc-repos/smart_watch_dvc/models/fusion/SC-20201117/SC_smt_it_st_s12_newanns_weighted_hybrid_v20/',
        '/media/joncrall/raid/home/joncrall/data/dvc-repos/smart_watch_dvc/models/fusion/SC-20201117/SC_deit_newanns_weighted_hybrid_v21/',
        '/media/joncrall/raid/home/joncrall/data/dvc-repos/smart_watch_dvc/models/fusion/SC-20201117/SC_smt_it_stm_p8_newanns_cs64_t5_perframe_hybrid_v31/',
        '/media/joncrall/raid/home/joncrall/data/dvc-repos/smart_watch_dvc/models/fusion/SC-20201117/SC_deit_newanns_weighted_rgb_v23/',
        '/media/joncrall/raid/home/joncrall/data/dvc-repos/smart_watch_dvc/models/fusion/SC-20201117/SC_smt_it_stm_p8_newanns_cs64_t5_perframe_rgb_v30/',
        '/media/joncrall/raid/home/joncrall/data/dvc-repos/smart_watch_dvc/models/fusion/SC-20201117/SC_smt_it_st_s12_newanns_weighted_rgb_v22/',
        '/media/joncrall/raid/home/joncrall/data/dvc-repos/smart_watch_dvc/models/fusion/SC-20201117/SC_smt_it_stm_p8_newanns_weighted_hybrid_v27/',
        '/media/joncrall/raid/home/joncrall/data/dvc-repos/smart_watch_dvc/models/fusion/SC-20201117/SC_smt_it_stm_p8_newanns_cs96_t3_perframe_rgb_v28/',
        '/media/joncrall/raid/home/joncrall/data/dvc-repos/smart_watch_dvc/models/fusion/SC-20201117/SC_smt_it_stm_p8_newanns_cs96_t3_hybrid_v29/',
        '/media/joncrall/raid/home/joncrall/data/dvc-repos/smart_watch_dvc/models/fusion/SC-20201117/SC_smt_it_st_s12_newanns_weighted_pfnorm_rgb_v25/',
        '/media/joncrall/raid/home/joncrall/data/dvc-repos/smart_watch_dvc/models/fusion/SC-20201117/SC_smt_it_stm_p8_newanns_weighted_rgb_v26/',
        '/media/joncrall/raid/home/joncrall/data/dvc-repos/smart_watch_dvc/models/fusion/SC-20201117/SC_smt_it_st_s12_newanns_weighted_pfnorm_hybrid_v24/',
    ]
    submit_prog = ub.ProgIter(desc='submit jobs')
    with submit_prog:
        for dpath in dpaths:
            for r, ds, fs in os.walk(dpath):
                for fname in fs:
                    if fname.endswith('.tif'):
                        fpath = join(r, fname)
                        submit_prog.step()
                        jobs.submit(lazy_reencode_worker, fpath)

    for job in jobs.as_completed(desc='reencode'):
        job.result()


def lazy_reencode_worker(fpath):
    info = ub.cmd(['gdalinfo', fpath])
    compress = 'DEFLATE'
    need_reformat = (f'COMPRESSION={compress}' not in {line.strip() for line in info['out'].split('\n')})

    if need_reformat:
        reencode_gdal_image_inplace(fpath, compress=compress)


def reencode_gdal_image_inplace(fpath, compress='DEFLATE', blocksize=128,
                                overviews='AUTO'):
    gdal_options = {
        'compress': compress,
        'blocksize': blocksize,
        'overviews': overviews,
    }
    code = ub.hash_data(gdal_options)[0:8]
    prefix = '.tmp.' + code + '.'
    tmp_fpath = ub.augpath(fpath, prefix=prefix)
    src = fpath
    dst = tmp_fpath
    command = ub.paragraph(
        f'''
        gdal_translate
        -of COG
        -co OVERVIEWS=AUTO
        -co BLOCKSIZE={blocksize}
        -co COMPRESS={compress}
        -co NUM_THREADS=2
        {src} {dst}
        ''')
    ub.cmd(command, check=True)
    os.rename(dst, src)
    return dst


def test_settings():
    def check_reformat_image(fpath, compress='DEFLATE', blocksize=128,
                             overviews='AUTO'):
        gdal_options = {
            'compress': compress,
            'blocksize': blocksize,
            'overviews': overviews,
        }
        code = ub.hash_data(gdal_options)[0:8]
        prefix = '.tmp.' + code + '.'
        tmp_fpath = ub.augpath(fpath, prefix=prefix)
        src = fpath
        dst = tmp_fpath
        command = ub.paragraph(
            f'''
            gdal_translate
            -of COG
            -co OVERVIEWS=AUTO
            -co BLOCKSIZE={blocksize}
            -co COMPRESS={compress}
            -co NUM_THREADS=2
            {src} {dst}
            ''')
        ub.cmd(command, check=True)
        src_stat = os.stat(src)
        dst_stat = os.stat(dst)
        new_size_frac = dst_stat.st_size / src_stat.st_size
        return new_size_frac

    gdal_option_basis = {
        'compress': ['RAW', 'DEFLATE', 'LZW', 'LZMA'],
        'blocksize': [64, 128],
    }
    gdal_option_grid = list(ub.named_product(gdal_option_basis))

    fpath = '/media/joncrall/raid/home/joncrall/data/dvc-repos/smart_watch_dvc/models/fusion/SC-20201117/SC_smt_it_st_s12_newanns_weighted_rgb_v22/pred_SC_smt_it_st_s12_newanns_weighted_rgb_v22_epoch=83-step=3596291/combo_vali_nowv.kwcoco/_assets/None/crop_20181101T20181101_N37.734945E128.856510_N37.788316E128.938643_S2_0_18828489793eeee6.tif'
    rows = []
    for gdal_option in ub.ProgIter(gdal_option_grid):
        row = {}
        row.update(gdal_option)
        result = check_reformat_image(fpath, **gdal_option)
        row['size_frac'] = result
        rows.append(row)

    import pandas as pd
    table = pd.DataFrame(rows)
    table = table.sort_values('size_frac')
    print(table)
