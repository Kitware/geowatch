"""
cd /home/joncrall/remote/toothbrush/data/dvc-repos/smart_expt_dvc/_testpipe
rm -rf _cmd_queue_schedule
"""


def cleanup_mlops():
    import ubelt as ub
    root_dpath = ub.Path('/home/joncrall/remote/toothbrush/data/dvc-repos/smart_expt_dvc/_testpipe')

    # Reduce site model sizes
    json_fpaths = list((root_dpath / 'pred/flat').glob('*/*/*.json'))

    # site_dpaths = list((root_dpath / 'pred/flat').glob('*/*/sites'))
    # sitesumary_dpaths = list((root_dpath / 'pred/flat').glob('*/*/site_summaries'))

    from watch.utils import util_progress
    import xdev
    pman = util_progress.ProgressManager()
    with pman:
        size = 0
        globbers = ub.flatten([
            (root_dpath / 'pred/flat').glob('*/*/sites/*.geojson'),
            (root_dpath / 'pred/flat').glob('*/*/site_summaries/*.geojson'),
            (root_dpath / 'eval/flat').glob('*/*/sites/*.geojson'),
            (root_dpath / 'eval/flat').glob('*/*/site_summaries/*.geojson'),
        ])
        for p in pman(globbers):
            size += p.stat().st_size
            pman.update_info(xdev.byte_str(size))

    size_size = sum(p.stat().st_size  )

    # Reduce kwcoco sizes
    kwcoco_fpaths = list((root_dpath / 'pred/flat').glob('*/*/*.kwcoco.json'))
    total_kwcoco_size = sum([p.stat().st_size for p in kwcoco_fpaths])
    max([(p.stat().st_size, p) for p in kwcoco_fpaths])
    import xdev as xd
    xd.byte_str(total_kwcoco_size)

    zipped_kwcoco_fpaths = list((root_dpath / 'pred/flat').glob('*/*/*.kwcoco.json.zip'))
    total_zip_size = sum([p.stat().st_size for p in zipped_kwcoco_fpaths])
    xd.byte_str(total_zip_size)
    remaining_paths = [p for p in kwcoco_fpaths if p.exists()]
    total_remain_kwcoco_size = sum([p.stat().st_size for p in remaining_paths])

    removed_size = total_kwcoco_size - total_remain_kwcoco_size

    compress_jobs = ub.JobPool('thread', max_workers=20)
    delete_jobs = ub.JobPool('thread', max_workers=20)
    for p in remaining_paths:
        if p.exists():
            zip_fpath = p.augment(tail='.zip')
            if zip_fpath.exists():
                delete_jobs.submit(p.delete)
            else:
                job = compress_jobs.submit(compress_file, p)
                job.p = p

    for job in compress_jobs.as_completed(desc='compressing'):
        job.result()
        delete_jobs.submit(job.p.delete)

    for job in delete_jobs.as_completed(desc='deleting uncompressed files'):
        job.result()

    # Remove temporary files
    to_remove = []
    to_remove.append(root_dpath / '_cmd_queue_schedule')

    for poly_eval_dpath in [(root_dpath / 'eval/flat/sc_poly_eval'),
                            (root_dpath / 'eval/flat/bas_poly_eval')]:
        to_remove.extend(list(poly_eval_dpath.glob('*/tmp')))
    jobs = ub.JobPool('thread', max_workers=10)
    for p in ub.ProgIter(to_remove, desc='removing'):
        jobs.submit(p.delete)

    for job in jobs.as_completed(desc='deleting'):
        job.result()




def compress_coco_files(kwcoco_fpaths):
    import ubelt as ub
    for p in ub.ProgIter(kwcoco_fpaths, desc='compressing'):
        if p.exists():
            compress_file(p, remove_src=True)




def test_compression_ratios(src_fpath):
    """
    src_fpath = ub.Path('/home/joncrall/remote/toothbrush/data/dvc-repos/smart_expt_dvc/_testpipe/pred/flat/bas_poly/bas_poly_id_9bb69c12/site_summaries/US_R007.geojson')

    Requires:
        pip install zipfile-zstd
        pip install zstandard
    """
    import ubelt as ub
    src_fpath = ub.Path(src_fpath)
    grid = []
    grid += list(ub.named_product({
        'compression': ['ZIP_DEFLATED'],
        'compresslevel': list(range(9 + 1)),
    }))
    grid += list(ub.named_product({
        'compression': ['ZIP_BZIP2'],
        'compresslevel': list(range(1, 9 + 1)),
    }))
    grid += list(ub.named_product({
        'compression': ['ZIP_STORED'],
    }))
    grid += list(ub.named_product({
        'compression': ['ZIP_LZMA'],
    }))
    try:
        import zipfile_zstd  # NOQA
    except ImportError:
        ...
    else:
        grid += list(ub.named_product({
            'compression': ['ZIP_ZSTANDARD'],
            'compresslevel': list(range(1, 22 + 1)),
        }))

    old_size = src_fpath.stat().st_size
    rows = []
    ziptest_dpath = (src_fpath.parent / 'ziptest').ensuredir()
    for kw in ub.ProgIter(grid):
        suffix = ub.hash_data(kw)[0:8] + kw['compression']
        zip_fname = src_fpath.name + '.' + suffix + '.zip'
        zip_fpath = ziptest_dpath / zip_fname
        with ub.Timer('writing') as write_time:
            compress_file(src_fpath, zip_fpath=zip_fpath, **kw)
        with ub.Timer('reading') as read_time:
            with ub.zopen(zip_fpath + '/' + src_fpath.name) as file:
                data = file.read()
        row = {
            'write_time': write_time.elapsed,
            'read_time': read_time.elapsed,
            'new_size': zip_fpath.stat().st_size,
            'old_size': old_size,
            'suffix': suffix,
            **kw,
        }
        rows.append(row)
    import pandas as pd
    df = pd.DataFrame(rows)
    df['compress_ratio'] = df['old_size'] / df['new_size']
    df['write_efficiency'] = df['compress_ratio'] / df['write_time']
    df['read_efficiency'] = df['compress_ratio'] / df['read_time']
    import rich
    df = df.sort_values('compress_ratio')
    rich.print(df)

    big_data = src_fpath.read_bytes()
    small_data = zstd.compress(big_data)

    # df = df.sort_values('write_efficiency')
    # rich.print(df)
    # df = df.sort_values('read_efficiency')
    # rich.print(df)


def compress_file(src_fpath, zip_fpath=None, compression='auto', compresslevel='auto', remove_src=False):
    """
    To enable zstd, we require:

    pip install zipfile-zstd
    pip install zstandard
    """
    import zipfile
    import safer

    if isinstance(compression, str):
        if compression == 'auto':
            compression = 'ZIP_LZMA'
        compression = getattr(zipfile, compression)

    if compresslevel == 'auto':
        if compression in {zipfile.ZIP_DEFLATED}:
            compression = 9
        else:
            compresslevel = None

    if zip_fpath is None:
        zip_fpath = src_fpath.augment(tail='.zip')
    with safer.open(zip_fpath, 'wb') as file:
        with zipfile.ZipFile(file, 'w', compression=compression, compresslevel=compresslevel) as zfile:
            zfile.write(src_fpath, arcname=src_fpath.name)

    if remove_src:
        assert zip_fpath.exists()
        assert src_fpath.exists()
        src_fpath.delete()

    return zip_fpath


def rsync_datas():
    import ubelt as ub
    remote = 'yardrat'

    root_dpath = ub.Path.home() / 'data/dvc-repos/smart_expt_dvc/_testpipe'

    zipped_kwcoco_fpaths = list(ub.ProgIter((root_dpath / 'pred/flat').glob('*/*/*.kwcoco.json.zip')))
    text = '\n'.join([str(p.relative_to(root_dpath)) for p in zipped_kwcoco_fpaths])

    asset_dpaths = list(ub.ProgIter((root_dpath / 'pred/flat').glob('*/*/_assets')))
    text = '\n'.join([str(p.relative_to(root_dpath)) for p in asset_dpaths])
    print(text)
    dpath = ub.Path.appdir('watch', 'rsync').ensuredir()
    fpath = dpath / ('rsync_todo' + ub.timestamp() + '.txt')
    fpath.write_text(text)

    remote_relpath = root_dpath.relative_to(ub.Path.home())
    cmd = f'rsync -avprPR --files-from={fpath} {root_dpath} {remote}:{remote_relpath}'
    ub.cmd(cmd, system=True)


"""
Send to yardrat


rsync -avprPR /home/joncrall/remote/toothbrush/data/dvc-repos/smart_expt_dvc/./_testpipe yardrat:data/dvc-repos/smart_expt_dvc
rsync -avprPR --exclude _assets --exclude "_viz*" /home/joncrall/remote/toothbrush/data/dvc-repos/smart_expt_dvc/./_testpipe yardrat:data/dvc-repos/smart_expt_dvc

"""
