"""
Doing this hack on horologic

QA Bands had unsigned_nodata=255, but it should have beeen unsigned_nodata=0

cp /data/joncrall/dvc-repos/smart_phase3_data/Aligned-Drop8-ARA/KR_R001/L8/affine_warp/crop_20140225T010000Z_N37.643680E128.649453_N37.683356E128.734073_L8_0/crop_20140225T010000Z_N37.643680E128.649453_N37.683356E128.734073_L8_0_quality.tif qademo.tif

gdalinfo qademo.tif
gdalinfo -stats qademo.tif

# Modify
gdal_edit.py -a_nodata 0 -oo IGNORE_COG_LAYOUT_BREAK=YES qademo.tif


python ~/code/watch/geowatch/tasks/fusion/datamodules/qa_bands.py qademo.tif --show

import kwimage
data = kwimage.imread('qademo.tif')
np.unique(data)
np.isnan(data).sum()

dvc unprotect -- Aligned-Drop8-ARA/*/L8 Aligned-Drop8-ARA/*/S2 Aligned-Drop8-ARA/*/PD Aligned-Drop8-ARA/*/WV
dvc unprotect -- Aligned-Drop8-ARA/*/*.kwcoco.zip

"""
import kwcoco
import ubelt as ub


def simple_coco_stats():
    bundle_dpath = ub.Path('/data2/projects/smart/smart_phase3_data/Aligned-Drop8-ARA')
    coco_fpaths = list(bundle_dpath.glob('*/imganns-*.kwcoco.zip'))
    coco_datasets = list(kwcoco.CocoDataset.coerce_multiple(coco_fpaths, workers=8))

    to_remove = []

    main_table = []
    detail_table = []
    for dset in coco_datasets:
        region_id = ub.Path(dset.fpath).stem.split('.')[0].split('-')[1]
        dset.n_images
        if dset.n_images == 0:
            to_remove.append(ub.Path(dset.fpath).parent)
            continue
        main_table.append({
            'region_id': region_id,
            'region_type': region_id.split('_')[1][0],
            'num_images': dset.n_images,
        })

        from kwutil.util_time import datetime

        groups = ub.group_items(dset.images().objs, key=lambda r: (r['sensor_coarse'], datetime.coerce(r['date_captured']).year))
        for k, group in groups.items():
            sensor, year = k
            detail_table.append({
                'region_id': region_id,
                'region_type': region_id.split('_')[1][0],
                'sensor': sensor,
                'year': year,
                'num_images': len(group),
            })

    table = sorted(main_table, key=lambda x: x['region_type'])
    import pandas as pd
    import rich
    df = pd.DataFrame(table)
    rich.print(df.to_string())

    detail_table = sorted(detail_table, key=lambda x: x['region_type'])
    df = pd.DataFrame(detail_table)
    detail_table = df.pivot(index=['region_type', 'region_id', 'sensor'], columns=['year'], values=['num_images'])
    rich.print(detail_table.to_string())


def hack_fixups(coco_datasets):
    registered_qa_fpaths = []
    for dset in ub.ProgIter(coco_datasets, desc='iter'):
        for coco_img in dset.images().coco_images:
            qa_asset = coco_img.find_asset('quality')
            qa_fpath = qa_asset.image_filepath()
            registered_qa_fpaths.append(qa_fpath)

    bundle_dpath = ub.Path('/data2/projects/smart/smart_phase3_data/Aligned-Drop8-ARA')
    all_qa_fpaths = list(bundle_dpath.glob('*/*/*/*/*_quality.tif'))
    unregistered_qa_bands = set(all_qa_fpaths) - set(registered_qa_fpaths)
    unregistered_qa_bands = [p for p in unregistered_qa_bands if not p.name.startswith('.')]

    # Fixup all QA bands
    for qa_fpath in ub.ProgIter(unregistered_qa_bands, desc='fix unregistered qa bands'):
        ub.cmd(f'gdal_edit.py -a_nodata 0 -oo IGNORE_COG_LAYOUT_BREAK=YES {qa_fpath}', verbose=3)


def main():
    bundle_dpath = ub.Path('/data2/projects/smart/smart_phase3_data/Aligned-Drop8-ARA')
    coco_fpaths = list(bundle_dpath.glob('*/imganns-*.kwcoco.zip'))

    import kwutil
    pman = kwutil.util_progress.ProgressManager()
    jobs = ub.JobPool(mode='thread', max_workers=16)
    with pman, jobs:
        for coco_fpath in pman.progiter(coco_fpaths, desc='submit jobs', transient=True):
            jobs.submit(fixup_qa_bands, coco_fpath, pman=pman, dry=0)

        for job in pman.progiter(jobs.as_completed(), total=len(jobs), desc='collect'):
            job.result()


def fixup_qa_bands(coco_fpath, pman=None, dry=1):

    if 'imganns' in coco_fpath.name:
        alt_fpath = coco_fpath.parent / coco_fpath.name.replace('imganns', 'imgonly')
    else:
        assert False
    assert alt_fpath.exists()

    dset = kwcoco.CocoDataset(coco_fpath)
    stem = coco_fpath.stem

    coco_img_iter = list(dset.images().coco_images)
    if len(coco_img_iter) == 0:
        return
    if pman:
        coco_img_iter = pman.progiter(coco_img_iter, desc=f'Fix main {stem}', transient=True)

    # Force the new nodata value
    for coco_img in coco_img_iter:
        qa_asset = coco_img.find_asset('quality')
        qa_asset['qa_encoding'] = 'ARA-4'
        qa_asset['band_metas'][0]['nodata'] = 0
        qa_fpath = qa_asset.image_filepath()
        assert not qa_fpath.is_symlink()
        # Doesnt seem to actually break things
        if not dry:
            ub.cmd(f'gdal_edit.py -a_nodata 0 -oo IGNORE_COG_LAYOUT_BREAK=YES {qa_fpath}', verbose=3)
    if not dry:
        dset.dump()

    # Fix both datasets
    dset2 = kwcoco.CocoDataset(alt_fpath)

    coco_img_iter2 = list(dset2.images().coco_images)
    if pman:
        coco_img_iter2 = pman.progiter(coco_img_iter2, desc=f'Fix alt {stem}', transient=True)

    for coco_img in coco_img_iter2:
        qa_asset = coco_img.find_asset('quality')
        qa_asset['qa_encoding'] = 'ARA-4'
        qa_asset['band_metas'][0]['nodata'] = 0
        qa_fpath = qa_asset.image_filepath()
        assert not qa_fpath.is_symlink()

    if not dry:
        # Update the dataset
        dset2.dump()
