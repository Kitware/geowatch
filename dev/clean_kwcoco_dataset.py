"""
Based on ~/code/watch/dev/oneoffs/clean_drop3.py

Ignore:
    import watch
    import sys, ubelt
    sys.path.append(ubelt.expandpath('~/code/watch/dev'))
    from clean_kwcoco_dataset import *  # NOQA
    src = watch.find_smart_dvc_dpath(hardware='ssd') / 'Aligned-Drop4-2022-07-28-c20-TA1-S2-L8-ACC/testdata.kwcoco.json'
    cmdline = 0
    config = kwargs = CleanKwcocoDatasetConfig(src=src)


Ignore:
    export AWS_DEFAULT_PROFILE=iarpa
    export GDAL_DISABLE_READDIR_ON_OPEN=EMPTY_DIR

    gdalinfo /vsis3/smart-data-accenture/ta-1/ta1-s2-acc/50/R/QU/2020/12/24/S2B_50RQU_20201224_0_L1C_ACC/S2B_50RQU_20201224_0_L1C_ACC_B04.tif
    gdal_translate /vsis3/smart-data-accenture/ta-1/ta1-s2-acc/50/R/QU/2020/12/24/S2B_50RQU_20201224_0_L1C_ACC/S2B_50RQU_20201224_0_L1C_ACC_B04.tif demo.tif

    # This image has good nodata
    gdalinfo /vsis3/smart-data-accenture/ta-1/ta1-ls-acc/43/R/FM/2017/9/20/LC08_L1TP_147040_20170920_20200903_02_T1_ACC/LC08_L1TP_147040_20170920_20200903_02_T1_ACC_B04.tif
    gdal_translate /vsis3/smart-data-accenture/ta-1/ta1-ls-acc/43/R/FM/2017/9/20/LC08_L1TP_147040_20170920_20200903_02_T1_ACC/LC08_L1TP_147040_20170920_20200903_02_T1_ACC_B04.tif red.tif


    # This image is MISSING a nodata value:
    /vsis3/smart-data-accenture/ta-1/ta1-s2-acc/17/S/QD/2017/4/9/S2A_17SQD_20170409_0_L1C_ACC/S2A_17SQD_20170409_0_L1C_ACC_B02.tif

    # But our crop seems bad.
    kwplot /home/joncrall/data/dvc-repos/smart_watch_dvc-ssd/Aligned-Drop4-2022-07-28-c20-TA1-S2-L8-ACC/CN_C001/S2/affine_warp/crop_20201224T020000Z_N30.114986E119.908343_N30.593740E120.466058_S2_0/crop_20201224T020000Z_N30.114986E119.908343_N30.593740E120.466058_S2_0_red.tif

    gid = 16



"""
import os
import numpy as np
import scriptconfig as scfg
import kwcoco
import ubelt as ub


class CleanKwcocoDatasetConfig(scfg.DataConfig):
    """
    Attempt to find and fix issues in a kwcoco dataset. This includes:
        * Empty images [TODO]
        * Fix nodata regions [TODO]
        * Remove duplicate channels [TODO]
        * Remove unused fields [TODO]
    """
    src = scfg.Value(None, help='Input coco dataset')
    dst = scfg.Value(None, help='Output coco dataset')
    remove_extra_image_attrs = scfg.Value(True, help='remove image attrs that we dont really need')
    remove_extra_annot_attrs = scfg.Value(True, help='remove annot attrs that we dont really need')
    remove_empty_images = scfg.Value(True, help='remove images without data')
    fix_nodata_values = scfg.Value(True, help='try and fix nodata values')


def main(cmdline=True, **kwargs):

    config = CleanKwcocoDatasetConfig.legacy(cmdline=cmdline, data=kwargs)
    dset = kwcoco.CocoDataset(config.src)

    if config.remove_extra_image_attrs:
        # This can be a 40% reduction in file size
        for img in ub.ProgIter(dset.dataset['images'], desc='remove extra image attrs'):
            coco_img = kwcoco.CocoImage(img, dset)
            objs = list(coco_img.iter_asset_objs()) + [coco_img.img]
            # This might be a bit too agressive
            for obj in objs:
                obj.pop('utm_corners', None)
                obj.pop('geos_corners', None)
                obj.pop('wgs84_corners', None)
                obj.pop('utm_crs_info', None)
                obj.pop('wld_crs_info', None)
                obj.pop('is_rpc', None)
                obj.pop('warp_to_wld', None)
                obj.pop('valid_region_utm', None)
                obj.pop('wld_to_pxl', None)

    if config.remove_extra_annot_attrs:
        for ann in ub.ProgIter(dset.dataset['annotations'], desc='remove extra annot attrs'):
            ann.pop('segmentation_geos', None)

    image_stats_workers = 0
    pool = ub.JobPool('process', max_workers=image_stats_workers)
    all_gids = list(dset.index.imgs.keys())
    for gid in ub.ProgIter(all_gids, desc='checking image stats'):
        coco_img = dset.coco_image(gid).detach()
        job = pool.submit(get_imagedata_stats, coco_img)
        job.coco_img = coco_img

    import kwarray
    stats = kwarray.RunningStats()


    img_info_list = []
    prog = ub.ProgIter(pool.as_completed(), total=len(pool), desc='collect image stats')
    for job in prog:
        coco_img = job.coco_img
        img_info = job.result()
        img_info_list.append(img_info)

        for chan, chan_info in img_info['chan_infos'].items():
            chan_info['num_masked']
            chan_info['num_samecolor']
            chan_info['num_pixels']

        prog.set_postfix_str(f'num_bad = {len(bad_images)} / {len(all_gids)}')

    bad_images = []
    good_images = []
    prog = ub.ProgIter(pool.as_completed(), total=len(pool), desc='collect image stats')
    for img_info in img_info_list:
        if img_info['is_bad']:
            bad_images.append(img_info)
        else:
            good_images.append(img_info)

    for good in good_images:
        img_info = good
        num_bad = img_info['num_bad']
        num_exist = img_info['num_exist']
        is_bad = (num_bad == num_exist and num_exist > 0)
        img_info['is_bad'] = is_bad
        print('good = {!r}'.format(good))
        if is_bad:
            bad_images.append(img_info)

    bad = dset.images([b['gid'] for b in bad_images])
    sensor_to_num_bad = ub.dict_hist(bad.lookup("sensor_coarse"))
    region_to_num_bad = ub.dict_hist(dset.videos(bad.lookup("video_id")).lookup("name"))
    print('sensor_to_num_bad = {}'.format(ub.repr2(sensor_to_num_bad, nl=1)))
    print('region_to_num_bad = {}'.format(ub.repr2(region_to_num_bad, nl=1)))

    bad_stats = ub.ddict(lambda: 0)
    for bad in bad_images:
        gid = bad['gid']
        coco_img = dset.coco_image(gid)
        for chan, chan_info in bad['chan_infos'].items():
            sensor = coco_img.img["sensor_coarse"]
            if chan_info["max_val"] is np.ma.masked:
                bad_stats[f'{sensor}:{chan}.max_masked'] += 1
            elif chan_info["max_val"] == 0:
                bad_stats[f'{sensor}:{chan}.max_zero'] += 1
                chan_info["num_masked"]

    # bad_gids = [bad['gid'] for bad in bad_images]
    # dset.remove_images(bad_gids)


def get_imagedata_stats(coco_img, main_channels='red'):
    from osgeo import gdal
    from watch.utils import util_kwimage
    main_channels = kwcoco.FusedChannelSpec.coerce(main_channels)

    delayed = coco_img.delay(channels=main_channels, nodata_method='ma')
    chan_infos = {}
    for delayed_chan in delayed.parts:
        chan_infos[delayed_chan.channels.spec] = chan_info = {}
        # Find the raw delayed load leaf node
        found = None
        for _, node in delayed_chan._traverse():
            if hasattr(node, 'fpath'):
                found = node
        chan_node = found
        assert found is not None
        chan_fpath = ub.Path(chan_node.fpath)
        chan_info['exists'] = chan_fpath.exists()
        if chan_info['exists']:
            num_overviews = chan_node.prepare().num_overviews
            num_overviews = min(0, num_overviews)
            chan_overview = chan_node.get_overview(num_overviews).optimize()
            imdata = chan_overview.finalize()
            max_val = imdata.max()
            min_val = imdata.min()
            imdata_f = imdata.data.astype(np.float32)
            imdata_f[imdata.mask] = np.nan
            labels = util_kwimage.find_samecolor_regions(imdata_f)
            chan_info['gdal_info'] = gdal.Info(os.fspath(chan_fpath), format='json')
            chan_info['max_val'] = max_val
            chan_info['min_val'] = min_val
            chan_info['num_samecolor'] = labels.sum()
            chan_info['num_pixels'] = imdata.size
            chan_info['num_masked'] = imdata.mask.sum()

    img_info = {
        'chan_infos': chan_infos,
        'gid': coco_img.img['id'],
    }
    num_exist = 0
    num_bad = 0
    for chan, info in chan_infos.items():
        if info['exists']:
            num_exist += 1
            maxval = info['max_val']
            if maxval is np.ma.masked or maxval == 0:
                num_bad += 1

    is_bad = (num_bad == num_exist and num_exist > 0)
    img_info['is_bad'] = is_bad
    img_info['num_bad'] = num_bad
    img_info['num_exist'] = num_exist
    return img_info


def debug_cases(dset):
    images = dset.videos(names=['US_C000']).images[0]
    coco_images = images.coco_images
    coco_img1 = coco_images[0]
    coco_img2 = coco_images[2]

    coco_img1.img['auxiliary'][0]['parent_file_name']
    coco_img2.img['auxiliary'][0]['parent_file_name']

    delay1 = coco_img2.delay(channels='red|green|blue', nodata_method='float').warp({'scale': 0.25}).optimize()
    delay2 = coco_img1.delay(channels='red|green|blue', nodata_method='float').warp({'scale': 0.25}).optimize()

    for _, node in delay1._traverse():
        if hasattr(node, 'fpath'):
            print(f'node.fpath={node.fpath}')

    data1 = delay1.finalize()
    data2 = delay2.finalize()

    import kwimage
    canvas1 = kwimage.normalize_intensity(data1)
    canvas2 = kwimage.normalize_intensity(data2)
    canvas1 = kwimage.fill_nans_with_checkers(canvas1)
    canvas2 = kwimage.fill_nans_with_checkers(canvas2)

    import kwplot
    kwplot.autompl()
    kwplot.imshow(canvas1, pnum=(1, 2, 1))
    kwplot.imshow(canvas2, pnum=(1, 2, 2))

    # Parent image with no metadaa?
    # gdal_translate /vsis3/smart-data-accenture/ta-1/ta1-s2-acc/17/S/QD/2017/4/9/S2A_17SQD_20170409_0_L1C_ACC/S2A_17SQD_20170409_0_L1C_ACC_B02.tif part1.tif
    # gdal_translate /vsis3/smart-data-accenture/ta-1/ta1-s2-acc/18/S/TJ/2017/4/9/S2A_18STJ_20170409_0_L1C_ACC/S2A_18STJ_20170409_0_L1C_ACC_B02.tif part2.tif

    # util_gdal.gdal_multi_warp()
    # fpath1 = [
    #     '/vsis3/smart-data-accenture/ta-1/ta1-s2-acc/17/S/QD/2017/4/9/S2A_17SQD_20170409_0_L1C_ACC/S2A_17SQD_20170409_0_L1C_ACC_B02.tif',
    #     '/vsis3/smart-data-accenture/ta-1/ta1-s2-acc/18/S/TJ/2017/4/9/S2A_18STJ_20170409_0_L1C_ACC/S2A_18STJ_20170409_0_L1C_ACC_B02.tif',
    # ]

    # space_box = poly.bounding_box()
    # out_fpath = ub.Path.appdir('watch/test/gdal-warp/').ensuredir() / 'acc_red_nodata.tif'
    # gdal_single_warp(in_fpath, out_fpath, space_box=space_box, verbose=3)
