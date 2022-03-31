"""

Ignore:
    import watch
    import kwcoco
    dvc_dpath = watch.find_smart_dvc_dpath()
    fpath = dvc_dpath / 'Aligned-Drop3-TA1-2022-03-10/imgonly.kwcoco.json'
    dset = kwcoco.CocoDataset(fpath)
"""
import kwimage
import ubelt as ub
import kwcoco
import numpy as np


def is_image_empty(coco_img, main_channels='red'):
    bundle_dpath = ub.Path(coco_img.bundle_dpath)

    main_channels = kwcoco.FusedChannelSpec.coerce(main_channels)
    # main_channels = coco_img.channels

    chan_infos = {}
    for obj in coco_img.iter_asset_objs():
        chan = kwcoco.FusedChannelSpec.coerce(obj['channels'])
        if (main_channels & chan).numel():
            gpath = bundle_dpath / obj['file_name']
            chan_infos[chan.spec] = chan_info = {}
            chan_info['exists'] = gpath.exists()
            if chan_info['exists']:
                try:
                    imdata = kwimage.imread(gpath, backend='gdal', nodata='ma', overview=-1)
                except Exception:
                    imdata = kwimage.imread(gpath, backend='gdal', nodata='ma')
                max_val = imdata.max()
                min_val = imdata.min()
                chan_info['max_val'] = max_val
                chan_info['min_val'] = min_val
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


def remove_empty_images(dset):
    gid_to_infos = {}

    pool = ub.JobPool('process', max_workers=8)
    all_gids = list(dset.index.imgs.keys())
    for gid in ub.ProgIter(all_gids, desc='find empty images'):
        if gid not in gid_to_infos:
            coco_img = dset.coco_image(gid).detach()
            job = pool.submit(is_image_empty, coco_img)
            job.coco_img = coco_img

    bad_images = []
    good_images = []
    prog = ub.ProgIter(pool.as_completed(), total=len(pool), desc='collect jobs')
    for job in prog:
        coco_img = job.coco_img
        img_info = job.result()
        if img_info['is_bad']:
            bad_images.append(img_info)
            prog.set_postfix_str(f'num_bad = {len(bad_images)} / {len(all_gids)}')
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

    bad_gids = [bad['gid'] for bad in bad_images]
    dset.remove_images(bad_gids)


def clean_drop3():
    """
    Post processing hack
    """
    import os
    import watch
    import kwcoco
    dvc_dpath = watch.find_smart_dvc_dpath()
    coco_fpath = dvc_dpath / 'Aligned-Drop3-TA1-2022-03-10/data.kwcoco.json'
    dset = kwcoco.CocoDataset(coco_fpath)

    no_activity_cid = dset.index.name_to_cat['No Activity']['id']
    annots = dset.annots()
    bad_annots = [c == no_activity_cid for c in annots.lookup('category_id')]
    bad_annots = annots.compress(bad_annots)
    dset.remove_annotations(list(bad_annots))

    for img in dset.dataset['images']:
        coco_img = kwcoco.CocoImage(img, dset)
        objs = list(coco_img.iter_asset_objs()) + [coco_img.img]
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

    for ann in dset.dataset['annotations']:
        ann.pop('segmentation_geos', None)

    remove_empty_images(dset)

    import kwimage
    images = dset.images()
    valid_fracs = []
    for img in images.objs:
        img_poly = kwimage.Boxes([[0, 0, img['width'], img['height']]], 'ltrb').to_shapely()[0]
        valid_poly = kwimage.MultiPolygon.coerce(img['valid_region']).to_shapely()
        valid_frac = (valid_poly.area / img_poly.area)
        valid_fracs.append(valid_frac)

    valid_fracs = np.array(valid_fracs)
    import pandas as pd
    df = pd.DataFrame({
        'gid': images.lookup('id'),
        'sensor': images.lookup('sensor_coarse', ''),
        'video_name': dset.videos(images.lookup('video_id')).lookup('name'),
        'valid_fracs': valid_fracs,
    })

    flags = (df.valid_fracs < 0.5) & (df.sensor == 'S2')
    flags |= (df.valid_fracs < 0.5) & (df.sensor == 'L8')
    flags |= (df.valid_fracs < 0.01) & (df.sensor == 'WV')

    invalid_df = df[flags]
    valid_df = df[~flags]

    print(ub.dict_hist(df.sensor))
    print(ub.dict_hist(valid_df.sensor))
    print(ub.dict_hist(invalid_df.sensor))

    print(ub.dict_hist(df.video_name))
    print(ub.dict_hist(valid_df.video_name))
    print(ub.dict_hist(invalid_df.video_name))
    print(invalid_df.describe())

    dset.remove_images(invalid_df.gid.values.tolist())

    orig_fpath = dset.fpath
    dset.fpath = os.fspath(ub.Path(orig_fpath).augment(suffix='-clean3'))
    dset.dump(dset.fpath, newlines=True)

    # ----
    import os
    import watch
    import kwcoco
    dvc_dpath = watch.find_smart_dvc_dpath()
    coco_fpath = dvc_dpath / 'Aligned-Drop3-TA1-2022-03-10/data.kwcoco-clean2.json'
    dset = kwcoco.CocoDataset(coco_fpath)
