#!/bin/env python
import kwimage
import ubelt as ub
import kwcoco
import numpy as np
import scriptconfig as scfg


class RemoveEmptyImagesConfig(scfg.Config):
    """
    Updates image transforms in a kwcoco json file to align all videos to a
    target GSD.
    """
    default = {
        'src': scfg.Value('data.kwcoco.json', help='input kwcoco filepath'),

        'dst': scfg.Value(None, help='output kwcoco filepath'),

        'workers': scfg.Value(0, type=str, help='number of io threads'),

        'mode': scfg.Value('process', help='can be thread, process, or serial'),

        'channels': scfg.Value(None, help='If specified, check only these channels for bad pixels'),

        'delete_assets': scfg.Value(False, help='if True actually deletes the assets')
    }


def main(cmdline=True, **kwargs):
    """
    Ignore:
        kwargs = {}
        kwargs['src'] = 'imgonly.kwcoco.json'
        kwargs['dst'] = 'imgonly2.kwcoco.json'
        kwargs['workers'] = 8
        # kwargs['channels'] = 'red|green|blue'
        kwargs['channels'] = 'red'
        cmdline = False
    """
    config = RemoveEmptyImagesConfig(cmdline=cmdline, data=kwargs)
    mode = config['mode']
    workers = config['workers']
    main_channels = config['channels']
    if main_channels is not None:
        main_channels = kwcoco.FusedChannelSpec.coerce(main_channels)

    dset = kwcoco.CocoDataset(config['src'])

    bad_gids = find_empty_images(dset, main_channels, mode=mode,
                                 workers=workers)

    if config['delete_assets']:

        bad_fpaths = []
        for bad_gid in ub.ProgIter(bad_gids, desc='collect empty assets'):
            coco_img = dset.coco_image(bad_gid)
            bad_fpaths.extend(list(coco_img.iter_image_filepaths()))

        for bad_fpath in ub.ProgIter(bad_fpaths, desc='delete empty assets'):
            ub.delete(bad_fpath)

    dset.remove_images(bad_gids)
    # dset.fpath = config['dst']
    import safer
    with safer.open(config['dst'], 'w', temp_file=True) as file:
        dset.dump(file, indent='    ', newlines=True)


def is_image_empty(coco_img, main_channels=None):
    """
    Run heristics to determine if a coco image is empty.
    """
    bundle_dpath = ub.Path(coco_img.bundle_dpath)

    if main_channels is not None:
        main_channels = kwcoco.FusedChannelSpec.coerce(main_channels)

    chan_infos = {}
    for obj in coco_img.iter_asset_objs():
        chan = kwcoco.FusedChannelSpec.coerce(obj['channels'])
        if main_channels is None or (main_channels & chan).numel():
            gpath = bundle_dpath / obj['file_name']
            chan_infos[chan.spec] = chan_info = {}
            chan_info['exists'] = gpath.exists()
            if chan_info['exists']:
                try:
                    imdata = kwimage.imread(gpath, backend='gdal', nodata='ma', overview=-1)
                except Exception:
                    imdata = kwimage.imread(gpath, backend='gdal', nodata='ma')

                valid_values = imdata.data[~imdata.mask]
                num_masked = imdata.mask.sum()
                num_zero = (valid_values == 0).sum()

                num_iffy = num_masked + num_zero
                total = imdata.mask.size

                if len(valid_values) == 0:
                    max_val = np.ma.masked
                    min_val = np.ma.masked
                    num_min = 0
                    num_max = 0
                else:
                    max_val = valid_values.max()
                    min_val = valid_values.min()
                    num_min = (valid_values == max_val).sum()
                    num_max = (valid_values == min_val).sum()
                    if max_val != 0 and min_val != 0:
                        num_iffy += num_min

                chan_info['max_val'] = max_val
                chan_info['min_val'] = min_val

                chan_info['num_masked'] = num_masked
                chan_info['num_zero'] = num_zero
                chan_info['num_min'] = num_min
                chan_info['num_max'] = num_max
                chan_info['num_iffy'] = num_iffy

                chan_info['frac_masked'] = num_masked / total
                chan_info['frac_zero'] = num_zero / total
                chan_info['frac_iffy'] = num_iffy / total

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


def find_empty_images(dset, main_channels, mode='process', workers=0):

    gid_to_infos = {}
    pool = ub.JobPool('process', max_workers=8)
    all_gids = list(dset.index.imgs.keys())
    for gid in ub.ProgIter(all_gids, desc='find empty images'):
        if gid not in gid_to_infos:
            coco_img = dset.coco_image(gid).detach()
            job = pool.submit(is_image_empty, coco_img,
                              main_channels=main_channels)
            job.coco_img = coco_img

    image_infos = []
    num_bad = 0
    prog = ub.ProgIter(pool.as_completed(), total=len(pool), desc='find empty images')
    for job in prog:
        coco_img = job.coco_img
        img_info = job.result()
        if img_info['is_bad']:
            num_bad += 1
            prog.set_postfix_str(f'num_empty = {num_bad} / {len(all_gids)}')
        image_infos.append(img_info)

    for img_info in image_infos:
        img_iffys = [b['frac_iffy'] for b in img_info['chan_infos'].values()]
        img_info['frac_iffy'] = min(img_iffys)
        # img_info['frac_iffy'] = sum(img_iffys) / len(img_iffys)

    if 1:
        iffy_fracs = [d['frac_iffy'] for d in image_infos]
        iffy_fracs = np.array(iffy_fracs)
        iffy_bins = [0, 0.25, 0.5, 0.75, 0.85, .90, .95, 0.98, 1.0]
        iffy_freq, iffy_bins = np.histogram(iffy_fracs, bins=iffy_bins)
        iffy_hist = ub.dzip(ub.iter_window(iffy_bins, 2), iffy_freq)
        print('iffy_hist = {}'.format(ub.repr2(iffy_hist, nl=1)))

    # TODO: different iffy thresh per sensor
    iffy_thresh = 0.95
    bad_images = []
    for img_info in image_infos:
        if img_info['frac_iffy'] > iffy_thresh:
            bad_images.append(img_info)

    print('{len(bad_images)=}')
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
    print('bad_stats = {}'.format(ub.repr2(bad_stats, nl=1)))

    bad_gids = [bad['gid'] for bad in bad_images]
    return bad_gids
