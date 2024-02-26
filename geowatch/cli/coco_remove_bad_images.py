#!/bin/env python
import ubelt as ub
import scriptconfig as scfg


class CocoRemoveBadImagesConfig(scfg.DataConfig):
    """
    Remove coco images that are mostly nodata.
    Can also delete the on-disk assets if specified.
    """
    __command__ = 'remove_bad_images'
    __default__ = {
        'src': scfg.Value('data.kwcoco.json', help='input kwcoco filepath', position=1),

        'dst': scfg.Value(None, help='output kwcoco filepath', position=2),

        'workers': scfg.Value(0, type=str, help='number of io threads'),

        'mode': scfg.Value('process', help='can be thread, process, or serial'),

        'channels': scfg.Value(None, help='If specified, check only these channels for bad pixels'),

        'delete_assets': scfg.Value('auto', help='if True actually deletes the assets. If auto and interactive, will ask the user to choose'),

        'interactive': scfg.Value(True, isflag=1, help='if true, ask the user to confirm deletion'),

        # 'overview': scfg.Value('coarsest', help='set to "coarsest" for fastest method, and 0 for most accurate method, or a non negative integer for that level of overview'),
        'overview': scfg.Value(0, help='set to "coarsest" for fastest method, and 0 for most accurate method, or a non negative integer for that level of overview'),
    }


def main(cmdline=True, **kwargs):
    """
    Ignore:
        from geowatch.cli.coco_bad_empty_images import *  # NOQA
        kwargs = {}
        kwargs['src'] = 'imgonly_S2_L8_WV.kwcoco.json'
        kwargs['dst'] = 'imgonly_S2_L8_WV.kwcoco.json.tmp'
        kwargs['workers'] = 8
        # kwargs['channels'] = 'red|green|blue'
        kwargs['channels'] = 'red'
        cmdline = False
    """
    config = CocoRemoveBadImagesConfig.cli(cmdline=cmdline, data=kwargs,
                                           strict=True)
    mode = config['mode']

    import kwcoco
    from kwutil import util_parallel
    from rich.prompt import Confirm
    import safer
    workers = util_parallel.coerce_num_workers(config['workers'])

    main_channels = config['channels']
    if main_channels is not None:
        main_channels = kwcoco.FusedChannelSpec.coerce(main_channels)

    dset = kwcoco.CocoDataset.coerce(config['src'])

    delete_assets = config['delete_assets']
    if delete_assets == 'auto':
        if not config['interactive']:
            delete_assets = False

    overview = config['overview']
    bad_gids = find_empty_images(dset, main_channels, mode=mode,
                                 workers=workers, overview=overview)

    if config['interactive']:
        if delete_assets == 'auto':
            total_bytes = compute_asset_disk_usage(dset, bad_gids, mode, workers)
            total_megabytes = total_bytes / 2 ** 20
            print(f'Total bad space: {total_megabytes:0.4f} MB')

        flag = Confirm.ask('Do you want to remove these empty images from the output kwcoco?')
        if not flag:
            return
        if delete_assets == 'auto':
            delete_assets = Confirm.ask('[red] Do you want to delete the on-disk assets too? (DESTRUCTIVE)')

    if delete_assets:
        bad_fpaths = []
        for bad_gid in ub.ProgIter(bad_gids, desc='collect empty assets'):
            coco_img = dset.coco_image(bad_gid)
            bad_fpaths.extend(list(coco_img.iter_image_filepaths()))

        for bad_fpath in ub.ProgIter(bad_fpaths, desc='delete empty assets'):
            ub.delete(bad_fpath)

    dset.remove_images(bad_gids)

    FIX_ASSET_ORDER = 1
    if FIX_ASSET_ORDER:
        # TODO: this should be part of the crop script
        for img in dset.dataset['images']:
            if 'auxiliary' in img:
                img['auxiliary'] = sorted(img['auxiliary'], key=lambda aux: aux['channels'])

    # dset.fpath = config['dst']
    dst_fpath = config['dst']
    print('Write to dst_fpath = {!r}'.format(dst_fpath))
    with safer.open(dst_fpath, 'w', temp_file=not ub.WIN32) as file:
        dset.dump(file, indent='    ', newlines=True)
    print('Wrote to dst_fpath = {!r}'.format(dst_fpath))


def compute_asset_disk_usage(dset, gids, mode, workers):
    from kwutil import util_progress

    calc_jobs = ub.JobPool(mode=mode, max_workers=workers)

    pman = util_progress.ProgressManager()

    with pman:
        for gid in pman.progiter(gids, desc='calc asset space'):
            coco_img = dset.coco_image(gid)
            for fpath in coco_img.iter_image_filepaths():
                fpath = ub.Path(fpath).resolve()
                calc_jobs.submit(fpath.stat)

        total_bytes = 0
        prog = pman.progiter(calc_jobs.as_completed(), desc='collect size jobs',
                             total=len(calc_jobs))
        for job in prog:
            stat = job.result()
            total_bytes += stat.st_size
            total_megabytes = total_bytes / 2 ** 20
            msg = f'Current size: {total_megabytes:0.4f} MB'
            prog.set_postfix_str(msg)
    return total_bytes


def is_image_empty(coco_img, main_channels=None, overview=-1):
    """
    Run heristics to determine if a coco image is empty.
    """
    import kwimage
    import numpy as np
    import kwcoco
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
                    imdata = kwimage.imread(gpath, backend='gdal', nodata='ma',
                                            overview=overview)
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


def find_empty_images(dset, main_channels, overview=-1, mode='process',
                      workers=0):
    import numpy as np
    import pandas as pd
    from kwutil import util_progress
    import rich

    gid_to_infos = {}
    all_gids = list(dset.index.imgs.keys())

    pool = ub.JobPool(mode=mode, max_workers=workers)
    pman = util_progress.ProgressManager()

    with pman, pool:
        for gid in pman.progiter(all_gids, desc='submit find empty image jobs',
                                 freq=1000, adjust=0):
            if gid not in gid_to_infos:
                coco_img = dset.coco_image(gid).detach()
                job = pool.submit(is_image_empty, coco_img,
                                  main_channels=main_channels, overview=overview)
                job.coco_img = coco_img

        image_infos = []
        num_bad = 0
        prog = pman.progiter(pool.as_completed(), total=len(pool),
                             desc='collect find empty images', freq=1000,
                             adjust=False)
        for job in prog:
            coco_img = job.coco_img
            img_info = job.result()
            if img_info['is_bad']:
                num_bad += 1
                pman.update_info(f'num_empty = {num_bad} / {len(all_gids)}')
            image_infos.append(img_info)

    for img_info in image_infos:
        img_iffys = [b['frac_iffy'] for b in img_info['chan_infos'].values()]
        if img_iffys:
            img_info['frac_iffy'] = min(img_iffys)
        else:
            img_info['frac_iffy'] = -1

    if 1:
        iffy_fracs = [d['frac_iffy'] for d in image_infos]
        iffy_fracs = np.array(iffy_fracs)
        iffy_bins = [-1, 0, 0.25, 0.5, 0.75, 0.85, .90, .95, 0.98, 1.0]
        iffy_freq, iffy_bins = np.histogram(iffy_fracs, bins=iffy_bins)
        iffy_hist = ub.dzip(ub.iter_window(iffy_bins, 2), iffy_freq)
        print('iffy_hist = {}'.format(ub.urepr(iffy_hist, nl=1)))

    # TODO: different iffy thresh per sensor
    iffy_thresh = 0.95
    bad_img_infos = []
    for img_info in image_infos:
        if img_info['frac_iffy'] > iffy_thresh:
            bad_img_infos.append(img_info)

    bad_gids = [b['gid'] for b in bad_img_infos]

    all_images = dset.images()
    bad_images = dset.images(bad_gids)
    sensor_to_num_bad = ub.dict_hist(bad_images.lookup("sensor_coarse"))
    sensor_to_total = ub.dict_hist(all_images.lookup("sensor_coarse"))
    sensor_bad_df = pd.DataFrame({'num_bad': sensor_to_num_bad, 'num_total': sensor_to_total})

    print(f'{len(bad_images)=}')

    print('Sensor Versus num bad / total')
    rich.print(sensor_bad_df.to_string())

    vidname_to_num_bad = ub.dict_hist(dset.videos(bad_images.lookup("video_id")).lookup("name"))
    vidname_to_num_total = ub.dict_hist(dset.videos(all_images.lookup("video_id")).lookup("name"))
    vidname_bad_df = pd.DataFrame({'num_bad': vidname_to_num_bad, 'num_total': vidname_to_num_total})
    vidname_bad_df = vidname_bad_df.fillna(0)
    vidname_bad_df = vidname_bad_df.sort_index()
    print('Video Versus num bad')
    rich.print(vidname_bad_df.to_string())
    # print('sensor_to_num_bad = {}'.format(ub.urepr(sensor_to_num_bad, nl=1)))
    # print('region_to_num_bad = {}'.format(ub.urepr(region_to_num_bad, nl=1)))

    bad_stats = ub.ddict(lambda: 0)
    for bad in bad_img_infos:
        gid = bad['gid']
        coco_img = dset.coco_image(gid)
        for chan, chan_info in bad['chan_infos'].items():
            sensor = coco_img.img["sensor_coarse"]
            if chan_info["max_val"] is np.ma.masked:
                bad_stats[f'{sensor}:{chan}.max_masked'] += 1
            elif chan_info["max_val"] == 0:
                bad_stats[f'{sensor}:{chan}.max_zero'] += 1
                chan_info["num_masked"]
    rich.print('bad_stats = {}'.format(ub.urepr(bad_stats, nl=1)))

    return bad_gids


__config__ = CocoRemoveBadImagesConfig


if __name__ == '__main__':
    """
    CommandLine:
        python ~/code/watch/geowatch/cli/coco_bad_empty_images.py
    """
    main()
