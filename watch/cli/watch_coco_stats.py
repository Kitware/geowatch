#!/usr/bin/env python
import scriptconfig as scfg
import ubelt as ub
import rich
from watch.utils import kwcoco_extensions  # NOQA


class WatchCocoStats(scfg.Config):
    """
    Print watch-relevant information about a kwcoco dataset.

    This provides summary information about:

        * Basic kwcoco stats (number of annotations / images / videos / categories)
        * Average GSDs
        * sensor / channel histograms
        * image / annotation / video attribute historams
        * Breakdowns over sensor / channel / video / dataset
        * Per video summaries

    CommandLine:
        smartwatch stats special:shapes8 vidshapes vidshapes-msi vidshapes-watch

    TODO:
        - [ ] Add other useful watch stats to this script

    SeeAlso:
        kwcoco stats
    """
    default = {
        'src': scfg.Value(
            ['special:shapes8'], nargs='+', help=ub.paragraph(
                '''
                one or more datasets coercables, i.e. a path, live dataset, or
                demodata code. Example demo codes are:
                    special:watch_msi
                    special:vidshapes8
                    special:vidshapes8-msi
                '''), position=1),
    }

    @classmethod
    def main(cls, cmdline=True, **kw):
        import pandas as pd
        config = WatchCocoStats(kw, cmdline=cmdline)

        fpaths = config['src']
        rich.print('config = {}'.format(ub.repr2(config, nl=1, sort=0)))

        if isinstance(fpaths, str):
            if ',' in fpaths:
                print('warning: might not handle this case well')
            fpaths = [fpaths]

        # TODO: tabulate stats when possible.
        import watch
        collatables = []
        video_sensor_rows = []
        all_sensors = set()
        for fpath in ub.ProgIter(fpaths, verbose=3, desc='Load dataset stats'):
            print('\n--- Single Dataset Stats ---')
            dset = watch.demo.coerce_kwcoco(fpath)
            print('dset = {!r}'.format(dset))
            stat_info = coco_watch_stats(dset)

            collatable = {
                'dset': stat_info['dset'],
                **stat_info['basic_stats'],
                **stat_info['chan_hist'],
                **stat_info['sensor_hist'],
                **stat_info['sensorchan_hist2'],
            }
            collatables.append(collatable)

            for video_info_row in stat_info['video_summary_rows']:
                video_sensor_freq = video_info_row['sensor_freq']
                all_sensors.update(set(video_sensor_freq))
                video_sensor_row = {
                    'dset': stat_info['dset'],
                    'name': video_info_row['name'],
                    **video_sensor_freq,
                }
                video_sensor_rows.append(video_sensor_row)

        print('\n--- Multi Dataset Stats --')

        import math
        all_sensors = sorted(all_sensors)
        if video_sensor_rows:
            video_sensor_df = pd.DataFrame(video_sensor_rows)
            piv = video_sensor_df.pivot(index=['name', 'dset'], columns=[], values=all_sensors)
            piv = piv.sort_index()
            piv = piv.astype(object)
            piv = piv.applymap(lambda x: None if math.isnan(x) else int(x))
            piv['total'] = piv.sum(axis=1)
            print('Per-Video Sensor Frequency')
            rich.print(piv.to_string(float_format='%0.0f'))
        else:
            print('No per-video stats')

        print('collatables = {}'.format(ub.repr2(collatables, nl=2, sort=0)))
        summary = pd.DataFrame(collatables)

        from watch.utils import slugify_ext
        col_name_map = {}
        for cname in summary.columns:
            new_cname = slugify_ext.smart_truncate(
                cname, max_length=10, trunc_loc=1.0)
            if cname != new_cname:
                col_name_map[cname] = new_cname

        if col_name_map:
            print('Remap names for readability:')
            print('col_name_map = {}'.format(ub.repr2(
                ub.invert_dict(col_name_map), nl=1, sort=0)))

        summary = summary.rename(col_name_map, axis=1)
        summary_string = summary.to_string()
        max_colwidth = max(map(len, summary_string.split('\n')))
        COLWIDTH_LIMIT = 1600
        if max_colwidth > COLWIDTH_LIMIT:
            rich.print(summary)
        else:
            rich.print(summary_string)

        print('Other helpful commands:')
        for fpath in fpaths:
            'smartwatch visualize {fpath:!r} --channels='
            pass


def coco_watch_stats(dset):
    """
    Args:
        dset (kwcoco.CocoDataset)

    Example:
        >>> from watch.cli.watch_coco_stats import *  # NOQA
        >>> from watch.demo import smart_kwcoco_demodata
        >>> dset = smart_kwcoco_demodata.demo_smart_aligned_kwcoco()
        >>> coco_watch_stats(dset)
    """
    from kwcoco.util import util_truncate
    import dateutil
    num_videos = len(dset.index.videos)
    rich.print('num_videos = {!r}'.format(num_videos))
    print('Per-video stats summary')

    video_summary_rows = []

    all_image_ids = set(dset.images())
    all_image_ids_with_video = set()

    all_sensor_entries = []
    for vidid, gids in dset.index.vidid_to_gids.items():
        all_image_ids_with_video.update(gids)
        avail_sensors = dset.images(gids).lookup('sensor_coarse', None)
        sensor_freq = ub.dict_hist(avail_sensors)
        video = dset.index.videos[vidid]
        video = ub.dict_diff(video, ['regions', 'properties'])
        # video_str = ub.repr2(video, nl=-1, sort=False)
        # video_str = util_truncate.smart_truncate(
        #     video_str, max_length=512, trunc_loc=0.7)
        # print('video = {}'.format(video_str))

        frame_dates = dset.images(gids).lookup('date_captured', None)
        frame_dt = sorted([dateutil.parser.parse(d) for d in frame_dates if d])
        if frame_dt:
            date_range = (min(frame_dt).date().isoformat(), max(frame_dt).date().isoformat())
        else:
            date_range = None

        video_info = ub.udict({
            'name': video['name'],
            **ub.dict_isect(video, ['width', 'height']),
            'num_frames': len(gids),
            'sensor_freq': sensor_freq,
            'date_range': date_range,
        }) | video
        video_info.pop('regions', None)
        video_info.pop('properties', None)
        vid_info_str = ub.repr2(video_info, nl=-1, sort=False)
        vid_info_str = util_truncate.smart_truncate(
            vid_info_str, max_length=512, trunc_loc=0.6)
        print('video_info = {}'.format(vid_info_str))
        all_sensor_entries.extend(avail_sensors)
        # video_summary_rows.append(ub.dict_diff(video_info, {'sensor_freq', 'warp_wld_to_vid'}))
        video_summary_rows.append(video_info - {'warp_wld_to_vid'})

    print('dset.tag = {!r}'.format(dset.tag))

    basic_stats = dset.basic_stats()
    ext_stats = dset.extended_stats()
    rich.print('basic_stats = {}'.format(ub.repr2(basic_stats, nl=1, sort=0)))
    rich.print('ext_stats = {}'.format(ub.repr2(ext_stats, nl=1, align=':', precision=3)))

    attrs = dset.videos().attribute_frequency()
    rich.print('video_attrs = {}'.format(ub.repr2(attrs, nl=1, sort=0)))
    attrs = dset.images().attribute_frequency()
    rich.print('image_attrs = {}'.format(ub.repr2(attrs, nl=1, sort=0)))
    attrs = dset.annots().attribute_frequency()
    rich.print('annot_attrs = {}'.format(ub.repr2(attrs, nl=1, sort=0)))

    loose_image_ids = sorted(all_image_ids - all_image_ids_with_video)
    rich.print('len(loose_image_ids) = {!r}'.format(len(loose_image_ids)))

    import pandas as pd
    video_summary = pd.DataFrame(video_summary_rows)
    video_summary = video_summary.drop(video_summary.columns & [
        'valid_region_geos', 'wld_crs_info', 'valid_region'], axis=1)
    rich.print(video_summary)

    # coco_dset = dset
    # all_images = coco_dset.images()
    # wv_images = all_images.compress([s == 'WV' for s in all_images.lookup('sensor_coarse')])
    # coco_images = [coco_dset.coco_image(gid) for gid in wv_images]
    # ub.dict_hist(['|'.join(sorted(coco_img.channels.fuse().parsed)) for coco_img in coco_images])
    # ub.dict_hist([(coco_img.channels.fuse() & kwcoco.FusedChannelSpec.coerce('red|green|blue|panchromatic')).spec for coco_img in coco_images])
    # all_images.lookup('sensor_coarse')

    # coco_img = dset.images().take([0]).coco_images[0]
    # fpath = coco_img.primary_image_filepath()
    # _ = ub.cmd('gdalinfo {}'.format(fpath), verbose=3)

    sensorchan_gsd_stats = coco_sensorchan_gsd_stats(dset)
    rich.print(sensorchan_gsd_stats)

    sensor_hist = ub.dict_hist(all_sensor_entries)
    rich.print('Sensor Histogram = {}'.format(ub.repr2(sensor_hist, nl=1, sort=0)))

    print('MSI channel stats')
    info = kwcoco_extensions.coco_channel_stats(dset)
    rich.print(ub.repr2(info, nl=4, sort=0))

    dset_bundle_suffix = '/'.join(ub.Path(dset.fpath).parts[-2:])

    stat_info = {
        'dset': dset_bundle_suffix,
        'basic_stats': basic_stats,
        'chan_hist': info['chan_hist'],
        'sensor_hist': info['sensor_hist'],
        'sensorchan_hist2': info['sensorchan_hist2'],
        'video_summary_rows': video_summary_rows,
    }
    return stat_info


def coco_sensorchan_gsd_stats(coco_dset):
    """
    Checks the GSD of each band.
    """
    import pandas as pd
    import math
    import numpy as np
    import kwimage
    longform_rows = []
    for image_id in coco_dset.images():
        coco_img = coco_dset.coco_image(image_id)

        asset_rows = []
        assets = list(coco_img.iter_asset_objs())
        missing_gsd_idxs = []
        for idx, asset in enumerate(assets):
            gsd = asset.get('approx_meter_gsd', float('nan'))
            sensor = asset.get('sensor_coarse', '*')
            channels = asset.get('channels', '?')
            asset_rows.append({
                'sensor': sensor,
                'channels': channels,
                'gsd': gsd,
            })
            if math.isnan(gsd):
                missing_gsd_idxs.append(idx)

        if missing_gsd_idxs:
            # If we have a GSD for some but not all assets,
            # we can relate them.
            flags = ~np.array(ub.boolmask(missing_gsd_idxs, len(assets)))
            if np.any(flags):
                reference_idx = np.where(flags[0])[0][0]
                ref_asset = assets[reference_idx]
                img_from_ref = kwimage.Affine.coerce(
                    ref_asset.get('warp_aux_to_img', ref_asset.get('warp_asset_to_img')))
                for miss_idx in missing_gsd_idxs:
                    mis_asset = assets[miss_idx]
                    img_from_mis = kwimage.Affine.coerce(
                        mis_asset.get('warp_aux_to_img', mis_asset.get('warp_asset_to_img')))
                    mis_from_img = img_from_mis.inv()
                    mis_from_ref = mis_from_img @ img_from_ref
                    approx_scale = np.mean(mis_from_ref.decompose()['scale'])
                    mis_gsd = ref_asset['approx_meter_gsd'] / approx_scale
                    asset_rows[miss_idx]['gsd'] = mis_gsd

        longform_rows.extend(asset_rows)

    gsd_table = pd.DataFrame(longform_rows)
    groups = gsd_table.groupby(['sensor', 'channels'])
    sensorchan_gsd_stats = groups.describe()
    return sensorchan_gsd_stats


_SubConfig = WatchCocoStats

if __name__ == '__main__':
    """
    CommandLine:
        python -m watch.cli.watch_coco_stats --src=special:vidshapes8-multispectral

        smartwatch stats drop1/data.kwcoco.json
    """
    WatchCocoStats.main()
