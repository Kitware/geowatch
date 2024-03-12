#!/usr/bin/env python3
import scriptconfig as scfg
import ubelt as ub


class WatchCocoStats(scfg.DataConfig):
    """
    Print geowatch-relevant information about a kwcoco dataset.

    This provides summary information about:

        * Basic kwcoco stats (number of annotations / images / videos / categories)
        * Average GSDs
        * sensor / channel histograms
        * image / annotation / video attribute historams
        * Breakdowns over sensor / channel / video / dataset
        * Per video summaries

    CommandLine:
        geowatch stats special:shapes8 vidshapes vidshapes-msi vidshapes-geowatch

    TODO:
        - [ ] Add other useful geowatch stats to this script

    SeeAlso:
        kwcoco stats
    """
    __command__ = 'stats'
    __default__ = {
        'src': scfg.Value(
            None, nargs='+', help=ub.paragraph(
                '''
                one or more datasets coercables, i.e. a path, live dataset, or
                demodata code. Example demo codes are:
                    special:watch_msi
                    special:vidshapes8
                    special:vidshapes8-msi
                '''), position=1),

        'io_workers': scfg.Value('avail', help='number of workers used to read multiple datasets'),

        'with_video_info': scfg.Value(
            False, isflag=True, help='Show more per-video details')
    }

    @classmethod
    def main(cls, cmdline=True, **kw):
        """
        Example:
            >>> from geowatch.cli import watch_coco_stats
            >>> import geowatch
            >>> dset1 = geowatch.coerce_kwcoco('geowatch-msi', geodata=True, dates=True, heatmap=True)
            >>> dset2 = geowatch.coerce_kwcoco('vidshapes8')
            >>> kw = dict(src=[dset1.fpath, dset2.fpath])
            >>> cmdline = 0
            >>> watch_coco_stats.__config__.main(cmdline=cmdline, **kw)

        Example:
            >>> from geowatch.cli import watch_coco_stats
            >>> import geowatch
            >>> dset1 = geowatch.coerce_kwcoco('geowatch-msi', geodata=True, dates=True, heatmap=True)
            >>> kw = dict(src=dset1.fpath)
            >>> cmdline = 0
            >>> watch_coco_stats.__config__.main(cmdline=cmdline, **kw)
        """
        config = WatchCocoStats.cli(data=kw, cmdline=cmdline, strict=True)
        import rich
        rich.print('config = {}'.format(ub.urepr(config, nl=1, sort=0)))

        import pandas as pd
        import kwcoco
        import math
        import os
        from kwutil import slugify_ext

        fpaths = config['src']

        if isinstance(fpaths, os.PathLike):
            fpaths = [fpaths]

        if isinstance(fpaths, str):
            if ',' in fpaths:
                print('warning: might not handle this case well')
            fpaths = [fpaths]

        if fpaths is None or len(fpaths) == 0:
            raise ValueError('no files to compute stats on')

        # TODO: tabulate stats when possible.
        collatables = []
        video_sensor_rows = []
        all_sensors = set()

        dset_iter = kwcoco.CocoDataset.coerce_multiple(
            fpaths, workers=config.io_workers)
        for dset in dset_iter:
            print('\n--- Single Dataset Stats ---')
            # dset = geowatch.coerce_kwcoco(fpath)
            print('dset = {!r}'.format(dset))
            stat_info = coco_watch_stats(
                dset, with_video_info=config['with_video_info'])

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

        try:
            all_sensors = sorted(all_sensors)
        except TypeError:
            ...
        if video_sensor_rows:
            if config['with_video_info']:
                video_sensor_df = pd.DataFrame(video_sensor_rows)
                piv = video_sensor_df.pivot(index=['name', 'dset'], columns=[], values=all_sensors)
                piv = piv.sort_index()
                piv = piv.astype(object)
                piv = piv.applymap(lambda x: None if math.isnan(x) else int(x))
                piv['total'] = piv.sum(axis=1)
                print('Per-Video Sensor Frequency')
                rich.print(piv.to_string(float_format='%0.0f', max_rows=500))
        else:
            print('No per-video stats')

        print('collatables = {}'.format(ub.urepr(collatables, nl=2, sort=0)))
        summary = pd.DataFrame(collatables)

        col_name_map = {}
        for cname in summary.columns:
            new_cname = slugify_ext.smart_truncate(
                cname, max_length=10, trunc_loc=1.0)
            if cname != new_cname:
                col_name_map[cname] = new_cname

        if col_name_map:
            print('Remap names for readability:')
            print('col_name_map = {}'.format(ub.urepr(
                ub.invert_dict(col_name_map), nl=1, sort=0)))

        summary = summary.rename(col_name_map, axis=1)
        summary_string = summary.to_string(max_rows=500)
        max_colwidth = max(map(len, summary_string.split('\n')))
        COLWIDTH_LIMIT = 1600
        if max_colwidth > COLWIDTH_LIMIT:
            rich.print(summary)
        else:
            rich.print(summary_string)

        # print('Other helpful commands:')
        # for fpath in fpaths:
        #     'geowatch visualize {fpath:!r} --channels='
        #     pass


def coco_watch_stats(dset, with_video_info=False):
    """
    Args:
        dset (kwcoco.CocoDataset)

    Returns:
        Dict[str, Any]: stat_info

    Example:
        >>> from geowatch.cli.watch_coco_stats import *  # NOQA
        >>> import geowatch
        >>> dset = geowatch.coerce_kwcoco('geowatch-msi-geodata-heatmap-dates')
        >>> stat_info = coco_watch_stats(dset)
    """
    from kwutil import slugify_ext
    from kwutil import util_time
    from geowatch.utils import kwcoco_extensions
    import rich
    import pandas as pd
    num_videos = len(dset.index.videos)
    rich.print('num_videos = {!r}'.format(num_videos))
    print('Per-video stats summary')

    video_summary_rows = []
    image_rows = []

    all_image_ids = set(dset.images())
    all_image_ids_with_video = set()

    all_sensor_entries = []
    for vidid, gids in dset.index.vidid_to_gids.items():
        all_image_ids_with_video.update(gids)
        video = dset.index.videos[vidid]
        video = ub.dict_diff(video, ['regions', 'properties'])
        # video_str = ub.urepr(video, nl=-1, sort=False)
        # video_str = slugify_ext.smart_truncate(
        #     video_str, max_length=512, trunc_loc=0.7)
        # print('video = {}'.format(video_str))

        images = dset.images(gids)
        annots_per_img = images.annots

        flat_annots = dset.annots(list(ub.flatten(annots_per_img)))
        # annots_per_img.lookup('track_id', None)
        unique_trackids = set(flat_annots.lookup('track_id', None))
        num_tracks = len(unique_trackids - {None})
        num_annots = len(flat_annots)

        # catname_freq = ub.udict(ub.dict_hist(
        #     flat_annots.lookup('category_id'))).map_keys(
        #         lambda x: dset._resolve_to_cat(x)['name'])

        avail_sensors = images.lookup('sensor_coarse', None)
        frame_dates = images.lookup('date_captured', None)
        sensor_freq = ub.dict_hist(avail_sensors)
        frame_dt = sorted([util_time.coerce_datetime(d) for d in frame_dates if d])
        if frame_dt:
            date_range = (min(frame_dt).date().isoformat(), max(frame_dt).date().isoformat())
        else:
            date_range = None

        for img in images.objs:
            dt = util_time.coerce_datetime(img.get('date_captured', None))
            image_rows.append({
                'video': video['name'],
                'year': None if dt is None else dt.year,
                'sensor': img.get('sensor_coarse', None)
            })

        video_info = ub.udict({
            'name': video['name'],
            **ub.dict_isect(video, ['width', 'height']),
            'num_frames': len(gids),
            'num_tracks': num_tracks,
            'num_annots': num_annots,
            # 'catname_freq': catname_freq,
            'sensor_freq': sensor_freq,
            'date_range': date_range,
        }) | video
        video_info.pop('regions', None)
        video_info.pop('properties', None)
        vid_info_str = ub.urepr(video_info, nl=-1, sort=False)
        vid_info_str = slugify_ext.smart_truncate(
            vid_info_str, max_length=512, trunc_loc=0.6)

        if with_video_info:
            print('video_info = {}'.format(vid_info_str))
        all_sensor_entries.extend(avail_sensors)
        # video_summary_rows.append(ub.dict_diff(video_info, {'sensor_freq', 'warp_wld_to_vid'}))
        video_summary_rows.append(video_info - {'warp_wld_to_vid'})

    print('dset.tag = {!r}'.format(dset.tag))

    basic_stats = dset.basic_stats()

    # Note: kwcoco should be doing this.
    basic_stats['n_tracks'] = len(dset.index.trackid_to_aids)

    ext_stats = dset.extended_stats()
    rich.print('basic_stats = {}'.format(ub.urepr(basic_stats, nl=1, sort=0)))
    rich.print('ext_stats = {}'.format(ub.urepr(ext_stats, nl=1, align=':', precision=3)))

    attrs = dset.videos().attribute_frequency()
    rich.print('video_attrs = {}'.format(ub.urepr(attrs, nl=1, sort=0)))
    attrs = dset.images().attribute_frequency()
    rich.print('image_attrs = {}'.format(ub.urepr(attrs, nl=1, sort=0)))
    attrs = dset.annots().attribute_frequency()
    rich.print('annot_attrs = {}'.format(ub.urepr(attrs, nl=1, sort=0)))

    loose_image_ids = sorted(all_image_ids - all_image_ids_with_video)
    rich.print('len(loose_image_ids) = {!r}'.format(len(loose_image_ids)))

    video_summary = pd.DataFrame(video_summary_rows)
    video_summary = video_summary.drop(video_summary.columns.intersection([
        'valid_region_geos', 'wld_crs_info', 'valid_region']), axis=1)
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

    image_df = pd.DataFrame(image_rows)
    try:
        year_pivot = build_year_summary(image_df)
    except (TypeError, KeyError):
        print('unable to build year analysis')
        year_pivot = None
    else:
        rich.print('Sensor Date Range Histograms')
        rich.print(year_pivot.to_string(max_rows=500))

    sensorchan_gsd_stats = coco_sensorchan_gsd_stats(dset)
    rich.print(sensorchan_gsd_stats.to_string(max_rows=500))

    sensor_hist = ub.dict_hist(all_sensor_entries)
    rich.print('Sensor Histogram = {}'.format(ub.urepr(sensor_hist, nl=1, sort=0)))

    print('MSI channel stats')
    info = kwcoco_extensions.coco_channel_stats(dset)
    rich.print(ub.urepr(info, nl=4, sort=0))

    dset_bundle_suffix = '/'.join(ub.Path(dset.fpath).parts[-2:])

    stat_info = {
        'dset': dset_bundle_suffix,
        'basic_stats': basic_stats,
        'chan_hist': info['chan_hist'],
        'sensor_hist': info['sensor_hist'],
        'sensorchan_hist2': info['sensorchan_hist2'],
        'video_summary_rows': video_summary_rows,
        'year_pivot': year_pivot,
    }
    return stat_info


def build_year_summary(image_df):
    import pandas as pd
    import numpy as np
    _, year_bins = np.histogram(image_df['year'])
    year_bins = sorted(np.unique(np.ceil(year_bins)))
    year_bins = year_bins + [year_bins[-1] + 1]
    # max_bins = max(18 // len(image_df['sensor'].unique()), 2)
    max_bins = 15
    if len(year_bins) > max_bins:
        _, year_bins = np.histogram(image_df['year'], bins=max_bins)
        year_bins = sorted(np.unique(np.ceil(year_bins)))

    bin_labels = []
    for a, b in ub.iter_window(year_bins, 2):
        a = int(a)
        b = int(b)
        if a == b or (a + 1) == b:
            bin_labels += [str(a)]
        else:
            bin_labels += [f'{a} - {b}']

    year_summaries = []
    group_keys = ['video', 'sensor']
    for group_vals, group in image_df.groupby(group_keys):
        counts = np.histogram(group['year'], bins=year_bins)[0]
        group_id = ub.dzip(group_keys, group_vals)
        summaries = [{'count': c, 'time': b, **group_id} for c, b in zip(counts, bin_labels)]
        year_summaries.extend(summaries)
    year_summary_df = pd.DataFrame(year_summaries)
    year_summary_df = year_summary_df.sort_values('time')

    year_pivot = year_summary_df.pivot(
        index=['video', 'sensor'], columns=['time'], values=['count'])
    year_pivot = year_pivot.fillna('0').astype(int)
    return year_pivot


def coco_sensorchan_gsd_stats(coco_dset):
    """
    Checks the GSD of each band.
    """
    import pandas as pd
    import math
    import numpy as np
    import kwimage
    from geowatch.utils import util_pandas
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
    groupers = list(gsd_table.columns.intersection(['sensor', 'channels']))
    if len(groupers) == 0:
        sensorchan_gsd_stats = gsd_table
    else:
        print(f'groupers={groupers}')
        groups = util_pandas.pandas_fixed_groupby(gsd_table, groupers)
        sensorchan_gsd_stats = groups.describe()
    return sensorchan_gsd_stats


__config__ = WatchCocoStats
main = __config__.main

if __name__ == '__main__':
    """
    CommandLine:
        python -m geowatch.cli.watch_coco_stats --src=special:vidshapes8-multispectral

        geowatch stats drop1/data.kwcoco.json
    """
    WatchCocoStats.main()
