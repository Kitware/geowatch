#!/usr/bin/env python
import scriptconfig as scfg
import ubelt as ub
from watch.utils import kwcoco_extensions  # NOQA


class WatchCocoStats(scfg.Config):
    """
    Print watch-relevant information about a kwcoco dataset

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
        print('config = {}'.format(ub.repr2(config, nl=1)))

        if isinstance(fpaths, str):
            if ',' in fpaths:
                print('warning: might not handle this case well')
            fpaths = [fpaths]

        # TODO: tabulate stats when possible.
        import watch
        collatables = []
        # print('collatables = {!r}'.format(collatables))
        for fpath in ub.ProgIter(fpaths, verbose=3, desc='Load dataset stats'):
            print('--')
            dset = watch.demo.coerce_kwcoco(fpath)
            print('dset = {!r}'.format(dset))
            colltable = coco_watch_stats(dset)
            collatables.append(colltable)

        print('collatables = {}'.format(ub.repr2(collatables, nl=2)))
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
                ub.invert_dict(col_name_map), nl=1)))

        summary = summary.rename(col_name_map, axis=1)
        print(summary.to_string())


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
    print('num_videos = {!r}'.format(num_videos))
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
        video_str = ub.repr2(video, nl=-1, sort=False)
        video_str = util_truncate.smart_truncate(
            video_str, max_length=512, trunc_loc=0.7)
        print('video = {}'.format(video_str))

        frame_dates = dset.images(gids).lookup('date_captured', None)
        frame_dt = sorted([dateutil.parser.parse(d) for d in frame_dates if d])
        if frame_dt:
            date_range = (min(frame_dt).isoformat(), max(frame_dt).isoformat())
        else:
            date_range = None

        video_info = ub.dict_union({
            'name': video['name'],
            'vidid': vidid,
            'sensor_freq': sensor_freq,
            'num_frames': len(gids),
            'date_range': date_range,
        }, video)
        video_info.pop('regions', None)
        video_info.pop('properties', None)
        vid_info_str = ub.repr2(video_info, nl=-1, sort=False)
        vid_info_str = util_truncate.smart_truncate(
            vid_info_str, max_length=512, trunc_loc=0.6)
        print('video_info = {}'.format(vid_info_str))
        all_sensor_entries.extend(all_sensor_entries)
        video_summary_rows.append(ub.dict_diff(video_info, {'sensor_freq', 'warp_wld_to_vid'}))

    print('dset.tag = {!r}'.format(dset.tag))

    basic_stats = dset.basic_stats()
    ext_stats = dset.extended_stats()
    print('basic_stats = {}'.format(ub.repr2(basic_stats, nl=1)))
    print('ext_stats = {}'.format(ub.repr2(ext_stats, nl=1, align=':', precision=3)))

    attrs = dset.videos().attribute_frequency()
    print('video_attrs = {}'.format(ub.repr2(attrs, nl=1)))
    attrs = dset.images().attribute_frequency()
    print('image_attrs = {}'.format(ub.repr2(attrs, nl=1)))
    attrs = dset.annots().attribute_frequency()
    print('annot_attrs = {}'.format(ub.repr2(attrs, nl=1)))

    loose_image_ids = sorted(all_image_ids - all_image_ids_with_video)
    print('len(loose_image_ids) = {!r}'.format(len(loose_image_ids)))

    import pandas as pd
    video_summary = pd.DataFrame(video_summary_rows)
    print(video_summary)

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

    sensor_hist = ub.dict_hist(all_sensor_entries)
    print('Sensor Histogram = {}'.format(ub.repr2(sensor_hist, nl=1)))

    print('MSI channel stats')
    info = kwcoco_extensions.coco_channel_stats(dset)
    print(ub.repr2(info, nl=4))

    import pathlib
    dset_bundle_suffix = '/'.join(pathlib.Path(dset.fpath).parts[-2:])

    colltable = {
        'dset': dset_bundle_suffix,
        **basic_stats,
        **info['chan_hist'],
        **info['sensor_hist'],
    }

    return colltable


_SubConfig = WatchCocoStats

if __name__ == '__main__':
    """
    CommandLine:
        python -m watch.cli.watch_coco_stats --src=special:vidshapes8-multispectral

        smartwatch stats drop1/data.kwcoco.json
    """
    WatchCocoStats.main()
