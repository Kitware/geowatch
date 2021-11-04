#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
KWCoco video visualization script

CommandLine:
    # A demo of this script on toydata is as follows

    TEMP_DPATH=$(mktemp -d)
    echo "TEMP_DPATH = $TEMP_DPATH"
    cd $TEMP_DPATH

    KWCOCO_BUNDLE_DPATH=$TEMP_DPATH/toy_bundle
    KWCOCO_FPATH=$KWCOCO_BUNDLE_DPATH/data.kwcoco.json
    VIZ_DPATH=$KWCOCO_BUNDLE_DPATH/_viz
    python -m kwcoco toydata --key=vidshapes3-msi-frames7 --dst=$KWCOCO_FPATH
    python -m watch.cli.coco_visualize_videos --src=$KWCOCO_FPATH --viz_dpath=$VIZ_DPATH --animate=True
    python -m watch.cli.coco_visualize_videos --src=$KWCOCO_FPATH --viz_dpath=$VIZ_DPATH --zoom_to_tracks=True --start_frame=1 --num_frames=2 --animate=True
"""
import kwcoco
import kwimage
import pathlib
import scriptconfig as scfg
import ubelt as ub


class CocoVisualizeConfig(scfg.Config):
    """
    Visualizes annotations on kwcoco video frames on each band

    CommandLine:
        # Point to your kwcoco file
        DVC_DPATH=$HOME/data/dvc-repos/smart_watch_dvc
        COCO_FPATH=$DVC_DPATH/drop1-S2-L8-aligned/data.kwcoco.json

        python -m watch.cli.coco_visualize_videos --src $COCO_FPATH --viz_dpath ./viz_out --channels="red|green|blue" --space="video"

        COCO_FPATH=/home/joncrall/data/dvc-repos/smart_watch_dvc/drop1-S2-L8-WV-aligned/KR_R001/subdata.kwcoco.json
        COCO_FPATH=/home/joncrall/data/dvc-repos/smart_watch_dvc/drop1-S2-L8-WV-aligned/data.kwcoco.json
        python -m watch.cli.coco_visualize_videos --src $COCO_FPATH --space="image"

        # Also note you can make an animated gif
        python -m watch.cli.gifify -i "./viz_out/US_Jacksonville_R01/_anns/red|green|blue/" -o US_Jacksonville_R01_anns.gif

        # NEW: as of 2021-11-04 : helper animation script

        python -m watch.cli.animate_visualizations --viz_dpath ./viz_out

    """
    default = {
        'src': scfg.Value('data.kwcoco.json', help='input dataset'),

        'viz_dpath': scfg.Value(None, help=ub.paragraph(
            '''
            Where to save the visualizations. If unspecified,
            writes them adjacent to the input kwcoco file
            ''')),

        'num_workers': scfg.Value(0, help='number of parallel draw jobs'),

        'space': scfg.Value('video', help='can be image or video space'),

        'channels': scfg.Value(None, type=str, help='only viz these channels'),

        'draw_imgs': scfg.Value(True),
        'draw_anns': scfg.Value(True),

        'animate': scfg.Value(False, help='if True, make an animated gif from the output'),

        # 'channels': scfg.Value(None, type=str, help='only viz these channels'),
        'num_frames': scfg.Value(None, type=str, help='show the first N frames from each video, if None, all are shown'),
        'start_frame': scfg.Value(0, type=str, help='If specified each video will start on this frame'),

        # TODO: better support for this
        # TODO: use the kwcoco_video_data, has good logic for this
        'zoom_to_tracks': scfg.Value(False, type=str, help='if True, zoom to tracked annotations. Experimental, might not work perfectly yet.'),
    }


def main(cmdline=True, **kwargs):
    """

    Example:
        >>> import ubelt as ub
        >>> dpath = ub.ensure_app_cache_dir('watch/test/viz_video')
        >>> ub.delete(dpath)
        >>> ub.ensuredir(dpath)
        >>> import kwcoco
        >>> from watch.utils import kwcoco_extensions
        >>> dset = kwcoco.CocoDataset.demo('vidshapes8-multispectral', num_frames=5)
        >>> img = dset.dataset['images'][0]
        >>> coco_img = kwcoco_extensions.CocoImage(img, dset)
        >>> channel_chunks = list(ub.chunks(coco_img.channels.fuse().parsed, chunksize=3))
        >>> channels = ','.join(['|'.join(p) for p in channel_chunks])
        >>> kwargs = {
        >>>     'src': dset.fpath,
        >>>     'viz_dpath': dpath,
        >>>     'space': 'video',
        >>>     'channels': channels,
        >>>     'zoom_to_tracks': True,
        >>> }
        >>> from watch.cli.coco_visualize_videos import *  # NOQA
        >>> cmdline = False
        >>> main(cmdline=cmdline, **kwargs)

    Ignore:
        src = ub.expandpath('$HOME/data/dvc-repos/smart_watch_dvc/drop1-S2-L8-aligned/data.kwcoco.json')
        cmdline = False
        kwargs = {
            'src': src,
        }
    """
    from watch.utils.lightning_ext import util_globals
    config = CocoVisualizeConfig(default=kwargs, cmdline=cmdline)
    space = config['space']
    channels = config['channels']
    print('config = {}'.format(ub.repr2(dict(config), nl=2)))

    num_workers = util_globals.coerce_num_workers(config['num_workers'])
    print('num_workers = {!r}'.format(num_workers))

    coco_dset = kwcoco.CocoDataset.coerce(config['src'])
    print('coco_dset.fpath = {!r}'.format(coco_dset.fpath))
    print('coco_dset = {!r}'.format(coco_dset))

    bundle_dpath = pathlib.Path(coco_dset.bundle_dpath)
    if config['viz_dpath'] is not None:
        viz_dpath = pathlib.Path(config['viz_dpath'])
    else:
        viz_dpath = bundle_dpath / '_viz'

    prog = ub.ProgIter(
        coco_dset.index.videos.items(), total=len(coco_dset.index.videos),
        desc='viz videos', verbose=3)

    pool = ub.JobPool(mode='thread', max_workers=num_workers)

    # TODO:
    from scriptconfig.smartcast import smartcast
    num_frames = smartcast(config['num_frames'])
    start_frame = smartcast(config['start_frame'])
    end_frame = None if num_frames is None else start_frame + num_frames

    video_names = []
    for vidid, video in prog:
        sub_dpath = viz_dpath / video['name']
        sub_dpath.mkdir(parents=True, exist_ok=1)
        video_names.append(video['name'])

        gids = coco_dset.index.vidid_to_gids[vidid]
        if config['zoom_to_tracks']:
            assert space == 'video'
            tid_to_info = video_track_info(coco_dset, vidid)
            for tid, track_info in tid_to_info.items():
                track_dpath = sub_dpath / '_tracks' / 'tid_{:04d}'.format(tid)
                track_dpath.mkdir(parents=True, exist_ok=1)
                vid_crop_box = track_info['full_vid_box']

                # Add context (todo: parameterize how much)
                vid_crop_box = vid_crop_box.scale(1.5, about='center')
                vid_crop_box = vid_crop_box.clip(
                    0, 0, video['width'] - 2, video['height'] - 2)
                vid_crop_box = vid_crop_box.to_xywh()
                vid_crop_box = vid_crop_box.quantize()

                gid_subset = gids[start_frame:end_frame]
                for gid in gid_subset:
                    img = coco_dset.index.imgs[gid]
                    anns = coco_dset.annots(gid=gid).objs

                    _header_hack = f'tid={tid} gid={gid}'
                    pool.submit(_write_ann_visualizations2,
                                coco_dset, img, anns, track_dpath, space=space,
                                channels=channels, vid_crop_box=vid_crop_box,
                                _header_hack=_header_hack)

        else:
            gid_subset = gids[start_frame:end_frame]
            for gid in gid_subset:
                img = coco_dset.index.imgs[gid]
                anns = coco_dset.annots(gid=gid).objs

                pool.submit(_write_ann_visualizations2,
                            coco_dset, img, anns, sub_dpath, space=space,
                            channels=channels,
                            draw_imgs=config['draw_imgs'],
                            draw_anns=config['draw_anns'], _header_hack=None)

        for job in ub.ProgIter(pool.as_completed(), total=len(pool), desc='write imgs'):
            job.result()

        pool.jobs.clear()

    if config['animate']:
        from watch.cli import animate_visualizations
        animate_visualizations.animate_visualizations(
            viz_dpath=viz_dpath,
            channels=channels,
            video_names=video_names,
            draw_imgs=config['draw_imgs'],
            draw_anns=config['draw_anns'],
            num_workers=config['num_workers'],
            zoom_to_tracks=config['zoom_to_tracks'],
        )
        pass


def video_track_info(coco_dset, vidid):
    vid_annots = coco_dset.images(vidid=vidid).annots
    track_ids = set(ub.flatten(vid_annots.lookup('track_id')))
    tid_to_info = {}
    for tid in track_ids:
        track_aids = coco_dset.index.trackid_to_aids[tid]
        vidspace_boxes = []
        for aid in track_aids:
            ann = coco_dset.index.anns[aid]
            gid = ann['image_id']
            img = coco_dset.index.imgs[gid]
            bbox = ann['bbox']
            vid_from_img = kwimage.Affine.coerce(img.get('warp_img_to_vid', None))
            imgspace_box = kwimage.Boxes([bbox], 'xywh')
            vidspace_box = imgspace_box.warp(vid_from_img)
            vidspace_boxes.append(vidspace_box)
        all_vidspace_boxes = kwimage.Boxes.concatenate(vidspace_boxes)
        full_vid_box = all_vidspace_boxes.bounding_box().to_xywh()
        tid_to_info[tid] = {
            'tid': tid,
            'full_vid_box': full_vid_box,
        }
    return tid_to_info


_CLI = CocoVisualizeConfig


def _write_ann_visualizations2(coco_dset : kwcoco.CocoDataset,
                               img : dict,
                               anns : list,
                               sub_dpath : str,
                               space : str,
                               channels=None,
                               vid_crop_box=None,
                               request_grouped_bands='default',
                               draw_imgs=True,
                               draw_anns=True, _header_hack=None):
    """
    Dumps an intensity normalized "space-aligned" kwcoco image visualization
    (with or without annotation overlays) for specific bands to disk.
    """
    # See if we can look at what we made
    from kwcoco import channel_spec
    from watch.utils.util_norm import normalize_intensity
    from watch.utils.kwcoco_extensions import CocoImage
    from watch.utils import util_kwimage

    sensor_coarse = img.get('sensor_coarse', 'unknown')
    align_method = img.get('align_method', 'unknown')
    name = img.get('name', 'unknown')

    vidname = coco_dset.index.videos[img['video_id']]['name']
    date_captured = img.get('date_captured', '')
    gid = img.get('id', None)
    if _header_hack is None:
        _header_hack = f'gid={gid}'
    header_line_infos = [
        [vidname, _header_hack],
        [sensor_coarse, date_captured],
    ]
    header_lines = []
    for line_info in header_line_infos:
        header_line = ' '.join([p for p in line_info if p])
        if header_line:
            header_lines.append(header_line)

    delayed = coco_dset.delayed_load(img['id'], space=space)

    if channels is not None:
        if isinstance(channels, list):
            channels = ','.join(channels)  # hack
        channels = channel_spec.ChannelSpec.coerce(channels)
        chan_groups = channels.streams()
    else:
        coco_img = CocoImage(img)
        channels = coco_img.channels
        if request_grouped_bands == 'default':
            # Use false color for special groups
            request_grouped_bands = [
                'red|green|blue',
                'r|g|b',
                # 'nir|swir16|swir22',
            ]
        for cand in request_grouped_bands:
            cand = kwcoco.FusedChannelSpec.coerce(cand)
            has_cand = (channels & cand).numel() == cand.numel()
            if has_cand:
                channels = channels - cand
                # todo: nicer way to join streams
                channels = kwcoco.ChannelSpec.coerce(channels.spec + ',' + cand.spec)

        initial_groups = channels.streams()
        chan_groups = []
        for group in initial_groups:
            if group.numel() > 3:
                # For large group, just take the first 3 channels
                if group.numel() > 8:
                    group = group.normalize()[0:3]
                    chan_groups.append(group)
                else:
                    # For smaller groups split them into singles
                    for part in group:
                        chan_groups.append(kwcoco.FusedChannelSpec.coerce(part))
            else:
                chan_groups.append(group)

    img_view_dpath = sub_dpath / '_imgs'
    ann_view_dpath = sub_dpath / '_anns'

    try:
        dets = kwimage.Detections.from_coco_annots(anns, dset=coco_dset)
    except Exception:
        # hack
        anns = [ub.dict_diff(ann, ['keypoints']) for ann in anns]
        dets = kwimage.Detections.from_coco_annots(anns, dset=coco_dset)

    if space == 'video':
        vid_from_img = kwimage.Affine.coerce(img['warp_img_to_vid'])
        dets = dets.warp(vid_from_img)

    if vid_crop_box is not None:
        # Ensure the crop box is in the proper space
        if space == 'image':
            img_from_vid = kwimage.Affine.coerce(img['warp_img_to_vid']).inv()
            img_crop_box = vid_crop_box.warp(img_from_vid).quantize()
            crop_box = img_crop_box
        elif space == 'video':
            crop_box = vid_crop_box
        else:
            raise KeyError(space)
        ann_shift = (
            -crop_box.tl_x.ravel()[0],
            -crop_box.tl_y.ravel()[0])
        dets = dets.translate(ann_shift)
        delayed = delayed.crop(crop_box.to_slices()[0])

    for chan_group in chan_groups:
        chan_group = chan_group.spec

        # spec = str(chan.channels.spec)
        img_chan_dpath = img_view_dpath / chan_group
        ann_chan_dpath = ann_view_dpath / chan_group
        ann_chan_dpath.mkdir(parents=True, exist_ok=1)
        img_chan_dpath.mkdir(parents=True, exist_ok=1)
        suffix = '_'.join([chan_group, sensor_coarse, align_method])
        view_img_fpath = ub.augpath(name, dpath=img_chan_dpath) + '_' + suffix + '.view_img.jpg'
        view_ann_fpath = ub.augpath(name, dpath=ann_chan_dpath) + '_' + suffix + '.view_ann.jpg'

        try:
            chan = delayed.take_channels(chan_group)
        except KeyError:
            # hack
            from kwcoco.util import util_delayed_poc
            chan = util_delayed_poc.DelayedChannelConcat([delayed]).take_channels(chan_group)

        canvas = chan.finalize()
        canvas = normalize_intensity(canvas)
        canvas = util_kwimage.ensure_false_color(canvas)

        if len(canvas.shape) > 2 and canvas.shape[2] > 4:
            # hack for wv
            canvas = canvas[..., 0]

        canvas = kwimage.ensure_float01(canvas)

        chan_header_lines = header_lines.copy()
        chan_header_lines.append(chan_group)
        header_text = '\n'.join(chan_header_lines)

        if draw_imgs:
            img_canvas = kwimage.ensure_uint255(canvas)
            img_canvas = util_kwimage.draw_header_text(image=img_canvas,
                                                       text=header_text,
                                                       stack=True,
                                                       fit='shrink')
            kwimage.imwrite(view_img_fpath, img_canvas)

        if draw_anns:
            try:
                ann_canvas = dets.draw_on(canvas, color='classes')
            except Exception:
                ann_canvas = dets.draw_on(canvas)
            ann_canvas = kwimage.ensure_uint255(ann_canvas)

            ann_canvas = util_kwimage.draw_header_text(image=ann_canvas,
                                                       text=header_text,
                                                       stack=True,
                                                       fit='shrink')
            kwimage.imwrite(view_ann_fpath, ann_canvas)


if __name__ == '__main__':
    """
    CommandLine:
        python ~/code/watch/watch/cli/coco_visualize_videos.py
    """
    main(cmdline=True)
