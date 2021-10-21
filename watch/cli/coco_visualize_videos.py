import kwimage
import ubelt as ub
import kwcoco
import scriptconfig as scfg


class CocoVisualizeConfig(scfg.Config):
    """
    Visualizes annotations on kwcoco video frames on each band

    TODO:
        - [X] Could parameterize which bands are displayed if that is useful
        - [ ] Could finalize by creating an animation if we need these for slides
        - [X] Could parallelize with ub.JobPool

    CommandLine:
        # Point to your kwcoco file
        DVC_DPATH=$HOME/data/dvc-repos/smart_watch_dvc
        COCO_FPATH=$DVC_DPATH/drop1-S2-L8-aligned/data.kwcoco.json

        python -m watch.cli.coco_visualize_videos --src $COCO_FPATH --viz_dpath ./viz_out --channels="red|green|blue" --space="video"

        # Also note you can make an animated gif
        python -m watch.cli.gifify -i "./viz_out/US_Jacksonville_R01/_anns/red|green|blue/" -o US_Jacksonville_R01_anns.gif

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

        # 'channels': scfg.Value(None, type=str, help='only viz these channels'),

        # 'num_frames': scfg.Value('inf', type=str, help='show the first N frames from each video'),


        # TODO: better support for this
        'zoom_to_tracks': scfg.Value(False, type=str, help='if True, zoom to tracked annotations'),
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
    import kwcoco
    import pathlib
    config = CocoVisualizeConfig(default=kwargs, cmdline=cmdline)
    space = config['space']
    channels = config['channels']
    print('config = {}'.format(ub.repr2(dict(config), nl=2)))

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

    pool = ub.JobPool(mode='thread', max_workers=config['num_workers'])

    config['zoom_to_tracks']

    # TODO:
    # from scriptconfig.smartcast import smartcast
    # num = smartcast(config['num_frames'])
    # if isinstance(num, int):
    #     time_sl = slice(0, num)
    # else:
    #     time_sl = slice(None)

    for vidid, video in prog:
        sub_dpath = viz_dpath / video['name']
        sub_dpath.mkdir(parents=True, exist_ok=1)

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

                for gid in gids:
                    img = coco_dset.index.imgs[gid]
                    anns = coco_dset.annots(gid=gid).objs

                    pool.submit(_write_ann_visualizations2,
                                coco_dset, img, anns, track_dpath, space=space,
                                channels=channels, vid_crop_box=vid_crop_box)

        else:
            # gids = gids[0:3] + gids[len(gids) // 2 - 2:len(gids) // 2 + 1] + gids[-3:]
            # time_sl]
            for gid in gids:
                img = coco_dset.index.imgs[gid]
                anns = coco_dset.annots(gid=gid).objs

                pool.submit(_write_ann_visualizations2,
                            coco_dset, img, anns, sub_dpath, space=space,
                            channels=channels)

        for job in ub.ProgIter(pool.as_completed(), total=len(pool), desc='write imgs'):
            job.result()

        pool.jobs.clear()


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


def ensure_false_color(canvas):
    """
    Given a canvas with more than 3 colors, (or 2 colors) do
    something to get it into a colorized space.

    I have no idea how well this works. Probably better methods exist.

    Example:
        >>> demo_img = kwimage.ensure_float01(kwimage.grab_test_image('astro'))
        >>> canvas = demo_img @ np.random.rand(3, 2)
        >>> rgb_canvas2 = ensure_false_color(canvas)
        >>> canvas = np.tile(demo_img, (1, 1, 10))
        >>> rgb_canvas10 = ensure_false_color(canvas)
        >>> import kwplot
        >>> kwplot.autompl()
        >>> kwplot.imshow(rgb_canvas2, pnum=(1, 2, 1))
        >>> kwplot.imshow(rgb_canvas10, pnum=(1, 2, 2))
    """
    import kwarray
    import numpy as np
    canvas = kwarray.atleast_nd(canvas, 3)

    if canvas.shape[2] in {1, 3}:
        rgb_canvas = canvas
    # elif canvas.shape[2] == 2:
    #     # Use LAB to colorize
    #     L_part = np.ones_like(canvas[..., 0:1]) * 50
    #     a_min = -86.1875
    #     a_max = 98.234375
    #     b_min = -107.859375
    #     b_max = 94.46875
    #     a_part = (canvas[..., 0:1] - a_min) / (a_max - a_min)
    #     b_part = (canvas[..., 1:2] - b_min) / (b_max - b_min)
    #     lab_canvas = np.concatenate([L_part, a_part, b_part], axis=2)
    #     rgb_canvas = kwimage.convert_colorspace(lab_canvas, src_space='lab', dst_space='rgb')
    else:
        rng = kwarray.ensure_rng(canvas.shape[2])
        seedmat = rng.rand(canvas.shape[2], 3).T
        h, tau = np.linalg.qr(seedmat, mode='raw')
        false_colored = (canvas @ h)
        rgb_canvas = kwimage.normalize(false_colored)
    return rgb_canvas
    # rgb_canvas = canvas
    # m = np.random.rand(7, 5)
    # q, r = np.linalg.qr(m, mode='raw')
    # print('m.shape = {!r}'.format(m.shape))
    # print('q.shape = {!r}'.format(q.shape))
    # print('r.shape = {!r}'.format(r.shape))


def _write_ann_visualizations2(coco_dset : kwcoco.CocoDataset,
                               img : dict,
                               anns : list,
                               sub_dpath : str,
                               space : str,
                               channels=None,
                               vid_crop_box=None):
    """
    TODO:
        refactor because similar code is also used in coco_align_geotiffs
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

    header_info = []
    header_info.append(vidname)
    if date_captured:
        header_info.append(date_captured + ' ' + sensor_coarse)

    delayed = coco_dset.delayed_load(img['id'], space=space)

    if channels is not None:
        if isinstance(channels, list):
            channels = ','.join(channels)  # hack
        channels = channel_spec.ChannelSpec.coerce(channels)
        chan_groups = channels.streams()
    else:
        coco_img = CocoImage(img)
        chan_groups = coco_img.channels.streams()

    img_view_dpath = sub_dpath / '_imgs'
    ann_view_dpath = sub_dpath / '_anns'
    dets = kwimage.Detections.from_coco_annots(anns, dset=coco_dset)

    if space == 'video':
        vid_from_img = kwimage.Affine.coerce(img['warp_img_to_vid'])
        dets = dets.warp(vid_from_img)

    print('vid_crop_box = {!r}'.format(vid_crop_box))
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
        # overlap = dets.boxes.iooas(vid_crop_box)
        # print('overlap = {!r}'.format(overlap.max()))
        delayed = delayed.crop(crop_box.to_slices()[0])

    for chan_group in chan_groups:
        chan_group = chan_group.spec
        chan = delayed.take_channels(chan_group)

        # spec = str(chan.channels.spec)
        import xdev
        with xdev.embed_on_exception_context:
            canvas = chan.finalize()
            canvas = normalize_intensity(canvas)
            canvas = ensure_false_color(canvas)

        if len(canvas.shape) > 2 and canvas.shape[2] > 4:
            # hack for wv
            canvas = canvas[..., 0]

        img_chan_dpath = img_view_dpath / chan_group
        ann_chan_dpath = ann_view_dpath / chan_group

        ann_chan_dpath.mkdir(parents=True, exist_ok=1)
        img_chan_dpath.mkdir(parents=True, exist_ok=1)

        canvas = kwimage.ensure_float01(canvas)

        suffix = '_'.join([chan_group, sensor_coarse, align_method])

        view_img_fpath = ub.augpath(name, dpath=img_chan_dpath) + '_' + suffix + '.view_img.jpg'

        chan_header_info = header_info.copy()
        chan_header_info.append(chan_group)
        header_text = '\n'.join(chan_header_info)

        img_canvas = kwimage.ensure_uint255(canvas)
        img_canvas = util_kwimage.draw_header_text(image=img_canvas,
                                                   text=header_text,
                                                   stack=True)
        kwimage.imwrite(view_img_fpath, img_canvas)

        view_ann_fpath = ub.augpath(name, dpath=ann_chan_dpath) + '_' + suffix + '.view_ann.jpg'
        try:
            ann_canvas = dets.draw_on(canvas, color='classes')
        except Exception:
            ann_canvas = dets.draw_on(canvas)
        ann_canvas = kwimage.ensure_uint255(ann_canvas)

        ann_canvas = util_kwimage.draw_header_text(image=ann_canvas,
                                                   text=header_text,
                                                   stack=True)
        kwimage.imwrite(view_ann_fpath, ann_canvas)


if __name__ == '__main__':
    """
    CommandLine:
        python ~/code/watch/watch/cli/coco_visualize_videos.py
    """
    main(cmdline=True)
