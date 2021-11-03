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

        COCO_FPATH=/home/joncrall/data/dvc-repos/smart_watch_dvc/drop1-S2-L8-WV-aligned/KR_R001/subdata.kwcoco.json
        COCO_FPATH=/home/joncrall/data/dvc-repos/smart_watch_dvc/drop1-S2-L8-WV-aligned/data.kwcoco.json
        python -m watch.cli.coco_visualize_videos --src $COCO_FPATH --space="image"

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
        # TODO: use the kwcoco_video_data, has good logic for this
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


def _write_ann_visualizations2(coco_dset : kwcoco.CocoDataset,
                               img : dict,
                               anns : list,
                               sub_dpath : str,
                               space : str,
                               channels=None,
                               vid_crop_box=None,
                               request_grouped_bands='default'):
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
        channels = coco_img.channels
        print('---')
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
    # print('anns = {}'.format(ub.repr2(anns, nl=1)))

    try:
        dets = kwimage.Detections.from_coco_annots(anns, dset=coco_dset)
    except Exception:
        # hack
        anns = [ub.dict_diff(ann, ['keypoints']) for ann in anns]
        dets = kwimage.Detections.from_coco_annots(anns, dset=coco_dset)

    if space == 'video':
        vid_from_img = kwimage.Affine.coerce(img['warp_img_to_vid'])
        dets = dets.warp(vid_from_img)

    # print('vid_crop_box = {!r}'.format(vid_crop_box))
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

        # spec = str(chan.channels.spec)
        img_chan_dpath = img_view_dpath / chan_group
        ann_chan_dpath = ann_view_dpath / chan_group
        ann_chan_dpath.mkdir(parents=True, exist_ok=1)
        img_chan_dpath.mkdir(parents=True, exist_ok=1)
        suffix = '_'.join([chan_group, sensor_coarse, align_method])
        view_img_fpath = ub.augpath(name, dpath=img_chan_dpath) + '_' + suffix + '.view_img.jpg'
        view_ann_fpath = ub.augpath(name, dpath=ann_chan_dpath) + '_' + suffix + '.view_ann.jpg'

        chan = delayed.take_channels(chan_group)
        try:
            canvas = chan.finalize()
        except Exception:
            if sensor_coarse in {'L8', 'S2'}:
                bundle_dpath = coco_dset.bundle_dpath
                _hack_check_and_fix_broken(bundle_dpath, img)
                canvas = chan.finalize()
            else:
                raise

        canvas = normalize_intensity(canvas)
        canvas = util_kwimage.ensure_false_color(canvas)

        if len(canvas.shape) > 2 and canvas.shape[2] > 4:
            # hack for wv
            canvas = canvas[..., 0]

        canvas = kwimage.ensure_float01(canvas)

        chan_header_info = header_info.copy()
        chan_header_info.append(chan_group)
        header_text = '\n'.join(chan_header_info)

        img_canvas = kwimage.ensure_uint255(canvas)
        img_canvas = util_kwimage.draw_header_text(image=img_canvas,
                                                   text=header_text,
                                                   stack=True)
        kwimage.imwrite(view_img_fpath, img_canvas)

        try:
            ann_canvas = dets.draw_on(canvas, color='classes')
        except Exception:
            ann_canvas = dets.draw_on(canvas)
        ann_canvas = kwimage.ensure_uint255(ann_canvas)

        ann_canvas = util_kwimage.draw_header_text(image=ann_canvas,
                                                   text=header_text,
                                                   stack=True)
        kwimage.imwrite(view_ann_fpath, ann_canvas)


class GdalErrorHandler(object):
    """
    References:
        https://gdal.org/api/python_gotchas.html#exceptions-raised-in-custom-error-handlers-do-not-get-caught

    SeeAlso:
        'Error',
        'ErrorReset',
        'GARIO_ERROR',
        'GetErrorCounter',
        'GetLastErrorMsg',
        'GetLastErrorNo',
        'GetLastErrorType',
        'OF_VERBOSE_ERROR',
        'PopErrorHandler',
        'PushErrorHandler',
        'SetCurrentErrorHandlerCatchDebug',
        'SetErrorHandler',
        'VSIErrorReset',
        'VSIGetLastErrorMsg',
        'VSIGetLastErrorNo'
    """
    def __init__(self):
        self.err_level = None
        self.err_no = None
        self.err_msg = None
        self.was_using_exceptions = None
        self.reset()

    def handler(self, err_level, err_no, err_msg):
        self.err_level = err_level
        self.err_no = err_no
        self.err_msg = err_msg

    def reset(self):
        from osgeo import gdal
        self.err_level = gdal.CE_None
        self.err_no = 0
        self.err_msg = ''

    def __enter__(self):
        from osgeo import gdal
        self.was_using_exceptions = gdal.GetUseExceptions()
        gdal.UseExceptions()
        gdal.PushErrorHandler(self.handler)

    def __exit__(self, a, b, c):
        from osgeo import gdal
        if not self.was_using_exceptions:
            gdal.DontUseExceptions()
        gdal.PopErrorHandler()


def _hack_check_and_fix_broken(bundle_dpath, img):
    print('HACK CHECK AND FIXING!!!!')
    from os.path import join
    from osgeo import gdal
    from kwcoco.coco_image import CocoImage
    coco_img = CocoImage(img)

    err = GdalErrorHandler()
    bad_bands = []
    with err:
        for obj in coco_img.iter_asset_objs():
            fpath = join(bundle_dpath, obj['file_name'])
            gdal_ds = gdal.Open(fpath, gdal.GA_ReadOnly)
            if err.err_level == gdal.CE_Warning:
                err.reset()
                bad_bands.append(obj['channels'])

            # print('err.err_level = {!r}'.format(err.err_level))
            # for band_idx in range(gdal_ds.RasterCount):
            #     band = gdal_ds.GetRasterBand(band_idx + 1)
            #     print('band_idx = {!r}'.format(band_idx))
            #     print('band = {!r}'.format(band))
            gdal_ds = None  # NOQA

    for chan_group in bad_bands:
        print('BAD chan_group = {!r}'.format(chan_group))
        _hack_check_and_fix_broken(bundle_dpath, img, chan_group)


def _hack_fix_align_warp(bundle_dpath, img, chan_group):
    print('HACK FIXING!!!!')
    import kwimage
    from os.path import join, exists
    # HACK IT: TODO: make the align script to consistency checks
    found = None

    from kwcoco.coco_image import CocoImage
    coco_img = CocoImage(img)

    for obj in coco_img.iter_asset_objs():
        if obj['channels'] == chan_group:
            found = obj
            break

    if found is None:
        import xdev
        xdev.embed()

    parent_fpath = join(bundle_dpath, found['parent_file_name'])
    if not exists(parent_fpath):
        # SUPER HACK
        parent_fpath = join(bundle_dpath, '..', found['parent_file_name'])

    if not exists(parent_fpath):
        raise Exception('cannot fix, cannot find parent')

    bad_fpath = join(bundle_dpath, found['file_name'])
    corner = kwimage.Polygon.coerce(found['geos_corners'])
    lonmax, latmax = corner.data['exterior'].data.max(axis=0)
    lonmin, latmin = corner.data['exterior'].data.min(axis=0)

    import watch
    candidate_utm_codes = [
        watch.gis.spatial_reference.utm_epsg_from_latlon(latmin, lonmin),
        watch.gis.spatial_reference.utm_epsg_from_latlon(latmax, lonmax),
        watch.gis.spatial_reference.utm_epsg_from_latlon(latmax, lonmin),
        watch.gis.spatial_reference.utm_epsg_from_latlon(latmin, lonmax),
        watch.gis.spatial_reference.utm_epsg_from_latlon(
            ((latmin + latmax) / 2), ((lonmin + lonmax) / 2)),
    ]
    utm_epsg_zone = ub.argmax(ub.dict_hist(candidate_utm_codes))

    compress = 'NONE'
    blocksize = 64
    crop_coordinate_srs = 'epsg:4326'
    target_srs = 'epsg:{}'.format(utm_epsg_zone)
    src_gpath = parent_fpath
    # dst_gpath = './tmp.tif'
    dst_gpath = bad_fpath

    # Use the new COG output driver
    prefix_template = (
        '''
        gdalwarp
        -multi
        --config GDAL_CACHEMAX 500 -wm 500
        --debug off
        -te {xmin} {ymin} {xmax} {ymax}
        -te_srs {crop_coordinate_srs}
        -t_srs {target_srs}
        -of COG
        -co OVERVIEWS=NONE
        -co BLOCKSIZE={blocksize}
        -co COMPRESS={compress}
        -co NUM_THREADS=2
        -overwrite
        ''')

    template_kw = {
        'crop_coordinate_srs': crop_coordinate_srs,
        'target_srs': target_srs,
        'ymin': latmin,
        'xmin': lonmin,
        'ymax': latmax,
        'xmax': lonmax,
        'blocksize': blocksize,
        'compress': compress,
        'SRC': src_gpath,
        'DST': dst_gpath,
    }
    template = ub.paragraph(
        prefix_template +
        '{SRC} {DST}')
    command = template.format(**template_kw)
    cmd_info = ub.cmd(command, verbose=0)  # NOQA
    if cmd_info['ret'] != 0:
        print('\n\nCOMMAND FAILED: {!r}'.format(command))
        raise Exception(cmd_info['err'])


if __name__ == '__main__':
    """
    CommandLine:
        python ~/code/watch/watch/cli/coco_visualize_videos.py
    """
    main(cmdline=True)
