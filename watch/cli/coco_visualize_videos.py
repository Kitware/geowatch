#!/usr/bin/env python
"""
KWCoco video visualization script

TODO:
    - [ ] Option to interpret a channel as a heatmap and overlay it on top of
          another set of channels interpreted as a grayscale image.

    - [ ] Migrate to kwcoco proper

    - [ ] Show valid image regions

CommandLine:
    # A demo of this script on toydata is as follows

    # TEMP_DPATH=$(mktemp -d)
    TEMP_DPATH=$HOME/.cache/kwcoco/demo/viz
    mkdir -p $TEMP_DPATH
    echo "TEMP_DPATH = $TEMP_DPATH"
    cd $TEMP_DPATH
    KWCOCO_BUNDLE_DPATH=$TEMP_DPATH/toy_bundle
    KWCOCO_FPATH=$KWCOCO_BUNDLE_DPATH/data.kwcoco.json
    VIZ_DPATH=$KWCOCO_BUNDLE_DPATH/_viz
    python -m kwcoco toydata --key=vidshapes3-msi-multisensor-frames7 --dst=$KWCOCO_FPATH

    python -m watch.cli.coco_visualize_videos --src=$KWCOCO_FPATH --viz_dpath=$VIZ_DPATH --animate=True --workers=0 --any3=only

    python -m watch.cli.coco_visualize_videos --src=$KWCOCO_FPATH --viz_dpath=$VIZ_DPATH --zoom_to_tracks=True --start_frame=1 --num_frames=2 --animate=True
"""
import kwcoco
import kwimage
import scriptconfig as scfg
import numpy as np
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
        'src': scfg.Value('data.kwcoco.json', help='input dataset', position=1),

        'viz_dpath': scfg.Value(None, help=ub.paragraph(
            '''
            Where to save the visualizations. If unspecified,
            writes them adjacent to the input kwcoco file
            ''')),

        'workers': scfg.Value(0, help='number of parallel procs'),
        'max_workers': scfg.Value(None, help='DEPRECATED USE workers'),

        'space': scfg.Value('video', help='can be image or video space'),

        'channels': scfg.Value(None, type=str, help='only viz these channels'),

        'any3': scfg.Value(False, help='if True, ensure the "any3" channels are drawn. If set to "only", then other per-channel visualizations are supressed. TODO: better name?'),

        'draw_imgs': scfg.Value(True),
        'draw_anns': scfg.Value('auto', help='auto means only draw anns if they exist'),

        'cmap': scfg.Value('viridis', help='colormap for single channel data'),

        'animate': scfg.Value(False, help='if True, make an animated gif from the output'),

        # 'channels': scfg.Value(None, type=str, help='only viz these channels'),
        'num_frames': scfg.Value(None, type=str, help='show the first N frames from each video, if None, all are shown'),
        'start_frame': scfg.Value(0, type=str, help='If specified each video will start on this frame'),

        'skip_missing': scfg.Value(True, help='If true, skip any image that does not have the requested channels. Otherwise a nan image will be shown'),

        'only_boxes': scfg.Value(False, help='If false, draws full annotation - which can be time consuming if there are a lot'),

        # TODO: better support for this
        # TODO: use the kwcoco_video_data, has good logic for this
        'zoom_to_tracks': scfg.Value(False, type=str, help='if True, zoom to tracked annotations. Experimental, might not work perfectly yet.'),

        'norm_over_time': scfg.Value(False, help='if True, normalize data over time'),

        'fixed_normalization_scheme': scfg.Value(
            None, type=str, help='Use a fixed normalization scheme for visualization; e.g. "scaled_25percentile"'),

        'extra_header': scfg.Value(None, help='extra text to include in the header'),

        'include_sensors': scfg.Value(None, help='if specified can be comma separated valid sensors'),
        'exclude_sensors': scfg.Value(None, help='if specified can be comma separated invalid sensors'),

        'select_images': scfg.Value(
            None, type=str, help=ub.paragraph(
                '''
                A json query (via the jq spec) that specifies which images
                belong in the subset. Note, this is a passed as the body of
                the following jq query format string to filter valid ids
                '.images[] | select({select_images}) | .id'.

                Examples for this argument are as follows:
                '.id < 3' will select all image ids less than 3.
                '.file_name | test(".*png")' will select only images with
                file names that end with png.
                '.file_name | test(".*png") | not' will select only images
                with file names that do not end with png.
                '.myattr == "foo"' will select only image dictionaries
                where the value of myattr is "foo".
                '.id < 3 and (.file_name | test(".*png"))' will select only
                images with id less than 3 that are also pngs.
                .myattr | in({"val1": 1, "val4": 1}) will take images
                where myattr is either val1 or val4.

                Requries the "jq" python library is installed.
                ''')),

        'select_videos': scfg.Value(
            None, help=ub.paragraph(
                '''
                A json query (via the jq spec) that specifies which videos
                belong in the subset. Note, this is a passed as the body of
                the following jq query format string to filter valid ids
                '.videos[] | select({select_images}) | .id'.

                Examples for this argument are as follows:
                '.name | startswith("foo")' will select only videos
                where the name starts with foo.

                Only applicable for dataset that contain videos.

                Requries the "jq" python library is installed.
                ''')),

        'verbose': scfg.Value(0, help='verbosity level')
    }


def _dataset_id(coco_dset):
    """ A possible good default for a coco candidate name """
    try:
        if hasattr(coco_dset, '_dataset_id'):
            return coco_dset._dataset_id()
    except Exception:
        return _hack_dataset_id(coco_dset)


def main(cmdline=True, **kwargs):
    """

    Example:
        >>> import ubelt as ub
        >>> dpath = ub.Path.appdir('watch/test/viz_video').delete().ensuredir()
        >>> import kwcoco
        >>> from watch.utils import kwcoco_extensions
        >>> dset = kwcoco.CocoDataset.demo('vidshapes8-multispectral', num_frames=2)
        >>> img = dset.dataset['images'][0]
        >>> coco_img = dset.coco_image(img['id'])
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

    if config['max_workers'] is not None:
        max_workers = util_globals.coerce_num_workers(config['max_workers'])
    else:
        max_workers = util_globals.coerce_num_workers(config['workers'])
    print('max_workers = {!r}'.format(max_workers))

    coco_dset = kwcoco.CocoDataset.coerce(config['src'])
    print('coco_dset.fpath = {!r}'.format(coco_dset.fpath))
    print('coco_dset = {!r}'.format(coco_dset))

    if config['draw_anns'] == 'auto':
        config['draw_anns'] = coco_dset.n_annots > 0

    bundle_dpath = ub.Path(coco_dset.bundle_dpath)
    dset_idstr = _dataset_id(coco_dset)
    if config['viz_dpath'] is not None:
        viz_dpath = ub.Path(config['viz_dpath'])
    else:
        viz_dpath = bundle_dpath / '_viz_{}'.format(dset_idstr)
    print('viz_dpath = {!r}'.format(viz_dpath))

    prog = ub.ProgIter(
        coco_dset.index.videos.items(), total=len(coco_dset.index.videos),
        desc='viz videos', verbose=3)

    pool = ub.JobPool(mode='thread', max_workers=max_workers)

    from scriptconfig.smartcast import smartcast
    num_frames = smartcast(config['num_frames'])
    start_frame = smartcast(config['start_frame'])
    end_frame = None if num_frames is None else start_frame + num_frames

    from watch.utils import kwcoco_extensions
    selected_gids = None
    selected_gids = kwcoco_extensions.filter_image_ids(
        coco_dset,
        gids=selected_gids,
        include_sensors=config['include_sensors'],
        exclude_sensors=config['exclude_sensors'],
        select_images=config['select_images'],
        select_videos=config['select_videos'],
    )

    if config['skip_missing'] and channels is not None:
        requested_channels = kwcoco.ChannelSpec.coerce(channels).fuse().as_set()
        coco_images = coco_dset.images(selected_gids).coco_images
        keep = []
        for coco_img in coco_images:
            code = coco_img.channels.fuse().as_set()
            if requested_channels & code:
                keep.append(coco_img.img['id'])
        print(f'Filtered {len(coco_images) - len(keep)} images without requested channels. Keeping {len(keep)}')
        selected_gids = keep

    video_names = []
    for vidid, video in prog:
        sub_dpath = viz_dpath / video['name']

        gids = coco_dset.index.vidid_to_gids[vidid]
        if selected_gids is not None:
            gids = list(ub.oset(gids) & set(selected_gids))

        if len(gids) == 0:
            print(f'Skip {video["name"]=!r} with no selected images')
            continue

        sub_dpath.ensuredir()
        video_names.append(video['name'])

        norm_over_time = config['norm_over_time']
        if not norm_over_time:
            chan_to_normalizer = None
        else:
            coco_images = [coco_dset.coco_image(gid) for gid in gids]

            # quick and dirty:
            # Find the first image for each visualization channel
            # to use as the normalizer.
            # Probably better to use multiple images from the sequence
            # to do normalization
            if channels is not None:
                requested_channels = kwcoco.ChannelSpec.coerce(channels).fuse().as_set()
            else:
                requested_channels = set()
                for coco_img in coco_images:
                    code = coco_img.channels.fuse().as_set()
                    requested_channels.update(code)

            chan_to_ref_imgs = {}
            for code in requested_channels:
                chan_to_ref_imgs[code] = []

            _remain = requested_channels.copy()
            for coco_img in coco_images:
                imghas = coco_img.channels.fuse().as_set()
                common = imghas & _remain
                for c in common:
                    chan_to_ref_imgs[c].append(coco_img)

            from watch.utils import util_kwarray
            chan_to_normalizer = {}
            for chan, coco_imgs in chan_to_ref_imgs.items():
                s = max(1, len(coco_imgs) // 10)
                obs = []
                for coco_img in coco_imgs[::s]:
                    rawdata = coco_img.delay(channels=chan).finalize()
                    mask = rawdata != 0
                    obs.append(rawdata[mask].ravel())
                allobs = np.hstack(obs)
                normalizer = util_kwarray.find_robust_normalizers(allobs, params={
                    'high': 0.90,
                    'mid': 0.5,
                    'low': 0.01,
                    'mode': 'linear',
                    # 'mode': 'sigmoid',
                })
                chan_to_normalizer[chan] = normalizer
            print('chan_to_normalizer = {}'.format(ub.repr2(chan_to_normalizer, nl=1)))

        if config['zoom_to_tracks']:
            assert space == 'video'
            tid_to_info = video_track_info(coco_dset, vidid)
            for tid, track_info in tid_to_info.items():
                track_dpath = sub_dpath / '_tracks' / 'tid_{:04d}'.format(tid)
                track_dpath.ensuredir()
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

                    if config['extra_header']:
                        _header_extra = f'tid={tid}' + config['extra_header']
                    else:
                        _header_extra = f'tid={tid}'

                    pool.submit(_write_ann_visualizations2,
                                coco_dset, img, anns, track_dpath, space=space,
                                channels=channels, vid_crop_box=vid_crop_box,
                                _header_extra=_header_extra,
                                chan_to_normalizer=chan_to_normalizer,
                                fixed_normalization_scheme=config.get(
                                    'fixed_normalization_scheme'),
                                cmap=config['cmap'],
                                verbose=config['verbose'],
                                only_boxes=config['only_boxes'],
                                any3=config['any3'], dset_idstr=dset_idstr)

        else:
            gid_subset = gids[start_frame:end_frame]
            for gid in gid_subset:
                img = coco_dset.index.imgs[gid]
                anns = coco_dset.annots(gid=gid).objs

                if config['extra_header']:
                    _header_extra = config['extra_header']
                else:
                    _header_extra = ''

                pool.submit(_write_ann_visualizations2,
                            coco_dset, img, anns, sub_dpath, space=space,
                            channels=channels,
                            draw_imgs=config['draw_imgs'],
                            draw_anns=config['draw_anns'],
                            _header_extra=_header_extra,
                            chan_to_normalizer=chan_to_normalizer,
                            verbose=config['verbose'],
                            only_boxes=config['only_boxes'],
                            fixed_normalization_scheme=config.get(
                                'fixed_normalization_scheme'),
                            any3=config['any3'], dset_idstr=dset_idstr,
                            cmap=config['cmap'],
                            skip_missing=config['skip_missing'])

        for job in ub.ProgIter(pool.as_completed(), total=len(pool), desc='write imgs'):
            job.result()

        pool.jobs.clear()

    print('Wrote images to viz_dpath = {!r}'.format(viz_dpath))

    if config['animate']:
        from watch.cli import animate_visualizations
        animate_visualizations.animate_visualizations(
            viz_dpath=viz_dpath,
            channels=channels,
            video_names=video_names,
            draw_imgs=config['draw_imgs'],
            draw_anns=config['draw_anns'],
            workers=max_workers,
            zoom_to_tracks=config['zoom_to_tracks'],
            frames_per_second=0.7,
        )
        pass


def video_track_info(coco_dset, vidid):
    vid_annots = coco_dset.images(vidid=vidid).annots
    track_ids = set(ub.flatten(vid_annots.lookup('track_id')))
    tid_to_info = {}
    for tid in track_ids:
        track_aids = coco_dset.index.trackid_to_aids[tid]
        vidspace_boxes = []
        track_gids = []
        for aid in track_aids:
            ann = coco_dset.index.anns[aid]
            gid = ann['image_id']
            img = coco_dset.index.imgs[gid]
            bbox = ann['bbox']
            vid_from_img = kwimage.Affine.coerce(img.get('warp_img_to_vid', None))
            imgspace_box = kwimage.Boxes([bbox], 'xywh')
            vidspace_box = imgspace_box.warp(vid_from_img)
            vidspace_boxes.append(vidspace_box)
            track_gids.append(gid)
        all_vidspace_boxes = kwimage.Boxes.concatenate(vidspace_boxes)
        full_vid_box = all_vidspace_boxes.bounding_box().to_xywh()
        tid_to_info[tid] = {
            'tid': tid,
            'full_vid_box': full_vid_box,
            'track_gids': track_gids,
            'track_aids': track_aids,
        }
    return tid_to_info


_CLI = CocoVisualizeConfig


def select_fixed_normalization(fixed_normalization_scheme, sensor_coarse):
    chan_to_normalizer = {}
    if fixed_normalization_scheme == 'scaled':
        if sensor_coarse in {'L8', 'S2'}:
            for c in ['blue', 'green', 'red', 'nir', 'swir16', 'swir22']:
                chan_to_normalizer[c] = {'type': 'normalize', 'mode': 'linear',
                                         'min_val': 0, 'max_val': 10_000}

    elif fixed_normalization_scheme == 'scaled_50percentile':
        if sensor_coarse in {'L8', 'S2'}:
            for c in ['blue', 'green', 'red', 'nir', 'swir16', 'swir22']:
                chan_to_normalizer[c] = {'type': 'normalize', 'mode': 'linear',
                                         'min_val': 0, 'max_val': 5_000}

    elif fixed_normalization_scheme == 'scaled_25percentile':
        if sensor_coarse in {'L8', 'S2'}:
            for c in ['blue', 'green', 'red', 'nir', 'swir16', 'swir22']:
                chan_to_normalizer[c] = {'type': 'normalize', 'mode': 'linear',
                                         'min_val': 0, 'max_val': 2_500}

    elif fixed_normalization_scheme == 'scaled_raw':
        if sensor_coarse == 'L8':
            for c in ['blue', 'green', 'red', 'nir', 'swir16', 'swir22']:
                chan_to_normalizer[c] = {'type': 'normalize', 'mode': 'linear',
                                         'min_val': 7_272, 'max_val': 36_363}
        if sensor_coarse == 'S2':
            for c in ['blue', 'green', 'red', 'nir', 'swir16', 'swir22']:
                chan_to_normalizer[c] = {'type': 'normalize', 'mode': 'linear',
                                         'min_val': 1, 'max_val': 10_000}

    elif fixed_normalization_scheme == 'scaled_raw_50percentile':
        if sensor_coarse == 'L8':
            for c in ['blue', 'green', 'red', 'nir', 'swir16', 'swir22']:
                chan_to_normalizer[c] = {'type': 'normalize', 'mode': 'linear',
                                         'min_val': 7_272, 'max_val': 21_818}
        if sensor_coarse == 'S2':
            for c in ['blue', 'green', 'red', 'nir', 'swir16', 'swir22']:
                chan_to_normalizer[c] = {'type': 'normalize', 'mode': 'linear',
                                         'min_val': 1, 'max_val': 5_000}

    elif fixed_normalization_scheme == 'scaled_raw_25percentile':
        if sensor_coarse == 'L8':
            for c in ['blue', 'green', 'red', 'nir', 'swir16', 'swir22']:
                chan_to_normalizer[c] = {'type': 'normalize', 'mode': 'linear',
                                         'min_val': 7_272, 'max_val': 14_544}
        if sensor_coarse == 'S2':
            for c in ['blue', 'green', 'red', 'nir', 'swir16', 'swir22']:
                chan_to_normalizer[c] = {'type': 'normalize', 'mode': 'linear',
                                         'min_val': 1, 'max_val': 2_500}

    else:
        raise NotImplementedError('Unsupported fixed normalization scheme')

    return chan_to_normalizer


def _write_ann_visualizations2(coco_dset : kwcoco.CocoDataset,
                               img : dict,
                               anns : list,
                               sub_dpath : str,
                               space : str,
                               channels=None,
                               vid_crop_box=None,
                               request_grouped_bands='default',
                               draw_imgs=True,
                               draw_anns=True, _header_extra=None,
                               chan_to_normalizer=None,
                               fixed_normalization_scheme=None,
                               any3=True, dset_idstr='',
                               skip_missing=False,
                               only_boxes=1,
                               cmap='viridis', verbose=0):
    """
    Dumps an intensity normalized "space-aligned" kwcoco image visualization
    (with or without annotation overlays) for specific bands to disk.
    """
    # See if we can look at what we made
    from kwcoco import channel_spec
    from watch.utils import util_kwimage

    sensor_coarse = img.get('sensor_coarse', 'unknown')
    align_method = img.get('align_method', 'unknown')
    name = img.get('name', 'unknown')

    from watch import heuristics
    header_lines = heuristics.build_image_header_text(
        img=img,
        name=None,
        _header_extra=None,
        coco_dset=coco_dset,
    )

    delayed = coco_dset.delayed_load(img['id'], space=space)

    if fixed_normalization_scheme is not None:
        chan_to_normalizer = select_fixed_normalization(
            fixed_normalization_scheme, sensor_coarse)
        # Hacks for common "heatmap" channels
        chan_to_normalizer['depth'] = {'type': 'normalize', 'mode': 'linear',
                                       'min_val': 0, 'max_val': 255}

    if verbose > 0:
        print(f'fixed_normalization_scheme={fixed_normalization_scheme}')
        print(f'chan_to_normalizer={chan_to_normalizer}')

    if channels is not None:
        if isinstance(channels, list):
            channels = ','.join(channels)  # hack
        channels = channel_spec.ChannelSpec.coerce(channels)
        chan_groups = [
            {'chan': chan_obj} for chan_obj in channels.streams()
        ]
    else:
        coco_img = coco_dset.coco_image(img['id'])
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
        group : kwcoco.FusedChannelSpec
        for group in initial_groups:
            if group.numel() > 3:
                # For large group, just take the first 3 channels
                if group.numel() > 8:
                    group = group.normalize()[0:3]
                    chan_groups.append({
                        'chan': group,
                    })
                else:
                    # For smaller groups split them into singles
                    for part in group:
                        chan_groups.append({
                            'chan': kwcoco.FusedChannelSpec.coerce(part)
                        })
            else:
                chan_groups.append({
                    'chan': group,
                })

    for row in chan_groups:
        row['pname'] = row['chan'].path_sanitize()

    if any3:
        if any3 == 'only':
            # Kick everything else out
            chan_groups = []
        # Try to visualize any3 channels to get a nice viewable sequence
        avail_channels = channels.fuse()
        common_visualizers = list(map(kwcoco.FusedChannelSpec.coerce, [
            'red|green|blue',
            'r|g|b',
            'pan',
            'panchromatic',
        ]))
        found = None
        for cand in common_visualizers:
            flag = (cand & avail_channels).spec == cand.spec
            if flag:
                found = cand
                break

        # Just show false color from the first few channels
        if found is None:
            first3 = avail_channels.as_list()[0:3]
            found = kwcoco.FusedChannelSpec.coerce('|'.join(first3))
        chan_groups.append({
            'pname': 'any3',
            'chan': found,
        })

    img_view_dpath = sub_dpath / '_imgs'
    ann_view_dpath = sub_dpath / '_anns'

    with ub.Timer('build dets', verbose=verbose):
        # Ignore keypoints
        anns_ = [ub.dict_diff(ann, ['keypoints']) for ann in anns]
        dets = kwimage.Detections.from_coco_annots(anns_, dset=coco_dset)

    if space == 'video':
        vid_from_img = kwimage.Affine.coerce(img['warp_img_to_vid'])
        with ub.Timer('warp dets', verbose=verbose > 2):
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

    for chan_row in chan_groups:
        chan_pname = chan_row['pname']
        chan_group_obj = chan_row['chan']
        chan_list = chan_group_obj.parsed
        chan_group = chan_group_obj.spec

        # spec = str(chan.channels.spec)
        img_chan_dpath = img_view_dpath / chan_pname
        ann_chan_dpath = ann_view_dpath / chan_pname

        if draw_anns:
            ann_chan_dpath.ensuredir()

        if draw_imgs:
            img_chan_dpath.ensuredir()

        if len(chan_pname) > 10:
            # Hack to prevent long names for docker (limit is 242 chars)
            chan_pname2 = kwcoco.FusedChannelSpec.coerce(chan_group).path_sanitize(maxlen=10)
            # chan_pname2 = '{}_{}'.format(ub.hash_data(chan_pname, base='abc')[0:8], num_bands)
        else:
            chan_pname2 = chan_pname
        suffix = '_'.join([chan_pname2, sensor_coarse, align_method])
        view_img_fpath = ub.augpath(name, dpath=img_chan_dpath) + '_' + suffix + '.view_img.jpg'
        view_ann_fpath = ub.augpath(name, dpath=ann_chan_dpath) + '_' + suffix + '.view_ann.jpg'

        try:
            chan = delayed.take_channels(chan_group)
        except KeyError:
            # hack
            from kwcoco.util import util_delayed_poc
            chan = util_delayed_poc.DelayedChannelConcat([delayed]).take_channels(chan_group)

        '''
        import kwcoco
        dset = kwcoco.CocoDataset('/home/joncrall/data/dvc-repos/smart_watch_dvc/Aligned-Drop2-TA1-2022-02-24/data.kwcoco_c9ea8bb9.json')
        coco_img = dset.videos(names=['AE_C002']).images[0].coco_images[3]
        imdata = coco_img.delay('red').finalize(nodata='float')

        import kwplot
        import kwimage
        kwplot.autompl()
        imdata_norm = kwimage.normalize_intensity(imdata)
        kwplot.imshow(imdata_norm)
        '''

        # Note: Using 'nearest' here since we're just visualizing (and
        # otherwise nodata values can affect interpolated pixel
        # values)
        # canvas = chan.finalize(interpolation='nearest', nodata='float')
        with ub.Timer('load channels', verbose=verbose):
            raw_canvas = canvas = chan.finalize(interpolation='linear', nodata='float')

        if verbose > 1:
            import kwarray
            print('raw_canvas.shape = {!r}'.format(raw_canvas.shape))
            print('chan_list = {!r}'.format(chan_list))
            try:
                chan_stats = kwarray.stats_dict(raw_canvas, axis=2, nan=True)
                print('chan_stats = {}'.format(ub.repr2(chan_stats, nl=1)))
            except Exception:
                import warnings
                warnings.warn('Error printing chan stats, probably need kwarray >= 0.6.1')

        if skip_missing and np.all(np.isnan(raw_canvas)):
            continue

        # FLAG = np.any(np.isnan(canvas)) and not np.all(np.isnan(canvas))
        # if FLAG:
        #     print('input nans', np.nansum(raw_canvas))
        #     print('input nans', np.isnan(raw_canvas).sum())

        # canvas = chan.finalize(, nodata='float')
        # import kwarray
        # kwarray.atleast_nd(canvas, 3)

        if chan_to_normalizer is None:
            dmax = np.nanmax(raw_canvas)
            # dmin = canvas.min()
            needs_norm = dmax > 1.0
            # if canvas.max() <= 0 or canvas.min() >= 255:
            # Hack to only do noramlization on "non-standard" data ranges
            with ub.Timer('normalize1', verbose=verbose > 2):
                if needs_norm:
                    mask = ~np.isnan(raw_canvas)
                    # from watch.utils import util_norm
                    norm_canvas = kwimage.normalize_intensity(raw_canvas, mask=mask, params={
                        'high': 0.90,
                        'mid': 0.5,
                        'low': 0.01,
                        'mode': 'linear',
                    })
                    # if FLAG:
                    #     print('norm nans', np.isnan(norm_canvas).sum())
                    #     print('norm canvas', np.nansum(norm_canvas))
                    canvas = norm_canvas
                canvas = np.clip(canvas, 0, None)
        else:
            # from watch.utils import util_kwarray
            with ub.Timer('normalize2', verbose=verbose > 2):
                new_parts = []
                for cx, c in enumerate(chan_list):
                    normalizer = chan_to_normalizer.get(c, None)
                    data = canvas[..., cx]
                    mask = ~np.isnan(data)
                    if normalizer is None:
                        p = kwimage.normalize_intensity(data, params={
                            'high': 0.90,
                            'mid': 0.5,
                            'low': 0.01,
                            'mode': 'linear',
                        })
                    else:
                        p = kwarray.apply_normalizer(data, normalizer, mask=mask,
                                                     set_value_at_mask=0.)
                    new_parts.append(p)
                canvas = np.stack(new_parts, axis=2)

        # invalid_mask = np.isnan(canvas)
        canvas = fill_nans_with_checkers(canvas)

        if cmap is not None:
            if kwimage.num_channels(canvas) == 1:
                import matplotlib as mpl
                import matplotlib.cm  # NOQA
                cmap_ = mpl.cm.get_cmap(cmap)
                canvas = np.nan_to_num(canvas)
                if len(canvas.shape) == 3:
                    canvas = canvas[..., 0]
                    canvas = cmap_(canvas)[..., 0:3].astype(np.float32)

        with ub.Timer('false color', verbose=verbose > 2):
            canvas = util_kwimage.ensure_false_color(canvas)

        if len(canvas.shape) > 2 and canvas.shape[2] > 4:
            # hack for wv
            canvas = canvas[..., 0]

        chan_header_lines = header_lines.copy()
        chan_header_lines.append(chan_group)
        header_text = '\n'.join(chan_header_lines)

        valid_region = img.get('valid_region', None)
        if valid_region:
            with ub.Timer('valid region', verbose=verbose > 2):
                valid_poly: kwimage.MultiPolygon = kwimage.MultiPolygon.coerce(valid_region)
                if space == 'video':
                    vid_from_img = kwimage.Affine.coerce(img['warp_img_to_vid'])
                    valid_poly = valid_poly.warp(vid_from_img)

                if any([p.data['exterior'].data.size for p in valid_poly.data]):
                    canvas = valid_poly.draw_on(canvas, color='green', fill=False,
                                                border=True)

        # force_min_dim = 512
        # min_dim = force_min_dim  # TODO: parameter
        min_dim = max(min(canvas.shape[0:2]), 384)

        if draw_imgs:
            with ub.Timer('prep img_canvas', verbose=verbose):
                img_canvas = kwimage.ensure_uint255(canvas)
                img_canvas = kwimage.imresize(img_canvas, min_dim=min_dim)
                img_canvas = util_kwimage.draw_header_text(image=img_canvas,
                                                           text=header_text,
                                                           stack=True,
                                                           fit='shrink')
            with ub.Timer('write img_canvas', verbose=verbose):
                kwimage.imwrite(view_img_fpath, img_canvas)

        if draw_anns:
            canvas = kwimage.ensure_float01(canvas)
            ann_canvas, info = kwimage.imresize(canvas, min_dim=min_dim,
                                                return_info=True)
            ann_canvas = ann_canvas.clip(0, 1)
            dets = dets.scale(info['scale'])
            dets = dets.translate(info['offset'])
            # info['scale']
            ONLY_BOXES = only_boxes
            if ONLY_BOXES:
                with ub.Timer('dets.draw_on 1', verbose=verbose):
                    ann_canvas = dets.draw_on(ann_canvas, sseg=False,
                                              labels=False, color='classes')
                    # ann_canvas = dets.boxes.draw_on(ann_canvas, color='blue')
            else:
                # THERE IS A IN DRAW POLY WITH LARGE POLYS. THIS IS FINE FOR
                # REAL DATA BUT A TEST FAILS HARD. HACKING THIS OFF FOR NOW
                with ub.Timer('dets.draw_on 2', verbose=verbose):
                    try:
                        # kwimage 0.8.4 fixes this error
                        ann_canvas = dets.draw_on(ann_canvas, color='classes')
                    except Exception:
                        ann_canvas = dets.draw_on(ann_canvas)

            ann_canvas = kwimage.ensure_uint255(ann_canvas)
            ann_canvas = util_kwimage.draw_header_text(image=ann_canvas,
                                                       text=header_text,
                                                       stack=True,
                                                       fit='shrink')
            with ub.Timer('write ann_canvas', verbose=verbose):
                kwimage.imwrite(view_ann_fpath, ann_canvas)


def _hack_cached_hashid(self):
    """
    TODO: remove when kwcoco 0.22 is out
    """
    cache_miss = True
    import json
    enable_cache = (
        self._state['was_loaded'] and
        not self._state['was_modified']
    )
    if enable_cache:
        coco_fpath = ub.Path(self.fpath)
        enable_cache = coco_fpath.exists()

    if enable_cache:
        hashid_sidecar_fpath = ub.Path(str(coco_fpath) + '.hashid.cache')
        # Generate current lookup key
        fpath_stat = coco_fpath.stat()
        status_key = {
            'st_size': fpath_stat.st_size,
            'st_mtime': fpath_stat.st_mtime
        }
        if hashid_sidecar_fpath.exists():
            cached_data = json.loads(hashid_sidecar_fpath.read_text())
            if cached_data['status_key'] == status_key:
                self.hashid = cached_data['hashid']
                self.hashid_parts = cached_data['hashid_parts']
                cache_miss = False

    if cache_miss:
        self._build_hashid()
        if enable_cache:
            hashid_cache_data = {
                'hashid': self.hashid,
                'hashid_parts': self.hashid_parts,
                'status_key': status_key,
            }
            hashid_sidecar_fpath.write_text(json.dumps(hashid_cache_data))
    return self.hashid


def _hack_dataset_id(self):
    """
    A human interpretable name that can be used to uniquely identify the
    dataset.

    Note:
        This function is currently subject to change.
    """
    hashid = _hack_cached_hashid(self)
    coco_fpath = ub.Path(self.fpath)
    dset_id = '_'.join([coco_fpath.parent.stem, coco_fpath.stem, hashid[0:8]])
    return dset_id


def fill_nans_with_checkers(canvas):
    """
    TODO: move to kwimage

    Example:
        >>> import kwplot
        >>> import kwimage
        >>> orig_img = kwimage.ensure_float01(kwimage.grab_test_image())
        >>> poly1 = kwimage.Polygon.random().scale(orig_img.shape[0] // 2)
        >>> poly2 = kwimage.Polygon.random().scale(orig_img.shape[0])
        >>> img = orig_img.copy()
        >>> img = poly1.fill(img, np.nan)
        >>> img[:, :, 0] = poly2.fill(np.ascontiguousarray(img[:, :, 0]), np.nan)
        >>> canvas = img.copy()
        >>> canvas = fill_nans_with_checkers(canvas)
        >>> kwplot.autompl()
        >>> kwplot.imshow(img, pnum=(1, 2, 1))
        >>> kwplot.imshow(canvas, pnum=(1, 2, 2))
    """
    import kwarray
    canvas = kwarray.atleast_nd(canvas, 3)
    invalid_mask = np.isnan(canvas)
    allchan_invalid_mask = invalid_mask.all(axis=2, keepdims=1)
    anychan_invalid_mask = invalid_mask.any(axis=2, keepdims=1)

    some_invalid_mask = (~allchan_invalid_mask) * anychan_invalid_mask

    invalid_mask.all(axis=2)
    # canvas[invalid_mask] = 0
    dsize = canvas.shape[0:2][::-1]

    checkers2d = None

    if np.any(allchan_invalid_mask):
        if checkers2d is None:
            checkers2d = checkerboard(square_shape=8, dsize=dsize)
        # canvas = kwimage.ensure_alpha_channel(canvas, (1 - invalid_mask))
        # checkers = kwimage.ensure_alpha_channel(checkers, 1)
        locs = np.where(allchan_invalid_mask)
        canvas[locs[0:2]] = checkers2d[..., None][locs[0:2]]

    if np.any(some_invalid_mask):
        if checkers2d is None:
            checkers2d = checkerboard(square_shape=8, dsize=dsize)

        locs = np.where(some_invalid_mask)
        canvas[locs] = checkers2d[locs[0:2]]

    return canvas


def checkerboard(num_squares='auto', square_shape='auto', dsize=(512, 512)):
    """
    Remove when kwimage 0.8.3 lands
    """
    import numpy as np
    if num_squares == 'auto' and square_shape == 'auto':
        num_squares = 8

    want_w, want_h = dsize

    if square_shape != 'auto':
        if not ub.iterable(square_shape):
            square_shape = [square_shape, square_shape]
        h, w = square_shape
        gen_h, gen_w = _next_multiple_of(want_h, h * 2), _next_multiple_of(want_w, w * 2)
    else:
        gen_h, gen_w = _next_multiple_of(want_h, 4), _next_multiple_of(want_w, 4)

    if num_squares == 'auto':
        assert square_shape != 'auto'
        if not ub.iterable(square_shape):
            square_shape = [square_shape, square_shape]
        h, w = square_shape
        num_w = gen_w // w
        num_h = gen_h // h
        num_squares = num_h, num_w
    elif square_shape == 'auto':
        assert num_squares != 'auto'
        if not ub.iterable(num_squares):
            num_squares = [num_squares, num_squares]
        num_h, num_w = num_squares
        w = gen_w // num_w
        h = gen_h // num_h
        square_shape = (h, w)
    else:
        if not ub.iterable(num_squares):
            num_squares = [num_squares, num_squares]
        if not ub.iterable(square_shape):
            square_shape = [square_shape, square_shape]

    num_h, num_w = num_squares

    num_pairs_w = int(num_w // 2)
    num_pairs_h = int(num_h // 2)
    # img_size = 512
    base = np.array([[1, 0] * num_pairs_w, [0, 1] * num_pairs_w] * num_pairs_h)
    expansion = np.ones((h, w))
    img = np.kron(base, expansion)[0:want_h, 0:want_w]
    return img


def _next_multiple_of(x, m):
    """
    References:
        https://stackoverflow.com/questions/14267555/find-the-smallest-power-of-2-greater-than-or-equal-to-n-in-python
    """
    return (x // m) * m + m
    # + (x % 2)


if __name__ == '__main__':
    """

    DVC_DPATH=$(smartwatch_dvc)
    KWCOCO_BUNDLE_DPATH=$DVC_DPATH/Drop2-Aligned-TA1-2022-02-15
    python -m watch visualize $KWCOCO_BUNDLE_DPATH/data.kwcoco.json \
        --animate=True --channels="red|green|blue" --skip_missing=True \
        --select_images '.sensor_coarse == "S2"' --workers=4 --draw_anns=False

    DVC_DPATH=$(smartwatch_dvc)
    KWCOCO_BUNDLE_DPATH=$DVC_DPATH/Aligned-Drop3-TA1-2022-03-10/
    python -m watch visualize $KWCOCO_BUNDLE_DPATH/data.kwcoco.json \
        --animate=True --channels="red|green|blue" --skip_missing=True \
        --select_videos '.name == "BR_R002"' --workers=4 --draw_anns=True

    CommandLine:
        python ~/code/watch/watch/cli/coco_visualize_videos.py
    """
    main(cmdline=True)
