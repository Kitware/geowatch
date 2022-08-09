import kwarray
import kwimage
import numpy as np
import ubelt as ub
from watch.utils import util_kwimage
from watch import heuristics


def visualize_sample_grid(dset, sample_grid, max_vids=2, max_frames=6):
    """
    Debug visualization for sampling grid

    Draws multiple frames.

    Draws a yellow polygon over invalid spatial regions.

    Places a red dot where there is a negative sample (at the center of the negative window)

    Places a blue dot where there is a positive sample

    Notes:
        * Dots are more intense when there are more temporal coverage of that dot.

        * Dots are placed on the center of the window.
          They do not indicate its extent.

        * Dots are blue if they overlap any annotation in their temporal region
          so they may visually be near an annotation.

    Example:
        >>> from watch.tasks.fusion.datamodules.kwcoco_video_data import *  # NOQA
        >>> from watch.tasks.fusion.datamodules.spacetime_grid_builder import *  # NOQA
        >>> from watch.demo.smart_kwcoco_demodata import demo_kwcoco_multisensor
        >>> dset = coco_dset = demo_kwcoco_multisensor(dates=True, geodata=True, heatmap=True)
        >>> window_overlap = 0.0
        >>> window_dims = (3, 32, 32)
        >>> keepbound = False
        >>> time_sampling = 'soft2+distribute'
        >>> use_centered_positives = True
        >>> use_grid_positives = 0
        >>> sample_grid = sample_video_spacetime_targets(
        >>>     dset, window_dims, window_overlap, time_sampling=time_sampling,
        >>>     use_grid_positives=use_grid_positives, use_centered_positives=use_centered_positives)
        >>> # xdoctest: +REQUIRES(--show)
        >>> import kwplot
        >>> kwplot.autompl()
        >>> canvas = visualize_sample_grid(dset, sample_grid, max_vids=1,
        >>>                                max_frames=6)
        >>> kwplot.imshow(canvas, doclf=1)
        >>> kwplot.show_if_requested()

    Example:
        >>> # xdoctest: +REQUIRES(env:DVC_DPATH)
        >>> from watch.tasks.fusion.datamodules.kwcoco_video_data import *  # NOQA
        >>> import watch
        >>> # dset = coco_dset = demo_kwcoco_multisensor(dates=True, geodata=True, heatmap=True)
        >>> dvc_dpath = watch.find_smart_dvc_dpath()
        >>> #coco_fpath = dvc_dpath / 'Drop2-Aligned-TA1-2022-02-15/combo_DILM_train.kwcoco.json'
        >>> coco_fpath = dvc_dpath / 'Aligned-Drop3-TA1-2022-03-10/data_nowv_vali.kwcoco.json'
        >>> big_dset = kwcoco.CocoDataset(coco_fpath)
        >>> dset = big_dset.subset(big_dset.videos(names=['KR_R002']).images.lookup('id')[0])
        >>> window_overlap = 0.0
        >>> window_dims = (3, 32, 32)
        >>> keepbound = False
        >>> time_sampling = 'soft2+distribute'
        >>> use_centered_positives = True
        >>> use_grid_positives = 0
        >>> sample_grid = sample_video_spacetime_targets(
        >>>     dset, window_dims, window_overlap, time_sampling=time_sampling,
        >>>     use_grid_positives=True, use_centered_positives=use_centered_positives)
        >>> # xdoctest: +REQUIRES(--show)
        >>> import kwplot
        >>> kwplot.autompl()
        >>> canvas = visualize_sample_grid(dset, sample_grid, max_vids=1, max_frames=12)
        >>> kwplot.imshow(canvas, doclf=1)
        >>> kwplot.show_if_requested()
    """
    # Visualize the sample grid
    import pandas as pd
    targets = pd.DataFrame(sample_grid['targets'])

    dataset_canvases = []

    # max_vids = 2
    # max_frames = 6

    vidid_to_videodf = dict(list(targets.groupby('video_id')))

    orientation = 0

    for vidid, video_df in vidid_to_videodf.items():
        video = dset.index.videos[vidid]
        vidname = video['name']
        gid_to_infos = ub.ddict(list)
        for _, row in video_df.iterrows():
            for gid in row['gids']:
                gid_to_infos[gid].append({
                    'gid': gid,
                    'space_slice': row['space_slice'],
                    'label': row['label'],
                })

        video_canvases = []
        common = ub.oset(dset.images(vidid=vidid)) & (gid_to_infos)

        if True:
            # HACK: Use a temporal sampler once to get a nice overview of the
            # dataset in time.
            from dateutil import parser
            from watch.tasks.fusion.datamodules import temporal_sampling as tsm  # NOQA
            images = dset.images(common)
            datetimes = [None if date is None else parser.parse(date) for date in images.lookup('date_captured', None)]
            unixtimes = np.array([np.nan if dt is None else dt.timestamp() for dt in datetimes])
            sensors = images.lookup('sensor_coarse', None)
            time_sampler = tsm.TimeWindowSampler(
                unixtimes=unixtimes, sensors=sensors, time_window=max_frames,
                time_span='1y', affinity_type='hardish3',
                update_rule='distribute+pairwise')
            sample = time_sampler.sample()
            common = list(ub.take(common, sample))

        for gid in common:
            infos = gid_to_infos[gid]
            label_to_items = ub.group_items(infos, key=lambda x: x['label'])
            video = dset.index.videos[vidid]

            shape = (video['height'], video['width'], 4)
            canvas = np.zeros(shape, dtype=np.float32)
            shape = (2, video['height'], video['width'])
            accum = np.zeros(shape, dtype=np.float32)

            for label, items in label_to_items.items():
                label_idx = {'positive_grid': 1, 'positive_center': 1,
                             'negative_grid': 0}[label]
                for info in items:
                    space_slice = info['space_slice']
                    y_sl, x_sl = space_slice
                    # ww = x_sl.stop - x_sl.start
                    # wh = y_sl.stop - y_sl.start
                    ss = accum[(label_idx,) + space_slice].shape
                    if np.prod(ss) > 0:
                        vals = util_kwimage.upweight_center_mask(ss)
                        vals = np.maximum(vals, 0.1)
                        accum[(label_idx,) + space_slice] += vals
                        # Add extra weight to borders for viz
                        accum[label_idx, y_sl.start:y_sl.start + 1, x_sl.start: x_sl.stop] += 0.15
                        accum[label_idx, y_sl.stop - 1:y_sl.stop, x_sl.start:x_sl.stop] += 0.15
                        accum[label_idx, y_sl.start:y_sl.stop, x_sl.start: x_sl.start + 1] += 0.15
                        accum[label_idx, y_sl.start:y_sl.stop, x_sl.stop - 1: x_sl.stop] += 0.15

            neg_accum = accum[0]
            pos_accum = accum[1]

            neg_alpha = neg_accum / (neg_accum.max() * 2)
            pos_alpha = pos_accum / (pos_accum.max() * 2)
            bg_canvas = canvas.copy()
            bg_canvas[..., 0:4] = [0., 0., 0., 1.0]
            pos_canvas = canvas.copy()
            neg_canvas = canvas.copy()
            pos_canvas[..., 0:3] = kwimage.Color('dodgerblue').as01()
            neg_canvas[..., 0:3] = kwimage.Color('orangered').as01()
            neg_canvas[..., 3] = neg_alpha
            pos_canvas[..., 3] = pos_alpha
            neg_canvas = np.nan_to_num(neg_canvas)
            pos_canvas = np.nan_to_num(pos_canvas)

            warp_vid_from_img = kwimage.Affine.coerce(
                dset.index.imgs[gid]['warp_img_to_vid'])

            vid_poly = kwimage.Boxes([[0, 0, video['width'], video['height']]], 'xywh').to_polygons()[0]
            coco_poly = dset.index.imgs[gid].get('valid_region', None)
            if coco_poly is None:
                kw_invalid_poly = None
            else:
                kw_poly_img = kwimage.MultiPolygon.coerce(coco_poly)
                valid_poly = kw_poly_img.warp(warp_vid_from_img)
                sh_invalid_poly = vid_poly.to_shapely().difference(valid_poly.to_shapely())
                kw_invalid_poly = kwimage.MultiPolygon.coerce(sh_invalid_poly)

            final_canvas = kwimage.overlay_alpha_layers([pos_canvas, neg_canvas, bg_canvas])
            final_canvas = kwimage.ensure_uint255(final_canvas)

            annot_dets = dset.annots(gid=gid).detections
            vid_annot_dets = annot_dets.warp(warp_vid_from_img)

            if 1:
                final_canvas = vid_annot_dets.draw_on(final_canvas, color='white')

            if kw_invalid_poly is not None:
                final_canvas = kw_invalid_poly.draw_on(final_canvas, color='yellow', alpha=0.5)

            # from watch import heuristics
            img = dset.index.imgs[gid]
            header_lines = heuristics.build_image_header_text(
                img=img, vidname=vidname)
            header_text = '\n'.join(header_lines)

            final_canvas = kwimage.draw_header_text(final_canvas, header_text)
            video_canvases.append(final_canvas)

            if len(video_canvases) >= max_frames:
                break

        if max_vids == 1:
            video_canvas = kwimage.stack_images_grid(
                video_canvases, axis=orientation, pad=10)
        else:
            video_canvas = kwimage.stack_images(video_canvases, axis=1 - orientation, pad=10)
        dataset_canvases.append(video_canvas)
        if len(dataset_canvases) >= max_vids:
            break

    dataset_canvas = kwimage.stack_images(dataset_canvases, axis=orientation, pad=20)
    if 0:
        import kwplot
        kwplot.autompl()
        kwplot.imshow(dataset_canvas, doclf=1)
    return dataset_canvas


def sample_video_spacetime_targets(dset, window_dims, window_overlap=0.0,
                                   negative_classes=None, keepbound=False,
                                   exclude_sensors=None,
                                   time_sampling='hard+distribute',
                                   time_span='2y', use_annot_info=True,
                                   use_grid_positives=True,
                                   use_centered_positives=True,
                                   set_cover_algo=None):
    """
    This is the main driver that builds the sample grid.

    The basic idea is that you will slide a spacetime window over the dataset
    and mark where positive andnegative "windows" are. We also put windows
    directly on positive annotations if desired.

    See the above :func:`visualize_sample_grid` for a visualization of what the
    sample grid looks like.

    Ask jon about what the params mean if you need this.
    This code badly needs a refactor.

    Args:
        set_cover_algo (str | None):
            Algorithm used to find set cover of image IDs. Options are 'approx' (a greedy solution)
            or 'exact' (an ILP solution). If None is passed, set cover is not computed. The 'exact'
            method requires the packe pulp, available at PyPi.

    Example:
        >>> # xdoctest: +REQUIRES(env:DVC_DPATH)
        >>> import os
        >>> from watch.tasks.fusion.datamodules.kwcoco_video_data import *  # NOQA
        >>> import watch
        >>> dvc_dpath = watch.find_smart_dvc_dpath()
        >>> coco_fpath = dvc_dpath / 'drop1-S2-L8-aligned/data.kwcoco.json'
        >>> dset = kwcoco.CocoDataset(coco_fpath)
        >>> window_overlap = 0.0
        >>> window_dims = (2, 128, 128)
        >>> keepbound = False
        >>> exclude_sensors = None
        >>> sample_grid = sample_video_spacetime_targets(dset, window_dims, window_overlap)
        >>> time_sampling = 'hard+distribute'
        >>> positives = list(ub.take(sample_grid['targets'], sample_grid['positives_indexes']))
        _ = xdev.profile_now(sample_video_spacetime_targets)(dset, window_dims, window_overlap)

        >>> import os
        >>> from watch.tasks.fusion.datamodules.kwcoco_video_data import *  # NOQA
        >>> import watch
        >>> dvc_dpath = watch.find_smart_dvc_dpath()
        >>> coco_fpath = dvc_dpath / 'Drop1-Aligned-L1/vali_data_wv.kwcoco.json'
        >>> dset = kwcoco.CocoDataset(coco_fpath)
        >>> window_overlap = 0.0
        >>> window_dims = (2, 128, 128)
        >>> keepbound = False
        >>> exclude_sensors = None
        >>> sample_grid = sample_video_spacetime_targets(dset, window_dims, window_overlap)
        >>> time_sampling = 'hard+distribute'
        >>> time_span = '2y'
        >>> use_annot_info = True
        >>> time_sampling = 'hard+distribute'
        >>> positives = list(ub.take(sample_grid['targets'], sample_grid['positives_indexes']))

    Example:
        >>> # xdoctest: +REQUIRES(env:DVC_DPATH)
        >>> import os
        >>> from watch.tasks.fusion.datamodules.kwcoco_video_data import *  # NOQA
        >>> import watch
        >>> dvc_dpath = watch.find_smart_dvc_dpath()
        >>> coco_fpath = dvc_dpath / 'Aligned-Drop3-TA1-2022-03-10/combo_LM_nowv_vali.kwcoco.json'
        >>> dset = kwcoco.CocoDataset(coco_fpath)
        >>> window_overlap = 0.5
        >>> window_dims = (2, 128, 128)
        >>> keepbound = False
        >>> exclude_sensors = None
        >>> set_cover_algo = 'approx'
        >>> sample_grid = sample_video_spacetime_targets(dset, window_dims, window_overlap, set_cover_algo=set_cover_algo)
        >>> time_sampling = 'hard+distribute'
        >>> positives = list(ub.take(sample_grid['targets'], sample_grid['positives_indexes']))
        _ = xdev.profile_now(sample_video_spacetime_targets)(dset, window_dims, window_overlap)


    Example:
        >>> from watch.tasks.fusion.datamodules.kwcoco_video_data import *  # NOQA
        >>> import ndsampler
        >>> import kwcoco
        >>> dset = kwcoco.CocoDataset.demo('vidshapes2-multispectral', num_frames=30)
        >>> window_overlap = 0.0
        >>> window_dims = (3, 256, 256)
        >>> keepbound = False
        >>> time_sampling = 'soft2+distribute'
        >>> sample_grid1 = sample_video_spacetime_targets(
        >>>     dset, window_dims, window_overlap,
        >>>     time_sampling='soft2+distribute')
        >>> boxes = [kwimage.Boxes.from_slice(target['space_slice'], clip=False).to_xywh() for target in sample_grid1['targets']]
        >>> all_boxes = kwimage.Boxes.concatenate(boxes)
        >>> assert np.all(all_boxes.height == window_dims[1])
        >>> assert np.all(all_boxes.width == window_dims[2])

    Example:
        >>> from watch.tasks.fusion.datamodules.kwcoco_video_data import *  # NOQA
        >>> import ndsampler
        >>> import kwcoco
        >>> dset = kwcoco.CocoDataset.demo('vidshapes2-multispectral', num_frames=30)
        >>> window_overlap = 0.0
        >>> window_dims = (3, 32, 32)
        >>> keepbound = False
        >>> time_sampling = 'soft2+distribute'
        >>> sample_grid1 = sample_video_spacetime_targets(
        >>>     dset, window_dims, window_overlap,
        >>>     time_sampling='soft2+distribute')
        >>> sample_grid2 = sample_video_spacetime_targets(
        >>>     dset, window_dims, window_overlap,
        >>>     time_sampling='contiguous+pairwise')

        ub.peek(sample_grid1['vidid_to_time_sampler'].values()).show_summary(fnum=1)
        ub.peek(sample_grid2['vidid_to_time_sampler'].values()).show_summary(fnum=2)
        _ = xdev.profile_now(sample_video_spacetime_targets)(dset, window_dims, window_overlap)

        import xdev
        globals().update(xdev.get_func_kwargs(sample_video_spacetime_targets))

    Ignore:
        >>> from watch.tasks.fusion.datamodules.kwcoco_video_data import *  # NOQA
        >>> import watch
        >>> dvc_dpath = watch.find_smart_dvc_dpath()
        >>> coco_fpath = dvc_dpath / 'Drop1-Aligned-TA1-2022-01/data.kwcoco.json'
        >>> dset = kwcoco.CocoDataset(coco_fpath)
        >>> window_overlap = 0.0
        >>> window_dims = (3, 128, 128)
    """
    # Create a sliding window object for each specific image (because they may
    # have different sizes, technically we could memoize this)
    import pyqtree
    from watch.tasks.fusion.datamodules import temporal_sampling as tsm  # NOQA

    window_space_dims = window_dims[1:3]
    window_time_dims = window_dims[0]
    print('window_time_dims = {!r}'.format(window_time_dims))

    # It is important that keepbound is True at test time, otherwise
    # we may not predict on the bottom right of the image.
    keepbound = True

    vidid_to_space_slider = {}
    for vidid, video in dset.index.videos.items():
        full_dims = [video['height'], video['width']]
        window_dims_ = full_dims if window_space_dims == 'full' else window_space_dims
        slider = kwarray.SlidingWindow(full_dims, window_dims_,
                                       overlap=window_overlap,
                                       keepbound=keepbound,
                                       allow_overshoot=True)
        vidid_to_space_slider[vidid] = slider

    # from ndsampler import isect_indexer
    # _isect_index = isect_indexer.FrameIntersectionIndex.from_coco(dset)
    targets = []
    positive_idxs = []
    negative_idxs = []

    vidid_to_time_sampler = {}
    vidid_to_valid_gids = {}

    parts = set(time_sampling.split('+'))
    affinity_type_parts = parts & {
        'hard', 'hardish', 'contiguous', 'soft2', 'soft', 'hardish2',
        'hardish3'}
    update_rule_parts = parts & {'distribute', 'pairwise'}
    unknown = (parts - affinity_type_parts) - update_rule_parts
    if unknown:
        raise ValueError('Unknown time-sampling parts: {}'.format(unknown))

    affinity_type = '+'.join(list(affinity_type_parts))
    update_rule = '+'.join(list(update_rule_parts))
    if not update_rule:
        update_rule = 'distribute'

    dset_hashid = dset._cached_hashid()

    @ub.memoize
    def get_image_valid_region_in_vidspace(gid):
        coco_poly = dset.index.imgs[gid].get('valid_region', None)
        if not coco_poly:
            sh_poly_vid = None
        else:
            warp_vid_from_img = kwimage.Affine.coerce(
                dset.index.imgs[gid]['warp_img_to_vid'])
            kw_poly_img = kwimage.MultiPolygon.coerce(coco_poly)
            kw_poly_vid = kw_poly_img.warp(warp_vid_from_img)
            sh_poly_vid = kw_poly_vid.to_shapely()
        return sh_poly_vid

    if negative_classes is None:
        negative_classes = heuristics.BACKGROUND_CLASSES

    # Given an video
    all_vid_ids = list(dset.index.videos.keys())
    for video_id in ub.ProgIter(all_vid_ids, desc='sample video regions', verbose=3):
        slider = vidid_to_space_slider[video_id]

        video_info = dset.index.videos[video_id]
        all_video_gids = list(dset.index.vidid_to_gids[video_id])

        if exclude_sensors is not None:
            sensor_coarse = dset.images(all_video_gids).lookup('sensor_coarse', '')
            flags = [s not in exclude_sensors for s in sensor_coarse]
            video_gids = list(ub.compress(all_video_gids, flags))
        else:
            video_gids = all_video_gids
        # video_frame_idxs = np.array(list(range(len(video_gids))))

        # TODO: allow for multiple time samplers
        time_sampler = tsm.TimeWindowSampler.from_coco_video(
            dset, video_id, gids=video_gids, time_window=window_time_dims,
            affinity_type=affinity_type, update_rule=update_rule,
            name=video_info['name'], time_span=time_span)
        time_sampler.video_gids = np.array(video_gids)
        time_sampler.determenistic = True

        depends = [
            dset_hashid,
            negative_classes,
            affinity_type,
            update_rule,
            video_info['name'],
            window_dims, window_overlap,
            negative_classes, keepbound,
            exclude_sensors,
            time_sampling,
            time_span, use_annot_info,
            use_grid_positives,
            use_centered_positives,
            'cache_v2',
        ]
        cacher = ub.Cacher('sliding-window-cache', appname='watch',
                           depends=depends)
        _cached = cacher.tryload()
        if _cached is None:

            video_targets = []
            video_positive_idxs = []
            video_negative_idxs = []
            # For each frame, determenistically compute an initial list of which
            # supporting frames we will look at when making a prediction for the
            # "main" frame. Initially this is only based on temporal metadata.  We
            # may modify this later depending on spatial properties.
            main_idx_to_gids = {
                main_idx: list(ub.take(video_gids, time_sampler.sample(main_idx)))
                for main_idx in time_sampler.main_indexes
            }

            if use_annot_info:
                # Build a distribution of where annotations exist in this dataset
                qtree = pyqtree.Index((0, 0, video_info['width'], video_info['height']))
                qtree.aid_to_tlbr = {}
                # qtree.idx_to_tlbr = {}
                tid_to_infos = ub.ddict(list)
                video_aids = dset.images(video_gids).annots.lookup('id')
                annot_vid_tlbr = []
                aids_to_track = []
                for aids, gid in zip(video_aids, video_gids):
                    warp_vid_from_img = kwimage.Affine.coerce(
                        dset.index.imgs[gid]['warp_img_to_vid'])
                    img_info = dset.index.imgs[gid]
                    frame_index = img_info['frame_index']
                    tids = dset.annots(aids).lookup('track_id', None)
                    cids = dset.annots(aids).lookup('category_id', None)
                    cnames = dset.categories(cids).name

                    for tid, aid, cid, cname in zip(tids, aids, cids, cnames):
                        if cname not in negative_classes:
                            aids_to_track.append(aid)
                            imgspace_box = kwimage.Boxes([
                                dset.index.anns[aid]['bbox']], 'xywh')
                            vidspace_box = imgspace_box.warp(warp_vid_from_img)
                            vidspace_box = vidspace_box.clip(
                                0, 0, video_info['width'], video_info['height'])
                            if vidspace_box.area.ravel()[0] > 0:
                                tlbr_box = vidspace_box.to_tlbr().data[0]
                                annot_vid_tlbr.append(tlbr_box)
                                if tid is not None:
                                    tid_to_infos[tid].append({
                                        'gid': gid,
                                        'cid': cid,
                                        'frame_index': frame_index,
                                        'vidspace_box': tlbr_box,
                                        'cname': dset._resolve_to_cat(cid)['name'],
                                        'aid': aid,
                                    })
                                qtree.insert(aid, tlbr_box)
                                qtree.aid_to_tlbr[aid] = tlbr_box

            # TODO: we probably can disable respect valid regions here and then
            # just do it on the fly in the datalaoder.
            RESPECT_VALID_REGIONS = True
            for space_region in ub.ProgIter(list(slider), desc='Sliding window'):
                y_sl, x_sl = space_region

                kw_space_box = kwimage.Boxes.from_slice(space_region).to_tlbr()

                # Find all annotations that pass through this spatial region
                if use_annot_info:
                    query = kw_space_box.data[0]
                    isect_aids = list(qtree.intersect(query))
                    # isect_aids = set(isect_aids)
                    isect_gids = set(dset.annots(isect_aids).lookup('image_id'))

                if RESPECT_VALID_REGIONS:
                    # Reselect the keyframes if we overlap an invalid region
                    # (as denoted in the metadata, further filtering may happen later)
                    # todo: refactor to be cleaner
                    try:
                        main_idx_to_gids2, resampled = _refine_time_sample(
                            dset, main_idx_to_gids, kw_space_box, time_sampler,
                            get_image_valid_region_in_vidspace)
                    except tsm.TimeSampleError:
                        # Hack, just skip the region
                        # We might be able to sample less and still be ok
                        continue
                else:
                    main_idx_to_gids2 = main_idx_to_gids
                    resampled = False

                if set_cover_algo is not None:
                    debug = True
                    if debug:
                        print('before applying set cover, len of main_idx_to_gids2', len(main_idx_to_gids2))
                    main_idx_to_gids2 = kwarray.setcover(main_idx_to_gids2, algo=set_cover_algo)
                    if debug:
                        print('after applying set cover', len(main_idx_to_gids2))

                for main_idx, gids in main_idx_to_gids2.items():
                    main_gid = time_sampler.video_gids[main_idx]
                    label = 'unknown'

                    if use_annot_info:
                        if isect_aids:
                            has_annot = bool(isect_gids & set(gids))
                        else:
                            has_annot = False
                        if has_annot:
                            label = 'positive_grid'
                        else:
                            # Hack: exclude all annotated regions from negative sampling
                            label = 'negative_grid'

                    # Or do that on the fly?
                    if False:
                        for gid in gids:
                            coco_img = dset.coco_image(gid)
                            coco_img.channels
                            part = coco_img.delay(space='video')
                            cropped = part.crop(space_region)
                            arr = cropped.finalize(as_xarray=True)
                            if np.all(arr == 0):
                                print('BLACK REGION')

                    if label == 'positive_grid':
                        if not use_grid_positives:
                            continue
                        video_positive_idxs.append(len(video_targets))
                    elif label == 'negative_grid':
                        video_negative_idxs.append(len(video_targets))

                    video_targets.append({
                        'main_idx': main_idx,
                        'video_id': video_id,
                        'gids': gids,
                        'main_gid': main_gid,
                        'space_slice': space_region,
                        'resampled': resampled,
                        'label': label,
                    })

            INSERT_CENTERED_ANNOT_WINDOWS = use_centered_positives
            if INSERT_CENTERED_ANNOT_WINDOWS and use_annot_info:
                # FIXME: This code is too slow
                # in addition to the sliding window sample, add positive samples
                # centered around each annotation.
                window_width = window_space_dims[1]
                window_height = window_space_dims[0]
                for tid, infos in ub.ProgIter(list(tid_to_infos.items()), desc='Centered annots'):
                    # existing_gids = [info['gid'] for info in infos]
                    for info in infos:
                        main_gid = info['gid']
                        ann_box = kwimage.Boxes([info['vidspace_box']], 'tlbr').to_cxywh()
                        ann_box.data[:, 2] = window_width
                        ann_box.data[:, 3] = window_height
                        kw_space_region = ann_box.to_tlbr()
                        kw_space_region = kw_space_region.quantize()
                        kw_space_region = kw_space_region.resize(width=window_width, height=window_height)
                        space_region = kw_space_region.to_slices()[0]
                        #  FIXME, this code is ugly
                        # TODO: we could make frames where the phase transitions
                        # more likely here.
                        _hack_main_idx = np.where(time_sampler.video_gids == main_gid)[0][0]
                        sample_gids = list(ub.take(video_gids, time_sampler.sample(_hack_main_idx)))
                        _hack = {_hack_main_idx: sample_gids}
                        if 0:
                            # Too slow to handle here, will have to handle
                            # in getitem or be more efficient
                            # 86% of the time is spent here
                            _hack2, _ = _refine_time_sample(
                                dset, _hack, kw_space_box, time_sampler,
                                get_image_valid_region_in_vidspace)
                        else:
                            _hack2 = _hack
                        if _hack2:
                            gids = _hack2[_hack_main_idx]
                            label = 'positive_center'
                            video_positive_idxs.append(len(video_targets))
                            video_targets.append({
                                'main_idx': _hack_main_idx,
                                'video_id': video_id,
                                'gids': gids,
                                'main_gid': main_gid,
                                'space_slice': space_region,
                                'label': label,
                                'resampled': -1,
                            })

            _cached = {
                'video_targets': video_targets,
                'video_positive_idxs': video_positive_idxs,
                'video_negative_idxs': video_negative_idxs,
            }
            cacher.save(_cached)

        offset = len(targets)
        targets.extend(_cached['video_targets'])
        positive_idxs.extend([idx + offset for idx in _cached['video_positive_idxs']])
        negative_idxs.extend([idx + offset for idx in _cached['video_negative_idxs']])

        # Disable determenism
        time_sampler.determenistic = False
        vidid_to_time_sampler[video_id] = time_sampler
        vidid_to_valid_gids[video_id] = video_gids

    print('Found {} targets'.format(len(targets)))
    if use_annot_info:
        print('Found {} positives'.format(len(positive_idxs)))
        print('Found {} negatives'.format(len(negative_idxs)))

    sample_grid = {
        'positives_indexes': positive_idxs,
        'negatives_indexes': negative_idxs,
        'targets': targets,
        'vidid_to_valid_gids': vidid_to_valid_gids,
        'vidid_to_time_sampler': vidid_to_time_sampler,
    }
    return sample_grid


def _refine_time_sample(dset, main_idx_to_gids, kw_space_box, time_sampler, get_image_valid_region_in_vidspace):
    """
    Refine the time sample based on spatial information
    """
    from watch.tasks.fusion.datamodules import temporal_sampling as tsm  # NOQA
    video_gids = time_sampler.video_gids

    iooa_thresh = 0.2  # parametarize?

    gid_to_isbad = {}
    for gid in video_gids:
        valid_poly = get_image_valid_region_in_vidspace(gid)
        gid_to_isbad[gid] = False
        if valid_poly is not None:
            sh_space_poly = kw_space_box.to_shapley()[0]
            # flag = valid_poly.intersects(sh_space_poly)
            isect = valid_poly.intersection(sh_space_poly)
            iooa = isect.area / sh_space_poly.area
            if iooa < iooa_thresh:
                gid_to_isbad[gid] = True

    all_bad_gids = [gid for gid, flag in gid_to_isbad.items() if flag]

    try:
        resampled = 0
        refined_sample = {}
        for main_idx, gids in main_idx_to_gids.items():
            main_gid = video_gids[main_idx]
            # Skip the sample when the "main" frame is bad.
            if not gid_to_isbad[main_gid]:
                good_gids = [gid for gid in gids if not gid_to_isbad[gid]]
                if good_gids != gids:
                    include_idxs = np.where(kwarray.isect_flags(video_gids, good_gids))[0]
                    exclude_idxs = np.where(kwarray.isect_flags(video_gids, all_bad_gids))[0]
                    chosen = time_sampler.sample(include=include_idxs, exclude=exclude_idxs, error_level=1, return_info=False)
                    new_gids = list(ub.take(video_gids, chosen))
                    # Are we allowed to return less than the initial expected
                    # number of frames? For transformers yes, but we should be
                    # careful to ask the user if they expect this.
                    new_are_bad = [g for g in new_gids if gid_to_isbad[g]]
                    if not new_are_bad:
                        resampled += 1
                        refined_sample[main_idx] = new_gids
                else:
                    refined_sample[main_idx] = gids
    except tsm.TimeSampleError:
        raise

    return refined_sample, resampled


def lookup_track_info(coco_dset, tid):
    """
    Find the spatio-temporal extent of a track
    """
    track_aids = coco_dset.index.trackid_to_aids[tid]
    vidspace_boxes = []
    track_gids = []
    for aid in track_aids:
        ann = coco_dset.index.anns[aid]
        gid = ann['image_id']
        track_gids.append(gid)
        img = coco_dset.index.imgs[gid]
        bbox = ann['bbox']
        vid_from_img = kwimage.Affine.coerce(img.get('warp_img_to_vid', None))
        imgspace_box = kwimage.Boxes([bbox], 'xywh')
        vidspace_box = imgspace_box.warp(vid_from_img)
        vidspace_boxes.append(vidspace_box)
    all_vidspace_boxes = kwimage.Boxes.concatenate(vidspace_boxes)
    full_vid_box = all_vidspace_boxes.bounding_box().to_xywh()

    frame_index = coco_dset.images(track_gids).lookup('frame_index')
    track_gids = list(ub.take(track_gids, ub.argsort(frame_index)))

    track_info = {
        'tid': tid,
        'full_vid_box': full_vid_box,
        'track_gids': track_gids,
    }
    return track_info


def make_track_based_spatial_samples(coco_dset):
    """
    Ignore:
        >>> # xdoctest: +REQUIRES(env:DVC_DPATH)
        >>> import os
        >>> from watch.tasks.fusion.datamodules.kwcoco_video_data import *  # NOQA
        >>> import watch
        >>> dvc_dpath = watch.find_smart_dvc_dpath()
        >>> coco_fpath = dvc_dpath / 'Drop1-Aligned-L1/data.kwcoco.json'
        >>> coco_dset = kwcoco.CocoDataset(coco_fpath)
    """
    tid_list = list(coco_dset.index.trackid_to_aids.keys())
    tid_to_trackinfo = {}
    for tid in tid_list:
        track_info = lookup_track_info(coco_dset, tid)
        gid = track_info['track_gids'][0]
        vidid = coco_dset.index.imgs[gid]['video_id']
        track_info['vidid'] = vidid
        tid_to_trackinfo[tid] = track_info

    vidid_to_tracks = ub.group_items(tid_to_trackinfo.values(), key=lambda x: x['vidid'])

    window_space_dims = [96, 96]

    for vidid, trackinfos in vidid_to_tracks.items():
        positive_boxes = []
        for track_info in trackinfos:
            boxes = track_info['full_vid_box']
            positive_boxes.append(boxes.to_cxywh())
        positives = kwimage.Boxes.concatenate(positive_boxes)
        positives_samples = positives.to_cxywh()
        positives_samples.data[:, 2] = window_space_dims[0]
        positives_samples.data[:, 3] = window_space_dims[1]
        print('positive_boxes = {}'.format(ub.repr2(positive_boxes, nl=1)))

        video = coco_dset.index.videos[vidid]
        full_dims = [video['height'], video['width']]
        window_overlap = 0.0
        keepbound = 0

        window_dims_ = full_dims if window_space_dims == 'full' else window_space_dims
        slider = kwarray.SlidingWindow(full_dims, window_dims_,
                                       overlap=window_overlap,
                                       keepbound=keepbound,
                                       allow_overshoot=True)

        sliding_boxes = kwimage.Boxes.concatenate(list(map(kwimage.Boxes.from_slice, slider)))
        ious = sliding_boxes.ious(positives)
        overlaps = ious.sum(axis=1)
        negative_boxes = sliding_boxes.compress(overlaps == 0)

        if 1:
            import kwplot
            kwplot.autompl()
            fig = kwplot.figure(fnum=vidid)
            ax = fig.gca()
            ax.set_title(video['name'])
            negative_boxes.draw(setlim=1, color='red', fill=True)
            positives.draw(color='limegreen')
            positives_samples.draw(color='green')
