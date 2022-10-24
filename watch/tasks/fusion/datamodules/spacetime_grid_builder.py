import kwarray
import kwimage
import numpy as np
import ubelt as ub
from watch.utils import util_kwimage
from watch import heuristics


def sample_video_spacetime_targets(dset, window_dims, window_overlap=0.0,
                                   negative_classes=None, keepbound=False,
                                   exclude_sensors=None,
                                   time_sampling='hard+distribute',
                                   time_span='2y',
                                   use_annot_info=True,
                                   use_grid_positives=True,
                                   use_centered_positives=True,
                                   window_space_scale=None,
                                   set_cover_algo=None,
                                   workers=0,
                                   use_cache=1):
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
        dset (kwcoco.CocoDataset): coco dataset

        window_dims (Tuple[int, int, int]):
            time, height, width of the sample region.

        window_overlap (float):
            fractional spatial overlap

        set_cover_algo (str | None):
            Algorithm used to find set cover of image IDs. Options are 'approx' (a greedy solution)
            or 'exact' (an ILP solution). If None is passed, set cover is not computed. The 'exact'
            method requires the packe pulp, available at PyPi.

        window_space_scale (str):
            Code indicating the scale at which to sample.

        use_grid_positives (bool):
            if False, will remove any grid sample that contains a positive
            example. In this case use_centered_positives should be True.

        use_centered_positives (bool):
            extend the grid with extra off-axis samples where positive
            annotations are centered. TODO: we could do a box packing
            to reduce the potential size here.

        use_annot_info (bool):
            if True allows using annotation information to get a better
            train-time grid. Should not be used at test-time.

        time_span (str):
            indicates the desired start/stop date range of the sample

        time_sampling (str):
            code for specific temporal sampler: see temporal_sampling.py for
            more information.

        exclude_sensors (List[str]):
            A list of sensors to exclude from the grid

        negative_classes (List[str]):
            indicate class names that should not count towards a region being
            marked as positive.

        workers (int): parallel workers

        use_cache (bool): uses a disk cache if True

    Example:
        >>> # xdoctest: +REQUIRES(env:DVC_DPATH)
        >>> import os
        >>> from watch.tasks.fusion.datamodules.spacetime_grid_builder import *  # NOQA
        >>> import watch
        >>> dvc_dpath = watch.find_dvc_dpath(tags='phase2_data', hardware='ssd')
        >>> coco_fpath = dvc_dpath / 'Aligned-Drop4-2022-08-08-TA1-S2-L8-ACC/data_train.kwcoco.json'
        >>> dset = kwcoco.CocoDataset(coco_fpath)
        >>> window_overlap = 0.0
        >>> window_dims = (11, 128, 128)
        >>> keepbound = False
        >>> exclude_sensors = None
        >>> sample_grid = sample_video_spacetime_targets(dset, window_dims)
        >>> time_sampling = 'hard+distribute'
        >>> positives = list(ub.take(sample_grid['targets'], sample_grid['positives_indexes']))

        _ = xdev.profile_now(sample_video_spacetime_targets)(dset, window_dims)

        >>> import os
        >>> from watch.tasks.fusion.datamodules.spacetime_grid_builder import *  # NOQA
        >>> import watch
        >>> dvc_dpath = watch.find_dvc_dpath(tags='phase2_data', hardware='ssd')
        >>> coco_fpath = dvc_dpath / 'Aligned-Drop4-2022-08-08-TA1-S2-L8-ACC/data_vali.kwcoco.json'
        >>> dset = kwcoco.CocoDataset(coco_fpath)
        >>> window_overlap = 0.0
        >>> window_dims = (2, 128, 128)
        >>> keepbound = False
        >>> exclude_sensors = None
        >>> set_cover_algo = 'approx'
        >>> window_space_scale = '13GSD'
        >>> negative_classes = None
        >>> time_span = '2y'
        >>> use_annot_info = True
        >>> use_grid_positives = 1
        >>> use_centered_positives = 1
        >>> time_sampling = 'hard+distribute'
        >>> with ub.Timer('sample'):
        >>>     sample_grid = sample_video_spacetime_targets(
        >>>         dset, window_dims, window_overlap, set_cover_algo=set_cover_algo,
        >>>         window_space_scale=window_space_scale, time_sampling=time_sampling,
        >>>         use_annot_info=use_annot_info, time_span=time_span, use_cache=1, workers=0)
        >>> positives = list(ub.take(sample_grid['targets'], sample_grid['positives_indexes']))
        >>> list(ub.unique([t['space_slice'] for t in sample_grid['targets']], key=ub.hash_data))

    Example:
        >>> # xdoctest: +REQUIRES(env:DVC_DPATH)
        >>> import os
        >>> from watch.tasks.fusion.datamodules.spacetime_grid_builder import *  # NOQA
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
        >>> from watch.tasks.fusion.datamodules.spacetime_grid_builder import *  # NOQA
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
        >>> from watch.tasks.fusion.datamodules.spacetime_grid_builder import *  # NOQA
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
        >>> from watch.tasks.fusion.datamodules.spacetime_grid_builder import *  # NOQA
        >>> import watch
        >>> dvc_dpath = watch.find_smart_dvc_dpath()
        >>> coco_fpath = dvc_dpath / 'Drop1-Aligned-TA1-2022-01/data.kwcoco.json'
        >>> dset = kwcoco.CocoDataset(coco_fpath)
        >>> window_overlap = 0.0
        >>> window_dims = (3, 128, 128)
    """
    winspace_space_dims = window_dims[1:3]
    winspace_time_dims = window_dims[0]

    # from ndsampler import isect_indexer
    # _isect_index = isect_indexer.FrameIntersectionIndex.from_coco(dset)
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

    if negative_classes is None:
        negative_classes = heuristics.BACKGROUND_CLASSES

    dset_hashid = dset._cached_hashid()

    # Given an video
    all_vid_ids = list(dset.index.videos.keys())

    depends = [
        dset_hashid,
        negative_classes,
        affinity_type,
        update_rule,
        window_dims,
        window_overlap,
        window_space_scale,
        negative_classes, keepbound,
        exclude_sensors,
        affinity_type, update_rule,
        time_span, use_annot_info,
        set_cover_algo,
        use_grid_positives,
        use_centered_positives,
        'cache_v7',
    ]
    # Higher level cacher (not sure if adding this secondary level of caching
    # is faster or not).
    cache_dpath = ub.Path.appdir('watch', 'grid_cache').ensuredir()
    cacher = ub.Cacher('sample_grid-dataset-cache', dpath=cache_dpath,
                       depends=depends, enabled=use_cache)
    sample_grid = cacher.tryload()
    if sample_grid is None:
        from watch.utils.lightning_ext import util_globals
        workers = util_globals.coerce_num_workers(workers)
        workers = min(len(all_vid_ids), workers)
        if workers == 1:
            workers = 0
        mode = 'process'
        jobs = ub.JobPool(mode=mode, max_workers=workers)

        # TODO: Reducing the information that needs to be passed to each worker
        # would help improve speed here. The dset itself is the biggest offender.
        verbose = 1 if workers == 0 else 0
        for video_id in ub.ProgIter(all_vid_ids, desc='Submit sample video regions'):
            job = jobs.submit(
                _sample_single_video_spacetime_targets, dset, dset_hashid,
                video_id, winspace_time_dims, winspace_space_dims, window_dims,
                window_overlap, negative_classes, keepbound, exclude_sensors,
                affinity_type, update_rule, time_span, use_annot_info,
                use_grid_positives, use_centered_positives, window_space_scale,
                set_cover_algo, use_cache, verbose)
            job.video_id = video_id

        targets = []
        positive_idxs = []
        negative_idxs = []
        vidid_to_time_sampler = {}
        vidid_to_valid_gids = {}
        vidid_to_meta = ub.ddict(dict)
        for job in jobs.as_completed(desc='Collect region sample grids',
                                     progkw=dict(verbose=3)):
            video_id = job.video_id
            _cached, meta, time_sampler, video_gids = job.result()
            offset = len(targets)
            targets.extend(_cached['video_targets'])
            positive_idxs.extend([idx + offset for idx in _cached['video_positive_idxs']])
            negative_idxs.extend([idx + offset for idx in _cached['video_negative_idxs']])
            vidid_to_time_sampler[video_id] = time_sampler
            vidid_to_valid_gids[video_id] = video_gids
            vidid_to_meta[video_id] = meta

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
            'vidid_to_meta': vidid_to_meta,
        }
        cacher.save(sample_grid)
    vidid_to_meta = sample_grid['vidid_to_meta']
    print('vidid_to_meta = {}'.format(ub.repr2(vidid_to_meta, nl=-1)))
    return sample_grid


def _sample_single_video_spacetime_targets(
        dset, dset_hashid, video_id, winspace_time_dims, winspace_space_dims,
        window_dims, window_overlap, negative_classes,
        keepbound, exclude_sensors, affinity_type, update_rule, time_span,
        use_annot_info, use_grid_positives, use_centered_positives,
        window_space_scale, set_cover_algo, use_cache, verbose):
    """
    Do this for a single video so we can parallelize.

    Called as the main worker function in
    :func:`sample_video_spacetime_targets`.

    Note this introduces a new temporary space:
        window-space, which is a scaled version of video-space.
        This is only used internally. The final targets are returned in video
        space.
    """
    from watch.tasks.fusion.datamodules import temporal_sampling as tsm  # NOQA
    from watch.tasks.fusion.datamodules import data_utils

    @ub.memoize
    def get_warp_vidspace_from_imgspace(gid):
        """
        Abstract the transform to bring us into whatever the internal "window
        space" is. Depends on whatever "window_scale" is computed as.
        """
        coco_img = dset.coco_image(gid)
        warp_vid_from_img = coco_img.warp_vid_from_img
        return warp_vid_from_img

    @ub.memoize
    def get_image_valid_region_in_vidspace(gid):
        coco_poly = dset.index.imgs[gid].get('valid_region', None)
        if not coco_poly:
            sh_poly_vid = None
        else:
            warp_vid_from_img = get_warp_vidspace_from_imgspace(gid)
            kw_poly_img = kwimage.MultiPolygon.coerce(coco_poly)
            kw_poly_vid = kw_poly_img.warp(warp_vid_from_img)
            sh_poly_vid = kw_poly_vid.to_shapely()
        return sh_poly_vid

    refine_iooa_thresh = 0.2  # parametarize?

    # TODO: we can disable respect valid regions here and then just do it on
    # the fly in the dataloader, but it is unclear which is more efficient.
    respect_valid_regions = True

    # It is important that keepbound is True at test time, otherwise we may not
    # predict on the bottom right of the image.
    keepbound = True
    video_info = dset.index.videos[video_id]
    video_name = video_info['name']

    # Create a box to represent the "window-space" extent, and determine how we
    # are going to slide a window over it.
    vidspace_gsd = video_info.get('target_gsd', None)

    resolved_scale = data_utils.resolve_scale_request(
        request=window_space_scale, data_gsd=vidspace_gsd)
    window_scale = resolved_scale['scale']

    all_video_gids = list(dset.index.vidid_to_gids[video_id])

    if exclude_sensors is not None:
        sensor_coarse = dset.images(all_video_gids).lookup('sensor_coarse', '')
        flags = [s not in exclude_sensors for s in sensor_coarse]
        video_gids = list(ub.compress(all_video_gids, flags))
    else:
        video_gids = all_video_gids

    vidspace_time_dims = winspace_time_dims

    # TODO: allow for multiple time samplers
    time_sampler = tsm.TimeWindowSampler.from_coco_video(
        dset, video_id, gids=video_gids, time_window=vidspace_time_dims,
        affinity_type=affinity_type, update_rule=update_rule,
        name=video_name, time_span=time_span)
    time_sampler.video_gids = np.array(video_gids)
    time_sampler.determenistic = True

    # Convert winspace to vidspace and use that for the rest of the function
    vidspace_video_height = video_info['height']
    vidspace_video_width = video_info['width']
    vidspace_full_dims = [vidspace_video_height, vidspace_video_width]
    if winspace_space_dims == 'full':
        vidspace_window_dims = vidspace_full_dims
    else:
        winspace_window_height, winspace_window_width = winspace_space_dims
        winspace_window_box = kwimage.Boxes([
            [0, 0, winspace_window_width, winspace_window_height]], 'xywh')
        # The window is scaled inversely to the data
        vidspace_window_box = winspace_window_box.scale(1 / window_scale).quantize()
        vidspace_window_height = vidspace_window_box.height.ravel()[0]
        vidspace_window_width = vidspace_window_box.width.ravel()[0]
        vidspace_window_dims = (vidspace_window_height, vidspace_window_width)

    depends = [
        dset_hashid,
        negative_classes,
        affinity_type,
        update_rule,
        video_name,
        vidspace_window_dims, window_overlap,
        negative_classes, keepbound,
        exclude_sensors,
        affinity_type, update_rule,
        time_span, use_annot_info,
        use_grid_positives,
        use_centered_positives,
        'cache_v5',
    ]
    cache_dpath = ub.Path.appdir('watch', 'grid_cache').ensuredir()
    cacher = ub.Cacher('sliding-window-cache', dpath=cache_dpath,
                       depends=depends, enabled=use_cache)
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
            qtree, tid_to_infos = _build_vidspace_track_qtree(
                dset, video_gids, negative_classes, vidspace_video_width,
                vidspace_video_height, get_warp_vidspace_from_imgspace)
        else:
            qtree = None
            tid_to_infos = None

        # Do a spatial sliding window (in video space) and handle all the
        # temporal stuff for that window in the internal function.
        slider = kwarray.SlidingWindow(vidspace_full_dims,
                                       vidspace_window_dims,
                                       overlap=window_overlap,
                                       keepbound=keepbound,
                                       allow_overshoot=True)

        for vidspace_region in ub.ProgIter(list(slider), desc='Sliding window',
                                           verbose=verbose):

            new_targets = _build_targets_in_spatial_region(
                dset, video_id, vidspace_region, use_annot_info, qtree,
                main_idx_to_gids, refine_iooa_thresh, time_sampler,
                get_image_valid_region_in_vidspace, respect_valid_regions,
                set_cover_algo)

            for target in new_targets:
                label = target['label']
                if label == 'positive_grid':
                    if not use_grid_positives:
                        continue
                    video_positive_idxs.append(len(video_targets))
                elif label == 'negative_grid':
                    video_negative_idxs.append(len(video_targets))
                video_targets.append(target)

        if use_centered_positives and use_annot_info:
            # FIXME: This code is too slow
            # in addition to the sliding window sample, add positive samples
            # centered around each annotation.
            for tid, infos in ub.ProgIter(list(tid_to_infos.items()),
                                          desc='Centered annots',
                                          verbose=verbose):

                new_targets = _build_targets_around_track(
                    video_id, tid, infos, video_gids, vidspace_window_dims,
                    time_sampler)
                for target in new_targets:
                    video_positive_idxs.append(len(video_targets))
                    video_targets.append(target)

        _cached = {
            'video_targets': video_targets,
            'video_positive_idxs': video_positive_idxs,
            'video_negative_idxs': video_negative_idxs,
        }
        cacher.save(_cached)

    # Disable determenism in the returned sampler
    time_sampler.determenistic = False
    meta = {
        'resolved_scale': resolved_scale,
        'vidspace_window_space_dims': vidspace_window_dims,
        'winspace_window_space_dims': winspace_space_dims,
        'vidspace_full_dims': vidspace_full_dims,
    }
    return _cached, meta, time_sampler, video_gids


def _build_targets_around_track(video_id, tid, infos, video_gids,
                                vidspace_window_dims, time_sampler):
    """
    Given information about a track, build targets to ensure the network trains
    with it.
    """
    window_height, window_width = vidspace_window_dims
    for info in infos:
        main_gid = info['gid']
        vidspace_ann_box = kwimage.Boxes([info['vidspace_box']], 'tlbr')
        vidspace_ann_box = vidspace_ann_box.quantize()
        vidspace_ann_box = vidspace_ann_box.resize(width=window_width, height=window_height)
        #  FIXME, this code is ugly
        # TODO: we could make frames where the phase transitions
        # more likely here.
        _hack_main_idx = np.where(time_sampler.video_gids == main_gid)[0][0]
        sample_gids = list(ub.take(video_gids, time_sampler.sample(_hack_main_idx)))
        _hack = {_hack_main_idx: sample_gids}
        # if 0:
        #     # Too slow to handle here, will have to handle
        #     # in getitem or be more efficient
        #     # 86% of the time is spent here
        #     _hack2, _ = _refine_time_sample(
        #         dset, _hack, winspace_box,
        #         refine_iooa_thresh, time_sampler,
        #         get_image_valid_region_in_vidspace)
        # else:
        _hack2 = _hack
        if _hack2:
            gids = _hack2[_hack_main_idx]
            label = 'positive_center'
            vidspace_region = vidspace_ann_box.to_slices()[0]

            target = {
                'main_idx': _hack_main_idx,
                'video_id': video_id,
                'gids': gids,
                'main_gid': main_gid,
                'space_slice': vidspace_region,
                'label': label,
                'resampled': -1,
            }
            yield target


def _build_targets_in_spatial_region(dset, video_id, vidspace_region,
                                     use_annot_info, qtree, main_idx_to_gids,
                                     refine_iooa_thresh, time_sampler,
                                     get_image_valid_region_in_vidspace,
                                     respect_valid_regions, set_cover_algo):
    """
    Called for each spatial grid in the sliding window. This adds multiple
    targets

    Called as part of :func:`_sample_single_video_spacetime_targets`.
    """
    from watch.tasks.fusion.datamodules import temporal_sampling as tsm  # NOQA
    y_sl, x_sl = vidspace_region

    vidspace_box = kwimage.Boxes.from_slice(vidspace_region).to_ltrb()

    # Find all annotations that pass through this spatial region
    if use_annot_info:
        query = vidspace_box.data[0]
        isect_aids = list(qtree.intersect(query))
        isect_gids = set(dset.annots(isect_aids).lookup('image_id'))

    if respect_valid_regions:
        # Reselect the keyframes if we overlap an invalid region (as
        # denoted in the metadata, further filtering may happen later)
        # todo: refactor to be cleaner
        try:
            main_idx_to_gids2, resampled = _refine_time_sample(
                dset, main_idx_to_gids, vidspace_box,
                refine_iooa_thresh, time_sampler,
                get_image_valid_region_in_vidspace)
        except tsm.TimeSampleError:
            # Hack, just skip the region
            # We might be able to sample less and still be ok
            raise
            # continue
    else:
        main_idx_to_gids2 = main_idx_to_gids
        resampled = False

    if set_cover_algo is not None:
        # TODO: allow there to be a slightly denser overlapped sampling
        # debug = True
        debug = 0
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

        target = {
            'main_idx': main_idx,
            'video_id': video_id,
            'gids': gids,
            'main_gid': main_gid,
            'space_slice': vidspace_region,
            'resampled': resampled,
            'label': label,
        }
        yield target


def _build_vidspace_track_qtree(dset, video_gids, negative_classes,
                                vidspace_video_width, vidspace_video_height,
                                get_warp_vidspace_from_imgspace):
    """
    Build a data structure that allows for fast lookup of which annotations
    exist in the in the requested "Window Space".

    Called as part of :func:`_sample_single_video_spacetime_targets`.
    """
    import pyqtree
    qtree = pyqtree.Index((0, 0, vidspace_video_width, vidspace_video_height))
    qtree.aid_to_tlbr = {}
    tid_to_infos = ub.ddict(list)
    video_images = dset.images(video_gids)
    video_coco_images = video_images.coco_images
    video_aids = video_images.annots.lookup('id')
    for aids, coco_img in zip(video_aids, video_coco_images):
        img_info = coco_img.img
        gid = img_info['id']
        frame_index = img_info['frame_index']
        tids = dset.annots(aids).lookup('track_id', None)
        cids = dset.annots(aids).lookup('category_id', None)
        cnames = dset.categories(cids).name

        warp_vid_from_img = get_warp_vidspace_from_imgspace(gid)

        for tid, aid, cid, cname in zip(tids, aids, cids, cnames):
            if cname not in negative_classes:
                imgspace_box = kwimage.Boxes([
                    dset.index.anns[aid]['bbox']], 'xywh')
                vidspace_box = imgspace_box.warp(warp_vid_from_img)
                vidspace_box = vidspace_box.clip(
                    0, 0, vidspace_video_width, vidspace_video_height)
                if vidspace_box.area.ravel()[0] > 0:
                    tlbr_box = vidspace_box.to_ltrb().data[0]
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
    return qtree, tid_to_infos


def _refine_time_sample(dset, main_idx_to_gids, vidspace_box, iooa_thresh, time_sampler, get_image_valid_region_in_vidspace):
    """
    Refine the time sample based on spatial information
    """
    from watch.tasks.fusion.datamodules import temporal_sampling as tsm  # NOQA
    video_gids = time_sampler.video_gids

    gid_to_isbad = {}
    for gid in video_gids:
        vidspace_valid_poly = get_image_valid_region_in_vidspace(gid)
        gid_to_isbad[gid] = False
        if vidspace_valid_poly is not None:
            vidspace_box_poly = vidspace_box.to_shapley()[0]
            # flag = winspace_valid_poly.intersects(vidspace_box_poly)
            isect = vidspace_valid_poly.intersection(vidspace_box_poly)
            iooa = isect.area / vidspace_box_poly.area
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
    UNUSED. DEPRECATED OR FIND USE.

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
    UNUSED. DEPRECATED OR FIND USE.

    Ignore:
        >>> # xdoctest: +REQUIRES(env:DVC_DPATH)
        >>> import os
        >>> from watch.tasks.fusion.datamodules.spacetime_grid_builder import *  # NOQA
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

    winspace_space_dims = [96, 96]

    for vidid, trackinfos in vidid_to_tracks.items():
        positive_boxes = []
        for track_info in trackinfos:
            boxes = track_info['full_vid_box']
            positive_boxes.append(boxes.to_cxywh())
        positives = kwimage.Boxes.concatenate(positive_boxes)
        positives_samples = positives.to_cxywh()
        positives_samples.data[:, 2] = winspace_space_dims[0]
        positives_samples.data[:, 3] = winspace_space_dims[1]
        print('positive_boxes = {}'.format(ub.repr2(positive_boxes, nl=1)))

        video = coco_dset.index.videos[vidid]
        full_dims = [video['height'], video['width']]
        window_overlap = 0.0
        keepbound = 0

        window_dims_ = full_dims if winspace_space_dims == 'full' else winspace_space_dims
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


def visualize_sample_grid(dset, sample_grid, max_vids=2, max_frames=6):
    r"""
    Debug visualization for sampling grid

    Draws multiple frames.

    Places a red dot where there is a negative sample (at the center of the negative window)

    Places a blue dot where there is a positive sample

    Draws a yellow polygon over invalid spatial regions.

    Notes:
        * Dots are more intense when there are more temporal coverage of that dot.

        * Dots are placed on the center of the window.  They do not indicate its extent.

        * Dots are blue if they overlap any annotation in their temporal region
          so they may visually be near an annotation.

    Example:
        >>> from watch.tasks.fusion.datamodules.spacetime_grid_builder import *  # NOQA
        >>> from watch.demo.smart_kwcoco_demodata import demo_kwcoco_multisensor
        >>> dset = coco_dset = demo_kwcoco_multisensor(num_frames=3, dates=True, geodata=True, heatmap=True, rng=10)
        >>> window_overlap = 0.0
        >>> window_dims = (3, 32, 32)
        >>> keepbound = False
        >>> time_sampling = 'soft2+distribute'
        >>> use_centered_positives = True
        >>> use_grid_positives = 1
        >>> sample_grid = sample_video_spacetime_targets(
        >>>     dset, window_dims, window_overlap, time_sampling=time_sampling,
        >>>     use_grid_positives=use_grid_positives, use_centered_positives=use_centered_positives)
        >>> # xdoctest: +REQUIRES(--show)
        >>> import kwplot
        >>> plt = kwplot.autoplt()
        >>> canvas = visualize_sample_grid(dset, sample_grid, max_vids=1,
        >>>                                max_frames=3)
        >>> kwplot.imshow(canvas, doclf=1)
        >>> plt.gca().set_title(ub.codeblock(
            '''
            Places a red dot where there is a negative sample (at the center of the negative window)

            Places a blue dot where there is a positive sample

            Draws a yellow polygon over invalid spatial regions.
            '''))
        >>> kwplot.show_if_requested()
        >>> #
        >>> # Now demo this same grid, but where we are sampling at a different resolution
        >>> window_space_scale = 0.3
        >>> sample_grid2 = sample_video_spacetime_targets(
        >>>     dset, window_dims, window_overlap, time_sampling=time_sampling,
        >>>     use_grid_positives=use_grid_positives,
        >>>     use_centered_positives=use_centered_positives,
        >>>     window_space_scale=window_space_scale)
        >>> # xdoctest: +REQUIRES(--show)
        >>> import kwplot
        >>> plt = kwplot.autoplt()
        >>> canvas = visualize_sample_grid(dset, sample_grid2, max_vids=1,
        >>>                                max_frames=3)
        >>> kwplot.imshow(canvas, doclf=1, fnum=2)
        >>> plt.gca().set_title(ub.codeblock(
            '''
            Sampled using larger scaled windows
            '''))
        >>> kwplot.show_if_requested()

    Example:
        >>> # xdoctest: +REQUIRES(env:DVC_DPATH)
        >>> from watch.tasks.fusion.datamodules.spacetime_grid_builder import *  # NOQA
        >>> import watch
        >>> # dset = coco_dset = demo_kwcoco_multisensor(dates=True, geodata=True, heatmap=True)
        >>> dvc_dpath = watch.find_dvc_dpath(hardware='ssd', tags='phase2_data')
        >>> #coco_fpath = dvc_dpath / 'Drop2-Aligned-TA1-2022-02-15/combo_DILM_train.kwcoco.json'
        >>> coco_fpath = dvc_dpath / 'Aligned-Drop4-2022-08-08-TA1-S2-L8-ACC/data_vali.kwcoco.json'
        >>> big_dset = kwcoco.CocoDataset(coco_fpath)
        >>> vid_gids = big_dset.videos(names=['KR_R002']).images.lookup('id')[0]
        >>> idxs = np.linspace(0, len(vid_gids) - 1, 12).round().astype(int)
        >>> vid_gids = list(ub.take(vid_gids, idxs))
        >>> dset = big_dset.subset(vid_gids)
        >>> window_overlap = 0.0
        >>> window_dims = (3, 256, 256)
        >>> keepbound = False
        >>> time_sampling = 'soft2+distribute'
        >>> use_centered_positives = 0
        >>> use_grid_positives = 1
        >>> window_space_scale = '30GSD'
        >>> sample_grid = sample_video_spacetime_targets(
        >>>     dset, window_dims, window_overlap, time_sampling=time_sampling,
        >>>     use_grid_positives=True,
        >>>     use_centered_positives=use_centered_positives,
        >>>     window_space_scale=window_space_scale)
        >>> print(list(ub.unique([t['space_slice'] for t in sample_grid['targets']], key=ub.hash_data)))
        >>> # xdoctest: +REQUIRES(--show)
        >>> import kwplot
        >>> plt = kwplot.autoplt()
        >>> canvas = visualize_sample_grid(dset, sample_grid, max_vids=1, max_frames=12)
        >>> kwplot.imshow(canvas, doclf=1)
        >>> kwplot.show_if_requested()
        >>> plt.gca().set_title(ub.codeblock(
        >>>     f'''
        >>>     Sample window {window_dims} @ {window_space_scale}
        >>>     '''))
    """
    # Visualize the sample grid
    import pandas as pd
    targets = pd.DataFrame(sample_grid['targets'])

    dataset_canvases = []

    # max_vids = 2
    # max_frames = 6

    vidid_to_videodf = dict(list(targets.groupby('video_id')))

    orientation = 1

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
                final_canvas = vid_annot_dets.draw_on(
                    final_canvas, color='white', alpha=0.5, kpts=False,
                    labels=False)

            if kw_invalid_poly is not None:
                final_canvas = kw_invalid_poly.draw_on(final_canvas, color='yellow', alpha=0.5)

            # from watch import heuristics
            img = dset.index.imgs[gid]
            header_lines = heuristics.build_image_header_text(
                img=img, vidname=vidname)
            header_text = '\n'.join(header_lines)

            final_canvas = kwimage.draw_header_text(final_canvas, header_text, fit=True)
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
