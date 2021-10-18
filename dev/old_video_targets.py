def sample_video_spacetime_targets(dset, window_dims, window_overlap=0.0,
                                   negative_classes=None, keepbound=False,
                                   exclude_sensors=None,
                                   time_sampling='hard+distribute',
                                   use_annot_info=True):
    """
    Example:
        >>> # xdoctest: +REQUIRES(env:DVC_DPATH)
        >>> import os
        >>> from watch.tasks.fusion.datamodules.kwcoco_video_data import *  # NOQA
        >>> from watch.utils.util_data import find_smart_dvc_dpath
        >>> dvc_dpath = find_smart_dvc_dpath()
        >>> coco_fpath = dvc_dpath / 'drop1-S2-L8-aligned/data.kwcoco.json'
        >>> dset = kwcoco.CocoDataset(coco_fpath)
        >>> # Create a sliding window object for each specific image (because they may
        >>> # have different sizes, technically we could memoize this)
        >>> import kwarray
        >>> window_overlap = 0.5
        >>> window_dims = (2, 64, 64)
        >>> keepbound = False
        >>> exclude_sensors = None
        >>> sample_grid = sample_video_spacetime_targets(dset, window_dims, window_overlap)
        >>> time_sampling = 'hard+distribute'
        >>> positives = list(ub.take(sample_grid['targets'], sample_grid['positives_indexes']))
        _ = xdev.profile_now(sample_video_spacetime_targets)(dset, window_dims, window_overlap)

    Example:
        >>> from watch.tasks.fusion.datamodules.kwcoco_video_data import *  # NOQA
        >>> import ndsampler
        >>> import kwcoco
        >>> dset = kwcoco.CocoDataset.demo('vidshapes2-multispectral', num_frames=30)
        >>> # Create a sliding window object for each specific image (because they may
        >>> # have different sizes, technically we could memoize this)
        >>> import kwarray
        >>> window_overlap = 0.0
        >>> window_dims = (2, 96, 96)
        >>> keepbound = False
        >>> sample_grid = sample_video_spacetime_targets(dset, window_dims, window_overlap, time_sampling='hard+distribute')
        >>> sample_grid = sample_video_spacetime_targets(dset, window_dims, window_overlap, time_sampling='contiguous')

        _ = xdev.profile_now(sample_video_spacetime_targets)(dset, window_dims, window_overlap)
    """
    # Create a sliding window object for each specific image (because they may
    # have different sizes, technically we could memoize this)
    import pyqtree
    import itertools as it

    # window_overlap = 0.5
    window_space_dims = window_dims[1:3]
    window_time_dims = window_dims[0]
    print('window_time_dims = {!r}'.format(window_time_dims))
    keepbound = False
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

    if use_annot_info:
        # FIXME: HARD CODED CONSTANTS
        print('dset.cats = {}'.format(ub.repr2(dset.cats, nl=1)))
        special_cids = ub.ddict(set)
        special_aliases = {
            'pre_cids': {'background', 'No Activity'},
            'ignore_cids': {'ignore', 'Unknown', 'clouds'},

            'active': {'Active Construction'},
            'post_cids': {'Post Construction'},
        }
        for key, aliases in special_aliases.items():
            for name in aliases:
                if name in dset.index.name_to_cat:
                    special_cids[key].add(dset.index.name_to_cat[name]['id'])

    # Given an video
    all_vid_ids = list(dset.index.videos.keys())
    for video_id in ub.ProgIter(all_vid_ids, desc='sample video regions'):
        slider = vidid_to_space_slider[video_id]

        video_info = dset.index.videos[video_id]
        all_video_gids = list(dset.index.vidid_to_gids[video_id])

        if exclude_sensors is not None:
            sensor_coarse = dset.images(all_video_gids).lookup('sensor_coarse', '')
            flags = [s not in exclude_sensors for s in sensor_coarse]
            video_gids = list(ub.compress(all_video_gids, flags))
        else:
            video_gids = all_video_gids
        video_frame_idxs = np.array(list(range(len(video_gids))))

        # If the dataset has dates, we can use that
        gid_to_datetime = {}
        frame_dates = dset.images(video_gids).lookup('date_captured', None)
        for gid, date in zip(video_gids, frame_dates):
            if date is not None:
                gid_to_datetime[gid] = parser.parse(date)
        unixtimes = np.array([
            gid_to_datetime[gid].timestamp()
            if gid in gid_to_datetime else np.nan
            for gid in video_gids])

        if use_annot_info:
            qtree = pyqtree.Index((0, 0, video_info['width'], video_info['height']))
            qtree.aid_to_tlbr = {}
            tid_to_infos = ub.ddict(list)
            video_aids = dset.images(video_gids).annots.lookup('id')
            for aids, gid in zip(video_aids, video_gids):
                warp_vid_from_img = kwimage.Affine.coerce(
                    dset.index.imgs[gid]['warp_img_to_vid'])
                img_info = dset.index.imgs[gid]
                frame_index = img_info['frame_index']
                tids = dset.annots(aids).lookup('track_id')
                cids = dset.annots(aids).lookup('category_id')
                for tid, aid, cid in zip(tids, aids, cids):
                    imgspace_box = kwimage.Boxes([dset.index.anns[aid]['bbox']], 'xywh')
                    vidspace_box = imgspace_box.warp(warp_vid_from_img)
                    tlbr_box = vidspace_box.to_tlbr().data[0]
                    qtree.insert(aid, tlbr_box)
                    qtree.aid_to_tlbr[aid] = tlbr_box
                    dset.index.anns[aid]['bbox']
                    tid_to_infos[tid].append({
                        'gid': gid,
                        'cid': cid,
                        'frame_index': frame_index,
                        'vidspace_box': tlbr_box,
                        'cname': dset._resolve_to_cat(cid)['name'],
                        'aid': aid,
                    })

            tid_to_dframe = ub.map_vals(kwarray.DataFrameLight.from_dict, tid_to_infos)
            for track_dframe in tid_to_dframe.values():
                track_dframe['gid'] = np.array(track_dframe['gid'])
                track_dframe['frame_index'] = np.array(track_dframe['frame_index'])
                # Precompute for speed
                track_boxes = kwimage.Boxes(np.array(track_dframe['vidspace_box']), 'ltrb')
                track_dframe['track_pairwise_ious'] = track_boxes.ious(track_boxes)
                track_dframe['track_boxes'] = track_boxes

        if time_sampling == 'hard+distribute':
            sample_idxs = dilated_template_sample(unixtimes, window_time_dims)
            # sample_pattern = kwarray.one_hot_embedding(sample_idxs, len(unixtimes), dim=1).sum(axis=2)

        elif time_sampling == 'contiguous':
            time_slider = kwarray.SlidingWindow(
                (len(unixtimes),), (window_time_dims,), stride=(1,), keepbound=True,
                allow_overshoot=True)
            all_indexes = np.arange(len(unixtimes))
            sample_idxs = [all_indexes[sl] for sl in time_slider]
        else:
            raise NotImplementedError(time_sampling)

        for space_region in list(slider):
            y_sl, x_sl = space_region

            # Find all annotations that pass through this spatial region
            if use_annot_info:
                vid_box = kwimage.Boxes.from_slice((y_sl, x_sl))
                query = vid_box.to_tlbr().data[0]
                isect_aids = set(qtree.intersect(query))

            for frame_idxs in sample_idxs:
                gids = list(ub.take(video_gids, frame_idxs))

                if use_annot_info:
                    if isect_aids:
                        has_annot = any(
                            bool(isect_aids & _aids)
                            for _aids in dset.index.gid_to_aids.values())
                    else:
                        has_annot = False

                    if has_annot:
                        positive_idxs.append(len(targets))
                    else:
                        # Hack: exclude all annotated regions from negative sampling
                        negative_idxs.append(len(targets))

                targets.append({
                    'gids': gids,
                    'space_slice': space_region,
                    # 'changes': ','.join(changes),
                    # 'region_tracks': region_tracks,
                })

        if 0:
            # For each frame, calculate a weight proportional to how much we would
            # like to include any other frame in the sample.
            sensors = np.array(dset.images(video_gids).lookup('sensor_coarse', None))
            dilated_weights = dilated_time_weights(unixtimes)['final']
            same_sensor = sensors[:, None] == sensors[None, :]
            sensor_weights = ((same_sensor * 0.5) + 0.5)
            pair_weights = dilated_weights * sensor_weights
            pair_weights[np.eye(len(pair_weights), dtype=bool)] = 1.0

            classes = dset.object_categories()
            nancx = len(classes) + 1
            track_phase_mat = []
            # bg_cid = classes.node_to_cid['No Activity']
            for tid, track_dframe in tid_to_dframe.items():
                # FIXME; BROKEN, NOT THE RIGHT INDEX
                at_idxs = np.searchsorted(video_frame_idxs, track_dframe['frame_index'])
                track_phase = np.full(len(video_frame_idxs), fill_value=nancx)
                track_cids = np.array(track_dframe['cid'])
                track_cxs = [classes.id_to_idx[cid] for cid in track_cids]
                track_phase[at_idxs] = track_cxs
                track_phase_mat.append(track_phase)
            track_phase_mat = np.array(track_phase_mat)

            tid_to_track_changemat = {}
            for tid, track_dframe in tid_to_dframe.items():
                # For each track, find frames where phase boundries occur
                track_cids = np.array(track_dframe['cid'])
                at_idxs = np.searchsorted(video_frame_idxs, track_dframe['frame_index'])
                track_phase = np.full(len(video_frame_idxs), fill_value=np.nan)
                track_phase[at_idxs] = track_cids
                track_missing = np.isnan(track_phase)
                is_change = (track_phase[:, None] != track_phase[None, :])
                is_change[track_missing, :] = 0
                is_change[:, track_missing] = 0
                tid_to_track_changemat[tid] = is_change

            # print('tid_to_info = {}'.format(ub.repr2(tid_to_info, nl=2, sort=0)))
            for space_region in list(slider):
                y_sl, x_sl = space_region

                # Find all annotations that pass through this spatial region
                vid_box = kwimage.Boxes.from_slice((y_sl, x_sl))
                query = vid_box.to_tlbr().data[0]
                isect_aids = sorted(set(qtree.intersect(query)))

                isect_annots = dset.annots(isect_aids)
                unique_tracks = set(isect_annots.lookup('track_id'))
                region_tid_to_info = ub.dict_subset(tid_to_dframe, unique_tracks)

                # precompute track metrics over pairwise frames for speed
                tid_to_space_window_iooa = {}
                window_change_weights = []
                for tid, track_dframe in region_tid_to_info.items():
                    track_boxes = track_dframe['track_boxes']
                    track_fxs = track_dframe['frame_index']
                    # track_boxes = kwimage.Boxes(track_dframe['vidspace_box'], 'tlbr')
                    track_iooa = vid_box.iooas(track_boxes)[0, :]
                    is_visible = track_iooa > 0
                    visible_fxs = track_fxs[is_visible]
                    invisible_fxs = track_fxs[~is_visible]
                    track_change_weight = tid_to_track_changemat[tid].copy().astype(np.float32)
                    track_change_weight[invisible_fxs, :][:, invisible_fxs] = 0
                    track_change_weight[visible_fxs, :][:, invisible_fxs] = 0.3
                    track_change_weight[invisible_fxs, :][:, visible_fxs] = 0.3
                    track_change_weight[invisible_fxs, :][:, visible_fxs] = 0.3
                    tid_to_space_window_iooa[tid] = track_iooa
                    window_change_weights.append(track_change_weight)

                if window_change_weights:
                    has_annot = True
                    frame_change_w = np.add.reduce(window_change_weights)
                    min_p = 0.1
                    frame_change_w = (frame_change_w * (1 - min_p)) + min_p
                    frame_w = (dilated_weights * frame_change_w)
                else:
                    has_annot = False
                    frame_w = (dilated_weights)

                for base_idx in video_frame_idxs:
                    chosen = affinity_sample(
                        frame_w, window_time_dims, include_indices=[base_idx],
                        jit=0)
                    gids = list(ub.take(video_gids, chosen))

                    if has_annot:
                        positive_idxs.append(len(targets))
                    else:
                        # Hack: exclude all annotated regions from negative sampling
                        negative_idxs.append(len(targets))

                    # TODO: would it be more efficient to simply iterate over
                    # spatial positions and then return information about what the
                    # dataset was allowed to to to augment the target?  In other
                    # words, allow training to choose a different dilated temporal
                    # sample every time?

                    targets.append({
                        'gids': gids,
                        'space_slice': space_region,
                        # 'changes': ','.join(changes),
                        # 'region_tracks': region_tracks,
                    })

            if 0:
                # PPR Review hacks:
                # Generate all combinations of sample frames
                # TODO: ITERATING THROUGH ALL COMBINATIONS IS SLOW!
                # Could likely reparametarize to sample implicitly in getitem
                for frame_idxs in list(it.combinations(video_frame_idxs, window_time_dims)):

                    # Default is to assume this spacetime region has no change
                    changes = []

                    any_visible = False

                    gids = list(ub.take(video_gids, frame_idxs))
                    region_tracks = []
                    # For each track that passes through this region
                    for tid, track_dframe in region_tid_to_info.items():

                        # Interpolate / Extrapolate track annotations onto the
                        # sample frame indexes The track might not be annotated on
                        # each frame. For each timestep check the most recent state
                        # of the track.
                        sampled_info = ub.ddict(list)
                        _explicit_track_fxs = track_dframe['frame_index']
                        _space_window_iooa = tid_to_space_window_iooa[tid]
                        most_recent_idxs = np.searchsorted(_explicit_track_fxs, frame_idxs, 'right') - 1
                        # prev_box = None
                        prev_idx = None
                        for idx in most_recent_idxs:
                            if idx < 0:
                                # The sample frame is before this track starts
                                cid = ub.peek(special_cids['pre_cids'])
                                curr_box = None
                                space_window_iooa = 0
                                prev_iou = np.nan
                            elif idx > _explicit_track_fxs[-1]:
                                # The sampled frame is after this track ends
                                cid = track_dframe['cid'][idx]
                                curr_box = track_dframe['vidspace_box'][idx]
                                space_window_iooa = _space_window_iooa[idx]
                                if prev_idx is None:
                                    prev_iou = np.nan
                                else:
                                    prev_iou = track_dframe['track_pairwise_ious'][idx][prev_idx]
                            else:
                                cid = track_dframe['cid'][idx]
                                curr_box = track_dframe['vidspace_box'][idx]
                                space_window_iooa = _space_window_iooa[idx]
                                if prev_idx is None:
                                    prev_iou = np.nan
                                else:
                                    prev_iou = track_dframe['track_pairwise_ious'][idx][prev_idx]
                            sampled_info['cid'].append(cid)
                            sampled_info['box'].append(curr_box)
                            sampled_info['space_window_iooa'].append(space_window_iooa)
                            sampled_info['prev_iou'].append(prev_iou)
                            prev_idx = idx

                        # Heuristic: flag this region as a positive if any of these
                        # heuristics are detected.  TODO: we need to figure out the
                        # best method for determening if a space-time window
                        # contains a positive example of change or not. What is the
                        # best way to encode this in a kwcoco dataset?

                        # Detect if the category of track changes.
                        is_visible = np.array(sampled_info['space_window_iooa']) > 0.1
                        is_change_visible = (is_visible[0:-1] & is_visible[1:])
                        is_moving = np.array(sampled_info['prev_iou'][1:]) < 0.6
                        is_visibly_moving = is_change_visible & is_moving

                        if is_visible.any():
                            any_visible = True

                        if is_visibly_moving.any():
                            changes.append('visibly_moving')

                        if 1:
                            has_pre = set(sampled_info['cid']) & special_cids['pre_cids']
                            has_active = set(sampled_info['cid']) & special_cids['active']
                            if len(has_pre) and len(has_active):
                                changes.append('hack')

                        if 0:
                            unique_cids = set(sampled_info['cid']) - special_cids['ignore_cids']
                            # TODO: dont mark change when post_cid moves to background
                            if len(unique_cids) > 1:
                                changes.append('category change')

                        region_tracks.append({
                            'tid': tid,
                            'sampled_info': sampled_info,
                            # 'is_visibly_moving': is_visibly_moving,
                            # 'is_moving': is_moving,
                            # 'is_change_visible': is_change_visible,
                            # 'is_visible': is_visible,
                        })

                    if changes:
                        positive_idxs.append(len(targets))
                    else:
                        # Hack: exclude all annotated regions from negative sampling
                        if any_visible:
                            continue
                        negative_idxs.append(len(targets))

                    targets.append({
                        'gids': gids,
                        'space_slice': space_region,
                        'changes': ','.join(changes),
                        'region_tracks': region_tracks,
                    })

    print('Found {} targets'.format(len(targets)))
    if use_annot_info:
        print('Found {} positives'.format(len(positive_idxs)))
        print('Found {} negatives'.format(len(negative_idxs)))

    sample_grid = {
        'positives_indexes': positive_idxs,
        'negatives_indexes': negative_idxs,
        'targets': targets,
    }
    return sample_grid
