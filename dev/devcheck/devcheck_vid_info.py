def debug_video_information(dset, video_id):
    """
    Ignore:
        >>> # xdoctest: +REQUIRES(env:DVC_DPATH)
        >>> from watch.tasks.fusion.datamodules.kwcoco_video_data import *  # NOQA
        >>> from watch.tasks.fusion.datamodules.temporal_sampling import *  # NOQA
        >>> from watch.utils.util_data import find_smart_dvc_dpath
        >>> dvc_dpath = find_smart_dvc_dpath()
        >>> coco_fpath = dvc_dpath / 'drop1-S2-L8-aligned/data.kwcoco.json'
        >>> dset = kwcoco.CocoDataset(coco_fpath)
        >>> video_id = 3

        for video_id in dset.index.videos.keys():
            debug_video_information(dset, video_id)
    """
    exclude_sensors = None
    # exclude_sensors = {'L8'}
    video_info = dset.index.videos[video_id]
    video_name = video_info['name']
    all_video_gids = list(dset.index.vidid_to_gids[video_id])

    if exclude_sensors is not None:
        sensor_coarse = dset.images(all_video_gids).lookup('sensor_coarse', '')
        flags = [s not in exclude_sensors for s in sensor_coarse]
        video_gids = list(ub.compress(all_video_gids, flags))
    else:
        video_gids = all_video_gids
    video_gids = np.array(video_gids)

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

    window_time_dims = 5

    sample_idxs = dilated_template_sample(unixtimes, window_time_dims)
    sample_pattern_v1 = kwarray.one_hot_embedding(sample_idxs, len(unixtimes), dim=1).sum(axis=2)

    # For each frame, calculate a weight proportional to how much we would
    # like to include any other frame in the sample.
    sensors = np.array(dset.images(video_gids).lookup('sensor_coarse', None))
    dilated_weights = soft_frame_affinity(unixtimes)['final']
    same_sensor = sensors[:, None] == sensors[None, :]
    sensor_weights = ((same_sensor * 0.5) + 0.5)
    pair_weights = dilated_weights * sensor_weights
    pair_weights[np.eye(len(pair_weights), dtype=bool)] = 1.0

    # Get track info in this video
    classes = dset.object_categories()
    tid_to_infos = ub.ddict(list)
    video_aids = dset.images(video_gids).annots.lookup('id')
    for aids, gid, frame_idx in zip(video_aids, video_gids, video_frame_idxs):
        tids = dset.annots(aids).lookup('track_id')
        cids = dset.annots(aids).lookup('category_id')
        for tid, aid, cid in zip(tids, aids, cids):
            dset.index.anns[aid]['bbox']
            tid_to_infos[tid].append({
                'gid': gid,
                'cid': cid,
                'aid': aid,
                'cx': classes.id_to_idx[cid],
                'frame_idx': frame_idx,
            })

    nancx = len(classes) + 1
    track_phase_mat = []
    # bg_cid = classes.node_to_cid['No Activity']
    for _tid, track_infos in tid_to_infos.items():
        track_phase = np.full(len(video_frame_idxs), fill_value=nancx)
        at_idxs = np.array([row['frame_idx'] for row in track_infos])
        track_cxs = np.array([row['cx'] for row in track_infos])
        track_phase[at_idxs] = track_cxs
        track_phase_mat.append(track_phase)
    track_phase_mat = np.array(track_phase_mat)

    if 1:
        import kwplot
        import pandas as pd
        kwplot.autompl()
        sns = kwplot.autosns()

        fnum = video_id

        utils.category_tree_ensure_color(classes)
        color_lut = np.zeros((nancx + 1, 3))
        for _node, node_data in classes.graph.nodes.items():
            cx = classes.id_to_idx[node_data['id']]
            color_lut[cx] = node_data['color']
        color_lut[nancx] = (0, 0, 0)
        colored_track_phase = color_lut[track_phase_mat]

        if 0:
            fig = kwplot.figure(fnum=fnum, pnum=(3, 4, slice(0, 3)), doclf=True)
            ax = fig.gca()
            kwplot.imshow(colored_track_phase, ax=ax)
            ax.set_xlabel('observation index')
            ax.set_ylabel('track')
            ax.set_title(f'{video_name} tracks')

            fig = kwplot.figure(fnum=fnum, pnum=(3, 4, 4))
            label_to_color = {
                node: data['color']
                for node, data in classes.graph.nodes.items()}
            label_to_color = ub.sorted_keys(label_to_color)
            legend_img = utils._memo_legend(label_to_color)
            kwplot.imshow(legend_img)

            # pairwise affinity
            fig = kwplot.figure(fnum=fnum, pnum=(3, 1, 2))
            ax = fig.gca()
            kwplot.imshow(kwimage.normalize(pair_weights), ax=ax)
            ax.set_title('pairwise affinity')

            # =====================
            # Show Sample Pattern in heatmap
            datetimes = np.array([datetime.datetime.fromtimestamp(t) for t in unixtimes])
            # dates = np.array([datetime.datetime.fromtimestamp(t).date() for t in unixtimes])
            #
            df = pd.DataFrame(sample_pattern_v1)
            df.index.name = 'index'
            #
            df.columns = pd.to_datetime(datetimes).date
            df.columns.name = 'date'
            #
            kwplot.figure(fnum=fnum, pnum=(3, 1, 3))
            ax = sns.heatmap(data=df)
            # ax.set_title(f'Sample Pattern wrt Available Observations: {video_name}')
            ax.set_title('Sample pattern')
            ax.set_xlabel('Observation Index')
            ax.set_ylabel('Sample Index')

        # extract track animations
        for tid, track_infos in tid_to_infos.items():

            vidspace_boxes = []
            for info in track_infos:
                aid = info['aid']
                gid = info['gid']
                warp_vid_from_img = kwimage.Affine.coerce(
                    dset.index.imgs[gid]['warp_img_to_vid'])
                imgspace_box = kwimage.Boxes([dset.index.anns[aid]['bbox']], 'xywh')
                vidspace_box = imgspace_box.warp(warp_vid_from_img).quantize()
                vidspace_boxes.append(vidspace_box)

            all_vidspace_boxes = kwimage.Boxes.concatenate(vidspace_boxes)
            full_vid_box = all_vidspace_boxes.bounding_box().to_xywh()

            gid_to_track = ub.group_items(track_infos, key=lambda x: x['gid'])

            temporal_stack = []
            for frame_idx, gid in zip(video_frame_idxs, video_gids):

                img = dset.index.imgs[gid]

                delayed = dset.delayed_load(gid, channels='red|green|blue', space='video')
                delayed_chip = delayed.crop(full_vid_box.to_slices()[0])
                chip = delayed_chip.finalize()
                chip = kwimage.normalize_intensity(chip)

                info = gid_to_track.get(gid, [None])[0]
                if info is not None:
                    aid = info['aid']
                    gid = info['gid']
                    cid = info['cid']

                    warp_vid_from_img = kwimage.Affine.coerce(
                        dset.index.imgs[gid]['warp_img_to_vid'])
                    imgspace_box = kwimage.Boxes([dset.index.anns[aid]['bbox']], 'xywh')
                    vidspace_box = imgspace_box.warp(warp_vid_from_img).quantize()
                    relvidspace_box = vidspace_box.translate(-full_vid_box.data[0, 0:2])
                    class_color = color_lut[classes.id_to_idx[cid]]
                    chip = relvidspace_box.draw_on(chip, color=class_color)

                chip = kwimage.imresize(chip, min_dim=128).clip(0, 1)

                max_dim = chip.shape[1]

                sensor_coarse = img.get('sensor_coarse', '')

                vertical_stack = []
                header_dims = {'width': max_dim}
                header_part = util_kwimage.draw_header_text(
                    image=header_dims, fit='shrink',
                    text=f't={frame_idx} gid={gid} {sensor_coarse}', color='salmon')
                vertical_stack.append(header_part)

                date_captured = img.get('date_captured', '')
                if date_captured:
                    header_part = util_kwimage.draw_header_text(
                        header_dims, fit='shrink', text=f'{date_captured}',
                        color='salmon')
                    vertical_stack.append(header_part)

                chip = kwimage.ensure_uint255(chip)
                vertical_stack.append(chip)

                frame_viz = kwimage.stack_images(vertical_stack, axis=0)

                temporal_stack.append(frame_viz)

            track_viz = kwimage.stack_images_grid(temporal_stack, pad=3, chunksize=20)
            track_viz = util_kwimage.draw_header_text(track_viz, f'{video_name}, tid={tid}')

            dpath = pathlib.Path(ub.ensuredir('./track_viz'))
            fpath = dpath / f'viz_track_{video_name}_tid={tid}.jpg'
            kwimage.imwrite(str(fpath), track_viz)

            # kwplot.imshow(track_viz)
