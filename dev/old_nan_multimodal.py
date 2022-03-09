            # ------------------
            # OLD CODE

            # collect sample
            sample = sampler.load_sample(
                tr_, with_annots=with_annots,
                padkw={'constant_values': np.nan}
            )

            if ALLOW_RESAMPLE:
                # If any image is junk allow for a resample
                is_frame_bad = np.isnan(sample['im']).all(axis=(1, 2, 3))
                if np.any(is_frame_bad):
                    gids = np.array(tr_['gids'])
                    good_gids = gids[~is_frame_bad].tolist()
                    vidid = tr_['video_id']
                    time_sampler = self.new_sample_grid['vidid_to_time_sampler'][vidid]
                    video_gids = time_sampler.video_gids
                    bad_gids = gids[is_frame_bad].tolist()
                    new_bad_gids = bad_gids
                    iter_idx = 0
                    while len(new_bad_gids):
                        print('resampling: {}'.format(index))
                        include_idxs = np.where(kwarray.isect_flags(time_sampler.video_gids, good_gids))[0]
                        exclude_idxs = np.where(kwarray.isect_flags(time_sampler.video_gids, bad_gids))[0]
                        chosen, info = time_sampler.sample(include=include_idxs, exclude=exclude_idxs, error_level=1, return_info=True)
                        new_idxs = np.setdiff1d(chosen, include_idxs)
                        new_gids = video_gids[new_idxs]

                        tr_test = tr_.copy()
                        tr_test['gids'] = new_gids
                        test_sample = sampler.load_sample(
                            tr_test, with_annots=False, padkw={'constant_values': np.nan}
                        )
                        new_is_frame_bad = np.isnan(test_sample['im']).all(axis=(1, 2, 3))
                        new_good_gids = new_gids[~new_is_frame_bad].tolist()
                        new_bad_gids = new_gids[new_is_frame_bad].tolist()
                        bad_gids.extend(new_bad_gids)
                        good_gids.extend(new_good_gids)
                        iter_idx += 1
                        if iter_idx > 100:
                            raise Exception('Something is wrong')

                    resampled_gids = ub.oset(video_gids) & good_gids
                    tr_['gids'] = resampled_gids
                    # finalize resample sample
                    sample = sampler.load_sample(
                        tr_, with_annots=with_annots,
                        padkw={'constant_values': np.nan}
                    )

            vidid = sample['tr']['vidid']
            video = self.sampler.dset.index.videos[vidid]

            if self.normalize_perframe:
                im = sample['im']
                # mask = ~np.isnan(im)
                # mask[im == 0] = False
                reorg = einops.rearrange(im, 't h w c -> (t c) h w')
                to_restack = []
                for item in reorg:
                    mask = (item != 0) & np.isfinite(item)
                    norm_item = kwimage.normalize_intensity(item, params={
                        'high': 0.90,
                        'mid': 0.5,
                        'low': 0.01,
                        'mode': 'linear',
                    }, mask=mask)
                    to_restack.append(norm_item)
                norm_im = einops.rearrange(np.stack(to_restack, axis=0), '(t c) h w -> t h w c', c=im.shape[3])
                sample['im'] = norm_im

            if self.special_inputs or self.diff_inputs:
                import xarray as xr
                im = sample['im']
                # chan_coords = list(self.sample_channels.values())[0].split('|')
                chan_coords = self.sample_channels.streams()[0].parsed
                sample_im = xr.DataArray(
                    im, dims=('t', 'h', 'w', 'c'),
                    coords={'c': chan_coords})
                special_ims = []
                if self.special_inputs:
                    bands = {c: sample_im.sel(c=c).data for c in chan_coords}
                    indexes = util_bands.specialized_index_bands(bands=bands)
                    indexes = ub.map_vals(np.nan_to_num, indexes)
                    special_ims = [
                        xr.DataArray(
                            indexes[v][..., None],
                            dims=('t', 'h', 'w', 'c'),
                            coords={'c': [v]}
                        )
                        for _, values in self.special_inputs.items()
                        for v in values
                    ]
                concat1 = xr.concat([sample_im] + special_ims, dim='c')

                main_idx_ = tr.get('main_idx', 0)
                if self.match_histograms:
                    nodata_mask = np.isnan(concat1)  # NOQA
                    tmp = np.nan_to_num(concat1)
                    # Hack: do before diff
                    from skimage import exposure  # NOQA
                    from skimage.exposure import match_histograms
                    main_idx_ = min(main_idx_, len(tmp) - 1)
                    reference = tmp[main_idx_]
                    for idx, raw_frame in enumerate(tmp):
                        if idx != main_idx_:
                            new_frame = match_histograms(raw_frame, reference, multichannel=True)
                            tmp[idx] = new_frame
                    concat1[...] = tmp

                # TODO: add the matching step somewhere around here

                if self.diff_inputs:
                    diff_ims = np.abs(concat1.diff(dim='t'))
                    diff_ims.coords.update({
                        'c': ['D' + s for s in diff_ims.coords['c'].data]
                    })
                    diff_ims = diff_ims.pad({'t': (1, 0)}).fillna(0)
                    concat2 = xr.concat([concat1, diff_ims], dim='c')
                else:
                    concat2 = concat1

                # TODO: multi-modal inputs
                requested_channel_order = self.input_channels.spec.split('|')
                final = concat2.sel(c=requested_channel_order)
                raw_frame_list = final
            else:
                # Access the sampled image and annotation data
                raw_frame_list = sample['im']

            # TODO: use this
            # TODO: read QA bands, input other special QA bands
            nodata_mask = np.isnan(raw_frame_list)  # NOQA
            raw_frame_list = np.nan_to_num(raw_frame_list)

            if not self.special_inputs and not self.diff_inputs:
                main_idx_ = tr.get('main_idx', 0)
                if self.match_histograms:
                    nodata_mask = nkey_to_learned_tensorp.isnan(raw_frame_list)  # NOQA
                    raw_frame_list = np.nan_to_num(raw_frame_list)
                    # Hack: do before diff
                    from skimage import exposure  # NOQA
                    from skimage.exposure import match_histograms
                    main_idx_ = min(main_idx_, len(raw_frame_list) - 1)
                    reference = raw_frame_list[main_idx_]
                    for idx, raw_frame in enumerate(raw_frame_list):
                        if idx != main_idx_:
                            new_frame = match_histograms(raw_frame, reference, multichannel=True)
                            raw_frame_list[idx] = new_frame

            raw_det_list = sample['annots']['frame_dets']
            raw_gids = sample['tr']['gids']

            # channel_keys = sample['tr']['_coords']['c'].values.tolist()
            # print('channel_keys = {!r}'.format(channel_keys))

            stream_specs = self.input_channels.streams()
            assert len(stream_specs) == 1, 'no late fusion yet'
            mode_key = self.input_channels.fuse().spec

            # print('mode_key = {!r}'.format(mode_key))
            # mode_key = '|'.join(channel_keys)

            # Break data down on a per-frame basis so we can apply image-based
            # augmentations.
            frame_items = []

            if self.sample_shape is None:
                input_dsize = raw_frame_list[0].shape[0:2][::-1]
            else:
                input_dsize = self.sample_shape[-2:][::-1]

            # hack for augmentation
            # TODO: make a nice "augmenter" pipeline
            do_hflip = False
            do_vflip = False
            if not self.disable_augmenter and self.mode == 'fit':
                def make_hflipper(width):
                    def hflip(pt):
                        new = np.hstack([width - pt[:, 0:1], pt[:, 1:2]])
                        return new
                    return hflip
                hflipper = make_hflipper(input_dsize[0])
                do_hflip = np.random.rand() > 0.5

                def make_vflipper(height):
                    def vflip(pt):
                        new = np.hstack([pt[:, 0:1], height - pt[:, 1:2]])
                        return new
                    return vflip
                vflipper = make_vflipper(input_dsize[1])
                do_vflip = np.random.rand() > 0.5

            prev_frame_cidxs = None

            if not self.inference_only:
                num_frames = len(raw_frame_list)
                frame0 = raw_frame_list[0]
                # from watch.utils import util_kwimage
                # Learn more from the center of the space-time patch
                time_weights = kwimage.gaussian_patch((1, num_frames))[0]
                time_weights = time_weights / time_weights.max()
                space_weights = util_kwimage.upweight_center_mask(frame0.shape[0:2])
                # time_weights[1:] = 0

            for time_idx, (frame, dets, gid) in enumerate(zip(raw_frame_list, raw_det_list, raw_gids)):
                img = self.sampler.dset.imgs[gid]

                frame = np.asarray(frame, dtype=np.float32)

                if do_hflip:
                    frame = np.fliplr(frame)
                    dets = dets.warp(hflipper)

                if do_vflip:
                    frame = np.flipud(frame)
                    dets = dets.warp(vflipper)

                # Resize the sampled window to the target space for the network
                frame, info = kwimage.imresize(frame, dsize=input_dsize,
                                               interpolation='linear',
                                               antialias=True,
                                               return_info=True)
                # Remember to apply any transform to the dets as well
                dets = dets.scale(info['scale'])
                dets = dets.translate(info['offset'])

                # ensure channel dim is not squeezed
                frame_hwc = kwarray.atleast_nd(frame, 3)
                # catch nans
                frame_hwc[np.isnan(frame_hwc)] = -1.
                # rearrange image axes for pytorch
                frame_chw = einops.rearrange(frame_hwc, 'h w c -> c h w')
                input_chw = frame_chw

                if not self.inference_only:
                    # allocate class masks
                    bg_idx = self.bg_idx

                    space_shape = frame.shape[:2]
                    frame_cidxs = np.full(space_shape, dtype=np.int32,
                                          fill_value=bg_idx)

                    ohe_shape = (len(self.classes),) + space_shape
                    frame_class_ohe = np.full(ohe_shape, dtype=np.uint8,
                                              fill_value=0)

                    # Ignore for saliency
                    saliency_ignore = np.full(space_shape, dtype=np.uint8,
                                              fill_value=0)

                    # Ignore for class
                    frame_class_ignore = np.full(space_shape, dtype=np.uint8,
                                                 fill_value=0)

                    # Rasterize frame targets
                    ann_polys = dets.data['segmentations'].to_polygon_list()
                    ann_aids = dets.data['aids']
                    ann_cids = dets.data['cids']
                    # Note: it is important to respect class indexes, ids, and name
                    # mappings
                    # TODO: layer ordering? Multiclass prediction?
                    for poly, aid, cid in zip(ann_polys, ann_aids, ann_cids):  # NOQA
                        cidx = self.classes.id_to_idx[cid]
                        catname = self.classes.id_to_node[cid]
                        if catname in self.background_classes:
                            pass
                        elif catname in self.ignore_classes:
                            poly.fill(saliency_ignore, value=1)
                            poly.fill(frame_class_ignore, value=1)
                        else:
                            if catname in self.undistinguished_classes:
                                poly.fill(frame_class_ohe[cidx], value=0)
                            else:
                                poly.fill(frame_class_ohe[cidx], value=1)

                    # Postprocess (Dilate?) the truth map
                    for cidx, class_map in enumerate(frame_class_ohe):
                        # class_map = util_kwimage.morphology(class_map, 'dilate', kernel=5)
                        frame_cidxs[class_map > 0] = cidx

                    if self.upweight_centers:
                        frame_weights = space_weights * time_weights[time_idx]
                    else:
                        frame_weights = 1.0

                    saliency_weights = frame_weights * (1 - saliency_ignore)
                    class_weights = frame_weights * (1 - frame_class_ignore)

                    # convert annotations into a change detection task suitable for
                    # the network.
                    if self.with_change:
                        if prev_frame_cidxs is None:
                            frame_change = None
                        else:
                            frame_change = (frame_cidxs != prev_frame_cidxs).astype(np.uint8)
                            # Clean up the change target
                            frame_change = util_kwimage.morphology(frame_change, 'open', kernel=3)
                            frame_change = torch.from_numpy(frame_change)
                    else:
                        frame_change = None

                # convert to torch
                frame_item = {
                    'gid': gid,
                    'date_captured': img.get('date_captured', ''),
                    'timestamp': np.nan,  # this code will 90% get removed, so no need to fix
                    'sensor': img.get('sensor_coarse', ''),
                    'modes': {
                        mode_key: torch.from_numpy(input_chw),
                    },
                    'change': None,
                    'class_idxs': None,
                    'ignore': None,
                    'time_index': time_idx,
                }

                if not self.inference_only:
                    frame_item.update({
                        'change': frame_change,
                        'class_idxs': torch.from_numpy(frame_cidxs),
                        'ignore': torch.from_numpy(saliency_ignore),
                        'class_weights': torch.from_numpy(class_weights),
                        'saliency_weights': torch.from_numpy(saliency_weights),
                    })
                    prev_frame_cidxs = frame_cidxs
                frame_items.append(frame_item)

