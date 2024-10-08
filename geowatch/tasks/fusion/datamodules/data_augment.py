"""
Data augmentation utilities
"""
import kwimage
import ubelt as ub
import numpy as np

from geowatch.tasks.fusion.datamodules import data_utils


try:
    import line_profiler
    profile = line_profiler.profile
except Exception:
    profile = ub.identity


class SpacetimeAugmentMixin:

    def _expand_targets_time(self, n_time_expands):
        """
        Increase the number of test-time targets by expanding them in time.
        """
        sample_grid = self.new_sample_grid
        expanded_targets = []
        assert not sample_grid['positives_indexes'], 'unhandled'
        assert not sample_grid['negatives_indexes'], 'unhandled'
        targets = sample_grid['targets']
        for target in targets:
            seen_ = set()
            # Add the original sample
            expanded_targets.append(target)
            seen_.add(tuple(target['gids']))
            # Add the expanded samples
            for _ in range(n_time_expands):
                target_ = target.copy()
                target_ = self._augment_target_time(target_)
                new_gids = tuple(target_['gids'])
                if new_gids not in seen_:
                    expanded_targets.append(target_)
                    seen_.add(tuple(target_['gids']))
        print(f'Temporal augmentation expanded {len(targets)=} '
              f'to {len(expanded_targets)=}')
        sample_grid['targets'] = expanded_targets
        self.length = len(expanded_targets)

    def _expand_targets_fliprot(self, n_fliprot):
        """
        Increase the number of test-time targets via flips
        """

        sample_grid = self.new_sample_grid
        expanded_targets = []
        assert not sample_grid['positives_indexes'], 'unhandled'
        assert not sample_grid['negatives_indexes'], 'unhandled'
        targets = sample_grid['targets']
        # See data_utils.fliprot_annot for a visualization of various
        # flip/rotations
        unique_fliprots = [
            {'rot_k': 0, 'flip_axis': None},
            {'rot_k': 0, 'flip_axis': (0,)},
            {'rot_k': 1, 'flip_axis': None},
            {'rot_k': 1, 'flip_axis': (0,)},
            {'rot_k': 2, 'flip_axis': None},
            {'rot_k': 2, 'flip_axis': (0,)},
            {'rot_k': 3, 'flip_axis': None},
            {'rot_k': 3, 'flip_axis': (0,)},
        ]
        for target in targets:
            # Add the original sample
            expanded_targets.append(target)
            # Add the expanded samples
            assert n_fliprot <= 7
            for idx in range(1, n_fliprot + 1):
                target_ = target.copy()
                target_['fliprot_params'] = unique_fliprots[idx]
                expanded_targets.append(target_)

        print(f'Fliprot augmentation expanded {len(targets)=} '
              f'to {len(expanded_targets)=}')

        sample_grid['targets'] = expanded_targets
        self.length = len(expanded_targets)

    @profile
    def _augment_target_time(self, target_):
        """
        Jitters the time sample in a target
        """
        vidid = target_['video_id']
        valid_gids = self.new_sample_grid['vidid_to_valid_gids'][vidid]
        if len(valid_gids) > 1:
            # optimization: dont time augment when there is only 1 frame
            time_sampler = self.new_sample_grid['vidid_to_time_sampler'][vidid]
            new_idxs = time_sampler.sample(target_['main_idx'])
            new_gids = list(ub.take(valid_gids, new_idxs))
            target_['gids'] = new_gids
        return target_

    @profile
    def _augment_spacetime_target(self, target_):
        """
        Given a target dictionary, shift around the space and time slice

        Ignore:
            >>> from geowatch.tasks.fusion.datamodules.kwcoco_dataset import *  # NOQA
            >>> import ndsampler
            >>> import kwcoco
            >>> import geowatch
            >>> coco_dset = geowatch.coerce_kwcoco('geowatch')
            >>> sampler = ndsampler.CocoSampler(coco_dset)
            >>> self = KWCocoVideoDataset(sampler, time_dims=2, window_dims=(128, 128))
            >>> index = 0
            >>> target = self.new_sample_grid['targets'][index]
            >>> target_ = target.copy()
            >>> target_ = self._augment_spacetime_target(target_)
            >>> print('target  = {!r}'.format(target))
            >>> print('target_ = {!r}'.format(target_))
        """
        # TODO: make a nice "augmenter" pipeline
        augment_time_resample_rate = self.config['augment_time_resample_rate']
        augment_space_shift_rate = self.config['augment_space_shift_rate']
        augment_space_xflip = self.config['augment_space_xflip']
        augment_space_yflip = self.config['augment_space_yflip']
        augment_space_rot = self.config['augment_space_rot']

        do_shift = False
        if not self.disable_augmenter and self.mode == 'fit':
            # do_shift = np.random.rand() > 0.5
            do_shift = True
        if do_shift:
            # Spatial augmentation
            rng = self.augment_rng

            vidid = target_['video_id']
            try:
                video = self.sampler.dset.index.videos[vidid]
            except KeyError:
                # Hack for loose images
                assert len(target_['gids']) == 1
                gid = target_['gids'][0]
                video = self.sampler.dset.index.imgs[gid]
            vid_width = video['width']
            vid_height = video['height']

            # Spatial augmentation:
            if augment_space_shift_rate and rng.rand() < augment_space_shift_rate:
                space_box = kwimage.Boxes.from_slice(
                    target_['space_slice'], clip=False,
                    endpoint=True)
                w = space_box.width.ravel()[0]
                h = space_box.height.ravel()[0]
                # hack: this prevents us from assuming there is a target in the
                # window, but it lets us get the benefit of chip_overlap=0.5 while
                # still having it at 0 for faster epochs.

                rand_w = rng.randint(-w // 2.7, w // 2.7)
                rand_h = rng.randint(-h // 2.7, h // 2.7)
                if 0:
                    """
                    Benchmark:
                        # Shows that it is about 5x better to use regular translate
                        # over an affine warp, with inplace being even better.
                        import kwimage
                        import kwarray
                        box = kwimage.Box.random()
                        import timerit
                        ti = timerit.Timerit(100, bestof=10, verbose=2)
                        rng = kwarray.ensure_rng()

                        w = box.width.ravel()[0]
                        h = box.height.ravel()[0]

                        rand_w = rng.randint(-w // 2.7, w // 2.7)
                        rand_h = rng.randint(-h // 2.7, h // 2.7)
                        aff = kwimage.Affine.coerce(offset=(rand_w, rand_h))

                        for timer in ti.reset('affine translate'):
                            with timer:
                                box.warp(aff)

                        for timer in ti.reset('direct translate'):
                            with timer:
                                box.translate((rand_w, rand_h))

                        for timer in ti.reset('direct translate'):
                            with timer:
                                box.translate((rand_w, rand_h), inplace=True)
                    """
                    aff = kwimage.Affine.coerce(offset=(rand_w, rand_h))
                    space_box = space_box.warp(aff)
                else:
                    # Faster than the above code
                    space_box = space_box.translate((rand_w, rand_h), inplace=True)

                space_box = space_box.quantize(inplace=True)
                # Keep the original box size
                space_box = space_box.resize(width=w, height=h, inplace=0)

                # prevent shifting the target off the edge of the video
                snap_target = kwimage.Boxes(np.array(
                    [[0, 0, vid_width, vid_height]]), 'ltrb')
                space_box = data_utils._boxes_snap_to_edges(
                    space_box, snap_target, inplace=True)

                target_['space_slice'] = space_box.astype(int).to_slices()[0]

            gids = target_['gids']
            if len(gids) > 1:

                # Temporal augmentation
                if augment_time_resample_rate and rng.rand() < augment_time_resample_rate:
                    self._augment_target_time(target_)

                # Temporal dropout
                temporal_dropout_rate = self.config['temporal_dropout_rate']
                frame_dropout_thresh = self.config['temporal_dropout']
                do_temporal_dropout = rng.rand() < temporal_dropout_rate
                if do_temporal_dropout and frame_dropout_thresh > 0:
                    main_gid = target_['main_gid']
                    main_frame_idx = gids.index(main_gid)
                    keep_score = rng.rand(len(gids))
                    keep_score[main_frame_idx] = 1.0
                    keep_flags = keep_score >= frame_dropout_thresh
                    gids = list(ub.compress(gids, keep_flags))
                    # target_['main_idx'] = gids.index(main_gid)
                    target_['gids'] = gids

        # force_flip = target_.get('flip_axis', None)
        # See data_utils.fliprot_annot for a visualization of various
        # flip/rotations
        unique_fliprots = [
            {'rot_k': 0, 'flip_axis': None},  # nothing
        ]

        if augment_space_rot:
            unique_fliprots += [
                {'rot_k': 1, 'flip_axis': None},  # ccw rotation
                {'rot_k': 3, 'flip_axis': None},  # cw rotation
            ]
            if augment_space_xflip:
                unique_fliprots += [
                    {'rot_k': 1, 'flip_axis': (0,)},  # ccw rotation + xflip
                    {'rot_k': 3, 'flip_axis': (0,)},  # cw rotation x-filp
                ]
        if augment_space_yflip:
            unique_fliprots += [
                {'rot_k': 0, 'flip_axis': (0,)},  # y-flip
            ]
            if augment_space_xflip:
                unique_fliprots += [
                    {'rot_k': 2, 'flip_axis': None},  # y-flip + x-flip
                ]
        if augment_space_xflip:
            unique_fliprots += [
                {'rot_k': 2, 'flip_axis': (0,)},  # x-flip
            ]

        # Force an augmentation
        FLIP_AUGMENTATION = (not self.disable_augmenter and self.mode == 'fit')
        if FLIP_AUGMENTATION:
            # Choose a unique flip/rot
            fliprot_idx = rng.randint(0, len(unique_fliprots))
            fliprot_params = unique_fliprots[fliprot_idx]
            target_['fliprot_params'] = fliprot_params

        return target_
