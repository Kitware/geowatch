"""
Data augmentation utilities
"""
import kwarray
import kwimage
import ubelt as ub

from watch.tasks.fusion.datamodules import data_utils


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

    def _augment_target_time(self, target_):
        """
        Jitters the time sample in a target
        """
        vidid = target_['video_id']
        time_sampler = self.new_sample_grid['vidid_to_time_sampler'][vidid]
        valid_gids = self.new_sample_grid['vidid_to_valid_gids'][vidid]
        new_idxs = time_sampler.sample(target_['main_idx'])
        new_gids = list(ub.take(valid_gids, new_idxs))
        target_['gids'] = new_gids
        return target_

    def _augment_spacetime_target(self, target_):
        """
        Given a target dictionary, shift around the space and time slice

        Ignore:
            >>> from watch.tasks.fusion.datamodules.kwcoco_dataset import *  # NOQA
            >>> import ndsampler
            >>> import kwcoco
            >>> import watch
            >>> coco_dset = watch.demo.coerce_kwcoco('watch')
            >>> sampler = ndsampler.CocoSampler(coco_dset)
            >>> sample_shape = (2, 128, 128)
            >>> self = KWCocoVideoDataset(sampler, sample_shape=sample_shape)
            >>> index = 0
            >>> target = self.new_sample_grid['targets'][index]
            >>> target_ = target.copy()
            >>> target_ = self._augment_spacetime_target(target_)
            >>> print('target  = {!r}'.format(target))
            >>> print('target_ = {!r}'.format(target_))
        """

        # TODO: make a nice "augmenter" pipeline
        # TODO: parameteraize
        temporal_augment_rate = 0.8
        spatial_augment_rate = 0.9

        do_shift = False
        if not self.disable_augmenter and self.mode == 'fit':
            # do_shift = np.random.rand() > 0.5
            do_shift = True
        if do_shift:
            # Spatial augmentation
            rng = kwarray.ensure_rng(None)

            vidid = target_['video_id']
            video = self.sampler.dset.index.videos[vidid]
            vid_width = video['width']
            vid_height = video['height']

            # Spatial augmentation:
            if rng.rand() < spatial_augment_rate:
                space_box = kwimage.Boxes.from_slice(
                    target_['space_slice'], clip=False,
                    endpoint=True)
                w = space_box.width.ravel()[0]
                h = space_box.height.ravel()[0]
                # hack: this prevents us from assuming there is a target in the
                # window, but it lets us get the benefit of chip_overlap=0.5 while
                # still having it at 0 for faster epochs.
                aff = kwimage.Affine.coerce(offset=(
                    rng.randint(-w // 2.7, w // 2.7),
                    rng.randint(-h // 2.7, h // 2.7)))
                space_box = space_box.warp(aff).quantize()
                # Keep the original box size
                space_box = space_box.resize(width=w, height=h)

                # prevent shifting the target off the edge of the video
                snap_target = kwimage.Boxes([[0, 0, vid_width, vid_height]], 'ltrb')
                space_box = data_utils._boxes_snap_to_edges(space_box, snap_target)

                target_['space_slice'] = space_box.astype(int).to_slices()[0]

            # Temporal augmentation
            if rng.rand() < temporal_augment_rate:
                self._augment_target_time(target_)

            temporal_dropout_rate = self.temporal_dropout
            do_temporal_dropout = rng.rand() < temporal_dropout_rate
            if do_temporal_dropout:
                # Temporal dropout
                gids = target_['gids']
                main_gid = target_['main_gid']
                main_frame_idx = gids.index(main_gid)
                flags = rng.rand(len(gids)) > 0.5
                flags[main_frame_idx] = True
                flags[0] = True
                flags[-1] = True
                gids = list(ub.compress(gids, flags))
                # target_['main_idx'] = gids.index(main_gid)
                target_['gids'] = gids

        # force_flip = target_.get('flip_axis', None)
        unique_fliprots = [
            {'rot_k': 0, 'flip_axis': None},
            {'rot_k': 1, 'flip_axis': None},
            {'rot_k': 2, 'flip_axis': None},
            {'rot_k': 3, 'flip_axis': None},
            {'rot_k': 0, 'flip_axis': (0,)},
            {'rot_k': 1, 'flip_axis': (0,)},
            {'rot_k': 2, 'flip_axis': (0,)},
            {'rot_k': 3, 'flip_axis': (0,)},
        ]

        # Force an augmentation
        FLIP_AUGMENTATION = (not self.disable_augmenter and self.mode == 'fit')
        if FLIP_AUGMENTATION:
            # Choose a unique flip/rot
            fliprot_idx = rng.randint(0, len(unique_fliprots))
            fliprot_params = unique_fliprots[fliprot_idx]
            target_['fliprot_params'] = fliprot_params

        return target_
