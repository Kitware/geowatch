import ubelt as ub

try:
    from xdev import profile
except Exception:
    profile = ub.identity


class TrackFunction:
    """
    Abstract class that all track functions should inherit from.
    """

    def __call__(self, sub_dset):
        """
        Ensure each annotation in coco_dset has a track_id.

        Returns:
            kwcoco.CocoDataset
        """
        raise NotImplementedError('must be implemented by subclasses')

    def apply_per_video(self, coco_dset, overwrite=False):
        """
        Main entrypoint for this class.
        """
        import kwcoco
        legacy = False

        assert not overwrite, 'overwrite should always be false'

        tracked_subdsets = []
        vid_gids = coco_dset.index.vidid_to_gids.values()
        total = len(coco_dset.index.vidid_to_gids)
        for gids in ub.ProgIter(vid_gids,
                                total=total,
                                desc='apply_per_video',
                                verbose=3):

            # Beware, in the past there was a crash here that required
            # wrapping the rest of this loop in a try/except. -csg
            sub_dset = self.safe_apply(coco_dset,
                                       gids,
                                       overwrite,
                                       legacy=legacy)
            if legacy:
                coco_dset = sub_dset
            else:
                tracked_subdsets.append(sub_dset)

        if not legacy:
            # Tracks were either updated or added.
            # In the case they were updated the existing track ids should
            # be disjoint. All new tracks should not overlap with

            _debug = 0

            from watch.utils import kwcoco_extensions
            new_trackids = kwcoco_extensions.TrackidGenerator(None)
            fixed_subdataset = []
            for sub_dset in ub.ProgIter(tracked_subdsets,
                                        desc='Ensure ok tracks',
                                        verbose=3):

                if _debug:
                    sub_dset = sub_dset.copy()

                # Rebuild the index to ensure any hacks are removed.
                # We should be able to remove this step.
                # sub_dset._build_index()

                sub_annots = sub_dset.annots()
                sub_tids = sub_annots.lookup('track_id')
                existing_tids = set(sub_tids)

                collisions = existing_tids & new_trackids.used_trackids
                if _debug:
                    print('existing_tids = {!r}'.format(existing_tids))
                    print('collisions = {!r}'.format(collisions))

                new_trackids.exclude_trackids(existing_tids)
                if collisions:
                    old_tid_to_aids = ub.group_items(sub_annots, sub_tids)
                    assert len(old_tid_to_aids) == len(existing_tids)
                    print(f'Resolve {len(collisions)} track-id collisions')
                    # Change the track ids of any collisions
                    for old_tid in collisions:
                        new_tid = next(new_trackids)
                        # Note: this does not update the index, but we
                        # are about to clobber it anyway, so it doesnt matter
                        for aid in old_tid_to_aids[old_tid]:
                            ann = sub_dset.index.anns[aid]
                            ann['track_id'] = new_tid
                        existing_tids.add(new_tid)
                new_trackids.exclude_trackids(existing_tids)

                if _debug:
                    after_tids = set(sub_annots.lookup('track_id'))
                    print('collisions = {!r}'.format(collisions))
                    print(f'{after_tids=}')

                fixed_subdataset.append(sub_dset)

            # Is this safe to do? It would be more efficient
            coco_dset = kwcoco.CocoDataset.union(*fixed_subdataset,
                                                 disjoint_tracks=False)

            if _debug:
                x = coco_dset.annots().images.get('video_id')
                y = coco_dset.annots().get('track_id')
                z = ub.group_items(x, y)
                track_to_num_videos = ub.map_vals(set, z)
                if track_to_num_videos:
                    assert max(map(len, track_to_num_videos.values())) == 1, (
                        'track belongs to multiple videos!')
        return coco_dset

    @profile
    def safe_apply(self, coco_dset, gids, overwrite, legacy=True):
        assert not legacy, 'todo: remove legacy code'

        import numpy as np
        DEBUG_JSON_SERIALIZABLE = 0
        if DEBUG_JSON_SERIALIZABLE:
            from watch.utils.util_json import debug_json_unserializable

        if DEBUG_JSON_SERIALIZABLE:
            debug_json_unserializable(coco_dset.dataset,
                                      'Input to safe_apply: ')

        if legacy:
            sub_dset, rest_dset = self.safe_partition(coco_dset,
                                                      gids,
                                                      remove=True)
        else:
            sub_dset = self.safe_partition(coco_dset, gids, remove=False)

        if DEBUG_JSON_SERIALIZABLE:
            debug_json_unserializable(sub_dset.dataset, 'Before __call__')

        if overwrite:
            raise AssertionError('overwrite should always be False')

            sub_dset = self(sub_dset)
            if DEBUG_JSON_SERIALIZABLE:
                debug_json_unserializable(sub_dset.dataset,
                                          'After __call__ (overwrite)')
        else:
            orig_annots = sub_dset.annots()
            orig_tids = orig_annots.get('track_id', None)
            orig_trackless_flags = np.array([tid is None for tid in orig_tids])
            orig_aids = list(orig_annots)

            # TODO more sophisticated way to check if we can skip self()

            ####
            # APPLY THE TRACKING FUNCTION.
            # THIS IS THE MAIN WORK. SEE SPECIFIC __call__ FUNCTIOSN
            sub_dset = self(sub_dset)
            ####

            if DEBUG_JSON_SERIALIZABLE:
                debug_json_unserializable(sub_dset.dataset, 'After __call__')

            # if new annots were not created, rollover the old tracks
            new_annots = sub_dset.annots()
            if new_annots.aids == orig_aids:
                new_tids = new_annots.get('track_id', None)
                # Only overwrite track ids for annots that didn't have them
                new_tids = np.where(orig_trackless_flags, new_tids, orig_tids)

                # Ensure types are json serializable
                import numbers

                def _fixtype(tid):
                    # need to keep strings the same, but integers need to be
                    # case from numpy to python ints.
                    if isinstance(tid, numbers.Integral):
                        return int(tid)
                    else:
                        return tid

                new_tids = list(map(_fixtype, new_tids))

                new_annots.set('track_id', new_tids)

        # TODO: why is this assert here?
        assert None not in sub_dset.annots().lookup('track_id', None)

        if legacy:
            out_dset = self.safe_union(rest_dset, sub_dset)
        else:
            out_dset = sub_dset

        if DEBUG_JSON_SERIALIZABLE:
            debug_json_unserializable(out_dset.dataset,
                                      'Output of safe_apply: ')
        return out_dset

    @staticmethod
    @profile
    def safe_partition(coco_dset, gids, remove=True):

        assert not remove, 'should never remove'

        sub_dset = coco_dset.subset(gids=gids, copy=True)
        # HACK ensure tracks are not duplicated between videos
        # (if they are, this is fixed in dedupe_tracks anyway)
        sub_dset.index.trackid_to_aids.update(coco_dset.index.trackid_to_aids)
        if remove:
            rest_gids = list(set(coco_dset.imgs.keys()) - set(gids))
            rest_dset = coco_dset.subset(rest_gids)
            return sub_dset, rest_dset
        else:
            return sub_dset

    @staticmethod
    @profile
    def safe_union(coco_dset, new_dset, existing_aids=[]):
        raise AssertionError('scheduled for removal')
        coco_dset._build_index()
        new_dset._build_index()
        # we handle tracks in normalize.dedupe_tracks anyway, and
        # disjoint_tracks=True interferes with keeping site_ids around as
        # track_ids.
        # return coco_dset.union(new_dset, disjoint_tracks=True)
        return coco_dset.union(new_dset, disjoint_tracks=False)


class NoOpTrackFunction(TrackFunction):
    """
    Use existing tracks.
    """

    def __init__(self, **kwargs):
        self.kwargs = kwargs  # Unused

    def __call__(self, sub_dset):
        return sub_dset


class NewTrackFunction(TrackFunction):
    """
    Specialization of TrackFunction to create polygons that do not yet exist
    in coco_dset, and add them as new annotations
    """

    def __call__(self, sub_dset):
        # print(f'Enter {self.__class__} __call__ function')
        # print('Create tracks')
        tracks = self.create_tracks(sub_dset)
        # print('Add tracks to dset')
        sub_dset = self.add_tracks_to_dset(sub_dset, tracks)
        # print('After tracking sub_dset.stats(): ' +
        #       ub.urepr(sub_dset.basic_stats()))
        # print(f'Exit {self.__class__} __call__ function')
        return sub_dset

    def create_tracks(self, sub_dset):
        """
        Args:
            sub_dset (CocoDataset):

        Returns:
            GeoDataFrame
        """
        raise NotImplementedError('must be implemented by subclasses')

    def add_tracks_to_dset(self, sub_dset, tracks):
        """
        Args:
            tracks (GeoDataFrame):

        Returns:
            kwcoco.CocoDataset
        """
        raise NotImplementedError('must be implemented by subclasses')
