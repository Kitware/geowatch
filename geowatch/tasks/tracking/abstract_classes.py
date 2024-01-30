import ubelt as ub

try:
    from xdev import profile
except Exception:
    profile = ub.identity


class TrackFunction:
    """
    Abstract class that all track functions should inherit from.
    """

    def __call__(self, sub_dset, video_id, **kwargs):
        # The original impl with __call__ methods made it difficult for me to
        # grep for things, so I'm disabling it.
        raise AssertionError("Use the explicit .forward method instead")
        return self.forward(sub_dset, video_id,  **kwargs)

    def forward(self, sub_dset, video_id):
        """
        Detect new annotations and track them in images for a single video.

        Args:
            sub_dset (CocoDataset):
            video_id (int): video to track

        Returns:
            kwcoco.CocoDataset
        """
        raise NotImplementedError('must be implemented by subclasses')

    def apply_per_video(self, coco_dset):
        """
        Main entrypoint for this class.

        Calls :func:`safe_apply` on each video in the coco dataset.

        Args:
            coco_dset (kwcoco.CocoDataset):
                the dataset to run tracking on

        Returns:
            kwcoco.CocoDataset:
                A modified or copied version of the input with new annotations
                and tracks.

        Example:
            >>> from geowatch.tasks.tracking.abstract_classes import *  # NOQA
            >>> import kwcoco
            >>> self = NoOpTrackFunction()
            >>> coco_dset = kwcoco.CocoDataset.coerce('vidshapes8')
            >>> self.apply_per_video(coco_dset)
        """
        import kwcoco
        from geowatch.utils import kwcoco_extensions

        # Note (2024-01-24): previously this was implemented in a way that
        # broke up a larger coco dataset into one for each video, but we have
        # modified the implementation to avoid the "subset" and "union"
        # overhead.

        # If there are downstream issues set to True for old behavior
        # otherwise, when stable the else branch should be deleted.
        DO_SUBSET_HACK = False  # True for old behavior

        tracking_results = []
        vid_gids = list(coco_dset.index.vidid_to_gids.items())
        total = len(coco_dset.index.vidid_to_gids)
        for video_id, gids in ub.ProgIter(vid_gids,
                                          total=total,
                                          desc='apply_per_video',
                                          verbose=3):

            # Beware, in the past there was a crash here that required
            # wrapping the rest of this loop in a try/except. -csg
            sub_dset = self.safe_apply(coco_dset, video_id, gids,
                                       DO_SUBSET_HACK=DO_SUBSET_HACK)

            # Store a reference to the dataset (which may be a modified subset,
            # or an unmodified reference with the image ids that were tracked)
            tracking_results.append({
                'sub_dset': sub_dset,
                'video_id': video_id,
                'image_ids': gids,
            })

        # Tracks were either updated or added.
        # In the case they were updated the existing track ids should
        # be disjoint. All new tracks should not overlap with

        _debug = 1

        new_trackids = kwcoco_extensions.TrackidGenerator(None)
        for tracking_result in ub.ProgIter(tracking_results,
                                           desc='Ensure ok tracks',
                                           verbose=3):

            sub_dset = tracking_result['sub_dset']
            sub_gids = tracking_result['image_ids']

            # Rebuild the index to ensure any hacks are removed.
            # We should be able to remove this step.
            # sub_dset._build_index()

            sub_aids = list(ub.flatten(sub_dset.images(sub_gids).annots))
            sub_annots = sub_dset.annots(sub_aids)
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

        if DO_SUBSET_HACK:
            # If we broke up the dataset into subsets, we need to combine them
            # back together.
            fixed_subdataset = [r['sub_dset'] for r in tracking_results]
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
    def safe_apply(self, coco_dset, video_id, gids, DO_SUBSET_HACK=False):
        import numpy as np
        DEBUG_JSON_SERIALIZABLE = 0
        if DEBUG_JSON_SERIALIZABLE:
            from kwutil.util_json import debug_json_unserializable

        if DEBUG_JSON_SERIALIZABLE:
            debug_json_unserializable(coco_dset.dataset,
                                      'Input to safe_apply: ')

        if DO_SUBSET_HACK:
            # Try not to do this if we can avoid it.
            sub_dset = self.safe_partition(coco_dset, gids)
        else:
            # Simulate a single-video dataset (and maintain relevant caches)
            sub_dset = coco_dset
            # TODO: holding context to revent re-looking up annotations for
            # each video-id will likely improve performance.
            # sub_dset = WrappedSingleVideoCocoDataset(coco_dset, video_id, gids)

        if DEBUG_JSON_SERIALIZABLE:
            debug_json_unserializable(sub_dset.dataset, 'Before __call__')

        orig_aids = list(ub.flatten(sub_dset.images(gids).annots))
        orig_annots = sub_dset.annots(orig_aids)
        # orig_annots = sub_dset.annots()
        orig_tids = orig_annots.get('track_id', None)
        orig_trackless_flags = np.array([tid is None for tid in orig_tids])
        orig_aids = list(orig_annots)

        ####
        # APPLY THE TRACKING FUNCTION.
        # THIS IS THE MAIN WORK. SEE SPECIFIC forward FUNCTIONS
        sub_dset = self.forward(sub_dset, video_id)
        ####

        if DEBUG_JSON_SERIALIZABLE:
            debug_json_unserializable(sub_dset.dataset, 'After __call__')

        # if new annots were not created, rollover the old tracks
        new_aids = list(ub.flatten(sub_dset.images(gids).annots))
        new_annots = sub_dset.annots(new_aids)
        if new_annots.aids == orig_aids:
            new_tids = new_annots.get('track_id', None)
            # Only overwrite track ids for annots that didn't have them
            new_tids = np.where(orig_trackless_flags, new_tids, orig_tids)

            # Ensure types are json serializable
            import numbers

            def _fixtype(tid):
                # need to keep strings the same, but integers need to be
                # cast from numpy to python ints.
                if isinstance(tid, numbers.Integral):
                    return int(tid)
                else:
                    return tid

            new_tids = list(map(_fixtype, new_tids))

            new_annots.set('track_id', new_tids)

        # TODO: why is this assert here?
        sub_track_ids = new_annots.lookup('track_id', None)
        if None in sub_track_ids:
            raise AssertionError(f'None in track ids: {sub_track_ids}')

        out_dset = sub_dset

        if DEBUG_JSON_SERIALIZABLE:
            debug_json_unserializable(out_dset.dataset,
                                      'Output of safe_apply: ')
        return out_dset

    @staticmethod
    @profile
    def safe_partition(coco_dset, gids):

        sub_dset = coco_dset.subset(gids=gids, copy=True)
        # HACK ensure tracks are not duplicated between videos
        # (if they are, this is fixed in dedupe_tracks anyway)
        sub_dset.index.trackid_to_aids.update(coco_dset.index.trackid_to_aids)
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

    def forward(self, sub_dset, video_id):
        return sub_dset


class NewTrackFunction(TrackFunction):
    """
    Specialization of TrackFunction to create polygons that do not yet exist
    in coco_dset, and add them as new annotations
    """

    def forward(self, sub_dset, video_id):
        # print(f'Enter {self.__class__} __call__ function')
        # print('Create tracks')
        tracks = self.create_tracks(sub_dset, video_id)
        # print('Add tracks to dset')
        sub_dset = self.add_tracks_to_dset(sub_dset, tracks)
        # print('After tracking sub_dset.stats(): ' +
        #       ub.urepr(sub_dset.basic_stats()))
        # print(f'Exit {self.__class__} __call__ function')
        return sub_dset

    def create_tracks(self, sub_dset, video_id):
        """
        Args:
            sub_dset (CocoDataset):
            video_id (int): video to create tracks for

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


class WrappedSingleVideoCocoDataset:
    def __init__(self, dset, video_id, gids):
        self.dset = dset
        self.video_id = video_id
        self.gids = gids
