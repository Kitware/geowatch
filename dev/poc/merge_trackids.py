

def union_with_merge_trackids(dset1, dset2):
    """
    Example:
        >>> import kwcoco
        >>> dset1 = kwcoco.CocoDataset.demo('vidshapes8')
        >>> dset2 = kwcoco.CocoDataset.demo('vidshapes8')
        >>> new_dset = union_with_merge_trackids(dset1, dset2)
    """
    import ubelt as ub
    import kwimage

    # At some point we will need to make new track ids, ensure we never reuse
    # anything that exists.
    from watch.utils import kwcoco_extensions
    trackid_gen = kwcoco_extensions.TrackidGenerator()
    trackid_gen.exclude_trackids(dset1.index.trackid_to_aids.keys())
    trackid_gen.exclude_trackids(dset2.index.trackid_to_aids.keys())

    print(f'Merging datasets {dset1=} {dset2=}')

    # Need to have some mechanism for figuring out which videos are the same in
    # the two datasets. Let's assume the names will give us that.
    video_names1 = set(dset1.index.name_to_video.keys())
    video_names2 = set(dset2.index.name_to_video.keys())
    common_video_names = video_names1 & video_names2
    unmatched_videos1 = video_names1 - video_names2
    unmatched_videos2 = video_names2 - video_names1
    print(ub.codeblock(
        f'''
        Checking for assignment between videos:
          * Found {len(common_video_names)} videos with the same name.
          * Found {len(unmatched_videos1)} unmatched videos in dset1
          * Found {len(unmatched_videos2)} unmatched videos in dset2
        '''))

    # For each video we need to change their track-ids to link tracks we want
    # to match and separate tracks we dont want to match

    trackid_mapping1 = {}
    trackid_mapping2 = {}

    # For sanity initialize the default mapping so everything in dset1
    # keeps the same trackid id and everything in dset2 gets a new trackid.
    trackid_mapping1.update({
        tid: tid for tid in dset1.index.trackid_to_aids.keys()
    })
    trackid_mapping2.update({
        tid: next(trackid_gen) for tid in dset2.index.trackid_to_aids.keys()
    })

    for video_name in common_video_names:
        video1 = dset1.index.name_to_video[video_name]
        video2 = dset2.index.name_to_video[video_name]

        video_id1 = video1['id']
        video_id2 = video2['id']

        # Get the (ordered) sequence of image ids for each frame in the videos
        images1 = dset1.images(video_id=video_id1)
        images2 = dset2.images(video_id=video_id2)

        # Assume video1 is temporally before video2 (todo: we could
        # programtically check this via timestamps to determine ordering)

        vid1_trackid_to_last_aid = {}
        # For video1, find the last frame each track occurs on.
        for annots in images1.annots:
            tids = annots.lookup('track_id')
            aids = annots.aids
            for tid, aid in zip(tids, aids):
                vid1_trackid_to_last_aid[tid] = aid

        # For video1, find the first frame each track occurs on.
        vid2_trackid_to_first_aid = {}
        for annots in images2.annots[::-1]:
            tids = annots.lookup('track_id')
            aids = annots.aids
            for tid, aid in zip(tids, aids):
                vid2_trackid_to_first_aid[tid] = aid

        # Now this is a tricky part, because we need to be sure the annotations
        # are in a comparable space before we test overlaps. Let's assume that
        # the videos have exactly the same coordinates for now and compare
        # annotations in video space.
        vid1_trackid_to_poly = {}
        vid2_trackid_to_poly = {}

        for tid, aid in vid1_trackid_to_last_aid.items():
            gid = dset1.index.anns[aid]['image_id']
            coco_img = dset1.coco_image(gid)
            sseg = dset1.annots([aid]).lookup('segmentation')[0]
            img_poly1 = kwimage.MultiPolygon.coerce(sseg)
            vid_poly1 = img_poly1.warp(coco_img.warp_vid_from_img)
            poly1 = vid_poly1.to_shapely()
            vid1_trackid_to_poly[tid] = poly1

        for tid, aid in vid2_trackid_to_first_aid.items():
            gid = dset2.index.anns[aid]['image_id']
            coco_img = dset2.coco_image(gid)
            sseg = dset2.annots([aid]).lookup('segmentation')[0]
            img_poly2 = kwimage.MultiPolygon.coerce(sseg)
            vid_poly2 = img_poly2.warp(coco_img.warp_vid_from_img)
            poly2 = vid_poly2.to_shapely()
            vid2_trackid_to_poly[tid] = poly2

        # Now we have a polygon representing each track in a comparable space,
        # let's push them into a geopandas object to reuse some existing code
        import geopandas as gpd
        vid1_geoms = gpd.GeoDataFrame([
            {'tid': tid, 'geometry': geom} for tid, geom in vid1_trackid_to_poly.items()
        ])
        vid2_geoms = gpd.GeoDataFrame([
            {'tid': tid, 'geometry': geom} for tid, geom in vid2_trackid_to_poly.items()
        ])

        # Find candidate matches between tracks
        from watch.utils import util_gis
        idx1_to_idxs2 = util_gis.geopandas_pairwise_overlaps(vid1_geoms, vid2_geoms, predicate='intersects')

        # TODO: probably have an assignemnt algo we can reuse somwehere
        # For now this greedy approach is fine.
        used = set()
        assigned_tid1s = []
        assigned_tid2s = []
        for idx1, idx2_cands in idx1_to_idxs2.items():
            tid1 = vid1_geoms.iloc[idx1]['tid']
            geom1 = vid1_geoms.iloc[idx1]['geometry']
            geom2_cands = vid2_geoms.iloc[idx2_cands]['geometry']
            geom2_tids = vid2_geoms.iloc[idx2_cands]['tid']

            curr_score = None
            curr_tid2 = None
            for geom2, tid2 in zip(geom2_cands, geom2_tids):
                if tid2 not in used:
                    # intersection overmin area (might want to change)
                    ioma = geom1.intersection(geom2).area / min(geom1.area, geom2.area)
                    if curr_score is None or curr_score >= ioma:
                        curr_score = ioma
                        curr_tid2 = tid2
            if curr_tid2 is not None:
                used.add(curr_tid2)
                assigned_tid1s.append(tid1)
                assigned_tid2s.append(tid2)

        unassigned_tid1s = set(vid1_trackid_to_last_aid) - set(assigned_tid1s)
        unassigned_tid2s = set(vid2_trackid_to_first_aid) - set(assigned_tid2s)

        # We now create mapping from old trackids to new trackids that can be
        # unioned normally with disjoint_tracks=False.
        # Allow dataset1 to keep it's track ids
        for tid1 in unassigned_tid1s:
            trackid_mapping1[tid1] = tid1

        # For dataset1 force anything unassigned into a new trackid
        for tid1 in unassigned_tid2s:
            trackid_mapping2[tid2] = next(trackid_gen)

        # For matched trackids, use the dataset1 id
        for tid1, tid2 in zip(unassigned_tid1s, unassigned_tid2s):
            # keep the same id for dataset1
            trackid_mapping1[tid1] = tid1
            # use the new id for dataset2
            trackid_mapping2[tid2] = tid1

    # Now that we have gone over every video we can remap the ids and do a
    # union.

    # It would be nice if kwcoco had a "remap trackid" function
    def remap_trackids(dset, trackid_mapping):
        # For every annotation, change its trackid
        for ann in dset.index.anns.values():
            old_tid = ann['track_id']
            ann['track_id'] = trackid_mapping[old_tid]
        # above code does not udpate the index nicely, need to rebuild it
        dset._build_index()

    remap_trackids(dset1, trackid_mapping1)
    remap_trackids(dset2, trackid_mapping2)

    new_dset = dset1.union(dset2, disjoint_tracks=False)
    return new_dset
