def add_to_track(ann, gen, prev_ann=None):
    if 'track_id' not in ann:
        if prev_ann is None or 'track_id' not in prev_ann:
            ann['track_id'] = next(gen)
            ann['track_index'] = 0
        else:
            ann['track_id'] = prev_ann['track_id']
            ann['track_index'] = prev_ann['track_index'] + 1

    return ann


def mono(video):
        # HACK for mono-site
    if coerce_site_boundary:
        for gid in coco_dset.imgs:
            annots = coco_dset.annots(gid=gid)
            if len(annots) == 0:
                continue

            template_ann = annots.peek()

            # print(list(np.unique(annots.lookup('category_id'))), [coco_dset.name_to_cat['change']['id']])
            assert list(np.unique(annots.lookup('category_id'))) == [coco_dset.name_to_cat['change']['id']]
            try:
                sseg_geos = [kwimage.MultiPolygon.from_shapely(
                    shapely.ops.unary_union([
                        kwimage.MultiPolygon.from_geojson(seg_geo).to_shapely().buffer(0)
                        for seg_geo in (annots.lookup('segmentation_geos'))])).to_geojson()]
            except TypeError:
                xdev.embed()

            template_ann.pop('segmentation', None)
            template_ann.pop('bbox', None)
            template_ann['score'] == np.mean(annots.lookup('score'))
            template_ann['segmentation_geos'] = sseg_geos

            coco_dset.remove_annotations(annots.aids[1:])


def from_overlap(ann, vid_id, coco_dset, phase, gen):
    """
    For each annotation, look forward in time and find the closest overlapping
    annotation that has a different construction phase. Add them to the same track.
    """
    def _shp(seg_geo):
        # xdev.embed()
        if isinstance(seg_geo, list):
            seg_geo = seg_geo[0]
        return kwimage.MultiPolygon.from_geojson(seg_geo).to_shapely().buffer(
            0)
    # Default prediction if one cannot be found
    prediction = {
        'predicted_phase': None,
        'predicted_phase_start_date': None,
    }
    ann = add_to_track(ann, gen)

    # If we're part of an existing track and not the last entry in it
    # use the next entry
    if 'track_id' in ann:
        track = coco_dset.annots(trackid=ann['track_id'])
        aids = list(track.aids)
        ix = aids.index(ann['id'])
        try:
            ann_obs = coco_dset.anns[aids[ix+1]]

            cat = coco_dset.cats[ann_obs['category_id']]
            predict_phase = category_dict.get(cat['name'], cat['name'])
            obs_img = coco_dset.index.imgs[ann_obs['image_id']]
            date = dateutil.parser.parse(obs_img['date_captured']).date()

            prediction = {
                'predicted_phase': predict_phase,
                'predicted_phase_start_date': date.isoformat(),
            }
            return prediction, coco_dset
        except IndexError:
            pass

    # Else, search the future for an overlapping untracked ann
    # and add it to the track
    if phase != 'Post Construction':
        img_id = ann['image_id']
        min_overlap = 0

        # Find all images that come after this one
        video_gids = coco_dset.index.vidid_to_gids[vid_id]
        img_index = video_gids.index(img_id)
        future_gids = video_gids[img_index:]
        cand_aids = []
        for frame_gid in future_gids:
            cand_aids.extend(coco_dset.index.gid_to_aids[frame_gid])
        
        # TODO check this
        union_poly_ann = _shp(ann['segmentation_geos'])
        for cand_aid in cand_aids:

            ann_obs = coco_dset.anns[cand_aid]

            # skip anns that are already part of a track
            if 'track_id' in ann_obs:
                continue

            cat = coco_dset.cats[ann_obs['category_id']]
            predict_phase = category_dict.get(cat['name'], cat['name'])
            # HACK for change-only preds
            if (phase != predict_phase) or predict_phase == 'change':
                # TODO check this
                union_poly_obs = _shp(ann_obs['segmentation_geos'])
                intersect = union_poly_obs.intersection(union_poly_ann).area
                if intersect == 0:
                    continue
                overlap = intersect / union_poly_ann.area
                if overlap > min_overlap:
                    obs_img = coco_dset.index.imgs[ann_obs['image_id']]
                    date = dateutil.parser.parse(obs_img['date_captured']).date()
                    # We found a valid prediction
                    prediction = {
                        'predicted_phase': predict_phase,
                        'predicted_phase_start_date': date.isoformat(),
                    }
                    ann_obs = add_to_track(ann_obs, gen, ann)
                    break
    return prediction, coco_dset

