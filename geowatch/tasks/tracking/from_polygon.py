from geowatch.tasks.tracking.abstract_classes import TrackFunction
import scriptconfig as scfg
import ubelt as ub


class MonoTrack(TrackFunction):
    '''
    Combine all polygons into the same track.
    '''

    def __init__(self, **kwargs):
        self.kwargs = kwargs  # Unused

    def forward(self, coco_dset, video_id):
        from geowatch.utils.kwcoco_extensions import TrackidGenerator
        aids = list(ub.flatten(coco_dset.images(video_id=video_id).annots))
        annots = coco_dset.annots(aids)

        annots.set('track_id', next(TrackidGenerator(coco_dset)))
        return coco_dset


def as_shapely_polys(annots):
    return map(lambda poly: poly.to_shapely().buffer(0),
               annots.detections.data['segmentations'].to_polygon_list())


class OverlapTrack(scfg.DataConfig, TrackFunction):
    '''
    Put polygons in the same track if their areas overlap.
    '''
    min_overlap: float = 0

    def forward(self, coco_dset, video_id):
        from geowatch.utils.kwcoco_extensions import TrackidGenerator
        new_trackids = TrackidGenerator(coco_dset)

        aids = list(ub.flatten(coco_dset.images(video_id=video_id).annots))
        annots = coco_dset.annots(aids)

        aid_to_poly = dict(zip(annots.aids, as_shapely_polys(annots)))

        def _search(aid, aid_groups):
            poly1 = aid_to_poly[aid]

            def _search_group(aids):
                for aid2 in aids:
                    if 'track_id' not in coco_dset.anns[aid2]:
                        poly2 = aid_to_poly[aid2]
                        # check overlap
                        if poly1.intersects(poly2):
                            if (poly1.intersection(poly2).area /
                                    poly2.area) > self.min_overlap:
                                return aid2

            try:
                return next(filter(None, map(_search_group, aid_groups)))
            except StopIteration:
                return None

        # update tracks one frame at a time
        aids_by_frame = list(
            map(coco_dset.gid_to_aids.get,
                coco_dset.index._set_sorted_by_frame_index(coco_dset.imgs)))

        for frame_ix, aids in enumerate(aids_by_frame):

            for aid in aids:

                ann = coco_dset.anns[aid]
                if 'track_id' not in ann:
                    trackid = next(new_trackids)
                    ann['track_id'] = trackid
                else:
                    trackid = ann['track_id']

                next_aid = _search(aid, aids_by_frame[frame_ix + 1:])
                if next_aid is not None:
                    next_ann = coco_dset.anns[next_aid]
                    next_ann['track_id'] = trackid

                DEBUG_JSON_SERIALIZABLE = 0
                if DEBUG_JSON_SERIALIZABLE:
                    from kwcoco.util import util_json
                    unserializable = list(util_json.find_json_unserializable(next_ann))
                    if unserializable:
                        raise Exception('Inside OverlapTrack: ' + ub.urepr(unserializable))

        return coco_dset
