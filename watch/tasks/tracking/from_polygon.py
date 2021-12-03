from dataclasses import dataclass
from watch.tasks.tracking.utils import TrackFunction
from watch.utils.kwcoco_extensions import TrackidGenerator


class MonoTrack(TrackFunction):
    '''
    Combine all polygons into the same track.
    '''
    def __call__(self, coco_dset):
        coco_dset.annots().set('track_id', next(TrackidGenerator(coco_dset)))

        return coco_dset


def as_shapely_polys(annots):
    return map(lambda poly: poly.to_shapely().buffer(0),
               annots.detections.data['segmentations'].to_polygon_list())


@dataclass
class OverlapTrack(TrackFunction):
    '''
    Put polygons in the same track if their areas overlap.
    '''
    min_overlap: float = 0

    def __call__(self, coco_dset):
        new_trackids = TrackidGenerator(coco_dset)

        aid_to_poly = dict(
            zip(coco_dset.annots().aids, as_shapely_polys(coco_dset.annots())))

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

        return coco_dset
