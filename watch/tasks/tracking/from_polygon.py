import itertools
import geojson
import json
import os
import sys
import argparse
import kwcoco
import dateutil.parser
import kwimage
import shapely
import shapely.ops
from os.path import join
from collections import defaultdict
from progiter import ProgIter
import numpy as np
import ubelt as ub

import watch
from watch.utils.kwcoco_extensions import TrackidGenerator


def mono(coco_dset):
    '''
    Track function.

    Combine all polygons into the same track.
    '''
    coco_dset.annots().set('track_id', next(TrackidGenerator(coco_dset)))

    return coco_dset


def overlap(coco_dset, min_overlap=0):
    '''
    Track function.

    Put polygons in the same track if their areas overlap.
    '''
    new_trackids = TrackidGenerator(coco_dset)

    aid_to_poly = dict(
        zip(
            coco_dset.annots().aids,
            map(
                lambda poly: poly.to_shapely().buffer(0),
                coco_dset.annots().detections.data['segmentations'].
                to_polygon_list())))

    def _search(aid, aid_groups):
        poly1 = aid_to_poly[aid]

        def _search_group(aids):
            for aid2 in aids:
                if 'track_id' not in coco_dset.anns[aid2]:
                    poly2 = aid_to_poly[aid2]
                    # check overlap
                    if poly1.intersects(poly2):
                        if (poly1.intersection(poly2).area /
                                poly2.area) > min_overlap:
                            return aid2

        return next(filter(None, map(_search_group, aid_groups)))

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
