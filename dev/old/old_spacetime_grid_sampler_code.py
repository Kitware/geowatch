import kwarray
import kwimage
import numpy as np
import ubelt as ub
from watch import heuristics
from watch.utils import util_kwimage


def lookup_track_info(coco_dset, tid):
    """
    UNUSED. DEPRECATED OR FIND USE.

    Find the spatio-temporal extent of a track
    """
    track_aids = coco_dset.index.trackid_to_aids[tid]
    vidspace_boxes = []
    track_gids = []
    for aid in track_aids:
        ann = coco_dset.index.anns[aid]
        gid = ann['image_id']
        track_gids.append(gid)
        img = coco_dset.index.imgs[gid]
        bbox = ann['bbox']
        vid_from_img = kwimage.Affine.coerce(img.get('warp_img_to_vid', None))
        imgspace_box = kwimage.Boxes([bbox], 'xywh')
        vidspace_box = imgspace_box.warp(vid_from_img)
        vidspace_boxes.append(vidspace_box)
    all_vidspace_boxes = kwimage.Boxes.concatenate(vidspace_boxes)

    full_vid_box = all_vidspace_boxes.bounding_box().to_xywh()

    frame_index = coco_dset.images(track_gids).lookup('frame_index')
    track_gids = list(ub.take(track_gids, ub.argsort(frame_index)))

    track_info = {
        'tid': tid,
        'full_vid_box': full_vid_box,
        'track_gids': track_gids,
    }
    return track_info


def make_track_based_spatial_samples(coco_dset):
    """
    UNUSED. DEPRECATED OR FIND USE.

    Ignore:
        >>> # xdoctest: +REQUIRES(env:DVC_DPATH)
        >>> import os
        >>> from watch.tasks.fusion.datamodules.spacetime_grid_builder import *  # NOQA
        >>> import watch
        >>> dvc_dpath = watch.find_dvc_dpath(tags='phase2_data', hardware='ssd')
        >>> coco_fpath = dvc_dpath / 'Drop1-Aligned-L1/data.kwcoco.json'
        >>> coco_dset = kwcoco.CocoDataset(coco_fpath)
    """
    tid_list = list(coco_dset.index.trackid_to_aids.keys())
    tid_to_trackinfo = {}
    for tid in tid_list:
        track_info = lookup_track_info(coco_dset, tid)
        gid = track_info['track_gids'][0]
        vidid = coco_dset.index.imgs[gid]['video_id']
        track_info['vidid'] = vidid
        tid_to_trackinfo[tid] = track_info

    vidid_to_tracks = ub.group_items(tid_to_trackinfo.values(), key=lambda x: x['vidid'])

    winspace_space_dims = [96, 96]

    for vidid, trackinfos in vidid_to_tracks.items():
        positive_boxes = []
        for track_info in trackinfos:
            boxes = track_info['full_vid_box']
            positive_boxes.append(boxes.to_cxywh())
        positives = kwimage.Boxes.concatenate(positive_boxes)
        positives_samples = positives.to_cxywh()
        positives_samples.data[:, 2] = winspace_space_dims[0]
        positives_samples.data[:, 3] = winspace_space_dims[1]
        print('positive_boxes = {}'.format(ub.urepr(positive_boxes, nl=1)))

        video = coco_dset.index.videos[vidid]
        full_dims = [video['height'], video['width']]
        window_overlap = 0.0
        keepbound = 0

        window_dims_ = full_dims if winspace_space_dims == 'full' else winspace_space_dims
        slider = kwarray.SlidingWindow(full_dims, window_dims_,
                                       overlap=window_overlap,
                                       keepbound=keepbound,
                                       allow_overshoot=True)

        sliding_boxes = kwimage.Boxes.concatenate(list(map(kwimage.Boxes.from_slice, slider)))
        ious = sliding_boxes.ious(positives)
        overlaps = ious.sum(axis=1)
        negative_boxes = sliding_boxes.compress(overlaps == 0)

        if 1:
            import kwplot
            kwplot.autompl()
            fig = kwplot.figure(fnum=vidid)
            ax = fig.gca()
            ax.set_title(video['name'])
            negative_boxes.draw(setlim=1, color='red', fill=True)
            positives.draw(color='limegreen')
            positives_samples.draw(color='green')
