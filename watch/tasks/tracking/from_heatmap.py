from watch.utils import kwcoco_extensions
from watch.utils import util_kwimage
import kwarray
import kwimage
import numpy as np


def mask_to_scored_polygons(probs, thresh):
    """
    Example:
        >>> from watch.tasks.tracking.from_heatmap import *  # NOQA
        >>> import kwimage
        >>> probs = kwimage.Heatmap.random(dims=(64, 64), rng=0).data['class_probs'][0]
        >>> thresh = 0.5
        >>> poly1, score1 = list(mask_to_scored_polygons(probs, thresh))[0]
        >>> # xdoctest: +IGNORE_WANT
        >>> import kwplot
        >>> kwplot.autompl()
        >>> kwplot.imshow(probs > 0.5)
    """
    # Threshold scores
    hard_mask = probs > thresh
    # Convert to polygons
    polygons = kwimage.Mask(hard_mask, 'c_mask').to_multi_polygon()
    for poly in polygons:
        # Compute a score for the polygon
        # First compute the valid bounds of the polygon
        # And create a mask for only the valid region of the polygon
        box = poly.bounding_box().quantize().to_xywh()
        # Ensure w/h are positive
        box.data[:, 2:4] = np.maximum(box.data[:, 2:4], 1)
        x, y, w, h = box.data[0]
        rel_poly = poly.translate((-x, -y))
        rel_mask = rel_poly.to_mask((h, w)).data
        # Slice out the corresponding region of probabilities
        rel_probs = probs[y:y + h, x:x + w]
        total = rel_mask.sum()
        score = 0 if total == 0 else (rel_mask * rel_probs).sum() / total
        yield poly, score


def time_aggregated_polys(coco_dset, thresh=0.15, morph_kernel=3, key='salient'):
    '''
    Track function.

    Aggregate heatmaps across time, threshold them to get polygons,
    and add one track per polygon.
    '''
    running = kwarray.RunningStats()
    for img in coco_dset.imgs.values():
        coco_img = kwcoco_extensions.CocoImage(img, coco_dset)
        if key in coco_img.channels:
            img_probs = coco_img.delay(key, space='video').finalize()
            running.update(img_probs)

    probs = running.summarize(axis=2, keepdims=False)['mean']

    hard_probs = util_kwimage.morphology(probs > thresh, 'close', morph_kernel)
    modulated_probs = probs * hard_probs

    scored_polys = list(mask_to_scored_polygons(modulated_probs, thresh))

    print('time aggregation: number of polygons:', len(scored_polys))
    # Add each polygon to every images as a track
    new_trackids = kwcoco_extensions.TrackidGenerator(coco_dset)
    change_cid = coco_dset.ensure_category(key)
    for vid_poly, score in scored_polys:
        track_id = next(new_trackids)
        for gid, img in coco_dset.imgs.items():
            vid_from_img = kwimage.Affine.coerce(img['warp_img_to_vid'])
            img_from_vid = vid_from_img.inv()

            # Transform the video polygon into image space
            img_poly = vid_poly.warp(img_from_vid)
            bbox = list(img_poly.bounding_box().to_coco())[0]
            # Add the polygon as an annotation on the image
            coco_dset.add_annotation(
                image_id=gid, category_id=change_cid,
                bbox=bbox, segmentation=img_poly, score=score,
                track_id=track_id)

    return coco_dset