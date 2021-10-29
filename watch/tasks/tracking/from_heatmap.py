from watch.utils import kwcoco_extensions
from watch.utils import util_kwimage
from collections import defaultdict
import kwarray
import kwimage
import numpy as np
import kwcoco


def _score(poly, probs):
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
    # hacking to solve a bug: sometimes shape of rel_probs is x,y,1
    if len(rel_probs.shape) == 3:
        rel_probs = rel_probs[:, :, 0]
    total = rel_mask.sum()
    score = 0 if total == 0 else (rel_mask * rel_probs).sum() / total
    return score


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
    hard_mask_c = (probs > thresh).astype(np.uint8)
    # Convert to polygons
    hard_mask = kwimage.Mask(hard_mask_c, 'c_mask')
    polygons = hard_mask.to_multi_polygon()
    for poly in polygons:
        yield poly, _score(poly, probs)


def time_aggregated_polys(coco_dset,
                          thresh=0.15,
                          morph_kernel=3,
                          key='salient',
                          bg_key=None):
    '''
    Track function.

    Aggregate heatmaps across time, threshold them to get polygons,
    and add one track per polygon.

    Args:
        key (String | List[String]): foreground key(s).

        bg_key (String | List[String] | None): background key(s).
            If None, background heatmaps become 1 - sum(foreground keys)

        thresh (float): For each frame, if sum of foreground heatmaps > thresh,
            class is max(foreground keys).
            else, class is max(background keys).

        morph_kernel (int): height/width in px of close-kernel
    '''

    # for backwards compatibility
    if isinstance(key, str):
        key = [key]
    if bg_key is None:
        bg_key = []
    elif isinstance(bg_key, str):
        bg_key = [bg_key]

    # error checking
    if len(key) < 1:
        raise ValueError('must have at least one key')
    if (len(key) > len(set(key)) or len(bg_key) > len(set(bg_key))):
        raise ValueError('keys are duplicated')
    if not set(key).isdisjoint(set(bg_key)):
        raise ValueError('cannot have a key in foreground and background')
    _all_keys = set(key + bg_key)
    #if all(
    #        _all_keys.isdisjoint(
    #            kwcoco_extensions.CocoImage(img, coco_dset).channels)
    #        for img in coco_dset.imgs.values()):
    #    raise ValueError(f'{coco_dset.tag} has no keys {key} or {bg_key}')
    has_requested_chans_list = []
    for img in coco_dset.imgs.values():
        coco_img = kwcoco_extensions.CocoImage(img, coco_dset)
        chan_codes = coco_img.channels.normalize().fuse().as_set()
        flag = bool(_all_keys & chan_codes)
        has_requested_chans_list.append(flag)

    if not all(has_requested_chans_list):
        raise ValueError(f'{coco_dset.tag} has no keys {key} or {bg_key}')

    # record fg and bg keys across frames, and partial sums of fg and bg
    # this guarantees RunningStats of equal length for all keys,
    # even with partial/nonexistence
    running_dct = defaultdict(kwarray.RunningStats)
    for img in coco_dset.imgs.values():

        coco_img = coco_dset.coco_image(img['id'])
        zeros = np.zeros_like(
            coco_img.delay(coco_img.channels.fuse()[0], space='video').finalize()).astype('float32')
        fg_img_probs = zeros.copy()
        bg_img_probs = zeros.copy()

        for k in key:
            k2 = kwcoco.FusedChannelSpec.coerce(k)
            common = kwcoco.FusedChannelSpec.coerce(coco_img.channels.fuse()).intersection(k2)
            if len(k2) == len(common):
                img_probs = coco_img.delay(k, space='video').finalize()
                fg_img_probs += img_probs
                running_dct[k].update(img_probs)
            else:
                # not sure if it is correct to assign zeros for images without predictions,
                # commenting out for now
                # running_dct[k].update(zeros)
                pass
        running_dct['fg'].update(fg_img_probs)

        for k in bg_key:
            k2 = kwcoco.FusedChannelSpec.coerce(k)
            common = kwcoco.FusedChannelSpec.coerce(coco_img.channels.fuse()).intersection(k2)
            if len(k2) == len(common):
                img_probs = coco_img.delay(k, space='video').finalize()
                bg_img_probs += img_probs
                running_dct[k].update(img_probs)
            else:
                # commenting out for now
                #running_dct[k].update(zeros)
                pass
        running_dct['bg'].update(bg_img_probs)

    # turn heatmaps into scores and polygons
    def probs(running):
        probs = running.summarize(axis=2, keepdims=False)['mean']

        hard_probs = util_kwimage.morphology(probs > thresh, 'close',
                                             morph_kernel)
        modulated_probs = probs * hard_probs
        return modulated_probs

    # TODO this still restricts to same-shape polygon in every frame
    # only the label (key) changes per-frame
    # to generalize this, have to get scored_polys from all keys
    # and associate them somehow
    scored_polys = list(
        mask_to_scored_polygons(probs(running_dct['fg']), thresh))

    print('time aggregation: number of polygons:', len(scored_polys))
    # Add each polygon to every images as a track
    new_trackids = kwcoco_extensions.TrackidGenerator(coco_dset)

    for vid_poly, score in scored_polys:
        track_id = next(new_trackids)
        for gid, img in coco_dset.imgs.items():

            # assign category (key) from max score
            coco_img = kwcoco_extensions.CocoImage(img, coco_dset)
            if score > thresh or len(bg_key) == 0:
                cand_keys = key
            else:
                cand_keys = bg_key
            cand_scores = []
            for k in cand_keys:
                if k in coco_img.channels:
                    img_probs = coco_img.delay(k, space='video').finalize()
                    cand_scores.append(_score(vid_poly, img_probs))
                else:
                    cand_scores.append(0)

            #cat_name = np.max(cand_keys, where=cand_scores)
            cat_name = cand_keys[np.argmax(cand_scores)]
            cid = coco_dset.ensure_category(cat_name)

            vid_from_img = kwimage.Affine.coerce(img['warp_img_to_vid'])
            img_from_vid = vid_from_img.inv()

            # Transform the video polygon into image space
            img_poly = vid_poly.warp(img_from_vid)
            bbox = list(img_poly.bounding_box().to_coco())[0]
            # Add the polygon as an annotation on the image
            coco_dset.add_annotation(image_id=gid,
                                     category_id=cid,
                                     bbox=bbox,
                                     segmentation=img_poly,
                                     score=score,
                                     track_id=track_id)

    return coco_dset


def time_aggregated_polys_bas(coco_dset, thresh=0.15, morph_kernel=3):
    '''
    Wrapper for BAS that looks for change heatmaps.
    '''
    change_keys = ['salient', 'change_prob', 'change']
    for change_key in change_keys:
        try:
            return time_aggregated_polys(coco_dset, thresh, morph_kernel,
                                         change_key)
        except ValueError:
            pass

    raise ValueError(
        f'{coco_dset.tag} does not contain any image channel {change_keys}')


def time_aggregated_polys_sc(coco_dset,
                             thresh=0.15,
                             morph_kernel=3,
                             bg_thresh=True):
    '''
    Wrapper for Site Characterization that looks for phase heatmaps.

    Args:
        bg_thresh: If True, treat No Activity as background and other classes as foreground,
            using thresh to switch between them. Else, all classes are foreground.
            TODO enable using a different thresh here than in mask_to_scored_polygons
    '''
    if bg_thresh:
        return time_aggregated_polys(coco_dset,
                                     thresh,
                                     morph_kernel,
                                     key=[
                                         'Site Preparation',
                                         'Active Construction',
                                         'Post Construction'
                                     ],
                                     bg_key=['No Activity'])
    else:
        return time_aggregated_polys(coco_dset,
                                     thresh,
                                     morph_kernel,
                                     key=[
                                         'Site Preparation',
                                         'Active Construction',
                                         'Post Construction', 'No Activity'
                                     ])
