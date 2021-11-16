from watch.utils import kwcoco_extensions
from watch.utils import util_kwimage
from collections import defaultdict
import kwarray
import kwimage
import numpy as np
import kwcoco
from rasterio import features


def _score(poly, probs):
    # Compute a score for the polygon
    # First compute the valid bounds of the polygon
    # And create a mask for only the valid region of the polygon
    box = poly.bounding_box().quantize().to_xywh()
    # Ensure w/h are positive
    box.data[:, 2:4] = np.maximum(box.data[:, 2:4], 1)
    x, y, w, h = box.data[0]
    y = max(y, 0)
    x = max(x, 0)
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


def filter_small_polys(scored_polys, min_area_px=80):
    for poly, score in scored_polys:
        if poly.to_shapely().area > min_area_px:
            yield poly, score


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


def mask_to_scored_polygons_v2(probs, thresh):
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
    binary_mask = probs > thresh
    shapes = list(features.shapes(binary_mask.astype(np.int16)))
    polygons = [kwimage.Polygon.from_geojson(s).translate((-0.5, -0.5)) for s, v in shapes if v]
    for poly in polygons:
        yield poly, _score(poly, probs)


def get_poly_overlap(poly, prob_map, threshold):
    """
    Get overlap of a polygon with a heatmap
    ToDo: this is inefficient, replace with ratserization of
    a smaller region around the bbox of poly
    """
    prob_map = prob_map[:, :, 0]
    hard_prob = prob_map > threshold
    poly_mask = poly.to_mask(prob_map.shape).numpy().data

    overlap = (hard_prob * poly_mask).sum()
    total_poly_area = poly_mask.sum()

    return overlap/total_poly_area


def get_poly_response(poly, dset, gid, key):
    """
    Find average respons of a heatmap within a polygon
    """
    img = dset.index.imgs[gid]
    coco_img = kwcoco_extensions.CocoImage(img, dset)
    try:
        if key[0] in coco_img.channels:
            img_probs = coco_img.delay(key[0], space='video').finalize()
    except:
        import xdev; xdev.embed()
    prob_map = img_probs[:,:,0]

    poly_mask = poly.to_mask(prob_map.shape).numpy().data

    response = (poly_mask*prob_map).mean()
    return response


def get_poly_response_sc(poly, dset, gid, key):
    """
    Find average respons of a heatmap within a polygon
    """
    img = dset.index.imgs[gid]
    coco_img = kwcoco_extensions.CocoImage(img, dset)
    zeros = np.zeros_like(coco_img.delay(coco_img.channels.fuse()[0], space='video').finalize()).astype('float32')
    fg_img_probs = zeros.copy()
    for k in key:
        k2 = kwcoco.FusedChannelSpec.coerce(k)
        common = kwcoco.FusedChannelSpec.coerce(coco_img.channels.fuse()).intersection(k2)
        if len(k2) == len(common):
            img_probs = coco_img.delay(k, space='video').finalize()
            fg_img_probs += img_probs

    prob_map = fg_img_probs[:, :, 0]

    poly_mask = poly.to_mask(prob_map.shape).numpy().data

    response = (poly_mask * prob_map).mean()
    return response


def get_poly_time_ind(scored_polys, threshold, dset, vidid, key, reverse=False):
    """
    Given a set of polygons, compute index of the first match of a polygon
    with a mask; mask is computed by comparing heatmaps with threshold.
    """
    gids = dset.index.vidid_to_gids[vidid]

    poly_started = set()
    poly_start_ind = [0 for (p, s) in scored_polys]
    if isinstance(key, str):
        key = [key]

    if reverse:
        gids = list(reversed(gids))

    for image_ind, gid in enumerate(gids):
        img = dset.index.imgs[gid]
        coco_img = kwcoco_extensions.CocoImage(img, dset)
        zeros = np.zeros_like(
        coco_img.delay(coco_img.channels.fuse()[0], space='video').finalize()).astype('float32')
        fg_img_probs = zeros.copy()
        for k in key:
            k2 = kwcoco.FusedChannelSpec.coerce(k)
            common = kwcoco.FusedChannelSpec.coerce(coco_img.channels.fuse()).intersection(k2)
            if len(k2) == len(common):
                img_probs = coco_img.delay(k, space='video').finalize()
                fg_img_probs += img_probs
                #if key in coco_img.channels:
                #img_probs = coco_img.delay(key, space='video').finalize()

                for poly_ind, (p, score) in enumerate(scored_polys):
                    if p not in poly_started:
                        overlap = get_poly_overlap(p, fg_img_probs, threshold=threshold)
                        if overlap > 0.5:
                            poly_started.add(p)
                            poly_start_ind[poly_ind] = image_ind
            else:
                print('image', gid, 'does not have predictions')

    if reverse:
        poly_start_ind = [len(gids) - i for i in poly_start_ind]

    return poly_start_ind


def filter_polys_response(polys, responses, response_thresh=0.001):
    response_aggregate = np.asarray(responses).mean(axis=1)
    for i, (poly, score) in enumerate(polys):
        if response_aggregate[i] > response_thresh:
            yield poly, score


def _validate_keys(key, bg_key):
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
    return key, bg_key


def vidpolys_to_tracks(coco_dset,
                       scored_polys,
                       thresh,
                       key,
                       bg_key=None,
                       coco_dset_sc=None,
                       poly_start_ind=None,
                       poly_end_ind=None):
    '''
    Take a set of scored single polygons (vidpolys) and add them to each frame
    of coco_dset as tracks using the categories/heatmaps from coco_dset_sc.
    '''
    key, bg_key = _validate_keys(key, bg_key)
    if coco_dset_sc is None:
        coco_dset_sc = coco_dset
    if poly_start_ind is None:
        poly_start_ind = defaultdict(lambda: -1)
    if poly_end_ind is None:
        poly_end_ind = defaultdict(lambda: int(1e99))

    new_trackids = kwcoco_extensions.TrackidGenerator(coco_dset)

    for poly_ind, (vid_poly, score) in enumerate(scored_polys):
        track_id = next(new_trackids)
        for image_ind, (gid, img) in enumerate(coco_dset_sc.imgs.items()):

            save_this_polygon = (
                poly_start_ind[poly_ind] < image_ind < poly_end_ind[poly_ind])

            if save_this_polygon:
                # assign category (key) from max score
                coco_img = kwcoco_extensions.CocoImage(img, coco_dset_sc)
                if score > thresh or len(bg_key) == 0:
                    cand_keys = key
                else:
                    cand_keys = bg_key

                if len(cand_keys) > 1:
                    cand_scores = []
                    for k in cand_keys:
                        if k in coco_img.channels:
                            img_probs = coco_img.delay(
                                k, space='video').finalize()
                            cand_scores.append(_score(vid_poly, img_probs))
                        else:
                            cand_scores.append(0)

                    cat_name = cand_keys[np.argmax(cand_scores)]
                else:
                    cat_name = cand_keys[0]

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


def time_aggregated_polys(coco_dset,
                          thresh=0.15,
                          morph_kernel=3,
                          key='salient',
                          bg_key=None,
                          time_filtering=False,
                          response_filtering=False,
                          return_only_polys=False):
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
    key, bg_key = _validate_keys(key, bg_key)
    _all_keys = set(key + bg_key)
    has_requested_chans_list = []
    for img in coco_dset.imgs.values():
        coco_img = kwcoco_extensions.CocoImage(img, coco_dset)
        chan_codes = coco_img.channels.normalize().fuse().as_set()
        flag = bool(_all_keys & chan_codes)
        has_requested_chans_list.append(flag)

    if not all(has_requested_chans_list):
        raise KeyError(f'{coco_dset.tag} has no keys {key} or {bg_key}')

    # record fg and bg keys across frames, and partial sums of fg and bg
    # this guarantees RunningStats of equal length for all keys,
    # even with partial/nonexistence
    running_dct = defaultdict(kwarray.RunningStats)
    for img in coco_dset.imgs.values():

        coco_img = coco_dset.coco_image(img['id'])
        zeros = np.zeros_like(
            coco_img.delay(coco_img.channels.fuse()[0],
                           space='video').finalize()).astype('float32')
        fg_img_probs = zeros.copy()
        bg_img_probs = zeros.copy()

        for k in key:
            k2 = kwcoco.FusedChannelSpec.coerce(k)
            common = kwcoco.FusedChannelSpec.coerce(
                coco_img.channels.fuse()).intersection(k2)
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
            common = kwcoco.FusedChannelSpec.coerce(
                coco_img.channels.fuse()).intersection(k2)
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

        hard_probs = util_kwimage.morphology(probs > thresh, 'dilate',
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

    scored_polys = list(filter_small_polys(scored_polys))
    print('removed small: remaining polygons:', len(scored_polys))

    if response_filtering:
        # get polygon responses
        responses = [[] for (p, score) in scored_polys]
        for track_id, (vid_poly, score) in enumerate(scored_polys, start=1):
            vidid = list(coco_dset.index.videos)[0]
            gids = coco_dset.index.vidid_to_gids[vidid]
            for image_ind, gid in enumerate(gids):
                response = get_poly_response(vid_poly, coco_dset, gid, key)
                # response = get_poly_response_sc(vid_poly, coco_dset, gid, key)
                responses[track_id - 1].append(response)
        scored_polys = list(
            filter_polys_response(scored_polys,
                                  responses,
                                  response_thresh=0.0002))  # 0.0005
        print('after filtering based on per-polygon response',
              len(scored_polys))

    if time_filtering:
        vidid = list(coco_dset.index.videos)[0]
        poly_start_ind = get_poly_time_ind(scored_polys, thresh, coco_dset,
                                           vidid, key)
        poly_end_ind = get_poly_time_ind(scored_polys,
                                         thresh,
                                         coco_dset,
                                         vidid,
                                         key,
                                         reverse=True)
    else:
        poly_start_ind = None
        poly_end_ind = None

    if return_only_polys:
        return coco_dset, scored_polys, poly_start_ind, poly_end_ind

    coco_dset = vidpolys_to_tracks(coco_dset, scored_polys, thresh, key,
                                   bg_key, None, poly_start_ind, poly_end_ind)
    return coco_dset


def time_aggregated_polys_bas(coco_dset,
                              thresh=0.3,
                              morph_kernel=3,
                              time_filtering=True,
                              response_filtering=False):
    '''
    Wrapper for BAS that looks for change heatmaps.
    '''
    change_keys = ['salient', 'change_prob', 'change']
    for change_key in change_keys:
        try:
            return time_aggregated_polys(coco_dset,
                                         thresh,
                                         morph_kernel,
                                         change_key,
                                         time_filtering=time_filtering,
                                         response_filtering=response_filtering)
        except KeyError:
            pass

    raise KeyError(
        f'{coco_dset.tag} does not contain any image channel {change_keys}')


def time_aggregated_polys_sc(coco_dset,
                             thresh=0.1,
                             morph_kernel=3,
                             time_filtering=False,
                             response_filtering=False):
    '''
    Wrapper for Site Characterization that looks for phase heatmaps.
    '''
    return time_aggregated_polys(coco_dset,
                                 thresh,
                                 morph_kernel,
                                 key=[
                                     'Site Preparation',
                                     'Active Construction',
                                     'Post Construction'
                                 ],
                                 bg_key=['No Activity'],
                                 time_filtering=time_filtering,
                                 response_filtering=response_filtering)


def time_aggregated_polys_hybrid(coco_dset,
                                 coco_dset_sc,
                                 thresh=0.3,
                                 morph_kernel=3,
                                 time_filtering=True,
                                 response_filtering=False):
    '''
    This method uses predictions from a BAS model to generate polygons.
    Predicted heatmaps from a Site Charachterization model are used to assign
    activity label to every polygon.
    coco_dset: KWCOCO file with BAS predictions
    coco_dset_sc: KWCOCO file with site characterization predictions

    '''
    return_tuple = time_aggregated_polys(coco_dset,
                                         thresh,
                                         morph_kernel,
                                         key=['salient'],
                                         time_filtering=time_filtering,
                                         response_filtering=response_filtering,
                                         return_only_polys=True)
    coco_dset, scored_polys, poly_start_ind, poly_end_ind = return_tuple

    get_poly_labels_from_SC = vidpolys_to_tracks
    coco_dset = get_poly_labels_from_SC(coco_dset,
                                        scored_polys,
                                        thresh,
                                        key=[
                                            'Site Preparation',
                                            'Active Construction',
                                            'Post Construction'
                                        ],
                                        bg_key=['No Activity'],
                                        coco_dset_sc=coco_dset_sc,
                                        poly_start_ind=poly_start_ind,
                                        poly_end_ind=poly_end_ind)

    return coco_dset
