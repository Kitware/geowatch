"""
Main tracker logic

SeeAlso:
    * ../../cli/kwcoco_to_geojson.py
"""
from watch.utils import kwcoco_extensions
from watch.heuristics import SITE_SUMMARY_CNAME, CNAMES_DCT
import kwimage
import kwcoco
import numpy as np
import ubelt as ub
import geopandas as gpd
import itertools
import math
from typing import Iterable, Tuple, Union, Optional, Literal
from dataclasses import dataclass
from shapely.ops import unary_union
from watch.tasks.tracking.utils import (NewTrackFunction, mask_to_polygons,
                                        Poly, _validate_keys, pop_tracks,
                                        build_heatmaps, trackid_is_default,
                                        gpd_sort_by_gid, gpd_len,
                                        gpd_compute_scores)

try:
    from xdev import profile
except Exception:
    profile = ub.identity

#
# --- aggregation functions for heatmaps ---
#


def _norm(heatmaps, norm_ord):
    heatmaps = np.array(heatmaps)
    if norm_ord == np.inf:
        probs = np.nansum(heatmaps)
    else:
        probs = np.power(np.nansum(np.power(heatmaps, norm_ord), axis=0),
                         1. / norm_ord)
        if norm_ord > 0:
            n_nonzero = np.count_nonzero(~np.isnan(heatmaps), axis=0)
            n_nonzero[n_nonzero == 0] = 1
            probs /= np.power(n_nonzero, 1. / norm_ord)
    return probs


# give all these the same signature so they can be swapped out


def probs(heatmaps, norm_ord, morph_kernel, thresh):
    probs = _norm(heatmaps, norm_ord)

    hard_probs = kwimage.morphology(probs > thresh, 'dilate', morph_kernel)
    modulated_probs = probs * hard_probs

    return modulated_probs


def mean_normalized(heatmaps, norm_ord=1, morph_kernel=1, thresh=None):
    '''
    Normalize average_heatmap by applying a scaling based on max(heatmaps) and
    max(average_heatmap)
    '''
    average = _norm(heatmaps, norm_ord)

    scale_factor = np.max(heatmaps) / (np.max(average) + 1e-9)
    print('max heatmaps', np.max(heatmaps))
    print('max average', np.max(average))

    # average *= scale_factor
    average = 0.75 * average * scale_factor
    print('scale_factor', scale_factor)
    print('After scaling, max:', np.max(average))

    average = kwimage.morphology(average, 'dilate', morph_kernel)

    return average


def frequency_weighted_mean(heatmaps, thresh, norm_ord=0, morph_kernel=3):
    '''
    Convert a list of heatmaps to an aggregated score, averaging is computed
    based on samples for every pixel
    '''
    heatmaps = np.array(heatmaps)

    masks = 1 * (heatmaps > thresh)
    pixel_wise_samples = masks.sum(0) + 1e-9
    print('pixel_wise_samples', pixel_wise_samples)

    # compute sum
    aggregated_probs = _norm(masks * heatmaps, norm_ord)

    # divide by number of samples for every pixel
    aggregated_probs /= pixel_wise_samples

    aggregated_probs = kwimage.morphology(aggregated_probs, 'dilate',
                                          morph_kernel)

    return aggregated_probs


AGG_FN_REGISTRY = {
    'frequency_weighted_mean': frequency_weighted_mean,
    'mean_normalized': mean_normalized,
    'probs': probs,
}

#
# --- track/polygon filters ---
#


class TimePolygonFilter:
    '''
    Cuts off start and end of each track based on min response.
    '''

    def __init__(self, threshold):
        self.threshold = threshold

    def __call__(self, gdf):

        def _edit(grp):
            magic_thresh = 0.5
            (ok_ixs, ) = (grp[('fg', self.threshold)] >
                          magic_thresh).values.nonzero()
            if len(ok_ixs) == 0:
                start_ix, end_ix = len(grp), len(grp)
            else:
                start_ix, end_ix = ok_ixs[[0, -1]]
            # print(grp.name, start_ix, end_ix+1)
            return grp.iloc[start_ix:end_ix + 1]

        if len(gdf) > 0:
            group = gdf.groupby('track_idx', group_keys=False)
            result = group.apply(_edit)
        else:
            result = gdf
        return result


class ResponsePolygonFilter:
    '''
    Filters each track based on the average response of all tracks.
    '''

    def __init__(self, gdf, threshold):

        self.threshold = threshold

        gids = gdf['gid'].unique()
        mean_response = gdf[('fg', None)].mean()

        self.gids = gids
        self.mean_response = mean_response

    def __call__(self, gdf, gids=None, threshold=None, cross=True):
        if gids is None:
            gids = self.gids
        if threshold is None:
            threshold = self.threshold
        if cross:

            def _filter(grp):
                this_response = grp[grp['gid'].isin(self.gids)][('fg',
                                                                 None)].mean()
                return this_response / self.mean_response > threshold

            return gdf.groupby('track_idx', group_keys=False).filter(_filter)
        else:
            cond = (gdf[('fg', None)] / self.mean_response > threshold)
            return gdf[cond]


#
# --- main logic ---
#


@profile
def _add_tracks_to_dset(sub_dset, tracks, thresh, key, bg_key=None):
    key, bg_key = _validate_keys(key, bg_key)

    @ub.memoize
    def _warp_img_from_vid(gid):
        # Memoize the conversion to a matrix
        coco_img = sub_dset.coco_image(gid)
        img_from_vid = coco_img.warp_img_from_vid
        return img_from_vid

    def make_new_annotation(gid, poly, this_score, scores_dct, track_id,
                            space='video'):

        # assign category (key) from max score
        if this_score > thresh or len(bg_key) == 0:
            cand_keys = key
        else:
            cand_keys = bg_key
        if len(cand_keys) > 1:
            # TODO ensure bg classes are scored if there are >1 of them
            cat_name = cand_keys[np.argmax([scores_dct[k] for k in cand_keys])]
        else:
            cat_name = cand_keys[0]
        cid = sub_dset.ensure_category(cat_name)

        assert space in {'image', 'video'}
        if space == 'video':
            # Transform the video polygon into image space
            img_from_vid = _warp_img_from_vid(gid)
            poly = kwimage.MultiPolygon.coerce(poly).warp(img_from_vid)

        bbox = list(poly.bounding_box().to_coco())[0]
        segmentation = poly.to_coco(style='new')

        # Add the polygon as an annotation on the image
        new_ann = dict(image_id=gid,
                       category_id=cid,
                       bbox=bbox,
                       segmentation=segmentation,
                       score=this_score,
                       track_id=track_id)
        return new_ann

    new_trackids = kwcoco_extensions.TrackidGenerator(sub_dset)

    all_new_anns = []

    def _add(obs, tid):
        if not trackid_is_default(tid):
            track_id = tid
            new_trackids.exclude_trackids([track_id])
        else:
            track_id = next(new_trackids)

        for o in obs:
            new_ann = make_new_annotation(*o, track_id)
            all_new_anns.append(new_ann)

    try:
        groups = tracks.groupby('track_idx', axis=0)
    except ValueError:
        import warnings
        warnings.warn('warning: no tracks to add the the kwcoco dataset')
    else:
        for tid, grp in groups:
            score_chan = kwcoco.ChannelSpec('|'.join(key))
            this_score = grp[(score_chan.spec, None)]
            scores_dct = {k: grp[(k, None)] for k in score_chan.unique()}
            scores_dct = [dict(zip(scores_dct, t))
                          for t in zip(*scores_dct.values())]
            _add(zip(grp['gid'], grp['poly'], this_score, scores_dct), tid)

    # TODO: Faster to add annotations in bulk, but we need to construct the
    # "ids" first
    for new_ann in all_new_anns:
        sub_dset.add_annotation(**new_ann)

    DEBUG_JSON_SERIALIZABLE = 0
    if DEBUG_JSON_SERIALIZABLE:
        from watch.utils.util_json import debug_json_unserializable
        debug_json_unserializable(sub_dset.dataset)

    return sub_dset


@profile
def time_aggregated_polys(
        sub_dset,
        thresh,
        morph_kernel=3,
        key='salient',
        bg_key=None,
        time_thresh=1,
        response_thresh=None,
        use_boundaries=False,
        norm_ord=1,
        agg_fn='probs',
        moving_window_size=None,  # 150
        min_area_sqkm=0.072,  # 80px@30GSD
        # min_area_sqkm=0.018,  # 80px@15GSD
        # min_area_sqkm=0.008,  # 80px@10GSD
        max_area_sqkm=None,
        # max_area_sqkm=2.25,  # ~1.5x upper tail of truth
        max_area_behavior='drop',
        thresh_hysteresis=None,
        polygon_simplify_tolerance=None,
        resolution=None,
        ):
    '''
    Track function.

    Aggregate heatmaps across time, threshold them to get polygons,
    and add one track per polygon.

    Args:
        sub_dset (kwcoco.CocoDataset): a kwcoco dataset with exactly 1 video

        key (String | List[String]): foreground key(s).

        bg_key (String | List[String] | None): background key(s).
            If None, background heatmaps become 1 - sum(foreground keys)

        thresh (float): For each frame, if sum of foreground heatmaps > thresh,
            class is max(foreground keys).
            else, class is max(background keys).

        morph_kernel (int): height/width in px of close or dilate kernel

        norm_ord: order of norm to aggregate heatmap pixels across time.
            1: average [default]
            2: euclidean
            0: sum
            np.inf, 'inf', or None: max

        agg_fn: (3d heatmaps -> 2d heatmaps), calling convention TBD

        resolution (str | None):
            A resolution in units understood by kwcoco. E.g. "10GSD"

    Ignore:
        # For debugging
        import xdev
        from watch.tasks.tracking.from_heatmap import *  # NOQA
        from watch.tasks.tracking.from_heatmap import _validate_keys
        globals().update(xdev.get_func_kwargs(time_aggregated_polys))

    Example:
        >>> # test interpolation
        >>> from watch.tasks.tracking.from_heatmap import time_aggregated_polys
        >>> from watch.demo import demo_kwcoco_with_heatmaps
        >>> import watch
        >>> #sub_dset = demo_kwcoco_with_heatmaps(
        >>> #                num_frames=5, image_size=(480, 640))
        >>> sub_dset = watch.coerce_kwcoco(
        >>>     'watch-msi', num_videos=1, num_frames=5, image_size=(480, 640),
        >>>     geodata=True, heatmap=True)
        >>> thresh = 0.01
        >>> min_area_sqkm = None
        >>> orig_track = time_aggregated_polys(
        >>>                 sub_dset, thresh, min_area_sqkm=min_area_sqkm, time_thresh=None)
        >>> # Test robustness to frames that are missing heatmaps
        >>> skip_gids = [1,3]
        >>> for gid in skip_gids:
        >>>      sub_dset.imgs[gid]['auxiliary'].pop()
        >>> inter_track = time_aggregated_polys(
        >>>                 sub_dset, thresh, min_area_sqkm=min_area_sqkm, time_thresh=None)
        >>> assert inter_track.iloc[0][('fg', None)] == 0
        >>> assert inter_track.iloc[1][('fg', None)] > 0
    '''
    #
    # --- input validation ---
    #

    key, bg_key = _validate_keys(key, bg_key)
    _all_keys = set(key + bg_key)
    has_requested_chans_list = []

    coco_videos = sub_dset.videos()
    assert len(coco_videos) == 1, 'we expect EXACTLY one video here'
    video = coco_videos.objs[0]
    video_name = video.get('name', None)
    video_id = video['id']

    video_gids = list(sub_dset.images(video_id=video_id))
    for gid in video_gids:
        coco_img = sub_dset.coco_image(gid)
        chan_codes = coco_img.channels.normalize().fuse().as_set()
        flag = bool(_all_keys & chan_codes)
        has_requested_chans_list.append(flag)

    scale_vid_from_trk = None
    video_gsd = None
    if len(video_gids):
        # Determine resolution information for videospace (what we will return
        # in) and tracking space (what we will build heatmaps in)
        first_gid = video_gids[0]
        first_coco_img = sub_dset.coco_image(first_gid)
        try:
            vidspace_resolution = first_coco_img.resolution(space='video')
            video_gsd = np.mean(vidspace_resolution['mag'])
            if resolution is not None:
                # Get the transform from tracking space back to video space
                scale_trk_from_vid = first_coco_img._scalefactor_for_resolution(
                    space='video', resolution=resolution)
                scale_vid_from_trk = 1 / np.array(scale_trk_from_vid)
        except Exception:
            ...

    if video_gsd is None:
        default_gsd = 30
        video_gsd = default_gsd
        print(f'warning: video {video["name"]} in dset {sub_dset.tag} '
              f'has no listed resolution; assuming {default_gsd}')

    if not any(has_requested_chans_list):
        raise KeyError(f'no imgs in dset {sub_dset.tag} '
                       f'have keys {key} or {bg_key}.')
    if not all(has_requested_chans_list):
        n_total = len(has_requested_chans_list)
        n_have = sum(has_requested_chans_list)
        n_missing = (n_total - n_have)
        print(f'warning: {n_missing} / {n_total} imgs in dset {sub_dset.tag} '
              f'with video {video_name} have no keys {key} or {bg_key}. '
              'Interpolating...')

    if norm_ord in {'inf', None}:
        norm_ord = np.inf

    #
    # --- main logic ---
    #

    gids_polys = _gids_polys(sub_dset,
                             key=key,
                             agg_fn=agg_fn,
                             thresh=thresh,
                             morph_kernel=morph_kernel,
                             thresh_hysteresis=thresh_hysteresis,
                             norm_ord=norm_ord,
                             moving_window_size=moving_window_size,
                             resolution=resolution,
                             bounds=use_boundaries)
    orig_gid_polys = list(gids_polys)  # 26% of runtime
    gids_polys = orig_gid_polys

    print('time aggregation: number of polygons: ', len(gids_polys))

    # polys here are vidpolys.
    # size and response filters should operate on each vidpoly separately.

    if max_area_sqkm:
        max_area_px = max_area_sqkm * 1e6 / (video_gsd**2)
        n_orig = len(gids_polys)
        if max_area_behavior == 'drop':
            gids_polys = [(t, p) for t, p in gids_polys
                          if p.to_shapely().area < max_area_px]
            print('filter large: remaining polygons: '
                  f'{len(gids_polys)} / {n_orig}')
        elif max_area_behavior == 'grid':
            # edits tracks instead of removing them
            raise NotImplementedError

    if min_area_sqkm:
        min_area_px = min_area_sqkm * 1e6 / (video_gsd**2)
        n_orig = len(gids_polys)
        gids_polys = [(t, p) for t, p in gids_polys
                      if p.to_shapely().area > min_area_px]
        print('filter small: remaining polygons: '
              f'{len(gids_polys)} / {n_orig}')

    # now we start needing scores, so bulk-compute them

    gids_polys_T = list(zip(*gids_polys))
    if gids_polys_T:
        gids, polys = gids_polys_T
    else:
        gids, polys = [], []

    polys = [p.to_shapely() for p in polys]

    _TRACKS = gpd.GeoDataFrame(dict(gid=gids, poly=polys), geometry='poly')

    if polygon_simplify_tolerance is not None:
        _TRACKS['poly'] = _TRACKS['poly'].simplify(tolerance=polygon_simplify_tolerance)

    # _TRACKS['track_idx'] = range(len(_TRACKS))
    _TRACKS = _TRACKS.reset_index().rename(columns={'index': 'track_idx'})
    _TRACKS = _TRACKS.explode('gid')

    # ensure index is sorted in video order
    sorted_gids = sub_dset.images(vidid=video['id']).gids
    _TRACKS = gpd_sort_by_gid(_TRACKS, sorted_gids)

    # awk, find better way of bookkeeping and indexing into scores needed
    thrs = {None}
    if response_thresh:
        thrs.add(None)
    if time_thresh:
        thrs.add(time_thresh * thresh)
    thrs = list(thrs)

    ks = {'fg': key, 'bg': bg_key}

    # 95% of runtime
    _TRACKS = gpd_compute_scores(_TRACKS, sub_dset, thrs, ks, USE_DASK=False,
                                 resolution=resolution)
    # 63% of runtime
    # _TRACKS = gpd_compute_scores(_TRACKS, sub_dset, thrs, ks, USE_DASK=True)
    # dask could unsort
    # _TRACKS = gpd_sort_by_gid(_TRACKS.reset_index(), sorted_gids)

    # response_thresh = 0.9
    if response_thresh:

        n_orig = gpd_len(_TRACKS)
        rsp_filter = ResponsePolygonFilter(_TRACKS, key, response_thresh)
        _TRACKS = rsp_filter(_TRACKS)
        print('filter based on per-polygon response: remaining tracks '
              f'{gpd_len(_TRACKS)} / {n_orig}')

    # TimePolygonFilter edits tracks instead of removing them
    if time_thresh:  # as a fraction of thresh
        time_filter = TimePolygonFilter(time_thresh * thresh)
        n_orig = gpd_len(_TRACKS)
        _TRACKS = time_filter(_TRACKS)  # 7% of runtime? could be next line
        print('filter based on time overlap: remaining tracks '
              f'{gpd_len(_TRACKS)} / {n_orig}')

    # The tracker assumes the polygons will be output in video space.
    if scale_vid_from_trk is not None and len(_TRACKS):
        # If a tracking resolution was specified undo the extra scale factor
        _TRACKS['poly'] = _TRACKS['poly'].scale(*scale_vid_from_trk, origin=(0, 0))

    # try to ignore this error
    # TODO: do we need to convert to MultiPolygon here? Or can that be handled
    # by consumers of this method?
    _TRACKS['poly'] = _TRACKS['poly'].map(kwimage.MultiPolygon.from_shapely)
    return _TRACKS


#
# --- time_aggregated_polys utilities ---
#


def _merge_polys(p1, p2):
    '''
    Given two lists of polygons, p1 and p2, merge these according to:
      - add all unique polygons in the merged list
      - for overlapping polygons, add the union of both polygons

    Ignore:
        from watch.tasks.tracking.from_heatmap import * # NOQA
        from watch.tasks.tracking.from_heatmap import _merge_polys  # NOQA

        p1 = [kwimage.Polygon.random().to_shapely() for _ in range(10)]
        p2 = [kwimage.Polygon.random().to_shapely() for _ in range(10)]
        _merge_polys(p1, p2)

        p1_kw = kwimage.Polygon(exterior=np.array([(0, 0), (1, 0), (0.5, 1)]))
        p2_kw = kwimage.Polygon(exterior=np.array([(0, 2), (1, 2), (0.5, 1)]))
        _p1 = p2_kw.to_shapely()
        _p2 = p1_kw.to_shapely()
        print(_p1.intersects(_p2))
        print(_p1.overlaps(_p2))
        print(unary_union([_p1, _p2]))

        p1_kw = kwimage.Boxes([[0, 0, 10, 10]], 'xywh').to_polygons()[0]
        p2_kw = kwimage.Boxes([[10, 0, 10, 10]], 'xywh').to_polygons()[0]
        _p1 = p2_kw.to_shapely()
        _p2 = p1_kw.to_shapely()
        print(_p1.intersects(_p2))
        print(_p1.overlaps(_p2))
        print(unary_union([_p1, _p2]))

        while True:
            _p1 = kwimage.Polygon.random().to_shapely()
            _p2 = kwimage.Polygon.random().to_shapely()
            if 1 or _p1.intersects(_p2):
                combo = unary_union([_p1, _p2])
                if combo.type != 'Polygon':
                    raise Exception('!')
    '''
    merged_polys = []

    p1_seen = set()
    p2_seen = set()

    # add all polygons that overlap
    for j, _p1 in enumerate(p1):
        if j in p1_seen:
            continue
        for i, _p2 in enumerate(p2):
            if (i in p2_seen) or (i > len(p2) - 1):
                continue
            if _p1.intersects(_p2):
                combo = unary_union([_p1, _p2])
                if combo.type == 'Polygon':
                    merged_polys.append(combo)
                elif combo.type == 'MultiPolygon':
                    # Can this ever happen? It seems to have occurred in a test
                    # run. Bowties can cause this.
                    # import warnings
                    # warnings.warn('Found two intersecting polygons where the
                    # union was a multipolygon')
                    merged_polys.extend(list(combo.geoms))
                else:
                    raise AssertionError(
                        f'Unexpected type {combo.type} from {_p1} and {_p2}')

                p1_seen.add(j)
                p2_seen.add(i)

    # all polygons that did not overlap with any polygon
    all_p1 = set(np.arange(len(p1)))
    remaining_p1 = all_p1 - p1_seen

    for index in remaining_p1:
        merged_polys.append(p1[index])

    all_p2 = set(np.arange(len(p2)))
    remaining_p2 = all_p2 - p2_seen
    for index in remaining_p2:
        merged_polys.append(p2[index])

    return merged_polys


@profile
def heatmaps_to_polys(heatmaps, bounds, agg_fn, thresh, morph_kernel,
                      thresh_hysteresis, norm_ord, moving_window_size):
    '''
    Use parameters: agg_fn, thresh, morph_kernel, thresh_hysteresis, norm_ord
    '''

    def convert_to_shapely(polys):
        return [p.to_shapely() for p in polys]

    def convert_to_kwimage_poly(shapely_polys):
        return [kwimage.Polygon.from_shapely(p) for p in shapely_polys]

    _agg_fn = AGG_FN_REGISTRY[agg_fn]

    def _process_1_step(heatmaps):
        aggregated = _agg_fn(heatmaps,
                             thresh=thresh,
                             morph_kernel=morph_kernel,
                             norm_ord=norm_ord)

        polygons = list(
            mask_to_polygons(aggregated,
                             thresh,
                             thresh_hysteresis=thresh_hysteresis,
                             bounds=bounds))
        return polygons

    if isinstance(moving_window_size, float) and math.isnan(moving_window_size):
        moving_window_size = None

    # calculate number of moving-window steps, based on window_size and number
    # of heatmaps
    if moving_window_size is not None:
        total_n = len(heatmaps)
        final_size = int(total_n // np.ceil((total_n / moving_window_size)))
        n_steps = total_n // final_size
    else:
        final_size = len(heatmaps)
        n_steps = 1

    # initialize heatmaps and initial polygons on the first set of heatmaps
    h_init = heatmaps[:final_size]
    polys_final = _process_1_step(h_init)

    if n_steps > 1:
        polys_final = convert_to_shapely(polys_final)

        for i in range(n_steps - 1):
            h1 = heatmaps[(i + 1) * final_size:(i + 2) * final_size]
            p1 = _process_1_step(h1)
            p1 = convert_to_shapely(p1)
            polys_final = _merge_polys(polys_final, p1)

        polys_final = convert_to_kwimage_poly(polys_final)

    return polys_final


def _gids_polys(
        sub_dset,
        key,
        agg_fn,
        thresh,
        morph_kernel,
        thresh_hysteresis,
        norm_ord,
        resolution=None,
        moving_window_size=None,  # 150
        bounds=False) -> Iterable[Union[int, Poly]]:
    if bounds:  # for SC
        raw_boundary_tracks = pop_tracks(sub_dset, [SITE_SUMMARY_CNAME])
        assert len(raw_boundary_tracks) > 0, 'need valid site boundaries!'
        gids = raw_boundary_tracks['gid'].unique()
        print('generating polys in bounds: number of bounds: ',
              gpd_len(raw_boundary_tracks))
        boundary_tracks = list(raw_boundary_tracks.groupby('track_idx'))

    else:
        boundary_tracks = [(None, None)]
        # TODO WARNING this is wrong!!! need to make sure this is never used.
        # The gids are lexically sorted, not sorted by order in video!
        # gids = list(sub_dset.imgs.keys())
        vidid = list(sub_dset.index.vidid_to_gids.keys())[0]
        gids = sub_dset.images(vidid=vidid).gids

    _heatmaps = build_heatmaps(sub_dset,
                               gids, {'fg': key},
                               skipped='interpolate',
                               resolution=resolution)['fg']

    def _process(track):

        # TODO when bounds are time-varying, this lets individual frames
        # go outside them; only enforces the union. Problem?
        # currently bounds come from site summaries, which are not
        # time-varying.
        if track is None:
            track_bounds = None
            _heatmaps_in_track = _heatmaps
        else:
            track_bounds = track['poly'].unary_union
            track_gids = track['gid']
            flags = np.in1d(gids, track_gids)
            _heatmaps_in_track = np.compress(flags, _heatmaps, axis=0)

        # this is another hot spot, heatmaps_to_polys -> mask_to_polygons ->
        # rasterize. Figure out how to vectorize over bounds.
        track_polys = heatmaps_to_polys(_heatmaps_in_track, track_bounds,
                                        agg_fn, thresh, morph_kernel,
                                        thresh_hysteresis, norm_ord,
                                        moving_window_size)
        if track is None:
            yield from zip(itertools.repeat(gids), track_polys)
            # for poly in polys:  # convert to shapely to check this
            # if poly.is_valid and not poly.is_empty:
            # yield (gids, poly)
        else:
            poly = unary_union([p.to_shapely() for p in track_polys])

            if poly.is_valid and not poly.is_empty:

                yield (track['gid'], kwimage.MultiPolygon.from_shapely(poly))

    # no benefit so far
    exc = ub.Executor('serial', max_workers=8)
    jobs = []
    for _, track in boundary_tracks:
        jobs.append(exc.submit(_process, track))

    result_gen = itertools.chain.from_iterable(j.result() for j in jobs)
    result_gen = list(result_gen)
    return result_gen

#
# --- wrappers ---
#
# Note:
#     The following are valid choices of `track_fn` in
#     ../../cli/kwcoco_to_geojson.py and will be called by ./normalize.py


__devnote__ = """

See Also kwcoco_to_geojson.KWCocoToGeoJSONConfig comment

TODO:
    it may make sense to change this into a scriptconfig.DataConfig in
    order to provide richer introspection to tools that want to know
    what parameters are available.

The following are the common and differing settings between BAS / SC

I include some candidate scfg logic that I may implement

COMMON:

from scriptconfig import DataConfig
from scriptconfig import Value as V

# Note that the V wrapping can/will be optional
# Do something like __defaults__ = Common.__defaults__  # Can do this with a metaclass

class CommonTrackerConfig(DataConfig):
    agg_fn                     : V[str]           = V('probs', help='agg method')
    max_area_behavior          : str              = 'drop'
    morph_kernel               : V[int]           = V(3, help='...')
    moving_window_size         : V[int | None]    = V(None, help='...')
    norm_ord                   : V[int | str]     = V(1, help='...')
    polygon_simplify_tolerance : V[float | None]  = V(None, help='...')
    resolution                 : V[str | None]    = V(None, help='...')
    response_thresh            : Optional[float]  = None
    thresh_hysteresis          : Optional[float]  = None


class BAS_TrackerConfig(CommonTrackerConfig):

    # Common but different defaults
    key: str = 'salient'
    max_area_sqkm: Optional[float] = 2.25
    min_area_sqkm: Optional[float] = 0.072
    time_thresh: Optional[float] = 1
    thresh: float = 0.2


VALID_BOUNDRY_ALGOS = Literal['bounds', 'polys', 'none']
BACKGROUND_NAMES = CNAMES_DCT['negative']['scored']

class SC_TrackerConfig(CommonTrackerConfig):

    # Common but different defaults
    key           : Tuple[str]      = tuple(CNAMES_DCT['positive']['scored'])
    max_area_sqkm : Optional[float] = None
    min_area_sqkm : Optional[float] = None
    thresh        : float           = 0.01
    time_thresh   : Optional[float] = None

    # Unique to SC
    boundaries_as : VALID_BOUNDRY_ALGOS = 'bounds'
    bg_key        : Tuple[str]          = tuple(BACKGROUND_NAMES)
"""


@dataclass
class TimeAggregatedBAS(NewTrackFunction):
    '''
    Wrapper for BAS that looks for change heatmaps.
    '''
    thresh: float = 0.2
    morph_kernel: int = 3
    time_thresh: Optional[float] = 1
    response_thresh: Optional[float] = None
    key: str = 'salient'
    norm_ord: Optional[Union[int, str]] = 1
    agg_fn: str = 'probs'
    thresh_hysteresis: Optional[float] = None
    moving_window_size: Optional[int] = None
    min_area_sqkm: Optional[float] = 0.072
    max_area_sqkm: Optional[float] = 2.25
    max_area_behavior: str = 'drop'
    polygon_simplify_tolerance: Union[None, float] = None
    resolution: Optional[str] = None

    def create_tracks(self, sub_dset):
        aggkw = ub.compatible(self.__dict__, time_aggregated_polys)
        tracks = time_aggregated_polys(sub_dset, **aggkw)
        return tracks

    def add_tracks_to_dset(self, sub_dset, tracks):
        sub_dset = _add_tracks_to_dset(sub_dset, tracks, self.thresh, self.key)
        return sub_dset


@dataclass
class TimeAggregatedSC(NewTrackFunction):
    '''
    Wrapper for Site Characterization that looks for phase heatmaps.

    Alias: class_heatmaps

    Note:
        This is a valid choice of `track_fn` in ../../cli/kwcoco_to_geojson.py
    '''
    thresh: float = 0.01
    morph_kernel: int = 3
    time_thresh: Optional[float] = None
    response_thresh: Optional[float] = None
    key: Tuple[str] = tuple(CNAMES_DCT['positive']['scored'])
    bg_key: Tuple[str] = tuple(CNAMES_DCT['negative']['scored'])
    boundaries_as: Literal['bounds', 'polys', 'none'] = 'bounds'
    norm_ord: Optional[Union[int, str]] = 1
    agg_fn: str = 'probs'
    thresh_hysteresis: Optional[float] = None
    moving_window_size: Optional[int] = None
    min_area_sqkm: Optional[float] = None
    max_area_sqkm: Optional[float] = None
    max_area_behavior: str = 'drop'
    polygon_simplify_tolerance: Union[None, float] = None
    resolution: Optional[str] = None

    def create_tracks(self, sub_dset):
        '''
        boundaries_as: use for Site Boundary annots in coco_dset
            'bounds': generated polys will lie inside the boundaries
            'polys': generated polys will be the boundaries
            'none': generated polys will ignore the boundaries
        '''
        # TODO last use of Track here
        if self.boundaries_as == 'polys':
            tracks = pop_tracks(
                sub_dset,
                cnames=[SITE_SUMMARY_CNAME],
                # these are SC scores, not BAS, so this is not a
                # true reproduction of hybrid.
                score_chan=kwcoco.ChannelSpec('|'.join(self.key)))
            # hack in always-foreground instead
            # tracks[(score_chan, None)] = 1

            # try to ignore this error
            tracks['poly'] = tracks['poly'].map(
                kwimage.MultiPolygon.from_shapely)

        else:
            aggkw = ub.compatible(self.__dict__, time_aggregated_polys)
            aggkw['use_boundaries'] = aggkw.get('boundaries_as', 'none') != 'none'
            tracks = time_aggregated_polys(sub_dset, **aggkw)
        return tracks

    def add_tracks_to_dset(self, sub_dset, tracks, **kwargs):
        if self.boundaries_as != 'polys':
            col_map = {}
            for c in tracks.columns:
                if c[0] == 'fg':
                    k = kwcoco.ChannelSpec('|'.join(self.key)).spec
                    col_map[c] = (k, *c[1:])
                elif c[0] == 'bg':
                    k = kwcoco.ChannelSpec('|'.join(self.bg_key)).spec
                    col_map[c] = (k, *c[1:])
            # weird effect here - reassignment casts from GeoDataFrame to
            # DataFrame. Related to invalid geometry column?
            # tracks = tracks.rename(columns=col_map)
            tracks.rename(columns=col_map, inplace=True)
        sub_dset = _add_tracks_to_dset(sub_dset, tracks, self.thresh, self.key,
                                       self.bg_key, **kwargs)

        return sub_dset
