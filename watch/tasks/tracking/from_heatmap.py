from watch.utils import kwcoco_extensions
from watch.heuristics import SITE_SUMMARY_CNAME, CNAMES_DCT
import kwarray
import kwimage
import kwcoco
import numpy as np
import ubelt as ub
import itertools
from typing import Iterable, Tuple, Union, Optional, Literal
from dataclasses import dataclass
from shapely.ops import unary_union
from watch.tasks.tracking.utils import (
    Track, NewTrackFunction, mask_to_polygons, build_heatmap,
    score_poly, Poly, CocoDsetFilter, _validate_keys, Observation, pop_tracks,
    build_heatmaps, trackid_is_default)

try:
    from xdev import profile
except Exception:
    profile = ub.identity

#
# --- aggregation functions for heatmaps ---
#


def _norm(heatmaps, norm_ord):
    heatmaps = np.array(heatmaps)
    probs = np.linalg.norm(heatmaps, norm_ord, axis=0)
    if 0 < norm_ord < np.inf:
        probs /= np.power(len(heatmaps), 1 / norm_ord)
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


class TimePolygonFilter(CocoDsetFilter):

    def get_poly_time_ind(self, gids_polys: Iterable[Tuple[int, Poly]]):
        """
        Given a potential track, compute index of the first match of the track
        with its mask.
        Mask is computed by comparing heatmaps with threshold.
        """
        found = None
        magic_thresh = 0.5
        for image_ind, (gid, poly) in enumerate(gids_polys):
            try:
                overlap = self.score(poly,
                                     gid,
                                     threshold=self.threshold)
                if overlap > magic_thresh:
                    found = image_ind
                    break
            except AssertionError as e:
                print(f'image {gid} does not have all predictions: {e}')

        # return None  # TODO error handling
        return found

    def on_observations(self, observations):
        observations = list(observations)
        len_obs = len(observations)
        gids_polys = [(o.gid, o.poly) for o in observations]
        start_idx = self.get_poly_time_ind(gids_polys)
        if start_idx is None:
            return []
        rev_end_idx = self.get_poly_time_ind(reversed(gids_polys))
        end_idx = len_obs - rev_end_idx
        return observations[start_idx:end_idx]

    def on_augmented_polys(self, aug_polys):
        raise NotImplementedError('need gids for time filtering')


# TODO memoize or gpd
class ResponsePolygonFilter:
    '''
    Filters each track based on the average response of all tracks.
    '''

    def __init__(self, tracks: Iterable[Track], key, threshold=0.001):

        dsets = {track.dset for track in tracks}
        assert len(dsets) == 1, 'Tracks refer to different CocoDatasets!'
        dset = dsets.pop()

        self.dset = dset
        self.key = key
        self.threshold = threshold

        gids = set()
        all_responses = kwarray.RunningStats()
        for track in tracks:  # could disambiguate these for better stats
            for obs in track.observations:
                rsp = np.array(self.response(obs.poly, obs.gid))
                all_responses.update(rsp)
                gids.add(obs.gid)
        mean_response = all_responses.summarize(keepdims=False)['mean']

        self.gids = gids
        self.mean_response = mean_response

    def response(self, poly, gid):
        return score_poly(poly, build_heatmap(self.dset, gid, self.key))

    def on_augmented_polys(self, aug_polys, gids=None, threshold=None):
        '''
        Mode for filtering each poly against each gid (cross product)
        '''
        if gids is None:
            gids = self.gids
        if threshold is None:
            threshold = self.threshold
        for aug, poly in aug_polys:
            # TODO nanmean
            this_response = np.mean([self.response(poly, gid) for gid in gids])
            if this_response / self.mean_response > threshold:
                yield aug, poly

    def on_observations(self, observations, threshold=None):
        '''
        Mode for filtering each poly against only its associated gid
        '''
        if threshold is None:
            threshold = self.threshold
        for obs in observations:
            if self.response(obs.poly, obs.gid) / self.mean_response > threshold:
                yield obs


#
# --- main logic ---
#


@profile
def add_tracks_to_dset(sub_dset,
                       tracks,
                       thresh,
                       key,
                       bg_key=None,
                       coco_dset_sc=None):
    '''
    Add tracks to sub_dset using the categories/heatmaps from coco_dset_sc.
    '''
    key, bg_key = _validate_keys(key, bg_key)
    if coco_dset_sc is None:
        coco_dset_sc = sub_dset

    @ub.memoize
    def _heatmap(gid, key, space):
        probs_tot, probs_dct = build_heatmap(
            coco_dset_sc, gid, key, return_chan_probs=True, space=space)
        return probs_dct

    @ub.memoize
    def _warp_img_from_vid(gid):
        # Memoize the conversion to a matrix
        coco_img = coco_dset_sc.coco_image(gid)
        img_from_vid = coco_img.warp_img_from_vid
        return img_from_vid

    def make_new_annotation(gid, poly, this_score, track_id, space='video'):

        # assign category (key) from max score
        if this_score > thresh or len(bg_key) == 0:
            cand_keys = key
        else:
            cand_keys = bg_key
        if len(cand_keys) > 1:
            cand_scores = [
                score_poly(poly, probs)  # awk, this could be a class
                for probs in _heatmap(gid, key, space).values()
            ]
            cat_name = cand_keys[np.argmax(cand_scores)]
        else:
            cat_name = cand_keys[0]
        cid = sub_dset.ensure_category(cat_name)

        assert space in {'image', 'video'}
        if space == 'video':
            # Transform the video polygon into image space
            img_from_vid = _warp_img_from_vid(gid)
            poly = poly.warp(img_from_vid)

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
    for track in tracks:
        if not trackid_is_default(track.track_id):
            track_id = track.track_id
            new_trackids.exclude_trackids([track_id])
        else:
            track_id = next(new_trackids)

        for obs in track.observations:
            new_ann = make_new_annotation(obs.gid, obs.poly, obs.score,
                                          track_id)
            all_new_anns.append(new_ann)

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
def time_aggregated_polys(sub_dset,
                          thresh,
                          morph_kernel=3,
                          key='salient',
                          bg_key=None,
                          time_filtering=False,
                          response_thresh=None,
                          use_boundaries=False,
                          norm_ord=1,
                          agg_fn='probs',
                          moving_window_size=None,  # 150
                          min_area_sqkm=0.072,  # 80px@30GSD
                          # min_area_sqkm=0.018,  # 80px@15GSD
                          # min_area_sqkm=0.008,  # 80px@10GSD
                          max_area_sqkm=None,
                          # max_area_sqkm=0.5,  # ~1/4 of KR_R002 vid area
                          max_area_behavior='drop',
                          thresh_hysteresis=None):
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

    Example:
        >>> # test interpolation
        >>> from watch.tasks.tracking.from_heatmap import time_aggregated_polys
        >>> from watch.demo import demo_kwcoco_with_heatmaps
        >>> sub_dset = demo_kwcoco_with_heatmaps(num_frames=5, image_size=(480, 640))
        >>> orig_track = time_aggregated_polys(sub_dset, thresh=0.15)[0].observations
        >>> skip_gids = [1,3]
        >>> for gid in skip_gids:
        >>>      # remove salient channel
        >>>      sub_dset.imgs[gid]['auxiliary'].pop()
        >>> inter_track = time_aggregated_polys(sub_dset, thresh=0.15)[0].observations
        >>> assert inter_track[0].score == 0, inter_track[1].score > 0
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
    vidname = video.get('name', None)

    for gid in sub_dset.imgs:
        coco_img = sub_dset.coco_image(gid)
        chan_codes = coco_img.channels.normalize().fuse().as_set()
        flag = bool(_all_keys & chan_codes)
        has_requested_chans_list.append(flag)

    if not any(has_requested_chans_list):
        raise KeyError(f'no imgs in dset {sub_dset.tag} '
                       f'have keys {key} or {bg_key}.')
    if not all(has_requested_chans_list):
        n_total = len(has_requested_chans_list)
        n_have = sum(has_requested_chans_list)
        n_missing = (n_total - n_have)
        print(f'warning: {n_missing} / {n_total} imgs in dset {sub_dset.tag} '
              f'with video {vidname} have no keys {key} or {bg_key}. '
               'Interpolating...')

    if norm_ord in {'inf', None}:
        norm_ord = np.inf

    #
    # --- main logic ---
    #

    if use_boundaries:
        tracks_polys = tracks_polys_bounds(
            sub_dset, key, agg_fn, thresh, morph_kernel, thresh_hysteresis,
            norm_ord, moving_window_size)
    else:
        tracks_polys = tracks_polys_nobounds(
            sub_dset, key, agg_fn, thresh, morph_kernel, thresh_hysteresis,
            norm_ord, moving_window_size)

    print('time aggregation: number of polygons: ', len(tracks_polys))

    # size and response filters should operate on each vidpoly separately, so
    # have to bookkeep both vidpolys and tracks in a list track_polys

    video_gsd = video.get('target_gsd', None)
    if video_gsd is None:
        default_gsd = 30
        video_gsd = default_gsd
        print(f'warning: video {video["name"]} in dset {sub_dset.tag} '
               'has no listed GSD; assuming {default_gsd}')

    if min_area_sqkm:
        min_area_px = min_area_sqkm * 1e6 / (video_gsd ** 2)
        n_orig = len(tracks_polys)
        tracks_polys = [(t, p) for t, p in tracks_polys if p.to_shapely().area > min_area_px]
        print(f'removed small: remaining polygons: {len(tracks_polys)} / {n_orig}')

    if max_area_sqkm:
        max_area_px = max_area_sqkm * 1e6 / (video_gsd ** 2)
        n_orig = len(tracks_polys)
        if max_area_behavior == 'drop':
            tracks_polys = [(t, p) for t, p in tracks_polys if p.to_shapely().area < max_area_px]
            print(f'removed large: remaining polygons: {len(tracks_polys)} / {n_orig}')
        elif max_area_behavior == 'grid':
            # edits tracks instead of removing them
            raise NotImplementedError


    # response_thresh = 0.9
    if response_thresh:
        rsp_filter = ResponsePolygonFilter(
            [t for t, _ in tracks_polys], key, response_thresh)
        tracks_polys = list(rsp_filter.on_augmented_polys(tracks_polys))
        print('after filtering based on per-polygon response {len(track_polys)} / {n_orig}')

    # TimePolygonFilter edits tracks instead of removing them, so we can
    # discard 'polys' and focus on 'tracks'
    tracks = [t for t, _ in tracks_polys]
    if time_filtering:
        # TODO investigate different thresh here
        time_thresh = thresh
        time_filter = TimePolygonFilter(sub_dset, tuple(key), time_thresh)
        _filtered = []
        for _, t in enumerate(tracks):
            _t = time_filter(t)
            _filtered.append(_t)
        _filtered = list(map(time_filter, tracks))
        tracks = [t for t in _filtered if len(list(t.observations)) > 0]

    return tracks


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
                    # warnings.warn('Found two intersecting polygons where the union was a multipolygon')
                    merged_polys.extend(list(combo.geoms))
                else:
                    raise AssertionError(f'Unexpected type {combo.type} from {_p1} and {_p2}')

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
        aggregated = _agg_fn(heatmaps, thresh=thresh, morph_kernel=morph_kernel,
                             norm_ord=norm_ord)

        polygons = list(mask_to_polygons(aggregated, thresh,
                                         thresh_hysteresis=thresh_hysteresis,
                                         bounds=bounds))
        return polygons

    # calculate number of moving-window steps, based on window_size and number of heatmaps
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


def tracks_polys_bounds(sub_dset,
                        key,
                        agg_fn,
                        thresh,
                        morph_kernel,
                        thresh_hysteresis,
                        norm_ord,
                        moving_window_size=None,  # 150
                       ) -> Iterable[Tuple[Track, Poly]]:
    import shapely.ops
    boundary_tracks = list(pop_tracks(sub_dset, [SITE_SUMMARY_CNAME]))
    assert len(boundary_tracks) > 0, 'need valid site boundaries!'
    '''
    # TODO these obnoxious fors will be removed with gpd support in Track
    # unused
    bounds = shapely.ops.unary_union(
        list(
            itertools.chain.from_iterable(
                [obs.poly for obs in track.observations]
                for track in boundary_tracks)))
    '''
    gids = list(
        np.unique(
            np.concatenate([[obs.gid for obs in track.observations]
                            for track in boundary_tracks])))
    _heatmaps = build_heatmaps(
        sub_dset,
        gids, {'fg': key},
        skipped='interpolate')['fg']

    def fill_boundary_track(track) -> Optional[Tuple[Track, Poly]]:
        # TODO when bounds are time-varying, this lets individual frames
        # go outside them; only enforces the union. Problem?
        # currently bounds come from site summaries, which are not
        # time-varying.
        track_bounds = shapely.ops.unary_union(
            [obs.poly.to_shapely() for obs in track.observations])
        _heatmaps_in_track = np.compress(np.in1d(
            gids, [obs.gid for obs in track.observations]),
                                         _heatmaps,
                                         axis=0)

        bounds = track_bounds
        track_polys = heatmaps_to_polys(_heatmaps_in_track,
                                     bounds,
                                     agg_fn,
                                     thresh, morph_kernel,
                                     thresh_hysteresis,
                                     norm_ord,
                                     moving_window_size)

        poly = shapely.ops.unary_union(
            [p.to_shapely() for p in track_polys])
        if poly.is_valid and not poly.is_empty:
            poly = kwimage.MultiPolygon.from_shapely(poly)
            out_track = Track(
                [
                    Observation(
                        poly=poly,
                        gid=obs.gid,
                        score=score_poly(
                            poly,
                            # TODO optimize .index()
                            _heatmaps[gids.index(obs.gid)]))
                    for obs in track.observations
                ],
                dset=sub_dset,
                track_id=track.track_id)
            return out_track, poly

    print('generating polys in bounds: number of bounds: ',
          len(boundary_tracks))
    return list(filter(None, map(fill_boundary_track, boundary_tracks)))


def tracks_polys_nobounds(sub_dset,
                          key,
                          agg_fn,
                          thresh,
                          morph_kernel,
                          thresh_hysteresis,
                          norm_ord,
                          moving_window_size=None,  # 150
                         ) -> Iterable[Tuple[Track, Poly]]:
    gids = list(sub_dset.imgs.keys())
    keys = {'fg': key}
    skipped = 'interpolate'
    heatmaps = build_heatmaps(sub_dset, gids, keys, skipped)['fg']

    bounds = None
    polys = heatmaps_to_polys(heatmaps, bounds, agg_fn, thresh, morph_kernel,
                               thresh_hysteresis, norm_ord, moving_window_size)

    # turn each polygon into a list of polygons (map them across gids)
    tracks = [Track.from_polys(itertools.repeat(poly), sub_dset, probs=heatmaps)
              for poly in polys]

    tracks_polys = list(zip(tracks, polys))

    return tracks_polys


#
# --- wrappers ---
#


@dataclass
class TimeAggregatedBAS(NewTrackFunction):
    '''
    Wrapper for BAS that looks for change heatmaps.
    '''
    thresh: float = 0.2
    morph_kernel: int = 3
    time_filtering: bool = True
    response_thresh: Optional[float] = None
    key: str = 'salient'
    norm_ord: Optional[Union[int, str]] = 1
    agg_fn: str = 'probs'
    thresh_hysteresis: Optional[float] = None
    moving_window_size: Optional[int] = None
    min_area_sqkm: Optional[float] = 0.072
    max_area_sqkm: Optional[float] = 0.5
    max_area_behavior: str = 'drop'
    response_thresh: Optional[float] = None

    def create_tracks(self, sub_dset):
        tracks = time_aggregated_polys(
            sub_dset,
            self.thresh,
            self.morph_kernel,
            key=self.key,
            time_filtering=self.time_filtering,
            response_thresh=self.response_thresh,
            norm_ord=self.norm_ord,
            agg_fn=self.agg_fn,
            thresh_hysteresis=self.thresh_hysteresis,
            moving_window_size=self.moving_window_size,
            min_area_sqkm=self.min_area_sqkm,
            max_area_sqkm=self.max_area_sqkm,
            max_area_behavior=self.max_area_behavior,
        )
        return tracks

    def add_tracks_to_dset(self, sub_dset, tracks):
        sub_dset = add_tracks_to_dset(sub_dset, tracks, self.thresh,
                                      self.key)
        return sub_dset


@dataclass
class TimeAggregatedSC(NewTrackFunction):
    '''
    Wrapper for Site Characterization that looks for phase heatmaps.

    Alias: class_heatmaps
    '''
    thresh: float = 0.01
    morph_kernel: int = 3
    time_filtering: bool = False
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
    response_thresh: Optional[float] = None

    def create_tracks(self, sub_dset):
        '''
        boundaries_as: use for Site Boundary annots in coco_dset
            'bounds': generated polys will lie inside the boundaries
            'polys': generated polys will be the boundaries
            'none': generated polys will ignore the boundaries
        '''
        if self.boundaries_as == 'polys':
            tracks = list(pop_tracks(
                sub_dset,
                cnames=[SITE_SUMMARY_CNAME],
                # these are SC scores, not BAS, so this is not a
                # true reproduction of hybrid.
                score_chan=kwcoco.ChannelSpec('|'.join(self.key))))
            # hack in always-foreground instead
            if 0:  # TODO
                for track in list(tracks):
                    for obs in track.observations:
                        obs.score = 1

            tracks = list(filter(
                lambda track: len(list(track.observations)) > 0, tracks))
        else:
            tracks = time_aggregated_polys(
                sub_dset,
                self.thresh,
                self.morph_kernel,
                key=self.key,
                bg_key=self.bg_key,
                time_filtering=self.time_filtering,
                response_thresh=self.response_thresh,
                use_boundaries=(self.boundaries_as != 'none'),
                norm_ord=self.norm_ord,
                agg_fn=self.agg_fn,
                thresh_hysteresis=self.thresh_hysteresis,
                moving_window_size=self.moving_window_size,
                min_area_sqkm=self.min_area_sqkm,
                max_area_sqkm=self.max_area_sqkm,
                max_area_behavior=self.max_area_behavior,
            )

        return tracks

    def add_tracks_to_dset(self, sub_dset, tracks, **kwargs):
        sub_dset = add_tracks_to_dset(sub_dset, tracks, self.thresh, self.key,
                                      self.bg_key, **kwargs)

        return sub_dset
