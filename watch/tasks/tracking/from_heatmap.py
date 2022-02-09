from watch.utils import kwcoco_extensions
from watch.utils import util_kwimage
from watch.heuristics import SITE_SUMMARY_CNAME, CNAMES_DCT
import kwarray
import kwimage
import kwcoco
import numpy as np
import ubelt as ub
import itertools
from typing import Iterable, Tuple, Set, Union, Optional, Literal
from dataclasses import dataclass
from watch.tasks.tracking.utils import (Track, PolygonFilter, NewTrackFunction,
                                        mask_to_polygons, heatmap, score, Poly,
                                        CocoDsetFilter, _validate_keys,
                                        Observation, pop_tracks, heatmaps)


try:
    from xdev import profile
except Exception:
    profile = ub.identity


@dataclass
class SmallPolygonFilter(PolygonFilter):
    min_area_px: float = 80

    def on_augmented_polys(self, aug_polys):
        for aug, poly in aug_polys:
            if poly.to_shapely().area > self.min_area_px:
                yield aug, poly


class TimePolygonFilter(CocoDsetFilter):
    def get_poly_time_ind(self, gids_polys: Iterable[Tuple[int, Poly]]):
        """
        Given a potential track, compute index of the first match of the track
        with its mask.
        Mask is computed by comparing heatmaps with threshold.
        """
        for image_ind, (gid, poly) in enumerate(gids_polys):
            try:
                overlap = self.score(poly,
                                     gid,
                                     mode='overlap',
                                     threshold=self.threshold)
                if overlap > 0.5:
                    return image_ind
            except AssertionError as e:
                print(f'image {gid} does not have all predictions: {e}')

        return None  # TODO error handling

    def on_observations(self, observations):
        start_idx = self.get_poly_time_ind(
            map(lambda o: (o.gid, o.poly), observations))
        end_idx = self.get_poly_time_ind(
            map(lambda o: (o.gid, o.poly), reversed(observations)))
        len_obs = sum(1 for _ in observations)
        # have to make sure this doesn't get consumed
        return list(
            itertools.islice(observations, start_idx, len_obs - end_idx))

    def on_augmented_polys(self, aug_polys):
        raise NotImplementedError('need gids for time filtering')


class ResponsePolygonFilter(CocoDsetFilter):
    '''
    Filters each track based on the average response of all tracks.
    '''
    mean_response: float
    gids: Set[int] = {}

    def __init__(self, tracks: Iterable[Track], key, threshold=0.001):

        self.threshold = threshold
        self.key = key

        dsets = {track.dset for track in tracks}
        assert len(dsets) == 1, 'Tracks refer to different CocoDatasets!'
        self.dset = dsets.pop()

        self.gids = {}
        all_responses = kwarray.RunningStats()
        for track in tracks:  # could disambiguate these for better stats
            for obs in track.observation:
                all_responses.update(np.array(self.response(obs.poly,
                                                            obs.gid)))
                self.gids.add(obs.gid)
        self.mean_response = all_responses.summarize(keepdims=False)['mean']

    def response(self, poly, gid):
        return self.score(poly, gid, mode='response')

    def on_augmented_polys(self, aug_polys, gids=None, threshold=None):
        '''
        Mode for filtering each poly against each gid (cross product)
        '''
        if gids is None:
            gids = self.gids
        if threshold is None:
            threshold = self.threshold
        for aug, poly in aug_polys:
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
            if self.response(obs.poly,
                             obs.gid) / self.mean_response > threshold:
                yield obs


@profile
def add_tracks_to_dset(coco_dset,
                       tracks,
                       thresh,
                       key,
                       bg_key=None,
                       coco_dset_sc=None):
    '''
    Add tracks to coco_dset using the categories/heatmaps from coco_dset_sc.
    '''
    key, bg_key = _validate_keys(key, bg_key)
    if coco_dset_sc is None:
        coco_dset_sc = coco_dset

    @ub.memoize
    def _heatmap(gid, key, space):
        probs_tot, probs_dct = heatmap(coco_dset_sc,
                                       gid,
                                       key,
                                       return_chan_probs=True,
                                       space=space)
        return probs_dct

    @ub.memoize
    def _warp_img_from_vid(gid):
        # Memoize the conversion to a matrix
        img = coco_dset_sc.imgs[gid]
        vid_from_img = kwimage.Affine.coerce(img['warp_img_to_vid'])
        img_from_vid = vid_from_img.inv()
        return img_from_vid

    def make_new_annotation(gid, poly, this_score, track_id, space='video'):

        # assign category (key) from max score
        if this_score > thresh or len(bg_key) == 0:
            cand_keys = key
        else:
            cand_keys = bg_key
        if len(cand_keys) > 1:
            cand_scores = [
                score(poly, probs)  # awk, this could be a class
                for probs in _heatmap(gid, key, space).values()
            ]
            cat_name = cand_keys[np.argmax(cand_scores)]
        else:
            cat_name = cand_keys[0]
        cid = coco_dset.ensure_category(cat_name)

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

    new_trackids = kwcoco_extensions.TrackidGenerator(coco_dset)

    for track in tracks:
        if track.track_id is not None:
            track_id = track.track_id
            new_trackids.exclude_trackids([track_id])
        else:
            track_id = next(new_trackids)

        new_anns = []
        for obs in track.observations:
            new_ann = make_new_annotation(obs.gid, obs.poly, obs.score,
                                          track_id)
            new_anns.append(new_ann)

        for new_ann in new_anns:
            coco_dset.add_annotation(**new_ann)
        # TODO: Faster to add annotations in bulk, but we need to construct the
        # "ids" first
        # coco_dset.add_annotations(new_anns)

    return coco_dset


def time_aggregated_polys(coco_dset,
                          thresh=0.15,
                          morph_kernel=3,
                          key='salient',
                          bg_key=None,
                          time_filtering=False,
                          response_filtering=False,
                          use_boundaries=False,
                          norm_ord=1):
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

        morph_kernel (int): height/width in px of close or dilate kernel

        norm_ord: order of norm to aggregate heatmap pixels across time.
            1: average [default]
            2: euclidean
            0: sum
            np.inf, 'inf', or None: max

    Example:
        >>> # test interpolation
        >>> from watch.tasks.tracking.from_heatmap import time_aggregated_polys
        >>> from watch.demo import demo_kwcoco_with_heatmaps
        >>> d = demo_kwcoco_with_heatmaps(num_frames=5, image_size=(480, 640))
        >>> orig_track = time_aggregated_polys(d)[0].observations
        >>> skip_gids = [1,3]
        >>> for gid in skip_gids:
        >>>      # remove salient channel
        >>>      d.imgs[gid]['auxiliary'].pop()
        >>> inter_track = time_aggregated_polys(d)[0].observations
        >>> assert inter_track[0].score == 0, inter_track[1].score > 0
    '''

    #
    # --- input validation ---
    #

    key, bg_key = _validate_keys(key, bg_key)
    _all_keys = set(key + bg_key)
    has_requested_chans_list = []
    for gid in coco_dset.imgs:
        coco_img = coco_dset.coco_image(gid)
        chan_codes = coco_img.channels.normalize().fuse().as_set()
        flag = bool(_all_keys & chan_codes)
        has_requested_chans_list.append(flag)

    if not any(has_requested_chans_list):
        raise KeyError(f'no imgs in dset {coco_dset.tag} '
                       f'have keys {key} or {bg_key}.')
    if not all(has_requested_chans_list):
        n_missing = (len(has_requested_chans_list) -
                     sum(has_requested_chans_list))
        print(f'warning: {n_missing} imgs in dset {coco_dset.tag} '
              f'have no keys {key} or {bg_key}. Interpolating...')

    if norm_ord in {'inf', None}:
        norm_ord = np.inf

    #
    # --- utilities ---
    #

    # turn heatmaps into polygons
    def probs(heatmaps):
        probs = np.linalg.norm(np.stack(heatmaps, axis=0), norm_ord, axis=0)
        if 0 < norm_ord < np.inf:
            probs /= np.power(len(heatmaps), 1/norm_ord)

        hard_probs = util_kwimage.morphology(probs > thresh, 'dilate',
                                             morph_kernel)
        modulated_probs = probs * hard_probs

        return modulated_probs

    def tracks_polys_bounds() -> Iterable[Tuple[Track, Poly]]:
        import shapely.ops
        boundary_tracks = list(pop_tracks(coco_dset, [SITE_SUMMARY_CNAME]))
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
        _heatmaps = heatmaps(coco_dset,
                             gids, {'fg': key},
                             skipped='interpolate')['fg']

        def fill_boundary_track(track) -> Optional[Tuple[Track, Poly]]:
            # TODO when bounds are time-varying, this lets individual frames
            # go outside them; only enforces the union. Problem?
            # currently bounds come from site summaries, which are not
            # time-varying.
            track_bounds = shapely.ops.unary_union(
                [obs.poly.to_shapely() for obs in track.observations])
            _heatmaps_in_track = np.compress(
                np.in1d(gids, [obs.gid for obs in track.observations]),
                _heatmaps,
                axis=0)

            track_polys = mask_to_polygons(probs(_heatmaps_in_track),
                                           thresh,
                                           bounds=track_bounds)
            poly = shapely.ops.unary_union(
                [p.to_shapely() for p in track_polys])
            if poly.is_valid and not poly.is_empty:
                poly = kwimage.MultiPolygon.from_shapely(poly)
                out_track = Track(
                    [
                        Observation(
                            poly=poly,
                            gid=obs.gid,
                            score=score(
                                poly,
                                # TODO optimize .index()
                                _heatmaps[gids.index(obs.gid)]))
                        for obs in track.observations
                    ],
                    dset=coco_dset,
                    track_id=track.track_id)
                return out_track, poly

        print('generating polys in bounds: number of bounds: ',
              len(boundary_tracks))
        return list(filter(None, map(fill_boundary_track, boundary_tracks)))

    def tracks_polys_nobounds() -> Iterable[Tuple[Track, Poly]]:
        gids = list(coco_dset.imgs.keys())
        _heatmaps = heatmaps(coco_dset,
                             gids, {'fg': key},
                             skipped='interpolate')['fg']

        polys = list(mask_to_polygons(probs(_heatmaps), thresh))

        # turn each polygon into a list of polygons (map them across gids)
        tracks = [
            Track.from_polys(itertools.repeat(poly),
                             coco_dset,
                             probs=_heatmaps) for poly in polys
        ]

        return list(zip(tracks, polys))

    #
    # --- main logic ---
    #

    if use_boundaries:
        tracks_polys = tracks_polys_bounds()
    else:
        tracks_polys = tracks_polys_nobounds()

    print('time aggregation: number of polygons: ', len(tracks_polys))

    # SmallPolygonFilter and ResponsePolygonFilter should operate on each
    # vidpoly separately, so have to bookkeep both vidpolys and tracks
    # in a list track_polys

    tracks_polys = list(SmallPolygonFilter(min_area_px=80)(tracks_polys))
    print('removed small: remaining polygons: ', len(tracks_polys))

    if response_filtering:
        response_thresh = 0.0002  # 0.0005
        tracks_polys = list(
            ResponsePolygonFilter([t for t, _ in tracks_polys], key,
                                  response_thresh)(tracks_polys))
        print('after filtering based on per-polygon response ',
              len(tracks_polys))

    # TimePolygonFilter edits tracks instead of removing them, so we can
    # discard 'polys' and focus on 'tracks'
    tracks = [t for t, _ in tracks_polys]
    if time_filtering:
        # TODO investigate different thresh here
        time_thresh = thresh
        time_filter = TimePolygonFilter(coco_dset, tuple(key), time_thresh)
        tracks = list(
            filter(lambda track: len(list(track.observations)) > 0,
                   map(time_filter, tracks)))

    return tracks


@dataclass
class TimeAggregatedBAS(NewTrackFunction):
    '''
    Wrapper for BAS that looks for change heatmaps.
    '''
    thresh: float = 0.3
    morph_kernel: int = 3
    time_filtering: bool = True
    response_filtering: bool = False
    key: str = 'salient'
    norm_ord: Optional[Union[int, str]] = 1

    def create_tracks(self, coco_dset):
        tracks = time_aggregated_polys(
            coco_dset,
            self.thresh,
            self.morph_kernel,
            key=self.key,
            time_filtering=self.time_filtering,
            response_filtering=self.response_filtering,
            norm_ord=self.norm_ord)
        return tracks

    def add_tracks_to_dset(self, coco_dset, tracks):
        coco_dset = add_tracks_to_dset(coco_dset, tracks, self.thresh,
                                       self.key)
        return coco_dset


@dataclass
class TimeAggregatedSC(NewTrackFunction):
    '''
    Wrapper for Site Characterization that looks for phase heatmaps.
    '''
    thresh: float = 0.01
    morph_kernel: int = 3
    time_filtering: bool = False
    response_filtering: bool = False
    key: Tuple[str] = tuple(CNAMES_DCT['positive']['scored'])  # TODO unscored?
    bg_key: Tuple[str] = ('No Activity')  # TODO other negative classes?
    boundaries_as: Literal['bounds', 'polys', 'none'] = 'bounds'
    norm_ord: Optional[Union[int, str]] = 1

    def create_tracks(self, coco_dset):
        '''
        boundaries_as: use for Site Boundary annots in coco_dset
            'bounds': generated polys will lie inside the boundaries
            'polys': generated polys will be the boundaries
            'none': generated polys will ignore the boundaries
        '''
        if self.boundaries_as == 'polys':
            tracks = pop_tracks(
                coco_dset,
                cnames=[SITE_SUMMARY_CNAME],
                # these are SC scores, not BAS, so this is not a
                # true reproduction of hybrid.
                score_chan=kwcoco.ChannelSpec('|'.join(self.key)))
            # hack in always-foreground instead
            if 0:  # TODO
                for track in list(tracks):
                    for obs in track.observations:
                        obs.score = 1

            tracks = list(filter(
                lambda track: len(list(track.observations)) > 0,
                tracks))
        else:
            tracks = time_aggregated_polys(
                coco_dset,
                self.thresh,
                self.morph_kernel,
                key=self.key,
                bg_key=self.bg_key,
                time_filtering=self.time_filtering,
                response_filtering=self.response_filtering,
                use_boundaries=(self.boundaries_as != 'none'),
                norm_ord=self.norm_ord)

        return tracks

    def add_tracks_to_dset(self, coco_dset, tracks, **kwargs):
        coco_dset = add_tracks_to_dset(coco_dset, tracks, self.thresh,
                                       self.key, self.bg_key, **kwargs)

        return coco_dset


@dataclass
class TimeAggregatedHybrid(NewTrackFunction):
    '''
    This method uses predictions from a BAS model to generate polygons.
    Predicted heatmaps from a Site Characterization model are used to assign
    activity label to every polygon.
    coco_dset: KWCOCO file with BAS predictions
    coco_dset_sc: KWCOCO file with site characterization predictions
    '''
    coco_dset_sc: Union[str, kwcoco.CocoDataset]

    def __post_init__(self):
        if isinstance(self.coco_dset_sc, str):
            self.coco_dset_sc = kwcoco.CocoDataset.coerce(self.coco_dset_sc)

    def create_tracks(self, coco_dset):
        return TimeAggregatedBAS().create_tracks(coco_dset)

    def add_tracks_to_dset(self, coco_dset, tracks):
        return TimeAggregatedSC(use_boundary_annots=False).add_tracks_to_dset(
            coco_dset, tracks, coco_dset_sc=self.coco_dset_sc)

    def safe_apply(self, coco_dset, gids, overwrite):
        '''
        Handle subsetting coco_dset_sc at the same time as coco_dset
        '''
        tmp = self.coco_dset_sc.copy()
        self.coco_dset_sc = self.safe_partition(self.coco_dset_sc,
                                                gids,
                                                remove=False)
        # TODO this might not call self.add_tracks_to_dset as intended
        result = super().safe_apply(coco_dset, gids, overwrite)
        self.coco_dset_sc = tmp
        return result
