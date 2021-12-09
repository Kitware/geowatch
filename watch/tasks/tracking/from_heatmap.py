from watch.utils import kwcoco_extensions
from watch.utils import util_kwimage
from collections import defaultdict
import kwarray
import kwimage
import kwcoco
import numpy as np
import ubelt as ub
import itertools
from typing import Iterable, Tuple, Set
from dataclasses import dataclass
from watch.tasks.tracking.utils import (Track, PolygonFilter, NewTrackFunction,
                                        mask_to_polygons, heatmap, score, Poly,
                                        CocoDsetFilter, _validate_keys)


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
                map(lambda o: (o.gid, o.poly),  observations))
        end_idx = self.get_poly_time_ind(
                map(lambda o: (o.gid, o.poly), reversed(observations)))
        len_obs = sum(1 for _ in observations)
        return itertools.islice(observations, start_idx, len_obs - end_idx)

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

    def add_annotation(gid, poly, this_score, track_id, space='video'):
        assert space in {'image', 'video'}

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

        if space == 'video':
            # Transform the video polygon into image space
            img = coco_dset_sc.imgs[gid]
            vid_from_img = kwimage.Affine.coerce(img['warp_img_to_vid'])
            img_from_vid = vid_from_img.inv()
            poly = poly.warp(img_from_vid)

        bbox = list(poly.bounding_box().to_coco())[0]
        # Add the polygon as an annotation on the image
        coco_dset.add_annotation(image_id=gid,
                                 category_id=cid,
                                 bbox=bbox,
                                 segmentation=poly,
                                 score=this_score,
                                 track_id=track_id)

    new_trackids = kwcoco_extensions.TrackidGenerator(coco_dset)

    for track in tracks:
        track_id = next(new_trackids)
        for obs in track.observations:
            add_annotation(obs.gid, obs.poly, obs.score, track_id)

    return coco_dset


def time_aggregated_polys(coco_dset,
                          thresh=0.15,
                          morph_kernel=3,
                          key='salient',
                          bg_key=None,
                          time_filtering=False,
                          response_filtering=False,
                          use_boundary_annots=False):
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
    '''
    key, bg_key = _validate_keys(key, bg_key)
    _all_keys = set(key + bg_key)
    has_requested_chans_list = []
    for gid in coco_dset.imgs:
        coco_img = coco_dset.coco_image(gid)
        chan_codes = coco_img.channels.normalize().fuse().as_set()
        flag = bool(_all_keys & chan_codes)
        has_requested_chans_list.append(flag)

    if not all(has_requested_chans_list):
        raise KeyError(f'{coco_dset.tag} has no keys {key} or {bg_key}')

    if use_boundary_annots:
        # get spatial bounds
        import shapely.ops
        annots = coco_dset.annots()
        boundary_annots = annots.compress(
            np.array(annots.cnames, dtype=str) == 'Site Boundary')
        if len(annots) < 1:
            print(f'warning: no Site Boundary annots in dset {coco_dset.tag}!')
        boundary_polys = [
            poly.to_shapely() for poly in
            boundary_annots.detections.data['segmentations'].to_polygon_list()
        ]
        bounds = shapely.ops.unary_union(boundary_polys)
        # get temporal bounds
        # TODO this works fine for one site boundary, for multiple sites each
        # trackid with Site Boundary annots should be handled separately.
        gids = np.unique(boundary_annots.gids)
        # remove boundary annots so they don't interfere with site labels
        # TODO preserve trackid for new polys?
        coco_dset.remove_categories(['Site Boundary'], keep_annots=False)
    else:
        bounds = None
        gids = coco_dset.imgs.keys()

    # record fg and bg keys across frames, and partial sums of fg and bg
    # this guarantees RunningStats of equal length for all keys,
    # even with partial/nonexistence
    running_dct = defaultdict(kwarray.RunningStats)
    for gid in gids:

        # TODO change assertion behavior to allow partial failure here
        fg_img_probs, fg_chan_probs = heatmap(coco_dset,
                                              gid,
                                              key,
                                              return_chan_probs=True)
        running_dct['fg'].update(fg_img_probs[:, :, np.newaxis])
        for k in key:
            running_dct[k].update(fg_chan_probs[k][:, :, np.newaxis])

        if len(bg_key) > 0:
            bg_img_probs, bg_chan_probs = heatmap(coco_dset,
                                                  gid,
                                                  bg_key,
                                                  return_chan_probs=True)
            running_dct['bg'].update(bg_img_probs[:, :, np.newaxis])
            for k in bg_key:
                running_dct[k].update(bg_chan_probs[k][:, :, np.newaxis])
        else:
            running_dct['bg'].update(
                    np.zeros_like(fg_img_probs)[:, :, np.newaxis])

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
    polys = list(mask_to_polygons(probs(running_dct['fg']), thresh, bounds=bounds))

    print('time aggregation: number of polygons:', len(polys))

    polys = list(SmallPolygonFilter(min_area_px=80)(polys))
    print('removed small: remaining polygons:', len(polys))

    # turn each polygon into a list of polygons (map them across gids)
    heatmaps = [heatmap(coco_dset, gid, key) for gid in coco_dset.imgs]

    def as_track(vidpoly):
        return Track.from_polys(itertools.repeat(vidpoly),
                                coco_dset,
                                probs=heatmaps)

    if response_filtering:
        response_thresh = 0.0002  # 0.0005
        # don't convert these to tracks yet, to make ResponsePolygonFilter
        # operate on each one as a separate track
        polys = list(
            ResponsePolygonFilter(map(as_track, polys), key,
                                  response_thresh)(polys))
        print('after filtering based on per-polygon response ', len(polys))

    tracks = map(as_track, polys)

    if time_filtering:
        # TODO investigate different thresh here
        time_thresh = thresh
        time_filter = TimePolygonFilter(coco_dset, tuple(key), time_thresh)
        tracks = map(time_filter, tracks)

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
    change_keys: Tuple[str] = ('salient', 'change_prob', 'change')

    def create_tracks(self, coco_dset):
        for change_key in self.change_keys:
            try:
                # import xdev; xdev.embed()
                tracks = time_aggregated_polys(
                    coco_dset,
                    self.thresh,
                    self.morph_kernel,
                    key=change_key,
                    time_filtering=self.time_filtering,
                    response_filtering=self.response_filtering)
                self.change_key = change_key
                return tracks
            except KeyError:
                pass

        raise KeyError(f'{coco_dset.tag} does not contain any image channel '
                       f'{self.change_keys}')

    def add_tracks_to_dset(self, coco_dset, tracks):
        coco_dset = add_tracks_to_dset(coco_dset, tracks, self.thresh,
                                       self.change_key)

        return coco_dset


@dataclass
class TimeAggregatedSC(NewTrackFunction):
    '''
    Wrapper for Site Characterization that looks for phase heatmaps.
    '''
    thresh: float = 0.1
    morph_kernel: int = 3
    time_filtering: bool = False
    response_filtering: bool = False
    key: Tuple[str] = (
        'Site Preparation', 'Active Construction', 'Post Construction'
    )
    bg_key: Tuple[str] = ('No Activity')
    use_boundary_annots: bool = True

    def create_tracks(self, coco_dset):
        tracks = time_aggregated_polys(
            coco_dset,
            self.thresh,
            self.morph_kernel,
            key=self.key,
            bg_key=self.bg_key,
            time_filtering=self.time_filtering,
            response_filtering=self.response_filtering,
            use_boundary_annots=self.use_boundary_annots)
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
    coco_dset_sc: kwcoco.CocoDataset

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
