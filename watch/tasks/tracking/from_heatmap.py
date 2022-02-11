from watch.utils import kwcoco_extensions
from watch.utils import util_kwimage
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


def mean_normalized(heatmaps):
    '''
    Normalize average_heatmap by applying a scaling based on max(heatmaps) and max(average_heatmap)
    '''
    average = np.average(heatmaps, axis=0)

    scale_factor = np.max(heatmaps) / (np.max(average) + 1e-9)
    print('max heatmaps', np.max(heatmaps))
    print('max average', np.max(average))

    #average *= scale_factor
    average = 0.75 * average * scale_factor
    print('scale_factor', scale_factor)
    print('After scaling, max:', np.max(average))

    return average


def frequency_weighted_mean(heatmaps, mask_thresh, morph_kernel=3):
    '''
    Convert a list of heatmaps to an aggregated score, averaging is computed based on samples for every pixel
    '''
    heatmaps = np.array(heatmaps)

    masks = 1 * (heatmaps > mask_thresh)
    pixel_wise_samples = masks.sum(0) + 1e-9
    print('pixel_wise_samples', pixel_wise_samples)

    # compute sum
    aggregated_probs = (masks * heatmaps).sum(0)

    # divide by number of samples for every pixel
    aggregated_probs /= pixel_wise_samples

    aggregated_probs = util_kwimage.morphology(aggregated_probs, 'dilate', morph_kernel)

    return aggregated_probs


def viterbi(input_sequence, transition_probs, emission_probs):
    """
    Viterbi decoding function.

    Obtain a MAP estimate for the most likely sequence of hidden states using a
    hidden Markov model.

    Args:
        input_sequence (ndarray[int]):
            Input sequence of shape (T,) encoding the sequence we believe we
            observed. Items are integers ranging from 0 to (S - 1), where S is
            the number of possible states. These indicate the "observed" state.

        transition_probs (ndarray[float]):
            Transition probabilities of shape (S, S), where
            ``transition_probs[i, j]`` indicates the probability that state
            ``i`` transitions to state ``j``. Rows should sum to 1.

        emission_probs (ndarray[float]):
            Emission probabilities of shape (S, S), where
            ``transition_probs[i, j]`` indicates the probability that
            when we observed state ``i`` the real state was actually ``j``.
            This encodes now noisy we believe the observations are.

    Returns:
        ndarray[int]: best_path
            The sequence of most likely true states

    References:
        - https://en.wikipedia.org/wiki/Viterbi_algorithm#Pseudocode
        - https://stackoverflow.com/questions/9729968/python-implementation-of-viterbi-algorithm

    Example:
        >>> # Demo based loosely on a star's simplified life sequence
        >>> states = ['cloud', 'small', 'giant', 'dwarf', 'large',
        >>>           'supergiant', 'supernova', 'neutron_star', 'black_hole']
        >>> # How likely is it for a state to change at any given time?
        >>> transitions = [
        >>>     {'src': 'cloud',        'dst': 'cloud',        'prob': 0.9},
        >>>     {'src': 'small',        'dst': 'small',        'prob': 0.9},
        >>>     {'src': 'giant',        'dst': 'giant',        'prob': 0.9},
        >>>     {'src': 'dwarf',        'dst': 'dwarf',        'prob': 0.9},
        >>>     {'src': 'large',        'dst': 'large',        'prob': 0.9},
        >>>     {'src': 'supergiant',   'dst': 'supergiant',   'prob': 0.9},
        >>>     {'src': 'supernova',    'dst': 'supernova',    'prob': 0.9},
        >>>     {'src': 'neutron_star', 'dst': 'neutron_star', 'prob': 0.9},
        >>>     {'src': 'black_hole',   'dst': 'black_hole',   'prob': 0.9},
        >>>     #
        >>>     {'src': 'cloud',      'dst': 'small',        'prob': 0.8},
        >>>     {'src': 'cloud',      'dst': 'large',        'prob': 0.2},
        >>>     {'src': 'small',      'dst': 'giant',        'prob': 1.0},
        >>>     {'src': 'giant',      'dst': 'dwarf',        'prob': 1.0},
        >>>     {'src': 'large',      'dst': 'supergiant',   'prob': 1.0},
        >>>     {'src': 'supergiant', 'dst': 'supernova',    'prob': 1.0},
        >>>     {'src': 'supernova',  'dst': 'neutron_star', 'prob': 6.0},
        >>>     {'src': 'supernova',  'dst': 'black_hole',   'prob': 4.0},
        >>> ]
        >>> # How likely is it that we made an error in observation?
        >>> emissions = [
        >>>     {'obs': 'cloud',        'real': 'cloud',        'prob': 0.5},
        >>>     {'obs': 'small',        'real': 'small',        'prob': 0.5},
        >>>     {'obs': 'giant',        'real': 'giant',        'prob': 0.5},
        >>>     {'obs': 'dwarf',        'real': 'dwarf',        'prob': 0.5},
        >>>     {'obs': 'large',        'real': 'large',        'prob': 0.5},
        >>>     {'obs': 'supergiant',   'real': 'supergiant',   'prob': 0.5},
        >>>     {'obs': 'supernova',    'real': 'supernova',    'prob': 0.5},
        >>>     {'obs': 'neutron_star', 'real': 'neutron_star', 'prob': 0.5},
        >>>     {'obs': 'black_hole',   'real': 'black_hole',   'prob': 0.5},
        >>> ]
        >>> emission_table = pd.DataFrame.from_dict(emissions)
        >>> emission_df = emission_table.pivot(['obs'], ['real'], ['prob'])
        >>> # Fill unspecified values in pairwise probability tables
        >>> import kwarray
        >>> rng = kwarray.ensure_rng(42110)
        >>> randfill = rng.rand(*emission_df.shape) * 0.01
        >>> flags = emission_df.isnull().astype(int)
        >>> emission_df = emission_df.fillna(0) + randfill * flags
        >>> transition_table = pd.DataFrame.from_dict(transitions)
        >>> transition_df = transition_table.pivot(['src'], ['dst'], ['prob']).fillna(0)
        >>> # Normalize probs
        >>> emission_df = emission_df.div(emission_df.groupby(axis=1, level=0).sum(), level=0)
        >>> transition_df = transition_df.div(transition_df.groupby(axis=1, level=0).sum(), level=0)
        >>> # Reorder indexes so we can use integer states
        >>> transition_df2 = transition_df.droplevel(axis=1, level=0)
        >>> emission_df2 = emission_df.droplevel(axis=1, level=0)
        >>> transition_df2 = transition_df2[states].loc[states]
        >>> emission_df2 = emission_df2[states].loc[states]
        >>> #
        >>> # Convert to ndarrays
        >>> transition_probs = transition_df2.values
        >>> emission_probs = emission_df2.values
        >>> #
        >>> observed_states = ['cloud', 'small', 'cloud', 'small', 'large', 'supergiant', 'black_hole', 'giant', 'dwarf', 'dwarf']
        >>> input_sequence = np.array([states.index(s) for s in observed_states], dtype=int)
        >>> best_path = viterbi(input_sequence, transition_probs, emission_probs)
        >>> predicted_states = [states[idx] for idx in best_path]
        >>> print('predicted_states = {!r}'.format(predicted_states))
        predicted_states = ['cloud', 'small', 'small', 'small', 'small', 'small', 'giant', 'giant', 'dwarf', 'dwarf']
    """
    # total number of states
    num_states = transition_probs.shape[0]

    # initialize prior from a uniform distribution
    Pi = (1 / num_states) * np.ones(num_states)

    # sequence length
    T = len(input_sequence)

    # probs of most likely path
    trellis = np.empty((num_states, T), dtype=np.float)

    # previous state of the most likely path
    pointers = np.empty((num_states, T), dtype=int)

    # determine each stat's prob at time 0
    trellis[:, 0] = Pi * emission_probs[:, input_sequence[0]]
    pointers[:, 0] = 0

    # track each state's most likely prior state
    for i in range(1, T):
        trellis[:, i] = np.max(trellis[:, i - 1] * transition_probs.T * emission_probs[np.newaxis, :, input_sequence[i]].T, 1)
        pointers[:, i] = np.argmax(trellis[:, i - 1] * transition_probs.T, 1)

    # best path
    best_path = np.empty(T, dtype=int)
    best_path[-1] = np.argmax(trellis[:, T - 1])
    for i in reversed(range(1, T)):
        best_path[i - 1] = pointers[best_path[i], i]

    return best_path


def class_label_smoothing(anns_in_track, categories_in_track, name_to_cat):
    """
    Given a sequence of class labels from SC model, perform smoothing on this sequence
    """
    # make class ids
    cid_no_activity = name_to_cat['No Activity']['id']
    cid_site_prep = name_to_cat['Site Preparation']['id']
    cid_active = name_to_cat['Active Construction']['id']
    cid_post = name_to_cat['Post Construction']['id']

    id_to_names = ['', 'No Activity', 'Site Preparation', 'Active Construction', 'Post Construction']
    #
    # Set up Viterbi Decoding
    #

    input_sequence = anns_in_track.cids

    # mapping of cids to canonical indices
    mapping_to_canonical = {
        cid_no_activity: 0,
        cid_site_prep: 1,
        cid_active: 2,
        cid_post: 3}

    # mapping from canonical to original cids
    mapping_to_original = {
        0: cid_no_activity,
        1: cid_site_prep,
        2: cid_active,
        3: cid_post}

    def replace(original_list, existing_val, desired_val):
        return [desired_val if i == existing_val else i for i in original_list]

    canonical_sequence = [mapping_to_canonical.get(number, number) for number in input_sequence]

    # Pre-processing: add post construction label at the end IF
    # 1) there was at least one active construction AND
    # 2) if the last frame is not active construction
    if (canonical_sequence[-1] != 2) and (2 in canonical_sequence):
        # canonical index of active construction is 2
        active_indices = [i for i, x in enumerate(canonical_sequence) if x == 2]
        last_active_ind = max(active_indices)

        # assign the frame after the last Active construction as Post Construction
        canonical_sequence[last_active_ind + 1] = 3

    # transition matrix
    transition_probs = np.array([
        [0.7, 0.1, 0.10, 0.10],
        [0.0, 0.7, 0.15, 0.15],
        [0.25, 0.0, 0.7, 0.05],
        [0.0, 0.0, 0.00, 1.00]
    ])

    # emission probability matrix
    emission_probs = np.array([
        [0.75, 0.10, 0.10, 0.10],
        [0.10, 0.75, 0.10, 0.10],
        [0.10, 0.10, 0.75, 0.10],
        [0.25, 0.25, 0.25, 0.75]
    ])

    smoothed_sequence = list(viterbi(canonical_sequence, transition_probs, emission_probs))

    # keep first post construction, mark others as no activity
    post_indices = [i for i, x in enumerate(smoothed_sequence) if x == 3]
    if (len(post_indices) > 0):
        for index in post_indices:
            smoothed_sequence[index] = 0

    # map canonical IDs back to original category IDs
    smoothed_fixed_cids = [mapping_to_original.get(number, number) for number in smoothed_sequence]

    smoothed_categories = []
    for cid in smoothed_fixed_cids:
        smoothed_categories.append(id_to_names[cid])
    return smoothed_categories


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
            new_ann = make_new_annotation(obs.gid, obs.poly, obs.score, track_id)
            new_anns.append(new_ann)

        for new_ann in new_anns:
            coco_dset.add_annotation(**new_ann)
        # TODO: Faster to add annotations in bulk, but we need to construct the
        # "ids" first
        # coco_dset.add_annotations(new_anns)

        # --- apply viterbi decoding ---
        if 0:
            anns_in_track = coco_dset.annots(trackid=track_id)
            categories_in_track = anns_in_track.cnames
            name_to_cat = coco_dset.name_to_cat
            # Viterbi decoding
            new_categories = class_label_smoothing(anns_in_track, categories_in_track, name_to_cat)

            # baseline: no smoothing uncomment the following line to skip viterbi decoding
            # new_categories = categories_in_track.copy()

            anns_in_track.set('category_id', [coco_dset.name_to_cat[name]['id'] for name in new_categories])
        # ------------------------------

    return coco_dset


def time_aggregated_polys(coco_dset,
                          thresh=0.15,
                          morph_kernel=3,
                          key='salient',
                          bg_key=None,
                          time_filtering=False,
                          response_filtering=False,
                          use_boundaries=False):
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

    #
    # --- utilities ---
    #

    # turn heatmaps into polygons
    def probs(heatmaps, weights=None):
        probs = np.average(heatmaps, axis=0, weights=weights)

        hard_probs = util_kwimage.morphology(probs > thresh, 'dilate',
                                             morph_kernel)
        modulated_probs = probs * hard_probs

        return modulated_probs

    def tracks_polys_bounds() -> Iterable[Tuple[Track, Poly]]:
        import shapely.ops
        boundary_tracks = list(pop_tracks(coco_dset, ['Site Boundary']))
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
            gid_ixs = np.in1d(gids, [obs.gid for obs in track.observations])
            track_polys = mask_to_polygons(probs(_heatmaps, weights=gid_ixs),
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

        polys = list(mask_to_polygons(mean_normalized(_heatmaps), thresh))

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

    def create_tracks(self, coco_dset):
        tracks = time_aggregated_polys(
            coco_dset,
            self.thresh,
            self.morph_kernel,
            key=self.key,
            time_filtering=self.time_filtering,
            response_filtering=self.response_filtering)
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
    key: Tuple[str] = ('Site Preparation', 'Active Construction',
                       'Post Construction')
    bg_key: Tuple[str] = ('No Activity')
    boundaries_as: Literal['bounds', 'polys', 'none'] = 'bounds'

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
                cnames=['Site Boundary'],
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
                use_boundaries=(self.boundaries_as != 'none'))

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
        #return TimeAggregatedSC(**self.sc_kwargs,
        return TimeAggregatedSC(boundaries_as='none').add_tracks_to_dset(
            coco_dset,
            tracks,
            coco_dset_sc=self.coco_dset_sc)

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
