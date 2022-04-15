import dateutil.parser
from datetime import timedelta
from typing import List
import itertools
import numpy as np
import kwcoco
import kwimage

from watch.tasks.tracking.utils import build_heatmaps, score_poly
from watch.heuristics import CNAMES_DCT


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
        - https://stackoverflow.com/q/9729968
    Example:
        >>> # Demo based loosely on a star's simplified life sequence
        >>> import numpy as np
        >>> import pandas as pd
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
        >>> transition_df = transition_table.pivot(
        >>>     ['src'], ['dst'], ['prob']).fillna(0)
        >>> # Normalize probs
        >>> emission_df = emission_df.div(emission_df.groupby(
        >>>     axis=1, level=0).sum(), level=0)
        >>> transition_df = transition_df.div(transition_df.groupby(
        >>>     axis=1, level=0).sum(), level=0)
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
        >>> observed_states = ['cloud', 'small', 'cloud', 'small', 'large',
        >>>     'supergiant', 'black_hole', 'giant', 'dwarf', 'dwarf']
        >>> input_sequence = np.array(
        >>>     [states.index(s) for s in observed_states], dtype=int)
        >>> from watch.tasks.tracking.phase import viterbi
        >>> best_path = viterbi(
        >>>     input_sequence, transition_probs, emission_probs)
        >>> predicted_states = [states[idx] for idx in best_path]
        >>> print('predicted_states = {!r}'.format(predicted_states))
        predicted_states = ['cloud', 'small', 'small', 'small', 'small',
                            'small', 'giant', 'giant', 'dwarf', 'dwarf']
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
        trellis[:, i] = np.max(
            trellis[:, i - 1] * transition_probs.T *
            emission_probs[np.newaxis, :, input_sequence[i]].T, 1)
        pointers[:, i] = np.argmax(trellis[:, i - 1] * transition_probs.T, 1)

    # best path
    best_path = np.empty(T, dtype=int)
    best_path[-1] = np.argmax(trellis[:, T - 1])
    for i in reversed(range(1, T)):
        best_path[i - 1] = pointers[best_path[i], i]

    return best_path


phase_classes = (CNAMES_DCT['negative']['scored'] +
                 CNAMES_DCT['positive']['scored'])
# {
# 'No Activity': 0,
# 'Site Preparation': 1,
# 'Active Construction': 2,
# 'Post Construction': 3
# }
index_to_class = dict(zip(range(len(phase_classes)), phase_classes))
class_to_index = {v: k for k, v in index_to_class.items()}

# sanity check: allowed transitions in the finite state machine of states
# (rows x cols) == (from_states x to_states)
# post construction to post construction is seen in the data, but should really
# only exist for 1 frame outside the edge case of subsite merging.
post_to_post = 0
allowed_transition_matrix = np.array(
    [[1, 1, 1, 0], [1, 1, 0, 0], [1, 0, 1, 1], [1, 0, 0, post_to_post]],
    dtype=bool)


registered_transition_probs = {
    'default': default_transition_probs,
}

# transition matrix
default_transition_probs = np.array([[0.7, 0.1, 0.10, 0.10],
                                     [0.0, 0.7, 0.15, 0.15],
                                     [0.25, 0.0, 0.7, 0.05],
                                     [0.0, 0.0, 0.00, 1.00]])

# emission probability matrix
# based on model predictions including "no activity"
#default_emission_probs = np.array([[0.233, 0.133, 0.633, 0.000],
#                                   [0.000, 0.133, 0.867, 0.000],
#                                   [0.020, 0.013, 0.966, 0.000],
#                                   [0.000, 0.000, 1.000, 0.000]])

# based on model predictions without "no activity"
default_emission_probs = np.array([[0.250, 0.250, 0.250, 0.250],
                                   [0.000, 0.133, 0.867, 0.000],
                                   [0.000, 0.014, 0.986, 0.000],
                                   [0.000, 0.000, 1.000, 0.000]])

# Based on CropDrop3_SC_V006_epoch=71-step=18431.pt
default_emission_probs = np.array([[0.25, 0.25, 0.25, 0.25],
                                   [0.00, 0.52, 0.48, 0.00],
                                   [0.00, 0.35, 0.65, 0.000],
                                   [0.00, 0.00, 1.000, 0.000]])


def _load_probs(arr_or_file, default=None):
    if arr_or_file is None:
        if default is None:
            raise ValueError(arr_or_file)
        else:
            return default
    if isinstance(arr_or_file, np.ndarray):
        arr = arr_or_file
    elif (isinstance(arr_or_file, list) and isinstance(arr_or_file[0],
                                                       (np.ndarray, list))):
        arr = np.ndarray(arr_or_file)
    else:
        arr = np.loadtxt(arr_or_file)
    assert arr.shape == (len(phase_classes), len(phase_classes)), arr
    return arr


def class_label_smoothing(track_cats, transition_probs=None,
                          emission_probs=None):
    """
    Args:
        track_cats: a list of scored SC phase names.
        Ex. ['Site Preparation', 'Active Construction', 'Site Preparation']

       transition_probs, emission_probs: see viterbi().
        These can be an (n_classes x n_classes) == (4x4) ndarray,
        or a format read by np.loadtxt - pathlike or list of strings,
        or None (use default).

    Returns:
        A smoothed list using Viterbi decoding.
        Ex. ['Site Preparation', 'Active Construction', 'Active Construction']

    # TODO make this work for subsites
    """

    transition_probs = _load_probs(transition_probs,
                                   default=default_transition_probs)
    emission_probs = _load_probs(emission_probs,
                                 default=default_emission_probs)

    #
    # Set up Viterbi Decoding
    #

    canonical_sequence = [class_to_index[i] for i in track_cats]

    # Pre-processing: add post construction label at the end IF
    # 1) there was at least one active construction AND
    # 2) if the last frame is not active construction
    if (canonical_sequence[-1] != 2) and (2 in canonical_sequence):
        # canonical index of active construction is 2
        active_indices = [
            i for i, x in enumerate(canonical_sequence) if x == 2
        ]
        last_active_ind = max(active_indices)

        # assign the frame after the last Active construction as Post
        # Construction
        canonical_sequence[last_active_ind + 1] = 3

    smoothed_sequence = list(
        viterbi(canonical_sequence, transition_probs, emission_probs))

    # keep first post construction, mark others as no activity
    post_indices = [i for i, x in enumerate(smoothed_sequence) if x == 3]
    for index in post_indices[1:]:
        smoothed_sequence[index] = 0

    smoothed_cats = [index_to_class[i] for i in smoothed_sequence]
    return smoothed_cats


def interpolate(coco_dset,
                track_id,
                cnames_to_keep=CNAMES_DCT['positive']['scored']):
    '''
    Replace any annot's cat not in cnames_to_keep with the most recent of
    cnames_to_keep
    '''
    annots = coco_dset.annots(trackid=track_id)
    cnames = annots.cnames
    cnames_to_replace = set(cnames) - set(cnames_to_keep)

    cids = np.array(annots.cids)
    good_ixs = np.in1d(cnames, list(cnames_to_replace), invert=True)
    ix_to_cid = dict(zip(range(len(good_ixs)), cids[good_ixs]))
    interp = np.interp(range(len(cnames)), good_ixs, range(len(good_ixs)))
    annots.set('category_id', [ix_to_cid[int(ix)] for ix in np.round(interp)])
    return annots


def baseline(coco_dset,
             track_id,
             cnames_to_insert=CNAMES_DCT['positive']['scored']):
    '''
    Predict site prep for the first half of the track and then active
    construction for the second half with post construction on the last frame
    '''
    annots = coco_dset.annots(trackid=track_id)

    assert len(cnames_to_insert) == 3, 'TODO generalize this with by_gid(anns)'
    siteprep_cid, active_cid, post_cid = map(coco_dset.ensure_category,
                                             cnames_to_insert)

    if len(set(annots.gids)) > 1:

        gids_first_half, gids_second_half = np.array_split(
            np.array(coco_dset.index._set_sorted_by_frame_index(annots.gids)),
            len(cnames_to_insert) - 1)
        gids = np.array(annots.gids)
        cids = np.where([g in gids_first_half for g in gids], siteprep_cid,
                        active_cid)
        cids = np.where(gids == gids_second_half[-1], post_cid, cids)
        annots.set('category_id', cids)

    return coco_dset


def sort_by_gid(coco_dset, track_id, prune=True):
    '''
    Group annots by image and return in sorted order by frame_index.

    Args:
        prune: if True, remove gids with no anns, else, return whole video

    Returns:
        (Images, AnnotGroups)
    '''
    images = coco_dset.images(
        coco_dset.index._set_sorted_by_frame_index(
            coco_dset.annots(trackid=track_id).gids))
    if len(images) == 0:
        print('warning: {track_id=} in {coco_dset.tag=} is empty')
        return images, []
    vidids = np.unique(images.get('video_id', None))
    assert len(vidids) == 1, f'track {track_id} spans multiple videos {vidids}'
    vidid = vidids[0]
    aids = set(coco_dset.index.trackid_to_aids[track_id])
    if not prune:
        images = coco_dset.images(vidid=vidid)
    return (images,
            kwcoco.coco_objects1d.AnnotGroups([
                coco_dset.annots(aids.intersection(img_aids))
                for img_aids in images.aids
            ], coco_dset))


def ensure_post(coco_dset,
                track_id,
                post_cname=CNAMES_DCT['positive']['scored'][-1],
                neg_cnames=CNAMES_DCT['negative']['scored']):
    '''
    If the track ends before the end of the video, and the last frame is
    not post construction, add another frame of post construction

    TODO this is not a perfect approach, since we don't have per-subsite
    tracking across frames. We can run into a case where:
    frame  1   2   3
    ss1    AC  AC  PC
    ss2    AC  AC
    it is ambiguous whether ss2 ends on AC or merges with ss1.
    Ignore this edge case (assume merge) for now.
    '''
    images, annot_groups = sort_by_gid(coco_dset, track_id)
    if len(list(annot_groups)) > 1:
        last_gid = images.gids[-1]
        gids = coco_dset.index._set_sorted_by_frame_index(
            coco_dset.images(vidid=images.get('vidid', None)[0]).gids)
        current_gid = gids[-1]

        def img_to_vid(gid):
            return kwimage.Affine.coerce(coco_dset.imgs[gid].get(
                'warp_img_to_vid', {'scale': 1}))

        post_cid = coco_dset.ensure_category(post_cname)

        if last_gid != current_gid:
            next_gid = gids[gids.index(last_gid) + 1]

            annots = annot_groups[-1]
            annots = annots.compress(np.array(annots.cnames) == post_cname)
            dets = annots.detections.warp(img_to_vid(last_gid)).warp(
                img_to_vid(next_gid).inv())
            for ann, seg, bbox in zip(
                    annots.objs,
                    dets.data['segmentations'].to_coco(style='new'),
                    dets.boxes.to_coco(style='new')):

                post_ann = ann.copy()
                post_ann.pop('id')
                if 'track_index' in post_ann:
                    post_ann['track_index'] += 1
                post_ann.update(
                    dict(image_id=next_gid,
                         category_id=post_cid,
                         segmentation=seg,
                         bbox=bbox))
                coco_dset.add_annotation(**post_ann)
                print(f'ensure_post {track_id=}: duplicating {ann["id"]=} '
                      f'from {last_gid=} to {next_gid=}')

    return coco_dset


def dedupe_background_anns(coco_dset,
                           track_id,
                           post_cname=CNAMES_DCT['positive']['scored'][-1],
                           neg_cnames=CNAMES_DCT['negative']['scored']):
    '''
    Chop off extra Post Construction and No Activity annots from the end of the
    track so they don't count as FPs.

    TODO same edge case as ensure_post() for lack of subsite tracking
    '''
    images, annot_groups = sort_by_gid(coco_dset, track_id)
    relevant_cnames = set(neg_cnames + [post_cname])
    end_annots = []
    for annots in reversed(list(annot_groups)):
        if set(annots.cnames).issubset(relevant_cnames):
            end_annots.append(annots)
        else:
            break
    removed_dct = coco_dset.remove_annotations(
        list(
            itertools.chain.from_iterable(
                [annots.aids for annots in end_annots[:-1]])))
    n_removed = removed_dct['annotations']
    if (n_removed or 0) > 0:
        print(
            f'removed {n_removed} background anns from end of track {track_id}'
        )

    return coco_dset


def current_date(annots):
    current_date_dct = dict(
        zip(annots.images.gids, [
            dateutil.parser.parse(dt).date()
            for dt in annots.images.get('date_captured', '1970-01-01')
        ]))
    return np.array([current_date_dct[gid] for gid in annots.gids])


def phase_prediction_baseline(annots) -> List[float]:
    '''
    Number of days until the next expected activity phase transition.

    Baseline: (average days in current_phase - elapsed days in current_phase)
    '''
    # from watch.dev.check_transition_probs
    phase_avg_days = {
        'No Activity': 60,
        'Site Preparation': 57,
        'Active Construction': 56,
        'Post Construction': 58
    }
    phase_avg_days = {k: timedelta(v) for k, v in phase_avg_days.items()}

    today = current_date(annots)

    # per metrics definition doc v2.2, the site enters <phase> on the date
    # when the last subsite enters <phase>, unless only some subsites are
    # in <phase> on <current_date>, in which case it is the first date a
    # subsite enters <phase>.
    # this doesn't make any sense.
    # ignore this, and just take the first subsite date.
    first_date = dict()
    for phase, date in zip(annots.cnames, today):
        missing_phases = set(phase_avg_days) - set(first_date)
        if missing_phases:
            if phase in missing_phases:
                first_date[phase] = date
        else:
            break

    predicted = np.array(
        [first_date[phase] + phase_avg_days[phase] for phase in annots.cnames])

    return np.where(predicted > today,
                    (predicted - today).astype('timedelta64[D]').astype(float),
                    1)


def phase_prediction_heatmap(annots, coco_dset, key) -> List[float]:
    '''
    Get phase prediction heatmaps from model output and score them against
    annotation polygons to get predicted dates
    '''
    gids = list(set(annots.gids))
    heatmaps_dct = dict(
        zip(gids,
            build_heatmaps(coco_dset, gids, [key], skipped='interpolate')[key]))
    # TODO generalize this as annots.detections.responses()?
    return [
        score_poly(poly, heatmaps_dct[gid])
        for poly, gid in zip(
            annots.detections.data['segmentations'].to_polygon_list(),
            annots.gids)
    ]
