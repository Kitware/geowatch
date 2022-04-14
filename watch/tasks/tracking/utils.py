import kwimage
import numpy as np
import kwcoco
from rasterio import features
import scipy.ndimage.measurements as ndm
from copy import deepcopy
import shapely.geometry
import ubelt as ub
from dataclasses import dataclass, astuple
# import functools
import itertools
import collections
from abc import abstractmethod
from typing import Union, Iterable, Optional, Any, Tuple, List, Dict
import warnings


def trackid_is_default(trackid):
    '''
    Hack to decide if a trackid is really a site_id or if it was randomly
    assigned
    '''
    if trackid is None:
        return True
    try:
        int(trackid)
        return True
    except ValueError:
        return False


try:
    from xdev import profile
except Exception:
    profile = ub.identity


Poly = Union[kwimage.Polygon, kwimage.MultiPolygon]


# TODO use geopandas for this instead?
# Pros:
# - vectorization
# - geometry handling
# - table structure/row-col slicing
# Cons:
# - geopandas
#
# store these in a Track as a geodataframe, then instantiate as Obs as needed?
# this avoids generators everywhere on access
@dataclass
class Observation:
    poly: Poly
    gid: Optional[int] = None
    # As of now, scores are mostly recalculated every time they are needed
    # for a different gid, key, etc. There's potential to store them
    # here instead.
    score: Optional[int] = None  # TODO restrict this to [0,1]


@dataclass
class Track:
    observations: Iterable[Observation]
    # dset: Optional[kwcoco.CocoDataset]
    dset: Any  # omg, I can't believe type errors are breaking runtime now. I must have some weird IPython package installed that does that.
    vidid: Optional[int] = None
    track_id: Optional[int] = None

    @classmethod
    def from_polys(cls, polys, dset, probs=None, vidid=None, **kwargs):
        if vidid is not None:
            gids = dset.index.vidid_to_gids[vidid]
        else:
            gids = dset.imgs.keys()

        if probs is not None:
            obs = [
                Observation(poly, gid, score_poly(poly, prob))
                for poly, gid, prob in zip(polys, gids, probs)
            ]
        else:
            obs = [Observation(poly, gid) for poly, gid in zip(polys, gids)]

        return cls(observations=obs, dset=dset, vidid=vidid, **kwargs)


class PolygonFilter(collections.abc.Callable):
    '''
    The type signatures are not enforced by python in child classes, but this
    is the intended use.
    TODO use mypy to enforce this? https://stackoverflow.com/q/25183424

    Many filters will only need polygons, but we could be keeping track of
    stuff along with it, such as "enumerate(polys)". The core function here
    is a "lifted" filter function that operates on a "polygon plus stuff".

    Given that assumption, PolygonFilters can be called on any supported type
    containing polygons and return a filtered version of it.
    '''
    # @functools.singledispatchmethod
    '''
    singledispatch does not work on parameterized container types until
    collections.abc.*[] is supported in Python 3.9; see:
    https://bugs.python.org/issue34499
    https://bugs.python.org/issue34498

    So, do this by hand instead.
    '''
    def __call__(self, obj):
        if isinstance(obj, Track):
            return self.on_track(obj)
        # could use more_itertools.peekable instead
        obj, obj2 = itertools.tee(obj)
        try:
            sample_object = next(obj2)
        except StopIteration:
            return obj
        if isinstance(sample_object, Observation):
            return self.on_observations(obj)
        # 'TypeError: Subscripted generics cannot be used with class and
        # instance checks'
        # breaking change in py3.7...
        # if isinstance(sample_object, Poly):
        if isinstance(sample_object, (kwimage.Polygon, kwimage.MultiPolygon)):
            return self.on_polys(obj)
        if isinstance(sample_object[1],
                      (kwimage.Polygon, kwimage.MultiPolygon)):
            return self.on_augmented_polys(obj)
        raise NotImplementedError(
            f'cannot filter polys like {sample_object}: unsupported type')

    # @__call__.register
    def on_track(self, track: Track):
        track.observations = self.on_observations(track.observations)
        return track

    # @__call__.register
    def on_observations(self, observations: Iterable[Observation]):
        # extract the polygon from each Observation to operate on
        augmented_polys = ((astuple(obs), obs.poly) for obs in observations)
        # filter them and rebuild them
        return (Observation(**obs)
                for obs, _ in self.on_augmented_polys(augmented_polys))

    # @__call__.register
    def on_polys(self, polys: Iterable[Poly]):
        augmented_polys = ((None, p) for p in polys)
        return (poly for _, poly in self.on_augmented_polys(augmented_polys))

    # @__call__.register
    @abstractmethod
    def on_augmented_polys(self, aug_polys: Iterable[Tuple[Any, Poly]]):
        raise NotImplementedError('must be implemented by subclasses')


@dataclass(frozen=True)  # to prevent cache invalidation
class CocoDsetFilter(PolygonFilter):
    '''
    Specialization of PolygonFilter for polygons that are tied to images,
    registered as gids in a CocoDataset
    '''
    dset: kwcoco.CocoDataset
    key: Tuple[str]
    threshold: float

    @ub.memoize_method
    def _heatmap(self, gid):
        return build_heatmap(self.dset, gid, self.key)

    @ub.memoize_method
    def score(self, poly, gid, mode, threshold=None):
        return score_poly(poly, self._heatmap(gid), mode=mode, threshold=threshold)


class TrackFunction(collections.abc.Callable):
    '''
    Abstract class that all track functions should inherit from.
    '''
    @abstractmethod
    def __call__(self, sub_dset) -> kwcoco.CocoDataset:
        '''
        Ensure each annotation in coco_dset has a track_id.
        '''
        raise NotImplementedError('must be implemented by subclasses')

    def apply_per_video(self, coco_dset, overwrite=False):
        '''
        Main entrypoint for this class.
        '''
        legacy = False

        tracked_subdsets = []
        vid_gids = coco_dset.index.vidid_to_gids.values()
        total = len(coco_dset.index.vidid_to_gids)
        for gids in ub.ProgIter(vid_gids, total=total, desc='apply_per_video', verbose=3):
            sub_dset = self.safe_apply(coco_dset, gids, overwrite, legacy=legacy)
            if legacy:
                coco_dset = sub_dset
            else:
                tracked_subdsets.append(sub_dset)

        if not legacy:
            # Tracks were either updated or added.
            # In the case they were updated the existing track ids should
            # be disjoint. All new tracks should not overlap with

            _debug = 0

            from watch.utils import kwcoco_extensions
            new_trackids = kwcoco_extensions.TrackidGenerator(None)
            fixed_subdataset = []
            for sub_dset in ub.ProgIter(tracked_subdsets, desc='Ensure ok tracks', verbose=3):

                if _debug:
                    sub_dset = sub_dset.copy()

                # Rebuild the index to ensure any hacks are removed.
                # We should be able to remove this step.
                # sub_dset._build_index()

                sub_annots = sub_dset.annots()
                sub_tids = sub_annots.lookup('track_id')
                existing_tids = set(sub_tids)

                collisions = existing_tids & new_trackids.used_trackids
                if _debug:
                    print('existing_tids = {!r}'.format(existing_tids))
                    print('collisions = {!r}'.format(collisions))

                new_trackids.exclude_trackids(existing_tids)
                if collisions:
                    old_tid_to_aids = ub.group_items(sub_annots, sub_tids)
                    assert len(old_tid_to_aids) == len(existing_tids)
                    print(f'Resolve {len(collisions)} track collisions')
                    # Change the track ids of any collisions
                    for old_tid in collisions:
                        new_tid = next(new_trackids)
                        # Note: this does not update the index, but we
                        # are about to clobber it anyway, so it doesnt matter
                        for aid in old_tid_to_aids[old_tid]:
                            ann = sub_dset.index.anns[aid]
                            ann['track_id'] = new_tid
                        existing_tids.add(new_tid)
                new_trackids.exclude_trackids(existing_tids)

                if _debug:
                    after_tids = set(sub_annots.lookup('track_id'))
                    print('collisions = {!r}'.format(collisions))
                    print(f'{after_tids=}')

                fixed_subdataset.append(sub_dset)

            # Is this safe to do? It would be more efficient
            coco_dset = kwcoco.CocoDataset.union(
                *fixed_subdataset, disjoint_tracks=False)

            if _debug:
                x = coco_dset.annots().images.get('video_id')
                y = coco_dset.annots().get('track_id')
                z = ub.group_items(x, y)
                track_to_num_videos = ub.map_vals(set, z)
                assert max(map(len, track_to_num_videos.values())) == 1, (
                    'track belongs to multiple videos!'
                )
        return coco_dset

    @profile
    def safe_apply(self, coco_dset, gids, overwrite, legacy=True):

        if legacy:
            sub_dset, rest_dset = self.safe_partition(coco_dset, gids, remove=True)
        else:
            sub_dset = self.safe_partition(coco_dset, gids, remove=False)

        if overwrite:
            sub_dset = self(sub_dset)
        else:
            orig_annots = sub_dset.annots()
            orig_tids = orig_annots.get('track_id', None)
            orig_trackless_flags = np.array([tid is None for tid in orig_tids])
            orig_aids = list(orig_annots)

            # TODO more sophisticated way to check if we can skip self()
            sub_dset = self(sub_dset)

            # if new annots were not created, rollover the old tracks
            new_annots = sub_dset.annots()
            if new_annots.aids == orig_aids:
                new_tids = new_annots.get('track_id', None)
                # Only overwrite track ids for annots that didn't have them
                new_tids = np.where(orig_trackless_flags, new_tids, orig_tids)
                new_annots.set('track_id', new_tids)

        # TODO: why is this assert here?
        assert None not in sub_dset.annots().lookup('track_id', None)
        if legacy:
            return self.safe_union(rest_dset, sub_dset)
        else:
            return sub_dset

    @staticmethod
    @profile
    def safe_partition(coco_dset, gids, remove=True):
        sub_dset = coco_dset.subset(gids=gids, copy=True)
        # HACK ensure tracks are not duplicated between videos
        # (if they are, this is fixed in dedupe_tracks anyway)
        sub_dset.index.trackid_to_aids.update(coco_dset.index.trackid_to_aids)
        if remove:
            rest_gids = list(set(coco_dset.imgs.keys()) - set(gids))
            rest_dset = coco_dset.subset(rest_gids)
            return sub_dset, rest_dset
        else:
            return sub_dset

    @staticmethod
    @profile
    def safe_union(coco_dset, new_dset, existing_aids=[]):
        coco_dset._build_index()
        new_dset._build_index()
        # we handle tracks in normalize.dedupe_tracks anyway, and
        # disjoint_tracks=True interferes with keeping site_ids around as
        # track_ids.
        # return coco_dset.union(new_dset, disjoint_tracks=True)
        return coco_dset.union(new_dset, disjoint_tracks=False)


class NoOpTrackFunction(TrackFunction):
    '''
    Use existing tracks.
    '''
    def __call__(self, sub_dset):
        return sub_dset


class NewTrackFunction(TrackFunction):
    '''
    Specialization of TrackFunction to create polygons that do not yet exist
    in coco_dset, and add them as new annotations
    '''
    def __call__(self, sub_dset):
        tracks = self.create_tracks(sub_dset)
        sub_dset = self.add_tracks_to_dset(sub_dset, tracks)
        return sub_dset

    @abstractmethod
    def create_tracks(self, sub_dset) -> Iterable[Track]:
        raise NotImplementedError('must be implemented by subclasses')

    @abstractmethod
    def add_tracks_to_dset(self, sub_dset,
                           tracks: Iterable[Track]) -> kwcoco.CocoDataset:
        raise NotImplementedError('must be implemented by subclasses')


def check_only_bg(category_sequence, bg_name=['No Activity']):
    if len( set(category_sequence) - set(bg_name) ) == 0:
        return True
    else:
        return False


def pop_tracks(
        coco_dset: kwcoco.CocoDataset,
        cnames: Iterable[str],
        remove: bool = True,
        score_chan: Optional[kwcoco.ChannelSpec] = None) -> Iterable[Track]:
    '''
    Convert kwcoco annotations into Track objects.

    Args:
        coco_dset
        cnames: category names
        remove: remove the annotations from coco_dset
        score_chan: score the track polygons by image overlap with this channel

    Returns:
        Track objects.
        Mutates coco_dset if remove=True.
    '''
    # TODO could refactor to work on coco_dset.annots() and integrate
    cnames = list(set(cnames))

    annots = coco_dset.annots()
    annots = annots.compress(
        np.in1d(np.array(annots.cnames, dtype=str), cnames))
    if len(annots) < 1:
        print(f'warning: no {cnames} annots in dset {coco_dset.tag}!')

    annots = deepcopy(annots)
    if remove:
        coco_dset.remove_categories(cnames, keep_annots=False)

    polys = annots.detections.data['segmentations'].to_polygon_list()
    assert len(polys) == len(annots), ('TODO handle multipolygon boundaries')

    if score_chan is not None:
        # bookkeep unique gids only
        # hackish, pretend it's all one big track for efficient interpolation
        # TODO make this work for multiple videos
        gids = coco_dset.index._set_sorted_by_frame_index(annots.gids)
        keys = {score_chan.spec: list(score_chan.unique())}
        heatmaps = build_heatmaps(coco_dset, gids, keys)[score_chan.spec]
        heatmaps_by_gid = dict(zip(gids, heatmaps))
        scores = [
            score_poly(poly, heatmaps_by_gid[gid])
            for poly, gid in zip(polys, annots.gids)
        ]
    else:
        scores = [None] * len(annots)
        # scores = annots.get('score', None)

    for track_id, track_info in ub.group_items(zip(polys, annots.gids, scores),
                                               annots.get('track_id',
                                                          None)).items():
        track_polys, track_gids, track_scores = zip(*track_info)
        yield Track(list(
            map(Observation, track_polys, track_gids, track_scores)),
                    dset=coco_dset,
                    track_id=track_id)


def score_poly(poly, probs, mode='score', threshold=0, use_rasterio=True):
    '''
    Args:
        poly: kwimage.Polygon or MultiPolygon in pixel coords

        probs: heatmap to compare poly against

        mode: return value.
            'score': fraction of probs contained in poly
            'response': average value of probs in poly
            'overlap': fraction of poly with probs > threshold

        use_rasterio: use rasterio.features module instead of kwimage

        threshold: only used for mode='overlap'
    '''
    # try converting from shapely
    # TODO standard coerce fns between kwimage, shapely, and __geo_interface__
    if not isinstance(poly, (kwimage.Polygon, kwimage.MultiPolygon)):
        poly = kwimage.MultiPolygon.from_shapely(poly)
    if 0:
        # naive computation across the whole image
        poly_mask = poly.to_mask(probs.shape).numpy().data
        rel_mask, rel_probs = poly_mask, probs
    else:
        # First compute the valid bounds of the polygon
        # And create a mask for only the valid region of the polygon
        box = poly.bounding_box().quantize().to_xywh()
        # Ensure box is inside probs
        ymax, xmax = probs.shape[:2]
        box = box.clip(0, 0, xmax, ymax).to_xywh()
        if box.area[0][0] == 0:
            warnings.warn('warning: scoring a polygon against an img with no overlap!')
            return 0
        x, y, w, h = box.data[0]
        if use_rasterio:  # rasterio inverse
            rel_poly = poly.translate((0.5 - x, 0.5 - y))
            rel_mask = features.rasterize([rel_poly.to_geojson()],
                                          out_shape=(h, w))
        else:  # kwimage inverse
            rel_poly = poly.translate((-x, -y))
            rel_mask = rel_poly.to_mask((h, w)).data
        # Slice out the corresponding region of probabilities
        rel_probs = probs[y:y + h, x:x + w]
        # hacking to solve a bug: sometimes shape of rel_probs is x,y,1
        if len(rel_probs.shape) == 3:
            rel_probs = rel_probs[:, :, 0]

    # TODO these are preserved for backwards compatibility, but they should
    # actually be the same. Test and remove them.
    if mode == 'response':
        response = (rel_mask * rel_probs).mean()
        return response
    elif mode == 'score':
        total = rel_mask.sum()
        score = 0 if total == 0 else (rel_mask * rel_probs).sum() / total
        return score
    elif mode == 'overlap':
        hard_prob = rel_probs > threshold
        overlap = (hard_prob * rel_mask).sum()
        total_poly_area = rel_mask.sum()
        return overlap / total_poly_area
    else:
        raise ValueError(mode)


def mask_to_polygons(probs,
                     thresh,
                     bounds=None,
                     scored=False,
                     use_rasterio=True,
                     thresh_hysteresis=None):
    """
    Args:
        probs: aka heatmap, image of probability values
        thresh: to turn probs into a hard mask
        bounds: a kwimage or shapely polygon to crop the results to
        scored: return Iterable[Tuple[score, poly]] instead of Iterable[Poly]
        use_rasterio: use rasterio.features module instead of kwimage
        thresh_hysteresis: if not None, only keep polygons with at least one
            pixel of score >= thresh_hysteresis

    Returns:
        Iterable[kwcoco.Polygon]

    Example:
        >>> from watch.tasks.tracking.utils import mask_to_polygons
        >>> import kwimage
        >>> probs = kwimage.Heatmap.random(dims=(64, 64),
        >>>                                rng=0).data['class_probs'][0]
        >>> thresh = 0.5
        >>> polys = mask_to_polygons(probs, thresh, scored=True)
        >>> score1, poly1 = list(polys)[0]
        >>> # xdoctest: +IGNORE_WANT
        >>> import kwplot
        >>> kwplot.autompl()
        >>> kwplot.imshow(probs > 0.5)
    """
    # Threshold scores
    if thresh_hysteresis is None:
        binary_mask = (probs > thresh).astype(np.uint8)
    else:
        mask = probs > thresh
        seeds = probs > thresh_hysteresis
        label_img = ndm.label(mask)[0]
        selected = np.unique(np.extract(seeds, label_img))
        binary_mask = np.isin(label_img, selected).astype(np.uint8)
    if bounds is not None:
        try:  # is this a shapely or geojson object?
            # asShape is being deprecated:
            # https://github.com/shapely/shapely/issues/1100
            bounds = shapely.geometry.shape(bounds)
        except ValueError:  # is this a kwimage object?
            bounds = bounds.to_shapely()
        if use_rasterio:
            # TODO needed?
            # x, y order for shapely
            bounds = shapely.affinity.translate(bounds, 0.5, 0.5)
            # TODO investigate all_touched option
            bounds_mask = features.rasterize([bounds],
                                             dtype=np.uint8,
                                             out_shape=binary_mask.shape[:2])
        else:
            bounds_mask = kwimage.Polygon.from_shapely(bounds).to_mask(
                probs.shape).numpy().data.astype(np.uint8)
        binary_mask *= bounds_mask

    if use_rasterio:
        shapes = features.shapes(binary_mask)
        polygons = [
            kwimage.Polygon.from_geojson(s).translate((-0.5, -0.5))
            for s, v in shapes if v
        ]
    else:
        polygons = kwimage.Mask(binary_mask, 'c_mask').to_multi_polygon()

    if scored:
        for poly in polygons:
            yield score_poly(poly, probs, use_rasterio=use_rasterio), poly
    else:
        yield from polygons


def _validate_keys(key, bg_key):
    # for backwards compatibility
    if bg_key is None:
        bg_key = []
    key = list(key) if ub.iterable(key) else [key]
    bg_key = list(bg_key) if ub.iterable(bg_key) else [bg_key]

    # error checking
    if len(key) < 1:
        raise ValueError('must have at least one key')
    if (len(key) > len(set(key)) or len(bg_key) > len(set(bg_key))):
        raise ValueError('keys are duplicated')
    if not set(key).isdisjoint(set(bg_key)):
        raise ValueError('cannot have a key in foreground and background')
    return key, bg_key


def build_heatmaps(sub_dset: kwcoco.CocoDataset,
                   gids: List[int],
                   keys: Union[List[str], Dict[str, List[str]]],
                   missing='fill',
                   skipped='interpolate',
                   video_id=None) -> Dict[str, List[np.array]]:
    '''
    Vectorized version of watch.tasks.tracking.utils.build_heatmap across gids.

    Can also sum keys using group names.

    Example:
        build_heatmaps(dset, gids=[1,2], ['key1', 'key2', 'key3']) == {
            'key1': heats1,
            'key2': heats2,
            'key3': heats3
        }
        build_heatmaps(dset, gids=[1,2], {'group1': ['key1', 'key2', 'key3']}) == {
            'key1': heats1,
            'key2': heats2,
            'key3': heats3,
            'group1': heats1 + heats2 + heats3
        }
        where len(heats) == len(gids) == 2.

    Restrictions wrt heatmap():
        - uses video space
        - returns chan probs

    Args:
        sub_dset (kwcoco.CocoDataset): must have exactly 1 video
        gids: List[image id]
        key: List[str] list of channel names
        space: 'video' or 'image'
        missing: behavior for missing keys.
            'fill': return probs and chan_probs of zeros
            'skip': return probs of zeros, skip chan_probs
            'raise': raise exception
        skipped: behavior for missing keys across gids.
            'interpolate': use heatmap from last gid
            'zeros': insert zeros
            # 'remove': do not return this gid  # TODO w/ different signature
        video_id (int | None): if specified, get heatmaps for this video
            otherwise assert that there is exactly one video

    Returns:
        {key: [heatmap for each gid]}
    '''
    # TODO use ChannelSpec objects
    # TODO doctest

    if isinstance(keys, list):
        key_groups = {'__dummy__': keys}
        _dummy_groups = ['__dummy__']
    elif isinstance(keys, dict):
        key_groups = keys
        _dummy_groups = []
    else:
        raise TypeError(type(keys))

    # Would use RunningStats, but it can't support indexed/subsetted access
    # for multiple site boundaries over different times.
    # This solution is more efficient when len(tracks) > len(gids).
    #
    # running_dct = defaultdict(kwarray.RunningStats)
    heatmaps_dct = collections.defaultdict(list)

    # record previous heatmaps in video space to propagate thru missing
    # frames
    if video_id is None:
        assert len(sub_dset.index.videos) == 1
        video_id = ub.peek(sub_dset.index.videos.values())['id']

    vid = sub_dset.index.videos[video_id]
    vid_shape = (vid['height'], vid['width'])
    prev_heatmap_dct = collections.defaultdict(lambda: np.zeros(vid_shape))

    for gid in gids:
        for group, key in key_groups.items():

            # we are working only in vid space, so forget about warping
            img_probs, chan_probs = build_heatmap(sub_dset, gid, key,
                                                  space='video',
                                                  return_chan_probs=True)
            # TODO make this more efficient using missing='skip'
            if any(np.flatnonzero(img_probs)):
                heatmaps_dct[group].append(img_probs)
            elif skipped == 'interpolate':
                heatmaps_dct[group].append(prev_heatmap_dct[group])
            elif skipped == 'zeros':
                heatmaps_dct[group].append(np.zeros(vid_shape))
            else:
                raise ValueError(skipped)

            for k in key:
                if k in chan_probs:
                    heatmaps_dct[k].append(chan_probs[k])
                    prev_heatmap_dct[k] = chan_probs[k]
                elif skipped == 'interpolate':
                    heatmaps_dct[k].append(prev_heatmap_dct[k])
                elif skipped == 'zeros':
                    heatmaps_dct[k].append(np.zeros(vid_shape))
                else:
                    raise ValueError(skipped)

    for dummy in _dummy_groups:
        heatmaps_dct.pop(dummy)
    return heatmaps_dct


def build_heatmap(dset, gid, key, return_chan_probs=False, space='video',
                  missing='fill'):
    """
    Find the total heatmap of key within gid

    Args:
        dset: kwcoco.CocoDataset
        gid: image id
        key: List[str] list of channel names
        return_chan_probs:
            if True, also return a dict {k: build_heatmap(k) for k in keys}
        space: 'video' or 'image'
        missing: behavior for missing keys.
            'fill': return probs and chan_probs of zeros
            'skip': return probs of zeros, skip chan_probs
            'raise': raise exception
    """
    key, _ = _validate_keys(key, None)
    coco_img = dset.coco_image(gid)

    channels_request = kwcoco.FusedChannelSpec.coerce(key)
    channels_have = coco_img.channels.fuse().intersection(channels_request)

    if missing == 'raise':
        if channels_have.numel() != channels_request.numel():
            raise ValueError(ub.paragraph(
                f'''
                Requeted {channels_request=} in the image {gid=} of {dset=}
                but only {channels_have=} existed.
                '''))

    w, h = coco_img.delay(space=space).dsize

    common = channels_have

    if len(common) == 0:  # for bg_key
        fg_img_probs = np.zeros((h, w))
        if return_chan_probs:
            if missing == 'skip':
                return fg_img_probs, {}
            else:
                return fg_img_probs, {k: fg_img_probs for k in key}
        else:
            return fg_img_probs

    if 0 and __debug__:
        if common.numel() > 1:
            print('WARNING: Im not sure about that sum axis=-1, '
                  'I hope there is only ever one channel here')

    key_img_probs = coco_img.delay(channels=common, space=space).finalize(nodata='float')
    # Not sure about that sum axis=-1 here
    fg_img_probs = key_img_probs.sum(axis=-1)
    if return_chan_probs:
        # some awkwardness here from non-invertible mapping from
        # ChannelSpec to FusedChannelSpec
        chan_probs = {}
        idxs = common.component_indices()
        for k in key:
            codes = common.intersection([k]).as_list()
            probs = [key_img_probs[idxs[code]] for code in codes]
            if len(probs) == 0:
                if missing == 'skip':
                    continue
                else:
                    probs.append(np.zeros((h, w)))
            # Again, I'm not sure about this sum here.
            chan_probs[k] = np.sum(probs, axis=0)
        return fg_img_probs, chan_probs
    else:
        return fg_img_probs
