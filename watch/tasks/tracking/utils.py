from watch.utils import kwcoco_extensions
import kwimage
import numpy as np
import kwcoco
from rasterio import features
import shapely.geometry
import ubelt as ub
from dataclasses import dataclass, astuple
# import functools
import itertools
import collections
from abc import abstractmethod
from typing import Union, Iterable, Optional, Any, Tuple, List

Poly = Union[kwimage.Polygon, kwimage.MultiPolygon]


# TODO use geopandas for this instead?
# Pros:
# - vectorization
# - geometry handling
# - table structure/row-col slicing
# Cons:
# - geopandas
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
    dset: Optional[kwcoco.CocoDataset]
    vidid: Optional[int]

    @classmethod
    def from_polys(cls, polys, dset, vidid=None, probs=None):
        if vidid is not None:
            gids = dset.index.vidid_to_gids[vidid]
        else:
            gids = dset.imgs.keys()

        if probs is not None:
            obs = [
                Observation(poly, gid, score(poly, prob))
                for poly, gid, prob in zip(polys, gids, probs)
            ]
        else:
            obs = [Observation(poly, gid) for poly, gid in zip(polys, gids)]

        return cls(observations=obs, dset=dset, vidid=vidid)


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
        sample_object = next(iter(obj2))
        if isinstance(sample_object, Observation):
            return self.on_observations(obj)
        # TypeError: Subscripted generics cannot be used with class and instance checks
        # breaking change in py3.7...
        if isinstance(sample_object, (kwimage.Polygon, kwimage.MultiPolygon)):
            return self.on_polys(obj)
        if isinstance(sample_object[1], (kwimage.Polygon, kwimage.MultiPolygon)):
            return self.on_augmented_polys(obj)
        raise NotImplementedError(f'cannot filter polygons from {obj}:'
                                  ' unsupported type')

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

    @ub.memoize
    def _heatmap(self, gid):
        return heatmap(self.dset, gid, self.key)

    @ub.memoize
    def score(self, poly, gid, mode, threshold=None):
        return score(poly, self._heatmap(gid), mode=mode, threshold=threshold)


class TrackFunction(collections.abc.Callable):
    '''
    Abstract class that all track functions should inherit from.
    '''
    @abstractmethod
    def __call__(self, coco_dset) -> kwcoco.CocoDataset:
        '''
        Ensure each annotation in coco_dset has a track_id.
        '''
        raise NotImplementedError('must be implemented by subclasses')

    def apply_per_video(self, coco_dset, overwrite=False):
        '''
        Main entrypoint for this class.
        '''
        for gids in coco_dset.index.vidid_to_gids.values():
            coco_dset = self.safe_apply(coco_dset, gids, overwrite)
        return coco_dset

    def safe_apply(self, coco_dset, gids, overwrite):

        sub_dset, coco_dset = self.safe_partition(coco_dset, gids, remove=True)

        existing_aids = sub_dset.anns.copy().keys()

        def tracks(annots):
            return annots.get('track_id', None)

        def are_trackless(annots):
            return np.array(tracks(annots)) == None  # noqa

        if overwrite:

            sub_dset = self(sub_dset)

        else:
            existing_tracks = tracks(sub_dset.annots())
            _are_trackless = are_trackless(sub_dset.annots())
            if np.any(_are_trackless) or len(existing_tracks) == 0:

                sub_dset = self(sub_dset)

                # if new annots were not created, rollover the old tracks
                annots = sub_dset.annots()
                if annots.aids == existing_aids:
                    annots.set(
                        'track_id',
                        np.where(_are_trackless, tracks(annots),
                                 existing_tracks))

        assert not any(are_trackless(sub_dset.annots()))
        return self.safe_union(coco_dset, sub_dset)

    @staticmethod
    def safe_partition(coco_dset, gids, remove=True):
        assert set(gids).issubset(coco_dset.imgs.keys())
        sub_dset = coco_dset.subset(gids=gids, copy=True)  # copy necessary?
        # HACK ensure tracks are not duplicated between videos
        # (if they are, this is fixed in dedupe_tracks anyway)
        sub_dset.index.trackid_to_aids.update(coco_dset.index.trackid_to_aids)
        if remove:
            coco_dset = coco_dset.subset(coco_dset.imgs.keys() - gids)
            return sub_dset, coco_dset
        else:
            return sub_dset

    @staticmethod
    def safe_union(coco_dset, new_dset, existing_aids=[]):
        coco_dset._build_index()
        new_dset._build_index()

        if 0:
            # could maybe use coco_dset.union, but it doesn't reuse IDs
            # TODO an ensure_annotations to do this properly
            # coco_dset.anns.update(sub_dset.anns)
            for cat in new_dset.cats.values():
                cat.pop('id')
                coco_dset.ensure_category(**cat)

            coco_dset.remove_annotations(existing_aids)
            anns_to_add = new_dset.anns.copy().values()
            for ann in anns_to_add:
                ann.pop('id')
                coco_dset.add_annotation(**ann)
            return coco_dset
        else:
            return coco_dset.union(new_dset, disjoint_tracks=True)


class NoOpTrackFunction(TrackFunction):
    '''
    Use existing tracks.
    '''
    def __call__(self, coco_dset):
        return coco_dset


class NewTrackFunction(TrackFunction):
    '''
    Specialization of TrackFunction to create polygons that do not yet exist
    in coco_dset, and add them as new annotations
    '''
    def __call__(self, coco_dset):
        tracks = self.create_tracks(coco_dset)
        coco_dset = self.add_tracks_to_dset(coco_dset, tracks)
        return coco_dset

    @abstractmethod
    def create_tracks(self, coco_dset) -> Iterable[Track]:
        raise NotImplementedError('must be implemented by subclasses')

    @abstractmethod
    def add_tracks_to_dset(self, coco_dset,
                           tracks: Iterable[Track]) -> kwcoco.CocoDataset:
        raise NotImplementedError('must be implemented by subclasses')


def score(poly, probs, mode='score', threshold=0):
    '''
    Args:
        poly: kwimage.Polygon or MultiPolygon in pixel coords

        probs: heatmap to compare poly against

        mode: return value.
            'score': fraction of probs contained in poly
            'response': average value of probs in poly
            'overlap': fraction of poly with probs > threshold

        threshold: only used for mode='overlap'
    '''
    if 0:
        # naive computation across the whole image
        poly_mask = poly.to_mask(probs.shape).numpy().data
        rel_mask, rel_probs = poly_mask, probs
    else:
        # First compute the valid bounds of the polygon
        # And create a mask for only the valid region of the polygon
        box = poly.bounding_box().quantize().to_xywh()
        # Ensure w/h are positive
        box.data[:, 2:4] = np.maximum(box.data[:, 2:4], 1)
        x, y, w, h = box.data[0]
        if 1:  # rasterio inverse
            rel_poly = poly.translate((0.5 - x, 0.5 - y))
            rel_mask = np.zeros((h, w))
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


def mask_to_polygons(probs, thresh, bounds=None, scored=False):
    """
    Args:
        probs: aka heatmap, image of probability values
        thresh: to turn probs into a hard mask
        bounds: a kwimage or shapely polygon to crop the results to
    Example:
        >>> from watch.tasks.tracking.from_heatmap import *  # NOQA
        >>> import kwimage
        >>> probs = kwimage.Heatmap.random(dims=(64, 64), rng=0).data['class_probs'][0]
        >>> thresh = 0.5
        >>> polys = mask_to_polygons(probs, thresh, scored=True)
        >>> poly1, score1 = list(polys)[0]
        >>> # xdoctest: +IGNORE_WANT
        >>> import kwplot
        >>> kwplot.autompl()
        >>> kwplot.imshow(probs > 0.5)
    """
    # Threshold scores
    binary_mask = probs > thresh
    shapes = features.shapes(binary_mask.astype(np.int16))

    if bounds is not None:
        try:  # is this a shapely or geojson object?
            bounds = shapely.geometry.asShape(bounds)
        except ValueError:  # is this a kwimage object?
            bounds = bounds.to_shapely()
        shp_polygons = [
            shapely.geometry.asShape(s).intersection(bounds) for s, v in shapes
            if v
        ]
        polygons = [
            kwimage.Polygon.from_shapely(s).translate((-0.5, -0.5))
            for s in shp_polygons if not s.is_empty
        ]

    else:
        polygons = [
            kwimage.Polygon.from_geojson(s).translate((-0.5, -0.5))
            for s, v in shapes if v
        ]

    if scored:
        for poly in polygons:
            yield score(poly, probs), poly
    else:
        yield from polygons


def mask_to_scored_polygons(probs, thresh):
    '''
    For backwards compatibility.
    '''
    return mask_to_polygons(probs, thresh, scored=True)


def _validate_keys(key, bg_key):
    # for backwards compatibility
    if isinstance(key, str):
        key = [key]
    elif isinstance(key, tuple):
        key = list(key)
    if bg_key is None:
        bg_key = []
    elif isinstance(bg_key, str):
        bg_key = [bg_key]
    elif isinstance(bg_key, tuple):
        bg_key = list(bg_key)

    # error checking
    if len(key) < 1:
        raise ValueError('must have at least one key')
    if (len(key) > len(set(key)) or len(bg_key) > len(set(bg_key))):
        raise ValueError('keys are duplicated')
    if not set(key).isdisjoint(set(bg_key)):
        raise ValueError('cannot have a key in foreground and background')
    return key, bg_key


def heatmap(dset, gid, key, return_chan_probs=False, space='video'):
    """
    Find the total heatmap of key within gid

    Args:
        dset: kwcoco.CocoDataset
        gid: image id
        key: List[str] list of channel names
        return_chan_probs:
            if True, also return a dict {k:heatmap(k) for k in keys}
        space: 'video' or 'image'
    """
    key, _ = _validate_keys(key, None)
    img = dset.index.imgs[gid]
    coco_img = kwcoco_extensions.CocoImage(img, dset)
    w, h = coco_img.delay(space=space).dsize
    fg_img_probs = np.zeros((h, w))
    common = kwcoco.FusedChannelSpec.coerce(
        coco_img.channels.fuse()).intersection(
            kwcoco.FusedChannelSpec.coerce(key))
    assert len(key) == len(common), (dset, gid, key)

    if len(key) == 0:  # for bg_key
        if return_chan_probs:
            return fg_img_probs, {}
        else:
            return fg_img_probs

    key_img_probs = coco_img.delay(common, space=space).finalize()
    fg_img_probs += key_img_probs.sum(axis=-1)
    if return_chan_probs:
        # some awkwardness here from non-invertible mapping from
        # ChannelSpec to FusedChannelSpec
        chan_probs = {}
        idxs = common.component_indices()
        for k in key:
            codes = common.intersection(
                kwcoco.FusedChannelSpec.coerce(k)).code_list()
            chan_probs[k] = np.sum(
                [key_img_probs[idxs[code]] for code in codes], axis=0)
        return fg_img_probs, chan_probs
    else:
        return fg_img_probs
