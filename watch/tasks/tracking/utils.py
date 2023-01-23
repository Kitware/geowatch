import collections
import geopandas as gpd
import itertools
import kwcoco
import kwimage
import numpy as np
import pandas as pd
import shapely.geometry
import ubelt as ub
import warnings
from abc import abstractmethod
from scipy.ndimage import label as ndm_label
from typing import Union, Iterable, Optional, List, Dict


try:
    from xdev import profile
except Exception:
    profile = ub.identity


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

Poly = Union[kwimage.Polygon, kwimage.MultiPolygon]


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
        for gids in ub.ProgIter(vid_gids,
                                total=total,
                                desc='apply_per_video',
                                verbose=3):

            # Beware, in the past there was a crash here that required
            # wrapping the rest of this loop in a try/except. -csg
            sub_dset = self.safe_apply(coco_dset,
                                       gids,
                                       overwrite,
                                       legacy=legacy)
            if legacy:
                coco_dset = sub_dset
            else:
                tracked_subdsets.append(sub_dset)

        if not legacy:
            # Tracks were either updated or added.
            # In the case they were updated the existing track ids should
            # be disjoint. All new tracks should not overlap with

            _debug = 1

            from watch.utils import kwcoco_extensions
            new_trackids = kwcoco_extensions.TrackidGenerator(None)
            fixed_subdataset = []
            for sub_dset in ub.ProgIter(tracked_subdsets,
                                        desc='Ensure ok tracks',
                                        verbose=3):

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
            coco_dset = kwcoco.CocoDataset.union(*fixed_subdataset,
                                                 disjoint_tracks=False)

            if _debug:
                x = coco_dset.annots().images.get('video_id')
                y = coco_dset.annots().get('track_id')
                z = ub.group_items(x, y)
                track_to_num_videos = ub.map_vals(set, z)
                if track_to_num_videos:
                    assert max(map(len, track_to_num_videos.values())) == 1, (
                        'track belongs to multiple videos!')
        return coco_dset

    @profile
    def safe_apply(self, coco_dset, gids, overwrite, legacy=True):
        DEBUG_JSON_SERIALIZABLE = 0
        if DEBUG_JSON_SERIALIZABLE:
            from watch.utils.util_json import debug_json_unserializable

        if DEBUG_JSON_SERIALIZABLE:
            debug_json_unserializable(coco_dset.dataset,
                                      'Input to safe_apply: ')

        if legacy:
            sub_dset, rest_dset = self.safe_partition(coco_dset,
                                                      gids,
                                                      remove=True)
        else:
            sub_dset = self.safe_partition(coco_dset, gids, remove=False)

        if DEBUG_JSON_SERIALIZABLE:
            debug_json_unserializable(sub_dset.dataset, 'Before __call__')
        if overwrite:
            sub_dset = self(sub_dset)
            if DEBUG_JSON_SERIALIZABLE:
                debug_json_unserializable(sub_dset.dataset,
                                          'After __call__ (overwrite)')
        else:
            orig_annots = sub_dset.annots()
            orig_tids = orig_annots.get('track_id', None)
            orig_trackless_flags = np.array([tid is None for tid in orig_tids])
            orig_aids = list(orig_annots)

            # TODO more sophisticated way to check if we can skip self()
            sub_dset = self(sub_dset)
            if DEBUG_JSON_SERIALIZABLE:
                debug_json_unserializable(sub_dset.dataset, 'After __call__')

            # if new annots were not created, rollover the old tracks
            new_annots = sub_dset.annots()
            if new_annots.aids == orig_aids:
                new_tids = new_annots.get('track_id', None)
                # Only overwrite track ids for annots that didn't have them
                new_tids = np.where(orig_trackless_flags, new_tids, orig_tids)

                # Ensure types are json serializable
                import numbers

                def _fixtype(tid):
                    # need to keep strings the same, but integers need to be
                    # case from numpy to python ints.
                    if isinstance(tid, numbers.Integral):
                        return int(tid)
                    else:
                        return tid

                new_tids = list(map(_fixtype, new_tids))

                new_annots.set('track_id', new_tids)

        # TODO: why is this assert here?
        assert None not in sub_dset.annots().lookup('track_id', None)

        if legacy:
            out_dset = self.safe_union(rest_dset, sub_dset)
        else:
            out_dset = sub_dset

        if DEBUG_JSON_SERIALIZABLE:
            debug_json_unserializable(out_dset.dataset,
                                      'Output of safe_apply: ')
        return out_dset

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

    def __init__(self, **kwargs):
        self.kwargs = kwargs  # Unused

    def __call__(self, sub_dset):
        return sub_dset


class NewTrackFunction(TrackFunction):
    '''
    Specialization of TrackFunction to create polygons that do not yet exist
    in coco_dset, and add them as new annotations
    '''

    def __call__(self, sub_dset):
        print('Create tracks')
        tracks = self.create_tracks(sub_dset)
        print('Add tracks to dset')
        sub_dset = self.add_tracks_to_dset(sub_dset, tracks)
        print('After tracking')
        print(sub_dset.basic_stats())
        return sub_dset

    @abstractmethod
    def create_tracks(self, sub_dset) -> gpd.GeoDataFrame:
        raise NotImplementedError('must be implemented by subclasses')

    @abstractmethod
    def add_tracks_to_dset(self, sub_dset,
                           tracks: gpd.GeoDataFrame) -> kwcoco.CocoDataset:
        raise NotImplementedError('must be implemented by subclasses')


def check_only_bg(category_sequence, bg_name=['No Activity']):
    if len(set(category_sequence) - set(bg_name)) == 0:
        return True
    else:
        return False


# --- geopandas utils ---
# make this a class?


def gpd_sort_by_gid(gdf, sorted_gids):

    dct = dict(zip(sorted_gids, range(len(sorted_gids))))
    gdf['gid_order'] = gdf['gid'].map(dct)

    # assert gdf['gid'].map(dct).groupby(
    # lambda x: x).is_monotonic_increasing.all()

    return gdf.groupby('track_idx', group_keys=False).apply(
        lambda x: x.sort_values('gid_order')).reset_index(drop=True).drop(
            columns=['gid_order'])


def gpd_len(gdf):
    return gdf['track_idx'].nunique()


@profile
def gpd_compute_scores(
        gdf,
        sub_dset,
        thrs: Iterable,
        ks: Dict,
        USE_DASK=False,
        resolution=None):

    def compute_scores(grp, thrs=[], keys=[]):
        # TODO handle keys as channelcodes
        # port over better handling from utils.build_heatmaps
        gid = getattr(grp, 'name', None)
        if gid is None:
            for thr in thrs:
                grp[[(k, thr) for k in keys]] = 0
        else:
            heatmaps = []
            for k in keys:
                # TODO use nans instead of fill
                heatmap = build_heatmap(sub_dset, gid, k, missing='fill',
                                       resolution=resolution)
                heatmaps.append(heatmap)
            heatmaps = np.stack(heatmaps, axis=0)
            if 0:
                h = build_heatmaps(sub_dset, [gid], keys, missing='fill')
                heatmaps2 = np.stack([h[k][0] for k in keys], axis=0)
                # TODO why does this fail?
                # TODO deprecate build_heatmaps, interpolation behavior
                # unneeded and confusing
                assert np.allclose(heatmaps, heatmaps2)
            if 0:
                scores = grp['poly'].map(
                    lambda p: score_poly(p, heatmaps, threshold=thrs))
                scores = scores.explode().explode()
                scores.index = pd.MultiIndex.from_product((keys, thrs, grp.index))
                scores_wide = scores.reset_index([0, 1]).pivot(
                    columns=['level_0', 'level_1'], values=scores.name)
                score_cols = scores_wide.columns.values
                grp[score_cols] = scores_wide.values
            else:
                # import xdev; xdev.embed()
                score_cols = list(itertools.product(keys, thrs))
                scores = grp['poly'].apply(
                    lambda p: pd.Series(dict(zip(
                        score_cols,
                        ub.flatten(score_poly(p, heatmaps, threshold=thrs))))
                    ))
                grp[score_cols] = scores
        return grp

    ks = {k: v for k, v in ks.items() if v}
    _valid_keys = list(set().union(itertools.chain.from_iterable(
        ks.values())))  # | ks.keys()
    score_cols = list(itertools.product(_valid_keys, thrs))

    # if USE_DASK:  # 63% runtime
    if 0:
        import dask_geopandas
        # https://github.com/geopandas/dask-geopandas
        # _col_order = gdf.columns  # doesn't matter
        gdf = gdf.set_index('gid')
        # npartitions and chunksize are mutually exclusive
        gdf = dask_geopandas.from_geopandas(gdf, npartitions=8)
        meta = gdf._meta.join(pd.DataFrame(columns=score_cols, dtype=float))
        gdf = gdf.groupby('gid', group_keys=False).apply(compute_scores,
                                                         thrs=thrs,
                                                         keys=_valid_keys,
                                                         meta=meta)
        # raises this, which is probably fine:
        # /home/local/KHQ/matthew.bernstein/.local/conda/envs/watch/lib/python3.9/site-packages/rasterio/features.py:362:
        # NotGeoreferencedWarning: Dataset has no geotransform, gcps, or rpcs.
        # The identity matrix will be returned.
        # _rasterize(valid_shapes, out, transform, all_touched, merge_alg)

        gdf = gdf.compute()  # gdf is now a GeoDataFrame again
        gdf = gdf.reset_index()
        # gdf = gdf.reindex(columns=_col_order)

    else:  # 95% runtime
        grouped = gdf.groupby('gid', group_keys=False)
        gdf = grouped.apply(compute_scores, thrs=thrs, keys=_valid_keys)

    # fill nan scores from nodata pxls
    # groupby track instead of gid
    def _fillna(grp):
        if len(grp) == 0:
            grp = grp.reindex(list(ub.oset(grp.columns) | score_cols), axis=1)
        else:
            grp[score_cols] = grp[score_cols].fillna(method='ffill').fillna(0)
        return grp

    grouped = gdf.groupby('track_idx', group_keys=False)
    scored_gdf = grouped.apply(_fillna)

    # copy over to summed fg/bg channels
    for thr in thrs:
        for k, kk in ks.items():
            if kk:
                # https://github.com/pandas-dev/pandas/issues/20824#issuecomment-384432277
                sum_cols = [(ki, thr) for ki in kk]
                sum_cols = list(ub.oset(scored_gdf.columns) & sum_cols)
                scored_gdf[(k, thr)] = scored_gdf[sum_cols].sum(axis=1)

    return scored_gdf


# -----------------------


@profile
def pop_tracks(coco_dset: kwcoco.CocoDataset,
               cnames: Iterable[str],
               remove: bool = True,
               score_chan: Optional[kwcoco.ChannelSpec] = None,
               resolution: Optional[str] = None):
    '''
    Convert kwcoco annotations into tracks.

    Args:
        coco_dset
        cnames: category names
        remove: remove the annotations from coco_dset
        score_chan: score the track polygons by image overlap with this channel

    Returns:
        gpd dataframe.
        Mutates coco_dset if remove=True.
    '''
    # TODO could refactor to work on coco_dset.annots() and integrate
    cnames = list(set(cnames))

    annots = coco_dset.annots()
    annots = annots.compress(
        np.in1d(np.array(annots.cnames, dtype=str), cnames))
    if len(annots) < 1:
        print(f'warning: no {cnames} annots in dset {coco_dset.tag}!')

    # Load polygon annotation segmentation in video space
    coco_imgs = annots.images.coco_images
    polys = []
    for coco_img, ann in zip(coco_imgs, annots.objs):
        poly = coco_img._annot_segmentation(ann, space='video')
        polys.append(poly)

    assert len(polys) == len(annots), ('TODO handle multipolygon boundaries')

    polys = [p.to_shapely() for p in polys]
    gdf = gpd.GeoDataFrame(dict(gid=annots.gids,
                                poly=polys,
                                track_idx=annots.get('track_id')),
                           geometry='poly')
    if score_chan is not None:
        keys = {score_chan.spec: list(score_chan.unique())}
        gdf = gpd_compute_scores(gdf, coco_dset, [-1], keys, USE_DASK=True,
                                 resolution=resolution)
    # TODO standard way to access sorted_gids
    sorted_gids = coco_dset.index._set_sorted_by_frame_index(
        np.unique(annots.gids))
    gdf = gpd_sort_by_gid(gdf, sorted_gids)

    if remove:
        coco_dset.remove_categories(cnames, keep_annots=False)

    return gdf


@profile
def score_poly(poly, probs, threshold=-1, use_rasterio=True):
    '''
    Args:
        poly: kwimage.Polygon or MultiPolygon in pixel coords

        probs: heatmap to compare poly against. Any leading batch dimensions
        will be preserved in output, e.g. (gid chan w h) -> (gid chan)

        use_rasterio: use rasterio.features module instead of kwimage

        threshold: Return fraction of poly with probs > threshold.
        If -1, return average value of probs in poly. Can be a list of values,
        in which case returns all of them.

    '''
    # try converting from shapely
    # TODO standard coerce fns between kwimage, shapely, and __geo_interface__
    if not isinstance(poly, (kwimage.Polygon, kwimage.MultiPolygon)):
        poly = kwimage.MultiPolygon.from_shapely(poly)  # 2.4% of runtime

    # First compute the valid bounds of the polygon
    # And create a mask for only the valid region of the polygon
    box = poly.bounding_box().quantize().to_xywh()
    # Ensure box is inside probs
    ymax, xmax = probs.shape[-2:]
    box = box.clip(0, 0, xmax, ymax).to_xywh()
    if box.area[0][0] == 0:
        warnings.warn(
            'warning: scoring a polygon against an img with no overlap!')
        return 0
    x, y, w, h = box.data[0]
    pixels_are = 'areas' if use_rasterio else 'points'
    # kwimage inverse
    # 95% of runtime... would batch be faster?
    rel_poly = poly.translate((-x, -y))
    rel_mask = rel_poly.to_mask((h, w), pixels_are=pixels_are).data
    # Slice out the corresponding region of probabilities
    rel_probs = probs[..., y:y + h, x:x + w]

    _return_list = isinstance(threshold, Iterable)
    if not _return_list:
        threshold = [threshold]
    result = []

    # handle nans
    msk = (np.isfinite(rel_probs) * rel_mask).astype(bool)
    for t in threshold:
        if not msk.any():
            result.append(np.nan * np.ones(rel_probs.shape[:-2]))
        elif t == -1:
            mskd = np.ma.array(rel_probs, mask=~msk)
            result.append(mskd.mean(axis=(-2, -1)))
        else:
            hard_prob = rel_probs > t
            mskd = np.ma.array(hard_prob, mask=~msk)
            result.append(mskd.mean(axis=(-2, -1)))

    return result if _return_list else result[0]


@profile
def mask_to_polygons(probs,
                     thresh,
                     bounds=None,
                     use_rasterio=True,
                     thresh_hysteresis=None):
    """
    Args:
        probs: aka heatmap, image of probability values
        thresh: to turn probs into a hard mask
        bounds: a kwimage or shapely polygon to crop the results to
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
        >>> polys = mask_to_polygons(probs, thresh)
        >>> poly1 = list(polys)[0]
        >>> # xdoctest: +REQUIRES(--show)
        >>> import kwplot
        >>> kwplot.autompl()
        >>> kwplot.imshow(probs > 0.5)

    Example:
        >>> from watch.tasks.tracking.utils import mask_to_polygons
        >>> import kwimage
        >>> import kwarray
        >>> rng = kwarray.ensure_rng(1043462368)
        >>> probs = kwimage.Heatmap.random(dims=(256, 256), rng=rng,
        >>>                                 ).data['class_probs'][0]
        >>> thresh = 0.5
        >>> polys1 = list(mask_to_polygons(
        >>>             probs, thresh, use_rasterio=0))
        >>> polys2 = list(mask_to_polygons(
        >>>             probs, thresh, use_rasterio=1))
        >>> # xdoctest: +REQUIRES(--show)
        >>> import kwplot
        >>> kwplot.autompl()
        >>> plt = kwplot.autoplt()
        >>> pnum_ = kwplot.PlotNums(nSubplots=4)
        >>> kwplot.imshow(probs, pnum=pnum_(), title='pixels_are=points')
        >>> for poly in polys1:
        >>>     poly.draw(facecolor='none', edgecolor='kitware_blue', alpha=0.5, linewidth=8)
        >>> kwplot.imshow(probs, pnum=pnum_(), title='pixels_are=areas')
        >>> for poly in polys2:
        >>>     poly.draw(facecolor='none', edgecolor='kitware_green', alpha=0.5, linewidth=8)
    """
    # Threshold scores
    if thresh_hysteresis is None:
        binary_mask = (probs > thresh).astype(np.uint8)
    else:
        mask = probs > thresh
        seeds = probs > thresh_hysteresis
        label_img = ndm_label(mask)[0]
        selected = np.unique(np.extract(seeds, label_img))
        binary_mask = np.isin(label_img, selected).astype(np.uint8)

    pixels_are = 'areas' if use_rasterio else 'points'
    if bounds is not None:
        try:  # is this a shapely or geojson object?
            # asShape is being deprecated:
            # https://github.com/shapely/shapely/issues/1100
            bounds = shapely.geometry.shape(bounds)
        except ValueError:  # is this a kwimage object?
            bounds = bounds.to_shapely()
        bounds_mask = kwimage.Polygon.from_shapely(bounds).to_mask(
            probs.shape, pixels_are=pixels_are).numpy().data.astype(np.uint8)
        binary_mask *= bounds_mask

    polygons = kwimage.Mask(
        binary_mask, 'c_mask').to_multi_polygon(pixels_are=pixels_are)

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


@profile
def build_heatmaps(
        sub_dset: kwcoco.CocoDataset,
        gids: List[int],
        # TODO debug List
        keys: Union[List[str], Dict[str, List[str]]],
        missing='fill',
        skipped='interpolate',
        space='video',
        resolution=None,
        video_id=None) -> Dict[str, List[np.array]]:
    '''
    Vectorized version of watch.tasks.tracking.utils.build_heatmap across gids.

    Can also sum keys using group names.

    Restrictions wrt heatmap():
        - uses video space
        - returns chan probs

    Args:
        sub_dset (kwcoco.CocoDataset): must have exactly 1 video

        gids: List[image id]

        key: List[str] list of channel names

        space (str):
            The "space" the heatmaps are loaded in.
            Can be 'video' or 'image'. Should generally be "video".

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

        resolution (str | None):
            desired resolution (e.g. 10GSD) that will define a scale factor
            on top of the "space" (i.e. video space) used to build the heatmaps.

    SeeAlso:
        :func:`build_heatmap`
        :func:`build_heatmaps`

    Returns:
        Dict : {key: [heatmap for each gid]}

    Example:
        >>> from watch.tasks.tracking.utils import *  # NOQA
        >>> import watch
        >>> dset = watch.coerce_kwcoco('watch-msi', heatmap=True, geodata=True, dates=True)
        >>> keys = 'salient|notsalient|distri'
        >>> videos = dset.videos()
        >>> video_id = videos._ids[0]
        >>> gids = videos.images[0][0:2]
        >>> heatmaps = build_heatmaps(dset, gids=gids, keys=keys, video_id=video_id)
        >>> assert all(len(v) == len(gids) for v in heatmaps.values())
        >>> heatmaps = build_heatmaps(dset, gids=gids, keys={'group1': ['salient', 'notsalient']}, video_id=video_id)
        >>> assert all(len(v) == len(gids) for v in heatmaps.values())
    '''
    assert space == 'video'

    if isinstance(keys, list):
        _dummy_key = '__dummy__'
        key_groups = {'__dummy__': keys}
    elif isinstance(keys, dict):
        _dummy_key = None
        key_groups = keys
    elif isinstance(keys, str):
        import kwcoco
        _dummy_key = '__dummy__'
        key_groups = {'__dummy__': kwcoco.FusedChannelSpec.coerce(keys).to_list()}
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
    first_coco_img = sub_dset.coco_image(
        sub_dset.images(video_id=video_id).peek()['id'])
    # should this be special-cased in _scalefactor_for_resolution?
    if resolution is not None:
        scale_trk_from_vid = first_coco_img._scalefactor_for_resolution(
            space='video', resolution=resolution)
        trk_shape = (np.array(vid_shape) * scale_trk_from_vid).astype(int)
    else:
        trk_shape = vid_shape

    prev_heatmap_dct = collections.defaultdict(lambda: np.zeros(trk_shape))

    for gid in gids:
        for group, key in key_groups.items():

            # we are working only in vid space, so forget about warping
            img_probs, chan_probs = build_heatmap(sub_dset,
                                                  gid,
                                                  key,
                                                  space=space,
                                                  resolution=resolution,
                                                  return_chan_probs=True)
            # TODO make this more efficient using missing='skip'
            if np.any(img_probs):
                heatmaps_dct[group].append(img_probs)
            elif skipped == 'interpolate':
                heatmaps_dct[group].append(prev_heatmap_dct[group])
            elif skipped == 'zeros':
                heatmaps_dct[group].append(np.zeros(trk_shape))
            else:
                raise ValueError(skipped)

            for k in key:
                if k in chan_probs:
                    heatmaps_dct[k].append(chan_probs[k])
                    prev_heatmap_dct[k] = chan_probs[k]
                elif skipped == 'interpolate':
                    heatmaps_dct[k].append(prev_heatmap_dct[k])
                elif skipped == 'zeros':
                    heatmaps_dct[k].append(np.zeros(trk_shape))
                else:
                    raise ValueError(skipped)

    if _dummy_key is not None:
        heatmaps_dct.pop(_dummy_key)
    return heatmaps_dct


@profile
def build_heatmap(dset,
                  gid,
                  key,
                  return_chan_probs=False,
                  space='video',
                  missing='fill',
                  resolution=None):
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

    SeeAlso:
        :func:`build_heatmap`
        :func:`build_heatmaps`

    Example:
        >>> from watch.tasks.tracking.utils import *  # NOQA
        >>> import watch
        >>> dset = watch.coerce_kwcoco(
        >>>     data='watch-msi', heatmap=True, geodata=True, dates=True)
        >>> gid = dset.images()[0]
        >>> key = 'salient'
        >>> space = 'video'
        >>> missing = 'fill'
        >>> # With probs
        >>> return_chan_probs = True
        >>> fg_img_probs1, chan_probs = build_heatmap(dset, gid, key, return_chan_probs, space, missing)
        >>> # FG only
        >>> return_chan_probs = False
        >>> fg_img_probs2 = build_heatmap(dset, gid, key, return_chan_probs, space, missing)
        >>> #
        >>> # Test with a non-existing key
        >>> return_chan_probs = True
        >>> key = 'eludium'
        >>> fg_img_probs1, chan_probs = build_heatmap(dset, gid, key, return_chan_probs, space, missing)

    Example:
        >>> from watch.tasks.tracking.utils import *  # NOQA
        >>> import watch
        >>> dset = watch.coerce_kwcoco(data='watch-msi', heatmap=True, geodata=True, dates=True)
        >>> gid = dset.images()[0]
        >>> key = 'salient'
        >>> # Test dynamic resolution
        >>> fg_img_probs1 = build_heatmap(dset, gid, key, resolution='3GSD')
        >>> fg_img_probs2 = build_heatmap(dset, gid, key, resolution='7GSD')
        >>> a = np.array(fg_img_probs1.shape)
        >>> b = np.array(fg_img_probs2.shape)
        >>> assert np.isclose((a / b).mean(), 7 / 3, atol=.1)
    """
    key, _ = _validate_keys(key, None)
    coco_img = dset.coco_image(gid)

    channels_request = kwcoco.FusedChannelSpec.coerce(key)
    channels_have = coco_img.channels.fuse().intersection(channels_request)

    if missing == 'raise':
        if channels_have.numel() != channels_request.numel():
            raise ValueError(
                ub.paragraph(f'''
                Requested {channels_request=} in the image {gid=} of {dset=}
                but only {channels_have=} existed.
                '''))

    w, h = coco_img.delay(space=space).dsize
    if resolution is not None:
        scale_trk_from_vid = coco_img._scalefactor_for_resolution(
            space='video', resolution=resolution)
        w, h = (np.array((w, h)) * scale_trk_from_vid).astype(int)

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

    delayed = coco_img.delay(
        channels=common, resolution=resolution, space=space,
        nodata_method='float')
    key_img_probs = delayed.finalize()

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
