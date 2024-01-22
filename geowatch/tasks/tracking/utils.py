import itertools
import ubelt as ub
import warnings
from functools import lru_cache
from typing import Iterable, Optional, Dict

try:
    from xdev import profile
except Exception:
    profile = ub.identity


def trackid_is_default(trackid):
    """
    Hack to decide if a trackid is really a site_id or if it was randomly
    assigned
    """
    if trackid is None:
        return True
    try:
        int(trackid)
        return True
    except ValueError:
        return False


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

    if gdf.empty:
        return gdf
    else:
        return gdf.groupby('track_idx', group_keys=False).apply(
            lambda x: x.sort_values('gid_order')).reset_index(drop=True).drop(
                columns=['gid_order'])


def gpd_len(gdf):
    return gdf['track_idx'].nunique()


@profile
def gpd_compute_scores(gdf, sub_dset, thrs: Iterable, ks: Dict, USE_DASK=False,
                       resolution=None, modulate=None):
    """
    TODO: This needs docs and examples for the BAS and SC/AC cases.

    Args:
        gdf (gdf.GeoDataFrame):
            input data frame tracks dataframe containing
            track_idx, gid, and poly columns.

        sub_dset (kwcoco.CocoDataset):
            dataset with reference to images

        thrs (List[float]):
            thresholds (-1) means take the average response, other values is
            the fraction of pixels with responses above that value.

        ks (Dict[str, List[str]]):
            mapping from "fg" to a list of "foreground classes"
            optionally also
            mapping from "bg" to a list of "background classes"

        resolution (str | None):
            resolution spec to compute scores at (e.g. "2GSD").

    Calls :func:`_compute_group_scores` on each dataframe row, which will
    execute the read for the image prediction scores for polygons with
    :func:`score_poly`.

    Returns:
        gdf.GeoDataFrame - a scored variant of the input data frame

             returns the per-channels scores as well as summed groups of
             channels. (not sure if that last one is necessary, might need to
             refactor to simplify)

    Example:
        >>> import kwcoco
        >>> from geowatch.tasks.tracking.utils import *  # NOQA
        >>> from geowatch.tasks.tracking.utils import _compute_group_scores, _build_annot_gdf
        >>> sub_dset = kwcoco.CocoDataset.demo('vidshapes1')
        >>> gdf, _ = _build_annot_gdf(sub_dset)
        >>> thrs = [-1, 'median']
        >>> ks = {'r|g': ['r', 'g'], 'bg': ['b']}
        >>> USE_DASK = 0
        >>> resolution = None
        >>> gdf2 = gpd_compute_scores(gdf, sub_dset, thrs, ks, USE_DASK, resolution)
    """
    import pandas as pd

    ks = {k: v for k, v in ks.items() if v}
    _valid_keys = list(set().union(itertools.chain.from_iterable(
        ks.values())))  # | ks.keys()

    # score_cols = list(itertools.product(_valid_keys, thrs))
    score_cols = [t[::-1] for t in itertools.product(thrs, _valid_keys)]

    if USE_DASK:  # 63% runtime
        import dask_geopandas
        # https://github.com/geopandas/dask-geopandas
        # _col_order = gdf.columns  # doesn't matter
        gdf = gdf.set_index('gid')
        # npartitions and chunksize are mutually exclusive
        gdf = dask_geopandas.from_geopandas(gdf, npartitions=8)
        meta = gdf._meta.join(pd.DataFrame(columns=score_cols, dtype=float))
        groups = gdf.groupby('gid', group_keys=False)
        gdf = groups.apply(_compute_group_scores,
                           thrs=thrs,
                           keys=_valid_keys,
                           meta=meta,
                           resolution=resolution,
                           sub_dset=sub_dset, modulate=modulate)
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
        """
        grp = gdf.iloc[0:1]
        grp.name = grp.iloc[0].gid
        """
        gdf = grouped.apply(_compute_group_scores, thrs=thrs, _valid_keys=_valid_keys,
                            resolution=resolution, sub_dset=sub_dset, modulate=modulate)

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


def _compute_group_scores(grp, thrs=[], _valid_keys=[], resolution=None,
                          sub_dset=None, modulate=None):
    """
    Helper for :func:`gpd_compute_scores`.
    """
    import kwcoco
    import pandas as pd
    """
    Note:
        "name" is an attribute groupby only seems to give in the apply step.

        We can get a reference to the group object via:

            obj = list(grouped._iterate_slices())[0]
            ??? not sure if this is right

        The following is a MWE:

    Ignore:
        # Test groupby
        import pandas as pd
        df = pd.DataFrame({'a': [1, 2, 3], 'b': [2, 2, 1]})
        def foo(grp):
            print(f'grp.name={grp.name}')
            print(type(grp))
            print(f'grp={grp}')
        groups = df.groupby('b',group_keys=False)
        grp = list(groups._iterate_slices())[0]
        groups.apply(foo)

    Args:
        grp : A Pandas Group Object for data in the same iamge

    Example:
        >>> import kwcoco
        >>> from geowatch.tasks.tracking.utils import *  # NOQA
        >>> from geowatch.tasks.tracking.utils import _compute_group_scores, _build_annot_gdf
        >>> _valid_keys = ['r', 'g', 'b']
        >>> sub_dset = kwcoco.CocoDataset.demo('vidshapes1')
        >>> aids = list(sub_dset.images().annots[0])
        >>> grp, _ = _build_annot_gdf(sub_dset, aids=aids)
        >>> thrs = [-1, 'mean', 'max', 'min', 'median']
        >>> modulate = {'r': 0.0001}
        >>> gdf2 = _compute_group_scores(grp, thrs=thrs, _valid_keys=_valid_keys, sub_dset=sub_dset, modulate=modulate)
        >>> print(gdf2)
    """
    import numpy as np
    gid = getattr(grp, 'name', None)
    if gid is None:
        if len(grp) > 0:
            gid = grp.iloc[0]['gid']

    if gid is None:
        for thr in thrs:
            grp[[(k, thr) for k in _valid_keys]] = 0
    else:
        img = sub_dset.coco_image(gid)

        # Load the channels to score
        channels = kwcoco.FusedChannelSpec.coerce(_valid_keys)
        heatmaps_hwc = img.imdelay(channels, space='video', resolution=resolution).finalize()
        heatmaps_chw = heatmaps_hwc.transpose(2, 0, 1)
        if heatmaps_chw.dtype.kind != 'f':
            heatmaps_chw = heatmaps_chw.astype(np.float32)
        heatmaps_chw = np.ascontiguousarray(heatmaps_chw)
        if modulate is not None:
            chan_list = channels.to_list()
            for k, v in modulate.items():
                idx = chan_list.index(k)
                assert idx >= 0
                heatmaps_chw[idx] *= v

        assert isinstance(thrs, list)
        # score_cols = list(itertools.product(_valid_keys, thrs))
        score_cols = [t[::-1] for t in itertools.product(thrs, _valid_keys)]

        # Compute scores for each polygon.
        new_scores_rows = []
        for poly in grp['poly']:
            poly_scores_ = score_poly(poly, heatmaps_chw, threshold=thrs)
            # awk, making this serializable for kwcoco dataset
            poly_scores = list(ub.flatten(poly_scores_))
            col_to_score = dict(zip(score_cols, poly_scores))
            new_scores_rows.append(pd.Series(col_to_score))
        grp[score_cols] = new_scores_rows

        # scores = grp['poly'].apply(
        #     lambda p: pd.Series(dict(zip(
        #         score_cols,
        #         # awk, making this serializable for kwcoco dataset
        #         list(ub.flatten(score_poly(p, heatmaps_chw, threshold=thrs)))))
        #     ))
        # grp[score_cols] = scores
    return grp


# -----------------------


@profile
def score_track_polys(coco_dset,
                      video_id,
                      cnames=None,
                      score_chan=None,
                      resolution: Optional[str] = None):
    """
    Score the polygons in a kwcoco dataset based on heatmaps without modifying
    the polygon boundaries.

    Args:
        coco_dset (kwcoco.CocoDataset):

        video_id (int): video to score tracks for

        cnames (Iterable[str] | None):
            category names. Only annotations with these names will be
            considered.

        score_chan (kwcoco.ChannelSpec | None):
            score the track polygons by image overlap with this channel

    Note:
        This function needs a rename because we don't want this to mutate the
        kwcoco dataset ever.

    Returns:
        gpd.GeoDataFrame:
            With columns:
                gid: the image id
                poly: the polygon in video space
                track_idx: the annotation trackid

                And then for each score chan: c you get a column:
                    (c, -1) where the -1 indicates that no threshold was
                    applied, and that this is the mean of that channel
                    intensity under the polygon.

                Then there is another column where all channels are fused: f
                and you get a column: (f, -1)

    Note:
        The returned unerlying GDF should return polygons in video space as it
        will be consumed by :func:`_add_tracks_to_dset`.

    Example:
        >>> import kwcoco
        >>> coco_dset = kwcoco.CocoDataset.demo('vidshapes8-msi')
        >>> video_id = list(coco_dset.videos())[0]
        >>> cnames = None
        >>> resolution = None
        >>> score_chan = kwcoco.ChannelSpec.coerce('B1|B8|B8a|B10|B11')
        >>> gdf = score_track_polys(coco_dset, video_id, cnames, score_chan, resolution)
        >>> print(gdf)
    """
    # TODO could refactor to work on coco_dset.annots() and integrate
    import numpy as np

    aids = list(ub.flatten(coco_dset.images(video_id=video_id).annots))
    annots = coco_dset.annots(aids)
    # annots = coco_dset.annots()
    gdf, flat_scales = _build_annot_gdf(
        coco_dset, aids=aids, cnames=cnames, resolution=resolution)

    if score_chan is not None:
        # USE_DASK = True
        USE_DASK = False
        keys = {score_chan.spec: list(score_chan.unique())}
        sub_dset = coco_dset
        thrs = [-1]
        ks = keys
        gdf = gpd_compute_scores(gdf, sub_dset, thrs, ks, USE_DASK=USE_DASK,
                                 resolution=resolution)
    # TODO standard way to access sorted_gids
    sorted_gids = coco_dset.index._set_sorted_by_frame_index(
        np.unique(annots.gids))
    gdf = gpd_sort_by_gid(gdf, sorted_gids)

    if resolution is not None:
        # It should be the case that all of the scale factors are the same
        # because it is wrt to video space. Check for this and then
        # just apply a single warp.
        assert ub.allsame(flat_scales)
        if flat_scales:
            inverse_scale = 1 / flat_scales[0][0], 1 / flat_scales[0][1]
            gdf['poly'] = gdf['poly'].scale(
                xfact=inverse_scale[0],
                yfact=inverse_scale[1],
                origin=(0, 0, 0))

    return gdf


def _build_annot_gdf(coco_dset, aids=None, cnames=None, resolution=None):
    import geopandas as gpd
    import numpy as np

    annots = coco_dset.annots(aids)

    if cnames is not None:
        cnames = list(set(cnames))
        annots = annots.compress(
            np.in1d(np.array(annots.cnames, dtype=str), cnames))
        if len(annots) < 1:
            print(f'warning: no cnames={cnames} annots in dset dset.tag={coco_dset.tag}!')

    # Load polygon annotation segmentation in video space at the target
    # resolution
    gids = annots.images.ids
    gid_to_anns = ub.group_items(annots.objs, gids)

    flat_polys = []
    flat_gids = []
    flat_track_ids = []
    flat_scales = []
    for image_id, anns in gid_to_anns.items():
        coco_img = coco_dset.coco_image(image_id)

        img_polys = coco_img._annot_segmentations(anns, space='video',
                                                  resolution=resolution)
        flat_polys.extend(img_polys)
        flat_gids.extend([image_id] * len(img_polys))
        flat_track_ids.extend([ann['track_id'] for ann in anns])
        if resolution is not None:
            # Need to remember the inverse scale factor to get back to video
            # space.
            scale = coco_img._scalefactor_for_resolution(space='video',
                                                         resolution=resolution)
            flat_scales.append(scale)

    assert len(flat_polys) == len(annots), ('TODO handle multipolygon boundaries')

    flat_polys = [p.to_shapely() for p in flat_polys]
    gdf = gpd.GeoDataFrame({
        'gid': flat_gids,
        'poly': flat_polys,
        'track_idx': flat_track_ids,
    }, geometry='poly')
    return gdf, flat_scales


@lru_cache(maxsize=512)
def _rasterized_poly(shp_poly, h, w, pixels_are):
    import kwimage
    poly = kwimage.MultiPolygon.from_shapely(shp_poly)
    mask = poly.to_mask((h, w), pixels_are=pixels_are).data
    return mask


@profile
def score_poly(poly, probs, threshold=-1, use_rasterio=True):
    """
    Compute the average heatmap response of a heatmap inside of a polygon.

    Args:
        poly (kwimage.Polygon | MultiPolygon):
            in pixel coords

        probs (ndarray):
            heatmap to compare poly against in [..., c, h, w] format.
            The last two dimensions should be height, and width.
            Any leading batch dimensions will be preserved in output,
            e.g. (gid, chan, h, w) -> (gid, chan)

        use_rasterio (bool):
            use rasterio.features module instead of kwimage

        threshold (float | List[float | str]):
            Return fraction of poly with probs > threshold.  If -1, return
            average value of probs in poly. Can be a list of values, in which
            case returns all of them.

    Returns:
        List[ndarray] | ndarray:

            When thresholds is a list, returns a corresponding list of ndarrays
            with an entry keeping the leading dimensions of probs and
            marginalizing over the last two.

    Example:
        >>> import numpy as np
        >>> import kwimage
        >>> from geowatch.tasks.tracking.utils import score_poly
        >>> h = w = 64
        >>> poly = kwimage.Polygon.random().scale((w, h))
        >>> probs = np.random.rand(1, 3, h, w)
        >>> # Test with one threshold
        >>> threshold = [0.1, 0.2]
        >>> result = score_poly(poly, probs, threshold=threshold, use_rasterio=True)
        >>> print('result = {}'.format(ub.urepr(result, nl=1)))
        >>> # Test with multiple thresholds
        >>> threshold = 0.1
        >>> result = score_poly(poly, probs, threshold=threshold, use_rasterio=True)
        >>> print('result = {}'.format(ub.urepr(result, nl=1)))
        >>> # Test with -1 threshold
        >>> threshold = [-1, 'min', 'median']
        >>> result = score_poly(poly, probs, threshold=threshold, use_rasterio=True)
        >>> print('result = {}'.format(ub.urepr(result, nl=1)))

    Example:
        ### Grid of cases

        basis = {
            'threshold':
        }

    """
    import kwimage
    import numpy as np
    if not isinstance(poly, (kwimage.Polygon, kwimage.MultiPolygon)):
        poly = kwimage.MultiPolygon.from_shapely(poly)  # 2.4% of runtime

    _return_list = isinstance(threshold, Iterable)
    if not _return_list:
        threshold = [threshold]

    # First compute the valid bounds of the polygon
    # And create a mask for only the valid region of the polygon

    box = poly.box().quantize().to_xywh()

    # Ensure box is inside probs
    ymax, xmax = probs.shape[-2:]
    box = box.clip(0, 0, xmax, ymax).to_xywh()
    if box.area == 0:
        warnings.warn(
            'warning: scoring a polygon against an img with no overlap!')
        zeros = np.zeros(probs.shape[:-2])
        return [zeros] * len(threshold) if _return_list else zeros
    # sl_y, sl_x = box.to_slice()
    x, y, w, h = box.data
    pixels_are = 'areas' if use_rasterio else 'points'
    # kwimage inverse
    # 95% of runtime... would batch be faster?
    rel_poly = poly.translate((-x, -y))
    # rel_mask = rel_poly.to_mask((h, w), pixels_are=pixels_are).data
    # shapely polys hash correctly (based on shape, not memory location)
    # kwimage polys don't
    rel_mask = _rasterized_poly(rel_poly.to_shapely(), h, w, pixels_are)
    # Slice out the corresponding region of probabilities
    rel_probs = probs[..., y:y + h, x:x + w]

    result = []

    # handle nans
    msk = (np.isfinite(rel_probs) * rel_mask).astype(bool)
    all_non_finite = not msk.any()

    mskd_rel_probs = np.ma.array(rel_probs, mask=~msk)

    for t in threshold:
        if all_non_finite:
            stat = np.full(rel_probs.shape[:-2], fill_value=np.nan)
        elif t == 'max':
            stat = mskd_rel_probs.max(axis=(-2, -1)).filled(0)
        elif t == 'min':
            stat = mskd_rel_probs.min(axis=(-2, -1)).filled(0)
        elif t == 'mean':
            stat = mskd_rel_probs.mean(axis=(-2, -1)).filled(0)
        elif t == 'median':
            stat = np.ma.median(mskd_rel_probs, axis=(-2, -1)).filled(0)
        elif t == -1:
            # Alias for mean, todo: deprecate
            stat = mskd_rel_probs.mean(axis=(-2, -1)).filled(0)
        else:
            # Real threshold case
            hard_prob = rel_probs > t
            mskd_hard_prob = np.ma.array(hard_prob, mask=~msk)
            stat = mskd_hard_prob.mean(axis=(-2, -1)).filled(0)
        result.append(stat)

    return result if _return_list else result[0]


@profile
def mask_to_polygons(probs,
                     thresh,
                     bounds=None,
                     use_rasterio=True,
                     thresh_hysteresis=None):
    """
    Extract a polygon from a 2D heatmap. Optionally within the bounds of
    another mask or polygon.

    Args:
        probs (ndarray): aka heatmap, image of probability values

        thresh (float): to turn probs into a hard mask

        bounds (kwimage.Polygon): a kwimage or shapely polygon to crop the results to

        use_rasterio (bool): use rasterio.features module instead of kwimage

        thresh_hysteresis: if not None, only keep polygons with at least one
            pixel of score >= thresh_hysteresis

    Yields:
        kwcoco.Polygon: extracted polygons.

    Example:
        >>> from geowatch.tasks.tracking.utils import mask_to_polygons
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
        >>> from geowatch.tasks.tracking.utils import mask_to_polygons
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
    import kwimage
    import numpy as np
    import shapely.geometry
    from scipy.ndimage import label as ndm_label
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
        try:
            bounds = shapely.geometry.shape(bounds)
        except ValueError:
            ...
        bounds_poly = kwimage.Polygon.coerce(bounds)
        bounds_mask = bounds_poly.to_mask(probs.shape, pixels_are=pixels_are)
        bounds_mask_ = bounds_mask.numpy().data.astype(np.uint8)
        binary_mask *= bounds_mask_

    final_mask = kwimage.Mask(binary_mask, 'c_mask')
    polygons = final_mask.to_multi_polygon(pixels_are=pixels_are)

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

    if 0:
        # Hack this off
        if not set(key).isdisjoint(set(bg_key)):
            raise ValueError('cannot have a key in foreground and background')

    return key, bg_key
