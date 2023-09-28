"""
Main tracker logic

SeeAlso:
    * ../../cli/kwcoco_to_geojson.py
"""
import ubelt as ub
import itertools
import math
from typing import Optional
from typing import Tuple
from typing import Literal
import scriptconfig as scfg

from watch.heuristics import SITE_SUMMARY_CNAME, CNAMES_DCT
from watch.tasks.tracking.utils import NoOpTrackFunction  # NOQA
from watch.tasks.tracking.utils import (
    NewTrackFunction,
    mask_to_polygons,
    _validate_keys,
    score_track_polys,
    trackid_is_default,
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
    """
    Computes the generalized mean over axis=0.

    Args:
        heatmaps (List[ndarray]) pixel aligned heatmaps
        norm_ord (int | float): the exponent of the generalized mean.

    Returns:
        ndarray : the axis=0 is marginalized over.

    Notes:
        like np.linalg.norm but with special nan handling and a division factor

    References:
        https://en.wikipedia.org/wiki/Generalized_mean
        https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.pmean.html

    Example:
        >>> from watch.tasks.tracking.from_heatmap import *  # NOQA
        >>> from watch.tasks.tracking.from_heatmap import _norm
        >>> import kwimage
        >>> import numpy as np
        >>> num_frames = 16
        >>> num_sequences = 6
        >>> # Setup 5 sequences to norm
        >>> heatmaps = [np.empty(num_sequences) for _ in range(num_frames)]
        >>> heatmaps = np.array(heatmaps)
        >>> # Sequence 0 is all nan
        >>> heatmaps[:, 0] = np.nan
        >>> # Sequence 1 is random
        >>> heatmaps[:, 1] = np.random.rand(num_frames)
        >>> # Sequence 2 is Sequence1, but half of the data is nan
        >>> heatmaps[0:, 2] = heatmaps[:, 1]
        >>> heatmaps[0:num_frames // 2, 2] = np.nan
        >>> # Sequence 3 is all zero except for an impulse
        >>> heatmaps[0:, 3] = 0
        >>> heatmaps[num_frames // 2, 3] = 1
        >>> # Sequence 4 is a gaussian response
        >>> heatmaps[0:, 4] = kwimage.gaussian_patch(shape=(1, num_frames))[0]
        >>> # Sequence 5 is a a gaussian response 1 / 4 nans
        >>> heatmaps[0:, 5] = kwimage.gaussian_patch(shape=(1, num_frames))[0]
        >>> heatmaps[0:num_frames // 4, 5] = np.nan
        >>> norm_ord = 1
        >>> x = _norm(heatmaps, norm_ord)
        >>> y = np.linalg.norm(heatmaps, ord=norm_ord, axis=0)
        >>> print('heatmaps = {}'.format(ub.urepr(heatmaps, nl=1, precision=2)))
        >>> print(x)
        >>> print(y)
        >>> # xdoctest: +REQUIRES(--show)
        >>> # Visualize how this works for random signals
        >>> import kwplot
        >>> sns = kwplot.sns
        >>> kwplot.plt.ion()
        >>> # kwplot.close_figures()
        >>> # Add in the original signals
        >>> rows = []
        >>> for c in range(num_sequences):
        >>>     for x in range(num_frames):
        >>>         rows.append(
        >>>             {'x': x, 'col': c, 'ord': 'raw-signal', 'value': heatmaps[x, c]})
        >>> #
        >>> import pandas as pd
        >>> import scipy.stats
        >>> for norm_ord in [1, 2, 3, float('inf')]:
        >>>     v1 = _norm(heatmaps, norm_ord)
        >>>     v2 = scipy.stats.pmean(heatmaps, p=norm_ord, axis=0, nan_policy='omit')
        >>>     print(f'norm_ord={norm_ord}')
        >>>     print(f'v1={v1}')
        >>>     print(f'v2={v2}')
        >>>     for c in range(num_sequences):
        >>>         for x in range(num_frames):
        >>>             rows.append({'x': x, 'col': c, 'ord': norm_ord, 'value': v1[c]})
        >>>     ...
        >>> df = pd.DataFrame(rows)
        >>> pnum_ = kwplot.PlotNums(nSubplots=num_sequences)
        >>> for c in range(num_sequences):
        >>>     kwplot.figure(fnum=1, pnum=pnum_())
        >>>     subdata = df[df['col'] == c]
        >>>     sns.lineplot(data=subdata, x='x', y='value', hue='ord')
    """
    import numpy as np
    heatmaps = np.array(heatmaps)
    if norm_ord == 0:
        import scipy.stats
        probs = scipy.stats.pmean(heatmaps, p=norm_ord, axis=0, nan_policy='omit')
        probs = np.nan_to_num(probs)
    elif norm_ord == np.inf:
        probs = np.nanmax(heatmaps, axis=0)
    else:
        # The np.linalg.norm part
        probs = np.power(np.nansum(np.power(heatmaps, norm_ord), axis=0),
                         1. / norm_ord)
        if norm_ord > 0:
            n_nonzero = np.count_nonzero(~np.isnan(heatmaps), axis=0)
            # Force the denominator to be positive.
            n_nonzero[n_nonzero == 0] = 1
            probs /= np.power(n_nonzero, 1. / norm_ord)
    return probs


# give all these the same signature so they can be swapped out


def binary(heatmaps, norm_ord, morph_kernel, thresh, viz_dpath=None):
    import kwimage
    probs = _norm(heatmaps, norm_ord)

    hard_probs = kwimage.morphology(probs > thresh, 'dilate', morph_kernel)

    return hard_probs.astype('float')


def rescaled_binary(heatmaps, norm_ord, morph_kernel, thresh, upper_quantile=0.999, viz_dpath=None):
    import kwimage
    import kwarray
    import numpy as np
    probs = _norm(heatmaps, norm_ord)
    probs = kwarray.normalize(probs, min_val=0, max_val=np.quantile(probs, upper_quantile))

    hard_probs = kwimage.morphology(probs > thresh, 'dilate', morph_kernel)

    return hard_probs.astype('float')


def probs(heatmaps, norm_ord, morph_kernel, thresh, viz_dpath=None):
    import kwimage
    probs = _norm(heatmaps, norm_ord)

    hard_probs = kwimage.morphology(probs > thresh, 'dilate', morph_kernel)
    modulated_probs = probs * hard_probs

    if viz_dpath is not None:
        kwimage.imwrite(viz_dpath / '0.png', kwimage.ensure_uint255(probs))
        kwimage.imwrite(viz_dpath / '1.png', kwimage.ensure_uint255(hard_probs))
        kwimage.imwrite(viz_dpath / '2.png', kwimage.ensure_uint255(modulated_probs))

    return modulated_probs


def rescaled_probs(heatmaps, norm_ord, morph_kernel, thresh, upper_quantile=0.999, viz_dpath=None):
    import kwimage
    import kwarray
    import numpy as np
    probs = _norm(heatmaps, norm_ord)
    probs = kwarray.normalize(probs, min_val=0, max_val=np.quantile(probs, upper_quantile))

    hard_probs = kwimage.morphology(probs > thresh, 'dilate', morph_kernel)
    modulated_probs = probs * hard_probs

    return modulated_probs


def mean_normalized(heatmaps, norm_ord=1, morph_kernel=1, thresh=None, viz_dpath=None):
    """
    Normalize average_heatmap by applying a scaling based on max(heatmaps) and
    max(average_heatmap)
    """
    import numpy as np
    import kwimage
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


def frequency_weighted_mean(heatmaps, thresh, norm_ord=0, morph_kernel=3, viz_dpath=None):
    """
    Convert a list of heatmaps to an aggregated score, averaging is computed
    based on samples for every pixel
    """
    import kwimage
    import numpy as np
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
    'rescaled_probs': rescaled_probs,
    'probs': probs,
    'rescaled_binary': rescaled_binary,
    'binary': binary,
}

#
# --- track/polygon filters ---
#


class TimePolygonFilter:
    """
    Cuts off start and end of each track based on min response.
    """

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
    """
    Filters each track based on the average response of all tracks.
    """

    def __init__(self, gdf, threshold):

        self.threshold = threshold

        gids = gdf['gid'].unique()
        mean_response = gdf[('fg', -1)].mean()

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
                                                                 -1)].mean()
                return this_response / self.mean_response > threshold

            return gdf.groupby('track_idx', group_keys=False).filter(_filter)
        else:
            cond = (gdf[('fg', -1)] / self.mean_response > threshold)
            return gdf[cond]


#
# --- main logic ---
#


@profile
def _add_tracks_to_dset(sub_dset, tracks, thresh, key, bg_key=None):
    """
    This takes the GeoDataFrame with computed or modified tracks and adds them
    to ``sub_dset``.

    We are assuming the polygon geometry in "tracks" is in video space.
    """
    import kwcoco
    import kwimage
    from watch.utils import kwcoco_extensions
    key, bg_key = _validate_keys(key, bg_key)

    if tracks.empty:
        print('no tracks to add!')
        return sub_dset

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
        if 1:
            # HACK for eval16, need to be nicer about what we do here
            cand_keys = ub.oset(cand_keys) - {'ac_salient'}
        if len(cand_keys) > 1:
            # TODO ensure bg classes are scored if there are >1 of them
            cat_name = ub.argmax(ub.udict.subdict(scores_dct, cand_keys))
        else:
            cat_name = cand_keys[0]
        cid = sub_dset.ensure_category(cat_name)

        assert space in {'image', 'video'}
        if space == 'video':
            # Transform the video polygon into image space
            img_from_vid = _warp_img_from_vid(gid)
            poly = kwimage.MultiPolygon.coerce(poly).warp(img_from_vid)

        bbox = list(poly.box().boxes.to_coco())[0]
        segmentation = poly.to_coco(style='new')

        # Add the polygon as an annotation on the image
        new_ann = dict(image_id=gid,
                       category_id=cid,
                       bbox=bbox,
                       segmentation=segmentation,
                       score=this_score,
                       scores=scores_dct,
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
        tracks.groupby('track_idx', axis=0)
    except ValueError:
        import warnings
        warnings.warn('warning: no tracks to add the the kwcoco dataset')
    else:
        score_chan = kwcoco.ChannelSpec('|'.join(key))
        for tid, grp in tracks.groupby('track_idx', axis=0):
            this_score = grp[(score_chan.spec, -1)]
            scores_dct = {k: grp[(k, -1)] for k in score_chan.unique()}
            scores_dct = [dict(zip(scores_dct, t))
                          for t in zip(*scores_dct.values())]
            _obs_iter = zip(grp['gid'], grp['poly'], this_score, scores_dct)
            _add(_obs_iter, tid)

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
def site_validation(sub_dset, thresh=0.25, span_steps=15):
    """
    Example:
        >>> import watch
        >>> from watch.tasks.tracking.from_heatmap import *  # NOQA
        >>> coco_dset = watch.coerce_kwcoco(
        >>>     'watch-msi', heatmap=True, geodata=True, dates=True)
        >>> vid_id = coco_dset.videos()[0]
        >>> sub_dset = coco_dset.subset(list(coco_dset.images(video_id=vid_id)))
        >>> import numpy as np
        >>> for ann in sub_dset.anns.values():
        >>>     ann['score'] = float(np.random.rand())
        >>> sub_dset.remove_annotations(sub_dset.index.trackid_to_aids[None])
        >>> sub_dset = site_validation(sub_dset)
    """

    # Turn annotations into table we can query
    # annots = pd.DataFrame([
    #     (ub.udict(ann) & {'score', 'track_id', 'track_idx'})
    #     # {
    #     #     'score': ann['score'],
    #     #     'track_id': ann.get('track_id', None),
    #     #     'track_idx': ann.get('track_idx', None),
    #     # }
    #     for ann in sub_dset.dataset["annotations"]
    # ])
    import pandas as pd
    imgs = pd.DataFrame(sub_dset.dataset['images'])
    if 'timestamp' not in imgs.columns:
        imgs['timestamp'] = imgs['id']

    annots = pd.DataFrame(sub_dset.dataset['annotations'])

    if annots.shape[0] == 0:
        print('Nothing to filter')
        return sub_dset

    annots = annots[[
        'id', 'image_id', 'track_id', 'score'
    ]].join(
        imgs[['timestamp']],
        on='image_id',
    )

    track_ids_to_drop = []
    ann_ids_to_drop = []

    for track_id, track_group in annots.groupby('track_id', axis=0):

        # Scores are inherently noisy. We smooth them out with a
        # `span_steps`-wide weighted moving average. The maximum
        # value of this decides whether to keep the track.
        # TODO: do something more elegant here?
        score = track_group['score'].ewm(span=span_steps).mean().max()
        if score < thresh:
            track_ids_to_drop.append(track_id)
            ann_ids_to_drop.extend(track_group['id'].tolist())

    print(f"Dropping {len(ann_ids_to_drop)} annotations from {len(track_ids_to_drop)} tracks.")
    if len(ann_ids_to_drop) > 0:
        sub_dset.remove_annotations(ann_ids_to_drop)

    return sub_dset


@profile
def time_aggregated_polys(sub_dset, **kwargs):
    """
    Track function.

    Aggregate heatmaps across time, threshold them to get polygons,
    and add one track per polygon.

    Args:
        sub_dset (kwcoco.CocoDataset): a kwcoco dataset with exactly 1 video

        **kwargs: see TimeAggregatedPolysConfig

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
        >>>     geodata=True, heatmap=True, dates=True)
        >>> thresh = 0.01
        >>> min_area_square_meters = None
        >>> orig_track = time_aggregated_polys(
        >>>                 sub_dset, thresh=thresh, min_area_square_meters=min_area_square_meters, time_thresh=None)
        >>> # Test robustness to frames that are missing heatmaps
        >>> skip_gids = [1,3]
        >>> for gid in skip_gids:
        >>>      sub_dset.imgs[gid]['auxiliary'].pop()
        >>> inter_track = time_aggregated_polys(
        >>>                 sub_dset, thresh=thresh, min_area_square_meters=min_area_square_meters, time_thresh=None)
        >>> assert inter_track.iloc[0][('fg', -1)] == 0
        >>> assert inter_track.iloc[1][('fg', -1)] > 0
    """
    #
    # --- input validation ---
    #
    import kwimage
    import geopandas as gpd
    import numpy as np
    config = TimeAggregatedPolysConfig(**kwargs)
    config.key, config.bg_key = _validate_keys(config.key, config.bg_key)

    _all_keys = set(config.key + config.bg_key)
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
    tracking_gsd = None
    if len(video_gids) and (config.resolution is not None):
        # Determine resolution information for videospace (what we will return
        # in) and tracking space (what we will build heatmaps in)
        first_gid = video_gids[0]
        first_coco_img = sub_dset.coco_image(first_gid)
        # (w, h)
        vidspace_resolution = first_coco_img.resolution(space='video')['mag']
        vidspace_resolution = np.array(vidspace_resolution)

        # (w, h)
        scale_trk_from_vid = first_coco_img._scalefactor_for_resolution(
            space='video', resolution=config.resolution)
        scale_trk_from_vid = np.array(scale_trk_from_vid)

        # Determinethe pixel size of tracking space
        tracking_resolution = vidspace_resolution / scale_trk_from_vid
        if not np.isclose(*tracking_resolution):
            print(f'warning: nonsquare pxl size of {tracking_resolution}')
        tracking_gsd = np.mean(tracking_resolution)

        # Get the transform from tracking space back to video space
        scale_vid_from_trk = 1 / scale_trk_from_vid
    else:
        scale_vid_from_trk = (1, 1)

    if tracking_gsd is None:
        if len(video_gids):
            # Use whatever is in the kwcoco file as the default.
            first_gid = video_gids[0]
            first_coco_img = sub_dset.coco_image(first_gid)
            # (w, h)
            vidspace_resolution = first_coco_img.resolution(space='video')['mag']
            default_gsd = np.mean(vidspace_resolution)
        else:
            default_gsd = 30
            print(f'warning: video {video["name"]} in dset {sub_dset.tag} '
                  f'has no listed resolution; assuming {default_gsd}')
        tracking_gsd = default_gsd

    if not any(has_requested_chans_list):
        raise KeyError(f'no imgs in dset {sub_dset.tag} '
                       f'have keys {config.key} or {config.bg_key}.')
    if not all(has_requested_chans_list):
        n_total = len(has_requested_chans_list)
        n_have = sum(has_requested_chans_list)
        n_missing = (n_total - n_have)
        print(f'warning: {n_missing} / {n_total} imgs in dset {sub_dset.tag} '
              f'with video {video_name} have no keys {config.key} or {config.bg_key}. '
              'Interpolating...')

    #
    # --- main logic ---
    #

    # polys are in "tracking-space", i.e. video-space up to a scale factor.
    gid_poly_config = _GidPolyConfig(**ub.udict(config).subdict(_GidPolyConfig.__default__.keys()))
    gids_polys = _gids_polys(sub_dset, **gid_poly_config)

    orig_gid_polys = list(gids_polys)  # 26% of runtime
    gids_polys = orig_gid_polys

    print('time aggregation: number of polygons: ', len(gids_polys))

    # size and response filters should operate on each vidpoly separately.
    if config.max_area_square_meters:
        max_area_sqpx = config.max_area_square_meters / (tracking_gsd ** 2)
        n_orig = len(gids_polys)
        if config.max_area_behavior == 'drop':
            gids_polys = [(t, p) for t, p in gids_polys
                          if p.to_shapely().area < max_area_sqpx]
            print('filter large: remaining polygons: '
                  f'{len(gids_polys)} / {n_orig}')
        elif config.max_area_behavior == 'grid':
            # edits tracks instead of removing them
            raise NotImplementedError

    if config.min_area_square_meters:
        min_area_sqpx = config.min_area_square_meters / (tracking_gsd ** 2)
        n_orig = len(gids_polys)
        gids_polys = [(t, p) for t, p in gids_polys
                      if p.to_shapely().area > min_area_sqpx]
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

    if config.polygon_simplify_tolerance is not None:
        _TRACKS['poly'] = _TRACKS['poly'].simplify(tolerance=config.polygon_simplify_tolerance)

    # _TRACKS['track_idx'] = range(len(_TRACKS))
    _TRACKS = _TRACKS.reset_index().rename(columns={'index': 'track_idx'})
    _TRACKS = _TRACKS.explode('gid')

    # ensure index is sorted in video order
    sorted_gids = sub_dset.images(vidid=video['id']).gids
    _TRACKS = gpd_sort_by_gid(_TRACKS, sorted_gids)

    # awk, find better way of bookkeeping and indexing into scores needed
    thrs = {-1}
    if config.response_thresh:
        thrs.add(-1)
    if config.time_thresh:
        thrs.add(config.time_thresh * config.thresh)
    #####
    ## Jon C: I'm not sure about this. Going from a set to a list, and then having
    ## the resulting function depend on the order of the list makes me nerevous.
    #####
    thrs = list(thrs)

    ks = {'fg': config.key, 'bg': config.bg_key}

    # TODO dask gives different results on polys that overlap nodata area, need
    # to debug this. (6% of polygons in KR_R001, so not a huge difference)
    # USE_DASK = True
    USE_DASK = False
    _TRACKS = gpd_compute_scores(_TRACKS, sub_dset, thrs, ks,
                                 USE_DASK=USE_DASK,
                                 resolution=config.resolution)

    if _TRACKS.empty:
        return _TRACKS

    # dask could unsort
    _TRACKS = gpd_sort_by_gid(_TRACKS.reset_index(), sorted_gids)

    # response_thresh = 0.9
    if config.response_thresh:

        n_orig = gpd_len(_TRACKS)
        rsp_filter = ResponsePolygonFilter(_TRACKS, config.key, config.response_thresh)
        _TRACKS = rsp_filter(_TRACKS)
        print('filter based on per-polygon response: remaining tracks '
              f'{gpd_len(_TRACKS)} / {n_orig}')

    # TimePolygonFilter edits tracks instead of removing them
    if config.time_thresh:  # as a fraction of thresh
        time_filter = TimePolygonFilter(config.time_thresh * config.thresh)
        n_orig = gpd_len(_TRACKS)
        _TRACKS = time_filter(_TRACKS)  # 7% of runtime? could be next line
        print('filter based on time overlap: remaining tracks '
              f'{gpd_len(_TRACKS)} / {n_orig}')

    # The tracker assumes the polygons will be output in video space.
    if scale_vid_from_trk is not None and len(_TRACKS):
        # If a tracking resolution was specified undo the extra scale factor
        _TRACKS['poly'] = _TRACKS['poly'].scale(*scale_vid_from_trk, origin=(0, 0))

    # TODO: do we need to convert to MultiPolygon here? Or can that be handled
    # by consumers of this method?
    _TRACKS['poly'] = _TRACKS['poly'].map(kwimage.MultiPolygon.from_shapely)
    return _TRACKS


#
# --- time_aggregated_polys utilities ---
#


def _merge_polys(p1, p2, poly_merge_method=None):
    """
    Given two lists of polygons, p1 and p2, merge these according to:
      - add all unique polygons in the merged list
      - for overlapping polygons, add the union of both polygons

    Ignore:
        from watch.tasks.tracking.from_heatmap import * # NOQA
        from watch.tasks.tracking.from_heatmap import _merge_polys  # NOQA

        p1 = [kwimage.Polygon.random().scale(0.2).to_shapely() for _ in range(1)]
        p2 = [kwimage.Polygon.random().to_shapely() for _ in range(1)]
        unary_union(p1 + p2)
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
                if combo.geom_type != 'Polygon':
                    raise Exception('!')
    """
    import numpy as np
    merged_polys = []
    if poly_merge_method is None:
        poly_merge_method = 'v1'

    if poly_merge_method == 'v2':
        # Just combine anything that touches in both frames together
        from watch.utils import util_gis
        import geopandas as gpd
        geom_df = gpd.GeoDataFrame(geometry=p1 + p2)
        isect_idxs = util_gis.geopandas_pairwise_overlaps(geom_df, geom_df)
        level_sets = {frozenset(v.tolist()) for v in isect_idxs.values()}
        level_sets = list(map(sorted, level_sets))

        merged_polys = []
        for idxs in level_sets:
            if len(idxs) == 1:
                combo = geom_df['geometry'].iloc[idxs[0]]
                merged_polys.append(combo)
            else:
                combo = geom_df['geometry'].iloc[idxs].unary_union
                if combo.geom_type == 'Polygon':
                    merged_polys.append(combo)
                elif combo.geom_type == 'MultiPolygon':
                    # Can this ever happen? It seems to have occurred in a test
                    # run. Bowties can cause this.
                    # import warnings
                    # warnings.warn('Found two intersecting polygons where the
                    # union was a multipolygon')
                    # combo = combo.buffer(0)
                    merged_polys.append(combo.convex_hull)
                    # merged_polys.extend(list(combo.geoms))
                else:
                    raise AssertionError(f'Unexpected type {combo.geom_type}')
    elif poly_merge_method == 'v1':
        from shapely.ops import unary_union
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
                    if combo.geom_type == 'Polygon':
                        merged_polys.append(combo)
                    elif combo.geom_type == 'MultiPolygon':
                        # Can this ever happen? It seems to have occurred in a test
                        # run. Bowties can cause this.
                        # import warnings
                        # warnings.warn('Found two intersecting polygons where the
                        # union was a multipolygon')
                        merged_polys.extend(list(combo.geoms))
                    else:
                        raise AssertionError(
                            f'Unexpected type {combo.geom_type} from {_p1} and {_p2}')

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
    else:
        raise ValueError(poly_merge_method)

    return merged_polys


def _process(track, _heatmaps, image_dates, gids, config):
    from shapely.ops import unary_union
    import kwimage
    import numpy as np
    # TODO when bounds are time-varying, this lets individual frames
    # go outside them; only enforces the union. Problem?
    # currently bounds come from site summaries, which are not
    # time-varying.
    if track is None:
        track_bounds = None
        _heatmaps_in_track = _heatmaps
        heatmap_dates = image_dates
    else:
        track_bounds = track['poly'].unary_union
        track_gids = track['gid']
        flags = np.in1d(gids, track_gids)
        _heatmaps_in_track = np.compress(flags, _heatmaps, axis=0)
        heatmap_dates = list(ub.compress(image_dates, flags))

    # this is another hot spot, heatmaps_to_polys -> mask_to_polygons ->
    # rasterize. Figure out how to vectorize over bounds.
    track_polys = heatmaps_to_polys(
        _heatmaps_in_track, track_bounds, heatmap_dates=heatmap_dates,
        config=config,
    )
    if track is None:
        # BUG: The polygons retunred from heatmap-to-polys might not be
        # corresponding to the gids in this case.
        yield from zip(itertools.repeat(gids), track_polys)
        # for poly in polys:  # convert to shapely to check this
        # if poly.is_valid and not poly.is_empty:
        # yield (gids, poly)
    else:
        poly = unary_union([p.to_shapely() for p in track_polys])
        if poly.is_valid and not poly.is_empty:
            yield (track['gid'], kwimage.MultiPolygon.from_shapely(poly))


viz_n_window = 0  # FIXME, no dynamic globals


@profile
def heatmaps_to_polys(heatmaps, track_bounds, heatmap_dates=None, config=None):
    """
    Use parameters: agg_fn, thresh, morph_kernel, thresh_hysteresis, norm_ord
    """
    global viz_n_window
    import numpy as np

    # TODO: rename moving window size to "outer_window_size"

    def convert_to_shapely(polys):
        return [p.to_shapely() for p in polys]

    def convert_to_kwimage_poly(shapely_polys):
        import kwimage
        return [kwimage.Polygon.from_shapely(p) for p in shapely_polys]

    _agg_fn = AGG_FN_REGISTRY[config.agg_fn]

    if isinstance(config.inner_window_size, str):
        # TODO: generalize if needed
        assert heatmap_dates is not None

        if config.inner_agg_fn == 'mean':
            inner_ord = 1
        elif config.inner_agg_fn == 'max':
            inner_ord = float('inf')
        else:
            raise NotImplementedError(config.inner_agg_fn)

        # Do inner aggregation before outer aggregation
        from kwutil import util_time
        import kwarray
        delta = util_time.coerce_timedelta(config.inner_window_size).total_seconds()
        image_unixtimes = np.array([d.timestamp() for d in heatmap_dates])
        bucket_ids = (image_unixtimes // delta).astype(int)
        unique_ids, groupxs = kwarray.group_indices(bucket_ids)
        new_heatmaps = []
        for idxs in groupxs:
            inner = _norm(heatmaps[idxs], norm_ord=inner_ord)
            new_heatmaps.append(inner)
        new_heatmaps = np.array(new_heatmaps)
        heatmaps = new_heatmaps
    else:
        if config.inner_window_size is not None:
            raise NotImplementedError(
                'only temporal deltas for inner agg window for now')

    # calculate number of moving-window steps, based on window_size and number
    # of heatmaps
    if config.moving_window_size is not None:
        total_n = len(heatmaps)
        final_size = int(total_n // np.ceil((total_n / config.moving_window_size)))
        n_steps = total_n // final_size
    else:
        final_size = len(heatmaps)
        n_steps = 1

    # initialize heatmaps and initial polygons on the first set of heatmaps
    h_init = heatmaps[:final_size]

    prog = ub.ProgIter(total=n_steps, desc='process-step')
    with prog:
        step = 0
        polys_final = _process_1_step(h_init, _agg_fn, track_bounds, step, config)
        prog.step()

        if n_steps > 1:
            polys_final = convert_to_shapely(polys_final)

            for step in range(1, n_steps):
                prog.step()
                h1 = heatmaps[step * final_size:(step + 1) * final_size]
                p1 = _process_1_step(h1, _agg_fn, track_bounds, step, config)
                p1 = convert_to_shapely(p1)
                polys_final = _merge_polys(polys_final, p1,
                                           poly_merge_method=config.poly_merge_method)

            polys_final = convert_to_kwimage_poly(polys_final)
    return polys_final


def _process_1_step(heatmaps, _agg_fn, track_bounds, step, config):
    # FIXME: no dynamic globals.
    if config.viz_out_dir is not None:
        viz_dpath = (config.viz_out_dir / f'heatmaps_{step}').ensuredir()
    else:
        viz_dpath = None

    aggregated = _agg_fn(heatmaps,
                         thresh=config.thresh,
                         morph_kernel=config.morph_kernel,
                         norm_ord=config.norm_ord,
                         viz_dpath=viz_dpath)
    polygons = list(
        mask_to_polygons(aggregated,
                         thresh=config.thresh,
                         bounds=track_bounds,
                         thresh_hysteresis=config.thresh_hysteresis))
    return polygons


def _gids_polys(sub_dset, **kwargs):
    """
    Example:
        >>> from watch.tasks.tracking.from_heatmap import *  # NOQA
        >>> from watch.tasks.tracking.from_heatmap import _gids_polys
        >>> import watch
        >>> coco_dset = watch.coerce_kwcoco(data='watch-msi', dates=True, geodata=True, heatmap=True)
        >>> sub_dset = coco_dset.subset(coco_dset.videos().images[0])
        >>> key = 'salient'
        >>> agg_fn = 'probs'
        >>> thresh = 0.01
        >>> morph_kernel = 3
        >>> thresh_hysteresis = 0
        >>> norm_ord = 1
        >>> resolution = None
        >>> moving_window_size = None
        >>> inner_window_size = '1year'
        >>> results = list(_gids_polys(
        >>>     sub_dset,
        >>>     key=key,
        >>>     agg_fn=agg_fn,
        >>>     thresh=thresh,
        >>>     morph_kernel=morph_kernel,
        >>>     thresh_hysteresis=thresh_hysteresis,
        >>>     norm_ord=norm_ord,
        >>>     resolution=resolution,
        >>>     moving_window_size=moving_window_size,
        >>>     use_boundaries=None,
        >>> ))

    Returns:
        Iterable[int | kwimage.Polygon | kwimage.MultiPolygon]
    """
    from kwutil import util_time
    import numpy as np
    config = _GidPolyConfig(**kwargs)

    if config.use_boundaries:  # for SC
        raw_boundary_tracks = score_track_polys(sub_dset, [SITE_SUMMARY_CNAME])
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

    images = sub_dset.images(gids)
    image_dates = [util_time.coerce_datetime(d)
                   for d in images.lookup('date_captured')]
    # image_years = [d.year for d in image_dates]

    key = '|'.join(config.key)
    coco_images = sub_dset.images(gids).coco_images

    load_workers = 0  # TODO: configure
    load_jobs = ub.JobPool(mode='process', max_workers=load_workers)

    with load_jobs:
        for coco_img in ub.ProgIter(coco_images, desc='submit heatmap jobs'):
            delayed = coco_img.imdelay(channels=key, space='video', resolution=config.resolution)
            load_jobs.submit(delayed.finalize)

        _heatmaps = []
        for job in ub.ProgIter(load_jobs.jobs, desc='collect heatmap jobs'):
            _heatmap = job.result()
            _heatmaps.append(_heatmap)
    _heatmaps = np.stack(_heatmaps, axis=0)
    print(f'(presum) _heatmaps.shape={_heatmaps.shape}')
    _heatmaps = _heatmaps.sum(axis=-1)  # sum over channels
    print(f'_heatmaps.shape={_heatmaps.shape}')
    missing_ix = np.invert([key in i.channels for i in coco_images])
    # TODO this was actually broken in orig, so turning it off here for now
    interpolate = 0
    if interpolate:
        diffed = np.concatenate((np.diff(missing_ix), [False]))
        src = ~missing_ix & diffed
        _heatmaps[missing_ix] = _heatmaps[src]
        if missing_ix[0]:
            _heatmaps[:np.searchsorted(diffed, True)] = 0
        assert np.isnan(_heatmaps).all(axis=(1, 2)).sum() == 0
    else:
        _heatmaps[missing_ix] = 0

    # no benefit so far
    proc_jobs = ub.JobPool('process', max_workers=0)
    with proc_jobs:

        for _, track in ub.ProgIter(boundary_tracks, desc='submit proc jobs'):
            proc_jobs.submit(_process, track, _heatmaps, image_dates, gids, config)

        result_gen = itertools.chain.from_iterable(
            j.result() for j in ub.ProgIter(proc_jobs.jobs, desc='collect proc jobs'))
        result_gen = list(result_gen)
        return result_gen

#
# --- wrappers ---
#
# Note:
#     The following are valid choices of `track_fn` in
#     ../../cli/kwcoco_to_geojson.py and will be called by ./normalize.py


def _resolve_deprecated_args(self):
    """
    Ignore:
        # Logic to check the conversion constant is correct
        import pint
        ureg = pint.UnitRegistry()
        sqm = ureg.meters ** 2
        sqkm = ureg.kilometers ** 2
        sqm_to_skqm_scale_factor = float(((1 * sqkm) / (1 * sqm)).to_base_units())
        print(f'sqm_to_skqm_scale_factor={sqm_to_skqm_scale_factor}')
        print((0.072 * sqkm).to(sqm))
        print(0.072 * sqm_to_skqm_scale_factor)
        0.072 * sqkm
    """
    sqm_to_skqm_scale_factor = 1_000_000

    if self.min_area_sqkm is not None:
        ub.schedule_deprecation(
            'watch', 'min_area_sqkm', 'tracking param',
            migration='use min_area_square_meters instead',
            deprecate='now', error='now')

        if self.min_area_square_meters is not None:
            raise ValueError('Cannot specify min_area_sqkm and min_area_square_meters')

        self.min_area_square_meters = self.min_area_sqkm * sqm_to_skqm_scale_factor
        self.min_area_sqkm = None

    if self.max_area_sqkm is not None:
        ub.schedule_deprecation(
            'watch', 'max_area_sqkm', 'tracking param',
            migration='use max_area_square_meters instead',
            deprecate='now', error='now')

        if self.max_area_square_meters is not None:
            raise ValueError('Cannot specify min_area_sqkm and max_area_square_meters')

        self.max_area_square_meters = self.max_area_sqkm * sqm_to_skqm_scale_factor
        self.max_area_sqkm = None


def _resolve_arg_values(self):
    if isinstance(self.norm_ord, str) and self.norm_ord.lower() == 'inf':
        self.norm_ord = float('inf')


class _GidPolyConfig(scfg.DataConfig):
    # This is the base config that all from-heatmap trackers have in common
    # which has to do with how heatmaps are loaded, normalized, and aggregated.

    key = scfg.Value('salient', help=ub.paragraph(
        '''
        One or more channels to use as positive class for binary heatmap
        polygon extraction and scoring.
        '''))

    agg_fn = scfg.Value('probs', help=ub.paragraph(
        '''
        The aggregation method to preprocess heatmaps.
        See ``AGG_FN_REGISTRY`` for available options.
        '''))

    thresh = scfg.Value(0.0, help=ub.paragraph(
        '''
        The threshold for polygon extraction from heatmaps.
        E.g. this threshold binarizes the heatmaps.
        '''))

    morph_kernel = scfg.Value(3, help=ub.paragraph(
        '''
        Morphology kernel for preprocessing the heatmaps with dilation.
        '''))

    thresh_hysteresis = scfg.Value(None, help=ub.paragraph(
        '''
        I dont remember. Help wanted to document this
        '''))

    # TODO: rename to outer_agg_fn
    norm_ord = scfg.Value(1, help=ub.paragraph(
        '''
        The generalized mean order used to average heatmaps over the
        "moving_window_size". A value of 1 is the normal mean. A value of inf
        is the max function. Note: this is effectively an outer_agg_fn.
        '''))

    # TODO: rename to outer_window_size
    moving_window_size = scfg.Value(None, help=ub.paragraph(
        '''
        The outer moving window size. The number of consecutive inner window
        results to aggregate together. If None, then all inner window results
        are combined into a single final heatmap.
        '''), alias=['outer_window_size'])

    inner_window_size = scfg.Value(None, help=ub.paragraph(
        '''
        The inner moving window time range (e.g. 1y).  The bucket size (in
        time) of time-consecutive heatmaps to combine using an inner moving
        window. If None, then no inner windowing is used.
        '''))

    inner_agg_fn = scfg.Value('mean', help=ub.paragraph(
        '''
        The method used for aggregating heatmaps scores over the inner window.
        Note, this roughtly corresponds to norm_ord, which is like the
        outer_agg_fn.
        '''))

    resolution = scfg.Value(None, help=ub.paragraph(
        '''
        The resolution for loading and processing the heatmaps at. E.g. 10GSD.
        '''))

    use_boundaries = scfg.Value(False, help=ub.paragraph(
        '''
        If False, then extracted polygons are used as new site boundaries.  If
        True, then we keep existing annotation boundaries unchanged and only
        used the heatmaps to update scores of the existing boundary polyons.
        '''))

    poly_merge_method = scfg.Value('v1', help=ub.paragraph(
        '''
        Method for handling overlaping polygons across multiple timesteps.
        Currently can be "v1" or "v2". There isn't much difference.
        We should find a better way of handling this.
        '''))

    viz_out_dir = scfg.Value(None, help=ub.paragraph(
        '''
        Directory to output intermediate visualizations
        '''))


class TimeAggregatedPolysConfig(_GidPolyConfig):
    """
    This is an intermediate config that we will use to transition between the
    current dataclass configuration and a new scriptconfig based one.

    python -c "if 1:
        from watch.tasks.tracking.from_heatmap import TimeAggregatedBAS
        TimeAggregatedBAS().argparse().print_help()
    "

    """
    bg_key = scfg.Value(None, help=ub.paragraph(
        '''
        Zero or more channels to use as the negative class for polygon scoring.
        '''))

    time_thresh = scfg.Value(1, help=ub.paragraph(
        '''
        Multiplier on the regular threshold used to determine the temporal
        extent of the polygon over time. All polygons must have an aggregate
        score over ``thresh * time_thresh``.  Typically set this a bit less
        than 1. (e.g. 0.8).
        '''))

    response_thresh = scfg.Value(None, help=ub.paragraph(
        '''
        I dont remember what this does. Help wanted with documenting.
        '''))

    min_area_square_meters = scfg.Value(None, help=ub.paragraph(
        '''
        If specified, any site with an area less than this threshold is
        removed.
        '''))

    max_area_square_meters = scfg.Value(None, help=ub.paragraph(
        '''
        If specified, any site with an area greater than this threshold is
        removed.
        '''))

    max_area_behavior = scfg.Value('drop', help=ub.paragraph(
        '''
        How to handle polygons that are over the max area threshold.
        '''))

    polygon_simplify_tolerance = scfg.Value(None, help=ub.paragraph(
        '''
        The pixel size (at the specified heatmap resolution) to use for polygon
        simplification.
        '''))

    def __post_init__(self):
        super().__post_init__()
        if self.norm_ord in {'inf', None}:
            self.norm_ord = float('inf')
        # self.key, self.bg_key = _validate_keys(self.key, self.bg_key)

        if isinstance(self.inner_window_size, float) and math.isnan(self.inner_window_size):
            self.inner_window_size = None

        if isinstance(self.moving_window_size, float) and math.isnan(self.moving_window_size):
            self.moving_window_size = None


class CommonTrackFn(NewTrackFunction, TimeAggregatedPolysConfig):
    min_area_sqkm: Optional[float] = None  # was 0.072  # 80px@30GSD  # deprecate
    max_area_sqkm: Optional[float] = None  # was 2.25  # deprecate

    def __post_init__(self):
        super().__post_init__()
        _resolve_deprecated_args(self)
        _resolve_arg_values(self)


class TrackFnWithSV(CommonTrackFn):
    site_validation: bool = False
    site_validation_span_steps: int = 120
    site_validation_thresh: float = 0.1


class TimeAggregatedBAS(TrackFnWithSV):
    """
    Wrapper for BAS that looks for change heatmaps.
    """
    thresh: float = 0.2
    key: str = 'salient'
    agg_fn: str = 'probs'

    def create_tracks(self, sub_dset):
        aggkw = ub.udict(self) & TimeAggregatedPolysConfig.__default__.keys()
        tracks = time_aggregated_polys(sub_dset, **aggkw)
        return tracks

    def add_tracks_to_dset(self, sub_dset, tracks):
        sub_dset = _add_tracks_to_dset(sub_dset, tracks, self.thresh, self.key)
        if self.site_validation:
            sub_dset = site_validation(
                sub_dset,
                thresh=self.site_validation_thresh,
                span_steps=self.site_validation_span_steps,
            )
        return sub_dset


class TimeAggregatedSC(TrackFnWithSV):
    """
    Wrapper for Activity Characterization / Site Characterization that looks
    for phase heatmaps.

    Alias: class_heatmaps

    Note:
        This is a valid choice of `track_fn` in ../../cli/kwcoco_to_geojson.py
    """
    thresh: float = 0.01

    # key: Tuple[str] = tuple(CNAMES_DCT['positive']['scored'])

    # HACK TO REMEMBER ALL SCORES
    # TODO: Ensure this does not  break anything and refactor such that the
    # default behavior is to aggreate the score from all available classes when
    # only scoring. When refining polygons, a different approach is needed.
    key: Tuple[str] = tuple(['Site Preparation', 'Active Construction', 'Post Construction', 'No Activity', 'ac_salient'])

    # IS THIS USED?
    bg_key: Tuple[str] = tuple(CNAMES_DCT['negative']['scored'])

    boundaries_as: Literal['bounds', 'polys', 'none'] = 'bounds'
    time_thresh = None

    def create_tracks(self, sub_dset):
        """
        boundaries_as: use for Site Boundary annots in coco_dset
            'bounds': generated polys will lie inside the boundaries
            'polys': generated polys will be the boundaries
            'none': generated polys will ignore the boundaries
        """
        import kwcoco
        import kwimage

        if self.boundaries_as == 'polys':
            tracks = score_track_polys(
                sub_dset,
                cnames=[SITE_SUMMARY_CNAME],
                # these are SC scores, not BAS, so this is not a
                # true reproduction of hybrid.
                score_chan=kwcoco.ChannelSpec('|'.join(self.key)),
                resolution=self.resolution,
            )
            # hack in always-foreground instead
            # tracks[(score_chan, -1)] = 1

            # try to ignore this error
            tracks['poly'] = tracks['poly'].map(
                kwimage.MultiPolygon.from_shapely)

        else:
            aggkw = ub.udict(self) & TimeAggregatedPolysConfig.__default__.keys()
            aggkw['use_boundaries'] = aggkw.get('boundaries_as', 'none') != 'none'
            tracks = time_aggregated_polys(sub_dset, **aggkw)
        return tracks

    def add_tracks_to_dset(self, sub_dset, tracks, **kwargs):
        import kwcoco
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

        thresh = self.thresh
        key = self.key
        bg_key = self.bg_key
        sub_dset = _add_tracks_to_dset(sub_dset, tracks=tracks, thresh=thresh,
                                       key=key, bg_key=bg_key, **kwargs)
        if self.site_validation:
            sub_dset = site_validation(
                sub_dset,
                thresh=self.site_validation_thresh,
                span_steps=self.site_validation_span_steps,
            )

        return sub_dset


class TimeAggregatedSV(CommonTrackFn):
    """
    Wrapper for Site Validation that looks for phase heatmaps.

    Alias:
        site_validation

    Note:
        This is a valid choice of `track_fn` in ../../cli/kwcoco_to_geojson.py
    """
    thresh: float = 0.1
    key: str = 'salient'
    boundaries_as: Literal['bounds', 'polys', 'none'] = 'polys'
    span_steps: int = 120

    def create_tracks(self, sub_dset):
        """
        boundaries_as: use for Site Boundary annots in coco_dset
            'bounds': generated polys will lie inside the boundaries
            'polys': generated polys will be the boundaries
            'none': generated polys will ignore the boundaries
        """
        import kwcoco
        import kwimage
        if self.boundaries_as == 'polys':
            tracks = score_track_polys(
                sub_dset,
                cnames=[SITE_SUMMARY_CNAME],
                # these are SC scores, not BAS, so this is not a
                # true reproduction of hybrid.
                score_chan=kwcoco.ChannelSpec('|'.join((self.key,))),
                resolution=self.resolution,
            )
            # hack in always-foreground instead
            # tracks[(score_chan, None)] = 1

            # try to ignore this error
            tracks['poly'] = tracks['poly'].map(
                kwimage.MultiPolygon.from_shapely)

        else:
            raise NotImplementedError
        return tracks

    def add_tracks_to_dset(self, sub_dset, tracks, **kwargs):
        # if self.boundaries_as != 'polys':
        #     col_map = {}
        #     for c in tracks.columns:
        #         if c[0] == 'fg':
        #             k = kwcoco.ChannelSpec('|'.join(self.key)).spec
        #             col_map[c] = (k, *c[1:])
        #         elif c[0] == 'bg':
        #             k = kwcoco.ChannelSpec('|'.join(self.bg_key)).spec
        #             col_map[c] = (k, *c[1:])
        #     # weird effect here - reassignment casts from GeoDataFrame to
        #     # DataFrame. Related to invalid geometry column?
        #     # tracks = tracks.rename(columns=col_map)
        #     tracks.rename(columns=col_map, inplace=True)

        sub_dset = _add_tracks_to_dset(sub_dset, tracks, self.thresh, self.key,
                                       **kwargs)
        sub_dset = site_validation(
            sub_dset,
            thresh=self.thresh,
            span_steps=self.span_steps,
        )

        return sub_dset
