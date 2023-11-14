"""
Main tracker logic

SeeAlso:
    * ../../cli/kwcoco_to_geojson.py
"""
import ubelt as ub
import itertools
import math
from typing import Tuple
from typing import Literal
import scriptconfig as scfg

from watch.heuristics import SITE_SUMMARY_CNAME, CNAMES_DCT
from watch.tasks.tracking import agg_functions
from watch.tasks.tracking.abstract_classes import NewTrackFunction

from watch.tasks.tracking.utils import (
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


class TimeSplitFilter:
    """
    Splits tracks based on start and end of each subtracks min response.
    """

    def __init__(self, threshold, frame_buffer):
        self.threshold = threshold
        self.frame_buffer = frame_buffer

    def __call__(self, gdf):
        import geopandas as gpd
        import pandas as pd

        def buffer_by(tracks, by):
            new_tracks = []
            for start, end in tracks:
                new_tracks.append((start - by, end + by))
            return new_tracks

        def merge_neighbors(tracks):
            new_tracks = []
            prev = None
            for curr in tracks:
                if prev is None:
                    prev = curr
                    continue

                if curr[0] <= prev[1]:
                    prev = (prev[0], curr[1])
                else:
                    new_tracks.append(prev)
                    prev = curr

            new_tracks.append(prev)
            return new_tracks

        def _edit(scores):
            magic_thresh = 0.5
            track_start = None
            sub_tracks = []
            for idx, score in enumerate(scores):

                if (score > magic_thresh) and (track_start is None):
                    # print(f"track started at {idx}")
                    track_start = idx

                if (score < magic_thresh) and (track_start is not None):
                    # print(f"track ended at {idx-1}")
                    sub_tracks.append((track_start, idx))
                    track_start = None

            if (track_start is not None):
                sub_tracks.append((track_start, len(scores)))

            return sub_tracks

        if len(gdf) > 0:
            subtracks = []
            subtrack_idx = 1
            for track_id, group in gdf.groupby('track_idx'):

                subtrack_startstops = _edit(list(group[('fg', self.threshold)]))
                subtrack_startstops = buffer_by(subtrack_startstops, self.frame_buffer)
                subtrack_startstops = merge_neighbors(subtrack_startstops)

                if len(subtrack_startstops) == 0:
                    return gpd.GeoDataFrame()

                if subtrack_startstops[0][0] < 0:
                    subtrack_startstops[0] = (0, subtrack_startstops[0][1])
                if subtrack_startstops[-1][0] >= len(group):
                    subtrack_startstops[-1] = (subtrack_startstops[-1][0], len(group))

                for sub_id, (start, stop) in enumerate(subtrack_startstops):
                    subtrack = group.iloc[start:stop]
                    subtrack["track_idx"] = subtrack_idx
                    subtrack_idx += 1

                    subtracks.append(subtrack)
            result = gpd.GeoDataFrame(
                pd.concat(subtracks, ignore_index=True)
            )
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

    print('Add tracks to dset')
    print(f'bg_key={bg_key}')
    print(f'key={key}')
    print('tracks:')
    print(tracks)

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
            if len(cand_keys) > 1:
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
    Polygon extraction and tracking function.

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
        >>> sub_dset = watch.coerce_kwcoco(
        >>>     'watch-msi', num_videos=1, num_frames=5, image_size=(480, 640),
        >>>     geodata=True, heatmap=True, dates=True)
        >>> thresh = 0.01
        >>> min_area_square_meters = None
        >>> kwargs = dict(thresh=thresh, min_area_square_meters=min_area_square_meters, time_thresh=None)
        >>> orig_track = time_aggregated_polys(sub_dset, **kwargs)
        >>> # Test robustness to frames that are missing heatmaps
        >>> skip_gids = [1,3]
        >>> for gid in skip_gids:
        >>>      sub_dset.imgs[gid]['auxiliary'].pop()
        >>> inter_track = time_aggregated_polys(sub_dset,  **kwargs)
        >>> assert inter_track.iloc[0][('fg', -1)] == 0
        >>> assert inter_track.iloc[1][('fg', -1)] > 0

    Example:
        >>> # test interpolation
        >>> from watch.tasks.tracking.from_heatmap import time_aggregated_polys
        >>> from watch.demo import demo_kwcoco_with_heatmaps
        >>> import watch
        >>> sub_dset = watch.coerce_kwcoco(
        >>>     'watch-msi', num_videos=1, num_frames=5, image_size=(480, 640),
        >>>     geodata=True, heatmap=True, dates=True)
        >>> thresh = 0.01
        >>> min_area_square_meters = None
        >>> kwargs = dict(thresh=thresh, min_area_square_meters=min_area_square_meters, time_thresh=None)
        >>> orig_track = time_aggregated_polys(sub_dset, **kwargs)
        >>> # Test robustness to frames that are missing heatmaps
        >>> skip_gids = [1,3]
        >>> for gid in skip_gids:
        >>>      sub_dset.imgs[gid]['auxiliary'].pop()
        >>> inter_track = time_aggregated_polys(sub_dset,  **kwargs)
        >>> assert inter_track.iloc[0][('fg', -1)] == 0
        >>> assert inter_track.iloc[1][('fg', -1)] > 0
    """
    #
    # --- input validation ---
    #
    import kwimage
    import geopandas as gpd
    import rich
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

    scale_vid_from_trk, tracking_gsd = _determine_tracking_scale(
        config, sub_dset, video_gids, video)

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
    gid_poly_config = PolygonExtractConfig(**ub.udict(config).subdict(PolygonExtractConfig.__default__.keys()))
    gids_polys = _gids_polys(sub_dset, **gid_poly_config)

    orig_gid_polys = list(gids_polys)  # 26% of runtime
    gids_polys = orig_gid_polys

    if len(gids_polys):
        rich.print('[green] time aggregation: number of polygons: ', len(gids_polys))
    else:
        rich.print('[red] time aggregation: number of polygons: ', len(gids_polys))

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
    if config.time_split_thresh:
        thrs.add(config.time_split_thresh)
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
    print('Begin compute track scores:')
    print(_TRACKS)
    _TRACKS = gpd_compute_scores(_TRACKS, sub_dset, thrs, ks,
                                 USE_DASK=USE_DASK,
                                 resolution=config.resolution)

    print('Finished computing track scores:')
    print(_TRACKS)
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

    if config.time_split_thresh:
        split_filter = TimeSplitFilter(config.time_split_thresh, config.time_split_frame_buffer)
        n_orig = gpd_len(_TRACKS)
        _TRACKS = split_filter(_TRACKS)
        n_result = gpd_len(_TRACKS)
        print('filter based on time splitting: remaining tracks '
              f'{n_result} / {n_orig}')

    # The tracker assumes the polygons will be output in video space.
    if scale_vid_from_trk is not None and len(_TRACKS):
        # If a tracking resolution was specified undo the extra scale factor
        _TRACKS['poly'] = _TRACKS['poly'].scale(*scale_vid_from_trk, origin=(0, 0))

    # TODO: do we need to convert to MultiPolygon here? Or can that be handled
    # by consumers of this method?
    _TRACKS['poly'] = _TRACKS['poly'].map(kwimage.MultiPolygon.from_shapely)
    print('Returning Tracks')
    print(_TRACKS)
    return _TRACKS


def _determine_tracking_scale(config, sub_dset, video_gids, video):
    """
    Factored out code from :func:`time_aggregated_polys`
    """
    import numpy as np
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
    return scale_vid_from_trk, tracking_gsd


#
# --- time_aggregated_polys utilities ---
#


def _merge_polys(p1, t1, p2, t2, poly_merge_method=None):
    """
    Given two lists of polygons, p1 and p2, merge these according to:
      - add all unique polygons in the merged list
      - for overlapping polygons, add the union of both polygons

    Args:
        p1 (List[shapely.geometry.polygon.Polygon]):
            List of polygons in group1

        t1 (List[float]):
            List of times corresponding with polygons in group1

        p2 (List[shapely.geometry.polygon.Polygon]):
            List of polygons in group1

        t2 (List[float]):
            List of times corresponding with polygons in group2

        poly_merge_method (str):
            Codename for the algorithm used. Can be "v1", "v2", "v3", or "v3_noop".

    Example:
        >>> from watch.tasks.tracking.from_heatmap import * # NOQA
        >>> from watch.tasks.tracking.from_heatmap import _merge_polys  # NOQA
        >>> import kwimage
        >>> import numpy as np
        >>> #
        >>> p1 = [kwimage.Polygon.random().scale(0.2).to_shapely() for _ in range(1)]
        >>> t1 = np.arange(len(p1) * 2).reshape(-1, 2)
        >>> p2 = [kwimage.Polygon.random().to_shapely() for _ in range(1)]
        >>> t2 = np.arange(len(p2) * 2).reshape(-1, 2)
        >>> poly_merge_method = 'v3'
        >>> #
        >>> _merge_polys(p1, t1, p2, t2, poly_merge_method)

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
    from shapely.ops import unary_union
    merged_polys = []
    merged_times = []

    if poly_merge_method is None:
        poly_merge_method = 'v1'

    elif poly_merge_method == 'v3_noop':
        merged_polys = p1 + p2
        merged_times = t1 + t2

    elif poly_merge_method == 'v3':
        p1_seen = set()
        p2_seen = set()
        # add all polygons that overlap
        for j, (_p1, _t1) in enumerate(zip(p1, t1)):
            if j in p1_seen:
                continue
            for i, (_p2, _t2) in enumerate(zip(p2, t2)):

                # if timestamps dont line up, skip
                if _t1[1] != _t2[0]:
                    continue
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
            merged_times.append(t1[index])

        all_p2 = set(np.arange(len(p2)))
        remaining_p2 = all_p2 - p2_seen
        for index in remaining_p2:
            merged_polys.append(p2[index])
            merged_times.append(t2[index])

    elif poly_merge_method == 'v2':
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

    return merged_polys, merged_times


def _process(track, _heatmaps, image_dates, gids, config):
    """
    Yields:
        Tuple[List[int], MultiPolygon] -
            a list of image ids a polygon is valid for, and
            the single polygon corresponding.
    """
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


@profile
def heatmaps_to_polys_new(heatmaps, track_bounds, heatmap_dates=None, config=None):
    """
    Use parameters: agg_fn, thresh, morph_kernel, thresh_hysteresis, norm_ord

    Args:
        heatmaps (ndarray): A [T, H, W] heatmap

        track_bounds (kwimage.MultiPolygon | None):
            a valid region in the heatmaps where new polygons can be extracted.

        heatmap_dates (List[datetime] | None):
            dates corresponding with each heatmap time dimension

        config (PolygonExtractConfig): polygon extraction config

    Example:
        >>> from watch.tasks.tracking.from_heatmap import *  # NOQA
        >>> import kwimage
        >>> from kwutil import util_time
        >>> import numpy as np
        >>> from watch.tasks.tracking.from_heatmap import PolygonExtractConfig  # NOQA
        >>> config = PolygonExtractConfig()
        >>> heatmaps = np.zeros((7, 64, 64))
        >>> heatmaps[2, 20:40, 20:40] = 1
        >>> heatmaps[5, 30:50, 30:50] = 1
        >>> heatmap_dates = [util_time.coerce_datetime(x) for x in [
        >>>     '2020-01-01', '2020-02-01', '2020-03-01', '2020-04-01',
        >>>     '2020-05-01', '2020-06-01', '2020-07-01', ]]
        >>> track_bounds = kwimage.Polygon.random(rng=0).scale((64, 64))
        >>> # V1 merges everything together across all time
        >>> config.poly_merge_method = 'v1'
        >>> polygons_final = heatmaps_to_polys(heatmaps, track_bounds, heatmap_dates=heatmap_dates, config=config)
        >>> # V3 does some time separation
        >>> config.poly_merge_method = 'v3'
        >>> polygons_final = heatmaps_to_polys(heatmaps, track_bounds, heatmap_dates=heatmap_dates, config=config)
    """
    from kwutil import util_time
    import numpy as np

    # TODO: rename moving window size to "outer_window_size"

    def convert_to_shapely(polys):
        return [p.to_shapely() for p in polys]

    def convert_to_kwimage_poly(shapely_polys):
        import kwimage
        return [kwimage.Polygon.from_shapely(p) for p in shapely_polys]

    # outer agg function
    _agg_fn = agg_functions.AGG_FN_REGISTRY[config.agg_fn]

    if config.inner_window_size is not None:
        # TODO: generalize if needed
        assert heatmap_dates is not None

        if config.inner_agg_fn == 'mean':
            inner_ord = 1
        elif config.inner_agg_fn == 'max':
            inner_ord = float('inf')
        else:
            raise NotImplementedError(config.inner_agg_fn)

        # Do inner aggregation before outer aggregation
        groupxs = _compute_time_window(
            config.inner_window_size, num_frames=len(heatmaps),
            heatmap_dates=heatmap_dates)

        new_heatmaps = []
        new_intervals = []
        for idxs in groupxs:
            new_start_date = min(ub.take(heatmap_dates, idxs))
            new_end_date = max(ub.take(heatmap_dates, idxs))
            inner = agg_functions._norm(heatmaps[idxs], norm_ord=inner_ord)
            new_intervals.append((new_start_date, new_end_date))
            new_heatmaps.append(inner)

        heatmap_date_intervals = new_intervals
        heatmaps = np.array(new_heatmaps)
    else:
        if config.inner_window_size is not None:
            raise NotImplementedError(
                'only temporal deltas for inner agg window for now')

        if heatmap_dates is None:
            heatmap_dates = [util_time.coerce_datetime(0)] * len(heatmaps)
        heatmap_date_intervals = [(t, t) for t in heatmap_dates]

    heatmap_unixtime_intervals = np.array([
        (a.timestamp(), b.timestamp())
        for a, b in heatmap_date_intervals
    ], dtype=np.float64)
    heatmap_dates = [util_time.coerce_datetime(a) for (a, b) in
                     heatmap_unixtime_intervals]

    # calculate number of moving-window steps, based on window_size and number
    # of heatmaps
    groupxs = _compute_time_window(
        config.inner_window_size, num_frames=len(heatmaps),
        heatmap_dates=heatmap_dates)

    # initialize heatmaps and initial polygons on the first set of heatmaps
    n_steps = len(groupxs)
    xs_init = groupxs[0]
    h_init = heatmaps[xs_init]
    t_init = heatmap_unixtime_intervals[xs_init]

    prog = ub.ProgIter(total=n_steps, desc='process-step')
    # prog.begin()
    with prog:
        step_idx = 0
        polys_final = _process_1_step(h_init, _agg_fn, track_bounds, step_idx, config)
        times_final = [[t_init[0][0], t_init[-1][1]]] * len(polys_final)
        prog.step()

        if n_steps > 1:
            polys_final = convert_to_shapely(polys_final)

            for step_idx in range(1, n_steps):
                idxs = groupxs[step_idx]
                prog.step()
                h1 = heatmaps[idxs]
                t1 = heatmap_unixtime_intervals[idxs]

                p1 = _process_1_step(h1, _agg_fn, track_bounds, step_idx, config)
                t1 = [[t1[0][0], t1[-1][1]]] * len(p1)
                p1 = convert_to_shapely(p1)

                polys_final, times_final = _merge_polys(
                    polys_final, times_final,
                    p1, t1,
                    poly_merge_method=config.poly_merge_method,
                )

            polys_final = convert_to_kwimage_poly(polys_final)
    return polys_final


def heatmaps_to_polys_orig(heatmaps, track_bounds, heatmap_dates=None, config=None):
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

    _agg_fn = agg_functions.AGG_FN_REGISTRY[config.agg_fn]

    image_unixtimes = np.array([d.timestamp() for d in heatmap_dates])

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
        bucket_ids = (image_unixtimes // delta).astype(int)

        unique_ids, groupxs = kwarray.group_indices(bucket_ids)

        new_heatmaps = []
        for idxs in groupxs:
            inner = agg_functions._norm(heatmaps[idxs], norm_ord=inner_ord)
            new_heatmaps.append(inner)
        new_heatmaps = np.array(new_heatmaps)
        heatmaps = new_heatmaps

        new_heatmap_dates = []
        for idxs in groupxs:
            new_start_date = np.min(image_unixtimes[idxs])
            new_end_date = np.max(image_unixtimes[idxs])
            new_heatmap_dates.append([new_start_date, new_end_date])
        new_heatmap_dates = np.array(new_heatmap_dates)
        image_unixtimeframes = new_heatmap_dates

    else:
        if config.inner_window_size is not None:
            raise NotImplementedError(
                'only temporal deltas for inner agg window for now')
        image_unixtimeframes = np.stack([image_unixtimes, image_unixtimes], axis=-1)

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
    t_init = image_unixtimeframes[:final_size]

    prog = ub.ProgIter(total=n_steps, desc='process-step')
    with prog:
        step = 0
        polys_final = _process_1_step(h_init, _agg_fn, track_bounds, step, config)
        times_final = [[t_init[0][0], t_init[-1][1]]] * len(polys_final)
        prog.step()

        if n_steps > 1:
            polys_final = convert_to_shapely(polys_final)

            for step in range(1, n_steps):
                prog.step()
                h1 = heatmaps[step * final_size:(step + 1) * final_size]
                t1 = image_unixtimeframes[step * final_size:(step + 1) * final_size]

                p1 = _process_1_step(h1, _agg_fn, track_bounds, step, config)
                t1 = [[t1[0][0], t1[-1][1]]] * len(p1)
                p1 = convert_to_shapely(p1)

                polys_final, times_final = _merge_polys(
                    polys_final, times_final,
                    p1, t1,
                    poly_merge_method=config.poly_merge_method,
                )

            polys_final = convert_to_kwimage_poly(polys_final)
    return polys_final


heatmaps_to_polys = heatmaps_to_polys_orig


def _compute_time_window(window, num_frames=None, heatmap_dates=None):
    """
    Example:
        >>> window = 5
        >>> num_frames = 23
        >>> groupxs = _compute_time_window(window, num_frames)
        >>> print(f'groupxs={groupxs}')
        >>> #
        >>> window = '7days'
        >>> from kwutil import util_time
        >>> heatmap_dates = list(map(util_time.coerce_datetime, [
        >>>     '2020-01-01', '2020-01-02', '2020-02-01',
        >>>     '2020-02-02', '2020-03-14', '2020-03-23',
        >>>     '2020-04-01', '2020-06-23', '2020-06-26',
        >>>     '2020-06-27', '2020-06-28', ]))
        >>> groupxs = _compute_time_window(window, num_frames, heatmap_dates)
        >>> print(f'groupxs={groupxs}')
        >>> groupxs = _compute_time_window(None, num_frames, heatmap_dates)
        >>> print(f'groupxs={groupxs}')
    """
    import kwarray
    from kwutil import util_time
    import numpy as np
    if window is None:
        bucket_ids = np.arange(num_frames)
    elif isinstance(window, str):
        assert heatmap_dates is not None
        delta = util_time.coerce_timedelta(window).total_seconds()
        image_unixtimes = np.array([d.timestamp() for d in heatmap_dates])
        image_unixtimes = image_unixtimes - image_unixtimes[0]
        bucket_ids = (image_unixtimes // delta).astype(int)
    elif isinstance(window, int):
        assert num_frames is not None
        frame_indexes = np.arange(num_frames)
        bucket_ids = frame_indexes // window
    else:
        raise NotImplementedError('')
    unique_ids, groupxs = kwarray.group_indices(bucket_ids)
    return groupxs


def _process_1_step(heatmaps, _agg_fn, track_bounds, step_idx, config):
    """
    Args:
        heatmaps (ndarray):
        _agg_fn (Callable):
        track_bounds (None | Coercable[kwimage.MultiPolygon]):
        step_idx (int):
        config (DataConfig):

    Returns:
        List[kwimage.Polygon]
    """
    # FIXME: no dynamic globals.
    if config.viz_out_dir is not None:
        viz_dpath = (config.viz_out_dir / f'heatmaps_{step_idx}').ensuredir()
        # print('\nviz_dpath = {}\n'.format(ub.urepr(viz_dpath, nl=1)))
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
    This is associated with :class:`PolygonExtractConfig`

    Example:
        >>> from watch.tasks.tracking.from_heatmap import *  # NOQA
        >>> from watch.tasks.tracking.from_heatmap import _gids_polys
        >>> import watch
        >>> coco_dset = watch.coerce_kwcoco(data='watch-msi', dates=True, geodata=True, heatmap=True)
        >>> sub_dset = coco_dset.subset(coco_dset.videos().images[0])
        >>> key = ['salient']
        >>> agg_fn = 'probs'
        >>> thresh = 0.001
        >>> morph_kernel = 3
        >>> thresh_hysteresis = 0
        >>> norm_ord = 1
        >>> resolution = None
        >>> outer_window_size = None
        >>> inner_window_size = '1year'
        >>> kwargs = dict(key=key,
        >>>     agg_fn=agg_fn,
        >>>     thresh=thresh,
        >>>     morph_kernel=morph_kernel,
        >>>     thresh_hysteresis=thresh_hysteresis,
        >>>     norm_ord=norm_ord,
        >>>     resolution=resolution,
        >>>     outer_window_size=outer_window_size,
        >>>     use_boundaries=None)
        >>> results1 = list(_gids_polys(sub_dset, **kwargs))
        >>> kwargs['new_algo'] = 'crall'
        >>> results2 = list(_gids_polys(sub_dset, **kwargs))

    Returns:
        Iterable[Tuple[List[int], MultiPolygon]] -
            For each track return a list of image ids and a single associated
            polygon.
    """
    from kwutil import util_time
    import numpy as np
    import rich
    config = PolygonExtractConfig(**kwargs)

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

    channels_list = config.key
    channels = '|'.join(config.key)
    coco_images = sub_dset.images(gids).coco_images

    load_workers = 0  # TODO: configure
    load_jobs = ub.JobPool(mode='process', max_workers=load_workers)

    print(f'Reading heatmaps with: channels={channels}')

    with load_jobs:
        for coco_img in ub.ProgIter(coco_images, desc='submit heatmap jobs'):
            delayed = coco_img.imdelay(channels=channels, space='video', resolution=config.resolution)
            load_jobs.submit(delayed.finalize)

        _heatmaps = []
        for job in ub.ProgIter(load_jobs.jobs, desc='collect heatmap jobs'):
            _heatmap = job.result()
            _heatmaps.append(_heatmap)

    _heatmaps_thwc = np.stack(_heatmaps, axis=0)

    if config.new_algo is not None:
        from watch.tasks.tracking import polygon_extraction
        extractor = polygon_extraction.PolygonExtractor(
            _heatmaps_thwc,
            heatmap_time_intervals=image_dates,
            bounds=None, classes=channels_list,
            config=config.asdict())
        polygons = extractor.predict_polygons()

        # Conform to expected output
        result_gen = []
        for poly in polygons:
            single_result = (gids, poly)
            result_gen.append(single_result)
    else:
        print(f'(presum) _heatmaps_thwc.shape={_heatmaps_thwc.shape}')
        _heatmaps = _heatmaps_thwc.sum(axis=-1)  # sum over channels
        print(f'_heatmaps.shape={_heatmaps.shape}')
        missing_ix = np.array([channels not in i.channels for i in coco_images])

        num_missing = missing_ix.sum()
        rich.print(f'[yellow]There are {num_missing} images that are missing {channels} channels')

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


class PolygonExtractConfig(scfg.DataConfig):
    # This is the base config that all from-heatmap trackers have in common
    # which has to do with how heatmaps are loaded, normalized, and aggregated.
    # This is associated with :func:`_gids_polys`

    new_algo = scfg.Value(None, help=ub.paragraph(
        '''
        If None, use the old algorithm, otherwise use one of the new algorithm
        '''))

    key = scfg.Value('salient', help=ub.paragraph(
        '''
        One or more channels to use as positive class for binary heatmap
        polygon extraction and scoring.
        '''))

    agg_fn = scfg.Value('probs', help=ub.paragraph(
        '''
        The aggregation method to preprocess heatmaps.
        See ``AGG_FN_REGISTRY`` for available options.
        '''), alias=['outer_agg_fn'])

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

    # TODO: Consolidate into agg_fn
    norm_ord = scfg.Value(1, help=ub.paragraph(
        '''
        The generalized mean order used to average heatmaps over the
        "outer_window_size". A value of 1 is the normal mean. A value of inf
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


class TimeAggregatedPolysConfig(PolygonExtractConfig):
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

    time_split_thresh = scfg.Value(None, help=ub.paragraph(
        '''
        time splitting parameter. if set, tracks will be broken into subtracks
        based on when the score is above this threshold.
        '''))

    time_split_frame_buffer = scfg.Value(2, help=ub.paragraph(
        '''
        time splitting parameter. if set, subtracks will be buffered by the specified
        number of frames. if this causes subtracks to overlap, they are merged together.
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
    def __post_init__(self):
        super().__post_init__()
        if isinstance(self.norm_ord, str) and self.norm_ord.lower() == 'inf':
            self.norm_ord = float('inf')


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
        print('Tracks:')
        print(tracks)
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
        boundaries_as: use for Site Boundary annots in coco_dsennjk
            'bounds': generated polys will lie inside the boundaries
            'polys': generated polys will be the boundaries
            'none': generated polys will ignore the boundaries
        """
        import kwcoco
        import kwimage

        print(f'self={self}')
        print(f'self.boundaries_as={self.boundaries_as}')
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
        print('Tracks:')
        print(tracks)
        return tracks

    def add_tracks_to_dset(self, sub_dset, tracks, **kwargs):
        import kwcoco
        if self.boundaries_as != 'polys':
            if 0:
                col_map = {}
                for c in tracks.columns:
                    if c[0] == 'fg':
                        k = kwcoco.ChannelSpec('|'.join(self.key)).spec
                        col_map[c] = (k, *c[1:])
                    elif c[0] == 'bg':
                        k = kwcoco.ChannelSpec('|'.join(self.bg_key)).spec
                        col_map[c] = (k, *c[1:])
                print(f'col_map={col_map}')
                # weird effect here - reassignment casts from GeoDataFrame to
                # DataFrame. Related to invalid geometry column?
                # tracks = tracks.rename(columns=col_map)
                tracks.rename(columns=col_map, inplace=True)

        thresh = self.thresh
        key = self.key
        bg_key = self.bg_key
        print(tracks)
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
