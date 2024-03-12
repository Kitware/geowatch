"""
Main tracker logic

SeeAlso:
    * ../../cli/run_tracker.py
"""
import ubelt as ub
import math
from typing import Tuple
from typing import Literal
import scriptconfig as scfg

from geowatch.heuristics import SITE_SUMMARY_CNAME, CNAMES_DCT
from geowatch.tasks.tracking.abstract_classes import NewTrackFunction

from geowatch.tasks.tracking.old_polygon_extraction import PolygonExtractConfig
from geowatch.tasks.tracking.old_polygon_extraction import _gids_polys

from geowatch.tasks.tracking.utils import (
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


class DataFrameFilter:

    def __call__(self, gdf):
        raise AssertionError("Use the explicit .filter_dataframe method instead")
        return self.filter_dataframe(gdf)

    def filter_dataframe(self, gdf):
        raise NotImplementedError


class TimePolygonFilter(DataFrameFilter):
    """
    Cuts off start and end of each track based on min response.
    """

    def __init__(self, threshold):
        self.threshold = threshold

    def filter_dataframe(self, gdf):

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


class TimeSplitFilter(DataFrameFilter):
    """
    Splits tracks based on start and end of each subtracks min response.
    """

    def __init__(self, threshold, frame_buffer):
        self.threshold = threshold
        self.frame_buffer = frame_buffer

    def filter_dataframe(self, gdf):
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


class ResponsePolygonFilter(DataFrameFilter):
    """
    Filters each track based on the average response of all tracks.
    """

    def __init__(self, gdf, threshold):

        self.threshold = threshold

        gids = gdf['gid'].unique()
        mean_response = gdf[('fg', -1)].mean()

        self.gids = gids
        self.mean_response = mean_response

    def filter_dataframe(self, gdf, gids=None, threshold=None, cross=True):
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
    from geowatch.utils import kwcoco_extensions
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
        # ASSIGN_ANNOTS_SCORE
        score_chan = kwcoco.ChannelSpec('|'.join(key))

        for tid, grp in tracks.groupby('track_idx', axis=0):

            try:
                this_score = grp[(score_chan.spec, -1)]
            except Exception:
                # HACK
                this_score = grp[('ac_salient', -1)]

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
        from kwutil.util_json import debug_json_unserializable
        debug_json_unserializable(sub_dset.dataset)

    return sub_dset


@profile
def site_validation(sub_dset, thresh=0.25, span_steps=15):
    """
    Example:
        >>> import geowatch
        >>> from geowatch.tasks.tracking.from_heatmap import *  # NOQA
        >>> coco_dset = geowatch.coerce_kwcoco(
        >>>     'geowatch-msi', heatmap=True, geodata=True, dates=True)
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
def time_aggregated_polys(sub_dset, video_id, **kwargs):
    """
    Polygon extraction and tracking function.

    Aggregate heatmaps across time, threshold them to get polygons,
    and add one track per polygon.

    Args:
        sub_dset (kwcoco.CocoDataset): a kwcoco dataset with exactly 1 video

        video_id (int): The video-id to track.

        **kwargs:
            see :class:`TimeAggregatedPolysConfig` and
            :class:`PolygonExtractConfig`.

    Ignore:
        # For debugging
        import xdev
        from geowatch.tasks.tracking.from_heatmap import *  # NOQA
        from geowatch.tasks.tracking.from_heatmap import _validate_keys
        globals().update(xdev.get_func_kwargs(time_aggregated_polys))

    Example:
        >>> # test interpolation
        >>> from geowatch.tasks.tracking.from_heatmap import time_aggregated_polys
        >>> from geowatch.demo import demo_kwcoco_with_heatmaps
        >>> import geowatch
        >>> sub_dset = geowatch.coerce_kwcoco(
        >>>     'geowatch-msi', num_videos=1, num_frames=5, image_size=(128, 128),
        >>>     geodata=True, heatmap=True, dates=True)
        >>> thresh = 0.01
        >>> video_id = list(sub_dset.videos())[0]
        >>> min_area_square_meters = None
        >>> kwargs = dict(thresh=thresh, min_area_square_meters=min_area_square_meters, time_thresh=None)
        >>> orig_track = time_aggregated_polys(sub_dset, video_id, **kwargs)
        >>> # Test robustness to frames that are missing heatmaps
        >>> skip_gids = [1,3]
        >>> for gid in skip_gids:
        >>>      sub_dset.imgs[gid]['auxiliary'].pop()
        >>> inter_track = time_aggregated_polys(sub_dset, video_id, **kwargs)
        >>> assert inter_track.iloc[0][('fg', -1)] == 0
        >>> assert inter_track.iloc[1][('fg', -1)] > 0

    Example:
        >>> # test interpolation
        >>> from geowatch.tasks.tracking.from_heatmap import time_aggregated_polys
        >>> from geowatch.demo import demo_kwcoco_with_heatmaps
        >>> import geowatch
        >>> sub_dset = geowatch.coerce_kwcoco(
        >>>     'geowatch-msi', num_videos=1, num_frames=5, image_size=(128, 128),
        >>>     geodata=True, heatmap=True, dates=True)
        >>> video_id = list(sub_dset.videos())[0]
        >>> thresh = 0.01
        >>> min_area_square_meters = None
        >>> kwargs = dict(thresh=thresh, min_area_square_meters=min_area_square_meters, time_thresh=None)
        >>> orig_track = time_aggregated_polys(sub_dset, video_id, **kwargs)
        >>> # Test robustness to frames that are missing heatmaps
        >>> skip_gids = [1,3]
        >>> for gid in skip_gids:
        >>>      sub_dset.imgs[gid]['auxiliary'].pop()
        >>> inter_track = time_aggregated_polys(sub_dset, video_id, **kwargs)
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

    # coco_videos = sub_dset.videos()
    # assert len(coco_videos) == 1, 'we expect EXACTLY one video here'
    assert video_id is not None
    coco_videos = sub_dset.videos(video_ids=[video_id])
    video = coco_videos.objs[0]
    video_name = video.get('name', None)
    # video_id = video['id']

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
    gids_polys = _gids_polys(sub_dset, video_id, **gid_poly_config)

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

    # At this point each row corresponds to a single track and the each gid
    # cell contains a list of image ids.
    _TRACKS_COMPACT = gpd.GeoDataFrame({'gid': gids, 'poly': polys}, geometry='poly')
    if config.polygon_simplify_tolerance is not None:
        _TRACKS_COMPACT['poly'] = _TRACKS_COMPACT['poly'].simplify(tolerance=config.polygon_simplify_tolerance)
    _TRACKS_COMPACT = _TRACKS_COMPACT.reset_index(names='track_idx')

    # Explode takes each row with multiple gids and expands it creating a new
    # row for each item in the exploeded column. That means we go from a
    # dataframe that looks like:
    # [
    #   {'gid': [1, 2, 3], 'track_idx': 0, 'poly': POLY1},
    #   {'gid': [5, 7], 'track_idx': 1, 'poly': POLY2},
    # ]
    # TO:
    # [
    #   {'gid': 1, 'track_idx': 0, 'poly': POLY1},
    #   {'gid': 2, 'track_idx': 0, 'poly': POLY1},
    #   {'gid': 3, 'track_idx': 0, 'poly': POLY1},
    #   {'gid': 5, 'track_idx': 1, 'poly': POLY2},
    #   {'gid': 7, 'track_idx': 1, 'poly': POLY2},
    # ]
    _TRACKS = _TRACKS_COMPACT.explode('gid')

    # ensure index is sorted in video order
    sorted_gids = sub_dset.images(video_id=video['id']).gids
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

    """
    Cases:
        For BAS:
            ks = {'fg': ['salient'], 'bg': []}

        For AC:
            {'fg': ['Site Preparation', 'Active Construction', 'Post Construction', 'No Activity', 'ac_salient'],
             'bg': ['No Activity']}

    """

    ks = {'fg': config.key, 'bg': config.bg_key}

    # TODO dask gives different results on polys that overlap nodata area, need
    # to debug this. (6% of polygons in KR_R001, so not a huge difference)
    # USE_DASK = True
    USE_DASK = False
    print('Begin compute track scores:')
    # Note: this is the function also called by
    # :func:`score_track_polys`

    modulate = None
    if config.modulate_post_construction is not None:
        modulate = {}
        modulate['Post Construction'] = float(config.modulate_post_construction)

    _TRACKS = gpd_compute_scores(_TRACKS, sub_dset, thrs, ks,
                                 USE_DASK=USE_DASK,
                                 resolution=config.resolution,
                                 modulate=modulate)

    rich.print('[green]Finished computing track scores:')
    rich.print(_TRACKS)
    if _TRACKS.empty:
        return _TRACKS

    # dask could unsort
    _TRACKS = gpd_sort_by_gid(_TRACKS.reset_index(), sorted_gids)

    # response_thresh = 0.9
    if config.response_thresh:

        n_orig = gpd_len(_TRACKS)
        rsp_filter = ResponsePolygonFilter(_TRACKS, config.key, config.response_thresh)
        _TRACKS = rsp_filter.filter_dataframe(_TRACKS)
        print('filter based on per-polygon response: remaining tracks '
              f'{gpd_len(_TRACKS)} / {n_orig}')

    # TimePolygonFilter edits tracks instead of removing them
    if config.time_thresh:  # as a fraction of thresh
        time_filter = TimePolygonFilter(config.time_thresh * config.thresh)
        n_orig = gpd_len(_TRACKS)
        _TRACKS = time_filter.filter_dataframe(_TRACKS)  # 7% of runtime? could be next line
        print('filter based on time overlap: remaining tracks '
              f'{gpd_len(_TRACKS)} / {n_orig}')

    if config.time_split_thresh:
        split_filter = TimeSplitFilter(config.time_split_thresh, config.time_split_frame_buffer)
        n_orig = gpd_len(_TRACKS)
        _TRACKS = split_filter.filter_dataframe(_TRACKS)
        n_result = gpd_len(_TRACKS)
        print('filter based on time splitting: remaining tracks '
              f'{n_result} / {n_orig}')

    # The tracker assumes the polygons will be output in video space.
    # rich.print('[red]!!!!!!!!!')
    # print(f'scale_vid_from_trk={scale_vid_from_trk}')
    # print(f'scale_vid_from_trk={scale_vid_from_trk}')
    # print(f'scale_vid_from_trk={scale_vid_from_trk}')
    # print(f'scale_vid_from_trk={scale_vid_from_trk}')
    # rich.print('[red]!!!!!!!!!')
    if scale_vid_from_trk is not None and len(_TRACKS):
        # If a tracking resolution was specified undo the extra scale factor
        _TRACKS['poly'] = _TRACKS['poly'].scale(*scale_vid_from_trk, origin=(0, 0))

    # TODO: do we need to convert to MultiPolygon here? Or can that be handled
    # by consumers of this method?
    _TRACKS['poly'] = _TRACKS['poly'].map(kwimage.MultiPolygon.from_shapely)
    rich.print('[green]Returning Tracks')
    rich.print(_TRACKS)
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
# --- wrappers ---
#
# Note:
#     The following are valid choices of `track_fn` in
#     ../../cli/run_tracker.py and will be called by ./normalize.py

class TimeAggregatedPolysConfig(PolygonExtractConfig):
    """
    This is an intermediate config that we will use to transition between the
    current dataclass configuration and a new scriptconfig based one.

    python -c "if 1:
        from geowatch.tasks.tracking.from_heatmap import TimeAggregatedBAS
        TimeAggregatedBAS().argparse().print_help()
    "

    """
    bg_key = scfg.Value(None, help=ub.paragraph(
        '''
        Zero or more channels to use as the negative class for polygon scoring.

        bg_key (String | List[String] | None): background key(s).
            If None, background heatmaps become 1 - sum(foreground keys)
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

    modulate_post_construction = scfg.Value(None, help=ub.paragraph(
        '''
        Hacked in POC command to multiply post scores.
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

    def create_tracks(self, sub_dset, video_id):
        aggkw = ub.udict(self) & TimeAggregatedPolysConfig.__default__.keys()
        tracks = time_aggregated_polys(sub_dset, video_id, **aggkw)
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
        This is a valid choice of `track_fn` in ../../cli/run_tracker.py
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

    def create_tracks(self, sub_dset, video_id):
        """
        boundaries_as: use for Site Boundary annots in coco_dsennjk
            'bounds': generated polys will lie inside the boundaries
            'polys': generated polys will be the boundaries
            'none': generated polys will ignore the boundaries
        """
        import kwcoco
        import kwimage
        import rich
        rich.print('[white] --- Create Tracks ---')
        if self.boundaries_as == 'polys':

            # Just score the polygons, no need to extract
            tracks = score_track_polys(
                sub_dset,
                video_id,
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
            # Need to extract and score
            aggkw = ub.udict(self) & TimeAggregatedPolysConfig.__default__.keys()
            aggkw['use_boundaries'] = str(self.get('boundaries_as', 'none')).lower() not in {'none', 'null'}
            tracks = time_aggregated_polys(sub_dset, video_id, **aggkw)
        print('Tracks:')
        print(tracks)
        rich.print('[white] ---')
        return tracks

    def add_tracks_to_dset(self, sub_dset, tracks, **kwargs):
        import rich
        rich.print('[white] --- Add Tracks To Dataset ---')
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
        rich.print('[white] ---')
        return sub_dset


class TimeAggregatedSV(CommonTrackFn):
    """
    Wrapper for Site Validation that looks for phase heatmaps.

    Alias:
        site_validation

    Note:
        This is a valid choice of `track_fn` in ../../cli/run_tracker.py
    """
    thresh: float = 0.1
    key: str = 'salient'
    boundaries_as: Literal['bounds', 'polys', 'none'] = 'polys'
    span_steps: int = 120

    def create_tracks(self, sub_dset, video_id):
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
                video_id,
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
