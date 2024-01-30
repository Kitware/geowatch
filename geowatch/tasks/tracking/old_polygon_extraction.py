"""
The original heatmap -> polygon extraction code.
"""
import ubelt as ub
import itertools
import scriptconfig as scfg

from geowatch.heuristics import SITE_SUMMARY_CNAME
from geowatch.tasks.tracking import agg_functions

from geowatch.tasks.tracking.utils import (
    mask_to_polygons,
    score_track_polys,
    gpd_len)


class PolygonExtractConfig(scfg.DataConfig):
    # This is the base config that all from-heatmap trackers have in common
    # which has to do with how heatmaps are loaded, normalized, and aggregated.
    # This is associated with :func:`_gids_polys`

    new_algo = scfg.Value(None, help=ub.paragraph(
        '''
        If None, use the old algorithm, otherwise use one of the new algorithm.
        This is a hack that needs to get redesigned and integrated in a better
        way.
        '''))

    key = scfg.Value('salient', help=ub.paragraph(
        '''
        One or more channels to use as positive class for binary heatmap
        polygon extraction and scoring.
        '''))

    agg_fn = scfg.Value('probs', help=ub.paragraph(
        '''
        The aggregation method to preprocess heatmaps.
        See ``agg_functions.AGG_FN_REGISTRY`` for available options.
        (3d heatmaps -> 2d heatmaps), calling convention TBD
        '''), alias=['outer_agg_fn'])

    thresh = scfg.Value(0.0, help=ub.paragraph(
        '''
        The threshold for polygon extraction from heatmaps.
        E.g. this threshold binarizes the heatmaps.

        For each frame, if sum of foreground heatmaps > thresh,
            class is max(foreground keys).
            else, class is max(background keys).
        '''))

    morph_kernel = scfg.Value(3, help=ub.paragraph(
        '''
        Morphology kernel for preprocessing the heatmaps with dilation.
        Height/width in px of close or dilate kernel
        '''))

    thresh_hysteresis = scfg.Value(None, help=ub.paragraph(
        '''
        I dont remember. Help wanted to document this, or remove it.
        '''))

    # TODO: Consolidate into agg_fn
    norm_ord = scfg.Value(1, help=ub.paragraph(
        '''
        The generalized mean order used to average heatmaps over the
        "outer_window_size". A value of 1 is the normal mean. A value of inf
        is the max function. Note: this is effectively an outer_agg_fn.

        order of norm to aggregate heatmap pixels across time.
        1: average [default]
        2: euclidean
        0: sum
        np.inf, 'inf', or None: max
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


def _gids_polys(sub_dset, video_id, **kwargs):
    """
    This is associated with :class:`PolygonExtractConfig`

    Example:
        >>> from geowatch.tasks.tracking.old_polygon_extraction import *  # NOQA
        >>> from geowatch.tasks.tracking.old_polygon_extraction import _gids_polys
        >>> import geowatch
        >>> coco_dset = geowatch.coerce_kwcoco(data='geowatch-msi', dates=True, geodata=True, heatmap=True)
        >>> sub_dset = coco_dset.subset(coco_dset.videos().images[0])
        >>> video_id = list(sub_dset.videos())[0]
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
        >>> results1 = list(_gids_polys(sub_dset, video_id, **kwargs))
        >>> kwargs['new_algo'] = 'crall'
        >>> results2 = list(_gids_polys(sub_dset, video_id, **kwargs))

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
        raw_boundary_tracks = score_track_polys(sub_dset, video_id, [SITE_SUMMARY_CNAME],
                                                resolution=config.resolution)

        if len(raw_boundary_tracks) == 0:
            gids = sub_dset.images(video_id=video_id).gids

            print(f'SITE_SUMMARY_CNAME={SITE_SUMMARY_CNAME}')
            print(f'config.resolution={config.resolution}')
            print(f'sub_dset={sub_dset}')
            print(f'video_id={video_id}')
            # anns = sub_dset.annots(video_id=video_id)
            anns = sub_dset.annots()
            set(anns.images.lookup('video_id'))
            boundary_tracks = [(None, None)]
            import warnings
            msg = ('need valid site boundaries!')
            warnings.warn(msg)
            # raise AssertionError(msg)
        else:
            gids = raw_boundary_tracks['gid'].unique()
            print('generating polys in bounds: number of bounds: ',
                  gpd_len(raw_boundary_tracks))
            boundary_tracks = list(raw_boundary_tracks.groupby('track_idx'))

    else:
        boundary_tracks = [(None, None)]
        # TODO WARNING this is wrong!!! need to make sure this is never used.
        # The gids are lexically sorted, not sorted by order in video!
        # gids = list(sub_dset.imgs.keys())
        # vidid = list(sub_dset.index.vidid_to_gids.keys())[0]
        gids = sub_dset.images(video_id=video_id).gids

    images = sub_dset.images(gids)
    image_dates = [util_time.coerce_datetime(d)
                   for d in images.lookup('date_captured')]
    # image_years = [d.year for d in image_dates]

    channels_list = config.key
    channels = '|'.join(config.key)
    coco_images = images.coco_images

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
        # HACK in new algo
        import kwimage
        if config.use_boundaries:
            # HACK TO LOAD BOUNDS FOR POLYGONS
            from geowatch.tasks.tracking.utils import _build_annot_gdf
            cnames = [SITE_SUMMARY_CNAME]
            resolution = config.resolution
            aids = list(ub.flatten(images.annots))
            gdf, flat_scales = _build_annot_gdf(sub_dset, aids=aids, cnames=cnames, resolution=resolution)
            if len(gdf) == 0:
                # assert len(gdf) > 0, 'need valid site boundaries!'
                bounds = None
            else:
                union_poly = gdf.unary_union
                bounds = kwimage.MultiPolygon.from_shapely(union_poly)
        else:
            bounds = None

        video_name = sub_dset.index.videos[video_id]['name']
        if config.viz_out_dir is None:
            extractor_viz_dir = None
        else:
            extractor_viz_dir = ub.Path(config.viz_out_dir) / video_name

        from geowatch.tasks.tracking import polygon_extraction
        import kwcoco
        classes = kwcoco.CategoryTree.from_mutex(channels_list)
        extractor = polygon_extraction.PolygonExtractor(
            _heatmaps_thwc,
            heatmap_time_intervals=image_dates,
            bounds=bounds,
            classes=classes,
            config={
                'scale_factor': 1,
                'thresh': config.thresh,
                'algo': config.new_algo,
                'viz_out_dir': extractor_viz_dir,
            })
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
        num_images  = len(missing_ix)
        if num_missing > 0:
            rich.print(f'[yellow]There are {num_missing} / {num_images} images that are missing {channels} channels')
        else:
            rich.print(f'[green]There are {num_missing} / {num_images} images that are missing {channels} channels')

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


def heatmaps_to_polys(heatmaps, track_bounds, heatmap_dates=None, config=None):
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
        >>> from geowatch.tasks.tracking.old_polygon_extraction import *  # NOQA
        >>> import kwimage
        >>> from kwutil import util_time
        >>> import numpy as np
        >>> from geowatch.tasks.tracking.old_polygon_extraction import PolygonExtractConfig  # NOQA
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
    if config.moving_window_size is None:
        # Hack: none meant window size is infinite here.
        groupxs = [np.arange(len(heatmaps))]
    else:
        groupxs = _compute_time_window(
            config.moving_window_size, num_frames=len(heatmaps),
            heatmap_dates=heatmap_dates)

    # initialize heatmaps and initial polygons on the first set of heatmaps
    n_steps = len(groupxs)
    xs_init = groupxs[0]
    h_init = heatmaps[xs_init]
    t_init = heatmap_unixtime_intervals[xs_init]

    # print(f'n_steps={n_steps}')
    # print('!!!!')
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
                prog.ensure_newline()
                h1 = heatmaps[idxs]
                t1 = heatmap_unixtime_intervals[idxs]
                print(h1.sum())

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
        >>> from geowatch.tasks.tracking.old_polygon_extraction import * # NOQA
        >>> from geowatch.tasks.tracking.old_polygon_extraction import _merge_polys  # NOQA
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
        from geowatch.tasks.tracking.old_polygon_extraction import * # NOQA
        from geowatch.tasks.tracking.old_polygon_extraction import _merge_polys  # NOQA

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
        from geowatch.utils import util_gis
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
