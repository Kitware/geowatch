"""
CommandLine:
    # Benchmark time sampling
    LINE_PROFILE=1 xdoctest -m geowatch.tasks.fusion.datamodules.spacetime_grid_builder __doc__:0


TODO:
    - [ ] The functions that take too many arguments should be refactored as
          object oriented code and use object attribute to store the extra
          data.

Example:
    >>> # xdoctest: +REQUIRES(env:LINE_PROFILE)
    >>> from geowatch.tasks.fusion.datamodules.spacetime_grid_builder import *  # NOQA
    >>> import ndsampler
    >>> import kwcoco
    >>> dset = kwcoco.CocoDataset.demo('vidshapes2', num_frames=100)
    >>> window_overlap = 0.0
    >>> time_dims = 1
    >>> window_dims = (32, 32)
    >>> keepbound = False
    >>> time_sampling = 'soft2+distribute'
    >>> sample_grid1 = SpacetimeGridBuilder(
    >>>     dset, time_dims, window_dims, window_overlap, use_cache=0,
    >>>     use_centered_positives=0,
    >>>     use_annot_info=0,
    >>>     time_sampling='contiguous').build()
    >>> boxes = [kwimage.Boxes.from_slice(target['space_slice'], clip=False).to_xywh() for target in sample_grid1['targets']]
    >>> all_boxes = kwimage.Boxes.concatenate(boxes)
    >>> assert np.all(all_boxes.height == window_dims[0])
    >>> assert np.all(all_boxes.width == window_dims[1])

Example:
    >>> # xdoctest: +REQUIRES(env:SMART_DATA_DVC_DPATH)
    >>> from geowatch.tasks.fusion.datamodules.spacetime_grid_builder import *  # NOQA
    >>> import geowatch
    >>> import kwcoco
    >>> dvc_dpath = geowatch.find_dvc_dpath(tags='phase3_data', hardware='ssd')
    >>> coco_fpath = dvc_dpath / 'Drop8-ARA-Median10GSD-V1/KR_R002/imganns-KR_R002-rawbands.kwcoco.zip'
    >>> coco_dset = kwcoco.CocoDataset(coco_fpath)
    >>> window_dims = 128
    >>> time_dims = 5
    >>> builder = SpacetimeGridBuilder(
    >>>     coco_dset,
    >>>     time_dims,
    >>>     window_dims,
    >>>     time_sampling='soft2+distribute',
    >>>     time_kernel='-1y,-8m,-2w,0,2w,8m,1y',
    >>>     keepbound=True,
    >>>     use_annot_info=1,
    >>>     use_grid_positives=1,
    >>>     use_centered_positives=True,
    >>>     respect_valid_regions=False,  # enabling this is slow
    >>>     use_cache=0
    >>> )
    >>> grid = builder.build()
    >>> # xdoctest: +REQUIRES(--show)
    >>> import kwplot
    >>> plt = kwplot.autoplt()
    >>> canvas = builder.visualize(max_vids=5, max_frames=5)
    >>> kwplot.imshow(canvas, doclf=1, fnum=2)
    >>> plt.gca().set_title(ub.codeblock(
        '''
        Sampled using larger scaled windows
        '''))
    >>> kwplot.show_if_requested()

Example:
    >>> # xdoctest: +REQUIRES(env:SMART_DATA_DVC_DPATH)
    >>> from geowatch.tasks.fusion.datamodules.spacetime_grid_builder import *  # NOQA
    >>> import geowatch
    >>> import kwcoco
    >>> dvc_dpath = geowatch.find_dvc_dpath(tags='phase3_data', hardware='hdd')
    >>> #coco_fpath = dvc_dpath / 'submodules/Drop8-ARA-Cropped2GSD-V1/KR_R002/imganns-KR_R002-rawbands.kwcoco.zip'
    >>> coco_fpath = dvc_dpath / 'submodules/Drop8-ARA-Cropped2GSD-V1/AE_R001/imganns-AE_R001-rawbands.kwcoco.zip'
    >>> coco_dset = kwcoco.CocoDataset(coco_fpath)
    >>> wh_lut = coco_dset.videos().lookup(['width', 'height'])
    >>> dsizes = list(zip(wh_lut['width'], wh_lut['height']))
    >>> window_dims = 128
    >>> builder = SpacetimeGridBuilder(
    >>>     dset=coco_dset,
    >>>     window_dims=window_dims,
    >>>     time_sampling='soft2+distribute',
    >>>     time_kernel='-1y,-8m,-2w,0,2w,8m,1y',
    >>>     dynamic_fixed_resolution=None,
    >>>     #dynamic_fixed_resolution={'max_winspace_full_dims': [1024, 1024]},
    >>>     keepbound=True,
    >>>     use_annot_info=1,
    >>>     use_grid_positives=1,
    >>>     use_centered_positives=True,
    >>>     respect_valid_regions=False,  # enabling this is slow
    >>>     use_cache=1
    >>> )
    >>> grid = builder.build()
    >>> # xdoctest: +REQUIRES(--show)
    >>> import kwplot
    >>> plt = kwplot.autoplt()
    >>> #infos = [ub.udict(v) & {'video_name', 'vidspace_full_dims'} for v in grid['vidid_to_meta'].values()]
    >>> #infos = sorted([d | {'area': np.prod(d['vidspace_full_dims'])} for d in infos], key=lambda d: d['area'])
    >>> # Show big and small clusters
    >>> #chosen = [infos[30], infos[-1]]
    >>> video_ids = None
    >>> video_ids = coco_dset.videos(names=['AE_R001_CLUSTER_073', 'AE_R001_CLUSTER_000']).ids
    >>> canvas = builder.visualize(max_vids=10, max_frames=1, video_ids=video_ids)
    >>> print(f'canvas.shape={canvas.shape}')
    >>> kwimage.imwrite('canvas.jpg', canvas)
    >>> kwplot.imshow(canvas, doclf=1, fnum=2)
    >>> relevant_params = ub.udict(builder.kw) & {'window_dims', 'window_overlap', 'time_kernel', 'use_grid_positives', 'use_centered_positives', 'dynamic_fixed_resolution'}
    >>> relevant_text = ub.urepr(relevant_params, concise=1, nobr=1, si=1, nl=0)
    >>> print(f'relevant_text = {ub.urepr(relevant_text, nl=1)}')
    >>> plt.gca().set_title(ub.codeblock(
        f'''
        {relevant_text}

        Places a red dot where there is a negative sample (at the center of the negative window)

        Places a blue dot where there is a positive sample

        Draws a yellow polygon over invalid spatial regions.
        '''))
    >>> kwplot.show_if_requested()
"""

import kwarray
import kwimage
import numpy as np
import ubelt as ub
import warnings
from geowatch.utils import kwcoco_extensions
from geowatch import heuristics

try:
    from line_profiler import profile
except ImportError:
    profile = ub.identity


# Bump this if the process for sampling the spacetime grid changes and old
# caches are no longer valid.
SPACETIME_CACHE_VERSION = 'spacetime_cache_v20'


class SpacetimeGridBuilder:
    """
    A helper class to help build a grid of spacetime windows for a coco
    dataset.

    The basic idea is that you will slide a spacetime window over the
    dataset and mark where positive andnegative "windows" are. We also put
    windows directly on positive annotations if desired.

    See the :func:`visualize_sample_grid` for a visualization of what the
    sample grid looks like.

    Example:
        >>> # xdoctest: +REQUIRES(env:DVC_DPATH)
        >>> import os
        >>> from geowatch.tasks.fusion.datamodules.spacetime_grid_builder import *  # NOQA
        >>> import geowatch
        >>> dvc_dpath = geowatch.find_dvc_dpath(tags='phase2_data', hardware='ssd')
        >>> coco_fpath = dvc_dpath / 'Drop6/data_train_split1.kwcoco.zip'
        >>> dset = geowatch.coerce_kwcoco(coco_fpath)
        >>> window_overlap = 0.0
        >>> window_dims = (128, 128)
        >>> time_dims = 2
        >>> sample_grid = SpacetimeGridBuilder(dset, time_dims, window_dims).build()
        >>> time_sampling = 'hard+distribute'
        >>> positives = list(ub.take(sample_grid['targets'], sample_grid['positives_indexes']))
    """
    def __init__(
        builder,
        dset,
        time_dims=None,
        window_dims=None,
        window_overlap=0.0,
        negative_classes=None,
        keepbound=True,
        include_sensors=None,
        exclude_sensors=None,
        select_images=None,
        select_videos=None,
        time_sampling='hard+distribute',
        time_span=None,
        time_kernel=None,
        use_annot_info=True,
        use_grid_positives=True,
        use_grid_negatives=True,
        use_centered_positives=True,
        window_space_scale=None,
        set_cover_algo=None,
        respect_valid_regions=True,
        dynamic_fixed_resolution=None,
        workers=0,
        use_cache=1
    ):
        """
        Args:
            dset (kwcoco.CocoDataset): coco dataset

            time_dims (int):
                number of time steps (Deprecated, use time_kernel)

            window_dims (Tuple[int, int] | str):
                spatial height, width of the sample region or a string code.

            window_overlap (float):
                fractional spatial overlap

            set_cover_algo (str | None):
                Algorithm used to find set cover of image IDs. Options are 'approx' (a greedy solution)
                or 'exact' (an ILP solution). If None is passed, set cover is not computed. The 'exact'
                method requires the packe pulp, available at PyPi.

            window_space_scale (str):
                Code indicating the scale at which to sample. If None uses the
                videospace GSD.

            use_grid_positives (bool):
                if False, will remove any grid sample that contains a positive
                example. In this case use_centered_positives should be True.

            use_grid_negatives (bool | str):
                Use non-annotation locations as negatives. Can be "cleared".

            use_centered_positives (bool):
                extend the grid with extra off-axis samples where positive
                annotations are centered. TODO: we could do a box packing
                to reduce the potential size here.

            use_annot_info (bool):
                if True allows using annotation information to get a better
                train-time grid. Should not be used at test-time.

            time_span (str):
                indicates the desired start/stop date range of the sample
                (Deprecated, use time_kernel)

            time_kernel (str):
                mutually exclusive with time span.

            time_sampling (str):
                code for specific temporal sampler: see
                temporal_sampling/sampler.py for more information.

            exclude_sensors (List[str]):
                A list of sensors to exclude from the grid

            negative_classes (List[str]):
                indicate class names that should not count towards a region being
                marked as positive. NOTE: This is old and deprecated.

            respect_valid_regions (bool):
                if True, only place windows in valid regions

            dynamic_fixed_resolution (None | Any):
                Experimental. Added in 0.17.1
                example value: {'max_winspace_full_dims': [1000, 1000]}

            workers (int): parallel workers

            use_cache (bool): uses a disk cache if True
        """
        builder.kw = ub.udict(locals()) - {'builder'}
        builder.__dict__.update(ub.udict(locals()) - {'builder'})

    def build(builder):
        r"""
        This is the main driver that builds the sample grid.

        Example:
            >>> # xdoctest: +REQUIRES(env:DVC_DPATH)
            >>> import os
            >>> from geowatch.tasks.fusion.datamodules.spacetime_grid_builder import *  # NOQA
            >>> import geowatch
            >>> dvc_dpath = geowatch.find_dvc_dpath(tags='phase2_data', hardware='ssd')
            >>> coco_fpath = dvc_dpath / 'Aligned-Drop3-TA1-2022-03-10/combo_LM_nowv_vali.kwcoco.json'
            >>> dset = kwcoco.CocoDataset(coco_fpath)
            >>> window_overlap = 0.5
            >>> time_dims = 2
            >>> window_dims = (128, 128)
            >>> keepbound = False
            >>> exclude_sensors = None
            >>> set_cover_algo = 'approx'
            >>> sample_grid = SpacetimeGridBuilder(dset, time_dims, window_dims, window_overlap, set_cover_algo=set_cover_algo).build()
            >>> time_sampling = 'hard+distribute'
            >>> positives = list(ub.take(sample_grid['targets'], sample_grid['positives_indexes']))

        Example:
            >>> from geowatch.tasks.fusion.datamodules.spacetime_grid_builder import *  # NOQA
            >>> import ndsampler
            >>> import kwcoco
            >>> dset = kwcoco.CocoDataset.demo('vidshapes2-multispectral', num_frames=30)
            >>> window_overlap = 0.0
            >>> time_dims = 3
            >>> window_dims = (256, 256)
            >>> keepbound = False
            >>> time_sampling = 'soft2+distribute'
            >>> sample_grid1 = SpacetimeGridBuilder(
            >>>     dset, time_dims, window_dims, window_overlap,
            >>>     time_sampling='soft2+distribute').build()
            >>> boxes = [kwimage.Boxes.from_slice(target['space_slice'], clip=False).to_xywh() for target in sample_grid1['targets']]
            >>> all_boxes = kwimage.Boxes.concatenate(boxes)
            >>> assert np.all(all_boxes.height == window_dims[0])
            >>> assert np.all(all_boxes.width == window_dims[1])

        Example:
            >>> from geowatch.tasks.fusion.datamodules.spacetime_grid_builder import *  # NOQA
            >>> import ndsampler
            >>> import kwcoco
            >>> dset = kwcoco.CocoDataset.demo('vidshapes2-multispectral', num_frames=30)
            >>> window_overlap = 0.0
            >>> time_dims = 3
            >>> window_dims = (32, 32)
            >>> keepbound = False
            >>> time_sampling = 'soft2+distribute'
            >>> sample_grid1 = SpacetimeGridBuilder(
            >>>     dset, time_dims, window_dims, window_overlap, exclude_sensors='Foo',
            >>>     time_sampling='soft2+distribute').build()
            >>> sample_grid2 = SpacetimeGridBuilder(
            >>>     dset, time_dims, window_dims, window_overlap,
            >>>     time_sampling='contiguous+pairwise').build()

            ub.peek(sample_grid1['vidid_to_time_sampler'].values()).show_summary(fnum=1)
            ub.peek(sample_grid2['vidid_to_time_sampler'].values()).show_summary(fnum=2)
            _ = xdev.profile_now(sample_video_spacetime_targets)(dset, time_dims, window_dims, window_overlap)

            import xdev
            globals().update(xdev.get_func_kwargs(sample_video_spacetime_targets))
        """
        builder.sample_grid = _build_grid(builder)
        return builder.sample_grid

    def visualize(builder, max_vids=2, max_frames=6, video_ids=None):
        r"""
        Debug visualization for sampling grid

        Draws multiple frames.

        Places a red dot where there is a negative sample (at the center of the negative window)

        Places a blue dot where there is a positive sample

        Draws a yellow polygon over invalid spatial regions.

        TODO:
            - [x] Make this the `SpacetimeGridBuilder.visualize` method

        Notes:
            * Dots are more intense when there are more temporal coverage of that dot.

            * Dots are placed on the center of the window.  They do not indicate its extent.

            * Dots are blue if they overlap any annotation in their temporal region
              so they may visually be near an annotation.

        Args:
            max_vids (int): maximum videos that can be shown
            max_frames (int): maximum frame from each video to draw
            video_ids (List[int] | None): if specified, show these videos

        Example:
            >>> from geowatch.tasks.fusion.datamodules.spacetime_grid_builder import *  # NOQA
            >>> from geowatch.demo.smart_kwcoco_demodata import demo_kwcoco_multisensor
            >>> dset = coco_dset = demo_kwcoco_multisensor(num_frames=3, dates=True, geodata=True, heatmap=True, rng=10)
            >>> window_overlap = 0.2
            >>> window_dims = (64, 64)
            >>> time_sampling = 'soft2+distribute'
            >>> use_centered_positives = True
            >>> use_grid_positives = 1
            >>> time_dims = 3
            >>> time_kernel = '-1d,0d,1d'
            >>> builder = SpacetimeGridBuilder(
            >>>     dset=dset, time_dims=time_dims, time_kernel=time_kernel, window_dims=window_dims,
            >>>     window_overlap=window_overlap, time_sampling=time_sampling,
            >>>     use_grid_positives=use_grid_positives,
            >>>     use_centered_positives=use_centered_positives)
            >>> grid = builder.build()
            >>> # xdoctest: +REQUIRES(--show)
            >>> import kwplot
            >>> plt = kwplot.autoplt()
            >>> canvas = builder.visualize(max_vids=1, max_frames=3)
            >>> kwplot.imshow(canvas, doclf=1)
            >>> relevant_params = ub.udict(builder.kw) & {'window_dims', 'window_overlap', 'time_dims', 'use_grid_positives', 'use_centered_positives'}
            >>> relevant_text = ub.urepr(relevant_params, concise=1, nobr=1, si=1, nl=0)
            >>> print(f'relevant_text = {ub.urepr(relevant_text, nl=1)}')
            >>> plt.gca().set_title(ub.codeblock(
                f'''
                {relevant_text}

                Places a red dot where there is a negative sample (at the center of the negative window)

                Places a blue dot where there is a positive sample

                Draws a yellow polygon over invalid spatial regions.
                '''))
            >>> kwplot.show_if_requested()
            >>> #
            >>> # Now demo this same grid, but where we are sampling at a different resolution
            >>> window_space_scale = 0.3
            >>> builder2 = SpacetimeGridBuilder(
            >>>     dset=dset, time_dims=time_dims, window_dims=window_dims, window_overlap=window_overlap,
            >>>     time_sampling=time_sampling, use_grid_positives=use_grid_positives,
            >>>     use_centered_positives=use_centered_positives,
            >>>     window_space_scale=window_space_scale)
            >>> sample_grid2 = builder2.build()
            >>> # xdoctest: +REQUIRES(--show)
            >>> import kwplot
            >>> plt = kwplot.autoplt()
            >>> canvas = builder2.visualize(max_vids=1, max_frames=3)
            >>> kwplot.imshow(canvas, doclf=1, fnum=2)
            >>> plt.gca().set_title(ub.codeblock(
                '''
                Sampled using larger scaled windows
                '''))
            >>> kwplot.show_if_requested()

        Example:
            >>> # xdoctest: +REQUIRES(env:DVC_DPATH)
            >>> from geowatch.tasks.fusion.datamodules.spacetime_grid_builder import *  # NOQA
            >>> import geowatch
            >>> # dset = coco_dset = demo_kwcoco_multisensor(dates=True, geodata=True, heatmap=True)
            >>> dvc_dpath = geowatch.find_dvc_dpath(hardware='ssd', tags='phase2_data')
            >>> #coco_fpath = dvc_dpath / 'Drop2-Aligned-TA1-2022-02-15/combo_DILM_train.kwcoco.json'
            >>> coco_fpath = dvc_dpath / 'Drop6/data_vali_split1.kwcoco.zip'
            >>> big_dset = kwcoco.CocoDataset(coco_fpath)
            >>> vid_gids = big_dset.videos(names=['KR_R002']).images.lookup('id')[0]
            >>> idxs = np.linspace(0, len(vid_gids) - 1, 12).round().astype(int)
            >>> vid_gids = list(ub.take(vid_gids, idxs))
            >>> dset = big_dset.subset(vid_gids)
            >>> window_overlap = 0.0
            >>> window_dims = (3, 256, 256)
            >>> keepbound = False
            >>> time_sampling = 'soft2+distribute'
            >>> use_centered_positives = 0
            >>> use_grid_positives = 1
            >>> window_space_scale = '30GSD'
            >>> sample_grid = sample_video_spacetime_targets(
            >>>     dset, window_dims, window_overlap, time_sampling=time_sampling,
            >>>     use_grid_positives=True,
            >>>     use_centered_positives=use_centered_positives,
            >>>     window_space_scale=window_space_scale)
            >>> print(list(ub.unique([t['space_slice'] for t in sample_grid['targets']], key=ub.hash_data)))
            >>> # xdoctest: +REQUIRES(--show)
            >>> import kwplot
            >>> plt = kwplot.autoplt()
            >>> canvas = _visualize_sample_grid(dset, sample_grid, max_vids=1, max_frames=12)
            >>> kwplot.imshow(canvas, doclf=1)
            >>> kwplot.show_if_requested()
            >>> plt.gca().set_title(ub.codeblock(
            >>>     f'''
            >>>     Sample window {window_dims} @ {window_space_scale}
            >>>     '''))
        """
        dset = builder.kw['dset']
        sample_grid = builder.sample_grid
        return _visualize_sample_grid(dset, sample_grid, max_vids=max_vids,
                                      max_frames=max_frames,
                                      video_ids=video_ids)


@profile
def _build_grid(builder):
    dset = builder.dset
    time_dims = builder.time_dims
    window_dims = builder.window_dims
    window_overlap = builder.window_overlap
    negative_classes = builder.negative_classes
    keepbound = builder.keepbound
    include_sensors = builder.include_sensors
    exclude_sensors = builder.exclude_sensors
    select_images = builder.select_images
    select_videos = builder.select_videos
    time_sampling = builder.time_sampling
    time_span = builder.time_span
    time_kernel = builder.time_kernel
    use_annot_info = builder.use_annot_info
    use_grid_positives = builder.use_grid_positives
    use_grid_negatives = builder.use_grid_negatives
    use_centered_positives = builder.use_centered_positives
    window_space_scale = builder.window_space_scale
    set_cover_algo = builder.set_cover_algo
    respect_valid_regions = builder.respect_valid_regions
    dynamic_fixed_resolution = builder.dynamic_fixed_resolution

    workers = builder.workers
    use_cache = builder.use_cache

    if isinstance(window_dims, int):
        window_dims = (window_dims, window_dims)

    if window_dims is not None and isinstance(window_dims, tuple) and len(window_dims) == 3:
        ub.schedule_deprecation(
            'geowatch', 'window_dims', 'argument in spacetime_grid_builder',
            migration='window_dims no longer supports T,H,W specification, use time_times=T, window_dims=(H, W)',
            deprecate='now')
        winspace_time_dims = window_dims[0]
        winspace_space_dims = window_dims[1:3]
        assert time_dims is None, 'cant do old style and new style'
    else:
        winspace_time_dims = time_dims
        winspace_space_dims = window_dims

    if isinstance(time_kernel, str) and time_kernel.lower() == 'none':
        time_kernel = None

    if isinstance(time_span, str) and time_span.lower() == 'none':
        time_span = None

    if time_kernel is not None and time_span is not None:
        raise ValueError('time_span and time_kernel are mutually exclusive')

    # from ndsampler import isect_indexer
    # _isect_index = isect_indexer.FrameIntersectionIndex.from_coco(dset)

    # FIXME: refactor to get rid of this hardcoded nonsense
    parts = set(time_sampling.split('+'))
    affinity_type_parts = parts & {
        'hard', 'hardish', 'contiguous', 'soft2', 'soft', 'hardish2',
        'hardish3', 'soft2-contiguous-hardish3', 'uniform',
        'uniform-soft2-contiguous-hardish3',
        'uniform-soft3-contiguous-hardish3',
        'uniform-soft5-soft4-contiguous',
        'soft4', 'soft5',
    }
    update_rule_parts = parts & {'distribute', 'pairwise'}
    unknown = (parts - affinity_type_parts) - update_rule_parts
    if unknown:
        raise ValueError('Unknown time-sampling parts: {}'.format(unknown))

    affinity_type = '+'.join(list(affinity_type_parts))
    update_rule = '+'.join(list(update_rule_parts))
    if not update_rule:
        update_rule = 'distribute'

    if negative_classes is not None:
        if 0:
            warnings.warn(ub.paragraph(
                '''
                Grid sampler no longer handles negative classes. Instead it
                provides the user the information to balance positive/negatives
                after the fact.
                '''))

    dset_hashid = dset._cached_hashid()

    # Intersection over smaller area wrt to window vs valid regions.
    refine_iosa_thresh = 0.2  # parametarize?

    # TODO: we can disable respect valid regions here and then just do it on
    # the fly in the dataloader, but it is unclear which is more efficient.
    if winspace_time_dims is None:
        if time_kernel is not None:
            from geowatch.tasks.fusion.datamodules.temporal_sampling.utils import coerce_multi_time_kernel
            _multi_time_kernel = coerce_multi_time_kernel(time_kernel)
            if len(_multi_time_kernel) == 1:
                winspace_time_dims = len(_multi_time_kernel[0])
            print(f'INFER: winspace_time_dims = {ub.urepr(winspace_time_dims, nl=1)}')
            print(f'time_kernel = {ub.urepr(time_kernel, nl=1)}')
        else:
            raise ValueError('time dims required if time_kernel not given')

    print('🛝 🌐⌚🪟 Gather Sliding Spacetime Windows')
    print(f'winspace_time_dims={winspace_time_dims}')
    print(f'winspace_space_dims={winspace_space_dims}')
    depends = [
        dset_hashid,
        affinity_type,
        update_rule,
        winspace_space_dims,
        winspace_time_dims,
        window_overlap,
        window_space_scale,
        keepbound,
        include_sensors,
        exclude_sensors,
        select_videos,
        select_images,
        affinity_type,
        time_span, time_kernel, use_annot_info,
        set_cover_algo,
        use_grid_positives,
        use_grid_negatives,
        use_centered_positives,
        refine_iosa_thresh,
        respect_valid_regions,
        dynamic_fixed_resolution,
        SPACETIME_CACHE_VERSION,
        negative_classes
    ]
    # Higher level cacher (not sure if adding this secondary level of caching
    # is faster or not).
    dset_name = ub.Path(dset.fpath).name
    cache_dpath = ub.Path.appdir('geowatch', 'grid_cache').ensuredir()
    cacher = ub.Cacher('sample_grid-dataset-cache_' + dset_name,
                       dpath=cache_dpath, depends=depends, enabled=use_cache,
                       verbose=4)
    sample_grid = cacher.tryload()
    if sample_grid is None:
        print('🖥️🛝 🌐⌚🪟 Compute Sliding Spacetime Windows')

        # Only build a grid over the selected images / videos
        selected_gids = kwcoco_extensions.filter_image_ids(
            dset, include_sensors=include_sensors,
            exclude_sensors=exclude_sensors,
            select_videos=select_videos,
            select_images=select_images,
        )

        selected_images = dset.images(selected_gids)
        if hasattr(selected_images._dset, '_column_lookup'):
            print(f'Query {len(selected_images)} sql rows, may take time')
            # self = selected_images
            # key = 'video_id'
            # with ub.Timer('sql query'):
            selected_vidid_per_gid = selected_images.lookup('video_id', default=None)
            # selected_vidid_per_gid = self._dset._column_lookup(
            #     tablename=self._key, key=key, rowids=self._ids)
            # f1dfe5897bbf49a09a9e0a4e63809dc95248d2d3b3cbd993f0d8b399dba60d746dc537a3bde46a8a066b227efafbbb7b38dc254c7bac1bae9ff09f4d64976956
        else:
            selected_vidid_per_gid = selected_images.lookup('video_id', default=None)

        selected_vidid_per_gid = selected_images.lookup('video_id', default=None)
        selected_vidid_to_gids = ub.group_items(selected_gids, selected_vidid_per_gid)

        loose_gids = selected_vidid_to_gids.pop(None, [])

        # Hack: pretend loose images are videos of length 1
        for gid in loose_gids:
            selected_vidid_to_gids[f'fakevid_loose_{gid}'] = [gid]

        # Given an video
        all_vid_ids = sorted(set(selected_vidid_to_gids.keys()))

        from kwutil import util_parallel
        workers = util_parallel.coerce_num_workers(workers)
        workers = min(len(all_vid_ids), workers)
        if workers == 1:
            workers = 0
        jobs = ub.JobPool(mode='process', max_workers=workers)

        # TODO: Reducing the information that needs to be passed to each worker
        # would help improve speed here. The dset itself is the biggest offender.
        verbose = 1 if workers == 0 else 0
        for video_id in ub.ProgIter(all_vid_ids, desc='Submit sample video regions'):
            # Ensure the ordering of image ids is valid
            video_gids = selected_vidid_to_gids[video_id]
            if len(video_gids) > 1:
                sortx = ub.argsort(dset.images(video_gids).lookup('frame_index'))
                video_gids = list(ub.take(video_gids, sortx))

            job = jobs.submit(
                _sample_single_video_spacetime_targets, dset, dset_hashid,
                video_id, video_gids, winspace_time_dims, winspace_space_dims,
                window_overlap, keepbound,
                affinity_type, update_rule, time_span, time_kernel, use_annot_info,
                use_grid_positives, use_grid_negatives, use_centered_positives, window_space_scale,
                set_cover_algo, use_cache, respect_valid_regions,
                refine_iosa_thresh, dynamic_fixed_resolution, negative_classes, verbose)
            job.video_id = video_id

        targets = []
        positive_idxs = []
        negative_idxs = []
        vidid_to_time_sampler = {}
        vidid_to_valid_gids = {}
        vidid_to_meta = ub.ddict(dict)
        for job in jobs.as_completed(desc='Collect region sample grids',
                                     progkw=dict(verbose=3)):
            video_id = job.video_id
            _cached, meta, time_sampler, video_gids = job.result()
            offset = len(targets)
            targets.extend(_cached['video_targets'])
            positive_idxs.extend([idx + offset for idx in _cached['video_positive_idxs']])
            negative_idxs.extend([idx + offset for idx in _cached['video_negative_idxs']])
            vidid_to_time_sampler[video_id] = time_sampler
            vidid_to_valid_gids[video_id] = video_gids
            vidid_to_meta[video_id] = meta

        print('Found {} targets'.format(len(targets)))
        if use_annot_info:
            print('Found {} positives'.format(len(positive_idxs)))
            print('Found {} negatives'.format(len(negative_idxs)))

        sample_grid = {
            'positives_indexes': positive_idxs,
            'negatives_indexes': negative_idxs,
            'targets': targets,
            'vidid_to_valid_gids': vidid_to_valid_gids,
            'vidid_to_time_sampler': vidid_to_time_sampler,
            'vidid_to_meta': vidid_to_meta,
        }
        cacher.save(sample_grid)

    SHOW_INFO = 1
    if SHOW_INFO:
        # Using urepr is quite slow here for large number of targets
        # reduce to a subset of video ids to reduce the time this takes
        from kwutil.slugify_ext import smart_truncate
        from geowatch.utils.util_kwutil import distributed_subitems
        vidid_to_meta = sample_grid['vidid_to_meta']
        subvidid_to_meta = distributed_subitems(vidid_to_meta, 6)
        _text = ub.urepr(subvidid_to_meta, nl=-1)
        # _text = repr(subvidid_to_meta)
        _trunc_text = smart_truncate(_text, max_length=1600, head='\n~TRUNCATED...', tail='\n...~')
        print(f'vidid_to_meta = {_trunc_text}')
    return sample_grid


class ImagePropertyCacher:
    """
    Helper class for caching image property lookups
    """
    def __init__(image_props, dset):
        image_props.dset = dset

    @ub.memoize_method
    def get_warp_vid_from_img(image_props, gid):
        """
        Abstract the transform to bring us into whatever the internal "window
        space" is. Depends on whatever "window_scale" is computed as.
        """
        coco_img = image_props.dset.coco_image(gid)
        warp_vid_from_img = coco_img.warp_vid_from_img
        return warp_vid_from_img

    @ub.memoize_method
    def get_image_valid_region_in_vidspace(image_props, gid):
        coco_poly = image_props.dset.index.imgs[gid].get('valid_region', None)
        if not coco_poly:
            sh_poly_vid = None
        else:
            warp_vid_from_img = image_props.get_warp_vid_from_img(gid)
            kw_poly_img = kwimage.MultiPolygon.coerce(coco_poly)
            kw_poly_vid = kw_poly_img.warp(warp_vid_from_img)
            sh_poly_vid = kw_poly_vid.to_shapely()
        return sh_poly_vid


@profile
def _sample_single_video_spacetime_targets(
        dset, dset_hashid, video_id, video_gids, winspace_time_dims, winspace_space_dims,
        window_overlap,
        keepbound, affinity_type, update_rule, time_span, time_kernel,
        use_annot_info, use_grid_positives, use_grid_negatives, use_centered_positives,
        window_space_scale, set_cover_algo, use_cache, respect_valid_regions,
        refine_iosa_thresh, dynamic_fixed_resolution, negative_classes, verbose):
    """
    Do this for a single video so we can parallelize.

    Called as the main worker function in
    :func:`sample_video_spacetime_targets`.

    Note this introduces a new temporary space:
        window-space, which is a scaled version of video-space.
        This is only used internally. The final targets are returned in video
        space.
    """
    from geowatch.tasks.fusion.datamodules import temporal_sampling as tsm  # NOQA
    from geowatch.tasks.fusion.datamodules import data_utils
    import kwutil

    # It is important that keepbound is True at test time, otherwise we may not
    # predict on the bottom right of the image.
    keepbound = True

    try:
        video_info = dset.index.videos[video_id]
    except KeyError:
        # Probably a fake video
        assert len(video_gids) == 1
        video_info = {
            'name': video_id,
            'width': dset.index.imgs[video_gids[0]]['width'],
            'height': dset.index.imgs[video_gids[0]]['height'],
        }

    try:
        video_name = video_info['name']
        vidspace_video_height = video_info['height']
        vidspace_video_width = video_info['width']
    except KeyError as ex:
        from kwutil.util_exception import add_exception_note
        raise add_exception_note(ex, 'KWCoco videos must contain: name, height, and width')

    vidspace_full_dims = [vidspace_video_height, vidspace_video_width]

    # Create a box to represent the "window-space" extent, and determine how we
    # are going to slide a window over it.
    vidspace_gsd = video_info.get('target_gsd', None)
    resolved_scale = data_utils.resolve_scale_request(
        request=window_space_scale, data_gsd=vidspace_gsd)
    window_scale = resolved_scale['scale']
    # Convert winspace to vidspace and use that for the rest of the function
    winspace_full_dims = np.ceil(np.array(vidspace_full_dims) * window_scale)

    # If the grid is going to force resolutions, we populate this
    forced_resolutions = {}

    # Hack: modify the window resolution if the dynamic fixed resolution
    # behavior is on.
    dynamic_fixed_resolution = kwutil.util_yaml.Yaml.coerce(dynamic_fixed_resolution)
    """
    dynamic_fixed_resolution = {'max_winspace_full_dims': [1000, 1000]}
    """
    if dynamic_fixed_resolution is not None:
        _debug = 1
        # if _debug:
        #     print(f'dynamic_fixed_resolution = {ub.urepr(dynamic_fixed_resolution, nl=1)}')
        if isinstance(dynamic_fixed_resolution, dict):
            max_winspace_full_dims = dynamic_fixed_resolution.get('max_winspace_full_dims', None)
            if max_winspace_full_dims is not None:
                max_winspace_full_dims = np.array(max_winspace_full_dims)
                dynamic_factor = (max_winspace_full_dims / winspace_full_dims).min()
                if dynamic_factor < 1.0:
                    if _debug:
                        print('---')
                        print('Handle dynamic resolution adjustment')
                        print(f'before: winspace_full_dims={winspace_full_dims}')
                        print(f'before: resolved_scale = {ub.urepr(resolved_scale, nl=1)}')
                        print(f'window_scale = {ub.urepr(window_scale, nl=1)}')
                    # Adjust the scale
                    window_scale = resolved_scale['scale'] * dynamic_factor
                    resolved_scale = data_utils.resolve_scale_request(
                        request=window_scale, data_gsd=vidspace_gsd)
                    # Convert winspace to vidspace and use that for the rest of the function
                    winspace_full_dims = np.ceil(np.array(vidspace_full_dims) * window_scale)
                    # Note: these names should change to input_resolution / output_resolution at some point
                    forced_resolutions['input_space_scale'] = str(resolved_scale['gsd']) + 'GSD'
                    forced_resolutions['output_space_scale'] = str(resolved_scale['gsd']) + 'GSD'
                    if _debug:
                        print(f'dynamic_factor = {ub.urepr(dynamic_factor, nl=1)}')
                        print(f'max_winspace_full_dims = {ub.urepr(max_winspace_full_dims, nl=1)}')
                        print(f'after: winspace_full_dims={winspace_full_dims}')
                        print(f'after: resolved_scale = {ub.urepr(resolved_scale, nl=1)}')
                        print('---')
                # raise NotImplementedError
            else:
                raise NotImplementedError
        else:
            raise NotImplementedError('todo')
        # TODO: handle dynamic resolution requests here.
        ...

    vidspace_time_dims = winspace_time_dims
    time_sampler = tsm.MultiTimeWindowSampler.from_coco_video(
        dset, video_id, gids=video_gids, time_window=vidspace_time_dims,
        affinity_type=affinity_type, update_rule=update_rule, name=video_name,
        time_kernel=time_kernel, time_span=time_span
    )

    gid_arr = np.array(video_gids)
    time_sampler.video_gids = gid_arr
    time_sampler.gid_to_index = ub.udict(enumerate(time_sampler.video_gids)).invert()
    time_sampler.deterministic = True

    if isinstance(winspace_space_dims, str):
        if winspace_space_dims == 'full':
            vidspace_window_dims = vidspace_full_dims
        else:
            raise KeyError(winspace_space_dims)
    else:
        try:
            winspace_window_height, winspace_window_width = winspace_space_dims
        except Exception:
            print(f'winspace_space_dims = {ub.urepr(winspace_space_dims, nl=1)}')
            raise

        winspace_window_box = kwimage.Boxes([
            [0, 0, winspace_window_width, winspace_window_height]], 'xywh')
        # The window is scaled inversely to the data
        # NOTE: This window size can be too big wrt what is actually requested.
        vidspace_window_box = winspace_window_box.scale(1 / window_scale).quantize()
        vidspace_window_height = vidspace_window_box.height.ravel()[0]
        vidspace_window_width = vidspace_window_box.width.ravel()[0]
        vidspace_window_dims = (vidspace_window_height, vidspace_window_width)

    if use_grid_negatives == 'cleared':
        use_grid_negatives = video_info.get('cleared', False)

    depends = [
        dset_hashid,
        affinity_type,
        update_rule,
        video_name,
        gid_arr,
        vidspace_window_dims, window_overlap,
        keepbound,
        affinity_type,
        time_span, use_annot_info,
        use_grid_positives,
        use_grid_negatives,
        use_centered_positives,
        refine_iosa_thresh,
        respect_valid_regions,
        set_cover_algo,
        SPACETIME_CACHE_VERSION,
        negative_classes
    ]

    # Only use the cache if this is probably going to be a slow operation.
    # Otherwise punt, the entire thing gets cached at the end, so the extra
    # disk IO may just slow things down.
    rough_num_windows = np.prod(np.array(vidspace_full_dims) / np.array(vidspace_window_dims))
    rough_num_cells = len(gid_arr) * rough_num_windows
    probably_slow = rough_num_cells > (16 * 30)

    cache_dpath = ub.Path.appdir('geowatch', 'grid_cache').ensuredir()
    safe_video_name = kwutil.util_path.sanitize_path_name(video_name)
    cacher = ub.Cacher('sliding-window-cache-' + safe_video_name,
                       dpath=cache_dpath, depends=depends,
                       enabled=(use_cache and probably_slow))
    _cached = cacher.tryload()
    if _cached is None:

        # Helper that caches repeatedly access values in the next parts
        image_props = ImagePropertyCacher(dset)

        video_targets = []
        video_positive_idxs = []
        video_negative_idxs = []
        # For each frame, deterministically compute an initial list of which
        # supporting frames we will look at when making a prediction for the
        # "main" frame. Initially this is only based on temporal metadata.  We
        # may modify this later depending on spatial properties.
        main_idx_to_sample_idxs = {
            main_idx: time_sampler.sample(main_idx)
            for main_idx in time_sampler.indexes
        }
        # print('Time index hash: ' + ub.hash_data(time_sampler.indexes))
        # print('Time affinity hash: ' + ub.hash_data(time_sampler.affinity))
        # print('Time time hash: ' + ub.hash_data(time_sampler.unixtimes))
        # print('Time sensor hash: ' + ub.hash_data(time_sampler.sensors))
        # print('First sample hash: ' + ub.hash_data(main_idx_to_sample_idxs))

        main_idx_to_gids = {
            main_idx: list(ub.take(video_gids, sample_idxs))
            for main_idx, sample_idxs in main_idx_to_sample_idxs.items()
        }

        if use_annot_info:
            qtree, tid_to_infos, loose_aid_to_infos = _build_vidspace_track_qtree(
                dset, video_gids, vidspace_video_width, vidspace_video_height,
                image_props)
        else:
            qtree = None
            tid_to_infos = None
            loose_aid_to_infos = None

        # Structured grid sampling:
        # Consider each spatial location in the sliding window as a candidate
        # sample and determine what label we want to give it.
        needs_sliding_window = (
            use_grid_negatives or
            use_grid_positives or
            use_centered_positives or
            (not use_annot_info)
        )
        if needs_sliding_window:
            # Do a spatial sliding window (in video space) and handle all the
            # temporal stuff for that window in the internal function.
            slider = kwarray.SlidingWindow(vidspace_full_dims,
                                           vidspace_window_dims,
                                           overlap=window_overlap,
                                           keepbound=keepbound,
                                           allow_overshoot=True)
            slices = list(slider)
            num_cells = len(slices) * len(video_gids)
            probably_slow = num_cells > (16 * 30)

            for vidspace_region in ub.ProgIter(slices,
                                               desc='Build targets over sliding window',
                                               enabled=probably_slow,
                                               verbose=verbose * probably_slow):

                new_targets = _build_targets_in_spatial_region(
                    dset, video_id, vidspace_region, use_annot_info, qtree,
                    main_idx_to_gids, refine_iosa_thresh, time_sampler,
                    image_props, respect_valid_regions,
                    set_cover_algo, negative_classes)
                new_targets = list(new_targets)

                if not use_annot_info:
                    video_targets.extend(new_targets)
                else:
                    for target in new_targets:
                        label = target['label']
                        if label == 'positive_grid':
                            if not use_grid_positives:
                                continue
                            video_positive_idxs.append(len(video_targets))
                        elif label == 'negative_grid':
                            if not use_grid_negatives:
                                continue
                            video_negative_idxs.append(len(video_targets))
                        video_targets.append(target)

        # Unstructured grid sampling:
        # Consider explicit samples placed around annotations.
        if use_centered_positives and use_annot_info:
            # FIXME: This code is too slow
            # in addition to the sliding window sample, add positive samples
            # centered around each annotation.
            track_infos = list(tid_to_infos.items())
            for tid, track_info_group in ub.ProgIter(track_infos,
                                                     desc='Build targets around centered track annotations',
                                                     enabled=len(track_infos) > 4 and probably_slow,
                                                     verbose=verbose * (len(track_infos) > 4 and probably_slow)):

                new_targets = _build_targets_around_track(
                        dset, use_annot_info, qtree, video_id, track_info_group,
                        video_gids, vidspace_window_dims,
                        time_sampler)
                new_targets = list(new_targets)
                for target in new_targets:
                    video_positive_idxs.append(len(video_targets))
                    video_targets.append(target)

            # For annotations without track information, consider them as
            # tracks of length 1
            loose_annot_infos = list(loose_aid_to_infos.items())
            for aid, info in ub.ProgIter(loose_annot_infos,
                                         desc='Build targets around centered trackless annots',
                                         enabled=len(loose_aid_to_infos) > 4 and probably_slow,
                                         verbose=verbose * (len(loose_aid_to_infos) > 4 and probably_slow)):
                track_info_group = [info]
                new_targets = _build_targets_around_track(
                        dset, use_annot_info, qtree, video_id, track_info_group,
                        video_gids, vidspace_window_dims,
                        time_sampler)
                new_targets = list(new_targets)
                for target in new_targets:
                    video_positive_idxs.append(len(video_targets))
                    video_targets.append(target)

        if forced_resolutions:
            for target in video_targets:
                target.update(forced_resolutions)

        _cached = {
            'video_targets': video_targets,
            'video_positive_idxs': video_positive_idxs,
            'video_negative_idxs': video_negative_idxs,
        }
        cacher.save(_cached)

    # Disable determenism in the returned sampler
    time_sampler.deterministic = False
    meta = {
        'video_name': video_name,
        'resolved_scale': resolved_scale,
        'vidspace_window_space_dims': vidspace_window_dims,
        'winspace_window_space_dims': winspace_space_dims,
        'vidspace_full_dims': vidspace_full_dims,
        'winspace_full_dims': winspace_full_dims,
        'num_available_frames': len(time_sampler.indexes),
        'num_samples': len(_cached['video_targets']),
        # TODO: if we are not classifying things as positive / negative
        # then we should not include this to prevent confusion
        'num_pos_samples': len(_cached['video_positive_idxs']),
        'num_neg_samples': len(_cached['video_negative_idxs']),
    }
    return _cached, meta, time_sampler, video_gids


@profile
def _build_targets_around_track(dset, use_annot_info, qtree, video_id, track_info_group, video_gids,
                                vidspace_window_dims, time_sampler):
    """
    Given information about a track, build targets to ensure the network trains
    with it.

    Args:
        track_info_group (List[Dict]):
            each row contains gid, aid, cid, tid, frame-index for the track of
            interest.
    """
    window_height, window_width = vidspace_window_dims
    # For every frame in the track
    for info in track_info_group:
        main_gid = info['gid']
        vidspace_ann_box = kwimage.Box.coerce(info['vidspace_box'], format='tlbr')
        vidspace_ann_box = vidspace_ann_box.quantize()
        vidspace_ann_box = vidspace_ann_box.resize(
            width=window_width, height=window_height, about='cxy')
        #  FIXME, this code is ugly
        # TODO: we could make frames where the phase transitions
        # more likely here.
        _hack_main_idx = np.where(time_sampler.video_gids == main_gid)[0][0]
        _sample_idxs = time_sampler.sample(_hack_main_idx)
        sample_gids = list(ub.take(video_gids, _sample_idxs))
        _hack = {_hack_main_idx: sample_gids}
        _hack2 = _hack
        if _hack2:
            gids = _hack2[_hack_main_idx]
            label = 'positive_center'
            annot_info = None
            vidspace_region = vidspace_ann_box.to_slice()

            # Find all annotations that pass through this spatial region
            vidspace_box = kwimage.Box.from_slice(vidspace_region).to_ltrb()
            query = vidspace_box.data
            isect_aids = list(qtree.intersect(query))
            isect_gids = set(dset.annots(isect_aids).lookup('image_id'))
            isect_aids_catnames = dset.annots(isect_aids).category_names
            aid_to_catname = dict(zip(isect_aids, isect_aids_catnames))
            gid_to_aids = {x: dset.index.gid_to_aids[x] & set(isect_aids) for x in isect_gids}
            gid_to_catnames = {k: list(ub.take(aid_to_catname, v)) for k, v in gid_to_aids.items()}

            if use_annot_info:
                annot_info = {
                    'isect_aids': isect_aids,
                    'isect_aids_catnames': isect_aids_catnames,
                    'gid_to_aids': ub.dict_subset(gid_to_aids, gids, default=[]),
                    'gid_to_catnames': ub.dict_subset(gid_to_catnames, gids, default=[]),
                    'main_gid_catnames': ub.dict_hist(list(ub.take(gid_to_catnames, [main_gid], default=[]))[0]),
                }

            target = {
                'main_idx': _hack_main_idx,
                'video_id': video_id,
                'gids': gids,
                'main_gid': main_gid,
                'space_slice': vidspace_region,
                'label': label,
                'resampled': -1,
                'annot_info': annot_info,
            }
            yield target


@profile
def _build_targets_in_spatial_region(dset, video_id, vidspace_region,
                                     use_annot_info, qtree, main_idx_to_gids,
                                     refine_iosa_thresh, time_sampler,
                                     image_props, respect_valid_regions,
                                     set_cover_algo, negative_classes):
    """
    Called for each spatial grid in the sliding window.
    This adds multiple targets

    Called as part of :func:`_sample_single_video_spacetime_targets`.
    """
    from geowatch.tasks.fusion.datamodules import temporal_sampling as tsm  # NOQA
    y_sl, x_sl = vidspace_region
    vidspace_box = kwimage.Box.from_slice(vidspace_region).to_ltrb()

    # Find all annotations that pass through this spatial region
    if use_annot_info:
        query = vidspace_box.data
        isect_aids = list(qtree.intersect(query))
        isect_gids = set(dset.annots(isect_aids).lookup('image_id'))
        isect_aids_catnames = dset.annots(isect_aids).category_names
        if negative_classes is not None:
            isect_positive_catnames = set(isect_aids_catnames) - set(negative_classes)
        else:
            isect_positive_catnames = set(isect_aids_catnames)
        aid_to_catname = dict(zip(isect_aids, isect_aids_catnames))
        gid_to_aids = {x: dset.index.gid_to_aids[x] & set(isect_aids) for x in isect_gids}
        gid_to_catnames = {k: list(ub.take(aid_to_catname, v)) for k, v in gid_to_aids.items()}

    if respect_valid_regions:
        # Reselect the keyframes if we overlap an invalid region (as
        # denoted in the metadata, further filtering may happen later)
        # todo: refactor to be cleaner
        try:
            main_idx_to_gids2, resampled = _refine_time_sample(
                dset, main_idx_to_gids, vidspace_box,
                refine_iosa_thresh, time_sampler, image_props)
        except tsm.TimeSampleError as ex:
            # Hack, just skip the region
            # We might be able to sample less and still be ok
            print(f'TSM ex={ex}')
            raise
            # continue
    else:
        main_idx_to_gids2 = main_idx_to_gids
        resampled = False

    if set_cover_algo is not None:
        # TODO: allow there to be a slightly denser overlapped sampling
        # debug = True
        debug = 0
        if debug:
            print('before applying set cover, len of main_idx_to_gids2', len(main_idx_to_gids2))
        main_idx_to_gids2 = kwarray.setcover(main_idx_to_gids2, algo=set_cover_algo)
        if debug:
            print('after applying set cover', len(main_idx_to_gids2))

    for main_idx, gids in main_idx_to_gids2.items():
        main_gid = time_sampler.video_gids[main_idx]
        label = 'unknown'
        annot_info = None

        if use_annot_info:
            if isect_aids:
                has_annot = bool(isect_gids & set(gids)) and bool(isect_positive_catnames)
            else:
                has_annot = False
            if has_annot:
                label = 'positive_grid'
            else:
                # Hack: exclude all annotated regions from negative sampling
                label = 'negative_grid'

            annot_info = {
                'isect_aids': isect_aids,
                'isect_aids_catnames': isect_aids_catnames,
                'gid_to_aids': ub.dict_subset(gid_to_aids, gids, default=[]),
                'gid_to_catnames': ub.dict_subset(gid_to_catnames, gids, default=[]),
                'main_gid_catnames': ub.dict_hist(list(ub.take(gid_to_catnames, [main_gid], default=[]))[0]),
            }

        target = {
            'main_idx': main_idx,
            'video_id': video_id,
            'gids': gids,
            'main_gid': main_gid,
            'space_slice': vidspace_region,
            'resampled': resampled,
            'label': label,
            'annot_info': annot_info,
        }
        yield target


@profile
def _build_vidspace_track_qtree(dset, video_gids, vidspace_video_width,
                                vidspace_video_height, image_props):
    """
    Build a data structure that allows for fast lookup of which annotations
    exist in the in the requested "Window Space".

    Called as part of :func:`_sample_single_video_spacetime_targets`.

    TODO:
        - [ ] For tracks with multiple annotations, we should build a range of
        values indicating where the intersection occurs in spacetime instead of
        putting each annotation on each frame. This should increase efficiency
        by a lot.

    Returns:
        Tuple[pyqtree.Index, Dict, Dict]
    """
    import pyqtree

    # Initialize a single qtree to represent where objects will be found across
    # all time in the video. We do need a better 3D way of doing this for long
    # videos where objects are moving.
    qtree = pyqtree.Index((0, 0, vidspace_video_width, vidspace_video_height))
    tid_to_infos = ub.ddict(list)
    loose_aid_to_infos = ub.ddict(list)
    video_images = dset.images(video_gids)
    video_coco_images = video_images.coco_images
    video_aids = video_images.annots.lookup('id')

    # For each image in the video sequence consider its annotations.
    for aids, coco_img in zip(video_aids, video_coco_images):

        # Consider the data in this frame only.
        img_info = coco_img.img
        gid = img_info['id']
        # frame_index = img_info['frame_index']
        annots = dset.annots(aids)
        tids = annots.lookup('track_id', None)
        cids = annots.lookup('category_id', None)
        cnames = dset.categories(cids).name

        # Gather data in imagespace so we can group expensive steps together
        imgspace_xywh = []
        infos = []
        for tid, aid, cid, cname in zip(tids, aids, cids, cnames):
            imgspace_xywh.append(dset.index.anns[aid]['bbox'])
            infos.append({
                'gid': gid,
                'cid': cid,
                'tid': tid,
                # 'frame_index': frame_index,
                'cname': dset._resolve_to_cat(cid)['name'],
                'aid': aid,
            })

        imgspace_boxes = kwimage.Boxes(np.array(imgspace_xywh), 'xywh')

        # Convert to image space in a single call
        warp_vid_from_img = image_props.get_warp_vid_from_img(gid)
        vidspace_boxes = imgspace_boxes.warp(warp_vid_from_img)
        vidspace_boxes = vidspace_boxes.clip(
            0, 0, vidspace_video_width, vidspace_video_height)

        isvalid_flags = vidspace_boxes.area.ravel() > 0
        valid_vidspace_boxes = vidspace_boxes.compress(isvalid_flags)
        valid_infos = list(ub.compress(infos, isvalid_flags))
        valid_ltrbs = valid_vidspace_boxes.to_ltrb().data

        for info, ltrb in zip(valid_infos, valid_ltrbs):
            tid = info['tid']
            aid = info['aid']
            info.update({
                'vidspace_box': ltrb,
            })
            if tid is not None:
                tid_to_infos[tid].append(info)
            else:
                # Remember annotations that are no associated with a particular
                # track.
                loose_aid_to_infos[aid] = info
            # A faster qtree insert would be helpful.
            # This takes 50% of the time in spacetime grid building.
            qtree.insert(aid, ltrb)

    return qtree, tid_to_infos, loose_aid_to_infos


@profile
def _refine_time_sample(dset, main_idx_to_gids, vidspace_box, refine_iosa_thresh, time_sampler, image_props):
    """
    Refine the time sample based on spatial information

    Attempt to remove images where valid data does not spatially intersect the
    query box.
    """
    from geowatch.tasks.fusion.datamodules import temporal_sampling as tsm  # NOQA
    video_gids = time_sampler.video_gids

    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", category=RuntimeWarning,
            message="invalid value encountered in intersection")

        # Mark images where the valid region does not intersect the
        # query vidspace box.
        gid_to_isbad = {}
        if refine_iosa_thresh > 0:
            for gid in video_gids:
                vidspace_valid_poly = image_props.get_image_valid_region_in_vidspace(gid)
                gid_to_isbad[gid] = False
                # If the area is of the valid polygon is less than zero, there
                # was probably an issue. treat it as if it didn't specify a
                # valid region.
                if vidspace_valid_poly is not None and vidspace_valid_poly.area > 0:
                    vidspace_box_poly = vidspace_box.to_shapely()
                    # Intersection over smaller area
                    isect = vidspace_valid_poly.intersection(vidspace_box_poly)
                    iosa = isect.area / min(vidspace_box_poly.area, vidspace_valid_poly.area)
                    if iosa < refine_iosa_thresh:
                        gid_to_isbad[gid] = True

    all_bad_gids = [gid for gid, flag in gid_to_isbad.items() if flag]

    rng = kwarray.ensure_rng(1093881655714)

    try:
        resampled = 0
        refined_sample = {}
        for main_idx, gids in main_idx_to_gids.items():
            main_gid = video_gids[main_idx]
            # Skip the sample when the "main" frame is bad.
            if not gid_to_isbad[main_gid]:
                good_gids = [gid for gid in gids if not gid_to_isbad[gid]]
                if good_gids != gids:
                    include_idxs = np.where(kwarray.isect_flags(video_gids, good_gids))[0]
                    exclude_idxs = np.where(kwarray.isect_flags(video_gids, all_bad_gids))[0]

                    chosen = time_sampler.sample(
                        include=include_idxs, exclude=exclude_idxs,
                        error_level=1, return_info=False, rng=rng)

                    new_gids = list(ub.take(video_gids, chosen))
                    # Are we allowed to return less than the initial expected
                    # number of frames? For transformers yes, but we should be
                    # careful to ask the user if they expect this.
                    new_are_bad = [g for g in new_gids if gid_to_isbad[g]]
                    if not new_are_bad:
                        resampled += 1
                        refined_sample[main_idx] = new_gids
                else:
                    refined_sample[main_idx] = gids
    except tsm.TimeSampleError:
        print('had a hard time resampling')
        raise

    # print(f'number of resampled grid cells={resampled}')
    return refined_sample, resampled


def _visualize_sample_grid(dset, sample_grid, max_vids=2, max_frames=6, video_ids=None):
    # Visualize the sample grid
    import pandas as pd
    from geowatch.utils import util_kwimage
    targets = pd.DataFrame(sample_grid['targets'])

    dataset_canvases = []

    # max_vids = 2
    # max_frames = 6

    vidid_to_videodf = dict(list(targets.groupby('video_id')))

    orientation = 1

    if video_ids is None:
        video_ids = list(vidid_to_videodf.keys())

    for vidid in video_ids:
        video_df = vidid_to_videodf[vidid]
        video = dset.index.videos[vidid]
        vidname = video['name']
        gid_to_infos = ub.ddict(list)
        for _, row in video_df.iterrows():
            for gid in row['gids']:
                gid_to_infos[gid].append({
                    'gid': gid,
                    'space_slice': row['space_slice'],
                    'label': row['label'],
                })

        video_canvases = []
        common = ub.oset(dset.images(video_id=vidid)) & (gid_to_infos)

        if True:
            # HACK: Use a temporal sampler once to get a nice overview of the
            # dataset in time.
            from kwutil import util_time
            from geowatch.tasks.fusion.datamodules import temporal_sampling as tsm  # NOQA
            images = dset.images(common)
            datetimes = [util_time.coerce_datetime(date) for date in images.lookup('date_captured', None)]
            unixtimes = np.array([np.nan if dt is None else dt.timestamp() for dt in datetimes])
            sensors = images.lookup('sensor_coarse', None)
            time_sampler = tsm.MultiTimeWindowSampler(
                unixtimes=unixtimes, sensors=sensors, time_window=max_frames,
                time_span='1y', affinity_type='hardish3',
                update_rule='distribute+pairwise')
            sample = time_sampler.sample(rng=0)
            common = list(ub.take(common, sample))

        for gid in common:
            infos = gid_to_infos[gid]
            label_to_items = ub.group_items(infos, key=lambda x: x['label'])
            video = dset.index.videos[vidid]

            shape = (video['height'], video['width'], 4)
            canvas = np.zeros(shape, dtype=np.float32)
            shape = (2, video['height'], video['width'])
            accum = np.zeros(shape, dtype=np.float32)

            for label, items in label_to_items.items():
                label_idx = {'positive_grid': 1, 'positive_center': 1,
                             'negative_grid': 0}[label]
                for info in items:
                    space_slice = info['space_slice']

                    # Hack to fix fractional slices
                    box = kwimage.Box.from_slice(space_slice)
                    box = box.quantize()
                    space_slice = box.to_slice()

                    y_sl, x_sl = space_slice
                    # ww = x_sl.stop - x_sl.start
                    # wh = y_sl.stop - y_sl.start
                    _index = (label_idx,) + space_slice
                    ss = accum[_index].shape
                    if np.prod(ss) > 0:
                        vals = util_kwimage.upweight_center_mask(ss)
                        vals = np.maximum(vals, 0.1)
                        accum[(label_idx,) + space_slice] += vals
                        # Add extra weight to borders for viz
                        accum[label_idx, y_sl.start:y_sl.start + 1, x_sl.start: x_sl.stop] += 0.15
                        accum[label_idx, y_sl.stop - 1:y_sl.stop, x_sl.start:x_sl.stop] += 0.15
                        accum[label_idx, y_sl.start:y_sl.stop, x_sl.start: x_sl.start + 1] += 0.15
                        accum[label_idx, y_sl.start:y_sl.stop, x_sl.stop - 1: x_sl.stop] += 0.15

            neg_accum = accum[0]
            pos_accum = accum[1]

            neg_alpha = neg_accum / (neg_accum.max() * 2)
            pos_alpha = pos_accum / (pos_accum.max() * 2)
            bg_canvas = canvas.copy()
            bg_canvas[..., 0:4] = [0., 0., 0., 1.0]
            pos_canvas = canvas.copy()
            neg_canvas = canvas.copy()
            pos_canvas[..., 0:3] = kwimage.Color('dodgerblue').as01()
            neg_canvas[..., 0:3] = kwimage.Color('orangered').as01()
            neg_canvas[..., 3] = neg_alpha
            pos_canvas[..., 3] = pos_alpha
            neg_canvas = np.nan_to_num(neg_canvas)
            pos_canvas = np.nan_to_num(pos_canvas)

            warp_vid_from_img = kwimage.Affine.coerce(
                dset.index.imgs[gid]['warp_img_to_vid'])

            vid_poly = kwimage.Boxes([[0, 0, video['width'], video['height']]], 'xywh').to_polygons()[0]
            coco_poly = dset.index.imgs[gid].get('valid_region', None)
            if coco_poly is None:
                kw_invalid_poly = None
            else:
                kw_poly_img = kwimage.MultiPolygon.coerce(coco_poly)
                valid_poly = kw_poly_img.warp(warp_vid_from_img)
                sh_invalid_poly = vid_poly.to_shapely().difference(valid_poly.to_shapely())
                kw_invalid_poly = kwimage.MultiPolygon.coerce(sh_invalid_poly)

            final_canvas = kwimage.overlay_alpha_layers([pos_canvas, neg_canvas, bg_canvas])
            final_canvas = kwimage.ensure_uint255(final_canvas)

            annot_dets = dset.annots(gid=gid).detections
            vid_annot_dets = annot_dets.warp(warp_vid_from_img)

            if 1:
                final_canvas = vid_annot_dets.draw_on(
                    final_canvas, color='white', alpha=0.5, kpts=False,
                    labels=False)

            if kw_invalid_poly is not None:
                final_canvas = kw_invalid_poly.draw_on(final_canvas, color='yellow', alpha=0.5)

            # from geowatch import heuristics
            img = dset.index.imgs[gid]
            header_lines = heuristics.build_image_header_text(
                img=img, vidname=vidname)
            header_text = '\n'.join(header_lines)

            # if 1:
            #     final_canvas = kwimage.imresize(final_canvas, dsize=(None, 1024))

            final_canvas = kwimage.draw_header_text(final_canvas, header_text, fit=True)
            video_canvases.append(final_canvas)

            if len(video_canvases) >= max_frames:
                break

        if max_vids == 1:
            video_canvas = kwimage.stack_images_grid(
                video_canvases, axis=orientation, pad=10)
        else:
            video_canvas = kwimage.stack_images(video_canvases, axis=1 - orientation, pad=10)
        dataset_canvases.append(video_canvas)
        if len(dataset_canvases) >= max_vids:
            break

    dataset_canvas = kwimage.stack_images(dataset_canvases, axis=orientation, pad=20)
    if 0:
        import kwplot
        kwplot.autompl()
        kwplot.imshow(dataset_canvas, doclf=1)
    return dataset_canvas


@profile
def sample_video_spacetime_targets(dset,
                                   time_dims=None,
                                   window_dims=None,
                                   window_overlap=0.0,
                                   negative_classes=None,
                                   keepbound=True,
                                   include_sensors=None,
                                   exclude_sensors=None,
                                   select_images=None,
                                   select_videos=None,
                                   time_sampling='hard+distribute',
                                   time_span='2y',
                                   time_kernel=None,
                                   use_annot_info=True,
                                   use_grid_positives=True,
                                   use_grid_negatives=True,
                                   use_centered_positives=True,
                                   window_space_scale=None,
                                   set_cover_algo=None,
                                   respect_valid_regions=True,
                                   dynamic_fixed_resolution=None,
                                   workers=0,
                                   use_cache=1):
    """
    Old API. Deprecated.
    """
    # ub.schedule_deprecation
    SpacetimeGridBuilder(**locals()).build()
