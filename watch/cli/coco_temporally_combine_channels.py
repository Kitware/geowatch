#!/usr/bin/env python3
import os

import kwcoco
import kwarray
import kwimage
import numpy as np
import ubelt as ub
import scriptconfig as scfg

from watch.utils import util_progress
from watch.utils import util_time, kwcoco_extensions
from watch.tasks.fusion.coco_stitcher import CocoStitchingManager


class TimeAverageConfig(scfg.DataConfig):
    """
    Averages kwcoco images over a sliding temporal window in a video.
    """
    kwcoco_fpath = scfg.Value(None, help=ub.paragraph(
            '''
            The path to the kwcoco file containing the image data to be
            combined.
            '''))

    output_kwcoco_fpath = scfg.Value(None, help=ub.paragraph(
            '''
            The path where the combined image data will be saved to in a
            kwcoco file.
            '''))

    channels = scfg.Value('red|green|blue', help=ub.paragraph(
            '''
            The channels to get and combine the spatial data from. E.g.
            "red|green|blue". Note: Separate channels with "|".
            '''))

    temporal_window_duration = scfg.Value('1month', help=ub.paragraph(
            '''
            The amount of time the temporal window should cover in days.
            E.g. 365 for a year.
            '''))

    merge_method = scfg.Value('mean', help=ub.paragraph(
            '''
            The combine method to use. Choices: "mean", "median".
            '''))

    resolution = scfg.Value('10GSD', help=ub.paragraph(
            '''
            The resolution the imagery will be loaded during the
            combination operation and saved to the output kwcoco file.
            '''))

    filter_with_cloudmasks = scfg.Value(True, isflag=True, help=ub.paragraph(
            '''
            If active the cloudmasks will be used to filter out pixels
            with too much cloud coverage or missing data.
            '''))

    s2_weight_factor = scfg.Value(1.0, help=ub.paragraph(
            '''
            A weighting factor to scale the impact of Sentinel-2 pixels
            during the combination operation. Note: Only effects the
            merge method "mean".
            '''))
    separate_sensors = scfg.Value(True, isflag=True, help='Combine images by sensor separately.')

    workers = scfg.Value(0, help=ub.paragraph(
            '''
            The number of CPU cores to compute the combination operation
            with.
            '''))

    include_sensors = scfg.Value(None, help=ub.paragraph(
            '''
            A list of sensors to include in the combination operation.
            '''))

    exclude_sensors = scfg.Value(None, help=ub.paragraph(
            '''
            A list of sensors to exclude from the combination operation.
            '''))

    select_images = scfg.Value(None, help='TODO:')

    select_videos = scfg.Value(None, help='TODO:')


def main(cmdline=1, **kwargs):
    """
    CommandLine:
        DEVEL_TEST=1 xdoctest -m watch.cli.coco_temporally_combine_channels main

        from watch.cli.coco_temporally_combine_channels import *  # NOQA
        import watch
        data_dvc_dpath = watch.find_dvc_dpath(tags='phase2_data', hardware='auto')
        cmdline = 0
        channels='red|green|blue'
        kwargs = dict(
            kwcoco_fpath=data_dvc_dpath / 'Drop6/imgonly-KR_R001.kwcoco.json',
            output_kwcoco_fpath=data_dvc_dpath / 'TestAveDrop6/test-timeave-valid_split1_1yr_mean_test.kwcoco.json',
            workers=4,
            filter_with_cloudmasks=True,
            temporal_window_duration='1 year',
            channels=channels,
        )
        output_coco_dset = main(cmdline=cmdline, **kwargs)
        coco_visualize_videos.main(cmdline=cmdline, src=kwargs['output_kwcoco_fpath'], smart=True)

    Example:
        >>> # 0: Baseline run.
        >>> # xdoctest: +REQUIRES(env:DEVEL_TEST)
        >>> from watch.cli.coco_temporally_combine_channels import *  # NOQA
        >>> import watch
        >>> data_dvc_dpath = watch.find_dvc_dpath(tags='phase2_data', hardware='auto')
        >>> cmdline = 0
        >>> kwargs = dict(
        >>>     kwcoco_fpath=data_dvc_dpath / 'Drop6/imgonly-KR_R001.kwcoco.json',
        >>>     output_kwcoco_fpath=data_dvc_dpath / 'Drop6/test-timeave-imgonly-KR_R001.kwcoco.json',
        >>>     workers=11,
        >>>     filter_with_cloudmasks=False,
        >>>     temporal_window_duration='1 year',
        >>>     channels='red|green|blue',
        >>> )
        >>> output_coco_dset = main(cmdline=cmdline, **kwargs)
        >>> from watch.cli import coco_visualize_videos
        >>> coco_visualize_videos.main(cmdline=0, src=kwargs['output_kwcoco_fpath'], smart=True)

    Ignore:
        DVC_DATA_DPATH=$(smartwatch_dvc --tags='phase2_data' --hardware=auto)
        smartwatch stats $DVC_DATA_DPATH/Drop6/test-timeave-imgonly-KR_R001.kwcoco.json

    Example:
        >>> # 1: Check cloudmasking.
        >>> # xdoctest: +REQUIRES(env:DEVEL_TEST)
        >>> from watch.cli.coco_temporally_combine_channels import *  # NOQA
        >>> import watch
        >>> data_dvc_dpath = watch.find_dvc_dpath(tags='phase2_data', hardware='auto')
        >>> cmdline = 0
        >>> kwargs = dict(
        >>>     kwcoco_fpath=data_dvc_dpath / 'Drop6/imgonly-KR_R001.kwcoco.json',
        >>>     output_kwcoco_fpath=data_dvc_dpath / 'Drop6/test-timeave-KR_R001-cloudmask.kwcoco.json',
        >>>     workers=11,
        >>>     temporal_window_duration='1 year',
        >>>     channels='red|green|blue',
        >>>     filter_with_cloudmasks=True,
        >>> )
        >>> output_coco_dset = main(cmdline=cmdline, **kwargs)
        >>> from watch.cli import coco_visualize_videos
        >>> coco_visualize_videos.main(cmdline=0, src=kwargs['output_kwcoco_fpath'], smart=True)

    Ignore:
        DVC_DATA_DPATH=$(smartwatch_dvc --tags='phase2_data' --hardware=auto)
        smartwatch stats $DVC_DATA_DPATH/Drop6/test-timeave-KR_R001.kwcoco.json

    Example:
        >>> # 2: Check that resolution can be updated.
        >>> # xdoctest: +REQUIRES(env:DEVEL_TEST)
        >>> from watch.cli.coco_temporally_combine_channels import *  # NOQA
        >>> import watch
        >>> data_dvc_dpath = watch.find_dvc_dpath(tags='phase2_data', hardware='auto')
        >>> cmdline = 0
        >>> kwargs = dict(
        >>>     kwcoco_fpath=data_dvc_dpath / 'Drop6/imgonly-KR_R001.kwcoco.json',
        >>>     output_kwcoco_fpath=data_dvc_dpath / 'Drop6/test-timeave-KR_R001-cloudmask-5GSD.kwcoco.json',
        >>>     workers=11,
        >>>     temporal_window_duration='1 year',
        >>>     channels='red|green|blue',
        >>>     filter_with_cloudmasks=True,
        >>>     resolution='5GSD',
        >>> )
        >>> output_coco_dset = main(cmdline=cmdline, **kwargs)
        >>> from watch.cli import coco_visualize_videos
        >>> coco_visualize_videos.main(cmdline=0, src=kwargs['output_kwcoco_fpath'], smart=True)

    Ignore:
        DVC_DATA_DPATH=$(smartwatch_dvc --tags='phase2_data' --hardware=auto)
        smartwatch stats $DVC_DATA_DPATH/Drop6/test-timeave-KR_R001-cloudmask.kwcoco.json

    Example:
        >>> # 3: Median combining.
        >>> # xdoctest: +REQUIRES(env:DEVEL_TEST)
        >>> from watch.cli.coco_temporally_combine_channels import *  # NOQA
        >>> import watch
        >>> data_dvc_dpath = watch.find_dvc_dpath(tags='phase2_data', hardware='auto')
        >>> cmdline = 0
        >>> kwargs = dict(
        >>>     kwcoco_fpath=data_dvc_dpath / 'Drop6/imgonly-KR_R001.kwcoco.json',
        >>>     output_kwcoco_fpath=data_dvc_dpath / 'Drop6/test-timeave-KR_R001-median.kwcoco.json',
        >>>     workers=11,
        >>>     merge_method='median',
        >>>     temporal_window_duration='1 year',
        >>>     channels='red|green|blue',
        >>>     filter_with_cloudmasks=False,
        >>>     resolution='10GSD',
        >>> )
        >>> output_coco_dset = main(cmdline=cmdline, **kwargs)
        >>> from watch.cli import coco_visualize_videos
        >>> coco_visualize_videos.main(cmdline=0, src=kwargs['output_kwcoco_fpath'], smart=True)

    Ignore:
        DVC_DATA_DPATH=$(smartwatch_dvc --tags='phase2_data' --hardware=auto)
        smartwatch stats $DVC_DATA_DPATH/Drop6/test-timeave-KR_R001-median.kwcoco.json

    Example:
        >>> # 4: Median combining with cloudmask.
        >>> # xdoctest: +REQUIRES(env:DEVEL_TEST)
        >>> from watch.cli.coco_temporally_combine_channels import *  # NOQA
        >>> import watch
        >>> data_dvc_dpath = watch.find_dvc_dpath(tags='phase2_data', hardware='auto')
        >>> cmdline = 0
        >>> kwargs = dict(
        >>>     kwcoco_fpath=data_dvc_dpath / 'Drop6/imgonly-KR_R001.kwcoco.json',
        >>>     output_kwcoco_fpath=data_dvc_dpath / 'Drop6/test-timeave-KR_R001-cloudmask-median.kwcoco.json',
        >>>     workers=11,
        >>>     merge_method='median',
        >>>     temporal_window_duration='1 year',
        >>>     channels='red|green|blue',
        >>>     filter_with_cloudmasks=True,
        >>>     resolution='10GSD',
        >>> )
        >>> output_coco_dset = main(cmdline=cmdline, **kwargs)
        >>> from watch.cli import coco_visualize_videos
        >>> coco_visualize_videos.main(cmdline=0, src=kwargs['output_kwcoco_fpath'], smart=True)

    Ignore:
        DVC_DATA_DPATH=$(smartwatch_dvc --tags='phase2_data' --hardware=auto)
        smartwatch stats $DVC_DATA_DPATH/Drop6/test-timeave-KR_R001-cloudmask-median.kwcoco.json

    Example:
        >>> # 5: Dont separate sensors.
        >>> # xdoctest: +REQUIRES(env:DEVEL_TEST)
        >>> from watch.cli.coco_temporally_combine_channels import *  # NOQA
        >>> import watch
        >>> data_dvc_dpath = watch.find_dvc_dpath(tags='phase2_data', hardware='auto')
        >>> cmdline = 0
        >>> kwargs = dict(
        >>>     kwcoco_fpath=data_dvc_dpath / 'Drop6/imgonly-KR_R001.kwcoco.json',
        >>>     output_kwcoco_fpath=data_dvc_dpath / 'Drop6/test-timeave-KR_R001-cloudmask-no_sensor_separate.kwcoco.json',
        >>>     workers=11,
        >>>     merge_method='mean',
        >>>     temporal_window_duration='1 year',
        >>>     channels='red|green|blue',
        >>>     filter_with_cloudmasks=True,
        >>>     resolution='10GSD',
        >>>     separate_sensors=False,
        >>> )
        >>> output_coco_dset = main(cmdline=cmdline, **kwargs)
        >>> from watch.cli import coco_visualize_videos
        >>> coco_visualize_videos.main(cmdline=0, src=kwargs['output_kwcoco_fpath'], smart=True)

    Ignore:
        DVC_DATA_DPATH=$(smartwatch_dvc --tags='phase2_data' --hardware=auto)
        smartwatch stats $DVC_DATA_DPATH/Drop6/test-timeave-KR_R001-cloudmask-no_sensor_separate.kwcoco.json

    Example:
        >>> # 6: Adjust the effect of S2 imagery.
        >>> # xdoctest: +REQUIRES(env:DEVEL_TEST)
        >>> from watch.cli.coco_temporally_combine_channels import *  # NOQA
        >>> import watch
        >>> data_dvc_dpath = watch.find_dvc_dpath(tags='phase2_data', hardware='auto')
        >>> cmdline = 0
        >>> kwargs = dict(
        >>>     kwcoco_fpath=data_dvc_dpath / 'Drop6/imgonly-KR_R001.kwcoco.json',
        >>>     output_kwcoco_fpath=data_dvc_dpath / 'Drop6/test-timeave-KR_R001-cloudmask-s2w_10.kwcoco.json',
        >>>     workers=11,
        >>>     merge_method='mean',
        >>>     temporal_window_duration='1 year',
        >>>     channels='red|green|blue',
        >>>     filter_with_cloudmasks=True,
        >>>     resolution='10GSD',
        >>>     s2_weight_factor=10.0,
        >>>     separate_sensors=False,
        >>> )
        >>> output_coco_dset = main(cmdline=cmdline, **kwargs)
        >>> from watch.cli import coco_visualize_videos
        >>> coco_visualize_videos.main(cmdline=0, src=kwargs['output_kwcoco_fpath'], smart=True)

    Ignore:
        DVC_DATA_DPATH=$(smartwatch_dvc --tags='phase2_data' --hardware=auto)
        smartwatch stats $DVC_DATA_DPATH/Drop6/test-timeave-KR_R001-cloudmask-s2w_10.kwcoco.json
    """
    config = TimeAverageConfig.cli(cmdline=cmdline, data=kwargs, strict=True)
    print('config = ' + ub.urepr(dict(config), nl=1))

    output_coco_dset = combine_kwcoco_channels_temporally(config)
    return output_coco_dset


def combine_kwcoco_channels_temporally(config):
    """Combine spatial data within a temporal window from a kwcoco dataset and save the result to a new kwcoco dataset.

    High level steps:
    1. Load kwcoco dataset.
    2. Divide the dataset into temporal windows.
    3. For each temporal window, combine the spatial data from each channel.
    4. Save the combined image result to a new kwcoco dataset.
    """
    # Check inputs.
    space = 'video'

    ## Check input kwcoco file path exists.
    if os.path.exists(config.kwcoco_fpath) is False:
        raise FileNotFoundError(f'Input kwcoco file path does not exist: {config.kwcoco_fpath}')

    ## Check the S2 weight factor and merge method combination.
    if config.s2_weight_factor != 1.0 and config.merge_method != 'mean':
        print('WARNING: S2 weight factor only effects the merge method "mean".')

    # 1. Load kwcoco dataset.
    coco_dset = kwcoco.CocoDataset.coerce(config.kwcoco_fpath)

    selected_gids = kwcoco_extensions.filter_image_ids(
        coco_dset,
        include_sensors=config['include_sensors'],
        exclude_sensors=config['exclude_sensors'],
        select_images=config['select_images'],
        select_videos=config['select_videos'],
    )

    if selected_gids is not None:
        coco_dset = coco_dset.subset(selected_gids)

    output_coco_dset = coco_dset.copy()
    output_coco_dset.clear_images()  # we will write all new images

    requested_sensorchan = kwcoco.SensorChanSpec.coerce(config.channels)
    requested_chans = requested_sensorchan.chans

    ## Get saved asset directory.
    output_kwcoco_fpath = ub.Path(config.output_kwcoco_fpath)
    new_bundle_dpath = output_kwcoco_fpath.parent
    new_bundle_dpath.ensuredir()

    # 2. Divide the dataset into temporal windows (per video).

    ## Convert temporal_window_duration from days to seconds.
    time_delta = util_time.coerce_timedelta(config.temporal_window_duration)
    time_delta_seconds = time_delta.total_seconds()

    video_ids = [vid for vid in coco_dset.videos()]

    # pman = util_progress.ProgressManager(backend='rich')
    pman = util_progress.ProgressManager(backend='progiter')  # uncomment if you need to embed

    with pman:

        for vid in pman.progiter(video_ids, desc='Combining channel info within temporal windows'):
            # Get all image ids for the video.
            images = coco_dset.images(video_id=vid)

            # Determine what channels in each image we want to merge
            image_merge_channels = []
            for coco_img in images.coco_images:
                if requested_chans.spec == '*':
                    # Merge everything
                    mergable = coco_img.channels.fuse().to_set()
                else:
                    mergable = requested_chans.to_set() & coco_img.channels.fuse().to_set()
                image_merge_channels.append(frozenset(mergable))

            # Filter to only images that have a channel to merge
            flags = [len(c) > 0 for c in image_merge_channels]
            image_merge_channels = list(ub.compress(image_merge_channels, flags))
            images = images.compress(flags)

            image_datetimes = list(map(util_time.coerce_datetime, images.lookup('timestamp')))

            image_unixtimes = np.array([t.timestamp() for t in image_datetimes])

            # Get number of seconds after the first image
            image_rel_unixtimes = image_unixtimes - image_unixtimes[0]

            # We might group via multiple criteria
            groupers = {}

            # Assign each image to a temporal bucket/window
            image_buckets = np.floor(image_rel_unixtimes / time_delta_seconds).astype(int)
            groupers['bucket'] = image_buckets

            # groupers['channels'] = image_merge_channels  # not sure if we do this or not

            if config.separate_sensors:
                image_sensors = images.lookup('sensor_coarse', None)
                groupers['sensor'] = image_sensors

            # Construct a group-id for each image
            image_groupids = list(zip(*groupers.values()))

            # Group all image-indexes within the same group-id together.
            # unique_groupids, grouped_idxs = kwarray.group_indices(image_groupids)
            # groupid_to_idxs = dict(zip(unique_groupids, grouped_idxs))
            groupid_to_idxs = ub.group_items(range(len(image_groupids)), image_groupids)

            # DEBUG: Print the distribution of images per window.
            if 0:
                region_name = coco_dset.index.videos[vid]['name']
                # Get the histogram for the number of images per window.
                bucket_to_num_images = ub.udict(groupid_to_idxs).map_values(len)
                bucket_stats = kwarray.stats_dict(list(bucket_to_num_images.values()))
                print(f'[{region_name}] Distribution of images per window:')
                print('Histogram: = {}'.format(ub.urepr(bucket_to_num_images, nl=1)))
                print('N images per window: = {}'.format(ub.urepr(bucket_stats, nl=1)))

            jobs = ub.JobPool(mode='process', max_workers=config.workers)

            # 3. For each temporal window, combine the spatial data from each channel.
            chunk_image_idxs = list(groupid_to_idxs.values())
            prog = pman.progiter(chunk_image_idxs, transient=True, desc='Submit combine within temporal windows jobs')
            for chunk_image_idxs in prog:
                window_image = images.take(chunk_image_idxs)
                window_coco_images = window_image.coco_images
                # Detach for process parallelization
                window_coco_images = [g.detach() for g in window_coco_images]
                # video_id = window_coco_images[0]['video_id']

                # (window_coco_images, merge_method, requested_chans, space, resolution, save_assest_dir,
                #  filter_with_cloudmasks, s2_weight_factor, og_kwcoco_fpath) = (
                #      window_coco_images, config.merge_method, requested_chans,
                #      space, config.resolution, save_assest_dir,
                #      config.filter_with_cloudmasks, config.s2_weight_factor,
                #      config.kwcoco_fpath)

                jobs.submit(merge_images, window_coco_images, config.merge_method, requested_chans, space,
                            config.resolution, new_bundle_dpath, config.filter_with_cloudmasks, config.s2_weight_factor,
                            config.kwcoco_fpath)

            for job in pman.progiter(jobs.as_completed(),
                                     total=len(jobs),
                                     desc='Collect combine within temporal windows jobs'):
                new_img = job.result()
                output_coco_dset.add_image(**new_img)

    # Save kwcoco file.
    print(f"Saving ouput kwcoco file to: {output_kwcoco_fpath}")
    output_coco_dset.fpath = output_kwcoco_fpath
    output_coco_dset.validate()
    output_coco_dset.dump()
    print(f"Saved ouput kwcoco file to: {output_kwcoco_fpath}")
    return output_coco_dset


def merge_images(window_coco_images, merge_method, requested_chans, space, resolution, new_bundle_dpath,
                 filter_with_cloudmasks, s2_weight_factor, og_kwcoco_fpath):
    """
    Args:
        window_coco_images (List[kwcoco.CocoImage]): images with channels to merge

    Returns:
        Dict: a new coco image that points at the merged image on disk.
    """
    if requested_chans.spec == '*':
        merge_chans_set = set(ub.flatten([g.channels.fuse().to_set() for g in window_coco_images]))
    else:
        merge_chans_set = requested_chans

    merge_chans = kwcoco.FusedChannelSpec.coerce(sorted(merge_chans_set))

    # TODO: we should merge each asset at the highest resolution for that
    # asset, but no more.  For instance, when given red|green|blue|swir16 we
    # should handle rgb separately from swir16. This will involve grouping by
    # the asset resolution of each image. We need to maintain video space
    # alignment, but we can adjust by scalefactors. For now we are ignoring all
    # of this.

    # For now we just choose video space
    assert space == 'video'

    first_coco_img = window_coco_images[0]
    # Scales to the resolution from the requested (i.e. video space)
    scale_asset_from_vid = first_coco_img._scalefactor_for_resolution(
        resolution=resolution, space='video')
    # warp_asset_from_vid = kwimage.Affine.scale(scale_asset_from_vid)

    canvas_dsize = kwimage.Box.from_dsize(
        (first_coco_img.video['width'], first_coco_img.video['height'])).scale(scale_asset_from_vid).quantize().dsize
    canvas_dims = canvas_dsize[::-1]
    canvas_shape = canvas_dims + (merge_chans.numel(), )

    # Load and combine the images within this range.
    if merge_method == 'mean':
        accum = kwarray.Stitcher(canvas_shape)

        # TODO: we can also parallelize this in case the window is really big
        for coco_img in window_coco_images:
            delayed = coco_img.imdelay(merge_chans, space=space, resolution=resolution)
            image_data = delayed.finalize(nodata_method='float')

            pxl_weight = (1 - np.isnan(image_data)).astype(np.float32)

            if filter_with_cloudmasks:
                qa_data = coco_img.imdelay('quality',
                                           space=space,
                                           interpolation='nearest',
                                           antialias=False,
                                           resolution=resolution).finalize()

                from watch.tasks.fusion.datamodules.qa_bands import QA_SPECS
                # We don't have the exact right information here, so we can
                # punt for now and assume "Drop4"
                iffy_qa_names = [
                    'cloud',
                    'cloud_adjacent',
                    'cloud_shadow',
                ]
                spec_name = 'ACC-1'
                sensor = coco_img.img.get('sensor_coarse', '*')
                try:
                    table = QA_SPECS.find_table(spec_name, sensor)
                except AssertionError as ex:
                    print(f'warning ex={ex}')
                    is_iffy = None
                else:
                    is_iffy = table.mask_any(qa_data, iffy_qa_names)

                quality_mask = (1 - is_iffy)

                # Update pixel weights based on quality pixel values.
                pxl_weight *= quality_mask

            if coco_img['sensor_coarse'] == 'S2':
                pxl_weight *= s2_weight_factor

            image_data = np.nan_to_num(image_data)
            accum.add((slice(None), slice(None)), image_data, pxl_weight)

        import warnings
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', 'invalid')
            combined_image_data = accum.finalize()

    elif merge_method == 'median':
        # TODO: Make this less computationally expensive.
        median_stack = []
        for coco_img in window_coco_images:
            delayed = coco_img.imdelay(merge_chans, space=space, resolution=resolution)
            image_data = delayed.finalize(nodata_method='float')

            if filter_with_cloudmasks:
                qa_data = coco_img.imdelay('quality',
                                           space=space,
                                           interpolation='nearest',
                                           antialias=False,
                                           resolution=resolution).finalize()

                from watch.tasks.fusion.datamodules.qa_bands import QA_SPECS
                # We don't have the exact right information here, so we can
                # punt for now and assume "Drop4"
                iffy_qa_names = [
                    'cloud',
                    'cloud_adjacent',
                    'cloud_shadow',
                ]
                spec_name = 'ACC-1'
                sensor = coco_img.img.get('sensor_coarse', '*')
                try:
                    table = QA_SPECS.find_table(spec_name, sensor)
                except AssertionError as ex:
                    print(f'warning ex={ex}')
                    is_iffy = None
                else:
                    is_iffy = table.mask_any(qa_data, iffy_qa_names)

                quality_mask = (1 - is_iffy)

                # Update pixel weights based on quality pixel values.
                M = np.ma.masked_array(data=image_data, mask=~np.repeat(quality_mask, repeats=3, axis=2))
                image_data = M.filled(np.nan)

            median_stack.append(image_data)

        combined_image_data = np.nanmedian(median_stack, axis=0)

    else:
        raise NotImplementedError

    # 4. Save the combined image result to a new kwcoco dataset.
    ## Use the first image as the standin for the new image.

    # Create a dictionary for the new image
    # TODO: Should likely update properties to indicate what went into this new
    # image.
    new_img = first_coco_img.img.copy()
    new_img.pop('auxiliary', None)
    new_img.pop('assets', None)
    new_coco_img = kwcoco.CocoImage(new_img)

    # We are currently writing all merged assets in video space, so
    # the transform from asset space to image space is the transform
    # from video to image space.
    # warp_vid_from_asset = warp_asset_from_vid.inv()
    # new_warp_img_from_asset = first_coco_img.warp_img_from_vid @ warp_vid_from_asset

    # Create the same image name.
    hash_depends = {
        'names': [g.img['name'] for g in window_coco_images],
        'chans': merge_chans.spec,
        'resolution': resolution,
        'space': space,
        'version': 0,
        'og_coco_path': og_kwcoco_fpath,
        'merge_method': merge_method,
    }
    hashid = ub.hash_data(hash_depends)[0:16]

    timestr = util_time.isoformat(util_time.coerce_datetime(new_coco_img.img['timestamp']))
    chanstr = merge_chans.path_sanitize()
    new_name = f'ave_{timestr}_{len(window_coco_images):03d}_{chanstr}_{hashid}'
    new_coco_img.img['name'] = new_name
    new_coco_img.img['sensor_coarse'] = '_'.join(sorted(set([coco_img['sensor_coarse'] for coco_img in window_coco_images])))
    # new_coco_img.bundle_dapth = new_bundle_dpath
    dname = f'ave_{merge_chans.path_sanitize()}'
    # average_rel_fpath = ub.Path('_assets') / dname / new_name + '.tif'
    # average_fpath = new_bundle_dpath / average_rel_fpath
    # average_fpath.parent.ensuredir()

    # TODO: this should be set to the union of any valid_region_utms in the
    # input images.
    new_coco_img.img.pop('valid_region_utm', None)
    # new_coco_img.add_asset(
    #     file_name=os.fspath(average_rel_fpath),
    #     channels=merge_chans,
    #     warp_aux_to_img=new_warp_img_from_asset,
    #     width=combined_image_data.shape[1],
    #     height=combined_image_data.shape[0],
    # )

    # TODO: Figure out how to add geo-metadata to the new image from previous images.
    tmp_dset = kwcoco.CocoDataset()
    tmp_dset.bundle_dpath = new_bundle_dpath
    tmp_dset.add_video(**first_coco_img.video)
    tmp_dset.add_image(**new_coco_img.img)
    stitch_manager = CocoStitchingManager(
        tmp_dset,
        short_code=dname,
        chan_code=merge_chans.spec,
        stiching_space='video',
        # quantize=
        # expected_minmax=
    )

    gid = new_coco_img.img['id']
    stitch_manager.accumulate_image(gid, None, combined_image_data,
                                    asset_dsize=canvas_dsize,
                                    scale_asset_from_stitchspace=scale_asset_from_vid)
    stitch_manager.finalize_image(gid)
    final_img = tmp_dset.imgs[gid]

    return final_img


if __name__ == '__main__':
    """

    CommandLine:
        python -m watch.cli.coco_temporally_combine_channels


    """
    main()
