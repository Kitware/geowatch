#!/usr/bin/env python3
r"""

SeeAlso:
    ~/code/watch/geowatch/cli/queue_cli/prepare_time_combined_dataset.py


CommandLine:
    DVC_DATA_DPATH=$(geowatch_dvc --tags='phase2_data' --hardware=auto)

    python -m geowatch.cli.coco_time_combine \
        --input_kwcoco_fpath="$DVC_DATA_DPATH/Drop6/imgonly-KR_R002.kwcoco.zip" \
        --output_kwcoco_fpath="$DVC_DATA_DPATH/Drop6_MeanYear/imgonly-KR_R002.kwcoco.zip" \
        --channels="red|green|blue|nir|swir16|swir22" \
        --resolution=10GSD \
        --time_window=1year \
        --merge_method=mean \
        --workers=4

    python -m geowatch reproject_annotations \
        --src $DVC_DATA_DPATH/Drop6_MeanYear/imgonly-KR_R002.kwcoco.zip \
        --dst $DVC_DATA_DPATH/Drop6_MeanYear/imganns-KR_R002.kwcoco.zip \
        --site_models="$DVC_DATA_DPATH/annotations/drop6/site_models/*.geojson"


Ignore:
    # Debugging

    python -m geowatch.cli.coco_time_combine \
        --kwcoco_fpath="$HOME/data/dvc-repos/smart_data_dvc/Aligned-Drop7/VN_C002/imgonly-VN_C002-rawbands.kwcoco.zip" \
        --output_kwcoco_fpath="$HOME/data/dvc-repos/smart_data_dvc/Drop7-MedianNoWinter10GSD-V2/VN_C002/_unfielded_imgonly-VN_C002-rawbands.kwcoco.zip" \
        --channels="red|green|blue|nir|swir16|swir22|pan|coastal|cirrus|B05|B06|B07|B8A|B09" \
        --resolution="10GSD" \
        --time_window=1y \
        --remove_seasons=winter \
        --merge_method=median \
        --spatial_tile_size=1024 \
        --mask_low_quality=True \
        --start_time=2010-03-01 \
        --assets_dname="raw_bands" \
        --workers=0


Example:
    >>> # Toydata example for CI
    >>> import geowatch
    >>> from geowatch.cli import coco_time_combine
    >>> import ubelt as ub
    >>> dpath = ub.Path.appdir('geowatch/tests/cli/time_combine/t0')
    >>> dset = geowatch.coerce_kwcoco(
    >>>     'geowatch-msi', geodata=True,
    >>>     dates={'start_time': '2020-01-01', 'end_time': '2020-06-01'},
    >>>     image_size=(32, 32)
    >>> )
    >>> dpath.delete().ensuredir()
    >>> output_fpath = dpath / 'time_combined/data.kwcoco.json'
    >>> gsd = dset.videos().objs[0]['target_gsd']
    >>> kwargs = coco_time_combine.TimeCombineConfig(
    >>>     input_kwcoco_fpath=dset.fpath,
    >>>     output_kwcoco_fpath=output_fpath,
    >>>     time_window='2month',
    >>>     merge_method='mean',
    >>>     resolution=f'{gsd}GSD',
    >>>     start_time='2019-06-01',
    >>> )
    >>> cmdline = 0
    >>> coco_time_combine.main(cmdline=cmdline, **kwargs)
    >>> import kwcoco
    >>> out_dset = kwcoco.CocoDataset(output_fpath)
    >>> assert len(out_dset.videos()) == len(dset.videos())
    >>> assert out_dset.n_images < dset.n_images

    from geowatch.cli import coco_visualize_videos
    coco_visualize_videos.main(cmdline=0, src=output_fpath, stack='only', workers='avail')

"""
import os
import ubelt as ub
import scriptconfig as scfg


class TimeCombineConfig(scfg.DataConfig):
    """
    Averages kwcoco images over a sliding temporal window in a video.
    """
    __command__ = 'time_combine'

    input_kwcoco_fpath = scfg.Value(None, help=ub.paragraph(
            '''
            The path to the kwcoco file containing the image data to be
            combined.
            '''), alias=['kwcoco_fpath'], position=1)

    output_kwcoco_fpath = scfg.Value(None, help=ub.paragraph(
            '''
            The path where the combined image data will be saved to in a
            kwcoco file.
            '''), position=2)

    channels = scfg.Value('*', help=ub.paragraph(
            '''
            The channels to get and combine the spatial data from. E.g.
            "red|green|blue". Note: Separate channels with "|". A ``*``
            means combine everything.
            '''))

    time_window = scfg.Value('1month', help=ub.paragraph(
            '''
            The temporal window that will group images. E.g. 1y (or 365d) will
            group all images in every year. Buckets are non-overlapping.  The first
            image defines where the buckets start.
            '''), alias=['temporal_window_duration'])

    start_time = scfg.Value(None, help=ub.paragraph(
        '''
        The datetime to start time window partitioning. If unspecified the
        first date in the dataset is used.
        '''))

    merge_method = scfg.Value('mean', help=ub.paragraph(
            '''
            How to combine multiple observations over each time_window.
            '''), choices=['mean', 'median', 'max'])

    resolution = scfg.Value('10GSD', help=ub.paragraph(
            '''
            The resolution the imagery will be loaded during the
            combination operation and saved to the output kwcoco file.
            '''))

    mask_low_quality = scfg.Value(True, isflag=True, help=ub.paragraph(
            '''
            If True, use the quality masks to prevent low-quality pixels from
            being used in the average.
            '''), alias=['filter_with_cloudmasks'])

    sensor_weights = scfg.Value(0, help=ub.paragraph(
            '''
            Weight the contribution of each sensor in the average operation
            by a scalar value.
            '''))

    separate_sensors = scfg.Value(True, isflag=True, help=ub.paragraph(
        '''
        Combine images by sensor separately, otherwise bands from multiple
        sensors are averaged together.
        '''))

    workers = scfg.Value(0, help=ub.paragraph(
            '''
            The number of CPU cores to compute the combination operation with.
            '''))

    include_sensors = scfg.Value(None, help=ub.paragraph(
            '''
            A list of sensors to include in the combination operation.
            '''))

    assets_dname = scfg.Value('_assets', help=ub.paragraph(
        '''
        The name of the top-level directory to write new assets.
        '''))

    exclude_sensors = scfg.Value(None, help=ub.paragraph(
            '''
            A list of sensors to exclude from the combination operation.
            '''))

    select_images = scfg.Value(None, help='see kwcoco_extensions.filter_image_ids docs')

    select_videos = scfg.Value(None, help='see kwcoco_extensions.filter_image_ids docs')

    spatial_tile_size = scfg.Value(None, help=ub.paragraph(
        '''
            The size of the tiling over space to use when computing the average.
            If None, the average is computed on the entire stack of images at
            once.
        '''))

    filter_season = scfg.Value([], nargs='+', help=ub.paragraph(
        '''
            Images within this season(s) will be excluded from the average operation.
            Season options: ['spring', 'summer', 'fall', 'winter']
        '''), alias=['remove_seasons'])


def main(cmdline=1, **kwargs):
    """
    CommandLine:
        DEVEL_TEST=1 xdoctest -m geowatch.cli.coco_time_combine main

        from geowatch.cli.coco_time_combine import *  # NOQA
        import geowatch
        data_dvc_dpath = geowatch.find_dvc_dpath(tags='phase2_data', hardware='auto')
        cmdline = 0
        channels='red|green|blue'
        kwargs = dict(
            input_kwcoco_fpath=data_dvc_dpath / 'Drop6/imgonly-KR_R001.kwcoco.zip',
            output_kwcoco_fpath=data_dvc_dpath / 'TestAveDrop6/test-timeave-valid_split1_1yr_mean_test.kwcoco.zip',
            workers=4,
            mask_low_quality=True,
            time_window='1 year',
            channels=channels,
        )
        output_coco_dset = main(cmdline=cmdline, **kwargs)
        coco_visualize_videos.main(cmdline=cmdline, src=kwargs['output_kwcoco_fpath'], smart=True)

    Example:
        >>> # 0: Baseline run.
        >>> # xdoctest: +REQUIRES(env:DEVEL_TEST)
        >>> from geowatch.cli.coco_time_combine import *  # NOQA
        >>> import geowatch
        >>> data_dvc_dpath = geowatch.find_dvc_dpath(tags='phase2_data', hardware='auto')
        >>> cmdline = 0
        >>> kwargs = dict(
        >>>     input_kwcoco_fpath=data_dvc_dpath / 'Drop6/imgonly-KR_R001.kwcoco.zip',
        >>>     output_kwcoco_fpath=data_dvc_dpath / 'Drop6/test-timeave-imgonly-KR_R001.kwcoco.zip',
        >>>     workers=11,
        >>>     mask_low_quality=False,
        >>>     time_window='1 year',
        >>>     channels='red|green|blue',
        >>> )
        >>> output_coco_dset = main(cmdline=cmdline, **kwargs)
        >>> from geowatch.cli import coco_visualize_videos
        >>> coco_visualize_videos.main(cmdline=0, src=kwargs['output_kwcoco_fpath'], smart=True)

    Ignore:
        DVC_DATA_DPATH=$(geowatch_dvc --tags='phase2_data' --hardware=auto)
        geowatch stats $DVC_DATA_DPATH/Drop6/test-timeave-imgonly-KR_R001.kwcoco.zip

    Example:
        >>> # 1: Check cloudmasking.
        >>> # xdoctest: +REQUIRES(env:DEVEL_TEST)
        >>> from geowatch.cli.coco_time_combine import *  # NOQA
        >>> import geowatch
        >>> data_dvc_dpath = geowatch.find_dvc_dpath(tags='phase2_data', hardware='auto')
        >>> cmdline = 0
        >>> kwargs = dict(
        >>>     input_kwcoco_fpath=data_dvc_dpath / 'Drop6/imgonly-KR_R001.kwcoco.zip',
        >>>     output_kwcoco_fpath=data_dvc_dpath / 'Drop6/test-timeave-KR_R001-cloudmask.kwcoco.zip',
        >>>     workers=11,
        >>>     time_window='1 year',
        >>>     channels='red|green|blue',
        >>>     mask_low_quality=True,
        >>> )
        >>> output_coco_dset = main(cmdline=cmdline, **kwargs)
        >>> from geowatch.cli import coco_visualize_videos
        >>> coco_visualize_videos.main(cmdline=0, src=kwargs['output_kwcoco_fpath'], smart=True)

    Ignore:
        DVC_DATA_DPATH=$(geowatch_dvc --tags='phase2_data' --hardware=auto)
        geowatch stats $DVC_DATA_DPATH/Drop6/test-timeave-KR_R001.kwcoco.zip

    Example:
        >>> # 2: Check that resolution can be updated.
        >>> # xdoctest: +REQUIRES(env:DEVEL_TEST)
        >>> from geowatch.cli.coco_time_combine import *  # NOQA
        >>> import geowatch
        >>> data_dvc_dpath = geowatch.find_dvc_dpath(tags='phase2_data', hardware='auto')
        >>> cmdline = 0
        >>> kwargs = dict(
        >>>     input_kwcoco_fpath=data_dvc_dpath / 'Drop6/imgonly-KR_R001.kwcoco.zip',
        >>>     output_kwcoco_fpath=data_dvc_dpath / 'Drop6/test-timeave-KR_R001-cloudmask-5GSD.kwcoco.zip',
        >>>     workers=11,
        >>>     time_window='1 year',
        >>>     channels='red|green|blue',
        >>>     mask_low_quality=True,
        >>>     resolution='5GSD',
        >>> )
        >>> output_coco_dset = main(cmdline=cmdline, **kwargs)
        >>> from geowatch.cli import coco_visualize_videos
        >>> coco_visualize_videos.main(cmdline=0, src=kwargs['output_kwcoco_fpath'], smart=True)

    Ignore:
        DVC_DATA_DPATH=$(geowatch_dvc --tags='phase2_data' --hardware=auto)
        geowatch stats $DVC_DATA_DPATH/Drop6/test-timeave-KR_R001-cloudmask.kwcoco.zip

    Example:
        >>> # 3: Median combining.
        >>> # xdoctest: +REQUIRES(env:DEVEL_TEST)
        >>> from geowatch.cli.coco_time_combine import *  # NOQA
        >>> import geowatch
        >>> data_dvc_dpath = geowatch.find_dvc_dpath(tags='phase2_data', hardware='auto')
        >>> cmdline = 0
        >>> kwargs = dict(
        >>>     input_kwcoco_fpath=data_dvc_dpath / 'Drop6/imgonly-KR_R001.kwcoco.zip',
        >>>     output_kwcoco_fpath=data_dvc_dpath / 'Drop6/test-timeave-KR_R001-median.kwcoco.zip',
        >>>     workers=11,
        >>>     merge_method='median',
        >>>     time_window='1 year',
        >>>     channels='red|green|blue',
        >>>     mask_low_quality=False,
        >>>     resolution='10GSD',
        >>> )
        >>> output_coco_dset = main(cmdline=cmdline, **kwargs)
        >>> from geowatch.cli import coco_visualize_videos
        >>> coco_visualize_videos.main(cmdline=0, src=kwargs['output_kwcoco_fpath'], smart=True)

    Ignore:
        DVC_DATA_DPATH=$(geowatch_dvc --tags='phase2_data' --hardware=auto)
        geowatch stats $DVC_DATA_DPATH/Drop6/test-timeave-KR_R001-median.kwcoco.zip

    Example:
        >>> # 4: Median combining with cloudmask.
        >>> # xdoctest: +REQUIRES(env:DEVEL_TEST)
        >>> from geowatch.cli.coco_time_combine import *  # NOQA
        >>> import geowatch
        >>> data_dvc_dpath = geowatch.find_dvc_dpath(tags='phase2_data', hardware='auto')
        >>> cmdline = 0
        >>> kwargs = dict(
        >>>     input_kwcoco_fpath=data_dvc_dpath / 'Drop6/imganns-KR_R001.kwcoco.zip',
        >>>     output_kwcoco_fpath=data_dvc_dpath / 'Drop6/test-timeave-KR_R001-cloudmask-median.kwcoco.zip',
        >>>     workers=11,
        >>>     merge_method='median',
        >>>     time_window='1 year',
        >>>     channels='red|green|blue',
        >>>     mask_low_quality=True,
        >>>     resolution='10GSD',
        >>>     separate_sensors=True,
        >>> )
        >>> output_coco_dset = main(cmdline=cmdline, **kwargs)
        >>> from geowatch.cli import coco_visualize_videos
        >>> coco_visualize_videos.main(cmdline=0, src=kwargs['output_kwcoco_fpath'], smart=True)

    Ignore:
        DVC_DATA_DPATH=$(geowatch_dvc --tags='phase2_data' --hardware=auto)
        geowatch stats $DVC_DATA_DPATH/Drop6/test-timeave-KR_R001-cloudmask-median.kwcoco.zip

    Example:
        >>> # 5: Dont separate sensors.
        >>> # xdoctest: +REQUIRES(env:DEVEL_TEST)
        >>> from geowatch.cli.coco_time_combine import *  # NOQA
        >>> import geowatch
        >>> data_dvc_dpath = geowatch.find_dvc_dpath(tags='phase2_data', hardware='auto')
        >>> cmdline = 0
        >>> kwargs = dict(
        >>>     input_kwcoco_fpath=data_dvc_dpath / 'Drop6/imgonly-KR_R001.kwcoco.zip',
        >>>     output_kwcoco_fpath=data_dvc_dpath / 'Drop6/test-timeave-KR_R001-cloudmask-no_sensor_separate.kwcoco.zip',
        >>>     workers=11,
        >>>     merge_method='mean',
        >>>     time_window='1 year',
        >>>     channels='red|green|blue',
        >>>     mask_low_quality=True,
        >>>     resolution='10GSD',
        >>>     separate_sensors=False,
        >>> )
        >>> output_coco_dset = main(cmdline=cmdline, **kwargs)
        >>> from geowatch.cli import coco_visualize_videos
        >>> coco_visualize_videos.main(cmdline=0, src=kwargs['output_kwcoco_fpath'], smart=True)

    Ignore:
        DVC_DATA_DPATH=$(geowatch_dvc --tags='phase2_data' --hardware=auto)
        geowatch stats $DVC_DATA_DPATH/Drop6/test-timeave-KR_R001-cloudmask-no_sensor_separate.kwcoco.zip

    Example:
        >>> # 6: Adjust the effect of S2 imagery.
        >>> from geowatch.cli.coco_time_combine import *  # NOQA
        >>> # xdoctest: +REQUIRES(env:DEVEL_TEST)
        >>> import geowatch
        >>> data_dvc_dpath = geowatch.find_dvc_dpath(tags='phase2_data', hardware='auto')
        >>> cmdline = 0
        >>> kwargs = dict(
        >>>     input_kwcoco_fpath=data_dvc_dpath / 'Drop6/imganns-KR_R001.kwcoco.zip',
        >>>     output_kwcoco_fpath=data_dvc_dpath / 'Drop6/test-timeave-test_6-KR_R001.kwcoco.zip',
        >>>     workers=11,
        >>>     merge_method='mean',
        >>>     time_window='1 year',
        >>>     channels='red|green|blue',
        >>>     mask_low_quality=True,
        >>>     resolution='10GSD',
        >>>     separate_sensors=False,
        >>>     sensor_weights=dict(S2=2.5, L8=1.0, WV=5.0)
        >>> )
        >>> output_coco_dset = main(cmdline=cmdline, **kwargs)
        >>> from geowatch.cli import coco_visualize_videos
        >>> coco_visualize_videos.main(cmdline=0, src=kwargs['output_kwcoco_fpath'], smart=True)

    Ignore:
        DVC_DATA_DPATH=$(geowatch_dvc --tags='phase2_data' --hardware=auto)
        geowatch stats $DVC_DATA_DPATH/Drop6/test-timeave-KR_R001-cloudmask-s2w_10.kwcoco.zip

    Example:
        >>> # 7: Tile images instead of computing average all at once.
        >>> from geowatch.cli.coco_time_combine import *  # NOQA
        >>> # xdoctest: +REQUIRES(env:DEVEL_TEST)
        >>> import geowatch
        >>> data_dvc_dpath = geowatch.find_dvc_dpath(tags='phase2_data', hardware='auto')
        >>> cmdline = 0
        >>> kwargs = dict(
        >>>     input_kwcoco_fpath=data_dvc_dpath / 'Drop6/imganns-KR_R001.kwcoco.zip',
        >>>     output_kwcoco_fpath=data_dvc_dpath / 'Drop6/test-timeave-test_7-KR_R001.kwcoco.zip',
        >>>     workers=11,
        >>>     merge_method='median',
        >>>     time_window='1 year',
        >>>     channels='red|green|blue',
        >>>     mask_low_quality=True,
        >>>     resolution='10GSD',
        >>>     separate_sensors=False,
        >>>     spatial_tile_size=200,
        >>> )
        >>> output_coco_dset = main(cmdline=cmdline, **kwargs)
        >>> from geowatch.cli import coco_visualize_videos
        >>> coco_visualize_videos.main(cmdline=0, src=kwargs['output_kwcoco_fpath'], smart=True)

    Ignore:
        DVC_DATA_DPATH=$(geowatch_dvc --tags='phase2_data' --hardware=auto)
        geowatch stats $DVC_DATA_DPATH/Drop6/test-timeave-KR_R001-cloudmask-s2w_10.kwcoco.zip

    Example:
        >>> # 8: Tile images on large kwcoco file.
        >>> # xdoctest: +REQUIRES(env:DEVEL_TEST)
        >>> from geowatch.cli.coco_time_combine import *  # NOQA
        >>> import geowatch
        >>> data_dvc_dpath = geowatch.find_dvc_dpath(tags='phase2_data', hardware='auto')
        >>> cmdline = 0
        >>> kwargs = dict(
        >>>     input_kwcoco_fpath=data_dvc_dpath / 'Drop6/data_train_split1.kwcoco.zip',
        >>>     output_kwcoco_fpath=data_dvc_dpath / 'Drop6/test-timeave-test_8-train_split1.kwcoco.zip',
        >>>     workers=4,
        >>>     merge_method='median',
        >>>     time_window='1 year',
        >>>     channels='red|green|blue',
        >>>     mask_low_quality=True,
        >>>     resolution='10GSD',
        >>>     separate_sensors=False,
        >>>     spatial_tile_size=400,
        >>>     include_sensors=['S2', 'L8'],
        >>> )
        >>> output_coco_dset = main(cmdline=cmdline, **kwargs)
        >>> from geowatch.cli import coco_visualize_videos
        >>> coco_visualize_videos.main(cmdline=0, src=kwargs['output_kwcoco_fpath'], smart=True)

    Ignore:
        DVC_DATA_DPATH=$(geowatch_dvc --tags='phase2_data' --hardware=auto)
        geowatch stats $DVC_DATA_DPATH/Drop6/test-timeave-KR_R001-cloudmask-s2w_10.kwcoco.zip

    Example:
        >>> # 9: Exclude winter seasons for time average.
        >>> from geowatch.cli.coco_time_combine import *  # NOQA
        >>> # xdoctest: +REQUIRES(env:DEVEL_TEST)
        >>> import geowatch
        >>> data_dvc_dpath = geowatch.find_dvc_dpath(tags='phase2_data', hardware='auto')
        >>> cmdline = 0
        >>> kwargs = dict(
        >>>     input_kwcoco_fpath=data_dvc_dpath / 'Drop6/imganns-KR_R001.kwcoco.zip',
        >>>     output_kwcoco_fpath=data_dvc_dpath / 'Drop6/test-timeave-test_9-KR_R001.kwcoco.zip',
        >>>     workers=11,
        >>>     merge_method='mean',
        >>>     time_window='1 year',
        >>>     channels='red|green|blue',
        >>>     mask_low_quality=True,
        >>>     resolution='10GSD',
        >>>     separate_sensors=False,
        >>>     filter_season='winter',
        >>> )
        >>> output_coco_dset = main(cmdline=cmdline, **kwargs)
        >>> from geowatch.cli import coco_visualize_videos
        >>> coco_visualize_videos.main(cmdline=0, src=kwargs['output_kwcoco_fpath'], smart=True)

    Ignore:
        DVC_DATA_DPATH=$(geowatch_dvc --tags='phase2_data' --hardware=auto)
        geowatch stats $DVC_DATA_DPATH/Drop6/test-timeave-KR_R001-cloudmask-s2w_10.kwcoco.zip

    """
    config = TimeCombineConfig.cli(cmdline=cmdline, data=kwargs, strict=True)
    import rich
    rich.print('config = ' + ub.urepr(config, nl=1))

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
    import kwarray
    import kwcoco
    import numpy as np
    from kwutil import util_progress
    from kwutil import util_time
    from geowatch.utils import kwcoco_extensions
    from kwutil import util_parallel
    from kwutil.util_yaml import Yaml
    # Check inputs.

    space = 'video'

    workers = util_parallel.coerce_num_workers(config.workers)

    ## Check input kwcoco file path exists.
    if os.path.exists(config.input_kwcoco_fpath) is False:
        raise FileNotFoundError(f'Input kwcoco file path does not exist: {config.input_kwcoco_fpath}')

    sensor_weights = Yaml.coerce(config.sensor_weights)

    ## Check the S2 weight factor and merge method combination.
    if isinstance(sensor_weights, dict) is False:
        sensor_weights = {}

    for sensor_code, sensor_weight in sensor_weights.items():
        if sensor_weight != 1.0 and config.merge_method != 'mean':
            print(f'WARNING: {sensor_code} weight factor only effects the merge method "mean".')

    # 1. Load kwcoco dataset.
    coco_dset = kwcoco.CocoDataset.coerce(config.input_kwcoco_fpath)

    selected_gids = kwcoco_extensions.filter_image_ids(
        coco_dset,
        include_sensors=config['include_sensors'],
        exclude_sensors=config['exclude_sensors'],
        select_images=config['select_images'],
        select_videos=config['select_videos'],
    )

    # Optional: Filter image IDs by season.
    if config.filter_season is not None:
        selected_gids = filter_image_ids_by_season(coco_dset, selected_gids, config.filter_season)

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

    output_coco_dset.fpath = output_kwcoco_fpath

    # 2. Divide the dataset into temporal windows (per video).

    ## Convert time_window from days to seconds.
    time_delta = util_time.coerce_timedelta(config.time_window)
    start_time = util_time.coerce_datetime(config.start_time)
    time_delta_seconds = time_delta.total_seconds()

    video_ids = list(coco_dset.videos())

    pman = util_progress.ProgressManager()
    # pman = util_progress.ProgressManager(backend='progiter')  # uncomment if you need to use breakpoints

    resolution = config.resolution
    merge_method = config.merge_method
    mask_low_quality = config.mask_low_quality
    og_kwcoco_fpath = config.input_kwcoco_fpath
    if isinstance(config.spatial_tile_size, tuple) is False:
        spatial_tile_size = (config.spatial_tile_size, config.spatial_tile_size)
    else:
        spatial_tile_size = config.spatial_tile_size

    n_combined_images = 0
    n_failed_merges = 0

    with pman:

        for video_id in pman.progiter(video_ids, desc='Combining channel info within temporal windows'):
            # Get all image ids for the video.
            images = coco_dset.images(video_id=video_id)

            # Determine what channels in each image we want to merge
            image_merge_channels = []
            for coco_img in images.coco_images:
                if requested_chans.spec == '*':
                    # Merge everything
                    mergable = coco_img.channels.fuse().to_set()
                else:
                    mergable = requested_chans.to_set() & coco_img.channels.fuse().to_set()
                image_merge_channels.append(frozenset(mergable))

            video = coco_dset.index.videos[video_id]
            video_name = video['name']
            video_dsize = (video['width'], video['height'])
            print(f'video_dsize={video_dsize}')

            # Filter to only images that have a channel to merge
            flags = [len(c) > 0 for c in image_merge_channels]
            image_merge_channels = list(ub.compress(image_merge_channels, flags))
            images = images.compress(flags)

            image_datetimes = list(map(util_time.coerce_datetime, images.lookup('timestamp')))

            image_unixtimes = np.array([t.timestamp() for t in image_datetimes])

            # Get number of seconds after the start time (or first image)
            if start_time is not None:
                image_rel_unixtimes = image_unixtimes - start_time.timestamp()
            else:
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

            print(f'{len(images)} / {len(flags)} images in {video_name} have {len(groupid_to_idxs)} mergable groups')

            SHOW_INFO = 0
            # DEBUG: Print the distribution of images per window.
            if SHOW_INFO:
                # Get the histogram for the number of images per window.
                bucket_to_num_images = ub.udict(groupid_to_idxs).map_values(len)
                bucket_stats = kwarray.stats_dict(list(bucket_to_num_images.values()))
                print(f'[{video_name}] Distribution of images per window:')
                print('Histogram: = {}'.format(ub.urepr(bucket_to_num_images, nl=1)))
                print('N images per window: = {}'.format(ub.urepr(bucket_stats, nl=1)))

            jobs = ub.JobPool(mode='process', max_workers=workers)

            # 3. For each temporal window, combine the spatial data from each channel.
            chunk_image_idxs = list(groupid_to_idxs.values())
            prog = pman.progiter(chunk_image_idxs, transient=True, desc='Submit combine within temporal windows jobs')
            for chunk_image_idxs in prog:
                # if len(chunk_image_idxs) == 1:
                #     import xdev
                #     xdev.embed()
                # else:
                #     continue
                window_images = images.take(chunk_image_idxs)
                window_coco_images = window_images.coco_images
                # Detach for process parallelization
                window_coco_images = [g.detach() for g in window_coco_images]

                job = jobs.submit(merge_images, window_coco_images,
                                  merge_method, requested_chans, space,
                                  resolution, new_bundle_dpath,
                                  mask_low_quality, sensor_weights,
                                  og_kwcoco_fpath, spatial_tile_size,
                                  config=config)
                job.merge_images = merge_images

            for job in pman.progiter(jobs.as_completed(),
                                     total=len(jobs),
                                     desc='Collect combine within temporal windows jobs'):
                new_img = job.result()
                if new_img is None:
                    n_failed_merges += 1
                    continue
                output_coco_dset.add_image(**new_img)
                n_combined_images += 1
                pman.update_info(ub.codeblock(
                    f'''
                    {n_combined_images=}
                    {n_failed_merges=}
                    '''))

    if n_combined_images == 0:
        raise ValueError('No images were combined with non-NaN values')

    # from geowatch.utils import util_resolution
    from kwcoco.coco_image import coerce_resolution
    target_gsd = float(np.mean(coerce_resolution(config.resolution)['mag']))
    print(f'Reset geowatch feilds target_gsd={target_gsd}')
    kwcoco_extensions.populate_watch_fields(
        output_coco_dset, target_gsd=target_gsd, overwrite=True)

    # video = output_coco_dset.index.videos[video_id]
    # after_video_dsize = (video['width'], video['height'])
    # print(f'after_video_dsize={after_video_dsize}')

    # for video_id in ub.ProgIter(vidids, total=len(vidids), desc='populate videos'):
    #     coco_populate_geo_video_stats(coco_dset, video_id, target_gsd=target_gsd)

    # Debugging:
    if 0:
        kwcoco_extensions.check_kwcoco_spatial_transforms(output_coco_dset)

    # Save kwcoco file.
    print(f"Saving ouput kwcoco file to: {output_kwcoco_fpath}")
    output_coco_dset.validate()
    output_coco_dset.dump()
    print(f"Saved ouput kwcoco file to: {output_kwcoco_fpath}")
    return output_coco_dset


def get_quality_mask(coco_image, space, resolution, avoid_quality_values=None, crop_slice=None):
    """Get a binary mask of the quality data.

    Args:
        coco_image (kwcoco.coco_image.CocoImage): Object that contains references to the image and assets including the quality mask.
        space (str): The space that the quality mask will be loaded in. Choices: 'image', 'video', 'asset'
        resolution (str, int): The resolution that the quality mask will be loaded in. E.g. '10GSD'.
        avoid_quality_values (list, optional): The values to include as bad quality according to the bitmask. Defaults to ['cloud', 'cloud_shadow', 'cloud_adjacent'].
        crop_slice (tuple(slice, slice), optional): The height and width crop slices to load from quality mask. Defaults to None, which loads full quality mask.
    Returns:
        np.ndarray: A binary numpy array of shape [H, W, 1] where the 1 values corresponds to a quality pixel vice versa for 0 values.
    """
    import numpy as np
    delay = coco_image.imdelay('quality',
                               space=space,
                               interpolation='nearest',
                               antialias=False,
                               resolution=resolution)
    if crop_slice:
        delay = delay.crop(crop_slice)
    qa_data = delay.finalize(antialias=False, interpolation='nearest')

    if qa_data.dtype.kind == 'f' or avoid_quality_values is None:
        # If the qa band is a float, then it must be a nan channel
        return np.ones_like(qa_data, dtype=np.uint8)

    from geowatch.tasks.fusion.datamodules.qa_bands import QA_SPECS
    # We don't have the exact right information here, so we can
    # punt for now and assume "Drop4"
    spec_name = 'ACC-1'
    sensor = coco_image.img.get('sensor_coarse', '*')
    try:
        table = QA_SPECS.find_table(spec_name, sensor)
    except AssertionError as ex:
        print(f'warning ex={ex}')
        is_iffy = None
    else:
        is_iffy = table.mask_any(qa_data, avoid_quality_values)

    quality_mask = (1 - is_iffy)

    return quality_mask


def merge_images(window_coco_images, merge_method, requested_chans, space,
                 resolution, new_bundle_dpath, mask_low_quality,
                 sensor_weights, og_kwcoco_fpath, spatial_tile_size, config):
    """
    Args:
        window_coco_images (List[kwcoco.CocoImage]): images with channels to merge

    Returns:
        Dict: a new coco image that points at the merged image on disk.
    """
    import kwimage
    import kwarray
    import kwcoco
    import numpy as np
    from geowatch.tasks.fusion.coco_stitcher import CocoStitchingManager
    from kwutil import util_time

    # Determine what channels are available in this image set
    available_chans_set = set(ub.flatten([g.channels.fuse().to_set() for g in window_coco_images]))

    if requested_chans.spec == '*':
        requested_chans_set = available_chans_set
    else:
        requested_chans_set = requested_chans

    merge_chans_set = requested_chans_set & available_chans_set
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

    video = first_coco_img.video
    video_name = video['name']

    # Scales to the resolution from the requested (i.e. video space)
    scale_asset_from_vid = first_coco_img._scalefactor_for_resolution(resolution=resolution, space='video')
    # warp_asset_from_vid = kwimage.Affine.scale(scale_asset_from_vid)

    video_dsize = kwimage.Box.from_dsize((video['width'], video['height']))

    canvas_dsize = video_dsize.scale(scale_asset_from_vid).quantize().dsize
    canvas_dims = canvas_dsize[::-1]
    video_height, video_width = canvas_dims
    canvas_shape = canvas_dims + (merge_chans.numel(), )

    if all(spatial_tile_size) is False:
        # Set tiles as the same size as the video.
        slider = [(slice(0, video_height, None), slice(0, video_width, None))]
    else:
        # Create tiling slices for given video shape and tile size.

        ## Check that crop size is not bigger than the video size.
        if video_height < spatial_tile_size[0]:
            spatial_height_size = video_height
        else:
            spatial_height_size = spatial_tile_size[0]

        if video_width < spatial_tile_size[1]:
            spatial_width_size = video_height
        else:
            spatial_width_size = spatial_tile_size[1]

        # step = min(spatial_height_size, spatial_width_size)

        slider = kwarray.SlidingWindow(
            shape=(video_height, video_width),
            window=(spatial_height_size, spatial_width_size),
            overlap=0,
            keepbound=True,
            allow_overshoot=True
        )

    # avoid_quality_values = ['cloud', 'cloud_shadow', 'cloud_adjacent']
    avoid_quality_values = ['cloud']
    # avoid_quality_values += ['ice']

    # Create canvas to combine averaged tiles into.
    average_canvas = kwarray.Stitcher(canvas_shape)

    import warnings
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', 'All-NaN')
        warnings.filterwarnings('ignore', 'invalid')

        # Load and combine the images within this range.
        for crop_slice in slider:
            if merge_method == 'mean':
                accum = kwarray.Stitcher(canvas_shape)

                # TODO: we can also parallelize this in case the window is really big
                for coco_img in window_coco_images:
                    delayed = coco_img.imdelay(merge_chans, space=space, resolution=resolution)
                    delayed = delayed.crop(crop_slice)
                    image_data = delayed.finalize(nodata_method='float')

                    pxl_weight = (~np.isnan(image_data)).astype(np.float32)

                    if mask_low_quality:
                        # Load quality mask.
                        quality_mask = get_quality_mask(coco_img, space, resolution, avoid_quality_values=avoid_quality_values, crop_slice=crop_slice)

                        # Update pixel weights based on quality pixel values.
                        pxl_weight *= quality_mask

                    try:
                        sensor_weight = sensor_weights[coco_img['sensor_coarse']]
                    except KeyError:
                        sensor_weight = 1.0
                    pxl_weight *= sensor_weight

                    image_data = np.nan_to_num(image_data)
                    accum.add((slice(None), slice(None)), image_data, pxl_weight)

                combined_image_data = accum.finalize()

            elif merge_method == 'median':
                # TODO: Make this less computationally expensive.
                median_stack = []
                for coco_img in window_coco_images:
                    delayed = coco_img.imdelay(merge_chans, space=space, resolution=resolution)
                    delayed = delayed.crop(crop_slice)
                    image_data = delayed.finalize(nodata_method='float')

                    if mask_low_quality:
                        # Load quality mask.
                        quality_mask = get_quality_mask(coco_img, space, resolution, avoid_quality_values=avoid_quality_values, crop_slice=crop_slice)

                        # Update pixel weights based on quality pixel values.
                        x, y = np.where(quality_mask[..., 0] == 0)
                        image_data[x, y, :] = np.nan

                        # TODO: Fix the logic below to match above because it should be faster.
                        # matched_quality_mask = np.repeat(quality_mask, repeats=3, axis=2)
                        # masked_image_data = np.ma.masked_array(data=image_data2, mask=~matched_quality_mask, fill_value=np.nan)
                        # image_data = M.filled(np.nan)
                        # masked_image_data = M.filled(np.nan)

                    median_stack.append(image_data)

                combined_image_data = np.nanmedian(median_stack, axis=0)

            elif merge_method == 'max':
                # TODO: Combine with other methods.
                median_stack = []
                for coco_img in window_coco_images:
                    delayed = coco_img.imdelay(merge_chans, space=space, resolution=resolution)
                    delayed = delayed.crop(crop_slice)
                    image_data = delayed.finalize(nodata_method='float')
                    if mask_low_quality:
                        # Load quality mask.
                        quality_mask = get_quality_mask(coco_img, space, resolution, avoid_quality_values=avoid_quality_values, crop_slice=crop_slice)
                        # Update pixel weights based on quality pixel values.
                        x, y = np.where(quality_mask[..., 0] == 0)
                        image_data[x, y, :] = np.nan
                    median_stack.append(image_data)
                combined_image_data = np.nanmax(median_stack, axis=0)
            else:
                raise NotImplementedError

            # Add the combined image data to the average canvas.
            average_canvas.add(crop_slice, combined_image_data)

        # Rename the average canvas to the combined image data.
        combined_image_data = average_canvas.finalize()

    # Check if the image contains data after cloud masking.
    if np.all(np.isnan(combined_image_data)):
        return None

    # 4. Save the combined image result to a new kwcoco dataset.
    ## Use the first image as the standin for the new image.

    # Create a dictionary for the new image
    # TODO: Should likely update properties to indicate what went into this new
    # image.
    new_img = first_coco_img.img.copy()
    new_img.pop('auxiliary', None)
    new_img.pop('assets', None)
    new_img.pop('name', None)
    new_img.pop('align_method', None)
    new_img.pop('valid_region', None)
    new_img.pop('valid_region_utm', None)
    new_img.pop('parent_name', None)
    new_img.pop('parent_canonical_name', None)
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
    sensors = '_'.join(sorted(set([coco_img['sensor_coarse'] for coco_img in window_coco_images])))
    new_name = f'ave_{timestr}_{len(window_coco_images):03d}_{chanstr}_{sensors}_{hashid}'
    new_coco_img.img['name'] = new_name

    # TODO: Figure out how to better handle this case.
    # Issue is that the unique sensor combination does not get processed in the predict script.
    new_coco_img.img['sensor_coarse'] = first_coco_img['sensor_coarse']
    new_coco_img.img['_parent_sensors'] = sensors
    new_coco_img.img['_num_parents'] = len(window_coco_images)
    new_coco_img.img['_first_timestamp'] = window_coco_images[0].img['timestamp']
    new_coco_img.img['_last_timestamp'] = window_coco_images[-1].img['timestamp']

    # new_coco_img.img['sensor_coarse'] = '_'.join(
    #     sorted(set([coco_img['sensor_coarse'] for coco_img in window_coco_images])))

    # new_coco_img.bundle_dapth = new_bundle_dpath
    dname = f'{video_name}/ave_{merge_chans.path_sanitize()}'

    # TODO:
    # We could recompute the valid_region and valid_region_utm here
    # as the union of the input items' properties
    new_coco_img.img.pop('valid_region_utm', None)

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
        write_prediction_attrs=False,
        assets_dname=config.assets_dname,
    )

    gid = new_coco_img.img['id']
    stitch_manager.accumulate_image(gid,
                                    None,
                                    combined_image_data,
                                    asset_dsize=canvas_dsize,
                                    scale_asset_from_stitchspace=scale_asset_from_vid)
    stitch_manager.finalize_image(gid)
    final_img = tmp_dset.imgs[gid]

    return final_img


def filter_image_ids_by_season(coco_dset, image_ids, filtered_seasons, ignore_winter_torrid_zone=True):
    """Filter a sequence of image ids by season and geolocation.

    Args:
        coco_dset (kwcoco.CocoDataset): A KWCOCO dataset object.
        image_ids (List(int)): A list of image ids that belong in coco_dset.
        filtered_seasons (str | List(str) | None): Which seasons to not include in the
             returned image ids.
        ignore_winter_torrid_zone (bool, optional): Do not filter images within the
            Torrid region when winter is one of the filtered seasons. Defaults to True.

    Raises:
        ValueError: Check if the filtered seasons varible is a correctable type.
        ValueError: Check if one of the filtered seasons is not a valid season.

    Returns:
        List[int]: A list of filtered image ids. Should never be longer than the input
             image ids.

    Example:
        >>> # 0: Baseline run.
        >>> # xdoctest: +REQUIRES(env:DEVEL_TEST)
        >>> from geowatch.cli.coco_time_combine import *  # NOQA
        >>> import geowatch
        >>> data_dvc_dpath = geowatch.find_dvc_dpath(tags='phase2_data', hardware='auto')
        >>> # Load KWCOCO dataset.
        >>> input_kwcoco_fpath = data_dvc_dpath / 'Drop6/imgonly-AE_C001.kwcoco.json'
        >>> coco_dset = kwcoco.CocoDataset(input_kwcoco_fpath)
        >>> image_ids = coco_dset.images().gids
        >>> ignore_torrid_regions_gids = filter_image_ids_by_season(coco_dset,
        >>>                                image_ids,
        >>>                                filtered_seasons='winter',
        >>>                                ignore_winter_torrid_zone=True)
        >>> all_filtered_gids = filter_image_ids_by_season(coco_dset,
        >>>                       image_ids,
        >>>                       filtered_seasons='winter',
        >>>                       ignore_winter_torrid_zone=True)
        >>> assert len(all_filtered_gids) > len(ignore_torrid_regions_gids)
    """
    from kwutil import util_time
    hemipshere_to_season_map = {
        'northern': {
            'spring': [3, 4, 5],
            'summer': [6, 7, 8],
            'fall': [9, 10, 11],
            'winter': [12, 1, 2]
        },
        'southern': {
            'spring': [9, 10, 11],
            'summer': [12, 1, 2],
            'fall': [3, 4, 5],
            'winter': [6, 7, 8]
        }
    }

    # Check if there are any images to filter.
    if len(image_ids) == 0:
        print('WARNING: No images to filter.')
        return []

    # Check type of filtered_seasons variable and try to convert to list.
    if isinstance(filtered_seasons, str):
        filtered_seasons = [filtered_seasons]
    elif filtered_seasons is None:
        filtered_seasons = []
    elif isinstance(filtered_seasons, list):
        # Remove nones to workaround cli issue
        filtered_seasons = [f for f in filtered_seasons if f is not None]
        pass
    else:
        raise ValueError(f'Filtered seasons must be a list or string. Got "{type(filtered_seasons)}" type.')

    # Get seasons to filter (invalid seasons).
    filtered_seasons = set(filtered_seasons)
    valid_seasons = {'spring', 'summer', 'fall', 'winter'}

    invalid_seasons = filtered_seasons - valid_seasons
    if invalid_seasons:
        raise ValueError(f'Invalid seasons: {invalid_seasons}')

    # Get hemisphere of region.
    coco_img = coco_dset.coco_image(image_ids[0])
    import numpy as np
    geo_corner_coords = np.asarray(coco_img.img['geos_corners']['coordinates'])
    center_coord = geo_corner_coords.mean(axis=1)[0]
    _, lat = center_coord[0], center_coord[1]

    # Check if the latitude is in the northern or southern hemisphere.
    if lat > 0:
        hemisphere = 'northern'
    else:
        hemisphere = 'southern'

    # Check if the region is within the torrid zone.
    # Torrid zone is generally warm and less likely to contain snow.
    # https://en.wikipedia.org/wiki/Geographical_zone
    # https://www.toppr.com/guides/chemistry/environmental-chemistry/torrid-zone/
    # Should technically be 23.5 but bumpping up to 27 to include AE regions
    within_torrid_zone = abs(lat) < 27

    month_to_season_map = {}
    for season, months in hemipshere_to_season_map[hemisphere].items():
        for month in months:
            month_to_season_map[month] = season

    final_image_ids = []
    for image_id in image_ids:
        coco_img = coco_dset.coco_image(image_id)

        # Get month image was taken in.
        dt = util_time.coerce_datetime(coco_img['date_captured'])
        month = dt.month

        # Get season of month (depends on northern or southern hemisphere).
        img_season = month_to_season_map[month]

        # Do not filter image if ALL conditions are met:
        # 1) We are not removing images within the torrid zone during winter,
        # 2) the region latitude is within the torrid zone (|lat| < 23.5),
        # 3) winter is in the filtered seasons,
        # 4) the image was captured during winter.
        if ignore_winter_torrid_zone and within_torrid_zone and (img_season == 'winter') and ("winter" in filtered_seasons):
            final_image_ids.append(image_id)
        else:
            # Check if the season is not in the filtered seasons.
            if img_season not in filtered_seasons:
                final_image_ids.append(image_id)

    return final_image_ids


__config__ = TimeCombineConfig

if __name__ == '__main__':
    """

    CommandLine:
        python -m geowatch.cli.coco_time_combine
    """
    main()
