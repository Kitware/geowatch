import os
import kwcoco
import kwarray
import kwimage
import numpy as np
import ubelt as ub
# from tqdm import tqdm
# from watch import exceptions
# from watch.tasks.fusion.predict import quantize_float01
# from watch.utils.kwcoco_extensions import transfer_geo_metadata
import scriptconfig as scfg
from watch.utils import util_time
from watch.utils import util_progress


class TimeAverageConfig(scfg.DataConfig):
    kwcoco_fpath = '/data4/datasets/dvc-repos/smart_data_dvc/Drop4-BAS/baseline_fusion_valid_M2.kwcoco.json'
    output_kwcoco_fpath = '/data4/datasets/dvc-repos/smart_data_dvc/Drop4-BAS/M2_time_merge_1month_mean_10GSD.kwcoco.json'
    channels = 'salient'
    temporal_window_duration = '1month'
    merge_method = 'mean'
    resolution = '10GSD'
    workers = 0


def main(cmdline=1, **kwargs):
    """
    CommandLine:
        DEVEL_TEST=1 xdoctest -m watch.cli.coco_temporally_combine_channels main

    Example:
        >>> # xdoctest: +REQUIRES(env:DEVEL_TEST)
        >>> from watch.cli.coco_temporally_combine_channels import *  # NOQA
        >>> import watch
        >>> data_dvc_dpath = watch.find_dvc_dpath(tags='phase2_data', hardware='auto')
        >>> cmdline = 0
        >>> kwargs = dict(
        >>>     kwcoco_fpath=data_dvc_dpath / 'Drop6/imgonly-KR_R001.kwcoco.json',
        >>>     output_kwcoco_fpath=data_dvc_dpath / 'Drop6/test-timeave-imgonly-KR_R001.kwcoco.json',
        >>>     workers=16,
        >>>     temporal_window_duration='1 year',
        >>>     channels='red|green|blue',
        >>> )
        >>> output_coco_dset = main(cmdline=cmdline, **kwargs)

    Ignore:
        DVC_DATA_DPATH=$(smartwatch_dvc --tags='phase2_data' --hardware=auto)
        smartwatch stats $DVC_DATA_DPATH/Drop6/test-timeave-imgonly-KR_R001.kwcoco.json
        smartwatch visualize $DVC_DATA_DPATH/Drop6/test-timeave-imgonly-KR_R001.kwcoco.json --smart=True
    """
    config = TimeAverageConfig.cli(cmdline=cmdline, data=kwargs, strict=True)
    print('config = ' + ub.urepr(dict(config), nl=1))

    if 0:
        globals().update(**config)

    output_coco_dset = combine_kwcoco_channels_temporally(**config)
    return output_coco_dset

    # config = TimeAverageConfig(
    #     kwcoco_fpath='/data4/datasets/dvc-repos/smart_data_dvc/Drop4-BAS/baseline_fusion_valid_M2.kwcoco.json',
    #     output_kwcoco_fpath='/data4/datasets/dvc-repos/smart_data_dvc/Drop4-BAS/M2_time_merge_1month_mean_5GSD.kwcoco.json',
    #     channel_name='salient',
    #     temporal_window_duration=365 / 12,  # 1 month on average.
    #     merge_method='mean',
    #     resolution='5GSD',
    # )
    # combine_kwcoco_channels_temporally(**config)

    # kwcoco_fpath = '/data4/datasets/dvc-repos/smart_data_dvc/Drop4-BAS/baseline_fusion_valid_M3.kwcoco.json'
    # output_kwcoco_fpath = '/data4/datasets/dvc-repos/smart_data_dvc/Drop4-BAS/M3_time_merge_1month_mean_15GSD.kwcoco.json'
    # channel_name = 'salient'
    # temporal_window_duration = 365 / 12  # 1 month on average.
    # merge_method = 'mean'
    # resolution = '15GSD'
    # combine_kwcoco_channels_temporally(kwcoco_fpath, output_kwcoco_fpath, channel_name, temporal_window_duration,
    #                                    merge_method, resolution)

    # Baseline: Year
    # kwcoco_fpath = '/data4/datasets/dvc-repos/smart_data_dvc/Drop4-BAS/baseline_fusion_valid_M2.kwcoco.json'
    # output_kwcoco_fpath = '/data4/datasets/dvc-repos/smart_data_dvc/Drop4-BAS/M2_time_merge_1year_mean_10GSD.kwcoco.json'
    # channel_name = 'salient'
    # temporal_window_duration = 365
    # merge_method = 'mean'
    # resolution = '10GSD'
    # combine_kwcoco_channels_temporally(kwcoco_fpath, output_kwcoco_fpath, channel_name, temporal_window_duration,
    #                                    merge_method, resolution)

    # kwcoco_fpath = '/data4/datasets/dvc-repos/smart_data_dvc/Drop4-BAS/baseline_fusion_valid_M3.kwcoco.json'
    # output_kwcoco_fpath = '/data4/datasets/dvc-repos/smart_data_dvc/Drop4-BAS/M3_time_merge_1year_mean_15GSD.kwcoco.json'
    # channel_name = 'salient'
    # temporal_window_duration = 365
    # merge_method = 'mean'
    # resolution = '15GSD'
    # combine_kwcoco_channels_temporally(kwcoco_fpath, output_kwcoco_fpath, channel_name, temporal_window_duration,
    #                                    merge_method, resolution)

    # # Baseline: 1/2 year
    # kwcoco_fpath = '/data4/datasets/dvc-repos/smart_data_dvc/Drop4-BAS/baseline_fusion_valid_M2.kwcoco.json'
    # output_kwcoco_fpath = '/data4/datasets/dvc-repos/smart_data_dvc/Drop4-BAS/M2_time_merge_6month_mean_10GSD.kwcoco.json'
    # channel_name = 'salient'
    # temporal_window_duration = 365 / 2
    # merge_method = 'mean'
    # resolution = '10GSD'
    # combine_kwcoco_channels_temporally(kwcoco_fpath, output_kwcoco_fpath, channel_name, temporal_window_duration,
    #                                    merge_method, resolution)

    # kwcoco_fpath = '/data4/datasets/dvc-repos/smart_data_dvc/Drop4-BAS/baseline_fusion_valid_M3.kwcoco.json'
    # output_kwcoco_fpath = '/data4/datasets/dvc-repos/smart_data_dvc/Drop4-BAS/M3_time_merge_6month_mean_15GSD.kwcoco.json'
    # channel_name = 'salient'
    # temporal_window_duration = 365 / 2
    # merge_method = 'mean'
    # resolution = '15GSD'
    # combine_kwcoco_channels_temporally(kwcoco_fpath, output_kwcoco_fpath, channel_name, temporal_window_duration,
    #                                    merge_method, resolution)


def combine_kwcoco_channels_temporally(kwcoco_fpath, output_kwcoco_fpath,
                                       channels, temporal_window_duration,
                                       merge_method, resolution, workers):
    """Combine spatial data within a temporal window from a kwcoco dataset and save the result to a new kwcoco dataset.

    High level steps:
    1. Load kwcoco dataset.
    2. Divide the dataset into temporal windows.
    3. For each temporal window, combine the spatial data from each channel.
    4. Save the combined image result to a new kwcoco dataset.

    Args:
        kwcoco_fpath (str): _description_
        output_kwcoco_fpath (str): _description_
        channels (str): _description_
        temporal_window_duration (int): How many days the window should be.
        merge_method (str): _description_
        resolution (str): _description_
    """
    # Check inputs.
    space = 'video'
    # space = 'image'

    ## Check input kwcoco file path exists.
    if os.path.exists(kwcoco_fpath) is False:
        raise FileNotFoundError(f'Input kwcoco file path does not exist: {kwcoco_fpath}')

    # 1. Load kwcoco dataset.
    coco_dset = kwcoco.CocoDataset.coerce(kwcoco_fpath)
    output_coco_dset = coco_dset.copy()
    output_coco_dset.clear_images()  # we will write all new images

    requested_sensorchan = kwcoco.SensorChanSpec.coerce(channels)
    requested_chans = requested_sensorchan.chans

    ## Get saved asset directory.
    output_kwcoco_fpath = ub.Path(output_kwcoco_fpath)
    save_assest_dir = (output_kwcoco_fpath.parent / "_assets").ensuredir()

    # 2. Divide the dataset into temporal windows (per video).

    ## Convert temporal_window_duration from days to seconds.
    time_delta = util_time.coerce_timedelta(temporal_window_duration)
    time_delta_seconds = time_delta.total_seconds()

    video_ids = [vid for vid in coco_dset.videos()]

    pman = util_progress.ProgressManager(backend='rich')
    # pman = util_progress.ProgressManager(backend='progiter')  # uncomment if you need to embed

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

            image_datetimes = list(map(util_time.coerce_datetime,
                                       images.lookup('timestamp')))

            image_unixtimes = np.array([t.timestamp() for t in image_datetimes])

            # Get number of seconds after the first image
            image_rel_unixtimes = image_unixtimes - image_unixtimes[0]

            # We might group via multiple criteria
            groupers = {}

            # Assign each image to a temporal bucket/window
            image_buckets = np.floor(image_rel_unixtimes / time_delta_seconds).astype(int)
            groupers['bucket'] = image_buckets

            # groupers['channels'] = image_merge_channels  # not sure if we do this or not

            RESPECT_SENSOR = True
            if RESPECT_SENSOR:
                image_sensors = images.lookup('sensor_coarse', None)
                groupers['sensor'] = image_sensors

            # Construct a group-id for each image
            image_groupids = list(zip(*groupers.values()))
            # Group all image-indexes within the same group-id together.
            unique_groupids, grouped_idxs = kwarray.group_indices(image_groupids)
            groupid_to_idxs = dict(zip(unique_groupids, grouped_idxs))

            # Print the distribution of images per window.
            if 0:
                region_name = coco_dset.index.videos[vid]['name']
                # Get the histogram for the number of images per window.
                bucket_to_num_images = ub.udict(groupid_to_idxs).map_values(len)
                bucket_stats = kwarray.stats_dict(list(bucket_to_num_images.values()))
                print(f'[{region_name}] Distribution of images per window:')
                print('Histogram: = {}'.format(ub.urepr(bucket_to_num_images, nl=1)))
                print('N images per window: = {}'.format(ub.urepr(bucket_stats, nl=1)))

            jobs = ub.JobPool(mode='process', max_workers=workers)

            # 3. For each temporal window, combine the spatial data from each channel.
            chunk_image_idxs = list(groupid_to_idxs.values())
            prog = pman.progiter(chunk_image_idxs, transient=True,
                                 desc='Submit combine within temporal windows jobs')
            for chunk_image_idxs in prog:
                window_image = images.take(chunk_image_idxs)
                window_coco_images = window_image.coco_images
                # Detach for process parallelization
                window_coco_images = [g.detach() for g in window_coco_images]
                jobs.submit(merge_images, window_coco_images, merge_method,
                            requested_chans, space, resolution,
                            save_assest_dir)

            for job in pman.progiter(jobs.as_completed(), total=len(jobs),
                                     desc='Collect combine within temporal windows jobs'):
                new_img = job.result()
                output_coco_dset.add_image(**new_img)

    # Save kwcoco file.
    print(f"Saving ouput kwcoco file to: {output_kwcoco_fpath}")
    output_coco_dset.validate()
    output_coco_dset.dump(output_kwcoco_fpath)
    print(f"Saved ouput kwcoco file to: {output_kwcoco_fpath}")
    return output_coco_dset


def merge_images(window_coco_images, merge_method, requested_chans, space, resolution, save_assest_dir):
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
    scale = first_coco_img._scalefactor_for_resolution(resolution='10GSD', space='video')
    canvas_dsize = kwimage.Box.from_dsize((first_coco_img.video['width'], first_coco_img.video['height'])).scale(scale).quantize().dsize
    canvas_dims = canvas_dsize[::-1]
    canvas_shape = canvas_dims + (merge_chans.numel(),)

    # Load and combine the images within this range.
    if merge_method == 'mean':
        accum = kwarray.Stitcher(canvas_shape)

        # TODO: we can also parallelize this in case the window is really big
        for coco_img in window_coco_images:
            delayed = coco_img.imdelay(merge_chans, space=space, resolution=resolution)
            image_data = delayed.finalize(nodata_method='float')
            pxl_weight = (1 - np.isnan(image_data))
            image_data = np.nan_to_num(image_data)
            accum.add((slice(None), slice(None)), image_data, pxl_weight)

            # if __debug__:
            #     axes[count].imshow(np.nan_to_num(image_data), vmin=0, vmax=0.5)
            #     count += 1
        import warnings
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', 'invalid')
            combined_image_data = accum.finalize()
        # if __debug__:
        #     axes[count].imshow(np.nan_to_num(combined_image_data), vmin=0, vmax=0.5)
        #     axes[count].set_title('combined')
        #     plt.savefig(f'debug_temp_combine_{space}_{region_name}_{w_index}_{resolution}_new.png')
        #     w_index += 1

        #     if w_index == 20:
        #         return None

    elif merge_method == 'median':
        raise NotImplementedError
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
    new_warp_img_from_asset = first_coco_img.warp_img_from_vid

    # Create the same image name.
    hash_depends = {
        'names': [g.img['name'] for g in window_coco_images],
        'chans': merge_chans.spec,
        'resolution': resolution,
        'space': space,
        'version': 0,
    }
    hashid = ub.hash_data(hash_depends)[0:16]

    timestr = util_time.isoformat(util_time.coerce_datetime(new_coco_img.img['timestamp']))
    chanstr = merge_chans.path_sanitize()
    new_name = f'ave_{timestr}_{len(window_coco_images):03d}_{chanstr}_{hashid}'
    new_coco_img.img['name'] = new_name
    dname = f'ave_{merge_chans.path_sanitize()}'
    average_fpath = (save_assest_dir / dname).ensuredir() / new_name + '.tif'

    new_coco_img.add_asset(
        file_name=average_fpath,
        channels=merge_chans,
        warp_aux_to_img=new_warp_img_from_asset,
        width=combined_image_data.shape[1],
        height=combined_image_data.shape[0],
    )
    new_asset = new_coco_img.img['auxiliary'][-1]

    DO_QUANTIZATION = 0
    if DO_QUANTIZATION:
        # TODO: quantization
        # NOTE: We will be able to use CocoStitchingManager after quantization bugs are fixed.
        quantization = ...
        new_asset['quantization'] = quantization
        nodata = quantization['nodata']
        writeable_imdata = ...
    else:
        writeable_imdata = combined_image_data
        nodata = None
        ...
    # Get default kwargs from ../tasks/fusion/predict.py:CocoStitchingManager.finalize_image
    write_kwargs = {}
    write_kwargs['blocksize'] = 128
    write_kwargs['compress'] = 'DEFLATE'

    # TODO: Possibly add `wld_crs_info` to write_kwargs.
    # Write the averaged image data
    kwimage.imwrite(
        average_fpath, writeable_imdata, backend="gdal",
        nodata=nodata,
        **write_kwargs)

    # Update all channels with projection info.
    # try:
    #     transfer_geo_metadata(output_coco_dset, first_gid)
    # except exceptions.GeoMetadataNotFound as ex:
    #     print("warning ex = {!r}".format(ex))
    #     pass
    return new_coco_img.img


if __name__ == '__main__':
    """

    CommandLine:
        python -m watch.cli.coco_temporally_combine_channels
    """
    main()
