import os
import hashlib

import kwcoco
import kwarray
import kwimage
import numpy as np
import ubelt as ub
from tqdm import tqdm

from watch import exceptions
from watch.tasks.fusion.predict import quantize_float01
from watch.utils.kwcoco_extensions import transfer_geo_metadata


def create_image_name(image_name, kwcoco_file_path, output_kwcoco_path, channel_name, temporal_window_duration,
                      merge_method, resolution):
    str_to_encode = f"{image_name}{kwcoco_file_path}{output_kwcoco_path}{channel_name}{str(temporal_window_duration)}{merge_method}{resolution}"
    hashed_str = hashlib.sha256(str_to_encode.encode('utf-8')).hexdigest()
    return hashed_str


def combine_kwcoco_channels_temporally(kwcoco_file_path, output_kwcoco_path, channel_name, temporal_window_duration,
                                       merge_method, resolution):
    """Combine spatial data within a temporal window from a kwcoco dataset and save the result to a new kwcoco dataset.

    High level steps:
    1. Load kwcoco dataset.
    2. Divide the dataset into temporal windows.
    3. For each temporal window, combine the spatial data from each channel.
    4. Save the combined image result to a new kwcoco dataset.

    Args:
        kwcoco_file_path (str): _description_
        output_kwcoco_path (str): _description_
        channel_name (str): _description_
        temporal_window_duration (int): How many days the window should be.
        merge_method (str): _description_
        resolution (str): _description_    
    """
    # Check inputs.
    space = 'video'
    # space = 'image'

    ## Check input kwcoco file path exists.
    if os.path.exists(kwcoco_file_path) is False:
        raise FileNotFoundError(f'Input kwcoco file path does not exist: {kwcoco_file_path}')

    # 1. Load kwcoco dataset.
    coco_dset = kwcoco.CocoDataset.coerce(kwcoco_file_path)
    output_coco_dset = coco_dset.copy()

    ## Get saved asset directory.
    output_kwcoco_path = ub.Path(output_kwcoco_path)
    save_assest_dir = (output_kwcoco_path.parent / "_assets").ensuredir()

    # 2. Divide the dataset into temporal windows (per video).

    ## Convert temporal_window_duration from days to seconds.
    time_delta_sec = temporal_window_duration * 24 * 60 * 60

    video_ids = [vid for vid in coco_dset.videos()]

    image_ids_to_remove = []
    for vid in tqdm(video_ids, colour='green', desc='Combining channel info within temporal windows'):
        # Get all image ids for the video.
        image_ids = coco_dset.index.vidid_to_gids[vid]

        # Get the range of dates for the video.
        early_timestamp = coco_dset.index.imgs[image_ids[0]]['timestamp']
        late_timestamp = coco_dset.index.imgs[image_ids[-1]]['timestamp']
        n_windows = int(np.ceil((late_timestamp - early_timestamp) / time_delta_sec))

        # Map timestamps of image ids to time windows.
        chunk_image_ids = [[] for _ in range(n_windows)]
        for gid in image_ids:
            # Get the timestamp for the image.
            timestamp = coco_dset.index.imgs[gid]['timestamp']

            # Get the window index for the image.
            window_index = int(np.floor((timestamp - early_timestamp) / time_delta_sec))

            # Add the image id to the window.
            chunk_image_ids[window_index].append(gid)

        # Print the distribution of images per window.
        # if __debug__:
        #     region_name = coco_dset.index.videos[vid]['name']
        #     # Get the histogram for the number of images per window.
        #     window_sizes = [len(chunk_image_ids[i]) for i in range(n_windows)]
        #     hist, indices = np.histogram(window_sizes, bins=range(0, max(window_sizes) + 1))
        #     print(f'[{region_name}] Distribution of images per window:')
        #     print(f'N images per window: {indices}')
        #     print(f'Histogram:           {hist}')

        # 3. For each temporal window, combine the spatial data from each channel.

        # Get the video size to use for the output images.
        coco_img = coco_dset.coco_image(image_ids[0])
        delayed = coco_img.delay(channel_name, space=space, resolution=resolution)
        video_image_shape = delayed.shape

        # if __debug__:
        #     w_index = 0
        for window_image_ids in tqdm(chunk_image_ids,
                                     desc='Combining channel info within temporal windows',
                                     colour='green',
                                     leave=False):
            if len(window_image_ids) == 0:
                # Skip empty windows.
                continue
            else:
                # Load and combine the images within this range.
                # if __debug__:
                #     import matplotlib.pyplot as plt
                #     _, axes = plt.subplots(1, len(window_image_ids) + 1)
                #     count = 0

                if merge_method == 'mean':
                    # accum = kwarray.Stitcher(video_image_shape)
                    new_accum = np.zeros(video_image_shape, dtype=float)
                    for gid in window_image_ids:
                        coco_img = coco_dset.coco_image(gid)
                        delayed = coco_img.delay(channel_name, space=space, resolution=resolution)
                        image_data = delayed.finalize(nodata_method='float')

                        pxl_weight = (1 - np.isnan(image_data))
                        new_accum += np.nan_to_num(image_data) * pxl_weight
                        # accum.add((slice(None), slice(None)), image_data, pxl_weight)

                        # if __debug__:
                        #     axes[count].imshow(np.nan_to_num(image_data), vmin=0, vmax=0.5)
                        #     count += 1

                    # combined_image_data = accum.finalize()
                    combined_image_data = new_accum / len(window_image_ids)
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
            ## Overwrite the image data for the first image in the window.
            ## Then delete the other images in the window.
            first_gid = window_image_ids[0]

            ## Add the other images to the remove list.
            if len(window_image_ids) > 1:
                image_ids_to_remove.extend(window_image_ids[1:])
            else:
                pass

            scale_target_from_vid = kwimage.Affine.scale(
                coco_img._scalefactor_for_resolution(space=space, channel=channel_name, resolution=resolution))

            warp_target_from_img = scale_target_from_vid @ coco_img.warp_vid_from_img
            warp_img_from_target = warp_target_from_img.inv()

            quant_data, quantization = quantize_float01(combined_image_data)

            combined_coco_img = output_coco_dset.coco_image(first_gid)
            output_obj = combined_coco_img.find_asset_obj(channel_name)

            # Create the same image name.
            img_name = combined_coco_img.img.get("name", "")
            average_fname = create_image_name(img_name, kwcoco_file_path, output_kwcoco_path, channel_name,
                                              temporal_window_duration, merge_method, resolution)
            average_fpath = save_assest_dir / average_fname

            if output_obj is not None:
                # Overwrite the data in the output auxiliary item.
                output_obj["file_name"] = os.fspath(average_fpath)
                output_obj['height'] = combined_image_data.shape[0]
                output_obj['width'] = combined_image_data.shape[1]
                if len(combined_image_data.shape) > 2:
                    # average_image_data: (H, W, C)
                    output_obj['num_bands'] = combined_image_data.shape[-1]
                else:
                    # average_image_data: (H, W)
                    output_obj['num_bands'] = 1
                output_obj['warp_aux_to_img'] = warp_img_from_target.concise()
                output_obj['quantization'] = quantization

            # Get default kwargs from ../tasks/fusion/predict.py:CocoStitchingManager.finalize_image
            write_kwargs = {}
            write_kwargs['blocksize'] = 128
            write_kwargs['compress'] = 'DEFLATE'

            # TODO: Possibly add `wld_crs_info` to write_kwargs.

            # Write the averaged image data
            kwimage.imwrite(average_fpath, quant_data, backend="gdal", nodata=quantization['nodata'], **write_kwargs)

            # Update all channels with projection info.
            try:
                transfer_geo_metadata(output_coco_dset, first_gid)
            except exceptions.GeoMetadataNotFound as ex:
                print("warning ex = {!r}".format(ex))
                pass

    # Get rid of image_ids without combined data.
    output_coco_dset.remove_images(image_ids_to_remove)

    # Save kwcoco file.
    output_coco_dset.validate()
    output_coco_dset.dump(output_kwcoco_path)
    print(f"Saved ouput kwcoco file to: {output_kwcoco_path}")


def main():
    # Baseline: Month
    kwcoco_path = '/data4/datasets/dvc-repos/smart_data_dvc/Drop4-BAS/baseline_fusion_valid_M2.kwcoco.json'
    output_kwcoco_path = '/data4/datasets/dvc-repos/smart_data_dvc/Drop4-BAS/M2_time_merge_1month_mean_10GSD.kwcoco.json'
    channel_name = 'salient'
    temporal_window_duration = 365 / 12  # 1 month on average.
    merge_method = 'mean'
    resolution = '10GSD'
    combine_kwcoco_channels_temporally(kwcoco_path, output_kwcoco_path, channel_name, temporal_window_duration,
                                       merge_method, resolution)

    kwcoco_path = '/data4/datasets/dvc-repos/smart_data_dvc/Drop4-BAS/baseline_fusion_valid_M2.kwcoco.json'
    output_kwcoco_path = '/data4/datasets/dvc-repos/smart_data_dvc/Drop4-BAS/M2_time_merge_1month_mean_5GSD.kwcoco.json'
    channel_name = 'salient'
    temporal_window_duration = 365 / 12  # 1 month on average.
    merge_method = 'mean'
    resolution = '5GSD'
    combine_kwcoco_channels_temporally(kwcoco_path, output_kwcoco_path, channel_name, temporal_window_duration,
                                       merge_method, resolution)

    # kwcoco_path = '/data4/datasets/dvc-repos/smart_data_dvc/Drop4-BAS/baseline_fusion_valid_M3.kwcoco.json'
    # output_kwcoco_path = '/data4/datasets/dvc-repos/smart_data_dvc/Drop4-BAS/M3_time_merge_1month_mean_15GSD.kwcoco.json'
    # channel_name = 'salient'
    # temporal_window_duration = 365 / 12  # 1 month on average.
    # merge_method = 'mean'
    # resolution = '15GSD'
    # combine_kwcoco_channels_temporally(kwcoco_path, output_kwcoco_path, channel_name, temporal_window_duration,
    #                                    merge_method, resolution)

    # Baseline: Year
    # kwcoco_path = '/data4/datasets/dvc-repos/smart_data_dvc/Drop4-BAS/baseline_fusion_valid_M2.kwcoco.json'
    # output_kwcoco_path = '/data4/datasets/dvc-repos/smart_data_dvc/Drop4-BAS/M2_time_merge_1year_mean_10GSD.kwcoco.json'
    # channel_name = 'salient'
    # temporal_window_duration = 365
    # merge_method = 'mean'
    # resolution = '10GSD'
    # combine_kwcoco_channels_temporally(kwcoco_path, output_kwcoco_path, channel_name, temporal_window_duration,
    #                                    merge_method, resolution)

    # kwcoco_path = '/data4/datasets/dvc-repos/smart_data_dvc/Drop4-BAS/baseline_fusion_valid_M3.kwcoco.json'
    # output_kwcoco_path = '/data4/datasets/dvc-repos/smart_data_dvc/Drop4-BAS/M3_time_merge_1year_mean_15GSD.kwcoco.json'
    # channel_name = 'salient'
    # temporal_window_duration = 365
    # merge_method = 'mean'
    # resolution = '15GSD'
    # combine_kwcoco_channels_temporally(kwcoco_path, output_kwcoco_path, channel_name, temporal_window_duration,
    #                                    merge_method, resolution)

    # # Baseline: 1/2 year
    # kwcoco_path = '/data4/datasets/dvc-repos/smart_data_dvc/Drop4-BAS/baseline_fusion_valid_M2.kwcoco.json'
    # output_kwcoco_path = '/data4/datasets/dvc-repos/smart_data_dvc/Drop4-BAS/M2_time_merge_6month_mean_10GSD.kwcoco.json'
    # channel_name = 'salient'
    # temporal_window_duration = 365 / 2
    # merge_method = 'mean'
    # resolution = '10GSD'
    # combine_kwcoco_channels_temporally(kwcoco_path, output_kwcoco_path, channel_name, temporal_window_duration,
    #                                    merge_method, resolution)

    # kwcoco_path = '/data4/datasets/dvc-repos/smart_data_dvc/Drop4-BAS/baseline_fusion_valid_M3.kwcoco.json'
    # output_kwcoco_path = '/data4/datasets/dvc-repos/smart_data_dvc/Drop4-BAS/M3_time_merge_6month_mean_15GSD.kwcoco.json'
    # channel_name = 'salient'
    # temporal_window_duration = 365 / 2
    # merge_method = 'mean'
    # resolution = '15GSD'
    # combine_kwcoco_channels_temporally(kwcoco_path, output_kwcoco_path, channel_name, temporal_window_duration,
    #                                    merge_method, resolution)


if __name__ == '__main__':
    main()
