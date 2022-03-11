import os
import argparse

import kwcoco
import kwimage
import numpy as np
from tqdm import tqdm
from osgeo import gdal

from watch.utils.kwcoco_extensions import transfer_geo_metadata


def check_kwcoco_file(kwcoco_path, channel_name, sensor_name=None):
    """Make sure that kwcoco files exist and contain required channel name.

    Args:
        kwcoco_path (str): Path to local kwcoco file.
        channel_name (str): Name of channel thats required to be in kwcoco file.
        sensor_name (str, optional): Only check images of from this type of sensor. Defaults to None.
    """

    # Check if kwcoco file exists.
    if os.path.isfile(kwcoco_path) is False:
        raise FileNotFoundError(f"KWCOCO file not found at {kwcoco_path}")

    # Load kwcoco file.
    kwcoco_file = kwcoco.CocoDataset(kwcoco_path)

    # Get all images in kwcoco file.
    images: kwcoco.coco_dataset.Videos = kwcoco_file.images()

    if sensor_name is not None:
        # Filter to only images with a chosen sensor
        flags = [s == sensor_name for s in images.lookup("sensor_coarse", None)]
        images = images.compress(flags)

    for coco_img in images.coco_images:
        if channel_name not in list(coco_img.channels.keys()):
            raise AssertionError(
                f"Channel '{channel_name}' not found in image {coco_img.img['id']} of kwcoco file {kwcoco_path}. Only channels found: {coco_img.channels}"
            )


def merge_kwcoco_channels(
    kwcoco_file_paths, output_kwcoco_path, channel_names, weights, merged_channel_name, sensor_name=None
):
    """Compute a weighted mean of channels from separate kwcoco file and save into merged kwcoco file.

    Args:
        kwcoco_file_paths (list(str)): A list of paths representing pathes to kwcoco files to be merged.
        output_kwcoco_path (str): Local path to the kwcoco file with merged channels.
        channel_names (list(str)): A list of channel names corresponding to the channel name to merge from each kwcoco
            file. Note, the length of the channel names be equal to the number of kwcoco file paths.
        weights (list(int)): A list of floats representing how much weight a particular kwcoco file should contribute
            to the final merged prediction.
        sensor_name (str, optional): Only merge images belonging to this sensor. Defaults to None.
    """
    # Load and merge images from kwcoco files.
    ## Load kwcoco files.
    kwcoco_files = [kwcoco.CocoDataset(kwcoco_file_path) for kwcoco_file_path in kwcoco_file_paths]

    ## Create output kwcoco by copying first kwcoco file.
    merge_kwcoco = kwcoco_files[0].copy()

    ## Load channel images from each viable image_id.
    save_assest_dir = os.path.join(".".join(output_kwcoco_path.split(".")[:-2]), "_assests")
    os.makedirs(save_assest_dir, exist_ok=True)
    pbar = tqdm(merge_kwcoco.index.imgs.items(), desc="Merging images", colour="green")
    for image_id, image_info in pbar:
        # If sensor name spacified, only merge channels for images from this sensor.
        if sensor_name is not None:
            if image_info["sensor_coarse"] != sensor_name:
                continue

        # Load channel image from each kwcoco file.
        weighted_images = []
        for kwcoco_index, kwcoco_file in enumerate(kwcoco_files):
            # Get the channel band.
            delayed_image = kwcoco_file.delayed_load(image_id, channels=channel_names[kwcoco_index], space="video")
            image = delayed_image.finalize(as_xarray=False)
            weighted_images.append(image * weights[kwcoco_index])

        # Combine images using weight factors.
        merged_image = np.add.reduce(weighted_images) / sum(weights)  # [height, width, n_channels]

        # Save merged image to disk and onto kwcoco file.
        ## Save merged image onto disk.
        save_path = os.path.join(save_assest_dir, str(image_id) + "_merged.tif")
        kwimage.imwrite(save_path, merged_image, backend="gdal")

        ## Get project and geo info.
        unmerged_image = merge_kwcoco.index.imgs[image_id]
        vid_from_img = kwimage.Affine.coerce(unmerged_image["warp_img_to_vid"])
        img_from_vid = vid_from_img.inv()
        unmerged_image.get("auxiliary", []).append(
            {
                "file_name": save_path,
                "channels": merged_channel_name,
                "height": merged_image.shape[0],
                "width": merged_image.shape[1],
                "num_bands": 2,
                "warp_aux_to_img": img_from_vid.concise(),
            }
        )
        merge_kwcoco.index.imgs[image_id] = unmerged_image

        # Update all channels with projection info.
        transfer_geo_metadata(merge_kwcoco, image_id)

    # Save kwcoco file.
    merge_kwcoco.validate()
    merge_kwcoco.dump(output_kwcoco_path)
    print(f"Saved merged kwcoco file to: {output_kwcoco_path}")


def main(cmdline=True):
    """
    Example call:
    python watch/cli/coco_merge_features.py --kwcoco_file_paths \
         /data4/datasets/smart_watch_dvc/Drop2-Aligned-TA1-2022-02-15/output_iarpa_drop2v2_total_bin_change_early_fusion_0014.kwcoco.json \
         /data4/datasets/smart_watch_dvc/Drop2-Aligned-TA1-2022-02-15/output_iarpa_drop2v2_total_bin_change_early_fusion_0013.kwcoco.json  \
         --output_kwcoco_path /data4/datasets/smart_watch_dvc/Drop2-Aligned-TA1-2022-02-15/test_comb.kwcoco.json \
         --channel_name "not_salient|salient" --sensor S2
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--kwcoco_file_paths",
        type=str,
        nargs="+",
        required=True,
        help="Path to at least two kwcoco paths with predictions.",
    )
    parser.add_argument(
        "--output_kwcoco_path", type=str, required=True, help="Path to the combined features kwcoco file."
    )
    parser.add_argument(
        "--channel_name",
        type=str,
        nargs="+",
        required=True,
        help="Name of the channel in kwcoco files to merge.",
    )
    parser.add_argument(
        "--weights",
        type=float,
        help="Combination weight value for each prediction from kwcoco file. Default: All predictions are equally weighted.",
    )
    parser.add_argument(
        "--sensor",
        type=str,
        default="all",
        choices=["all", "S2", "L8", "WV"],
        help="Only merge channels from this type of sensor.",
    )
    parser.add_argument(
        "--merge_channel_name",
        type=str,
        help="Name of channel with merged features. By default using the first input argument channel_name.",
    )
    args = parser.parse_args()

    # Check input arguments.
    ## Must have at least two kwcoco files.
    if len(args.kwcoco_file_paths) <= 1:
        raise ValueError(f"Need at least 2 kwcoco files to merge and only recieved {len(args.kwcoco_file_paths)}")

    ## If weights argument is not None then make sure it is the same length as kwcoco files.
    if args.weights is not None:
        if len(args.weights) != len(args.kwcoco_file_paths):
            raise ValueError(
                f"If weights is not None, number of weights ({len(args.weights)}) must be equal to number of kwcoco files ({len(args.kwcoco_file_paths)})."
            )
    else:
        # Set weight of all files to 1.
        args.weights = [1] * len(args.kwcoco_file_paths)

    ## If channels is not one value then make sure that it has the same number of values as kwcoco files.
    if len(args.channel_name) != 1:
        if len(args.channel_name) != len(args.kwcoco_file_paths):
            raise ValueError(
                f"If more than one channel name, number of channel names ({len(args.channel_name)}) must be equal to number of kwcoco files ({len(args.kwcoco_file_paths)})."
            )
    else:
        args.channel_name = args.channel_name * len(args.kwcoco_file_paths)

    ## If no merge_channel_name given then use first
    if args.merge_channel_name is None:
        args.merge_channel_name = args.channel_name[0]
        print(f"INFO: No output channel name given, using channel name: {args.channel_name[0]}")

    ## Check kwcoco files to see that they exist and contain the required channels.
    for kwcoco_file_path, channel_name in zip(args.kwcoco_file_paths, args.channel_name):
        check_kwcoco_file(kwcoco_file_path, channel_name, sensor_name=args.sensor)
    print("INFO: Verified that kwcoco files contain correct channel name(s).")

    # Merge kwcoco files along certain channels.
    merge_kwcoco_channels(
        args.kwcoco_file_paths,
        args.output_kwcoco_path,
        args.channel_name,
        args.weights,
        args.merge_channel_name,
        sensor_name=args.sensor,
    )


if __name__ == "__main__":
    main()
