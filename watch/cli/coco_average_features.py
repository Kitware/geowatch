import os
import re
import argparse
import kwcoco
import kwimage
import ubelt as ub
from tqdm import tqdm

from watch import exceptions
from watch.utils.kwcoco_extensions import transfer_geo_metadata


def split_channel_names_by_grammar(channel_names):
    """Split a string containing channel names by commas (,) and pipes (|).

    Args:
        channel_names (str):
            A string that may contain commas and pipe characters.

    Returns:
        list(str): A list of strings that were originally divided by certain characters.
    """
    channel_names = re.split(r",|\|", channel_names)
    return channel_names


def check_kwcoco_file(kwcoco_path, channel_name, sensor_name=None, flexible_merge=False):
    """Make sure that kwcoco files exist and contain required channel name.

    Args:
        kwcoco_path (str): Path to local kwcoco file.
        channel_name (str): Name of channel thats required to be in kwcoco file.
        sensor_name (str, optional): Only check images of from this type of sensor. Defaults to None.
        flexible_merge (bool, optional): Skip images that do not contain channel_name. Defaults to False.

    Returns:
        missing_image_names (list): A list of names corresponding to images without the channel name.
    """

    # Check if kwcoco file exists.
    if os.path.isfile(kwcoco_path) is False:
        raise FileNotFoundError(f"KWCOCO file not found at {kwcoco_path}")

    # Load kwcoco file.
    kwcoco_file = kwcoco.CocoDataset(kwcoco_path)

    # Get all images in kwcoco file.
    images: kwcoco.coco_dataset.Videos = kwcoco_file.images()

    n_all_images = len(images)
    if sensor_name is not None:
        # Filter to only images with a chosen sensor
        flags = [s == sensor_name for s in images.lookup("sensor_coarse", None)]
        images = images.compress(flags)
        print(f"INFO: Number of {sensor_name} images: [{len(images)}/{n_all_images}]")

        if len(images) == 0:
            raise ValueError(f"No images in {kwcoco_path} containing images of sensor {sensor_name}.")

    # Breakup channels names based on , and | characters and convert channel_name from str to list.
    channel_names = split_channel_names_by_grammar(channel_name)

    available_image_names, missing_image_names = [], []
    for coco_img in images.coco_images:
        # Check if all requested channel names were found in image.
        channel_matches = []
        merged_image_channel_names = ",".join(list(coco_img.channels.keys()))
        image_channels = split_channel_names_by_grammar(merged_image_channel_names)
        for channel_name in channel_names:
            channel_matches.append(channel_name in image_channels)

        # If all channel matches are true then skip.
        image_name = kwcoco_file.index.imgs[coco_img.img["id"]]["name"]
        if not all(channel_matches):
            missing_image_names.append(image_name)
            if flexible_merge is False:
                raise AssertionError(
                    f"Channel(s) '{channel_names}' not found in image {coco_img.img['id']} of kwcoco file {kwcoco_path}. Only channels found: {coco_img.channels}"
                )
        else:
            available_image_names.append(image_name)

    if flexible_merge:
        print(kwcoco_path)
        print(
            f"INFO: Number of images without requested channel names from [{len(missing_image_names)}/{len(images.coco_images)}]"
        )

        # Make sure that at least one image contains all requested channel names.
        if len(missing_image_names) == len(images.coco_images):
            raise ValueError(f"All of the images in {kwcoco_path} were missing channel(s) {channel_names}")

    return available_image_names, missing_image_names


def merge_kwcoco_channels(
    kwcoco_file_paths,
    output_kwcoco_path,
    channel_names,
    weights,
    output_channel_names,
    sensor_name=None,
    common_image_names=[],
):
    """
    Compute a weighted mean of channels from separate kwcoco file and save into
    merged kwcoco file.

    Args:
        kwcoco_file_paths (list(str)):
            A list of paths representing pathes to kwcoco files to be merged.

        output_kwcoco_path (str):
            Local path to the kwcoco file with merged channels.

        channel_names (list(str)):
            A list of channel names corresponding to the channel name to merge
            from each kwcoco file. Note, the length of the channel names be
            equal to the number of kwcoco file paths.

        weights (list(int)):
            A list of floats representing how much weight a particular kwcoco
            file should contribute to the final merged prediction.

        output_channel_names (list(str)):
            A list containing the names of the output channel names. Must
            contain the same number of channel names as the input channel
            names.

        sensor_name (str, optional):
            Only merge images belonging to this sensor. Defaults to None.

        missing_image_names (list, optional):
            Skip combining these image names. Defaults to False.


    Example:
        >>> from watch.cli.coco_average_features import *  # NOQA
        >>> import watch
        >>> from kwcoco.demo.perterb import perterb_coco
        >>> dpath = ub.Path.appdir('watch/test/coco_average_features')
        >>> base_dset = watch.demo.coerce_kwcoco('watch-msi')
        >>> # Construct two copies of the same data with slightly different heatmaps
        >>> dset1 = perterb_coco(base_dset.copy(), box_noise=0.5, cls_noise=0.5, n_fp=10, n_fn=10, rng=32)
        >>> dset2 = base_dset.copy()
        >>> dset1.fpath = ub.Path(dset1.fpath).augment(suffix='_heatmap1')
        >>> dset2.fpath = ub.Path(dset2.fpath).augment(suffix='_heatmap2')
        >>> watch.demo.smart_kwcoco_demodata.hack_in_heatmaps(dset1, heatmap_dname='dummy_heatmap1', with_nan=0, rng=423432)
        >>> watch.demo.smart_kwcoco_demodata.hack_in_heatmaps(dset2, heatmap_dname='dummy_heatmap2', with_nan=0, rng=132129)
        >>> dset1.dump(dset1.fpath)
        >>> dset2.dump(dset2.fpath)
        >>> # Build method args
        >>> kwcoco_file_paths = [dset1.fpath, dset2.fpath]
        >>> output_bundle_dpath = (dpath / 'merge_bundle').delete().ensuredir()
        >>> output_kwcoco_path = output_bundle_dpath / 'data.kwcoco.json'
        >>> channel_names = ['notsalient|salient'] * 2
        >>> weights = [1.0, 1.0]
        >>> output_channel_names = 'notsalient|salient'
        >>> sensor_name = None
        >>> # Execute merge
        >>> merge_kwcoco_channels(kwcoco_file_paths, output_kwcoco_path,
        >>>                       channel_names, weights, output_channel_names,
        >>>                       sensor_name)
        >>> # Check results
        >>> output_dset = kwcoco.CocoDataset(output_kwcoco_path)
        >>> gid = 1
        >>> imdata1 = dset1.coco_image(gid).delay('salient').finalize()
        >>> imdata2 = dset2.coco_image(gid).delay('salient').finalize()
        >>> imdataM = output_dset.coco_image(gid).delay('salient').finalize()
        >>> import kwplot
        >>> kwplot.autompl()
        >>> kwplot.imshow(kwimage.normalize_intensity(imdata1), title='img1', pnum=(1, 3, 1), fnum=1)
        >>> kwplot.imshow(kwimage.normalize_intensity(imdata2), title='img2', pnum=(1, 3, 2), fnum=1)
        >>> kwplot.imshow(kwimage.normalize_intensity(imdataM), title='mean', pnum=(1, 3, 3), fnum=1)
    """
    # Load and merge images from kwcoco files.
    ## Load kwcoco files.
    kwcoco_files = [kwcoco.CocoDataset.coerce(p) for p in kwcoco_file_paths]

    ## Create output kwcoco by copying first kwcoco file.
    merge_kwcoco = kwcoco_files[0].copy()

    ## Load channel images from each viable image_id.
    output_kwcoco_path = ub.Path(output_kwcoco_path)
    save_assest_dir = (output_kwcoco_path.parent / "_assets").ensuredir()

    # Get channel names for each file.
    kwcoco_channel_names = []
    for channel_name in channel_names:
        split_channel_names = split_channel_names_by_grammar(channel_name)
        combined_channel_names = kwcoco.FusedChannelSpec.coerce(split_channel_names)
        kwcoco_channel_names.append(combined_channel_names)

    # Generate output channel names in (channel_name1|channel_name2|channel_name3) format.
    output_channels = kwcoco.FusedChannelSpec.coerce(output_channel_names)

    pbar = tqdm(merge_kwcoco.index.imgs.items(), desc="Merging images", colour="green")
    for image_id, image_info in pbar:
        # Skip this image if its name is in the missing image name list.
        image_name = merge_kwcoco.index.imgs[image_id]["name"]
        if image_name not in common_image_names:
            continue

        # If sensor name specified, only merge channels for images from this sensor.
        if sensor_name is not None:
            if image_info["sensor_coarse"] != sensor_name:
                continue

        # Get the merged kwcoco image.
        merge_coco_img = merge_kwcoco.coco_image(image_id)

        # Get asset channels from each kwcoco file based on image_name.
        accum_data = None
        for file_index, kwcoco_file in enumerate(kwcoco_files):
            # Get the kwcoco specific image id from image name.
            kwfile_image_id = kwcoco_file.index.name_to_img[image_name]["id"]

            # Load the coco image based on image id.
            coco_img = kwcoco_file.coco_image(kwfile_image_id)

            # Get the channel data based on channel names.
            # image_channels: numpy float array of shape [height, width, n_channels].
            image_data = coco_img.delay(kwcoco_channel_names[file_index]).finalize()

            # Weight image data.
            weighted_image_data = image_data * weights[file_index]

            if accum_data is None:
                accum_data = weighted_image_data
            else:
                accum_data += weighted_image_data

        # Normalize averaged data by weights
        average_image_data = accum_data / sum(weights)

        # TODO: better backend name
        path_chan = output_channels.path_sanitize()
        img_name = merge_coco_img.img.get("name", "")
        average_fname = f"merged_{img_name}_{path_chan}.tif"
        average_fpath = save_assest_dir / average_fname

        # Check if there is already an asset with the same channel_names.
        output_obj = None
        for cand_obj in merge_coco_img.iter_asset_objs():
            cand_channels = kwcoco.FusedChannelSpec.coerce(cand_obj["channels"])
            if output_channels == cand_channels:
                output_obj = cand_obj
                break

        # Overwrite the data in the output auxiliary item.
        if output_obj is not None:
            output_obj["file_name"] = os.fspath(average_fpath)

        # Write the averaged image data
        kwimage.imwrite(average_fpath, average_image_data, backend="gdal")

        # Update all channels with projection info.
        try:
            transfer_geo_metadata(merge_kwcoco, image_id)
        except exceptions.GeoMetadataNotFound as ex:
            print("warning ex = {!r}".format(ex))
            pass

    # Save kwcoco file.
    merge_kwcoco.validate()
    merge_kwcoco.dump(output_kwcoco_path)
    print(f"Saved merged kwcoco file to: {output_kwcoco_path}")


def main(cmdline=True):
    """
    Example call:
    python watch/cli/coco_average_features.py --kwcoco_file_paths \
         /data4/datasets/smart_watch_dvc/Drop2-Aligned-TA1-2022-02-15/output_iarpa_drop2v2_total_bin_change_early_fusion_0014.kwcoco.json \
         /data4/datasets/smart_watch_dvc/Drop2-Aligned-TA1-2022-02-15/output_iarpa_drop2v2_total_bin_change_early_fusion_0013.kwcoco.json  \
         --output_kwcoco_path /data4/datasets/smart_watch_dvc/Drop2-Aligned-TA1-2022-02-15/test_comb.kwcoco.json \
         --channel_name "not_salient|salient" --sensor S2

    python watch/cli/coco_average_features.py --kwcoco_file_paths \
         /data4/datasets/smart_watch_dvc/Drop2-Aligned-TA1-2022-02-15/output_iarpa_drop2v2_total_bin_change_early_fusion_0014.kwcoco.json \
         /data4/datasets/smart_watch_dvc/Drop2-Aligned-TA1-2022-02-15/output_iarpa_drop2v2_total_bin_change_early_fusion_0013.kwcoco.json  \
         --output_kwcoco_path /data4/datasets/smart_watch_dvc/Drop2-Aligned-TA1-2022-02-15/test_comb.kwcoco.json \
         --channel_name "not_salient|salient" --sensor S2 --weights 0.5 0.1

    python watch/cli/coco_average_features.py --kwcoco_file_paths \
        "/data4/datasets/smart_watch_dvc/training/core534-SYS-4028GR-TRT/purri/Drop2-Aligned-TA1-2022-02-15/runs/FUSION_EXPERIMENT_ML_V157/lightning_logs/pred/checkpoints/pred_FUSION_EXPERIMENT_ML_V157_epoch=12-step=831-v3/Drop2-Aligned-TA1-2022-02-15_data_vali.kwcoco/predcfg_abd043ec/pred.kwcoco.json" \
        /data4/datasets/smart_watch_dvc/Drop2-Aligned-TA1-2022-02-15/materials_v2_pred.kwcoco.json \
        --output_kwcoco_path /data4/datasets/smart_watch_dvc/Drop2-Aligned-TA1-2022-02-15/jon_mat_fusion.kwcoco.json \
        --channel_name "Site Preparation|Active Construction|Post Construction|No Activity" \
        --sensor S2  --flexible_merge

    python watch/cli/coco_average_features.py --kwcoco_file_paths \
        "/data4/datasets/smart_watch_dvc/training/core534-SYS-4028GR-TRT/purri/Drop2-Aligned-TA1-2022-02-15/runs/FUSION_EXPERIMENT_ML_V157/lightning_logs/pred/checkpoints/pred_FUSION_EXPERIMENT_ML_V157_epoch=12-step=831-v3/Drop2-Aligned-TA1-2022-02-15_data_vali.kwcoco/predcfg_abd043ec/pred.kwcoco.json" \
        /data4/datasets/smart_watch_dvc/Drop2-Aligned-TA1-2022-02-15/materials_v2_pred.kwcoco.json \
        --output_kwcoco_path /data4/datasets/smart_watch_dvc/Drop2-Aligned-TA1-2022-02-15/jon_mat_fusion.kwcoco.json \
        --channel_name "Site Preparation|Active Construction|Post Construction|No Activity" \
        --output_channel_names "site_prep|active_con|post_con|no_activity" --sensor S2  --flexible_merge

    python watch/cli/coco_average_features.py --kwcoco_file_paths \
        "/data4/datasets/smart_watch_dvc/training/core534-SYS-4028GR-TRT/purri/Drop2-Aligned-TA1-2022-02-15/runs/FUSION_EXPERIMENT_ML_V157/lightning_logs/pred/checkpoints/pred_FUSION_EXPERIMENT_ML_V157_epoch=12-step=831-v3/Drop2-Aligned-TA1-2022-02-15_data_vali.kwcoco/predcfg_abd043ec/pred.kwcoco.json" \
        "/data4/datasets/smart_watch_dvc/models/fusion/eval3_candidates/pred/Drop3_SpotCheck_V323/pred_Drop3_SpotCheck_V323_epoch=19-step=13659-v1/Aligned-Drop3-TA1-2022-03-10_combo_LM_nowv_vali.kwcoco/predcfg_chipoverlap0/pred.kwcoco.json" \
        --output_kwcoco_path /data4/datasets/smart_watch_dvc/Drop2-Aligned-TA1-2022-02-15/average_sc_2.kwcoco.json \
        --channel_name "salient" \
        --sensor S2  --flexible_merge
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
        nargs="+",
        type=float,
        default=None,
        help="Combination weight value for each prediction from kwcoco file. Default: All predictions are equally weighted.",
    )
    parser.add_argument(
        "--sensor",
        type=str,
        default=None,
        choices=[None, "all", "S2", "L8", "WV"],
        help="Only merge channels from this type of sensor.",
    )
    parser.add_argument(
        "--output_channel_names",
        type=str,
        help="What the name of the output channels will be after averaging. Needs to have the same number of channel names as both of "
        "the input channel names. NOTE: Channel names can be separated by ',' or '|' characters.",
    )
    parser.add_argument(
        "--flexible_merge",
        default=False,
        action="store_true",
        help="If active, skip images that dont contain band when merging.",
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
        # Make sure the number of channel names per file are equal.
        n_channels_per_file = []
        for channel_names in args.channel_name:
            n_channels_per_file.append(len(split_channel_names_by_grammar(channel_names)))

        if len(set(n_channels_per_file)) != 1:
            raise ValueError(f"Number of requested channels per kwcoco file are not equal: {n_channels_per_file}")
    else:
        args.channel_name = args.channel_name * len(args.kwcoco_file_paths)

    ## If no merge_channel_name given then use first
    if args.output_channel_names is None:
        args.output_channel_names = args.channel_name[0]
        print(f"INFO: No output channel name given, using channel name: {args.channel_name[0]}")
    else:
        # Make sure that the size of output channel names and input channel names are equal in length.
        output_channel_names = split_channel_names_by_grammar(args.output_channel_names)

        # Get the number of channels per kwcoco file.
        n_channels_per_file = []
        for channel_names in args.channel_name:
            n_channels_per_file.append(len(split_channel_names_by_grammar(channel_names)))

        if len(output_channel_names) != list(set(n_channels_per_file))[0]:
            raise ValueError(
                f"Number of output channels ({(output_channel_names)}) does not match number of requested channels ({list(set(n_channels_per_file))[0]})."
            )

    ## Check kwcoco files to see that they exist and contain the required channels.
    all_available_image_names, all_missing_image_names = [], []
    for kwcoco_file_path, channel_name in zip(args.kwcoco_file_paths, args.channel_name):
        available_image_names, missing_image_names = check_kwcoco_file(
            kwcoco_file_path, channel_name, sensor_name=args.sensor, flexible_merge=args.flexible_merge
        )
        all_missing_image_names.append(set(missing_image_names))
        all_available_image_names.append(set(available_image_names))

    # Find common image names.
    common_image_names = list(set.intersection(*all_available_image_names))
    if len(all_available_image_names) == 0:
        raise ValueError("No common images found between all kwcoco file.")

    # Find common missing image names.
    # if len(all_missing_image_names) != 0:
    #     common_missing_image_names = list(set.union(*all_missing_image_names))

    print("INFO: Verified that kwcoco files contain correct channel name(s).")

    # Merge kwcoco files along certain channels.
    merge_kwcoco_channels(
        args.kwcoco_file_paths,
        args.output_kwcoco_path,
        args.channel_name,
        args.weights,
        args.output_channel_names,
        sensor_name=args.sensor,
        common_image_names=common_image_names,
    )


if __name__ == "__main__":
    main()
