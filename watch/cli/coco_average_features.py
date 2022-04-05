import os
import argparse
import kwcoco
import kwimage
import numpy as np
import ubelt as ub
from tqdm import tqdm

from watch import exceptions
# from watch.cli.coco_combine_features import associate_images
from watch.utils.kwcoco_extensions import transfer_geo_metadata


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
        print(f"INFO: Number of {sensor_name} images: [{n_all_images - len(images)}/{n_all_images}]")

    missing_image_names = []
    for coco_img in images.coco_images:
        if channel_name not in list(coco_img.channels.keys()):
            missing_image_names.append(kwcoco_file.index.imgs[coco_img.img["id"]]["name"])
            if flexible_merge is False:
                raise AssertionError(
                    f"Channel '{channel_name}' not found in image {coco_img.img['id']} of kwcoco file {kwcoco_path}. Only channels found: {coco_img.channels}"
                )
    if flexible_merge:
        print(kwcoco_path)
        print(f"INFO: Number of images without channel from [{len(missing_image_names)}/{len(images.coco_images)}]")
    return missing_image_names


def merge_kwcoco_channels(
    kwcoco_file_paths,
    output_kwcoco_path,
    channel_names,
    weights,
    merged_channel_name,
    sensor_name=None,
    missing_image_names=[],
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
        >>> merged_channel_name = 'notsalient|salient'
        >>> sensor_name = None
        >>> # Execute merge
        >>> merge_kwcoco_channels(kwcoco_file_paths, output_kwcoco_path,
        >>>                       channel_names, weights, merged_channel_name,
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

    input_channel_objs = [kwcoco.FusedChannelSpec.coerce(c) for c in channel_names]
    output_channels = kwcoco.FusedChannelSpec.coerce(merged_channel_name)
    assert ub.allsame(input_channel_objs), "expecting same channels for now"
    assert output_channels == input_channel_objs[0], "expecting same channels for now"

    pbar = tqdm(merge_kwcoco.index.imgs.items(), desc="Merging images", colour="green")
    for image_id, image_info in pbar:
        image_name = merge_kwcoco.index.imgs[image_id]["name"]
        if image_name in missing_image_names:
            continue

        # If sensor name spacified, only merge channels for images from this sensor.
        if sensor_name is not None:
            if image_info["sensor_coarse"] != sensor_name:
                continue

        # Find which asset the merged channels will belong to
        merge_coco_img = merge_kwcoco.coco_image(image_id)
        output_obj = None
        for cand_obj in merge_coco_img.iter_asset_objs():
            cand_channels = kwcoco.FusedChannelSpec.coerce(cand_obj["channels"])
            if output_channels == cand_channels:
                output_obj = cand_obj
                break
        assert output_obj is not None, "missing, todo: make more flexible"

        # Find which assets will be inputs
        input_objs = []
        input_dpaths = []
        for kwcoco_index, kwcoco_file in enumerate(kwcoco_files):
            coco_img = kwcoco_file.coco_image(image_id)
            input_obj = None
            for cand_obj in coco_img.iter_asset_objs():
                cand_channels = kwcoco.FusedChannelSpec.coerce(cand_obj["channels"])
                if output_channels == cand_channels:
                    input_obj = cand_obj
            assert output_obj is not None, "missing, todo: make more flexible"
            input_dpaths.append(kwcoco_file.bundle_dpath)
            input_objs.append(input_obj)

        average_imdata = average_auxiliary_datas(input_objs, input_dpaths, weights)

        # TODO: better backend name
        path_chan = output_channels.path_sanitize()
        img_name = merge_coco_img.img.get("name", "")
        average_fname = f"merged_{img_name}_{path_chan}.tif"
        average_fpath = save_assest_dir / average_fname

        # Overwrite the data in the output auxiliary item
        output_obj["file_name"] = os.fspath(average_fpath)

        kwimage.imwrite(average_fpath, average_imdata, backend="gdal")

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


# output_obj, output_dpath, weights, save_assest_dir):


def average_auxiliary_datas(input_objs, input_dpaths, weights):
    """
    Args:
        input_objs (list[dict]): list of input auxiliary items with same channels
        input_dpaths (list[str]): directory each input obj is relative to
        weights (list[float]): weight for each input obj

    Returns:
        np.ndarray : averaged heatmap in auxiliary space
    """
    accum_imdata = None
    accum_weights = None
    for obj, dpath, weight in zip(input_objs, input_dpaths, weights):
        # Assuming auxiliary data is perfectly alignable
        # TODO: Need a better way to handle case that auxiliary channels could have different
        # sizes between datasets. Right now the first dataset has to contain the largest
        # image which we can't rely on using the correct order.
        fpath = os.path.join(dpath, obj["file_name"])
        imdata = kwimage.imread(fpath, nodata="float")
        mask = np.isnan(imdata)
        imweights = np.full(imdata.shape, fill_value=weight, dtype="float")
        imweights[mask] = 0
        imdata[mask] = 0
        weighted_imdata = imdata * imweights
        if accum_imdata is None:
            accum_imdata = weighted_imdata
            accum_weights = imweights
        else:
            try:
                accum_imdata += weighted_imdata
            except ValueError:
                h, w = weighted_imdata.shape
                accum_imdata[:h, :w] += weighted_imdata

            try:
                accum_weights += imweights
            except ValueError:
                h, w = imweights.shape
                accum_weights[:h, :w] += imweights
    import warnings

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", "invalid value")
        average_imdata = accum_imdata / accum_weights

    return average_imdata


def main(cmdline=True):
    """
    Example call:
    python watch/cli/coco_average_features.py --kwcoco_file_paths \
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
        nargs="+",
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
        "--merge_channel_name",
        type=str,
        help="Name of channel with merged features. By default using the first input argument channel_name.",
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
    else:
        args.channel_name = args.channel_name * len(args.kwcoco_file_paths)

    ## If no merge_channel_name given then use first
    if args.merge_channel_name is None:
        args.merge_channel_name = args.channel_name[0]
        print(f"INFO: No output channel name given, using channel name: {args.channel_name[0]}")

    ## Check kwcoco files to see that they exist and contain the required channels.
    all_missing_image_names = []
    for kwcoco_file_path, channel_name in zip(args.kwcoco_file_paths, args.channel_name):
        missing_image_names = check_kwcoco_file(
            kwcoco_file_path, channel_name, sensor_name=args.sensor, flexible_merge=args.flexible_merge
        )
        all_missing_image_names.append(set(missing_image_names))

    # Find common image names
    if len(all_missing_image_names) != 0:
        common_missing_image_names = list(set.union(*all_missing_image_names))
    print("INFO: Verified that kwcoco files contain correct channel name(s).")

    # Merge kwcoco files along certain channels.
    merge_kwcoco_channels(
        args.kwcoco_file_paths,
        args.output_kwcoco_path,
        args.channel_name,
        args.weights,
        args.merge_channel_name,
        sensor_name=args.sensor,
        missing_image_names=common_missing_image_names,
    )


if __name__ == "__main__":
    main()
