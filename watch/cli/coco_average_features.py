import os
import re

import kwcoco
import kwimage
import numpy as np
import ubelt as ub
from tqdm import tqdm
import scriptconfig as scfg

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


def check_kwcoco_file(kwcoco_file, channel_name, sensor_names=None, flexible_merge=False):
    """Make sure that kwcoco files exist and contain required channel name.

    Args:
        kwcoco_file (kwcoco.CocoDataset): kwcoco file containing images.
        channel_name (str): Name of channel thats required to be in kwcoco file.
        sensor_names (list(str), optional): Only check images of from these types of sensors. Defaults to None.
        flexible_merge (bool, optional): Skip images that do not contain channel_name. Defaults to False.

    Returns:
        missing_image_names (list): A list of names corresponding to images without the channel name.
    """

    # Get all images in kwcoco file.
    images: kwcoco.coco_dataset.Videos = kwcoco_file.images()

    n_all_images = len(images)
    if sensor_names is not None:
        # Filter to only images with a chosen sensor
        flags = [s in sensor_names for s in images.lookup("sensor_coarse", None)]
        images = images.compress(flags)
        print(f"INFO: Number of {sensor_names} images: [{len(images)}/{n_all_images}]")

        if len(images) == 0:
            raise ValueError(f"No images in {kwcoco_file.fpath} containing images of sensor {sensor_names}.")

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
                    f"Channel(s) '{channel_names}' not found in image {coco_img.img['id']} of kwcoco file {kwcoco_file.fpath}. Only channels found: {coco_img.channels}"
                )
        else:
            available_image_names.append(image_name)

    if flexible_merge:
        print(kwcoco_file.fpath)
        print(
            f"INFO: Number of images without requested channel names from [{len(missing_image_names)}/{len(images.coco_images)}]"
        )

        # Make sure that at least one image contains all requested channel names.
        if len(missing_image_names) == len(images.coco_images):
            raise ValueError(f"All of the images in {kwcoco_file.fpath} were missing channel(s) {channel_names}")

    return available_image_names, missing_image_names


def merge_kwcoco_channels(kwcoco_file_paths,
                          output_kwcoco_path,
                          channel_names,
                          weights,
                          output_channel_names,
                          sensor_names=None,
                          target_resolution=None,
                          flexible_merge=False):
    """
    Compute a weighted mean of channels from separate kwcoco file and save into
    merged kwcoco file.

    Assumptions:
        - The channel_nams to merge are not a subset of a group of channels targeted in the kwcoco file.
            - I.e. 'salient' in 'salient|notsalient' will not work.

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

        sensor_names (list(str), optional):
            Only merge images belonging to sensors in this list. Defaults to None (aka do not filter by sensor).

        target_resolution (int | str, optional): 
            GSD to resize the resolution of the images to. Defaults to None.

        flexible_merge (bool, optional):
            Skip images that do not contain channel_name. Defaults to False.

    Example:
        >>> # TEST 1: Merge two kwcoco files with the same number of images and plot results.
        >>> from watch.cli.coco_average_features import *  # NOQA
        >>> import watch
        >>> from kwcoco.demo.perterb import perterb_coco
        >>> dpath = ub.Path.appdir('watch/test/coco_average_features')
        >>> base_dset = watch.demo.coerce_kwcoco('watch-msi')
        >>> # Construct two copies of the same data with slightly different heatmaps
        >>> dset1 = perterb_coco(base_dset.copy(), box_noise=0.5, cls_noise=0.5, n_fp=10, n_fn=10, rng=32)
        >>> dset2 = base_dset.copy()
        >>> dset1.fpath = ub.Path(dset1.fpath).augment(stemsuffix='_heatmap1')
        >>> dset2.fpath = ub.Path(dset2.fpath).augment(stemsuffix='_heatmap2')
        >>> watch.demo.smart_kwcoco_demodata.hack_in_heatmaps(dset1, heatmap_dname='dummy_heatmap1', with_nan=0, rng=423432)
        >>> watch.demo.smart_kwcoco_demodata.hack_in_heatmaps(dset2, heatmap_dname='dummy_heatmap2', with_nan=0, rng=132129)
        >>> dset1.dump(dset1.fpath)
        >>> dset2.dump(dset2.fpath)
        >>> # Build method args
        >>> kwcoco_file_paths = [dset1.fpath, dset2.fpath]
        >>> output_bundle_dpath = (dpath / 'merge_bundle').delete().ensuredir()
        >>> output_kwcoco_path = output_bundle_dpath / 'data.kwcoco.json'
        >>> channel_names = ['notsalient|salient'] * 2
        >>> weights = [1.0, 0.0]
        >>> output_channel_names = 'notsalient|salient'
        >>> sensor_name = None
        >>> # Execute merge
        >>> merge_kwcoco_channels(kwcoco_file_paths, output_kwcoco_path,
        >>>                       channel_names, weights, output_channel_names,
        >>>                       sensor_name)
        >>> # Check results
        >>> output_dset = kwcoco.CocoDataset(output_kwcoco_path)
        >>> gid = 5
        >>> imdata1 = dset1.coco_image(gid).delay('salient').finalize()
        >>> imdata2 = dset2.coco_image(gid).delay('salient').finalize()
        >>> imdataM = output_dset.coco_image(gid).delay('salient').finalize()
        >>> # imdata1 = dset1.delayed_load(gid, channels='notsalient|salient').finalize()[:,:,0]
        >>> # imdata2 = dset2.delayed_load(gid, channels='notsalient|salient').finalize()[:,:,0]
        >>> # imdataM = output_dset.delayed_load(gid, channels='notsalient|salient').finalize()[:,:,0]
        >>> import kwplot
        >>> kwplot.autompl()
        >>> kwplot.imshow(kwimage.normalize_intensity(imdata1), title='img1', pnum=(1, 3, 1), fnum=1)
        >>> kwplot.imshow(kwimage.normalize_intensity(imdata2), title='img2', pnum=(1, 3, 2), fnum=1)
        >>> kwplot.imshow(kwimage.normalize_intensity(imdataM), title='mean', pnum=(1, 3, 3), fnum=1)
        >>> save_figure_path = dpath / 'test_1_result_plot.png'
        >>> import matplotlib.pyplot as plt
        >>> plt.savefig(save_figure_path)
        >>> print(f'Test 1 plot saved to: {save_figure_path}')
        >>> print(f'Weights: {weights}')
        >>> print(f'Img1  mean: {np.nan_to_num(imdata1).mean()}')
        >>> print(f'Img2  mean: {np.nan_to_num(imdata2).mean()}')
        >>> print(f'Merge mean: {np.nan_to_num(imdataM).mean()}')
        >>> print()
        >>> print(f'Img1  shape: {imdata1.shape}')
        >>> print(f'Img2  shape: {imdata2.shape}')
        >>> print(f'Merge shape: {imdataM.shape}')
        >>> os.remove(dset1.fpath)
        >>> os.remove(dset2.fpath)
        >>> os.remove(output_dset.fpath)
    """

    # Check args.
    if len(kwcoco_file_paths) <= 1:
        raise ValueError(f"Need at least 2 kwcoco files to merge and only recieved {len(kwcoco_file_paths)}")

    ## If weights argument is not None then make sure it is the same length as kwcoco files.
    if weights is not None:
        if len(weights) != len(kwcoco_file_paths):
            raise ValueError(
                f"If weights is not None, number of weights ({len(weights)}) must be equal to number of kwcoco files ({len(kwcoco_file_paths)})."
            )
    else:
        # Set weight of all files to 1.
        weights = [1] * len(kwcoco_file_paths)

    ## If channels is not one value then make sure that it has the same number of values as kwcoco files.
    if len(channel_names) != 1:
        if len(channel_names) != len(kwcoco_file_paths):
            raise ValueError(
                f"If more than one channel name, number of channel names ({len(channel_names)}) must be equal to number of kwcoco files ({len(kwcoco_file_paths)})."
            )
        # Make sure the number of channel names per file are equal.
        n_channels_per_file = []
        for kwcoco_file_channel_names in channel_names:
            n_channels_per_file.append(len(split_channel_names_by_grammar(kwcoco_file_channel_names)))

        if len(set(n_channels_per_file)) != 1:
            raise ValueError(f"Number of requested channels per kwcoco file are not equal: {n_channels_per_file}")
    else:
        channel_names = channel_names * len(kwcoco_file_paths)

    ## If no merge_channel_name given then use first
    if output_channel_names is None:
        output_channel_names = channel_names[0]
        print(f"INFO: No output channel name given, using channel name: {channel_names[0]}")
    else:
        # Make sure that the size of output channel names and input channel names are equal in length.
        split_output_channel_names = split_channel_names_by_grammar(output_channel_names)

        # Get the number of channels per kwcoco file.
        n_channels_per_file = []
        for kwcoco_file_channel_names in channel_names:
            n_channels_per_file.append(len(split_channel_names_by_grammar(kwcoco_file_channel_names)))

        if len(split_output_channel_names) != list(set(n_channels_per_file))[0]:
            raise ValueError(
                f"Number of output channels ({(output_channel_names)}) does not match number of requested channels ({list(set(n_channels_per_file))[0]})."
            )

    # Load and merge images from kwcoco files.
    ## Load kwcoco files.
    kwcoco_files = [kwcoco.CocoDataset.coerce(p) for p in kwcoco_file_paths]

    ## Check kwcoco files to see that they exist and contain the required channels.
    all_available_image_names, all_missing_image_names = [], []
    for kwcoco_file_path, channel_name in zip(kwcoco_files, channel_names):
        available_image_names, missing_image_names = check_kwcoco_file(kwcoco_file_path,
                                                                       channel_name,
                                                                       sensor_names=sensor_names,
                                                                       flexible_merge=flexible_merge)
        all_missing_image_names.append(set(missing_image_names))
        all_available_image_names.append(set(available_image_names))

    ## Find common image names.
    all_image_names = list(set.union(*all_available_image_names))
    common_image_names = list(set.intersection(*all_available_image_names))
    missing_image_names = list(set.difference(set(common_image_names), set(all_image_names)))
    if len(common_image_names) == 0:
        raise ValueError("No common images found between all kwcoco files.")

    if len(missing_image_names) != 0:
        print(f'Out of all images, {len(missing_image_names)} are missing from at least one kwcoco file.')

    # Create output kwcoco by copying first kwcoco file.
    merge_kwcoco = kwcoco_files[0].copy()

    ## Remove missing images from kwcoco file.
    merge_kwcoco.remove_images(missing_image_names)

    # Load channel images from each viable image_id.
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
        if sensor_names is not None:
            if image_info["sensor_coarse"] not in sensor_names:
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
            # image_data = coco_img.delay(kwcoco_channel_names[file_index], space='video',
            #                             resolution=target_resolution).finalize()
            image_data = coco_img.delay(kwcoco_channel_names[file_index]).finalize()

            # FIXME:
            image_data = np.nan_to_num(image_data)

            if image_id == 5:
                if file_index == 0:
                    first_img = image_data
                else:
                    last_img = image_data

            # Weight image data.
            weighted_image_data = image_data * weights[file_index]

            if accum_data is None:
                accum_data = weighted_image_data
            else:
                accum_data += weighted_image_data

        # breakpoint()
        # pass
        # img_data_no_nan = np.nan_to_num(image_data)
        # print(img_data_no_nan.min(), img_data_no_nan.mean(), img_data_no_nan.max())

        # accum_data_no_nan = np.nan_to_num(accum_data)
        # print(accum_data_no_nan.min(), accum_data_no_nan.mean(), accum_data_no_nan.max())

        # FIXME: Does not work.
        # accum_data = image_data

        # Normalize averaged data by weights.
        average_image_data = accum_data / sum(weights)
        if image_id == 5:
            print(
                f'{image_id} | First: {first_img[:,:,0].mean()} | Second: {last_img[:,:,0].mean()} | Merge: {average_image_data[:,:,0].mean()}'
            )
            print(
                f'{image_id} | First: {first_img.mean()} | Second: {last_img.mean()} | Merge: {average_image_data.mean()}'
            )
            print(
                f'{image_id} | First: {first_img.shape} | Second: {last_img.shape} | Merge: {average_image_data.shape}')

        # average_image_data_no_nan = np.nan_to_num(average_image_data)
        # print(average_image_data_no_nan.min(), average_image_data_no_nan.mean(), average_image_data_no_nan.max())

        # TODO: better backend name.
        path_chan = output_channels.path_sanitize()
        img_name = merge_coco_img.img.get("name", "")
        average_fname = f"merged_{img_name}_{path_chan}.tif"
        average_fpath = save_assest_dir / average_fname

        # Check if there is already an asset with the same channel_names.
        output_obj = None
        for cand_obj in merge_coco_img.iter_asset_objs():
            cand_channels = kwcoco.FusedChannelSpec.coerce(cand_obj["channels"])
            # Check if output channels matches candidate channels.
            if output_channels == cand_channels:
                output_obj = cand_obj
                break

        if output_obj is not None:
            # Overwrite the data in the output auxiliary item.
            output_obj["file_name"] = os.fspath(average_fpath)
            # output_obj['height'] = average_image_data.shape[0]
            # output_obj['width'] = average_image_data.shape[1]
        # else:
        #     # Add this as a new auxiliary item.
        #     merge_kwcoco.add_auxiliary_item(image_id,
        #                                     os.fspath(average_fpath),
        #                                     channels=output_channels,
        #                                     height=average_image_data.shape[0],
        #                                     width=average_image_data.shape[1])

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


class CocoAverageFeaturesConfig(scfg.DataConfig):
    """
    Create a new kwcoco file with averaged features from multiple kwcoco files.

    High Level Steps:
        1. Load kwcoco files. Must have at least two kwcoco files.
        2. Create new kwcoco file by copying first kwcoco file.
        3. For each image ID in the kwcoco file, load the features from each kwcoco file.
            a. Average the features from each kwcoco file.
            b. Save the averaged features to the new kwcoco image.
        4. Save the new kwcoco file.
    
    """
    kwcoco_file_paths = scfg.Value(None,
                                   type=str,
                                   required=True,
                                   help=ub.paragraph('''
            Path to at least two kwcoco paths with predictions.
            '''),
                                   nargs='+')
    output_kwcoco_path = scfg.Value(None,
                                    type=str,
                                    required=True,
                                    help=ub.paragraph('''
            Path to the combined features kwcoco file.
            '''))
    channel_name = scfg.Value(None,
                              type=str,
                              required=True,
                              help=ub.paragraph('''
            Name of the channel in kwcoco files to merge.
            '''),
                              nargs='+')
    weights = scfg.Value(None,
                         type=float,
                         help=ub.paragraph('''
            Combination weight value for each prediction from kwcoco
            file. Default: All predictions are equally weighted.
            '''),
                         nargs='+')
    sensors = scfg.Value(None,
                         type=str,
                         choices=[None, 'all', 'S2', 'L8', 'WV'],
                         help=ub.paragraph('''
            Only merge channels from this type of sensor.
            '''))
    output_channel_names = scfg.Value(None,
                                      type=str,
                                      help=ub.paragraph('''
            What the name of the output channels will be after
            averaging. Needs to have the same number of channel names as
            both of the input channel names. NOTE: Channel names can be
            separated by ',' or '|' characters.
            '''))
    flexible_merge = scfg.Value(False,
                                isflag=True,
                                help=ub.paragraph('''
            If active, skip images that dont contain band when merging.
            '''))
    target_resolution = scfg.Value(None,
                                   type=float,
                                   help=ub.paragraph('''
            Set the target resolution that the features will be loaded at 
            and then merged.
            '''))


def main(cmdline=True, **kw):
    """
    Main function for merge_kwcoco_channels.
    See :class:``CocoAverageFeaturesConfig` for details

    TODO: Add examples
    """
    config = CocoAverageFeaturesConfig(default=kw, cmdline=cmdline)
    config_dict = config.to_dict()

    # Merge kwcoco files along certain channels.
    merge_kwcoco_channels(
        config_dict['kwcoco_file_paths'],
        config_dict['output_kwcoco_path'],
        config_dict['channel_name'],
        config_dict['weights'],
        config_dict['output_channel_names'],
        sensor_names=config_dict['sensors'],
        target_resolution=config_dict['target_resolution'],
    )


if __name__ == "__main__":
    main()
