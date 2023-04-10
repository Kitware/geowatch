import kwcoco
import imageio
import cv2
import argparse

import pandas as pd
import numpy as np

from tqdm import tqdm
from pathlib import Path
from shapely.geometry import shape


def _get_image(coco_image: kwcoco.CocoImage, channels: list, scale=0.2):

    delay_image = coco_image.delay(channels=channels)
    image = delay_image.scale(scale).finalize(nodata='float')

    num_of_channels = image.shape[2]
    if (num_of_channels == 3) or (num_of_channels == 1):

        if num_of_channels == 1:
            # TODO - do this somewhere else; convert single-band to three-bands image
            image = np.repeat(image[:, :, np.newaxis], 3, axis=2)
            image = np.squeeze(image)

        # Clip image to be between the 2nd and 98th percentiles
        image = image.astype(np.float32)
        percentile98 = np.nanpercentile(image, 98)
        percentile2 = np.nanpercentile(image, 2)

        if (percentile98 != percentile2):
            result = (image - percentile2) / (percentile98 - percentile2)
        else:
            result = image

        # Only values between 0 and 1 are valid
        result[result < 0] = 0
        result[result > 1] = 1

        # Convert result to uint8
        result = result * 255
        result = result.astype(np.uint8)

        return result
    else:
        raise RuntimeError(f'Unsupported number of channels: {num_of_channels}')


def _parse_channels(channels: str):
    splits = channels.split(',')
    result = []

    for split in splits:
        entry = split.split('|')
        result.append(entry)

    return result


def _apply_sensor_filter(data: pd.DataFrame, sensors: list):
    data['valid_sensor'] = data['sensor_coarse'].apply(lambda x: x in sensors)
    data = data[data['valid_sensor']]
    data = data.drop('valid_sensor', axis=1)

    return data


def _apply_channel_filter(data: pd.DataFrame, channels: list):
    flatten_channels = [element for sublist in channels for element in sublist]
    data['has_channels'] = data['auxiliary'].apply(lambda x: _has_channels(x, flatten_channels))
    data = data[data['has_channels']]
    data = data.drop('has_channels', axis=1)

    return data


def _has_channels(auxs: list, channels: list):
    available_channels = [aux['channels'] for aux in auxs]
    return set(channels).issubset(available_channels)


def _generate_gif(images: list, texts: list, filename: Path, duration=0.1):

    # Save the list of images to a GIF animation file
    with imageio.get_writer(filename, mode='I', duration=duration) as writer:
        for image, text in zip(images, texts):
            overlayed_image = _add_text_overlay(image, text)
            writer.append_data(overlayed_image)


def _add_text_overlay(image, text):

    # Define the text position, font, and color
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1
    color = (255, 255, 255)
    thickness = 2

    # Calculate the size of the text and the optimal text position
    height, width, _ = image.shape
    (text_width, text_height), _ = cv2.getTextSize(text, font, font_scale, thickness)
    text_x = int((width - text_width) / 10) * 5
    text_y = int((height + text_height) / 10)

    # Add the text overlay to the image
    result = cv2.putText(np.copy(image), text, (text_x, text_y), font, font_scale, color, thickness, cv2.LINE_AA)

    return result


def _get_args():
    parser = argparse.ArgumentParser(description='Create gifs for each stack of images per region')
    parser.add_argument('-i', '--input', type=Path, required=True, help='input kwcoco json file')
    parser.add_argument('-o', '--output_dir', type=Path, required=True, help='output directory where to store gifs')
    parser.add_argument('-s', '--sensors', type=str, default=None, help='Sensor(s) to select images from; comma delimited')
    parser.add_argument('-rc', '--required_channels', type=str, default='red|green|blue,depth', help='required channels to use for image creation')
    parser.add_argument('-oc', '--optional_channels', type=str, default='change', help='optional channels to use for image creation')
    parser.add_argument('-d', '--duration', type=float, default=0.2, help='Display time of each frame in seconds')
    return parser.parse_args()


if __name__ == '__main__':

    """
    Example Usage:
        python -W ignore -m watch.tasks.dzyne_misc.create_region_gif \
            -i /smart_data_dvc/Validation-V1/positive_annotated/AE_R001_positive_annotated/data_filteredWV_depth_change.kwcoco.json \
            -o /smart_data_dvc/Validation-V1/positive_annotated/AE_R001_positive_annotated/_assets/gifs \
            -s WV -rc 'red|green|blue,depth' \
            -oc 'change' \
            -d 0.5
    """

    args = _get_args()

    required_channels_list = _parse_channels(args.required_channels)
    optional_channels_list = _parse_channels(args.optional_channels)

    data = kwcoco.CocoDataset(args.input)
    df = pd.DataFrame(data.dataset['images'])
    print(f'Number of images: {len(df)}')

    # apply sensor(s) filtering
    if args.sensors is not None:
        df = _apply_sensor_filter(df, args.sensors.split(','))
        print(f'Number of images after sensor filtering: {len(df)}')

    # apply required channel(s) filtering
    df = _apply_channel_filter(df, required_channels_list)
    print(f'Number of images after channel filtering: {len(df)}')

    # iterate through all regions
    df['geos_corners'] = df.geos_corners.apply(lambda x: shape(x).wkt)
    combined_channels_list = required_channels_list + optional_channels_list
    del required_channels_list
    del optional_channels_list

    region_uuid = 0
    for region_name, region_df in tqdm(df.groupby('geos_corners')):
        region_df = region_df.sort_values(by='timestamp')

        # iterate through images in region
        dst_file = args.output_dir / f'{region_uuid}.gif'

        if dst_file.exists() is False:

            gif_images = []
            gif_texts = []

            for row in region_df.itertuples():

                coco_image = data.coco_image(row.id)

                images = []
                for channel in combined_channels_list:
                    if _has_channels(coco_image['auxiliary'], channel):
                        image = _get_image(coco_image, channel)
                    else:
                        image = _get_image(coco_image, channel)
                        image = np.zeros_like(image)

                    images.append(image)

                image_hstack = np.hstack(images)
                del images

                gif_images.append(image_hstack)
                gif_texts.append(f'{row.id}')
                del image_hstack

            dst_file.parent.mkdir(parents=True, exist_ok=True)
            _generate_gif(gif_images, gif_texts, dst_file, duration=args.duration)
            del gif_images

        region_uuid += 1
