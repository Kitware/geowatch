import math
import json
import kwcoco
import imageio
import cv2
import argparse

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm
from pathlib import Path
from shapely.geometry import shape


def load_json(file: Path):
    data = None
    with open(file, 'r') as f:
        data = json.load(f)
    return data


def load_kwcoco(file: Path):
    return kwcoco.CocoDataset(file)


def _preprocess_image_dataframe(data: pd.DataFrame):

    # convert geos_corners from dict -> wkt; for grouping regions
    data['geos_corners'] = data.geos_corners.apply(lambda x: shape(x).wkt)
    data['image_index'] = data.index

    # apply some filtering to scope dataset
    data = data[data['sensor_coarse'] == 'WV']
    data['has_required_channels'] = data['auxiliary'].apply(lambda x: _has_required_channels(x))
    data = data[data['has_required_channels'] is True]
    data = data.drop('has_required_channels', axis=1)

    return data


def _has_required_channels(aux_data):
    required_channels = _get_required_channels()
    available_channels = [x['channels'] for x in aux_data]
    return set(required_channels).issubset(available_channels)


def _get_required_channels():
    return ['red', 'green', 'blue']


def _create_rgb_image(coco_img: kwcoco.coco_image, bundle_dpath: Path):

    # Loads RGB image
    coco_delay = coco_img.delay(channels=_get_required_channels(), bundle_dpath=bundle_dpath)
    rgb_img = coco_delay.scale(0.2).finalize(nodata='float')  # TODO - rm scaling on release; scaling used to speed up image loading

    # Clip image to be between the 2nd and 98th percentiles
    rgb_img = rgb_img.astype(np.float32)
    percentile98 = np.percentile(rgb_img, 98)
    percentile2 = np.percentile(rgb_img, 2)

    if (percentile98 != percentile2):
        image = (rgb_img - percentile2) / (percentile98 - percentile2)
    else:
        image = rgb_img

    # Only values between 0 and 1 are valid
    image[image < 0] = 0
    image[image > 1] = 1

    # Convert image to uint8
    image = image * 255
    image = image.astype(np.uint8)

    return image


def _plot_images(images: list, num_cols=4):
    num_images = len(images)
    num_rows = math.ceil(num_images / num_cols)
    subplot_size = 5

    fig, axes = plt.subplots(num_rows, num_cols, figsize=(subplot_size * num_cols, subplot_size * num_rows))

    for i, ax in enumerate(axes.flat):
        if i < num_images:
            ax.imshow(images[i]['data'])
            ax.axis('off')
            ax.set_title(images[i]['title'])

    # plt.tight_layout()
    plt.show()


def _compute_bw_percentage(rgb):
    """
    Compute the percentage of black and white pixels in an RGB image.

    Parameters:
    rgb (np.ndarray): Input RGB image as a numpy array of shape (height, width, 3)

    Returns:
    float: Percentage of black pixels in the image
    float: Percentage of white pixels in the image
    """

    # Convert the RGB values to grayscale
    gray = np.dot(rgb[..., :3], [0.2989, 0.5870, 0.1140])

    # Compute the percentage of black pixels
    black_percentage = np.sum(gray == 0) / (gray.shape[0] * gray.shape[1])

    # Compute the percentage of white pixels
    white_percentage = np.sum(gray == 255) / (gray.shape[0] * gray.shape[1])

    return black_percentage, white_percentage


def _generate_gif(images: list, filename: Path, duration=0.1):

    # Save the list of images to a GIF animation file
    with imageio.get_writer(filename, mode='I', duration=duration) as writer:
        for image in images:
            overlayed_image = _add_text_overlay(image['data'], image['text'])
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
    text_x = int((width - text_width) / 10)
    text_y = int((height + text_height) / 10)

    # Add the text overlay to the image
    result = cv2.putText(np.copy(image), text, (text_x, text_y), font, font_scale, color, thickness, cv2.LINE_AA)

    return result


def _bw_range(value: float):
    fvalue = float(value)
    if (fvalue > 1.0) or (fvalue < 0.0):
        raise argparse.ArgumentTypeError(f'{fvalue} must be between 0 and 1; e.g. 0.5')
    return fvalue


def _get_args():
    parser = argparse.ArgumentParser(description='Marks useuable images that are from the WV sensor')
    parser.add_argument('-i', '--input', type=Path, required=True, help='input kwcoco json file')
    parser.add_argument('-o', '--output', type=Path, required=True, help='output kwcoco json file')
    parser.add_argument('-bp', '--black_percentage', type=lambda x: _bw_range(x), default=0.05, help='Acceptable percentage of pure black pixels; 0 (image cannot have any black pixels) - 1 (image can be pure black)')
    parser.add_argument('-wp', '--white_percentage', type=lambda x: _bw_range(x), default=0.05, help='Acceptable percentage of pure white pixels; 0 (image cannot have any white pixels) - 1 (image can be pure white)')
    parser.add_argument('--create_gif', type=bool, default=True, help='save stack of images as a gif to args.output.parent/_assets/filter')
    return parser.parse_args()


if __name__ == '__main__':

    args = _get_args()
    data = load_kwcoco(args.input)
    df = _preprocess_image_dataframe(pd.DataFrame(data.dataset['images']))

    # Group sub-regions using geos_corners; using string-representation of coordinates as uuid... seems to work...
    region_uuid = 0
    for region_name, region_df in tqdm(df.groupby('geos_corners'), desc='Regions'):

        region_df = region_df.sort_values(by=['timestamp'])
        stats = {'blanks' : 0, 'bad_gsd': 0, 'valid': 0, 'skipped': 0, 'total': len(region_df)}
        gif_images = []  # for gif-generations

        for row in region_df.itertuples():

            # Obtain CocoImage
            gid = row.id
            coco_img = data.coco_image(gid)

            # GSD filtering
            min_gsd_meter = np.array(coco_img.resolution()['mag']).min()
            if min_gsd_meter > 1:
                # TODO - up/down-sample all images to 1GSD?
                stats['bad_gsd'] = stats['bad_gsd'] + 1
                continue

            # Load image
            rgb_image = _create_rgb_image(coco_img, args.input.parent)

            # Validate image
            black_percentage, white_percentage = _compute_bw_percentage(rgb_image)

            # Check if image is blank; blank means image is mostly white or black
            if (black_percentage <= args.black_percentage) and (white_percentage < args.white_percentage):

                if args.create_gif is True:
                    text = '{}, ({:.3f},{:.3f})'.format(row.id, black_percentage, white_percentage)
                    gif_images.append({'data': rgb_image, 'text': text})

                data.dataset['images'][row.image_index]['is_valid_wv'] = True
                stats['valid'] = stats['valid'] + 1
            else:
                stats['blanks'] = stats['blanks'] + 1

        # gif-generation
        if len(gif_images) > 0:
            gif_file = args.output.parent / f'_assets/filter/{region_uuid}.gif'
            gif_file.parent.mkdir(parents=True, exist_ok=True)

            _generate_gif(gif_images, gif_file, 0.20)

            del gif_images  # clean-up memory

        # dump results and update region_uuid
        stats['skipped'] = abs(stats['total'] - stats['bad_gsd'] - stats['blanks'] - stats['valid'])
        print(f'{region_uuid} | {stats}')
        region_uuid += 1

    # Save results
    print(f'Saving results to... {args.output}')
    args.output.parent.mkdir(parents=True, exist_ok=True)

    with open(args.output, 'w') as f:
        json.dump(data.dataset, f)
