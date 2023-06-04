import os
import argparse

import kwcoco
import numpy as np
from tqdm import tqdm

from watch.tasks.rutgers_material_seg_v2.matseg.utils.utils_misc import create_gif
from watch.tasks.rutgers_material_seg_v2.matseg.utils.utils_dataset import colorize_material_mask


def normalize_rgb(image, gamma=0.6):
    """Convert a raw image to beautiful RGB color.

    Args:
        image (np.ndarray): A uint16 tensor of shape (H, W, 3).
        gamma (float, optional): The factor to adjust color values. Defaults to 0.6.

    Returns:
        np.ndarray: The uint8 tensor of shape (H, W, 3).
    """
    # Normalize from uint16 to float32.
    image = np.clip(image / 2**16, 0, 1)

    # Adjust color balance.
    image *= 2**4
    image = np.clip(image**gamma, 0, 1)

    # Convert to uint8.
    image = (image * 255).astype('uint8')

    return image


def rgbize_features(features):
    """_summary_

    Args:
        features (_type_): A 

    Returns:
        np.ndarray: _description_
    """
    if features.shape[-1] == 1:
        feat_img = np.repeat(features, 3, axis=-1)
    else:
        feat_img = features[..., :3]
    feat_img = np.nan_to_num(feat_img)
    max_value = np.quantile(feat_img, 0.95, axis=(0, 1, 2))
    min_value = np.quantile(feat_img, 0.05, axis=(0, 1, 2))
    feat_img = (feat_img - min_value) / (max_value - min_value)

    # Convert to uint8.
    feat_img = (feat_img * 255).astype('uint8')

    return feat_img


def visualize_material_features():
    parser = argparse.ArgumentParser()
    parser.add_argument('kwcoco_fpath', type=str, help='Path to kwcoco file')
    parser.add_argument('save_dir', type=str, help='Path to save images.')
    args = parser.parse_args()

    # Check if save directory exists.
    os.makedirs(args.save_dir, exist_ok=True)

    # Load kwcoco file.
    dset = kwcoco.CocoDataset.coerce(args.kwcoco_fpath)

    # For each video load material and RGB frames, visualize with GIF.
    video_ids = list(dset.videos())
    for video_id in tqdm(video_ids, desc='Video'):
        # Get all image ids for video.
        video_image_ids = dset.index.vidid_to_gids[video_id]

        mat_rgb_imgs, metadata_strings = [], []
        region_name = None
        for image_id in tqdm(video_image_ids, desc='Image', leave=False):
            coco_image = dset.coco_image(image_id)

            # Get region name.
            if region_name is None:
                region_name = coco_image.video['name']

            # TODO: Check if the materials are empty for this image.
            # missing_material_instance = False
            # for asset in coco_image['assets']:
            # if not .has('materials'):
            #     missing_material_images += 1
            #     continue

            # Load and normalize RGB image.
            rgb_image = coco_image.delay('red|green|blue', space='video',
                                         resolution='10GSD').finalize()
            rgb_image = normalize_rgb(rgb_image)

            # Load and colorize material image.
            materials = coco_image.delay('materials', space='video', resolution='10GSD').finalize()
            materials = np.argmax(materials, axis=-1)
            rgb_mats = colorize_material_mask(materials)

            # Load and colorize feature image.
            mat_feats = coco_image.delay('mat_feats', space='video', resolution='10GSD').finalize()
            mat_feat_img = rgbize_features(mat_feats)

            # Combine images and append to list.
            mat_rgb_imgs.append(np.concatenate([rgb_image, rgb_mats, mat_feat_img], axis=1))

            # Get image metadata in string format.

            ## Get date of image taken.
            img_name = str(coco_image.primary_image_filepath()).split('/')[-1]
            datetime_str = img_name.split('_')[1]
            year = datetime_str[:4]
            month = datetime_str[4:6]
            day = datetime_str[6:8]

            ## Get sensor type.
            try:
                sensor_type = coco_image.primary_asset()['sensor_coarse']
            except KeyError:
                sensor_type = None

            if sensor_type is None:
                metadata_string = f'{year}-{month}-{day} {image_id}'
            else:
                metadata_string = f'{year}-{month}-{day} {sensor_type} {image_id}'

            metadata_strings.append(metadata_string)

        # Generate GIF.
        save_path = f'{args.save_dir}/{region_name}.gif'
        create_gif(mat_rgb_imgs, save_path, image_text=metadata_strings)


if __name__ == '__main__':
    visualize_material_features()
