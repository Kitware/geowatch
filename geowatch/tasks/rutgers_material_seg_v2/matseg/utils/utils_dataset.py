import os
import sys
import pickle

import numpy as np
from glob import glob
from tqdm import tqdm

from geowatch.tasks.rutgers_material_seg_v2.matseg.utils.utils_misc import get_repo_paths

MATERIAL_TO_MATID = {
    'unknown': 0,
    'water': 1,
    'soil': 2,
    'metal': 3,
    'vegetation': 4,
    'snow': 5,
    'polymer': 6,
    'asphalt': 7,
    'concrete': 8
}

MATID_TO_MATERIAL = dict((v, k) for k, v in MATERIAL_TO_MATID.items())

MATERIAL_TO_COLOR = {
    'unknown': [0, 0, 0],
    'water': [0, 255, 255],
    'soil': [171, 105, 0],
    'metal': [255, 255, 0],
    'vegetation': [13, 209, 65],
    'snow': [0, 72, 255],
    'polymer': [255, 0, 247],
    'asphalt': [117, 117, 117],
    'concrete': [222, 222, 222]
}


def colorize_material_mask(mat_pred):
    """Convert material prediction image to an RGB mask corresponding to material colors.

    Args:
        mat_pred (np.array): An int numpy array of shape [height, width] containing values
            corresponding to material classes.

    Returns:
        np.array: A numpy array of type `uint8` with shape [height, width, 3].
    """

    height, width = mat_pred.shape
    mat_color_mask = np.zeros([height, width, 3], dtype='uint8')

    mat_ids = list(np.unique(mat_pred))

    for mat_id in mat_ids:
        if mat_id != 0:
            x, y = np.where(mat_pred == mat_id)

            material = MATID_TO_MATERIAL[mat_id]
            mat_color = MATERIAL_TO_COLOR[material]

            mat_color_mask[x, y, :] = mat_color

    return mat_color_mask


def get_image_path_from_dir(image_dir, channel_name):
    image_paths = glob(image_dir + '/*.tif')

    channel_image_path = None
    for image_path in image_paths:
        if channel_name in image_path:
            channel_image_path = image_path

    if channel_image_path is None:
        raise FileNotFoundError(f'Cannot find "{channel_name}" in image directory "{image_dir}"')

    return channel_image_path


def paste_annotations_onto_mask(annos, mat_to_num, height, width, simplified_taxonomy=False):
    """Get a mask of material labels from a list of annotations.

    Args:
        annos (??): An object containing annotations.
        mat_to_num (Dict[str, int]): A dictionary mapping material names to material ids.
        height (int): The height of the canvas to paste the annotations onto.
        width (int): The width of the canvas to paste the annotations onto.
        simplified_taxonomy (bool, optional): Group some of the material together.
            Defaults to False.

    Returns:
        numpy.ndarray: A numpy array of type `uint8` with shape [height, width].
    """
    canvas = np.zeros([height, width], dtype='uint8')
    for annotation in tqdm(annos, file=sys.stdout):
        rgb_canvas = annotation.value.draw(height=height, width=width, color=[1, 1, 1])
        x, y = np.where(rgb_canvas.sum(axis=2) > 1)
        mat_name = annotation.name.lower()
        if simplified_taxonomy:
            if mat_name == 'metal':
                mat_name = 'polymer'
            elif mat_name == 'concrete':
                mat_name = 'asphalt'
        canvas[x, y] = mat_to_num[mat_name]
    return canvas


def get_lb_anno_mask(label, simplified_taxonomy=True):
    """From a labelbox label, get the annotation mask.

    Args:
        label (_type_): A LabelBox annotation object.
        simplified_taxonomy (bool, optional): Convert some material values to others.
            Defaults to True.

    Raises:
        FileExistsError: Raises if the image directory cannot be found.

    Returns:
        list: A list of dictionaries containing the following keys: label, mat_mask, image_dir,
            region_name.
        np.array: An array of shape n_materials containing the number of pixels found per material.
    """
    # Initialize mat dict.
    mat_dict = np.zeros(len(MATERIAL_TO_MATID.keys()))
    external_id_split = label.data.external_id.split('_')
    region_name = '_'.join(external_id_split[:2])
    image_name = '_'.join(external_id_split[2:])[:-4]

    print(f'{region_name}: {image_name}', flush=True)

    # FIXME: Hack
    if (region_name == 'KR_R001') & (
            image_name == 'crop_20181025T020000Z_N37.643680E128.649453_N37.683356E128.734073_S2_1'):
        image_name = 'crop_20181025T020000Z_N37.643680E128.649453_N37.683356E128.734073_S2_0'

    dataset_root_dir = '/data4/datasets/dvc-repos/smart_data_dvc/Drop4-BAS/'
    image_dir = os.path.join(dataset_root_dir, region_name, 'S2', 'affine_warp', image_name)

    if os.path.exists(image_dir) is False:
        raise FileExistsError(f'Cannot locate: "{image_dir}"')

    # Get image height and width.
    height, width = label.data.value.shape[0], label.data.value.shape[1]

    # Create annotation mask.
    mask = paste_annotations_onto_mask(label.annotations,
                                       MATERIAL_TO_MATID,
                                       height,
                                       width,
                                       simplified_taxonomy=simplified_taxonomy)

    # Update material distribution.
    unique_values = list(np.unique(mask))
    for value in unique_values:
        if value == 0:
            continue

        if simplified_taxonomy:
            # Map: metal (3) -> polymer (7)
            if value == MATERIAL_TO_MATID['metal']:
                value = MATERIAL_TO_MATID['polymer']

            # Map: concrete (8) -> asphalt (7)
            if value == MATERIAL_TO_MATID['concrete']:
                value = MATERIAL_TO_MATID['asphalt']

        x, _ = np.where(mask == value)
        mat_dict[value] += x.shape[0]

    mat_label_info = {
        'label': label,
        'mat_mask': mask,
        'image_dir': image_dir,
        'region_name': region_name
    }

    return mat_label_info, mat_dict


def load_region_bas_annos(drop_version=4, gsd=10):
    """Find path of pickle file and load BAS annos as dictionary with region_name keys.

    Args:
        drop_version (int, optional): Name of the drop version to get BAS annos for. Defaults to 4.

    Raises:
        NotImplementedError: If the BAS annos have not been created for the specified drop version.

    Returns:
        dict[region_name]: np.array: A dictionary with region_name keys and np.array BAS annos
            as values.
    """
    if drop_version == 4:
        if gsd is None:
            bas_anno_path = get_repo_paths('bas_annos')
        else:
            raise NotImplementedError
    elif drop_version == 6:
        if gsd == 10 or gsd is None:
            bas_anno_path = get_repo_paths('bas_annos_10GSD')
        else:
            raise NotImplementedError
    else:
        raise NotImplementedError(f'No BAS annos predicted for drop version "{drop_version}"')

    bas_annos = pickle.load(open(bas_anno_path, 'rb'))
    return bas_annos
