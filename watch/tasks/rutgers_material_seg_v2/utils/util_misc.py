from datetime import datetime

import numpy as np
import omegaconf

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

def sigmoid(x):
    return 1 / (1 + np.exp(x))



def load_cfg_file(path):
    with open(path, "r") as fp:
        cfg = omegaconf.OmegaConf.load(fp.name)
    return cfg


def colorize_material_mask(mat_pred):
    """TODO: _summary_

    Args:
        mat_pred (np.array): A int numpy array of shape [height, width] containing values corresponding to material classes.

    Returns:
        TODO: _type_: _description_
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


def date_from_image_name(image_name):
    date_str = image_name.split('_')[1]
    year = int(date_str[:4])
    month = int(date_str[4:6])
    day = int(date_str[6:8])
    return datetime(year, month, day)


def softmax(vector, axis=None):
    e = np.exp(vector)

    try:
        sm = e / e.sum(axis=axis)
    except ValueError:
        sm = e / e.sum(axis=axis)[:, None]
    return sm

class VideoSlice:
    def __init__(self, height, width, n_frames, scale, stride):
        self.height = height

        if width is None:
            width = height

        self.width = width
        self.n_frames = n_frames
        self.scale = scale
        self.stride = stride


def generate_video_slice_object(height, width=None, n_frames=None, scale=None, stride=None):
    """Create a more easily queried object for image dimension information.

    Args:
        height (int): Height of crop slice.
        width (int, optional): Width of crop slice. If None, then use equal to height. Defaults to None.
        n_frames (int, optional): Number of frames to sample from video data. Defaults to None.
        scale (float, optional): Scale the height and width by this factor. Note: The scale is used to resize the 
          height and width crop sizes. Defaults to None.
        stride (int, optional): Value to determine the amount to move a crop over an image vertically or
          horizontally. Defaults to None.

    Returns:
        namedtuple: [description]
    """
    # VideoSlice = collections.namedtuple('VideoSlice', ['height', 'width', 'n_frames', 'scale', 'stride'])
    # VideoSlice.height = height

    if width is None:
        width = height


    video_slice = VideoSlice(height, width, n_frames, scale, stride)

    return video_slice

def get_crop_slices(height, width, crop_height, crop_width, step=None, mode='exact'):
    """Given an image size and desried crop, return all possible crop slices over space.

    Args:
        height (int): The height of the image to be cropped (y-axis).
        width (int): The width of the image to be cropped (x-axis).
        crop_height (int): The size of the crop height. Note: For certain modes, 
            e.g. mode = 'under', crop height must be less than original image height.
        crop_width (int): The size of the crop width. Note: For certain modes, 
            e.g. mode = 'under', crop width must be less than original image width.
        step (int): Distance in pixels to move crop window, defauls to size of the crop along that direction, i.e. no overlap.
        mode (str, optional): Method for how to handle edge cases. Defaults to 'exact'.
            - exact: Returns slices that do not go over original image size
            - over: Returns slices that have fixed crop size, covers full image
            - under: Returns slices that have fixed crop size, may not cover full image

    Raises:
        NotImplementedError: If invalid crop mode given.

    Returns:
        list: A list of crop slices. Each crop slice has the following form [h0, w0, h, w].
    """
    if step is not None:
        if type(step) is tuple:
            h_step, w_step = step[0], step[1]
        elif type(step) is int:
            h_step, w_step = step, step
        else:
            raise TypeError(f'Invalid step type: {type(step)}')

        if h_step <= 0:
            raise ValueError(f'Step of size {h_step} is too small.')
        if w_step <= 0:
            raise ValueError(f'Step of size {w_step} is too small.')

        if h_step > height:
            raise ValueError(f'Step of size {h_step} is too large for height {height}')
        if w_step > width:
            raise ValueError(f'Step of size {w_step} is too large for width {width}')
    else:
        # No step so use crop size for height.
        h_step, w_step = crop_height, crop_width

    crop_slices = []
    if mode == 'over':
        num_h_crops = 0
        while True:
            if ((num_h_crops * h_step) + crop_height) > height:
                break
            num_h_crops += 1
        num_w_crops = 0
        while True:
            if ((num_w_crops * w_step) + crop_width) > width:
                break
            num_w_crops += 1
        num_h_crops += 1
        num_w_crops += 1

        for i in range(num_h_crops):
            for j in range(num_w_crops):
                crop_slices.append([i * h_step, j * w_step, crop_height, crop_width])
    elif mode == 'under':
        num_h_crops = 0
        while True:
            if ((num_h_crops * h_step) + crop_height) > height:
                break
            num_h_crops += 1
        num_w_crops = 0
        while True:
            if ((num_w_crops * w_step) + crop_width) > width:
                break
            num_w_crops += 1

        for i in range(num_h_crops):
            for j in range(num_w_crops):
                crop_slices.append([i * h_step, j * w_step, crop_height, crop_width])
    elif mode == 'exact':
        # Get number of crops fit in target image
        num_h_crops = 0
        while True:
            if ((num_h_crops * h_step) + crop_height) > height:
                break
            num_h_crops += 1
        num_w_crops = 0
        while True:
            if ((num_w_crops * w_step) + crop_width) > width:
                break
            num_w_crops += 1

        for i in range(num_h_crops):
            for j in range(num_w_crops):
                crop_slices.append([i * h_step, j * w_step, crop_height, crop_width])

        # Get the remaining portion of the images
        rem_h = height - (num_h_crops * h_step)
        rem_w = width - (num_w_crops * w_step)

        # Get reminder crops along width axis
        if rem_w != 0:
            for i in range(num_h_crops):
                crop_slices.append([i * h_step, num_w_crops * w_step, crop_height, rem_w])

        # Get reminder crops along height axis
        if rem_h != 0:
            for j in range(num_w_crops):
                crop_slices.append([num_h_crops * h_step, j * w_step, rem_h, crop_height])

        # Get final crop corner
        if (rem_h != 0) and (rem_w != 0):
            crop_slices.append([num_h_crops * h_step, num_w_crops * w_step, rem_h, rem_w])
    else:
        raise NotImplementedError(f'Invalid mode: {mode}')

    return crop_slices
