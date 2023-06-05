import os
import json
import hashlib
# import collections

import omegaconf
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# ImageSlice = collections.namedtuple("ImageSlice", ["height", "width", "scale", "stride"])


class ImageSlice:
    """Image slice object.
    """

    def __init__(self, height, width, stride):
        self.height = height
        self.width = width
        self.stride = stride

    def __str__(self):
        return f'ImageSlice(height={self.height}, width={self.width}, stride={self.stride})'


def generate_image_slice_object(height, width=None, stride=None, scale=1):
    """Create a more easily queried object for image dimension information.
    Args:
        height (int): Height of crop slice.
        width (int, optional): Width of crop slice. If None, then use equal to height. Defaults to None.
        stride (int, optional): Value to determine the amount to move a crop over an image vertically or
          horizontally. Defaults to None.
        scale (float, optional): Scale the height and width by this factor. Note: The scale is used to resize the
          height and width crop sizes. Defaults to 1.
    Returns:
        namedtuple: [description]
    """
    ImageSlice.height = height

    if width is None:
        width = height

    if stride is None:
        stride = height

    ImageSlice.width = width
    ImageSlice.scale = scale
    ImageSlice.stride = stride

    return ImageSlice


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


def get_repo_paths(path_name):
    # Get repo paths file.
    current_dir = os.path.dirname(os.path.abspath(__file__))
    repo_paths_path = os.path.join('/'.join(current_dir.split('/')[:-2]), 'repo_paths.json')
    try:
        select_path = json.load(open(repo_paths_path, 'r'))[path_name]
    except KeyError:
        raise KeyError(f'No path name "{path_name}" found in {repo_paths_path}')
    return select_path


def get_secrets(secret_name):
    # Get repo paths file.
    current_dir = os.path.dirname(os.path.abspath(__file__))
    secrets_path = os.path.join('/'.join(current_dir.split('/')[:-2]), 'secrets.json')
    try:
        secret = json.load(open(secrets_path, 'r'))[secret_name]
    except KeyError:
        raise KeyError(f'No secret named "{secret_name}" found in {secrets_path}')
    return secret


def sigmoid(x):
    return 1 / (1 + np.exp(x))


def load_cfg_file(path):
    with open(path, "r") as fp:
        cfg = omegaconf.OmegaConf.load(fp.name)
    cfg = populate_old_config(cfg)
    return cfg


def create_conf_matrix_pred_image(pred, target):
    # Pred: numpy
    # target: numpy
    out_image = np.zeros([pred.shape[0], pred.shape[1], 3], dtype='uint8')

    # TP - White
    x, y = np.where((pred == 1) & (target == 1))
    out_image[x, y, :] = np.array([255, 255, 255])

    # FP - Teal
    x, y = np.where((pred == 1) & (target == 0))
    out_image[x, y, :] = np.array([0, 255, 255])

    # FN - Red
    x, y = np.where((pred == 0) & (target == 1))
    out_image[x, y, :] = np.array([255, 0, 0])

    return out_image


def create_gif(image_list,
               save_path,
               fps=1,
               image_text=None,
               fontpct=5,
               overlay_images=None,
               optimize=False):
    """Create a gif image from a collection of numpy arrays.

    Args:
        image_list (list[numpy array]): A list of images in numpy format of type uint8.
        save_path (str): Path to save gif file.
        fps (float, optional): Frames per second. Defaults to 1.
        image_text (list[str], optional): A list of text to add to each frame of the gif.
            Must be the same length as image_list.
    """

    # Check dtype of images in image list.
    assert isinstance(image_list, list), f'image_list must be a list. Not type {type(image_list)}'
    assert all([img.dtype == 'uint8' for img in image_list]), 'Not all images are of type uint8.'

    if len(image_list) < 2:
        print(f'Cannot create a GIF with less than 2 images, only {len(image_list)} provided.')
        return None
    elif len(image_list) == 2:
        img, imgs = Image.fromarray(image_list[0]), [Image.fromarray(image_list[1])]
    else:
        img, *imgs = [Image.fromarray(img) for img in image_list]

    if overlay_images is not None:
        assert len(overlay_images) == len(image_list)

        # Overlay images together
        images = [img]
        images.extend(imgs)

        images_comb = []
        for image_1, image_2 in zip(images, overlay_images):
            # Make sure images have alpha channel
            image_1.putalpha(1)
            image_2.putalpha(1)

            # Overlay images
            image_comb = Image.alpha_composite(image_1, image_2)
            images_comb.append(image_comb)

        img, *imgs = [img for img in images_comb]

    if image_text is not None:
        assert len(image_text) == len(image_list)

        # Have an issue loading larger font
        H = image_list[0].shape[0]
        if fontpct is None:
            font = ImageFont.load_default()
        else:
            if H < 200:
                font = ImageFont.load_default()
            else:
                fontsize = int(H * fontpct / 100)
                # Find fonts via "locate .ttf"
                try:
                    font = ImageFont.truetype("/usr/share/fonts/truetype/ubuntu/Ubuntu-R.ttf",
                                              fontsize)
                except:  # NOQA
                    print('Cannot find font at: /usr/share/fonts/truetype/ubuntu/Ubuntu-R.ttf')
                    font = ImageFont.load_default()

        images = [img]
        images.extend(imgs)
        for i, (img, text) in enumerate(zip(images, image_text)):
            draw = ImageDraw.Draw(img)
            draw.text((0, 0), text, (255, 0, 0), font=font)
            images[i] = img

        img, *imgs = images

    # Convert the images to higher quality
    images = [img]
    images.extend(imgs)
    img, *imgs = [img.quantize(dither=Image.NONE) for img in images]

    duration = int(1000 / fps)
    img.save(fp=save_path,
             format='GIF',
             append_images=imgs,
             save_all=True,
             duration=duration,
             loop=0,
             optimize=optimize)


def create_hash_name(input_str, method_name='sha256'):
    if method_name == 'sha256':
        sha = hashlib.sha256()
        sha.update(input_str.encode())
        output_str = sha.hexdigest()
    else:
        raise NotImplementedError(f'Not method "{method_name}" for hashing.')

    return output_str


def populate_old_config(cfg):
    attributes = {
        'log_image_iter': 100,
    }
    for attr_name, default_value in attributes.items():
        if hasattr(cfg, attr_name) is False:
            setattr(cfg, attr_name, default_value)

    return cfg


def create_hash_str(method_name='sha256', **kwargs):

    hashed_str = ''
    for name, value in kwargs.items():
        if isinstance(value, str):
            pass
        if isinstance(value, list):
            kwargs[name] = '_'.join(value)
        else:
            kwargs[name] = str(value)

        hashed_str += kwargs[name]

    if method_name == 'sha256':
        sha = hashlib.sha256()
        sha.update(hashed_str.encode())
        output_str = sha.hexdigest()
    else:
        raise NotImplementedError(f'Not method "{method_name}" for hashing.')

    return output_str
