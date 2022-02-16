import os
import json
import hydra
import torch
import collections
from omegaconf import OmegaConf
from PIL import Image, ImageDraw, ImageFont


def get_device():
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
        print("No GPUs are available on system.")
    return device


def get_repo_dir():
    try:
        base_dir = hydra.utils.get_original_cwd()
    except ValueError:
        base_dir = os.getcwd()
    return base_dir


def load_cfg_file(path):
    with open(path, "r") as fp:
        cfg = OmegaConf.load(fp.name)
    return cfg


def create_gif(image_list, save_path, fps=1, image_text=None, fontpct=5, overlay_images=None, optimize=False):
    """Create a gif image from a collection of numpy arrays.

    Args:
        image_list (list[numpy array]): A list of images in numpy format of type uint8.
        save_path (str): Path to save gif file.
        fps (float, optional): Frames per second. Defaults to 1.
        image_text (list[str], optional): A list of text to add to each frame of the gif.
            Must be the same length as iimage_list.
    """

    # Check dtype of images in image list.
    assert type(image_list) is list
    assert all([img.dtype == "uint8" for img in image_list])

    if len(image_list) < 2:
        print(f"Cannot create a GIF with less than 2 images, only {len(image_list)} provided.")
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
                    font = ImageFont.truetype("/usr/share/fonts/truetype/ubuntu/Ubuntu-R.ttf", fontsize)
                except FileNotFoundError:
                    print("Cannot find font at: /usr/share/fonts/truetype/ubuntu/Ubuntu-R.ttf")
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
    img.save(
        fp=save_path, format="GIF", append_images=imgs, save_all=True, duration=duration, loop=0, optimize=optimize
    )


def save_to_json(dict_obj, save_path):
    with open(save_path, "w") as f:
        json.dump(dict_obj, f, indent=4, sort_keys=True)


def get_n_frames(n_frames, task_mode):

    if n_frames is not None:
        if task_mode not in ["ss_triplet", "ss_mat_recon"]:
            assert n_frames > 1

    return n_frames


def flatten_deep_dictionary(d, parent_key="", sep="_"):
    # https://stackoverflow.com/questions/6027558/flatten-nested-dictionaries-compressing-keys
    if type(d).__name__ == "DictConfig":
        d = OmegaConf.to_container(d)

    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, collections.MutableMapping):
            items.extend(flatten_deep_dictionary(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def get_crop_slices(height, width, crop_height, crop_width, step=None, mode="exact"):
    """Given an image size and desried crop, return all possible crop slices over space.

    Args:
        height (int): The height of the image to be cropped (y-axis).
        width (int): The width of the image to be cropped (x-axis).
        crop_height (int): The size of the crop height. Note: For certain modes,
            e.g. mode = 'under', crop height must be less than original image height.
        crop_width (int): The size of the crop width. Note: For certain modes,
            e.g. mode = 'under', crop width must be less than original image width.
        step (int): TODO. Defaults to None.
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
            raise TypeError(f"Invalid step type: {type(step)}")

        if h_step > height:
            raise ValueError(f"Step of size {h_step} is too large for height {height}")
        if w_step > width:
            raise ValueError(f"Step of size {w_step} is too large for width {height}")
    else:
        # No step so use crop size for height.
        h_step, w_step = crop_height, crop_width

    crop_slices = []
    if mode == "over":
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
    elif mode == "under":
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
    elif mode == "exact":
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
        raise NotImplementedError(f"Invalid mode: {mode}")

    return crop_slices


def generate_video_slice_object(height, width=None, n_frames=None, scale=None, stride=None):
    """[summary]

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
    VideoSlice = collections.namedtuple("VideoSlice", ["height", "width", "n_frames", "scale"])
    VideoSlice.height = height

    if width is None:
        width = height

    VideoSlice.width = width
    VideoSlice.n_frames = n_frames
    VideoSlice.scale = scale
    VideoSlice.stride = stride

    return VideoSlice


class EmptyScheduler:
    """A class that has the methods of a scheduler object but does nothing."""

    def __init__(self):
        pass

    def step(self):
        pass


def to_numpy(x):
    try:
        return x.detach().cpu().numpy()
    except AttributeError:
        return x.detach().numpy()


def update_gradient_map(model, gradmap):
    for tag, value in model.named_parameters():
        if value.grad is not None:
            gradmap[tag] = value.grad.cpu()
    return gradmap


def log_gradients(experiment, gradmap, step):
    for k, v in gradmap.items():
        experiment.log_histogram_3d(to_numpy(v), name=k, step=step)


def log_weights(model, step):
    for name, layer in zip(model._modules, model.children()):
        if "activ" in name:
            continue
