import os

import cv2
import kwimage
import numpy as np
from PIL import Image
from glob import glob
from tqdm import tqdm
from osgeo import gdal
from tifffile import tifffile

from watch.tasks.rutgers_material_seg_v2.matseg.utils.utils_misc import sigmoid


def load_S2_image(image_dir, channels, crop_slice=None, backend='gdal'):
    """_summary_

    Args:
        image_dir (_type_): Assumes image paths are sorted.
        channels (str): TODO
        crop_slice (_type_): _description_
        backend (str, optional): _description_. Defaults to 'gdal'.
    """

    def load_image_file(image_path, crop_info, backend_name='gdal'):
        if crop_info is None:
            if backend_name == 'tifffile':
                frame = tifffile.imread(image_path)
            elif backend_name == 'gdal':
                frame = gdal.Open(image_path).ReadAsArray()
            else:
                raise NotImplementedError(
                    f'No implementation for loading images with "{backend_name}" backend.')
        else:
            h0, w0, h, w = crop_info
            if backend_name == 'tifffile':
                frame = tifffile.imread(image_path)[h0:(h0 + h), w0:(w0 + w)]
            elif backend_name == 'gdal':
                frame = gdal.Open(image_path).ReadAsArray(w0, h0, w, h)
            else:
                raise NotImplementedError(
                    f'No implementation for loading images with "{backend_name}" backend.')
        return frame

    # Get all band images in the
    channel_image_paths = sorted(glob(image_dir + '/*.tif'))
    band_names = [bp.split('_')[-1][:-4] for bp in sorted(channel_image_paths)]

    if channels == 'RGB':
        # Choose RGB bands
        red_band_index = band_names.index('red')
        green_band_index = band_names.index('green')
        blue_band_index = band_names.index('blue')
        r_channel_img_path = channel_image_paths[red_band_index]
        g_channel_img_path = channel_image_paths[green_band_index]
        b_channel_img_path = channel_image_paths[blue_band_index]

        r_channel_img = load_image_file(r_channel_img_path, crop_slice, backend_name=backend)
        if r_channel_img.sum() == 0:
            return None
        g_channel_img = load_image_file(g_channel_img_path, crop_slice, backend_name=backend)
        b_channel_img = load_image_file(b_channel_img_path, crop_slice, backend_name=backend)

        frame = np.stack([r_channel_img, g_channel_img, b_channel_img], axis=0)
    elif channels == 'RGB_NIR':
        red_band_index = band_names.index('red')
        green_band_index = band_names.index('green')
        blue_band_index = band_names.index('blue')
        nir_band_index = band_names.index('nir')

        r_channel_img_path = channel_image_paths[red_band_index]
        g_channel_img_path = channel_image_paths[green_band_index]
        b_channel_img_path = channel_image_paths[blue_band_index]
        nir_channel_img_path = channel_image_paths[nir_band_index]

        r_channel_img = load_image_file(r_channel_img_path, crop_slice, backend_name=backend)
        if r_channel_img.sum() == 0:
            return None

        g_channel_img = load_image_file(g_channel_img_path, crop_slice, backend_name=backend)
        b_channel_img = load_image_file(b_channel_img_path, crop_slice, backend_name=backend)
        nir_channel_img = load_image_file(nir_channel_img_path, crop_slice, backend_name=backend)

        frame = np.stack([r_channel_img, g_channel_img, b_channel_img, nir_channel_img], axis=0)

    elif channels == 'cloudmask':
        cloudmask_index = band_names.index('cloudmask')
        cloud_channel_img_path = channel_image_paths[cloudmask_index]
        frame = load_image_file(cloud_channel_img_path, crop_slice, backend_name=backend)

    elif channels == 'ALL':

        height, width = tifffile.imread(channel_image_paths[3]).shape
        channel_images = []
        for img_path in channel_image_paths:
            if 'cloudmask' in img_path:
                # Do not add this to image.
                continue

            if backend == 'tifffile':

                channel_image = tifffile.imread(img_path)
                if channel_image.sum() == 0:
                    return None

                h_img, w_img = channel_image.shape

                if (h_img != height) or (w_img != width):
                    channel_image = cv2.resize(channel_image, dsize=(width, height))

                # Get crop subset.
                h0, w0, h, w = crop_slice
                channel_image = channel_image[h0:(h0 + h), w0:(w0 + w)]

                channel_images.append(channel_image)

            elif backend == 'gdal':
                # Get channel resolution.
                img_ds = gdal.Open(img_path)
                h0, w0, h, w = crop_slice
                c_height, c_width = img_ds.RasterYSize, img_ds.RasterXSize

                # Load cropped version of channel image.
                h_adj = round(height / c_height)
                w_adj = round(width / c_width)
                if (c_height != height) or (c_width != width):
                    # Adjust crop slice.
                    if h_adj != w_adj:
                        breakpoint()
                        pass
                    # assert h_adj == w_adj, 'Images should be equally scaled'

                    c_h0 = h0 // h_adj
                    c_w0 = w0 // h_adj
                    c_h = h // h_adj
                    c_w = w // h_adj

                    c_h = np.clip(c_h, 1, a_max=None).astype(int)
                    c_w = np.clip(c_w, 1, a_max=None).astype(int)

                    wE = c_w0 + c_w
                    if wE >= c_width:
                        c_w = c_width - c_w0
                    hE = c_h0 + c_h
                    if hE >= c_height:
                        c_h = c_height - c_h0
                    crop = img_ds.ReadAsArray(int(c_w0), int(c_h0), int(c_w), int(c_h))
                else:
                    # Just crop
                    wE = w0 + w
                    if wE >= width:
                        w = width - w0
                    hE = h0 + h
                    if hE >= height:
                        h = height - h0
                    crop = img_ds.ReadAsArray(int(w0), int(h0), int(w), int(h))

                if crop is None:
                    breakpoint()
                    pass

                # Scale the crop if not correct resolution.
                if (crop.shape[0] != h) or (crop.shape[1] != w):
                    crop = cv2.resize(crop, dsize=(w, h))

                channel_images.append(crop)

        try:
            frame = np.stack(channel_images, axis=0)
        except:
            hh, ww = channel_images[0].shape
            new_channel_images = []
            for ci in channel_images:
                if (ci.shape[0] != hh) or (ci.shape[1] != ww):
                    new_channel_images.append(ci[:hh, :ww])
                else:
                    new_channel_images.append(ci)
            frame = np.stack(new_channel_images, axis=2)
        if crop_slice is None:
            assert (frame.shape[0] == height) and (frame.shape[1] == width)
    else:
        raise NotImplementedError(
            f'No implementation for loading S2 images with "{channels}" channels.')

    return frame


def load_norm_image(kwcoco_dset,
                    image_id,
                    crop_slice=None,
                    channels=None,
                    image_scale_mode='direct'):
    """Load an image from kwcoco file and normalize to float.

    Args:
        image_id (int): Image ID (aka gid) to use as reference in kwcoco dataset.
        crop_slice (list, optional): Crop image to this rectangle, [h0, w0, hE, wE]. Defaults to None.
        channels (str, optional): Language specifying which channels to load from an image. Defaults to None.

    Returns:
        np.array: A float array of shape [channels, height, width].
    """

    img_delayed = kwcoco_dset.delayed_load(image_id, channels=channels)

    # Crop image (optional).
    if crop_slice is not None:
        h0, w0, dh, dw = crop_slice
        hE, wE = h0 + dh, w0 + dw
        img_delayed = img_delayed.crop((slice(h0, hE), slice(w0, wE)))

    # Finalize image loading.
    image = img_delayed.finalize()  # [height, width, channels]

    image = image.transpose(2, 0, 1)  # [channels, height, width]

    # Handle cases of NaNs.
    if np.isnan(image.max()):
        image = np.nan_to_num(image).astype('uint16')

    # Normalize image to [0, 1] range.
    image = scale_image(image, scale_mode=image_scale_mode)

    return image


def scale_image(image, scale_mode='direct'):
    """Scales the pixel values of image.

    Args:
        image (np.array): A numpy array of shape [channels, height, width].
        scale_mode (str, optional): Parameter to determine how to scale image. Defaults to 'direct'.

    Returns:
        TODO: 
    """

    def __direct_scale(image):
        """Map image values from whatever range to [0,1].

        Args:
            image (np.array): A np.array of any shape.

        Raises:
            NotImplementedError: Throws error when dtype conversion is not set.

        Returns:
            np.array: A np.arrary of any shape but with values between [0, 1].
        """
        if image.dtype == 'uint16':
            scaled_image = image / 2**16
        elif image.dtype == 'int16':
            scaled_image = image / 2**16
        elif image.dtype == 'float32':
            scaled_image = image
        else:
            raise NotImplementedError(
                f'method not implemented for scaling "{image.dtype}" type to [0,1] range.')

        # Make sure image is bounded between 0 and 1.
        scaled_image = np.clip(scaled_image, 0, 1)

        return scaled_image

    if scale_mode == 'direct':
        scaled_image = __direct_scale(image)
    elif scale_mode == 'sigmoid':
        # Idea taken from Dynamic World paper.

        # Apply log to values in image.
        log_image = np.log(image)

        # Apply sigmoid to map to [0,1] range.
        scaled_image = sigmoid(log_image)

    elif scale_mode == 'unit':
        scaled_image = image / np.linalg.norm(axis=0)

    else:
        raise NotImplementedError(f'Scale mode "{scale_mode}" not implemented.')

    return scaled_image


class ImageStitcher:

    def __init__(self, save_dir, save_backend='gdal', save_ext='.tif'):
        """Class to combine crops for a single type of image.

        Args:
            save_dir (str): TODO: _description_
        """
        self.save_dir = save_dir
        self.save_ext = save_ext
        self.save_backend = save_backend

        # Create save directory if it does not already exist.
        os.makedirs(save_dir, exist_ok=True)

        # Initialize canvas
        self.image_canvas = {}
        self.weight_canvas = {}

    def add_images(self, images, image_names, crop_info, og_heights, og_widths, image_weights=None):
        # Populate image weights if it is None.
        if image_weights is None:
            image_weights = [None] * len(images)

        for img, name, crop, og_height, og_width, img_weight in zip(images, image_names, crop_info,
                                                                    og_heights, og_widths,
                                                                    image_weights):
            self.add_image(img, name, crop, og_height, og_width, image_weight=img_weight)

    def add_image(self, image, image_name, crop_info, og_height, og_width, image_weight=None):
        # Get crop info.
        h0, w0, dh, dw = crop_info
        # h0, w0, dh, dw = h0.item(), w0.item(), dh.item(), dw.item()
        hE, wE = h0 + dh, w0 + dw

        # Check if canvas image has already been generated.
        try:
            self.image_canvas[image_name]
            self.weight_canvas[image_name]
        except KeyError:
            # Create canvases for this image.

            ## Initialize canvas for this image.
            if len(image.shape) == 2:
                self.image_canvas[image_name] = np.zeros([og_height, og_width], dtype=type(image))
            elif len(image.shape) == 3:
                self.image_canvas[image_name] = np.zeros([image.shape[0], og_height, og_width],
                                                         dtype=image.dtype)
            else:
                raise NotImplementedError

            ## Initialize weight canvas for this image.
            self.weight_canvas[image_name] = np.zeros([og_height, og_width], dtype='float')

        # Upadate canvases and weights.
        if len(image.shape) == 2:
            self.image_canvas[image_name][h0:hE, w0:wE] += image[:dh, :dw]
            self.weight_canvas[image_name][h0:hE, w0:wE] += np.ones([dh, dw], dtype='float')
        elif len(image.shape) == 3:
            self.image_canvas[image_name][:, h0:hE, w0:wE] += image[:, :dh, :dw]
            self.weight_canvas[image_name][h0:hE, w0:wE] += np.ones([dh, dw], dtype='float')
        else:
            raise NotImplementedError

    def save_images(self):
        save_paths, image_names, image_sizes = [], [], []
        for image_name in tqdm(self.image_canvas.keys(), colour='green', desc='Saving images'):
            # Normalize canvases based on weights.
            if len(self.image_canvas[image_name].shape) == 2:
                self.image_canvas[image_name] = self.image_canvas[image_name] / (
                    self.weight_canvas[image_name] + 1e-5)
            elif len(self.image_canvas[image_name].shape) == 3:
                self.image_canvas[image_name] = self.image_canvas[image_name] / (
                    self.weight_canvas[image_name][None] + 1e-5)
            else:
                raise NotImplementedError(
                    f'Cannot save image in ImageStitcher with {len(self.image_canvas[image_name])} dimensions.'
                )

            ## Make sure there are no NaN values kept during normalization stage.
            self.image_canvas[image_name] = np.nan_to_num(self.image_canvas[image_name])

            # Save normalized image.
            save_path = os.path.join(self.save_dir, image_name + self.save_ext)

            ## Call save image method.
            self._save_image(self.image_canvas[image_name], save_path)

            ## Record image sizes, save path, and image name.
            image_sizes.append(self.image_canvas[image_name].shape)
            save_paths.append(save_path)
            image_names.append(image_name)

        return save_paths, image_names, image_sizes

    def _save_image(self, image, save_path):
        if self.save_backend == 'gdal':
            # kwimage.imwrite(save_path, image, backend='gdal')
            driver = gdal.GetDriverByName("GTiff")
            height, width = image.shape[-2], image.shape[-1]
            if len(image.shape) == 2:
                outdata = driver.Create(save_path, width, height, 1, gdal.GDT_Float32)
                outdata.GetRasterBand(1).WriteArray(image)
                outdata.FlushCache()
            elif len(image.shape) == 3:
                n_channels = image.shape[0]
                outdata = driver.Create(save_path, width, height, n_channels, gdal.GDT_Float32)
                for i in range(n_channels):
                    outdata.GetRasterBand(i + 1).WriteArray(image[i])
                    outdata.FlushCache()
            else:
                raise NotImplementedError
            del outdata
            driver = None

        elif self.save_backend == 'tifffile':
            tifffile.imwrite(save_path, image)
        elif self.save_backend == 'kwimage':
            kwimage.imwrite(save_path, image)
        else:
            raise NotImplementedError


class ImageStitcher_v2:

    def __init__(self, save_dir, image_type_name='', save_backend='gdal', save_ext=None):
        """Class to combine crops for a single type of image.
        Args:
            save_dir (str): TODO: _description_
        """
        self.save_dir = save_dir
        self.save_backend = save_backend
        self.image_type_name = image_type_name

        if save_ext is None:
            if save_backend == 'PIL':
                self.save_ext = '.png'
            elif save_backend in ['gdal', 'tiffile']:
                self.save_ext = '.tif'
            else:
                raise NotImplementedError(f'{save_backend}')
        else:
            self.save_ext = save_ext

        self._images_combined = False

        # Create save directory if it does not already exist.
        os.makedirs(save_dir, exist_ok=True)

        # Initialize canvas
        self.image_canvas = {}
        self.weight_canvas = {}

    def add_images(self, images, image_names, crop_info, og_heights, og_widths, image_weights=None):
        # Populate image weights if it is None.
        if image_weights is None:
            image_weights = [None] * len(images)

        for img, name, crop, og_height, og_width, img_weight in zip(images, image_names, crop_info,
                                                                    og_heights, og_widths,
                                                                    image_weights):
            self.add_image(img, name, crop, og_height, og_width, image_weight=img_weight)

    def add_image(self, image, image_name, crop_info, og_height, og_width, image_weight=None):
        # Get crop info.
        h0, w0, dh, dw = crop_info
        hE, wE = h0 + dh, w0 + dw
        # h0 = crop_info.h0
        # w0 = crop_info.w0
        # hE = crop_info.hE
        # wE = crop_info.wE
        # dh = hE - h0
        # dw = wE - w0

        n_dims = len(image.shape)

        # Check if canvas image has already been generated.
        try:
            self.image_canvas[image_name]
            self.weight_canvas[image_name]
        except KeyError:
            # Create canvases for this image.

            ## Initialize canvas for this image.
            if n_dims == 2:
                self.image_canvas[image_name] = np.zeros([og_height, og_width], dtype=type(image))
            elif n_dims == 3:
                self.image_canvas[image_name] = np.zeros([image.shape[0], og_height, og_width],
                                                         dtype=image.dtype)
            else:
                raise NotImplementedError

            ## Initialize weight canvas for this image.
            self.weight_canvas[image_name] = np.zeros([og_height, og_width], dtype='float')

        # Upadate canvases and weights.
        if n_dims == 2:
            self.image_canvas[image_name][h0:hE, w0:wE] += image[:dh, :dw]
        elif n_dims == 3:
            self.image_canvas[image_name][:, h0:hE, w0:wE] += image[:, :dh, :dw]
        self.weight_canvas[image_name][h0:hE, w0:wE] += np.ones([dh, dw], dtype='float')

    def _combine_images(self):
        if self._images_combined:
            pass
        else:
            for image_name in tqdm(self.image_canvas.keys(),
                                   colour='green',
                                   desc='Combining images'):
                # Normalize canvases based on weights.
                if len(self.image_canvas[image_name].shape) == 2:
                    self.image_canvas[image_name] = self.image_canvas[image_name] / (
                        self.weight_canvas[image_name] + 1e-5)
                elif len(self.image_canvas[image_name].shape) == 3:
                    self.image_canvas[image_name] = self.image_canvas[image_name] / (
                        self.weight_canvas[image_name][None] + 1e-5)
                else:
                    raise NotImplementedError(
                        f'Cannot save image in ImageStitcher with {len(self.image_canvas[image_name])} dimensions.'
                    )

                ## Make sure there are no NaN values kept during normalization stage.
                self.image_canvas[image_name] = np.nan_to_num(self.image_canvas[image_name])
            self._images_combined = True

    def save_images(self):
        save_paths, image_names, image_sizes = [], [], []
        self._combine_images()
        for image_name in tqdm(self.image_canvas.keys(), colour='green', desc='Saving images'):
            # Save normalized image.
            img_save_dir = os.path.join(self.save_dir, image_name)
            os.makedirs(img_save_dir, exist_ok=True)
            save_path = os.path.join(img_save_dir, self.image_type_name + self.save_ext)

            ## Call save image method.
            self._save_image(self.image_canvas[image_name], save_path)

            ## Record image sizes, save path, and image name.
            image_sizes.append(self.image_canvas[image_name].shape)
            save_paths.append(save_path)
            image_names.append(image_name)

        return save_paths, image_names, image_sizes

    def _save_image(self, image, save_path):
        if self.save_backend == 'tifffile':
            tifffile.imwrite(save_path, image)
        elif self.save_backend == 'PIL':
            # Check the input type of image.
            if isinstance(image, int) is False:
                if image.max() < 1:
                    image = image * 255
                image = image.astype('uint8')
            try:
                Image.fromarray(image).save(save_path)
            except:
                breakpoint()
                pass
        elif self.save_backend == 'gdal':
            # kwimage.imwrite(save_path, image, backend='gdal')
            driver = gdal.GetDriverByName("GTiff")
            height, width = image.shape[-2], image.shape[-1]
            if len(image.shape) == 2:
                outdata = driver.Create(save_path, width, height, 1, gdal.GDT_Float32)
                outdata.GetRasterBand(1).WriteArray(image)
                outdata.FlushCache()
                del outdata
            elif len(image.shape) == 3:
                n_channels = image.shape[0]
                outdata = driver.Create(save_path, width, height, n_channels, gdal.GDT_Float32)
                for i in range(n_channels):
                    outdata.GetRasterBand(i + 1).WriteArray(image[i])
                    outdata.FlushCache()
            else:
                raise NotImplementedError
            del outdata
            driver = None

        else:
            raise NotImplementedError

    def get_combined_images(self):
        self._combine_images()
        return self.image_canvas


def save_geotiff(image, save_path, save_backend='gdal', dtype=gdal.GDT_Float32):
    if save_backend == 'gdal':
        # kwimage.imwrite(save_path, image, backend='gdal')
        driver = gdal.GetDriverByName("GTiff")
        height, width = image.shape[-2], image.shape[-1]
        if len(image.shape) == 2:
            outdata = driver.Create(save_path, width, height, 1, gdal.GDT_Float32)
            outdata.GetRasterBand(1).WriteArray(image)
            outdata.FlushCache()
        elif len(image.shape) == 3:
            n_channels = image.shape[0]
            outdata = driver.Create(save_path, width, height, n_channels, gdal.GDT_Float32)
            for i in range(n_channels):
                outdata.GetRasterBand(i + 1).WriteArray(image[i])
                outdata.FlushCache()
        else:
            raise NotImplementedError
        del outdata
        driver = None
    elif save_backend == 'tifffile':
        tifffile.imwrite(save_path, image)
    elif save_backend == 'kwimage':
        kwimage.imwrite(save_path, image)
    else:
        raise NotImplementedError


def add_buffer_to_image(image,
                        desired_height,
                        desired_width,
                        buffer_mode='constant',
                        constant_value=0):
    """Extend image to desired size if smaller in resolution.
    Do nothing if the image is larger than the desired size.
    Args:
        image (np.array): A numpy array of shape [height, width] or [channels, height, width].
        desired_height (int): Desired height the image should be.
        desired_width (int): Desired width the image should be.
        buffer_mode (str, optional): Method mode for determining how to fill buffered regions. Defaults to 'constant'.
        constant_value (int, optional): For constant method, what value to assign to default canvas value. Defaults to 0.
    Raises:
        NotImplementedError: No method to handle images with number of dimensions other than 2 or 3. 
        NotImplementedError: No method to handle images with number of dimensions other than 2 or 3. 
    Returns:
        np.array: A numpy array of shape [desired_height, desired_width].
    """
    # Get image dimensions.
    n_dims = len(image.shape)
    if n_dims == 2:
        image_height, image_width = image.shape
    elif n_dims == 3:
        n_channels, image_height, image_width = image.shape
    else:
        raise NotImplementedError(f'Cannot add buffer to image with "{n_dims}" dimensions.')

    # Check if image is smaller than desired resolution.
    if (image_height >= desired_height) and (image_width >= desired_width):
        return image, np.ones([image_height, image_width], dtype=int)
    else:
        if buffer_mode == 'constant':
            # Create buffer canvas.
            buffer_mask = np.zeros([desired_height, desired_width], dtype=int)
            if n_dims == 2:
                buffer_canvas = np.ones([desired_height, desired_width],
                                        dtype=image.dtype) * constant_value
                buffer_canvas[:image_height, :image_width] = image

            elif n_dims == 3:
                buffer_canvas = np.ones([n_channels, desired_height, desired_width],
                                        dtype=image.dtype) * constant_value
                buffer_canvas[:, :image_height, :image_width] = image

                buffer_mask = np.zeros([desired_height, desired_width], dtype=int)
            buffer_mask[:image_height, :image_width] = 1
            image = buffer_canvas
        else:
            raise NotImplementedError(f'No method to handle buffer mode of "{buffer_mode}"')

    return image, buffer_mask
