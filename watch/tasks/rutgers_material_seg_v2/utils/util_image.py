import os

import kwimage
import numpy as np
from tqdm import tqdm
from osgeo import gdal
from tifffile import tifffile

from watch.tasks.rutgers_material_seg_v2.utils.util_misc import sigmoid


def load_norm_image(kwcoco_dset, image_id, crop_slice=None, channels=None, image_scale_mode='direct'):
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
            raise NotImplementedError(f'method not implemented for scaling "{image.dtype}" type to [0,1] range.')

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

        for img, name, crop, og_height, og_width, img_weight in zip(images, image_names, crop_info, og_heights,
                                                                    og_widths, image_weights):
            self.add_image(img, name, crop, og_height, og_width, image_weight=img_weight)

    def add_image(self, image, image_name, crop_info, og_height, og_width, image_weight=None):
        # Get crop info.
        h0, w0, dh, dw = crop_info
        h0, w0, dh, dw = h0.item(), w0.item(), dh.item(), dw.item()
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
                self.image_canvas[image_name] = np.zeros([image.shape[0], og_height, og_width], dtype=image.dtype)
            else:
                raise NotImplementedError

            ## Initialize weight canvas for this image.
            self.weight_canvas[image_name] = np.zeros([og_height, og_width], dtype='float')

        # Upadate canvases and weights.
        self.image_canvas[image_name][:, h0:hE, w0:wE] += image[:, :dh, :dw]
        self.weight_canvas[image_name][h0:hE, w0:wE] += np.ones([dh, dw], dtype='float')

    def save_images(self):
        save_paths, image_names, image_sizes = [], [], []
        for image_name in tqdm(self.image_canvas.keys(), colour='green', desc='Saving images'):
            # Normalize canvases based on weights.
            if len(self.image_canvas[image_name].shape) == 2:
                self.image_canvas[image_name] = self.image_canvas[image_name] / (self.weight_canvas[image_name] + 1e-5)
            elif len(self.image_canvas[image_name].shape) == 3:
                self.image_canvas[image_name] = self.image_canvas[image_name] / (self.weight_canvas[image_name][None] +
                                                                                 1e-5)
            else:
                raise NotImplementedError(
                    f'Cannot save image in ImageStitcher with {len(self.image_canvas[image_name])} dimensions.')

            ## Make sure there are no NaN values kept during normalization stage.
            self.image_canvas[image_name] = np.nan_to_num(self.image_canvas[image_name])

            # Save normalized image.
            save_path = os.path.join(self.save_dir, image_name + self.save_ext)

            ## Call save image method.
            try:
                self._save_image(self.image_canvas[image_name], save_path)
            except:
                breakpoint()
                pass

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

        elif self.save_backend == 'tifffile':
            tifffile.imwrite(save_path, image)
        elif self.save_backend == 'kwimage':
            kwimage.imwrite(save_path, image)
        else:
            raise NotImplementedError


def save_geotiff(image, save_path, save_backend='gdal', dtype=gdal.GDT_Float32):
    if save_backend == 'gdal':
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
    elif save_backend == 'tifffile':
        tifffile.imwrite(save_path, image)
    elif save_backend == 'kwimage':
        kwimage.imwrite(save_path, image)
    else:
        raise NotImplementedError