import numpy as np
from torch.utils.data import Dataset

from watch.tasks.rutgers_material_seg_v2.utils.util_misc import sigmoid


class BaseDataset(Dataset):
    def __init__(self, kwcoco_path, split, image_scale_mode='direct', seed_num=0, sensor='S2'):
        super().__init__()

        self.split = split
        self.sensor = sensor
        self.seed_num = seed_num
        self.kwcoco_path = kwcoco_path
        self.image_scale_mode = image_scale_mode

        # Get number of channels of input data.
        self.n_channels = self.get_n_channels()

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, index):
        raise NotImplementedError

    def get_n_channels(self):
        raise NotImplementedError

    def _scale_image(self, image, scale_mode='direct'):
        """Scales the pixel values of image.

        Args:
            image (np.array): A numpy array of shape [channels, height, width].
            scale_mode (str, optional): Parameter to determine how to scale image. Defaults to 'direct'.
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
            image = __direct_scale(image)
            scaled_image = image / (np.linalg.norm(image, axis=0, ord=1) + 1e-4)

        else:
            raise NotImplementedError(f'Scale mode "{scale_mode}" not implemented.')

        return scaled_image

    def _load_norm_image(self, image_id, crop_slice=None, channels=None):
        """Load an image from kwcoco file and normalize to float.

        Args:
            image_id (int): Image ID (aka gid) to use as reference in kwcoco dataset.
            crop_slice (list, optional): Crop image to this rectangle, [h0, w0, hE, wE]. Defaults to None.
            channels (str, optional): Language specifying which channels to load from an image. Defaults to None.

        Returns:
            np.array: A float array of shape [channels, height, width].
        """
        if channels is None:
            channels = self.channels

        img_delayed = self.coco_dset.delayed_load(image_id, channels=channels)

        # Crop image (optional).
        if crop_slice is not None:
            h0, w0, dh, dw = crop_slice
            hE, wE = h0 + dh, w0 + dw
            img_delayed = img_delayed.crop((slice(h0, hE), slice(w0, wE)))

        # Finalize image loading.
        image = img_delayed.finalize()  # [height, width, channels]

        image = image.transpose(2, 0, 1)  # [channels, height, width]

        # Normalize image to [0, 1] range.
        image = self._scale_image(image, scale_mode=self.image_scale_mode)

        return image

    def _compute_dataset_stats(self, pixel_data):
        # Less sensitive to outliers
        mean = np.median(pixel_data, axis=-1)
        # mean = pixel_data.mean(axis=-1)
        std = pixel_data.std(axis=-1)

        return mean, std

    def _add_buffer_to_image(self, image, desired_height, desired_width, buffer_mode='constant', constant_value=0):
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
                    buffer_canvas = np.ones([desired_height, desired_width], dtype=image.dtype) * constant_value
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

    def _label_mask_manipulation(self, label_mask, label_polygon_raster=None):
        # TODO: Label smoothing
        ## Convert 1-hot encoding to label-distribution.

        # Label condensing (fuse material labels such as concrete and asphalt)

        # Attentuate label confidence (weight) by distnace to edge of polygon.
        ## Need to figure out algorithm for this.
        pass


if __name__ == '__main__':
    pass
