import os
import pickle
import hashlib

import cv2
import tifffile
import numpy as np
from tqdm import tqdm
from einops import rearrange

import torch
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF

from matseg.utils.utils_misc import get_crop_slices, get_repo_paths
from matseg.utils.utils_image import load_S2_image, add_buffer_to_image


class Material_LB_Dataset(Dataset):

    def __init__(self,
                 mat_labels,
                 crop_size,
                 split,
                 channels,
                 rnd_seed=0,
                 sensors='S2',
                 sample_positives=True,
                 ndvi_input=False,
                 refresh_labels=False,
                 resize_factor=1):
        # Initialize private variables.
        self.split = split
        self.sensors = sensors
        self.channels = channels
        self.rnd_seed = rnd_seed
        self.crop_size = crop_size
        self.ndvi_input = ndvi_input
        self.resize_factor = resize_factor
        self.refresh_labels = refresh_labels
        self.sample_positives = sample_positives

        # Set random seed.
        if self.rnd_seed is not None:
            self._set_random_seed()

        # Compute the number of channels.
        self.n_channels = self._get_n_channels()

        # Split material labels into train or valid split.
        train_split_pct = 0.9
        np.random.shuffle(mat_labels)
        n_train_values = int(train_split_pct * len(mat_labels))
        if split == 'train':
            sub_mat_labels = mat_labels[:n_train_values]
        elif split == 'valid':
            sub_mat_labels = mat_labels[n_train_values:]
        elif split == 'all':
            sub_mat_labels = mat_labels
        else:
            raise ValueError(f'Split "{self.split}" not handled.')

        cache_dir = get_repo_paths('material_dataset_cache')
        cache_save_path = os.path.join(cache_dir, self._create_cache_file_name())
        if os.path.exists(cache_save_path) is False or (self.refresh_labels):
            self.examples = self._cache_examples(cache_save_path, sub_mat_labels)
            print(f'Created: {cache_save_path}')
        else:
            self.examples = pickle.load(open(cache_save_path, 'rb'))
            print(f'Loaded: {cache_save_path}')

        print(self.split, len(self.examples))

    def _set_random_seed(self):
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
        torch.manual_seed(self.rnd_seed)
        torch.cuda.manual_seed_all(self.rnd_seed)
        np.random.seed(self.rnd_seed)

    def _get_n_channels(self):
        if self.channels == 'RGB_NIR':
            n_channels = 4
        elif self.channels == 'RGB_NIR':
            n_channels = 3
        else:
            raise NotImplementedError

        return n_channels

    def _create_cache_file_name(self):
        input_str = f'{self.split}_{self.rnd_seed}_{self.crop_size.height}_{self.crop_size.stride}_{self.channels}_{self.sensors}_{self.sample_positives}'
        sha = hashlib.sha256()
        sha.update(input_str.encode())
        output_str = sha.hexdigest()

        return output_str + '.p'

    def _cache_examples(self, save_path, mat_labels):
        examples = []
        for mat_label in tqdm(mat_labels, desc='Filtering dataset examples'):
            # Get size of image.
            H, W = mat_label['mat_mask'].shape
            crop_slices = get_crop_slices(H,
                                          W,
                                          self.crop_size.height,
                                          self.crop_size.width,
                                          step=self.crop_size.stride,
                                          mode='exact')
            for crop_slice in crop_slices:
                example = {}

                region_name = mat_label['region_name']

                # Check that this slice contains material masks.
                h0, w0, h, w = crop_slice
                mat_mask_crop = mat_label['mat_mask'][h0:h0 + h, w0:w0 + w]

                if (mat_mask_crop.shape[0] != self.crop_size.height) or (mat_mask_crop.shape[1] !=
                                                                         self.crop_size.width):
                    buffer_image = np.zeros([self.crop_size.height, self.crop_size.width],
                                            dtype=mat_mask_crop.dtype)
                    buffer_image[:mat_mask_crop.shape[0], :mat_mask_crop.shape[1]] = mat_mask_crop
                    mat_mask_crop = buffer_image

                mat_mask_crop = mat_mask_crop.astype('int64')
                if self.sample_positives:
                    if mat_mask_crop.sum() > 0:
                        example['crop_slice'] = crop_slice
                        example['mat_crop'] = mat_mask_crop
                        example['image_dir'] = mat_label['image_dir']
                        example['region_name'] = region_name
                        example['og_size'] = [H, W]
                        examples.append(example)
                else:
                    example['crop_slice'] = crop_slice
                    example['mat_crop'] = mat_mask_crop
                    example['image_dir'] = mat_label['image_dir']
                    example['region_name'] = region_name
                    example['og_size'] = [H, W]
                    examples.append(example)

        pickle.dump(examples, open(save_path, 'wb'))
        return examples

    def apply_transforms(self, image, target, aux_items=[]):
        # Vertical and horizonal flips.
        if np.random.rand() > 0.5:
            image = TF.hflip(image)
            aux_items = [TF.hflip(aux_item) for aux_item in aux_items]
            target = TF.hflip(target)
        if np.random.rand() > 0.5:
            image = TF.vflip(image)
            target = TF.vflip(target)
            aux_items = [TF.vflip(aux_item) for aux_item in aux_items]

        # Color adjustments.
        ## Gamma
        if np.random.rand() > 0.5:
            random_gamma = np.random.uniform(0.9, 1.1)
            image = image**random_gamma
        ## Contrast
        if np.random.rand() > 0.5:
            random_contrast_factor = np.random.uniform(0.75, 1.25)
            image = torch.clamp(image * random_contrast_factor, 0, 1)

        return image, target, aux_items

    def __len__(self):
        return len(self.examples)

    def _load_mat_mask(self, mat_path, crop_slice=None):
        mat_mask = tifffile.imread(mat_path)

        if crop_slice is not None:
            h0, w0, h, w = crop_slice
            mat_mask = mat_mask[h0:h0 + h, w0:w0 + w]

            if (mat_mask.shape[0] != self.crop_size.height) or (mat_mask.shape[1] !=
                                                                self.crop_size.width):
                buffer_image = np.zeros([self.crop_size.height, self.crop_size.width],
                                        dtype=mat_mask.dtype)
                buffer_image[:mat_mask.shape[0], :mat_mask.shape[1]] = mat_mask
                mat_mask = buffer_image

        mat_mask = mat_mask.astype('int64')

        return mat_mask

    def _compute_ndvi(self, image, eps=1e-5):
        # NDVI = (NIR-RED) / (NIR+RED)
        nir_band = image[3]
        red_band = image[0]
        ndvi = (nir_band - red_band) / (nir_band + red_band + eps)

        # Scale between [0, 1]
        ndvi = (np.clip(ndvi, -1, 1) + 1) / 2
        return ndvi

    def __getitem__(self, index, get_metadata=False):
        example = self.examples[index]

        # Get material mask.
        mat_mask = example['mat_crop']

        # Load and crop image.
        # [C, H, W]
        crop = load_S2_image(example['image_dir'], self.channels, crop_slice=example['crop_slice'])

        # Convert from int16 to float.
        crop = crop / 2**12

        # Potentially resize image.
        if self.resize_factor != 1:
            crop = rearrange(crop, 'c h w -> h w c')
            crop = cv2.resize(crop,
                              dsize=(crop.shape[0] * self.resize_factor,
                                     crop.shape[1] * self.resize_factor),
                              interpolation=cv2.INTER_LINEAR)
            crop = rearrange(crop, 'h w c -> c h w')

            # Resize material mask as well.
            mat_mask = cv2.resize(mat_mask,
                                  dsize=(mat_mask.shape[0] * self.resize_factor,
                                         mat_mask.shape[1] * self.resize_factor),
                                  interpolation=cv2.INTER_NEAREST)

        # Add padding to the image.
        image, buffer_mask = add_buffer_to_image(crop,
                                                 self.crop_size.height * self.resize_factor,
                                                 self.crop_size.width * self.resize_factor,
                                                 constant_value=0)

        output = {}
        output['image'] = torch.tensor(image.astype('float32'))
        output['target'] = torch.tensor(mat_mask).long()
        output['crop_slice'] = example['crop_slice']
        output['image_dir'] = example['image_dir']
        output['og_size'] = example['og_size']
        output['region_name'] = example['region_name']
        output['buffer_mask'] = buffer_mask

        return output

    def to_RGB(self, image):
        red_band = image[0]
        green_band = image[1]
        blue_band = image[2]

        red_band = red_band**(0.6)
        green_band = green_band**(0.6)
        blue_band = blue_band**(0.6)

        rgb_image = np.stack([red_band, green_band, blue_band])
        # rgb_image = rgb_image.transpose((1, 2, 0))
        rgb_image *= 2**2

        rgb_image = (np.clip(rgb_image, 0, 1) * 255).astype('uint8')

        return rgb_image
