import os
import pickle
import hashlib
from datetime import datetime

import cv2
import kwcoco
import tifffile
import numpy as np
from tqdm import tqdm
from einops import rearrange

import torch
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF

from matseg.utils.utils_image import add_buffer_to_image
from matseg.utils.utils_dataset import get_labelbox_material_labels
from matseg.utils.utils_misc import get_crop_slices, get_repo_paths, generate_image_slice_object


class Material_Multi_Sensor_Dataset(Dataset):

    def __init__(self,
                 mat_labels,
                 crop_size,
                 split,
                 channels,
                 rnd_seed=0,
                 sensors=['S2', 'L8'],
                 sample_positives=True,
                 ndvi_input=False,
                 refresh_labels=False,
                 resize_factor=1,
                 dset_version='drop4',
                 temporal_window=5,
                 image_transforms=False):
        """Constructor.

        Args:
            mat_labels (_type_): _description_
            crop_size (_type_): _description_
            split (_type_): _description_
            channels (_type_): _description_
            rnd_seed (int, optional): _description_. Defaults to 0.
            sensors (list, optional): _description_. Defaults to ['S2', 'L8'].
            sample_positives (bool, optional): _description_. Defaults to True.
            ndvi_input (bool, optional): _description_. Defaults to False.
            refresh_labels (bool, optional): _description_. Defaults to False.
            resize_factor (int, optional): _description_. Defaults to 1.
            dset_version (str, optional): _description_. Defaults to 'drop4'.
            temporal_window (int, optional): Number of days +- to attach material annotations to. Defaults to 5.

        Raises:
            ValueError: _description_
        """
        # Initialize private variables.
        self.split = split
        self.sensors = sensors
        self.channels = channels
        self.rnd_seed = rnd_seed
        self.crop_size = crop_size
        self.ndvi_input = ndvi_input
        self.dset_version = dset_version
        self.resize_factor = resize_factor
        self.refresh_labels = refresh_labels
        self.temporal_window = temporal_window
        self.image_transforms = image_transforms
        self.sample_positives = sample_positives

        # Set random seed.
        if self.rnd_seed is not None:
            self._set_random_seed()

        # Compute the number of channels.
        self.n_channels = self._get_n_channels()

        # Get channel code.
        if self.channels == 'RGB_NIR':
            self.channel_code = 'red|green|blue|nir'
        elif self.channels == 'RGB':
            self.channel_code = 'red|green|blue'
        else:
            raise NotImplementedError

        # Load kwcoco file.
        self.coco_dset = self._load_kwcoco_file(self.dset_version)

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
        elif self.channels == 'RGB':
            n_channels = 3
        else:
            raise NotImplementedError

        return n_channels

    def _load_kwcoco_file(self, dset_name):
        if dset_name.lower() == 'drop4':
            print('WARNING: Drop 4 dataset path is hardcoded!')
            base_dset_path = '/data4/datasets/dvc-repos/smart_data_dvc/Drop4-BAS'
            if self.split == 'train':
                dset_name = 'data_train.kwcoco.json'
            elif self.split in ['valid', 'test']:
                dset_name = 'data_vali.kwcoco.json'
            elif self.split == 'all':
                dset_name = 'data.kwcoco.json'
            else:
                raise NotImplementedError
            dset_path = os.path.join(base_dset_path, dset_name)
            print(f'Loading KWCOCO file: {dset_path}')
            coco_dset = kwcoco.CocoDataset(dset_path)
        else:
            raise NotImplementedError

        return coco_dset

    def _create_cache_file_name(self):
        input_str = f'mat_multi_sensor_{self.split}_{self.rnd_seed}_{self.crop_size.height}_{self.crop_size.stride}_{self.channels}_{self.sensors}_{self.sample_positives}_{self.temporal_window}'
        sha = hashlib.sha256()
        sha.update(input_str.encode())
        output_str = sha.hexdigest()
        return output_str + '.p'

    def _cache_examples(self, save_path, mat_labels):
        # Get image ids for kwcoco file and map them to material labels.
        ## Create mapping for region_names to video id.
        video_ids = [vid for vid in self.coco_dset.videos()]
        region_name_to_vid = {}
        for vid in video_ids:
            region_name = self.coco_dset.index.videos[vid]['name']
            region_name_to_vid[region_name] = vid

        ## Find image id associated with particular label and valid image ids within temporal window.
        region_id_mapping = {}
        region_id_tracking = {}
        for mat_label in mat_labels:

            region_name = mat_label['region_name']
            vid = region_name_to_vid[region_name]

            if region_name not in region_id_mapping.keys():
                region_id_mapping[region_name] = {}
                region_id_tracking[region_name] = {'base': 0, 'extra': 0}

            image_ids = self.coco_dset.index.vidid_to_gids[vid]

            # Get only S2 and L8 images.
            image_ids_list = []
            for img_id in image_ids:
                sensor_id = self.coco_dset.index.imgs[img_id]['sensor_coarse']
                if sensor_id in ['S2', 'L8']:
                    image_ids_list.append(img_id)

            # Find img_id from label image_dir.
            for img_id in image_ids:
                sensor_id = self.coco_dset.index.imgs[img_id]['sensor_coarse']
                if sensor_id == 'S2':
                    # Get image directory.
                    img_name = self.coco_dset.index.imgs[img_id]['name']
                    img_dir = os.path.join(self.coco_dset.data_root, region_name, sensor_id,
                                           'affine_warp', img_name)

                    # Make sure the directory exists.
                    if os.path.exists(img_dir) is False:
                        breakpoint()
                        pass

                    if img_dir == mat_label['image_dir']:
                        # Create mapping.
                        region_id_mapping[region_name][img_id] = [mat_label, 'og']
                        region_id_tracking[region_name]['base'] += 1

                        # Get datetime for image id.
                        dt_str = img_name.split('_')[1]
                        year = int(dt_str[:4])
                        month = int(dt_str[4:6])
                        day = int(dt_str[6:8])
                        img_id_dt = datetime(year, month, day)

                        # Find image ids within temporal range.
                        img_id_index = image_ids_list.index(img_id)
                        min_img_id_index = np.clip(img_id_index - 10, 0, None)
                        max_img_id_index = np.clip(img_id_index + 10, 0, len(image_ids_list))

                        for i in range(min_img_id_index, max_img_id_index):
                            temp_img_id = image_ids_list[i - 1]

                            # Check if image id is already in the data structure.
                            if temp_img_id in region_id_mapping[region_name].keys():
                                continue

                            # Get datetime.
                            temp_img_name = self.coco_dset.index.imgs[temp_img_id]['name']
                            dt_str = temp_img_name.split('_')[1]
                            year = int(dt_str[:4])
                            month = int(dt_str[4:6])
                            day = int(dt_str[6:8])
                            temp_img_id_dt = datetime(year, month, day)

                            # Check if the temp dt is within the temporal span.
                            if abs((img_id_dt - temp_img_id_dt).days) < self.temporal_window:
                                region_id_mapping[region_name][temp_img_id] = [
                                    mat_label, 'transfered'
                                ]
                                region_id_tracking[region_name]['extra'] += 1

        for region_name, stats in region_id_tracking.items():
            print(f"{region_name} | Base: {stats['base']} | Extra: {stats['extra']}")

        # Create examples for each image id that has a label.
        examples = []
        for region_name, region_data in tqdm(region_id_mapping.items(), desc='Regions'):
            for img_id, (mat_label, label_status) in tqdm(region_data.items(),
                                                          desc='Images',
                                                          leave=False):
                # Get image sensor type.
                sensor_id = self.coco_dset.index.imgs[img_id]['sensor_coarse']

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

                    if label_status == 'og':
                        # Don't need to load quality mask here.
                        q_mask = None
                        bin_q_mask = None
                    elif label_status == 'transfered':
                        # Get quality mask associated with this image.
                        q_mask = self.coco_dset.delayed_load(img_id,
                                                             channels='cloudmask').finalize()

                        if len(q_mask.shape) == 3:
                            q_mask = q_mask[..., 0]
                        bin_q_mask = np.zeros(q_mask.shape)

                        x, y = np.where((q_mask == 1) | (q_mask == 33))
                        bin_q_mask[x, y] = 1

                    # Check that this slice contains material masks.
                    h0, w0, h, w = crop_slice
                    mat_mask_crop = mat_label['mat_mask'][h0:h0 + h, w0:w0 + w]

                    if (mat_mask_crop.shape[0] != self.crop_size.height) or (
                            mat_mask_crop.shape[1] != self.crop_size.width):
                        buffer_image = np.zeros([self.crop_size.height, self.crop_size.width],
                                                dtype=mat_mask_crop.dtype)
                        buffer_image[:mat_mask_crop.shape[0], :mat_mask_crop.
                                     shape[1]] = mat_mask_crop
                        mat_mask_crop = buffer_image

                        # If a transferred label region then filter the material labels by the q_mask.
                        if bin_q_mask is None:
                            # An originally labeled region so do not filter by quality mask.
                            pass
                        else:
                            # Crop and buffer quality mask to match the above crop.
                            bin_q_mask_crop = bin_q_mask[h0:h0 + h, w0:w0 + w]
                            buffered_bin_q_mask_crop = np.zeros(
                                [self.crop_size.height, self.crop_size.width],
                                dtype=mat_mask_crop.dtype)
                            buffered_bin_q_mask_crop[:bin_q_mask_crop.shape[0], :bin_q_mask_crop.
                                                     shape[1]] = bin_q_mask_crop

                            # Filter the material labels by the quality mask.
                            mat_mask_crop = mat_mask_crop * buffered_bin_q_mask_crop

                    mat_mask_crop = mat_mask_crop.astype('int64')
                    if self.sample_positives:
                        if mat_mask_crop.sum() > 0:
                            example['crop_slice'] = crop_slice
                            example['mat_crop'] = mat_mask_crop
                            example['image_dir'] = mat_label['image_dir']
                            example['img_id'] = img_id
                            example['region_name'] = region_name
                            example['sensor'] = sensor_id
                            example['og_size'] = [H, W]
                            examples.append(example)
                    else:
                        example['crop_slice'] = crop_slice
                        example['mat_crop'] = mat_mask_crop
                        example['image_dir'] = mat_label['image_dir']
                        example['region_name'] = region_name
                        example['img_id'] = img_id
                        example['sensor'] = sensor_id
                        example['og_size'] = [H, W]
                        examples.append(example)

        pickle.dump(examples, open(save_path, 'wb'))
        return examples

    def apply_transforms(self, image, target, aux_items=[], color_transforms=False):
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
        if color_transforms:
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
        img_id = example['img_id']
        h0, w0, h, w = example['crop_slice']

        # Get material mask.
        mat_mask = example['mat_crop']

        # Load and crop image.
        delayed_crop = self.coco_dset.delayed_load(img_id, channels=self.channel_code)
        if example['sensor'] == 'L8':
            delayed_crop = delayed_crop.scale(3)
        delayed_crop = delayed_crop.crop((slice(h0, h0 + h), slice(w0, w0 + w)))
        crop = delayed_crop.finalize(as_xarray=False).transpose((2, 0, 1))
        # [n_channels, height, width]

        # Convert from int16 to float.
        crop = np.clip(crop / 2**16, 0, 1)

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

        # Apply transform.
        if self.image_transforms:
            image, mat_mask, _ = self.apply_transforms(image, mat_mask)

        output = {}
        output['image'] = torch.tensor(image.astype('float32'))
        output['target'] = torch.tensor(mat_mask).long()
        output['img_id'] = img_id
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


if __name__ == '__main__':
    # Test 1: Make sure a class object can be generated.
    crop_height, crop_width, crop_stride = 160, 160, 40
    mat_labels, _ = get_labelbox_material_labels(refresh_labels=False)
    slice_params = generate_image_slice_object(crop_height, crop_width, crop_stride)
    temporal_window = 5
    split = 'all'
    channels = 'RGB_NIR'
    image_transforms = False
    dset = Material_Multi_Sensor_Dataset(mat_labels,
                                         slice_params,
                                         split,
                                         channels,
                                         temporal_window,
                                         image_transforms=image_transforms)
    ex = dset.__getitem__(0)
    print('Test #1: Passed')
