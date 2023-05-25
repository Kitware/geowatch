import os
import pickle

import cv2
import torch
import kwcoco
import numpy as np
from einops import rearrange
from torch.utils.data import Dataset

from matseg.utils.utils_image import add_buffer_to_image, load_S2_image
from matseg.utils.utils_misc import generate_image_slice_object, get_repo_paths, get_crop_slices


class BAS_Dataset(Dataset):

    def __init__(self,
                 channels,
                 slice_params,
                 resize_factor=1,
                 first_last_samples=5,
                 select_regions=[],
                 dset_name='drop4',
                 sensors=['S2'],
                 quality_masks=True,
                 split='all'):
        # Initialize private variables.
        self.split = split
        self.sensors = sensors
        self.channels = channels
        self.dset_name = dset_name
        self.slice_params = slice_params
        self.quality_masks = quality_masks
        self.resize_factor = resize_factor
        self.select_regions = select_regions
        self.first_last_samples = first_last_samples
        self.n_channels = self._get_n_channels(channels)
        self.channel_code = self._get_channel_str(channels)

        # Load/compute BAS masks for each region.
        region_bas_annos_paths = get_repo_paths('bas_annos')
        self.bas_annos = pickle.load(open(region_bas_annos_paths, 'rb'))

        # Load kwcoco file.
        self.coco_dset = self._load_kwcoco_file(self.dset_name)

        # Create examples to get
        self.examples = self._create_examples()

    def _get_n_channels(self, channels):
        if channels == 'RGB_NIR':
            n_channels = 4
        else:
            raise NotImplementedError
        return n_channels

    def _get_channel_str(self, channels):
        if channels == 'RGB_NIR':
            channel_str = 'red|blue|green|nir'
        else:
            raise NotImplementedError
        return channel_str

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

    def _create_examples(self):
        # Create mapping from region_names to video id.
        video_ids = [vid for vid in self.coco_dset.videos()]
        region_name_to_vid = {}
        for vid in video_ids:
            region_name = self.coco_dset.index.videos[vid]['name']
            region_name_to_vid[region_name] = vid

        # Subselect regions that should be included.
        if self.select_regions == 'all':
            vids = video_ids
        elif self.select_regions == []:
            raise ValueError('No regions selected to create examples for!')
        else:
            print(f'Creating examples for regions: {self.select_regions}')
            vids = []
            for region_name in self.select_regions:
                vids.append(region_name_to_vid[region_name])

        # For each region get the first/last X images in video sequence.
        examples = []
        for vid in vids:
            region_name = self.coco_dset.index.videos[vid]['name']
            image_ids = self.coco_dset.index.vidid_to_gids[vid]

            # Subselect based on sensor_type.
            up_image_ids = []
            for img_id in image_ids:
                sensor_id = self.coco_dset.index.imgs[img_id]['sensor_coarse']
                if sensor_id in self.sensors:
                    up_image_ids.append(img_id)

            # Only get S2 later images.
            s2_image_ids = []
            for img_id in image_ids:
                sensor_id = self.coco_dset.index.imgs[img_id]['sensor_coarse']
                if sensor_id == 'S2':
                    s2_image_ids.append(img_id)

            height = self.coco_dset.index.imgs[s2_image_ids[0]]['height']
            width = self.coco_dset.index.imgs[s2_image_ids[0]]['width']
            # if region_name not in region_sizes.keys():
            #     region_sizes[region_name] = {'height': height, 'width': width}

            # Get first and last image_ids.
            early_img_ids = up_image_ids[:self.first_last_samples]
            late_img_ids = s2_image_ids[-self.first_last_samples:]

            # Create example.
            ## Early images.
            for i, img_id in enumerate(early_img_ids):
                height = self.coco_dset.index.imgs[img_id]['height']
                width = self.coco_dset.index.imgs[img_id]['width']
                sensor_id = self.coco_dset.index.imgs[img_id]['sensor_coarse']
                img_name = self.coco_dset.index.imgs[img_id]['name']

                img_dir = os.path.join(self.coco_dset.data_root, region_name, sensor_id,
                                       'affine_warp', img_name)

                if sensor_id == 'L8':
                    height = int(height * 3)
                    width = int(width * 3)

                crop_slices = get_crop_slices(height,
                                              width,
                                              self.slice_params.height,
                                              self.slice_params.width,
                                              step=self.slice_params.stride,
                                              mode='exact')
                for crop_slice in crop_slices:
                    example = {}
                    example['crop_slice'] = crop_slice
                    example['img_id'] = img_id
                    example['sensor'] = sensor_id
                    example['region_name'] = region_name
                    example['first_last'] = 0
                    example['first_last_index'] = i
                    example['og_size'] = [height, width]
                    example['img_name'] = img_name
                    example['img_dir'] = img_dir
                    examples.append(example)

            ## Late images.
            for i, img_id in enumerate(late_img_ids):
                height = self.coco_dset.index.imgs[img_id]['height']
                width = self.coco_dset.index.imgs[img_id]['width']
                sensor_id = self.coco_dset.index.imgs[img_id]['sensor_coarse']
                img_name = self.coco_dset.index.imgs[img_id]['name']

                img_dir = os.path.join(self.coco_dset.data_root, region_name, sensor_id,
                                       'affine_warp', img_name)

                if sensor_id == 'L8':
                    height = int(height * 3)
                    width = int(width * 3)

                crop_slices = get_crop_slices(height,
                                              width,
                                              self.slice_params.height,
                                              self.slice_params.width,
                                              step=self.slice_params.stride,
                                              mode='exact')
                for crop_slice in crop_slices:
                    example = {}
                    example['crop_slice'] = crop_slice
                    example['img_id'] = img_id
                    example['sensor'] = sensor_id
                    example['region_name'] = region_name
                    example['first_last'] = 1
                    example['first_last_index'] = i
                    example['og_size'] = [height, width]
                    example['img_name'] = img_name
                    example['img_dir'] = img_dir
                    examples.append(example)

        if len(examples) == 0:
            raise ValueError(f'No examples could be generated!')

        return examples

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, index):
        example = self.examples[index]
        img_id = example['img_id']
        h0, w0, h, w = example['crop_slice']

        # Load and frame image.

        if example['sensor'] == 'L8':
            crop = load_S2_image(example['img_dir'], self.channels)
            bands = []
            for i in range(crop.shape[0]):
                band = cv2.resize(crop[i], (example['og_size'][1], example['og_size'][0]),
                                  interpolation=cv2.INTER_LINEAR)
                bands.append(band)
            crop = np.stack(bands)
            crop = crop[:, h0:h0 + h, w0:w0 + w]
        else:
            crop = load_S2_image(example['img_dir'],
                                 self.channels,
                                 crop_slice=example['crop_slice'])
        # delayed_crop = self.coco_dset.delayed_load(img_id, channels=self.channel_code)
        # if example['sensor'] == 'L8':
        #     delayed_crop = delayed_crop.scale(3)
        # delayed_crop = delayed_crop.crop((slice(h0, h0 + h), slice(w0, w0 + w)))
        # crop = delayed_crop.finalize(as_xarray=False).transpose((2, 0, 1))
        # [n_channels, height, width]

        # Get quality mask.
        if self.quality_masks:
            # delayed_crop = self.coco_dset.delayed_load(img_id, channels='cloudmask')
            # if example['sensor'] == 'L8':
            #     delayed_crop = delayed_crop.scale(3)
            # delayed_crop = delayed_crop.crop((slice(h0, h0 + h), slice(w0, w0 + w)))
            # q_mask = delayed_crop.finalize(as_xarray=False)[:, :, 0]

            if example['sensor'] == 'L8':
                q_mask = load_S2_image(example['img_dir'], 'cloudmask')
                q_mask = cv2.resize(q_mask, (example['og_size'][1], example['og_size'][0]),
                                    interpolation=cv2.INTER_NEAREST)
                q_mask = q_mask[h0:h0 + h, w0:w0 + w]
            else:
                q_mask = load_S2_image(example['img_dir'], 'cloudmask', example['crop_slice'])

            # Convert to binary quality mask.
            q_mask_bin = np.zeros(q_mask.shape)
            x, y = np.where((q_mask == 1) | (q_mask == 33))  # include non-mask and water values.
            q_mask_bin[x, y] = 1
            q_mask = q_mask_bin

        # Convert from int16 to float.
        frame = crop / 2**12

        # Potentially resize image.
        if self.resize_factor != 1:
            frame = rearrange(frame, 'c h w -> h w c')
            frame = cv2.resize(frame,
                               dsize=(frame.shape[0] * self.resize_factor,
                                      frame.shape[1] * self.resize_factor),
                               interpolation=cv2.INTER_LINEAR)
            frame = rearrange(frame, 'h w c -> c h w')

            # Resize material mask as well.
            if self.quality_masks:
                q_mask = cv2.resize(q_mask,
                                    dsize=(q_mask.shape[0] * self.resize_factor,
                                           q_mask.shape[1] * self.resize_factor),
                                    interpolation=cv2.INTER_NEAREST)

        # Add padding to the image.
        image, buffer_mask = add_buffer_to_image(frame,
                                                 self.slice_params.height * self.resize_factor,
                                                 self.slice_params.width * self.resize_factor,
                                                 constant_value=0)

        if self.quality_masks:
            q_mask, _ = add_buffer_to_image(q_mask,
                                            self.slice_params.height * self.resize_factor,
                                            self.slice_params.width * self.resize_factor,
                                            constant_value=0)

        output = {}
        output['img_id'] = img_id
        output['image'] = torch.tensor(image.astype('float32'))
        output['buffer_mask'] = buffer_mask
        output['region_name'] = example['region_name']
        output['first_last'] = example['first_last']
        output['crop_slice'] = example['crop_slice']
        output['og_size'] = example['og_size']
        output['first_last_index'] = example['first_last_index']
        output['img_name'] = example['img_name']
        if self.quality_masks:
            output['q_mask'] = q_mask

        return output


if __name__ == '__main__':

    channels = 'RGB_NIR'
    slice_params = generate_image_slice_object(300, 300, 300)
    select_regions = ['KR_R001', 'KR_R002']
    dataset = BAS_Dataset(channels, slice_params, select_regions=select_regions)
    ex = dataset.__getitem__(0)