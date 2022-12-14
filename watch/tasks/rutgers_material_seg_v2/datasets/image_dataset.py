import os
import pickle
import hashlib
import multiprocessing
from datetime import datetime

import kwcoco
import numpy as np
from einops import rearrange
from omegaconf import OmegaConf

from watch.tasks.rutgers_material_seg_v2.datasets.base_dataset import BaseDataset
from watch.tasks.rutgers_material_seg_v2.utils.util_dataset import filter_image_ids_by_sensor
from watch.tasks.rutgers_material_seg_v2.utils.util_misc import generate_video_slice_object, get_crop_slices


def process_example(image_id, image_slice, kwcoco_dset, channels):
    examples = []
    # Get datetime for image.
    image_name = kwcoco_dset.index.imgs[image_id]['name']
    year = int(image_name.split('_')[1][:4])
    month = int(image_name.split('_')[1][4:6])
    day = int(image_name.split('_')[1][6:8])
    dt = datetime(year, month, day)

    pct_load_image = 0.0
    if np.random.rand() < pct_load_image:
        # Load and scale image.
        # image = self._load_norm_image(image_id)
        img_delayed = kwcoco_dset.delayed_load(image_id, channels=channels)
        image = img_delayed.finalize()  # [height, width, channels]
        image = image.transpose(2, 0, 1)  # [channels, height, width]
        image = image / 2**16

        img_h, img_w = image.shape[-2:]

        # Randomly sample pixels for computing mean and std.
        ## TODO: Make percentage of pixels taken adjustable.
        pct = 0.001
        n_pixels = img_h * img_w
        n_sampled_pixels = int(pct * n_pixels)

        X = np.random.randint(0, img_h, size=n_sampled_pixels)
        Y = np.random.randint(0, img_w, size=n_sampled_pixels)

        sampled_pixels = image[:, X, Y]
    else:
        img_h = kwcoco_dset.index.imgs[image_id]['height']
        img_w = kwcoco_dset.index.imgs[image_id]['width']
        sampled_pixels = None

    # Get viable crop slices.
    crop_slices = get_crop_slices(img_h, img_w, crop_height=image_slice.height, crop_width=image_slice.width)

    for crop_slice in crop_slices:
        example = {}
        example['image_id'] = image_id
        example['crop_slice'] = crop_slice
        example['og_height'] = img_h
        example['og_width'] = img_w
        example['datetime'] = dt
        examples.append(example)

    return examples, sampled_pixels


class ImageDataset(BaseDataset):
    def __init__(self,
                 kwcoco_path,
                 split,
                 image_slice_cfg=None,
                 feature_type='pixel',
                 n_clusters=5,
                 region_shape='square',
                 square_size=5,
                 channels='ALL',
                 seed_num=0,
                 sensor='S2',
                 image_scale_mode='direct',
                 image_norm_mode=None,
                 overwrite_stats=None):

        # Create private variables.
        self.channels = channels
        self.n_clusters = n_clusters
        self.square_size = square_size
        self.feature_type = feature_type
        self.region_shape = region_shape
        self.image_norm_mode = image_norm_mode

        # Check image slice.
        if image_slice_cfg is None:
            raise ValueError('No Image Slice parameters given for ImageDataset.')
        else:
            self.image_slice = generate_video_slice_object(height=image_slice_cfg.height,
                                                           width=image_slice_cfg.width,
                                                           n_frames=image_slice_cfg.n_frames,
                                                           scale=image_slice_cfg.scale,
                                                           stride=image_slice_cfg.stride)

        # Call super function.
        super(ImageDataset, self).__init__(kwcoco_path,
                                           split,
                                           image_scale_mode=image_scale_mode,
                                           seed_num=seed_num,
                                           sensor=sensor)

        # Get examples.
        self._build_dataset()

        if overwrite_stats:
            self.mean = overwrite_stats['mean']
            self.std = overwrite_stats['std']

    def get_n_channels(self):
        # Update channels.
        # TODO: Fix this hack.
        if self.channels == 'ALL':
            print('WARNING: ALL channels does NOT include cloud channel or materials.')
            self.channels = 'red|green|blue|nir|swir16|swir22'

        n_channels = len(self.channels.split('|'))

        return n_channels

    def _create_cach_file_name(self):
        input_str = f'{self.split}_{self.feature_type}_{self.n_clusters}_{self.image_scale_mode}_{self.square_size}_{self.sensor}_{self.channels}'
        sha = hashlib.sha256()
        sha.update(input_str.encode())
        output_str = sha.hexdigest()

        return output_str + '.p'

    def _build_dataset(self):
        # Check to see if examples are already generated.
        self.dset_root = os.path.split(self.kwcoco_path)[0]
        cache_dir = os.path.join(self.dset_root, '_cache', 'image')
        os.makedirs(cache_dir, exist_ok=True)
        cache_file_name = self._create_cach_file_name()
        cache_path = os.path.join(cache_dir, cache_file_name)

        if os.path.exists(cache_path) is False:
            # Create examples.
            print(f'No cached examples found at: {cache_path}')

            # Load KWCOCO file.
            self.coco_dset = kwcoco.CocoDataset(self.kwcoco_path)

            # Get all image ids from sensor.

            ## Filter image IDs based on input sensor.
            image_ids = filter_image_ids_by_sensor(self.coco_dset, self.sensor)
            print(f'Number of image IDs after filtering by sensor ({self.sensor}): {len(image_ids)}')

            print(f'Multiprocessing on {multiprocessing.cpu_count()} cores: Gathering examples')
            self.examples, pixel_data = [], []
            # with multiprocessing.Pool() as pool:
            #     input_to_func = [(image_id, self.image_slice, self.coco_dset, self.channels) for image_id in image_ids]
            #     for (examples, ex_pixel_data) in pool.starmap(process_example,
            #                                                   tqdm(input_to_func, total=len(input_to_func))):
            #         self.examples.extend(examples)
            #         pixel_data.append(ex_pixel_data)
            for image_id in image_ids:
                examples, ex_pixel_data = process_example(image_id, self.image_slice, self.coco_dset, self.channels)
                self.examples.extend(examples)
                pixel_data.append(ex_pixel_data)

            # Combine pixel data.
            ## Filter Nones.
            # filtered_pixel_data = []
            # for data in pixel_data:
            #     if data is not None:
            #         filtered_pixel_data.append(data)
            # pixel_data = np.concatenate(filtered_pixel_data, axis=1)

            # Compute mean and std of this slice of dataset if split == 'train'.
            # mean, std = self._compute_dataset_stats(pixel_data)
            mean, std = None, None

            # Save examples and info needed to load from examples.
            cache_object = {'examples': self.examples, 'coco_dset': self.coco_dset, 'mean': mean, 'std': std}
            pickle.dump(cache_object, open(cache_path, 'wb'))
            print(f'Saved cached dataset to: {cache_path}')
        else:
            # Load cached examples.
            print(f'Loading examples from: {cache_path}')
            cache_object = pickle.load(open(cache_path, 'rb'))

        self.examples = cache_object['examples']
        self.coco_dset = cache_object['coco_dset']
        # self.mean, self.std = cache_object['mean'], cache_object['std']
        self.mean, self.std = None, None

    def __getitem__(self, index):
        example = self.examples[index]

        # Load image.
        crop = self._load_norm_image(example['image_id'], crop_slice=example['crop_slice'], channels=self.channels)

        # Buffer image.
        crop, buffer_mask = self._add_buffer_to_image(crop,
                                                      self.image_slice.height,
                                                      self.image_slice.width,
                                                      constant_value=-1)

        # Normalize data.
        # crop = (crop - self.mean[:, None, None]) / (self.std[:, None, None] + 1e-5)

        output = {}
        output['image'] = crop
        output['image_id'] = example['image_id']
        output['og_height'] = example['og_height']
        output['og_width'] = example['og_width']
        output['datetime'] = example['datetime']
        output['crop_slice'] = example['crop_slice']
        output['buffer_mask'] = buffer_mask
        return output

    def to_RGB(self, image, gamma=0.4):
        # image: [C, H, W]
        rgb = image[:3]
        rgb = np.clip(rgb**gamma, 0, 1)

        # rgb: [3, H, W]
        return rgb.transpose(1, 2, 0)


def test_get_pixel_dist(args, n_samples=200):
    # Build dataset.
    print('Building dataset ...')
    image_slice_dict = {
        'height': args.crop_size,
        'width': args.crop_size,
        'scale': 1,
        'stride': args.crop_size,
        'n_frames': None
    }
    image_slice_cfg = OmegaConf.create(image_slice_dict)
    dataset = ImageDataset(args.dataset_path, split=args.split, image_slice_cfg=image_slice_cfg)

    # Gather examples from dataset.
    print(f'\nCollecting {n_samples} samples from each dataset ...')
    indices = np.random.choice(list(range(dataset.__len__())), n_samples, replace=False)
    sample_data = []
    for index in indices:
        ex = dataset.__getitem__(index)
        image = ex['image']
        pixels = rearrange(image, 'c h w -> (h w) c')
        indices = np.where(pixels >= 0)
        subpixels = pixels[indices[0], :]
        sample_data.append(subpixels)
    sample_data = np.concatenate(sample_data, axis=0)

    print('\nPixel Distribution')
    print('-' * 20)
    print(f'Pixel dist: {sample_data.mean(axis=0)}')


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_path', type=str, help='Path to dataset file.')
    parser.add_argument('--split', type=str, default='valid')
    parser.add_argument('--crop_size', type=int, default=300)
    parser.add_argument('--feature_type', type=str, default='pixel')
    parser.add_argument('--test_type', type=str, default=None)
    parser.add_argument('--seed_num', type=int, default=0)
    args = parser.parse_args()

    # Set random seeds.
    import torch
    np.random.seed(args.seed_num)
    torch.random.manual_seed(args.seed_num)

    if args.test_type is None:
        image_slice_dict = {
            'height': args.crop_size,
            'width': args.crop_size,
            'scale': 1,
            'stride': args.crop_size,
            'n_frames': None
        }
        image_slice_cfg = OmegaConf.create(image_slice_dict)
        dataset = ImageDataset(args.dataset_path, split=args.split, image_slice_cfg=image_slice_cfg)
        dataset.__getitem__(0)

    elif args.test_type == 'get_pixel_dist_values':
        test_get_pixel_dist(args)
