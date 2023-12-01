import os
from glob import glob

import numpy as np
from PIL import Image
from tifffile import tifffile
import torchvision.transforms.functional as FT
import torchvision
from torch.utils.data import Dataset
# import torch


class DynEarthNetDataset(Dataset):
    """
    Assumptions:
    - Only using data with labels (10 regions)
    - Test split does not have labels
        - Need to submit results to:
        https://competitions.codalab.org/competitions/30441#participate-submit_results
    - Divide splits based on regions.
    """

    def __init__(self,
                 root,
                 transforms,
                 split,
                 crop_size=300,
                 channels='B02|B03|B04',
                 #  sensor_type='S2',
                 seed_num=0):

        self.split = split
        self.root = root
        self.transforms = transforms
        self.sensor_type = 'S2'
        self.crop_size = crop_size
        self.randomcrop_transform = torchvision.transforms.RandomCrop(
            size=(crop_size, crop_size))

        # Get requested channels
        if channels is None:
            self.band_names = None
        else:
            self.band_names = channels.split('|')

        # Check inputs
        assert split in ['train', 'valid', 'test']
        assert self.sensor_type in ['S2', 'planet']

        # Handle train/validation splits separately from test split.

        if split in ['train', 'valid']:
            # Get all regions with labels.
            label_dirs = sorted(glob(os.path.join(root, 'Labels') + '/*/'))

            # There are no train/valid/test splits so we have to make our own splits.
            # Use random seed to make the split consistent.
            np.random.seed(seed_num)
            np.random.shuffle(label_dirs)

            # Use a 70/20/10 split for train/valid/test
            num_examples = len(label_dirs)
            num_train = int(num_examples * 0.7)

            if split == 'train':
                label_dirs = label_dirs[:num_train]
            elif split == 'valid':
                label_dirs = label_dirs[num_train:]

        else:
            # Handle test split.
            breakpoint()

        # Get image paths based on sensor type.
        self.dataset = []
        if self.sensor_type == 'S2':
            # Parse label names for S2-specific region names.
            sensor_dir_names = [
                '_'.join(p.split('/')[-2].split('_')[1:]) for p in label_dirs]

            # Find region directories for S2 imagery.
            base_sensor_dir = os.path.join(root, 'sentinel-2')
            region_dirs = [os.path.join(base_sensor_dir, name)
                           for name in sensor_dir_names]

            # Get sequential pairs of image paths.
            for region_dir, label_dir in zip(region_dirs, label_dirs):
                # Get all images in region.
                image_paths = sorted(glob(region_dir + '/*.tif'))

                # Get all label images in region.
                label_paths = sorted(glob(label_dir + '/*.png'))

                # Check that there are correct number of labels.
                assert len(label_paths) == len(image_paths) - 1

                # Get sequential pairs of image paths.
                for i in range(len(label_paths)):
                    example = {}
                    example['label_path'] = label_paths[i]
                    example['image_path_1'] = image_paths[i]
                    example['image_path_2'] = image_paths[i + 1]

                    self.dataset.append(example)

            # Get band indices to sample.
            # TODO: Figure out
            if self.band_names is None:
                self.band_indices = None
            else:
                band_indices_dict = {
                    'B01': 0,
                    'B02': 1,
                    'B03': 2,
                    'B04': 3,
                    'B05': 4,
                    'B06': 5,
                    'B07': 6,
                    'B08': 7}
                self.band_indices = []
                for band_name in self.band_names:
                    self.band_indices.append(band_indices_dict[band_name])

        elif self.sensor_type == 'planet':
            # TODO: Need to sample planet images temporally.
            # 120 Planet images for 23 labels.
            raise NotImplementedError(
                'Not sure how to handle Planet sensor images.')
            sensor_dir_names = [p.split('/')[-2].split('_')[0]
                                for p in label_dirs]

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        example = self.dataset[index]

        # Load images.
        image_1 = tifffile.imread(example['image_path_1']).astype(np.float32)
        image_2 = tifffile.imread(example['image_path_2']).astype(np.float32)

        # Sample bands based on requested channels.
        image_1 = np.take(image_1, self.band_indices, axis=2)
        image_2 = np.take(image_2, self.band_indices, axis=2)

        # Load label image.
        label = Image.open(example['label_path'])
        label = FT.to_tensor(label) * 255
        # Normalize images from uint16 to float
        # image_1 = image_1 / 2**16
        # image_2 = image_2 / 2**16

        # Compute transforms on image
        if self.transforms:
            # TODO: Make this transforms consistent.
            image_1 = self.transforms(image_1)
            image_2 = self.transforms(image_2)
            # label = self.transforms(label)

        crop_params = self.randomcrop_transform.get_params(
            image_1, output_size=(self.crop_size, self.crop_size))
        image_1 = FT.crop(image_1, *crop_params)
        image_2 = FT.crop(image_2, *crop_params)
        label = FT.crop(label, *crop_params)

        outputs = {}
        outputs['image1'] = image_1
        outputs['image2'] = image_2
        outputs['mask'] = label

        return outputs
