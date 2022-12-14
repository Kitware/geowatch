import os
import pickle

import torch
import kwcoco
import hashlib
import numpy as np
from tqdm import tqdm
from einops import rearrange
from tifffile import tifffile
from sklearn.cluster import KMeans

from watch.tasks.rutgers_material_seg_v2.datasets.base_dataset import BaseDataset
from watch.tasks.rutgers_material_seg_v2.utils.util_features import compute_residual_feature


class MaterialPixelDataset(BaseDataset):
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
                 image_norm_mode=None):

        self.channels = channels
        self.n_clusters = n_clusters
        self.square_size = square_size
        self.feature_type = feature_type
        self.region_shape = region_shape
        self.image_norm_mode = image_norm_mode
        self.image_scale_mode = image_scale_mode

        super(MaterialPixelDataset, self).__init__(kwcoco_path,
                                                   split,
                                                   image_scale_mode=image_scale_mode,
                                                   seed_num=seed_num,
                                                   sensor=sensor)

        # Get examples.
        self._build_dataset()

    def get_n_channels(self):
        # TODO: Update channels.
        if self.channels == 'ALL':
            print('WARNING: ALL channels does NOT include cloud channel or materials.')
            self.channels = 'red|green|blue|nir|swir16|swir22'

        if self.feature_type == 'pixel':
            n_channels = len(self.channels.split('|'))
        elif self.feature_type == 'residual':
            n_channels = self.n_clusters
        else:
            raise NotImplementedError

        return n_channels

    def _create_cach_file_name(self):
        input_str = f'{self.split}_{self.feature_type}_{self.n_clusters}_{self.image_scale_mode}_{self.square_size}_{self.sensor}_{self.channels}'
        sha = hashlib.sha256()
        sha.update(input_str.encode())
        output_str = sha.hexdigest()

        return output_str + '.p'

    def _build_dataset(self):
        # Check to see if examples are already generated.
        cache_dir = os.path.join(self.dset_root, '_cache', 'residual_feature')
        os.makedirs(cache_dir, exist_ok=True)
        cache_file_name = self._create_cach_file_name()
        cache_path = os.path.join(cache_dir, cache_file_name)

        if os.path.exists(cache_path) is False:
            # Create examples.
            print(f'No cached examples found at: {cache_path}')

            # Load KWCOCO file.
            self.coco_dset = kwcoco.CocoDataset(self.kwcoco_path)

            # Get all image ids with auxiliary materials.
            self.examples, pixel_data = [], []
            material_class_ids = []
            image_ids = list(self.coco_dset.imgs.keys())
            for image_id in tqdm(image_ids, colour='green', desc='Generating examples'):
                aux_data = self.coco_dset.imgs[image_id]['auxiliary']

                # Check if image has a material mask.
                if aux_data[-1]['channels'] == 'mat_mask':
                    # Load image.
                    image = self._load_norm_image(image_id)
                    image_height, image_width = image.shape[1], image.shape[2]

                    # Load image and get pixels with material labels.
                    file_path = aux_data[-1]['file_name']
                    mat_label_mask = tifffile.imread(file_path)
                    mask_height, mask_width = mat_label_mask.shape[0], mat_label_mask.shape[1]

                    if (image_height != mask_height) or (image_width != mask_height):
                        if (mask_height == image_width) and (mask_width == image_height):
                            mat_label_mask = mat_label_mask[:image_width, :image_height]
                            mat_label_mask = mat_label_mask.T
                        else:
                            mat_label_mask = mat_label_mask[:image_height, :image_width]
                    mask_height, mask_width = mat_label_mask.shape[0], mat_label_mask.shape[1]

                    # Find pixels where there are labels and the pixels contain information.
                    X, Y = np.where((mat_label_mask != 0) & (image.sum(axis=0) != 0))

                    for i in range(X.shape[0]):
                        example = {}
                        x, y = X[i], Y[i]

                        example['center_loc'] = [x, y]
                        example['image_id'] = image_id
                        example['mat_id'] = mat_label_mask[x, y]

                        self.examples.append(example)

                    # Get pixel data for k-Means computation.
                    img_pixels = image[:, X, Y]  # [channels, n_pixels]

                    # Take mean of pixels
                    pixel_data.append(img_pixels)

                    # Get unique material ids.
                    img_mat_ids = list(np.unique(mat_label_mask))
                    material_class_ids.extend(img_mat_ids)

            # Combine pixel data.
            pixel_data = np.concatenate(pixel_data, axis=1)

            # Compute mean and std of this slice of dataset if split == 'train'.
            mean, std = self._compute_dataset_stats(pixel_data)

            # Compute kMeans
            if self.feature_type != 'pixel':
                pixel_data = (pixel_data - mean[:, None]) / (std[:, None] + 1e-5)
                centroid_clusters = self._compute_clusters(pixel_data, self.n_clusters)
            else:
                centroid_clusters = None

            # Save examples and info needed to load from examples.
            cache_object = {
                'examples': self.examples,
                'coco_dset': self.coco_dset,
                'centroid_centers': centroid_clusters,
                'mean': mean,
                'std': std,
                'n_classes': len(set(material_class_ids))
            }
            pickle.dump(cache_object, open(cache_path, 'wb'))
        else:
            # Load cached examples.
            print(f'Loading examples from: {cache_path}')
            cache_object = pickle.load(open(cache_path, 'rb'))

        self.examples = cache_object['examples']
        self.n_classes = cache_object['n_classes']
        self.coco_dset = cache_object['coco_dset']
        self.centroid_clusters = cache_object['centroid_centers']
        self.mean, self.std = cache_object['mean'], cache_object['std']

    def __getitem__(self, index):
        example = self.examples[index]

        # Load pixel or residual feature.
        pixel_data = self._load_pixel_info(self.feature_type, example['image_id'],
                                           example['center_loc'])  # [channels, height, width]

        if self.feature_type == 'pixel':
            pixel_data = pixel_data.reshape([pixel_data.shape[0], -1]).mean(axis=-1)
        else:
            pixel_data = compute_residual_feature(pixel_data,
                                                  self.centroid_clusters,
                                                  img_norm_mode=self.image_norm_mode,
                                                  local_context_mode=f'precomputed|{self.region_shape}')
            if np.isnan(pixel_data).sum() > 0:
                pixel_data = np.nan_to_num(pixel_data)

        output = {}
        output['pixel_data'] = pixel_data
        output['image_id'] = example['image_id']
        output['center_loc'] = example['center_loc']
        output['target'] = example['mat_id'].astype(int)

        return output

    def _load_pixel_info(self, feature_type, image_id, center_loc):
        # Load pixel information.
        if self.region_shape == 'square':
            # Get pixel indices.
            x, y = center_loc
            radius = (self.square_size - 1) // 2
            h0, _ = x - radius, x + radius + 1  # Add 1 because slice is not inclusive
            w0, _ = y - radius, y + radius + 1  # Add 1 because slice is not inclusive

            # Clip box dimensions to not extend past image.
            h0, w0 = np.clip(h0, 0, a_max=None), np.clip(w0, 0, a_max=None)

            # Get pixel values within the square region.
            pixel_data = self._load_norm_image(image_id, crop_slice=[h0, w0, radius * 2 + 1, radius * 2 + 1])
        else:
            raise NotImplementedError

        # TODO: Optionally compute features.

        return pixel_data

    def _compute_clusters(self, pixels, n_clusters):
        """Compute the centroid clusters of the K-Means algorithm.

        Args:
            pixels (np.array): A float array of shape [feature_dim, n_features]
            n_clusters (int): The number of cluster centroids to fit to data.

        Returns:
            np.array: _description_
        """
        print(f'Computing K-Means of {n_clusters} clusters over {pixels.shape[1]} samples ...')
        kmeans = KMeans(n_clusters=n_clusters, random_state=self.seed_num)
        kmeans.fit(pixels.T)
        return kmeans.cluster_centers_.T

    def format_image_data(self, image_data, buffer_mask):
        batch_ssls = rearrange(image_data, 'b c h w -> b c (h w)')
        return {'pixel_data': batch_ssls}


def test_compare_train_valid_dists(args, n_samples=10000):
    # Create train and valid datasets.
    print('Building datasets ...')
    train_dset = MaterialPixelDataset(args.dataset, split='train', feature_type=args.feature_type)
    valid_dset = MaterialPixelDataset(args.dataset, split='valid', feature_type=args.feature_type)

    # Gather some examples from each dataset.
    print(f'\nCollecting {n_samples} samples from each dataset ...')
    ## Train dataset.
    train_indices = np.random.choice(list(range(train_dset.__len__())), n_samples, replace=False)
    train_sample_data = []
    for index in train_indices:
        train_ex = train_dset.__getitem__(index)
        train_sample_data.append(train_ex['pixel_data'])
    train_sample_data = np.stack(train_sample_data, axis=0)

    ## Valid dataset.
    valid_indices = np.random.choice(list(range(valid_dset.__len__())), n_samples, replace=False)
    valid_sample_data = []
    for index in valid_indices:
        valid_ex = valid_dset.__getitem__(index)
        valid_sample_data.append(valid_ex['pixel_data'])
    valid_sample_data = np.stack(valid_sample_data, axis=0)

    print('\nDistribution comparison')
    print('-' * 20)
    print(f'Train dist: {train_sample_data.mean(axis=0)}')
    print(f'Valid dist: {valid_sample_data.mean(axis=0)}')


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='iarpa_drop4_materials')
    parser.add_argument('--split', type=str, default='valid')
    parser.add_argument('--feature_type', type=str, default='pixel')
    parser.add_argument('--test_type', type=str, default=None)
    parser.add_argument('--seed_num', type=int, default=0)
    args = parser.parse_args()

    # Set random seeds.
    np.random.seed(args.seed_num)
    torch.random.manual_seed(args.seed_num)

    # Run a test.
    if args.test_type is None:
        dataset = MaterialPixelDataset(args.dataset, split=args.split, feature_type=args.feature_type)
        example = dataset.__getitem__(0)
        print(example['pixel_data'].shape)
    elif args.test_type == 'compare_train_valid_dists':
        test_compare_train_valid_dists(args)
