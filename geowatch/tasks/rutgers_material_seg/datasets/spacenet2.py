# flake8: noqa

from PIL import Image
import geowatch.tasks.rutgers_material_seg.utils.utils as utils
import torchvision.transforms.functional as FT
import torch
import numpy as np
from scipy.spatial import distance
import random
import torchvision
# import torchvision.transforms.functional.pad
# torch.backends.cudnn.enabled = True
# torch.backends.cudnn.deterministic = True
import itertools
from geowatch.tasks.rutgers_material_seg.utils import utils
from tifffile import tifffile
import geopandas
import rasterio.mask
from rasterio import features
import albumentations as A
# if 1:
#     torch.set_printoptions(precision=6, sci_mode=False)
torch.manual_seed(128)
torch.cuda.manual_seed(128)
np.random.seed(128)
random.seed(128)
IMG_EXTENSIONS = ['*.png', '*.jpeg', '*.jpg', '*.npy']


# FIXME: Hard coding mean/std is not a good idea
mean_std = ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])


class RandomCrop(object):

    def __init__(self, size, padding=None, pad_if_needed=True, fill=0, padding_mode='constant'):

        self.size = size
        self.padding = padding
        self.pad_if_needed = pad_if_needed
        self.fill = fill
        self.padding_mode = padding_mode

    @staticmethod
    def get_params(img, output_size):
        w, h = img.size
        th, tw = output_size
        if w == tw and h == th:
            return 0, 0, h, w

        i = random.randint(0, h - th)
        j = random.randint(0, w - tw)
        return i, j, th, tw

    def __call__(self, image, mask):

        # pad the width if needed
        if self.pad_if_needed and image.size[0] < self.size[1]:
            image = FT.pad(image, (self.size[1] - image.size[0], 0), self.fill, self.padding_mode)
            mask = FT.pad(mask, (self.size[1] - mask.size[0], 0), self.fill, self.padding_mode)
        # pad the height if needed
        if self.pad_if_needed and image.size[1] < self.size[0]:
            image = FT.pad(image, (0, self.size[0] - image.size[1]), self.fill, self.padding_mode)
            mask = FT.pad(mask, (0, self.size[0] - mask.size[1]), self.fill, self.padding_mode)

        i, j, h, w = self.get_params(image, self.size)
        crop_image = FT.crop(image, i, j, h, w)
        crop_mask = FT.crop(mask, i, j, h, w)

        return crop_image, crop_mask


class SpaceNet2Dataset(object):
    def __init__(self, root, transforms, channels=None, split=False, crop_size=64):
        self.root = root
        self.transforms = transforms
        self.split = split
        self.randomcrop_transform = torchvision.transforms.RandomCrop(size=(crop_size, crop_size))
        self.crop_size = crop_size
        self.ms_transforms = A.Compose([
                                        A.RandomBrightnessContrast(p=0.2),
                                        # A.ColorJitter(),
                                        ])

        self.images_root = f"{self.root}/*/*/MUL/"
        # self.masks_root = f"{self.root}/labels_land_cover_2012/"
        self.images_paths = utils.dictionary_contents(path=self.images_root, types=['*.tif'])
        num_examples = len(self.images_paths)
        num_train = int(num_examples * 0.7)
        if split == 'train':
            self.images_paths = self.images_paths[:num_train]
        elif split == 'val':
            self.images_paths = self.images_paths[num_train:]

    def __getitem__(self, idx):

        image_path = self.images_paths[idx]
        base_path = '/'.join(image_path.split('/')[:-2])
        mask_base_path = f"{base_path}/geojson/buildings/"
        image_name = image_path.split('/')[-1].split(".")[0]
        mask_base_name = "_".join(image_name.split("_")[1:])
        mask_name = f"buildings_{mask_base_name}.geojson"
        mask_path = f"{mask_base_path}{mask_name}"
        img = tifffile.imread(image_path).astype(np.float32)
        og_height, og_width, channels = img.shape

        gdf = geopandas.read_file(mask_path)
        # else:
            # mask = features.rasterize(((polygon, 255) for polygon in gdf['geometry']), out_shape=[og_height, og_width])
            # mask = features.bounds([polygon for polygon in gdf['geometry']], out_shape=[og_height, og_width])
            # features = [feature for feature in gdf["geometry"]]
            # out_image, out_transform = rasterio.mask.mask(img, features, crop=True)

        if len(gdf) == 0:
            mask = np.zeros((og_height, og_width), dtype="int64")
        else:
            with rasterio.open(image_path) as src:
                features = [feature for feature in gdf["geometry"]]
                mask, out_transform = rasterio.mask.mask(src, features, crop=False)
            mask = mask.mean(axis=0)
            mask[mask > 0] = 1

        #     new_image = self.ms_transforms(image=img)

        new_image = self.transforms(img)
        new_mask = FT.to_tensor(mask)  # * 255
        # print(torch.unique(new_image))
        # print(f"new_image min:{new_image.min()}, max:{new_image.max()}")

        crop_params = self.randomcrop_transform.get_params(new_image, output_size=(self.crop_size, self.crop_size))
        new_image = FT.crop(new_image, *crop_params)
        new_mask = FT.crop(new_mask, *crop_params)

        if self.split == "train":
            new_image, new_mask = utils.random_horizonal_flip(new_image, new_mask)
            new_image, new_mask = utils.random_vertical_flip(new_image, new_mask)

        # total_pixels = new_mask.shape[2] * new_mask.shape[1]
        # label_inds, label_counts = torch.unique(new_mask, return_counts=True)
        # label_inds = label_inds.long()
        # distribution = label_counts / total_pixels  # NOQA

        # for label_ind, label_count in zip(label_inds, label_counts):
        #     labels[0, label_ind] = label_count / total_pixels

        # distances = distance.cdist(self.possible_combinations, labels, 'cityblock')
        # label = np.argmin(distances).item()
        outputs = {}
        # outputs['visuals'] = {'image': new_image, 'mask': new_mask, 'image_name': image_name}
        # outputs['inputs'] = {'image': new_image, 'mask': new_mask}
        outputs['image'] = new_image
        outputs['mask'] = new_mask
        outputs['image_name'] = image_name

        return outputs

    def __len__(self):
        return len(self.images_paths)
