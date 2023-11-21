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
# if 1:
#     torch.set_printoptions(precision=6, sci_mode=False)

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


class HRSCDDataset(object):
    def __init__(self, root, transforms, channels=None, split=False, crop_size=64):
        self.root = root
        self.transforms = transforms
        self.split = split
        self.randomcrop_transform = torchvision.transforms.RandomCrop(size=(crop_size, crop_size))
        self.crop_size = crop_size

        self.images_root = f"{self.root}/{split}/images_2012/"
        self.masks_root = f"{self.root}/labels_land_cover_2012/"
        self.masks_paths = utils.dictionary_contents(path=self.masks_root, types=['*.tif'], recursive=True)
        num_examples = len(self.masks_paths)
        num_train = int(num_examples * 0.7)
        if split == 'train':
            self.masks_paths = self.masks_paths[:num_train]
        elif split == 'val':
            self.masks_paths = self.masks_paths[num_train:]
        # print(len(self.masks_paths))

    def __getitem__(self, idx):

        mask_path = self.masks_paths[idx]
        img_path = mask_path.split('/')
        img_path[-4] = 'images_2012'
        img_path = '/'.join(img_path)
        # print(img_path)
        image_name = mask_path.split('/')[-1].split('.')[0]
        # img_path = f"{self.images_root}/{image_name}.png"

        img = tifffile.imread(img_path).astype(np.float32)
        mask = tifffile.imread(mask_path).astype(np.float32)

        # img, mask = self.randomcrop_transform(img, mask)

        new_image = self.transforms(img)
        new_mask = FT.to_tensor(mask)  # * 255

        crop_params = self.randomcrop_transform.get_params(new_image, output_size=(self.crop_size, self.crop_size))
        new_image = FT.crop(new_image, *crop_params)
        new_mask = FT.crop(new_mask, *crop_params)

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

        return outputs

    def __len__(self):
        return len(self.masks_paths)
