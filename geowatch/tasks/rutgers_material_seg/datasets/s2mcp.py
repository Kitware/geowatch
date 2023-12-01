# flake8: noqa

from PIL import Image
import torchvision.transforms.functional as FT
import torchvision
import torch
import numpy as np
from scipy.spatial import distance
import random
# import torchvision.transforms.functional.pad
# torch.backends.cudnn.enabled = True
# torch.backends.cudnn.deterministic = True
import itertools
from geowatch.tasks.rutgers_material_seg.utils import utils

# if 1:
#     torch.set_printoptions(precision=6, sci_mode=False)

IMG_EXTENSIONS = ['*.png', '*.jpeg', '*.jpg', '*.npy']


# FIXME: Hard coding mean/std is not a good idea
mean_std = ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])


# class RandomCrop(object):

#     def __init__(self, size, padding=None, pad_if_needed=True, fill=0, padding_mode='constant'):

#         self.shape = size
#         self.padding = padding
#         self.pad_if_needed = pad_if_needed
#         self.fill = fill
#         self.padding_mode = padding_mode

#     @staticmethod
#     def get_params(img, output_size):

#         w, h, c  = img.shape
#         th, tw = output_size
#         if w == tw and h == th:
#             return 0, 0, h, w

#         i = random.randint(0, h - th)
#         j = random.randint(0, w - tw)
#         return i, j, th, tw

#     def __call__(self, image, mask):

#         i, j, h, w = self.get_params(image, self.shape)
#         crop_image = FT.crop(image, i, j, h, w)
#         crop_mask = FT.crop(mask, i, j, h, w)

#         return crop_image, crop_mask


class S2MCPDataset(object):
    def __init__(self, root, transforms, split=False, crop_size=300):
        self.root = root
        self.transforms = transforms
        self.split = split
        self.crop_size = crop_size
        self.randomcrop_transform = torchvision.transforms.RandomCrop(size=(crop_size, crop_size))

        self.images_root = f"{self.root}/"
        self.images1_paths = utils.dictionary_contents(path=self.images_root, types=['*a.npy'])

    def __getitem__(self, idx):
        negative_idx = random.choice([i for i in range(0, self.__len__()) if i != idx])

        img1_path = self.images1_paths[idx]
        image_name = img1_path.split('/')[-1].split('.')[0].split('_')[0]

        img2_path = f"{self.images_root}/{image_name}_b.npy"

        negative_image_name = self.images1_paths[negative_idx].split('/')[-1].split('.')[0]
        negative_image_path = f"{self.images_root}/{negative_image_name}.npy"

        img1 = np.load(img1_path)[:, :, :13]
        img2 = np.load(img2_path)[:, :, :13]
        negative_img = np.load(negative_image_path)

        new_image1 = self.transforms(img1)
        new_image2 = self.transforms(img2)
        new_negative_image = self.transforms(negative_img)

        crop_params = self.randomcrop_transform.get_params(new_image1, output_size=(self.crop_size, self.crop_size))
        new_image1 = FT.crop(new_image1, *crop_params)
        new_image2 = FT.crop(new_image2, *crop_params)
        new_negative_image = FT.crop(new_negative_image, *crop_params)

        outputs = {}
        outputs['visuals'] = {'image1': new_image1, 'image2': new_image2, 'image_name': image_name}
        outputs['inputs'] = {'image1': new_image1, 'image2': new_image2, 'negative_image': new_negative_image}

        return outputs

    def __len__(self):
        return len(self.images1_paths)
