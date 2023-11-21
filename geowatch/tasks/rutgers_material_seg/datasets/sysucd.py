# flake8: noqa

from PIL import Image
import torchvision.transforms.functional as FT
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


class SYSUCDDataset(object):
    def __init__(self, root, transforms, split=False):
        self.root = root
        self.transforms = transforms
        self.split = split

        self.images1_root = f"{self.root}/{split}/time1/"
        self.images2_root = f"{self.root}/{split}/time2/"
        self.masks_root = f"{self.root}/{split}/label/"
        self.masks_paths = utils.dictionary_contents(path=self.masks_root, types=['*.png'])

    def __getitem__(self, idx):
        negative_idx = random.choice([i for i in range(0, self.__len__()) if i != idx])

        mask_path = self.masks_paths[idx]
        negative_image_name = self.masks_paths[negative_idx].split('/')[-1].split('.')[0]
        image_name = mask_path.split('/')[-1].split('.')[0]
        img1_path = f"{self.images1_root}/{image_name}.png"
        img2_path = f"{self.images2_root}/{image_name}.png"
        negative_image_path = f"{self.images1_root}/{negative_image_name}.png"

        img1 = Image.open(img1_path).convert("RGB")
        img2 = Image.open(img2_path).convert("RGB")
        negative_img = Image.open(negative_image_path).convert("RGB")
        mask = Image.open(mask_path)  # .convert("L"))

        new_image1 = self.transforms(img1)
        new_image2 = self.transforms(img2)
        new_negative_image = self.transforms(negative_img)

        print(new_image1.dtype)
        # import matplotlib.pyplot as plt
        # plt.imshow(mask)
        # plt.show()
        new_mask = FT.to_tensor(mask)

        outputs = {}
        outputs['visuals'] = {'image1': new_image1, 'image2': new_image2, 'mask': new_mask, 'image_name': image_name}
        outputs['inputs'] = {'image1': new_image1, 'image2': new_image2, 'negative_image': new_negative_image , 'mask': new_mask}

        return outputs

    def __len__(self):
        return len(self.masks_paths)
