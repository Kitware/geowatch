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
from PIL import Image
from geowatch.tasks.rutgers_material_seg.utils import utils
from tifffile import tifffile
# if 1:
#     torch.set_printoptions(precision=6, sci_mode=False)

IMG_EXTENSIONS = ['*.png', '*.jpeg', '*.jpg', '*.npy']


# FIXME: Hard coding mean/std is not a good idea
mean_std = ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

channels_dict =  {'B01': 0,  # (Coastal)
            'B02': 1,  # (Blue)
            'B03': 2,  # (Green)
            'B04': 3,  # (Red)
            'B05': 4,  # (RedEdge-1)
            'B06': 5,  # (RedEdge-2)
            'B07': 6,  # (RedEdge-3)
            'B08': 7,  # (NIR)
            'B8A': 8,  # (Narrow NIR)
            'B09': 9,  # (Water Vapor)
            'B10': 10,  # (Cirrus)
            'B11': 11,  # (SWIR-1)
            'B12': 12  # (SWIR-2)
            }


class S2SelfCollectDataset(object):
    def __init__(self, root, transforms, split=False, crop_size=300, channels='B02|B03|B04'):
        self.root = root
        self.transforms = transforms
        self.split = split
        self.crop_size = crop_size
        self.randomcrop_transform = torchvision.transforms.RandomCrop(size=(crop_size, crop_size))

        self.images_root = f"{self.root}regions"
        self.images1_paths = utils.dictionary_contents(path=self.images_root, types=['*.tif'], recursive=True)
        self.channels_idx = [channels_dict[x] for x in channels.split('|')]
        self.channels_delete = [e for e in channels_dict.values() if e not in self.channels_idx]
        # print(self.channels_delete)

    def __getitem__(self, idx):

        img1_path = self.images1_paths[idx]
        base_path = f"{'/'.join(img1_path.split('/')[:-1])}/"
        base_region_path = f"{'/'.join(img1_path.split('/')[:-3])}/"

        unusable_negatives_path = utils.dictionary_contents(path=base_region_path, types=['*.tif'])
        image_name = img1_path.split('/')[-1].split('.')[0].split('_')[0]
        same_region_crops = utils.dictionary_contents(path=base_path, types=['*.tif'])
        negative_region_crops = [e for e in self.images1_paths if e not in unusable_negatives_path]
        same_region_crops.remove(img1_path)

        img2_path = random.sample(same_region_crops, 1)[0]
        negative_image_path = random.sample(negative_region_crops, 1)[0]

        # img1 = Image.open(img1_path)
        img1 = np.delete(tifffile.imread(img1_path), self.channels_delete, axis=-1) / 2
        img2 = np.delete(tifffile.imread(img2_path), self.channels_delete, axis=-1) / 2
        negative_img = np.delete(tifffile.imread(negative_image_path), self.channels_delete, axis=-1) / 2

        new_image1 = self.transforms(img1)
        new_image2 = self.transforms(img2)
        new_negative_image = self.transforms(negative_img)

        crop_params = self.randomcrop_transform.get_params(new_image1, output_size=(self.crop_size, self.crop_size))
        new_image1 = FT.crop(new_image1, *crop_params)
        new_image2 = FT.crop(new_image2, *crop_params)
        new_negative_image = FT.crop(new_negative_image, *crop_params)

        # print(new_image1.dtype)
        # new_image1 = new_image1.double()
        # print(new_image1.dtype)
        # import matplotlib.pyplot as plt
        # plt.imshow(mask)
        # plt.show()

        outputs = {}
        outputs['visuals'] = {'image1': new_image1, 'image2': new_image2, 'image_name': image_name}
        outputs['inputs'] = {'image1': new_image1, 'image2': new_image2, 'negative_image': new_negative_image}

        return outputs

    def __len__(self):
        return len(self.images1_paths)
