# import os
# import cv2
import json
import numpy as np
from glob import glob
from tifffile import tifffile

import torch
import torchvision
from torch.utils.data import Dataset
# import torchvision.transforms.functional as FT
# from PIL import Image


class BigEarthNetDataset(Dataset):
    def __init__(self, root, transforms, split='train',
                 crop_size=300, channels='B02|B03|B04', rnd_seed=0):
        self.root = root
        self.split = split
        self.crop_size = crop_size
        self.transforms = transforms
        self.randomcrop_transform = torchvision.transforms.RandomCrop(
            size=(crop_size, crop_size))

        # Get requested channels
        self.band_names = channels.split('|')

        # Check inputs
        assert split in ['train', 'val', 'test']

        # Get all image files
        image_dirs = glob(root + '/*/')
        assert len(image_dirs) > 0

        # There are no train/valid/test splits so we have to make our own splits.
        # Use random seed to make the split consistent.
        np.random.seed(rnd_seed)
        np.random.shuffle(image_dirs)

        # Use a 70/20/10 split for train/valid/test
        num_examples = len(image_dirs)
        num_train = int(num_examples * 0.5)
        num_valid = int(num_examples * 0.1)
        self.resize_transform = torchvision.transforms.Resize((120, 120))
        if split == 'train':
            self.image_dirs = image_dirs[:num_train]
        elif split == 'val':
            self.image_dirs = image_dirs[num_train:(num_train + num_valid)]
        elif split == 'test':
            self.image_dirs = image_dirs[(num_train + num_valid):]
        else:
            NameError(f'Invalid split name: {split}')

    def __len__(self):
        return len(self.image_dirs)

    def __getitem__(self, index):
        # Get all image paths in image directory.
        image_dir = self.image_dirs[index]
        image_paths = glob(image_dir + '/*.tif')

        # Find correct band image path by matching the band name to the image
        # name.
        band_paths = []
        for band_name in self.band_names:
            band_path = None
            for image_path in image_paths:
                if band_name in image_path:
                    band_path = image_path
            if band_path is None:
                raise FileNotFoundError(
                    f'Could not find image path for band name: {band_name} \nIn this dir: {image_dir}')
            band_paths.append(band_path)
        band_images = []
        for p in band_paths:
            band = tifffile.imread(p).astype(np.float32)
            band_tensor = torch.tensor(band).unsqueeze(0)
            band_tensor = self.resize_transform(band_tensor).squeeze(0)
            band_images.append(band_tensor)
        # band_images = [FT.resize(torch.tensor(tifffile.imread(p).astype(np.float32)), size=(120,120)) for p in band_paths]

        stack_image = torch.stack(band_images, axis=0)
        # print(f"stack_image min: {stack_image.min()} max: {stack_image.max()}")
        # Resize the image to desired crop size if smaller.
        C, H, W = stack_image.shape

        # if H < self.crop_size or W < self.crop_size:
        #     stack_image = cv2.resize(stack_image.transpose(1, 2, 0),
        #                              (self.crop_size, self.crop_size)).transpose(2, 0, 1)

        # Normalize image from uint16 to float
        # stack_image = stack_image / 2**16

        # Compute transforms on image
        # if self.transforms:
        #     stack_image = self.transforms(stack_image)

        # Load labels for this image
        label_path = glob(image_dir + '/*.json')[0]
        metadata = json.load(open(label_path, 'r'))
        str_labels = metadata['labels']
        # print(str_labels)
        onehot_label = self.label_names_to_onehot(str_labels)

        outputs = {}
        outputs['image'] = stack_image
        outputs['label'] = onehot_label

        return outputs

    def label_names_to_onehot(self, label_names: list):
        # "original_labels":{
        #                 "Continuous urban fabric": 0,
        #                 "Discontinuous urban fabric": 1,
        #                 "Industrial or commercial units": 2,
        #                 "Road and rail networks and associated land": 3,
        #                 "Port areas": 4,
        #                 "Airports": 5,
        #                 "Mineral extraction sites": 6,
        #                 "Dump sites": 7,
        #                 "Construction sites": 8,
        #                 "Green urban areas": 9,
        #                 "Sport and leisure facilities": 10,
        #                 "Non-irrigated arable land": 11,
        #                 "Permanently irrigated land": 12,
        #                 "Rice fields": 13,
        #                 "Vineyards": 14,
        #                 "Fruit trees and berry plantations": 15,
        #                 "Olive groves": 16,
        #                 "Pastures": 17,
        #                 "Annual crops associated with permanent crops": 18,
        #                 "Complex cultivation patterns": 19,
        #                 "Land principally occupied by agriculture, with significant areas of natural vegetation": 20,
        #                 "Agro-forestry areas": 21,
        #                 "Broad-leaved forest": 22,
        #                 "Coniferous forest": 23,
        #                 "Mixed forest": 24,
        #                 "Natural grassland": 25,
        #                 "Moors and heathland": 26,
        #                 "Sclerophyllous vegetation": 27,
        #                 "Transitional woodland/shrub": 28,
        #                 "Beaches, dunes, sands": 29,
        #                 "Bare rock": 30,
        #                 "Sparsely vegetated areas": 31,
        #                 "Burnt areas": 32,
        #                 "Inland marshes": 33,
        #                 "Peatbogs": 34,
        #                 "Salt marshes": 35,
        #                 "Salines": 36,
        #                 "Intertidal flats": 37,
        #                 "Water courses": 38,
        #                 "Water bodies": 39,
        #                 "Coastal lagoons": 40,
        #                 "Estuaries": 41,
        #                 "Sea and ocean": 42

        # label_conversion = [[0, 1], "Continuous urban fabric": 0,
        #         #                 "Discontinuous urban fabric": 1,
        #                     [2], #"Industrial or commercial units": 2,
        #                     [11, 12, 13],  #"Non-irrigated arable land": 11,
        #                 #                 "Permanently irrigated land": 12,
        #                 #                 "Rice fields": 13,
        #                     [14, 15, 16, 18],  #"Vineyards": 14,
        #                     #                 "Fruit trees and berry plantations": 15,
        #                     #                 "Olive groves": 16,
        #                                     #    "Annual crops associated with permanent crops": 18,
        #                     [17],   #"Pastures": 17,
        #                     [19], # "Complex cultivation patterns": 19,
        #                     [20], # "Land principally occupied by agriculture, with significant areas of natural vegetation": 20,
        #                     [21], "Agro-forestry areas": 21,
        #                     [22], "Broad-leaved forest": 22,
        #                     [23], "Coniferous forest": 23,
        #                     [24], "Mixed forest": 24,
        #                     [25, 31], "Natural grassland": 25,
        #                                 "Sparsely vegetated areas": 31,
        #                     [26, 27], "Moors and heathland": 26,
        #             #                 "Sclerophyllous vegetation": 27,
        #                     [28],  "Transitional woodland/shrub": 28,
        #                     [29], "Beaches, dunes, sands": 29,
        #                     [33, 34], "Inland marshes": 33,
        #             #                 "Peatbogs": 34,
        #                     [35, 36], "Salt marshes": 35,
        #             #                 "Salines": 36,
        #                     [38, 39], "Water courses": 38,
        #             #                 "Water bodies": 39,
        #                     [40, 41, 42] "Coastal lagoons": 40,
        #                 #                 "Estuaries": 41,
        #                 #                 "Sea and ocean": 42
        #                     ]

        dset_label_map = {
            "Urban fabric": 0,
            "Continuous urban fabric": 0,
            "Discontinuous urban fabric": 0,
            'Road and rail networks and associated land': 0,
            'Green urban areas': 0,
            'Port areas': 0,
            "Sport and leisure facilities": 0,
            "Industrial or commercial units": 1,
            "Airports": 1,
            'Construction sites': 1,
            'Dump sites': 1,
            "Arable land": 2,
            "Non-irrigated arable land": 2,
            "Permanently irrigated land": 2,
            "Rice fields": 2,
            "Permanent crops": 3,
            "Vineyards": 3,
            'Annual crops associated with permanent crops': 3,
            "Fruit trees and berry plantations": 3,
            "Olive groves": 3,
            "Pastures": 4,
            "Complex cultivation patterns": 5,
            "Land principally occupied by agriculture, with significant areas of natural vegetation": 6,
            "Agro-forestry areas": 7,
            "Broad-leaved forest": 8,
            "Coniferous forest": 9,
            "Mixed forest": 10,
            "Natural grassland and sparsely vegetated areas": 11,
            "Natural grassland": 11,
            "Sparsely vegetated areas": 11,
            "Moors, heathland and sclerophyllous vegetation": 12,
            "Moors and heathland": 12,
            "Mineral extraction sites": 12,
            'Bare rock': 12,
            'Burnt areas': 12,
            "Sclerophyllous vegetation": 12,
            "Transitional woodland, shrub": 13,
            "Transitional woodland/shrub": 13,
            "Beaches, dunes, sands": 14,
            "Inland wetlands": 15,
            "Inland marshes": 15,
            "Peatbogs": 15,
            "Coastal wetlands": 16,
            "Salt marshes": 16,
            "Intertidal flats": 16,
            "Salines": 16,
            "Inland waters": 17,
            "Water courses": 17,
            "Water bodies": 17,
            "Marine waters": 18,
            "Coastal lagoons": 18,
            "Estuaries": 18,
            "Sea and ocean": 18
        }

        assert isinstance(label_names, list)
        assert len(label_names) > 0

        # 19 classes based on http://bigearth.net/
        onehot = torch.zeros(19, dtype=torch.long)

        for label_name in label_names:
            label_index = dset_label_map[label_name]
            onehot[label_index] = 1

        return onehot
