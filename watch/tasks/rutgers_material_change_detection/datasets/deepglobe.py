import os
import numpy as np
import torch
from PIL import Image
import torchvision
import matplotlib.pyplot as plt
import math
from torch.utils.data.dataset import random_split
import material_seg.utils.utils as utils
from torchvision import transforms
import matplotlib.patches as patches
import cv2
import torch.nn.functional as F
import torchvision.transforms.functional as FT
import xml.etree.ElementTree as ET
import collections
import random

# torch.backends.cudnn.enabled = True
# torch.backends.cudnn.deterministic = True
# torch.set_printoptions(precision=6, sci_mode=False)

IMG_EXTENSIONS = ['*.png', '*.jpeg', '*.jpg','*.npy']

mean_std = ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

class DeepGlobeDataset(object):
    def __init__(self, root, transforms, split=False):
        self.root = root
        self.transforms = transforms
        self.split = split

        self.images_root = f"{self.root}/{split}/images/"
        self.masks_root = f"{self.root}/{split}/masks/"
        self.masks_paths = utils.dictionary_contents(path=self.masks_root,types=['*.png'])

        self.mask_mapping = {
                                0:   0, # 0, unknown
                                179:   1, # 179, urban land
                                226: 2, # 226, agriculture land
                                105: 3, # 105, rangeland, non-forest, non farm, green land
                                150: 4, # 150, forest land
                                29: 5, # 29, water
                                255: 6, # 255, barren land, mountain, rock, dessert
                            }


    def __getitem__(self, idx):
        
        mask_path = self.masks_paths[idx]
        image_name = mask_path.split('/')[-1].split('.')[0]
        img_path = f"{self.images_root}/{image_name}.png"

        img = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path)#.convert("L"))

        new_mask = FT.to_tensor(mask)*255

        new_image = self.transforms(img)
        outputs = {}
        outputs['visuals'] = {'image':new_image, 'mask':new_mask, 'image_name':image_name}
        outputs['inputs'] = {'image':new_image, 'mask':new_mask}
        
        return outputs

    def __len__(self):
        return len(self.masks_paths)
