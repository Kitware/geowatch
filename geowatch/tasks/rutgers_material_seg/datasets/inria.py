# import os
import numpy as np
import torch
from PIL import Image
import torchvision
# import matplotlib.pyplot as plt
# import math
# from torch.utils.data.dataset import random_split
from geowatch.tasks.rutgers_material_seg.utils import utils
# from torchvision import transforms
# import matplotlib.patches as patches
# import cv2
# import torch.nn.functional as F
import torchvision.transforms.functional as FT
# import xml.etree.ElementTree as ET
# import collections
import random

if 0:
    torch.manual_seed(128)
    torch.cuda.manual_seed(128)
    np.random.seed(128)
    random.seed(128)
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.deterministic = True
    # warnings.filterwarnings('ignore')
    # torch.autograd.set_detect_anomaly(True)
    torch.set_printoptions(precision=6, sci_mode=False)

IMG_EXTENSIONS = ['*.png', '*.jpeg', '*.jpg', '*.npy']


mean_std = ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])


class InriaDataset(object):
    def __init__(self, root: str, transforms: object,
                 split: str = False) -> None:
        self.root = root
        self.transforms = transforms
        self.load_points = False
        self.full_supervision = False

        self.imgs = utils.dictionary_contents(
            path=f"{root}/images/", types=['*.png'])
        # self.imgs = utils.dictionary_contents(path=f"{root}/iarpa/",types=['*.png'])
        if split == "train" or split == "trainaug":
            self.masks_root = f"{self.root}/{split}/full_masks/"
            self.masks_paths = utils.dictionary_contents(
                path=self.perturbed_points_root, types=['*.png'])
        elif split == "val":
            self.full_supervision = True
            self.masks_root = f"{self.root}/{split}/SegmentationClass/"
            self.masks_paths = utils.dictionary_contents(
                path=self.masks_root, types=['*.png'])
        elif split == "test":
            self.test_transform = torchvision.transforms.Compose([
                # transforms.Resize((300,300)),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(
                    [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
            self.masks_root = f"{self.root}/{split}/SegmentationClass/"
            self.masks_paths = utils.dictionary_contents(
                path=self.masks_root, types=['*.png'])

    def __getitem__(self, idx: int) -> dict:
        # load images ad masks
        if self.load_points:
            perturbed_points_mask_path = self.perturbed_masks_paths[idx]
            image_name = perturbed_points_mask_path.split(
                "/")[-1].split(".")[0]
            points_mask_path = f"{self.points_root}{image_name}.png"
            img_path = f"{self.root}images/{image_name}.png"

            points_mask = Image.open(points_mask_path)  # .convert("L")
            perturbed_points_mask = Image.open(
                perturbed_points_mask_path)  # .convert("L")

        elif self.full_supervision:
            mask_path = self.masks_paths[idx]
            image_name = mask_path.split("/")[-1].split(".")[0]
            img_path = f"{self.root}images/{image_name}.png"

            mask = Image.open(mask_path).convert("L")
            mask = np.array(mask)
            mask[mask == 255] = 1
            mask = Image.fromarray(mask)
        else:
            img_path = self.imgs[idx]
            image_name = img_path.split("/")[-1].split(".")[0]

        img = Image.open(img_path).convert("RGB")
        labels_logits = [0 for x in range(2)]

        if self.full_supervision:
            labels = np.unique(mask)
            for label in labels:
                labels_logits[label] = 1
        elif self.load_points:
            labels = np.unique(perturbed_points_mask)
            for label in labels[:-1]:
                labels_logits[label] = 1
        labels_logits.pop(0)

        visual_image = FT.to_tensor(FT.resize(img, (300, 300)))
        if self.full_supervision:
            new_image, new_mask = utils.random_horizonal_flip(img, mask)
            new_image, new_mask = utils.random_vertical_flip(
                new_image, new_mask)
            new_mask = np.array(new_mask).astype(np.float)
            new_mask = FT.to_tensor(new_mask)
        elif self.load_points:
            new_image, new_perturbed_points_mask, new_points_mask = utils.random_horizonal_flip(
                img, perturbed_points_mask, points_mask)
            new_image, new_perturbed_points_mask, new_points_mask = utils.random_vertical_flip(
                new_image, new_perturbed_points_mask, new_points_mask)
            new_perturbed_points_mask = np.array(
                new_perturbed_points_mask).astype(np.float)
            new_perturbed_points_mask = FT.to_tensor(new_perturbed_points_mask)

            new_points_mask = np.array(new_points_mask).astype(np.float)
            new_points_mask = FT.to_tensor(new_points_mask)
        else:
            new_image = img

        labels = torch.FloatTensor(labels_logits)

        if self.full_supervision:
            new_image = self.transforms(new_image)
            outputs = {}
            outputs['visuals'] = {
                'image': visual_image,
                'mask': new_mask,
                'labels': labels,
                'image_name': image_name}
            outputs['inputs'] = {
                'image': new_image,
                'mask': new_mask,
                'labels': labels}
        elif self.load_points:
            new_image = self.transforms(new_image)
            outputs = {}
            outputs['visuals'] = {
                'image': visual_image,
                'mask': new_perturbed_points_mask,
                'labels': labels,
                'image_name': image_name}
            outputs['inputs'] = {
                'image': new_image,
                'mask': new_perturbed_points_mask,
                'labels': labels,
                'points': new_points_mask}
        else:
            new_image = self.test_transform(new_image)
            outputs = {}
            outputs['visuals'] = {
                'image': visual_image,
                'mask': new_image,
                'labels': labels,
                'image_name': image_name}
            outputs['inputs'] = {
                'image': new_image,
                'mask': new_image,
                'labels': labels,
                'points': new_image}

        return outputs

    def __len__(self):
        if self.full_supervision:
            return len(self.masks_paths)
        elif self.load_points:
            return len(self.perturbed_masks_paths)
        else:
            return len(self.imgs)
