import pytorch_lightning as pl
from torch.utils import data

import pathlib
from PIL import Image
import numpy as np

import albumentations as A
from torchvision import transforms

from .. import utils

class SUN_RGBD_Dataset(data.Dataset):
    def __init__(self, 
                 data_root, 
                 split="train", 
                 augment_step=None, 
                 transform_step=None,
                ):
        self.data_root = data_root
        self.split = split
        self.augment_step = augment_step
        self.transform_step = transform_step
        
        if split == "train":
            self.size = 5285
        elif split == "test":
            self.size = 5050
        else:
            raise "Unknown split"
            
    def __len__(self):
        return self.size
    
    def __getitem__(self, idx):
        idx += 1 # dataset is one indexed, blergh
        
        image = np.array(Image.open(self.data_root / self.split / f"images/img-{idx:06d}.jpg"))
        depth = np.array(Image.open(self.data_root / self.split / f"depth/depth-{idx:06d}.png"))
        labels = np.array(Image.open(self.data_root / self.split / f"labels/img13labels-{idx:06d}.png")).astype("int")
        
        if self.augment_step:
            augmented = self.augment_step(image=image, depth=depth, mask=labels)
            image = augmented["image"]
            depth = augmented["depth"]
            labels = augmented["mask"]
        
        inputs = np.concatenate([image, depth[...,None]], axis=-1)
        
        if self.transform_step:
            inputs = self.transform_step(inputs)
            
        labels -= 1
        labels[labels == -1] = -100
            
        return {
            "images": inputs,
            "labels": labels,
        }
    
class SUN_RGBD(pl.LightningDataModule):
    def __init__(
        self, 
        data_root,
        valid_pct=0.1,
        batch_size=4,
        num_workers=4,
        preprocessing_step=None,
        tfms_train_channel_size=1000,
    ):
        super().__init__()
        
        self.data_root = data_root
        self.valid_pct = valid_pct
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.preprocessing_step = preprocessing_step
        self.tfms_train_channel_size = tfms_train_channel_size

        self.train_tfms = transforms.Compose([
            self.preprocessing_step,
            utils.DimensionDropout(0, self.tfms_train_channel_size),
        ])
        self.test_tfms = transforms.Compose([
            self.preprocessing_step,
        ])
        
    def setup(self, stage):
        
        if stage == "fit" or stage is None:
            aug = A.Compose([
                #     A.RandomCrop(256, 256),
                    A.RandomResizedCrop(128, 128),
                    A.RandomToneCurve(),
                    A.ColorJitter(),
                    A.HorizontalFlip(),
                ], additional_targets={"depth": "mask"})
            

            train_val_ds = SUN_RGBD_Dataset(
                pathlib.Path(self.data_root), 
                split="train",
                augment_step=aug,
                transform_step=self.train_tfms,
            )

            num_examples = len(train_val_ds)
            num_valid = int(self.valid_pct * num_examples)
            num_train = num_examples - num_valid

            self.train_dataset, self.valid_dataset = data.random_split(
                train_val_ds,
                [num_train, num_valid],
            )
        
        if stage == "test" or stage is None:
            
            self.test_dataset = SUN_RGBD_Dataset(
                pathlib.Path(self.data_root), 
                split="test",
                transform_step=self.test_tfms,
            )

    def train_dataloader(self):
        return data.DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
            pin_memory=True,
        )

    def val_dataloader(self):
        return data.DataLoader(
            self.valid_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            pin_memory=True,
        )

    def test_dataloader(self):
        return data.DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            pin_memory=True,
        )

    @staticmethod
    def add_data_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("SUN_RGBD")
        parser.add_argument("--data_root", required=True, type=pathlib.Path)
        parser.add_argument("--valid_pct", default=0.1, type=float)
        parser.add_argument("--batch_size", default=4, type=int)
        parser.add_argument("--num_workers", default=4, type=int)
        parser.add_argument("--tfms_train_channel_size", default=1000, type=int)
        return parent_parser