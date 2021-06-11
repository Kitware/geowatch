import kwcoco
import ndsampler
import matplotlib.pyplot as plt
import tifffile
import itertools as it
import pytorch_lightning as pl
from torch.utils import data
from datasets import common
import utils
import pathlib

class OneraCD_2018(pl.LightningDataModule):
    def __init__(
        self, 
        train_kwcoco_path=None, 
        test_kwcoco_path=None,
        time_steps=2,
        chip_size=128,
        time_overlap=0,
        chip_overlap=0.1,
        channels='B01|B02|B03|B04|B05|B06|B07|B08|B09|B10|B11|B12|B8A',
        valid_pct=0.1,
        batch_size=4,
        num_workers=4,
    ):
        super().__init__()
        self.train_kwcoco_path = train_kwcoco_path
        self.test_kwcoco_path = test_kwcoco_path
        self.time_steps = time_steps
        self.chip_size = chip_size
        self.time_overlap = time_overlap
        self.chip_overlap = chip_overlap
        self.channels = channels
        self.valid_pct = valid_pct
        self.batch_size = batch_size
        self.num_workers = num_workers
        
    def setup(self, stage):
        transform = utils.Lambda(lambda x: x/2000.)
        
        if stage == "fit" or stage is None:
            kwcoco_ds = kwcoco.CocoDataset(str(self.train_kwcoco_path.expanduser()))
            kwcoco_sampler = ndsampler.CocoSampler(kwcoco_ds)
            train_val_ds = common.VideoDataset(
                kwcoco_sampler,
                sample_shape=(self.time_steps, self.chip_size, self.chip_size),
                window_overlap=(self.time_overlap, self.chip_overlap, self.chip_overlap),
                channels=self.channels,
                transform=transform,
            )
            
            num_examples = len(train_val_ds)
            num_valid = int(self.valid_pct * num_examples)
            num_train = num_examples - num_valid
            
            self.train_dataset, self.valid_dataset = data.random_split(
                train_val_ds, 
                [num_train, num_valid],
            )
            
        if stage == "test" or stage is None:
            kwcoco_ds = kwcoco.CocoDataset(str(self.test_kwcoco_path.expanduser()))
            kwcoco_sampler = ndsampler.CocoSampler(kwcoco_ds)
            test_ds = common.VideoDataset(
                kwcoco_sampler,
                sample_shape=(self.time_steps, self.chip_size, self.chip_size),
                window_overlap=(self.time_overlap, self.chip_overlap, self.chip_overlap),
                channels=self.channels,
                transform=transform,
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
            shuffle=false,
            pin_memory=True,
        )
    
    @staticmethod
    def add_data_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("OneraCD_2018")
        parser.add_argument("--train_kwcoco_path", default=None, type=pathlib.Path) 
        parser.add_argument("--test_kwcoco_path", default=None, type=pathlib.Path)
        parser.add_argument("--time_steps", default=2, type=int)
        parser.add_argument("--chip_size", default=128, type=int)
        parser.add_argument("--time_overlap", default=0, type=int)
        parser.add_argument("--chip_overlap", default=0.1, type=float)
        parser.add_argument("--channels", default='B01|B02|B03|B04|B05|B06|B07|B08|B09|B10|B11|B12|B8A', type=str)
        parser.add_argument("--valid_pct", default=0.1, type=float)
        parser.add_argument("--batch_size", default=4, type=int)
        parser.add_argument("--num_workers", default=4, type=int)
        return parent_parser
