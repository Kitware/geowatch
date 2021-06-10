import kwcoco
import ndsampler
import matplotlib.pyplot as plt
import tifffile
import itertools as it
import pytorch_lightning as pl
from torch.utils import data
from datasets import common
import utils

class Drop0AlignMSI_S2(pl.LightningDataModule):
    def __init__(
        self, 
        train_kwcoco_path=None, 
        test_kwcoco_path=None,
        time_steps=2,
        chip_size=128,
        time_overlap=0,
        chip_overlap=0.1,
        channels='costal|blue|green|red|B05|B06|B07|nir|B09|cirrus|swir16|swir22|B8A',
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
        
    def preprocess_ds(self, project_ds):
        project_ds.remove_images([
            image
            for image in project_ds.dataset["images"]
            if "S2-TrueColor" not in image["sensor_candidates"]
        ])
        project_ds.remove_videos([2,3])

        for image in project_ds.dataset["images"]:
            image["warp_img_to_vid"] = {
                'type': 'affine',
                'matrix': [
                    [1.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0],
                    [0.0, 0.0, 1.0],
                ]}

            for band in image["auxiliary"]:
                band["warp_aux_to_img"]["type"] = "affine"

        for video in project_ds.dataset["videos"]:
            video_id = video["id"]

            video["width"] = max([
                    image["width"]
                    for image in project_ds.dataset["images"]
                    if image["video_id"] == video["id"]
                ])
            video["height"] = max([
                    image["height"]
                    for image in project_ds.dataset["images"]
                    if image["video_id"] == video["id"]
                ])
            video["num_frames"] = len([
                    image
                    for image in project_ds.dataset["images"]
                    if image["video_id"] == video["id"]
                ])
            video["available_channels"] = list(set(it.chain.from_iterable([
                    [
                        band["channels"]
                        for band in image["auxiliary"]
                    ]
                    for image in project_ds.dataset["images"]
                    if image["video_id"] == video_id
                ])))
            video["warp_wld_to_vid"] = {
                'type': 'affine',
                'matrix': [
                    [1.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0],
                    [0.0, 0.0, 1.0],
                ]}

        project_ds.validate()
        return project_ds
        
    def setup(self, stage):
        transform = utils.Lambda(lambda x: x/2000.)
        
        if stage == "fit" or stage is None:
            kwcoco_ds = kwcoco.CocoDataset(self.train_kwcoco_path)
            kwcoco_ds = self.preprocess_ds(kwcoco_ds)
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
            kwcoco_ds = kwcoco.CocoDataset(self.test_kwcoco_path)
            kwcoco_ds = self.preprocess_ds(kwcoco_ds)
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