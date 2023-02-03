import albumentations as A
import kwcoco
import ndsampler
import ubelt as ub
from utils.masking_generator import RandomMaskingGenerator
import torchvision
from torchvision import transforms
import datetime
import sys
import torch
import torch.nn as nn
import pytorch_lightning as pl
from pytorch_lightning import LightningModule
from torch.nn import L1Loss as MSE
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms, datasets
#import albumentations as A
import torch.nn.functional as F
from einops import repeat, rearrange
from einops.layers.torch import Rearrange
from pytorch_lightning.callbacks import ModelCheckpoint
import numpy as np
from vit_pytorch.vit import Transformer
#from albumentations.pytorch import transforms
import wandb
from contextlib import redirect_stdout, redirect_stderr
from pytorch_lightning.loggers import WandbLogger
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
sys.path.append('/storage1/fs1/jacobsn/Active/user_s.sastry/watch/')
from ffcv.writer import DatasetWriter
from ffcv.fields import NDArrayField
class WatchDataset(Dataset):
    S2_l2a_channel_names = [
        'B02.tif', 'B01.tif', 'B03.tif', 'B04.tif', 'B05.tif', 'B06.tif', 'B07.tif', 'B08.tif', 'B09.tif', 'B11.tif', 'B12.tif', 'B8A.tif'
    ]
    S2_channel_names = [
        'coastal', 'blue', 'green', 'red', 'B05', 'B06', 'B07', 'nir', 'B09', 'cirrus', 'swir16', 'swir22', 'B8A'
    ]
    L8_channel_names = [
        'coastal', 'lwir11', 'lwir12', 'blue', 'green', 'red', 'nir', 'swir16', 'swir22', 'pan', 'cirrus'
    ]
    def __init__(self, coco_dset, sensor=['S2'], bands=['shared'],
                 segmentation=False, patch_size=224, mask_patch_size=16, num_images=2,
                 mode='train', patch_overlap=.25, bas=True, rng=None, mask_pct=.5, mask_time_width=2,
                 temporal_mode='cat'):
        super().__init__()
        if not isinstance(bands, list):
            bands = [bands]
        if not isinstance(sensor, list):
            sensor = [sensor]
        assert(temporal_mode in ['cat', 'stack'])

        # initialize dataset
        print('load dataset')
        self.coco_dset: kwcoco.CocoDataset = kwcoco.CocoDataset.coerce(coco_dset)

        print('filter dataset')
        # Filter out worldview images (better to use subset than remove)
        images: kwcoco.coco_objects1d.Images = self.coco_dset.images()
        flags = [s in sensor for s in images.lookup('sensor_coarse')]
        valid_image_ids : list[int] = list(images.compress(flags))
        self.coco_dset = self.coco_dset.subset(valid_image_ids)

        self.images : kwcoco.coco_objects1d.Images = self.coco_dset.images()
        self.sampler = ndsampler.CocoSampler(self.coco_dset)

        window_dims = [num_images, patch_size, patch_size]

        NEW_GRID = 1
        if NEW_GRID:
            print('make grid')
            from watch.tasks.fusion.datamodules.kwcoco_video_data import sample_video_spacetime_targets
            sample_grid = sample_video_spacetime_targets(
                self.coco_dset, window_dims=window_dims,
                window_overlap=patch_overlap,
                time_sampling='hardish3', time_span='1y',
                use_annot_info=False,
                keepbound=True,
                exclude_sensors=['WV'],
                use_centered_positives=False,
            )
            samples = sample_grid['targets']
            for tr in samples:
                tr['vidid'] = tr['video_id']  # hack
            print('made grid')
        else:
            grid = self.sampler.new_sample_grid(**{
                'task': 'video_detection',
                'window_dims': [num_images, patch_size, patch_size],
                'window_overlap': patch_overlap,
            })
            if segmentation:
                samples = grid['positives']
            else:
                samples = grid['positives'] + grid['negatives']

        # vidid_to_patches = ub.group_items(samples, key=lambda x: x['vidid'])
        # self.vidid_to_patches = vidid_to_patches
        print('build patches')
        grouped = ub.group_items(
                samples,
                lambda x: tuple(
                    [x['vidid']] + [gid for gid in x['gids']]
                )
                )
        grouped = ub.sorted_keys(grouped)
        self.patches : list[dict] = list(ub.flatten(grouped.values()))

        all_bands = [aux.get('channels', None) for aux in self.coco_dset.index.imgs[self.images._ids[0]].get('auxiliary', [])]

        if 'r|g|b' in all_bands:
            all_bands.remove('r|g|b')
        self.bands = []
        # no channels selected
        if len(bands) < 1:
            raise ValueError(f'bands must be specified. Options are {", ".join(all_bands)}, or all')
        # all channels selected
        elif len(bands) == 1:
            if bands[0].lower() == 'all':
                self.bands = all_bands
            elif bands[0].lower() == 'shared':
                self.bands = ['red', 'green', 'blue', 'nir', 'swir16', 'swir22']
            elif bands[0] == 'r|g|b':
                self.bands.append('r|g|b')
        else:
            for band in bands:
                if band in all_bands:
                    self.bands.append(band)
        self.num_channels = len(self.bands)
        self.bands = "|".join(self.bands)

        # define augmentations
        print('build augs')
        additional_targets = dict()
        self.num_images = num_images

        for i in range(self.num_images):
            additional_targets['image{}'.format(1 + i)] = 'image'
            additional_targets['seg{}'.format(i + 1)] = 'mask'

        self.transforms = A.NoOp()
        # if mode == 'train':
        #     self.transforms = A.Compose([A.OneOf([
        #                     A.MotionBlur(p=.5),
        #                     A.Blur(blur_limit=7, p=1),
        #                 ], p=.9),
        #                 A.GaussNoise(var_limit=.002),
        #                 A.RandomBrightnessContrast(brightness_limit=.3, contrast_limit=.3, brightness_by_max=False, always_apply=True)
        #             ],
        #             additional_targets=additional_targets)
        # else:
        #     ### deterministic transforms for test mode
        #     self.transforms = A.Compose([
        #                     A.Blur(blur_limit=[4, 4], p=1),
        #                     A.RandomBrightnessContrast(brightness_limit=[.2, .2], contrast_limit=[.2, .2], brightness_by_max=False, always_apply=True)
        #             ],
        #             additional_targets=additional_targets)

        self.mode = mode
        self.segmentation = segmentation
        self.patch_size = patch_size
        self.bas = bas
        if self.bas:
            self.positive_indices = [0, 1, 3]
            self.ignore_indices = [2, 6]
        else:
            self.positive_indices = [0, 1, 2, 3]
            self.ignore_indices = [6]
        print('finished dataset init')
        """
        if temporal_mode == 'cat':
            self.mask_generator = RandomMaskingGenerator((int(patch_size * num_images / (mask_time_width * mask_patch_size)), int(patch_size / mask_patch_size)), mask_pct)
        else:
            self.mask_generator = RandomMaskingGenerator((int(patch_size * num_images / (mask_time_width * mask_patch_size)), int(patch_size / mask_patch_size)), mask_pct)
    """
        self.temporal_mode = temporal_mode

    def __len__(self):
        return len(self.patches)

    def __getitem__(self, idx):
        tr : dict = self.patches[idx]
        tr['channels'] = self.bands
        # vidid = tr['vidid']
        gids : list[int] = tr['gids']

        sample = self.sampler.load_sample(tr, nodata='float')
        images : np.ndarray = sample['im']
        std = np.nanstd(images)
        mean = np.nanmean(images)
        if std != 0:
            images = np.nan_to_num((images - mean) / std)
        else:
            images = np.zeros_like(images)

        if self.temporal_mode=='cat':
            images = torch.cat([torch.tensor(x) for x in images], dim=0).permute(2, 0, 1)
        else:
            images = images.transpose(0, 3, 1, 2)

        return (images,)

if __name__=='__main__':
    #f = open("eur_train.txt", "w")
    if True:
        torch.cuda.empty_cache()
        #logger = WandbLogger(project="Watch-MAE")
        train_data_path = "/storage1/fs1/jacobsn/Active/proj_smart/smart_dvc/smart_data_dvc/Aligned-Drop4-2022-08-08-TA1-S2-L8-ACC/data_train.kwcoco.json"
        val_data_path = "/storage1/fs1/jacobsn/Active/proj_smart/smart_dvc/smart_data_dvc/Aligned-Drop4-2022-08-08-TA1-S2-L8-ACC/data_vali.kwcoco.json"
        trainset = WatchDataset(train_data_path, sensor=['S8', 'L8'], bands=['shared'],
                    segmentation=False, patch_size=128, mask_patch_size=16, num_images=4,
                    mode='train', mask_pct=0.5,
                    temporal_mode='stack',
                    mask_time_width=2)
        valset = WatchDataset(val_data_path, sensor=['S8', 'L8'], bands=['shared'],
                    segmentation=False, patch_size=128, mask_patch_size=16, num_images=4,
                    mode='train', mask_pct=0.5,
                    temporal_mode='stack',
                    mask_time_width=2)

        print("Dataset load finished ...")
        writer1 = DatasetWriter('./watchtrain.beton', {'train': NDArrayField(shape=(4, 6, 128, 128,), dtype=np.dtype('float32'))}, num_workers=32)
        writer2 = DatasetWriter('./watchtest.beton', {'val': NDArrayField(shape=(4, 6, 128, 128,), dtype=np.dtype('float32'))}, num_workers=32)
        writer1.from_indexed_dataset(trainset)
        writer2.from_indexed_dataset(valset)



"""
Here is an example to write a ffcv train and valid files.
1:28
The you can create ffcv dataloader -

def train_dataloader(self):
        return Loader('./watchtrain.beton',
        batch_size=self.batch_size,
        num_workers=self.num_workers,
        order=OrderOption.SEQUENTIAL,
        os_cache=True,
        distributed=True,
        pipelines={'image': [NDArrayDecoder(), transforms.ToTensor()]}
        )

def val_dataloader(self):
        return Loader('./watchtest.beton',
        batch_size=self.batch_size,
        num_workers=self.num_workers,
        order=OrderOption.SEQUENTIAL,
        os_cache=True,
        distributed=True,
        pipelines={'image': [NDArrayDecoder(), transforms.ToTensor()]}
        )

"""
