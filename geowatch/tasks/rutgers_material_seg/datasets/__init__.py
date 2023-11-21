# flake8: noqa

from geowatch.tasks.rutgers_material_seg.datasets import deepglobe
from geowatch.tasks.rutgers_material_seg.datasets import iarpa_dataset

from geowatch.tasks.rutgers_material_seg.datasets.deepglobe import (
    DeepGlobeDataset, IMG_EXTENSIONS, mean_std,)
from geowatch.tasks.rutgers_material_seg.datasets.iarpa_dataset import (
    SequenceDataset, decollate_batch, worker_init_fn,)
from geowatch.tasks.rutgers_material_seg.datasets.sysucd import SYSUCDDataset
from geowatch.tasks.rutgers_material_seg.datasets.s2mcp import S2MCPDataset
from geowatch.tasks.rutgers_material_seg.datasets.s2_self import S2SelfCollectDataset
from geowatch.tasks.rutgers_material_seg.datasets.bigearthnet import BigEarthNetDataset
from geowatch.tasks.rutgers_material_seg.datasets.dynamicearthnet import DynEarthNetDataset
from geowatch.tasks.rutgers_material_seg.datasets.hrscd import HRSCDDataset
from geowatch.tasks.rutgers_material_seg.datasets.inria import InriaDataset
from geowatch.tasks.rutgers_material_seg.datasets.spacenet2 import SpaceNet2Dataset

from torchvision import transforms
from torch.utils.data import DataLoader

datasets = {'deepglobe': DeepGlobeDataset,
            'iarpa': SequenceDataset,
            'sysucd': SYSUCDDataset,
            's2mcp': S2MCPDataset,
            's2self': S2SelfCollectDataset,
            'bigearthnet': BigEarthNetDataset,
            'dynamicearthnet': DynEarthNetDataset,
            'hrscd': HRSCDDataset,
            'spacenet2': SpaceNet2Dataset
            }


def build_dataset(dataset_name: str, root: str, batch_size: int,
                  num_workers: int, split: str, **kwargs) -> DataLoader:
    """Dataset builder

    Parameters
    ----------
    dataset_name : str
        dataset used
    root : str
        directory root in which images and masks are located
    batch_size : int
        batch size
    num_workers : int
        [description]
    split : str
        train, validation, or test data

    Returns
    -------
    DataLoader
        torch loader
    """
    # height, width = int(kwargs["image_size"].split("x")[0]), int(kwargs["image_size"].split("x")[1])  # NOQA
    transformer = transforms.Compose([
                                      # transforms.Resize((height, width)),
                                      # transforms.ColorJitter(),
                                      transforms.ToTensor(),
                                    #   transforms.Normalize(*mean_std)
                                     ])
    # transformer = transforms
    print(f"Building {split} dataset {dataset_name} with root: {root}")
    dataset = datasets[dataset_name](root=root, transforms=transformer, split=split, **kwargs)
    dataloader = DataLoader(dataset,
                            batch_size=batch_size,
                            shuffle=False,
                            num_workers=num_workers,
                            pin_memory=True,
                            drop_last=True)
    print(f"The dataset has length of {len(dataloader)}")
    return dataloader


__all__ = ['DeepGlobeDataset', 'IARPAVideoDataset', 'IMG_EXTENSIONS',
           'decollate_batch', 'deepglobe', 'draw_multispectral_batch',
           'draw_multispectral_item', 'iarpa_dataset', 'mean_std',
           'worker_init_fn', 'mean_std']
