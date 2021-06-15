from watch.tasks.rutgers_material_seg.datasets import deepglobe
from watch.tasks.rutgers_material_seg.datasets import iarpa_dataset

from watch.tasks.rutgers_material_seg.datasets.deepglobe import (
    DeepGlobeDataset, IMG_EXTENSIONS, mean_std,)
from watch.tasks.rutgers_material_seg.datasets.iarpa_dataset import (
    SequenceDataset, decollate_batch, worker_init_fn,)

from torchvision import transforms
from torch.utils.data import DataLoader

datasets = {
    'deepglobe': DeepGlobeDataset,
    'iarpa': SequenceDataset,
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
    height, width = int(kwargs["image_size"].split("x")[0]), int(kwargs["image_size"].split("x")[1])
    transformer = transforms.Compose([
        transforms.Resize((height, width)),
        # transforms.ColorJitter(),
        transforms.ToTensor(),
        transforms.Normalize(*mean_std),
    ])
    print(f"Building {split} dataset {dataset_name} with root: {root}")
    dataset = datasets[dataset_name](root=root, transforms=transformer, split=split)
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
