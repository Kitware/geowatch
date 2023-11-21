import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Union

import kwcoco
import torch.utils.data

from . import detector
from .datasets import S2Dataset, WVDataset

log = logging.getLogger(__name__)


class ModelInfo(ABC):

    @abstractmethod
    def create_dataset(self, coco_dset: Union[kwcoco.CocoDataset, str]) -> torch.utils.data.Dataset:
        """
        Create a torch dataset compatible with this model using a CocoDataset
        Args:
            coco_dset: CocoDataset of string filepath

        Returns: torch dataset

        """
        pass

    @property
    @abstractmethod
    def model_outputs(self):
        pass

    @abstractmethod
    def load_model(self, weights_filename: Path, device):
        pass


class S2ModelInfo(ModelInfo):
    """
    This model was trained on 13-band Sentinel-2 data with 5
    segmentation classes
    """

    def create_dataset(self, coco_dset):
        return S2Dataset(coco_dset)

    @property
    def model_outputs(self):
        return [
            'water', 'forest', 'field', 'impervious', 'barren',
        ]

    def load_model(self, weights_filename, device):
        assert len(self.model_outputs) == 5
        return detector.load_model(weights_filename,
                                   num_outputs=5,
                                   num_channels=13,
                                   device=device)


class WVModelInfo(ModelInfo):
    """
    This model was trained on 8-band WorldView-3 data with 5
    segmentation classes
    """

    def create_dataset(self, coco_dset):
        return WVDataset(coco_dset)

    @property
    def model_outputs(self):
        return [
            'water', 'forest', 'field', 'impervious', 'barren',
        ]

    def load_model(self, weights_filename, device):
        assert len(self.model_outputs) == 5
        return detector.load_model(weights_filename,
                                   num_outputs=5,
                                   num_channels=8,
                                   device=device)


__mapping = {
    'sentinel2': S2ModelInfo,
    'worldview': WVModelInfo,
}


def lookup_model_info(weights_filename: Path) -> ModelInfo:
    model_info_class = __mapping.get(weights_filename.stem)
    if not model_info_class:
        raise Exception('unknown weights file')
    return model_info_class()
