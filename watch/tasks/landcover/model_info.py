import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Union

import kwcoco
import torch.utils.data
from torch.utils.data import ConcatDataset

from . import detector
from .datasets import L8asWV3Dataset, S2asWV3Dataset, S2Dataset

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


class WV3ModelInfo(ModelInfo):
    """
    This model was trained on 8-band WV3 data with 15 segmentation classes
    """

    def create_dataset(self, coco_dset):
        return ConcatDataset([
            L8asWV3Dataset(coco_dset),
            S2asWV3Dataset(coco_dset)
        ])

    @property
    def model_outputs(self):
        return [
            'rice_field', 'cropland', 'water', 'inland_water', 'river_or_stream',
            'sebkha', 'snow_or_ice_field', 'bare_ground', 'sand_dune', 'built_up',
            'grassland', 'brush', 'forest', 'wetland', 'road'
        ]

    def load_model(self, weights_filename, device):
        assert len(self.model_outputs) == 15
        return detector.load_model(weights_filename,
                                   num_outputs=15,
                                   num_channels=8,
                                   device=device)


class S2ModelInfo(ModelInfo):
    """
    This model was trained on 13-band Sentinel 2 data with 22 segmentation classes
    """

    def create_dataset(self, coco_dset):
        return S2Dataset(coco_dset)

    @property
    def model_outputs(self):
        return [
            'forest_deciduous', 'forest_evergreen', 'brush', 'grassland', 'bare_ground',
            'built_up', 'cropland', 'rice_field', 'marsh', 'swamp',
            'inland_water', 'snow_or_ice_field', 'reef', 'sand_dune', 'sebkha',
            'ocean<10m', 'ocean>10m', 'lake', 'river', 'beach',
            'alluvial_deposits', 'med_low_density_built_up'
        ]

    def load_model(self, weights_filename, device):
        assert len(self.model_outputs) == 22
        return detector.load_model(weights_filename,
                                   num_outputs=22,
                                   num_channels=13,
                                   device=device)


__mapping = {
    'visnav_osm': WV3ModelInfo,
    'visnav_sentinel2': S2ModelInfo
}


def lookup_model_info(weights_filename: Path) -> ModelInfo:
    model_info_class = __mapping.get(weights_filename.stem)
    if not model_info_class:
        raise Exception('unknown weights file')
    return model_info_class()
