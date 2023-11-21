"""
python -m geowatch.tasks.fusion


mkinit ~/code/watch/geowatch/tasks/fusion/datamodules/__init__.py --nomods -w
"""

__submodules__ = [
    'kwcoco_video_data',

]
from geowatch.tasks.fusion.datamodules.kwcoco_video_data import (
    KWCocoVideoDataModule,
    KWCocoVideoDataset,)

__all__ = ['KWCocoVideoDataModule', 'KWCocoVideoDataset']
