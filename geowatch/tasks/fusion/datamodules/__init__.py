"""
python -m watch.tasks.fusion


mkinit ~/code/watch/watch/tasks/fusion/datamodules/__init__.py --nomods -w
"""

__submodules__ = [
    'kwcoco_video_data',

]
from watch.tasks.fusion.datamodules.kwcoco_video_data import (
    KWCocoVideoDataModule,
    KWCocoVideoDataset,)

__all__ = ['KWCocoVideoDataModule', 'KWCocoVideoDataset']
