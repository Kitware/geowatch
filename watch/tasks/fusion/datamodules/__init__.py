"""
python -m watch.tasks.fusion


mkinit ~/code/watch/watch/tasks/fusion/datamodules/__init__.py --nomods -w
"""

__submodules__ = [
    'kwcoco_data',

]
from watch.tasks.fusion.datamodules.kwcoco_data import (
    KWCocoDataModule,
    WatchVideoDataset,)

__all__ = ['KWCocoDataModule', 'WatchVideoDataset']
