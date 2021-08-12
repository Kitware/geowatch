"""
python -m watch.tasks.fusion


mkinit ~/code/watch/watch/tasks/fusion/datamodules/__init__.py --nomods -w
"""

__submodules__ = [
    'common',
    'sun_rgbd',
    'watch_data',
    'project_data',

]
from watch.tasks.fusion.datamodules.common import (AddPositionalEncoding,
                                                   VideoDataset,)
from watch.tasks.fusion.datamodules.sun_rgbd import (SUN_RGBD,
                                                     SUN_RGBD_Dataset,)
from watch.tasks.fusion.datamodules.watch_data import (AddPositionalEncoding,
                                                       WatchDataModule,
                                                       WatchVideoDataset,
                                                       category_tree_ensure_color,
                                                       coco_channel_profiles,
                                                       morphology, profile,
                                                       simple_video_sample_grid,)
from watch.tasks.fusion.datamodules.project_data import (Drop0AlignMSI_S2,
    Drop0Raw_S2,)

__all__ = ['AddPositionalEncoding', 'Drop0AlignMSI_S2', 'Drop0Raw_S2',
           'SUN_RGBD', 'SUN_RGBD_Dataset', 'VideoDataset', 'WatchDataModule',
           'WatchVideoDataset', 'category_tree_ensure_color',
           'coco_channel_profiles', 'morphology', 'profile',
           'simple_video_sample_grid']
