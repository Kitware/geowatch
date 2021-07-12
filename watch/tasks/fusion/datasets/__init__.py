"""
mkinit ~/code/watch/watch/tasks/fusion/datasets/__init__.py --nomods -w
"""
from watch.tasks.fusion.datasets.common import (AddPositionalEncoding,
                                                VideoDataset,)
from watch.tasks.fusion.datasets.kwcoco_video import (WatchDataModule,)
from watch.tasks.fusion.datasets.onera_2018 import (OneraCD_2018,)
from watch.tasks.fusion.datasets.project_data import (Drop0AlignMSI_S2,
                                                      Drop0Raw_S2,)

__all__ = ['AddPositionalEncoding', 'Drop0AlignMSI_S2', 'Drop0Raw_S2',
           'OneraCD_2018', 'VideoDataset', 'WatchDataModule']
