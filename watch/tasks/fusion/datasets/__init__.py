"""
python -m watch.tasks.fusion


mkinit ~/code/watch/watch/tasks/fusion/datasets/__init__.py --nomods -w
"""
from watch.tasks.fusion.datasets.common import (AddPositionalEncoding,
                                                VideoDataset,)
from watch.tasks.fusion.datasets.drop0_align_msi_s2_preprocess_split import (
    main,)
from watch.tasks.fusion.datasets.watch_data import (WatchDataModule,)
from watch.tasks.fusion.datasets.onera_2018 import (OneraCD_2018,)
from watch.tasks.fusion.datasets.project_data import (Drop0AlignMSI_S2,
                                                      Drop0Raw_S2,)
from watch.tasks.fusion.datasets.sun_rgbd import (SUN_RGBD, SUN_RGBD_Dataset,)

__all__ = ['AddPositionalEncoding', 'Drop0AlignMSI_S2', 'Drop0Raw_S2',
           'OneraCD_2018', 'SUN_RGBD', 'SUN_RGBD_Dataset', 'VideoDataset',
           'WatchDataModule', 'main']
