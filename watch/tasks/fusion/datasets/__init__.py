"""
python -m watch.tasks.fusion

mkinit ~/code/watch/watch/tasks/fusion/datasets/__init__.py --nomods -w
"""
from watch.tasks.fusion.datasets import common
from watch.tasks.fusion.datasets import drop0_align_msi_s2_preprocess_split
from watch.tasks.fusion.datasets import kwcoco_video
from watch.tasks.fusion.datasets import onera_2018
from watch.tasks.fusion.datasets import project_data
from watch.tasks.fusion.datasets import sun_rgbd

__all__ = ['common', 'drop0_align_msi_s2_preprocess_split', 'kwcoco_video',
           'onera_2018', 'project_data', 'sun_rgbd']
