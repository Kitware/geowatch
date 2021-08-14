"""
mkinit watch.tasks.fusion.lightning_extensions --noattr -w
"""
from watch.tasks.fusion.lightning_extensions import demo
from watch.tasks.fusion.lightning_extensions import draw_batch
from watch.tasks.fusion.lightning_extensions import kwtrainer
from watch.tasks.fusion.lightning_extensions import tensorboard_plotter
from watch.tasks.fusion.lightning_extensions import trainer

__all__ = ['demo', 'draw_batch', 'kwtrainer', 'tensorboard_plotter', 'trainer']
