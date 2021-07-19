__dev__ = """
mkinit -m watch.tasks.fusion.models -w --lazy --noattrs
"""
from watch.tasks.fusion.models import transformer
from watch.tasks.fusion.models import unet_blur

__all__ = ['transformer', 'unet_blur']
