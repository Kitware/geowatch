__dev__ = """
mkinit -m watch.tasks.fusion.architectures -w --lazy --noattrs
"""
from watch.tasks.fusion.architectures import transformer
from watch.tasks.fusion.architectures import unet_blur

__all__ = ['transformer', 'unet_blur']
