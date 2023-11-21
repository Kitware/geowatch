__dev__ = """
mkinit -m geowatch.tasks.fusion.architectures -w --lazy --noattrs
"""
from geowatch.tasks.fusion.architectures import transformer
from geowatch.tasks.fusion.architectures import unet_blur

__all__ = ['transformer', 'unet_blur']
