"""
mkinit -m watch.tasks.fusion.methods -w
"""
from watch.tasks.fusion.methods import channelwise_transformer
from watch.tasks.fusion.methods.channelwise_transformer import (
    MultimodalTransformer)

__all__ = ['MultimodalTransformer', 'channelwise_transformer']
