"""
mkinit -m watch.tasks.fusion.methods -w
"""
from watch.tasks.fusion.methods import channelwise_transformer
from watch.tasks.fusion.methods import common

from watch.tasks.fusion.methods.channelwise_transformer import (
    MultimodalTransformerDirectCD, MultimodalTransformerDotProdCD,
    MultimodalTransformerSegmentation)
from watch.tasks.fusion.methods.common import (ChangeDetectorBase,
                                               SemanticSegmentationBase)

__all__ = ['ChangeDetectorBase', 'MultimodalTransformerDirectCD',
           'MultimodalTransformerDotProdCD',
           'MultimodalTransformerSegmentation', 'SemanticSegmentationBase',
           'channelwise_transformer', 'common']
