"""
mkinit -m watch.tasks.fusion.methods -w
"""
from watch.tasks.fusion.methods import channelwise_transformer
from watch.tasks.fusion.methods.channelwise_transformer import (
    MultimodalTransformer)
from watch.tasks.fusion.methods import heterogeneous
from watch.tasks.fusion.methods.heterogeneous import (
    HeterogeneousModel)
from watch.tasks.fusion.methods import noop_model
from watch.tasks.fusion.methods.noop_model import (
    NoopModel)

__all__ = ['MultimodalTransformer', 'channelwise_transformer', 'heterogeneous', 'HeterogeneousModel', 'noop_model', 'NoopModel']
