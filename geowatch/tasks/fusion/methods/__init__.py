"""
mkinit -m geowatch.tasks.fusion.methods -w
"""
from geowatch.tasks.fusion.methods import channelwise_transformer
from geowatch.tasks.fusion.methods.channelwise_transformer import (
    MultimodalTransformer)
from geowatch.tasks.fusion.methods import heterogeneous
from geowatch.tasks.fusion.methods.heterogeneous import (
    HeterogeneousModel)
from geowatch.tasks.fusion.methods import unet_baseline
from geowatch.tasks.fusion.methods.unet_baseline import (
    UNetBaseline)
from geowatch.tasks.fusion.methods import noop_model
from geowatch.tasks.fusion.methods.noop_model import (
    NoopModel)

__all__ = ['MultimodalTransformer', 'channelwise_transformer', 'heterogeneous', 'HeterogeneousModel', 'unet_baseline', 'UNetBaseline', 'noop_model', 'NoopModel']
