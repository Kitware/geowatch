from watch.tasks.fusion.methods import baseline
from watch.tasks.fusion.methods import channelwise_transformer
from watch.tasks.fusion.methods import common
from watch.tasks.fusion.methods import transformer
from watch.tasks.fusion.methods import voting

from watch.tasks.fusion.methods.baseline import (UNetChangeDetector,)
from watch.tasks.fusion.methods.channelwise_transformer import (
    MultimodalTransformerDirectCD, MultimodalTransformerDotProdCD,
    MultimodalTransformerSegmentation,)
from watch.tasks.fusion.methods.common import (ChangeDetectorBase,
                                               SemanticSegmentationBase,)
from watch.tasks.fusion.methods.transformer import (TransformerChangeDetector,)
from watch.tasks.fusion.methods.voting import (End2EndVotingModel,
                                               VotingModel,)

__all__ = ['ChangeDetectorBase', 'End2EndVotingModel',
           'MultimodalTransformerDirectCD', 'MultimodalTransformerDotProdCD',
           'MultimodalTransformerSegmentation', 'SemanticSegmentationBase',
           'TransformerChangeDetector', 'UNetChangeDetector', 'VotingModel',
           'baseline', 'channelwise_transformer', 'common', 'transformer',
           'voting']
