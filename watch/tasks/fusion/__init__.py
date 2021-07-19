"""
mkinit -m watch.tasks.fusion --lazy --noattrs -w
"""


from watch.tasks.fusion import datasets
from watch.tasks.fusion import evaluate
from watch.tasks.fusion import fit
from watch.tasks.fusion import fit_bigvoter
from watch.tasks.fusion import fit_voter
from watch.tasks.fusion import fusion_transformer_v1
from watch.tasks.fusion import methods
from watch.tasks.fusion import models
from watch.tasks.fusion import predict
from watch.tasks.fusion import utils

__all__ = ['datasets', 'evaluate', 'fit', 'fit_bigvoter', 'fit_voter',
           'fusion_transformer_v1', 'methods', 'models', 'predict', 'utils']
