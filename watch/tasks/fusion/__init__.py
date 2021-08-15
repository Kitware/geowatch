"""
mkinit -m watch.tasks.fusion --lazy --noattrs -w
mkinit -m watch.tasks.fusion --noattrs -w
"""

# Hack to supress pytorch-lightning warning

from watch.tasks.fusion import datamodules
from watch.tasks.fusion import evaluate
from watch.tasks.fusion import fit
from watch.tasks.fusion import fit_bigvoter
from watch.tasks.fusion import fit_voter
from watch.tasks.fusion import methods
from watch.tasks.fusion import models
from watch.tasks.fusion import predict
from watch.tasks.fusion import utils

__all__ = ['datamodules', 'evaluate', 'fit', 'fit_bigvoter', 'fit_voter',
           'methods', 'models', 'predict', 'utils']
