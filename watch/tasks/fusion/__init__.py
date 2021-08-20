"""
mkinit -m watch.tasks.fusion --lazy --noattrs -w
mkinit -m watch.tasks.fusion --noattrs -w
"""

# Hack to supress pytorch-lightning warning

from watch.tasks.fusion import architectures
from watch.tasks.fusion import datamodules
from watch.tasks.fusion import evaluate
from watch.tasks.fusion import fit
from watch.tasks.fusion import methods
from watch.tasks.fusion import predict
from watch.tasks.fusion import utils

__all__ = ['architectures', 'datamodules', 'evaluate', 'fit', 'methods',
           'predict', 'utils']
