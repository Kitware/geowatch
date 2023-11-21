"""
Defines a torch Dataset and lightning DataModule for kwcoco video data.

The parameters to each are handled by scriptconfig objects, which prevents us
from needing to specify what the available options are in multiple places.
"""
from geowatch.tasks.fusion.datamodules.kwcoco_dataset import *  # NOQA
from geowatch.tasks.fusion.datamodules.kwcoco_datamodule import *  # NOQA
