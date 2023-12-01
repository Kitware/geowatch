"""
A submodule where each file contains information and functionality specific to
a particular GEO sensor.
"""

__dev__ = """
mkinit ~/code/watch/geowatch/gis/sensors/__init__.py -w
"""
from geowatch.gis.sensors import sentinel2

from geowatch.gis.sensors.sentinel2 import (sentinel2_grid,)

__all__ = ['sentinel2', 'sentinel2_grid']
