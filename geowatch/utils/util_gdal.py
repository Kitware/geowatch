"""
This module was moved to kwgis.utils.util_gdal, prefer using that directly.

SeeAlso
    util_raster.py

References:
    https://gdal.org/programs/gdalwarp.html#cmdoption-gdalwarp-multi
    https://gis.stackexchange.com/a/241810
    https://trac.osgeo.org/gdal/wiki/UserDocs/GdalWarp#WillincreasingRAMincreasethespeedofgdalwarp
    https://github.com/OpenDroneMap/ODM/issues/778


TODO:
    TODO test this and see if it's safe to add:
        --config GDAL_PAM_ENABLED NO
    Removes .aux.xml sidecar files and puts them in the geotiff metadata
    ex. histogram from fmask
    https://stackoverflow.com/a/51075774
    https://trac.osgeo.org/gdal/wiki/ConfigOptions#GDAL_PAM_ENABLED
    https://gdal.org/drivers/raster/gtiff.html#georeferencing
"""
import ubelt as ub


def __getattr__(key):
    ub.schedule_deprecation(
        'geowatch', key, 'attribute of util_gdal',
        migration='use kwgis.util_gdal directly instead',
        deprecate='0.18.3', error='0.20.0', remove='0.21.0'
    )
    from kwgis.utils import util_gdal
    return getattr(util_gdal, key)
