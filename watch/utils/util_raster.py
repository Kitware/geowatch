import numpy as np
import os
from contextlib import contextmanager

import gdal

import rasterio
from rasterio import Affine, MemoryFile
from rasterio.enums import Resampling

from lxml import etree
from copy import deepcopy


# use context manager so DatasetReader and MemoryFile get cleaned up automatically
@contextmanager
def resample_raster(raster, scale=2, read=True):
    '''
    Context manager to rescale a raster on the fly using rasterio
    
    This changes the number of pixels in the raster while maintaining its geographic bounds, that is, it changes the raster's GSD.

    Args:
        raster: a DatasetReader (the object returned by rasterio.open)
        scale: factor to upscale the resolution, aka downscale the GSD, by
        read: if True, read and return the resampled data (an expensive operation)
            else, return the resampled dataset's .profile attribute (metadata)

    Example:
        >>> path = 'path/to/band.jp2'
        >>> current_gsd_meters = 60
        >>> desired_gsd_meters = 10
        >>> scale = current_gsd_meters / desired_gsd_meters
        >>> with rasterio.open(path) as old:
        >>>     assert old.profile['width'] == 1830
        >>>     with resample_raster(old, scale=scale, read=False) as new_profile:
        >>>         assert new_profile['width'] == 10980
        >>>         assert new_profile['crs'] == old.profile['crs']
        >>>     with resample_raster(old, scale=scale) as new:
        >>>         assert new.profile['width'] == 10980
        >>>         assert new.profile['crs'] == old.profile['crs']
        >>>         # do other stuff with new

    References:
        https://gis.stackexchange.com/a/329439
        https://rasterio.readthedocs.io/en/latest/topics/reading.html
        https://rasterio.readthedocs.io/en/latest/topics/profiles.html
    '''
    t = raster.transform

    # rescale the metadata
    transform = Affine(t.a / scale, t.b, t.c, t.d, t.e / scale, t.f)
    # TODO get rid of floating point weirdness here
    height = int(raster.height * scale)
    width = int(raster.width * scale)

    profile = raster.profile
    profile.update(transform=transform,
                   driver='GTiff',
                   height=height,
                   width=width)

    if return_data:

        data = raster.read(  # Note changed order of indexes, arrays are band, row, col order not row, col, band
            out_shape=(raster.count, height, width),
            resampling=Resampling.bilinear,
        )

        with MemoryFile() as memfile:
            with memfile.open(**profile) as dataset:  # Open as DatasetWriter
                dataset.write(data)
                del data

            with memfile.open() as dataset:  # Reopen as DatasetReader
                yield dataset  # Note yield not return

    else:

        yield profile


@contextmanager
def gdal_open(path):
    '''
    A simple context manager for friendlier gdal use.

    Example:
        >>> # standard use:
        >>> dataset = gdal.Open(path)
        >>> print(dataset.getDescription())  # [do stuff]
        >>> del dataset  # or 'dataset = None'
        >>> 
        >>> # equivalent:
        >>> with gdal_open(path) as dataset:
        >>>     print(dataset.GetDescription())  # [do stuff]

    '''
    try:
        f = gdal.Open(path)
        yield f
    finally:
        # gdal.GDALClose(f)  # not implemented in this version of gdal?
        del f  # this is ugly, but it works...
        # gdal.Unlink(path)  # THIS DELETES THE FILE


def reroot_vrt(old_path, new_path, keep_old=True):
    '''
    Copy a VRT file, fixing relative paths to its component images

    Example:
        >>> imgs_dpath = 'long/path/to/imgs/'
        >>> # VRT must be created in the imgs' subtree
        >>> gdal.BuildVRT(imgs_dpath + 'imgs.vrt', sorted(glob(imgs_dpath + '*.tif')))
        >>> # now move it somewhere more convenient
        >>> reroot_vrt(imgs_dpath + 'imgs.vrt', 'imgs_vrt', keep_old=False)
    '''
    path_diff = os.path.join(
        os.path.curdir,
        os.path.relpath(os.path.dirname(old_path),
                        start=os.path.dirname(new_path)))

    tree = deepcopy(etree.parse(old_path))
    for elem in tree.iterfind('.//SourceFilename'):
        assert elem.get('relativeToVRT') == '1', old_path
        elem.text = os.path.join(path_diff, elem.text)

    with open(new_path, 'wb') as f:
        tree.write(f, encoding='utf-8')

    if not keep_old:
        os.remove(old_path)

