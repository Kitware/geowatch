import os
from copy import deepcopy
from contextlib import contextmanager
from tempfile import NamedTemporaryFile
from lxml import etree

import gdal

from rasterio import Affine, MemoryFile
from rasterio.enums import Resampling


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

    if read:

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
    if os.path.abspath(old_path) == os.path.abspath(new_path):
        return

    path_diff = os.path.relpath(os.path.dirname(os.path.abspath(old_path)),
                                start=os.path.dirname(
                                    os.path.abspath(new_path)))

    tree = deepcopy(etree.parse(old_path))
    for elem in tree.iterfind('.//SourceFilename'):
        if elem.get('relativeToVRT') == '1':
            elem.text = os.path.join(path_diff, elem.text)
        else:
            if not os.path.isabs(elem.text):
                raise ValueError(f'''VRT file:
                    {old_path}
                cannot be rerooted because it contains path: 
                    {elem.text}
                relative to an unknown location [the original calling location].
                To produce a rerootable VRT, call gdal.BuildVRT() with out_path relative to in_paths.'''
                                 )
            if not os.path.isfile(elem.text):
                raise ValueError(f'''VRT file:
                    {old_path}
                references an nonexistent path: 
                    {elem.text}''')

    with open(new_path, 'wb') as f:
        tree.write(f, encoding='utf-8')

    if not keep_old:
        os.remove(old_path)


def make_vrt(in_paths, out_path, mode, relative_to_path=None, **kwargs):
    '''
    Stack multiple band files in the same directory into a single VRT

    Args:
        in_paths: list(path)
        out_path: path to save to; standard is '*.vrt'. If None, a path will be generated.
        mode:
            'stacked': Stack multiple band files covering the same area
            'mosaicked': Mosaic/merge scenes with overlapping areas. Content will be taken from the first in_path without nodata.
        relative_to_path: if this function is being called from another process, pass in the cwd of the calling process, to trick gdal into creating a rerootable VRT
        kwargs: passed to gdal.BuildVRTOptions [1,2]

    Returns:
        path to VRT

    References:
        [1] https://gdal.org/programs/gdalbuildvrt.html
        [2] https://gdal.org/python/osgeo.gdal-module.html#BuildVRTOptions
    '''

    if mode == 'stacked':
        kwargs['separate'] = True
    elif mode == 'mosaicked':
        kwargs['separate'] = False
    else:
        raise ValueError(f'mode: {mode} should be "stacked" or "mosaicked"')

    # set sensible defaults
    if 'resolution' not in kwargs:
        kwargs['resolution'] = 'highest'
    if 'resampleAlg' not in kwargs:
        kwargs['resampleAlg'] = 'bilinear'

    opts = gdal.BuildVRTOptions(**kwargs)

    if len(in_paths) > 1:
        common = os.path.commonpath(in_paths)
    else:
        common = os.path.dirname(in_paths[0])

    if relative_to_path is None:
        relative_to_path = os.path.dirname(os.path.abspath(__file__))

    # validate out_path
    if out_path is not None:
        out_path = os.path.abspath(out_path)
        if os.path.splitext(out_path)[1]:  # is a file
            os.makedirs(os.path.dirname(out_path), exist_ok=True)
        elif os.path.isdir(out_path):  # is a dir
            raise ValueError(f'{out_path} is an existing directory.')

    # generate an unused name
    with NamedTemporaryFile(dir=common,
                            suffix='.vrt',
                            mode='r+',
                            delete=(out_path is not None)) as f:

        # First, create VRT in a place where it can definitely see the input files.
        # Use a relative instead of absolute path to ensure that
        # <SourceFilename> refs are relative, and therefore the VRT is rerootable
        vrt = gdal.BuildVRT(os.path.relpath(f.name, start=relative_to_path),
                            in_paths,
                            options=opts)
        del vrt  # write to disk

        # then, move it to the desired location
        if out_path is None:
            out_path = f.name
        elif os.path.isfile(out_path):
            print(f'warning: {out_path} already exists! Removing...')
            os.remove(out_path)
        reroot_vrt(f.name, out_path, keep_old=True)

    return out_path
