import os
from copy import deepcopy
from tempfile import NamedTemporaryFile
from dataclasses import dataclass
from typing import Union
from contextlib import ExitStack
import numpy as np
from tempenv import TemporaryEnvironment
from lxml import etree
import shapely as shp
import shapely.geometry

from osgeo import gdal, osr

import rasterio
from rasterio import Affine, MemoryFile
from rasterio.enums import Resampling

from watch.gis.spatial_reference import utm_epsg_from_latlon


@dataclass
class ResampledRaster(ExitStack):
    '''
    Context manager to rescale a raster on the fly using rasterio
    
    This changes the number of pixels in the raster while maintaining its geographic bounds, that is, it changes the raster's GSD.

    Args:
        raster: a DatasetReader (the object returned by rasterio.open) or path to a dataset
        scale: factor to upscale the resolution, aka downscale the GSD, by
        read: if True, read and return the resampled data (an expensive operation if scale>1)
            else, return the resampled dataset's .profile attribute (metadata)
        resampling: resampling algorithm, from rasterio.enums.Resampling [1]

    Example:
        >>> # xdoctest: +REQUIRES(--network)
        >>> from watch.utils.util_raster import *
        >>> from watch.demo.landsat_demodata import grab_landsat_product
        >>> path = grab_landsat_product()['bands'][0]
        >>> 
        >>> current_gsd_meters = 60
        >>> desired_gsd_meters = 10
        >>> scale = current_gsd_meters / desired_gsd_meters
        >>> 
        >>> with rasterio.open(path) as f:
        >>>     old_profile = f.profile
        >>> 
        >>> # can instantiate this class in a with-block
        >>> with ResampledRaster(path, scale=scale, read=False) as f:
        >>>     pass
        >>> 
        >>> # or have it stick around and change the resampling on the fly
        >>> resampled = ResampledRaster(path, scale=scale, read=False)
        >>> 
        >>> # the computation only happens when you invoke 'with'
        >>> with resampled as new_profile:
        >>>     assert new_profile['width'] == int(old_profile['width'] * scale)
        >>>     assert new_profile['crs'] == old_profile['crs']
        >>> 
        >>> resampled.scale = scale / 2
        >>> resampled.read = True
        >>> 
        >>> with resampled as new:
        >>>     assert new.profile['width'] == int(old_profile['width'] * scale / 2)
        >>>     assert new.profile['crs'] == old_profile['crs']
        >>>     # do other stuff with new

    References:
        https://gis.stackexchange.com/a/329439
        https://rasterio.readthedocs.io/en/latest/topics/reading.html
        https://rasterio.readthedocs.io/en/latest/topics/profiles.html
        [1] https://rasterio.readthedocs.io/en/latest/api/rasterio.enums.html#rasterio.enums.Resampling
    '''
    raster: Union[str, rasterio.DatasetReader]
    scale: float = 2
    read: bool = True
    resampling: Resampling = Resampling.bilinear

    def __post_init__(self):
        super().__init__()

    def __enter__(self):
        if not isinstance(self.raster, rasterio.DatasetReader) or self.raster.closed:
            # workaround for
            # https://rasterio.readthedocs.io/en/latest/faq.html#why-can-t-rasterio-find-proj-db-rasterio-from-pypi-versions-1-2-0
            with TemporaryEnvironment({'PROJ_LIB': None}):
                self.raster = rasterio.open(self.raster)

        t = self.raster.transform

        # rescale the metadata
        transform = Affine(t.a / self.scale, t.b, t.c, t.d, t.e / self.scale, t.f)
        height = int(np.ceil(self.raster.height * self.scale))
        width = int(np.ceil(self.raster.width * self.scale))

        profile = self.raster.profile
        profile.update(transform=transform,
                       driver='GTiff',
                       height=height,
                       width=width)

        if self.read:

            data = self.raster.read(  # Note changed order of indexes, arrays are band, row, col order not row, col, band
                out_shape=(self.raster.count, height, width),
                resampling=self.resampling,
            )

            # enter_context is from contextlib.ExitStack, which takes care of closing these
            memfile = self.enter_context(MemoryFile())
            with memfile.open(**profile) as dataset:  # Open as DatasetWriter
                dataset.write(data)
                del data

            dataset = self.enter_context(memfile.open())  # Reopen as DatasetReader
            return dataset

        else:

            return profile

    def __exit__(self, *exc):
        pass


@dataclass
class GdalOpen:
    '''
    A simple context manager for friendlier gdal use.

    Example:
        >>> # xdoctest: +REQUIRES(--network)
        >>> from watch.utils.util_raster import *
        >>> from watch.demo.landsat_demodata import grab_landsat_product
        >>> path = grab_landsat_product()['bands'][0]
        >>> 
        >>> # standard use:
        >>> dataset = gdal.Open(path)
        >>> print(dataset.GetDescription())  # do stuff
        >>> del dataset  # or 'dataset = None'
        >>> 
        >>> # equivalent:
        >>> with GdalOpen(path) as dataset:
        >>>     print(dataset.GetDescription())  # do stuff

    '''
    path: str

    def __enter__(self):
        self.f = gdal.Open(self.path)
        return self.f
    
    def __exit__(self, *exc):
        # gdal.GDALClose(f)  # not implemented in this version of gdal?
        del self.f  # this is ugly, but it works...
        # gdal.Unlink(path)  # THIS DELETES THE FILE


def reroot_vrt(old_path, new_path, keep_old=True):
    '''
    Copy a VRT file, fixing relative paths to its component images

    Example:
        >>> # xdoctest: +REQUIRES(--network)
        >>> import os
        >>> from osgeo import gdal
        >>> from watch.utils.util_raster import *
        >>> from watch.demo.landsat_demodata import grab_landsat_product
        >>> bands = grab_landsat_product()['bands']
        >>> 
        >>> # VRT must be created in the imgs' subtree
        >>> tmp_path = os.path.join(os.path.dirname(bands[0]), 'all_bands.vrt')
        >>> # (consider using the wrapper util_raster.make_vrt instead of this)
        >>> gdal.BuildVRT(tmp_path, sorted(bands))
        >>> # now move it somewhere more convenient
        >>> reroot_vrt(tmp_path, './bands.vrt', keep_old=False)
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

    Example:
        >>> # xdoctest: +REQUIRES(--network)
        >>> import os
        >>> from osgeo import gdal
        >>> from watch.utils.util_raster import *
        >>> from watch.demo.landsat_demodata import grab_landsat_product
        >>> bands = grab_landsat_product()['bands']
        >>> 
        >>> # stack bands from a scene
        >>> make_vrt(sorted(bands), './bands1.vrt', mode='stacked', relative_to_path=os.getcwd())
        >>> # pretend this is a different scene
        >>> make_vrt(sorted(bands), './bands2.vrt', mode='stacked', relative_to_path=os.getcwd())
        >>> # now, if they overlap, mosaic/merge them
        >>> make_vrt(['./bands1.vrt', './bands2.vrt'], 'full_scene.vrt', mode='mosaicked', relative_to_path=os.getcwd())
        >>> with GdalOpen('full_scene.vrt') as f:
        >>>     print(f.GetDescription())
        >>> 
        >>> # clean up
        >>> os.remove('./bands1.vrt')
        >>> os.remove('./bands2.vrt')
        >>> os.remove('./full_scene.vrt')

    References:
        [1] https://gdal.org/programs/gdalbuildvrt.html
        [2] https://gdal.org/python/osgeo.gdal-module.html#BuildVRTOptions
    '''

    if mode == 'stacked':
        kwargs['separate'] = True
    elif mode == 'mosaicked':
        kwargs['separate'] = False
        kwargs['srcNodata']= 0 # this ensures nodata doesn't overwrite data
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


def scenes_to_vrt(scenes, vrt_root, relative_to_path):
    '''
    Search for band files from compatible scenes and stack them in a single mosaicked VRT

    A simple wrapper around watch.utils.util_raster.make_vrt that performs both
    the 'stacked' and 'mosaicked' modes

    Args:
        scenes: list(scene), where scene := list(path) [of band files]
        vrt_root: root dir to save VRT under

    Returns:
        path to the VRT

    Example:
        >>> # xdoctest: +REQUIRES(--network)
        >>> import os
        >>> from osgeo import gdal
        >>> from watch.utils.util_raster import *
        >>> from watch.demo.landsat_demodata import grab_landsat_product
        >>> bands = grab_landsat_product()['bands']
        >>> 
        >>> # pretend there are more scenes here
        >>> out_path = scenes_to_vrt([sorted(bands)] , vrt_root='.', relative_to_path=os.getcwd())
        >>> with GdalOpen(out_path) as f:
        >>>     print(f.GetDescription())
        >>> 
        >>> # clean up
        >>> os.remove(out_path)
    '''
    # first make VRTs for individual tiles
    # TODO use https://rasterio.readthedocs.io/en/latest/topics/memory-files.html
    # for these intermediate files?
    tmp_vrts = [
        make_vrt(
            scene,
            os.path.join(vrt_root, f'{hash(scene[0])}.vrt'),
            mode='stacked',
            relative_to_path=relative_to_path
        ) for scene in scenes
    ]

    # then mosaic them
    final_vrt = make_vrt(
        tmp_vrts,
        os.path.join(vrt_root, f'{hash(scenes[0][0] + "final")}.vrt'),
        mode='mosaicked',
        relative_to_path=relative_to_path)

    if 0:  # can't do this because final_vrt still references them
        for t in tmp_vrts:
            os.remove(t)

    return final_vrt


def reproject_crop(in_path, aoi, code=None, out_path=None, vrt_root=None):
    '''
    Crop an image to an AOI and reproject to its UTM CRS (or another CRS)

    Unfortunately, this cannot be done in a single step in scenes_to_vrt
    because gdal.BuildVRT does not support warping between CRS.
    Cropping alone could be done in scenes_to_vrt. Note gdal.BuildVRTOptions has
    an outputBounds(=-te) kwarg for cropping, but not an equivalent of -te_srs.

    This means another intermediate file is necessary for each warp operation.

    TODO check for this quantization error: https://gis.stackexchange.com/q/139906

    Args:
        in_path: A georeferenced image. GTiff, VRT, etc.
        aoi: A geojson Feature in epsg:4326 CRS to crop to.
        code: EPSG code [1] of the CRS to convert to.
            if None, use the UTM CRS containing aoi.
        out_path: Name of output file to write to. If None, create a VRT file.
        vrt_root: Root directory for VRT output. If None, same dir as input.

    Returns:
        Path to a new VRT or out_path

    References:
        [1] http://epsg.io/

    Example:
        >>> # xdoctest: +REQUIRES(--network)
        >>> import os
        >>> from osgeo import gdal
        >>> from watch.utils.util_raster import *
        >>> from watch.demo.landsat_demodata import grab_landsat_product
        >>> band1 = grab_landsat_product()['bands'][0]
        >>> 
        >>> # pick the AOI from the drop0 KR site
        >>> # (this doesn't actually intersect the demodata)
        >>> top, left = (128.6643, 37.6601)
        >>> bottom, right = (128.6749, 37.6639)
        >>> geojson_bbox = {
        >>>     "type":
        >>>     "Polygon",
        >>>     "coordinates": [[[top, left], [top, right], [bottom, right],
        >>>                      [bottom, left], [top, left]]]
        >>> }
        >>> 
        >>> out_path = reproject_crop(band1, geojson_bbox)
        >>> 
        >>> # clean up
        >>> os.remove(out_path)
    '''
    if out_path is None:
        root, name = os.path.split(in_path)
        if vrt_root is None:
            vrt_root = root
        os.makedirs(vrt_root, exist_ok=True)
        out_path = os.path.join(vrt_root, f'{hash(name + "warp")}.vrt')
        if os.path.isfile(out_path):
            print(f'Warning: {out_path} already exists! Removing...')
            os.remove(out_path)

    if code is None:
        # find the UTM zone(s) of the AOI
        codes = [
            utm_epsg_from_latlon(lat, lon) for lon, lat in aoi['coordinates'][0]
        ]
        u, counts = np.unique(codes, return_counts=True)
        if len(u) > 1:
            print(f'Warning: AOI crosses UTM zones {u}. Taking majority vote...')
        code = int(u[np.argsort(-counts)][0])

    dst_crs = osr.SpatialReference()
    dst_crs.ImportFromEPSG(code)

    bounds_crs = osr.SpatialReference()
    bounds_crs.ImportFromEPSG(4326)

    opts = gdal.WarpOptions(
        outputBounds=shp.geometry.shape(aoi).buffer(0).bounds,
        outputBoundsSRS=bounds_crs,
        dstSRS=dst_crs)
    vrt = gdal.Warp(out_path, in_path, options=opts)
    del vrt

    return out_path
