import kwimage
import numpy as np
import os
import pygeos
import shapely as shp
import shapely.geometry
import shapely.ops
import ubelt as ub
import warnings
from skimage.morphology import convex_hull_image

from contextlib import ExitStack
from copy import deepcopy
from dataclasses import dataclass
from lxml import etree
from tempenv import TemporaryEnvironment
from tempfile import NamedTemporaryFile
from typing import Union, List, Literal, Optional
from osgeo import gdal, osr
import pyproj

import rasterio
import rasterio.features
import rasterio.mask
from rasterio import Affine, MemoryFile
from rasterio.enums import Resampling

from watch.gis.spatial_reference import utm_epsg_from_latlon

try:
    from xdev import profile
except Exception:
    profile = ub.identity


def _ensure_open(
        raster: Union[rasterio.DatasetReader, str]) -> rasterio.DatasetReader:
    if not isinstance(raster, rasterio.DatasetReader) or raster.closed:
        # workaround for
        # https://rasterio.readthedocs.io/en/latest/faq.html#why-can-t-rasterio-find-proj-db-rasterio-from-pypi-versions-1-2-0
        with TemporaryEnvironment({'PROJ_LIB': None, 'PROJ_DEBUG': '3'}):
            return rasterio.open(raster)
    else:
        return raster


def _swapxy(poly: shp.geometry.Polygon) -> shp.geometry.Polygon:
    return kwimage.Polygon.from_shapely(poly).swap_axes().to_shapely()


def open_cropped(raster: Union[rasterio.DatasetReader, str],
                 poly: shp.geometry.Polygon,
                 rect=True,
                 out_fpath=None) -> np.ndarray:
    '''
    Open and return the part of raster that falls within poly. Essentially the
    opposite of watch.util.util_raster.crop_to().

    This is safe to use on a remote path (eg s3 or WMS); rasterio will
    correctly only download the tiles needed instead of the entire file.

    Args:
        raster: Path to a dataset poly: to crop raster to in geo-space.
            Expressed in WGS84 lon-lat units.
        rect: if True, return the full rectangular data window that includes
            poly; else, the data will be tightly cropped to poly and filled
            outside it with nodata.
        out_fpath: if given, save the cropped data to this local path as a new
            image.

    Returns:
        data: the contents of the cropped raster.
    '''
    poly_epsg = 4326  # only true after swapping from lon-lat to lat-lon!
    poly = _swapxy(poly)

    raster = _ensure_open(raster)
    profile = raster.profile.copy()
    profile.pop('driver')

    # these methods return near-identical data and transform (+/- rounding
    # errors in data.shape and sigfigs in transform) for a rectangular polygon.
    if rect:  # use windowed read
        # https://rasterio.readthedocs.io/en/latest/topics/windowed-rw.html
        bounds = pyproj.Transformer.from_crs(
            poly_epsg, raster.crs).transform_bounds(*poly.bounds,
                                                    errcheck=True)
        window = rasterio.windows.from_bounds(*bounds, raster.transform)
        transform = rasterio.windows.transform(window, raster.transform)
        data = raster.read(window=window)
        profile.update(transform=transform,
                       height=window.height,
                       width=window.width)

    else:  # use mask
        # https://automating-gis-processes.github.io/CSC18/lessons/L6/clipping-raster.html
        # https://gis.stackexchange.com/a/127432
        tfm = pyproj.Transformer.from_crs(poly_epsg, raster.crs).transform
        bounds = shapely.ops.transform(tfm, poly)
        data, transform = rasterio.mask.mask(raster,
                                             shapes=[bounds],
                                             crop=True)
        profile.update(transform=transform,
                       height=data.shape[1],
                       width=data.shape[2])

    # TODO option to use MemoryFile?
    # Or is that handled by passing out_fpath=/vsimem/... ?
    if out_fpath is not None:
        with rasterio.open(out_fpath, **profile) as f:
            f.write(data)

    return data


@profile
def mask(raster: Union[rasterio.DatasetReader, str],
         default_nodata=None,
         save=False,
         convex_hull=False,
         as_poly=True,
         tolerance=None,
         max_polys=None):
    """
    Compute a raster's valid data mask in pixel coordinates.

    Note that this is the rasterio mask, which for multi-band rasters is the
    binary OR of the individual band masks. This is different from the gdal
    mask, which is always per-band.

    Args:
        raster (str): Path to a dataset (raster image file)

        default_nodata (int): if raster's nodata value is None, default to this

        save (bool): if True and raster's nodata value is None, write the
            default to it. If False, performance overhead is incurred from
            creating a tempfile

        convex_hull (bool):
            if True, return the convex hull of the mask image or poly

        as_poly (bool): if True, return the mask as a shapely Polygon or
            MultiPolygon instead of a raster image, in (w, h) order (opposite
            of Python convention).

        tolerance (int): if specified, simplifies the valid polygon.

    Returns:
        If as_poly, a shapely Polygon or MultiPolygon bounding the valid
        data region(s) in pixel coordinates.

        Else, a uint8 raster mask of the same shape as the input, where
        255 == valid and 0 == invalid.

    Ignore:
        raster = '/home/joncrall/data/dvc-repos/smart_watch_dvc/drop1/../drop1/_assets/google-cloud/LS/LC08_L1TP_016039_20160216_20170224_01_T1/LC08_L1TP_016039_20160216_20170224_01_T1_B1.TIF'  # noqa
        from watch.utils.util_raster import *
        mask_img = mask(raster, as_poly=False)

    Example:
        >>> # xdoctest: +REQUIRES(--network)
        >>> from watch.utils.util_raster import *
        >>> from watch.demo.landsat_demodata import grab_landsat_product
        >>> path = grab_landsat_product()['bands'][0]
        >>> #
        >>> mask_img = mask(path, as_poly=False)
        >>> import kwimage as ki
        >>> assert mask_img.shape == ki.load_image_shape(path)[:2]
        >>> assert set(np.unique(mask_img)) == {0, 255}
        >>> #
        >>> mask_poly = mask(path, as_poly=True)
        >>> import shapely
        >>> assert isinstance(mask_poly, shapely.geometry.Polygon)
        >>> # xdoctest: +REQUIRES(--show)
        >>> import kwplot
        >>> kwplot.autompl()
        >>> kwplot.figure(fnum=1, doclf=True)
        >>> imdata = kwimage.imread(path)
        >>> imdata = kwimage.normalize_intensity(imdata)
        >>> kw_poly = kwimage.Polygon.coerce(mask_poly.buffer(0).simplify(10))
        >>> canvas = imdata.copy()
        >>> mask_alpha = kwimage.ensure_alpha_channel(mask_img, alpha=(mask_img > 0))
        >>> canvas = kwimage.overlay_alpha_layers([mask_alpha, canvas])
        >>> canvas = kw_poly.scale(0.9, about='center').draw_on(canvas, color='green', alpha=0.6)
        >>> kw_poly.scale(1.1, about='center').draw(alpha=0.5, color='red', setlim=True)
        >>> kwplot.imshow(canvas)

    Example:
        >>> # Test how the "save" functionality modifies the data
        >>> import kwimage
        >>> from watch.utils.util_raster import *
        >>> import pathlib
        >>> dpath = pathlib.Path(ub.ensure_app_cache_dir('watch/tests/empty_raster'))
        >>> raster = dpath / 'empty.tif'
        >>> ub.delete(raster)
        >>> kwimage.imwrite(raster, np.zeros((3, 3, 5)))
        >>> info1 = ub.cmd('gdalinfo {}'.format(raster))
        >>> nodata = 0
        >>> mask_img = mask(raster, as_poly=False)
        >>> print('mask_img = {!r}'.format(mask_img))
        >>> info2 = ub.cmd('gdalinfo {}'.format(raster))
        >>> mask_poly = mask(raster, as_poly=True)
        >>> info3 = ub.cmd('gdalinfo {}'.format(raster))
        >>> print(info1['out'])
        >>> print(info2['out'])
        >>> print(info3['out'])
    """
    scale_factor = None

    with warnings.catch_warnings():
        warnings.filterwarnings(
            'ignore', category=rasterio.errors.NotGeoreferencedWarning)
        # workaround for
        # https://rasterio.readthedocs.io/en/latest/faq.html#why-can-t-rasterio-find-proj-db-rasterio-from-pypi-versions-1-2-0
        with TemporaryEnvironment({'PROJ_LIB': None, 'PROJ_DEBUG': '3'}):

            img = _ensure_open(raster)

            # Work at the coarsest overview level for speed
            overviews = {
                tuple(img.overviews(bandx))
                for bandx in range(1, img.count + 1)
            }
            if len(overviews) == 1:
                overview_levels = ub.peek(overviews)
                if len(overview_levels):
                    img.close()
                    # Open image with a higher overview level
                    # https://github.com/rasterio/rasterio/issues/1504
                    requested_overview = len(overview_levels) - 1
                    img = rasterio.open(img.name,
                                        'r',
                                        overview_level=requested_overview)
                    scale_factor = overview_levels[requested_overview]

            try:
                mask_img = None
                if default_nodata is None:
                    nodata = img.nodata
                    use_disk_nodata = True
                else:
                    nodata = default_nodata
                    use_disk_nodata = False

                if nodata is None:
                    # Not specified, and not introspectable
                    # TODO: early return
                    # if as_poly:
                    #     pass
                    # else:
                    mask_img = np.full((img.height, img.width),
                                       fill_value=255,
                                       dtype=np.uint8)

                else:
                    if save:
                        raise NotImplementedError(
                            'Dont update here. It can be unsafe. '
                            'Probably should be done in a separate script')

                    # if needs_nodata:
                    #     # The image was closed, so we must open a new one
                    #     if save:
                    #         img.close()
                    #         # Add on necessary information in footer
                    #         with rasterio.open(raster, 'r+') as img:
                    #             img.nodata = nodata
                    #         img = rasterio.open(raster, 'r')
                    #     else:
                    #         profile = img.profile.copy()
                    #         profile['nodata'] = nodata
                    #         # TODO could optimize this with rasterio.shutil.copy
                    #         # or https://rasterio.readthedocs.io/en/latest/topics/windowed-rw.html#blocks
                    #         data = img.read()
                    #         img.close()
                    #         profile.update(nodata=nodata)
                    #         memfile = MemoryFile()
                    #         img = memfile.open(**profile)
                    #         img.write(data)
                    if use_disk_nodata:
                        mask_img = img.dataset_mask()
                    else:
                        # simulate 0 = nodata, 255=valid data
                        # operate inplace when possible
                        imdata = img.read(1, out_dtype=np.uint8)
                        np.not_equal(imdata,
                                     nodata,
                                     dtype=np.uint8,
                                     out=imdata)
                        np.multiply(imdata, 255, out=imdata)
                        mask_img = imdata

            finally:
                img.close()

        if convex_hull:
            mask_img = convex_hull_image(mask_img).astype(np.uint8)

        if not as_poly:
            return mask_img

        if mask_img is None:
            raise AssertionError('mask image was None')

        # mask has values 0 and 255
        polys = []
        for poly, val in rasterio.features.shapes(mask_img, connectivity=4):
            if val > 0:
                polys.append(shp.geometry.shape(poly))
                if max_polys is not None and len(polys) > max_polys:
                    break

        if tolerance is not None:
            polys = [poly.buffer(0).simplify(tolerance) for poly in polys]
        mask_poly = shp.ops.unary_union(polys).buffer(0)
        if tolerance is not None:
            mask_poly.simplify(tolerance)

        # do this again to fix any weirdness from union
        if convex_hull:
            mask_poly = mask_poly.convex_hull

        if scale_factor is not None:
            mask_poly = shapely.affinity.scale(mask_poly,
                                               xfact=scale_factor,
                                               yfact=scale_factor,
                                               origin=(0, 0))

    return mask_poly


def crop_to(
    pxl_polys: List[shp.geometry.Polygon],
    raster: str,
    bounds_policy: Literal['none', 'bounds', 'valid'],
    intersect_policy: Literal['keep', 'crop', 'discard'] = 'crop'
) -> List[Optional[shp.geometry.Polygon]]:
    """
    Crop pxl_polys to raster in one of several ways.

    Computation is independent per pxl_poly, but vectorized for speed.

    Args:
        pxl_polys (List[shapely.Polygon]): In pixel coordinates in (w,h) order
            (opposite of Python convention).

        raster: Path to a dataset (raster image file).

        bounds_policy: "none", "bounds", or "valid"
            "none": Do not crop polygons. Makes this function a no-op.
            "bounds": Crop polygons that fall outside the image's height/width
            "valid": Crop polygons that fall outside the image's valid data
                mask. This is more restrictive than "bounds".

        intersect_policy: "keep", "crop", or "discard". Polygons that fall
            completely outside the bounds are discarded. This arg decides
            how to handle polygons that intersect the bounds.
            "keep": Return the polygon unchanged.
            "crop": Crop the polygon to the bounds.
            "discard": Discard the polygon (replace it with None).
    Returns:
        List[shapely.Polygon] of cropped polys, some of which may be None. This
        maintains indexing relative to the input polygons.

    Example:
        >>> # xdoctest: +REQUIRES(--network)
        >>> from watch.utils.util_raster import *
        >>> from watch.demo.landsat_demodata import grab_landsat_product
        >>> path = grab_landsat_product()['bands'][0]
        >>> #
        >>> # a polygon that partially intersects this image's bounds
        >>> # and valid mask
        >>> import shapely
        >>> poly = shapely.geometry.box(-500, -500, 2000, 2000)
        >>> #
        >>> # no-op for testing purposes
        >>> assert crop_to([poly], path, bounds_policy='none')[0] == poly
        >>> #
        >>> # handle intersecting polygons
        >>> assert crop_to([poly], path, bounds_policy='bounds',
        >>>                intersect_policy='keep')[0] == poly
        >>> assert crop_to([poly], path, bounds_policy='bounds',
        >>>                intersect_policy='discard')[0] == None
        >>> cropped = crop_to([poly], path, bounds_policy='bounds',
        >>>                intersect_policy='crop')[0]  # default
        >>> assert cropped.bounds == (0, 0, 2000, 2000)
        >>> #
        >>> # same with valid mask
        >>> cropped = crop_to([poly], path, bounds_policy='valid')[0]
        >>> assert cropped.bounds == (924, 14, 2000, 2000)


    """
    assert bounds_policy in {'none', 'bounds', 'valid'}, bounds_policy
    assert intersect_policy in {'keep', 'crop', 'discard'}, intersect_policy

    if bounds_policy == 'none':
        return pxl_polys

    # convert to pygeos for vectorized operations
    pxl_polys = pygeos.from_shapely(pxl_polys)

    if bounds_policy == 'bounds':
        h, w = kwimage.load_image_shape(raster)[:2]
        geom = pygeos.box(0, 0, w, h)

    elif bounds_policy == 'valid':
        mask_poly = mask(raster, as_poly=True)
        geom = pygeos.from_shapely(mask_poly)

    else:
        raise ValueError(bounds_policy)

    contains = pygeos.contains(geom, pxl_polys)
    overlaps = pygeos.overlaps(geom, pxl_polys)
    if intersect_policy == 'crop':  # optimization
        intersections = pygeos.intersection(geom, pxl_polys)
    else:
        intersections = pxl_polys

    # I don't see an obvious way to vectorize this part
    # due to the branching
    result = []
    for poly, c, o, inter in zip(pygeos.to_shapely(pxl_polys), contains,
                                 overlaps, pygeos.to_shapely(intersections)):
        if c:
            result.append(poly)
        elif o:
            if intersect_policy == 'keep':
                result.append(poly)
            elif intersect_policy == 'crop':
                result.append(inter)
            elif intersect_policy == 'discard':
                result.append(None)
            else:
                raise ValueError(intersect_policy)
        else:
            result.append(None)

    return result


def list_gdal_drivers():
    '''
    List all drivers currently available to GDAL to create a raster

    Returns:
        list((driver_shortname, driver_longname, list(driver_file_extension)))

    Example:
        >>> from watch.utils.util_raster import *
        >>> drivers = list_gdal_drivers()
        >>> assert ('GTiff', 'GeoTIFF', ['tif', 'tiff']) in drivers
    '''
    result = []
    for idx in range(gdal.GetDriverCount()):
        driver = gdal.GetDriver(idx)
        if driver:
            metadata = driver.GetMetadata()
            if metadata.get(gdal.DCAP_CREATE) == 'YES' and metadata.get(
                    gdal.DCAP_RASTER) == 'YES':
                name = driver.GetDescription()
                longname = metadata.get("DMD_LONGNAME")
                exts = metadata.get("DMD_EXTENSIONS")
                if exts is None:
                    exts = []
                else:
                    exts = exts.split(' ')
                result.append((name, longname, exts))
    return result


@dataclass
class ResampledRaster(ExitStack):
    """
    Context manager to rescale a raster on the fly using rasterio

    This changes the number of pixels in the raster while maintaining its
    geographic bounds, that is, it changes the raster's GSD.

    Args:
        raster: a DatasetReader (the object returned by rasterio.open) or path
            to a dataset
        scale: factor to upscale the resolution, aka downscale the GSD, by
        read: if True, read and return the resampled data (an expensive
            operation if scale>1) else, return the resampled dataset's .profile
            attribute (metadata)
        resampling: resampling algorithm, from rasterio.enums.Resampling [1]

    Example:
        >>> # xdoctest: +REQUIRES(--network)
        >>> from watch.utils.util_raster import *
        >>> from watch.demo.landsat_demodata import grab_landsat_product
        >>> path = grab_landsat_product()['bands'][0]
        >>> #
        >>> current_gsd_meters = 60
        >>> desired_gsd_meters = 10
        >>> scale = current_gsd_meters / desired_gsd_meters
        >>> #
        >>> with rasterio.open(path) as f:
        >>>     old_profile = f.profile
        >>> #
        >>> # can instantiate this class in a with-block
        >>> with ResampledRaster(path, scale=scale, read=False) as f:
        >>>     pass
        >>> #
        >>> # or have it stick around and change the resampling on the fly
        >>> resampled = ResampledRaster(path, scale=scale, read=False)
        >>> #
        >>> # the computation only happens when you invoke 'with'
        >>> with resampled as new_profile:
        >>>     assert new_profile['width'] == int(old_profile['width'] * scale)
        >>>     assert new_profile['crs'] == old_profile['crs']
        >>> #
        >>> resampled.scale = scale / 2
        >>> resampled.read = True
        >>> #
        >>> with resampled as new:
        >>>     assert new.profile['width'] == int(old_profile['width'] * scale / 2)
        >>>     assert new.profile['crs'] == old_profile['crs']
        >>>     # do other stuff with new

    References:
        https://gis.stackexchange.com/a/329439
        https://rasterio.readthedocs.io/en/latest/topics/reading.html
        https://rasterio.readthedocs.io/en/latest/topics/profiles.html
        [1] https://rasterio.readthedocs.io/en/latest/api/rasterio.enums.html#rasterio.enums.Resampling
    """
    raster: Union[str, rasterio.DatasetReader]
    scale: float = 2
    read: bool = True
    resampling: Resampling = Resampling.bilinear

    def __post_init__(self):
        super().__init__()

    def __enter__(self):
        self.raster = _ensure_open(self.raster)
        t = self.raster.transform

        # rescale the metadata
        transform = Affine(t.a / self.scale, t.b, t.c, t.d, t.e / self.scale,
                           t.f)
        height = int(np.ceil(self.raster.height * self.scale))
        width = int(np.ceil(self.raster.width * self.scale))

        profile = self.raster.profile
        profile.update(transform=transform,
                       driver='GTiff',
                       height=height,
                       width=width)

        if self.read:

            # Note changed order of indexes, arrays are band, row, col order
            # not row, col, band
            data = self.raster.read(
                out_shape=(self.raster.count, height, width),
                resampling=self.resampling,
            )

            # enter_context is from contextlib.ExitStack, which takes care of
            # closing these
            memfile = self.enter_context(MemoryFile())
            with memfile.open(**profile) as dataset:  # Open as DatasetWriter
                dataset.write(data)
                del data

            dataset = self.enter_context(
                memfile.open())  # Reopen as DatasetReader
            return dataset

        else:

            return profile

    def __exit__(self, *exc):
        pass


@dataclass
class GdalOpen:
    """
    A simple context manager for friendlier gdal use.

    Example:
        >>> # xdoctest: +REQUIRES(--network)
        >>> from watch.utils.util_raster import *
        >>> from watch.demo.landsat_demodata import grab_landsat_product
        >>> path = grab_landsat_product()['bands'][0]
        >>> #
        >>> # standard use:
        >>> dataset = gdal.Open(path)
        >>> print(dataset.GetDescription())  # do stuff
        >>> del dataset  # or 'dataset = None'
        >>> #
        >>> # equivalent:
        >>> with GdalOpen(path) as dataset:
        >>>     print(dataset.GetDescription())  # do stuff
        >>>
        >>> # open for writing:
        >>> with GdalOpen(path, gdal.GA_Update) as dataset:
        >>>     print(dataset.GetDescription())  # do stuff

    """
    path: str
    mode: int = gdal.GA_ReadOnly

    def __enter__(self):
        self.f = gdal.Open(self.path, self.mode)
        return self.f

    def __exit__(self, *exc):
        # gdal.GDALClose(f)  # not implemented in this version of gdal?
        del self.f  # this is ugly, but it works...
        # gdal.Unlink(path)  # THIS DELETES THE FILE


def reroot_vrt(old_path, new_path, keep_old=True):
    """
    Copy a VRT file, fixing relative paths to its component images

    Example:
        >>> # xdoctest: +REQUIRES(--network)
        >>> import os
        >>> from osgeo import gdal
        >>> from watch.utils.util_raster import *
        >>> from watch.demo.landsat_demodata import grab_landsat_product
        >>> bands = grab_landsat_product()['bands']
        >>> #
        >>> # VRT must be created in the imgs' subtree
        >>> tmp_path = os.path.join(os.path.dirname(bands[0]), 'all_bands.vrt')
        >>> # (consider using the wrapper util_raster.make_vrt instead of this)
        >>> gdal.BuildVRT(tmp_path, sorted(bands))
        >>> # now move it somewhere more convenient
        >>> reroot_vrt(tmp_path, './bands.vrt', keep_old=False)
        >>> #
        >>> # clean up
        >>> os.remove('bands.vrt')
    """
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
    """
    Stack multiple band files in the same directory into a single VRT

    Args:
        in_paths: list(path)
        out_path: path to save to; standard is '*.vrt'. If None, a path will be
            generated.
        mode:
            'stacked': Stack multiple band files covering the same area
            'mosaicked': Mosaic/merge scenes with overlapping areas. Content
                will be taken from the first in_path without nodata.
        relative_to_path: if this function is being called from another
            process, pass in the cwd of the calling process, to trick gdal into
            creating a rerootable VRT
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
        >>> #
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
        >>> os.remove('bands1.vrt')
        >>> os.remove('bands2.vrt')
        >>> os.remove('full_scene.vrt')

    References:
        [1] https://gdal.org/programs/gdalbuildvrt.html
        [2] https://gdal.org/python/osgeo.gdal-module.html#BuildVRTOptions
    """

    if mode == 'stacked':
        kwargs['separate'] = True
    elif mode == 'mosaicked':
        kwargs['separate'] = False
        kwargs['srcNodata'] = 0  # this ensures nodata doesn't overwrite data
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

        # First, create VRT in a place where it can definitely see the input
        # files.  Use a relative instead of absolute path to ensure that
        # <SourceFilename> refs are relative, and therefore the VRT is
        # rerootable
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
    """
    Search for band files from compatible scenes and stack them in a single
    mosaicked VRT

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
        >>> #
        >>> # pretend there are more scenes here
        >>> out_path = scenes_to_vrt([sorted(bands)] , vrt_root='.', relative_to_path=os.getcwd())
        >>> with GdalOpen(out_path) as f:
        >>>     print(f.GetDescription())
        >>> #
        >>> # clean up
        >>> os.remove(out_path)
    """
    # first make VRTs for individual tiles
    # TODO use https://rasterio.readthedocs.io/en/latest/topics/memory-files.html
    # for these intermediate files?
    tmp_vrts = [
        make_vrt(scene,
                 os.path.join(vrt_root, f'{hash(scene[0])}.vrt'),
                 mode='stacked',
                 relative_to_path=relative_to_path) for scene in scenes
    ]

    # then mosaic them
    final_vrt = make_vrt(tmp_vrts,
                         os.path.join(vrt_root,
                                      f'{hash(scenes[0][0] + "final")}.vrt'),
                         mode='mosaicked',
                         relative_to_path=relative_to_path)

    if 0:  # can't do this because final_vrt still references them
        for t in tmp_vrts:
            os.remove(t)

    return final_vrt


def reproject_crop(in_path, aoi, code=None, out_path=None, vrt_root=None):
    """
    Crop an image to an AOI and reproject to its UTM CRS (or another CRS)

    Unfortunately, this cannot be done in a single step in scenes_to_vrt
    because gdal.BuildVRT does not support warping between CRS.
    Cropping alone could be done in scenes_to_vrt. Note gdal.BuildVRTOptions
    has an outputBounds(=-te) kwarg for cropping, but not an equivalent of
    -te_srs.

    This means another intermediate file is necessary for each warp operation.

    TODO check for this quantization error:
        https://gis.stackexchange.com/q/139906

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
        >>> #
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
        >>> #
        >>> out_path = reproject_crop(band1, geojson_bbox)
        >>> #
        >>> # clean up
        >>> os.remove(out_path)
    """
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
            utm_epsg_from_latlon(lat, lon)
            for lon, lat in aoi['coordinates'][0]
        ]
        u, counts = np.unique(codes, return_counts=True)
        if len(u) > 1:
            print(
                f'Warning: AOI crosses UTM zones {u}. Taking majority vote...')
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
