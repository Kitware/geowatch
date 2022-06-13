"""
Utilities for rasterio

SeeAlso
    util_gdal.py
"""
import kwimage
import numpy as np
import pygeos
import shapely as shp
import shapely.geometry
import shapely.ops
import ubelt as ub
import warnings

from contextlib import ExitStack
from dataclasses import dataclass
from tempenv import TemporaryEnvironment
from typing import Union, List, Literal, Optional
import pyproj

import rasterio
import rasterio.features
import rasterio.mask
from rasterio import Affine, MemoryFile
from rasterio.enums import Resampling

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
    """
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
    """
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
         max_polys=None,
         use_overview=0):
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

        use_overview (int):
            if non-zero uses the closest overview if it is available.
            This increases computation time, but gives a better polygon when
            use_overview is closer to 0.

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
            img_height = img.height
            img_width = img.width

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
                    if use_overview < 0:
                        use_overview = len(overview_levels) + use_overview

                    requested_overview = min(max(use_overview, 0), len(overview_levels) - 1)
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
            from skimage.morphology import convex_hull_image
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    'ignore', 'Input image is entirely zero',
                    category=UserWarning)
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
            scaled_tolerance = tolerance
            if scale_factor is not None:
                scaled_tolerance = tolerance / scale_factor
            polys = [poly.buffer(0).simplify(scaled_tolerance) for poly in polys]

        mask_poly = shp.ops.unary_union(polys).buffer(0)

        if tolerance is not None:
            mask_poly = mask_poly.simplify(scaled_tolerance)

        # do this again to fix any weirdness from union
        if convex_hull:
            mask_poly = mask_poly.convex_hull

        if scale_factor is not None:
            # Move from area space into point space?
            # mask_poly = shapely.affinity.translate(mask_poly, xoff=-0.5, yoff=-0.5)
            mask_poly = shapely.affinity.scale(mask_poly,
                                               xfact=scale_factor,
                                               yfact=scale_factor,
                                               origin=(0.0, 0.0))
            # Using overviews to compute a polygon has slack.
            # Buffer to account for this.
            mask_poly = mask_poly.buffer(scale_factor)

            # Clip to the bounds
            bounds = shapely.geometry.box(0, 0, img_width, img_height)
            mask_poly = mask_poly.intersection(bounds)

            if tolerance is not None:
                mask_poly = mask_poly.simplify(tolerance)

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
        List[shapely.Polygon | None]:
            of cropped polys, some of which may be None. This maintains
            indexing relative to the input polygons.

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
