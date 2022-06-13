import kwimage
import numpy as np
import pygeos
import shapely as shp
import shapely.geometry
import shapely.ops

from typing import Union, List, Literal, Optional
import pyproj

import rasterio
import rasterio.features
import rasterio.mask

from watch.utils.util_raster import mask, _swapxy, _ensure_open


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
