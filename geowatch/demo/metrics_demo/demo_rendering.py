"""
Logic for rendering toy images for generated demo-data.
"""
import kwimage
import kwarray
import numpy as np
from osgeo import osr


def render_toy_georeferenced_image(img_dpath, renderable, rng=None):
    """
    Args:
        img_dpath (PathLike):
            path where the images can be written

        renderable (Dict):
            information about how to render a demo image

        rng : random state

    Returns:
        PathLike:
            The filepath where the image was written

     FIXME:
         I might be doing something wrong with warping polygons from image to
         world space, the y-axis is flipped. Tracking the issue here:
         https://github.com/rasterio/rasterio/issues/2565

    Example:
        >>> from geowatch.demo.metrics_demo.demo_rendering import *  # NOQA
        >>> from geowatch.demo.metrics_demo import demo_utils
        >>> import tempfile
        >>> import kwarray
        >>> from datetime import datetime as datetime_cls
        >>> import ubelt as ub
        >>> img_dpath = ub.Path(tempfile.mkdtemp())
        >>> rng = kwarray.ensure_rng(4324)
        >>> region_poly_wld = demo_utils.random_geo_polygon(rng=rng)
        >>> site_poly_wld = region_poly_wld.scale(0.5, about='center')
        >>> wld_polygon = region_poly_wld
        >>> img_width, img_height = 200, 300
        >>> image_box = kwimage.Boxes([[0, 0, img_width, img_height]], "xywh")
        >>> image_corners = image_box.corners().astype(float)
        >>> wld_box = wld_polygon.bounding_box()
        >>> wld_corners = wld_box.corners()
        >>> tf_img_from_wld = kwimage.Affine.fit(wld_corners, image_corners)
        >>> tf_wld_from_img = tf_img_from_wld.inv()
        >>> site_poly_img = site_poly_wld.warp(tf_img_from_wld)
        >>> renderable = {
        >>>     'sensor': 'foobar',
        >>>     'date': datetime_cls.now(),
        >>>     'frame_idx': 0,
        >>>     'image_dsize': (img_width, img_height),
        >>>     'visible_polys': [site_poly_img],
        >>>     'wld_polygon': wld_polygon,
        >>> }
        >>> fpath = render_toy_georeferenced_image(img_dpath, renderable, rng)
        >>> # xdoctest: +REQUIRES(--show)
        >>> # xdoctest: +REQUIRES(module:kwplot)
        >>> import geopandas as gpd
        >>> gdf = gpd.GeoDataFrame(geometry=[region_poly_wld.to_shapely(),
        >>>                                  site_poly_wld.to_shapely()], crs='crs84')
        >>> import kwplot
        >>> kwplot.autompl()
        >>> imdata = kwimage.imread(fpath)
        >>> SHOW_RASTERIO = 1
        >>> if SHOW_RASTERIO:
        >>>     kwplot.imshow(imdata, pnum=(1, 2, 1))
        >>> else:
        >>>     kwplot.imshow(imdata, pnum=(1, 1, 1))
        >>> site_poly_img.draw(edgecolor='black', fill=False)
        >>> # Rasterio rendering has a bug here
        >>> # https://github.com/rasterio/rasterio/issues/2565
        >>> if SHOW_RASTERIO:
        >>>     ax = kwplot.figure(pnum=(1, 2, 2)).gca()
        >>>     import rasterio
        >>>     from rasterio import plot  # NOQA
        >>>     f = rasterio.open(fpath)
        >>>     with rasterio.open(fpath, crs='crs84') as f:
        >>>         rasterio.plot.show(f, ax=ax, alpha=0.8)
        >>>         dst_crs = str(f.crs).lower()
        >>>     site_poly_wld.draw(edgecolor='black', fill=0)
        >>> gdf.to_crs(dst_crs).boundary.plot(ax=ax, color='green')
        >>> kwplot.show_if_requested()
    """
    rng = kwarray.ensure_rng(rng)

    sensor = renderable["sensor"]
    frame_idx = renderable["frame_idx"]
    yymmdd = renderable["date"].strftime("%Y%m%d")

    # This is the type of format that can be visualized
    fname = f"frame_{frame_idx}_{yymmdd}_{sensor}.jp2"
    img_fpath = img_dpath / fname

    image_dsize = renderable["image_dsize"]
    wld_polygon = renderable["wld_polygon"]

    # Make dummy image data
    _imdata = kwimage.grab_test_image("amazon")
    imdata = kwimage.imresize(_imdata, dsize=image_dsize)
    imdata = kwimage.ensure_float01(imdata)
    img_h, img_w = imdata.shape[0:2]

    # Speckle noise
    imdata += rng.rand(*imdata.shape) * 0.1
    imdata = imdata.clip(0, 1)

    # Render the polygons on the image
    visible_polys = renderable["visible_polys"]
    for poly in visible_polys:
        color = poly.meta.get("color", None)
        if color is None:
            color = kwimage.Color.random().as01()
        alpha = (rng.rand() * 0.4) + 0.6  # vary alpha
        imdata = poly.draw_on(imdata, color=color, alpha=alpha)

    if 0:
        # More realistic
        nodata_value = -9999
        imdata = (imdata * 10000).astype(np.int16)
    else:
        # Easier to work with
        nodata_value = 0
        imdata = kwimage.ensure_uint255(imdata)

    # Set a region to be nodata_value
    imdata[-10:, 10:] = nodata_value
    imdata[0:10:, -200:-180] = nodata_value
    imdata = imdata[:, :, 0:3]

    write_demo_geotiff(img_fpath=img_fpath, imdata=imdata,
                       wld_polygon=wld_polygon, nodata_value=nodata_value)
    return img_fpath


def write_demo_geotiff(img_fpath=None, imdata=None, wld_polygon=None,
                       nodata_value=None, rng=None, metadata=None):
    """
    Create a demo geotiff at a specified path. Arguments that are not specified
    will be randomly generated.

    Example:
        >>> from geowatch.demo.metrics_demo.demo_rendering import *  # NOQA
        >>> img_fpath = write_demo_geotiff()

    """
    rng = kwarray.ensure_rng(rng)

    if img_fpath is None:
        import ubelt as ub
        import tempfile
        dpath = ub.Path.appdir('geowatch/test/geotiff/demo').ensuredir()
        img_fpath = tempfile.mktemp(dir=dpath, prefix='demo_geotiff_', suffix='.tif')

    # Generate unspecified data
    if imdata is None:
        imdata = {}

    if isinstance(imdata, dict):
        img_w = imdata.get('width', None)
        img_h = imdata.get('height', None)
        image_dsize = img_w, img_h
        if img_w is None and img_h is None:
            image_dsize = None
        imdata = kwimage.grab_test_image("amazon", dsize=image_dsize)
        imdata = kwimage.ensure_float01(imdata)
        imdata += rng.rand(*imdata.shape) * 0.1
        imdata = imdata.clip(0, 1)
        img_h, img_w = imdata.shape[0:2]

    if metadata is None:
        metadata = {
            'SENSOR': 'kwimage-demo',
            'arbitrary': 'some arbitrary metadata',
        }

    if wld_polygon is None:
        from geowatch.demo.metrics_demo import demo_utils
        wld_polygon = demo_utils.random_geo_polygon(rng=rng)

    # Setup the geo metadata
    wld_box = wld_polygon.bounding_box()
    wld_corners = wld_box.corners()

    img_dsize = imdata.shape[0:2][::-1]
    img_width, img_height = img_dsize

    image_box = kwimage.Boxes([[0, 0, img_width, img_height]], "xywh")
    image_corners = image_box.corners().astype(float)

    # Compute values to trasnform from the world corners to the image corners.
    tf_wld_from_img = kwimage.Affine.fit(image_corners, wld_corners)

    # The CRS should be CRS-84
    srs = osr.SpatialReference()
    srs.ImportFromEPSG(4326)
    srs.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
    crs = srs.ExportToWkt()

    kwimage.imwrite(
        img_fpath,
        imdata,
        backend="gdal",
        nodata_value=nodata_value,
        crs=crs,
        transform=tf_wld_from_img,
        overviews="auto",
        metadata=metadata,
    )
    return img_fpath
