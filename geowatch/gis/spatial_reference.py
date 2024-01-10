"""
Tools relating to coordinate reference systems

Notes:
    * GEOJSON is assumed to be specified as reversed WGS84 with reversed axes
      (i.e. lon/lat) **unless** another CRS is specified.

    * WGS84 (aka EPSG 4326) is always given as latitude longitude

    * EPSG stands for "European Petroleum Survey Group"
"""
import ubelt as ub
import numpy as np
from functools import partial
from geowatch.gis.elevation import ElevationDatabase


class RPCTransform:
    r"""
    Wrapper around rasterio RPC transforms

    Args:
        rpcs (rasterio.rpc.RPC): rasterio RPC data
        elevation (str): method used to determine the elevation when RPC
            information is used. Available options are:
                * "open-elevation"
                * "gtop30"

    Notes:
        * By definition rpcs are always referenced against WGS84 (EPSG:4326)
          coordinates.
        * However, we are accept and return lon/lat (i.e. reversed WGS84) points here.
        * TODO: don't use reversed. Use authority compliant

    References:
        https://rasterio.readthedocs.io/en/latest/topics/reproject.html#reprojecting-with-other-georeferencing-metadata
        http://geotiff.maptools.org/rpc_prop.html
        https://github.com/mapbox/rasterio/blob/master/rasterio/rpc.py
        https://github.com/mapbox/rasterio/blob/master/rasterio/_transform.pyx
        https://gdal.org/doxygen/gdal__alg_8h.html#af4c3c0d4c79218995b3a1f0bac3700a0

    Example:
        >>> # xdoctest: +REQUIRES(env:SLOW_DOCTEST)
        >>> from geowatch.gis.spatial_reference import *  # NOQA
        >>> self = RPCTransform.demo()
        >>> pxl_pts = np.array([
        >>>     [    0,     0],
        >>>     [    0, 20000],
        >>>     [20000,     0],
        >>>     [20000, 20000],
        >>> ])
        >>> wld_pts = self.warp_world_from_pixel(pxl_pts)
        >>> print('wld_pts =\n{}'.format(ub.urepr(wld_pts, nl=1, precision=2)))
        wld_pts =
        np.array([[128.87,  37.8 ],
                  [128.87,  37.7 ],
                  [129.  ,  37.81],
                  [129.  ,  37.71]]...)

        >>> wld_pts = np.array([[self.rpcs.long_off, self.rpcs.lat_off]])
        >>> pxl_pts = self.warp_pixel_from_world(wld_pts)
        >>> print('pxl_pts = {}'.format(ub.urepr(pxl_pts, nl=1, precision=2)))
        pxl_pts = np.array([[17576.2 , 10690.73]]...)
    """

    def __init__(self, rpcs, elevation='gtop30',
                 axis_mapping='OAMS_TRADITIONAL_GIS_ORDER'):
        self.rpcs = rpcs
        self.elevation = ElevationDatabase.coerce(elevation)
        self.axis_mapping = axis_mapping
        assert axis_mapping == 'OAMS_TRADITIONAL_GIS_ORDER', (
            'we dont handle lat/lon yet, TODO')

    # TODO
    # def __json__():
    #     pass

    # def coerce():
    #     pass

    @classmethod
    def demo(cls, **kw):
        """
        Return a demo RPCTransform for testing purposes
        """
        rpc_info = {
            'HEIGHT_OFF': '134',
            'HEIGHT_SCALE': '501',
            'LAT_OFF': '37.7574',
            'LAT_SCALE': '0.063',
            'LINE_DEN_COEFF': (
                '1 -0.001280647 0.002106012 -0.0001507726 -3.584366e-05 '
                '0 -3.407159e-06 -1.13468e-05 8.050045e-05 -2.154705e-05 '
                '3.680717e-06 -1.18494e-06 -0.0001362055 2.194475e-08 '
                '2.225298e-05 0.0002853284 1.715958e-07 -2.953517e-07 '
                '-1.154067e-05 0 '),
            'LINE_NUM_COEFF': (
                '0.005421624 0.1940749 -1.211795 0.01633917 -0.0004186136 '
                '2.647701e-05 -4.734004e-05 -0.001422265 -0.003701303 '
                '6.023496999999999e-08 4.668106e-07 -4.195505e-06 '
                '-2.337797e-05 -4.190499e-06 1.710599e-05 '
                '7.084406000000001e-05 2.61384e-05 -4.056353e-07 '
                '-1.79412e-06 -3.535734e-07 '),
            'LINE_OFF': '10679',
            'LINE_SCALE': '10680',
            'LONG_OFF': '128.9868',
            'LONG_SCALE': '0.1238',
            'MAX_LAT': '37.7889',
            'MAX_LONG': '129.0487',
            'MIN_LAT': '37.7259',
            'MIN_LONG': '128.9249',
            'SAMP_DEN_COEFF': (
                '1 0.0004235249 0.0006144441 -0.0003215715 -4.071902e-08 '
                '-2.171622e-06 1.747825e-06 9.980383e-06 -3.155285e-05 '
                '-4.147756e-07 -1.752229e-08 -1.628482e-07 5.040285e-07 '
                '0 -1.813282e-07 7.100189e-07 0 -3.722191e-08 '
                '1.117271e-08 0 '),
            'SAMP_NUM_COEFF': (
                '-0.00757565 1.026796 0.0007400306 -0.0254972 '
                '-0.000593794 0.0001250929 -6.722757e-05 0.007166261 '
                '6.205097e-05 -4.390415e-06 -1.18329e-06 4.165168e-05 '
                '2.937798e-05 -3.408414e-07 -1.045831e-05 -0.0001462765 '
                '-1.455117e-07 2.673684e-06 6.849923e-06 1.029223e-08 '),
            'SAMP_OFF': '17589',
            'SAMP_SCALE': '17590',
        }
        self = cls.from_gdal(rpc_info, **kw)
        return self

    @ub.memoize_method
    def _default_elevation(self):
        """
        Determine a default elevation if none is provided.
        """
        approx_elevation = self.elevation.query(
            self.rpcs.lat_off, self.rpcs.long_off)
        return approx_elevation

    @classmethod
    def from_gdal(cls, rpc_info, **kwargs):
        """
        Build an RPC transform from GDAL-style metadata

        Args:
            rpc_info (dict): gdal RPC dictionary. Typically aquired from
                ``gdal.Open(<path>).GetMetadata(domain='RPC')``

            **kwargs : passed to `class`:RPCTransform
        """
        import rasterio.rpc
        rpcs = rasterio.rpc.RPC.from_gdal(rpc_info)
        self = cls(rpcs, **kwargs)
        return self

    def _ensure_xyz(self, pts_in):
        """
        Ensure that points include elevation data. If not specified,
        attempts to determine a "reasonable" default elevation.
        """
        if len(pts_in.shape) != 2:
            raise ValueError('Expected a 2D array of points')
        N, D = pts_in.shape
        if D == 2:
            # Note: we can actually supply a DEM
            # via passing kwargs **{'RPC_DEM': <path>}
            from rasterio import _transform
            if hasattr(_transform, '_rpc_transform'):
                _rpc_transform = partial(_transform._rpc_transform, self.rpcs)
                transform_direction = 0
            else:
                # TODO: use the new API more correctly
                from rasterio.transform import RPCTransformer
                transformer = RPCTransformer(self.rpcs)
                _rpc_transform = transformer._transform
                transform_direction = 1
            xs, ys = pts_in.T
            default_z = self._default_elevation()
            zs = np.full_like(xs, fill_value=default_z)

            max_iters = 20
            converge_thresh = 3
            num_same = 0

            for iter_num in range(max_iters):
                lons, lats = _rpc_transform(xs, ys, zs, transform_direction)
                new_zs = self.elevation.query(lats, lons)

                if np.all(new_zs == zs):
                    num_same += 1
                else:
                    num_same = 0

                zs = new_zs

                if num_same > converge_thresh:
                    # converged
                    break
        else:
            xs, ys, zs = pts_in.T
        return xs, ys, zs

    def _ensure_lonlatz(self, pts_in):
        """
        Ensure that points include elevation data. If not specified,
        attempts to determine a "reasonable" default elevation.
        """
        if len(pts_in.shape) != 2:
            raise ValueError('Expected a 2D array of points')
        N, D = pts_in.shape
        if D == 2:
            xs, ys = pts_in.T
            zs = self.elevation.query(ys, xs)
            # default_z = self._default_elevation()
            # zs = np.full_like(xs, fill_value=default_z)
        else:
            xs, ys, zs = pts_in.T
        return xs, ys, zs

    def warp_pixel_from_world(self, pts_in, return_elevation=False):
        """
        Args:
            pts_in (ndarray):
                Either an Nx2 array of lon/lat WGS84 coordinates or
                an Nx3 array of lon/lat/elevation coordinates.

        TODO:
            - [ ] Handle CRS

        Returns:
            pts_out (ndarray): An Nx2 array of pixel coordinates
        """
        from rasterio import _transform
        if hasattr(_transform, '_rpc_transform'):
            _rpc_transform = partial(_transform._rpc_transform, self.rpcs)
            transform_direction = 1
        else:
            # TODO: use the new API more correctly
            from rasterio.transform import RPCTransformer
            transformer = RPCTransformer(self.rpcs)
            _rpc_transform = transformer._transform
            transform_direction = 0

        lons, lats, elev = self._ensure_lonlatz(pts_in)
        x_out, y_out = _rpc_transform(lons, lats, elev,
                                      transform_direction)
        x_out = np.array(x_out)[:, None]
        y_out = np.array(y_out)[:, None]

        if return_elevation:
            elev = np.array(elev)[:, None]
            pts_out = np.concatenate([x_out, y_out, elev], axis=1)
        else:
            pts_out = np.concatenate([x_out, y_out], axis=1)
        return pts_out

    def warp_world_from_pixel(self, pts_in, return_elevation=False):
        r"""
        Args:
            pts_in (ndarray):
                Either an Nx2 array of x,y pixel coordinates, or
                an Nx3 array of x,y,elevation coordinates.

        TODO:
            - [ ] Handle CRS

        Returns:
            pts_out (ndarray): An Nx2 array of lon/lat WGS84 coordinates

        Example:
            >>> # xdoctest: +SKIP
            >>> # xdoctest: +REQUIRES(env:SLOW_DOCTEST)
            >>> from geowatch.gis.spatial_reference import *  # NOQA
            >>> self = RPCTransform.demo(elevation='gtop30')
            >>> pts_in = pxl_pts = np.array([
            >>>     [    0,     0],
            >>>     [    0, 20000],
            >>>     [20000,     0],
            >>>     [20000, 20000],
            >>> ])
            >>> wld_pts = self.warp_world_from_pixel(pxl_pts)
            >>> print('wld_pts =\n{}'.format(ub.urepr(wld_pts, nl=1, precision=2)))
            >>> #
            >>> import osgeo
            >>> from geowatch.gis.spatial_reference import *  # NOQA
            >>> gpath = '/home/joncrall/data/dvc-repos/smart_watch_dvc/drop0/KR-Pyeongchang-WV/_assets/20140131_a_KRG_011778204_10_0/011778204010_01_003/011778204010_01/011778204010_01_P002_PAN/14JAN31020440-P1BS-011778204010_01_P002.NTF'
            >>> #gpath = '/home/joncrall/data/dvc-repos/smart_watch_dvc/drop0/KR-Pyeongchang-WV/_assets/20140131_a_KRG_011778204_10_0/011778204010_01_003/011778204010_01/011778204010_01_P001_PAN/14JAN31020439-P1BS-011778204010_01_P001.NTF'
            >>> from osgeo import gdal
            >>> ref = gdal.Open(gpath)
            >>> rpc_info = ref.GetMetadata(domain='RPC')
            >>> pts_in = pxl_pts = np.array([
            >>>     [    0,                         0],
            >>>     [    0,           ref.RasterYSize],
            >>>     [ref.RasterXSize, ref.RasterYSize],
            >>>     [ref.RasterXSize,               0],
            >>> ])
            >>> self = RPCTransform.from_gdal(rpc_info, elevation='gtop30')
            >>> wld_pts = self.warp_world_from_pixel(pxl_pts)
            >>> kwimage.Polygon(exterior=wld_pts).draw(color='purple', alpha=0.5)
            >>> #
            >>> self = RPCTransform.from_gdal(rpc_info, elevation='open-elevation')
            >>> self.warp_world_from_pixel(pxl_pts)
        """
        from rasterio import _transform
        if hasattr(_transform, '_rpc_transform'):
            _rpc_transform = partial(_transform._rpc_transform, self.rpcs)
            transform_direction = 0
        else:
            # TODO: use the new API more correctly
            from rasterio.transform import RPCTransformer
            transformer = RPCTransformer(self.rpcs)
            _rpc_transform = transformer._transform
            transform_direction = 1

        xs, ys, zs = self._ensure_xyz(pts_in)
        x_out, y_out = _rpc_transform(xs, ys, zs, transform_direction)
        lons = np.array(x_out)[:, None]
        lats = np.array(y_out)[:, None]
        if return_elevation:
            zs = np.array(zs)[:, None]
            pts_out = np.concatenate([lons, lats, zs], axis=1)
        else:
            pts_out = np.concatenate([lons, lats], axis=1)
        return pts_out

    def make_warp_pixel_from_world(self):
        """
        Hack for pickelability
        """
        return RPCPixelFromWorldTransform(self)

    def make_warp_world_from_pixel(self):
        """
        Hack for pickelability
        """
        return RPCWorldFromPixelTransform(self)


class RPCPixelFromWorldTransform:
    """
    Helper class to pickle the warp_pixel_from_world method.
    I'm not sure if this is really needed. I would think a method can be
    pickled if its class can be pickeld...
    """

    def __init__(self, rpc_transform):
        self.rpc_transform = rpc_transform

    def __call__(self, pts_in, return_elevation=False):
        return self.rpc_transform.warp_pixel_from_world(pts_in, return_elevation)


class RPCWorldFromPixelTransform:
    """
    Helper class to pickle the warp_world_from_pixel method.
    I'm not sure if this is really needed. I would think a method can be
    pickled if its class can be pickeld...
    """

    def __init__(self, rpc_transform):
        self.rpc_transform = rpc_transform

    def __call__(self, pts_in, return_elevation=False):
        return self.rpc_transform.warp_world_from_pixel(pts_in, return_elevation)
