"""
Tools relating to coordinate reference systems

Notes:
    * GEOJSON is assumed to be specified as reversed WGS84 with reversed axes
      (i.e. lon/lat) **unless** another CRS is specified.

    * WGS84 (aka EPSG 4326) is always given as latitude longitude
"""
import ubelt as ub
import numpy as np
import numbers


def utm_epsg_from_latlon(lat, lon):
    """
    Find a reasonable UTM CRS for a given lat / lon

    The purpose of this function is to get a reasonable CRS for computing
    distances in meters. If the region of interest is very large, this may not
    be valid.

    Args:
        lat (float): degrees in latitude
        lon (float): degrees in longitude

    Returns:
        int : the ESPG code of the UTM zone

    References:
        https://gis.stackexchange.com/questions/190198/how-to-get-appropriate-crs-for-a-position-specified-in-lat-lon-coordinates
        https://gis.stackexchange.com/questions/365584/convert-utm-zone-into-epsg-code

    Example:
        >>> from watch.gis.spatial_reference import *  # NOQA
        >>> epsg_code = utm_epsg_from_latlon(0, 0)
        >>> print('epsg_code = {!r}'.format(epsg_code))
        epsg_code = 32631
    """
    import utm
    # easting, northing, zone_num, zone_code = utm.from_latlon(min_lat, min_lon)
    zone_num = utm.latlon_to_zone_number(lat, lon)

    # Construction of EPSG code from UTM zone number
    south = lat < 0
    epsg_code = 32600
    epsg_code += int(zone_num)
    if south is True:
        epsg_code += 100
    return epsg_code


def check_latlons(lat, lon):
    """
    Quick check to see if latitudes and longitudes are valid.

    Longitude is always between -180 and 180 (degrees east)
    Latitude is always between -90 and 90 (degrees north)

    Example:
        >>> from watch.gis.spatial_reference import *  # NOQA
        >>> import pytest
        >>> assert not check_latlons(1000, 1000)
        >>> assert check_latlons(0, 0)
    """
    lat = np.asarray(lat)
    lon = np.asarray(lon)
    bad_lat = (lat < -90).any() or (lat > 90).any()
    bad_lon = (lon < -180).any() or (lon > 180).any()
    in_ranges = not (bad_lon or bad_lat)
    return in_ranges


class RPCTransform(object):
    r"""
    Wrapper around rasterio RPC transforms

    Args:
        rpcs (rasterio.rpc.RPC): rasterio RPC data
        elevation (str): method used to determine the elevation when RPC
            information is used. Currently only "open-elevation" is available.

    Notes:
        * By definition rpcs are always referenced against WGS84 (EPSG:4326)
          coordinates.
        * However, we are accept and return reversed lon/lat points here.

    References:
        https://rasterio.readthedocs.io/en/latest/topics/reproject.html#reprojecting-with-other-georeferencing-metadata

    Example:
        >>> from watch.gis.spatial_reference import *  # NOQA
        >>> self = RPCTransform.demo()
        >>> pxl_pts = np.array([
        >>>     [    0,     0],
        >>>     [    0, 20000],
        >>>     [20000,     0],
        >>>     [20000, 20000],
        >>> ])
        >>> wld_pts = self.warp_pixel_to_world(pxl_pts)
        >>> print('wld_pts =\n{}'.format(ub.repr2(wld_pts, nl=1, precision=2)))
        wld_pts =
        np.array([[128.87,  37.8 ],
                  [128.87,  37.7 ],
                  [129.  ,  37.81],
                  [129.  ,  37.71]]...)

        >>> wld_pts = np.array([[self.rpcs.long_off, self.rpcs.lat_off]])
        >>> pxl_pts = self.warp_world_to_pixel(wld_pts)
        >>> print('pxl_pts = {}'.format(ub.repr2(pxl_pts, nl=1, precision=2)))
        pxl_pts = np.array([[17576.2 , 10690.73]]...)
    """
    def __init__(self, rpcs, elevation='open-elevation'):
        self.rpcs = rpcs
        self.elevation = elevation

    @classmethod
    def demo(cls):
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
        self = cls.from_gdal(rpc_info)
        return self

    @ub.memoize_method
    def _default_elevation(self):
        """
        Determine a default elevation if none is provided.
        """
        if isinstance(self.elevation, numbers.Number):
            return self.elevation
        elif self.elevation == 'open-elevation':
            from watch.gis.elevation import query_open_elevation
            approx_elevation = query_open_elevation(
                self.rpcs.lat_off, self.rpcs.long_off)
        else:
            raise NotImplementedError(self.elevation)
        return approx_elevation

    @classmethod
    def from_gdal(cls, rpc_info, **kwargs):
        """
        Build an RPC transform from GDAL-style metadata

        Args:
            rpc_info (dict): gdal RPC dictionary. Typically aquired from
                ``osgeo.gdal.Dataset(<path>).GetMetadata(domain='RPC')``

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
            xs, ys = pts_in.T
            default_z = self._default_elevation()
            zs = np.full_like(xs, fill_value=default_z)
        else:
            xs, ys, zs = pts_in.T
        return xs, ys, zs

    def warp_world_to_pixel(self, pts_in):
        """
        Args:
            pts_in (ndarray):
                Either an Nx2 array of lon/lat WGS84 coordinates or
                an Nx3 array of lon/lat/elevation coordinates.

        Returns:
            pts_out (ndarray): An Nx2 array of pixel coordinates
        """
        from rasterio._transform import _rpc_transform
        xs, ys, zs = self._ensure_xyz(pts_in)
        transform_direction = 1
        x_out, y_out = _rpc_transform(self.rpcs, xs, ys, zs,
                                      transform_direction)
        x_out = np.array(x_out)[:, None]
        y_out = np.array(y_out)[:, None]
        pts_out = np.concatenate([x_out, y_out], axis=1)
        return pts_out

    def warp_pixel_to_world(self, pts_in):
        """
        Args:
            pts_in (ndarray):
                Either an Nx2 array of x,y pixel coordinates, or
                an Nx3 array of x,y,elevation coordinates.

        Returns:
            pts_out (ndarray): An Nx2 array of lon/lat WGS84 coordinates
        """
        from rasterio._transform import _rpc_transform
        xs, ys, zs = self._ensure_xyz(pts_in)
        transform_direction = 0
        x_out, y_out = _rpc_transform(self.rpcs, xs, ys, zs,
                                      transform_direction)
        x_out = np.array(x_out)[:, None]
        y_out = np.array(y_out)[:, None]
        pts_out = np.concatenate([x_out, y_out], axis=1)
        return pts_out
