"""
Tools for accessing querying for elevation data.
"""
import numbers
import numpy as np
import os
import random
import requests
import time
import ubelt as ub
from os.path import join


class ElevationDatabase:
    """
    An object that might use various backends to query elevation for a given
    latitude and longitude.

    """

    def __init__(self):
        pass

    @classmethod
    def coerce(cls, key):
        """
        Attempt to resolve key to an elevation database.

        Args:
            key (str): Available options are:
                * "open-elevation"
                * "gtop30"
                * A number for a constant elevation
        """
        if key is False:
            self = ConstantElevationDatabase(0)
        elif isinstance(key, numbers.Number):
            self = ConstantElevationDatabase(key)
        elif key == 'open-elevation':
            self = OpenElevationDatabase()
        elif key == 'gtop30':
            self = girder_gtop30_elevation_dem()
        else:
            raise KeyError(key)
        return self


class ConstantElevationDatabase(ElevationDatabase):
    """
    Fallback compatibility API when no elevation information is available
    """

    def __init__(self, const):
        self.const = const

    def query(self, lats, lons):
        lats_, lons_, was_iterable = ensure_iterable_latlons(lats, lons)
        if was_iterable:
            return np.array([self.const] * len(lats_))
        else:
            return self.const


class OpenElevationDatabase(ElevationDatabase):
    """
    Use open-elevation to query the elevation for a lat/lon point.

    This issues a web request, so it can be slow.

    Args:
        lat (float): degrees in latitude
        lon (float): degrees in longitude
        cache (bool): if True uses on-disk caching
        attempts (int): number of attempts before giving up
        verbose (int): verbosity flag

    Returns:
        float : elevation in meters

    References:
        https://gis.stackexchange.com/questions/338392/getting-elevation-for-multiple-lat-long-coordinates-in-python
        https://gis.stackexchange.com/questions/212106/seeking-alternative-to-google-maps-elevation-api
        https://open-elevation.com/
        https://www.freemaptools.com/elevation-finder.htm

    Example:
        >>> # xdoctest: +REQUIRES(--network)
        >>> # xdoctest: +REQUIRES(--slow)
        >>> from geowatch.gis.elevation import *  # NOQA
        >>> lat = 37.65026538818887
        >>> lon = 128.81096081618637
        >>> eldb = OpenElevationDatabase()
        >>> elevation = eldb.query(lat, lon, verbose=3)
        >>> print('elevation = {!r}'.format(elevation))
        elevation = 449
    """

    def query(self, lat, lon, **kwargs):
        # We could vectorize this, but if is very slow, and we probably should
        # always use gtop30 instead.
        return _query_open_elevation(lat, lon, **kwargs)


def _query_open_elevation(lat, lon, cache=True, attempts=10, verbose=0):
    url = 'https://api.open-elevation.com/api/v1/lookup?'
    suffix = 'locations={},{}'.format(float(lat), float(lon))
    query_url = url + suffix

    cacher = ub.Cacher('elevation', depends=query_url,
                       appname='geowatch/elevation_query', verbose=verbose)
    body = cacher.tryload()
    if body is None:
        for _i in range(attempts):
            result = requests.get(query_url)
            if result.status_code != 200:
                if verbose:
                    print('REQUEST FAILED')
                    print(result.text)
                    print('RETRY')
                time.sleep(3 + random.random() * 3)
            else:
                body = result.json()
                break
        if body is None:
            raise Exception('Failed to query')
        cacher.save(body)
    elevation = body['results'][0]['elevation']
    return elevation


class DEM_Collection(ElevationDatabase):
    """
    Manage a collection of DEM geotiffs

    Example:
        >>> # xdoctest: +REQUIRES(--network)
        >>> # xdoctest: +REQUIRES(--slow)
        >>> # Use the gtop30 DEM dataset from GIRDER to find elevation.
        >>> from geowatch.gis.elevation import *  # NOQA
        >>> dems = DEM_Collection.gtop30()
        >>> lat, lon = (37.7455555555556, 128.780555555556)
        >>> print(dems.query(lat, lon))
        499.0

        >>> # Running one query on the CI is good enough
        >>> # xdoctest: +REQUIRES(--slow)
        >>> # xdoctest: +REQUIRES(--slow)
        >>> lat, lon = (37.7280555555556, 129.008888888889)
        >>> print(dems.query(lat, lon))
        0
        >>> lat, lon = (37.6947222222222, 129.008888888889)
        >>> print(dems.query(lat, lon))
        95.0
        >>> lat, lon = (37.7127777777778, 128.780555555556)
        >>> print(dems.query(lat, lon))
        452.0
        >>> lat, lon = (0, 0)
        >>> print(dems.query(lat, lon))
        0
        >>> lon_basis = np.linspace(-175, 175, 100)
        >>> lat_basis = np.linspace(-85, 85, 100)
        >>> lats_, lons_ = np.meshgrid(lat_basis, lon_basis)
        >>> lats = lats_.ravel()
        >>> lons = lons_.ravel()
        >>> elevations = dems.query(lats, lons)
    """
    def __init__(dems, dem_paths):
        from geowatch.gis.geotiff import geotiff_crs_info
        from geowatch.utils import util_gis
        dem_infos = []
        for dem_fpath in dem_paths:
            dem_info = geotiff_crs_info(dem_fpath)
            dem_infos.append(dem_info)

        dem_crs84_polys = []
        for info in dem_infos:
            gdf = util_gis.crs_geojson_to_gdf(info['geos_corners'])
            gdf_crs84 = gdf.to_crs(util_gis.get_crs84())
            sh_poly = gdf_crs84['geometry'].iloc[0]
            dem_crs84_polys.append(sh_poly)

        dems.dem_paths = dem_paths
        dems.dem_infos = dem_infos
        dems.dem_crs84_polys = dem_crs84_polys

    def __reduce__(self):
        """
        Make this object pickleable by saving references, and then reconstruct
        infos on unpickle. Note: this is inefficient and could be improved.

        Example:
            >>> # xdoctest: +REQUIRES(--network)
            >>> # xdoctest: +REQUIRES(--slow)
            >>> # Check that we can pickle this object
            >>> from geowatch.gis.elevation import *  # NOQA
            >>> dems = DEM_Collection.gtop30()
            >>> import pickle
            >>> text = pickle.dumps(dems)
            >>> recon = pickle.loads(text)
        """
        cls = self.__class__
        args = (self.dem_paths,)
        return (cls, args)

    def find_reference_fpath(dems, lat, lon):
        import shapely

        crs84_query = shapely.geometry.Point(lon, lat)
        flags = [poly.contains(crs84_query) for poly in dems.dem_crs84_polys]
        idxs = np.where(flags)[0]
        assert len(idxs) == 1

        idx = idxs[0]
        dem_fpath = dems.dem_paths[idx]
        dem_info = dems.dem_infos[idx]
        return dem_fpath, dem_info

    def query(dems, lats, lons):
        """
        TODO: the API supports vectorization, but we need to make it more
        efficient.
        """
        from osgeo import gdal
        import kwimage

        lats_, lons_, was_iterable = ensure_iterable_latlons(lats, lons)

        elevations = []

        prev_dem_fpath = None
        prev_dem_band = None

        for lat_, lon_ in zip(lats_, lons_):
            dem_fpath, dem_info = dems.find_reference_fpath(lat_, lon_)

            latlon = kwimage.Coords(np.array([[lat_, lon_]], dtype=np.float64))
            xy = latlon.warp(dem_info['wgs84_to_wld']).warp(dem_info['wld_to_pxl'])

            x, y = np.floor(xy.data[0])
            gdalkw = dict(xoff=x, yoff=y, win_xsize=1, win_ysize=1)

            if prev_dem_fpath != dem_fpath:
                dem_ref = gdal.Open(dem_fpath, gdal.GA_ReadOnly)
                dem_band = dem_ref.GetRasterBand(1)
                prev_dem_band = dem_band
            else:
                dem_band = prev_dem_band

            nodata = dem_band.GetNoDataValue()

            data = dem_band.ReadAsArray(**gdalkw)
            data = data.ravel()
            assert len(data) == 1
            elevation = float(data[0])
            if elevation == nodata:
                # Assume when there is no data
                elevation = 0
            elevations.append(elevation)

        if was_iterable:
            elevations = np.array(elevations)
            return elevations
        else:
            elevation = elevations[0]
            return elevation

    @classmethod
    def gtop30(cls):
        """
        Build the gtop30 dataset

        Example:
            >>> # xdoctest: +REQUIRES(--network)
            >>> # xdoctest: +REQUIRES(--slow)
            >>> dems = DEM_Collection.gtop30()
            >>> lats = [1, 2, 3]
            >>> lons = [1, 2, 3]
            >>> dems.query(lats, lons)
        """
        gtop_dpath = ensure_girder_gtop30_elevation_maps()

        dem_paths = []
        for r, ds, fs in os.walk(gtop_dpath):
            for f in fs:
                dem_paths.append(join(r, f))

        dems = DEM_Collection(dem_paths)
        return dems


def ensure_iterable_latlons(lats, lons):
    lat_was_iterable = ub.iterable(lats)
    lon_was_iterable = ub.iterable(lons)
    was_iterable = lon_was_iterable or lat_was_iterable

    if was_iterable:
        if lon_was_iterable and not lat_was_iterable:
            lons_ = lons
            lats_ = [lats] * len(lons)
        elif lat_was_iterable and not lon_was_iterable:
            lats_ = lats
            lons_ = [lons] * len(lats)
        else:
            lats_ = lats
            lons_ = lons
    else:
        lats_ = [lats]
        lons_ = [lons]

    return lats_, lons_, was_iterable


@ub.memoize
def girder_gtop30_elevation_dem():
    dems = DEM_Collection.gtop30()
    return dems


@ub.memoize
def ensure_girder_gtop30_elevation_maps():
    """
    Ensure that we have the GTOP30 Digital Elevation Maps (DEMS) available
    locally.

    References:
        https://data.kitware.com/#collection/59eb64168d777f31ac6477e7/folder/59fb784d8d777f31ac6480fb
        https://www.google.com/url?sa=j&url=https%3A%2F%2Fwww.usgs.gov%2Fcenters%2Feros%2Fscience%2Fusgs-eros-archive-digital-elevation-global-30-arc-second-elevation-gtopo30%3Fqt-science_center_objects%3D0%23qt-science_center_objects&uct=1599876275&usg=jBvv8w64RCBJd2SyQA3kUtKhMQ4.&source=chat
    """
    print('Building elevation map')
    from geowatch.utils import util_girder
    api_url = 'https://data.kitware.com/api/v1'
    resource_id = '59fb784d8d777f31ac6480fb'
    dl_path = util_girder.grabdata_girder(api_url, resource_id)
    return dl_path
