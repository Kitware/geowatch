"""
Functional utilities for generating basic components of the demodata.
"""
import ubelt as ub
import kwimage
import kwarray
import numpy as np
import geopandas as gpd
from shapely import geometry
import datetime as datetime_mod
from datetime import datetime as datetime_cls


def random_geo_points(num, rng=None):
    """
    Generate a uniformly random longitude, latitude.

    Based on logic described in [SO68298220]_.

    Args:
        num (int) : number of random points to generate
        rng : random seed or number generator

    Returns:
        ndarray:
            An Nx2 array with columns corresponding to CRS84 points
            (i.e. longitude, latitude)

    References:
        .. [SO68298220] https://stackoverflow.com/questions/68298220/random-geocoords

    Example:
        >>> from geowatch.demo.metrics_demo.demo_utils import *  # NOQA
        >>> latlon = random_geo_points(num=3, rng=0)
        >>> print(ub.urepr(latlon, precision=4))
        np.array([[ 16.1579,   5.6025],
                  [-27.4843,  25.4916],
                  [ 52.5219,  11.8603]], dtype=np.float64)

    Example:
        >>> # This example demonstrates that the points are randomly spread out
        >>> from geowatch.demo.metrics_demo.demo_utils import *  # NOQA
        >>> # xdoctest: +REQUIRES(--show)
        >>> # xdoctest: +REQUIRES(module:kwplot)
        >>> import kwplot
        >>> kwplot.autompl()
        >>> # Create random geopoints and place them into a GeoDataFrame
        >>> latlons = random_geo_points(300)
        >>> crs84 = get_crs84()
        >>> pts_gdf = gpd.GeoDataFrame(geometry=[geometry.Point(p) for p in latlons], crs=crs84)
        >>> # Plot the map of the world in the background
        >>> wld_map_gdf = gpd.read_file(
        >>>     gpd.datasets.get_path('naturalearth_lowres')
        >>> ).to_crs(crs84)
        >>> ax = wld_map_gdf.plot()
        >>> pts_gdf.plot(ax=ax, color='orange', alpha=0.8)
        >>> kwplot.show_if_requested()
    """
    rng = kwarray.ensure_rng(rng)
    u0, u1 = rng.rand(2, num)
    rad_lat = np.arcsin(2 * u0 - 1.0)  # angle with Equator   - from +pi/2 to -pi/2
    rad_lon = (2 * u1 - 1) * np.pi     # longitude in radians - from -pi to +pi
    rad_lonlat = np.stack([rad_lon, rad_lat], axis=1)
    lonlat = np.rad2deg(rad_lonlat)
    return lonlat


def random_geo_polygon(max_rt_area=10_000, rng=None):
    """
    Creates a random polygon of a "reasonable size" for a region in CRS84

    Args:
        max_rt_area (float):
            Maximum root area (i.e. sqrt(area)) of the polygon in meters.
            The generated polygon will usually have a root-area that is between
            a factor of 0.1 and 0.6 of this number.  Defaults to 10,000 meters.

        rng : random state or seed

    Returns:
        kwimage.Polygon : polygon in CRS84 space

    Example:
        >>> from geowatch.demo.metrics_demo.demo_utils import *  # NOQA
        >>> max_rt_area = 10000
        >>> region_poly = random_geo_polygon(max_rt_area, rng=321)
        >>> geo_poly = region_poly.round(6).to_geojson()
        >>> print('geo_poly = {}'.format(ub.urepr(geo_poly, nl=-1)))
        geo_poly = {
            'type': 'Polygon',
            'coordinates': [
                [
                    [-151.983974, 50.530122],
                    [-151.982547, 50.520243],
                    [-151.951446, 50.506376],
                    [-151.925555, 50.514069],
                    [-151.930728, 50.543657],
                    [-151.941043, 50.541291],
                    [-151.983974, 50.530122]
                ]
            ]
        }

    Example:
        >>> # This example demonstrates the distribution of random polygons
        >>> from geowatch.demo.metrics_demo.demo_utils import *  # NOQA
        >>> # xdoctest: +REQUIRES(--show)
        >>> # xdoctest: +REQUIRES(module:kwplot)
        >>> import kwplot
        >>> kwplot.autompl()
        >>> # Create random polygons and place them into a GeoDataFrame
        >>> rng = kwarray.ensure_rng(32321)
        >>> # Make the polygons very large, so they show up at world scale
        >>> max_rt_area = 1_000_000
        >>> polys = [random_geo_polygon(max_rt_area, rng=rng) for _ in range(10)]
        >>> crs84 = get_crs84()
        >>> poly_gdf = gpd.GeoDataFrame(geometry=[p.to_shapely() for p in polys], crs=crs84)
        >>> # Plot the map of the world in the background
        >>> wld_map_gdf = gpd.read_file(
        >>>     gpd.datasets.get_path('naturalearth_lowres')
        >>> ).to_crs(crs84)
        >>> ax = wld_map_gdf.plot()
        >>> poly_gdf.plot(ax=ax, color='limegreen', alpha=0.8)
        >>> poly_gdf.centroid.plot(ax=ax, color='orangered', alpha=0.5)
        >>> kwplot.show_if_requested()
    """
    # Choose a random centroid of the polygon
    lon, lat = random_geo_points(num=1, rng=rng).ravel()

    # Determine what UTM zone this random centroid falls in
    epsg_code = utm_epsg_from_latlon(lat, lon)

    crs84 = get_crs84()
    pt_crs84_gdf = gpd.GeoDataFrame(geometry=[
        geometry.Point(lon, lat)], crs=crs84)

    # Project the random lon/lat point into a UTM space
    pt_utm_gdf = pt_crs84_gdf.to_crs(epsg_code)
    utm_xy = pt_utm_gdf['geometry'].iloc[0].xy
    utm_x = utm_xy[0][0]
    utm_y = utm_xy[1][0]

    # Create a random polygon with an area between 0 and 1 (usually in 0.1-0.6)
    poly = kwimage.Polygon.random(rng=rng)
    # centered it at (0, 0)
    poly = poly.translate(-np.array(poly.centroid))

    # Scale the polygon to the appropriate area in UTM space, and translate it
    # into the randomly chosen center position.
    poly_utm = poly.scale(max_rt_area).translate((utm_x, utm_y))
    utm_gdf = gpd.GeoDataFrame(geometry=[poly_utm.to_shapely()],
                               crs=pt_utm_gdf.crs)
    crs84_gdf = utm_gdf.to_crs(crs84)
    poly_crs84 = crs84_gdf['geometry'].iloc[0]
    region_poly = kwimage.Polygon.from_shapely(poly_crs84)
    return region_poly


def random_time_sequence(min_date_iso, max_date_iso, num_observations, rng=None):
    """
    Generate a list of random timestamps between the specified dates

    Args:
        min_date_iso (str): minimum possible date
        max_date_iso (str): maximum possible date
        num_observations (int): number of dates to generate
        rng: random state or seed

    Returns:
        List[datetime_cls]: sampled dates in the UTC timezone

    Example:
        >>> from geowatch.demo.metrics_demo.demo_utils import *  # NOQA
        >>> min_date_iso = '1996-06-23'
        >>> max_date_iso = '2017-10-27'
        >>> num_observations = 3
        >>> obs_sequence = random_time_sequence(min_date_iso, max_date_iso, num_observations, rng=9320)
        >>> print('obs_sequence = {}'.format(ub.urepr(obs_sequence, nl=1)))
        obs_sequence = [
            datetime.datetime(1997, 9, 17, 20, 49, 7, 888653, tzinfo=datetime.timezone.utc),
            datetime.datetime(2001, 6, 29, 17, 43, 36, 108233, tzinfo=datetime.timezone.utc),
            datetime.datetime(2009, 4, 2, 10, 26, 54, 429200, tzinfo=datetime.timezone.utc),
        ]

    Ignore:
        # In Docker, test that the code provides the same number in different
        # locale.
        export TZ="Africa/Lusaka"
        rm -rf /etc/localtime
        ln -s /usr/share/zoneinfo/Africa/Lusaka /etc/localtime
        python -c "import datetime as datetime_mod; print(datetime_mod.datetime.fromisoformat('1996-06-23').replace(tzinfo=datetime_mod.timezone.utc).timestamp())"
    """
    rng = kwarray.ensure_rng(rng)
    from kwutil import util_time
    min_date = util_time.coerce_datetime(min_date_iso)
    max_date = util_time.coerce_datetime(max_date_iso)
    # min_date = datetime_cls.fromisoformat(min_date_iso)
    # max_date = datetime_cls.fromisoformat(max_date_iso)
    # # Assume user is specifying UTC by default.
    # if min_date.tzinfo is None:
    #     min_date = min_date.replace(tzinfo=datetime_mod.timezone.utc)
    # if max_date.tzinfo is None:
    #     max_date = max_date.replace(tzinfo=datetime_mod.timezone.utc)
    min_ts = min_date.timestamp()
    max_ts = max_date.timestamp()
    obs_timestamps = rng.rand(num_observations) * (max_ts - min_ts) + min_ts
    obs_sequence = [
        # util_time.coerce_datetime(ts)
        datetime_cls.fromtimestamp(ts, tz=datetime_mod.timezone.utc)
        for ts in sorted(obs_timestamps)
    ]
    return obs_sequence


def utm_epsg_from_latlon(lat, lon):
    """
    Find a reasonable UTM CRS for a given lat / lon

    The purpose of this function is to get a reasonable CRS for computing
    distances in meters. If the region of interest is very large, this may not
    be valid.

    See [SE190198]_ and [SE365584]_.

    Args:
        lat (float): degrees in latitude
        lon (float): degrees in longitude

    Returns:
        int : the ESPG code of the UTM zone

    References:
        .. [SE190198] https://gis.stackexchange.com/questions/190198/how-to-get-appropriate-crs-for-a-position-specified-in-lat-lon-coordinates
        .. [SE365584] https://gis.stackexchange.com/questions/365584/convert-utm-zone-into-epsg-code

    Example:
        >>> from geowatch.demo.metrics_demo.demo_utils import *  # NOQA
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


def project_gdf_to_local_utm(gdf_crs84, max_utm_zones=None):
    """
    Find the local UTM zone for a geo data frame and project to it.

    Assumes geometry is in CRS-84.

    All geometry in the GDF must be in the same UTM zone.

    Args:
        gdf_crs84 (geopandas.GeoDataFrame):
            The data with CRS-84 geometry to project into a local UTM

        max_utm_zones (int | None):
            If the data spans more than this many UTM zones, error.
            Otherwise, we take the first one.

    Returns:
        geopandas.GeoDataFrame

    Example:
        >>> import geopandas as gpd
        >>> import kwarray
        >>> import kwimage
        >>> rng = kwarray.ensure_rng(0)
        >>> # Gen lat/lons between 0 and 1, which is in UTM zone 31N
        >>> gdf_crs84 = gpd.GeoDataFrame({'geometry': [
        >>>     kwimage.Polygon.random(rng=rng).to_shapely(),
        >>>     kwimage.Polygon.random(rng=rng).to_shapely(),
        >>>     kwimage.Polygon.random(rng=rng).to_shapely(),
        >>> ]}, crs=get_crs84())
        >>> gdf_utm = project_gdf_to_local_utm(gdf_crs84)
        >>> assert gdf_utm.crs.name == 'WGS 84 / UTM zone 31N'

    Example:
        >>> import geopandas as gpd
        >>> import kwarray
        >>> import kwimage
        >>> # If the data is too big for a single UTM zone,
        >>> rng = kwarray.ensure_rng(0)
        >>> gdf_crs84 = gpd.GeoDataFrame({'geometry': [
        >>>     kwimage.Polygon.random(rng=rng).scale(90).to_shapely(),
        >>>     kwimage.Polygon.random(rng=rng).scale(90).to_shapely(),
        >>>     kwimage.Polygon.random(rng=rng).scale(90).to_shapely(),
        >>> ]}, crs=get_crs84())
        >>> import pytest
        >>> with pytest.raises(ValueError):
        >>>     gdf_utm = project_gdf_to_local_utm(gdf_crs84, max_utm_zones=1)
    """
    epsg_zones = []
    for geom_crs84 in gdf_crs84.geometry:
        epsg_zone = find_local_meter_epsg_crs(geom_crs84)
        epsg_zones.append(epsg_zone)
    if not ub.allsame(epsg_zones):
        unique_utm = set(epsg_zones)
        if max_utm_zones is not None and len(unique_utm) > max_utm_zones:
            raise ValueError(ub.paragraph(
                '''
                Input data spanned multiple UTM zones.
                This is currently not allowed. {}
                '''
            ).format(unique_utm))
    # TODO: if there are more than one is there a way to get "the best one?"
    epsg_zone = epsg_zones[0]
    gdf_utm = gdf_crs84.to_crs(epsg_zone)
    return gdf_utm


def find_local_meter_epsg_crs(geom_crs84):
    """
    Find the "best" meter based CRS for a smallish geographic region.

    Currently this only returns UTM zones. Might be better to return an Albers
    projection if the geometry spans more than one UTM zone.

    Args:
        geom_crs84 (shapely.geometry.base.BaseGeometry):
            shapely geometry in CRS84 (lon/lat wgs84)

    Returns:
        int: epsg code

    References:
        [1] https://gis.stackexchange.com/questions/148181/choosing-projection-crs-for-short-distance-based-analysis/148187
        [2] http://projfinder.com/

    TODO:
        - [ ] Better UTM zone intersection
        - [ ] Fix edge cases

    Example:
        >>> import kwimage
        >>> geom_crs84 = kwimage.Polygon.random().translate(-0.5).scale((180, 90)).to_shapely()
        >>> epsg_zone = find_local_meter_epsg_crs(geom_crs84)
    """
    lonmin, latmin, lonmax, latmax = geom_crs84.bounds
    # Hack: this doesnt work on boundries (or for larger regions)
    # correct way of doing this would be lookup candiate CRS zones,
    # and find the one with highest intersection area weighted by distance
    # to the center of the valid region.
    latmid = (latmin + latmax) / 2
    lonmid = (lonmin + lonmax) / 2
    candidate_utm_codes = [
        utm_epsg_from_latlon(latmin, lonmin),
        utm_epsg_from_latlon(latmax, lonmax),
        utm_epsg_from_latlon(latmax, lonmin),
        utm_epsg_from_latlon(latmin, lonmax),
        utm_epsg_from_latlon(latmid, lonmid),
    ]
    epsg_zone = ub.argmax(ub.dict_hist(candidate_utm_codes))
    return epsg_zone


@ub.memoize
def get_crs84():
    """
    Constructing the CRS84 is slow.
    This function memoizes it so it only happens once.

    Returns:
        pyproj.crs.crs.CRS
    """
    from pyproj import CRS
    crs84 = CRS.from_user_input('crs84')
    return crs84
