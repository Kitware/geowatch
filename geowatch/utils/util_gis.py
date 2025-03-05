"""
DEPRECATED: moved to kwgis

Utilities for geopandas and other geographic information system tools
"""
from kwgis.utils.util_gis import plot_geo_background  # NOQA
from kwgis.utils.util_gis import geopandas_pairwise_overlaps  # NOQA
from kwgis.utils.util_gis import latlon_text  # NOQA
from kwgis.utils.util_gis import demo_regions_geojson_text  # NOQA
from kwgis.utils.util_gis import load_geojson  # NOQA
from kwgis.utils.util_gis import read_geojson  # NOQA
from kwgis.utils.util_gis import get_crs84  # NOQA
from kwgis.utils.util_gis import _get_crs84  # NOQA
from kwgis.utils.util_gis import _flip  # NOQA
from kwgis.utils.util_gis import shapely_flip_xy  # NOQA
from kwgis.utils.util_gis import project_gdf_to_local_utm  # NOQA
from kwgis.utils.util_gis import UTM_TransformContext  # NOQA
from kwgis.utils.util_gis import _demo_convert_latlon_to_utm  # NOQA
from kwgis.utils.util_gis import find_local_meter_epsg_crs  # NOQA
from kwgis.utils.util_gis import check_latlons  # NOQA
from kwgis.utils.util_gis import coerce_geojson_datas  # NOQA
from kwgis.utils.util_gis import coerce_geojson_paths  # NOQA
from kwgis.utils.util_gis import _coerce_raw_geojson  # NOQA
from kwgis.utils.util_gis import _load_json_from_path  # NOQA
from kwgis.utils.util_gis import load_geojson_datas  # NOQA
from kwgis.utils.util_gis import crs_geojson_to_gdf  # NOQA
from kwgis.utils.util_gis import coerce_crs  # NOQA
from kwgis.utils.util_gis import utm_epsg_from_latlon  # NOQA


def utm_epsg_from_zoneletter(zone, letter):
    """
    Example:
        >>> lon, lat = -99.1149386165299831, 19.4867151641771414
        >>> espg1 = utm_epsg_from_latlon(lat, lon)
        >>> zone = 14
        >>> letter = 'Q'
        >>> espg2 = utm_epsg_from_zoneletter(zone, letter)
        >>> assert espg2 == espg1

    References:
        https://www.maptools.com/tutorials/grid_zone_details

    Notes:
        The UTM coordinate system divides the earth into 60 zones each 6
        degrees of longitude wide. These zones define the reference point for
        UTM grid coordinates within the zone. UTM zones extend from a latitude
        of 80° S to 84° N. In the polar regions the Universal Polar
        Stereographic (UPS) grid system is used. Note that there are a few
        exceptions to zone width in Northern Europe to keep small countries in
        a single zone.

        UTM zones are numbered 1 through 60, starting at the international date
        line, longitude 180°, and proceeding east. Zone 1 extends from 180° W
        to 174° W and is centered on 177° W.

        Each zone is divided into horizontal bands spanning 8 degrees of
        latitude. These bands are lettered, south to north, beginning at 80° S
        with the letter C and ending with the letter X at 84° N. The letters I
        and O are skipped to avoid confusion with the numbers one and zero. The
        band lettered X spans 12° of latitude.
    """
    if letter.upper() >= 'N':
        epsg_code = 32600 + int(zone)  # Northern hemisphere
    else:
        epsg_code = 32700 + int(zone)  # Southern hemisphere
    return epsg_code
