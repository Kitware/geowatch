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
from kwgis.utils.util_gis import utm_epsg_from_latlon  # NOQA
from kwgis.utils.util_gis import utm_epsg_from_zoneletter  # NOQA
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
