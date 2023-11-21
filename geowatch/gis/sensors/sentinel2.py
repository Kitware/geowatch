import ubelt as ub


def sentinel2_grid():
    """
    Grabs sentinel2 grid information from the web (if needed)

    Returns:
        geopandas.GeoDataFrame:
            A data frame where each row contains the tile name and the
            reversed-WGS84 (GeoJSON style) coordinates in
            shapely.geometry.Polygon format.

    References:
        https://gisgeography.com/arcgis-shapefile-files-types-extensions/
        https://github.com/justinelliotmeyers/Sentinel-2-Shapefile-Index

    CommandLine:
        xdoctest -m /home/joncrall/code/watch/geowatch/gis/sensors/sentinel2.py

    Example:
        >>> # xdoctest: +SKIP("slow, not currently used")
        >>> from geowatch.gis.sensors.sentinel2 import *  # NOQA
        >>> s2_tiles = sentinel2_grid()
        >>> assert s2_tiles.crs.name == 'WGS 84'
        >>> # Print out the first 5 tile rows
        >>> # print(s2_tiles.iloc[0:5])  # xdoctest: +IGNORE_WANT
        >>> '''
        >>>     Name                                           geometry
        >>> 0  01CCV  POLYGON Z ((180.00000 -73.05974 0.00000, 176.8...
        >>> 1  01CCV  POLYGON Z ((-180.00000 -72.07333 0.00000, -179...
        >>> 2  01CDH  POLYGON Z ((180.00000 -83.80855 0.00000, 174.7...
        >>> 3  01CDH  POLYGON Z ((-180.00000 -82.82590 0.00000, -176...
        >>> 4  01CDJ  POLYGON Z ((180.00000 -82.91344 0.00000, 175.7...
        >>> '''
        >>> #
        >>> # Demo how to convert each polygon into its UTM zone
        >>> import kwimage
        >>> from geowatch.utils import util_gis
        >>> import pyproj
        >>> utm_codes = []
        >>> # Only take some of the tiles for test speed
        >>> s2_tiles = s2_tiles.iloc[0:100]
        >>> for poly in ub.ProgIter(s2_tiles.geometry, desc='find utm'):
        >>>     lon = poly.centroid.x
        >>>     lat = poly.centroid.y
        >>>     utm_epsg = util_gis.utm_epsg_from_latlon(lat, lon)
        >>>     utm_codes.append(utm_epsg)
        >>> s2_tiles['utm_epsg'] = utm_codes
        >>> # Group all tiles within the same zone together
        >>> groups = dict(list(s2_tiles.groupby('utm_epsg')))
        >>> utm_groups = {}
        >>> for utm_epsg in ub.ProgIter(groups, desc='proj to utm'):
        >>>     utm_crs = pyproj.CRS.from_epsg(utm_epsg)
        >>>     # convert group into its utm coordinates
        >>>     utm_group = groups[utm_epsg].to_crs(utm_crs)
        >>>     utm_groups[utm_epsg] = utm_group
        >>> # Measure the area of each UTM Polygon
        >>> all_utm_areas = []
        >>> for utm_epsg, utm_group in utm_groups.items():
        >>>     all_utm_areas.extend([s.area for s in utm_group.geometry])
        >>> # Print out statistics about the chosen S2-tile UTM areas
        >>> import kwarray
        >>> area_stats = kwarray.stats_dict(all_utm_areas)
        >>> print('area_stats = {}'.format(ub.urepr(
        >>>     area_stats, nl=1, precision=0, align=':')))

        ...
        area_stats = {
            'mean' : 8463072768,
            'std'  : 3774869504,
            'min'  : 65260336,
            'max'  : 12056040448,
            ...
            'shape': (100,),
        }
    """
    from os.path import join
    import geopandas as gpd
    base_url = 'https://raw.githubusercontent.com/justinelliotmeyers/Sentinel-2-Shapefile-Index/9c553c340ee0b04f4d0c54c6f4d6c04504067dc6'
    items = {
        # Contains main geometry information
        'shp': {
            'fname': 'sentinel_2_index_shapefile.shp',
            'hash_prefix': '5343d73fa84e81cdcef03da86f4d21087f52cf41e4776aab1'},

        # contains name information
        'dbf': {
            'fname': 'sentinel_2_index_shapefile.dbf',
            'hash_prefix': 'ca65ffe251c0a1c4c0a27bf9aa915c475a56a4d6fb7fa465b'},

        # Contains CRS projection information
        'prj': {
            'fname': 'sentinel_2_index_shapefile.prj',
            'hash_prefix': 'b3ddd6fbfb5cca02d69acf67f986a7d1bd0400d72db141b61'},

        # Speeds up spatial queries
        'sbn': {
            'fname': 'sentinel_2_index_shapefile.sbn',
            'hash_prefix': 'f0f0118be76b3c53f44613f85d68c02d0b16f019b0c7630d3'},

        # Speeds up spatial queries
        'sbx': {
            'fname': 'sentinel_2_index_shapefile.sbx',
            'hash_prefix': 'ffe9c1dd8b55af0bc41f6cb354d4e8d107133b2d4e6f40e53'},

        # Required for spatial information?
        'shx': {
            'fname': 'sentinel_2_index_shapefile.shx',
            'hash_prefix': '4574fe1fe1be31e212b9c2ce0b09a162ccd518f67ccd455aa'},
    }
    for item in items.values():
        url = join(base_url, item['fname'])
        fpath = ub.grabdata(url, appname='geowatch', hash_prefix=item['hash_prefix'])
        item['fpath'] = fpath
    fpath = items['shp']['fpath']
    s2_tiles = gpd.read_file(fpath)
    return s2_tiles


def s2_grid_tiles_for_geometry(geometry):
    """
    Takes in a GeoJSON geometry and returns MGRS tile names that
    intersect it, ordered by intersection area (descending)

    Returns:
        List of strings:
            List of MGRS tile names

    Example:
        >>> # xdoctest: +SKIP("slow, not currently used")
        >>> from geowatch.gis.sensors.sentinel2 import *  # NOQA
        >>> from shapely.geometry import shape
        >>> geometry = shape({
        >>>  "type": "Polygon",
        >>>  "coordinates": [
        >>>     [
        >>>       [
        >>>         126.84333152242021,
        >>>         38.51998197874137
        >>>       ],
        >>>       [
        >>>         129.5107776938883,
        >>>         38.53873352863998
        >>>       ],
        >>>       [
        >>>         129.49645974534624,
        >>>         36.40500129696213
        >>>       ],
        >>>       [
        >>>         126.90373488362242,
        >>>         36.38763332541521
        >>>       ],
        >>>       [
        >>>         126.84333152242021,
        >>>         38.51998197874137
        >>>       ]
        >>>     ]
        >>>   ]
        >>>  })
        >>> assert(s2_grid_tiles_for_geometry(geometry) ==
        >>>        ['52SDG', '52SCG', '52SDH', '52SDF', '52SCH', '52SCF', '52SEG', '52SEH', '52SEF'])
    """

    s2_tiles = sentinel2_grid()

    # TODO: Convert geometries to UTM for more appropriate 'area'
    # computation

    s2_tiles['intersection_area'] =\
        s2_tiles.geometry.intersection(geometry).area

    return s2_tiles[s2_tiles['intersection_area'] > 0].sort_values(
        'intersection_area', ascending=False)['Name'].tolist()
