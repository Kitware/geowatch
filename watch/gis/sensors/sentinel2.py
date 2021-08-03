import ubelt as ub


def sentinel2_grid():
    """
    Grabs sentinel2 grid information from the web if needed

    Returns:
        geopandas.geodataframe.GeoDataFrame

    References:
        https://gisgeography.com/arcgis-shapefile-files-types-extensions/
        https://github.com/justinelliotmeyers/Sentinel-2-Shapefile-Index

    Example:
        >>> from watch.gis.sensors.sentinel2 import *  # NOQA
        >>> s2_tiles = sentinel2_grid()
        >>> assert s2_tiles.crs.name == 'WGS 84'
        >>> print(s2_tiles.iloc[0:5])
            Name                                           geometry
        0  01CCV  POLYGON Z ((180.000 -73.060 0.000, 176.865 -72...
        1  01CCV  POLYGON Z ((-180.000 -72.073 0.000, -179.627 -...
        2  01CDH  POLYGON Z ((180.000 -83.809 0.000, 174.713 -83...
        3  01CDH  POLYGON Z ((-180.000 -82.826 0.000, -176.297 -...
        4  01CDJ  POLYGON Z ((180.000 -82.913 0.000, 175.748 -82...
    """
    from os.path import join
    import geopandas as gpd
    base_url = 'https://raw.githubusercontent.com/justinelliotmeyers/Sentinel-2-Shapefile-Index/9c553c340ee0b04f4d0c54c6f4d6c04504067dc6'
    items = [
        # Contains main geometry information
        {'fname': 'sentinel_2_index_shapefile.shp',
         'hash_prefix': '5343d73fa84e81cdcef03da86f4d21087f52cf41e4776aab1b21'},

        # contains name information
        {'fname': 'sentinel_2_index_shapefile.dbf',
         'hash_prefix': 'ca65ffe251c0a1c4c0a27bf9aa915c475a56a4d6fb7fa465b2f8'},

        # Contains CRS projection information
        {'fname': 'sentinel_2_index_shapefile.prj',
         'hash_prefix': 'b3ddd6fbfb5cca02d69acf67f986a7d1bd0400d72db141b610a'},

        # Speeds up spatial queries
        {'fname': 'sentinel_2_index_shapefile.sbn',
         'hash_prefix': 'f0f0118be76b3c53f44613f85d68c02d0b16f019b0c7630d3aa'},

        # Speeds up spatial queries
        {'fname': 'sentinel_2_index_shapefile.sbx',
         'hash_prefix': 'ffe9c1dd8b55af0bc41f6cb354d4e8d107133b2d4e6f40e5387'},

        # Required for spatial information?
        {'fname': 'sentinel_2_index_shapefile.shx',
         'hash_prefix': '4574fe1fe1be31e212b9c2ce0b09a162ccd518f67ccd455aa0'},
    ]
    for item in items:
        url = join(base_url, item['fname'])
        fpath = ub.grabdata(url, appname='watch', hash_prefix=item['hash_prefix'])
        item['fpath'] = fpath
    fpath = items[0]['fpath']
    s2_tiles = gpd.read_file(fpath)
    return s2_tiles
