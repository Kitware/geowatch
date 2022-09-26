

def _crs_demo():
    """
    The axis mapping of CRS objects is absolutely FUBAR, nobody seems to care
    enough that there is no clear unambiguous way of doing this.

    EPSG:4326 *SHOULD* be wgs84 with lat/lon, but thats not always the case

    OCG:CRS84 actually is always wgs84 lon/lat, but gdal doesnt really support it

    Geopandas always uses x/y even when it's crs says that its not.
    This is absolutely insane.
    """
    from osgeo import osr
    # georef_crs_info['axis_mapping']
    # osr.OAMS_AUTHORITY_COMPLIANT
    # aux_wld_crs = osr.SpatialReference()
    # aux_wld_crs.ImportFromEPSG(4326)  # 4326 is the EPSG id WGS84 of lat/lon crs
    # aux_wld_crs.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
    # http://spatialreference.org/ref/epsg/26912/proj4/
    # urn:ogc:def:crs:OGC:1.3:CRS84
    # crs.ImportFromEPSG(4326)
    crs = osr.SpatialReference()
    crs.ImportFromProj4('+proj=longlat')
    print(crs.GetAxisMappingStrategy())
    print(crs.ExportToWkt())
    print(crs.ExportToProj4())

    crs = osr.SpatialReference()
    crs.ImportFromEPSG(4326)
    crs.SetAxisMappingStrategy(osr.OAMS_AUTHORITY_COMPLIANT)
    print(crs.GetAxisMappingStrategy())
    print(crs.ExportToWkt())
    print(crs.ExportToProj4())

    crs = osr.SpatialReference()
    crs.ImportFromEPSG(4326)
    crs.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
    print(crs.GetAxisMappingStrategy())
    print(crs.ExportToWkt())
    print(crs.ExportToProj4())

    import geopandas as gpd
    wld_map_gdf = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))

    wld_map_crs84 = wld_map_gdf.to_crs('crs84')
    wld_map_wgs84 = wld_map_gdf.to_crs('epsg:4326')
    print('wld_map_crs84.crs = {!r}'.format(wld_map_crs84.crs))
    print('wld_map_wgs84.crs = {!r}'.format(wld_map_wgs84.crs))
