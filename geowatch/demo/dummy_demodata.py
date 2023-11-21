def dummy_rpc_geotiff_fpath():
    """
    Create a blank tif with RPC information for testing
    """
    import rasterio
    import ubelt as ub
    from os.path import join
    from geowatch.gis import spatial_reference as watch_crs
    dpath = ub.Path.appdir('geowatch/demodata').ensuredir()
    gpath = join(dpath, 'test_rpc.tif')
    rpcs = watch_crs.RPCTransform.demo()

    with rasterio.open(gpath, 'w', driver='GTiff', dtype='uint8', count=1,
                       width=2000, height=2000, rpcs=rpcs.rpcs) as dst:
        dst,  # do nothing

    return gpath
