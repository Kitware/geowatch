"""
Tools to work with geotiff metadata.
"""
import numpy as np
from watch.gis import spatial_reference as watch_crs


def demodata_rpc_geotiff():
    """
    Create a tif with RPC information for testing
    """
    import rasterio
    import ubelt as ub
    from os.path import join
    dpath = ub.ensure_app_cache_dir('smart-watch/demodata')
    gpath = join(dpath, 'test_rpc.tif')
    rpcs = watch_crs.RPCTransform.demo()

    with rasterio.open(gpath, 'w', driver='GTiff', dtype='uint8', count=1,
                       width=2000, height=2000, rpcs=rpcs.rpcs) as dst:
        dst,  # do nothing

    return gpath


def geotiff_crs_info(gpath, force_affine=False, elevation='open-elevation',
                     verbose=0):
    """
    Use GDAL to extract coordinate system information about a geo_tiff.

    Builds transformations between pixel, geotiff-world, utm, and wgs84 spaces

    Args:
        gpath (str): path to the image file
        force_affine (bool): if True ignores RPC information
        elevation (str): method used to determine the elevation when RPC
            information is used. Currently only "open-elevation" is available.

    Example:
        >>> from watch.gis.geotiff import *  # NOQA
        >>> import ubelt as ub
        >>> gpath = demodata_rpc_geotiff()
        >>> info = geotiff_crs_info(gpath)
        >>> print('info = {}'.format(ub.repr2(info, nl=1, sort=False)))
        >>> assert info['is_rpc']
        >>> assert info['img_shape'] == (2000, 2000)

        >>> # xdoctest: +REQUIRES(--network)
        >>> gpath = ub.grabdata(
        >>>     'https://download.osgeo.org/geotiff/samples/gdal_eg/cea.tif',
        >>>     appname='smart-watch/demodata', hash_prefix='10a2ebcdcd95582')
        >>> info = geotiff_crs_info(gpath)
        >>> print('info = {}'.format(ub.repr2(info, nl=1, sort=False)))
        >>> assert not info['is_rpc']
        >>> assert info['img_shape'] == (515, 514)

        >>> from watch.demo.nitf_demodata import grab_nitf_fpath
        >>> gpath = grab_nitf_fpath('i_3004g.ntf')
        >>> info = geotiff_crs_info(gpath)
        >>> print('info = {}'.format(ub.repr2(info, nl=1, sort=False)))
        >>> assert not info['is_rpc']
        >>> assert info['img_shape'] == (512, 512)
    """
    import gdal
    import osr
    import affine
    import kwimage
    ref = gdal.Open(gpath, gdal.GA_ReadOnly)

    info = {}

    if ref is None:
        raise Exception('gpath={} is not an image file'.format(gpath))

    wgs84_crs = osr.SpatialReference()
    wgs84_crs.ImportFromEPSG(4326)  # 4326 is the EPSG id WGS84 of lat/lon crs

    tags = ref.GetMetadataDomainList()
    if 'RPC' in tags:
        rpc_info = ref.GetMetadata(domain='RPC')
        rpc_transform = watch_crs.RPCTransform.from_gdal(
            rpc_info, elevation=elevation)
        approx_elevation = rpc_transform._default_elevation()
    else:
        rpc_transform = None
        approx_elevation = None

    # TODO: understand the conditions for when these will not be populated
    # will proj always exist if wld_crs exists?
    wld_crs = ref.GetSpatialRef()
    proj = ref.GetProjection()
    if wld_crs is None or proj == '':
        ref = gdal.Open(gpath, gdal.GA_ReadOnly)
        gcps = ref.GetGCPs()
        wld_crs = ref.GetGCPSpatialRef()
        wld_crs_type = 'gcp'  # mark the wld crs as coming from the gcp
        if len(gcps) == 0 or wld_crs is None:
            if rpc_transform is not None:
                wld_crs = osr.SpatialReference()
                wld_crs.ImportFromEPSG(4326)  # 4326 is the EPSG id WGS84 of lat/lon crs
                # Set traditional because our rpc transform returns x,y
                wld_crs.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
                # Not 100% sure this is correct to do
                wld_crs_type = 'assume-rpc-wgs84-reverse'
                _geo_transform = (0.0, 1.0, 0.0, 0.0, 0.0, 1.0)
                if force_affine:
                    raise Exception(
                        'cant force affine without dataset or gcp ref')
            else:
                raise Exception('no dataset or gcps refs or rpc')
        else:
            _geo_transform = gdal.GCPsToGeoTransform(gcps)
    else:
        if wld_crs is None:
            wld_crs = osr.SpatialReference()
            wld_crs.ImportFromWkt(proj)
        else:
            # mark the wld crs as coming from the dataset
            # is there a better name for this?
            wld_crs_type = 'dataset'
        _geo_transform = ref.GetGeoTransform()

    _aff = affine.Affine.from_gdal(*_geo_transform)
    _aff_pxl_to_wld = np.vstack([np.array(_aff.column_vectors).T, [0, 0, 1]])
    _aff_wld_to_pxl = np.linalg.inv(_aff_pxl_to_wld)

    def axis_mapping_int_to_text(axis_mapping_int):
        """
        References:
            https://gdal.org/tutorials/osr_api_tut.html#crs-and-axis-order

        Notes:
            * OAMS_TRADITIONAL_GIS_ORDER means that for geographic CRS with
                lat/long order, the data will still be long/lat ordered.
                Similarly for a projected CRS with northing/easting order, the
                data will still be easting/northing ordered

            * OAMS_AUTHORITY_COMPLIANT means that the data axis will be
                identical to the CRS axis. This is the default value when
                instantiating OGRSpatialReference

            * OAMS_CUSTOM means that the data axis are customly defined with
                SetDataAxisToSRSAxisMapping
        """
        if axis_mapping_int == osr.OAMS_TRADITIONAL_GIS_ORDER:
            axis_mapping = 'OAMS_TRADITIONAL_GIS_ORDER'
        elif axis_mapping_int == osr.OAMS_AUTHORITY_COMPLIANT:
            axis_mapping = 'OAMS_AUTHORITY_COMPLIANT'
        elif axis_mapping_int == osr.OAMS_CUSTOM:
            axis_mapping = 'OAMS_CUSTOM'
        else:
            raise KeyError(axis_mapping_int)
        return axis_mapping

    wld_axis_mapping_int = wld_crs.GetAxisMappingStrategy()
    wld_axis_mapping = axis_mapping_int_to_text(wld_axis_mapping_int)

    wgs84_axis_mapping_int = wgs84_crs.GetAxisMappingStrategy()
    wgs84_axis_mapping = axis_mapping_int_to_text(wgs84_axis_mapping_int)

    wld_to_wgs84 = osr.CoordinateTransformation(wld_crs, wgs84_crs)
    wgs84_to_wld = osr.CoordinateTransformation(wgs84_crs, wld_crs)

    if force_affine or rpc_transform is None:
        is_rpc = False
        pxl_to_wld = _aff_pxl_to_wld
        wld_to_pxl = _aff_wld_to_pxl
    else:
        is_rpc = True
        wld_to_pxl = rpc_transform.warp_world_to_pixel
        pxl_to_wld = rpc_transform.warp_pixel_to_world

    shape = (ref.RasterYSize, ref.RasterXSize)

    # warp pixel corner points to lat / lon
    xy_corners = np.array([
        [0, 0],
        [0, ref.RasterYSize],
        [ref.RasterXSize, 0],
        [ref.RasterXSize, ref.RasterYSize],
    ])
    pxl_corners = kwimage.Coords(xy_corners)
    wld_corners = pxl_corners.warp(pxl_to_wld)

    wgs84_corners = wld_corners.warp(wld_to_wgs84)
    lat1 = wgs84_corners.data[:, 0].min()
    lat2 = wgs84_corners.data[:, 0].max()
    lon1 = wgs84_corners.data[:, 1].min()
    lon2 = wgs84_corners.data[:, 1].max()
    min_lon, max_lon = sorted([lon1, lon2])
    min_lat, max_lat = sorted([lat1, lat2])

    assert watch_crs.check_latlons(
        wgs84_corners.data[:, 0], wgs84_corners.data[:, 1]), (
            'bad WGS84 coordinates'
        )

    WITH_UTM_INFO = True
    if WITH_UTM_INFO:
        lat, lon = min_lat, min_lon
        epsg_int = watch_crs.utm_epsg_from_latlon(lat, lon)
        utm_crs = osr.SpatialReference()
        utm_crs.ImportFromEPSG(epsg_int)

        wld_to_utm = osr.CoordinateTransformation(wld_crs, utm_crs)
        utm_to_wld = osr.CoordinateTransformation(utm_crs, wld_crs)
        utm_corners = wld_corners.warp(wld_to_utm)

        min_utm = utm_corners.data.min(axis=0)
        max_utm = utm_corners.data.max(axis=0)
        meter_extent = max_utm - min_utm
        pxl_extent = np.array([ref.RasterXSize, ref.RasterYSize])
        meter_per_pxl = meter_extent / pxl_extent

        utm_axis_mapping_int = utm_crs.GetAxisMappingStrategy()
        utm_axis_mapping = axis_mapping_int_to_text(utm_axis_mapping_int)
    else:
        meter_per_pxl = None
        meter_extent = None
        wld_to_utm = None
        utm_to_wld = None
        utm_crs = None
        utm_axis_mapping = None

    if verbose:
        print('pxl_corners = {!r}'.format(pxl_corners))
        print('wld_corners = {!r}'.format(wld_corners))
        print('wgs84_corners = {!r}'.format(wgs84_corners))
        print('LAT: {} - {}'.format(min_lat, max_lat))
        print('LON: {} - {}'.format(min_lon, max_lon))

    bbox_geos = [min_lat, min_lon, max_lat, max_lon]

    if 1:
        # Is this a reasonable thing to do?
        from pyproj import CRS
        utm_crs = utm_crs.ExportToWkt()
        utm_crs = CRS.from_wkt(utm_crs).to_authority(min_confidence=100)

        wld_crs = wld_crs.ExportToWkt()
        wld_crs = CRS.from_wkt(wld_crs).to_authority(min_confidence=100)

        wgs84_crs = wgs84_crs.ExportToWkt()
        wgs84_crs = CRS.from_wkt(wgs84_crs).to_authority(min_confidence=100)

    info.update({
        'is_rpc': is_rpc,

        'meter_per_pxl': meter_per_pxl,
        'meter_extent': meter_extent,
        'approx_elevation': approx_elevation,

        'pxl_corners': pxl_corners,
        'utm_corners': utm_corners,
        'wld_corners': wld_corners,
        'wgs84_corners': wgs84_corners,

        'pxl_to_wld': pxl_to_wld,
        'wgs84_to_wld': wgs84_to_wld,
        'utm_to_wld': utm_to_wld,

        'wld_to_pxl': wld_to_pxl,
        'wld_to_wgs84': wld_to_wgs84,
        'wld_to_utm': wld_to_utm,

        'wld_axis_mapping': wld_axis_mapping,
        'wgs84_axis_mapping': wgs84_axis_mapping,
        'utm_axis_mapping': utm_axis_mapping,

        'utm_crs': utm_crs,
        'wld_crs': wld_crs,
        'wgs84_crs': wgs84_crs,

        'wld_crs_type': wld_crs_type,

        'bbox_geos': bbox_geos,
        'img_shape': shape,

        'gpath': gpath,
    })
    return info
