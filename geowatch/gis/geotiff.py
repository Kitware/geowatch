"""
Tools to work with geotiff metadata.
"""
import numpy as np
import ubelt as ub
from geowatch.gis import spatial_reference as watch_crs
# from geowatch.utils.util_bands import LANDSAT7
from geowatch.utils import util_gis
from geowatch.utils.util_bands import SENTINEL2, LANDSAT8
import parse
from os.path import basename, isfile
from dateutil.parser import isoparse
from geowatch import exceptions


def geotiff_metadata(gpath, elevation='gtop30', strict=False,
                     supress_warnings=False):
    """
    Extract all relevant metadata we know how to extract.

    Args:
        gpath (str): path to the geotiff of interest
        elevation (str): method for extracting elevation data.

    Example:
        >>> # xdoctest: +REQUIRES(env:SLOW_DOCTEST)
        >>> # xdoctest: +REQUIRES(--network)
        >>> from geowatch.gis.geotiff import *  # NOQA
        >>> url = ('http://storage.googleapis.com/gcp-public-data-landsat/'
        ...        'LC08/01/044/034/LC08_L1GT_044034_20130330_20170310_01_T2/'
        ...        'LC08_L1GT_044034_20130330_20170310_01_T2_B11.TIF')
        >>> gpath = ub.grabdata(url, appname='geowatch')
        >>> info = geotiff_metadata(gpath)
        >>> print('info = {}'.format(ub.urepr(info, nl=1)))

        >>> import ubelt as ub
        >>> url = ('http://storage.googleapis.com/gcp-public-data-landsat/'
        ...        'LC08/01/037/029/LC08_L1TP_037029_20130602_20170310_01_T1/'
        ...        'LC08_L1TP_037029_20130602_20170310_01_T1_B2.TIF')
        >>> gpath = ub.grabdata(url, appname='geowatch')

        >>> info = geotiff_metadata(gpath)
        >>> print('info = {}'.format(ub.urepr(info, nl=1)))


    Ignore:
        source secrets/secrets
        export AWS_PROFILE=iarpa

        export XDEV_PROFILE=1
        pyblock "
        gpath = '/vsis3/smart-data-accenture/ta-1/ta1-wv-acc/52/S/DG/2020/4/13/20APR13020407-M1BS-014418171010_01_P004_ACC/20APR13020407-M1BS-014418171010_01_P004_ACC_B05.tif'
        from geowatch.gis.geotiff import geotiff_metadata
        info = geotiff_metadata(gpath)
        print(info)
        "
    """
    from geowatch.utils import util_gdal
    infos = {}
    ref = util_gdal.GdalDataset.open(gpath, 'r', virtual_retries=3)

    if supress_warnings:
        context = util_gdal.GdalSupressWarnings()
    else:
        import contextlib
        context = contextlib.nullcontext()

    infos['fname'] = geotiff_filepath_info(gpath)
    try:
        # TODO: we probably shouldn't suppress warnings here, remove once we
        # figure out why we are getting the current ones.
        with context:
            infos['crs'] = geotiff_crs_info(ref, elevation=elevation)
    except exceptions.GeoMetadataNotFound as ex:
        if strict:
            raise
        infos['crs'] = {'crs_error': str(ex)}
    infos['header'] = geotiff_header_info(ref)

    # Combine sensor candidates
    sensor_candidates = list(ub.flatten([
        v.get('sensor_candidates', []) for v in infos.values()]))

    info = ub.dict_union(*infos.values())
    info['sensor_candidates'] = sensor_candidates
    return info


def geotiff_header_info(gpath_or_ref):
    """
    Extract relevant metadata information from a geotiff header.

    Example:
        >>> # xdoctest: +REQUIRES(env:SLOW_DOCTEST)
        >>> from geowatch.gis.geotiff import *  # NOQA
        >>> from geowatch.demo.dummy_demodata import dummy_rpc_geotiff_fpath
        >>> from geowatch.demo.landsat_demodata import grab_landsat_product
        >>> gpath_or_ref = gpath = dummy_rpc_geotiff_fpath()
        >>> info = geotiff_header_info(gpath)
        >>> print('info = {}'.format(ub.urepr(info, nl=1)))
        >>> gpath_or_ref = gpath = grab_landsat_product()['bands'][0]
        >>> info = geotiff_header_info(gpath)
        >>> print('info = {}'.format(ub.urepr(info, nl=1)))
    """
    from geowatch.utils import util_gdal
    ref = util_gdal.GdalDataset.coerce(gpath_or_ref)
    keys_of_interest = [
        'NITF_CSEXRA_MAX_GSD',
        'NITF_PIAIMC_MEANGSD',
        'NITF_ISORCE',
        'NITF_PIAIMC_SENSNAME',
        'NITF_USE00A_MEAN_GSD',
        'NITF_CSEXRA_GEO_MEAN_GSD',
    ]
    sensor_candidates = []
    ignore_domains = {
        'xml:TRE', 'NITF_METADATA', 'DERIVED_SUBDATASETS', 'IMAGE_STRUCTURE',
        'TRE'}
    all_domain_img_meta = {}
    for ns in ref.GetMetadataDomainList():
        if ns in ignore_domains:
            continue
        meta = ref.GetMetadata(ns)
        if 0:
            print('meta = {}'.format(ub.urepr(meta, nl=1)))
        for key, value in ub.dict_isect(meta, keys_of_interest).items():
            all_domain_img_meta['{}.{}'.format(ns, key)] = value
    img_info = {}
    img_info['img_meta' ] = all_domain_img_meta
    img_info['num_bands'] = ref.RasterCount
    img_info['sensor_candidates'] = sensor_candidates

    band_metas = []
    for i in range(1, ref.RasterCount + 1):
        band = ref.GetRasterBand(i)
        # band.ComputeBandStats()
        if 0:
            band.GetStatistics(0, 1)
        band_meta_domains = band.GetMetadataDomainList() or []
        all_domain_band_meta = {}
        all_domain_band_meta['nodata'] = band.GetNoDataValue()
        for ns in band_meta_domains:
            band_meta = band.GetMetadata(ns)
            # band_meta = ub.dict_diff(band_meta, {'COMPRESSION'})
            for key, value in band_meta.items():
                all_domain_band_meta['{}.{}'.format(ns, key)] = value
        band_metas.append(all_domain_band_meta)

    if band_metas and any(band_metas):
        img_info['band_metas'] = band_metas

    if '.NITF_PIAIMC_SENSNAME' in all_domain_img_meta:
        sensor_candidates.append(all_domain_img_meta['.NITF_PIAIMC_SENSNAME'])
    return img_info


def geotiff_crs_info(gpath_or_ref, force_affine=False,
                     elevation='gtop30', verbose=0):
    """
    Use GDAL to extract coordinate system information about a geo_tiff.

    Builds transformations between pixel, geotiff-world, utm, and wgs84 spaces

    Args:
        gpath (str): path to the image file
        force_affine (bool): if True ignores RPC information
        elevation (str): method used to determine the elevation when RPC
            information is used. Available options are:
                * "open-elevation"
                * "gtop30"

    Example:
        >>> # xdoctest: +REQUIRES(env:SLOW_DOCTEST)
        >>> from geowatch.gis.geotiff import *  # NOQA
        >>> from geowatch.demo.dummy_demodata import dummy_rpc_geotiff_fpath
        >>> gpath_or_ref = gpath = dummy_rpc_geotiff_fpath()
        >>> info = geotiff_crs_info(gpath)
        >>> print('info = {}'.format(ub.urepr(info, nl=1, sort=False)))
        >>> assert info['is_rpc']
        >>> assert info['img_shape'] == (2000, 2000)

        >>> # xdoctest: +REQUIRES(--network)
        >>> gpath_or_ref = gpath = ub.grabdata(
        >>>     'https://download.osgeo.org/geotiff/samples/gdal_eg/cea.tif',
        >>>     appname='geowatch/demodata', hash_prefix='10a2ebcdcd95582')
        >>> info = geotiff_crs_info(gpath)
        >>> print('info = {}'.format(ub.urepr(info, nl=1, sort=False)))
        >>> assert not info['is_rpc']
        >>> assert info['img_shape'] == (515, 514)

        >>> # The public gateways seem to be too slow in serving the content
        >>> # xdoctest: +REQUIRES(--ipfs)
        >>> from geowatch.demo.nitf_demodata import grab_nitf_fpath
        >>> gpath_or_ref = gpath = grab_nitf_fpath('i_3004g.ntf')
        >>> info = geotiff_crs_info(gpath)
        >>> print('info = {}'.format(ub.urepr(info, nl=1, sort=False)))
        >>> assert not info['is_rpc']
        >>> assert info['img_shape'] == (512, 512)

        tf = info['wgs84_to_wld']
    """
    import affine
    import kwimage
    from geowatch.utils import util_gdal
    gdal = util_gdal.import_gdal()
    osr = util_gdal.import_osr()

    info = {}
    ref = util_gdal.GdalDataset.coerce(gpath_or_ref)

    if 0:
        # TODO: is it more efficient to use a gdal info call?  Do we get all
        # the information we need from it so we can serialize the data?
        json_info = gdal.Info(ref, format='json')
        json_info['coordinateSystem']
        aff_geo_transform = json_info['geoTransform']

        aff_wld_crs = osr.SpatialReference()
        aff_wld_crs.ImportFromWkt(json_info['coordinateSystem']['wkt'])
        # Not sure about how to import this info best
        if json_info['coordinateSystem']['dataAxisToSRSAxisMapping'] == [1, 2]:
            aff_wld_crs.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
        else:
            aff_wld_crs.SetAxisMappingStrategy(osr.OAMS_AUTHORITY_COMPLIANT)

    # tags = ref.GetMetadataDomainList()  # 7.5% of the execution time
    rpc_info = ref.GetMetadata(domain='RPC')  # 5% of execution time

    # if 'RPC' in tags:
    if len(rpc_info):
        rpc_info = ref.GetMetadata(domain='RPC')
        rpc_transform = watch_crs.RPCTransform.from_gdal(
            rpc_info, elevation=elevation)
        approx_elevation = rpc_transform._default_elevation()
    else:
        rpc_transform = None
        approx_elevation = None

    # TODO: understand the conditions for when these will not be populated
    # will proj always exist if wld_crs exists?
    if not hasattr(ref, 'GetSpatialRef'):
        import warnings
        warnings.warn('ref has no attribute GetSpatialRef, gdal version issue?')
        raise AssertionError('ref has no attribute GetSpatialRef, gdal version issue?')
        aff_wld_crs = None
    else:
        aff_wld_crs = ref.GetSpatialRef()  # 20% of the execution time
    aff_proj = ref.GetProjection()

    gcps = ref.GetGCPs()
    aff_gcp_wld_crs = ref.GetGCPSpatialRef()

    if gcps is not None:
        gcp_wld_coords = kwimage.Coords(np.array([[p.GCPX, p.GCPY] for p in gcps]))
        gcp_pxl_coords = kwimage.Coords(np.array([[p.GCPPixel, p.GCPLine] for p in gcps]))
    else:
        gcp_wld_coords = None
        gcp_pxl_coords = None

    if aff_wld_crs is None or aff_proj == '':
        aff_wld_crs = aff_gcp_wld_crs
        aff_wld_crs_type = 'gcp'  # mark the aff_wld crs as coming from the gcp
        if len(gcps) == 0 or aff_wld_crs is None:
            if rpc_transform is not None:
                # raise AssertionError('I dont think this should be possible')
                # Oh but it is, tests hit it.
                aff_wld_crs = osr.SpatialReference()
                aff_wld_crs.ImportFromEPSG(4326)  # 4326 is the EPSG id WGS84 of lat/lon crs
                # Set traditional because our rpc transform returns x,y
                aff_wld_crs.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
                # Not 100% sure this is correct to do
                aff_wld_crs_type = 'assume-rpc-wgs84-reverse'
                aff_geo_transform = (0.0, 1.0, 0.0, 0.0, 0.0, 1.0)
                if force_affine:
                    raise exceptions.GeoMetadataNotFound(
                        'cant force affine without dataset or gcp ref')
            else:
                raise exceptions.GeoMetadataNotFound('no dataset or gcps refs or rpc')
        else:
            # gcp_ids = [p.Id for p in gcps]
            aff_geo_transform = gdal.GCPsToGeoTransform(gcps)
    else:
        if aff_wld_crs is None:
            aff_wld_crs = osr.SpatialReference()
            aff_wld_crs.ImportFromWkt(aff_proj)
            aff_wld_crs_type = 'unknown?affine-projection-maybe?'
            raise AssertionError('can this ever happen?')
        else:
            # mark the aff_wld crs as coming from the dataset
            # is there a better name for this?
            aff_wld_crs_type = 'dataset'
        aff_geo_transform = ref.GetGeoTransform()

    if True:
        # TODO: do we need this info? If not, delete.
        # wld_crs = ref.GetSpatialRef()
        if aff_wld_crs_type == 'dataset':
            info['SpatialRefLinearUnits'] = aff_wld_crs.GetLinearUnits()
            info['SpatialRefLinearUnitsName'] = aff_wld_crs.GetLinearUnitsName()
            gt = aff_geo_transform
            pixelSizeX =  gt[1]
            pixelSizeY = -gt[5]
            info['GeoTransformGSD'] = (pixelSizeX, pixelSizeY)

        elif aff_wld_crs_type == 'gcp':

            aff_wld_crs = ref.GetGCPSpatialRef()
            gcps = ref.GetGCPs()
            if len(gcps) == 0 or aff_wld_crs is None:
                raise Exception('no gcps')
            gt = gdal.GCPsToGeoTransform(gcps)
            gt = ref.GetGeoTransform()
            pixelSizeX =  gt[1]
            pixelSizeY = -gt[5]
            info['GCPSpatialRefLinearUnits'] = aff_wld_crs.GetLinearUnits()
            info['GCPSpatialRefLinearUnitsName'] = aff_wld_crs.GetLinearUnitsName()
            info['GCPGeoTransformGSD'] = (pixelSizeX, pixelSizeY)

    aff = affine.Affine.from_gdal(*aff_geo_transform)
    aff_wld_from_pxl = np.vstack([np.array(aff.column_vectors).T, [0, 0, 1]])
    aff_pxl_from_wld = np.linalg.inv(aff_wld_from_pxl)

    is_rpc = not (force_affine or rpc_transform is None)

    if is_rpc:
        # If we are using RPC, we will ignore the "affine world CRS"
        # and define the standard one the RPC will warp into.
        # RPC is always wrt to WGS84.
        # TODO: Do we need to port an axis mapping strategy here?

        # FIXME: Our RPC transforms are currently using traditional
        # axis ordering. This is not compliant.

        wld_axis_mapping_int = osr.OAMS_TRADITIONAL_GIS_ORDER
        wld_axis_mapping = axis_mapping_int_to_text(wld_axis_mapping_int)
        wld_crs_type = 'hard-coded-rpc-crs'
        wld_crs = osr.SpatialReference()
        wld_crs.ImportFromEPSG(4326)  # 4326 is the EPSG id WGS84 of lat/lon crs
        wld_crs.SetAxisMappingStrategy(wld_axis_mapping_int)

        # aff_wld_axis_mapping_int = aff_wld_crs.GetAxisMappingStrategy()
        # wld_crs.SetAxisMappingStrategy(aff_wld_axis_mapping_int)
        # wld_axis_mapping_int = wld_crs.GetAxisMappingStrategy()
    else:
        wld_crs = aff_wld_crs
        wld_crs_type = aff_wld_crs
        wld_axis_mapping_int = aff_wld_crs.GetAxisMappingStrategy()
        wld_axis_mapping = axis_mapping_int_to_text(wld_axis_mapping_int)

    wgs84_crs = osr.SpatialReference()
    wgs84_crs.ImportFromEPSG(4326)  # 4326 is the EPSG id WGS84 of lat/lon crs
    wgs84_axis_mapping_int = wgs84_crs.GetAxisMappingStrategy()
    wgs84_axis_mapping = axis_mapping_int_to_text(wgs84_axis_mapping_int)

    wgs84_from_wld = osr.CoordinateTransformation(wld_crs, wgs84_crs)
    wld_from_wgs84 = osr.CoordinateTransformation(wgs84_crs, wld_crs)  # 10% of the execution time

    if is_rpc:
        pxl_from_wld = rpc_transform.make_warp_pixel_from_world()
        wld_from_pxl = rpc_transform.make_warp_world_from_pixel()
    else:
        wld_from_pxl = aff_wld_from_pxl
        pxl_from_wld = aff_pxl_from_wld

    shape = (ref.RasterYSize, ref.RasterXSize)

    # warp pixel corner points to lat / lon
    # Note, CCW order for conversion to polys
    xy_corners = np.array([
        [0, 0],
        [0, ref.RasterYSize],
        [ref.RasterXSize, ref.RasterYSize],
        [ref.RasterXSize, 0],
    ])
    pxl_corners = kwimage.Coords(xy_corners)
    wld_corners = pxl_corners.warp(wld_from_pxl)  # 10% of the execution time

    wgs84_corners = wld_corners.warp(wgs84_from_wld)
    min_lat, min_lon = wgs84_corners.data[:, 0:2].min(axis=0)
    max_lat, max_lon = wgs84_corners.data[:, 0:2].max(axis=0)

    # lat1 = wgs84_corners.data[:, 0].min()
    # lat2 = wgs84_corners.data[:, 0].max()
    # lon1 = wgs84_corners.data[:, 1].min()
    # lon2 = wgs84_corners.data[:, 1].max()

    # min_lon, max_lon = sorted([lon1, lon2])
    # min_lat, max_lat = sorted([lat1, lat2])

    assert util_gis.check_latlons(
        wgs84_corners.data[:, 0], wgs84_corners.data[:, 1]), (
            'bad WGS84 coordinates'
        )

    WITH_UTM_INFO = True
    if WITH_UTM_INFO:
        epsg_int = util_gis.utm_epsg_from_latlon(min_lat, min_lon)
        utm_crs = osr.SpatialReference()
        utm_crs.ImportFromEPSG(epsg_int)
        utm_axis_mapping_int = utm_crs.GetAxisMappingStrategy()
        utm_axis_mapping = axis_mapping_int_to_text(utm_axis_mapping_int)

        utm_from_wld = osr.CoordinateTransformation(wld_crs, utm_crs)  # 4% time
        # wld_from_utm = osr.CoordinateTransformation(utm_crs, wld_crs)
        utm_corners = wld_corners.warp(utm_from_wld)  # 2% time

        min_utm = utm_corners.data.min(axis=0)
        max_utm = utm_corners.data.max(axis=0)
        meter_extent = max_utm - min_utm
        pxl_extent = np.array([ref.RasterXSize, ref.RasterYSize])
        meter_per_pxl = meter_extent / pxl_extent
    else:
        meter_per_pxl = None
        meter_extent = None
        utm_from_wld = None
        # wld_from_utm = None
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
        utm_wkt = utm_crs.ExportToWkt()
        wld_wkt = wld_crs.ExportToWkt()
        wgs84_wkt = wgs84_crs.ExportToWkt()

        utm_crs_info = {
            'auth': memo_auth_from_wkt(utm_wkt),
            'axis_mapping': utm_axis_mapping,
        }

        wld_crs_info = {
            'auth': memo_auth_from_wkt(wld_wkt),
            'axis_mapping': wld_axis_mapping,
            'type': wld_crs_type,
        }

        wgs84_crs_info = {
            'auth': memo_auth_from_wkt(wgs84_wkt),
            'axis_mapping': wgs84_axis_mapping,
        }

        # Convert to the more general geos corners
        wgs84_crs_info = ub.dict_diff(wgs84_crs_info, {'type'})
        if wgs84_crs_info['axis_mapping'] == 'OAMS_AUTHORITY_COMPLIANT':
            geos_corners = kwimage.Polygon.coerce(
                wgs84_corners).swap_axes().to_geojson()
        else:
            geos_corners = kwimage.Polygon.coerce(
                wgs84_corners).to_geojson()
        geos_crs_info = {
            'axis_mapping': 'OAMS_TRADITIONAL_GIS_ORDER',
            'auth': ('EPSG', '4326')
        }
        geos_corners['properties'] = {'crs_info': geos_crs_info}

    info.update({
        'is_rpc': is_rpc,

        'meter_per_pxl': meter_per_pxl,
        'meter_extent': meter_extent,
        'approx_elevation': approx_elevation,

        'gcp_wld_coords': gcp_wld_coords,
        'gcp_pxl_coords': gcp_pxl_coords,

        'rpc_transform': rpc_transform,

        'pxl_corners': pxl_corners,
        'utm_corners': utm_corners,
        'wld_corners': wld_corners,
        'wgs84_corners': wgs84_corners,
        'geos_corners': geos_corners,

        # TODO: we changed the internal names to the "_from_" varaint but we
        # are keeping the "_to_" variant in this interface for now In the
        # future we should change things over to the "_from_" variant, which
        # is easier to reason about when chaining transforms.
        'pxl_to_wld': wld_from_pxl,
        'wgs84_to_wld': wld_from_wgs84,

        # 'utm_to_wld': wld_from_utm,  # unused, and 10% overhead


        'wld_to_pxl': pxl_from_wld,
        'wld_to_wgs84': wgs84_from_wld,
        'wld_to_utm': utm_from_wld,

        'utm_crs_info': utm_crs_info,
        'wld_crs_info': wld_crs_info,
        'wgs84_crs_info': wgs84_crs_info,

        'bbox_geos': bbox_geos,
        'img_shape': shape,
    })

    if info['utm_corners'] is not None:
        utm_box = kwimage.Polygon(exterior=info['utm_corners']).box()
        meter_w = float(utm_box.width)
        meter_h = float(utm_box.height)
        meter_hw = np.mean([meter_h , meter_w])
        pxl_hw = np.array(info['img_shape'])
        gsd = (meter_hw / pxl_hw).mean()
        minx, miny = info['utm_corners'].data.min(axis=0)
        maxx, maxy = info['utm_corners'].data.min(axis=0)
        info['approx_meter_gsd'] = gsd
    return info


def make_crs_info_object(osr_crs):
    """
    Args:
        osr_crs (osr.SpatialReference): an osr object from gdal

    Example:
        >>> from geowatch.gis.geotiff import *  # NOQA
        >>> from geowatch.utils import util_gdal
        >>> osr = util_gdal.import_osr()
        >>> osr_crs = osr.SpatialReference()
        >>> osr_crs.ImportFromEPSG(4326)
        >>> crs_info = make_crs_info_object(osr_crs)
        >>> print('crs_info = {}'.format(ub.urepr(crs_info, nl=1)))
        >>> osr_crs = osr.SpatialReference()
        >>> osr_crs.ImportFromEPSG(4326)
        >>> osr_crs.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
        >>> crs_info = make_crs_info_object(osr_crs)
        >>> print('crs_info = {}'.format(ub.urepr(crs_info, nl=1)))
        >>> osr_crs.ImportFromEPSG(32744)
        >>> crs_info = make_crs_info_object(osr_crs)
        >>> print('crs_info = {}'.format(ub.urepr(crs_info, nl=1)))
    """
    wkt = osr_crs.ExportToWkt()
    auth = memo_auth_from_wkt(wkt)
    axis_mapping_int = osr_crs.GetAxisMappingStrategy()
    axis_mapping_text = axis_mapping_int_to_text(axis_mapping_int)
    crs_info = {
        'auth': auth,
        'axis_mapping': axis_mapping_text,
    }
    return crs_info


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
    from geowatch.utils import util_gdal
    osr = util_gdal.import_osr()
    if axis_mapping_int == osr.OAMS_TRADITIONAL_GIS_ORDER:
        axis_mapping = 'OAMS_TRADITIONAL_GIS_ORDER'
    elif axis_mapping_int == osr.OAMS_AUTHORITY_COMPLIANT:
        axis_mapping = 'OAMS_AUTHORITY_COMPLIANT'
    elif axis_mapping_int == osr.OAMS_CUSTOM:
        axis_mapping = 'OAMS_CUSTOM'
    else:
        raise KeyError(axis_mapping_int)
    return axis_mapping


# def new_spatial_reference(axis_mapping='OAMS_AUTHORITY_COMPLIANT'):
#     """
#     Creates a new spatial reference

#     Args:
#         axis_mapping (int | str) : can be
#             OAMS_TRADITIONAL_GIS_ORDER, OAMS_AUTHORITY_COMPLIANT, or
#             OAMS_CUSTOM or the integer gdal code.

#     References:
#         https://gdal.org/tutorials/osr_api_tut.html#crs-and-axis-order
#     """
#     raise NotImplementedError
#     from osgeo import osr
#     if isinstance(axis_mapping, int):
#         axis_mapping_int = axis_mapping
#     else:
#         assert axis_mapping in {
#             'OAMS_TRADITIONAL_GIS_ORDER',
#             'OAMS_AUTHORITY_COMPLIANT',
#             'OAMS_CUSTOM',
#         }
#         axis_mapping_int = getattr(osr, axis_mapping)


@ub.memoize
def memo_auth_from_wkt(wkt):
    """
    This benchmarks as an expensive operation, memoize it.
    """
    from pyproj import CRS
    return CRS.from_wkt(wkt).to_authority(min_confidence=100)


class InvalidFormat(Exception):
    pass


def geotiff_filepath_info(gpath, fast=True):
    """
    Attempt to parse information out of a path to a geotiff file.

    Information provided here is purely heuristic. Generally filepath
    information is not robust.

    Several huerstics are currently implemented for:
        * Sentinel 2
        * Landsat-8
        * WorldView3

    See [S2_Name_2016]_ and [S3_Name]_.

    Args:
        gpath (str): a path to an image that uses a standard naming convention
            (may include subdirectories that contain relevant information) .

        fast (bool):
            if True stops when a hueristic matches well enough, otherwise tries
            multiple hueristics. Defaults to True.

    SeeAlso:
        * parse_landsat_product_id - specific to the landsat spec

    Example:
        >>> from geowatch.gis.geotiff import *  # NOQA
        >>> inputs = [
        >>>     '18JUN18071532-S3DMR1C2.NTF',
        >>>     'LC08_L1GT_029030_20151209_20160131_01_RT',
        >>>     'LC08_L1TP_162043_20130401_20170505_01_T1_B8.tif',
        >>>     'LC08_L1GT_017037_20190303_20190309_01_T2_B8.TIF',
        >>>     '03AUG16WV031000016AUG03125811-P1BS-500827464010_01_P002_________AAE_0AAAAABPABB0.NTF',
        >>>     '13APR16WV030900016APR13131153-P1BS_R1C1-500659828010_01_P001____AAE_0AAAAABPABM0.NTF.NTF',
        >>>     '011777481010_01_P001_MUL/17SEP07021826-M1BS-011777481010_01_P001.TIF',
        >>>     '011778213010_01_P001_PAN/14FEB03022210-P1BS-011778213010_01_P001.NTF',
        >>>     'T17SMS_20151020T162042_TCI.jp2',
        >>>     'T39QXG_20160711T065622_TCI.jp2',
        >>>     'S2B_MSIL1C_20181219T022109_N0207_R003_T52SDG_20181219T051028.SAFE/GRANULE/L1C_T52SDG_A009324_20181219T022640/IMG_DATA/T52SDG_20181219T022109_TCI.jp2',
        >>>     'S2B_MSIL1C_20181219T022109_N0207_R003_T52SDG_20181219T051028/S2B_MSIL1C_20181219T022109_N0207_R003_T52SDG_20181219T051028.SAFE/GRANULE/L1C_T52SDG_A009324_20181219T022640/IMG_DATA/T52SDG_20181219T022109_B01.jp2',
        >>>     'S2A_MSIL1C_20151021T022702_N0204_R003_T52SDG_20151021T022701.SAFE/GRANULE/L1C_T52SDG_A001716_20151021T022701/IMG_DATA/T52SDG_20151021T022702_TCI.jp2',
        >>>     'LC08_L2SP_217076_20190107_20211008_02_T1_T23KPQ_B1.tif',
        >>>     'S2B_MSI_L2A_T23KPQ_20190114_20211008_SR_B01.tif',
        >>>     'S2A_MSI_L2A_T39RVK_20180803_20211102_SR_SOZ4.tif',
        >>> ]
        >>> gpath = inputs[-1]
        >>> print('gpath = {!r}'.format(gpath))
        >>> for gpath in inputs:
        >>>     print('----')
        >>>     info = geotiff_filepath_info(gpath)
        >>>     print('gpath = {}'.format(ub.urepr(gpath, nl=1)))
        >>>     print('info = {}'.format(ub.urepr(info, nl=2)))
        >>>     if len(info['sensor_candidates']) == 0:
        >>>         print(ub.color_text('NO HUERISTIC', 'red'))
        >>>     else:
        >>>         assert len(info['sensor_candidates']) == len(set(info['sensor_candidates']))
        >>>         assert info['filename_meta']

    Example:
        >>> from geowatch.gis.geotiff import *  # NOQA
        >>> gpath = 'LC08_L1TP_037029_20130602_20170310_01_T1_B1.TIF'
        >>> info = geotiff_filepath_info(gpath)
        >>> print('info = {}'.format(ub.urepr(info, nl=1)))
        >>> assert info['filename_meta']['sensor_code'] == 'C'
        >>> assert info['filename_meta']['sat_code'] == '08'
        >>> assert info['filename_meta']['sat_code'] == '08'
        >>> assert info['filename_meta']['collection_category'] == 'T1'
        >>> assert info['filename_meta']['suffix'] == 'B1'

    Example:
        >>> # xdoctest: +REQUIRES(--network)
        >>> # Test extact info from real landsat product files
        >>> from geowatch.demo.landsat_demodata import grab_landsat_product  # NOQA
        >>> from geowatch.gis.geotiff import *  # NOQA
        >>> product = grab_landsat_product()
        >>> band_infos = [geotiff_filepath_info(gpath) for gpath in product['bands']]
        >>> meta_infos = [geotiff_filepath_info(gpath) for gpath in product['meta'].values()]
        >>> assert not any(d is None for d in band_infos)
        >>> assert not any(d is None for d in meta_infos)

    Ignore:
        >>> # TODO : demodata for a digital globe archive
        >>> from geowatch.gis.geotiff import *  # NOQA
        >>> import ubelt as ub
        >>> gpath = ub.expandpath('$HOME/remote/namek/data/dvc-repos/smart_watch_dvc/drop0/KR-Pyeongchang-WV/_assets/20170907_a_KRP_011777481_10_0/011777481010_01_003/011777481010_01/011777481010_01_P001_MUL/17SEP07021826-M1BS-011777481010_01_P001.TIF')
        >>> info = geotiff_filepath_info(gpath)
        >>> print('info = {}'.format(ub.urepr(info, nl=1)))

    Ignore:
        >>> from geowatch.gis.geotiff import *  # NOQA
        >>> base_dpath = '/home/joncrall/data/grab_tiles_out/fels'
        >>> paths = sorted(walk_geotiff_products(base_dpath, with_product_dirs=1, with_loose_images=0))
        >>> infos = []
        >>> for path in paths:
        >>>     info = geotiff_filepath_info(path)
        >>>     info['basename'] = basename(path)
        >>>     print('info = {}'.format(ub.urepr(info, nl=1)))
        >>>     infos.append(info)

        >>> paths = sorted(walk_geotiff_products(base_dpath, with_product_dirs=0, with_loose_images=1))
        >>> infos = []
        >>> for path in paths:
        >>>     info = geotiff_filepath_info(path)
        >>>     info['basename'] = basename(path)
        >>>     print('info = {}'.format(ub.urepr(info, nl=1)))
        >>>     infos.append(info)
        >>> print('infos = {}'.format(ub.urepr(infos[0:10], nl=1)))
        >>> print('infos = {}'.format(ub.urepr(infos[-10:], nl=1)))

        >>> infos = []
        >>> for path in paths:
        >>>     info = geotiff_filepath_info(path)
        >>>     info['path'] = path
        >>>     infos.append(info)

    References:
        .. [S2_Name_2016] https://sentinel.esa.int/web/sentinel/user-guides/sentinel-2-msi/naming-convention
        .. [S3_Name] https://sentinel.esa.int/web/sentinel/user-guides/sentinel-3-altimetry/naming-conventions

    """

    base_ext = basename(gpath)
    base, *exts = base_ext.split('.')
    ext = '.'.join(exts)  # NOQA
    parts = gpath.split('/')

    info = {
        'sensor_candidates': [],
        'filename_meta': {},
    }
    sensor_candidates = info['sensor_candidates']
    meta = info['filename_meta']

    ls_meta = parse_landsat_product_id(base)
    if ls_meta is not None:
        sensor_cand = 'L' + ls_meta['sensor_code'] + ls_meta['sat_code']
        meta.update(ls_meta)
        meta['product_guess'] = 'landsat'
        meta['guess_heuristic'] = 'landsat_parse'
        sensor_candidates.append(sensor_cand)

    if not sensor_candidates or not fast:
        s2_meta = parse_sentinel2_product_id(parts)
        if s2_meta is not None:
            meta.update(s2_meta)
            sensor_candidates.append(meta['mission_id'])
            if meta.get('suffix', None) == 'TCI':
                sensor_candidates.append('S2-TrueColor')
            meta['product_guess'] = 'sentinel2'
            meta['guess_heuristic'] = 'sentinel2_parse'

    dg_bundle = None
    if not sensor_candidates or not fast:
        # WorldView3
        # TODO: find a reference for the spec
        # TODO fix date handling for eg 03JUL15WV020200015JUL03021500-P1BS-011777484010_01_P001.NTF or 15JUL03021500
        wv3_pat = '{date1:w}-{part1:w}-{date2:w}_{num:w}_{part2:w}'
        wv3_parser = _parser_lut(wv3_pat)
        prefix = base.split('____')[0]
        result = wv3_parser.parse(prefix)
        if result is not None:
            try:
                wv2_meta = {}
                wv2_meta['date1'] = result.named['date1']
                wv2_meta['part1'] = result.named['part1']
                wv2_meta['date2'] = result.named['date2']
                wv2_meta['num'] = result.named['num']
                wv2_meta['part2'] = result.named['part2']
                wv2_meta['product_guess'] = 'worldview'
                wv2_meta['guess_heuristic'] = 'WV_heuristic1'
                meta.update(wv2_meta)
            except InvalidFormat:
                pass
            else:
                meta.update(wv2_meta)
                sensor_candidates.append('WV03')

        # Add DG information if it exists
        from geowatch.gis import digital_globe as dg_parser
        try:
            # technically, this does read files, so its not all about the path
            dg_bundle = dg_parser.DigitalGlobeBundle.from_pointer(gpath)
            info['dg_info'] = dg_bundle.data
        except Exception:
            dg_bundle = None

        if dg_bundle is not None:
            for prod_meta in dg_bundle.data['product_metas']:
                sensor_candidates.append(prod_meta['sensorVehicle'])

    # TODO: handle landsat and sentinel2 bundles
    info['is_dg_bundle'] = dg_bundle is not None

    if 0:
        # This is slow, and I don't think it is ever hit

        def _is_rgb(gpath):
            # fallback for 'channels'
            # often, a gtiff is a TCI that was postprocessed in some way that destroys
            # the original naming convention
            #
            # this opens the image to check for that case as a fallback
            from geowatch.utils import util_gdal
            gdal = util_gdal.import_gdal()
            info = gdal.Info(gpath, format='json')
            if len(info['bands']) == 3:
                # TODO sometimes colorInterpretation is stripped, but it's still RGB
                # should this return True for any gpath with 3 bands?
                if [b['colorInterpretation'] for b in info['bands']] == ['Red', 'Green', 'Blue']:
                    return True
            return False

        if 'channels' not in meta:
            if isfile(gpath) and _is_rgb(gpath):
                meta['channels'] = 'r|g|b'

    return info


@ub.memoize
def _parser_lut(pattern):
    """
    Calling parse.parse is about 14x slower than creating a parser object
    once and then using it.
    """
    return parse.Parser(pattern)


def parse_sentinel2_product_id(parts):
    """
    Try to parse the Sentinel-2 pre-2016 and post-2016 safedir formats.

    Note that unlike parse_landsat_product_id, which expects a band file basename,
    this presently purports to plurally parse pieces of path postfixedly
    (it parses the whole path, backwards :))

    TODO extend this to parsing the granuledir and band file formats as well.
    For now, we just need all names to be minimally parseable, even if some info is incorrect.

    General plan is to check the old formats strictly first, and then check the new safedir loosely as a default

    Example:

        parts = ['S2B_MSI_L2A_T23KPQ_20190114_20211008_SR_B01.tif']
        parts = ['S2A_MSI_L2A_T39RVK_20180803_20211102_SR_SOZ4.tif']
        parse_sentinel2_product_id(parts)

    """

    def _dt(name):
        # expand to a named ISO 8601 datetime without separators, which is not supported by parse
        # example: 20190901T234135
        # {{ escapes {

        # trying to be too clever here...
        # return f'{{{name}.Y:04d}}{{{name}.M:02d}}{{{name}.D:02d}}T{{{name}.h:02d}}{{{name}.m:02d}}{{{name}.s:02d}}'
        # return f'{{{name}.date:08d}}T{{{name}.time:06d}}'
        return f'{{{name}:.15}}'

    # unfortunately parse() doesn't seem to support a format specifier for "string of length exactly n"
    # {name:n} is ">= n"
    # {name:.n} is "<= n"  <- going with this one as a better approximation of "exactly n"

    # this also EXCLUDES the trailing '.SAFE' for all safedirs, again because parse can't handle optional pieces

    s2_safedir_2015 = '{MMM:.3}_{CCCC:.4}_PRD_{MSIXXX:.6}_{ssss:.4}_' + _dt('creation') + '_R{OOO:03d}_V' + _dt('sensing_start') + '_' + _dt('sensing_end')
    s2_safedir_2015_parser = _parser_lut(s2_safedir_2015)

    # parse also doesn't support 'or'
    # A could instead be the Validity Start Time
    # T could instead be the Detector ID
    # but these are the more common (and useful) choices
    s2_granuledir_2015 = '{MMM:.3}_{CCCC:.4}_MSI_{YYY:.3}_{ZZ:.2}_{ssss:.4}_' + _dt('validity_start') + '_A{ffffff:06d}_T{xxxxx:.5}_N{xx:02d}.{yy:02d}'
    s2_granuledir_2015_parser = _parser_lut(s2_granuledir_2015)

    # Sentinel-2 2016+ filename pattern. See [S2_Name_2016]_
    # These filenames are often directories
    # MMM_MSIXXX_YYYYMMDDHHMMSS_Nxxyy_ROOO_Txxxxx_<Product Discriminator>.SAFE
    # SAFE = Standard Archive Format for Europe
    s2_name_2016 = '{MMM:.3}_{MSIXXX:.6}_{YYYYMMDDHHMMSS:.15}_{Nxxyy:.5}_{ROOO:.4}_{Txxxxx:.6}_{Discriminator}'
    s2_2016_parser = _parser_lut(s2_name_2016)

    # use util_bands for this
    s2_channel_alias = {
        band['name']: band['common_name'] for band in SENTINEL2 if 'common_name' in band
    }
    # ...except for TCI, which is not a true band, but often included anyway
    # and this channel code is more specific to kwcoco
    s2_channel_alias.update({'TCI': 'r|g|b'})

    meta = {}

    # TODO allow for parsing multiple parts once safedir, granuledir, and band file are all implemented
    for _part in reversed(parts):

        part = _part.split('.')[0]
        result = s2_safedir_2015_parser.parse(part)
        if result:
            s2_meta = {}
            try:
                mission_id = result.named['MMM']
                if mission_id not in {'S2A', 'S2B'}:
                    raise InvalidFormat
                s2_meta['mission_id'] = mission_id
                s2_meta['product_level'] = result.named['MSIXXX']
                s2_meta['sense_start_time'] = isoparse(result.named['sensing_start']
                                                       ).isoformat()
                s2_meta['relative_orbit_num'] = result.named['ROOO']
                s2_meta['discriminator'] = result.named['Discriminator']
                s2_meta['product_guess'] = 'sentinel2'
                s2_meta['guess_heuristic'] = 'S2_safedir_2015_format'
            except InvalidFormat:
                pass
            else:
                meta.update(s2_meta)
                break

        part = '.'.join(_part.split('.')[:2])
        result = s2_granuledir_2015_parser.parse(part)
        if result:
            s2_meta = {}
            try:
                mission_id = result.named['MMM']
                if mission_id not in {'S2A', 'S2B'}:
                    raise InvalidFormat
                s2_meta['mission_id'] = mission_id
                s2_meta['product_level'] = result.named['YYY']
                # TODO warning, this date can differ between the 2 in safedir and 1 in granuledir!
                # it's an open question which should be "canonical"
                s2_meta['sense_start_time'] = isoparse(result.named['validity_start']
                                                       ).isoformat()
                s2_meta['pdgs_num'] = f"N{result.named['xx']:02d}{result.named['yy']:02d}"
                s2_meta['absolute_orbit_num'] = result.named['ffffff']
                s2_meta['tile_number'] = 'T' + result.named['xxxxx']
                s2_meta['product_guess'] = 'sentinel2'
                s2_meta['guess_heuristic'] = 'S2_2016_format'
            except InvalidFormat:
                pass
            else:
                meta.update(s2_meta)
                break

        part = _part.split('.')[0]
        result = s2_2016_parser.parse(part)
        if result:
            s2_meta = {}
            try:
                mission_id = result.named['MMM']
                if mission_id not in {'S2A', 'S2B'}:
                    raise InvalidFormat
                s2_meta['mission_id'] = mission_id
                s2_meta['product_level'] = result.named['MSIXXX']
                s2_meta['sense_start_time'] = result.named['YYYYMMDDHHMMSS']
                s2_meta['pdgs_num'] = result.named['Nxxyy']
                s2_meta['relative_orbit_num'] = result.named['ROOO']
                s2_meta['tile_number'] = result.named['Txxxxx']
                s2_meta['discriminator'] = result.named['Discriminator']
                s2_meta['product_guess'] = 'sentinel2'
                s2_meta['guess_heuristic'] = 'S2_2016_format'
            except InvalidFormat:
                pass
            else:
                meta.update(s2_meta)
                break

    # Files ending in _TCI are true color images based on the Sentinel 2
    # Handbook I'm not sure what the standard for this format is I just know a
    # suffix of _TCI means true color image. ANd these are from sentinel2, I'm
    # guessing on the rest of the format.
    s2_format_guess1 = '{tile_number:.6}_{date:.15}_{band:.3}.{ext}'
    s2_parser = _parser_lut(s2_format_guess1)
    result = s2_parser.parse(parts[-1])
    if result:
        tile_number = result.named['tile_number']
        if len(tile_number) == 6 and tile_number.startswith('T'):
            if 'sense_start_time' in meta:
                assert meta['sense_start_time'] == result.named['date']
            # Changed from acquisition_date to match other S2 information
            meta['sense_start_time'] = result.named['date']
            band = result.named['band']
            # TODO: normalized consise channel code
            meta['tile_number'] = tile_number
            meta['suffix'] = band
            channels = s2_channel_alias.get(band, band)
            meta['channels'] = channels
            meta['product_guess'] = 'sentinel2'
            meta['guess_heuristic'] = 'S2_tile_date_band_format'
            if 'mission_id' not in meta:
                meta['mission_id'] = 'S2'

    # This is another guess based on a file that failed to ingest
    s2_format_guess2 = '{MMM:.3}_MSI_{XXX:.3}_{Txxxxx:.6}_{SENSE_YYYYMMDD:.8}_{PROC_YYYYMMDD:.8}_{correction_code}_{band}.{ext}'
    s2_parser2 = _parser_lut(s2_format_guess2)
    result = s2_parser2.parse(parts[-1])
    if result:
        s2_meta = {}
        mission_id = result.named['MMM']
        if mission_id not in {'S2A', 'S2B'}:
            raise InvalidFormat
        s2_meta['mission_id'] = mission_id
        s2_meta['product_level'] = result.named['XXX']
        s2_meta['tile_number'] = result.named['Txxxxx']
        s2_meta['sense_start_time'] = result.named['SENSE_YYYYMMDD']
        s2_meta['processing_date'] = result.named['PROC_YYYYMMDD']  # is this corret?
        s2_meta['correction_code'] = result.named['correction_code']
        s2_meta['product_guess'] = 'sentinel2'
        s2_meta['guess_heuristic'] = 's2_format_guess2'
        s2_meta['band'] = band = result.named['band']
        s2_meta['channels'] = s2_channel_alias.get(band, band)
        # l8_channel_alias = {
        #     band['name']: band['common_name'] for band in SENTINEL2 if 'common_name' in band
        # }

        meta.update(s2_meta)

    if meta:
        if 'sense_start_time' in meta:
            # Write data to a consistent key
            meta['date_captured'] = isoparse(meta['sense_start_time']).isoformat()
        return meta


def parse_landsat_product_id(product_id):
    """
    Extract information from a landsat produt id

    See [LanSatName]_, [LS_578]_, [ExampleLandSat]_, [LandSatSuffixFormat]_,
    [LandsatProcLevels]_, [LandsatL2Names]_, and [LandSatARDDocs]_.

    Args:
        product_id (str): this is typically the filename (without extension!)
            of a landsat product, as described in [LanSatName]_.

    Example:
        >>> from geowatch.gis.geotiff import *  # NOQA
        >>> product_id = 'LC08_L1TP_037029_20130602_20170310_01_T1'
        >>> ls_meta = parse_landsat_product_id(product_id)

    Example:
        >>> from geowatch.gis.geotiff import *  # NOQA
        >>> gpath = 'LC08_L1TP_037029_20130602_20170310_01_T1_B1'
        >>> info = parse_landsat_product_id(gpath)
        >>> print('info = {}'.format(ub.urepr(info, nl=1)))
        >>> assert info['sensor_code'] == 'C'
        >>> assert info['sat_code'] == '08'
        >>> assert info['sat_code'] == '08'
        >>> assert info['collection_category'] == 'T1'
        >>> assert info['suffix'] == 'B1'
        >>> assert info['band_num'] == 1


        >>> from geowatch.gis.geotiff import *  # NOQA
        >>> gpath = 'LC08_CU_029005_20181208_20210503_02_QA_LINEAGE.TIF'
        >>> info = parse_landsat_product_id(gpath)
        >>> print('info = {}'.format(ub.urepr(info, nl=1)))

    Example:
        >>> # xdoctest: +REQUIRES(--network)
        >>> # Test on real landsat data
        >>> from geowatch.demo.landsat_demodata import grab_landsat_product  # NOQA
        >>> from geowatch.gis.geotiff import *  # NOQA
        >>> product = grab_landsat_product()
        >>> band_prodids = [ub.augpath(gpath, dpath='', ext='') for gpath in product['bands']]
        >>> band_infos = [parse_landsat_product_id(product_id) for product_id in band_prodids]
        >>> assert ub.allsame([ub.dict_diff(d, ['band_num', 'suffix', 'channels', 'band']) for d in band_infos])
        >>> meta_prodids = [ub.augpath(gpath, dpath='', ext='') for gpath in product['meta'].values()]
        >>> meta_infos = [parse_landsat_product_id(product_id) for product_id in meta_prodids]
        >>> assert ub.allsame([ub.dict_diff(d, ['suffix', 'channels', 'band']) for d in meta_infos])

    References:
        .. [LanSatName] https://www.usgs.gov/faqs/what-naming-convention-landsat-collections-level-1-scenes
        .. [LS_578] https://github.com/dgketchum/Landsat578#-1
        .. [ExampleLandSat]  https://console.cloud.google.com/storage/browser/gcp-public-data-landsat/LC08/01/044/034/LC08_L1GT_044034_20130330_20170310_01_T2?_ga=2.210779154.665659046.1615242530-37570621.1615242530
        .. [LandSatSuffixFormat] https://prd-wret.s3.us-west-2.amazonaws.com/assets/palladium/production/atoms/files/LSDS-750_Landsat8_Level-0-Reformatted_DataFormatControlBook-v15.pdf (page 26 / 99)
        .. [LandsatProcLevels] https://www.usgs.gov/core-science-systems/nli/landsat/landsat-levels-processing
        .. [LandsatL2Names] https://www.usgs.gov/faqs/what-naming-convention-landsat-collection-2-level-1-and-level-2-scenes?qt-news_science_products=0#qt-news_science_products

        .. [LandSatARDDocs] https://d9-wret.s3.us-west-2.amazonaws.com/assets/palladium/production/s3fs-public/atoms/files/LSDS-1873_US-Landsat%20C1-ARD-DFCB-v7.pdf
    """
    # Landsat filename pattern. See [LanSatName]_
    # LXSS_LLLL_PPPRRR_YYYYMMDD_yyyymmdd_CC_TX
    #                   LXSS     _ LLLL_  PPPRRR _ YYYYMMDD _ yyyymmdd _ CC _ TX
    from dateutil.parser import isoparse
    landsat_pattern = 'L{X:1}{SS}_{LLLL}_{PPPRRR}_{YYYYMMDD}_{yyyymmdd}_{CC}_{TX}'
    landsat_parser = _parser_lut(landsat_pattern)
    result = landsat_parser.parse(product_id)
    if result:
        ls_sensor_code_to_text = {
            'C': 'OLI/TIRS',
            'O': 'OLI',
            'T': 'TIRS',
            'E': 'ETM+',
            # 'T': 'TM',  # ambiguous? That's in the spec
            'M': 'MSS',
        }

        correction_code_to_text = {
            'L1TP': 'Precision Terrain',
            'L1GT': 'Systematic Terrain',
            'L1GS': 'Systematic',
            'L2SP': 'Science Package',
            'L2SR': 'Surface Reflectance',
            'L2ST': 'Surface Temperature',  # this is a guess
            'CU': 'CU?',  # No idea what this is. Seen in quality ands.
        }

        # use util_bands for this
        # l7_channel_alias = {
        #     band['name']: band['common_name'] for band in LANDSAT7 if 'common_name' in band
        # }
        l8_channel_alias = {
            band['name']: band['common_name'] for band in LANDSAT8 if 'common_name' in band
        }

        # When accessing files from google API, there might be an additional
        # field specifying band information.
        trailing = result.named['TX'].split('_')
        tx = trailing[0]

        wrs = result.named['PPPRRR']
        sensor_code = result.named['X']
        sat_code = result.named['SS']

        correction_code = result.named['LLLL']

        ls_meta = {}
        ls_meta['sensor_text']: ls_sensor_code_to_text[sensor_code]
        ls_meta['correction_level_text'] = correction_code_to_text[correction_code]

        ls_meta['sensor_code'] = sensor_code
        ls_meta['sat_code'] = sat_code
        ls_meta['WRS_path'] = wrs[:3]
        ls_meta['WRS_row'] = wrs[3:]
        ls_meta['tile_number'] = wrs
        ls_meta['correction_level_code'] = correction_code
        ls_meta['acquisition_date'] = result.named['YYYYMMDD']
        ls_meta['processing_date'] = result.named['yyyymmdd']
        ls_meta['collection_number'] = result.named['CC']
        ls_meta['collection_category'] = tx

        # Common key
        ls_meta['date_captured'] = isoparse(ls_meta['acquisition_date']).isoformat()

        if len(trailing) > 1:
            suffix = '_'.join(trailing[1:])
            ls_meta['suffix'] = suffix

            name_suffix = suffix
            removable_exts = ['.tif', '.tiff']
            for s in removable_exts:
                if name_suffix.lower().endswith(s):
                    name_suffix = name_suffix[:-len(s)]

            ls_meta['band'] = name_suffix
            ls_meta['channels'] = l8_channel_alias.get(name_suffix, name_suffix)

            if name_suffix == 'ANC':
                ls_meta['is_ancillary'] = True
            elif name_suffix == 'MTA':
                ls_meta['is_metadata'] = True
            elif name_suffix == 'MD5':
                ls_meta['is_checksum'] = True
            else:
                # The suffix might represent something about band
                # information, we may parse it.
                # See [LandSatSuffixFormat]_.
                band_suffix_pat = 'B{band_num:d}'
                band_suffix_parser = _parser_lut(band_suffix_pat)
                band_result = band_suffix_parser.parse(name_suffix)
                if band_result is not None:
                    ls_meta['band_num'] = band_result.named['band_num']
        return ls_meta


# def normalize_sensor():
#     pass


def walk_geotiff_products(dpath, with_product_dirs=True,
                          with_loose_images=True, recursive=True):
    """
    Walks a file path and returns directories and files that look
    like standalone geotiff products.

    Args:
        dpath (str): directory to search

    Yields:
        str: paths of files or recognized geotiff product bundle directories

    Example:
        >>> # xdoctest: +REQUIRES(env:SLOW_DOCTEST)
        >>> # xdoctest: +REQUIRES(--network)
        >>> # Test on real landsat data
        >>> from geowatch.gis.geotiff import *  # NOQA
        >>> import geowatch
        >>> from os.path import dirname
        >>> product = geowatch.demo.landsat_demodata.grab_landsat_product()
        >>> dpath = dirname(dirname(ub.peek(product['bands'])))
        >>> print(list(walk_geotiff_products(dpath)))
        >>> print(list(walk_geotiff_products(dpath, with_product_dirs=False)))
    """
    import os
    from os.path import join
    # blocklist = set()
    GEOTIFF_EXTENSIONS = ('.vrt', '.tiff', '.tif', '.jp2')

    for r, ds, fs in os.walk(dpath):
        handled = []
        if with_product_dirs:
            for didx, dname in enumerate(ds):
                if dname.startswith('LE07'):
                    dpath = join(r, dname)
                    handled.append(didx)
                    yield dpath
                elif dname.startswith('LC08_'):
                    dpath = join(r, dname)
                    handled.append(didx)
                    yield dpath
                elif dname.startswith(('S2A_', 'S2B_')):
                    handled.append(didx)
                    dpath = join(r, dname)
                    yield dpath
                else:
                    pass
            for didx in reversed(handled):
                del ds[didx]

        if with_loose_images:
            for fname in fs:
                if fname.lower().endswith(GEOTIFF_EXTENSIONS):
                    fpath = join(r, fname)
                    yield fpath

        if not recursive:
            break
