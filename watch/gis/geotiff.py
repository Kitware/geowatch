"""
Tools to work with geotiff metadata.
"""
import numpy as np
import ubelt as ub
from watch.gis import spatial_reference as watch_crs


def geotiff_metadata(gpath):
    """
    Extract all relevant metadata we know how to extract.
    """
    import gdal
    infos = {}
    ref = gdal.Open(gpath, gdal.GA_ReadOnly)
    infos['fname'] = geotiff_filepath_info(gpath)
    infos['cfs'] = geotiff_crs_info(ref)
    infos['header'] = geotiff_header_info(ref)

    # Combine sensor candidates
    sensor_candidates = list(ub.flatten([
        v.get('sensor_candidates', []) for v in infos.values()]))

    info = ub.dict_union(*infos.values())
    info['sensor_candidates'] = sensor_candidates

    if info['utm_corners'] is not None:
        import kwimage
        utm_box = kwimage.Polygon(exterior=info['utm_corners']).bounding_box()
        meter_w = float(utm_box.width.ravel()[0])
        meter_h = float(utm_box.height.ravel()[0])
        meter_hw = np.mean([meter_h , meter_w])
        pxl_hw = np.array(info['img_shape'])
        gsd = (meter_hw / pxl_hw).mean()
        minx, miny = info['utm_corners'].data.min(axis=0)
        maxx, maxy = info['utm_corners'].data.min(axis=0)
        info['approx_meter_gsd'] = gsd
    return info


def _coerce_gdal_dataset(data):
    import gdal
    if isinstance(data, str):
        ref = gdal.Open(data, gdal.GA_ReadOnly)
    elif isinstance(data, gdal.Dataset):
        ref = data
    else:
        raise TypeError(type(data))

    if ref is None:
        raise Exception('data={} is not a gdal dataset'.format(data))
    return ref


def geotiff_header_info(gpath_or_ref):
    """
    Extract relevant metadata information from a geotiff header.
    """
    # import gdal
    ref = _coerce_gdal_dataset(gpath_or_ref)
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
            print('meta = {}'.format(ub.repr2(meta, nl=1)))
        for key, value in ub.dict_isect(meta, keys_of_interest).items():
            all_domain_img_meta['{}.{}'.format(ns, key)] = value
    img_info = {}
    img_info['img_meta' ] = all_domain_img_meta
    img_info['num_bands'] = ref.RasterCount
    img_info['sensor_candidates'] = sensor_candidates

    band_metas = []
    for i in range(1, ref.RasterCount + 1):
        band = ref.GetRasterBand(i)
        band_meta_domains = band.GetMetadataDomainList() or []
        all_domain_band_meta = {}
        for ns in band_meta_domains:
            band_meta = band.GetMetadata(ns)
            band_meta = ub.dict_diff(band_meta, {'COMPRESSION'})
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
        >>> from watch.gis.geotiff import *  # NOQA
        >>> from watch.demo.dummy_demodata import dummy_rpc_geotiff_fpath
        >>> gpath = dummy_rpc_geotiff_fpath()
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

    info = {}
    ref = _coerce_gdal_dataset(gpath_or_ref)

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
    aff_wld_crs = ref.GetSpatialRef()
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
                aff_wld_crs = osr.SpatialReference()
                aff_wld_crs.ImportFromEPSG(4326)  # 4326 is the EPSG id WGS84 of lat/lon crs
                # Set traditional because our rpc transform returns x,y
                aff_wld_crs.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
                # Not 100% sure this is correct to do
                aff_wld_crs_type = 'assume-rpc-wgs84-reverse'
                aff_geo_transform = (0.0, 1.0, 0.0, 0.0, 0.0, 1.0)
                if force_affine:
                    raise Exception(
                        'cant force affine without dataset or gcp ref')
            else:
                raise Exception('no dataset or gcps refs or rpc')
        else:
            # gcp_ids = [p.Id for p in gcps]
            aff_geo_transform = gdal.GCPsToGeoTransform(gcps)
    else:
        if aff_wld_crs is None:
            aff_wld_crs = osr.SpatialReference()
            aff_wld_crs.ImportFromWkt(aff_proj)
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
    aff_pxl_to_wld = np.vstack([np.array(aff.column_vectors).T, [0, 0, 1]])
    aff_wld_to_pxl = np.linalg.inv(aff_pxl_to_wld)

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
        wld_crs_type = 'har-coded-rpc-crs'
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

    wgs84_axis_mapping_int = wgs84_crs.GetAxisMappingStrategy()
    wgs84_axis_mapping = axis_mapping_int_to_text(wgs84_axis_mapping_int)

    wld_to_wgs84 = osr.CoordinateTransformation(wld_crs, wgs84_crs)
    wgs84_to_wld = osr.CoordinateTransformation(wgs84_crs, wld_crs)

    if is_rpc:
        wld_to_pxl = rpc_transform.warp_world_to_pixel
        pxl_to_wld = rpc_transform.warp_pixel_to_world
    else:
        pxl_to_wld = aff_pxl_to_wld
        wld_to_pxl = aff_wld_to_pxl

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
        utm_axis_mapping_int = utm_crs.GetAxisMappingStrategy()
        utm_axis_mapping = axis_mapping_int_to_text(utm_axis_mapping_int)

        wld_to_utm = osr.CoordinateTransformation(wld_crs, utm_crs)
        utm_to_wld = osr.CoordinateTransformation(utm_crs, wld_crs)
        utm_corners = wld_corners.warp(wld_to_utm)

        min_utm = utm_corners.data.min(axis=0)
        max_utm = utm_corners.data.max(axis=0)
        meter_extent = max_utm - min_utm
        pxl_extent = np.array([ref.RasterXSize, ref.RasterYSize])
        meter_per_pxl = meter_extent / pxl_extent
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
        utm_wkt = utm_crs.ExportToWkt()
        wld_wkt = wld_crs.ExportToWkt()
        wgs84_wkt = wgs84_crs.ExportToWkt()

        utm_crs_info = {
            'auth': CRS.from_wkt(utm_wkt).to_authority(min_confidence=100),
            'axis_mapping': utm_axis_mapping,
        }

        wld_crs_info = {
            'auth': CRS.from_wkt(wld_wkt).to_authority(min_confidence=100),
            'axis_mapping': wld_axis_mapping,
            'type': wld_crs_type,
        }

        wgs84_crs_info = {
            'auth': CRS.from_wkt(wgs84_wkt).to_authority(min_confidence=100),
            'axis_mapping': wgs84_axis_mapping,
        }

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

        'pxl_to_wld': pxl_to_wld,
        'wgs84_to_wld': wgs84_to_wld,
        'utm_to_wld': utm_to_wld,

        'wld_to_pxl': wld_to_pxl,
        'wld_to_wgs84': wld_to_wgs84,
        'wld_to_utm': wld_to_utm,

        'utm_crs_info': utm_crs_info,
        'wld_crs_info': wld_crs_info,
        'wgs84_crs_info': wgs84_crs_info,

        'bbox_geos': bbox_geos,
        'img_shape': shape,
    })
    return info


def geotiff_filepath_info(gpath):
    """
    Attempt to parse information out of a path to a geotiff file.

    Information provided here is purely heuristic. Generally filepath
    information is not robust.

    Several huerstics are currently implemented for:
        * Sentinal 2
        * Landsat-8
        * WorldView3

    Args:
        gpath (str): a path to an image that uses a standard naming convention
            (may include subdirectories that contain relevant information) .

    Example:
        >>> from watch.gis.geotiff import *  # NOQA
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
        >>> ]
        >>> gpath = inputs[-1]
        >>> print('gpath = {!r}'.format(gpath))
        >>> for gpath in inputs:
        >>>     print('----')
        >>>     info = geotiff_filepath_info(gpath)
        >>>     print('gpath = {}'.format(ub.repr2(gpath, nl=1)))
        >>>     print('info = {}'.format(ub.repr2(info, nl=2)))
        >>>     if len(info['sensor_candidates']) == 0:
        >>>         print(ub.color_text('NO HUERISTIC', 'red'))
        >>>     else:
        >>>         assert len(info['sensor_candidates']) == len(set(info['sensor_candidates']))
        >>>         assert info['filename_meta']

    Ignore:
        >>> from watch.gis.geotiff import *  # NOQA
        >>> import ubelt as ub
        >>> gpath = ub.expandpath('$HOME/remote/namek/data/dvc-repos/smart_watch_dvc/drop1/KR-Pyeongchang-WV/_assets/20170907_a_KRP_011777481_10_0/011777481010_01_003/011777481010_01/011777481010_01_P001_MUL/17SEP07021826-M1BS-011777481010_01_P001.TIF')
        >>> info = geotiff_filepath_info(gpath)
        >>> print('info = {}'.format(ub.repr2(info, nl=1)))


    References:
        .. [LanSatName] https://www.usgs.gov/faqs/what-naming-convention-landsat-collections-level-1-scenes
        .. [S2_Name_2016] https://sentinel.esa.int/web/sentinel/user-guides/sentinel-2-msi/naming-convention
        .. [S3_Name] https://sentinel.esa.int/web/sentinel/user-guides/sentinel-3-altimetry/naming-conventions
    """
    from os.path import basename
    import parse
    base_ext = basename(gpath)
    base, *exts = base_ext.split('.')
    ext = '.'.join(exts)  # NOQA
    parts = gpath.split('/')

    class InvalidFormat(Exception):
        pass

    info = {
        'sensor_candidates': [],
        'filename_meta': {},
    }
    sensor_candidates = info['sensor_candidates']
    meta = info['filename_meta']

    # Landsat filename pattern. See [LanSatName]_
    # LXSS_LLLL_PPPRRR_YYYYMMDD_yyyymmdd_CC_TX
    landsat_pattern = 'L{X}{SS}_{LLL}_{PPPRRR}_{YYYYMMDD}_{yyyymmdd}_{CC}_{TX}'
    result = parse.parse(landsat_pattern, base)
    if result:
        ls_sensor_code_to_text = {
            'C': 'OLI/TIRS',
            'O': 'OLI',
            'T': 'TIRS',
            'E': 'ETM+',
            # 'T': 'TM',  # ambiguous? That's in the spec
            'M': 'MSS',
        }
        try:
            wrs = result.named['PPPRRR']
            sensor_code = result.named['X']
            sat_code = result.named['SS']
            ls_meta = {}
            ls_meta['sensor_code'] = sensor_code
            ls_meta['sat_code'] = sat_code
            ls_meta['sensor_text']: ls_sensor_code_to_text[sensor_code]
            ls_meta['WRS_path'] = wrs[:3]
            ls_meta['WRS_now'] = wrs[3:]
            ls_meta['acquisition_date'] = result.named['YYYYMMDD']
            ls_meta['processing_date'] = result.named['yyyymmdd']
            ls_meta['collection_number'] = result.named['CC']
            ls_meta['collection_category'] = result.named['TX']
            sensor_cand = 'L' + sensor_code + sat_code
        except InvalidFormat:
            pass
        else:
            meta.update(ls_meta)
            sensor_candidates.append(sensor_cand)

    # Sentinal-2 2016+ filename pattern. See [S2_Name_2016]_
    # These filenames are often directories
    # MMM_MSIXXX_YYYYMMDDHHMMSS_Nxxyy_ROOO_Txxxxx_<Product Discriminator>.SAFE
    # SAFE = Standard Archive Format for Europe
    s2_name_2016 = '{MMM}_{MSIXXX}_{YYYYMMDDHHMMSS}_{Nxxyy}_{ROOO}_{Txxxxx}_{Discriminator}'

    for part_ in reversed(parts):
        part = part_.split('.')[0]
        result = parse.parse(s2_name_2016, part)
        if result:
            try:
                mission_id = result.named['MMM']
                if mission_id not in {'S2A', 'S2B'}:
                    raise InvalidFormat
                s2_meta = {}
                s2_meta['mission_id'] = mission_id
                s2_meta['product_level'] = result.named['MSIXXX']
                s2_meta['sense_start_time'] = result.named['YYYYMMDDHHMMSS']
                s2_meta['pdgs_num'] = result.named['Nxxyy']
                s2_meta['relative_oribt_num'] = result.named['ROOO']
                s2_meta['tile_number'] = result.named['Txxxxx']
            except InvalidFormat:
                pass
            else:
                meta.update(s2_meta)
                sensor_candidates.append(mission_id)
                break

    # Files ending in _TCI are true color images based on the Sentinal 2
    # Handbook I'm not sure what the standard for this format is I just know a
    # suffix of _TCI means true color image. ANd these are from sentinal2, I'm
    # guessing on the rest of the format.
    s2_format_guess = '{part1}_{date}_{part2}'
    result = parse.parse(s2_format_guess, base)
    if result:
        if result.named['part2'] == 'TCI':
            if 'acquisition_date' in meta:
                assert meta['acquisition_date'] == result.named['date']
            meta['acquisition_date'] = result.named['date']
            sensor_candidates.append('S2-TrueColor')

    # WorldView3
    # TODO: find a reference for the spec
    wv3_pat = '{date1:w}-{part1:w}-{date2:w}_{num:w}_{part2:w}'
    prefix = base.split('____')[0]
    result = parse.parse(wv3_pat, prefix)
    if result is not None:
        try:
            wv2_meta = {}
            wv2_meta['date1'] = result.named['date1']
            wv2_meta['part1'] = result.named['part1']
            wv2_meta['date2'] = result.named['date2']
            wv2_meta['num'] = result.named['num']
            wv2_meta['part2'] = result.named['part2']
            meta.update(wv2_meta)
        except InvalidFormat:
            pass
        else:
            meta.update(wv2_meta)
            sensor_candidates.append('WV03')

    # Add DG information if it exists
    from watch.io import digital_globe as dg_parser
    try:
        # technically, this does read files, so its not all about the path
        dg_bundle = dg_parser.DigitalGlobeBundle.from_pointer(gpath)
        info['dg_info'] = dg_bundle.data
    except Exception:
        dg_bundle = None

    if dg_bundle is not None:
        for prod_meta in dg_bundle.data['product_metas']:
            sensor_candidates.append(prod_meta['sensorVehicle'])

    info['is_dg_bundle'] = dg_bundle is not None
    return info
