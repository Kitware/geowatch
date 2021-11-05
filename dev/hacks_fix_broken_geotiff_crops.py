class GdalErrorHandler(object):
    """
    References:
        https://gdal.org/api/python_gotchas.html#exceptions-raised-in-custom-error-handlers-do-not-get-caught

    SeeAlso:
        'Error',
        'ErrorReset',
        'GARIO_ERROR',
        'GetErrorCounter',
        'GetLastErrorMsg',
        'GetLastErrorNo',
        'GetLastErrorType',
        'OF_VERBOSE_ERROR',
        'PopErrorHandler',
        'PushErrorHandler',
        'SetCurrentErrorHandlerCatchDebug',
        'SetErrorHandler',
        'VSIErrorReset',
        'VSIGetLastErrorMsg',
        'VSIGetLastErrorNo'
    """
    def __init__(self):
        self.err_level = None
        self.err_no = None
        self.err_msg = None
        self.was_using_exceptions = None
        self.reset()

    def handler(self, err_level, err_no, err_msg):
        self.err_level = err_level
        self.err_no = err_no
        self.err_msg = err_msg

    def reset(self):
        from osgeo import gdal
        self.err_level = gdal.CE_None
        self.err_no = 0
        self.err_msg = ''

    def __enter__(self):
        from osgeo import gdal
        self.was_using_exceptions = gdal.GetUseExceptions()
        gdal.UseExceptions()
        gdal.PushErrorHandler(self.handler)

    def __exit__(self, a, b, c):
        from osgeo import gdal
        if not self.was_using_exceptions:
            gdal.DontUseExceptions()
        gdal.PopErrorHandler()


def _hack_check_and_fix_broken(bundle_dpath, img):
    """
    NOTE: this should be an option elsewhere, perhaps in the align script
    """
    print('HACK CHECK AND FIXING!!!!')
    from os.path import join
    from osgeo import gdal
    from kwcoco.coco_image import CocoImage
    coco_img = CocoImage(img)

    err = GdalErrorHandler()
    bad_bands = []
    with err:
        for obj in coco_img.iter_asset_objs():
            fpath = join(bundle_dpath, obj['file_name'])
            gdal_ds = gdal.Open(fpath, gdal.GA_ReadOnly)
            if err.err_level == gdal.CE_Warning:
                err.reset()
                bad_bands.append(obj['channels'])

            # print('err.err_level = {!r}'.format(err.err_level))
            # for band_idx in range(gdal_ds.RasterCount):
            #     band = gdal_ds.GetRasterBand(band_idx + 1)
            #     print('band_idx = {!r}'.format(band_idx))
            #     print('band = {!r}'.format(band))
            gdal_ds = None  # NOQA

    for chan_group in bad_bands:
        print('BAD chan_group = {!r}'.format(chan_group))
        _hack_check_and_fix_broken(bundle_dpath, img, chan_group)


def _hack_fix_align_warp(bundle_dpath, img, chan_group):
    print('HACK FIXING!!!!')
    import kwimage
    from os.path import join, exists
    # HACK IT: TODO: make the align script to consistency checks
    found = None

    from kwcoco.coco_image import CocoImage
    coco_img = CocoImage(img)

    for obj in coco_img.iter_asset_objs():
        if obj['channels'] == chan_group:
            found = obj
            break

    if found is None:
        import xdev
        xdev.embed()

    parent_fpath = join(bundle_dpath, found['parent_file_name'])
    if not exists(parent_fpath):
        # SUPER HACK
        parent_fpath = join(bundle_dpath, '..', found['parent_file_name'])

    if not exists(parent_fpath):
        raise Exception('cannot fix, cannot find parent')

    bad_fpath = join(bundle_dpath, found['file_name'])
    corner = kwimage.Polygon.coerce(found['geos_corners'])
    lonmax, latmax = corner.data['exterior'].data.max(axis=0)
    lonmin, latmin = corner.data['exterior'].data.min(axis=0)

    import watch
    candidate_utm_codes = [
        watch.gis.spatial_reference.utm_epsg_from_latlon(latmin, lonmin),
        watch.gis.spatial_reference.utm_epsg_from_latlon(latmax, lonmax),
        watch.gis.spatial_reference.utm_epsg_from_latlon(latmax, lonmin),
        watch.gis.spatial_reference.utm_epsg_from_latlon(latmin, lonmax),
        watch.gis.spatial_reference.utm_epsg_from_latlon(
            ((latmin + latmax) / 2), ((lonmin + lonmax) / 2)),
    ]
    utm_epsg_zone = ub.argmax(ub.dict_hist(candidate_utm_codes))

    compress = 'NONE'
    blocksize = 64
    crop_coordinate_srs = 'epsg:4326'
    target_srs = 'epsg:{}'.format(utm_epsg_zone)
    src_gpath = parent_fpath
    # dst_gpath = './tmp.tif'
    dst_gpath = bad_fpath

    # Use the new COG output driver
    prefix_template = (
        '''
        gdalwarp
        -multi
        --config GDAL_CACHEMAX 500 -wm 500
        --debug off
        -te {xmin} {ymin} {xmax} {ymax}
        -te_srs {crop_coordinate_srs}
        -t_srs {target_srs}
        -of COG
        -co OVERVIEWS=NONE
        -co BLOCKSIZE={blocksize}
        -co COMPRESS={compress}
        -co NUM_THREADS=2
        -overwrite
        ''')

    template_kw = {
        'crop_coordinate_srs': crop_coordinate_srs,
        'target_srs': target_srs,
        'ymin': latmin,
        'xmin': lonmin,
        'ymax': latmax,
        'xmax': lonmax,
        'blocksize': blocksize,
        'compress': compress,
        'SRC': src_gpath,
        'DST': dst_gpath,
    }
    template = ub.paragraph(
        prefix_template +
        '{SRC} {DST}')
    command = template.format(**template_kw)
    cmd_info = ub.cmd(command, verbose=0)  # NOQA
    if cmd_info['ret'] != 0:
        print('\n\nCOMMAND FAILED: {!r}'.format(command))
        raise Exception(cmd_info['err'])
