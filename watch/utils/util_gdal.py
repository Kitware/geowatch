"""
SeeAlso
    util_raster.py

References:
    https://gdal.org/programs/gdalwarp.html#cmdoption-gdalwarp-multi
    https://gis.stackexchange.com/a/241810
    https://trac.osgeo.org/gdal/wiki/UserDocs/GdalWarp#WillincreasingRAMincreasethespeedofgdalwarp
    https://github.com/OpenDroneMap/ODM/issues/778


TODO:
    TODO test this and see if it's safe to add:
        --config GDAL_PAM_ENABLED NO
    Removes .aux.xml sidecar files and puts them in the geotiff metadata
    ex. histogram from fmask
    https://stackoverflow.com/a/51075774
    https://trac.osgeo.org/gdal/wiki/ConfigOptions#GDAL_PAM_ENABLED
    https://gdal.org/drivers/raster/gtiff.html#georeferencing
"""
import kwimage
import os
import ubelt as ub


GDAL_VIRTUAL_FILESYSTEM_PREFIX = '/vsi'

# https://gdal.org/user/virtual_file_systems.
# GDAL_VIRTUAL_FILESYSTEMS = [
#     {'prefix': 'vsizip', 'type': 'zip'},
#     {'prefix': 'vsigzip', 'type': None},
#     {'prefix': 'vsitar', 'type': None},
#     {'prefix': 'vsitar', 'type': None},

#     # Networks
#     {'prefix': 'vsicurl', 'type': 'curl'},
#     {'prefix': 'vsicurl_streaming', 'type': None},
#     {'prefix': 'vsis3', 'type': None},
#     {'prefix': 'vsis3_streaming', 'type': None},
#     {'prefix': 'vsigs', 'type': None},
#     {'prefix': 'vsigs_streaming', 'type': None},
#     {'prefix': 'vsiaz', 'type': None},
#     {'prefix': 'vsiaz_streaming', 'type': None},
#     {'prefix': 'vsiadls', 'type': None},
#     {'prefix': 'vsioss', 'type': None},
#     {'prefix': 'vsioss_streaming', 'type': None},
#     {'prefix': 'vsiswift', 'type': None},
#     {'prefix': 'vsiswift_streaming', 'type': None},
#     {'prefix': 'vsihdfs', 'type': None},
#     {'prefix': 'vsiwebhdfs', 'type': None},

#     #
#     {'prefix': 'vsistdin', 'type': None},
#     {'prefix': 'vsistdout', 'type': None},
#     {'prefix': 'vsimem', 'type': None},
#     {'prefix': 'vsisubfile', 'type': None},
#     {'prefix': 'vsisparse', 'type': None},
#     {'prefix': 'vsicrypt', 'type': None},
# ]


gdalwarp_performance_opts = ub.paragraph('''
        -multi
        --config GDAL_CACHEMAX 15%
        -wm 15%
        -co NUM_THREADS=ALL_CPUS
        -wo NUM_THREADS=1
        ''')


def _demo_geoimg_with_nodata():
    """
    Example:
        fpath = _demo_geoimg_with_nodata()
        self = LazyGDalFrameFile.demo()

    """
    import kwimage
    import numpy as np
    from osgeo import osr
    # gdal.UseExceptions()

    # Make a dummy geotiff
    imdata = kwimage.grab_test_image('airport')
    dpath = ub.Path.appdir('watch/test/geotiff').ensuredir()
    geo_fpath = dpath / 'dummy_geotiff.tif'

    # compute dummy values for a geotransform to CRS84
    img_h, img_w = imdata.shape[0:2]
    img_box = kwimage.Boxes([[0, 0, img_w, img_h]], 'xywh')
    img_corners = img_box.corners()

    # wld_box = kwimage.Boxes([[lon_x, lat_y, 0.0001, 0.0001]], 'xywh')
    # wld_corners = wld_box.corners()
    lat_y = 40.060759
    lon_x = 116.613095
    # lat_y_off = 0.0001
    # lat_x_off = 0.0001
    # Pretend this is a big spatial region
    lat_y_off = 0.1
    lat_x_off = 0.1
    # hard code so north is up
    wld_corners = np.array([
        [lon_x - lat_x_off, lat_y + lat_y_off],
        [lon_x - lat_x_off, lat_y - lat_y_off],
        [lon_x + lat_x_off, lat_y - lat_y_off],
        [lon_x + lat_x_off, lat_y + lat_y_off],
    ])
    transform = kwimage.Affine.fit(img_corners, wld_corners)

    nodata = -9999

    srs = osr.SpatialReference()
    srs.ImportFromEPSG(4326)
    crs = srs.ExportToWkt()

    # Set a region to be nodata
    imdata = imdata.astype(np.int16)
    imdata[-100:] = nodata
    imdata[0:200:, -200:-180] = nodata

    kwimage.imwrite(geo_fpath, imdata, backend='gdal', nodata=-9999, crs=crs, transform=transform)
    return geo_fpath


def gdal_single_translate(in_fpath, out_fpath, pixel_box, blocksize=256,
                          compress='DEFLATE'):
    """
    Crops using pixels

    Example:
        >>> from watch.utils.util_gdal import *  # NOQA
        >>> from watch.utils.util_gdal import _demo_geoimg_with_nodata
        >>> from watch.gis import geotiff
        >>> in_fpath = ub.Path(_demo_geoimg_with_nodata())
        >>> info = geotiff.geotiff_crs_info(in_fpath)

        >>> # Test CRS84 cropping
        >>> wgs84_poly = kwimage.Polygon(exterior=info['wgs84_corners'])
        >>> assert info['wgs84_crs_info']['axis_mapping'] == 'OAMS_AUTHORITY_COMPLIANT'
        >>> crs84_epsg = int(info['wgs84_crs_info']['auth'][1])
        >>> crs84_space_box = wgs84_poly.scale(0.5, about='center').to_boxes().transpose()
        >>> crs84_out_fpath = in_fpath.augment(suffix='_crs84_crop')
        >>> gdal_single_warp(in_fpath, crs84_out_fpath, local_epsg=crs84_epsg, space_box=crs84_space_box)

        >>> # Test UTM cropping
        >>> utm_poly = kwimage.Polygon(exterior=info['utm_corners'])
        >>> utm_epsg = int(info['utm_crs_info']['auth'][1])
        >>> utm_space_box = utm_poly.scale(0.5, about='center').to_boxes()
        >>> utm_out_fpath = in_fpath.augment(suffix='_utmcrop')
        >>> gdal_single_warp(in_fpath, utm_out_fpath, local_epsg=utm_epsg, space_box=utm_space_box, box_epsg=utm_epsg)

        >>> # Test Pixel cropping
        >>> pxl_poly = kwimage.Polygon(exterior=info['pxl_corners'])
        >>> pixel_box = pxl_poly.scale(0.5, about='center').to_boxes()
        >>> pxl_out_fpath = in_fpath.augment(suffix='_pxlcrop')
        >>> gdal_single_translate(in_fpath, pxl_out_fpath, pixel_box=pixel_box)

        >>> # xdoctest: +REQUIRES(--show)
        >>> import kwplot
        >>> kwplot.autompl()
        >>> imdata0 = kwimage.normalize(kwimage.imread(in_fpath, nodata='float'))
        >>> imdata1 = kwimage.normalize(kwimage.imread(crs84_out_fpath, nodata='float'))
        >>> imdata2 = kwimage.normalize(kwimage.imread(utm_out_fpath, nodata='float'))
        >>> imdata3 = kwimage.normalize(kwimage.imread(pxl_out_fpath, nodata='float'))
        >>> kwplot.imshow(imdata0, pnum=(1, 4, 1), title='orig')
        >>> kwplot.imshow(imdata1, pnum=(1, 4, 2), title='crs84-crop')
        >>> kwplot.imshow(imdata2, pnum=(1, 4, 3), title='utm-crop')
        >>> kwplot.imshow(imdata3, pnum=(1, 4, 4), title='pxl-crop')

        print(ub.cmd('gdalinfo ' + str(in_fpath))['out'])
        print(ub.cmd('gdalinfo ' + str(crs84_out_fpath))['out'])
        print(ub.cmd('gdalinfo ' + str(utm_out_fpath))['out'])
        print(ub.cmd('gdalinfo ' + str(pxl_out_fpath))['out'])
    """
    xoff, yoff, xsize, ysize = pixel_box.to_xywh().data[0]
    template_parts = [
        '''
        gdal_translate
        --debug off
        '''
    ]

    if compress == 'RAW':
        compress = 'NONE'

    # Use the new COG output driver
    template_parts.append(f'''
        -of COG
        -co OVERVIEWS=AUTO
        -co BLOCKSIZE={blocksize}
        -co COMPRESS={compress}
        ''')

    tmp_fpath = out_fpath.augment(suffix='.tmp')

    template_parts.append(f'-srcwin {xoff} {yoff} {xsize} {ysize}')
    template_parts.append(f'{in_fpath} {tmp_fpath}')
    template = ' '.join(template_parts)

    command = template.format(template)
    command = ub.paragraph(command)
    if 0:
        print(ub.paragraph(command))
    cmd_info = ub.cmd(command, verbose=0, check=True)  # NOQA
    # if cmd_info['ret'] != 0:
    #     # print('\n\nCOMMAND FAILED: {!r}'.format(command))
    #     # print(cmd_info['out'])
    #     # print(cmd_info['err'])
    #     raise Exception(cmd_info['err'])

    os.rename(tmp_fpath, out_fpath)


def gdal_single_warp(in_fpath,
                     out_fpath,
                     space_box=None,
                     local_epsg=4326,
                     box_epsg=4326,
                     nodata=None,
                     rpcs=None,
                     blocksize=256,
                     compress='DEFLATE',
                     use_perf_opts=False,
                     as_vrt=False,
                     use_te_geoidgrid=False,
                     dem_fpath=None,
                     tries=0,
                     verbose=0):
    r"""
    Wrapper around gdalwarp

    Args:
        in_fpath (PathLike): input geotiff path

        out_fpath (PathLike): output geotiff path

        space_box (kwimage.Boxes):
            Should be traditional crs84 ltrb (or lbrt?) -- i.e.
            (lonmin, latmin, lonmax, latmax) - when box_epsg is 4326

        local_epsg (int):
            EPSG code for the CRS the final geotiff will be projected into.
            This should be the UTM zone for the region if known. Otherwise
            It can be 4326 to project into WGS84 or CRS84 (not sure which
            axis ordering it will use by default).

        box_epsg (int):
            this is the EPSG of the bounding box. Should usually be 4326.

        nodata (int | None):
            only specify if in_fpath does not already have a nodata value

        rpcs (dict): the "rpc_transform" from
            ``watch.gis.geotiff.geotiff_crs_info``, if that information
            is available and orthorectification is desired.

        use_perf_opts (bool): undocumented
        as_vrt (bool): undocumented
        use_te_geoidgrid (bool): undocumented
        dem_fpath (bool): undocumented

        tries (int): gdal can be flakey, set to force some number of retries

    TODO:
        - [ ] This should be a kwgeo function?

    Ignore:
        in_fpath =
        s3://landsat-pds/L8/001/002/LC80010022016230LGN00/LC80010022016230LGN00_B1.TIF?useAnon=true&awsRegion=US_WEST_2

        gdalwarp 's3://landsat-pds/L8/001/002/LC80010022016230LGN00/LC80010022016230LGN00_B1.TIF?useAnon=true&awsRegion=US_WEST_2' foo.tif

    aws s3 --profile iarpa cp s3://kitware-smart-watch-data/processed/ta1/drop1/mtra/f9e7e52029bb4dfaadfe306e92641481/S2A_MSI_L2A_T23KPQ_20190509_20211103_SR_B05.tif foo.tif

    gdalwarp 's3://kitware-smart-watch-data/processed/ta1/drop1/mtra/f9e7e52029bb4dfaadfe306e92641481/S2A_MSI_L2A_T23KPQ_20190509_20211103_SR_B05.tif' bar.tif

    Note:
        Proof of concept for warp from S3:

        aws s3 --profile iarpa ls s3://kitware-smart-watch-data/processed/ta1/drop1/mtra/0bff43f318c14a97b19b682a36f28d26/

        gdalinfo \
            --config AWS_DEFAULT_PROFILE "iarpa" \
            "/vsis3/kitware-smart-watch-data/processed/ta1/drop1/mtra/0bff43f318c14a97b19b682a36f28d26/LC08_L2SP_017039_20190404_20211102_02_T1_T17RMP_B7_BRDFed.tif"

        gdalwarp \
            --config AWS_DEFAULT_PROFILE "iarpa" \
            -te_srs epsg:4326 \
            -te -81.51 29.99 -81.49 30.01 \
            -t_srs epsg:32617 \
            -overwrite \
            -of COG \
            -co OVERVIEWS=AUTO \
            "/vsis3/kitware-smart-watch-data/processed/ta1/drop1/mtra/0bff43f318c14a97b19b682a36f28d26/LC08_L2SP_017039_20190404_20211102_02_T1_T17RMP_B7_BRDFed.tif" \
            partial_crop2.tif
        gdalinfo partial_crop2.tif
        kwplot partial_crop2.tif

        gdalinfo \
            --config AWS_DEFAULT_PROFILE "iarpa" \
            --config AWS_CONFIG_FILE "$HOME/.aws/config" \
            --config CPL_AWS_CREDENTIALS_FILE "$HOME/.aws/credentials" \
            "/vsis3/kitware-smart-watch-data/processed/ta1/drop1/mtra/f9e7e52029bb4dfaadfe306e92641481/S2A_MSI_L2A_T23KPQ_20190509_20211103_SR_B05.tif"

        gdalwarp \
            --config AWS_DEFAULT_PROFILE "iarpa" \
            --config AWS_CONFIG_FILE "$HOME/.aws/config" \
            --config CPL_AWS_CREDENTIALS_FILE "$HOME/.aws/credentials" \
            -te_srs epsg:4326 \
            -te -43.51 -23.01 -43.49 -22.99 \
            -t_srs epsg:32723 \
            -overwrite \
            -of COG \
            "/vsis3/kitware-smart-watch-data/processed/ta1/drop1/mtra/f9e7e52029bb4dfaadfe306e92641481/S2A_MSI_L2A_T23KPQ_20190509_20211103_SR_B05.tif" \
            partial_crop.tif
        kwplot partial_crop.tif

    Notes:
        In gdalwarp:
            -s_srs - Set source spatial reference
            -t_srs - Set target spatial reference

            -te_srs - Specifies the SRS in which to interpret the coordinates given with -te.
            -te - Set georeferenced extents of output file to be created
    """

    # Coordinate Reference System of the "target" destination image
    # t_srs = target spatial reference for output image
    if local_epsg is None:
        target_srs = 'epsg:4326'
    else:
        target_srs = 'epsg:{}'.format(local_epsg)

    template_parts = [
        '''
        gdalwarp
        --debug off
        -t_srs {target_srs}
        -overwrite
        '''
    ]

    template_kw = {
        'target_srs': target_srs,
        'SRC': in_fpath,
        'DST': out_fpath,
    }

    if as_vrt:
        template_parts.append('''
            -of VRT
            ''')
    else:
        if compress == 'RAW':
            compress = 'NONE'

        # Use the new COG output driver
        template_parts.append('''
            -of COG
            -co OVERVIEWS=AUTO
            -co BLOCKSIZE={blocksize}
            -co COMPRESS={compress}
            ''')
        template_kw.update(**{
            'blocksize': blocksize,
            'compress': compress,
        })

    if space_box is not None:
        # Data is from geo-pandas so this should be traditional order
        lonmin, latmin, lonmax, latmax = space_box.data[0]

        # Coordinate Reference System of the "te" crop coordinates
        # te_srs = spatial reference of query points
        # This means space_box currently MUST be in CRS84
        # crop_coordinate_srs = 'epsg:4326'
        crop_coordinate_srs = 'epsg:{}'.format(box_epsg)

        template_parts.append('''
            -te {xmin} {ymin} {xmax} {ymax}
            -te_srs {crop_coordinate_srs}
            ''')
        template_kw.update(
            **{
                'crop_coordinate_srs': crop_coordinate_srs,
                'ymin': latmin,
                'xmin': lonmin,
                'ymax': latmax,
                'xmax': lonmax,
            })

    if nodata is not None:
        # TODO: Use cloudmask?
        template_parts.append('''
            -srcnodata {NODATA_VALUE} -dstnodata {NODATA_VALUE}
            ''')
        template_kw['NODATA_VALUE'] = nodata

    # HACK TO FIND an appropriate DEM file
    if rpcs is not None:
        if dem_fpath is not None:
            template_parts.append(
                ub.paragraph('''
                -rpc -et 0
                -to RPC_DEM={dem_fpath}
                '''))
            template_kw['dem_fpath'] = dem_fpath
        else:
            dems = rpcs.elevation
            if hasattr(dems, 'find_reference_fpath'):
                # TODO: get a better DEM path for this image if possible
                dem_fpath, dem_info = dems.find_reference_fpath(latmin, lonmin)
                template_parts.append(
                    ub.paragraph('''
                    -rpc -et 0
                    -to RPC_DEM={dem_fpath}
                    '''))
                template_kw['dem_fpath'] = dem_fpath
            else:
                dem_fpath = None
                template_parts.append('-rpc -et 0')

    if use_te_geoidgrid:
        # assumes source CRS is WGS84
        # https://smartgitlab.com/TE/annotations/-/wikis/WorldView-Annotations#notes-on-the-egm96-geoidgrid-file
        from watch.rc import geoidgrid_path
        template_parts.append('''
            -s_srs "+proj=longlat +datum=WGS84 +no_defs +geoidgrids={geoidgrid_path}"
            ''')
        template_kw['geoidgrid_path'] = geoidgrid_path()

    if use_perf_opts:
        template_parts.append(gdalwarp_performance_opts)
    else:
        # use existing options
        template_parts.append(
            ub.paragraph('''
            -multi
            --config GDAL_CACHEMAX 500
            -wm 500
            -co NUM_THREADS=2
            '''))

    template_parts.append('{SRC} {DST}')
    template = ' '.join(template_parts)

    command = template.format(**template_kw)
    command = ub.paragraph(command)
    # if verbose:
    #     print(ub.paragraph(command))
    # cmd_info = ub.cmd(command, verbose=verbose, check=True)  # NOQA
    import subprocess
    try:
        if 1:
            import retry
            retry.api.retry_call(
                ub.cmd, (command,), dict(check=True, verbose=verbose),
                tries=tries, delay=1, exceptions=subprocess.CalledProcessError)
        else:
            ub.cmd(command, check=True, verbose=verbose)
    except subprocess.CalledProcessError as ex:
        print('\n\nCOMMAND FAILED: {!r}'.format(ex.cmd))
        print(ex.stdout)
        print(ex.stderr)
        raise
    # if cmd_info['ret'] != 0:
    #     import subprocess
    #     print('\n\nCOMMAND FAILED: {!r}'.format(command))
    #     print(cmd_info['out'])
    #     print(cmd_info['err'])
    #     raise Exception(cmd_info['err'])


def gdal_multi_warp(in_fpaths, out_fpath, *args, nodata=None, tries=0, **kwargs):
    """
    See gdal_single_warp() for args

    Ignore:
        # Uses data from the data cube with extra=1
        from watch.cli.coco_align_geotiffs import *  # NOQA
        cube, region_df = SimpleDataCube.demo(with_region=True, extra=True)
        local_epsg = 32635
        space_box = kwimage.Polygon.from_shapely(region_df.geometry.iloc[1]).bounding_box().to_ltrb()
        dpath = ub.ensure_app_cache_dir('smart_watch/test/gdal_multi_warp')
        out_fpath = join(dpath, 'test_multi_warp.tif')
        in_fpath1 = cube.coco_dset.get_image_fpath(2)
        in_fpath2 = cube.coco_dset.get_image_fpath(3)
        in_fpaths = [in_fpath1, in_fpath2]
        rpcs = None
        gdal_multi_warp(in_fpaths, out_fpath, space_box, local_epsg, rpcs)
    """
    # Warp then merge
    import tempfile

    # Write to a temporary file and then rename the file to the final
    # Destination so ctrl+c doesn't break everything
    tmp_out_fpath = ub.augpath(out_fpath, prefix='.tmp.')

    tempfiles = []  # hold references
    warped_gpaths = []
    for in_fpath in in_fpaths:
        tmpfile = tempfile.NamedTemporaryFile(suffix='.tif')
        tempfiles.append(tmpfile)
        tmp_out = tmpfile.name
        gdal_single_warp(in_fpath, tmp_out, *args, nodata=nodata, tries=tries,
                         **kwargs)
        warped_gpaths.append(tmp_out)

    if nodata is not None:
        from watch.utils import util_raster
        valid_polygons = []
        for tmp_out in warped_gpaths:
            sh_poly = util_raster.mask(tmp_out,
                                       tolerance=10,
                                       default_nodata=nodata)
            valid_polygons.append(sh_poly)
        valid_areas = [p.area for p in valid_polygons]
        # Determine order by valid data
        warped_gpaths = list(
            ub.sorted_vals(ub.dzip(warped_gpaths, valid_areas)).keys())
        warped_gpaths = warped_gpaths[::-1]
    else:
        # Last image is copied over earlier ones, but we expect first image to
        # be the primary one, so reverse order
        warped_gpaths = warped_gpaths[::-1]

    merge_cmd_parts = ['gdal_merge.py']
    if nodata is not None:
        merge_cmd_parts.extend(['-n', str(nodata)])
    merge_cmd_parts.extend(['-o', tmp_out_fpath])
    merge_cmd_parts.extend(warped_gpaths)
    merge_cmd = ' '.join(merge_cmd_parts)
    verbose = kwargs.get('verbose', 0)

    import subprocess
    try:
        if 1:
            import retry
            retry.api.retry_call(
                ub.cmd, (merge_cmd,), dict(check=True, verbose=verbose),
                tries=tries, delay=1)
        else:
            ub.cmd(merge_cmd, check=True, verbose=verbose)
    except subprocess.CalledProcessError as ex:
        print('\n\nCOMMAND FAILED: {!r}'.format(ex.cmd))
        print(ex.stdout)
        print(ex.stderr)
        raise
    os.rename(tmp_out_fpath, out_fpath)

    if 0:
        # Debugging
        datas = []
        for p in warped_gpaths:
            d = kwimage.imread(p)
            d = kwimage.normalize_intensity(d, nodata=0)
            datas.append(d)

        import kwplot
        kwplot.autompl()
        combo = kwimage.imread(out_fpath)
        combo = kwimage.normalize_intensity(combo, nodata=0)
        datas.append(combo)
        kwplot.imshow(kwimage.stack_images(datas, axis=1))

        datas2 = []
        for p in in_fpaths:
            d = kwimage.imread(p)
            d = kwimage.normalize_intensity(d, nodata=0)
            datas2.append(d)
        kwplot.imshow(kwimage.stack_images(datas2, axis=1), fnum=2)


def list_gdal_drivers():
    """
    List all drivers currently available to GDAL to create a raster

    Returns:
        list((driver_shortname, driver_longname, list(driver_file_extension)))

    Example:
        >>> from watch.utils.util_gdal import *
        >>> drivers = list_gdal_drivers()
        >>> print('drivers = {}'.format(ub.repr2(drivers, nl=1)))
        >>> assert ('GTiff', 'GeoTIFF', ['tif', 'tiff']) in drivers
    """
    from osgeo import gdal
    result = []
    for idx in range(gdal.GetDriverCount()):
        driver = gdal.GetDriver(idx)
        if driver:
            metadata = driver.GetMetadata()
            if metadata.get(gdal.DCAP_CREATE) == 'YES' and metadata.get(
                    gdal.DCAP_RASTER) == 'YES':
                name = driver.GetDescription()
                longname = metadata.get('DMD_LONGNAME')
                exts = metadata.get('DMD_EXTENSIONS')
                if exts is None:
                    exts = []
                else:
                    exts = exts.split(' ')
                result.append((name, longname, exts))
    return result


def GdalOpen(path, mode='r', **kwargs):
    """
    A simple context manager for friendlier gdal use.

    Returns:
        GdalDataset

    Example:
        >>> # xdoctest: +REQUIRES(--network)
        >>> from watch.utils.util_gdal import *
        >>> from osgeo import gdal
        >>> from watch.demo.landsat_demodata import grab_landsat_product
        >>> path = grab_landsat_product()['bands'][0]
        >>> #
        >>> # standard use:
        >>> dataset = gdal.Open(path)
        >>> print(dataset.GetDescription())  # do stuff
        >>> del dataset  # or 'dataset = None'
        >>> #
        >>> # equivalent:
        >>> with GdalOpen(path) as dataset:
        >>>     print(dataset.GetDescription())  # do stuff
        >>> #
        >>> # open for writing:
        >>> with GdalOpen(path, gdal.GA_Update) as dataset:
        >>>     print(dataset.GetDescription())  # do stuff
    """
    return GdalDataset.open(path, mode=mode, **kwargs)


class GdalDataset(ub.NiceRepr):
    """
    A wrapper around `gdal.Open` and the underlying dataset it returns.

    This object is completely transparent and offers the same API as the
    :class:`osgeo.gdal.Dataset` returned by :func`:`osgeo.gdal.GDalOpen``.

    This object can be used as a context manager. By default the GDAL dataset
    is opened when the object is created, and it is closed when either
    ``close`` is called or the `__exit__` method is called by a context
    manager. When the object is closed the underlying GDAL objet is
    dereferenced and garbage collected.

    Args:
        path (PathLike): a path or string referencing a gdal image file

        mode (str | int): a gdal GA (Gdal Access) integer code or
            a string that can be: 'readonly' or 'update' or the equivalent
            standard mode codes: 'r' and 'w+'.

        virtual_retries (int):
            If the path is a reference to a virtual file system
            (i.e. starts with vsi) then we try to open it this many times
            before we finally fail.

    Example:
        >>> # Demonstrate use cases of this object
        >>> from watch.utils.util_gdal import *
        >>> import kwimage
        >>> # Grab demo path we can test with
        >>> path = kwimage.grab_test_image_fpath()
        >>> #
        >>> #
        >>> # Method1: Use GDalOpen exactly the same as gdal.Open
        >>> ref = GdalDataset.open(path)
        >>> print(f'{ref=!s}')
        >>> assert not ref.closed
        >>> ref.GetDescription()  # use GDAL API exactly as-is
        >>> assert not ref.closed
        >>> ref.close()  # Except you can now do this
        >>> print(f'{ref=!s}')
        >>> assert ref.closed
        >>> #
        >>> #
        >>> # Method2: Use GDalOpen exactly the same as gdal.GdalDataset
        >>> with GdalDataset.open(path, mode='r') as ref:
        >>>     ref.GetDescription()  # do stuff
        >>>     print(f'{ref=!s}')
        >>>     assert not ref.closed
        >>> print(f'{ref=!s}')
        >>> assert ref.closed

    Example:
        >>> # Test virtual filesystem
        >>> # xdoctest: +REQUIRES(--network)
        >>> from watch.utils.util_gdal import *  # NOQA
        >>> path = '/vsicurl/https://i.imgur.com/KXhKM72.png'
        >>> ref = GdalDataset.open(path)
        >>> data = ref.GetRasterBand(1).ReadAsArray()
        >>> assert data.sum() == 37109758
    """

    def __init__(self, __ref, _path='?', _str_mode='?'):
        """
        Do not call this method directly. Use `GdalDataset.open`
        """
        self.__ref = __ref  # This is a private variable
        self._path = _path
        self._str_mode = _str_mode

    @classmethod
    def open(cls, path, mode='r', virtual_retries=3):
        """
        Create a new dataset
        """
        from osgeo import gdal
        _path = os.fspath(path)

        if isinstance(mode, str):
            # https://mkyong.com/python/python-difference-between-r-w-and-a-in-open/
            if mode in {'readonly', 'r'}:
                mode = gdal.GA_ReadOnly
            elif mode == {'update', 'w+'}:
                mode = gdal.GA_Update
            else:
                raise KeyError(mode)
        if mode == gdal.GA_ReadOnly:
            _str_mode = 'r'
        elif mode == gdal.GA_Update:
            _str_mode = 'w+'
        else:
            raise ValueError(mode)

        # Exceute gdal open with retries if it is a virtual system
        __ref = None
        try:
            __ref = gdal.Open(_path, mode)
            if __ref is None:
                # gdal.GetLastErrorType()
                # gdal.GetLastErrorNo()
                msg = gdal.GetLastErrorMsg()
                raise RuntimeError(msg)
        except Exception:
            import time
            if _path.startswith(GDAL_VIRTUAL_FILESYSTEM_PREFIX):
                for _ in range(virtual_retries):
                    try:
                        __ref = gdal.Open(_path, mode)
                        if __ref is None:
                            msg = gdal.GetLastErrorMsg()
                            raise RuntimeError(msg)
                    except Exception:
                        time.sleep(0.01)
                    else:
                        break
            if __ref is None:
                raise
        self = cls(__ref, _path, _str_mode)
        return self

    @classmethod
    def coerce(cls, data, mode=None, **kwargs):
        """
        Ensures the underlying object is a gdal dataset.
        """
        from osgeo import gdal
        import pathlib
        if mode is None:
            mode = gdal.GA_ReadOnly
        if isinstance(data, str):
            ref = cls.open(data, mode, **kwargs)
        elif isinstance(data, pathlib.Path):
            ref = cls.open(data, mode, **kwargs)
        elif isinstance(data, gdal.Dataset):
            ref = cls(data)
        elif isinstance(data, GdalDataset):
            ref = data
        else:
            raise TypeError(type(data))
        if ref is None:
            raise Exception('data={} is not a gdal dataset'.format(data))
        return ref

    @property
    def closed(self):
        return self.__ref is None

    @property
    def mode(self):
        return self._mode

    def close(self):
        """
        Closes this dataset.

        Part of the GDalOpen Wrapper.
        Closes this dataset and dereferences the underlying GDAL object.

        Note: this will not work if the `__ref` attribute as accessed outside of
        this wrapper class.
        """
        self.__ref = None

    def __nice__(self):
        mode_part = 'closed' if self.closed else f'mode={self._str_mode!r}'
        return f'{self._path!r} {mode_part}'

    def __dir__(self):
        attrs = super().__dir__()
        if self.__ref is not None:
            attrs = attrs + dir(self.__ref)
        return attrs

    def __getattr__(self, key):
        """
        Expose the API of the underlying gdal.Dataset object

        References:
            https://stackoverflow.com/questions/26091833/proxy-object-in-python
        """
        if self.__ref is None:
            raise AttributeError(key)
        return getattr(self.__ref, key)

    def __enter__(self):
        """
        Entering the context manager simply returns
        """
        return self

    def __exit__(self, *exc):
        """
        Exiting the context manager forces the gdal object closed.
        """
        self.close()
