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
import subprocess
import retry


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


class DummyLogger:
    def warning(self, msg, *args):
        print(msg % args)


def _demo_geoimg_with_nodata():
    """
    Example:
        fpath = _demo_geoimg_with_nodata()
        self = LazyGDalFrameFile.demo()

    """
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


# TODO: simplified API for gdalwarp and gdal_translate
# def gdal_single_crop(in_fpath, out_fpath, space_box=None, local_epsg=4326,
#                      box_epsg=4326, nodata=None, rpcs=None, blocksize=256,
#                      compress='DEFLATE', as_vrt=False,
#                      use_te_geoidgrid=False, dem_fpath=None, tries=1,
#                      verbose=0):
#     """
#     Wrapper around gdal_single_translate and gdal_single_warp

#     Args:
#         in_fpath (PathLike): geotiff to translate

#         out_fpath (PathLike): output geotiff

#         pixel_box (kwimage.Boxes): box to crop to in pixel space.

#         blocksize (int): COG tile size

#         compress (str): gdal compression

#         verbose (int): verbosity level

#     Ignore:
#         print(ub.cmd('gdalinfo ' + str(in_fpath))['out'])
#         print(ub.cmd('gdalinfo ' + str(crs84_out_fpath))['out'])
#         print(ub.cmd('gdalinfo ' + str(utm_out_fpath))['out'])
#         print(ub.cmd('gdalinfo ' + str(pxl_out_fpath))['out'])
#     """

#     if box_epsg == 'pixel':
#         return gdal_single_translate()
#     else:
#         return gdal_single_warp()


class GDalCommandBuilder:
    """
    Helper for templating out gdal commands

    Example:
        >>> from watch.utils.util_gdal import *  # NOQA
        >>> builder = GDalCommandBuilder('madeupcmd')
        >>> builder.flags.append('-flag1')
        >>> builder.flags.append('-flag2')
        >>> builder['-of'] = 'VRT'
        >>> builder['-f1'] = 'flag1'
        >>> builder['-f2'] = 'a b c d'
        >>> builder['-xywh'] = ['x', 'y', 'w', 'h']
        >>> builder.options['-oo']['FOO'] = 'bar'
        >>> builder.options['-oo']['BAZ'] = 'biz'
        >>> builder.options['--config']['CPL_LOG'] = '/dev/null'
        >>> builder.options['--config']['SPACE_PATH'] = '/path with/spaces'
        >>> builder.append('pos1')
        >>> builder.append('pos2')
        >>> builder.set_cog_options()
        >>> command = builder.finalize()
        >>> print(command)

    """
    def __init__(self, command_name):
        self.command_name = command_name

        # hyphen prefixed flags that have no value
        self._flags = []

        # key/value options where the key can only be specified once
        self._single_lut = {}

        # superkey -> (key/value) options where the superkey can be
        # specified more than once
        self._multi_lut = ub.ddict(dict)
        self._multi_lut['-co'] = {}
        self._multi_lut['--config'] = {}

        # positional arguments at the end of the command
        self._positional = []

    @property
    def config_options(self):
        # https://gdal.org/user/configoptions.html#configoptions
        return self._multi_lut['--config']

    @property
    def common_options(self):
        # https://gdal.org/programs/gdalwarp.html#cmdoption-gdalwarp-co
        return self._multi_lut['-co']

    @property
    def options(self):
        return self._multi_lut

    def __setitem__(self, key, val):
        self._single_lut[key] = val

    def __getitem__(self, key):
        return self._single_lut[key]

    @property
    def flags(self):
        return self._flags

    def append(self, item):
        self._positional.append(item)

    def extend(self, items):
        self._positional.extend(items)

    def finalize(self):
        import pathlib
        import os
        parts = [self.command_name]
        parts.extend(self._flags)

        def _rectify_value(v):
            import shlex
            if isinstance(v, list):
                v = v
            elif isinstance(v, pathlib.Path):
                v = [os.fspath(v)]
            elif isinstance(v, str):
                v = [shlex.quote(v)]
            else:
                raise TypeError(type(v))
            return v

        for key, val in self._single_lut.items():
            val = _rectify_value(val)
            parts.extend([key])
            parts.extend(_rectify_value(val))

        # hack, config needs space separated data
        sq_sep_option_keys = ['--config']
        eq_sep_options = ub.dict_diff(self._multi_lut, sq_sep_option_keys)
        sq_sep_options = ub.dict_isect(self._multi_lut, sq_sep_option_keys)

        for superkey, lut in eq_sep_options.items():
            for key, val in lut.items():
                parts.extend([superkey, f'{key}={val}'])

        for superkey, lut in sq_sep_options.items():
            for key, val in lut.items():
                parts.extend([superkey, key])
                parts.extend(_rectify_value(val))

        for v in self._positional:
            parts.extend(_rectify_value(v))

        command = ub.paragraph(' '.join(parts))
        return command

    def set_cog_options(self, compress='DEFLATE', blocksize=256,
                        overviews='AUTO'):
        if compress == 'RAW':
            compress = 'NONE'
        self['-of'] = 'COG'
        self.options['-co']['OVERVIEWS'] = str(overviews)
        self.options['-co']['BLOCKSIZE'] = str(blocksize)
        self.options['-co']['COMPRESS'] = compress


def gdal_single_translate(in_fpath, out_fpath, pixel_box=None, blocksize=256,
                          compress='DEFLATE', tries=1, verbose=0):
    """
    Crops geotiffs using pixels

    Args:
        in_fpath (PathLike): geotiff to translate

        out_fpath (PathLike): output geotiff

        pixel_box (kwimage.Boxes): box to crop to in pixel space.

        blocksize (int): COG tile size

        compress (str): gdal compression

        verbose (int): verbosity level

    CommandLine:
        xdoctest -m watch.utils.util_gdal gdal_single_translate

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
        >>> crs84_out_fpath = ub.augpath(in_fpath, prefix='_crs84_crop')
        >>> gdal_single_warp(in_fpath, crs84_out_fpath, local_epsg=crs84_epsg, space_box=crs84_space_box)

        >>> # Test UTM cropping
        >>> utm_poly = kwimage.Polygon(exterior=info['utm_corners'])
        >>> utm_epsg = int(info['utm_crs_info']['auth'][1])
        >>> utm_space_box = utm_poly.scale(0.5, about='center').to_boxes()
        >>> utm_out_fpath = ub.augpath(in_fpath, prefix='_utmcrop')
        >>> gdal_single_warp(in_fpath, utm_out_fpath, local_epsg=utm_epsg, space_box=utm_space_box, box_epsg=utm_epsg)

        >>> # Test Pixel cropping
        >>> pxl_poly = kwimage.Polygon(exterior=info['pxl_corners'])
        >>> pixel_box = pxl_poly.scale(0.5, about='center').to_boxes()
        >>> pxl_out_fpath = ub.augpath(in_fpath, prefix='_pxlcrop')
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

    Ignore:
        print(ub.cmd('gdalinfo ' + str(in_fpath))['out'])
        print(ub.cmd('gdalinfo ' + str(crs84_out_fpath))['out'])
        print(ub.cmd('gdalinfo ' + str(utm_out_fpath))['out'])
        print(ub.cmd('gdalinfo ' + str(pxl_out_fpath))['out'])
    """
    tmp_fpath = ub.Path(ub.augpath(out_fpath, prefix='.tmptrans.'))

    builder = GDalCommandBuilder('gdal_translate')
    # builder['--debug'] = 'off'
    builder.set_cog_options(compress=compress, blocksize=blocksize)
    # Use the new COG output driver
    # Perf options
    if 1:
        builder.options['--config']['GDAL_CACHEMAX'] = '15%'
        builder.options['-co']['NUM_THREADS'] = 'ALL_CPUS'

    if pixel_box is not None:
        xoff, yoff, xsize, ysize = pixel_box.to_xywh().data[0]
        builder['-srcwin'] = [f'{xoff}', f'{yoff}', f'{xsize}', f'{ysize}']

    builder.append(f'{in_fpath}')
    builder.append(f'{tmp_fpath}')
    gdal_translate_command = builder.finalize()

    _execute_gdal_command_with_checks(
        gdal_translate_command, tmp_fpath, tries=tries, verbose=verbose)
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
                     as_vrt=False,
                     use_te_geoidgrid=False,
                     dem_fpath=None,
                     error_logfile=None,
                     tries=1,
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

        as_vrt (bool): undocumented

        use_te_geoidgrid (bool): undocumented

        dem_fpath (bool): undocumented

        error_logfile (None | PathLike):
            If specified, errors will be logged to this filepath.

        tries (int): gdal can be flakey, set to force some number of retries

    Notes:
        In gdalwarp:
            -s_srs - Set source spatial reference
            -t_srs - Set target spatial reference

            -te_srs - Specifies the SRS in which to interpret the coordinates given with -te.
            -te - Set georeferenced extents of output file to be created

    Ignore:
        import xdev
        import sys, ubelt
        from watch.utils.util_gdal import *  # NOQA
        globals().update(xdev.get_func_kwargs(gdal_single_warp))

    Example:
        >>> import kwimage
        >>> from watch.utils.util_gdal import gdal_single_warp
        >>> in_fpath = '/vsicurl/https://sentinel-cogs.s3.us-west-2.amazonaws.com/sentinel-s2-l2a-cogs/23/K/PQ/2019/6/S2B_23KPQ_20190623_0_L2A/B03.tif'
        >>> from osgeo import gdal
        >>> info = gdal.Info(in_fpath, format='json')
        >>> bound_poly = kwimage.Polygon.coerce(info['wgs84Extent'])
        >>> crop_poly = bound_poly.scale(0.02, about='centroid')
        >>> space_box = crop_poly.to_boxes()
        >>> out_fpath = ub.Path.appdir('fds').ensuredir() / 'cropped.tif'
        >>> error_logfile = '/dev/null'
        >>> gdal_single_warp(in_fpath, out_fpath, space_box=space_box, error_logfile=error_logfile, verbose=3)
        >>> # xdoctest: +REQUIRES(--show)
        >>> import kwplot
        >>> kwplot.autompl()
        >>> data = kwimage.imread(out_fpath)
        >>> canvas = kwimage.normalize_intensity(data)
        >>> kwplot.imshow(canvas)

    Ignore:
        from kwcoco.util import util_archive
        sample_zip_fpath = ub.grabdata('https://maxar-marketing.s3.amazonaws.com/product-samples/Rome_Colosseum_2022-03-22_WV03_HD.zip')
        util_archive.Archive.extractall(sample_zip_fpath)


    Ignore:
        import os
        import kwimage
        os.environ['AWS_DEFAULT_PROFILE'] = 'iarpa'
        in_fpath = '/vsis3/smart-data-accenture/ta-1/ta1-wv-acc/14/T/QL/2014/9/29/14SEP29174805-M1BS-014484503010_01_P003_ACC/14SEP29174805-M1BS-014484503010_01_P003_ACC_B05.tif'

        gdal_multi_warp([in_fpath], out_fpath)
        --config CPL_LOG image.log
    """
    tmp_out_fpath = ub.augpath(out_fpath, prefix='.tmpwarp.')

    # Coordinate Reference System of the "target" destination image
    # t_srs = target spatial reference for output image
    if local_epsg is None:
        target_srs = 'epsg:4326'
    else:
        target_srs = 'epsg:{}'.format(local_epsg)

    builder = GDalCommandBuilder('gdalwarp')
    builder.flags.append('-overwrite')
    builder['--debug'] = 'off'
    builder['-t_srs'] = f'{target_srs}'

    # https://gdal.org/api/gdalwarp_cpp.html#_CPPv4N15GDALWarpOptions16papszWarpOptionsE
    if as_vrt:
        builder['-of'] = 'VRT'
    else:
        builder.set_cog_options(compress=compress, blocksize=blocksize,
                                overviews='AUTO')

    if space_box is not None:
        # Data is from geo-pandas so this should be traditional order
        lonmin, latmin, lonmax, latmax = space_box.data[0]
        ymin, xmin, ymax, xmax = latmin, lonmin, latmax, lonmax

        # Coordinate Reference System of the "te" crop coordinates
        # te_srs = spatial reference of query points
        # This means space_box currently MUST be in CRS84
        # crop_coordinate_srs = 'epsg:4326'
        crop_coordinate_srs = f'epsg:{box_epsg}'

        builder['-te'] = [f'{xmin}', f'{ymin}', f'{xmax}', f'{ymax}']
        builder['-te_srs'] = crop_coordinate_srs

    if nodata is not None:
        builder['-srcnodata'] = str(nodata)
        builder['-dstnodata'] = str(nodata)

    # HACK TO FIND an appropriate DEM file
    if rpcs is not None:
        builder.flags.append('-rpc')
        builder['-et'] = '0'
        if dem_fpath is None:
            dems = rpcs.elevation
            if hasattr(dems, 'find_reference_fpath'):
                # TODO: get a better DEM path for this image if possible
                dem_fpath, dem_info = dems.find_reference_fpath(latmin, lonmin)
        if dem_fpath is not None:
            builder.options['-to']['RPC_DEM'] = dem_fpath

    if use_te_geoidgrid:
        # assumes source CRS is WGS84
        # https://smartgitlab.com/TE/annotations/-/wikis/WorldView-Annotations#notes-on-the-egm96-geoidgrid-file
        from watch.rc import geoidgrid_path
        geoidgrids = geoidgrid_path()
        builder['-s_srs'] = f'"+proj=longlat +datum=WGS84 +no_defs +geoidgrids={geoidgrids}"'

    # use multithreaded warping implementation
    builder.flags.append('-multi')

    if error_logfile is not None:
        builder.options['--config']['CPL_LOG'] = error_logfile

    # if use_perf_opts:
    #     warp_memory = '15%'
    #     config_options['GDAL_CACHEMAX'] = '15%'
    #     common_options['NUM_THREADS'] = 'ALL_CPUS'
    #     warp_options['NUM_THREADS'] = '1'
    #     builder.options['-wo']['NUM_THREADS'] = '1'
    # else:
    # GDAL_CACHEMAX is in megabytes
    # https://gdal.org/programs/gdalwarp.html#cmdoption-gdalwarp-wm
    warp_memory = '1500'
    builder['-wm'] = str(warp_memory)
    builder.options['-co']['NUM_THREADS'] = '2'
    builder.options['--config']['GDAL_CACHEMAX'] = '1500'

    builder.append(in_fpath)
    builder.append(tmp_out_fpath)

    gdal_warp_command = builder.finalize()

    # Execute the command with checks to ensure gdal doesnt fail
    _execute_gdal_command_with_checks(
        gdal_warp_command, tmp_out_fpath, tries=tries, verbose=verbose)
    os.rename(tmp_out_fpath, out_fpath)


def gdal_multi_warp(in_fpaths, out_fpath, nodata=None, tries=1, blocksize=256,
                    compress='DEFLATE', error_logfile=None,
                    _intermediate_vrt=False, verbose=0, **kwargs):
    """
    See gdal_single_warp() for args

    Example:
        >>> # xdoctest +REQUIRES(--slow)
        >>> # Uses data from the data cube with extra=1
        >>> from watch.utils.util_gdal import *  # NOQA
        >>> from watch.cli.coco_align_geotiffs import *  # NOQA
        >>> import ubelt as ub
        >>> cube, region_df = SimpleDataCube.demo(with_region=True, extra=True)
        >>> local_epsg = 32635
        >>> space_box = kwimage.Polygon.from_shapely(region_df.geometry.iloc[1]).bounding_box().to_ltrb()
        >>> dpath = ub.ensure_app_cache_dir('watch/test/gdal_multi_warp')
        >>> out_fpath = ub.Path(dpath) / 'test_multi_warp.tif'
        >>> out_fpath.delete()
        >>> in_fpath1 = cube.coco_dset.get_image_fpath(2)
        >>> in_fpath2 = cube.coco_dset.get_image_fpath(3)
        >>> in_fpaths = [in_fpath1, in_fpath2]
        >>> rpcs = None
        >>> gdal_multi_warp(in_fpaths, out_fpath=out_fpath, space_box=space_box,
        >>>                 local_epsg=local_epsg, rpcs=rpcs, verbose=3)

    Ignore:
        >>> # xdoctest +REQUIRES(--slow)
        >>> import kwimage
        >>> from watch.utils.util_gdal import gdal_single_warp
        >>> in_fpath = '/vsicurl/https://sentinel-cogs.s3.us-west-2.amazonaws.com/sentinel-s2-l2a-cogs/23/K/PQ/2019/6/S2B_23KPQ_20190623_0_L2A/B03.tif'
        >>> from osgeo import gdal
        >>> info = gdal.Info(in_fpath, format='json')
        >>> bound_poly = kwimage.Polygon.coerce(info['wgs84Extent'])
        >>> crop_poly = bound_poly.scale(0.2, about='centroid')
        >>> space_box = crop_poly.to_boxes()
        >>> out_fpath = ub.Path.appdir('fds').ensuredir() / 'gdal_multi_warp_demo_out.tif'
        >>> error_logfile = '/dev/null'
        >>> in_fpaths = [in_fpath]
        >>> out_fpath1 = ub.Path.appdir('fds').ensuredir() / 'gdal_multi_warp_demo_out1.tif'
        >>> out_fpath2 = ub.Path.appdir('fds').ensuredir() / 'gdal_multi_warp_demo_out2.tif'
        >>> with ub.Timer('no_vrt'):
        >>>     gdal_multi_warp(in_fpaths, out_fpath2, space_box=space_box, verbose=3, _intermediate_vrt=False)
        >>> with ub.Timer('with_vrt'):
        >>>     gdal_multi_warp(in_fpaths, out_fpath1, space_box=space_box, verbose=3, _intermediate_vrt=True)
        >>> # xdoctest: +REQUIRES(--show)
        >>> import kwplot
        >>> kwplot.autompl()
        >>> data = kwimage.imread(out_fpath)
        >>> canvas = kwimage.normalize_intensity(data)
        >>> kwplot.imshow(canvas)

    Ignore:
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
    """
    # Warp then merge
    # Write to a temporary file and then rename the file to the final
    # Destination so ctrl+c doesn't break everything
    tmp_out_fpath = ub.augpath(out_fpath, prefix='.tmpmerge.')

    single_warp_kwargs = kwargs.copy()
    single_warp_kwargs['compress'] = compress
    single_warp_kwargs['blocksize'] = blocksize
    single_warp_kwargs['tries'] = tries
    single_warp_kwargs['nodata'] = nodata
    single_warp_kwargs['verbose'] = verbose
    single_warp_kwargs['error_logfile'] = error_logfile
    # Delay the actual execution of the partial warps until merge is called.

    if _intermediate_vrt:
        # Not using the intermediate VRT is faster.
        # Building the VRT requires network calls, so might as well
        # pull down all of the data and cache it on disk
        single_warp_kwargs['as_vrt'] = True

    USE_REAL_TEMPFILES = 1
    if USE_REAL_TEMPFILES:
        import tempfile
        tempfiles = []  # hold references

    warped_gpaths = []
    for in_idx, in_fpath in enumerate(in_fpaths):
        if USE_REAL_TEMPFILES:
            tmpfile = tempfile.NamedTemporaryFile(suffix='.tif')
            tempfiles.append(tmpfile)
            part_out_fpath = tmpfile.name
        else:
            part_out_fpath = ub.augpath(
                out_fpath, prefix=f'.tmpmerge.part{in_idx:02d}.')
        gdal_single_warp(in_fpath, part_out_fpath, **single_warp_kwargs)
        warped_gpaths.append(part_out_fpath)

    if nodata is not None:
        # FIXME: might not be necessary
        from watch.utils import util_raster
        valid_polygons = []
        for part_out_fpath in warped_gpaths:
            sh_poly = util_raster.mask(part_out_fpath,
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

    builder = GDalCommandBuilder('gdal_merge.py')
    builder.options['-co']['NUM_THREADS'] = '2'
    builder.options['--config']['GDAL_CACHEMAX'] = '1500'
    # builder.set_cog_options()
    if error_logfile is not None:
        builder.options['--config']['CPL_LOG'] = error_logfile
    if nodata is not None:
        builder['-n'] = str(nodata)
    builder['-o'] = tmp_out_fpath
    builder.extend(warped_gpaths)

    gdal_merge_cmd = builder.finalize()
    _execute_gdal_command_with_checks(
        gdal_merge_cmd, tmp_out_fpath, tries=tries, verbose=verbose)

    # Merge does not output cogs, we need to do another call to make that
    # happen
    tmp_out_fpath2 = ub.augpath(out_fpath, prefix='.tmpcog.')
    gdal_single_translate(tmp_out_fpath, tmp_out_fpath2, compress=compress,
                          blocksize=blocksize, verbose=verbose)
    ub.Path(tmp_out_fpath).delete()

    os.rename(tmp_out_fpath2, out_fpath)


def _execute_gdal_command_with_checks(command, out_fpath, tries=1, shell=False,
                                      check_after=True, verbose=0):
    """
    Given a gdal command and the expected file that it should produce
    try to execute a few times and check that it produced a valid result.
    """

    def _execute_command():
        cmd_info = ub.cmd(command, check=True, verbose=verbose, shell=shell)
        if not ub.Path(out_fpath).exists():
            raise FileNotFoundError(f'Error: gdal did not write {out_fpath}')
        if check_after:
            try:
                GdalOpen(out_fpath, mode='r')
            except RuntimeError:
                raise
        return cmd_info
    got = -1
    try:
        logger = DummyLogger()
        got = retry.api.retry_call(
            _execute_command,
            tries=tries, delay=1, exceptions=(
                subprocess.CalledProcessError, FileNotFoundError,
                RuntimeError),
            logger=logger)
    except subprocess.CalledProcessError as ex:
        if verbose:
            print('\n\nCOMMAND FAILED: {!r}'.format(ex.cmd))
            print(ex.stdout)
            print(ex.stderr)
        raise
    except FileNotFoundError:
        if verbose:
            print(
                'Error: gdal seems to have returned with a valid exist code, '
                'but the target file was not written')
            print('got = {}'.format(ub.repr2(got, nl=1)))
            print(command)
        raise
    except RuntimeError:
        if verbose:
            print(
                'Error: gdal has written a file, but its contents '
                'appear to be invalid')
            print('got = {}'.format(ub.repr2(got, nl=1)))
            print(command)
        raise


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


class GdalSupressWarnings(object):
    """
    References:
        https://gdal.org/api/python_gotchas.html

    This currently will suppress warnings entirely. We could build this into
    something a big nicer.
    """
    def __init__(self):
        from osgeo import gdal
        self.err_level = gdal.CE_None
        self.err_no = 0
        self.err_msg = ''

    def handler(self, err_level, err_no, err_msg):
        self.err_level = err_level
        self.err_no = err_no
        self.err_msg = err_msg

    def __enter__(self):
        from osgeo import gdal
        gdal.PushErrorHandler(self.handler)
        gdal.UseExceptions()  # Exceptions will get raised on anything >= gdal.CE_Failure

    def __exit__(self, a, b, c):
        from osgeo import gdal
        gdal.PopErrorHandler()
        pass


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

    Ignore:
        gpath = '/vsis3/smart-data-accenture/ta-1/ta1-wv-acc/14/T/QL/2014/9/29/14SEP29174805-M1BS-014484503010_01_P003_ACC/14SEP29174805-M1BS-014484503010_01_P003_ACC_B05.tif'

        from watch.utils import util_gdal
        import sys, ubelt
        from watch.utils.util_gdal import *  # NOQA
        os.environ['AWS_DEFAULT_PROFILE'] = 'iarpa'
        gpath = '/vsis3/smart-data-accenture/ta-1/ta1-s2-acc/23/K/PQ/2017/9/6/S2A_23KPQ_20170906_0_L1C_ACC/S2A_23KPQ_20170906_0_L1C_ACC_B10.tif'
        ref = util_gdal.GdalDataset.open(gpath, 'r', virtual_retries=3)
        with GdalErrorHandler() as handler:
            ref.GetMetadataDomainList()

        from watch.gis.geotiff import *  # NOQA
        _ = geotiff_metadata(gpath)


        # Test 404 handling
        from watch.utils import util_gdal
        gpath = '/vsicurl/https://sentinel-cogs.s3.us-west-2.amazonaws.com/sentinel-s2-l2a-cogs/51R/TP2020/8/S2A_51RTP_20200811_0_L2A/B11.tif'
        import watch
        meta = watch.gis.geotiff.geotiff_metadata(gpath)
        from watch.utils import util_gdal
        infos = {}
        try:
            util_gdal.GdalDataset.open(gpath, 'r', virtual_retries=3)
        except Exception as e:
            ex = e
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
                raise RuntimeError(msg + f' for {_path}')
        except Exception:
            import time
            if _path.startswith(GDAL_VIRTUAL_FILESYSTEM_PREFIX):
                wait_time = 0.1
                for _ in range(virtual_retries):
                    try:
                        __ref = gdal.Open(_path, mode)
                        if __ref is None:
                            msg = gdal.GetLastErrorMsg()
                            raise RuntimeError(msg + f' for {_path}')
                    except Exception:
                        time.sleep(wait_time)
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
