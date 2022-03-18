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
import numpy as np
import kwimage
import os
import ubelt as ub
import shapely
from osgeo import gdal, osr
from watch.gis.spatial_reference import utm_epsg_from_latlon
from copy import deepcopy
from lxml import etree
from tempfile import NamedTemporaryFile


GDAL_VIRTUAL_FILESYSTEM_PREFIX = 'vsi'

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


def gdal_multi_warp(in_fpaths, out_fpath, *args, nodata=None, **kwargs):
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
        gdal_single_warp(in_fpath, tmp_out, *args, nodata=nodata, **kwargs)
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
    cmd_info = ub.cmd(merge_cmd_parts, check=True)
    if cmd_info['ret'] != 0:
        print('\n\nCOMMAND FAILED: {!r}'.format(merge_cmd))
        print(cmd_info['out'])
        print(cmd_info['err'])
        raise Exception(cmd_info['err'])
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


def gdal_single_warp(in_fpath,
                     out_fpath,
                     space_box=None,
                     local_epsg=4326,
                     nodata=None,
                     rpcs=None,
                     blocksize=256,
                     compress='DEFLATE',
                     use_perf_opts=False,
                     as_vrt=False,
                     use_te_geoidgrid=False,
                     dem_fpath=None):
    r"""
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
            -of VRT'
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
        crop_coordinate_srs = 'epsg:4326'

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
    cmd_info = ub.cmd(command, verbose=0)  # NOQA
    if cmd_info['ret'] != 0:
        print('\n\nCOMMAND FAILED: {!r}'.format(command))
        print(cmd_info['out'])
        print(cmd_info['err'])
        raise Exception(cmd_info['err'])


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
    """
    return GdalDataset(path, mode=mode, **kwargs)


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
            (i.e. starts with vis) then we try to open it this many times
            before we finally fail.

    Example:
        >>> # xdoctest: +REQUIRES(--network)
        >>> from watch.utils.util_gdal import *
        >>> from watch.demo.landsat_demodata import grab_landsat_product
        >>> path = grab_landsat_product()['bands'][0]
        >>> #
        >>> # standard use:
        >>> dataset = gdal.Open(path)
        >>> print(dataset.GetDescription())  # do stuff
        >>> del dataset  # or 'dataset = None'
        >>> #
        >>> # equivalent:
        >>> with GdalDataset(path) as dataset:
        >>>     print(dataset.GetDescription())  # do stuff
        >>> #
        >>> # open for writing:
        >>> with GdalDataset(path, gdal.GA_Update) as dataset:
        >>>     print(dataset.GetDescription())  # do stuff

    Example:
        >>> # Demonstrate use cases of this object
        >>> from watch.utils.util_gdal import *
        >>> import kwimage
        >>> # Grab demo path we can test with
        >>> path = kwimage.grab_test_image_fpath()
        >>> #
        >>> #
        >>> # Method1: Use GDalOpen exactly the same as gdal.Open
        >>> ref = GdalDataset(path)
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
        >>> with GdalDataset(path, mode='r') as ref:
        >>>     ref.GetDescription()  # do stuff
        >>>     print(f'{ref=!s}')
        >>>     assert not ref.closed
        >>> print(f'{ref=!s}')
        >>> assert ref.closed
    """

    def __init__(self, path, mode='r', virtual_retries=3):
        if isinstance(mode, str):
            # https://mkyong.com/python/python-difference-between-r-w-and-a-in-open/
            if mode in {'readonly', 'r'}:
                mode = gdal.GA_ReadOnly
            elif mode == {'update', 'w+'}:
                mode = gdal.GA_Update
            else:
                raise KeyError(mode)
        if mode == gdal.GA_ReadOnly:
            str_mode = 'r'
        elif mode == gdal.GA_Update:
            str_mode = 'w+'
        else:
            raise ValueError(mode)
        self.__ref = None  # This is a private variable
        self._path = _path = os.fspath(path)
        self._str_mode = str_mode

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
            if virtual_retries and _path.startswith('/vsi'):
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

        self.__ref = __ref

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


def reproject_crop(in_path, aoi, code=None, out_path=None, vrt_root=None):
    """
    Crop an image to an AOI and reproject to its UTM CRS (or another CRS)

    Unfortunately, this cannot be done in a single step in scenes_to_vrt
    because gdal.BuildVRT does not support warping between CRS.
    Cropping alone could be done in scenes_to_vrt. Note gdal.BuildVRTOptions
    has an outputBounds(=-te) kwarg for cropping, but not an equivalent of
    -te_srs.

    This means another intermediate file is necessary for each warp operation.

    TODO check for this quantization error:
        https://gis.stackexchange.com/q/139906

    Args:
        in_path: A georeferenced image. GTiff, VRT, etc.
        aoi: A geojson Feature in epsg:4326 CRS to crop to.
        code: EPSG code [1] of the CRS to convert to.
            if None, use the UTM CRS containing aoi.
        out_path: Name of output file to write to. If None, create a VRT file.
        vrt_root: Root directory for VRT output. If None, same dir as input.

    Returns:
        Path to a new VRT or out_path

    References:
        [1] http://epsg.io/

    Example:
        >>> # xdoctest: +REQUIRES(--network)
        >>> import os
        >>> from osgeo import gdal
        >>> from watch.utils.util_gdal import *
        >>> from watch.demo.landsat_demodata import grab_landsat_product
        >>> band1 = grab_landsat_product()['bands'][0]
        >>> #
        >>> # pick the AOI from the drop0 KR site
        >>> # (this doesn't actually intersect the demodata)
        >>> top, left = (128.6643, 37.6601)
        >>> bottom, right = (128.6749, 37.6639)
        >>> geojson_bbox = {
        >>>     "type":
        >>>     "Polygon",
        >>>     "coordinates": [[[top, left], [top, right], [bottom, right],
        >>>                      [bottom, left], [top, left]]]
        >>> }
        >>> #
        >>> out_path = reproject_crop(band1, geojson_bbox)
        >>> #
        >>> # clean up
        >>> os.remove(out_path)
    """
    if out_path is None:
        root, name = os.path.split(in_path)
        if vrt_root is None:
            vrt_root = root
        os.makedirs(vrt_root, exist_ok=True)
        out_path = os.path.join(vrt_root, f'{hash(name + "warp")}.vrt')
        if os.path.isfile(out_path):
            print(f'Warning: {out_path} already exists! Removing...')
            os.remove(out_path)

    if code is None:
        # find the UTM zone(s) of the AOI
        codes = [
            utm_epsg_from_latlon(lat, lon)
            for lon, lat in aoi['coordinates'][0]
        ]
        u, counts = np.unique(codes, return_counts=True)
        if len(u) > 1:
            print(
                f'Warning: AOI crosses UTM zones {u}. Taking majority vote...')
        code = int(u[np.argsort(-counts)][0])

    dst_crs = osr.SpatialReference()
    dst_crs.ImportFromEPSG(code)

    bounds_crs = osr.SpatialReference()
    bounds_crs.ImportFromEPSG(4326)

    opts = gdal.WarpOptions(
        outputBounds=shapely.geometry.shape(aoi).buffer(0).bounds,
        outputBoundsSRS=bounds_crs,
        dstSRS=dst_crs)
    vrt = gdal.Warp(out_path, in_path, options=opts)
    del vrt

    return out_path


def scenes_to_vrt(scenes, vrt_root, relative_to_path):
    """
    Search for band files from compatible scenes and stack them in a single
    mosaicked VRT

    A simple wrapper around watch.utils.util_gdal.make_vrt that performs both
    the 'stacked' and 'mosaicked' modes

    Args:
        scenes: list(scene), where scene := list(path) [of band files]
        vrt_root: root dir to save VRT under

    Returns:
        path to the VRT

    Example:
        >>> # xdoctest: +REQUIRES(--network)
        >>> import os
        >>> from osgeo import gdal
        >>> from watch.utils.util_gdal import *
        >>> from watch.demo.landsat_demodata import grab_landsat_product
        >>> bands = grab_landsat_product()['bands']
        >>> #
        >>> # pretend there are more scenes here
        >>> out_path = scenes_to_vrt([sorted(bands)] , vrt_root='.', relative_to_path=os.getcwd())
        >>> with GdalOpen(out_path) as f:
        >>>     print(f.GetDescription())
        >>> #
        >>> # clean up
        >>> os.remove(out_path)
    """
    # first make VRTs for individual tiles
    # TODO use https://rasterio.readthedocs.io/en/latest/topics/memory-files.html
    # for these intermediate files?
    tmp_vrts = [
        make_vrt(scene,
                 os.path.join(vrt_root, f'{hash(scene[0])}.vrt'),
                 mode='stacked',
                 relative_to_path=relative_to_path) for scene in scenes
    ]

    # then mosaic them
    final_vrt = make_vrt(tmp_vrts,
                         os.path.join(vrt_root,
                                      f'{hash(scenes[0][0] + "final")}.vrt'),
                         mode='mosaicked',
                         relative_to_path=relative_to_path)

    if 0:  # can't do this because final_vrt still references them
        for t in tmp_vrts:
            os.remove(t)

    return final_vrt


def make_vrt(in_paths, out_path, mode, relative_to_path=None, **kwargs):
    """
    Stack multiple band files in the same directory into a single VRT

    Args:
        in_paths: list(path)
        out_path: path to save to; standard is '*.vrt'. If None, a path will be
            generated.
        mode:
            'stacked': Stack multiple band files covering the same area
            'mosaicked': Mosaic/merge scenes with overlapping areas. Content
                will be taken from the first in_path without nodata.
        relative_to_path: if this function is being called from another
            process, pass in the cwd of the calling process, to trick gdal into
            creating a rerootable VRT
        kwargs: passed to gdal.BuildVRTOptions [1,2]

    Returns:
        path to VRT

    Example:
        >>> # xdoctest: +REQUIRES(--network)
        >>> import os
        >>> from osgeo import gdal
        >>> from watch.utils.util_gdal import *
        >>> from watch.demo.landsat_demodata import grab_landsat_product
        >>> bands = grab_landsat_product()['bands']
        >>> #
        >>> # stack bands from a scene
        >>> make_vrt(sorted(bands), './bands1.vrt', mode='stacked', relative_to_path=os.getcwd())
        >>> # pretend this is a different scene
        >>> make_vrt(sorted(bands), './bands2.vrt', mode='stacked', relative_to_path=os.getcwd())
        >>> # now, if they overlap, mosaic/merge them
        >>> make_vrt(['./bands1.vrt', './bands2.vrt'], 'full_scene.vrt', mode='mosaicked', relative_to_path=os.getcwd())
        >>> with GdalOpen('full_scene.vrt') as f:
        >>>     print(f.GetDescription())
        >>> #
        >>> # clean up
        >>> os.remove('bands1.vrt')
        >>> os.remove('bands2.vrt')
        >>> os.remove('full_scene.vrt')

    References:
        [1] https://gdal.org/programs/gdalbuildvrt.html
        [2] https://gdal.org/python/osgeo.gdal-module.html#BuildVRTOptions
    """

    if mode == 'stacked':
        kwargs['separate'] = True
    elif mode == 'mosaicked':
        kwargs['separate'] = False
        kwargs['srcNodata'] = 0  # this ensures nodata doesn't overwrite data
    else:
        raise ValueError(f'mode: {mode} should be "stacked" or "mosaicked"')

    # set sensible defaults
    if 'resolution' not in kwargs:
        kwargs['resolution'] = 'highest'
    if 'resampleAlg' not in kwargs:
        kwargs['resampleAlg'] = 'bilinear'

    opts = gdal.BuildVRTOptions(**kwargs)

    if len(in_paths) > 1:
        common = os.path.commonpath(in_paths)
    else:
        common = os.path.dirname(in_paths[0])

    if relative_to_path is None:
        relative_to_path = os.path.dirname(os.path.abspath(__file__))

    # validate out_path
    if out_path is not None:
        out_path = os.path.abspath(out_path)
        if os.path.splitext(out_path)[1]:  # is a file
            os.makedirs(os.path.dirname(out_path), exist_ok=True)
        elif os.path.isdir(out_path):  # is a dir
            raise ValueError(f'{out_path} is an existing directory.')

    # generate an unused name
    with NamedTemporaryFile(dir=common,
                            suffix='.vrt',
                            mode='r+',
                            delete=(out_path is not None)) as f:

        # First, create VRT in a place where it can definitely see the input
        # files.  Use a relative instead of absolute path to ensure that
        # <SourceFilename> refs are relative, and therefore the VRT is
        # rerootable
        vrt = gdal.BuildVRT(os.path.relpath(f.name, start=relative_to_path),
                            in_paths,
                            options=opts)
        del vrt  # write to disk

        # then, move it to the desired location
        if out_path is None:
            out_path = f.name
        elif os.path.isfile(out_path):
            print(f'warning: {out_path} already exists! Removing...')
            os.remove(out_path)
        reroot_vrt(f.name, out_path, keep_old=True)

    return out_path


def reroot_vrt(old_path, new_path, keep_old=True):
    """
    Copy a VRT file, fixing relative paths to its component images

    Example:
        >>> # xdoctest: +REQUIRES(--network)
        >>> import os
        >>> from osgeo import gdal
        >>> from watch.utils.util_gdal import *
        >>> from watch.demo.landsat_demodata import grab_landsat_product
        >>> bands = grab_landsat_product()['bands']
        >>> #
        >>> # VRT must be created in the imgs' subtree
        >>> tmp_path = os.path.join(os.path.dirname(bands[0]), 'all_bands.vrt')
        >>> # (consider using the wrapper util_gdal.make_vrt instead of this)
        >>> gdal.BuildVRT(tmp_path, sorted(bands))
        >>> # now move it somewhere more convenient
        >>> reroot_vrt(tmp_path, './bands.vrt', keep_old=False)
        >>> #
        >>> # clean up
        >>> os.remove('bands.vrt')
    """
    if os.path.abspath(old_path) == os.path.abspath(new_path):
        return

    path_diff = os.path.relpath(os.path.dirname(os.path.abspath(old_path)),
                                start=os.path.dirname(
                                    os.path.abspath(new_path)))

    tree = deepcopy(etree.parse(old_path))
    for elem in tree.iterfind('.//SourceFilename'):
        if elem.get('relativeToVRT') == '1':
            elem.text = os.path.join(path_diff, elem.text)
        else:
            if not os.path.isabs(elem.text):
                raise ValueError(f'''VRT file:
                    {old_path}
                cannot be rerooted because it contains path:
                    {elem.text}
                relative to an unknown location [the original calling location].
                To produce a rerootable VRT, call gdal.BuildVRT() with out_path relative to in_paths.'''
                                 )
            if not os.path.isfile(elem.text):
                raise ValueError(f'''VRT file:
                    {old_path}
                references an nonexistent path:
                    {elem.text}''')

    with open(new_path, 'wb') as f:
        tree.write(f, encoding='utf-8')

    if not keep_old:
        os.remove(old_path)
