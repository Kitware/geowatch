from tempfile import NamedTemporaryFile, TemporaryDirectory
import os
import shlex

from watch.utils import util_raster


def _save_cloudmask(scenedir, out_fpath, driver, sensor, args):

    with TemporaryDirectory() as d:

        args = args + ['-o', out_fpath, '-e', d]
        if sensor == 'S2':
            if os.path.isdir(os.path.join(scenedir, 'IMG_DATA')):
                args += ['--granuledir', scenedir]
            else:
                args += ['--safedir', scenedir]

            # default is 20m GSD for S2 (changeable)
            # and 30m GSD for LS (fixed).
            # assume that unless someone knows what they're doing, they want 30m for both
            if '--pixsize' not in args:
                args += ['--pixsize', '20']

            # os.system instead of subprocess.run to inherit the current conda env
            os.system(f'RIOS_DFLT_DRIVER={driver} fmask_sentinel2Stacked.py ' +
                      shlex.join(args))

        elif sensor == 'LS':
            args += ['--scenedir', scenedir]

            os.system(
                f'RIOS_DFLT_DRIVER={driver} fmask_usgsLandsatStacked.py ' +
                shlex.join(args))


def cloudmask(in_dpath,
              out_fpath=None,
              driver=None,
              sensor='S2',
              args=[]):
    '''
    Use Fmask 3 to generate a cloud mask for a S2 or LS scene.

    Args:
        in_dpath: path to the root directory of a S2 or LS scene
        out_fpath: path to save the cloud mask to.
            Will also create out_fpath+'.aux.xml', a histogram
            if None, read and return the array in memory
        driver: gdal raster driver shortname to use when creating the image
            If None, will guess from out_fpath file extension. (Default: GTiff)
            For available drivers, see [1] for your $ gdalinfo --version
            or run $ gdalinfo --formats
        sensor: 'S2' or 'LS'
        thresh: TODO cloud probability threshold.
            If None, return the proba map instead of a mask
        args: other arguments for fmask in the form ['--arg1', 'val1', '--arg2', 'val2']
            Some relevant arguments are:
            --mincloudsize MINCLOUDSIZE
                Mininum cloud size (in pixels) to retain, before any buffering. Default=0)
            --cloudbufferdistance CLOUDBUFFERDISTANCE
                Distance (in metres) to buffer final cloud objects (default=150)
            --shadowbufferdistance SHADOWBUFFERDISTANCE
                Distance (in metres) to buffer final cloud shadow objects (default=300)
            --cloudprobthreshold CLOUDPROBTHRESHOLD
                Cloud probability threshold (percentage) (default=20.0). This is the constant
                term at the end of equation 17, given in the paper as 0.2 (i.e. 20%). To reduce
                commission errors, increase this value, but this will also increase omission
                errors.
            --nirsnowthreshold NIRSNOWTHRESHOLD
                Threshold for NIR reflectance (range [0-1]) for snow detection (default=0.11).
                Increase this to reduce snow commission errors
            --greensnowthreshold GREENSNOWTHRESHOLD
                Threshold for Green reflectance (range [0-1]) for snow detection (default=0.1).
                Increase this to reduce snow commission errors
            To see all:
                $ fmask_sentinel2Stacked.py -h
                $ fmask_usgsLandsatStacked.py -h
            (they have different args)

    Returns:
        out_fpath or cloud mask array
        The cloud mask is a uint8, 30m GSD raster with the following values [2]:
            0: null
            1: clear
            2: cloud
            3: shadow
            4: snow
            5: water
            6-7: unused

    References:
        [1] https://gdal.org/drivers/raster/index.html
        [2] https://github.com/ubarsc/python-fmask/blob/master/fmask/fmask.py#L82

    Example:
        >>> # xdoctest: +REQUIRES(--network)
        >>> import os
        >>> import numpy as np
        >>> from watch.datacube.cloud.fmask3 import *
        >>> from watch.utils.util_raster import GdalOpen
        >>> from watch.demo.landsat_demodata import grab_landsat_product
        >>>
        >>> root = os.path.commonpath(grab_landsat_product()['bands'])
        >>>
        >>> data = cloudmask(root, sensor='LS')
        >>> assert data.dtype == np.uint8
        >>> assert set(np.unique(data)).issubset({0, 1, 2, 3, 4, 5})
        >>>
        >>> out_path = cloudmask(root, 'cloudmask.tif', sensor='LS')
        >>> assert out_path == 'cloudmask.tif'
        >>> with GdalOpen(out_path) as f:
        >>>     assert f.GetDriver().ShortName == 'GTiff'
        >>>     assert np.all(f.ReadAsArray() == data)
        >>>
        >>> # clean up
        >>> os.remove('cloudmask.tif')
        >>> os.remove('cloudmask.tif.aux.xml')
    '''

    if driver is None:
        ext = ('' if out_fpath is None else os.path.splitext(out_fpath)[1])
        for d, _, exts in util_raster.list_gdal_drivers():
            if ext in exts:
                driver = d
                break
        # override gdal default driver 'ENVI'
        if driver is None or ext == '':
            driver = 'GTiff'

    if out_fpath is None:
        with NamedTemporaryFile(mode='r+') as f:
            _save_cloudmask(in_dpath, f.name, driver, sensor, args)
            with util_raster.GdalOpen(f.name) as result:
                return result.ReadAsArray()

    else:
        _save_cloudmask(in_dpath, out_fpath, driver, sensor, args)
        return out_fpath
