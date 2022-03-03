"""
Try to figure out how to work with coordinate transforms
"""
import ubelt as ub
from watch.gis.geotiff import geotiff_crs_info
from tempenv import TemporaryEnvironment  # NOQA
from osgeo import osr


def demo_temp_env_error(include_fix=False):
    gpath = ub.grabdata(
        'https://download.osgeo.org/geotiff/samples/gdal_eg/cea.tif',
        appname='smart_watch/demodata', hash_prefix='10a2ebcdcd95582')

    if include_fix:
        osr.GetPROJSearchPaths()

    with TemporaryEnvironment({'PROJ_LIB': None, 'PROJ_DEBUG': '3'}):
        info = geotiff_crs_info(gpath)
        assert info['img_shape'] == (515, 514)


def demo_thread_error(include_fix=False):
    import watch  # NOQA
    # from watch.utils import util_gis
    # from shapely import geometry
    # from watch.utils import util_gis
    # from shapely.ops import unary_union
    # import shapely
    # util_gis._get_crs84()

    # There must be some condition here that breaks the threads like it does in
    # the align script... not sure what

    # from osgeo import osr
    gpaths = []
    gpath = ub.grabdata(
        'https://download.osgeo.org/geotiff/samples/gdal_eg/cea.tif',
        appname='smart_watch/demodata', hash_prefix='10a2ebcdcd95582')
    gpaths.append(gpath)

    # url = ('http://storage.googleapis.com/gcp-public-data-landsat/'
    #         'LC08/01/044/034/LC08_L1GT_044034_20130330_20170310_01_T2/'
    #         'LC08_L1GT_044034_20130330_20170310_01_T2_B11.TIF')
    # gpath = ub.grabdata(url, appname='smart_watch')
    # gpaths.append(gpath)

    jobs = ub.JobPool('thread', max_workers=3)
    for gpath in gpaths:
        jobs.submit(worker, gpath)

    for job in jobs.as_completed(desc='get jobs'):
        got = job.result()
        print('got = {!r}'.format(got))


def worker(gpath):
    from watch.utils import util_raster
    info = geotiff_crs_info(gpath)
    util_raster.mask(gpath, tolerance=10, convex_hull=True)
    return info

if __name__ == '__main__':
    """
    CommandLine:
        python ~/code/watch/dev/devcheck_coordinate_transform.py demo_temp_env_error --include_fix=True
        python ~/code/watch/dev/devcheck_coordinate_transform.py demo_temp_env_error --include_fix=False

        python ~/code/watch/dev/devcheck_coordinate_transform.py demo_thread_error --include_fix=True
        python ~/code/watch/dev/devcheck_coordinate_transform.py demo_thread_error --include_fix=False
    """
    import fire
    fire.Fire()
    # demo_temp_env_error()
    # demo_thread_error()
