"""
Grab specific sentinel2 images for testing
"""
import ubelt as ub
import os
import pathlib
from glob import glob


def grab_sentinel2_product(index=0, overwrite=False):
    """
    Download and cache all items for a Sentinel-2 product.

    TODO when RGD supports API keys, give one to this repo and use that instead of fels.

    Args:
        index: 0, 1, or 2. Currently this function just picks 3 scenes over KR in Nov 2018.
        overwrite (bool, default=False): if True, always downloads the files

    Returns:
        rgd_client.rgdc.RasterDownload(
            path: pathlib.Path,
            images: List[pathlib.Path],
            ancillary: List[pathlib.Path]
        )

    Example:
        >>> # xdoctest: +SKIP("too many https errors")
        >>> # xdoctest: +REQUIRES(--network)
        >>> from geowatch.demo.sentinel2_demodata import *  # NOQA
        >>> from geowatch.utils.util_rgdc import bands_sentinel2
        >>> product = grab_sentinel2_product()
        >>> assert len(bands_sentinel2(product)) == 13

    SeeAlso:
        geowatch.util.util_rgdc.bands_sentinel2
    """
    try:
        from rgd_imagery_client import RasterDownload
    except ImportError:
        from rgd_client.rgdc import RasterDownload

    from os.path import join, basename, relpath
    urls = [
        'http://storage.googleapis.com/gcp-public-data-sentinel-2/tiles/52/S/DG/S2A_MSIL1C_20181101T020821_N0206_R103_T52SDG_20181101T040328.SAFE',
        'http://storage.googleapis.com/gcp-public-data-sentinel-2/tiles/52/S/DG/S2A_MSIL1C_20181104T021841_N0206_R003_T52SDG_20181104T055443.SAFE',
        'http://storage.googleapis.com/gcp-public-data-sentinel-2/tiles/52/S/DG/S2B_MSIL1C_20181106T020849_N0207_R103_T52SDG_20181106T034331.SAFE'
    ]
    url = urls[index]

    # By default cache to the $XDG_CACHE_HOME/geowatch
    dset_dpath = ub.Path.appdir('geowatch/demo/grab_s2').ensuredir()

    # Cache the scene using the same path used by google cloud storage
    tile_hierarchy = os.path.sep.join(url.split('tiles/')[1].split('/')[:3])
    scene_dpath = ub.ensuredir((dset_dpath, tile_hierarchy))

    safe_dpath = pathlib.Path(scene_dpath) / basename(url)

    was_failed_download = (
        not safe_dpath.exists() or not list(safe_dpath.glob('*'))
    )

    """
    # We could vendor code from fels if we wanted.
    import liberator
    from fels.sentinel2 import get_sentinel2_image
    lib = liberator.Liberator()
    lib.add_dynamic(get_sentinel2_image)
    print(lib.current_sourcecode()
    """
    try:
        from fels.sentinel2 import get_sentinel2_image
    except ImportError:
        from fels import get_sentinel2_image

    if was_failed_download:
        # This is really slow even if the data is cached with default fels
        # This PR: https://github.com/vascobnunes/fetchLandsatSentinelFromGoogleCloud/pull/58
        # can help speed it up, which is merged but not releases as of
        # 2021-10-30

        # Download the scene
        assert get_sentinel2_image(url,
                                   scene_dpath,
                                   # overwrite=was_failed_download,
                                   overwrite=True,  # makes this really slow
                                   reject_old=True)

    # Build a rgdc object to return the scene
    root = join(scene_dpath, url.split('/')[-1])
    fpaths = sorted(glob(join(root, '**', '*.*'), recursive=True))
    fpaths = [pathlib.Path(relpath(f, start=root)) for f in fpaths]
    product = RasterDownload(
        path=pathlib.Path(root),
        images=[f for f in fpaths if f.suffix == '.jp2'],
        ancillary=[f for f in fpaths if f.suffix != '.jp2'])
    return product
