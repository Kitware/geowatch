import ubelt as ub
import os
from pathlib import Path
from glob import glob

import fels
from rgdc.rgdc import RasterDownload


def grab_sentinel2_product(index=0):
    """
    Download and cache all items for a Sentinel-2 product.
    
    TODO when RGD supports API keys, give one to this repo and use that instead of fels.
    
    Args:
        index: 0, 1, or 2. Currently this function just picks 3 scenes over KR in Nov 2018.

    Returns:
        rgdc.rgdc.RasterDownload(
            path: pathlib.Path,
            images: List[pathlib.Path],
            ancillary: List[pathlib.Path]
        )

    Example:
        >>> # xdoctest: +REQUIRES(--network)
        >>> from watch.demo.sentinel2_demodata import *  # NOQA
        >>> from watch.utils.util_rgdc import bands_sentinel2
        >>> product = grab_sentinel2_product()
        >>> assert len(bands_sentinel2(product)) == 13

    SeeAlso:
        watch.util.util_rgdc.bands_sentinel2
    """

    urls = [
        'http://storage.googleapis.com/gcp-public-data-sentinel-2/tiles/52/S/DG/S2A_MSIL1C_20181104T021841_N0206_R003_T52SDG_20181104T055443.SAFE',
        'http://storage.googleapis.com/gcp-public-data-sentinel-2/tiles/52/S/DG/S2A_MSIL1C_20181101T020821_N0206_R103_T52SDG_20181101T040328.SAFE',
        'http://storage.googleapis.com/gcp-public-data-sentinel-2/tiles/52/S/DG/S2B_MSIL1C_20181106T020849_N0207_R103_T52SDG_20181106T034331.SAFE'
    ]
    url = urls[index]

    # By default cache to the $XDG_CACHE_HOME/smart_watch
    dset_dpath = ub.ensure_app_cache_dir('smart_watch')

    # Cache the scene using the same path used by google cloud storage
    scene_dpath = ub.ensuredir((dset_dpath, url.split('tiles/')[1]))

    # Download the scene
    assert fels.get_sentinel2_image(url,
                                    scene_dpath,
                                    overwrite=False,
                                    reject_old=True)

    # Build a rgdc object to return the scene
    fpaths = sorted(
        glob(os.path.join(scene_dpath, '**', '*.*'), recursive=True))
    fpaths = [Path(os.path.relpath(f, start=scene_dpath)) for f in fpaths]
    return RasterDownload(path=Path(scene_dpath),
                          images=[f for f in fpaths if f.suffix == '.jp2'],
                          ancillary=[f for f in fpaths if f.suffix != '.jp2'])
