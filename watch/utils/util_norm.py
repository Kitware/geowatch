import numpy as np
import kwimage as ki
import rasterio
from tempenv import TemporaryEnvironment
import ubelt as ub


def thumbnail(in_path, sat=None, max_dim=1000):
    """
    Create a small, true-color [1] thumbnail from a satellite image.

    Args:
        in_path: path to a S2, L7, or L8 scene readable by gdal
        sat: if None, assume this is a true color (3-channel) or panchromatic (1-channel) image.
            Otherwise, grab 3 color bands for 'WV', 'S2', 'L7', or 'L8'.
        max_dim: max dimension of the thumbnail

    Returns:
        uint8, 3-channel RGB image content.

    References:
        [1] https://github.com/sentinel-hub/custom-scripts/tree/master
    """
    # workaround for
    # https://rasterio.readthedocs.io/en/latest/faq.html#why-can-t-rasterio-find-proj-db-rasterio-from-pypi-versions-1-2-0
    import kwimage
    with TemporaryEnvironment({'PROJ_LIB': None}):
        with rasterio.open(in_path) as f:
            if sat is None:
                assert f.indexes.issubset((1,), (3,)), f'{in_path} is not PAN or TCI.'
                bands = f.read()
                bands = ki.atleast_3channels(bands)
            elif sat == 'S2':
                bands = np.stack([f.read(4), f.read(3), f.read(2)], axis=-1)
            elif sat == 'L8':
                bands = np.stack([f.read(4), f.read(3), f.read(2)], axis=-1)
            elif sat == 'L7':
                bands = np.stack([f.read(3), f.read(2), f.read(1)], axis=-1)
            elif sat == 'WV':
                bands = np.stack([f.read(5), f.read(3), f.read(2)], axis=-1)

    bands = kwimage.normalize_intensity(bands)
    bands = ki.ensure_uint255(bands)
    bands = ki.imresize(bands, max_dim=max_dim)
    return bands


def find_robust_normalizers(data, params='auto'):
    """
    DEPRECATE: THIS IS IN KWARRAY
    """
    import kwimage
    ub.schedule_deprecation(
        'watch', 'find_robust_normalizers', 'use kwarray.find_robust_normalizers instead.',
        deprecate='now',
    )
    normalizer = kwimage.find_robust_normalizers(data, params=params)
    return normalizer
