# -*- coding: utf-8 -*-
import numpy as np
import kwimage as ki
import rasterio
from tempenv import TemporaryEnvironment
import ubelt as ub
import math

# NOTE: This exists in kwimage now


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

    bands = normalize_intensity(bands, nodata=0)
    bands = ki.ensure_uint255(bands)
    bands = ki.imresize(bands, max_dim=max_dim)
    return bands


def find_robust_normalizers(data, params='auto'):
    """
    Finds robust normalization statistics for a single observation

    Args:
        data (ndarray): a 1D numpy array where invalid data has already been removed

        params (str | dict): normalization params

    Returns:
        Dict[str, str | float]: normalization parameters

    TODO:
        - [ ] No Magic Numbers! Use first principles to deterimine defaults.
        - [ ] Probably a lot of literature on the subject.
        - [ ] Is this a kwarray function in general?

    Example:
        >>> data = np.random.rand(100)
        >>> norm_params1 = find_robust_normalizers(data, params='auto')
        >>> norm_params2 = find_robust_normalizers(data, params={'low': 0, 'high': 1.0})
        >>> norm_params3 = find_robust_normalizers(np.empty(0), params='auto')
        >>> print('norm_params1 = {}'.format(ub.repr2(norm_params1, nl=1)))
        >>> print('norm_params2 = {}'.format(ub.repr2(norm_params2, nl=1)))
        >>> print('norm_params3 = {}'.format(ub.repr2(norm_params3, nl=1)))
    """
    if data.size == 0:
        normalizer = {
            'type': None,
            'min_val': np.nan,
            'max_val': np.nan,
        }
    else:
        # should center the desired distribution to visualize on zero
        # beta = np.median(imdata)
        default_params = {
            'low': 0.01,
            'mid': 0.5,
            'high': 0.9,
            'mode': 'sigmoid',
        }
        if isinstance(params, str):
            if params == 'auto':
                params = {}
            else:
                raise KeyError(params)

        params = ub.dict_union(default_params, params)
        quant_low = params['low']
        quant_mid = params['mid']
        quant_high = params['high']
        qvals = [0, quant_low, quant_mid, quant_high, 1]
        quantile_vals = np.quantile(data, qvals)

        (quant_low_abs, quant_low_val, quant_mid_val, quant_high_val,
         quant_high_abs) = quantile_vals

        # TODO: we could implement a hueristic where we do a numerical inspection
        # of the intensity distribution. We could apply a normalization that is
        # known to work for data with that sort of histogram distribution.
        # This might involve fitting several parametarized distributions to the
        # data and choosing the one with the best fit. (check how many modes there
        # are).

        # inner_range = quant_high_val - quant_low_val
        # upper_inner_range = quant_high_val - quant_mid_val
        # upper_lower_range = quant_mid_val - quant_low_val
        # http://mathcenter.oxford.emory.edu/site/math117/shapeCenterAndSpread/

        # Compute amount of weight in each quantile
        quant_center_amount = (quant_high_val - quant_low_val)
        quant_low_amount = (quant_mid_val - quant_low_val)
        quant_high_amount = (quant_high_val - quant_mid_val)

        if math.isclose(quant_center_amount, 0):
            high_weight = 0.5
            low_weight = 0.5
        else:
            high_weight = quant_high_amount / quant_center_amount
            low_weight = quant_low_amount / quant_center_amount

        quant_high_residual = (1.0 - quant_high)
        quant_low_residual = (quant_low - 0.0)
        # todo: verify, having slight head fog, not 100% sure
        low_pad_val = quant_low_residual * (low_weight * quant_center_amount)
        high_pad_val = quant_high_residual * (high_weight * quant_center_amount)

        min_val = max(quant_low_abs, quant_low_val - low_pad_val)
        max_val = max(quant_high_abs, quant_high_val - high_pad_val)

        beta = quant_mid_val
        # division factor
        # from scipy.special import logit
        # alpha = max(abs(old_min - beta), abs(old_max - beta)) / logit(0.998)
        # This chooses alpha such the original min/max value will be pushed
        # towards -1 / +1.
        alpha = max(abs(min_val - beta), abs(max_val - beta)) / 6.212606

        normalizer = {
            'type': 'normalize',
            'mode': params['mode'],
            'min_val': min_val,
            'max_val': max_val,
            'beta': beta,
            'alpha': alpha,
        }
    return normalizer


# NOTE: This exists in kwimage now
def normalize_intensity(imdata, return_info=False, nodata=None, axis=None,
                        dtype=np.float32, params='auto', mask=None):
    """
    Normalize data intensities using heuristics to help put sensor data with
    extremely high or low contrast into a visible range.

    This function is designed with an emphasis on getting something that is
    reasonable for visualization.

    TODO:
        - [ ] Move to kwarray and renamed to robust_normalize?
        - [ ] Support for M-estimators?

    Args:
        imdata (ndarray): raw intensity data

        return_info (bool, default=False):
            if True, return information about the chosen normalization
            heuristic.

        params (str | dict):
            can contain keys, low, high, or center

        nodata:
            A value representing nodata to leave unchanged during
            normalization, for example 0

        dtype : can be float32 or float64

    Returns:
        ndarray: a floating point array with values between 0 and 1.

    Example:
        >>> import ubelt as ub
        >>> import kwimage
        >>> import kwarray
        >>> s = 512
        >>> bit_depth = 11
        >>> dtype = np.uint16
        >>> max_val = int(2 ** bit_depth)
        >>> min_val = int(0)
        >>> rng = kwarray.ensure_rng(0)
        >>> background = np.random.randint(min_val, max_val, size=(s, s), dtype=dtype)
        >>> poly1 = kwimage.Polygon.random(rng=rng).scale(s / 2)
        >>> poly2 = kwimage.Polygon.random(rng=rng).scale(s / 2).translate(s / 2)
        >>> forground = np.zeros_like(background, dtype=np.uint8)
        >>> forground = poly1.fill(forground, value=255)
        >>> forground = poly2.fill(forground, value=122)
        >>> forground = (kwimage.ensure_float01(forground) * max_val).astype(dtype)
        >>> imdata = background + forground
        >>> normed, info = normalize_intensity(imdata, return_info=True)
        >>> print('info = {}'.format(ub.repr2(info, nl=1)))
        >>> # xdoctest: +REQUIRES(--show)
        >>> import kwplot
        >>> kwplot.autompl()
        >>> kwplot.imshow(imdata, pnum=(1, 2, 1), fnum=1)
        >>> kwplot.imshow(normed, pnum=(1, 2, 2), fnum=1)

    Example:
        >>> import ubelt as ub
        >>> import kwimage
        >>> # Test on an image that is already normalized to test how it
        >>> # degrades
        >>> imdata = kwimage.grab_test_image() / 255

        >>> quantile_basis = {
        >>>     'mode': ['linear', 'sigmoid'],
        >>>     'high': [0.8, 0.9, 1.0],
        >>> }
        >>> quantile_grid = list(ub.named_product(quantile_basis))
        >>> quantile_grid += ['auto']
        >>> rows = []
        >>> rows.append({'key': 'orig', 'result': imdata})
        >>> for params in quantile_grid:
        >>>     key = ub.repr2(params, compact=1)
        >>>     result, info = normalize_intensity(imdata, return_info=True, params=params)
        >>>     print('key = {}'.format(key))
        >>>     print('info = {}'.format(ub.repr2(info, nl=1)))
        >>>     rows.append({'key': key, 'info': info, 'result': result})
        >>> # xdoctest: +REQUIRES(--show)
        >>> import kwplot
        >>> kwplot.autompl()
        >>> pnum_ = kwplot.PlotNums(nSubplots=len(rows))
        >>> for row in rows:
        >>>     _, ax = kwplot.imshow(row['result'], fnum=1, pnum=pnum_())
        >>>     ax.set_title(row['key'])
    """
    import kwarray

    if axis is not None:
        # Hack, normalize each channel individually. This could
        # be implementd more effciently.
        assert not return_info
        reorg = imdata.swapaxes(0, axis)
        if mask is None:
            parts = []
            for item in reorg:
                part = normalize_intensity(item, nodata=nodata, axis=None)
                parts.append(part[None, :])
        else:
            reorg_mask = mask.swapaxes(0, axis)
            parts = []
            for item, item_mask in zip(reorg, reorg_mask):
                part = normalize_intensity(item, nodata=nodata, axis=None, mask=item_mask)
                parts.append(part[None, :])
        recomb = np.concatenate(parts, axis=0)
        final = recomb.swapaxes(0, axis)
        return final

    if mask is None:
        if nodata is not None:
            mask = imdata != nodata
            imdata_valid = imdata[mask]
        else:
            mask = None
            imdata_valid = imdata
    else:
        imdata_valid = imdata[mask]

    normalizer = find_robust_normalizers(imdata_valid, params=params)

    if normalizer['type'] is None:
        imdata_normalized = imdata.astype(dtype)
    elif normalizer['type'] == 'normalize':
        # Note: we are using kwarray normalize, the one in kwimage is deprecated
        imdata_normalized = kwarray.normalize(
            imdata.astype(dtype), mode=normalizer['mode'],
            beta=normalizer['beta'], alpha=normalizer['alpha'],
        )
    else:
        raise KeyError(normalizer['type'])

    if mask is not None:
        result = np.where(mask, imdata_normalized, imdata)
    else:
        result = imdata_normalized

    if return_info:
        return result, normalizer
    else:
        return result
