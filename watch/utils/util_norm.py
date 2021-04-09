# -*- coding: utf-8 -*-
import numpy as np


def normalize_intensity(imdata, return_info=False, nodata=None):
    """
    Normalize data intensities, with an emphasis on visualization.

    Args:
        imdata (ndarray): raw data read from the geotiff.
        return_info (bool, default=False): if True, return information about
            the chosen normalization heuristic.
        nodata: A value representing nodata to leave unchanged during
            normalization, for example 0

    Example:
        >>> from watch.utils.util_norm import *  # NOQA
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
    """
    import kwimage

    info = {}

    # if len(imdata.shape) > 2 and imdata.shape[2] > 4:
    #     # Just select one channel for now
    #     imdata = imdata[:, :, 0]

    if nodata is not None:
        imdata_valid = imdata[imdata != nodata]
    else:
        imdata_valid = imdata

    if imdata.dtype.itemsize > 1 and imdata.dtype.kind in {'i', 'u', 'f'}:
        # should center the desired distribution to visualize on zero
        # beta = np.median(imdata)
        quant_low = 0.01
        quand_mid = 0.5
        quant_high = 0.99

        quantiles = np.quantile(imdata_valid, [0, quant_low, quand_mid, quant_high, 1])
        (quant_low_abs, quant_low_val, quant_mid_val, quant_high_val,
         quant_high_abs) = quantiles

        # Compute amount of weight in each quantile
        quant_center_amount = (quant_high_val - quant_low_val)
        quant_low_amount = (quant_mid_val - quant_low_val)
        quant_high_amount = (quant_high_val - quant_mid_val)

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
        # This chooses alpha such the original min/max value will be pushed
        # towards -1 / +1.
        alpha = max(abs(min_val - beta), abs(max_val - beta)) / 6.212606

        info.update({
            'min_val': min_val,
            'max_val': max_val,
            'beta': beta,
            'alpha': alpha,
            'mode': 'sigmoid'
        })

        imdata_normalized = kwimage.normalize(
            imdata.astype(np.float32), mode='sigmoid', beta=beta, alpha=alpha)
    else:
        imdata_normalized = imdata

    if nodata is not None:
        result = np.where(imdata != nodata, imdata_normalized, imdata)
    else:
        result = imdata_normalized

    if return_info:
        return result, info
    else:
        return result
