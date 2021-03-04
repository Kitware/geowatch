

def normalize_intensity(imdata, return_info=False):
    """
    Normalize data intensities, with an emphasis on visualization.

    Args:
        imdata (ndarray): raw data read from the geotiff.
    """
    import kwimage
    import numpy as np

    info = {}

    if len(imdata.shape) > 2 and imdata.shape[2] > 4:
        # Just select one channel for now
        imdata = imdata[:, :, 0]

    if imdata.dtype.itemsize > 1 and imdata.dtype.kind in {'i', 'u'}:
        # should center the desired distribution to visualize on zero
        # beta = np.median(imdata)
        quant_low = 0.01
        quand_mid = 0.5
        quant_high = 0.99

        quantiles = np.quantile(imdata, [0, quant_low, quand_mid, quant_high, 1])
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

        imdata = kwimage.normalize(
            imdata.astype(np.float32), mode='sigmoid', beta=beta, alpha=alpha)

    if return_info:
        return imdata, info
    else:
        return imdata
