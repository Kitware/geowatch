"""
aggregation functions for heatmaps
"""


def _norm(heatmaps, norm_ord):
    """
    Computes the generalized mean over axis=0.

    Args:
        heatmaps (List[ndarray]) pixel aligned heatmaps
        norm_ord (int | float): the exponent of the generalized mean.

    Returns:
        ndarray : the axis=0 is marginalized over.

    Notes:
        like np.linalg.norm but with special nan handling and a division factor

    References:
        https://en.wikipedia.org/wiki/Generalized_mean
        https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.pmean.html

    Example:
        >>> from geowatch.tasks.tracking.agg_functions import *  # NOQA
        >>> from geowatch.tasks.tracking.agg_functions import _norm
        >>> import ubelt as ub
        >>> import kwimage
        >>> import numpy as np
        >>> num_frames = 16
        >>> num_sequences = 6
        >>> # Setup 5 sequences to norm
        >>> heatmaps = [np.empty(num_sequences) for _ in range(num_frames)]
        >>> heatmaps = np.array(heatmaps)
        >>> # Sequence 0 is all nan
        >>> heatmaps[:, 0] = np.nan
        >>> # Sequence 1 is random
        >>> heatmaps[:, 1] = np.random.rand(num_frames)
        >>> # Sequence 2 is Sequence1, but half of the data is nan
        >>> heatmaps[0:, 2] = heatmaps[:, 1]
        >>> heatmaps[0:num_frames // 2, 2] = np.nan
        >>> # Sequence 3 is all zero except for an impulse
        >>> heatmaps[0:, 3] = 0
        >>> heatmaps[num_frames // 2, 3] = 1
        >>> # Sequence 4 is a gaussian response
        >>> heatmaps[0:, 4] = kwimage.gaussian_patch(shape=(1, num_frames))[0]
        >>> # Sequence 5 is a a gaussian response 1 / 4 nans
        >>> heatmaps[0:, 5] = kwimage.gaussian_patch(shape=(1, num_frames))[0]
        >>> heatmaps[0:num_frames // 4, 5] = np.nan
        >>> norm_ord = 1
        >>> x = _norm(heatmaps, norm_ord)
        >>> y = np.linalg.norm(heatmaps, ord=norm_ord, axis=0)
        >>> print('heatmaps = {}'.format(ub.urepr(heatmaps, nl=1, precision=2)))
        >>> print(x)
        >>> print(y)
        >>> # xdoctest: +REQUIRES(--show)
        >>> # Visualize how this works for random signals
        >>> import kwplot
        >>> sns = kwplot.sns
        >>> kwplot.plt.ion()
        >>> # kwplot.close_figures()
        >>> # Add in the original signals
        >>> rows = []
        >>> for c in range(num_sequences):
        >>>     for x in range(num_frames):
        >>>         rows.append(
        >>>             {'x': x, 'col': c, 'ord': 'raw-signal', 'value': heatmaps[x, c]})
        >>> #
        >>> import pandas as pd
        >>> import scipy.stats
        >>> for norm_ord in [1, 2, 3, float('inf')]:
        >>>     v1 = _norm(heatmaps, norm_ord)
        >>>     v2 = scipy.stats.pmean(heatmaps, p=norm_ord, axis=0, nan_policy='omit')
        >>>     print(f'norm_ord={norm_ord}')
        >>>     print(f'v1={v1}')
        >>>     print(f'v2={v2}')
        >>>     for c in range(num_sequences):
        >>>         for x in range(num_frames):
        >>>             rows.append({'x': x, 'col': c, 'ord': norm_ord, 'value': v1[c]})
        >>>     ...
        >>> df = pd.DataFrame(rows)
        >>> pnum_ = kwplot.PlotNums(nSubplots=num_sequences)
        >>> for c in range(num_sequences):
        >>>     kwplot.figure(fnum=1, pnum=pnum_())
        >>>     subdata = df[df['col'] == c]
        >>>     sns.lineplot(data=subdata, x='x', y='value', hue='ord')
    """
    import numpy as np
    heatmaps = np.array(heatmaps)
    if norm_ord == 0:
        import scipy.stats
        probs = scipy.stats.pmean(heatmaps, p=norm_ord, axis=0, nan_policy='omit')
        probs = np.nan_to_num(probs)
    elif norm_ord == np.inf:
        probs = np.nanmax(heatmaps, axis=0)
    else:
        # The np.linalg.norm part
        probs = np.power(np.nansum(np.power(heatmaps, norm_ord), axis=0),
                         1. / norm_ord)
        if norm_ord > 0:
            n_nonzero = np.count_nonzero(~np.isnan(heatmaps), axis=0)
            # Force the denominator to be positive.
            n_nonzero[n_nonzero == 0] = 1
            probs /= np.power(n_nonzero, 1. / norm_ord)
    return probs


# give all these the same signature so they can be swapped out


def binary(heatmaps, norm_ord, morph_kernel, thresh, viz_dpath=None):
    import kwimage
    probs = _norm(heatmaps, norm_ord)

    hard_probs = kwimage.morphology(probs > thresh, 'dilate', morph_kernel)

    return hard_probs.astype('float')


def rescaled_binary(heatmaps, norm_ord, morph_kernel, thresh, upper_quantile=0.999, viz_dpath=None):
    import kwimage
    import kwarray
    import numpy as np
    probs = _norm(heatmaps, norm_ord)
    probs = kwarray.normalize(probs, min_val=0, max_val=np.quantile(probs, upper_quantile))

    hard_probs = kwimage.morphology(probs > thresh, 'dilate', morph_kernel)

    return hard_probs.astype('float')


def probs(heatmaps, norm_ord, morph_kernel, thresh, viz_dpath=None):
    import kwimage
    probs = _norm(heatmaps, norm_ord)

    hard_probs = kwimage.morphology(probs > thresh, 'dilate', morph_kernel)
    modulated_probs = probs * hard_probs

    if viz_dpath is not None:
        kwimage.imwrite(viz_dpath / 'probs_raw.png', kwimage.ensure_uint255(probs))
        kwimage.imwrite(viz_dpath / 'probs_hard.png', kwimage.ensure_uint255(hard_probs))
        kwimage.imwrite(viz_dpath / 'probs_modulated.png', kwimage.ensure_uint255(modulated_probs))

    return modulated_probs


def rescaled_probs(heatmaps, norm_ord, morph_kernel, thresh, upper_quantile=0.999, viz_dpath=None):
    import kwimage
    import kwarray
    import numpy as np
    probs = _norm(heatmaps, norm_ord)
    probs = kwarray.normalize(probs, min_val=0, max_val=np.quantile(probs, upper_quantile))

    hard_probs = kwimage.morphology(probs > thresh, 'dilate', morph_kernel)
    modulated_probs = probs * hard_probs

    return modulated_probs


def mean_normalized(heatmaps, norm_ord=1, morph_kernel=1, thresh=None, viz_dpath=None):
    """
    Normalize average_heatmap by applying a scaling based on max(heatmaps) and
    max(average_heatmap)
    """
    import numpy as np
    import kwimage
    average = _norm(heatmaps, norm_ord)

    scale_factor = np.max(heatmaps) / (np.max(average) + 1e-9)
    print('max heatmaps', np.max(heatmaps))
    print('max average', np.max(average))

    # average *= scale_factor
    average = 0.75 * average * scale_factor
    print('scale_factor', scale_factor)
    print('After scaling, max:', np.max(average))

    average = kwimage.morphology(average, 'dilate', morph_kernel)

    return average


def frequency_weighted_mean(heatmaps, thresh, norm_ord=0, morph_kernel=3, viz_dpath=None):
    """
    Convert a list of heatmaps to an aggregated score, averaging is computed
    based on samples for every pixel
    """
    import kwimage
    import numpy as np
    heatmaps = np.array(heatmaps)

    masks = 1 * (heatmaps > thresh)
    pixel_wise_samples = masks.sum(0) + 1e-9
    print('pixel_wise_samples', pixel_wise_samples)

    # compute sum
    aggregated_probs = _norm(masks * heatmaps, norm_ord)

    # divide by number of samples for every pixel
    aggregated_probs /= pixel_wise_samples

    aggregated_probs = kwimage.morphology(aggregated_probs, 'dilate',
                                          morph_kernel)

    return aggregated_probs


AGG_FN_REGISTRY = {
    'frequency_weighted_mean': frequency_weighted_mean,
    'mean_normalized': mean_normalized,
    'rescaled_probs': rescaled_probs,
    'probs': probs,
    'rescaled_binary': rescaled_binary,
    'binary': binary,
}
