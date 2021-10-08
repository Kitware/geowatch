"""
Functions that may eventually be moved to kwarray
"""
import numpy as np


def cartesian_product(*arrays):
    """
    Fast numpy version of itertools.product

    TODO: Move to kwarray

    Referencs:
        https://stackoverflow.com/a/11146645/887074
    """
    la = len(arrays)
    dtype = np.result_type(*arrays)
    arr = np.empty([len(a) for a in arrays] + [la], dtype=dtype)
    for i, a in enumerate(np.ix_(*arrays)):
        arr[..., i] = a
    return arr.reshape(-1, la)


def tukey_biweight_loss(r, c=4.685):
    """
    Beaton Tukey Biweight

    Computes the function :
        L(r) = (
            (c ** 2) / 6 * (1 - 1 * (r / c) ** 2) ** 3) if abs(r) <= c else
            (c ** 2)
        )

    Args:
        r (float | ndarray): residual parameter
        c (float): tuning constant (defaults to 4.685 which is 95% efficient
            for normal distributions of residuals)

    TODO:
        - [ ] Move elsewhere or find a package that provides it
        - [ ] Move elsewhere (kwarray?) or find a package that provides it

    Returns:
        float | ndarray

    References:
        https://en.wikipedia.org/wiki/Robust_statistics
        https://mathworld.wolfram.com/TukeysBiweight.html
        https://statisticaloddsandends.wordpress.com/2021/04/23/what-is-the-tukey-loss-function/
        https://arxiv.org/pdf/1505.06606.pdf

    Example:
        >>> from watch.utils.util_kwarray import *  # NOQA
        >>> import ubelt as ub
        >>> r = np.linspace(-20, 20, 1000)
        >>> data = {'r': r}
        >>> grid = ub.named_product({
        >>>     'c': [4.685, 2, 6],
        >>> })
        >>> for kwargs in grid:
        >>>     key = ub.repr2(kwargs, compact=1)
        >>>     loss = tukey_biweight_loss(r, **kwargs)
        >>>     data[key] = loss
        >>> import pandas as pd
        >>> melted = pd.DataFrame(data).melt(['r'])
        >>> # xdoctest: +REQUIRES(--show)
        >>> import kwplot
        >>> sns = kwplot.autosns()
        >>> kwplot.figure(fnum=1, doclf=True)
        >>> ax = sns.lineplot(data=melted, x='r', y='value', hue='variable', style='variable')
        >>> #ax.set_ylim(*robust_limits(melted.value))
    """
    # https://statisticaloddsandends.wordpress.com/2021/04/23/what-is-the-tukey-loss-function/
    is_inside = np.abs(r) < c
    c26 = (c ** 2) / 6
    loss = np.full_like(r, fill_value=c26, dtype=np.float32)
    r_inside = r[is_inside]
    loss_inside = c26 * (1 - (1 - (r_inside / c) ** 2) ** 3)
    loss[is_inside] = loss_inside
    return loss


def asymptotic(x, offset=1, gamma=1, horizontal=1):
    """
    A function with a horizontal asymptote at ``horizontal``

    Args:
        x (ndarray): input parameter
        offset (float): shifts function to the left or the right
        gamma (float): higher values approach the asymptote more slowly
        horizontal (float): location of the horiztonal asymptote

    TODO:
        - [ ] Move elsewhere (kwarray?) or find a package that provides it

    Example:
        >>> from watch.utils.util_kwarray import *  # NOQA
        >>> import ubelt as ub
        >>> x = np.linspace(-5, 29, 1000)
        >>> data = {'x': x}
        >>> grid = ub.named_product({
        >>>     'gamma': [0.5, 1.0, 2.0],
        >>>     'offset': [0, 2],
        >>>     'horizontal': [1],
        >>> })
        >>> for kwargs in grid:
        >>>     key = ub.repr2(kwargs, compact=1)
        >>>     data[key] = asymptotic(x, **kwargs)
        >>> import pandas as pd
        >>> melted = pd.DataFrame(data).melt(['x'])
        >>> print(melted)
        >>> # xdoctest: +REQUIRES(--show)
        >>> import kwplot
        >>> sns = kwplot.autosns()
        >>> kwplot.figure(fnum=1, doclf=True)
        >>> ax = sns.lineplot(data=melted, x='x', y='value', hue='variable', style='variable')
        >>> ax.set_ylim(0, 2)
    """
    return ((x + offset) ** gamma / (x + offset + 1) ** gamma) + (horizontal - 1)


def robust_limits(values):
    """
    # TODO: Proper Robust estimator for matplotlib ylim and general use

    values = np.array([-1000, -4, -3, -2, 0, 2.7, 3.1415, 1, 2, 3, 4, 100000])
    robust_limits(values)
    """
    quants = [0.0, 0.05, 0.08, 0.2, 0.5, 0.8, 0.9, 0.5, 1.0]
    values = values[~np.isnan(values)]
    quantiles = np.quantile(values, quants)
    print('quantiles = {!r}'.format(quantiles))

    lower_idx1 = 1
    upper_idx1 = 2
    part = quantiles[upper_idx1] - quantiles[lower_idx1]
    inner_w = quants[upper_idx1] - quants[lower_idx1]
    extrap_w = quants[lower_idx1] - quants[0]

    extrap_part = part * extrap_w / inner_w
    low_value = quantiles[lower_idx1]
    robust_min = low_value - extrap_part
    #
    lower_idx2 = -3
    upper_idx2 = -2
    high_value = quantiles[upper_idx2]
    part = quantiles[upper_idx2] - quantiles[lower_idx2]
    inner_w = quants[upper_idx2] - quants[lower_idx2]
    extrap_w = quants[lower_idx1] - quants[0]

    extrap_part = part * extrap_w / inner_w
    robust_max = high_value + extrap_part

    robust_min
    return robust_min, robust_max


def unique_rows(arr, ordered=False):
    """
    Note: function also added to kwarray and will be available in >0.5.20

    Example:
        >>> import kwarray
        >>> from kwarray.util_numpy import *  # NOQA
        >>> rng = kwarray.ensure_rng(0)
        >>> arr = rng.randint(0, 2, size=(12, 3))
        >>> arr_unique = unique_rows(arr)
        >>> print('arr_unique = {!r}'.format(arr_unique))
    """
    dtype_view = np.dtype((np.void, arr.dtype.itemsize * arr.shape[1]))
    arr_view = arr.view(dtype_view)
    if ordered:
        arr_view_unique, idxs = np.unique(arr_view, return_index=True)
        arr_flat_unique = arr_view_unique.view(arr.dtype)
        arr_unique = arr_flat_unique.reshape(-1, arr.shape[1])
        arr_unique = arr_unique[np.argsort(idxs)]
    else:
        arr_view_unique = np.unique(arr_view)
        arr_flat_unique = arr_view_unique.view(arr.dtype)
        arr_unique = arr_flat_unique.reshape(-1, arr.shape[1])
    return arr_unique
