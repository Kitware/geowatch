"""
Functions that may eventually be moved to kwarray
"""
import functools
import itertools as it
import math
import numpy as np
import os
import ubelt as ub
import warnings

try:
    from packaging.version import parse as Version
except ImportError:
    from distutils.version import LooseVersion as Version


try:
    import importlib.metadata
    try:
        _TORCH_VERSION = Version(importlib.metadata.version('torch'))
    except importlib.metadata.PackageNotFoundError:
        _TORCH_VERSION = None
except ImportError:
    import pkg_resources
    try:
        _TORCH_VERSION = Version(pkg_resources.get_distribution('torch').version)
    except pkg_resources.DistributionNotFound:
        _TORCH_VERSION = None

if _TORCH_VERSION is None:
    _TORCH_LT_1_7_0 = None
    _TORCH_LT_2_1_0 = None
    _TORCH_HAS_MAX_BUG = None
else:
    _TORCH_LT_1_7_0 = _TORCH_VERSION < Version('1.7')
    _TORCH_LT_2_1_0 = _TORCH_VERSION < Version('2.1')
    _TORCH_HAS_MAX_BUG = _TORCH_LT_1_7_0


try:
    # The math variant only exists in Python 3+ but is faster for scalars
    # so try and use it
    from math import isclose
except Exception:
    from numpy import isclose


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
        >>> from geowatch.utils.util_kwarray import *  # NOQA
        >>> import ubelt as ub
        >>> r = np.linspace(-20, 20, 1000)
        >>> data = {'r': r}
        >>> grid = ub.named_product({
        >>>     'c': [4.685, 2, 6],
        >>> })
        >>> for kwargs in grid:
        >>>     key = ub.urepr(kwargs, compact=1)
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


def asymptotic(x, offset=1, gamma=1, degree=0, horizontal=1):
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
        >>> from geowatch.utils.util_kwarray import *  # NOQA
        >>> import ubelt as ub
        >>> x = np.linspace(0, 27, 1000)
        >>> data = {'x': x}
        >>> grid = ub.named_product({
        >>>     #'gamma': [0.5, 1.0, 2.0, 3.0],
        >>>     'gamma': [1.0, 3.0],
        >>>     'degree': [0, 1, 2, 3],
        >>>     'offset': [0, 2],
        >>>     'horizontal': [1],
        >>> })
        >>> for kwargs in grid:
        >>>     key = ub.urepr(kwargs, compact=1)
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
    gamma_denom = gamma + degree
    gamma_numer = gamma
    assert gamma_numer <= gamma_denom
    hz_offset = horizontal - 1 if gamma_numer == gamma_denom else horizontal
    numer = (x + offset) ** gamma_numer
    denom = (x + offset + 1) ** gamma_denom
    return (numer / denom) + hz_offset


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
        - [ ] https://arxiv.org/pdf/1707.09752.pdf
        - [ ] https://www.tandfonline.com/doi/full/10.1080/02664763.2019.1671961
        - [ ] https://www.rips-irsp.com/articles/10.5334/irsp.289/

    Example:
        >>> data = np.random.rand(100)
        >>> norm_params1 = find_robust_normalizers(data, params='auto')
        >>> norm_params2 = find_robust_normalizers(data, params={'low': 0, 'high': 1.0})
        >>> norm_params3 = find_robust_normalizers(np.empty(0), params='auto')
        >>> print('norm_params1 = {}'.format(ub.urepr(norm_params1, nl=1)))
        >>> print('norm_params2 = {}'.format(ub.urepr(norm_params2, nl=1)))
        >>> print('norm_params3 = {}'.format(ub.urepr(norm_params3, nl=1)))
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
            'extrema': 'custom-quantile',
            'scaling': 'linear',
            'low': 0.01,
            'mid': 0.5,
            'high': 0.9,
        }
        fense_extremes = None
        if isinstance(params, str):
            if params == 'auto':
                params = {}
            elif params == 'tukey':
                params = {
                    'extrema': 'tukey',
                }
            elif params == 'std':
                pass
            else:
                raise KeyError(params)

        # hack
        params = ub.dict_union(default_params, params)

        if params['extrema'] == 'tukey':
            # TODO:
            # https://github.com/derekbeaton/OuRS
            # https://en.wikipedia.org/wiki/Feature_scaling
            fense_extremes = _tukey_quantile_extreme_estimator(data)
        elif params['extrema'] == 'custom-quantile':
            fense_extremes = _custom_quantile_extreme_estimator(data, params)
        else:
            raise KeyError(params['extrema'])

        min_val, mid_val, max_val = fense_extremes

        beta = mid_val
        # division factor
        # from scipy.special import logit
        # alpha = max(abs(old_min - beta), abs(old_max - beta)) / logit(0.998)
        # This chooses alpha such the original min/max value will be pushed
        # towards -1 / +1.
        alpha = max(abs(min_val - beta), abs(max_val - beta)) / 6.212606

        normalizer = {
            'type': 'normalize',
            'mode': params['scaling'],
            'min_val': min_val,
            'max_val': max_val,
            'beta': beta,
            'alpha': alpha,
        }
    return normalizer


def _custom_quantile_extreme_estimator(data, params):
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
    mid_val = quant_mid_val
    return (min_val, mid_val, max_val)


def _tukey_quantile_extreme_estimator(data):
    # Tukey method for outliers
    # https://www.youtube.com/watch?v=zY1WFMAA-ec
    q1, q2, q3 = np.quantile(data, [0.25, 0.5, 0.75])
    iqr = q3 - q1
    # One might wonder where the 1.5 in the above interval comes from -- Paul
    # Velleman, a statistician at Cornell University, was a student of John
    # Tukey, who invented this test for outliers. He wondered the same thing.
    # When he asked Tukey, "Why 1.5?", Tukey answered, "Because 1 is too small
    # and 2 is too large."
    # Cite: http://mathcenter.oxford.emory.edu/site/math117/shapeCenterAndSpread/
    fence_lower = q1 - 1.5 * iqr
    fence_upper = q1 + 1.5 * iqr
    return fence_lower, q2, fence_upper


def apply_normalizer(data, normalizer, mask=None, set_value_at_mask=float('nan')):
    dtype = np.float32
    result = data.astype(dtype).copy()

    if normalizer['type'] is None:
        data_normalized = result
    else:
        if mask is not None:
            valid_data = result[mask]
        else:
            valid_data = result

        if valid_data.size > 0:
            data_normalized = normalize(
                valid_data.astype(dtype), mode=normalizer['mode'],
                beta=normalizer.get('beta'), alpha=normalizer.get('alpha'),
                min_val=normalizer.get('min_val'),
                max_val=normalizer.get('max_val')
            )
        else:
            data_normalized = valid_data

    if mask is not None:
        mask_flat = mask.ravel()
        result_flat = result.ravel()
        result_flat[mask_flat] = data_normalized
        result_flat[~mask_flat] = set_value_at_mask
    else:
        result = data_normalized
    return result


def normalize(arr, mode='linear', alpha=None, beta=None, out=None,
              min_val=None, max_val=None):
    """
    Rebalance signal values via contrast stretching.

    By default linearly stretches array values to minimum and maximum values.

    Args:
        arr (ndarray): array to normalize, usually an image

        out (ndarray | None): output array. Note, that we will create an
            internal floating point copy for integer computations.

        mode (str): either linear or sigmoid.

        alpha (float): Only used if mode=sigmoid.  Division factor
            (pre-sigmoid). If unspecified computed as:
            ``max(abs(old_min - beta), abs(old_max - beta)) / 6.212606``.
            Note this parameter is sensitive to if the input is a float or
            uint8 image.

        beta (float): subtractive factor (pre-sigmoid). This should be the
            intensity of the most interesting bits of the image, i.e. bring
            them to the center (0) of the distribution.
            Defaults to ``(max - min) / 2``.  Note this parameter is sensitive
            to if the input is a float or uint8 image.

        min_val: override minimum value

        max_val: override maximum value

    References:
        https://en.wikipedia.org/wiki/Normalization_(image_processing)

    Example:
        >>> raw_f = np.random.rand(8, 8)
        >>> norm_f = normalize(raw_f)
        >>> raw_f = np.random.rand(8, 8) * 100
        >>> norm_f = normalize(raw_f)
        >>> assert isclose(norm_f.min(), 0)
        >>> assert isclose(norm_f.max(), 1)
        >>> raw_u = (np.random.rand(8, 8) * 255).astype(np.uint8)
        >>> norm_u = normalize(raw_u)
        >>> raw_m = (np.zeros((8, 8)) + 10)
        >>> norm_m = normalize(raw_m, min_val=0, max_val=20)
        >>> assert isclose(norm_m.min(), 0.5)
        >>> assert isclose(norm_m.max(), 0.5)
        >>> # Ensure that we're clamping if explicit min or max values
        >>> # are provided
        >>> raw_m = (np.zeros((8, 8)) + 10)
        >>> norm_m = normalize(raw_m, min_val=0, max_val=5)
        >>> assert isclose(norm_m.min(), 1.0)
        >>> assert isclose(norm_m.max(), 1.0)

    Example:
        >>> # xdoctest: +REQUIRES(module:kwimage)
        >>> import kwimage
        >>> arr = kwimage.grab_test_image('lowcontrast')
        >>> arr = kwimage.ensure_float01(arr)
        >>> norms = {}
        >>> norms['arr'] = arr.copy()
        >>> norms['linear'] = normalize(arr, mode='linear')
        >>> norms['sigmoid'] = normalize(arr, mode='sigmoid')
        >>> # xdoctest: +REQUIRES(--show)
        >>> import kwplot
        >>> kwplot.autompl()
        >>> kwplot.figure(fnum=1, doclf=True)
        >>> pnum_ = kwplot.PlotNums(nSubplots=len(norms))
        >>> for key, img in norms.items():
        >>>     kwplot.imshow(img, pnum=pnum_(), title=key)
    """
    if out is None:
        out = arr.copy()

    # TODO:
    # - [ ] Parametarize new_min / new_max values
    #     - [ ] infer from datatype
    #     - [ ] explicitly given
    new_min = 0.0
    if arr.dtype.kind in ('i', 'u'):
        # Need a floating point workspace
        float_out = out.astype(np.float32)
        new_max = float(np.iinfo(arr.dtype).max)
    elif arr.dtype.kind == 'f':
        float_out = out
        new_max = 1.0
    else:
        raise NotImplementedError

    # TODO:
    # - [ ] Parametarize old_min / old_max strategies
    #     - [X] explicitly given min and max
    #     - [ ] raw-naive min and max inference
    #     - [ ] outlier-aware min and max inference
    if min_val is not None:
        old_min = min_val
        float_out[float_out < min_val] = min_val
    else:
        old_min = float_out.min()

    if max_val is not None:
        old_max = max_val
        float_out[float_out > max_val] = max_val
    else:
        old_max = float_out.max()

    old_span = old_max - old_min
    new_span = new_max - new_min

    if mode == 'linear':
        # linear case
        # out = (arr - old_min) * (new_span / old_span) + new_min
        factor = 1.0 if old_span == 0 else (new_span / old_span)
        if old_min != 0:
            float_out -= old_min
    elif mode == 'sigmoid':
        # nonlinear case
        # out = new_span * sigmoid((arr - beta) / alpha) + new_min
        from scipy.special import expit as sigmoid
        if beta is None:
            # should center the desired distribution to visualize on zero
            beta = old_max - old_min

        if alpha is None:
            # division factor
            # from scipy.special import logit
            # alpha = max(abs(old_min - beta), abs(old_max - beta)) / logit(0.998)
            # This chooses alpha such the original min/max value will be pushed
            # towards -1 / +1.
            alpha = max(abs(old_min - beta), abs(old_max - beta)) / 6.212606

        if isclose(alpha, 0):
            alpha = 1

        energy = float_out
        energy -= beta
        energy /= alpha
        # Ideally the data of interest is roughly in the range (-6, +6)
        float_out = sigmoid(energy, out=float_out)
        factor = new_span
    else:
        raise KeyError(mode)

    # Stretch / shift to the desired output range
    if factor != 1:
        float_out *= factor

    if new_min != 0:
        float_out += new_min

    if float_out is not out:
        out[:] = float_out.astype(out.dtype)

    return out


def balanced_number_partitioning(items, num_parts):
    """
    Greedy approximation to multiway number partitioning

    Uses Greedy number partitioning method to minimize the size of the largest
    partition.


    Args:
        items (np.ndarray): list of numbers (i.e. weights) to split
            between paritions.
        num_parts (int): number of partitions

    Returns:
        List[np.ndarray]:
            A list for each parition that contains the index of the items
            assigned to it.

    References:
        https://en.wikipedia.org/wiki/Multiway_number_partitioning
        https://en.wikipedia.org/wiki/Balanced_number_partitioning

    Example:
        >>> from geowatch.utils.util_kwarray import *  # NOQA
        >>> items = np.array([1, 3, 29, 22, 4, 5, 9])
        >>> num_parts = 3
        >>> bin_assignments = balanced_number_partitioning(items, num_parts)
        >>> import kwarray
        >>> groups = kwarray.apply_grouping(items, bin_assignments)
        >>> bin_weights = [g.sum() for g in groups]
    """
    item_weights = np.asanyarray(items)
    sortx = np.argsort(item_weights)[::-1]

    bin_assignments = [[] for _ in range(num_parts)]
    bin_sums = np.zeros(num_parts)

    for item_index in sortx:
        # Assign item to the smallest bin
        item_weight = item_weights[item_index]
        bin_index = bin_sums.argmin()
        bin_assignments[bin_index].append(item_index)
        bin_sums[bin_index] += item_weight

    bin_assignments = [np.array(p, dtype=int) for p in bin_assignments]
    return bin_assignments


def torch_array_equal(data1, data2, equal_nan=False) -> bool:
    """
    Example:
        >>> # xdoctest: +REQUIRES(module:torch)
        >>> import torch
        >>> data1 = torch.rand(5, 5)
        >>> data2 = data1 + 1
        >>> result1 = torch_array_equal(data1, data2)
        >>> result3 = torch_array_equal(data1, data1)
        >>> assert result1 is False
        >>> assert result3 is True

    Example:
        >>> # xdoctest: +REQUIRES(module:torch)
        >>> import torch
        >>> data1 = torch.rand(5, 5)
        >>> data1[0] = np.nan
        >>> data2 = data1
        >>> result1 = torch_array_equal(data1, data2)
        >>> result3 = torch_array_equal(data1, data2, equal_nan=True)
        >>> assert result1 is False
        >>> assert result3 is True
    """
    # TODO: just use
    # return kwarray.ArrayAPI.coerce('torch').array_equal(data1, data2, equal_nan)
    import torch
    if equal_nan:
        val_flags = torch.eq(data1, data2)
        nan_flags = (data1.isnan() & data2.isnan())
        flags = val_flags | nan_flags
        return bool(flags.all())
    else:
        if _TORCH_LT_2_1_0:
            return torch.equal(data1, data2)
        else:
            # Torch 2.1 introduced a bug so we need an alternate
            # implementation.
            # References:
            #     https://github.com/pytorch/pytorch/issues/111251
            return bool(torch.eq(data1, data2).all())


def combine_mean_stds(means, stds, nums=None, axis=None, keepdims=False,
                       bessel=True):
    r"""
    Args:
        means (array): means[i] is the mean of the ith entry to combine

        stds (array): stds[i] is the std of the ith entry to combine

        nums (array | None):
            nums[i] is the number of samples in the ith entry to combine.
            if None, assumes sample sizes are infinite.

        axis (int | Tuple[int] | None):
            axis to combine the statistics over

        keepdims (bool):
            if True return arrays with the same number of dimensions they were
            given in.

        bessel (int):
            Set to 1 to enables bessel correction to unbias the combined std
            estimate.  Only disable if you have the true population means, or
            you think you know what you are doing.

    References:
        https://stats.stackexchange.com/questions/55999/is-it-possible-to-find-the-combined-standard-deviation

    SeeAlso:
        development kwarray has a similar hidden function in util_averages.
        Might expose later.

    Example:
        >>> means = np.stack([np.array([1.2, 3.2, 4.1])] * 100, axis=0)
        >>> stds = np.stack([np.array([4.2, 0.2, 2.1])] * 100, axis=0)
        >>> nums = np.stack([np.array([10, 100, 10])] * 100, axis=0)
        >>> cm1, cs1, _ = combine_mean_stds(means, stds, nums, axis=None)
        >>> print('combo_mean = {}'.format(ub.urepr(cm1, nl=1)))
        >>> print('combo_std  = {}'.format(ub.urepr(cs1, nl=1)))
        >>> means = np.stack([np.array([1.2, 3.2, 4.1])] * 1, axis=0)
        >>> stds = np.stack([np.array([4.2, 0.2, 2.1])] * 1, axis=0)
        >>> nums = np.stack([np.array([10, 100, 10])] * 1, axis=0)
        >>> cm2, cs2, _ = combine_mean_stds(means, stds, nums, axis=None)
        >>> print('combo_mean = {}'.format(ub.urepr(cm2, nl=1)))
        >>> print('combo_std  = {}'.format(ub.urepr(cs2, nl=1)))
        >>> means = np.stack([np.array([1.2, 3.2, 4.1])] * 5, axis=0)
        >>> stds = np.stack([np.array([4.2, 0.2, 2.1])] * 5, axis=0)
        >>> nums = np.stack([np.array([10, 100, 10])] * 5, axis=0)
        >>> cm3, cs3, combo_num = combine_mean_stds(means, stds, nums, axis=1)
        >>> print('combo_mean = {}'.format(ub.urepr(cm3, nl=1)))
        >>> print('combo_std  = {}'.format(ub.urepr(cs3, nl=1)))
        >>> assert np.allclose(cm1, cm2) and np.allclose(cm2,  cm3)
        >>> assert not np.allclose(cs1, cs2)
        >>> assert np.allclose(cs2, cs3)

    Example:
        >>> from geowatch.utils.util_kwarray import *  # NOQA
        >>> means = np.random.rand(2, 3, 5, 7)
        >>> stds = np.random.rand(2, 3, 5, 7)
        >>> nums = (np.random.rand(2, 3, 5, 7) * 10) + 1
        >>> cm, cs, cn = combine_mean_stds(means, stds, nums, axis=1, keepdims=1)
        >>> print('cs = {}'.format(ub.urepr(cs, nl=1)))
        >>> assert cm.shape == cs.shape == cn.shape
        ...
        >>> print(f'cm.shape={cm.shape}')
        >>> cm, cs, cn = combine_mean_stds(means, stds, nums, axis=(0, 2), keepdims=1)
        >>> assert cm.shape == cs.shape == cn.shape
        >>> print(f'cm.shape={cm.shape}')
        >>> cm, cs, cn = combine_mean_stds(means, stds, nums, axis=(1, 3), keepdims=1)
        >>> assert cm.shape == cs.shape == cn.shape
        >>> print(f'cm.shape={cm.shape}')
        >>> cm, cs, cn = combine_mean_stds(means, stds, nums, axis=None)
        >>> assert cm.shape == cs.shape == cn.shape
        >>> print(f'cm.shape={cm.shape}')
        cm.shape=(2, 1, 5, 7)
        cm.shape=(1, 3, 1, 7)
        cm.shape=(2, 1, 5, 1)
        cm.shape=()
    """
    if nums is None:
        # Assume the limit as nums -> infinite
        combo_num = None
        combo_mean = np.average(means, weights=None, axis=axis)
        combo_mean = _postprocess_keepdims(means, combo_mean, axis)
        numer_p1 = stds.sum(axis=axis, keepdims=1)
        numer_p2 = (((means - combo_mean) ** 2)).sum(axis=axis, keepdims=1)
        numer = numer_p1 + numer_p2
        denom = len(stds)
        # if denom == 0:
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', 'invalid value encountered', category=RuntimeWarning)
            combo_std = np.sqrt(numer / denom)
        # else:
        #     combo_std = np.full_like(numer, fill_value=np.nan)
    else:
        combo_num = nums.sum(axis=axis, keepdims=1)
        weights = nums / combo_num
        combo_mean = np.average(means, weights=weights, axis=axis)
        combo_mean = _postprocess_keepdims(means, combo_mean, axis)
        numer_p1 = (np.maximum(nums - bessel, 0) * stds).sum(axis=axis, keepdims=1)
        numer_p2 = (nums * ((means - combo_mean) ** 2)).sum(axis=axis, keepdims=1)
        numer = numer_p1 + numer_p2
        denom = np.maximum(combo_num - bessel, 0)
        # if denom == 0:
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', 'invalid value encountered', category=RuntimeWarning)
            combo_std = np.sqrt(numer / denom)
        # else:
        #     combo_std = np.full_like(numer, fill_value=np.nan)

    if not keepdims:
        indexer = _no_keepdim_indexer(combo_mean, axis)
        combo_mean = combo_mean[indexer]
        combo_std = combo_std[indexer]
        if combo_num is not None:
            combo_num = combo_num[indexer]

    return combo_mean, combo_std, combo_num


def _no_keepdim_indexer(result, axis):
    """
    Computes an indexer to postprocess a result with keepdims=True
    that will modify the result as if keepdims=False
    """
    if axis is None:
        indexer = [0] * len(result.shape)
    else:
        indexer = [slice(None)] * len(result.shape)
        if isinstance(axis, (list, tuple)):
            for a in axis:
                indexer[a] = 0
        else:
            indexer[axis] = 0
    indexer = tuple(indexer)
    return indexer


def _postprocess_keepdims(original, result, axis):
    """
    Can update the result of a function that does not support keepdims to look
    as if keepdims was supported.
    """
    # Newer versions of numpy have keepdims on more functions
    if axis is not None:
        expander = [slice(None)] * len(original.shape)
        if isinstance(axis, (list, tuple)):
            for a in axis:
                expander[a] = None
        else:
            expander[axis] = None
        result = result[tuple(expander)]
    else:
        expander = [None] * len(original.shape)
        result = np.array(result)[tuple(expander)]
    return result


def apply_robust_normalizer(normalizer, imdata, imdata_valid, mask, dtype, copy=True):
    """
        data = [self.dataset[idx] for idx in possibly_batched_index]
      File "/home/joncrall/code/watch/geowatch/tasks/fusion/datamodules/kwcoco_dataset.py", line 1004, in __getitem__
        return self.getitem(index)
      File "/home/joncrall/code/watch/geowatch/tasks/fusion/datamodules/kwcoco_dataset.py", line 1375, in getitem
        imdata_normalized = apply_robust_normalizer(
      File "/home/joncrall/code/watch/geowatch/tasks/fusion/datamodules/kwcoco_dataset.py", line 2513, in apply_robust_normalizer
        imdata_valid_normalized = kwarray.normalize(
      File "/home/joncrall/code/kwarray/kwarray/util_numpy.py", line 760, in normalize
        old_min = np.nanmin(float_out)
      File "<__array_function__ internals>", line 5, in nanmin
      File "/home/joncrall/.pyenv/versions/3.10.5/envs/pyenv3.10.5/lib/python3.10/site-packages/numpy/lib/nanfunctions.py", line 319, in nanmin
        res = np.fmin.reduce(a, axis=axis, out=out, **kwargs)
    """
    import kwarray
    if normalizer['type'] is None:
        imdata_normalized = imdata.astype(dtype, copy=copy)
    elif normalizer['type'] == 'normalize':
        # Note: we are using kwarray normalize, the one in kwimage is deprecated
        arr = imdata_valid.astype(dtype, copy=copy)
        imdata_valid_normalized = kwarray.normalize(
            arr, mode=normalizer['mode'],
            beta=normalizer['beta'], alpha=normalizer['alpha'],
        )
        if mask is None:
            imdata_normalized = imdata_valid_normalized
        else:
            imdata_normalized = imdata.copy() if copy else imdata
            imdata_normalized[mask] = imdata_valid_normalized
    else:
        raise KeyError(normalizer['type'])
    return imdata_normalized


@functools.cache
def biased_1d_weights(upweight_time, num_frames):
    """
    import kwplot
    plt = kwplot.autoplt()

    kwplot.figure()
    import sys, ubelt
    sys.path.append(ubelt.expandpath('~/code/watch'))
    from geowatch.tasks.fusion.datamodules.kwcoco_dataset import *  # NOQA

    kwplot.figure(fnum=1, doclf=1)
    num_frames = 5
    values = biased_1d_weights(0.5, num_frames)
    plt.plot(values)
    values = biased_1d_weights(0.1, num_frames)
    plt.plot(values)
    values = biased_1d_weights(0.0, num_frames)
    plt.plot(values)
    values = biased_1d_weights(0.9, num_frames)
    plt.plot(values)
    values = biased_1d_weights(1.0, num_frames)
    plt.plot(values)
    """
    # from kwarray.distributions import TruncNormal
    from scipy.stats import norm
    import kwimage
    # from kwarray.distributions import TruncNormal
    sigma = kwimage.im_cv2._auto_kernel_sigma(kernel=((num_frames, 1)))[1][0]
    mean = upweight_time * (num_frames - 1) + 0.5
    # rv = TruncNormal(mean=mean, std=sigma, low=0.0, high=num_frames).rv
    rv = norm(mean, sigma)
    locs = np.arange(num_frames) + 0.5
    values = rv.pdf(locs)
    return values


def argsort_threshold(arr, threshold=None, num_top=None, objective='maximize'):
    """
    Find all indexes over a threshold, but always return at least the
    `num_top`, and potentially more.

    Args:
        arr (ndarray): array of scores

        threshold (float):
            return indexes that are better than this threshold.

        num_top (int):
            always return at least this number of "best" indexes.

        objective (str):
            if maximize, filters things above the threshold, otherwise filters
            below the threshold.

    Returns:
        ndarray: top indexes

    Example:
        >>> from geowatch.utils.util_kwarray import *  # NOQA
        >>> arr = np.array([0.3, .2, 0.1, 0.15, 0.11, 0.15, 0.2, 0.6, 0.32])
        >>> argsort_threshold(arr, threshold=0.5, num_top=0)
        array([7])
        >>> argsort_threshold(arr, threshold=0.5, num_top=3)
        array([7, 8, 0])
        >>> argsort_threshold(arr, threshold=0.0, num_top=3)
    """
    # Find the "best" indices and their scores
    ascending_sortx = arr.argsort()
    # Mark any index "better" than the score threshold
    if objective == 'maximize':
        sortx = ascending_sortx[::-1]
        sorted_arr = arr[sortx]
        flags = sorted_arr > threshold
    elif objective == 'minimize':
        sortx = ascending_sortx
        sorted_arr = arr[sortx]
        flags = sorted_arr < threshold
    else:
        raise KeyError(objective)

    if num_top is not None:
        # Always return at least `num_top`
        flags[0:num_top] = True

        fallback_thresh = sorted_arr[num_top - 1]
        threshold = min(fallback_thresh, threshold)

    top_inds = sortx[flags]
    return top_inds


from geowatch.utils.remedian import Remedian  # NOQA


"""
Defines the :class:`SlidingWindow` and :class:`Sticher` classes.

The :class:`SlidingWindow` generates a grid of slices over an
:func:`numpy.ndarray`, which can then be used to compute on subsets of the
data. The :class:`Stitcher` can then take these results and recombine them into
a final result that matches the larger array.
"""


class SlidingWindow(ub.NiceRepr):
    """
    Slide a window of a certain shape over an array with a larger shape.

    This can be used for iterating over a grid of sub-regions of 2d-images,
    3d-volumes, or any n-dimensional array.

    Yields slices of shape `window` that can be used to index into an array
    with shape `shape` via numpy / torch fancy indexing. This allows for fast
    fast iteration over subregions of a larger image. Because we generate a
    grid-basis using only shapes, the larger image does not need to be in
    memory as long as its width/height/depth/etc...

    Args:
        shape (Tuple[int, ...]): shape of source array to slide across.

        window (Tuple[int, ...]): shape of window that will be slid over the
            larger image.

        overlap (float, default=0): a number between 0 and 1 indicating the
            fraction of overlap that parts will have. Specifying this is
            mutually exclusive with `stride`.  Must be `0 <= overlap < 1`.

        stride (int, default=None): the number of cells (pixels) moved on each
            step of the window. Mutually exclusive with overlap.

        keepbound (bool, default=False): if True, a non-uniform stride will be
            taken to ensure that the right / bottom of the image is returned as
            a slice if needed. Such a slice will not obey the overlap
            constraints.  (Defaults to False)

        allow_overshoot (bool, default=False): if False, we will raise an
            error if the window doesn't slide perfectly over the input shape.

    Attributes:
        basis_shape - shape of the grid corresponding to the number of strides
            the sliding window will take.
        basis_slices - slices that will be taken in every dimension

    Yields:
        Tuple[slice, ...]: slices used for numpy indexing, the number of slices
            in the tuple

    Note:
        For each dimension, we generate a basis (which defines a grid), and we
        slide over that basis.

    TODO:
        - [ ] have an option that is allowed to go outside of the window bounds
              on the right and bottom when the slider overshoots.

    Example:
        >>> shape = (10, 10)
        >>> window = (5, 5)
        >>> self = SlidingWindow(shape, window)
        >>> for i, index in enumerate(self):
        >>>     print('i={}, index={}'.format(i, index))
        i=0, index=(slice(0, 5, None), slice(0, 5, None))
        i=1, index=(slice(0, 5, None), slice(5, 10, None))
        i=2, index=(slice(5, 10, None), slice(0, 5, None))
        i=3, index=(slice(5, 10, None), slice(5, 10, None))

    Example:
        >>> shape = (16, 16)
        >>> window = (4, 4)
        >>> self = SlidingWindow(shape, window, overlap=(.5, .25))
        >>> print('self.stride = {!r}'.format(self.stride))
        self.stride = [2, 3]
        >>> list(ub.chunks(self.grid, 5))
        [[(0, 0), (0, 1), (0, 2), (0, 3), (0, 4)],
         [(1, 0), (1, 1), (1, 2), (1, 3), (1, 4)],
         [(2, 0), (2, 1), (2, 2), (2, 3), (2, 4)],
         [(3, 0), (3, 1), (3, 2), (3, 3), (3, 4)],
         [(4, 0), (4, 1), (4, 2), (4, 3), (4, 4)],
         [(5, 0), (5, 1), (5, 2), (5, 3), (5, 4)],
         [(6, 0), (6, 1), (6, 2), (6, 3), (6, 4)]]

    Example:
        >>> # Test shapes that dont fit
        >>> # When the window is bigger than the shape, the left-aligned slices
        >>> # are returend.
        >>> self = SlidingWindow((3, 3), (12, 12), allow_overshoot=True, keepbound=True)
        >>> print(list(self))
        [(slice(0, 12, None), slice(0, 12, None))]
        >>> print(list(SlidingWindow((3, 3), None, allow_overshoot=True, keepbound=True)))
        [(slice(0, 3, None), slice(0, 3, None))]
        >>> print(list(SlidingWindow((3, 3), (None, 2), allow_overshoot=True, keepbound=True)))
        [(slice(0, 3, None), slice(0, 2, None)), (slice(0, 3, None), slice(1, 3, None))]
    """
    def __init__(self, shape, window, overlap=None, stride=None,
                 keepbound=False, allow_overshoot=False):

        stride, overlap, window = self._compute_stride(
            overlap, stride, shape, window)

        stide_kw = [dict(margin=d, stop=D, step=s, keepbound=keepbound,
                         check=not keepbound and not allow_overshoot)
                      for d, D, s in zip(window, shape, stride)]

        undershot_shape = []
        overshoots = []
        for kw in stide_kw:
            final_pos = (kw['stop'] - kw['margin'])
            n_steps = final_pos // kw['step']
            overshoot = final_pos % kw['step']
            undershot_shape.append(n_steps + 1)
            overshoots.append(overshoot)

        self._final_step = overshoots

        if not allow_overshoot and any(overshoots):
            raise ValueError('overshoot={} stide_kw={}'.format(overshoots,
                                                               stide_kw))

        # make a slice generator for each dimension
        self.stride = stride
        self.overlap = overlap

        self.window = window
        self.input_shape = shape

        # The undershot basis shape, only contains indices that correspond
        # perfectly to the input. It may crop a bit of the ends.  If this is
        # equal to basis_shape, then the self perfectly fits the input.
        self.undershot_shape = undershot_shape

        # NOTE: if we have overshot, then basis shape will not perfectly
        # align to the original image. This shape will be a bit bigger.
        self.basis_slices = [list(_slices1d(**kw)) for kw in stide_kw]
        self.basis_shape = [len(b) for b in self.basis_slices]
        self.n_total = np.prod(self.basis_shape)

    def __nice__(self):
        return 'bshape={}, shape={}, window={}, stride={}'.format(
            tuple(self.basis_shape),
            tuple(self.input_shape),
            self.window,
            tuple(self.stride)
        )

    def _compute_stride(self, overlap, stride, shape, window):
        """
        Ensures that stride hasoverlap the correct shape.  If stride is not
        provided, compute stride from desired overlap.
        """
        if window is None:
            window = shape

        if isinstance(stride, np.ndarray):
            stride = tuple(stride)

        # TODO: some auto overlap?

        if isinstance(overlap, np.ndarray):
            overlap = tuple(overlap)

        if len(window) != len(shape):
            raise ValueError('incompatible dims: {} {}'.format(len(window),
                                                               len(shape)))

        if any(d is None for d in window):
            window = [D if d is None else d for d, D in zip(window, shape)]

        if overlap is None and stride is None:
            overlap = 0

        if not (overlap is None) ^ (stride is None):
            raise ValueError('specify overlap({}) XOR stride ({})'.format(
                overlap, stride))
        if stride is None:
            if not isinstance(overlap, (list, tuple)):
                overlap = [overlap] * len(window)
            if any(frac < 0 or frac >= 1 for frac in overlap):
                raise ValueError((
                    'part overlap was {}, but fractional overlaps must be '
                    'in the range [0, 1)').format(overlap))
            stride = [int(round(d - d * frac))
                      for frac, d in zip(overlap, window)]
        else:
            if not isinstance(stride, (list, tuple)):
                stride = [stride] * len(window)
        # Recompute fractional overlap after integer stride is computed
        overlap = [(d - s) / d for s, d in zip(stride, window)]

        assert len(stride) == len(shape), 'incompatible dims'

        if not all(stride):
            raise ValueError(
                'Step must be positive everywhere. Got={}'.format(stride))
        return stride, overlap, window

    def __len__(self):
        return self.n_total

    def _iter_basis_frac(self):
        for slices in self:
            frac = [sl.start / D for sl, D in zip(slices, self.source.shape)]
            yield frac

    def __iter__(self):
        for slices in it.product(*self.basis_slices):
            yield slices

    def __getitem__(self, index):
        """
        Get a specific item by its flat (raveled) index

        Example:
            >>> from kwarray.util_slider import *  # NOQA
            >>> window = (10, 10)
            >>> shape = (20, 20)
            >>> self = SlidingWindow(shape, window, stride=5)
            >>> itered_items = list(self)
            >>> assert len(itered_items) == len(self)
            >>> indexed_items = [self[i] for i in range(len(self))]
            >>> assert itered_items[0] == self[0]
            >>> assert itered_items[-1] == self[-1]
            >>> assert itered_items == indexed_items
        """
        if index < 0:
            index = len(self) + index
        # Find the nd location in the grid
        basis_idx = np.unravel_index(index, self.basis_shape)
        # Take the slice for each of the n dimensions
        slices = tuple([bdim[i]
                        for bdim, i in zip(self.basis_slices, basis_idx)])
        return slices

    @property
    def grid(self):
        """
        Generate indices into the "basis" slice for each dimension.
        This enumerates the nd indices of the grid.

        Yields:
            Tuple[int, ...]
        """
        # Generates basis for "sliding window" slices to break a large image
        # into smaller pieces. Use it.product to slide across the coordinates.
        basis_indices = map(range, self.basis_shape)
        for basis_idxs in it.product(*basis_indices):
            yield basis_idxs

    @property
    def slices(self):
        """
        Generate slices for each window (equivalent to iter(self))

        Example:
            >>> shape = (220, 220)
            >>> window = (10, 10)
            >>> self = SlidingWindow(shape, window, stride=5)
            >>> list(self)[41:45]
            [(slice(0, 10, None), slice(205, 215, None)),
             (slice(0, 10, None), slice(210, 220, None)),
             (slice(5, 15, None), slice(0, 10, None)),
             (slice(5, 15, None), slice(5, 15, None))]
            >>> print('self.overlap = {!r}'.format(self.overlap))
            self.overlap = [0.5, 0.5]
        """
        return iter(self)

    @property
    def centers(self):
        """
        Generate centers of each window

        Yields:
            Tuple[float, ...]: the center coordinate of the slice

        Example:
            >>> shape = (4, 4)
            >>> window = (3, 3)
            >>> self = SlidingWindow(shape, window, stride=1)
            >>> list(zip(self.centers, self.slices))
            [((1.0, 1.0), (slice(0, 3, None), slice(0, 3, None))),
             ((1.0, 2.0), (slice(0, 3, None), slice(1, 4, None))),
             ((2.0, 1.0), (slice(1, 4, None), slice(0, 3, None))),
             ((2.0, 2.0), (slice(1, 4, None), slice(1, 4, None)))]
            >>> shape = (3, 3)
            >>> window = (2, 2)
            >>> self = SlidingWindow(shape, window, stride=1)
            >>> list(zip(self.centers, self.slices))
            [((0.5, 0.5), (slice(0, 2, None), slice(0, 2, None))),
             ((0.5, 1.5), (slice(0, 2, None), slice(1, 3, None))),
             ((1.5, 0.5), (slice(1, 3, None), slice(0, 2, None))),
             ((1.5, 1.5), (slice(1, 3, None), slice(1, 3, None)))]
        """
        for slices in self:
            center = tuple(sl_.start + (sl_.stop - sl_.start - 1) / 2
                           for sl_ in slices)
            yield center


class Stitcher(ub.NiceRepr):
    """
    From kwarray: v0.6.19

    Stitches multiple possibly overlapping slices into a larger array.

    This is used to invert the SlidingWindow.  For semenatic segmentation the
    patches are probability chips. Overlapping chips are averaged together.

    SeeAlso:
        :class:`kwarray.RunningStats` - similarly performs running means, but
           can also track other statistics.

    Example:
        >>> # Build a high resolution image and slice it into chips
        >>> highres = np.random.rand(5, 200, 200).astype(np.float32)
        >>> target_shape = (1, 50, 50)
        >>> slider = SlidingWindow(highres.shape, target_shape, overlap=(0, .5, .5))
        >>> # Show how Sticher can be used to reconstruct the original image
        >>> stitcher = Stitcher(slider.input_shape)
        >>> for sl in list(slider):
        ...     chip = highres[sl]
        ...     stitcher.add(sl, chip)
        >>> assert stitcher.weights.max() == 4, 'some parts should be processed 4 times'
        >>> recon = stitcher.finalize()

    Example:
        >>> # Demo stitching 3 patterns where one has nans
        >>> pat1 = np.full((32, 32), fill_value=0.2)
        >>> pat2 = np.full((32, 32), fill_value=0.4)
        >>> pat3 = np.full((32, 32), fill_value=0.8)
        >>> pat1[:, 16:] = 0.6
        >>> pat2[16:, :] = np.nan
        >>> # Test with nan_policy=omit
        >>> stitcher = Stitcher(shape=(32, 64), nan_policy='omit')
        >>> stitcher[0:32, 0:32](pat1)
        >>> stitcher[0:32, 16:48](pat2)
        >>> stitcher[0:32, 33:64](pat3[:, 1:])
        >>> final1 = stitcher.finalize()
        >>> # Test without nan_policy=propogate
        >>> stitcher = Stitcher(shape=(32, 64), nan_policy='propogate')
        >>> stitcher[0:32, 0:32](pat1)
        >>> stitcher[0:32, 16:48](pat2)
        >>> stitcher[0:32, 33:64](pat3[:, 1:])
        >>> final2 = stitcher.finalize()
        >>> # Checks
        >>> assert np.isnan(final1).sum() == 16, 'only should contain nan where no data was stiched'
        >>> assert np.isnan(final2).sum() == 512, 'should contain nan wherever a nan was stitched'
        >>> # xdoctest: +REQUIRES(--show)
        >>> # xdoctest: +REQUIRES(module:kwplot)
        >>> import kwplot
        >>> import kwimage
        >>> kwplot.autompl()
        >>> kwplot.imshow(pat1, title='pat1', pnum=(3, 3, 1))
        >>> kwplot.imshow(kwimage.nodata_checkerboard(pat2, square_shape=1), title='pat2 (has nans)', pnum=(3, 3, 2))
        >>> kwplot.imshow(pat3, title='pat3', pnum=(3, 3, 3))
        >>> kwplot.imshow(kwimage.nodata_checkerboard(final1, square_shape=1), title='stitched (nan_policy=omit)', pnum=(3, 1, 2))
        >>> kwplot.imshow(kwimage.nodata_checkerboard(final2, square_shape=1), title='stitched (nan_policy=propogate)', pnum=(3, 1, 3))
    """
    def __init__(self, shape, device='numpy', dtype='float32',
                 nan_policy='propogate', memmap=False):
        """
        Args:
            shape (tuple): dimensions of the large image that will be created from
                the smaller pixels or patches.

            device (str | int | torch.device):
                default is 'numpy', but if given as a torch device, then
                underlying operations will be done with torch tensors instead.

            dtype (str):
                the datatype to use in the underlying accumulator.

            nan_policy (str):
                if omit, check for nans and convert any to zero weight items in
                stitching.

            memmap (bool | PathLike):
                if truthy, the stitcher will use a memory map. If this
                pathlike, then we use this as the directory for the memmap.
                If True, a temp directory is used.
        """
        self.nan_policy = nan_policy
        self.shape = shape
        self.device = device
        self.paths = None

        use_memmap = bool(memmap)
        if use_memmap:
            import uuid
            uuid = uuid.uuid4()
            if isinstance(memmap, (str, os.PathLike)):
                memmap_dpath = ub.Path(memmap)
            else:
                from tempfile import mkdtemp
                memmap_dpath = ub.Path(mkdtemp())
                memmap_sums_fpath = memmap_dpath / f'{uuid}-sums.npy'
                memmap_weights_fpath = memmap_dpath / f'{uuid}-weights.npy'
                self.paths = {
                    'sums': memmap_sums_fpath,
                    'weights': memmap_weights_fpath,
                }
        else:
            memmap_dpath = None
            memmap_sums_fpath = None
            memmap_weights_fpath = None

        if device == 'numpy':

            if use_memmap:
                # Seems to always init to zero
                self.sums = np.memmap(memmap_sums_fpath, dtype=dtype, mode='w+', shape=shape)
                self.weights = np.memmap(memmap_weights_fpath, dtype=dtype, mode='w+', shape=shape)
            else:
                self.sums = np.zeros(shape, dtype=dtype)
                self.weights = np.zeros(shape, dtype=dtype)

            # self.sumview = self.sums.ravel()
            # self.weightview = self.weights.ravel()
        else:
            import torch
            if memmap:
                raise NotImplementedError('cannot do torch memmaping')
            else:
                self.sums = torch.zeros(shape, device=device)
                self.weights = torch.zeros(shape, device=device)
            # self.sumview = self.sums.view(-1)
            # self.weightview = self.weights.view(-1)
        if self.nan_policy in {'omit', 'raise'}:
            if device == 'numpy':
                self._isnan = np.isnan
                self._any = np.any
            else:
                self._isnan = torch.isnan
                self._any = torch.any
        elif self.nan_policy != 'propogate':
            raise ValueError(self.nan_policy)

    def __nice__(self):
        return str(self.sums.shape)

    def add(self, indices, patch, weight=None):
        """
        Incorporate a new (possibly overlapping) patch or pixel using a
        weighted sum.

        Args:
            indices (slice | tuple | None):
                typically a Tuple[slice] of pixels or a single pixel, but this
                can be any numpy fancy index.

            patch (ndarray): data to patch into the bigger image.

            weight (float | ndarray): weight of this patch (default to 1.0)
        """
        if self.nan_policy == 'omit':
            mask = self._isnan(patch)
            if self._any(mask):
                # Detect nans, set weight and value to zero
                if weight is None:
                    weight = (~mask).astype(self.weights.dtype)
                else:
                    weight = weight * (~mask).astype(self.weights.dtype)
                patch = patch.copy()
                patch[mask] = 0
        elif self.nan_policy == 'raise':
            mask = self._isnan(patch)
            if self._any(mask):
                raise ValueError('nan_policy is raise')

        if weight is None:
            self.sums[indices] += patch
            self.weights[indices] += 1.0
        else:
            self.sums[indices] += (patch * weight)
            self.weights[indices] += weight

    def __getitem__(self, indices):
        """
        Convinience function to use slice notation directly.
        """
        from functools import partial
        return partial(self.add, indices)

    def average(self):
        """
        Averages out contributions from overlapping adds using weighted average

        Returns:
            ndarray: out - the stitched image
        """
        out = self.sums / self.weights
        return out

    def finalize(self, indices=None):
        """
        Averages out contributions from overlapping adds

        Args:
            indices (None | slice | tuple): if None, finalize the entire
                block, otherwise only finalize a subregion.

        Returns:
            ndarray: final - the stitched image
        """
        if indices is None:
            final = self.sums / self.weights
        else:
            final = self.sums[indices] / self.weights[indices]
        return final


def _slices1d(margin, stop, step=None, start=0, keepbound=False, check=True):
    """
    Helper to generates slices in a single dimension.

    Args:

        margin (int): the length of the slice (window)

        stop (int): the length of the image dimension

        step (int, default=None): the length of each step / distance between
            slices

        start (int, default=0): starting point (in most cases set this to 0)

        keepbound (bool): if True, a non-uniform step will be taken to ensure
            that the right / bottom of the image is returned as a slice if
            needed. Such a slice will not obey the overlap constraints.
            (Defaults to False)

        check (bool): if True an error will be raised if the window does not
            cover the entire extent from start to stop, even if keepbound is
            True.

    Yields:
        slice : slice in one dimension of size (margin)

    Example:
        >>> stop, margin, step = 2000, 360, 360
        >>> keepbound = True
        >>> strides = list(_slices1d(margin, stop, step, keepbound, check=False))
        >>> assert all([(s.stop - s.start) == margin for s in strides])

    Example:
        >>> stop, margin, step = 200, 46, 7
        >>> keepbound = True
        >>> strides = list(_slices1d(margin, stop, step, keepbound=False, check=True))
        >>> starts = np.array([s.start for s in strides])
        >>> stops = np.array([s.stop for s in strides])
        >>> widths = stops - starts
        >>> assert np.all(np.diff(starts) == step)
        >>> assert np.all(widths == margin)

    Example:
        >>> import pytest
        >>> stop, margin, step = 200, 36, 7
        >>> with pytest.raises(ValueError):
        ...     list(_slices1d(margin, stop, step))
    """
    if step is None:
        step = margin

    if check:
        # see how far off the end we would fall if we didnt check bounds
        perfect_final_pos = (stop - start - margin)
        overshoot = perfect_final_pos % step
        if overshoot > 0:
            raise ValueError(
                ('margin={} and step={} overshoot endpoint={} '
                 'by {} units when starting from={}').format(
                     margin, step, stop, overshoot, start))
    pos = start
    # probably could be more efficient with numpy here
    while True:
        endpos = pos + margin
        yield slice(pos, endpos)
        # Stop once we reached the end
        if endpos == stop:
            break
        pos += step
        if pos + margin > stop:
            if keepbound:
                # Ensure the boundary is always used even if steps
                # would overshoot Could do some other strategy here
                pos = stop - margin
                if pos < 0:
                    break
            else:
                break
