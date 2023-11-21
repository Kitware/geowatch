"""
Functions that may eventually be moved to kwarray
"""
import numpy as np
import functools
import math
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

    Benchmark:
        # Our method is faster than standard in-line implementations.

        import timerit
        ti = timerit.Timerit(100, bestof=10, verbose=2, unit='ms')
        arr = kwimage.grab_test_image('lowcontrast', dsize=(512, 512))

        print('--- uint8 ---')
        arr = ensure_float01(arr)
        out = arr.copy()
        for timer in ti.reset('naive1-float'):
            with timer:
                (arr - arr.min()) / (arr.max() - arr.min())

        import timerit
        for timer in ti.reset('simple-float'):
            with timer:
                max_ = arr.max()
                min_ = arr.min()
                result = (arr - min_) / (max_ - min_)

        for timer in ti.reset('normalize-float'):
            with timer:
                normalize(arr)

        for timer in ti.reset('normalize-float-inplace'):
            with timer:
                normalize(arr, out=out)

        print('--- float ---')
        arr = ensure_uint255(arr)
        out = arr.copy()
        for timer in ti.reset('naive1-uint8'):
            with timer:
                (arr - arr.min()) / (arr.max() - arr.min())

        import timerit
        for timer in ti.reset('simple-uint8'):
            with timer:
                max_ = arr.max()
                min_ = arr.min()
                result = (arr - min_) / (max_ - min_)

        for timer in ti.reset('normalize-uint8'):
            with timer:
                normalize(arr)

        for timer in ti.reset('normalize-uint8-inplace'):
            with timer:
                normalize(arr, out=out)

    Ignore:
        globals().update(xdev.get_func_kwargs(normalize))
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


class Remedian:
    """
    Remedian object for a robust averaging method for large data sets.
    Implementation of the Remedian algorithm, see [1]_ [2]_ [3]_ for
    references. This algorithm is used to approximate the median of several
    data chunks if these data chunks cannot (or should not) be loaded into
    memory at once. See "Notes" section for further information.
    Parameters
    ----------
    obs_size : ndarray
        The shape of each data chunk (=observation) to be fed into the Remedian
        object.
    n_obs : int
        The number of observations to be stored within each array.
        If `n_obs` >= `t`, Remedian will equal the median. The smaller this
        parameter, the fewer data have to be loaded into memory at once, but
        the less accurate the approximation of the median will be.
    t : int
        The total number of observations from which a median should be
        approximated.
    Attributes
    ----------
    obs_count : int
        Counter of number of observations that have already been given
        to the Remedian object.
    remedian : None | ndarray, shape(obs_size)
        The calculated remedian of the same shape as the input data.
        Will be None until all observations `n_obs` have been fed into
        the object using the add_obs method.
    Notes
    -----
    Given a data chunk of size `obs_size`, and `t` data chunks overall, the
    Remedian class sets up a number `k_arrs` of arrays of length `n_obs`.
    The median of the `t` data chunks of size `obs_size` is then approximated
    as follows: One data chunk after another is fed into the `n_obs` positions
    of the first array. When the first array is full, its median is calculated
    and stored in the first position of the second array. After this, the first
    array is re-used to fill the second position of the second array, etc.
    When the second array is full, the median of its values is stored in the
    first position of the third array, and so on.
    The final "Remedian" is the median of the last array, after all `t` data
    chunks have been fed into the object.
    In other words, given an n-dimensional array, the Remedian class
    approximates the median of this array across the ith dimension and you have
    to break up your n-dimensional array into `t` n-1-dimensional arrays that
    are given to Remedian one after another.
    References
    ----------
    .. [1] P.J. Rousseeuw, G.W. Bassett Jr., "The remedian: A robust averaging
       method for large data sets", Journal of the American Statistical
       Association, vol. 85 (1990), pp. 97-104
    .. [2] M. Chao, G. Lin, "The asymptotic distributions of the remedians",
       Journal of Statistical Planning and Inference, vol. 37 (1993), pp. 1-11
    .. [3] Domenico Cantone, Micha Hofri, "Further analysis of the remedian
       algorithm", Theoretical Computer Science, vol. 495 (2013), pp. 1-16

       https://stackoverflow.com/questions/62352762/is-it-possible-to-compute-median-while-data-is-still-being-generated-python-onl
       https://wis.kuleuven.be/stat/robust/papers/publications-1990/rousseeuwbassett-remedian-jasa-1990.pdf
       https://github.com/sappelhoff/remedian/blob/master/remedian/remedian.py


    TODO:
        - [ ] Can we make this work so we don't need t?

    Examples:
        >>> import sys, ubelt
        >>> from geowatch.utils.util_kwarray import Remedian
        >>> #
        >>> shape = (7, 5)
        >>> self = Remedian(shape, n_obs=11, t=101)
        >>> #
        >>> obs0 = np.arange(7 * 5).reshape(7, 5).astype(np.float32)
        >>> raw_obs = []
        >>> for _ in range(self.t):
        >>>     obs = obs0.copy()
        >>>     obs[2:, 3:] = np.random.rand(5, 2)
        >>>     raw_obs.append(obs.copy())
        >>>     self.add_obs(obs)
        >>> approx_median = self.remedian
        >>> real_median = np.median(np.stack(raw_obs, -1), axis=-1)
        >>> delta = approx_median - real_median
    """

    def __init__(self, obs_size, n_obs, t):
        """Initialize the Remedian object.
        See class docstring for more thorough information.
        Parameters
        ----------
        obs_size : ndarray
            Size of the observations. Must be (1,) for scalars.
        n_obs : int
            Observations per array.
        t : int
            Number of total observations.
        """
        if n_obs <= 1:
            raise ValueError('`n_obs` of <= 1 does not make sense.')

        self.obs_size = list(obs_size)
        self.n_obs = n_obs
        self.t = t

        # Calculate the number of arrays needed and their sizes
        self.k_arrs = self._calc_k_arrs()
        self.k_arr_sizes = self._calc_k_arr_sizes()

        # Initialize the arrays
        self.arrs = [np.zeros(self.obs_size + [s]) for s in self.k_arr_sizes]

        # counter for observations within each array
        self.obs_idx_counter = [0 for arr in range(self.k_arrs)]

        # Modulos of observations to assign to correct array later
        self.modulos = [self.n_obs**i for i in range(1, 1 + self.k_arrs)]

        # Counter of received observations
        self.obs_count = 0

        # Set the median value to None until we have it
        self.remedian = None

    def _calc_k_arrs(self):
        """Calculate number of arrays to accommodate the observations."""
        tmp = self.n_obs
        k_arrs = 1
        while tmp <= self.t:
            tmp *= self.n_obs
            k_arrs += 1
        return k_arrs

    def _calc_k_arr_sizes(self):
        """Calculate the size of each array to accomodate the observations."""
        k_arr_sizes = [self.n_obs for i in range(self.k_arrs)]
        k_arr_sizes[-1] = int(np.ceil(self.t / (self.n_obs**(self.k_arrs - 1))))
        return k_arr_sizes

    def add_obs(self, obs):
        """Add an observation to the Remedian.
        Parameters
        ----------
        obs : ndarray, shape(obs_size)
            A single data observation.
        """
        # We only work if:
        # ... we get an observation of correct size
        # ... we have not yet received all observations already
        if list(obs.shape) != self.obs_size:
            raise ValueError('Expected observation of size {} but received: '
                             '{}'.format(self.obs_size, list(obs.shape)))
        if self.obs_count > (self.t - 1):
            raise RuntimeError('Already collected {} observations out of t={} '
                               'The remedian is {}'.format(self.obs_count,
                                                           self.t,
                                                           self.remedian))

        # We accept a new observation
        self.obs_count += 1

        # Add the data to the first array
        # and increment the counter for the next data
        obs_idx = self.obs_idx_counter[0]
        self.arrs[0][..., obs_idx] = obs
        self.obs_idx_counter[0] += 1

        # We can notice whenever an array is full using modulo operations
        # on the observation counter self.obs_count.
        # When an array is full, calculate the median and put the result
        # into the next array. Then reset the counters and start filling
        # previous arrays again
        for arr_i, mod in enumerate(self.modulos):
            if self.obs_count % mod == 0:
                data = self.arrs[arr_i]
                m_tmp = np.median(data, axis=-1, overwrite_input=True)
                self.arrs[arr_i + 1][..., self.obs_idx_counter[arr_i + 1]] = m_tmp
                self.obs_idx_counter[arr_i + 1] += 1
                self.obs_idx_counter[arr_i] = 0

        # If all observations have been received,
        # calculate the median of the last array.
        # This is the robust approximation of the median
        if self.obs_count == self.t:
            self.remedian = np.median(self.arrs[-1], axis=-1,
                                      overwrite_input=True)


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
