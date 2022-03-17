"""
Functions that may eventually be moved to kwarray
"""
import numpy as np
import math
import ubelt as ub

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
        >>> from watch.utils.util_kwarray import *  # NOQA
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
        >>> from watch.utils.util_kwarray import *  # NOQA
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
