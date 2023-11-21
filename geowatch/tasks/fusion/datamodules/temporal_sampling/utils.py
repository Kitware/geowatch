import numpy as np
import ubelt as ub


def guess_missing_unixtimes(unixtimes, assume_delta=86400):
    """
    Hueristic solution to fill in missing time values via interpolation /
    extrapolation.

    To succesfully interpolate nan values must be between two non-nan values.
    In all other cases we have to make an assumption about the timedelta
    between frames, which can be specified and is one day by default.

    Args:
        unixtimes (ndarray): numpy array of numeric unix timestamps that may
            contain nan values.

        assume_delta (float):
            The fallback delta between timesteps when surrounding context is
            unavailable.  Defaults to 86400 seconds - i.e. 1 day.

    Returns:
        ndarray:
            The same array, but nan values are filled with interpolated or
            extrapolated values.

    Example:
        >>> from geowatch.tasks.fusion.datamodules.temporal_sampling.utils import *  # NOQA
        >>> import ubelt as ub
        >>> cases = [
        >>>     np.array([np.nan, np.nan, np.nan, np.nan, np.nan]),
        >>>     np.array([np.nan, 20, 30, np.nan, np.nan]),
        >>>     np.array([0, np.nan, np.nan, np.nan, 10]),
        >>>     np.array([np.nan, np.nan, 9001, np.nan, np.nan]),
        >>>     np.array([1, 2, 3, 4, 5]),
        >>>     np.array([1, 2, np.nan, 4, 5]),
        >>> ]
        >>> for case_ in cases:
        >>>     unixtimes = case_
        >>>     print('case_ = {}'.format(ub.urepr(case_, nl=1)))
        >>>     guess = guess_missing_unixtimes(unixtimes)
        >>>     print('guess = {}'.format(ub.urepr(guess, nl=1)))
    """
    missing_date = np.isnan(unixtimes)
    num_missing = missing_date.sum()
    missing_any_dates = num_missing > 0

    if missing_any_dates:
        num_have = len(unixtimes) - num_missing
        have_exactly1 = num_have == 1
        have_atleast2 = num_have > 1

        if have_exactly1:
            # Only 1 date still means we have to extrapolate.
            have_idx = np.where(~missing_date)[0][0]
            have_time = unixtimes[have_idx]
            unixtimes = unixtimes.copy()
            unixtimes[0] = have_time - (have_idx * assume_delta)
            unixtimes[-1] = have_time + (len(unixtimes) - (have_idx + 1)) * assume_delta
            missing_date = np.isnan(unixtimes)
            have_atleast2 = True

        if have_atleast2:
            from scipy import interpolate
            frame_idxs = np.arange(len(unixtimes))
            miss_idxs = frame_idxs[missing_date]
            have_idxs = frame_idxs[~missing_date]
            have_values = unixtimes[have_idxs]
            interp = interpolate.interp1d(have_idxs, have_values, fill_value='extrapolate')
            interp_vals = interp(miss_idxs)
            unixtimes = unixtimes.copy()
            unixtimes[miss_idxs] = interp_vals
        else:
            # No information.
            unixtimes = np.linspace(0, len(unixtimes) * assume_delta, len(unixtimes))
    return unixtimes


def coerce_time_kernel(pattern):
    """
    Obtain a time kernel from user input

    Args:
        pattern (str | Iterable[str | Number]):
            A string code or a iterable of time coercable time deltas in
            ascending order. A pattern code is a ',' separated string of
            coercable time deltas.

    Returns:
        ndarray : ascending timedelta offsets in seconds

    Example:
        >>> from geowatch.tasks.fusion.datamodules.temporal_sampling.utils import *  # NOQA
        >>> import ubelt as ub
        >>> valid_patterns = [
        >>>     '-1y,-30d,-1d,0,1d,30d,1y',
        >>>     '-60s,0s,60s',
        >>>     '-1d,-60s,20s,60s,1d',
        >>>     '1,1,1,1,1',
        >>>     '(1,1,1,1,1)',
        >>> ]
        >>> for pattern in valid_patterns:
        >>>     kernel = coerce_time_kernel(pattern)
        >>>     assert np.all(kernel == coerce_time_kernel(pattern)), 'should be idempotent'
        >>>     print('kernel = {}'.format(ub.urepr(kernel.tolist(), nl=0)))
        kernel = [-31536000.0, -2592000.0, -86400.0, 0.0, 86400.0, 2592000.0, 31536000.0]
        kernel = [-60.0, 0.0, 60.0]
        kernel = [-86400.0, -60.0, 20.0, 60.0, 86400.0]
        kernel = [1.0, 1.0, 1.0, 1.0, 1.0]
        kernel = [1.0, 1.0, 1.0, 1.0, 1.0]
        >>> import pytest
        >>> invalid_patterns = [
        >>>     '3s,2s,1s'
        >>>     '-10,5,3,-2,0,1'
        >>> ]
        >>> for pattern in invalid_patterns:
        >>>     with pytest.raises(ValueError):
        >>>         kernel = coerce_time_kernel(pattern)
        >>> with pytest.raises(TypeError):
        >>>     coerce_time_kernel(3.14)
    """
    from geowatch.tasks.fusion.datamodules.temporal_sampling.time_kernel_grammar import parse_multi_time_kernel
    from kwutil.util_time import coerce_timedelta
    if isinstance(pattern, str):
        if '(' in pattern:
            multi_kernel = parse_multi_time_kernel(pattern)
            assert len(multi_kernel) == 1, 'only expecting a single kernel here'
            kernel_deltas = multi_kernel[0]
        else:
            kernel_deltas = pattern.split(',')
    elif ub.iterable(pattern):
        kernel_deltas = pattern
    else:
        print(f'SINGLE-coerce error: pattern={pattern}')
        raise TypeError(type(pattern))
    parsed = [coerce_timedelta(d) for d in kernel_deltas]
    kernel = np.array([v.total_seconds() for v in parsed])
    diffs = np.diff(kernel)
    if not np.all(diffs >= 0):
        raise ValueError('Inputs must be in ascending order')
    return kernel


def coerce_multi_time_kernel(pattern):
    """
    Obtain a list of time kernels from user input.

    Example:
        >>> from geowatch.tasks.fusion.datamodules.temporal_sampling.utils import *  # NOQA
        >>> import ubelt as ub
        >>> valid_patterns = [
        >>>     '(-1d,0,1d),(1,2)',
        >>>     '1,1,1',
        >>>     '(1,1,1,1,1)',
        >>>     ['-1,0,+1', '0'],
        >>> ]
        >>> for pattern in valid_patterns:
        >>>     multi_kernel = coerce_multi_time_kernel(pattern)
        >>>     recon = coerce_multi_time_kernel(multi_kernel)
        >>>     a = [r.tolist() for r in recon]
        >>>     b = [r.tolist() for r in multi_kernel]
        >>>     assert a == b, 'should be idempotent'
        >>>     print('multi_kernel = {}'.format(ub.urepr(multi_kernel, nl=1)))
        multi_kernel = [
            np.array([-86400.,      0.,  86400.], dtype=np.float64),
            np.array([1., 2.], dtype=np.float64),
        ]
        multi_kernel = [
            np.array([1., 1., 1.], dtype=np.float64),
        ]
        multi_kernel = [
            np.array([1., 1., 1., 1., 1.], dtype=np.float64),
        ]
        multi_kernel = [
            np.array([-1.,  0.,  1.], dtype=np.float64),
            np.array([0.], dtype=np.float64),
        ]

    Example:
        >>> from geowatch.tasks.fusion.datamodules.temporal_sampling.utils import *  # NOQA
        >>> import ubelt as ub
        >>> pattern = ('-3y', '-2.5y', '-2y', '-1.5y', '-1y', 0, '1y', '1.5y', '2y', '2.5y', '3y')
        >>> multi_kernel = coerce_multi_time_kernel(pattern)
        >>> print('multi_kernel = {}'.format(ub.urepr(multi_kernel, nl=2)))

        >>> # FIXME: Bug ambigous case
        >>> pattern = ('-3y', '-2.5y', '-2y', '-1.5y', '-1y', '0', '1y', '1.5y', '2y', '2.5y', '3y')
        >>> multi_kernel = coerce_multi_time_kernel(pattern)
        >>> print('multi_kernel = {}'.format(ub.urepr(multi_kernel, nl=2)))
    """
    if pattern is None:
        return [None]
    from geowatch.tasks.fusion.datamodules.temporal_sampling.time_kernel_grammar import parse_multi_time_kernel
    from kwutil.util_time import coerce_timedelta
    if isinstance(pattern, str):
        multi_kernel = parse_multi_time_kernel(pattern)
    elif ub.iterable(pattern):
        if len(pattern) == 0:
            multi_kernel = []
        else:
            first = pattern[0]
            if ub.iterable(first):
                # Assume we are given a list of pre-parsed kernels
                multi_kernel = pattern
                # multi_kernel = [coerce_timedelta(d) for d in pattern]
            else:
                # We might be given a list of multiple kernels
                try:
                    multi_kernel = [coerce_time_kernel(p) for p in pattern]
                    WORKAROUND_SINGLE_KERNELS = 1
                    if WORKAROUND_SINGLE_KERNELS:
                        # This is not a total fix for the ambiguous case, but
                        # it will get us by for now. If all sub-kernels have a
                        # length of 1, the user probably didnt mean that.
                        if all(len(k) == 1 for k in multi_kernel):
                            raise Exception('ambiguity workaround')
                except Exception:
                    # Or we might be given just a single kernel.
                    multi_kernel = [[coerce_timedelta(d) for d in pattern]]
                    ...

    else:
        print(f'MULTI-coerce error: pattern={pattern}')
        raise TypeError(type(pattern))
    return multi_kernel
