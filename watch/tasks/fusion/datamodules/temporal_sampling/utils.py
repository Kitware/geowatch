import numpy as np


def guess_missing_unixtimes(unixtimes):
    """
    Hueristic solution to fill in missing time values via interpolation /
    extrapolation.

    Example:
        >>> from watch.tasks.fusion.datamodules.temporal_sampling.utils import *  # NOQA
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

        assume_delta = 60 * 60 * 24  # 1 day
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
