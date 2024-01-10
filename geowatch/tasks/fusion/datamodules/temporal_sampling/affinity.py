import datetime as datetime_mod
import kwarray
import kwimage
import math
import numpy as np
import ubelt as ub
from geowatch.utils import util_kwarray
from kwutil.util_time import coerce_timedelta
from datetime import datetime as datetime_cls  # NOQA
from .exceptions import TimeSampleError
from .utils import guess_missing_unixtimes
from .utils import coerce_time_kernel

try:
    from xdev import profile
except ImportError:
    profile = ub.identity


@profile
def affinity_sample(affinity, size, include_indices=None, exclude_indices=None,
                    allow_fewer=False, update_rule='pairwise', gamma=1,
                    deterministic=False, time_kernel=None, unixtimes=None,
                    error_level=2, rng=None, return_info=False, jit=False):
    """
    Randomly select ``size`` timesteps from a larger pool based on ``affinity``.

    Given an NxN affinity matrix between frames and an initial set of indices
    to include, chooses a sample of other frames to complete the sample.  Each
    row and column in the affinity matrix represent a "selectable" timestamp.
    Given an initial set of ``include_indices`` that indicate which timesteps
    must be included in the sample. An iterative process is used to select
    remaining indices such that ``size`` timesteps are returned. In each
    iteration we choose the "next" timestep based on a probability distribution
    derived from (1) the affinity matrix (2) the currently included set of
    indexes and (3) the update rule.

    Args:
        affinity (ndarray):
            pairwise affinity matrix

        size (int):
            Number of sample indices to return

        include_indices (List[int]):
            Indices that must be included in the sample

        exclude_indices (List[int]):
            Indices that cannot be included in the sample

        allow_fewer (bool):
            if True, we will allow fewer than the requested "size" samples to
            be returned.

        update_rule (str):
            Modifies how the affinity matrix is used to create the
            probability distribution for the "next" frame that will be
            selected.
            a "+" separated string of codes which can contain:
                * pairwise - if included, each newly chosen sample will
                    modulate the initial "main" affinity with it's own
                    affinity.  Otherwise, only the affinity of the initially
                    included rows are considered.
                * distribute - if included, every step of weight updates will
                    downweight samples temporally close to the most recently
                    selected sample.

        gamma (float, default=1.0):
            Exponent that modulates the probability distribution. Lower gamma
            will "flatten" the probability curve. At gamma=0, all frames will
            be equally likely regardless of affinity. As gamma -> inf, the rule
            becomes more likely to sample the maximum probability at each
            timestep. In the limit this becomes equivalent to
            ``deterministic=True``.

        deterministic (bool):
            if True, on each step we choose the next timestamp with maximum
            probability. Otherwise, we randomly choose a timestep, but with
            probability according to the current distribution.

        error_level (int):
            Error and fallback behavior if perfect sampling is not possible.
            error level 0:
                might return excluded, duplicate indexes, or 0-affinity indexes
                if everything else is exhausted.
            error level 1:
                duplicate indexes will raise an error
            error level 2:
                duplicate and excluded indexes will raise an error
            error level 3:
                duplicate, excluded, and 0-affinity indexes will raise an error

        rng (Coercible[RandomState]):
            random state for reproducible sampling

        return_info (bool):
            If True, includes a dictionary of information that details the
            internal steps the algorithm took.

        jit (bool):
            NotImplemented - do not use

        time_kernel (ndarray):
            if specified, the sample will attempt to conform to this time
            kernel.

    Returns:
        ndarray | Tuple[ndarray, Dict]:
            The ``chosen`` indexes for the sample, or if return_info is True,
            then returns a tuple of ``chosen`` and the info dictionary.

    Raises:
        TimeSampleError : if sampling is impossible

    Possible Related Work:
        * Random Stratified Sampling Affinity Matrix
        * A quasi-random sampling approach to image retrieval

    Example:
        >>> from geowatch.tasks.fusion.datamodules.temporal_sampling import *  # NOQA
        >>> from geowatch.tasks.fusion.datamodules.temporal_sampling.affinity import *  # NOQA
        >>> low = datetime_cls.now().timestamp()
        >>> high = low + datetime_mod.timedelta(days=365 * 5).total_seconds()
        >>> rng = kwarray.ensure_rng(0)
        >>> unixtimes = np.array(sorted(rng.randint(low, high, 113)), dtype=float)
        >>> #
        >>> affinity = soft_frame_affinity(unixtimes, version=2, time_span='1d')['final']
        >>> include_indices = [5]
        >>> size = 5
        >>> chosen, info = affinity_sample(affinity, size, include_indices, update_rule='pairwise',
        >>>                                return_info=True, deterministic=True)
        >>> # xdoctest: +REQUIRES(--show)
        >>> import kwplot
        >>> from geowatch.tasks.fusion.datamodules.temporal_sampling.plots import show_affinity_sample_process
        >>> sns = kwplot.autosns()
        >>> plt = kwplot.autoplt()
        >>> show_affinity_sample_process(chosen, info)

    Example:
        >>> from geowatch.tasks.fusion.datamodules.temporal_sampling import *  # NOQA
        >>> low = datetime_cls.now().timestamp()
        >>> high = low + datetime_mod.timedelta(days=365 * 5).total_seconds()
        >>> rng = kwarray.ensure_rng(0)
        >>> unixtimes = np.array(sorted(rng.randint(low, high, 5)), dtype=float)
        >>> self = TimeWindowSampler(unixtimes, sensors=None, time_window=4,
        >>>     affinity_type='soft2', time_span='0.3y',
        >>>     update_rule='distribute+pairwise', allow_fewer=False)
        >>> self.deterministic = False
        >>> import pytest
        >>> with pytest.raises(IndexError):
        >>>     self.sample(0, exclude=[1, 2, 4], error_level=3)
        >>> with pytest.raises(IndexError):
        >>>     self.sample(0, exclude=[1, 2, 4], error_level=2)
        >>> self.sample(0, exclude=[1, 2, 4], error_level=1)
        >>> # xdoctest: +REQUIRES(--show)
        >>> import kwplot
        >>> kwplot.autompl()
        >>> chosen, info = self.show_procedure(idx=0, fnum=10, exclude=[1, 2, 4])
        >>> print('info = {}'.format(ub.urepr(info, nl=4)))

    Ignore:
        >>> from geowatch.tasks.fusion.datamodules.temporal_sampling import *  # NOQA
        >>> low = datetime_cls.now().timestamp()
        >>> high = low + datetime_mod.timedelta(days=365 * 5).total_seconds()
        >>> rng = kwarray.ensure_rng(0)
        >>> unixtimes = np.array(sorted(rng.randint(low, high, 113)), dtype=float)
        >>> affinity = soft_frame_affinity(unixtimes)['final']
        >>> include_indices = [5]
        >>> size = 20
        >>> xdev.profile_now(affinity_sample)(affinity, size, include_indices)

    Example:
        >>> # xdoctest: +REQUIRES(env:SMART_DATA_DVC_DPATH)
        >>> from geowatch.tasks.fusion.datamodules.temporal_sampling import *  # NOQA
        >>> from geowatch.tasks.fusion.datamodules.temporal_sampling.utils import coerce_time_kernel
        >>> import kwarray
        >>> import geowatch
        >>> data_dvc_dpath = geowatch.find_dvc_dpath(tags='phase2_data', hardware='auto')
        >>> coco_fpath = data_dvc_dpath / 'Drop6/imgonly-KR_R001.kwcoco.json'
        >>> dset = geowatch.coerce_kwcoco(coco_fpath)
        >>> vidid = dset.dataset['videos'][0]['id']
        >>> time_kernel_code = '-3m,-1w,0,3m,1y'
        >>> self = TimeWindowSampler.from_coco_video(
        >>>     dset, vidid,
        >>>     time_window=5,
        >>>     time_kernel=time_kernel_code,
        >>>     affinity_type='soft3',
        >>>     update_rule='')
        >>> self.deterministic = False
        >>> self.show_affinity()
        >>> include_indices = [len(self.unixtimes) // 2]
        >>> exclude_indices = []
        >>> affinity = self.affinity
        >>> size = self.time_window
        >>> deterministic = self.deterministic
        >>> update_rule = self.update_rule
        >>> unixtimes = self.unixtimes
        >>> gamma = self.gamma
        >>> time_kernel = self.time_kernel
        >>> rng = kwarray.ensure_rng(None)
        >>> deterministic = True
        >>> return_info = True
        >>> error_level = 2
        >>> chosen, info = affinity_sample(
        >>>     affinity=affinity,
        >>>     size=size,
        >>>     include_indices=include_indices,
        >>>     exclude_indices=exclude_indices,
        >>>     update_rule=update_rule,
        >>>     gamma=gamma,
        >>>     deterministic=deterministic,
        >>>     error_level=error_level,
        >>>     rng=rng,
        >>>     return_info=return_info,
        >>>     time_kernel=time_kernel,
        >>>     unixtimes=unixtimes,
        >>> )
        >>> # xdoctest: +REQUIRES(--show)
        >>> import kwplot
        >>> kwplot.autompl()
        >>> info['title_suffix'] = chr(10) + time_kernel_code
        >>> from geowatch.tasks.fusion.datamodules.temporal_sampling.plots import show_affinity_sample_process
        >>> show_affinity_sample_process(chosen, info, fnum=1)
    """
    rng = kwarray.ensure_rng(rng)

    if include_indices is None:
        include_indices = []
    if exclude_indices is None:
        exclude_indices = []

    chosen = list(include_indices)

    if len(chosen) == 0:
        # Need to find a seed frame
        avail = list(set(range(len(affinity))) - set(exclude_indices))
        if len(avail) == 0:
            raise Exception('nothing is available')
        avail_idx = rng.randint(0, len(avail))
        chosen = [avail_idx]

    col_idxs = np.arange(0, affinity.shape[1])

    if time_kernel is not None:
        # TODO: let the user pass in the primary idx
        primary_idx = chosen[0]

        primary_unixtime = unixtimes[primary_idx]
        relative_unixtimes = unixtimes - primary_unixtime

        kernel_masks, kernel_attrs = make_soft_mask(time_kernel, relative_unixtimes)

        kernel_distance = np.abs(relative_unixtimes[:, None] - time_kernel[None:, ])
        # Partition the pool based on which part of the kernel they most satisfy
        kernel_idxs = np.arange(len(time_kernel))
        kernel_groups = kernel_distance.argmin(axis=1)
        satisfied_kernel_idxs = kernel_groups[chosen]
        unsatisfied_kernel_idxs = np.setdiff1d(kernel_idxs, satisfied_kernel_idxs)
        _, kernel_idx_to_groupxs = kwarray.group_indices(kernel_groups)

    update_rules = {r for r in update_rule.split('+') if r}
    config = ub.dict_subset({'pairwise': True, 'distribute': True}, update_rules)
    do_pairwise = config.get('pairwise', False)
    do_distribute = config.get('distribute', False)

    if len(chosen) == 1:
        initial_weights = affinity[chosen[0]]
    else:
        initial_weights = affinity[chosen].prod(axis=0)

    initial_weights[exclude_indices] = 0
    update_weights = 1

    if do_pairwise:
        update_weights = initial_weights * update_weights

    if do_distribute:
        update_weights *= (np.abs(col_idxs - np.array(chosen)[:, None]) / len(col_idxs)).min(axis=0)

    current_weights = initial_weights * update_weights
    current_weights[chosen] = 0

    num_sample = size - len(chosen)

    if not allow_fewer:
        num_available = len(affinity)
        if size > num_available:
            raise TimeSampleError(
                '{size=} is greater than {num_available=}. '
                'Set allow_fewer=True if returning less than the requested '
                'number of samples is ok.')

    if jit:
        raise NotImplementedError('A cython version of this would be useful')
        # cython_mod = cython_aff_samp_mod()
        # return cython_mod.cython_affinity_sample(affinity, num_sample, current_weights, chosen, rng)

    if time_kernel is not None:
        if len(unsatisfied_kernel_idxs) < num_sample:
            raise AssertionError(f'{len(unsatisfied_kernel_idxs)}, {num_sample}')
        if len(unsatisfied_kernel_idxs) == 0:
            raise AssertionError(f'{len(unsatisfied_kernel_idxs)}, {num_sample}')
        kernel_idx = unsatisfied_kernel_idxs[0]
        next_mask = kernel_masks[kernel_idx]
        unsatisfied_kernel_idxs = unsatisfied_kernel_idxs[1:]
    else:
        next_mask = None
        # next_ideal_idx = None

    current_mask = initial_mask = next_mask

    if return_info:
        denom = current_weights.sum()
        if denom == 0:
            denom = 1
        initial_probs = current_weights / denom
        info = {
            'steps': [],

            'initial_weights': initial_weights.copy(),
            'initial_update_weights': update_weights.copy() if hasattr(update_weights, 'copy') else update_weights,
            'initial_mask': initial_mask,
            'initial_probs': initial_probs,

            'initial_chosen': chosen.copy(),

            'include_indices': include_indices,

            'affinity': affinity,
            'unixtimes': unixtimes,
            'time_kernel': time_kernel,
        }

    # Errors will be accumulated in this list if we encounter them and the user
    # requested debug info.
    errors = []

    try:
        for _ in range(num_sample):
            # Choose the next image based on combined sample affinity

            if return_info:
                errors = []

            total_weight = current_weights.sum()

            # If we zeroed out all of the probabilities try two things before
            # punting and setting everything to uniform.
            if total_weight == 0:
                current_weights = _handle_degenerate_weights(
                    affinity, size, chosen, exclude_indices, errors, error_level,
                    return_info, rng)

            if current_mask is not None:
                masked_current_weights = current_weights * current_mask

                total_weight = masked_current_weights.sum()
                if total_weight == 0:
                    masked_current_weights = _handle_degenerate_weights(
                        affinity, size, chosen, exclude_indices, errors,
                        error_level, return_info, rng)
            else:
                masked_current_weights = current_weights

            if deterministic:
                next_idx = masked_current_weights.argmax()
            else:
                cumprobs = (masked_current_weights ** gamma).cumsum()
                dart = rng.rand() * cumprobs[-1]
                next_idx = np.searchsorted(cumprobs, dart)

            update_weights = 1

            if do_pairwise:
                if next_idx < affinity.shape[0]:
                    update_weights = affinity[next_idx] * update_weights

            if do_distribute:
                update_weights = (np.abs(col_idxs - next_idx) / len(col_idxs)) * update_weights

            chosen.append(next_idx)

            if current_mask is not None:
                # Build the next mask
                if len(unsatisfied_kernel_idxs):
                    kernel_idx = unsatisfied_kernel_idxs[0]
                    next_mask = kernel_masks[kernel_idx]
                    unsatisfied_kernel_idxs = unsatisfied_kernel_idxs[1:]
                else:
                    next_mask = None

            if return_info:
                if total_weight == 0:
                    probs = masked_current_weights.copy()
                else:
                    probs = masked_current_weights / total_weight
                probs = masked_current_weights
                info['steps'].append({
                    'probs': probs,
                    'next_idx': next_idx,
                    'update_weights': update_weights,
                    'next_mask': next_mask,
                    'errors': errors,
                })

            # Modify weights / mask to impact next sample
            current_weights = current_weights * update_weights
            current_mask = next_mask

            # Don't resample the same item
            current_weights[next_idx] = 0
    except TimeSampleError:
        if len(chosen) == 0:
            raise
        if not allow_fewer:
            raise

    chosen = sorted(chosen)
    if return_info:
        return chosen, info
    else:
        return chosen


def make_soft_mask(time_kernel, relative_unixtimes):
    """
    Assign probabilities to real observations based on an ideal time kernel

    Args:
        time_kernel (ndarray):
            A list of relative seconds in the time kernel. Each element in this
            list is referred to as a "kernel entry".

        relative_unixtimes (ndarray):
            A list of available unixtimes corresponding to real observations.
            These should be relative to an "ideal" center. I.e. the "main"
            observation the kernel is centered around should have a relative
            unixtime of zero.

    Returns:
        Tuple[List[ndarray], List[Dict]]:
            A tuple of (kernel_masks, kernel_attrs).  For each element in the
            time kernel there is a corresponding entry in the output
            kernel_masks and kernel_attrs list, with the former being a
            probability assigned to each observation for that particular kernel
            entry, and the latter is a dictionary of information about that
            kernel entry.

    Example:
        >>> # Generates the time kernel visualization
        >>> from geowatch.tasks.fusion.datamodules.temporal_sampling.affinity import *  # NOQA
        >>> time_kernel = coerce_time_kernel('-1H,-5M,0,5M,1H')
        >>> relative_unixtimes = coerce_time_kernel('-90M,-70M,-50M,0,1sec,10S,30M')
        >>> # relative_unixtimes = coerce_time_kernel('-90M,-70M,-50M,-20M,-10M,0,1sec,10S,30M,57M,87M')
        >>> kernel_masks, kernel_attrs = make_soft_mask(time_kernel, relative_unixtimes)
        >>> #
        >>> min_t = min(kattr['left'] for kattr in kernel_attrs)
        >>> max_t = max(kattr['right'] for kattr in kernel_attrs)
        >>> # xdoctest: +REQUIRES(--show)
        >>> import kwimage
        >>> import kwplot
        >>> plt = kwplot.autoplt()
        >>> kwplot.figure(fnum=1, doclf=1)
        >>> kernel_color = kwimage.Color.coerce('kitware_green').as01()
        >>> obs_color = kwimage.Color.coerce('kitware_blue').as01()
        >>> #
        >>> kwplot.figure(fnum=1, pnum=(2, 1, 1))
        >>> plt.plot(time_kernel, [0] * len(time_kernel), '-o', color=kernel_color, label='kernel')
        >>> #
        >>> for kattr in kernel_attrs:
        >>>     rv = kattr['rv']
        >>>     xs = np.linspace(min_t, max_t, 1000)
        >>>     ys = rv.pdf(xs)
        >>>     ys_norm = ys / ys.sum()
        >>>     plt.plot(xs, ys_norm)
        >>> #
        >>> ax = plt.gca()
        >>> ax.legend()
        >>> ax.set_xlabel('time')
        >>> ax.set_ylabel('ideal probability')
        >>> ax.set_title('ideal kernel')
        >>> #
        >>> kwplot.figure(fnum=1, pnum=(2, 1, 2))
        >>> plt.plot(relative_unixtimes, [0] * len(relative_unixtimes), '-o', color=obs_color, label='observation')
        >>> ax = plt.gca()
        >>> #
        >>> for kattr in kernel_attrs:
        >>>     rv = kattr['rv']
        >>>     xs = relative_unixtimes
        >>>     ys = rv.pdf(xs)
        >>>     ys_norm = ys / ys.sum()
        >>>     plt.plot(xs, ys_norm)
        >>> ax.legend()
        >>> ax.set_xlabel('time')
        >>> ax.set_ylabel('sample probability')
        >>> ax.set_title('discrete observations')
        >>> plt.subplots_adjust(top=0.9, hspace=.3)
        >>> kwplot.show_if_requested()

    Example:
        >>> from geowatch.tasks.fusion.datamodules.temporal_sampling.affinity import *  # NOQA
        >>> time_kernel = coerce_time_kernel('-1H,-5M,0,5M,1H')
        >>> relative_unixtimes = [np.nan] * 10
        >>> # relative_unixtimes = coerce_time_kernel('-90M,-70M,-50M,-20M,-10M,0,1sec,10S,30M,57M,87M')
        >>> kernel_masks, kernel_attrs = make_soft_mask(time_kernel, relative_unixtimes)

    """
    if len(time_kernel) == 1:
        raise Exception(f'Time kernel has a length of 1: time_kernel={time_kernel!r}')

    # Fix any possible missing values
    from geowatch.tasks.fusion.datamodules.temporal_sampling.utils import guess_missing_unixtimes
    relative_unixtimes = guess_missing_unixtimes(relative_unixtimes)

    first = time_kernel[0]
    second = time_kernel[1]
    penult = time_kernel[-2]
    last = time_kernel[-1]
    left_pad = second - first
    right_pad = last - penult

    padded_time_kernel = [first - left_pad] + list(time_kernel) + [last + right_pad]

    kernel_attrs = []

    from scipy.stats import norm
    for a, b, c in ub.iter_window(padded_time_kernel, 3):
        left_extent = b - a
        right_extent = c - b
        extent = min(left_extent, right_extent)
        mean = b
        std = extent / 3  # 3sigma
        rv = norm(mean, std)
        kernel_attrs.append({
            'left': a,
            'mid': b,
            'right': c,
            'extent': extent,
            'rv': rv,
        })

    kernel_masks = []
    import warnings
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', 'invalid value encountered', category=RuntimeWarning)

        for kattr in kernel_attrs:
            probs = kattr['rv'].pdf(relative_unixtimes)
            pmf = probs / probs.sum()
            kernel_masks.append(pmf)

    return kernel_masks, kernel_attrs


@profile
def _handle_degenerate_weights(affinity, size, chosen, exclude_indices, errors,
                               error_level, return_info, rng):
    """
    Called by :func:`affinity_sample` when the exact requested sampling is
    impossible. Depending on the error level this function either tries to
    recover or raises an error with debug info.
    """

    debug_parts = [
        f'{size=}',
        f'{affinity.shape=}',
        f'{len(chosen)=}',
        f'{len(exclude_indices)=}',
        f'{error_level=}',
    ]
    if error_level == 3:
        msg3 = '\n'.join(['all probability is exhausted.'] + debug_parts)
        raise TimeSampleError(msg3)

    current_weights = affinity[chosen[0]].copy()
    current_weights[chosen] = 0
    current_weights[exclude_indices] = 0

    total_weight = current_weights.sum()

    if return_info:
        errors.append('all indices were chosen, excluded, or had no affinity')

    if total_weight == 0:
        # Should really never get here in day-to-day, but just in case
        if error_level == 2:
            msg2 = '\n'.join(['all included probability is exhausted.'] + debug_parts)
            raise TimeSampleError(msg2)

        # Zero weight method: neighbors
        zero_weight_method = 'neighbors'
        if zero_weight_method == 'neighbors':
            if len(chosen) == 0:
                zero_weight_method = 'random'
            else:
                chosen_neighbor_idxs = np.hstack([np.array(chosen) + 1, np.array(chosen) - 1])
                chosen_neighbor_idxs = np.unique(np.clip(chosen_neighbor_idxs, 0, len(current_weights) - 1))
                ideal_idxs = np.setdiff1d(chosen_neighbor_idxs, chosen)
                if len(ideal_idxs) == 0:
                    ideal_idxs = chosen_neighbor_idxs
                current_weights[ideal_idxs] = 1
        elif zero_weight_method == 'random':
            current_weights[:] = rng.rand(len(current_weights))
        else:
            raise KeyError(zero_weight_method)

        current_weights[chosen] = 0
        total_weight = current_weights.sum()
        if return_info:
            errors.append('all indices were chosen, excluded')
        if total_weight == 0:

            if error_level == 1:
                debug_parts.append(f'{total_weight=}')
                msg1 = '\n'.join(['all chosen probability is exhausted.'] + debug_parts)
                raise TimeSampleError(msg1)

            if zero_weight_method == 'neighbors':
                current_weights[:] = rng.rand(len(current_weights))
            elif zero_weight_method == 'random':
                current_weights[:] = rng.rand(len(current_weights))
            else:
                raise KeyError(zero_weight_method)

            if return_info:
                errors.append('all indices were chosen, punting')

    if return_info:
        errors.append('\n'.join(debug_parts))

    return current_weights


@profile
def hard_time_sample_pattern(unixtimes, time_window, time_kernel=None, time_span=None):
    """
    Finds hard time sampling indexes

    Args:
        unixtimes (ndarray):
            list of unix timestamps indicating available temporal samples

        time_window (int):
            number of frames per sample

    References:
        https://docs.google.com/presentation/d/1GSOaY31cKNERQObl_L3vk0rGu6zU7YM_ZFLrdksHSC0/edit#slide=id.p

    Example:
        >>> low = datetime_cls.now().timestamp()
        >>> high = low + datetime_mod.timedelta(days=365 * 5).total_seconds()
        >>> rng = kwarray.ensure_rng(0)
        >>> base_unixtimes = np.array(sorted(rng.randint(low, high, 20)), dtype=float)
        >>> unixtimes = base_unixtimes.copy()
        >>> #unixtimes[rng.rand(*unixtimes.shape) < 0.1] = np.nan
        >>> time_window = 5
        >>> sample_idxs = hard_time_sample_pattern(unixtimes, time_window, time_span='2y')
        >>> name = 'demo-data'

        >>> #unixtimes[:] = np.nan
        >>> time_window = 5
        >>> sample_idxs = hard_time_sample_pattern(unixtimes, time_window, time_span='2y')
        >>> name = 'demo-data'

    Ignore:
        >>> # xdoctest: +REQUIRES(env:SMART_DATA_DVC_DPATH)
        >>> from geowatch.tasks.fusion.datamodules.temporal_sampling import *  # NOQA
        >>> import geowatch
        >>> from kwutil import util_time
        >>> data_dvc_dpath = geowatch.find_dvc_dpath(tags='phase2_data', hardware='auto')
        >>> coco_fpath = data_dvc_dpath / 'Drop6/imgonly-KR_R001.kwcoco.json'
        >>> dset = kwcoco.CocoDataset(coco_fpath)
        >>> video_ids = list(ub.sorted_vals(dset.index.vidid_to_gids, key=len).keys())
        >>> vidid = video_ids[0]
        >>> video = dset.index.videos[vidid]
        >>> name = (video['name'])
        >>> print('name = {!r}'.format(name))
        >>> images = dset.images(video_id=vidid)
        >>> datetimes = [util_time.coerce_datetime(date) for date in images.lookup('date_captured')]
        >>> unixtimes = np.array([dt.timestamp() for dt in datetimes])
        >>> time_window = 5
        >>> time_kernel = '-1y-3m,-1w,0,1w,3m,1y'
        >>> sample_idxs = hard_time_sample_pattern(unixtimes, time_window, time_kernel=time_kernel)
        >>> # xdoctest: +REQUIRES(--show)
        >>> from geowatch.tasks.fusion.datamodules.temporal_sampling.plots import plot_dense_sample_indices
        >>> from geowatch.tasks.fusion.datamodules.temporal_sampling.plots import plot_temporal_sample_indices
        >>> import kwplot
        >>> kwplot.autoplt()
        >>> kwplot.figure(fnum=1, doclf=1)
        >>> plot_dense_sample_indices(sample_idxs, unixtimes, title_suffix=f': {name}')
        >>> kwplot.figure(fnum=2, doclf=1)
        >>> plot_temporal_sample_indices(sample_idxs, unixtimes, title_suffix=f': {name}')

    Ignore:
        >>> import kwplot
        >>> import numpy as np
        >>> sns = kwplot.autosns()

        >>> # =====================
        >>> # Show Sample Pattern in heatmap
        >>> from geowatch.tasks.fusion.datamodules.temporal_sampling.plots import plot_dense_sample_indices
        >>> plot_dense_sample_indices(sample_idxs, unixtimes, title_suffix=f': {name}')

        >>> datetimes = np.array([datetime_mod.datetime.fromtimestamp(t) for t in unixtimes])
        >>> dates = np.array([datetime_mod.datetime.fromtimestamp(t).date() for t in unixtimes])
        >>> #
        >>> sample_pattern = kwarray.one_hot_embedding(sample_idxs, len(unixtimes), dim=1).sum(axis=2)
        >>> kwplot.imshow(sample_pattern)
        >>> import pandas as pd
        >>> df = pd.DataFrame(sample_pattern)
        >>> df.index.name = 'index'
        >>> #
        >>> df.columns = pd.to_datetime(datetimes).date
        >>> df.columns.name = 'date'
        >>> #
        >>> kwplot.figure(fnum=1, doclf=True)
        >>> ax = sns.heatmap(data=df)
        >>> ax.set_title(f'Sample Pattern wrt Available Observations: {name}')
        >>> ax.set_xlabel('Observation Index')
        >>> ax.set_ylabel('Sample Index')
        >>> #
        >>> #import matplotlib.dates as mdates
        >>> #ax.figure.autofmt_xdate()

        >>> # =====================
        >>> # Show Sample Pattern WRT to time
        >>> fig = kwplot.figure(fnum=2, doclf=True)
        >>> ax = fig.gca()
        >>> for t in datetimes:
        >>>     ax.plot([t, t], [0, len(sample_idxs) + 1], color='orange')
        >>> for sample_ypos, sample in enumerate(sample_idxs, start=1):
        >>>     ax.plot(datetimes[sample], [sample_ypos] * len(sample), '-x')
        >>> ax.set_title(f'Sample Pattern wrt Time Range: {name}')
        >>> ax.set_xlabel('Time')
        >>> ax.set_ylabel('Sample Index')
        >>> # import matplotlib.dates as mdates
        >>> # ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        >>> # ax.xaxis.set_major_locator(mdates.DayLocator(interval=1))
        >>> # ax.figure.autofmt_xdate()

        >>> # =====================
        >>> # Show Available Samples
        >>> import july
        >>> from july.utils import date_range
        >>> datetimes = [datetime_cls.fromtimestamp(t) for t in unixtimes]
        >>> grid_dates = date_range(
        >>>     datetimes[0].date().isoformat(),
        >>>     (datetimes[-1] + datetime_mod.timedelta(days=1)).date().isoformat()
        >>> )
        >>> grid_unixtime = np.array([
        >>>     datetime_cls.combine(d, datetime_cls.min.time()).timestamp()
        >>>     for d in grid_dates
        >>> ])
        >>> positions = np.searchsorted(grid_unixtime, unixtimes)
        >>> indicator = np.zeros_like(grid_unixtime)
        >>> indicator[positions] = 1
        >>> dates_unixtimes = [d for d in dates]
        >>> july.heatmap(grid_dates, indicator, title=f'Available Observations: {name}', cmap="github")
    """
    if time_span is not None and time_kernel is not None:
        raise ValueError('time_span and time_kernel are mutex')

    if isinstance(time_window, int):
        # TODO: formulate how to choose template delta for given window dims Or
        # pass in a delta
        if time_window == 1:
            template_deltas = np.array([
                datetime_mod.timedelta(days=0).total_seconds(),
            ])
        else:
            if time_span is not None:
                time_span = coerce_timedelta(time_span).total_seconds()
                min_time = -datetime_mod.timedelta(seconds=time_span).total_seconds()
                max_time = datetime_mod.timedelta(seconds=time_span).total_seconds()
                template_deltas = np.linspace(min_time, max_time, time_window).round().astype(int)
                # Always include a delta of 0
                template_deltas[np.abs(template_deltas).argmin()] = 0
            elif time_kernel is not None:
                time_kernel = coerce_time_kernel(time_kernel)
                template_deltas = time_kernel
            else:
                raise Exception('need time span or time kernel')
    else:
        raise NotImplementedError
        template_deltas = time_window

    unixtimes = guess_missing_unixtimes(unixtimes)

    rel_unixtimes = unixtimes - unixtimes[0]
    temporal_sampling = rel_unixtimes[:, None] + template_deltas[None, :]

    # Wraparound (this is a bit of a hack)
    hackit = 1
    if hackit:
        wraparound = 1
        last_time = rel_unixtimes[-1] + 1
        is_oob_left = temporal_sampling < 0
        is_oob_right = temporal_sampling >= last_time
        is_oob = (is_oob_right | is_oob_left)
        is_ib = ~is_oob

        tmp = temporal_sampling.copy()
        tmp[is_oob] = np.nan
        # filter warn
        max_ib = np.nanmax(tmp, axis=1)
        min_ib = np.nanmin(tmp, axis=1)

        row_oob_flag = is_oob.any(axis=1)

        # TODO: rewrite this with reasonable logic for fixing oob samples
        # This is horrible logic, I'd be ashamed, but it works.
        for rx in np.where(row_oob_flag)[0]:
            is_bad = is_oob[rx]
            mx = max_ib[rx]
            mn = min_ib[rx]
            row = temporal_sampling[rx].copy()
            valid_data = row[is_ib[rx]]
            if not len(valid_data):
                wraparound = 1
                temporal_sampling[rx, :] = 0.0
            else:
                step = (mx - mn) / len(valid_data)
                if step == 0:
                    step = np.diff(template_deltas[0:2])[0]

                if step <= 0:
                    wraparound = 1
                else:
                    avail_after = last_time - mx
                    avail_before = mn

                    if avail_after > 0:
                        avail_after_steps = avail_after / step
                    else:
                        avail_after_steps = 0

                    if avail_before > 0:
                        avail_before_steps = avail_before / step
                    else:
                        avail_before_steps = 0

                    need = is_bad.sum()
                    before_oob = is_oob_left[rx]
                    after_oob = is_oob_right[rx]

                    take_after = min(before_oob.sum(), int(avail_after_steps))
                    take_before = min(after_oob.sum(), int(avail_before_steps))

                    extra_after = np.linspace(mx, mx + take_after * step, take_after)
                    extra_before = np.linspace(mn - take_before * step, mn, take_before)

                    extra = np.hstack([extra_before, extra_after])

                    bad_idxs = np.where(is_bad)[0]
                    use = min(min(len(bad_idxs), need), len(extra))
                    row[bad_idxs[:use]] = extra[:use]
                    temporal_sampling[rx] = row

    wraparound = 1
    if wraparound:
        temporal_sampling = temporal_sampling % last_time

    losses = np.abs(temporal_sampling[:, :, None] - rel_unixtimes[None, None, :])
    losses[losses == 0] = -np.inf
    all_rows = []
    for loss_for_row in losses:
        # For each row find the closest available frames to the ideal
        # sample without duplicates.
        sample_idxs = np.array(kwarray.mincost_assignment(loss_for_row)[0]).T[1]
        all_rows.append(sorted(sample_idxs))

    sample_idxs = np.vstack(all_rows)
    return sample_idxs


@profile
def soft_frame_affinity(unixtimes, sensors=None, time_kernel=None,
                        time_span=None, version=1, heuristics='default'):
    """
    Produce a pairwise affinity weights between frames based on a dilated time
    heuristic.

    Example:
        >>> from geowatch.tasks.fusion.datamodules.temporal_sampling.affinity import *  # NOQA
        >>> low = datetime_mod.datetime.now().timestamp()
        >>> high = low + datetime_mod.timedelta(days=365 * 5).total_seconds()
        >>> rng = kwarray.ensure_rng(0)
        >>> base_unixtimes = np.array(sorted(rng.randint(low, high, 113)), dtype=float)

        >>> # Test no missing data case
        >>> unixtimes = base_unixtimes.copy()
        >>> allhave_weights = soft_frame_affinity(unixtimes, version=2)
        >>> #
        >>> # Test all missing data case
        >>> unixtimes = np.full_like(unixtimes, fill_value=np.nan)
        >>> allmiss_weights = soft_frame_affinity(unixtimes, version=2)
        >>> #
        >>> # Test partial missing data case
        >>> unixtimes = base_unixtimes.copy()
        >>> unixtimes[rng.rand(*unixtimes.shape) < 0.1] = np.nan
        >>> anymiss_weights_1 = soft_frame_affinity(unixtimes, version=2)
        >>> unixtimes = base_unixtimes.copy()
        >>> unixtimes[rng.rand(*unixtimes.shape) < 0.5] = np.nan
        >>> anymiss_weights_2 = soft_frame_affinity(unixtimes, version=2)
        >>> unixtimes = base_unixtimes.copy()
        >>> unixtimes[rng.rand(*unixtimes.shape) < 0.9] = np.nan
        >>> anymiss_weights_3 = soft_frame_affinity(unixtimes, version=2)

        >>> # xdoctest: +REQUIRES(--show)
        >>> import kwplot
        >>> kwplot.autoplt()
        >>> pnum_ = kwplot.PlotNums(nCols=5)
        >>> kwplot.figure(fnum=1, doclf=True)
        >>> # kwplot.imshow(kwimage.normalize(daylight_weights))
        >>> kwplot.imshow(kwimage.normalize(allhave_weights['final']), pnum=pnum_(), title='no missing dates')
        >>> kwplot.imshow(kwimage.normalize(anymiss_weights_1['final']), pnum=pnum_(), title='any missing dates (0.1)')
        >>> kwplot.imshow(kwimage.normalize(anymiss_weights_2['final']), pnum=pnum_(), title='any missing dates (0.5)')
        >>> kwplot.imshow(kwimage.normalize(anymiss_weights_3['final']), pnum=pnum_(), title='any missing dates (0.9)')
        >>> kwplot.imshow(kwimage.normalize(allmiss_weights['final']), pnum=pnum_(), title='all missing dates')

        >>> import pandas as pd
        >>> sns = kwplot.autosns()
        >>> fig = kwplot.figure(fnum=2, doclf=True)
        >>> kwplot.imshow(kwimage.normalize(allhave_weights['final']), pnum=(1, 3, 1), title='pairwise affinity')
        >>> row_idx = 5
        >>> df = pd.DataFrame({k: v[row_idx] for k, v in allhave_weights.items()})
        >>> df['index'] = np.arange(df.shape[0])
        >>> data = df.drop(['final'], axis=1).melt(['index'])
        >>> kwplot.figure(fnum=2, pnum=(1, 3, 2))
        >>> sns.lineplot(data=data, x='index', y='value', hue='variable')
        >>> fig.gca().set_title('Affinity components for row={}'.format(row_idx))
        >>> kwplot.figure(fnum=2, pnum=(1, 3, 3))
        >>> sns.lineplot(data=df, x='index', y='final')
        >>> fig.gca().set_title('Affinity components for row={}'.format(row_idx))

    Example:
        >>> # xdoctest: +REQUIRES(env:SMART_DATA_DVC_DPATH)
        >>> from geowatch.tasks.fusion.datamodules.temporal_sampling import *  # NOQA
        >>> import geowatch
        >>> import kwimage
        >>> data_dvc_dpath = geowatch.find_dvc_dpath(tags='phase2_data', hardware='auto')
        >>> coco_fpath = data_dvc_dpath / 'Drop6/imgonly-KR_R001.kwcoco.json'
        >>> dset = geowatch.coerce_kwcoco(coco_fpath)
        >>> vidid = dset.dataset['videos'][0]['id']
        >>> self = TimeWindowSampler.from_coco_video(dset, vidid, time_window=5, time_kernel='-1y,-3m,0,3m,1y', affinity_type='soft3')
        >>> unixtimes = self.unixtimes
        >>> sensors = self.sensors
        >>> time_kernel = self.time_kernel
        >>> time_span = None
        >>> version = 4
        >>> heuristics = 'default'
        >>> weights = soft_frame_affinity(unixtimes, sensors, time_kernel, time_span, version, heuristics)
        >>> # xdoctest: +REQUIRES(--show)
        >>> import kwplot
        >>> kwplot.autoplt()
        >>> pnum_ = kwplot.PlotNums(nCols=5)
        >>> kwplot.figure(fnum=1, doclf=True)
        >>> kwplot.imshow(kwimage.normalize(weights['final']), pnum=pnum_(), title='all missing dates')

        >>> import pandas as pd
        >>> sns = kwplot.autosns()
        >>> fig = kwplot.figure(fnum=2, doclf=True)
        >>> kwplot.imshow(weights['final'], pnum=(1, 3, 1), title='pairwise affinity', cmap='viridis')
        >>> row_idx = 200
        >>> df = pd.DataFrame({k: v[row_idx] for k, v in weights.items()})
        >>> df['index'] = np.arange(df.shape[0])
        >>> data = df.drop(['final'], axis=1).melt(['index'])
        >>> kwplot.figure(fnum=2, pnum=(1, 3, 2))
        >>> sns.lineplot(data=data, x='index', y='value', hue='variable')
        >>> fig.gca().set_title('Affinity components for row={}'.format(row_idx))
        >>> kwplot.figure(fnum=2, pnum=(1, 3, 3))
        >>> sns.lineplot(data=df, x='index', y='final')
        >>> fig.gca().set_title('Affinity components for row={}'.format(row_idx))

    """
    if time_span is not None and time_kernel is not None:
        raise ValueError('time_span and time_kernel are mutex')

    if heuristics == 'default':
        if version in {1, 2}:
            heuristics = {'daylight', 'season', 'sensor_similiarty'}
        elif version == 3:
            heuristics = {'daylight', 'season', 'sensor_similiarty', 'sensor_value'}
        elif version in {4, 'sval'}:
            heuristics = {'sensor_value'}
        elif version in {5, 'ssim'}:
            heuristics = {'sensor_similiarty'}

    missing_date = np.isnan(unixtimes)
    missing_any_dates = np.any(missing_date)
    have_any_dates = not np.all(missing_date)

    weights = {}

    if have_any_dates:
        # unixtimes[np.random.rand(*unixtimes.shape) > 0.1] = np.nan
        seconds_per_year = datetime_mod.timedelta(days=365).total_seconds()
        seconds_per_day = datetime_mod.timedelta(days=1).total_seconds()

        second_deltas = np.abs(unixtimes[None, :] - unixtimes[:, None])

        # Upweight similar seasons
        if 'season' in heuristics:
            year_deltas = second_deltas / seconds_per_year
            season_weights = (1 + np.cos(year_deltas * math.tau)) / 2.0

        # Upweight similar times of day
        if 'daylight' in heuristics:
            day_deltas = second_deltas / seconds_per_day
            daylight_weights = ((1 + np.cos(day_deltas * math.tau)) / 2.0) * 0.95 + 0.95

        if version == 1:
            # backwards compat
            # Upweight times in the future
            # future_weights = year_deltas ** 0.25
            # future_weights = util_kwarray.asymptotic(year_deltas, degree=1)
            future_weights = util_kwarray.tukey_biweight_loss(year_deltas, c=0.5)
            future_weights = future_weights - future_weights.min()
            future_weights = (future_weights / future_weights.max())
            future_weights = future_weights * 0.8 + 0.2
            weights['future'] = future_weights
        elif version in {2, 3}:
            # TODO:
            # incorporate the time_span?
            # if version == 2:
            if time_span is not None:
                try:
                    time_span = coerce_timedelta(time_span).total_seconds()
                except Exception:
                    print(f'time_span={time_span!r}')
                    print('time_span = {}'.format(ub.urepr(time_span, nl=1)))
                    raise
                span_delta = (second_deltas - time_span) ** 2
                norm_span_delta = span_delta / (time_span ** 2)
                weights['time_span'] = (1 - np.minimum(norm_span_delta, 1)) * 0.5 + 0.5

            # Modify the influence of season / daylight
            if 'daylight' in heuristics:
                # squash daylight weight influence
                try:
                    middle = np.nanmean(daylight_weights)
                except Exception:
                    middle = 0
                daylight_weights = (daylight_weights - middle) * 0.1 + (middle / 2)
            if 'season' in heuristics:
                season_weights = ((season_weights - 0.5) / 2) + 0.5

        if version == 3:
            season_weights = (season_weights / 32) + 0.3

        if 'daylight' in heuristics:
            weights['daylight'] = daylight_weights

        if 'season' in heuristics:
            weights['season'] = season_weights

    if sensors is not None:
        sensors = np.asarray(sensors)

        if 'sensor_similiarty' in heuristics:
            same_sensor = sensors[:, None] == sensors[None, :]
            sensor_similarity_weight = ((same_sensor * 0.5) + 0.5)
            if version >= 3:
                sensor_similarity_weight = sensor_similarity_weight / 32 + .4
            weights['sensor_similarity'] = sensor_similarity_weight

        # TODO: this info does not belong here. Pass this information in.
        if 'sensor_value' in heuristics:
            from geowatch.heuristics import SENSOR_TEMPORAL_SAMPLING_VALUES
            sensor_value = SENSOR_TEMPORAL_SAMPLING_VALUES
            values = np.array(list(ub.take(sensor_value, sensors, default=1))).astype(float)
            values /= values.max()
            sensor_value_weight = np.sqrt(values[:, None] * values[None, :])
            weights['sensor_value'] = sensor_value_weight

    if time_kernel is not None:
        if 0:
            # Don't do anything with the time kernel here. That will be handled
            # at sample time.
            delta_diff = (unixtimes[:, None] - unixtimes[None, :])
            diff = np.abs((delta_diff - time_kernel[:, None, None]))
            sdiff = diff - diff.min(axis=0)[None, :, :]
            s = 1 / sdiff.mean(axis=0)
            flags = np.isinf(s)
            s[flags] = 0
            s = (s / s.max())
            s[flags] = 1
            # kwplot.autoplt().imshow(s, cmap='magma')
            kernel_weight = s
            weights['kernel_weight'] = kernel_weight

    if len(weights) == 0:
        frame_weights = np.ones((len(unixtimes), len(unixtimes)))
    else:
        frame_weights = np.prod(np.stack(list(weights.values())), axis=0)

    if missing_any_dates:
        # For the frames that don't have dates on them, we use indexes to
        # calculate a proxy weight.
        frame_idxs = np.arange(len(unixtimes))
        frame_dist = np.abs(frame_idxs[:, None] - frame_idxs[None, ])
        index_weight = (frame_dist / len(frame_idxs)) ** 0.33
        weights['index'] = index_weight

        # Interpolate over any existing values
        # https://stackoverflow.com/questions/21690608/numpy-inpaint-nans-interpolate-and-extrapolate
        if have_any_dates:
            from scipy import interpolate
            miss_idxs = frame_idxs[missing_date]
            have_idxs = frame_idxs[~missing_date]

            miss_coords = np.vstack([
                util_kwarray.cartesian_product(miss_idxs, frame_idxs),
                util_kwarray.cartesian_product(have_idxs, miss_idxs)])
            have_coords = util_kwarray.cartesian_product(have_idxs, have_idxs)
            have_values = frame_weights[tuple(have_coords.T)]

            interp = interpolate.LinearNDInterpolator(have_coords, have_values, fill_value=0.8)
            interp_vals = interp(miss_coords)

            miss_coords_fancy = tuple(miss_coords.T)
            frame_weights[miss_coords_fancy] = interp_vals

            # Average interpolation with the base case
            frame_weights[miss_coords_fancy] = (
                frame_weights[miss_coords_fancy] +
                index_weight[miss_coords_fancy]) / 2
        else:
            # No data to use, just use
            frame_weights = index_weight

    weights['final'] = frame_weights
    return weights


@profile
def hard_frame_affinity(unixtimes, sensors, time_window, time_kernel=None, time_span=None, blur=False):
    # Hard affinity

    sample_idxs = hard_time_sample_pattern(unixtimes, time_window,
                                           time_kernel=time_kernel,
                                           time_span=time_span)
    affinity = kwarray.one_hot_embedding(
        sample_idxs, len(unixtimes), dim=1).sum(axis=2)
    affinity[np.eye(len(affinity), dtype=bool)] = 0
    if blur:
        if blur is True:
            affinity = kwimage.gaussian_blur(affinity, kernel=(5, 1))
        else:
            affinity = kwimage.gaussian_blur(affinity, sigma=blur)

    affinity[np.eye(len(affinity), dtype=bool)] = 0
    # affinity = affinity * 0.99 + 0.01
    affinity = affinity / max(affinity.max(), 1e-9)
    affinity[np.eye(len(affinity), dtype=bool)] = 1
    return affinity


@ub.memoize
def cython_aff_samp_mod():
    """ Old JIT code, no longer works """
    import os
    from geowatch.tasks.fusion.datamodules import temporal_sampling
    fpath = os.path.join(os.path.dirname(temporal_sampling.__file__), 'affinity_sampling.pyx')
    import xdev
    cython_mod = xdev.import_module_from_pyx(fpath, verbose=0, annotate=True)
    return cython_mod
