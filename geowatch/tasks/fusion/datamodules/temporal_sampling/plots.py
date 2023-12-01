from datetime import datetime as datetime_cls
import kwarray
import kwimage
import numpy as np
import ubelt as ub
from .utils import guess_missing_unixtimes


def show_affinity_sample_process(chosen, info, fnum=1):
    """
    Debugging / demo visualization of the iterative sample algorithm.
    For details see :func:`TimeWindowSampler.show_procedure`.
    """
    # import seaborn as sns
    import kwplot

    mask_color = kwimage.Color.coerce('kitware_yellow').as01()
    probability_color = kwimage.Color.coerce('kitware_blue').as01()
    update_weight_color = kwimage.Color.coerce('kitware_green').as01()
    prev_chosen_color = kwimage.Color.coerce('kitware_darkgray').as01()
    chosen_text_color = 'black'
    chosen_arrow_color = 'orange'
    chosen_line_color = 'orange'

    # from matplotlib import pyplot as plt
    steps = info['steps']
    unixtimes = info.get('unixtimes', None)
    if unixtimes is not None:
        unixtimes = guess_missing_unixtimes(unixtimes)

    _include_summary_row = 0

    pnum_ = kwplot.PlotNums(nCols=2, nRows=len(steps) + (1 + _include_summary_row))
    fig = kwplot.figure(fnum=fnum, doclf=True)

    fig = kwplot.figure(pnum=pnum_(), fnum=fnum)
    ax = fig.gca()

    # initial_weights = info['initial_weights']
    # initial_indexes = info['include_indices']
    initial_indexes = info['initial_chosen']

    # if len(initial_indexes):
    idx = initial_indexes[0]
    # else:
    #     idx = None
    probs = info['initial_weights']
    ymax = probs.max()
    xmax = len(probs)

    SHOW_UNIXTIMES_IN_TOP_LEFT = 0

    if unixtimes is None or not SHOW_UNIXTIMES_IN_TOP_LEFT:
        for x_ in initial_indexes:
            ax.plot([x_, x_], [0, ymax], color=prev_chosen_color)
        ax.plot(np.arange(xmax), probs)
    else:
        datetimes = np.array([datetime_cls.fromtimestamp(t) for t in unixtimes])
        initial_datetimes = datetimes[initial_indexes]
        for x_ in initial_datetimes:
            ax.plot([x_, x_], [0, ymax], color=prev_chosen_color)
        ax.plot(datetimes, probs)
    ax.set_title('Initialize included indices')

    fig = kwplot.figure(pnum=pnum_())
    ax = fig.gca()

    initial_mask = info.get('initial_mask', None)
    xidxs = np.arange(xmax)
    if initial_mask is not None:
        ax.fill_between(xidxs, initial_mask, color=mask_color, alpha=0.5)

    try:
        ax.plot(xidxs, info['initial_update_weights'], color=update_weight_color)
    except ValueError:
        ax.plot(xidxs, [info['initial_update_weights']] * xmax, color=update_weight_color)
        ...
    if initial_mask is not None:
        ax.set_title('Initialize update weights & Mask')
    else:
        ax.set_title('Initialize update weights')

    # kwplot.imshow(kwimage.normalize(affinity), title='Pairwise Affinity')

    chosen_so_far = list(initial_indexes)

    start_index = list(initial_indexes)
    for step_idx, step in enumerate(steps, start=len(initial_indexes)):
        fig = kwplot.figure(pnum=pnum_())
        ax = fig.gca()
        idx = step['next_idx']
        probs = step['probs']
        ymax = probs.max()
        if ymax == 0:
            ymax = 1
        xmax = len(probs)
        x, y = idx, probs[idx]
        for x_ in chosen_so_far:
            ax.plot([x_, x_], [0, ymax], color=prev_chosen_color)
        ax.plot(np.arange(xmax), probs, color=probability_color)
        xpos = x + xmax * 0.0 if x < (xmax / 2) else x - xmax * 0.0
        ypos = y + ymax * 0.3 if y < (ymax / 2) else y - ymax * 0.3
        ax.annotate('chosen', (x, y), xytext=(xpos, ypos), color=chosen_text_color, arrowprops=dict(color=chosen_arrow_color, arrowstyle='->'))
        ax.plot([x, x], [0, ymax], color=chosen_line_color)
        #ax.annotate('chosen', (x, y), color='black')
        ax.set_title('Iteration {}: sample'.format(step_idx))

        chosen_so_far.append(idx)

        fig = kwplot.figure(pnum=pnum_())
        ax = fig.gca()
        if step_idx < len(steps):
            next_mask = step.get('next_mask', None)
            xidxs = np.arange(xmax)
            try:
                if next_mask is not None:
                    ax.fill_between(xidxs, next_mask, color=mask_color, alpha=0.5)
                ax.plot(xidxs, step['update_weights'], color=update_weight_color)
                ax.plot([x, x], [0, step['update_weights'].max()], color=chosen_line_color)
            except ValueError:
                ax.plot(xidxs, [step['update_weights']] * xmax, color=update_weight_color)
                ax.plot([x, x], [0, step['update_weights']], color=chosen_line_color)
            if next_mask is not None:
                ax.set_title('Iteration {}: update & mask weights'.format(step_idx))
            else:
                ax.set_title('Iteration {}: update weights'.format(step_idx))
        else:
            if unixtimes is None:
                for x_ in chosen_so_far:
                    ax.plot([x_, x_], [0, ymax], color=prev_chosen_color)
            else:
                chosen_unixtimes = unixtimes[chosen_so_far]
                chosen_datetimes = np.array([datetime_cls.fromtimestamp(t) for t in chosen_unixtimes])
                for x_ in chosen_datetimes:
                    ax.plot([x_, x_], [0, ymax], color=prev_chosen_color)

            ax.set_title('Final sample')

    if _include_summary_row:
        # This last row is not helpful, don't include it.
        affinity = info['affinity']
        fig = kwplot.figure(pnum=pnum_())
        ax = fig.gca()
        for row in affinity[chosen]:
            ax.plot(row)
        ax.set_title('Chosen affinities')
        # kwplot.imshow(kwimage.normalize(), pnum=pnum_(), title='Chosen Affinities')

        final_mat = affinity[chosen][:, chosen]
        final_mat[np.isnan(final_mat)] = 0
        final_mat = kwimage.normalize(final_mat)
        kwplot.imshow(final_mat, pnum=pnum_(), title='Final affinities')

    title_suffix = info.get('title_suffix', '')
    fig.suptitle(f'Sample procedure: {start_index}{title_suffix}')
    fig.subplots_adjust(hspace=0.4)
    return fig


def plot_dense_sample_indices(sample_idxs, unixtimes, title_suffix='', linewidths=0):
    """
    Visualization helper

    Args:
        sample_idxs (List[List[int]] | ArrayLike[ndim=2]):
            A list of frame indexes that index into unixtimes.
            I.e. multiple samples of frame index groups.

        unixtimes (List | ArrayLike[ndim=1] | None):[
            An array of unix timestamps corresonding to frame indexes.
            If unspecified, then frame indexes are shown directly.

    Example:
        >>> unixtimes = None
        >>> sample_idxs = [
        >>>     [0, 1, 2],
        >>>     [3, 5, 6],
        >>>     [2, 3, 6],
        >>> ]
        >>> plot_dense_sample_indices(sample_idxs, unixtimes)
    """
    import seaborn as sns
    import pandas as pd

    use_datetimes = unixtimes is not None
    if not use_datetimes:
        max_frame = max([max(s) for s in sample_idxs])
        unixtimes = np.arange(max_frame + 1)

    num_keyframes = len(unixtimes)
    try:
        # Fast homogeneous path
        dense_sample = kwarray.one_hot_embedding(sample_idxs, num_keyframes, dim=1).sum(axis=2)
    except AttributeError:
        # Slower heterogeneous path
        rows = []
        for frame_idxs in sample_idxs:
            frame_idxs = np.array(frame_idxs)
            row = kwarray.one_hot_embedding(frame_idxs, num_keyframes, dim=0).sum(axis=1)
            rows.append(row)
        dense_sample = np.array(rows)

    unixtimes = guess_missing_unixtimes(unixtimes)

    # =====================
    # Show Sample Pattern in heatmap
    datetimes = np.array([datetime_cls.fromtimestamp(t) for t in unixtimes])
    # dates = np.array([datetime_cls.fromtimestamp(t).date() for t in unixtimes])
    df = pd.DataFrame(dense_sample)
    df.index.name = 'index'
    if use_datetimes:
        df.columns = pd.to_datetime(datetimes).date
        df.columns.name = 'date'
    ax = sns.heatmap(data=df, cbar=False, linewidths=linewidths, linecolor='darkgray')
    ax.set_title('Sample Indexes' + title_suffix)
    ax.set_xlabel('Observation Index')
    ax.set_ylabel('Sample Index')
    return ax


def plot_temporal_sample_indices(sample_idxs, unixtimes=None, sensors=None, title_suffix=''):
    """
    Visualization helper

    Args:
        sample_idxs (List[List[int]]):
            A list of frame indexes that index into unixtimes.
            I.e. multiple samples of frame index groups.

        unixtimes (List | None):
            An array of unix timestamps corresonding to frame indexes.
            If unspecified, then frame indexes are shown directly.

    Example:
        >>> unixtimes = None
        >>> sample_idxs = [
        >>>     [0, 1, 2],
        >>>     [3, 5, 6],
        >>>     [2, 3, 6],
        >>> ]
        >>> plot_temporal_sample_indices(sample_idxs, unixtimes)
    """
    import matplotlib.pyplot as plt
    import kwimage

    if unixtimes is None:
        xlabel = 'Frame Index'
        max_frame = max([max(s) for s in sample_idxs])
        datetimes = np.arange(max_frame + 1)
    else:
        xlabel = 'Time'
        unixtimes = guess_missing_unixtimes(unixtimes)
        datetimes = np.array([datetime_cls.fromtimestamp(t) for t in unixtimes])
    # =====================
    # Show Sample Pattern WRT to time
    ax = plt.gca()

    if sensors:
        unique_sensors = set(sensors)
        unique_colors = kwimage.Color.distinct(len(unique_sensors))
        sensor_to_color = ub.dzip(unique_sensors, unique_colors)
        colors = [sensor_to_color[s] for s in sensors]
    else:
        colors = ['darkblue'] * len(datetimes)

    # Mark available observation locations
    for t, color in zip(datetimes, colors):
        ax.plot([t, t], [0, len(sample_idxs) + 1], color=color, alpha=0.5)

    # Order the samples along the y-axis
    sample_ordering = 'duration'
    sample_ordering = 'start_time'

    if sample_ordering == 'start_time':
        sample_idxs = sorted(sample_idxs, key=lambda x: tuple([min(x), max(x)]))  # start time
    elif sample_ordering == 'end_time':
        sample_idxs = sorted(sample_idxs, key=lambda x: tuple([max(x), min(x)]))
    elif sample_ordering == 'duration':
        sample_idxs = sorted(sample_idxs, key=lambda x: tuple([max(x) - min(x), min(x), max(x)]))
    else:
        raise KeyError(sample_ordering)

    # Mark specific sample locations
    for sample_ypos, sample in enumerate(sample_idxs, start=1):
        ax.plot(datetimes[sample], [sample_ypos] * len(sample), '-', marker='.')

    ax.set_title('Sample Times' + title_suffix)
    ax.set_xlabel(xlabel)
    ax.set_ylabel('Sample Index')
    return ax
    # import matplotlib.dates as mdates
    # ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    # ax.xaxis.set_major_locator(mdates.DayLocator(interval=1))
    # ax.figure.autofmt_xdate()


def plot_temporal_sample(affinity, sample_idxs, unixtimes, sensors=None, fnum=1):
    """
    Visualization helper
    """
    import kwplot
    kwplot.autompl()

    # =====================
    # Show Sample Pattern in heatmap
    kwplot.figure(fnum=fnum, pnum=(2, 1, 1))
    plot_dense_sample_indices(sample_idxs, unixtimes, title_suffix='')

    # =====================
    # Show Sample Pattern WRT to time
    kwplot.figure(fnum=fnum, pnum=(2, 1, 2))
    plot_temporal_sample_indices(sample_idxs, unixtimes, sensors=sensors)
