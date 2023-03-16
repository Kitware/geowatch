def _dev_1darray_sample():
    """
    What are other options to sample roughly uniformly from clumpy data?

    References:
        https://stats.stackexchange.com/questions/122668/is-there-a-measure-of-evenness-of-spread
    """
    import numpy as np
    import ubelt as ub
    # The idea is that we are given cluttered datetimes
    from watch.utils.util_time import coerce_timedelta, coerce_datetime
    from watch.tasks.fusion.datamodules.temporal_sampling.plots import plot_temporal_sample_indices
    # Generate a "clumpy" sample

    def demo_clumpy_data(N, rng):
        uniform = np.linspace(0, 1, N)
        noise = rng.randn(N) / N
        initial = kwarray.normalize(uniform + noise)
        num_clumps = int(N // 10)
        # ranks = initial.argsort()
        ranks = rng.rand(N).argsort()
        clump_size = 7
        clump_indexes = ranks[0:num_clumps]
        # eaten_indexes = ranks[num_clumps:num_clumps + num_clumps * clump_size]
        remain_indexes = ranks[num_clumps + num_clumps * clump_size:]
        clump_seeds = uniform[clump_indexes]
        clump_members = (clump_seeds[:, None] + rng.randn(num_clumps, clump_size) / N).ravel()
        clumpy = kwarray.normalize(np.r_[initial[clump_indexes], initial[remain_indexes], clump_members])
        clumpy.sort()
        return clumpy
    import kwarray
    N = 100
    rng = kwarray.ensure_rng()
    clumpy = demo_clumpy_data(N, rng)
    start_time = coerce_datetime('now').timestamp()
    obs_time = coerce_timedelta('10years').total_seconds()
    # time_span = N * sample_delta
    # delta = time_span.total_seconds()
    unixtimes = (clumpy * obs_time) + start_time
    # time_span
    import kwplot
    plt = kwplot.autoplt()
    plt.plot(unixtimes, 'o')

    k_sizes = [3, 5, 10, 20]
    all_idxs = np.arange(len(unixtimes))
    with ub.Timer('kmeans'):
        sample_idxs = [all_idxs]
        for k in k_sizes:
            from sklearn.cluster import KMeans
            km = KMeans(n_clusters=k)
            km = km.fit(unixtimes[:, None])
            labels = km.predict(unixtimes[:, None])
            # idx_to_cluster = kwarray.group_items(unixtimes, labels)
            cx_to_sxs = ub.dzip(*kwarray.group_indices(labels))
            km_sample_idxs = []
            for cx, sxs in cx_to_sxs.items():
                cluster = unixtimes[sxs]
                midx = sxs[np.abs(cluster - km.cluster_centers_[cx]).argmin()]
                km_sample_idxs.append(midx)
            km_sample_idxs = np.array(km_sample_idxs)
            sample_idxs.append(km_sample_idxs)
        # plot_temporal_sample_indices(sample_idxs, unixtimes)

    # from sklearn.cluster import MeanShift
    # ms = MeanShift()
    # ms.fit(unixtimes[:, None])
    # Uniform bins method
    with ub.Timer('1step'):
        norm_data =  kwarray.normalize(unixtimes)
        for k in k_sizes:
            centers = np.linspace(0, 1, k, endpoint=0) + (1 / (k * 2))
            labels = np.abs(norm_data[None, :] - centers[:, None]).argmin(axis=0)
            cx_to_sxs = ub.dzip(*kwarray.group_indices(labels))
            km_sample_idxs = []
            for cx, sxs in cx_to_sxs.items():
                cluster = unixtimes[sxs]
                midx = sxs[np.abs(cluster - km.cluster_centers_[cx]).argmin()]
                km_sample_idxs.append(midx)
            km_sample_idxs = np.array(km_sample_idxs)
            sample_idxs.append(km_sample_idxs)

    plot_temporal_sample_indices(sample_idxs, unixtimes)


def test_time_strategy():
    # Basic overview demo of the algorithm
    from watch.tasks.fusion.datamodules.temporal_sampling import TimeWindowSampler
    import watch
    dset = watch.coerce_kwcoco('watch-msi', geodata=True, dates=True, num_frames=128, image_size=(32, 32))
    vidid = dset.dataset['videos'][0]['id']
    self = TimeWindowSampler.from_coco_video(
        dset, vidid,
        time_window=11,
        affinity_type='soft2', time_span='8m', update_rule='distribute',
    )
    # xdoctest: +REQUIRES(--show)
    import kwplot
    kwplot.autosns()

    # xdoctest: +REQUIRES(env:SMART_DATA_DVC_DPATH)
    # from watch.tasks.fusion.datamodules.temporal_sampling import *  # NOQA
    import watch
    data_dvc_dpath = watch.find_dvc_dpath(tags='phase2_data', hardware='auto')
    coco_fpath = data_dvc_dpath / 'Drop6/imgonly-KR_R001.kwcoco.json'
    dset = watch.coerce_kwcoco(coco_fpath)
    vidid = dset.dataset['videos'][0]['id']
    self = TimeWindowSampler.from_coco_video(
        dset, vidid,
        time_window=11,
        affinity_type='hardish3', time_span='3m', update_rule='pairwise+distribute', determenistic=True
    )

    # from scipy.special import expit
    from watch.tasks.fusion.datamodules.temporal_sampling.utils import coerce_time_kernel
    from watch.utils.util_time import coerce_timedelta
    # import kwarray
    import numpy as np
    pattern = '-1y,-60d,-30d,-1d,0,1d,30d,60d,1y'
    kernel = coerce_time_kernel(pattern)

    delta_diff = (self.unixtimes[:, None] - self.unixtimes[None, :])
    kwplot.autoplt().imshow(delta_diff, cmap='coolwarm')

    [coerce_timedelta(d) for d in delta_diff[0]][0:10]

    diff = np.abs((delta_diff - kernel[:, None, None]))
    sdiff = diff - diff.min(axis=0)[None, :, :]
    energy = sdiff.mean()
    kwplot.autoplt().imshow(sdiff.mean(axis=0), cmap='magma')

    s = 1 / sdiff.mean(axis=0)
    flags = np.isinf(s)
    s[flags] = 0
    s = (s / s.max())
    s[flags] = 1
    kwplot.autoplt().imshow(s, cmap='magma')
    distance_weight = s

    # delta_diff01 = kwarray.normalize(delta_diff)
    # kernel01 = kwarray.normalize(kernel, min_val=delta_diff.min(), max_val=delta_diff.max())

    sensor_value = {
        'WV': 10,
        'WV1': 9,
        'S2': 1,
        'PD': 7,
        'L8': 0.3,
        'sensor1': 11,
        'sensor2': 7,
        'sensor3': 5,
        'sensor4': 3,
    }
    import ubelt as ub
    values = np.array(list(ub.take(sensor_value, self.sensors, default=1))).astype(float)
    values /= values.max()
    sensor_weight = np.sqrt(values[:, None] * values[None, :])
    # distance = (1 - np.abs((delta_diff01 - kernel01[:, None, None])).min(axis=0))

    # kwplot.autoplt().imshow(distance, cmap='plasma')
    # kwplot.autoplt().imshow(distance ** 4, cmap='plasma')

    energy = distance_weight * sensor_weight
    # energy = distance_weight
    energy = (energy / np.diag(energy)[:, None]).clip(None, 1)
    energy.max(axis=0)

    self.affinity = self.affinity * energy

    kwplot.autoplt().imshow(energy, cmap='plasma')

    import kwplot
    plt = kwplot.autoplt()
    self.show_summary(samples_per_frame=11, fnum=1)
    self.show_procedure(fnum=4)
    plt.subplots_adjust(top=0.9)


def _format_xaxis_as_timedelta(ax):
    """
    Add to util_kwplot
    """
    import datetime as datetime_mod
    def timeTicks(x, pos):
        d = datetime_mod.timedelta(seconds=x)
        return str(d.days) #+ ' days'
    import matplotlib as mpl
    formatter = mpl.ticker.FuncFormatter(timeTicks)
    ax.xaxis.set_major_formatter(formatter)


def _visualize_temporal_sampling():
    from watch.utils import util_time
    import numpy as np
    from watch.tasks.fusion.datamodules.temporal_sampling.utils import coerce_time_kernel
    import kwarray

    # time_kernel = coerce_time_kernel('-1y,-6m,-3m,0,3m,6m,1y')
    time_kernel = coerce_time_kernel('-1y,-3m,0,3m,1y')
    time_range = util_time.coerce_timedelta('4y')

    rng = kwarray.ensure_rng()
    relative_unixtimes = rng.rand(10) * time_range.total_seconds()
    idx = ((relative_unixtimes - (time_range.total_seconds() / 2)) ** 2).argmin()
    mid = relative_unixtimes[idx]
    relative_unixtimes = relative_unixtimes - mid

    # time_kernel = coerce_time_kernel('-1H,-5M,0,5M,1H')
    # relative_unixtimes = coerce_time_kernel('-90M,-70M,-50M,0,1sec,10S,30M')
    # relative_unixtimes = coerce_time_kernel('-90M,-70M,-50M,-20M,-10M,0,1sec,10S,30M,57M,87M')

    from watch.tasks.fusion.datamodules.temporal_sampling.affinity import make_soft_mask
    kernel_masks, kernel_attrs = make_soft_mask(time_kernel, relative_unixtimes)

    min_t = min(kattr['left'] for kattr in kernel_attrs)
    max_t = max(kattr['right'] for kattr in kernel_attrs)

    min_t = min(min_t, relative_unixtimes[0])
    max_t = max(max_t, relative_unixtimes[-1])

    import kwplot
    from watch.utils import util_kwplot
    plt = kwplot.autoplt()
    import kwimage
    kwplot.close_figures()
    kwplot.figure(fnum=1, doclf=1)
    kernel_color = kwimage.Color.coerce('kitware_green').as01()
    obs_color = kwimage.Color.coerce('kitware_blue').as01()

    kwplot.figure(fnum=1, pnum=(1, 1, 1))

    kwplot.phantom_legend({
        'ideal sample': kernel_color,
        'available observation': obs_color,
    })

    for kattr in kernel_attrs:
        rv = kattr['rv']
        xs = np.linspace(min_t, max_t, 1000)
        ys = rv.pdf(xs)
        kattr['_our_norm'] = ys.sum()
        ys_norm = ys / ys.sum()
        plt.plot(xs, ys_norm)

    ax = plt.gca()
    # ax.set_ylim(0, 1)
    ax.set_xlabel('relative time (days)')
    ax.set_ylabel('sample probability')
    # ax.set_title('ideal sample location')
    ax.set_yticks([])

    obs_line_segments = []
    for x in relative_unixtimes:
        y = 0
        for kattr in kernel_attrs:
            rv = kattr['rv']
            y = max(y, rv.pdf(x) / kattr['_our_norm'])
        obs_line_segments.append([x, y])
    for x, y in obs_line_segments:
        plt.plot([x, x], [0, y], '-', color=obs_color)

    kern_line_segments = []
    for x in time_kernel:
        y = 0
        for kattr in kernel_attrs:
            rv = kattr['rv']
            y = max(y, rv.pdf(x) / kattr['_our_norm'])
        kern_line_segments.append([x, y])
    for x, y in kern_line_segments:
        plt.plot([x, x], [0, y], '--', color=kernel_color)

    # plt.plot(time_kernel, [0] * len(time_kernel), '-o', color=kernel_color, label='ideal frame location')

    kwplot.phantom_legend(label_to_attrs={
        'ideal sample': {'color': kernel_color, 'linestyle': '--'},
        'available observation': {'color': obs_color},
    })

    _format_xaxis_as_timedelta(ax)
    plt.subplots_adjust(top=0.99, bottom=0.1, hspace=.3, left=0.1)
    fig = plt.gcf()
    fig.set_size_inches(np.array([4, 3]) * 1.5)
    fig.tight_layout()
    finalizer = util_kwplot.FigureFinalizer()
    finalizer(fig, 'time_sampling_example.png')
