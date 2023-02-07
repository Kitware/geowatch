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
    from watch.tasks.fusion.datamodules.temporal_sampling import *  # NOQA
    import watch
    data_dvc_dpath = watch.find_dvc_dpath(tags='phase2_data', hardware='auto')
    coco_fpath = data_dvc_dpath / 'Drop4-BAS/KR_R001.kwcoco.json'
    dset = watch.coerce_kwcoco(coco_fpath)
    vidid = dset.dataset['videos'][0]['id']
    self = TimeWindowSampler.from_coco_video(
        dset, vidid,
        time_window=11,
        affinity_type='hardish3', time_span='3m', update_rule='pairwise+distribute', determenistic=True
    )

    # from scipy.special import expit
    from watch.utils.util_time import coerce_timedelta
    # ideal_pattern = 'time_kernel:60d-30d-0-30d-60d'
    ideal_pattern = 'time_kernel:3y-1y-60d-30d-1d-0-1d-30d-60d-1y-3y'
    kernel_deltas = ideal_pattern.split(':')[1].split('-')
    parsed = [coerce_timedelta(d) for d in kernel_deltas]
    import numpy as np
    prekernel = np.array([v.total_seconds() for v in parsed])
    presign = np.sign(np.diff(prekernel))
    kernel = prekernel.copy()
    kernel[0:len(prekernel) - 1] *= presign

    import kwarray
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
