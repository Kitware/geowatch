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
