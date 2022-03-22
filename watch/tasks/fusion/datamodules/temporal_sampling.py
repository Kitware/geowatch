import kwarray
import kwimage
import numpy as np
import ubelt as ub
import math
import datetime
from dateutil import parser
from watch.utils import util_kwarray
from watch.utils.util_time import  coerce_timedelta


class TimeSampleError(IndexError):
    pass


class MultiTimeWindowSampler:
    """
    A wrapper that contains multiple time window samplers with different
    affinity matrices to increase the diversity of temporal sampling.

    Example:
        >>> from watch.tasks.fusion.datamodules.temporal_sampling import *  # NOQA
        >>> low = datetime.datetime.now().timestamp()
        >>> high = low + datetime.timedelta(days=365 * 5).total_seconds()
        >>> rng = kwarray.ensure_rng(0)
        >>> unixtimes = np.array(sorted(rng.randint(low, high, 113)), dtype=float)
        >>> sensors = ['a' for _ in range(len(unixtimes))]
        >>> time_window = 5
        >>> self = MultiTimeWindowSampler(
        >>>     unixtimes=unixtimes, sensors=sensors, time_window=time_window, update_rule='pairwise+distribute',
        >>>     #time_spans=['2y', '1y', '5m'])
        >>>     time_spans='7d-1m', affinity_type='soft2')
        >>> self.sample()
        >>> # xdoctest: +REQUIRES(--show)
        >>> import kwplot
        >>> kwplot.autosns()
        >>> self.show_summary(10)
    """
    def __init__(self, unixtimes, sensors, time_window, affinity_type='hard',
                 update_rule='distribute', determenistic=False, gamma=1,
                 time_spans=['2y', '1y', '5m'], name='?'):

        if isinstance(time_spans, str):
            time_spans = time_spans.split('-')

        self.sensors = sensors
        self.unixtimes = unixtimes
        self.time_window = time_window
        self.update_rule = update_rule
        self.affinity_type = affinity_type
        self.determenistic = determenistic
        self.gamma = gamma
        self.name = name
        self.num_frames = len(unixtimes)
        self.time_spans = time_spans
        self.sub_samplers = {}
        self._build()

    def _build(self):
        for time_span in self.time_spans:
            sub_sampler = TimeWindowSampler(
                unixtimes=self.unixtimes,
                sensors=self.sensors,
                time_window=self.time_window,
                update_rule=self.update_rule,
                affinity_type=self.affinity_type,
                determenistic=self.determenistic,
                gamma=self.gamma,
                name=self.name,
                time_span=time_span,
            )
            self.sub_samplers[time_span] = sub_sampler

    def sample(self, main_frame_idx=None, include=None, exclude=None,
               return_info=False, error_level=0, rng=None):
        """
        Chooses a sub-sampler and samples from it.

        Args:
            main_frame_idx (int): "main" sample index.
            include (List[int]): other indexes forced to be included
            exclude (List[int]): other indexes forced to be excluded
            return_info (bool): for debugging / introspection
            error_level (int): See :func:`affinity_sample`.

        Returns:
            ndarray | Tuple[ndarray, Dict]
        """
        rng = kwarray.ensure_rng(rng)
        chosen_key = rng.choice(list(self.sub_samplers.keys()))
        chosen_sampler = self.sub_samplers[chosen_key]
        return chosen_sampler.sample(main_frame_idx, include=include,
                                     exclude=exclude, return_info=return_info,
                                     rng=rng, error_level=error_level)

    @property
    def affinity(self):
        """
        Approximate combined affinity, for this multi-sampler
        """
        affinity = np.mean(np.stack([sampler.affinity for sampler in self.sub_samplers.values()]), axis=0)
        return affinity

    def show_summary(self, samples_per_frame=1, fnum=1):
        """
        Similar to :func:`TimeWindowSampler.show_summary`
        """
        import kwplot
        kwplot.autompl()

        sample_idxs = []
        for idx in range(len(self.unixtimes)):
            for _ in range(samples_per_frame):
                idxs = self.sample(idx)
                sample_idxs.append(idxs)

        if 0:
            sample_idxs = np.array(sorted(map(tuple, sample_idxs)))
        else:
            sample_idxs = np.array(sample_idxs)

        title_info = ub.codeblock(
            f'''
            name={self.name}
            affinity_type={self.affinity_type} determenistic={self.determenistic}
            update_rule={self.update_rule} gamma={self.gamma}
            ''')

        pnum_ = kwplot.PlotNums(nCols=3)

        fig = kwplot.figure(fnum=fnum, doclf=True)

        fig = kwplot.figure(fnum=fnum, pnum=pnum_())
        ax = fig.gca()

        affinity = self.affinity

        kwplot.imshow(affinity, ax=ax)
        ax.set_title('combined frame affinity')

        fig = kwplot.figure(fnum=fnum, pnum=pnum_())
        if samples_per_frame < 5:
            ax = plot_dense_sample_indices(sample_idxs, self.unixtimes, linewidths=0.1)
            ax.set_aspect('equal')
        else:
            ax = plot_dense_sample_indices(sample_idxs, self.unixtimes, linewidths=0.001)

        kwplot.figure(fnum=fnum, pnum=pnum_())
        plot_temporal_sample_indices(sample_idxs, self.unixtimes)
        fig.suptitle(title_info)


class TimeWindowSampler:
    """
    Object oriented API to produce random temporal samples given a set of
    keyframes with metadata.

    This works by computing a pairwise "affinity" NxN matrix for each of the N
    keyframes. The details of the affinity matrix depend on parameters passed
    to this object. Intuitively, the value at ``Affinity[i, j]`` represents how
    much frame-i "wants" to be in the same sample as frame-j.

    Args:
        unixtimes (List[int]):
            list of unix timestamps for each frame

        sensors (List[str]):
            list of attributes for each frame

        time_window (int):
            number of frames to sample

        affinity_type (str):
            Method for computing the affinity matrix for the underlying
            sampling algorithm. Can be:
                "soft" - The old generalized random affinity matrix.
                "soft2" - The new generalized random affinity matrix.
                "hard" - A simplified affinity algorithm.
                "hardish" - Like hard, but with a blur.
                "contiguous" - Neighboring frames get high affinity.

        update_rule (str):
            "+" separated string that can contain {"distribute", "pairwise"}.
            See :func:`affinity_sample` for details.

        gamma (float):
            Modulates sampling probability. Higher values
            See :func:`affinity_sample` for details.

        time_span (Coercable[datetime.timedelta]):
            The ideal distince in time that frames should be separated in.
            This is typically a string code. E.g. "1y" is one year.

        name (str):
            A name for this object.  For developer convinience, has no
            influence on the algorithm.

        determenistic (bool):
            if True, on each step we choose the next timestamp with maximum
            probability. Otherwise, we randomly choose a timestep, but with
            probability according to the current distribution.  This is an
            attribute, which can be modified to change behavior (not thread
            safe).

    Example:
        >>> # xdoctest: +REQUIRES(env:DVC_DPATH)
        >>> import os
        >>> from watch.tasks.fusion.datamodules.temporal_sampling import *  # NOQA
        >>> import watch
        >>> import kwcoco
        >>> dvc_dpath = watch.find_smart_dvc_dpath()
        >>> coco_fpath = dvc_dpath / 'Drop1-Aligned-L1-2022-01/data.kwcoco.json'
        >>> dset = kwcoco.CocoDataset(coco_fpath)
        >>> vidid = dset.dataset['videos'][0]['id']
        >>> self = TimeWindowSampler.from_coco_video(
        >>>     dset, vidid,
        >>>     time_window=5,
        >>>     affinity_type='soft2', time_span='1y',
        >>>     update_rule='distribute')
        >>> self.determenistic = False
        >>> self.show_summary(samples_per_frame=1, fnum=1)
        >>> self.determenistic = True
        >>> self.show_summary(samples_per_frame=3, fnum=2)
    """

    def __init__(self, unixtimes, sensors, time_window,
                 affinity_type='hard', update_rule='distribute',
                 determenistic=False, gamma=1, time_span='2y', name='?'):
        self.sensors = sensors
        self.unixtimes = unixtimes
        self.time_window = time_window
        self.update_rule = update_rule
        self.affinity_type = affinity_type
        self.determenistic = determenistic
        self.gamma = gamma
        self.name = name
        self.num_frames = len(unixtimes)
        self.time_span = time_span

        self.compute_affinity()

    @classmethod
    def from_coco_video(cls, dset, vidid, gids=None, **kwargs):
        if gids is None:
            gids = dset.images(vidid=vidid).lookup('id')
        images = dset.images(gids)
        name = dset.index.videos[ub.peek(images.lookup('video_id'))].get('name', '<no-name?>')
        datetimes = [None if date is None else parser.parse(date) for date in images.lookup('date_captured', None)]
        unixtimes = np.array([np.nan if dt is None else dt.timestamp() for dt in datetimes])
        sensors = images.lookup('sensor_coarse', None)
        kwargs['unixtimes'] = unixtimes
        kwargs['sensors'] = sensors
        kwargs['name'] = name
        self = cls(**kwargs)
        return self

    def compute_affinity(self):
        """
        Construct the affinity matrix given the current ``affinity_type``.

        Example:
            >>> # xdoctest: +REQUIRES(env:DVC_DPATH)
            >>> import os
            >>> from watch.tasks.fusion.datamodules.temporal_sampling import *  # NOQA
            >>> from watch.utils.util_data import find_smart_dvc_dpath
            >>> dvc_dpath = find_smart_dvc_dpath()
            >>> coco_fpath = dvc_dpath / 'Drop1-Aligned-L1-2022-01/data.kwcoco.json'
            >>> dset = kwcoco.CocoDataset(coco_fpath)
            >>> vidid = dset.dataset['videos'][0]['id']
            >>> self = TimeWindowSampler.from_coco_video(
            >>>     dset, vidid,
            >>>     time_window=5,
            >>>     affinity_type='contiguous',
            >>>     update_rule='pairwise')
            >>> self.determenistic = True
            >>> self.show_procedure(fnum=1)
        """
        if self.affinity_type.startswith('soft'):
            if self.affinity_type == 'soft':
                version = 1
            else:
                version = int(self.affinity_type[4:])
            # Soft affinity
            self.affinity = soft_frame_affinity(self.unixtimes, self.sensors,
                                                self.time_span,
                                                version=version)['final']
        elif self.affinity_type == 'hard':
            # Hard affinity
            self.affinity = hard_frame_affinity(self.unixtimes, self.sensors,
                                                time_window=self.time_window,
                                                blur=False,
                                                time_span=self.time_span)
        elif self.affinity_type == 'hardish':
            # Hardish affinity
            self.affinity = hard_frame_affinity(self.unixtimes, self.sensors,
                                                time_window=self.time_window,
                                                blur=True,
                                                time_span=self.time_span)
        elif self.affinity_type == 'hardish2':
            # Hardish affinity
            self.affinity = hard_frame_affinity(self.unixtimes, self.sensors,
                                                time_window=self.time_window,
                                                blur=3.0,
                                                time_span=self.time_span)
        elif self.affinity_type == 'hardish3':
            # Hardish affinity
            self.affinity = hard_frame_affinity(self.unixtimes, self.sensors,
                                                time_window=self.time_window,
                                                blur=6.0,
                                                time_span=self.time_span)
        elif self.affinity_type == 'contiguous':
            # Recovers the original method that we used to sample time.
            time_window = self.time_window
            unixtimes = self.unixtimes

            # Note: cant just use a sliding window because we currently need
            # and NxN affinity matrix, can remove duplicates at sample time.
            # This allows us to assume that indexes always correspond with
            # frames, and each frame has an "ideal" sample. This assumption
            # might not be necsesary, but other code would need to change to
            # break it, so we keep it in for now.
            n_before = time_window // 2
            n_after = (time_window - n_before) - 1
            num_samples = len(unixtimes)
            all_indexes = np.arange(num_samples)
            sample_idxs = []
            for idx in all_indexes:
                start_idx = idx - n_before
                stop_idx = idx + n_after
                if stop_idx > num_samples:
                    offset = num_samples - start_idx
                    start_idx -= offset
                    stop_idx -= offset
                if start_idx < 0:
                    offset = 0 - start_idx
                    start_idx += offset
                    stop_idx += offset
                sample_idxs.append(all_indexes[start_idx:stop_idx])
            sample_idxs = np.array(sample_idxs)

            # Old, and somewhat better way, but does not give us NxN
            # time_slider = kwarray.SlidingWindow(
            #     (len(self.unixtimes),), (self.time_window,), stride=(1,),
            #     keepbound=True, allow_overshoot=True)
            # sample_idxs = np.array([all_indexes[sl] for sl in time_slider])

            # Construct the contiguous sliding window affinity.
            # (Note: an alternate approach would be to give the first
            # and last frames out-of-bounds padding, so they actually
            # dont give full affinity. That may be more natural)
            self.affinity = kwarray.one_hot_embedding(
                sample_idxs, len(self.unixtimes), dim=1).sum(axis=2)
        else:
            raise KeyError(self.affinity_type)

        self.main_indexes = np.arange(self.affinity.shape[0])

    def sample(self, main_frame_idx=None, include=None, exclude=None,
               return_info=False, error_level=0, rng=None):
        """
        Args:
            main_frame_idx (int): "main" sample index.
            include (List[int]): other indexes forced to be included
            exclude (List[int]): other indexes forced to be excluded
            return_info (bool): for debugging / introspection
            error_level (int): See :func:`affinity_sample`.

        Returns:
            ndarray | Tuple[ndarray, Dict]

        Example:
            >>> # xdoctest: +REQUIRES(env:DVC_DPATH)
            >>> import os
            >>> import kwcoco
            >>> from watch.tasks.fusion.datamodules.temporal_sampling import *  # NOQA
            >>> from watch.utils.util_data import find_smart_dvc_dpath
            >>> dvc_dpath = find_smart_dvc_dpath()
            >>> coco_fpath = dvc_dpath / 'Drop2-Aligned-TA1-2022-02-15/data.kwcoco.json'
            >>> dset = kwcoco.CocoDataset(coco_fpath)
            >>> vidid = dset.dataset['videos'][0]['id']
            >>> self = TimeWindowSampler.from_coco_video(
            >>>     dset, vidid,
            >>>     time_span='1y',
            >>>     time_window=3,
            >>>     affinity_type='soft2',
            >>>     update_rule='distribute+pairwise')
            >>> self.determenistic = False
            >>> self.show_summary(samples_per_frame=1 if self.determenistic else 10, fnum=1)
            >>> self.show_procedure(fnum=2)
        """
        if main_frame_idx is None:
            include_indices = []
        else:
            include_indices = [main_frame_idx]
        if include is not None:
            include_indices.extend(include)
        exclude_indices = exclude
        affinity = self.affinity
        size = self.time_window
        determenistic = self.determenistic
        update_rule = self.update_rule
        gamma = self.gamma
        rng = kwarray.ensure_rng(rng)

        # Ret could be an ndarray | Tuple[ndarray, Dict]
        ret = affinity_sample(
            affinity=affinity,
            size=size,
            include_indices=include_indices,
            exclude_indices=exclude_indices,
            update_rule=update_rule,
            gamma=gamma,
            determenistic=determenistic,
            error_level=error_level,
            rng=rng,
            return_info=return_info,
        )
        return ret

    def show_summary(self, samples_per_frame=1, fnum=1, with_temporal=True):
        """
        Visualize the affinity matrix and two views of a selected sample.

        Plots a figure with three subfigures.

        (1) The affinity matrix.

        (2) A visualization of a random sampled over "index-space".
        A matrix M, where each row is a sample index, each column is a
        timestep, ``M[i,j] = 1`` (the cell is colored white) to indicate that a
        sample-i includes timestep-j.

        (3) A visualization of the same random sample over "time-space".  A
        plot where x is the time-axis is drawn, and vertical lines indicate the
        selectable time indexes. For each sample, a horizontal line indicates
        the timespan of the sample and an "x" denotes exactly which timesteps
        are included in that sample.

        Example:
            >>> # xdoctest: +REQUIRES(env:DVC_DPATH)
            >>> import os
            >>> from watch.tasks.fusion.datamodules.temporal_sampling import *  # NOQA
            >>> from watch.utils.util_data import find_smart_dvc_dpath
            >>> import kwcoco
            >>> # xdoctest: +REQUIRES(--show)
            >>> dvc_dpath = find_smart_dvc_dpath()
            >>> coco_fpath = dvc_dpath / 'Drop2-Aligned-TA1-2022-02-15/data.kwcoco.json'
            >>> dset = kwcoco.CocoDataset(coco_fpath)
            >>> video_ids = list(ub.sorted_vals(dset.index.vidid_to_gids, key=len).keys())
            >>> vidid = video_ids[2]
            >>> # Demo behavior over a grid of parameters
            >>> grid = list(ub.named_product({
            >>>     'affinity_type': ['hard', 'soft2', 'hardish3', 'hardish2'],
            >>>     'update_rule': ['distribute', 'pairwise+distribute'],
            >>>     #'determenistic': [False, True],
            >>>     'determenistic': [False],
            >>>     'time_window': [2],
            >>> }))
            >>> import kwplot
            >>> kwplot.autompl()
            >>> for idx, kwargs in enumerate(grid):
            >>>     print('kwargs = {!r}'.format(kwargs))
            >>>     self = TimeWindowSampler.from_coco_video(dset, vidid, **kwargs)
            >>>     self.show_summary(samples_per_frame=30, fnum=idx, with_temporal=False)
        """
        import kwplot
        kwplot.autompl()

        sample_idxs = []
        for idx in range(self.affinity.shape[0]):
            for _ in range(samples_per_frame):
                idxs = self.sample(idx)
                sample_idxs.append(idxs)

        if 0:
            sample_idxs = np.array(sorted(map(tuple, sample_idxs)))
        else:
            sample_idxs = np.array(sample_idxs)

        title_info = ub.codeblock(
            f'''
            name={self.name}
            affinity_type={self.affinity_type} determenistic={self.determenistic}
            update_rule={self.update_rule} gamma={self.gamma}
            ''')

        with_dense = True
        with_mat = True

        num_subplots = (with_mat + with_dense + with_temporal)

        pnum_ = kwplot.PlotNums(nCols=num_subplots)

        fig = kwplot.figure(fnum=fnum, doclf=True)

        fig = kwplot.figure(fnum=fnum, pnum=pnum_())
        ax = fig.gca()
        kwplot.imshow(self.affinity, ax=ax, cmap='viridis')
        ax.set_title('frame affinity')

        fig = kwplot.figure(fnum=fnum, pnum=pnum_())
        if samples_per_frame < 5:
            ax = plot_dense_sample_indices(sample_idxs, self.unixtimes, linewidths=0.1)
            ax.set_aspect('equal')
        else:
            ax = plot_dense_sample_indices(sample_idxs, self.unixtimes, linewidths=0.001)

        if with_temporal:
            kwplot.figure(fnum=fnum, pnum=pnum_())
            plot_temporal_sample_indices(sample_idxs, self.unixtimes)
        fig.suptitle(title_info)

    def show_affinity(self, fnum=3):
        """
        Simple drawing of the affinity matrix.
        """
        import kwplot
        kwplot.autompl()
        fig = kwplot.figure(fnum=fnum)
        ax = fig.gca()
        kwplot.imshow(self.affinity, ax=ax)
        ax.set_title('frame affinity')

    def show_procedure(self, idx=None, exclude=None, fnum=2, rng=None):
        """
        Draw a figure that shows the process of performing on call to
        :func:`TimeWindowSampler.sample`. Each row illustrates an iteration of
        the algorithm. The left column draws the current indicies included in
        the sample and the right column draws how that sample (corresponding to
        the current row) influences the probability distribution for the next
        row.

        Example:
            >>> # xdoctest: +REQUIRES(env:DVC_DPATH)
            >>> import os
            >>> from watch.tasks.fusion.datamodules.temporal_sampling import *  # NOQA
            >>> from watch.utils.util_data import find_smart_dvc_dpath
            >>> dvc_dpath = find_smart_dvc_dpath()
            >>> coco_fpath = dvc_dpath / 'Drop1-Aligned-L1-2022-01/data.kwcoco.json'
            >>> dset = kwcoco.CocoDataset(coco_fpath)
            >>> vidid = dset.dataset['videos'][0]['id']
            >>> self = TimeWindowSampler.from_coco_video(
            >>>     dset, vidid,
            >>>     time_window=5,
            >>>     affinity_type='soft2',
            >>>     update_rule='distribute+pairwise')
            >>> self.determenistic = False
            >>> self.show_procedure(idx=0, fnum=10)
            >>> self.show_affinity(fnum=100)

            for idx in xdev.InteractiveIter(list(range(self.num_frames))):
                self.show_procedure(idx=idx, fnum=1)
                xdev.InteractiveIter.draw()


            self = TimeWindowSampler.from_coco_video(dset, vidid, time_window=5, affinity_type='soft2', update_rule='distribute+pairwise')
            self.determenistic = True
            self.show_summary(samples_per_frame=20, fnum=1)
            self.determenistic = False
            self.show_summary(samples_per_frame=20, fnum=2)

            self = TimeWindowSampler.from_coco_video(dset, vidid, time_window=5, affinity_type='hard', update_rule='distribute')
            self.determenistic = True
            self.show_summary(samples_per_frame=20, fnum=3)
            self.determenistic = False
            self.show_summary(samples_per_frame=20, fnum=4)

            self = TimeWindowSampler.from_coco_video(dset, vidid, time_window=5, affinity_type='hardish', update_rule='distribute')
            self.determenistic = True
            self.show_summary(samples_per_frame=20, fnum=5)
            self.determenistic = False
            self.show_summary(samples_per_frame=20, fnum=6)

            >>> self.show_procedure(fnum=1)
            >>> self.determenistic = True
            >>> self.show_procedure(fnum=2)
            >>> self.show_procedure(fnum=3)
            >>> self.show_procedure(fnum=4)
            >>> self.determenistic = False
            >>> self.show_summary(samples_per_frame=3, fnum=10)

        """
        rng = kwarray.ensure_rng(rng)
        # if idx is None:
        #     idx = self.num_frames // 2
        title_info = ub.codeblock(
            f'''
            name={self.name}
            affinity_type={self.affinity_type} determenistic={self.determenistic}
            update_rule={self.update_rule} gamma={self.gamma}
            ''')
        chosen, info = self.sample(idx, return_info=True, exclude=exclude,
                                   rng=rng)
        info['title_suffix'] = title_info
        show_affinity_sample_process(chosen, info, fnum=fnum)
        return chosen, info


def affinity_sample(affinity, size, include_indices=None, exclude_indices=None,
                    update_rule='pairwise', gamma=1, determenistic=False,
                    error_level=2, rng=None, return_info=False, jit=False):
    """
    Randomly select `size` timesteps from a larger pool based on "affinity".

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
            Indicies that must be included in the sample

        exclude_indices (List[int]):
            Indicies that cannnot be included in the sample

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
            becomes more likely to sample the maximum probaility at each
            timestep. In the limit this becomes equivalent to
            ``determenistic=True``.

        determenistic (bool):
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

        rng (Coercable[RandomState]):
            random state for reproducible sampling

        return_info (bool):
            If True, includes a dictionary of information that details the
            internal steps the algorithm took.

        jit (bool):
            NotImplemented - do not use

    Returns:
        ndarray | Tuple[ndarray, Dict] -
            The ``chosen`` indexes for the sample, or if return_info is True,
            then returns a tuple of ``chosen`` and the info dictionary.

    Possible Related Work:
        * Random Stratified Sampling Affinity Matrix
        * A quasi-random sampling approach to image retrieval

    Example:
        >>> from watch.tasks.fusion.datamodules.temporal_sampling import *  # NOQA
        >>> low = datetime.datetime.now().timestamp()
        >>> high = low + datetime.timedelta(days=365 * 5).total_seconds()
        >>> rng = kwarray.ensure_rng(0)
        >>> unixtimes = np.array(sorted(rng.randint(low, high, 113)), dtype=float)
        >>> #
        >>> affinity = soft_frame_affinity(unixtimes, version=2, time_span='1d')['final']
        >>> include_indices = [5]
        >>> size = 5
        >>> chosen, info = affinity_sample(affinity, size, include_indices, update_rule='pairwise',
        >>>                                return_info=True, determenistic=True)
        >>> # xdoctest: +REQUIRES(--show)
        >>> import kwplot
        >>> sns = kwplot.autosns()
        >>> plt = kwplot.autoplt()
        >>> show_affinity_sample_process(chosen, info)

    Example:
        >>> from watch.tasks.fusion.datamodules.temporal_sampling import *  # NOQA
        >>> low = datetime.datetime.now().timestamp()
        >>> high = low + datetime.timedelta(days=365 * 5).total_seconds()
        >>> rng = kwarray.ensure_rng(0)
        >>> unixtimes = np.array(sorted(rng.randint(low, high, 5)), dtype=float)
        >>> self = TimeWindowSampler(unixtimes, sensors=None, time_window=4,
        >>>     affinity_type='soft2', time_span='0.3y',
        >>>     update_rule='distribute+pairwise')
        >>> self.determenistic = False
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
        >>> print('info = {}'.format(ub.repr2(info, nl=4)))

    Ignore:
        >>> from watch.tasks.fusion.datamodules.temporal_sampling import *  # NOQA
        >>> low = datetime.datetime.now().timestamp()
        >>> high = low + datetime.timedelta(days=365 * 5).total_seconds()
        >>> rng = kwarray.ensure_rng(0)
        >>> unixtimes = np.array(sorted(rng.randint(low, high, 113)), dtype=float)
        >>> affinity = soft_frame_affinity(unixtimes)['final']
        >>> include_indices = [5]
        >>> size = 20
        >>> xdev.profile_now(affinity_sample)(affinity, size, include_indices)
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

    update_rules = set(update_rule.split('+'))
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
        col_idxs = np.arange(0, affinity.shape[1])
        update_weights *= (np.abs(col_idxs - np.array(chosen)[:, None]) / len(col_idxs)).min(axis=0)

    current_weights = initial_weights * update_weights
    current_weights[chosen] = 0

    num_sample = size - len(chosen)

    if jit:
        raise NotImplementedError
        # out of date
        cython_mod = cython_aff_samp_mod()
        return cython_mod.cython_affinity_sample(affinity, num_sample, current_weights, chosen, rng)

    # available_idxs = np.arange(affinity.shape[0])
    if return_info:
        denom = current_weights.sum()
        if denom == 0:
            denom = 1
        initial_probs = current_weights / denom
        info = {
            'steps': [],

            'initial_weights': initial_weights.copy(),
            'initial_update_weights': update_weights.copy(),
            'initial_probs': initial_probs,

            'initial_chosen': chosen.copy(),

            'include_indices': include_indices,
            'affinity': affinity,
        }

    for _ in range(num_sample):
        # Choose the next image based on combined sample affinity

        total_weight = current_weights.sum()

        if return_info:
            errors = []

        # If we zeroed out all of the probabilities try two things before
        # punting and setting everything to uniform.
        if total_weight == 0:
            if error_level == 3:
                raise TimeSampleError('all probability is exhausted')
            current_weights = affinity[chosen[0]].copy()
            current_weights[chosen] = 0
            current_weights[exclude_indices] = 0
            total_weight = current_weights.sum()
            if return_info:
                errors.append('all indices were chosen, excluded, or had no affinity')
            if total_weight == 0:
                # Should really never get here in day-to-day, but just in case
                if error_level == 2:
                    raise TimeSampleError('all included probability is exhausted')
                current_weights[:] = rng.rand(len(current_weights))
                current_weights[chosen] = 0
                total_weight = current_weights.sum()
                if return_info:
                    errors.append('all indices were chosen, excluded')
                if total_weight == 0:
                    if error_level == 1:
                        raise TimeSampleError('all chosen probability is exhausted')
                    current_weights[:] = rng.rand(len(current_weights))
                    if return_info:
                        errors.append('all indices were chosen, punting')

        if determenistic:
            next_idx = current_weights.argmax()
        else:
            cumprobs = (current_weights ** gamma).cumsum()
            dart = rng.rand() * cumprobs[-1]
            next_idx = np.searchsorted(cumprobs, dart)

        update_weights = 1

        if do_pairwise:
            if next_idx < affinity.shape[0]:
                update_weights = affinity[next_idx] * update_weights

        if do_distribute:
            update_weights = (np.abs(col_idxs - next_idx) / len(col_idxs)) * update_weights

        chosen.append(next_idx)

        if return_info:
            if total_weight == 0:
                probs = current_weights.copy()
            else:
                probs = current_weights / total_weight
            probs = current_weights
            info['steps'].append({
                'probs': probs,
                'next_idx': next_idx,
                'update_weights': update_weights,
                'errors': errors,
            })

        # Modify weights to impact next sample
        current_weights = current_weights * update_weights

        # Don't resample the same item
        current_weights[next_idx] = 0

    chosen = sorted(chosen)
    if return_info:
        return chosen, info
    else:
        return chosen


def hard_time_sample_pattern(unixtimes, time_window, time_span='2y'):
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
        >>> low = datetime.datetime.now().timestamp()
        >>> high = low + datetime.timedelta(days=365 * 5).total_seconds()
        >>> rng = kwarray.ensure_rng(0)
        >>> base_unixtimes = np.array(sorted(rng.randint(low, high, 20)), dtype=float)
        >>> unixtimes = base_unixtimes.copy()
        >>> #unixtimes[rng.rand(*unixtimes.shape) < 0.1] = np.nan
        >>> time_window = 5
        >>> sample_idxs = hard_time_sample_pattern(unixtimes, time_window)
        >>> name = 'demo-data'

        >>> #unixtimes[:] = np.nan
        >>> time_window = 5
        >>> sample_idxs = hard_time_sample_pattern(unixtimes, time_window)
        >>> name = 'demo-data'

    Ignore:
        >>> # xdoctest: +REQUIRES(env:DVC_DPATH)
        >>> import os
        >>> from watch.utils.util_data import find_smart_dvc_dpath
        >>> dvc_dpath = find_smart_dvc_dpath()
        >>> coco_fpath = dvc_dpath / 'Drop1-Aligned-L1-2022-01/data.kwcoco.json'
        >>> dset = kwcoco.CocoDataset(coco_fpath)
        >>> video_ids = list(ub.sorted_vals(dset.index.vidid_to_gids, key=len).keys())
        >>> vidid = video_ids[0]
        >>> video = dset.index.videos[vidid]
        >>> name = (video['name'])
        >>> print('name = {!r}'.format(name))
        >>> images = dset.images(vidid=vidid)
        >>> datetimes = [parser.parse(date) for date in images.lookup('date_captured')]
        >>> unixtimes = np.array([dt.timestamp() for dt in datetimes])
        >>> time_window = 5
        >>> sample_idxs = hard_time_sample_pattern(unixtimes, time_window)
        >>> # xdoctest: +REQUIRES(--show)
        >>> import kwplot
        >>> kwplot.autoplt()
        >>> plot_dense_sample_indices(sample_idxs, unixtimes, title_suffix=f': {name}')

    Ignore:
        >>> import kwplot
        >>> import numpy as np
        >>> sns = kwplot.autosns()

        >>> # =====================
        >>> # Show Sample Pattern in heatmap
        >>> plot_dense_sample_indices(sample_idxs, unixtimes, title_suffix=f': {name}')

        >>> datetimes = np.array([datetime.datetime.fromtimestamp(t) for t in unixtimes])
        >>> dates = np.array([datetime.datetime.fromtimestamp(t).date() for t in unixtimes])
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
        >>> datetimes = [datetime.datetime.fromtimestamp(t) for t in unixtimes]
        >>> grid_dates = date_range(
        >>>     datetimes[0].date().isoformat(),
        >>>     (datetimes[-1] + datetime.timedelta(days=1)).date().isoformat()
        >>> )
        >>> grid_unixtime = np.array([
        >>>     datetime.datetime.combine(d, datetime.datetime.min.time()).timestamp()
        >>>     for d in grid_dates
        >>> ])
        >>> positions = np.searchsorted(grid_unixtime, unixtimes)
        >>> indicator = np.zeros_like(grid_unixtime)
        >>> indicator[positions] = 1
        >>> dates_unixtimes = [d for d in dates]
        >>> july.heatmap(grid_dates, indicator, title=f'Available Observations: {name}', cmap="github")
    """
    if isinstance(time_window, int):
        # TODO: formulate how to choose template delta for given window dims Or
        # pass in a delta
        if time_window == 1:
            template_deltas = np.array([
                datetime.timedelta(days=0).total_seconds(),
            ])
        else:
            time_span = coerce_timedelta(time_span).total_seconds()
            min_time = -datetime.timedelta(seconds=time_span).total_seconds()
            max_time = datetime.timedelta(seconds=time_span).total_seconds()
            template_deltas = np.linspace(min_time, max_time, time_window).round().astype(int)
            # Always include a delta of 0
            template_deltas[np.abs(template_deltas).argmin()] = 0
    else:
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


def soft_frame_affinity(unixtimes, sensors=None, time_span='2y', version=1):
    """
    Produce a pairwise affinity weights between frames based on a dilated time
    heuristic.

    Example:
        >>> low = datetime.datetime.now().timestamp()
        >>> high = low + datetime.timedelta(days=365 * 5).total_seconds()
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

    """
    missing_date = np.isnan(unixtimes)
    missing_any_dates = np.any(missing_date)
    have_any_dates = not np.all(missing_date)

    weights = {}

    if have_any_dates:
        # unixtimes[np.random.rand(*unixtimes.shape) > 0.1] = np.nan
        seconds_per_year = datetime.timedelta(days=365).total_seconds()
        seconds_per_day = datetime.timedelta(days=1).total_seconds()

        second_deltas = np.abs(unixtimes[None, :] - unixtimes[:, None])
        year_deltas = second_deltas / seconds_per_year
        day_deltas = second_deltas / seconds_per_day

        # Upweight similar seasons
        season_weights = (1 + np.cos(year_deltas * math.tau)) / 2.0

        # Upweight similar times of day
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
        elif version == 2:
            # TODO:
            # incorporate the time_span?
            time_span = coerce_timedelta(time_span).total_seconds()

            span_delta = (second_deltas - time_span) ** 2
            norm_span_delta = span_delta / (time_span ** 2)
            weights['time_span'] = (1 - np.minimum(norm_span_delta, 1)) * 0.5 + 0.5

            # squash daylight weight influence
            try:
                middle = np.nanmean(daylight_weights)
            except Exception:
                middle = 0

            # Modify the influence of season / daylight
            daylight_weights = (daylight_weights - middle) * 0.1 + (middle / 2)
            season_weights = ((season_weights - 0.5) / 2) + 0.5

        weights['daylight'] = daylight_weights
        weights['season'] = season_weights

        frame_weights = np.prod(np.stack(list(weights.values())), axis=0)
        # frame_weights = season_weights * daylight_weights
        # frame_weights = frame_weights * future_weights
    else:
        frame_weights = None

    if sensors is not None:
        sensors = np.asarray(sensors)
        same_sensor = sensors[:, None] == sensors[None, :]
        sensor_weights = ((same_sensor * 0.5) + 0.5)
        weights['sensor'] = sensor_weights
        if frame_weights is None:
            frame_weights = frame_weights
        else:
            frame_weights = frame_weights * sensor_weights

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


def hard_frame_affinity(unixtimes, sensors, time_window, time_span='2y', blur=False):
    # Hard affinity
    sample_idxs = hard_time_sample_pattern(unixtimes, time_window, time_span=time_span)
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
    affinity = affinity / affinity.max()
    affinity[np.eye(len(affinity), dtype=bool)] = 1
    return affinity


def guess_missing_unixtimes(unixtimes):
    """
    Hueristic solution to fill in missing time values via interpolation /
    extrapolation.
    """
    missing_date = np.isnan(unixtimes)
    missing_any_dates = np.any(missing_date)
    have_any_dates = not np.all(missing_date)

    if missing_any_dates:
        if have_any_dates:
            from scipy import interpolate
            frame_idxs = np.arange(len(unixtimes))
            miss_idxs = frame_idxs[missing_date]
            have_idxs = frame_idxs[~missing_date]
            have_values = unixtimes[have_idxs]
            interp = interpolate.interp1d(have_idxs, have_values, fill_value=np.nan)
            interp_vals = interp(miss_idxs)
            unixtimes = unixtimes.copy()
            unixtimes[miss_idxs] = interp_vals
        else:
            unixtimes = np.linspace(0, len(unixtimes) * 60 * 60 * 24, len(unixtimes))
    return unixtimes


@ub.memoize
def cython_aff_samp_mod():
    """ Old JIT code, no longer works """
    import os
    from watch.tasks.fusion.datamodules import temporal_sampling
    fpath = os.path.join(os.path.dirname(temporal_sampling.__file__), 'affinity_sampling.pyx')
    import xdev
    cython_mod = xdev.import_module_from_pyx(fpath, verbose=0, annotate=True)
    return cython_mod


def show_affinity_sample_process(chosen, info, fnum=1):
    """
    Debugging / demo visualization of the iterative sample algorithm.
    For details see :func:`TimeWindowSampler.show_procedure`.
    """
    # import seaborn as sns
    import kwplot
    # from matplotlib import pyplot as plt
    steps = info['steps']

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
    for x_ in initial_indexes:
        ax.plot([x_, x_], [0, ymax], color='gray')
    ax.plot(np.arange(xmax), probs)
    if idx is not None:
        x, y = idx, probs[idx]
        xpos = x + xmax * 0.0 if x < (xmax / 2) else x - xmax * 0.0
        ypos = y + ymax * 0.3 if y < (ymax / 2) else y - ymax * 0.3
        ax.plot([x, x], [0, ymax], color='gray')
    ax.set_title('Initialize included indices')

    fig = kwplot.figure(pnum=pnum_())
    ax = fig.gca()
    ax.plot(np.arange(xmax), info['initial_update_weights'], color='orange')
    ax.set_title('Initialize Update weights')

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
            ax.plot([x_, x_], [0, ymax], color='gray')
        ax.plot(np.arange(xmax), probs)
        xpos = x + xmax * 0.0 if x < (xmax / 2) else x - xmax * 0.0
        ypos = y + ymax * 0.3 if y < (ymax / 2) else y - ymax * 0.3
        ax.annotate('chosen', (x, y), xytext=(xpos, ypos), color='black', arrowprops=dict(color='orange', arrowstyle='->'))
        ax.plot([x, x], [0, ymax], color='orange')
        #ax.annotate('chosen', (x, y), color='black')
        ax.set_title('Iteration {}: sample'.format(step_idx))

        chosen_so_far.append(idx)

        fig = kwplot.figure(pnum=pnum_())
        ax = fig.gca()
        if step_idx < len(steps):
            ax.plot(np.arange(xmax), step['update_weights'], color='orange')
            #ax.annotate('chosen', (x, y), xytext=(xpos, ypos), color='black', arrowprops=dict(color='black', arrowstyle="->"))
            ax.plot([x, x], [0, step['update_weights'].max()], color='orangered')
            ax.set_title('Iteration {}: update weights'.format(step_idx))
        else:
            for x_ in chosen_so_far:
                ax.plot([x_, x_], [0, ymax], color='gray')
            ax.set_title('Final Sample')

    if _include_summary_row:
        # This last row is not helpful, don't include it.
        affinity = info['affinity']
        fig = kwplot.figure(pnum=pnum_())
        ax = fig.gca()
        for row in affinity[chosen]:
            ax.plot(row)
        ax.set_title('Chosen Affinities')
        # kwplot.imshow(kwimage.normalize(), pnum=pnum_(), title='Chosen Affinities')

        final_mat = affinity[chosen][:, chosen]
        final_mat[np.isnan(final_mat)] = 0
        final_mat = kwimage.normalize(final_mat)
        kwplot.imshow(final_mat, pnum=pnum_(), title='Final Affinities')

    title_suffix = info.get('title_suffix', '')
    fig.suptitle(f'Sample procedure: {start_index}{title_suffix}')
    fig.subplots_adjust(hspace=0.4)
    return fig


def plot_dense_sample_indices(sample_idxs, unixtimes, title_suffix='', linewidths=0):
    """
    Visualization helper
    """
    import seaborn as sns
    import pandas as pd

    dense_sample = kwarray.one_hot_embedding(sample_idxs, len(unixtimes), dim=1).sum(axis=2)
    unixtimes = guess_missing_unixtimes(unixtimes)

    # =====================
    # Show Sample Pattern in heatmap
    datetimes = np.array([datetime.datetime.fromtimestamp(t) for t in unixtimes])
    # dates = np.array([datetime.datetime.fromtimestamp(t).date() for t in unixtimes])
    df = pd.DataFrame(dense_sample)
    df.index.name = 'index'
    df.columns = pd.to_datetime(datetimes).date
    df.columns.name = 'date'
    ax = sns.heatmap(data=df, cbar=False, linewidths=linewidths, linecolor='darkgray')
    ax.set_title('Sample Indexes' + title_suffix)
    ax.set_xlabel('Observation Index')
    ax.set_ylabel('Sample Index')
    return ax


def plot_temporal_sample_indices(sample_idxs, unixtimes, title_suffix=''):
    """
    Visualization helper
    """
    import matplotlib.pyplot as plt
    unixtimes = guess_missing_unixtimes(unixtimes)
    datetimes = np.array([datetime.datetime.fromtimestamp(t) for t in unixtimes])
    # =====================
    # Show Sample Pattern WRT to time
    ax = plt.gca()
    for t in datetimes:
        ax.plot([t, t], [0, len(sample_idxs) + 1], color='darkblue', alpha=0.5)
    for sample_ypos, sample in enumerate(sample_idxs, start=1):
        ax.plot(datetimes[sample], [sample_ypos] * len(sample), '-x')

    ax.set_title('Sample Times' + title_suffix)
    ax.set_xlabel('Time')
    ax.set_ylabel('Sample Index')
    return ax
    # import matplotlib.dates as mdates
    # ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    # ax.xaxis.set_major_locator(mdates.DayLocator(interval=1))
    # ax.figure.autofmt_xdate()


def plot_temporal_sample(affinity, sample_idxs, unixtimes, fnum=1):
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
    plot_temporal_sample_indices(sample_idxs, unixtimes)
