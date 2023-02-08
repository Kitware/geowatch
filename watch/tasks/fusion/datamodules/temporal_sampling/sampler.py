"""
This module defines our generalized time sampling strategy.

This is used to define our dilated time sampling.

This following doctest illustrates the method on project data.

Example:
    >>> # Basic overview demo of the algorithm
    >>> from watch.tasks.fusion.datamodules.temporal_sampling import *  # NOQA
    >>> import watch
    >>> dset = watch.coerce_kwcoco('watch-msi', geodata=True, dates=True, num_frames=128, image_size=(32, 32))
    >>> vidid = dset.dataset['videos'][0]['id']
    >>> self = TimeWindowSampler.from_coco_video(
    >>>     dset, vidid,
    >>>     time_window=11,
    >>>     affinity_type='soft2', time_span='8m', update_rule='distribute',
    >>> )
    >>> # xdoctest: +REQUIRES(--show)
    >>> import kwplot
    >>> plt = kwplot.autoplt()
    >>> self.show_summary(samples_per_frame=3, fnum=3)
    >>> self.show_procedure(fnum=1)
    >>> plt.subplots_adjust(top=0.9)

Example:
    >>> # Demo multiple different settings
    >>> from watch.tasks.fusion.datamodules.temporal_sampling import *  # NOQA
    >>> import watch
    >>> dset = watch.coerce_kwcoco('watch-msi', geodata=True, dates=True, num_frames=128, image_size=(32, 32))
    >>> vidid = dset.dataset['videos'][0]['id']
    >>> self = TimeWindowSampler.from_coco_video(
    >>>     dset, vidid,
    >>>     time_window=11,
    >>>     affinity_type='uniform', time_span='8m', update_rule='',
    >>> )
    >>> # xdoctest: +REQUIRES(--show)
    >>> import kwplot
    >>> plt = kwplot.autoplt()
    >>> self.update_affinity(affinity_type='contiguous')
    >>> self.show_summary(samples_per_frame=3, fnum=1)
    >>> self.show_procedure(fnum=4)
    >>> plt.subplots_adjust(top=0.9)
    >>> self.update_affinity(affinity_type='soft2')
    >>> self.show_summary(samples_per_frame=3, fnum=2)
    >>> self.show_procedure(fnum=5)
    >>> plt.subplots_adjust(top=0.9)
    >>> self.update_affinity(affinity_type='hardish3')
    >>> self.show_summary(samples_per_frame=3, fnum=3)
    >>> self.show_procedure(fnum=6)
    >>> self.update_affinity(affinity_type='uniform')
    >>> self.show_summary(samples_per_frame=3, fnum=3)
    >>> self.show_procedure(fnum=6)
    >>> plt.subplots_adjust(top=0.9)

Example:
    >>> # Demo corner case where there are too few observations
    >>> from watch.tasks.fusion.datamodules.temporal_sampling import *  # NOQA
    >>> import watch
    >>> dset = watch.coerce_kwcoco('watch-msi', geodata=True, dates=True, num_frames=1, num_videos=1, image_size=(32, 32))
    >>> vidid = dset.dataset['videos'][0]['id']
    >>> self = TimeWindowSampler.from_coco_video(
    >>>     dset, vidid,
    >>>     time_window=2,
    >>>     affinity_type='hardish3', time_span='1y', update_rule='',
    >>> )
    >>> idxs = self.sample()
    >>> print(f'idxs={idxs}')
    idxs=[0, 0]
    >>> # xdoctest: +REQUIRES(--show)
    >>> import kwplot
    >>> plt = kwplot.autoplt()
    >>> self.show_summary(samples_per_frame=3, fnum=1)
    >>> self.show_procedure(fnum=4)
    >>> plt.subplots_adjust(top=0.9)

Example:
    >>> # xdoctest: +REQUIRES(env:SMART_DATA_DVC_DPATH)
    >>> from watch.tasks.fusion.datamodules.temporal_sampling import *  # NOQA
    >>> import watch
    >>> data_dvc_dpath = watch.find_dvc_dpath(tags='phase2_data', hardware='auto')
    >>> coco_fpath = data_dvc_dpath / 'Drop4-BAS/KR_R001.kwcoco.json'
    >>> dset = watch.coerce_kwcoco(coco_fpath)
    >>> vidid = dset.dataset['videos'][0]['id']
    >>> self = TimeWindowSampler.from_coco_video(
    >>>     dset, vidid,
    >>>     time_window=5,
    >>>     affinity_type='hardish3', time_span='3m', update_rule='pairwise+distribute', determenistic=True
    >>> )
    >>> # xdoctest: +REQUIRES(--show)
    >>> import kwplot
    >>> plt = kwplot.autoplt()
    >>> self.show_summary(samples_per_frame=5, fnum=1)
    >>> self.show_procedure(fnum=4)
    >>> plt.subplots_adjust(top=0.9)

Example:
    >>> # xdoctest: +REQUIRES(env:SMART_DATA_DVC_DPATH)
    >>> from watch.tasks.fusion.datamodules.temporal_sampling import *  # NOQA
    >>> import watch
    >>> data_dvc_dpath = watch.find_dvc_dpath(tags='phase2_data', hardware='auto')
    >>> coco_fpath = data_dvc_dpath / 'Drop4-BAS/KR_R001.kwcoco.json'
    >>> dset = watch.coerce_kwcoco(coco_fpath)
    >>> vidid = dset.dataset['videos'][0]['id']
    >>> self = MultiTimeWindowSampler.from_coco_video(
    >>>     dset, vidid,
    >>>     time_window=11,
    >>>     affinity_type='uniform-soft2-hardish3', update_rule=[''], gamma=2,
    >>>     time_span='6m-1y')
    >>> self.sample()
    >>> # xdoctest: +REQUIRES(--show)
    >>> import kwplot
    >>> kwplot.autosns()
    >>> self.show_summary(3)
"""
import kwarray
import numpy as np
import ubelt as ub
import itertools as it
from dateutil import parser
from .plots import plot_dense_sample_indices
from .plots import plot_temporal_sample_indices
from .plots import show_affinity_sample_process
from .affinity import soft_frame_affinity
from .affinity import hard_frame_affinity
from .affinity import affinity_sample


class CommonSamplerMixin:
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

    @classmethod
    def from_datetimes(cls, datetimes, time_span='full', affinity_type='soft2',
                       **kwargs):
        unixtimes = np.array([dt.timestamp() for dt in datetimes])
        if isinstance(time_span, str) and time_span == 'full':
            time_span = max(datetimes) - min(datetimes)
        kwargs['unixtimes'] = unixtimes
        kwargs['sensors'] = None
        kwargs['time_span'] = time_span
        kwargs['affinity_type'] = affinity_type
        self = cls(**kwargs)
        return self


class MultiTimeWindowSampler(CommonSamplerMixin):
    """
    A wrapper that contains multiple time window samplers with different
    affinity matrices to increase the diversity of temporal sampling.

    Example:
        >>> from watch.tasks.fusion.datamodules.temporal_sampling import *  # NOQA
        >>> import datetime as datetime_mod
        >>> from datetime import datetime as datetime_cls
        >>> low = datetime_cls.now().timestamp()
        >>> high = low + datetime_mod.timedelta(days=365 * 5).total_seconds()
        >>> rng = kwarray.ensure_rng(0)
        >>> unixtimes = np.array(sorted(rng.randint(low, high, 113)), dtype=float)
        >>> sensors = ['a' for _ in range(len(unixtimes))]
        >>> time_window = 5
        >>> self = MultiTimeWindowSampler(
        >>>     unixtimes=unixtimes, sensors=sensors, time_window=time_window, update_rule='pairwise+distribute',
        >>>     #time_span=['2y', '1y', '5m'])
        >>>     time_span='7d-1m', affinity_type='soft2')
        >>> self.sample()
        >>> # xdoctest: +REQUIRES(--show)
        >>> import kwplot
        >>> kwplot.autosns()
        >>> self.show_summary(10)
    """

    def __init__(self, unixtimes, sensors, time_window, affinity_type='hard',
                 update_rule='distribute', determenistic=False, gamma=1,
                 time_span=['2y', '1y', '5m'], name='?'):

        if isinstance(time_span, str):
            time_span = time_span.split('-')
        if isinstance(affinity_type, str):
            affinity_type = affinity_type.split('-')
        if isinstance(update_rule, str):
            update_rule = update_rule.split('-')

        if len(update_rule) == 0:
            update_rule = ['']

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
        self.sub_samplers = {}
        self._build()

    def _build(self):
        for time_span, affinity_type, update_rule in it.product(self.time_span, self.affinity_type, self.update_rule):
            sub_sampler = TimeWindowSampler(
                unixtimes=self.unixtimes,
                sensors=self.sensors,
                time_window=self.time_window,
                update_rule=update_rule,
                affinity_type=affinity_type,
                determenistic=self.determenistic,
                gamma=self.gamma,
                name=self.name,
                time_span=time_span,
            )
            key = ':'.join([time_span, affinity_type, update_rule])
            self.sub_samplers[key] = sub_sampler
            self.indexes = sub_sampler.indexes
        # self.indexes = np.arange(self.affinity.shape[0])

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
        affinity = np.mean(np.stack([
            sampler.affinity
            for sampler in self.sub_samplers.values()
        ]), axis=0)
        return affinity

    @affinity.setter
    def affinity(self, value):
        # Define setter for backwards compatibility
        if len(self.sub_samplers) > 1:
            raise Exception(
                'no way to hack affinity directly when there is '
                'more than one subsampler')
        sub_sampler = ub.peek(self.sub_samplers.values())
        sub_sampler.affinity = value

    def show_summary(self, samples_per_frame=1, show_indexes=0, fnum=1):
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

        sample_idxs = np.array(sample_idxs)

        title_info = ub.codeblock(
            f'''
            name={self.name}
            affinity_type={self.affinity_type} determenistic={self.determenistic}
            update_rule={self.update_rule} gamma={self.gamma}
            ''')

        pnum_ = kwplot.PlotNums(nCols=2 + show_indexes)

        fig = kwplot.figure(fnum=fnum, doclf=True)

        fig = kwplot.figure(fnum=fnum, pnum=pnum_())
        ax = fig.gca()

        affinity = self.affinity

        kwplot.imshow(affinity, ax=ax)
        ax.set_title('combined frame affinity')

        if show_indexes:
            fig = kwplot.figure(fnum=fnum, pnum=pnum_())
            if samples_per_frame < 2:
                ax = plot_dense_sample_indices(sample_idxs, self.unixtimes, linewidths=0.1)
                ax.set_aspect('equal')
            else:
                ax = plot_dense_sample_indices(sample_idxs, self.unixtimes, linewidths=0.001)

        kwplot.figure(fnum=fnum, pnum=pnum_())
        plot_temporal_sample_indices(sample_idxs, self.unixtimes)
        fig.suptitle(title_info)


class TimeWindowSampler(CommonSamplerMixin):
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

    Attributes:
        main_indexes

    Example:
        >>> # xdoctest: +REQUIRES(env:SMART_DATA_DVC_DPATH)
        >>> import os
        >>> from watch.tasks.fusion.datamodules.temporal_sampling import *  # NOQA
        >>> import kwcoco
        >>> import watch
        >>> data_dvc_dpath = watch.find_dvc_dpath(tags='phase2_data', hardware='auto')
        >>> coco_fpath = data_dvc_dpath / 'Drop4-SC/data_vali.kwcoco.json'
        >>> dset = kwcoco.CocoDataset(coco_fpath)
        >>> vidid = dset.dataset['videos'][0]['id']
        >>> self = TimeWindowSampler.from_coco_video(
        >>>     dset, vidid,
        >>>     time_window=5,
        >>>     affinity_type='hardish3', time_span='1y',
        >>>     update_rule='distribute')
        >>> self.determenistic = False
        >>> self.show_summary(samples_per_frame=1, fnum=1)
        >>> self.determenistic = True
        >>> self.show_summary(samples_per_frame=3, fnum=2)
    """

    def __init__(self, unixtimes, sensors, time_window,
                 affinity_type='hard', update_rule='distribute',
                 determenistic=False, gamma=1, time_span='2y',
                 affkw=None, name='?'):
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
        self.affkw = affkw  # extra args to affinity matrix building

        self.compute_affinity()

    def update_affinity(self, affinity_type=None, update_rule=None):
        """
        Construct the affinity matrix given the current ``affinity_type``.

        Example:
            >>> # xdoctest: +REQUIRES(env:SMART_DATA_DVC_DPATH)
            >>> import os
            >>> from watch.tasks.fusion.datamodules.temporal_sampling import *  # NOQA
            >>> import watch
            >>> data_dvc_dpath = watch.find_dvc_dpath(tags='phase2_data', hardware='auto')
            >>> coco_fpath = dvc_dpath / 'Drop4-SC/data_vali.kwcoco.json'
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
        if self.affkw is None:
            self.affkw = {}

        if affinity_type is not None:
            self.affinity_type = affinity_type

        if update_rule is not None:
            self.update_rule = update_rule

        # update_rules = set(update_rule.split('+'))
        # config = ub.dict_subset({'pairwise': True, 'distribute': True}, update_rules)
        # do_pairwise = config.get('pairwise', False)
        # do_distribute = config.get('distribute', False)

        if self.affinity_type.startswith('uniform'):
            n = len(self.unixtimes)
            self.affinity = np.full((n, n), fill_value=1 / n, dtype=np.float32)
        elif self.affinity_type.startswith('soft'):
            if self.affinity_type == 'soft':
                version = 1
            else:
                version = int(self.affinity_type[4:])
            # Soft affinity
            self.affinity = soft_frame_affinity(self.unixtimes, self.sensors,
                                                self.time_span,
                                                version=version,
                                                **self.affkw)['final']
        elif self.affinity_type == 'hard':
            # Hard affinity
            self.affinity = hard_frame_affinity(self.unixtimes, self.sensors,
                                                time_window=self.time_window,
                                                blur=False,
                                                time_span=self.time_span,
                                                **self.affkw)
        elif self.affinity_type == 'hardish':
            # Hardish affinity
            self.affinity = hard_frame_affinity(self.unixtimes, self.sensors,
                                                time_window=self.time_window,
                                                blur=True,
                                                time_span=self.time_span,
                                                **self.affkw)
        elif self.affinity_type == 'hardish2':
            # Hardish affinity
            self.affinity = hard_frame_affinity(self.unixtimes, self.sensors,
                                                time_window=self.time_window,
                                                blur=3.0,
                                                time_span=self.time_span,
                                                **self.affkw)
        elif self.affinity_type == 'hardish3':
            # Hardish affinity
            self.affinity = hard_frame_affinity(self.unixtimes, self.sensors,
                                                time_window=self.time_window,
                                                blur=6.0,
                                                time_span=self.time_span,
                                                **self.affkw)
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

        self.indexes = np.arange(self.affinity.shape[0])

    compute_affinity = update_affinity

    # TODO: deprecate wrapper for moving APIs
    __todo__ = '''
    def deprecate_alias(func):
        def _wrapper(*args, **kwargs):
            ub.schedule_deprecation(...)
            return func(*args, **kwargs)
        return _wrapper
    compute_affinity = deprecate_alias(update_affinity)
    '''

    @property
    def main_indexes(self):
        ub.schedule_deprecation(
            'watch', 'main_indexes', 'use indexes instead', deprecate='now')
        return self.indexes

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
            >>> # xdoctest: +REQUIRES(env:SMART_DATA_DVC_DPATH)
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

        Example:
            >>> import os
            >>> import kwcoco
            >>> from watch.tasks.fusion.datamodules.temporal_sampling import *  # NOQA
            >>> import watch
            >>> dset = watch.coerce_kwcoco('watch-msi', geodata=True, dates=True, num_frames=128, image_size=(32, 32))
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

        Ignore:
            import xdev
            globals().update(xdev.get_func_kwargs(TimeWindowSampler.sample))
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

    def show_summary(self, samples_per_frame=1, fnum=1, show_indexes=False,
                     with_temporal=True, compare_determ=True, title_suffix=''):
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
            >>> # xdoctest: +REQUIRES(env:SMART_DATA_DVC_DPATH)
            >>> import os
            >>> from watch.tasks.fusion.datamodules.temporal_sampling import *  # NOQA
            >>> from watch.utils.util_data import find_smart_dvc_dpath
            >>> import kwcoco
            >>> # xdoctest: +REQUIRES(--show)
            >>> dvc_dpath = find_smart_dvc_dpath()
            >>> coco_fpath = dvc_dpath / 'Aligned-Drop4-2022-07-28-c20-TA1-S2-L8-ACC/data_vali.kwcoco.json'
            >>> dset = kwcoco.CocoDataset(coco_fpath)
            >>> video_ids = list(ub.sorted_vals(dset.index.vidid_to_gids, key=len).keys())
            >>> vidid = video_ids[2]
            >>> # Demo behavior over a grid of parameters
            >>> grid = list(ub.named_product({
            >>>     'affinity_type': ['hard', 'soft2', 'hardish3', 'hardish2'],
            >>>     'update_rule': ['distribute', 'pairwise+distribute'][0:1],
            >>>     #'determenistic': [False, True],
            >>>     'determenistic': [False],
            >>>     'time_window': [5],
            >>> }))
            >>> import kwplot
            >>> kwplot.autompl()
            >>> for idx, kwargs in enumerate(grid):
            >>>     print('kwargs = {!r}'.format(kwargs))
            >>>     self = TimeWindowSampler.from_coco_video(dset, vidid, **kwargs)
            >>>     self.show_summary(samples_per_frame=30, fnum=idx, show_indexes=False, determenistic=True)
        """
        import kwplot
        kwplot.autompl()

        if compare_determ:
            _prev_determenistic = self.determenistic
            self.determenistic = False

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
            affinity_type={self.affinity_type}
            update_rule={self.update_rule} gamma={self.gamma}
            ''') + title_suffix
        # determenistic={self.determenistic}

        with_mat = True

        n_temp_plots = (compare_determ + 1) * with_temporal

        num_subplots = (with_mat + show_indexes + n_temp_plots)

        pnum_ = kwplot.PlotNums(nCols=num_subplots)

        fig = kwplot.figure(fnum=fnum, doclf=True)

        fig = kwplot.figure(fnum=fnum, pnum=pnum_())
        ax = fig.gca()
        kwplot.imshow(self.affinity, ax=ax, cmap='viridis')
        ax.set_title('frame affinity')

        if show_indexes:
            fig = kwplot.figure(fnum=fnum, pnum=pnum_())
            if samples_per_frame < 5:
                ax = plot_dense_sample_indices(sample_idxs, self.unixtimes, linewidths=0.1)
                ax.set_aspect('equal')
            else:
                ax = plot_dense_sample_indices(sample_idxs, self.unixtimes, linewidths=0.001)

        if with_temporal:
            kwplot.figure(fnum=fnum, pnum=pnum_())
            plot_temporal_sample_indices(sample_idxs, self.unixtimes, sensors=self.sensors, title_suffix=': non-determenistic')

            if compare_determ:
                self.determenistic = True
                sample_idxs = []
                for idx in range(self.affinity.shape[0]):
                    idxs = self.sample(idx)
                    sample_idxs.append(idxs)
                kwplot.figure(fnum=fnum, pnum=pnum_())
                plot_temporal_sample_indices(sample_idxs, self.unixtimes, sensors=self.sensors, title_suffix=': determenistic')

        fig.suptitle(title_info)

        if compare_determ:
            self.determenistic = _prev_determenistic

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
            >>> # xdoctest: +REQUIRES(env:SMART_DATA_DVC_DPATH)
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

        Ignore:
            import xdev
            globals().update(xdev.get_func_kwargs(TimeWindowSampler.show_procedure))
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
