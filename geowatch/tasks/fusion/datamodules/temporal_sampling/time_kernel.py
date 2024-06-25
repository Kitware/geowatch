"""
Defines :class:`TimeKernel`.

Note:
    Kernel may not be the best name for this, which is a highly overloaded
    term. The name was chosen based on the use of a statistical kernels or
    convolutional kernels used in image processing
    [WikiKernelImageProcessing]_.  It has nothing to do with kernel methods in
    optimization or null space in linear algebra.

    It is really an ideal time distribution that will be used as a template.

References:
    .. [WikiKernelImageProcessing] https://en.wikipedia.org/wiki/Kernel_(image_processing)
"""
import numpy as np


class TimeKernel(np.ndarray):
    """
    Represents an ideal relative time sampling pattern.

    This is just an ndarray with offsets specified in seconds.

    Notes:
        https://numpy.org/doc/stable/user/basics.subclassing.html#extra-gotchas-custom-del-methods-and-ndarray-base

    CommandLine:
        xdoctest -m geowatch.tasks.fusion.datamodules.temporal_sampling.time_kernel TimeKernel --show

    Example:
        >>> from geowatch.tasks.fusion.datamodules.temporal_sampling.time_kernel import TimeKernel
        >>> from geowatch.tasks.fusion.datamodules.temporal_sampling.time_kernel import _random_discrete_relative_times
        >>> self = TimeKernel.coerce('-1y,-3m,0,3m,1y')
        >>> print(f'self={self!r}')
        >>> # xdoctest: +REQUIRES(--show)
        >>> import kwplot
        >>> kwplot.autosns()
        >>> # Pretend we have discrete observations with the following relative
        >>> # time differences
        >>> import kwutil
        >>> time_range = kwutil.timedelta.coerce('4y')
        >>> relative_unixtimes = _random_discrete_relative_times(time_range)
        >>> self.plot(relative_unixtimes)
        >>> kwplot.show_if_requested()
    """
    def __new__(cls, *args, **kwargs):
        print('In __new__ with class %s' % cls)
        return super().__new__(cls, *args, **kwargs)

    @classmethod
    def coerce(cls, data):
        from geowatch.tasks.fusion.datamodules.temporal_sampling.utils import coerce_time_kernel
        kernel = np.array(coerce_time_kernel(data))
        self = kernel.view(cls)
        return self

    @classmethod
    def coerce_multiple(cls, data):
        """
        Example:
            >>> from geowatch.tasks.fusion.datamodules.temporal_sampling.time_kernel import TimeKernel
            >>> import ubelt as ub
            >>> pattern = ('-3y', '-2.5y', '-2y', '-1.5y', '-1y', 0, '1y', '1.5y', '2y', '2.5y', '3y')
            >>> multi_kernel = TimeKernel.coerce_multiple(pattern)
            >>> print('multi_kernel = {}'.format(ub.urepr(multi_kernel, nl=2)))
        """
        from geowatch.tasks.fusion.datamodules.temporal_sampling.utils import coerce_multi_time_kernel
        kernels = coerce_multi_time_kernel(data)
        kernels = [np.array(k).view(cls) for k in kernels]
        return kernels

    def make_soft_mask(self, relative_unixtimes):
        from geowatch.tasks.fusion.datamodules.temporal_sampling.affinity import make_soft_mask
        kernel_masks, kernel_attrs = make_soft_mask(self, relative_unixtimes)
        return kernel_masks, kernel_attrs

    def plot(self, relative_unixtimes):
        time_kernel = self
        kernel_masks, kernel_attrs = self.make_soft_mask(relative_unixtimes)

        min_t = min(kattr['left'] for kattr in kernel_attrs)
        max_t = max(kattr['right'] for kattr in kernel_attrs)

        min_t = min(min_t, relative_unixtimes[0])
        max_t = max(max_t, relative_unixtimes[-1])

        import kwplot
        from geowatch.utils import util_kwplot
        plt = kwplot.autoplt()
        import kwimage
        kwplot.close_figures()
        kwplot.figure(fnum=1, doclf=1)
        kernel_color = kwimage.Color.coerce('kitware_green').as01()
        obs_color = kwimage.Color.coerce('kitware_darkblue').as01()

        kwplot.figure(fnum=1, pnum=(1, 1, 1))

        kwplot.phantom_legend({
            'Ideal Sample': kernel_color,
            'Discrete Observation': obs_color,
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

        lw = 2

        obs_line_segments = []
        for x in relative_unixtimes:
            y = 0
            for kattr in kernel_attrs:
                rv = kattr['rv']
                y = max(y, rv.pdf(x) / kattr['_our_norm'])
            obs_line_segments.append([x, y])
        for x, y in obs_line_segments:
            plt.plot([x, x], [0, y], '-', color=obs_color, linewidth=lw)

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
            'Ideal Sample': {'color': kernel_color, 'linestyle': '--'},
            'Discrete Observation': {'color': obs_color, 'linewidth': lw},
        })

        util_kwplot._format_xaxis_as_timedelta(ax)

        # plt.subplots_adjust(top=0.99, bottom=0.1, hspace=.3, left=0.1)
        # fig = plt.gcf()
        # fig.set_size_inches(np.array([4, 3]) * 1.5)
        # fig.tight_layout()
        # finalizer = util_kwplot.FigureFinalizer()
        # finalizer(fig, 'time_sampling_example.png')


def _random_discrete_relative_times(time_range):
    import kwarray
    rng = kwarray.ensure_rng()
    relative_unixtimes = rng.rand(10) * time_range.total_seconds()
    idx = ((relative_unixtimes - (time_range.total_seconds() / 2)) ** 2).argmin()
    mid = relative_unixtimes[idx]
    relative_unixtimes = relative_unixtimes - mid
    return relative_unixtimes
