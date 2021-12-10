"""
Compute intensity histograms of the underlying images in this dataset.

Example:
    smartwatch intensity_histograms --src special:vidshapes8-msi --show=True --stat=density
    smartwatch intensity_histograms --src special:photos --show=True --fill=False
    smartwatch intensity_histograms --src special:shapes8 --show=True --stat=count --cumulative=True --multiple=stack

    DVC_DPATH=$HOME/data/dvc-repos/smart_watch_dvc
    KWCOCO_FPATH=$DVC_DPATH/Drop1-Aligned-L1/vali_data_nowv.kwcoco.json
    smartwatch intensity_histograms --src $KWCOCO_FPATH --show=True --show=True
"""
import kwcoco
import pickle
import kwarray
import ubelt as ub
import scriptconfig as scfg
import pathlib
import kwimage
import pandas as pd
import numpy as np


class IntensityHistogramConfig(scfg.Config):
    """
    Updates image transforms in a kwcoco json file to align all videos to a
    target GSD.
    """
    default = {
        'src': scfg.Value('in.geojson.json', help='input dataset to chip'),

        'dst': scfg.Value(None, help='if specified dump the figure to disk at this path'),

        'show': scfg.Value(False, help='if True, do a plt.show()'),

        'workers': scfg.Value(0, help='number of io workers'),
        'mode': scfg.Value('process', help='type of parallelism'),

        'include_sensors': scfg.Value(None, help='if specified can be comma separated valid sensors'),
        'exclude_sensors': scfg.Value(None, help='if specified can be comma separated invalid sensors'),

        # Histogram modifiers
        'kde': scfg.Value(True, help='if True compute a kernel density estimate to smooth the distribution'),
        'cumulative': scfg.Value(False, help='If True, plot the cumulative counts as bins increase.'),

        # 'bins': scfg.Value(256, help='Generic bin parameter that can be the name of a reference rule or the number of bins.'),
        'bins': scfg.Value('auto', help='Generic bin parameter that can be the name of a reference rule or the number of bins.'),

        'fill': scfg.Value(True, help='If True, fill in the space under the histogram.'),
        'element': scfg.Value('step', help='Visual representation of the histogram statistic.', choices=['bars', 'step', 'poly']),
        'multiple': scfg.Value('layer', choices=['layer', 'dodge', 'stack', 'fill']
                               , help='Approach to resolving multiple elements when semantic mapping creates subsets.'),

        'stat': scfg.Value('density', choices={'count', 'frequency', 'density', 'probability'}, help=ub.paragraph(
            '''
            Aggregate statistic to compute in each bin.

            - ``count`` shows the number of observations
            - ``frequency`` shows the number of observations divided by the bin width
            - ``density`` normalizes counts so that the area of the histogram is 1
            - ``probability`` normalizes counts so that the sum of the bar heights is 1
            '''))
    }


class HistAccum:
    """
    Helper to accumulate histograms
    """
    def __init__(self):
        self.accum = {}
        self.n = 0

    def update(self, data, sensor, channel):
        self.n += 1
        if sensor not in self.accum:
            self.accum[sensor] = {}
        sensor_accum = self.accum[sensor]
        if channel not in sensor_accum:
            sensor_accum[channel] = ub.ddict(lambda: 0)
        final_accum = sensor_accum[channel]
        for k, v in data.items():
            final_accum[k] += v


def main(**kwargs):
    r"""
    Example:
        >>> from watch.cli.coco_intensity_histograms import *  # NOQA
        >>> import kwcoco
        >>> test_dpath = ub.ensure_app_cache_dir('watch/tests')
        >>> image_fpath = test_dpath + '/intensityhist_demo.jpg'
        >>> coco_dset = kwcoco.CocoDataset.demo('vidshapes8-multispectral')
        >>> kwargs = {'src': coco_dset, 'dst': image_fpath, 'mode': 'thread'}
        >>> kwargs['multiple'] = 'layer'
        >>> kwargs['element'] = 'step'
        >>> main(**kwargs)
    """
    from watch.utils import kwcoco_extensions
    from watch.utils.lightning_ext import util_globals
    import kwplot
    kwplot.autosns()

    config = IntensityHistogramConfig(kwargs, cmdline=True)
    coco_dset = kwcoco.CocoDataset.coerce(config['src'])

    valid_gids = kwcoco_extensions.filter_image_ids(
        coco_dset,
        include_sensors=config['include_sensors'],
        exclude_sensors=config['exclude_sensors'],
    )

    images = coco_dset.images(valid_gids)
    workers = util_globals.coerce_num_workers(config['workers'])

    jobs = ub.JobPool(mode='process', max_workers=workers)
    for coco_img in ub.ProgIter(images.coco_images, desc='submit stats jobs'):
        coco_img.detach()
        job = jobs.submit(ensure_intensity_stats, coco_img)
        job.coco_img = coco_img

    accum = HistAccum()
    for job in jobs.as_completed(desc='accumulate stats'):
        intensity_stats = job.result()
        sensor = job.coco_img.get('sensor_coarse', 'unknown_sensor')
        for band_stats in intensity_stats['bands']:
            band_name = band_stats['band_name']
            intensity_hist = band_stats['intensity_hist']
            accum.update(intensity_hist, sensor, band_name)

    fig = plot_intensity_histograms(accum, config)

    dst_fpath = config['dst']
    if dst_fpath is not None:
        fig.set_size_inches(np.array([6.4, 4.8]) * 1.68)
        fig.tight_layout()
        fig.savefig(dst_fpath)

    if config['show']:
        from matplotlib import pyplot as plt
        plt.show()


def _fill_missing_colors(label_to_color):
    """
    label_to_color = {'foo': kwimage.Color('red').as01(), 'bar': None}
    """
    from distinctipy import distinctipy
    import kwarray
    import numpy as np
    given = {k: kwimage.Color(v).as01() for k, v in label_to_color.items() if v is not None}
    needs_color = sorted(set(label_to_color) - set(given))

    seed = 6777939437
    # hack in our code
    def _patched_get_random_color(pastel_factor=0, rng=None):
        rng = kwarray.ensure_rng(seed, api='python')
        color = [(rng.random() + pastel_factor) / (1.0 + pastel_factor) for _ in range(3)]
        return tuple(color)
    distinctipy.get_random_color = _patched_get_random_color

    exclude_colors = [
        tuple(map(float, (d, d, d)))
        for d in np.linspace(0, 1, 5)
    ] + list(given.values())

    final = given.copy()
    new_colors = distinctipy.get_colors(len(needs_color), exclude_colors=exclude_colors)
    for key, new_color in zip(needs_color, new_colors):
        final[key] = tuple(map(float, new_color))
    return final


def ensure_intensity_sidecar(fpath, recompute=False):
    """
    Write statistics next to the image
    """
    stats_fpath = pathlib.Path(fpath + '.stats.pkl')

    if recompute or not stats_fpath.exists():
        imdata = kwimage.imread(fpath, backend='gdal')
        imdata = kwarray.atleast_nd(imdata, 3)
        # TODO: better float handling
        # if imdata.dtype.kind == 'f':
        stats_info = {'bands': []}
        for imband in imdata.transpose(2, 0, 1):
            data = imband.ravel()
            intensity_hist = ub.dict_hist(data)
            intensity_hist = ub.sorted_keys(intensity_hist)
            stats_info['bands'].append({
                'intensity_hist': intensity_hist,
            })
        with open(stats_fpath, 'wb') as file:
            pickle.dump(stats_info, file)
    return stats_fpath


def ensure_intensity_stats(coco_img, recompute=False):
    from os.path import join
    intensity_stats = {'bands': []}
    for obj in coco_img.iter_asset_objs():
        fpath = join(coco_img.bundle_dpath, obj['file_name'])
        channels = obj.get('channels', None)
        if channels is None:
            shape = kwimage.load_image_shape(fpath)
            num_channels = shape[2]
            if num_channels == 3:
                channels = 'r|g|b'
            elif num_channels == 4:
                channels = 'r|g|b|a'
            elif num_channels == 1:
                channels = 'r|g|b'
            else:
                channels = kwcoco.FusedChannelSpec.coerce(num_channels)

        channels = kwcoco.FusedChannelSpec.coerce(channels).as_list()
        stats_fpath = ensure_intensity_sidecar(fpath, recompute=recompute)
        with open(stats_fpath, 'rb') as file:
            stat_info = pickle.load(file)

        obj.get('channels', None)

        for band_idx, band_stat in enumerate(stat_info['bands']):
            try:
                band_name = channels[band_idx]
            except IndexError:
                print('bad channel declaration fpath = {!r}'.format(fpath))
                if 0:
                    print('obj = {}'.format(ub.repr2(obj, nl=1)))
                    print('coco_img = {!r}'.format(coco_img))
                    print('fpath = {!r}'.format(fpath))
                    print('stats_fpath = {!r}'.format(stats_fpath))
                    print(len(stat_info['bands']))
                    print('band_idx = {!r}'.format(band_idx))
                    print('channels = {!r}'.format(channels))
                # raise
                band_name = 'unknown'
            band_stat['band_name'] = band_name
            intensity_stats['bands'].append(band_stat)
    return intensity_stats


def plot_intensity_histograms(accum, config):
    import kwplot
    sns = kwplot.autosns()

    # Stack all accuulated histograms into a longform dataframe
    to_stack = {}
    unique_channels = set()
    unique_sensors = set()
    for sensor, sub in accum.accum.items():
        unique_sensors.add(sensor)
        for channel, hist in sub.items():
            unique_channels.add(channel)
            hist = ub.sorted_keys(hist)
            hist.pop(0)
            df = pd.DataFrame({
                'intensity_bin': np.array(list(hist.keys())),
                'value': np.array(list(hist.values())),
                'channel': [channel] * len(hist),
                'sensor': [sensor] * len(hist),
            })
            to_stack[(channel, sensor)] = df

    full_df = pd.concat(list(to_stack.values()))
    is_finite = np.isfinite(full_df.intensity_bin)
    num_nonfinite = (~is_finite).sum()
    if num_nonfinite > 0:
        import warnings
        warnings.warn('There were {} non-finite values'.format(num_nonfinite))
        full_df = full_df[is_finite]

    # sns.lineplot(full_df=df, x='intensity_bin', y='value')
    # max_val = np.iinfo(np.uint16).max
    # import kwimage

    palette = {
        # 'red': kwimage.Color('red').as01(),
        # 'blue': kwimage.Color('blue').as01(),
        # 'green': kwimage.Color('green').as01(),
        # 'cirrus': kwimage.Color('skyblue').as01(),
        # 'coastal': kwimage.Color('purple').as01(),
        # 'nir': kwimage.Color('orange').as01(),
        # 'swir16': kwimage.Color('pink').as01(),
        # 'swir22': kwimage.Color('hotpink').as01(),
        'red': 'red',
        'blue': 'blue',
        'green': 'green',
        'cirrus': 'skyblue',
        'coastal': 'purple',
        'nir': 'orange',
        'swir16': 'pink',
        'swir22': 'hotpink',
    }
    if 'red' not in unique_channels and 'r' in unique_channels:
        # hack
        palette['r'] = 'red'
        palette['g'] = 'green'
        palette['b'] = 'blue'
    for channel in unique_channels:
        if channel not in palette:
            palette[channel] = None
    palette = _fill_missing_colors(palette)

    __hisplot_notes__ = """
    # TODO: play with these:

    binwidth : number or pair of numbers
        Width of each bin, overrides ``bins`` but can be used with
        ``binrange``.
    binrange : pair of numbers or a pair of pairs
        Lowest and highest value for bin edges; can be used either
        with ``bins`` or ``binwidth``. Defaults to data extremes.

    discrete : bool
        If True, default to ``binwidth=1`` and draw the bars so that they are
        centered on their corresponding data points. This avoids "gaps" that may
        otherwise appear when using discrete (integer) data.
    common_bins : bool
        If True, use the same bins when semantic variables produce multiple
        plots. If using a reference rule to determine the bins, it will be computed
        with the full dataset.

    common_norm : bool
        If True and using a normalized statistic, the normalization will apply over
        the full dataset. Otherwise, normalize each histogram independently.

    fill : bool
        If True, fill in the space under the histogram.
    shrink : number
        Scale the width of each bar relative to the binwidth by this factor.

    pthresh : number or None
        Like ``thresh``, but a value in [0, 1] such that cells with aggregate counts
        (or other statistics, when used) up to this proportion of the total will be
        transparent.

    color : :mod:`matplotlib color <matplotlib.colors>`
        Single color specification for when hue mapping is not used. Otherwise, the
        plot will try to hook into the matplotlib property cycle.
    log_scale : bool or number, or pair of bools or numbers
        Set a log scale on the data axis (or axes, with bivariate data) with the
        given base (default 10), and evaluate the KDE in log space.

    # Probably ignorable

    kde_kws : dict
        Parameters that control the KDE computation, as in :func:`kdeplot`.
    line_kws : dict
        Parameters that control the KDE visualization, passed to
        :meth:`matplotlib.axes.Axes.plot`.

    # Ignorable
    thresh : number or None
        Cells with a statistic less than or equal to this value will be transparent.
        Only relevant with bivariate data.
    legend : bool
        If False, suppress the legend for semantic variables.
    ax : :class:`matplotlib.axes.Axes`
        Pre-existing axes for the plot. Otherwise, call :func:`matplotlib.pyplot.gca`
        internally.
    cbar : bool
        If True, add a colorbar to annotate the color mapping in a bivariate plot.
        Note: Does not currently support plots with a ``hue`` variable well.
    cbar_ax : :class:`matplotlib.axes.Axes`
        Pre-existing axes for the colorbar.
    cbar_kws : dict
        Additional parameters passed to :meth:`matplotlib.figure.Figure.colorbar`.
    pmax : number or None
        A value in [0, 1] that sets that saturation point for the colormap at a value
        such that cells below is constistute this proportion of the total count (or
        other statistic, when used).
    palette : string, list, dict, or :class:`matplotlib.colors.Colormap`
        Method for choosing the colors to use when mapping the ``hue`` semantic.
        String values are passed to :func:`color_palette`. List or dict values
        imply categorical mapping, while a colormap object implies numeric mapping.
    hue_order : vector of strings
        Specify the order of processing and plotting for categorical levels of the
        ``hue`` semantic.
    hue_norm : tuple or :class:`matplotlib.colors.Normalize`
        Either a pair of values that set the normalization range in data units
        or an object that will map from data units into a [0, 1] interval. Usage
        implies numeric mapping.
    """
    __hisplot_notes__
    hist_data_kw = dict(
        x='intensity_bin',
        weights='value',
        bins=config['bins'],
        stat=config['stat'],
        hue='channel',
    )
    hist_style_kw = dict(
        palette=palette,
        fill=config['fill'],
        element=config['element'],
        multiple=config['multiple'],
        kde=config['kde'],
        cumulative=config['cumulative'],
    )

    #  For S2 that is supposed to be divide by 10000.  For L8 it is multiply by 2.75e-5 and subtract 0.2.
    # 1 / 2.75e-5
    fig = kwplot.figure(fnum=1, doclf=True)
    pnum_ = kwplot.PlotNums(nSubplots=len(unique_sensors))
    for sensor_name, sensor_df in full_df.groupby('sensor'):

        info_rows = []
        for channel, chan_df in sensor_df.groupby('channel'):
            info = {
                'min': chan_df.intensity_bin.min(),
                'max': chan_df.intensity_bin.max(),
                'channel': channel,
                'sensor': sensor_name,
            }
            info_rows.append(info)
        print(pd.DataFrame(info_rows))

        ax = kwplot.figure(fnum=1, pnum=pnum_()).gca()
        sns.histplot(ax=ax, data=sensor_df, **hist_data_kw, **hist_style_kw)
        ax.set_title(sensor_name)
        # maxx = sensor_df.intensity_bin.max()
        # maxx = sensor_maxes[sensor_name]
        # ax.set_xlim(0, maxx)
    return fig


_SubConfig = IntensityHistogramConfig

if __name__ == '__main__':
    main()
