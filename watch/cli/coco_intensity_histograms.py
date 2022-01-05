"""
Compute intensity histograms of the underlying images in this dataset.

CommandLine:
    smartwatch intensity_histograms --src special:watch-msi --show=True --stat=density
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
import math


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

        'include_channels': scfg.Value(None, help='if specified can be | separated valid channels'),
        'exclude_channels': scfg.Value(None, help='if specified can be | separated invalid channels'),

        'include_sensors': scfg.Value(None, help='if specified can be comma separated valid sensors'),
        'exclude_sensors': scfg.Value(None, help='if specified can be comma separated invalid sensors'),

        'max_images': scfg.Value(None, help='if given only sample this many images when computing statistics'),

        'valid_range': scfg.Value(None, help='Only include values within this range; specified as <min_val>:<max_val> e.g. (0:10000)'),

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

    def finalize(self):
        # Stack all accuulated histograms into a longform dataframe
        to_stack = {}
        for sensor, sub in self.accum.items():
            for channel, hist in sub.items():
                hist = ub.sorted_keys(hist)
                # hist.pop(0)
                df = pd.DataFrame({
                    'intensity_bin': np.array(list(hist.keys()), dtype=int),
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
        full_df = full_df.reset_index()
        return full_df


def demo_kwcoco_multisensor(num_videos=4, num_frames=10, **kwargs):
    """
    Note:
        dev/flow21 has main implementation. remove this after this is merged

    Ignore:
        import watch
        coco_dset = watch.demo.demo_kwcoco_multisensor()
        coco_dset = watch.demo.demo_kwcoco_multisensor(max_speed=0.5)
    """
    demo_kwargs = {
        'num_frames': num_frames,
        'num_videos': num_videos,
        'rng': 9111665008,
        'multisensor': True,
        'multispectral': True,
        'image_size': 'random',
    }
    demo_kwargs.update(kwargs)
    coco_dset = kwcoco.CocoDataset.demo('vidshapes', **demo_kwargs)
    # Hack in sensor_coarse
    images = coco_dset.images()
    groups = ub.sorted_keys(ub.group_items(images.coco_images, lambda x: x.channels.spec))
    for idx, (k, g) in enumerate(groups.items()):
        for coco_img in g:
            coco_img.img['sensor_coarse'] = 'sensor{}'.format(idx)
    return coco_dset


def coerce_kwcoco(data='watch-msi', **kwargs):
    """
    Note:
        dev/flow21 has main implementation. remove this after this is merged

    coerce with watch special datasets
    """
    if isinstance(data, str) and 'watch' in data.split('special:', 1)[-1].split('-'):
        return demo_kwcoco_multisensor(**kwargs)
    else:
        return kwcoco.CocoDataset.coerce(data, **kwargs)


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

    Example:
        >>> from watch.cli.coco_intensity_histograms import *  # NOQA
        >>> import kwcoco
        >>> import watch
        >>> test_dpath = ub.ensure_app_cache_dir('watch/tests')
        >>> image_fpath = test_dpath + '/intensityhist_demo2.jpg'
        >>> coco_dset = coerce_kwcoco('watch-msi')
        >>> kwargs = {
        >>>     'src': coco_dset,
        >>>     'dst': image_fpath,
        >>>     'mode': 'thread',
        >>>     'valid_range': '10:2000',
        >>> }
        >>> kwargs['multiple'] = 'layer'
        >>> kwargs['element'] = 'step'
        >>> main(**kwargs)
    """
    from watch.utils import kwcoco_extensions
    from watch.utils.lightning_ext import util_globals
    import kwplot
    kwplot.autosns()

    config = IntensityHistogramConfig(kwargs, cmdline=True)
    print('config = {}'.format(ub.repr2(config.to_dict(), nl=1)))

    # coco_dset = kwcoco.CocoDataset.coerce(config['src'])
    coco_dset = coerce_kwcoco(config['src'])

    valid_gids = kwcoco_extensions.filter_image_ids(
        coco_dset,
        include_sensors=config['include_sensors'],
        exclude_sensors=config['exclude_sensors'],
    )

    images = coco_dset.images(valid_gids)
    workers = util_globals.coerce_num_workers(config['workers'])
    print('workers = {!r}'.format(workers))

    if config['max_images'] is not None:
        print('images = {!r}'.format(images))
        images = coco_dset.images(list(images)[:int(config['max_images'])])
        print('filter images = {!r}'.format(images))

    include_channels = config['include_channels']
    exclude_channels = config['exclude_channels']
    include_channels = None if include_channels is None else kwcoco.FusedChannelSpec.coerce(include_channels)
    exclude_channels = None if exclude_channels is None else kwcoco.FusedChannelSpec.coerce(exclude_channels)

    jobs = ub.JobPool(mode=config['mode'], max_workers=workers)
    for coco_img in ub.ProgIter(images.coco_images, desc='submit stats jobs'):
        coco_img.detach()
        job = jobs.submit(ensure_intensity_stats, coco_img,
                          include_channels=include_channels,
                          exclude_channels=exclude_channels)
        job.coco_img = coco_img

    if config['valid_range'] is not None:
        valid_min, valid_max = map(float, config['valid_range'].split(':'))
    else:
        valid_min = -math.inf
        valid_max = math.inf

    accum = HistAccum()
    for job in jobs.as_completed(desc='accumulate stats'):
        intensity_stats = job.result()
        sensor = job.coco_img.get('sensor_coarse', 'unknown_sensor')
        for band_stats in intensity_stats['bands']:
            band_name = band_stats['band_name']
            intensity_hist = band_stats['intensity_hist']
            intensity_hist = {k: v for k, v in intensity_hist.items()
                              if k >= valid_min and k <= valid_max}
            accum.update(intensity_hist, sensor, band_name)

    full_df = accum.finalize()

    COMPARSE_SENSORS = True
    if COMPARSE_SENSORS:
        distance_metrics = compare_sensors(full_df)
        request_columns = ['emd', 'energy_dist', 'mean_diff', 'std_diff']
        have_columns = list(ub.oset(request_columns) & ub.oset(distance_metrics.columns))
        harmony_scores = distance_metrics[have_columns].mean()
        extra_text = ub.repr2(harmony_scores.to_dict(), precision=4, compact=1)
        print('extra_text = {!r}'.format(extra_text))
    else:
        extra_text = None

    fig = plot_intensity_histograms(full_df, config)

    if extra_text is not None:
        fig.suptitle(extra_text)

    dst_fpath = config['dst']
    if dst_fpath is not None:
        print('dump to dst_fpath = {!r}'.format(dst_fpath))
        fig.set_size_inches(np.array([6.4, 4.8]) * 1.68)
        fig.tight_layout()
        fig.savefig(dst_fpath)

    if config['show']:
        from matplotlib import pyplot as plt
        plt.show()


def compare_sensors(full_df):
    import itertools as it
    import scipy
    import scipy.stats

    distance_metrics = []
    sensor_channel_to_vw = {}
    for channel, chan_df in full_df.groupby('channel'):
        for _sensor, sensor_df in chan_df.groupby('sensor'):
            _values = sensor_df['intensity_bin']
            _weights = sensor_df['value']
            sensor_channel_to_vw[(_sensor, channel)] = (_values, _weights)

    print('comparing sensors')
    print(ub.repr2(list(sensor_channel_to_vw.keys()), nl=1))
    chan_to_group = ub.group_items(
        sensor_channel_to_vw.keys(),
        [t[1] for t in sensor_channel_to_vw.keys()]
    )
    chan_to_combos = {
        chan: list(it.combinations(group, 2)) for chan, group in chan_to_group.items()
    }
    to_compare = list(ub.flatten(chan_to_combos.values()))

    # ub.Timerit()
    for item1, item2 in ub.ProgIter(to_compare, desc='comparse_sensors', verbose=3):
        sensor1, channel1 = item1
        sensor2, channel2 = item2
        assert channel1 == channel2
        channel = channel1

        row = {
            'sensor1': sensor1,
            'sensor2': sensor2,
            'channel': channel,
        }

        u_values, u_weights = sensor_channel_to_vw[(sensor1, channel1)]
        v_values, v_weights = sensor_channel_to_vw[(sensor2, channel1)]

        # https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.wasserstein_distance.html#scipy.stats.wasserstein_distance
        # https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.energy_distance.html#scipy.stats.energy_distance
        dist_inputs = dict(
            u_values=u_values, v_values=v_values, u_weights=u_weights,
            v_weights=v_weights)

        if 1:
            row['emd'] = scipy.stats.wasserstein_distance(**dist_inputs)

        if 1:
            row['energy_dist'] = scipy.stats.energy_distance(**dist_inputs)

        u_mean = np.average(u_values.values, weights=u_weights.values)
        v_mean = np.average(v_values.values, weights=v_weights.values)
        row['mean_diff'] = abs(u_mean - v_mean)

        u_variance = np.average((u_values.values - u_mean) ** 2, weights=u_weights.values)
        u_variance = u_variance * sum(u_weights.values) / (sum(u_weights.values) - 1)
        u_std = np.sqrt(u_variance)

        v_variance = np.average((v_values.values - v_mean) ** 2, weights=v_weights.values)
        v_variance = v_variance * sum(v_weights.values) / (sum(v_weights.values) - 1)
        v_std = np.sqrt(v_variance)

        row['std_diff'] = abs(u_std - v_std)

        # TODO: robust alignment of pdfs
        if 0:
            cuv_values = np.union1d(u_values, v_values)
            cu_indexes = np.where(kwarray.one_hot_embedding(u_values.values, cuv_values.max() + 1, dim=0).sum(axis=1) > 0)
            cv_indexes = np.where(kwarray.one_hot_embedding(u_values.values, cuv_values.max() + 1, dim=0).sum(axis=1) > 0)
            cu_weights = np.zeros(cuv_values.shape, dtype=np.float32)
            cu_weights[cu_indexes] = u_weights
            cv_weights = np.zeros(cuv_values.shape, dtype=np.float32)
            cv_weights[cv_indexes] = v_weights
            row['kld'] = scipy.stats.entropy(cu_weights, cv_weights)

        distance_metrics.append(row)
        print('row = {}'.format(ub.repr2(row, nl=1)))
    distance_metrics = pd.DataFrame(distance_metrics)
    print(distance_metrics.to_string())
    return distance_metrics


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


def ensure_intensity_stats(coco_img, recompute=False, include_channels=None, exclude_channels=None):
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

        channels = kwcoco.FusedChannelSpec.coerce(channels)
        declared_channel_list = channels.as_list()

        requested_channels = channels
        if include_channels:
            requested_channels = requested_channels & include_channels
        if exclude_channels:
            requested_channels = requested_channels - exclude_channels

        if len(requested_channels) > 0:
            stats_fpath = ensure_intensity_sidecar(fpath, recompute=recompute)
            try:
                with open(stats_fpath, 'rb') as file:
                    stat_info = pickle.load(file)
            except Exception as ex:
                print('ex = {!r}'.format(ex))
                print('error loading stats_fpath = {!r}'.format(stats_fpath))
                raise

            alwaysappend = requested_channels.numel() == channels.numel()

            for band_idx, band_stat in enumerate(stat_info['bands']):
                try:
                    band_name = declared_channel_list[band_idx]
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
                if alwaysappend or band_name in requested_channels:
                    band_stat['band_name'] = band_name
                    intensity_stats['bands'].append(band_stat)
    return intensity_stats


def plot_intensity_histograms(full_df, config):

    unique_channels = full_df['channel'].unique()
    unique_sensors = full_df['sensor'].unique()

    import kwplot
    sns = kwplot.autosns()

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
            # print(chan_df)
            values = chan_df.intensity_bin
            weights = chan_df.value

            # Note: the calculation of the variance depends on the type of
            # weighting we choose
            average = np.average(values, weights=weights)
            variance = np.average((values - average) ** 2, weights=weights)
            variance = variance * sum(weights) / (sum(weights) - 1)
            stddev = np.sqrt(variance)

            pytype = float if values.values.dtype.kind == 'f' else int

            info = {
                'min': pytype(values.min()),
                'max': pytype(values.max()),
                'mean': average,
                'std': stddev,
                'total_weight': chan_df.value.sum(),
                'channel': channel,
                'sensor': sensor_name,
            }
            assert info['max'] >= info['min']
            # print('info = {}'.format(ub.repr2(info, nl=1, sort=False)))
            info_rows.append(info)

        sensor_chan_stats = pd.DataFrame(info_rows)
        print(sensor_chan_stats)

        hist_data_kw_ = hist_data_kw.copy()
        if hist_data_kw_['bins'] == 'auto':
            xvar = hist_data_kw['x']
            weightvar = hist_data_kw['weights']
            hist_data_kw_['bins'] = _weighted_auto_bins(sensor_df, xvar, weightvar)

        ax = kwplot.figure(fnum=1, pnum=pnum_()).gca()
        # z = [tuple(a.values()) for a in sensor_df[['intensity_bin', 'channel', 'sensor']].to_dict('records')]
        # ub.find_duplicates(z)
        try:
            # https://github.com/mwaskom/seaborn/issues/2709
            sns.histplot(ax=ax, data=sensor_df.reset_index(), **hist_data_kw_, **hist_style_kw)
        except Exception:
            print('hist_data_kw_ = {}'.format(ub.repr2(hist_data_kw_, nl=1)))
            print('hist_style_kw = {}'.format(ub.repr2(hist_style_kw, nl=1)))
            print('ERROR')
            print(sensor_df)
            raise
            pass
        ax.set_title(sensor_name)
        # maxx = sensor_df.intensity_bin.max()
        # maxx = sensor_maxes[sensor_name]
        # ax.set_xlim(0, maxx)

    return fig


# def _weighted_quantile(weights, qs):
#     cumtotal = np.cumsum(weights)
#     quantiles = cumtotal / cumtotal[-1]


def _weighted_auto_bins(data, xvar, weightvar):
    """
    Generalized histogram bandwidth estimators for weighted univariate data

    References:
        https://github.com/mwaskom/seaborn/issues/2710

    Example:
        >>> import pandas as pd
        >>> import numpy as np
        >>> n = 100
        >>> to_stack = []
        >>> rng = np.random.RandomState(432)
        >>> for group_idx in range(3):
        >>>     part_data = pd.DataFrame({
        >>>         'x': np.arange(n),
        >>>         'weights': rng.randint(0, 100, size=n),
        >>>         'hue': [f'group_{group_idx}'] * n,
        >>>     })
        >>>     to_stack.append(part_data)
        >>> data = pd.concat(to_stack).reset_index()
        >>> xvar = 'x'
        >>> weightvar = 'weights'
        >>> n_equal_bins = _weighted_auto_bins(data, xvar, weightvar)
        >>> # xdoctest: +REQUIRES(--show)
        >>> import seaborn as sns
        >>> sns.histplot(data=data, bins=n_equal_bins, x='x', weights='weights', hue='hue')
    """
    sort_df = data.sort_values(xvar)
    values = sort_df[xvar]
    weights = sort_df[weightvar]
    minval = values.iloc[0]
    maxval = values.iloc[-1]

    total = weights.sum()
    ptp = maxval - minval

    # _hist_bin_sqrt = ptp / np.sqrt(total)
    _hist_bin_sturges = ptp / (np.log2(total) + 1.0)

    cumtotal = weights.cumsum().values
    quantiles = cumtotal / cumtotal[-1]
    idx2, idx1 = np.searchsorted(quantiles, [0.75, 0.25])
    # idx2, idx1 = _weighted_quantile(weights, [0.75, 0.25])
    iqr = values.iloc[idx2] - values.iloc[idx1]
    _hist_bin_fd = 2.0 * iqr * total ** (-1.0 / 3.0)

    fd_bw = _hist_bin_fd  # Freedman-Diaconis
    sturges_bw = _hist_bin_sturges

    if fd_bw:
        bw_est = min(fd_bw, sturges_bw)
    else:
        # limited variance, so we return a len dependent bw estimator
        bw_est = sturges_bw

    from numpy.lib.histograms import _get_outer_edges, _unsigned_subtract
    first_edge, last_edge = _get_outer_edges(values, None)
    if bw_est:
        n_equal_bins = int(np.ceil(_unsigned_subtract(last_edge, first_edge) / bw_est))
    else:
        # Width can be zero for some estimators, e.g. FD when
        # the IQR of the data is zero.
        n_equal_bins = 1

    # Take the minimum of this and the number of actual bins
    n_equal_bins = min(n_equal_bins, len(values))
    return n_equal_bins


_SubConfig = IntensityHistogramConfig

if __name__ == '__main__':
    main()
