import ubelt as ub
import matplotlib as mpl
import matplotlib.text  # NOQA


def phantom_legend(label_to_color, mode='line', ax=None, legend_id=None, loc=0):
    import kwplot
    import kwimage
    plt = kwplot.autoplt()

    if ax is None:
        ax = plt.gca()

    _phantom_legends = getattr(ax, '_phantom_legends', None)
    if _phantom_legends is None:
        _phantom_legends = ax._phantom_legends = ub.ddict(dict)

    phantom = _phantom_legends[legend_id]
    handles = phantom['handles'] = []
    handles.clear()

    alpha = 1.0
    for label, color in label_to_color.items():
        color = kwimage.Color(color).as01()
        if mode == 'line':
            phantom_actor = plt.Line2D(
                (0, 0), (1, 1), color=color, label=label, alpha=alpha)
        elif mode == 'circle':
            phantom_actor = plt.Circle(
                (0, 0), 1, fc=color, label=label, alpha=alpha)
        else:
            raise KeyError
        handles.append(phantom_actor)

    legend_artist = ax.legend(handles=handles, loc=loc)
    phantom['artist'] = legend_artist

    # Re-add other legends
    for _phantom in _phantom_legends.values():
        artist = _phantom['artist']
        if artist is not legend_artist:
            ax.add_artist(artist)


def cropwhite_ondisk(fpath):
    import kwimage
    from kwplot.mpl_make import crop_border_by_color
    imdata = kwimage.imread(fpath)
    imdata = crop_border_by_color(imdata)
    kwimage.imwrite(fpath, imdata)


def dataframe_table(table, fpath, title=None, fontsize=12,
                    table_conversion='auto', dpi=None, fnum=None, show=False):
    """
    Use dataframe_image (dfi) to render a pandas dataframe.

    Args:
        table (pandas.DataFrame | pandas.io.formats.style.Styler)
        fpath (str | PathLike): where to save the image
        table_conversion (str):
            can be auto, chrome, or matplotlib (auto tries to default to
            chrome)

    Example:
        >>> # xdoctest: +REQUIRES(module:dataframe_image)
        >>> from geowatch.utils.util_kwplot import *  # NOQA
        >>> import ubelt as ub
        >>> dpath = ub.Path.appdir('kwplot/tests/test_dfi').ensuredir()
        >>> import pandas as pd
        >>> table = pd.DataFrame({'foo': ['one', 'one', 'one', 'two', 'two', 'two'],
        ...                       'bar': ['A', 'B', 'C', 'A', 'B', 'C'],
        ...                       'baz': [1, 2, 3, 4, 5, 6],
        ...                       'zoo': ['x', 'y', 'z', 'q', 'w', 't']})
        >>> fpath = dpath / 'dfi.png'
        >>> dataframe_table(table, fpath, title='A caption / title')
    """
    import kwimage
    import kwplot
    import dataframe_image as dfi
    import pandas as pd
    # table_conversion = "chrome"  # matplotlib

    if table_conversion == 'chrome':
        if ub.find_exe('google-chrome'):
            table_conversion = 'chrome'
        else:
            table_conversion = 'matplotlib'

    if isinstance(table, pd.DataFrame):
        style = table.style
    else:
        style = table

    if title is not None:
        style = style.set_caption(title)

    dfi.export(
        style,
        str(fpath),
        table_conversion=table_conversion,
        fontsize=fontsize,
        max_rows=-1,
        dpi=dpi,
    )
    if show == 'imshow':
        imdata = kwimage.imread(fpath)
        kwplot.imshow(imdata, fnum=fnum)
    elif show == 'eog':
        import xdev
        xdev.startfile(fpath)
    elif show:
        raise KeyError(f'Show can be "imshow" or "eog", not {show!r}')


def humanize_dataframe(df, col_formats=None, human_labels=None, index_format=None,
                       title=None):
    import humanize
    df2 = df.copy()
    if col_formats is not None:
        for col, fmt in col_formats.items():
            if fmt == 'intcomma':
                df2[col] = df[col].apply(humanize.intcomma)
            if fmt == 'concice_si_display':
                from kwcoco.metrics.drawing import concice_si_display
                for row in df2.index:
                    val = df2.loc[row, col]
                    # if isinstance(val, str):
                    #     try:
                    #         val = float(val)
                    #     except Exception:
                    #         ...
                    # print(f'val: {type(val)}={val}')
                    if isinstance(val, float):
                        val = concice_si_display(val)
                        df2.loc[row, col] = val
                df2[col] = df[col].apply(humanize.intcomma)
            if callable(fmt):
                df2[col] = df[col].apply(fmt)
    if human_labels:
        df2 = df2.rename(human_labels, axis=1)

    indexes = [df2.index, df2.columns]
    if human_labels:

        for index in indexes:
            if index.name is not None:
                index.name = human_labels.get(index.name, index.name)
            if index.names:
                index.names = [human_labels.get(n, n) for n in index.names]

    if index_format == 'capcase':
        def capcase(x):
            if '_' in x or x.islower():
                return ' '.join([w.capitalize() for w in x.split('_')])
            return x
        df2.index.values[:] = [human_labels.get(x, x) for x in df2.index.values]
        df2.index.values[:] = list(map(capcase, df2.index.values))
        # human_df = human_df.applymap(lambda x: str(x) if isinstance(x, int) else '{:0.2f}'.format(x))
        pass

    df2_style = df2.style
    if title:
        df2_style = df2_style.set_caption(title)
    return df2_style


def scatterplot_highlight(data, x, y, highlight, size=10, color='orange',
                          marker='*', val_to_color=None, ax=None, linewidths=None):
    if ax is None:
        import kwplot
        plt = kwplot.autoplt()
        ax = plt.gca()
    _starkw = {
        's': size,
        # 'edgecolor': color,
        'facecolor': 'none',
    }
    flags = data[highlight].apply(bool)
    star_data = data[flags]
    star_x = star_data[x]
    star_y = star_data[y]

    if linewidths is not None:
        _starkw['linewidths'] = linewidths

    if color != 'group':
        _starkw['edgecolor'] = color
        ax.scatter(star_x, star_y, marker=marker, **_starkw)
    else:
        val_to_group = dict(list(star_data.groupby(highlight)))
        if val_to_color is None:
            import kwimage
            val_to_color = ub.dzip(val_to_group, kwimage.Color.distinct(len(val_to_group)))
        for val, group in val_to_group.items():
            star_x = group[x]
            star_y = group[y]
            edgecolor = val_to_color[val]
            ax.scatter(star_x, star_y, marker=marker, edgecolor=edgecolor,
                       **_starkw)


def humanize_labels():
    ...


def relabel_xticks(mapping, ax=None):
    """
    Change the tick labels on the x-axis.

    Args:
        mapping (dict):
        ax (Axes | None):
    """
    if ax is None:
        import kwplot
        ax = kwplot.autoplt().gca()
    relabeler = LabelModifier(mapping)
    new_xticklabels = [
        relabeler._modify_labels(label)
        for label in ax.get_xticklabels()
    ]
    ax.set_xticklabels(new_xticklabels)


class LabelModifier:
    """
    Registers multiple ways to relabel text on axes

    TODO:
        - [ ] Maybe rename to label manager?

    Example:
        >>> # xdoctest: +SKIP
        >>> import sys, ubelt
        >>> from geowatch.utils.util_kwplot import *  # NOQA
        >>> import pandas as pd
        >>> import kwarray
        >>> rng = kwarray.ensure_rng(0)
        >>> models = ['category1', 'category2', 'category3']
        >>> data = pd.DataFrame([
        >>>     {
        >>>         'node.metrics.tpr': rng.rand(),
        >>>         'node.metrics.fpr': rng.rand(),
        >>>         'node.metrics.f1': rng.rand(),
        >>>         'node.param.model': rng.choice(models),
        >>>     } for _ in range(100)])
        >>> # xdoctest: +REQUIRES(env:PLOTTING_DOCTESTS)
        >>> import kwplot
        >>> sns = kwplot.autosns()
        >>> fig = kwplot.figure(fnum=1, pnum=(1, 2, 1), doclf=1)
        >>> ax1 = sns.boxplot(data=data, x='node.param.model', y='node.metrics.f1')
        >>> ax1.set_title('My node.param.model boxplot')
        >>> kwplot.figure(fnum=1, pnum=(1, 2, 2))
        >>> ax2 = sns.scatterplot(data=data, x='node.metrics.tpr', y='node.metrics.f1', hue='node.param.model')
        >>> ax2.set_title('My node.param.model scatterplot')
        >>> ax = ax2
        >>> #
        >>> def mapping(text):
        >>>     text = text.replace('node.param.', '')
        >>>     text = text.replace('node.metrics.', '')
        >>>     return text
        >>> #
        >>> self = LabelModifier(mapping)
        >>> self.add_mapping({'category2': 'FOO', 'category3': 'BAR'})
        >>> #fig.canvas.draw()
        >>> #
        >>> self.relabel(ax=ax1)
        >>> self.relabel(ax=ax2)
        >>> fig.canvas.draw()
    """

    def __init__(self, mapping=None):
        self._dict_mapper = {}
        self._func_mappers = []
        self.add_mapping(mapping)

    def copy(self):
        new = self.__class__()
        new.add_mapping(self._dict_mappem.copy())
        for m in self._func_mappers:
            new.add_mapping(m)
        return new

    def add_mapping(self, mapping):
        if mapping is not None:
            if callable(mapping):
                self._func_mappers.append(mapping)
            elif hasattr(mapping, 'get'):
                self._dict_mapper.update(mapping)
                self._dict_mapper.update(ub.udict(mapping).map_keys(str))
        return self

    def update(self, dict_mapping):
        self._dict_mapper.update(dict_mapping)
        self._dict_mapper.update(ub.udict(dict_mapping).map_keys(str))
        return self

    def _modify_text(self, text: str):
        # Handles strings, which we call text by convention, but that is
        # confusing here.
        new_text = text
        mapper = self._dict_mapper
        new_text = mapper.get(str(new_text), new_text)
        new_text = mapper.get(new_text, new_text)
        for mapper in self._func_mappers:
            new_text = mapper(new_text)
        return new_text

    def _modify_labels(self, label: mpl.text.Text):
        # Handles labels, which are mpl Text objects
        text = label.get_text()
        new_text = self._modify_text(text)
        label.set_text(new_text)
        return label

    def _modify_legend(self, legend):
        leg_title = legend.get_title()
        if isinstance(leg_title, str):
            new_leg_title = self._modify_text(leg_title)
            legend.set_text(new_leg_title)
        else:
            self._modify_labels(leg_title)
        for label in legend.texts:
            self._modify_labels(label)

    def relabel_yticks(self, ax=None):
        old_ytick_labels = ax.get_yticklabels()
        new_yticklabels = [self._modify_labels(label) for label in old_ytick_labels]
        ax.set_yticks(ax.get_yticks())
        ax.set_yticklabels(new_yticklabels)

    def relabel_xticks(self, ax=None):
        # Set xticks and yticks first before setting tick labels
        # https://stackoverflow.com/questions/63723514/userwarning-fixedformatter-should-only-be-used-together-with-fixedlocator
        # print(f'new_xlabel={new_xlabel}')
        # print(f'new_ylabel={new_ylabel}')
        # print(f'old_xticks={old_xticks}')
        # print(f'old_yticks={old_yticks}')
        # print(f'old_xtick_labels={old_xtick_labels}')
        # print(f'old_ytick_labels={old_ytick_labels}')
        # print(f'new_xticklabels={new_xticklabels}')
        # print(f'new_yticklabels={new_yticklabels}')
        old_xtick_labels = ax.get_xticklabels()
        new_xticklabels = [self._modify_labels(label) for label in old_xtick_labels]

        ax.set_xticks(ax.get_xticks())
        ax.set_xticklabels(new_xticklabels)

    def relabel_axes_labels(self, ax=None):
        old_xlabel = ax.get_xlabel()
        old_ylabel = ax.get_ylabel()
        old_title = ax.get_title()

        new_xlabel = self._modify_text(old_xlabel)
        new_ylabel = self._modify_text(old_ylabel)
        new_title = self._modify_text(old_title)

        ax.set_xlabel(new_xlabel)
        ax.set_ylabel(new_ylabel)
        ax.set_title(new_title)

    def relabel_legend(self, ax=None):
        if ax.legend_ is not None:
            self._modify_legend(ax.legend_)

    def relabel(self, ax=None, ticks=True, axes_labels=True, legend=True):
        if axes_labels:
            self.relabel_axes_labels(ax)
        if ticks:
            self.relabel_xticks(ax)
            self.relabel_yticks(ax)
        if legend:
            self.relabel_legend(ax)

    def __call__(self, ax=None):
        self.relabel(ax)


class FigureFinalizer(ub.NiceRepr):
    """
    Helper for defining where and how figures will be saved on disk.

    Known Parameters:
        dpi : float
        format : str
        metadata : dict
        bbox_inches : str
        pad_inches : float
        facecolor : color
        edgecolor : color
        backend : str
        orientation :
        papertype :
        transparent :
        bbox_extra_artists :
        pil_kwargs :

    Example:
        from geowatch.utils.util_kwplot import *  # NOQA
        self = FigureFinalizer()
        print('self = {}'.format(ub.urepr(self, nl=1)))
        self.update(dpi=300)

    """

    def __init__(
        self,
        dpath='.',
        size_inches=None,
        cropwhite=True,
        tight_layout=True,
        **kwargs
    ):
        locals_ = ub.udict(locals())
        locals_ -= {'self', 'kwargs'}
        locals_.update(kwargs)
        self.update(locals_)

    def __nice__(self):
        return ub.urepr(self.__dict__)

    def copy(self):
        """
        Create a copy of this object.
        """
        new = self.__class__(**self.__dict__)
        return new

    def update(self, *args, **kwargs):
        """
        Modify this config
        """
        self.__dict__.update(*args, **kwargs)

    def finalize(self, fig, fpath, **kwargs):
        """
        Sets the figure properties, like size, tight layout, etc, writes to
        disk, and then crops the whitespace out.

        Args:
            fig (matplotlib.figure.Figure): figure to safe

            fpath (str | PathLike): where to save the figure image

            **kwargs: overrides this config for this finalize only
        """
        config = ub.udict(self.__dict__) | kwargs

        dpath = ub.Path(config['dpath']).ensuredir()
        final_fpath = dpath / fpath
        savekw = {}
        if config.get('dpi', None) is not None:
            savekw['dpi'] = config['dpi']
            # fig.set_dpi(savekw['dpi'])
        if config['size_inches'] is not None:
            fig.set_size_inches(config['size_inches'])
        if config['tight_layout'] is not None:
            fig.tight_layout()
        # TODO: could save to memory and then write as an image
        fig.savefig(final_fpath, **savekw)
        if self.cropwhite:
            cropwhite_ondisk(final_fpath)
        return final_fpath

    def __call__(self, fig, fpath, **kwargs):
        """
        Alias for finalize
        """
        return self.finalize(fig, fpath, **kwargs)


def fix_matplotlib_dates(dates, format='mdate'):
    """

    Args:
        dates (List[None | Coerceble[datetime]]):
            input dates to fixup

        format (str):
            can be mdate for direct matplotlib usage or datetime for seaborn usage.

    Note:
        seaborn seems to do just fine with timestamps...
        todo:
            add regular matplotlib test for a real demo of where this is useful

    Example:
        >>> from geowatch.utils.util_kwplot import *  # NOQA
        >>> from kwutil.util_time import coerce_datetime
        >>> from kwutil.util_time import coerce_timedelta
        >>> import pandas as pd
        >>> import numpy as np
        >>> delta = coerce_timedelta('1 day')
        >>> n = 100
        >>> min_date = coerce_datetime('2020-01-01').timestamp()
        >>> max_date = coerce_datetime('2021-01-01').timestamp()
        >>> from kwarray.distributions import Uniform
        >>> distri = Uniform(min_date, max_date)
        >>> timestamps = distri.sample(n)
        >>> timestamps[np.random.rand(n) > 0.5] = np.nan
        >>> dates = list(map(coerce_datetime, timestamps))
        >>> scores = np.random.rand(len(dates))
        >>> table = pd.DataFrame({
        >>>     'isodates': [None if d is None else d.isoformat() for d in dates],
        >>>     'dates': dates,
        >>>     'timestamps': timestamps,
        >>>     'scores': scores
        >>> })
        >>> table['fixed_dates'] = fix_matplotlib_dates(table.dates, format='datetime')
        >>> table['fixed_timestamps'] = fix_matplotlib_dates(table.timestamps, format='datetime')
        >>> table['fixed_isodates'] = fix_matplotlib_dates(table.isodates, format='datetime')
        >>> table['mdate_dates'] = fix_matplotlib_dates(table.dates, format='mdate')
        >>> table['mdate_timestamps'] = fix_matplotlib_dates(table.timestamps, format='mdate')
        >>> table['mdate_isodates'] = fix_matplotlib_dates(table.isodates, format='mdate')
        >>> # xdoctest: +REQUIRES(env:PLOTTING_DOCTESTS)
        >>> import kwplot
        >>> sns = kwplot.autosns()
        >>> pnum_ = kwplot.PlotNums(nSubplots=8)
        >>> ax = kwplot.figure(fnum=1, doclf=1)
        >>> for key in table.columns.difference({'scores'}):
        >>>     ax = kwplot.figure(fnum=1, doclf=0, pnum=pnum_()).gca()
        >>>     sns.scatterplot(data=table, x=key, y='scores', ax=ax)
        >>>     if key.startswith('mdate_'):
        >>>         # TODO: make this formatter fixup work better.
        >>>         import matplotlib.dates as mdates
        >>>         ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        >>>         ax.xaxis.set_major_locator(mdates.DayLocator(interval=90))
    """
    from kwutil import util_time
    import matplotlib.dates as mdates
    new = []
    for d in dates:
        n = util_time.coerce_datetime(d)
        if n is not None:
            if format == 'mdate':
                n = mdates.date2num(n)
            elif format == 'datetime':
                ...
            else:
                raise KeyError(format)
        new.append(n)
    return new


def fix_matplotlib_timedeltas(deltas):
    from kwutil import util_time
    # import matplotlib.dates as mdates
    new = []
    for d in deltas:
        if d is None:
            n = None
        else:
            try:
                n = util_time.coerce_timedelta(d)
            except util_time.TimeValueError:
                n = None
        # if n is not None:
        #     n = mdates.num2timedelta(n)
        new.append(n)
    return new


def extract_legend(ax):
    """
    Creates a new figure that contains the original legend.
    """
    # ax.get_legend().remove()
    orig_legend = ax.get_legend()
    if orig_legend is None:
        raise RuntimeError('no legend')
    orig_legend_title = orig_legend.get_title().get_text()
    legend_handles = ax.get_legend_handles_labels()

    # fnum = 321
    import kwplot
    fig_onlylegend = kwplot.figure(
        fnum=str(ax.figure.number) + '_onlylegend', doclf=1)
    new_ax = fig_onlylegend.gca()
    new_ax.axis('off')
    new_ax.legend(*legend_handles, title=orig_legend_title,
                            loc='lower center')
    return new_ax


class ArtistManager:
    """
    Accumulates artist collections (e.g. lines, patches, ellipses) the user is
    interested in drawing so we can draw them efficiently.

    References:
        https://matplotlib.org/stable/api/collections_api.html
        https://matplotlib.org/stable/gallery/shapes_and_collections/ellipse_collection.html
        https://stackoverflow.com/questions/32444037/how-can-i-plot-many-thousands-of-circles-quickly

    Example:
        >>> # xdoctest: +SKIP
        >>> from geowatch.utils.util_kwplot import *  # NOQA
        >>> # xdoctest: +REQUIRES(env:PLOTTING_DOCTESTS)
        >>> import kwplot
        >>> sns = kwplot.autosns()
        >>> fig = kwplot.figure(fnum=1)
        >>> self = ArtistManager()
        >>> import kwimage
        >>> points = kwimage.Polygon.star().data['exterior'].data
        >>> self.add_linestring(points)
        >>> ax = fig.gca()
        >>> self.add_to_axes(ax)
        >>> ax.relim()
        >>> ax.set_xlim(-1, 1)
        >>> ax.set_ylim(-1, 1)

    Example:
        >>> # xdoctest: +SKIP
        >>> from geowatch.utils.util_kwplot import *  # NOQA
        >>> # xdoctest: +REQUIRES(env:PLOTTING_DOCTESTS)
        >>> import kwplot
        >>> sns = kwplot.autosns()
        >>> fig = kwplot.figure(fnum=1)
        >>> self = ArtistManager()
        >>> import kwimage
        >>> points = kwimage.Polygon.star().data['exterior'].data
        >>> y = 1
        >>> self.add_linestring([(0, y), (1, y)], color='kitware_blue')
        >>> y = 2
        >>> self.add_linestring([(0, y), (1, y)], color='kitware_green')
        >>> y = 3
        >>> self.add_circle((0, y), r=.1, color='kitware_darkgreen')
        >>> self.add_circle((0.5, y), r=.1, color='kitware_darkblue')
        >>> self.add_circle((0.2, y), r=.1, color='kitware_darkblue')
        >>> self.add_circle((1.0, y), r=.1, color='kitware_darkblue')
        >>> self.add_ellipse((0.2, 1), .1, .2, angle=10, color='kitware_gray')
        >>> self.add_linestring([(0, y), (1, y)], color='kitware_blue')
        >>> y = 4
        >>> self.add_linestring([(0, y), (1, y)], color='kitware_blue')
        >>> self.add_circle_marker((0, y), r=10, color='kitware_darkgreen')
        >>> self.add_circle_marker((0.5, y), r=10, color='kitware_darkblue')
        >>> self.add_circle_marker((0.2, y), r=10, color='kitware_darkblue')
        >>> self.add_circle_marker((1.0, y), r=10, color='kitware_darkblue')
        >>> self.add_ellipse_marker((0.2, 2), 10, 20, angle=10, color='kitware_gray')
        >>> self.add_linestring(np.array([
        ...     (0.2, 0.5),
        ...     (0.45, 1.6),
        ...     (0.62, 2.3),
        ...     (0.82, 4.9),
        >>> ]), color='kitware_yellow')
        >>> self.add_to_axes()
        >>> ax = fig.gca()
        >>> ax.set_xlim(0, 1)
        >>> ax.set_ylim(0, 5)
        >>> ax.autoscale_view()
    """

    def __init__(self):
        self.group_to_line_segments = ub.ddict(list)
        self.group_to_patches = ub.ddict(lambda : ub.ddict(list))
        self.group_to_ellipse_markers = ub.ddict(lambda: {
            'xy': [],
            'rx': [],
            'ry': [],
            'angle': [],
        })
        self.group_to_attrs = {}

    def _normalize_attrs(self, attrs):
        import kwimage
        attrs = ub.udict(attrs)
        if 'color' in attrs:
            attrs['color'] = kwimage.Color.coerce(attrs['color']).as01()
        if 'hashid' in attrs:
            attrs = attrs - {'hashid'}
        hashid = ub.hash_data(sorted(attrs.items()))[0:8]
        return hashid, attrs

    def plot(self, xs, ys, **attrs):
        """
        Alternative way to add lines
        """
        import numpy as np
        ys = [ys] if not ub.iterable(ys) else ys
        xs = [xs] if not ub.iterable(xs) else xs
        if len(ys) == 1 and len(xs) > 1:
            ys = ys * len(xs)
        if len(xs) == 1 and len(ys) > 1:
            xs = xs * len(ys)
        points = np.array(list(zip(xs, ys)))
        self.add_linestring(points, **attrs)

    def add_linestring(self, points, **attrs):
        """
        Args:
            points (List[Tuple[float, float]] | ndarray):
                an Nx2 set of ordered points

        NOTE:
            perhaps allow adding markers based on ax.scatter?
        """
        hashid, attrs = self._normalize_attrs(attrs)
        self.group_to_line_segments[hashid].append(points)
        self.group_to_attrs[hashid] = attrs

    def add_ellipse(self, xy, rx, ry, angle=0, **attrs):
        """
        Real ellipses in dataspace
        """
        hashid, attrs = self._normalize_attrs(attrs)
        ell = mpl.patches.Ellipse(xy, rx, ry, angle=angle, **attrs)
        self.group_to_patches[hashid]['ellipse'].append(ell)
        self.group_to_attrs[hashid] = attrs

    def add_circle(self, xy, r, **attrs):
        """
        Real ellipses in dataspace
        """
        hashid, attrs = self._normalize_attrs(attrs)
        ell = mpl.patches.Circle(xy, r, **attrs)
        self.group_to_patches[hashid]['circle'].append(ell)
        self.group_to_attrs[hashid] = attrs

    def add_ellipse_marker(self, xy, rx, ry, angle=0, color=None, **attrs):
        """
        Args:
            xy : center
            rx : radius in the first axis (size is in points, i.e. same way plot markers are sized)
            ry : radius in the second axis
            angle (float): The angles of the first axes, degrees CCW from the x-axis.

        """
        import numpy as np
        import kwimage
        if color is not None:
            if 'edgecolors' not in attrs:
                attrs['edgecolors'] = kwimage.Color.coerce(color).as01()
            if 'facecolors' not in attrs:
                attrs['facecolors'] = kwimage.Color.coerce(color).as01()

        hashid, attrs = self._normalize_attrs(attrs)
        cols = self.group_to_ellipse_markers[hashid]

        xy = np.array(xy)
        if len(xy.shape) == 1:
            assert xy.shape[0] == 2
            xy = xy[None, :]
        elif len(xy.shape) == 2:
            assert xy.shape[1] == 2
        else:
            raise ValueError

        # Broadcast shapes
        rx = [rx] if not ub.iterable(rx) else rx
        ry = [ry] if not ub.iterable(ry) else ry
        angle = [angle] if not ub.iterable(angle) else angle
        nums = list(map(len, (xy, rx, ry, angle)))
        if not ub.allsame(nums):
            new_n = max(nums)
            for n in nums:
                assert n == 1 or n == new_n
            if len(xy) == 1:
                xy = np.repeat(xy, new_n, axis=0)
            if len(rx) == 1:
                rx = np.repeat(rx, new_n, axis=0)
            if len(ry) == 1:
                ry = np.repeat(ry, new_n, axis=0)
            if len(angle) == 1:
                ry = np.repeat(ry, new_n, axis=0)

        cols['xy'].append(xy)
        cols['rx'].append(rx)
        cols['ry'].append(ry)
        cols['angle'].append(angle)
        self.group_to_attrs[hashid] = attrs

    def add_circle_marker(self, xy, r, **attrs):
        """
        Args:
            xy (List[Tuple[float, float]] | ndarray):
                an Nx2 set of circle centers
            r (List[float] | ndarray):
                an Nx1 set of circle radii
        """
        self.add_ellipse_marker(xy, rx=r, ry=r, angle=0, **attrs)

    def build_collections(self, ax=None):
        import numpy as np
        collections = []
        for hashid, segments in self.group_to_line_segments.items():
            attrs = self.group_to_attrs[hashid]
            collection = mpl.collections.LineCollection(segments, **attrs)
            collections.append(collection)

        for hashid, type_to_patches in self.group_to_patches.items():
            attrs = self.group_to_attrs[hashid]
            for ptype, patches in type_to_patches.items():
                collection = mpl.collections.PatchCollection(patches, **attrs)
                collections.append(collection)

        for hashid, cols in self.group_to_ellipse_markers.items():
            attrs = self.group_to_attrs[hashid] - {'hashid'}
            xy = np.concatenate(cols['xy'], axis=0)
            rx = np.concatenate(cols['rx'], axis=0)
            ry = np.concatenate(cols['ry'], axis=0)
            angles = np.concatenate(cols['angle'], axis=0)
            collection = mpl.collections.EllipseCollection(
                widths=rx, heights=ry, offsets=xy, angles=angles,
                units='points',
                # units='x',
                # units='xy',
                transOffset=ax.transData,
                **attrs
            )
            # collection.set_transOffset(ax.transData)
            collections.append(collection)

        return collections

    def add_to_axes(self, ax=None):
        import kwplot
        if ax is None:
            plt = kwplot.autoplt()
            ax = plt.gca()

        collections = self.build_collections(ax=ax)
        for collection in collections:
            ax.add_collection(collection)

    def bounds(self):
        import numpy as np
        all_lines = []
        for segments in self.group_to_line_segments.values():
            for lines in segments:
                lines = np.array(lines)
                all_lines.append(lines)

        all_coords = np.concatenate(all_lines, axis=0)
        import pandas as pd
        flags = pd.isnull(all_coords)
        all_coords[flags] = np.nan
        all_coords = all_coords.astype(float)

        minx, miny = np.nanmin(all_coords, axis=0) if len(all_coords) else 0
        maxx, maxy = np.nanmax(all_coords, axis=0) if len(all_coords) else 1
        ltrb = minx, miny, maxx, maxy
        return ltrb

    def setlims(self, ax=None):
        import kwplot
        if ax is None:
            plt = kwplot.autoplt()
            ax = plt.gca()

        from kwimage.structs import _generic
        minx, miny, maxx, maxy = self.bounds()
        _generic._setlim(minx, miny, maxx, maxy, 1.1, ax=ax)
        # ax.set_xlim(minx, maxx)
        # ax.set_ylim(miny, maxy)


def time_sample_arcplot(time_samples, yloc=1, ax=None):
    """
    Example:
        >>> import sys, ubelt
        >>> sys.path.append(ubelt.expandpath('~/code/watch'))
        >>> from geowatch.utils.util_kwplot import *  # NOQA
        >>> time_samples = [
        >>>     [1, 3, 5, 7, 9],
        >>>     [2, 3, 4, 6, 8],
        >>>     [1, 5, 6, 7, 9],
        >>> ]
        >>> import kwplot
        >>> kwplot.autompl()
        >>> time_sample_arcplot(time_samples)

    References:
        https://stackoverflow.com/questions/42162787/arc-between-points-in-circle
    """

    import kwplot
    import numpy as np
    # import matplotlib.patches as patches

    USE_BEZIER_PACKAGE = 0
    if USE_BEZIER_PACKAGE:
        import bezier
    else:
        # bezier_path = np.arange(0, 1.01, 0.01)
        num_path_points = 20
        s_vals = np.linspace(0, 1, num_path_points)

    if ax is None:
        ax = kwplot.plt.gca()
        # ax.cla()
        # maxx = 0

    num_samples = len(time_samples)
    for idx, xlocs in enumerate(time_samples):
        assert sorted(xlocs) == xlocs
        ylocs = [yloc] * len(xlocs)
        # ax.plot(xlocs, ylocs, 'o-')
        # maxx = max(max(xlocs), maxx)

        dist = ((idx + 1) / num_samples)
        dist = (dist * 0.5) + 0.5
        # print(f'dist={dist}')

        xy_sequence = list(zip(xlocs, ylocs))
        curve_path = []

        for xy1, xy2 in ub.iter_window(xy_sequence, 2):

            x1, y1 = xy1
            x2, y2 = xy2

            dx = x2 - x1
            dy = y2 - y1
            # normals are
            raw_normal1 = (-dy, dx)
            # normal2 = (dy, -dx)
            unit_normal1 = np.array(raw_normal1) / np.linalg.norm(raw_normal1)

            nx, ny = unit_normal1 * dist

            xm, ym = [(x1 + x2) / 2, (y1 + y2) / 2]
            xb, yb = [xm + nx, ym + ny]

            if USE_BEZIER_PACKAGE:
                # Create random bezier control points
                nodes_f = np.array([
                    [x1, y1],
                    [xb, yb],
                    [x2, y2],
                ]).T
                curve = bezier.Curve(nodes_f, degree=2)
                num = 10
                s_vals = np.linspace(0, 1, num)
                # s_vals = np.linspace(*sorted(rng.rand(2)), num)
                path_f = curve.evaluate_multi(s_vals)
                curve_path += list(path_f)
            else:
                # Compute and store the Bezier curve points
                # pure numpy version
                curve_x = (1 - s_vals) ** 2 * x1 + 2 * (1 - s_vals) * s_vals * xb + s_vals ** 2 * x2
                curve_y = (1 - s_vals) ** 2 * y1 + 2 * (1 - s_vals) * s_vals * yb + s_vals ** 2 * y2
                curve_path += list(zip(curve_x, curve_y))

        curve_path = np.array(curve_path)
        ax.plot(curve_path.T[0], curve_path.T[1], '-', alpha=0.5)

    # ax.set_xlim(0, maxx)
    # ax.set_ylim(0, 3)


class Palette(ub.udict):
    """
    Dictionary subclass that maps a label to a particular color.

    Explicit colors per label can be given, but for other unspecified labels we
    attempt to generate a distinct color.

    Example:
        >>> from geowatch.utils.util_kwplot import *  # NOQA
        >>> self1 = Palette()
        >>> self1.add_labels(labels=['a', 'b'])
        >>> self1.update({'foo': 'blue'})
        >>> self1.update(['bar', 'baz'])
        >>> self2 = Palette.coerce({'foo': 'blue'})
        >>> self2.update(['a', 'b', 'bar', 'baz'])
        >>> self1 = self1.sorted_keys()
        >>> self2 = self2.sorted_keys()
        >>> # xdoctest: +REQUIRES(env:PLOTTING_DOCTESTS)
        >>> import kwplot
        >>> kwplot.autoplt()
        >>> canvas1 = self1.make_legend_img()
        >>> canvas2 = self2.make_legend_img()
        >>> canvas = kwimage.stack_images([canvas1, canvas2])
        >>> kwplot.imshow(canvas)
    """

    @classmethod
    def coerce(cls, data):
        self = cls()
        self.update(data)
        return self

    def update(self, other):
        if isinstance(other, dict):
            self.add_labels(label_to_color=other)
        else:
            self.add_labels(labels=other)

    def add_labels(self, label_to_color=None, labels=None):
        """
        Forces particular labels to take a specific color and then chooses
        colors for any other unspecified label.

        Args:
            label_to_color (Dict[str, Any] | None): mapping to colors that are forced
            labels (List[str] | None): new labels that should take distinct colors
        """
        import kwimage
        # Given an existing set of colors, add colors to things without it.
        if label_to_color is None:
            label_to_color = {}
        if labels is None:
            labels = []

        # Determine which labels in the input mapping are not explicitly given
        specified = {k: kwimage.Color.coerce(v).as01()
                     for k, v in label_to_color.items() if v is not None}
        unspecified = ub.oset(label_to_color.keys()) - specified

        # Merge specified colors into this pallet
        super().update(specified)

        # Determine which labels need a color.
        new_labels = (unspecified | ub.oset(labels)) - set(self.keys())
        num_new = len(new_labels)
        if num_new:
            existing_colors = list(self.values())
            new_colors = kwimage.Color.distinct(num_new,
                                                existing=existing_colors,
                                                legacy=False)
            new_label_to_color = dict(zip(new_labels, new_colors))
            super().update(new_label_to_color)

    def make_legend_img(self, dpi=300, **kwargs):
        import kwplot
        legend = kwplot.make_legend_img(self, dpi=dpi, **kwargs)
        return legend

    def sorted_keys(self):
        return self.__class__(super().sorted_keys())

    def reorder(self, head=None, tail=None):
        if head is None:
            head = []
        if tail is None:
            tail = []
        head_part = self.subdict(head)
        tail_part = self.subdict(tail)
        end_keys = (head_part.keys() | tail_part.keys())
        mid_part = self - end_keys
        new = self.__class__(head_part | mid_part | tail_part)
        return new

    """
    # Do we want to offer standard pallets for small datas

    # if num_regions < 10:
    #     colors = sns.color_palette(n_colors=num_regions)
    # else:
    #     colors = kwimage.Color.distinct(num_regions, legacy=False)
    #     colors = [kwimage.Color.coerce(c).adjust(saturate=-0.3, lighten=-0.1).as01()
    #               for c in kwimage.Color.distinct(num_regions, legacy=False)]
    """


class PaletteManager:
    """
    Manages colors that should be kept constant across different labels for
    multiple parameters.

    self = PaletteManager()
    self.update_params('region_id', {'region1': 'red'})
    """
    def __init__(self):
        self.param_to_palette = {}


def color_new_labels(label_to_color, labels):
    import kwimage
    # Given an existing set of colors, add colors to things without it.
    missing = {k for k, v in label_to_color.items() if v is None}
    has_color = set(label_to_color) - missing
    missing |= set(labels) - has_color
    existing_colors = list((ub.udict(label_to_color) & has_color).values())
    new_colors = kwimage.Color.distinct(len(missing), existing=existing_colors, legacy=False)
    new_label_to_color = label_to_color.copy()
    new_label_to_color.update(dict(zip(missing, new_colors)))
    return new_label_to_color


def autompl2():
    """
    New autompl with inline logic for notebooks
    """
    import kwplot
    try:
        import IPython
        ipy = IPython.get_ipython()
        ipy.config
        # TODO: general test to see if we are in a notebook where
        # we want to inline things.
        if 'colab' in str(ipy.config['IPKernelApp']['kernel_class']):
            ipy.run_line_magic('matplotlib', 'inline')
    except NameError:
        ...
    kwplot.autompl()
