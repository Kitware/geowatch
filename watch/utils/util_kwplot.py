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
                    table_conversion='auto', fnum=None, show=False):
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
        >>> from watch.utils.util_kwplot import *  # NOQA
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
    )
    if show == 'imshow':
        imdata = kwimage.imread(fpath)
        kwplot.imshow(imdata, fnum=fnum)
    elif show == 'eog':
        import xdev
        xdev.startfile(fpath)


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


def scatterplot_highlight(data, x, y, highlight, size=10, color='orange', marker='*', ax=None):
    if ax is None:
        import kwplot
        plt = kwplot.autoplt()
        ax = plt.gca()
    _starkw = {
        's': size,
        'edgecolor': color,
        'facecolor': 'none',
    }
    flags = data[highlight].apply(bool)
    star_data = data[flags]
    star_x = star_data[x]
    star_y = star_data[y]
    ax.scatter(star_x, star_y, marker=marker, **_starkw)


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

    Example:
        >>> # xdoctest: +SKIP
        >>> import sys, ubelt
        >>> from watch.utils.util_kwplot import *  # NOQA
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
        ax.set_xticks(ax.get_xticks())
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
        ax.set_yticks(ax.get_yticks())
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


class FigureFinalizer:
    """
    Helper for defining where and how figures will be saved on disk.
    """

    def __init__(
        self,
        dpath='.',
        size_inches=None,
        cropwhite=True,
        tight_layout=True
    ):
        self.update(ub.udict(locals()) - {'self'})

    def update(self, *args, **kwargs):
        self.__dict__.update(*args, **kwargs)

    def finalize(self, fig, fpath, **kwargs):
        config = ub.udict(self.__dict__) | kwargs
        final_fpath = ub.Path(config['dpath']) / fpath
        if config['size_inches'] is not None:
            fig.set_size_inches(config['size_inches'])
        if config['tight_layout'] is not None:
            fig.tight_layout()
        fig.savefig(final_fpath)
        cropwhite_ondisk(final_fpath)

    def __call__(self, fig, fpath, **kwargs):
        return self.finalize(fig, fpath, **kwargs)


def fix_matplotlib_dates(dates):
    from watch.utils import util_time
    import matplotlib.dates as mdates
    new = []
    for d in dates:
        n = util_time.coerce_datetime(d)
        if n is not None:
            n = mdates.date2num(n)
        new.append(n)
    return new


def fix_matplotlib_timedeltas(deltas):
    from watch.utils import util_time
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
