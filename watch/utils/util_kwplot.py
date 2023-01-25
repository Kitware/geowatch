import ubelt as ub


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


def relabel_xticks(mapping, ax=None):
    """
    Change the tick labels on the x-axis.

    Args:
        mapping (dict):
        ax (Axes | None):
    """
    if ax is None:
        import kwplot
        plt = kwplot.autoplt()
        ax = plt.gca()
    xtick_labels = list(ax.get_xticklabels())

    new_labels = []
    for label in xtick_labels:
        text = label.get_text()
        new_text = mapping.get(text, mapping.get(str(text), text))
        label.set_text(new_text)
        new_labels.append(label)

    ax.set_xticklabels(new_labels)


def dataframe_table(table_style, fpath, fontsize=12, fnum=None, show='eog'):
    import kwimage
    import kwplot
    import dataframe_image as dfi
    dfi_converter = "chrome"  # matplotlib
    dfi.export(
        table_style,
        str(fpath),
        table_conversion=dfi_converter,
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
