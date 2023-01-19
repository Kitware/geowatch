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
