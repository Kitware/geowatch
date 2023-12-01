import matplotlib
# import matplotlib as mpl
import numpy as np

distinct_colors = ["#FFFF00", "#FF4A46", "#FF34FF", "#1CE6FF", "#008941", "#006FA6", "#A30059",
                   "#00FF23", "#7A4900", "#0000A6", "#63FFAC", "#B79762", "#004D43", "#8FB0FF", "#997D87",
                   "#5A0007", "#809693",  # "#FEFFE6",
                   "#1B4400", "#4FC601", "#3B5DFF", "#4A3B53", "#FF2F80",
                   "#61615A", "#BA0900", "#6B7900", "#00C2A0", "#FFAA92", "#FF90C9", "#B903AA", "#D16100",
                   "#DDEFFF", "#000035", "#7B4F4B", "#A1C299", "#300018", "#0AA6D8", "#013349", "#00846F",
                   "#372101", "#FFB500", "#C2FFED", "#A079BF", "#CC0744", "#C0B9B2", "#C2FF99", "#001E09",
                   "#00489C", "#6F0062", "#0CBD66", "#EEC3FF", "#456D75", "#B77B68", "#7A87A1", "#788D66",
                   "#885578", "#FAD09F", "#FF8A9A", "#D157A0", "#BEC459", "#456648", "#0086ED", "#886F4C",
                   "#34362D", "#B4A8BD", "#00A6AA", "#452C2C", "#636375", "#A3C8C9", "#FF913F", "#938A81",
                   "#575329", "#00FECF", "#B05B6F", "#8CD0FF", "#3B9700", "#04F757", "#C8A1A1", "#1E6E00",
                   "#7900D7", "#A77500", "#6367A9", "#A05837", "#6B002C", "#772600", "#D790FF", "#9B9700",
                   "#549E79", "#FFF69F", "#201625", "#72418F", "#BC23FF", "#99ADC0", "#3A2465", "#922329",
                   "#5B4534", "#FDE8DC", "#404E55", "#0089A3", "#CB7E98", "#A4E804", "#324E72", "#6A3A4C",
                   "#83AB58", "#001C1E", "#D1F7CE", "#004B28", "#C8D0F6", "#A3A489", "#806C66", "#222800",
                   "#BF5650", "#E83000", "#66796D", "#DA007C", "#FF1A59", "#8ADBB4", "#1E0200", "#5B4E51",
                   "#C895C5", "#320033", "#FF6832", "#66E1D3", "#CFCDAC", "#D0AC94", "#7ED379", "#012C58",
                   "#000000", "#FFFF00", "#1CE6FF", "#FF34FF", "#FF4A46", "#008941", "#006FA6", "#A30059",
                   "#FFDBE5", "#7A4900", "#0000A6", "#63FFAC", "#B79762", "#004D43", "#8FB0FF", "#997D87",
                   "#5A0007", "#809693", "#FEFFE6", "#1B4400", "#4FC601", "#3B5DFF", "#4A3B53", "#FF2F80",
                   "#61615A", "#BA0900", "#6B7900", "#00C2A0", "#FFAA92", "#FF90C9", "#B903AA", "#D16100",
                   "#DDEFFF", "#000035", "#7B4F4B", "#A1C299", "#300018", "#0AA6D8", "#013349", "#00846F",
                   "#372101", "#FFB500", "#C2FFED", "#A079BF", "#CC0744", "#C0B9B2", "#C2FF99", "#001E09",
                   "#00489C", "#6F0062", "#0CBD66", "#EEC3FF", "#456D75", "#B77B68", "#7A87A1", "#788D66",
                   "#885578", "#FAD09F", "#FF8A9A", "#D157A0", "#BEC459", "#456648", "#0086ED", "#886F4C",
                   "#34362D", "#B4A8BD", "#00A6AA", "#452C2C", "#636375", "#A3C8C9", "#FF913F", "#938A81",
                   "#575329", "#00FECF", "#B05B6F", "#8CD0FF", "#3B9700", "#04F757", "#C8A1A1", "#1E6E00",
                   "#7900D7", "#A77500", "#6367A9", "#A05837", "#6B002C", "#772600", "#D790FF", "#9B9700",
                   "#549E79", "#FFF69F", "#201625", "#72418F", "#BC23FF", "#99ADC0", "#3A2465", "#922329",
                   "#5B4534", "#FDE8DC", "#404E55", "#0089A3", "#CB7E98", "#A4E804", "#324E72", "#6A3A4C",
                   "#83AB58", "#001C1E", "#D1F7CE", "#004B28", "#C8D0F6", "#A3A489", "#806C66", "#222800",
                   "#BF5650", "#E83000", "#66796D", "#DA007C", "#FF1A59", "#8ADBB4", "#1E0200", "#5B4E51",
                   "#C895C5", "#320033", "#FF6832", "#66E1D3", "#CFCDAC", "#D0AC94", "#7ED379", "#012C58"]


def hex_to_rgb(hex):
    hex = hex.lstrip('#')
    hlen = len(hex)
    rgb = tuple(int(hex[i:i + hlen // 3], 16)
                for i in range(0, hlen, hlen // 3))
    return rgb


def n_distinguishable_colors(nlabels: int = 10, first_color_black=True, last_color_black=True,
                             bg_alpha=0.75, fg_alpha=0.9):

    hex_list = distinct_colors[:nlabels]
    rgb_list = []
    for hex_item in hex_list:
        rgb = hex_to_rgb(hex_item)
        rgb_list.append((rgb[0], rgb[1], rgb[2], fg_alpha))

    if first_color_black:
        hex_list[0] = "#000000"

    # if last_color_black:
    #     rgb_list[-1] = (0, 0, 0, bg_alpha)

    cmap = matplotlib.colors.ListedColormap(hex_list, name='cmap')
    # cmap = mpl.colors.ListedColormap(rgb_list, N=nlabels)
    # cmap = cmap.from_list('Custom cmap', cmaplist[0:], cmap.N)
    return cmap


def rand_cmap(nlabels, type='bright', first_color_black=True, last_color_black=False,
              verbose=False, bg_alpha=0.75, fg_alpha=0.9, random_colors=False):
    """
    Creates a random colormap to be used together with matplotlib. Useful for segmentation tasks
    :param nlabels: Number of labels (size of colormap)
    :param type: 'bright' for strong colors, 'soft' for pastel colors
    :param first_color_black: Option to use first color as black, True or False
    :param last_color_black: Option to use last color as black, True or False
    :param verbose: Prints the number of labels and shows the colormap. True or False
    :return: colormap for matplotlib
    """
    from matplotlib.colors import LinearSegmentedColormap
    import colorsys
    import numpy as np

    if type not in ('bright', 'soft'):
        print('Please choose "bright" or "soft" for type')
        return

    if verbose:
        print('Number of labels: ' + str(nlabels))

    # Generate color map for bright colors, based on hsv
    if type == 'bright':
        if random_colors:
            randHSVcolors = [(np.random.uniform(low=0.0, high=1),
                              np.random.uniform(low=0.4, high=1),
                              np.random.uniform(low=0.9, high=1)) for i in range(nlabels)]
        else:
            randHSVcolors = [(np.random.uniform(low=0.0, high=1),
                              np.random.uniform(low=0.4, high=1),
                              np.random.uniform(low=0.9, high=1)) for i in range(nlabels)]

        # Convert HSV list to RGB
        randRGBcolors = []
        for HSVcolor in randHSVcolors:
            randRGBcolors.append((*colorsys.hsv_to_rgb(HSVcolor[0], HSVcolor[1], HSVcolor[2]), fg_alpha))
        # randRGBcolors[17] = (*matplotlib.colors.to_rgb('magenta'),fg_alpha)
        if first_color_black:
            randRGBcolors[0] = (0, 0, 0, bg_alpha)

        if last_color_black:
            randRGBcolors[-1] = (0, 0, 0, bg_alpha)

        random_colormap = LinearSegmentedColormap.from_list('new_map', randRGBcolors, N=nlabels)

    # Generate soft pastel colors, by limiting the RGB spectrum
    if type == 'soft':
        low = 0.6
        high = 0.95
        randRGBcolors = [(np.random.uniform(low=low, high=high),
                          np.random.uniform(low=low, high=high),
                          np.random.uniform(low=low, high=high)) for i in range(nlabels)]

        if first_color_black:
            randRGBcolors[0] = [0, 0, 0]

        if last_color_black:
            randRGBcolors[-1] = [0, 0, 0]
        random_colormap = LinearSegmentedColormap.from_list('new_map', randRGBcolors, N=nlabels)

    # Display colorbar
    if verbose:
        from matplotlib import colors, colorbar
        from matplotlib import pyplot as plt
        fig, ax = plt.subplots(1, 1, figsize=(15, 0.5))

        bounds = np.linspace(0, nlabels, nlabels + 1)
        norm = colors.BoundaryNorm(bounds, nlabels)

        cb = colorbar.ColorbarBase(ax, cmap=random_colormap, norm=norm,  # NOQA
                                   spacing='proportional', ticks=None,
                                   boundaries=bounds, format='%1i',
                                   orientation=u'horizontal')

    return random_colormap


def plot_grad_flow(named_parameters):
    import matplotlib.pyplot as plt
    ave_grads = []
    layers = []
    for n, p in named_parameters:
        if (p.requires_grad) and ("bias" not in n):
            layers.append(n)
            ave_grads.append(p.grad.abs().mean().cpu().detach().numpy())
    fig = plt.figure()
    ax1 = fig.add_subplot(1, 1, 1)
    ax1.plot(ave_grads, alpha=0.3, color="b")
    ax1.set_hlines(0, 0, len(ave_grads) + 1, linewidth=1, color="k" )
    ax1.set_xticks(range(0, len(ave_grads), 1), layers, rotation="vertical")
    ax1.set_xlim(xmin=0, xmax=len(ave_grads))
    ax1.set_xlabel("Layers")
    ax1.set_ylabel("average gradient")
    ax1.set_title("Gradient flow")
    ax1.grid(True)
    return fig


def plot_grad_flow_v2(named_parameters):
    '''Plots the gradients flowing through different layers in the net during training.
    Can be used for checking for possible gradient vanishing / exploding problems.

    Usage: Plug this function in Trainer class after loss.backwards() as
    "plot_grad_flow(self.model.named_parameters())" to visualize the gradient flow'''
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    ave_grads = []
    max_grads = []
    layers = []
    for n, p in named_parameters:
        if (p.requires_grad) and ("bias" not in n):
            layers.append(n)
            ave_grads.append(p.grad.abs().mean())
            max_grads.append(p.grad.abs().max())
    fig = plt.figure()
    ax1 = fig.add_subplot(1, 1, 1)
    ax1.bar(np.arange(len(max_grads)), max_grads, alpha=0.1, lw=1, color="c")
    ax1.bar(np.arange(len(max_grads)), ave_grads, alpha=0.1, lw=1, color="b")
    ax1.set_hlines(0, 0, len(ave_grads) + 1, lw=2, color="k" )
    ax1.set_xticks(range(0, len(ave_grads), 1), layers, rotation="vertical")
    ax1.set_xlim(left=0, right=len(ave_grads))
    ax1.set_ylim(bottom=-0.001, top=0.02)  # zoom in on the lower gradient regions
    ax1.set_xlabel("Layers")
    ax1.set_ylabel("average gradient")
    ax1.set_title("Gradient flow")
    ax1.set_grid(True)
    ax1.legend([Line2D([0], [0], color="c", lw=4),
                Line2D([0], [0], color="b", lw=4),
                Line2D([0], [0], color="k", lw=4)], ['max-gradient', 'mean-gradient', 'zero-gradient'])
    return fig
