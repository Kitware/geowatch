import sys
import os
current_path = os.getcwd().split("/")
if 'projects' in current_path:
    sys.path.append("/home/native/projects/cranberry_attention/")
else:
    sys.path.append("/data/cranberry_attention/")
import matplotlib; 
# matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import matplotlib as mpl
from scipy import ndimage
from skimage.segmentation import find_boundaries
from skimage import morphology
from matplotlib import colors
from random import shuffle
import random
np.random.seed(128)
random.seed(128)


def rand_cmap(nlabels, type='bright', first_color_black=True, last_color_black=False, verbose=True, bg_alpha=0.75, fg_alpha=0.9, random_colors=False):
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
        print ('Please choose "bright" or "soft" for type')
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
            randRGBcolors.append((*colorsys.hsv_to_rgb(HSVcolor[0], HSVcolor[1], HSVcolor[2]),fg_alpha))
        # randRGBcolors[17] = (*matplotlib.colors.to_rgb('magenta'),fg_alpha)
        if first_color_black:
            randRGBcolors[0] = (0,0,0, bg_alpha)

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

        cb = colorbar.ColorbarBase(ax, cmap=random_colormap, norm=norm, spacing='proportional', ticks=None,
                                   boundaries=bounds, format='%1i', orientation=u'horizontal')

    return random_colormap
    
