"""
These functions might be added to kwimage
"""
from functools import lru_cache
import numpy as np
import cv2
import ubelt as ub


def draw_header_text(image, text, fit=False, color='red', halign='center',
                     stack='auto'):
    """
    Places a black bar on top of an image and writes text in it

    TODO:
        This can likely be derecated in favor of the kwimage version

    Args:

        image (ndarray | dict | None):
            numpy image or dictionary containing a key width

        text (str) :
            text to draw

        fit (bool | str):
            If False, will draw as much text within the given width as possible.
            If True, will draw all text and then resize to fit in the given width
            If "shrink", will only resize the text if it is too big to fit, in
            other words this is like fit=True, but it wont enlarge the text.

        color (str | Tuple) :
            a color coercable to :class:`kwimage.Color`.

        halign (str) :
            Horizontal alignment. Can be left, center, or right.

        stack (bool | str):
            if True returns the stacked image, otherwise just returns the
            header. If 'auto', will only stack if an image is given as an
            ndarray.

    Returns:
        ndarray

    Example:
        >>> from watch.utils.util_kwimage import *  # NOQA
        >>> import kwimage
        >>> image = kwimage.grab_test_image()
        >>> tiny_image = kwimage.imresize(image, dsize=(64, 64))
        >>> canvases = []
        >>> canvases += [draw_header_text(image=image, text='unfit long header ' * 5, fit=False)]
        >>> canvases += [draw_header_text(image=image, text='shrunk long header ' * 5, fit='shrink')]
        >>> canvases += [draw_header_text(image=image, text='left header', fit=False, halign='left')]
        >>> canvases += [draw_header_text(image=image, text='center header', fit=False, halign='center')]
        >>> canvases += [draw_header_text(image=image, text='right header', fit=False, halign='right')]
        >>> canvases += [draw_header_text(image=image, text='shrunk header', fit='shrink', halign='left')]
        >>> canvases += [draw_header_text(image=tiny_image, text='shrunk header-center', fit='shrink', halign='center')]
        >>> canvases += [draw_header_text(image=image, text='fit header', fit=True, halign='left')]
        >>> canvases += [draw_header_text(image={'width': 200}, text='header only', fit=True, halign='left')]
        >>> # xdoctest: +REQUIRES(--show)
        >>> import kwplot
        >>> kwplot.autompl()
        >>> pnum_ = kwplot.PlotNums(nSubplots=len(canvases))
        >>> for c in canvases:
        >>>     kwplot.imshow(c, pnum=pnum_())
        >>> kwplot.show_if_requested()
    """
    # import cv2
    import kwimage

    if stack == 'auto':
        stack = isinstance(image, np.ndarray)

    if isinstance(image, dict):
        width = image['width']
        if stack:
            raise ValueError('Must pass in the actual image if stack is True')
    else:
        width = image.shape[1]

    if stack:
        h, w = image.shape[0:2]
        min_pixels = 32
        if w < min_pixels or h < min_pixels:
            image = kwimage.imresize(image, min_dim=min_pixels)
        width = image.shape[1]

    if fit:
        # TODO: allow a shrink-to-fit only option
        try:
            # needs new kwimage to work
            header = kwimage.draw_text_on_image(
                None, text, org=None,
                valign='top', halign=halign, color=color)
        except Exception:
            header = kwimage.draw_text_on_image(
                None, text, org=(1, 1),
                valign='top', halign='left', color=color)

        if fit == 'shrink':
            if header.shape[1] > width:
                # print('header.shape = {!r}'.format(header.shape))
                # print('width = {!r}'.format(width))
                header = kwimage.imresize(header, dsize=(width, None))
            elif header.shape[1] < width:
                header = np.pad(header, [(0, 0), ((width - header.shape[1]) // 2, 0), (0, 0)])
            else:
                pass
        else:
            header = kwimage.imresize(header, dsize=(width, None))
    else:
        # Allows for however much height is needed
        if halign == 'left':
            org = (1, 1)
        elif halign == 'center':
            org = (width // 2, 1)
        elif halign == 'right':
            org = (width - 1, 1)
        else:
            raise KeyError(halign)

        header = kwimage.draw_text_on_image(
            {'width': width}, text, org=org,
            valign='top', halign=halign, color=color)

    if stack:
        stacked = kwimage.stack_images([header, image], axis=0, overlap=-1)
        return stacked
    else:
        return header


"""
items = {k.split('_')[1].lower(): 'cv2.' + k for k in dir(cv2) if k.startswith('MORPH_')}
items = ub.sorted_vals(items, key=lambda x: eval(x, {'cv2': cv2}))
print('_CV2_MORPH_MODES = {}'.format(ub.repr2(items, nl=1, sv=1, align=':')))
"""
_CV2_STRUCT_ELEMENTS = {
    'rect'    : cv2.MORPH_RECT,
    'cross'   : cv2.MORPH_CROSS,
    'ellipse' : cv2.MORPH_ELLIPSE,
}


_CV2_MORPH_MODES = {
    'erode'   : cv2.MORPH_ERODE,
    'dilate'  : cv2.MORPH_DILATE,
    'open'    : cv2.MORPH_OPEN,
    'close'   : cv2.MORPH_CLOSE,
    'gradient': cv2.MORPH_GRADIENT,
    'tophat'  : cv2.MORPH_TOPHAT,
    'blackhat': cv2.MORPH_BLACKHAT,
    'hitmiss' : cv2.MORPH_HITMISS,
}


@lru_cache
def _morph_kernel_core(h, w, element):
    struct_shape = _CV2_STRUCT_ELEMENTS.get(element, element)
    element = cv2.getStructuringElement(struct_shape, (h, w))
    return element
    # return np.ones((h, w), np.uint8)


def _morph_kernel(size, element='rect'):
    """
    Example:
        >>> from watch.utils.util_kwimage import *  # NOQA
        >>> from watch.utils.util_kwimage import _CV2_MORPH_MODES  # NOQA
        >>> from watch.utils.util_kwimage import _CV2_STRUCT_ELEMENTS  # NOQA
        >>> kernel = 20
        >>> results = {}
        >>> for element in _CV2_STRUCT_ELEMENTS.keys():
        ...     results[element] = _morph_kernel(kernel, element)
        >>> # xdoc: +REQUIRES(--show)
        >>> import kwplot
        >>> kwplot.autompl()
        >>> pnum_ = kwplot.PlotNums(nSubplots=len(results))
        >>> for k, result in results.items():
        >>>     kwplot.imshow(result, pnum=pnum_(), title=k)
        >>> kwplot.show_if_requested()

    """
    if isinstance(size, int):
        h = size
        w = size
    else:
        h, w = size
        # raise NotImplementedError
    return _morph_kernel_core(h, w, element)


def morphology(data, mode, kernel=5, element='rect', iterations=1):
    """
    Executes a morphological operation.

    Args:
        input (ndarray[dtype=uint8 | float64]): data
            (note if mode is hitmiss data must be uint8)

        mode (str) : morphology mode, can be one of: erode, rect, cross,
            dilate, ellipse, open, close, gradient, tophat, blackhat, or
            hitmiss

        kernel (int | Tuple[int, int]): size of the morphology kernel

        element (str):
            structural element, can be rect, cross, or ellipse.

        iterations (int):
            numer of times to repeat the operation

    TODO:
        borderType
        borderValue

    Example:
        >>> from watch.utils.util_kwimage import *  # NOQA
        >>> from watch.utils.util_kwimage import _CV2_MORPH_MODES  # NOQA
        >>> from watch.utils.util_kwimage import _CV2_STRUCT_ELEMENTS  # NOQA
        >>> #shape = (32, 32)
        >>> shape = (64, 64)
        >>> data = (np.random.rand(*shape) > 0.5).astype(np.uint8)
        >>> import kwimage
        >>> data = kwimage.gaussian_patch(shape)
        >>> data = data / data.max()
        >>> data = kwimage.ensure_uint255(data)
        >>> results = {}
        >>> kernel = 5
        >>> for mode in _CV2_MORPH_MODES.keys():
        ...     for element in _CV2_STRUCT_ELEMENTS.keys():
        ...         results[f'{mode}+{element}'] = morphology(data, mode, kernel=kernel, element=element, iterations=2)
        >>> results['raw'] = data
        >>> # xdoc: +REQUIRES(--show)
        >>> import kwplot
        >>> kwplot.autompl()
        >>> pnum_ = kwplot.PlotNums(nCols=3, nSubplots=len(results))
        >>> for k, result in results.items():
        >>>     kwplot.imshow(result, pnum=pnum_(), title=k)
        >>> kwplot.show_if_requested()

    References:
        https://opencv24-python-tutorials.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_morphological_ops/py_morphological_ops.html

    """
    import cv2
    if data.dtype.kind == 'b':
        data = data.astype(np.uint8)
    kernel = _morph_kernel(kernel, element=element)
    if isinstance(mode, str):
        morph_mode = _CV2_MORPH_MODES[mode]
    elif isinstance(mode, int):
        morph_mode = mode
    else:
        raise TypeError(type(mode))

    new = cv2.morphologyEx(
        data, op=morph_mode, kernel=kernel, iterations=iterations)
    return new


def _auto_kernel_sigma(kernel=None, sigma=None, autokernel_mode='ours'):
    """
    Attempt to determine sigma and kernel size from heuristics

    Example:
        >>> _auto_kernel_sigma(None, None)
        >>> _auto_kernel_sigma(3, None)
        >>> _auto_kernel_sigma(None, 0.8)
        >>> _auto_kernel_sigma(7, None)
        >>> _auto_kernel_sigma(None, 1.4)

    Ignore:
        >>> # xdoctest: +REQUIRES(--demo)
        >>> rows = []
        >>> for k in np.arange(3, 101, 2):
        >>>     s = _auto_kernel_sigma(k, None)[1][0]
        >>>     rows.append({'k': k, 's': s, 'type': 'auto_sigma'})
        >>> #
        >>> sigmas = np.array([r['s'] for r in rows])
        >>> other = np.linspace(0, sigmas.max() + 1, 100)
        >>> sigmas = np.unique(np.hstack([sigmas, other]))
        >>> sigmas.sort()
        >>> for s in sigmas:
        >>>     k = _auto_kernel_sigma(None, s, autokernel_mode='cv2')[0][0]
        >>>     rows.append({'k': k, 's': s, 'type': 'auto_kernel (cv2)'})
        >>> #
        >>> for s in sigmas:
        >>>     k = _auto_kernel_sigma(None, s, autokernel_mode='ours')[0][0]
        >>>     rows.append({'k': k, 's': s, 'type': 'auto_kernel (ours)'})
        >>> import pandas as pd
        >>> df = pd.DataFrame(rows)
        >>> p = df.pivot(['s'], ['type'], ['k'])
        >>> print(p[~p.droplevel(0, axis=1).auto_sigma.isnull()])
        >>> # xdoctest: +REQUIRES(--show)
        >>> import kwplot
        >>> sns = kwplot.autosns()
        >>> sns.lineplot(data=df, x='s', y='k', hue='type')
    """
    import numbers
    if kernel is None and sigma is None:
        kernel = 3

    if kernel is not None:
        if isinstance(kernel, numbers.Integral):
            k_x = k_y = kernel
        else:
            k_x, k_y = kernel

    if sigma is None:
        # https://github.com/egonSchiele/OpenCV/blob/09bab41/modules/imgproc/src/smooth.cpp#L344
        sigma_x = 0.3 * ((k_x - 1) * 0.5 - 1) + 0.8
        sigma_y = 0.3 * ((k_y - 1) * 0.5 - 1) + 0.8
    else:
        if isinstance(sigma, numbers.Number):
            sigma_x = sigma_y = sigma
        else:
            sigma_x, sigma_y = sigma

    if kernel is None:
        if autokernel_mode == 'zero':
            # When 0 computed internally via cv2
            k_x = k_y = 0
        elif autokernel_mode == 'cv2':
            # if USE_CV2_DEF:
            # This is the CV2 definition
            # https://github.com/egonSchiele/OpenCV/blob/09bab41/modules/imgproc/src/smooth.cpp#L387
            depth_factor = 3  # or 4 for non-uint8
            k_x = int(round(sigma_x * depth_factor * 2 + 1)) | 1
            k_y = int(round(sigma_y * depth_factor * 2 + 1)) | 1
        elif autokernel_mode == 'ours':
            # But I think this definition makes more sense because it keeps
            # sigma and the kernel in agreement more often
            """
            # Our hueristic is computed via solving the sigma heuristic for k
            import sympy as sym
            s, k = sym.symbols('s, k', rational=True)
            sa = sym.Rational('3 / 10') * ((k - 1) / 2 - 1) + sym.Rational('8 / 10')
            sym.solve(sym.Eq(s, sa), k)
            """
            k_x = max(3, round(20 * sigma_x / 3 - 7 / 3)) | 1
            k_y = max(3, round(20 * sigma_y / 3 - 7 / 3)) | 1
        else:
            raise KeyError(autokernel_mode)
    sigma = (sigma_x, sigma_y)
    kernel = (k_x, k_y)
    return kernel, sigma


@ub.memoize
def upweight_center_mask(shape):
    """
    Example:
        >>> from watch.utils.util_kwimage import *  # NOQA
        >>> shapes = [32, 64, 96, 128, 256]
        >>> results = {}
        >>> for shape in shapes:
        >>>     results[str(shape)] = upweight_center_mask(shape)
        >>> # xdoc: +REQUIRES(--show)
        >>> import kwplot
        >>> kwplot.autompl()
        >>> pnum_ = kwplot.PlotNums(nSubplots=len(results))
        >>> for k, result in results.items():
        >>>     kwplot.imshow(result, pnum=pnum_(), title=k)
        >>> kwplot.show_if_requested()
    """
    import kwimage
    shape, sigma = _auto_kernel_sigma(kernel=shape)
    sigma_x, sigma_y = sigma
    weights = kwimage.gaussian_patch(shape, sigma=(sigma_x, sigma_y))
    weights = weights / weights.max()
    # weights = kwimage.ensure_uint255(weights)
    kernel = np.maximum(np.array(shape) // 8, 3)
    kernel = kernel + (1 - (kernel % 2))
    weights = morphology(
        weights, kernel=kernel, mode='dilate', element='rect', iterations=1)
    weights = kwimage.ensure_float01(weights)
    weights = np.maximum(weights, 0.001)
    return weights


def ensure_false_color(canvas, method='ortho'):
    """
    Given a canvas with more than 3 colors, (or 2 colors) do
    something to get it into a colorized space.

    TODO:
        - [ ] I have no idea how well this works. Probably better methods exist. Find them.

    Example:
        >>> import kwimage
        >>> import numpy as np
        >>> demo_img = kwimage.ensure_float01(kwimage.grab_test_image('astro'))
        >>> canvas = demo_img @ np.random.rand(3, 2)
        >>> rgb_canvas2 = ensure_false_color(canvas)
        >>> canvas = np.tile(demo_img, (1, 1, 10))
        >>> rgb_canvas10 = ensure_false_color(canvas)
        >>> # xdoctest: +REQUIRES(--show)
        >>> import kwplot
        >>> kwplot.autompl()
        >>> kwplot.imshow(rgb_canvas2, pnum=(1, 2, 1))
        >>> kwplot.imshow(rgb_canvas10, pnum=(1, 2, 2))
    """
    import kwarray
    import numpy as np
    import kwimage
    canvas = kwarray.atleast_nd(canvas, 3)

    if canvas.shape[2] in {1, 3}:
        rgb_canvas = canvas
    # elif canvas.shape[2] == 2:
    #     # Use LAB to colorize
    #     L_part = np.ones_like(canvas[..., 0:1]) * 50
    #     a_min = -86.1875
    #     a_max = 98.234375
    #     b_min = -107.859375
    #     b_max = 94.46875
    #     a_part = (canvas[..., 0:1] - a_min) / (a_max - a_min)
    #     b_part = (canvas[..., 1:2] - b_min) / (b_max - b_min)
    #     lab_canvas = np.concatenate([L_part, a_part, b_part], axis=2)
    #     rgb_canvas = kwimage.convert_colorspace(lab_canvas, src_space='lab', dst_space='rgb')
    else:

        if method == 'ortho':
            rng = kwarray.ensure_rng(canvas.shape[2])
            seedmat = rng.rand(canvas.shape[2], 3).T
            h, tau = np.linalg.qr(seedmat, mode='raw')
            false_colored = (canvas @ h)
            rgb_canvas = kwimage.normalize(false_colored)
        elif method == 'PCA':
            import sklearn
            ndim = canvas.ndim
            dims = canvas.shape[0:2]
            if ndim == 2:
                in_channels = 1
            else:
                in_channels = canvas.shape[2]

            if in_channels > 1:
                model = sklearn.decomposition.PCA(1)
                X = canvas.reshape(-1, in_channels)
                X_ = model.fit_transform(X)
                gray = X_.reshape(dims)
                viz = kwimage.make_heatmask(gray, with_alpha=1)[:, :, 0:3]
            else:
                gray = canvas.reshape(dims)
                viz = gray
            return viz
    return rgb_canvas


def colorize_label_image(labels, with_legend=True):
    """
    Replace an image with integer labels with colors
    """
    import kwimage
    label_colors = kwimage.Color.distinct(labels.max())
    index_to_color = np.array([kwimage.Color('black').as01()] + label_colors)
    colored_label_img = index_to_color[labels]
    if with_legend:
        import kwplot
        legend = kwplot.make_legend_img(ub.dzip(range(len(index_to_color)), index_to_color))
        canvas = kwimage.stack_images([colored_label_img, legend], axis=1, resize='smaller')
    else:
        colored_label_img = canvas
    return canvas


def local_variance(image, kernel, handle_nans=True):
    """
    The local variance at each point in the image (take the sqrt to get the
    local std)

    Args:
        image (ndarray)
        kernel (int | Tuple[int, int]) kernel size (w, h)

    Returns:
        ndarray: the image with the variance at each point

    References:
        https://answers.opencv.org/question/193393/local-mean-and-variance/
        https://stackoverflow.com/questions/11456565/opencv-mean-sd-filter

    Example:
        >>> # Test with nans
        >>> from watch.utils.util_kwimage import *  # NOQA
        >>> import kwimage
        >>> import kwarray
        >>> shape = (512, 512)
        >>> dsize = shape[::-1]
        >>> image = np.zeros(shape + (3,), dtype=np.uint8)
        >>> image = kwimage.ensure_float01(image)
        >>> rng = kwarray.ensure_rng(0)
        >>> image[:, 256:, :] = rng.rand(512, 256, 3)  # high frequency noise
        >>> poly1 = kwimage.Polygon.random(rng=rng).scale(dsize)
        >>> poly2 = kwimage.Polygon.random(rng=rng).scale(dsize)
        >>> poly3 = kwimage.Polygon.random(rng=rng).scale(dsize)
        >>> poly1.draw_on(image, color='kitware_blue')
        >>> poly2.draw_on(image, color='pink')
        >>> poly3.draw_on(image, color='kitware_green')
        >>> image[50:70, :, :] = 1  # a line of ones
        >>> image[150:170, :, :] = np.nan  # a line of nans
        >>> #image = kwimage.convert_colorspace(image, 'rgb', 'gray')
        >>> varimg = local_variance(image, kernel=7)
        >>> # xdoctest: +REQUIRES(--show)
        >>> import kwplot
        >>> kwplot.autompl()
        >>> kwplot.imshow(kwimage.fill_nans_with_checkers(image), pnum=(1, 2, 1), title='input image')
        >>> kwplot.imshow(kwimage.fill_nans_with_checkers(kwarray.normalize(varimg)), pnum=(1, 2, 2), title='variance image')
    """
    ksize = (kernel, kernel) if isinstance(kernel, int) else kernel

    image_f = image.astype(np.float32, copy=True)
    invalid_mask = np.isnan(image_f)
    has_mask = np.any(invalid_mask)
    if has_mask:
        if len(invalid_mask.shape) > 2:
            invalid_mask = invalid_mask.any(axis=2)
        # What is a good replacement value?
        image_f[invalid_mask] = 0
    local_mean = cv2.boxFilter(image_f, ddepth=-1, ksize=ksize)
    diff = (image_f - local_mean)
    square_diff = diff * diff
    local_vari = cv2.boxFilter(square_diff, ddepth=-1, ksize=ksize)
    if has_mask:
        local_vari[invalid_mask] = np.nan
    return local_vari


def find_lowvariance_regions(image, kernel=7):
    """
    The idea is that we want to detect large region in an image that are filled
    entirely with the same color.

    The approach is that we are going to find the local variance of the image
    in a KxK window (K is the size of a kernel and corresponds to a minimum
    size of homogenous region that we care to segment).  Then we are going to
    find all regions with zero variance. The connected components of that
    binary image should be roughly what we want.

    We can postprocess this with floodfills to get nearly exacly what we want.

    Example:
        >>> from watch.utils.util_kwimage import *  # NOQA
        >>> import kwimage
        >>> import kwarray
        >>> shape = (512, 512)
        >>> dsize = shape[::-1]
        >>> image = np.zeros(shape + (3,), dtype=np.uint8)
        >>> rng = kwarray.ensure_rng(0)
        >>> image[:, 256:, :] = (rng.rand(512, 256, 3) * 255)  # high frequency noise
        >>> poly1 = kwimage.Polygon.random(rng=rng).scale(dsize)
        >>> poly2 = kwimage.Polygon.random(rng=rng).scale(dsize)
        >>> poly3 = kwimage.Polygon.random(rng=rng).scale(dsize)
        >>> poly1.draw_on(image, color='kitware_blue')
        >>> poly2.draw_on(image, color='pink')
        >>> poly3.draw_on(image, color='kitware_green')
        >>> image[50:70, :, :] = 255  # a "thin" line
        >>> labels = find_lowvariance_regions(image)
        >>> # xdoctest: +REQUIRES(--show)
        >>> import kwplot
        >>> kwplot.autompl()
        >>> canvas = colorize_label_image(labels)
        >>> kwplot.imshow(image, pnum=(1, 2, 1), title='input image')
        >>> kwplot.imshow(canvas, pnum=(1, 2, 2), title='labeled regions')

    Example:
        >>> # Test with nans
        >>> from watch.utils.util_kwimage import *  # NOQA
        >>> import kwimage
        >>> import kwarray
        >>> shape = (512, 512)
        >>> dsize = shape[::-1]
        >>> image = np.zeros(shape + (3,), dtype=np.uint8)
        >>> image = kwimage.ensure_float01(image)
        >>> rng = kwarray.ensure_rng(0)
        >>> image[:, 256:, :] = rng.rand(512, 256, 3)  # high frequency noise
        >>> poly1 = kwimage.Polygon.random(rng=rng).scale(dsize)
        >>> poly2 = kwimage.Polygon.random(rng=rng).scale(dsize)
        >>> poly3 = kwimage.Polygon.random(rng=rng).scale(dsize)
        >>> poly1.draw_on(image, color='kitware_blue')
        >>> poly2.draw_on(image, color='pink')
        >>> poly3.draw_on(image, color='kitware_green')
        >>> image[50:70, :, :] = 1  # a line of ones
        >>> image[150:170, :, :] = np.nan  # a line of nans
        >>> image = kwimage.convert_colorspace(image, 'rgb', 'gray')
        >>> labels = find_lowvariance_regions(image, kernel=7)
        >>> # xdoctest: +REQUIRES(--show)
        >>> import kwplot
        >>> kwplot.autompl()
        >>> canvas = colorize_label_image(labels)
        >>> kwplot.imshow(image, pnum=(1, 2, 1), title='input image')
        >>> kwplot.imshow(canvas, pnum=(1, 2, 2), title='labeled regions')
    """
    h, w = image.shape[0:2]
    # standard deviation filter
    # https://stackoverflow.com/questions/11456565/opencv-mean-sd-filter
    kernel = (kernel, kernel) if isinstance(kernel, int) else kernel
    vari_image = local_variance(image, kernel)

    if len(vari_image.shape) > 2:
        binary_image = (vari_image == 0).all(axis=2).astype(np.uint8)
    else:
        binary_image = (vari_image == 0).astype(np.uint8)
    # import kwimage
    labels, info = connected_components(binary_image, with_stats=False)
    return labels


def find_samecolor_regions(image, min_region_size=49, seed_method='grid',
                           connectivity=8):
    """
    Alternative approach to find_samecolor_regions, but the idea is we check a
    set of seed points and perform a flood fill.

    Args:
        image (ndarray):
            image to find regions of the same color

        min_region_size (int):
            the minimum number of pixels in a region for it to be
            considered valid.

        seed_method (str): can be grid or variance

        connectivity (int): cc connectivity. Either 4 or 8.

    References:
        https://docs.opencv.org/3.4/d7/d1b/group__imgproc__misc.html#ga366aae45a6c1289b341d140839f18717

    Example:
        >>> from watch.utils.util_kwimage import *  # NOQA
        >>> import kwimage
        >>> import kwarray
        >>> shape = (512, 512)
        >>> dsize = shape[::-1]
        >>> image = np.zeros(shape + (3,), dtype=np.uint8)
        >>> rng = kwarray.ensure_rng(0)
        >>> image[:, 256:, :] = (rng.rand(512, 256, 3) * 255)  # high frequency noise
        >>> poly1 = kwimage.Polygon.random(rng=rng).scale(dsize)
        >>> poly2 = kwimage.Polygon.random(rng=rng).scale(dsize)
        >>> poly3 = kwimage.Polygon.random(rng=rng).scale(dsize)
        >>> poly1.draw_on(image, color='kitware_blue')
        >>> poly2.draw_on(image, color='pink')
        >>> poly3.draw_on(image, color='kitware_green')
        >>> image[50:70, :, :] = 255  # a "thin" line
        >>> #labels = find_samecolor_regions(image, seed_method='grid')
        >>> labels = find_samecolor_regions(image, seed_method='variance')
        >>> # xdoctest: +REQUIRES(--show)
        >>> import kwplot
        >>> kwplot.autompl()
        >>> canvas = colorize_label_image(labels)
        >>> kwplot.imshow(image, pnum=(1, 2, 1), title='input image')
        >>> kwplot.imshow(canvas, pnum=(1, 2, 2), title='labeled regions')

    Example:
        >>> # Test with nans
        >>> from watch.utils.util_kwimage import *  # NOQA
        >>> import kwimage
        >>> import kwarray
        >>> shape = (512, 512)
        >>> dsize = shape[::-1]
        >>> image = np.zeros(shape + (3,), dtype=np.uint8)
        >>> image = kwimage.ensure_float01(image).astype(np.float32)
        >>> rng = kwarray.ensure_rng(0)
        >>> image[:, 256:, :] = rng.rand(512, 256, 3)  # high frequency noise
        >>> poly1 = kwimage.Polygon.random(rng=rng).scale(dsize)
        >>> poly2 = kwimage.Polygon.random(rng=rng).scale(dsize)
        >>> poly3 = kwimage.Polygon.random(rng=rng).scale(dsize)
        >>> poly1.draw_on(image, color='kitware_blue')
        >>> poly2.draw_on(image, color='pink')
        >>> poly3.draw_on(image, color='kitware_green')
        >>> image[50:70, :, :] = 1  # a line of ones
        >>> image[150:170, :, :] = np.nan  # a line of nans
        >>> image = kwimage.convert_colorspace(image, 'rgb', 'gray')
        >>> labels = find_samecolor_regions(image, seed_method='grid')
        >>> #labels = find_samecolor_regions(image, seed_method='variance')
        >>> # xdoctest: +REQUIRES(--show)
        >>> import kwplot
        >>> kwplot.autompl()
        >>> canvas = colorize_label_image(labels)
        >>> kwplot.imshow(image, pnum=(1, 2, 1), title='input image')
        >>> kwplot.imshow(canvas, pnum=(1, 2, 2), title='labeled regions')

    Ignore:
        import xdev
        xdev.profile_now(find_lowvariance_regions)(image)
        xdev.profile_now(find_samecolor_regions)(image)
        find_samecolor_regions(image)

        import timerit
        ti = timerit.Timerit(30, bestof=3, verbose=2)
        for timer in ti.reset('time'):
            with timer:
                find_lowvariance_regions(image)

        ti = timerit.Timerit(30, bestof=3, verbose=2)
        for timer in ti.reset('time'):
            with timer:
                find_samecolor_regions(image)
    """
    import cv2
    import kwimage
    h, w = image.shape[0:2]

    if not image.flags['C_CONTIGUOUS']:
        image = np.ascontiguousarray(image)

    # Enumerate a set of pixel positions that we will try to flood fill.
    if seed_method == 'grid':
        # Seed method, uniform grid
        # This method is a lot faster, but it will miss any component
        # that a sampling point doesn't land on.
        stride = int(np.ceil(np.sqrt(min_region_size)))
        x_grid = np.arange(0, w, stride)
        y_grid = np.arange(0, w, stride)
        x_locs, y_locs = np.meshgrid(x_grid, y_grid)
        x_locs = x_locs.ravel()
        y_locs = y_locs.ravel()
        check_xy = np.stack([x_locs, y_locs], axis=1)
    elif seed_method == 'variance':
        # Seed method, low variance
        ksize = int(np.ceil(np.sqrt(min_region_size)))
        ksize = ksize + (1 - (ksize % 2))
        kernel = (ksize, ksize)
        seed_labels = find_lowvariance_regions(image, kernel)
        unique_labels, unique_pos = np.unique(seed_labels, return_index=True)
        seed_y, seed_x = np.unravel_index(unique_pos, seed_labels.shape)
        seed_xy = np.stack([seed_x, seed_y], axis=1)
        check_xy = seed_xy[unique_labels > 0]
        # return seed_labels
    else:
        raise KeyError(seed_method)

    # Initialize the floodfill mask and our output labels
    accum_labels = np.zeros((h + 2, w + 2), dtype=np.uint8)
    mask = accum_labels.copy()
    mask[0, :] = 1
    mask[-1, :] = 1
    mask[:, 0] = 1
    mask[:, -1] = 1

    # Initialize floodfill flags
    ff_flags_base = 0
    ff_flags_base |= connectivity
    ff_flags_base |= cv2.FLOODFILL_FIXED_RANGE
    ff_flags_base |= cv2.FLOODFILL_MASK_ONLY

    # Start at 2 because 1 is used as an internal value
    cluster_label = 2
    for check_x, check_y in check_xy:
        already_filled = accum_labels[check_y + 1, check_x + 1]
        if not already_filled:
            seed_point = (check_x, check_y)
            # The value of the mask is specified in the flags Note: we can only
            # handle 254 different regions, which should be fine, but its a
            # limitaiton (we could work around it if needed)
            ff_flags = ff_flags_base | (cluster_label << 8)
            num, im, mask, rect = cv2.floodFill(image, mask=mask, seedPoint=seed_point,
                                                newVal=1, flags=ff_flags)
            if num > min_region_size:
                # Accept this as a cluster of similar colors
                if 1:
                    # Faster method where we only copy data in the filled region
                    fx, fy, fw, fh = rect
                    sl = kwimage.Boxes(np.array([
                        [fx, fy, fw + 1, fh + 1]]), 'xywh').to_slices()[0]
                    mask_part = mask[sl]
                    label_part = accum_labels[sl]
                    label_part[mask_part == cluster_label] = cluster_label
                else:
                    accum_labels[mask == cluster_label] = cluster_label
                cluster_label += 1

    final_labels = accum_labels[1:-1, 1:-1]
    # is_labeled = final_labels
    # Make labeles start at 1 instead of 2.
    # final_labels[is_labeled] = final_labels[is_labeled] - 1
    return final_labels


def connected_components(image, connectivity=8, ltype=np.int32,
                         with_stats=True, algo='default'):
    """
    TODO: remove and use when landed in kwiamge >= 0.9.5

    Find connected components in a binary image.

    Wrapper around :func:`cv2.connectedComponentsWithStats`.

    Args:
        image (ndarray): a binary uint8 image. Zeros denote the background, and
            non-zeros numbers are foreground regions that will be partitioned
            into connected components.

        connectivity (int): either 4 or 8

        ltype (dtype | str | int):
            The dtype for the output label array.
            Can be either 'int32' or 'uint16', and this can be specified as a
            cv2 code or a numpy dtype.

        algo (str):
            The underlying algorithm to use. See [Cv2CCAlgos]_ for details.
            Options are spaghetti, sauf, bbdt. (default is spaghetti)

    Returns:
        Tuple[ndarray, dict]:
            The label array and an information dictionary

    TODO:
        Document the details of which type of coordinates we are using.
        I.e. are pixels points or areas? (I think this uses the points
        convention?)

    Note:
        opencv 4.5.5 will segfault if connectivity=4
        See: https://github.com/opencv/opencv/issues/21366

    References:
        .. [SO35854197] https://stackoverflow.com/questions/35854197/how-to-use-opencvs-connectedcomponentswithstats-in-python
        .. [Cv2CCAlgos] https://docs.opencv.org/3.4/d3/dc0/group__imgproc__shape.html#ga5ed7784614678adccb699c70fb841075

    CommandLine:
        xdoctest -m kwimage.im_cv2 connected_components:0 --show

    Example:
        >>> # xdoctest: +SKIP
        >>> import kwimage
        >>> from kwimage.im_cv2 import *  # NOQA
        >>> mask = kwimage.Mask.demo()
        >>> image = mask.data
        >>> labels, info = connected_components(image)
        >>> # xdoc: +REQUIRES(--show)
        >>> import kwplot
        >>> kwplot.autompl()
        >>> canvas0 = kwimage.atleast_3channels(mask.data * 255)
        >>> canvas2 = canvas0.copy()
        >>> canvas3 = canvas0.copy()
        >>> boxes = info['label_boxes']
        >>> centroids = info['label_centroids']
        >>> label_colors = kwimage.Color.distinct(info['num_labels'])
        >>> index_to_color = np.array([kwimage.Color('black').as01()] + label_colors)
        >>> canvas2 = centroids.draw_on(canvas2, color=label_colors, radius=None)
        >>> boxes.draw_on(canvas3, color=label_colors, thickness=1)
        >>> legend = kwplot.make_legend_img(ub.dzip(range(len(index_to_color)), index_to_color))
        >>> colored_label_img = index_to_color[labels]
        >>> canvas1 = kwimage.stack_images([colored_label_img, legend], axis=1, resize='smaller')
        >>> kwplot.imshow(canvas0, pnum=(1, 4, 1), title='input image')
        >>> kwplot.imshow(canvas1, pnum=(1, 4, 2), title='label image (colored w legend)')
        >>> kwplot.imshow(canvas2, pnum=(1, 4, 3), title='component centroids')
        >>> kwplot.imshow(canvas3, pnum=(1, 4, 4), title='component bounding boxes')
    """

    if isinstance(ltype, str):
        if ltype in {'int32', 'CV2_32S'}:
            ltype = np.int32
        elif ltype in {'uint16', 'CV_16U'}:
            ltype = np.uint16
    if ltype is np.int32:
        ltype = cv2.CV_32S
    elif ltype is np.int16:
        ltype = cv2.CV_16U
    if not isinstance(ltype, int):
        raise TypeError('type(ltype) = {}'.format(type(ltype)))

    # It seems very easy for a segfault to happen here.
    image = np.ascontiguousarray(image)
    if image.dtype.kind != 'u' or image.dtype.itemsize != 1:
        raise ValueError('input image must be a uint8')

    if algo != 'default':
        if algo in {'spaghetti', 'bolelli'}:
            ccltype = cv2.CCL_SPAGHETTI
        elif algo in {'sauf', 'wu'}:
            ccltype = cv2.CCL_SAUF
        elif algo in {'bbdt', 'grana'}:
            ccltype = cv2.CCL_BBDT
        else:
            raise KeyError(algo)

        if with_stats:
            num_labels, labels, stats, centroids = cv2.connectedComponentsWithStatsWithAlgorithm(
                image, connectivity=connectivity, ccltype=ccltype, ltype=ltype)
        else:
            num_labels, labels = cv2.connectedComponentsWithAlgorithm(
                image, connectivity=connectivity, ccltype=ccltype, ltype=ltype)
    else:
        if with_stats:
            num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
                image, connectivity=connectivity, ltype=ltype)
        else:
            num_labels, labels = cv2.connectedComponents(
                image, connectivity=connectivity, ltype=ltype)

    info = {
        'num_labels': num_labels,
    }

    if with_stats:
        # Transform stats into a kwimage boxes object for each label
        import kwimage
        info['label_boxes'] = kwimage.Boxes(stats[:, [
            cv2.CC_STAT_LEFT, cv2.CC_STAT_TOP,
            cv2.CC_STAT_WIDTH, cv2.CC_STAT_HEIGHT]], 'ltwh')
        info['label_areas'] = stats[:, cv2.CC_STAT_AREA]
        info['label_centroids'] = kwimage.Points(xy=centroids)

    return labels, info
