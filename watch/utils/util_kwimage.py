"""
These functions might be added to kwimage
"""
from functools import lru_cache
import numpy as np
import cv2


def draw_header_text(image, text, fit=False, color='red', halign='center',
                     stack='auto'):
    """
    Places a black bar on top of an image and writes text in it

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
        >>> canvases = []
        >>> canvases += [draw_header_text(image=image, text='unfit long header ' * 5, fit=False)]
        >>> canvases += [draw_header_text(image=image, text='shrunk long header ' * 5, fit='shrink')]
        >>> canvases += [draw_header_text(image=image, text='left header', fit=False, halign='left')]
        >>> canvases += [draw_header_text(image=image, text='center header', fit=False, halign='center')]
        >>> canvases += [draw_header_text(image=image, text='right header', fit=False, halign='right')]
        >>> canvases += [draw_header_text(image=image, text='shrunk header', fit='shrink', halign='left')]
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

    if fit:
        # TODO: allow a shrink-to-fit only option
        header = kwimage.draw_text_on_image(
            None, text, org=(1, 1),
            valign='top', halign='left', color=color)
        # header = cv2.copyMakeBorder(header, 3, 3, 3, 3,
        #                             cv2.BORDER_CONSTANT)

        if fit == 'shrink':
            if header.shape[1] > width:
                header = kwimage.imresize(header, dsize=(width, None))
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
        >>> from watch.tasks.fusion.predict import *  # NOQA
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
    from watch.utils import util_kwimage
    shape, sigma = _auto_kernel_sigma(kernel=shape)
    sigma_x, sigma_y = sigma
    weights = kwimage.gaussian_patch(shape, sigma=(sigma_x, sigma_y))
    weights = weights / weights.max()
    # weights = kwimage.ensure_uint255(weights)
    kernel = np.maximum(np.array(shape) // 8, 3)
    kernel = kernel + (1 - (kernel % 2))
    weights = util_kwimage.morphology(
        weights, kernel=kernel, mode='dilate', element='rect', iterations=1)
    weights = kwimage.ensure_float01(weights)
    weights = np.maximum(weights, 0.001)
    return weights
