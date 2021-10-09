"""
These functions might be added to kwimage
"""
from functools import lru_cache
import numpy as np


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


@lru_cache
def _morph_kernel_core(h, w):
    return np.ones((h, w), np.uint8)


def _morph_kernel(size):
    if isinstance(size, int):
        h = size
        w = size
    else:
        raise NotImplementedError
    return _morph_kernel_core(h, w)


def morphology(data, mode, kernel=5):
    """
    Executes a morphological operation.

    Args:
        input (ndarray): data
        mode (str) : morphology mode.  currently only open

    Example:
        >>> data = (np.random.rand(32, 32) > 0.5).astype(np.uint8)
        >>> mode = 'open'
        >>> kernel = 5
        >>> morphology(data, mode, kernel=5)

    """
    import cv2
    if data.dtype.kind == 'b':
        data = data.astype(np.uint8)
    kernel = _morph_kernel(kernel)
    if mode == 'open':
        new = cv2.morphologyEx(data, cv2.MORPH_OPEN, kernel)
    elif mode == 'close':
        new = cv2.morphologyEx(data, cv2.MORPH_CLOSE, kernel)
    elif mode == 'dilate':
        new = cv2.morphologyEx(data, cv2.MORPH_DILATE, kernel)
    elif mode == 'erode':
        new = cv2.morphologyEx(data, cv2.MORPH_ERODE, kernel)
    else:
        raise NotImplementedError
    return new
