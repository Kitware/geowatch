"""
These functions might be added to kwimage
"""
from functools import lru_cache


def draw_header_text(image, text, fit=False, color='red', halign='center',
                     stack=True):
    """
    Places a black bar on top of an image and writes text in it

    Args:
        stack (bool): if True returns the stacked image, otherwise just returns
            the header.

    Example:
        >>> from watch.utils.util_kwimage import *  # NOQA
        >>> import kwimage
        >>> image = kwimage.grab_test_image()
        >>> canvases = []
        >>> canvases += [draw_header_text(image, 'unfit long header ' * 5, fit=False)]
        >>> canvases += [draw_header_text(image, 'shrunk long header ' * 5, fit='shrink')]
        >>> canvases += [draw_header_text(image, 'left header', fit=False, halign='left')]
        >>> canvases += [draw_header_text(image, 'center header', fit=False, halign='center')]
        >>> canvases += [draw_header_text(image, 'right header', fit=False, halign='right')]
        >>> canvases += [draw_header_text(image, 'shrunk header', fit='shrink', halign='left')]
        >>> canvases += [draw_header_text(image, 'fit header', fit=True, halign='left')]
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
        # Allows for howeverm much height is needed
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
    import numpy as np
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
    import numpy as np
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
