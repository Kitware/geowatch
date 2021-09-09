"""
These functions might be added to kwimage
"""


def draw_header_text(image, text, fit=False, color='red', halign='center'):
    """
    Places a black bar on top of an image and writes text in it

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

    stacked = kwimage.stack_images([header, image], axis=0, overlap=-1)
    return stacked
