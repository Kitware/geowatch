"""
These functions might be added to kwimage
"""


def draw_header_text(image, text, fit=False, color='white'):
    """
    Places a black bar on top of an image and writes text in it

    Example:
        >>> from watch.utils.util_kwimage import *  # NOQA
        >>> import kwimage
        >>> image = kwimage.grab_test_image()
        >>> canvas1 = draw_header_text(image, 'a long header' * 5, fit=False)
        >>> canvas2 = draw_header_text(image, 'a long header' * 5, fit=True)
        >>> # xdoctest: +REQUIRES(--show)
        >>> import kwplot
        >>> kwplot.autompl()
        >>> kwplot.imshow(canvas1, pnum=(1, 2, 1))
        >>> kwplot.imshow(canvas2, pnum=(1, 2, 2))
        >>> kwplot.show_if_requested()
    """
    import cv2
    import kwimage
    width = image.shape[1]
    if fit:
        # TODO: allow a shrink-to-fit only option
        header = kwimage.draw_text_on_image(
            None, text, org=(1, 1),
            valign='top', halign='left', color=color)
        header = cv2.copyMakeBorder(header, 3, 3, 3, 3,
                                    cv2.BORDER_CONSTANT)
        header = kwimage.imresize(header, dsize=(width, None))
    else:
        header = kwimage.draw_text_on_image(
            {'width': width}, text, org=(width // 2, 1),
            valign='top', halign='center', color=color)

    stacked = kwimage.stack_images([header, image], axis=0, overlap=-1)
    return stacked
