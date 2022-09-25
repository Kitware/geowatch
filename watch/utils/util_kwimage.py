"""
These functions might be added to kwimage
"""
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
    weights = kwimage.morphology(
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
                           connectivity=8, scale=1.0):
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

        scale (float): scale at which the computation is done.
            Should be a value between 0 and 1. The default is 1.  Setting to
            less than 1 will resize the image, perform the computation, and
            then upsample the output. This can cause a significant speed
            increase at the cost of some accuracy.

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
        for timer in ti.reset('find_lowvariance_regions'):
            with timer:
                find_lowvariance_regions(image)

        ti = timerit.Timerit(30, bestof=3, verbose=2)
        for timer in ti.reset('find_samecolor_regions'):
            with timer:
                labels = find_samecolor_regions(image)


        # Test to see the overhead compared to different levels of downscale / upscale
        ti = timerit.Timerit(30, bestof=3, verbose=2)
        for timer in ti.reset('find_samecolor_regions + resize'):
            with timer:
                labels = find_samecolor_regions(image, scale=0.5)

        # Test to see the overhead compared to different levels of downscale / upscale
        ti = timerit.Timerit(30, bestof=3, verbose=2)
        for timer in ti.reset('find_samecolor_regions + resize'):
            with timer:
                labels = find_samecolor_regions(image, scale=0.25)
    """
    import cv2
    import kwimage

    if scale != 1.0:
        assert 0 < scale <= 1, 'scale should be in the range (0, 1]'
        orig_dsize = image.shape[0:2][::-1]
        image = kwimage.imresize(image, scale=scale, interpolation='nearest')

    h, w = image.shape[0:2]

    if not image.flags['C_CONTIGUOUS'] or not image.flags['OWNDATA']:
        # Cv2 only likes certain types of numpy arrays
        image = np.ascontiguousarray(image).copy()

    # Enumerate a set of pixel positions that we will try to flood fill.
    if seed_method == 'grid':
        # Seed method, uniform grid
        # This method is a lot faster, but it will miss any component
        # that a sampling point doesn't land on.
        stride = int(np.ceil(np.sqrt(min_region_size)))
        x_grid = np.arange(0, w, stride)
        y_grid = np.arange(0, h, stride)
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
            num, im, mask, rect = cv2.floodFill(
                image, mask=mask, seedPoint=seed_point, newVal=1, loDiff=0, upDiff=0,
                # rect=None,
                flags=ff_flags)
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

    if scale != 1.0:
        final_labels = kwimage.imresize(
            final_labels, dsize=orig_dsize, interpolation='nearest')
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
        >>> from watch.utils.util_kwimage import *  # NOQA
        >>> from watch.utils.util_kwimage import _morph_kernel_core, _morph_kernel, _auto_kernel_sigma
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


def polygon_distance_transform(poly, shape, dtype=np.uint8):
    """
    The API needs work, but I think the idea could be useful

    Example:
        >>> from watch.utils.util_kwimage import *  # NOQA
        >>> import cv2
        >>> import kwimage
        >>> poly = kwimage.Polygon.random().scale(32)
        >>> dtype = np.uint8
        >>> shape = (32, 32)
        >>> dist, poly_mask = polygon_distance_transform(poly, shape, dtype)
        >>> # xdoctest: +REQUIRES(--show)
        >>> import kwplot
        >>> kwplot.autompl()
        >>> import kwplot
        >>> kwplot.autompl()
        >>> kwplot.imshow(dist, cmap='viridis', doclf=1)
        >>> poly.draw(fill=0, border=1)
    """
    import cv2
    poly_mask = np.zeros(shape, dtype=dtype)
    poly_mask = poly.fill(poly_mask, value=1)
    dist = cv2.distanceTransform(
        src=poly_mask, distanceType=cv2.DIST_L2, maskSize=3)
    return dist, poly_mask


def devcheck_frame_poly_weights(poly, shape, dtype=np.uint8):
    """
    import kwimage
    import kwplot
    kwplot.autompl()
    from watch.utils import util_kwimage
    space_shape = (380, 380)
    weights1 = util_kwimage.upweight_center_mask(space_shape)
    weights2 = kwimage.normalize(kwimage.gaussian_patch(space_shape))
    sigma3 = 4.8 * ((space_shape[0] - 1) * 0.5 - 1) + 0.8
    weights3 = kwimage.normalize(kwimage.gaussian_patch(space_shape, sigma=sigma3))

    min_spacetime_weight = 0.5

    weights1 = np.maximum(weights1, min_spacetime_weight)
    weights2 = np.maximum(weights2, min_spacetime_weight)
    weights3 = np.maximum(weights3, min_spacetime_weight)

    # Hack so color bar goes to 0
    weights3[0, 0] = 0
    weights2[0, 0] = 0
    weights1[0, 0] = 0

    kwplot.imshow(weights1, pnum=(1, 3, 1), title='current', cmap='viridis', data_colorbar=1)
    kwplot.imshow(weights2, pnum=(1, 3, 2), title='variant1', cmap='viridis', data_colorbar=1)
    kwplot.imshow(weights3, pnum=(1, 3, 3), title='variant2', cmap='viridis', data_colorbar=1)
    """
    import kwimage
    space_shape = (128, 128)
    space_dsize = space_shape[::-1]
    polys = [
        kwimage.Polygon.random().scale(space_dsize).scale(0.25, about='center'),
        kwimage.Polygon.random().scale(space_dsize).scale(0.25, about='center'),
        kwimage.Polygon.random().scale(space_dsize).scale(0.25, about='center'),
        kwimage.Polygon.random().scale(space_dsize).scale(0.25, about='center'),
    ]

    frame_poly_weights_v1 = np.ones(space_shape, dtype=np.float32)
    frame_poly_weights_v2 = np.zeros(space_shape, dtype=np.float32)
    for poly in polys:
        dist, poly_mask = polygon_distance_transform(poly, space_shape)
        max_dist = dist.max()
        if max_dist > 0:
            dist_weight = dist / max_dist
            weight_mask = dist_weight + (1 - poly_mask)
            frame_poly_weights_v1 = frame_poly_weights_v1 * weight_mask
            frame_poly_weights_v2 = np.maximum(frame_poly_weights_v2, dist_weight)

    sigma = (
        (4.8 * ((space_shape[1] - 1) * 0.5 - 1) + 0.8),
        (4.8 * ((space_shape[0] - 1) * 0.5 - 1) + 0.8),
    )
    min_spacetime_weight = 0.5
    frame_poly_weights = frame_poly_weights_v2
    frame_poly_weights = np.maximum(frame_poly_weights, min_spacetime_weight)
    space_weights = kwimage.normalize(kwimage.gaussian_patch(space_shape, sigma=sigma))
    import kwplot
    kwplot.autompl()
    kwplot.imshow(frame_poly_weights_v1, pnum=(1, 3, 1))
    kwplot.imshow(frame_poly_weights, pnum=(1, 3, 2))
    kwplot.imshow(np.maximum(frame_poly_weights, space_weights), pnum=(1, 3, 3))


def find_low_overlap_covering_boxes(polygons, scale, min_box_dim, max_box_dim, merge_thresh=0.001, max_iters=100):
    """
    Given a set of polygons we want to find a small set of boxes that
    completely cover all of those polygons.

    We are going to do some set-cover shenanigans by making a bunch of
    candidate boxes based on some hueristics and find a set cover of those.

    Then we will search for small boxes that can be merged, and iterate.

    References:
        https://aip.scitation.org/doi/pdf/10.1063/1.5090003?cookieSet=1
        Mercantile - https://pypi.org/project/mercantile/0.4/
        BingMapsTiling - XYZ Tiling for webmap services
        https://mercantile.readthedocs.io/en/stable/api/mercantile.html#mercantile.bounding_tile
        https://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.713.6709&rep=rep1&type=pdf

    Ignore:
        >>> # Create random polygons as test data
        >>> import kwimage
        >>> import kwarray
        >>> from kwarray import distributions
        >>> rng = kwarray.ensure_rng(934602708841)
        >>> num = 200
        >>> #
        >>> canvas_width = 2000
        >>> offset_distri = distributions.Uniform(canvas_width, rng=rng)
        >>> scale_distri = distributions.Uniform(10, 150, rng=rng)
        >>> #
        >>> polygons = []
        >>> for _ in range(num):
        >>>     poly = kwimage.Polygon.random(rng=rng)
        >>>     poly = poly.scale(scale_distri.sample())
        >>>     poly = poly.translate(offset_distri.sample(2))
        >>>     polygons.append(poly)
        >>> polygons = kwimage.PolygonList(polygons)
        >>> #
        >>> import kwplot
        >>> plt = kwplot.autoplt()
        >>> kwplot.figure(doclf=1)
        >>> plt.gca().set_xlim(0, canvas_width)
        >>> plt.gca().set_ylim(0, canvas_width)
        >>> _ = polygons.draw(fill=0, border=1, color='pink')
        >>> #
        >>> scale = 1.0
        >>> min_box_dim = 240
        >>> max_box_dim = 500
        >>> #
        >>> keep_bbs, overlap_idxs = find_low_overlap_covering_boxes(polygons, scale, min_box_dim, max_box_dim)
        >>> # xdoctest: +REQUIRES(--show)
        >>> #
        >>> import kwplot
        >>> plt = kwplot.autoplt()
        >>> kwplot.figure(fnum=1, doclf=1)
        >>> polygons.draw(color='pink')
        >>> # candidate_bbs.draw(color='blue', setlim=1)
        >>> keep_bbs.draw(color='orange', setlim=1)
        >>> plt.gca().set_title('find_low_overlap_covering_boxes')
    """
    import kwimage
    import kwarray
    import numpy as np
    import geopandas as gpd
    import ubelt as ub
    from watch.utils import util_gis
    import networkx as nx

    polygons_sh = [p.to_shapely() for p in polygons]
    polygons_gdf = gpd.GeoDataFrame(geometry=polygons_sh)

    polybbs = kwimage.Boxes.concatenate([p.to_boxes() for p in polygons])
    initial_candiate_bbs = polybbs.scale(scale, about='center')
    initial_candiate_bbs = initial_candiate_bbs.to_cxywh()
    initial_candiate_bbs.data[..., 2] = np.maximum(initial_candiate_bbs.data[..., 2], min_box_dim)
    initial_candiate_bbs.data[..., 3] = np.maximum(initial_candiate_bbs.data[..., 3], min_box_dim)

    candidate_bbs = initial_candiate_bbs

    def refine_candidates(candidate_bbs, iter_idx):
        # Add some translated boxes to the mix to see if they do any better
        extras = [
            candidate_bbs.translate((-min_box_dim / 10, 0)),
            candidate_bbs.translate((+min_box_dim / 10, 0)),
            candidate_bbs.translate((0, -min_box_dim / 10)),
            candidate_bbs.translate((0, +min_box_dim / 10)),
            candidate_bbs.translate((-min_box_dim / 3, 0)),
            candidate_bbs.translate((+min_box_dim / 3, 0)),
            candidate_bbs.translate((0, -min_box_dim / 3)),
            candidate_bbs.translate((0, +min_box_dim / 3)),
        ]
        candidate_bbs = kwimage.Boxes.concatenate([candidate_bbs] + extras)

        # Find the minimum boxes that cover all of the regions
        # xs, ys = centroids.T
        # ws = hs = np.full(len(xs), fill_value=site_meters)
        # utm_boxes = kwimage.Boxes(np.stack([xs, ys, ws, hs], axis=1), 'cxywh').to_xywh()

        boxes_gdf = gpd.GeoDataFrame(geometry=candidate_bbs.to_shapley(), crs=polygons_gdf.crs)
        box_poly_overlap = util_gis.geopandas_pairwise_overlaps(boxes_gdf, polygons_gdf, predicate='contains')
        cover_idxs = list(kwarray.setcover(box_poly_overlap).keys())
        keep_bbs = candidate_bbs.take(cover_idxs)
        box_ious = keep_bbs.ious(keep_bbs)

        if iter_idx > 0:
            # Dont do it on the first iter to compare to old algo
            laplace = box_ious - np.diag(np.diag(box_ious))
            mergable = laplace > merge_thresh
            g = nx.Graph()
            g.add_edges_from(list(zip(*np.where(mergable))))
            cliques = sorted(nx.find_cliques(g), key=len)[::-1]

            used = set()
            merged_boxes = []
            for clique in cliques:
                if used & set(clique):
                    continue

                new_box = keep_bbs.take(clique).bounding_box()
                w = new_box.width.ravel()[0]
                h = new_box.height.ravel()[0]
                if w < max_box_dim and h < max_box_dim:
                    merged_boxes.append(new_box)
                    used.update(clique)

            unused = sorted(set(range(len(keep_bbs))) - used)
            post_merge_bbs = kwimage.Boxes.concatenate([keep_bbs.take(unused)] + merged_boxes)

            boxes_gdf = gpd.GeoDataFrame(geometry=post_merge_bbs.to_shapley(), crs=polygons_gdf.crs)
            box_poly_overlap = util_gis.geopandas_pairwise_overlaps(boxes_gdf, polygons_gdf, predicate='contains')
            cover_idxs = list(kwarray.setcover(box_poly_overlap).keys())
            new_cand_bbs = post_merge_bbs.take(cover_idxs)
        else:
            new_cand_bbs = keep_bbs

        new_cand_overlaps = list(ub.take(box_poly_overlap, cover_idxs))
        return new_cand_bbs, new_cand_overlaps

    new_cand_overlaps = None

    for iter_idx in range(max_iters):
        old_candidate_bbs = candidate_bbs
        candidate_bbs, new_cand_overlaps = refine_candidates(candidate_bbs, iter_idx)
        num_old = len(old_candidate_bbs)
        num_new = len(candidate_bbs)
        if num_old == num_new:
            residual = (old_candidate_bbs.data - candidate_bbs.data).max()
            if residual > 0:
                print('improving residual = {}'.format(ub.repr2(residual, nl=1)))
            else:
                print('converged')
                break
        else:
            print(f'improving: {num_old} -> {num_new}')
    else:
        print('did not converge')
    keep_bbs = candidate_bbs
    overlap_idxs = new_cand_overlaps

    if 0:
        import kwplot
        kwplot.autoplt()
        kwplot.figure(fnum=1, doclf=1)
        polygons.draw(color='pink')
        # candidate_bbs.draw(color='blue', setlim=1)
        keep_bbs.draw(color='orange', setlim=1)

    return keep_bbs, overlap_idxs


def find_low_overlap_covering_boxes_optimize(polygons, scale, min_box_dim, max_box_dim, merge_thresh=0.001, max_iters=100):
    """
    A variant of the covering problem that doesn't work that well, but might in
    the future with tweaks.

    Ignore:
        >>> # Create random polygons as test data
        >>> import kwimage
        >>> import kwarray
        >>> from kwarray import distributions
        >>> rng = kwarray.ensure_rng(934602708841)
        >>> num = 200
        >>> #
        >>> canvas_width = 2000
        >>> offset_distri = distributions.Uniform(canvas_width, rng=rng)
        >>> scale_distri = distributions.Uniform(10, 150, rng=rng)
        >>> #
        >>> polygons = []
        >>> for _ in range(num):
        >>>     poly = kwimage.Polygon.random(rng=rng)
        >>>     poly = poly.scale(scale_distri.sample())
        >>>     poly = poly.translate(offset_distri.sample(2))
        >>>     polygons.append(poly)
        >>> polygons = kwimage.PolygonList(polygons)
        >>> #
        >>> import kwplot
        >>> plt = kwplot.autoplt()
        >>> kwplot.figure(doclf=1)
        >>> plt.gca().set_xlim(0, canvas_width)
        >>> plt.gca().set_ylim(0, canvas_width)
        >>> _ = polygons.draw(fill=0, border=1, color='pink')
        >>> #
        >>> scale = 1.0
        >>> min_box_dim = 240
        >>> max_box_dim = 500
        >>> #

    """
    import kwimage
    # import kwarray

    start_scale = 2.0
    polygon_boxes = kwimage.Boxes.concatenate([p.to_boxes() for p in polygons]).to_ltrb()
    candidate_bbs = polygon_boxes.scale(start_scale, about='center').to_ltrb()
    orig_candidates = candidate_bbs.copy()
    import torch

    device = 'cpu'
    # device = 0
    polygon_ltrb = polygon_boxes.tensor().data.float().to(device)
    candidate_ltrb = torch.nn.Parameter(candidate_bbs.tensor().data.float().to(device))

    # These will be soft bits that will indicate 1 or 0, and we will try to
    # force into an integer solution via rounding.
    indicator_logits = torch.nn.Parameter(
        torch.rand(len(candidate_ltrb), dtype=torch.float, device=candidate_ltrb.device) * 0.2 + 0.8
    )

    parameters = [
        candidate_ltrb,
        indicator_logits,
    ]
    import torch_optimizer as optim
    optimizer = optim.RangerQH(parameters, lr=1e-2)
    # from torch.optim import SGD
    # optimizer = SGD(parameters, lr=1-1, weight_decay=1e-7)
    # from torch.optim import AdamW
    # optimizer = AdamW(parameters, lr=1e-3, weight_decay=1e-6)

    target_boxes = kwimage.Boxes(polygon_ltrb, 'ltrb')
    cover_boxes = kwimage.Boxes(candidate_ltrb, 'ltrb')

    # ltrb1 = target_boxes.data
    # ltrb2 = cover_boxes.data
    # _impl = target_boxes._impl

    # num_targets = len(target_boxes)
    target_area = target_boxes.area.sum()
    # eps = kwarray.dtype_info(target_area.dtype).eps

    denom = target_area

    disatisfaction_penalty = 100

    def forward():
        # TODO: restrict what boxes can cover what objects via grouping to
        # reduce computational complexity here.
        iooa = target_boxes.iooas(cover_boxes)

        areas = target_boxes.area
        self_ious = target_boxes.ious(target_boxes, impl='py')

        indicator_bits = indicator_logits.sigmoid()
        chosen_area = (indicator_bits * (areas / denom)).sum()
        chosen_self_overlap = (indicator_bits * self_ious).sum()

        # We want to minimize...
        objective = (
            # Total chosen area covered
            chosen_area +
            # Overlap of the chosen boxes
            chosen_self_overlap
        )

        # Subject to the constraint (which we relax for optimization)
        relaxed_iooa = iooa * indicator_bits[:, None]

        # TODO: Getting this loss right is the key to this problem.
        # The current version doesn't work that well. But a more numerically
        # stable version might do better.

        # All of the polygons must be completely covered by at least one box
        sat_critical = relaxed_iooa.max(dim=0)[0].min()
        sat_overall = relaxed_iooa.max(dim=0)[0].mean()
        satisfaction = (sat_critical + sat_overall) / 2

        bottom_line_loss = disatisfaction_penalty * (1 - sat_critical)
        overall_sat_loss = disatisfaction_penalty * (1 - sat_overall)
        loss = objective / (satisfaction + 0.01) + bottom_line_loss + overall_sat_loss
        # loss = bottom_line_loss + overall_sat_loss

        outputs = {
            'item_losses': {
                'chosen_area': chosen_area[None, None, ...],
                'chosen_self_overlap': chosen_self_overlap[None, None, ...],
                'sat_critical': sat_critical[None, None, ...],
                'sat_overall': sat_overall[None, None, ...],
                'satisfaction': satisfaction[None, None, ...],
                'total': loss[None, None, ...],
            },
            'loss': loss,
        }
        return outputs

    if 1:
        prog = ub.ProgIter(range(100000))
        for i in prog:
            optimizer.zero_grad()
            outputs = forward()
            loss = outputs['loss']
            loss.backward()
            total_grad = candidate_ltrb.grad.sum()
            mean_grad = total_grad / candidate_ltrb.numel()
            drift = (candidate_ltrb.data - orig_candidates.data).abs().max().item()
            sat_overall = outputs['item_losses']['sat_overall'].sum().item()
            sat_critical = outputs['item_losses']['sat_critical'].sum().item()
            prog.set_extra(f'{loss=} {total_grad=} {mean_grad=} {drift=} {sat_critical=} {sat_overall=}')
            optimizer.step()

    else:

        def draw_batch():
            import kwarray
            indicator_bits = kwarray.ArrayAPI.numpy(indicator_logits.sigmoid())
            orig_candidates.draw(color='red', linewidth=6)
            cover_boxes.numpy().draw(color='blue', setlim=1, alpha=indicator_bits, u=3)
            target_boxes.numpy().draw(color='orange', setlim='grow', linwidth=2)

        import kwplot
        sns = kwplot.autosns()
        plt = kwplot.autoplt()
        kwplot.figure(fnum=1, doclf=1)
        draw_batch()
        plt.gca().set_title('find_low_overlap_covering_boxes')

        import xdev
        fnum = 2
        fig = kwplot.figure(fnum=fnum, doclf=True)
        fig.set_size_inches(15, 6)
        fig.subplots_adjust(left=0.05, top=0.9)
        prev = None
        _frame_idx = 0

        loss_records = []
        loss_records = [g[0] for g in ub.group_items(loss_records, lambda x: x['step']).values()]
        step = 0
        _frame_idx = 0

        for _frame_idx in xdev.InteractiveIter(list(range(_frame_idx + 1, 1000))):
            # for _frame_idx in list(range(_frame_idx, 1000)):
            num_steps = 100
            ex = None
            prog = ub.ProgIter(range(num_steps), desc='optimize')
            for _i in prog:
                optimizer.zero_grad()
                outputs = forward()
                loss = outputs['loss'].sum()
                if torch.any(torch.isnan(loss)):
                    print('NAN OUTPUT!!!')
                    print('loss = {!r}'.format(loss))
                    print('prev = {!r}'.format(prev))
                    ex = Exception('prev = {!r}'.format(prev))
                    break
                # elif loss > 1e4:
                #     # Turn down the learning rate when loss gets huge
                #     scale = (loss / 1e4).detach()
                #     loss /= scale
                prev = loss
                # import netharn as nh
                # item_losses_ = nh.data.collate.default_collate(outputs['item_losses'])
                item_losses_ = outputs['item_losses']
                item_losses = ub.map_vals(lambda x: sum(x).item(), item_losses_)
                loss_records.extend([{'part': key, 'val': val, 'step': step} for key, val in item_losses.items()])
                loss.backward()
                total_grad = candidate_ltrb.grad.sum()
                mean_grad = total_grad / candidate_ltrb.numel()
                drift = (candidate_ltrb.data - orig_candidates.data).abs().max().item()
                sat_overall = outputs['item_losses']['sat_overall'].sum().item()
                sat_critical = outputs['item_losses']['sat_critical'].sum().item()
                prog.set_extra(f'{loss=} {total_grad=} {mean_grad=} {drift=} {sat_critical=} {sat_overall=}')
                optimizer.step()
                step += 1

            draw_batch()

            kwplot.figure(pnum=(1, 2, 1), fnum=fnum, docla=1)
            draw_batch()

            fig = kwplot.figure(fnum=fnum, pnum=(1, 2, 2))
            #kwplot.imshow(canvas, pnum=(1, 2, 1))
            import pandas as pd
            df = pd.DataFrame(loss_records)
            total_df = dict(list((df.groupby('part'))))['total']
            print(total_df)
            ax = sns.lineplot(data=total_df, x='step', y='val', hue='part')
            ax
            # ax.set_ylim(0, df.groupby('part')['val'].median().max())
            # try:
            #     ax.set_yscale('logit')
            # except Exception:
            #     ...
            # from watch.utils.slugify_ext import smart_truncate
            # from kwplot.mpl_make import render_figure_to_image
            # fig.suptitle(smart_truncate(str(optimizer).replace('\n', ''), max_length=64))
            # img = render_figure_to_image(fig)
            # img = kwimage.convert_colorspace(img, src_space='bgr', dst_space='rgb')
            # fpath = join(dpath, 'frame_{:04d}.png'.format(_frame_idx))
            #kwimage.imwrite(fpath, img)
            xdev.InteractiveIter.draw()
            if ex:
                raise ex

    # polygon_boxes.tensor()
    # boxes_gdf = gpd.GeoDataFrame(geometry=candidate_bbs.to_shapley(), crs=polygons_gdf.crs)
    # box_poly_overlap = util_gis.geopandas_pairwise_overlaps(boxes_gdf, polygons_gdf, predicate='contains')
    # cover_idxs = list(kwarray.setcover(box_poly_overlap).keys())
    # keep_bbs = candidate_bbs.take(cover_idxs)
    # box_ious = keep_bbs.ious(keep_bbs)
    # import pulp
    # prob = pulp.LpProblem("Set Cover", pulp.LpMinimize)


class Box(ub.NiceRepr):
    """
    Like kwimage.Boxes, but only one of t/em.

    Currently implemented by storing a Boxes object with one item and indexing
    into it. Could be done more efficiently
    """

    def __init__(self, boxes):
        self.boxes = boxes

    @property
    def format(self):
        return self.boxes.format

    @property
    def data(self):
        return self.boxes.data[0]

    def __nice__(self):
        data_repr = repr(self.data)
        if '\n' in data_repr:
            data_repr = ub.indent('\n' + data_repr.lstrip('\n'), '    ')
        nice = '{}, {}'.format(self.format, data_repr)
        return nice

    @classmethod
    def from_slice(self, slice_):
        import kwimage
        boxes = kwimage.Boxes.from_slice(slice_)
        self = Box(boxes)
        return self

    @classmethod
    def from_shapely(self, geom):
        import kwimage
        boxes = kwimage.Boxes.from_shapely(geom)
        self = Box(boxes)
        return self

    @classmethod
    def from_dsize(self, dsize):
        width, height = dsize
        import kwimage
        boxes = kwimage.Boxes([[0, 0, width, height]], 'ltrb')
        self = Box(boxes)
        return self

    @classmethod
    def coerce(cls, data):
        if isinstance(data, Box):
            return data
        else:
            import kwimage
            return cls(kwimage.Boxes.coerce(data))

    @property
    def dsize(self):
        return (int(self.width), int(self.height))

    def translate(self, *args, **kwargs):
        new_boxes = self.boxes.translate(*args, **kwargs)
        new = self.__class__(new_boxes)
        return new

    def warp(self, *args, **kwargs):
        new_boxes = self.boxes.warp(*args, **kwargs)
        new = self.__class__(new_boxes)
        return new

    def scale(self, *args, **kwargs):
        new_boxes = self.boxes.scale(*args, **kwargs)
        new = self.__class__(new_boxes)
        return new

    def clip(self, *args, **kwargs):
        new_boxes = self.boxes.clip(*args, **kwargs)
        new = self.__class__(new_boxes)
        return new

    def quantize(self, *args, **kwargs):
        new_boxes = self.boxes.quantize(*args, **kwargs)
        new = self.__class__(new_boxes)
        return new

    def copy(self, *args, **kwargs):
        new_boxes = self.boxes.copy(*args, **kwargs)
        new = self.__class__(new_boxes)
        return new

    def round(self, *args, **kwargs):
        new_boxes = self.boxes.round(*args, **kwargs)
        new = self.__class__(new_boxes)
        return new

    def pad(self, *args, **kwargs):
        new_boxes = self.boxes.pad(*args, **kwargs)
        new = self.__class__(new_boxes)
        return new

    def resize(self, *args, **kwargs):
        new_boxes = self.boxes.resize(*args, **kwargs)
        new = self.__class__(new_boxes)
        return new

    def to_ltbr(self, *args, **kwargs):
        return self.__class__(self.boxes.to_ltbr(*args, **kwargs))

    def to_xywh(self, *args, **kwargs):
        return self.__class__(self.boxes.to_xywh(*args, **kwargs))

    def to_cxywh(self, *args, **kwargs):
        return self.__class__(self.boxes.to_cxywh(*args, **kwargs))

    def toformat(self, *args, **kwargs):
        return self.__class__(self.boxes.toformat(*args, **kwargs))

    def corners(self, *args, **kwargs):
        return self.boxes.corners(*args, **kwargs)[0]

    @property
    def width(self):
        return self.boxes.width.ravel()[0]

    @property
    def aspect_ratio(self):
        return self.boxes.aspect_ratio.ravel()[0]

    @property
    def height(self):
        return self.boxes.height.ravel()[0]

    @property
    def tl_x(self):
        return self.boxes.tl_x[0]

    @property
    def tl_y(self):
        return self.boxes.tl_y[0]

    @property
    def br_x(self):
        return self.boxes.br_x[0]

    @property
    def br_y(self):
        return self.boxes.br_y[0]

    @property
    def dtype(self):
        return self.boxes.dtype

    @property
    def area(self):
        return self.boxes.area.ravel()[0]

    def to_slice(self, endpoint=True):
        return self.boxes.to_slices(endpoint=endpoint)[0]

    def to_shapely(self):
        return self.boxes.to_shapely()[0]

    def to_polygon(self):
        return self.boxes.to_polygons()[0]

    def to_coco(self):
        return self.boxes.to_coco()[0]
