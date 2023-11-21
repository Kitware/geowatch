# from distutils.log import error
import logging

import dask.array as da
import numpy as np
from tqdm import tqdm
import kwarray

log = logging.getLogger(__name__)


def _process_image_chunked_with_kwarray(image,
                                        process_func,
                                        chip_size=(2048, 2048, 3),
                                        overlap=(128, 128, 0),
                                        output_dtype=np.uint8,
                                        verbose=1):

    gh, gw = image.shape[0:2]
    ch, cw = chip_size[0:2]

    # if gh <= ch and gw <= cw:
    #     overlap = 0
    # else:
    #     if (chip_size[0] == 0):
    #         overlap = 0
    #     else:
    #         overlap = float(overlap[0]) / chip_size[0]

    # if gh <= ch and gw <= cw:
    #     stride = 1
    # else:
    #     if (chip_size[0] == 0):
    #         stride = 1
    #     else:
    #         stride = (chip_size[0] - overlap[0], chip_size[1] - overlap[1])
    #         # float(overlap[0]) / chip_size[0]

    # HACK:
    slider = kwarray.SlidingWindow(image.shape[0:2], chip_size[0:2],
                                   # stride=stride,
                                   overlap=0.3,
                                   keepbound=True,
                                   allow_overshoot=True)

    output_shape = slider.input_shape
    stitcher = kwarray.Stitcher(output_shape)

    from geowatch.tasks.fusion.predict import CocoStitchingManager
    for sl in tqdm(slider, desc='sliding window'):

        chip = image[sl]
        new_chip = process_func(chip)

        CocoStitchingManager._stitcher_center_weighted_add(
            stitcher, sl, new_chip)

        # # Basic add that treats all locations equally
        # stitcher.add(sl, new_chip)

    final = stitcher.finalize()

    return final


def _process_image_chunked_with_dask(image, process_func,
                                     chip_size=(2048, 2048, 3),
                                     overlap=(128, 128, 0),
                                     output_dtype=np.uint8, verbose=1):

    def process_wrapper(img: np.ndarray, pbar, block_info=None):
        if block_info:
            # get total number of chunks and update the progress bar
            num_chunks = np.prod(block_info[0]['num-chunks'])
            pbar.total = num_chunks
            pbar.refresh()
        try:
            res = process_func(img)
        finally:
            pbar.update()
        return res

    gh, gw = image.shape[0:2]
    ch, cw = chip_size[0:2]
    if gh <= ch and gw <= cw:
        overlap = (0, 0, 0)

    # the actual size of the image passed to __process_chip is chunk_size + 2*overlap
    chunk_size = tuple(c - 2 * o for c, o in zip(chip_size, overlap))

    image: da.Array = da.asanyarray(image)
    image = image.rechunk(chunk_size)

    mapkw = {
        'boundary': 'none',
        'pbar': tqdm(unit=' chip', disable=not verbose),
    }
    if 0:
        print('overlap = {!r}'.format(overlap))
        print('image = {!r}'.format(image))

    pred = image.map_overlap(
        process_wrapper,
        # overlap on each dimension
        depth=overlap,
        # FIXME: dont do this? Drop after?
        # input is w,h,b output is w,h so tell map_overlay that we're dropping axis 2
        drop_axis=2,
        # output will have this dtype
        dtype=output_dtype,
        # meta=np.array((), dtype=output_dtype),
        # pass through
        **mapkw,
    )

    # Is there a leak or memory issue here?
    scheduler = 'single-threaded'
    # scheduler = 'synchronous'
    pred = pred.compute(scheduler=scheduler)
    mapkw['pbar'].close()

    return pred


def process_image_chunked(image,
                          process_func,
                          chip_size=(2048, 2048, 3),
                          overlap=(128, 128, 0),
                          output_dtype=np.uint8,
                          verbose=1,
                          sliding_window_method='kwarray'):
    """
    Args:
        chip_size : must be less than half of the overlap

    Example:
        >>> from geowatch.tasks.depth.utils import *  # NOQA
        >>> import kwimage
        >>> import kwarray
        >>> image = kwimage.ensure_float01(kwimage.grab_test_image(dsize=(512, 512)))
        >>> nan_poly = kwimage.Polygon.random(rng=None).scale(image.shape[0])
        >>> image = nan_poly.fill(image.copy(), np.nan)
        >>> process_func = lambda x: kwimage.gaussian_blur(x, sigma=7).mean(axis=2)
        >>> non_chunked = process_func(image)
        >>> chip_size = (128, 128, 3)
        >>> overlap = (32, 32, 0)
        >>> output_dtype = np.uint8
        >>> verbose = 0
        >>> print('kwarray')
        >>> result1 = process_image_chunked(image, process_func, chip_size, overlap, output_dtype, verbose=1, sliding_window_method='kwarray')
        >>> print('dask')
        >>> result2 = process_image_chunked(image, process_func, chip_size, overlap, output_dtype, verbose=1, sliding_window_method='dask')
        >>> # xdoctest: +REQUIRES(--show)
        >>> import kwplot
        >>> kwplot.autompl()
        >>> kwplot.imshow(image, pnum=(1, 3, 1), doclf=True)
        >>> kwplot.imshow(result1, pnum=(1, 3, 2), title='kwarray')
        >>> kwplot.imshow(result2, pnum=(1, 3, 3), title='dask')
    """

    if sliding_window_method == 'kwarray':
        return _process_image_chunked_with_kwarray(image, process_func, chip_size, overlap, output_dtype, verbose)
    elif sliding_window_method == 'dask':
        return _process_image_chunked_with_dask(image, process_func, chip_size, overlap, output_dtype, verbose)
    else:
        raise KeyError(sliding_window_method)
