import logging

import dask.array as da
import numpy as np
from tqdm import tqdm

log = logging.getLogger(__name__)


def process_image_chunked(image,
                          process_func,
                          chip_size=(2048, 2048, 3),
                          overlap=(128, 128, 0),
                          output_dtype=np.uint8,
                          verbose=1):
    """
    Args:
        chip_size : must be less than half of the overlap

    Example:
        >>> from watch.tasks.depth.utils import *  # NOQA
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
        >>> result = process_image_chunked(image, process_func, chip_size, overlap, output_dtype, verbose=0)
        >>> # xdoctest: +REQUIRES(--show)
        >>> import kwplot
        >>> kwplot.autompl()
        >>> kwplot.imshow(image, pnum=(1, 2, 1), doclf=True)
        >>> kwplot.imshow(result, pnum=(1, 2, 2))
    """
    log.info('processing image {}'.format(image.shape))

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
