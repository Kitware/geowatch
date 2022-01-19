import logging

import dask.array as da
import numpy as np
from tqdm import tqdm

log = logging.getLogger(__name__)


def process_image_chunked(image,
                          process_func,
                          chip_size=(2048, 2048, 3),
                          overlap=(128, 128, 0),
                          output_dtype=np.uint8
                          ):
    log.info('processing image {}'.format(image.shape))

    def process_wrapper(img: np.ndarray, pbar, block_info=None):
        if block_info:
            # get total number of chunks and update the progress bar
            num_chunks = np.prod(block_info[0]['num-chunks'])
            pbar.total = num_chunks
            pbar.refresh()
        try:
            return process_func(img)
        finally:
            pbar.update()

    if image.shape[0] <= chip_size[0] and image.shape[1] <= chip_size[1]:
        overlap = (0, 0, 0)

    # the actual size of the image passed to __process_chip is chunk_size + 2*overlap
    chunk_size = tuple(map(lambda c, o: c - 2 * o, chip_size, overlap))

    image: da.Array = da.asanyarray(image)
    image = image.rechunk(chunk_size)

    pbar = tqdm(unit=' chip')
    pred = image.map_overlap(
        process_wrapper,
        # overlap on each dimension
        depth=overlap,
        # input is w,h,b output is w,h so tell map_overlay that we're dropping axis 2
        drop_axis=2,
        # output will have this dtype
        dtype=output_dtype,
        meta=np.array((), dtype=output_dtype),
        # pass through
        pbar=pbar,
        boundary='none',
    )
    pred = pred.compute(scheduler='single-threaded')
    pbar.close()

    return pred
