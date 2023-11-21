from collections import deque
import itertools as it


def consume(iterator, n=None):
    """
    Consume n items from an iterator and discard them.

    Args:
        iterator (Iterable): an iterator to consume
        n (int | None): number of items to consume (or consume all if None)

    References:
        https://stackoverflow.com/questions/50937966/fastest-most-pythonic-way-to-consume-an-iterator
        https://docs.python.org/3/library/itertools.html#itertools-recipes

    Benchmark:
        >>> from geowatch.utils.util_iter import *  # NOQA
        >>> import timerit
        >>> ti = timerit.Timerit(100, bestof=10, verbose=2)
        >>> #
        >>> def make_iterator():
        >>>     return iter(range(100000))
        >>> #
        >>> for timer in ti.reset('our-consume'):
        >>>     iterator = make_iterator()
        >>>     with timer:
        >>>         consume(iterator)
        >>> #
        >>> for timer in ti.reset('list'):
        >>>     iterator = make_iterator()
        >>>     with timer:
        >>>         list(iterator)
        >>> #
        >>> for timer in ti.reset('consume-100'):
        >>>     iterator = make_iterator()
        >>>     with timer:
        >>>         consume(iterator, n=100)
        >>> #
        >>> for timer in ti.reset('list-100'):
        >>>     iterator = make_iterator()
        >>>     with timer:
        >>>         list(zip(iterator, range(100)))
    """
    if n is None:
        # consume the entire iterator
        deque(iterator, maxlen=0)
    else:
        # advance to the empty slice starting at position n
        next(it.islice(iterator, n, n), None)


def chunks(items, nchunks):
    """

    Note:
        ubelt.chunks does not handle this case (yet)

    TODO:
        - [ ] Fix ubelt.chunks to handle this remainder case

    Example:
        >>> from geowatch.utils.util_iter import *  # NOQA
        >>> items = list(range(11))
        >>> nchunks = 4
        >>> list(chunks(items, nchunks))
    """
    if nchunks == 0:
        return
    num_items = len(items)
    # Basic chunksize
    chunksize = num_items // nchunks
    # How should the remainder be distributed?
    # Evenly at the start or at the end? (for now do it at the start)
    remainder = num_items % nchunks

    # Build an iterator that describes how big each chunk will be
    chunksize_iter = it.chain(
        it.repeat(chunksize + 1, remainder),
        it.repeat(chunksize, nchunks - remainder)
    )
    iterator = iter(items)
    for _chunksize in chunksize_iter:
        yield list(it.islice(iterator, _chunksize))
