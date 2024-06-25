"""
Functions that may be moved to kwutil
"""
import itertools as it


def distributed_subitems(items, max_num):
    """
    Return a subset of items maximally distributed over the input index space.
    I.e. the chosen indexes are maximally dialted.

    Args:
        items (List | Dict): an ordered indexable object

    Returns:
        List | Dict: a subset of the input with a length at most ``max_num``.

    TODO:
        - [ ] Find a better name
        - [ ] Figure out where this lives

    Example:
        >>> from geowatch.utils.util_kwutil import *  # NOQA
        >>> items = list(range(100))
        >>> max_num = 5
        >>> sub_items = distributed_subitems(items, max_num)
        >>> print(sub_items)
        [0, 25, 50, 75, 99]

    Example:
        >>> from geowatch.utils.util_kwutil import *  # NOQA
        >>> items = {chr(i): i for i in range(ord('a'), ord('a') + 26)}
        >>> max_num = 5
        >>> sub_items = distributed_subitems(items, max_num)
        >>> print(sub_items)
        {'a': 97, 'g': 103, 'n': 110, 't': 116, 'z': 122}
    """
    sub_index_gen = farthest_from_previous(0, len(items))
    sub_indices = sorted(it.islice(sub_index_gen, max_num))
    if isinstance(items, dict):
        item_keys = list(items.keys())
        sub_keys = [item_keys[idx] for idx in sub_indices]
        sub_items = {key: items[key] for key in sub_keys}
    else:
        sub_items = [items[idx] for idx in sub_indices]
    return sub_items


def farthest_from_previous(start, stop):
    """
    Given a ordered list of items, incrementally yield indexes such that each
    new index maximizes the distance to all other previously chosen indexes.

    Args:
        start (int): The inclusive starting index (typically 0)
        stop (int): The exclusive maximum index (typically ``len(items)``)

    Yields:
        int: the next chosen index in the series

    TODO:
        - [ ] Find a Better Name

    Notes:
        * This is an instance of farthest-point traversal in 1D.

    References:
        .. [CSSE_167943] https://cs.stackexchange.com/questions/167943/is-this-knapsack-variant-named-studied-online-algorithm-for-farthest-from-pr

    CommandLine:
        xdoctest -m geowatch.utils.util_kwutil farthest_from_previous

    Example:
        >>> from geowatch.utils.util_kwutil import *  # NOQA
        >>> total = 10
        >>> start, stop = 0, 10
        >>> gen = farthest_from_previous(start, stop)
        >>> result = list(gen)
        >>> assert set(result) == set(range(start, stop))
        >>> print(result)
        [9, 0, 5, 2, 7, 1, 6, 3, 8, 4]
    """
    if start < stop:
        yield stop - 1
        yield from _farthest_from_previous_helper(start, stop - 1)


def _farthest_from_previous_helper(start, stop):
    if start < stop:
        low_mid: int = (start + stop) // 2
        high_mid: int = (start + stop + 1) // 2

        left_gen = _farthest_from_previous_helper(start, low_mid)
        right_gen = _farthest_from_previous_helper(high_mid, stop)

        pairgen = it.zip_longest(left_gen, right_gen)
        flatgen = it.chain.from_iterable(pairgen)
        filtgen = filter(lambda x: x is not None, flatgen)
        yield from filtgen
        if low_mid < high_mid:
            yield low_mid
