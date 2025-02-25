"""
Functions that may be moved to kwutil
"""
import itertools as it


# Backwards compatible variant used in geowatch util
def distributed_subitems(items, max_num=None):
    """
    Return a subset of items maximally distributed over the input index space.
    I.e. the chosen indexes maximize the space between them.

    Args:
        items (List | Dict): an ordered indexable object

    Returns:
        List | Dict: a subset of the input with a length at most ``max_num``.

    Note:
        Prefer using the generator variant :func:`farthest_first` instead.

    TODO:
        - [X] Find a better name. ChatGPT suggests using "spread", which I
              like. Maybe spreadsort, spreadshuffle, spredtraverse?
              spreadtake, takespread?
        - [ ] Figure out where this lives, probably kwutil.
        - [X] maybe we should force this to be a generator? Or make a generator variant?


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
    sub_index_gen = _farthest_first_indices(0, len(items))
    sub_indices = sorted(it.islice(sub_index_gen, max_num))
    if isinstance(items, dict):
        item_keys = list(items.keys())
        sub_keys = [item_keys[idx] for idx in sub_indices]
        sub_items = {key: items[key] for key in sub_keys}
    else:
        sub_items = [items[idx] for idx in sub_indices]
    return sub_items


def farthest_first(items, max_num=None, first=None):
    """
    Return a subset of items maximally distributed over the input index space.
    I.e. the chosen indexes maximize the space between them.

    Args:
        items (List | Dict): an ordered indexable object
        first (int | None): if specified, start with this index.

    Returns:
        List | Dict: a subset of the input with a length at most ``max_num``.

    Example:
        >>> from geowatch.utils.util_kwutil import *  # NOQA
        >>> items = list(range(100))
        >>> max_num = 5
        >>> sub_items = list(farthest_first(items, max_num))
        >>> print(list(sub_items))
        [99, 0, 50, 25, 75]

    Example:
        >>> from geowatch.utils.util_kwutil import *  # NOQA
        >>> items = {chr(i): i for i in range(ord('a'), ord('a') + 26)}
        >>> max_num = 5
        >>> sub_items = list(farthest_first(items, max_num))
        >>> print(dict(sub_items))
        {'z': 122, 'a': 97, 'n': 110, 'g': 103, 't': 116}
    """
    if first is None:
        sub_index_gen = _farthest_first_indices(0, len(items))
    else:
        sub_index_gen = _farthest_first_indices_generalized(0, len(items), start=first)
    sub_index_gen = it.islice(sub_index_gen, max_num)
    if isinstance(items, dict):
        item_keys = list(items.keys())
        for idx in sub_index_gen:
            key = item_keys[idx]
            value = items[key]
            yield (key, value)
    else:
        for idx in sub_index_gen:
            yield items[idx]


def _farthest_first_indices(start, stop):
    """
    Given a ordered list of items, incrementally yield indexes such that each
    new index maximizes the distance to all other previously chosen indexes.

    Args:
        start (int): The inclusive starting index (typically 0)
        stop (int): The exclusive maximum index (typically ``len(items)``)

    Yields:
        int: the next chosen index in the series

    TODO:
        - [ ] Find a Better Name,
                spread_indices?
                farthest_index_traversal
                spread_index_traversal

        - [ ] Allow the user to specify which point is first?
              Do we start at the beginning, end, or middle?
              Currently we default to the end.

    Notes:
        * This is an instance of farthest-point traversal in 1D.

    References:
        .. [CSSE_167943] https://cs.stackexchange.com/questions/167943/is-this-knapsack-variant-named-studied-online-algorithm-for-farthest-from-pr
        .. [WikiFarthestFirst] https://en.wikipedia.org/wiki/Farthest-first_traversal

    CommandLine:
        xdoctest -m geowatch.utils.util_kwutil _farthest_first_indices

    Example:
        >>> from geowatch.utils.util_kwutil import *  # NOQA
        >>> total = 10
        >>> start, stop = 0, 10
        >>> gen = _farthest_first_indices(start, stop)
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


def _farthest_first_indices_generalized(start, stop, first=None, distance_fn=None):
    """
    Yield indices in a farthest-first traversal order for a 1D uniform grid [start, stop),
    with an optional user-specified starting index.

    This implementation uses a heap to maintain for each candidate the minimum distance
    to any selected index, so that we can always pick the candidate that maximizes that distance.

    Args:
        start (int): inclusive start index.
        stop (int): exclusive stop index.
        first (int, optional): the starting index (must be in [start, stop)). Defaults to stop - 1.
        distance_fn (callable, optional): A function that computes distance between two indices.
            Defaults to absolute difference (L1 norm).

    Yields:
        int: the next index in the farthest-first order.

    References:
        https://chatgpt.com/c/67a37e86-f89c-8013-8ef8-400c209de0e0

    TODO:
        - [ ] Allow the user to specify a custom distance metric. The logic
              should still work because we only care about the maximum distance
              of unselected elements to their closest selected index.

    Example:
        >>> from geowatch.utils.util_kwutil import *  # NOQA
        >>> import ubelt as ub
        >>> total = 10
        >>> start, stop = 0, 10
        >>> results = []
        >>> for first in range(start, stop):
        >>>     results += [list(_farthest_first_indices_generalized(start, stop, first=first))]
        >>> print(f'results = {ub.urepr(results, nl=1)}')
        results = [
            [0, 9, 4, 2, 6, 1, 3, 5, 7, 8],
            [1, 9, 5, 3, 7, 0, 2, 4, 6, 8],
            [2, 9, 5, 0, 7, 1, 3, 4, 6, 8],
            [3, 9, 0, 6, 1, 2, 4, 5, 7, 8],
            [4, 9, 0, 2, 6, 1, 3, 5, 7, 8],
            [5, 0, 9, 2, 7, 1, 3, 4, 6, 8],
            [6, 0, 3, 9, 1, 2, 4, 5, 7, 8],
            [7, 0, 3, 5, 9, 1, 2, 4, 6, 8],
            [8, 0, 4, 2, 6, 1, 3, 5, 7, 9],
            [9, 0, 4, 2, 6, 1, 3, 5, 7, 8],
        ]
        >>> # This distance function just reverses the metric, so we find the
        >>> # closest item instead of the furthest. It is a bit contrived, but
        >>> # demos how a specific distance function can modify the results.
        >>> results = []
        >>> def distance_fn(i, j):
        >>>     return -abs(i - j)
        >>> for first in range(start, stop):
        >>>     results += [list(_farthest_first_indices_generalized(start, stop, first=first, distance_fn=distance_fn))]
        >>> print(f'results = {ub.urepr(results, nl=1)}')
        results = [
            [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
            [1, 0, 2, 3, 4, 5, 6, 7, 8, 9],
            [2, 1, 0, 3, 4, 5, 6, 7, 8, 9],
            [3, 2, 1, 0, 4, 5, 6, 7, 8, 9],
            [4, 3, 2, 1, 0, 5, 6, 7, 8, 9],
            [5, 4, 3, 2, 1, 0, 6, 7, 8, 9],
            [6, 5, 4, 3, 2, 1, 0, 7, 8, 9],
            [7, 6, 5, 4, 3, 2, 1, 0, 8, 9],
            [8, 7, 6, 5, 4, 3, 2, 1, 0, 9],
            [9, 8, 7, 6, 5, 4, 3, 2, 1, 0],
        ]
    """
    # Note: using sortedcontainers might have a more clear implementation this
    # is probably still O(n**2), but I think we must be able to do better.  not
    # going to think about it too hard, this code path is not very likely to be
    # needed, as the simple farther first implementation is what is mainly
    # needed.
    import heapq
    if first is None:
        first = stop - 1
    if not (start <= first < stop):
        raise ValueError("first must be in the range [start, stop)")

    if distance_fn is None:
        distance_fn = lambda a, b: abs(a - b)  # NOQA

    # Set of selected indices (kept sorted for convenience).
    selected = [first]
    yield first

    # For each candidate (not yet selected), store its best known distance to a selected index.
    best_dist = {
        i: distance_fn(i, first)
        for i in range(start, stop)
        if i != first
    }
    # Maintain a heap keyed by negative distance (since heapq is a min-heap).
    heap = []
    for i, d in best_dist.items():
        heapq.heappush(heap, (-d, i))

    # Greedily select the candidate with the maximum current distance.
    while heap:
        neg_d, candidate = heapq.heappop(heap)
        # If candidate was already selected, skip it.
        if candidate not in best_dist:
            continue
        # If the min distance in the heap is wrong, then the item is outdated,
        # and we need to skip it.
        current_d = best_dist[candidate]
        if -neg_d != current_d:
            # The stored distance is outdated, pop it and skip, the correct
            # value will exist in the queue later on.
            continue

        # Candidate is the farthest from any selected point.
        yield candidate
        selected.append(candidate)
        # Remove candidate from best_dist so it isn’t chosen again.
        best_dist.pop(candidate)

        # Update best distances for all remaining candidates.
        for other, old_d in best_dist.items():
            new_d = distance_fn(other, candidate)
            if new_d < old_d:
                best_dist[other] = new_d
                heapq.heappush(heap, (-new_d, other))
