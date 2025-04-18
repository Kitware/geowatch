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

    Note:
        A farthest first sequence is not unique in general. The particular
        order this function returns may change in the future.

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

    Example:
        >>> # Choose a custom starting location
        >>> from geowatch.utils.util_kwutil import *  # NOQA
        >>> import ubelt as ub
        >>> items = list(range(10))
        >>> results = []
        >>> start, stop = 0, 10
        >>> for first in range(start, stop):
        >>>     results += [list(farthest_first(items, first=first))]
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
            [9, 0, 5, 2, 7, 1, 6, 3, 8, 4],
        ]
    """
    sub_index_gen = _farthest_first_indices(0, len(items), first=first)
    if max_num is not None:
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


def _farthest_first_indices(start, stop, first=None):
    """
    Given a ordered list of items, incrementally yield indexes such that each
    new index maximizes the distance to the closest previously chosen index.

    Args:
        start (int): The inclusive starting index (typically 0)
        stop (int): The exclusive maximum index (typically ``len(items)``)
        first (int | None): The index to start from.
            If unspecified the largest index is first.

    Yields:
        int: the next chosen index in the series

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

    Example:
        >>> from geowatch.utils.util_kwutil import *  # NOQA
        >>> from geowatch.utils.util_kwutil import _farthest_first_indices
        >>> from geowatch.utils.util_kwutil import _check_farthest_first_index_sequence
        >>> import ubelt as ub
        >>> start, stop = 0, 10
        >>> gen = _farthest_first_indices(start, stop)
        >>> results = []
        >>> for first in range(start, stop):
        >>>     result = list(_farthest_first_indices(start, stop, first=first))
        >>>     assert _check_farthest_first_index_sequence(result)
        >>>     results += [result]
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
            [9, 0, 5, 2, 7, 1, 6, 3, 8, 4],
        ]
        >>> result = list(gen)
        >>> assert set(result) == set(range(start, stop))
        >>> print(result)

        print(list(_farthest_first_indices(0, 32)))

        start = 0
        for stop in range(100):
            for first in range(start, stop):
                result = list(_farthest_first_indices(start, stop, first=first))
                assert _check_farthest_first_index_sequence(result)
    """
    import ubelt as ub
    if first is None or first == stop - 1:
        # Standard case
        if start < stop:
            yield stop - 1
            yield from _farthest_from_previous_helper(start, stop - 1)
    else:
        yield from _farthest_first_indices_generalized(start, stop, first)

    # Broken
    if 0:
        # Case where first is specified
        yield first
        # In this case, the left and right are not always balanced, so we generate
        # from whatever one is further until they balance. We have to include
        # the "first" element in both left and right generators to ensure
        # are generating elments far away from it, so we have to filter them out.
        left_gen = _farthest_first_indices(start, first + 1)
        right_gen = _farthest_first_indices(first, stop)

        DEBUG = 1
        if DEBUG:
            from itertools import tee
            right_gen, right_gen1 = tee(right_gen)
            left_gen, left_gen1 = tee(left_gen)
            print(f'start = {ub.urepr(start, nl=1)}')
            print(f'stop={stop}')
            print(f'first = {ub.urepr(first, nl=1)}')
            print('setup left: ' + str(list(left_gen1)))
            print('setup right: ' + str(list(right_gen1)))

        # the first item in the left gen will be "first", which we already yeilded
        # pop it off and ignore it.
        next(left_gen)

        # As long as first isn't the same as start or stop, then next will not raise StopIteration here.
        next_left = next(left_gen, None)
        next_right = next(right_gen, None)

        """
        Observation:
            the "best" distance halfs every time we take an item past the first.

            def best_dist(sequence):
                for idx, item in enumerate(sequence):
                    if idx == 0:
                        yield len(sequence)
                    else:
                        yield min([abs(p - item) for p in sequence[:idx]])
            sequence = list(_farthest_first_indices(0, 32))
            distance = list(best_dist(sequence))
            print(ub.urepr(ub.dzip(sequence, distance), align=':'))
            print(ub.urepr(ub.dict_hist(distance)))

            If the previous distance id d:
                then the next distance can only be on of:
                    d
                    d - 1
                    (d // 2)
                    (d // 2) + 1

            sequence: [31, 0, 16, 8, 24, 4, 20, 12, 28, 2, 18, 10, 26, 6, 22, 14, 30, 1, 17, 9, 25, 5, 21, 13, 29, 3, 19, 11, 27, 7, 23, 15]
            bestdist:  ∞  31, 15, 8, 7, 4, 4, 4, 3, 2, 2, 2, 2, 2, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]

            The 0th selection is basically free and doesn't have a value.
            the 1st will always have a distance of length of the sequence.
            the 2nd selection will roughly half the distance
            the 3rd will roughly half the distance again
            the 4th will be within 1 of the halfed distance
            the 5th will roughly half the distance again and there will be 4 of this magnitude
            the 9th will be the next halving and have 8 items at the magnitude
            The 17th item will be the next halving
            The 33th will be the next half

            so the pattern is the i-th item's distance is within at most this
            number of halfings:

                from numpy import log2, ceil

                def num_halfings(i):
                    if i == 0:
                        return 0
                    else:
                        return ceil(log2(i))

                def max_distance_at_step(i, total):
                    if i == 0:
                        return total
                    return total / (2 ** ceil(log2(i)))

                for total in range(2048):
                    bound2 = np.array([max_distance_at_step(i, total) for i in range(total)], dtype=int)
                    halvings = np.array([num_halfings(i) for i in range(total)], dtype=int)
                    bound = np.floor(total / 2 ** halvings).astype(int)
                    sequence = np.array(list(_farthest_first_indices(0, total)))
                    distance = np.array(list(best_dist(sequence)))
                    print(sequence)
                    print(distance)
                    print(bound)
                    np.array(distance <= bound)
                    tightness = distance - bound
                    assert np.all(np.abs(tightness) <= 1)

               so if we have a first element,
               a left sequence an an optimal right sequence, we know the
               size of the left and right sequence.

               without loss of generality assume the right sequence is larger.

               the right size has size R
               and the left side has size L
               in the case R > L

               in either sequence, the distance will roughly half with the sequence:

                   0, 1, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 4, 4, 5, ...  # https://oeis.org/A029837


               say
               start=0
               stop=81
               first=27

               result = list(_farthest_first_indices(start, stop, first=first))

               so

               R = 53
               L = 27

               The max distance in each case will be...

               [int(max_distance_at_step(i, R)) for i in range(1, 4)]
               [int(max_distance_at_step(i, L)) for i in range(1, 4)]


        """

        if next_left is None:
            if next_right is not None:
                # FIXME; something wrong here in the case where first = stop
                yield next_right
                # Ignore the next item in next right, it "first"
                next(right_gen, None)
                yield from right_gen
            return
        elif next_right is None:
            yield next_left
            yield from left_gen
            return

        left_dist = abs(next_left - first)
        right_dist = abs(next_right - first)

        left_size = first - start
        right_size = stop - first - 1
        print(f'left_size={left_size}')
        print(f'right_size={right_size}')

        # The second item in the right gen will be "first", so we pop it off
        # to ignore it.
        next(right_gen, None)

        DEBUG = 1
        if DEBUG:
            from itertools import tee
            right_gen, right_gen1 = tee(right_gen)
            left_gen, left_gen1 = tee(left_gen)
            print(f'start left: {next_left}, ' + str(list(left_gen1)))
            print(f'start right: {next_right}, ' + str(list(right_gen1)))
            #### FFF this doesn't work..

        while True:
            if left_dist == right_dist:
                # At this point the distance to first heuristic breaks in the
                # case where start=0, stop=10, first=2. I think it might be the
                # case that when the heuristic breaks we can just interleave
                # the remaining iterables, but I'm not sure if this holds.
                total = (stop - start)
                if DEBUG:
                    print('At a point where left and right dist are the same')
                    right_gen, right_gen1 = tee(right_gen)
                    left_gen, left_gen1 = tee(left_gen)
                    print(f'curr left: {next_left}, ' + str(list(left_gen1)))
                    print(f'curr right: {next_right}, ' + str(list(right_gen1)))
                    print(f'total = {ub.urepr(total, nl=1)}')

                # if total % 2 == 0:
                pairgen = it.zip_longest(left_gen, right_gen)
                flatgen = it.chain([next_left], [next_right], *pairgen)
                filtgen = filter(lambda x: x is not None, flatgen)
                yield from filtgen
                # else:
                #     pairgen = it.zip_longest(right_gen, left_gen)
                #     flatgen = it.chain([next_right], [next_left], *pairgen)
                #     filtgen = filter(lambda x: x is not None, flatgen)
                #     yield from filtgen
                return
            if left_dist > right_dist:
                if DEBUG:
                    print(f'chose left {next_left=} {next_right=} {left_dist=} {right_dist=}')
                yield next_left
                next_left = next(left_gen, None)
                if next_left is None:
                    # No more data on the left side, finish the right side
                    yield next_right
                    yield from right_gen
                    break
                left_dist = abs(next_left - first)
            else:
                if DEBUG:
                    print(f'chose right {next_left=} {next_right=} {left_dist=} {right_dist=}')
                yield next_right
                next_right = next(right_gen, None)
                if next_right is None:
                    # No more data on the right side, finish the left side
                    yield next_left
                    yield from left_gen
                    break
                right_dist = abs(next_right - first)


def _check_farthest_first_index_sequence(sequence):
    """
    Checks that the sequence of indices satisfies the farthest-first objective.

    For each item in the sequence, it should be the further from any seen value
    than any unseen value.

    Example:
        >>> # There are multiple ways a sequence can be valie
        >>> assert _check_farthest_first_index_sequence([3, 0, 2, 1])
        >>> assert _check_farthest_first_index_sequence([0, 3, 2, 1])
        >>> assert not _check_farthest_first_index_sequence([0, 2, 3, 1])
    """
    for idx, index in enumerate(sequence):
        seen = sequence[:idx]
        unseen = sequence[idx + 1:]
        # Look at the distances from the chosen item to all seen items
        this_dist_to_seens = {prev: abs(index - prev) for prev in seen}
        # Look at the distances from the unchosen item to all seen items
        other_dist_to_seens = {
            other: {prev: abs(other - prev) for prev in seen}
            for other in unseen
        }
        import ubelt as ub
        if seen and unseen:
            # The minimum distance from this to the seen items should be greater than
            this_min_dist_to_seen_value = min(this_dist_to_seens.values())
            # The minimum distance from any unseen item to the seen items
            other_min_dist_to_seen_value = max([min(o.values()) for o in other_dist_to_seens.values()])
            next_best = ub.argmax({k: min(o.values()) for k, o in other_dist_to_seens.items()})
            if this_min_dist_to_seen_value < other_min_dist_to_seen_value:
                print('----')
                print(f'index={index}')
                print(f'seen = {ub.urepr(seen, nl=0)}')
                print(f'unseen = {ub.urepr(unseen, nl=0)}')
                print(f'this_dist_to_seens = {ub.urepr(this_dist_to_seens, nl=1)}')
                print(f'other_dist_to_seens = {ub.urepr(other_dist_to_seens, nl=1)}')
                print(f'this_min_dist_to_seen_value={this_min_dist_to_seen_value}')
                print(f'other_min_dist_to_seen_value={other_min_dist_to_seen_value}')
                print(f'We chose the index {index}, but we could have chosen {next_best}')
                print(f'sequence = {ub.urepr(sequence, nl=0)}')
                return False

    return True


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
