"""
pip install pygtrie
"""
import numpy as np  # NOQA


def shortest_unique_prefixes(items, sep=None, allow_simple=True, min_length=0, allow_end=False):
    r"""
    The shortest unique prefix algorithm.

    Args:
        items (list of str): returned prefixes will be unique wrt this set
        sep (str): if specified, all characters between separators are treated
            as a single symbol. Makes the algo much faster.
        allow_simple (bool): if True tries to construct a simple feasible
            solution before resorting to the optimal trie algorithm.
        min_length (int): minimum length each prefix can be
        allow_end (bool): if True allows for string terminators to be
            considered in the prefix

    Returns:
        list of str: a prefix for each item that uniquely identifies it
           wrt to the original items.

    References:
        http://www.geeksforgeeks.org/find-all-shortest-unique-prefixes-to-represent-each-word-in-a-given-list/
        https://github.com/Briaares/InterviewBit/blob/master/Level6/Shortest%20Unique%20Prefix.cpp

    Requires:
        pip install pygtrie

    Example:
        >>> # xdoctest: +REQUIRES(--pygtrie)
        >>> items = ["zebra", "dog", "duck", "dove"]
        >>> shortest_unique_prefixes(items)
        ['z', 'dog', 'du', 'dov']

    Timeing:
        >>> # DISABLE_DOCTEST
        >>> # make numbers larger to stress test
        >>> # L = max length of a string, N = number of strings,
        >>> # C = smallest gaurenteed common length
        >>> # (the setting N=10000, L=100, C=20 is feasible we are good)
        >>> import ubelt as ub
        >>> import random
        >>> def make_data(N, L, C):
        >>>     rng = random.Random(0)
        >>>     return [''.join(['a' if i < C else chr(rng.randint(97, 122))
        >>>                      for i in range(L)]) for _ in range(N)]
        >>> items = make_data(N=1000, L=10, C=0)
        >>> ub.Timerit(3).call(shortest_unique_prefixes, items).print()
        Timed for: 3 loops, best of 3
            time per loop: best=24.54 ms, mean=24.54 ± 0.0 ms
        >>> items = make_data(N=1000, L=100, C=0)
        >>> ub.Timerit(3).call(shortest_unique_prefixes, items).print()
        Timed for: 3 loops, best of 3
            time per loop: best=155.4 ms, mean=155.4 ± 0.0 ms
        >>> items = make_data(N=1000, L=100, C=70)
        >>> ub.Timerit(3).call(shortest_unique_prefixes, items).print()
        Timed for: 3 loops, best of 3
            time per loop: best=232.8 ms, mean=232.8 ± 0.0 ms
        >>> items = make_data(N=10000, L=250, C=20)
        >>> ub.Timerit(3).call(shortest_unique_prefixes, items).print()
        Timed for: 3 loops, best of 3
            time per loop: best=4.063 s, mean=4.063 ± 0.0 s
    """
    import pygtrie
    if len(set(items)) != len(items):
        raise ValueError('inputs must be unique')

    # construct trie
    if sep is None:
        trie = pygtrie.CharTrie.fromkeys(items, value=0)
    else:
        # In some simple cases we can avoid constructing a trie
        if allow_simple:
            tokens = [item.split(sep) for item in items]
            simple_solution = [t[0] for t in tokens]
            if len(simple_solution) == len(set(simple_solution)):
                return simple_solution
            for i in range(2, 10):
                # print('return simple solution at i = {!r}'.format(i))
                simple_solution = ['-'.join(t[:i]) for t in tokens]
                if len(simple_solution) == len(set(simple_solution)):
                    return simple_solution

        trie = pygtrie.StringTrie.fromkeys(items, value=0, separator=sep)

    # Set the value (frequency) of all nodes to zero.
    for node in _trie_iternodes(trie):
        node.value = 0

    # For each item trace its path and increment frequencies
    for item in items:
        final_node, trace = trie._get_node(item)
        for key, node in trace:
            node.value += 1

    # if not isinstance(node.value, int):
    #     node.value = 0

    # Query for the first prefix with frequency 1 for each item.
    # This is the shortest unique prefix over all items.
    unique = []
    for item in items:
        freq = None
        for prefix, freq in trie.prefixes(item):
            if freq == 1 and len(prefix) >= min_length:
                break
        if not allow_end:
            assert freq == 1, 'item={} has no unique prefix. freq={}'.format(item, freq)
        # print('items = {!r}'.format(items))
        unique.append(prefix)
    return unique


def _trie_iternodes(self):
    """
    Generates all nodes in the trie

    # Hack into the internal structure and insert frequencies at each node
    """
    from collections import deque
    stack = deque([[self._root]])
    while stack:
        for node in stack.pop():
            yield node
            try:
                # only works in pygtrie-2.2 broken in pygtrie-2.3.2
                stack.append(node.children.values())
            except AttributeError:
                stack.append([v for k, v in node.children.iteritems()])


def shortest_unique_suffixes(items, sep=None):
    r"""
    Example:
        >>> # xdoctest: +REQUIRES(--pygtrie)
        >>> items = ["zebra", "dog", "duck", "dove"]
        >>> shortest_unique_suffixes(items)
        ['a', 'g', 'k', 'e']

    Example:
        >>> # xdoctest: +REQUIRES(--pygtrie)
        >>> items = ["aa/bb/cc", "aa/bb/bc", "aa/bb/dc", "aa/cc/cc"]
        >>> shortest_unique_suffixes(items)
    """
    snoitpo = [p[::-1] for p in items]
    sexiffus = shortest_unique_prefixes(snoitpo, sep=sep)
    suffixes = [s[::-1] for s in sexiffus]
    return suffixes
