"""
Utilities for dictionaries where dots in keys represent nestings
"""
import ubelt as ub
import pygtrie


class DotDict(ub.UDict):
    """
    I'm sure this data structure exists on pypi.
    This should be replaced with that if we find it.

    SeeAlso:
        DotDictDataFrame

    Example:
        >>> from geowatch.utils.util_dotdict import *  # NOQA
        >>> self = DotDict({
        >>>     'proc1.param1': 1,
        >>>     'proc1.param2': 2,
        >>>     'proc2.param1': 3,
        >>>     'proc2.param2': 4,
        >>>     'proc3.param1': 5,
        >>>     'proc3.param2': 6,
        >>>     'proc4.part1.param1': 7,
        >>>     'proc4.part1.param2': 8,
        >>>     'proc4.part2.param2': 9,
        >>>     'proc4.part2.param2': 10,
        >>> })
        >>> self.get('proc1')
        >>> self.prefix_get('proc4')
        >>> 'proc1' in self

        >>> nested = self.to_nested()
        >>> recon = DotDict.from_nested(nested)
        >>> assert nested != self
        >>> assert recon == self
    """

    def __init__(self, /, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Tries work well with prefix stuff, but they may be too complex for
        # what we really need to do here.
        self._trie_cache = {}

    @classmethod
    def from_nested(cls, data):
        """
        Args:
            data (Dict):
                nested data
        """
        flat = cls()
        walker = ub.IndexableWalker(data, list_cls=tuple())
        for path, value in walker:
            if not isinstance(value, dict):
                spath = list(map(str, path))
                key = '.'.join(spath)
                flat[key] = value
        return flat

    def to_nested(self):
        """
        Converts this flat DotDict into a nested representation.  I.e. keys are
        broken using the "." separtor, with each separator becoming a new
        nesting level.

        Example:
            >>> from geowatch.utils.util_dotdict import *  # NOQA
            >>> self = DotDict(**{
            >>>     'foo.bar.baz': 1,
            >>>     'foo.bar.biz': 1,
            >>>     'foo.spam': 1,
            >>>     'eggs.spam': 1,
            >>> })
            >>> nested = self.to_nested()
            >>> print(f'nested = {ub.urepr(nested, nl=2)}')
            nested = {
                'foo': {
                    'bar': {'baz': 1, 'biz': 1},
                    'spam': 1,
                },
                'eggs': {
                    'spam': 1,
                },
            }
        """
        auto = ub.AutoDict()
        walker = ub.IndexableWalker(auto)
        d = self
        for k, v in d.items():
            path = k.split('.')
            walker[path] = v
        return auto.to_dict()

    def to_nested_keys(self):
        """
        Converts this flat DotDict into a nested key representation.
        The difference between this and to_nested is that the leafs are
        sets of keys whereas the leafs in DotDict are dicts

        Example:
            >>> from geowatch.utils.util_dotdict import *  # NOQA
            >>> self = DotDict(**{
            >>>     'foo.bar.baz': 1,
            >>>     'foo.bar.biz': 1,
            >>>     'foo.spam': 1,
            >>>     'eggs.spam': 1,
            >>> })
            >>> nested = self.to_nested_keys()
            >>> print(f'nested = {ub.urepr(nested, nl=2)}')
            nested = {
                'foo': {
                    'bar': {'baz': 'foo.bar.baz', 'biz': 'foo.bar.biz'},
                    'spam': 'foo.spam',
                },
                'eggs': {
                    'spam': 'eggs.spam',
                },
            }
        """
        auto = ub.AutoDict()
        walker = ub.IndexableWalker(auto)
        for k in self:
            path = k.split('.')
            walker[path] = k
        # print(ub.urepr(auto))
        return auto.to_dict()

    @property
    def _prefix_trie(self):
        if self._trie_cache.get('prefix_trie', None) is None:
            _trie_data = ub.dzip(self.keys(), self.keys())
            _trie = pygtrie.StringTrie(_trie_data, separator='.')
            self._trie_cache['prefix_trie'] = _trie
        return self._trie_cache['prefix_trie']

    @property
    def _suffix_trie(self):
        if 'suffix_trie' not in self._trie_cache:
            reversed_keys = {
                '.'.join(reversed(k.split('.'))): k
                for k in self.keys()
            }
            _trie = pygtrie.StringTrie(reversed_keys, separator='.')
            self._trie_cache['suffix_trie'] = _trie
        return self._trie_cache['suffix_trie']

    def suffix_get(self, suffix, default=ub.NoParam, backend='trie'):
        """
        Retrieve all key-value pairs whose keys end with a given dot-suffix.

        Args:
            suffix (str): dot-separated suffix string
            default: fallback if no matches found
            backend (str): 'trie' or 'loop'

        Returns:
            DotDict

        Example:
            >>> from geowatch.utils.util_dotdict import *  # NOQA
            >>> self = DotDict({
            >>>     'a.b.c': 1,
            >>>     'x.b.c': 2,
            >>>     'z.y': 3,
            >>> })
            >>> self.suffix_get('b.c')
            {'a.b.c': 1, 'x.b.c': 2}
        """
        if backend == 'loop':
            matches = DotDict({
                k: v for k, v in self.items()
                if k.endswith('.' + suffix) or k == suffix
            })
        elif backend == 'trie':
            rev_suffix = '.'.join(reversed(suffix.split('.')))
            try:
                matches = DotDict({
                    k: self[k] for k in self._suffix_trie.values(rev_suffix)
                })
            except KeyError:
                if default is not ub.NoParam:
                    return default
                raise
        else:
            raise ValueError(f'Unknown backend={backend}')

        if not matches and default is not ub.NoParam:
            return default
        return matches

    def prefix_get(self, key, default=ub.NoParam):
        """
        Example:
            >>> from geowatch.utils.util_dotdict import *  # NOQA
            >>> self = DotDict(**{
            >>>     'foo.bar.baz': 1,
            >>>     'foo.bar.biz': 1,
            >>>     'foo.spam': 1,
            >>>     'eggs.spam': 1,
            >>> })
            >>> self.prefix_get('foo')
            {'bar.baz': 1, 'bar.biz': 1, 'spam': 1}
        """
        try:
            suffix_dict = DotDict()
            full_keys = self._prefix_trie.values(key)
        except KeyError:
            if default is not ub.NoParam:
                return default
            else:
                raise
        else:
            for full_key in full_keys:
                sub_key = full_key[len(key) + 1:]
                suffix_dict[sub_key] = self[full_key]
            return suffix_dict

    def suffix_subdict(self, suffixes, backend='trie'):
        """
        Filter DotDict to only contain keys ending with any given suffixes.

        Args:
            suffixes (List[str]): list of dot-suffixes
            backend (str): 'trie' or 'loop'

        Returns:
            DotDict

        References:
            https://chatgpt.com/c/6841a161-2cd4-8002-ad9b-5593f5a2d70c

        Example:
            >>> from geowatch.utils.util_dotdict import *  # NOQA
            >>> self = DotDict({
            >>>     'proc1.param1': 1,
            >>>     'proc2.param1': 2,
            >>>     'proc3.param2': 3,
            >>>     'proc4.part1.param1': 4,
            >>>     'proc4.part2.param2': 5,
            >>> })
            >>> new = self.suffix_subdict(['param1', 'part2.param2'])
            >>> print(f'new = {ub.urepr(new, nl=1, sort=1)}')
            new = {
                'proc1.param1': 1,
                'proc2.param1': 2,
                'proc4.part1.param1': 4,
                'proc4.part2.param2': 5,
            }
        """
        if backend == 'loop':
            result = {
                k: v for k, v in self.items()
                if any(k.endswith('.' + suf) or k == suf for suf in suffixes)
            }
        elif backend == 'trie':
            reversed_trie = self._suffix_trie
            result_keys = set()
            for suf in suffixes:
                rev_suf = '.'.join(reversed(suf.split('.')))
                result_keys.update(reversed_trie.values(rev_suf))
            result = {k: self[k] for k in result_keys}
        else:
            raise ValueError(f'Unknown backend={backend}')
        return self.__class__(result)

    def prefix_subdict(self, prefixes, backend='trie'):
        """
        Filter DotDict to only contain keys starting with any given prefixes.

        Args:
            prefixes (List[str]): list of dot-prefixes
            backend (str): 'trie' or 'loop'

        Returns:
            DotDict

        Example:
            >>> from geowatch.utils.util_dotdict import *  # NOQA
            >>> self = DotDict({
            >>>     'proc1.param1': 1,
            >>>     'proc1.param2': 2,
            >>>     'proc2.param1': 3,
            >>>     'proc3.param2': 4,
            >>>     'proc4.part1.param1': 5,
            >>>     'proc4.part2.param2': 6,
            >>> })
            >>> new = self.prefix_subdict(['proc1', 'proc4.part1'])
            >>> print(f'new = {ub.urepr(new, nl=1, sort=1)}')
            new = {
                'proc1.param1': 1,
                'proc1.param2': 2,
                'proc4.part1.param1': 5,
            }
        """
        if backend == 'loop':
            result = {
                k: v for k, v in self.items()
                if any(k.startswith(pref + '.') or k == pref for pref in prefixes)
            }
        elif backend == 'trie':
            trie = self._prefix_trie
            result_keys = set()
            for pref in prefixes:
                try:
                    result_keys.update(trie.values(pref))
                except KeyError:
                    pass  # It's okay if a prefix has no matches
            result = {k: self[k] for k in result_keys}
        else:
            raise ValueError(f'Unknown backend={backend}')
        return self.__class__(result)

    def add_prefix(self, prefix):
        """
        Adds a prefix to all items
        """
        new = self.__class__([(prefix + '.' + k, v) for k, v in self.items()])
        return new

    def insert_prefix(self, prefix, index):
        """
        Adds a prefix to all items

        Args:
            prefix (str): prefix to insert
            index (int): the depth to insert the new param

        Example:
            >>> from geowatch.utils.util_dotdict import *  # NOQA
            >>> self = DotDict({
            >>>     'proc1.param1': 1,
            >>>     'proc1.param2': 2,
            >>>     'proc2.param1': 3,
            >>>     'proc4.part1.param2': 8,
            >>>     'proc4.part2.param2': 9,
            >>>     'proc4.part2.param2': 10,
            >>> })
            >>> new = self.insert_prefix('foo', index=1)
            >>> print('self = {}'.format(ub.urepr(self, nl=1)))
            >>> print('new = {}'.format(ub.urepr(new, nl=1)))
        """
        def _generate_new_items():
            sep = '.'
            for k, v in self.items():
                path = k.split(sep)
                path.insert(index, prefix)
                k2 = sep.join(path)
                yield k2, v
        new = self.__class__(_generate_new_items())
        return new

    def query_keys(self, col):
        """
        Finds columns where one level has this key

        Example:
            >>> from geowatch.utils.util_dotdict import *  # NOQA
            >>> self = DotDict({
            >>>     'proc1.param1': 1,
            >>>     'proc1.param2': 2,
            >>>     'proc2.param1': 3,
            >>>     'proc4.part1.param2': 8,
            >>>     'proc4.part2.param2': 9,
            >>>     'proc4.part2.param2': 10,
            >>> })
            >>> list(self.query_keys('param1'))

        Ignore:
            could use _trie_iteritems
            trie = self._prefix_trie
        """
        for key in self.keys():
            if col in set(key.split('.')):
                yield key

    def print_graph(self):
        explore_nested_dict(self)

    # def __contains__(self, key):
    #     if super().__contains__(key):
    #         return True
    #     else:
    #         subkeys = []
    #         subkeys.extend(self._prefix_trie.values(key))
    #         return bool(subkeys)

    # def get(self, key, default=ub.NoParam):
    #     if default is ub.NoParam:
    #         return self[key]
    #     else:
    #         try:
    #             return self[key]
    #         except KeyError:
    #             return default

    # def __getitem__(self, key):
    #     try:
    #         return super().__getitem__(key)
    #     except KeyError:
    #         subkeys = []
    #         subkeys.extend(self._prefix_trie.values(key))
    #         return self.__class__([(k, self[k]) for k in subkeys])


def dotdict_to_nested(d):
    return DotDict.dotdict_to_nested(d)


def dotkeys_to_nested(keys):
    """
    Args:
        keys (List[str]): a list of dotted key names
    """
    return DotDict.to_nested_keys(keys)


def indexable_to_graph(data):
    import networkx as nx
    graph = nx.DiGraph()
    walker = ub.IndexableWalker(data)
    for path, value in walker:
        spath = list(map(str, path))
        key = '.'.join(spath)
        graph.add_node(key)
        label = spath[-1]
        if not isinstance(value, walker.indexable_cls):
            label = f'{label} : {type(value).__name__} = {value}'

        graph.nodes[key].update({
            'path': path,
            'value': value,
            'label': label,
        })
        if len(path) > 1:
            parent_key = '.'.join(spath[:-1])
            graph.add_edge(parent_key, key)
    return graph


def explore_nested_dict(data):
    """
    TODO: some sort of textual interface
    """
    graph = indexable_to_graph(data)

    from cmd_queue.util.util_networkx import write_network_text
    import rich
    write_network_text(graph, path=rich.print, end='')
