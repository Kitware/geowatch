"""
Utilities for dictionaries where dots in keys represent nestings
"""
import ubelt as ub
import pygtrie


def dotdict_to_nested(d):
    auto = ub.AutoDict()
    walker = ub.IndexableWalker(auto)
    for k, v in d.items():
        path = k.split('.')
        walker[path] = v
    return auto.to_dict()


def dotkeys_to_nested(keys):
    """
    Args:
        keys (List[str]): a list of dotted key names
    """
    auto = ub.AutoDict()
    walker = ub.IndexableWalker(auto)
    for k in keys:
        path = k.split('.')
        walker[path] = k
    # print(ub.urepr(auto))
    return auto.to_dict()


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
        return dotdict_to_nested(self)

    def to_nested_keys(self):
        return dotkeys_to_nested(self)

    @property
    def _prefix_trie(self):
        if self._trie_cache.get('prefix_trie', None) is None:
            _trie_data = ub.dzip(self.keys(), self.keys())
            _trie = pygtrie.StringTrie(_trie_data, separator='.')
            self._trie_cache['prefix_trie'] = _trie
        return self._trie_cache['prefix_trie']

    # @property
    # def _suffix_trie(self):
    #     if self._trie_cache.get('suffix_trie', None) is None:
    #         _trie_data = ub.dzip(self.keys(), self.keys())
    #         _trie = pygtrie.StringTrie(_trie_data, separator='.')
    #         self._trie_cache['suffix_trie'] = _trie
    #     return self._trie_cache['suffix_trie']

    def prefix_get(self, key, default=ub.NoParam):
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

    def print_graph(self):
        explore_nested_dict(self)

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
