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
    # print(ub.repr2(auto))
    return auto.to_dict()


class DotDict(dict):
    """
    I'm sure this data structure exists on pypi.
    This should be replaced with that if we find it.

    Example:
        >>> from watch.utils.util_dotdict import *  # NOQA
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
    """

    def __init__(self, /, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._trie_cache = {}

    @property
    def _prefix_trie(self):
        if self._trie_cache.get('prefix_trie', None) is None:
            _trie_data = ub.dzip(self.keys(), self.keys())
            _trie = pygtrie.StringTrie(_trie_data, separator='.')
            self._trie_cache['prefix_trie'] = _trie
        return self._trie_cache['prefix_trie']

    @property
    def _suffix_trie(self):
        if self._trie_cache.get('prefix_trie', None) is None:
            _trie_data = ub.dzip(self.keys(), self.keys())
            _trie = pygtrie.StringTrie(_trie_data, separator='.')
            self._trie_cache['prefix_trie'] = _trie
        return self._trie_cache['prefix_trie']

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
