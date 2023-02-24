import ubelt as ub
import os
import pandas as pd
import pygtrie
from watch.utils import slugify_ext
from watch.utils.util_stringalgo import shortest_unique_suffixes


def pandas_argmaxima(data, columns, k=1):
    """
    Finds the top K indexes for each column.

    Args:
        data : pandas data frame
        columns : columns to maximize
        k : number of top entries per column

    Returns:
        List: indexes into subset of data that are in the top k for any of the
            requested columns.

    Example:
        >>> from watch.mlops.aggregate import *  # NOQA
        >>> import numpy as np
        >>> data = pd.DataFrame({k: np.random.rand(10) for k in 'abcde'})
        >>> columns = ['b', 'd', 'e']
        >>> k = 1
        >>> top_indexes = pandas_argmaxima(data=data, columns=columns, k=k)
        >>> print(data.loc[top_indexes])
    """
    top_indexes = None
    for col in columns:
        ranked_data = data[col].sort_values(ascending=False)
        ranked_idxs = ranked_data.index
        if top_indexes is None:
            top_indexes = ranked_idxs[0:k]
        else:
            top_indexes = top_indexes.union(ranked_idxs[0:k], sort=False)
    return top_indexes


def pandas_suffix_columns(data, suffixes):
    """
    Return columns that end with this suffix
    """
    return [c for c in data.columns if any(c.endswith(s) for s in suffixes)]


def pandas_nan_eq(a, b):
    nan_flags1 = pd.isna(a)
    nan_flags2 = pd.isna(b)
    eq_flags = a == b
    both_nan = nan_flags1 & nan_flags2
    flags = eq_flags | both_nan
    return flags


def pandas_shorten_columns(summary_table, return_mapping=False):
    """
    Shorten column names
    """
    import ubelt as ub
    # fixme
    old_cols = summary_table.columns
    new_cols = shortest_unique_suffixes(old_cols, sep='.')
    mapping = ub.dzip(old_cols, new_cols)
    summary_table = summary_table.rename(columns=mapping)
    if return_mapping:
        return summary_table, mapping
    else:
        return summary_table


def pandas_condense_paths(colvals):
    """
    Condense a column of paths
    """
    is_valid = ~pd.isnull(colvals)
    valid_vals = colvals[is_valid].apply(os.fspath)
    unique_valid_vals = valid_vals.unique().tolist()
    unique_short_vals = shortest_unique_suffixes(unique_valid_vals, sep='/')
    new_vals = [p.split('.')[0] for p in unique_short_vals]
    mapper = ub.dzip(unique_valid_vals, new_vals)
    condensed = colvals.apply(lambda x: mapper.get(x, x))
    return condensed, mapper


def pandas_truncate_items(data, paths=False, max_length=16):
    """
    from watch.utils.util_pandas import pandas_truncate_items

    Args:
        data (pd.DataFrame): data frame to truncate

    Returns:
        Tuple[pd.DataFrame, Dict[str, str]]
    """
    def truncate(x):
        if not isinstance(x, (str, os.PathLike)):
            return x
        return slugify_ext.smart_truncate(str(x), max_length=max_length, trunc_loc=0,
                                          hash_len=4, head='', tail='')
    mappings = {}
    if len(data):
        # only check the first row to see if we want to truncate the columns or
        # not
        trunc_str_cols = set()
        trunc_path_cols = set()
        for _, check_row in data.iloc[0:10].iterrows():
            for k, v in check_row.items():
                if paths:
                    # Check if probably a path or not
                    if isinstance(v, os.PathLike):
                        trunc_path_cols.add(k)
                        continue
                    elif isinstance(v, str) and '/' in v:
                        trunc_path_cols.add(k)
                        continue
                if isinstance(v, (str, os.PathLike)) and len(str(v)) > max_length:
                    trunc_str_cols.add(k)

        trunc_str_cols = list(ub.oset(data.columns) & trunc_str_cols)
        trunc_path_cols = list(ub.oset(data.columns) & trunc_path_cols)

        trunc_data = data.copy()
        trunc_data[trunc_str_cols] = trunc_data[trunc_str_cols].applymap(truncate)
        for c in trunc_data[trunc_str_cols]:
            v2 = pd.Categorical(data[c])
            data[c] = v2
            v1 = v2.map(truncate)
            mapping = list(zip(v1.categories, v2.categories))
            mappings[c] = mapping

        for c in trunc_path_cols:
            colvals = trunc_data[c]
            condensed, mapping = pandas_condense_paths(colvals)
            trunc_data[c] = condensed
            mappings[c] = mapping
    else:
        mapping = {}
        trunc_data = data
    return trunc_data, mappings


class DotDictDataFrame(pd.DataFrame):
    """
    A proof-of-concept wrapper around pandas that lets us walk down the nested
    structure a little easier.

    The API is a bit weird, and the caches are not invalidated if any column
    changes, but it does a reasonable job otherwise.

    Is there another library out there that does this?

    SeeAlso:
        DotDict

    Example:
        >>> from watch.utils.util_pandas import *  # NOQA
        >>> rows = [
        >>>     {'node1.id': 1, 'node2.id': 2, 'node1.metrics.ap': 0.5, 'node2.metrics.ap': 0.8},
        >>>     {'node1.id': 1, 'node2.id': 2, 'node1.metrics.ap': 0.5, 'node2.metrics.ap': 0.8},
        >>>     {'node1.id': 1, 'node2.id': 2, 'node1.metrics.ap': 0.5, 'node2.metrics.ap': 0.8},
        >>>     {'node1.id': 1, 'node2.id': 2, 'node1.metrics.ap': 0.5, 'node2.metrics.ap': 0.8},
        >>> ]
        >>> self = DotDictDataFrame(rows)
        >>> # Test prefix lookup
        >>> assert set(self['node1'].columns) == {'node1.id', 'node1.metrics.ap'}
        >>> # Test suffix lookup
        >>> assert set(self['id'].columns) == {'node1.id', 'node2.id'}
        >>> # Test mid-node lookup
        >>> assert set(self['metrics'].columns) == {'node1.metrics.ap', 'node2.metrics.ap'}
        >>> # Test single lookup
        >>> assert set(self[['node1.id']].columns) == {'node1.id'}
        >>> # Test glob
        >>> assert set(self.find_columns('*metri*')) == {'node1.metrics.ap', 'node2.metrics.ap'}
    """

    def __init__(self, *args, **kw):
        super().__init__(*args, **kw)
        self.__dict__['_trie_cache'] = {}

    def _clear_column_caches(self):
        self._trie_cache = {}

    @property
    def _column_prefix_trie(self):
        # TODO: cache the trie correctly
        if self._trie_cache.get('prefix_trie', None) is None:
            _trie_data = ub.dzip(self.columns, self.columns)
            _trie = pygtrie.StringTrie(_trie_data, separator='.')
            self._trie_cache['prefix_trie'] = _trie
        return self._trie_cache['prefix_trie']

    @property
    def _column_suffix_trie(self):
        if self._trie_cache.get('suffix_trie', None) is None:
            reversed_columns = ['.'.join(col.split('.')[::-1])
                                for col in self.columns]
            _trie_data = ub.dzip(reversed_columns, reversed_columns)
            _trie = pygtrie.StringTrie(_trie_data, separator='.')
            self._trie_cache['suffix_trie'] = _trie
        return self._trie_cache['suffix_trie']

    @property
    def _column_node_groups(self):
        if self._trie_cache.get('node_groups', None) is None:
            paths = [col.split('.') for col in self.columns]
            lut = ub.ddict(list)
            for path in paths:
                col = '.'.join(path)
                for part in path:
                    lut[part].append(col)
            self._trie_cache['node_groups'] = lut
        return self._trie_cache['node_groups']

    @property
    def nested_columns(self):
        from watch.utils.util_dotdict import dotkeys_to_nested
        return dotkeys_to_nested(self.columns)

    def find_column(self, col):
        result = self.query_column(col)
        if len(result) == 0:
            raise KeyError
        elif len(result) > 1:
            raise RuntimeError
        return list(result)[0]

    def query_column(self, col):
        # Might be better to do a globby sort of pattern
        parts = col.split('.')
        return ub.oset.intersection(*[self._column_node_groups[p] for p in parts])
        # try:
        #     candiates.update(self._column_prefix_trie.values(col))
        # except KeyError:
        #     ...
        # try:
        #     candiates.update(self._column_suffix_trie.values(col))
        # except KeyError:
        #     ...
        # return candiates

    def lookup_suffix_columns(self, col):
        return self._column_suffix_trie.values(col)

    def lookup_prefix_columns(self, col):
        return self._column_prefix_trie.values(col)

    def find_columns(self, pat, hint='glob'):
        from watch.utils import util_pattern
        pat = util_pattern.Pattern.coerce(pat, hint=hint)
        found = [c for c in self.columns if pat.match(c)]
        return found

    def subframe(self, key, drop_prefix=True):
        """
        Given a prefix key, return the subet columns that match it with the
        stripped prefix.
        """
        lookup_keys = []
        new_keys = []
        for c in self.columns:
            path = c.split('.')
            if path[0] == key:
                lookup_keys.append(c)
                new_keys.append('.'.join(path[1:]))
        new = self.loc[:, lookup_keys]
        if drop_prefix:
            new.rename(ub.dzip(lookup_keys, new_keys), inplace=True, axis=1)
        return new

    def __getitem__(self, cols):
        if isinstance(cols, str):
            if cols not in self.columns:
                cols = self.query_column(cols)
                if not cols:
                    print(f'Available columns={self.columns}')
                    raise KeyError
        elif isinstance(cols, list):
            cols = list(ub.flatten([self.query_column(c) for c in cols]))
        return super().__getitem__(cols)


def pandas_add_prefix(data, prefix):
    return data.add_prefix(prefix)
    # mapper = {c: prefix + c for c in data.columns}
    # return data.rename(mapper, axis=1)
