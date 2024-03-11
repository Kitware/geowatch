import ubelt as ub
import os
import math
import pandas as pd
import pygtrie
import kwarray
import wrapt
from kwutil import slugify_ext
from geowatch.utils.util_stringalgo import shortest_unique_suffixes


class DataFrame(pd.DataFrame):
    """
    https://stackoverflow.com/questions/22155951/how-can-i-subclass-a-pandas-dataframe

    Example:
        from geowatch.utils.util_pandas import *  # NOQA
        from geowatch.utils import util_pandas
        df = util_pandas.DataFrame.random()
    """
    # _metadata = ['added_property']
    # added_property = 1  # This will be passed to copies

    @property
    def _constructor(self):
        return DataFrame

    @classmethod
    def random(cls, rows=10, columns='abcde', rng=None):
        """
        rows=10
        columns='abcde'
        rng = None
        cls = util_pandas.DataFrame
        """
        import kwarray
        rng = kwarray.ensure_rng(rng)

        def coerce_index(data):
            if isinstance(data, int):
                return list(range(data))
            else:
                return list(data)
        columns = coerce_index(columns)
        index = coerce_index(rows)
        random_data = [{c: rng.rand() for c in columns} for r in index]
        self = cls(random_data, index=index, columns=columns)
        return self

    def safe_drop(self, labels, axis=0):
        """
        Args:
            df (pd.DataFrame): df
            labels (List): ...
            axis (int): todo

        Example:
            >>> from geowatch.utils.util_pandas import *  # NOQA
            >>> import numpy as np
            >>> self = DataFrame({k: np.random.rand(10) for k in 'abcde'})
            >>> self.safe_drop(list('bdf'), axis=1)
        """
        existing = self.axes[axis]
        labels = existing.intersection(labels)
        return self.drop(labels, axis=axis)

    def reorder(self, labels, axis=0, intersect=False):
        """
        Change the order of the row or column index. Unspecified labels will
        keep their existing order after the specified labels.

        Args:
            intersect (bool):
                if True ignores labels that doen't exist, otherwise an error
                will occur if a label is specified that does not exist.

        Example:
            >>> from geowatch.utils import util_pandas
            >>> self = util_pandas.DataFrame.random(columns=['a', 'b', 'c', 'd', 'e', 'f'])
            >>> self.reorder(['b', 'c'], axis=1)
            >>> self.reorder([1, 0], axis=0)
            >>> self.reorder(['q'], axis=1)
            >>> self.reorder(['q'], axis=1, intersect=True)
        """
        existing = self.axes[axis]
        if intersect:
            resolved_labels = ub.oset(labels) & ub.oset(list(existing))
        else:
            resolved_labels = labels
        remain = existing.difference(resolved_labels)
        new_labels = list(resolved_labels) + list(remain)
        return self.reindex(labels=new_labels, axis=axis)

    def _orig_groupby(self, by=None, **kwargs):
        return super().groupby(by=by, **kwargs)

    def groupby(self, by=None, **kwargs):
        """
        Fixed groupby behavior so length-one arguments are handled correctly

        Args:
            df (DataFrame):
            ** kwargs: groupby kwargs

        Example:
            >>> from geowatch.utils import util_pandas
            >>> df = util_pandas.DataFrame({
            >>>     'Animal': ['Falcon', 'Falcon', 'Parrot', 'Parrot'],
            >>>     'Color': ['Blue', 'Blue', 'Blue', 'Yellow'],
            >>>     'Max Speed': [380., 370., 24., 26.]
            >>>     })
            >>> new1 = dict(list(df.groupby(['Animal', 'Color'])))
            >>> new2 = dict(list(df.groupby(['Animal'])))
            >>> new3 = dict(list(df.groupby('Animal')))
            >>> assert sorted(new1.keys())[0] == ('Falcon', 'Blue')
            >>> assert sorted(new3.keys())[0] == 'Falcon'
            >>> # This is the case that is fixed.
            >>> assert sorted(new2.keys())[0] == ('Falcon',)
        """
        groups = super().groupby(by=by, **kwargs)
        fixed_groups = _fix_groupby(groups)
        return fixed_groups

    def match_columns(self, pat, hint='glob'):
        """
        Find matching columns in O(N)
        """
        from kwutil import util_pattern
        pat = util_pattern.Pattern.coerce(pat, hint=hint)
        found = [c for c in self.columns if pat.match(c)]
        return found

    def search_columns(self, pat, hint='glob'):
        """
        Find matching columns in O(N)
        """
        from kwutil import util_pattern
        pat = util_pattern.Pattern.coerce(pat, hint=hint)
        found = [c for c in self.columns if pat.search(c)]
        return found

    def varied_values(self, **kwargs):
        """
        SeeAlso:
            :func:`geowatch.utils.result_analysis.varied_values`
        """
        from geowatch.utils.result_analysis import varied_values
        varied = varied_values(self, **kwargs)
        return varied

    def varied_value_counts(self, **kwargs):
        """
        SeeAlso:
            :func:`geowatch.utils.result_analysis.varied_value_counts`
        """
        from geowatch.utils.result_analysis import varied_value_counts
        varied = varied_value_counts(self, **kwargs)
        return varied


def pandas_reorder_columns(df, columns):
    # Use DataFrame.reorder instead
    remain = df.columns.difference(columns)
    return df.reindex(columns=(columns + list(remain)))


def pandas_argmaxima(data, columns, k=1):
    """
    Finds the top K indexes for given columns.

    Args:
        data : pandas data frame

        columns : columns to maximize.
            If multiple are given, then secondary columns are used as
            tiebreakers.

        k : number of top entries

    Returns:
        List: indexes into subset of data that are in the top k for any of the
            requested columns.

    Example:
        >>> from geowatch.utils.util_pandas import *  # NOQA
        >>> import numpy as np
        >>> import pandas as pd
        >>> data = pd.DataFrame({k: np.random.rand(10) for k in 'abcde'})
        >>> columns = ['b', 'd', 'e']
        >>> k = 1
        >>> top_indexes = pandas_argmaxima(data=data, columns=columns, k=k)
        >>> assert len(top_indexes) == k
        >>> print(data.loc[top_indexes])
    """
    ranked_data = data.sort_values(columns, ascending=False)
    if isinstance(k, float) and math.isinf(k):
        k = None
    top_locs = ranked_data.index[0:k]
    return top_locs


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


def pandas_shorten_columns(summary_table, return_mapping=False, min_length=0):
    """
    Shorten column names

    Example:
        >>> from geowatch.utils.util_pandas import *  # NOQA
        >>> df = pd.DataFrame([
        >>>     {'param_hashid': 'badbeaf', 'metrics.eval.f1': 0.9, 'metrics.eval.mcc': 0.8, 'metrics.eval.acc': 0.3},
        >>>     {'param_hashid': 'decaf', 'metrics.eval.f1': 0.6, 'metrics.eval.mcc': 0.2, 'metrics.eval.acc': 0.4},
        >>>     {'param_hashid': 'feedcode', 'metrics.eval.f1': 0.5, 'metrics.eval.mcc': 0.3, 'metrics.eval.acc': 0.1},
        >>> ])
        >>> print(df.to_string(index=0))
        >>> df2 = pandas_shorten_columns(df)
        param_hashid  metrics.eval.f1  metrics.eval.mcc  metrics.eval.acc
             badbeaf              0.9               0.8               0.3
               decaf              0.6               0.2               0.4
            feedcode              0.5               0.3               0.1
        >>> print(df2.to_string(index=0))
        param_hashid  f1  mcc  acc
             badbeaf 0.9  0.8  0.3
               decaf 0.6  0.2  0.4
            feedcode 0.5  0.3  0.1

    Example:
        >>> from geowatch.utils.util_pandas import *  # NOQA
        >>> df = pd.DataFrame([
        >>>     {'param_hashid': 'badbeaf', 'metrics.eval.f1.mean': 0.9, 'metrics.eval.f1.std': 0.8},
        >>>     {'param_hashid': 'decaf', 'metrics.eval.f1.mean': 0.6, 'metrics.eval.f1.std': 0.2},
        >>>     {'param_hashid': 'feedcode', 'metrics.eval.f1.mean': 0.5, 'metrics.eval.f1.std': 0.3},
        >>> ])
        >>> df2 = pandas_shorten_columns(df, min_length=2)
        >>> print(df2.to_string(index=0))
        param_hashid  f1.mean  f1.std
             badbeaf      0.9     0.8
               decaf      0.6     0.2
            feedcode      0.5     0.3
    """
    import ubelt as ub
    # fixme
    old_cols = summary_table.columns
    new_cols = shortest_unique_suffixes(old_cols, sep='.', min_length=min_length)
    mapping = ub.dzip(old_cols, new_cols)
    summary_table = summary_table.rename(columns=mapping)
    if return_mapping:
        return summary_table, mapping
    else:
        return summary_table


def pandas_condense_paths(colvals):
    """
    Condense a column of paths to keep only the shortest distinguishing
    suffixes

    Args:
        colvals (pd.Series): a column containing paths to condense

    Returns:
        Tuple: the condensed series and a mapping from old to new

    Example:
        >>> from geowatch.utils.util_pandas import *  # NOQA
        >>> rows = [
        >>>     {'path1': '/path/to/a/file1'},
        >>>     {'path1': '/path/to/a/file2'},
        >>> ]
        >>> colvals = pd.DataFrame(rows)['path1']
        >>> pandas_condense_paths(colvals)
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
    from geowatch.utils.util_pandas import pandas_truncate_items

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
        >>> from geowatch.utils.util_pandas import *  # NOQA
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

    # Not sure how safe it is to do this.
    # Consider the case of the dataframe with columns ['a.b.c', 'b.c'].
    # Asking for ['b.c'] would return both, there is no way to just get the
    # specific b.c column because its a suffix of another columns.
    # We would need to disallow the current __getitem__ behavior in order to
    # make a consistent variant of this, and perhaps thats an ok idea. Perhaps
    # we do a df.nest[<index>] localizer similar to df.loc or df.iloc and keep
    # default getitem behavior. That seems cleaner.
    #
    # @property
    # def _constructor(self):
    #     return DotDictDataFrame

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
        from geowatch.utils.util_dotdict import dotkeys_to_nested
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
        # DEPRECATE use match or search columns instead.
        from kwutil import util_pattern
        pat = util_pattern.Pattern.coerce(pat, hint=hint)
        found = [c for c in self.columns if pat.match(c)]
        return found

    def match_columns(self, pat, hint='glob'):
        from kwutil import util_pattern
        pat = util_pattern.Pattern.coerce(pat, hint=hint)
        found = [c for c in self.columns if pat.match(c)]
        return found

    def search_columns(self, pat, hint='glob'):
        from kwutil import util_pattern
        pat = util_pattern.Pattern.coerce(pat, hint=hint)
        found = [c for c in self.columns if pat.search(c)]
        return found

    def subframe(self, key, drop_prefix=True):
        """
        Given a prefix key, return the subet columns that match it with the
        stripped prefix.
        """
        lookup_keys = []
        new_keys = []
        key_parts = key.split('.')
        nlevel = len(key_parts)
        for c in self.columns:
            path = c.split('.')
            if path[0:nlevel] == key_parts:
                lookup_keys.append(c)
                if drop_prefix:
                    new_keys.append('.'.join(path[nlevel:]))
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


def aggregate_columns(df, aggregator=None, fallback='const',
                      nonconst_policy='error'):
    """
    Aggregates parameter columns based on per-column strategies / functions
    specified in ``aggregator``.

    Args:
        hash_cols (None | List[str]):
            columns whos values should be hashed together.

        aggregator (Dict[str, str | callable]):
            a dictionary mapping column names to a callable function that
            should be used to aggregate them. There a special string codes that
            we accept as well.
            Special functions are: hist, hash, min-max, const,

        fallback (str | callable):
            Aggregator function for any column without an explicit aggregator.
            Defaults to "const", which passes one value from the columns
            through if they are constant. If they are not constant, the
            nonconst-policy is triggered.

        nonconst_policy (str):
            Behavior when the aggregator is "const", but the input is
            non-constant. The policies are:
                * 'error' - error if unhandled non-uniform columns exist
                * 'drop' - remove unhandled non-uniform columns

    Returns:
        pd.Series

    TODO:
        - [ ] optimize this

    CommandLine:
        xdoctest -m geowatch.utils.util_pandas aggregate_columns

    Example:
        >>> from geowatch.utils.util_pandas import *  # NOQA
        >>> import numpy as np
        >>> num_rows = 10
        >>> columns = {
        >>>     'nums1': np.random.rand(num_rows),
        >>>     'nums2': np.random.rand(num_rows),
        >>>     'nums3': (np.random.rand(num_rows) * 10).astype(int),
        >>>     'nums4': (np.random.rand(num_rows) * 10).astype(int),
        >>>     'cats1': np.random.randint(0, 3, num_rows),
        >>>     'cats2': np.random.randint(0, 3, num_rows),
        >>>     'cats3': np.random.randint(0, 3, num_rows),
        >>>     'const1': ['a'] * num_rows,
        >>>     'strs1': [np.random.choice(list('abc')) for _ in range(num_rows)],
        >>> }
        >>> df = pd.DataFrame(columns)
        >>> aggregator = ub.udict({
        >>>     'nums1': 'mean',
        >>>     'nums2': 'max',
        >>>     'nums3': 'min-max',
        >>>     'nums4': 'stats',
        >>>     'cats1': 'histogram',
        >>>     'cats3': 'first',
        >>>     'cats2': 'hash12',
        >>>     'strs1': 'hash12',
        >>> })
        >>> #
        >>> # Test that the const fallback works
        >>> row = aggregate_columns(df, aggregator, fallback='const')
        >>> print('row = {}'.format(ub.urepr(row.to_dict(), nl=1)))
        >>> assert row['const1'] == 'a'
        >>> row = aggregate_columns(df.iloc[0:1], aggregator, fallback='const')
        >>> assert row['const1'] == 'a'
        >>> #
        >>> # Test that the drop fallback workds
        >>> row = aggregate_columns(df, aggregator, fallback='drop')
        >>> print('row = {}'.format(ub.urepr(row.to_dict(), nl=1)))
        >>> assert 'const1' not in row
        >>> row = aggregate_columns(df.iloc[0:1], aggregator, fallback='drop')
        >>> assert 'const1' not in row
        >>> #
        >>> # Test that non-constant policy triggers
        >>> aggregator_ = aggregator - {'cats3'}
        >>> import pytest
        >>> with pytest.raises(NonConstantError):
        >>>     row = aggregate_columns(df, aggregator_, nonconst_policy='error')
        >>> row = aggregate_columns(df, aggregator_, nonconst_policy='drop')
        >>> assert 'cats3' not in row
        >>> row = aggregate_columns(df, aggregator_, nonconst_policy='hash')
        >>> assert 'cats3' in row
        >>> #
        >>> # Test an empty dataframe returns an empty series
        >>> row = aggregate_columns(df.iloc[0:0], aggregator)
        >>> assert len(row) == 0
        >>> #
        >>> # Test single column cases work fine.
        >>> for col in df.columns:
        ...     subdf = df[[col]]
        ...     subagg = aggregate_columns(subdf, aggregator, fallback='const')
        ...     assert len(subagg) == 1
        >>> #
        >>> # Test single column drop case works
        >>> subagg = aggregate_columns(df[['cats3']], aggregator_, fallback='const', nonconst_policy='drop')
        >>> assert len(subagg) == 0
        >>> subagg = aggregate_columns(df[['cats3']], aggregator_, fallback='drop')
        >>> assert len(subagg) == 0

    Example:
        >>> from geowatch.utils.util_pandas import *  # NOQA
        >>> import numpy as np
        >>> num_rows = 10
        >>> columns = {
        >>>     'dates': ['2101-01-01', '1970-01-01', '2000-01-01'],
        >>>     'lists': [['a'], ['a', 'b'], []],
        >>>     'nums':  [1, 2, 3],
        >>> }
        >>> df = pd.DataFrame(columns)
        >>> aggregator = ub.udict({
        >>>     'dates': 'min-max',
        >>>     'lists': 'hash',
        >>>     'nums':  'mean',
        >>> })
        >>> row = aggregate_columns(df, aggregator)
        >>> print('row = {}'.format(ub.urepr(row.to_dict(), nl=1)))

    Example:
        >>> from geowatch.utils.util_pandas import *  # NOQA
        >>> import numpy as np
        >>> num_rows = 10
        >>> columns = {
        >>>     'items': [['a'], ['bcd', 'ef'], [], ['3', '234', '2343']],
        >>> }
        >>> df = pd.DataFrame(columns)
        >>> row = aggregate_columns(df, 'last', fallback='const')
        >>> columns = {
        >>>     'items': ['a', 'c', 'c', 'd'],
        >>>     'items2': [['a'], ['bcd', 'ef'], [], ['3', '234', '2343']],
        >>> }
        >>> df = pd.DataFrame(columns)
        >>> row = aggregate_columns(df, 'unique')
    """
    import pandas as pd
    # import numpy as np
    if len(df) == 0:
        return pd.Series(dtype=object)

    if aggregator is None:
        aggregator = {}
    if isinstance(aggregator, str):
        # If given as a string apply the aggreagatr to all columns
        aggregator = {c: aggregator for c in df.columns}

    aggregator = ub.udict(aggregator)

    # Handle columns that can be aggregated
    aggregated = []
    handled_keys = df.columns.intersection(aggregator.keys())
    unhandled_keys = df.columns.difference(handled_keys)

    if len(df) == 1 and fallback == 'const':
        agg_row = df.iloc[0]
        return agg_row
    elif len(df) == 1 and fallback == 'drop':
        agg_row = df.iloc[0][handled_keys]
        return agg_row
    else:
        aggregator = aggregator & handled_keys
        op_to_cols = ub.group_items(aggregator.keys(), aggregator.values())
        if len(unhandled_keys):
            op_to_cols[fallback] = unhandled_keys

        # Handle all columns with the same aggregator in a single call.
        for agg_op, cols in op_to_cols.items():
            toagg = df[cols]
            # toagg = toagg.select_dtypes(include=np.number)
            if isinstance(agg_op, str):
                agg_op_norm = SpecialAggregators.normalize_special_key(agg_op)
                if agg_op_norm == 'drop':
                    continue
                elif agg_op_norm == 'const':
                    # Special case where we will skip aggregation
                    part = _handle_const(toagg, nonconst_policy)
                    aggregated.append(part)
                    continue
                else:
                    agg_op = SpecialAggregators.special_lut.get(agg_op_norm, agg_op)

            # Using apply instead of pandas aggregate because we are allowed to
            # return a list result and have that be a single cell.
            part = toagg.apply(agg_op, result_type='reduce')
            # old: part = toagg.aggregate(agg_op)

            aggregated.append(part)
        if len(aggregated):
            agg_parts = pd.concat(aggregated)
            agg_row = agg_parts
        else:
            agg_row = pd.Series(dtype=object)
    return agg_row


def _handle_const(toagg, nonconst_policy):
    # Check which of the columns are actually constant
    is_const_cols = {
        k: ub.allsame(vs, eq=nan_eq)
        for k, vs in toagg.T.iterrows()}
    nonconst_cols = [k for k, v in is_const_cols.items() if not v]
    if nonconst_cols:
        if nonconst_policy == 'drop':
            toagg = toagg.iloc[0:1].drop(nonconst_cols, axis=1)
        elif nonconst_policy == 'error':
            raise NonConstantError(f'Values are non-constant in columns: {nonconst_cols}')
        elif nonconst_policy == 'hash':
            nonconst_data = toagg[nonconst_cols]
            const_part = toagg.iloc[0:1].drop(nonconst_cols, axis=1).iloc[0]
            nonconst_part = nonconst_data.apply(SpecialAggregators.hash, result_type='reduce')
            part = pd.concat([const_part, nonconst_part])
            return part
        else:
            raise KeyError(nonconst_policy)
    part = toagg.iloc[0]
    return part


class SpecialAggregators:

    def hash(x):
        return ub.hash_data(x.values.tolist())

    def hash12(x):
        return ub.hash_data(x.values.tolist())[0:12]

    def unique(x):
        try:
            return list(ub.unique(x))
        except Exception:
            return list(ub.unique(x, key=ub.hash_data))

    def min_max(x):
        return (x.min(), x.max())
        # ret = {
        #     'min': x.min(),
        #     'max': x.max(),
        # }
        # return ret

    @staticmethod
    def normalize_special_key(k):
        return k.replace('-', '_')

    special_lut = {
        'hash': hash,
        'hash12': hash12,
        'min_max': min_max,
        'stats': kwarray.stats_dict,
        'hist': ub.dict_hist,
        'unique': unique,
        'histogram': ub.dict_hist,
        'first': lambda x: x.iloc[0],
        'last': lambda x: x.iloc[-1],
    }


class NonConstantError(ValueError):
    ...


def nan_eq(a, b):
    if isinstance(a, float) and isinstance(b, float) and math.isnan(a) and math.isnan(b):
        return True
    else:
        return a == b


# Fix pandas groupby so it uses the new behavior with a list of len 1

class GroupbyFutureWrapper(wrapt.ObjectProxy):
    """
    Wraps a groupby object to get the new behavior sooner.
    """

    def __iter__(self):
        keys = self.keys
        if isinstance(keys, list) and len(keys) == 1:
            # Handle this special case to avoid a warning
            for key, group in self.grouper.get_iterator(self._selected_obj, axis=self.axis):
                yield (key,), group
        else:
            # Otherwise use the parent impl
            yield from self.__wrapped__.__iter__()


def _fix_groupby(groups):
    keys = groups.keys
    if isinstance(keys, list) and len(keys) == 1:
        return GroupbyFutureWrapper(groups)
    else:
        return groups


def pandas_fixed_groupby(df, by=None, **kwargs):
    """
    Fixed groupby behavior so length-one arguments are handled correctly

    Args:
        df (DataFrame):
        ** kwargs: groupby kwargs

    Example:
        >>> from geowatch.utils.util_pandas import *  # NOQA
        >>> df = pd.DataFrame({
        >>>     'Animal': ['Falcon', 'Falcon', 'Parrot', 'Parrot'],
        >>>     'Color': ['Blue', 'Blue', 'Blue', 'Yellow'],
        >>>     'Max Speed': [380., 370., 24., 26.]
        >>>     })
        >>> # Old behavior
        >>> old1 = dict(list(df.groupby(['Animal', 'Color'])))
        >>> old2 = dict(list(df.groupby(['Animal'])))
        >>> old3 = dict(list(df.groupby('Animal')))
        >>> new1 = dict(list(pandas_fixed_groupby(df, ['Animal', 'Color'])))
        >>> new2 = dict(list(pandas_fixed_groupby(df, ['Animal'])))
        >>> new3 = dict(list(pandas_fixed_groupby(df, 'Animal')))
        >>> assert sorted(new1.keys())[0] == ('Falcon', 'Blue')
        >>> assert sorted(old1.keys())[0] == ('Falcon', 'Blue')
        >>> assert sorted(new3.keys())[0] == 'Falcon'
        >>> assert sorted(old3.keys())[0] == 'Falcon'
        >>> # This is the case that is fixed.
        >>> assert sorted(new2.keys())[0] == ('Falcon',)
        >>> assert sorted(old2.keys())[0] == 'Falcon'
    """
    groups = df.groupby(by=by, **kwargs)
    fixed_groups = _fix_groupby(groups)
    return fixed_groups
