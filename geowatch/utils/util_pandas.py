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
    Extension of pandas dataframes with quality-of-life improvements.

    Refernces:
        .. [SO22155951] https://stackoverflow.com/questions/22155951/how-can-i-subclass-a-pandas-dataframe

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
        Create a random data frame for testing.

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

    @classmethod
    def coerce(cls, data):
        """
        Ensures that the input is an instance of our extended DataFrame.

        Pandas is generally good about input coercion via its normal
        constructors, the purpose of this classmethod is to quickly ensure that
        a DataFrame has all of the extended methods defined by this class
        without incurring a copy. In this sense it is more similar to
        :func:numpy.asarray`.

        Args:
            data (DataFrame | ndarray | Iterable | dict):
                generally another dataframe, otherwise normal inputs that would
                be given to the regular pandas dataframe constructor

        Returns:
            DataFrame:

        Example:
            >>> # xdoctest: +REQUIRES(--benchmark)
            >>> # This example demonstrates the speed difference between
            >>> # recasting as a DataFrame versus using coerce
            >>> from geowatch.utils.util_pandas import DataFrame
            >>> data = DataFrame.random(rows=10_000)
            >>> import timerit
            >>> ti = timerit.Timerit(100, bestof=10, verbose=2)
            >>> for timer in ti.reset('constructor'):
            >>>     with timer:
            >>>         DataFrame(data)
            >>> for timer in ti.reset('coerce'):
            >>>     with timer:
            >>>         DataFrame.coerce(data)
            >>> # xdoctest: +IGNORE_WANT
            Timed constructor for: 100 loops, best of 10
                time per loop: best=2.594 µs, mean=2.783 ± 0.1 µs
            Timed coerce for: 100 loops, best of 10
                time per loop: best=246.000 ns, mean=283.000 ± 32.4 ns
        """
        if isinstance(data, cls):
            return data
        else:
            return cls(data)

    def safe_drop(self, labels, axis=0):
        """
        Like :func:`self.drop`, but does not error if the specified labels do
        not exist.

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

    def reorder(self, head=None, tail=None, axis=0, missing='error',
                fill_value=float('nan'), **kwargs):
        """
        Change the order of the row or column index. Unspecified labels will
        keep their existing order after the specified labels.

        Args:
            head (List | None):
                The order of the labels to put at the start of the re-indexed
                data frame. Unspecified labels keep their relative order and
                are placed after specified these "head" labels.

            tail (List | None):
                The order of the labels to put at the end of the re-indexed
                data frame. Unspecified labels keep their relative order and
                are placed after before these "tail" labels.

            axis (int):
                The axis 0 for rows, 1 for columns to reorder.

            missing (str):
                Policy to handle specified labels that do not exist in the
                specified axies. Can be either "error", "drop", or "fill".
                If "drop", then drop any specified labels that do not exist.
                If "error", then raise an error non-existing labels are given.
                If "fill", then fill in values for labels that do not exist.

            fill_value (Any):
                fill value to use when missing is "fill".

        Returns:
            Self - DataFrame with modified indexes

        Example:
            >>> from geowatch.utils import util_pandas
            >>> self = util_pandas.DataFrame.random(rows=5, columns=['a', 'b', 'c', 'd', 'e', 'f'])
            >>> new = self.reorder(['b', 'c'], axis=1)
            >>> assert list(new.columns) == ['b', 'c', 'a', 'd', 'e', 'f']
            >>> # Set the order of the first and last of the columns
            >>> new = self.reorder(head=['b', 'c'], tail=['e', 'd'], axis=1)
            >>> assert list(new.columns) == ['b', 'c', 'a', 'f', 'e', 'd']
            >>> # Test reordering the rows
            >>> new = self.reorder([1, 0], axis=0)
            >>> assert list(new.index) == [1, 0, 2, 3, 4]
            >>> # Test reordering with a non-existent column
            >>> new = self.reorder(['q'], axis=1, missing='drop')
            >>> assert list(new.columns) == ['a', 'b', 'c', 'd', 'e', 'f']
            >>> new = self.reorder(['q'], axis=1, missing='fill')
            >>> assert list(new.columns) == ['q', 'a', 'b', 'c', 'd', 'e', 'f']
            >>> import pytest
            >>> with pytest.raises(ValueError):
            >>>     self.reorder(['q'], axis=1, missing='error')
            >>> # Should error if column is given in both head and tail
            >>> with pytest.raises(ValueError):
            >>>     self.reorder(['c'], ['c'], axis=1, missing='error')
        """
        if 'intersect' in kwargs:
            raise Exception('The intersect argument was removed. Set missing=drop')
        if kwargs:
            raise ValueError(f'got unknown kwargs: {list(kwargs.keys())}')

        existing = self.axes[axis]
        if head is None:
            head = []
        if tail is None:
            tail = []
        head_set = set(head)
        tail_set = set(tail)
        duplicate_labels = head_set & tail_set
        if duplicate_labels:
            raise ValueError(
                'Cannot specify the same label in both the head and tail.'
                f'Duplicate labels: {duplicate_labels}')
        if missing == 'drop':
            orig_order = ub.oset(list(existing))
            resolved_head = ub.oset(head) & orig_order
            resolved_tail = ub.oset(tail) & orig_order
        elif missing == 'error':
            requested = (head_set | tail_set)
            unknown = requested - set(existing)
            if unknown:
                raise ValueError(
                    f"Requested labels that don't exist unknown={unknown}. "
                    "Specify intersect=True to ignore them.")
            resolved_head = head
            resolved_tail = tail
        elif missing == 'fill':
            resolved_head = head
            resolved_tail = tail
        else:
            raise KeyError(missing)
        remain = existing.difference(resolved_head).difference(resolved_tail)
        new_labels = list(resolved_head) + list(remain) + list(resolved_tail)
        return self.reindex(labels=new_labels, axis=axis,
                            fill_value=fill_value)

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
        Kwargs:
            min_variations=0, max_variations=None, default=ub.NoParam,
            dropna=False, on_error='raise'

        SeeAlso:
            :func:`geowatch.utils.result_analysis.varied_values`
        """
        from geowatch.utils.result_analysis import varied_values
        varied = varied_values(self, **kwargs)
        return varied

    def varied_value_counts(self, **kwargs):
        """
        Kwargs:
            min_variations=0, max_variations=None, default=ub.NoParam,
            dropna=False, on_error='raise'

        SeeAlso:
            :func:`geowatch.utils.result_analysis.varied_value_counts`
        """
        from geowatch.utils.result_analysis import varied_value_counts
        varied = varied_value_counts(self, **kwargs)
        return varied

    def shorten_columns(self, return_mapping=False, min_length=0):
        """
        Shorten column names by separating unique suffixes based on the "."
        separator.

        Args:
            return_mapping (bool):
                if True, returns the

            min_length (int):
                minimum size of the new column names in terms of parts.

        Returns:
            DataFrame | Tuple[DataFrame, Dict[str, str]]:
                Either the new data frame with shortened column names or that
                data frame and the mapping from old column names to new column
                names.

        Example:
            >>> from geowatch.utils.util_pandas import DataFrame
            >>> # If all suffixes are unique, then they are used.
            >>> self = DataFrame.random(columns=['id', 'params.metrics.f1', 'params.metrics.acc', 'params.fit.model.lr', 'params.fit.data.seed'])
            >>> new = self.shorten_columns()
            >>> assert list(new.columns) == ['id', 'f1', 'acc', 'lr', 'seed']
            >>> # Conflicting suffixes impose limitations on what can be shortened
            >>> self = DataFrame.random(columns=['id', 'params.metrics.magic', 'params.metrics.acc', 'params.fit.model.lr', 'params.fit.data.magic'])
            >>> new = self.shorten_columns()
            >>> assert list(new.columns) == ['id', 'metrics.magic', 'metrics.acc', 'model.lr', 'data.magic']
        """
        import ubelt as ub
        old_cols = self.columns
        new_cols = shortest_unique_suffixes(old_cols, sep='.', min_length=min_length)
        mapping = ub.dzip(old_cols, new_cols)
        new = self.rename(columns=mapping)
        if return_mapping:
            return new, mapping
        else:
            return new

    def _to_dotdict(self):
        """
        Experimental, convert a a dotdict (should maybe give useful dotdict
        methods to this class?)
        """
        return DotDictDataFrame(self)

    def argextrema(self, columns, objective='maximize', k=1):
        """
        Finds the top K indexes (locs) for given columns.

        Args:
            columns (str | List[str]) : columns to find extrema of.
                If multiple are given, then secondary columns are used as
                tiebreakers.

            objective (str | List[str]) :
                Either maximize or minimize (max and min are also accepted).
                If given as a list, it specifies the criteria for each column,
                which allows for a mix of maximization and minimization.

            k : number of top entries

        Returns:
            List: indexes into subset of data that are in the top k for any of the
                requested columns.

        Example:
            >>> from geowatch.utils.util_pandas import DataFrame
            >>> # If all suffixes are unique, then they are used.
            >>> self = DataFrame.random(columns=['id', 'f1', 'loss'], rows=10)
            >>> self.loc[3, 'f1'] = 1.0
            >>> self.loc[4, 'f1'] = 1.0
            >>> self.loc[5, 'f1'] = 1.0
            >>> self.loc[3, 'loss'] = 0.2
            >>> self.loc[4, 'loss'] = 0.3
            >>> self.loc[5, 'loss'] = 0.1
            >>> columns = ['f1', 'loss']
            >>> k = 4
            >>> top_indexes = self.argextrema(columns=columns, k=k, objective=['max', 'min'])
            >>> assert len(top_indexes) == k
            >>> print(self.loc[top_indexes])
        """
        ascending = None
        def rectify_ascending(objective_str):
            if objective_str in {'max', 'maximize'}:
                ascending = False
            elif objective_str in {'min', 'minimize'}:
                ascending = True
            else:
                raise KeyError(objective)
            return ascending

        if isinstance(objective, str):
            ascending = rectify_ascending(objective)
        else:
            ascending = [rectify_ascending(o) for o in objective]

        ranked_data = self.sort_values(columns, ascending=ascending)
        if isinstance(k, float) and math.isinf(k):
            k = None
        top_locs = ranked_data.index[0:k]
        return top_locs


def pandas_reorder_columns(df, columns):
    """
    DEPRECATED: Use :func:`DataFrame.reorder` instead
    """
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

    DEPRECATED: Use :func:`DataFrame.shorten_columns` instead.

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

    def _column_graph(self):
        import networkx as nx
        graph = nx.DiGraph()
        # root = '__root__'
        # graph.add_node(root)
        for c in self.columns:
            parts = c.split('.')
            # prev_node = root
            prev_node = None
            for i in range(1, len(parts)):
                node = '.'.join(parts[:i + 1])
                graph.add_node(node, label=f'{parts[i]}')
                if prev_node is not None:
                    graph.add_edge(prev_node, node)
                prev_node = node
        nx.write_network_text(graph, with_labels=1)

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
        >>> import numpy as np
        >>> if np.lib.NumpyVersion(pd.__version__) < '2.0.0':
        >>>     assert sorted(old2.keys())[0] == 'Falcon'
    """
    groups = df.groupby(by=by, **kwargs)
    fixed_groups = _fix_groupby(groups)
    return fixed_groups
