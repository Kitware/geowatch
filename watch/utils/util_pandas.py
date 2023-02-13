import ubelt as ub
import os
import pandas as pd
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
