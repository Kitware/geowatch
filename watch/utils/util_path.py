import pathlib
import os


def coercepath(path_like):
    """
    Args:
        path_like (str | pathlib.Path | os.PathLike):
            an object representing a filesystem path

    Example:
        >>> from watch.utils.util_path import *  # NOQA
        >>> #
        >>> path_like = '.'
        >>> path = coercepath(path_like)
        >>> print('path = {!r}'.format(path))
        >>> #
        >>> path_like = pathlib.Path('.')
        >>> path = coercepath(path_like)
        >>> print('path = {!r}'.format(path))
        >>> #
        >>> path_like = pathlib.PurePath('.')
        >>> path = coercepath(path_like)
        >>> print('path = {!r}'.format(path))
    """
    if isinstance(path_like, str):
        path = pathlib.Path(path_like)
    elif isinstance(path_like, pathlib.Path):
        path = path_like
    elif isinstance(path_like, os.PathLike):
        path = path_like
    else:
        raise TypeError('Unable to coerce {} to Path'.format(type(path_like)))
    return path


def tree(path):
    """
    Like os.walk but yields a flat list of file and directory paths

    Args:
        path (str | os.PathLike)

    Yields:
        str: path

    Example:
        >>> import itertools as it
        >>> from watch.utils.util_path import *  # NOQA
        >>> import ubelt as ub
        >>> path = pathlib.Path('.')
        >>> gen = tree(path)
        >>> results = list(it.islice(gen, 5))
        >>> print('results = {}'.format(ub.repr2(results, nl=1)))
    """
    import os
    from os.path import join
    for r, fs, ds in os.walk(path):
        for f in fs:
            yield join(r, f)
        for d in ds:
            yield join(r, d)


def coerce_patterned_paths(data, expected_extension=None):
    from os.path import isdir, isfile, join
    import glob
    data = os.fspath(data)
    globpat = None
    if '*' in data:
        globpat = data
    else:
        if isfile(data):
            paths = [data]
        elif isdir(data):
            globpat = join(data, '*' + expected_extension)
    if globpat is not None:
        paths = list(glob.glob(globpat, recursive=True))
    return paths
