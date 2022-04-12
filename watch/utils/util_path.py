import os
import ubelt as ub
import pathlib
from watch.utils import util_pattern


def coercepath(path_like):
    """
    Args:
        path_like (str | ub.Path | os.PathLike):
            an object representing a filesystem path

    Example:
        >>> from watch.utils.util_path import *  # NOQA
        >>> #
        >>> path_like = '.'
        >>> path = coercepath(path_like)
        >>> print('path = {!r}'.format(path))
        >>> #
        >>> path_like = ub.Path('.')
        >>> path = coercepath(path_like)
        >>> print('path = {!r}'.format(path))
        >>> #
        >>> path_like = pathlib.PurePath('.')
        >>> path = coercepath(path_like)
        >>> print('path = {!r}'.format(path))
    """
    if isinstance(path_like, str):
        path = ub.Path(path_like)
    elif isinstance(path_like, pathlib.PurePath):
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
        >>> path = ub.Path('.')
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
    """
    Coerce input to a list of paths.

    Args:
        data (str | List[str]):
            a glob pattern or list of glob patterns

    Returns:
        List[ubelt.Path]: Multiple paths that match the query

    Example:
        >>> # xdoctest: +SKIP
        >>> import watch
        >>> dvc_dpath = watch.find_smart_dvc_dpath()
        >>> data = [dvc_dpath / 'annotations/region_models/*.geojson']
        >>> from watch.utils.util_path import *  # NOQA
        >>> print(coerce_patterned_paths(dvc_dpath / 'annotations/region_models'))
        >>> print(coerce_patterned_paths([dvc_dpath / 'annotations/region_models/*.geojson']))
        >>> print(coerce_patterned_paths('./'))
    """
    from os.path import isdir, join
    import glob
    datas = data if ub.iterable(data) else [data]
    datas = list(map(os.fspath, datas))
    paths = []
    for data_ in datas:
        if expected_extension is not None and isdir(data_):
            globpat = join(data, '*' + expected_extension)
        else:
            globpat = data_
        paths.extend(list(glob.glob(globpat, recursive=True)))
    paths = [ub.Path(p) for p in paths]
    return paths


def find(pattern=None, dpath=None, include=None, exclude=None, type=None,
         recursive=True, followlinks=False):
    """
    Find all paths in a root subject to a search criterion

    Args:
        pattern (str):
            The glob pattern the path name must match to be returned

        dpath (str):
            The root direcotry to search. Default to cwd.

        include (str | List[str]):
            Pattern or list of patterns. If specified, search only files whose
            base name matches this pattern. By default the pattern is GLOB.

        exclude (str | List[str]):
            Pattern or list of patterns. Skip any file with a name suffix that
            matches the pattern. By default the pattern is GLOB.

        type (str | List[str]):
            A list of 1 character codes indicating what types of file can be
            returned. Currently we only allow either "f" for file or "d" for
            directory. Symbolic links are not currently distinguished. In the
            future we may support posix codes, see [1]_ for details.

        recursive:
            search all subdirectories recursively

        followlinks (bool, default=False):
            if True will follow directory symlinks

    References:
        _[1] https://linuxconfig.org/identifying-file-types-in-linux

    TODO:
        mindepth

        maxdepth

        ignore_case

        regex_match


    Example:
        >>> from watch.utils.util_path import *  # NOQA
        >>> paths = list(find(pattern='*'))
        >>> paths = list(find(pattern='*', type='f'))
        >>> print('paths = {!r}'.format(paths))
        >>> print('paths = {!r}'.format(paths))
    """
    from os.path import join

    if pattern is None:
        pattern = '*'

    if type is None:
        with_dirs = True
        with_files = True
    else:
        with_dirs = False
        with_files = False
        if 'd' in type:
            with_dirs = True
        if 'f' in type:
            with_files = True

    if dpath is None:
        dpath = os.getcwd()

    include_ = (None if include is None else
                util_pattern.MultiPattern(include, hint='glob'))
    exclude_ = (None if exclude is None else
                util_pattern.MultiPattern(exclude, hint='glob'))

    main_pattern = util_pattern.Pattern.coerce(pattern, hint='glob')

    def is_included(name):
        if not main_pattern.match(name):
            return False

        if exclude_ is not None:
            if exclude_.match(name):
                return False

        if include_ is not None:
            if include_.match(name):
                return True
            else:
                return False
        return True

    for root, dnames, fnames in os.walk(dpath, followlinks=followlinks):

        if with_files:
            for fname in fnames:
                if is_included(fname):
                    yield join(root, fname)

        if with_dirs:
            for dname in dnames:
                if is_included(dname):
                    yield join(root, dname)

        if not recursive:
            break


def resolve_relative_to(path, dpath, strict=False):
    """
    Given a path, try to resolve its symlinks such that it is relative to the
    given dpath.

    Ignore:
        def _symlink(self, target, verbose=0):
            return ub.Path(ub.symlink(target, self, verbose=verbose))
        ub.Path._symlink = _symlink

        # TODO: try to enumerate all basic cases

        base = ub.Path.appdir('kwcoco/tests/reroot')
        base.delete().ensuredir()

        drive1 = (base / 'drive1').ensuredir()
        drive2 = (base / 'drive2').ensuredir()

        data_repo1 = (drive1 / 'data_repo1').ensuredir()
        cache = (data_repo1 / '.cache').ensuredir()
        real_file1 = (cache / 'real_file1').touch()

        real_bundle = (data_repo1 / 'real_bundle').ensuredir()
        real_assets = (real_bundle / 'assets').ensuredir()

        # Symlink file outside of the bundle
        link_file1 = (real_assets / 'link_file1')._symlink(real_file1)
        real_file2 = (real_assets / 'real_file2').touch()
        link_file2 = (real_assets / 'link_file2')._symlink(real_file2)


        # A symlink to the data repo
        data_repo2 = (drive1 / 'data_repo2')._symlink(data_repo1)
        data_repo3 = (drive2 / 'data_repo3')._symlink(data_repo1)
        data_repo4 = (drive2 / 'data_repo4')._symlink(data_repo2)

        # A prediction repo TODO
        pred_repo5 = (drive2 / 'pred_repo5').ensuredir()

        _ = ub.cmd(f'tree -a {base}', verbose=3)

        fpaths = []
        for r, ds, fs in os.walk(base, followlinks=True):
            for f in fs:
                if 'file' in f:
                    fpath = ub.Path(r) / f
                    fpaths.append(fpath)


        dpath = real_bundle.resolve()

        for path in fpaths:
            # print(f'{path}')
            # print(f'{path.resolve()=}')
            resolved_rel = resolve_relative_to(path, dpath)
            print('resolved_rel = {!r}'.format(resolved_rel))
    """
    try:
        resolved_abs = resolve_directory_symlinks(path)
        resolved_rel = resolved_abs.relative_to(dpath)
    except ValueError:
        if strict:
            raise
        else:
            return path
    return resolved_rel


def resolve_directory_symlinks(path):
    """
    Only resolve symlinks of directories
    """
    return path.parent.resolve() / path.name
    # prev = path
    # curr = prev.parent
    # while prev != curr:
    #     if curr.is_symlink():
    #         rhs = path.relative_to(curr)
    #         resolved_lhs = curr.resolve()
    #         new_path = resolved_lhs / rhs
    #         return new_path
    #     prev = curr
    #     curr = prev.parent
    # return path
