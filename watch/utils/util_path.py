import os
import ubelt as ub
import pathlib
from watch.utils import util_pattern


def coercepath(path_like):
    """
    Args:
        path_like (str | ub.Path | os.PathLike):
            an object representing a filesystem path

    DEPRECATE:
        you canjust call ub.Path instead

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
            a glob pattern or list of glob patterns or a yaml list of glob
            patterns

    Returns:
        List[ubelt.Path]: Multiple paths that match the query

    Example:
        >>> # xdoctest: +SKIP
        >>> import watch
        >>> dvc_dpath = watch.find_dvc_dpath(tags='phase2_data')
        >>> data = [dvc_dpath / 'annotations/region_models/*.geojson']
        >>> from watch.utils.util_path import *  # NOQA
        >>> print(coerce_patterned_paths(dvc_dpath / 'annotations/region_models'))
        >>> print(coerce_patterned_paths([dvc_dpath / 'annotations/region_models/*.geojson']))
        >>> print(coerce_patterned_paths('./'))

        import watch
        dvc_dpath = watch.find_dvc_dpath(tags='phase2_data')
        data = [ub.Path(dvc_dpath / 'annotations/region_models')]
        expected_extension = ['*.geojson', '*.txt']

    Example:
        >>> import watch
        >>> empty_fpaths = coerce_patterned_paths(None)
        >>> assert len(empty_fpaths) == 0

    Example:
        >>> from watch.utils.util_path import *  # NOQA
        >>> import watch
        >>> import ubelt as ub
        >>> dpath = ub.Path.appdir('watch/test/utils/path/').ensuredir()
        >>> (dpath / 'file1.txt').touch()
        >>> (dpath / 'dir').ensuredir()
        >>> (dpath / 'dir' / 'subfile1.txt').touch()
        >>> (dpath / 'dir' / 'subfile2.txt').touch()
        >>> paths = coerce_patterned_paths(
        ...     f'''
        ...     - {dpath / 'file1.txt'}
        ...     - {dpath / 'file2.txt'}
        ...     - {dpath / 'dir'}
        ...     ''', expected_extension='.txt')
        >>> paths = [p.shrinkuser() for p in paths]
        >>> print('paths = {}'.format(ub.urepr(paths, nl=1)))

        paths = [
            Path('~/.cache/watch/test/utils/path/file1.txt'),
            Path('~/.cache/watch/test/utils/path/dir/subfile1.txt'),
            Path('~/.cache/watch/test/utils/path/dir/subfile2.txt'),
        ]
    """
    from watch.utils import util_yaml
    from os.path import isdir, join
    import glob

    if data is None:
        datas = []
    elif ub.iterable(data):
        datas = data
    else:
        datas = [data]

    # Resolve any yaml
    resolved_globs = []
    for data in datas:
        if isinstance(data, str):
            loaded = util_yaml.yaml_loads(data)
            if isinstance(loaded, str):
                loaded = [loaded]
            resolved_globs.extend(loaded)
        else:
            resolved_globs.append(data)

    datas = list(map(os.fspath, resolved_globs))
    paths = []
    for data_ in resolved_globs:
        if expected_extension is not None and isdir(data_):
            exts = expected_extension if ub.iterable(expected_extension) else [expected_extension]
            globpats = [join(data_, '*' + e) for e in exts]
        else:
            globpats = [data_]
        for globpat in globpats:
            paths.extend(list(glob.glob(os.fspath(globpat), recursive=True)))
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


class ChDir:
    """
    Context manager that changes the current working directory and then
    returns you to where you were.

    Args:
        dpath (PathLike | None):
            The new directory to work in.
            If None, then the context manager is disabled.

    Example:
        >>> from watch.utils.util_path import *  # NOQA
        >>> dpath = ub.Path.appdir('xdev/tests/chdir').ensuredir()
        >>> dir1 = (dpath / 'dir1').ensuredir()
        >>> dir2 = (dpath / 'dir2').ensuredir()
        >>> with ChDir(dpath):
        >>>     assert ub.Path.cwd() == dpath
        >>>     # changes to the given directory, and then returns back
        >>>     with ChDir(dir1):
        >>>         assert ub.Path.cwd() == dir1
        >>>         with ChDir(dir2):
        >>>             assert ub.Path.cwd() == dir2
        >>>             # changes inside the context manager will be reset
        >>>             os.chdir(dpath)
        >>>         assert ub.Path.cwd() == dir1
        >>>     assert ub.Path.cwd() == dpath
        >>>     with ChDir(dir1):
        >>>         assert ub.Path.cwd() == dir1
        >>>         with ChDir(None):
        >>>             assert ub.Path.cwd() == dir1
        >>>             # When disabled, the cwd does *not* reset at context exit
        >>>             os.chdir(dir2)
        >>>         assert ub.Path.cwd() == dir2
        >>>         os.chdir(dir1)
        >>>         # Dont change dirs, but reset to your cwd at context end
        >>>         with ChDir('.'):
        >>>             os.chdir(dir2)
        >>>         assert ub.Path.cwd() == dir1
        >>>     assert ub.Path.cwd() == dpath
    """

    def __init__(self, dpath):
        self.context_dpath = dpath
        self.orig_dpath = None

    def __enter__(self):
        if self.context_dpath is not None:
            self.orig_dpath = os.getcwd()
            os.chdir(self.context_dpath)
        return self

    def __exit__(self, a, b, c):
        if self.context_dpath is not None:
            os.chdir(self.orig_dpath)


def sidecar_glob(main_pat, sidecar_ext, main_key='main', sidecar_key=None,
                 recursive=0):
    """
    Similar to a regular glob, but returns a dictionary with associated
    main-file / sidecar-file pairs.

    TODO:
        add as a general option to Pattern.paths?

    Args:
        main_pat (str | PathLike):
            glob pattern for the main non-sidecar file

    Yields:
        Dict[str, Path | None]

    Notes:
        A sidecar file is defined by the sidecar extension. We usually use this
        for .dvc sidecars.

        When the pattern includes a .dvc suffix, the result will include those .dvc
        files and any matching main files they correspond to. Note: if you search
        for paths like `foo_*.dvc` this might skiped unstaged files. Therefore it
        is recommended to only include the .dvc suffix in the pattern ONLY if you
        do not want any unstaged files.

        If you want both staged and unstaged files, ensure the pattern does not
        exclude objects without a .dvc suffix (i.e. don't end the pattern with
        .dvc).

        When the pattern does not include a .dvc suffix, we include all those
        files, for other files that exist by adding a .dvc suffix.

        With the pattern matches both a dvc and non-dvc file, they are grouped
        together.

    Example:
        >>> # xdoctest: +REQUIRES(env:DVC_DPATH)
        >>> import watch
        >>> dvc_dpath = watch.find_smart_dvc_dpath()
        >>> bundle_dpath = dvc_dpath / 'Cropped-Drop3-TA1-2022-03-10'
        >>> sidecar_ext = '.dvc'
        >>> print(ub.repr2(list(sidecar_glob(bundle_dpath / '*R*', sidecar_ext)), nl=2))
        >>> print(ub.repr2(list(sidecar_glob(bundle_dpath / '*.dvc', sidecar_ext)), nl=2))

    Example:
        >>> from watch.utils.util_path import *  # NOQA
        >>> dpath = ub.Path.appdir('xdev/tests/sidecar_glob')
        >>> dpath.delete().ensuredir()
        >>> (dpath / 'file1').touch()
        >>> (dpath / 'file1.ext').touch()
        >>> (dpath / 'file1.ext.car').touch()
        >>> (dpath / 'file2.ext').touch()
        >>> (dpath / 'file3.ext.car').touch()
        >>> (dpath / 'file4.car').touch()
        >>> (dpath / 'file5').touch()
        >>> (dpath / 'file6').touch()
        >>> (dpath / 'file6.car').touch()
        >>> (dpath / 'file7.bike').touch()
        >>> def _handle_resulst(results):
        ...     results = list(results)
        ...     for row in results:
        ...         for k, v in row.items():
        ...             if v is not None:
        ...                 row[k] = v.relative_to(dpath)
        ...     print(ub.repr2(results, sv=1))
        ...     return results
        >>> main_key = 'main',
        >>> sidecar_key = '.car'
        >>> sidecar_ext = '.car'
        >>> main_pat = dpath / '*'
        >>> _handle_resulst(sidecar_glob(main_pat, sidecar_ext))
        >>> _handle_resulst(sidecar_glob(dpath / '*.ext', '.car'))
        >>> _handle_resulst(sidecar_glob(dpath / '*.car', '.car'))
        >>> _handle_resulst(sidecar_glob(dpath / 'file*.ext', '.car'))
        >>> _handle_resulst(sidecar_glob(dpath / '*', '.ext'))
    """
    from watch.utils import util_pattern
    import warnings
    import os
    _len_ext = len(sidecar_ext)
    main_pat = os.fspath(main_pat)
    glob_patterns = [main_pat]
    if main_pat.endswith(sidecar_ext):
        warnings.warn(
            'The main path query should not end with the sidecar extension.'
            ' {main_pat=} {sidecar_ext=}'
        )
        # We could have a variant that removes the extension, but lets not do
        # that and document it.
        # glob_patterns.append(pat[:-_len_ext])
    else:
        if main_pat.endswith('/*'):
            # Optimization dont need an extra pattern in this case
            pass
        else:
            glob_patterns.append(main_pat + sidecar_ext)

    mpat = util_pattern.MultiPattern.coerce(glob_patterns)
    if sidecar_key is None:
        sidecar_key = sidecar_ext
    default = {main_key: None, sidecar_key: None}
    id_to_row = ub.ddict(default.copy)
    paths = mpat.paths(recursive=recursive)

    def _gen():
        for path in paths:
            parent = path.parent
            name = path.name
            if name.endswith(sidecar_ext):
                this_key = sidecar_key
                other_key = main_key
                main_path = parent / name[:-_len_ext]
                other_path = main_path
            else:
                this_key = main_key
                other_key = sidecar_key
                main_path = path
                other_path = parent / (name + sidecar_ext)
            needs_yeild = main_path not in id_to_row
            row = id_to_row[main_path]
            row[this_key] = path
            if row[other_key] is None:
                if other_path.exists():
                    row[other_key] = other_path
            if needs_yeild:
                yield row
    # without this, yilded rows might modify themselves later, that is
    # confusing for a user. Don't do it or come up with a scheme where we
    # detect if a row is "complete" and only yeild it then
    # We could more easilly do this if we used a walk-style find and pattern
    # match mechanism
    rows = list(_gen())
    yield from rows
