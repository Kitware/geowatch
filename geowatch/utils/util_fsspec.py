"""
fsspec wrappers that should make working with S3 / the local file system
seemless.

TODO:
    Someone must have already implemented this somewhere. Find that to
    either use directly or as a reference.

Look into:
    https://github.com/fsspec/universal_pathlib
    https://pypi.org/project/pathlibfs/
"""
from os.path import join
import pathlib
import ubelt as ub
import os
import fsspec
import types

NOOP_CALLBACK = fsspec.callbacks.NoOpCallback()


class FSPath(str):
    """
    Provide a pathlib.Path-like way of interacting with fsspec.

    This has a few notable differences with pathlib.Path. We inherit from
    :class:`str` because :class:`pathlib.Path` semantics can break protocols
    sections of URIs. This means we have to use :mod:`os.path` functions
    to implement things like :meth:`FSPath.relative_to` and
    :meth:`FSPath.joinpath` (which behave differently than pathlib)

    Note:
        Not all of the fsspec / pathlib operations are currently implemented,
        add as needed.

    Example:
        >>> cwd = FSPath.coerce('.')
        >>> print(cwd)
        >>> print(cwd.fs)
    """
    # Final subclasses must define this as a string to be passed to
    # fsspec.filesystem(__protocol__)
    __protocol__ : str | types.NotImplementedType = NotImplemented

    @classmethod
    def _new_fs(cls, **kwargs):
        """
        Create a new filesystem instance based on __protocol__
        """
        fs_cls = cls.get_filesystem_class()
        fs = fs_cls(**kwargs)
        return fs

    @classmethod
    def get_filesystem_class(cls):
        fs_cls = fsspec.get_filesystem_class(cls.__protocol__)
        return fs_cls

    @classmethod
    def _current_fs(cls):
        """
        The "default" FileSystem object.  Get the most recent filesystem with
        this protocol, or create a new one with defaults.

        Returns:
            AbstractFileSystem
        """
        fs_cls = cls.get_filesystem_class()
        fs = fs_cls.current()
        return fs

    def __new__(cls, path, *, fs=None):
        # Note: the value of the string is set in the __new__ method because
        # strings are immutable. So we dont need to call super or anything.
        # The first argument will be the value of the string.
        if fs is None:
            # Lazy creation of a new fs
            fs = cls._current_fs()
        self = str.__new__(cls, path)
        self._fs: fsspec.AbstractFileSystem = fs
        return self

    @property
    def fs(self) -> fsspec.AbstractFileSystem:
        return self._fs

    @fs.setter
    def fs(self, value: fsspec.AbstractFileSystem):
        self._fs = value

    @classmethod
    def coerce(cls, path):
        """
        Determine which backend to use automatically

        Example:
            >>> path2 = FSPath.coerce('/local/path')
            >>> print(f'path2={path2}')
            >>> assert path2.is_local()
            >>> # xdoctest: +REQUIRES(module:s3fs)
            >>> path1 = FSPath.coerce('s3://demo_bucket')
            >>> print(f'path1={path1}')
            >>> assert path1.is_remote()
        """
        path = os.fspath(path)
        if path.startswith(('s3:', '/vsis3/')):
            self = S3Path.coerce(path)
        elif path.startswith(('ssh:')):
            raise NotImplementedError('getting ssh to work with uris is tricky')
        else:
            self = LocalPath(path)
        return self

    @property
    def path(self):
        """
        By default the string representation is assumed to be the entire path,
        however, for subclasses like SSHPath it is necessary to overwrite this
        so the core object represents the entire URI, but this just returns the
        path part, which is what the fsspec.FileSystem object expects.
        """
        return self

    def relative_to(self, other):
        return self.__class__(os.path.relpath(self, other))

    def is_remote(self):
        return isinstance(self, RemotePath)

    def is_local(self):
        return isinstance(self, LocalPath)

    def open(self, mode='rb', block_size=None, cache_options=None, compression=None):
        """
        Example:
            >>> from geowatch.utils.util_fsspec import *  # NOQA
            >>> from geowatch.utils import util_fsspec
            >>> dpath = util_fsspec.LocalPath.appdir('geowatch/fsspec/tests/open').ensuredir()
            >>> fpath = dpath / 'file.txt'
            >>> file = fpath.open(mode='w')
            >>> file.write('hello world')
            >>> file.close()
            >>> assert fpath.read_text() == fpath.open('r').read()
        """
        return self.fs.open(self.path, mode=mode, block_size=block_size,
                            cache_options=cache_options,
                            compression=compression)

    def ls(self, detail=False, **kwargs):
        """
        Example:
            >>> from geowatch.utils.util_fsspec import *  # NOQA
            >>> import ubelt as ub
            >>> dpath = ub.Path.appdir('geowatch', 'tests', 'fsspec', 'ls').ensuredir()
            >>> (dpath / 'file1').touch()
            >>> (dpath / 'file2').touch()
            >>> (dpath / 'subfolder').ensuredir()
            >>> self = FSPath.coerce(dpath)
            >>> results = self.ls()
            >>> assert sorted(results) == sorted(map(str, dpath.ls()))
        """
        cls = self.__class__
        results = self.fs.ls(self.path, detail=detail, **kwargs)
        if detail:
            return results
        else:
            # Hack:
            if self.__protocol__ == 'file':
                return [cls(p, fs=self.fs) for p in results]
            else:
                return [cls(self.__protocol__ + '://' + p, fs=self.fs) for p in results]

    def touch(self, truncate=False, **kwargs):
        """
        Example:
            >>> import ubelt as ub
            >>> dpath = ub.Path.appdir('geowatch', 'tests', 'fsspec', 'touch').ensuredir()
            >>> dpath_ = FSPath.coerce(dpath)
            >>> self = (dpath_ / 'file')
            >>> self.touch()
            >>> assert self.exists()
            >>> assert (dpath / 'file').exists()
        """
        self.fs.touch(self.path, truncate=truncate, **kwargs)

    def move(self, path2, recursive='auto', maxdepth=None, verbose=1, **kwargs):
        """
        Note: this may work differently than ubelt.Path.move, ideally we should
        rectify this. The difference case is what happens when you move:

            ./path/to/dir -> ./path/to/other/dir

        Does `./path/to/dir` merge into `./path/to/other/dir`, or do you get
        all of the src contents in `./path/to/other/dir/dir`?

        Ignore:
            import ubelt as ub
            root = ub.Path.appdir('geowatch', 'tests', 'fsspec', 'move').delete().ensuredir()
            dpath1 = (root / 'path/to/dir').ensuredir()
            dpath2 = (root / 'path/to/other/dir').ensuredir()

            # Add content to both dirs
            (dpath1 / 'file1').write_text('a1')
            (dpath1 / 'file2').write_text('a2')
            (dpath2 / 'file1').write_text('b1')
            (dpath2 / 'file3').write_text('b3')

            dpath1.ls()
            dpath2.ls()

            # ubelt will complain moves are only allowed to locs that dont
            # exist cool, that makes sense.
            dpath1.move(dpath2)

            dpath1_alt = FSPath.coerce(dpath1)
            dpath2_alt = FSPath.coerce(dpath2)

            if 0:
                # This moves it into the directory, which is not the behavior I
                # want.
                dpath1_alt.move(dpath2_alt)
                assert not dpath1.exists()
                dpath2.ls()
            else:
                # This has the desired behavior, but requries an akward call
                # sig
                dpath1_alt.move(dpath2_alt.parent)
                assert not dpath1.exists()
                dpath2.ls()
        """
        if recursive == 'auto':
            recursive = self.is_dir()
        if verbose:
            print(f'Move {self} -> {path2}')
        self.fs.move(self.path, path2, recursive=recursive, maxdepth=maxdepth,
                     **kwargs)

    def delete(self, recursive='auto', maxdepth=True, verbose=1):
        """
        Deletes this file or this directory (and all of its contents)

        Unlike fs.delete, this will not error if the file doesnt exist. See
        :func:`FSPath.rm` if you want standard error-ing behavior.
        """
        if verbose:
            print(f'Delete {self}')
        if recursive == 'auto':
            recursive = self.is_dir()
        try:
            return self.fs.delete(self.path, recursive=recursive, maxdepth=maxdepth)
        except FileNotFoundError:
            ...

    def rm(self, recursive='auto', maxdepth=True):
        """
        Deletes this file or this directory (and all of its contents)
        """
        if recursive == 'auto':
            recursive = self.is_dir()
        return self.fs.rm(self.path, recursive=recursive, maxdepth=maxdepth)

    def mkdir(self, create_parents=True, **kwargs):
        """
        Note:
            does nothing on some filesystems (e.g. S3)
        """
        return self.fs.mkdir(self.path, create_parents=create_parents, **kwargs)

    def stat(self):
        return self.fs.stat(self.path)

    def is_dir(self):
        return self.fs.isdir(self.path)

    def is_file(self):
        return self.fs.isfile(self.path)

    def is_link(self):
        try:
            return self.fs.islink(self.path)
        except AttributeError:
            return False

    def exists(self):
        return self.fs.exists(self.path)

    def write_text(self, value, **kwargs):
        return self.fs.write_text(self.path, value, **kwargs)

    def read_text(self, **kwargs):
        return self.fs.read_text(self.path, **kwargs)

    def walk(self, include_protocol='auto', **kwargs):
        """
        Yields:
            Tuple[Self, List[str], List[str]] - root, dir names, file names
        """
        if include_protocol == 'auto':
            include_protocol = self.is_remote()
        if include_protocol:
            for root, dnames, fnames in self.fs.walk(self.path, **kwargs):
                root = self.__class__(self.fs.unstrip_protocol(root), fs=self.fs)
                yield root, dnames, fnames
        else:
            for root, dnames, fnames in self.fs.walk(self, **kwargs):
                root = self.__class__(root, fs=self.fs)
                yield root, dnames, fnames

    @property
    def parent(self):
        """
        Example:
            >>> self = FSPath.coerce('foo/bar/baz.jaz.raz')
            >>> assert str(ub.Path(self).parent) == self.parent
            >>> assert self.parent == 'foo/bar'
        """
        return self.__class__(os.path.dirname(self), fs=self.fs)

    @property
    def name(self):
        """
        Example:
            >>> self = FSPath.coerce('foo/bar/baz.jaz')
            >>> assert ub.Path(self).name == self.name
            >>> assert self.name == 'baz.jaz'
        """
        return os.path.basename(self.path)

    @property
    def stem(self):
        """
        Example:
            >>> self = FSPath.coerce('foo/bar/baz.jaz')
            >>> assert ub.Path(self).stem == self.stem
            >>> assert self.stem == 'baz'
        """
        return os.path.splitext(self.path.name)[0]

    @property
    def suffix(self):
        """
        Example:
            >>> self = FSPath.coerce('foo/bar/baz.jaz.raz')
            >>> assert ub.Path(self).suffix == self.suffix
            >>> assert self.suffix == '.raz'
        """
        return os.path.splitext(self.path.name)[1]

    @property
    def suffixes(self):
        """
        Example:
            >>> self = FSPath.coerce('foo/bar/baz.jaz.raz')
            >>> assert ub.Path(self).suffixes == self.suffixes
            >>> assert self.suffixes == ['.jaz', '.raz']
        """
        return ['.' + x for x in self.name.split('.')[1:]]

    @property
    def parts(self):
        """
        Example:
            >>> self = FSPath.coerce('foo/bar/baz.jaz.raz')
            >>> assert ub.Path(self).parts == self.parts
            >>> assert self.parts == ('foo', 'bar', 'baz.jaz.raz')
        """
        return tuple(self.path.split(self.fs.sep))

    def copy(self, dst, recursive='auto', maxdepth=None, on_error=None,
             callback=None, verbose=1, idempotent=True, overwrite=False,
             **kwargs):
        """
        Copies this file or directory to dst

        Abtracts fsspec copy / put / get.

        If dst ends with a "/", it will be assumed to be a directory, and
        target files will go within.

        Unlike fsspec, this attempts to be idempotent. See [FSSpecCopy]_.

        Args:
            dst (FSPath): location to copy to

            recursive (bool | str):
                If 'auto' (the default), attempt to determine if this is a
                directory or a file. Set to True if it is a directory and False
                otherwise. If you know what this is beforehand, you can set it
                explicitly to be more efficient.

            maxdepth (int | None):
                only makes sense when recursive is True

            callback (None | callable):
                for put / get cases

            on_error (str):
                either "raise", "ignore". Only applicable in the "copy" case.

            idempotent (bool):
                if False, use standard fsspec behavior, otherwise attempt to
                be idempotent.

            overwrite (bool):
                if True, overwrite existing data instead of erroring. Defaults
                to False.

        Note:
            There are different functions depending on if we are going from
            remote->remote (copy), local->remote (put), or remote->local (get)

        References:
            .. [FSSpecCopy] https://filesystem-spec.readthedocs.io/en/latest/copying.html

        Example:
            >>> from geowatch.utils import util_fsspec
            >>> dpath = util_fsspec.LocalPath.appdir('geowatch/fsspec/tests/copy').ensuredir()
            >>> src_dpath = (dpath / 'src').ensuredir()
            >>> for i in range(100):
            ...     (src_dpath / 'file_{i:03d}.txt').write_text('hello world' * 100)
            >>> dst_dpath = (dpath / 'dst')
            >>> dst_dpath.delete()
            >>> src_dpath.copy(dst_dpath, verbose=3)
            >>> dst_dpath.delete()
            >>> if 0:
            >>>     from fsspec.callbacks import TqdmCallback
            >>>     callback = TqdmCallback(tqdm_kwargs={"desc": "Your tqdm description"})
            >>>     src_dpath.copy(dst_dpath, callback=callback)
        """
        if recursive == 'auto':
            # On S3, asking if something is a dir can give permission
            # issues, whereas asking if something is a file seems ok.
            # try:
            # recursive = self.is_dir()
            # except PermissionError:
            recursive = not self.is_file()

        if callback is None:
            callback = NOOP_CALLBACK

        commonkw = {
            'recursive': recursive,
            'maxdepth': maxdepth,
            **kwargs,
        }

        dst: FSPath

        if overwrite:
            raise NotImplementedError

        if verbose:
            print(f'Copy {self} -> {dst}')

        # HANDLE SPECIAL CASE WHERE FSSPEC IS NOT IDEMPOTENT
        if idempotent:
            if recursive and dst.exists():
                dst = dst.parent + '/'

        if isinstance(self, LocalPath):
            if not self.exists():
                raise IOError(f'{self} does not exist')
            if isinstance(dst, RemotePath):

                # TODO: test if we are an empty directory and fail because
                # generally we cant copy an empty directory to a remote.
                try:
                    if recursive:
                        if verbose >= 3:
                            print(' * local -> remote (put recursive)')
                        return dst.fs.put(self.path, dst, **commonkw, callback=callback)
                    else:
                        if verbose >= 3:
                            print(' * local -> remote (put_file)')
                        return dst.fs.put_file(self.path, dst, callback=callback)
                except FileExistsError:
                    # TODO: overwrite
                    raise

            elif isinstance(dst, LocalPath):
                if verbose >= 3:
                    print(' * local -> local')
                return self.fs.copy(self.path, dst, **commonkw, callback=callback)
            else:
                raise TypeError(type(dst))
        elif isinstance(self, RemotePath):
            if isinstance(dst, RemotePath):
                if verbose >= 3:
                    print(' * remote -> remote')
                return self.fs.copy(self.path, dst, **commonkw, on_error=on_error)
            elif isinstance(dst, (LocalPath, pathlib.Path)):

                if recursive:
                    if verbose >= 3:
                        print(' * remote -> local (get recursive)')
                    return self.fs.get(self.path, dst, **commonkw, callback=callback)
                else:
                    # Using put on an s3 bucket seems like it fails when
                    # directories have tight permissions, but put file
                    # seems ok. Need to ensure that we are actually just
                    # using a file in this instance.
                    if verbose >= 3:
                        print(' * remote -> local (get_file)')
                    return self.fs.get_file(self.path, dst, callback=callback)
            else:
                raise TypeError(type(dst))
        else:
            raise TypeError(type(self))

    def joinpath(self, *others):
        return self.__class__(join(self, *others), fs=self.fs)

    def __truediv__(self, other):
        return self.__class__(join(self, other), fs=self.fs)

    def __add__(self, other):
        """
        Returns a new string starting with this fspath representation.
        """
        return self.__class__(str(self) + other, fs=self.fs)

    def __radd__(self, other):
        """
        Returns a new string ending with this fspath representation.
        """
        return self.__class__(other + str(self), fs=self.fs)

    def tree(self, max_files=100, dirblocklist=None, show_nfiles='auto',
             return_text=False, return_tree=True, pathstyle='name',
             max_depth=None, with_type=False, abs_root_label=True,
             colors=not ub.NO_COLOR):
        """
        Filesystem tree representation

        Like the unix util tree, but allow writing numbers of files per directory
        when given -d option

        Ported from xdev.misc.tree_repr

        TODO:
            instead of building the networkx structure and then waiting to
            display everything, build and display simultaniously. Will require
            using a modified version of write_network_text

        Args:
            max_files (int | None) : maximum files to print before supressing a directory
            pathstyle (str): can be rel, name, or abs
            return_tree (bool): if True return the tree
            return_text (bool): if True return the text
            maxdepth (int | None): maximum depth to descend
            abs_root_label (bool): if True force the root to always be absolute
            colors (bool): if True use rich
        """
        import io
        import os
        import networkx as nx
        from cmd_queue.util.util_networkx import write_network_text
        from kwutil.util_pattern import MultiPattern
        # tree = nx.OrderedDiGraph()
        tree = nx.DiGraph()

        if dirblocklist is not None:
            dirblocklist = MultiPattern.coerce(dirblocklist, hint='glob')

        def _make_label(p, force_abs=False):
            if force_abs:
                pathrep = p
            elif pathstyle == 'rel':
                pathrep = p.relative_to(self)
            elif pathstyle == 'name':
                pathrep = p.name
            elif pathstyle == 'abs':
                pathrep = p
            else:
                KeyError(pathstyle)

            types = []
            islink = p.is_link()
            isdir = p.is_dir()
            isfile = p.is_file()
            isbroken = False
            scolor = ''
            tcolor = ''
            L_scolor = ''
            L_tcolor = ''
            if islink:
                if colors:
                    L_scolor = '[cyan]'
                    L_tcolor = '[/cyan]'
                types.append('L')
                if not isfile and not isdir:
                    isbroken = True
                    if isbroken:
                        if colors:
                            L_scolor = '[red]'
                            L_tcolor = '[/red]'
                    types.append('B')

            if isfile:
                if colors:
                    scolor = '[reset]'
                    tcolor = '[/reset]'
                    if os.access(p, os.X_OK):
                        scolor = '[green]'
                        tcolor = '[/green]'
                types.append('F')
            if isdir:
                if colors:
                    scolor = f'[blue][link={p.absolute()}]'
                    tcolor = '[/link][/blue]'
                types.append('D')

            if islink:
                target = os.readlink(p)
                pathrep = L_scolor + pathrep + L_tcolor + ' -> ' + scolor + target + tcolor
            else:
                pathrep = scolor + pathrep + tcolor

            if with_type:
                typelabel = ''.join(types)
                return f'({typelabel}) ' + pathrep
            else:
                return pathrep

        # TODO: rectify with "find"
        start_depth = self.count(self.fs.sep)
        for root, dnames, fnames in self.walk():
            curr_depth = root.count(self.fs.sep)

            if max_depth is not None:
                if (curr_depth - start_depth):
                    del dnames[:]

            if dirblocklist is not None:
                dnames[:] = [
                    dname for dname in dnames if not dirblocklist.match(dname)]

            dnames[:] = sorted(dnames)
            tree.add_node(root)

            too_many_files = max_files is not None and len(fnames) >= max_files

            if show_nfiles == 'auto':
                show_nfiles_ = too_many_files
            else:
                show_nfiles_ = show_nfiles

            num_files = len(fnames)
            if show_nfiles_:
                prefix = '[ {} ] '.format(num_files)
            else:
                prefix = ''

            force_abs = abs_root_label and curr_depth == start_depth
            label = '{}{}'.format(prefix, _make_label(root, force_abs))

            tree.nodes[root]['label'] = label

            if not too_many_files:
                for fname in fnames:
                    fpath = root / fname
                    tree.add_node(fpath)
                    tree.nodes[fpath]['label'] = _make_label(fpath)
                    tree.add_edge(root, fpath)

            for dname in dnames:
                dpath = root / dname
                tree.add_node(dpath)
                tree.nodes[dpath]['label'] = _make_label(dpath)
                tree.add_edge(root, dpath)

        file = io.StringIO()
        write_network_text(tree, file)
        file.seek(0)
        text = file.read()

        info = {}

        if return_text:
            info['text'] = text
        else:
            if colors:
                from rich import print as rprint
                rprint(text)
            else:
                print(text)

        if return_tree:
            info['tree'] = tree
        return info


class LocalPath(FSPath):
    """
    The implementation for the local filesystem

    CommandLine:
        xdoctest -m geowatch.utils.util_fsspec LocalPath
        xdoctest geowatch/utils/util_fsspec.py

    Example:
        >>> from geowatch.utils.util_fsspec import *  # NOQA
        >>> dpath = ub.Path.appdir('geowatch/tests/util_fsspec/demo')
        >>> dpath.delete().ensuredir()
        >>> (dpath / 'file1.txt').write_text('data')
        >>> (dpath / 'dpath').ensuredir()
        >>> (dpath / 'dpath/file2.txt').write_text('data')
        >>> self = LocalPath(dpath).absolute()
        >>> print(f'self={self}')
        >>> print(self.ls())
        >>> info = self.tree()
        >>> fsspec_dpath = (dpath / 'dpath')
        >>> fsspec_fpath = (dpath / 'file1.txt')
        >>> pathlib_dpath = ub.Path(dpath / 'pathlib_dpath')
        >>> pathlib_fpath = ub.Path(dpath / 'pathlib_fpath')
        >>> assert not pathlib_dpath.exists()
        >>> assert not pathlib_fpath.exists()
        >>> fsspec_dpath.copy(pathlib_dpath)
        >>> fsspec_fpath.copy(pathlib_fpath)
        >>> assert pathlib_dpath.exists()
        >>> assert pathlib_fpath.exists()
    """
    __protocol__ = 'file'

    def ensuredir(self, mode=0o0777):
        pathlib.Path(self).mkdir(mode=mode, parents=True, exist_ok=True)
        return self

    def absolute(self):
        return self.__class__(os.path.abspath(self), fs=self.fs)

    @classmethod
    def appdir(cls, *args, **kw):
        return cls(str(ub.Path.appdir(*args, **kw)))


class MemoryPath(FSPath):
    """
    Ignore:
        self = MemoryPath('/')
        self.ls()
        (self / 'file').write_text('data')
        self.ls()

        ref_mempath = MemoryPath('/')
        ref_mempath.ls()

        # The MemoryFileSystem is global for the entire program
        new_mempath = MemoryPath('/', fs=MemoryPath._new_fs())
        new_mempath.ls()

        # Show the entire contents of the memory filesystem
        new_mempath.fs.store
    """
    __protocol__ = 'memory'


class RemotePath(FSPath):
    """
    Abstract implementation for all remote filesystems
    """


class S3Path(RemotePath):
    """
    The specific S3 remote filesystem.

    Control credentials with the environment variables: AWS_ACCESS_KEY_ID,
    AWS_SECRET_ACCESS_KEY, and AWS_SESSION_TOKEN.

    A single S3 filesystem is used by default, but you can work with multiple
    of them if you pass in the fs object. E.g.

        fs = S3Path._new_fs(profile='iarpa')
        self = S3Path('s3://kitware-smart-watch-data/', fs=fs)
        self.ls()

        # Can also do
        S3Path.register_bucket('s3://kitware-smart-watch-data', profile='iarpa')
        self = S3Path.coerce('s3://kitware-smart-watch-data/')
        self.ls()

        # Demo of multiple registered buckets
        S3Path.register_bucket('s3://usgs-landsat-ard', profile='iarpa', requester_pays=True)
        self = S3Path.coerce('s3://usgs-landsat-ard/collection02')
        self.ls()

        self = S3Path.coerce('/vsis3/usgs-landsat-ard/collection02')
        self.ls()

    SeeAlso:
        geowatch.heuristics.register_known_fsspec_s3_buckets

    To work with different S3 filesystems,

    See [S3FS_Docs]_.

    Requirements:
        s3fs>=2023.6.0

    References:
        .. [S3FS_Docs] https://s3fs.readthedocs.io/en/latest/?badge=latest

    Example:
        >>> # xdoctest: +REQUIRES(module:s3fs)
        >>> fs = S3Path._new_fs()
    """
    __protocol__ = 's3'

    _bucket_registry = {}

    def _as_gdal_vsi(self):
        # replace the s3:// part with /vsis3
        return '/vsis3/' + self[len(self.__protocol__) + 3:]

    def ensuredir(self, mode=0o0777):
        # Does nothing on S3 because you cannot create an empty directory
        # they are always auto-created for you.
        ...
        return self

    @classmethod
    def register_bucket(cls, bucket, **kwargs):
        # Parse out the bucket part
        assert '/' not in bucket[5], 'not a bucket'
        fs = cls._new_fs(**kwargs)
        # want to store and associate buckets with fs objects
        cls._bucket_registry[bucket] = fs

    @classmethod
    def coerce(cls, path):
        if path.startswith('/vsis3/'):
            # convert gdal virtual filesystems to s3 paths
            path = 's3://' + path[7:]
        # Parse out the bucket part
        bucket = path[:path.find('/', 5)]
        fs = cls._bucket_registry.get(bucket, None)
        self = cls(path, fs=fs)
        return self


class SSHPath(RemotePath):
    """
    Ignore:
        fs = SSHPath._new_fs(host='localhost')
        self = SSHPath('.', fs=fs)
        self.ls()
        self = SSHPath('misc/notes', fs=fs)
        self.tree()
    """
    __protocol__ = 'ssh'

    @property
    def host(self):
        return self.fs.host


class _BROKEN_SSHURI(RemotePath):
    """
    The idea here is to do something like the bucket registery in S3Path, but
    weird corner cases pop up here making the path forward not easy to
    identify.

    Requires a RFC3986 style URI.

    The idea here is that we keep the remote information in the string so we
    can automatically lookup the appropriate fs object and only have to
    authenticate it once.

    Note this is a mess, and we may just drop this idea...

    Ignore:
        from geowatch.utils.util_fsspec import *  # NOQA
        self = SSHPath('ssh://localhost/.')
        print(f'self={self}')
        self.ls()
        (self / 'misc/notes').tree()

        self = SSHPath('misc/notes', fs=fs)
        self.tree()

        fs = SSHPath._new_fs(host='localhost')
        path = 'ssh://localhost/.'
        self = SSHPath(path, fs=fs)
    """
    __protocol__ = 'ssh'

    _fs_registry = {}

    @property
    def host(self):
        return self.fs.host

    @classmethod
    def _lookup_fs_remote(cls, path, fs=None, **kwargs):
        import uritools
        uri_parsed = uritools.urisplit(path)

        remote = uritools.uricompose(
            scheme=uri_parsed.scheme, authority=uri_parsed.authority,
            userinfo=uri_parsed.userinfo, host=uri_parsed.host,
            port=uri_parsed.port)

        if remote in cls._fs_registry:
            fs = cls._fs_registry[remote]

        if fs is None:
            fs = cls.register_remote(remote, **kwargs)

        return fs

    @classmethod
    def register_remote(cls, remote, **kwargs):
        import uritools
        uri_parsed = uritools.urisplit(remote)
        assert not uri_parsed.path
        remote = uritools.uricompose(
            scheme=uri_parsed.scheme, authority=uri_parsed.authority,
            userinfo=uri_parsed.userinfo, host=uri_parsed.host,
            port=uri_parsed.port)
        ssh_kwargs = {}
        ssh_kwargs['host'] = uri_parsed.host
        if uri_parsed.port is not None:
            ssh_kwargs['port'] = uri_parsed.port
        if uri_parsed.userinfo is not None:
            ssh_kwargs['username'] = uri_parsed.userinfo
        ssh_kwargs.update(kwargs)
        print('new remote')
        print(f'remote={remote}')
        print(f'ssh_kwargs={ssh_kwargs}')
        fs = cls._new_fs(**ssh_kwargs)
        print(f'fs={fs}')
        cls._fs_registry[remote] = fs
        return fs

    def __new__(cls, path, *, fs=None):
        parsed, remote = _uri_parse(path)
        if fs is None:
            fs = cls._lookup_fs_remote(path)
        self = super().__new__(cls, path, fs=fs)
        self._path = parsed.path
        return self

    @property
    def path(self):
        return self._path


def _uri_parse(uri):
    import uritools
    uri_parsed = uritools.urisplit(uri)
    remote = uritools.uricompose(
        scheme=uri_parsed.scheme, authority=uri_parsed.authority,
        userinfo=uri_parsed.userinfo, host=uri_parsed.host,
        port=uri_parsed.port)
    return uri_parsed, remote


def _devtest_ssh_registry1():
    """
    TODO:
        - [ ] Generate a test that setups the docker ssh server

    This assumes you've run the instructions in
        ~/code/ci-docker/ubuntu_sshd.dockerfile

    and have a local ssh server running
    """
    from geowatch.utils.util_fsspec import SSHPath

    ssh_kwargs = {
        'port': 2222,
        'username': 'ubuntu',
        # 'password': 'ubuntu',
        'key_filename': str(ub.Path('~/code/ci-docker/tmp_keys/id_ed25519').expand()),
    }
    fs1 = SSHPath._new_fs(host='localhost', **ssh_kwargs)

    fs2 = SSHPath._new_fs(host='localhost')
    fs1.open('testfile2', mode='w').write('foo')

    # Should be files on the docker ssh server
    fs1.ls('.')

    # Should be files on the local ssh server (todo: test with 2 docker instances)
    fs2.ls('.')

    # import fsspec
    # uri1 = 'ssh://localhost:22/testfile1'
    # uri2 = 'ssh://ubuntu@localhost:2222/testfile2'
    # file1 = fsspec.open(uri1, mode='w')
    # file2 = fsspec.open(uri2)

    # p1 = SSHPath(uri1, fs=fs1)
    # p2 = SSHPath(uri1, fs=fs1)
    # fsspec.core.get_fs_token_paths(uri1)
    # fsspec.core.get_fs_token_paths(uri2)

    # # uri1 = 'ssh://localhost:22/.ssh'
    # uri2 = 'ssh://ubuntu@localhost:2222/.ssh'

    # import uritools
    # # uri_parsed1 = uritools.urisplit(uri1)
    # # uri_parsed2 = uritools.urisplit(uri2)

    # remote = uri2
    # options = {
    #     'key_filename': str(ub.Path('~/code/ci-docker/tmp_keys/id_ed25519').expand()),
    # }
    # registry = {}
    # if remote in registry:
    #     raise ValueError('already exists')
    # uri_parsed = uritools.urisplit(remote)
    # options['port'] = uri_parsed.port
    # options['host'] = uri_parsed.host
    # options['username'] = uri_parsed.userinfo
    # fs = SSHPath._new_fs(**options)
    # # registry
    # prefix = uritools.uricompose(
    #     scheme=uri_parsed.scheme, authority=uri_parsed.authority,
    #     userinfo=uri_parsed.userinfo, host=uri_parsed.host,
    #     port=uri_parsed.port)
    # registry[prefix] = fs

    # uri = 's3://usgs-landsat-ard/collection02/oli-tirs/2016/CU/003/008/LC08_CU_003008_20161026_20210502_02/LC08_CU_003008_20161026_20210502_02_SR_B2.TIF'
    # parsed, prefix = _uri_parse(uri)
    # prefix = 's3://usgs-landsat-ard'
    # parsed, prefix = _uri_parse(uri)

    from uritools import uricompose
    from uritools import urisplit
    uricompose(scheme='foo', host='example.com', port=8042, path='/over/there',
               query={'name': 'ferret'}, fragment='nose')

    x = uricompose(scheme='foo', host='example.com', port=8042, path='/.')
    urisplit(x).authority
    uricompose(scheme='foo', host='example.com', path='/.')


def _devtest_ssh_registry2():
    # from geowatch.utils.util_fsspec import SSHPath, FSPath
    prefix1 = 'ssh://localhost:22'
    prefix2 = 'ssh://ubuntu@localhost:2222'
    options1 = {
        'port': 22,
        'host': 'localhost',
    }
    options2 = {
        'port': 2222,
        'host': 'localhost',
        'username': 'ubuntu',
        'key_filename': str(ub.Path('~/code/ci-docker/tmp_keys/id_ed25519').expand()),
    }
    f1 = SSHPath.register_remote(prefix1, **options1)
    f2 = SSHPath.register_remote(prefix2, **options2)
    print(f'f1={f1}')
    print(f'f2={f2}')

    path1 = SSHPath('ssh://ubuntu@localhost:2222/.')
    path1.ls()
    path1.fs.ls('/')
