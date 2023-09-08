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
    __protocol__ = NotImplemented

    @classmethod
    def _new_fs(cls, **kwargs):
        """
        Create a new filesystem instance based on __protocol__
        """
        return fsspec.filesystem(cls.__protocol__, **kwargs)

    @classmethod
    def _current_fs(cls, **kwargs):
        """
        The "default" FileSystem object.  Get the most recent filesystem with
        this protocol, or create a new one with defaults.

        Returns:
            AbstractFileSystem
        """
        fs_cls = fsspec.get_filesystem_class(cls.__protocol__)
        return fs_cls.current()

    def __new__(cls, path, *, fs=None):
        # Note: the value of the string is set in the __new__ method because
        # strings are immutable. So we dont need to call super or anything.
        # The first argument will be the value of the string.
        if fs is None:
            # Lazy creation of a new fs
            fs = cls._current_fs()
        self = str.__new__(cls, path)
        self.fs = fs
        return self

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
        if path.startswith('s3:'):
            self = S3Path(path)
        # elif path.startswith('/vsis3/'):
        #     # convert gdal virtual filesystems to s3 paths?
        #     self = S3Path('s3://' + path[7:])
        else:
            self = LocalPath(path)
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
            >>> from watch.utils import util_fsspec
            >>> dpath = util_fsspec.LocalPath.appdir('watch/fsspec/tests/open').ensuredir()
            >>> fpath = dpath / 'file.txt'
            >>> file = fpath.open(mode='w')
            >>> file.write('hello world')
            >>> file.close()
            >>> assert fpath.read_text() == fpath.open('r').read()
        """
        return self.fs.open(self, mode=mode, block_size=block_size,
                            cache_options=cache_options,
                            compression=compression)

    def ls(self, detail=False, **kwargs):
        return self.fs.ls(self, detail=detail, **kwargs)

    def touch(self, truncate=False, **kwargs):
        self.fs.touch(self, truncate=truncate, **kwargs)

    def move(self, path1, path2, recursive='auto', maxdepth=None, **kwargs):
        if recursive == 'auto':
            recursive = self.is_dir()
        self.fs.move(self, path2, recursive=recursive, maxdepth=maxdepth,
                     **kwargs)

    def delete(self, recursive='auto', maxdepth=True):
        """
        Deletes this file or this directory (and all of its contents)

        Unlike fs.delete, this will not error if the file doesnt exist. See
        :func:`FSPath.rm` if you want standard error-ing behavior.
        """
        if recursive == 'auto':
            recursive = self.is_dir()

        try:
            return self.fs.delete(self, recursive=recursive, maxdepth=maxdepth)
        except FileNotFoundError:
            ...

    def rm(self, recursive='auto', maxdepth=True):
        """
        Deletes this file or this directory (and all of its contents)
        """
        if recursive == 'auto':
            recursive = self.is_dir()
        return self.fs.rm(self, recursive=recursive, maxdepth=maxdepth)

    def mkdir(self, create_parents=True, **kwargs):
        """
        Note:
            does nothing on some filesystems (e.g. S3)
        """
        return self.fs.mkdir(self, create_parents=create_parents, **kwargs)

    def stat(self):
        return self.fs.stat(self)

    def is_dir(self):
        return self.fs.isdir(self)

    def is_file(self):
        return self.fs.isfile(self)

    def is_link(self):
        try:
            return self.fs.islink(self)
        except AttributeError:
            return False

    def exists(self):
        return self.fs.exists(self)

    def write_text(self, value, **kwargs):
        return self.fs.write_text(self, value, **kwargs)

    def read_text(self, **kwargs):
        return self.fs.read_text(self, **kwargs)

    def walk(self, include_protocol='auto', **kwargs):
        """
        Yields:
            Tuple[Self, List[str], List[str]] - root, dir names, file names
        """
        if include_protocol == 'auto':
            include_protocol = self.is_remote()
        if include_protocol:
            for root, dnames, fnames in self.fs.walk(self, **kwargs):
                root = self.__class__(self.fs.unstrip_protocol(root), fs=self.fs)
                yield root, dnames, fnames
        else:
            for root, dnames, fnames in self.fs.walk(self, **kwargs):
                root = self.__class__(root, fs=self.fs)
                yield root, dnames, fnames

    @property
    def parent(self):
        return self.__class__(os.path.dirname(self), fs=self.fs)

    @property
    def name(self):
        return os.path.basename(self)

    @property
    def stem(self):
        return os.path.splitext(self.name)[0]

    @property
    def suffix(self):
        return os.path.splitext(self.name)[1]

    @property
    def suffixes(self):
        return self.name.split('.')[1:]

    @property
    def parts(self):
        return self.split(self.fs.sep)

    def copy(self, dst, recursive='auto', maxdepth=None, on_error=None,
             callback=None, verbose=1, idempotent=True, overwrite=False,
             **kwargs):
        """
        Copies this file or directory to dst

        Abtracts fsspec copy / put / get.

        If dst ends with a "/", it will be assumed to be a directory, and
        target files will go within.

        Unlike fsspec, this attempts to be idempotent.

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
            https://filesystem-spec.readthedocs.io/en/latest/copying.html
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
                        return dst.fs.put(self, dst, **commonkw, callback=callback)
                    else:
                        return dst.fs.put_file(self, dst, callback=callback)
                except FileExistsError:
                    # TODO: overwrite
                    raise

            elif isinstance(dst, LocalPath):
                return self.fs.copy(self, dst, **commonkw, callback=callback)
            else:
                raise TypeError(type(dst))
        elif isinstance(self, RemotePath):
            if isinstance(dst, RemotePath):
                return self.fs.copy(self, dst, **commonkw, on_error=on_error)
            elif isinstance(dst, (LocalPath, pathlib.Path)):

                if recursive:
                    return self.fs.get(self, dst, **commonkw, callback=callback)
                else:
                    # Using put on an s3 bucket seems like it fails when
                    # directories have tight permissions, but put file
                    # seems ok. Need to ensure that we are actually just
                    # using a file in this instance.
                    return self.fs.get_file(self, dst, callback=callback)
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
        xdoctest -m watch.utils.util_fsspec LocalPath
        xdoctest watch/utils/util_fsspec.py

    Example:
        >>> from watch.utils.util_fsspec import *  # NOQA
        >>> dpath = ub.Path.appdir('watch/tests/util_fsspec/demo')
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

    To work with different S3 filesystems,

    Requirements:
        s3fs>=2023.6.0

    References:
        .. [S3FS_Docs] https://s3fs.readthedocs.io/en/latest/?badge=latest

    Example:
        >>> # xdoctest: +REQUIRES(module:s3fs)
        >>> fs = S3Path._new_fs()
    """
    __protocol__ = 's3'

    def _as_gdal_vsi(self):
        return '/vsis3/' + self[len(self.__protocol__) + 3:]


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
