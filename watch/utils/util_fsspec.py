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
        Not all of the fsspec / pathlib operations are currently implemented
    """

    fs: fsspec.spec.AbstractFileSystem = NotImplemented

    def __init__(self, path):
        # I don't know why this works without a super().__init__(path)
        self.path = path

    @classmethod
    def coerce(cls, path):
        """
        Determine which backend to use automatically

        Example:
            >>> path2 = FSPath.coerce('/local/path')
            >>> assert path2.is_local()
            >>> # xdoctest: +REQUIRES(module:s3fs)
            >>> path1 = FSPath.coerce('s3://demo_bucket')
            >>> assert path1.is_remote()
        """
        if path.startswith('s3:'):
            self = S3Path(path)
        else:
            self = LocalPath(path)
        return self

    def relative_to(self, other):
        return self.__class__(os.path.relpath(self, other))

    def is_remote(self):
        return isinstance(self, RemotePath)

    def is_local(self):
        return isinstance(self, LocalPath)

    def ls(self, detail=False, **kwargs):
        return self.fs.ls(self, detail=detail, **kwargs)

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
        return self.fs.mkdir(self, create_parents=create_parents, **kwargs)

    def info(self):
        return self.fs.info(self)

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
                root = self.__class__(self.fs.unstrip_protocol(root))
                yield root, dnames, fnames
        else:
            for root, dnames, fnames in self.fs.walk(self, **kwargs):
                root = self.__class__(root)
                yield root, dnames, fnames

    @property
    def parent(self):
        return self.__class__(os.path.dirname(self))

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
             callback=None, verbose=1, **kwargs):
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

        Note:
            There are different functions depending on if we are going from
            remote->remote (copy), local->remote (put), or remote->local (get)

        References:
            https://filesystem-spec.readthedocs.io/en/latest/copying.html
        """
        if recursive == 'auto':
            recursive = self.is_dir()

        if callback is None:
            callback = NOOP_CALLBACK

        commonkw = {
            'recursive': recursive,
            'maxdepth': maxdepth,
            **kwargs,
        }

        idempotent = True

        if verbose:
            print(f'Copy {self} -> {dst}')

        # HANDLE SPECIAL CASE WHERE FSSPEC IS NOT IDEMPOTENT
        if idempotent:
            if recursive and dst.exists():
                dst = dst.parent + '/'

        if isinstance(self, LocalPath):
            if isinstance(dst, RemotePath):
                return dst.fs.put(self, dst, **commonkw, callback=callback)
            elif isinstance(dst, LocalPath):
                return self.fs.copy(self, dst, **commonkw, callback=callback)
            else:
                raise TypeError(type(dst))
        elif isinstance(self, RemotePath):
            if isinstance(dst, RemotePath):
                return self.fs.copy(self, dst, **commonkw, on_error=on_error)
            elif isinstance(dst, LocalPath):
                return self.fs.get(self, dst, **commonkw, callback=callback)
            else:
                raise TypeError(type(dst))
        else:
            raise TypeError(type(self))

    def joinpath(self, *others):
        return self.__class__(join(self, *others))

    def __truediv__(self, other):
        return self.__class__(join(self, other))

    def __add__(self, other):
        """
        Returns a new string starting with this fspath representation.
        """
        return self.__class__(str(self) + other)

    def __radd__(self, other):
        """
        Returns a new string ending with this fspath representation.
        """
        return self.__class__(other + str(self))

    def tree(self, max_files=100, dirblocklist=None, show_nfiles='auto',
             return_text=False, return_tree=True, pathstyle='name',
             max_depth=None, with_type=False, abs_root_label=True,
             colors=not ub.NO_COLOR):
        """
        Filesystem tree representation

        Like the unix util tree, but allow writing numbers of files per directory
        when given -d option

        Args:
            cwd (None | str | PathLike) : directory to print
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
                    scolor = '[blue]'
                    tcolor = '[/blue]'
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
    """
    fs = fsspec.filesystem('file')  # type: fsspec.implementations.local.LocalFileSystem

    def ensuredir(self, mode=0o0777):
        pathlib.Path(self).mkdir(mode=mode, parents=True, exist_ok=True)
        return self


class RemotePath(FSPath):
    """
    Abstract implementation for all remote filesystems
    """


class S3Path(RemotePath):
    """
    The specific S3 remote filesystem.
    """
    fs = fsspec.filesystem('s3')  # type: s3fs.core.S3FileSystem
