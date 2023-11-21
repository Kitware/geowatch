#!/usr/bin/env python3
"""
A simplified Python DVC API
"""
import ubelt as ub
import os
from kwutil import util_path
from kwutil.util_yaml import Yaml


def __test_simple_dvc():
    """
    Builds a medium complexity dvc repo, todo:
        implement some tests
    """
    import ubelt as ub
    test_root = ub.Path.appdir('simpledvc', 'tests', 'basic')
    dvc_root = test_root / 'repo'
    dvc_root.delete()
    SimpleDVC.init(dvc_root, no_scm=True)

    dvc = SimpleDVC.coerce(dvc_root)

    # Build basic data
    (dvc_root / 'test-set1').ensuredir()
    assets_dpath = (dvc_root / 'test-set1/assets').ensuredir()
    for idx in range(1, 21):
        fpath = assets_dpath / f'asset_{idx:03d}.data'
        fpath.write_text(str(idx) * 100)
    manifest_fpath = (dvc_root / 'test-set1/manifest.txt')
    manifest_fpath.write_text('pretend-data')

    root_fpath = dvc_root / 'root_file'
    root_fpath.write_text('----' * 100)

    root_dpath = dvc_root / 'root_dir'

    # Use networkx to make a random complex directory structure
    import networkx as nx
    graph = nx.erdos_renyi_graph(30, p=0.2, directed=True)
    tree = nx.minimum_spanning_arborescence(graph)
    nx.write_network_text(tree)
    sources = [n for n in tree.nodes if not tree.pred[n]]
    sinks = [n for n in tree.nodes if not tree.succ[n]]

    node_paths = []
    for t in sinks:
        for s in sources:
            paths = list(nx.all_simple_edge_paths(tree, s, t))
            if paths:
                node_path = [u for (u, v) in paths[0]] + [t]
                node_paths.append(node_path)

    for node_path in node_paths:
        rel_fpath = ub.Path(*[f'dir_{n}' for n in node_path[0:-1]]) / ('file_' + str(node_path[-1]) + '.data')
        fpath = root_dpath / rel_fpath
        fpath.parent.ensuredir()
        fpath.write_text(str(node_path))

    dvc.add(root_dpath)
    dvc.add(root_fpath)
    dvc.add(manifest_fpath)
    dvc.add(assets_dpath)

    # xdev.tree_repr(dvc_root)


class SimpleDVC(ub.NiceRepr):
    """
    A Simple DVC API

    Args:
        dvc_root (Path): path to DVC repo directory
        remote (str): dvc remote to sync to by default

    Ignore:
        >>> # xdoctest: +REQUIRES(--dvc-test)
        >>> import sys, ubelt
        >>> from geowatch.utils.simple_dvc import *  # NOQA
        >>> dvc_dpath = SimpleDVC.demo_dpath(reset=0)
        >>> self = SimpleDVC(dvc_dpath)
        >>> a_file_fpath = dvc_dpath / 'a_file.txt'
        >>> if not a_file_fpath.exists():
        >>>     a_file_fpath.write_text('hello')
        >>> self.add(a_file_fpath)

    """

    def __init__(self, dvc_root=None, remote=None):
        self.dvc_root = dvc_root
        self.remote = remote

    def __nice__(self):
        return f'dvc_root={self.dvc_root}'

    @classmethod
    def init(cls, dpath, no_scm=False, force=False, verbose=0):
        """
        Initialize a DVC repo in a path
        """
        dpath = ub.Path(dpath.ensuredir())
        args = ['dvc', 'init']
        if verbose:
            args += ['--verbose']
        if force:
            args += ['--force']
        if no_scm:
            args += ['--no-scm']
        ub.cmd(args, cwd=dpath, verbose=3, check=True)
        self = cls(dpath)
        return self

    @ub.memoize_property
    def cache_dir(self):
        info = ub.cmd('dvc cache dir', cwd=self.dvc_root, check=True)
        cache_dpath = ub.Path(info['out'].strip())
        return cache_dpath

    @classmethod
    def demo_dpath(cls, reset=False):
        dvc_dpath = ub.Path.appdir('simple_dvc/test/test_dvc_repo')
        if reset:
            dvc_dpath.delete()
        if not dvc_dpath.exists():
            dvc_dpath.ensuredir()
            verbose = 2
            # Init empty git repo
            ub.cmd('git init --quiet', cwd=dvc_dpath, verbose=verbose)
            ub.cmd('git config --local receive.denyCurrentBranch "warn"', cwd=dvc_dpath, verbose=verbose)
            # Init empty dvc repo
            ub.cmd('dvc init --quiet', cwd=dvc_dpath, verbose=verbose)
            ub.cmd('dvc config core.autostage true', cwd=dvc_dpath, verbose=verbose)
            ub.cmd('dvc config cache.type "symlink,hardlink,copy"', cwd=dvc_dpath, verbose=verbose)
            ub.cmd('dvc config cache.shared group', cwd=dvc_dpath, verbose=verbose)
            ub.cmd('dvc config cache.protected true', cwd=dvc_dpath, verbose=verbose)
        return dvc_dpath

    @classmethod
    def coerce(cls, dvc_path, **kw):
        """
        Given a path inside DVC, finds the root.
        """
        dvc_root = cls.find_root(dvc_path)
        return cls(dvc_root, **kw)

    @classmethod
    def find_root(cls, path=None):
        """
        Given a path, search its ancestors to find the root of a dvc repo.

        Returns:
            Path | None
        """
        if path is None:
            raise Exception('no way to find dvc root')
        # Need to find it from the path
        path = ub.Path(path).resolve()
        max_parts = len(path.parts)
        curr = path
        found = None
        for _ in range(max_parts + 1):
            cand = curr / '.dvc'
            if cand.exists():
                found = curr
                break
            curr = curr.parent
        return found

    def _ensure_root(self, paths):
        if self.dvc_root is None:
            self.dvc_root = self.find_root(paths[0])
            print('found new self.dvc_root = {!r}'.format(self.dvc_root))
        return self.dvc_root

    def _ensure_remote(self, remote):
        if remote is None:
            remote = self.remote
        return remote

    def _resolve_root_and_relative_paths(self, paths):
        # try:
        #     dvc_root = self._ensure_root(paths)
        #     rel_paths = [os.fspath(p.relative_to(dvc_root)) for p in paths]
        # except Exception as ex:
        #     print(f'ex={ex}')
        # Handle symlinks: https://dvc.org/doc/user-guide/troubleshooting#add-symlink
        # not sure if this is safe
        dvc_root = self._ensure_root(paths)
        if dvc_root is None:
            raise Exception('unable to find a DVC root')
        dvc_root = dvc_root.resolve()
        # Note: this could resolve the symlink to the dvc cache which we dont want
        # rel_paths = [os.fspath(p.resolve().relative_to(dvc_root)) for p in paths]
        # Fixed version?
        parent_resolved = [p.parent.resolve() / p.name for p in paths]
        rel_paths = [os.fspath(p.relative_to(dvc_root)) for p in parent_resolved]

        return dvc_root, rel_paths

    def add(self, path, verbose=0):
        """
        Args:
            path (str | PathLike | Iterable[str | PathLike]):
                a single or multiple paths to add
        """
        dvc_root, rel_paths = self._dvc_path_op('add', path, verbose)
        has_autostage = ub.cmd('dvc config core.autostage', cwd=dvc_root, check=True)['out'].strip() == 'true'
        if not has_autostage:
            print('warning: Need autostage to complete the git commit')
            # raise NotImplementedError('Need autostage to complete the git commit')

    def pathsremove(self, path, verbose=0):
        """
        Args:
            path (str | PathLike | Iterable[str | PathLike]):
                a single or multiple paths to add
        """
        self._dvc_path_op('remove', path, verbose)

    def _dvc_path_op(self, op, path, verbose=0):
        """
        Args:
            path (str | PathLike | Iterable[str | PathLike]):
                a single or multiple paths to add
        """
        dvc_main = _import_dvc_main()
        paths = list(map(ub.Path, _ensure_iterable(path)))
        if len(paths) == 0:
            print('No paths to add')
            return
        dvc_root, rel_paths = self._resolve_root_and_relative_paths(paths)
        with util_path.ChDir(dvc_root):
            dvc_command = [op] + rel_paths
            extra_args = self._verbose_extra_args(verbose)
            dvc_command = dvc_command + extra_args
            ret = dvc_main(dvc_command)
        if ret != 0:
            raise Exception(f'Failed to {op} files')
        return dvc_root, rel_paths

    def check_ignore(self, path, details=0, verbose=0):
        dvc_main = _import_dvc_main()
        paths = list(map(ub.Path, _ensure_iterable(path)))
        if len(paths) == 0:
            print('No paths to add')
            return
        dvc_root, rel_paths = self._resolve_root_and_relative_paths(paths)
        with util_path.ChDir(dvc_root):
            dvc_command = ['check-ignore'] + rel_paths
            if details:
                dvc_command += ['--details']
            extra_args = self._verbose_extra_args(verbose)
            dvc_command = dvc_command + extra_args
            ret = dvc_main(dvc_command)
        if ret != 0:
            raise Exception('Failed check-ignore')

    def git_pull(self):
        ub.cmd('git pull', verbose=3, check=True, cwd=self.dvc_root)

    def git_push(self):
        ub.cmd('git push', verbose=3, check=True, cwd=self.dvc_root)

    def git_commit(self, message):
        ub.cmd(f'git commit -m "{message}"', verbose=3, check=True, cwd=self.dvc_root)

    def git_commitpush(self, message='', pull_on_fail=True):
        """
        TODO: better name here?
        """
        # dangerous?
        try:
            self.git_commit(message)
        except Exception as e:
            ex = e
            if 'nothing added to commit' not in ex.output:
                raise
        else:
            try:
                self.git_push()
            except Exception:
                if pull_on_fail:
                    print('Initial push failed, will pull and then try once more')
                    self.git_pull()
                    self.git_push()
                else:
                    raise

    def _verbose_extra_args(self, verbose):
        extra_args = []
        if verbose:
            verbose = max(min(3, verbose), 1)
            extra_args += ['-' + 'v' * verbose]
        return extra_args

    def _remote_extra_args(self, remote, recursive, jobs, verbose):
        extra_args = self._verbose_extra_args(verbose)
        if remote:
            extra_args += ['-r', remote]
        if jobs is not None:
            extra_args += ['--jobs', str(jobs)]
        if recursive:
            extra_args += ['--recursive']
        return extra_args

    def push(self, path, remote=None, recursive=False, jobs=None, verbose=0):
        """
        Push the content tracked by .dvc files to remote storage.

        Args:
            path (Path | List[Path):
                one or more file paths that should have an associated .dvc
                sidecar file or if recursive is true, a directory containing
                multiple tracked files.

            remote (str):
                the name of the remote registered in the .dvc/config to push to

            recursive (bool):
                if True, then items in ``path`` can be a directory.

            jobs (int): number of parallel workers
        """
        dvc_main = _import_dvc_main()
        paths = list(map(ub.Path, _ensure_iterable(path)))
        if len(paths) == 0:
            print('No paths to push')
            return
        remote = self._ensure_remote(remote)
        dvc_root, rel_paths = self._resolve_root_and_relative_paths(paths)
        extra_args = self._remote_extra_args(remote, recursive, jobs, verbose)
        with util_path.ChDir(dvc_root):
            dvc_command = ['push'] + extra_args + [str(p) for p in rel_paths]
            dvc_main(dvc_command)

    def pull(self, path, remote=None, recursive=False, jobs=None, verbose=0):
        dvc_main = _import_dvc_main()
        paths = list(map(ub.Path, _ensure_iterable(path)))
        if len(paths) == 0:
            print('No paths to pull')
            return
        remote = self._ensure_remote(remote)
        dvc_root, rel_paths = self._resolve_root_and_relative_paths(paths)
        extra_args = self._remote_extra_args(remote, recursive, jobs, verbose)
        with util_path.ChDir(dvc_root):
            dvc_command = ['pull'] + extra_args + [str(p) for p in rel_paths]
            dvc_main(dvc_command)

    def request(self, path, remote=None):
        """
        Requests to ensure that a specific file from DVC exists.

        Any files that do not exist, check to see if there is an associated
        .dvc sidecar file. If any sidecar files are missing, an error is
        thrown.  Otherwise we attempt to pull the missing files.

        Args:
            path (Path | List[Path):
                one or more file paths that should have an associated .dvc
                sidecar file.
        """
        paths = list(map(ub.Path, _ensure_iterable(path)))
        missing_data = [path for path in paths if not path.exists()]

        if missing_data:
            dvc_root, rel_paths = self._resolve_root_and_relative_paths(missing_data)

            def _find_sidecar(rel_path):
                rel_path = ub.Path(rel_path)
                first_cand = dvc_root / rel_path.augment(stem=rel_path.name, ext='.dvc')
                if first_cand.exists():
                    return first_cand
                rel_parts = rel_path.parts
                for i in reversed(range(len(rel_parts))):
                    parts = rel_parts[0:i]
                    cand_dat = dvc_root.joinpath(*parts)
                    cand_dvc = cand_dat.augment(stem=cand_dat.name, ext='.dvc')
                    if cand_dvc.exists():
                        return cand_dvc
                raise IOError(f'Could not find sidecar for: {rel_path=} in {dvc_root=}. Wrong path, or need to git pull?')

            to_pull = [_find_sidecar(rel_path) for rel_path in rel_paths]
            missing_sidecar = [dvc_fpath for dvc_fpath in to_pull if not dvc_fpath.exists()]

            if missing_sidecar:
                if len(missing_sidecar) < 10:
                    print(f'missing_sidecar={missing_sidecar}')
                raise Exception(f'There were {len(missing_sidecar)} / {len(paths)} missing sidecar files')

            if to_pull:
                self.pull(to_pull, remote=remote)

    def unprotect(self, path):
        dvc_main = _import_dvc_main()
        paths = list(map(ub.Path, _ensure_iterable(path)))
        if len(paths) == 0:
            print('No paths to unprotect')
            return
        dvc_root, rel_paths = self._resolve_root_and_relative_paths(paths)
        with util_path.ChDir(dvc_root):
            dvc_command = ['unprotect'] + rel_paths
            dvc_main(dvc_command)

    def is_tracked(self, path):
        path = ub.Path(path)
        tracker_fpath = self.find_file_tracker(path)
        if tracker_fpath is not None:
            return True
        else:
            tracker_fpath = self.find_dir_tracker(path)
            if tracker_fpath is not None:
                raise NotImplementedError

    @classmethod
    def find_file_tracker(cls, path):
        assert not path.name.endswith('.dvc')
        tracker_fpath = path.augment(tail='.dvc')
        if tracker_fpath.exists():
            return tracker_fpath

    def find_dir_tracker(cls, path):
        # Find if an ancestor parent dpath is tracked
        path = ub.Path(path).absolute()
        prev = path
        dpath = path.parent
        while (not (dpath / '.dvc').exists()) and prev != dpath:
            tracker_fpath = dpath.augment(tail='.dvc')
            if tracker_fpath.exists():
                return tracker_fpath
            prev = dpath
            dpath = dpath.parent
        tracker_fpath = dpath.augment(tail='.dvc')
        if tracker_fpath.exists():
            return tracker_fpath

    def read_dvc_sidecar(self, sidecar_fpath):
        sidecar_fpath = ub.Path(sidecar_fpath)
        data = Yaml.loads(sidecar_fpath.read_text())
        return data

    def resolve_cache_paths(self, sidecar_fpath):
        """
        Given a .dvc file, enumerate the paths in the cache associated with it.

        Args:
            sidecar_fpath (PathLike | str): path to the .dvc file
        """
        sidecar_fpath = ub.Path(sidecar_fpath)
        data = Yaml.loads(sidecar_fpath.read_text())

        dvc3_cache_base = (self.cache_dir / 'files/md5')
        try_dvc3 = dvc3_cache_base.exists()

        # TODO: dvc 3.0 added new hashes! Yay! But we have to support this.
        for item in data['outs']:
            md5 = item['md5']

            if try_dvc3:
                cache_fpath = self.cache_dir / 'files' / 'md5' / md5[0:2] / md5[2:]
                if not cache_fpath.exists():
                    cache_fpath = self.cache_dir / md5[0:2] / md5[2:]
            else:
                cache_fpath = self.cache_dir / md5[0:2] / md5[2:]
                if not cache_fpath.exists():
                    cache_fpath = self.cache_dir / 'files' / 'md5' / md5[0:2] / md5[2:]

            if md5.endswith('.dir') and cache_fpath.exists():
                dir_data = Yaml.loads(cache_fpath.read_text())
                for item in dir_data:
                    file_md5 = item['md5']
                    assert not file_md5.endswith('.dir'), 'unhandled'
                    if try_dvc3:
                        file_cache_fpath = self.cache_dir / 'files' / 'md5' / file_md5[0:2] / file_md5[2:]
                    else:
                        file_cache_fpath = self.cache_dir / file_md5[0:2] / file_md5[2:]

                    yield file_cache_fpath
            yield cache_fpath

    def find_sidecar_paths(self, dpath):
        """
        Args:
            dpath (Path | str): directory in dvc repo to search

        Yields:
            ub.Path: existing dvc sidecar files
        """
        # TODO: handle .dvcignore
        dpath = ub.Path(dpath)
        for r, ds, fs in dpath.walk():
            for f in fs:
                if f.endswith('.dvc'):
                    yield r / f

    def resolve_sidecar(self, path):
        """
        Given a path in a DVC repo, resolve it to a sidecar file that it
        corresponds to. If the input is a .dvc file return it.

        If it is inside a directory that corresponds to a dvc repo, search for
        that.

        Args:
            path (Path | str): directory or file in dvc repo to search

        Yields:
            ub.Path: existing dvc sidecar files
        """
        # TODO: handle .dvcignore
        path = ub.Path(path).absolute()
        if path.name.endswith('.dvc'):
            return path
        elif path.augment(tail='.dvc').exists():
            return path.augment(tail='.dvc')
        else:
            return self.find_dir_tracker(path)


def _ensure_iterable(inputs):
    return inputs if ub.iterable(inputs) else [inputs]


####
# SimpleDVC CLI Stuff (should move to a new file)
import scriptconfig as scfg  # NOQA


class SimpleDVC_CLI(scfg.ModalCLI):
    """
    A DVC CLI That uses our simplified (and more permissive) interface.

    The main advantage is that you can run these commands outside a DVC repo as
    long as you point to a valid in-repo path.
    """

    class Add(scfg.DataConfig):
        """
        Add data to the DVC repo.
        """
        __command__ = 'add'

        paths = scfg.Value([], nargs='+', position=1, help='Input files / directories to add')

        @classmethod
        def main(cls, cmdline=1, **kwargs):
            config = cls.cli(cmdline=cmdline, data=kwargs, strict=True)
            dvc = SimpleDVC()
            dvc.add(config.paths)

    class Request(scfg.DataConfig):
        """
        Pull data if the requested file doesn't exist.
        """
        __command__ = 'request'

        paths = scfg.Value([], nargs='+', position=1, help='Data to attempt to pull')
        remote = scfg.Value(None, short_alias=['r'], help='remote to pull from if needed')

        @classmethod
        def main(cls, cmdline=1, **kwargs):
            config = cls.cli(cmdline=cmdline, data=kwargs, strict=True)
            dvc = SimpleDVC()
            dvc.request(config.paths)

    class CacheDir(scfg.DataConfig):
        """
        Print the cache directory
        """
        __command__ = 'cache_dir'

        dvc_root = scfg.Value('.', position=1, help='get the cache path for this DVC repo')

        @classmethod
        def main(cls, cmdline=1, **kwargs):
            config = cls.cli(cmdline=cmdline, data=kwargs, strict=True)
            dvc = SimpleDVC(dvc_root=config.dvc_root)
            print(dvc.cache_dir)


def _import_dvc_main():
    try:
        from dvc import main as dvc_main_mod
        dvc_main = dvc_main_mod.main
    except (ImportError, ModuleNotFoundError):
        from dvc.cli import main as dvc_main
    return dvc_main


if __name__ == '__main__':
    """

    CommandLine:
        python -m geowatch.utils.simple_dvc --help
        python -m geowatch.utils.simple_dvc request --help
        python -m geowatch.utils.simple_dvc cache_dir
    """
    SimpleDVC_CLI.main()
