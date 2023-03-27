"""
A simplified Python DVC API

TODO:
    - [ ] Replace "jobs" with "workers" to keep variables consistent across the
          project? Or keep "jobs" because that's what DVC uses?
"""
import ubelt as ub
import os
from watch.utils import util_path


class SimpleDVC(ub.NiceRepr):
    """
    A Simple DVC API

    Args:
        dvc_root (Path): path to DVC repo directory
        remote (str): dvc remote to sync to by default

    Ignore:
        >>> # xdoctest: +REQUIRES(--dvc-test)
        >>> import sys, ubelt
        >>> from watch.utils.simple_dvc import *  # NOQA
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
        return str(self.dvc_root)

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
        if path is None:
            raise Exception('no way to find dvc root')
        # Need to find it from the path
        path = ub.Path(path)
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

    def add(self, path, verbose=0):
        """
        Args:
            path (str | PathLike | Iterable[str | PathLike]):
                a single or multiple paths to add
        """
        from dvc import main as dvc_main
        paths = list(map(ub.Path, _ensure_iterable(path)))
        if len(paths) == 0:
            print('No paths to add')
            return
        if 1:
            # Handle symlinks: https://dvc.org/doc/user-guide/troubleshooting#add-symlink
            # not sure if this is safe
            dvc_root = self._ensure_root(paths).resolve()
            rel_paths = [os.fspath(p.resolve().relative_to(dvc_root)) for p in paths]
        else:
            dvc_root = self._ensure_root(paths)
            rel_paths = [os.fspath(p.relative_to(dvc_root)) for p in paths]
        with util_path.ChDir(dvc_root):
            dvc_command = ['add'] + rel_paths
            extra_args = self._verbose_extra_args(verbose)
            dvc_command = dvc_command + extra_args
            ret = dvc_main.main(dvc_command)
        if ret != 0:
            raise Exception('Failed to add files')

        has_autostage = ub.cmd('dvc config core.autostage', cwd=dvc_root, check=True)['out'].strip() == 'true'
        if not has_autostage:
            raise NotImplementedError('Need autostage to complete the git commit')

    def check_ignore(self, path, details=0, verbose=0):
        from dvc import main as dvc_main
        paths = list(map(ub.Path, _ensure_iterable(path)))
        if len(paths) == 0:
            print('No paths to add')
            return
        dvc_root = self._ensure_root(paths)
        rel_paths = [os.fspath(p.relative_to(dvc_root)) for p in paths]
        with util_path.ChDir(dvc_root):
            dvc_command = ['check-ignore'] + rel_paths
            if details:
                dvc_command += ['--details']
            extra_args = self._verbose_extra_args(verbose)
            dvc_command = dvc_command + extra_args
            ret = dvc_main.main(dvc_command)
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
        from dvc import main as dvc_main
        paths = list(map(ub.Path, _ensure_iterable(path)))
        if len(paths) == 0:
            print('No paths to push')
            return
        remote = self._ensure_remote(remote)
        dvc_root = self._ensure_root(paths)
        extra_args = self._remote_extra_args(remote, recursive, jobs, verbose)
        with util_path.ChDir(dvc_root):
            dvc_command = ['push'] + extra_args + [str(p.relative_to(dvc_root)) for p in paths]
            dvc_main.main(dvc_command)

    def pull(self, path, remote=None, recursive=False, jobs=None, verbose=0):
        from dvc import main as dvc_main
        paths = list(map(ub.Path, _ensure_iterable(path)))
        if len(paths) == 0:
            print('No paths to pull')
            return
        remote = self._ensure_remote(remote)
        dvc_root = self._ensure_root(paths)
        extra_args = self._remote_extra_args(remote, recursive, jobs, verbose)
        with util_path.ChDir(dvc_root):
            dvc_command = ['pull'] + extra_args + [str(p.relative_to(dvc_root)) for p in paths]
            dvc_main.main(dvc_command)

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
            dvc_root = self._ensure_root(paths)
            def _find_sidecar(path):
                first_cand = path.augment(stem=path.name, ext='.dvc')
                if first_cand.exists():
                    return first_cand
                rel_path = path.relative_to(dvc_root)
                rel_parts = rel_path.parts
                for i in reversed(range(len(rel_parts))):
                    parts = rel_parts[0:i]
                    cand_dat = dvc_root.joinpath(*parts)
                    cand_dvc = cand_dat.augment(stem=cand_dat.name, ext='.dvc')
                    if cand_dvc.exists():
                        return cand_dvc

            to_pull = [_find_sidecar(path) for path in missing_data]
            missing_sidecar = [dvc_fpath for dvc_fpath in to_pull if not dvc_fpath.exists()]

            if missing_sidecar:
                if len(missing_sidecar) < 10:
                    print(f'missing_sidecar={missing_sidecar}')
                raise Exception(f'There were {len(missing_sidecar)} / {len(paths)} missing sidecar files')

            if to_pull:
                self.pull(to_pull, remote=remote)

    def unprotect(self, path):
        from dvc import main as dvc_main
        paths = list(map(ub.Path, _ensure_iterable(path)))
        if len(paths) == 0:
            print('No paths to unprotect')
            return
        dvc_root = self._ensure_root(paths)
        rel_paths = [os.fspath(p.relative_to(dvc_root)) for p in paths]
        with util_path.ChDir(dvc_root):
            dvc_command = ['unprotect'] + rel_paths
            dvc_main.main(dvc_command)

    def is_tracked(self, path):
        path = ub.Path(path)
        tracker_fpath = self.find_file_tracker(path)
        if tracker_fpath is not None:
            return True
        else:
            tracker_fpath = self.find_dir_tracker(path)
            if tracker_fpath is not None:
                raise NotImplementedError

    # @classmethod
    # def find_dvc_tracking_fpath(cls, path):
    @classmethod
    def find_file_tracker(cls, path):
        assert not path.name.endswith('.dvc')
        tracker_fpath = path.augment(tail='.dvc')
        if tracker_fpath.exists():
            return tracker_fpath

    def find_dir_tracker(cls, path):
        # Find if an ancestor parent dpath is tracked
        path = ub.Path(path)
        prev = path
        dpath = path.parent
        while (not (dpath / '.dvc').exists()) and prev != dpath:
            tracker_fpath = dpath.augment(tail='.dvc')
            if tracker_fpath.exists():
                return tracker_fpath
            prev = dpath
            dpath = dpath.parent

    def read_dvc_sidecar(self, sidecar_fpath):
        from watch.utils import util_yaml
        sidecar_fpath = ub.Path(sidecar_fpath)
        data = util_yaml.yaml_loads(sidecar_fpath.read_text())
        return data

    def resolve_cache_paths(self, sidecar_fpath):
        """
        Given a .dvc file, enumerate the paths in the cache associated with it.

        Args:
            sidecar_fpath (PathLike | str): path to the .dvc file
        """
        from watch.utils import util_yaml
        sidecar_fpath = ub.Path(sidecar_fpath)
        data = util_yaml.yaml_loads(sidecar_fpath.read_text())
        for item in data['outs']:
            md5 = item['md5']
            cache_fpath = self.cache_dir / md5[0:2] / md5[2:]
            if md5.endswith('.dir') and cache_fpath.exists():
                dir_data = util_yaml.yaml_loads(cache_fpath.read_text())
                for item in dir_data:
                    file_md5 = item['md5']
                    assert not file_md5.endswith('.dir'), 'unhandled'
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


def _ensure_iterable(inputs):
    return inputs if ub.iterable(inputs) else [inputs]
