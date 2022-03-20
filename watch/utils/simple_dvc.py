"""
A simplified Python DVC API
"""
import ubelt as ub
import os

"""
Python API to make DVC easier to work with
"""


class ChDir:
    """
    Context manager that changes the current working directory and then
    returns you to where you were.
    """
    def __init__(self, dpath):
        self.context_dpath = dpath
        self.orig_dpath = None

    def __enter__(self):
        self.orig_dpath = os.getcwd()
        os.chdir(self.context_dpath)
        return self

    def __exit__(self, a, b, c):
        os.chdir(self.orig_dpath)


class SimpleDVC():
    """
    A Simple DVC API

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

    def __init__(self, dvc_root):
        self.dvc_root = dvc_root

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

    def find_root(self, path=None):
        if self.dvc_root is not None:
            return self.dvc_root
        else:
            if path is None:
                raise Exception('no way to find dvc root')
            # Need to find it from the path
            max_parts = len(path.parts)
            curr = path
            found = None
            for _ in range(max_parts):
                cand = curr / '.dvc'
                if cand.exists():
                    found = curr
                    break
                curr = curr.parent
            return found

    def add(self, paths):
        import dvc.main
        if not ub.iterable(paths):
            paths = [paths]
        dvc_root = self.find_root(paths[0])
        rel_paths = [os.fspath(p.relative_to(dvc_root)) for p in paths]
        with ChDir(dvc_root):
            dvc_command = ['add'] + rel_paths
            dvc.main.main(dvc_command)

        has_autostage = ub.cmd('dvc config core.autostage', cwd=dvc_root, check=True)['out'].strip() == 'true'
        if not has_autostage:
            raise NotImplementedError('Need autostage to complete the git commit')

    def push(self, path, remote=None):
        import dvc.main
        if not ub.iterable(path):
            paths = [path]
        else:
            paths = path
        dvc_root = self.find_root(paths[0])
        extra_args = []
        if remote:
            extra_args += ['-r', remote]

        with ChDir(dvc_root):
            dvc_command = ['push', '--recursive'] + [str(p.relative_to(dvc_root)) for p in paths] + extra_args
            dvc.main.main(dvc_command)
