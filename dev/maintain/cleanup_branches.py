"""
Cleanup old local branches

See Also:

    Erotemic/local:
    ~/local/git_tools/git_devbranch.py
"""
from git_devbranch import *  # NOQA
import git
repo = git.Repo('.')
branch_names = [p.split(' ')[-1] for p in repo.git.branch().split('\n')]
to_remove = [b for b in branch_names if b.startswith('dev/flow')]
repo.git.branch(*to_remove, '-D')
