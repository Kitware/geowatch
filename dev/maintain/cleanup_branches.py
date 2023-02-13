"""
Cleanup old local branches

See Also:

    Erotemic/local:
    ~/local/git_tools/git_devbranch.py
"""
import sys
import git
import ubelt
sys.path.append(ubelt.expandpath('~/local/git_tools'))

repo = git.Repo('.')
branch_names = [p.split(' ')[-1] for p in repo.git.branch().split('\n')]
to_remove = [b for b in branch_names if b.startswith('dev/flow')]
if to_remove:
    repo.git.branch(*to_remove, '-D')
