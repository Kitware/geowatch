# import pycodestyle
import ubelt as ub

VERBOSE = 3


def exec_flake8(dpaths, select=None, ignore=None, max_line_length=79):
    if VERBOSE > 1:
        print('ignore = {!r}'.format(ignore))
    args_list = ['--max-line-length', f'{max_line_length:d}']
    if select is not None:
        args_list += ['--select=' + ','.join(select)]
    if ignore is not None:
        args_list += ['--ignore=' + ','.join(ignore)]
    args_list += ['--statistics']
    info = ub.cmd(['flake8'] + args_list + dpaths, verbose=VERBOSE, check=False)
    if info['ret'] not in {0, 1}:
        raise Exception(ub.repr2(ub.dict_diff(info, ['out'])))
    return info['ret']


def exec_autopep8(dpaths, autofix, mode='diff'):
    if VERBOSE > 1:
        print('autofix = {!r}'.format(autofix))
    args_list = ['--select', ','.join(autofix), '--recursive']
    if mode == 'diff':
        args_list += ['--diff']
    elif mode == 'apply':
        args_list += ['--in-place']
    else:
        raise AssertionError(mode)
    info = ub.cmd(['autopep8'] + args_list + dpaths, verbose=VERBOSE, check=False)
    if info['ret'] not in {0, 1}:
        raise Exception(ub.repr2(ub.dict_diff(info, ['out'])))
    return info['ret']


def custom_lint(dpath: str, mode=False):
    """
    Runs our custom "watch" linting rules on a specific directory and
    optionally "fixes" them.

    Args:
        dpath (str|list):
            the path or paths to lint

        mode (bool, default="lint"):
            Options and effects are:
                * "show": display linting results
                * "diff": show the autopep8 diff that would autofix some errors
                * "apply": apply the autopep8 diff that would autofix some errors

    Ignore:
        dpath = ub.expandpath('~/code/watch/watch')
    """
    dpaths = [d] if isinstance(d := dpath, str) else d
    ignore = {
        'E123': 'closing braket indentation',
        'E126': 'continuation line hanging-indent',
        'E127': 'continuation line over-indented for visual indent',
        'E201': 'whitespace after "("',
        'E202': 'whitespace before "]"',
        'E203': 'whitespace before ", "',
        'E221': 'multiple spaces before operator  (TODO: I wish I could make an exception for the equals operator. Is there a way to do this?)',
        'E222': 'multiple spaces after operator',
        'E241': 'multiple spaces after ,',
        'E265': 'block comment should start with "# "',
        'E271': 'multiple spaces after keyword',
        'E272': 'multiple spaces before keyword',
        'E301': 'expected 1 blank line, found 0',
        'E305': 'expected 1 blank line after class / func',
        'E306': 'expected 1 blank line before func',
        #'E402': 'module import not at top',
        'E501': 'line length > 79',
        'W602': 'Old reraise syntax',
        'E266': 'too many leading # for block comment',
        'N801': 'function name should be lowercase [N806]',
        'N802': 'function name should be lowercase [N806]',
        'N803': 'argument should be lowercase [N806]',
        'N805': 'first argument of a method should be named "self"',
        'N806': 'variable in function should be lowercase [N806]',
        'N811': 'constant name imported as non constant',
        'N813': 'camel case',
        'W503': 'line break before binary operator',
        'W504': 'line break after binary operator',

        'I201': 'Newline between Third party import groups',
        'I100': 'Wrong import order',

        'E26 ': 'Fix spacing after comment hash for inline comments.',
        # 'E265': 'Fix spacing after comment hash for block comments.',
        # 'E266': 'Fix too many leading # for block comments.',
    }

    modifiers = {
        'whitespace': 1,
        'newlines': 0,
        'warnings': 1,
    }
    autofix = {}
    if modifiers['whitespace']:
        autofix.update({
            'E225': 'Fix missing whitespace around operator.',
            'E226': 'Fix missing whitespace around arithmetic operator.',
            'E227': 'Fix missing whitespace around bitwise/shift operator.',
            'E228': 'Fix missing whitespace around modulo operator.',

            # 'E231': 'Add missing whitespace.',

            'E241': 'Fix extraneous whitespace around keywords.',
            'E242': 'Remove extraneous whitespace around operator.',
            'E251': 'Remove whitespace around parameter "=" sign.',
            'E252': 'Missing whitespace around parameter equals.',

            # 'E26 ': 'Fix spacing after comment hash for inline comments.',
            # 'E265': 'Fix spacing after comment hash for block comments.',
            # 'E266': 'Fix too many leading # for block comments.',

            'E27' : 'Fix extraneous whitespace around keywords.',
        })
    if modifiers['newlines']:
        autofix.update({
            'E301': 'Add missing blank line.',
            'E302': 'Add missing 2 blank lines.',
            'E303': 'Remove extra blank lines.',
            'E304': 'Remove blank line following function decorator.',
            'E305': 'Expected 2 blank lines after end of function or class.',
            'E306': 'Expected 1 blank line before a nested definition.',
        })
    if modifiers['warnings']:
        autofix.update({
            'W291': 'Remove trailing whitespace.',
            'W292': 'Add a single newline at the end of the file.',
            'W293': 'Remove trailing whitespace on blank line.',
            'W391': 'Remove trailing blank lines.',
        })

    autofix = sorted(autofix)
    ignore = sorted(ignore)
    select = autofix

    if VERBOSE > 1:
        print('mode = {!r}'.format(mode))
    if mode == 'diff':
        return exec_autopep8(dpaths, autofix)
    elif mode == 'apply':
        return exec_autopep8(dpaths, autofix)
    elif mode == 'lint':
        max_line_length = 79
        return exec_flake8(dpaths, select, ignore, max_line_length)
    else:
        raise KeyError(mode)


if __name__ == '__main__':
    """
    CommandLine:
        python ~/code/watch/dev/lint.py --help

        cd $HOME/code/watch

        python ~/code/watch/dev/lint.py watch --mode=lint
        python ~/code/watch/dev/lint.py watch --mode=diff
        python ~/code/watch/dev/lint.py watch --mode=apply

        python ~/code/watch/dev/lint.py [watch,atk]
    """
    import fire
    import sys
    sys.exit(fire.Fire(custom_lint))
