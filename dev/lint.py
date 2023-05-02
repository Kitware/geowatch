# import pycodestyle
import ubelt as ub
import os

VERBOSE = 3
DEFAULT_MODE = os.environ.get('WATCH_LINT_DEFAULT_MODE', 'lint')
DEFAULT_DPATH = os.environ.get('WATCH_LINT_DEFAULT_DPATH', '.')


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

    import re
    unique_errors = []
    summary_line = re.compile(r'(\d+)\s+(\w\d+) ([^ ].*)')
    for line in info['out'].split('\n'):
        match = summary_line.match(line)
        if match:
            freq, code, desc = match.groups()
            unique_errors.append((freq, code, desc))

    if unique_errors:
        print('Unique errors:')
        for freq, code, desc in unique_errors:
            print(f'{code!r}: {desc!r},')

    if info['ret'] not in {0, 1}:
        raise Exception(ub.urepr(ub.dict_diff(info, ['out'])))
    return info['ret']


def exec_autopep8(dpaths, autofix, mode='diff'):
    if VERBOSE > 1:
        print('autofix = {!r}'.format(autofix))

    args_list = ['--select', ','.join(autofix), '--recursive']
    # args_list += ['--ignore '
    if mode == 'diff':
        args_list += ['--diff']
    elif mode in {'apply', 'fix'}:
        args_list += ['--in-place']
    else:
        raise AssertionError(mode)

    if mode == 'diff':
        args = ['autopep8'] + args_list + dpaths
        print('command = {!r}'.format(' '.join(args)))
        info = ub.cmd(args, verbose=0, check=False)
        text = info['out']
        if not ub.util_colors.NO_COLOR:
            import pygments
            import pygments.lexers
            import pygments.formatters
            import pygments.formatters.terminal
            formater = pygments.formatters.terminal.TerminalFormatter(bg='dark')
            kwargs = {}
            lexer = pygments.lexers.get_lexer_by_name('diff', **kwargs)
            new_text = pygments.highlight(text, lexer, formater)
            print(new_text)
        else:
            print(text)
    else:
        info = ub.cmd(['autopep8'] + args_list + dpaths, verbose=VERBOSE, check=False)
    if info['ret'] not in {0, 1}:
        raise Exception(ub.urepr(ub.dict_diff(info, ['out'])))
    return info['ret']


def custom_lint(mode : str = DEFAULT_MODE, dpath : str = DEFAULT_DPATH, index=None, interact=None):
    """
    Runs our custom "watch" linting rules on a specific directory and
    optionally "fixes" them.

    Args:
        dpath (str|list):
            the path or paths to lint

        mode (bool, default="lint"):
            Options and effects are:
                * "lint": display all linting results
                * "show": display autofixable linting results (sort of works)
                * "diff": show the autopep8 diff that would autofix some errors
                * "fix": apply the autopep8 diff that would autofix some errors
                * "ask": show the diff and then choose to apply or not

        index(int, default=None):
            if given only does one error at a time for autopep8

    Ignore:
        dpath = ub.expandpath('~/code/watch/watch')
    """
    d = dpath  # no walrus on the CI :(=
    dpaths = [d] if isinstance(d, str) else d
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

        'C408': ' Unncessary dict call, rewrite as literal',
        'C409': ' Unncessary tuple call, rewrite as literal',
        'C405': ' Unnecessary list literal - rewrite as a set literal.',

        'N804': ' first argument of classmethod should be named cls',
        'B006': 'Do not use mutable data structures for argument defaults.  They are created during function definition time. All calls to the function reuse this one instance of that data structure, persisting changes between them.',

        'B007': 'Loop control variable not used within the loop body. If this is intended, start the name with an underscore.',
        'B009': 'Do not call getattr with a constant attribute value, it is not any safer than normal property access.',
        'C401': 'Unnecessary generator - rewrite as a set comprehension.',
        'C414': 'Unnecessary list call within sorted().',
        'C416': 'Unnecessary list comprehension - rewrite using list().',
        'EXE001': 'Shebang is present but the file is not executable.',
        'EXE002': 'The file is executable but no shebang is present.',
        'N807': 'function name should not start and end with __',
        'N812': 'lowercase imported as non lowercase',
        'N814': 'camelcase imported as constant',
        'N817': 'camelcase imported as acronym',
        'N818': 'exception name be named with an Error suffix',
    }

    context_ignore = {
        # 'whitespace': 0,
        'whitespace': 0,
        'indentation': 0,
        'style': 0,

    }

    if context_ignore['indentation']:
        ignore.update({
            'E101': 'indentation contains mixed spaces and tabs',
            'E117': 'over-indented',
            'E121': 'continuation line under-indented for hanging indent',
            'E122': 'continuation line missing indentation or outdented',
            'E124': 'closing bracket does not match visual indentation',
            'E128': 'continuation line under-indented for visual indent',
            'E129': 'visually indented line with same indent as next logical line',
            'E131': 'continuation line unaligned for hanging indent',
        })

    if context_ignore['style']:
        ignore.update({
            'E401': 'multiple imports on one line',
            'E402': 'module level import not at top of file',
            'E701': 'multiple statements on one line (colon)',
            'E703': 'statement ends with a semicolon',
            # 'E712': "comparison to True should be 'if cond is True:' or 'if cond:'",
            # 'E713': "test for membership should be 'not in'",
            # 'E721': "do not compare types, use 'isinstance()'",
            # 'E722': "do not use bare 'except'",
            'E731': 'do not assign a lambda expression, use a def',
            'E741': "ambiguous variable name 'l'",
            'E742': "ambiguous class definition 'I'",
        })
    if context_ignore['whitespace']:
        ignore.update({
            'E252': 'missing whitespace around parameter equals',
            'E302': 'expected 2 blank lines, found 1',
            'E303': 'too many blank lines (3)',
            'E211': "whitespace before '('",
            'E225': 'missing whitespace around operator',
            'E226': 'missing whitespace around arithmetic operator',
            'E228': 'missing whitespace around modulo operator',
            'E231': "missing whitespace after ','",
            'E251': 'unexpected spaces around keyword / parameter equals',

            'W291': 'trailing whitespace',
            'W292': 'no newline at end of file',
            'W293': 'blank line contains whitespace',
            'W391': 'blank line at end of file',
            'W605': r'invalid escape sequence "\w"',
            'W191': 'indentation contains tabs',
        })

    modifiers = {
        'whitespace': 1,
        'newlines': 1,
        'warnings': 1,
    }
    autofix = {}
    if modifiers['whitespace']:
        autofix.update({
            'E225': 'Fix missing whitespace around operator.',
            'E226': 'Fix missing whitespace around arithmetic operator.',
            'E227': 'Fix missing whitespace around bitwise/shift operator.',
            'E228': 'Fix missing whitespace around modulo operator.',

            'E231': 'Add missing whitespace.',

            'E241': 'Fix extraneous whitespace around keywords.',
            'E242': 'Remove extraneous whitespace around operator.',
            'E251': 'Remove whitespace around parameter "=" sign.',
            'E252': 'Missing whitespace around parameter equals.',

            'E261': 'Fix spacing after comment hash for inline comments.',
            # 'E265': 'Fix spacing after comment hash for block comments.',
            # 'E266': 'Fix too many leading # for block comments.',

            'E271' : 'Fix extraneous whitespace around keywords.',
            'E272' : 'Fix extraneous whitespace around keywords.',
            'E273' : 'Fix extraneous whitespace around keywords.',
            'E274' : 'Fix extraneous whitespace around keywords.',

            # autopep8_ignore = {
            # 'E112': Fix under-indented comments.
            # 'E115': Fix under-indented comments.
            # 'E116': Fix over-indented comments.
            # 'E261': Fix spacing after comment hash.
            # 'E262': Fix spacing after comment hash.
            # 'E265': Format block comments.
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
    if index is not None and index is not False:
        print('index = {!r}'.format(index))
        try:
            index = int(index)
        except ValueError:
            pass
        else:
            autofix = autofix[index: index + 1]

    select = None
    ignore = sorted(ignore)

    if VERBOSE > 1:
        print('mode = {!r}'.format(mode))

    if mode in {'ask', 'interact'}:
        interact = True

    if interact:
        mode = 'diff'

    if mode in {'diff', 'apply', 'fix'}:
        if VERBOSE > 1:
            print('autofix = {!r}'.format(autofix))
        ret = exec_autopep8(dpaths, autofix, mode=mode)
        if interact:
            from rich import prompt
            if prompt.Confirm.ask('Apply this diff?'):
                mode = 'apply'
                ret = exec_autopep8(dpaths, autofix, mode=mode)
    elif mode == 'show':
        select = autofix
        if VERBOSE > 1:
            print('select = {!r}'.format(select))
        ret = exec_flake8(dpaths, select, None, 300)
    elif mode == 'lint':
        max_line_length = 79
        if VERBOSE > 1:
            print('ignore = {!r}'.format(ignore))
        ret = exec_flake8(dpaths, select, ignore, max_line_length)

        our_code_checks(dpath)
    else:
        raise KeyError(mode)

    if index:
        print('autofix = {!r}'.format(autofix))

    return ret


def our_code_checks(dpath):
    import re
    import xdev
    results = xdev.grep(re.compile(r'^ *\bxdev.embed\b'), dpath=dpath, include='*.py', exclude=['.git'])
    if len(results):
        print('ERROR: remove xdev.embed statements')
        return 1
    return 0


if __name__ == '__main__':
    """
    CommandLine:
        python ~/code/watch/dev/lint.py --help

        cd $HOME/code/watch

        # List ALL the errors
        python ~/code/watch/dev/lint.py watch --mode=lint

        # List the fixable errors
        python ~/code/watch/dev/lint.py watch --mode=show

        # Show the diff that fixes the fixable errors
        python ~/code/watch/dev/lint.py watch --mode=diff | colordiff

        # Apply the fixable diffs
        python ~/code/watch/dev/lint.py watch --mode=apply

        # First one should show nothing, but second might still show stuff
        python ~/code/watch/dev/lint.py watch --mode=show
        python ~/code/watch/dev/lint.py watch --mode=lint

        python ~/code/watch/dev/lint.py [watch,atk]

        # WORKFLOW
        python ~/code/watch/dev/lint.py watch atk --mode=diff --interact

        python ~/code/watch/dev/lint.py watch --mode=diff --interact
        python ~/code/watch/dev/lint.py atk --mode=diff --interact

        python ~/code/watch/dev/lint.py watch/tasks/rutgers_material_change_detection --mode=diff --interact
        python ~/code/watch/dev/lint.py watch/tasks/rutgers_material_seg/configs --mode=diff --interact
        python ~/code/watch/dev/lint.py watch/tasks/rutgers_material_seg/datasets --mode=diff --interact
        python ~/code/watch/dev/lint.py watch/tasks/rutgers_material_seg/experiments --mode=diff --interact
        python ~/code/watch/dev/lint.py watch/tasks/rutgers_material_seg/models --mode=diff --interact
        python ~/code/watch/dev/lint.py watch/tasks/rutgers_material_seg/scripts --mode=diff --interact
        python ~/code/watch/dev/lint.py watch/tasks/rutgers_material_seg/utils --mode=diff --interact
        python ~/code/watch/dev/lint.py watch/tasks/rutgers_material_seg/*.py --mode=diff --interact
        python ~/code/watch/dev/lint.py watch/tasks/rutgers_material_seg/ --mode=diff --interact
        python ~/code/watch/dev/lint.py watch/tasks/ --mode=diff --interact
        python ~/code/watch/dev/lint.py watch/ --mode=diff --interact

        python ~/code/watch/dev/lint.py atk --mode=diff --interact

        python ~/code/watch/dev/lint.py ~/code/watch/watch/tasks/rutgers_material_change_detection/utils/utils.py --mode=diff
    """
    import fire
    import sys
    sys.exit(fire.Fire(custom_lint))
