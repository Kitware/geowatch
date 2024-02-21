"""
This logic was ported to ub.Path.chmod in 1.3.5, can remove and depend on that
when it is ready.
"""


def _parse_chmod_code(code):
    """
    Expand a chmod code into a list of actions.

    Args:
        code (str): of the form: [ugoa…][-+=]perms…[,…]
            perms is either zero or more letters from the set rwxXst, or a
            single letter from the set ugo.

    Yields:
        Tuple[str, str, str]: target, op, and perms.

            The target is modified by the operation using the value.
            target -- specified 'u' for user, 'g' for group, 'o' for other.
            op -- specified as '+' to add, '-' to remove, or '=' to assign.
            val -- specified as 'r' for read, 'w' for write, or 'x' for execute.

    Example:
        >>> print(list(_parse_chmod_code('ugo+rw,+r,g=rwx')))
        >>> print(list(_parse_chmod_code('o+x')))
        >>> print(list(_parse_chmod_code('u-x')))
        >>> print(list(_parse_chmod_code('x')))
        >>> print(list(_parse_chmod_code('ugo+rwx')))
        [('ugo', '+', 'rw'), ('ugo', '+', 'r'), ('g', '=', 'rwx')]
        [('o', '+', 'x')]
        [('u', '-', 'x')]
        [('u', '+', 'x')]
        [('ugo', '+', 'rwx')]
        >>> import pytest
        >>> with pytest.raises(ValueError):
        >>>     list(_parse_chmod_code('a+b+c'))
    """
    import re
    pat = re.compile(r'([\+\-\=])')
    parts = code.split(',')
    for part in parts:
        ab = pat.split(part)
        len_ab = len(ab)
        if len_ab == 3:
            targets, op, perms = ab
        elif len_ab == 1:
            perms = ab[0]
            op = '+'
            targets = 'u'
        else:
            raise ValueError('unknown chmod code pattern: part={part}')
        if targets == '' or targets == 'a':
            targets = 'ugo'
        yield (targets, op, perms)


def _resolve_chmod_code(old_mode, code):
    """
    Modifies integer stat permissions based on a string code.

    Args:
        old_mode (int): old mode from st_stat
        code (str): chmod style codeold mode from st_stat

    Returns:
        int : new code

    Example:
        >>> print(oct(_resolve_chmod_code(0, '+rwx')))
        >>> print(oct(_resolve_chmod_code(0, 'ugo+rwx')))
        >>> print(oct(_resolve_chmod_code(0, 'a-rwx')))
        >>> print(oct(_resolve_chmod_code(0, 'u+rw,go+r,go-wx')))
        >>> print(oct(_resolve_chmod_code(0o777, 'u+rw,go+r,go-wx')))
        0o777
        0o777
        0o0
        0o644
        0o744
        >>> import pytest
        >>> with pytest.raises(NotImplementedError):
        >>>     print(oct(_resolve_chmod_code(0, 'u=rw')))
        >>> with pytest.raises(ValueError):
        >>>     _resolve_chmod_code(0, 'u?w')
    """
    import stat
    import itertools as it
    action_lut = {
        # TODO: handle suid, sgid, and sticky?
        # suid = stat.S_ISUID
        # sgid = stat.S_ISGID
        # sticky = stat.S_ISVTX
        'ur' : stat.S_IRUSR,
        'uw' : stat.S_IWUSR,
        'ux' : stat.S_IXUSR,

        'gr' : stat.S_IRGRP,
        'gw' : stat.S_IWGRP,
        'gx' : stat.S_IXGRP,

        'or' : stat.S_IROTH,
        'ow' : stat.S_IWOTH,
        'ox' : stat.S_IXOTH,
    }
    actions = _parse_chmod_code(code)
    new_mode = int(old_mode)  # (could optimize to modify inplace if needed)
    for action in actions:
        targets, op, perms = action
        try:
            action_keys = (target + perm for target, perm in it.product(targets, perms))
            action_values = (action_lut[key] for key in action_keys)
            action_values = list(action_values)
            if op == '+':
                for val in action_values:
                    new_mode |= val
            elif op == '-':
                for val in action_values:
                    new_mode &= (~val)
            elif op == '=':
                raise NotImplementedError(f'new chmod code for op={op}')
            else:
                raise AssertionError(
                    f'should not be able to get here. unknown op code: op={op}')
        except KeyError:
            # Give a better error message if something goes wrong
            raise ValueError(f'Unknown action: {action}')
    return new_mode


def _encode_chmod_int(int_code):
    """
    Convert a chmod integer code to a string

    Currently unused, but may be useful in the future.

    Example:
        >>> int_code = 0o744
        >>> print(_encode_chmod_int(int_code))
        u=rwx,g=r,o=r
    """
    import stat
    action_lut = {
        'ur' : stat.S_IRUSR,
        'uw' : stat.S_IWUSR,
        'ux' : stat.S_IXUSR,

        'gr' : stat.S_IRGRP,
        'gw' : stat.S_IWGRP,
        'gx' : stat.S_IXGRP,

        'or' : stat.S_IROTH,
        'ow' : stat.S_IWOTH,
        'ox' : stat.S_IXOTH,
    }
    from collections import defaultdict
    target_to_perms = defaultdict(list)
    for key, val in action_lut.items():
        target, perm = key
        if int_code & val:
            target_to_perms[target].append(perm)
    parts = [k + '=' + ''.join(vs) for k, vs in target_to_perms.items()]
    code = ','.join(parts)
    return code


def new_chmod(path, code):
    """
    dpath = ub.Path.appdir('util/chmod/test').ensuredir()
    path = (dpath / 'file').touch()
    code = 'g+x,g+r'
    new_chmod(path, code)
    import stat
    stat.filemode(path.stat().st_mode)
    """
    old_mode = path.stat().st_mode
    new_mode = _resolve_chmod_code(old_mode, code)
    path.chmod(new_mode)
