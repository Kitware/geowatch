"""
This logic might get ported to ub.Path.chmod
"""


def parse_perms(code):
    """
    Ignore:
        [ugoa…][-+=]perms…[,…]
        from geowatch.tasks.fusion.fit_lightning import *  # NOQA
        print(parse_perms('ugo+rw,+r,g=rwx'))
        print(parse_perms('o+x'))
        print(parse_perms('u-x'))
        print(parse_perms('x'))
    """
    import re
    pat = re.compile(r'([\+\-\=])')
    parts = code.split(',')
    actions = []
    for part in parts:
        ab = pat.split(part)
        if len(ab) == 3:
            targets, op, perms = ab
        elif len(ab) == 2:
            op, perms = ab
            assert set('+-=') & set(op)
            targets = 'a'
        elif len(ab) == 1:
            perms = ab[0]
            op = '+'
            targets = 'u'
        else:
            raise Exception
        if targets == '' or targets == 'a':
            targets = 'ugo'
        actions.append((targets, op, perms))
    return actions


def new_chmod(path, code):
    """
    dpath = ub.Path.appdir('util/chmod/test').ensuredir()
    path = (dpath / 'file').touch()
    code = 'g+x,g+r'
    new_chmod(path, code)
    import stat
    stat.filemode(path.stat().st_mode)
    """
    import stat
    actions = parse_perms(code)
    lut = {
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

    def make_eq(new_val):
        def _close(current):
            return new_val
        return _close

    new_mode = path.stat().st_mode
    for action in actions:
        targets, op, perms = action
        for target, perm in zip(targets, perms):
            key = target + perm
            val = lut[key]
            if op == '+':
                new_mode |= val
            elif op == '-':
                new_mode &= (~val)
            elif op == '=':
                raise NotImplementedError
            else:
                raise AssertionError
    path.chmod(new_mode)
