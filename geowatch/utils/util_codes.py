
def parse_delimited_argstr(data):
    """
    Special suffixes can be added to generic demo names. Parse them out here.
    Arguments are `-` separated, only known defaulted values are parsed. Bare
    default names are interpreted as a value of True, otherwise the value
    should be numeric. TODO: generalize this and conslidate in the kwcoco
    demo method.

    Example:
        >>> from geowatch.utils.util_codes import *  # NOQA
        >>> data = 'foo-bar-baz1-biz2.3'
        >>> defaults = {}
        >>> alias_to_key = None
        >>> parse_delimited_argstr(data)
        {'foo': True, 'bar': True, 'baz': 1, 'biz': 2.3}
    """
    import re
    from scriptconfig.smartcast import smartcast
    parts = data.split('-')
    parsed = {}
    for part in parts:
        match = re.search(r'[\d]', part)
        if match is None:
            value = True
            key = part
        else:
            key = part[:match.span()[0]]
            value = smartcast(part[match.span()[0]:], allow_split=False)
        parsed[key] = value
    return parsed
