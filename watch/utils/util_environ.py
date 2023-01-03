import os


TRUTHY_ENVIRONS = {'true', 'on', 'yes', '1', 't'}


def envflag(key, default=None, environ=None):
    """
    Determine if an environment variable is specified and truthy or falsy.

    Args:
        key (str): the environment variable name to check

        default (Any):
            the default value to return if the environment variable is not
            specified.

        environ (None | Dict):
            Uses this to get the environment variable. If unspecified, defaults
            to ``os.environ``.

    Returns:
        True if the environment variable exist and matches a truthy pattern.
        (e.g. true, on, yes, 1, or t). Otherwise returns False.

    Note:
        This will return false on any setting of the environ that is not
        truthy. (e.g. YESPLEASE is not a registered TRUTHY_ENVIRON so
        it will return False).

    Example:
        >>> from watch.utils import util_environ
        >>> environ = {
        >>>     'foo': '1',
        >>>     'bar': 'YES',
        >>>     'baz': '0',
        >>>     'biz': '1111',
        >>> }
        >>> assert util_environ.envflag('foo', 0, environ=environ)
        >>> assert util_environ.envflag('bar', 0, environ=environ)
        >>> assert not util_environ.envflag('baz', 0, environ=environ)
        >>> assert not util_environ.envflag('biz', 0, environ=environ)
        >>> assert not util_environ.envflag('buzz', 0, environ=environ)
        >>> assert util_environ.envflag('buzz', 1, environ=environ)
    """
    if environ is None:
        environ = os.environ
    if key not in environ:
        return default
    value = environ[key]
    if isinstance(value, str):
        value = value.lower() in TRUTHY_ENVIRONS
    else:
        assert value is None, 'environ values should all be strings'
    return value
