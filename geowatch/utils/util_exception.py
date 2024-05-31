"""
TODO: port to kwutil
"""


def add_exception_note(ex, note, force_legacy=False):
    """
    Support for PEP 678 in Python < 3.11

    If PEP 678 is available, use it, otherwise create a new exeception based on
    the old one with an updated note.

    Args:
        ex (BaseException): the exception to modify
        note (str): extra unstructured information to add
        force_legacy (bool): for testing

    Returns:
        BaseException: modified exception

    Example:
        >>> from geowatch.utils.util_exception import add_exception_note
        >>> ex = Exception('foo')
        >>> new_ex = add_exception_note(ex, 'hello world', force_legacy=False)
        >>> print(new_ex)
        >>> new_ex = add_exception_note(ex, 'hello world', force_legacy=True)
        >>> print(new_ex)
    """
    if not force_legacy and hasattr(ex, 'add_note'):
        # Requires python.311 PEP 678
        ex.add_note(note)
        return ex
    else:
        return type(ex)(str(ex) + chr(10) + note)
