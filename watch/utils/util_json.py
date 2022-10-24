

def debug_json_unserializable(data, msg=''):
    """
    Raises an exception if the data is not serializable and prints information
    about it.
    """
    from kwcoco.util import util_json
    import ubelt as ub
    unserializable = list(util_json.find_json_unserializable(data))
    if unserializable:
        raise Exception(msg + ub.repr2(unserializable))
