"""
A registry of resource files bundled with the watch package
"""


def load_site_model_schema():
    """
    Exapmle:
        >>> from watch.rc.registry import *  # NOQA
        >>> data = load_site_model_schema()
        >>> print('data = {!r}'.format(data))
    """
    import json
    from importlib import resources as importlib_resources
    # with importlib_resources.path('watch.rc', name) as path:
    #     print('path = {!r}'.format(path))
    # list(importlib_resources.contents('watch.rc'))
    file = importlib_resources.open_text('watch.rc', 'site-model.schema.json')
    data = json.load(file)
    return data


def load_region_model_schema():
    """
    Exapmle:
        >>> from watch.rc.registry import *  # NOQA
        >>> data = load_region_model_schema()
        >>> print('data = {!r}'.format(data))
    """
    import json
    from importlib import resources as importlib_resources
    file = importlib_resources.open_text('watch.rc', 'region-model.schema.json')
    data = json.load(file)
    return data
