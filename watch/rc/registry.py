"""
A registry of resource files bundled with the watch package
"""


# def resource_path(name):
#     import sys
#     if sys.version_info >= (3, 7):
#         from importlib import resources as importlib_resources
#     else:
#         import importlib_resources
#         import importlib_resources

#     name = 'site-model.schema.json'
#     with importlib_resources.path('watch.rc', name) as path:
#         print('path = {!r}'.format(path))

#     list(importlib_resources.contents('watch.rc'))
#     p = importlib_resources.open_text('watch.rc', 'site-model.schema.json')


def load_site_model_schema():
    """
    Exapmle:
        >>> from watch.rc.registry import *  # NOQA
        >>> load_site_model_schema()
    """
    import sys
    if sys.version_info >= (3, 7):
        from importlib import resources as importlib_resources
    else:
        import importlib_resources
    import json
    file = importlib_resources.open_text('watch.rc', 'site-model.schema.json')
    data = json.load(file)
    return data
