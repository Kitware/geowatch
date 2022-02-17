"""
A registry of resource files bundled with the watch package

Schemas are taken from
https://smartgitlab.com/TE/standards/-/snippets/18

Previous:
https://smartgitlab.com/infrastructure/docs/-/tree/main/pages/schemas
commit fe4343521d05e433d4ccfcf080d9bcf46c9d2e83
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
    file = importlib_resources.open_text('watch.rc',
                                         'region-model.schema.json')
    data = json.load(file)
    return data


def load_job_schema():
    """
    Exapmle:
        >>> from watch.rc.registry import *  # NOQA
        >>> data = load_job_schema()
        >>> print('data = {!r}'.format(data))
    """
    import json
    from importlib import resources as importlib_resources
    file = importlib_resources.open_text('watch.rc', 'job.schema.json')
    data = json.load(file)
    return data
