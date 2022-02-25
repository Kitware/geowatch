"""
A registry of resource files bundled with the watch package

Schemas are taken from
https://smartgitlab.com/TE/standards/-/snippets/18

Previous:
https://smartgitlab.com/infrastructure/docs/-/tree/main/pages/schemas
commit fe4343521d05e433d4ccfcf080d9bcf46c9d2e83

Geoidgrid is taken from
https://smartgitlab.com/TE/annotations/-/wikis/WorldView-Annotations#notes-on-the-egm96-geoidgrid-file
"""
import json
from importlib import resources as importlib_resources
import ubelt as ub


def load_site_model_schema():
    """
    Example:
        >>> from watch.rc.registry import *  # NOQA
        >>> data = load_site_model_schema()
        >>> print('data = {!r}'.format(data))
    """
    # with importlib_resources.path('watch.rc', name) as path:
    #     print('path = {!r}'.format(path))
    # list(importlib_resources.contents('watch.rc'))
    file = importlib_resources.open_text('watch.rc', 'site-model.schema.json')
    data = json.load(file)
    return data


def load_region_model_schema():
    """
    Example:
        >>> from watch.rc.registry import *  # NOQA
        >>> data = load_region_model_schema()
        >>> print('data = {!r}'.format(data))
    """
    file = importlib_resources.open_text('watch.rc',
                                         'region-model.schema.json')
    data = json.load(file)
    return data


def load_job_schema():
    """
    Example:
        >>> from watch.rc.registry import *  # NOQA
        >>> data = load_job_schema()
        >>> print('data = {!r}'.format(data))
    """
    file = importlib_resources.open_text('watch.rc', 'job.schema.json')
    data = json.load(file)
    return data


def geoidgrid_path():
    with importlib_resources.path('watch.rc', 'egm96_15.gtx') as p:
        return ub.Path(p)


def dem_path(cache_dir=None, overwrite=False):
    with importlib_resources.path('watch.rc', 'dem.xml') as p:
        orig_pth = ub.Path(p)

    if cache_dir is None:
        cache_dir = ub.Path.appdir('smart_watch/dem')
    cache_dir = ub.Path(cache_dir).ensuredir()

    cached_pth = ub.Path(cache_dir) / orig_pth.name
    if overwrite or not cached_pth.is_file():
        with open(orig_pth) as orig_f, open(cached_pth.delete(),
                                            'w+') as cached_f:
            cached_f.write(orig_f.read().replace('./gdalwmscache',
                                                 str(cache_dir.absolute())))

    return cached_pth
