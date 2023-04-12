"""
A registry of resource files bundled with the watch package

Schemas are taken from
https://smartgitlab.com/TE/standards/-/snippets/18

Previous:
https://smartgitlab.com/infrastructure/docs/-/tree/main/pages/schemas
commit fe4343521d05e433d4ccfcf080d9bcf46c9d2e83

Geoidgrid is taken from
https://smartgitlab.com/TE/annotations/-/wikis/WorldView-Annotations#notes-on-the-egm96-geoidgrid-file

SeeAlso:
    ../geoannots/geomodels.py
"""
import json
from importlib import resources as importlib_resources
import ubelt as ub


def load_site_model_schema(strict=True):
    """
    Args:
        strict (bool):
            if True we make a few changes the schema to be more permissive
            towards things like region names and originator.

    Example:
        >>> from watch.rc.registry import *  # NOQA
        >>> data = load_site_model_schema(strict=False)
        >>> import rich
        >>> rich.print('data = {}'.format(ub.urepr(data, nl=-2)))
    """
    # with importlib_resources.path('watch.rc', name) as path:
    #     print('path = {!r}'.format(path))
    # list(importlib_resources.contents('watch.rc'))
    file = importlib_resources.open_text('watch.rc', 'site-model.schema.json')
    data = json.load(file)
    if not strict:
        from kwcoco.util.jsonschema_elements import STRING
        from kwcoco.util.jsonschema_elements import ONEOF
        from kwcoco.util.jsonschema_elements import NULL
        any_identifier = STRING(pattern='^[A-Za-z0-9_-]+$')
        walker = ub.IndexableWalker(data)
        if 0:
            # Identify the paths to the schema element we are going to modify
            for p, v in walker:
                if p[-1] in {'region_id', 'site_id', 'originator', 'sensor_name'}:
                    print(f'p={p}')
                    print(f'v={v}')
        walker[['definitions', 'associated_site_properties', 'properties',
                'region_id']] = any_identifier
        walker[['definitions', 'associated_site_properties', 'properties',
                'site_id']] = any_identifier
        walker[['definitions', 'unassociated_site_properties', 'properties',
                'region_id']] = ONEOF(NULL, any_identifier)
        walker[['definitions', 'unassociated_site_properties', 'properties',
                'site_id']] = any_identifier
        walker[['definitions', '_site_properties', 'properties',
                'originator']] = any_identifier
        walker[['definitions', 'observation_properties', 'properties',
                'sensor_name']] = ONEOF(NULL, any_identifier)
    return data


def load_region_model_schema(strict=True):
    """
    Args:
        strict (bool):
            if True we make a few changes the schema to be more permissive
            towards things like region names and originator.

    Example:
        >>> from watch.rc.registry import *  # NOQA
        >>> data = load_site_model_schema(strict=False)
        >>> import rich
        >>> rich.print('data = {}'.format(ub.urepr(data, nl=-2)))
    """
    file = importlib_resources.open_text('watch.rc',
                                         'region-model.schema.json')
    data = json.load(file)
    if not strict:
        from kwcoco.util.jsonschema_elements import STRING
        any_identifier = STRING(pattern='^[A-Za-z0-9_-]+$')
        walker = ub.IndexableWalker(data)
        # Allow any alphanumeric region id
        walker[['definitions', 'region_properties',
                'properties', 'region_id']] = any_identifier
        walker[['definitions', 'region_properties',
                'properties', 'originator']] = any_identifier
        walker[['definitions', 'site_summary_properties',
                'properties', 'site_id']] = any_identifier
        walker[['definitions', 'site_summary_properties',
                'properties', 'originator']] = any_identifier
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
        cache_dir = ub.Path.appdir('watch/dem')
    cache_dir = ub.Path(cache_dir).ensuredir()

    cached_pth = ub.Path(cache_dir) / orig_pth.name
    if overwrite or not cached_pth.is_file():
        with open(orig_pth) as orig_f, open(cached_pth.delete(),
                                            'w+') as cached_f:
            cached_f.write(orig_f.read().replace('./gdalwmscache',
                                                 str(cache_dir.absolute())))

    return cached_pth
