"""
A registry of resource files bundled with the geowatch package

Schemas are from
* https://smartgitlab.com/TE/aristeas/-/tree/main/src/aristeas/schemas
* https://smartgitlab.com/TE/standards
* https://smartgitlab.com/TE/standards/-/snippets/18

Previous:
* https://smartgitlab.com/infrastructure/docs/-/tree/main/pages/schemas
* commit fe4343521d05e433d4ccfcf080d9bcf46c9d2e83

Geoidgrid is from
* https://smartgitlab.com/TE/annotations/-/wikis/WorldView-Annotations#notes-on-the-egm96-geoidgrid-file

SeeAlso:
    ../geoannots/geomodels.py
    https://github.com/ResonantGeoData/RD-WATCH/blob/main/django/src/rdwatch/schemas/site_model.py
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
        >>> from geowatch.rc.registry import *  # NOQA
        >>> data1 = load_site_model_schema(strict=True)
        >>> data2 = load_site_model_schema(strict=False)
        >>> import rich
        >>> rich.print('data = {}'.format(ub.urepr(data1, nl=-2)))
        >>> rich.print('data = {}'.format(ub.urepr(data2, nl=-2)))
        >>> import jsonschema
        >>> cls = jsonschema.validators.validator_for(data1)
        >>> cls.check_schema(data1)
        >>> cls = jsonschema.validators.validator_for(data2)
        >>> cls.check_schema(data2)
    """
    rc_dpath = importlib_resources.files('geowatch.rc')
    schema_fpath = rc_dpath / 'site-model.schema.json'
    data = json.loads(schema_fpath.read_text())
    # file = importlib_resources.open_text('geowatch.rc', 'site-model.schema.json')
    # data = json.load(file)
    if not strict:
        from kwcoco.util.jsonschema_elements import STRING
        from kwcoco.util.jsonschema_elements import ONEOF
        # from kwcoco.util.jsonschema_elements import ANYOF
        from kwcoco.util.jsonschema_elements import NULL
        any_identifier = STRING(pattern='^[A-Za-z_][A-Za-z0-9_-]*$')
        any_sensor = STRING(pattern=r'^[A-Za-z_][A-Za-z0-9\s_-]*$')
        walker = ub.IndexableWalker(data)
        if 0:
            # Identify the paths to the schema element we are going to modify
            leaf_to_loose = {
                'region_id': 'any_identifier',
                'site_id': 'any_identifier',
                'originator': 'any_identifier',
                'sensor_name': 'ONEOF(NULL, any_identifier)',
            }
            suggestions = []
            for p, v in walker:
                if p[-1] in leaf_to_loose:
                    print(f'p={p}')
                    print(f'v={v}')
                    loose_val = leaf_to_loose[p[-1]]
                    suggestions.append(f'walker[{p}] = {loose_val}  # previous: {v!r}')
            print('Suggestions: ')
            print('\n'.join(suggestions))

        walker[['$defs', 'site_properties', 'allOf', 4, 'then', 'properties', 'site_id']] = any_identifier  # previous: {'type': 'string', 'pattern': '^[A-Z]{2}_([RS]\\d{3}|C[0-7]\\d{2})_\\d{4}$'}
        walker[['$defs', 'site_properties', 'allOf', 4, 'if', 'properties', 'region_id']] = any_identifier  # previous: {'type': 'string', 'pattern': '^[A-Z]{2}_([RS]\\d{3}|C[0-7]\\d{2})$'}
        walker[['$defs', 'site_properties', 'allOf', 3, 'then', 'properties', 'site_id']] = any_identifier  # previous: {'type': 'string', 'pattern': '^[A-Z]{2}_[RC][Xx]{3}_\\d{4}$'}
        walker[['$defs', 'site_properties', 'allOf', 3, 'if', 'properties', 'region_id']] = any_identifier  # previous: {'type': 'string', 'pattern': '^[A-Z]{2}_[RC][Xx]{3}$'}
        walker[['$defs', 'site_properties', 'allOf', 0, 'then', 'properties', 'originator']] = any_identifier  # previous: {'enum': ['te', 'iMERIT', 'pmo']}
        walker[['$defs', 'site_properties', 'properties', 'site_id']] = any_identifier  # previous: {'type': 'string', 'pattern': '^[A-Z]{2}_([RS]\\d{3}|C[0-7]\\d{2}|[RC][Xx]{3})_\\d{4}$'}
        walker[['$defs', 'site_properties', 'properties', 'region_id']] = any_identifier  # previous: {'oneOf': [{'type': 'null'}, {'type': 'string', 'pattern': '^[A-Z]{2}_([RS]\\d{3}|C[0-7]\\d{2}|[RC][Xx]{3})$'}]}
        walker[['$defs', 'site_properties', 'properties', 'originator']] = any_identifier  # previous: {'enum': ['te', 'pmo', 'acc', 'ast', 'ast', 'bla', 'iai', 'kit', 'str', 'iMERIT']}
        walker[['$defs', 'observation_properties', 'properties', 'sensor_name']] = ONEOF(NULL, any_sensor)  # previous: {'oneOf': [{'type': 'null'}, {'type': 'string', 'pattern': '^((Landsat 8|Sentinel-2|WorldView|Planet), )*(Landsat 8|Sentinel-2|WorldView|Planet)$'}]}

        # walker[['definitions', 'associated_site_properties', 'properties',
        #         'region_id']] = any_identifier
        # walker[['definitions', 'associated_site_properties', 'properties',
        #         'site_id']] = any_identifier
        # walker[['definitions', 'unassociated_site_properties', 'properties',
        #         'region_id']] = ONEOF(NULL, any_identifier)
        # walker[['definitions', 'unassociated_site_properties', 'properties',
        #         'site_id']] = any_identifier
        # walker[['definitions', '_site_properties', 'properties',
        #         'originator']] = any_identifier
        # walker[['definitions', 'observation_properties', 'properties',
        #         'sensor_name']] = ONEOF(NULL, any_identifier)

        # By setting strict=False, unassociated and associated site properties
        # are no longer distinguished, so we have to just pick one.
        # walker[['properties', 'features', 'items', 'anyOf', 0, 'properties',
        #         'properties']] = ANYOF(
        #     {'$ref': '#/definitions/associated_site_properties'},
        #     {'$ref': '#/definitions/unassociated_site_properties'},
        # )

    return data


def load_region_model_schema(strict=True):
    """
    Args:
        strict (bool):
            if True we make a few changes the schema to be more permissive
            towards things like region names and originator.

    Returns:
        Dict: the schema

    CommandLine:
        xdoctest -m geowatch.rc.registry load_region_model_schema

    Example:
        >>> from geowatch.rc.registry import *  # NOQA
        >>> data1 = load_region_model_schema(strict=True)
        >>> data2 = load_region_model_schema(strict=False)
        >>> import rich
        >>> rich.print('data = {}'.format(ub.urepr(data1, nl=-2)))
        >>> rich.print('data = {}'.format(ub.urepr(data2, nl=-2)))
        >>> import jsonschema
        >>> cls = jsonschema.validators.validator_for(data1)
        >>> cls.check_schema(data1)
        >>> cls = jsonschema.validators.validator_for(data2)
        >>> cls.check_schema(data2)
    """
    rc_dpath = importlib_resources.files('geowatch.rc')
    schema_fpath = rc_dpath / 'region-model.schema.json'
    data = json.loads(schema_fpath.read_text())
    # file = importlib_resources.open_text('geowatch.rc',
    #                                      'region-model.schema.json')
    # data = json.load(file)
    if not strict:
        from kwcoco.util.jsonschema_elements import STRING

        # Allow any alphanumeric region id
        any_identifier = STRING(pattern='^[A-Za-z_][A-Za-z0-9_-]*$')
        walker = ub.IndexableWalker(data)
        if 0:
            # Identify the paths to the schema element we are going to modify
            leaf_to_loose = {
                'region_id': 'any_identifier',
                'site_id': 'any_identifier',
                'originator': 'any_identifier',
            }
            suggestions = []
            for p, v in walker:
                if p[-1] in leaf_to_loose:
                    print(f'p={p}')
                    print(f'v={v}')
                    loose_val = leaf_to_loose[p[-1]]
                    suggestions.append(f'walker[{p}] = {loose_val}  # previous: {v!r}')
            print('Suggestions: ')
            print('\n'.join(suggestions))

        # walker[['$defs', 'site_summary_properties', 'properties', 'site_id']] = any_identifier  # previous: {'type': 'string', 'pattern': '^[A-Z]{2}_([RS]\\d{3}|C[0-7]\\d{2})_\\d{4}$'}
        # walker[['$defs', 'site_summary_properties', 'properties', 'originator']] = any_identifier  # previous: {'enum': ['te', 'pmo', 'acc', 'ast', 'ast', 'bla', 'iai', 'kit', 'str', 'iMERIT']}
        # walker[['$defs', 'region_properties', 'properties', 'region_id']] = any_identifier  # previous: {'type': 'string', 'pattern': '^[A-Z]{2}_([RS]\\d{3}|C[0-7]\\d{2})$'}
        # walker[['$defs', 'region_properties', 'properties', 'originator']] = any_identifier  # previous: {'enum': ['te', 'pmo', 'acc', 'ara', 'ast', 'bla', 'iai', 'kit', 'str', 'iMERIT']}
        # walker[['$defs', 'proposed_originator', 'then', 'properties', 'originator']] = any_identifier  # previous: {'enum': ['acc', 'ara', 'ast', 'bla', 'iai', 'kit', 'str']}
        # walker[['$defs', 'annotation_originator', 'then', 'properties', 'originator']] = any_identifier  # previous: {'enum': ['te', 'iMERIT', 'pmo']}

        # New schema
        walker[['$defs', 'site_summary_properties', 'allOf', 4, 'else', 'properties', 'site_id']] = any_identifier  # previous: {'type': 'string', 'pattern': '^[A-Z]{2}_([RS]\\d{3}|C[0-7]\\d{2})_\\d{4}$'}
        walker[['$defs', 'site_summary_properties', 'allOf', 4, 'then', 'properties', 'site_id']] = any_identifier  # previous: {'type': 'string', 'pattern': '^[A-Z]{2}_(T\\d{3})_\\d{4}$'}
        walker[['$defs', 'site_summary_properties', 'properties', 'site_id']] = any_identifier  # previous: {'type': 'string', 'pattern': '^[A-Z]{2}_([RST]\\d{3}|C[0-7]\\d{2})_\\d{4}$'}
        walker[['$defs', 'site_summary_properties', 'properties', 'originator']] = any_identifier  # previous: {'enum': ['te', 'pmo', 'acc', 'ast', 'ast', 'bla', 'iai', 'kit', 'str', 'iMERIT']}
        walker[['$defs', 'region_properties', 'properties', 'region_id']] = any_identifier  # previous: {'type': 'string', 'pattern': '^[A-Z]{2}_([RST]\\d{3}|C[0-7]\\d{2})$'}
        walker[['$defs', 'region_properties', 'properties', 'originator']] = any_identifier  # previous: {'enum': ['te', 'pmo', 'acc', 'ara', 'ast', 'bla', 'iai', 'kit', 'str', 'iMERIT']}
        walker[['$defs', 'proposed_originator', 'then', 'properties', 'originator']] = any_identifier  # previous: {'enum': ['acc', 'ara', 'ast', 'bla', 'iai', 'kit', 'str']}
        walker[['$defs', 'annotation_originator', 'then', 'properties', 'originator']] = any_identifier  # previous: {'enum': ['te', 'iMERIT', 'pmo']}

    return data


def load_job_schema():
    """
    Example:
        >>> from geowatch.rc.registry import *  # NOQA
        >>> data = load_job_schema()
        >>> print('data = {!r}'.format(data))
    """
    file = importlib_resources.open_text('geowatch.rc', 'job.schema.json')
    data = json.load(file)
    return data


def geoidgrid_path():
    with importlib_resources.path('geowatch.rc', 'egm96_15.gtx') as p:
        return ub.Path(p)


def dem_path(cache_dir=None, overwrite=False):
    with importlib_resources.path('geowatch.rc', 'dem.xml') as p:
        orig_pth = ub.Path(p)

    if cache_dir is None:
        cache_dir = ub.Path.appdir('geowatch/dem')
    cache_dir = ub.Path(cache_dir).ensuredir()

    cached_pth = ub.Path(cache_dir) / orig_pth.name
    if overwrite or not cached_pth.is_file():
        with open(orig_pth) as orig_f, open(cached_pth.delete(),
                                            'w+') as cached_f:
            cached_f.write(orig_f.read().replace('./gdalwmscache',
                                                 str(cache_dir.absolute())))

    return cached_pth


def requirement_path(fname):
    """

    CommandLine:
        xdoctest -m geowatch.rc.registry requirement_path

    Example:
        >>> from geowatch.rc.registry import requirement_path
        >>> fname = 'runtime.txt'
        >>> requirement_path(fname)
    """
    with importlib_resources.path('geowatch.rc.requirements', f'{fname}') as p:
        orig_pth = ub.Path(p)
        return orig_pth
