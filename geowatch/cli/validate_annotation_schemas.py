#!/usr/bin/env python3
r"""
CommandLine:
    python ~/code/watch/dev/validate_annotation_schemas.py

Prereq:


    TEST_DPATH=$HOME/tmp/test_smart_schema
    mkdir -p $TEST_DPATH
    cd $TEST_DPATH
    git clone git@smartgitlab.com:TE/annotations.git
    git clone git@smartgitlab.com:infrastructure/docs.git

    pip install jsonschema ubelt -U


SeeAlso:
    ~/code/watch/geowatch/geoannots/geomodels.py
    ~/code/watch/geowatch/cli/validate_annotation_schemas.py
    ~/code/watch/geowatch/cli/fix_region_models.py


References:
    https://smartgitlab.com/TE/annotations/-/issues/17
    https://smartgitlab.com/TE/standards/-/snippets/18


Example:

    DVC_DATA_DPATH=$(geowatch_dvc --tags='phase2_data' --hardware=auto)

    python -m geowatch.cli.validate_annotation_schemas \
        --site_models="$DVC_DATA_DPATH"/annotations/drop7/site_models \
        --region_models="$DVC_DATA_DPATH"/annotations/drop7/region_models

    python -m geowatch.cli.validate_annotation_schemas \
        --site_models="<path-to-site-models>" \
        --region_models="<path-to-region-models>"

    python -m geowatch.cli.validate_annotation_schemas \
        --region_models="$DVC_DATA_DPATH"/annotations/drop6/region_models/AE_C001.geojson
"""

import ubelt as ub
import jsonschema
import scriptconfig as scfg


class ValidateAnnotationConfig(scfg.DataConfig):
    """
    Validate the site / region model schemas
    """
    __command__ = 'site_validate'
    __alias__ = ['validate_sites']

    models = scfg.Value(None, help='site OR region models coercables (the script will attempt to distinguish them)', nargs='+', position=1)

    site_models = scfg.Value(None, nargs='+', help='coercable site models')

    region_models = scfg.Value(None, nargs='+', help='coercable region models')

    io_workers = scfg.Value('avail', help='number of workers for parallel io')

    fixup = scfg.Value(False, isflag=True, help='if True then run fixups before validation to check if the fixed version works')

    strict = scfg.Value(False, help='if True use the strict schema (i.e. enforces IARPA naming)')


def main(cmdline=1, **kwargs):
    """
    Example:
        >>> import ubelt as ub
        >>> import sys, ubelt
        >>> from geowatch.cli.validate_annotation_schemas import ValidateAnnotationConfig
        >>> from geowatch.geoannots import geomodels
        >>> dpath = ub.Path.appdir('geowatch', 'tests', 'test_validate_geoannot_schema')
        >>> dpath.ensuredir()
        >>> region, sites = geomodels.RegionModel.random(with_sites=True)
        >>> region_fpath = dpath / (region.region_id + '.geojson')
        >>> region_fpath.write_text(region.dumps())
        >>> for site in sites:
        >>>     site_fpath = dpath / (site.site_id + '.geojson')
        >>>     site_fpath.write_text(site.dumps())
        >>> kwargs = {
        >>>     'models': str(dpath / '*.geojson')
        >>> }
        >>> cmdline = 0
        >>> ValidateAnnotationConfig.main(cmdline=cmdline, **kwargs)
    """
    config = ValidateAnnotationConfig.cli(data=kwargs, cmdline=cmdline)
    from geowatch.utils import util_gis
    import rich
    rich.print(ub.urepr(config))

    from geowatch.geoannots import geomodels
    from kwutil import util_parallel

    io_workers = util_parallel.coerce_num_workers(config['io_workers'])

    if config.models:
        if config.site_models:
            raise ValueError('the models and site_models arguments are mutex')
        if config.region_models:
            raise ValueError('the models and region_models arguments are mutex')
        model_infos = list(util_gis.coerce_geojson_datas(config.models, format='json', workers=io_workers))
        site_model_infos = []
        region_model_infos = []
        for model_info in model_infos:
            model_data = model_info.pop('data')
            model = geomodels.coerce_site_or_region_model(model_data)
            model_info['model'] = model
            if isinstance(model, geomodels.SiteModel):
                site_model_infos.append(model_info)
            elif isinstance(model, geomodels.RegionModel):
                region_model_infos.append(model_info)
            else:
                raise AssertionError
    else:
        site_model_infos = list(util_gis.coerce_geojson_datas(config['site_models'], format='json', workers=io_workers))
        region_model_infos = list(util_gis.coerce_geojson_datas(config['region_models'], format='json', workers=io_workers))

        for model_info in region_model_infos + site_model_infos:
            model_data = model_info.pop('data')
            model = geomodels.coerce_site_or_region_model(model_data)
            model_info['model'] = model

    if config.fixup:
        for info in region_model_infos:
            info['model'].fixup()

        for info in site_model_infos:
            info['model'].fixup()

    print('Validate Data Content')
    validate_data_contents(region_model_infos, site_model_infos)

    print('Validate Schemas')
    validate_schemas(region_model_infos, site_model_infos,
                     strict=config.strict)


def validate_data_contents(region_model_infos, site_model_infos):
    """
    Content validation (i.e. dates look sane)
    """
    # Finish reading region models and build reports
    region_reports = []
    for region_info in region_model_infos:
        region_model = region_info['model']
        region_fpath = region_info['fpath']
        region_df = region_model.pandas()
        region_report = validate_region_model_content(region_df, region_fpath)
        region_reports.append(region_report)
    region_reports = sorted(region_reports, key=lambda x: (x['num_errors'], x['region_id']))
    print('region_reports = {}'.format(ub.urepr(region_reports, sort=0, nl=-1)).replace('\'', '"').replace('None', 'null'))

    # Finish reading site models, build reports while this is happening.
    site_reports = []
    for site_info in site_model_infos:
        site_model = site_info['model']
        site_fpath = site_info['fpath']
        site_df = site_model.pandas()
        site_report = validate_site_content(site_df, site_fpath)
        site_reports.append(site_report)

    region_id_to_site_reports = ub.group_items(site_reports, lambda x: x['region_id'])
    region_id_to_report_group = {}
    for region_id, sub_reports in region_id_to_site_reports.items():
        total_errors = sum([r['num_errors'] for r in sub_reports])
        invalid_site_reports = [r for r in sub_reports if r['num_errors']]
        for r in invalid_site_reports:
            r.pop('region_id', None)
        region_id_to_report_group[region_id] = {
            'region_id': region_id,
            'num_errors': total_errors,
            'valid_sites': len(sub_reports) - len(invalid_site_reports),
            'invalid_site_reports': invalid_site_reports,
        }

    grouped_site_reports = sorted(region_id_to_report_group.values(), key=lambda x: (x['num_errors'], x['region_id']))
    print('site_reports = {}'.format(ub.urepr(grouped_site_reports, sort=0, nl=-1)).replace('\'', '"').replace('None', 'null'))


def validate_schemas(region_model_infos, site_model_infos, strict=False):

    region_errors = []
    prog = ub.ProgIter(region_model_infos, desc='check region models')
    for region_model_info in prog:
        region_model_fpath = region_model_info['fpath']
        region_model = region_model_info['model']
        try:
            region_model.validate(verbose=0, strict=strict)
        except jsonschema.ValidationError as ex:
            error_info = {
                'type': 'region_model_error',
                'name': region_model_fpath.stem,
                'ex': ex,
            }
            region_errors.append(error_info)
            prog.set_description(f'check region models, errors: {len(region_errors)}')
        except Exception as ex:
            error_info = {
                'type': 'region_model_error',
                'name': region_model_fpath.stem,
                'ex': ex,
            }
            region_errors.append(error_info)
            prog.set_description(f'check region models, errors: {len(region_errors)}')

    site_errors = []
    prog = ub.ProgIter(site_model_infos, desc='check site models')
    for site_model_info in prog:
        site_model_fpath = site_model_info['fpath']
        site_model = site_model_info['model']
        try:
            site_model.validate(verbose=0, strict=strict)
            # site_model._validate_parts(strict=strict)
        except jsonschema.ValidationError as ex:
            error_info = {
                'type': 'site_model_error',
                'name': site_model_fpath.stem,
                'ex': ex,
            }
            site_errors.append(error_info)
            prog.set_description(f'check site models, errors: {len(site_errors)}')
        except Exception as ex:
            error_info = {
                'type': 'site_model_error',
                'name': site_model_fpath.stem,
                'ex': ex,
            }
            site_errors.append(error_info)
            prog.set_description(f'check site models, errors: {len(site_errors)}')

    print('site_errors = {}'.format(ub.urepr(site_errors, nl=1)))
    print('region_errors = {}'.format(ub.urepr(region_errors, nl=1)))

    print(f'{len(site_errors)} / {len(site_model_infos)} site model errors')
    print(f'{len(region_errors)} / {len(region_model_infos)} region model errors')


def validate_region_model_content(region_df, fpath):
    # import pandas as pd
    import os
    from kwutil import util_time
    is_region = region_df['type'] == 'region'
    region_part = region_df[is_region]
    assert len(region_part) == 1, 'must have exactly one region in each region file'
    assert region_part['region_id'].apply(lambda x: x is not None).all(), 'regions must have region ids'

    region_rel_fpath = fpath.relative_to(fpath.parent.parent.parent)

    region_row = region_part.iloc[0]
    region_id = region_row['region_id']
    region_stem = fpath.stem

    errors = []
    region_report = {
        'region_id': region_id,
        'rel_fpath': os.fspath(region_rel_fpath),
        'errors': errors,
    }

    if region_id != region_stem:
        errors.append({
            'description': 'A region file name does not match its region id',
            'offending_info': {
                'region_id': region_id,
                'filename': fpath.name,
            }
        })

    dummy_start_date = '1970-01-01'  # hack, could be more robust here
    dummy_end_date = '2101-01-01'

    class AbortCheck(Exception):
        ...

    # Hack to set all region-ids
    sites_part = region_df[~is_region]
    try:
        if not (sites_part['type'] == 'site_summary').all():
            bad_types = sites_part['type'].unique()
            errors.append({
                'description': f'rest of data must be site summaries, but got: {bad_types}'
            })
            raise AbortCheck

        region_start_date = region_row['start_date'] or dummy_start_date
        region_end_date = region_row['end_date'] or dummy_end_date

        region_start_datetime = util_time.coerce_datetime(region_start_date)
        region_end_datetime = util_time.coerce_datetime(region_end_date)
        if region_end_datetime < region_start_datetime:
            errors.append(f'Bad region dates: {region_start_datetime=}, {region_end_datetime=}')

        # Check datetime errors
        sitesum_start_dates = sites_part['start_date'].apply(lambda x: util_time.coerce_datetime(x or region_start_date))
        sitesum_end_dates = sites_part['end_date'].apply(lambda x: util_time.coerce_datetime(x or region_end_date))
        has_bad_time_range = sitesum_start_dates > sitesum_end_dates

        bad_date_rows = sites_part[has_bad_time_range]
        if len(bad_date_rows):
            bad_row_info = bad_date_rows[['site_id', 'start_date', 'end_date', 'originator']].to_dict('records')
            errors.append({
                'description': f'Site summary rows with start_dates > end_date or outside of region start/end dates ({region_start_date} - {region_end_date})',
                'offending_rows': bad_row_info,
            })
    except AbortCheck:
        ...
    except Exception as ex:
        errors.append({
            'description': 'Unknown error {}'.format(repr(ex))
        })

    if not region_report['errors']:
        region_report['errors'] = None
    region_report['num_errors'] = len(errors)
    return region_report


def validate_site_content(site_df, site_fpath):
    from kwutil import util_time
    import os
    import numpy as np
    dummy_start_date = '1970-01-01'  # hack, could be more robust here
    dummy_end_date = '2101-01-01'
    first = site_df.iloc[0]
    rest = site_df.iloc[1:]
    assert first['type'] == 'site', 'first row must have type of site'
    assert first['region_id'] is not None, 'first row must have a region id'
    assert rest['type'].apply(lambda x: x == 'observation').all(), (
        'rest of row must have type observation')
    assert rest['region_id'].isna().all(), (
        'rest of row must have region_id=None')

    site_start_date = first['start_date'] or dummy_start_date
    site_end_date = first['end_date'] or dummy_end_date
    site_start_datetime = util_time.coerce_datetime(site_start_date)
    site_end_datetime = util_time.coerce_datetime(site_end_date)

    rel_fpath = site_fpath.relative_to(site_fpath.parent.parent.parent)

    errors = []
    site_report = {
        'region_id': first['region_id'],
        'site_id': first['site_id'],
        'rel_fpath': os.fspath(rel_fpath),
        'errors': errors,
    }

    if site_end_datetime is not None and site_start_datetime is not None:
        if site_end_datetime < site_start_datetime:
            offending = first[['site_id', 'start_date', 'end_date', 'originator']].to_dict()
            errors.append({
                'description': 'Site summary row has a start_date > end_date',
                'offending_info': offending,
            })

    # Check datetime errors in observations
    null_obs_sites = []
    try:
        obs_dates = [None if x is None else util_time.coerce_datetime(x) for x in rest['observation_date']]
        obs_isvalid = [x is not None for x in obs_dates]
        valid_obs_dates = list(ub.compress(obs_dates, obs_isvalid))
        if not all(valid_obs_dates):
            null_obs_sites.append(first[['site_id', 'status']].to_dict())
        valid_deltas = np.array([d.total_seconds() for d in np.diff(valid_obs_dates)])
        if not (valid_deltas >= 0).all():
            errors.append({
                'description': 'observations are not sorted temporally',
                'offending_info': valid_obs_dates,
            })
    except AssertionError as ex:
        print('ex = {!r}'.format(ex))
        print(site_df)
        raise

    if null_obs_sites:
        errors.append({
            'description': 'sites contained null observations',
            'offending_info': null_obs_sites,
        })

    if not site_report['errors']:
        site_report['errors'] = None
    site_report['num_errors'] = len(errors)

    return site_report


__config__ = ValidateAnnotationConfig
__config__.main = main


if __name__ == '__main__':
    main()
