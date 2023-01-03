"""
CommandLine:
    python ~/code/watch/dev/validate_annotation_schemas.py

Prereq:


    TEST_DPATH=$HOME/tmp/test_smart_schema
    mkdir -p $TEST_DPATH
    cd $TEST_DPATH
    git clone git@smartgitlab.com:TE/annotations.git
    git clone git@smartgitlab.com:infrastructure/docs.git

    pip install jsonschema ubelt -U


Also:

    python -m watch.demo.metrics_demo.generate_demodata --reset
    python ~/code/watch/dev/validate_annotation_schemas.py


References:
    https://smartgitlab.com/TE/annotations/-/issues/17
    https://smartgitlab.com/TE/standards/-/snippets/18


Example:

    DVC_DATA_DPATH=$(smartwatch_dvc --tags='phase2_data' --hardware=auto)

    python -m watch.cli.validate_annotation_schemas \
        --site_model_dpath="$DVC_DATA_DPATH"/annotations/drop6/site_models \
        --region_model_dpath="$DVC_DATA_DPATH"/annotations/drop6/region_models

"""

import ubelt as ub
import jsonschema
import json
import scriptconfig as scfg


class ValidateAnnotationConfig(scfg.DataConfig):
    """
    Validate the site / region model schemas
    """

    site_model_dpath = scfg.Value('auto', help='path to the site model directory')

    region_model_dpath = scfg.Value('auto', help='path to the region model directory')

    site_schema = scfg.Value('auto', help='path to the site model directory')

    region_schema = scfg.Value('auto', help='path to the region model directory')


def main():
    # Point to the cloned repos
    # docs_dpath = ub.Path('docs')
    # annotations_dpath = dvc_dpath / 'annotations/drop6'
    config = ValidateAnnotationConfig.cli()
    # config['site_model_dpath']

    # Load schemas
    # site_model_schema_fpath = docs_dpath / ('pages/schemas/site-model.schema.json')
    # region_model_schema_fpath = docs_dpath / ('pages/schemas/region-model.schema.json')
    # site_model_schema = json.loads(site_model_schema_fpath.read_text())
    # region_model_schema = json.loads(region_model_schema_fpath.read_text())

    if config['site_schema'] == 'auto':
        import watch
        site_schema = site_model_schema = watch.rc.registry.load_site_model_schema()
    else:
        raise NotImplementedError

    if config['region_schema'] == 'auto':
        import watch
        region_schema = region_model_schema = watch.rc.registry.load_region_model_schema()
    else:
        raise NotImplementedError

    if config['site_model_dpath'] == 'auto':
        import watch
        dvc_dpath = watch.find_dvc_dpath(tags='phase2_data', hardware='auto')
        site_model_dpath = dvc_dpath / 'annotations/drop6/site_models'
    else:
        site_model_dpath = ub.Path(config['site_model_dpath'])

    if config['region_model_dpath'] == 'auto':
        import watch
        dvc_dpath = watch.find_dvc_dpath(tags='phase2_data', hardware='auto')
        region_model_dpath = dvc_dpath / 'annotations/drop6/region_models'
    else:
        region_model_dpath = ub.Path(config['region_model_dpath'])

    site_model_fpaths = list(site_model_dpath.glob('*.geojson'))
    region_model_fpaths = list(region_model_dpath.glob('*.geojson'))

    #  hack to remove known invalid site models
    if 0:
        site_model_fpaths = [s for s in site_model_fpaths if 'Rxxx' not in s.stem]

    print('Validate Data Content')
    validate_data_contents(region_model_fpaths, site_model_fpaths)

    print('Validate Schemas')
    validate_schemas(site_model_fpaths, region_model_fpaths, site_model_schema,
                     region_model_schema)

    print('Sam Script')
    sam_script(site_model_fpaths, region_model_fpaths, site_schema, region_schema)


def sam_script(site_model_fpaths, region_model_fpaths, site_schema, region_schema):
    for test_file in site_model_fpaths + region_model_fpaths:
        try:
            model = json.load(ub.Path(test_file).open())
        except FileNotFoundError:
            print("Can't find the file {:s}".format(test_file))
            continue
        except json.decoder.JSONDecodeError:
            print("Can't load the JSON content of file {:s}".format(test_file))
            continue

        try:
            feature0 = model['features'][0]
        except IndexError:
            print('No features in the file {:s}'.format(test_file))
            continue

        try:
            feature_properties = feature0['properties']
        except KeyError:
            print('Properties missing from the first feature of {:s}'.format(test_file))
            continue

        try:
            feature_type = feature_properties['type']
        except KeyError:
            print('No feature type specified in the first feature of {:s}'.format(test_file))
            continue

        try:
            if feature_type in ['region', 'site_summary']:
                jsonschema.validate(instance=model, schema=region_schema)
            elif feature_type in ['site', 'observation']:
                jsonschema.validate(instance=model, schema=site_schema)
            else:
                raise ValueError(feature_type)
        except jsonschema.ValidationError as e:
            print('Validation failed for {}'.format(test_file))
            print(e)
            continue
        except ValueError:
            print('No type inferrable from the feature type {:s}'.format(feature_type))
            continue

        print('{} is valid'.format(test_file))


def validate_schemas(site_model_fpaths, region_model_fpaths, site_model_schema,
                     region_model_schema):

    region_errors = []
    prog = ub.ProgIter(region_model_fpaths, desc='check region models')
    for region_model_fpath in prog:
        region_model = json.loads(region_model_fpath.read_text())
        try:
            jsonschema.validate(region_model, region_model_schema)
        except jsonschema.ValidationError as ex:
            error_info = {
                'type': 'region_model_error',
                'name': region_model_fpath.stem,
                'ex': ex,
            }
            region_errors.append(error_info)
            prog.set_description(f'check region models, errors: {len(region_errors)}')

    site_errors = []
    prog = ub.ProgIter(site_model_fpaths, desc='check site models')
    for site_model_fpath in prog:
        site_model = json.loads(site_model_fpath.read_text())
        try:
            jsonschema.validate(site_model, site_model_schema)
        except jsonschema.ValidationError as ex:
            error_info = {
                'type': 'site_model_error',
                'name': site_model_fpath.stem,
                'ex': ex,
            }
            site_errors.append(error_info)
            prog.set_description(f'check site models, errors: {len(site_errors)}')

    print('site_errors = {}'.format(ub.repr2(site_errors, nl=1)))
    print('region_errors = {}'.format(ub.repr2(region_errors, nl=1)))

    print(f'{len(site_errors)} / {len(site_model_fpaths)} site model errors')
    print(f'{len(region_errors)} / {len(region_model_fpaths)} region model errors')


def validate_site_dataframe(site_df):
    from dateutil.parser import parse
    import numpy as np
    dummy_start_date = '1970-01-01'  # hack, could be more robust here
    dummy_end_date = '2101-01-01'
    first = site_df.iloc[0]
    rest = site_df.iloc[1:]
    assert first['type'] == 'site', 'first row must have type of site'
    assert first['region_id'] is not None, 'first row must have a region id'
    assert rest['type'].apply(lambda x: x == 'observation').all(), (
        'rest of row must have type observation')
    assert rest['region_id'].apply(lambda x: x is None).all(), (
        'rest of row must have region_id=None')

    site_start_date = first['start_date'] or dummy_start_date
    site_end_date = first['end_date'] or dummy_end_date
    site_start_datetime = parse(site_start_date)
    site_end_datetime = parse(site_end_date)

    if site_end_datetime < site_start_datetime:
        print('\n\nBAD SITE DATES:')
        print(first)

    # Check datetime errors in observations
    try:
        obs_dates = [None if x is None else parse(x) for x in rest['observation_date']]
        obs_isvalid = [x is None for x in obs_dates]
        valid_obs_dates = list(ub.compress(obs_dates, obs_isvalid))
        if not all(valid_obs_dates):
            # null_obs_sites.append(first[['site_id', 'status']].to_dict())
            pass
        valid_deltas = np.array([d.total_seconds() for d in np.diff(valid_obs_dates)])
        assert (valid_deltas >= 0).all(), 'observations must be sorted temporally'
    except AssertionError as ex:
        print('ex = {!r}'.format(ex))
        print(site_df)
        raise


def validate_region_model_content(region_df):
    from dateutil.parser import parse
    import os
    is_region = region_df['type'] == 'region'
    region_part = region_df[is_region]
    assert len(region_part) == 1, 'must have exactly one region in each region file'
    assert region_part['region_id'].apply(lambda x: x is not None).all(), 'regions must have region ids'

    region_rel_fpath = region_df.fpath.relative_to(region_df.fpath.parent.parent.parent)

    region_row = region_part.iloc[0]
    region_id = region_row['region_id']
    region_stem = region_df.fpath.stem

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
                'filename': region_df.fpath.name,
            }
        })

    dummy_start_date = '1970-01-01'  # hack, could be more robust here
    dummy_end_date = '2101-01-01'

    # Hack to set all region-ids
    sites_part = region_df[~is_region]
    assert (sites_part['type'] == 'site_summary').all(), 'rest of data must be site summaries'
    assert sites_part['region_id'].apply(lambda x: x is None).all(), ('site-summaries do not have region ids')

    region_start_date = region_row['start_date'] or dummy_start_date
    region_end_date = region_row['end_date'] or dummy_end_date

    region_start_datetime = parse(region_start_date)
    region_end_datetime = parse(region_end_date)
    if region_end_datetime < region_start_datetime:
        errors.append(f'Bad region dates: {region_start_datetime=}, {region_end_datetime=}')

    # Check datetime errors
    sitesum_start_dates = sites_part['start_date'].apply(lambda x: parse(x or region_start_date))
    sitesum_end_dates = sites_part['end_date'].apply(lambda x: parse(x or region_end_date))
    has_bad_time_range = sitesum_start_dates > sitesum_end_dates

    bad_date_rows = sites_part[has_bad_time_range]
    if len(bad_date_rows):
        bad_row_info = bad_date_rows[['site_id', 'start_date', 'end_date', 'originator']].to_dict('records')
        errors.append({
            'description': 'Site summary rows with start_dates > end_date',
            'offending_rows': bad_row_info,
        })
    if not region_report['errors']:
        region_report['errors'] = None
    region_report['num_errors'] = len(errors)
    return region_report


def validate_site_content(site_df):
    from dateutil.parser import parse
    import os
    dummy_start_date = '1970-01-01'  # hack, could be more robust here
    dummy_end_date = '2101-01-01'
    first = site_df.iloc[0]
    rest = site_df.iloc[1:]
    assert first['type'] == 'site', 'first row must have type of site'
    assert first['region_id'] is not None, 'first row must have a region id'
    assert rest['type'].apply(lambda x: x == 'observation').all(), (
        'rest of row must have type observation')
    assert rest['region_id'].apply(lambda x: x is None).all(), (
        'rest of row must have region_id=None')

    site_start_date = first['start_date'] or dummy_start_date
    site_end_date = first['end_date'] or dummy_end_date
    site_start_datetime = parse(site_start_date)
    site_end_datetime = parse(site_end_date)

    rel_fpath = site_df.fpath.relative_to(site_df.fpath.parent.parent.parent)

    errors = []
    site_report = {
        'region_id': first['region_id'],
        'site_id': first['site_id'],
        'rel_fpath': os.fspath(rel_fpath),
        'errors': errors,
    }

    if site_end_datetime < site_start_datetime:
        offending = first[['site_id', 'start_date', 'end_date', 'originator']].to_dict()
        errors.append({
            'description': 'Site summary row has a start_date > end_date',
            'offending_info': offending,
        })

    # Check datetime errors in observations
    null_obs_sites = []
    import numpy as np
    try:
        obs_dates = [None if x is None else parse(x) for x in rest['observation_date']]
        obs_isvalid = [x is not None for x in obs_dates]
        valid_obs_dates = list(ub.compress(obs_dates, obs_isvalid))
        if not all(valid_obs_dates):
            null_obs_sites.append(first[['site_id', 'status']].to_dict())
        valid_deltas = np.array([d.total_seconds() for d in np.diff(valid_obs_dates)])
        assert (valid_deltas >= 0).all(), 'observations must be sorted temporally'
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


def validate_data_contents(region_model_fpaths, site_model_fpaths):
    import geopandas as gpd
    # Start reading sites while we are doing other work
    site_read_pool = ub.JobPool(mode='process', max_workers=8)
    for site_model_fpath in ub.ProgIter(site_model_fpaths, desc='load site models'):
        job = site_read_pool.submit(gpd.read_file, site_model_fpath)
        job.fpath = site_model_fpath

    # Construct the region reports
    region_read_pool = ub.JobPool(mode='thread', max_workers=0)
    for region_model_fpath in ub.ProgIter(region_model_fpaths, desc='load region models'):
        job = region_read_pool.submit(gpd.read_file, region_model_fpath)
        job.fpath = region_model_fpath

    # Finish reading region models and build reports
    region_models = []
    region_reports = []
    for job in region_read_pool.as_completed(desc='collect region models'):
        region_df = job.result()
        region_df.fpath = job.fpath
        region_report = validate_region_model_content(region_df)
        region_models.append(region_df)
        region_reports.append(region_report)
    region_reports = sorted(region_reports, key=lambda x: (x['num_errors'], x['region_id']))
    print('region_reports = {}'.format(ub.repr2(region_reports, sort=0, nl=-1)).replace('\'', '"').replace('None', 'null'))

    # Finish reading site models, build reports while this is happening.
    site_reports = []
    site_models = []
    for job in site_read_pool.as_completed(desc='collect site models'):
        site_df = job.result()
        site_df.fpath = job.fpath
        site_report = validate_site_content(site_df)
        site_reports.append(site_report)
        site_models.append(site_df)

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
    print('site_reports = {}'.format(ub.repr2(grouped_site_reports, sort=0, nl=-1)).replace('\'', '"').replace('None', 'null'))


if __name__ == '__main__':
    main()
