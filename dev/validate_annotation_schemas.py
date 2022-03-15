"""
Prereq:

    TEST_DPATH=$HOME/tmp/test_smart_schema
    mkdir -p $TEST_DPATH
    cd $TEST_DPATH
    git clone git@smartgitlab.com:TE/annotations.git
    git clone git@smartgitlab.com:infrastructure/docs.git

    pip install jsonschema ubelt -U

References:
    https://smartgitlab.com/TE/annotations/-/issues/17
    https://smartgitlab.com/TE/standards/-/snippets/18
"""

import ubelt as ub
import jsonschema
import json


def main():
    # Point to the cloned repos
    # docs_dpath = ub.Path('docs')
    import watch
    annotations_dpath = ub.Path('annotations')
    annotations_dpath = watch.find_smart_dvc_dpath() / 'annotations'

    # Load schemas
    # site_model_schema_fpath = docs_dpath / ('pages/schemas/site-model.schema.json')
    # region_model_schema_fpath = docs_dpath / ('pages/schemas/region-model.schema.json')
    # site_model_schema = json.loads(site_model_schema_fpath.read_text())
    # region_model_schema = json.loads(region_model_schema_fpath.read_text())
    site_schema = site_model_schema = watch.rc.registry.load_site_model_schema()
    region_schema  = region_model_schema = watch.rc.registry.load_region_model_schema()

    site_model_dpath = annotations_dpath / 'site_models'
    region_model_dpath = annotations_dpath / 'region_models'

    site_model_fpaths = list(site_model_dpath.glob('*.geojson'))
    region_model_fpaths = list(region_model_dpath.glob('*.geojson'))

    validate_schemas(site_model_fpaths, region_model_fpaths, site_model_schema,
                     region_model_schema)

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


def validate_data_contents(region_model_fpaths, site_model_fpaths):
    from dateutil.parser import parse
    import geopandas as gpd
    import numpy as np

    dummy_start_date = '1970-01-01'  # hack, could be more robust here
    dummy_end_date = '2101-01-01'

    site_models = []
    for site_model_fpath in ub.ProgIter(site_model_fpaths, desc='load site models'):
        site_model = gpd.read_file(site_model_fpath)
        site_models.append(site_model)

    region_models = []
    for region_model_fpath in ub.ProgIter(region_model_fpaths, desc='load region models'):
        region_model = gpd.read_file(region_model_fpath)
        region_models.append(region_model)

    null_obs_sites = []

    for site_df in site_models:
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
                null_obs_sites.append(first[['site_id', 'status']].to_dict())
            valid_deltas = np.array([d.total_seconds() for d in np.diff(valid_obs_dates)])
            assert (valid_deltas >= 0).all(), 'observations must be sorted temporally'
        except AssertionError as ex:
            print('ex = {!r}'.format(ex))
            print(site_df)
            raise

    if null_obs_sites:
        print('Warning: sites have null observation dates')
        print('null_obs_sites = {!r}'.format(null_obs_sites))

    for region_df in region_models:
        is_region = region_df['type'] == 'region'
        region_part = region_df[is_region]
        assert len(region_part) == 1, 'must have exactly one region in each region file'
        assert region_part['region_id'].apply(lambda x: x is not None).all(), 'regions must have region ids'

        region_row = region_part.iloc[0]
        region_id = region_row['region_id']

        # Hack to set all region-ids
        sites_part = region_df[~is_region]
        assert (sites_part['type'] == 'site_summary').all(), 'rest of data must be site summaries'
        assert sites_part['region_id'].apply(lambda x: x is None).all(), ('site-summaries do not have region ids')

        region_start_date = region_row['start_date'] or dummy_start_date
        region_end_date = region_row['end_date'] or dummy_end_date

        region_start_datetime = parse(region_start_date)
        region_end_datetime = parse(region_end_date)
        if region_end_datetime < region_start_datetime:
            print(f'\n\nBAD REGION DATES: {region_id=}')
            print(region_row)

        # Check datetime errors
        sitesum_start_dates = sites_part['start_date'].apply(lambda x: parse(x or region_start_date))
        sitesum_end_dates = sites_part['end_date'].apply(lambda x: parse(x or region_end_date))
        has_bad_time_range = sitesum_start_dates > sitesum_end_dates

        bad_dates = sites_part[has_bad_time_range]
        if len(bad_dates):
            print(f'\n\nBAD SITE SUMMARY DATES: {region_id=}')
            print(bad_dates)


if __name__ == '__main__':
    main()
