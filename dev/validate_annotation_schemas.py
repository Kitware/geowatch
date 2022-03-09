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
"""

import ubelt as ub
import jsonschema
import json


def main():
    # Point to the cloned repos
    docs_dpath = ub.Path('docs')
    annotations_dpath = ub.Path('annotations')

    # Load schemas
    site_model_schema_fpath = docs_dpath / ('pages/schemas/site-model.schema.json')
    region_model_schema_fpath = docs_dpath / ('pages/schemas/region-model.schema.json')
    site_model_schema = json.loads(site_model_schema_fpath.read_text())
    region_model_schema = json.loads(region_model_schema_fpath.read_text())

    site_model_dpath = annotations_dpath / 'site_models'
    site_model_fpaths = list(site_model_dpath.glob('*.geojson'))

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

    region_errors = []
    region_model_dpath = annotations_dpath / 'region_models'
    region_model_fpaths = list(region_model_dpath.glob('*.geojson'))
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

    print('site_errors = {}'.format(ub.repr2(site_errors, nl=1)))
    print('region_errors = {}'.format(ub.repr2(region_errors, nl=1)))

    print(f'{len(site_errors)} / {len(site_model_fpaths)} site model errors')
    print(f'{len(region_errors)} / {len(region_model_fpaths)} region model errors')


if __name__ == '__main__':
    main()
