"""
Runs tests on a small region over different known stac collections with
different processing levels to ensure we populate kwcoco metadata correctly.


SeeAlso:
    ../watch/stac/stac_search_builder.py
    ../watch/cli/stac_search.py
    ../watch/demo/demo_region.py
"""

import pytest


def run_metadata_test():
    pytest.skip('unfinished')

    import ubelt as ub
    from watch.demo import demo_region
    from watch.utils import util_gis
    region_fpath = demo_region.demo_smart_region_fpath()
    region_gdf = list(util_gis.coerce_geojson_datas(region_fpath))[0]['data']
    region_row = region_gdf[region_gdf['type'] == 'region'].iloc[0]

    demo_dpath = ub.Path.appdir('watch/demo/datasets/smart_test').ensuredir()
    search_fpath = demo_dpath / 'stac_search.json'

    result_fpath = demo_dpath / (region_row['region_id'] + '.input')

    import watch
    repo_dpath = ub.Path(watch.__file__).parent.parent
    secrets_fpath = repo_dpath / 'secrets/secrets'
    if secrets_fpath.exists():
        # hack for environs
        secret_text = secrets_fpath.read_text()
        for line in secret_text.split('\n'):
            if line.startswith('export SMART_STAC_API_KEY'):
                import os
                key = line.split('=')[1].strip()
                os.environ['SMART_STAC_API_KEY'] = key

    from watch.stac import stac_search_builder
    stac_search_builder.main(cmdline=0, **{
        'start_date': region_row['start_date'],
        'end_date': region_row['end_date'],
        'cloud_cover': 40,
        # 'sensors': 'sentinel-s2-l2a-cogs',
        # 'sensors': 'sentinel-s2-l2a-cogs',
        # 'sensors': 'landsat-c2l2-sr',
        'sensors': 'ta1-ls-acc',
        'out_fpath': search_fpath,
    })

    from watch.cli import stac_search
    stac_search.main(cmdline=0, **{
        'region_file': region_fpath,
        'search_json': search_fpath,
        'mode': 'area',
        'verbose': 100,
        'outfile': result_fpath,
    })

    from watch.cli import prepare_ta2_dataset
    prepare_ta2_dataset.main(cmdline=0, **{
        'dataset_suffix': 'testing',
        's3_fpath': [result_fpath],
        'collated': [False],
        'dvc_dpath': demo_dpath,
        'aws_profile': 'iarpa',
        'region_globstr': region_fpath,
        'site_globstr': None,
        'fields_workers': 0,
        'convert_workers': 0,
        'align_workers': 0,
        'cache': 0,
        'ignore_duplicates': 0,
        'visualize': 0,
        'backend': 'serial',
        'run': 0,
    })
