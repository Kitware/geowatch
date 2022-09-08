def test_geotiff_ingest():
    """
    python -m watch.cli.geotiffs_to_kwcoco --geotiff_dpath=test_geotiffs --dst test_geotiffs/raw.kwcoco.json
    """
    import pytest
    import os
    import pathlib
    if not os.environ.get('GIRDER_API_KEY') and not os.environ.get('GIRDER_USERNAME'):
        pytest.skip('This test requires girder credentials')

    # Requires credentials
    import watch
    api_url = 'https://data.kitware.com/api/v1'
    resource_id = '6168a18f2fa25629b92c051c'
    geotiff_dpath = watch.utils.util_girder.grabdata_girder(api_url, resource_id)

    from watch.cli import geotiffs_to_kwcoco
    coco_dset = geotiffs_to_kwcoco.main(**{
        'geotiff_dpath': geotiff_dpath,
        'dst': pathlib.Path(geotiff_dpath) / 'raw.kwcoco.json',
        'strict': True,
    })

    assert coco_dset.n_images == 2
