import ubelt as ub
import pytest
import kwcoco
from geowatch.cli import coco_align
from geowatch.geoannots import geomodels


def test_coco_align_with_empty_site_summary():
    coco_dset = kwcoco.CocoDataset()

    # Create arguments to the script
    dpath = ub.Path.appdir('geowatch/tests/align_bundle_empty1').delete().ensuredir()
    region_dpath = (dpath / 'region_models').ensuredir()

    # Write a region model with no sites, and run align in "site-summary" mode.
    region = geomodels.RegionModel.random(num_sites=0)
    region_fpath = region_dpath / 'region.geojson'
    region.dump(region_fpath)
    dst_dpath = (dpath / 'dst_bundle').ensuredir()
    dst_fpath = dst_dpath / 'dst.kwcoco.zip'

    kw = {
        'src': coco_dset,
        'dst': dst_fpath,
        'keep': 'img',
        'regions': region_fpath,
        'workers': 2,
        'aux_workers': 2,
        'convexify_regions': True,
        'convexify_regions': True,
        'num_start_frames': 3,
        'num_end_frames': 3,
        'site_summary': True,
        'minimum_size': '128x128@10GSD',
    }
    coco_align.main(cmdline=False, **kw)
    assert dst_fpath.exists()
    dst = kwcoco.CocoDataset(dst_fpath)
    assert dst.n_images == 0

    # Test with empty region policy as raise
    kw['empty_region_policy'] = 'raise'
    with pytest.raises(ValueError):
        coco_align.main(cmdline=False, **kw)


def test_coco_align_with_empty_inputs_coco():
    import kwcoco
    from geowatch.cli import coco_align
    from geowatch.geoannots import geomodels
    coco_dset = kwcoco.CocoDataset()

    # Create arguments to the script
    dpath = ub.Path.appdir('geowatch/tests/align_bundle_empty2').delete().ensuredir()
    region_dpath = (dpath / 'region_models').ensuredir()

    region = geomodels.RegionModel.random()
    region_fpath = region_dpath / 'region.geojson'
    region.dump(region_fpath)
    dst_dpath = (dpath / 'dst_bundle').ensuredir()
    dst_fpath = dst_dpath / 'dst.kwcoco.zip'

    kw = {
        'src': coco_dset,
        'dst': dst_fpath,
        'keep': 'img',
        'regions': region_fpath,
        'workers': 2,
        'aux_workers': 2,
        'convexify_regions': True,
        'convexify_regions': True,
        'num_start_frames': 3,
        'num_end_frames': 3,
        'site_summary': True,
        'minimum_size': '128x128@10GSD',
    }
    coco_align.main(cmdline=False, **kw)
    assert dst_fpath.exists()
    dst = kwcoco.CocoDataset(dst_fpath)
    assert dst.n_images == 0
