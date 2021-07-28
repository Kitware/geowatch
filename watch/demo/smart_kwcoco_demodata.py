"""
"""
from dateutil.parser import isoparse
from os.path import dirname
from os.path import join
from watch.cli import coco_align_geotiffs
from watch.cli import geotiffs_to_kwcoco
from watch.demo import landsat_demodata
from watch.demo import sentinel2_demodata
import datetime
import geopandas as gpd
import kwarray
import kwcoco
import kwimage
import ubelt as ub
import watch


def demo_smart_raw_kwcoco():
    """
    Creates a kwcoco dataset that attempts to exhibit common corner cases with
    special watch geospatial attributes. This is a raws dataset of
    tiles with annotations that are not organized into videos.

    Example:
        >>> from watch.demo.smart_kwcoco_demodata import *  # NOQA
        >>> raw_coco_dset = demo_smart_raw_kwcoco()
        >>> print('raw_coco_dset = {!r}'.format(raw_coco_dset))
    """
    cache_dpath = ub.ensure_app_cache_dir('watch/demo/kwcoco')
    raw_coco_fpath = join(cache_dpath, 'demo_smart_raw.kwcoco.json')
    stamp = ub.CacheStamp('raw_stamp', dpath=cache_dpath, depends=['v1'],
                          product=raw_coco_fpath)
    if stamp.expired():
        s2_demo_paths1 = sentinel2_demodata.grab_sentinel2_product(index=0)
        s2_demo_paths2 = sentinel2_demodata.grab_sentinel2_product(index=1)
        l8_demo_paths = landsat_demodata.grab_landsat_product()

        s2_demo_paths1.images
        s2_demo_paths2.images

        raw_coco_dset = kwcoco.CocoDataset()

        categories = [
            'Active Construction',
            'Site Preparation',
            'Post Construction',
            'Unknown'
        ]
        for catname in categories:
            raw_coco_dset.add_category(catname)

        rng = kwarray.ensure_rng(542370, api='python')

        # Add only the TCI for the first S2 image
        cands = [fname for fname in s2_demo_paths1.images if fname.name.endswith('TCI.jp2')]
        assert len(cands) == 1
        fname = cands[0]
        fpath = str(s2_demo_paths1.path / fname)
        img = geotiffs_to_kwcoco.make_coco_img_from_geotiff(fpath)
        baseinfo = watch.gis.geotiff.geotiff_filepath_info(fpath)
        capture_time = isoparse(baseinfo['filename_meta']['sense_start_time'])
        img['date_captured'] = datetime.datetime.isoformat(capture_time)
        img['sensor_coarse'] = 'S2'
        raw_coco_dset.add_image(**img)
        # Fix issue
        try:
            raw_coco_dset.imgs[1]['warp_pxl_to_wld'] = raw_coco_dset.imgs[1]['warp_pxl_to_wld'].concise()
        except Exception:
            pass

        # Add all bands for the seconds S2 image
        img = geotiffs_to_kwcoco.ingest_sentinel2_directory(str(s2_demo_paths2.path))
        raw_coco_dset.add_image(**img)

        # Add all bands of the L8 image
        lc_dpath = dirname(l8_demo_paths['bands'][0])
        img = geotiffs_to_kwcoco.ingest_landsat_directory(lc_dpath)
        raw_coco_dset.add_image(**img)

        # Add random annotations on each image
        for img in raw_coco_dset.imgs.values():
            utm_corners = kwimage.Polygon(exterior=img['utm_corners']).to_shapely()
            utm_gdf = gpd.GeoDataFrame(
                {'geometry': [utm_corners]},
                geometry='geometry', crs=img['utm_crs_info']['auth'])
            wgs_corners = utm_gdf.to_crs('wgs84').geometry.iloc[0]
            corner_poly = kwimage.Polygon.from_shapely(wgs_corners)

            # Create a dummy annotation which is a scaled down version of the corner points
            dummy_sseg_geos = corner_poly.scale(0.02, about='center')
            random_cat = rng.choice(raw_coco_dset.dataset['categories'])
            raw_coco_dset.add_annotation(
                image_id=img['id'],
                category_id=random_cat['id'],
                segmentation_geos=dummy_sseg_geos.to_geojson())

        raw_coco_dset.fpath = raw_coco_fpath
        raw_coco_dset.dump(raw_coco_dset.fpath, newlines=True)
        stamp.renew()

    raw_coco_dset = kwcoco.CocoDataset(raw_coco_fpath)
    return raw_coco_dset


def demo_smart_aligned_kwcoco():
    """
    This is an aligned dataset of videos

    Example:
        >>> from watch.demo.smart_kwcoco_demodata import *  # NOQA
        >>> aligned_coco_dset = demo_smart_aligned_kwcoco()
        >>> print('aligned_coco_dset = {!r}'.format(aligned_coco_dset))
    """
    cache_dpath = ub.ensure_app_cache_dir('watch/demo/kwcoco')
    aligned_kwcoco_dpath = join(cache_dpath, 'demo_aligned')
    aligned_coco_fpath = join(aligned_kwcoco_dpath, 'data.kwcoco.json')
    stamp = ub.CacheStamp('aligned_stamp', dpath=cache_dpath, depends=['v1'],
                          product=[aligned_coco_fpath])
    if stamp.expired():
        raw_coco_dset = demo_smart_raw_kwcoco()
        coco_align_geotiffs.main(
            src=raw_coco_dset,
            regions='annots',
            max_workers=0,
            context_factor=3.0,
            dst=aligned_kwcoco_dpath,
        )
        stamp.renew()

    aligned_coco_dset = kwcoco.CocoDataset(aligned_coco_fpath)
    return aligned_coco_dset
