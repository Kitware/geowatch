import os
import sys
import argparse
import re
from uuid import uuid4
import glob
import traceback

import pystac
from shapely.geometry import shape

from watch.datacube.registration.s2_coreg_l1c import (
    s2_coregister_all_tiles, s2_coregister)
from watch.datacube.registration.l8_coreg_l1 import (
    l8_coregister)
from watch.gis.sensors.sentinel2 import s2_grid_tiles_for_geometry
from watch.utils.util_stac import parallel_map_items, maps


# TODO: Fully specify or re-use something already in WATCH module?
S2_L1C_RE = re.compile(r'S2[AB]_MSI.*')
S2_L1C_GRANULE_RE = re.compile(r'S2[AB]_.*_L(1C|2A)')
L8_L1_RE = re.compile(r'^L[COTEM]08_L(1(TP|GT|GS)|2SP)_\d{3}\d{3}_\d{4}\d{2}\d{2}_\d{4}\d{2}\d{2}_\d{2}_(RT|T1|T2)')  # noqa
GRANULE_DIR_RE = re.compile(r'(.*)/GRANULE/(.*)')
BAND_NAME_RE = re.compile(r'.*(B[0-9A-Z]+|S[EO][AZ][0-9]?|[SV][ZA]A|cloudmask|QA_PIXEL|QA_RADSAT)\.(tiff?|vrt)$', re.I)  # noqa


SUPPORTED_S2_PLATFORMS = {'S2A',
                          'S2B',
                          'sentinel-2a',
                          'sentinel-2b'}
SUPPORTED_LS_PLATFORMS = {'LANDSAT_8'}


def main():
    parser = argparse.ArgumentParser(
        description="Report changes in detected objects")

    parser.add_argument('stac_catalog',
                        type=str,
                        help="Path to input STAC catalog")
    parser.add_argument("-o", "--outdir",
                        type=str,
                        help="Output directory for coregistered scenes and "
                             "updated STAC catalog")

    parser.add_argument("-j", "--jobs",
                        type=int,
                        default=1,
                        required=False,
                        help="Number of jobs to run in parallel")

    run_s2_coreg_l1c(**vars(parser.parse_args()))

    return 0


def _determine_basedir_for_item(stac_item):
    item_base_dir = None
    for link in stac_item.get_links('self'):
        item_base_dir = os.path.dirname(link.get_href())
        break

    if item_base_dir is None:
        raise RuntimeError("Couldn't determine STAC item basedir")

    return item_base_dir


def compute_baseline_scenes_only(stac_catalog, outdir):
    if isinstance(stac_catalog, str):
        catalog = pystac.read_file(href=stac_catalog).full_copy()
    else:
        catalog = stac_catalog.full_copy()

    s2_item_dirs = []
    for item in catalog.get_all_items():
        if item.properties['platform'] not in SUPPORTED_S2_PLATFORMS:
            continue

        s2_item_dirs.append(_determine_basedir_for_item(item))

    if len(s2_item_dirs) > 0:
        scenes, baseline_scenes = s2_coregister_all_tiles(
            list(s2_item_dirs),
            outdir,
            dry_run=True,  # Don't actually coregister
            granuledirs_input=True)

    return baseline_scenes


def run_s2_coreg_l1c(stac_catalog, outdir, jobs=1):
    baseline_scenes = compute_baseline_scenes_only(stac_catalog, outdir)

    if isinstance(stac_catalog, str):
        catalog = pystac.read_file(href=stac_catalog).full_copy()
    else:
        catalog = stac_catalog.full_copy()

    output_catalog = parallel_map_items(
        catalog,
        coreg_stac_item,
        max_workers=jobs,
        mode='process' if jobs > 1 else 'serial',
        extra_args=[outdir, baseline_scenes])

    output_catalog.set_self_href(os.path.join(outdir, 'catalog.json'))
    output_catalog.save(catalog_type=pystac.CatalogType.ABSOLUTE_PUBLISHED)


@maps(history_entry='coregistration')
def coreg_stac_item(stac_item, outdir, baseline_scenes):
    platform = stac_item.properties['platform']

    print("* Running coregistration for item: {}".format(stac_item.id))
    if platform in SUPPORTED_S2_PLATFORMS:
        return coreg_s2_stac_item(stac_item, outdir, baseline_scenes)
    elif platform in SUPPORTED_LS_PLATFORMS:
        return coreg_ls_stac_item(stac_item, outdir, baseline_scenes)
    else:
        print("* Warning * Unsupported platform '{}' for "
              "coregistration, skipping!".format(platform))
        return stac_item


def coreg_s2_stac_item(stac_item, outdir, baseline_scenes):
    mgrs_tile = ''.join(
        map(str, (stac_item.properties["sentinel:utm_zone"],
                  stac_item.properties["sentinel:latitude_band"],
                  stac_item.properties["sentinel:grid_square"])))

    item_basedir = _determine_basedir_for_item(stac_item)
    # FIXME: Somewhat loose check for whether or not the item being
    # processed is a baseline scene
    is_baseline = (os.path.basename(item_basedir) ==
                   os.path.basename(baseline_scenes[mgrs_tile]))

    s2_coregister([item_basedir],
                  outdir,
                  baseline_scenes[mgrs_tile],
                  mgrs_tile)

    asset_path_basedir = os.path.join(
        outdir, "T{}".format(mgrs_tile), os.path.basename(item_basedir))
    asset_paths = glob.glob(
        os.path.join(asset_path_basedir, "*.tif"))

    processed_assets = {}
    for asset_path in sorted(asset_paths):
        band_name = re.match(BAND_NAME_RE, asset_path).group(1)

        if band_name == 'cloudmask':
            roles = ['cloudmask']
        else:
            roles = ['data']

        # Is there some proper convention here for S2 asset names?
        processed_assets["image-{}".format(band_name)] =\
            pystac.Asset.from_dict(
                {'href': os.path.abspath(asset_path),
                 'title': os.path.join(os.path.basename(item_basedir),
                                       os.path.basename(asset_path)),
                 'eo:bands': [{'name': band_name}],
                 'roles': roles})

    vrt_asset_paths = glob.glob(
        os.path.join(asset_path_basedir, "*.vrt"))

    for vrt_asset_path in sorted(vrt_asset_paths):
        m = re.match(BAND_NAME_RE, vrt_asset_path)
        if m is None:
            # Shouldn't match temporary VRT filesnames (e.g. *_tmp.vrt)
            continue

        band_name = m.group(1)

        # Is there some proper convention here for S2 asset names?
        processed_assets["image-vrt-{}".format(band_name)] =\
            pystac.Asset.from_dict(
                {'href': os.path.abspath(vrt_asset_path),
                 'title': os.path.join(
                     os.path.basename(item_basedir),
                     os.path.basename(vrt_asset_path)),
                 'eo:bands': [{'name': band_name}],
                 'roles': ['metadata']})

    # Building off of the original STAC item data (but
    # replacing it's assets), which may or may not be the
    # right thing to do here as some of the metadata may no
    # longer be correct.
    stac_item.id = uuid4().hex
    stac_item.assets = processed_assets

    # Adding mgrs extension information
    stac_item.properties['mgrs:utm_zone'] = mgrs_tile[0:2]
    stac_item.properties['mgrs:latitude_band'] = mgrs_tile[2]
    stac_item.properties['mgrs:grid_square'] = mgrs_tile[3:5]

    # Adding WATCH specific metadata to the STAC item
    # properties; we could formalize this at some point by
    # specifying a proper STAC extension, but I don't think
    # this is necessary for now
    stac_item.properties['watch:s2_coreg_l1c:is_baseline'] =\
        is_baseline

    return stac_item


def coreg_ls_stac_item(stac_item, outdir, baseline_scenes):
    item_geometry = shape(stac_item.geometry)
    mgrs_tiles = s2_grid_tiles_for_geometry(item_geometry)

    item_basedir = _determine_basedir_for_item(stac_item)

    output_stac_items = []
    for mgrs_tile in set(mgrs_tiles).intersection(baseline_scenes.keys()):
        try:
            l8_coregister(mgrs_tile,
                          item_basedir,
                          outdir,
                          baseline_scenes[mgrs_tile])
        except Exception:
            print("Couldn't coregister scene: [{}] for {}, skipping!".format(
                os.path.basename(item_basedir), mgrs_tile))
            traceback.print_exception(*sys.exc_info())
            continue

        l8_scene_image_base = os.path.basename(item_basedir)

        asset_path_basedir = os.path.join(
            outdir, "T{}".format(mgrs_tile),
            "{}_T{}".format(l8_scene_image_base, mgrs_tile))
        asset_paths = glob.glob(
            os.path.join(asset_path_basedir, "*.tif"))

        processed_assets = {}
        for asset_path in sorted(asset_paths):
            band_name = re.match(BAND_NAME_RE, asset_path).group(1)

            if band_name == 'cloudmask':
                roles = ['cloudmask']
            else:
                roles = ['data']

            # Is there some proper convention here for S2 asset names?
            processed_assets["image-{}".format(band_name)] =\
                pystac.Asset.from_dict(
                    {'href': os.path.abspath(asset_path),
                     'title': os.path.join(l8_scene_image_base,
                                           os.path.basename(asset_path)),
                     'eo:bands': [{'name': band_name}],
                     'roles': roles})

        vrt_asset_paths = glob.glob(
            os.path.join(asset_path_basedir, "*.vrt"))

        for vrt_asset_path in sorted(vrt_asset_paths):
            m = re.match(BAND_NAME_RE, vrt_asset_path)
            if m is None:
                # Shouldn't match temporary VRT filesnames (e.g. *_tmp.vrt)
                continue

            band_name = m.group(1)

            # Is there some proper convention here for S2 asset names?
            processed_assets["image-vrt-{}".format(band_name)] =\
                pystac.Asset.from_dict(
                    {'href': os.path.abspath(vrt_asset_path),
                     'title': os.path.join(
                         l8_scene_image_base,
                         os.path.basename(vrt_asset_path)),
                     'eo:bands': [{'name': band_name}],
                     'roles': ['metadata']})

        # Building off of the original STAC item data (but
        # replacing it's assets), which may or may not be the
        # right thing to do here as some of the metadata may no
        # longer be correct.
        processed_item = stac_item.clone()
        processed_item.id = uuid4().hex
        processed_item.assets = processed_assets

        # Adding mgrs extension information
        processed_item.properties['mgrs:utm_zone'] = mgrs_tile[0:2]
        processed_item.properties['mgrs:latitude_band'] = mgrs_tile[2]
        processed_item.properties['mgrs:grid_square'] = mgrs_tile[3:5]

        # Adding WATCH specific metadata to the STAC item
        # properties; we could formalize this at some point by
        # specifying a proper STAC extension, but I don't think
        # this is necessary for now
        processed_item.properties['watch:s2_coreg_l1c:is_baseline'] =\
            False

        output_stac_items.append(processed_item)

    return output_stac_items


if __name__ == "__main__":
    sys.exit(main())
