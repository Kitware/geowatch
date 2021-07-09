import os
import sys
import argparse
import re
from uuid import uuid4
import glob

import pystac

from watch.datacube.registration.s2_coreg_l1c import (
    s2_coregister_all_tiles)


# TODO: Fully specify or re-use something already in WATCH module?
S2_L1C_RE = re.compile(r'S2[AB]_MSIL1C_.*')
GRANULE_DIR_RE = re.compile(r'(.*)/GRANULE/(.*)')
BAND_NAME_RE = re.compile(r'.*_(B\w+)\.(tiff?|vrt)', re.I)


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

    run_s2_coreg_l1c(**vars(parser.parse_args()))

    return 0


def run_s2_coreg_l1c(stac_catalog, outdir):
    if isinstance(stac_catalog, str):
        catalog = pystac.read_file(href=stac_catalog)
    else:
        catalog = stac_catalog

    s2_l1c_items = {}
    for item in catalog.get_all_items():
        if re.match(S2_L1C_RE, item.id):
            # Get base directory for assets
            item_base_dir = None
            for link in item.links:
                if link.rel == 'self':
                    item_base_dir = os.path.dirname(link.get_href())
                    break

            if item_base_dir is None:
                raise RuntimeError("Couldn't determine STAC item basedir")

            s2_l1c_items[item_base_dir] = item.id

    s2_l1c_item_dirs = set(s2_l1c_items.keys())

    os.makedirs(outdir, exist_ok=True)
    scenes, baseline_scenes = s2_coregister_all_tiles(
        list(s2_l1c_item_dirs),
        outdir)

    catalog_outpath = os.path.abspath(os.path.join(outdir, 'catalog.json'))
    catalog.set_self_href(catalog_outpath)

    original_item_ids = set()
    baseline_images = set(baseline_scenes.values())
    for scene, scene_images in scenes.items():
        for scene_image in scene_images:
            is_baseline = scene_image in baseline_images

            match = re.match(GRANULE_DIR_RE, scene_image)
            if match is not None:
                scene_image_base_dir, granule_id = match.groups()
            else:
                raise RuntimeError("Unexpected scene output path returned")

            scene_image_item_id = s2_l1c_items[scene_image_base_dir]
            original_item_ids.add(scene_image_item_id)
            original_item = catalog.get_item(scene_image_item_id)

            asset_path_basedir = os.path.join(
                outdir, "T{}".format(scene), granule_id)
            asset_paths = glob.glob(
                os.path.join(asset_path_basedir, "*.tif"))

            processed_assets = {}
            for asset_path in sorted(asset_paths):
                band_name = re.match(BAND_NAME_RE, asset_path).group(1)

                # Is there some proper convention here for S2 asset names?
                processed_assets["image-{}".format(band_name)] =\
                    pystac.Asset.from_dict(
                        {'href': os.path.abspath(asset_path),
                         'title': os.path.join(granule_id,
                                               os.path.basename(asset_path)),
                         'eo:bands': [{'name': band_name}],
                         'roles': ['data']})

            vrt_asset_paths = glob.glob(
                os.path.join(asset_path_basedir, "*.vrt"))

            for vrt_asset_path in sorted(vrt_asset_paths):
                band_name = re.match(BAND_NAME_RE, vrt_asset_path).group(1)

                # Is there some proper convention here for S2 asset names?
                processed_assets["image-vrt-{}".format(band_name)] =\
                    pystac.Asset.from_dict(
                        {'href': os.path.abspath(vrt_asset_path),
                         'title': os.path.join(
                             granule_id,
                             os.path.basename(vrt_asset_path)),
                         'eo:bands': [{'name': band_name}],
                         'roles': ['metadata']})

            # Generate a unique random name
            new_id = uuid4().hex
            processed_item_outpath = os.path.abspath(os.path.join(
                outdir, new_id, "{}.json".format(new_id)))

            # Building off of the original STAC item data (but
            # replacing it's assets), which may or may not be the
            # right thing to do here as some of the metadata may no
            # longer be correct.
            processed_item = original_item.clone()
            processed_item.id = new_id
            processed_item.assets = processed_assets

            # Adding WATCH specific metadata to the STAC item
            # properties; we could formalize this at some point by
            # specifying a proper STAC extension, but I don't think
            # this is necessary for now
            processed_item.properties['watch:s2_coreg_l1c:is_baseline'] =\
                is_baseline

            # Roughly keeping track of what WATCH processes have been
            # run on this particular item
            processed_item.properties.setdefault(
                'watch:process_history', []).append('s2_coreg_l1c')

            # Adding a reference back to the previous unprocessed STAC
            # item
            processed_item.links.append(pystac.Link.from_dict(
                {'rel': 'previous',
                 'href': original_item.get_self_href(),
                 'type': 'application/json'}))

            processed_item.set_self_href(processed_item_outpath)
            pystac.write_file(processed_item,
                              include_self_link=True,
                              dest_href=processed_item_outpath)

            catalog.add_item(processed_item)

    # Removing the original STAC items from the catalog
    for original_item_id in original_item_ids:
        catalog.remove_item(original_item_id)

    pystac.write_file(catalog,
                      include_self_link=True,
                      dest_href=catalog_outpath)

    return catalog


if __name__ == "__main__":
    sys.exit(main())
