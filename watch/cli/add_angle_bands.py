import argparse
import os
import sys
import re
from contextlib import contextmanager
import tempfile
import subprocess
import glob

import pystac


L7_RE = re.compile(r'^L[COTEM]07')
L8_RE = re.compile(r'^L[COTEM]08')
S2_RE = re.compile(r'^S2[AB]')

LANDSAT_ANGLES_FILE_RE = re.compile(r'ANG.txt$', re.IGNORECASE)


def main():
    parser = argparse.ArgumentParser(
        description="Align all STAC item data assets to the same CRS")

    parser.add_argument('stac_catalog',
                        type=str,
                        help="Path to input STAC catalog")
    parser.add_argument("-o", "--outdir",
                        type=str,
                        help="Output directory for coregistered scenes and "
                             "updated STAC catalog")

    add_angle_bands(**vars(parser.parse_args()))

    return 0


def add_angle_bands(stac_catalog, outdir):
    if isinstance(stac_catalog, str):
        catalog = pystac.read_file(href=stac_catalog).full_copy()
    elif isinstance(stac_catalog, dict):
        catalog = pystac.Catalog.from_dict(stac_catalog).full_copy()
    else:
        catalog = stac_catalog.full_copy()

    os.makedirs(outdir, exist_ok=True)

    def _item_map(stac_item):
        # This assumes we're not changing the stac_item ID in any of
        # the mapping functions
        item_outdir = os.path.join(outdir, stac_item.id)

        if re.search(L7_RE, stac_item.id):
            return add_angles_l7(stac_item, item_outdir)
        elif re.search(L8_RE, stac_item.id):
            return add_angles_l8(stac_item, item_outdir)
        elif re.search(S2_RE, stac_item.id):
            return add_angles_s2(stac_item, item_outdir)
        else:
            return stac_item

    catalog.normalize_hrefs(outdir)
    output_catalog = catalog.map_items(_item_map)

    output_catalog.save(catalog_type=pystac.CatalogType.ABSOLUTE_PUBLISHED)


@contextmanager
def change_working_dir(destination_dir, create=True):
    previous_wd = os.getcwd()

    if create:
        os.makedirs(destination_dir, exist_ok=True)

    os.chdir(destination_dir)

    try:
        yield
    finally:
        os.chdir(previous_wd)


def add_angles_l7(stac_item, item_outdir):
    angles_file = None
    for asset_name, asset in stac_item.assets.items():
        if re.search(LANDSAT_ANGLES_FILE_RE, asset.href):
            angles_file = asset.href
            break

    if angles_file is None:
        return stac_item

    with tempfile.TemporaryDirectory() as tmpdirname:
        with change_working_dir(tmpdirname):
            subprocess.run(['landsat_angles', angles_file], check=True)

        for angle_file in glob.glob(os.path.join(tmpdirname, "*.img")):
            angle_file_basename, _ = os.path.splitext(
                os.path.basename(angle_file))

            # Convert to COG (original format is HDR)
            angle_file_outpath = os.path.join(
                item_outdir, "{}.tif".format(angle_file_basename))
            subprocess.run(['gdalwarp', '-of', 'COG',
                            angle_file,
                            angle_file_outpath])

            stac_item.assets[angle_file_basename] = pystac.Asset.from_dict(
                {'href': angle_file_outpath,
                 'title': os.path.join(stac_item.id, angle_file_basename),
                 'roles': ['metadata']})

    return stac_item


def add_angles_l8(stac_item, item_outdir):
    return stac_item


def add_angles_s2(stac_item, item_outdir):
    return stac_item


if __name__ == "__main__":
    sys.exit(main())
