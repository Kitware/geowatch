import argparse
import glob
import os
import sys
import re
from contextlib import contextmanager
import tempfile
import subprocess
import shutil

from s2angs.cli import generate_anglebands
import pystac

from watch.utils.util_stac import parallel_map_items, maps


L7_RE = re.compile(r'^L[COTEM]07')
L8_RE = re.compile(r'^L[COTEM]08')
S2_RE = re.compile(r'^S2[AB]')

LANDSAT_ANGLES_FILE_RE = re.compile(r'^(.*_)ANG\.txt$', re.IGNORECASE)
S2_MTD_TL_FILE_RE = re.compile(r'MTD_TL\.xml$', re.IGNORECASE)


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
    parser.add_argument("-j", "--jobs",
                        type=int,
                        default=1,
                        required=False,
                        help="Number of jobs to run in parallel")

    add_angle_bands(**vars(parser.parse_args()))

    return 0


@maps(history_entry='add_angle_bands')
def add_angle_bands_to_item(stac_item, outdir):

    print("* Generating angle bands for item: '{}'".format(stac_item.id))

    if re.search(L7_RE, stac_item.id):
        output_stac_item = add_angles_landsat(
            stac_item, outdir, 'L7')
    elif re.search(L8_RE, stac_item.id):
        output_stac_item = add_angles_landsat(
            stac_item, outdir, 'L8')
    elif re.search(S2_RE, stac_item.id):
        output_stac_item = add_angles_s2(stac_item, outdir)
    else:
        print("** No angle band generation implemented for item, "
              "skipping!")
        output_stac_item = stac_item

    return output_stac_item


def add_angle_bands(stac_catalog, outdir, jobs=1):
    if isinstance(stac_catalog, str):
        catalog = pystac.read_file(href=stac_catalog).full_copy()
    elif isinstance(stac_catalog, dict):
        catalog = pystac.Catalog.from_dict(stac_catalog).full_copy()
    else:
        catalog = stac_catalog.full_copy()

    os.makedirs(outdir, exist_ok=True)

    output_catalog = parallel_map_items(
        catalog,
        add_angle_bands_to_item,
        max_workers=jobs,
        mode='process' if jobs > 1 else 'serial',
        extra_args=[outdir])

    output_catalog.set_self_href(os.path.join(outdir, 'catalog.json'))
    output_catalog.save(catalog_type=pystac.CatalogType.ABSOLUTE_PUBLISHED)

    return output_catalog


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


def add_angles_landsat(stac_item,
                       item_outdir,
                       landsat_version,
                       selected_bands=['4']):

    supported_landsat_versions = {'L8', 'L7'}
    if landsat_version not in supported_landsat_versions:
        raise NotImplementedError("landsat_version: '{}' not supported, "
                                  "expecting one of: {}".format(
                                      landsat_version,
                                      ", ".join(supported_landsat_versions)))

    angles_file = None
    angles_file_prefix = None
    for asset_name, asset in stac_item.assets.items():
        m = re.match(LANDSAT_ANGLES_FILE_RE, asset.href)
        if m is not None:
            angles_file = asset.href
            angles_file_prefix = m.group(1)
            break

    if angles_file is None:
        return stac_item

    with tempfile.TemporaryDirectory() as tmpdirname:
        with change_working_dir(tmpdirname):
            if landsat_version == 'L8':
                # Command line arguments for `l8_angles`:
                # l8_angles <*_ANG.txt> <AngleType> <SubsampleFactor> -f
                # <FillPixelValue> -b <BandList>
                cmd = ['l8_angles', angles_file, 'BOTH', '1']
                if selected_bands is not None:
                    cmd.extend(['-b', ",".join(selected_bands)])

                subprocess.run(cmd,
                               check=True)

            elif landsat_version == 'L7':
                subprocess.run(['landsat_angles', angles_file], check=True)

        for angle_file in os.listdir(tmpdirname):
            if selected_bands is not None:
                selected_bands_re = r'.*({}).img$'.format(
                    '|'.join(("B{:0>2s}".format(b) for b in selected_bands)))
                if not re.match(selected_bands_re, angle_file):
                    continue

            m = re.search(r'(solar|sensor)', angle_file)
            if m is not None:
                angle_type = m.group(1)
            else:
                print("* Warning * Unexpected file '{}' in angle files dir, "
                      "skipping!".format(angle_file))
                continue

            angle_file_basename, _ = os.path.splitext(
                os.path.basename(angle_file))

            # Convert to COG (original format is HDR)
            azimuth_angle_file_outpath = "{}S{}A4.tif".format(
                angles_file_prefix,
                'O' if angle_type == 'solar' else 'E')

            subprocess.run(['gdal_calc.py',
                            '--calc="A"',
                            '--outfile={}'.format(azimuth_angle_file_outpath),
                            '-A', os.path.join(tmpdirname, angle_file),
                            '--A_band=1'])

            stac_item.assets['{}_azimuth'.format(angle_type)] =\
                pystac.Asset.from_dict(
                    {'href': azimuth_angle_file_outpath,
                     'title': os.path.join(
                         stac_item.id, os.path.basename(
                             azimuth_angle_file_outpath)),
                     'roles': ['metadata']})

            # Convert to COG (original format is HDR)
            zenith_angle_file_outpath = "{}S{}Z4.tif".format(
                angles_file_prefix,
                'O' if angle_type == 'solar' else 'E')

            subprocess.run(['gdal_calc.py',
                            '--calc="A"',
                            '--outfile={}'.format(zenith_angle_file_outpath),
                            '-A', os.path.join(tmpdirname, angle_file),
                            '--A_band=2'])

            stac_item.assets['{}_zenith'.format(angle_type)] =\
                pystac.Asset.from_dict(
                    {'href': zenith_angle_file_outpath,
                     'title': os.path.join(
                         stac_item.id, os.path.basename(
                             zenith_angle_file_outpath)),
                     'roles': ['metadata']})

    return stac_item


def add_angles_s2(stac_item, item_outdir):
    mtd_tl_xml_file = None
    for asset_name, asset in stac_item.assets.items():
        if re.search(S2_MTD_TL_FILE_RE, asset.href):
            mtd_tl_xml_file = asset.href
            break

    mtd_base_dir = os.path.dirname(mtd_tl_xml_file)
    output_path_prefix = os.path.commonpath(
        glob.glob(os.path.join(mtd_base_dir, "**", "*04.jp2"), recursive=True))

    if not os.path.isdir(output_path_prefix):
        output_path_prefix = os.path.dirname(output_path_prefix)

    with tempfile.TemporaryDirectory() as tmpdirname:
        angle_file_paths = generate_anglebands(mtd_tl_xml_file, tmpdirname)

        for angle_file_path in angle_file_paths:
            m = re.search(r'(solar|sensor)_(azimuth|zenith)', angle_file_path)
            if m is not None:
                ss, az = m.groups()

                output_path = "{}/S{}{}4.tif".format(
                    output_path_prefix,
                    'O' if ss == 'solar' else 'E',
                    'A' if az == 'azimuth' else 'Z')

                angle_file_basename, _ = os.path.splitext(
                    os.path.basename(output_path))

                shutil.move(angle_file_path, output_path)

                stac_item.assets[m.group(0)] = pystac.Asset.from_dict(
                    {'href': output_path,
                     'title': os.path.join(stac_item.id, angle_file_basename),
                     'roles': ['metadata']})
            else:
                print("* Warning: unexpected filename pattern for generated "
                      "S2 angle band files")

    return stac_item


if __name__ == "__main__":
    sys.exit(main())
