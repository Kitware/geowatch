import argparse
import subprocess
from contextlib import contextmanager
import tempfile
import os
import sys
import glob
import re

import pystac

from watch.utils.util_gdal import GdalOpen
from watch.stac.util_stac import parallel_map_items


SUPPORTED_S2_PLATFORMS = {'S2A',
                          'S2B',
                          'sentinel-2a',
                          'sentinel-2b'}  # Sentinel
SUPPORTED_LS_PLATFORMS = {'OLI_TIRS',
                          'LANDSAT_8'}  # Landsat
SUPPORTED_PLATFORMS = (SUPPORTED_S2_PLATFORMS |
                       SUPPORTED_LS_PLATFORMS)

# Values are (band_interpretation, gsd_meters)
SUPPORTED_S2_BANDS = {'B02': ('Blue', 10),
                      'B03': ('Green', 10),
                      'B04': ('Red', 10),
                      'B8A': ('NIR', 20),
                      'B11': ('SWIR1', 20),
                      'B12': ('SWIR2', 20)}

# Values are (band_interpretation, gsd_meters)
SUPPORTED_LS_BANDS = {'B2': ('Blue', 30),
                      'B3': ('Green', 30),
                      'B4': ('Red', 30),
                      'B5': ('NIR', 30),
                      'B6': ('SWIR1', 30),
                      'B7': ('SWIR2', 30)}

ANGLE_FILE_RE = re.compile(r'(.*)(_S[EO][ZA][0-9]?\.tif)$', re.I)


def main():
    parser = argparse.ArgumentParser(
        description="Run UConn's BRDF correction algorithm")

    parser.add_argument('stac_catalog',
                        type=str,
                        help="Path to input STAC catalog")
    parser.add_argument("-o", "--outdir",
                        type=str,
                        help="Output directory for BRDF corrected imagery")
    parser.add_argument("-j", "--jobs",
                        type=int,
                        default=1,
                        required=False,
                        help="Number of jobs to run in parallel")

    run_brdf(**vars(parser.parse_args()))

    return 0


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


def _get_res_for_bandfile(filepath):
    with GdalOpen(filepath) as ds:
        _, xres, _, _, _, yres = ds.GetGeoTransform()
        return abs(xres), abs(yres)


def _ensure_angle_maps_at_res(angle_files_prefix, outdir, gsd):
    output_angle_prefix = None
    for angle_filename in glob.glob("{}*".format(angle_files_prefix)):
        m = re.match(ANGLE_FILE_RE, angle_filename)
        if m is None:
            continue

        base, suffix = m.groups()
        base = os.path.join(outdir, os.path.basename(base))

        outpath = "{}_{}{}".format(base, gsd, suffix)

        if output_angle_prefix is None:
            output_angle_prefix = "{}_{}".format(base, gsd)

        if not os.path.isfile(outpath):
            subprocess.run(['gdalwarp',
                            '-overwrite',
                            '-of', 'GTiff',
                            '-tr', str(gsd), str(gsd),
                            angle_filename,
                            outpath], check=True)

    return output_angle_prefix


def brdf_correct_item(stac_item, platform):
    if platform in SUPPORTED_LS_PLATFORMS:
        scale_factor = 0.01
        bands_to_correct = SUPPORTED_LS_BANDS
    elif platform in SUPPORTED_S2_PLATFORMS:
        scale_factor = 1.0
        bands_to_correct = SUPPORTED_S2_BANDS
    else:
        # Shouldn't be processing
        print("* Warning * trying to BRDF correct data from "
              "unsupported platform, skipping!")
        return stac_item

    print("* Running BRDF correction on item '{}'".format(
        stac_item.id))

    assets_dict = {}
    for asset_name, asset in stac_item.assets.items():
        # Want to process the individual band files, not the
        # stacked one from mtra_preprocessed
        if asset_name == 'mtra_preprocessed':
            continue

        asset_dict = asset.to_dict()
        if 'roles' in asset_dict:
            if 'data' in asset_dict['roles']:
                for band_name in asset_dict.get('eo:bands', ()):
                    assets_dict[band_name['name']] =\
                        (asset_dict['href'], asset_name)

    if platform in SUPPORTED_S2_PLATFORMS:
        # HACK: Ensure that we have at least some prefix for out angle
        # band files due to filepath assumptions in BRDF correction
        # code
        for angleband_asset in ('SOZ4', 'SOA4', 'SEZ4', 'SEA4'):
            asset_href, asset_name = assets_dict[angleband_asset]

            print(asset_href)
            if os.path.basename(asset_href).startswith("S"):
                new_asset_href = os.path.join(
                    os.path.dirname(asset_href),
                    "dummyprefix_{}".format(
                        os.path.basename(asset_href)))

                os.symlink(asset_href, new_asset_href)
                assets_dict[angleband_asset] = (new_asset_href, asset_name)

    with tempfile.TemporaryDirectory() as tmpdirname:
        with change_working_dir(tmpdirname):
            with open(os.path.join(
                    tmpdirname, "BRDF_Parameters.txt"), 'w') as f:
                print(scale_factor, file=f)

            for band, (band_interp, band_gsd) in bands_to_correct.items():
                sample_angle_band_file, _ = assets_dict['SOZ4']
                angle_band_xres, angle_band_yres = _get_res_for_bandfile(
                    sample_angle_band_file)

                angle_bands_prefix, _ = sample_angle_band_file.split("_SOZ4")
                if angle_band_xres != band_gsd or angle_band_yres != band_gsd:
                    # Have to resize the angle band files in this case
                    angle_bands_prefix = _ensure_angle_maps_at_res(
                        angle_bands_prefix, tmpdirname, band_gsd)

                angle_bands_dir = os.path.dirname(angle_bands_prefix)
                angle_bands_base = os.path.basename(angle_bands_prefix)

                band_filepath, asset_name = assets_dict[band]
                with open(os.path.join(tmpdirname,
                                       "BRDF_Dir_SingleBand.txt"), 'w') as f:
                    print(band_filepath, file=f)
                    print(angle_bands_dir, file=f)
                    print(angle_bands_base, file=f)
                    print(band_interp, file=f)

                subprocess.run(
                    ['main_BRDF_SingleBand'], check=True)

                # 'BRDFed' version put into same directory as input band file
                brdf_corrected_band_filepath = '_BRDFed'.join(
                    os.path.splitext(band_filepath))

                # Copy nodata values from original band file (to
                # ensure same nodata value is used):
                with GdalOpen(band_filepath) as ds:
                    band = ds.GetRasterBand(1)
                    nodata_value = band.GetNoDataValue()

                subprocess.run([
                    'gdal_calc.py',
                    '-A', band_filepath,
                    '-B', brdf_corrected_band_filepath,
                    f'--calc={nodata_value}*(A=={nodata_value})+B*(A!={nodata_value})',  # noqa
                    '--NoDataValue', str(nodata_value),
                    '--outfile', brdf_corrected_band_filepath,
                    '--quiet',
                    '--hideNoData',
                    '--overwrite'], check=True)

                stac_item.assets[asset_name].href =\
                    brdf_corrected_band_filepath

    # Remove angle band assets from output STAC item as they're no
    # longer needed
    for angleband_asset in ('SOZ4', 'SOA4', 'SEZ4', 'SEA4'):
        _, asset_name = assets_dict[angleband_asset]
        del stac_item.assets[asset_name]

    return stac_item


def brdf_item_map(stac_item, outdir):
    platform = stac_item.properties.get('platform')

    output_stac_item = brdf_correct_item(stac_item, platform)

    stac_item.set_self_href(os.path.join(
        outdir, stac_item.id, "{}.json".format(stac_item.id)))

    # Roughly keeping track of what WATCH processes have been
    # run on this particular item
    output_stac_item.properties.setdefault(
        'watch:process_history', []).append('run_brdf')

    return output_stac_item


def run_brdf(stac_catalog, outdir, jobs=1):
    if isinstance(stac_catalog, str):
        catalog = pystac.read_file(href=stac_catalog).full_copy()
    elif isinstance(stac_catalog, dict):
        catalog = pystac.Catalog.from_dict(stac_catalog).full_copy()
    else:
        catalog = stac_catalog.full_copy()

    os.makedirs(outdir, exist_ok=True)

    output_catalog = parallel_map_items(
        catalog,
        brdf_item_map,
        max_workers=jobs,
        mode='process' if jobs > 1 else 'serial',
        extra_args=[outdir])

    output_catalog.set_self_href(os.path.join(outdir, 'catalog.json'))
    output_catalog.save(catalog_type=pystac.CatalogType.ABSOLUTE_PUBLISHED)

    return output_catalog


if __name__ == "__main__":
    sys.exit(main())
