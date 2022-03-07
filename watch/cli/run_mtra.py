import argparse
import subprocess
from contextlib import contextmanager
import tempfile
import os
import sys
import itertools
from dateutil.parser import parse
from concurrent.futures import as_completed

import pystac
import ubelt

from watch.utils.util_gdal import GdalOpen
from watch.cli.mtra_preprocess import stac_item_map


SUPPORTED_S2_PLATFORMS = {'S2A',
                          'S2B',
                          'sentinel-2a',
                          'sentinel-2b'}  # Sentinel
SUPPORTED_LS_PLATFORMS = {'OLI_TIRS',
                          'LANDSAT_8'}  # Landsat
SUPPORTED_PLATFORMS = (SUPPORTED_S2_PLATFORMS |
                       SUPPORTED_LS_PLATFORMS)


def main():
    parser = argparse.ArgumentParser(
        description="Preprocess imagery for MTRA algorithm")

    parser.add_argument('stac_catalog',
                        type=str,
                        help="Path to input STAC catalog")
    parser.add_argument("-o", "--outdir",
                        type=str,
                        required=True,
                        help="Output directory for the MTRA harmonized output "
                             "data and updated STAC catalog")
    parser.add_argument("--num_pairs",
                        required=False,
                        type=int,
                        help="Number of best Landsat and S2 pairs to select "
                             "for building the harmonization model")
    parser.add_argument("-g", "--gsd",
                        type=int,
                        default=60,
                        help="Destination GSD for preprocessed images "
                             "(default: 60)")
    parser.add_argument("--remap_cloudmask_to_hls",
                        action='store_true',
                        default=False,
                        help="Remap cloudmask (assumed to be generated from "
                             "fmask) values to HLS quality mask values")
    parser.add_argument("-j", "--jobs",
                        type=int,
                        default=1,
                        required=False,
                        help="Number of jobs to run in parallel")

    run_mtra(**vars(parser.parse_args()))

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


def compute_harmonization(stac_items, outdir):
    os.makedirs(outdir, exist_ok=True)

    with tempfile.TemporaryDirectory() as tmpdirname:
        with change_working_dir(tmpdirname):
            data_dir = os.path.join(tmpdirname, "data")
            os.makedirs(data_dir, exist_ok=True)

            for stac_item in stac_items:
                mtra_preprocessed_asset =\
                    stac_item.get_assets().get('mtra_preprocessed')

                if mtra_preprocessed_asset is None:
                    print("* Warning * No mtra_preprocessed asset for item "
                          "'{}', skipping".format(stac_item.id))
                    continue

                mtra_pre_path = mtra_preprocessed_asset.get_absolute_href()

                mtra_pre_out_base =\
                    os.path.basename(mtra_pre_path).replace('_stacked')

                stacked_outpath = os.path.join(data_dir, mtra_pre_out_base)

                os.link(mtra_pre_path, stacked_outpath)

            with open(os.path.join(tmpdirname, "MTRA_Dir.txt"), 'w') as f:
                print(os.path.join(data_dir, "L*"), file=f)
                print(os.path.join(data_dir, "S*"), file=f)
                print(outdir, file=f)

            subprocess.run(['mainMTRA_HLS'],
                           check=True)

    return (os.path.join(outdir, "MTRAMap", "SlopeMap.tif"),
            os.path.join(outdir, "MTRAMap", "InterceptMap.tif"))


def _get_res_for_bandfile(filepath):
    with GdalOpen(filepath) as ds:
        _, xres, _, _, _, yres = ds.GetGeoTransform()
        return abs(xres), abs(yres)


def _default_s2_item_selector(stac_item):
    return stac_item.properties.get('platform') in SUPPORTED_S2_PLATFORMS


def _ensure_map_at_res(map_file, xres, yres):
    base, ext = os.path.splitext(map_file)

    outpath = "{}_{}x{}{}".format(base, xres, yres, ext)

    if not os.path.isfile(outpath):
        subprocess.run(['gdalwarp',
                        '-overwrite',
                        '-of', 'GTiff',
                        '-tr', str(xres), str(yres),
                        map_file,
                        outpath], check=True)

    return outpath


def apply_harmonization_item_map(stac_item,
                                 outdir,
                                 item_selector,
                                 bands_to_harmonize,
                                 slope_map,
                                 intercept_map):
    # Only process select items
    if not item_selector(stac_item):
        return stac_item

    item_outdir = os.path.join(outdir, stac_item.id)
    os.makedirs(item_outdir, exist_ok=True)

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

    for i, band in enumerate(bands_to_harmonize):
        band_href, asset_name = assets_dict[band]
        band_basename = os.path.basename(band_href)
        band_outpath = os.path.join(item_outdir, band_basename)

        xres, yres = _get_res_for_bandfile(band_href)

        subprocess.run(
            ['gdal_calc.py',
             '--calc="A * S + I"',
             '--outfile={}'.format(band_outpath),
             '-A', band_href,
             '--A_band=1',
             '-S', _ensure_map_at_res(slope_map, xres, yres),
             '--S_band={}'.format(i + 1),
             '-I', _ensure_map_at_res(intercept_map, xres, yres),
             '--I_band={}'.format(i + 1),
             '--type', 'Int16'])

        # Replace asset href with harmonized version
        stac_item.assets[asset_name].href = band_outpath

    stac_item.set_self_href(os.path.join(
        item_outdir,
        "{}.json".format(stac_item.id)))

    # Roughly keeping track of what WATCH processes have been
    # run on this particular item
    stac_item.properties.setdefault(
        'watch:process_history', []).append('mtra_harmonization')

    return stac_item


def apply_harmonization(
        items,
        outdir,
        slope_map,
        intercept_map,
        item_selector=_default_s2_item_selector,
        bands_to_harmonize=['B02', 'B03', 'B04', 'B8A', 'B11', 'B12'],
        jobs=1):
    os.makedirs(outdir, exist_ok=True)

    harmonized_stac_items = []
    executor = ubelt.Executor(mode='process' if jobs > 1 else 'serial',
                              max_workers=jobs)
    harmonization_jobs = [executor.submit(apply_harmonization_item_map,
                                          stac_item,
                                          outdir,
                                          item_selector,
                                          bands_to_harmonize,
                                          slope_map, intercept_map)
                          for stac_item in items]
    for mapped_item in (harmonization_job.result() for harmonization_job
                        in as_completed(harmonization_jobs)):
        harmonized_stac_items.append(mapped_item)

    return harmonized_stac_items


def _item_has_cloudmask(stac_item):
    # Ensure that the STAC item has a cloudmask (needed by MTRA)
    for asset_name, asset in stac_item.assets.items():
        asset_dict = asset.to_dict()
        if('roles' in asset_dict and
           'cloudmask' in asset_dict['roles']):
            return True

    return False


def select_best_pairs(stac_items, num_pairs):
    landsat_items =\
        [item for item in stac_items
         if(item.properties.get('platform') in SUPPORTED_LS_PLATFORMS and
            _item_has_cloudmask(item))]
    sentinel_items =\
        [item for item in stac_items
         if(item.properties.get('platform') in SUPPORTED_S2_PLATFORMS and
            _item_has_cloudmask(item))]

    potential_pairs = []
    for ls_item, s2_item in itertools.product(landsat_items, sentinel_items):
        ls_item_datetime = parse(ls_item.properties['datetime'])
        s2_item_datetime = parse(s2_item.properties['datetime'])

        time_delta = abs(ls_item_datetime - s2_item_datetime)

        potential_pairs.append((ls_item, s2_item, time_delta))

    sorted_potential_pairs = sorted(potential_pairs, key=lambda t: t[2])

    selected_ls_items = set()
    selected_s2_items = set()
    selected_items = []
    selected_pairs = 0
    for ls_item, s2_item, _ in sorted_potential_pairs:
        if selected_pairs >= num_pairs:
            break

        if(ls_item not in selected_ls_items and
           s2_item not in selected_s2_items):
            selected_items.append(ls_item)
            selected_ls_items.add(ls_item)

            selected_items.append(s2_item)
            selected_s2_items.add(s2_item)

            selected_pairs += 1

    print("* Selected {} pairs for computing harmonization model".format(
        selected_pairs))
    return selected_items


def run_mtra(stac_catalog,
             outdir,
             num_pairs=None,
             gsd=60,
             remap_cloudmask_to_hls=False,
             jobs=1):
    if isinstance(stac_catalog, str):
        catalog = pystac.read_file(href=stac_catalog).full_copy()
    elif isinstance(stac_catalog, dict):
        catalog = pystac.Catalog.from_dict(stac_catalog).full_copy()
    else:
        catalog = stac_catalog.full_copy()

    output_catalog = catalog.full_copy()
    output_catalog.clear_items()

    grouped_items = {}
    for item in catalog.get_all_items():
        mgrs_tile = ''.join((item.properties.get('mgrs:utm_zone', '??'),
                             item.properties.get('mgrs:latitude_band', '?'),
                             item.properties.get('mgrs:grid_square', '??')))
        grouped_items.setdefault(mgrs_tile, []).append(item)

    for mgrs_tile, stac_items in grouped_items.items():
        if '?' in mgrs_tile:
            print("* Warning * Couldn't parse MGRS tile ({}) for {} items, "
                  "dropping them from output!".format(
                      mgrs_tile, len(stac_items)))
            continue

        print("* Running MTRA harmonization for MGRS tile {}: "
              "{} items..".format(mgrs_tile, len(stac_items)))

        if num_pairs is not None and num_pairs > 0:
            print("* Selecting best items for harmonization model computation")
            stac_items_for_harmonization_model =\
                select_best_pairs(stac_items, num_pairs)
        else:
            stac_items_for_harmonization_model = stac_items

        print("* Preprocessing items for harmonization model computation")
        preprocess_dir = os.path.join(outdir, '_preprocessed')
        os.makedirs(preprocess_dir, exist_ok=True)
        preprocessed_stac_items = []
        executor = ubelt.Executor(mode='process' if jobs > 1 else 'serial',
                                  max_workers=jobs)
        preprocess_jobs = [executor.submit(stac_item_map, stac_item,
                                           preprocess_dir, gsd,
                                           remap_cloudmask_to_hls)
                           for stac_item in stac_items_for_harmonization_model]
        for mapped_item in (preprocess_job.result() for preprocess_job
                            in as_completed(preprocess_jobs)):
            preprocessed_stac_items.append(mapped_item)

        print("* Computing harmonization model")
        slope_map, intercept_map = compute_harmonization(
            preprocessed_stac_items, os.path.join(outdir, mgrs_tile))

        # Precompute different GSD slope & intercept map files to avoid
        # having difference processes try to do it at the same time.
        # TODO: Use a lock instead
        _ensure_map_at_res(slope_map, 10.0, 10.0)
        _ensure_map_at_res(intercept_map, 20.0, 20.0)
        _ensure_map_at_res(slope_map, 10.0, 10.0)
        _ensure_map_at_res(intercept_map, 20.0, 20.0)

        print("* Applying harmonization model to select items")
        harmonized_items = apply_harmonization(
            stac_items, outdir, slope_map, intercept_map, jobs=jobs)

        for output_item in harmonized_items:
            output_catalog.add_link(
                pystac.Link('item', output_item, pystac.MediaType.JSON))

    output_catalog.set_self_href(os.path.join(outdir, 'catalog.json'))
    output_catalog.save(catalog_type=pystac.CatalogType.ABSOLUTE_PUBLISHED)

    return output_catalog


if __name__ == "__main__":
    sys.exit(main())
