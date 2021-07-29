import os
import sys
import argparse
import json
from uuid import uuid4
import itertools

import pystac
from osgeo import gdal, osr

from watch.gis.spatial_reference import utm_epsg_from_latlon


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
    parser.add_argument("--aoi_bounds",
                        type=str,
                        required=True,
                        help="AOI Bounds as a serialized JSON string in the "
                             "form '[min_lon, min_lat, max_lon, max_lat]' "
                             "WGS84 (EPSG4326)")

    align_crs(**vars(parser.parse_args()))

    return 0


def _aoi_bounds_to_utm_zone(aoi_bounds):
    '''
    Return majority EPSG code from aoi_bounds cornerpoints.  Returns
    first EPSG code if there's a tie
    '''
    lon0, lat0, lon1, lat1 = aoi_bounds

    codes = [utm_epsg_from_latlon(lat, lon)
             for lat, lon in itertools.product((lat0, lat1), (lon0, lon1))]

    code_counts = {}
    selected_code = None
    highest_count = 0
    for code in codes:
        code_counts[code] = code_counts.get(code, 0) + 1

        if code_counts[code] > highest_count:
            selected_code = code

    return selected_code


def align_crs(stac_catalog, outdir, aoi_bounds):
    if isinstance(stac_catalog, str):
        catalog = pystac.read_file(href=stac_catalog).full_copy()
    else:
        catalog = stac_catalog.full_copy()

    catalog_outpath = os.path.abspath(os.path.join(outdir, 'catalog.json'))
    catalog.set_self_href(catalog_outpath)
    catalog.set_root(catalog)

    if isinstance(aoi_bounds, str):
        aoi_bounds = json.loads(aoi_bounds)

    epsg_code = _aoi_bounds_to_utm_zone(aoi_bounds)

    output_items = []
    for original_item in catalog.get_all_items():
        new_id = uuid4().hex
        processed_assets = {}
        for asset_name, asset in original_item.assets.items():
            # 'worldview-nitf' from T&E STAC doesn't include roles,
            # but asset_name is "data"
            if((asset.roles is None
                or 'data' not in asset.roles)
               and asset_name != 'data'):
                print("Asset '{}' for item '{}' is not data, skipping "
                      "conversion".format(asset_name, original_item.id))
                continue

            input_path = asset.href
            input_base, _ = os.path.splitext(
                os.path.basename(input_path))

            asset_output_dir = os.path.join(outdir, new_id)
            os.makedirs(asset_output_dir, exist_ok=True)

            output_path = os.path.join(
                asset_output_dir, "{}.tif".format(input_base))

            # Should probably check that the asset isn't already in
            # the destination CRS so that we don't have to process it
            # or throw away metadata
            dst_crs = osr.SpatialReference()
            dst_crs.ImportFromEPSG(epsg_code)
            opts = gdal.WarpOptions(dstSRS=dst_crs, format="COG")
            print("* Warping asset '{}' for item '{}'".format(
                asset_name, original_item.id))
            out = gdal.Warp(output_path, input_path, options=opts)
            del out  # this is necessary, it writes out to disk

            output_asset_dict = {'href': os.path.abspath(output_path),
                                 'title': os.path.join(
                                     new_id,
                                     os.path.basename(output_path)),
                                 'roles': ['data']}

            input_asset_dict = asset.to_dict()
            if 'eo:bands' in input_asset_dict:
                output_asset_dict['eo:bands'] = input_asset_dict['eo:bands']

            processed_assets[asset_name] = pystac.Asset.from_dict(
                output_asset_dict)

        processed_item_outpath = os.path.abspath(os.path.join(
            outdir, new_id, "{}.json".format(new_id)))

        # Building off of the original STAC item data (but
        # replacing it's assets), which may or may not be the
        # right thing to do here as some of the metadata may no
        # longer be correct.
        processed_item = original_item.clone()
        processed_item.id = new_id
        processed_item.assets = processed_assets

        # Roughly keeping track of what WATCH processes have been
        # run on this particular item
        processed_item.properties.setdefault(
            'watch:process_history', []).append('align_crs')

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

        output_items.append(processed_item)

    catalog.clear_items()
    catalog.add_items(output_items)

    pystac.write_file(catalog,
                      include_self_link=True,
                      dest_href=catalog_outpath)

    return catalog


if __name__ == "__main__":
    sys.exit(main())
