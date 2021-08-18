import argparse
import os
import sys

import pystac

from watch.datacube.cloud.fmask3 import cloudmask


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

    run_fmask(**vars(parser.parse_args()))

    return 0


def run_fmask(stac_catalog, outdir):
    if isinstance(stac_catalog, str):
        catalog = pystac.read_file(href=stac_catalog).full_copy()
    elif isinstance(stac_catalog, dict):
        catalog = pystac.Catalog.from_dict(stac_catalog).full_copy()
    else:
        catalog = stac_catalog.full_copy()

    os.makedirs(outdir, exist_ok=True)

    sensor_mapping = {'S2A': 'S2',
                      'S2B': 'S2',
                      'OLI_TIRS': 'LS'}

    def _item_map(stac_item):
        sensor = sensor_mapping.get(stac_item.properties['platform'])
        if sensor is None:
            return stac_item

        # This assumes we're not changing the stac_item ID in any of
        # the mapping functions
        item_outdir = os.path.join(outdir, stac_item.id)
        os.makedirs(item_outdir, exist_ok=True)

        data_asset_hrefs = []
        for asset_name, asset in stac_item.assets.items():
            if((asset.roles is None
                or 'data' not in asset.roles)
               and asset_name != 'data'):
                continue
            else:
                data_asset_hrefs.append(asset.href)

        if len(data_asset_hrefs) == 0:
            print("* Warning * Couldn't find any data assets for "
                  "item '{}', skipping!".format(stac_item.id))
            return stac_item

        assets_root = os.path.commonpath(data_asset_hrefs)

        cloudmask_basename = 'cloudmask.tif'
        print("* Generating cloudmask for item '{}'".format(stac_item.id))
        # NOTE ** We're putting the output cloud mask in the input
        # item's directory as this makes it easier to hand off to
        # UMD's registration component.
        cloudmask_outpath = cloudmask(
            assets_root,
            os.path.join(assets_root, cloudmask_basename),
            sensor=sensor)
        cloudmask_outpath = os.path.join(assets_root, cloudmask_basename)
        print("** Cloudmask written to '{}'".format(cloudmask_outpath))

        stac_item.assets['cloudmask'] = pystac.Asset.from_dict(
                {'href': cloudmask_outpath,
                 'title': os.path.join(stac_item.id, cloudmask_basename),
                 'roles': ['cloudmask']})

        # Roughly keeping track of what WATCH processes have been
        # run on this particular item
        stac_item.properties.setdefault(
            'watch:process_history', []).append('run_fmask')

        return stac_item

    catalog.normalize_hrefs(outdir)
    output_catalog = catalog.map_items(_item_map)

    output_catalog.save(catalog_type=pystac.CatalogType.ABSOLUTE_PUBLISHED)

    return output_catalog


if __name__ == "__main__":
    sys.exit(main())
