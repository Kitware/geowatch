import argparse
import os
import sys
from glob import glob
import tempfile
import re

import pystac

from watch.datacube.cloud.fmask3 import cloudmask
from watch.stac.util_stac import parallel_map_items


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

    run_fmask(**vars(parser.parse_args()))

    return 0


SENSOR_MAPPING = {'S2A': 'S2',
                  'S2B': 'S2',
                  'sentinel-2a': 'S2',
                  'sentinel-2b': 'S2',
                  'OLI_TIRS': 'LS',
                  'LANDSAT_8': 'LS'}


def run_fmask_for_item(stac_item, outdir):
    sensor = SENSOR_MAPPING.get(stac_item.properties['platform'])
    if sensor is None:
        return stac_item

    # This assumes we're not changing the stac_item ID in any of
    # the mapping functions
    item_outdir = os.path.join(outdir, stac_item.id)
    os.makedirs(item_outdir, exist_ok=True)

    data_asset_hrefs = []
    for asset_name, asset in stac_item.assets.items():
        if ((asset.roles is not None and 'data' in asset.roles) or
           asset_name == 'data' or
           re.search(r'\.(tiff?|jp2)$', asset.href, re.I) is not None):
            data_asset_hrefs.append(asset.href)
        else:
            continue

    if len(data_asset_hrefs) == 0:
        print("* Warning * Couldn't find any data assets for "
              "item '{}', skipping!".format(stac_item.id))
        return stac_item

    assets_root = os.path.commonpath(data_asset_hrefs)
    cloudmask_basename = 'cloudmask.tif'
    # NOTE ** We're putting the output cloud mask in the input
    # item's directory as this makes it easier to hand off to
    # UMD's registration component.
    cloudmask_outpath = os.path.join(assets_root, cloudmask_basename)

    print("* Generating cloudmask for item '{}'".format(stac_item.id))
    if (sensor == 'S2'
       and not os.path.isdir(os.path.join(assets_root, 'IMG_DATA'))):
        # Fake a S2 granuledir structure, if not dealing with a
        # true granuledir ..
        with tempfile.TemporaryDirectory() as tmpdirname:
            tmp_imgdata_dir = os.path.join(tmpdirname, 'IMG_DATA')
            os.makedirs(tmp_imgdata_dir, exist_ok=True)
            for fpath in glob(os.path.join(assets_root, '*.jp2')):
                # python-fmask code is using the following glob:
                # IMG_DATA/*_{band_name}.jp2, so we ensure that we
                # have an underscore in the case that our input
                # band files are simply {band_name}.jp2
                out_fpath = os.path.join(
                    tmp_imgdata_dir,
                    '_{}'.format(os.path.basename(fpath)))
                os.symlink(fpath, out_fpath)
            for xml_fpath in glob(os.path.join(assets_root, '*.xml')):
                out_xml_fpath = os.path.join(
                    tmpdirname, os.path.basename(xml_fpath))
                os.symlink(xml_fpath, out_xml_fpath)

            returned_cloudmask_outpath = cloudmask(
                tmpdirname,
                cloudmask_outpath,
                sensor=sensor)
    else:
        returned_cloudmask_outpath = cloudmask(
            assets_root,
            cloudmask_outpath,
            sensor=sensor)

    print("** Cloudmask written to '{}'".format(
        returned_cloudmask_outpath))

    stac_item.assets['cloudmask'] = pystac.Asset.from_dict(
            {'href': cloudmask_outpath,
             'title': os.path.join(stac_item.id, cloudmask_basename),
             'roles': ['cloudmask']})

    stac_item.set_self_href(os.path.join(
        item_outdir,
        "{}.json".format(stac_item.id)))

    # Roughly keeping track of what WATCH processes have been
    # run on this particular item
    stac_item.properties.setdefault(
        'watch:process_history', []).append('run_fmask')

    return stac_item


def run_fmask(stac_catalog, outdir, jobs=1):
    if isinstance(stac_catalog, str):
        catalog = pystac.read_file(href=stac_catalog).full_copy()
    elif isinstance(stac_catalog, dict):
        catalog = pystac.Catalog.from_dict(stac_catalog).full_copy()
    else:
        catalog = stac_catalog.full_copy()

    os.makedirs(outdir, exist_ok=True)

    output_catalog = parallel_map_items(
        catalog,
        run_fmask_for_item,
        max_workers=jobs,
        mode='process' if jobs > 1 else 'serial',
        extra_args=[outdir])

    output_catalog.set_self_href(os.path.join(outdir, 'catalog.json'))
    output_catalog.save(catalog_type=pystac.CatalogType.ABSOLUTE_PUBLISHED)

    return output_catalog


if __name__ == "__main__":
    sys.exit(main())
