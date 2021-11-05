import argparse
import os
import sys
import shapely.geometry
import ubelt as ub
import numpy as np
from copy import deepcopy
import pystac

from watch.utils.util_stac import parallel_map_items
from watch.datacube.registration.wv_to_s2 import wv_to_s2_coregister
from watch.cli.ortho_wv import maps, associate_msi_pan


def main():
    parser = argparse.ArgumentParser(
        description="Coregister WorldView images to Sentinel-2 images")

    parser.add_argument(
        'wv_catalog',
        type=str,
        help="Path to input STAC catalog with ortho'd WV items")
    parser.add_argument(
        's2_catalog',
        type=str,
        help="Path to input STAC catalog with coregistered S2 items")
    parser.add_argument("-o",
                        "--outdir",
                        type=str,
                        help="Output directory for coregistered images and "
                        "updated STAC catalog")
    parser.add_argument("-j",
                        "--jobs",
                        type=int,
                        default=1,
                        required=False,
                        help="Number of jobs to run in parallel")

    coreg_wv(**vars(parser.parse_args()))

    return 0


def _sanitize_catalog(stac_catalog):
    if isinstance(stac_catalog, str):
        catalog = pystac.read_file(href=stac_catalog).full_copy()
    elif isinstance(stac_catalog, dict):
        catalog = pystac.Catalog.from_dict(stac_catalog).full_copy()
    else:
        catalog = stac_catalog.full_copy()
    return catalog


def coreg_wv(wv_catalog, s2_catalog, outdir, jobs=1):
    '''
    Returns an updated wv_catalog coregistered to the baseline images in the
    already-coregistered s2_catalog.

    Also merges MSI and PAN WV items using a previously established mapping.
    Merged items contain assets from both items and item-level metadata from
    the MSI item.
    '''
    wv_catalog = _sanitize_catalog(wv_catalog)
    s2_catalog = _sanitize_catalog(s2_catalog)

    os.makedirs(outdir, exist_ok=True)

    # Look for the baseline s2 scenes among the s2 stac items
    def _is_baseline(s2_item):
        return ('s2_coreg_l1c' in s2_item.properties['watch:process_history']
                and s2_item.properties['watch:s2_coreg_l1c:is_baseline'])

    baseline_s2_items = list(filter(_is_baseline, s2_catalog.get_all_items()))

    # Establish the MSI-PAN mapping in the WV catalog
    # and remove the paired PAN items so they don't get duplicated
    item_pairs_dct = associate_msi_pan(wv_catalog)
    for item in item_pairs_dct.values():
        wv_catalog.remove_item(
            item.id)  # TODO make sure this works as intended

    output_catalog = parallel_map_items(
        wv_catalog,
        _coreg_map,
        max_workers=jobs,
        mode='process' if jobs > 1 else 'serial',
        extra_kwargs=dict(outdir=outdir,
                          baseline_s2_items=baseline_s2_items,
                          item_pairs_dct=item_pairs_dct))

    output_catalog.set_self_href(os.path.join(outdir, 'catalog.json'))
    output_catalog.save(catalog_type=pystac.CatalogType.ABSOLUTE_PUBLISHED)

    return output_catalog


@maps
def _coreg_map(stac_item, outdir, baseline_s2_items, item_pairs_dct):

    print("* Coregistering WV item: '{}'".format(stac_item.id))

    is_ortho = (stac_item.properties['nitf:image_preprocessing_level'] == '2G'
                or any('ortho_wv' in i
                       for i in stac_item.properties['watch:process_history']))
    if stac_item.properties['constellation'] == 'worldview' and is_ortho:

        # Perform coregistration

        output_stac_item = stac_item.clone()
        msi_fpath = stac_item.assets['data'].href
        out_msi_fpath = NotImplemented()

        pan_fpath = ''
        out_pan_fpath = None
        if stac_item.id in item_pairs_dct:
            pan_item = item_pairs_dct[stac_item.id]
            print("** Merging in PAN WV item: '{}'".format(pan_item.id))
            pan_fpath = pan_item.assets['data']
            out_pan_fpath = NotImplemented()

        wv_to_s2_coregister(
            msi_fpath, band_path(best_match(baseline_s2_items, stac_item)),
            pan_fpath)

    else:
        print("** Not a orthorectified WorldView item, skipping!")
        output_stac_item = stac_item

    return output_stac_item


def band_path(s2_item):
    '''
    Get href to S2 band 4 from STAC item
    '''
    def _is_b04(asset):
        eo_bands = asset.properties.get('eo:bands', [])
        if any(i.get('name') in {'B4', 'B04'} for i in eo_bands):
            return True
        return any(i.get('common_name') == 'red' for i in eo_bands)

    asset_name = next(filter(_is_b04, s2_item.assets))

    try:
        return s2_item.assets[asset_name].href
    except KeyError:
        raise ValueError(
            f'Baseline band B04 not found in S2 item {s2_item.id}')


def best_match(s2_items, wv_item):
    '''
    Return the s2_item with the greatest spatial overlap with the wv_item.

    TODO add datetime bounds to this?
    Can use pygeos to parallelize at the wv_item or catalog level if needed
    '''
    wv_shp = shapely.geometry.asShape(wv_item.geometry)
    return max(s2_items,
               key=lambda item: shapely.geometry.asShape(item.geometry).
               intersection(wv_shp).area)


if __name__ == "__main__":
    sys.exit(main())
