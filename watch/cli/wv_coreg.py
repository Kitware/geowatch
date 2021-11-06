import argparse
import os
import sys
import shapely.geometry
from copy import deepcopy
import pystac

from watch.utils.util_stac import parallel_map_items
from watch.datacube.registration.wv_to_s2 import wv_to_s2_coregister
from watch.cli.wv_ortho import maps, associate_msi_pan


def main():
    parser = argparse.ArgumentParser(
        description="Coregister WorldView images to Sentinel-2 images")

    parser.add_argument(
        'wv_catalog',
        type=str,
        help="Path to input STAC catalog with ortho'd WV items")
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
    parser.add_argument("--drop_empty",
                        action='store_true',
                        help='Remove empty items from the catalog after '
                        'coregistration')
    parser.add_argument(
        '--s2_catalog',
        type=str,
        default=None,
        required=False,
        help="Path to input STAC catalog with coregistered S2 items. "
        "If None, look for these in wv_catalog as well.")

    wv_coreg(**vars(parser.parse_args()))

    return 0


def _sanitize_catalog(stac_catalog):
    if isinstance(stac_catalog, str):
        catalog = pystac.read_file(href=stac_catalog).full_copy()
    elif isinstance(stac_catalog, dict):
        catalog = pystac.Catalog.from_dict(stac_catalog).full_copy()
    else:
        catalog = stac_catalog.full_copy()
    return catalog


def wv_coreg(wv_catalog, outdir, jobs=1, drop_empty=False, s2_catalog=None):
    '''
    Returns an updated wv_catalog coregistered to the baseline images in the
    already-coregistered s2_catalog.

    Also merges MSI and PAN WV items using a previously established mapping.
    Merged items contain assets from both items and item-level metadata from
    the MSI item.

    Example:
        >>> # create a dummy catalog of 10 WV PAN-MSI pairs and 1 S2 item
        >>> 
        >>> # xdoctest: +REQUIRES(env:API_KEY)
        >>> from pystac_client import Client
        >>> from watch.cli.wv_ortho import wv_ortho
        >>> from watch.cli.wv_coreg import *
        >>> catalog = Client.open('https://api.smart-stac.com/', headers={
        >>>                           "x-api-key": os.environ['API_KEY']})
        >>> ids = ['21OCT13071603-P1BS-014507674010_01_P002',  # og
        >>>        '21OCT13071603-M1BS-014507674010_01_P002',  # og
        >>>        '21OCT13071602-P1BS-014507674010_01_P001',  # og
        >>>        '21OCT13071602-M1BS-014507674010_01_P001',  # og
        >>>        '21OCT02085712-M1BS-014502876010_01_P002',  #
        >>>        '21OCT02085712-P1BS-014502876010_01_P002',  #
        >>>        '21OCT02085711-M1BS-014502876010_01_P001',  # o
        >>>        '21OCT02085711-P1BS-014502876010_01_P001',  # o
        >>>        '21SEP18105055-P1BS-014395056010_01_P003',  # o
        >>>        '21SEP18105055-M1BS-014395056010_01_P003']  # o
        >>> search = catalog.search(collections=['worldview-nitf'], ids=ids)
        >>> items = list(search.get_items())
        >>> assert len(items) == len(ids)
        >>> # s2_id = 'S2B_31TEJ_20210606_0_L1C'
        >>> s2_id = 'S2A_39QXG_20210605_0_L1C'
        >>> s2_item = next(Client.open(
        >>>         'https://earth-search.aws.element84.com/v0').search(
        >>>         collections=['sentinel-s2-l1c'], ids=[s2_id]).get_items())
        >>> s2_item.properties['watch:s2_coreg_l1c:is_baseline'] = True
        >>> s2_item.properties['watch:process_history'] = ['s2_coreg_l1c']
        >>> items.append(s2_item)
        >>> # remove inaccessible URI
        >>> # https://api.smart-stac.com/collections/worldview-nitf
        >>> for item in items:
        >>>     item.set_collection(None)
        >>>     item.set_parent(None)
        >>>     item.set_root(None)
        >>> catalog_dct = catalog.to_dict()
        >>> catalog_dct['links'] = []
        >>> catalog = pystac.Catalog.from_dict(catalog_dct)
        >>> in_dir = os.path.abspath('wv/in/')
        >>> os.makedirs(in_dir, exist_ok=True)
        >>> def download(asset_name, asset):
        >>>     fpath = os.path.join(in_dir, os.path.basename(asset.href))
        >>>     if not os.path.isfile(fpath):
        >>>         os.system(f'aws s3 cp {asset.href} {fpath} '
        >>>                    '--profile iarpa --request-payer')
        >>>     asset.href = fpath
        >>>     return asset
        >>> catalog.add_items(items)
        >>> catalog = catalog.map_assets(download)
        >>>  
        >>> # run orthorectification and pansharpening
        >>>  
        >>> ortho_dir = os.path.abspath('wv/out/')
        >>> os.makedirs(ortho_dir, exist_ok=True)
        >>> ortho_catalog = os.path.join(ortho_dir, 'catalog.json')
        >>> if not os.path.isfile(ortho_catalog):  # caching
        >>>     ortho_catalog = wv_ortho(catalog, ortho_dir, jobs=16,
        >>>                            te_dems=False, pansharpen=True)
        >>> 
        >>> # run coregistration
        >>> 
        >>> coreg_dir = os.path.abspath('wv/coreg/')
        >>> os.makedirs(coreg_dir, exist_ok=True)
        >>> coreg_catalog = wv_coreg(ortho_catalog, coreg_dir, jobs=1,
        >>>                          drop_empty=True)

    '''
    if s2_catalog is None:
        s2_catalog = wv_catalog
    wv_catalog = _sanitize_catalog(wv_catalog)
    s2_catalog = _sanitize_catalog(s2_catalog)

    os.makedirs(outdir, exist_ok=True)

    # Look for the baseline s2 scenes among the s2 stac items
    def _is_baseline(s2_item):
        return (s2_item.properties['constellation'] == 'sentinel-2' and
                's2_coreg_l1c' in s2_item.properties['watch:process_history']
                and s2_item.properties['watch:s2_coreg_l1c:is_baseline'])

    # TODO does this need parallelized?
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
                          item_pairs_dct=item_pairs_dct,
                          drop_empty=drop_empty))

    output_catalog.set_self_href(os.path.join(outdir, 'catalog.json'))
    output_catalog.save(catalog_type=pystac.CatalogType.ABSOLUTE_PUBLISHED)

    return output_catalog


@maps
def _coreg_map(stac_item, outdir, baseline_s2_items, item_pairs_dct,
               drop_empty):

    print("* Coregistering WV item: '{}'".format(stac_item.id))

    is_wv_ortho = (
        stac_item.properties['constellation'] == 'worldview'
        and (stac_item.properties['nitf:image_preprocessing_level'] == '2G'
             or any('wv_ortho' in i
                    for i in stac_item.properties['watch:process_history'])))
    if is_wv_ortho:

        # Setup new STAC item with expected output paths
        # TODO keep the old assets in the new item?

        output_stac_item = stac_item.clone()

        def build_fpaths(fpath):
            # Setup output paths expected from coreg
            base = os.path.splitext(
                os.path.join(outdir, os.path.basename(fpath)))[0]
            out_fpath = base + '_coreg.tif'
            vrt_fpath = base + '_coreg.vrt'
            return out_fpath, vrt_fpath

        fpaths = {}

        # bit of a misnomer, this could alse be an unpaired PAN
        msi_fpath = stac_item.assets['data'].href
        out_msi_fpath, vrt_msi_fpath = build_fpaths(msi_fpath)
        fpaths['data'] = (out_msi_fpath, vrt_msi_fpath)

        pan_fpath = ''
        if stac_item.id in item_pairs_dct:

            pan_item = item_pairs_dct[stac_item.id]
            print("** Merging in PAN WV item: '{}'".format(pan_item.id))
            dct = pan_item.to_dict()
            dct.pop('assets')
            output_stac_item.properties['pan'] = dct
            output_stac_item.assets['data_pan'] = deepcopy(
                pan_item.assets['data'])

            pan_fpath = pan_item.assets['data'].href
            out_pan_fpath, vrt_pan_fpath = build_fpaths(pan_fpath)
            fpaths['data_pan'] = out_pan_fpath, vrt_pan_fpath

        # Perform coregistration

        wv_to_s2_coregister(
            msi_fpath, band_path(best_match(baseline_s2_items, stac_item)),
            outdir, pan_fpath)

        # Perform coregistration on pansharpened image

        if 'data_pansharpened' in output_stac_item.assets:

            ps_fpath = output_stac_item.assets['data_pansharpened'].href
            out_ps_fpath, vrt_ps_fpath = build_fpaths(ps_fpath)
            fpaths['data_pansharpened'] = out_ps_fpath, vrt_ps_fpath

            copy_coreg(ps_fpath, vrt_msi_fpath, out_ps_fpath, vrt_ps_fpath)

        # Update assets and error checking - coreg could have failed due to
        # not enough GCPs found

        def update_asset(item, asset_key, out_fpath, vrt_fpath, drop_empty):
            # Add coreg output to stac item
            backup_vrt_fpath = vrt_fpath.replace('_coreg', '')
            vrt_key = '_'.join(['vrt'] + asset_key.split('_')[1:])

            if os.path.isfile(out_fpath) and os.path.isfile(vrt_fpath):
                # add successful output
                item.assets[asset_key].href = out_fpath
                item.assets[vrt_key] = deepcopy(item.assets[asset_key])
                item.assets[vrt_key].href = vrt_fpath

            elif drop_empty:
                # remove the asset
                item.assets.pop(asset_key)

            elif os.path.isfile(backup_vrt_fpath):
                # just add the vrt of the original file that was created
                item.assets[vrt_key] = deepcopy(item.assets[asset_key])
                item.assets[vrt_key].href = backup_vrt_fpath

            return item

        for asset_key, (out_fpath, vrt_fpath) in fpaths.items():
            output_stac_item = update_asset(output_stac_item, asset_key,
                                            out_fpath, vrt_fpath, drop_empty)

        if drop_empty and not set(fpaths).issubset(output_stac_item.assets):
            print("** WV item is empty after orthorectification, dropping!")
            output_stac_item = None

    else:
        print("** Not a orthorectified WorldView item, skipping!")
        output_stac_item = stac_item

    return output_stac_item


def band_path(s2_item):
    '''
    Get href to S2 band 4 from STAC item
    '''
    def _is_b04(asset):
        eo_bands = asset.extra_fields.get('eo:bands', [])
        if any(i.get('name') in {'B4', 'B04'} for i in eo_bands):
            return True
        return any(i.get('common_name') == 'red' for i in eo_bands)

    try:
        return next(filter(_is_b04, s2_item.assets.values())).href
    except StopIteration:
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


def copy_coreg(in_fpath, vrt_fpath, out_fpath, out_vrt_fpath):
    '''
    wv_to_s2_coregister operates on up to 2 images. But we could have a third-
    the pansharpened MSI. If this exists, its extent is a strict subset of the
    originals', so "copy over" the coreg transform by rewriting a coreg VRT.
    '''
    backup_vrt_fpath = vrt_fpath.replace('_coreg', '')
    pass


if __name__ == "__main__":
    sys.exit(main())
