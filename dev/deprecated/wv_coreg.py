import argparse
import os
import sys
import shapely.geometry
from copy import deepcopy
from osgeo import gdal, osr
from tempfile import NamedTemporaryFile
import pystac

from watch.stac.util_stac import parallel_map_items, maps
from watch.datacube.registration.wv_to_s2 import (wv_to_s2_coregister,
                                                  gdal_translate_prefix,
                                                  gdalwarp_prefix)
from watch.cli.wv_ortho import associate_msi_pan


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
    if isinstance(stac_catalog, pystac.Catalog):
        return stac_catalog.full_copy()
    elif isinstance(stac_catalog, dict):
        return pystac.Catalog.from_dict(stac_catalog).full_copy()
    else:
        return pystac.read_file(href=stac_catalog).full_copy()


def wv_coreg(wv_catalog, outdir, jobs=1, drop_empty=False, s2_catalog=None):
    '''
    Returns an updated wv_catalog coregistered to the baseline images in the
    already-coregistered s2_catalog.

    Also merges MSI and PAN WV items using a previously established mapping.
    Merged items contain assets from both items and item-level metadata from
    the MSI item.

    Example:
        >>> #
        >>> # create a dummy catalog of 10 WV PAN-MSI pairs and 1 S2 item
        >>> #
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
        >>> catalog_dct = catalog.to_dict()
        >>> catalog_dct['links'] = []
        >>> catalog = pystac.Catalog.from_dict(catalog_dct)
        >>> in_dir = os.path.abspath('wv/in/')
        >>> catalog_fpath = os.path.join(in_dir, 'catalog.json')
        >>> catalog.set_self_href(catalog_fpath)
        >>> os.makedirs(in_dir, exist_ok=True)
        >>> # remove inaccessible URI
        >>> # https://api.smart-stac.com/collections/worldview-nitf
        >>> for item in items:
        >>>     item.set_self_href(os.path.join(in_dir, item.id + '.json'))
        >>>     item.set_collection(None)
        >>>     item.set_parent(catalog)
        >>>     item.set_root(catalog)
        >>> def download(asset_name, asset):
        >>>     fpath = os.path.join(in_dir, os.path.basename(asset.href))
        >>>     if (not os.path.isfile(fpath)
        >>>         and asset.href.startswith('s3://')):
        >>>         os.system(f'aws s3 cp {asset.href} {fpath} '
        >>>                    '--profile iarpa --request-payer')
        >>>     asset.href = fpath
        >>>     return asset
        >>> catalog.add_items(items)
        >>> catalog = catalog.map_assets(download)
        >>> catalog.save(catalog_type=pystac.CatalogType.ABSOLUTE_PUBLISHED)
        >>> #
        >>> # run orthorectification and pansharpening
        >>> #
        >>> ortho_dir = os.path.abspath('wv/out/')
        >>> os.makedirs(ortho_dir, exist_ok=True)
        >>> ortho_catalog = os.path.join(ortho_dir, 'catalog.json')
        >>> if not os.path.isfile(ortho_catalog):  # caching
        >>>     ortho_catalog = wv_ortho(catalog, ortho_dir, jobs=16,
        >>>                            te_dems=False, pansharpen=True)
        >>> #
        >>> # run coregistration
        >>> #
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
        return (s2_item.properties.get('constellation') == 'sentinel-2' and
                'coregistration' in s2_item.properties['watch:process_history']
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
        coreg_map,
        max_workers=jobs,
        mode='process' if jobs > 1 else 'serial',
        extra_kwargs=dict(outdir=outdir,
                          baseline_s2_items=baseline_s2_items,
                          item_pairs_dct=item_pairs_dct,
                          drop_empty=drop_empty))

    output_catalog.set_self_href(os.path.join(outdir, 'catalog.json'))
    output_catalog.save(catalog_type=pystac.CatalogType.ABSOLUTE_PUBLISHED)

    return output_catalog


@maps(history_entry='coregistration')
def coreg_map(stac_item, outdir, baseline_s2_items, item_pairs_dct,
              drop_empty):

    print("* Coregistering WV item: '{}'".format(stac_item.id))

    is_wv_ortho = (
        stac_item.properties.get('constellation') == 'worldview'
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

            copy_coreg(ps_fpath, msi_fpath, vrt_msi_fpath, out_ps_fpath,
                       vrt_ps_fpath)

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
            print("** WV item is empty after coregistration, dropping!")
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
        # Ignore "overview" (TCI) asset
        if('overview' in asset.roles or
           asset.title == 'True color image'):
            return False

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
    wv_shp = shapely.geometry.shape(wv_item.geometry)
    return max(s2_items,
               key=lambda item: shapely.geometry.shape(item.geometry).
               intersection(wv_shp).area)


def copy_coreg(in_fpath, orig_fpath, vrt_fpath, out_fpath, out_vrt_fpath):
    '''
    wv_to_s2_coregister operates on up to 2 images. But we could have a third-
    the pansharpened MSI. If this exists, its extent is a strict subset of the
    originals', so copy over the coreg transform from a reference image.
    '''
    backup_vrt_fpath = vrt_fpath.replace('_coreg', '')

    if os.path.isfile(orig_fpath) and os.path.isfile(vrt_fpath):
        # coreg succeeded
        wv_ds = gdal.Open(in_fpath)
        wv_proj = wv_ds.GetProjectionRef()
        proj_ref = osr.SpatialReference()
        proj_ref.ImportFromWkt(wv_proj)  # Well known format
        proj4 = proj_ref.ExportToProj4()

        xsize = wv_ds.RasterXSize
        ysize = wv_ds.RasterYSize

        wv_gt = wv_ds.GetGeoTransform()
        wv_xres = wv_gt[1]
        wv_yres = abs(wv_gt[5])

        x_min = wv_gt[0]
        y_max = wv_gt[3]
        x_max = wv_gt[0] + wv_gt[1] * xsize
        y_min = wv_gt[3] + wv_gt[5] * ysize

        orig_gt = gdal.Open(orig_fpath).GetGeoTransform()

        # get GCPs in CRS of input image
        gcps = gdal.Open(vrt_fpath).GetGCPs()
        for gcp in gcps:
            x_geo = orig_gt[0] + orig_gt[1] * gcp.GCPPixel
            y_geo = orig_gt[3] + orig_gt[5] * gcp.GCPLine

            gcp.GCPPixel = (x_geo - wv_gt[0]) / wv_gt[1]
            gcp.GCPLine = (y_geo - wv_gt[3]) / wv_gt[5]

        # GCPs might be too long to fit in a shell command, so read them from
        # a text file
        with NamedTemporaryFile(mode='w+') as gcp_file:

            with open(gcp_file.name, 'w') as f:
                f.writelines([
                    f'-gcp {gcp.GCPPixel:.5f} {gcp.GCPLine:.5f} '
                    f'{gcp.GCPX:.5f} {gcp.GCPY:.5f} 0 ' for gcp in gcps
                ])
                f.seek(0)

            com_gdal_translate_prefix = gdal_translate_prefix(
                gcp_file.name, proj4)

            com_gdalwarp_prefix = gdalwarp_prefix(wv_xres, wv_yres, x_min,
                                                  y_min, x_max, y_max, proj4)

            os.system(
                f'{com_gdal_translate_prefix} {in_fpath} {out_vrt_fpath}')
            os.system(f'{com_gdalwarp_prefix} {out_vrt_fpath} {out_fpath}')

    elif os.path.isfile(backup_vrt_fpath):
        # coreg failed
        os.system(f'gdal_translate -of VRT {in_fpath} '
                  f'{out_vrt_fpath.replace("_coreg", "")}')

    else:
        # coreg crashed
        # raise ValueError(f'No coreg result for {vrt_fpath}')
        pass

    pass


if __name__ == "__main__":
    sys.exit(main())
