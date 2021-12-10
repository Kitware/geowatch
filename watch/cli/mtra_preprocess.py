import sys
import argparse
import os
import subprocess

import pystac

from watch.utils.util_stac import parallel_map_items


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
                        help="Output directory for data preprocessed for MTRA "
                             "algorithm and updated STAC catalog")
    parser.add_argument("-g", "--gsd",
                        type=int,
                        default=60,
                        help="Destination GSD for preprocessed images "
                             "(default: 60)")
    parser.add_argument("-j", "--jobs",
                        type=int,
                        default=1,
                        required=False,
                        help="Number of jobs to run in parallel")
    parser.add_argument("--remap_cloudmask_to_hls",
                        action='store_true',
                        default=False,
                        help="Remap cloudmask (assumed to be generated from "
                             "fmask) values to HLS quality mask values")

    mtra_preprocess(**vars(parser.parse_args()))

    return 0


def stac_item_map(stac_item, outdir, gsd, remap_cloudmask_to_hls=False):
    platform = stac_item.properties.get('platform')
    if platform is None or platform not in SUPPORTED_PLATFORMS:
        return stac_item

    # Order of selected_bands is important here, as the bands in
    # the output file will be in this order (same bands / ordering
    # as HLS with the addition of the cloudmask)
    if platform in SUPPORTED_S2_PLATFORMS:
        selected_bands = ['B02', 'B03', 'B04', 'B8A', 'B11', 'B12', 'cloudmask']  # NOQA: E501
    elif platform in SUPPORTED_LS_PLATFORMS:
        selected_bands = ['B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'cloudmask']

    output_stac_item = mtra_preprocess_item(
        stac_item, selected_bands, outdir, gsd,
        remap_cloudmask_to_hls=remap_cloudmask_to_hls)

    # Roughly keeping track of what WATCH processes have been
    # run on this particular item
    output_stac_item.properties.setdefault(
        'watch:process_history', []).append('mtra_preprocess')

    return output_stac_item


def mtra_preprocess(stac_catalog,
                    outdir,
                    gsd,
                    jobs=1,
                    remap_cloudmask_to_hls=False):
    if isinstance(stac_catalog, str):
        catalog = pystac.read_file(href=stac_catalog).full_copy()
    elif isinstance(stac_catalog, dict):
        catalog = pystac.Catalog.from_dict(stac_catalog).full_copy()
    else:
        catalog = stac_catalog.full_copy()

    os.makedirs(outdir, exist_ok=True)

    output_catalog = parallel_map_items(
        catalog,
        stac_item_map,
        max_workers=jobs,
        mode='process' if jobs > 1 else 'serial',
        extra_args=[outdir, gsd],
        extra_kwargs={'remap_cloudmask_to_hls': remap_cloudmask_to_hls})

    output_catalog.set_self_href(os.path.join(outdir, 'catalog.json'))
    output_catalog.save(catalog_type=pystac.CatalogType.ABSOLUTE_PUBLISHED)

    return output_catalog


def mtra_preprocess_item(stac_item,
                         selected_bands,
                         outdir,
                         gsd,
                         remap_cloudmask_to_hls=False):
    item_outdir = os.path.join(outdir, stac_item.id)
    os.makedirs(item_outdir, exist_ok=True)

    assets_dict = {}
    for asset_name, asset in stac_item.assets.items():
        asset_dict = asset.to_dict()
        if 'roles' in asset_dict:
            if 'data' in asset_dict['roles']:
                for band_name in asset_dict.get('eo:bands', ()):
                    assets_dict[band_name['name']] = asset_dict['href']
            elif 'cloudmask' in asset_dict['roles']:
                assets_dict['cloudmask'] = asset_dict['href']

    vrts = {}
    for band in selected_bands:
        resampling_method = 'cubic'

        if band == 'cloudmask':
            resampling_method = 'near'

        asset_basename, asset_ext = os.path.splitext(
            os.path.basename(assets_dict[band]))
        vrt_outpath = os.path.join(item_outdir,
                                   '{}.vrt'.format(asset_basename))
        subprocess.run([
            'gdalwarp',
            '-overwrite',
            '-of', 'VRT',
            '-r', resampling_method,
            '-tr', str(abs(gsd)), str(abs(gsd)),
            assets_dict[band],
            vrt_outpath], check=True)

        if band == 'cloudmask' and remap_cloudmask_to_hls:
            print("** Remapping cloudmask to HLS values")
            # The cloud mask is a uint8, 30m GSD raster with the
            # following values [2]:
            #
            # 0: null
            # 1: clear
            # 2: cloud
            # 3: shadow
            # 4: snow
            # 5: water
            # 6-7: unused
            #
            # [2] https://github.com/ubarsc/python-fmask/blob/master/fmask/fmask.py#L82  # noqa
            remapped_cloudmask_outpath = os.path.join(
                item_outdir, 'remapped_cloudmask.tif')
            subprocess.run([
                'gdal_calc.py',
                '-A', vrt_outpath,
                '--outfile', remapped_cloudmask_outpath,
                '--calc',
                '0*(A==1)+32*(A==5)+8*(A==3)+16*(A==4)+2*(A==2)+255*(A==255)',
                '--NoDataValue', '255'], check=True)

            vrt_outpath = remapped_cloudmask_outpath

        vrts[band] = vrt_outpath

    # Create merged file
    output_base = '{}_stack'.format(stac_item.id)
    output_filepath = os.path.join(
        item_outdir,
        '{}.tif'.format(output_base))
    subprocess.run([
        'gdal_merge.py',
        '-of', 'GTiff',
        '-separate',
        '-o', output_filepath,
        *vrts.values()])

    stac_item.add_asset('mtra_preprocessed',
                        pystac.Asset.from_dict(
                            {'href': output_filepath,
                             'title': os.path.join(stac_item.id, output_base),
                             'eo:bands': [{'name': b} for b in selected_bands],
                             'roles': ['data']}))

    processed_item_outpath = os.path.abspath(os.path.join(
        outdir, stac_item.id, "{}.json".format(stac_item.id)))
    stac_item.set_self_href(processed_item_outpath)

    return stac_item


if __name__ == "__main__":
    sys.exit(main())
