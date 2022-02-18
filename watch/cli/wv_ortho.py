import argparse
import os
import sys
import shapely.geometry
import ubelt as ub
import numpy as np
from copy import deepcopy
import shutil
import pystac

from osgeo_utils.gdal_pansharpen import gdal_pansharpen

import watch
from watch.utils.util_stac import parallel_map_items, maps, associate_msi_pan
from watch.utils.util_gdal import gdal_single_warp


def main():
    parser = argparse.ArgumentParser(
        description='Orthorectify WorldView images to a DEM, '
        'and merge matching PAN and MSI items.')

    parser.add_argument('stac_catalog',
                        type=str,
                        help='Path to input STAC catalog')
    parser.add_argument('-o',
                        '--outdir',
                        type=str,
                        help='Output directory for orthorectified images and '
                        'updated STAC catalog')
    parser.add_argument('-j',
                        '--jobs',
                        type=int,
                        default=1,
                        required=False,
                        help='Number of jobs to run in parallel')
    parser.add_argument('--no_te_dems',
                        dest='te_dems',
                        action='store_false',
                        help='Use GTOP30 DEMs instead of IARPA T&E DEMs')
    parser.add_argument('--drop_empty',
                        action='store_true',
                        help='Remove empty items from the catalog after '
                        'orthorectification')
    parser.add_argument('--as_vrt',
                        action='store_true',
                        help='use VRT instead of COG as ortho output format')
    parser.add_argument('--as_utm',
                        action='store_true',
                        help='Use UTM instead of WGS84 as ortho output CRS')
    parser.add_argument('--skip_ortho',
                        action='store_true',
                        help='Skip orthorectification.')
    parser.add_argument('--pansharpen',
                        action='store_true',
                        help='Additionally pan-sharpen any MSI images')
    parser.add_argument('--as_rgb',
                        action='store_true',
                        help='Create pansharpened image as 3-band RGB instead '
                        'of all MSI bands')

    wv_ortho(**vars(parser.parse_args()))

    return 0


def wv_ortho(stac_catalog,
             outdir,
             jobs=1,
             te_dems=True,
             drop_empty=False,
             as_vrt=False,
             as_utm=False,
             skip_ortho=False,
             pansharpen=False,
             as_rgb=False):
    '''
    Performs the following steps.

    - Orthorectifies WV images
        - Converts them from NTF to COG
        - Converts them from variable GSD to constant (per-image) GSD
        - Removes dependency on a DEM, converting to fixed CRS (UTM or WGS84)
    - For each MSI WV image, if a matching PAN image is available,
      adds a pansharpened image as a new Asset in the MSI Item.
    Example:
        >>> #
        >>> # create a dummy catalog of 10 local items
        >>> #
        >>> # xdoctest: +REQUIRES(env:API_KEY)
        >>> from pystac_client import Client
        >>> from watch.cli.wv_ortho import *
        >>> catalog = Client.open('https://api.smart-stac.com/', headers={
        >>>                           "x-api-key": os.environ['API_KEY']})
        >>> ids = ['21OCT13071603-P1BS-014507674010_01_P002',
        >>>        '21OCT13071603-M1BS-014507674010_01_P002',
        >>>        '21OCT13071602-P1BS-014507674010_01_P001',
        >>>        '21OCT13071602-M1BS-014507674010_01_P001',
        >>>        '21OCT02085712-M1BS-014502876010_01_P002',
        >>>        '21OCT02085712-P1BS-014502876010_01_P002',
        >>>        '21OCT02085711-M1BS-014502876010_01_P001',
        >>>        '21OCT02085711-P1BS-014502876010_01_P001',
        >>>        '21SEP18105055-P1BS-014395056010_01_P003',
        >>>        '21SEP18105055-M1BS-014395056010_01_P003']
        >>> search = catalog.search(collections=['worldview-nitf'], ids=ids)
        >>> items = list(search.get_items())
        >>> assert len(items) == len(ids)
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
        >>>     if not os.path.isfile(fpath):
        >>>         os.system(f'aws s3 cp {asset.href} {fpath} '
        >>>                    '--profile iarpa')
        >>>     asset.href = fpath
        >>>     return asset
        >>> catalog.add_items(items)
        >>> catalog = catalog.map_assets(download)
        >>> catalog.save(catalog_type=pystac.CatalogType.ABSOLUTE_PUBLISHED)
        >>> #
        >>> # run orthorectification and pansharpening
        >>> #
        >>> out_dir = os.path.abspath('wv/out/')
        >>> os.makedirs(out_dir, exist_ok=True)
        >>> out_catalog = wv_ortho(catalog, out_dir, jobs=10, drop_empty=True,
        >>>                        te_dems=False, pansharpen=True)


    '''
    if isinstance(stac_catalog, str):
        catalog = pystac.read_file(href=stac_catalog).full_copy()
    elif isinstance(stac_catalog, dict):
        catalog = pystac.Catalog.from_dict(stac_catalog).full_copy()
    else:
        catalog = stac_catalog.full_copy()

    os.makedirs(outdir, exist_ok=True)

    if skip_ortho:
        orthorectified_catalog = catalog
    else:
        orthorectified_catalog = parallel_map_items(
            catalog,
            ortho_map,
            max_workers=jobs,
            mode='process' if jobs > 1 else 'serial',
            extra_kwargs=dict(outdir=outdir,
                              te_dems=te_dems,
                              drop_empty=drop_empty,
                              as_vrt=as_vrt,
                              as_utm=as_utm))

    if pansharpen:
        pansharpened_catalog = parallel_map_items(
            orthorectified_catalog,
            _pan_map,
            max_workers=jobs,
            mode='process' if jobs > 1 else 'serial',
            extra_kwargs=dict(
                outdir=outdir,
                item_pairs_dct=associate_msi_pan(orthorectified_catalog),
                as_rgb=as_rgb))
    else:
        pansharpened_catalog = orthorectified_catalog

    output_catalog = pansharpened_catalog

    output_catalog.set_self_href(os.path.join(outdir, 'catalog.json'))
    output_catalog.save(catalog_type=pystac.CatalogType.ABSOLUTE_PUBLISHED)

    return output_catalog


@maps(history_entry='wv_ortho')
def ortho_map(stac_item, outdir, drop_empty=False, *args, **kwargs):
    def is_empty(fpath):
        '''
        Check for a failed gdalwarp resulting in an image of all zeros

        This is expensive.
        '''
        import rasterio
        try:
            with rasterio.open(fpath) as f:
                return len(np.unique(f.read().flat)) <= 1
        except rasterio.RasterioIOError:
            return True

    print("* Orthorectifying WV item: '{}'".format(stac_item.id))

    if (stac_item.properties.get('constellation') == 'worldview'
            and stac_item.properties.get('nitf:image_preprocessing_level')
            == '1R'):

        in_fpath = ub.Path(stac_item.assets['data'].href)
        out_fpath = ub.Path(outdir) / in_fpath.name
        geometry = shapely.geometry.shape(stac_item.geometry)

        # warning: can change out_fpath's extension
        out_fpath = orthorectify(in_fpath, out_fpath, geometry, *args,
                                 **kwargs)

        output_stac_item = deepcopy(stac_item)
        output_stac_item.assets['data'].href = str(out_fpath)

        if drop_empty and is_empty(output_stac_item.assets['data'].href):
            output_stac_item = None
            print("** WV item is empty after orthorectification, dropping!")
    else:
        print("** Not a 1R WorldView item, skipping!")
        output_stac_item = stac_item

    return output_stac_item


def orthorectify(in_fpath, out_fpath, geometry: shapely.geometry.Polygon,
                 te_dems: bool, as_vrt: bool, as_utm: bool):
    '''
    Orthorectify a WV image to a DEM using RPCs.

    Example:
        >>> # xdoctest: +SKIP
        >>> import seaborn_image as isns
        >>> import matplotlib.pyplot as plt
        >>> from pystac import Item
        >>> import ubelt as ub
        >>> import os
        >>> import rasterio
        >>> import shapely.geometry
        >>> from watch.utils.util_raster import mask, ResampledRaster
        >>> from watch.cli.wv_ortho import orthorectify
        >>> stac_item = Item.from_file(
        >>>     'original_18JAN07023100-M1BS-014525269010_01_P001.json')
        >>> href = ub.Path(stac_item.assets['data'].href)
        >>> in_fpath = ub.Path(href.name)
        >>> if not in_fpath.is_file():
        >>>    os.system(f'aws s3 cp {href} {in_fpath} --profile iarpa')
        >>> with ResampledRaster(in_fpath, scale=0.1) as f:
        >>>     arr1 = f.read()
        >>>     img1 = f
        >>> geometry = shapely.geometry.shape(stac_item.geometry)
        >>> out_fpath = ub.Path('ortho') / ('gtx' + in_fpath.name)
        >>> out_fpath = orthorectify(in_fpath, out_fpath, geometry,
        >>>                          te_dems=False, as_vrt=False, as_utm=True)
        >>> with ResampledRaster(out_fpath, scale=0.1) as f:
        >>>     arr2 = f.read()
        >>>     img2 = f
        >>> out_fpath = ub.Path('ortho') / ('gtxte' + in_fpath.name)
        >>> out_fpath = orthorectify(in_fpath, out_fpath, geometry,
        >>>                          te_dems=True, as_vrt=False, as_utm=True)
        >>> with ResampledRaster(out_fpath, scale=0.1) as f:
        >>>     arr3 = f.read()
        >>>     img3 = f
        >>> m1 = mask(img1.files[0], save=False, as_poly=False)
        >>> m2 = mask(img2.files[0], save=False, as_poly=False)
        >>> m3 = mask(img3.files[0], save=False, as_poly=False)
        >>> isns.ImageGrid([m1, m2, m3, arr1[0], arr2[0], arr3[0]],
        >>>                col_wrap=3, axis=0)
        >>> plt.show()
    '''
    if as_vrt:
        out_fpath = ub.Path(out_fpath).with_suffix('.vrt')
    else:
        out_fpath = ub.Path(out_fpath).with_suffix('.tif')

    lon, lat = np.concatenate(geometry.centroid.xy)

    if te_dems:

        # TODO should geoidgrid s_srs be used for DEM too? If so, how?
        # Surprisingly, passing in the WMS xml file just works! No need to crop
        # the DEM to the image first. But see
        # watch.utils.util_raster.open_cropped() for those tests.
        from watch.rc import dem_path
        dem_fpath = dem_path()

    else:

        dems = watch.gis.elevation.girder_gtop30_elevation_dem()
        dem_fpath, dem_info = dems.find_reference_fpath(lat, lon)

    # TODO: is this necessary for epsg=utm?
    # https://gis.stackexchange.com/questions/193094/can-gdalwarp-reproject-from-espg4326-wgs84-to-utm
    # -t_srs '+proj=utm +zone=12 +datum=WGS84 +units=m +no_defs'
    if as_utm:
        epsg = watch.gis.spatial_reference.utm_epsg_from_latlon(lat, lon)
    else:
        epsg = 4326

    gdal_single_warp(in_fpath,
                     out_fpath,
                     local_epsg=epsg,
                     nodata=0,
                     rpcs=True,
                     use_perf_opts=True,
                     as_vrt=as_vrt,
                     use_te_geoidgrid=True,
                     dem_fpath=dem_fpath)
    return out_fpath


@maps
def _pan_map(stac_item, outdir, item_pairs_dct, as_rgb):

    print("* Pansharpening WV item: '{}'".format(stac_item.id))

    if stac_item.id in item_pairs_dct:
        output_stac_item = pansharpen(item_pairs_dct[stac_item.id], stac_item,
                                      outdir, as_rgb)
    else:
        print("** Not a WV MSI image or no paired PAN image, skipping!")
        output_stac_item = stac_item

    return output_stac_item


def pansharpen(stac_item_pan, stac_item_msi, outdir, as_rgb):
    '''
    Returns a modified copy of stac_item_msi with a new asset containing
    the pansharpened image.

    NOTE: Item's 'gsd' and 'nitf:*gsd*' properties will refer to the
    original asset. 'pansharpened_to' asset property enables lookup of new GSD
    '''
    pan_fpath = stac_item_pan.assets['data'].href
    msi_fpath = stac_item_msi.assets['data'].href
    out_fpath = os.path.join(outdir,
                             'pansharpened_' + os.path.basename(msi_fpath))

    # build both cli and python commands, in case there's a difference
    cmd_str = ub.paragraph(f'''
        gdal_pansharpen.py
        -threads ALL_CPUS
        -nodata 0
        -of COG
        -co BLOCKSIZE=256
        -co COMPRESS=NONE
        --config GDAL_CACHEMAX 10%
        -co NUM_THREADS=ALL_CPUS
        {pan_fpath} {msi_fpath} {out_fpath}
        ''')
    # -r nearest? (instead of cubic)
    kwargs = {
        'pan_name': pan_fpath,
        'spectral_ds': [msi_fpath],
        'dst_filename': out_fpath,
        'num_threads': 'ALL_CPUS',
        'nodata_value': 0,
        'driver_name': 'COG',
        'creation_options': {
            'BLOCKSIZE': 256,
            'COMPRESS': 'NONE',
            'NUM_THREADS': 'ALL_CPUS'
        },
    }

    eo_bands = deepcopy(stac_item_msi.assets['data'].properties['eo:bands'])

    if as_rgb:
        try:
            common_names = [dct.get('common_name', None) for dct in eo_bands]
            red_band = common_names.index('red')
            green_band = common_names.index('green')
            blue_band = common_names.index('blue')
        except ValueError:  # some items are missing common_names
            # TODO should really split this by mission
            # GeoEye, QB could be different
            names = [dct['name'] for dct in eo_bands]
            red_band = names.index('B5')
            green_band = names.index('B3')
            blue_band = names.index('B2')

        cmd_str = ' '.join((cmd_str,
                            ub.paragraph(f'''
        -b {red_band + 1} –b {green_band + 1} –b {blue_band + 1}
        -spat_adjust intersection
        -co PHOTOMETRIC=RGB
        ''')))
        kwargs.update({
            'band_nums': [red_band + 1, green_band + 1, blue_band + 1],
            'spat_adjust':
            'intersection',
            'creation_options': {
                **kwargs['creation_options'], 'PHOTOMETRIC': 'RGB'
            },
        })
        eo_bands = [
            eo_bands[red_band], eo_bands[green_band], eo_bands[blue_band]
        ]

    if 1:
        cmd = ub.cmd(cmd_str, check=True, verbose=0)  # noqa
    else:
        gdal_pansharpen(**kwargs)

    item = deepcopy(stac_item_msi)
    if 0:  # don't need to transfer the rest of the assets to outdir
        for asset in item.assets.values():
            shutil.copy(asset.href, outdir)
            asset.href = os.path.join(outdir, os.path.basename(asset.href))

    item.assets['data_pansharpened'] = item.assets['data'].clone()
    item.assets['data_pansharpened'].href = out_fpath
    item.assets['data_pansharpened'].properties['eo:bands'] = eo_bands
    item.assets['data_pansharpened'].properties[
        'pansharpened_to_item'] = stac_item_pan.id
    item.assets['data_pansharpened'].properties[
        'pansharpened_to_gsd'] = stac_item_pan.properties['gsd']

    return item


if __name__ == "__main__":
    sys.exit(main())
