import argparse
import os
import sys
import shapely.geometry
import ubelt as ub
import numpy as np
from copy import deepcopy

import pystac

import watch
from watch.utils.util_stac import parallel_map_items


def main():
    parser = argparse.ArgumentParser(
        description="Orthorectify WorldView images to a DEM")

    parser.add_argument('stac_catalog',
                        type=str,
                        help="Path to input STAC catalog")
    parser.add_argument("-o", "--outdir",
                        type=str,
                        help="Output directory for orthorectified images and "
                             "updated STAC catalog")
    parser.add_argument("-j", "--jobs",
                        type=int,
                        default=1,
                        required=False,
                        help="Number of jobs to run in parallel")
    parser.add_argument("--as_cog",
                        action='store_true',
                        help='Convert NITFs to COGs')
    parser.add_argument("--te_dems",
                        action='store_true',
                        help='Use IARPA T&E DEMs instead of GTOP30 DEMs')
    parser.add_argument("--to_utm",
                        action='store_true',
                        help='reproject to UTM')
    parser.add_argument("--pansharpen",
                        action='store_true',
                        help='Additionally pan-sharpen any MSI images')
    parser.add_argument("-g", "--gsd",
                        type=float,
                        default=None,
                        required=False,
                        help="")

    ortho_wv(**vars(parser.parse_args()))

    return 0


def ortho_wv(stac_catalog, outdir, jobs=1, as_cog=False, te_dems=False, to_utm=False, pansharpen=False, gsd=None):
    if isinstance(stac_catalog, str):
        catalog = pystac.read_file(href=stac_catalog).full_copy()
    elif isinstance(stac_catalog, dict):
        catalog = pystac.Catalog.from_dict(stac_catalog).full_copy()
    else:
        catalog = stac_catalog.full_copy()

    os.makedirs(outdir, exist_ok=True)

    # output_catalog = catalog.map_items(_item_map)
    orthorectified_catalog = parallel_map_items(
        catalog,
        _item_map,
        max_workers=jobs,
        mode='process' if jobs > 1 else 'serial',
        extra_args=[outdir, as_cog, te_dems, to_utm])

    if pansharpen:
        pansharpened_catalog = parallel_map_items(
            orthorectified_catalog,
            _pan_map,
            max_workers=jobs,
            mode='process' if jobs > 1 else 'serial',
            extra_args=[outdir])
    else:
        pansharpened_catalog = orthorectified_catalog

    if gsd is None:
        output_catalog = pansharpened_catalog
    else:
        output_catalog = parallel_map_items(
            pansharpened_catalog,
            _item_map,
            max_workers=jobs,
            mode='process' if jobs > 1 else 'serial',
            extra_args=[outdir])

    output_catalog.set_self_href(os.path.join(outdir, 'catalog.json'))
    output_catalog.save(catalog_type=pystac.CatalogType.ABSOLUTE_PUBLISHED)

    return output_catalog


def _item_map(stac_item, outdir, as_cog, te_dems, to_utm):
    """
    Args:
        stac_item (pystac.Item)
    """
    # This assumes we're not changing the stac_item ID in any of
    # the mapping functions
    item_outdir = os.path.join(outdir, stac_item.id)
    os.makedirs(item_outdir, exist_ok=True)

    # Adding a reference back to the original STAC
    # item if not already present
    if len(stac_item.get_links('original')) == 0:
        stac_item.links.append(pystac.Link.from_dict(
            {'rel': 'original',
             'href': stac_item.get_self_href(),
             'type': 'application/json'}))

    print("* Orthorectifying WV item: '{}'".format(stac_item.id))

    if stac_item.properties.get('constellation') == 'worldview' and stac_item.properties.get('nitf:image_preprocessing_level') == '1R':
        output_stac_item = orthorectify(
            stac_item, item_outdir, as_cog, te_dems, to_utm=to_utm)
    else:
        print("** Not a 1R WorldView item, skipping!")
        output_stac_item = stac_item

    output_stac_item.set_self_href(os.path.join(
        item_outdir,
        "{}.json".format(output_stac_item.id)))

    # Roughly keeping track of what WATCH processes have been
    # run on this particular item
    output_stac_item.properties.setdefault(
        'watch:process_history', []).append('ortho_wv')

    return output_stac_item


def orthorectify(stac_item, outdir, as_cog, te_dems, to_utm=False):
    """
    Args:
        stac_item (pystac.Item)
    """
    in_fpath = stac_item.assets['data'].href
    out_fpath = os.path.join(outdir, os.path.basename(in_fpath))

    # parametarize?
    blocksize = 64  # FIXME: should this be 256 for TA1?
    compress = 'NONE'

    # NOTE: The names of the driver options depend on the `driver`. The options
    # for the COG driver are given here:
    # https://gdal.org/drivers/raster/cog.html and these differ from
    # options for other drivers (e.g. GTIFF
    # https://gdal.org/drivers/raster/gtiff.html#raster-gtiff)
    if as_cog:
        out_fpath = os.path.splitext(out_fpath)[0] + '.tif'
        driver = '-of cog'
        driver_options = ub.paragraph(
            f'''
            -co TILED=YES
            -co OVERVIEWS=AUTO
            -co BLOCKSIZE={blocksize}
            -co COMPRESS={compress}
            ''')
    else:
        # Regular gtiff driver has no overviews options
        driver = ''
        driver_options = ub.paragraph(
            f'''
            -co TILED=YES
            -co BLOCKXSIZE={blocksize}
            -co BLOCKYSIZE={blocksize}
            -co COMPRESS={compress}
            ''')

    lon, lat = np.concatenate(shapely.geometry.asShape(stac_item.geometry).centroid.xy)
    if te_dems:
        raise NotImplementedError('TODO: point to T&E DEMs')
    else:
        dems = watch.gis.elevation.girder_gtop30_elevation_dem()
        dem_fpath, dem_info = dems.find_reference_fpath(lat, lon)

    # TODO: is this necessary for epsg=utm?
    # https://gis.stackexchange.com/questions/193094/can-gdalwarp-reproject-from-espg4326-wgs84-to-utm
    # '+proj=utm +zone=12 +datum=WGS84 +units=m +no_defs'
    if to_utm:
        epsg = watch.gis.spatial_reference.utm_epsg_from_latlon(lat, lon)
    else:
        epsg = 4326

    cmd_str = ub.paragraph(
        f'''
        gdalwarp
        -multi
        --config GDAL_CACHEMAX 500 -wm 500
        --debug off
        -t_srs EPSG:{epsg} -et 0
        -rpc -to RPC_DEM={dem_fpath}
        {driver}
        {driver_options}
        -overwrite
        {in_fpath} {out_fpath}
        ''')
    ub.cmd(cmd_str, check=True, verbose=0)

    item = deepcopy(stac_item)
    item.assets['data'].href = out_fpath

    if as_cog:
        item.assets['data'].media_type = 'image/vnd.stac.geotiff; cloud-optimized=true'
    else:
        item.assets['data'].media_type = 'image/vnd.stac.geotiff'

    return item


def _pan_map(stac_item_pan, stac_item_msi, outdir):
    '''
    Returns a modified copy of stac_item_msi.

    NOTE: new item will have item.properties['gsd'] updated to match
    stac_item_pan's, but 'nitf:*gsd*' removed as these will be invalid.
    '''
    # to get rgb instead of input 4- or 8-channel, add:
    # -b red_band -b blue_band -b green_band
    # -spat_adjust intersection
    # -co PHOTOMETRIC=RGB
    pan_fpath = stac_item_pan.assets['data'].href
    msi_fpath = stac_item_msi.assets['data'].href
    out_fpath = os.path.join(outdir, os.path.basename(msi_fpath))
    cmd_str = ub.paragraph(f'''
        gdal_pansharpen.py
        {pan_fpath} {msi_fpath} {out_fpath}
        -threads ALL_CPUS
        -co BIGTIFF=IF_NEEDED
        ''')
    ub.cmd(cmd_str, check=True, verbose=0)
    item = deepcopy(stac_item_msi)
    item.href = out_fpath
    item.properties = ub.dict_isect(item.properties, [k for k in item.properties if 'gsd' not in k])
    item.properties['gsd'] = stac_item_pan.properties['gsd']
    return item


def set_gsd(stac_item, outdir):
    raise NotImplementedError


if __name__ == "__main__":
    sys.exit(main())
