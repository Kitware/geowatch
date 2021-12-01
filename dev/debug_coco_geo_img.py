"""
/data/joncrall/dvc-repos/smart_watch_dvc/Drop1-Aligned-L1/_test/_debug_regions/debug_20150811T093006_N54.892730E023.897986_N54.972815E024.089080_S2_text.py

    # images = parent_dset.images()
    # wv_images = images.compress([s == 'WV' for s in images.lookup('sensor_coarse')])
    # coco_imgs = wv_images.coco_images[10:11]
"""

debug_info = {
    'coco_fpath': '/home/joncrall/data/dvc-repos/smart_watch_dvc/drop1/data.kwcoco.json',
    'gids': [468],
}


def main():
    import kwcoco
    import kwimage
    from os.path import join
    import ubelt as ub
    from watch.utils import kwcoco_extensions
    from watch.utils import util_raster  # NOQA
    import geopandas as gpd
    from shapely.ops import unary_union
    parent_dset = kwcoco.CocoDataset(debug_info['coco_fpath'])
    coco_imgs = parent_dset.images(debug_info['gids']).coco_images

    # image_summaries = []
    # for coco_img in coco_imgs:
    #     gid = coco_img.img['id']

    image_summaries = []
    for coco_img in coco_imgs:
        gid = coco_img.img['id']

        kwcoco_extensions.coco_populate_geo_img_heuristics(
            parent_dset, gid, overwrite=True, keep_geotiff_metadata=True)

        aux_geo_rows = []
        for obj in coco_img.iter_asset_objs():
            info = obj['geotiff_metadata']
            fpath = join(parent_dset.bundle_dpath, obj['file_name'])

            sh_geos_corners_crs84 = kwimage.Polygon.coerce(obj['geos_corners']).to_shapely()

            kw_valid_poly_utm = kwimage.MultiPolygon.coerce(coco_img.img['valid_region_utm'])
            # sh_valid_poly_pxl = util_raster.mask(fpath, tolerance=10, default_nodata=0, convex_hull=True)
            # kw_valid_poly_pxl = kwimage.MultiPolygon.coerce(sh_valid_poly_pxl)
            # kw_valid_poly_utm = kw_valid_poly_pxl.warp(info['pxl_to_wld']).warp(info['wld_to_utm'])
            sh_valid_poly_utm = kw_valid_poly_utm.to_shapely()

            utm_epsg = int(info['utm_crs_info']['auth'][1])
            aux_geo_rows.append({
                'channels': obj['channels'],
                'geos_corners': sh_geos_corners_crs84,
                'valid_poly_utm': sh_valid_poly_utm,
                'fpath': fpath,
                'utm_epsg': utm_epsg,
            })

        img_df = gpd.GeoDataFrame(aux_geo_rows, crs='crs84', geometry='geos_corners')
        unique_epsg = set(img_df['utm_epsg'])
        assert len(unique_epsg) == 1
        utm_epsg = ub.peek(unique_epsg)

        sh_corner_crs84 = img_df.geos_corners.unary_union
        sh_valid_crs84 = img_df.set_geometry('valid_poly_utm', crs=utm_epsg).to_crs('crs84').unary_union
        image_summaries.append({
            'gid': gid,
            'corners': sh_corner_crs84,
            'valid': sh_valid_crs84,
        })

    corners_crs84 = gpd.GeoDataFrame(image_summaries, geometry='corners', crs='crs84')
    valid_crs84 = corners_crs84.set_geometry('valid')
    bounds_crs84 = gpd.GeoDataFrame({
        'geometry': [unary_union([
            corners_crs84.corners.unary_union,
            valid_crs84.corners.unary_union,
        ]).convex_hull]}, crs='crs84')

    wld_map_crs84_gdf = gpd.read_file(
        gpd.datasets.get_path('naturalearth_lowres')
    ).to_crs('crs84')
    import kwplot
    kwplot.autompl()

    fig = kwplot.figure(fnum=1, doclf=True)
    ax = fig.gca()
    wld_map_crs84_gdf.plot(ax=ax)
    corners_crs84.plot(ax=ax, color='blue', edgecolor='black', linewidth=4, alpha=0.6)
    valid_crs84.plot(ax=ax, color='pink', alpha=0.6)
    bounds = bounds_crs84.geometry.scale(2.5, 2.5).bounds.iloc[0]
    ax.set_xlim(bounds.minx, bounds.maxx)
    ax.set_ylim(bounds.miny, bounds.maxy)
    kwplot.phantom_legend({'valid region': 'pink', 'corners': 'black'}, ax=ax)


def debug_valid_regions_utm():

    parent_dset.images().get('valid_region_utm', None)
    pass


if __name__ == '__main__':
    main()
