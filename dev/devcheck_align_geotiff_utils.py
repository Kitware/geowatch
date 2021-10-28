import kwimage
import ubelt as ub
# from watch.cli.coco_align_geotiffs import *  # NOQA


def coco_geopandas_images(dset):
    """
    TODO:
        - [ ] This is unused in this file and thus should move to the dev
        folder or somewhere else for to keep useful scratch work.
    """
    import geopandas as gpd
    df_input = []
    for gid, img in dset.imgs.items():
        info  = img['geotiff_metadata']
        kw_img_poly = kwimage.Polygon(exterior=info['wgs84_corners'])
        sh_img_poly = kw_img_poly.to_shapely()
        df_input.append({
            'gid': gid,
            'name': img.get('name', None),
            'video_id': img.get('video_id', None),
            'bounds': sh_img_poly,
        })
    img_geos_df = gpd.GeoDataFrame(df_input, geometry='bounds', crs='epsg:4326')
    return img_geos_df


def visualize_rois(dset, kw_all_box_rois):
    """
    matplotlib visualization of image and annotation regions on a world map

    Developer function, unused in the script

    TODO:
        - [ ] This is unused in this file and thus should move to the dev
        folder or somewhere else for to keep useful scratch work.
    """
    import geopandas as gpd
    sh_coverage_rois = find_covered_regions(dset)
    sh_coverage_rois_trad = [flip_xy(p) for p in sh_coverage_rois]
    kw_coverage_rois_trad = list(map(kwimage.Polygon.from_shapely, sh_coverage_rois_trad))
    print('kw_coverage_rois_trad = {}'.format(ub.repr2(kw_coverage_rois_trad, nl=1)))
    cov_poly_crs = 'epsg:4326'
    cov_poly_gdf = gpd.GeoDataFrame({'cov_rois': sh_coverage_rois_trad},
                                    geometry='cov_rois', crs=cov_poly_crs)

    sh_all_box_rois = [p.to_shapely()for p in kw_all_box_rois]
    sh_all_box_rois_trad = [flip_xy(p) for p in sh_all_box_rois]
    kw_all_box_rois_trad = list(map(kwimage.Polygon.from_shapely, sh_all_box_rois_trad))
    roi_poly_crs = 'epsg:4326'
    roi_poly_gdf = gpd.GeoDataFrame({'roi_polys': sh_all_box_rois_trad},
                                    geometry='roi_polys', crs=roi_poly_crs)
    print('kw_all_box_rois_trad = {}'.format(ub.repr2(kw_all_box_rois_trad, nl=1)))

    if True:
        import kwplot
        kwplot.autompl()

        wld_map_gdf = gpd.read_file(
            gpd.datasets.get_path('naturalearth_lowres')
        )
        ax = wld_map_gdf.plot()

        cov_centroids = cov_poly_gdf.geometry.centroid
        cov_poly_gdf.plot(ax=ax, facecolor='none', edgecolor='green', alpha=0.5)
        cov_centroids.plot(ax=ax, marker='o', facecolor='green', alpha=0.5)
        # img_centroids = img_poly_gdf.geometry.centroid
        # img_poly_gdf.plot(ax=ax, facecolor='none', edgecolor='red', alpha=0.5)
        # img_centroids.plot(ax=ax, marker='o', facecolor='red', alpha=0.5)

        roi_centroids = roi_poly_gdf.geometry.centroid
        roi_poly_gdf.plot(ax=ax, facecolor='none', edgecolor='orange', alpha=0.5)
        roi_centroids.plot(ax=ax, marker='o', facecolor='orange', alpha=0.5)

        kw_zoom_roi = kw_all_box_rois_trad[1]
        kw_zoom_roi = kw_coverage_rois_trad[2]
        kw_zoom_roi = kw_all_box_rois_trad[3]

        bb = kw_zoom_roi.bounding_box()

        min_x, min_y, max_x, max_y = bb.scale(1.5, about='center').to_ltrb().data[0]
        ax.set_xlim(min_x, max_x)
        ax.set_ylim(min_y, max_y)


def find_covered_regions(dset):
    """
    Find the intersection of all image bounding boxes in world space
    to see what spatial regions are covered by the imagery.
    """
    from shapely import ops
    gid_to_poly = {}
    for gid, img in dset.imgs.items():
        info  = img['geotiff_metadata']
        kw_img_poly = kwimage.Polygon(exterior=info['wgs84_corners'])
        sh_img_poly = kw_img_poly.to_shapely()
        gid_to_poly[gid] = sh_img_poly

    # df_input = [
    #     {'gid': gid, 'bounds': poly, 'name': dset.imgs[gid].get('name', None),
    #      'video_id': dset.imgs[gid].get('video_id', None) }
    #     for gid, poly in gid_to_poly.items()
    # ]
    # img_geos = gpd.GeoDataFrame(df_input, geometry='bounds', crs='epsg:4326')

    # Can merge like this, but we lose membership info
    # coverage_df = gpd.GeoDataFrame(img_geos.unary_union)

    coverage_rois_ = ops.unary_union(gid_to_poly.values())
    if hasattr(coverage_rois_, 'geoms'):
        # Iteration over shapely objects was deprecated, test for geoms
        # attribute instead.
        coverage_rois = list(coverage_rois_.geoms)
    else:
        coverage_rois = [coverage_rois_]
    return coverage_rois


def flip_xy(poly):
    """
    TODO:
        - [ ] This is unused in this file and thus should move to the dev
        folder or somewhere else for to keep useful scratch work.
    """
    if hasattr(poly, 'reorder_axes'):
        new_poly = poly.reorder_axes((1, 0))
    else:
        kw_poly = kwimage.Polygon.from_shapely(poly)
        kw_poly.data['exterior'].data = kw_poly.data['exterior'].data[:, ::-1]
        sh_poly_ = kw_poly.to_shapely()
        new_poly = sh_poly_
    return new_poly
