import watch
import kwcoco
import ubelt as ub
from watch.utils import util_gis
from geowatch.geoannots import geomodels
import kwimage
from kwimage import Color
import pandas as pd
import geopandas as gpd
from watch import heuristics
import fiona

fiona.drvsupport.supported_drivers['kml'] = 'rw'  # enable KML support which is disabled by default
fiona.drvsupport.supported_drivers['KML'] = 'rw'  # enable KML support which is disabled by default

import kwplot  # NOQA
kwplot.autompl()

POSITIVE_STATUS_LABELS = {
    row['status']
    for row in heuristics.HUERISTIC_STATUS_DATA
    if 'positive' in row['status']
}

to_UTM = util_gis.project_gdf_to_local_utm

# Setup paths to data
dvc_expt_dpath = watch.find_dvc_dpath(tags='phase2_expt')
dvc_annot_dpath = watch.find_dvc_dpath(tags='phase2_data', hardware='hdd')
dvc_data_dpath = watch.find_dvc_dpath(tags='drop7_data')
true_rm_dpath = dvc_annot_dpath / 'annotations/drop7/region_models'
true_sm_dpath = dvc_annot_dpath / 'annotations/drop7/site_models'
bundle_dpath = dvc_data_dpath / 'Drop7-StaticACTestSet-2GSD'

new_rm_dpath = (bundle_dpath / 'bas_small_truth/region_models').ensuredir()
new_sm_dpath = (bundle_dpath / 'bas_small_truth/site_models').ensuredir()

bas_rm_dpath = bundle_dpath / 'bas_output/region_models'
new_bas_rm_dpath = bundle_dpath / 'bas_small_output/region_models'

# Use KR_R002 to define a crop box size
ref_region_fpath = true_rm_dpath / 'KR_R002.geojson'
ref_region = geomodels.RegionModel.coerce(ref_region_fpath)
ref_region_crs84_gdf = ref_region.pandas_region()
ref_region_utm_gdf = to_UTM(ref_region_crs84_gdf, mode=1)
ref_box = kwimage.Polygon.coerce(ref_region_utm_gdf.geometry.iloc[0]).box().to_xywh()

if 0:
    # Copy over kr2 true and predicted sites
    ref_name = 'KR_R002'
    (true_rm_dpath / f'{ref_name}.geojson').copy(new_rm_dpath / f'{ref_name}.geojson', follow_file_symlinks=True)
    (bas_rm_dpath / f'{ref_name}.geojson').copy(new_bas_rm_dpath / f'{ref_name}.geojson', follow_file_symlinks=True)
    for p in true_sm_dpath.glob(f'{ref_name}*'):
        p.copy(new_sm_dpath / p.name, follow_file_symlinks=True)

region_stats_rows = []

region_names = ['KW_C001', 'CN_C000', 'CO_C001']
for region_name in region_names:
    true_region_fpath = true_rm_dpath / f'{region_name}.geojson'
    bas_region_fpath = bas_rm_dpath / f'{region_name}.geojson'

    cluster_regions = list(geomodels.RegionModel.coerce_multiple(((bundle_dpath / region_name) / 'clusters')))

    cluster_crs84_gdf = pd.concat([r.pandas_region() for r in cluster_regions])

    bas_region = geomodels.RegionModel.coerce(bas_region_fpath)
    true_region = geomodels.RegionModel.coerce(true_region_fpath)

    bas_ss_utm_gdf = to_UTM(bas_region.pandas_summaries(), mode=1)
    utm_crs = bas_ss_utm_gdf.crs
    true_ss_utm_gdf = true_region.pandas_summaries().to_crs(utm_crs)
    true_rm_utm_gdf = true_region.pandas_region().to_crs(utm_crs)
    cluster_utm_gdf = cluster_crs84_gdf.to_crs(utm_crs)
    region_box = kwimage.Polygon.coerce(true_rm_utm_gdf.geometry.iloc[0]).box()

    # Determine how many 'positive' truth sites have no overlap in BAS
    is_pos = true_ss_utm_gdf['status'].apply(lambda x: x in POSITIVE_STATUS_LABELS)
    pos_true_gdf = true_ss_utm_gdf[is_pos]
    idx1_to_idx2 = util_gis.geopandas_pairwise_overlaps(pos_true_gdf, bas_ss_utm_gdf)
    num_bas_missed_truth = sum([len(idxs) == 0 for idxs in idx1_to_idx2.values()])

    ref_box.data[0:2] = [region_box.center_x, region_box.center_y]

    try:
        manual_region_fpath = f'/home/joncrall/Downloads/{region_name}_R1.kml'
        maual_region_crs84 = gpd.read_file(manual_region_fpath)
    except Exception:
        ref_poly = ref_box.to_polygon()
        raise NotImplementedError('no way to choose the polygon')
    else:
        maual_region_utm = maual_region_crs84.to_crs(true_rm_utm_gdf.crs)
        ref_poly = kwimage.Polygon.coerce(maual_region_utm.iloc[0].geometry)

        ref_poly.to_shapely()

    # Build the polygon where we will keep
    flags = cluster_utm_gdf.intersects(ref_poly.to_shapely())
    keep_clusters = cluster_utm_gdf[flags]
    keep_poly = keep_clusters.unary_union | ref_poly.to_shapely()

    # Determine which clusters / truth sites to keep
    true_flags = true_ss_utm_gdf.intersects(keep_poly)
    keep_true_ss_gdm = true_ss_utm_gdf[true_flags]
    keep_true_site_ids = keep_true_ss_gdm['site_id'].tolist()
    keep_cluster_ids = keep_clusters['region_id'].tolist()

    keep_bas_flags = bas_ss_utm_gdf.intersects(keep_poly)
    keep_bas_ss_gdf = bas_ss_utm_gdf[keep_bas_flags]

    keep_is_pos = keep_true_ss_gdm['status'].apply(lambda x: x in POSITIVE_STATUS_LABELS)
    keep_pos_ss_gdm = keep_true_ss_gdm[is_pos]
    idx1_to_idx2 = util_gis.geopandas_pairwise_overlaps(keep_pos_ss_gdm, keep_bas_ss_gdf)
    num_keep_bas_missed_truth = sum([len(idxs) == 0 for idxs in idx1_to_idx2.values()])

    stat_row = {
        'region_name': region_name,
        'num_bas_missed': num_bas_missed_truth,
        'num_true_sites': len(pos_true_gdf),
        'num_small_bas_missed': num_keep_bas_missed_truth,
        'num_small_true_sites': len(keep_pos_ss_gdm),
    }
    region_stats_rows.append(stat_row)
    print('stat_row = {}'.format(ub.urepr(stat_row, nl=1)))

    if 1:
        # Copy the selected predicted site models over
        keep_bas_site_ids = set(keep_bas_ss_gdf['site_id'].tolist())
        new_bas_rm_dpath.ensuredir()
        new_bas_region_fpath = new_bas_rm_dpath / f'{region_name}.geojson'

        # Filter the region model
        bas_region = geomodels.RegionModel.coerce(bas_region_fpath)
        for feat in list(bas_region.site_summaries()):
            if feat['properties']['site_id'] not in keep_bas_site_ids:
                bas_region['features'].remove(feat)
        new_bas_region_fpath.write_text(bas_region.dumps())

    if 1:

        # Copy the selected true region / site models over
        true_region = geomodels.RegionModel.coerce(true_region_fpath)
        for feat in list(true_region.body_features()):
            if feat['properties']['site_id'] not in keep_true_site_ids:
                true_region['features'].remove(feat)
        new_region_fpath = new_rm_dpath / f'{region_name}.geojson'
        new_region_fpath.write_text(true_region.dumps())

        for site_id in keep_true_site_ids:
            fpath = true_sm_dpath / f'{site_id}.geojson'
            assert fpath.exists()
            new_fpath = new_sm_dpath / fpath.name
            new_fpath.delete()
            fpath.copy(new_fpath, overwrite=True, follow_file_symlinks=True)

    # Filter the kwcoco file
    if 0:
        cluster_coco_fpath = (bundle_dpath / region_name / f'imgonly-{region_name}-rawbands.kwcoco.zip')
        new_cluster_coco_fpath = (bundle_dpath / region_name / f'imgonly-{region_name}-rawbands-small.kwcoco.zip')
        dset = kwcoco.CocoDataset.coerce(cluster_coco_fpath)
        remove_video_ids = [o['id'] for o in dset.videos().objs if o['name'] not in keep_cluster_ids]
        dset.remove_videos(remove_video_ids)
        dset.fpath = new_cluster_coco_fpath
        dset.dump()

    DRAW = 1
    if DRAW:
        ax = kwplot.plt.gca()
        ax.cla()
        bas_ss_utm_gdf.plot(
            ax=ax,
            facecolor=Color.coerce('kitware_blue', alpha=0.5).as01(),
            edgecolor=Color.coerce('kitware_darkblue', alpha=0.5).as01()
        )
        true_ss_utm_gdf.plot(
            ax=ax,
            facecolor=Color('kitware_green', alpha=0.5).as01(),
            edgecolor=Color('black', alpha=0.5).as01(),
        )
        cluster_utm_gdf.plot(
            ax=ax,
            facecolor='none',
            edgecolor=Color('kitware_gray', alpha=0.5).as01(),
        )
        keep_clusters.plot(
            ax=ax,
            facecolor='none',
            edgecolor=Color('hotpink', alpha=0.9).as01(),
        )
        ref_poly.draw(edgecolor='kitware_yellow', fill=False)

        ax.set_title(ub.urepr(stat_row, nobr=1, si=1))

        keep_true_ss_gdm.plot(
            ax=ax,
            facecolor=Color('kitware_green', alpha=0.8).as01(),
            edgecolor=Color('black', alpha=0.8).as01(),
        )

        imdata = kwplot.render_figure_to_image(ax.figure)
        kwimage.imwrite(f'{region_name}_overlaps.png', imdata)


df = pd.DataFrame(region_stats_rows)
print(df)


### Change names

def change_names():
    """
    Manual execute then

    rm -rf bas_small_output bas_small_truth
    mv bas_small_output5 bas_small_output
    mv bas_small_truth5 bas_small_truth
    dvc add bas_output bas_small_output
    git commit -am "Renamed small sites"
    dvc push -r aws bas_output bas_small_output

    """
    from geowatch.geoannots import geomodels
    dpath1 = ub.Path('/media/joncrall/flash1/smart_drop7/Drop7-StaticACTestSet-2GSD/bas_small_truth/region_models')
    dpath3 = ub.Path('/media/joncrall/flash1/smart_drop7/Drop7-StaticACTestSet-2GSD/bas_small_truth/site_models')
    dpath2 = ub.Path('/media/joncrall/flash1/smart_drop7/Drop7-StaticACTestSet-2GSD/bas_small_output/region_models')

    dpath1_out = ub.Path('/media/joncrall/flash1/smart_drop7/Drop7-StaticACTestSet-2GSD/bas_small_truth5/region_models').ensuredir()
    dpath3_out = ub.Path('/media/joncrall/flash1/smart_drop7/Drop7-StaticACTestSet-2GSD/bas_small_truth5/site_models').ensuredir()
    dpath2_out = ub.Path('/media/joncrall/flash1/smart_drop7/Drop7-StaticACTestSet-2GSD/bas_small_output5/region_models').ensuredir()

    old_names = [
        'CN_C000',
        'CO_C001',
        'KW_C001',
    ]

    for old_region_id in old_names:
        new_region_id = old_region_id.replace('0', '5', 1)

        region = geomodels.RegionModel.coerce(dpath2 / f'{old_region_id}.geojson')
        region.header['region_id'] = new_region_id
        for feat in region.body_features():
            feat['properties']['site_id'] = feat['properties']['site_id'].replace('0', '5', 1)
        (dpath2_out / f'{new_region_id}.geojson').write_text(region.dumps())

        region = geomodels.RegionModel.coerce(dpath1 / f'{old_region_id}.geojson')
        region.header['region_id'] = new_region_id
        for feat in region.body_features():
            feat['properties']['site_id'] = feat['properties']['site_id'].replace('0', '5', 1)
        (dpath1_out / f'{new_region_id}.geojson').write_text(region.dumps())

        old_site_fpaths = list(dpath3.glob(old_region_id + '*'))
        for fpath in old_site_fpaths:
            site = geomodels.SiteModel.coerce(fpath)
            old_site_id = site.header['properties']['site_id']
            new_site_id = old_site_id.replace('0', '5', 1)
            site.header['properties']['region_id'] = new_region_id
            site.header['properties']['site_id'] = new_site_id
            new_fpath = dpath3_out / (new_site_id + '.geojson')
            new_fpath.write_text(site.dumps())

    keep_names = [
        'KR_R002',
    ]
    for region_id in keep_names:
        region = geomodels.RegionModel.coerce(dpath2 / f'{region_id}.geojson')
        (dpath2_out / f'{region_id}.geojson').write_text(region.dumps())
        region = geomodels.RegionModel.coerce(dpath1 / f'{region_id}.geojson')
        (dpath1_out / f'{region_id}.geojson').write_text(region.dumps())
        site_fpaths = list(dpath3.glob(region_id + '*'))
        for fpath in site_fpaths:
            site = geomodels.SiteModel.coerce(fpath)
            new_fpath = dpath3_out / fpath.name
            new_fpath.write_text(site.dumps())

    if 0:
        from geowatch.geoannots import geomodels
        path = ub.Path('/media/joncrall/flash1/smart_drop7/Drop7-StaticACTestSet-2GSD/bas_small_output/region_models/CN_C500.geojson')
        path = ub.Path('/media/joncrall/flash1/smart_drop7/Drop7-StaticACTestSet-2GSD/bas_small_output/region_models/KW_C501.geojson')
        path = ub.Path('/media/joncrall/flash1/smart_drop7/Drop7-StaticACTestSet-2GSD/bas_small_output/region_models/CO_C501.geojson')

        path = ub.Path('/media/joncrall/flash1/smart_drop7/Drop7-StaticACTestSet-2GSD/bas_small_truth/region_models/CN_C500.geojson')
        path = ub.Path('/media/joncrall/flash1/smart_drop7/Drop7-StaticACTestSet-2GSD/bas_small_truth/region_models/CN_C500.geojson')
        assert 'C00' not in path.read_text()

        region = geomodels.RegionModel.coerce(path)
        for feat in region.features:
            if 'cache' in feat['properties']:
                feat['properties']['cache'].pop('region_id', None)
        region.header.pop('region_id')
        region.header['properties']['region_id'] = path.name.split('.')[0]
        path.write_text(region.dumps())
