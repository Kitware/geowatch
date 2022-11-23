"""
Find hard negative detections and add them as annotations
"""


def main():
    import watch
    import pandas as pd
    from watch.utils import util_gis
    import ubelt as ub
    import numpy as np
    # from watch.mlops import smart_pipeline

    data_dvc_dpath = watch.find_dvc_dpath(tags='phase2_data')

    if 0:
        # On Jons' machine
        pred_bas_dpath = ub.Path('/home/joncrall/data/dvc-repos/smart_expt_dvc/models/fusion/Aligned-Drop4-2022-08-08-TA1-S2-L8-ACC/pred/trk/Drop4_BAS_Retrain_V002_epoch=31-step=16384.pt.pt/Aligned-Drop4-2022-08-08-TA1-S2-L8-ACC_data.kwcoco/trk_pxl_b788335d')
        pred_path_pat = (pred_bas_dpath / 'trk_poly*' / 'site-summaries' / '*_R*.geojson')
        pred_data_paths = list(watch.utils.util_gis.coerce_geojson_paths(pred_path_pat))

        # For Matt
        dump_path = ub.Path('./pred_regions_for_verification').ensuredir()
        for p in pred_data_paths:
            new_p = dump_path / (p.name.split('.')[0] + '_' + ub.hash_data(p.parts)[0:8] + '.geojson')
            p.copy(new_p)
    else:
        pred_path_pat = ub.Path('./pred_regions_for_verification')

    pred_data_infos = list(watch.utils.util_gis.coerce_geojson_datas(pred_path_pat, workers=4))

    true_region_dpath = data_dvc_dpath / 'annotations/region_models'
    true_data_infos = list(watch.utils.util_gis.coerce_geojson_datas(true_region_dpath / '*_R*.geojson', workers=4))

    all_true_sitesum_accum = []
    all_true_region_accum = []

    for true_info in true_data_infos:
        df = true_info['data']
        region_rows = df[df['type'] == 'region']
        assert len(region_rows) == 1
        region_row = region_rows.iloc[0]
        subdf = df[df['type'] != 'region']
        subdf.loc[subdf.index, 'region_id'] = region_row['region_id']
        all_true_sitesum_accum.append(subdf)
        all_true_region_accum.append(region_rows)

    true_region_df = pd.concat(all_true_region_accum).reset_index()
    true_sitesum_df = pd.concat(all_true_sitesum_accum).reset_index()

    # ---

    all_pred_sitesum_accum = []
    all_pred_region_accum = []
    for pred_info in pred_data_infos:
        df = pred_info['data']
        region_rows = df[df['type'] == 'region']
        assert len(region_rows) == 1
        region_row = region_rows.iloc[0]
        subdf = df[df['type'] != 'region']
        subdf.loc[subdf.index, 'region_id'] = region_row['region_id']
        all_pred_sitesum_accum.append(subdf)
        all_pred_region_accum.append(region_rows)

    # pred_region_df = pd.concat(all_pred_region_accum).reset_index()
    pred_sitesum_df = pd.concat(all_pred_sitesum_accum).reset_index()

    cand_pred_sitesum_df = pred_sitesum_df.copy()

    # Restrict to only the data inside of the region bounds
    idx1_to_idxs2 = util_gis.geopandas_pairwise_overlaps(true_region_df, cand_pred_sitesum_df, predicate='contains')
    contained_idxs = sorted(ub.unique(ub.flatten(idx1_to_idxs2.values())))
    is_contained = np.array(ub.boolmask(contained_idxs, len(cand_pred_sitesum_df)))
    cand_pred_sitesum_df = cand_pred_sitesum_df[is_contained]

    # Remove any prediction that touches any truth annotation
    # idx1_to_idxs2 = util_gis.geopandas_pairwise_overlaps(true_sitesum_df, cand_pred_sitesum_df, predicate='intersects')
    # idxs2 = sorted(ub.unique(ub.flatten(idx1_to_idxs2.values())))
    # is_intersecting = np.array(ub.boolmask(idxs2, len(cand_pred_sitesum_df)))
    # cand_pred_sitesum_df = cand_pred_sitesum_df[~is_intersecting]

    region_id_to_true_sitesum = dict(list(true_sitesum_df.groupby('region_id')))
    region_id_to_region = dict(list(true_region_df.groupby('region_id')))

    import kwplot
    import kwimage
    import geopandas as gpd  # NOQA
    # kwplot.autompl()

    # Visualize candidates
    region_id_to_pred_subdf = dict(list(cand_pred_sitesum_df.groupby('region_id')))
    for region_id, pred_subdf in region_id_to_pred_subdf.items():
        pred_subdf = region_id_to_pred_subdf[region_id]

        fig = kwplot.figure(fnum=region_id, doclf=True)
        ax = fig.gca()

        true_subdf = region_id_to_true_sitesum[region_id]
        region_rows = region_id_to_region[region_id]

        # hack
        # pred_union_geom = pred_subdf['geometry'].geometry.unary_union
        # pred_union_df = gpd.GeoDataFrame({'geometry': [pred_union_geom]})
        # pred_final_df = pred_union_df
        pred_final_df = pred_subdf

        true_color = kwimage.Color.coerce('kitware_green').as01()
        fp_color = kwimage.Color.coerce('kitware_red').as01()
        bound_color = kwimage.Color.coerce('kitware_blue').as01()

        region_rows.plot(ax=ax, facecolor='none', edgecolor=bound_color, alpha=0.9)
        pred_final_df.plot(ax=ax, facecolor='none', edgecolor=fp_color, alpha=0.9)
        true_subdf.plot(ax=ax, facecolor='none', edgecolor=true_color, alpha=0.9)

        ax.set_title(f'region {region_id} - hard negatives')
        fpath = f'hardneg_{region_id}.png'
        fig.savefig(fpath)
        from watch.utils import util_kwplot
        util_kwplot.cropwhite_ondisk(fpath)
