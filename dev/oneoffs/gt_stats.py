import watch
import ubelt as ub
import shapestats  # NOQA
# from watch.utils import util_kwimage
import kwimage
import pandas as pd
import numpy as np


def geopandas_shape_stats(df):
    """
    Compute shape statistics about a geopandas dataframe (assume UTM CRS)
    """
    # df['area'] = df.geometry.area
    # df['hull_area'] = df.geometry.convex_hull.area
    df['hull_rt_area'] = np.sqrt(df.geometry.convex_hull.area)
    df['rt_area'] = np.sqrt(df.geometry.area)

    obox_whs = [kwimage.MultiPolygon.from_shapely(s).oriented_bounding_box().extent
                for s in df.geometry]

    df['obox_major'] = [max(e) for e in obox_whs]
    df['obox_minor'] = [min(e) for e in obox_whs]
    df['major_obox_ratio'] = df['obox_major'] / df['obox_minor']

    # df['ch_aspect_ratio'] =
    # df['isoperimetric_quotient'] = df.geometry.apply(shapestats.ipq)
    # df['boundary_amplitude'] = df.geometry.apply(shapestats.compactness.boundary_amplitude)
    # df['eig_seitzinger'] = df.geometry.apply(shapestats.compactness.eig_seitzinger)
    return df


def main():
    data_dvc_dpath = watch.find_dvc_dpath(tags='phase2_data')

    from watch.mlops import smart_result_parser

    # MODE = 'pred'
    MODE = 'true'

    if MODE == 'pred':
        # pred_dpath = ub.Path('/home/joncrall/data/dvc-repos/smart_expt_dvc/models/fusion/Aligned-Drop4-2022-08-08-TA1-S2-L8-ACC/pred/trk/Drop4_BAS_Retrain_V002_epoch=31-step=16384.pt.pt/Aligned-Drop4-2022-08-08-TA1-S2-L8-ACC_data.kwcoco/trk_pxl_b788335d/trk_poly_0be8a0ab/site-summaries/')

        pred_bas_dpath = ub.Path('/home/joncrall/data/dvc-repos/smart_expt_dvc/models/fusion/Aligned-Drop4-2022-08-08-TA1-S2-L8-ACC/pred/trk/Drop4_BAS_Retrain_V002_epoch=31-step=16384.pt.pt/Aligned-Drop4-2022-08-08-TA1-S2-L8-ACC_data.kwcoco/trk_pxl_b788335d')
        pred_path_pat = (pred_bas_dpath / 'trk_poly*' / 'site-summaries' / '*_R*.geojson')
        datas = list(watch.utils.util_gis.coerce_geojson_datas(pred_path_pat, workers=4))

    if MODE == 'true':
        # site_dpath = data_dvc_dpath / 'annotations/site_models'
        # datas = list(watch.utils.util_gis.coerce_geojson_datas(site_dpath / '*_R0*_*.geojson', workers=4))
        region_dpath = data_dvc_dpath / 'annotations/region_models'
        datas = list(watch.utils.util_gis.coerce_geojson_datas(region_dpath / '*_R*.geojson', workers=4))

    site_stats_accum = []
    for site_info in datas:

        data_crs84 = site_info['data']

        flags = data_crs84['type'] == 'site_summary'
        subdata_crs84 = data_crs84[flags]

        region_row = data_crs84[data_crs84['type'] == 'region']
        assert len(region_row) == 1
        region_id = region_row.iloc[0]['region_id']

        subdata_crs84['region_id'] = region_id

        utm_crs = subdata_crs84.estimate_utm_crs()
        subdata_utm = subdata_crs84.to_crs(utm_crs)
        df = subdata_utm
        df = geopandas_shape_stats(df)
        metric_keys = ['rt_area', 'major_obox_ratio', 'obox_major', 'obox_minor', 'hull_rt_area']
        keep_keys = ['status', 'region_id', 'site_id'] + metric_keys
        stats_df = df[keep_keys].copy()

        pred_info_fpath = site_info['fpath'].parent.parent / 'site_tracks_manifest.json'
        if pred_info_fpath.exists():
            import json
            info_section = smart_result_parser.parse_json_header(pred_info_fpath)
            track_kw = json.loads(info_section[-1]['properties']['args']['track_kwargs'])
            stats_df.loc[:, 'thresh'] = track_kw['thresh']
        site_stats_accum.append(stats_df)

    site_stats = pd.concat(site_stats_accum)

    site_stats = site_stats[site_stats['region_id'] == 'AE_R001']

    import kwplot
    import kwimage
    sns = kwplot.autosns()

    if MODE == 'pred':
        a = kwimage.Color('blue')
        b = kwimage.Color('orange')
        thresh = sorted(site_stats['thresh'].unique())
        palette = {}
        for t in thresh:
            alpha = (t - thresh[0]) / (thresh[-1] - thresh[0])
            c = a.interpolate(b, alpha=alpha, ispace='lab', ospace='rgb')
            palette[t] = np.array(c.as01()).clip(0, 1).tolist() + [0.8]
        snskw = dict(
            # palette=palette,
            hue='thresh',
            size='thresh')
    else:
        snskw = dict(
            hue='site_id',
        )

    # with sns.color_palette("flare"):
    # sns.scatterplot(data=site_stats, x='rt_area', y='major_obox_ratio', hue='status')
    ax = kwplot.figure(doclf=True, fnum=1).gca()
    sns.scatterplot(data=site_stats, x='obox_major', y='obox_minor',  ax=ax, **snskw)

    ax = kwplot.figure(doclf=True, fnum=2).gca()
    sns.scatterplot(data=site_stats, x='rt_area', y='major_obox_ratio', ax=ax, **snskw)

    ax = kwplot.figure(doclf=True, fnum=3).gca()
    sns.scatterplot(data=site_stats, x='rt_area', y='hull_rt_area', ax=ax, **snskw)
