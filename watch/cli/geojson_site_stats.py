import scriptconfig as scfg
import numpy as np


class GeojsonSiteStatConfig(scfg.DataConfig):
    """
    Compute statistics about geojson sites
    """
    site_models = scfg.Value(None, help='site model coercable')

    viz_dpath = None


def main(cmdline=0, **kwargs):
    """
    Ignore:
        import watch
        data_dvc_dpath = watch.find_dvc_dpath(tags='phase2_data', hardware='auto')
        kwargs = {
            'site_models': data_dvc_dpath / 'annotations/site_models/BR_R001_*',
        }
        cmdline = 0

        kwargs['site_models'] = '/data/joncrall/dvc-repos/smart_expt_dvc/smartflow_evals/kit_pre_eval_8_20230131/BR_R001/sc_out_site_models'
    """
    from watch.utils import util_gis
    config = GeojsonSiteStatConfig.legacy(cmdline=cmdline, data=kwargs)
    site_infos = list(util_gis.coerce_geojson_datas(config['site_models']))

    obs_stats_accum = []
    site_stat_accum = []
    for site_info in site_infos:

        data_crs84 = site_info['data']
        data_utm = util_gis.project_gdf_to_local_utm(data_crs84)
        type_to_datas = dict(list(data_utm.groupby('type')))
        obs_df = type_to_datas.pop('observation', [])
        site_df = type_to_datas.pop('site', [])
        assert len(type_to_datas) == 0
        site_row = site_df.iloc[0]

        site_df = geopandas_shape_stats(site_df)

        obs_df = geopandas_shape_stats(obs_df)
        obs_df['region_id'] = site_row.region_id
        metric_keys = ['rt_area', 'major_obox_ratio', 'obox_major', 'obox_minor', 'hull_rt_area']

        keep_keys = ['status', 'region_id', 'site_id', 'current_phase', 'observation_date'] + metric_keys
        obs_subdf = obs_df[keep_keys].copy()

        keep_keys = ['status', 'region_id', 'site_id', 'start_date', 'end_date'] + metric_keys
        site_subdf = site_df[keep_keys].copy()

        # pred_info_fpath = site_info['fpath'].parent.parent / 'site_tracks_manifest.json'
        # if pred_info_fpath.exists():
        #     import json
        #     info_section = smart_result_parser.parse_json_header(pred_info_fpath)
        #     track_kw = json.loads(info_section[-1]['properties']['args']['track_kwargs'])
        #     stats_df.loc[:, 'thresh'] = track_kw['thresh']
        obs_stats_accum.append(obs_subdf)
        site_stat_accum.append(site_subdf)

    import pandas as pd
    import matplotlib.dates as mdates
    site_stats = pd.concat(site_stat_accum)
    obs_stats = pd.concat(obs_stats_accum)

    def fix_matplotlib_dates(dates):
        from watch.utils import util_time
        new = []
        for d in dates:
            n = util_time.coerce_datetime(d)
            if n is not None:
                n = mdates.date2num(n)
                # n = n.timestamp()
            # n = pd.to_datetime(n)
            # if n is not None:
            #     n = n.to_datetime64()
            new.append(n)
        return new
    # dates = site_stats['start_date']
    site_stats['start_date'] = fix_matplotlib_dates(site_stats['start_date'])
    site_stats['end_date'] = fix_matplotlib_dates(site_stats['end_date'])
    obs_stats['observation_date'] = fix_matplotlib_dates(obs_stats['observation_date'])

    import kwplot
    sns = kwplot.autosns()

    # with sns.color_palette("flare"):
    # sns.scatterplot(data=site_stats, x='rt_area', y='major_obox_ratio', hue='status')
    ax = kwplot.figure(doclf=True, fnum=1).gca()

    regions = site_stats['region_id'].unique().tolist()

    ax = kwplot.figure(doclf=True, fnum=2).gca()
    ax.set_title(f'regions={regions}')
    sns.boxplot(data=site_stats, x='status', y='end_date', ax=ax)
    ax.yaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    ax.yaxis.set_major_locator(mdates.DayLocator(interval=356))

    ax = kwplot.figure(doclf=True, fnum=3).gca()
    ax.set_title(f'regions={regions}')
    sns.boxplot(data=site_stats, x='status', y='start_date', ax=ax)
    ax.yaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    ax.yaxis.set_major_locator(mdates.DayLocator(interval=356))

    ax = kwplot.figure(doclf=True, fnum=4).gca()
    ax.set_title(f'regions={regions}')
    sns.boxplot(data=site_stats, x='status', y='rt_area', ax=ax)

    ax = kwplot.figure(doclf=True, fnum=5).gca()
    sns.scatterplot(data=site_stats, x='obox_major', y='obox_minor', ax=ax, hue='status')
    ax.set_title(f'regions={regions}')

    import kwplot
    sns = kwplot.autosns()

    # with sns.color_palette("flare"):
    # sns.scatterplot(data=site_stats, x='rt_area', y='major_obox_ratio', hue='status')
    ax = kwplot.figure(doclf=True, fnum=1).gca()

    regions = obs_stats['region_id'].unique().tolist()
    ax = kwplot.figure(doclf=True, fnum=6).gca()
    ax.set_title(f'regions={regions}')
    sns.boxplot(data=obs_stats, x='current_phase', y='observation_date', ax=ax)
    ax.yaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    ax.yaxis.set_major_locator(mdates.DayLocator(interval=356))

    ax = kwplot.figure(doclf=True, fnum=4).gca()
    ax.set_title(f'regions={regions}')
    sns.boxplot(data=obs_stats, x='current_phase', y='rt_area', ax=ax)

    ax = kwplot.figure(doclf=True, fnum=5).gca()
    sns.scatterplot(data=obs_stats, x='obox_major', y='obox_minor', ax=ax, hue='current_phase')
    ax.set_title(f'regions={regions}')


def geopandas_shape_stats(df):
    """
    Compute shape statistics about a geopandas dataframe (assume UTM CRS)
    """
    import kwimage
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
