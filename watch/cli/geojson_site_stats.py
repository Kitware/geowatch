"""


CommandLine:
    DVC_DATA_DPATH=$(smartwatch_dvc --tags='phase2_data' --hardware='auto')
    python -m watch.cli.geojson_site_stats.py \
        --site_models="$DVC_DATA_DPATH/annotations/drop6/site_models/*"


"""

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
        from watch.cli.geojson_site_stats import *  # NOQA
        import watch
        data_dvc_dpath = watch.find_dvc_dpath(tags='phase2_data', hardware='auto')
        kwargs = {
            'site_models': data_dvc_dpath / 'annotations/drop6/site_models/*',
        }
        cmdline = 0

        kwargs['site_models'] = '/data/joncrall/dvc-repos/smart_expt_dvc/smartflow_evals/kit_pre_eval_8_20230131/BR_R001/sc_out_site_models'
    """
    import ubelt as ub
    import pandas as pd
    from watch.utils import util_gis
    from watch.utils import util_time
    import matplotlib.dates as mdates

    config = GeojsonSiteStatConfig.legacy(cmdline=cmdline, data=kwargs)
    site_infos = list(util_gis.coerce_geojson_datas(config['site_models']))

    viz_dpath = config['viz_dpath']
    if viz_dpath is None:
        viz_dpath = '_viz_sitestat'
    viz_dpath = ub.Path(viz_dpath).ensuredir()

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

        valid_obs_rows = obs_subdf[~pd.isnull(obs_df['observation_date'])]
        valid_obs_rows = valid_obs_rows.sort_values('observation_date')
        obs_unixtimes = [util_time.coerce_datetime(d).timestamp() for d in valid_obs_rows['observation_date']]
        duration = np.diff(obs_unixtimes)
        obs_subdf.loc[valid_obs_rows.index[0:-1], 'duration'] = duration

        multiphase_to_duration = obs_subdf.groupby('current_phase')['duration'].sum()
        phase_to_duration = ub.ddict(lambda: 0)
        for mk, v in multiphase_to_duration.items():
            for k in mk.split(','):
                phase_to_duration['duration.' + k.strip()] += v

        keep_keys = ['status', 'region_id', 'site_id', 'start_date', 'end_date'] + metric_keys
        site_subdf = site_df[keep_keys].copy()
        durr_df = pd.DataFrame(ub.udict(phase_to_duration).map_values(lambda x: [x]))
        site_subdf = pd.concat([site_subdf, durr_df], axis=1)

        # pred_info_fpath = site_info['fpath'].parent.parent / 'site_tracks_manifest.json'
        # if pred_info_fpath.exists():
        #     import json
        #     info_section = smart_result_parser.parse_json_header(pred_info_fpath)
        #     track_kw = json.loads(info_section[-1]['properties']['args']['track_kwargs'])
        #     stats_df.loc[:, 'thresh'] = track_kw['thresh']
        obs_stats_accum.append(obs_subdf)
        site_stat_accum.append(site_subdf)

    site_stats = pd.concat(site_stat_accum)
    obs_stats = pd.concat(obs_stats_accum)

    from watch.utils import util_kwplot

    # dates = site_stats['start_date']
    site_stats['start_date'] = util_kwplot.fix_matplotlib_dates(site_stats['start_date'])
    site_stats['end_date'] = util_kwplot.fix_matplotlib_dates(site_stats['end_date'])
    obs_stats['observation_date'] = util_kwplot.fix_matplotlib_dates(obs_stats['observation_date'])

    phase_duration_keys = [k for k in site_stats.columns if k.startswith('duration.')]
    for phase_duration_key in phase_duration_keys:
        deltas = site_stats[phase_duration_key]
        site_stats[phase_duration_key] = util_kwplot.fix_matplotlib_timedeltas(deltas)

    import kwplot
    from watch.mlops.aggregate import hash_regions
    sns = kwplot.autosns()

    finalize_figure = util_kwplot.FigureFinalizer(
        size_inches=np.array([6.4, 4.8]) * 1.0,
        dpath=viz_dpath,
    )
    label_modifier = util_kwplot.LabelModifier()
    for phase_duration_key in phase_duration_keys:
        x = phase_duration_key
        label_modifier.add_mapping({x: x + ' (days)'})
        label_modifier.add_mapping({'rt_area': 'sqrt(area)'})

    fnum = 1
    def new_figure():
        fig = kwplot.figure(doclf=True, fnum=fnum)
        # if 0:
        #     fnum += 1
        return fig

    kwplot.close_figures()

    ### PER-SITE PLOTS
    regions = site_stats['region_id'].unique().tolist()
    region_title = hash_regions(regions)

    ax = new_figure().gca()
    sns.boxplot(data=site_stats, x='status', y='end_date', ax=ax)
    ax.set_title(f'End Date Distribution: regions={region_title}')
    ax.yaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    ax.yaxis.set_major_locator(mdates.DayLocator(interval=356))
    finalize_figure(ax.figure, 'date_distri_stop.png')

    ax = new_figure().gca()
    sns.boxplot(data=site_stats, x='status', y='start_date', ax=ax)
    ax.set_title(f'Start Date Distribution: regions={region_title}')
    ax.yaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    ax.yaxis.set_major_locator(mdates.DayLocator(interval=356))
    finalize_figure(ax.figure, 'date_distri_start.png')

    ax = new_figure().gca()
    ax.set_title(f'regions={region_title}')
    sns.boxplot(data=site_stats, x='status', y='rt_area', ax=ax)
    finalize_figure(ax.figure, 'status_vs_area.png')

    ax = new_figure().gca()
    sns.scatterplot(data=site_stats, x='obox_major', y='obox_minor', ax=ax, hue='status')
    ax.set_title(f'regions={region_title}')
    finalize_figure(ax.figure, 'obox_aspect_ratio.png')

    ax = new_figure().gca()
    sns.scatterplot(data=site_stats, x='obox_major', y='obox_minor', ax=ax, hue='status')
    ax.set_title(f'regions={region_title}')
    finalize_figure(ax.figure, 'obox_aspect_ratio.png')

    def corr_label(group, x, y):
        metrics_of_interest = group[[x, y]]
        metric_corr_mat = metrics_of_interest.corr()
        metric_corr = metric_corr_mat.stack()
        metric_corr.name = 'corr'
        stack_idx = metric_corr.index
        valid_idxs = [(a, b) for (a, b) in ub.unique(map(tuple, map(sorted, stack_idx.to_list()))) if a != b]
        if valid_idxs:
            metric_corr = metric_corr.loc[valid_idxs]
            # corr_lbl = 'corr({},{})={:0.4f}'.format(*metric_corr.index[0], metric_corr.iloc[0])
            corr_lbl = 'corr={:0.4f}'.format(metric_corr.iloc[0])
        else:
            corr_lbl = ''
        return corr_lbl

    phase_duration_keys = [k for k in site_stats.columns if k.startswith('duration.')]
    max_durration = site_stats[phase_duration_keys].max().max()

    import rich
    from watch.utils import util_pandas
    table = site_stats[phase_duration_keys].describe().applymap(lambda x: ''.join(str(x).partition('days')[0:2]))
    table = util_pandas.pandas_shorten_columns(table)
    rich.print(table)

    y = 'rt_area'
    ddd = site_stats[phase_duration_keys + ['site_id', 'rt_area']].melt(id_vars=['site_id', 'rt_area'])
    ddd['value'] = [getattr(d, 'days', None) for d in ddd['value']]
    ax = new_figure().gca()
    sns.scatterplot(data=ddd, x='value', y=y, ax=ax, hue='variable')
    ax.set_title(f'Durations\nregions={region_title}')
    ax.set_xscale('log')
    ax.set_yscale('log')
    for k, group in ddd.groupby('variable'):
        x = 'value'
        label_modifier.update({k: k + ': ' + corr_label(group, x, y)})
    label_modifier(ax)
    finalize_figure(ax.figure, 'area_vs_duration_melt.png')

    for k, group in ddd.groupby('variable'):
        label_modifier.update({k: k})
    ax = new_figure().gca()
    sns.boxplot(data=ddd, x='variable', y=x, ax=ax)
    ax.set_ylim(1, max_durration.days)
    label_modifier(ax)
    ax.set_title(f'Durations\nregions={region_title}')
    finalize_figure(ax.figure, 'duration_boxplot_melt.png')

    for phase_duration_key in phase_duration_keys:
        ax = new_figure().gca()
        valid_duration_df = site_stats[~pd.isnull(site_stats[phase_duration_key])].copy()
        valid_duration_df[phase_duration_key] = [d.days for d in valid_duration_df[phase_duration_key]]
        hue = 'status'
        x = phase_duration_key
        y = 'rt_area'
        for k, group in valid_duration_df.groupby(hue):
            label_modifier.update({k: k + ': ' + corr_label(group, x, y)})
            ...
        sns.scatterplot(data=valid_duration_df, x=x, y=y, ax=ax, hue=hue)
        ax.set_title(f'{phase_duration_key}\nregions={region_title}')
        ax.set_xscale('log')
        ax.set_yscale('log')
        label_modifier(ax)
        finalize_figure(ax.figure, f'area_vs_{phase_duration_key}.png')

        for k, group in valid_duration_df.groupby(hue):
            label_modifier.update({k: k})
        ax = new_figure().gca()
        sns.boxplot(data=valid_duration_df, x=hue, y=x, ax=ax)
        ax.set_ylim(1, max_durration.days)
        label_modifier(ax)
        ax.set_title(f'{phase_duration_key}\nregions={region_title}')
        finalize_figure(ax.figure, f'duration_boxplot_{phase_duration_key}.png')

    # with sns.color_palette("flare"):
    # sns.scatterplot(data=site_stats, x='rt_area', y='major_obox_ratio', hue='status')

    ### PER-OBSERVATION PLOTS

    regions = obs_stats['region_id'].unique().tolist()
    region_title = hash_regions(regions)

    ax = new_figure().gca()
    ax.set_title(f'regions={regions}')
    sns.boxplot(data=obs_stats, x='current_phase', y='observation_date', ax=ax)
    ax.yaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    ax.yaxis.set_major_locator(mdates.DayLocator(interval=356))
    finalize_figure(ax.figure, 'phase_vs_obsdate.png')

    ax = kwplot.figure(doclf=True, fnum=4).gca()
    ax.set_title(f'regions={regions}')
    sns.boxplot(data=obs_stats, x='current_phase', y='rt_area', ax=ax)
    finalize_figure(ax.figure, 'phase_vs_size.png')

    ax = kwplot.figure(doclf=True, fnum=5).gca()
    sns.scatterplot(data=obs_stats, x='obox_major', y='obox_minor', ax=ax, hue='current_phase')
    ax.set_title(f'regions={regions}')
    finalize_figure(ax.figure, 'obox_aspect_ratio_phase.png')

    ax = kwplot.figure(doclf=True, fnum=5).gca()
    sns.scatterplot(data=obs_stats, x='obox_major', y='obox_minor', ax=ax, hue='current_phase')
    ax.set_title(f'regions={regions}')
    finalize_figure(ax.figure, 'size_verus_duration.png')

    # import xdev
    # xdev.view_diri


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
