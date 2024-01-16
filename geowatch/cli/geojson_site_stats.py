#!/usr/bin/env python3
"""
CommandLine:
    DVC_DATA_DPATH=$(geowatch_dvc --tags='phase2_data' --hardware='auto')
    python -m geowatch.cli.geojson_site_stats \
        --site_models="$DVC_DATA_DPATH/annotations/drop6/site_models/*" \
        --region_models="$DVC_DATA_DPATH/annotations/drop6/region_models/*"

    geowatch geomodel_stats <paths-to-site-or-region-models>
"""
import scriptconfig as scfg
import ubelt as ub


class GeojsonSiteStatsConfig(scfg.DataConfig):
    """
    Compute statistics about geojson sites.

    TODO:
        - [ ] Rename to geojson stats? Or geomodel stats?
        - [ ] make text output more  consistent and more useful.
    """
    __command__ = 'site_stats'
    __alias__ = ['geojson_stats', 'geomodel_stats']

    models = scfg.Value(None, help='site OR region models coercables (the script will attempt to distinguish them)', nargs='+', position=1)

    site_models = scfg.Value(None, help='site model coercable', nargs='+', alias=['sites'])

    region_models = scfg.Value(None, help='region model coercable', nargs='+', alias=['regions'])

    viz_dpath = scfg.Value(None, help='if specified will write stats visualizations and plots to this directory')

    io_workers = scfg.Value('avail', help='number of workers for parallel io')


def main(cmdline=1, **kwargs):
    """
    Ignore:
        from geowatch.cli.geojson_site_stats import *  # NOQA
        import geowatch
        data_dvc_dpath = geowatch.find_dvc_dpath(tags='phase2_data', hardware='auto')
        kwargs = {
            'site_models': data_dvc_dpath / 'annotations/drop6/site_models/*',
        }
        cmdline = 0

        kwargs['site_models'] = '/data/joncrall/dvc-repos/smart_expt_dvc/smartflow_evals/kit_pre_eval_8_20230131/BR_R001/sc_out_site_models'

        DVC_DATA_DPATH=$(geowatch_dvc --tags='phase2_data' --hardware='auto')
        python -m geowatch.cli.geojson_site_stats \
            --region_models="$DVC_DATA_DPATH/annotations/drop6/region_models/PE_R001.geojson" \
            --site_models="$DVC_DATA_DPATH/annotations/drop6/site_models/PE_R001_*.geojson"
    """
    config = GeojsonSiteStatsConfig.cli(cmdline=cmdline, data=kwargs, strict=True)
    import rich
    rich.print('config = ' + ub.repr2(config))
    import copy
    import numpy as np
    import pandas as pd
    from kwutil import util_time
    from geowatch.geoannots import geomodels
    from geowatch.utils import util_gis
    from kwutil import util_parallel

    site_models = []
    region_models = []

    io_workers = util_parallel.coerce_num_workers(config['io_workers'])
    print(f'io_workers={io_workers}')

    if config.models:
        if config.site_models:
            raise ValueError('the models and site_models arguments are mutex')
        if config.region_models:
            raise ValueError('the models and region_models arguments are mutex')
        models = list(util_gis.coerce_geojson_datas(config.models, format='json', workers=io_workers))
        for model_info in models:
            model_data = model_info['data']
            model = geomodels.coerce_site_or_region_model(model_data)
            if isinstance(model, geomodels.SiteModel):
                site_models.append(model)
            elif isinstance(model, geomodels.RegionModel):
                region_models.append(model)
            else:
                raise AssertionError
    else:
        site_models = list(geomodels.SiteModel.coerce_multiple(config['site_models'], workers=io_workers))
        region_models = list(geomodels.RegionModel.coerce_multiple(config['region_models'], workers=io_workers))

    print(f'len(region_models) = {len(region_models)}')
    print(f'len(site_models) = {len(site_models)}')

    region_to_sites = ub.ddict(list)
    region_to_regions = ub.ddict(list)
    for site in site_models:
        region_to_sites[site.region_id].append(site)
    for region in region_models:
        region_to_regions[region.region_id].append(region)

    unique_region_ids = sorted(set(region_to_regions.keys()) | set(region_to_sites.keys()))
    print('unique_region_ids = {}'.format(ub.urepr(unique_region_ids, nl=1)))

    region_to_obs_accum = {}
    region_to_site_accum = {}

    for region_id in unique_region_ids:
        sites = region_to_sites.get(region_id, [])
        regions = region_to_regions.get(region_id, [])
        if len(regions) > 1:
            raise AssertionError('should only be 1 region file per region_id')
        elif len(regions) == 0:
            region = None
        else:
            region = regions[0]

        if region is None:
            site_summaries = []
            print(f'Region: {region_id} does not have a region model')
        else:
            site_summaries = list(region.site_summaries())
            print(f'Region: {region_id} has a region model with {len(site_summaries)} site summaries')

        print(f'Region: {region_id} has {len(sites)} site models')

        region_to_obs_accum[region_id] = region_obs_accum = []
        region_to_site_accum[region_id] = region_site_accum = []

        # Build up per-site information to expand summary stats if we have it.
        site_id_to_stats = {}
        for site in sites:
            obs_df = site.pandas_observations()
            obs_utm = util_gis.project_gdf_to_local_utm(obs_df, mode=1)
            obs_utm = geopandas_shape_stats(obs_utm)

            metric_keys = ['rt_area', 'major_obox_ratio', 'obox_major', 'obox_minor', 'hull_rt_area']
            keep_keys = ['current_phase', 'observation_date'] + metric_keys
            obs_subdf = obs_utm[keep_keys].copy()

            valid_obs_rows = obs_subdf[~pd.isnull(obs_utm['observation_date'])]
            valid_obs_rows = valid_obs_rows.sort_values('observation_date')
            obs_datetimes = [util_time.coerce_datetime(d) for d in valid_obs_rows['observation_date']]
            obs_unixtimes = [d.timestamp() for d in obs_datetimes]
            duration = np.diff(obs_unixtimes)
            obs_subdf.loc[valid_obs_rows.index[0:-1], 'duration'] = duration

            multiphase_to_duration = obs_subdf.groupby('current_phase')['duration'].sum()
            phase_to_duration = ub.ddict(lambda: 0)
            for mk, v in multiphase_to_duration.items():
                for k in mk.split(','):
                    phase_to_duration['duration.' + k.strip()] += v

            keep_keys = ['status', 'region_id', 'site_id', 'start_date', 'end_date'] + metric_keys
            site_df = site.pandas_site()
            site_df = util_gis.project_gdf_to_local_utm(site_df, mode=1)
            site_df = geopandas_shape_stats(site_df)
            site_subdf = site_df[keep_keys].copy()
            durr_df = pd.DataFrame(ub.udict(phase_to_duration).map_values(lambda x: [x]))

            site_subdf = pd.concat([site_subdf, durr_df], axis=1)

            region_obs_accum.append(obs_subdf)
            region_site_accum.append(site_subdf)

            phases = [p for p in obs_subdf['current_phase'].unique() if not pd.isnull(p)]
            site_row = copy.deepcopy(site.header['properties'])
            site_row['num_phases'] = len(phases)
            site_id_to_stats[site.site_id] = site_row
            if region is not None:
                is_earlier = region.start_date > np.array(obs_datetimes)
                is_later = region.end_date < np.array(obs_datetimes)
                num_obs_outside_time_bounds = is_earlier.sum() + is_later.sum()
                site_row['num_obs_outside_time_bounds'] = num_obs_outside_time_bounds
                site_row['num_obs'] = len(obs_utm)

        if region is not None:
            summary_df = region.pandas_summaries()
            summary_df['status'].value_counts()
            summary_df = summary_df.sort_values('status')
            summary_utm = util_gis.project_gdf_to_local_utm(summary_df, mode=1)

            region_df = region.pandas_region()
            region_utm = util_gis.project_gdf_to_local_utm(region_df, mode=1)

            # Find spatial intersection within the region
            if 1:
                region_start_date = region.start_date
                region_end_date = region.end_date
                gdf_site_overlaps(summary_utm, region_start_date, region_end_date)

            display_summary = summary_utm.drop(['type', 'geometry'], axis=1)
            display_summary['area_square_meters'] = summary_utm.geometry.area
            # rich.print(display_summary)

            new_rows = []
            for sitesum in display_summary.to_dict('records'):
                row = site_id_to_stats.get(sitesum['site_id'], sitesum)
                inconsistency = 0
                if row['status'] != sitesum['status']:
                    inconsistency += 1
                if row['start_date'] != sitesum['start_date']:
                    inconsistency += 1
                if row['end_date'] != sitesum['end_date']:
                    inconsistency += 1
                row.update(sitesum)
                row.pop('type', None)
                row.pop('misc_info', None)
                row.pop('cache', None)
                new_rows.append(row)

            new_sitesums = pd.DataFrame(new_rows)
            print('Site Summaries:')
            rich.print(new_sitesums.to_string())

            status_summaries = []
            for status, group in display_summary.groupby('status'):
                start_dates = [d for d in group['start_date'] if not pd.isna(d)]
                end_dates = [d for d in group['end_date'] if not pd.isna(d)]
                max_date = max(end_dates) if end_dates else None
                min_date = min(start_dates) if start_dates else None
                row = {
                    'status': status,
                    'num': len(group),
                    'min_area': group.area_square_meters.min(),
                    'max_area': group.area_square_meters.max(),
                    'min_date': min_date,
                    'max_date': max_date,
                }
                status_summaries.append(row)
            print('Summary Stats:')
            summary_stats = pd.DataFrame(status_summaries)
            rich.print(summary_stats)

            import pint
            ureg = pint.UnitRegistry()
            region_area = region_utm.area.iloc[0] * (ureg.meter ** 2)
            region_area = region_area.to(ureg.kilometer ** 2)
            duration = region.end_date - region.start_date
            duration = util_time.format_timedelta(duration)

            region_stats = {
                'region_area': region_area,
                'duration': duration,
            }
            # print(f'region_stats = {ub.urepr(region_stats, nl=1)}')
            print(f'region_stats = {repr(region_stats)}')
        else:
            # Only site models are given, show their summaries
            import geopandas as gpd
            summary_rows = list(site_id_to_stats.values())
            summary_df = gpd.GeoDataFrame(summary_rows)
            rich.print(summary_df)

            if len(sites) == 1:
                # obs_df2 = pd.concat(region_obs_accum)
                # rich.print(obs_df2)
                rich.print(obs_df)
                # Show observations in this case as well
                ...

    viz_dpath = config['viz_dpath']
    if viz_dpath is not None:
        viz_dpath = ub.Path(viz_dpath).ensuredir()
        viz_site_stats(unique_region_ids, region_to_obs_accum,
                       region_to_site_accum, viz_dpath)


def gdf_site_overlaps(summary_utm, region_start_date, region_end_date):
    import kwutil
    from geowatch.utils import util_gis
    import numpy as np
    import rich
    import pandas as pd
    overlap_rows = []
    idx_to_idxs = util_gis.geopandas_pairwise_overlaps(summary_utm, summary_utm)
    for idx, idxs in idx_to_idxs.items():
        other_idxs = np.setdiff1d(idxs, [idx])
        if len(other_idxs):
            geoms1 = summary_utm.iloc[[idx]]
            geoms2 = summary_utm.iloc[other_idxs]
            s1 = geoms1.iloc[0]

            start1 = kwutil.util_time.coerce_datetime(s1['start_date']) or region_start_date
            end1 = kwutil.util_time.coerce_datetime(s1['end_date']) or region_end_date
            delta1 = end1 - start1

            g1 = s1.geometry
            isects = [g1.intersection(g2) for g2 in geoms2.geometry]
            unions = [g1.union(g2) for g2 in geoms2.geometry]
            isect_areas = np.array([g.area for g in isects])
            union_areas = np.array([g.area for g in unions])
            ious = isect_areas / union_areas
            site_id1 = geoms1['site_id'].iloc[0]
            for _ix, site_id2 in enumerate(geoms2['site_id']):
                s2 = geoms2.iloc[_ix]
                g2 = s2.geometry
                start2 = kwutil.util_time.coerce_datetime(s2['start_date']) or region_start_date
                end2 = kwutil.util_time.coerce_datetime(s2['end_date']) or region_end_date
                delta2 = end2 - start2

                start3 = max(start1, start2)
                end3 = min(end1, end2)

                start4 = min(start1, start2)
                end4 = max(end1, end2)

                isect_delta = end3 - start3
                union_delta = end4 - start4
                iot = isect_delta / union_delta

                overlap_rows.append({
                    'site_id1': site_id1,
                    'site_id2': site_id2,
                    'space_iou': ious[_ix],
                    'time_iou': iot,
                    'space_io1': isect_areas[_ix] / g1.area,
                    'space_io2': isect_areas[_ix] / g2.area,
                    'time_io1': isect_delta / delta1,
                    'time_io2': isect_delta / delta2,
                })
    overlaps = pd.DataFrame(overlap_rows)
    if len(overlaps):
        piv = overlaps.pivot(index='site_id1', columns='site_id2', values=['space_iou', 'time_iou'])
        piv = piv.fillna(0)

        # piv.sort_values('space_iou')
        # piv = piv.loc[piv.sum(axis=1).argsort().index[::-1]]
        # piv = piv[piv.sum(axis=1).sort_values().index[::-1]]
        site_order = piv['space_iou'].sum(axis=0).sort_values().index[::-1]
        piv = piv.loc[site_order]
        piv = piv.swaplevel(axis=1)[site_order]
        piv = piv[site_order]

        piv[piv == 0] = '-'
        rich.print(piv)
    else:
        rich.print('[green]no overlaps')


def viz_site_stats(unique_region_ids, region_to_obs_accum, region_to_site_accum, viz_dpath):
    from geowatch.mlops.aggregate import hash_regions
    from geowatch.utils import util_kwplot
    from geowatch.utils import util_pandas
    import matplotlib.dates as mdates
    import numpy as np
    import pandas as pd
    import rich

    import kwplot
    sns = kwplot.autosns()

    obs_stats_accum = list(ub.flatten(region_to_obs_accum.values()))
    site_stat_accum = list(ub.flatten(region_to_site_accum.values()))

    if site_stat_accum:
        site_stats = pd.concat(site_stat_accum)
    else:
        site_stats = None

    if obs_stats_accum:
        obs_stats = pd.concat(obs_stats_accum)
    else:
        obs_stats = None

    if obs_stats is not None:
        obs_stats['observation_date'] = util_kwplot.fix_matplotlib_dates(obs_stats['observation_date'])

    # dates = site_stats['start_date']
    if site_stats is not None:
        site_stats['start_date'] = util_kwplot.fix_matplotlib_dates(site_stats['start_date'])
        site_stats['end_date'] = util_kwplot.fix_matplotlib_dates(site_stats['end_date'])

        phase_duration_keys = [k for k in site_stats.columns if k.startswith('duration.')]
        for phase_duration_key in phase_duration_keys:
            deltas = site_stats[phase_duration_key]
            site_stats[phase_duration_key] = util_kwplot.fix_matplotlib_timedeltas(deltas)

    finalize_figure = util_kwplot.FigureFinalizer(
        size_inches=np.array([6.4, 4.8]) * 1.0,
        dpath=viz_dpath,
    )

    fnum = 1

    def new_figure():
        fig = kwplot.figure(doclf=True, fnum=fnum)
        # if 0:
        #     fnum += 1
        return fig

    kwplot.close_figures()

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

    ### PER-SITE PLOTS
    if site_stats is not None:
        label_modifier = util_kwplot.LabelModifier()
        for phase_duration_key in phase_duration_keys:
            x = phase_duration_key
            label_modifier.add_mapping({x: x + ' (days)'})
            label_modifier.add_mapping({'rt_area': 'sqrt(area)'})

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

        phase_duration_keys = [k for k in site_stats.columns if k.startswith('duration.')]
        max_durration = site_stats[phase_duration_keys].max().max()

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
    if obs_stats is not None:
        # regions = obs_stats['region_id'].unique().tolist()
        # region_title = hash_regions(regions)
        region_title = hash_regions(unique_region_ids)

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
    import numpy as np
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


__config__ = GeojsonSiteStatsConfig
__config__.main = main

if __name__ == '__main__':
    """
    CommandLine:
        python ~/code/watch/geowatch/cli/geojson_site_stats.py
    """
    main()
