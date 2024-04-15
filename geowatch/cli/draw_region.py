#!/usr/bin/env python3
import scriptconfig as scfg
import ubelt as ub


class DrawRegionCLI(scfg.DataConfig):
    """
    Ignore:
        from geowatch.cli.draw_region import *  # NOQA
        cls = DrawRegionCLI
        cmdline = 0
        kw = kwargs = {}
        kw['models'] = '/home/joncrall/temp/debug_smartflow_latest/ingress/sv_out_region_models/KR_R001.geojson'
        kw['models'] = '/home/joncrall/temp/debug_smartflow_eval20/ingress/sv_out_region_models/KR_R001.geojson'
        kw['models'] = 'sv_out_region_models/KR_R001.geojson'
        kw['models'] = 'sc_out_region_models/KR_R001.geojson'

        kw['models'] = 'cropped_region_models_sc/KR_R001.geojson'
        kw['extra_header'] = '03-cropped_region_models_sc'
        kw['fpath'] = '03-cropped_region_models_sc.png'
        cls.main(cmdline=cmdline, **kwargs)

        kw['models'] = 'sc_out_region_models/KR_R001.geojson'
        kw['extra_header'] = '02-sc_out_region_models'
        kw['fpath'] = '02-sc_out_region_models.png'
        cls.main(cmdline=cmdline, **kwargs)

        kw['models'] = 'sv_out_region_models/KR_R001.geojson'
        kw['extra_header'] = '01-sv_out_region_models'
        kw['fpath'] = '01-sv_out_region_models.png'
        cls.main(cmdline=cmdline, **kwargs)

        ub.cmd('kwimage stack_images *.png')
    """
    # param1 = scfg.Value(None, help='param1')

    models = scfg.Value(None, help='site OR region models coercables (the script will attempt to distinguish them)', nargs='+', position=1)

    site_models = scfg.Value(None, help='site model coercable', nargs='+', alias=['sites'])

    region_models = scfg.Value(None, help='region model coercable', nargs='+', alias=['regions'])
    # viz_dpath = scfg.Value(None, help='if specified will write stats visualizations and plots to this directory')

    extra_header = scfg.Value(None)

    io_workers = scfg.Value('avail', help='number of workers for parallel io')

    with_timeline = scfg.Value(True, help='if True draw the timeline')

    fpath = scfg.Value('region.png', help='path to write viz to')

    @classmethod
    def main(cls, cmdline=1, **kwargs):
        """
        Example:
            >>> # xdoctest: +SKIP
            >>> from geowatch.cli.draw_region import *  # NOQA
            >>> cmdline = 0
            >>> kwargs = dict()
            >>> cls = DrawRegionCLI
            >>> cls.main(cmdline=cmdline, **kwargs)
        """
        import rich
        config = cls.cli(cmdline=cmdline, data=kwargs, strict=True)
        rich.print('config = ' + ub.urepr(config, nl=1))

        # import copy
        import numpy as np
        import pandas as pd
        # from kwutil import util_time
        from geowatch.geoannots import geomodels
        from geowatch.utils import util_gis
        from kwutil import util_parallel
        import kwimage
        import kwplot
        from geowatch.utils import util_kwplot

        # Note: these colors are kinda not useful.
        from geowatch import heuristics
        status_to_color = {r['status']: r['color'] for r in heuristics.HUERISTIC_STATUS_DATA}

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

        dataframes = ub.ddict(list)

        for region in region_models:
            dataframes['region'] += [region.pandas_region()]
            dataframes['site_summary'] += [region.pandas_summaries()]

        for site in site_models:
            dataframes['sites'] += [site.pandas_site()]
            # dataframes['observations'] += [site.pandas_observations()]

        # if dataframes['region']:
        if len(dataframes['region']):
            _region_df = pd.concat(dataframes['region'])
            _summary_df = pd.concat(dataframes['site_summary'], axis=0)
            region_df = util_gis.project_gdf_to_local_utm(_region_df)
            summary_df = _summary_df.to_crs(region_df.crs)
        else:
            region_df = None
            summary_df = None

        if len(dataframes['sites']):
            _site_df = pd.concat(dataframes['sites'])
            site_df = util_gis.project_gdf_to_local_utm(_site_df)
        else:
            site_df = None

        # if config.viz_dpath is None:
        #     config.viz_dpath = ub.Path('.').resolve()

        region_ids = set()
        if region_df is not None:
            region_ids.update(set(region_df['region_id']))
        if site_df is not None:
            region_ids.update(set(site_df['region_id']))

        unique_status = set()
        if summary_df is not None:
            unique_status.update(summary_df['status'].unique())
        if site_df is not None:
            unique_status.update(site_df['status'].unique())

        draw_backend = 'cv2'
        draw_backend = 'mpl'
        if draw_backend == 'mpl':
            sns = kwplot.autosns()

            title = util_kwplot.TitleBuilder()

            if len(str(config.models)) < 255:
                title.ensure_newline()
                title.add_part(str(config.models))
                title.ensure_newline()

            region_ids = sorted(set(region_ids))
            if len(region_ids) == 1:
                region_title = f'Region: {region_ids[0]}'
            else:
                region_title = f'Regions: {",".join(region_ids)}'
            title.add_part(region_title)

            if config.extra_header is not None:
                title.add_part(config.extra_header)

            extra_header = ub.Path('.').resolve().parent.name
            title.add_part(extra_header)
            # extra_header = 'Latest'
            # title_row.append(f'{geowatch.__version__=}')

            unique_colors = [kwimage.Color.coerce(status_to_color.get(s, 'cyan')) for s in unique_status]
            unique_status_to_color = ub.dzip(unique_status, unique_colors)
            legend_canvas = kwplot.make_legend_img(unique_status_to_color, dpi=600)

            # Assign colors to each site summary
            unique_status_to_face_color01 = {s: c.as01() for s, c in unique_status_to_color.items()}
            unique_status_to_edge_color01 = {s: c.adjust(lighten=-.05).as01() for s, c in unique_status_to_color.items()}
            if summary_df is not None:
                summary_df['face_color'] = summary_df['status'].apply(unique_status_to_face_color01.__getitem__)
                summary_df['edge_color'] = summary_df['status'].apply(unique_status_to_edge_color01.__getitem__)
            if site_df is not None:
                site_df['face_color'] = site_df['status'].apply(unique_status_to_face_color01.__getitem__)
                site_df['edge_color'] = site_df['status'].apply(unique_status_to_edge_color01.__getitem__)

            num_rows = 2 if config.with_timeline else 1
            figman = util_kwplot.FigureManager(
                dpath=ub.Path('.'),
                dpi=300,
                size_inches=(12, 8),
                verbose=True,
            )
            fig = figman.figure(fnum=1, pnum=(num_rows, 2, 1), doclf=True)
            ax = fig.gca()

            # Plot Region Bounds
            if region_df is not None:
                region_df['geometry'].plot(
                    facecolor='none',
                    edgecolor='black',
                    ax=ax
                )

            if summary_df is not None:
                # Plot Site Summary Bounds
                summary_df['geometry'].plot(
                    facecolor=summary_df['face_color'],
                    edgecolor=summary_df['edge_color'],
                    # linewidth=1,
                    alpha=0.5,
                    ax=ax
                )
            if site_df is not None:
                # Plot Site Bounds
                # Note: if both sites and sites summary exist, this will
                # overlay both, which might not look very nice.
                site_df['geometry'].plot(
                    facecolor=site_df['face_color'],
                    edgecolor=site_df['edge_color'],
                    # linewidth=1,
                    alpha=0.5,
                    ax=ax
                )

            miny, maxy = ax.get_ylim()
            ax = figman.figure(fnum=1, pnum=(num_rows, 2, 2)).gca()
            kwplot.imshow(legend_canvas, ax=ax)

            figman.set_figtitle(str(title))

            if config.with_timeline:
                ax = kwplot.figure(fnum=1, pnum=(num_rows, 1, 2), docla=1).gca()

                if site_df is not None:
                    centroid = site_df.geometry.centroid
                    site_df['y'] = centroid.y
                    sub = site_df[['site_id', 'y']]
                    stacked = pd.concat([sub] * 2, ignore_index=True)
                    stacked['date'] = pd.concat([site_df.start_date, site_df.end_date], ignore_index=True)
                    stacked['frame_idx'] = np.concatenate([[0] * len(site_df), [1] * len(site_df)])
                    sns.lineplot(data=stacked, x='date', y='y', ax=ax, hue='site_id', legend=False)

                if summary_df is not None:
                    centroid = summary_df.geometry.centroid
                    summary_df['y'] = centroid.y
                    sub = summary_df[['site_id', 'y']]
                    stacked = pd.concat([sub] * 2, ignore_index=True)
                    stacked['date'] = pd.concat([summary_df.start_date, summary_df.end_date], ignore_index=True)
                    stacked['frame_idx'] = np.concatenate([[0] * len(summary_df), [1] * len(summary_df)])
                    sns.lineplot(data=stacked, x='date', y='y', ax=ax, hue='site_id', legend=False)
                ax.set_ylim(miny, maxy)

            fpath = config.fpath
            final_fpath = figman.finalize(fpath)
            dpath = final_fpath.absolute().parent
            rich.print(f'Wrote to: [link={dpath}]{dpath}[/link]')

        else:
            raise NotImplementedError


__cli__ = DrawRegionCLI
main = __cli__.main

if __name__ == '__main__':
    """

    CommandLine:
        python ~/code/geowatch/geowatch/cli/draw_region.py
        python -m geowatch.cli.draw_region
    """
    main()
