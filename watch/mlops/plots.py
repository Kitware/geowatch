import kwcoco
import math
import numpy as np
import pandas as pd
import ubelt as ub
from watch import heuristics

fit_param_keys = heuristics.fit_param_keys
pred_param_keys = heuristics.pred_param_keys
trk_param_keys = heuristics.trk_param_keys
act_param_keys = heuristics.act_param_keys
DSET_CODE_TO_GSD = heuristics.DSET_CODE_TO_GSD


def expt_over_time(merged_df, metrics, human_mapping, dpath, huevar='expt', fnum=None):
    import kwplot
    if fnum is None:
        fnum = kwplot.next_fnum()

    expt_group = dict(list(merged_df.groupby(['test_dset', 'type'])))
    for code_type, group in expt_group.items():

        test_dset, type = code_type
        selected = group

        plotkw = ub.udict({
            'x': 'step',
            'y': 'value',
            'star': 'in_production',
        })
        metrics = metrics if ub.iterable(metrics) else [metrics]
        metrics_key = '_'.join(metrics)

        missing = set((plotkw & {'x'}).values()) - set(group.columns)
        if missing:
            print(f'Cannot plot plot_pixel_ap_verus_iarpa for {code_type} missing={missing}')

        if plotkw['x'] not in group.columns:
            continue

        if all(m not in group.columns for m in metrics):
            continue

        melted = selected.melt(
            ['step', 'in_production', 'expt', 'pred_cfg'],
            metrics, var_name='metric')

        if len(metrics) > 1:
            plotkw['style'] = 'metric'

        plotkw['hue'] = huevar

        plot_name = 'expt_over_time_' + metrics_key
        prefix = f'{test_dset}_{type}_'

        def make_fig(fnum, legend=True):
            fig = kwplot.figure(fnum=fnum, doclf=True)
            ax = fig.gca()
            humanized_scatterplot(human_mapping, plot_type='line', data=melted, ax=ax, legend=0, **plotkw)
            humanized_scatterplot(human_mapping, plot_type='scatter', data=melted, ax=ax, legend=legend,  s=80, **plotkw)
            if len(metrics) == 1:
                ax.set_ylabel(metrics[0])
            nice_type = human_mapping.get(type, type)
            ax.set_title(f'Metric over time - {nice_type} - {test_dset}')

        fnum = plot_name + prefix
        run_make_fig(make_fig, fnum, dpath, human_mapping, plot_name, prefix)


def run_make_fig(make_fig, fnum, dpath, human_mapping, plot_name, prefix):
    """
    Runs a function that plots a figure with and without a legend.
    Also saves the legend to its own file.
    """
    import kwplot
    plt = kwplot.autoplt()

    plot_dpath_main = (dpath / plot_name).ensuredir()
    plot_dpath_parts = (dpath / (plot_name + '_parts')).ensuredir()

    make_fig(str(fnum) + '_legend', legend=True)
    fig = plt.gcf()
    fname = f'{prefix}{plot_name}.png'
    fpath = plot_dpath_main / fname
    fig.set_size_inches(np.array([6.4, 4.8]) * 1.4)
    fig.tight_layout()
    fig.savefig(fpath)

    SAVE_PARTS = 1
    if SAVE_PARTS:
        ax = fig.gca()
        legend_handles = ax.get_legend_handles_labels()

        # TODO: incorporate that
        make_fig(str(fnum) + '_nolegend', legend=False)
        fig_nolegend = plt.gcf()
        fname = f'{prefix}{plot_name}_nolegend.png'
        fpath = plot_dpath_parts / fname
        fig_nolegend.set_size_inches(np.array([6.4, 4.8]) * 1.4)
        fig_nolegend.tight_layout()
        fig_nolegend.savefig(fpath)

        fig_onlylegend = kwplot.figure(fnum=str(fnum) + '_onlylegend', doclf=1)
        ax2 = fig_onlylegend.gca()
        ax2.axis('off')
        new_legend = ax2.legend(*legend_handles, loc='lower center')
        humanize_legend(new_legend, human_mapping)
        fname = f'{prefix}{plot_name}_onlylegend.png'
        fpath = plot_dpath_parts / fname
        try:
            new_extent = new_legend.get_window_extent()
            inv_scale = fig_onlylegend.dpi_scale_trans.inverted()
            bbox = new_extent.transformed(inv_scale)
            newkw = {'bbox_inches': bbox}
        except Exception:
            newkw = {'bbox_inches': None}
        fig_onlylegend.tight_layout()
        fig_onlylegend.savefig(fpath, **newkw)
        kwplot.close_figures([fig_onlylegend, fig_nolegend])
        cropwhite_ondisk(fpath)


def plot_pixel_ap_verus_iarpa(merged_df, human_mapping, iarpa_metric_lut, pixel_metric_lut, common_plotkw, dpath, huevar='sensorchan'):
    import kwplot
    # expt_group = dict(list(merged_df.groupby(['test_dset', 'type'])))
    expt_group = dict(list(merged_df.groupby(['test_dset', 'type'])))
    plot_name = 'pxl_vs_iarpa'

    for code_type, group in expt_group.items():

        test_dset, type = code_type
        if type == 'eval_act+pxl':
            plotkw = ub.udict({
                'x': pixel_metric_lut[type],
                'y': iarpa_metric_lut[type],
                'hue': huevar,
                **common_plotkw,
                # 'hue': 'trk_use_viterbi',
                # 'style': 'trk_thresh',
                # 'size': 'trk_thresh',
                # 'hue': 'pred_cfg',
                # 'hue': 'expt',
            })
        elif type == 'eval_trk+pxl':
            # hacks
            plotkw = ub.udict({
                'x': pixel_metric_lut[type],
                'y': iarpa_metric_lut[type],
                'hue': huevar,
                **common_plotkw,
                # 'hue': 'trk_thresh',
                # 'size': 'trk_thresh_hysteresis',
                # 'style': 'track_agg_fn',
                # 'hue': 'trk_cfg',
                # 'hue': 'pred_cfg',
                # 'hue': 'expt',
            })
        else:
            raise KeyError(type)

        missing = set((plotkw & {'x', 'y'}).values()) - set(group.columns)
        if missing:
            print(f'Cannot plot plot_pixel_ap_verus_iarpa for {code_type} missing={missing}')

        if plotkw['x'] not in group.columns or plotkw['y'] not in group.columns:
            continue

        metrics_of_interest = group[[plotkw['x'], plotkw['y']]]
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
        data = group

        def make_fig(fnum, legend=True):
            fig = kwplot.figure(fnum=fnum, doclf=True)
            ax = fig.gca()
            n = len(data)
            ax = humanized_scatterplot(human_mapping, data=data, ax=ax, legend=legend, **plotkw)
            nice_type = human_mapping.get(type, type)
            ax.set_title(f'Pixelwise Vs IARPA metrics - {nice_type} - {test_dset}\n{corr_lbl}, n={n}')

        prefix = f'{test_dset}_{type}_'
        fnum = 'plot_pixel_ap_verus_iarpa' + prefix
        run_make_fig(make_fig, fnum, dpath, human_mapping, plot_name, prefix)


def plot_pixel_ap_verus_auc(merged_df, human_mapping, iarpa_metric_lut,
                            pixel_metric_lut, common_plotkw, dpath, huevar='sensorchan'):
    import kwplot
    expt_group = dict(list(merged_df.groupby(['test_dset', 'type'])))
    plot_name = 'pxl_vs_auc'
    for code_type, group in expt_group.items():

        group = group[~group['sensorchan'].isnull()]
        # group['has_teamfeat'] = group['sensorchan'].apply(lambda x: (('depth' in x) or ('invariants' in x) or ('matseg' in x) or ('land' in x)))

        test_dset, type = code_type
        if type == 'eval_act+pxl':
            plotkw = ub.udict({
                'x': pixel_metric_lut[type],
                'y': 'coi_mAUC',
                'hue': huevar,
                **common_plotkw,
                # 'hue': 'trk_use_viterbi',
                # 'style': 'trk_thresh',
                # 'size': 'trk_thresh',
                # 'hue': 'pred_cfg',
                # 'hue': 'expt',
            })
        elif type == 'eval_trk+pxl':
            plotkw = ub.udict({
                'x': pixel_metric_lut[type],
                'y': 'salient_AUC',
                'hue': huevar,
                **common_plotkw,
                # 'hue': 'trk_cfg',
                # 'hue': 'pred_cfg',
                # 'hue': 'expt',
            })
        else:
            raise KeyError(type)
        missing = set((plotkw & {'x', 'y'}).values()) - set(group.columns)
        if missing:
            print(f'Cannot plot plot_pixel_ap_verus_auc for {code_type} missing={missing}')

        metric_corr_mat = group[[plotkw['x'], plotkw['y']]].corr()
        metric_corr = metric_corr_mat.stack()
        metric_corr.name = 'corr'
        stack_idx = metric_corr.index
        valid_idxs = [(a, b) for (a, b) in ub.unique(map(tuple, map(sorted, stack_idx.to_list()))) if a != b]
        metric_corr = metric_corr.loc[valid_idxs]
        # corr_lbl = 'corr({},{})={:0.4f}'.format(*metric_corr.index[0], metric_corr.iloc[0])
        corr_lbl = 'corr={:0.4f}'.format(metric_corr.iloc[0])

        def make_fig(fnum, legend=True):
            fig = kwplot.figure(fnum=fnum)
            ax = fig.gca()
            ax = humanized_scatterplot(human_mapping, data=group, ax=ax, **plotkw)
            nice_type = human_mapping.get(type, type)
            ax.set_title(f'Pixelwise metrics - {nice_type} - {test_dset}\n{corr_lbl}')
            fig.set_size_inches(16.85, 8.82)

        prefix = f'{test_dset}_{type}_'
        fnum = 'plot_pixel_ap_verus_auc' + prefix
        run_make_fig(make_fig, fnum, dpath, human_mapping, plot_name, prefix)


def plot_resource_versus_metric(merged_df, human_mapping, iarpa_metric_lut, pixel_metric_lut, common_plotkw, dpath, huevar='sensorchan'):
    import kwplot
    expt_group = dict(list(merged_df.groupby(['test_dset', 'type'])))
    plot_name = 'resource_vs_metric'
    if 1:
        expt_group.pop((10, 'eval_act+pxl'), None)
        expt_group.pop((1, 'eval_act+pxl'), None)
        expt_group.pop((1, 'eval_trk+pxl'), None)

    for resource_type in ['total_hours', 'co2_kg']:
        human_resource_type = human_mapping.get(resource_type, resource_type)

        # 'pixel']:
        # for metric_type in ['pixel']:
        for metric_type in ['iarpa']:
            if metric_type == 'iarpa':
                metric_lut = iarpa_metric_lut
                human_metric_type = 'IARPA'
            else:
                metric_lut = pixel_metric_lut
                human_metric_type = 'Pixelwise'

            # pnum_ = kwplot.PlotNums(nCols=len(expt_group))
            for code_type, group in expt_group.items():

                group['pred_tta_time'] = group['pred_tta_time'].astype(str)
                group['pred_tta_fliprot'] = group['pred_tta_fliprot'].astype(str)

                group.loc[group['pred_tta_time'] == 'nan', 'pred_tta_time'] = '0.0'
                group.loc[group['pred_tta_fliprot'] == 'nan', 'pred_tta_fliprot'] = '0.0'
                test_dset, type = code_type
                if type == 'eval_act+pxl':
                    plotkw = ub.udict({
                        'x': resource_type,
                        'y': metric_lut[type],
                        'hue': huevar,
                        **common_plotkw,
                        # 'style': 'pred_cfg',
                        # 'hue': 'pred_tta_fliprot',
                        # 'hue': 'pred_tta_time',
                        # 'size': 'pred_tta_fliprot',
                        'style': 'hardware',
                    })
                elif type == 'eval_trk+pxl':
                    plotkw = ub.udict({
                        'x': resource_type,
                        'y': metric_lut[type],
                        'hue': 'sensorchan',
                        **common_plotkw,
                        # 'hue': 'pred_tta_time',
                        # 'size': 'pred_tta_fliprot',
                        'style': 'hardware',
                    })
                else:
                    raise KeyError(type)

                missing = set((plotkw & {'x', 'y'}).values()) - set(group.columns)
                if missing:
                    print(f'Cannot plot plot_resource_versus_metric for {code_type} missing={missing}')
                    continue
                # fig = kwplot.figure(fnum=fnum, pnum=pnum_())
                # fig.get_size_inches()
                # fig.set_size_inches(17.85,  6.82)
                data = group

                def make_fig(fnum, legend=True):
                    fig = kwplot.figure(fnum=fnum)
                    ax = fig.gca()
                    ax = humanized_scatterplot(human_mapping, data=data, ax=ax, **plotkw)
                    nice_type = human_mapping.get(type, type)
                    ax.set_title(f'{human_resource_type} vs {human_metric_type} - {nice_type} - {test_dset}')
                    fig.set_size_inches(np.array([6.4, 4.8]) * 1.4)

                prefix = f'{test_dset}_{type}_'
                fnum = 'plot_resource_versus_metric' + prefix
                run_make_fig(make_fig, fnum, dpath, human_mapping, plot_name, prefix)


def plot_viterbii_analysis(merged_df, human_mapping, iarpa_metric_lut, pixel_metric_lut, common_plotkw):
    import kwplot
    expt_group = dict(list(merged_df.groupby(['test_dset', 'type'])))
    from watch.utils import result_analysis

    nan_defaults = {
        'modulate_class_weights': '',
        'trk_agg_fn': 'probs',
        'pred_tta_fliprot': 0,
        'pred_tta_time': 0,
        'pred_chip_overlap': 0.3,
        'decoder': 'mlp',
        'trk_morph_kernel': 3,
        'stream_channels': 8,
        'trk_thresh': 0.2,
        'trk_use_viterbi': 0,
        'trk_thresh_hysteresis': None,
        'trk_moving_window_size': None,
        'use_cloudmask': 0,
    }

    merged_df.loc[merged_df['use_cloudmask'].isnull(), 'use_cloudmask'] = 0

    code_type = ('Cropped-Drop3-TA1-2022-03-10', 'eval_act+pxl')
    test_dset, type = code_type
    group = expt_group[code_type]

    iarpa_metric = iarpa_metric_lut[type]
    pixel_metric = pixel_metric_lut[type]
    # metric = pixel_metric
    metric = iarpa_metric

    results_list = []
    for row in group.to_dict('records'):
        metric_val = row[metric]
        if math.isnan(metric_val):
            continue
        metrics = {
            metric: metric_val,
        }
        params = ub.dict_isect(row, fit_param_keys + pred_param_keys + trk_param_keys + act_param_keys)
        params['modulate_class_weights']

        for k, v in params.items():
            if isinstance(v, float) and math.isnan(v):
                if k == 'sensorchan':
                    params['sensorchan'] = params['channels']
                else:
                    params[k] = nan_defaults[k]

        result = result_analysis.Result(None, params, metrics)
        results_list.append(result)

    ignore_params = {'bad_channels'}
    ignore_metrics = {}
    abalation_orders = {1}
    analysis = result_analysis.ResultAnalysis(
        results_list, ignore_params=ignore_params,
        # metrics=['coi_mAPUC', 'coi_APUC'],
        # metrics=['salient_AP'],
        # metrics=['coi_mAP', 'salient_AP'],
        metrics=[metric],
        metric_objectives={
            'salient_AP': 'max',
            'coi_mAP': 'max',
            'mean_f1': 'max',
        },
        ignore_metrics=ignore_metrics,
        abalation_orders=abalation_orders
    )
    # try:
    #     analysis.run()
    # except TypeError:
    #     raise

    param = 'trk_use_viterbi'
    scored_obs = analysis.abalate(param)
    ab_rows = []
    pts1 = []
    pts2 = []
    for obs in scored_obs:
        ab_row = obs.melt(['trk_use_viterbi']).pivot(['variable'], ['trk_use_viterbi'], 'value').reset_index(drop=True)
        if (~ab_row.isnull()).values.sum() > 1:

            # hack
            obspt = obs.copy()
            obspt[param] = obs[param].astype(bool).astype(int)
            pts1.append(obspt.values[0])
            pts2.append(obspt.values[1])
            ab_rows.append(ab_row)
    # ab_df = pd.concat(ab_rows).reset_index(drop=True)
    # obs[param]

    plt = kwplot.autoplt()
    ax = plt.gca()
    kwplot.draw_line_segments(pts1, pts2)
    # ticks = ax.set_xticks([0, 1], ['0', 'v6,v1'])
    ax.set_ylabel(metric)
    ax.set_xlabel(param)
    ax.set_title('Viterbi A/B tests')

    return

    # ab_df
    # .reset_index()
    # ab_melt_df = ab_df.melt(id_vars=['index'], value_vars=ab_row.columns)
    # sns = kwplot.autosns()
    # sns.lineplot(data=ab_melt_df, x=param, y='value')

    import seaborn as sns
    kwplot.figure(fnum=1000)
    sns.violinplot(data=merged_df, x='temporal_dropout', y=pixel_metric)

    # TODO: translate to chip_dims
    # kwplot.figure(fnum=1001)
    # sns.violinplot(data=merged_df, x='chip_size', y=pixel_metric)

    kwplot.figure(fnum=1002, doclf=True)
    sns.violinplot(data=merged_df, x='time_steps', y=pixel_metric)

    kwplot.figure(fnum=1003, doclf=True)
    sns.violinplot(data=merged_df, x='trk_use_viterbi', y=pixel_metric)

    kwplot.figure(fnum=1004, doclf=True)
    ax = sns.violinplot(data=group, x='sensorchan', y=pixel_metric)
    for xtick in ax.get_xticklabels():
        xtick.set_rotation(90)

    kwplot.figure(fnum=3)
    sns.violinplot(data=merged_df, x='saliency_loss', y=pixel_metric)

    group[heuristics.fit_param_keys]
    group[heuristics.fit_param_keys]


def humanize_dataframe(df, col_formats, human_labels=None, index_format=None,
                       title=None):
    import humanize
    df2 = df.copy()
    for col, fmt in col_formats.items():
        if fmt == 'intcomma':
            df2[col] = df[col].apply(humanize.intcomma)
        if fmt == 'concice_si_display':
            from kwcoco.metrics.drawing import concice_si_display
            for row in df2.index:
                val = df2.loc[row, col]
                if isinstance(val, float):
                    val = concice_si_display(val)
                    df2.loc[row, col] = val
            df2[col] = df[col].apply(humanize.intcomma)
        if callable(fmt):
            df2[col] = df[col].apply(fmt)
    if human_labels:
        df2 = df2.rename(human_labels, axis=1)

    indexes = [df2.index, df2.columns]
    for index in indexes:
        if index.name is not None:
            index.name = human_labels.get(index.name, index.name)
        if index.names:
            index.names = [human_labels.get(n, n) for n in index.names]

    if index_format == 'capcase':
        def capcase(x):
            if '_' in x or x.islower():
                return ' '.join([w.capitalize() for w in x.split('_')])
            return x
        df2.index.values[:] = [human_labels.get(x, x) for x in df2.index.values]
        df2.index.values[:] = list(map(capcase, df2.index.values))
        # human_df = human_df.applymap(lambda x: str(x) if isinstance(x, int) else '{:0.2f}'.format(x))
        pass

    df2_style = df2.style
    if title:
        df2_style = df2_style.set_caption(title)
    return df2_style


def humanize_legend(legend, human_mapping):
    leg_title = legend.get_title()
    old_text = leg_title.get_text()
    new_text = human_mapping.get(old_text, old_text)
    leg_title.set_text(new_text)
    for leg_lbl in legend.texts:
        old_text = leg_lbl.get_text()
        new_text = human_mapping.get(old_text, old_text)
        leg_lbl.set_text(new_text)


def humanized_scatterplot(human_mapping, data, ax, plot_type='scatter', mesh=None, connect=None, star=None, starkw=None, **plotkw):
    """
    Example:
        import pandas as pd
        human_mapping = {}
        ax = None
        plotkw = {'x': 'x', 'y': 'y', 'hue': 'group'}
        n = 100
        data = pd.DataFrame({
             'x': np.random.rand(n),
             'y': np.random.rand(n),
             'group': (np.random.rand(n) * 5).astype(int),
             'star': (np.random.rand(n) > 0.9).astype(int),
        })
        mesh = 'group'
        import kwplot
        kwplot.autompl()
        kwplot.figure(fnum=32)
        humanized_scatterplot(human_mapping, data, ax, mesh, **plotkw)
    """
    import seaborn as sns
    import kwplot
    import kwimage
    plt = kwplot.autoplt()
    xkey = plotkw['x']
    ykey = plotkw['y']

    if ax is None:
        import kwplot
        plt = kwplot.autoplt()
        ax = plt.gca()

    if star is not None:
        _starkw = ub.dict_isect(plotkw, {'s'})
        _starkw = {
            's': _starkw.get('s', 10) + 280,
            'color': 'orange',
        }
        if starkw is not None:
            _starkw.update(starkw)
        flags = data[star].apply(bool)
        star_data = data[flags]
        star_x = star_data[xkey]
        star_y = star_data[ykey]
        ax.scatter(star_x, star_y, marker='*', **_starkw)

    if plot_type == 'scatter':
        ax = sns.scatterplot(data=data, ax=ax, **plotkw)
    else:
        ax = sns.lineplot(data=data, ax=ax, **plotkw)

    ax.set_xlabel(human_mapping.get(xkey, xkey))
    ax.set_ylabel(human_mapping.get(ykey, ykey))
    legend = ax.get_legend()
    if legend is not None:
        humanize_legend(legend, human_mapping)

    if connect:
        import scipy
        import scipy.spatial
        groups = data.groupby(connect)
        colors = kwimage.Color.distinct(len(groups))
        i = 0
        for gkey, subgroup in groups:
            if 'step' in subgroup.columns:
                subgroup = subgroup.sort_values('epoch')
            points = subgroup[[xkey, ykey]].values
            ax = plt.gca()
            ax.plot(points[:, 0], points[:, 1], '--', alpha=0.3, color='gray')
            i += 1

    if mesh:
        import scipy
        import scipy.spatial
        mesh_groups = data.groupby(mesh)
        colors = kwimage.Color.distinct(len(mesh_groups))
        i = 0
        for gkey, subgroup in mesh_groups:
            points = subgroup[[xkey, ykey]].values
            did_plot = 0
            if 1:
                if len(points) > 3:
                    try:
                        tri = scipy.spatial.Delaunay(points)
                        if 0:
                            # MSE
                            # todo: cut off non-mse edges, or order based on epoch?
                            import networkx as nx
                            g = nx.Graph()
                            g.add_edges_from(tri.simplices[:, 0:2])
                            g.add_edges_from(tri.simplices[:, 1:3])
                            g.add_edges_from(tri.simplices[:, [2, 0]])
                            mse_edges = list(nx.minimum_spanning_tree(g).edges)
                            segments = points[mse_edges, :]
                            pts1 = segments[:, 0, :]
                            pts2 = segments[:, 1, :]
                            kwplot.draw_line_segments(pts1, pts2, color=colors[i])
                        else:
                            plt.triplot(points[:, 0], points[:, 1], tri.simplices, alpha=0.2)
                    except Exception:
                        pass
                    else:
                        did_plot = 1
            if not did_plot:
                # Just trace the points in whatever order
                ax = plt.gca()
                ax.plot(points[:, 0], points[:, 1], alpha=0.2, color='gray')
            i += 1
    return ax


def cropwhite_ondisk(fpath):
    import kwimage
    from kwplot.mpl_make import crop_border_by_color
    imdata = kwimage.imread(fpath)
    imdata = crop_border_by_color(imdata)
    kwimage.imwrite(fpath, imdata)


def describe_varied(merged_df, dpath, human_mapping=None):
    # import pprint
    expt_group = dict(list(merged_df.groupby(['dataset_code', 'type'])))
    fnum = 40

    human_mapping.update({
        'time_steps': 'Time Steps (frames)',
        # 'chip_size': 'Chip Size (pxls)',  # TODO chip_dims
        'time_span': 'Time Span',
        'time_sampling': 'Temporal Sampling Method',
        'num_unique': 'Num Unique',
        'top_val': 'Most Frequent Value',
        'top_freq': 'Top Freq',
        'param': 'Param Name',
        'init': 'Network Initialization',
        'pred_tta_time': 'Temporal Test Time Augmentation',
        'pred_tta_time': 'Temporal Test Time Augmentation',
    })

    ignore_params = {
        'bad_channels',
        'true_multimodal',
        'modulate_class_weights',
        'channels',
    }

    varied_dpath = (dpath / 'varied_params').ensuredir()

    def varied_param_table(fnum, param_keys, dataset_code, param_type):
        have_params = list(ub.oset(param_keys) & set(group.columns))
        if len(have_params) == 0:
            return
        part = group[have_params].fillna('null')
        # varied_series = part.nunique()
        # varied_series = varied_series[varied_series > 1]
        # print(varied_series)
        # human_df = varied_df.rename(human_mapping, axis=0)

        param_to_row = ub.ddict(dict)
        param_to_hist = {k: part[k].value_counts() for k in have_params}
        rows = []
        for param, hist in param_to_hist.items():
            if param in ignore_params:
                continue
            row = param_to_row[param]
            row['param'] = param
            row['num_unique'] = len(hist)
            row['top_val'] = hist.idxmax()
            # row['top_freq'] = hist.max()
            if row['num_unique'] > 1:
                rows.append(row)
        if len(rows) == 0:
            return None
        param_summary = pd.DataFrame(rows).set_index('param', drop=True)

        col_formats = {
            'num_unique': int,
            'top_val': 'concice_si_display',
        }
        index_format = 'capcase'
        title = f'Varied {param_type} Parameters: {dataset_code}'
        df = param_summary
        df2_style = humanize_dataframe(df, col_formats,
                                       index_format=index_format,
                                       human_labels=human_mapping,
                                       title=title)
        fname = 'varied_' + dataset_code + '_' + param_type + '_' + ub.hash_data([fnum, code_type])[0:16] + '.png'
        fpath = varied_dpath / fname
        dfi_table(df2_style, fpath, fontsize=12, show=False)

    for code_type, group in expt_group.items():
        print(f'code_type={code_type}')
        fnum += 1
        dataset_code, type = code_type

        print('Varied fit params')
        print('type = {}'.format(ub.repr2(type, nl=1)))
        print('dataset_code = {}'.format(ub.repr2(dataset_code, nl=1)))
        param_keys = heuristics.fit_param_keys
        varied_param_table(fnum, param_keys, dataset_code, param_type='Fit')
        # print(ub.highlight_code(pprint.pformat(dict(varied), width=80)))
        # print('varied = {}'.format(ub.repr2(varied, nl=2)))

        print('Varied pred params')
        print('type = {}'.format(ub.repr2(type, nl=1)))
        print('dataset_code = {}'.format(ub.repr2(dataset_code, nl=1)))
        param_keys = pred_param_keys
        fnum += 1
        varied_param_table(fnum, param_keys, dataset_code, param_type='Predict')
        # part = group[pred_param_keys].fillna('null')
        # rows = part.to_dict('records')
        # varied = ub.varied_values(rows, 0)
        # print(ub.highlight_code(pprint.pformat(dict(varied), width=80)))

        print('Varied track params')
        print('type = {}'.format(ub.repr2(type, nl=1)))
        print('dataset_code = {}'.format(ub.repr2(dataset_code, nl=1)))
        param_keys = trk_param_keys
        fnum += 1
        varied_param_table(fnum, param_keys, dataset_code, param_type='BAS Tracking')
        # part = group[trk_param_keys].fillna('null')
        # rows = part.to_dict('records')
        # varied = ub.varied_values(rows, 0)
        # print(ub.highlight_code(pprint.pformat(dict(varied), width=80)))

        print('Varied activity params')
        print('type = {}'.format(ub.repr2(type, nl=1)))
        print('dataset_code = {}'.format(ub.repr2(dataset_code, nl=1)))
        param_keys = act_param_keys
        fnum += 1
        varied_param_table(fnum, param_keys, dataset_code, param_type='SC Classification')
        # part = group[act_param_keys].fillna('null')
        # rows = part.to_dict('records')
        # varied = ub.varied_values(rows, 0)
        # print(ub.highlight_code(pprint.pformat(dict(varied), width=80)))


memo_kwcoco_load = ub.memoize(kwcoco.CocoDataset)


def dataset_summary_tables(dpath):
    import watch

    dvc_expt_dpath = watch.find_smart_dvc_dpath()
    rows = []
    DSET_CODE_TO_TASK = {
        # 'Aligned-Drop3-TA1-2022-03-10': 'bas',
        # 'Aligned-Drop3-L1': 'bas',
        # 'Cropped-Drop3-TA1-2022-03-10': 'sc',
        'Aligned-Drop4-2022-08-08-TA1-S2-L8-ACC': 'bas',
        'Aligned-Drop4-2022-08-08-TA1-S2-WV-PD-ACC': 'sc',
    }
    for bundle_name in DSET_CODE_TO_TASK.keys():
        task = DSET_CODE_TO_TASK[bundle_name]
        gsd = DSET_CODE_TO_GSD.get(bundle_name, None)
        bundle_dpath = dvc_expt_dpath / bundle_name
        train_fpath = bundle_dpath / 'data_train.kwcoco.json'
        vali_fpath = bundle_dpath / 'data_vali.kwcoco.json'

        if not train_fpath.exists():
            raise Exception
            watch.utils.simple_dvc.SimpleDVC().pull(list(bundle_dpath.glob('splits*.dvc')))
            zip_fpaths = list(bundle_dpath.glob('splits.zip'))
            for p in zip_fpaths:
                ub.cmd(f'7z x {p}', verbose=3, check=1, cwd=bundle_dpath)
            pass

        print(f'read train_fpath={train_fpath}')
        train_dset = memo_kwcoco_load(train_fpath)
        print(f'read vali_fpath={vali_fpath}')
        vali_dset = memo_kwcoco_load(vali_fpath)

        if gsd is None:
            gsd = set(vid['target_gsd'] for vid in train_dset.videos().objs)

        if ub.iterable(gsd) and len(gsd) == 1:
            gsd = ub.peek(gsd)

        type_to_dset = {'train': train_dset, 'vali': vali_dset}
        for split, dset in type_to_dset.items():
            unique_regions = {'_'.join(n.split('_')[0:2]) for n in dset.videos().get('name')}
            unique_sites = set(dset.annots().get('track_id'))
            num_tracks = len(unique_sites)
            row = {
                'dataset': bundle_name,
                'task': task,
                'split': split,
                'gsd': gsd,
                'num_regions': len(unique_regions),
                'num_sites': num_tracks,
                'num_videos': dset.n_videos,
                'num_images': dset.n_images,
                'num_annots': dset.n_annots,
            }
            rows.append(row)

    human_labels = {
        'dataset': 'Dataset Codename',
        'task': 'Task',
        'split': 'Split',
        'gsd': 'GSD',
        'num_sites': 'Num Sites',
        'num_videos': 'Num Videos',
        'num_regions': 'Num Regions',
        'num_images': 'Num Images',
        'num_annots': 'Num Annots',
    }

    col_formats = {
        'gsd': 'intcomma',
        'num_sites': 'intcomma',
        'num_videos': 'intcomma',
        'num_images': 'intcomma',
        'num_annots': 'intcomma',
    }
    df = pd.DataFrame(rows)
    df['task'] = df['task'].apply(str.upper)
    df = df.set_index(['dataset', 'task', 'split'])
    import rich
    rich.print(df.to_string())
    title = 'Dataset Summary'
    df2_style = humanize_dataframe(df, col_formats, human_labels=human_labels,
                                   title=title)
    fpath = dpath / 'dataset_summary.png'
    dfi_table(df2_style, fpath, fontsize=32, show='eog')


def initial_summary(reporter, dpath=None):
    table = reporter.orig_merged_df.copy()
    # Alternate way to compute
    co2_rows = []
    kwh_rows = []
    hour_rows = []
    for row in reporter.big_rows:
        if row['type'] == 'eval_pxl':
            co2_rows += [row['info']['param_types']['resource'].get('co2_kg', np.nan)]
            kwh_rows += [row['info']['param_types']['resource'].get('kwh', np.nan)]
            hour_rows += [row['info']['param_types']['resource'].get('total_hours', np.nan)]

    def mean_imputer(arr):
        arr = np.array([np.nan if a is None else a for a in arr])
        val = np.nanmean(arr)
        arr[np.isnan(arr)] = val
        return arr

    co2_rows = mean_imputer(np.array(co2_rows))
    hour_rows = mean_imputer(np.array(hour_rows))
    kwh_rows = mean_imputer(np.array(kwh_rows))

    total_co2 = co2_rows.sum()
    total_hours = hour_rows.sum()
    total_kwh = kwh_rows.sum()

    num_expt_models = table[['expt', 'model']].nunique()
    num_eval_types = table['type'].value_counts()

    summary_df = pd.concat([num_expt_models, num_eval_types]).to_frame().T
    order = ub.oset(['expt', 'model', 'eval_pxl', 'eval_trk', 'eval_act']) & summary_df.columns
    summary_df = summary_df[order]
    summary_df['co2_kg'] = [total_co2]
    summary_df['kwh'] = [total_kwh]
    summary_df['hours'] = [total_hours]
    summary_df['index'] = ['Total']
    summary_df = summary_df.set_index('index', drop=True)
    summary_df.index.name = None
    humanize = {
        'expt': 'Num Training Runs (Experiments)',
        'model': 'Num Checkpoints Selected (Models)',
        'eval_pxl': 'Num Pixel Evaluations',
        'eval_trk': 'Num IARPA BAS Evaluations',
        'eval_act': 'Num IARPA SC Evaluations',
        'co2_kg': 'Total Carbon Cost (kg)',
        'kwh': 'Energy (kW/H)',
        'hours': 'Total Compute Time (hours)',
    }
    human_df = summary_df.rename(humanize, axis=1)
    human_df = human_df.applymap(lambda x: str(x) if isinstance(x, int) else '{:0.2f}'.format(x))
    table_style = human_df.T.style
    table_style = table_style.set_caption('Experimental Summary')

    import rich
    rich.print(human_df)

    fpath = dpath / 'big_picture_experiment_summary.png'
    dfi_table(table_style, fpath, fontsize=24, show='eog')


def dfi_table(table_style, fpath, fontsize=12, fnum=None, show='eog'):
    import kwimage
    import kwplot
    import dataframe_image as dfi
    dfi_converter = "chrome"  # matplotlib
    dfi.export(
        table_style,
        str(fpath),
        table_conversion=dfi_converter,
        fontsize=fontsize,
        max_rows=-1,
    )
    if show == 'imshow':
        imdata = kwimage.imread(fpath)
        kwplot.imshow(imdata, fnum=fnum)
    elif show == 'eog':
        import xdev
        xdev.startfile(fpath)


def plot_ta1_vs_l1(merged_df, human_mapping, iarpa_metric_lut, pixel_metric_lut, common_plotkw, dpath, fnum=None):
    import kwplot
    sns = kwplot.autosns()

    if fnum is None:
        fnum = kwplot.next_fnum()

    fnum = 0
    expt_group = dict(list(merged_df.groupby(['test_dset', 'type'])))
    k1 = ('Aligned-Drop3-TA1-2022-03-10', 'eval_trk+pxl')
    k2 = ('Aligned-Drop3-L1', 'eval_trk+pxl')
    plot_name = 'ta1_vs_l1'
    param = 'Processing'
    plotkw = ub.udict({
        'x': 'salient_AP',
        'y': 'BAS_F1',
        'hue': param,
        # 'hue': 'sensorchan',
        **common_plotkw,
        # 'hue': 'trk_use_viterbi',
        # 'style': 'trk_thresh',
        # 'size': 'trk_thresh',
        # 'hue': 'pred_cfg',
        # 'hue': 'expt',
    })
    x = plotkw['x']
    y = plotkw['y']
    plotkw.pop('style', None)

    plot_dpath = (dpath / plot_name).ensuredir()

    from watch.utils import result_analysis
    all_param_keys = ub.oset.union(trk_param_keys, pred_param_keys,
                                   fit_param_keys, act_param_keys)

    all_param_keys = {
        'trk_thresh',
        # 'trk_morph_kernel',
        'trk_agg_fn', 'trk_thresh_hysteresis', 'trk_moving_window_size',
        'pred_tta_fliprot', 'pred_tta_time', 'pred_chip_overlap',
        # 'sensorchan',
        # 'time_steps',
        # 'chip_size',
        # 'chip_overlap',
        # 'arch_name',
        # 'optimizer', 'time_sampling', 'time_span', 'true_multimodal',
        # 'accumulate_grad_batches', 'modulate_class_weights', 'tokenizer',
        # 'use_grid_positives', 'use_cloudmask', 'upweight_centers',
        # 'temporal_dropout', 'stream_channels', 'saliency_loss', 'class_loss',
        # 'init', 'learning_rate', 'decoder',
        'trk_use_viterbi'
    }

    data1 = expt_group[k1]
    data2 = expt_group[k2]
    data = pd.concat([data1, data2])
    data = data[~data['has_teamfeat']]

    if 0:
        # Reduce to comparable groups according to the abalate criterion
        results_list = []
        for row in data.to_dict('records'):
            params = ub.dict_isect(row, {'Processing', *all_param_keys})
            metrics = ub.dict_isect(row, {x, y})
            result = result_analysis.Result(None, params, metrics)
            results_list.append(result)
        self = analysis = result_analysis.ResultAnalysis(
            results_list, default_objective='max', metrics={'BAS_F1'})
        comparable_data = []
        for group in self.abalation_groups('Processing'):
            print(len(group))
            if len(group['Processing'].unique()) > 1:
                comparable_data.append(group)
        comparable = pd.concat(comparable_data)
        data = data.iloc[comparable.index]

    if 0:
        # Remove duplicates for clarity
        rows = []
        for model, group in data.groupby('model'):
            if len(group) > 1:
                # group.pred_cfg.value_counts()
                # group.trk_cfg.value_counts()
                idx = group[y].argmax()
                row = group.iloc[idx]
            else:
                row = group.iloc[0]
            rows.append(row)
        data = pd.DataFrame(rows)

    results_list = []
    for row in data.to_dict('records'):
        # params = ub.dict_isect(row, {'Processing', *all_param_keys})
        params = ub.dict_isect(row, {'Processing'})
        metrics = ub.dict_isect(row, {x, y})
        result = result_analysis.Result(None, params, metrics)
        results_list.append(result)
    analysis = result_analysis.ResultAnalysis(
        results_list, default_objective='max', metrics={'BAS_F1', 'salient_AP'})

    try:
        analysis.run()
    except TypeError:
        raise

    # kitware_green = '#3caf49'
    # kitware_blue = '#006ab6'
    kitware_green = '#3EAE2B'
    kitware_blue = '#0068C7'

    self = analysis
    conclusions = analysis.conclusions()

    fig = kwplot.figure(fnum=fnum, doclf=True)
    ax = fig.gca()
    palette = {
        'L1': kitware_blue,
        'TA1': kitware_green,
    }
    ax = humanized_scatterplot(human_mapping, data=data, ax=ax, legend=True, palette=palette, **plotkw)
    # nice_type = human_mapping.get(type, type)
    # ax.set_title('TA1 vs L1' + '\n' + '\n'.join(conclusions))
    ax.set_title('TA1 vs L1')
    fname = f'{plot_name}_scatter.png'
    fpath = plot_dpath / fname
    fig.set_size_inches(np.array([6.4, 4.8]) * 1.4)
    fig.tight_layout()
    fig.savefig(fpath)

    bas_conclusion = '\n'.join([c for c in conclusions if 'BAS_F1' in c])

    fnum = fnum + 1
    fig = kwplot.figure(fnum=fnum, doclf=True)
    ax = fig.gca()
    ax.set_title('BAS scores: TA1 vs L1')
    sns.violinplot(data=merged_df, x='Processing', y='BAS_F1', palette=palette)
    ax.set_title('TA1 vs L1' + '\n' + bas_conclusion)
    fname = f'{plot_name}_violin.png'
    fpath = plot_dpath / fname
    fig.set_size_inches(np.array([6.4, 4.8]) * 1.0)
    fig.tight_layout()
    fig.savefig(fpath)
    cropwhite_ondisk(fpath)

    fnum = fnum + 1
    fig = kwplot.figure(fnum=fnum, doclf=True)
    ax = fig.gca()
    sns.boxplot(data=merged_df, x='Processing', y='BAS_F1', palette=palette)
    ax.set_title('TA1 vs L1' + '\n' + bas_conclusion)
    fname = f'{plot_name}_boxwhisker.png'
    fpath = plot_dpath / fname
    fig.set_size_inches(np.array([6.4, 4.8]) * 1.0)
    fig.tight_layout()
    fig.savefig(fpath)
    cropwhite_ondisk(fpath)

    # ax.set_title('TA1 vs L1' + '\n' + '\n'.join(conclusions))
