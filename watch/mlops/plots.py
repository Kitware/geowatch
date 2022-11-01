import kwcoco
# import math
import numpy as np
import pandas as pd
import ubelt as ub
from watch import heuristics

fit_param_keys = heuristics.fit_param_keys
pred_param_keys = heuristics.pred_param_keys
trk_param_keys = heuristics.trk_param_keys
act_param_keys = heuristics.act_param_keys
DSET_CODE_TO_GSD = heuristics.DSET_CODE_TO_GSD


class UnableToPlot(Exception):
    pass


class Plotter:
    """
    A WIP refactor to hold the information we need to do plotting.
    """
    def __init__(plotter):
        pass

    @classmethod
    def from_reporter(cls, reporter, common_plotkw, dpath):
        plotter = Plotter()
        orig_merged_df = reporter.orig_merged_df
        iarpa_metric_lut = reporter.iarpa_metric_lut
        pixel_metric_lut = reporter.pixel_metric_lut
        # predcfg_to_label = reporter.predcfg_to_label
        # actcfg_to_label = reporter.actcfg_to_label
        human_mapping = reporter.human_mapping
        merged_df = orig_merged_df.copy()
        plotter.dpath = dpath
        plotter.metric_luts = {
            'pxl': pixel_metric_lut,
            'trk': iarpa_metric_lut,
            'act': iarpa_metric_lut,
        }
        common_plotkw = common_plotkw
        plotter.common_plotkw = common_plotkw
        plotter.human_mapping = human_mapping
        plotter.merged_df = merged_df

        # It's important to separate results by what dataset they were tested
        # on / what type of result they were evaluating
        plotter.group_keys = ['test_trk_dset', 'type']
        plotter.expt_groups = dict(list(merged_df.groupby(plotter.group_keys)))
        return plotter

    def plot_groups(plotter, plot_name, **kwargs):
        """
        plot_name = 'metric_over_training_time'
        metrics = ['BAS_F1']
        """
        plot_method = getattr(plotter, plot_name)
        for code_type, group in plotter.expt_groups.items():
            print(f'code_type={code_type}')
            try:
                plot_method(code_type, group, **kwargs)
            except UnableToPlot as ex:
                print(f'ex={ex}')
                continue

    def metric_over_training_time(plotter, code_type, group, metrics, huevar='expt'):
        test_dset, type = code_type
        metrics = metrics if ub.iterable(metrics) else [metrics]
        metrics_key = '_'.join(metrics)
        plot_name = 'metric_over_time_' + metrics_key
        prefix = f'{test_dset}_{type}_'
        plotkw = ub.udict({
            'x': 'step',
            'y': 'value',
            'star': 'in_production',
        })

        missing = set((plotkw & {'x'}).values()) - set(group.columns)
        if missing:
            raise UnableToPlot(f'Cannot plot {plot_name} for {code_type} missing={missing}')

        missing = set((plotkw & {'x'}).values()) - set(group.columns)
        if missing:
            raise UnableToPlot(f'Cannot plot {plot_name} for {code_type} missing={missing}')

        if plotkw['x'] not in group.columns:
            raise UnableToPlot

        if all(m not in group.columns for m in metrics):
            raise UnableToPlot

        melted = group.melt(
            ['step', 'in_production', 'expt', 'pred_cfg'],
            metrics, var_name='metric')

        if len(metrics) > 1:
            plotkw['style'] = 'metric'

        plotkw['hue'] = huevar

        def make_fig(fnum, legend=True):
            import kwplot
            fig = kwplot.figure(fnum=fnum, doclf=True)
            ax = fig.gca()
            humanized_scatterplot(plotter.human_mapping, plot_type='line',
                                  data=melted, ax=ax, legend=0, **plotkw)
            humanized_scatterplot(plotter.human_mapping, plot_type='scatter',
                                  data=melted, ax=ax, legend=legend,  s=80,
                                  **plotkw)
            if len(metrics) == 1:
                ax.set_ylabel(metrics[0])
            nice_type = plotter.human_mapping.get(type, type)
            ax.set_title(f'Metric over time - {nice_type} - {test_dset}')

        fnum = plot_name + prefix
        run_make_fig(make_fig, fnum, plotter.dpath, plotter.human_mapping,
                     plot_name, prefix)

    def plot_relationship(plotter, code_type, group, huevar='sensorchan'):
        import kwplot
        plot_name = 'plot_relationship'

        x = 'act.poly.metrics.micro_f1'
        y = 'trk.poly.metrics.bas_f1'

        test_dset, type = code_type
        plotkw = ub.udict({
            # 'x': plotter.metric_luts['pxl'][type],
            # 'y': plotter.metric_luts['trk'][type],
            'x': x,
            'y': y,
            'hue': huevar,
            **plotter.common_plotkw,
        })

        missing = set((plotkw & {'x', 'y'}).values()) - set(group.columns)
        if missing:
            raise UnableToPlot(f'Cannot plot {plot_name} for {code_type} missing={missing}')

        if plotkw['x'] not in group.columns or plotkw['y'] not in group.columns:
            raise UnableToPlot

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

        if huevar not in {'auto', 'random'}:
            allow_magic_huevar = True
            if allow_magic_huevar:
                if len(data[huevar].unique()) <= 1:
                    huevar = 'random'

        if huevar == 'random':
            import random
            huevar = random.choice(plotter.analysis.statistics)['param_name']
            # huevar = plotter.analysis.statistics[-1]['param_name']
            # huevar.replace('pxl', 'pred')
            plotkw['hue'] = huevar

        if huevar == 'auto':
            huevar = plotter.analysis.statistics[-1]['param_name']
            # huevar.replace('pxl', 'pred')
            plotkw['hue'] = huevar

        def make_fig(fnum, legend=True):
            fig = kwplot.figure(fnum=fnum, doclf=True)
            ax = fig.gca()
            n = len(data)
            ax = humanized_scatterplot(plotter.human_mapping, data=data, ax=ax,
                                       legend=legend, **plotkw)
            nice_type = plotter.human_mapping.get(type, type)
            ax.set_title(f'{nice_type} - {test_dset}\n{corr_lbl}, n={n}')

        prefix = f'{test_dset}_{type}_{huevar}'
        fnum = plot_name + prefix
        dpath = plotter.dpath
        run_make_fig(make_fig, fnum, dpath, plotter.human_mapping, plot_name, prefix)

    def plot_pixel_ap_verus_iarpa(plotter, code_type, group, huevar='sensorchan'):
        import kwplot
        plot_name = 'pxl_vs_iarpa'

        test_dset, type = code_type
        plotkw = ub.udict({
            'x': plotter.metric_luts['pxl'][type],
            'y': plotter.metric_luts['trk'][type],
            'hue': huevar,
            **plotter.common_plotkw,
        })

        missing = set((plotkw & {'x', 'y'}).values()) - set(group.columns)
        if missing:
            raise UnableToPlot(f'Cannot plot {plot_name} for {code_type} missing={missing}')

        if plotkw['x'] not in group.columns or plotkw['y'] not in group.columns:
            raise UnableToPlot

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

        if huevar not in {'auto', 'random'}:
            allow_magic_huevar = True
            if allow_magic_huevar:
                if len(data[huevar].unique()) <= 1:
                    huevar = 'random'

        if huevar == 'random':
            import random
            huevar = random.choice(plotter.analysis.statistics)['param_name']
            # huevar = plotter.analysis.statistics[-1]['param_name']
            # huevar.replace('pxl', 'pred')
            plotkw['hue'] = huevar

        if huevar == 'auto':
            huevar = plotter.analysis.statistics[-1]['param_name']
            # huevar.replace('pxl', 'pred')
            plotkw['hue'] = huevar

        def make_fig(fnum, legend=True):
            fig = kwplot.figure(fnum=fnum, doclf=True)
            ax = fig.gca()
            n = len(data)
            ax = humanized_scatterplot(plotter.human_mapping, data=data, ax=ax,
                                       legend=legend, **plotkw)
            nice_type = plotter.human_mapping.get(type, type)
            ax.set_title(f'Pixelwise Vs IARPA metrics - {nice_type} - {test_dset}\n{corr_lbl}, n={n}')

        prefix = f'{test_dset}_{type}_{huevar}'
        fnum = plot_name + prefix
        dpath = plotter.dpath
        run_make_fig(make_fig, fnum, dpath, plotter.human_mapping, plot_name, prefix)

    def plot_pixel_ap_verus_auc(plotter, code_type, group, huevar='sensorchan'):
        import kwplot
        plot_name = 'pxl_vs_auc'
        # group = group[~group['sensorchan'].isnull()]
        # group['has_teamfeat'] = group['sensorchan'].apply(lambda x: (('depth' in x) or ('invariants' in x) or ('matseg' in x) or ('land' in x)))

        test_dset, type = code_type
        if type == 'eval_act+pxl':
            plotkw = ub.udict({
                'x': plotter.metric_luts['pxl'][type],
                'y': 'coi_mAUC',
                'hue': huevar,
                **plotter.common_plotkw,
            })
        elif type == 'eval_trk+pxl':
            plotkw = ub.udict({
                'x': plotter.metric_luts['pxl'][type],
                'y': 'salient_AUC',
                'hue': huevar,
                **plotter.common_plotkw,
            })
        else:
            raise KeyError(type)
        missing = set((plotkw & {'x', 'y'}).values()) - set(group.columns)
        if missing:
            raise UnableToPlot(f'Cannot plot plot_pixel_ap_verus_auc for {code_type} missing={missing}')

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
            ax = humanized_scatterplot(plotter.human_mapping, data=group,
                                       ax=ax, legend=legend, **plotkw)
            nice_type = plotter.human_mapping.get(type, type)
            ax.set_title(f'Pixelwise metrics - {nice_type} - {test_dset}\n{corr_lbl}')
            fig.set_size_inches(16.85, 8.82)

        prefix = f'{test_dset}_{type}_{huevar}'
        fnum = 'plot_pixel_ap_verus_auc' + prefix
        run_make_fig(make_fig, fnum, plotter.dpath, plotter.human_mapping, plot_name, prefix)

    def plot_param_analysis(plotter, code_type, group, metrics, params_of_interest=None):
        """
        metrics = ['salient_AP']
        metrics = ['BAS_F1']
        """
        test_dset, type = code_type
        metrics = metrics if ub.iterable(metrics) else [metrics]
        metrics_key = '_'.join(metrics)
        plot_name = 'violin_' + metrics_key
        prefix = f'{test_dset}_{type}_'

        metrics_key = '_'.join(metrics)
        # pred_param_df = pd.DataFrame(group['pred_params'].tolist(), index=group.index)
        # track_param_df = pd.DataFrame(group['track_params'].tolist(), index=group.index)
        # fit_param_df = pd.DataFrame(group['fit_params'].tolist(), index=group.index)
        # varied_params = ub.varied_values(pred_param_df.to_dict('records'), min_variations=1)
        # blocklist = ub.oset(['step', 'epoch', 'pred_in_dataset_name'])
        # pred_param_df = pred_param_df.drop(blocklist & pred_param_df.columns, axis=1)

        # main_cols = ub.oset(['expt', 'model', 'step', 'test_dset'] + metrics)
        expanded = group

        if params_of_interest is None:
            params_of_interest = [
                'pred_use_cloudmask',
                'pred_resample_invalid_frames',
                'pred_input_space_scale',
                'pred_window_space_scale',
                'trk_thresh',
            ]

        x = 'act.pxl.properties.model_name'

        additional_needed_legends = []

        def make_make_fig(expanded, param_name):
            def make_fig(fnum, legend=True):
                fig = kwplot.figure(fnum=fnum, doclf=True)
                ax = fig.gca()
                n = len(expanded)

                assert len(metrics) == 1

                # sns.violinplot(data=expanded, x=param_name, y=metrics[0], hue='expt')
                # sns.violinplot(data=expanded, x='expt', y=metrics[0], hue=param_name, split=True)
                # sns.boxplot(data=expanded, x='expt', y=metrics[0], hue=param_name, notch=True)

                if param_name not in expanded.columns:
                    raise UnableToPlot

                sns.boxplot(data=expanded, x=x, y=metrics[0], hue=param_name,
                            medianprops={"color": "coral"})
                if not legend:
                    ax.get_legend().remove()
                humanize_axes(ax, plotter.human_mapping)

                # Relabel if it's too big to fit.
                relabels = {}
                needs_relabel = False
                from itertools import count
                counter = count(1)

                xtick_labels = list(ax.get_xticklabels())
                n_xticks = len(xtick_labels)
                if n_xticks == 1:
                    len_thresh = 60
                else:
                    len_thresh = 10

                for label in xtick_labels:
                    text = label.get_text()
                    if len(text) > len_thresh:
                        needs_relabel = True
                        relabels[text] = str(next(counter))

                if needs_relabel:
                    additional = {
                        'relabels': relabels,
                        'param_name': param_name,
                    }
                    additional_needed_legends.append(additional)
                    new_labels = []
                    for label in ax.get_xticklabels():
                        text = label.get_text()
                        if text in relabels:
                            label.set_text(relabels[text])
                        new_labels.append(label)
                    ax.set_xticklabels(new_labels)

                # for label in ax.get_xticklabels():
                #     label.set_rotation(90)

                nice_type = plotter.human_mapping.get(type, type)
                nice_param_name = plotter.human_mapping.get(param_name, param_name)
                ax.set_title(f'Varied {nice_param_name} - {nice_type}\n{test_dset}\nn={n}')
                fig.set_size_inches(12.85, 8.82)
                fig.tight_layout()
            return make_fig

        import kwplot
        import seaborn as sns
        for param_name in params_of_interest:
            try:
                expanded[[param_name] + metrics]
            except KeyError:
                continue
            make_fig = make_make_fig(expanded, param_name)
            prefix = f'{test_dset}_{type}_'
            fnum = plot_name + param_name + prefix
            dpath = plotter.dpath
            try:
                run_make_fig(make_fig, fnum, dpath, plotter.human_mapping, plot_name + param_name, prefix)
            except Exception:
                print(f'Error checking {param_name}')
                ...
            # sns.violinplot(data=expanded, x=x, y=metrics[0], hue=param_name,
            #                medianprops={"color": "coral"})

        # In the case we needed to relabel an axis, we also need to write a
        # figure indicating what that relabel was.
        for additional in additional_needed_legends:
            relabels = additional['relabels']
            param_name = additional['param_name']
            df = pd.DataFrame([{'orig': k, 'relabel': v} for k, v in relabels.items()])
            df = df.set_index('orig')
            df2_style = humanize_dataframe(df, title='Relabling')

            # Hack, write the new figure to the parts dir. We have to be sure
            # to reconstruct it correctly.

            plot_dpath_parts = (dpath / (plot_name + param_name + '_parts')).ensuredir()
            fpath = plot_dpath_parts / prefix + 'relabel.png'
            dfi_table(df2_style, fpath, fontsize=32, show=False)

    def plot_resource_versus_metric(plotter, code_type, group, huevar='sensorchan'):
        import kwplot
        plot_name = 'resource_vs_metric'

        resources_of_interest = [
            'trk.pxl.resource.total_hours',
            'trk.pxl.resource.co2_kg',
            'act.pxl.resource.total_hours',
            'act.pxl.resource.co2_kg',
            'act.poly.resource.total_hours',
            'trk.poly.resource.total_hours',
        ]

        metrics = [
            'trk.poly.metrics.macro_f1',
        ]

        for resource_type in resources_of_interest:
            human_resource_type = plotter.human_mapping.get(resource_type, resource_type)

            for metric in metrics:
                human_metric_type = plotter.human_mapping.get(metric, metric)

                # group['pred_tta_time'] = group['pred_tta_time'].astype(str)
                # group['pred_tta_fliprot'] = group['pred_tta_fliprot'].astype(str)
                # group.loc[group['pred_tta_time'] == 'nan', 'pred_tta_time'] = '0.0'
                # group.loc[group['pred_tta_fliprot'] == 'nan', 'pred_tta_fliprot'] = '0.0'

                test_dset, type = code_type
                plotkw = ub.udict({
                    'x': resource_type,
                    'y': metric,
                    'hue': huevar,
                    **plotter.common_plotkw,
                    'style': 'hardware',
                })

                missing = set((plotkw & {'x', 'y'}).values()) - set(group.columns)
                if missing:
                    print(f'Cannot plot plot_resource_versus_metric for {code_type} missing={missing}')
                    continue

                def make_fig(fnum, legend=True):
                    fig = kwplot.figure(fnum=fnum)
                    ax = fig.gca()
                    ax = humanized_scatterplot(plotter.human_mapping, data=group, ax=ax, **plotkw)
                    nice_type = plotter.human_mapping.get(type, type)
                    ax.set_title(f'{human_resource_type} vs {human_metric_type} - {nice_type} - {test_dset}')
                    fig.set_size_inches(np.array([6.4, 4.8]) * 1.4)

                prefix = f'{test_dset}_{type}_'
                fnum = 'plot_resource_versus_metric_' + resource_type + prefix
                run_make_fig(make_fig, fnum, plotter.dpath, plotter.human_mapping, plot_name, prefix)


def humanize_dataframe(df, col_formats=None, human_labels=None, index_format=None,
                       title=None):
    import humanize
    df2 = df.copy()
    if col_formats is not None:
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
    if human_labels:
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


def humanize_axes(ax, human_mapping):
    xkey = ax.get_xlabel()
    ykey = ax.get_ylabel()
    ax.set_xlabel(human_mapping.get(xkey, xkey))
    ax.set_ylabel(human_mapping.get(ykey, ykey))
    legend = ax.get_legend()
    if legend is not None:
        humanize_legend(legend, human_mapping)


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
        if star in data:
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

    humanize_axes(ax, human_mapping)

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

    expt_dvc_dpath = watch.find_smart_dvc_dpath()
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
        bundle_dpath = expt_dvc_dpath / bundle_name
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

    # type_to_rows = dict(list(reporter.orig_merged_df.groupby('type')))
    # type_to_resource = {}
    # pairs = [
    #     ('eval_trk_poly_fpath', 'trk.poly.resource.total_hours'),
    #     ('eval_act_poly_fpath', 'trk.poly.resource.total_hours'),
    # ]
    # if k1 in type_to_rows:
    #     type_to_rows[k1][k2].sum()

    reporter.orig_merged_df['trk.pxl.resource.co2_kg']
    reporter.orig_merged_df['trk.pxl.resource.total_hours']
    reporter.orig_merged_df['act.pxl.resource.co2_kg']
    reporter.orig_merged_df['act.pxl.resource.total_hours']

    # for row in reporter.merg
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


def run_make_fig(make_fig, fnum, dpath, human_mapping, plot_name, prefix):
    """
    Runs a function that plots a figure with and without a legend.
    Also saves the legend to its own file.
    """
    import kwplot
    plt = kwplot.autoplt()

    plot_dpath_main = (dpath / plot_name).ensuredir()
    plot_dpath_parts = (dpath / (plot_name + '_parts')).ensuredir()

    print(f'fnum={fnum}')
    make_fig(str(fnum) + '_legend', legend=True)
    fig = plt.gcf()
    # fname = f'{prefix}{plot_name}.png'
    fname = f'{fnum}.png'
    fpath = plot_dpath_main / fname
    fig.set_size_inches(np.array([6.4, 4.8]) * 1.4)
    fig.tight_layout()
    fig.savefig(fpath)

    SAVE_PARTS = 1
    if SAVE_PARTS:
        ax = fig.gca()
        orig_legend = ax.get_legend()
        orig_legend_title = orig_legend.get_title().get_text()
        legend_handles = ax.get_legend_handles_labels()

        make_fig(str(fnum) + '_nolegend', legend=False)
        fig_nolegend = plt.gcf()
        fname = f'{prefix}{plot_name}_nolegend.png'
        fpath = plot_dpath_parts / fname
        fig_nolegend.set_size_inches(np.array([6.4, 4.8]) * 1.4)
        fig_nolegend.tight_layout()
        fig_nolegend.savefig(fpath)

        fig_onlylegend = kwplot.figure(
            fnum=str(fnum) + '_onlylegend', doclf=1)
        ax2 = fig_onlylegend.gca()
        ax2.axis('off')
        new_legend = ax2.legend(*legend_handles, title=orig_legend_title,
                                loc='lower center')
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
