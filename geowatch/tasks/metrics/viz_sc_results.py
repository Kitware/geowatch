import pandas as pd
import numpy as np
import ubelt as ub
import kwplot
import kwimage
from matplotlib.collections import LineCollection
from matplotlib.colors import to_rgba
from matplotlib.dates import date2num
from typing import Literal
import datetime as datetime_mod


def viz_sc(region_dpaths, true_site_dpath, true_region_dpath, save_dpath):
    """
    Seems broken in 1.0.0

    Ignore:
        >>> # xdoctest: +REQUIRES(module:iarpa_smart_metrics)
        >>> from geowatch.cli.run_metrics_framework import *  # NOQA
        >>> from geowatch.demo.metrics_demo.generate_demodata import generate_demo_metrics_framework_data
        >>> cmdline = 0
        >>> base_dpath = ub.Path.appdir('geowatch', 'tests', 'test-iarpa-metrics2')
        >>> data_dpath = base_dpath / 'inputs'
        >>> dpath = base_dpath / 'outputs'
        >>> demo_info3 = generate_demo_metrics_framework_data(
        >>>     roi='DR_R003',
        >>>     num_sites=11, num_observations=10, noise=3, p_observe=0.5,
        >>>     p_transition=0.2, drop_noise=0.3, drop_limit=0.5)
        >>> print('demo_info3 = {}'.format(ub.urepr(demo_info3, nl=1)))
        >>> out_dpath = dpath / 'region_metrics'
        >>> merge_fpath = dpath / 'merged.json'
        >>> out_dpath.delete()
        >>> kwargs = {
        >>>     'pred_sites': demo_info3['pred_site_dpath'],
        >>>     'true_region_dpath': demo_info3['true_region_dpath'],
        >>>     'true_site_dpath': demo_info3['true_site_dpath'],
        >>>     'merge': True,
        >>>     'merge_fpath': merge_fpath,
        >>>     'out_dir': out_dpath,
        >>> }
        >>> main(cmdline=False, **kwargs)
        >>> #
        >>> from geowatch.tasks.metrics.viz_sc_results import *  # NOQA
        >>> region_dpaths = [save_dpath / 'region_metrics/DR_R003']
        >>> save_dpath = merge_fpath.parent
        >>> true_site_dpath = demo_info3['true_site_dpath']
        >>> true_region_dpath = demo_info3['true_region_dpath']
        >>> viz_sc(region_dpaths, true_site_dpath, true_region_dpath, save_dpath)
        >>> # TODO: visualize
    """
    from geowatch.tasks.metrics.merge_iarpa_metrics import RegionResult
    kwplot.autompl()

    results = []
    for pth in region_dpaths:
        try:
            results.append(
                RegionResult.from_dpath_and_anns_root(
                    pth, true_site_dpath, true_region_dpath)
            )
        except FileNotFoundError:
            print(f'warning: missing region {pth}')

    # merge SC
    sc_results = [r for r in results if r.sc_dpath]

    # check out:
    # kwimage.stack_image
    # kwplot.make_legend_image

    phs = [ph for r in sc_results if (ph := r.sc_phasetable) is not None]

    for ph in ub.ProgIter(phs, desc='visualize sc gantt'):

        rid = ph.index.get_level_values('region_id')[0]

        # site-level viz
        for (site, site_cand), df_ in ph.groupby(['site', 'site_candidate']):
            df = df_.droplevel([0, 1, 2])
            viz_sc_gantt(
                df,
                ' vs. '.join((site, site_cand)),
                ((save_dpath / rid).ensuredir() / ('_'.join((site, site_cand)) + '.png'))
            )

        # region-level viz
        df = ph
        plot_title = rid
        save_fpath = (save_dpath / f'sc_{rid}.png')
        viz_sc_multi(ph, plot_title, save_fpath)

    # merged viz
    print('Visualize merged')
    merged_df = pd.concat(phs, axis=0)
    viz_sc_multi(
        merged_df,
        ' '.join(merged_df.index.unique(level='region_id')),
        (save_dpath / 'sc_merged.png')
    )


def viz_sc_gantt(df, plot_title, save_fpath):
    from geowatch.heuristics import PHASES as phases
    plt = kwplot.autoplt()
    sns = kwplot.autosns()
    # TODO how to pick site boundary?
    df = df.apply(lambda s: s.str.split(', '))

    df['pred'] = df['pred'].fillna(method='ffill')
    # df = df[~df['true'].isna()]
    # df['pred'] = df['pred'].fillna('Unknown')
    df['pred'] = df['pred'].fillna(method='bfill')
    df = df.reset_index()
    df['date'] = pd.to_datetime(df['date']).dt.date
    df = df.melt(id_vars=['date'])
    df = df.dropna()

    # split out subsites
    subsite_ixs = df['value'].str.len() > 1
    var, val = [], []
    for _, row in df.loc[subsite_ixs].iterrows():
        for v in row['value']:
            var.append(row['variable'])
            val.append(v)
    df = df.loc[~subsite_ixs]
    df['value'] = df['value'].str[0]
    df = pd.concat(
        (df,
         pd.DataFrame(dict(variable=var, value=val))),
        axis=0,
        ignore_index=True,
    )

    # order hack for relplot
    phases_type = pd.api.types.CategoricalDtype(
        (['Unknown'] + phases)[::-1], ordered=True)
    df['value'] = df['value'].astype(phases_type)
    df = df.sort_values(by='value')

    # TODO threadsafe
    grid = sns.relplot(
        data=df,
        x='date',
        y='value',
        hue='variable',
        size='variable'
    )
    plt.title(plot_title)
    grid.savefig(save_fpath)
    plt.close()


def viz_sc_multi(ph, plot_title, save_fpath,
                 date: Literal['absolute', 'from_start', 'from_active'] = 'absolute',
                 how: Literal['residual', 'strip'] = 'strip'):
    from geowatch.heuristics import PHASES as phases
    import geowatch.heuristics
    plt = kwplot.autoplt()
    sns = kwplot.autosns()
    df = ph

    # df.index = [df.index.map('{0[0]} {0[1]} {0[2]}'.format), df.index.get_level_values(3)]

    # ignore subsites
    # TODO how to pick site boundary?
    df = df.apply(lambda s: s.str.split(', ').str[0])

    # could do just region_id for error bars after fillna
    df = df.reset_index()
    df['group'] = df[['region_id', 'site', 'site_candidate']].astype('string').agg('\n'.join, axis=1)
    df = df.drop(['region_id', 'site', 'site_candidate'], axis=1)

    if how == 'residual':  # true and pred must be aligned
        df['pred'] = (df.groupby('group')['pred']
                        .fillna(method='ffill')
                        .fillna('No Activity'))
        # df['pred'] = df['pred'].fillna('Unknown')
        # df['pred'] = df['pred'].fillna(method='bfill')

        df = df[~df['true'].isna()]
        # df = df[df['true'] != 'Unknown']  # Unk should be gone from df after this

    df['date'] = pd.to_datetime(df['date'])  # .dt.date

    # must do this before searchsorted
    phases_type = pd.api.types.CategoricalDtype(
        ['Unknown'] + phases, ordered=True)
    df['pred'] = df['pred'].astype(phases_type)
    df['true'] = df['true'].astype(phases_type)
    # assert df.groupby('group')['true'].is_monotonic_increasing.all()
    # assert df.groupby('group')['pred'].is_monotonic_increasing.all()

    if date == 'absolute':  # absolute date
        df['date'] = df['date'].dt.date
        x_var = 'date'
    elif date == 'from_start':  # relative date since start
        df['date'] = df.groupby('group')['date'].transform(lambda date: date - date.iloc[0])
        # df['date'] = df['date'].dt.days.fillna(0)
        df['date'] = df['date'].dt.days
        x_var = 'days since start'
    elif date == 'from_active':  # relative date since last NA; requires 1 NA to exist before SP
        df = df.groupby('group').apply(_align_start)
        df['date'] = df['date'].dt.days
        x_var = 'days since final No Activity'

    palette = {c['name']: c['color'] for c in geowatch.heuristics.CATEGORIES}

    if how == 'residual':
        jitter = 0.1
        df['diff'] = df['pred'].cat.codes - df['true'].cat.codes
        df['diff'] += np.random.uniform(low=-jitter, high=jitter, size=len(df['diff']))

        # TODO threadsafe
        grid = sns.relplot(
            kind='scatter',
            data=df,
            x='date',
            y='diff',
            hue='true',
            palette=palette,
            s=10,
        )
        grid.map(_add_colored_linesegments,
                 'date', 'diff', 'true', 'group',)
        y_var = 'pred phases ahead of true phase'
    elif how == 'strip':
        # get tp idxs before reshaping
        tps = df.groupby('group').apply(_tp_idxs)

        dfm = df.melt(id_vars=['date', 'group'], value_name='phase').dropna()
        dfm['phase'] = dfm['phase'].astype(phases_type)
        dfm['yval'], ylabels = pd.factorize(dfm['group'])
        tps.index = tps.index.map(dict(zip(ylabels, range(len(ylabels)))))
        with pd.option_context('mode.chained_assignment', None):
            dfm['yval'].loc[dfm['variable'] == 'pred'] -= 0.2
        grid = sns.relplot(
            kind='scatter',
            data=dfm,
            x='date',
            y='yval',
            hue='phase',
            palette=palette,
            s=10,
            facet_kws=dict(legend_out=False),
        )
        grid.map(_add_colored_linesegments,
                 'date', 'yval', 'phase', 'yval', palette=palette)

        grid.map(_highlight_tp, 'yval', tps=tps)
        if len(ylabels) <= 20:  # draw site names if they'll be readable
            grid.set(yticks=range(len(ylabels)))
            grid.set_yticklabels(ylabels, size=4)
        # and write them as an index
        yregion, ytrue, ypred = np.array(ylabels.str.split('\n').to_list()).T
        pd.DataFrame(dict(
            index=range(len(ylabels)),
            region=yregion,
            true=ytrue,
            pred=ypred,
        )).to_csv(save_fpath.with_suffix('.index.csv'))
        y_var = '[region, true, pred]'

    # df['group_phase'] = df[['group', 'true']].agg('_'.join, axis=1)

    sns.move_legend(grid, 'upper right')
    grid.set_axis_labels(x_var=x_var, y_var=y_var)
    plt.title(plot_title)
    grid.savefig(save_fpath)
    plt.close()


def _align_start(grp, phase='Site Preparation', before=True):
    grp = grp.sort_values(by='date')
    # assert grp['true'].iloc[0] != phase, grp['true']
    grp['date'] -= grp.iloc[grp['true'].searchsorted(phase) - int(before)]['date']
    return grp


def _tp_idxs(grp):
    grp = grp.sort_values(by='date')
    grp['pred'] = (grp['pred']
                   .fillna(method='ffill')
                   .fillna('No Activity'))
    grp = grp[~grp['true'].isna()]
    match = grp['pred'] == grp['true']
    ixs = grp[match.shift() != match].index.values
    x = grp['date'].map(date2num)
    blocks = [x.loc[[start, end]] for start, end, matches in zip(ixs, ixs[1:], match[ixs]) if matches]
    return blocks


def _highlight_tp(y, **kwargs):
    tps = kwargs.get('tps')
    sites = y.round().abs().astype(int).unique()
    for site in sites:
        boxes = kwimage.Boxes([[*xs, site - 0.5, site + 0.5] for xs in tps.loc[site]], format='xxyy')
        kwplot.draw_boxes(boxes, color='green', fill=True, lw=0, alpha=0.3)


def _add_colored_linesegments(x, y, phase, units, **kwargs):
    """
    x = dfm['date']
    y = dfm['yval']
    phase = dfm['phase']
    units = dfm['yval']
    """
    # need args instead of kwargs because of grid.map() weirdness
    plt = kwplot.autoplt()
    palette = kwargs.get('palette')

    if isinstance(x.iloc[0], datetime_mod.date):
        x = date2num(x)

    _df = pd.DataFrame(dict(
        xy=zip(x, y),
        hue=pd.Series(phase).map(palette).astype('string').map(to_rgba),
        phase=phase,  # need to keep this due to float comparisons in searchsorted
        units=units,
    ))

    lines = []
    colors = []
    for unit, grp in _df.groupby('units'):
        # drop consecutive dups
        ph = grp['phase']
        ixs = ph.loc[ph.shift() != ph].index.values
        for start, end, hue in zip(
                ixs,
                ixs[1:].tolist() + [None],
                grp['hue'].loc[ixs]
        ):
            line = grp['xy'].loc[start:end].values.tolist()
            if len(line) > 0:
                lines.append(line)
                colors.append(hue)

    lc = LineCollection(lines, alpha=0.5, colors=colors)
    ax = plt.gca()
    ax.add_collection(lc)
    # ax.autoscale()
