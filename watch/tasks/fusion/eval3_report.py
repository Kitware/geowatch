"""

python -m watch.tasks.fusion.dvc_sync_manager "push pull evals"

"""
import ubelt as ub
import math
import numpy as np
import pandas as pd
import functools  # NOQA


# TODO: move to heuristics
fit_param_keys = [
    'bad_channels', 'sensorchan', 'channels', 'time_steps',
    'chip_size', 'chip_overlap', 'arch_name', 'optimizer',
    'time_sampling', 'time_span', 'true_multimodal',
    'accumulate_grad_batches', 'modulate_class_weights', 'tokenizer',
    'use_grid_positives', 'use_cloudmask', 'upweight_centers',
    'temporal_dropout', 'stream_channels', 'saliency_loss',
    'class_loss', 'init', 'learning_rate', 'decoder',
]
pred_param_keys = [
    'pred_tta_fliprot',
    'pred_tta_time',
    'pred_chip_overlap',
]
trk_param_keys = [
    'trk_thresh',
    'trk_morph_kernel',
    'trk_agg_fn',
    'trk_thresh_hysteresis',
    'trk_moving_window_size',
]
act_param_keys = [
    'trk_use_viterbi',
    'trk_thresh',
]


def eval3_report():
    import kwplot
    kwplot.autosns()
    import watch
    try:
        dvc_dpath = watch.find_smart_dvc_dpath(hardware='hdd')
    except Exception:
        dvc_dpath = watch.find_smart_dvc_dpath()

    from watch.tasks.fusion import dvc_sync_manager
    dvc_manager = dvc_sync_manager.DVCSyncManager.coerce(dvc_dpath)
    table = dvc_manager.evaluation_table()

    description = table[['type', 'dataset_code', 'expt', 'model', 'pred_cfg', 'act_cfg', 'trk_cfg']].describe()
    print(description)

    dset_code_to_gsd = {
        'Aligned-Drop3-L1': 10.0,
        'Aligned-Drop3-TA1-2022-03-10': 10.0,
        'Cropped-Drop3-TA1-2022-03-10': 1.0,
    }
    summary_stats = []
    for dset_code, group in table.groupby(['dataset_code']):
        gsd = dset_code_to_gsd.get(dset_code, np.nan)
        table.loc[group.index, 'gsd'] = gsd

        type_hist = group.groupby('type').size()
        model_hist = group.groupby('model').size()
        expt_hist = group.groupby('expt').size()

        row = {
            'dataset_code': dset_code,
            'gsd': gsd,
            'num_experiments': len(expt_hist),
            'num_models': len(model_hist),
            'num_pxl_evals': type_hist.get('pxl', 0),
            'num_bas_evals': type_hist.get('trk', 0),
            'num_sc_evals': type_hist.get('act', 0),
        }
        summary_stats.append(row)
    _summary_df = pd.DataFrame(summary_stats)
    total_row = _summary_df.sum().to_dict()
    total_row['gsd'] = '*'
    total_row['dataset_code'] = '*'
    summary_df = pd.DataFrame(summary_stats + [total_row])
    print('Number of Models & Evaluations')
    print(summary_df.to_string(index=False))

    evaluations = table[~table['raw'].isnull()]
    raw_df = pd.DataFrame(evaluations)

    if 0:
        col_stats_df = unique_col_stats(raw_df)
        print('Column Unique Value Frequencies')
        print(col_stats_df.to_string())

        if len(group) > 1:
            print(group)

    test_dset_freq = raw_df['test_dset'].value_counts()
    print(f'test_dset_freq={test_dset_freq}')

    preference = {
        'Cropped-Drop3-TA1-2022-03-10_combo_DLM_s2_wv_vali.kwcoco': 0,
        'Cropped-Drop3-TA1-2022-03-10_combo_DL_s2_wv_vali.kwcoco': 1,
        'Cropped-Drop3-TA1-2022-03-10_data_wv_vali.kwcoco': 2,

        'Aligned-Drop3-TA1-2022-03-10_combo_LM_nowv_vali.kwcoco': 0,
        'Aligned-Drop3-TA1-2022-03-10_combo_LM_vali.kwcoco': 1,
    }

    # Filter out rows where models have predictions on "better" datasets
    FILTER_DUPS = 0
    if FILTER_DUPS:
        keep_locs = []
        for k, group in raw_df.groupby(['dataset_code', 'model', 'pred_cfg', 'type']):
            prefs = group['test_dset'].apply(lambda x: preference.get(x, 0))
            keep_flags = prefs == prefs.min()
            keep_locs.extend(group[keep_flags].index)
        print(f'Keep {len(keep_locs)} / {len(raw_df)} drop3 evals')
        df = filt_df = raw_df.loc[keep_locs]
        print('Column Unique Value Frequencies')
        # print(col_stats_df2.to_string())
        num_files_summary(filt_df)
    else:
        filt_df = raw_df.copy()

    # Load detailed data
    eval_types_to_locs = ub.ddict(list)
    for k, group in filt_df.groupby(['dataset_code', 'model', 'pred_cfg']):
        eval_types = tuple(sorted(group['type'].unique()))
        eval_types_to_locs[eval_types].extend(group.index)
    print('Comparable locs')
    print(ub.repr2(ub.map_vals(len, eval_types_to_locs)))
    comparable_locs = list(ub.flatten(v for k, v in eval_types_to_locs.items() if len(k) > 1))
    df = comp_df = filt_df.loc[comparable_locs]
    num_files_summary(comp_df)

    big_rows = load_extended_data(df)
    merged_df, other = clean_loaded_data(big_rows)
    plot_merged(merged_df, other)


def is_null(x):
    return (isinstance(x, float) and math.isnan(x)) or x is None or not bool(x)


def resolve_model_info(model_fpath):
    cacher = ub.Cacher('model_info_memo', depends=[str(model_fpath)], appname='watch')
    stats = cacher.tryload()
    if stats is None:
        from watch.cli.torch_model_stats import torch_model_stats
        stats = torch_model_stats(model_fpath)
        cacher.save(stats)
    return stats


def plot_merged(merged_df, other):
    expt_group = dict(list(merged_df.groupby(['dataset_code', 'type'])))
    import pprint
    for gsd_type, group in expt_group.items():
        gsd, type = gsd_type
        print('Varied fit params')
        print('type = {}'.format(ub.repr2(type, nl=1)))
        print('gsd = {}'.format(ub.repr2(gsd, nl=1)))
        part = group[fit_param_keys].fillna('null')
        part = part.drop('channels', axis=1)
        rows = part.to_dict('records')
        varied = ub.varied_values(rows, 1)
        print(ub.highlight_code(pprint.pformat(dict(varied), width=80)))
        # print('varied = {}'.format(ub.repr2(varied, nl=2)))

        print('Varied pred params')
        print('type = {}'.format(ub.repr2(type, nl=1)))
        print('gsd = {}'.format(ub.repr2(gsd, nl=1)))
        part = group[pred_param_keys].fillna('null')
        rows = part.to_dict('records')
        varied = ub.varied_values(rows, 0)
        print(ub.highlight_code(pprint.pformat(dict(varied), width=80)))

        print('Varied track params')
        print('type = {}'.format(ub.repr2(type, nl=1)))
        print('gsd = {}'.format(ub.repr2(gsd, nl=1)))
        part = group[trk_param_keys].fillna('null')
        rows = part.to_dict('records')
        varied = ub.varied_values(rows, 0)
        print(ub.highlight_code(pprint.pformat(dict(varied), width=80)))

        print('Varied activity params')
        print('type = {}'.format(ub.repr2(type, nl=1)))
        print('gsd = {}'.format(ub.repr2(gsd, nl=1)))
        part = group[act_param_keys].fillna('null')
        rows = part.to_dict('records')
        varied = ub.varied_values(rows, 0)
        print(ub.highlight_code(pprint.pformat(dict(varied), width=80)))

    human_mapping = {
        'coi_mAP': 'Pixelwise mAP (classes of interest)',
        'coi_mAUC': 'Pixelwise mAUC (classes of interest)',
        'salient_AP': 'Pixelwise Salient AP',
        'salient_AUC': 'Pixelwise Salient AUC',
        'mean_f1': 'IARPA SC mean F1',
        'BAS_F1': 'IARPA BAS F1',
        'act_cfg': 'SC Tracking Config',
        'trk_use_viterbi': 'Viterbi Enabled',
        'trk_thresh': 'SC Tracking Threshold',
        'co2_kg': 'CO2 Emissions (kg)',
        'total_hours': 'Time (hours)',
        'sensorchan': 'Sensor/Channel',
        'has_teamfeat': 'Has Team Features',
    }
    actcfg_to_label = other['actcfg_to_label']
    predcfg_to_label = other['predcfg_to_label']
    human_mapping.update(actcfg_to_label)
    human_mapping.update(predcfg_to_label)

    iarpa_metric_lut = {
        'act+pxl': 'mean_f1',
        'trk+pxl': 'BAS_F1',
    }
    pixel_metric_lut = {
        'act+pxl': 'coi_mAP',
        'trk+pxl': 'salient_AP',
    }

    # ['trk_thresh',
    #  'trk_morph_kernel',
    #  'trk_agg_fn',
    #  'trk_thresh_hysteresis',
    #  'trk_moving_window_size']

    markersize = 60

    import kwplot
    expt_group = dict(list(merged_df.groupby(['dataset_code', 'type'])))
    fnum = 0
    for gsd_type, group in expt_group.items():
        dataset_code, type = gsd_type
        if type == 'act+pxl':
            plotkw = {
                'x': pixel_metric_lut[type],
                'y': iarpa_metric_lut[type],
                'hue': 'channels',
                'style': 'has_teamfeat',
                # 'hue': 'trk_use_viterbi',
                # 'style': 'trk_thresh',
                # 'size': 'trk_thresh',
                # 'hue': 'pred_cfg',
                # 'hue': 'expt',
            }
        elif type == 'trk+pxl':
            # hacks
            group['track_agg_fn'] = group['trk_agg_fn'].fillna('probs')
            flags = 1 - group['trk_thresh_hysteresis'].isnull()
            group['trk_thresh_hysteresis'] = group['trk_thresh_hysteresis'].fillna(0) + (flags * group['trk_thresh'])

            plotkw = {
                'x': pixel_metric_lut[type],
                'y': iarpa_metric_lut[type],
                'hue': 'channels',
                'style': 'has_teamfeat',
                # 'hue': 'trk_thresh',
                # 'size': 'trk_thresh_hysteresis',
                # 'style': 'track_agg_fn',
                # 'hue': 'trk_cfg',
                # 'hue': 'pred_cfg',
                # 'hue': 'expt',
            }
        else:
            raise KeyError(type)
        fnum += 1
        fig = kwplot.figure(fnum=fnum)
        ax = fig.gca()

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
        plotkw['s'] = markersize
        ax = humanized_scatterplot(human_mapping, data=group, ax=ax, **plotkw)
        ax.set_title(f'Pixelwise Vs IARPA metrics - {type} - {dataset_code=}\n{corr_lbl}')

    fnum = 10
    import kwplot
    expt_group = dict(list(merged_df.groupby(['dataset_code', 'type'])))
    for gsd_type, group in expt_group.items():

        group = group[~group['sensorchan'].isnull()]
        # group['has_teamfeat'] = group['sensorchan'].apply(lambda x: (('depth' in x) or ('invariants' in x) or ('matseg' in x) or ('land' in x)))

        dataset_code, type = gsd_type
        if type == 'act+pxl':
            plotkw = {
                'x': pixel_metric_lut[type],
                'y': 'coi_mAUC',
                'hue': 'sensorchan',
                'style': 'has_teamfeat',
                # 'hue': 'trk_use_viterbi',
                # 'style': 'trk_thresh',
                # 'size': 'trk_thresh',
                # 'hue': 'pred_cfg',
                # 'hue': 'expt',
            }
        elif type == 'trk+pxl':
            plotkw = {
                'x': pixel_metric_lut[type],
                'y': 'salient_AUC',
                'hue': 'sensorchan',
                'style': 'has_teamfeat',
                # 'hue': 'trk_cfg',
                # 'hue': 'pred_cfg',
                # 'hue': 'expt',
            }
        else:
            raise KeyError(type)
        fnum += 1
        fig = kwplot.figure(fnum=fnum)
        ax = fig.gca()

        metric_corr_mat = group[[plotkw['x'], plotkw['y']]].corr()
        metric_corr = metric_corr_mat.stack()
        metric_corr.name = 'corr'
        stack_idx = metric_corr.index
        valid_idxs = [(a, b) for (a, b) in ub.unique(map(tuple, map(sorted, stack_idx.to_list()))) if a != b]
        metric_corr = metric_corr.loc[valid_idxs]
        # corr_lbl = 'corr({},{})={:0.4f}'.format(*metric_corr.index[0], metric_corr.iloc[0])
        corr_lbl = 'corr={:0.4f}'.format(metric_corr.iloc[0])
        plotkw['s'] = markersize
        ax = humanized_scatterplot(human_mapping, data=group, ax=ax, **plotkw)
        ax.set_title(f'Pixelwise metrics - {type} - {dataset_code=}\n{corr_lbl}')
        fig.set_size_inches(16.85,  8.82)

    expt_group = dict(list(merged_df.groupby(['dataset_code', 'type'])))
    if 1:
        expt_group.pop((10, 'act+pxl'), None)
        expt_group.pop((1, 'act+pxl'), None)
        expt_group.pop((1, 'trk+pxl'), None)
    for resource_type in ['total_hours', 'co2_kg']:
        human_resource_type = human_mapping.get(resource_type, resource_type)

        # for metric_type in ['iarpa', 'pixel']:
        for metric_type in ['pixel']:
            if metric_type == 'iarpa':
                metric_lut = iarpa_metric_lut
                human_metric_type = 'IARPA'
            else:
                metric_lut = pixel_metric_lut
                human_metric_type = 'Pixelwise'

            fnum += 1
            fig = kwplot.figure(fnum=fnum)
            pnum_ = kwplot.PlotNums(nCols=len(expt_group))
            for gsd_type, group in expt_group.items():

                group['pred_tta_time'] = group['pred_tta_time'].astype(str)
                group['pred_tta_fliprot'] = group['pred_tta_fliprot'].astype(str)

                group.loc[group['pred_tta_time'] == 'nan', 'pred_tta_time'] = '0.0'
                group.loc[group['pred_tta_fliprot'] == 'nan', 'pred_tta_fliprot'] = '0.0'
                dataset_code, type = gsd_type
                if type == 'act+pxl':
                    plotkw = {
                        'x': resource_type,
                        'y': metric_lut[type],
                        'hue': 'sensorchan',
                        # 'style': 'pred_cfg',
                        # 'hue': 'pred_tta_fliprot',
                        # 'hue': 'pred_tta_time',
                        # 'size': 'pred_tta_fliprot',
                        # 'style': 'hardware',
                    }
                elif type == 'trk+pxl':
                    plotkw = {
                        'x': resource_type,
                        'y': metric_lut[type],
                        'hue': 'sensorchan',
                        # 'hue': 'pred_tta_time',
                        # 'size': 'pred_tta_fliprot',
                        # 'style': 'hardware',
                    }
                else:
                    raise KeyError(type)
                fig = kwplot.figure(fnum=fnum, pnum=pnum_())
                ax = fig.gca()
                # fig.get_size_inches()
                fig.set_size_inches(17.85,  6.82)
                data = group
                plotkw['s'] = markersize
                ax = humanized_scatterplot(human_mapping, data=data, ax=ax, **plotkw)
                ax.set_title(f'{human_resource_type} vs {human_metric_type} - {type} - {dataset_code=}')

    expt_group = dict(list(merged_df.groupby(['dataset_code', 'type'])))
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

    gsd_type = ('Cropped-Drop3-TA1-2022-03-10', 'act+pxl')
    dataset_code, type = gsd_type
    group = expt_group[gsd_type]

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
    scored_obs = analysis.abalate_one(param)
    ab_rows = []
    pts1 = []
    pts2 = []
    for obs in scored_obs:
        ab_row = obs.melt(['trk_use_viterbi']).pivot(['variable'], ['trk_use_viterbi'],  'value').reset_index(drop=True)
        if (~ab_row.isnull()).values.sum() > 1:

            # hack
            obspt = obs.copy()
            obspt[param] = obs[param].astype(bool).astype(int)
            pts1.append(obspt.values[0])
            pts2.append(obspt.values[1])
            ab_rows.append(ab_row)
    # ab_df = pd.concat(ab_rows).reset_index(drop=True)
    # obs[param]
    kwplot.draw_line_segments(pts1, pts2)
    # ticks = ax.set_xticks([0, 1], ['0', 'v6,v1'])
    ax.set_ylabel(metric)
    ax.set_xlabel(param)
    ax.set_title('Viterbi A/B tests')

    plt = kwplot.autoplt()
    ax = plt.gca()

    return

    # ab_df
    # .reset_index()
    # ab_melt_df = ab_df.melt(id_vars=['index'], value_vars=ab_row.columns)
    # sns = kwplot.autosns()
    # sns.lineplot(data=ab_melt_df, x=param, y='value')

    import seaborn as sns
    kwplot.figure(fnum=1000)
    sns.violinplot(data=merged_df, x='temporal_dropout', y=pixel_metric)

    kwplot.figure(fnum=1001)
    sns.violinplot(data=merged_df, x='chip_size', y=pixel_metric)

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

    group[fit_param_keys]
    group[fit_param_keys]


def humanized_scatterplot(human_mapping, data, ax, **plotkw):
    import seaborn as sns
    ax = sns.scatterplot(data=data, ax=ax, **plotkw)
    xkey = plotkw['x']
    ykey = plotkw['y']
    ax.set_xlabel(human_mapping.get(xkey, xkey))
    ax.set_ylabel(human_mapping.get(ykey, ykey))
    legend = ax.get_legend()
    if legend is not None:
        leg_title = legend.get_title()
        old_text = leg_title.get_text()
        new_text = human_mapping.get(old_text, old_text)
        leg_title.set_text(new_text)
        for leg_lbl in legend.texts:
            old_text = leg_lbl.get_text()
            new_text = human_mapping.get(old_text, old_text)
            leg_lbl.set_text(new_text)
    return ax


def num_files_summary(df):
    expt_group = dict(list(df.groupby('dataset_code')))
    filt_summaries = []
    for gsd, group in sorted(expt_group.items())[::-1]:
        print('gsd = {!r}'.format(gsd))
        print('Column Unique Value Frequencies')
        col_stats_df2 = unique_col_stats(group)
        print(col_stats_df2.to_string())
        row = {}
        type_evals = ub.dict_hist(group['type'])
        row['gsd'] = gsd
        row['num_experiments'] = len(group['expt'].unique())
        row['num_models'] = len(group['model'].unique())
        row['num_pxl_evals'] = type_evals.get('pxl', 0)
        row['num_bas_evals'] = type_evals.get('trk', 0)
        row['num_sc_evals'] = type_evals.get('act', 0)
        filt_summaries.append(row)
    _summary_df = pd.DataFrame(filt_summaries)
    total_row = _summary_df.sum().to_dict()
    total_row['gsd'] = '*'
    summary_df = pd.DataFrame(filt_summaries + [total_row])
    print('Number of Models & Evaluations')
    print(summary_df.to_string(index=False))


def unique_col_stats(df):
    col_stats = ub.ddict(dict)
    import kwarray
    import numpy as np
    for key in df.columns:
        col_freq = np.array(list(ub.dict_hist(df[key]).values()))
        stats = kwarray.stats_dict(col_freq, median=True)
        stats['num_unique'] = stats.pop('shape')[0]
        col_stats[key] = stats
        # ['num_unique'] = len(unique_cols)
        # col_stats[key]['max'] = stats['max']
        # col_stats[key]['max'] = stats['max']
    col_stats_df = pd.DataFrame(col_stats)
    # Hack
    col_stats_df = col_stats_df.drop(['gsd', 'dvc', 'raw'], axis=1)
    col_stats_df = col_stats_df.astype(int)
    return col_stats_df


def load_extended_data(df, dvc_dpath):
    from watch.tasks.fusion import aggregate_results as agr
    filt_rows = df.to_dict('records')

    big_rows = []
    errors = []
    for row in ub.ProgIter(filt_rows, desc='load'):
        big_row = row.copy()
        fpath = row['raw']
        try:
            if row['type'] == 'pxl':
                pxl_info = agr.load_pxl_eval(fpath, dvc_dpath)
                big_row['info'] = pxl_info
            elif row['type'] == 'act':
                sc_info = agr.load_sc_eval(fpath, dvc_dpath)
                big_row['info'] = sc_info
            elif row['type'] == 'trk':
                bas_info = agr.load_bas_eval(fpath, dvc_dpath)
                big_row['info'] = bas_info
            else:
                raise KeyError(row['type'])
            big_rows.append(big_row)
        except Exception as ex:
            errors.append((ex, row))
    print(f'{len(errors)=}')
    return big_rows


def clean_loaded_data(big_rows):
    from watch.tasks.fusion import aggregate_results as agr
    try:
        from kwcoco._experimental.sensorchan import concise_sensor_chan, sensorchan_parts
    except Exception:
        concise_sensor_chan = ub.identity

    def _is_teamfeat(x):
        if isinstance(x, float) and math.isnan(x):
            return False
        return any([a in x for a in ['depth', 'invariant', 'invariants', 'matseg', 'land']])

    _actcfg_to_track_config = ub.ddict(list)
    _trkcfg_to_track_config = ub.ddict(list)
    _prdcfg_to_pred_config = ub.ddict(list)
    simple_rows = []
    missing_models = []
    blocklist = {
        'S2:|R|G',
        'S2:|G|R|,invariants:16)',
        'S2:(RGB|land:8,R|G,R|G|land:8)',
    }

    passlist = {
        'BGR',
        'RGB|near-ir1|near-ir2|red-edge|yellow',
        'BGR|near-ir1',
        'BGRNSH|land:8|matseg:4|mat_up5:64',
        'BGRNSH',
        'BGR|near-ir1|depth',
        'RGB',
        'RGB|near-ir1',
        'RGB|land:8',
        'RGB|near-ir1|near-ir2|depth',
        'RGB|near_ir1|near_ir2|depth',
        'land:8',
        'invariants:16',
        'matseg:4',
        'matseg:4|mat_up5:64',
    }
    chan_blocklist = {
        'R|G',
        'G|R',
        'G|R|N|S|H',
        'R|G|land:8',
        'RGB|near-ir1|depth',
        'G|R|N|S|H|land:8|matseg:4|mat_up5:64',
    }

    for big_row in ub.ProgIter(big_rows, desc='big rows'):
        # fpath = big_row['raw']
        row = ub.dict_diff(big_row, {'info'})
        info = big_row['info']

        param_type = info['param_types']

        fit_params = param_type['fit']
        pred_params = param_type['pred']
        model_fpath = pred_params['pred_model_fpath']

        fit_params['channels'] = agr.shrink_channels(fit_params['channels'])

        # if 'invariants' in fit_params['channels']:
        #     raise Exception

        # Dont trust what the model info says about channels, look
        # at the model stats to be sure.
        if model_fpath and model_fpath.exists():
            stats = resolve_model_info(model_fpath)
            real_chan_parts = ub.oset()
            senschan_parts = []
            real_sensors = []
            for input_row in stats['model_stats']['known_inputs']:
                real_chan = agr.shrink_channels(input_row['channel'])
                if real_chan not in chan_blocklist:
                    if real_chan not in passlist:
                        print(real_chan)
                    real_chan_parts.add(real_chan)
                    real_sensors.append(input_row['sensor'])
                    senschan_parts.append('{}:{}'.format(input_row['sensor'], real_chan))
            sensorchan = ','.join(sorted(set(senschan_parts)))
            sensorchan = concise_sensor_chan(sensorchan)
            request_chan_parts = set(fit_params['channels'].split(','))
            if not request_chan_parts.issubset(real_chan_parts):
                print(f'{real_chan_parts=}')
                print(f'{request_chan_parts=}')
                print(row['expt'])
                fit_params['bad_channels'] = True
            else:
                fit_params['bad_channels'] = False
        else:
            missing_models.append(model_fpath)

            if 'Cropped' in big_row['test_dset']:
                # Hack
                sensors = ['WV', 'S2']
            elif 'Cropped' in big_row['test_dset']:
                sensors = ['S2', 'L8']
            else:
                sensors = ['*']

            import kwcoco
            channels = kwcoco.ChannelSpec.coerce(fit_params['channels'])
            senschan_parts = []
            for sensor in sensors:
                for chan in channels.streams():
                    senschan_parts.append(f'{sensor}:{chan.spec}')

            sensorchan = ','.join(sorted(senschan_parts))
            sensorchan = concise_sensor_chan(sensorchan)
            request_chan_parts = set(fit_params['channels'].split(','))
            if not request_chan_parts.issubset(real_chan_parts):
                print(f'{real_chan_parts=}')
                print(f'{request_chan_parts=}')
                print(row['expt'])
                fit_params['bad_channels'] = True
            else:
                fit_params['bad_channels'] = False

        # MANUAL HACK:
        if 1:
            sensorchan = ','.join([p for p in sensorchan_parts(sensorchan) if p not in blocklist])

        fit_params['sensorchan'] = sensorchan
        row['has_teamfeat'] = _is_teamfeat(sensorchan)

        selected_fit_params = ub.dict_isect(fit_params, fit_param_keys)

        param_type['fit']
        act_cfg = row['act_cfg']
        if not is_null(act_cfg):
            track_cfg = param_type.get('track', None)
            row.update(track_cfg)
            _actcfg_to_track_config[act_cfg].append(track_cfg)

        trk_cfg = row['trk_cfg']
        if not is_null(trk_cfg):
            track_cfg = param_type.get('track', None)
            row.update(track_cfg)
            _trkcfg_to_track_config[trk_cfg].append(track_cfg)

        pred_cfg = row['pred_cfg']
        if not is_null(trk_cfg):
            pred_config = param_type.get('pred', None)
            pred_config = ub.dict_isect(pred_config, pred_param_keys)
            if pred_config.get('pred_tta_time', None) is None:
                pred_config['pred_tta_time'] = 0
            if pred_config.get('pred_tta_fliprot', None) is None:
                pred_config['pred_tta_fliprot'] = 0
            row.update(pred_config)
            _prdcfg_to_pred_config[pred_cfg].append(pred_config)

        resource = param_type.get('resource', {})
        row['model_fpath'] = model_fpath
        row.update(info['metrics'])
        row.update(resource)
        row.update(selected_fit_params)
        simple_rows.append(row)

    simple_df = pd.DataFrame(simple_rows)
    print(f'{len(simple_df)=}')
    # simple_df['sensorchan'].unique()
    # simple_df[simple_df['sensorchan'].isnull()]
    # simple_df['bad_channels']

    actcfg_to_track_config = {}
    for actcfg, track_cfgs in _actcfg_to_track_config.items():
        unique_configs = list(ub.unique(track_cfgs, key=ub.hash_data))
        assert len(unique_configs) == 1
        actcfg_to_track_config[actcfg] = unique_configs[0]

    trkcfg_to_track_config = {}
    for trkcfg, track_cfgs in _trkcfg_to_track_config.items():
        unique_configs = list(ub.unique(track_cfgs, key=ub.hash_data))
        assert len(unique_configs) == 1
        trkcfg_to_track_config[trkcfg] = unique_configs[0]

    prdcfg_to_pred_config = {}
    for predcfg, track_cfgs in _prdcfg_to_pred_config.items():
        unique_configs = list(ub.unique(track_cfgs, key=ub.hash_data))
        if len(unique_configs) == 1:
            prdcfg_to_pred_config[predcfg] = unique_configs[0]
        else:
            print(f'{unique_configs=}')
            print('predcfg = {}'.format(ub.repr2(predcfg, nl=1)))

    if True:
        # Get activity config labels
        actcfg_to_label = {}
        varied_act = ub.varied_values(actcfg_to_track_config.values(), 1, default=None)
        varied_act_keys = sorted(varied_act.keys())
        for k, v in actcfg_to_track_config.items():
            c = ub.dict_isect(v, varied_act_keys)
            label = ub.repr2(c, compact=1)
            actcfg_to_label[k] = label

        # Get activity config labels
        predcfg_to_label = {}
        varied_act = ub.varied_values(prdcfg_to_pred_config.values(), 1, default=None)
        varied_act_keys = sorted(varied_act.keys())
        for k, v in prdcfg_to_pred_config.items():
            c = ub.dict_isect(v, varied_act_keys)
            label = ub.repr2(c, compact=1)
            predcfg_to_label[k] = label

    bad_expts = simple_df[simple_df['bad_channels']]['expt']
    print('bad_expts =\n{}'.format(ub.repr2(bad_expts, nl=1)))

    ub.dict_hist(simple_df['channels'])
    merged_rows = []
    for pred_key, group in simple_df.groupby(['model', 'pred_cfg']):
        # Can propogate pixel metrics to child groups
        type_to_subgroup = dict(list(group.groupby('type')))
        pxl_group = type_to_subgroup.pop('pxl', None)
        if pxl_group is not None:
            if len(pxl_group) > 1:
                print(f'Warning more than one pixel group for {pred_key}')
            pxl_row = pxl_group.iloc[0]

            if len(type_to_subgroup) == 0:
                pxl_row = pxl_row.to_dict()
                if not math.isnan(pxl_row.get('coi_mAP', np.nan)):
                    srow = pxl_row.copy()
                    srow['type'] = 'act+pxl'
                    merged_rows.append(srow)

                if not math.isnan(pxl_row.get('salient_AP', np.nan)):
                    srow = pxl_row.copy()
                    srow['type'] = 'trk+pxl'
                    merged_rows.append(srow)

            for type, subgroup in type_to_subgroup.items():
                for srow in subgroup.to_dict('records'):
                    srow['type'] = srow['type'] + '+pxl'
                    for k1, v1 in pxl_row.items():
                        v2 = srow.get(k1, None)
                        if v2 is None or (isinstance(v2, float) and math.isnan(v2)):
                            srow[k1] = v1
                    merged_rows.append(srow)

    merged_df = pd.DataFrame(merged_rows)
    print(f'{len(merged_df)=}')

    total_carbon_cost = simple_df[simple_df['type'] == 'pxl']['co2_kg'].sum()
    # total_carbon_cost = merged_df['co2_kg'].sum()
    print(f'{total_carbon_cost=}')
    merged_df['gpu_name'] = merged_df['gpu_name'].fillna('?')
    merged_df['cpu_name'] = merged_df['cpu_name'].fillna('?')
    cpu_names = merged_df['cpu_name'].apply(lambda x: x.replace('Intel(R) Core(TM) ', ''))
    gpu_names = merged_df['gpu_name']
    merged_df['hardware'] = ['{} {}'.format(c, g) for c, g in zip(cpu_names, gpu_names)]

    other = {
        'actcfg_to_label': actcfg_to_label,
        'predcfg_to_label': predcfg_to_label,
    }
    return merged_df, other
