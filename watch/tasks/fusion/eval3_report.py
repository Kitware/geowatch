"""

python -m watch.tasks.fusion.dvc_sync_manager "push pull evals"

"""
import ubelt as ub
import math
import numpy as np
import pandas as pd
import os
import glob
import functools  # NOQA

from watch.tasks.fusion import dvc_sync_manager


def eval3_report():
    import kwplot
    kwplot.autosns()
    import watch
    try:
        dvc_dpath = watch.find_smart_dvc_dpath(hardware='hdd')
    except Exception:
        dvc_dpath = watch.find_smart_dvc_dpath()

    dvc_manager = dvc_sync_manager.DVCSyncManager(dvc_dpath)

    gsd10_dpath = dvc_dpath / 'models/fusion/eval3_candidates'
    gsd1_dpath = dvc_dpath / 'models/fusion/eval3_sc_candidates'

    summary_stats = []
    evaluations = []

    gsd_dpaths = {
        10: gsd10_dpath,
        1: gsd1_dpath,
    }
    for gsd, dpath in gsd_dpaths.items():
        experiments = list((dpath / 'packages').glob('*'))
        models = list((dpath / 'packages').glob('*/*'))

        eval_globstrs = {
            'pxl': dpath / 'eval/*/*/*/*/eval/curves/measures2.*',
            'bas': dpath / dpath / 'eval/*/*/*/*/eval/tracking/*/iarpa_eval/scores/merged/summary2.*',
            'sc': dpath / 'eval/*/*/*/*/eval/actclf/*/iarpa_*/scores/merged/summary3.*',
        }

        pxl_evals = dvc_globbed_info(eval_globstrs['pxl'], type='pxl', gsd=gsd)
        bas_evals = dvc_globbed_info(eval_globstrs['bas'], type='bas', gsd=gsd)
        sc_evals = dvc_globbed_info(eval_globstrs['sc'], type='sc', gsd=gsd)
        evaluations += pxl_evals
        evaluations += bas_evals
        evaluations += sc_evals

        row = {
            'gsd': gsd,
            'num_experiments': len(experiments),
            'num_models': len(models),
            'num_pxl_evals': len(pxl_evals),
            'num_bas_evals': len(bas_evals),
            'num_sc_evals': len(sc_evals),
        }
        summary_stats.append(row)

    if 1:
        # Check if dvc is synced
        missing_dvc = []
        missing_real = []
        for row in evaluations:
            if row['dvc_fpath'] is None:
                missing_dvc.append(row['fpath'])
            if row['fpath'] is None:
                missing_real.append(row['dvc_fpath'])
        print(f'{len(missing_real)=}')
        print(f'{len(missing_dvc)=}')

    if 0:
        # Ensure dvc is synced
        from watch.utils import simple_dvc
        if missing_real:
            dvc = simple_dvc.SimpleDVC.coerce(missing_real[0])
            dvc.pull(missing_real, remote='aws')

        if missing_dvc:
            dvc = simple_dvc.SimpleDVC.coerce(missing_dvc[0])
            dvc.add(missing_dvc)
            import platform
            dvc.git_commitpush('Adding evals from {}'.format(platform.node()))
            dvc.push(missing_dvc, remote='aws')

    _summary_df = pd.DataFrame(summary_stats)
    total_row = _summary_df.sum().to_dict()
    total_row['gsd'] = '*'
    summary_df = pd.DataFrame(summary_stats + [total_row])
    print('Number of Models & Evaluations')
    print(summary_df.to_string(index=False))

    # Filter down to just the evals we have on this machine.
    evaluations = [row for row in evaluations if row['fpath'] is not None]
    for row in evaluations:
        eval_fpath = ub.Path(row['fpath'])

        if row['type'] == 'sc':
            eval_dpath = eval_fpath.parent.parent.parent.parent.parent.parent
            row['act_cfgstr'] = eval_fpath.parent.parent.parent.parent.name

        if row['type'] == 'bas':
            eval_dpath = eval_fpath.parent.parent.parent.parent.parent.parent
            row['trk_cfgstr'] = eval_fpath.parent.parent.parent.parent.name

        if row['type'] == 'pxl':
            eval_dpath = eval_fpath.parent.parent

        assert eval_dpath.name == 'eval'

        row['pred_cfgstr'] = eval_dpath.parent.name
        row['test_dset'] = eval_dpath.parent.parent.name
        row['model_name'] = eval_dpath.parent.parent.parent.name.replace('pred_', '')
        row['expt_name'] = eval_dpath.parent.parent.parent.parent.name

    raw_df = pd.DataFrame(evaluations)
    col_stats_df = unique_col_stats(raw_df)
    print('Column Unique Value Frequencies')
    print(col_stats_df.to_string())

    raw_df['trk_cfgstr'].unique()
    raw_df['act_cfgstr'].unique()
    raw_df['pred_cfgstr'].unique()
    raw_df['expt_name'].unique()
    raw_df['model_name'].unique()
    # print(raw_df['test_dset'].unique())

    test_dset_blocklist = {
        'Drop2-Aligned-TA1-2022-02-15_combo_DILM_nowv_vali.kwcoco',
    }
    flags = raw_df['test_dset'].apply(lambda x: x in test_dset_blocklist)
    raw_df = raw_df[~flags]
    print('Remove {} drop2 evals'.format(flags.sum()))

    preference = {
        'Cropped-Drop3-TA1-2022-03-10_combo_DLM_s2_wv_vali.kwcoco': 0,
        'Cropped-Drop3-TA1-2022-03-10_combo_DL_s2_wv_vali.kwcoco': 1,
        'Cropped-Drop3-TA1-2022-03-10_data_wv_vali.kwcoco': 2,

        'Aligned-Drop3-TA1-2022-03-10_combo_LM_nowv_vali.kwcoco': 0,
        'Aligned-Drop3-TA1-2022-03-10_combo_LM_vali.kwcoco': 1,
    }

    # Filter out rows where models have predictions on "better" datasets
    keep_locs = []
    for expt_name, group in raw_df.groupby('expt_name'):
        # print('expt_name = {!r}'.format(expt_name))
        # dset_to_subgroup = dict(list(group.groupby('test_dset')))
        for model_name, subgroup in list(group.groupby('model_name')):
            prefs = subgroup['test_dset'].apply(lambda x: preference.get(x, 0))
            keep_flags = prefs == prefs.min()
            keep_locs.extend(subgroup[keep_flags].index)

    print(f'Keep {len(keep_locs)} / {len(raw_df)} drop3 evals')
    df = filt_df = raw_df.loc[keep_locs]
    print('Column Unique Value Frequencies')
    # print(col_stats_df2.to_string())
    num_files_summary(filt_df)

    # Load detailed data
    eval_types_to_locs = ub.ddict(list)
    gsd_groups = dict(list(filt_df.groupby('gsd')))
    for gsd, gsd_group in sorted(gsd_groups.items())[::-1]:
        for expt_name, expt_group in gsd_group.groupby(['expt_name']):
            for model_name, model_groups in expt_group.groupby(['model_name']):
                for pred_cfg, pred_group in model_groups.groupby('pred_cfgstr'):
                    eval_types = tuple(sorted(pred_group['type'].unique()))
                    eval_types_to_locs[eval_types].extend(pred_group.index)

    print('Comparable locs')
    print(ub.repr2(ub.map_vals(len, eval_types_to_locs)))
    comparable_locs = eval_types_to_locs[('bas', 'pxl')] + eval_types_to_locs[('pxl', 'sc')]
    df = comp_df = filt_df.loc[comparable_locs]
    num_files_summary(comp_df)

    load_extended_data(df)

    # gsd_groups = dict(list(comp_df.groupby('gsd')))
    # bas_df = gsd_groups[10]
    # sc_df = gsd_groups[1]


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


def load_extended_data(df, dvc_dpath):
    from watch.tasks.fusion import aggregate_results as agr
    filt_rows = df.to_dict('records')

    big_rows = []
    errors = []
    for row in ub.ProgIter(filt_rows, desc='load'):
        big_row = row.copy()
        fpath = row['fpath']
        try:
            if row['type'] == 'pxl':
                pxl_info = agr.load_pxl_eval(fpath, dvc_dpath)
                big_row['info'] = pxl_info
            elif row['type'] == 'sc':
                sc_info = agr.load_sc_eval(fpath, dvc_dpath)
                big_row['info'] = sc_info
            elif row['type'] == 'bas':
                bas_info = agr.load_bas_eval(fpath, dvc_dpath)
                big_row['info'] = bas_info
            else:
                raise KeyError(row['type'])
            big_rows.append(big_row)
        except Exception as ex:
            errors.append((ex, row))
    print(f'{len(errors)=}')

    try:
        from kwcoco._experimental.sensorchan import concise_sensor_chan
    except Exception:
        concise_sensor_chan = ub.identity

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

    _actcfg_to_track_config = ub.ddict(list)
    _trkcfg_to_track_config = ub.ddict(list)
    _prdcfg_to_pred_config = ub.ddict(list)
    simple_rows = []
    missing_models = []
    for big_row in ub.ProgIter(big_rows, desc='big rows'):
        fpath = big_row['fpath']
        row = ub.dict_diff(big_row, {'info'})
        info = big_row['info']

        param_type = info['param_types']

        fit_params = param_type['fit']
        pred_params = param_type['pred']
        model_fpath = pred_params['pred_model_fpath']

        fit_params['channels'] = agr.shrink_channels(fit_params['channels'])

        # Dont trust what the model info says about channels, look
        # at the model stats to be sure.
        if model_fpath and model_fpath.exists():
            stats = resolve_model_info(model_fpath)
            real_chan_parts = ub.oset()
            senschan_parts = []
            real_sensors = []
            for input_row in stats['model_stats']['known_inputs']:
                real_chan = agr.shrink_channels(input_row['channel'])
                real_chan_parts.add(real_chan)
                real_sensors.append(input_row['sensor'])
                senschan_parts.append('{}:{}'.format(input_row['sensor'], real_chan))

            sensorchan = ','.join(sorted(senschan_parts))
            sensorchan = concise_sensor_chan(sensorchan)
            fit_params['sensorchan'] = sensorchan
            request_chan_parts = set(fit_params['channels'].split(','))
            if not request_chan_parts.issubset(real_chan_parts):
                print(f'{real_chan_parts=}')
                print(f'{request_chan_parts=}')
                print(row['expt_name'])
                fit_params['bad_channels'] = True
            else:
                fit_params['bad_channels'] = False
        else:
            missing_models.append(model_fpath)
            fit_params['bad_channels'] = False

        selected_fit_params = ub.dict_isect(fit_params, fit_param_keys)

        param_type['fit']
        act_cfgstr = row['act_cfgstr']
        if not is_null(act_cfgstr):
            track_cfg = param_type.get('track', None)
            row.update(track_cfg)
            _actcfg_to_track_config[act_cfgstr].append(track_cfg)

        trk_cfgstr = row['trk_cfgstr']
        if not is_null(trk_cfgstr):
            track_cfg = param_type.get('track', None)
            row.update(track_cfg)
            _trkcfg_to_track_config[trk_cfgstr].append(track_cfg)

        pred_cfgstr = row['pred_cfgstr']
        if not is_null(trk_cfgstr):
            pred_config = param_type.get('pred', None)
            pred_config = ub.dict_isect(pred_config, pred_param_keys)
            if pred_config.get('pred_tta_time', None) is None:
                pred_config['pred_tta_time'] = 0
            if pred_config.get('pred_tta_fliprot', None) is None:
                pred_config['pred_tta_fliprot'] = 0
            row.update(pred_config)
            _prdcfg_to_pred_config[pred_cfgstr].append(pred_config)

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

    bad_expts = simple_df[simple_df['bad_channels']]['expt_name']
    print('bad_expts =\n{}'.format(ub.repr2(bad_expts, nl=1)))

    ub.dict_hist(simple_df['channels'])
    merged_rows = []
    for pred_key, group in simple_df.groupby(['model_name', 'pred_cfgstr']):
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
                    srow['type'] = 'sc+pxl'
                    merged_rows.append(srow)

                if not math.isnan(pxl_row.get('salient_AP', np.nan)):
                    srow = pxl_row.copy()
                    srow['type'] = 'bas+pxl'
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

    gsd_groups = dict(list(merged_df.groupby(['gsd', 'type'])))
    import pprint
    for gsd_type, group in gsd_groups.items():
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
        'act_cfgstr': 'SC Tracking Config',
        'trk_use_viterbi': 'Viterbi Enabled',
        'trk_thresh': 'SC Tracking Threshold',
        'co2_kg': 'CO2 Emissions (kg)',
        'total_hours': 'Time (hours)',
        'sensorchan': 'Sensor/Channel',
    }
    human_mapping.update(actcfg_to_label)
    human_mapping.update(predcfg_to_label)

    iarpa_metric_lut = {
        'sc+pxl': 'mean_f1',
        'bas+pxl': 'BAS_F1',
    }
    pixel_metric_lut = {
        'sc+pxl': 'coi_mAP',
        'bas+pxl': 'salient_AP',
    }

    # ['trk_thresh',
    #  'trk_morph_kernel',
    #  'trk_agg_fn',
    #  'trk_thresh_hysteresis',
    #  'trk_moving_window_size']

    merged_df['has_teamfeat'] = merged_df['sensorchan'].apply(lambda x: (not (isinstance(x, float) and math.isnan(x))) and (('depth' in x) or ('invariant' in x) or ('matseg' in x) or ('land' in x)))

    import kwplot
    gsd_groups = dict(list(merged_df.groupby(['gsd', 'type'])))
    fnum = 0
    for gsd_type, group in gsd_groups.items():
        gsd, type = gsd_type
        if type == 'sc+pxl':
            plotkw = {
                'x': pixel_metric_lut[type],
                'y': iarpa_metric_lut[type],
                'hue': 'channels',
                'style': 'has_teamfeat',
                # 'hue': 'trk_use_viterbi',
                # 'style': 'trk_thresh',
                # 'size': 'trk_thresh',
                # 'hue': 'pred_cfgstr',
                # 'hue': 'expt_name',
            }
        elif type == 'bas+pxl':
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
                # 'hue': 'trk_cfgstr',
                # 'hue': 'pred_cfgstr',
                # 'hue': 'expt_name',
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
        ax = humanized_scatterplot(human_mapping, data=group, ax=ax, **plotkw)
        ax.set_title(f'Pixelwise Vs IARPA metrics - {type} - {gsd=}\n{corr_lbl}')

    fnum = 10
    import kwplot
    gsd_groups = dict(list(merged_df.groupby(['gsd', 'type'])))
    for gsd_type, group in gsd_groups.items():

        group = group[~group['sensorchan'].isnull()]
        # group['has_teamfeat'] = group['sensorchan'].apply(lambda x: (('depth' in x) or ('invariant' in x) or ('matseg' in x) or ('land' in x)))

        gsd, type = gsd_type
        if type == 'sc+pxl':
            plotkw = {
                'x': pixel_metric_lut[type],
                'y': 'coi_mAUC',
                'hue': 'sensorchan',
                'style': 'has_teamfeat',
                # 'hue': 'trk_use_viterbi',
                # 'style': 'trk_thresh',
                # 'size': 'trk_thresh',
                # 'hue': 'pred_cfgstr',
                # 'hue': 'expt_name',
            }
        elif type == 'bas+pxl':
            plotkw = {
                'x': pixel_metric_lut[type],
                'y': 'salient_AUC',
                'hue': 'sensorchan',
                'style': 'has_teamfeat',
                # 'hue': 'trk_cfgstr',
                # 'hue': 'pred_cfgstr',
                # 'hue': 'expt_name',
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
        ax = humanized_scatterplot(human_mapping, data=group, ax=ax, **plotkw)
        ax.set_title(f'Pixelwise metrics - {type} - {gsd=}\n{corr_lbl}')
        fig.set_size_inches(16.85,  8.82)

    gsd_groups = dict(list(merged_df.groupby(['gsd', 'type'])))
    if 1:
        gsd_groups.pop((10, 'sc+pxl'), None)
        gsd_groups.pop((1, 'sc+pxl'), None)
        gsd_groups.pop((1, 'bas+pxl'), None)
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
            pnum_ = kwplot.PlotNums(nCols=len(gsd_groups))
            for gsd_type, group in gsd_groups.items():

                group['pred_tta_time'] = group['pred_tta_time'].astype(str)
                group['pred_tta_fliprot'] = group['pred_tta_fliprot'].astype(str)

                group.loc[group['pred_tta_time'] == 'nan', 'pred_tta_time'] = '0.0'
                group.loc[group['pred_tta_fliprot'] == 'nan', 'pred_tta_fliprot'] = '0.0'
                gsd, type = gsd_type
                if type == 'sc+pxl':
                    plotkw = {
                        'x': resource_type,
                        'y': metric_lut[type],
                        'hue': 'sensorchan',
                        # 'style': 'pred_cfgstr',
                        # 'hue': 'pred_tta_fliprot',
                        # 'hue': 'pred_tta_time',
                        # 'size': 'pred_tta_fliprot',
                        # 'style': 'hardware',
                    }
                elif type == 'bas+pxl':
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
                ax = humanized_scatterplot(human_mapping, data=data, ax=ax, **plotkw)
                ax.set_title(f'{human_resource_type} vs {human_metric_type} - {type} - {gsd=}')

    gsd_groups = dict(list(merged_df.groupby(['gsd', 'type'])))
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

    gsd_type = (1, 'sc+pxl')
    gsd, type = gsd_type
    group = gsd_groups[gsd_type]

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
    try:
        analysis.run()
    except TypeError:
        raise


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
    ab_df = pd.concat(ab_rows).reset_index(drop=True)
    obs[param]
    kwplot.draw_line_segments(pts1, pts2)
    ticks = ax.set_xticks([0, 1], ['0', 'v6,v1'])
    ax.set_ylabel(metric)
    ax.set_xlabel(param)
    ax.set_title('Viterbi A/B tests')

    plt = kwplot.autoplt()
    ax = plt.gca()


    # ab_df
    # .reset_index()
    ab_melt_df = ab_df.melt(id_vars=['index'], value_vars=ab_row.columns)
    sns.lineplot(data=ab_melt_df, x=param, y='value')

    scored_obs['']

    import seaborn as sns
    kwplot.figure(fnum=1)
    sns.violinplot(data=merged_df, x='temporal_dropout', y=pixel_metric)

    kwplot.figure(fnum=2)
    sns.violinplot(data=merged_df, x='chip_size', y=pixel_metric)

    kwplot.figure(fnum=2, doclf=True)
    sns.violinplot(data=merged_df, x='time_steps', y=pixel_metric)

    kwplot.figure(fnum=2, doclf=True)
    sns.violinplot(data=merged_df, x='trk_use_viterbi', y=pixel_metric)

    kwplot.figure(fnum=2, doclf=True)
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
    gsd_groups = dict(list(df.groupby('gsd')))
    filt_summaries = []
    for gsd, group in sorted(gsd_groups.items())[::-1]:
        print('gsd = {!r}'.format(gsd))
        print('Column Unique Value Frequencies')
        col_stats_df2 = unique_col_stats(group)
        print(col_stats_df2.to_string())

        row = {}
        type_evals = ub.dict_hist(group['type'])
        row['gsd'] = gsd
        row['num_experiments'] = len(group['expt_name'].unique())
        row['num_models'] = len(group['model_name'].unique())
        row['num_pxl_evals'] = type_evals.get('pxl', 0)
        row['num_bas_evals'] = type_evals.get('bas', 0)
        row['num_sc_evals'] = type_evals.get('sc', 0)
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
        stats = kwarray.stats_dict(col_freq, quantile=0, median=True)
        stats['num_unique'] = stats.pop('shape')[0]
        col_stats[key] = stats
        # ['num_unique'] = len(unique_cols)
        # col_stats[key]['max'] = stats['max']
        # col_stats[key]['max'] = stats['max']
    col_stats_df = pd.DataFrame(col_stats)
    # Hack
    col_stats_df = col_stats_df.drop(['gsd', 'dvc_fpath', 'fpath'], axis=1)
    col_stats_df = col_stats_df.astype(int)
    return col_stats_df


def dvc_globbed_info(pat, **extra):
    eval_fpaths = list(glob.glob(os.fspath(pat)))
    rows = []
    for k, group in ub.group_items(eval_fpaths, lambda x: (ub.Path(x).parent, ub.Path(x).name.split('.')[0])).items():
        dvc_fpath = None
        fpath = None
        for g in group:
            if g.endswith('.dvc'):
                dvc_fpath = g
            else:
                fpath = g

        row = {
            **extra,
            'fpath': fpath,
            'dvc_fpath': dvc_fpath,
        }
        rows.append(row)
    return rows
