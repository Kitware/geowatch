"""

python -m watch.tasks.fusion.dvc_sync_manager "list"
python -m watch.tasks.fusion.dvc_sync_manager "push pull evals"
python -m watch.tasks.fusion.dvc_sync_manager "pull evals"
python -m watch.tasks.fusion.dvc_sync_manager "pull packages"
python -m watch.tasks.fusion.dvc_sync_manager "push evals"

"""
import ubelt as ub
import math
import numpy as np
import pandas as pd
import functools  # NOQA
# APPLY Monkey Patches
from watch.tasks.fusion import monkey  # NOQA


# TODO: move to heuristics
fit_param_keys = [
    'sensorchan',
    # 'channels',
    'time_steps',
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

DSET_CODE_TO_GSD = {
    'Aligned-Drop3-L1': 10.0,
    'Aligned-Drop3-TA1-2022-03-10': 10.0,
    'Cropped-Drop3-TA1-2022-03-10': 1.0,
}


def eval3_report():
    """
    MAIN FUNCTION

    from watch.tasks.fusion.eval3_report import *  # NOQA
    """
    import kwplot
    kwplot.autosns()
    import watch
    try:
        dvc_dpath = watch.find_smart_dvc_dpath(hardware='hdd')
    except Exception:
        dvc_dpath = watch.find_smart_dvc_dpath()
    reporter = EvaluationReporter(dvc_dpath)
    reporter.load()
    reporter.summarize()
    plot_merged(reporter)
    self = reporter

    if 0:
        self = reporter
        merged_df = self.orig_merged_df.copy()
        merged_df[merged_df.expt.str.contains('invar')]['mean_f1']
        merged_df[merged_df.in_production]['mean_f1']

        selected = merged_df[merged_df.in_production].sort_values('mean_f1')
        selected = selected[['siteprep_f1', 'active_f1', 'mean_f1', 'model']]
        selected['coi_mean_f1'] = selected[['siteprep_f1', 'active_f1']].mean(axis=1)
        selected = selected.sort_values('coi_mean_f1')
        print(selected)


class EvaluationReporter:
    """
    Manages handing the data off to experiment plotting functions.
    """

    def __init__(self, dvc_dpath):
        from watch.tasks.fusion import dvc_sync_manager
        self.dvc_dpath = dvc_dpath
        self.dvc_manager = dvc_sync_manager.DVCSyncManager.coerce(dvc_dpath)
        # dvc_sync_manager.main(command='pull evals')
        # dvc_sync_manager.main(command='pull packages')

        self.raw_df = None
        self.filt_df = None
        self.comp_df = None

        self.dpath = ub.Path.appdir('watch/report').ensuredir()

    def summarize(self, table=None):
        if table is None:
            table = self.dvc_manager.evaluation_table()
        self.dvc_manager.summarize()
        if 0:
            loaded_table = load_extended_data(table, self.dvc_dpath)
            loaded_table = pd.DataFrame(loaded_table)
            # dataset_summary_tables(dpath)
            initial_summary(table, loaded_table, self.dpath)

    def load1(self):
        """
        Load basic data
        """
        table = self.dvc_manager.evaluation_table()
        self.summarize(table)
        evaluations = table[~table['raw'].isnull()]
        self.raw_df = raw_df = pd.DataFrame(evaluations)

        if 0:
            col_stats_df = unique_col_stats(raw_df)
            print('Column Unique Value Frequencies')
            print(col_stats_df.to_string())

        test_dset_freq = raw_df['test_dset'].value_counts()
        print(f'test_dset_freq={test_dset_freq}')

        print('\nRaw')
        num_files_summary(raw_df)

        # Remove duplicate predictions on effectively the same dataset.
        self.filt_df = filt_df = deduplicate_test_datasets(raw_df)

        print('\nDeduplicated (over test dataset)')
        num_files_summary(filt_df)

        eval_types_to_locs = ub.ddict(list)
        for k, group in filt_df.groupby(['dataset_code', 'model', 'pred_cfg']):
            eval_types = tuple(sorted(group['type'].unique()))
            eval_types_to_locs[eval_types].extend(group.index)
        print('Cross-Metric Comparable Locs')
        print(ub.repr2(ub.map_vals(len, eval_types_to_locs)))
        comparable_locs = list(ub.flatten(v for k, v in eval_types_to_locs.items() if len(k) > 0))
        self.comp_df = comp_df = filt_df.loc[comparable_locs]

        print('\nCross-Metric Comparable')
        num_files_summary(comp_df)

    def load2(self):
        """
        Load detailed data that might cross reference files
        """
        self.big_rows = load_extended_data(self.comp_df, self.dvc_dpath)
        set(r['expt'] for r in self.big_rows)

        orig_merged_df, other = clean_loaded_data(self.big_rows)
        self.orig_merged_df = orig_merged_df
        self.other = other

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
            'eval_act+pxl': 'SC',
            'eval_trk+pxl': 'BAS',
        }
        self.iarpa_metric_lut = {
            'eval_act+pxl': 'mean_f1',
            'eval_trk+pxl': 'BAS_F1',
        }
        self.pixel_metric_lut = {
            'eval_act+pxl': 'coi_mAP',
            'eval_trk+pxl': 'salient_AP',
        }
        self.human_mapping = human_mapping
        self.actcfg_to_label = other['actcfg_to_label']
        self.predcfg_to_label = other['predcfg_to_label']
        self.human_mapping.update(self.actcfg_to_label)
        self.human_mapping.update(self.predcfg_to_label)

    def load(self):
        self.load1()
        self.load2()


def plot_merged(reporter):
    self = reporter
    dpath = self.dpath
    orig_merged_df = self.orig_merged_df
    iarpa_metric_lut = self.iarpa_metric_lut
    pixel_metric_lut = self.pixel_metric_lut
    predcfg_to_label = self.predcfg_to_label
    actcfg_to_label = self.actcfg_to_label
    human_mapping = self.human_mapping

    # ['trk_thresh',
    #  'trk_morph_kernel',
    #  'trk_agg_fn',
    #  'trk_thresh_hysteresis',
    #  'trk_moving_window_size']

    common_plotkw = {
        # 'connect': 'expt',
        # 'mesh': 'model',
        # 'clique': 'model',
        'style': 'has_teamfeat',
        # 'star': 'in_production',
        'starkw': {'s': 500},
        's': 120,
    }

    merged_df = orig_merged_df.copy()

    if 0:
        from watch.utils import util_time
        deadline = util_time.coerce_datetime('2022-04-19')
        before_deadline = ((merged_df['pred_start_time'] < deadline) | merged_df['pred_start_time'].isnull())
        # after_deadline = ~before_deadline
        # merged_df = merged_df[after_deadline]
        merged_df = merged_df[before_deadline]

    if 0:
        chosen_pred_cfg = ub.invert_dict(predcfg_to_label)['pred_tta_time=0']
        # chosen_act_cfg = ub.invert_dict(actcfg_to_label)['trk_thresh=0.01,trk_use_viterbi=v1,v6']
        chosen_act_cfg = ub.invert_dict(actcfg_to_label)['trk_thresh=0,trk_use_viterbi=0']
        # chosen_pred_cfg = 'predcfg_abd043ec'
        # chosen_pred_cfg = 'predcfg_4d9147b0'
        # chosen_pred_cfg = 'predcfg_036fdb96'
        # chosen_act_cfg = 'actcfg_f1456a39'

        merged_df = merged_df[(
            (merged_df['pred_cfg'] == chosen_pred_cfg) &
            ((merged_df['act_cfg'] == chosen_act_cfg) | merged_df['act_cfg'].isnull())
        )]

    if 0:
        metrics = [
            'mean_f1',
            'BAS_F1',
            # 'salient_AP',
            # 'coi_mAP'
        ]
        # merged_df['pred_cfg'].value_counts()
        # merged_df['act_cfg'].value_counts()
        # HACK: need to maximize comparability, not the metric here.
        # Do this for viz purposes, dont present if it changes the conclusion
        # but might need to do this for visual clarity.
        rows = []
        for model, group in merged_df.groupby('model'):
            if len(group) > 1:
                chosen_idxs = [group[m].argmax() for m in metrics]
                row = group.iloc[sorted(set(chosen_idxs))]
            else:
                row = group
            rows.append(row)
        merged_df = pd.concat(rows)

    describe_varied(merged_df, dpath, human_mapping=human_mapping)

    plot_ta1_vs_l1(merged_df, human_mapping, iarpa_metric_lut, pixel_metric_lut, common_plotkw, dpath)

    plot_pixel_ap_verus_iarpa(merged_df, human_mapping, iarpa_metric_lut, pixel_metric_lut, common_plotkw, dpath)

    plot_pixel_ap_verus_auc(merged_df, human_mapping, iarpa_metric_lut, pixel_metric_lut, common_plotkw, dpath)

    plot_resource_versus_metric(merged_df, human_mapping, iarpa_metric_lut, pixel_metric_lut, common_plotkw, dpath)

    # plot_viterbii_analysis(merged_df, human_mapping, iarpa_metric_lut, pixel_metric_lut, common_plotkw, dpath)


def expt_over_time(merged_df, human_mapping, ):
    import kwplot
    fnum = 320
    merged_df['expt'].value_counts()
    # merged_df['CropDrop3_SC_s2wv_invar_scratch_V030'].value_counts()
    merged_df.query('in_production')
    # merged_df[merged_df['expt'] == 'BASELINE_EXPERIMENT_V001']

    metric = 'BAS_F1'
    # metric = 'mean_f1'

    expt = 'Drop3_SpotCheck_V323'
    # expt = 'CropDrop3_SC_V005'
    selected = merged_df[merged_df['expt'] == expt]

    for g, group in merged_df.groupby('expt'):
        if len(group['pred_cfg'].unique()) >= 2 and len(group['step'].unique()) > 2:
            print(group['expt'].unique())
            print(len(group))
            break
        pass

    selected['step']
    selected[metric]
    # subidx = selected.groupby(['step', 'pred_cfg'])[metric].idxmax().values
    subidx = selected.groupby(['step', 'pred_cfg'])[metric].idxmin().values
    selected = selected.loc[subidx]

    plotkw = {
        'x': 'step',
        'y': 'value',
        # 'star': 'in_production',
    }
    melted = selected.melt(['step', 'in_production', 'pred_cfg'], ['salient_AP', 'BAS_F1'])
    fig = kwplot.figure(fnum=fnum, doclf=True)
    ax = fig.gca()
    humanized_scatterplot(human_mapping, plot_type='line', data=melted, ax=ax, legend=0, style='pred_cfg', hue='variable', **plotkw)
    humanized_scatterplot(human_mapping, plot_type='scatter', data=melted, ax=ax, legend=True, style='pred_cfg', hue='variable', s=250, **plotkw)
    ax.set_title('Scores on Checkpoint Shortlist')

    """

    DVC_DPATH=$(smartwatch_dvc)
    jq '.images[] | .id' $DVC_DPATH/Aligned-Drop3-TA1-2022-03-10/data_nowv_vali_kr1.kwcoco.json

    kwcoco subset \
        --src $DVC_DPATH/Aligned-Drop3-TA1-2022-03-10/data_nowv_vali.kwcoco.json \
        --dst $DVC_DPATH/Aligned-Drop3-TA1-2022-03-10/data_nowv_vali_kr1_small.kwcoco.json \
        --select_videos '.name == "KR_R001"' \
        --select_images '.id <  6495 and .id >  6375'

    DVC_DPATH=$(smartwatch_dvc)
    TEST_DATASET=$DVC_DPATH/Aligned-Drop3-TA1-2022-03-10/data_nowv_vali_kr1_small.kwcoco.json
    EXPT_PATTERN="*"
    python -m watch.tasks.fusion.schedule_evaluation \
            --gpus="0,1" \
            --model_globstr="$DVC_DPATH/models/fusion/eval3_candidates/packages/Drop3_SpotCheck_V323/Drop3_SpotCheck_V323_epoch=18-step=12976.pt" \
            --test_dataset="$TEST_DATASET" \
            --workdir="$DVC_DPATH/_tmp/smalltest2" \
            --tta_fliprot=0 \
            --tta_time=0,6 \
            --chip_overlap=0.0,0.3 \
            --draw_heatmaps=0 \
            --enable_pred=1 \
            --enable_iarpa_eval=0 \
            --enable_eval=0 \
            --skip_existing=0 --backend=tmux --run=0

    DVC_DPATH=$(smartwatch_dvc)
    TEST_DATASET=$DVC_DPATH/Aligned-Drop3-TA1-2022-03-10/data_nowv_vali_kr1.kwcoco.json
    PRED_TTA0_OV0_DATASET=$DVC_DPATH/_tmp/smalltest2/pred/Drop3_SpotCheck_V323/pred_Drop3_SpotCheck_V323_epoch=18-step=12976/Aligned-Drop3-TA1-2022-03-10_data_nowv_vali_kr1_small.kwcoco/predcfg_4a02a01c/pred.kwcoco.json
    PRED_TTA6_OV3_DATASET=$DVC_DPATH/_tmp/smalltest2/pred/Drop3_SpotCheck_V323/pred_Drop3_SpotCheck_V323_epoch=18-step=12976/Aligned-Drop3-TA1-2022-03-10_data_nowv_vali_kr1_small.kwcoco/predcfg_4bef4048/pred.kwcoco.json
    WITHOUT_DATASET=$DVC_DPATH/_tmp/without_tta.kwcoco.json
    WITH_DATASET=$DVC_DPATH/_tmp/with_tta.kwcoco.json

    python -m watch.cli.coco_combine_features $TEST_DATASET $PRED_TTA0_OV0_DATASET --dst=$WITHOUT_DATASET --absolute=True
    python -m watch.cli.coco_combine_features $TEST_DATASET $PRED_TTA6_OV3_DATASET --dst=$WITH_DATASET --absolute=True

    smartwatch visualize $WITH_DATASET --channels="salient,red|green|blue" --animate=True --with_anns=True --only_boxes=True
    smartwatch visualize $WITHOUT_DATASET --channels="salient,red|green|blue" --animate=True --with_anns=True --only_boxes=True

    smartwatch visualize $DVC_DPATH/_tmp/smalltest/pred/Drop3_SpotCheck_V323/pred_Drop3_SpotCheck_V323_epoch=19-step=13659-v1/Aligned-Drop3-TA1-2022-03-10_data_nowv_vali_kr1.kwcoco/predcfg_8db7dd3b/pred.kwcoco.json --channels="salient,red|green|blue" --animate=True --with_anns=True

    echo $DVC_DPATH/_tmp/*/KR_R001/_anns/*


    python -m watch.tasks.fusion.predict \
            --
    ptyhon

    """
    pass


def plot_ta1_vs_l1(merged_df, human_mapping, iarpa_metric_lut, pixel_metric_lut, common_plotkw, dpath):
    import kwplot
    sns = kwplot.autosns()

    fnum = 0
    expt_group = dict(list(merged_df.groupby(['dataset_code', 'type'])))
    k1 = ('Aligned-Drop3-TA1-2022-03-10', 'eval_trk+pxl')
    k2 = ('Aligned-Drop3-L1', 'eval_trk+pxl')
    plot_name = 'ta1_vs_l1'
    param = 'Processing'
    plotkw = {
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
    }
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


def plot_pixel_ap_verus_iarpa(merged_df, human_mapping, iarpa_metric_lut, pixel_metric_lut, common_plotkw, dpath):
    import kwplot
    fnum = 0
    expt_group = dict(list(merged_df.groupby(['dataset_code', 'type'])))
    plot_name = 'pxl_vs_iarpa'

    plot_dpath_main = (dpath / plot_name).ensuredir()
    plot_dpath_parts = (dpath / (plot_name + '_parts')).ensuredir()

    for code_type, group in expt_group.items():

        dataset_code, type = code_type
        if type == 'eval_act+pxl':
            plotkw = {
                'x': pixel_metric_lut[type],
                'y': iarpa_metric_lut[type],
                'hue': 'sensorchan',
                **common_plotkw,
                # 'hue': 'trk_use_viterbi',
                # 'style': 'trk_thresh',
                # 'size': 'trk_thresh',
                # 'hue': 'pred_cfg',
                # 'hue': 'expt',
            }
        elif type == 'eval_trk+pxl':
            # hacks
            plotkw = {
                'x': pixel_metric_lut[type],
                'y': iarpa_metric_lut[type],
                'hue': 'sensorchan',
                **common_plotkw,
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

        fig = kwplot.figure(fnum=fnum, doclf=True)
        ax = fig.gca()
        n = len(data)
        ax = humanized_scatterplot(human_mapping, data=data, ax=ax, **plotkw)
        nice_type = human_mapping.get(type, type)
        ax.set_title(f'Pixelwise Vs IARPA metrics - {nice_type} - {dataset_code}\n{corr_lbl}, n={n}')
        # ax.set_xlim(0.1, 0.45)
        # ax.set_ylim(0.1, 0.45)
        fname = f'{dataset_code}_{type}_{plot_name}.png'
        fpath = plot_dpath_main / fname
        fig.set_size_inches(np.array([6.4, 4.8]) * 1.4)
        fig.tight_layout()
        fig.savefig(fpath)

        if 1:
            # TODO: incorporate that
            fig = kwplot.figure(fnum=fnum, doclf=True)
            ax = fig.gca()
            ax = humanized_scatterplot(human_mapping, data=data, ax=ax, legend=False, **plotkw)
            nice_type = human_mapping.get(type, type)
            ax.set_title(f'Pixelwise Vs IARPA metrics - {nice_type} - {dataset_code}\n{corr_lbl}, n={n}')
            # ax.set_xlim(0.1, 0.45)
            # ax.set_ylim(0.1, 0.45)
            fname = f'{dataset_code}_{type}_{plot_name}_nolegend.png'
            fpath = plot_dpath_parts / fname
            fig.set_size_inches(np.array([6.4, 4.8]) * 1.4)
            fig.tight_layout()
            fig.savefig(fpath)

            fig = kwplot.figure(fnum=fnum, doclf=True)
            ax = fig.gca()
            ax = humanized_scatterplot(human_mapping, data=data, ax=ax, legend=True, **plotkw)
            nice_type = human_mapping.get(type, type)
            ax.set_title(f'Pixelwise Vs IARPA metrics - {nice_type} - {dataset_code}\n{corr_lbl}, n={n}')
            fname = f'{dataset_code}_{type}_{plot_name}_nolegend.png'
            fpath = plot_dpath_parts / fname
            fig.set_size_inches(np.array([6.4, 4.8]) * 1.4)
            fig2 = kwplot.figure(fnum=1000 + fnum, doclf=True)
            fig2.set_size_inches(np.array([6.4, 4.8]) * 1.4)
            ax2 = fig2.gca()
            ax2.axis('off')
            handles = ax.get_legend_handles_labels()
            new_legend = ax2.legend(*handles, loc='lower center')
            humanize_legend(new_legend, human_mapping)
            fname = f'{dataset_code}_{type}_{plot_name}_onlylegend.png'
            fpath = plot_dpath_parts / fname
            try:
                new_extent = new_legend.get_window_extent()
                inv_scale = fig2.dpi_scale_trans.inverted()
                bbox = new_extent.transformed(inv_scale)
                newkw = {'bbox_inches': bbox}
            except Exception:
                newkw = {'bbox_inches': None}
            fig2.tight_layout()
            fig2.savefig(fpath, **newkw)
            kwplot.close_figures([fig2])
            cropwhite_ondisk(fpath)

        # legend = ax.get_legend()
        # plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0)
        # def export_legend(ax, filename="legend.png"):
        #     plt = kwplot.autoplt()
        #     fig2 = plt.figure()
        #     ax2 = fig2.add_subplot()
        #     ax2.axis('off')
        #     legend = ax2.legend(*ax.get_legend_handles_labels(), frameon=False, loc='lower center', ncol=10,)
        #     fig  = legend.figure
        #     fig.canvas.draw()
        #     bbox  = legend.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
        #     fig.savefig(filename, dpi="figure", bbox_inches=bbox)

        # def export_legend(legend, filename="legend.png", expand=[-5,-5,5,5]):
        #     fig = legend.figure
        #     fig.canvas.draw()
        #     bbox  = legend.get_window_extent()
        #     bbox = bbox.from_extents(*(bbox.extents + np.array(expand)))
        #     bbox = bbox.transformed(fig.dpi_scale_trans.inverted())
        #     fig.savefig(filename, dpi="figure", bbox_inches=bbox)

        # export_legend(legend)
        # https://stackoverflow.com/questions/4534480/get-legend-as-a-separate-picture-in-matplotlib

        # fname = f'{dataset_code}_{type}_{plot_name}.png'
        # fpath = plot_dpath / fname
        # fig.set_size_inches(np.array([6.4, 4.8]) * 1.4)
        # fig.tight_layout()
        # fig.savefig(fpath)


def plot_pixel_ap_verus_auc(merged_df, human_mapping, iarpa_metric_lut, pixel_metric_lut, common_plotkw, dpath):
    import kwplot
    fnum = 10
    expt_group = dict(list(merged_df.groupby(['dataset_code', 'type'])))
    plot_name = 'pxl_vs_auc'
    plot_dpath = (dpath / plot_name).ensuredir()
    for code_type, group in expt_group.items():

        group = group[~group['sensorchan'].isnull()]
        # group['has_teamfeat'] = group['sensorchan'].apply(lambda x: (('depth' in x) or ('invariants' in x) or ('matseg' in x) or ('land' in x)))

        dataset_code, type = code_type
        if type == 'eval_act+pxl':
            plotkw = {
                'x': pixel_metric_lut[type],
                'y': 'coi_mAUC',
                'hue': 'sensorchan',
                **common_plotkw,
                # 'hue': 'trk_use_viterbi',
                # 'style': 'trk_thresh',
                # 'size': 'trk_thresh',
                # 'hue': 'pred_cfg',
                # 'hue': 'expt',
            }
        elif type == 'eval_trk+pxl':
            plotkw = {
                'x': pixel_metric_lut[type],
                'y': 'salient_AUC',
                'hue': 'sensorchan',
                **common_plotkw,
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
        ax = humanized_scatterplot(human_mapping, data=group, ax=ax, **plotkw)
        nice_type = human_mapping.get(type, type)
        ax.set_title(f'Pixelwise metrics - {nice_type} - {dataset_code}\n{corr_lbl}')
        fig.set_size_inches(16.85, 8.82)
        fname = f'{dataset_code}_{type}_{plot_name}.png'
        fpath = plot_dpath / fname
        fig.savefig(fpath)


def plot_resource_versus_metric(merged_df, human_mapping, iarpa_metric_lut, pixel_metric_lut, common_plotkw, dpath):
    import kwplot
    fnum = 20
    expt_group = dict(list(merged_df.groupby(['dataset_code', 'type'])))
    plot_name = 'resource_vs_metric'
    plot_dpath = (dpath / plot_name).ensuredir()
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
                fnum += 1

                group['pred_tta_time'] = group['pred_tta_time'].astype(str)
                group['pred_tta_fliprot'] = group['pred_tta_fliprot'].astype(str)

                group.loc[group['pred_tta_time'] == 'nan', 'pred_tta_time'] = '0.0'
                group.loc[group['pred_tta_fliprot'] == 'nan', 'pred_tta_fliprot'] = '0.0'
                dataset_code, type = code_type
                if type == 'eval_act+pxl':
                    plotkw = {
                        'x': resource_type,
                        'y': metric_lut[type],
                        'hue': 'sensorchan',
                        **common_plotkw,
                        # 'style': 'pred_cfg',
                        # 'hue': 'pred_tta_fliprot',
                        # 'hue': 'pred_tta_time',
                        # 'size': 'pred_tta_fliprot',
                        # 'style': 'hardware',
                    }
                elif type == 'eval_trk+pxl':
                    plotkw = {
                        'x': resource_type,
                        'y': metric_lut[type],
                        'hue': 'sensorchan',
                        **common_plotkw,
                        # 'hue': 'pred_tta_time',
                        # 'size': 'pred_tta_fliprot',
                        # 'style': 'hardware',
                    }
                else:
                    raise KeyError(type)
                # fig = kwplot.figure(fnum=fnum, pnum=pnum_())
                # fig.get_size_inches()
                # fig.set_size_inches(17.85,  6.82)
                data = group

                fig = kwplot.figure(fnum=fnum)
                ax = fig.gca()
                ax = humanized_scatterplot(human_mapping, data=data, ax=ax, **plotkw)
                nice_type = human_mapping.get(type, type)
                ax.set_title(f'{human_resource_type} vs {human_metric_type} - {nice_type} - {dataset_code}')
                fig.set_size_inches(np.array([6.4, 4.8]) * 1.4)
                fname = f'{dataset_code}_{type}_{resource_type}_{plot_name}.png'
                fpath = plot_dpath / fname
                fig.tight_layout()
                fig.savefig(fpath)

                # if 1:
                #     # TODO: incorporate that
                #     fig = kwplot.figure(fnum=fnum, doclf=True)
                #     ax = humanized_scatterplot(human_mapping, data=data, ax=ax, legend=False, **plotkw)
                #     nice_type = human_mapping.get(type, type)
                #     ax.set_title(f'{human_resource_type} vs {human_metric_type} - {nice_type} - {dataset_code}')
                #     fig.set_size_inches(np.array([6.4, 4.8]) * 1.4)
                #     fname = f'{dataset_code}_{type}_{resource_type}_{plot_name}.png'
                #     fpath = plot_dpath / fname
                #     fig.tight_layout()
                #     fig.savefig(fpath)

                #     # fig2 = kwplot.figure(fnum=1000 + fnum, doclf=True)
                #     # fig2.set_size_inches(np.array([6.4, 4.8]) * 1.4)
                #     # ax2 = fig2.gca()
                #     # ax2.axis('off')
                #     # handles = ax.get_legend_handles_labels()
                #     # new_legend = ax2.legend(*handles, loc='lower center')
                #     # humanize_legend(new_legend, human_mapping)
                #     # fname = f'{dataset_code}_{type}_{plot_name}_onlylegend.png'
                #     # fpath = plot_dpath / fname
                #     # try:
                #     #     new_extent = new_legend.get_window_extent()
                #     #     inv_scale = fig2.dpi_scale_trans.inverted()
                #     #     bbox = new_extent.transformed(inv_scale)
                #     #     newkw = {'bbox_inches': bbox}
                #     # except Exception:
                #     #     newkw = {'bbox_inches': None}
                #     # fig2.tight_layout()
                #     # fig2.savefig(fpath, **newkw)


def plot_viterbii_analysis(merged_df, human_mapping, iarpa_metric_lut, pixel_metric_lut, common_plotkw):
    import kwplot
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

    code_type = ('Cropped-Drop3-TA1-2022-03-10', 'eval_act+pxl')
    dataset_code, type = code_type
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


def num_files_summary(df):
    expt_group = dict(list(df.groupby('dataset_code')))
    filt_summaries = []
    for dataset_code, group in sorted(expt_group.items())[::-1]:
        # print('dataset_code = {!r}'.format(dataset_code))
        # print('Column Unique Value Frequencies')
        # col_stats_df2 = unique_col_stats(group)
        # print(col_stats_df2.to_string())
        row = {}
        type_evals = ub.dict_hist(group['type'])
        row['dataset_code'] = dataset_code
        row['num_experiments'] = len(group['expt'].unique())
        row['num_models'] = len(group['model'].unique())
        row['num_pxl_evals'] = type_evals.get('eval_pxl', 0)
        row['num_bas_evals'] = type_evals.get('eval_trk', 0)
        row['num_sc_evals'] = type_evals.get('eval_act', 0)
        filt_summaries.append(row)
    _summary_df = pd.DataFrame(filt_summaries)
    total_row = _summary_df.sum().to_dict()
    total_row['dataset_code'] = '*'
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
    col_stats_df = col_stats_df.drop(['dataset_code', 'dvc', 'raw'], axis=1)
    col_stats_df = col_stats_df.astype(int)
    return col_stats_df


def load_extended_data(df, dvc_dpath):
    from watch.tasks.fusion import aggregate_results as agr
    rows = df.to_dict('records')
    big_rows = []
    errors = []
    for row in ub.ProgIter(rows, desc='load'):
        big_row = row.copy()
        fpath = row['raw']
        try:
            if row['type'] == 'eval_pxl':
                pxl_info = agr.load_pxl_eval(fpath, dvc_dpath)
                big_row['info'] = pxl_info
            elif row['type'] == 'eval_act':
                sc_info = agr.load_sc_eval(fpath, dvc_dpath)
                big_row['info'] = sc_info
            elif row['type'] == 'eval_trk':
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
    """
    Turn the nested "loaded" data into flat data for tabulation.
    Also combine eval types together into a single row per model / config.
    """
    from watch.tasks.fusion import aggregate_results as agr
    try:
        from kwcoco._experimental.sensorchan import concise_sensor_chan, sensorchan_parts
    except Exception:
        # hack
        def sensorchan_parts(x):
            return x.split(',')
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
        'G|R|N|S|H|land:8',
    }
    chan_blocklist = {
        'R|G',
        'G|R',
        'G|R|N|S|H',
        'R|G|land:8',
        'RGB|near-ir1|depth',
        'G|R|N|S|H|land:8|matseg:4|mat_up5:64',
        'BGRNSH|land:8',
    }

    for big_row in ub.ProgIter(big_rows, desc='big rows'):
        # fpath = big_row['raw']
        row = ub.dict_diff(big_row, {'info'})
        info = big_row['info']

        param_type = info['param_types']

        meta = param_type['meta']
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
                        print(f'Unknown real_chan={real_chan}')
                    real_chan_parts.add(real_chan)
                    real_sensors.append(input_row['sensor'])
                    senschan_parts.append('{}:{}'.format(input_row['sensor'], real_chan))
            sensorchan = ','.join(sorted(set(senschan_parts)))
            sensorchan = concise_sensor_chan(sensorchan)
            request_chan_parts = set(fit_params['channels'].split(','))
            if not request_chan_parts.issubset(real_chan_parts):
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
            fit_params['bad_channels'] = True

        # MANUAL HACK:
        if 1:
            sensorchan = ','.join([p for p in sensorchan_parts(sensorchan) if p not in blocklist])

        fit_params['sensorchan'] = sensorchan
        row['has_teamfeat'] = _is_teamfeat(sensorchan)

        fit_param_keys2 = list(fit_param_keys) + ['bad_channels', 'channels']
        selected_fit_params = ub.dict_isect(fit_params, fit_param_keys2)

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
        row.update(**meta)
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
        pxl_group = type_to_subgroup.pop('eval_pxl', None)
        if pxl_group is not None:
            if len(pxl_group) > 1:
                print(f'Warning more than one pixel group for {pred_key}')
            pxl_row = pxl_group.iloc[0]

            if len(type_to_subgroup) == 0:
                pxl_row = pxl_row.to_dict()
                if not math.isnan(pxl_row.get('coi_mAP', np.nan)):
                    srow = pxl_row.copy()
                    srow['type'] = 'eval_act+pxl'
                    merged_rows.append(srow)

                if not math.isnan(pxl_row.get('salient_AP', np.nan)):
                    srow = pxl_row.copy()
                    srow['type'] = 'eval_trk+pxl'
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

    total_carbon_cost = simple_df[simple_df['type'] == 'eval_pxl']['co2_kg'].sum()
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

    # TODO: compute total steps including with initialized continuations
    from watch.tasks.fusion import dvc_sync_manager
    epoch_info = merged_df['model'].apply(dvc_sync_manager.checkpoint_filepath_info).values
    merged_df['epoch'] = [e.get('epoch', None) if e else None for e in epoch_info]
    merged_df['step'] = [e.get('step', None) if e else None for e in epoch_info]

    merged_df['init'] = merged_df['init'].apply(lambda x: x[:-3] if x.endswith('.pt') else x)

    known_models = merged_df['model'].unique()
    init_models = merged_df['init'].unique()
    traceable = sorted(set(init_models) & set(known_models))

    merged_df = merged_df.reset_index(drop=True)  # because of enumerate
    init_to_idxs = ub.map_vals(sorted, ub.invert_dict(dict(enumerate(merged_df['init'])), 0))
    model_to_idxs = ub.map_vals(sorted, ub.invert_dict(dict(enumerate(merged_df['model'])), 0))
    import networkx as nx
    g = nx.DiGraph()
    for model in traceable:
        pred_idxs = model_to_idxs[model]
        succ_idxs = init_to_idxs[model]
        pred_df = merged_df.loc[pred_idxs]
        succ_df = merged_df.loc[succ_idxs]
        parents = pred_df['model'].unique()
        children = succ_df['model'].unique()
        for p in parents:
            for c in children:
                g.add_edge(p, c)

    if 0:
        from cmd_queue.util import graph_str
        print(graph_str(g))

    # Ensure we compute total epochs for earlier models first
    merged_df['total_steps'] = merged_df['step']
    for model in list(nx.topological_sort(g)):
        pred_idxs = model_to_idxs[model]
        succ_idxs = init_to_idxs.get(model, [])
        if len(succ_idxs):
            pred_df = merged_df.loc[pred_idxs]
            succ_df = merged_df.loc[succ_idxs]
            assert len(pred_df['total_steps'].unique()) == 1
            prev = pred_df['total_steps'].iloc[0]
            merged_df.loc[succ_idxs, 'total_steps'] += prev

    if 0:
        print(ub.repr2(merged_df.columns.tolist()))

    # Flag which models went into production.
    from watch.tasks.fusion import production
    import kwarray
    production_models = [row['name'].replace('.pt', '') for row in production.PRODUCTION_MODELS]
    stared_models = set(merged_df['model'].unique()) & set(production_models)
    star_flags = kwarray.isect_flags(merged_df['model'], stared_models)
    merged_df['in_production'] = star_flags

    merged_df.loc[merged_df['trk_use_viterbi'].isnull(), 'trk_use_viterbi'] = 0

    merged_df['track_agg_fn'] = merged_df['trk_agg_fn'].fillna('probs')
    flags = 1 - group['trk_thresh_hysteresis'].isnull()
    merged_df['trk_thresh_hysteresis'] = merged_df['trk_thresh_hysteresis'].fillna(0) + (flags * merged_df['trk_thresh'])

    actcfg_to_label = other['actcfg_to_label']
    predcfg_to_label = other['predcfg_to_label']
    label_to_cfgstr = ub.invert_dict(actcfg_to_label)
    try:
        # hack
        a = label_to_cfgstr['trk_thresh=0,trk_use_viterbi=0']
        b = label_to_cfgstr['trk_thresh=0.0,trk_use_viterbi=0']
        merged_df.loc[merged_df['act_cfg'] == b, 'act_cfg'] = a
    except Exception:
        pass

    is_l1 = np.array(['L1' in c for c in merged_df['dataset_code']])
    is_ta1 = np.array(['TA1' in c for c in merged_df['dataset_code']])
    merged_df.loc[is_l1, 'Processing'] = 'L1'
    merged_df.loc[is_ta1, 'Processing'] = 'TA1'

    return merged_df, other


def initial_summary(table, loaded_table=None, dpath=None):
    if 0:
        description = table[['type', 'dataset_code', 'expt', 'model', 'pred_cfg', 'act_cfg', 'trk_cfg']].describe()
        print(description)

        summary_stats = []
        for dset_code, group in table.groupby(['dataset_code']):
            # dataset_code = DSET_CODE_TO_GSD.get(dset_code, np.nan)
            # dataset_code = DSET_CODE_TO_GSD.get(dset_code, np.nan)
            gsd = DSET_CODE_TO_GSD.get(dset_code, np.nan)
            table.loc[group.index, 'gsd'] = gsd

            type_hist = group.groupby('type').size()
            model_hist = group.groupby('model').size()
            expt_hist = group.groupby('expt').size()

            row = {
                'dataset_code': dset_code,
                'num_experiments': len(expt_hist),
                'num_models': len(model_hist),
                'num_pxl_evals': type_hist.get('eval_pxl', 0),
                'num_bas_evals': type_hist.get('eval_trk', 0),
                'num_sc_evals': type_hist.get('eval_act', 0),
            }
            summary_stats.append(row)
        _summary_df = pd.DataFrame(summary_stats)
        total_row = _summary_df.sum().to_dict()
        total_row['dataset_code'] = '*'
        total_row['dataset_code'] = '*'
        summary_df = pd.DataFrame(summary_stats + [total_row])
        print('Number of Models & Evaluations')
        print(summary_df.to_string(index=False))

    # Alternate way to compute
    if loaded_table is not None:
        co2_rows = []
        hour_rows = []
        for _, row in loaded_table.iterrows():
            if row['type'] == 'eval_pxl':
                co2_rows += [row['info']['param_types']['resource'].get('co2_kg', np.nan)]
                hour_rows += [row['info']['param_types']['resource'].get('total_hours', np.nan)]
        co2_rows = np.array(co2_rows)
        hour_rows = np.array(hour_rows)
        est_co2_missing = np.nanmean(co2_rows)
        est_hours_missing = np.nanmean(hour_rows)
        co2_rows[np.isnan(co2_rows)] = est_co2_missing
        hour_rows[np.isnan(hour_rows)] = est_hours_missing
        total_co2 = co2_rows.sum()
        total_hours = hour_rows.sum()

        num_expt_models = table[['expt', 'model']].nunique()
        num_eval_types = table['type'].value_counts()

        summary_df = pd.concat([num_expt_models, num_eval_types]).to_frame().T
        order = ['expt', 'model', 'eval_pxl', 'eval_trk', 'eval_act']
        summary_df = summary_df[order]
        summary_df['co2_kg'] = [total_co2]
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
            'hours': 'Total Compute Time (hours)',
        }
        human_df = summary_df.rename(humanize, axis=1)
        human_df = human_df.applymap(lambda x: str(x) if isinstance(x, int) else '{:0.2f}'.format(x))
        table_style = human_df.T.style
        table_style = table_style.set_caption('Experimental Summary')

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


def describe_varied(merged_df, dpath, human_mapping=None):
    # import pprint
    expt_group = dict(list(merged_df.groupby(['dataset_code', 'type'])))
    fnum = 40

    human_mapping.update({
        'time_steps': 'Time Steps (frames)',
        'chip_size': 'Chip Size (pxls)',
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
        part = group[param_keys].fillna('null')
        # varied_series = part.nunique()
        # varied_series = varied_series[varied_series > 1]
        # print(varied_series)
        # human_df = varied_df.rename(human_mapping, axis=0)

        param_to_row = ub.ddict(dict)
        param_to_hist = {k: part[k].value_counts() for k in param_keys}
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
        param_keys = fit_param_keys
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


def deduplicate_test_datasets(raw_df):
    """
    The same model might have been run on two variants of the dataset.
    E.g. a RGB model might have run on data_vali.kwcoco.json and
    combo_DILM.kwcoco.json. The system sees these as different datasets
    even though the model will use the same subset of both. We define
    a heuristic ordering and then take just one of them.
    """
    preference = {
        'Cropped-Drop3-TA1-2022-03-10_combo_DLM_s2_wv_vali.kwcoco': 0,
        'Cropped-Drop3-TA1-2022-03-10_combo_DL_s2_wv_vali.kwcoco': 1,
        'Cropped-Drop3-TA1-2022-03-10_data_wv_vali.kwcoco': 2,
        'Aligned-Drop3-TA1-2022-03-10_combo_LM_nowv_vali.kwcoco': 0,
        'Aligned-Drop3-TA1-2022-03-10_combo_LM_vali.kwcoco': 1,
    }
    FILTER_DUPS = 1
    if FILTER_DUPS:
        keep_locs = []
        for k, group in raw_df.groupby(['dataset_code', 'model', 'pred_cfg', 'type']):
            prefs = group['test_dset'].apply(lambda x: preference.get(x, 0))
            keep_flags = prefs == prefs.min()
            keep_locs.extend(group[keep_flags].index)
        print(f'Keep {len(keep_locs)} / {len(raw_df)} drop3 evals')
        filt_df = raw_df.loc[keep_locs]
    else:
        filt_df = raw_df.copy()
    return filt_df


def dataset_summary_tables(dpath):
    import watch
    import kwcoco

    memo_kwcoco_load = ub.memoize(kwcoco.CocoDataset)

    dvc_dpath = watch.find_smart_dvc_dpath()
    rows = []
    DSET_CODE_TO_TASK = {
        'Aligned-Drop3-TA1-2022-03-10': 'bas',
        'Aligned-Drop3-L1': 'bas',
        'Cropped-Drop3-TA1-2022-03-10': 'sc',
    }
    for bundle_name in DSET_CODE_TO_TASK.keys():
        task = DSET_CODE_TO_TASK[bundle_name]
        gsd = DSET_CODE_TO_GSD[bundle_name]
        bundle_dpath = dvc_dpath / bundle_name
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
    title = 'Dataset Summary'
    df2_style = humanize_dataframe(df, col_formats, human_labels=human_labels,
                                   title=title)
    fpath = dpath / 'dataset_summary.png'
    dfi_table(df2_style, fpath, fontsize=32, show='eog')


def humanize_dataframe(df, col_formats, human_labels=None, index_format=None,
                       title=None):
    import humanize
    df2 = df.copy()
    for col, fmt in col_formats.items():
        print(f'col={col}')
        print(f'fmt={fmt}')
        if fmt == 'intcomma':
            df2[col] = df[col].apply(humanize.intcomma)
        if fmt == 'concice_si_display':
            print(f'fmt={fmt}')
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
            's': _starkw.get('s', 10) + 250,
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
