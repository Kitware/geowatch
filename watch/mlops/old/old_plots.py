# Old plots that we might remove entirely soon


# def plot_ta1_vs_l1(merged_df, human_mapping, iarpa_metric_lut, pixel_metric_lut, common_plotkw, dpath, fnum=None):
#     import kwplot
#     sns = kwplot.autosns()

#     if fnum is None:
#         fnum = kwplot.next_fnum()

#     fnum = 0
#     expt_group = dict(list(merged_df.groupby(['test_dset', 'type'])))
#     k1 = ('Aligned-Drop3-TA1-2022-03-10', 'eval_trk+pxl')
#     k2 = ('Aligned-Drop3-L1', 'eval_trk+pxl')
#     plot_name = 'ta1_vs_l1'
#     param = 'Processing'
#     plotkw = ub.udict({
#         'x': 'salient_AP',
#         'y': 'BAS_F1',
#         'hue': param,
#         # 'hue': 'sensorchan',
#         **common_plotkw,
#         # 'hue': 'trk_use_viterbi',
#         # 'style': 'trk_thresh',
#         # 'size': 'trk_thresh',
#         # 'hue': 'pred_cfg',
#         # 'hue': 'expt',
#     })
#     x = plotkw['x']
#     y = plotkw['y']
#     plotkw.pop('style', None)

#     plot_dpath = (dpath / plot_name).ensuredir()

#     from watch.utils import result_analysis
#     all_param_keys = ub.oset.union(trk_param_keys, pred_param_keys,
#                                    fit_param_keys, act_param_keys)

#     all_param_keys = {
#         'trk_thresh',
#         # 'trk_morph_kernel',
#         'trk_agg_fn', 'trk_thresh_hysteresis', 'trk_moving_window_size',
#         'pred_tta_fliprot', 'pred_tta_time', 'pred_chip_overlap',
#         # 'sensorchan',
#         # 'time_steps',
#         # 'chip_size',
#         # 'chip_overlap',
#         # 'arch_name',
#         # 'optimizer', 'time_sampling', 'time_span', 'true_multimodal',
#         # 'accumulate_grad_batches', 'modulate_class_weights', 'tokenizer',
#         # 'use_grid_positives', 'use_cloudmask', 'upweight_centers',
#         # 'temporal_dropout', 'stream_channels', 'saliency_loss', 'class_loss',
#         # 'init', 'learning_rate', 'decoder',
#         'trk_use_viterbi'
#     }

#     data1 = expt_group[k1]
#     data2 = expt_group[k2]
#     data = pd.concat([data1, data2])
#     data = data[~data['has_teamfeat']]

#     if 0:
#         # Reduce to comparable groups according to the abalate criterion
#         results_list = []
#         for row in data.to_dict('records'):
#             params = ub.dict_isect(row, {'Processing', *all_param_keys})
#             metrics = ub.dict_isect(row, {x, y})
#             result = result_analysis.Result(None, params, metrics)
#             results_list.append(result)
#         self = analysis = result_analysis.ResultAnalysis(
#             results_list, default_objective='max', metrics={'BAS_F1'})
#         comparable_data = []
#         for group in self.abalation_groups('Processing'):
#             print(len(group))
#             if len(group['Processing'].unique()) > 1:
#                 comparable_data.append(group)
#         comparable = pd.concat(comparable_data)
#         data = data.iloc[comparable.index]

#     if 0:
#         # Remove duplicates for clarity
#         rows = []
#         for model, group in data.groupby('model'):
#             if len(group) > 1:
#                 # group.pred_cfg.value_counts()
#                 # group.trk_cfg.value_counts()
#                 idx = group[y].argmax()
#                 row = group.iloc[idx]
#             else:
#                 row = group.iloc[0]
#             rows.append(row)
#         data = pd.DataFrame(rows)

#     results_list = []
#     for row in data.to_dict('records'):
#         # params = ub.dict_isect(row, {'Processing', *all_param_keys})
#         params = ub.dict_isect(row, {'Processing'})
#         metrics = ub.dict_isect(row, {x, y})
#         result = result_analysis.Result(None, params, metrics)
#         results_list.append(result)
#     analysis = result_analysis.ResultAnalysis(
#         results_list, default_objective='max', metrics={'BAS_F1', 'salient_AP'})

#     try:
#         analysis.run()
#     except TypeError:
#         raise

#     # kitware_green = '#3caf49'
#     # kitware_blue = '#006ab6'
#     kitware_green = '#3EAE2B'
#     kitware_blue = '#0068C7'

#     self = analysis
#     conclusions = analysis.conclusions()

#     fig = kwplot.figure(fnum=fnum, doclf=True)
#     ax = fig.gca()
#     palette = {
#         'L1': kitware_blue,
#         'TA1': kitware_green,
#     }
#     ax = humanized_scatterplot(human_mapping, data=data, ax=ax, legend=True, palette=palette, **plotkw)
#     # nice_type = human_mapping.get(type, type)
#     # ax.set_title('TA1 vs L1' + '\n' + '\n'.join(conclusions))
#     ax.set_title('TA1 vs L1')
#     fname = f'{plot_name}_scatter.png'
#     fpath = plot_dpath / fname
#     fig.set_size_inches(np.array([6.4, 4.8]) * 1.4)
#     fig.tight_layout()
#     fig.savefig(fpath)

#     bas_conclusion = '\n'.join([c for c in conclusions if 'BAS_F1' in c])

#     fnum = fnum + 1
#     fig = kwplot.figure(fnum=fnum, doclf=True)
#     ax = fig.gca()
#     ax.set_title('BAS scores: TA1 vs L1')
#     sns.violinplot(data=merged_df, x='Processing', y='BAS_F1', palette=palette)
#     ax.set_title('TA1 vs L1' + '\n' + bas_conclusion)
#     fname = f'{plot_name}_violin.png'
#     fpath = plot_dpath / fname
#     fig.set_size_inches(np.array([6.4, 4.8]) * 1.0)
#     fig.tight_layout()
#     fig.savefig(fpath)
#     cropwhite_ondisk(fpath)

#     fnum = fnum + 1
#     fig = kwplot.figure(fnum=fnum, doclf=True)
#     ax = fig.gca()
#     sns.boxplot(data=merged_df, x='Processing', y='BAS_F1', palette=palette)
#     ax.set_title('TA1 vs L1' + '\n' + bas_conclusion)
#     fname = f'{plot_name}_boxwhisker.png'
#     fpath = plot_dpath / fname
#     fig.set_size_inches(np.array([6.4, 4.8]) * 1.0)
#     fig.tight_layout()
#     fig.savefig(fpath)
#     cropwhite_ondisk(fpath)

#     # ax.set_title('TA1 vs L1' + '\n' + '\n'.join(conclusions))


# def plot_viterbii_analysis(merged_df, human_mapping, iarpa_metric_lut, pixel_metric_lut, common_plotkw):
#     import kwplot
#     expt_group = dict(list(merged_df.groupby(['test_dset', 'type'])))
#     from watch.utils import result_analysis

#     nan_defaults = {
#         'modulate_class_weights': '',
#         'trk_agg_fn': 'probs',
#         'pred_tta_fliprot': 0,
#         'pred_tta_time': 0,
#         'pred_chip_overlap': 0.3,
#         'decoder': 'mlp',
#         'trk_morph_kernel': 3,
#         'stream_channels': 8,
#         'trk_thresh': 0.2,
#         'trk_use_viterbi': 0,
#         'trk_thresh_hysteresis': None,
#         'trk_moving_window_size': None,
#         'use_cloudmask': 0,
#     }

#     merged_df.loc[merged_df['use_cloudmask'].isnull(), 'use_cloudmask'] = 0

#     code_type = ('Cropped-Drop3-TA1-2022-03-10', 'eval_act+pxl')
#     test_dset, type = code_type
#     group = expt_group[code_type]

#     iarpa_metric = iarpa_metric_lut[type]
#     pixel_metric = pixel_metric_lut[type]
#     # metric = pixel_metric
#     metric = iarpa_metric

#     results_list = []
#     for row in group.to_dict('records'):
#         metric_val = row[metric]
#         if math.isnan(metric_val):
#             continue
#         metrics = {
#             metric: metric_val,
#         }
#         params = ub.dict_isect(row, fit_param_keys + pred_param_keys + trk_param_keys + act_param_keys)
#         params['modulate_class_weights']

#         for k, v in params.items():
#             if isinstance(v, float) and math.isnan(v):
#                 if k == 'sensorchan':
#                     params['sensorchan'] = params['channels']
#                 else:
#                     params[k] = nan_defaults[k]

#         result = result_analysis.Result(None, params, metrics)
#         results_list.append(result)

#     ignore_params = {'bad_channels'}
#     ignore_metrics = {}
#     abalation_orders = {1}
#     analysis = result_analysis.ResultAnalysis(
#         results_list, ignore_params=ignore_params,
#         # metrics=['coi_mAPUC', 'coi_APUC'],
#         # metrics=['salient_AP'],
#         # metrics=['coi_mAP', 'salient_AP'],
#         metrics=[metric],
#         metric_objectives={
#             'salient_AP': 'max',
#             'coi_mAP': 'max',
#             'mean_f1': 'max',
#         },
#         ignore_metrics=ignore_metrics,
#         abalation_orders=abalation_orders
#     )
#     # try:
#     #     analysis.run()
#     # except TypeError:
#     #     raise

#     param = 'trk_use_viterbi'
#     scored_obs = analysis.abalate(param)
#     ab_rows = []
#     pts1 = []
#     pts2 = []
#     for obs in scored_obs:
#         ab_row = obs.melt(['trk_use_viterbi']).pivot(['variable'], ['trk_use_viterbi'], 'value').reset_index(drop=True)
#         if (~ab_row.isnull()).values.sum() > 1:

#             # hack
#             obspt = obs.copy()
#             obspt[param] = obs[param].astype(bool).astype(int)
#             pts1.append(obspt.values[0])
#             pts2.append(obspt.values[1])
#             ab_rows.append(ab_row)
#     # ab_df = pd.concat(ab_rows).reset_index(drop=True)
#     # obs[param]

#     plt = kwplot.autoplt()
#     ax = plt.gca()
#     kwplot.draw_line_segments(pts1, pts2)
#     # ticks = ax.set_xticks([0, 1], ['0', 'v6,v1'])
#     ax.set_ylabel(metric)
#     ax.set_xlabel(param)
#     ax.set_title('Viterbi A/B tests')

#     return

#     # ab_df
#     # .reset_index()
#     # ab_melt_df = ab_df.melt(id_vars=['index'], value_vars=ab_row.columns)
#     # sns = kwplot.autosns()
#     # sns.lineplot(data=ab_melt_df, x=param, y='value')

#     import seaborn as sns
#     kwplot.figure(fnum=1000)
#     sns.violinplot(data=merged_df, x='temporal_dropout', y=pixel_metric)

#     # TODO: translate to chip_dims
#     # kwplot.figure(fnum=1001)
#     # sns.violinplot(data=merged_df, x='chip_size', y=pixel_metric)

#     kwplot.figure(fnum=1002, doclf=True)
#     sns.violinplot(data=merged_df, x='time_steps', y=pixel_metric)

#     kwplot.figure(fnum=1003, doclf=True)
#     sns.violinplot(data=merged_df, x='trk_use_viterbi', y=pixel_metric)

#     kwplot.figure(fnum=1004, doclf=True)
#     ax = sns.violinplot(data=group, x='sensorchan', y=pixel_metric)
#     for xtick in ax.get_xticklabels():
#         xtick.set_rotation(90)

#     kwplot.figure(fnum=3)
#     sns.violinplot(data=merged_df, x='saliency_loss', y=pixel_metric)

#     group[heuristics.fit_param_keys]
#     group[heuristics.fit_param_keys]
