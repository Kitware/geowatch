# from kwutil.util_yaml import Yaml
from watch.mlops import smart_global_helper
# from watch.mlops import aggregate_plots
from watch.mlops.aggregate import AggregateEvluationConfig
from watch.utils import util_kwplot
from watch.utils import util_pandas
import kwimage
# import kwplot
import rich
import watch
import ubelt as ub

expt_dvc_dpath = watch.find_dvc_dpath(tags='phase2_expt', hardware='auto')

config = AggregateEvluationConfig(**{
    'target': expt_dvc_dpath / 'aggregate_results/mlops-2023-10-30/*/*.csv.zip',
    'pipeline': 'sc',
    'io_workers': 10,
})
rich.print('config = {}'.format(ub.urepr(config, nl=1)))

eval_type_to_aggregator = config.coerce_aggregators()

agg = eval_type_to_aggregator['sc_poly_eval']
agg.output_dpath = expt_dvc_dpath / 'aggregate_results/mlops-2023-10-30-summary'

min_date = min(agg.table['context.sc_pxl.start_timestamp'])
max_date = max(agg.table['context.sc_poly_eval.stop_timestamp'])
print(f'min_date={min_date}')
print(f'max_date={max_date}')

df = agg.table

df = util_pandas.DataFrame(agg.table)
print(agg.table['machine.sc_pxl.host'].unique())

flags = df['params.sc_pxl.package_fpath'].str.contains('Drop4_tune_V30_8GSD_V3_epoch=2-step=17334.pt')
print(f'{flags.sum()} / {len(flags)}')
df = df[flags]

flags = df['resolved_params.sc_poly.thresh'] == 0.07
# flags |= df['resolved_params.sc_poly.thresh'] == 0.1
df = df[flags]

score_zero = df['params.sc_poly.site_score_thresh'] == 0
score_zero |= df['params.sc_poly.site_score_thresh'].isna()
df = df[score_zero]

no_smoothing = df['params.sc_poly.smoothing'] == 0
no_smoothing |= df['params.sc_poly.smoothing'].isna()
df = df[no_smoothing]

flags = (df['params.sc_pxl.fixed_resolution'] == '8GSD')
df = df[flags]

flags = df['machine.sc_pxl.host'] != 'compute1-exec-197.ris.wustl.edu'
df = df[flags]

subagg = agg.filterto(index=df.index)
subagg.table['machine.sc_pxl.host'].unique()

subagg.index[['param_hashid', 'region_id']].sort_values('param_hashid').value_counts()

if 1:
    kw_df = df[df['region_id'] == 'KW_C501']
    top_idx = kw_df['metrics.sc_poly_eval.bas_f1'].idxmin()
    bot_idx = kw_df['metrics.sc_poly_eval.bas_f1'].idxmax()

    kw_cases = kw_df.loc[[bot_idx, top_idx]]
    kw_cases.to_dict('records')

    varied = kw_cases.varied_values(min_variations=2)
    constant = kw_cases.varied_values(max_variations=1)

    varied = ub.udict(varied).sorted_keys()
    constant = ub.udict(constant).sorted_keys()

    varied_text = ub.urepr(varied, nl=2)
    constant_text = ub.urepr(constant, nl=2)
    dpath = (ub.Path('~').expand() / 'tmp-kwc5').ensuredir()
    (dpath / 'constant_text.txt').write_text(constant_text)
    (dpath / 'varied_text.txt').write_text(varied_text)

    rich.print('varied = {}'.format(ub.urepr(varied, nl=2)))
    rich.print('constant = {}'.format(ub.urepr(constant, nl=2)))

    kw_records = kw_cases[['region_id', 'fpath', 'metrics.sc_poly_eval.bas_f1']].to_dict('records')
    print('kw_records = {}'.format(ub.urepr(kw_records, nl=2)))

    # Check for the best KW_C5001 example

    df2 = agg.table
    df2 = df2[df2['region_id'] == 'KW_C501']
    flags = df2['params.sc_pxl.package_fpath'].str.contains('Cropped2GSD_SC_bgrn_gnt_split6_V84_epoch17_step1548')
    df2 = df2[flags]
    max_idx = df2['metrics.sc_poly_eval.bas_f1'].idxmax()

    case = df2.loc[[max_idx]]
    case_record = case.to_dict('records')
    rich.print('case_record = {}'.format(ub.urepr(case_record, nl=2)))


subagg.output_dpath = agg.output_dpath / 'baseline'

rois = ['KR_R002', 'CN_C500', 'CO_C501', 'KW_C501']
plot_config = {}

plot_config = {
    'enabled': 1,
    'min_variations': 2,
    'plot_params': 1,
}
plot_config['vantage_points'] = [
    {
        'metric1': 'metrics.sc_poly_eval.sc_macro_f1',
        'metric2': 'metrics.sc_poly_eval.bas_faa_f1',
        'scale1': 'linear',
        'scale2': 'linear',
        'objective1': 'maximize',
        'objective2': 'maximize',
    },
    {
        'metric1': 'metrics.sc_poly_eval.bas_f1',
        'metric2': 'metrics.sc_poly_eval.bas_ffpa',
        'scale1': 'linear',
        'scale2': 'log',
        'objective1': 'maximize',
        'objective2': 'minimize',
    }
]

# 'NZ_R001', 'CH_R001']
baseline_rois = ['KR_R002', 'CN_C500', 'CO_C501', 'KW_C501']
roi_to_color = util_kwplot.Palette.coerce({
    'KR_R002': (230,  76, 230),
})
roi_to_color.update(baseline_rois)
roi_legend = roi_to_color.make_legend_img(dpi=300)

# roi_legend_fpath = (subagg.output_dpath / 'roi_legend.png')
# kwimage.imwrite(roi_legend_fpath, roi_legend)


plotter = subagg.build_plotter(rois=baseline_rois)
plotter.param_to_palette['region_id'] = roi_to_color
plotter.modifier.update(smart_global_helper.SMART_HELPER.LABEL_MAPPINGS)
plotter.plot_config['region_order'] = baseline_rois
plotter.plot_overviews()


if 1:
    all_metric_table = subagg._wip_build_per_region_variance_tables()
    col_order = ['bas_tp', 'bas_fp', 'bas_fn', 'bas_f1', 'bas_ffpa', 'sc_macro_f1', 'macro_f1_active', 'macro_f1_siteprep', 'count']
    metrics_table = all_metric_table
    metrics_table = util_pandas.DataFrame(metrics_table)
    index_order = ['KR_R002', 'CN_C500', 'CO_C501', 'KW_C501']
    metrics_table = metrics_table.reorder(index_order, axis=0, intersect=True)
    metrics_table = metrics_table[col_order]

    title = 'Small AC Baseline'
    fname = 'small_ac_baseline.png'
    bas_table_cols = [
        'region_id', 'bas_tp', 'bas_fp', 'bas_fn', 'bas_f1', 'bas_ffpa'
    ]
    bas_table = metrics_table.reset_index()
    bas_table.columns.name = None
    #bas_table = bas_table[bas_table_cols]
    bas_table_renamer = {
        'bas_tp': 'TP',
        'bas_fp': 'FP',
        'bas_fn': 'FN',
        'bas_f1': 'F1',
        'bas_ffpa': 'FFPA',
        'region_id': 'Region',
        'sc_macro_f1': 'AC-F1 (macro)',
        'macro_f1_siteprep': 'F1-SitePrep',
        'macro_f1_active': 'F1-Active',
    }
    bas_table_show = bas_table.rename(bas_table_renamer, axis=1)

    bas_table['region_id']

    col_formats = {c: 'concice_si_display' for c in bas_table_show.columns}
    df_style = util_kwplot.humanize_dataframe(bas_table_show, col_formats=col_formats)
    output_dpath = subagg.output_dpath
    fpath = output_dpath / fname
    print('output_dpath = {}'.format(ub.urepr(output_dpath, nl=1)))

    # df_style = bas_table_show.style.set_caption(title)
    util_kwplot.dataframe_table(df_style, fpath, title=title, dpi=200)
    from kwimage.im_draw import _draw_text_on_image_pil
    header = _draw_text_on_image_pil(None, title, fontsize=64)
    print(f'header.shape={header.shape}')
    ondisk = kwimage.imread(fpath)
    stacked = kwimage.stack_images([header, ondisk], axis=0)
    kwimage.imwrite(fpath, stacked)


index = subagg.table[subagg.table['region_id'] == 'KW_C501'].index
subsubagg = subagg.filterto(index=index)


if 1:
    # Full AGG
    #
    agg = eval_type_to_aggregator['sc_poly_eval']
    baseline_rois = ['KR_R002', 'CN_C500', 'CO_C501', 'KW_C501']
    import numpy as np
    flags = np.logical_or.reduce([(agg.table['region_id'] == r) for r in baseline_rois])
    big_agg = agg.filterto(index=agg.table.index[flags])

    plotter = big_agg.build_plotter()
    labels = ['KR_R002', 'CN_C500', 'CO_C501', 'KW_C501']
    roi_to_color = util_kwplot.Palette.coerce({
        'KR_R002': (230,  76, 230),
    })
    roi_to_color.update(labels)
    plotter.param_to_palette['region_id'] = roi_to_color
    plotter.plot_config['region_order'] = baseline_rois
    from watch.mlops import smart_global_helper
    plotter.modifier.update(smart_global_helper.SMART_HELPER.LABEL_MAPPINGS)
    plotter.plot_overviews()
    plotter.plot_resources()
