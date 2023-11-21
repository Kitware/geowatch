DVC_EXPT_DPATH=$(geowatch_dvc --tags='phase2_expt' --hardware=auto)



# Export on namek
python -m geowatch.mlops.aggregate \
    --pipeline=sc \
    --target "
        - $DVC_EXPT_DPATH/_ac_static_small_baseline_namek_v1
        - $DVC_EXPT_DPATH/_ac_static_small_baseline_v2
    " \
    --export_tables=True \
    --output_dpath="$DVC_EXPT_DPATH/mlops-2023-10-30/kit_namek" \
    --resource_report=0 \
    --eval_nodes="
        - sc_poly_eval
    " \
    --plot_params="
        enabled: 0
        stats_ranking: 0
        min_variations: 1
        params_of_interest:
            - params.sc_poly.thresh
    " \
    --stdout_report=0

# Export on toothbrush
python -m geowatch.mlops.aggregate \
    --pipeline=sc \
    --target "
        - $DVC_EXPT_DPATH/_ac_static_small_baseline_v1
        - $DVC_EXPT_DPATH/_ac_static_small_baseline_v2
    " \
    --export_tables=True \
    --output_dpath="/data/joncrall/dvc-repos/smart_expt_dvc/aggregate_results/mlops-2023-10-30/kit_toothbrush" \
    --resource_report=0 \
    --eval_nodes="
        - sc_poly_eval
    " \
    --plot_params="
        enabled: 0
        stats_ranking: 0
        min_variations: 1
        params_of_interest:
            - params.sc_poly.thresh
    " \
    --stdout_report=0
    # \
    # --query="(
    #     (df['params.sc_pxl.fixed_resolution'] == '8GSD') &
    #     (df['params.sc_poly.thresh'] == 0.07) &
    #     (df['params.sc_poly.site_score_thresh'] == 0) &
    #     df['params.sc_pxl.package_fpath'].str.contains('Drop4_tune_V30_8GSD_V3_epoch=2-step=17334.pt')
    # )"
    #--query="(df['param_hashid'] == 'fvlcyfgjmydd')"
        #(df['params.sc_poly.smoothing'] == None) &
    #--query="(df['param_hashid'] == 'fvlcyfgjmydd')"


python -m geowatch.mlops.aggregate \
    --pipeline=sc \
    --target "
        - $DVC_EXPT_DPATH/_ac_static_small_baseline_namek_v1
    " \
    --export_tables=True \
    --output_dpath="$DVC_EXPT_DPATH/aggregate_results/mlops-2023-10-30/_ac_static_small_baseline_v2" \
    --resource_report=0 \
    --eval_nodes="
        - sc_poly_eval
    " \
    --plot_params="
        enabled: 0
        stats_ranking: 0
        min_variations: 1
        params_of_interest:
            - params.sc_poly.thresh
    " \
    --stdout_report="
        top_k: 10
        per_group: 1
        macro_analysis: 0
        analyze: 0
        print_models: True
        reference_region: final
        concise: 1
        show_csv: 0
    "


# Aggregate all CSVs (including team csvs)
DVC_EXPT_DPATH=$(geowatch_dvc --tags='phase2_expt' --hardware=auto)
python -m geowatch.mlops.aggregate \
    --pipeline=sc \
    --target="$DVC_EXPT_DPATH/aggregate_results/mlops-2023-10-30/*/*.csv.zip" \
    --output_dpath="$DVC_EXPT_DPATH/aggregate_results/mlops-2023-10-30-summary" \
    --resource_report=0 \
    --eval_nodes="
        - sc_poly_eval
    " \
    --plot_params="
        enabled: 0
        stats_ranking: 0
        min_variations: 1
        params_of_interest:
            - params.sc_poly.thresh
    " \
    --stdout_report="
        top_k: 10
        per_group: 1
        macro_analysis: 0
        analyze: 0
        print_models: True
        reference_region: final
        concise: 1
        show_csv: 0
    " --embed \
    --rois "[KR_R002,]"


echo "Query:


from watch.utils import util_pandas
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
df = flags[df]


subagg = agg.filterto(index=df.index)
subagg.table['machine.sc_pxl.host'].unique()

subagg.index[['param_hashid', 'region_id']].sort_values('param_hashid').value_counts()

subagg.output_dpath = agg.output_dpath / 'baseline'

rois = ['KR_R002', 'CN_C500', 'CO_C501', 'KW_C501']
plot_config = {
}

from watch.mlops import aggregate_plots
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

import kwimage
import kwplot
from watch.utils import util_kwplot
labels = ['KR_R002', 'CN_C500', 'CO_C501', 'KW_C501', 'NZ_R001', 'CH_R001']
label_to_color = {
    'KR_R002': kwimage.Color.coerce((230,  76, 230,)).as01(),
    #'KW_R002': kwimage.Color.coerce('magenta').as01(),
}
roi_to_color = util_kwplot.color_new_labels(label_to_color, labels)
roi_legend = kwplot.make_legend_img(roi_to_color, dpi=300)
roi_legend_fpath = (subagg.output_dpath / 'roi_legend.png')
kwimage.imwrite(roi_legend_fpath, roi_legend)


plotter = subagg.build_plotter(rois=rois)
plotter.param_to_palette['region_id'] = roi_to_color
from watch.mlops import smart_global_helper
plotter.modifier.update(smart_global_helper.SMART_HELPER.LABEL_MAPPINGS)
plotter.plot_all()


all_metric_table = subagg._wip_build_per_region_variance_tables()
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


col_order = ['bas_tp', 'bas_fp', 'bas_fn', 'bas_f1', 'bas_ffpa', 'sc_macro_f1', 'macro_f1_active', 'macro_f1_siteprep', 'count']
metrics_table = all_metric_table
metrics_table = util_pandas.DataFrame(metrics_table)
index_order = ['KR_R002', 'CO_C001', 'KW_C001', 'CN_C000', 'CH_R001', 'NZ_R001']
metrics_table = metrics_table.reorder(index_order, axis=0, intersect=True)
metrics_table = metrics_table[col_order]

title = 'Small AC Baseline'

if 1:
    bas_table_cols = [
        'region_id', 'bas_tp', 'bas_fp', 'bas_fn', 'bas_f1', 'bas_ffpa'
    ]
    bas_table = metrics_table.reset_index()
    bas_table.columns.name = None
    #bas_table = bas_table[bas_table_cols]
    bas_table_show = bas_table.rename(bas_table_renamer, axis=1)
    col_formats = {c: 'concice_si_display' for c in bas_table_show.columns}
    df_style = util_kwplot.humanize_dataframe(bas_table_show, col_formats=col_formats)
    output_dpath = subagg.output_dpath
    fpath = output_dpath / f'{title}.png'

    # df_style = bas_table_show.style.set_caption(title)
    util_kwplot.dataframe_table(df_style, fpath, title=title)
    from kwimage.im_draw import _draw_text_on_image_pil
    header = _draw_text_on_image_pil(None, title)
    import kwplot
    kwplot.autompl()
    kwplot.imshow(header)
    import kwimage
    ondisk = kwimage.imread(fpath)
    stacked = kwimage.stack_images([header, ondisk], axis=0)
    kwimage.imwrite(fpath, stacked)



index = subagg.table[subagg.table['region_id'] == 'KW_C501'].index
subsubagg = subagg.filterto(index=index)


if 1:
    # Full AGG
    #
    agg = eval_type_to_aggregator['sc_poly_eval']
    rois = ['CN_C500', 'CO_C501', 'KR_R002', 'KW_C501']
    plotter = agg.build_plotter()
    plotter.param_to_palette['region_id'] = roi_to_color
    plotter.plot_config['plot_params'] = 0
    from watch.mlops import smart_global_helper
    plotter.modifier.update(smart_global_helper.SMART_HELPER.LABEL_MAPPINGS)
    plotter.plot_all()
    plotter.plot_resources()
"
