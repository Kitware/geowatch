DVC_DATA_DPATH=$(geowatch_dvc --tags='phase2_data' --hardware=hdd)
DVC_EXPT_DPATH=$(geowatch_dvc --tags='phase2_expt' --hardware=auto)

python -m geowatch.cli.prepare_teamfeats \
    --base_fpath="$DVC_DATA_DPATH/Drop6/imganns-*BH_R001.kwcoco.zip" \
    --expt_dpath="$DVC_EXPT_DPATH" \
    --with_landcover=0 \
    --with_materials=0 \
    --with_invariants=0 \
    --with_invariants2=1 \
    --with_cold=0 \
    --with_depth=0 \
    --do_splits=0 \
    --skip_existing=1 \
    --invariant_resolution="30GSD" \
    --gres=3, --workers=1 --backend=tmux --run=1

python -m geowatch.mlops.schedule_evaluation --params="
    matrix:
        bas_pxl.package_fpath:
            - $DVC_EXPT_DPATH/models/fusion/Drop4-BAS/packages/Drop4_TuneV323_BAS_30GSD_BGRNSH_V2/package_epoch0_step41.pt.pt
            - $DVC_EXPT_DPATH/models/fusion/Drop4-BAS/packages/Drop4_BAS_15GSD_BGRNSH_invar_V8/Drop4_BAS_15GSD_BGRNSH_invar_V8_epoch=16-step=8704.pt
            #- $DVC_EXPT_DPATH/models/fusion/Drop4-BAS/packages/Drop4_BAS_2022_12_10GSD_BGRN_V11/Drop4_BAS_2022_12_10GSD_BGRN_V11_v0_epoch0_step108.pt
            #- $DVC_EXPT_DPATH/models/fusion/Drop4-BAS/packages/Drop4_BAS_2022_12_10GSD_BGRN_V11/Drop4_BAS_2022_12_10GSD_BGRN_V11_v0_epoch100_step51712.pt
        bas_pxl.test_dataset:
            #- $DVC_DATA_DPATH/Drop6/combo_imganns-CH_R001_I.kwcoco.json
            - /home/joncrall/remote/toothbrush/data/dvc-repos/smart_data_dvc-ssd/Drop6/combo_imganns-BH_R001_I2.kwcoco.json
            - /home/joncrall/remote/toothbrush/data/dvc-repos/smart_data_dvc-ssd/Drop6/combo_imganns-BR_R001_I2.kwcoco.json
            - /home/joncrall/remote/toothbrush/data/dvc-repos/smart_data_dvc-ssd/Drop6/combo_imganns-CH_R001_I2.kwcoco.json
            - /home/joncrall/remote/toothbrush/data/dvc-repos/smart_data_dvc-ssd/Drop6/combo_imganns-KR_R001_I2.kwcoco.json
            - /home/joncrall/remote/toothbrush/data/dvc-repos/smart_data_dvc-ssd/Drop6/combo_imganns-LT_R001_I2.kwcoco.json
            - /home/joncrall/remote/toothbrush/data/dvc-repos/smart_data_dvc-ssd/Drop6/combo_imganns-NZ_R001_I2.kwcoco.json
            - /home/joncrall/remote/toothbrush/data/dvc-repos/smart_data_dvc-ssd/Drop6/combo_imganns-US_R001_I2.kwcoco.json
        bas_pxl.chip_overlap: 0.3
        bas_pxl.chip_dims:
            - auto
        bas_pxl.time_span:
            - auto
        bas_pxl.time_sampling:
            - auto
        bas_poly.thresh:
            - 0.17
        bas_poly.polygon_simplify_tolerance:
            - 1
        bas_poly.agg_fn:
            - probs
        bas_poly.resolution:
            - null
        bas_poly.moving_window_size:
            - null
            - 100
        bas_poly.min_area_sqkm:
            - 0.0072
        bas_poly.max_area_sqkm:
            - 8
        bas_poly_eval.true_site_dpath: $DVC_DATA_DPATH/drop6/annotations/site_models
        bas_poly_eval.true_region_dpath: $DVC_DATA_DPATH/drop6/annotations/region_models
        bas_pxl.enabled: 1
        bas_poly.enabled: 1
        bas_poly_eval.enabled: 1
        bas_pxl_eval.enabled: 1
        bas_poly_viz.enabled: 0
    " \
    --root_dpath="$DVC_EXPT_DPATH/_check_ch" \
    --devices="0,1" --queue_size=2 --print_commands=0 \
    --backend=tmux --queue_name "_check_ch" \
    --pipeline=bas --skip_existing=1 \
    --run=1

ls $DVC_DATA_DPATH/Drop6/combo_imganns-*_R001_I2.kwcoco.json


    # viz_dpath argument can be specified to visualize the algorithm details.
DVC_DATA_DPATH=$(geowatch_dvc --tags='phase2_data' --hardware=auto)
DVC_EXPT_DPATH=$(geowatch_dvc --tags='phase2_expt' --hardware=auto)
python -m geowatch reproject_annotations \
    --src "$DVC_EXPT_DPATH"/_check_ch/pred/flat/bas_pxl/bas_pxl_id_ac952ddc/pred.kwcoco.zip \
    --dst "$DVC_EXPT_DPATH"/_check_ch/pred/flat/bas_pxl/bas_pxl_id_ac952ddc/pred_with_truth.kwcoco.zip \
    --workers=4 \
    --site_models="$DVC_DATA_DPATH/annotations/drop6/site_models/*.geojson"

geowatch visualize "$DVC_EXPT_DPATH"/_check_ch/pred/flat/bas_pxl/bas_pxl_id_ac952ddc/pred_with_truth.kwcoco.zip --smart



# viz_dpath argument can be specified to visualize the algorithm details.
DVC_DATA_DPATH=$(geowatch_dvc --tags='phase2_data' --hardware=auto)
DVC_EXPT_DPATH=$(geowatch_dvc --tags='phase2_expt' --hardware=auto)
python -m geowatch reproject_annotations \
    --src "$DVC_EXPT_DPATH"/_check_ch/pred/flat/bas_pxl/bas_pxl_id_790d666a/pred.kwcoco.zip \
    --dst "$DVC_EXPT_DPATH"/_check_ch/pred/flat/bas_pxl/bas_pxl_id_790d666a/pred_with_truth.kwcoco.zip \
    --workers=4 \
    --site_models="$DVC_DATA_DPATH/annotations/drop6/site_models/*.geojson"

geowatch visualize "$DVC_EXPT_DPATH"/_check_ch/pred/flat/bas_pxl/bas_pxl_id_790d666a/pred_with_truth.kwcoco.zip --smart

#python -m geowatch.cli.run_tracker \
#    /home/joncrall/remote/toothbrush/data/dvc-repos/smart_expt_dvc/_check_ch/pred/flat/bas_pxl/bas_pxl_id_790d666a/pred.kwcoco.zip \
#    --default_track_fn saliency_heatmaps \
#    --track_kwargs '{"agg_fn": "probs", "thresh": 0.17, "polygon_simplify_tolerance": 1, "resolution": "auto", "moving_window_size": 100, "min_area_sqkm": 0.0072, "max_area_sqkm": 8}' \
#    --clear_annots \
#    --site_summary None \
#    --out_site_summaries_fpath /home/joncrall/remote/toothbrush/data/dvc-repos/smart_expt_dvc/_check_ch/pred/flat/bas_poly/bas_poly_id_0836eaa2/site_summaries_manifest.json \
#    --out_site_summaries_dir /home/joncrall/remote/toothbrush/data/dvc-repos/smart_expt_dvc/_check_ch/pred/flat/bas_poly/bas_poly_id_0836eaa2/site_summaries \
#    --out_sites_fpath /home/joncrall/remote/toothbrush/data/dvc-repos/smart_expt_dvc/_check_ch/pred/flat/bas_poly/bas_poly_id_0836eaa2/sites_manifest.json \
#    --out_sites_dir /home/joncrall/remote/toothbrush/data/dvc-repos/smart_expt_dvc/_check_ch/pred/flat/bas_poly/bas_poly_id_0836eaa2/sites \
#    --out_kwcoco /home/joncrall/remote/toothbrush/data/dvc-repos/smart_expt_dvc/_check_ch/pred/flat/bas_poly/bas_poly_id_0836eaa2/poly.kwcoco.zip


DVC_DATA_DPATH=$(geowatch_dvc --tags='phase2_data' --hardware=auto)
cmd_queue new "my_spectra_queue"
cmd_queue submit 'my_spectra_queue' --command "geowatch spectra --channels 'red|green|blue' --valid_range=0:10000 '$DVC_DATA_DPATH'/Drop6/imgonly-AE_R001.kwcoco.json --workers=4 --dst spectra-AE_R001.png"
cmd_queue submit 'my_spectra_queue' --command "geowatch spectra --channels 'red|green|blue' --valid_range=0:10000 '$DVC_DATA_DPATH'/Drop6/imgonly-BH_R001.kwcoco.json --workers=4 --dst spectra-BH_R001.png"
cmd_queue submit 'my_spectra_queue' --command "geowatch spectra --channels 'red|green|blue' --valid_range=0:10000 '$DVC_DATA_DPATH'/Drop6/imgonly-BR_R001.kwcoco.json --workers=4 --dst spectra-BR_R001.png"
cmd_queue submit 'my_spectra_queue' --command "geowatch spectra --channels 'red|green|blue' --valid_range=0:10000 '$DVC_DATA_DPATH'/Drop6/imgonly-BR_R002.kwcoco.json --workers=4 --dst spectra-BR_R002.png"
cmd_queue submit 'my_spectra_queue' --command "geowatch spectra --channels 'red|green|blue' --valid_range=0:10000 '$DVC_DATA_DPATH'/Drop6/imgonly-BR_R004.kwcoco.json --workers=4 --dst spectra-BR_R004.png"
cmd_queue submit 'my_spectra_queue' --command "geowatch spectra --channels 'red|green|blue' --valid_range=0:10000 '$DVC_DATA_DPATH'/Drop6/imgonly-BR_R005.kwcoco.json --workers=4 --dst spectra-BR_R005.png"
cmd_queue submit 'my_spectra_queue' --command "geowatch spectra --channels 'red|green|blue' --valid_range=0:10000 '$DVC_DATA_DPATH'/Drop6/imgonly-CH_R001.kwcoco.json --workers=4 --dst spectra-CH_R001.png"
cmd_queue submit 'my_spectra_queue' --command "geowatch spectra --channels 'red|green|blue' --valid_range=0:10000 '$DVC_DATA_DPATH'/Drop6/imgonly-KR_R001.kwcoco.json --workers=4 --dst spectra-KR_R001.png"
cmd_queue submit 'my_spectra_queue' --command "geowatch spectra --channels 'red|green|blue' --valid_range=0:10000 '$DVC_DATA_DPATH'/Drop6/imgonly-KR_R002.kwcoco.json --workers=4 --dst spectra-KR_R002.png"
cmd_queue submit 'my_spectra_queue' --command "geowatch spectra --channels 'red|green|blue' --valid_range=0:10000 '$DVC_DATA_DPATH'/Drop6/imgonly-LT_R001.kwcoco.json --workers=4 --dst spectra-LT_R001.png"
cmd_queue submit 'my_spectra_queue' --command "geowatch spectra --channels 'red|green|blue' --valid_range=0:10000 '$DVC_DATA_DPATH'/Drop6/imgonly-NZ_R001.kwcoco.json --workers=4 --dst spectra-NZ_R001.png"
cmd_queue submit 'my_spectra_queue' --command "geowatch spectra --channels 'red|green|blue' --valid_range=0:10000 '$DVC_DATA_DPATH'/Drop6/imgonly-PE_R001.kwcoco.json --workers=4 --dst spectra-PE_R001.png"
cmd_queue submit 'my_spectra_queue' --command "geowatch spectra --channels 'red|green|blue' --valid_range=0:10000 '$DVC_DATA_DPATH'/Drop6/imgonly-US_R001.kwcoco.json --workers=4 --dst spectra-US_R001.png"
cmd_queue submit 'my_spectra_queue' --command "geowatch spectra --channels 'red|green|blue' --valid_range=0:10000 '$DVC_DATA_DPATH'/Drop6/imgonly-US_R004.kwcoco.json --workers=4 --dst spectra-US_R004.png"
cmd_queue submit 'my_spectra_queue' --command "geowatch spectra --channels 'red|green|blue' --valid_range=0:10000 '$DVC_DATA_DPATH'/Drop6/imgonly-US_R005.kwcoco.json --workers=4 --dst spectra-US_R005.png"
cmd_queue submit 'my_spectra_queue' --command "geowatch spectra --channels 'red|green|blue' --valid_range=0:10000 '$DVC_DATA_DPATH'/Drop6/imgonly-US_R006.kwcoco.json --workers=4 --dst spectra-US_R006.png"
cmd_queue submit 'my_spectra_queue' --command "geowatch spectra --channels 'red|green|blue' --valid_range=0:10000 '$DVC_DATA_DPATH'/Drop6/imgonly-US_R007.kwcoco.json --workers=4 --dst spectra-US_R007.png"
cmd_queue show "my_spectra_queue"
cmd_queue run "my_spectra_queue" --backend=tmux --workers=8


# viz_dpath argument can be specified to visualize the algorithm details.
DVC_DATA_DPATH=$(geowatch_dvc --tags='phase2_data' --hardware=auto)
DVC_EXPT_DPATH=$(geowatch_dvc --tags='phase2_expt' --hardware=auto)
python -m geowatch reproject_annotations \
    --src "$DVC_EXPT_DPATH"/_check_ch/pred/flat/bas_pxl/bas_pxl_id_8e3baad5/pred.kwcoco.zip \
    --dst "$DVC_EXPT_DPATH"/_check_ch/pred/flat/bas_pxl/bas_pxl_id_8e3baad5/pred_with_truth_KR_R001.kwcoco.zip \
    --workers=4 \
    --site_models="$DVC_DATA_DPATH/annotations/drop6/site_models/*.geojson"

geowatch visualize "$DVC_EXPT_DPATH"/_check_ch/pred/flat/bas_pxl/bas_pxl_id_8e3baad5/pred_with_truth_KR_R001.kwcoco.zip --smart


geowatch spectra --channels 'salient' --valid_range=0:1 \
    "$DVC_EXPT_DPATH"/_check_ch/pred/flat/bas_pxl/bas_pxl_id_8e3baad5/pred_with_truth_KR_R001.kwcoco.zip \
    --workers=4 --dst "$DVC_EXPT_DPATH"/_check_ch/pred/flat/bas_pxl/bas_pxl_id_8e3baad5/spectra-KR_R001.png  --title="nov7 model"

ls /home/joncrall/remote/toothbrush/data/dvc-repos/smart_expt_dvc/_check_ch/pred/flat/bas_pxl/*/pred.kwcoco.zip
DVC_DATA_DPATH=$(geowatch_dvc --tags='phase2_data' --hardware=auto)
DVC_EXPT_DPATH=$(geowatch_dvc --tags='phase2_expt' --hardware=auto)
python -c "
import ubelt as ub
import json
node_dpath = ub.Path('/home/joncrall/remote/toothbrush/data/dvc-repos/smart_expt_dvc/_check_ch/pred/flat/bas_pxl')
print(node_dpath.exists())
for pred_fpath in list(node_dpath.glob('*/pred.kwcoco.zip')):
    dpath = pred_fpath.parent
    print(dpath)
    json_fpath = (dpath / 'job_config.json')
    config = json.loads(json_fpath.read_text())
    print(ub.urepr(config))

    model_name = ub.Path(config['bas_pxl.package_fpath']).name.split('.')[0]
    data_name = ub.Path(config['bas_pxl.test_dataset']).name.split('.')[0]

    pred_with_truth_fpath = dpath / ('pred_with_truth_' + data_name + '.kwcoco.zip')

    BACKSLASH = chr(92)

    analysis_script_text = ub.codeblock(
        f'''

        ##########
        # ANALYSIS
        ##########
        
        # viz_dpath argument can be specified to visualize the algorithm details.
        #geowatch reproject {BACKSLASH}
        #    --src {pred_fpath} {BACKSLASH}
        #    --dst {pred_with_truth_fpath} {BACKSLASH}
        #    --workers=4 {BACKSLASH}
        #    --site_models=$DVC_DATA_DPATH/annotations/drop6/site_models/*.geojson

        geowatch spectra --channels 'salient' --valid_range=0:1 {BACKSLASH}
            {pred_with_truth_fpath} {BACKSLASH}
            --workers=4 --dst {dpath}/spectra-salient-{data_name}.png  --title='{model_name}'

        #geowatch spectra --channels 'red|green|blue|nir' --valid_range=0:10000 {BACKSLASH}
        #    {pred_with_truth_fpath} {BACKSLASH}
        #    --workers=4 --dst {dpath}/spectra-rgb-{data_name}.png  --title='{model_name}'

        #geowatch visualize {BACKSLASH}
        #    '{pred_with_truth_fpath}' --smart
        ''')
    analysis_fpath = dpath / f'analysis1_{data_name}_{model_name}.sh'
    analysis_fpath.write_text(chr(10) + chr(10) + analysis_script_text)
"
ls /home/joncrall/remote/toothbrush/data/dvc-repos/smart_expt_dvc/_check_ch/pred/flat/bas_pxl/*/analysis1_*.sh


 source  /home/joncrall/remote/toothbrush/data/dvc-repos/smart_expt_dvc/_check_ch/pred/flat/bas_pxl/bas_pxl_id_001ee987/analysis1_combo_imganns-CH_R001_I2_package_epoch0_step41.sh
 source '/home/joncrall/remote/toothbrush/data/dvc-repos/smart_expt_dvc/_check_ch/pred/flat/bas_pxl/bas_pxl_id_19175383/analysis1_combo_imganns-CH_R001_I2_Drop4_BAS_15GSD_BGRNSH_invar_V8_epoch=16-step=8704.sh'
 source '/home/joncrall/remote/toothbrush/data/dvc-repos/smart_expt_dvc/_check_ch/pred/flat/bas_pxl/bas_pxl_id_219ac3d7/analysis1_combo_imganns-NZ_R001_I2_Drop4_BAS_15GSD_BGRNSH_invar_V8_epoch=16-step=8704.sh'
 source  /home/joncrall/remote/toothbrush/data/dvc-repos/smart_expt_dvc/_check_ch/pred/flat/bas_pxl/bas_pxl_id_3551e9be/analysis1_combo_imganns-BR_R001_I2_package_epoch0_step41.sh
 source  /home/joncrall/remote/toothbrush/data/dvc-repos/smart_expt_dvc/_check_ch/pred/flat/bas_pxl/bas_pxl_id_4cd9880a/analysis1_combo_imganns-BH_R001_I2_package_epoch0_step41.sh
 source '/home/joncrall/remote/toothbrush/data/dvc-repos/smart_expt_dvc/_check_ch/pred/flat/bas_pxl/bas_pxl_id_61b189d1/analysis1_combo_imganns-KR_R001_I2_Drop4_BAS_15GSD_BGRNSH_invar_V8_epoch=16-step=8704.sh'
 source '/home/joncrall/remote/toothbrush/data/dvc-repos/smart_expt_dvc/_check_ch/pred/flat/bas_pxl/bas_pxl_id_734c8270/analysis1_combo_imganns-US_R001_I2_Drop4_BAS_15GSD_BGRNSH_invar_V8_epoch=16-step=8704.sh'
 source '/home/joncrall/remote/toothbrush/data/dvc-repos/smart_expt_dvc/_check_ch/pred/flat/bas_pxl/bas_pxl_id_75adeafa/analysis1_combo_imganns-BR_R001_I2_Drop4_BAS_15GSD_BGRNSH_invar_V8_epoch=16-step=8704.sh'
 source '/home/joncrall/remote/toothbrush/data/dvc-repos/smart_expt_dvc/_check_ch/pred/flat/bas_pxl/bas_pxl_id_77e5b11b/analysis1_combo_imganns-BH_R001_I2_Drop4_BAS_15GSD_BGRNSH_invar_V8_epoch=16-step=8704.sh'
 source '/home/joncrall/remote/toothbrush/data/dvc-repos/smart_expt_dvc/_check_ch/pred/flat/bas_pxl/bas_pxl_id_790d666a/analysis1_combo_imganns-CH_R001_I_Drop4_BAS_15GSD_BGRNSH_invar_V8_epoch=16-step=8704.sh'
 source '/home/joncrall/remote/toothbrush/data/dvc-repos/smart_expt_dvc/_check_ch/pred/flat/bas_pxl/bas_pxl_id_7d426947/analysis1_combo_imganns-KR_R001_I2_Drop4_BAS_15GSD_BGRNSH_invar_V8_epoch=16-step=8704.sh'
 source  /home/joncrall/remote/toothbrush/data/dvc-repos/smart_expt_dvc/_check_ch/pred/flat/bas_pxl/bas_pxl_id_86f05acd/analysis1_combo_imganns-KR_R002_I2_package_epoch0_step41.sh
 source  /home/joncrall/remote/toothbrush/data/dvc-repos/smart_expt_dvc/_check_ch/pred/flat/bas_pxl/bas_pxl_id_8e3baad5/analysis1_combo_imganns-KR_R001_I2_package_epoch0_step41.sh
 source  /home/joncrall/remote/toothbrush/data/dvc-repos/smart_expt_dvc/_check_ch/pred/flat/bas_pxl/bas_pxl_id_9cbf8b96/analysis1_combo_imganns-LT_R001_I2_package_epoch0_step41.sh
 source /home/joncrall/remote/toothbrush/data/dvc-repos/smart_expt_dvc/_check_ch/pred/flat/bas_pxl/bas_pxl_id_a2a2ce87/analysis1_combo_imganns-KR_R001_I2_package_epoch0_step41.sh

 source /home/joncrall/remote/toothbrush/data/dvc-repos/smart_expt_dvc/_check_ch/pred/flat/bas_pxl/bas_pxl_id_ac952ddc/analysis1_combo_imganns-CH_R001_I_package_epoch0_step41.sh
source '/home/joncrall/remote/toothbrush/data/dvc-repos/smart_expt_dvc/_check_ch/pred/flat/bas_pxl/bas_pxl_id_bc1dd541/analysis1_combo_imganns-LT_R001_I2_Drop4_BAS_15GSD_BGRNSH_invar_V8_epoch=16-step=8704.sh'
 source /home/joncrall/remote/toothbrush/data/dvc-repos/smart_expt_dvc/_check_ch/pred/flat/bas_pxl/bas_pxl_id_e00a9eaa/analysis1_combo_imganns-NZ_R001_I2_package_epoch0_step41.sh
 source /home/joncrall/remote/toothbrush/data/dvc-repos/smart_expt_dvc/_check_ch/pred/flat/bas_pxl/bas_pxl_id_f9ec5f05/analysis1_combo_imganns-US_R001_I2_package_epoch0_step41.sh



# viz_dpath argument can be specified to visualize the algorithm details.
DVC_DATA_DPATH=$(geowatch_dvc --tags='phase2_data' --hardware=auto)
DVC_EXPT_DPATH=$(geowatch_dvc --tags='phase2_expt' --hardware=auto)
python -m geowatch reproject_annotations \
    --src "$DVC_EXPT_DPATH"/_check_ch/pred/flat/bas_pxl/bas_pxl_id_8e3baad5/pred.kwcoco.zip \
    --dst "$DVC_EXPT_DPATH"/_check_ch/pred/flat/bas_pxl/bas_pxl_id_8e3baad5/pred_with_truth_KR_R001.kwcoco.zip \
    --workers=4 \
    --site_models="$DVC_DATA_DPATH/annotations/drop6/site_models/*.geojson"

geowatch visualize "$DVC_EXPT_DPATH"/_check_ch/pred/flat/bas_pxl/bas_pxl_id_8e3baad5/pred_with_truth_KR_R001.kwcoco.zip --smart


geowatch spectra --channels 'salient' --valid_range=0:1 \
    "$DVC_EXPT_DPATH"/_check_ch/pred/flat/bas_pxl/bas_pxl_id_8e3baad5/pred_with_truth_KR_R001.kwcoco.zip \
    --workers=4 --dst "$DVC_EXPT_DPATH"/_check_ch/pred/flat/bas_pxl/bas_pxl_id_8e3baad5/spectra-KR_R001.png  --title="nov7 model"


python -c "
import ubelt as ub
import json
node_dpath = ub.Path('/home/joncrall/remote/toothbrush/data/dvc-repos/smart_expt_dvc/_check_ch/pred/flat/bas_pxl')
print(node_dpath.exists())
import kwimage

rows = []

for pred_fpath in list(node_dpath.glob('*/pred.kwcoco.zip')):
    dpath = pred_fpath.parent
    print(dpath)
    json_fpath = (dpath / 'job_config.json')
    config = json.loads(json_fpath.read_text())
    print(ub.urepr(config))

    model_name = ub.Path(config['bas_pxl.package_fpath']).name.split('.')[0]
    data_name = ub.Path(config['bas_pxl.test_dataset']).name.split('.')[0]

    pred_with_truth_fpath = dpath / ('pred_with_truth_' + data_name + '.kwcoco.zip')

    BACKSLASH = chr(92)

    salient_fpath = f'{dpath}/spectra-salient-{data_name}.png'
    rgb_fpath = f'{dpath}/spectra-rgb-{data_name}.png'

    rows.append({
        'data_name': data_name.split('-')[1],
        'model_name': model_name,
        'salient_fpath': salient_fpath,
        'rgb_fpath': rgb_fpath,
    })

import pandas as pd
df = pd.DataFrame(rows)
df.sort_values(['data_name', 'model_name'])

NL = chr(10)

im_rows = []
for _, group in df.groupby('data_name'):
    im_cols = []
    group = group.sort_values(['data_name', 'model_name'])
    for row in group.to_dict('records'):
        if len(im_cols) > 1:
            continue
        fpath1 = ub.Path(row['salient_fpath'])
        fpath2 = ub.Path(row['rgb_fpath'])
        if fpath1.exists():
            im1 = kwimage.imread(fpath1)
        else:
            im1 = np.zeros((806, 1075, 3), dtype=np.uint8)
        if fpath2.exists():
            im2 = kwimage.imread(fpath2)
        else:
            im2 = np.zeros((806, 1075, 3), dtype=np.uint8)
        im3 = kwimage.stack_images([im1, im2], axis=0)
        im3 = kwimage.draw_header_text(im3, row['data_name'] + f' {NL} ' + row['model_name'])
        im_cols.append(im3)
    im_col = kwimage.stack_images(im_cols, axis=1)

    im_rows.append(im_col)

final = kwimage.stack_images(im_rows, axis=0)
import kwplot
kwplot.plt.ion()
kwplot.imshow(final)

kwimage.imwrite('final.png', final)

"


python -m geowatch.tasks.invariants.predict --input_kwcoco /home/local/KHQ/jon.crall/remote/horologic/data/dvc-repos/smart_data_dvc/Drop6/imganns-BH_R001.kwcoco.zip --output_kwcoco /home/local/KHQ/jon.crall/remote/horologic/data/dvc-repos/smart_data_dvc/Drop6/imganns-BH_R001_uky_invariants.kwcoco.json --pretext_package_path /home/local/KHQ/jon.crall/remote/horologic/data/dvc-repos/smart_expt_dvc/models/uky/uky_invariants_2022_12_17/TA1_pretext_model/pretext_package.pt --pca_projection_path /home/local/KHQ/jon.crall/remote/horologic/data/dvc-repos/smart_expt_dvc/models/uky/uky_invariants_2022_03_21/pretext_model/pretext_pca_104.pt --input_resolution=30GSD --window_resolution=30GSD --patch_size=256 --do_pca 0 --patch_overlap=0.3 --num_workers=2 --write_workers 0 --tasks before_after pretext
