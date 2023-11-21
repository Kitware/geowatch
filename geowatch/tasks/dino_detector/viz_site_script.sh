#!/bin/bash

#EVAL_FPATH=/home/joncrall/remote/toothbrush/data/dvc-repos/smart_expt_dvc/_mlops_eval10_baseline/eval/flat/sv_poly_eval/sv_poly_eval_id_d6b58698/poly_eval.json
EVAL_DPATH=/home/joncrall/remote/toothbrush/data/dvc-repos/smart_expt_dvc/_mlops_eval10_baseline/eval/flat/sv_poly_eval/sv_poly_eval_id_d6b58698

EVAL_DPATH=/home/joncrall/remote/toothbrush/data/dvc-repos/smart_expt_dvc/_toothbrush_split6_landcover_MeanYear10GSD-V2/_custom/eval_links/KR_R002_sv_poly_eval_id_e865e066
FILTERED_DPATH=$(echo "$EVAL_DPATH"/.pred/sv_dino_filter/*)
BOXES_DPATH=$(echo "$FILTERED_DPATH"/.pred/sv_dino_boxes/*)
SVCROP_DPATH=$(echo "$BOXES_DPATH"/.pred/sv_crop/*)
BAS_POLY_DPATH=$(echo "$SVCROP_DPATH"/.pred/bas_poly/*/)

echo "
FILTERED_DPATH=$FILTERED_DPATH
BOXES_DPATH=$BOXES_DPATH
SVCROP_DPATH=$SVCROP_DPATH
BAS_POLY_DPATH=$BAS_POLY_DPATH
"

FILTERED_REGION_ID=$(python -c "import pathlib; print(list(pathlib.Path('$BAS_POLY_DPATH/site_summaries').glob('*'))[0].stem)")
echo "FILTERED_REGION_ID = $FILTERED_REGION_ID"


BOXES_KWCOCO_FPATH=$BOXES_DPATH/pred_boxes.kwcoco.zip
#FILTERED_REGION_FPATH=$FILTERED_DPATH/out_region.geojson
INPUT_REGION_FPATH=$BAS_POLY_DPATH/site_summaries_manifest.json

# /home/joncrall/remote/toothbrush/data/dvc-repos/smart_expt_dvc/_mlops_eval10_baseline/eval/flat/sv_poly_eval/sv_poly_eval_id_d235a5fd/poly_eval.json
#NODE_DPATH=/home/joncrall/remote/toothbrush/data/dvc-repos/smart_expt_dvc/_mlops_eval10_baseline/pred/flat/buildings/buildings_id_fd298dba/
DVC_DATA_DPATH=$(geowatch_dvc --tags='phase2_data' --hardware=auto)

echo "
FILTERED_REGION_ID=$FILTERED_REGION_ID
DVC_DATA_DPATH=$DVC_DATA_DPATH
INPUT_REGION_FPATH=$INPUT_REGION_FPATH
BOXES_KWCOCO_FPATH=$BOXES_KWCOCO_FPATH
"

ANALYSIS_DPATH=$EVAL_DPATH/analysis

mkdir -p $ANALYSIS_DPATH


### SV DINO BOX VIZ

geowatch reproject \
        --src "$BOXES_KWCOCO_FPATH" \
        --dst "$ANALYSIS_DPATH/pred_boxes_with_polys.kwcoco.zip" \
        --region_models "$INPUT_REGION_FPATH" \
        --status_to_catname="{system_confirmed: positive}" \
        --role=pred_poly \
        --validate_checks=False \
        --clear_existing=False

geowatch reproject \
        --src "$ANALYSIS_DPATH/pred_boxes_with_polys.kwcoco.zip" \
        --dst "$ANALYSIS_DPATH/pred_and_truth2.kwcoco.zip" \
        --region_models="$DVC_DATA_DPATH/annotations/drop6/region_models/${FILTERED_REGION_ID}.geojson" \
        --site_models="$DVC_DATA_DPATH/annotations/drop6/site_models/${FILTERED_REGION_ID}_*.geojson" \
        --status_to_catname="{system_confirmed: positive}" \
        --validate_checks=False \
        --role=truth \
        --clear_existing=False

geowatch visualize --smart 1 \
    --ann_score_thresh 0.5 \
    --draw_labels False \
    --alpha 0.5 \
    --src $ANALYSIS_DPATH/pred_and_truth2.kwcoco.zip \
    --viz_dpath $ANALYSIS_DPATH/_vizme

python ~/code/watch/dev/wip/grid_sitevali_crops.py \
    --sub=_anns \
    $ANALYSIS_DPATH/_vizme


### BAS FEATURE VIZ
BAS_PXL_DPATH=$(echo "$BAS_POLY_DPATH"/.pred/bas_pxl/*/)

BAS_PXL_FPATH=$BAS_PXL_DPATH/pred.kwcoco.zip
POLY_PRED_FPATH=$BAS_POLY_DPATH/poly.kwcoco.zip

geowatch stats "$BAS_PXL_FPATH"

geowatch visualize "$BAS_PXL_FPATH" --smart \
    --viz_dpath $ANALYSIS_DPATH/_viz_bas_heatmaps

geowatch visualize "$BAS_PXL_FPATH" --smart \
    --viz_dpath $ANALYSIS_DPATH/_viz_bas_landcover_hidden \
    --include_sensors=S2 \
    --channels="red|green|blue,pan,landcover_hidden.0:3,landcover_hidden.3:6,landcover_hidden.6:9"

geowatch visualize "$BAS_PXL_FPATH" --smart \
    --viz_dpath $ANALYSIS_DPATH/_viz_bas_landcover_output \
    --include_sensors=S2 \
    --channels="red|green|blue,pan,impervious|forest|water,barren|field|water"

geowatch visualize "$BAS_PXL_FPATH" --smart \
    --viz_dpath $ANALYSIS_DPATH/_viz_bas_invariants \
    --include_sensors=S2 \
    --channels="red|green|blue,pan,invariants.0:3,invariants.3:6,invariants.6:9"


geowatch reproject \
        --src "$POLY_PRED_FPATH" \
        --dst "$ANALYSIS_DPATH/bas_pred_and_truth.kwcoco.zip" \
        --region_models="$DVC_DATA_DPATH/annotations/drop6/region_models/${FILTERED_REGION_ID}.geojson" \
        --site_models="$DVC_DATA_DPATH/annotations/drop6/site_models/${FILTERED_REGION_ID}_*.geojson" \
        --status_to_catname="{system_confirmed: positive}" \
        --validate_checks=False \
        --role=truth \
        --clear_existing=False

geowatch visualize --smart 1 \
    --draw_labels False \
    --alpha 0.5 \
    --src "$ANALYSIS_DPATH"/bas_pred_and_truth.kwcoco.zip \
    --only_boxes=True \
    --viz_dpath "$ANALYSIS_DPATH"/_viz_bas_pred_and_truth


python -m geowatch.mlops.confusor_analysis \
    --detections_fpath "$EVAL_DPATH"/"$FILTERED_REGION_ID"/overall/bas/detections_tau=0.2_rho=0.5_min_area=0.csv \
    --proposals_fpath "$EVAL_DPATH"/"$FILTERED_REGION_ID"/overall/bas/proposals_tau=0.2_rho=0.5_min_area=0.csv \
    --bas_metric_dpath "$EVAL_DPATH"/"$FILTERED_REGION_ID"/overall/bas \
    --region_id "$FILTERED_REGION_ID" \
    --src_kwcoco "$BAS_PXL_FPATH" \
    --dst_kwcoco "$ANALYSIS_DPATH"/confusion_analysis/bas_confusion.kwcoco.zip \
    --pred_sites "$FILTERED_DPATH"/out_sites \
    --true_region_dpath="$DVC_DATA_DPATH/annotations/drop6/region_models" \
    --true_site_dpath="$DVC_DATA_DPATH/annotations/drop6/site_models" \
    --out_dpath "$ANALYSIS_DPATH"/confusion_analysis


### BAS SALIENCY VIZ



#### For BAS ONLY

#POLY_EVAL_DPATH=/home/joncrall/remote/toothbrush/data/dvc-repos/smart_expt_dvc/_toothbrush_split6_landcover_MeanYear10GSD-V2/eval/flat/bas_poly_eval/bas_poly_eval_id_1ed4acad/
#POLY_EVAL_DPATH=/home/joncrall/remote/toothbrush/data/dvc-repos/smart_expt_dvc/_toothbrush_split6_landcover_MeanYear10GSD-V2/eval/flat/bas_poly_eval/bas_poly_eval_id_afabb3af
#BAS_POLY_DPATH=$(echo $POLY_EVAL_DPATH/.pred/bas_poly/*)

#FILTERED_REGION_ID=$(python -c "import pathlib; print(list(pathlib.Path('$POLY_DPATH/site_summaries').glob('*'))[0].stem)")
#echo "FILTERED_REGION_ID = $FILTERED_REGION_ID"

##DVC_DATA_DPATH=$(geowatch_dvc --tags='phase2_data' --hardware=auto)
#geowatch reproject \
#        --src "$POLY_PRED_FPATH" \
#        --dst "$ANALYSIS_DPATH/bas_pred_and_truth.kwcoco.zip" \
#        --region_models="$DVC_DATA_DPATH/annotations/drop6/region_models/${FILTERED_REGION_ID}.geojson" \
#        --site_models="$DVC_DATA_DPATH/annotations/drop6/site_models/${FILTERED_REGION_ID}_*.geojson" \
#        --status_to_catname="{system_confirmed: positive}" \
#        --validate_checks=False \
#        --role=truth \
#        --clear_existing=False


#geowatch visualize --smart 1 \
#    --draw_labels True \
#    --alpha 0.5 \
#    --src $POLY_EVAL_DPATH/pred_and_truth.kwcoco.zip \
#    --viz_dpath $POLY_EVAL_DPATH/_kwviz \


#geowatch visualize --smart 1 \
#    --draw_labels False \
#    --alpha 0.5 \
#    --src $POLY_EVAL_DPATH/pred_and_truth.kwcoco.zip \
#    --only_boxes=True \
#    --viz_dpath $POLY_EVAL_DPATH/_kwviz_onlyboxes


##--ann_score_thresh 0.5 \
