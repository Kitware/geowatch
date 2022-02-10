#!/bin/bash
__doc__="
If you need to regenerate the regions use:
"

DATASET_SUFFIX=TA1_FULL_SEQ_KR_S001_CLOUD_LT_10
S3_FPATH=s3://kitware-smart-watch-data/processed/ta1/eval2/master_collation_working/KR_S001.unique.fixed_ls_ids.cloudcover_lt_10.output

#DATASET_SUFFIX=TA1_FULL_SEQ_KR_S001
#S3_FPATH=s3://kitware-smart-watch-data/processed/ta1/eval2/master_collation_working/KR_S001.unique.fixed_ls_ids.output


DVC_DPATH=$HOME/data/dvc-repos/smart_watch_dvc
#DVC_DPATH=$(python -m watch.cli.find_dvc)
#S3_DPATH=s3://kitware-smart-watch-data/processed/ta1/eval2/master_collation_working
QUERY_BASENAME=$(basename "$S3_FPATH")
ALIGNED_BUNDLE_NAME=Aligned-$DATASET_SUFFIX
UNCROPPED_BUNDLE_NAME=Uncropped-$DATASET_SUFFIX
#REGION_MODELS=$DVC_DPATH'/annotations/region_models/*.geojson'
REGION_MODELS=$DVC_DPATH/annotations/region_models/KR_R002.geojson
# Helper Variables
UNCROPPED_DPATH=$DVC_DPATH/$UNCROPPED_BUNDLE_NAME
UNCROPPED_QUERY_DPATH=$UNCROPPED_DPATH/_query/items
UNCROPPED_INGRESS_DPATH=$UNCROPPED_DPATH/ingress
UNCROPPED_KWCOCO_FPATH=$UNCROPPED_DPATH/data.kwcoco.json
ALIGNED_KWCOCO_BUNDLE=$DVC_DPATH/$ALIGNED_BUNDLE_NAME
ALIGNED_KWCOCO_FPATH=$ALIGNED_KWCOCO_BUNDLE/data.kwcoco.json
UNCROPPED_QUERY_FPATH=$UNCROPPED_QUERY_DPATH/$QUERY_BASENAME
UNCROPPED_CATALOG_FPATH=$UNCROPPED_INGRESS_DPATH/catalog.json



export AWS_DEFAULT_PROFILE=iarpa


mkdir -p "$UNCROPPED_QUERY_DPATH"
#aws s3 --profile iarpa ls "$S3_DPATH/"
aws s3 --profile iarpa cp $S3_FPATH "$UNCROPPED_QUERY_DPATH"
ls -al "$UNCROPPED_QUERY_DPATH"

#cat "$UNCROPPED_QUERY_FPATH" | sort -u > "$UNCROPPED_QUERY_FPATH.unique"
python -m watch.cli.baseline_framework_ingress \
    --aws_profile iarpa \
    --jobs 4 \
    --virtual \
    --outdir "$UNCROPPED_INGRESS_DPATH" \
    "$UNCROPPED_QUERY_FPATH"

python -m watch.cli.ta1_stac_to_kwcoco \
    "$UNCROPPED_CATALOG_FPATH" \
    --outpath="$UNCROPPED_KWCOCO_FPATH" \
    --populate-watch-fields \
    --from-collated \
    --jobs avail

python -m watch.cli.coco_align_geotiffs \
    --src "$UNCROPPED_KWCOCO_FPATH" \
    --dst "$ALIGNED_KWCOCO_FPATH" \
    --regions "$REGION_MODELS" \
    --workers=avail \
    --context_factor=1 \
    --geo_preprop=auto \
    --visualize False \
    --keep none \
    --rpc_align_method affine_warp


#DVC_DPATH=$(python -m watch.cli.find_dvc)
#python -m watch.cli.prepare_teamfeats \
#    --base_fpath="$ALIGNED_KWCOCO_BUNDLE/data.kwcoco.json" \
#    --gres="0," \
#    --with_depth=True \
#    --with_materials=False \
#    --with_invariants=False \
#    --with_landcover=True \
#    --keep_sessions=0 --run=1 \
#    --workers=0 --do_splits=0


#DEPTH_MODEL_SUFFIX=models/landcover/visnav_remap_s2_subset.pt
#DEPTH_MODEL_FPATH=$DVC_DPATH/$DEPTH_MODEL_SUFFIX
#[[ -f "$DEPTH_MODEL_FPATH" ]] || (cd "$DVC_DPATH" && dvc pull $DEPTH_MODEL_SUFFIX)
#export CUDA_VISIBLE_DEVICES=1
#python -m watch.tasks.depth.predict \
#    --dataset="$ALIGNED_KWCOCO_BUNDLE/data.kwcoco.json" \
#    --output="$ALIGNED_KWCOCO_BUNDLE/dzyne_depth.kwcoco.json" \
#    --deployed="$DVC_DPATH/models/depth/weights_v1.pt" \
#    --data_workers=2 \
#    --window_size=1536


LANDCOVER_MODEL_SUFFIX=models/landcover/visnav_remap_s2_subset.pt
LANDCOVER_MODEL_FPATH=$DVC_DPATH/$LANDCOVER_MODEL_SUFFIX
[[ -f "$LANDCOVER_MODEL_FPATH" ]] || (cd "$DVC_DPATH" && dvc pull $LANDCOVER_MODEL_SUFFIX)
export CUDA_VISIBLE_DEVICES=1
python -m watch.tasks.landcover.predict \
    --dataset="$ALIGNED_KWCOCO_BUNDLE/data.kwcoco.json" \
    --deployed="$LANDCOVER_MODEL_FPATH" \
    --output="$ALIGNED_KWCOCO_BUNDLE/dzyne_landcover.kwcoco.json" \
    --num_workers="avail/4" \
    --device=0

python -m watch.cli.coco_combine_features \
    --src "$ALIGNED_KWCOCO_BUNDLE/data.kwcoco.json" \
          "$ALIGNED_KWCOCO_BUNDLE/dzyne_landcover.kwcoco.json" \
    --dst "$ALIGNED_KWCOCO_BUNDLE/combo_L.kwcoco.json"


DVC_DPATH=$HOME/data/dvc-repos/smart_watch_dvc/
INPUT_DATASET=$ALIGNED_KWCOCO_BUNDLE/combo_L.kwcoco.json

BAS_MODEL_SUFFIX=models/fusion/SC-20201117/BAS_TA1_c001_v076/BAS_TA1_c001_v076_epoch=90-step=186367.pt
#BAS_MODEL_SUFFIX=models/fusion/SC-20201117/BAS_TA1_c001_v082/BAS_TA1_c001_v082_epoch=42-step=88063.pt
#BAS_MODEL_SUFFIX=models/fusion/SC-20201117/BAS_TA1_c001_v073/BAS_TA1_c001_v073_epoch=13-step=28671.pt
#BAS_MODEL_SUFFIX=models/fusion/SC-20201117/BAS_TA1_ALL_REGIONS_v084/BAS_TA1_ALL_REGIONS_v084_epoch=5-step=51917.pt


BAS_MODEL_PATH=$DVC_DPATH/$BAS_MODEL_SUFFIX
[[ -f "$BAS_MODEL_PATH" ]] || (cd "$DVC_DPATH" && dvc pull "$BAS_MODEL_SUFFIX")
SUGGESTIONS=$(
    python -m watch.tasks.fusion.organize suggest_paths  \
        --package_fpath="$BAS_MODEL_PATH"  \
        --test_dataset="$INPUT_DATASET")
OUTPUT_BAS_DATASET="$(echo "$SUGGESTIONS" | jq -r .pred_dataset)"

export CUDA_VISIBLE_DEVICES=1
python -m watch.tasks.fusion.predict \
       --write_preds False \
       --write_probs True \
       --with_change False \
       --with_saliency True \
       --with_class False \
       --test_dataset "$INPUT_DATASET" \
       --package_fpath "$BAS_MODEL_PATH" \
       --pred_dataset "$OUTPUT_BAS_DATASET" \
       --batch_size 8 \
       --gpus 1 

_debug(){
    python -m watch visualize \
        --src "$OUTPUT_BAS_DATASET" --channels="salient" \
        --extra_header="$(basename "$BAS_MODEL_PATH")" \
        --draw_anns=False --animate=True --workers=4 
}

# Site characterization
DVC_DPATH=$HOME/data/dvc-repos/smart_watch_dvc/
SC_MODEL_SUFFIX=models/fusion/SC-20201117/SC_smt_it_stm_p8_TA1_xfer55_v70/SC_smt_it_stm_p8_TA1_xfer55_v70_epoch=34-step=71679.pt
SC_MODEL_PATH=$DVC_DPATH/$SC_MODEL_SUFFIX
[[ -f "$SC_MODEL_PATH" ]] || (cd "$DVC_DPATH" && dvc pull "$SC_MODEL_PATH")
SUGGESTIONS=$(
    python -m watch.tasks.fusion.organize suggest_paths  \
        --package_fpath="$SC_MODEL_PATH"  \
        --test_dataset="$INPUT_DATASET")
OUTPUT_SC_DATASET="$(echo "$SUGGESTIONS" | jq -r .pred_dataset)"

export CUDA_VISIBLE_DEVICES=1
python -m watch.tasks.fusion.predict \
       --write_preds False \
       --write_probs True \
       --with_change False \
       --with_saliency False \
       --with_class True \
       --num_workers=4 \
       --test_dataset "$INPUT_DATASET" \
       --package_fpath "$SC_MODEL_PATH" \
       --pred_dataset "$OUTPUT_SC_DATASET" \
       --batch_size 32 \
       --gpus 1
_debug(){
    python -m watch visualize \
        --src "$OUTPUT_SC_DATASET" --channels="No Activity|Active Construction|Site Preparation" \
        --extra_header="$(basename "$SC_MODEL_PATH")" \
        --draw_anns=False --animate=True --workers=4 
}


_debug(){
    __notes__="

    Drop2 train set:

        'L8': {                                                                                                                                                                     
            'coastal|lwir11|lwir12|blue|green|red|nir|swir16|swir22|pan|cirrus|QA_PIXEL|QA_RADSAT|SAA|SEA4|SEZ4|SOA4|SOZ4|SZA|VAA|VZA|cloudmask|forest|brush|bare_ground|built_up|cr
opland|wetland|water|snow_or_ice_field': 344,                                                                                                                                       
            'coastal|lwir11|lwir12|blue|green|red|nir|swir16|swir22|pan|cirrus|QA_PIXEL|QA_RADSAT|SAA|SZA|VAA|VZA|cloudmask|forest|brush|bare_ground|built_up|cropland|wetland|water
|snow_or_ice_field': 66,                                                                                                                                                            
        },                                                                                                                                                                          
        'S2': {                                                                                                                                                                     
            'coastal|blue|green|red|B05|B06|B07|nir|B09|cirrus|swir16|swir22|B8A|SEA4|SEZ4|SOA4|SOZ4|cloudmask|forest|brush|bare_ground|built_up|cropland|wetland|water|snow_or_ice_
field': 593,                                                                                                                                                                        
            'coastal|blue|green|red|B05|B06|B07|nir|B09|cirrus|swir16|swir22|B8A|cloudmask|forest|brush|bare_ground|built_up|cropland|wetland|water|snow_or_ice_field': 66,         
        },                                                                                                                                                                          
        'WV': {                                                                                                                                                                     
            'blue|green|red|near-ir1': 29,                                                                                                                                          
            'blue|green|red|near-ir1|panchromatic': 5,                                                                                                                              
            'coastal|blue|green|yellow|red|red-edge|near-ir1|near-ir2': 197,                                                                                                        
            'coastal|blue|green|yellow|red|red-edge|near-ir1|near-ir2|panchromatic': 5,                                                                                             
            'panchromatic': 12,                                                                                                                                                     
            'panchromatic|blue|green|red|near-ir1': 1,                                                                                                                              
            'panchromatic|coastal|blue|green|yellow|red|red-edge|near-ir1|near-ir2': 2,                                                                                             
        },                                                                                                                                                                          
    
    "
    #kwcoco subset --src "$ALIGNED_KWCOCO_BUNDLE/combo_L.kwcoco.json" \
    #        --dst "$ALIGNED_KWCOCO_BUNDLE/combo_L_s2.kwcoco.json" \
    #        --select_images '.sensor_coarse == "S2"'

    python -m watch visualize \
        --src "$OUTPUT_BAS_DATASET" \
        --space="video" \
        --num_workers=2 \
        --channels="salient" \
        --draw_anns=False \
        --animate=True \
        --workers=4 \
        --any3=False 


    smartwatch stats "$DVC_DPATH/Drop1-Aligned-L1/combo_vali_nowv.kwcoco.json"

    python -m watch visualize \
        --src "$DVC_DPATH/Drop1-Aligned-L1/combo_vali_nowv.kwcoco.json" \
        --space="video" \
        --num_workers=avail \
        --channels="red|green|blue,forest|brush|bare_ground" \
        --viz_dpath="$DVC_DPATH/Drop1-Aligned-L1/_viz_combo" \
        --draw_anns=False \
        --animate=True \
        --workers=4 \
        --any3=False 

    python -m watch visualize \
        --src "$DVC_DPATH/Drop2-Aligned-TA1-2022-01/combo_L_nowv_vali.kwcoco.json" \
        --space="video" \
        --num_workers=avail \
        --channels="red|green|blue,forest|brush|bare_ground" \
        --viz_dpath="$DVC_DPATH/Drop2-Aligned-TA1-2022-01/_viz_combo_L" \
        --draw_anns=False \
        --animate=True \
        --workers=4 \
        --any3=False 

    python -m watch intensity_histograms \
        --src "$DVC_DPATH/Drop2-Aligned-TA1-2022-01/combo_L.kwcoco.json" \
        --dst="$DVC_DPATH/Drop2-Aligned-TA1-2022-01/_viz_combo_L/intensity.png" \
        --exclude_channels="cloudmask|cirrus|forest|brush|bare_ground|built_up|cropland|wetland|water|snow_or_ice_field" \
        --valid_range="1:6000" \
        --workers="0" 

    python -m watch visualize \
        --src "$ALIGNED_KWCOCO_BUNDLE/combo_L.kwcoco.json" \
        --space="video" \
        --num_workers=avail \
        --channels="red|green|blue,forest|brush|bare_ground" \
        --viz_dpath="$ALIGNED_KWCOCO_BUNDLE/_viz_combo_L" \
        --draw_anns=False \
        --animate=True \
        --any3=False 

    python -m watch intensity_histograms \
        --src "$ALIGNED_KWCOCO_BUNDLE/combo_L.kwcoco.json" \
        --dst="$ALIGNED_KWCOCO_BUNDLE/_viz_combo_L/intensity.png" \
        --exclude_channels="cirrus|forest|brush|bare_ground|built_up|cropland|wetland|water|snow_or_ice_field" \
        --valid_range="1:6000" \
        --workers="avail" 


    DVC_DPATH=$HOME/data/dvc-repos/smart_watch_dvc/
    BAS_MODEL_SUFFIX=models/fusion/SC-20201117/BAS_TA1_ALL_REGIONS_v084/BAS_TA1_ALL_REGIONS_v084_epoch=5-step=51917.pt
    BAS_MODEL_SUFFIX=models/fusion/SC-20201117/BAS_TA1_c001_v082/BAS_TA1_c001_v082_epoch=42-step=88063.pt
    BAS_MODEL_PATH=$DVC_DPATH/$BAS_MODEL_SUFFIX
    python -m watch.tasks.fusion.predict \
           --write_preds False \
           --write_probs True \
           --with_change False \
           --with_saliency True \
           --with_class False \
           --test_dataset "$DVC_DPATH/Drop2-Aligned-TA1-2022-01/combo_L_nowv_vali.kwcoco.json" \
           --package_fpath "$BAS_MODEL_PATH" \
           --pred_dataset "$DVC_DPATH/temp/poc" \
           --batch_size 8 \
           --gpus 1
            

    python -m kwcoco stats "$INPUT_DATASET" python -m watch stats "$INPUT_DATASET"
    python -m watch.cli.torch_model_stats "$BAS_MODEL_PATH"

    
    #--test_dataset="$DVC_DPATH/Drop2-Aligned-TA1-2022-01/combo_L_nowv_vali.kwcoco.json" \
    #--test_dataset="$DVC_DPATH/Aligned-TA1_FULL_SEQ_KR_S001/combo_L.kwcoco.json" \
    python -m watch.cli.torch_model_stats "$DVC_DPATH/models/fusion/SC-20201117/BAS_TA1_c001_v080/BAS_TA1_c001_v080_epoch=54-step=112639.pt"
    python -m watch.cli.torch_model_stats "$DVC_DPATH/models/fusion/SC-20201117/BAS_TA1_ALL_REGIONS_v084/BAS_TA1_ALL_REGIONS_v084_epoch=5-step=51917.pt"

    python -m watch.cli.torch_model_stats ~/data/dvc-repos/smart_watch_dvc/models/fusion/SC-20201117/BAS_TA1_ALL_REGIONS_v084/BAS_TA1_ALL_REGIONS_v084_epoch=3-step=34611.pt

    python -m watch.cli.torch_model_stats "$HOME/data/dvc-repos/smart_watch_dvc/models/fusion/SC-20201117/BAS_TA1_c001_v082/BAS_TA1_c001_v082_epoch=42-step=88063.pt"
    python -m watch.cli.torch_model_stats "$HOME/data/dvc-repos/smart_watch_dvc/models/fusion/SC-20201117/BAS_TA1_ALL_REGIONS_v084/BAS_TA1_ALL_REGIONS_v084_epoch=3-step=34611.pt"
    python -m watch.cli.torch_model_stats "$HOME/data/dvc-repos/smart_watch_dvc/models/fusion/SC-20201117/BAS_TA1_KOREA_v083/BAS_TA1_KOREA_v083_epoch=2-step=5594.pt"

    #--package_fpath="$DVC_DPATH/models/fusion/SC-20201117/BAS_TA1_c001_v080/BAS_TA1_c001_v080_epoch=54-step=112639.pt" \
    #--package_fpath="$DVC_DPATH/models/fusion/SC-20201117/BAS_TA1_ALL_REGIONS_v084/BAS_TA1_ALL_REGIONS_v084_epoch=5-step=51917.pt" \

    python -m watch stats "$DVC_DPATH/Aligned-TA1_FULL_SEQ_KR_S001/combo_L.kwcoco.json"

    DVC_DPATH=$HOME/data/dvc-repos/smart_watch_dvc/
    python -m watch.tasks.fusion.predict \
        --write_probs=True \
        --write_preds=False \
        --with_class=auto \
        --with_saliency=auto \
        --with_change=False \
        --pred_dataset="$DVC_DPATH/tmp2/pred.kwcoco.json" \
        --test_dataset="$DVC_DPATH/Aligned-TA1_FULL_SEQ_KR_S001/combo_L.kwcoco.json" \
        --package_fpath="$DVC_DPATH/models/fusion/SC-20201117/BAS_TA1_ALL_REGIONS_v084/BAS_TA1_ALL_REGIONS_v084_epoch=5-step=51917.pt" \
        --num_workers=5 \
        --compress=DEFLATE \
        --gpus="0," \
        --batch_size=1

    #jq .images[0] "$INPUT_DATASET"


    DVC_DPATH=$HOME/data/dvc-repos/smart_watch_dvc/
    python -m watch visualize \
        "$DVC_DPATH/models/fusion/SC-20201117/BAS_TA1_ALL_REGIONS_v084/pred_BAS_TA1_ALL_REGIONS_v084_epoch=5-step=51917/Drop2-Aligned-TA1-2022-01_combo_L_nowv_vali.kwcoco/pred.kwcoco.json" \
        --workers=4 \
        --channels="salient" \
        --draw_anns=False  \
        --animate=True \
        --extra_header="pred_BAS_TA1_ALL_REGIONS_v084_epoch=5-step=51917"

    DVC_DPATH=$HOME/data/dvc-repos/smart_watch_dvc/
    python -m watch visualize \
        "$DVC_DPATH/models/fusion/SC-20201117/BAS_TA1_c001_v076/pred_BAS_TA1_c001_v076_epoch=90-step=186367/Aligned-TA1_FULL_SEQ_KR_S001_combo_L.kwcoco/pred.kwcoco.json" \
        --workers=4 \
        --channels="salient" \
        --draw_anns=False  \
        --animate=True \
        --extra_header="BAS_TA1_c001_v076_epoch=90-step=186367.pt"

    DVC_DPATH=$HOME/data/dvc-repos/smart_watch_dvc/
    python -m watch visualize \
        "$DVC_DPATH/models/fusion/SC-20201117/BAS_TA1_c001_v082/pred_BAS_TA1_c001_v082_epoch=42-step=88063/Aligned-TA1_FULL_SEQ_KR_S001_combo_L.kwcoco/pred.kwcoco.json" \
        --workers=4 \
        --channels="salient" \
        --draw_anns=False  \
        --animate=True \
        --extra_header="BAS_TA1_c001_v082_epoch=42-step=88063.pt"

    smartwatch visualize \
        "$HOME/data/dvc-repos/smart_watch_dvc/Aligned-TA1_FULL_SEQ_KR_S001_CLOUD_LT_10/dzyne_landcover.kwcoco.json" \
        --channels="red|green|blue,bare_ground|forest|wetland" --animate=True --with_anns=False

        smartwatch stats "$HOME/data/dvc-repos/smart_watch_dvc/models/fusion/SC-20201117/SC_smt_it_stm_p8_TA1_xfer55_v70/pred_SC_smt_it_stm_p8_TA1_xfer55_v70_epoch=34-step=71679/Aligned-TA1_FULL_SEQ_KR_S001_combo_L.kwcoco/pred.kwcoco.json" 
    smartwatch visualize \
        "$HOME/data/dvc-repos/smart_watch_dvc/models/fusion/SC-20201117/SC_smt_it_stm_p8_TA1_xfer55_v70/pred_SC_smt_it_stm_p8_TA1_xfer55_v70_epoch=34-step=71679/Aligned-TA1_FULL_SEQ_KR_S001_combo_L.kwcoco/pred.kwcoco.json" \
        --channels="No Activity|Active Construction|Site Preparation" --animate=True 
}

