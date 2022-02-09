#!/bin/bash
__doc__="
If you need to regenerate the regions use:

"

DVC_DPATH=$HOME/data/dvc-repos/smart_watch_dvc
#DVC_DPATH=$(python -m watch.cli.find_dvc)
DATASET_SUFFIX=TA1_FULL_SEQ_KR_S001
S3_FPATH=s3://kitware-smart-watch-data/processed/ta1/eval2/master_collation_working/KR_S001.unique.fixed_ls_ids.output
S3_DPATH=s3://kitware-smart-watch-data/processed/ta1/eval2/master_collation_working
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
UNCROPPED_QUERY_FPATH=$UNCROPPED_QUERY_DPATH/KR_S001.unique.fixed_ls_ids.output
UNCROPPED_CATALOG_FPATH=$UNCROPPED_INGRESS_DPATH/catalog.json

export AWS_DEFAULT_PROFILE=iarpa


mkdir -p "$UNCROPPED_QUERY_DPATH"
aws s3 --profile iarpa ls "$S3_DPATH/"
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
python -m watch.cli.prepare_teamfeats \
    --base_fpath="$DVC_DPATH/Aligned-TA1_FULL_SEQ_KR_S001/data.kwcoco.json" \
    --gres="0," \
    --with_depth=False \
    --with_materials=False \
    --with_invariants=False \
    --with_landcover=True \
    --keep_sessions=0 --run=1 \
    --workers=0 --do_splits=0


export CUDA_VISIBLE_DEVICES=2
python -m watch.tasks.landcover.predict \
    --dataset="$ALIGNED_KWCOCO_BUNDLE/data.kwcoco.json" \
    --deployed="/home/local/KHQ/jon.crall/data/dvc-repos/smart_watch_dvc/models/landcover/visnav_remap_s2_subset.pt" \
    --output="$ALIGNED_KWCOCO_BUNDLE/dzyne_landcover.kwcoco.json" \
    --num_workers="4" \
    --device=0

python -m watch.cli.coco_combine_features \
    --src "$ALIGNED_KWCOCO_BUNDLE/data.kwcoco.json" \
          "$ALIGNED_KWCOCO_BUNDLE/dzyne_landcover.kwcoco.json" \
    --dst "$ALIGNED_KWCOCO_BUNDLE/combo_L.kwcoco.json"


DVC_DPATH=$HOME/data/dvc-repos/smart_watch_dvc/
BAS_MODEL_SUFFIX=models/fusion/SC-20201117/BAS_TA1_ALL_REGIONS_v084/BAS_TA1_ALL_REGIONS_v084_epoch=5-step=51917.pt
BAS_MODEL_PATH=$DVC_DPATH/$BAS_MODEL_SUFFIX
[[ -f "$BAS_MODEL_PATH" ]] || (cd "$DVC_DPATH" && dvc pull $BAS_MODEL_SUFFIX)



kwcoco subset --src "$ALIGNED_KWCOCO_BUNDLE/combo_L.kwcoco.json" \
        --dst "$ALIGNED_KWCOCO_BUNDLE/combo_L_s2.kwcoco.json" \
        --select_images '.sensor_coarse == "S2"'


INPUT_DATASET=$ALIGNED_KWCOCO_BUNDLE/combo_L.kwcoco.json
SUGGESTIONS=$(
    python -m watch.tasks.fusion.organize suggest_paths  \
        --package_fpath="$BAS_MODEL_PATH"  \
        --test_dataset="$INPUT_DATASET")
OUTPUT_BAS_DATASET="$(echo "$SUGGESTIONS" | jq -r .pred_dataset)"


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
        --src "$ALIGNED_KWCOCO_BUNDLE/combo_L.kwcoco.json" \
        --space="video" \
        --num_workers=avail \
        --channels="red|green|blue,forest|brush|bare_ground" \
        --viz_dpath="$ALIGNED_KWCOCO_BUNDLE/_viz_combo_L" \
        --draw_anns=False \
        --animate=True \
        --any3=False \
        --fixed_normalization_scheme=scaled_raw_25percentile

    python -m watch intensity_histograms \
        --src "$ALIGNED_KWCOCO_BUNDLE/combo_L.kwcoco.json" \
        --dst="$ALIGNED_KWCOCO_BUNDLE/_viz_combo_L/intensity.png" \
        --exclude_channels="forest|brush|bare_ground|built_up|cropland|wetland|water|snow_or_ice_field" \
        --valid_range="0;10000" \
        --workers="avail/2" 

    python -m kwcoco stats "$INPUT_DATASET"
    python -m watch stats "$INPUT_DATASET"
    python -m watch.cli.torch_model_stats "$BAS_MODEL_PATH"

    #jq .images[0] "$INPUT_DATASET"
}

