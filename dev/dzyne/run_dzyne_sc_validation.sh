#!/bin/bash
set -e

# geowatch_dvc add --name=smart_data_hdd --path=/smart_data_dvc --tags=data_dir --hardware=hdd
# geowatch_dvc add --name=smart_expt_hdd --path=/bigdata_smart/smart_expt_dvc --tags=expt_dir --hardware=hdd

# Determine the paths to your SMART data and experiment repositories.
DATA_DVC_DPATH=$(geowatch_dvc --tags='data_dir' --hardware=auto)
EXPT_DVC_DPATH=$(geowatch_dvc --tags='expt_dir' --hardware=auto)

echo "
DATA_DVC_DPATH=$DATA_DVC_DPATH
EXPT_DVC_DPATH=$EXPT_DVC_DPATH
"

OUTPUT_DIR="${DATA_DVC_DPATH}/Validation-V1/positive_annotated/AE_R001_positive_annotated"
STARTING_KWCOCO_FILE="${OUTPUT_DIR}/data.kwcoco.json"

# WV filter
FILTERED_KWCOCO_FILE="${OUTPUT_DIR}/data_filteredWV.kwcoco.json"
if [ ! -f "$FILTERED_KWCOCO_FILE" ]; then
    python -m geowatch.tasks.dzyne_misc.filter \
        --input=$STARTING_KWCOCO_FILE \
        --output=$FILTERED_KWCOCO_FILE \
        --create_gif=true
else
    echo "$FILTERED_KWCOCO_FILE exists... skipping filtering"
fi

# DEPTH 
DEPTH_KWCOCO_FILE="${OUTPUT_DIR}/data_filteredWV_depth.kwcoco.json"
if [ ! -f "$DEPTH_KWCOCO_FILE" ]; then
    python3 -m watch.tasks.depth.predict \
        --deployed=${EXPT_DVC_DPATH}/models/depth/RGB_LowRes_V1.pt \
        --dataset=$FILTERED_KWCOCO_FILE \
        --output=$DEPTH_KWCOCO_FILE \
        --asset_suffix=_assets/depth_1024 \
        --data_workers=8 \
        --window_size=1024 \
        --cache=1 \
        --select_images='.is_valid_wv == true'
else
    echo "${DEPTH_KWCOCO_FILE} exists... skipping depth detection"
fi

# CHANGE
CHANGE_KWCOCO_FILE="${OUTPUT_DIR}/data_filteredWV_depth_change.kwcoco.json"
if [ ! -f "$CHANGE_KWCOCO_FILE" ]; then
    python3 -W ignore -m watch.tasks.change_detection.pair_BIT_CD     \
        --project_name CD_base_transformer_pos_s4_dd8_LEVIR_b2_lr0.01_trainval_test_1000_linear     \
        --net_G base_transformer_pos_s4_dd8     \
        --output_folder=$OUTPUT_DIR \
        --src_kwcoco=$DEPTH_KWCOCO_FILE \
        --checkpoint_root=${EXPT_DVC_DPATH}/models/change_detection/
else
    echo "${CHANGE_KWCOCO_FILE} exists... skipping change detection"
fi

# PROBABILITY 
PROBABIlITY_KWCOCO_FILE="${OUTPUT_DIR}/data_filteredWV_depth_change_prob.kwcoco.json"
if [ ! -f "$PROBABIlITY_KWCOCO_FILE" ]; then
    python3 -W ignore -m watch.tasks.dzyne_misc.compute_change_probability \
        --input=$CHANGE_KWCOCO_FILE \
        --output=$PROBABIlITY_KWCOCO_FILE
else 
    echo "${PROBABIlITY_KWCOCO_FILE} exist... skipping probability scoring"
fi

# VISUALIZATION
if [  -f "$PROBABIlITY_KWCOCO_FILE" ]; then
    python -W ignore -m watch.tasks.dzyne_misc.create_region_gif \
        -i ${PROBABIlITY_KWCOCO_FILE} \
        -o ${DATA_DVC_DPATH}/Validation-V1/positive_annotated/AE_R001_positive_annotated/_assets/gifs \
        -s WV -rc 'red|green|blue,depth' \
        -oc 'change' \
        -d 0.5
else
    echo "${PROBABIlITY_KWCOCO_FILE} does not exist... skipping visualization"
fi