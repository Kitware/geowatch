#!/bin/bash

#DVC_DATA_DPATH=$(smartwatch_dvc --tags="phase2_data" --hardware="hdd")
DVC_DATA_DPATH=/home/local/KHQ/connor.greenwell/data/dvc-repos/smart_watch_dvc
DVC_EXPT_DPATH=$(smartwatch_dvc --tags="phase2_expt")
WORKDIR=$DVC_EXPT_DPATH/training/$HOSTNAME/$USER

DATASET_CODE=onera_2018

KWCOCO_BUNDLE_DPATH=$DVC_DATA_DPATH/extern/$DATASET_CODE
TEST_FPATH=$KWCOCO_BUNDLE_DPATH/onera_test.kwcoco.json

EXPERIMENT_NAME=OSCD_HeterogeneousModel_upsampled_0.1
DEFAULT_ROOT_DIR=$WORKDIR/$DATASET_CODE/runs/$EXPERIMENT_NAME
python predict.py \
    --model_path="$DEFAULT_ROOT_DIR"/final_package.pt  \
    --coco_dataset="$TEST_FPATH" \
    --chip_size=96 \
    --time_steps=2 \
    --window_overlap=0.5 \
    --channels="B02|B03|B04|B08,B01,B11|B12"

EXPERIMENT_NAME=OSCD_HeterogeneousModel_upsampled_1.0
DEFAULT_ROOT_DIR=$WORKDIR/$DATASET_CODE/runs/$EXPERIMENT_NAME
python predict.py \
    --model_path="$DEFAULT_ROOT_DIR"/final_package.pt  \
    --coco_dataset="$TEST_FPATH" \
    --chip_size=96 \
    --time_steps=2 \
    --window_overlap=0.5 \
    --channels="B02|B03|B04|B08,B01,B11|B12"

EXPERIMENT_NAME=OSCD_HeterogeneousModel_native_scaled_0.1
DEFAULT_ROOT_DIR=$WORKDIR/$DATASET_CODE/runs/$EXPERIMENT_NAME
python predict.py \
    --model_path="$DEFAULT_ROOT_DIR"/final_package.pt  \
    --coco_dataset="$TEST_FPATH" \
    --chip_size=96 \
    --time_steps=2 \
    --window_overlap=0.5 \
    --space_scale=native \
    --channels="B02|B03|B04|B08,B01,B11|B12"

EXPERIMENT_NAME=OSCD_HeterogeneousModel_native_scaled_1.0
DEFAULT_ROOT_DIR=$WORKDIR/$DATASET_CODE/runs/$EXPERIMENT_NAME
python predict.py \
    --model_path="$DEFAULT_ROOT_DIR"/final_package.pt  \
    --coco_dataset="$TEST_FPATH" \
    --chip_size=96 \
    --time_steps=2 \
    --window_overlap=0.5 \
    --space_scale=native \
    --channels="B02|B03|B04|B08,B01,B11|B12"

EXPERIMENT_NAME=OSCD_HeterogeneousModel_native_unscaled_0.1
DEFAULT_ROOT_DIR=$WORKDIR/$DATASET_CODE/runs/$EXPERIMENT_NAME
python predict.py \
    --model_path="$DEFAULT_ROOT_DIR"/final_package.pt  \
    --coco_dataset="$TEST_FPATH" \
    --chip_size=96 \
    --time_steps=2 \
    --window_overlap=0.5 \
    --space_scale=native \
    --channels="B02|B03|B04|B08,B01,B11|B12"

EXPERIMENT_NAME=OSCD_HeterogeneousModel_native_unscaled_1.0
DEFAULT_ROOT_DIR=$WORKDIR/$DATASET_CODE/runs/$EXPERIMENT_NAME
python predict.py \
    --model_path="$DEFAULT_ROOT_DIR"/final_package.pt  \
    --coco_dataset="$TEST_FPATH" \
    --chip_size=96 \
    --time_steps=2 \
    --window_overlap=0.5 \
    --space_scale=native \
    --channels="B02|B03|B04|B08,B01,B11|B12"
