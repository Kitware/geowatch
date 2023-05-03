#!/bin/bash

#DVC_DATA_DPATH=$(geowatch_dvc --tags="phase2_data" --hardware="hdd")
DVC_DATA_DPATH=/home/local/KHQ/connor.greenwell/data/dvc-repos/smart_watch_dvc
DVC_EXPT_DPATH=$(geowatch_dvc --tags="phase2_expt")
WORKDIR=$DVC_EXPT_DPATH/training/$HOSTNAME/$USER

DATASET_CODE=onera_2018

KWCOCO_BUNDLE_DPATH=$DVC_DATA_DPATH/extern/$DATASET_CODE
TEST_FPATH=$KWCOCO_BUNDLE_DPATH/onera_test.kwcoco.json

EXPERIMENT_NAME=OSCD_HeterogeneousModel_native_scaled_1.0
DEFAULT_ROOT_DIR=$WORKDIR/$DATASET_CODE/runs/$EXPERIMENT_NAME
MODEL_PATH=$DEFAULT_ROOT_DIR/lightning_logs/version_1/package-interupt/package_epoch7688_step138384.pt
python predict.py \
    --model_path="$MODEL_PATH"  \
    --coco_dataset="$TEST_FPATH" \
    --chip_size=96 \
    --time_steps=2 \
    --window_overlap=0.5 \
    --space_scale=native \
    --channels="B02|B03|B04|B08,B01,B11|B12"
