#!/bin/bash

#DVC_DATA_DPATH=$(geowatch_dvc --tags="phase2_data" --hardware="hdd")
DVC_DATA_DPATH=/home/local/KHQ/connor.greenwell/Projects/SMART/smart_watch_dvc
DVC_EXPT_DPATH=$(geowatch_dvc --tags="phase2_expt")
WORKDIR=$DVC_EXPT_DPATH/training/$HOSTNAME/$USER

DATASET_CODE=onera_2018

KWCOCO_BUNDLE_DPATH=$DVC_DATA_DPATH/extern/$DATASET_CODE
TRAIN_FPATH=$KWCOCO_BUNDLE_DPATH/onera_train.kwcoco.json
VALI_FPATH=$KWCOCO_BUNDLE_DPATH/onera_test.kwcoco.json
TEST_FPATH=$KWCOCO_BUNDLE_DPATH/onera_test.kwcoco.json

EXPERIMENT_NAME=OSCD_HeterogeneousModel_native_scaled_1.0_resnet18
DEFAULT_ROOT_DIR=$WORKDIR/$DATASET_CODE/runs/$EXPERIMENT_NAME
python -m geowatch.tasks.fusion fit \
    --config=config.yaml \
    --model.init_args.name=$EXPERIMENT_NAME \
    --data.input_space_scale=native \
    --data.train_dataset="$TRAIN_FPATH" \
    --data.vali_dataset="$VALI_FPATH" \
    --data.num_workers=8 \
    --trainer.default_root_dir="$DEFAULT_ROOT_DIR" \
    --trainer.accelerator="gpu" \
    --trainer.devices=1 \
    --trainer.precision=16 \
    --trainer.max_steps=200000
