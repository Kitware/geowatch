#!/bin/bash

#DVC_DATA_DPATH=$(smartwatch_dvc --tags="phase2_data" --hardware="hdd")
DVC_DATA_DPATH=/home/local/KHQ/connor.greenwell/data/dvc-repos/smart_watch_dvc
DVC_EXPT_DPATH=$(smartwatch_dvc --tags="phase2_expt")
WORKDIR=$DVC_EXPT_DPATH/training/$HOSTNAME/$USER

DATASET_CODE=onera_2018

KWCOCO_BUNDLE_DPATH=$DVC_DATA_DPATH/extern/$DATASET_CODE
TRAIN_FPATH=$KWCOCO_BUNDLE_DPATH/onera_train.kwcoco.json
VALI_FPATH=$KWCOCO_BUNDLE_DPATH/onera_test.kwcoco.json
TEST_FPATH=$KWCOCO_BUNDLE_DPATH/onera_test.kwcoco.json

EXPERIMENT_NAME=OSCD_HeterogeneousModel_upsampled_0.1
DEFAULT_ROOT_DIR=$WORKDIR/$DATASET_CODE/runs/$EXPERIMENT_NAME
python -m watch.tasks.fusion fit \
    --config=config.yaml \
    --model.init_args.name=$EXPERIMENT_NAME \
    --model.ignore_scale=true \
    --model.spatial_scale_base=0.1 \
    --data.batch_size=16 \
    --data.input_space_scale=null \
    --data.train_dataset="$TRAIN_FPATH" \
    --data.vali_dataset="$VALI_FPATH" \
    --data.num_workers=8 \
    --trainer.gradient_clip_val=5.0 \
    --trainer.gradient_clip_algorithm="norm" \
    --trainer.detect_anomaly=true \
    --trainer.default_root_dir="$DEFAULT_ROOT_DIR" \
    --trainer.accelerator="gpu" \
    --trainer.devices=1 \
    --trainer.precision=16 \
    --trainer.accumulate_grad_batches=4 \
    --trainer.max_steps=200000
