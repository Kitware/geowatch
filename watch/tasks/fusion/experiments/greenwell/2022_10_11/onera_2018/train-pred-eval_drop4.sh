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

EXPERIMENT_NAME=OSCD_HeterogeneousModel

DEFAULT_ROOT_DIR=$WORKDIR/$DATASET_CODE/runs/$EXPERIMENT_NAME
python -m watch.tasks.fusion fit \
    --config=config.yaml \
    --data.train_dataset="$TRAIN_FPATH" \
    --data.vali_dataset="$VALI_FPATH" \
    --model.init_args.name=$EXPERIMENT_NAME \
    --trainer.default_root_dir="$DEFAULT_ROOT_DIR" \
    --trainer.accelerator="gpu" \
    --trainer.devices=1 \
    --trainer.precision=16 \
    --trainer.max_steps=20000

# Predict 
python -m watch.tasks.fusion.predict \
    --test_dataset="$TEST_FPATH" \
    --package_fpath="$DEFAULT_ROOT_DIR"/final_package.pt  \
    --pred_dataset="$DVC_EXPT_DPATH"/predictions/pred.kwcoco.json


# Inspect the channels in the prediction file
smartwatch stats "$DVC_EXPT_DPATH"/predictions/pred.kwcoco.json


# Evaluate 
python -m watch.tasks.fusion.evaluate \
    --true_dataset="$TEST_FPATH" \
    --pred_dataset="$DVC_EXPT_DPATH"/predictions/pred.kwcoco.json \
    --eval_dpath="$DVC_EXPT_DPATH"/predictions/eval
