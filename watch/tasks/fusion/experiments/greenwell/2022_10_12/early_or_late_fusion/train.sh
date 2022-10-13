#!/bin/bash

#DVC_DATA_DPATH=$(smartwatch_dvc --tags="phase2_data" --hardware="hdd")
DVC_DATA_DPATH=/home/local/KHQ/connor.greenwell/Projects/SMART/smart_watch_dvc
# DVC_EXPT_DPATH=$(smartwatch_dvc --tags="phase2_expt")
DVC_EXPT_DPATH=/home/local/KHQ/connor.greenwell/data/dvc-repos/smart_expt_dvc
WORKDIR=$DVC_EXPT_DPATH/training/$HOSTNAME/$USER

DATASET_CODE=onera_2018

KWCOCO_BUNDLE_DPATH=$DVC_DATA_DPATH/extern/$DATASET_CODE
TRAIN_FPATH=$KWCOCO_BUNDLE_DPATH/onera_train.kwcoco.json
VALI_FPATH=$KWCOCO_BUNDLE_DPATH/onera_test.kwcoco.json
TEST_FPATH=$KWCOCO_BUNDLE_DPATH/onera_test.kwcoco.json

EXPERIMENT_NAME=OSCD_HeterogeneousModel_encoder0_decoder4
DEFAULT_ROOT_DIR=$WORKDIR/$DATASET_CODE/runs/$EXPERIMENT_NAME
python fit_lightning.py fit \
    --config=config.yaml \
    --model.init_args.name=$EXPERIMENT_NAME \
    --model.init_args.backbone_encoder_depth=0 \
    --model.init_args.backbone_encoder_depth=4 \
    --data.input_space_scale=native \
    --data.train_dataset="$TRAIN_FPATH" \
    --data.vali_dataset="$VALI_FPATH" \
    --trainer.default_root_dir="$DEFAULT_ROOT_DIR" \
    --trainer.accelerator="gpu" \
    --trainer.devices=1 \
    --trainer.precision=16 \
    --trainer.max_steps=20000

EXPERIMENT_NAME=OSCD_HeterogeneousModel_encoder1_decoder4
DEFAULT_ROOT_DIR=$WORKDIR/$DATASET_CODE/runs/$EXPERIMENT_NAME
python fit_lightning.py fit \
    --config=config.yaml \
    --model.init_args.name=$EXPERIMENT_NAME \
    --model.init_args.backbone_encoder_depth=1 \
    --model.init_args.backbone_encoder_depth=4 \
    --data.input_space_scale=native \
    --data.train_dataset="$TRAIN_FPATH" \
    --data.vali_dataset="$VALI_FPATH" \
    --trainer.default_root_dir="$DEFAULT_ROOT_DIR" \
    --trainer.accelerator="gpu" \
    --trainer.devices=1 \
    --trainer.precision=16 \
    --trainer.max_steps=20000

EXPERIMENT_NAME=OSCD_HeterogeneousModel_encoder3_decoder2
DEFAULT_ROOT_DIR=$WORKDIR/$DATASET_CODE/runs/$EXPERIMENT_NAME
python fit_lightning.py fit \
    --config=config.yaml \
    --model.init_args.name=$EXPERIMENT_NAME \
    --model.init_args.backbone_encoder_depth=3 \
    --model.init_args.backbone_encoder_depth=2 \
    --data.input_space_scale=native \
    --data.train_dataset="$TRAIN_FPATH" \
    --data.vali_dataset="$VALI_FPATH" \
    --trainer.default_root_dir="$DEFAULT_ROOT_DIR" \
    --trainer.accelerator="gpu" \
    --trainer.devices=1 \
    --trainer.precision=16 \
    --trainer.max_steps=20000

EXPERIMENT_NAME=OSCD_HeterogeneousModel_encoder4_decoder1
DEFAULT_ROOT_DIR=$WORKDIR/$DATASET_CODE/runs/$EXPERIMENT_NAME
python fit_lightning.py fit \
    --config=config.yaml \
    --model.init_args.name=$EXPERIMENT_NAME \
    --model.init_args.backbone_encoder_depth=4 \
    --model.init_args.backbone_encoder_depth=1 \
    --data.input_space_scale=native \
    --data.train_dataset="$TRAIN_FPATH" \
    --data.vali_dataset="$VALI_FPATH" \
    --trainer.default_root_dir="$DEFAULT_ROOT_DIR" \
    --trainer.accelerator="gpu" \
    --trainer.devices=1 \
    --trainer.precision=16 \
    --trainer.max_steps=20000
