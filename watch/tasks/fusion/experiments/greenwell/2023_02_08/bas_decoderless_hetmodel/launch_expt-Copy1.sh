#!/bin/bash

PHASE2_DATA_DPATH_SRC=/flash/smart_data_dvc
PHASE2_EXPT_DPATH_SRC=$HOME/data/dvc-repos/smart_expt_dvc

PHASE2_DATA_DPATH=/flash/smart_data_dvc
PHASE2_EXPT_DPATH=/data/smart_expt

WORKDIR=$PHASE2_EXPT_DPATH/training/$HOSTNAME/$USER
# DATASET_CODE=Aligned-Drop4-2022-08-08-TA1-S2-WV-PD-ACC
DATASET_CODE=Aligned-Drop4-2022-08-08-TA1-S2-L8-ACC
KWCOCO_BUNDLE_DPATH=$PHASE2_DATA_DPATH/$DATASET_CODE

TRAIN_FPATH=$KWCOCO_BUNDLE_DPATH/data_train.kwcoco.json
VALI_FPATH=$KWCOCO_BUNDLE_DPATH/data_vali.kwcoco.json
TEST_FPATH=$KWCOCO_BUNDLE_DPATH/data_vali.kwcoco.json

EXPERIMENT_NAME=Drop4_BAS_S2L8_NoDecoderHetModel
DEFAULT_ROOT_DIR=$WORKDIR/$DATASET_CODE/runs/$EXPERIMENT_NAME

docker run \
    --tty --memory=60g --cpus=12 \
    --runtime=nvidia \
    --env NVIDIA_VISIBLE_DEVICES=0,1 \
    --mount type=bind,source="$PHASE2_DATA_DPATH_SRC",target="$PHASE2_DATA_DPATH" \
    --mount type=bind,source="$PHASE2_EXPT_DPATH_SRC",target="$PHASE2_EXPT_DPATH" \
    --mount type=bind,source="$(pwd)/config_common.yaml",target="/config_common.yaml" \
    greenwell-test-build \
    conda run --no-capture-output -n watch \
    python -m watch.tasks.fusion fit \
        --config=/config_common.yaml \
        --model.init_args.name=$EXPERIMENT_NAME \
        --data.train_dataset="$TRAIN_FPATH" \
        --data.vali_dataset="$VALI_FPATH" \
        --data.num_workers=10 \
        --data.verbose=true \
        --trainer.default_root_dir="$DEFAULT_ROOT_DIR" \
        --trainer.max_steps=100 \
        --trainer.accelerator="gpu" \
        --trainer.devices=2 \
        --trainer.strategy="ddp" \
        --trainer.precision=16
        # --print_config
        # --trainer.detect_anomaly=true \
