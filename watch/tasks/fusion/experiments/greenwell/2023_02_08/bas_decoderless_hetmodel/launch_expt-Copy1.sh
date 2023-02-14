#!/bin/bash

PHASE2_DATA_DPATH_SRC=/flash/smart_data_dvc
PHASE2_EXPT_DPATH_SRC=$HOME/data/dvc-repos/smart_expt_dvc

PHASE2_DATA_DPATH=/flash/smart_data_dvc
PHASE2_EXPT_DPATH=/data/smart_expt

WORKDIR=$PHASE2_EXPT_DPATH/training/$HOSTNAME/$USER
DATASET_CODE=Drop6
KWCOCO_BUNDLE_DPATH=$PHASE2_DATA_DPATH/$DATASET_CODE

TRAIN_FPATH=$KWCOCO_BUNDLE_DPATH/data_train_split1.kwcoco.zip
VALI_FPATH=$KWCOCO_BUNDLE_DPATH/data_vali_split1.kwcoco.zip

EXPERIMENT_NAME=Drop4_BAS_S2L8_NoDecoderHetModel_TESTNOKEEP
DEFAULT_ROOT_DIR=$WORKDIR/$DATASET_CODE/runs/$EXPERIMENT_NAME

docker run \
    --user $(id -u):$(id -g) \
    --tty --shm-size=1g --memory=30g --cpus=6 --gpus='"device=3"' \
    --runtime=nvidia \
    --mount type=bind,source="$PHASE2_DATA_DPATH_SRC",target="$PHASE2_DATA_DPATH" \
    --mount type=bind,source="$PHASE2_EXPT_DPATH_SRC",target="$PHASE2_EXPT_DPATH" \
    --mount type=bind,source="$(pwd)/config_common.yaml",target="/config_common.yaml" \
    "feature/decoderless_heterogeneous_model" \
    conda run --no-capture-output -n watch \
    python -m watch.tasks.fusion fit \
        --config=/config_common.yaml \
        --model.init_args.name=$EXPERIMENT_NAME \
        --data.train_dataset="$TRAIN_FPATH" \
        --data.vali_dataset="$VALI_FPATH" \
        --data.num_workers=5 \
        --data.use_grid_cache=False \
        --data.verbose=true \
        --trainer.default_root_dir="$DEFAULT_ROOT_DIR" \
        --trainer.max_steps=100 \
        --trainer.accelerator="gpu" \
        --trainer.devices=1 \
        --trainer.strategy="ddp" \
        --trainer.precision=16
        # --print_config
