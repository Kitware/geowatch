#!/bin/bash

PHASE2_DATA_DPATH=/flash/smart_data_dvc
PHASE2_EXPT_DPATH=$HOME/data/dvc-repos/smart_expt_dvc

WORKDIR=$PHASE2_EXPT_DPATH/training/$HOSTNAME/$USER
# DATASET_CODE=Aligned-Drop4-2022-08-08-TA1-S2-WV-PD-ACC
# DATASET_CODE=Aligned-Drop4-2022-08-08-TA1-S2-L8-ACC
DATASET_CODE=Drop6
KWCOCO_BUNDLE_DPATH=$PHASE2_DATA_DPATH/$DATASET_CODE

TRAIN_FPATH=$KWCOCO_BUNDLE_DPATH/data_train_split1.kwcoco.zip
VALI_FPATH=$KWCOCO_BUNDLE_DPATH/data_vali_split1.kwcoco.zip
TEST_FPATH=$KWCOCO_BUNDLE_DPATH/data_vali_split1.kwcoco.zip

EXPERIMENT_NAME=Drop4_BAS_S2L8_UNet
DEFAULT_ROOT_DIR=$WORKDIR/$DATASET_CODE/runs/$EXPERIMENT_NAME
python -m watch.tasks.fusion fit \
    --config=config_common.yaml \
    --ckpt_path=$WORKDIR/Aligned-Drop4-2022-08-08-TA1-S2-L8-ACC/runs/Drop4_BAS_S2L8_UNet/lightning_logs/version_2/checkpoints/epoch=479-step=12000.ckpt \
    --model.init_args.name=$EXPERIMENT_NAME \
    --data.train_dataset="$TRAIN_FPATH" \
    --data.vali_dataset="$VALI_FPATH" \
    --data.num_workers=0 \
    --trainer.default_root_dir="$DEFAULT_ROOT_DIR" \
    --trainer.accelerator="gpu" \
    --trainer.devices=1 \
    --trainer.precision=16
    # --trainer.detect_anomaly=true \
