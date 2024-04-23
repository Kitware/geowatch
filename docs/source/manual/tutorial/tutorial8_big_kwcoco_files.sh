#!/bin/bash
__doc__="

Big KWCoco Files & SQL Views
============================

"

# For those windows folks:
if [[ "$(uname -a)" == "MINGW"* ]]; then
    echo "detected windows with mingw"
    export HOME=$USERPROFILE
    export USER=$USERNAME
fi


DVC_DATA_DPATH=$HOME/data/dvc-repos/toy_data_dvc
DVC_EXPT_DPATH=$HOME/data/dvc-repos/toy_expt_dvc

mkdir -p "$DVC_DATA_DPATH"
mkdir -p "$DVC_EXPT_DPATH"

# Define the names of the kwcoco files to generate
TRAIN_FPATH=$DVC_DATA_DPATH/vidshapes_rgb_big_train/data.kwcoco.json
VALI_FPATH=$DVC_DATA_DPATH/vidshapes_rgb_big_vali/data.kwcoco.json

# Generate toy datasets using the "kwcoco toydata" tool
kwcoco toydata vidshapes800-frames5-amazon --bundle_dpath "$DVC_DATA_DPATH"/vidshapes_rgb_big_train
kwcoco toydata vidshapes10-frames5-amazon --bundle_dpath "$DVC_DATA_DPATH"/vidshapes_rgb_big_vali

kwcoco validate "$TRAIN_FPATH"
kwcoco validate "$VALI_FPATH"


WORKDIR=$DVC_EXPT_DPATH/training/$HOSTNAME/$USER
EXPERIMENT_NAME=BigToyRGB_Demo_V002
DATASET_CODE=BigToyRGB
DEFAULT_ROOT_DIR=$WORKDIR/$DATASET_CODE/runs/$EXPERIMENT_NAME
CHANNELS="r|g|b"
MAX_STEPS=32

TARGET_LR=3e-4
WEIGHT_DECAY=$(python -c "print($TARGET_LR * 0.01)")

# To ensure anyone can run this tutorial we default to a "cpu" accelerator.
# However, if you have one or more GPUs available you should modify the
# ACCELERATOR and CUDA_VISIBLE_DEVICES variables based on your hardware


export ACCELERATOR="${ACCELERATOR:-cpu}"
export ACCELERATOR=cpu
# Uncomment if you are using a GPU, and set CUDA_VISIBLE_DEVICES
# to a comma separated list of the GPU #'s you want to use.
#export ACCELERATOR=gpu
#export CUDA_VISIBLE_DEVICES=0
 #export CUDA_VISIBLE_DEVICES=0,1

DEVICES=$(python -c "if 1:
    import os
    if os.environ.get('ACCELERATOR', '') == 'cpu':
        print('1')
    else:
        n = len(os.environ.get('CUDA_VISIBLE_DEVICES', '').split(','))
        print(','.join(list(map(str, range(n)))) + ',')
")
STRATEGY=$(python -c "if 1:
    import os
    if os.environ.get('ACCELERATOR', '') == 'cpu':
        print('auto')
    else:
        n = len(os.environ.get('CUDA_VISIBLE_DEVICES', '').split(','))
        print('ddp' if n > 1 else 'auto')
")
DDP_WORKAROUND=$(python -c "if 1:
    import os
    n = len(os.environ.get('CUDA_VISIBLE_DEVICES', '').split(','))
    print(int(n > 1))
")
echo "
CUDA_VISIBLE_DEVICES = $CUDA_VISIBLE_DEVICES
ACCELERATOR = $ACCELERATOR
DEVICES = $DEVICES
STRATEGY = $STRATEGY
DDP_WORKAROUND = $DDP_WORKAROUND
"

# The following parameters are related and impose constraints on each other.
MAX_STEPS=400
MAX_EPOCHS=20
TRAIN_BATCHES_PER_EPOCH=20
ACCUMULATE_GRAD_BATCHES=1
BATCH_SIZE=2
TRAIN_ITEMS_PER_EPOCH=$(python -c "print($TRAIN_BATCHES_PER_EPOCH * $BATCH_SIZE)")
# Thus we expose this recommendation script to suggest changes to that satisfy
# the constraints. (Its ok if it isnt perfect)
python -m geowatch.cli.experimental.recommend_size_adjustments \
    --MAX_STEPS=$MAX_STEPS \
    --MAX_EPOCHS=$MAX_EPOCHS \
    --BATCH_SIZE=$BATCH_SIZE \
    --ACCUMULATE_GRAD_BATCHES=$ACCUMULATE_GRAD_BATCHES \
    --TRAIN_BATCHES_PER_EPOCH="$TRAIN_BATCHES_PER_EPOCH" \
    --TRAIN_ITEMS_PER_EPOCH="$TRAIN_ITEMS_PER_EPOCH"


# Now that we have important variables defined in the environment we can run
# the actual fit command.
DDP_WORKAROUND=$DDP_WORKAROUND python -m geowatch.tasks.fusion fit --config "
    data:
        num_workers          : 4
        train_dataset        : $TRAIN_FPATH
        vali_dataset         : $VALI_FPATH
        channels             : '$CHANNELS'
        time_steps           : 5
        chip_dims            : 128
        batch_size           : $BATCH_SIZE
        sqlview              : 'postgresql'
    model:
        class_path: MultimodalTransformer
        init_args:
            name        : $EXPERIMENT_NAME
            arch_name   : smt_it_stm_p8
            window_size : 8
            dropout     : 0.1
            global_saliency_weight: 1.0
            global_class_weight:    1.0
            global_change_weight:   0.0
    lr_scheduler:
      class_path: torch.optim.lr_scheduler.OneCycleLR
      init_args:
        max_lr: $TARGET_LR
        total_steps: $MAX_STEPS
        anneal_strategy: cos
        pct_start: 0.05
    optimizer:
      class_path: torch.optim.Adam
      init_args:
        lr: $TARGET_LR
        weight_decay: $WEIGHT_DECAY
        betas:
          - 0.9
          - 0.99
    trainer:
      accumulate_grad_batches: $ACCUMULATE_GRAD_BATCHES
      default_root_dir     : $DEFAULT_ROOT_DIR
      accelerator          : $ACCELERATOR
      devices              : $DEVICES
      strategy             : $STRATEGY
      limit_train_batches  : $TRAIN_BATCHES_PER_EPOCH
      check_val_every_n_epoch: 1
      enable_checkpointing: true
      enable_model_summary: true
      log_every_n_steps: 5
      logger: true
      max_steps: $MAX_STEPS
      num_sanity_val_steps: 0
    initializer:
        init: noop
"
