#!/bin/bash
__notes__="
Optimizing CIFAR:

* Testing if a ramdisk helps throughput

sudo mkdir -p /mnt/tmpfs
sudo mount -o size=16G -t tmpfs none /mnt/tmpfs

kwcoco grab cifar10 --dpath /mnt/tmpfs
"
kwcoco grab cifar10

export CUDA_VISIBLE_DEVICES=0
DVC_EXPT_DPATH=$HOME/data/dvc-repos/cifar10
WORKDIR=$DVC_EXPT_DPATH/training/$HOSTNAME/$USER

DATASET_CODE=cifar10
#CIFAR_ROOT=/mnt/tmpfs/
CIFAR_ROOT=$HOME/.cache/kwcoco/data

TRAIN_FPATH=$CIFAR_ROOT/cifar10-train/cifar10-train.kwcoco.json
VALI_FPATH=$CIFAR_ROOT/cifar10-test/cifar10-test.kwcoco.json

inspect_kwcoco_files(){
    kwcoco stats "$TRAIN_FPATH" "$VALI_FPATH"
    kwcoco validate --corrupted "$TRAIN_FPATH" "$VALI_FPATH"
    kwcoco info "$VALI_FPATH" -g 1
    kwcoco info "$VALI_FPATH" -v 1
    #kwcoco info "$VALI_FPATH" -a 1
    #geowatch stats "$TRAIN_FPATH" "$VALI_FPATH"
}
#inspect_kwcoco_files
EXPERIMENT_NAME="cifar10_simple_resnet_v3"

CHANNELS="auto"
DEFAULT_ROOT_DIR=$WORKDIR/$DATASET_CODE/runs/$EXPERIMENT_NAME
TARGET_LR=1e-2
WEIGHT_DECAY=$(python -c "print($TARGET_LR * 0.01)")
#PERTERB_SCALE=$(python -c "print($TARGET_LR * 0.003)")
DEVICES=$(python -c "if 1:
    import os
    n = len(os.environ.get('CUDA_VISIBLE_DEVICES', '').split(','))
    print(','.join(list(map(str, range(n)))) + ',')
")
ACCELERATOR=gpu
STRATEGY=$(python -c "if 1:
    import os
    n = len(os.environ.get('CUDA_VISIBLE_DEVICES', '').split(','))
    print('ddp' if n > 1 else 'auto')
")
DDP_WORKAROUND=$(python -c "if 1:
    import os
    n = len(os.environ.get('CUDA_VISIBLE_DEVICES', '').split(','))
    print(int(n > 1))
")
echo "DEVICES = $DEVICES"
echo "DDP_WORKAROUND = $DDP_WORKAROUND"
echo "WEIGHT_DECAY = $WEIGHT_DECAY"


MAX_STEPS=60000
MAX_EPOCHS=3
TRAIN_ITEMS_PER_EPOCH=50000
VALI_ITEMS_PER_EPOCH=10000
ACCUMULATE_GRAD_BATCHES=1
BATCH_SIZE=1000
TRAIN_BATCHES_PER_EPOCH=$(python -c "print($TRAIN_ITEMS_PER_EPOCH // $BATCH_SIZE)")
VALI_BATCHES_PER_EPOCH=$(python -c "print($VALI_ITEMS_PER_EPOCH // $BATCH_SIZE)")
echo "TRAIN_ITEMS_PER_EPOCH = $TRAIN_ITEMS_PER_EPOCH"
echo "VALI_BATCHES_PER_EPOCH = $VALI_BATCHES_PER_EPOCH"

python -m geowatch.cli.experimental.recommend_size_adjustments \
    --MAX_STEPS=$MAX_STEPS \
    --MAX_EPOCHS=$MAX_EPOCHS \
    --BATCH_SIZE=$BATCH_SIZE \
    --ACCUMULATE_GRAD_BATCHES=$ACCUMULATE_GRAD_BATCHES \
    --TRAIN_BATCHES_PER_EPOCH="$TRAIN_BATCHES_PER_EPOCH" \
    --TRAIN_ITEMS_PER_EPOCH="$TRAIN_ITEMS_PER_EPOCH"


rm -rf "$DEFAULT_ROOT_DIR"

# Find the most recent checkpoint (TODO add utility for this)
PREV_CHECKPOINT_TEXT=$(python -m geowatch.cli.experimental.find_recent_checkpoint --default_root_dir="$DEFAULT_ROOT_DIR")
echo "PREV_CHECKPOINT_TEXT = $PREV_CHECKPOINT_TEXT"
if [[ "$PREV_CHECKPOINT_TEXT" == "None" ]]; then
    PREV_CHECKPOINT_ARGS=()
else
    PREV_CHECKPOINT_ARGS=(--ckpt_path "$PREV_CHECKPOINT_TEXT")
fi
echo "${PREV_CHECKPOINT_ARGS[@]}"


ulimit -n 1000000


DDP_WORKAROUND=1
LINE_PROFILE=0 DDP_WORKAROUND=$DDP_WORKAROUND python -m geowatch.tasks.fusion fit --config "
data:
    select_videos          : $SELECT_VIDEOS
    num_workers            : 2
    train_dataset          : $TRAIN_FPATH
    vali_dataset           : $VALI_FPATH
    window_dims            : '32,32'
    time_steps             : 0
    time_sampling          : uniform
    #time_kernel           : '[0.0s,]'
    window_resolution     : 1.0
    input_resolution      : 1.0
    output_resolution     : 1.0
    neg_to_pos_ratio       : 1.0
    batch_size             : $BATCH_SIZE
    normalize_perframe     : false
    normalize_peritem      : false
    max_items_per_epoch    : $TRAIN_ITEMS_PER_EPOCH
    channels               : '$CHANNELS'
    min_spacetime_weight   : 0.6
    temporal_dropout_rate  : 0.0
    channel_dropout_rate   : 0.0
    modality_dropout_rate  : 0.0
    temporal_dropout       : 0.0
    channel_dropout        : 0.0
    modality_dropout       : 0.0
    mask_low_quality       : False
    mask_samecolor_method  : None
    observable_threshold   : 0.0
    quality_threshold      : 0.0
    weight_dilate          : 0
    dist_weights           : False
    upweight_centers       : False
    downweight_nan_regions : False
    use_centered_positives : True
    use_grid_positives     : False
    use_grid_negatives     : False
    normalize_inputs       : 50000

    augment_time_resample_rate : 0
    augment_space_shift_rate : 0
    balance_areas          : false
    reduce_item_size       : true
    balance_options        : sequential_without_replacement

model:
    class_path: geowatch.tasks.fusion.methods.torchvision_nets.Resnet50
    init_args:
        heads:
            - name: nonlocal_class
              type: mlp
              hidden_channels: []
              out_channels: 11
              dropout: 0.1
              norm: batch
              loss:
                  type: focal
                  gamma: 2.0
              head_weight: 1.0
optimizer:
    class_path: torch.optim.AdamW
    init_args:
        lr           : $TARGET_LR
        weight_decay : $WEIGHT_DECAY
lr_scheduler:
  class_path: torch.optim.lr_scheduler.OneCycleLR
  init_args:
    max_lr: $TARGET_LR
    total_steps: $MAX_STEPS
    anneal_strategy: cos
    pct_start: 0.3
trainer:
    accumulate_grad_batches: $ACCUMULATE_GRAD_BATCHES
    default_root_dir     : $DEFAULT_ROOT_DIR
    accelerator          : $ACCELERATOR
    devices              : $DEVICES
    strategy             : $STRATEGY
    limit_train_batches  : $TRAIN_BATCHES_PER_EPOCH
    limit_val_batches    : $VALI_BATCHES_PER_EPOCH
    #limit_train_batches : 5
    #limit_val_batches   : 3
    log_every_n_steps    : 50
    check_val_every_n_epoch: 1
    enable_checkpointing: true
    enable_model_summary: true
    num_sanity_val_steps : 0
    # profiler: advanced
    # profiler: simple
    max_epochs: $MAX_EPOCHS
    #max_epochs: 1
    callbacks:
        - class_path: pytorch_lightning.callbacks.ModelCheckpoint
          init_args:
              monitor: val_loss
              mode: min
              save_top_k: 5
              filename: '{epoch:04d}-{step:06d}-{val_loss:.3f}.ckpt'
              save_last: true

torch_globals:
    float32_matmul_precision: auto

initializer:
    init: noop
"

#"${PREV_CHECKPOINT_ARGS[@]}"



python -m geowatch.tasks.fusion fit --config "
data:
    train_dataset          : $TRAIN_FPATH
    vali_dataset           : $VALI_FPATH
    window_dims            : '196,196'
    time_kernel            : [-1year, 0.0year, +1year]
    fixed_resolution       : 10GSD
    normalize_perframe     : true
    channels               : 'red|green|blue|nir'
    use_centered_positives : True
    use_grid_positives     : False
    use_grid_negatives     : True
    balance_options :
        - attribute: region
        - attribute: contains_annotation
          weights:
              False: 0.5
              True: 0.5
        - attribute: phases
          default_weight: 0.1
          weights:
              'No Activity': 0.05
              'Site Preparation': 0.8
              'Active Construction': 0.1
              'Post Construction': 0.05
model:
  class_path: watch.tasks.fusion.methods.MultimodalTransformer
  init_args:
    arch_name: smt_it_stm_p16
optimizer:
    class_path: torch.optim.SGD
    init_args:
        lr           : 1e-4
initializer:
    init: path/to/partial_pretrained_state.py
"


