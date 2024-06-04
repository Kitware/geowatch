#!/bin/bash

# --------
# Polygon BAS tuning based on Point Based Model (Yardrat)

#export CUDA_VISIBLE_DEVICES="0,1,2,3"
export CUDA_VISIBLE_DEVICES="0"
DVC_DATA_DPATH=$(geowatch_dvc --tags='phase3_data' --hardware='ssd')
DVC_EXPT_DPATH=$(geowatch_dvc --tags='phase3_expt' --hardware='hdd')
echo "DVC_EXPT_DPATH = $DVC_EXPT_DPATH"
WORKDIR=$DVC_EXPT_DPATH/training/$HOSTNAME/$USER
DATASET_CODE=Drop8-ARA-Median10GSD-V1
KWCOCO_BUNDLE_DPATH=$DVC_DATA_DPATH/$DATASET_CODE
#TRAIN_FPATH=$KWCOCO_BUNDLE_DPATH/data_train_rawbands_pointannv1_split6_n043_486cc4af.kwcoco.zip
#VALI_FPATH=$KWCOCO_BUNDLE_DPATH/data_vali_rawbands_pointannv1_split6_n005_ba3fd747.kwcoco.zip
TRAIN_FPATH=$KWCOCO_BUNDLE_DPATH/data_train_rawbands_split6_n043_4a7fbf74.kwcoco.zip
VALI_FPATH=$KWCOCO_BUNDLE_DPATH/data_vali_rawbands_split6_n005_14403917.kwcoco.zip
CHANNELS="(L8,S2,PD,WV):(blue|green|red)"
EXPERIMENT_NAME=Drop8_ARA_Median10GSD_allsensors_polyann_frompoint_v1
DEFAULT_ROOT_DIR=$WORKDIR/$DATASET_CODE/runs/$EXPERIMENT_NAME
TARGET_LR=1e-4

PERTERB_SCALE=$(python -c "print($TARGET_LR * 0.003)")
WEIGHT_DECAY=$(python -c "print($TARGET_LR * 0.01)")
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

echo "
CUDA_VISIBLE_DEVICES = $CUDA_VISIBLE_DEVICES
DEVICES        = $DEVICES
ACCELERATOR    = $ACCELERATOR
STRATEGY       = $STRATEGY

DDP_WORKAROUND = $DDP_WORKAROUND

TARGET_LR      = $TARGET_LR
WEIGHT_DECAY   = $WEIGHT_DECAY
PERTERB_SCALE  = $PERTERB_SCALE
"


MAX_STEPS=10000
MAX_EPOCHS=121
TRAIN_BATCHES_PER_EPOCH=2666
ACCUMULATE_GRAD_BATCHES=16
BATCH_SIZE=12
TRAIN_ITEMS_PER_EPOCH=$(python -c "print($TRAIN_BATCHES_PER_EPOCH * $BATCH_SIZE)")


python -m geowatch.cli.experimental.recommend_size_adjustments \
    --MAX_STEPS=$MAX_STEPS \
    --MAX_EPOCHS=$MAX_EPOCHS \
    --BATCH_SIZE=$BATCH_SIZE \
    --ACCUMULATE_GRAD_BATCHES=$ACCUMULATE_GRAD_BATCHES \
    --TRAIN_BATCHES_PER_EPOCH="$TRAIN_BATCHES_PER_EPOCH" \
    --TRAIN_ITEMS_PER_EPOCH="$TRAIN_ITEMS_PER_EPOCH"


DDP_WORKAROUND=$DDP_WORKAROUND WATCH_GRID_WORKERS=4 python -m geowatch.tasks.fusion fit --config "
data:
  batch_size              : $BATCH_SIZE
  num_workers             : 6
  train_dataset           : $TRAIN_FPATH
  vali_dataset            : $VALI_FPATH
  time_steps              : 9
  chip_dims               : 196,196
  window_space_scale      : 10.0GSD
  input_space_scale       : 10.0GSD
  output_space_scale      : 10.0GSD
  channels                : '$CHANNELS'
  chip_overlap            : 0
  dist_weights            : True
  min_spacetime_weight    : 0.6
  neg_to_pos_ratio        : 1.0
  normalize_inputs        : 1024
  normalize_perframe      : false
  resample_invalid_frames : 3
  temporal_dropout        : 0.5
  time_sampling           : uniform-soft5-soft4-contiguous
  time_kernel             : '(-3y,-2.5y,-2y,-1.5y,-1y,0,1y,1.5y,2y,2.5y,3y)'
  upweight_centers        : true
  use_centered_positives  : True
  use_grid_positives      : true
  verbose                 : 1
  max_epoch_length        : $TRAIN_ITEMS_PER_EPOCH
  mask_low_quality        : false
  mask_samecolor_method   : null
  num_balance_trees      : 4
  balance_options :
      - attribute: old_has_class_of_interest
        weights:
           True:  0.6
           False: 0.4
      - attribute: region
model:
  class_path: watch.tasks.fusion.methods.MultimodalTransformer
  init_args:
    arch_name: smt_it_stm_p24
    attention_impl: exact
    attention_kwargs: null
    backbone_depth: null
    change_head_hidden: 6
    change_loss: cce
    class_head_hidden: 6
    class_loss: dicefocal
    class_weights: auto
    config: null
    continual_learning: false
    decoder: mlp
    decouple_resolution: false
    dropout: 0.1
    focal_gamma: 2.0
    global_change_weight: 0.0
    global_class_weight: 0.0
    global_saliency_weight: 1.0
    input_channels: null
    input_sensorchan: null
    learning_rate: 0.001
    lr_scheduler: CosineAnnealingLR
    modulate_class_weights: ''
    multimodal_reduce: learned_linear
    name: unnamed_model
    negative_change_weight: 0.01
    ohem_ratio: null
    optimizer: RAdam
    perterb_scale          : $PERTERB_SCALE
    positional_dims: 48
    positive_change_weight: 1
    rescale_nans: null
    saliency_head_hidden: 12
    saliency_loss: focal
    saliency_weights:
        foreground: 1.0
        background: 1.0
    stream_channels: 16
    tokenizer: linconv
lr_scheduler:
  class_path: torch.optim.lr_scheduler.OneCycleLR
  init_args:
    max_lr: $TARGET_LR
    total_steps: $MAX_STEPS
    anneal_strategy: cos
    pct_start: 0.3
optimizer:
  class_path: torch.optim.AdamW
  init_args:
    lr: $TARGET_LR
    weight_decay : $WEIGHT_DECAY
    betas:
      - 0.9
      - 0.99
trainer:
  accumulate_grad_batches: $ACCUMULATE_GRAD_BATCHES
  default_root_dir     : $DEFAULT_ROOT_DIR
  accelerator          : $ACCELERATOR
  devices              : $DEVICES
  strategy             : $STRATEGY
  check_val_every_n_epoch: 1
  enable_checkpointing: true
  enable_model_summary: true
  log_every_n_steps: 50
  logger: true
  max_epochs: $MAX_EPOCHS
  num_sanity_val_steps: 0
  limit_val_batches: 512
  limit_train_batches: $TRAIN_BATCHES_PER_EPOCH
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
    init: $DVC_EXPT_DPATH/models/fusion/Drop8-ARA-Median10GSD-V1/packages/Drop8_ARA_Median10GSD_allsensors_pointannsv1_v1/Drop8_ARA_Median10GSD_allsensors_pointannsv1_v1_epoch56_step4788.pt
"

export DVC_DATA_DPATH=$(geowatch_dvc --tags="phase3_data")
export DVC_EXPT_DPATH=$(geowatch_dvc --tags="phase3_expt")
cd "$DVC_EXPT_DPATH"
python -m geowatch.mlops.manager "status" --dataset_codes "Drop8-ARA-Median10GSD-V1"
python -m geowatch.mlops.manager "list checkpoints" --dataset_codes "Drop8-ARA-Median10GSD-V1"
python -m geowatch.mlops.manager "repackage checkpoints" --dataset_codes "Drop8-ARA-Median10GSD-V1"
python -m geowatch.mlops.manager "gather packages" --dataset_codes "Drop8-ARA-Median10GSD-V1"
python -m geowatch.mlops.manager "push packages" --dataset_codes "Drop8-ARA-Median10GSD-V1"

