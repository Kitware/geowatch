#!/bin/bash

# --------
# Point Based BAS V1 on yardrat

#export CUDA_VISIBLE_DEVICES="0,1,2,3"
export CUDA_VISIBLE_DEVICES="0"
DVC_DATA_DPATH=$(geowatch_dvc --tags='phase3_data' --hardware='ssd')
DVC_EXPT_DPATH=$(geowatch_dvc --tags='phase3_expt' --hardware='hdd')
echo "DVC_EXPT_DPATH = $DVC_EXPT_DPATH"
WORKDIR=$DVC_EXPT_DPATH/training/$HOSTNAME/$USER
DATASET_CODE=Drop8-ARA-Median10GSD-V1
KWCOCO_BUNDLE_DPATH=$DVC_DATA_DPATH/$DATASET_CODE
TRAIN_FPATH=$KWCOCO_BUNDLE_DPATH/data_train_rawbands_pointannv1_split6_n043_486cc4af.kwcoco.zip
VALI_FPATH=$KWCOCO_BUNDLE_DPATH/data_vali_rawbands_pointannv1_split6_n005_ba3fd747.kwcoco.zip
CHANNELS="(L8,S2,PD,WV):(blue|green|red)"
EXPERIMENT_NAME=Drop8_ARA_Median10GSD_allsensors_pointannsv1_v1
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
ACCUMULATE_GRAD_BATCHES=32
BATCH_SIZE=6
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
  num_workers             : 4
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
model:
  class_path: watch.tasks.fusion.methods.MultimodalTransformer
  init_args:
    arch_name: smt_it_stm_p16
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
    saliency_head_hidden: 6
    saliency_loss: focal
    saliency_weights: auto
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
  limit_val_batches: $TRAIN_BATCHES_PER_EPOCH
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
    init: $DVC_EXPT_DPATH/models/fusion/Drop8-Median10GSD-V1/packages/Drop8_Median10GSD_allsensors_scratch_V7/Drop8_Median10GSD_allsensors_scratch_V7_epoch187_step2632.pt
"

export DVC_DATA_DPATH=$(geowatch_dvc --tags="phase3_data")
export DVC_EXPT_DPATH=$(geowatch_dvc --tags="phase3_expt")
cd "$DVC_EXPT_DPATH"
python -m geowatch.mlops.manager "status" --dataset_codes "Drop8-ARA-Median10GSD-V1"
python -m geowatch.mlops.manager "list checkpoints" --dataset_codes "Drop8-ARA-Median10GSD-V1"
python -m geowatch.mlops.manager "repackage checkpoints" --dataset_codes "Drop8-ARA-Median10GSD-V1"
python -m geowatch.mlops.manager "gather packages" --dataset_codes "Drop8-ARA-Median10GSD-V1"
python -m geowatch.mlops.manager "push packages" --dataset_codes "Drop8-ARA-Median10GSD-V1"

# Point Based BAS V2 on yardrat

#export CUDA_VISIBLE_DEVICES="0,1,2,3"
export CUDA_VISIBLE_DEVICES="0"
DVC_DATA_DPATH=$(geowatch_dvc --tags='phase3_data' --hardware='ssd')
DVC_EXPT_DPATH=$(geowatch_dvc --tags='phase3_expt' --hardware='hdd')
echo "DVC_EXPT_DPATH = $DVC_EXPT_DPATH"
WORKDIR=$DVC_EXPT_DPATH/training/$HOSTNAME/$USER
DATASET_CODE=Drop8-ARA-Median10GSD-V1
KWCOCO_BUNDLE_DPATH=$DVC_DATA_DPATH/$DATASET_CODE
TRAIN_FPATH=$KWCOCO_BUNDLE_DPATH/data_train_rawbands_pointannv1_split6_n043_486cc4af.kwcoco.zip
VALI_FPATH=$KWCOCO_BUNDLE_DPATH/data_vali_rawbands_pointannv1_split6_n005_ba3fd747.kwcoco.zip
CHANNELS="(L8,S2,PD,WV):(blue|green|red)"
EXPERIMENT_NAME=Drop8_ARA_Median10GSD_allsensors_pointannsv1_v2
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


MAX_STEPS=20000
MAX_EPOCHS=241
TRAIN_BATCHES_PER_EPOCH=2666
ACCUMULATE_GRAD_BATCHES=32
BATCH_SIZE=6
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
  num_workers             : 4
  train_dataset           : $TRAIN_FPATH
  vali_dataset            : $VALI_FPATH
  time_steps              : 9
  chip_dims               : 196,196
  window_space_scale      : 10.0GSD
  input_space_scale       : 10.0GSD
  output_space_scale      : 10.0GSD
  channels                : '$CHANNELS'
  chip_overlap            : 0
  dist_weights            : 0
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
model:
  class_path: watch.tasks.fusion.methods.MultimodalTransformer
  init_args:
    arch_name: smt_it_stm_p16
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
    saliency_head_hidden: 6
    saliency_loss: focal
    saliency_weights: auto
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
  limit_val_batches: $TRAIN_BATCHES_PER_EPOCH
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
    init: $DVC_EXPT_DPATH/models/fusion/Drop8-Median10GSD-V1/packages/Drop8_Median10GSD_allsensors_scratch_V7/Drop8_Median10GSD_allsensors_scratch_V7_epoch187_step2632.pt
"


export DVC_DATA_DPATH=$(geowatch_dvc --tags="phase3_data")
export DVC_EXPT_DPATH=$(geowatch_dvc --tags="phase3_expt")
cd "$DVC_EXPT_DPATH"
python -m geowatch.mlops.manager "status" --dataset_codes "Drop8-ARA-Median10GSD-V1"
python -m geowatch.mlops.manager "list packages" --dataset_codes "Drop8-ARA-Median10GSD-V1"
python -m geowatch.mlops.manager "list checkpoints" --dataset_codes "Drop8-ARA-Median10GSD-V1"
python -m geowatch.mlops.manager "repackage checkpoints" --dataset_codes "Drop8-ARA-Median10GSD-V1"
python -m geowatch.mlops.manager "gather packages" --dataset_codes "Drop8-ARA-Median10GSD-V1"
python -m geowatch.mlops.manager "push packages" --dataset_codes "Drop8-ARA-Median10GSD-V1"



export DVC_EXPT_DPATH=$(geowatch_dvc --tags="phase3_expt")
python -m geowatch.mlops.manager "list packages" --expt_dvc_dpath="$DVC_EXPT_DPATH" --dataset_codes "Drop8-ARA-Median10GSD-V1"

# Point based mlops evaluation
export DVC_DATA_DPATH=$(geowatch_dvc --tags='phase3_data' --hardware=ssd)
export DVC_EXPT_DPATH=$(geowatch_dvc --tags='phase3_expt' --hardware=auto)
TRUTH_DPATH=$DVC_DATA_DPATH/annotations/drop8-v1
MLOPS_NAME=_preeval22_point_bas_grid
MLOPS_DPATH=$DVC_EXPT_DPATH/$MLOPS_NAME

MODEL_SHORTLIST="
- $DVC_EXPT_DPATH/models/fusion/Drop8-ARA-Median10GSD-V1/packages/Drop8_ARA_Median10GSD_allsensors_pointannsv1_v1/Drop8_ARA_Median10GSD_allsensors_pointannsv1_v1_epoch56_step4788.pt
- $DVC_EXPT_DPATH/models/fusion/Drop8-ARA-Median10GSD-V1/packages/Drop8_ARA_Median10GSD_allsensors_pointannsv1_v1/Drop8_ARA_Median10GSD_allsensors_pointannsv1_v1_epoch119_step10001.pt
- $DVC_EXPT_DPATH/models/fusion/Drop8-ARA-Median10GSD-V1/packages/Drop8_ARA_Median10GSD_allsensors_pointannsv1_v1/Drop8_ARA_Median10GSD_allsensors_pointannsv1_v1_epoch62_step5292.pt
- $DVC_EXPT_DPATH/models/fusion/Drop8-ARA-Median10GSD-V1/packages/Drop8_ARA_Median10GSD_allsensors_pointannsv1_v1/Drop8_ARA_Median10GSD_allsensors_pointannsv1_v1_epoch61_step5208.pt
- $DVC_EXPT_DPATH/models/fusion/Drop8-ARA-Median10GSD-V1/packages/Drop8_ARA_Median10GSD_allsensors_pointannsv1_v1/Drop8_ARA_Median10GSD_allsensors_pointannsv1_v1_epoch60_step5124.pt
- $DVC_EXPT_DPATH/models/fusion/Drop8-ARA-Median10GSD-V1/packages/Drop8_ARA_Median10GSD_allsensors_pointannsv1_v1/Drop8_ARA_Median10GSD_allsensors_pointannsv1_v1_epoch66_step5628.pt
- $DVC_EXPT_DPATH/models/fusion/Drop8-ARA-Median10GSD-V1/packages/Drop8_ARA_Median10GSD_allsensors_pointannsv1_v2/Drop8_ARA_Median10GSD_allsensors_pointannsv1_v2_epoch5_step504.pt
- $DVC_EXPT_DPATH/models/fusion/Drop8-ARA-Median10GSD-V1/packages/Drop8_ARA_Median10GSD_allsensors_pointannsv1_v2/Drop8_ARA_Median10GSD_allsensors_pointannsv1_v2_epoch3_step336.pt
- $DVC_EXPT_DPATH/models/fusion/Drop8-ARA-Median10GSD-V1/packages/Drop8_ARA_Median10GSD_allsensors_pointannsv1_v2/Drop8_ARA_Median10GSD_allsensors_pointannsv1_v2_epoch144_step12180.pt
- $DVC_EXPT_DPATH/models/fusion/Drop8-ARA-Median10GSD-V1/packages/Drop8_ARA_Median10GSD_allsensors_pointannsv1_v2/Drop8_ARA_Median10GSD_allsensors_pointannsv1_v2_epoch2_step252.pt
- $DVC_EXPT_DPATH/models/fusion/Drop8-ARA-Median10GSD-V1/packages/Drop8_ARA_Median10GSD_allsensors_pointannsv1_v2/Drop8_ARA_Median10GSD_allsensors_pointannsv1_v2_epoch6_step588.pt
- $DVC_EXPT_DPATH/models/fusion/Drop8-ARA-Median10GSD-V1/packages/Drop8_ARA_Median10GSD_allsensors_pointannsv1_v2/Drop8_ARA_Median10GSD_allsensors_pointannsv1_v2_epoch149_step12600.pt
- $DVC_EXPT_DPATH/models/fusion/Drop8-ARA-Median10GSD-V1/packages/Drop8_ARA_Median10GSD_allsensors_pointannsv1_v2/Drop8_ARA_Median10GSD_allsensors_pointannsv1_v2_epoch118_step9996.pt
- $DVC_EXPT_DPATH/models/fusion/Drop8-ARA-Median10GSD-V1/packages/Drop8_ARA_Median10GSD_allsensors_pointannsv1_v2/Drop8_ARA_Median10GSD_allsensors_pointannsv1_v2_epoch147_step12432.pt
- $DVC_EXPT_DPATH/models/fusion/Drop8-ARA-Median10GSD-V1/packages/Drop8_ARA_Median10GSD_allsensors_pointannsv1_v2/Drop8_ARA_Median10GSD_allsensors_pointannsv1_v2_epoch119_step10080.pt
- $DVC_EXPT_DPATH/models/fusion/Drop8-ARA-Median10GSD-V1/packages/Drop8_ARA_Median10GSD_allsensors_pointannsv1_v2/Drop8_ARA_Median10GSD_allsensors_pointannsv1_v2_epoch4_step420.pt
- $DVC_EXPT_DPATH/models/fusion/Drop8-ARA-Median10GSD-V1/packages/Drop8_ARA_Median10GSD_allsensors_pointannsv1_v2/Drop8_ARA_Median10GSD_allsensors_pointannsv1_v2_epoch1_step168.pt
"


MODEL_SHORTLIST="
- $DVC_EXPT_DPATH/models/fusion/Drop8-ARA-Median10GSD-V1/packages/Drop8_ARA_Median10GSD_allsensors_pointannsv1_v1/Drop8_ARA_Median10GSD_allsensors_pointannsv1_v1_epoch56_step4788.pt
- $DVC_EXPT_DPATH/models/fusion/Drop8-ARA-Median10GSD-V1/packages/Drop8_ARA_Median10GSD_allsensors_pointannsv1_v1/Drop8_ARA_Median10GSD_allsensors_pointannsv1_v1_epoch60_step5124.pt
- $DVC_EXPT_DPATH/models/fusion/Drop8-ARA-Median10GSD-V1/packages/Drop8_ARA_Median10GSD_allsensors_pointannsv1_v2/Drop8_ARA_Median10GSD_allsensors_pointannsv1_v2_epoch144_step12180.pt
"

mkdir -p "$MLOPS_DPATH"
echo "$MODEL_SHORTLIST" > "$MLOPS_DPATH/shortlist.yaml"

cat "$MLOPS_DPATH/shortlist.yaml"

# FIXME: make sdvc request works with YAML lists of models
# sdvc request --verbose=3 "$MLOPS_DPATH/shortlist.yaml"

geowatch schedule --params="
    pipeline: bas

    matrix:
        bas_pxl.package_fpath: $MLOPS_DPATH/shortlist.yaml

        bas_pxl.test_dataset:
            - $DVC_DATA_DPATH/Drop8-ARA-Median10GSD-V1/KR_R002/imganns-KR_R002-rawbands.kwcoco.zip
            - $DVC_DATA_DPATH/Drop8-ARA-Median10GSD-V1/CN_C000/imganns-CN_C000-rawbands.kwcoco.zip
            - $DVC_DATA_DPATH/Drop8-ARA-Median10GSD-V1/KW_C001/imganns-KW_C001-rawbands.kwcoco.zip
            - $DVC_DATA_DPATH/Drop8-ARA-Median10GSD-V1/CO_C001/imganns-CO_C001-rawbands.kwcoco.zip
        bas_pxl.chip_overlap: 0.3
        bas_pxl.chip_dims: auto
        bas_pxl.time_span: auto
        bas_pxl.time_sampling: soft4
        bas_poly.thresh:
            - 0.36
            - 0.37
            - 0.375
            - 0.38
            - 0.39
            - 0.40
        bas_poly.inner_window_size: 1y
        bas_poly.inner_agg_fn: mean
        bas_poly.norm_ord: inf
        bas_poly.polygon_simplify_tolerance: 1
        bas_poly.agg_fn: probs
        bas_poly.time_thresh:
            - 0.85
            - 0.8
            - 0.75
        bas_poly.time_pad_after:
            - 0 months
            - 3 months
            - 12 months
        bas_poly.resolution: 10GSD
        bas_poly.moving_window_size: null
        bas_poly.poly_merge_method: 'v2'
        bas_poly.min_area_square_meters: 7200
        bas_poly.max_area_square_meters: 8000000
        bas_poly.boundary_region: $TRUTH_DPATH/region_models
        bas_poly_eval.true_site_dpath: $TRUTH_DPATH/site_models
        bas_poly_eval.true_region_dpath: $TRUTH_DPATH/region_models
        bas_pxl.enabled: 1
        bas_pxl_eval.enabled: 0
        bas_poly_viz.enabled: 0
        bas_poly.enabled: 1
        bas_poly_eval.enabled: 1
    include:
        - bas_poly.time_pad_after: 0 months
          bas_poly.time_pad_before: 0 months
        - bas_poly.time_pad_after: 3 months
          bas_poly.time_pad_before: 3 months
        - bas_poly.time_pad_after: 12 months
          bas_poly.time_pad_before: 12 months
    " \
    --root_dpath="$MLOPS_DPATH" \
    --devices="0,1,2,3" --tmux_workers=8 \
    --backend=tmux --queue_name "$MLOPS_NAME" \
    --skip_existing=1 \
    --run=1

# Point based mlops evaluation
export DVC_DATA_DPATH=$(geowatch_dvc --tags='phase3_data' --hardware=ssd)
export DVC_EXPT_DPATH=$(geowatch_dvc --tags='phase3_expt' --hardware=auto)
TRUTH_DPATH=$DVC_DATA_DPATH/annotations/drop8-v1
MLOPS_NAME=_preeval22_point_bas_grid
MLOPS_DPATH=$DVC_EXPT_DPATH/$MLOPS_NAME
DVC_EXPT_DPATH=$(geowatch_dvc --tags='phase3_expt' --hardware=auto)
echo "DVC_EXPT_DPATH = $DVC_EXPT_DPATH"
python -m geowatch.mlops.aggregate \
    --pipeline=bas \
    --target "
        - $MLOPS_DPATH
    " \
    --output_dpath="$MLOPS_DPATH/aggregate" \
    --resource_report=0 \
    --eval_nodes="
        - bas_poly_eval
        #- bas_pxl_eval
    " \
    --plot_params="
        enabled: 0
        stats_ranking: 0
        min_variations: 1
        params_of_interest:
            - params.bas_poly.thresh
            - resolved_params.bas_pxl.channels
    " \
    --stdout_report="
        top_k: 100
        per_group: 1
        macro_analysis: 0
        analyze: 0
        print_models: True
        reference_region: final
        concise: 1
        show_csv: 0
    " \
    --rois="KR_R002,CN_C000,KW_C001,CO_C001"
    #" --rois="KR_R002,CN_C000"
    #--rois="KR_R002"
    #--rois="CN_C000"


# --------
# Point Based BAS V3 on horologic (vinnie)

#export CUDA_VISIBLE_DEVICES="0,1,2,3"
export CUDA_VISIBLE_DEVICES="0"
DVC_DATA_DPATH=$(geowatch_dvc --tags='phase3_data' --hardware='ssd')
DVC_EXPT_DPATH=$(geowatch_dvc --tags='phase3_expt' --hardware='hdd')
echo "DVC_EXPT_DPATH = $DVC_EXPT_DPATH"
WORKDIR=$DVC_EXPT_DPATH/training/$HOSTNAME/$USER
DATASET_CODE=Drop8-ARA-Median10GSD-V1
KWCOCO_BUNDLE_DPATH=$DVC_DATA_DPATH/$DATASET_CODE
TRAIN_FPATH=$KWCOCO_BUNDLE_DPATH/data_train_rawbands_pointannv1_split6_n045_d1bf6e67.kwcoco.zip
VALI_FPATH=$KWCOCO_BUNDLE_DPATH/data_vali_rawbands_pointannv1_split6_n005_4f11856c.kwcoco.zip
CHANNELS="(L8,S2,PD,WV):(blue|green|red)"
EXPERIMENT_NAME=Drop8_ARA_Median10GSD_allsensors_pointannsv1_v3
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


MAX_STEPS=20000
MAX_EPOCHS=241
TRAIN_BATCHES_PER_EPOCH=2666
ACCUMULATE_GRAD_BATCHES=32
BATCH_SIZE=6
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
  num_workers             : 4
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
model:
  class_path: watch.tasks.fusion.methods.MultimodalTransformer
  init_args:
    arch_name: smt_it_stm_p16
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
    saliency_head_hidden: 6
    saliency_loss: focal
    saliency_weights: auto
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
  limit_val_batches: $TRAIN_BATCHES_PER_EPOCH
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
