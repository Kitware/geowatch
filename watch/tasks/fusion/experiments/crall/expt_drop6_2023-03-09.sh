#!/bin/bash

PHASE2_DATA_DPATH_SSD=$(geowatch_dvc --tags="phase2_data" --hardware="auto")
cd "$PHASE2_DATA_DPATH_SSD/Drop6"
python -m watch.cli.prepare_splits \
    --base_fpath "combo_imganns-*_L.kwcoco.json" \
    --suffix=fixquant \
    --constructive_mode=True


export CUDA_VISIBLE_DEVICES=0
DVC_DATA_DPATH=$(geowatch_dvc --tags='phase2_data' --hardware='auto')
DVC_EXPT_DPATH=$(geowatch_dvc --tags='phase2_expt' --hardware='auto')
echo "DVC_EXPT_DPATH = $DVC_EXPT_DPATH"
WORKDIR=$DVC_EXPT_DPATH/training/$HOSTNAME/$USER
DATASET_CODE=Drop6
KWCOCO_BUNDLE_DPATH=$DVC_DATA_DPATH/$DATASET_CODE
TRAIN_FPATH=$KWCOCO_BUNDLE_DPATH/data_train_fixquant_split2.kwcoco.zip
VALI_FPATH=$KWCOCO_BUNDLE_DPATH/data_vali_fixquant_split2.kwcoco.zip
CHANNELS="(L8,S2,PD):(blue|green|red|nir),(WV):(blue|green|red),(WV,WV1):pan,(S2):(water|forest|field|impervious|barren|landcover_hidden.0:32)"
EXPERIMENT_NAME=Drop6_BAS_scratch_landcover_10GSD_split2_V33
DEFAULT_ROOT_DIR=$WORKDIR/$DATASET_CODE/runs/$EXPERIMENT_NAME
TARGET_LR=1e-4
MAX_STEPS=80000
WATCH_GRID_WORKERS=4 python -m watch.tasks.fusion fit --config "
seed_everything: 1104562820
data:
  batch_size              : 4
  num_workers             : 4
  train_dataset           : $TRAIN_FPATH
  vali_dataset            : $VALI_FPATH
  time_steps              : 7
  chip_dims               : 128
  window_space_scale      : 10.0GSD
  input_space_scale       : 10.0GSD
  output_space_scale      : 300.0GSD
  channels                : '$CHANNELS'
  chip_overlap            : 0
  dist_weights            : 0
  min_spacetime_weight    : 0.5
  neg_to_pos_ratio        : 0.25
  normalize_inputs        : 16384
  normalize_perframe      : false
  resample_invalid_frames : true
  temporal_dropout        : 0.5
  time_sampling           : uniform-soft5-soft4-contiguous
  time_kernel             : '(-1y,-6w,-2w,0,2w,6w,1y)'
  upweight_centers        : true
  use_centered_positives  : True
  use_grid_positives      : true
  verbose                 : 1
  max_epoch_length        : 16384
  mask_low_quality        : true
  mask_samecolor_method   : null
model:
  class_path: watch.tasks.fusion.methods.HeterogeneousModel
  init_args:
    token_width: 8
    token_dim: 64
    position_encoder:
      class_path: watch.tasks.fusion.methods.heterogeneous.MipNerfPositionalEncoder
      init_args:
        in_dims: 3
        max_freq: 3
        num_freqs: 16
    backbone:
      class_path: watch.tasks.fusion.architectures.transformer.TransformerEncoderDecoder
      init_args:
        encoder_depth: 4
        decoder_depth: 0
        dim: 160
        queries_dim: 96
        logits_dim: 64
        latent_dim_head: 256
    spatial_scale_base: 1.0
    temporal_scale_base: 1.0
    global_change_weight: 0.0
    global_class_weight: 0.0
    global_saliency_weight: 1.0
    saliency_loss: focal
    decoder: simple_conv
lr_scheduler:
  class_path: torch.optim.lr_scheduler.OneCycleLR
  init_args:
    max_lr: $TARGET_LR
    total_steps: $MAX_STEPS
    anneal_strategy: cos
    pct_start: 0.1
optimizer:
  class_path: torch.optim.AdamW
  init_args:
    lr: $TARGET_LR
    weight_decay: 5e-7
    betas:
      - 0.9
      - 0.99
trainer:
  accumulate_grad_batches: 8
  #callbacks:
  #  - class_path: pytorch_lightning.callbacks.ModelCheckpoint
  #    init_args:
  #      monitor: val_loss
  #      mode: min
  #      save_top_k: 5
  #      auto_insert_metric_name: true
  default_root_dir     : $DEFAULT_ROOT_DIR
  accelerator          : gpu
  devices              : 0,
  #devices              : 0,1
  #strategy             : ddp
  check_val_every_n_epoch: 1
  enable_checkpointing: true
  enable_model_summary: true
  log_every_n_steps: 50
  logger: true
  max_steps: $MAX_STEPS
  num_sanity_val_steps: 0
  replace_sampler_ddp: true
  track_grad_norm: -1
  limit_val_batches: 32
  limit_train_batches: 500
"


### OMG I hope checkpoints work now!
export CUDA_VISIBLE_DEVICES=1
python -m watch.tasks.fusion fit \
    --config /home/joncrall/data/dvc-repos/smart_expt_dvc/training/toothbrush/joncrall/Drop6/runs/Drop6_BAS_scratch_landcover_10GSD_split2_V33/lightning_logs/version_1/config.yaml \
    --ckpt_path /home/joncrall/data/dvc-repos/smart_expt_dvc/training/toothbrush/joncrall/Drop6/runs/Drop6_BAS_scratch_landcover_10GSD_split2_V33/lightning_logs/version_1/checkpoints/epoch=647-step=40824.ckpt




# On Toothbrush (train longer, f16, cos aneal, adamw, big Heterogeneous)
export CUDA_VISIBLE_DEVICES=0
DVC_DATA_DPATH=$(geowatch_dvc --tags='phase2_data' --hardware='auto')
DVC_EXPT_DPATH=$(geowatch_dvc --tags='phase2_expt' --hardware='auto')
echo "DVC_EXPT_DPATH = $DVC_EXPT_DPATH"
WORKDIR=$DVC_EXPT_DPATH/training/$HOSTNAME/$USER
DATASET_CODE=Drop6
KWCOCO_BUNDLE_DPATH=$DVC_DATA_DPATH/$DATASET_CODE
TRAIN_FPATH=$KWCOCO_BUNDLE_DPATH/data_train_fixquant_split2.kwcoco.zip
VALI_FPATH=$KWCOCO_BUNDLE_DPATH/data_vali_fixquant_split2.kwcoco.zip
CHANNELS="(L8,S2,PD):(blue|green|red|nir),(WV):(blue|green|red),(WV,WV1):pan,(S2):(landcover_hidden.0:32)"
#water|forest|field|impervious|barren
EXPERIMENT_NAME=Drop6_BAS_scratch_validation_10GSD_split2_V34
DEFAULT_ROOT_DIR=$WORKDIR/$DATASET_CODE/runs/$EXPERIMENT_NAME
TARGET_LR=1e-3
MAX_STEPS=80000
WATCH_GRID_WORKERS=0 python -m watch.tasks.fusion fit --config "
seed_everything: 1104562820
data:
  batch_size              : 4
  num_workers             : 4
  train_dataset           : $TRAIN_FPATH
  vali_dataset            : $VALI_FPATH
  time_steps              : 7
  chip_dims               : 128
  window_space_scale      : 3.3GSD
  input_space_scale       : 3.3GSD
  output_space_scale      : 300.0GSD
  channels                : '$CHANNELS'
  chip_overlap            : 0
  dist_weights            : 0
  min_spacetime_weight    : 0.5
  neg_to_pos_ratio        : 0.5
  normalize_inputs        : 16384
  normalize_perframe      : false
  resample_invalid_frames : true
  temporal_dropout        : 0.5
  time_sampling           : uniform-soft5-soft4-contiguous
  time_kernel             : '(-1y,-6w,-2w,0,2w,6w,1y)'
  upweight_centers        : true
  use_centered_positives  : True
  use_grid_positives      : true
  verbose                 : 1
  max_epoch_length        : 16384
  mask_low_quality        : true
  mask_samecolor_method   : null
model:
  class_path: watch.tasks.fusion.methods.HeterogeneousModel
  init_args:
    token_width: 8
    token_dim: 64
    position_encoder:
      class_path: watch.tasks.fusion.methods.heterogeneous.MipNerfPositionalEncoder
      init_args:
        in_dims: 3
        max_freq: 3
        num_freqs: 16
    backbone:
      class_path: watch.tasks.fusion.architectures.transformer.TransformerEncoderDecoder
      init_args:
        encoder_depth: 4
        decoder_depth: 0
        dim: 160
        queries_dim: 96
        logits_dim: 64
        latent_dim_head: 256
    spatial_scale_base: 1.0
    temporal_scale_base: 1.0
    global_change_weight: 0.0
    global_class_weight: 0.0
    global_saliency_weight: 1.0
    saliency_loss: focal
    decoder: simple_conv
lr_scheduler:
  class_path: torch.optim.lr_scheduler.OneCycleLR
  init_args:
    max_lr: $TARGET_LR
    total_steps: $MAX_STEPS
    anneal_strategy: cos
    pct_start: 0.1
optimizer:
  class_path: torch.optim.AdamW
  init_args:
    lr: $TARGET_LR
    weight_decay: 5e-7
    betas:
      - 0.9
      - 0.99
trainer:
  accumulate_grad_batches: 8
  #callbacks:
  #  - class_path: pytorch_lightning.callbacks.ModelCheckpoint
  #    init_args:
  #      monitor: val_loss
  #      mode: min
  #      save_top_k: 5
  #      auto_insert_metric_name: true
  default_root_dir     : $DEFAULT_ROOT_DIR
  accelerator          : gpu
  devices              : 0,
  #devices              : 0,1
  #strategy             : ddp
  check_val_every_n_epoch: 1
  enable_checkpointing: true
  enable_model_summary: true
  log_every_n_steps: 50
  logger: true
  max_steps: $MAX_STEPS
  num_sanity_val_steps: 0
  replace_sampler_ddp: true
  track_grad_norm: -1
  limit_val_batches: 64
  limit_train_batches: 500
torch_globals:
    float32_matmul_precision: medium

#  #precision: bf16
#initializer:
#    init: /home/joncrall/remote/toothbrush/data/dvc-repos/smart_expt_dvc/training/toothbrush/joncrall/Drop6/runs/Drop6_BAS_scratch_landcover_nohidden_10GSD_split2_V31/lightning_logs/version_20/package-interupt/package_epoch2_step177.pt
#    #init: /home/joncrall/remote/toothbrush/data/dvc-repos/smart_expt_dvc/training/toothbrush/joncrall/Drop6/runs/Drop6_BAS_scratch_landcover_nohidden_10GSD_split2_V31/lightning_logs/version_14/package-interupt/package_epoch2_step522.pt
"



# MAE Backbone (toothbrush)
export CUDA_VISIBLE_DEVICES=1
DVC_DATA_DPATH=$(geowatch_dvc --tags='phase2_data' --hardware='auto')
DVC_EXPT_DPATH=$(geowatch_dvc --tags='phase2_expt' --hardware='auto')
echo "DVC_EXPT_DPATH = $DVC_EXPT_DPATH"
WORKDIR=$DVC_EXPT_DPATH/training/$HOSTNAME/$USER
DATASET_CODE=Drop6
KWCOCO_BUNDLE_DPATH=$DVC_DATA_DPATH/$DATASET_CODE
TRAIN_FPATH=$KWCOCO_BUNDLE_DPATH/data_train_fixquant_split1.kwcoco.zip
VALI_FPATH=$KWCOCO_BUNDLE_DPATH/data_vali_fixquant_split1.kwcoco.zip
CHANNELS="(L8,S2,PD):(blue|green|red|nir),(WV):(blue|green|red),(WV,WV1):pan"
#water|forest|field|impervious|barren
EXPERIMENT_NAME=Drop6_BAS_WUMAE_validation_3GSD_split1_V35
DEFAULT_ROOT_DIR=$WORKDIR/$DATASET_CODE/runs/$EXPERIMENT_NAME
TARGET_LR=1e-4
MAX_STEPS=50000
WATCH_GRID_WORKERS=0 python -m watch.tasks.fusion fit --config "
data:
  batch_size              : 4
  num_workers             : 4
  train_dataset           : $TRAIN_FPATH
  vali_dataset            : $VALI_FPATH
  time_steps              : 9
  chip_dims               : 128
  window_space_scale      : 3.3GSD
  input_space_scale       : 3.3GSD
  output_space_scale      : 165.0GSD
  channels                : '$CHANNELS'
  chip_overlap            : 0
  dist_weights            : 0
  min_spacetime_weight    : 0.5
  neg_to_pos_ratio        : 0.5
  normalize_inputs        : 16384
  normalize_perframe      : false
  resample_invalid_frames : true
  temporal_dropout        : 0.5
  time_sampling           : uniform-soft5-soft4-contiguous
  time_kernel             : '(-1y,-6m,-6w,-2w,0,2w,6w,6m,1y)'
  upweight_centers        : true
  use_centered_positives  : True
  use_grid_positives      : true
  verbose                 : 1
  max_epoch_length        : 16384
  mask_low_quality        : true
  mask_samecolor_method   : null
model:
  class_path: watch.tasks.fusion.methods.HeterogeneousModel
  init_args:
    token_width: 8
    token_dim: 16
    position_encoder:
      class_path: watch.tasks.fusion.methods.heterogeneous.MipNerfPositionalEncoder
      init_args:
        in_dims: 3
        max_freq: 3
        num_freqs: 16
    backbone: wu-vit
    spatial_scale_base: 1.0
    temporal_scale_base: 1.0
    global_change_weight: 0.0
    global_class_weight: 0.0
    global_saliency_weight: 1.0
    saliency_loss: focal
    decoder: simple_conv
lr_scheduler:
  class_path: torch.optim.lr_scheduler.OneCycleLR
  init_args:
    max_lr: $TARGET_LR
    total_steps: $MAX_STEPS
    anneal_strategy: cos
    pct_start: 0.05
optimizer:
  class_path: torch.optim.AdamW
  init_args:
    lr: $TARGET_LR
    weight_decay: 1e-6
    betas:
      - 0.9
      - 0.99
trainer:
  accumulate_grad_batches: 6
  default_root_dir     : $DEFAULT_ROOT_DIR
  accelerator          : gpu
  devices              : 0,
  check_val_every_n_epoch: 1
  enable_checkpointing: true
  enable_model_summary: true
  log_every_n_steps: 50
  logger: true
  max_steps: $MAX_STEPS
  num_sanity_val_steps: 0
  replace_sampler_ddp: true
  track_grad_norm: -1
  limit_val_batches: 64
  limit_train_batches: 500
torch_globals:
    float32_matmul_precision: medium
initializer:
    init: $DVC_EXPT_DPATH/models/wu/MAE-2023-02-09/goldenMae-epoch=07-val_loss=0.23.ckpt
"


# MAE Backbone (toothbrush)
export CUDA_VISIBLE_DEVICES=0
DVC_DATA_DPATH=$(geowatch_dvc --tags='phase2_data' --hardware='auto')
DVC_EXPT_DPATH=$(geowatch_dvc --tags='phase2_expt' --hardware='auto')
echo "DVC_EXPT_DPATH = $DVC_EXPT_DPATH"
WORKDIR=$DVC_EXPT_DPATH/training/$HOSTNAME/$USER
DATASET_CODE=Drop6
KWCOCO_BUNDLE_DPATH=$DVC_DATA_DPATH/$DATASET_CODE
TRAIN_FPATH=$KWCOCO_BUNDLE_DPATH/data_train_fixquant_split1.kwcoco.zip
VALI_FPATH=$KWCOCO_BUNDLE_DPATH/data_vali_fixquant_split1.kwcoco.zip
CHANNELS="(L8,S2,PD):(blue|green|red|nir),(WV):(blue|green|red),(WV,WV1):pan"
#water|forest|field|impervious|barren
EXPERIMENT_NAME=Drop6_BAS_WUMAE_validation_3GSD_split1_V37
DEFAULT_ROOT_DIR=$WORKDIR/$DATASET_CODE/runs/$EXPERIMENT_NAME
TARGET_LR=1e-4
MAX_STEPS=50000
WATCH_GRID_WORKERS=2 python -m watch.tasks.fusion fit --config "
data:
  batch_size              : 1
  num_workers             : 4
  train_dataset           : $TRAIN_FPATH
  vali_dataset            : $VALI_FPATH
  time_steps              : 7
  chip_dims               : 128
  window_space_scale      : 10GSD
  input_space_scale       : 10GSD
  output_space_scale      : 100.0GSD
  channels                : '$CHANNELS'
  chip_overlap            : 0
  dist_weights            : 0
  min_spacetime_weight    : 0.5
  neg_to_pos_ratio        : 0.5
  normalize_inputs        : 16384
  normalize_perframe      : false
  normalize_peritem       : true
  resample_invalid_frames : true
  temporal_dropout        : 0.5
  time_sampling           : uniform-soft5-soft4-contiguous
  time_kernel             : '(-1y,-6w,-2w,0,2w,6w,1y)'
  upweight_centers        : true
  use_centered_positives  : True
  use_grid_positives      : true
  verbose                 : 1
  max_epoch_length        : 16384
  mask_low_quality        : true
  mask_samecolor_method   : null
model:
  class_path: watch.tasks.fusion.methods.HeterogeneousModel
  init_args:
    token_width: 8
    token_dim: 16
    position_encoder:
      class_path: watch.tasks.fusion.methods.heterogeneous.MipNerfPositionalEncoder
      init_args:
        in_dims: 3
        max_freq: 3
        num_freqs: 16
    backbone: wu-vit
    spatial_scale_base: 1.0
    temporal_scale_base: 1.0
    global_change_weight: 0.0
    global_class_weight: 0.0
    global_saliency_weight: 1.0
    saliency_loss: focal
    decoder: simple_conv
lr_scheduler:
  class_path: torch.optim.lr_scheduler.OneCycleLR
  init_args:
    max_lr: $TARGET_LR
    total_steps: $MAX_STEPS
    anneal_strategy: cos
    pct_start: 0.05
optimizer:
  class_path: torch.optim.AdamW
  init_args:
    lr: $TARGET_LR
    weight_decay: 1e-6
    betas:
      - 0.9
      - 0.99
trainer:
  accumulate_grad_batches: 8
  default_root_dir     : $DEFAULT_ROOT_DIR
  accelerator          : gpu
  devices              : 0,
  check_val_every_n_epoch: 1
  enable_checkpointing: true
  enable_model_summary: true
  log_every_n_steps: 50
  logger: true
  max_steps: $MAX_STEPS
  num_sanity_val_steps: 0
  replace_sampler_ddp: true
  track_grad_norm: -1
  limit_val_batches: 128
  limit_train_batches: 1024
torch_globals:
    float32_matmul_precision: medium
initializer:
    init: $DVC_EXPT_DPATH/models/wu/MAE-2023-02-09/goldenMae-epoch=07-val_loss=0.23.ckpt
"


# MAE Backbone (ooo)
export CUDA_VISIBLE_DEVICES=1
DVC_DATA_DPATH=$(geowatch_dvc --tags='phase2_data' --hardware='auto')
DVC_EXPT_DPATH=$(geowatch_dvc --tags='phase2_expt' --hardware='auto')
echo "DVC_EXPT_DPATH = $DVC_EXPT_DPATH"
WORKDIR=$DVC_EXPT_DPATH/training/$HOSTNAME/$USER
DATASET_CODE=Drop6
KWCOCO_BUNDLE_DPATH=$DVC_DATA_DPATH/$DATASET_CODE
TRAIN_FPATH=$KWCOCO_BUNDLE_DPATH/data_train_split1.kwcoco.zip
VALI_FPATH=$KWCOCO_BUNDLE_DPATH/data_vali_split1.kwcoco.zip
CHANNELS="(L8,S2,PD):(blue|green|red|nir),(WV):(blue|green|red),(WV,WV1):pan"
#water|forest|field|impervious|barren
EXPERIMENT_NAME=Drop6_BAS_WUMAE_10GSD_split1_V36
DEFAULT_ROOT_DIR=$WORKDIR/$DATASET_CODE/runs/$EXPERIMENT_NAME
TARGET_LR=1e-4
MAX_STEPS=50000
WATCH_GRID_WORKERS=2 python -m watch.tasks.fusion fit --config "
data:
  batch_size              : 1
  num_workers             : 4
  train_dataset           : $TRAIN_FPATH
  vali_dataset            : $VALI_FPATH
  time_steps              : 9
  chip_dims               : 96
  window_space_scale      : 10.0GSD
  input_space_scale       : 10.0GSD
  output_space_scale      : 10.0GSD
  channels                : '$CHANNELS'
  chip_overlap            : 0
  dist_weights            : 0
  min_spacetime_weight    : 0.5
  neg_to_pos_ratio        : 0.5
  normalize_inputs        : 16384
  normalize_perframe      : false
  resample_invalid_frames : true
  temporal_dropout        : 0.5
  time_sampling           : uniform-soft5-soft4-contiguous
  time_kernel             : '(-1y,-6m,-6w,-2w,0,2w,6w,6m,1y)'
  upweight_centers        : true
  use_centered_positives  : True
  use_grid_positives      : true
  verbose                 : 1
  max_epoch_length        : 16384
  mask_low_quality        : true
  mask_samecolor_method   : null
model:
  class_path: watch.tasks.fusion.methods.HeterogeneousModel
  init_args:
    token_width: 8
    token_dim: 16
    position_encoder:
      class_path: watch.tasks.fusion.methods.heterogeneous.MipNerfPositionalEncoder
      init_args:
        in_dims: 3
        max_freq: 3
        num_freqs: 16
    backbone: wu-vit
    spatial_scale_base: 1.0
    temporal_scale_base: 1.0
    global_change_weight: 0.0
    global_class_weight: 0.0
    global_saliency_weight: 1.0
    saliency_loss: focal
    decoder: simple_conv
lr_scheduler:
  class_path: torch.optim.lr_scheduler.OneCycleLR
  init_args:
    max_lr: $TARGET_LR
    total_steps: $MAX_STEPS
    anneal_strategy: cos
    pct_start: 0.05
optimizer:
  class_path: torch.optim.AdamW
  init_args:
    lr: $TARGET_LR
    weight_decay: 1e-6
    betas:
      - 0.9
      - 0.99
trainer:
  accumulate_grad_batches: 6
  default_root_dir     : $DEFAULT_ROOT_DIR
  accelerator          : gpu
  devices              : 0,
  check_val_every_n_epoch: 1
  enable_checkpointing: true
  enable_model_summary: true
  log_every_n_steps: 50
  logger: true
  max_steps: $MAX_STEPS
  num_sanity_val_steps: 0
  replace_sampler_ddp: true
  track_grad_norm: -1
  limit_val_batches: 128
  limit_train_batches: 1024
#torch_globals:
#    float32_matmul_precision: medium
initializer:
    init: $DVC_EXPT_DPATH/models/wu/MAE-2023-02-09/goldenMae-epoch=07-val_loss=0.23.ckpt
"



export CUDA_VISIBLE_DEVICES=0
DVC_DATA_DPATH=$(geowatch_dvc --tags='phase2_data' --hardware='auto')
DVC_EXPT_DPATH=$(geowatch_dvc --tags='phase2_expt' --hardware='auto')
echo "DVC_EXPT_DPATH = $DVC_EXPT_DPATH"
WORKDIR=$DVC_EXPT_DPATH/training/$HOSTNAME/$USER
DATASET_CODE=Drop6
KWCOCO_BUNDLE_DPATH=$DVC_DATA_DPATH/$DATASET_CODE
TRAIN_FPATH=$KWCOCO_BUNDLE_DPATH/data_train_fixquant_split2.kwcoco.zip
VALI_FPATH=$KWCOCO_BUNDLE_DPATH/data_vali_fixquant_split2.kwcoco.zip
CHANNELS="(L8,S2,PD):(blue|green|red|nir),(WV):(blue|green|red),(WV,WV1):pan,(S2):(water|forest|field|impervious|barren|landcover_hidden.0:32)"
EXPERIMENT_NAME=Drop6_BAS_scratch_landcover_10GSD_split2_V33
DEFAULT_ROOT_DIR=$WORKDIR/$DATASET_CODE/runs/$EXPERIMENT_NAME
TARGET_LR=1e-4
MAX_STEPS=80000



export CUDA_VISIBLE_DEVICES=1
DVC_DATA_DPATH=$(geowatch_dvc --tags='phase2_data' --hardware='auto')
DVC_EXPT_DPATH=$(geowatch_dvc --tags='phase2_expt' --hardware='auto')
echo "DVC_EXPT_DPATH = $DVC_EXPT_DPATH"
WORKDIR=$DVC_EXPT_DPATH/training/$HOSTNAME/$USER
DATASET_CODE=Drop6
KWCOCO_BUNDLE_DPATH=$DVC_DATA_DPATH/$DATASET_CODE
TRAIN_FPATH=$KWCOCO_BUNDLE_DPATH/data_train_fixquant_split6.kwcoco.zip
VALI_FPATH=$KWCOCO_BUNDLE_DPATH/data_vali_fixquant_split6.kwcoco.zip
CHANNELS="(L8,S2,PD):(blue|green|red|nir),(WV):(blue|green|red),(WV,WV1):pan,(S2):(landcover_hidden.0:32)"
EXPERIMENT_NAME=Drop6_BAS_10GSD_split6_V40
DEFAULT_ROOT_DIR=$WORKDIR/$DATASET_CODE/runs/$EXPERIMENT_NAME
TARGET_LR=1e-4
WEIGHT_DECAY=$(python -c "print($TARGET_LR * 0.01)")
MAX_STEPS=80000
WATCH_GRID_WORKERS=2 python -m watch.tasks.fusion fit --config "
data:
    select_videos          : $SELECT_VIDEOS
    num_workers            : 4
    train_dataset          : $TRAIN_FPATH
    vali_dataset           : $VALI_FPATH
    chip_dims              : 256,256
    time_steps             : 9
    time_sampling          : uniform-soft5-soft4-contiguous
    time_kernel            : '(-1y,-6m,-2m,-1w,0,1w,2m,6m,1y)'
    window_space_scale     : 10.0GSD
    input_space_scale      : 10.0GSD
    output_space_scale     : 160.0GSD
    neg_to_pos_ratio       : 1.0
    batch_size             : 4
    normalize_perframe     : true
    normalize_peritem      : false
    max_epoch_length       : 1000000
    channels               : '$CHANNELS'
    min_spacetime_weight   : 0.6
    temporal_dropout       : 0.5
    mask_low_quality       : True
    mask_samecolor_method  : None
    observable_threshold   : 0.0
    quality_threshold      : 0.0
    weight_dilate          : 10
    use_centered_positives : True
    normalize_inputs       : 16384
    balance_areas          : True
model:
    class_path: MultimodalTransformer
    init_args:
        saliency_weights       : auto
        class_weights          : auto
        tokenizer              : linconv
        arch_name              : smt_it_stm_p8
        decoder                : mlp
        positive_change_weight : 1
        negative_change_weight : 0.01
        stream_channels        : 16
        class_loss            : 'dicefocal'
        saliency_loss          : 'focal'
        saliency_head_hidden   : 5
        change_head_hidden     : 5
        global_change_weight   : 0.00
        global_class_weight    : 0.00
        global_saliency_weight : 1.00
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
    pct_start: 0.05

trainer:
    accumulate_grad_batches: 8
    default_root_dir     : $DEFAULT_ROOT_DIR
    accelerator          : gpu
    devices              : 0,
    #devices              : 0,1
    #strategy             : ddp
    limit_val_batches    : 256
    limit_train_batches  : 2048
    num_sanity_val_steps : 0
    max_epochs           : 360

torch_globals:
    float32_matmul_precision: auto

initializer:
    init: $DVC_EXPT_DPATH/models/fusion/eval3_candidates/packages/Drop3_SpotCheck_V323/Drop3_SpotCheck_V323_epoch=18-step=12976.pt
"


export CUDA_VISIBLE_DEVICES=0
DVC_DATA_DPATH=$(geowatch_dvc --tags='phase2_data' --hardware='auto')
DVC_EXPT_DPATH=$(geowatch_dvc --tags='phase2_expt' --hardware='auto')
echo "DVC_EXPT_DPATH = $DVC_EXPT_DPATH"
WORKDIR=$DVC_EXPT_DPATH/training/$HOSTNAME/$USER
DATASET_CODE=Drop6
KWCOCO_BUNDLE_DPATH=$DVC_DATA_DPATH/$DATASET_CODE
TRAIN_FPATH=$KWCOCO_BUNDLE_DPATH/data_train_fixquant_split6.kwcoco.zip
VALI_FPATH=$KWCOCO_BUNDLE_DPATH/data_vali_fixquant_split6.kwcoco.zip
CHANNELS="(L8,S2,PD):(blue|green|red|nir),(WV):(blue|green|red),(WV,WV1):pan"
EXPERIMENT_NAME=Drop6_BAS_10GSD_split6_V41
DEFAULT_ROOT_DIR=$WORKDIR/$DATASET_CODE/runs/$EXPERIMENT_NAME
TARGET_LR=1e-4
WEIGHT_DECAY=$(python -c "print($TARGET_LR * 0.01)")
echo "WEIGHT_DECAY = $WEIGHT_DECAY"
MAX_STEPS=80000
WATCH_GRID_WORKERS=2 python -m watch.tasks.fusion fit --config "
data:
    select_videos          : $SELECT_VIDEOS
    num_workers            : 4
    train_dataset          : $TRAIN_FPATH
    vali_dataset           : $VALI_FPATH
    chip_dims              : 256,256
    time_steps             : 9
    time_sampling          : uniform-soft5-soft4-contiguous
    time_kernel            : '(-1y,-6m,-2m,-1w,0,1w,2m,6m,1y)'
    window_space_scale     : 10.0GSD
    input_space_scale      : 10.0GSD
    output_space_scale     : 160.0GSD
    neg_to_pos_ratio       : 1.0
    batch_size             : 4
    normalize_perframe     : true
    normalize_peritem      : false
    max_epoch_length       : 1000000
    channels               : '$CHANNELS'
    min_spacetime_weight   : 0.6
    temporal_dropout       : 0.5
    mask_low_quality       : True
    mask_samecolor_method  : None
    observable_threshold   : 0.0
    quality_threshold      : 0.0
    weight_dilate          : 10
    use_centered_positives : True
    normalize_inputs       : 16384
    balance_areas          : True
model:
    class_path: MultimodalTransformer
    init_args:
        saliency_weights       : auto
        class_weights          : auto
        tokenizer              : linconv
        arch_name              : smt_it_stm_p8
        decoder                : mlp
        positive_change_weight : 1
        negative_change_weight : 0.01
        stream_channels        : 16
        class_loss            : 'dicefocal'
        saliency_loss          : 'focal'
        saliency_head_hidden   : 5
        change_head_hidden     : 5
        global_change_weight   : 0.00
        global_class_weight    : 0.00
        global_saliency_weight : 1.00
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
    pct_start: 0.05
trainer:
    accumulate_grad_batches: 8
    default_root_dir     : $DEFAULT_ROOT_DIR
    accelerator          : gpu
    devices              : 0,
    #devices              : 0,1
    #strategy             : ddp
    limit_val_batches    : 256
    limit_train_batches  : 2048
    num_sanity_val_steps : 0
    max_epochs           : 360

torch_globals:
    float32_matmul_precision: auto

initializer:
    init: $DVC_EXPT_DPATH/models/fusion/eval3_candidates/packages/Drop3_SpotCheck_V323/Drop3_SpotCheck_V323_epoch=18-step=12976.pt
"


### Toothbrush long training with normalize_perframe
export CUDA_VISIBLE_DEVICES=1
DVC_DATA_DPATH=$(geowatch_dvc --tags='phase2_data' --hardware='auto')
DVC_EXPT_DPATH=$(geowatch_dvc --tags='phase2_expt' --hardware='auto')
echo "DVC_EXPT_DPATH = $DVC_EXPT_DPATH"
WORKDIR=$DVC_EXPT_DPATH/training/$HOSTNAME/$USER
DATASET_CODE=Drop6
KWCOCO_BUNDLE_DPATH=$DVC_DATA_DPATH/$DATASET_CODE
TRAIN_FPATH=$KWCOCO_BUNDLE_DPATH/data_train_fixquant_split6.kwcoco.zip
VALI_FPATH=$KWCOCO_BUNDLE_DPATH/data_vali_fixquant_split6.kwcoco.zip
CHANNELS="(L8,S2,PD):(blue|green|red|nir),(WV):(blue|green|red),(WV,WV1):pan"
EXPERIMENT_NAME=Drop6_BAS_10GSD_split6_V41_cont
DEFAULT_ROOT_DIR=$WORKDIR/$DATASET_CODE/runs/$EXPERIMENT_NAME
TARGET_LR=3e-4
WEIGHT_DECAY=$(python -c "print($TARGET_LR * 0.01)")
echo "WEIGHT_DECAY = $WEIGHT_DECAY"
MAX_STEPS=80000
WATCH_GRID_WORKERS=2 python -m watch.tasks.fusion fit --config "
data:
    select_videos          : $SELECT_VIDEOS
    num_workers            : 4
    train_dataset          : $TRAIN_FPATH
    vali_dataset           : $VALI_FPATH
    chip_dims              : 256,256
    time_steps             : 9
    time_sampling          : uniform-soft5-soft4-contiguous
    time_kernel            : '(-1y,-6m,-2m,-1w,0,1w,2m,6m,1y)'
    window_space_scale     : 10.0GSD
    input_space_scale      : 10.0GSD
    output_space_scale     : 10.0GSD
    neg_to_pos_ratio       : 1.0
    batch_size             : 4
    normalize_perframe     : true
    normalize_peritem      : false
    max_epoch_length       : 1000000
    channels               : '$CHANNELS'
    min_spacetime_weight   : 0.6
    temporal_dropout       : 0.5
    mask_low_quality       : True
    mask_samecolor_method  : None
    observable_threshold   : 0.0
    quality_threshold      : 0.0
    weight_dilate          : 10
    use_centered_positives : True
    normalize_inputs       : 16384
    balance_areas          : False
model:
    class_path: MultimodalTransformer
    init_args:
        saliency_weights       : auto
        class_weights          : auto
        tokenizer              : linconv
        arch_name              : smt_it_stm_p8
        decoder                : mlp
        positive_change_weight : 1
        negative_change_weight : 0.01
        stream_channels        : 16
        class_loss            : 'dicefocal'
        saliency_loss          : 'focal'
        saliency_head_hidden   : 3
        change_head_hidden     : 3
        global_change_weight   : 0.00
        global_class_weight    : 0.00
        global_saliency_weight : 1.00
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
    pct_start: 0.05
trainer:
    accumulate_grad_batches: 8
    default_root_dir     : $DEFAULT_ROOT_DIR
    accelerator          : gpu
    devices              : 0,
    #devices              : 0,1
    #strategy             : ddp
    limit_val_batches    : 256
    limit_train_batches  : 2048
    num_sanity_val_steps : 0
    max_epochs           : 360

torch_globals:
    float32_matmul_precision: auto

initializer:
    init: /home/joncrall/remote/toothbrush/data/dvc-repos/smart_expt_dvc/training/toothbrush/joncrall/Drop6/runs/Drop6_BAS_10GSD_split6_V41/lightning_logs/version_1/package-interupt/package_epoch94_step24084.pt
"


### Toothbrush long training with normalize_peritem and 1year averaged data.
export CUDA_VISIBLE_DEVICES=1
DVC_DATA_DPATH=$(geowatch_dvc --tags='phase2_data' --hardware='auto')
DVC_EXPT_DPATH=$(geowatch_dvc --tags='phase2_expt' --hardware='auto')
echo "DVC_EXPT_DPATH = $DVC_EXPT_DPATH"
WORKDIR=$DVC_EXPT_DPATH/training/$HOSTNAME/$USER
DATASET_CODE=Drop6-MeanYear10GSD
KWCOCO_BUNDLE_DPATH=$DVC_DATA_DPATH/$DATASET_CODE
TRAIN_FPATH=$KWCOCO_BUNDLE_DPATH/data_train_rawbands_split6.kwcoco.zip
VALI_FPATH=$KWCOCO_BUNDLE_DPATH/data_vali_rawbands_split6.kwcoco.zip
CHANNELS="(L8,S2,PD):(blue|green|red|nir),(WV):(blue|green|red),(WV,WV1):pan"
EXPERIMENT_NAME=Drop6_TCombo1Year_BAS_10GSD_split6_V41_cont2
ls "$WORKDIR"/$DATASET_CODE/runs/$EXPERIMENT_NAME/lightning_logs/version_*/checkpoints/*.ckpt
DEFAULT_ROOT_DIR=$WORKDIR/$DATASET_CODE/runs/$EXPERIMENT_NAME
TARGET_LR=3e-4
WEIGHT_DECAY=$(python -c "print($TARGET_LR * 0.01)")
echo "WEIGHT_DECAY = $WEIGHT_DECAY"
MAX_STEPS=80000
WATCH_GRID_WORKERS=2 python -m watch.tasks.fusion fit --config "
data:
    select_videos          : $SELECT_VIDEOS
    num_workers            : 6
    train_dataset          : $TRAIN_FPATH
    vali_dataset           : $VALI_FPATH
    chip_dims              : 256,256
    time_steps             : 9
    time_sampling          : uniform-soft5-soft4-contiguous
    time_kernel            : '(-2.5y,-2y,-1.5y,-1y,0,1y,1.5y,2y,2.5y)'
    window_space_scale     : 10.0GSD
    input_space_scale      : 10.0GSD
    output_space_scale     : 10.0GSD
    neg_to_pos_ratio       : 1.0
    batch_size             : 6
    normalize_perframe     : false
    normalize_peritem      : true
    max_epoch_length       : 1000000
    channels               : '$CHANNELS'
    min_spacetime_weight   : 0.6
    temporal_dropout       : 0.5
    mask_low_quality       : False
    mask_samecolor_method  : None
    observable_threshold   : 0.0
    quality_threshold      : 0.0
    weight_dilate          : 10
    use_centered_positives : True
    normalize_inputs       : 16384
    balance_areas          : False
model:
    class_path: MultimodalTransformer
    init_args:
        saliency_weights       : auto
        class_weights          : auto
        tokenizer              : linconv
        arch_name              : smt_it_stm_p8
        decoder                : mlp
        positive_change_weight : 1
        negative_change_weight : 0.01
        stream_channels        : 16
        class_loss            : 'dicefocal'
        saliency_loss          : 'focal'
        saliency_head_hidden   : 3
        change_head_hidden     : 3
        global_change_weight   : 0.00
        global_class_weight    : 0.00
        global_saliency_weight : 1.00
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
    pct_start: 0.05
trainer:
    accumulate_grad_batches: 8
    default_root_dir     : $DEFAULT_ROOT_DIR
    accelerator          : gpu
    devices              : 0,
    #devices              : 0,1
    #strategy             : ddp
    limit_val_batches    : 256
    limit_train_batches  : 2048
    num_sanity_val_steps : 0
    max_epochs           : 360

torch_globals:
    float32_matmul_precision: auto

initializer:
    init: /home/joncrall/remote/toothbrush/data/dvc-repos/smart_expt_dvc/training/toothbrush/joncrall/Drop6-MeanYear10GSD/runs/Drop6_TCombo1Year_BAS_10GSD_split6_V41_cont2/lightning_logs/version_2/package-interupt/package_epoch0_step195.pt
" \
    --ckpt_path=/home/joncrall/remote/toothbrush/data/dvc-repos/smart_expt_dvc/training/toothbrush/joncrall/Drop6-MeanYear10GSD/runs/Drop6_TCombo1Year_BAS_10GSD_split6_V41_cont2/lightning_logs/version_8/checkpoints/epoch=97-step=25088.ckpt

#--ckpt_path=/home/joncrall/remote/toothbrush/data/dvc-repos/smart_expt_dvc/training/toothbrush/joncrall/Drop6-MeanYear10GSD/runs/Drop6_TCombo1Year_BAS_10GSD_split6_V41_cont2/lightning_logs/version_7/checkpoints/epoch=69-step=17920.ckpt
 #--ckpt_path=/home/joncrall/remote/toothbrush/data/dvc-repos/smart_expt_dvc/training/toothbrush/joncrall/Drop6-MeanYear10GSD/runs/Drop6_TCombo1Year_BAS_10GSD_split6_V41_cont2/lightning_logs/version_3/checkpoints/epoch=40-step=10496.ckpt

#/home/joncrall/remote/toothbrush/data/dvc-repos/smart_expt_dvc/training/toothbrush/joncrall/Drop6-MeanYear10GSD/runs/Drop6_TCombo1Year_BAS_10GSD_split6_V41_cont2/lightning_logs/version_0/checkpoints/epoch=1-step=512.ckpt
ls "$WORKDIR"/$DATASET_CODE/runs/$EXPERIMENT_NAME/lightning_logs/version_*/checkpoints/*.ckpt


#


## Yardrat long train run
export CUDA_VISIBLE_DEVICES=0
DVC_DATA_DPATH=$(geowatch_dvc --tags='phase2_data' --hardware='auto')
DVC_EXPT_DPATH=$(geowatch_dvc --tags='phase2_expt' --hardware='auto')
echo "DVC_EXPT_DPATH = $DVC_EXPT_DPATH"
WORKDIR=$DVC_EXPT_DPATH/training/$HOSTNAME/$USER
DATASET_CODE=Drop6-MeanYear10GSD
KWCOCO_BUNDLE_DPATH=$DVC_DATA_DPATH/$DATASET_CODE
TRAIN_FPATH=$KWCOCO_BUNDLE_DPATH/data_train_rawbands_split6.kwcoco.zip
VALI_FPATH=$KWCOCO_BUNDLE_DPATH/data_vali_rawbands_split6.kwcoco.zip
CHANNELS="(L8,S2,PD):(blue|green|red|nir),(WV):(blue|green|red),(WV,WV1):pan"
EXPERIMENT_NAME=Drop6_TCombo1Year_BAS_10GSD_split6_V42
DEFAULT_ROOT_DIR=$WORKDIR/$DATASET_CODE/runs/$EXPERIMENT_NAME
TARGET_LR=1e-4
WEIGHT_DECAY=$(python -c "print($TARGET_LR * 0.01)")
echo "WEIGHT_DECAY = $WEIGHT_DECAY"
MAX_STEPS=80000
WATCH_GRID_WORKERS=2 python -m watch.tasks.fusion fit --config "
data:
    select_videos          : $SELECT_VIDEOS
    num_workers            : 5
    train_dataset          : $TRAIN_FPATH
    vali_dataset           : $VALI_FPATH
    chip_dims              : 196,196
    time_steps             : 11
    time_sampling          : uniform-soft5-soft4-contiguous
    time_kernel            : '(-3y,-2.5y,-2y,-1.5y,-1y,0,1y,1.5y,2y,2.5y,3y)'
    window_space_scale     : 10.0GSD
    input_space_scale      : 10.0GSD
    output_space_scale     : 10.0GSD
    neg_to_pos_ratio       : 1.0
    batch_size             : 4
    normalize_perframe     : false
    normalize_peritem      : true
    max_epoch_length       : 1000000
    channels               : '$CHANNELS'
    min_spacetime_weight   : 0.6
    temporal_dropout       : 0.5
    mask_low_quality       : False
    mask_samecolor_method  : None
    observable_threshold   : 0.0
    quality_threshold      : 0.0
    weight_dilate          : 10
    use_centered_positives : True
    normalize_inputs       : 16384
    balance_areas          : True
model:
    class_path: MultimodalTransformer
    init_args:
        #saliency_weights       : '1:1'
        #class_weights          : auto
        tokenizer              : linconv
        arch_name              : smt_it_stm_p8
        decoder                : mlp
        positive_change_weight : 1
        negative_change_weight : 0.01
        stream_channels        : 16
        class_loss             : 'dicefocal'
        saliency_loss          : 'focal'
        saliency_head_hidden   : 6
        change_head_hidden     : 6
        class_head_hidden      : 6
        global_change_weight   : 0.05
        global_class_weight    : 0.50
        global_saliency_weight : 1.00
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
    pct_start: 0.05
trainer:
    accumulate_grad_batches: 8
    default_root_dir     : $DEFAULT_ROOT_DIR
    accelerator          : gpu
    devices              : 0,
    #devices              : 0,1
    #strategy             : ddp
    limit_val_batches    : 256
    limit_train_batches  : 2048
    num_sanity_val_steps : 0
    max_epochs           : 360

torch_globals:
    float32_matmul_precision: auto

initializer:
    init: $DVC_EXPT_DPATH/models/fusion/Drop6/packages/Drop6_BAS_2022_12_10GSD_BGRN_V11_CONT4/Drop6_BAS_2022_12_10GSD_BGRN_V11_CONT4_epoch6_step22939.pt
"

DVC_EXPT_DPATH=$(geowatch_dvc --tags='phase2_expt' --hardware='auto')
rsync -avprPR yardrat:data/dvc-repos/smart_expt_dvc/./training/yardrat/jon.crall/Drop6-MeanYear10GSD/runs/Drop6_TCombo1Year_BAS_10GSD_split6_V42 "$DVC_EXPT_DPATH"

DVC_EXPT_DPATH=$(geowatch_dvc --tags='phase2_expt' --hardware='auto')
smartwatch model_stats "$DVC_EXPT_DPATH"/models/fusion/Drop6-MeanYear10GSD/packages/Drop6_TCombo1Year_BAS_10GSD_split6_V42_cont2/Drop6_TCombo1Year_BAS_10GSD_split6_V42_cont2_epoch3_step941.pt


## Yardrat long train run
export CUDA_VISIBLE_DEVICES=0
DVC_DATA_DPATH=$(geowatch_dvc --tags='phase2_data' --hardware='auto')
DVC_EXPT_DPATH=$(geowatch_dvc --tags='phase2_expt' --hardware='auto')
echo "DVC_EXPT_DPATH = $DVC_EXPT_DPATH"
WORKDIR=$DVC_EXPT_DPATH/training/$HOSTNAME/$USER
DATASET_CODE=Drop6-MeanYear10GSD
KWCOCO_BUNDLE_DPATH=$DVC_DATA_DPATH/$DATASET_CODE
TRAIN_FPATH=$KWCOCO_BUNDLE_DPATH/data_train_rawbands_split6.kwcoco.zip
VALI_FPATH=$KWCOCO_BUNDLE_DPATH/data_vali_rawbands_split6.kwcoco.zip
CHANNELS="(L8,S2,PD):(blue|green|red|nir),(WV):(blue|green|red),(WV,WV1):pan"
EXPERIMENT_NAME=Drop6_TCombo1Year_BAS_10GSD_split6_V42_cont3
DEFAULT_ROOT_DIR=$WORKDIR/$DATASET_CODE/runs/$EXPERIMENT_NAME
TARGET_LR=1e-4
WEIGHT_DECAY=$(python -c "print($TARGET_LR * 0.01)")
echo "WEIGHT_DECAY = $WEIGHT_DECAY"
MAX_STEPS=80000
WATCH_GRID_WORKERS=2 python -m watch.tasks.fusion fit --config "
data:
    select_videos          : $SELECT_VIDEOS
    num_workers            : 5
    train_dataset          : $TRAIN_FPATH
    vali_dataset           : $VALI_FPATH
    chip_dims              : 196,196
    time_steps             : 11
    time_sampling          : uniform-soft5-soft4-contiguous
    time_kernel            : '(-3y,-2.5y,-2y,-1.5y,-1y,0,1y,1.5y,2y,2.5y,3y)'
    window_space_scale     : 10.0GSD
    input_space_scale      : 10.0GSD
    output_space_scale     : 10.0GSD
    neg_to_pos_ratio       : 1.0
    batch_size             : 4
    normalize_perframe     : false
    normalize_peritem      : true
    max_epoch_length       : 1000000
    channels               : '$CHANNELS'
    min_spacetime_weight   : 0.6
    temporal_dropout       : 0.5
    mask_low_quality       : False
    mask_samecolor_method  : None
    observable_threshold   : 0.0
    quality_threshold      : 0.0
    weight_dilate          : 10
    use_centered_positives : True
    normalize_inputs       : 16384
    balance_areas          : False
model:
    class_path: MultimodalTransformer
    init_args:
        #saliency_weights       : '1:1'
        #class_weights          : auto
        tokenizer              : linconv
        arch_name              : smt_it_stm_p8
        decoder                : mlp
        positive_change_weight : 1
        negative_change_weight : 0.01
        stream_channels        : 16
        class_loss             : 'dicefocal'
        saliency_loss          : 'dicefocal'
        saliency_head_hidden   : 6
        change_head_hidden     : 6
        class_head_hidden      : 6
        global_change_weight   : 0.00
        global_class_weight    : 0.00
        global_saliency_weight : 1.00
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
    pct_start: 0.05
trainer:
    accumulate_grad_batches: 8
    default_root_dir     : $DEFAULT_ROOT_DIR
    accelerator          : gpu
    devices              : 0,
    #devices              : 0,1
    #strategy             : ddp
    limit_val_batches    : 256
    limit_train_batches  : 2048
    num_sanity_val_steps : 0
    max_epochs           : 360

torch_globals:
    float32_matmul_precision: auto

initializer:
    init: /home/local/KHQ/jon.crall/remote/yardrat/data/dvc-repos/smart_expt_dvc/training/yardrat/jon.crall/Drop6-MeanYear10GSD/runs/Drop6_TCombo1Year_BAS_10GSD_split6_V42_cont2/lightning_logs/version_0/package-interupt/package_epoch3_step941.pt
    #init: /home/local/KHQ/jon.crall/remote/yardrat/data/dvc-repos/smart_expt_dvc/training/yardrat/jon.crall/Drop6-MeanYear10GSD/runs/Drop6_TCombo1Year_BAS_10GSD_split6_V42/lightning_logs/version_3/package-interupt/package_epoch102_step26346.pt
    #init: $DVC_EXPT_DPATH/models/fusion/Drop6/packages/Drop6_BAS_2022_12_10GSD_BGRN_V11_CONT4/Drop6_BAS_2022_12_10GSD_BGRN_V11_CONT4_epoch6_step22939.pt
"


DVC_EXPT_DPATH=$(geowatch_dvc --tags='phase2_expt' --hardware='auto')
rsync -avprPR yardrat:data/dvc-repos/smart_expt_dvc/./training/yardrat/jon.crall/Drop6-MeanYear10GSD/runs/Drop6_TCombo1Year_BAS_10GSD_split6_V42 "$DVC_EXPT_DPATH"


### Toothbrush long training with normalize_peritem and 1year averaged data.
export CUDA_VISIBLE_DEVICES=1
DVC_DATA_DPATH=$(geowatch_dvc --tags='phase2_data' --hardware='auto')
DVC_EXPT_DPATH=$(geowatch_dvc --tags='phase2_expt' --hardware='auto')
echo "DVC_EXPT_DPATH = $DVC_EXPT_DPATH"
WORKDIR=$DVC_EXPT_DPATH/training/$HOSTNAME/$USER
DATASET_CODE=Drop6-MeanYear10GSD
KWCOCO_BUNDLE_DPATH=$DVC_DATA_DPATH/$DATASET_CODE
TRAIN_FPATH=$KWCOCO_BUNDLE_DPATH/data_train_rawbands_split6.kwcoco.zip
VALI_FPATH=$KWCOCO_BUNDLE_DPATH/data_vali_rawbands_split6.kwcoco.zip
CHANNELS="(L8,S2,PD):(blue|green|red|nir),(WV):(blue|green|red),(WV,WV1):pan"
EXPERIMENT_NAME=Drop6_TCombo1Year_BAS_10GSD_split6_V41_cont3
DEFAULT_ROOT_DIR=$WORKDIR/$DATASET_CODE/runs/$EXPERIMENT_NAME
TARGET_LR=1e-4
WEIGHT_DECAY=$(python -c "print($TARGET_LR * 0.01)")
echo "WEIGHT_DECAY = $WEIGHT_DECAY"
MAX_STEPS=80000
WATCH_GRID_WORKERS=2 python -m watch.tasks.fusion fit --config "
data:
    select_videos          : $SELECT_VIDEOS
    num_workers            : 6
    train_dataset          : $TRAIN_FPATH
    vali_dataset           : $VALI_FPATH
    chip_dims              : 256,256
    time_steps             : 9
    time_sampling          : uniform-soft5-soft4-contiguous
    time_kernel            : '(-2.5y,-2y,-1.5y,-1y,0,1y,1.5y,2y,2.5y)'
    window_space_scale     : 10.0GSD
    input_space_scale      : 10.0GSD
    output_space_scale     : 10.0GSD
    neg_to_pos_ratio       : 1.0
    batch_size             : 6
    normalize_perframe     : false
    normalize_peritem      : true
    max_epoch_length       : 1000000
    channels               : '$CHANNELS'
    min_spacetime_weight   : 0.6
    temporal_dropout       : 0.5
    mask_low_quality       : False
    mask_samecolor_method  : None
    observable_threshold   : 0.0
    quality_threshold      : 0.0
    weight_dilate          : 10
    use_centered_positives : True
    normalize_inputs       : 16384
    balance_areas          : False
model:
    class_path: MultimodalTransformer
    init_args:
        saliency_weights       : auto
        class_weights          : auto
        tokenizer              : linconv
        arch_name              : smt_it_stm_p8
        decoder                : mlp
        positive_change_weight : 1
        negative_change_weight : 0.01
        stream_channels        : 16
        class_loss             : 'dicefocal'
        saliency_loss          : 'focal'
        saliency_head_hidden   : 3
        change_head_hidden     : 3
        global_change_weight   : 0.00
        global_class_weight    : 0.50
        global_saliency_weight : 1.00
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
    pct_start: 0.05
trainer:
    accumulate_grad_batches: 16
    default_root_dir     : $DEFAULT_ROOT_DIR
    accelerator          : gpu
    devices              : 0,
    #devices              : 0,1
    #strategy             : ddp
    limit_val_batches    : 256
    limit_train_batches  : 2048
    num_sanity_val_steps : 0
    max_epochs           : 360

torch_globals:
    float32_matmul_precision: auto

initializer:
    init: /home/joncrall/remote/toothbrush/data/dvc-repos/smart_expt_dvc/training/toothbrush/joncrall/Drop6-MeanYear10GSD/runs/Drop6_TCombo1Year_BAS_10GSD_split6_V41_cont2/lightning_logs/version_9/checkpoints/epoch=110-step=28416.ckpt
"

#--ckpt_path=/home/joncrall/remote/toothbrush/data/dvc-repos/smart_expt_dvc/training/toothbrush/joncrall/Drop6-MeanYear10GSD/runs/Drop6_TCombo1Year_BAS_10GSD_split6_V41_cont2/lightning_logs/version_7/checkpoints/epoch=69-step=17920.ckpt
 #--ckpt_path=/home/joncrall/remote/toothbrush/data/dvc-repos/smart_expt_dvc/training/toothbrush/joncrall/Drop6-MeanYear10GSD/runs/Drop6_TCombo1Year_BAS_10GSD_split6_V41_cont2/lightning_logs/version_3/checkpoints/epoch=40-step=10496.ckpt

#/home/joncrall/remote/toothbrush/data/dvc-repos/smart_expt_dvc/training/toothbrush/joncrall/Drop6-MeanYear10GSD/runs/Drop6_TCombo1Year_BAS_10GSD_split6_V41_cont2/lightning_logs/version_0/checkpoints/epoch=1-step=512.ckpt
ls "$WORKDIR"/$DATASET_CODE/runs/$EXPERIMENT_NAME/lightning_logs/version_*/checkpoints/*.ckpt



## Yardrat continue good model
export CUDA_VISIBLE_DEVICES=0
DVC_DATA_DPATH=$(geowatch_dvc --tags='phase2_data' --hardware='auto')
DVC_EXPT_DPATH=$(geowatch_dvc --tags='phase2_expt' --hardware='auto')
echo "DVC_EXPT_DPATH = $DVC_EXPT_DPATH"
WORKDIR=$DVC_EXPT_DPATH/training/$HOSTNAME/$USER
DATASET_CODE=Drop6-MeanYear10GSD
KWCOCO_BUNDLE_DPATH=$DVC_DATA_DPATH/$DATASET_CODE
TRAIN_FPATH=$KWCOCO_BUNDLE_DPATH/data_train_rawbands_split6.kwcoco.zip
VALI_FPATH=$KWCOCO_BUNDLE_DPATH/data_vali_rawbands_split6.kwcoco.zip
CHANNELS="(L8,S2,PD):(blue|green|red|nir),(WV):(blue|green|red),(WV,WV1):pan"
EXPERIMENT_NAME=Drop6_TCombo1Year_BAS_10GSD_split6_V42_cont4
DEFAULT_ROOT_DIR=$WORKDIR/$DATASET_CODE/runs/$EXPERIMENT_NAME
TARGET_LR=1e-4
WEIGHT_DECAY=$(python -c "print($TARGET_LR * 0.01)")
echo "WEIGHT_DECAY = $WEIGHT_DECAY"
MAX_STEPS=80000
WATCH_GRID_WORKERS=2 python -m watch.tasks.fusion fit --config "
data:
    select_videos          : $SELECT_VIDEOS
    num_workers            : 5
    train_dataset          : $TRAIN_FPATH
    vali_dataset           : $VALI_FPATH
    chip_dims              : 196,196
    time_steps             : 11
    time_sampling          : uniform-soft5-soft4-contiguous
    time_kernel            : '(-3y,-2.5y,-2y,-1.5y,-1y,0,1y,1.5y,2y,2.5y,3y)'
    window_space_scale     : 10.0GSD
    input_space_scale      : 10.0GSD
    output_space_scale     : 10.0GSD
    neg_to_pos_ratio       : 1.0
    batch_size             : 4
    normalize_perframe     : false
    normalize_peritem      : true
    max_epoch_length       : 1000000
    channels               : '$CHANNELS'
    min_spacetime_weight   : 0.6
    temporal_dropout       : 0.5
    mask_low_quality       : False
    mask_samecolor_method  : None
    observable_threshold   : 0.0
    quality_threshold      : 0.0
    weight_dilate          : 10
    use_centered_positives : True
    normalize_inputs       : 16384
    balance_areas          : True
model:
    class_path: MultimodalTransformer
    init_args:
        #saliency_weights       : '1:1'
        #class_weights          : auto
        tokenizer              : linconv
        arch_name              : smt_it_stm_p8
        decoder                : mlp
        positive_change_weight : 1
        negative_change_weight : 0.01
        stream_channels        : 16
        class_loss             : 'dicefocal'
        saliency_loss          : 'focal'
        saliency_head_hidden   : 6
        change_head_hidden     : 6
        class_head_hidden      : 6
        global_change_weight   : 0.05
        global_class_weight    : 0.50
        global_saliency_weight : 1.00
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
    pct_start: 0.05
trainer:
    accumulate_grad_batches: 16
    default_root_dir     : $DEFAULT_ROOT_DIR
    accelerator          : gpu
    devices              : 0,
    #devices              : 0,1
    #strategy             : ddp
    limit_val_batches    : 256
    limit_train_batches  : 2048
    num_sanity_val_steps : 0
    max_epochs           : 360

torch_globals:
    float32_matmul_precision: auto

initializer:
    init: $DVC_EXPT_DPATH/models/fusion/Drop6-MeanYear10GSD/packages/Drop6_TCombo1Year_BAS_10GSD_split6_V42_cont2/Drop6_TCombo1Year_BAS_10GSD_split6_V42_cont2_epoch3_step941.pt
"

models/fusion/Drop6-MeanYear10GSD/packages/Drop6_TCombo1Year_BAS_10GSD_split6_V42_cont2/Drop6_TCombo1Year_BAS_10GSD_split6_V42_cont2_epoch3_step941.pt


DVC_EXPT_DPATH=$(geowatch_dvc --tags='phase2_expt' --hardware='auto')
smartwatch model_stats "$DVC_EXPT_DPATH"/models/fusion/Drop6-MeanYear10GSD/packages/Drop6_TCombo1Year_BAS_10GSD_split6_V42_cont2/Drop6_TCombo1Year_BAS_10GSD_split6_V42_cont2_epoch3_step941.pt
rsync -avprPR yardrat:data/dvc-repos/smart_expt_dvc/./training/yardrat/jon.crall/Drop6-MeanYear10GSD/runs/Drop6_TCombo1Year_BAS_10GSD_split6_V42 "$DVC_EXPT_DPATH"


#####


## Continue training on toothbrush with new annotations and hard negatives
export CUDA_VISIBLE_DEVICES=0
DVC_DATA_DPATH=$(geowatch_dvc --tags='phase2_data' --hardware='auto')
DVC_EXPT_DPATH=$(geowatch_dvc --tags='phase2_expt' --hardware='auto')
echo "DVC_EXPT_DPATH = $DVC_EXPT_DPATH"
WORKDIR=$DVC_EXPT_DPATH/training/$HOSTNAME/$USER
DATASET_CODE=Drop6-MeanYear10GSD
KWCOCO_BUNDLE_DPATH=$DVC_DATA_DPATH/$DATASET_CODE
TRAIN_FPATH=$KWCOCO_BUNDLE_DPATH/data_train_rawbands_split6.kwcoco.zip
VALI_FPATH=$KWCOCO_BUNDLE_DPATH/data_vali_rawbands_split6.kwcoco.zip
CHANNELS="(L8,S2,PD):(blue|green|red|nir),(WV):(blue|green|red),(WV,WV1):pan"
EXPERIMENT_NAME=Drop6_TCombo1Year_BAS_10GSD_split6_V43
DEFAULT_ROOT_DIR=$WORKDIR/$DATASET_CODE/runs/$EXPERIMENT_NAME
TARGET_LR=1e-4
WEIGHT_DECAY=$(python -c "print($TARGET_LR * 0.01)")
echo "WEIGHT_DECAY = $WEIGHT_DECAY"
MAX_STEPS=80000
WATCH_GRID_WORKERS=0 python -m watch.tasks.fusion fit --config "
data:
    select_videos          : $SELECT_VIDEOS
    num_workers            : 5
    train_dataset          : $TRAIN_FPATH
    vali_dataset           : $VALI_FPATH
    window_dims            : '196,196'
    time_steps             : 11
    time_sampling          : uniform-soft5-soft4-contiguous
    time_kernel            : '(-3y,-2.5y,-2y,-1.5y,-1y,0,1y,1.5y,2y,2.5y,3y)'
    window_resolution     : 10.0GSD
    input_resolution      : 10.0GSD
    output_resolution     : 10.0GSD
    neg_to_pos_ratio       : 1.0
    batch_size             : 4
    normalize_perframe     : false
    normalize_peritem      : true
    max_epoch_length       : 1000000
    channels               : '$CHANNELS'
    min_spacetime_weight   : 0.6
    temporal_dropout       : 0.5
    mask_low_quality       : False
    mask_samecolor_method  : None
    observable_threshold   : 0.0
    quality_threshold      : 0.0
    weight_dilate          : 10
    use_centered_positives : True
    use_grid_positives     : True
    use_grid_negatives     : 'cleared'
    normalize_inputs       : 16384
    balance_areas          : True
model:
    class_path: MultimodalTransformer
    init_args:
        #saliency_weights       : '1:1'
        #class_weights          : auto
        tokenizer              : linconv
        arch_name              : smt_it_stm_p16
        decoder                : mlp
        positive_change_weight : 1
        negative_change_weight : 0.01
        stream_channels        : 16
        class_loss             : 'dicefocal'
        saliency_loss          : 'focal'
        saliency_head_hidden   : 6
        change_head_hidden     : 6
        class_head_hidden      : 6
        global_change_weight   : 0.00
        global_class_weight    : 0.00
        global_saliency_weight : 1.00
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
    pct_start: 0.05
trainer:
    accumulate_grad_batches: 8
    default_root_dir     : $DEFAULT_ROOT_DIR
    accelerator          : gpu
    devices              : 0,
    #devices              : 0,1
    #strategy             : ddp
    limit_val_batches    : 256
    limit_train_batches  : 2048
    num_sanity_val_steps : 0
    max_epochs           : 360

torch_globals:
    float32_matmul_precision: auto

initializer:
    init: $DVC_EXPT_DPATH/models/fusion/Drop6-MeanYear10GSD/packages/Drop6_TCombo1Year_BAS_10GSD_split6_V42_cont2/Drop6_TCombo1Year_BAS_10GSD_split6_V42_cont2_epoch3_step941.pt
" \
    --ckpt_path=/home/joncrall/remote/toothbrush/data/dvc-repos/smart_expt_dvc/training/toothbrush/joncrall/Drop6-MeanYear10GSD/runs/Drop6_TCombo1Year_BAS_10GSD_split6_V43/lightning_logs/version_0/checkpoints/epoch=73-step=18944.ckpt

ls "$WORKDIR"/$DATASET_CODE/runs/$EXPERIMENT_NAME/lightning_logs/version_*/checkpoints/*.ckpt


## Continue training on toothbrush with new annotations and hard negatives
export CUDA_VISIBLE_DEVICES=1
DVC_DATA_DPATH=$(geowatch_dvc --tags='phase2_data' --hardware='auto')
DVC_EXPT_DPATH=$(geowatch_dvc --tags='phase2_expt' --hardware='auto')
echo "DVC_EXPT_DPATH = $DVC_EXPT_DPATH"
WORKDIR=$DVC_EXPT_DPATH/training/$HOSTNAME/$USER
DATASET_CODE=Drop6-MeanYear10GSD
KWCOCO_BUNDLE_DPATH=$DVC_DATA_DPATH/$DATASET_CODE
TRAIN_FPATH=$KWCOCO_BUNDLE_DPATH/data_train_rawbands_split6.kwcoco.zip
VALI_FPATH=$KWCOCO_BUNDLE_DPATH/data_vali_rawbands_split6.kwcoco.zip
CHANNELS="(L8,S2,PD):(blue|green|red|nir),(WV):(blue|green|red),(WV,WV1):pan"
EXPERIMENT_NAME=Drop6_TCombo1Year_BAS_10GSD_split6_V45
DEFAULT_ROOT_DIR=$WORKDIR/$DATASET_CODE/runs/$EXPERIMENT_NAME
TARGET_LR=5e-5
WEIGHT_DECAY=$(python -c "print($TARGET_LR * 0.01)")
echo "WEIGHT_DECAY = $WEIGHT_DECAY"
MAX_STEPS=80000
WATCH_GRID_WORKERS=4 python -m watch.tasks.fusion fit --config "
data:
    select_videos          : $SELECT_VIDEOS
    num_workers            : 5
    train_dataset          : $TRAIN_FPATH
    vali_dataset           : $VALI_FPATH
    window_dims            : '196,196'
    time_steps             : 11
    time_sampling          : uniform-soft5-soft4-contiguous
    time_kernel            : '(-3y,-2.5y,-2y,-1.5y,-1y,0,1y,1.5y,2y,2.5y,3y)'
    window_resolution     : 10.0GSD
    input_resolution      : 10.0GSD
    output_resolution     : 10.0GSD
    neg_to_pos_ratio       : 1.0
    batch_size             : 4
    normalize_perframe     : false
    normalize_peritem      : true
    max_epoch_length       : 1000000
    channels               : '$CHANNELS'
    min_spacetime_weight   : 0.6
    temporal_dropout       : 0.5
    mask_low_quality       : False
    mask_samecolor_method  : None
    observable_threshold   : 0.0
    quality_threshold      : 0.0
    weight_dilate          : 10
    use_centered_positives : True
    use_grid_positives     : True
    use_grid_negatives     : 'cleared'
    normalize_inputs       : 16384
    balance_areas          : True
model:
    class_path: MultimodalTransformer
    init_args:
        #saliency_weights       : '1:1'
        #class_weights          : auto
        tokenizer              : linconv
        arch_name              : smt_it_stm_p16
        decoder                : mlp
        positive_change_weight : 1
        negative_change_weight : 0.01
        stream_channels        : 16
        class_loss             : 'dicefocal'
        saliency_loss          : 'focal'
        saliency_head_hidden   : 6
        change_head_hidden     : 6
        class_head_hidden      : 6
        #global_change_weight   : 0.05
        global_class_weight    : 0.50
        global_saliency_weight : 1.00
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
    pct_start: 0.05
trainer:
    accumulate_grad_batches: 8
    default_root_dir     : $DEFAULT_ROOT_DIR
    accelerator          : gpu
    devices              : 0,
    #devices              : 0,1
    #strategy             : ddp
    limit_val_batches    : 256
    limit_train_batches  : 2048
    num_sanity_val_steps : 0
    max_epochs           : 360

torch_globals:
    float32_matmul_precision: auto

initializer:
    init: $DVC_EXPT_DPATH/models/fusion/Drop6-MeanYear10GSD/packages/Drop6_TCombo1Year_BAS_10GSD_split6_V42_cont2/Drop6_TCombo1Year_BAS_10GSD_split6_V42_cont2_epoch3_step941.pt
"


export CUDA_VISIBLE_DEVICES=0
DVC_DATA_DPATH=$(geowatch_dvc --tags='phase2_data' --hardware='auto')
DVC_EXPT_DPATH=$(geowatch_dvc --tags='phase2_expt' --hardware='auto')
echo "DVC_EXPT_DPATH = $DVC_EXPT_DPATH"
WORKDIR=$DVC_EXPT_DPATH/training/$HOSTNAME/$USER
DATASET_CODE=Drop6-MeanYear10GSD
KWCOCO_BUNDLE_DPATH=$DVC_DATA_DPATH/$DATASET_CODE
TRAIN_FPATH=$KWCOCO_BUNDLE_DPATH/data_train_rawbands_split6.kwcoco.zip
VALI_FPATH=$KWCOCO_BUNDLE_DPATH/data_vali_rawbands_split6.kwcoco.zip
CHANNELS="(L8,S2,PD):(blue|green|red|nir),(WV):(blue|green|red),(WV,WV1):pan"
EXPERIMENT_NAME=Drop6_TCombo1Year_BAS_10GSD_split6_V43_cont1
DEFAULT_ROOT_DIR=$WORKDIR/$DATASET_CODE/runs/$EXPERIMENT_NAME
TARGET_LR=3e-4
WEIGHT_DECAY=$(python -c "print($TARGET_LR * 0.01)")
echo "WEIGHT_DECAY = $WEIGHT_DECAY"
MAX_STEPS=80000
WATCH_GRID_WORKERS=0 python -m watch.tasks.fusion fit --config "
data:
    select_videos          : $SELECT_VIDEOS
    num_workers            : 5
    train_dataset          : $TRAIN_FPATH
    vali_dataset           : $VALI_FPATH
    window_dims            : '196,196'
    time_steps             : 11
    time_sampling          : uniform-soft5-soft4-contiguous
    time_kernel            : '(-3y,-2.5y,-2y,-1.5y,-1y,0,1y,1.5y,2y,2.5y,3y)'
    window_resolution     : 10.0GSD
    input_resolution      : 10.0GSD
    output_resolution     : 10.0GSD
    neg_to_pos_ratio       : 1.0
    batch_size             : 4
    normalize_perframe     : false
    normalize_peritem      : true
    max_epoch_length       : 1000000
    channels               : '$CHANNELS'
    min_spacetime_weight   : 0.6
    temporal_dropout       : 0.5
    mask_low_quality       : False
    mask_samecolor_method  : None
    observable_threshold   : 0.0
    quality_threshold      : 0.0
    weight_dilate          : 10
    use_centered_positives : True
    use_grid_positives     : True
    use_grid_negatives     : 'cleared'
    normalize_inputs       : 16384
    balance_areas          : True
model:
    class_path: MultimodalTransformer
    init_args:
        #saliency_weights       : '1:1'
        #class_weights          : auto
        tokenizer              : linconv
        arch_name              : smt_it_stm_p16
        decoder                : mlp
        positive_change_weight : 1
        negative_change_weight : 0.01
        stream_channels        : 16
        class_loss             : 'dicefocal'
        saliency_loss          : 'focal'
        saliency_head_hidden   : 6
        change_head_hidden     : 6
        class_head_hidden      : 6
        global_change_weight   : 0.00
        global_class_weight    : 0.00
        global_saliency_weight : 1.00
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
    pct_start: 0.05
trainer:
    accumulate_grad_batches: 64
    default_root_dir     : $DEFAULT_ROOT_DIR
    accelerator          : gpu
    devices              : 0,
    #devices              : 0,1
    #strategy             : ddp
    limit_val_batches    : 256
    limit_train_batches  : 2048
    num_sanity_val_steps : 0
    max_epochs           : 360

torch_globals:
    float32_matmul_precision: auto

initializer:
    init: /home/joncrall/remote/toothbrush/data/dvc-repos/smart_expt_dvc/training/toothbrush/joncrall/Drop6-MeanYear10GSD/runs/Drop6_TCombo1Year_BAS_10GSD_split6_V43/lightning_logs/version_0/checkpoints/epoch=73-step=18944.ckpt
" --ckpt_path=/home/joncrall/remote/toothbrush/data/dvc-repos/smart_expt_dvc/training/toothbrush/joncrall/Drop6-MeanYear10GSD/runs/Drop6_TCombo1Year_BAS_10GSD_split6_V43_cont1/lightning_logs/version_0/checkpoints/epoch=55-step=1792.ckpt


# On yardrat
export CUDA_VISIBLE_DEVICES=0
DVC_DATA_DPATH=$(geowatch_dvc --tags='phase2_data' --hardware='auto')
DVC_EXPT_DPATH=$(geowatch_dvc --tags='phase2_expt' --hardware='auto')
echo "DVC_EXPT_DPATH = $DVC_EXPT_DPATH"
WORKDIR=$DVC_EXPT_DPATH/training/$HOSTNAME/$USER
DATASET_CODE=Drop6-MeanYear10GSD
KWCOCO_BUNDLE_DPATH=$DVC_DATA_DPATH/$DATASET_CODE
TRAIN_FPATH=$KWCOCO_BUNDLE_DPATH/data_train_rawbands_split6.kwcoco.zip
VALI_FPATH=$KWCOCO_BUNDLE_DPATH/data_vali_rawbands_split6.kwcoco.zip
CHANNELS="(L8,S2,PD):(blue|green|red|nir),(WV):(blue|green|red),(WV,WV1):pan"
EXPERIMENT_NAME=Drop6_TCombo1Year_BAS_10GSD_split6_V46
DEFAULT_ROOT_DIR=$WORKDIR/$DATASET_CODE/runs/$EXPERIMENT_NAME
TARGET_LR=1e-5
WEIGHT_DECAY=$(python -c "print($TARGET_LR * 0.01)")
echo "WEIGHT_DECAY = $WEIGHT_DECAY"
MAX_STEPS=80000
WATCH_GRID_WORKERS=4 python -m watch.tasks.fusion fit --config "
data:
    select_videos          : $SELECT_VIDEOS
    num_workers            : 5
    train_dataset          : $TRAIN_FPATH
    vali_dataset           : $VALI_FPATH
    window_dims            : '196,196'
    time_steps             : 11
    time_sampling          : uniform-soft5-soft4-contiguous
    time_kernel            : '(-3y,-2.5y,-2y,-1.5y,-1y,0,1y,1.5y,2y,2.5y,3y)'
    window_resolution     : 10.0GSD
    input_resolution      : 10.0GSD
    output_resolution     : 10.0GSD
    neg_to_pos_ratio       : 1.0
    batch_size             : 3
    normalize_perframe     : false
    normalize_peritem      : true
    max_epoch_length       : 1000000
    channels               : '$CHANNELS'
    min_spacetime_weight   : 0.6
    temporal_dropout       : 0.5
    mask_low_quality       : False
    mask_samecolor_method  : None
    observable_threshold   : 0.0
    quality_threshold      : 0.0
    weight_dilate          : 10
    use_centered_positives : True
    use_grid_positives     : True
    use_grid_negatives     : 'cleared'
    normalize_inputs       : 16384
    balance_areas          : True
model:
    class_path: MultimodalTransformer
    init_args:
        #saliency_weights       : '1:1'
        #class_weights          : auto
        tokenizer              : linconv
        arch_name              : smt_it_stm_p16
        decoder                : mlp
        positive_change_weight : 1
        negative_change_weight : 0.01
        stream_channels        : 16
        class_loss             : 'dicefocal'
        saliency_loss          : 'focal'
        saliency_head_hidden   : 6
        change_head_hidden     : 6
        class_head_hidden      : 6
        #global_change_weight   : 0.05
        global_class_weight    : 0.50
        global_saliency_weight : 1.00
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
    pct_start: 0.05
trainer:
    accumulate_grad_batches: 11
    default_root_dir     : $DEFAULT_ROOT_DIR
    accelerator          : gpu
    devices              : 0,
    #devices              : 0,1
    #strategy             : ddp
    limit_val_batches    : 256
    limit_train_batches  : 2048
    num_sanity_val_steps : 0
    max_epochs           : 360

torch_globals:
    float32_matmul_precision: auto

initializer:
    init: $DVC_EXPT_DPATH/models/fusion/Drop6-MeanYear10GSD/packages/Drop6_TCombo1Year_BAS_10GSD_split6_V42_cont2/Drop6_TCombo1Year_BAS_10GSD_split6_V42_cont2_epoch3_step941.pt
"
DVC_EXPT_DPATH=$(geowatch_dvc --tags='phase2_expt' --hardware='auto')
rsync -avprPR yardrat:data/dvc-repos/smart_expt_dvc/./training/yardrat/jon.crall/Drop6-MeanYear10GSD/runs/Drop6_TCombo1Year_BAS_10GSD_split6_V46/lightning_logs/version_0/monitor/tensorboard "$DVC_EXPT_DPATH"
rsync -avprPR yardrat:data/dvc-repos/smart_expt_dvc/./training/yardrat/jon.crall/Drop6-MeanYear10GSD/runs/Drop6_TCombo1Year_BAS_10GSD_split6_V46 "$DVC_EXPT_DPATH"


# On toothbrush (split6 starting from namek point)
export CUDA_VISIBLE_DEVICES=1
DVC_DATA_DPATH=$(geowatch_dvc --tags='phase2_data' --hardware='auto')
DVC_EXPT_DPATH=$(geowatch_dvc --tags='phase2_expt' --hardware='auto')
echo "DVC_EXPT_DPATH = $DVC_EXPT_DPATH"
WORKDIR=$DVC_EXPT_DPATH/training/$HOSTNAME/$USER
DATASET_CODE=Drop6-MeanYear10GSD-V2
KWCOCO_BUNDLE_DPATH=$DVC_DATA_DPATH/$DATASET_CODE
TRAIN_FPATH=$KWCOCO_BUNDLE_DPATH/data_train_I2L_split6.kwcoco.zip
VALI_FPATH=$KWCOCO_BUNDLE_DPATH/data_vali_I2L_split6.kwcoco.zip
CHANNELS="(L8,S2):(blue|green|red|nir),(WV):(blue|green|red),(WV,WV1):pan,(S2):(water|forest|field|impervious|barren|landcover_hidden.0:32)"
EXPERIMENT_NAME=Drop6_TCombo1Year_BAS_10GSD_V2_landcover_split6_V47
DEFAULT_ROOT_DIR=$WORKDIR/$DATASET_CODE/runs/$EXPERIMENT_NAME
TARGET_LR=3e-4
WEIGHT_DECAY=$(python -c "print($TARGET_LR * 0.01)")
echo "WEIGHT_DECAY = $WEIGHT_DECAY"
MAX_STEPS=80000
WATCH_GRID_WORKERS=0 python -m watch.tasks.fusion fit --config "
data:
    select_videos          : $SELECT_VIDEOS
    num_workers            : 5
    train_dataset          : $TRAIN_FPATH
    vali_dataset           : $VALI_FPATH
    window_dims            : '196,196'
    time_steps             : 11
    time_sampling          : uniform-soft5-soft4-contiguous
    time_kernel            : '(-3y,-2.5y,-2y,-1.5y,-1y,0,1y,1.5y,2y,2.5y,3y)'
    window_resolution     : 10.0GSD
    input_resolution      : 10.0GSD
    output_resolution     : 10.0GSD
    neg_to_pos_ratio       : 1.0
    batch_size             : 3
    normalize_perframe     : false
    normalize_peritem      : 'blue|green|red|nir|pan'
    max_epoch_length       : 1000000
    channels               : '$CHANNELS'
    min_spacetime_weight   : 0.6
    temporal_dropout       : 0.5
    mask_low_quality       : False
    mask_samecolor_method  : None
    observable_threshold   : 0.0
    quality_threshold      : 0.0
    weight_dilate          : 10
    use_centered_positives : True
    use_grid_positives     : True
    use_grid_negatives     : 'cleared'
    normalize_inputs       : 1024
    balance_areas          : True
model:
    class_path: MultimodalTransformer
    init_args:
        #saliency_weights       : '1:1'
        #class_weights          : auto
        tokenizer              : linconv
        arch_name              : smt_it_stm_p16
        decoder                : mlp
        positive_change_weight : 1
        negative_change_weight : 0.01
        stream_channels        : 16
        class_loss             : 'dicefocal'
        saliency_loss          : 'focal'
        saliency_head_hidden   : 6
        change_head_hidden     : 6
        class_head_hidden      : 6
        global_change_weight   : 0.00
        global_class_weight    : 0.50
        global_saliency_weight : 1.00
        multimodal_reduce      : learned_linear
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
    pct_start: 0.05
trainer:
    accumulate_grad_batches: 32
    default_root_dir     : $DEFAULT_ROOT_DIR
    accelerator          : gpu
    devices              : 0,
    #devices              : 0,1
    #strategy             : ddp
    limit_val_batches    : 256
    limit_train_batches  : 2048
    num_sanity_val_steps : 0
    max_epochs           : 360

torch_globals:
    float32_matmul_precision: auto

initializer:
    init: $DVC_EXPT_DPATH/models/fusion/Drop6-MeanYear10GSD/packages/Drop6_TCombo1Year_BAS_10GSD_split6_V46/Drop6_TCombo1Year_BAS_10GSD_split6_V46_epoch118_step22253.pt
"


export CUDA_VISIBLE_DEVICES=1
DVC_DATA_DPATH=$(geowatch_dvc --tags='phase2_data' --hardware='auto')
DVC_EXPT_DPATH=$(geowatch_dvc --tags='phase2_expt' --hardware='auto')
echo "DVC_EXPT_DPATH = $DVC_EXPT_DPATH"
WORKDIR=$DVC_EXPT_DPATH/training/$HOSTNAME/$USER
DATASET_CODE=Drop6-MeanYear10GSD-V2
KWCOCO_BUNDLE_DPATH=$DVC_DATA_DPATH/$DATASET_CODE
TRAIN_FPATH=$KWCOCO_BUNDLE_DPATH/data_train_I2L_split6.kwcoco.zip
VALI_FPATH=$KWCOCO_BUNDLE_DPATH/data_vali_I2L_split6.kwcoco.zip
CHANNELS="(L8,S2):(blue|green|red|nir),(WV):(blue|green|red),(WV,WV1):pan,(S2):(water|forest|field|impervious|barren|landcover_hidden.0:32)"
EXPERIMENT_NAME=Drop6_TCombo1Year_BAS_10GSD_V2_landcover_split6_V48
DEFAULT_ROOT_DIR=$WORKDIR/$DATASET_CODE/runs/$EXPERIMENT_NAME
TARGET_LR=1e-4
WEIGHT_DECAY=$(python -c "print($TARGET_LR * 0.01)")
echo "WEIGHT_DECAY = $WEIGHT_DECAY"
MAX_STEPS=80000
WATCH_GRID_WORKERS=2 python -m watch.tasks.fusion fit --config "
data:
    select_videos          : $SELECT_VIDEOS
    num_workers            : 5
    train_dataset          : $TRAIN_FPATH
    vali_dataset           : $VALI_FPATH
    window_dims            : '196,196'
    time_steps             : 11
    time_sampling          : uniform-soft5-soft4-contiguous
    time_kernel            : '(-3y,-2.5y,-2y,-1.5y,-1y,0,1y,1.5y,2y,2.5y,3y)'
    window_resolution     : 10.0GSD
    input_resolution      : 10.0GSD
    output_resolution     : 10.0GSD
    neg_to_pos_ratio       : 1.0
    batch_size             : 4
    normalize_perframe     : false
    normalize_peritem      : 'blue|green|red|nir|pan'
    max_epoch_length       : 1000000
    channels               : '$CHANNELS'
    min_spacetime_weight   : 0.6
    temporal_dropout       : 0.5
    modality_dropout       : 0.5
    mask_low_quality       : False
    mask_samecolor_method  : None
    observable_threshold   : 0.0
    quality_threshold      : 0.0
    weight_dilate          : 10
    use_centered_positives : True
    use_grid_positives     : True
    use_grid_negatives     : 'cleared'
    normalize_inputs       : 1024
    balance_areas          : True
model:
    class_path: MultimodalTransformer
    init_args:
        #saliency_weights       : '1:1'
        #class_weights          : auto
        tokenizer              : linconv
        arch_name              : smt_it_stm_p16
        decoder                : mlp
        positive_change_weight : 1
        negative_change_weight : 0.01
        stream_channels        : 16
        class_loss             : 'dicefocal'
        saliency_loss          : 'focal'
        saliency_head_hidden   : 6
        change_head_hidden     : 6
        class_head_hidden      : 6
        global_change_weight   : 0.00
        global_class_weight    : 0.01
        global_saliency_weight : 1.00
        multimodal_reduce      : learned_linear
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
    pct_start: 0.05
trainer:
    accumulate_grad_batches: 32
    default_root_dir     : $DEFAULT_ROOT_DIR
    accelerator          : gpu
    devices              : 0,
    #devices              : 0,1
    #strategy             : ddp
    limit_val_batches    : 256
    limit_train_batches  : 2048
    num_sanity_val_steps : 0
    max_epochs           : 360

torch_globals:
    float32_matmul_precision: auto

initializer:
    init: $DVC_EXPT_DPATH/models/fusion/Drop6-MeanYear10GSD/packages/Drop6_TCombo1Year_BAS_10GSD_split6_V46/Drop6_TCombo1Year_BAS_10GSD_split6_V46_epoch118_step22253.pt
"


export CUDA_VISIBLE_DEVICES=1
DVC_DATA_DPATH=$(geowatch_dvc --tags='phase2_data' --hardware='auto')
DVC_EXPT_DPATH=$(geowatch_dvc --tags='phase2_expt' --hardware='auto')
echo "DVC_EXPT_DPATH = $DVC_EXPT_DPATH"
WORKDIR=$DVC_EXPT_DPATH/training/$HOSTNAME/$USER
DATASET_CODE=Drop6-MeanYear10GSD-V2
KWCOCO_BUNDLE_DPATH=$DVC_DATA_DPATH/$DATASET_CODE
TRAIN_FPATH=$KWCOCO_BUNDLE_DPATH/data_train_I2L_split6.kwcoco.zip
VALI_FPATH=$KWCOCO_BUNDLE_DPATH/data_vali_I2L_split6.kwcoco.zip
CHANNELS="(L8,S2):(blue|green|red|nir),(WV):(blue|green|red),(WV,WV1):pan,(S2):(water|forest|field|impervious|barren|landcover_hidden.0:32)"
EXPERIMENT_NAME=Drop6_TCombo1Year_BAS_10GSD_V2_landcover_split6_V49
DEFAULT_ROOT_DIR=$WORKDIR/$DATASET_CODE/runs/$EXPERIMENT_NAME
TARGET_LR=1e-5
WEIGHT_DECAY=$(python -c "print($TARGET_LR * 0.01)")
echo "WEIGHT_DECAY = $WEIGHT_DECAY"
MAX_STEPS=80000
WATCH_GRID_WORKERS=2 python -m watch.tasks.fusion fit --config "
data:
    select_videos          : $SELECT_VIDEOS
    num_workers            : 5
    train_dataset          : $TRAIN_FPATH
    vali_dataset           : $VALI_FPATH
    window_dims            : '196,196'
    time_steps             : 11
    time_sampling          : uniform-soft5-soft4-contiguous
    time_kernel            : '(-3y,-2.5y,-2y,-1.5y,-1y,0,1y,1.5y,2y,2.5y,3y)'
    window_resolution     : 10.0GSD
    input_resolution      : 10.0GSD
    output_resolution     : 10.0GSD
    neg_to_pos_ratio       : 1.0
    batch_size             : 4
    normalize_perframe     : false
    normalize_peritem      : 'blue|green|red|nir|pan'
    max_epoch_length       : 1000000
    channels               : '$CHANNELS'
    min_spacetime_weight   : 0.6
    temporal_dropout       : 0.5
    modality_dropout       : 0.5
    mask_low_quality       : False
    mask_samecolor_method  : None
    observable_threshold   : 0.0
    quality_threshold      : 0.0
    weight_dilate          : 10
    use_centered_positives : True
    use_grid_positives     : True
    use_grid_negatives     : 'cleared'
    normalize_inputs       : 1024
    balance_areas          : True
model:
    class_path: MultimodalTransformer
    init_args:
        #saliency_weights       : '1:1'
        #class_weights          : auto
        tokenizer              : linconv
        arch_name              : smt_it_stm_p16
        decoder                : mlp
        positive_change_weight : 1
        negative_change_weight : 0.01
        stream_channels        : 16
        class_loss             : 'dicefocal'
        saliency_loss          : 'focal'
        saliency_head_hidden   : 6
        change_head_hidden     : 6
        class_head_hidden      : 6
        global_change_weight   : 0.00
        global_class_weight    : 0.01
        global_saliency_weight : 1.00
        multimodal_reduce      : learned_linear
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
    pct_start: 0.05
trainer:
    accumulate_grad_batches: 64
    default_root_dir     : $DEFAULT_ROOT_DIR
    accelerator          : gpu
    devices              : 0,
    #devices              : 0,1
    #strategy             : ddp
    limit_val_batches    : 256
    limit_train_batches  : 2048
    num_sanity_val_steps : 0
    max_epochs           : 360

torch_globals:
    float32_matmul_precision: auto

initializer:
    init: $DVC_EXPT_DPATH/models/fusion/Drop6-MeanYear10GSD/packages/Drop6_TCombo1Year_BAS_10GSD_split6_V46/Drop6_TCombo1Year_BAS_10GSD_split6_V46_epoch118_step22253.pt
"


export CUDA_VISIBLE_DEVICES=1
DVC_DATA_DPATH=$(geowatch_dvc --tags='phase2_data' --hardware='auto')
DVC_EXPT_DPATH=$(geowatch_dvc --tags='phase2_expt' --hardware='auto')
echo "DVC_EXPT_DPATH = $DVC_EXPT_DPATH"
WORKDIR=$DVC_EXPT_DPATH/training/$HOSTNAME/$USER
DATASET_CODE=Drop6-MeanYear10GSD-V2
KWCOCO_BUNDLE_DPATH=$DVC_DATA_DPATH/$DATASET_CODE
TRAIN_FPATH=$KWCOCO_BUNDLE_DPATH/data_train_I2L_split6.kwcoco.zip
VALI_FPATH=$KWCOCO_BUNDLE_DPATH/data_vali_I2L_split6.kwcoco.zip
CHANNELS="(L8,S2):(blue|green|red|nir),(WV):(blue|green|red),(WV,WV1):pan,(S2):(water|forest|field|impervious|barren|landcover_hidden.0:32)"
EXPERIMENT_NAME=Drop6_TCombo1Year_BAS_10GSD_V2_landcover_split6_V50
DEFAULT_ROOT_DIR=$WORKDIR/$DATASET_CODE/runs/$EXPERIMENT_NAME
TARGET_LR=5e-6
WEIGHT_DECAY=$(python -c "print($TARGET_LR * 0.01)")
echo "WEIGHT_DECAY = $WEIGHT_DECAY"
MAX_STEPS=80000
WATCH_GRID_WORKERS=2 python -m watch.tasks.fusion fit --config "
data:
    select_videos          : $SELECT_VIDEOS
    num_workers            : 5
    train_dataset          : $TRAIN_FPATH
    vali_dataset           : $VALI_FPATH
    window_dims            : '196,196'
    time_steps             : 11
    time_sampling          : uniform-soft5-soft4-contiguous
    time_kernel            : '(-3y,-2.5y,-2y,-1.5y,-1y,0,1y,1.5y,2y,2.5y,3y)'
    window_resolution     : 10.0GSD
    input_resolution      : 10.0GSD
    output_resolution     : 10.0GSD
    neg_to_pos_ratio       : 1.0
    batch_size             : 4
    normalize_perframe     : false
    normalize_peritem      : 'blue|green|red|nir|pan'
    max_epoch_length       : 1000000
    channels               : '$CHANNELS'
    min_spacetime_weight   : 0.6
    temporal_dropout       : 0.5
    modality_dropout       : 0.5
    mask_low_quality       : False
    mask_samecolor_method  : None
    observable_threshold   : 0.0
    quality_threshold      : 0.0
    weight_dilate          : 10
    absolute_weighting     : True
    use_centered_positives : True
    use_grid_positives     : True
    use_grid_negatives     : 'cleared'
    normalize_inputs       : 1024
    balance_areas          : True
model:
    class_path: MultimodalTransformer
    init_args:
        #saliency_weights       : '1:1'
        #class_weights          : auto
        tokenizer              : linconv
        arch_name              : smt_it_stm_p16
        decoder                : mlp
        positive_change_weight : 1
        negative_change_weight : 0.01
        stream_channels        : 16
        class_loss             : 'dicefocal'
        saliency_loss          : 'focal'
        saliency_head_hidden   : 6
        change_head_hidden     : 6
        class_head_hidden      : 6
        global_change_weight   : 0.00
        global_class_weight    : 0.01
        global_saliency_weight : 1.00
        multimodal_reduce      : learned_linear
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
    pct_start: 0.05
trainer:
    accumulate_grad_batches: 64
    default_root_dir     : $DEFAULT_ROOT_DIR
    accelerator          : gpu
    devices              : 0,
    #devices              : 0,1
    #strategy             : ddp
    limit_val_batches    : 256
    limit_train_batches  : 2048
    num_sanity_val_steps : 0
    max_epochs           : 360

torch_globals:
    float32_matmul_precision: auto

initializer:
    #init: $DVC_EXPT_DPATH/models/fusion/Drop6-MeanYear10GSD-V2/packages/Drop6_TCombo1Year_BAS_10GSD_V2_landcover_split6_V47/Drop6_TCombo1Year_BAS_10GSD_V2_landcover_split6_V47_epoch47_step3026.pt
    #init: /home/joncrall/remote/toothbrush/data/dvc-repos/smart_expt_dvc/training/toothbrush/joncrall/Drop6-MeanYear10GSD-V2/runs/Drop6_TCombo1Year_BAS_10GSD_V2_landcover_split6_V50/lightning_logs/version_0/checkpoints/epoch=77-step=1248.ckpt
    init: /home/joncrall/remote/toothbrush/data/dvc-repos/smart_expt_dvc/training/toothbrush/joncrall/Drop6-MeanYear10GSD-V2/runs/Drop6_TCombo1Year_BAS_10GSD_V2_landcover_split6_V50/lightning_logs/version_2/checkpoints/epoch=8-step=144.ckpt
"

# On Yardrat
export CUDA_VISIBLE_DEVICES=0
DVC_DATA_DPATH=$(geowatch_dvc --tags='phase2_data' --hardware='auto')
DVC_EXPT_DPATH=$(geowatch_dvc --tags='phase2_expt' --hardware='auto')
echo "DVC_EXPT_DPATH = $DVC_EXPT_DPATH"
WORKDIR=$DVC_EXPT_DPATH/training/$HOSTNAME/$USER
DATASET_CODE=Drop6-MeanYear10GSD-V2
KWCOCO_BUNDLE_DPATH=$DVC_DATA_DPATH/$DATASET_CODE
TRAIN_FPATH=$KWCOCO_BUNDLE_DPATH/data_train_I2L_split6.kwcoco.zip
VALI_FPATH=$KWCOCO_BUNDLE_DPATH/data_vali_I2L_split6.kwcoco.zip
CHANNELS="(L8,S2):(blue|green|red|nir),(WV):(blue|green|red),(WV,WV1):pan,(S2):(water|forest|field|impervious|barren|landcover_hidden.0:32)"
EXPERIMENT_NAME=Drop6_TCombo1Year_BAS_10GSD_V2_landcover_split6_V51
DEFAULT_ROOT_DIR=$WORKDIR/$DATASET_CODE/runs/$EXPERIMENT_NAME
TARGET_LR=1e-5
WEIGHT_DECAY=$(python -c "print($TARGET_LR * 0.01)")
echo "WEIGHT_DECAY = $WEIGHT_DECAY"
MAX_STEPS=80000
WATCH_GRID_WORKERS=2 python -m watch.tasks.fusion fit --config "
data:
    select_videos          : $SELECT_VIDEOS
    num_workers            : 5
    train_dataset          : $TRAIN_FPATH
    vali_dataset           : $VALI_FPATH
    window_dims            : '196,196'
    time_steps             : 11
    time_sampling          : uniform-soft5-soft4-contiguous
    time_kernel            : '(-3y,-2.5y,-2y,-1.5y,-1y,0,1y,1.5y,2y,2.5y,3y)'
    window_resolution     : 10.0GSD
    input_resolution      : 10.0GSD
    output_resolution     : 10.0GSD
    neg_to_pos_ratio       : 1.0
    batch_size             : 2
    normalize_perframe     : false
    normalize_peritem      : 'blue|green|red|nir|pan'
    max_epoch_length       : 1000000
    channels               : '$CHANNELS'
    min_spacetime_weight   : 0.6
    temporal_dropout       : 0.5
    absolute_weighting     : True
    modality_dropout       : 0.5
    mask_low_quality       : False
    mask_samecolor_method  : None
    observable_threshold   : 0.0
    quality_threshold      : 0.0
    weight_dilate          : 10
    use_centered_positives : True
    use_grid_positives     : True
    #use_grid_negatives     : 'cleared'
    normalize_inputs       : 1024
    balance_areas          : True
model:
    class_path: MultimodalTransformer
    init_args:
        #saliency_weights       : '1:1'
        #class_weights          : auto
        tokenizer              : linconv
        arch_name              : smt_it_stm_p16
        decoder                : mlp
        positive_change_weight : 1
        negative_change_weight : 0.01
        stream_channels        : 16
        class_loss             : 'dicefocal'
        saliency_loss          : 'focal'
        saliency_head_hidden   : 6
        change_head_hidden     : 6
        class_head_hidden      : 6
        global_change_weight   : 0.00
        global_class_weight    : 0.01
        global_saliency_weight : 1.00
        multimodal_reduce      : learned_linear
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
    pct_start: 0.05
trainer:
    accumulate_grad_batches: 128
    default_root_dir     : $DEFAULT_ROOT_DIR
    accelerator          : gpu
    devices              : 0,
    #devices              : 0,1
    #strategy             : ddp
    limit_val_batches    : 256
    limit_train_batches  : 2048
    num_sanity_val_steps : 0
    max_epochs           : 360

torch_globals:
    float32_matmul_precision: auto

initializer:
    init: $DVC_EXPT_DPATH/models/fusion/Drop6-MeanYear10GSD/packages/Drop6_TCombo1Year_BAS_10GSD_split6_V46/Drop6_TCombo1Year_BAS_10GSD_split6_V46_epoch118_step22253.pt
"


# On Horologic
export CUDA_VISIBLE_DEVICES=3
DVC_DATA_DPATH=$(geowatch_dvc --tags='phase2_data' --hardware='auto')
DVC_EXPT_DPATH=$(geowatch_dvc --tags='phase2_expt' --hardware='auto')
echo "DVC_EXPT_DPATH = $DVC_EXPT_DPATH"
WORKDIR=$DVC_EXPT_DPATH/training/$HOSTNAME/$USER
DATASET_CODE=Drop6-MeanYear10GSD-V2
KWCOCO_BUNDLE_DPATH=$DVC_DATA_DPATH/$DATASET_CODE
TRAIN_FPATH=$KWCOCO_BUNDLE_DPATH/data_train_I2L_split6.kwcoco.zip
VALI_FPATH=$KWCOCO_BUNDLE_DPATH/data_vali_I2L_split6.kwcoco.zip
CHANNELS="(L8,S2):(blue|green|red|nir),(WV):(blue|green|red),(WV,WV1):pan,(S2):(water|forest|field|impervious|barren|landcover_hidden.0:32)"
EXPERIMENT_NAME=Drop6_TCombo1Year_BAS_10GSD_V2_landcover_split6_V51
DEFAULT_ROOT_DIR=$WORKDIR/$DATASET_CODE/runs/$EXPERIMENT_NAME
TARGET_LR=1e-3
WEIGHT_DECAY=$(python -c "print($TARGET_LR * 0.01)")
echo "WEIGHT_DECAY = $WEIGHT_DECAY"
MAX_STEPS=80000
WATCH_GRID_WORKERS=2 python -m watch.tasks.fusion fit --config "
data:
    select_videos          : $SELECT_VIDEOS
    num_workers            : 5
    train_dataset          : $TRAIN_FPATH
    vali_dataset           : $VALI_FPATH
    window_dims            : '196,196'
    time_steps             : 11
    time_sampling          : uniform-soft5-soft4-contiguous
    time_kernel            : '(-3y,-2.5y,-2y,-1.5y,-1y,0,1y,1.5y,2y,2.5y,3y)'
    window_resolution     : 10.0GSD
    input_resolution      : 10.0GSD
    output_resolution     : 10.0GSD
    neg_to_pos_ratio       : 1.0
    batch_size             : 4
    normalize_perframe     : false
    normalize_peritem      : 'blue|green|red|nir|pan'
    max_epoch_length       : 1000000
    channels               : '$CHANNELS'
    min_spacetime_weight   : 0.6
    temporal_dropout       : 0.1
    absolute_weighting     : True
    modality_dropout       : 0.1
    mask_low_quality       : False
    mask_samecolor_method  : None
    observable_threshold   : 0.0
    quality_threshold      : 0.0
    weight_dilate          : 10
    use_centered_positives : True
    use_grid_positives     : True
    #use_grid_negatives     : 'cleared'
    normalize_inputs       : 1024
    balance_areas          : True
model:
    class_path: MultimodalTransformer
    init_args:
        #saliency_weights       : '1:1'
        #class_weights          : auto
        tokenizer              : linconv
        arch_name              : smt_it_stm_p16
        decoder                : mlp
        positive_change_weight : 1
        negative_change_weight : 0.01
        stream_channels        : 16
        class_loss             : 'dicefocal'
        saliency_loss          : 'focal'
        saliency_head_hidden   : 6
        change_head_hidden     : 6
        class_head_hidden      : 6
        global_change_weight   : 0.00
        global_class_weight    : 0.01
        global_saliency_weight : 1.00
        multimodal_reduce      : learned_linear
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
    pct_start: 0.05
trainer:
    accumulate_grad_batches: 128
    default_root_dir     : $DEFAULT_ROOT_DIR
    accelerator          : gpu
    devices              : 0,
    #devices              : 0,1
    #strategy             : ddp
    limit_val_batches    : 256
    limit_train_batches  : 2048
    num_sanity_val_steps : 0
    max_epochs           : 360

torch_globals:
    float32_matmul_precision: auto

initializer:
    init: $DVC_EXPT_DPATH/models/fusion/Drop6-MeanYear10GSD/packages/Drop6_TCombo1Year_BAS_10GSD_split6_V46/Drop6_TCombo1Year_BAS_10GSD_split6_V46_epoch118_step22253.pt
"


export CUDA_VISIBLE_DEVICES=1
DVC_DATA_DPATH=$(geowatch_dvc --tags='phase2_data' --hardware='auto')
DVC_EXPT_DPATH=$(geowatch_dvc --tags='phase2_expt' --hardware='auto')
echo "DVC_EXPT_DPATH = $DVC_EXPT_DPATH"
WORKDIR=$DVC_EXPT_DPATH/training/$HOSTNAME/$USER
DATASET_CODE=Drop6-MeanYear10GSD-V2
KWCOCO_BUNDLE_DPATH=$DVC_DATA_DPATH/$DATASET_CODE
TRAIN_FPATH=$KWCOCO_BUNDLE_DPATH/data_train_I2LS_split6.kwcoco.zip
VALI_FPATH=$KWCOCO_BUNDLE_DPATH/data_vali_I2LS_split6.kwcoco.zip
CHANNELS="(L8,S2):(blue|green|red|nir),(WV):(blue|green|red),(WV,WV1):pan,(S2):(water|forest|field|impervious|barren|landcover_hidden.0:32),(L8,S2,WV,WV1):(sam.0:32)"
EXPERIMENT_NAME=Drop6_TCombo1Year_BAS_10GSD_V2_sam_landcover_split6_V52
DEFAULT_ROOT_DIR=$WORKDIR/$DATASET_CODE/runs/$EXPERIMENT_NAME
TARGET_LR=1e-4
WEIGHT_DECAY=$(python -c "print($TARGET_LR * 0.01)")
echo "WEIGHT_DECAY = $WEIGHT_DECAY"
MAX_STEPS=80000
WATCH_GRID_WORKERS=2 python -m watch.tasks.fusion fit --config "
data:
    select_videos          : $SELECT_VIDEOS
    num_workers            : 6
    train_dataset          : $TRAIN_FPATH
    vali_dataset           : $VALI_FPATH
    window_dims            : '196,196'
    time_steps             : 11
    time_sampling          : uniform-soft5-soft4-contiguous
    time_kernel            : '(-3y,-2.5y,-2y,-1.5y,-1y,0,1y,1.5y,2y,2.5y,3y)'
    window_resolution     : 10.0GSD
    input_resolution      : 10.0GSD
    output_resolution     : 10.0GSD
    neg_to_pos_ratio       : 1.0
    batch_size             : 2
    normalize_perframe     : false
    normalize_peritem      : 'blue|green|red|nir|pan'
    max_epoch_length       : 1000000
    channels               : '$CHANNELS'
    min_spacetime_weight   : 0.6
    temporal_dropout       : 0.5
    modality_dropout       : 0.5
    mask_low_quality       : False
    mask_samecolor_method  : None
    observable_threshold   : 0.0
    quality_threshold      : 0.0
    weight_dilate          : 10
    absolute_weighting     : True
    use_centered_positives : True
    use_grid_positives     : True
    use_grid_negatives     : 'cleared'
    normalize_inputs       : 1024
    balance_areas          : True
model:
    class_path: MultimodalTransformer
    init_args:
        #saliency_weights       : '1:1'
        #class_weights          : auto
        tokenizer              : linconv
        arch_name              : smt_it_stm_p16
        decoder                : mlp
        positive_change_weight : 1
        negative_change_weight : 0.01
        stream_channels        : 16
        class_loss             : 'dicefocal'
        saliency_loss          : 'focal'
        saliency_head_hidden   : 6
        change_head_hidden     : 6
        class_head_hidden      : 6
        global_change_weight   : 0.00
        global_class_weight    : 0.01
        global_saliency_weight : 1.00
        multimodal_reduce      : learned_linear
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
    pct_start: 0.05
trainer:
    accumulate_grad_batches: 64
    default_root_dir     : $DEFAULT_ROOT_DIR
    accelerator          : gpu
    devices              : 0,
    #devices              : 0,1
    #strategy             : ddp
    limit_val_batches    : 256
    limit_train_batches  : 2048
    num_sanity_val_steps : 0
    max_epochs           : 360

torch_globals:
    float32_matmul_precision: auto

initializer:
    init: /home/joncrall/remote/toothbrush/data/dvc-repos/smart_expt_dvc/training/toothbrush/joncrall/Drop6-MeanYear10GSD-V2/runs/Drop6_TCombo1Year_BAS_10GSD_V2_landcover_split6_V50/lightning_logs/version_8/package-interupt/package_epoch97_step3132.pt

    #init: $DVC_EXPT_DPATH/models/fusion/Drop6-MeanYear10GSD-V2/packages/Drop6_TCombo1Year_BAS_10GSD_V2_landcover_split6_V47/Drop6_TCombo1Year_BAS_10GSD_V2_landcover_split6_V47_epoch47_step3026.pt
    #init: /home/joncrall/remote/toothbrush/data/dvc-repos/smart_expt_dvc/training/toothbrush/joncrall/Drop6-MeanYear10GSD-V2/runs/Drop6_TCombo1Year_BAS_10GSD_V2_landcover_split6_V50/lightning_logs/version_0/checkpoints/epoch=77-step=1248.ckpt
    #init: /home/joncrall/remote/toothbrush/data/dvc-repos/smart_expt_dvc/training/toothbrush/joncrall/Drop6-MeanYear10GSD-V2/runs/Drop6_TCombo1Year_BAS_10GSD_V2_landcover_split6_V50/lightning_logs/version_2/checkpoints/epoch=8-step=144.ckpt
"
