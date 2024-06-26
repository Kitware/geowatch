#!/bin/bash

PHASE2_DATA_DPATH_SSD=$(geowatch_dvc --tags="phase2_data" --hardware="auto")
cd "$PHASE2_DATA_DPATH_SSD/Drop6"
python -m geowatch.cli.queue_cli.prepare_splits \
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
WATCH_GRID_WORKERS=4 python -m geowatch.tasks.fusion fit --config "
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
python -m geowatch.tasks.fusion fit \
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
WATCH_GRID_WORKERS=0 python -m geowatch.tasks.fusion fit --config "
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
WATCH_GRID_WORKERS=0 python -m geowatch.tasks.fusion fit --config "
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
WATCH_GRID_WORKERS=2 python -m geowatch.tasks.fusion fit --config "
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
WATCH_GRID_WORKERS=2 python -m geowatch.tasks.fusion fit --config "
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
WATCH_GRID_WORKERS=2 python -m geowatch.tasks.fusion fit --config "
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
WATCH_GRID_WORKERS=2 python -m geowatch.tasks.fusion fit --config "
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
WATCH_GRID_WORKERS=2 python -m geowatch.tasks.fusion fit --config "
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
WATCH_GRID_WORKERS=2 python -m geowatch.tasks.fusion fit --config "
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
WATCH_GRID_WORKERS=2 python -m geowatch.tasks.fusion fit --config "
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
geowatch model_stats "$DVC_EXPT_DPATH"/models/fusion/Drop6-MeanYear10GSD/packages/Drop6_TCombo1Year_BAS_10GSD_split6_V42_cont2/Drop6_TCombo1Year_BAS_10GSD_split6_V42_cont2_epoch3_step941.pt


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
WATCH_GRID_WORKERS=2 python -m geowatch.tasks.fusion fit --config "
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
WATCH_GRID_WORKERS=2 python -m geowatch.tasks.fusion fit --config "
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
WATCH_GRID_WORKERS=2 python -m geowatch.tasks.fusion fit --config "
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
geowatch model_stats "$DVC_EXPT_DPATH"/models/fusion/Drop6-MeanYear10GSD/packages/Drop6_TCombo1Year_BAS_10GSD_split6_V42_cont2/Drop6_TCombo1Year_BAS_10GSD_split6_V42_cont2_epoch3_step941.pt
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
WATCH_GRID_WORKERS=0 python -m geowatch.tasks.fusion fit --config "
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
WATCH_GRID_WORKERS=4 python -m geowatch.tasks.fusion fit --config "
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
WATCH_GRID_WORKERS=0 python -m geowatch.tasks.fusion fit --config "
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
WATCH_GRID_WORKERS=4 python -m geowatch.tasks.fusion fit --config "
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
WATCH_GRID_WORKERS=0 python -m geowatch.tasks.fusion fit --config "
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
WATCH_GRID_WORKERS=2 python -m geowatch.tasks.fusion fit --config "
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
WATCH_GRID_WORKERS=2 python -m geowatch.tasks.fusion fit --config "
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
WATCH_GRID_WORKERS=2 python -m geowatch.tasks.fusion fit --config "
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
WATCH_GRID_WORKERS=2 python -m geowatch.tasks.fusion fit --config "
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
WATCH_GRID_WORKERS=2 python -m geowatch.tasks.fusion fit --config "
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
WATCH_GRID_WORKERS=2 python -m geowatch.tasks.fusion fit --config "
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


export CUDA_VISIBLE_DEVICES=1
DVC_DATA_DPATH=$(geowatch_dvc --tags='phase2_data' --hardware='auto')
DVC_EXPT_DPATH=$(geowatch_dvc --tags='phase2_expt' --hardware='auto')
echo "DVC_EXPT_DPATH = $DVC_EXPT_DPATH"
WORKDIR=$DVC_EXPT_DPATH/training/$HOSTNAME/$USER
DATASET_CODE=Drop6-MeanYear10GSD-V2
KWCOCO_BUNDLE_DPATH=$DVC_DATA_DPATH/$DATASET_CODE
TRAIN_FPATH=$KWCOCO_BUNDLE_DPATH/data_train_I2LS_split6.kwcoco.zip
VALI_FPATH=$KWCOCO_BUNDLE_DPATH/data_vali_I2LS_split6.kwcoco.zip
CHANNELS="(L8,S2):(blue|green|red|nir),(WV):(blue|green|red),(WV,WV1):pan,(S2):(water|forest|field|impervious|barren|landcover_hidden.0:32),(L8,S2,WV,WV1):(sam.0:256)"
EXPERIMENT_NAME=Drop6_TCombo1Year_BAS_10GSD_V2_sam_landcover_split6_V53
DEFAULT_ROOT_DIR=$WORKDIR/$DATASET_CODE/runs/$EXPERIMENT_NAME
TARGET_LR=1e-5
WEIGHT_DECAY=$(python -c "print($TARGET_LR * 0.01)")
echo "WEIGHT_DECAY = $WEIGHT_DECAY"
MAX_STEPS=80000
WATCH_GRID_WORKERS=2 python -m geowatch.tasks.fusion fit --config "
data:
    select_videos          : $SELECT_VIDEOS
    num_workers            : 6
    train_dataset          : $TRAIN_FPATH
    vali_dataset           : $VALI_FPATH
    window_dims            : '164,164'
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
    temporal_dropout       : 0.1
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
    normalize_inputs       : 2048
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
        global_class_weight    : 0.0000001
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
    init: $DVC_EXPT_DPATH/models/fusion/Drop6-MeanYear10GSD-V2/packages/Drop6_TCombo1Year_BAS_10GSD_V2_landcover_split6_V47/Drop6_TCombo1Year_BAS_10GSD_V2_landcover_split6_V47_epoch47_step3026.pt
"


export CUDA_VISIBLE_DEVICES=1
DVC_DATA_DPATH=$(geowatch_dvc --tags='phase2_data' --hardware='auto')
DVC_EXPT_DPATH=$(geowatch_dvc --tags='phase2_expt' --hardware='auto')
echo "DVC_EXPT_DPATH = $DVC_EXPT_DPATH"
WORKDIR=$DVC_EXPT_DPATH/training/$HOSTNAME/$USER
DATASET_CODE=Drop6-MedianSummer10GSD
KWCOCO_BUNDLE_DPATH=$DVC_DATA_DPATH/$DATASET_CODE
TRAIN_FPATH=$KWCOCO_BUNDLE_DPATH/data_train_L_split6.kwcoco.zip
VALI_FPATH=$KWCOCO_BUNDLE_DPATH/data_vali_L_split6.kwcoco.zip
CHANNELS="(L8,S2):(blue|green|red|nir),(WV):(blue|green|red),(WV,WV1):pan,(S2):(water|forest|field|impervious|barren|landcover_hidden.0:32)"
EXPERIMENT_NAME=Drop6_TCombo1Year_BAS_10GSD_V2_landcover_split6_V54
DEFAULT_ROOT_DIR=$WORKDIR/$DATASET_CODE/runs/$EXPERIMENT_NAME
TARGET_LR=7e-6
WEIGHT_DECAY=$(python -c "print($TARGET_LR * 0.01)")
echo "WEIGHT_DECAY = $WEIGHT_DECAY"
MAX_STEPS=80000
WATCH_GRID_WORKERS=2 python -m geowatch.tasks.fusion fit --config "
data:
    select_videos          : $SELECT_VIDEOS
    num_workers            : 7
    train_dataset          : $TRAIN_FPATH
    vali_dataset           : $VALI_FPATH
    window_dims            : '160,160'
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
    temporal_dropout       : 0.1
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
    normalize_inputs       : 2048
    balance_areas          : True
model:
    class_path: MultimodalTransformer
    init_args:
        #saliency_weights       : '1:1'
        #class_weights          : auto
        tokenizer              : linconv
        arch_name              : smt_it_stm_p24
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
        global_class_weight    : 0.0000001
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
    accumulate_grad_batches: 72
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
    init: $DVC_EXPT_DPATH/models/fusion/Drop6-MeanYear10GSD-V2/packages/Drop6_TCombo1Year_BAS_10GSD_V2_landcover_split6_V47/Drop6_TCombo1Year_BAS_10GSD_V2_landcover_split6_V47_epoch47_step3026.pt
"

# Yardrat
export CUDA_VISIBLE_DEVICES=0
DVC_DATA_DPATH=$(geowatch_dvc --tags='phase2_data' --hardware='auto')
DVC_EXPT_DPATH=$(geowatch_dvc --tags='phase2_expt' --hardware='auto')
echo "DVC_EXPT_DPATH = $DVC_EXPT_DPATH"
WORKDIR=$DVC_EXPT_DPATH/training/$HOSTNAME/$USER
DATASET_CODE=Drop7-MedianSummer10GSD
KWCOCO_BUNDLE_DPATH=$DVC_DATA_DPATH/$DATASET_CODE
TRAIN_FPATH=$KWCOCO_BUNDLE_DPATH/data_train_I2L_split6.kwcoco.zip
VALI_FPATH=$KWCOCO_BUNDLE_DPATH/data_vali_I2L_split6.kwcoco.zip
CHANNELS="(L8,S2):(blue|green|red|nir),(WV):(blue|green|red),(WV,WV1):pan,(S2):(water|forest|field|impervious|barren|landcover_hidden.0:32,invariants:16)"
EXPERIMENT_NAME=Drop7-MedianSummer10GSD_BAS_10GSD_V2_invar_landcover_split6_V55
DEFAULT_ROOT_DIR=$WORKDIR/$DATASET_CODE/runs/$EXPERIMENT_NAME
TARGET_LR=7e-6
WEIGHT_DECAY=$(python -c "print($TARGET_LR * 0.01)")
echo "WEIGHT_DECAY = $WEIGHT_DECAY"
MAX_STEPS=80000
WATCH_GRID_WORKERS=2 python -m geowatch.tasks.fusion fit --config "
data:
    select_videos          : $SELECT_VIDEOS
    num_workers            : 7
    train_dataset          : $TRAIN_FPATH
    vali_dataset           : $VALI_FPATH
    window_dims            : '160,160'
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
    temporal_dropout       : 0.1
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
    normalize_inputs       : 2048
    balance_areas          : True
model:
    class_path: MultimodalTransformer
    init_args:
        #saliency_weights       : '1:1'
        #class_weights          : auto
        tokenizer              : linconv
        arch_name              : smt_it_stm_p24
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
        global_class_weight    : 0.0000001
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
    accumulate_grad_batches: 72
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
    init: $DVC_EXPT_DPATH/models/fusion/Drop6-MeanYear10GSD-V2/packages/Drop6_TCombo1Year_BAS_10GSD_V2_landcover_split6_V47/Drop6_TCombo1Year_BAS_10GSD_V2_landcover_split6_V47_epoch47_step3026.pt
"

DVC_EXPT_DPATH=$(geowatch_dvc --tags='phase2_expt' --hardware='auto')
rsync -avprPR yardrat:data/dvc-repos/smart_expt_dvc/./training/yardrat/jon.crall/Drop7-MedianSummer10GSD/runs/Drop7-MedianSummer10GSD_BAS_10GSD_V2_invar_landcover_split6_V55/lightning_logs/version_1 "$DVC_EXPT_DPATH"



export CUDA_VISIBLE_DEVICES=1
DVC_DATA_DPATH=$(geowatch_dvc --tags='phase2_data' --hardware='auto')
DVC_EXPT_DPATH=$(geowatch_dvc --tags='phase2_expt' --hardware='auto')
echo "DVC_EXPT_DPATH = $DVC_EXPT_DPATH"
WORKDIR=$DVC_EXPT_DPATH/training/$HOSTNAME/$USER
DATASET_CODE=Drop6-NoWinterMedian10GSD
KWCOCO_BUNDLE_DPATH=$DVC_DATA_DPATH/$DATASET_CODE
TRAIN_FPATH=$KWCOCO_BUNDLE_DPATH/data_train_I2L_split6.kwcoco.zip
VALI_FPATH=$KWCOCO_BUNDLE_DPATH/data_vali_I2L_split6.kwcoco.zip
CHANNELS="(L8,S2):(blue|green|red|nir),(WV):(blue|green|red),(WV,WV1):pan,(S2):(water|forest|field|impervious|barren|landcover_hidden.0:32,invariants:16)"
EXPERIMENT_NAME=Drop6_TCombo1Year_BAS_10GSD_V2_landcover_invar_split6_V56
DEFAULT_ROOT_DIR=$WORKDIR/$DATASET_CODE/runs/$EXPERIMENT_NAME
TARGET_LR=3e-6
WEIGHT_DECAY=$(python -c "print($TARGET_LR * 0.01)")
echo "WEIGHT_DECAY = $WEIGHT_DECAY"
MAX_STEPS=40000
WATCH_GRID_WORKERS=2 python -m geowatch.tasks.fusion fit --config "
data:
    select_videos          : $SELECT_VIDEOS
    num_workers            : 7
    train_dataset          : $TRAIN_FPATH
    vali_dataset           : $VALI_FPATH
    window_dims            : '160,160'
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
    max_epoch_length       : 10000
    channels               : '$CHANNELS'
    min_spacetime_weight   : 0.6
    temporal_dropout       : 0.1
    modality_dropout       : 0.2
    mask_low_quality       : False
    mask_samecolor_method  : None
    observable_threshold   : 0.0
    quality_threshold      : 0.0
    weight_dilate          : 10
    absolute_weighting     : True
    use_centered_positives : True
    use_grid_positives     : True
    use_grid_negatives     : 'cleared'
    normalize_inputs       : 2048
    balance_areas          : True
model:
    class_path: MultimodalTransformer
    init_args:
        #saliency_weights       : '1:1'
        #class_weights          : auto
        tokenizer              : linconv
        arch_name              : smt_it_stm_p24
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
        global_class_weight    : 0.0000001
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
    accumulate_grad_batches: 72
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
    init: $DVC_EXPT_DPATH/models/fusion/Drop6-MedianSummer10GSD/packages/Drop6_TCombo1Year_BAS_10GSD_V2_landcover_split6_V54/Drop6_TCombo1Year_BAS_10GSD_V2_landcover_split6_V54_epoch359_step10440.pt
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
CHANNELS="(L8,S2):(blue|green|red|nir),(WV):(blue|green|red),(WV,WV1):pan,(S2):(water|forest|field|impervious|barren|landcover_hidden.0:32,invariants:16),(L8,S2,WV,WV1):(sam.0:32)"
EXPERIMENT_NAME=Drop6_TCombo1Year_BAS_10GSD_V2_sam_landcover_split6_V57
DEFAULT_ROOT_DIR=$WORKDIR/$DATASET_CODE/runs/$EXPERIMENT_NAME
TARGET_LR=1e-6
WEIGHT_DECAY=$(python -c "print($TARGET_LR * 0.01)")
echo "WEIGHT_DECAY = $WEIGHT_DECAY"
MAX_STEPS=50000
WATCH_GRID_WORKERS=2 python -m geowatch.tasks.fusion fit --config "
data:
    select_videos          : $SELECT_VIDEOS
    num_workers            : 6
    train_dataset          : $TRAIN_FPATH
    vali_dataset           : $VALI_FPATH
    window_dims            : '196,196'
    time_steps             : 5
    time_sampling          : uniform-soft5-soft4-contiguous
    time_kernel            : '(-2y,-1y,0,1y,2y)'
    window_resolution     : 10.0GSD
    input_resolution      : 10.0GSD
    output_resolution     : 10.0GSD
    neg_to_pos_ratio       : 1.0
    batch_size             : 2
    normalize_perframe     : false
    normalize_peritem      : 'blue|green|red|nir|pan'
    max_epoch_length       : 10000
    channels               : '$CHANNELS'
    min_spacetime_weight   : 0.6
    temporal_dropout       : 0.1
    modality_dropout       : 0.4
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
        arch_name              : smt_it_stm_p24
        rescale_nans           : null
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
    three_phase: True
    anneal_strategy: cos
    pct_start: 0.05
trainer:
    accumulate_grad_batches: 64
    default_root_dir     : $DEFAULT_ROOT_DIR
    accelerator          : gpu
    devices              : 0,
    #devices              : 0,1
    #strategy             : ddp
    limit_val_batches    : 512
    limit_train_batches  : 4096
    num_sanity_val_steps : 0
    max_epochs           : 360

torch_globals:
    float32_matmul_precision: auto

initializer:
    init: $DVC_EXPT_DPATH/models/fusion/Drop6-MeanYear10GSD-V2/packages/Drop6_TCombo1Year_BAS_10GSD_V2_landcover_split6_V47/Drop6_TCombo1Year_BAS_10GSD_V2_landcover_split6_V47_epoch47_step3026.pt
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
CHANNELS="(L8,S2):(blue|green|red|nir),(WV):(blue|green|red),(WV,WV1):pan,(S2):(water|forest|field|impervious|barren|landcover_hidden.0:32,invariants:16)"
EXPERIMENT_NAME=Drop6_TCombo1Year_BAS_10GSD_V2_invariants_landcover_split6_V58
DEFAULT_ROOT_DIR=$WORKDIR/$DATASET_CODE/runs/$EXPERIMENT_NAME
TARGET_LR=1e-6
WEIGHT_DECAY=$(python -c "print($TARGET_LR * 0.01)")
echo "WEIGHT_DECAY = $WEIGHT_DECAY"
MAX_STEPS=50000
WATCH_GRID_WORKERS=2 python -m geowatch.tasks.fusion fit --config "
data:
    select_videos          : $SELECT_VIDEOS
    num_workers            : 6
    train_dataset          : $TRAIN_FPATH
    vali_dataset           : $VALI_FPATH
    window_dims            : '196,196'
    time_steps             : 5
    time_sampling          : uniform-soft5-soft4-contiguous
    time_kernel            : '(-2y,-1y,0,1y,2y)'
    window_resolution     : 10.0GSD
    input_resolution      : 10.0GSD
    output_resolution     : 10.0GSD
    neg_to_pos_ratio       : 1.0
    batch_size             : 3
    normalize_perframe     : false
    normalize_peritem      : 'blue|green|red|nir|pan'
    max_epoch_length       : 10000
    channels               : '$CHANNELS'
    min_spacetime_weight   : 0.6
    temporal_dropout       : 0.1
    modality_dropout       : 0.4
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
        arch_name              : smt_it_stm_p24
        rescale_nans           : null
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
    three_phase: True
    anneal_strategy: cos
    pct_start: 0.05
trainer:
    accumulate_grad_batches: 64
    default_root_dir     : $DEFAULT_ROOT_DIR
    accelerator          : gpu
    devices              : 0,
    #devices              : 0,1
    #strategy             : ddp
    limit_val_batches    : 512
    limit_train_batches  : 4096
    num_sanity_val_steps : 0
    max_epochs           : 360

torch_globals:
    float32_matmul_precision: auto

initializer:
    init: $DVC_EXPT_DPATH/models/fusion/Drop6-MeanYear10GSD-V2/packages/Drop6_TCombo1Year_BAS_10GSD_V2_landcover_split6_V47/Drop6_TCombo1Year_BAS_10GSD_V2_landcover_split6_V47_epoch47_step3026.pt
"


export CUDA_VISIBLE_DEVICES=0
DVC_DATA_DPATH=$(geowatch_dvc --tags='phase2_data' --hardware='auto')
DVC_EXPT_DPATH=$(geowatch_dvc --tags='phase2_expt' --hardware='auto')
echo "DVC_EXPT_DPATH = $DVC_EXPT_DPATH"
WORKDIR=$DVC_EXPT_DPATH/training/$HOSTNAME/$USER
DATASET_CODE=Drop6-MeanYear10GSD-V2
KWCOCO_BUNDLE_DPATH=$DVC_DATA_DPATH/$DATASET_CODE
TRAIN_FPATH=$KWCOCO_BUNDLE_DPATH/data_train_I2LS_split6.kwcoco.zip
VALI_FPATH=$KWCOCO_BUNDLE_DPATH/data_vali_I2LS_split6.kwcoco.zip
CHANNELS="(L8,S2):(blue|green|red|nir),(WV):(blue|green|red),(WV,WV1):pan,(S2):(water|forest|field|impervious|barren|landcover_hidden.0:32,invariants:16)"
EXPERIMENT_NAME=Drop6_TCombo1Year_BAS_10GSD_V2_invariants_landcover_split6_V59
DEFAULT_ROOT_DIR=$WORKDIR/$DATASET_CODE/runs/$EXPERIMENT_NAME
TARGET_LR=1e-5
WEIGHT_DECAY=$(python -c "print($TARGET_LR * 0.01)")
echo "WEIGHT_DECAY = $WEIGHT_DECAY"
MAX_STEPS=200000
WATCH_GRID_WORKERS=2 python -m geowatch.tasks.fusion fit --config "
data:
    select_videos          : $SELECT_VIDEOS
    num_workers            : 6
    train_dataset          : $TRAIN_FPATH
    vali_dataset           : $VALI_FPATH
    window_dims            : '128,128'
    time_steps             : 5
    time_sampling          : uniform-soft5-soft4-contiguous
    time_kernel            : '(-2y,-1y,0,1y,2y)'
    window_resolution     : 5.0GSD
    input_resolution      : 5.0GSD
    output_resolution     : 5.0GSD
    neg_to_pos_ratio       : 1.0
    batch_size             : 3
    normalize_perframe     : false
    normalize_peritem      : 'blue|green|red|nir|pan'
    max_epoch_length       : 10000
    channels               : '$CHANNELS'
    min_spacetime_weight   : 0.6
    temporal_dropout       : 0.1
    modality_dropout       : 0.4
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
        arch_name              : smt_it_stm_p24
        rescale_nans           : null
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
    three_phase: True
    anneal_strategy: cos
    pct_start: 0.05
trainer:
    accumulate_grad_batches: 12
    default_root_dir     : $DEFAULT_ROOT_DIR
    accelerator          : gpu
    devices              : 0,
    #devices              : 0,1
    #strategy             : ddp
    limit_val_batches    : 512
    limit_train_batches  : 4096
    num_sanity_val_steps : 0
    max_epochs           : 720

torch_globals:
    float32_matmul_precision: auto

initializer:
    init: $DVC_EXPT_DPATH/models/fusion/Drop6-MeanYear10GSD-V2/packages/Drop6_TCombo1Year_BAS_10GSD_V2_landcover_split6_V47/Drop6_TCombo1Year_BAS_10GSD_V2_landcover_split6_V47_epoch47_step3026.pt
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
CHANNELS="(L8,S2):(blue|green|red|nir),(WV):(blue|green|red),(WV,WV1):pan,(S2):(water|forest|field|impervious|barren|landcover_hidden.0:32,invariants:16)"
EXPERIMENT_NAME=Drop6_TCombo1Year_BAS_10GSD_V2_invariants_landcover_split6_V59
DEFAULT_ROOT_DIR=$WORKDIR/$DATASET_CODE/runs/$EXPERIMENT_NAME
TARGET_LR=1e-5
WEIGHT_DECAY=$(python -c "print($TARGET_LR * 0.01)")
echo "WEIGHT_DECAY = $WEIGHT_DECAY"
MAX_STEPS=200000
WATCH_GRID_WORKERS=2 python -m geowatch.tasks.fusion fit --config "
data:
    select_videos          : $SELECT_VIDEOS
    num_workers            : 6
    train_dataset          : $TRAIN_FPATH
    vali_dataset           : $VALI_FPATH
    window_dims            : '128,128'
    time_steps             : 7
    time_sampling          : uniform-soft5-soft4-contiguous
    time_kernel            : '(-3y,-2y,-1y,0,1y,2y,3y)'
    window_resolution     : 5.0GSD
    input_resolution      : 5.0GSD
    output_resolution     : 5.0GSD
    neg_to_pos_ratio       : 1.0
    batch_size             : 3
    normalize_perframe     : false
    normalize_peritem      : 'blue|green|red|nir|pan'
    max_epoch_length       : 10000
    channels               : '$CHANNELS'
    min_spacetime_weight   : 0.6
    temporal_dropout       : 0.1
    modality_dropout       : 0.4
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
        #saliency_weights      : '1:1'
        #class_weights         : auto
        tokenizer              : linconv
        arch_name              : smt_it_stm_p24
        rescale_nans           : null
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
        global_class_weight    : 1.00
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
    three_phase: True
    anneal_strategy: cos
    pct_start: 0.05
trainer:
    accumulate_grad_batches: 12
    default_root_dir     : $DEFAULT_ROOT_DIR
    accelerator          : gpu
    devices              : 0,
    #devices              : 0,1
    #strategy             : ddp
    limit_val_batches    : 512
    limit_train_batches  : 4096
    num_sanity_val_steps : 0
    max_epochs           : 720

torch_globals:
    float32_matmul_precision: auto

initializer:
    init: $DVC_EXPT_DPATH/models/fusion/Drop6-MeanYear10GSD-V2/packages/Drop6_TCombo1Year_BAS_10GSD_V2_landcover_split6_V47/Drop6_TCombo1Year_BAS_10GSD_V2_landcover_split6_V47_epoch47_step3026.pt
"






#/home/joncrall/remote/toothbrush/data/dvc-repos/smart_expt_dvc/training/toothbrush/joncrall/Drop7-MedianNoWinter10GSD/runs/Drop7-MedianNoWinter10GSD_landcover_invar_cold_split6_V60/lightning_logs/



# On toothbrush (split6 with COLD+Invariants+landcover)
export CUDA_VISIBLE_DEVICES=1
DVC_DATA_DPATH=$(geowatch_dvc --tags='phase2_data' --hardware='auto')
DVC_EXPT_DPATH=$(geowatch_dvc --tags='phase2_expt' --hardware='auto')
echo "DVC_EXPT_DPATH = $DVC_EXPT_DPATH"
WORKDIR=$DVC_EXPT_DPATH/training/$HOSTNAME/$USER
DATASET_CODE=Drop7-MedianNoWinter10GSD
KWCOCO_BUNDLE_DPATH=$DVC_DATA_DPATH/$DATASET_CODE
TRAIN_FPATH=$KWCOCO_BUNDLE_DPATH/data_train_I2LSC_split6.kwcoco.zip
VALI_FPATH=$KWCOCO_BUNDLE_DPATH/data_vali_I2LSC_split6.kwcoco.zip
CHANNELS="(L8,S2):(blue|green|red|nir),(WV):(blue|green|red),(WV,WV1):pan,(S2):(water|forest|field|impervious|barren|landcover_hidden.0:32,invariants:16),(L8):(blue_COLD_cv|green_COLD_cv|red_COLD_cv|nir_COLD_cv|swir16_COLD_cv|swir22_COLD_cv|blue_COLD_a0|green_COLD_a0|red_COLD_a0|nir_COLD_a0|swir16_COLD_a0|swir22_COLD_a0|blue_COLD_a1|green_COLD_a1|red_COLD_a1|nir_COLD_a1|swir16_COLD_a1|swir22_COLD_a1|blue_COLD_b1|green_COLD_b1|red_COLD_b1|nir_COLD_b1|swir16_COLD_b1|swir22_COLD_b1|blue_COLD_c1|green_COLD_c1|red_COLD_c1|nir_COLD_c1|swir16_COLD_c1|swir22_COLD_c1|blue_COLD_rmse|green_COLD_rmse|red_COLD_rmse|nir_COLD_rmse|swir16_COLD_rmse|swir22_COLD_rmse)"
EXPERIMENT_NAME=Drop7-MedianNoWinter10GSD_landcover_invar_cold_split6_V60
DEFAULT_ROOT_DIR=$WORKDIR/$DATASET_CODE/runs/$EXPERIMENT_NAME
TARGET_LR=3e-4
WEIGHT_DECAY=$(python -c "print($TARGET_LR * 0.01)")
echo "WEIGHT_DECAY = $WEIGHT_DECAY"
MAX_STEPS=80000
WATCH_GRID_WORKERS=0 python -m geowatch.tasks.fusion fit --config "
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
    accumulate_grad_batches: 48
    default_root_dir     : $DEFAULT_ROOT_DIR
    accelerator          : gpu
    devices              : 0,
    limit_val_batches    : 256
    limit_train_batches  : 2048
    num_sanity_val_steps : 0
    max_epochs           : 360

torch_globals:
    float32_matmul_precision: auto

initializer:
    init: $DVC_EXPT_DPATH/models/fusion/Drop6-MeanYear10GSD-V2/packages/Drop6_TCombo1Year_BAS_10GSD_V2_landcover_split6_V47/Drop6_TCombo1Year_BAS_10GSD_V2_landcover_split6_V47_epoch47_step3026.pt
"


# On namek - no teamfeat
export CUDA_VISIBLE_DEVICES=1
DVC_DATA_DPATH=$(geowatch_dvc --tags='phase2_data' --hardware='auto')
DVC_EXPT_DPATH=$(geowatch_dvc --tags='phase2_expt' --hardware='auto')
echo "DVC_EXPT_DPATH = $DVC_EXPT_DPATH"
WORKDIR=$DVC_EXPT_DPATH/training/$HOSTNAME/$USER
DATASET_CODE=Drop7-MedianNoWinter10GSD
KWCOCO_BUNDLE_DPATH=$DVC_DATA_DPATH/$DATASET_CODE
TRAIN_FPATH=$KWCOCO_BUNDLE_DPATH/data_train_I2LSC_split6.kwcoco.zip
VALI_FPATH=$KWCOCO_BUNDLE_DPATH/data_vali_I2LSC_split6.kwcoco.zip
CHANNELS="(L8,S2):(blue|green|red|nir),(WV):(blue|green|red),(WV,WV1):pan"
EXPERIMENT_NAME=Drop7-MedianNoWinter10GSD_bgr_split6_V61
DEFAULT_ROOT_DIR=$WORKDIR/$DATASET_CODE/runs/$EXPERIMENT_NAME
TARGET_LR=3e-4
WEIGHT_DECAY=$(python -c "print($TARGET_LR * 0.01)")
echo "WEIGHT_DECAY = $WEIGHT_DECAY"
MAX_STEPS=80000
WATCH_GRID_WORKERS=0 python -m geowatch.tasks.fusion fit --config "
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
    accumulate_grad_batches: 48
    default_root_dir     : $DEFAULT_ROOT_DIR
    accelerator          : gpu
    devices              : 0,
    limit_val_batches    : 256
    limit_train_batches  : 2048
    num_sanity_val_steps : 0
    max_epochs           : 360

torch_globals:
    float32_matmul_precision: auto

initializer:
    init: $DVC_EXPT_DPATH/models/fusion/Drop6-MeanYear10GSD-V2/packages/Drop6_TCombo1Year_BAS_10GSD_V2_landcover_split6_V47/Drop6_TCombo1Year_BAS_10GSD_V2_landcover_split6_V47_epoch47_step3026.pt
"


# On yardrat - COLD-only
export CUDA_VISIBLE_DEVICES=1
DVC_DATA_DPATH=$(geowatch_dvc --tags='phase2_data' --hardware='auto')
DVC_EXPT_DPATH=$(geowatch_dvc --tags='phase2_expt' --hardware='auto')
echo "DVC_EXPT_DPATH = $DVC_EXPT_DPATH"
WORKDIR=$DVC_EXPT_DPATH/training/$HOSTNAME/$USER
DATASET_CODE=Drop7-MedianNoWinter10GSD
KWCOCO_BUNDLE_DPATH=$DVC_DATA_DPATH/$DATASET_CODE
TRAIN_FPATH=$KWCOCO_BUNDLE_DPATH/data_train_I2LSC_split6.kwcoco.zip
VALI_FPATH=$KWCOCO_BUNDLE_DPATH/data_vali_I2LSC_split6.kwcoco.zip
CHANNELS="(L8,S2):(blue|green|red|nir),(WV):(blue|green|red),(WV,WV1):pan,(L8):(blue_COLD_cv|green_COLD_cv|red_COLD_cv|nir_COLD_cv|swir16_COLD_cv|swir22_COLD_cv|blue_COLD_a0|green_COLD_a0|red_COLD_a0|nir_COLD_a0|swir16_COLD_a0|swir22_COLD_a0|blue_COLD_a1|green_COLD_a1|red_COLD_a1|nir_COLD_a1|swir16_COLD_a1|swir22_COLD_a1|blue_COLD_b1|green_COLD_b1|red_COLD_b1|nir_COLD_b1|swir16_COLD_b1|swir22_COLD_b1|blue_COLD_c1|green_COLD_c1|red_COLD_c1|nir_COLD_c1|swir16_COLD_c1|swir22_COLD_c1|blue_COLD_rmse|green_COLD_rmse|red_COLD_rmse|nir_COLD_rmse|swir16_COLD_rmse|swir22_COLD_rmse)"
EXPERIMENT_NAME=Drop7-MedianNoWinter10GSD_bgr_cold_split6_V62
DEFAULT_ROOT_DIR=$WORKDIR/$DATASET_CODE/runs/$EXPERIMENT_NAME
TARGET_LR=3e-4
WEIGHT_DECAY=$(python -c "print($TARGET_LR * 0.01)")
echo "WEIGHT_DECAY = $WEIGHT_DECAY"
MAX_STEPS=80000
WATCH_GRID_WORKERS=0 python -m geowatch.tasks.fusion fit --config "
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
    accumulate_grad_batches: 48
    default_root_dir     : $DEFAULT_ROOT_DIR
    accelerator          : gpu
    devices              : 0,
    limit_val_batches    : 256
    limit_train_batches  : 2048
    num_sanity_val_steps : 0
    max_epochs           : 360

torch_globals:
    float32_matmul_precision: auto

initializer:
    init: $DVC_EXPT_DPATH/models/fusion/Drop6-MeanYear10GSD-V2/packages/Drop6_TCombo1Year_BAS_10GSD_V2_landcover_split6_V47/Drop6_TCombo1Year_BAS_10GSD_V2_landcover_split6_V47_epoch47_step3026.pt
"


# On ooo - no teamfeat
export CUDA_VISIBLE_DEVICES=1
DVC_DATA_DPATH=$(geowatch_dvc --tags='phase2_data' --hardware='auto')
DVC_EXPT_DPATH=$(geowatch_dvc --tags='phase2_expt' --hardware='auto')
echo "DVC_EXPT_DPATH = $DVC_EXPT_DPATH"
WORKDIR=$DVC_EXPT_DPATH/training/$HOSTNAME/$USER
DATASET_CODE=Drop7-MedianNoWinter10GSD
KWCOCO_BUNDLE_DPATH=$DVC_DATA_DPATH/$DATASET_CODE
TRAIN_FPATH=$KWCOCO_BUNDLE_DPATH/data_train_I2LSC_split6.kwcoco.zip
VALI_FPATH=$KWCOCO_BUNDLE_DPATH/data_vali_I2LSC_split6.kwcoco.zip
CHANNELS="(L8,S2):(blue|green|red|nir),(WV):(blue|green|red),(WV,WV1):pan"
EXPERIMENT_NAME=Drop7-MedianNoWinter10GSD_bgr_split6_V63
DEFAULT_ROOT_DIR=$WORKDIR/$DATASET_CODE/runs/$EXPERIMENT_NAME
TARGET_LR=3e-4
WEIGHT_DECAY=$(python -c "print($TARGET_LR * 0.01)")
echo "WEIGHT_DECAY = $WEIGHT_DECAY"
MAX_STEPS=80000
WATCH_GRID_WORKERS=0 python -m geowatch.tasks.fusion fit --config "
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
    accumulate_grad_batches: 48
    default_root_dir     : $DEFAULT_ROOT_DIR
    accelerator          : gpu
    devices              : 0,
    limit_val_batches    : 256
    limit_train_batches  : 2048
    num_sanity_val_steps : 0
    max_epochs           : 360

torch_globals:
    float32_matmul_precision: auto

initializer:
    init: $DVC_EXPT_DPATH/models/fusion/Drop6-MeanYear10GSD-V2/packages/Drop6_TCombo1Year_BAS_10GSD_V2_landcover_split6_V47/Drop6_TCombo1Year_BAS_10GSD_V2_landcover_split6_V47_epoch47_step3026.pt
"




# On toothbrush scratch (split6 with COLD+Invariants+landcover)
export CUDA_VISIBLE_DEVICES=1
DVC_DATA_DPATH=$(geowatch_dvc --tags='phase2_data' --hardware='auto')
DVC_EXPT_DPATH=$(geowatch_dvc --tags='phase2_expt' --hardware='auto')
echo "DVC_EXPT_DPATH = $DVC_EXPT_DPATH"
WORKDIR=$DVC_EXPT_DPATH/training/$HOSTNAME/$USER
DATASET_CODE=Drop7-MedianNoWinter10GSD
KWCOCO_BUNDLE_DPATH=$DVC_DATA_DPATH/$DATASET_CODE
TRAIN_FPATH=$KWCOCO_BUNDLE_DPATH/data_train_I2LSC_split6.kwcoco.zip
VALI_FPATH=$KWCOCO_BUNDLE_DPATH/data_vali_I2LSC_split6.kwcoco.zip
CHANNELS="(L8,S2):(blue|green|red|nir),(WV):(blue|green|red),(WV,WV1):pan,(S2):(water|forest|field|impervious|barren|landcover_hidden.0:32,invariants:16),(L8):(blue_COLD_cv|green_COLD_cv|red_COLD_cv|nir_COLD_cv|swir16_COLD_cv|swir22_COLD_cv|blue_COLD_a0|green_COLD_a0|red_COLD_a0|nir_COLD_a0|swir16_COLD_a0|swir22_COLD_a0|blue_COLD_a1|green_COLD_a1|red_COLD_a1|nir_COLD_a1|swir16_COLD_a1|swir22_COLD_a1|blue_COLD_b1|green_COLD_b1|red_COLD_b1|nir_COLD_b1|swir16_COLD_b1|swir22_COLD_b1|blue_COLD_c1|green_COLD_c1|red_COLD_c1|nir_COLD_c1|swir16_COLD_c1|swir22_COLD_c1|blue_COLD_rmse|green_COLD_rmse|red_COLD_rmse|nir_COLD_rmse|swir16_COLD_rmse|swir22_COLD_rmse),(L8,S2,WV,WV1):(sam.0:64)"
EXPERIMENT_NAME=Drop7-MedianNoWinter10GSD_landcover_invar_cold_sam_scratch_split6_V60
DEFAULT_ROOT_DIR=$WORKDIR/$DATASET_CODE/runs/$EXPERIMENT_NAME
TARGET_LR=3e-4
WEIGHT_DECAY=$(python -c "print($TARGET_LR * 0.01)")
echo "WEIGHT_DECAY = $WEIGHT_DECAY"
MAX_STEPS=80000
WATCH_GRID_WORKERS=0 python -m geowatch.tasks.fusion fit --config "
data:
    select_videos          : $SELECT_VIDEOS
    num_workers            : 5
    train_dataset          : $TRAIN_FPATH
    vali_dataset           : $VALI_FPATH
    window_dims            : '164,164'
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
    accumulate_grad_batches: 48
    default_root_dir     : $DEFAULT_ROOT_DIR
    accelerator          : gpu
    devices              : 0,
    limit_val_batches    : 256
    limit_train_batches  : 2048
    num_sanity_val_steps : 0
    max_epochs           : 360
    # TODO:
    #max_steps            : $MAX_STEPS

torch_globals:
    float32_matmul_precision: auto

initializer:
    init: noop
" --ckpt_path /home/joncrall/remote/toothbrush/data/dvc-repos/smart_expt_dvc/training/toothbrush/joncrall/Drop7-MedianNoWinter10GSD/runs/Drop7-MedianNoWinter10GSD_landcover_invar_cold_sam_scratch_split6_V60/lightning_logs/version_3/checkpoints/epoch=194-step=8385.ckpt


# On toothbrush scratch (split6 with COLD+Invariants+landcover+materials+mae)
export CUDA_VISIBLE_DEVICES=1
DVC_DATA_DPATH=$(geowatch_dvc --tags='phase2_data' --hardware='auto')
DVC_EXPT_DPATH=$(geowatch_dvc --tags='phase2_expt' --hardware='auto')
echo "DVC_EXPT_DPATH = $DVC_EXPT_DPATH"
WORKDIR=$DVC_EXPT_DPATH/training/$HOSTNAME/$USER
DATASET_CODE=Drop7-MedianNoWinter10GSD
KWCOCO_BUNDLE_DPATH=$DVC_DATA_DPATH/$DATASET_CODE
TRAIN_FPATH=$KWCOCO_BUNDLE_DPATH/data_train_EI2LMSC_split6.kwcoco.zip
VALI_FPATH=$KWCOCO_BUNDLE_DPATH/data_vali_EI2LMSC_split6.kwcoco.zip
CHANNELS="(L8,S2):(blue|green|red|nir),(WV):(blue|green|red),(WV,WV1):pan,(S2):(water|forest|field|impervious|barren|landcover_hidden.0:32,invariants:16),(L8):(blue_COLD_cv|green_COLD_cv|red_COLD_cv|nir_COLD_cv|swir16_COLD_cv|swir22_COLD_cv|blue_COLD_a0|green_COLD_a0|red_COLD_a0|nir_COLD_a0|swir16_COLD_a0|swir22_COLD_a0|blue_COLD_a1|green_COLD_a1|red_COLD_a1|nir_COLD_a1|swir16_COLD_a1|swir22_COLD_a1|blue_COLD_b1|green_COLD_b1|red_COLD_b1|nir_COLD_b1|swir16_COLD_b1|swir22_COLD_b1|blue_COLD_c1|green_COLD_c1|red_COLD_c1|nir_COLD_c1|swir16_COLD_c1|swir22_COLD_c1|blue_COLD_rmse|green_COLD_rmse|red_COLD_rmse|nir_COLD_rmse|swir16_COLD_rmse|swir22_COLD_rmse),(L8,S2,WV,WV1):(sam.0:64),(L8,S2,WV):(mat_feats.0:16|materials.0:9|mtm),(S2):(mae.0:16)"
EXPERIMENT_NAME=Drop7-MedianNoWinter10GSD_landcover_invar_cold_sam_mat_mae_scratch_split6_V64
DEFAULT_ROOT_DIR=$WORKDIR/$DATASET_CODE/runs/$EXPERIMENT_NAME
TARGET_LR=1e-4
WEIGHT_DECAY=$(python -c "print($TARGET_LR * 0.01)")
echo "WEIGHT_DECAY = $WEIGHT_DECAY"
MAX_STEPS=80000
WATCH_GRID_WORKERS=0 python -m geowatch.tasks.fusion fit --config "
data:
    select_videos          : $SELECT_VIDEOS
    num_workers            : 5
    train_dataset          : $TRAIN_FPATH
    vali_dataset           : $VALI_FPATH
    window_dims            : '164,164'
    time_steps             : 11
    time_sampling          : uniform-soft5-soft4-contiguous
    time_kernel            : '(-3y,-2.5y,-2y,-1.5y,-1y,0,1y,1.5y,2y,2.5y,3y)'
    window_resolution     : 10.0GSD
    input_resolution      : 10.0GSD
    output_resolution     : 10.0GSD
    neg_to_pos_ratio       : 1.0
    batch_size             : 1
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
    accumulate_grad_batches: 24
    default_root_dir     : $DEFAULT_ROOT_DIR
    accelerator          : gpu
    devices              : 0,
    limit_val_batches    : 256
    limit_train_batches  : 2048
    num_sanity_val_steps : 0
    max_steps            : $MAX_STEPS

    callbacks:
        - class_path: pytorch_lightning.callbacks.ModelCheckpoint
          init_args:
              monitor: val_loss
              mode: min
              save_top_k: 5
              filename: '{epoch}-{val_loss:.3f}.ckpt'
              save_last: true

torch_globals:
    float32_matmul_precision: auto

initializer:
    init: noop
" --ckpt_path /home/joncrall/remote/toothbrush/data/dvc-repos/smart_expt_dvc/training/toothbrush/joncrall/Drop7-MedianNoWinter10GSD/runs/Drop7-MedianNoWinter10GSD_landcover_invar_cold_sam_mat_mae_scratch_split6_V64/lightning_logs/version_1/checkpoints/epoch=1-step=172-v1.ckpt



# On horologic scratch (connor's channels)
export CUDA_VISIBLE_DEVICES=3
DVC_DATA_DPATH=$(geowatch_dvc --tags='phase2_data' --hardware='auto')
DVC_EXPT_DPATH=$(geowatch_dvc --tags='phase2_expt' --hardware='auto')
echo "DVC_EXPT_DPATH = $DVC_EXPT_DPATH"
WORKDIR=$DVC_EXPT_DPATH/training/$HOSTNAME/$USER
DATASET_CODE=Drop7-MedianNoWinter10GSD
KWCOCO_BUNDLE_DPATH=$DVC_DATA_DPATH/$DATASET_CODE
TRAIN_FPATH=$KWCOCO_BUNDLE_DPATH/data_train_EI2LMSC_split6.kwcoco.zip
VALI_FPATH=$KWCOCO_BUNDLE_DPATH/data_vali_EI2LMSC_split6.kwcoco.zip
CHANNELS="(L8,S2):blue|green|red|nir,(WV,WV1):pan,S2:(invariants:16,water|forest|field|impervious|barren|landcover_hidden:32),WV:blue|green|red"
EXPERIMENT_NAME=Drop7-MedianNoWinter10GSD_landcover_invar_scratch_split6_V65
DEFAULT_ROOT_DIR=$WORKDIR/$DATASET_CODE/runs/$EXPERIMENT_NAME
TARGET_LR=3e-4
WEIGHT_DECAY=$(python -c "print($TARGET_LR * 0.01)")
echo "WEIGHT_DECAY = $WEIGHT_DECAY"
MAX_STEPS=80000
WATCH_GRID_WORKERS=0 python -m geowatch.tasks.fusion fit --config "
data:
    select_videos          : $SELECT_VIDEOS
    num_workers            : 5
    train_dataset          : $TRAIN_FPATH
    vali_dataset           : $VALI_FPATH
    window_dims            : '164,164'
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
    accumulate_grad_batches: 48
    default_root_dir     : $DEFAULT_ROOT_DIR
    accelerator          : gpu
    devices              : 0,
    limit_val_batches    : 256
    limit_train_batches  : 2048
    num_sanity_val_steps : 0
    max_epochs           : 360
    callbacks:
        - class_path: pytorch_lightning.callbacks.ModelCheckpoint
          init_args:
              monitor: val_loss
              mode: min
              save_top_k: 5
              filename: '{epoch}-{val_loss:.3f}.ckpt'
              save_last: true

torch_globals:
    float32_matmul_precision: auto

initializer:
    init: noop
" --ckpt_path "$HOME"/remote/horologic/data/dvc-repos/smart_expt_dvc/training/horologic/jon.crall/Drop7-MedianNoWinter10GSD/runs/Drop7-MedianNoWinter10GSD_landcover_invar_scratch_split6_V65/lightning_logs/version_0/checkpoints/epoch=6-val_loss=2.360.ckpt.ckpt


# On Ooo scratch no team features
export CUDA_VISIBLE_DEVICES=1
DVC_DATA_DPATH=$(geowatch_dvc --tags='phase2_data' --hardware='auto')
DVC_EXPT_DPATH=$(geowatch_dvc --tags='phase2_expt' --hardware='auto')
echo "DVC_EXPT_DPATH = $DVC_EXPT_DPATH"
WORKDIR=$DVC_EXPT_DPATH/training/$HOSTNAME/$USER
DATASET_CODE=Drop7-MedianNoWinter10GSD
KWCOCO_BUNDLE_DPATH=$DVC_DATA_DPATH/$DATASET_CODE
TRAIN_FPATH=$KWCOCO_BUNDLE_DPATH/data_train_EI2LMSC_split6.kwcoco.zip
VALI_FPATH=$KWCOCO_BUNDLE_DPATH/data_vali_EI2LMSC_split6.kwcoco.zip
CHANNELS="(L8,S2):blue|green|red|nir,(WV,WV1):pan,WV:blue|green|red"
EXPERIMENT_NAME=Drop7-MedianNoWinter10GSD_bgrn_scratch_split6_V66
DEFAULT_ROOT_DIR=$WORKDIR/$DATASET_CODE/runs/$EXPERIMENT_NAME
TARGET_LR=3e-4
WEIGHT_DECAY=$(python -c "print($TARGET_LR * 0.01)")
echo "WEIGHT_DECAY = $WEIGHT_DECAY"
MAX_STEPS=80000
WATCH_GRID_WORKERS=0 python -m geowatch.tasks.fusion fit --config "
data:
    select_videos          : $SELECT_VIDEOS
    num_workers            : 5
    train_dataset          : $TRAIN_FPATH
    vali_dataset           : $VALI_FPATH
    window_dims            : '164,164'
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
    accumulate_grad_batches: 48
    default_root_dir     : $DEFAULT_ROOT_DIR
    accelerator          : gpu
    devices              : 0,
    limit_val_batches    : 256
    limit_train_batches  : 2048
    num_sanity_val_steps : 0
    max_epochs           : 360
    callbacks:
        - class_path: pytorch_lightning.callbacks.ModelCheckpoint
          init_args:
              monitor: val_loss
              mode: min
              save_top_k: 5
              filename: '{epoch}-{val_loss:.3f}.ckpt'
              save_last: true

torch_globals:
    float32_matmul_precision: auto

initializer:
    init: noop
"


# On Ooo scratch no team features (lower LR)
export CUDA_VISIBLE_DEVICES=0
DVC_DATA_DPATH=$(geowatch_dvc --tags='phase2_data' --hardware='auto')
DVC_EXPT_DPATH=$(geowatch_dvc --tags='phase2_expt' --hardware='auto')
echo "DVC_EXPT_DPATH = $DVC_EXPT_DPATH"
WORKDIR=$DVC_EXPT_DPATH/training/$HOSTNAME/$USER
DATASET_CODE=Drop7-MedianNoWinter10GSD
KWCOCO_BUNDLE_DPATH=$DVC_DATA_DPATH/$DATASET_CODE
TRAIN_FPATH=$KWCOCO_BUNDLE_DPATH/data_train_EI2LMSC_split6.kwcoco.zip
VALI_FPATH=$KWCOCO_BUNDLE_DPATH/data_vali_EI2LMSC_split6.kwcoco.zip
CHANNELS="(L8,S2):blue|green|red|nir,(WV,WV1):pan,WV:blue|green|red"
EXPERIMENT_NAME=Drop7-MedianNoWinter10GSD_bgrn_scratch_split6_V67
DEFAULT_ROOT_DIR=$WORKDIR/$DATASET_CODE/runs/$EXPERIMENT_NAME
TARGET_LR=1e-4
WEIGHT_DECAY=$(python -c "print($TARGET_LR * 0.01)")
echo "WEIGHT_DECAY = $WEIGHT_DECAY"
MAX_STEPS=80000
WATCH_GRID_WORKERS=0 python -m geowatch.tasks.fusion fit --config "
data:
    select_videos          : $SELECT_VIDEOS
    num_workers            : 5
    train_dataset          : $TRAIN_FPATH
    vali_dataset           : $VALI_FPATH
    window_dims            : '164,164'
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
    accumulate_grad_batches: 48
    default_root_dir     : $DEFAULT_ROOT_DIR
    accelerator          : gpu
    devices              : 0,
    limit_val_batches    : 256
    limit_train_batches  : 2048
    num_sanity_val_steps : 0
    max_epochs           : 360
    callbacks:
        - class_path: pytorch_lightning.callbacks.ModelCheckpoint
          init_args:
              monitor: val_loss
              mode: min
              save_top_k: 5
              filename: '{epoch}-{val_loss:.3f}.ckpt'
              save_last: true

torch_globals:
    float32_matmul_precision: auto

initializer:
    init: noop
"

# On toothbrush - no teamfeat / pan (lower LR)
export CUDA_VISIBLE_DEVICES=0
DVC_DATA_DPATH=$(geowatch_dvc --tags='phase2_data' --hardware='auto')
DVC_EXPT_DPATH=$(geowatch_dvc --tags='phase2_expt' --hardware='auto')
echo "DVC_EXPT_DPATH = $DVC_EXPT_DPATH"
WORKDIR=$DVC_EXPT_DPATH/training/$HOSTNAME/$USER
DATASET_CODE=Drop7-MedianNoWinter10GSD
KWCOCO_BUNDLE_DPATH=$DVC_DATA_DPATH/$DATASET_CODE
TRAIN_FPATH=$KWCOCO_BUNDLE_DPATH/data_train_EI2LMSC_split6.kwcoco.zip
VALI_FPATH=$KWCOCO_BUNDLE_DPATH/data_vali_EI2LMSC_split6.kwcoco.zip
CHANNELS="(L8,S2):(blue|green|red|nir),(WV):(blue|green|red)"
EXPERIMENT_NAME=Drop7-MedianNoWinter10GSD_bgrn_split6_V68
DEFAULT_ROOT_DIR=$WORKDIR/$DATASET_CODE/runs/$EXPERIMENT_NAME
TARGET_LR=1e-5
WEIGHT_DECAY=$(python -c "print($TARGET_LR * 0.01)")
echo "WEIGHT_DECAY = $WEIGHT_DECAY"
MAX_STEPS=80000
WATCH_GRID_WORKERS=0 python -m geowatch.tasks.fusion fit --config "
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
        #saliency_weights      : '1:1'
        #class_weights         : auto
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
    accumulate_grad_batches: 24
    default_root_dir     : $DEFAULT_ROOT_DIR
    accelerator          : gpu
    devices              : 0,
    limit_val_batches    : 256
    limit_train_batches  : 2048
    num_sanity_val_steps : 0
    max_epochs           : 360
    callbacks:
        - class_path: pytorch_lightning.callbacks.ModelCheckpoint
          init_args:
              monitor: val_loss
              mode: min
              save_top_k: 5
              filename: '{epoch}-{val_loss:.3f}.ckpt'
              save_last: true

torch_globals:
    float32_matmul_precision: auto

initializer:
    init: $DVC_EXPT_DPATH/models/fusion/Drop6-MeanYear10GSD-V2/packages/Drop6_TCombo1Year_BAS_10GSD_V2_landcover_split6_V47/Drop6_TCombo1Year_BAS_10GSD_V2_landcover_split6_V47_epoch47_step3026.pt
"



# On toothbrush - SC training
export CUDA_VISIBLE_DEVICES=0
DVC_DATA_DPATH=$(geowatch_dvc --tags='drop7_data' --hardware='auto')
DVC_EXPT_DPATH=$(geowatch_dvc --tags='phase2_expt' --hardware='auto')
echo "DVC_EXPT_DPATH = $DVC_EXPT_DPATH"
WORKDIR=$DVC_EXPT_DPATH/training/$HOSTNAME/$USER
DATASET_CODE=Drop7-Cropped2GSD
KWCOCO_BUNDLE_DPATH=$DVC_DATA_DPATH/$DATASET_CODE
TRAIN_FPATH=$KWCOCO_BUNDLE_DPATH/data_train_rawbands_split6.kwcoco.zip
VALI_FPATH=$KWCOCO_BUNDLE_DPATH/data_vali_rawbands_split6.kwcoco.zip
CHANNELS="(L8,S2):(blue|green|red|nir),(WV):(blue|green|red),(WV1):(pan)"
EXPERIMENT_NAME=Drop7-Cropped2GSD_SC_bgrn_split6_V05
DEFAULT_ROOT_DIR=$WORKDIR/$DATASET_CODE/runs/$EXPERIMENT_NAME
TARGET_LR=1e-4
WEIGHT_DECAY=$(python -c "print($TARGET_LR * 0.01)")
echo "WEIGHT_DECAY = $WEIGHT_DECAY"
MAX_STEPS=80000
WATCH_GRID_WORKERS=0 python -m geowatch.tasks.fusion fit --config "
data:
    select_videos          : $SELECT_VIDEOS
    num_workers            : 5
    train_dataset          : $TRAIN_FPATH
    vali_dataset           : $VALI_FPATH
    window_dims            : '256,256'
    time_steps             : 9
    time_sampling          : uniform-soft5-soft4-contiguous
    time_kernel            : '(-1.25y,-1.08y,-0.25y,-0.08y,0.0y,0.08y,0.25y,1.08y,1.25y)'
    window_resolution     : 2.0GSD
    input_resolution      : 2.0GSD
    output_resolution     : 2.0GSD
    neg_to_pos_ratio       : 1.0
    batch_size             : 4
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
    use_grid_positives     : False
    use_grid_negatives     : False
    normalize_inputs       : 1024
    balance_areas          : True
model:
    class_path: MultimodalTransformer
    init_args:
        #saliency_weights      : '1:1'
        #class_weights         : auto
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
        global_class_weight    : 1.00
        global_saliency_weight : 0.01
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
    accumulate_grad_batches: 24
    default_root_dir     : $DEFAULT_ROOT_DIR
    accelerator          : gpu
    devices              : 0,
    limit_val_batches    : 256
    limit_train_batches  : 2048
    num_sanity_val_steps : 0
    max_epochs           : 360
    callbacks:
        - class_path: pytorch_lightning.callbacks.ModelCheckpoint
          init_args:
              monitor: val_loss
              mode: min
              save_top_k: 5
              filename: '{epoch}-{step}-{val_loss:.3f}.ckpt'
              save_last: true

torch_globals:
    float32_matmul_precision: auto

initializer:
    init: $DVC_EXPT_DPATH/models/fusion/Drop6-MeanYear10GSD-V2/packages/Drop6_TCombo1Year_BAS_10GSD_V2_landcover_split6_V47/Drop6_TCombo1Year_BAS_10GSD_V2_landcover_split6_V47_epoch47_step3026.pt
"


# On toothbrush - SC training - test behavior?
export CUDA_VISIBLE_DEVICES=0
DVC_DATA_DPATH=$(geowatch_dvc --tags='drop7_data' --hardware='auto')
DVC_EXPT_DPATH=$(geowatch_dvc --tags='phase2_expt' --hardware='auto')
echo "DVC_EXPT_DPATH = $DVC_EXPT_DPATH"
WORKDIR=$DVC_EXPT_DPATH/training/$HOSTNAME/$USER
DATASET_CODE=Drop7-Cropped2GSD
KWCOCO_BUNDLE_DPATH=$DVC_DATA_DPATH/$DATASET_CODE
TRAIN_FPATH=$KWCOCO_BUNDLE_DPATH/PE_R001/imgannots-PE_R001.kwcoco.zip
VALI_FPATH=$KWCOCO_BUNDLE_DPATH/data_vali_rawbands_split6.kwcoco.zip
CHANNELS="(L8,S2):(blue|green|red|nir),(WV):(blue|green|red),(WV1):(pan)"
EXPERIMENT_NAME=Drop7-Cropped2GSD_SC_bgrn_petest_V06
DEFAULT_ROOT_DIR=$WORKDIR/$DATASET_CODE/runs/$EXPERIMENT_NAME
TARGET_LR=1e-4
WEIGHT_DECAY=$(python -c "print($TARGET_LR * 0.01)")
echo "WEIGHT_DECAY = $WEIGHT_DECAY"
MAX_STEPS=80000
WATCH_GRID_WORKERS=0 python -m geowatch.tasks.fusion fit --config "
data:
    select_videos          : $SELECT_VIDEOS
    num_workers            : 5
    train_dataset          : $TRAIN_FPATH
    vali_dataset           : $VALI_FPATH
    window_dims            : '196,196'
    time_steps             : 9
    time_sampling          : uniform-soft5-soft4-contiguous
    time_kernel            : '(-1.25y,-1.08y,-0.25y,-0.08y,0.0y,0.08y,0.25y,1.08y,1.25y)'
    window_resolution     : 2.0GSD
    input_resolution      : 2.0GSD
    output_resolution     : 2.0GSD
    neg_to_pos_ratio       : 1.0
    batch_size             : 4
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
    use_grid_positives     : False
    use_grid_negatives     : False
    normalize_inputs       : 1024
    balance_areas          : True
model:
    class_path: MultimodalTransformer
    init_args:
        #saliency_weights      : '1:1'
        #class_weights         : auto
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
        global_class_weight    : 1.00
        global_saliency_weight : 0.01
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
    accumulate_grad_batches: 24
    default_root_dir     : $DEFAULT_ROOT_DIR
    accelerator          : gpu
    devices              : 0,
    limit_val_batches    : 256
    limit_train_batches  : 2048
    num_sanity_val_steps : 0
    max_epochs           : 360
    callbacks:
        - class_path: pytorch_lightning.callbacks.ModelCheckpoint
          init_args:
              monitor: val_loss
              mode: min
              save_top_k: 5
              filename: '{epoch}-{step}-{val_loss:.3f}.ckpt'
              save_last: true

torch_globals:
    float32_matmul_precision: auto

initializer:
    init: $DVC_EXPT_DPATH/models/fusion/Drop6-MeanYear10GSD-V2/packages/Drop6_TCombo1Year_BAS_10GSD_V2_landcover_split6_V47/Drop6_TCombo1Year_BAS_10GSD_V2_landcover_split6_V47_epoch47_step3026.pt
"



# On toothbrush - SC training - test behavior?
export CUDA_VISIBLE_DEVICES=1
DVC_DATA_DPATH=$(geowatch_dvc --tags='drop7_data' --hardware='auto')
DVC_EXPT_DPATH=$(geowatch_dvc --tags='phase2_expt' --hardware='auto')
echo "DVC_EXPT_DPATH = $DVC_EXPT_DPATH"
WORKDIR=$DVC_EXPT_DPATH/training/$HOSTNAME/$USER
DATASET_CODE=Drop7-Cropped2GSD
KWCOCO_BUNDLE_DPATH=$DVC_DATA_DPATH/$DATASET_CODE
TRAIN_FPATH=$KWCOCO_BUNDLE_DPATH/data_train_rawbands_split6.kwcoco.zip
VALI_FPATH=$KWCOCO_BUNDLE_DPATH/data_vali_rawbands_split6.kwcoco.zip
CHANNELS="(L8,S2):(blue|green|red|nir),(WV):(blue|green|red)"
EXPERIMENT_NAME=Drop7-Cropped2GSD_SC_bgrn_split6_V07
DEFAULT_ROOT_DIR=$WORKDIR/$DATASET_CODE/runs/$EXPERIMENT_NAME
TARGET_LR=1e-4
WEIGHT_DECAY=$(python -c "print($TARGET_LR * 0.01)")
echo "WEIGHT_DECAY = $WEIGHT_DECAY"
MAX_STEPS=80000
WATCH_GRID_WORKERS=0 python -m geowatch.tasks.fusion fit --config "
data:
    select_videos          : $SELECT_VIDEOS
    num_workers            : 5
    train_dataset          : $TRAIN_FPATH
    vali_dataset           : $VALI_FPATH
    window_dims            : '196,196'
    time_steps             : 9
    time_sampling          : uniform-soft5-soft4-contiguous
    time_kernel            : '(-1.25y,-1.08y,-0.25y,-0.08y,0.0y,0.08y,0.25y,1.08y,1.25y)'
    window_resolution     : 2.0GSD
    input_resolution      : 2.0GSD
    output_resolution     : 2.0GSD
    neg_to_pos_ratio       : 1.0
    batch_size             : 4
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
    use_grid_positives     : False
    use_grid_negatives     : False
    normalize_inputs       : 1024
    balance_areas          : True
model:
    class_path: MultimodalTransformer
    init_args:
        #saliency_weights      : '1:1'
        #class_weights         : auto
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
        global_class_weight    : 1.00
        global_saliency_weight : 0.01
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
    accumulate_grad_batches: 24
    default_root_dir     : $DEFAULT_ROOT_DIR
    accelerator          : gpu
    devices              : 0,
    limit_val_batches    : 256
    limit_train_batches  : 2048
    num_sanity_val_steps : 0
    max_epochs           : 360
    callbacks:
        - class_path: pytorch_lightning.callbacks.ModelCheckpoint
          init_args:
              monitor: val_loss
              mode: min
              save_top_k: 5
              filename: '{epoch}-{step}-{val_loss:.3f}.ckpt'
              save_last: true

torch_globals:
    float32_matmul_precision: auto

initializer:
    init: $DVC_EXPT_DPATH/models/fusion/Drop6-MeanYear10GSD-V2/packages/Drop6_TCombo1Year_BAS_10GSD_V2_landcover_split6_V47/Drop6_TCombo1Year_BAS_10GSD_V2_landcover_split6_V47_epoch47_step3026.pt
"


# On toothbrush - SC training - test behavior?
export CUDA_VISIBLE_DEVICES=1
DVC_DATA_DPATH=$(geowatch_dvc --tags='drop7_data' --hardware='auto')
DVC_EXPT_DPATH=$(geowatch_dvc --tags='phase2_expt' --hardware='auto')
echo "DVC_EXPT_DPATH = $DVC_EXPT_DPATH"
WORKDIR=$DVC_EXPT_DPATH/training/$HOSTNAME/$USER
DATASET_CODE=Drop7-Cropped2GSD
KWCOCO_BUNDLE_DPATH=$DVC_DATA_DPATH/$DATASET_CODE
TRAIN_FPATH=$KWCOCO_BUNDLE_DPATH/data_train_rawbands_split6.kwcoco.zip
VALI_FPATH=$KWCOCO_BUNDLE_DPATH/data_vali_rawbands_split6.kwcoco.zip
CHANNELS="(L8,S2):(blue|green|red|nir),(WV):(blue|green|red)"
EXPERIMENT_NAME=Drop7-Cropped2GSD_SC_bgrn_split6_V07
DEFAULT_ROOT_DIR=$WORKDIR/$DATASET_CODE/runs/$EXPERIMENT_NAME
TARGET_LR=1e-4
WEIGHT_DECAY=$(python -c "print($TARGET_LR * 0.01)")
echo "WEIGHT_DECAY = $WEIGHT_DECAY"
MAX_STEPS=80000
WATCH_GRID_WORKERS=0 python -m geowatch.tasks.fusion fit --config "
data:
    select_videos          : $SELECT_VIDEOS
    num_workers            : 5
    train_dataset          : $TRAIN_FPATH
    vali_dataset           : $VALI_FPATH
    window_dims            : '224,224'
    time_steps             : 9
    time_sampling          : uniform-soft5-soft4-contiguous
    time_kernel            : '(-1.25y,-1.08y,-0.25y,-0.08y,0.0y,0.08y,0.25y,1.08y,1.25y)'
    window_resolution     : 2.0GSD
    input_resolution      : 2.0GSD
    output_resolution     : 2.0GSD
    neg_to_pos_ratio       : 1.0
    batch_size             : 4
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
    use_grid_positives     : False
    use_grid_negatives     : False
    normalize_inputs       : 1024
    balance_areas          : True
model:
    class_path: MultimodalTransformer
    init_args:
        #saliency_weights      : '1:1'
        #class_weights         : auto
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
        global_class_weight    : 1.00
        global_saliency_weight : 0.01
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
    accumulate_grad_batches: 24
    default_root_dir     : $DEFAULT_ROOT_DIR
    accelerator          : gpu
    devices              : 0,
    limit_val_batches    : 256
    limit_train_batches  : 2048
    num_sanity_val_steps : 0
    max_epochs           : 360
    callbacks:
        - class_path: pytorch_lightning.callbacks.ModelCheckpoint
          init_args:
              monitor: val_loss
              mode: min
              save_top_k: 5
              filename: '{epoch}-{step}-{val_loss:.3f}.ckpt'
              save_last: true

torch_globals:
    float32_matmul_precision: auto

initializer:
    init: $DVC_EXPT_DPATH/models/fusion/Drop6-MeanYear10GSD-V2/packages/Drop6_TCombo1Year_BAS_10GSD_V2_landcover_split6_V47/Drop6_TCombo1Year_BAS_10GSD_V2_landcover_split6_V47_epoch47_step3026.pt
"


# On toothbrush - SC training - test behavior?
export CUDA_VISIBLE_DEVICES=1
DVC_DATA_DPATH=$(geowatch_dvc --tags='drop7_data' --hardware='auto')
DVC_EXPT_DPATH=$(geowatch_dvc --tags='phase2_expt' --hardware='auto')
echo "DVC_EXPT_DPATH = $DVC_EXPT_DPATH"
WORKDIR=$DVC_EXPT_DPATH/training/$HOSTNAME/$USER
DATASET_CODE=Drop7-Cropped2GSD
KWCOCO_BUNDLE_DPATH=$DVC_DATA_DPATH/$DATASET_CODE
TRAIN_FPATH=$KWCOCO_BUNDLE_DPATH/data_train_rawbands_split6.kwcoco.zip
VALI_FPATH=$KWCOCO_BUNDLE_DPATH/data_vali_rawbands_split6.kwcoco.zip
CHANNELS="(L8,S2):(blue|green|red|nir),(WV):(blue|green|red),(WV,WV1):pan"
EXPERIMENT_NAME=Drop7-Cropped2GSD_SC_bgrn_split6_V08
DEFAULT_ROOT_DIR=$WORKDIR/$DATASET_CODE/runs/$EXPERIMENT_NAME
TARGET_LR=1e-4
WEIGHT_DECAY=$(python -c "print($TARGET_LR * 0.01)")
echo "WEIGHT_DECAY = $WEIGHT_DECAY"
MAX_STEPS=80000
WATCH_GRID_WORKERS=0 python -m geowatch.tasks.fusion fit --config "
data:
    select_videos          : $SELECT_VIDEOS
    num_workers            : 5
    train_dataset          : $TRAIN_FPATH
    vali_dataset           : $VALI_FPATH
    window_dims            : '224,224'
    time_steps             : 9
    time_sampling          : soft4
    time_kernel            : '(-1.08y,-1y,-0.25y,-0.08y,0.0y,0.08y,0.25y,1y,1.08y)'
    window_resolution     : 2.0GSD
    input_resolution      : 2.0GSD
    output_resolution     : 2.0GSD
    neg_to_pos_ratio       : 1.0
    batch_size             : 4
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
    use_grid_positives     : False
    use_grid_negatives     : False
    normalize_inputs       : 1024
    balance_areas          : True
model:
    class_path: MultimodalTransformer
    init_args:
        #saliency_weights      : '1:1'
        #class_weights         : auto
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
        global_class_weight    : 1.00
        global_saliency_weight : 0.0001
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
    accumulate_grad_batches: 24
    default_root_dir     : $DEFAULT_ROOT_DIR
    accelerator          : gpu
    devices              : 0,
    limit_val_batches    : 256
    limit_train_batches  : 2048
    num_sanity_val_steps : 0
    max_epochs           : 360
    callbacks:
        - class_path: pytorch_lightning.callbacks.ModelCheckpoint
          init_args:
              monitor: val_loss
              mode: min
              save_top_k: 5
              filename: '{epoch}-{step}-{val_loss:.3f}.ckpt'
              save_last: true

torch_globals:
    float32_matmul_precision: auto

initializer:
    init: $DVC_EXPT_DPATH/models/fusion/Drop6-MeanYear10GSD-V2/packages/Drop6_TCombo1Year_BAS_10GSD_V2_landcover_split6_V47/Drop6_TCombo1Year_BAS_10GSD_V2_landcover_split6_V47_epoch47_step3026.pt
"


python -m geowatch.tasks.fusion fit --optimizer.help=torch.optim.AdamW
python -m geowatch.tasks.fusion fit --lr_scheduler.help=torch.optim.lr_scheduler.OneCycleLR


# On toothbrush - SC training - continued with shuffled hyperparams
export CUDA_VISIBLE_DEVICES=0
DVC_DATA_DPATH=$(geowatch_dvc --tags='drop7_data' --hardware='auto')
DVC_EXPT_DPATH=$(geowatch_dvc --tags='phase2_expt' --hardware='auto')
echo "DVC_EXPT_DPATH = $DVC_EXPT_DPATH"
WORKDIR=$DVC_EXPT_DPATH/training/$HOSTNAME/$USER
DATASET_CODE=Drop7-Cropped2GSD
KWCOCO_BUNDLE_DPATH=$DVC_DATA_DPATH/$DATASET_CODE
TRAIN_FPATH=$KWCOCO_BUNDLE_DPATH/data_train_rawbands_split6.kwcoco.zip
VALI_FPATH=$KWCOCO_BUNDLE_DPATH/data_vali_rawbands_split6.kwcoco.zip
CHANNELS="(L8,S2):(blue|green|red|nir),(WV):(blue|green|red),(WV,WV1):pan"
EXPERIMENT_NAME=Drop7-Cropped2GSD_SC_bgrn_split6_V09
DEFAULT_ROOT_DIR=$WORKDIR/$DATASET_CODE/runs/$EXPERIMENT_NAME
TARGET_LR=1e-4
WEIGHT_DECAY=$(python -c "print($TARGET_LR * 0.01)")
echo "WEIGHT_DECAY = $WEIGHT_DECAY"
MAX_STEPS=80000
WATCH_GRID_WORKERS=0 python -m geowatch.tasks.fusion fit --config "
data:
    select_videos          : $SELECT_VIDEOS
    num_workers            : 5
    train_dataset          : $TRAIN_FPATH
    vali_dataset           : $VALI_FPATH
    window_dims            : '224,224'
    time_steps             : 9
    time_sampling          : soft4
    time_kernel            : '(-1.08y,-1y,-0.25y,-0.08y,0.0y,0.08y,0.25y,1y,1.08y)'
    window_resolution     : 2.0GSD
    input_resolution      : 2.0GSD
    output_resolution     : 2.0GSD
    neg_to_pos_ratio       : 1.0
    batch_size             : 2
    normalize_perframe     : false
    normalize_peritem      : 'blue|green|red|nir|pan'
    max_epoch_length       : 1000000
    channels               : '$CHANNELS'
    min_spacetime_weight   : 0.6
    temporal_dropout       : 0.5
    mask_low_quality       : False
    mask_samecolor_method  : None
    observable_threshold   : 0.1
    quality_threshold      : 0.0
    weight_dilate          : 10
    use_centered_positives : True
    use_grid_positives     : False
    use_grid_negatives     : False
    normalize_inputs       : 1024
    balance_areas          : True
model:
    class_path: MultimodalTransformer
    init_args:
        #saliency_weights      : '1:1'
        #class_weights         : auto
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
        global_class_weight    : 1.00
        global_saliency_weight : 0.00001
        multimodal_reduce      : learned_linear
optimizer:
    class_path: torch.optim.AdamW
    init_args:
        lr           : $TARGET_LR
        weight_decay : $WEIGHT_DECAY
        betas:
            - 0.85
            - 0.998
lr_scheduler:
  class_path: torch.optim.lr_scheduler.OneCycleLR
  init_args:
    max_lr: $TARGET_LR
    total_steps: $MAX_STEPS
    anneal_strategy: cos
    pct_start: 0.3
    div_factor: 10
    final_div_factor: 10000
    cycle_momentum: false
trainer:
    accumulate_grad_batches: 48
    default_root_dir     : $DEFAULT_ROOT_DIR
    accelerator          : gpu
    devices              : 0,
    limit_val_batches    : 256
    limit_train_batches  : 2048
    num_sanity_val_steps : 0
    max_epochs           : 560
    callbacks:
        - class_path: pytorch_lightning.callbacks.ModelCheckpoint
          init_args:
              monitor: val_loss
              mode: min
              save_top_k: 5
              filename: '{epoch}-{step}-{val_loss:.3f}.ckpt'
              save_last: true

torch_globals:
    float32_matmul_precision: auto

initializer:
    init: $DVC_EXPT_DPATH/models/fusion/Drop7-Cropped2GSD/packages/Drop7-Cropped2GSD_SC_bgrn_split6_V08/Drop7-Cropped2GSD_SC_bgrn_split6_V08_epoch336_step28982.pt
"



# On toothbrush - SC training - continued with shuffled hyperparams
export CUDA_VISIBLE_DEVICES=1
DVC_DATA_DPATH=$(geowatch_dvc --tags='drop7_data' --hardware='auto')
DVC_EXPT_DPATH=$(geowatch_dvc --tags='phase2_expt' --hardware='auto')
echo "DVC_EXPT_DPATH = $DVC_EXPT_DPATH"
WORKDIR=$DVC_EXPT_DPATH/training/$HOSTNAME/$USER
DATASET_CODE=Drop7-Cropped2GSD
KWCOCO_BUNDLE_DPATH=$DVC_DATA_DPATH/$DATASET_CODE
TRAIN_FPATH=$KWCOCO_BUNDLE_DPATH/data_train_rawbands_split6.kwcoco.zip
VALI_FPATH=$KWCOCO_BUNDLE_DPATH/data_vali_rawbands_split6.kwcoco.zip
CHANNELS="(L8,S2):(blue|green|red|nir),(WV):(blue|green|red),(WV,WV1):pan"
EXPERIMENT_NAME=Drop7-Cropped2GSD_SC_bgrn_split6_V10
DEFAULT_ROOT_DIR=$WORKDIR/$DATASET_CODE/runs/$EXPERIMENT_NAME
TARGET_LR=1e-4
WEIGHT_DECAY=$(python -c "print($TARGET_LR * 0.01)")
echo "WEIGHT_DECAY = $WEIGHT_DECAY"
MAX_STEPS=80000
WATCH_GRID_WORKERS=0 python -m geowatch.tasks.fusion fit --config "
data:
    select_videos          : $SELECT_VIDEOS
    num_workers            : 5
    train_dataset          : $TRAIN_FPATH
    vali_dataset           : $VALI_FPATH
    window_dims            : '224,224'
    time_steps             : 9
    time_sampling          : soft4
    time_kernel            : '(-1.08y,-1y,-0.25y,-0.08y,0.0y,0.08y,0.25y,1y,1.08y)'
    window_resolution     : 2.0GSD
    input_resolution      : 2.0GSD
    output_resolution     : 2.0GSD
    neg_to_pos_ratio       : 1.0
    batch_size             : 4
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
    use_grid_positives     : False
    use_grid_negatives     : False
    normalize_inputs       : 1024
    balance_areas          : True
model:
    class_path: MultimodalTransformer
    init_args:
        #saliency_weights      : '1:1'
        #class_weights         : auto
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
        global_class_weight    : 1.00
        global_saliency_weight : 0.00001
        multimodal_reduce      : learned_linear
optimizer:
    class_path: torch.optim.AdamW
    init_args:
        lr           : $TARGET_LR
        weight_decay : $WEIGHT_DECAY
        betas:
            - 0.95
            - 0.998
lr_scheduler:
  class_path: torch.optim.lr_scheduler.OneCycleLR
  init_args:
    max_lr: $TARGET_LR
    total_steps: $MAX_STEPS
    anneal_strategy: cos
    pct_start: 0.3
    div_factor: 10
    final_div_factor: 10000
    cycle_momentum: false
trainer:
    accumulate_grad_batches: 48
    default_root_dir     : $DEFAULT_ROOT_DIR
    accelerator          : gpu
    devices              : 0,
    limit_val_batches    : 256
    limit_train_batches  : 2048
    num_sanity_val_steps : 0
    max_epochs           : 560
    callbacks:
        - class_path: pytorch_lightning.callbacks.ModelCheckpoint
          init_args:
              monitor: val_loss
              mode: min
              save_top_k: 5
              filename: '{epoch}-{step}-{val_loss:.3f}.ckpt'
              save_last: true

torch_globals:
    float32_matmul_precision: auto

initializer:
    init: $DVC_EXPT_DPATH/models/fusion/Drop7-Cropped2GSD/packages/Drop7-Cropped2GSD_SC_bgrn_split6_V08/Drop7-Cropped2GSD_SC_bgrn_split6_V08_epoch336_step28982.pt
"


# On toothbrush - SC training - continued with shuffled hyperparams (And a fixed loss function)
export CUDA_VISIBLE_DEVICES=1
DVC_DATA_DPATH=$(geowatch_dvc --tags='drop7_data' --hardware='auto')
DVC_EXPT_DPATH=$(geowatch_dvc --tags='phase2_expt' --hardware='auto')
echo "DVC_EXPT_DPATH = $DVC_EXPT_DPATH"
WORKDIR=$DVC_EXPT_DPATH/training/$HOSTNAME/$USER
DATASET_CODE=Drop7-Cropped2GSD
KWCOCO_BUNDLE_DPATH=$DVC_DATA_DPATH/$DATASET_CODE
TRAIN_FPATH=$KWCOCO_BUNDLE_DPATH/data_train_rawbands_split6.kwcoco.zip
VALI_FPATH=$KWCOCO_BUNDLE_DPATH/data_vali_rawbands_split6.kwcoco.zip
CHANNELS="(L8,S2):(blue|green|red|nir),(WV):(blue|green|red),(WV,WV1):pan"
EXPERIMENT_NAME=Drop7-Cropped2GSD_SC_bgrn_split6_V11
DEFAULT_ROOT_DIR=$WORKDIR/$DATASET_CODE/runs/$EXPERIMENT_NAME
TARGET_LR=1e-4
WEIGHT_DECAY=$(python -c "print($TARGET_LR * 0.01)")
echo "WEIGHT_DECAY = $WEIGHT_DECAY"
MAX_STEPS=80000
WATCH_GRID_WORKERS=0 python -m geowatch.tasks.fusion fit --config "
data:
    select_videos          : $SELECT_VIDEOS
    num_workers            : 5
    train_dataset          : $TRAIN_FPATH
    vali_dataset           : $VALI_FPATH
    window_dims            : '224,224'
    time_steps             : 9
    time_sampling          : soft4
    time_kernel            : '(-1.08y,-1y,-0.25y,-0.08y,0.0y,0.08y,0.25y,1y,1.08y)'
    window_resolution     : 2.0GSD
    input_resolution      : 2.0GSD
    output_resolution     : 2.0GSD
    neg_to_pos_ratio       : 1.0
    batch_size             : 2
    normalize_perframe     : false
    normalize_peritem      : 'blue|green|red|nir|pan'
    max_epoch_length       : 1000000
    channels               : '$CHANNELS'
    min_spacetime_weight   : 0.6
    temporal_dropout       : 0.5
    mask_low_quality       : False
    mask_samecolor_method  : None
    observable_threshold   : 0.1
    quality_threshold      : 0.0
    weight_dilate          : 10
    use_centered_positives : True
    use_grid_positives     : False
    use_grid_negatives     : False
    normalize_inputs       : 1024
    balance_areas          : True
model:
    class_path: MultimodalTransformer
    init_args:
        #saliency_weights      : '1:1'
        #class_weights         : auto
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
        global_class_weight    : 1.00
        global_saliency_weight : 0.00001
        multimodal_reduce      : learned_linear
optimizer:
    class_path: torch.optim.AdamW
    init_args:
        lr           : $TARGET_LR
        weight_decay : $WEIGHT_DECAY
        betas:
            - 0.85
            - 0.998
lr_scheduler:
  class_path: torch.optim.lr_scheduler.OneCycleLR
  init_args:
    max_lr: $TARGET_LR
    total_steps: $MAX_STEPS
    anneal_strategy: cos
    pct_start: 0.3
    div_factor: 10
    final_div_factor: 10000
    cycle_momentum: false
trainer:
    accumulate_grad_batches: 48
    default_root_dir     : $DEFAULT_ROOT_DIR
    accelerator          : gpu
    devices              : 0,
    limit_val_batches    : 256
    limit_train_batches  : 2048
    num_sanity_val_steps : 0
    max_epochs           : 560
    callbacks:
        - class_path: pytorch_lightning.callbacks.ModelCheckpoint
          init_args:
              monitor: val_loss
              mode: min
              save_top_k: 5
              filename: '{epoch}-{step}-{val_loss:.3f}.ckpt'
              save_last: true

torch_globals:
    float32_matmul_precision: auto

initializer:
    init: $DVC_EXPT_DPATH/models/fusion/Drop7-Cropped2GSD/packages/Drop7-Cropped2GSD_SC_bgrn_split6_V08/Drop7-Cropped2GSD_SC_bgrn_split6_V08_epoch336_step28982.pt
"


# On yardrat - AC/SC depth
export CUDA_VISIBLE_DEVICES=0
DVC_DATA_DPATH=$(geowatch_dvc --tags='drop7_data' --hardware='auto')
DVC_EXPT_DPATH=$(geowatch_dvc --tags='phase2_expt' --hardware='auto')
echo "DVC_EXPT_DPATH = $DVC_EXPT_DPATH"
WORKDIR=$DVC_EXPT_DPATH/training/$HOSTNAME/$USER
DATASET_CODE=Drop7-Cropped2GSD
KWCOCO_BUNDLE_DPATH=$DVC_DATA_DPATH/$DATASET_CODE
TRAIN_FPATH=$KWCOCO_BUNDLE_DPATH/data_train_depth_split6.kwcoco.zip
VALI_FPATH=$KWCOCO_BUNDLE_DPATH/data_vali_depth_split6.kwcoco.zip
CHANNELS="(L8,S2):(blue|green|red|nir,depth),(WV):(blue|green|red,depth)"
EXPERIMENT_NAME=Drop7-Cropped2GSD_SC_bgrn_depth_split6_V12
DEFAULT_ROOT_DIR=$WORKDIR/$DATASET_CODE/runs/$EXPERIMENT_NAME
TARGET_LR=1e-4
WEIGHT_DECAY=$(python -c "print($TARGET_LR * 0.01)")
echo "WEIGHT_DECAY = $WEIGHT_DECAY"
MAX_STEPS=80000
WATCH_GRID_WORKERS=0 python -m geowatch.tasks.fusion fit --config "
data:
    select_videos          : $SELECT_VIDEOS
    num_workers            : 5
    train_dataset          : $TRAIN_FPATH
    vali_dataset           : $VALI_FPATH
    window_dims            : '196,196'
    time_steps             : 9
    time_sampling          : uniform-soft5-soft4-contiguous
    time_kernel            : '(-1.0y,-0.5y,-0.25y,-0.08y,0.0y,0.08y,0.25y,0.5y,1.0y)'
    window_resolution     : 2.0GSD
    input_resolution      : 2.0GSD
    output_resolution     : 2.0GSD
    neg_to_pos_ratio       : 1.0
    batch_size             : 2
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
    use_grid_positives     : False
    use_grid_negatives     : False
    normalize_inputs       : 1024
    balance_areas          : True
model:
    class_path: MultimodalTransformer
    init_args:
        #saliency_weights      : '1:1'
        #class_weights         : auto
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
        global_class_weight    : 1.00
        global_saliency_weight : 0.01
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
    accumulate_grad_batches: 24
    default_root_dir     : $DEFAULT_ROOT_DIR
    accelerator          : gpu
    devices              : 0,
    limit_val_batches    : 256
    limit_train_batches  : 2048
    num_sanity_val_steps : 0
    max_epochs           : 360
    callbacks:
        - class_path: pytorch_lightning.callbacks.ModelCheckpoint
          init_args:
              monitor: val_loss
              mode: min
              save_top_k: 5
              filename: '{epoch}-{step}-{val_loss:.3f}.ckpt'
              save_last: true

torch_globals:
    float32_matmul_precision: auto

initializer:
    init: $DVC_EXPT_DPATH/models/fusion/Drop4-SC/packages/Drop4_tune_V30_8GSD_V3/Drop4_tune_V30_8GSD_V3_epoch=2-step=17334.pt.pt
"


# '(-2.0y,-1.0y,-0.25y,-0.08y,0.0y,0.08y,0.25y,1.0y,2.0y)'

# On yardrat - AC/SC MAE
export CUDA_VISIBLE_DEVICES=1
DVC_DATA_DPATH=$(geowatch_dvc --tags='drop7_data' --hardware='auto')
DVC_EXPT_DPATH=$(geowatch_dvc --tags='phase2_expt' --hardware='auto')
echo "DVC_EXPT_DPATH = $DVC_EXPT_DPATH"
WORKDIR=$DVC_EXPT_DPATH/training/$HOSTNAME/$USER
DATASET_CODE=Drop7-Cropped2GSD
KWCOCO_BUNDLE_DPATH=$DVC_DATA_DPATH/$DATASET_CODE
TRAIN_FPATH=$KWCOCO_BUNDLE_DPATH/data_train_mae_split6.kwcoco.zip
VALI_FPATH=$KWCOCO_BUNDLE_DPATH/data_vali_mae_split6.kwcoco.zip
CHANNELS="(L8,S2):(blue|green|red),(S2):(mae.0:16),(WV):(blue|green|red)"
EXPERIMENT_NAME=Drop7-Cropped2GSD_SC_bgrn_depth_split6_V14
DEFAULT_ROOT_DIR=$WORKDIR/$DATASET_CODE/runs/$EXPERIMENT_NAME
TARGET_LR=1e-4
WEIGHT_DECAY=$(python -c "print($TARGET_LR * 0.01)")
echo "WEIGHT_DECAY = $WEIGHT_DECAY"
MAX_STEPS=80000
WATCH_GRID_WORKERS=0 python -m geowatch.tasks.fusion fit --config "
data:
    select_videos          : $SELECT_VIDEOS
    num_workers            : 5
    train_dataset          : $TRAIN_FPATH
    vali_dataset           : $VALI_FPATH
    window_dims            : '196,196'
    time_steps             : 9
    time_sampling          : uniform-soft5-soft4-contiguous
    time_kernel            : '(-1.0y,-0.5y,-0.25y,-0.08y,0.0y,0.08y,0.25y,0.5y,1.0y)'
    window_resolution     : 2.0GSD
    input_resolution      : 2.0GSD
    output_resolution     : 2.0GSD
    neg_to_pos_ratio       : 1.0
    batch_size             : 2
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
    use_grid_positives     : False
    use_grid_negatives     : False
    normalize_inputs       : 1024
    balance_areas          : True
model:
    class_path: MultimodalTransformer
    init_args:
        #saliency_weights      : '1:1'
        #class_weights         : auto
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
        global_class_weight    : 1.00
        global_saliency_weight : 0.01
        multimodal_reduce      : learned_linear
        attention_kwargs:
            add_zero_attn: true
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
    accumulate_grad_batches: 24
    default_root_dir     : $DEFAULT_ROOT_DIR
    accelerator          : gpu
    devices              : 0,
    limit_val_batches    : 256
    limit_train_batches  : 2048
    num_sanity_val_steps : 0
    max_epochs           : 360
    callbacks:
        - class_path: pytorch_lightning.callbacks.ModelCheckpoint
          init_args:
              monitor: val_loss
              mode: min
              save_top_k: 5
              filename: '{epoch}-{step}-{val_loss:.3f}.ckpt'
              save_last: true

torch_globals:
    float32_matmul_precision: auto

initializer:
    init: $DVC_EXPT_DPATH/models/fusion/Drop4-SC/packages/Drop4_tune_V30_8GSD_V3/Drop4_tune_V30_8GSD_V3_epoch=2-step=17334.pt.pt
"



# On namek - AC/SC MAE
export CUDA_VISIBLE_DEVICES=1
DVC_DATA_DPATH=$(geowatch_dvc --tags='drop7_data' --hardware='auto')
DVC_EXPT_DPATH=$(geowatch_dvc --tags='phase2_expt' --hardware='auto')
echo "DVC_EXPT_DPATH = $DVC_EXPT_DPATH"
WORKDIR=$DVC_EXPT_DPATH/training/$HOSTNAME/$USER
DATASET_CODE=Drop7-Cropped2GSD
KWCOCO_BUNDLE_DPATH=$DVC_DATA_DPATH/$DATASET_CODE
TRAIN_FPATH=$KWCOCO_BUNDLE_DPATH/data_train_rawbands_split6.kwcoco.zip
VALI_FPATH=$KWCOCO_BUNDLE_DPATH/data_vali_rawbands_split6.kwcoco.zip
CHANNELS="(L8,S2):(blue|green|red),(WV):(blue|green|red)"
EXPERIMENT_NAME=Drop7-Cropped2GSD_SC_bgrn_depth_split6_V15
DEFAULT_ROOT_DIR=$WORKDIR/$DATASET_CODE/runs/$EXPERIMENT_NAME
TARGET_LR=1e-4
WEIGHT_DECAY=$(python -c "print($TARGET_LR * 0.01)")
echo "WEIGHT_DECAY = $WEIGHT_DECAY"
MAX_STEPS=80000
WATCH_GRID_WORKERS=0 python -m geowatch.tasks.fusion fit --config "
data:
    select_videos          : $SELECT_VIDEOS
    num_workers            : 5
    train_dataset          : $TRAIN_FPATH
    vali_dataset           : $VALI_FPATH
    window_dims            : '128,128'
    time_steps             : 11
    time_sampling          : uniform-soft5-soft4-contiguous
    time_kernel            : '(-3.0y,-2.0y,-1.0y,-0.25y,-0.08y,0.0y,0.08y,0.25y,1.0y,2.0y,3.0y)'
    window_resolution     : 2.0GSD
    input_resolution      : 2.0GSD
    output_resolution     : 2.0GSD
    neg_to_pos_ratio       : 1.0
    batch_size             : 2
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
    use_grid_positives     : False
    use_grid_negatives     : False
    normalize_inputs       : 1024
    balance_areas          : True
model:
    class_path: MultimodalTransformer
    init_args:
        #saliency_weights      : '1:1'
        #class_weights         : auto
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
        global_class_weight    : 1.00
        global_saliency_weight : 0.01
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
    accumulate_grad_batches: 24
    default_root_dir     : $DEFAULT_ROOT_DIR
    accelerator          : gpu
    devices              : 0,
    limit_val_batches    : 256
    limit_train_batches  : 2048
    num_sanity_val_steps : 0
    max_epochs           : 360
    callbacks:
        - class_path: pytorch_lightning.callbacks.ModelCheckpoint
          init_args:
              monitor: val_loss
              mode: min
              save_top_k: 5
              filename: '{epoch}-{step}-{val_loss:.3f}.ckpt'
              save_last: true

torch_globals:
    float32_matmul_precision: auto

initializer:
    init: $DVC_EXPT_DPATH/models/fusion/Drop4-SC/packages/Drop4_tune_V30_8GSD_V3/Drop4_tune_V30_8GSD_V3_epoch=2-step=17334.pt.pt
"


# On ooo - AC/SC MAE
export CUDA_VISIBLE_DEVICES="1"
DVC_DATA_DPATH=$(geowatch_dvc --tags='drop7_data' --hardware='auto')
DVC_EXPT_DPATH=$(geowatch_dvc --tags='phase2_expt' --hardware='auto')
echo "DVC_EXPT_DPATH = $DVC_EXPT_DPATH"
WORKDIR=$DVC_EXPT_DPATH/training/$HOSTNAME/$USER
DATASET_CODE=Drop7-Cropped2GSD
KWCOCO_BUNDLE_DPATH=$DVC_DATA_DPATH/$DATASET_CODE
TRAIN_FPATH=$KWCOCO_BUNDLE_DPATH/data_train_rawbands_split6.kwcoco.zip
VALI_FPATH=$KWCOCO_BUNDLE_DPATH/data_vali_rawbands_split6.kwcoco.zip
CHANNELS="(L8,S2):(blue|green|red),(WV):(blue|green|red)"
EXPERIMENT_NAME=Drop7-Cropped2GSD_SC_bgrn_split6_V16
DEFAULT_ROOT_DIR=$WORKDIR/$DATASET_CODE/runs/$EXPERIMENT_NAME
TARGET_LR=1e-4
WEIGHT_DECAY=$(python -c "print($TARGET_LR * 0.01)")
echo "WEIGHT_DECAY = $WEIGHT_DECAY"
MAX_STEPS=80000
WATCH_GRID_WORKERS=0 python -m geowatch.tasks.fusion fit --config "
data:
    select_videos          : $SELECT_VIDEOS
    num_workers            : 1
    train_dataset          : $TRAIN_FPATH
    vali_dataset           : $VALI_FPATH
    window_dims            : '128,128'
    time_steps             : 11
    time_sampling          : uniform-soft5-soft4-contiguous
    time_kernel            : '(-3.0y,-2.0y,-1.0y,-0.25y,-0.08y,0.0y,0.08y,0.25y,1.0y,2.0y,3.0y)'
    window_resolution     : 2.0GSD
    input_resolution      : 2.0GSD
    output_resolution     : 2.0GSD
    neg_to_pos_ratio       : 1.0
    batch_size             : 5
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
    use_grid_positives     : False
    use_grid_negatives     : False
    normalize_inputs       : 1024
    balance_areas          : True
    sqlview                : sqlite
model:
    class_path: MultimodalTransformer
    init_args:
        #saliency_weights      : '1:1'
        #class_weights         : auto
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
        global_class_weight    : 1.00
        global_saliency_weight : 0.01
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
    accumulate_grad_batches: 24
    default_root_dir     : $DEFAULT_ROOT_DIR
    accelerator          : gpu
    devices              : 0,
    limit_val_batches    : 256
    limit_train_batches  : 2048
    num_sanity_val_steps : 0
    max_epochs           : 360
    callbacks:
        - class_path: pytorch_lightning.callbacks.ModelCheckpoint
          init_args:
              monitor: val_loss
              mode: min
              save_top_k: 5
              filename: '{epoch}-{step}-{val_loss:.3f}.ckpt'
              save_last: true

torch_globals:
    float32_matmul_precision: auto

initializer:
    init: $DVC_EXPT_DPATH/models/fusion/Drop4-SC/packages/Drop4_tune_V30_8GSD_V3/Drop4_tune_V30_8GSD_V3_epoch=2-step=17334.pt.pt
"




# On toothbrush - AC/SC MAE
export CUDA_VISIBLE_DEVICES="1"
DVC_DATA_DPATH=$(geowatch_dvc --tags='drop7_data' --hardware='auto')
DVC_EXPT_DPATH=$(geowatch_dvc --tags='phase2_expt' --hardware='auto')
echo "DVC_EXPT_DPATH = $DVC_EXPT_DPATH"
WORKDIR=$DVC_EXPT_DPATH/training/$HOSTNAME/$USER
DATASET_CODE=Drop7-Cropped2GSD
KWCOCO_BUNDLE_DPATH=$DVC_DATA_DPATH/$DATASET_CODE
TRAIN_FPATH=$KWCOCO_BUNDLE_DPATH/data_train_rawbands_split6.kwcoco.zip
VALI_FPATH=$KWCOCO_BUNDLE_DPATH/data_vali_rawbands_split6.kwcoco.zip
CHANNELS="(L8,S2):(blue|green|red),(WV):(blue|green|red)"
EXPERIMENT_NAME=Drop7-Cropped2GSD_SC_bgrn_split6_V17
DEFAULT_ROOT_DIR=$WORKDIR/$DATASET_CODE/runs/$EXPERIMENT_NAME
TARGET_LR=3e-4
WEIGHT_DECAY=$(python -c "print($TARGET_LR * 0.01)")
echo "WEIGHT_DECAY = $WEIGHT_DECAY"
MAX_STEPS=80000
WATCH_GRID_WORKERS=0 python -m geowatch.tasks.fusion fit --config "
data:
    select_videos          : $SELECT_VIDEOS
    num_workers            : 1
    train_dataset          : $TRAIN_FPATH
    vali_dataset           : $VALI_FPATH
    window_dims            : '128,128'
    time_steps             : 11
    time_sampling          : uniform-soft5-soft4-contiguous
    time_kernel            : '(-3.0y,-2.0y,-1.0y,-0.25y,-0.08y,0.0y,0.08y,0.25y,1.0y,2.0y,3.0y)'
    window_resolution     : 2.0GSD
    input_resolution      : 2.0GSD
    output_resolution     : 2.0GSD
    neg_to_pos_ratio       : 1.0
    batch_size             : 5
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
    use_grid_positives     : False
    use_grid_negatives     : False
    normalize_inputs       : 1024
    balance_areas          : True
    sqlview                : sqlite
model:
    class_path: MultimodalTransformer
    init_args:
        #saliency_weights      : '1:1'
        #class_weights         : auto
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
        global_class_weight    : 1.00
        global_saliency_weight : 0.01
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
    accumulate_grad_batches: 24
    default_root_dir     : $DEFAULT_ROOT_DIR
    accelerator          : gpu
    devices              : 0,
    limit_val_batches    : 256
    limit_train_batches  : 2048
    num_sanity_val_steps : 0
    max_epochs           : 360
    callbacks:
        - class_path: pytorch_lightning.callbacks.ModelCheckpoint
          init_args:
              monitor: val_loss
              mode: min
              save_top_k: 5
              filename: '{epoch}-{step}-{val_loss:.3f}.ckpt'
              save_last: true

torch_globals:
    float32_matmul_precision: auto

initializer:
    init: $DVC_EXPT_DPATH/models/fusion/Drop4-SC/packages/Drop4_tune_V30_8GSD_V3/Drop4_tune_V30_8GSD_V3_epoch=2-step=17334.pt.pt
"

# On yardrat - AC/SC MAE
export CUDA_VISIBLE_DEVICES=0
DVC_DATA_DPATH=$(geowatch_dvc --tags='drop7_data' --hardware='auto')
DVC_EXPT_DPATH=$(geowatch_dvc --tags='phase2_expt' --hardware='auto')
echo "DVC_EXPT_DPATH = $DVC_EXPT_DPATH"
WORKDIR=$DVC_EXPT_DPATH/training/$HOSTNAME/$USER
DATASET_CODE=Drop7-Cropped2GSD
KWCOCO_BUNDLE_DPATH=$DVC_DATA_DPATH/$DATASET_CODE
TRAIN_FPATH=$KWCOCO_BUNDLE_DPATH/data_train_rawbands_split6.kwcoco.zip
VALI_FPATH=$KWCOCO_BUNDLE_DPATH/data_vali_rawbands_split6.kwcoco.zip
CHANNELS="(S2):(blue|green|red),(WV):(blue|green|red)"
EXPERIMENT_NAME=Drop7-Cropped2GSD_SC_bgr_split6_V19
DEFAULT_ROOT_DIR=$WORKDIR/$DATASET_CODE/runs/$EXPERIMENT_NAME
TARGET_LR=1e-4
WEIGHT_DECAY=$(python -c "print($TARGET_LR * 0.01)")
echo "WEIGHT_DECAY = $WEIGHT_DECAY"
MAX_STEPS=80000
WATCH_GRID_WORKERS=0 python -m geowatch.tasks.fusion fit --config "
data:
    select_videos          : $SELECT_VIDEOS
    num_workers            : 5
    train_dataset          : $TRAIN_FPATH
    vali_dataset           : $VALI_FPATH
    window_dims            : '128,128'
    time_steps             : 13
    time_sampling          : uniform-soft5-soft4-contiguous
    time_kernel            : '(-4.0y, -3.0y,-2.0y,-1.0y,-0.25y,-0.08y,0.0y,0.08y,0.25y,1.0y,2.0y,3.0y,4.0y)'
    window_resolution     : 2.0GSD
    input_resolution      : 2.0GSD
    output_resolution     : 2.0GSD
    neg_to_pos_ratio       : 1.0
    batch_size             : 2
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
    use_grid_positives     : False
    use_grid_negatives     : False
    normalize_inputs       : 1024
    balance_areas          : True
model:
    class_path: MultimodalTransformer
    init_args:
        #saliency_weights      : '1:1'
        #class_weights         : auto
        tokenizer              : linconv
        arch_name              : smt_it_stm_p24
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
        global_class_weight    : 1.00
        global_saliency_weight : 0.01
        multimodal_reduce      : learned_linear
        attention_kwargs:
            add_zero_attn: true
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
    accumulate_grad_batches: 24
    default_root_dir     : $DEFAULT_ROOT_DIR
    accelerator          : gpu
    devices              : 0,
    limit_val_batches    : 256
    limit_train_batches  : 2048
    num_sanity_val_steps : 0
    max_epochs           : 360
    callbacks:
        - class_path: pytorch_lightning.callbacks.ModelCheckpoint
          init_args:
              monitor: val_loss
              mode: min
              save_top_k: 5
              filename: '{epoch}-{step}-{val_loss:.3f}.ckpt'
              save_last: true

torch_globals:
    float32_matmul_precision: auto

initializer:
    init: $DVC_EXPT_DPATH/models/fusion/Drop4-SC/packages/Drop4_tune_V30_8GSD_V3/Drop4_tune_V30_8GSD_V3_epoch=2-step=17334.pt.pt
"



# On ooo - AC/SC MAE
export CUDA_VISIBLE_DEVICES="1"
DVC_DATA_DPATH=$(geowatch_dvc --tags='drop7_data' --hardware='auto')
DVC_EXPT_DPATH=$(geowatch_dvc --tags='phase2_expt' --hardware='auto')
echo "DVC_EXPT_DPATH = $DVC_EXPT_DPATH"
WORKDIR=$DVC_EXPT_DPATH/training/$HOSTNAME/$USER
DATASET_CODE=Drop7-Cropped2GSD
KWCOCO_BUNDLE_DPATH=$DVC_DATA_DPATH/$DATASET_CODE
TRAIN_FPATH=$KWCOCO_BUNDLE_DPATH/data_train_rawbands_split6.kwcoco.zip
VALI_FPATH=$KWCOCO_BUNDLE_DPATH/data_vali_rawbands_split6.kwcoco.zip
CHANNELS="(L8,S2):(blue|green|red),(WV):(blue|green|red)"
EXPERIMENT_NAME=Drop7-Cropped2GSD_SC_bgrn_split6_V20
DEFAULT_ROOT_DIR=$WORKDIR/$DATASET_CODE/runs/$EXPERIMENT_NAME
TARGET_LR=1e-4
WEIGHT_DECAY=$(python -c "print($TARGET_LR * 0.01)")
echo "WEIGHT_DECAY = $WEIGHT_DECAY"
MAX_STEPS=80000
WATCH_GRID_WORKERS=0 python -m geowatch.tasks.fusion fit --config "
data:
    select_videos          : $SELECT_VIDEOS
    num_workers            : 1
    train_dataset          : $TRAIN_FPATH
    vali_dataset           : $VALI_FPATH
    window_dims            : '128,128'
    time_steps             : 11
    time_sampling          : uniform-soft5-soft4-contiguous
    time_kernel            : '(-3.0y,-2.0y,-1.0y,-0.25y,-0.08y,0.0y,0.08y,0.25y,1.0y,2.0y,3.0y)'
    window_resolution     : 2.0GSD
    input_resolution      : 2.0GSD
    output_resolution     : 2.0GSD
    neg_to_pos_ratio       : 1.0
    batch_size             : 5
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
    use_grid_positives     : False
    use_grid_negatives     : False
    normalize_inputs       : 1024
    balance_areas          : True
    sqlview                : sqlite
model:
    class_path: MultimodalTransformer
    init_args:
        #saliency_weights      : '1:1'
        #class_weights         : auto
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
        global_class_weight    : 1.00
        global_saliency_weight : 0.01
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
    accumulate_grad_batches: 24
    default_root_dir     : $DEFAULT_ROOT_DIR
    accelerator          : gpu
    devices              : 0,
    limit_val_batches    : 256
    limit_train_batches  : 2048
    num_sanity_val_steps : 0
    max_epochs           : 360
    callbacks:
        - class_path: pytorch_lightning.callbacks.ModelCheckpoint
          init_args:
              monitor: val_loss
              mode: min
              save_top_k: 5
              filename: '{epoch}-{step}-{val_loss:.3f}.ckpt'
              save_last: true

torch_globals:
    float32_matmul_precision: auto

initializer:
    init: $DVC_EXPT_DPATH/models/fusion/Drop4-SC/packages/Drop4_tune_V30_8GSD_V3/Drop4_tune_V30_8GSD_V3_epoch=2-step=17334.pt.pt
"


# On yardrat - retrain BAS with fixed annotations - no teamfeat / pan (lower LR)
export CUDA_VISIBLE_DEVICES=0
DVC_DATA_DPATH=$(geowatch_dvc --tags='phase2_data' --hardware='auto')
DVC_EXPT_DPATH=$(geowatch_dvc --tags='phase2_expt' --hardware='auto')
echo "DVC_EXPT_DPATH = $DVC_EXPT_DPATH"
WORKDIR=$DVC_EXPT_DPATH/training/$HOSTNAME/$USER
DATASET_CODE=Drop7-MedianNoWinter10GSD
KWCOCO_BUNDLE_DPATH=$DVC_DATA_DPATH/$DATASET_CODE
TRAIN_FPATH=$KWCOCO_BUNDLE_DPATH/data_train_EI2LMSC_split6.kwcoco.zip
VALI_FPATH=$KWCOCO_BUNDLE_DPATH/data_vali_EI2LMSC_split6.kwcoco.zip
CHANNELS="(L8,S2):(blue|green|red|nir),(WV):(blue|green|red)"
EXPERIMENT_NAME=Drop7-MedianNoWinter10GSD_bgrn_split6_V70
DEFAULT_ROOT_DIR=$WORKDIR/$DATASET_CODE/runs/$EXPERIMENT_NAME
TARGET_LR=1e-5
WEIGHT_DECAY=$(python -c "print($TARGET_LR * 0.01)")
echo "WEIGHT_DECAY = $WEIGHT_DECAY"
MAX_STEPS=80000
WATCH_GRID_WORKERS=0 python -m geowatch.tasks.fusion fit --config "
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
        #saliency_weights      : '1:1'
        #class_weights         : auto
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
    accumulate_grad_batches: 24
    default_root_dir     : $DEFAULT_ROOT_DIR
    accelerator          : gpu
    devices              : 0,
    limit_val_batches    : 256
    limit_train_batches  : 2048
    num_sanity_val_steps : 0
    max_epochs           : 360
    callbacks:
        - class_path: pytorch_lightning.callbacks.ModelCheckpoint
          init_args:
              monitor: val_loss
              mode: min
              save_top_k: 5
              filename: '{epoch}-{step}-{val_loss:.3f}.ckpt'
              save_last: true

torch_globals:
    float32_matmul_precision: auto

initializer:
    init: $DVC_EXPT_DPATH/models/fusion/Drop6-MeanYear10GSD-V2/packages/Drop6_TCombo1Year_BAS_10GSD_V2_landcover_split6_V47/Drop6_TCombo1Year_BAS_10GSD_V2_landcover_split6_V47_epoch47_step3026.pt
"


# On toothbrush - retrain BAS with fixed annotations - no teamfeat / pan (higher LR)
export CUDA_VISIBLE_DEVICES=1
DVC_DATA_DPATH=$(geowatch_dvc --tags='phase2_data' --hardware='auto')
DVC_EXPT_DPATH=$(geowatch_dvc --tags='phase2_expt' --hardware='auto')
echo "DVC_EXPT_DPATH = $DVC_EXPT_DPATH"
WORKDIR=$DVC_EXPT_DPATH/training/$HOSTNAME/$USER
DATASET_CODE=Drop7-MedianNoWinter10GSD
KWCOCO_BUNDLE_DPATH=$DVC_DATA_DPATH/$DATASET_CODE
TRAIN_FPATH=$KWCOCO_BUNDLE_DPATH/data_train_EI2LMSC_split6.kwcoco.zip
VALI_FPATH=$KWCOCO_BUNDLE_DPATH/data_vali_EI2LMSC_split6.kwcoco.zip
CHANNELS="(L8,S2):(blue|green|red|nir),(WV):(blue|green|red)"
EXPERIMENT_NAME=Drop7-MedianNoWinter10GSD_bgrn_split6_V71
DEFAULT_ROOT_DIR=$WORKDIR/$DATASET_CODE/runs/$EXPERIMENT_NAME
TARGET_LR=1e-4
WEIGHT_DECAY=$(python -c "print($TARGET_LR * 0.01)")
echo "WEIGHT_DECAY = $WEIGHT_DECAY"
MAX_STEPS=80000
WATCH_GRID_WORKERS=0 python -m geowatch.tasks.fusion fit --config "
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
    mask_low_quality       : False
    mask_samecolor_method  : histogram
    observable_threshold   : 0.1
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
        #saliency_weights      : '1:1'
        class_weights          : 'auto:ignore+0.000001,Unknown+0.000001,Unknown+0.000001,transient*0.1+0.000001'
        tokenizer              : linconv
        arch_name              : smt_it_stm_p24
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
    accumulate_grad_batches: 24
    default_root_dir     : $DEFAULT_ROOT_DIR
    accelerator          : gpu
    devices              : 0,
    limit_val_batches    : 256
    limit_train_batches  : 2048
    num_sanity_val_steps : 0
    max_epochs           : 360
    callbacks:
        - class_path: pytorch_lightning.callbacks.ModelCheckpoint
          init_args:
              monitor: val_loss
              mode: min
              save_top_k: 5
              filename: '{epoch}-{step}-{val_loss:.3f}.ckpt'
              save_last: true

torch_globals:
    float32_matmul_precision: auto

initializer:
    init: $DVC_EXPT_DPATH/models/fusion/Drop6-MeanYear10GSD-V2/packages/Drop6_TCombo1Year_BAS_10GSD_V2_landcover_split6_V47/Drop6_TCombo1Year_BAS_10GSD_V2_landcover_split6_V47_epoch47_step3026.pt
"



DVC_EXPT_DPATH=$(geowatch_dvc --tags='phase2_expt' --hardware='auto')
sdvc request "$DVC_EXPT_DPATH"/models/fusion/Drop6/packages/Drop6_BAS_2022_12_10GSD_BGRN_V11_CONT4/Drop6_BAS_2022_12_10GSD_BGRN_V11_CONT4_epoch6_step22939.pt

# On toothbrush - retrain BAS with fixed annotations - no teamfeat / pan (higher LR)
export CUDA_VISIBLE_DEVICES=1
DVC_DATA_DPATH=$(geowatch_dvc --tags='phase2_data' --hardware='auto')
DVC_EXPT_DPATH=$(geowatch_dvc --tags='phase2_expt' --hardware='auto')
echo "DVC_EXPT_DPATH = $DVC_EXPT_DPATH"
WORKDIR=$DVC_EXPT_DPATH/training/$HOSTNAME/$USER
DATASET_CODE=Drop7-MedianNoWinter10GSD
KWCOCO_BUNDLE_DPATH=$DVC_DATA_DPATH/$DATASET_CODE
TRAIN_FPATH=$KWCOCO_BUNDLE_DPATH/data_train_EI2LMSC_split6.kwcoco.zip
VALI_FPATH=$KWCOCO_BUNDLE_DPATH/data_vali_EI2LMSC_split6.kwcoco.zip
CHANNELS="(L8,S2):(blue|green|red|nir),(WV):(blue|green|red)"
EXPERIMENT_NAME=Drop7-MedianNoWinter10GSD_bgrn_split6_oldckpt_V72
DEFAULT_ROOT_DIR=$WORKDIR/$DATASET_CODE/runs/$EXPERIMENT_NAME
TARGET_LR=3e-4
WEIGHT_DECAY=$(python -c "print($TARGET_LR * 0.01)")
echo "WEIGHT_DECAY = $WEIGHT_DECAY"
MAX_STEPS=80000
WATCH_GRID_WORKERS=0 python -m geowatch.tasks.fusion fit --config "
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
    mask_samecolor_method  : histogram
    observable_threshold   : 0.1
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
        #saliency_weights      : '1:1'
        class_weights          : 'auto:ignore+0.000001,Unknown+0.000001,Unknown+0.000001,transient*0.1+0.000001'
        tokenizer              : linconv
        arch_name              : smt_it_stm_p32
        decoder                : mlp
        positive_change_weight : 1
        negative_change_weight : 0.01
        stream_channels        : 16
        class_loss             : 'dicefocal'
        saliency_loss          : 'dicefocal'
        saliency_head_hidden   : 6
        change_head_hidden     : 6
        class_head_hidden      : 6
        global_change_weight   : 0.01
        global_class_weight    : 0.5
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
    accumulate_grad_batches: 24
    default_root_dir     : $DEFAULT_ROOT_DIR
    accelerator          : gpu
    devices              : 0,
    limit_val_batches    : 256
    limit_train_batches  : 2048
    num_sanity_val_steps : 0
    max_epochs           : 360
    callbacks:
        - class_path: pytorch_lightning.callbacks.ModelCheckpoint
          init_args:
              monitor: val_loss
              mode: min
              save_top_k: 5
              filename: '{epoch}-{step}-{val_loss:.3f}.ckpt'
              save_last: true

torch_globals:
    float32_matmul_precision: auto

initializer:
    init: /home/joncrall/remote/toothbrush/data/dvc-repos/smart_expt_dvc/training/toothbrush/joncrall/Drop6/runs/Drop6_BAS_scratch_big_landcover_10GSD_split2_V11/lightning_logs/version_0/checkpoints/epoch=44-step=48015.ckpt
    # init: $DVC_EXPT_DPATH/models/fusion/Drop6/packages/Drop6_BAS_2022_12_10GSD_BGRN_V11_CONT4/Drop6_BAS_2022_12_10GSD_BGRN_V11_CONT4_epoch6_step22939.pt
" --ckpt_path /home/joncrall/remote/toothbrush/data/dvc-repos/smart_expt_dvc/training/toothbrush/joncrall/Drop7-MedianNoWinter10GSD/runs/Drop7-MedianNoWinter10GSD_bgrn_split6_oldckpt_V72/lightning_logs/version_1/checkpoints/epoch=14-step=1290-val_loss=5.528.ckpt.ckpt


# On toothbrush - retrain BAS with fixed annotations - no teamfeat / pan (higher LR)
export CUDA_VISIBLE_DEVICES=1
DVC_DATA_DPATH=$(geowatch_dvc --tags='phase2_data' --hardware='auto')
DVC_EXPT_DPATH=$(geowatch_dvc --tags='phase2_expt' --hardware='auto')
echo "DVC_EXPT_DPATH = $DVC_EXPT_DPATH"
WORKDIR=$DVC_EXPT_DPATH/training/$HOSTNAME/$USER
DATASET_CODE=Drop7-MedianNoWinter10GSD
KWCOCO_BUNDLE_DPATH=$DVC_DATA_DPATH/$DATASET_CODE
TRAIN_FPATH=$KWCOCO_BUNDLE_DPATH/data_train_EI2LMSC_split6.kwcoco.zip
VALI_FPATH=$KWCOCO_BUNDLE_DPATH/data_vali_EI2LMSC_split6.kwcoco.zip
CHANNELS="(L8,S2):(blue|green|red|nir),(WV):(blue|green|red)"
EXPERIMENT_NAME=Drop7-MedianNoWinter10GSD_bgrn_split6_oldckpt_V73
DEFAULT_ROOT_DIR=$WORKDIR/$DATASET_CODE/runs/$EXPERIMENT_NAME
TARGET_LR=3e-4
WEIGHT_DECAY=$(python -c "print($TARGET_LR * 0.01)")
echo "WEIGHT_DECAY = $WEIGHT_DECAY"
MAX_STEPS=80000
WATCH_GRID_WORKERS=0 python -m geowatch.tasks.fusion fit --config "
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
    mask_samecolor_method  : histogram
    observable_threshold   : 0.1
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
        #saliency_weights      : '1:1'
        class_weights          : 'auto:ignore+0.000001,Unknown+0.000001,Unknown+0.000001,transient*0.1+0.000001'
        tokenizer              : linconv
        arch_name              : smt_it_stm_p32
        decoder                : mlp
        positive_change_weight : 1
        negative_change_weight : 0.01
        stream_channels        : 16
        class_loss             : 'dicefocal'
        saliency_loss          : 'dicefocal'
        saliency_head_hidden   : 6
        change_head_hidden     : 6
        class_head_hidden      : 6
        global_change_weight   : 0.00000001
        global_class_weight    : 0.5
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
    accumulate_grad_batches: 24
    default_root_dir     : $DEFAULT_ROOT_DIR
    accelerator          : gpu
    devices              : 0,
    limit_val_batches    : 256
    limit_train_batches  : 2048
    num_sanity_val_steps : 0
    max_epochs           : 360
    callbacks:
        - class_path: pytorch_lightning.callbacks.ModelCheckpoint
          init_args:
              monitor: val_loss
              mode: min
              save_top_k: 5
              filename: '{epoch}-{step}-{val_loss:.3f}.ckpt'
              save_last: true

torch_globals:
    float32_matmul_precision: auto

initializer:
    init: /home/joncrall/remote/toothbrush/data/dvc-repos/smart_expt_dvc/training/toothbrush/joncrall/Drop7-MedianNoWinter10GSD/runs/Drop7-MedianNoWinter10GSD_bgrn_split6_oldckpt_V72/lightning_logs/version_2/checkpoints/epoch=35-step=3096-val_loss=4.938.ckpt.ckpt
"


DVC_EXPT_DPATH=$(geowatch_dvc --tags='phase2_expt' --hardware='auto')
sdvc request "$DVC_EXPT_DPATH"/models/fusion/Drop6/packages/Drop6_BAS_2022_12_10GSD_BGRN_V11_CONT4/Drop6_BAS_2022_12_10GSD_BGRN_V11_CONT4_epoch6_step22939.pt

# On toothbrush - retrain BAS with fixed annotations - no teamfeat / pan (higher LR)
export CUDA_VISIBLE_DEVICES=1
DVC_DATA_DPATH=$(geowatch_dvc --tags='phase2_data' --hardware='auto')
DVC_EXPT_DPATH=$(geowatch_dvc --tags='phase2_expt' --hardware='auto')
echo "DVC_EXPT_DPATH = $DVC_EXPT_DPATH"
WORKDIR=$DVC_EXPT_DPATH/training/$HOSTNAME/$USER
DATASET_CODE=Drop7-MedianNoWinter10GSD-iMERIT
KWCOCO_BUNDLE_DPATH=$DVC_DATA_DPATH/$DATASET_CODE
TRAIN_FPATH=$KWCOCO_BUNDLE_DPATH/data_train_rawbands_split6.kwcoco.zip
VALI_FPATH=$KWCOCO_BUNDLE_DPATH/data_vali_rawbands_split6.kwcoco.zip
CHANNELS="(L8,S2):(blue|green|red|nir),(WV):(blue|green|red)"
EXPERIMENT_NAME=Drop7-MedianNoWinter10GSD_bgrn_split6_V74
DEFAULT_ROOT_DIR=$WORKDIR/$DATASET_CODE/runs/$EXPERIMENT_NAME
TARGET_LR=3e-4
WEIGHT_DECAY=$(python -c "print($TARGET_LR * 0.01)")
echo "WEIGHT_DECAY = $WEIGHT_DECAY"
MAX_STEPS=80000
WATCH_GRID_WORKERS=0 python -m geowatch.tasks.fusion fit --config "
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
    mask_low_quality       : False
    mask_samecolor_method  : histogram
    observable_threshold   : 0.1
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
        #saliency_weights      : '1:1'
        class_weights          : 'auto'
        tokenizer              : linconv
        arch_name              : smt_it_stm_p24
        decoder                : mlp
        positive_change_weight : 1
        negative_change_weight : 0.01
        stream_channels        : 16
        class_loss             : 'dicefocal'
        saliency_loss          : 'dicefocal'
        saliency_head_hidden   : 6
        change_head_hidden     : 6
        class_head_hidden      : 6
        global_change_weight   : 0
        global_class_weight    : 0.0000000000001
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
    accumulate_grad_batches: 24
    default_root_dir     : $DEFAULT_ROOT_DIR
    accelerator          : gpu
    devices              : 0,
    limit_val_batches    : 256
    limit_train_batches  : 2048
    num_sanity_val_steps : 0
    max_epochs           : 360
    callbacks:
        - class_path: pytorch_lightning.callbacks.ModelCheckpoint
          init_args:
              monitor: val_loss
              mode: min
              save_top_k: 5
              filename: '{epoch}-{step}-{val_loss:.3f}.ckpt'
              save_last: true

torch_globals:
    float32_matmul_precision: auto

initializer:
    init: $DVC_EXPT_DPATH/models/fusion/Drop6-MeanYear10GSD-V2/packages/Drop6_TCombo1Year_BAS_10GSD_V2_landcover_split6_V47/Drop6_TCombo1Year_BAS_10GSD_V2_landcover_split6_V47_epoch47_step3026.pt
"


# On yardrat - retrain BAS with fixed annotations - no teamfeat / pan (lower LR)
export CUDA_VISIBLE_DEVICES=0
DVC_DATA_DPATH=$(geowatch_dvc --tags='phase2_data' --hardware='auto')
DVC_EXPT_DPATH=$(geowatch_dvc --tags='phase2_expt' --hardware='auto')
echo "DVC_EXPT_DPATH = $DVC_EXPT_DPATH"
WORKDIR=$DVC_EXPT_DPATH/training/$HOSTNAME/$USER
DATASET_CODE=Drop7-MedianNoWinter10GSD
KWCOCO_BUNDLE_DPATH=$DVC_DATA_DPATH/$DATASET_CODE
TRAIN_FPATH=$KWCOCO_BUNDLE_DPATH/data_train_EI2LMSC_split6.kwcoco.zip
VALI_FPATH=$KWCOCO_BUNDLE_DPATH/data_vali_EI2LMSC_split6.kwcoco.zip
CHANNELS="(L8,S2):(blue|green|red|nir),(WV):(blue|green|red)"
EXPERIMENT_NAME=Drop7-MedianNoWinter10GSD_bgrn_split6_V74
DEFAULT_ROOT_DIR=$WORKDIR/$DATASET_CODE/runs/$EXPERIMENT_NAME
TARGET_LR=1e-5
WEIGHT_DECAY=$(python -c "print($TARGET_LR * 0.01)")
echo "WEIGHT_DECAY = $WEIGHT_DECAY"
MAX_STEPS=80000
WATCH_GRID_WORKERS=0 python -m geowatch.tasks.fusion fit --config "
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
        #saliency_weights      : '1:1'
        #class_weights         : auto
        tokenizer              : linconv
        arch_name              : smt_it_stm_p16
        decoder                : mlp
        positive_change_weight : 1
        negative_change_weight : 0.01
        stream_channels        : 16
        class_loss             : 'focal'
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
    accumulate_grad_batches: 24
    default_root_dir     : $DEFAULT_ROOT_DIR
    accelerator          : gpu
    devices              : 0,
    limit_val_batches    : 256
    limit_train_batches  : 2048
    num_sanity_val_steps : 0
    max_epochs           : 360
    callbacks:
        - class_path: pytorch_lightning.callbacks.ModelCheckpoint
          init_args:
              monitor: val_loss
              mode: min
              save_top_k: 5
              filename: '{epoch}-{step}-{val_loss:.3f}.ckpt'
              save_last: true

torch_globals:
    float32_matmul_precision: auto

initializer:
    init: $DVC_EXPT_DPATH/models/fusion/Drop6-MeanYear10GSD-V2/packages/Drop6_TCombo1Year_BAS_10GSD_V2_landcover_split6_V47/Drop6_TCombo1Year_BAS_10GSD_V2_landcover_split6_V47_epoch47_step3026.pt
"


# On toothbrush - retrain BAS with fixed annotations - no teamfeat / pan (higher LR)
export CUDA_VISIBLE_DEVICES=1
DVC_DATA_DPATH=$(geowatch_dvc --tags='phase2_data' --hardware='auto')
DVC_EXPT_DPATH=$(geowatch_dvc --tags='phase2_expt' --hardware='auto')
echo "DVC_EXPT_DPATH = $DVC_EXPT_DPATH"
WORKDIR=$DVC_EXPT_DPATH/training/$HOSTNAME/$USER
DATASET_CODE=Drop7-MedianNoWinter10GSD
KWCOCO_BUNDLE_DPATH=$DVC_DATA_DPATH/$DATASET_CODE
TRAIN_FPATH=$KWCOCO_BUNDLE_DPATH/data_train_EI2LMSC_split6.kwcoco.zip
VALI_FPATH=$KWCOCO_BUNDLE_DPATH/data_vali_EI2LMSC_split6.kwcoco.zip
CHANNELS="(L8,S2):(blue|green|red|nir),(WV):(blue|green|red)"
EXPERIMENT_NAME=Drop7-MedianNoWinter10GSD_bgrn_split6_oldckpt_V75
DEFAULT_ROOT_DIR=$WORKDIR/$DATASET_CODE/runs/$EXPERIMENT_NAME
TARGET_LR=1e-4
WEIGHT_DECAY=$(python -c "print($TARGET_LR * 0.01)")
echo "WEIGHT_DECAY = $WEIGHT_DECAY"
MAX_STEPS=80000
WATCH_GRID_WORKERS=0 python -m geowatch.tasks.fusion fit --config "
data:
    select_videos          : $SELECT_VIDEOS
    num_workers            : 5
    train_dataset          : $TRAIN_FPATH
    vali_dataset           : $VALI_FPATH
    window_dims            : '164,164'
    time_steps             : 15
    time_sampling          : uniform-soft5-soft4-contiguous
    time_kernel            : '(-5y,-4y,-3y,-2.5y,-2y,-1.5y,-1y,0,1y,1.5y,2y,2.5y,3y,4y,5y)'
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
    mask_samecolor_method  : histogram
    observable_threshold   : 0.1
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
        #saliency_weights      : '1:1'
        class_weights          : 'auto'
        tokenizer              : linconv
        arch_name              : smt_it_stm_p32
        decoder                : mlp
        positive_change_weight : 1
        negative_change_weight : 0.01
        stream_channels        : 16
        class_loss             : 'dicefocal'
        saliency_loss          : 'dicefocal'
        saliency_head_hidden   : 9
        change_head_hidden     : 6
        class_head_hidden      : 6
        global_change_weight   : 0.0
        global_class_weight    : 0.00001
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
    accumulate_grad_batches: 24
    default_root_dir     : $DEFAULT_ROOT_DIR
    accelerator          : gpu
    devices              : 0,
    limit_val_batches    : 1024
    limit_train_batches  : 4096
    num_sanity_val_steps : 0
    max_epochs           : 360
    callbacks:
        - class_path: pytorch_lightning.callbacks.ModelCheckpoint
          init_args:
              monitor: val_loss
              mode: min
              save_top_k: 5
              filename: '{epoch}-{step}-{val_loss:.3f}.ckpt'
              save_last: true

torch_globals:
    float32_matmul_precision: auto

initializer:
    init: /home/joncrall/remote/toothbrush/data/dvc-repos/smart_expt_dvc/training/toothbrush/joncrall/Drop7-MedianNoWinter10GSD/runs/Drop7-MedianNoWinter10GSD_bgrn_split6_oldckpt_V73/lightning_logs/version_0/checkpoints/epoch=310-step=26746-val_loss=4.898.ckpt.ckpt
"


# On toothbrush - retrain BAS with fixed annotations - no teamfeat / pan (higher LR)
export CUDA_VISIBLE_DEVICES=1
DVC_DATA_DPATH=$(geowatch_dvc --tags='phase2_data' --hardware='ssd')
DVC_EXPT_DPATH=$(geowatch_dvc --tags='phase2_expt' --hardware='auto')
echo "DVC_EXPT_DPATH = $DVC_EXPT_DPATH"
WORKDIR=$DVC_EXPT_DPATH/training/$HOSTNAME/$USER
DATASET_CODE=Drop7-MedianNoWinter10GSD-Both
KWCOCO_BUNDLE_DPATH=$DVC_DATA_DPATH/$DATASET_CODE
TRAIN_FPATH=$KWCOCO_BUNDLE_DPATH/data_train_mixed_split6.kwcoco.zip
VALI_FPATH=$KWCOCO_BUNDLE_DPATH/data_vali_mixed_split6.kwcoco.zip
CHANNELS="(L8,S2):(blue|green|red|nir),(WV):(blue|green|red)"
EXPERIMENT_NAME=Drop7-MedianNoWinter10GSD_bgrn_mixed_split6_oldckpt_V76
DEFAULT_ROOT_DIR=$WORKDIR/$DATASET_CODE/runs/$EXPERIMENT_NAME
TARGET_LR=1e-4
WEIGHT_DECAY=$(python -c "print($TARGET_LR * 0.01)")
echo "WEIGHT_DECAY = $WEIGHT_DECAY"
MAX_STEPS=80000
WATCH_GRID_WORKERS=0 python -m geowatch.tasks.fusion fit --config "
data:
    select_videos          : $SELECT_VIDEOS
    num_workers            : 5
    train_dataset          : $TRAIN_FPATH
    vali_dataset           : $VALI_FPATH
    window_dims            : '164,164'
    time_steps             : 15
    time_sampling          : uniform-soft5-soft4-contiguous
    time_kernel            : '(-5y,-4y,-3y,-2.5y,-2y,-1.5y,-1y,0,1y,1.5y,2y,2.5y,3y,4y,5y)'
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
    mask_samecolor_method  : histogram
    observable_threshold   : 0.1
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
        #saliency_weights      : '1:1'
        class_weights          : 'auto'
        tokenizer              : linconv
        arch_name              : smt_it_stm_p32
        decoder                : mlp
        positive_change_weight : 1
        negative_change_weight : 0.01
        stream_channels        : 16
        class_loss             : 'dicefocal'
        saliency_loss          : 'dicefocal'
        saliency_head_hidden   : 9
        change_head_hidden     : 6
        class_head_hidden      : 6
        global_change_weight   : 0.0
        global_class_weight    : 0.00001
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
        accumulate_grad_batches: 24
        default_root_dir     : $DEFAULT_ROOT_DIR
        accelerator          : gpu
        devices              : 0,
        limit_val_batches    : 1024
        limit_train_batches  : 4096
        num_sanity_val_steps : 0
        max_epochs           : 360
        callbacks:
        - class_path: pytorch_lightning.callbacks.ModelCheckpoint
        init_args:
        monitor: val_loss
        mode: min
        save_top_k: 5
        filename: '{epoch}-{step}-{val_loss:.3f}.ckpt'
        save_last: true

        torch_globals:
        float32_matmul_precision: auto

        initializer:
        init: /home/joncrall/remote/toothbrush/data/dvc-repos/smart_expt_dvc/training/toothbrush/joncrall/Drop7-MedianNoWinter10GSD/runs/Drop7-MedianNoWinter10GSD_bgrn_split6_oldckpt_V75/lightning_logs/version_0/checkpoints/epoch=6-step=1197-val_loss=7.161.ckpt.ckpt
        "


# On toothbrush - retrain BAS with fixed annotations - no teamfeat / pan (higher LR)
# back to drop6 checkpoint
export CUDA_VISIBLE_DEVICES=1
DVC_DATA_DPATH=$(geowatch_dvc --tags='phase2_data' --hardware='ssd')
DVC_EXPT_DPATH=$(geowatch_dvc --tags='phase2_expt' --hardware='auto')
echo "DVC_EXPT_DPATH = $DVC_EXPT_DPATH"
WORKDIR=$DVC_EXPT_DPATH/training/$HOSTNAME/$USER
DATASET_CODE=Drop7-MedianNoWinter10GSD-NoMask
KWCOCO_BUNDLE_DPATH=$DVC_DATA_DPATH/$DATASET_CODE
TRAIN_FPATH=$KWCOCO_BUNDLE_DPATH/data_train_rawbands_split6.kwcoco.zip
VALI_FPATH=$KWCOCO_BUNDLE_DPATH/data_vali_rawbands_split6.kwcoco.zip
CHANNELS="(L8,S2):(blue|green|red|nir),(WV):(blue|green|red)"
EXPERIMENT_NAME=Drop7-MedianNoWinter10GSD_bgrn_mixed_split6_V77
DEFAULT_ROOT_DIR=$WORKDIR/$DATASET_CODE/runs/$EXPERIMENT_NAME
TARGET_LR=1e-4
WEIGHT_DECAY=$(python -c "print($TARGET_LR * 0.01)")
echo "WEIGHT_DECAY = $WEIGHT_DECAY"
MAX_STEPS=80000
test -d "$DVC_EXPT_DPATH" || echo "missing DVC_EXPT_DPATH"
test -d "$DVC_DATA_DPATH" || echo "missing DVC_DATA_DPATH"
test -f "$TRAIN_FPATH" || echo "missing TRAIN_FPATH"
test -f "$VALI_FPATH" || echo "missing VALI_FPATH"
test -d "$KWCOCO_BUNDLE_DPATH" || echo "missing KWCOCO_BUNDLE_DPATH"

kwcoco stats "$TRAIN_FPATH" "$VALI_FPATH"
echo "TRAIN_FPATH = $TRAIN_FPATH"
#kwcoco stats "$TRAIN_FPATH"
#kwcoco stats "$VALI_FPATH"
WATCH_GRID_WORKERS=0 python -m geowatch.tasks.fusion fit --config "
data:
select_videos          : $SELECT_VIDEOS
num_workers            : 5
train_dataset          : $TRAIN_FPATH
vali_dataset           : $VALI_FPATH
window_dims            : '164,164'
time_steps             : 15
    time_sampling          : uniform-soft5-soft4-contiguous
    time_kernel            : '(-5y,-4y,-3y,-2.5y,-2y,-1.5y,-1y,0,1y,1.5y,2y,2.5y,3y,4y,5y)'
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
    mask_low_quality       : False
    mask_samecolor_method  : histogram
    observable_threshold   : 0.1
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
        # saliency_weights      : '1:1'
        class_weights          : 'auto'
        tokenizer              : linconv
        arch_name              : smt_it_stm_p32
        decoder                : mlp
        positive_change_weight : 1
        negative_change_weight : 0.01
        stream_channels        : 16
        class_loss             : 'dicefocal'
        saliency_loss          : 'dicefocal'
        saliency_head_hidden   : 9
        change_head_hidden     : 6
        class_head_hidden      : 6
        global_change_weight   : 1.0
        global_class_weight    : 1.0
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
    accumulate_grad_batches: 24
    default_root_dir     : $DEFAULT_ROOT_DIR
    accelerator          : gpu
    devices              : 0,
    limit_val_batches    : 1024
    limit_train_batches  : 4096
    num_sanity_val_steps : 0
    max_epochs           : 360
    callbacks:
        - class_path: pytorch_lightning.callbacks.ModelCheckpoint
          init_args:
              monitor: val_loss
              mode: min
              save_top_k: 5
              filename: '{epoch}-{step}-{val_loss:.3f}.ckpt'
              save_last: true

torch_globals:
    float32_matmul_precision: auto

initializer:
    init: $DVC_EXPT_DPATH/models/fusion/Drop6-MeanYear10GSD-V2/packages/Drop6_TCombo1Year_BAS_10GSD_V2_landcover_split6_V47/Drop6_TCombo1Year_BAS_10GSD_V2_landcover_split6_V47_epoch47_step3026.pt
"

# back to drop6 checkpoint
export CUDA_VISIBLE_DEVICES=1
DVC_DATA_DPATH=$(geowatch_dvc --tags='phase2_data' --hardware='ssd')
DVC_EXPT_DPATH=$(geowatch_dvc --tags='phase2_expt' --hardware='auto')
echo "DVC_EXPT_DPATH = $DVC_EXPT_DPATH"
WORKDIR=$DVC_EXPT_DPATH/training/$HOSTNAME/$USER
DATASET_CODE=Drop7-MedianNoWinter10GSD-NoMask
KWCOCO_BUNDLE_DPATH=$DVC_DATA_DPATH/$DATASET_CODE
TRAIN_FPATH=$KWCOCO_BUNDLE_DPATH/data_train_rawbands_split6.kwcoco.zip
VALI_FPATH=$KWCOCO_BUNDLE_DPATH/data_vali_rawbands_split6.kwcoco.zip
CHANNELS="(L8,S2):(blue|green|red|nir),(WV):(blue|green|red)"
EXPERIMENT_NAME=Drop7-MedianNoWinter10GSD_bgrn_mixed_split6_V78
DEFAULT_ROOT_DIR=$WORKDIR/$DATASET_CODE/runs/$EXPERIMENT_NAME
TARGET_LR=3e-3
WEIGHT_DECAY=$(python -c "print($TARGET_LR * 0.01)")
echo "WEIGHT_DECAY = $WEIGHT_DECAY"
MAX_STEPS=80000
test -d "$DVC_EXPT_DPATH" || echo "missing DVC_EXPT_DPATH"
test -d "$DVC_DATA_DPATH" || echo "missing DVC_DATA_DPATH"
test -f "$TRAIN_FPATH" || echo "missing TRAIN_FPATH"
test -f "$VALI_FPATH" || echo "missing VALI_FPATH"
test -d "$KWCOCO_BUNDLE_DPATH" || echo "missing KWCOCO_BUNDLE_DPATH"
#kwcoco stats "$TRAIN_FPATH" "$VALI_FPATH"
#echo "TRAIN_FPATH = $TRAIN_FPATH"
#kwcoco stats "$TRAIN_FPATH"
#kwcoco stats "$VALI_FPATH"
WATCH_GRID_WORKERS=0 python -m geowatch.tasks.fusion fit --config "
data:
    select_videos          : $SELECT_VIDEOS
    num_workers            : 5
    train_dataset          : $TRAIN_FPATH
    vali_dataset           : $VALI_FPATH
    window_dims            : '164,164'
    time_steps             : 15
    time_sampling          : uniform-soft5-soft4-contiguous
    time_kernel            : '(-5y,-4y,-3y,-2.5y,-2y,-1.5y,-1y,0,1y,1.5y,2y,2.5y,3y,4y,5y)'
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
    mask_low_quality       : False
    mask_samecolor_method  : histogram
    observable_threshold   : 0.1
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
        # saliency_weights      : '1:1'
        class_weights          : 'auto'
        tokenizer              : linconv
        arch_name              : smt_it_stm_p32
        decoder                : mlp
        positive_change_weight : 1
        negative_change_weight : 0.01
        stream_channels        : 16
        class_loss             : 'dicefocal'
        saliency_loss          : 'dicefocal'
        saliency_head_hidden   : 9
        change_head_hidden     : 6
        class_head_hidden      : 6
        global_change_weight   : 1.0
        global_class_weight    : 1.0
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
    accumulate_grad_batches: 256
    default_root_dir     : $DEFAULT_ROOT_DIR
    accelerator          : gpu
    devices              : 0,
    limit_val_batches    : 1024
    limit_train_batches  : 4096
    num_sanity_val_steps : 0
    max_epochs           : 360
    log_every_n_steps    : 50
    callbacks:
        - class_path: pytorch_lightning.callbacks.ModelCheckpoint
          init_args:
              monitor: val_loss
              mode: min
              save_top_k: 5
              filename: '{epoch:03d}-{step:06d}-{val_loss:.3f}.ckpt'
              save_last: true

torch_globals:
    float32_matmul_precision: auto

initializer:
    init: $DVC_EXPT_DPATH/models/fusion/Drop7-MedianNoWinter10GSD-NoMask/packages/Drop7-MedianNoWinter10GSD_bgrn_mixed_split6_V77/Drop7-MedianNoWinter10GSD_bgrn_mixed_split6_V77_epoch16_step2907.pt
"


export CUDA_VISIBLE_DEVICES=1
DVC_DATA_DPATH=$(geowatch_dvc --tags='phase2_data' --hardware='ssd')
DVC_EXPT_DPATH=$(geowatch_dvc --tags='phase2_expt' --hardware='auto')
echo "DVC_EXPT_DPATH = $DVC_EXPT_DPATH"
WORKDIR=$DVC_EXPT_DPATH/training/$HOSTNAME/$USER
DATASET_CODE=Drop7-MedianNoWinter10GSD-NoMask
KWCOCO_BUNDLE_DPATH=$DVC_DATA_DPATH/$DATASET_CODE
TRAIN_FPATH=$KWCOCO_BUNDLE_DPATH/data_train_rawbands_split6.kwcoco.zip
VALI_FPATH=$KWCOCO_BUNDLE_DPATH/data_vali_rawbands_split6.kwcoco.zip
CHANNELS="(L8,S2):(blue|green|red|nir),(WV):(blue|green|red)"
EXPERIMENT_NAME=Drop7-MedianNoWinter10GSD_bgrn_mixed_split6_V79
DEFAULT_ROOT_DIR=$WORKDIR/$DATASET_CODE/runs/$EXPERIMENT_NAME
TARGET_LR=3e-3
WEIGHT_DECAY=$(python -c "print($TARGET_LR * 0.01)")
echo "WEIGHT_DECAY = $WEIGHT_DECAY"
MAX_STEPS=80000
test -d "$DVC_EXPT_DPATH" || echo "missing DVC_EXPT_DPATH"
test -d "$DVC_DATA_DPATH" || echo "missing DVC_DATA_DPATH"
test -f "$TRAIN_FPATH" || echo "missing TRAIN_FPATH"
test -f "$VALI_FPATH" || echo "missing VALI_FPATH"
test -d "$KWCOCO_BUNDLE_DPATH" || echo "missing KWCOCO_BUNDLE_DPATH"
#kwcoco stats "$TRAIN_FPATH" "$VALI_FPATH"
#echo "TRAIN_FPATH = $TRAIN_FPATH"
#kwcoco stats "$TRAIN_FPATH"
#kwcoco stats "$VALI_FPATH"
WATCH_GRID_WORKERS=0 python -m geowatch.tasks.fusion fit --config "
data:
    select_videos          : $SELECT_VIDEOS
    num_workers            : 5
    train_dataset          : $TRAIN_FPATH
    vali_dataset           : $VALI_FPATH
    window_dims            : '164,164'
    time_steps             : 15
    time_sampling          : uniform-soft5-soft4-contiguous
    time_kernel            : '(-5y,-4y,-3y,-2.5y,-2y,-1.5y,-1y,0,1y,1.5y,2y,2.5y,3y,4y,5y)'
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
    mask_low_quality       : False
    mask_samecolor_method  : histogram
    observable_threshold   : 0.1
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
        # saliency_weights      : '1:1'
        class_weights          : 'auto'
        tokenizer              : linconv
        arch_name              : smt_it_stm_p32
        decoder                : mlp
        positive_change_weight : 1
        negative_change_weight : 0.01
        stream_channels        : 16
        class_loss             : 'dicefocal'
        saliency_loss          : 'dicefocal'
        saliency_head_hidden   : 9
        change_head_hidden     : 6
        class_head_hidden      : 6
        global_change_weight   : 0.0
        global_class_weight    : 1.0
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
    accumulate_grad_batches: 256
    default_root_dir     : $DEFAULT_ROOT_DIR
    accelerator          : gpu
    devices              : 0,
    limit_val_batches    : 1024
    limit_train_batches  : 4096
    num_sanity_val_steps : 0
    max_epochs           : 360
    log_every_n_steps    : 1
    callbacks:
        - class_path: pytorch_lightning.callbacks.ModelCheckpoint
          init_args:
              monitor: val_loss
              mode: min
              save_top_k: 5
              filename: '{epoch:03d}-{step:06d}-{val_loss:.3f}.ckpt'
              save_last: true

torch_globals:
    float32_matmul_precision: auto

initializer:
    init: $DVC_EXPT_DPATH/models/fusion/Drop7-MedianNoWinter10GSD-NoMask/packages/Drop7-MedianNoWinter10GSD_bgrn_mixed_split6_V78/Drop7-MedianNoWinter10GSD_bgrn_mixed_split6_V78_epoch7_step126.pt
"


# Toothbrush SC with Shrink and Perterb
export CUDA_VISIBLE_DEVICES=1
DVC_DATA_DPATH=$(geowatch_dvc --tags='drop7_data' --hardware='auto')
DVC_EXPT_DPATH=$(geowatch_dvc --tags='phase2_expt' --hardware='auto')
echo "DVC_DATA_DPATH = $DVC_DATA_DPATH"
echo "DVC_EXPT_DPATH = $DVC_EXPT_DPATH"
WORKDIR=$DVC_EXPT_DPATH/training/$HOSTNAME/$USER
DATASET_CODE=Drop7-Cropped2GSD
KWCOCO_BUNDLE_DPATH=$DVC_DATA_DPATH/$DATASET_CODE
TRAIN_FPATH=$KWCOCO_BUNDLE_DPATH/data_train_rawbands_split6.kwcoco.zip
VALI_FPATH=$KWCOCO_BUNDLE_DPATH/data_vali_rawbands_split6.kwcoco.zip
CHANNELS="(L8,S2):(blue|green|red|nir),(WV):(blue|green|red)"
EXPERIMENT_NAME=Drop7-Cropped2GSD_SC_bgrn_snp_split6_V80
DEFAULT_ROOT_DIR=$WORKDIR/$DATASET_CODE/runs/$EXPERIMENT_NAME
TARGET_LR=1e-4
WEIGHT_DECAY=$(python -c "print($TARGET_LR * 0.01)")
echo "WEIGHT_DECAY = $WEIGHT_DECAY"
MAX_STEPS=80000
WATCH_GRID_WORKERS=0 python -m geowatch.tasks.fusion fit --config "
data:
    select_videos          : $SELECT_VIDEOS
    num_workers            : 5
    train_dataset          : $TRAIN_FPATH
    vali_dataset           : $VALI_FPATH
    window_dims            : '196,196'
    time_steps             : 9
    time_sampling          : uniform-soft5-soft4-contiguous
    time_kernel            : '(-1.0y,-0.5y,-0.25y,-0.08y,0.0y,0.08y,0.25y,0.5y,1.0y)'
    window_resolution     : 2.0GSD
    input_resolution      : 2.0GSD
    output_resolution     : 2.0GSD
    neg_to_pos_ratio       : 1.0
    batch_size             : 2
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
    use_grid_positives     : False
    use_grid_negatives     : False
    normalize_inputs       : 1024
    balance_areas          : True
model:
    class_path: MultimodalTransformer
    init_args:
        #saliency_weights      : '1:1'
        #class_weights         : auto
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
        global_class_weight    : 1.00
        global_saliency_weight : 0.01
        multimodal_reduce      : learned_linear
        perterb_scale          : 0.00001
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
    accumulate_grad_batches: 24
    default_root_dir     : $DEFAULT_ROOT_DIR
    accelerator          : gpu
    devices              : 0,
    limit_val_batches    : 256
    limit_train_batches  : 2048
    num_sanity_val_steps : 0
    max_epochs           : 360
    callbacks:
        - class_path: pytorch_lightning.callbacks.ModelCheckpoint
          init_args:
              monitor: val_loss
              mode: min
              save_top_k: 5
              filename: '{epoch}-{step}-{val_loss:.3f}.ckpt'
              save_last: true

torch_globals:
    float32_matmul_precision: auto

initializer:
    init: $DVC_EXPT_DPATH/models/fusion/Drop4-SC/packages/Drop4_tune_V30_8GSD_V3/Drop4_tune_V30_8GSD_V3_epoch=2-step=17334.pt.pt
"



# Toothbrush SC with Generate and Test
export CUDA_VISIBLE_DEVICES=0
DVC_DATA_DPATH=$(geowatch_dvc --tags='drop7_data' --hardware='auto')
DVC_EXPT_DPATH=$(geowatch_dvc --tags='phase2_expt' --hardware='auto')
echo "DVC_DATA_DPATH = $DVC_DATA_DPATH"
echo "DVC_EXPT_DPATH = $DVC_EXPT_DPATH"
WORKDIR=$DVC_EXPT_DPATH/training/$HOSTNAME/$USER
DATASET_CODE=Drop7-Cropped2GSD
KWCOCO_BUNDLE_DPATH=$DVC_DATA_DPATH/$DATASET_CODE
TRAIN_FPATH=$KWCOCO_BUNDLE_DPATH/data_train_rawbands_split6.kwcoco.zip
VALI_FPATH=$KWCOCO_BUNDLE_DPATH/data_vali_rawbands_split6.kwcoco.zip
CHANNELS="(L8,S2):(blue|green|red|nir),(WV):(blue|green|red)"
EXPERIMENT_NAME=Drop7-Cropped2GSD_SC_bgrn_gnt_split6_V81
DEFAULT_ROOT_DIR=$WORKDIR/$DATASET_CODE/runs/$EXPERIMENT_NAME
TARGET_LR=1e-4
WEIGHT_DECAY=$(python -c "print($TARGET_LR * 0.01)")
echo "WEIGHT_DECAY = $WEIGHT_DECAY"
MAX_STEPS=80000
WATCH_GRID_WORKERS=0 python -m geowatch.tasks.fusion fit --config "
data:
    select_videos          : $SELECT_VIDEOS
    num_workers            : 5
    train_dataset          : $TRAIN_FPATH
    vali_dataset           : $VALI_FPATH
    window_dims            : '196,196'
    time_steps             : 9
    time_sampling          : uniform-soft5-soft4-contiguous
    time_kernel            : '(-1.0y,-0.5y,-0.25y,-0.08y,0.0y,0.08y,0.25y,0.5y,1.0y)'
    window_resolution     : 2.0GSD
    input_resolution      : 2.0GSD
    output_resolution     : 2.0GSD
    neg_to_pos_ratio       : 1.0
    batch_size             : 2
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
    use_grid_positives     : False
    use_grid_negatives     : False
    normalize_inputs       : 1024
    balance_areas          : True
model:
    class_path: MultimodalTransformer
    init_args:
        #saliency_weights      : '1:1'
        #class_weights         : auto
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
        global_class_weight    : 1.00
        global_saliency_weight : 0.01
        multimodal_reduce      : learned_linear
        continual_learning     : true
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
    accumulate_grad_batches: 24
    default_root_dir     : $DEFAULT_ROOT_DIR
    accelerator          : gpu
    devices              : 0,
    limit_val_batches    : 256
    limit_train_batches  : 2048
    num_sanity_val_steps : 0
    max_epochs           : 360
    callbacks:
        - class_path: pytorch_lightning.callbacks.ModelCheckpoint
          init_args:
              monitor: val_loss
              mode: min
              save_top_k: 5
              filename: '{epoch}-{step}-{val_loss:.3f}.ckpt'
              save_last: true

torch_globals:
    float32_matmul_precision: auto

initializer:
    init: $DVC_EXPT_DPATH/models/fusion/Drop4-SC/packages/Drop4_tune_V30_8GSD_V3/Drop4_tune_V30_8GSD_V3_epoch=2-step=17334.pt.pt
"


# Horologic SC with Shrink and Perterb
#export CUDA_VISIBLE_DEVICES="0,1,2,3"
export CUDA_VISIBLE_DEVICES="1"
DVC_DATA_DPATH=$(geowatch_dvc --tags='drop7_data' --hardware='ssd')
DVC_EXPT_DPATH=$(geowatch_dvc --tags='phase2_expt' --hardware='auto')
echo "DVC_DATA_DPATH = $DVC_DATA_DPATH"
echo "DVC_EXPT_DPATH = $DVC_EXPT_DPATH"
WORKDIR=$DVC_EXPT_DPATH/training/$HOSTNAME/$USER
DATASET_CODE=Drop7-Cropped2GSD
KWCOCO_BUNDLE_DPATH=$DVC_DATA_DPATH/$DATASET_CODE
TRAIN_FPATH=$KWCOCO_BUNDLE_DPATH/data_train_rawbands_split6.kwcoco.zip
VALI_FPATH=$KWCOCO_BUNDLE_DPATH/data_vali_rawbands_split6.kwcoco.zip
CHANNELS="(L8,S2):(blue|green|red|nir),(WV):(blue|green|red)"
EXPERIMENT_NAME=Drop7-Cropped2GSD_SC_bgrn_bline_split6_V82
DEFAULT_ROOT_DIR=$WORKDIR/$DATASET_CODE/runs/$EXPERIMENT_NAME
TARGET_LR=1e-4
WEIGHT_DECAY=$(python -c "print($TARGET_LR * 0.01)")
echo "WEIGHT_DECAY = $WEIGHT_DECAY"
MAX_STEPS=80000
DDP_WORKAROUND=1 WATCH_GRID_WORKERS=0 python -m geowatch.tasks.fusion fit --config "
data:
    select_videos          : $SELECT_VIDEOS
    num_workers            : 5
    train_dataset          : $TRAIN_FPATH
    vali_dataset           : $VALI_FPATH
    window_dims            : '196,196'
    time_steps             : 9
    time_sampling          : uniform-soft5-soft4-contiguous
    time_kernel            : '(-1.0y,-0.5y,-0.25y,-0.08y,0.0y,0.08y,0.25y,0.5y,1.0y)'
    window_resolution     : 2.0GSD
    input_resolution      : 2.0GSD
    output_resolution     : 2.0GSD
    neg_to_pos_ratio       : 1.0
    batch_size             : 8
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
    use_grid_positives     : False
    use_grid_negatives     : False
    normalize_inputs       : 1024
    balance_areas          : True
model:
    class_path: MultimodalTransformer
    init_args:
        #saliency_weights      : '1:1'
        #class_weights         : auto
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
        global_class_weight    : 1.00
        global_saliency_weight : 0.01
        multimodal_reduce      : learned_linear
        perterb_scale          : 0
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
    accumulate_grad_batches: 24
    default_root_dir     : $DEFAULT_ROOT_DIR
    accelerator          : gpu
    devices              : 0,
    #devices              : 0,1,2,3
    #strategy             : ddp
    limit_val_batches    : 256
    limit_train_batches  : 2048
    num_sanity_val_steps : 0
    max_epochs           : 360
    callbacks:
        - class_path: pytorch_lightning.callbacks.ModelCheckpoint
          init_args:
              monitor: val_loss
              mode: min
              save_top_k: 5
              filename: '{epoch}-{step}-{val_loss:.3f}.ckpt'
              save_last: true

torch_globals:
    float32_matmul_precision: auto

initializer:
    init: $DVC_EXPT_DPATH/models/fusion/Drop4-SC/packages/Drop4_tune_V30_8GSD_V3/Drop4_tune_V30_8GSD_V3_epoch=2-step=17334.pt.pt
"


# Horologic SC with Shrink and Perterb MultiGPU?
export CUDA_VISIBLE_DEVICES="2,3"
DVC_DATA_DPATH=$(geowatch_dvc --tags='drop7_data' --hardware='ssd')
DVC_EXPT_DPATH=$(geowatch_dvc --tags='phase2_expt' --hardware='auto')
echo "DVC_DATA_DPATH = $DVC_DATA_DPATH"
echo "DVC_EXPT_DPATH = $DVC_EXPT_DPATH"
WORKDIR=$DVC_EXPT_DPATH/training/$HOSTNAME/$USER
DATASET_CODE=Drop7-Cropped2GSD
KWCOCO_BUNDLE_DPATH=$DVC_DATA_DPATH/$DATASET_CODE
TRAIN_FPATH=$KWCOCO_BUNDLE_DPATH/data_train_rawbands_split6.kwcoco.zip
VALI_FPATH=$KWCOCO_BUNDLE_DPATH/data_vali_rawbands_split6.kwcoco.zip
CHANNELS="(L8,S2):(blue|green|red|nir),(WV):(blue|green|red)"
EXPERIMENT_NAME=Drop7-Cropped2GSD_SC_bgrn_bline_split6_V83
DEFAULT_ROOT_DIR=$WORKDIR/$DATASET_CODE/runs/$EXPERIMENT_NAME
TARGET_LR=1e-4
WEIGHT_DECAY=$(python -c "print($TARGET_LR * 0.01)")
echo "WEIGHT_DECAY = $WEIGHT_DECAY"
MAX_STEPS=80000
DDP_WORKAROUND=1 WATCH_GRID_WORKERS=0 python -m geowatch.tasks.fusion fit --config "
data:
    select_videos          : $SELECT_VIDEOS
    num_workers            : 5
    train_dataset          : $TRAIN_FPATH
    vali_dataset           : $VALI_FPATH
    window_dims            : '196,196'
    time_steps             : 9
    time_sampling          : uniform-soft5-soft4-contiguous
    time_kernel            : '(-1.0y,-0.5y,-0.25y,-0.08y,0.0y,0.08y,0.25y,0.5y,1.0y)'
    window_resolution     : 2.0GSD
    input_resolution      : 2.0GSD
    output_resolution     : 2.0GSD
    neg_to_pos_ratio       : 1.0
    batch_size             : 8
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
    use_grid_positives     : False
    use_grid_negatives     : False
    normalize_inputs       : 1024
    balance_areas          : True
model:
    class_path: MultimodalTransformer
    init_args:
        #saliency_weights      : '1:1'
        #class_weights         : auto
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
        global_class_weight    : 1.00
        global_saliency_weight : 0.01
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
    accumulate_grad_batches: 24
    default_root_dir     : $DEFAULT_ROOT_DIR
    accelerator          : gpu
    devices              : 0,1
    strategy             : ddp
    limit_val_batches    : 256
    limit_train_batches  : 2048
    num_sanity_val_steps : 0
    max_epochs           : 360
    callbacks:
        - class_path: pytorch_lightning.callbacks.ModelCheckpoint
          init_args:
              monitor: val_loss
              mode: min
              save_top_k: 5
              filename: '{epoch}-{step}-{val_loss:.3f}.ckpt'
              save_last: true

torch_globals:
    float32_matmul_precision: auto

initializer:
    init: $DVC_EXPT_DPATH/models/fusion/Drop4-SC/packages/Drop4_tune_V30_8GSD_V3/Drop4_tune_V30_8GSD_V3_epoch=2-step=17334.pt.pt
"


export CUDA_VISIBLE_DEVICES=1
DVC_DATA_DPATH=$(geowatch_dvc --tags='drop7_data' --hardware='auto')
DVC_EXPT_DPATH=$(geowatch_dvc --tags='phase2_expt' --hardware='auto')
echo "DVC_DATA_DPATH = $DVC_DATA_DPATH"
echo "DVC_EXPT_DPATH = $DVC_EXPT_DPATH"
WORKDIR=$DVC_EXPT_DPATH/training/$HOSTNAME/$USER
DATASET_CODE=Drop7-Cropped2GSD
KWCOCO_BUNDLE_DPATH=$DVC_DATA_DPATH/$DATASET_CODE
TRAIN_FPATH=$KWCOCO_BUNDLE_DPATH/data_train_rawbands_split6.kwcoco.zip
VALI_FPATH=$KWCOCO_BUNDLE_DPATH/data_vali_rawbands_split6.kwcoco.zip
CHANNELS="(L8,S2):(blue|green|red|nir),(WV):(blue|green|red)"
EXPERIMENT_NAME=Drop7-Cropped2GSD_SC_bgrn_gnt_split6_V84
DEFAULT_ROOT_DIR=$WORKDIR/$DATASET_CODE/runs/$EXPERIMENT_NAME
TARGET_LR=5e-5
WEIGHT_DECAY=$(python -c "print($TARGET_LR * 0.01)")
echo "WEIGHT_DECAY = $WEIGHT_DECAY"
MAX_STEPS=80000
DDP_WORKAROUND=0 WATCH_GRID_WORKERS=0 python -m geowatch.tasks.fusion fit --config "
data:
    select_videos          : $SELECT_VIDEOS
    num_workers            : 5
    train_dataset          : $TRAIN_FPATH
    vali_dataset           : $VALI_FPATH
    window_dims            : '196,196'
    time_steps             : 9
    time_sampling          : uniform-soft5-soft4-contiguous
    time_kernel            : '(-1.0y,-0.5y,-0.25y,-0.08y,0.0y,0.08y,0.25y,0.5y,1.0y)'
    window_resolution     : 2.0GSD
    input_resolution      : 2.0GSD
    output_resolution     : 2.0GSD
    neg_to_pos_ratio       : 1.0
    batch_size             : 4
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
    use_grid_positives     : False
    use_grid_negatives     : False
    normalize_inputs       : 1024
    balance_areas          : True
model:
    class_path: MultimodalTransformer
    init_args:
        #saliency_weights      : '1:1'
        #class_weights         : auto
        class_weights          : 'auto'
        tokenizer              : linconv
        arch_name              : smt_it_stm_p24
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
        global_class_weight    : 1.00
        global_saliency_weight : 0.05
        multimodal_reduce      : learned_linear
        continual_learning     : true
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
    accumulate_grad_batches: 6
    default_root_dir     : $DEFAULT_ROOT_DIR
    accelerator          : gpu
    devices              : 0,
    #devices              : 0,1
    #strategy             : ddp
    limit_val_batches    : 256
    limit_train_batches  : 2048
    num_sanity_val_steps : 0
    max_epochs           : 360
    callbacks:
        - class_path: pytorch_lightning.callbacks.ModelCheckpoint
          init_args:
              monitor: val_loss
              mode: min
              save_top_k: 5
              filename: '{epoch}-{step}-{val_loss:.3f}.ckpt'
              save_last: true

torch_globals:
    float32_matmul_precision: auto

initializer:
    #init: /home/joncrall/quicklinks/toothbrush_training/Drop7-Cropped2GSD/runs/Drop7-Cropped2GSD_SC_bgrn_gnt_split6_V81/lightning_logs/version_19/checkpoints/epoch=186-step=16082-val_loss=1.418.ckpt.pt
    #init: /home/joncrall/data/dvc-repos/smart_expt_dvc/training/toothbrush/joncrall/Drop7-Cropped2GSD/runs/Drop7-Cropped2GSD_SC_bgrn_gnt_split6_V84/lightning_logs/version_1/checkpoints/epoch=44-step=3870-val_loss=5.761_weight_hacked.ckpt
    init: /home/joncrall/remote/toothbrush/data/dvc-repos/smart_expt_dvc/training/toothbrush/joncrall/Drop7-Cropped2GSD/runs/Drop7-Cropped2GSD_SC_bgrn_gnt_split6_V84/lightning_logs/version_3/checkpoints/last.pt
"


export CUDA_VISIBLE_DEVICES=1
DVC_DATA_DPATH=$(geowatch_dvc --tags='drop7_data' --hardware='auto')
DVC_EXPT_DPATH=$(geowatch_dvc --tags='phase2_expt' --hardware='auto')
echo "DVC_DATA_DPATH = $DVC_DATA_DPATH"
echo "DVC_EXPT_DPATH = $DVC_EXPT_DPATH"
WORKDIR=$DVC_EXPT_DPATH/training/$HOSTNAME/$USER
DATASET_CODE=Drop7-Cropped2GSD
KWCOCO_BUNDLE_DPATH=$DVC_DATA_DPATH/$DATASET_CODE
TRAIN_FPATH=$KWCOCO_BUNDLE_DPATH/data_train_rawbands_split6.kwcoco.zip
VALI_FPATH=$KWCOCO_BUNDLE_DPATH/data_vali_rawbands_split6.kwcoco.zip
CHANNELS="(L8,S2):(blue|green|red|nir),(WV):(blue|green|red)"
EXPERIMENT_NAME=Drop7-Cropped2GSD_SC_bgrn_gnt_snp_split6_V85
DEFAULT_ROOT_DIR=$WORKDIR/$DATASET_CODE/runs/$EXPERIMENT_NAME
TARGET_LR=3e-4
WEIGHT_DECAY=$(python -c "print($TARGET_LR * 0.01)")
echo "WEIGHT_DECAY = $WEIGHT_DECAY"
MAX_STEPS=80000
DDP_WORKAROUND=0 WATCH_GRID_WORKERS=0 python -m geowatch.tasks.fusion fit --config "
data:
    select_videos          : $SELECT_VIDEOS
    num_workers            : 5
    train_dataset          : $TRAIN_FPATH
    vali_dataset           : $VALI_FPATH
    window_dims            : '196,196'
    time_steps             : 11
    time_sampling          : uniform-soft5-soft4-contiguous
    time_kernel            : '(-1.0y,-0.9y,-0.5y,-0.25y,-0.08y,0.0y,0.08y,0.25y,0.5y,0.9y,1.0y)'
    window_resolution     : 2.0GSD
    input_resolution      : 2.0GSD
    output_resolution     : 2.0GSD
    neg_to_pos_ratio       : 1.0
    batch_size             : 4
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
    use_grid_positives     : False
    use_grid_negatives     : False
    normalize_inputs       : 1024
    balance_areas          : True
model:
    class_path: MultimodalTransformer
    init_args:
        #saliency_weights      : '1:1'
        #class_weights         : auto
        class_weights          : 'auto'
        tokenizer              : linconv
        arch_name              : smt_it_stm_p24
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
        global_class_weight    : 1.00
        global_saliency_weight : 0.05
        multimodal_reduce      : learned_linear
        continual_learning     : true
        perterb_scale          : 0.000001
optimizer:
    class_path: torch.optim.AdamW
    init_args:
        lr           : $TARGET_LR
        weight_decay : $WEIGHT_DECAY
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
    callbacks:
        - class_path: pytorch_lightning.callbacks.ModelCheckpoint
          init_args:
              monitor: val_loss
              mode: min
              save_top_k: 5
              filename: '{epoch}-{step}-{val_loss:.3f}.ckpt'
              save_last: true

torch_globals:
    float32_matmul_precision: auto

initializer:
    init: /home/joncrall/remote/toothbrush/data/dvc-repos/smart_expt_dvc/training/toothbrush/joncrall/Drop7-Cropped2GSD/runs/Drop7-Cropped2GSD_SC_bgrn_gnt_split6_V84/lightning_logs/version_9/checkpoints/epoch=91-step=31464-val_loss=2.286.ckpt.pt
"


export CUDA_VISIBLE_DEVICES=0,1
DVC_DATA_DPATH=$(geowatch_dvc --tags='drop7_data' --hardware='auto')
DVC_EXPT_DPATH=$(geowatch_dvc --tags='phase2_expt' --hardware='auto')
echo "DVC_DATA_DPATH = $DVC_DATA_DPATH"
echo "DVC_EXPT_DPATH = $DVC_EXPT_DPATH"
WORKDIR=$DVC_EXPT_DPATH/training/$HOSTNAME/$USER
DATASET_CODE=Drop7-Cropped2GSD
KWCOCO_BUNDLE_DPATH=$DVC_DATA_DPATH/$DATASET_CODE
TRAIN_FPATH=$KWCOCO_BUNDLE_DPATH/data_train_rawbands_split6.kwcoco.zip
VALI_FPATH=$KWCOCO_BUNDLE_DPATH/data_vali_rawbands_split6.kwcoco.zip
CHANNELS="(L8,S2):(blue|green|red|nir),(WV):(blue|green|red)"
EXPERIMENT_NAME=Drop7-Cropped2GSD_SC_bgrn_snp_sgd_split6_V86
DEFAULT_ROOT_DIR=$WORKDIR/$DATASET_CODE/runs/$EXPERIMENT_NAME
TARGET_LR=1e-4
WEIGHT_DECAY=$(python -c "print($TARGET_LR * 0.01)")
echo "WEIGHT_DECAY = $WEIGHT_DECAY"
MAX_STEPS=80000
DDP_WORKAROUND=1 python -m geowatch.tasks.fusion fit --config "
data:
    select_videos          : $SELECT_VIDEOS
    num_workers            : 5
    train_dataset          : $TRAIN_FPATH
    vali_dataset           : $VALI_FPATH
    window_dims            : '196,196'
    time_steps             : 11
    time_sampling          : uniform-soft5-soft4-contiguous
    time_kernel            : '(-1.0y,-0.9y,-0.5y,-0.25y,-0.08y,0.0y,0.08y,0.25y,0.5y,0.9y,1.0y)'
    window_resolution     : 2.0GSD
    input_resolution      : 2.0GSD
    output_resolution     : 2.0GSD
    neg_to_pos_ratio       : 1.0
    batch_size             : 4
    normalize_perframe     : false
    normalize_peritem      : 'blue|green|red|nir|pan'
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
    use_grid_positives     : False
    use_grid_negatives     : False
    normalize_inputs       : 1024
    balance_areas          : True
model:
    class_path: MultimodalTransformer
    init_args:
        #saliency_weights      : '1:1'
        #class_weights         : auto
        class_weights          : 'auto'
        tokenizer              : linconv
        arch_name              : smt_it_stm_p24
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
        global_class_weight    : 1.00
        global_saliency_weight : 1.00
        multimodal_reduce      : learned_linear
        continual_learning     : true
        perterb_scale          : 1e-7
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
    pct_start: 0.90
trainer:
    accumulate_grad_batches: 32
    default_root_dir     : $DEFAULT_ROOT_DIR
    accelerator          : gpu
    #devices              : 0,
    devices              : 0,1
    strategy             : ddp_find_unused_parameters_true
    limit_val_batches    : 256
    limit_train_batches  : 2048
    num_sanity_val_steps : 0
    max_epochs           : 360
    callbacks:
        - class_path: pytorch_lightning.callbacks.ModelCheckpoint
          init_args:
              monitor: val_loss
              mode: min
              save_top_k: 5
              filename: '{epoch}-{step}-{val_loss:.3f}.ckpt'
              save_last: true

torch_globals:
    float32_matmul_precision: auto

initializer:
    # init: $DVC_EXPT_DPATH/models/fusion/Drop4-SC/packages/Drop4_tune_V30_8GSD_V3/Drop4_tune_V30_8GSD_V3_epoch=2-step=17334.pt.pt
    # init: /home/joncrall/remote/toothbrush/data/dvc-repos/smart_expt_dvc/training/toothbrush/joncrall/Drop7-Cropped2GSD/runs/Drop7-Cropped2GSD_SC_bgrn_gnt_sgd_split6_V86/lightning_logs/version_0/checkpoints/epoch=39-step=1280-val_loss=3.681.ckpt.pt
    # init: /home/joncrall/remote/toothbrush/data/dvc-repos/smart_expt_dvc/training/toothbrush/joncrall/Drop7-Cropped2GSD/runs/Drop7-Cropped2GSD_SC_bgrn_snp_sgd_split6_V86/lightning_logs/version_2/package-interupt/package_epoch1_step81.pt
    #init: /home/joncrall/remote/toothbrush/data/dvc-repos/smart_expt_dvc/training/toothbrush/joncrall/Drop7-Cropped2GSD/runs/Drop7-Cropped2GSD_SC_bgrn_snp_sgd_split6_V86/lightning_logs/version_4/checkpoints/epoch=56-step=3648-val_loss=3.596.ckpt_weight_hacked.pt
    #init: /home/joncrall/remote/toothbrush/data/dvc-repos/smart_expt_dvc/training/toothbrush/joncrall/Drop7-Cropped2GSD/runs/Drop7-Cropped2GSD_SC_bgrn_snp_sgd_split6_V86/lightning_logs/version_6/checkpoints/epoch=17-step=2304-val_loss=2.604.ckpt.ckpt
    # init: /home/joncrall/remote/toothbrush/data/dvc-repos/smart_expt_dvc/training/toothbrush/joncrall/Drop7-Cropped2GSD/runs/Drop7-Cropped2GSD_SC_bgrn_snp_sgd_split6_V86/lightning_logs/version_8/checkpoints/epoch=0-step=128-val_loss=2.614.ckpt.ckpt
    init: /home/joncrall/remote/toothbrush/data/dvc-repos/smart_expt_dvc/training/toothbrush/joncrall/Drop7-Cropped2GSD/runs/Drop7-Cropped2GSD_SC_bgrn_snp_sgd_split6_V86/lightning_logs/version_9/checkpoints/epoch=9-step=640-val_loss=2.145.ckpt.ckpt
"


export CUDA_VISIBLE_DEVICES=""
DVC_DATA_DPATH=$(geowatch_dvc --tags='drop7_data' --hardware='auto')
DVC_EXPT_DPATH=$(geowatch_dvc --tags='phase2_expt' --hardware='auto')
echo "DVC_DATA_DPATH = $DVC_DATA_DPATH"
echo "DVC_EXPT_DPATH = $DVC_EXPT_DPATH"
WORKDIR=$DVC_EXPT_DPATH/training/$HOSTNAME/$USER
DATASET_CODE=Drop7-Cropped2GSD
KWCOCO_BUNDLE_DPATH=$DVC_DATA_DPATH/$DATASET_CODE
TRAIN_FPATH=$KWCOCO_BUNDLE_DPATH/data_train_rawbands_split6.kwcoco.zip
VALI_FPATH=$KWCOCO_BUNDLE_DPATH/data_vali_rawbands_split6.kwcoco.zip
CHANNELS="(L8,S2):(blue|green|red|nir),(WV):(blue|green|red)"
EXPERIMENT_NAME=Drop7-Cropped2GSD_SC_bgrn_snp_sgd_split6_V86_viz
DEFAULT_ROOT_DIR=$WORKDIR/$DATASET_CODE/runs/$EXPERIMENT_NAME
TARGET_LR=1e-4
WEIGHT_DECAY=$(python -c "print($TARGET_LR * 0.01)")
echo "WEIGHT_DECAY = $WEIGHT_DECAY"
MAX_STEPS=1
DDP_WORKAROUND=0 python -m geowatch.tasks.fusion fit --config "
data:
    select_videos          : $SELECT_VIDEOS
    num_workers            : 0
    train_dataset          : $TRAIN_FPATH
    vali_dataset           : $VALI_FPATH
    window_dims            : '196,196'
    time_steps             : 11
    time_sampling          : uniform-soft5-soft4-contiguous
    time_kernel            : '(-1.0y,-0.9y,-0.5y,-0.25y,-0.08y,0.0y,0.08y,0.25y,0.5y,0.9y,1.0y)'
    window_resolution     : 2.0GSD
    input_resolution      : 2.0GSD
    output_resolution     : 2.0GSD
    neg_to_pos_ratio       : 1.0
    batch_size             : 1
    normalize_perframe     : false
    normalize_peritem      : 'blue|green|red|nir|pan'
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
    use_grid_positives     : False
    use_grid_negatives     : False
    normalize_inputs       : 1024
    balance_areas          : True
model:
    class_path: MultimodalTransformer
    init_args:
        #saliency_weights      : '1:1'
        #class_weights         : auto
        class_weights          : 'auto'
        tokenizer              : linconv
        arch_name              : smt_it_stm_p24
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
        global_class_weight    : 1.00
        global_saliency_weight : 1.00
        multimodal_reduce      : learned_linear
        continual_learning     : true
        perterb_scale          : 1e-7
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
    pct_start: 0.90
trainer:
    accumulate_grad_batches: 32
    default_root_dir     : $DEFAULT_ROOT_DIR
    accelerator          : cpu
    limit_val_batches    : 256
    limit_train_batches  : 2048
    num_sanity_val_steps : 0
    max_epochs           : 360
    callbacks:
        - class_path: pytorch_lightning.callbacks.ModelCheckpoint
          init_args:
              monitor: val_loss
              mode: min
              save_top_k: 5
              filename: '{epoch}-{step}-{val_loss:.3f}.ckpt'
              save_last: true

torch_globals:
    float32_matmul_precision: auto

initializer:
    init: /home/joncrall/remote/toothbrush/data/dvc-repos/smart_expt_dvc/training/toothbrush/joncrall/Drop7-Cropped2GSD/runs/Drop7-Cropped2GSD_SC_bgrn_snp_sgd_split6_V86/lightning_logs/version_10/checkpoints/epoch=162-step=10432-val_loss=2.951.ckpt.ckpt
"


# Try to make the kwcoco file smaller
python -c "if 1:
import kwcoco
# dset = kwcoco.CocoDataset('data_train_rawbands_split6.kwcoco.zip')
# dset = kwcoco.CocoDataset('data_vali_rawbands_split6.kwcoco.zip')
# dset = kwcoco.CocoDataset('data_vali_rawbands_split6-subset.kwcoco.zip')
dset = kwcoco.CocoDataset('data_train_rawbands_split6-subset.kwcoco.zip')

for vid in dset.videos().objs:
    if 'domain' not in vid:
        vid['domain'] = vid.get('name').split('_CLUSTER')[0]

print(set(dset.videos().lookup('domain')))

# These regions are too big. Remove them for now.
remove_vidnames = {
    'SA_C005', 'CN_C001', 'QA_C001',
    'US_C010', 'US_C000', 'US_C014',
    'CN_C001', 'SA_C001', 'VN_C002',
}

remove_vids = []
for vid in dset.videos().objs:
    if vid['domain'] in remove_vidnames:
        remove_vids.append(vid)

dset.remove_videos(remove_vids)

to_remove = []
for img in ub.ProgIter(dset.dataset['images']):
    if img.get('sensor_coarse') == 'PD':
        to_remove.append(img)
dset.remove_images(to_remove)

for img in ub.ProgIter(dset.dataset['images']):
    img.pop('parent_stac_properties', None)
    img.pop('union_parent', None)
    for asset in img['auxiliary']:
        asset.pop('parent_file_names', None)
        asset.pop('geos_corners', None)

for ann in ub.ProgIter(dset.dataset['annotations']):
    ann.pop('segmentation_geos', None)


if '-subset' in dset.fpath:
    new_fpath = ub.Path(dset.fpath)
else:
    new_fpath = ub.Path(dset.fpath).augment(stemsuffix='-subset', multidot=True)
dset.fpath = new_fpath
dset.dump()
"

export CUDA_VISIBLE_DEVICES=1
DVC_DATA_DPATH=$(geowatch_dvc --tags='drop7_data' --hardware='auto')
DVC_EXPT_DPATH=$(geowatch_dvc --tags='phase2_expt' --hardware='auto')
echo "DVC_DATA_DPATH = $DVC_DATA_DPATH"
echo "DVC_EXPT_DPATH = $DVC_EXPT_DPATH"
WORKDIR=$DVC_EXPT_DPATH/training/$HOSTNAME/$USER
DATASET_CODE=Drop7-Cropped2GSD-V2
KWCOCO_BUNDLE_DPATH=$DVC_DATA_DPATH/$DATASET_CODE
TRAIN_FPATH=$KWCOCO_BUNDLE_DPATH/data_train_rawbands_split6-subset.kwcoco.zip
VALI_FPATH=$KWCOCO_BUNDLE_DPATH/data_vali_rawbands_split6.kwcoco.zip
CHANNELS="(L8,S2):(blue|green|red|nir),(WV):(blue|green|red)"
EXPERIMENT_NAME=Drop7-Cropped2GSD_SC_bgrn_sgd_gnt_split6_V87
DEFAULT_ROOT_DIR=$WORKDIR/$DATASET_CODE/runs/$EXPERIMENT_NAME
TARGET_LR=1e-4
WEIGHT_DECAY=$(python -c "print($TARGET_LR * 0.01)")
echo "WEIGHT_DECAY = $WEIGHT_DECAY"
MAX_STEPS=80000
DDP_WORKAROUND=0 python -m geowatch.tasks.fusion fit --config "
data:
    select_videos          : $SELECT_VIDEOS
    num_workers            : 2
    train_dataset          : $TRAIN_FPATH
    vali_dataset           : $VALI_FPATH
    window_dims            : '196,196'
    time_steps             : 11
    time_sampling          : uniform-soft5-soft4-contiguous
    time_kernel            : '(-1.0y,-0.9y,-0.5y,-0.25y,-0.08y,0.0y,0.08y,0.25y,0.5y,0.9y,1.0y)'
    window_resolution     : 2.0GSD
    input_resolution      : 2.0GSD
    output_resolution     : 2.0GSD
    neg_to_pos_ratio       : 1.0
    batch_size             : 1
    normalize_perframe     : false
    normalize_peritem      : 'blue|green|red|nir|pan'
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
    use_grid_positives     : False
    use_grid_negatives     : False
    normalize_inputs       : 1024
    balance_areas          : True
    sqlview                : sqlite
model:
    class_path: MultimodalTransformer
    init_args:
        #saliency_weights      : '1:1'
        #class_weights         : auto
        class_weights          : 'auto'
        tokenizer              : linconv
        arch_name              : smt_it_stm_p24
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
        global_class_weight    : 1.00
        global_saliency_weight : 1.00
        multimodal_reduce      : learned_linear
        continual_learning     : true
        perterb_scale          : 1e-8
optimizer:
    class_path: torch.optim.SGD
    init_args:
        lr           : $TARGET_LR
        weight_decay : $WEIGHT_DECAY
lr_scheduler:
  class_path: torch.optim.lr_scheduler.OneCycleLR
  init_args:
    max_lr: $TARGET_LR
    total_steps: $MAX_STEPS
    anneal_strategy: cos
    pct_start: 0.90
trainer:
    accumulate_grad_batches: 32
    default_root_dir     : $DEFAULT_ROOT_DIR
    accelerator          : gpu
    devices              : 0,
    #strategy             : ddp_find_unused_parameters_true
    limit_val_batches    : 256
    limit_train_batches  : 2048
    num_sanity_val_steps : 0
    max_epochs           : 360
    callbacks:
        - class_path: pytorch_lightning.callbacks.ModelCheckpoint
          init_args:
              monitor: val_loss
              mode: min
              save_top_k: 5
              filename: '{epoch}-{step}-{val_loss:.3f}.ckpt'
              save_last: true

torch_globals:
    float32_matmul_precision: auto

initializer:
    init: /home/joncrall/remote/toothbrush/data/dvc-repos/smart_expt_dvc/training/toothbrush/joncrall/Drop7-Cropped2GSD/runs/Drop7-Cropped2GSD_SC_bgrn_snp_sgd_split6_V86/lightning_logs/version_10/checkpoints/epoch=189-step=12160-val_loss=2.881.ckpt.ckpt
"


export CUDA_VISIBLE_DEVICES=0
DVC_DATA_DPATH=$(geowatch_dvc --tags='drop7_data' --hardware='auto')
DVC_EXPT_DPATH=$(geowatch_dvc --tags='phase2_expt' --hardware='auto')
echo "DVC_DATA_DPATH = $DVC_DATA_DPATH"
echo "DVC_EXPT_DPATH = $DVC_EXPT_DPATH"
WORKDIR=$DVC_EXPT_DPATH/training/$HOSTNAME/$USER
DATASET_CODE=Drop7-Cropped2GSD-V2
KWCOCO_BUNDLE_DPATH=$DVC_DATA_DPATH/$DATASET_CODE
TRAIN_FPATH=$KWCOCO_BUNDLE_DPATH/data_train_rawbands_split6-subset.kwcoco.zip
VALI_FPATH=$KWCOCO_BUNDLE_DPATH/data_vali_rawbands_split6.kwcoco.zip
CHANNELS="(L8,S2):(blue|green|red|nir),(WV):(blue|green|red)"
EXPERIMENT_NAME=Drop7-Cropped2GSD_SC_bgrn_sgd_gnt_split6_V87
DEFAULT_ROOT_DIR=$WORKDIR/$DATASET_CODE/runs/$EXPERIMENT_NAME
TARGET_LR=1e-4
WEIGHT_DECAY=$(python -c "print($TARGET_LR * 0.01)")
echo "WEIGHT_DECAY = $WEIGHT_DECAY"
MAX_STEPS=80000
DDP_WORKAROUND=0 python -m geowatch.tasks.fusion fit --config "
data:
    select_videos          : $SELECT_VIDEOS
    num_workers            : 2
    train_dataset          : $TRAIN_FPATH
    vali_dataset           : $VALI_FPATH
    window_dims            : '196,196'
    time_steps             : 11
    time_sampling          : uniform-soft5-soft4-contiguous
    time_kernel            : '(-1.0y,-0.9y,-0.5y,-0.25y,-0.08y,0.0y,0.08y,0.25y,0.5y,0.9y,1.0y)'
    window_resolution     : 2.0GSD
    input_resolution      : 2.0GSD
    output_resolution     : 2.0GSD
    neg_to_pos_ratio       : 1.0
    batch_size             : 3
    normalize_perframe     : false
    normalize_peritem      : 'blue|green|red|nir|pan'
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
    use_grid_positives     : False
    use_grid_negatives     : False
    normalize_inputs       : 1024
    balance_areas          : True
    sqlview                : sqlite
model:
    class_path: MultimodalTransformer
    init_args:
        #saliency_weights      : '1:1'
        #class_weights         : auto
        class_weights          : 'auto'
        tokenizer              : linconv
        arch_name              : smt_it_stm_p24
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
        global_class_weight    : 1.00
        global_saliency_weight : 1.00
        multimodal_reduce      : learned_linear
        continual_learning     : true
        perterb_scale          : 1e-8
optimizer:
    class_path: torch.optim.SGD
    init_args:
        lr           : $TARGET_LR
        weight_decay : $WEIGHT_DECAY
lr_scheduler:
  class_path: torch.optim.lr_scheduler.OneCycleLR
  init_args:
    max_lr: $TARGET_LR
    total_steps: $MAX_STEPS
    anneal_strategy: cos
    pct_start: 0.95
trainer:
    accumulate_grad_batches: 64
    default_root_dir     : $DEFAULT_ROOT_DIR
    accelerator          : gpu
    devices              : 0,
    #strategy            : ddp_find_unused_parameters_true
    limit_val_batches    : 256
    limit_train_batches  : 2048
    num_sanity_val_steps : 0
    max_epochs           : 360
    callbacks:
        - class_path: pytorch_lightning.callbacks.ModelCheckpoint
          init_args:
              monitor: val_loss
              mode: min
              save_top_k: 5
              filename: '{epoch}-{step}-{val_loss:.3f}.ckpt'
              save_last: true

torch_globals:
    float32_matmul_precision: auto

initializer:
    init: $DVC_EXPT_DPATH/models/fusion/Drop7-Cropped2GSD/packages/Drop7-Cropped2GSD_SC_bgrn_snp_sgd_split6_V86/Drop7-Cropped2GSD_SC_bgrn_snp_sgd_split6_V86_epoch189_step12160.pt
"



export CUDA_VISIBLE_DEVICES=0
DVC_DATA_DPATH=$(geowatch_dvc --tags='drop7_data' --hardware='auto')
DVC_EXPT_DPATH=$(geowatch_dvc --tags='phase2_expt' --hardware='auto')
echo "DVC_DATA_DPATH = $DVC_DATA_DPATH"
echo "DVC_EXPT_DPATH = $DVC_EXPT_DPATH"
WORKDIR=$DVC_EXPT_DPATH/training/$HOSTNAME/$USER
DATASET_CODE=Drop7-Cropped2GSD-V2
KWCOCO_BUNDLE_DPATH=$DVC_DATA_DPATH/$DATASET_CODE
TRAIN_FPATH=$KWCOCO_BUNDLE_DPATH/data_train_rawbands_split6-subset.kwcoco.zip
VALI_FPATH=$KWCOCO_BUNDLE_DPATH/data_vali_rawbands_split6.kwcoco.zip
CHANNELS="(L8,S2):(blue|green|red|nir),(WV):(blue|green|red)"
EXPERIMENT_NAME=Drop7-Cropped2GSD_SC_bgrn_sgd_gnt_4GSD_split6_V88
DEFAULT_ROOT_DIR=$WORKDIR/$DATASET_CODE/runs/$EXPERIMENT_NAME
TARGET_LR=1e-4
WEIGHT_DECAY=$(python -c "print($TARGET_LR * 0.01)")
echo "WEIGHT_DECAY = $WEIGHT_DECAY"
MAX_STEPS=80000
DDP_WORKAROUND=0 python -m geowatch.tasks.fusion fit --config "
data:
    select_videos          : $SELECT_VIDEOS
    num_workers            : 2
    train_dataset          : $TRAIN_FPATH
    vali_dataset           : $VALI_FPATH
    window_dims            : '196,196'
    time_steps             : 11
    time_sampling          : uniform-soft5-soft4-contiguous
    time_kernel            : '(-1.0y,-0.9y,-0.5y,-0.25y,-0.08y,0.0y,0.08y,0.25y,0.5y,0.9y,1.0y)'
    window_resolution     : 4.0GSD
    input_resolution      : 4.0GSD
    output_resolution     : 4.0GSD
    neg_to_pos_ratio       : 1.0
    batch_size             : 3
    normalize_perframe     : false
    normalize_peritem      : 'blue|green|red|nir|pan'
    max_epoch_length       : 1000000
    channels               : '$CHANNELS'
    min_spacetime_weight   : 0.6
    temporal_dropout       : 0.5
    mask_low_quality       : True
    mask_samecolor_method  : None
    observable_threshold   : 0.1
    quality_threshold      : 0.0
    weight_dilate          : 10
    use_centered_positives : True
    use_grid_positives     : False
    use_grid_negatives     : False
    normalize_inputs       : 1024
    balance_areas          : False
    sqlview                : sqlite
model:
    class_path: MultimodalTransformer
    init_args:
        #saliency_weights      : '1:1'
        #class_weights         : auto
        class_weights          : 'auto'
        tokenizer              : linconv
        arch_name              : smt_it_stm_p24
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
        global_class_weight    : 1.00
        global_saliency_weight : 1.00
        multimodal_reduce      : learned_linear
        continual_learning     : true
        perterb_scale          : 1e-8
optimizer:
    class_path: torch.optim.SGD
    init_args:
        lr           : $TARGET_LR
        weight_decay : $WEIGHT_DECAY
lr_scheduler:
  class_path: torch.optim.lr_scheduler.OneCycleLR
  init_args:
    max_lr: $TARGET_LR
    total_steps: $MAX_STEPS
    anneal_strategy: cos
    pct_start: 0.95
trainer:
    accumulate_grad_batches: 64
    default_root_dir     : $DEFAULT_ROOT_DIR
    accelerator          : gpu
    devices              : 0,
    #strategy            : ddp_find_unused_parameters_true
    limit_val_batches    : 256
    limit_train_batches  : 2048
    num_sanity_val_steps : 0
    max_epochs           : 360
    callbacks:
        - class_path: pytorch_lightning.callbacks.ModelCheckpoint
          init_args:
              monitor: val_loss
              mode: min
              save_top_k: 5
              filename: '{epoch}-{step}-{val_loss:.3f}.ckpt'
              save_last: true

torch_globals:
    float32_matmul_precision: auto

initializer:
    init: $DVC_EXPT_DPATH/models/fusion/Drop7-Cropped2GSD/packages/Drop7-Cropped2GSD_SC_bgrn_snp_sgd_split6_V86/Drop7-Cropped2GSD_SC_bgrn_snp_sgd_split6_V86_epoch189_step12160.pt
"



export CUDA_VISIBLE_DEVICES=0
DVC_DATA_DPATH=$(geowatch_dvc --tags='drop7_data' --hardware='auto')
DVC_EXPT_DPATH=$(geowatch_dvc --tags='phase2_expt' --hardware='auto')
echo "DVC_DATA_DPATH = $DVC_DATA_DPATH"
echo "DVC_EXPT_DPATH = $DVC_EXPT_DPATH"
WORKDIR=$DVC_EXPT_DPATH/training/$HOSTNAME/$USER
DATASET_CODE=Drop7-Cropped2GSD-V2
KWCOCO_BUNDLE_DPATH=$DVC_DATA_DPATH/$DATASET_CODE
TRAIN_FPATH=$KWCOCO_BUNDLE_DPATH/data_train_rawbands_split6-subset.kwcoco.zip
VALI_FPATH=$KWCOCO_BUNDLE_DPATH/data_vali_rawbands_split6.kwcoco.zip
CHANNELS="(L8,S2):(blue|green|red|nir),(WV):(blue|green|red)"
EXPERIMENT_NAME=Drop7-Cropped2GSD_SC_bgrn_sgd_gnt_8GSD_split6_V89
DEFAULT_ROOT_DIR=$WORKDIR/$DATASET_CODE/runs/$EXPERIMENT_NAME
TARGET_LR=1e-4
WEIGHT_DECAY=$(python -c "print($TARGET_LR * 0.01)")
echo "WEIGHT_DECAY = $WEIGHT_DECAY"
MAX_STEPS=80000
DDP_WORKAROUND=0 python -m geowatch.tasks.fusion fit --config "
data:
    select_videos          : $SELECT_VIDEOS
    num_workers            : 2
    train_dataset          : $TRAIN_FPATH
    vali_dataset           : $VALI_FPATH
    window_dims            : '196,196'
    time_steps             : 11
    time_sampling          : uniform-soft5-soft4-contiguous
    time_kernel            : '(-1.0y,-0.9y,-0.5y,-0.25y,-0.08y,0.0y,0.08y,0.25y,0.5y,0.9y,1.0y)'
    window_resolution     : 8.0GSD
    input_resolution      : 8.0GSD
    output_resolution     : 8.0GSD
    neg_to_pos_ratio       : 1.0
    batch_size             : 3
    normalize_perframe     : false
    normalize_peritem      : 'blue|green|red|nir|pan'
    max_epoch_length       : 1000000
    channels               : '$CHANNELS'
    min_spacetime_weight   : 0.6
    temporal_dropout       : 0.5
    mask_low_quality       : True
    mask_samecolor_method  : None
    observable_threshold   : 0.1
    quality_threshold      : 0.0
    weight_dilate          : 10
    use_centered_positives : True
    use_grid_positives     : False
    use_grid_negatives     : 'cleared'
    normalize_inputs       : 1024
    balance_areas          : False
    sqlview                : sqlite
model:
    class_path: MultimodalTransformer
    init_args:
        #saliency_weights      : '1:1'
        #class_weights         : auto
        class_weights          : 'auto'
        tokenizer              : linconv
        arch_name              : smt_it_stm_p24
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
        global_class_weight    : 1.00
        global_saliency_weight : 1.00
        multimodal_reduce      : learned_linear
        continual_learning     : true
        perterb_scale          : 1e-8
optimizer:
    class_path: torch.optim.SGD
    init_args:
        lr           : $TARGET_LR
        weight_decay : $WEIGHT_DECAY
lr_scheduler:
  class_path: torch.optim.lr_scheduler.OneCycleLR
  init_args:
    max_lr: $TARGET_LR
    total_steps: $MAX_STEPS
    anneal_strategy: cos
    pct_start: 0.95
trainer:
    accumulate_grad_batches: 256
    default_root_dir     : $DEFAULT_ROOT_DIR
    accelerator          : gpu
    devices              : 0,
    #strategy            : ddp_find_unused_parameters_true
    limit_val_batches    : 256
    limit_train_batches  : 2048
    num_sanity_val_steps : 0
    max_epochs           : 360
    callbacks:
        - class_path: pytorch_lightning.callbacks.ModelCheckpoint
          init_args:
              monitor: val_loss
              mode: min
              save_top_k: 5
              filename: '{epoch}-{step}-{val_loss:.3f}.ckpt'
              save_last: true

torch_globals:
    float32_matmul_precision: auto

initializer:
    #init: $DVC_EXPT_DPATH/models/fusion/Drop7-Cropped2GSD-V2/packages/Drop7-Cropped2GSD_SC_bgrn_sgd_gnt_4GSD_split6_V88/Drop7-Cropped2GSD_SC_bgrn_sgd_gnt_4GSD_split6_V88_epoch334_step10720.pt
    init: /home/local/KHQ/jon.crall/remote/yardrat/data/dvc-repos/smart_expt_dvc/training/yardrat/jon.crall/Drop7-Cropped2GSD-V2/runs/Drop7-Cropped2GSD_SC_bgrn_sgd_gnt_8GSD_split6_V89/lightning_logs/version_0/checkpoints/epoch=18-step=152-val_loss=3.508.ckpt.ckpt
"


export CUDA_VISIBLE_DEVICES=0
DVC_DATA_DPATH=$(geowatch_dvc --tags='drop7_data' --hardware='auto')
DVC_EXPT_DPATH=$(geowatch_dvc --tags='phase2_expt' --hardware='auto')
echo "DVC_DATA_DPATH = $DVC_DATA_DPATH"
echo "DVC_EXPT_DPATH = $DVC_EXPT_DPATH"
WORKDIR=$DVC_EXPT_DPATH/training/$HOSTNAME/$USER
DATASET_CODE=Drop7-Cropped2GSD-V2
KWCOCO_BUNDLE_DPATH=$DVC_DATA_DPATH/$DATASET_CODE
TRAIN_FPATH=$KWCOCO_BUNDLE_DPATH/data_train_rawbands_split6-subset.kwcoco.zip
VALI_FPATH=$KWCOCO_BUNDLE_DPATH/data_vali_rawbands_split6.kwcoco.zip
CHANNELS="(L8,S2):(blue|green|red|nir),(WV):(blue|green|red)"
EXPERIMENT_NAME=Drop7-Cropped2GSD_SC_bgrn_gnt_4GSD_split6_V90
DEFAULT_ROOT_DIR=$WORKDIR/$DATASET_CODE/runs/$EXPERIMENT_NAME
TARGET_LR=1e-4
WEIGHT_DECAY=$(python -c "print($TARGET_LR * 0.01)")
echo "WEIGHT_DECAY = $WEIGHT_DECAY"
MAX_STEPS=80000
DDP_WORKAROUND=0 python -m geowatch.tasks.fusion fit --config "
data:
    select_videos          : $SELECT_VIDEOS
    num_workers            : 2
    train_dataset          : $TRAIN_FPATH
    vali_dataset           : $VALI_FPATH
    window_dims            : '196,196'
    time_steps             : 7
    time_sampling          : uniform-soft5-soft4-contiguous
    time_kernel            : '(-1.0y,-0.5y,-0.25y,0.0y,0.25y,0.5y,1.0y)'
    window_resolution     : 4.0GSD
    input_resolution      : 4.0GSD
    output_resolution     : 4.0GSD
    neg_to_pos_ratio       : 1.0
    batch_size             : 3
    normalize_perframe     : false
    normalize_peritem      : 'blue|green|red|nir|pan'
    max_epoch_length       : 1000000
    channels               : '$CHANNELS'
    min_spacetime_weight   : 0.6
    temporal_dropout       : 0.5
    mask_low_quality       : True
    mask_samecolor_method  : None
    observable_threshold   : 0.2
    quality_threshold      : 0.2
    weight_dilate          : 10
    use_centered_positives : True
    use_grid_positives     : False
    use_grid_negatives     : 'cleared'
    normalize_inputs       : 1024
    balance_areas          : True
    sqlview                : sqlite
model:
    class_path: MultimodalTransformer
    init_args:
        #saliency_weights      : '1:1'
        #class_weights         : auto
        class_weights          : 'auto'
        tokenizer              : linconv
        arch_name              : smt_it_stm_p32
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
        global_class_weight    : 2.00
        global_saliency_weight : 1.00
        multimodal_reduce      : learned_linear
        continual_learning     : true
        perterb_scale          : 1e-8
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
    pct_start: 0.95
trainer:
    accumulate_grad_batches: 64
    default_root_dir     : $DEFAULT_ROOT_DIR
    accelerator          : gpu
    devices              : 0,
    #strategy            : ddp_find_unused_parameters_true
    limit_val_batches    : 256
    limit_train_batches  : 2048
    num_sanity_val_steps : 0
    max_epochs           : 360
    callbacks:
        - class_path: pytorch_lightning.callbacks.ModelCheckpoint
          init_args:
              monitor: val_loss
              mode: min
              save_top_k: 5
              filename: '{epoch}-{step}-{val_loss:.3f}.ckpt'
              save_last: true

torch_globals:
    float32_matmul_precision: auto

initializer:
    init: /home/local/KHQ/jon.crall/remote/yardrat/data/dvc-repos/smart_expt_dvc/training/yardrat/jon.crall/Drop7-Cropped2GSD-V2/runs/Drop7-Cropped2GSD_SC_bgrn_sgd_gnt_8GSD_split6_V89/lightning_logs/version_1/checkpoints/epoch=112-step=904-val_loss=3.499.ckpt.ckpt
"



export CUDA_VISIBLE_DEVICES=0
DVC_DATA_DPATH=$(geowatch_dvc --tags='drop7_data' --hardware='auto')
DVC_EXPT_DPATH=$(geowatch_dvc --tags='phase2_expt' --hardware='auto')
echo "DVC_DATA_DPATH = $DVC_DATA_DPATH"
echo "DVC_EXPT_DPATH = $DVC_EXPT_DPATH"
WORKDIR=$DVC_EXPT_DPATH/training/$HOSTNAME/$USER
DATASET_CODE=Drop7-Cropped2GSD-V2
KWCOCO_BUNDLE_DPATH=$DVC_DATA_DPATH/$DATASET_CODE
TRAIN_FPATH=$KWCOCO_BUNDLE_DPATH/data_train_rawbands_split6-subset.kwcoco.zip
VALI_FPATH=$KWCOCO_BUNDLE_DPATH/data_vali_rawbands_split6.kwcoco.zip
CHANNELS="(L8,S2):(blue|green|red|nir),(WV):(blue|green|red)"
EXPERIMENT_NAME=Drop7-Cropped2GSD_SC_bgrn_gnt_2GSD_split6_V91
DEFAULT_ROOT_DIR=$WORKDIR/$DATASET_CODE/runs/$EXPERIMENT_NAME
TARGET_LR=1e-4
WEIGHT_DECAY=$(python -c "print($TARGET_LR * 0.01)")
echo "WEIGHT_DECAY = $WEIGHT_DECAY"
MAX_STEPS=80000
DDP_WORKAROUND=0 python -m geowatch.tasks.fusion fit --config "
data:
    select_videos          : $SELECT_VIDEOS
    num_workers            : 2
    train_dataset          : $TRAIN_FPATH
    vali_dataset           : $VALI_FPATH
    window_dims            : '196,196'
    time_steps             : 7
    time_sampling          : uniform-soft5-soft4-contiguous
    time_kernel            : '(-1.0y,-0.5y,-0.25y,0.0y,0.25y,0.5y,1.0y)'
    window_resolution     : 2.0GSD
    input_resolution      : 2.0GSD
    output_resolution     : 2.0GSD
    neg_to_pos_ratio       : 1.0
    batch_size             : 3
    normalize_perframe     : false
    normalize_peritem      : 'blue|green|red|nir|pan'
    max_epoch_length       : 1000000
    channels               : '$CHANNELS'
    min_spacetime_weight   : 0.6
    temporal_dropout_rate  : 0.5
    channel_dropout_rate   : 0.5
    modality_dropout_rate  : 0.5
    temporal_dropout       : 0.1
    channel_dropout        : 0.1
    modality_dropout       : 0.1
    mask_low_quality       : True
    mask_samecolor_method  : None
    observable_threshold   : 0.2
    quality_threshold      : 0.2
    weight_dilate          : 10
    use_centered_positives : True
    use_grid_positives     : False
    use_grid_negatives     : 'cleared'
    normalize_inputs       : 1024
    balance_areas          : True
    sqlview                : sqlite
model:
    class_path: MultimodalTransformer
    init_args:
        #saliency_weights      : '1:1'
        #class_weights         : auto
        class_weights          : 'auto'
        tokenizer              : linconv
        arch_name              : smt_it_stm_p32
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
        global_class_weight    : 2.00
        global_saliency_weight : 1.00
        multimodal_reduce      : learned_linear
        continual_learning     : true
        perterb_scale          : 1e-7
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
    pct_start: 0.95
trainer:
    accumulate_grad_batches: 64
    default_root_dir     : $DEFAULT_ROOT_DIR
    accelerator          : gpu
    devices              : 0,
    #strategy            : ddp_find_unused_parameters_true
    limit_val_batches    : 256
    limit_train_batches  : 2048
    num_sanity_val_steps : 0
    max_epochs           : 360
    callbacks:
        - class_path: pytorch_lightning.callbacks.ModelCheckpoint
          init_args:
              monitor: val_loss
              mode: min
              save_top_k: 5
              filename: '{epoch}-{step}-{val_loss:.3f}.ckpt'
              save_last: true

torch_globals:
    float32_matmul_precision: auto

initializer:
    init: $DVC_EXPT_DPATH/models/fusion/Drop7-Cropped2GSD-V2/packages/Drop7-Cropped2GSD_SC_bgrn_gnt_4GSD_split6_V90/Drop7-Cropped2GSD_SC_bgrn_gnt_4GSD_split6_V90_epoch343_step11008.pt
"


export CUDA_VISIBLE_DEVICES=0
DVC_DATA_DPATH=$(geowatch_dvc --tags='drop7_data' --hardware='auto')
DVC_EXPT_DPATH=$(geowatch_dvc --tags='phase2_expt' --hardware='auto')
echo "DVC_DATA_DPATH = $DVC_DATA_DPATH"
echo "DVC_EXPT_DPATH = $DVC_EXPT_DPATH"
WORKDIR=$DVC_EXPT_DPATH/training/$HOSTNAME/$USER
DATASET_CODE=Drop7-Cropped2GSD-V2
KWCOCO_BUNDLE_DPATH=$DVC_DATA_DPATH/$DATASET_CODE
TRAIN_FPATH=$KWCOCO_BUNDLE_DPATH/data_train_rawbands_split6-subset.kwcoco.zip
VALI_FPATH=$KWCOCO_BUNDLE_DPATH/data_vali_rawbands_split6.kwcoco.zip
CHANNELS="(L8,S2):(blue|green|red|nir),(WV):(blue|green|red,blue|green|red|nir08,pan),(WV1):(pan)"
EXPERIMENT_NAME=Drop7-Cropped2GSD_SC_bgrn_gnt_2GSD_split6_V92
DEFAULT_ROOT_DIR=$WORKDIR/$DATASET_CODE/runs/$EXPERIMENT_NAME
TARGET_LR=3e-4
WEIGHT_DECAY=$(python -c "print($TARGET_LR * 0.01)")
echo "WEIGHT_DECAY = $WEIGHT_DECAY"
MAX_STEPS=80000
DDP_WORKAROUND=0 python -m geowatch.tasks.fusion fit --config "
data:
    select_videos          : $SELECT_VIDEOS
    num_workers            : 6
    train_dataset          : $TRAIN_FPATH
    vali_dataset           : $VALI_FPATH
    window_dims            : '196,196'
    time_steps             : 7
    time_sampling          : uniform-soft5-soft4-contiguous
    time_kernel            : '(-1.0y,-0.5y,-0.25y,0.0y,0.25y,0.5y,1.0y)'
    window_resolution     : 2.0GSD
    input_resolution      : 2.0GSD
    output_resolution     : 2.0GSD
    neg_to_pos_ratio       : 1.0
    batch_size             : 1
    normalize_perframe     : false
    normalize_peritem      : 'blue|green|red|nir|nir08|pan'
    max_epoch_length       : 1000000
    channels               : '$CHANNELS'
    min_spacetime_weight   : 0.6
    temporal_dropout_rate  : 0.5
    channel_dropout_rate   : 0.5
    modality_dropout_rate  : 0.5
    temporal_dropout       : 0.1
    channel_dropout        : 0.1
    modality_dropout       : 0.1
    mask_low_quality       : True
    mask_samecolor_method  : None
    observable_threshold   : 0.2
    quality_threshold      : 0.2
    weight_dilate          : 10
    use_centered_positives : True
    use_grid_positives     : False
    use_grid_negatives     : 'cleared'
    normalize_inputs       : 1024
    balance_areas          : True
    sqlview                : sqlite
model:
    class_path: MultimodalTransformer
    init_args:
        #saliency_weights      : '1:1'
        #class_weights         : auto
        class_weights          : 'auto'
        tokenizer              : linconv
        arch_name              : smt_it_stm_m24
        decoder                : mlp
        positive_change_weight : 1
        negative_change_weight : 0.01
        stream_channels        : 16
        class_loss             : 'dicefocal'
        saliency_loss          : 'focal'
        saliency_head_hidden   : 8
        change_head_hidden     : 6
        class_head_hidden      : 8
        global_change_weight   : 0.00
        global_class_weight    : 1.00
        global_saliency_weight : 1.00
        multimodal_reduce      : learned_linear
        continual_learning     : true
        perterb_scale          : 3e-7
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
    pct_start: 0.95
trainer:
    accumulate_grad_batches: 64
    default_root_dir     : $DEFAULT_ROOT_DIR
    accelerator          : gpu
    devices              : 0,
    #strategy            : ddp_find_unused_parameters_true
    limit_val_batches    : 2048
    limit_train_batches  : 2048
    num_sanity_val_steps : 0
    max_epochs           : 360
    callbacks:
        - class_path: pytorch_lightning.callbacks.ModelCheckpoint
          init_args:
              monitor: val_loss
              mode: min
              save_top_k: 5
              filename: '{epoch}-{step}-{val_loss:.3f}.ckpt'
              save_last: true

torch_globals:
    float32_matmul_precision: auto

initializer:
    init: $DVC_EXPT_DPATH/models/fusion/Drop7-MedianNoWinter10GSD-NoMask/packages/Drop7-MedianNoWinter10GSD_bgrn_mixed_split6_V78/Drop7-MedianNoWinter10GSD_bgrn_mixed_split6_V78_epoch5_step96.pt
"


export CUDA_VISIBLE_DEVICES=0
DVC_DATA_DPATH=$(geowatch_dvc --tags='drop7_data' --hardware='auto')
DVC_EXPT_DPATH=$(geowatch_dvc --tags='phase2_expt' --hardware='auto')
echo "DVC_DATA_DPATH = $DVC_DATA_DPATH"
echo "DVC_EXPT_DPATH = $DVC_EXPT_DPATH"
WORKDIR=$DVC_EXPT_DPATH/training/$HOSTNAME/$USER
DATASET_CODE=Drop7-Cropped2GSD-V2
KWCOCO_BUNDLE_DPATH=$DVC_DATA_DPATH/$DATASET_CODE
TRAIN_FPATH=$KWCOCO_BUNDLE_DPATH/data_train_rawbands_split6-subset.kwcoco.zip
VALI_FPATH=$KWCOCO_BUNDLE_DPATH/data_vali_rawbands_split6.kwcoco.zip
CHANNELS="(L8,S2):(blue|green|red|nir),(WV):(blue|green|red|nir08,pan),(WV1):(pan)"
EXPERIMENT_NAME=Drop7-Cropped2GSD_SC_bgrn_gnt_2GSD_split6_V93
DEFAULT_ROOT_DIR=$WORKDIR/$DATASET_CODE/runs/$EXPERIMENT_NAME
TARGET_LR=3e-4
WEIGHT_DECAY=$(python -c "print($TARGET_LR * 0.01)")
echo "WEIGHT_DECAY = $WEIGHT_DECAY"
MAX_STEPS=80000
DDP_WORKAROUND=0 python -m geowatch.tasks.fusion fit --config "
data:
    select_videos          : $SELECT_VIDEOS
    num_workers            : 6
    train_dataset          : $TRAIN_FPATH
    vali_dataset           : $VALI_FPATH
    window_dims            : '196,196'
    time_steps             : 7
    time_sampling          : uniform-soft5-soft4-contiguous
    time_kernel            : '(-1.0y,-0.5y,-0.25y,0.0y,0.25y,0.5y,1.0y)'
    window_resolution     : 2.0GSD
    input_resolution      : 2.0GSD
    output_resolution     : 2.0GSD
    neg_to_pos_ratio       : 1.0
    batch_size             : 1
    normalize_perframe     : false
    normalize_peritem      : 'blue|green|red|nir|nir08|pan'
    max_epoch_length       : 1000000
    channels               : '$CHANNELS'
    min_spacetime_weight   : 0.6
    temporal_dropout_rate  : 0.5
    channel_dropout_rate   : 0.5
    modality_dropout_rate  : 0.5
    temporal_dropout       : 0.1
    channel_dropout        : 0.1
    modality_dropout       : 0.1
    mask_low_quality       : True
    mask_samecolor_method  : None
    observable_threshold   : 0.2
    quality_threshold      : 0.2
    weight_dilate          : 10
    use_centered_positives : True
    use_grid_positives     : True
    use_grid_negatives     : 'cleared'
    normalize_inputs       : 1024
    balance_areas          : True
    sqlview                : sqlite
model:
    class_path: MultimodalTransformer
    init_args:
        #saliency_weights      : '1:1'
        #class_weights         : auto
        class_weights          : 'auto'
        tokenizer              : linconv
        arch_name              : smt_it_stm_m24
        decoder                : mlp
        positive_change_weight : 1
        negative_change_weight : 0.01
        stream_channels        : 16
        class_loss             : 'dicefocal'
        saliency_loss          : 'focal'
        saliency_head_hidden   : 8
        change_head_hidden     : 6
        class_head_hidden      : 8
        global_change_weight   : 0.00
        global_class_weight    : 1.00
        global_saliency_weight : 1.00
        multimodal_reduce      : learned_linear
        continual_learning     : true
        perterb_scale          : 3e-7
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
    pct_start: 0.95
trainer:
    accumulate_grad_batches: 64
    default_root_dir     : $DEFAULT_ROOT_DIR
    accelerator          : gpu
    devices              : 0,
    #strategy            : ddp_find_unused_parameters_true
    limit_val_batches    : 2048
    limit_train_batches  : 2048
    num_sanity_val_steps : 0
    max_epochs           : 360
    callbacks:
        - class_path: pytorch_lightning.callbacks.ModelCheckpoint
          init_args:
              monitor: val_loss
              mode: min
              save_top_k: 5
              filename: '{epoch}-{step}-{val_loss:.3f}.ckpt'
              save_last: true

torch_globals:
    float32_matmul_precision: auto

initializer:
    init: /home/local/KHQ/jon.crall/remote/yardrat/data/dvc-repos/smart_expt_dvc/training/yardrat/jon.crall/Drop7-Cropped2GSD-V2/runs/Drop7-Cropped2GSD_SC_bgrn_gnt_2GSD_split6_V92/lightning_logs/version_4/checkpoints/last_weight_hacked.ckpt
"


### ------ Fine tune the good COLD model on RGB only
#

# Hack
wads2
cd Drop7-MedianNoWinter10GSD-V2
python -c "if 1:
    import kwcoco
    dset = kwcoco.CocoDataset('data_train_rawbands_split6.kwcoco.zip')
    to_remove = list(dset.videos(names=['CO_C011', 'VN_C003']))
    dset.remove_videos(to_remove)
    dset.fpath = 'data_train_rawbands_split6_hack.kwcoco.zip'
    dset.validate()
    dset.dump()
"

export CUDA_VISIBLE_DEVICES=0
DVC_DATA_DPATH=$(geowatch_dvc --tags='phase2_data' --hardware='ssd')
DVC_EXPT_DPATH=$(geowatch_dvc --tags='phase2_expt' --hardware='hdd')
echo "DVC_EXPT_DPATH = $DVC_EXPT_DPATH"
WORKDIR=$DVC_EXPT_DPATH/training/$HOSTNAME/$USER
DATASET_CODE=Drop7-MedianNoWinter10GSD-V2
KWCOCO_BUNDLE_DPATH=$DVC_DATA_DPATH/$DATASET_CODE
TRAIN_FPATH=$KWCOCO_BUNDLE_DPATH/data_train_rawbands_split6_hack.kwcoco.zip
VALI_FPATH=$KWCOCO_BUNDLE_DPATH/data_vali_rawbands_split6.kwcoco.zip
CHANNELS="(L8,S2):(blue|green|red|nir),(WV):(blue|green|red)"
EXPERIMENT_NAME=Drop7_finetune_COLD_phase3_V01
DEFAULT_ROOT_DIR=$WORKDIR/$DATASET_CODE/runs/$EXPERIMENT_NAME
TARGET_LR=3e-4
MAX_STEPS=5000
WATCH_GRID_WORKERS=4 python -m geowatch.tasks.fusion fit --config "
data:
  batch_size              : 2
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
  max_epoch_length        : 2048
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
    continual_learning: true
    decoder: mlp
    decouple_resolution: false
    dropout: 0.1
    focal_gamma: 2.0
    global_change_weight: 0.0
    global_class_weight: 1.0
    global_saliency_weight: 0.01
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
    perterb_scale: 1.0e-07
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
    pct_start: 0.1
optimizer:
  class_path: torch.optim.AdamW
  init_args:
    lr: $TARGET_LR
    weight_decay: 3e-06
    betas:
      - 0.9
      - 0.99
trainer:
  accumulate_grad_batches: 48
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
  max_epochs: 720
  num_sanity_val_steps: 0
  limit_val_batches: 256
  limit_train_batches: 2048
"

### ------ Second fine tune cold experiment, but actually init from the COLD model this time...
#

export CUDA_VISIBLE_DEVICES="3,"
DVC_DATA_DPATH=$(geowatch_dvc --tags='phase2_data' --hardware='ssd')
DVC_EXPT_DPATH=$(geowatch_dvc --tags='phase2_expt' --hardware='hdd')
echo "DVC_EXPT_DPATH = $DVC_EXPT_DPATH"
#sdvc request "$DVC_EXPT_DPATH/models/fusion/uconn/D7-V2-COLD-candidate/epoch=203-step=4488.pt"
#geowatch model_stats "$DVC_EXPT_DPATH/models/fusion/uconn/D7-V2-COLD-candidate/epoch=203-step=4488.pt"
WORKDIR=$DVC_EXPT_DPATH/training/$HOSTNAME/$USER
DATASET_CODE=Drop7-MedianNoWinter10GSD-V2
KWCOCO_BUNDLE_DPATH=$DVC_DATA_DPATH/$DATASET_CODE
TRAIN_FPATH=$KWCOCO_BUNDLE_DPATH/data_train_rawbands_split6_hack.kwcoco.zip
VALI_FPATH=$KWCOCO_BUNDLE_DPATH/data_vali_rawbands_split6.kwcoco.zip
CHANNELS="(L8,S2):(blue|green|red|nir),(WV):(blue|green|red)"
EXPERIMENT_NAME=Drop7_finetune_COLD_phase3_V02
DEFAULT_ROOT_DIR=$WORKDIR/$DATASET_CODE/runs/$EXPERIMENT_NAME
TARGET_LR=3e-4
MAX_STEPS=5000
WATCH_GRID_WORKERS=4 python -m geowatch.tasks.fusion fit --config "
data:
  batch_size              : 2
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
  max_epoch_length        : 2048
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
    continual_learning: true
    decoder: mlp
    decouple_resolution: false
    dropout: 0.1
    focal_gamma: 2.0
    global_change_weight: 0.0
    global_class_weight: 1.0
    global_saliency_weight: 0.01
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
    perterb_scale: 1.0e-07
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
    pct_start: 0.1
optimizer:
  class_path: torch.optim.AdamW
  init_args:
    lr: $TARGET_LR
    weight_decay: 3e-06
    betas:
      - 0.9
      - 0.99
trainer:
  accumulate_grad_batches: 48
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
  max_epochs: 720
  num_sanity_val_steps: 0
  limit_val_batches: 256
  limit_train_batches: 2048
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
    init: $DVC_EXPT_DPATH/models/fusion/uconn/D7-V2-COLD-candidate/epoch=203-step=4488.pt
"


### ------ 3rd fine tune cold experiment, without the class head

export CUDA_VISIBLE_DEVICES="2,"
DVC_DATA_DPATH=$(geowatch_dvc --tags='phase2_data' --hardware='ssd')
DVC_EXPT_DPATH=$(geowatch_dvc --tags='phase2_expt' --hardware='hdd')
echo "DVC_EXPT_DPATH = $DVC_EXPT_DPATH"
WORKDIR=$DVC_EXPT_DPATH/training/$HOSTNAME/$USER
DATASET_CODE=Drop7-MedianNoWinter10GSD-V2
KWCOCO_BUNDLE_DPATH=$DVC_DATA_DPATH/$DATASET_CODE
TRAIN_FPATH=$KWCOCO_BUNDLE_DPATH/data_train_rawbands_split6_hack.kwcoco.zip
VALI_FPATH=$KWCOCO_BUNDLE_DPATH/data_vali_rawbands_split6.kwcoco.zip
CHANNELS="(L8,S2):(blue|green|red|nir),(WV):(blue|green|red)"
EXPERIMENT_NAME=Drop7_finetune_COLD_phase3_V03
DEFAULT_ROOT_DIR=$WORKDIR/$DATASET_CODE/runs/$EXPERIMENT_NAME
TARGET_LR=3e-4
MAX_STEPS=5000
WATCH_GRID_WORKERS=4 python -m geowatch.tasks.fusion fit --config "
data:
  batch_size              : 6
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
  max_epoch_length        : 2048
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
    continual_learning: true
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
    perterb_scale: 1.0e-07
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
    pct_start: 0.1
optimizer:
  class_path: torch.optim.AdamW
  init_args:
    lr: $TARGET_LR
    weight_decay: 3e-06
    betas:
      - 0.9
      - 0.99
trainer:
  accumulate_grad_batches: 16
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
  max_epochs: 720
  num_sanity_val_steps: 0
  limit_val_batches: 256
  limit_train_batches: 2048
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
    init: $DVC_EXPT_DPATH/models/fusion/uconn/D7-V2-COLD-candidate/epoch=203-step=4488.pt
"



### Package up and evaluate Models

# shellcheck disable=SC2155
export DVC_DATA_DPATH=$(geowatch_dvc --tags="phase2_data")
# shellcheck disable=SC2155
export DVC_EXPT_DPATH=$(geowatch_dvc --tags="phase2_expt")
cd "$DVC_EXPT_DPATH"
python -m geowatch.mlops.manager "status" --dataset_codes "Drop7-MedianNoWinter10GSD-V2"
#python -m geowatch.mlops.manager "add packages" --dataset_codes "Drop7-MedianNoWinter10GSD-V2"
python -m geowatch.mlops.manager "push packages" --dataset_codes "Drop7-MedianNoWinter10GSD-V2"
python -m geowatch.mlops.manager "list packages" --dataset_codes "Drop7-MedianNoWinter10GSD-V2"


### Evaluate
DVC_DATA_DPATH=$(geowatch_dvc --tags='phase2_data' --hardware=ssd)
DVC_EXPT_DPATH=$(geowatch_dvc --tags='phase2_expt' --hardware=auto)
geowatch schedule --params="
    pipeline: bas

    matrix:
        bas_pxl.package_fpath:
            - $DVC_EXPT_DPATH/models/fusion/uconn/D7-V2-COLD-candidate/epoch=203-step=4488.pt
            - $DVC_EXPT_DPATH/models/fusion/Drop7-MedianNoWinter10GSD-V2/packages/Drop7_finetune_COLD_phase3_V02/Drop7_finetune_COLD_phase3_V02_epoch60_step1342.pt
            - $DVC_EXPT_DPATH/models/fusion/Drop7-MedianNoWinter10GSD-V2/packages/Drop7_finetune_COLD_phase3_V02/Drop7_finetune_COLD_phase3_V02_epoch79_step1760.pt
            - $DVC_EXPT_DPATH/models/fusion/Drop7-MedianNoWinter10GSD-V2/packages/Drop7_finetune_COLD_phase3_V02/Drop7_finetune_COLD_phase3_V02_epoch214_step4730.pt
            - $DVC_EXPT_DPATH/models/fusion/Drop7-MedianNoWinter10GSD-V2/packages/Drop7_finetune_COLD_phase3_V02/Drop7_finetune_COLD_phase3_V02_epoch45_step1012.pt
            - $DVC_EXPT_DPATH/models/fusion/Drop7-MedianNoWinter10GSD-V2/packages/Drop7_finetune_COLD_phase3_V02/Drop7_finetune_COLD_phase3_V02_epoch88_step1958.pt
            - $DVC_EXPT_DPATH/models/fusion/Drop7-MedianNoWinter10GSD-V2/packages/Drop7_finetune_COLD_phase3_V02/Drop7_finetune_COLD_phase3_V02_epoch120_step2662.pt
            - $DVC_EXPT_DPATH/models/fusion/Drop7-MedianNoWinter10GSD-V2/packages/Drop7_finetune_COLD_phase3_V02/Drop7_finetune_COLD_phase3_V02_epoch63_step1408.pt
            - $DVC_EXPT_DPATH/models/fusion/Drop7-MedianNoWinter10GSD-V2/packages/Drop7_finetune_COLD_phase3_V01/Drop7_finetune_COLD_phase3_V01_epoch226_step4994.pt
            - $DVC_EXPT_DPATH/models/fusion/Drop7-MedianNoWinter10GSD-V2/packages/Drop7_finetune_COLD_phase3_V03/Drop7_finetune_COLD_phase3_V03_epoch208_step4598.pt
            - $DVC_EXPT_DPATH/models/fusion/Drop7-MedianNoWinter10GSD-V2/packages/Drop7_finetune_COLD_phase3_V03/Drop7_finetune_COLD_phase3_V03_epoch226_step4994.pt
            - $DVC_EXPT_DPATH/models/fusion/Drop7-MedianNoWinter10GSD-V2/packages/Drop7_finetune_COLD_phase3_V03/Drop7_finetune_COLD_phase3_V03_epoch209_step4620.pt
        bas_pxl.test_dataset:
            - $DVC_DATA_DPATH/Drop7-MedianNoWinter10GSD-V2/KR_R002/imganns-KR_R002-rawbands.kwcoco.zip
            - $DVC_DATA_DPATH/Drop7-MedianNoWinter10GSD-V2/CN_C000/imganns-CN_C000-rawbands.kwcoco.zip
            - $DVC_DATA_DPATH/Drop7-MedianNoWinter10GSD-V2/KW_C001/imganns-KW_C001-rawbands.kwcoco.zip
            - $DVC_DATA_DPATH/Drop7-MedianNoWinter10GSD-V2/CO_C001/imganns-CO_C001-rawbands.kwcoco.zip
        bas_pxl.chip_overlap: 0.3
        bas_pxl.chip_dims: auto
        bas_pxl.time_span: auto
        bas_pxl.time_sampling: soft4
        bas_poly.thresh:
            - 0.10
            - 0.20
            - 0.25
            - 0.275
            - 0.30
            - 0.325
            - 0.35
            - 0.375
            - 0.4
            - 0.425
        bas_poly.inner_window_size: 1y
        bas_poly.inner_agg_fn: mean
        bas_poly.norm_ord: inf
        bas_poly.polygon_simplify_tolerance: 1
        bas_poly.agg_fn: probs
        bas_poly.time_thresh:
            - 0.8
            #- 0.6
        bas_poly.resolution: 10GSD
        bas_poly.moving_window_size: null
        bas_poly.poly_merge_method: 'v2'
        bas_poly.min_area_square_meters: 7200
        bas_poly.max_area_square_meters: 8000000
        bas_poly.boundary_region: $DVC_DATA_DPATH/annotations/drop7/region_models
        bas_poly_eval.true_site_dpath: $DVC_DATA_DPATH/annotations/drop7/site_models
        bas_poly_eval.true_region_dpath: $DVC_DATA_DPATH/annotations/drop7/region_models
        bas_pxl.enabled: 1
        bas_pxl_eval.enabled: 0
        bas_poly.enabled: 1
        bas_poly_eval.enabled: 1
        bas_poly_viz.enabled: 0
    " \
    --root_dpath="$DVC_EXPT_DPATH/_instill_cold_finetune_test" \
    --devices="0,1,2,3" --tmux_workers=8 \
    --backend=tmux --queue_name "_instill_cold_finetune_test" \
    --skip_existing=1 \
    --run=1


DVC_EXPT_DPATH=$(geowatch_dvc --tags='phase2_expt' --hardware=auto)
echo "DVC_EXPT_DPATH = $DVC_EXPT_DPATH"
python -m geowatch.mlops.aggregate \
    --pipeline=bas \
    --target "
        - $DVC_EXPT_DPATH/_instill_cold_finetune_test
    " \
    --output_dpath="$DVC_EXPT_DPATH/_instill_cold_finetune_test/aggregate" \
    --resource_report=0 \
    --eval_nodes="
        - bas_poly_eval
        #- bas_pxl_eval
    " \
    --plot_params="
        enabled: 0
        stats_ranking: 0
        min_variations: 1
        #params_of_interest:
        #    - params.bas_poly.thresh
    " \
    --stdout_report="
        top_k: 1100
        per_group: 1
        macro_analysis: 0
        analyze: 0
        print_models: True
        reference_region: final
        concise: 1
        show_csv: 0
    " \
    --rois="KR_R002,CN_C000,KW_C001,CO_C001"

    #--rois="KR_R002"


DVC_EXPT_DPATH=$(geowatch_dvc --tags='phase2_expt' --hardware=auto)
echo "DVC_EXPT_DPATH = $DVC_EXPT_DPATH"
python -m geowatch.mlops.aggregate \
    --pipeline=bas \
    --target "
        - $DVC_EXPT_DPATH/aggregate_results/mlops-2023-10-30/uconn/bas_poly_eval_2023-10-07T103304-5.csv.zip
    " \
    --resource_report=0 \
    --eval_nodes="
        - bas_poly_eval
        #- bas_pxl_eval
    " \
    --plot_params="
        enabled: 1
        stats_ranking: 0
        min_variations: 1
        #params_of_interest:
        #    - params.bas_poly.thresh
    " \
    --stdout_report="
        top_k: 1100
        per_group: 1
        macro_analysis: 0
        analyze: 0
        print_models: True
        reference_region: final
        concise: 1
        show_csv: 0
    " \
    --output_dpath="$DVC_EXPT_DPATH/aggregate_results/mlops-2023-10-30/uconn/agg" \
    --rois="KR_R002,CN_C000,KW_C001,CO_C001"
    #--rois="KR_R002,CN_C000,KW_C001,CO_C001"



#### ----
# Yardrat training from scratch

export CUDA_VISIBLE_DEVICES=0
DVC_DATA_DPATH=$(geowatch_dvc --tags='phase2_data' --hardware='ssd')
DVC_EXPT_DPATH=$(geowatch_dvc --tags='phase2_expt' --hardware='hdd')
echo "DVC_EXPT_DPATH = $DVC_EXPT_DPATH"
WORKDIR=$DVC_EXPT_DPATH/training/$HOSTNAME/$USER
DATASET_CODE=Drop7-MedianNoWinter10GSD-V2
KWCOCO_BUNDLE_DPATH=$DVC_DATA_DPATH/$DATASET_CODE
TRAIN_FPATH=$KWCOCO_BUNDLE_DPATH/data_train_rawbands_split6_hack.kwcoco.zip
VALI_FPATH=$KWCOCO_BUNDLE_DPATH/data_vali_rawbands_split6.kwcoco.zip
CHANNELS="(L8,S2,WV,PD):(blue|green|red)"
EXPERIMENT_NAME=Drop7_scratch_V04
DEFAULT_ROOT_DIR=$WORKDIR/$DATASET_CODE/runs/$EXPERIMENT_NAME
TARGET_LR=3e-4
MAX_STEPS=10000
BATCH_SIZE=6
MAX_EPOCHS=641
ACCUMULATE_GRAD_BATCHES=19
TRAIN_ITEMS_PER_EPOCH=1777
TRAIN_BATCHES_PER_EPOCH=$(python -c "print($TRAIN_ITEMS_PER_EPOCH // $BATCH_SIZE)")
echo "TRAIN_BATCHES_PER_EPOCH = $TRAIN_BATCHES_PER_EPOCH"

python -m geowatch.cli.experimental.recommend_size_adjustments \
    --MAX_STEPS=$MAX_STEPS \
    --MAX_EPOCHS=$MAX_EPOCHS \
    --BATCH_SIZE=$BATCH_SIZE \
    --ACCUMULATE_GRAD_BATCHES=$ACCUMULATE_GRAD_BATCHES \
    --TRAIN_BATCHES_PER_EPOCH="$TRAIN_BATCHES_PER_EPOCH" \
    --TRAIN_ITEMS_PER_EPOCH="$TRAIN_ITEMS_PER_EPOCH"

PREV_CHECKPOINT=$(python -m geowatch.cli.experimental.find_recent_checkpoint --default_root_dir="$DEFAULT_ROOT_DIR")
echo "PREV_CHECKPOINT = $PREV_CHECKPOINT"

WATCH_GRID_WORKERS=4 python -m geowatch.tasks.fusion fit --config "
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
    continual_learning: true
    decoder: mlp
    decouple_resolution: false
    dropout: 0.1
    focal_gamma: 2.0
    global_change_weight: 0.0
    global_class_weight: 0.000
    global_saliency_weight: 1.00
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
    perterb_scale: 1.0e-07
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
    pct_start: 0.1
optimizer:
  class_path: torch.optim.AdamW
  init_args:
    lr: $TARGET_LR
    weight_decay: 3e-06
    betas:
      - 0.9
      - 0.99
trainer:
  accumulate_grad_batches: $ACCUMULATE_GRAD_BATCHES
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
  max_epochs: 743
  num_sanity_val_steps: 0
  limit_val_batches: 4096
  limit_train_batches: $TRAIN_BATCHES_PER_EPOCH
  callbacks:
        - class_path: pytorch_lightning.callbacks.ModelCheckpoint
          init_args:
              monitor: val_loss
              mode: min
              save_top_k: 5
              filename: '{epoch:04d}-{step:06d}-{val_loss:.3f}.ckpt'
              save_last: true
" --ckpt_path="$PREV_CHECKPOINT"

export DVC_DATA_DPATH=$(geowatch_dvc --tags="phase2_data")
export DVC_EXPT_DPATH=$(geowatch_dvc --tags="phase2_expt")
python -m geowatch.mlops.manager "list" --dataset_codes "Drop7-MedianNoWinter10GSD-V2"

#python -m geowatch.mlops.manager "add packages" --dataset_codes "Drop7-MedianNoWinter10GSD-V2"
python -m geowatch.mlops.manager "push packages" --dataset_codes "Drop7-MedianNoWinter10GSD-V2"
python -m geowatch.mlops.manager "list packages" --dataset_codes "Drop7-MedianNoWinter10GSD-V2"


#### ----
# Yardrat training from scratch (rework params)

export CUDA_VISIBLE_DEVICES=0
DVC_DATA_DPATH=$(geowatch_dvc --tags='phase2_data' --hardware='ssd')
DVC_EXPT_DPATH=$(geowatch_dvc --tags='phase2_expt' --hardware='hdd')
echo "DVC_EXPT_DPATH = $DVC_EXPT_DPATH"
WORKDIR=$DVC_EXPT_DPATH/training/$HOSTNAME/$USER
DATASET_CODE=Drop7-MedianNoWinter10GSD-V2
KWCOCO_BUNDLE_DPATH=$DVC_DATA_DPATH/$DATASET_CODE
TRAIN_FPATH=$KWCOCO_BUNDLE_DPATH/data_train_rawbands_split6_hack.kwcoco.zip
VALI_FPATH=$KWCOCO_BUNDLE_DPATH/data_vali_rawbands_split6.kwcoco.zip
CHANNELS="(L8,S2,WV,PD):(blue|green|red)"
EXPERIMENT_NAME=Drop7_scratch_V05
DEFAULT_ROOT_DIR=$WORKDIR/$DATASET_CODE/runs/$EXPERIMENT_NAME
TARGET_LR=3e-3
MAX_STEPS=80000
BATCH_SIZE=10
MAX_EPOCHS=743
ACCUMULATE_GRAD_BATCHES=38
TRAIN_ITEMS_PER_EPOCH=40960
TRAIN_BATCHES_PER_EPOCH=$(python -c "print($TRAIN_ITEMS_PER_EPOCH // $BATCH_SIZE)")
echo "TRAIN_BATCHES_PER_EPOCH = $TRAIN_BATCHES_PER_EPOCH"

python -m geowatch.cli.experimental.recommend_size_adjustments \
    --MAX_STEPS=$MAX_STEPS \
    --MAX_EPOCHS=$MAX_EPOCHS \
    --BATCH_SIZE=$BATCH_SIZE \
    --ACCUMULATE_GRAD_BATCHES=$ACCUMULATE_GRAD_BATCHES \
    --TRAIN_BATCHES_PER_EPOCH="$TRAIN_BATCHES_PER_EPOCH" \
    --TRAIN_ITEMS_PER_EPOCH="$TRAIN_ITEMS_PER_EPOCH"

PREV_CHECKPOINT=$(python -m geowatch.cli.experimental.find_recent_checkpoint --default_root_dir="$DEFAULT_ROOT_DIR")
echo "PREV_CHECKPOINT = $PREV_CHECKPOINT"

# shellcheck disable=SC2037
CKPT_ARG=python -c "if 1:
    if '$PREV_CHECKPOINT' == 'None':
        return ''
    else:
        return '--ckpt_fpath=$PREV_CHECKPOINT'
"
echo "CKPT_ARG = $CKPT_ARG"

# shellcheck disable=SC2086
WATCH_GRID_WORKERS=4 python -m geowatch.tasks.fusion fit --config "
data:
  batch_size              : $BATCH_SIZE
  num_workers             : 8
  train_dataset           : $TRAIN_FPATH
  vali_dataset            : $VALI_FPATH
  time_steps              : 9
  chip_dims               : 256,256
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
    continual_learning: true
    decoder: mlp
    decouple_resolution: false
    dropout: 0.1
    focal_gamma: 2.0
    global_change_weight: 0.0
    global_class_weight: 0.000
    global_saliency_weight: 1.00
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
    perterb_scale: 1.0e-07
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
    weight_decay: 3e-06
    betas:
      - 0.9
      - 0.99
trainer:
  accumulate_grad_batches: $ACCUMULATE_GRAD_BATCHES
  default_root_dir     : $DEFAULT_ROOT_DIR
  accelerator          : gpu
  devices              : 0,
  #devices              : 0,1
  #strategy             : ddp
  check_val_every_n_epoch: 1
  enable_checkpointing: true
  enable_model_summary: true
  log_every_n_steps: 1
  logger: true
  max_epochs: 743
  num_sanity_val_steps: 0
  limit_val_batches: 4096
  limit_train_batches: $TRAIN_BATCHES_PER_EPOCH
  callbacks:
        - class_path: pytorch_lightning.callbacks.ModelCheckpoint
          init_args:
              monitor: val_loss
              mode: min
              save_top_k: 5
              filename: '{epoch:04d}-{step:06d}-{val_loss:.3f}.ckpt'
              save_last: true
initializer:
    init: noop
" $CKPT_ARG


# --------
# ACC FineTune From xView2 (on namek)

#export CUDA_VISIBLE_DEVICES="0,1,2,3"
export CUDA_VISIBLE_DEVICES="0"
DVC_DATA_DPATH=$(geowatch_dvc --tags='phase2_data' --hardware='ssd')
DVC_EXPT_DPATH=$(geowatch_dvc --tags='phase2_expt' --hardware='hdd')
echo "DVC_EXPT_DPATH = $DVC_EXPT_DPATH"
WORKDIR=$DVC_EXPT_DPATH/training/$HOSTNAME/$USER
DATASET_CODE=Drop7-MedianNoWinter10GSD-V2
KWCOCO_BUNDLE_DPATH=$DVC_DATA_DPATH/$DATASET_CODE
TRAIN_FPATH=$KWCOCO_BUNDLE_DPATH/data_train_rawbands_split6_hack.kwcoco.zip
VALI_FPATH=$KWCOCO_BUNDLE_DPATH/data_vali_rawbands_split6.kwcoco.zip
CHANNELS="(L8,S2,WV,PD):(blue|green|red)"
EXPERIMENT_NAME=Drop7_from_xview2_V1
DEFAULT_ROOT_DIR=$WORKDIR/$DATASET_CODE/runs/$EXPERIMENT_NAME
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
CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES
STRATEGY=$STRATEGY
ACCELERATOR=$ACCELERATOR
DDP_WORKAROUND=$DDP_WORKAROUND
DEVICES=$DEVICES
"


TARGET_LR=1e-3
PERTERB_SCALE=$(python -c "print($TARGET_LR * 0.01)")
WEIGHT_DECAY=$(python -c "print($TARGET_LR * 0.01)")
MAX_STEPS=10000
BATCH_SIZE=9
MAX_EPOCHS=720
ACCUMULATE_GRAD_BATCHES=32
TRAIN_ITEMS_PER_EPOCH=4096
TRAIN_BATCHES_PER_EPOCH=$(python -c "print($TRAIN_ITEMS_PER_EPOCH // $BATCH_SIZE)")
echo "TRAIN_BATCHES_PER_EPOCH = $TRAIN_BATCHES_PER_EPOCH"

python -m geowatch.cli.experimental.recommend_size_adjustments \
    --MAX_STEPS=$MAX_STEPS \
    --MAX_EPOCHS=$MAX_EPOCHS \
    --BATCH_SIZE=$BATCH_SIZE \
    --ACCUMULATE_GRAD_BATCHES=$ACCUMULATE_GRAD_BATCHES \
    --TRAIN_BATCHES_PER_EPOCH="$TRAIN_BATCHES_PER_EPOCH" \
    --TRAIN_ITEMS_PER_EPOCH="$TRAIN_ITEMS_PER_EPOCH"

PREV_CHECKPOINT=$(python -m geowatch.cli.experimental.find_recent_checkpoint --default_root_dir="$DEFAULT_ROOT_DIR")
echo "PREV_CHECKPOINT = $PREV_CHECKPOINT"

# shellcheck disable=SC2037
CKPT_ARG=python -c "if 1:
    if '$PREV_CHECKPOINT' == 'None':
        return ''
    else:
        return '--ckpt_fpath=$PREV_CHECKPOINT'
"
echo "CKPT_ARG = $CKPT_ARG"

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
  max_epoch_length        : $ITEMS_PER_EPOCH
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
  log_every_n_steps: 1
  logger: true
  max_epochs: $MAX_EPOCHS
  num_sanity_val_steps: 0
  limit_val_batches: $ITEMS_PER_EPOCH
  limit_train_batches: $ITEMS_PER_EPOCH
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
    init: /home/joncrall/data/dvc-repos/xview2/models/XView2Continue_v4-epoch=1957-step=15664-val_loss=1.347.ckpt.pt
"

_out_='
# shellcheck disable=SC2155
export DVC_DATA_DPATH=$(geowatch_dvc --tags="phase2_data")
# shellcheck disable=SC2155
export DVC_EXPT_DPATH=$(geowatch_dvc --tags="phase2_expt")
cd "$DVC_EXPT_DPATH"
python -m geowatch.mlops.manager "pull packages" --dataset_codes "Drop7-MedianNoWinter10GSD-V2"

python -m geowatch.mlops.manager "status" --dataset_codes "Drop7-MedianNoWinter10GSD-V2"
#python -m geowatch.mlops.manager "add packages" --dataset_codes "Drop7-MedianNoWinter10GSD-V2"
python -m geowatch.mlops.manager "push packages" --dataset_codes "Drop7-MedianNoWinter10GSD-V2"
python -m geowatch.mlops.manager "list packages" --dataset_codes "Drop7-MedianNoWinter10GSD-V2"


# Recent models
#models/fusion/Drop7-MedianNoWinter10GSD-V2/packages/Drop7_finetune_COLD_phase3_V02/Drop7_finetune_COLD_phase3_V02_epoch63_step1408.pt
#models/fusion/Drop7-MedianNoWinter10GSD-V2/packages/Drop7_finetune_COLD_phase3_V02/Drop7_finetune_COLD_phase3_V02_epoch60_step1342.pt
#models/fusion/Drop7-MedianNoWinter10GSD-V2/packages/Drop7_finetune_COLD_phase3_V02/Drop7_finetune_COLD_phase3_V02_epoch79_step1760.pt
#models/fusion/Drop7-MedianNoWinter10GSD-V2/packages/Drop7_finetune_COLD_phase3_V02/Drop7_finetune_COLD_phase3_V02_epoch45_step1012.pt
#models/fusion/Drop7-MedianNoWinter10GSD-V2/packages/Drop7_finetune_COLD_phase3_V02/Drop7_finetune_COLD_phase3_V02_epoch120_step2662.pt

#models/fusion/Drop7-MedianNoWinter10GSD-V2/packages/Drop7_finetune_COLD_phase3_V03/Drop7_finetune_COLD_phase3_V03_epoch209_step4620.pt
#
#models/fusion/Drop7-MedianNoWinter10GSD-V2/packages/Drop7_finetune_COLD_phase3_V02/Drop7_finetune_COLD_phase3_V02_epoch214_step4730.pt
#models/fusion/Drop7-MedianNoWinter10GSD-V2/packages/Drop7_finetune_COLD_phase3_V03/Drop7_finetune_COLD_phase3_V03_epoch226_step4994.pt
'

# --------
# ACC FineTune From xView2 (on namek) (V2, lower LR)

#export CUDA_VISIBLE_DEVICES="0,1,2,3"
export CUDA_VISIBLE_DEVICES="0"
DVC_DATA_DPATH=$(geowatch_dvc --tags='phase2_data' --hardware='ssd')
DVC_EXPT_DPATH=$(geowatch_dvc --tags='phase2_expt' --hardware='hdd')
echo "DVC_EXPT_DPATH = $DVC_EXPT_DPATH"
WORKDIR=$DVC_EXPT_DPATH/training/$HOSTNAME/$USER
DATASET_CODE=Drop7-MedianNoWinter10GSD-V2
KWCOCO_BUNDLE_DPATH=$DVC_DATA_DPATH/$DATASET_CODE
TRAIN_FPATH=$KWCOCO_BUNDLE_DPATH/data_train_rawbands_split6_hack.kwcoco.zip
VALI_FPATH=$KWCOCO_BUNDLE_DPATH/data_vali_rawbands_split6.kwcoco.zip
CHANNELS="(L8,S2,WV,PD):(blue|green|red)"
EXPERIMENT_NAME=Drop7_from_xview2_V2
DEFAULT_ROOT_DIR=$WORKDIR/$DATASET_CODE/runs/$EXPERIMENT_NAME
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
CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES
STRATEGY=$STRATEGY
ACCELERATOR=$ACCELERATOR
DDP_WORKAROUND=$DDP_WORKAROUND
DEVICES=$DEVICES
"


TARGET_LR=3e-4
PERTERB_SCALE=$(python -c "print($TARGET_LR * 0.01)")
WEIGHT_DECAY=$(python -c "print($TARGET_LR * 0.01)")
MAX_STEPS=10000
BATCH_SIZE=9
MAX_EPOCHS=720
ACCUMULATE_GRAD_BATCHES=32
TRAIN_ITEMS_PER_EPOCH=4096
TRAIN_BATCHES_PER_EPOCH=$(python -c "print($TRAIN_ITEMS_PER_EPOCH // $BATCH_SIZE)")
echo "TRAIN_BATCHES_PER_EPOCH = $TRAIN_BATCHES_PER_EPOCH"

python -m geowatch.cli.experimental.recommend_size_adjustments \
    --MAX_STEPS=$MAX_STEPS \
    --MAX_EPOCHS=$MAX_EPOCHS \
    --BATCH_SIZE=$BATCH_SIZE \
    --ACCUMULATE_GRAD_BATCHES=$ACCUMULATE_GRAD_BATCHES \
    --TRAIN_BATCHES_PER_EPOCH="$TRAIN_BATCHES_PER_EPOCH" \
    --TRAIN_ITEMS_PER_EPOCH="$TRAIN_ITEMS_PER_EPOCH"

PREV_CHECKPOINT=$(python -m geowatch.cli.experimental.find_recent_checkpoint --default_root_dir="$DEFAULT_ROOT_DIR")
echo "PREV_CHECKPOINT = $PREV_CHECKPOINT"

# shellcheck disable=SC2037
CKPT_ARG=python -c "if 1:
    if '$PREV_CHECKPOINT' == 'None':
        return ''
    else:
        return '--ckpt_fpath=$PREV_CHECKPOINT'
"
echo "CKPT_ARG = $CKPT_ARG"

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
  max_epoch_length        : $ITEMS_PER_EPOCH
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
  log_every_n_steps: 1
  logger: true
  max_epochs: $MAX_EPOCHS
  num_sanity_val_steps: 0
  limit_val_batches: $ITEMS_PER_EPOCH
  limit_train_batches: $ITEMS_PER_EPOCH
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
    init: /home/joncrall/data/dvc-repos/xview2/models/XView2Continue_v4-epoch=1957-step=15664-val_loss=1.347.ckpt.pt
"
