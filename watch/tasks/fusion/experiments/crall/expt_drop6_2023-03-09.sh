#!/bin/bash

PHASE2_DATA_DPATH_SSD=$(smartwatch_dvc --tags="phase2_data" --hardware="auto")
cd "$PHASE2_DATA_DPATH_SSD/Drop6"
python -m watch.cli.prepare_splits \
    --base_fpath "combo_imganns-*_L.kwcoco.json" \
    --suffix=fixquant \
    --constructive_mode=True


export CUDA_VISIBLE_DEVICES=0
DVC_DATA_DPATH=$(smartwatch_dvc --tags='phase2_data' --hardware='auto')
DVC_EXPT_DPATH=$(smartwatch_dvc --tags='phase2_expt' --hardware='auto')
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
DVC_DATA_DPATH=$(smartwatch_dvc --tags='phase2_data' --hardware='auto')
DVC_EXPT_DPATH=$(smartwatch_dvc --tags='phase2_expt' --hardware='auto')
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
